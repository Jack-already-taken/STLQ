import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantizers.uniform import *
from quantizers.logarithm import *
from quant_layers.linear import *
import copy

class LogQuantLinear(PTQSLBatchingQuantLinear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 mode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 n_V = 1, 
                 fpcs = False,
                 steps = 4):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit,
                         calib_batch_size=calib_batch_size, search_round=search_round, eq_n=eq_n, n_V=n_V)
        self.fpcs = fpcs
        self.steps = steps
        
        del self.a_quantizer, self.w_quantizer
        self.w_quantizer = LogQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.a_quantizer = UniformIntQuantizer(n_bits = a_bit, symmetric = False, channel_wise = False)
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1)))
        self.a_quantizer.zero_point = nn.Parameter(torch.zeros((1)))
        self.w_quantizer.scale = nn.Parameter(torch.zeros((n_V, self.crb_rows, 1)))

    def _initialize_weight_scale(self):
        self.w_quantizer.scale.data.copy_(
            (self.weight.view(self.n_V, self.crb_rows, self.in_features).amax([2],keepdim=True) - 
                self.weight.view(self.n_V, self.crb_rows, self.in_features).amin([2],keepdim=True)) / 
            (2 * self.w_quantizer.n_levels - 1)
        )
        self.w_quantizer.inited = True

    def _initialize_activation_scale(self):
        tmp_a_scales = []
        tmp_a_max, tmp_a_min = [], []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            if self.a_quantizer.channel_wise:
                a_max = x_.abs().amax([i for i in range(x_.ndim-1)], keepdim=False).detach().view(1, -1)
                a_min = x_.abs().amin([i for i in range(x_.ndim-1)], keepdim=False).detach().view(1, -1)
            else:
                a_max = x_.abs().max().detach().view(1, 1)
                a_min = x_.abs().min().detach().view(1, 1)
            tmp_a_max.append(a_max)
            tmp_a_min.append(a_min)
        tmp_a_max = torch.cat(tmp_a_max, dim=0).amax(dim=0, keepdim=False)
        tmp_a_min = torch.cat(tmp_a_min, dim=0).amin(dim=0, keepdim=False)
        self.a_quantizer.scale.data.copy_((tmp_a_max - tmp_a_min) / (2 * self.a_quantizer.n_levels - 1))
        self.a_quantizer.zero_point.data.copy_(-tmp_a_min / self.a_quantizer.scale)
        self.a_quantizer.inited = True

    def _search_best_w_scale_self(self, weight_scale_candidates, topk=1):
        similarities = []
        tmp_w_quantizer = copy.deepcopy(self.w_quantizer)
        raw_weight = self.weight.view(self.n_V, self.crb_rows, self.in_features).unsqueeze(0) # shape: 1,n_V,crb_rows,in_features
        for p_st in range(0, self.eq_n, self.parallel_eq_n):
            p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
            cur_w_scale = weight_scale_candidates[p_st:p_ed]
            # quantize weight and bias 
            tmp_w_quantizer.scale = nn.Parameter(cur_w_scale)
            w_dequant = tmp_w_quantizer(raw_weight) # shape: 1,n_V,crb_rows,in_features
            similarity = self._get_similarity(raw_weight, w_dequant) # shape: parallel_eq_n,n_V,crb_rows,in_features
            similarity = torch.mean(similarity, dim=-1, keepdim=False) # shape: parallel_eq_n,n_V,crb_rows
            similarities.append(similarity)
        similarities = torch.cat(similarities, dim=0) # shape: eq_n,n_V,crb_rows
        _, best_index = torch.topk(similarities, k=topk, dim=0)
        best_index = best_index.reshape(topk, self.n_V, -1, 1)
        if topk == 1:
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(0))
            self.w_quantizer.inited = True
        return best_index.squeeze(0) # shape: (topk, n_V,crb_rows,1)

    def _search_best_a_scale_self(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_x = self.raw_input[b_st:b_ed].cuda().unsqueeze(-1) # shape: b,*,in_features,1
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[:, p_st:p_ed]
                # quantize input
                x_sim = x.unsqueeze(-1) # shape: B,*,in_features,1
                x_quant = ((x_sim / cur_a_scale).round_() + cur_a_zero_point).clamp_(0, 2 * self.a_quantizer.n_levels - 1) # shape: B,*,in_features,parallel_eq_n
                x_dequant = (x_quant - cur_a_zero_point) * cur_a_scale # shape: B,*,in_features,parallel_eq_n
                similarity = self._get_similarity(raw_x, x_dequant) # shape: b,*,in_features,parallel_eq_n
                if len(similarity.shape) > 3:
                    similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-2))) # shape: b, in_features, parallel_eq_n
                if not self.a_quantizer.channel_wise:
                    similarity = torch.mean(similarity, dim=1, keepdim=True) # shape: b, 1, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, in_features, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=-1) # shape: 1, in_features, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: in_features, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1) # shape: in_features, topk
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=-1, index=best_index)
            tmp_a_zero_point = torch.gather(input_zero_point_candidates, dim=-1, index=best_index)
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(-1))
            self.a_quantizer.zero_point.data.copy_(tmp_a_zero_point.squeeze(-1))
            self.a_quantizer.inited = True
        return best_index
    
    def _search_best_w_scale(self, weight_scale_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        tmp_w_quantizer = copy.deepcopy(self.w_quantizer)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,out_features
            raw_out_expanded = raw_out_expanded.view(*raw_out_expanded.shape[:-1], self.n_V, -1) # shape: b,*,1,n_V,crb_rows
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]
                # quantize weight and bias 
                tmp_w_quantizer.scale = nn.Parameter(cur_w_scale)
                w_dequant = tmp_w_quantizer(self.weight.view(1, self.n_V, self.crb_rows, self.in_features)) # shape: 1,n_V,crb_rows,in_features
                w_sim = w_dequant.view(-1,self.in_features) # shape: parallel_eq_n*out_features,in_features
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                x_sim = self.quant_input(x)
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n*out_features
                out_sim = out_sim.view(*out_sim.shape[:-1], p_ed-p_st, self.n_V, -1) # shape: b,*,parallel_eq_n,n_V,crb_rows
                similarity = self._get_similarity(raw_out_expanded, out_sim) # shape: b,*,parallel_eq_n,n_V,crb_rows
                if len(similarity.shape) > 4:
                    similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-3))) # shape: b, parallel_eq_n, n_V, crb_rows
                similarity = similarity.sum(dim=0, keepdim=True) # shape: (1, parallel_eq_n, n_V) or (1, parallel_eq_n, n_V, crb_rows)
                similarities.append(similarity)
            # store best weight scale of h into tmp_w_scale
            similarities = torch.cat(similarities, dim=1) # shape: (1, eq_n, n_V) or (1, eq_n, n_V, crb_rows)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: (eq_n, n_V) or (eq_n, n_V, crb_rows)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.reshape(topk, self.n_V, -1, 1)
        if topk == 1:
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(0))
        return best_index.squeeze(0) # shape: (topk, n_V,crb_rows,1)
    
    def _search_best_a_scale(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,oc
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[:, p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim = x.unsqueeze(-1) # shape: B,*,in_features,1
                x_quant = ((x_sim / cur_a_scale).round_() + cur_a_zero_point).clamp_(0, 2 * self.a_quantizer.n_levels - 1) # shape: B,*,in_features,parallel_eq_n
                x_dequant = (x_quant - cur_a_zero_point) * cur_a_scale # shape: B,*,in_features,parallel_eq_n
                x_sim = x_dequant.permute(*list(range(len(x_sim.shape)-2)),-1,-2) # shape: B,*,parallel_eq_n,in_features
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim) # shape: b,*,parallel_eq_n,out_features
                similarity = torch.mean(similarity, dim=-1) # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=True) # shape: 1, eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=-1) # shape: 1, topk
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=-1, index=best_index)
            tmp_a_zero_point = torch.gather(input_zero_point_candidates, dim=-1, index=best_index)
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(-1))
            self.a_quantizer.zero_point.copy_(tmp_a_zero_point.squeeze(-1))
        return best_index
        
    def calculate_percentile_weight_candidates(self, l=0.7 , r=1.0):
        num_scale = self.eq_n
        pct = torch.tensor([l, r])
        alpha_bounds = torch.quantile(
            self.weight.view(self.n_V, self.crb_rows, self.in_features), pct.to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shape: 2, n_V, crb_rows, 1
        alpha_lowers_candidates = alpha_bounds[0:1] # shape: 1, out_channels, 1
        alpha_uppers_candidates = alpha_bounds[1:]
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None, None] * (alpha_uppers_candidates - alpha_lowers_candidates)
        weight_scale_candidates = (alpha_lowers_candidates + splits)#.repeat(num_zp, 1, 1)
        return weight_scale_candidates
    
    def calculate_percentile_activation_candidates(self, l=0.9, r=1.0):
        num_zp = min(16, self.a_quantizer.n_levels * 2)
        num_scale = int(self.eq_n / num_zp)
        percentiles_uppers, percentiles_lowers = [], []
        pct = torch.tensor([l, r])
        x = self.raw_input.cuda()
        tensor_too_large = True
        mini_batch_size = 1
        if self.a_quantizer.channel_wise:
            a_uppers_candidates = torch.quantile(x.view(-1, x.shape[-1]), pct.to(x.device), dim=0).transpose(0, 1) # shape: in_features, 2
            a_lowers_candidates = torch.quantile(x.view(-1, x.shape[-1]), (1-pct).to(x.device), dim=0).transpose(0, 1) # shape: in_features, 2
        else:
            while tensor_too_large:
                try:
                    a_uppers_candidates = torch.quantile(x.view(mini_batch_size, -1), pct.to(x.device), dim=-1).mean(dim=-1).unsqueeze(0) # shape: 1, 2
                    a_lowers_candidates = torch.quantile(x.view(mini_batch_size, -1), (1-pct).to(x.device), dim=-1).mean(dim=-1).unsqueeze(0) # shape: 1, 2
                    tensor_too_large = False
                except:
                    mini_batch_size *= 2
        delta_min = a_uppers_candidates[:, 0:1] - a_lowers_candidates[:, 0:1]
        delta_max = a_uppers_candidates[:, 1:] - a_lowers_candidates[:, 1:]
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[None, :] * (delta_max - delta_min)
        a_scale_candidates = ((delta_min + splits).repeat(1, num_zp) / (2 * self.a_quantizer.n_levels - 1)).clamp(min=1e-4)
        zp_min = int(self.a_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.a_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()
        a_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[None, :]
        a_zero_point_candidates = a_zero_point_candidates.repeat(a_scale_candidates.shape[0], 1)
        return a_scale_candidates, a_zero_point_candidates

    def weight_fpcs(self, fpcs_width=16, steps=6, search_strategy=None):
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        weight_scale_candidates = self.calculate_percentile_weight_candidates()
        delta_scale = weight_scale_candidates[1:2] - weight_scale_candidates[0:1]
        topk_index = search_strategy(self, weight_scale_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
        remain_steps = steps - 1
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[:, None, None, None] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)
            weight_scale_candidates = (topk_scale_candidates.unsqueeze(1) + delta_scale_candidates.unsqueeze(0)).reshape(
                -1, *weight_scale_candidates.shape[1:])
            topk_index = search_strategy(self, weight_scale_candidates, topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
            remain_steps -= 1

    def activation_fpcs(self, fpcs_width=16, steps=6, search_strategy=None):
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates()
        delta_scale = a_scale_candidates[:, 1:2] - a_scale_candidates[:, 0:1]
        topk_index = search_strategy(self, a_scale_candidates, a_zero_point_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(a_scale_candidates, dim=-1, index=topk_index)
        topk_zp_candidates = torch.gather(a_zero_point_candidates, dim=-1, index=topk_index)
        remain_steps = steps - 1
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[None, :] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)
            a_scale_candidates = (topk_scale_candidates.unsqueeze(-1) + delta_scale_candidates.unsqueeze(-2)).reshape(
                *a_scale_candidates.shape[:-1], -1).clamp(min=1e-4)
            a_zero_point_candidates = topk_zp_candidates.repeat_interleave(fpcs_new_cnt, dim=-1)
            topk_index = search_strategy(self, a_scale_candidates, a_zero_point_candidates, 
                                         topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(a_scale_candidates, dim=-1, index=topk_index)
                topk_zp_candidates = torch.gather(a_zero_point_candidates, dim=-1, index=topk_index)
            remain_steps -= 1
    
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        if self.fpcs:
            self.weight_fpcs(steps=self.steps, search_strategy=LogQuantLinear._search_best_w_scale_self)
            if self.a_quantizer.n_bits <= 8:
                self.activation_fpcs(steps=self.steps, search_strategy=LogQuantLinear._search_best_a_scale_self)
        else:
            weight_scale_candidates = self.calculate_percentile_weight_candidates()
            a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates()
            self._search_best_w_scale_self(weight_scale_candidates)
            if self.a_quantizer.n_bits <= 8:
                self._search_best_a_scale_self(a_scale_candidates, a_zero_point_candidates)

        for e in range(self.search_round):
            if self.fpcs:
                self.weight_fpcs(steps=self.steps, search_strategy=LogQuantLinear._search_best_w_scale)
                if self.a_quantizer.n_bits <= 8:
                    self.activation_fpcs(steps=self.steps, search_strategy=LogQuantLinear._search_best_a_scale)
            else:
                self._search_best_w_scale(weight_scale_candidates)
                if self.a_quantizer.n_bits <= 8:
                    self._search_best_a_scale(a_scale_candidates, a_zero_point_candidates)
        print("log linear")
        self.calibrated = True

        self.weight_deq = self.quant_weight_bias()[0]
        self.act_deq = self.quant_input(self.raw_input[:].cuda())
        self.weight_q = self.w_quantizer.quant_val
        self.act_q = self.a_quantizer.quant_val
        if self.act_q is None:
            self.act_q = self.act_deq
        if self.weight_q is None:
            self.weight_q = self.weight_deq
        # del self.raw_input, self.raw_out
        del self.raw_out
        return None

class TwoWordLogQuantLinear(LogQuantLinear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 mode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 n_V = 1, 
                 fpcs = False,
                 steps = 4):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit,
                         calib_batch_size=calib_batch_size, search_round=search_round, eq_n=eq_n, n_V=n_V)
        self.fpcs = fpcs
        self.steps = steps

        del self.a_quantizer, self.w_quantizer
        self.w_quantizer = SelectiveTwoWordLogQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True, threshold=0.04, second_word_ratio=None)
        self.a_quantizer = UniformIntQuantizer(n_bits = a_bit, symmetric = False, channel_wise = False)
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1)))
        self.a_quantizer.zero_point = nn.Parameter(torch.zeros((1)))
        self.w_quantizer.scale = nn.Parameter(torch.zeros((n_V, self.crb_rows, 1)))

    def hyperparameter_searching(self):
        super().hyperparameter_searching()
        self.second_word_ratio = self.w_quantizer.mask.float().mean().item()
        print("Second Word Ratio:", self.second_word_ratio)
        print("Mask shape:", self.w_quantizer.mask.shape)
        print("Threshold:", self.w_quantizer.threshold)

class TwoWordLogQuantLinear2(LogQuantLinear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 mode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 n_V = 1, 
                 fpcs = False,
                 steps = 4):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit,
                         calib_batch_size=calib_batch_size, search_round=search_round, eq_n=eq_n, n_V=n_V)
        self.fpcs = fpcs
        self.steps = steps

        del self.a_quantizer, self.w_quantizer
        self.w_quantizer = SelectiveTwoWordLogQuantizer2(n_bits = w_bit, symmetric = False, channel_wise = True, threshold=0.08, second_word_ratio=None, scale2=0.1)
        self.a_quantizer = UniformIntQuantizer(n_bits = a_bit, symmetric = False, channel_wise = False)
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1)))
        self.a_quantizer.zero_point = nn.Parameter(torch.zeros((1)))
        self.w_quantizer.scale = nn.Parameter(torch.zeros((n_V, self.crb_rows, 1)))

    def hyperparameter_searching(self):
        super().hyperparameter_searching()
        self.second_word_ratio = self.w_quantizer.mask.float().mean().item()
        print("Second Word Ratio:", self.second_word_ratio)
        print("Mask shape:", self.w_quantizer.mask.shape)
        print("Threshold:", self.w_quantizer.threshold)