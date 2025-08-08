from typing import Tuple
from numpy import not_equal
import math
from torch import tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from quantizers.uniform import UniformQuantizer, UniformIntQuantizer, LogIntQuantizer
from quantizers.logarithm import LogQuantizer, ShiftLog2Quantizer, SelectiveTwoWordLogQuantizer, SelectiveTwoWordLogQuantizer2
import copy


class MinMaxQuantConv2d(nn.Conv2d):
    """
    MinMax quantize weight and output
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mode = mode
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = False)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        
    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward":
            out=self.quant_forward(x)
        elif self.mode == 'debug_only_quant_weight':
            out = self.debug_only_quant_weight(x)
        elif self.mode == 'debug_only_quant_act':
            out = self.debug_only_quant_act(x)
        else:
            raise NotImplementedError
        return out
            
    def quant_weight_bias(self):
        w_sim = self.w_quantizer(self.weight)
        return w_sim, self.bias if self.bias is not None else None
    
    def quant_input(self,x):
        if self.a_quantizer.n_bits >= 32:
            return x
        return self.a_quantizer(x)
    
    def quant_forward(self,x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    def debug_only_quant_weight(self, x):
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.conv2d(x, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    def debug_only_quant_act(self, x):
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    
class PTQSLQuantConv2d(MinMaxQuantConv2d):
    """
    PTQSL on Conv2d
    weight: (oc,ic,kw,kh) -> (oc,ic*kw*kh) -> divide into sub-matrixs and quantize
    input: (B,ic,W,H), keep this shape

    Only support SL quantization on weights.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit)
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.search_round = search_round
        self.eq_n = eq_n
        self.parallel_eq_n = eq_n
        
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        # self.a_quantizer.register_buffer('scale', torch.zeros((1, 1, 1, 1)))
    
    def _get_similarity(self, tensor_raw, tensor_sim):
        return -(tensor_raw - tensor_sim) ** 2

    def quant_weight_bias(self):
        # self.weight_scale shape: (1, 1) or (oc, 1) 
        # self.weight       shape: (oc,ic,kw,kh)
        oc, ic, kw, kh = self.weight.data.shape
        w_sim = self.w_quantizer(self.weight.view(oc, ic * kw * kh)).view(oc, ic, kw, kh)
        return w_sim, self.bias if self.bias is not None else None

    
class PTQSLBatchingQuantConv2d(PTQSLQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, search_round, eq_n)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        self.calib_size = self.raw_input.shape[0]
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
            # memory = free_memory // 2  # Use half of the free memory for calibration
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (2 * self.raw_input[:self.calib_batch_size].numel() + 
                 2 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        
    def _initialize_activation_scale(self):
        tmp_a_scales = []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            a_scale_=(x_.abs().max() / (self.a_quantizer.n_levels - 0.5)).detach().view(1, 1)
            tmp_a_scales.append(a_scale_)
        tmp_a_scale = torch.cat(tmp_a_scales, dim=1).amax(dim=1, keepdim=False).view(1, 1, 1, 1)
        self.a_quantizer.scale.data.copy_(tmp_a_scale) # shape: (1, 1, 1, 1)
        self.a_quantizer.inited = True
        
    def _search_best_a_scale(self, input_scale_candidates):
        batch_similarities = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                B,ic,iw,ih = x.shape
                x_sim = x.unsqueeze(0) # shape: 1,B,ic,iw,ih
                x_sim = (x_sim / (cur_a_scale)).round_().clamp_(-self.a_quantizer.n_levels, self.a_quantizer.n_levels - 1) * cur_a_scale # shape: parallel_eq_n,B,ic,iw,ih
                x_sim = x_sim.view(-1,ic,iw,ih)
                # calculate similarity and store them
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: parallel_eq_n*B,oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(0), chunks=p_ed-p_st, dim=1), dim=0) # shape: parallel_eq_n,B,oc,fw,fh
                out_sim = out_sim.transpose_(0, 1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = torch.mean(similarity, dim=[2,3,4]) # shape: B,parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1,parallel_eq_n
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1) # shape: 1,eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: eq_n
        best_index = batch_similarities.argmax(dim=0).view(1,1,1,1,1)
        tmp_a_scale = torch.gather(input_scale_candidates, dim=0, index=best_index)
        self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(0))
        
        
class AsymmetricallyBatchingQuantConv2d(PTQSLBatchingQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 fpcs = False,
                 steps = 4):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, calib_batch_size, search_round, eq_n)
        self.fpcs = fpcs
        self.steps = steps
        del self.w_quantizer
        # self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        # self.w_quantizer = Log2Quantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.w_quantizer = UniformIntQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros((self.out_channels, 1)))

        del self.a_quantizer
        self.a_quantizer = UniformIntQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        # self.tmp_w_quantizer = UniformIntQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        # self.tmp_w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        # self.tmp_w_quantizer.zero_point = nn.Parameter(torch.zeros((self.out_channels, 1)))
    
    def _search_best_w_scale(self, weight_scale_candidates, weight_zero_point_candidates, topk=1):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed] # shape: (parallel_eq_n, 1, 1) or (parallel_eq_n, oc, 1)
                cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]
                # quantize weight and bias 
                oc,ic,kw,kh = self.weight.data.shape
                w_sim = self.weight.view(oc, -1).unsqueeze(0) # shape: (1, oc, ic*kw*kh)
                w_quant = ((w_sim / cur_w_scale).round_() + cur_w_zero_point).clamp(0, 2 * self.w_quantizer.n_levels - 1)
                w_sim = (w_quant - cur_w_zero_point).mul_(cur_w_scale) # shape: (parallel_eq_n,oc,ic*kw*kh)
                # print(cur_w_scale.shape, cur_w_scale.squeeze(0).shape)
                # self.tmp_w_quantizer.scale.data.copy_(cur_w_scale.squeeze(0))
                # self.tmp_w_quantizer.zero_point.data.copy_(cur_w_zero_point.squeeze(0))
                # w_sim = self.tmp_w_quantizer(w_sim) # shape: (parallel_eq_n,oc,ic*kw*kh)
                w_sim = w_sim.view(-1,ic,kw,kh) # shape: parallel_eq_n*oc,ic,kw,kh
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                # quantize input
                x_sim = self.quant_input(x)
                # calculate similarity and store them
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: B,parallel_eq_n*oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = torch.mean(similarity, [3,4]) # shape: B,parallel_eq_n,oc
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: (1,parallel_eq_n) or (1,parallel_eq_n,oc)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1) # shape: (1,eq_n) or (1,eq_n,oc)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: (eq_n) or (eq_n,oc)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        # print(_)
        best_index = best_index.view(topk, -1, 1)
        if topk == 1:
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            tmp_w_zero_point = torch.gather(weight_zero_point_candidates, dim=0, index=best_index)
            # print(tmp_w_scale)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
            self.w_quantizer.zero_point.data.copy_(tmp_w_zero_point.squeeze(dim=0))
        return best_index
    
    def calculate_activation_candidates(self, l=0.1, r=1.0):
        input_scale_candidates =  torch.tensor(
            [l + i * (r - l) / self.eq_n for i in range(self.eq_n + 1)]
        ).cuda().view(-1,1,1,1,1) * self.a_quantizer.scale # shape: (eq_n,1,1,1,1)
        return input_scale_candidates
    
    def calculate_percentile_weight_candidates(self, l=0.6, r=1.0):
        num_zp = self.w_quantizer.n_levels // 4
        num_scale = int(self.eq_n / num_zp)
        pct = torch.tensor([l, r])
        w_uppers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), pct.to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shape: 2, out_channels, 1
        w_lowers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), (1-pct).to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shapeL 2, out_channels, 1
        delta_min = w_uppers_candidates[0:1] - w_lowers_candidates[0:1]
        delta_max = w_uppers_candidates[1:] - w_lowers_candidates[1:]
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None] * (delta_max - delta_min)
        weight_scale_candidates = (delta_min + splits).repeat(num_zp, 1, 1) / (2 * self.w_quantizer.n_levels - 1)
        # print(weight_scale_candidates)
        zp_min = int(self.w_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.w_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()
        weight_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None]
        weight_zero_point_candidates = weight_zero_point_candidates.repeat(1, self.out_channels, 1)
        return weight_scale_candidates, weight_zero_point_candidates
    
    def weight_fpcs(self, fpcs_width=16, steps=4, search_strategy=None):
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
        delta_scale = weight_scale_candidates[1:2] - weight_scale_candidates[0:1]
        topk_index = search_strategy(self, weight_scale_candidates, weight_zero_point_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
        topk_zp_candidates = torch.gather(weight_zero_point_candidates, dim=0, index=topk_index)
        remain_steps = steps - 1
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[:, None, None] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)
            weight_scale_candidates = (topk_scale_candidates.unsqueeze(1) + delta_scale_candidates.unsqueeze(0)).reshape(
                -1, *weight_scale_candidates.shape[1:])
            weight_zero_point_candidates = topk_zp_candidates.repeat_interleave(fpcs_new_cnt, dim=0)
            topk_index = search_strategy(self, weight_scale_candidates, weight_zero_point_candidates, 
                                         topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
                topk_zp_candidates = torch.gather(weight_zero_point_candidates, dim=0, index=topk_index)
            remain_steps -= 1
        
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        if self.a_quantizer.n_bits < 32:
            self._initialize_activation_scale()
            input_scale_candidates = self.calculate_activation_candidates()
        weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
        self.w_quantizer.scale.data.copy_(weight_scale_candidates[-2])
        self.w_quantizer.zero_point.data.copy_(weight_zero_point_candidates[-2])
        self.w_quantizer.inited = True
        
        for e in range(self.search_round):
            if self.fpcs:
                self.weight_fpcs(steps=self.steps, search_strategy=AsymmetricallyBatchingQuantConv2d._search_best_w_scale)
            else:
                self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
            if self.a_quantizer.n_bits < 32:
                self._search_best_a_scale(input_scale_candidates)
            else:
                break
        self.calibrated = True
        self.weight_deq = self.quant_weight_bias()[0]
        self.act_deq = self.quant_input(self.raw_input[:].cuda())
        self.weight_q = self.w_quantizer.quant_val
        self.act_q = self.a_quantizer.quant_val
        if self.act_q is None:
            self.act_q = self.act_deq
        if self.weight_q is None:
            self.weight_q = self.weight_deq
        del self.raw_out
        return None

    # def _calc_qparams_pc(self, lo: torch.Tensor,
    #                     hi: torch.Tensor,
    #                     eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     lo, hi: 1-D tensors of length = n_channels
    #     returns (scale, zero_point) each shape [n_channels]
    #     """
    #     if self.w_quantizer.sym:
    #         bound  = torch.maximum(lo.abs(), hi.abs()) + eps
    #         scale  = bound / ((2 ** (self.w_quantizer.n_bits - 1)) - 1)
    #         zp     = torch.zeros_like(scale)          # zp = 0 in symmetric case
    #         return scale, zp

    #     qmin, qmax = 0, 2 ** self.w_quantizer.n_bits - 1
    #     scale = (hi - lo) / (qmax - qmin) + eps
    #     zp    = (qmin - lo / scale).round().clamp(qmin, qmax)
    #     return scale, zp

    # def hyperparameter_searching(self):
    #     self._initialize_calib_parameters()
    #     if self.a_quantizer.n_bits < 8:
    #         self._initialize_activation_scale()
    #         self.calculate_activation_candidates()
    #     self.w_quantizer.inited = True
    #     lo = torch.quantile(self.weight.view(self.out_channels, -1), 0.001, dim=-1)
    #     hi = torch.quantile(self.weight.view(self.out_channels, -1), 0.999, dim=-1)
    #     scale, zero_point = self._calc_qparams_pc(lo, hi)
    #     self.w_quantizer.scale.data.copy_(scale.unsqueeze(-1))
    #     self.w_quantizer.zero_point.data.copy_(zero_point.unsqueeze(-1))

        
    #     self.calibrated = True
    #     del self.raw_input, self.raw_out
    #     return None
    

class LogQuantConv2d(AsymmetricallyBatchingQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 fpcs = False,
                 steps = 4,
                 a_mode = 'linear'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, calib_batch_size, search_round, eq_n)
        self.fpcs = fpcs
        self.steps = steps
        del self.w_quantizer

        # self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        # self.w_quantizer = Log2Quantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.w_quantizer = LogQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        # self.w_quantizer = UniformIntQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        # self.w_quantizer = SelectiveTwoWordLogQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True, threshold=0.4)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))

        self.a_mode = a_mode
        if a_mode == 'log':
            del self.a_quantizer
            self.a_quantizer = LogQuantizer(n_bits = a_bit, symmetric = False, channel_wise = True)
            self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))

    def calculate_percentile_weight_candidates(self, l=0.7, r=1.0):
        # num_zp = self.w_quantizer.n_levels
        # num_scale = int(self.eq_n / num_zp)
        num_scale = self.eq_n
        pct = torch.tensor([l, r])
        alpha_bounds = torch.quantile(
            self.weight.view(self.out_channels, -1).abs(), pct.to(self.weight.device), dim=-1
        ).unsqueeze(-1)
        alpha_lowers_candidates = alpha_bounds[0:1] # shape: 1, out_channels, 1
        alpha_uppers_candidates = alpha_bounds[1:]
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None] * (alpha_uppers_candidates - alpha_lowers_candidates)
        weight_scale_candidates = (alpha_lowers_candidates + splits)#.repeat(num_zp, 1, 1)
        # print(weight_scale_candidates)
        return weight_scale_candidates

    def _search_best_w_scale(self, weight_scale_candidates, topk=1):
        # print(self.w_quantizer.n_levels)
        batch_similarities = []
        tmp_w_quantizer = copy.deepcopy(self.w_quantizer)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]  # shape: (parallel_eq_n, oc, 1)
                
                # Quantize weight and bias with logarithmic quantization
                oc,ic,kw,kh = self.weight.data.shape
                
                w_sim = self.weight.view(oc, -1).unsqueeze(0)  # shape: (1, oc, ic*kw*kh)
                # sign = torch.sign(w_sim)
                # w_quant = (w_sim.abs() / cur_w_scale).clamp(min=1e-15, max=1.0)
                # e = -torch.log2(w_quant)
                # e = torch.round(e)
                # e = e.clamp(0, self.w_quantizer.n_levels - 1)
                # w_sim = (2.0 ** (-e)) * cur_w_scale * sign
                # print(cur_w_scale.shape, cur_w_scale.squeeze(0).shape)
                tmp_w_quantizer.scale = nn.Parameter(cur_w_scale.squeeze(0))
                w_sim = tmp_w_quantizer(w_sim)  # shape: (parallel_eq_n, oc, ic*kw*kh)
                w_sim = w_sim.view(-1, ic, kw, kh)  # shape: parallel_eq_n*oc,ic,kw,kh
                
                # Handle bias
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                
                # Quantize input
                x_sim = self.quant_input(x)
                
                # Calculate similarity
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1)
                similarity = self._get_similarity(raw_out, out_sim)
                similarity = torch.mean(similarity, [3,4])
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
                
            similarities = torch.cat(similarities, dim=1)
            batch_similarities.append(similarities)
            
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)
        
        # Get top-k best scales
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, -1, 1)
        
        if topk == 1:
            # print(_)
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
            
        return best_index

    def _initialize_activation_scale_log(self):
        tmp_a_scales = []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            a_scale_=(x_.abs().max()).detach().view(1, 1)
            tmp_a_scales.append(a_scale_)
        tmp_a_scale = torch.cat(tmp_a_scales, dim=1).amax(dim=1, keepdim=False).view(1, 1, 1, 1)
        self.a_quantizer.scale.data.copy_(tmp_a_scale) # shape: (1, 1, 1, 1)
        self.a_quantizer.inited = True

    def calculate_activation_candidates_log(self, l=0.1, r=1.0):
        input_scale_candidates =  torch.tensor(
            [l + i * (r - l) / self.eq_n for i in range(self.eq_n + 1)]
        ).cuda().view(-1,1,1,1,1) * self.a_quantizer.scale # shape: (eq_n,1,1,1,1)
        return input_scale_candidates

    def _search_best_a_scale_log(self, input_scale_candidates):
        batch_similarities = []
        tmp_a_quantizer = copy.deepcopy(self.a_quantizer)
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                B,ic,iw,ih = x.shape
                x_sim = x.unsqueeze(0) # shape: 1,B,ic,iw,ih
                tmp_a_quantizer.scale = nn.Parameter(cur_a_scale.squeeze(0))
                x_sim = tmp_a_quantizer(x_sim)  # shape: (parallel_eq_n, oc, ic*kw*kh)
                x_sim = x_sim.view(-1,ic,iw,ih)
                # calculate similarity and store them
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: parallel_eq_n*B,oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(0), chunks=p_ed-p_st, dim=1), dim=0) # shape: parallel_eq_n,B,oc,fw,fh
                out_sim = out_sim.transpose_(0, 1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = torch.mean(similarity, dim=[2,3,4]) # shape: B,parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1,parallel_eq_n
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1) # shape: 1,eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: eq_n
        best_index = batch_similarities.argmax(dim=0).view(1,1,1,1,1)
        tmp_a_scale = torch.gather(input_scale_candidates, dim=0, index=best_index)
        self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(0))

    def weight_fpcs(self, fpcs_width=16, steps=4, search_strategy=None):
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        weight_scale_candidates= self.calculate_percentile_weight_candidates()
        delta_scale = weight_scale_candidates[1:2] - weight_scale_candidates[0:1]
        topk_index = search_strategy(self, weight_scale_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
        remain_steps = steps - 1
        # print("Min candidate scale:", weight_scale_candidates.min().item())
        # print("Max candidate scale:", weight_scale_candidates.max().item())
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[:, None, None] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)
            weight_scale_candidates = (topk_scale_candidates.unsqueeze(1) + delta_scale_candidates.unsqueeze(0)).reshape(
                -1, *weight_scale_candidates.shape[1:])
            topk_index = search_strategy(self, weight_scale_candidates, topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(weight_scale_candidates, dim=0, index=topk_index)
            remain_steps -= 1
    
    def hyperparameter_searching(self):
        if self.a_quantizer.n_bits >= 32 and self.w_quantizer.n_bits >= 32:
            print("Skip calibration for 32-bit quantization")
            self.calibrated = True
            return None
        self._initialize_calib_parameters()
        if self.a_quantizer.n_bits < 32:
            if self.a_mode == 'log':
                self._initialize_activation_scale_log()
                input_scale_candidates = self.calculate_activation_candidates_log()
            else:
                self._initialize_activation_scale()
                input_scale_candidates = self.calculate_activation_candidates()
            
        # Get weight scale candidates
        weight_scale_candidates= self.calculate_percentile_weight_candidates()
        self.w_quantizer.scale.data.copy_(weight_scale_candidates[-2])
        self.w_quantizer.inited = True
        # self.fpcs = False if self.w_quantizer.n_bits >= 8 else self.fpcs
        for e in range(self.search_round):
            if self.fpcs:
                self.weight_fpcs(steps=self.steps, search_strategy=LogQuantConv2d._search_best_w_scale)
            else:
                self._search_best_w_scale(weight_scale_candidates)
                
            if self.a_quantizer.n_bits < 32:
                if self.a_mode == 'log':
                    self._search_best_a_scale_log(input_scale_candidates)
                else:
                    self._search_best_a_scale(input_scale_candidates)
            else:
                break

        print("Activation scale:", self.a_quantizer.scale.mean().item())
        print("Log scale:", self.w_quantizer.scale.mean().item())
        print("Mean weight:", self.weight.mean().item())
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
    
    # def hyperparameter_searching(self):
    #     if self.a_quantizer.n_bits >= 32 and self.w_quantizer.n_bits >= 32:
    #         print("Skip calibration for 32-bit quantization")
    #         self.calibrated = True
    #         return None
    #     self._initialize_calib_parameters()
    #     if self.a_quantizer.n_bits < 32:
    #         if self.a_mode == 'log':
    #             self._initialize_activation_scale_log()
    #             input_scale_candidates = self.calculate_activation_candidates_log()
    #         else:
    #             self._initialize_activation_scale()
    #             input_scale_candidates = self.calculate_activation_candidates()
            
    #     # Get weight scale candidates
    #     weight_scale_candidates= self.calculate_percentile_weight_candidates()
    #     print("Weight Shape:", self.weight.shape)
    #     print("Out Channels:", self.out_channels)
    #     hi = torch.quantile(self.weight.view(self.out_channels, -1).abs(), 1, dim=-1).unsqueeze(-1)
    #     self.w_quantizer.scale.data.copy_(hi)
    #     self.w_quantizer.inited = True

    #     print("Activation scale:", self.a_quantizer.scale.mean().item())
    #     print("Log scale:", self.w_quantizer.scale.mean().item())
    #     print("Mean weight:", self.weight.mean().item())
    #     self.calibrated = True
    #     self.weight_deq = self.quant_weight_bias()[0]
    #     self.act_deq = self.quant_input(self.raw_input[:].cuda())
    #     self.weight_q = self.w_quantizer.quant_val
    #     self.act_q = self.a_quantizer.quant_val
    #     if self.act_q is None:
    #         self.act_q = self.act_deq
    #     if self.weight_q is None:
    #         self.weight_q = self.weight_deq
    #     # del self.raw_input, self.raw_out
    #     del self.raw_out
    #     return None


class TwoWordLogQuantConv2d(LogQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 fpcs = False,
                 steps = 4,
                 a_mode = 'linear'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, calib_batch_size, search_round, eq_n)
        self.fpcs = fpcs
        self.steps = steps
        del self.w_quantizer
        self.w_quantizer = SelectiveTwoWordLogQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True, threshold=0.04, second_word_ratio=None)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.second_word_ratio = 0
        
    def _search_best_w_scale(self, weight_scale_candidates, topk=1):
        batch_similarities = []
        tmp_w_quantizer = copy.deepcopy(self.w_quantizer)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1)
            
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]
                
                # # First word quantization
                w_sim = self.weight.view(self.out_channels, -1).unsqueeze(0)
                # w1 = self._quantize_weight(w_sim, cur_w_scale)
                
                # # Second word quantization for large values
                # err = (w_sim - w1) / cur_w_scale
                # mask = err.abs() > self.w_quantizer.threshold
                # w_residual = err * mask.float()
                # w2 = self._quantize_weight(w_residual, 1, offset=2, mask=1)
                
                # # Combine quantized weights
                # w_sim = (w1 + w2).view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

                tmp_w_quantizer.scale = nn.Parameter(cur_w_scale.squeeze(0))
                w_sim = tmp_w_quantizer(w_sim)  # shape: (parallel_eq_n, oc, ic*kw*kh)
                w_sim = w_sim.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
                
                # Calculate similarity with two-word quantization
                x_sim = self.quant_input(x)
                out_sim = F.conv2d(x_sim, w_sim, None, self.stride, self.padding, self.dilation, self.groups)
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1)
                
                similarity = self._get_similarity(raw_out, out_sim)
                similarity = torch.mean(similarity, [3,4])
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
                
            similarities = torch.cat(similarities, dim=1)
            batch_similarities.append(similarities)
            
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, -1, 1)
        
        if topk == 1:
            # print(_)
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
            
        return best_index
    
    def _quantize_weight(self, w, scale, offset=0, mask = 1):
        sign = torch.sign(w)
        mag = (w.abs() / scale).clamp_(min=1e-15, max=1.0)
        exp = torch.round(-torch.log2(mag)) * mask
        exp = torch.clamp(exp, 0 + offset, self.w_quantizer.n_levels - 1 + offset)
        return sign * (2.0 ** (-exp)) * scale

    def hyperparameter_searching(self):
        if self.a_quantizer.n_bits >= 32 and self.w_quantizer.n_bits >= 32:
            print("Skip calibration for 32-bit quantization")
            self.calibrated = True
            return None
        self._initialize_calib_parameters()
        if self.a_quantizer.n_bits < 8:
            self._initialize_activation_scale()
            input_scale_candidates = self.calculate_activation_candidates()
            
        # Get weight scale candidates
        weight_scale_candidates= self.calculate_percentile_weight_candidates()
        self.w_quantizer.scale.data.copy_(weight_scale_candidates[-2])
        
        for e in range(self.search_round):
            if self.fpcs:
                self.weight_fpcs(steps=self.steps, search_strategy=TwoWordLogQuantConv2d._search_best_w_scale)
            else:
                self._search_best_w_scale(weight_scale_candidates)
                
            if self.a_quantizer.n_bits < 8:
                self._search_best_a_scale(input_scale_candidates)
            else:
                break
        
        print("Log scale:", self.w_quantizer.scale.mean().item())
        print("Mean weight:", self.weight.mean().item())
        self.calibrated = True
        self.w_quantizer.inited = True
        self.weight_deq = self.quant_weight_bias()[0]
        self.act_deq = self.quant_input(self.raw_input[:].cuda())
        self.weight_q = self.w_quantizer.quant_val
        self.act_q = self.a_quantizer.quant_val
        if self.act_q is None:
            self.act_q = self.act_deq
        if self.weight_q is None:
            self.weight_q = self.weight_deq
        # del self.raw_input, self.raw_out
        self.second_word_ratio = self.w_quantizer.mask.float().mean().item()
        print("Second Word Ratio:", self.second_word_ratio)
        if self.a_mode == 'log':
            self.a_second_word_ratio = self.a_quantizer.mask.float().mean().item()
            print("Activation Second Word Ratio:", self.a_second_word_ratio)
        print("Mask shape:", self.w_quantizer.mask.shape)
        print("Threshold:", self.w_quantizer.threshold)
        return None

class TwoWordLogQuantConv2d2(LogQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 fpcs = False,
                 steps = 4,
                 a_mode = 'linear'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, calib_batch_size, search_round, eq_n)
        self.fpcs = fpcs
        self.steps = steps
        del self.w_quantizer
        self.w_quantizer = SelectiveTwoWordLogQuantizer2(n_bits = w_bit, symmetric = False, channel_wise = True, threshold=0.04, scale2=0.1)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.second_word_ratio = 0

        self.a_mode = a_mode
        if a_mode == 'log':
            del self.a_quantizer
            self.a_quantizer = SelectiveTwoWordLogQuantizer2(n_bits = a_bit, symmetric = False, channel_wise = True, threshold=0.04, scale2=0.1)
            self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))

    def _initialize_calib_parameters(self):
        self.calib_size = self.raw_input.shape[0]
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 4
            # memory = free_memory // 2  # Use half of the free memory for calibration
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (2 * self.raw_input[:self.calib_batch_size].numel() + 
                 2 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        
    def _search_best_w_scale(self, weight_scale_candidates, topk=1):
        batch_similarities = []
        tmp_w_quantizer = copy.deepcopy(self.w_quantizer)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1)
            
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]
                
                # First word quantization
                w_sim = self.weight.view(self.out_channels, -1).unsqueeze(0)
                # w1 = self._quantize_weight(w_sim, cur_w_scale)
                
                # # Second word quantization for large values
                # err = (w_sim - w1) / cur_w_scale
                # mask = err.abs() > self.w_quantizer.threshold
                # w_residual = err
                # w2 = self._quantize_weight(w_residual, cur_w_scale * self.w_quantizer.scale2)
                
                # # Combine quantized weights
                # w_sim = (w1 + w2 * mask.float()).view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

                tmp_w_quantizer.scale = nn.Parameter(cur_w_scale.squeeze(0))
                w_sim = tmp_w_quantizer(w_sim)  # shape: (parallel_eq_n, oc, ic*kw*kh)
                w_sim = w_sim.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
                
                # Calculate similarity with two-word quantization
                x_sim = self.quant_input(x)
                out_sim = F.conv2d(x_sim, w_sim, None, self.stride, self.padding, self.dilation, self.groups)
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1)
                
                similarity = self._get_similarity(raw_out, out_sim)
                similarity = torch.mean(similarity, [3,4])
                similarity = torch.sum(similarity, dim=0, keepdim=True)
                similarities.append(similarity)
                
            similarities = torch.cat(similarities, dim=1)
            batch_similarities.append(similarities)
            
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, -1, 1)
        
        if topk == 1:
            # print(_)
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
            
        return best_index
    
    def _quantize_weight(self, w, scale, offset=0, mask = 1):
        sign = torch.sign(w)
        mag = (w.abs() / scale).clamp_(min=1e-15, max=1.0)
        exp = torch.round(-torch.log2(mag)) * mask
        exp = torch.clamp(exp, 0 + offset, self.w_quantizer.n_levels - 1 + offset)
        return sign * (2.0 ** (-exp)) * scale

    def hyperparameter_searching(self):
        if self.a_quantizer.n_bits >= 32 and self.w_quantizer.n_bits >= 32:
            print("Skip calibration for 32-bit quantization")
            self.calibrated = True
            return None
        self._initialize_calib_parameters()
        if self.a_quantizer.n_bits < 32:
            if self.a_mode == 'log':
                self._initialize_activation_scale_log()
                input_scale_candidates = self.calculate_activation_candidates_log()
            else:
                self._initialize_activation_scale()
                input_scale_candidates = self.calculate_activation_candidates()
            
        # Get weight scale candidates
        weight_scale_candidates= self.calculate_percentile_weight_candidates()
        self.w_quantizer.scale.data.copy_(weight_scale_candidates[-2])
        
        for e in range(self.search_round):
            if self.fpcs:
                self.weight_fpcs(steps=self.steps, search_strategy=TwoWordLogQuantConv2d2._search_best_w_scale)
            else:
                self._search_best_w_scale(weight_scale_candidates)
                
            if self.a_quantizer.n_bits < 32:
                if self.a_mode == 'log':
                    self._search_best_a_scale_log(input_scale_candidates)
                else:
                    self._search_best_a_scale(input_scale_candidates)
            else:
                break
        
        print("Log scale:", self.w_quantizer.scale.mean().item())
        print("Mean weight:", self.weight.mean().item())
        self.calibrated = True
        self.w_quantizer.inited = True
        self.weight_deq = self.quant_weight_bias()[0]
        self.act_deq = self.quant_input(self.raw_input[:].cuda())
        self.weight_q = self.w_quantizer.quant_val
        self.act_q = self.a_quantizer.quant_val
        if self.act_q is None:
            self.act_q = self.act_deq
        if self.weight_q is None:
            self.weight_q = self.weight_deq
        # del self.raw_input, self.raw_out
        self.second_word_ratio = self.w_quantizer.mask.float().mean().item()
        print("Second Word Ratio:", self.second_word_ratio)
        if self.a_mode == 'log':
            self.a_second_word_ratio = self.a_quantizer.mask.float().mean().item()
            print("Activation Second Word Ratio:", self.a_second_word_ratio)
        print("Mask shape:", self.w_quantizer.mask.shape)
        print("Threshold:", self.w_quantizer.threshold)
        del self.raw_out
        return None