import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from quant_layers import MinMaxQuantMatMul, MinMaxQuantConv2d, MinMaxQuantLinear, TwoWordLogQuantConv2d2, TwoWordLogQuantConv2d, TwoWordLogQuantLinear
import matplotlib.pyplot as plt  # <-- Add this import

class QuantCalibrator:
    def __init__(self, model, calib_loader):
        self.model = model
        self.calib_loader = calib_loader
        
    def single_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append(inp[0].cpu().detach())
        
    def double_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1].cpu().detach())
    
    def outp_forward_hook(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append(outp.cpu().detach())

    def batching_quant_calib(self):
        device = next(self.model.parameters()).device
        total = sum(1 for name, module in self.model.named_modules() if hasattr(module, 'calibrated') and not module.calibrated)
        second_word_ratio = []

        weight_hist_data = []
        input_hist_data = []

        weight_q_hist_data = []
        input_q_hist_data = []

        weight_deq_hist_data = []
        input_deq_hist_data = []

        with tqdm(total=total) as progress_bar:
            for name, module in self.model.named_modules():
                if not hasattr(module, 'calibrated') or module.calibrated:
                    continue
                progress_bar.set_description(f"calibrating {name}")
                hooks = []
                hooks.append(module.register_forward_hook(self.outp_forward_hook))
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantMatMul):
                    hooks.append(module.register_forward_hook(self.double_input_forward_hook))
                with torch.no_grad():
                    for i, (inp, target) in enumerate(self.calib_loader):
                        inp = inp.to(device)
                        _ = self.model(inp)
                # replace cached raw_inputs, raw_outs
                module.raw_out = torch.cat(module.tmp_out, dim=0)
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(_, dim=0) for _ in module.tmp_input]
                for hook in hooks:
                    hook.remove()
                module.tmp_input = module.tmp_out = None
                # run hyperparameter_searching
                with torch.no_grad():
                    module.hyperparameter_searching()
                    if isinstance(module, TwoWordLogQuantConv2d) or isinstance(module, TwoWordLogQuantConv2d2) or isinstance(module, TwoWordLogQuantLinear):
                        second_word_ratio.append(module.second_word_ratio)
                        pass
                    if hasattr(module, 'prev_layer') and module.prev_layer is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam()
                if hasattr(module, 'weight'):
                    w = module.weight.detach().cpu().numpy().flatten()
                    weight_hist_data.append((name, w))
                if hasattr(module, 'raw_input') and isinstance(module.raw_input, torch.Tensor):
                    inp = module.raw_input.detach().cpu().numpy().flatten()
                    input_hist_data.append((name, inp))
                    del module.raw_input  # save memory
                if hasattr(module, 'weight_q'):
                    wq = module.weight_q.detach().cpu().numpy().flatten()
                    weight_q_hist_data.append((name, wq))
                    del module.weight_q  # save memory
                if hasattr(module, 'act_q'):
                    inq = module.act_q.detach().cpu().numpy().flatten()
                    input_q_hist_data.append((name, inq))
                    del module.act_q  # save memory
                if hasattr(module, 'weight_deq'):
                    wq = module.weight_deq.detach().cpu().numpy().flatten()
                    weight_deq_hist_data.append((name, wq))
                    del module.weight_deq  # save memory
                if hasattr(module, 'act_deq'):
                    inq = module.act_deq.detach().cpu().numpy().flatten()
                    input_deq_hist_data.append((name, inq))
                    del module.act_deq  # save memory
                progress_bar.update()
        # end calibration
        # Plot all histograms in a single figure after calibration
        if weight_hist_data:
            n = len(weight_hist_data)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1)  # flatten in case axes is 2D

            for idx, (name, w) in enumerate(weight_hist_data):
                ax = axes[idx]
                ax.hist(w, bins=100, alpha=0.75)
                ax.set_title(f"Weight: {name}")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Frequency")
                ax.grid(True)
            # Hide unused subplots
            for idx in range(len(weight_hist_data), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig("all_weight_histograms.png")
            plt.close()

        if input_hist_data:
            n = len(input_hist_data)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1)
            for idx, (name, inp) in enumerate(input_hist_data):
                ax = axes[idx]
                inp = inp[np.isfinite(inp)]
                if inp.size == 0 or np.all(inp == inp[0]):
                    # Plot a single bin or skip
                    ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
                    ax.set_title(f"Input: {name}")
                    ax.set_xlabel("Input Value")
                    ax.set_ylabel("Ratio")
                    ax.grid(True)
                    continue
                ax.hist(inp, bins=100, alpha=0.75, color='orange')
                ax.set_title(f"Input: {name}")
                ax.set_xlabel("Input Value")
                ax.set_ylabel("Frequency")
                ax.grid(True)
            for idx in range(len(input_hist_data), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig("all_input_histograms.png")
            plt.close()
        
        if weight_q_hist_data:
            n = len(weight_q_hist_data)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1)
            for idx, (name, wq) in enumerate(weight_q_hist_data):
                ax = axes[idx]
                ax.hist(wq, bins=100, alpha=0.75, color='green')
                ax.set_title(f"Quantized Weight: {name}")
                ax.set_xlabel("Quantized Weight Value")
                ax.set_ylabel("Frequency")
                ax.grid(True)
            for idx in range(len(weight_q_hist_data), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig("all_quantized_weight_histograms.png")
            plt.close()
        
        if input_q_hist_data:
            n = len(input_q_hist_data)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1)
            for idx, (name, inq) in enumerate(input_q_hist_data):
                ax = axes[idx]
                ax.hist(inq, bins=100, alpha=0.75, color='red')
                ax.set_title(f"Quantized Input: {name}")
                ax.set_xlabel("Quantized Input Value")
                ax.set_ylabel("Frequency")
                ax.grid(True)
            for idx in range(len(input_q_hist_data), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig("all_quantized_input_histograms.png")
            plt.close()
        
        if weight_deq_hist_data:
            n = len(weight_deq_hist_data)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1)
            for idx, (name, wdq) in enumerate(weight_deq_hist_data):
                ax = axes[idx]
                ax.hist(wdq, bins=100, alpha=0.75, color='purple')
                ax.set_title(f"Dequantized Weight: {name}")
                ax.set_xlabel("Dequantized Weight Value")
                ax.set_ylabel("Frequency")
                ax.grid(True)
            for idx in range(len(weight_deq_hist_data), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig("all_dequantized_weight_histograms.png")
            plt.close()
        
        if input_deq_hist_data:
            n = len(input_deq_hist_data)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1)
            for idx, (name, indq) in enumerate(input_deq_hist_data):
                ax = axes[idx]
                ax.hist(indq, bins=100, alpha=0.75, color='brown')
                ax.set_title(f"Dequantized Input: {name}")
                ax.set_xlabel("Dequantized Input Value")
                ax.set_ylabel("Frequency")
                ax.grid(True)
            for idx in range(len(input_deq_hist_data), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.savefig("all_dequantized_input_histograms.png")
            plt.close()

        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"
        
        if len(second_word_ratio) > 0:
            print("Average Second Word Ratio:", sum(second_word_ratio) / len(second_word_ratio))
