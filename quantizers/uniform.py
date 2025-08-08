import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizers._ste import *
from torch.autograd import Function


class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False
        self.use_clip_forward = False

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        x_int = round_ste(x / self.scale) if self.training_mode else torch.round(x / self.scale)
        if self.sym:
            x_quant = x_int.clamp(-self.n_levels, self.n_levels - 1)
            x_dequant = x_quant * self.scale
        else:
            x_quant = (x_int + round_ste(self.zero_point)).clamp(0, 2 * self.n_levels - 1)
            x_dequant = (x_quant - round_ste(self.zero_point)) * self.scale
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'


class ShiftUniformQuantizer(UniformQuantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1)))
        self.register_buffer('bias_reparamed', torch.tensor(False))

    def forward(self, x):
        result = UniformQuantizer.forward(self, x + self.shift)
        return result if self.bias_reparamed else result - self.shift

    
class TwinUniformQuantizer(UniformQuantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
    
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited and self.scale.shape[0] == 2
        if self.training_mode:
            x_pos = round_ste(x / (self.scale[0])).clamp(0, self.n_levels - 1).mul(self.scale[0])
            x_neg = round_ste(x / (self.scale[1])).clamp(-self.n_levels, 0).mul(self.scale[1])
        else:
            x_pos = torch.round(x / (self.scale[0])).clamp(0, self.n_levels - 1).mul(self.scale[0])
            x_neg = torch.round(x / (self.scale[1])).clamp(-self.n_levels, 0).mul(self.scale[1])
        x_dequant = (x_pos + x_neg).reshape_as(x)
        return x_dequant
    
class _RoundSTE(Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g

round_auto = _RoundSTE.apply

class UniformIntQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False
        self.use_clip_forward = False
        self.register_buffer("quant_val", None)
        if not channel_wise:
            self.register_buffer("scale", torch.ones(1))
            self.register_buffer("zero_point", torch.zeros(1))

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
    
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.init_training()
        else:
            self.end_training()
        return self
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        zero_point = self.zero_point if not self.sym else 0
        x_int = x / self.scale + zero_point
        x_quant = round_ste(x_int) if self.training_mode else torch.round(x_int)
        x_quant = x_quant.clamp(-self.n_levels, self.n_levels - 1) if self.sym else x_quant.clamp(0, 2 * self.n_levels - 1)
        self.quant_val = x_quant
        x_dequant = (x_quant - zero_point) * self.scale
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'
    
class LogIntQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False
        self.use_clip_forward = False
        if not channel_wise:
            self.register_buffer("scale", torch.ones(1))

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False

    def _quant_exp(self, x):
        e = -torch.log2(x)
        e = round_ste(e) if self.training_mode else torch.round(e)
        return torch.clamp(e, 0, self.n_levels - 1)
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        sign = torch.sign(x)
        mag  = (x.abs() / self.scale).clamp(min=1e-15, max=1.0)
        e    = self._quant_exp(mag)
        return sign * (2.0 ** (-e)) * self.scale

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'


class TubLogQuantizer(nn.Module):
    """
    Log2 quantizer optimized for tubGEMM hardware.
    
    TubGEMM uses a modified 2-unary encoding scheme where each pulse
    represents a magnitude of 2. This quantizer aligns values with 
    powers of two while maximizing sparsity.
    """
    def __init__(self, n_bits=4, symmetric=True, channel_wise=False, sparsity_threshold=0.01):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** (n_bits - 1) if symmetric else 2 ** n_bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.sparsity_threshold = sparsity_threshold  # Values below this become zero
        self.inited = False
        
        # Register scale parameter for quantization - will be optimized during calibration
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        
        # For tubGEMM's temporal-unary scaling (each pulse = 2)
        self.register_buffer('base', torch.tensor(2.0))
        
        # Adaptive base parameters (from AdaLog)
        self.r = 37.0  # Fixed parameter from AdaLog
        self.register_parameter('q', nn.Parameter(torch.tensor([20.0])))  # Adaptive base parameter
        
        # Setup lookup table for quantized values
        self.register_buffer('lookup_table', torch.zeros(2 * self.n_levels))
        
        # For tracking bias reparameterization
        self.register_buffer('bias_reparamed', torch.tensor(False))
        
        # For activation-aware quantization
        self.register_buffer('act_running_mean', None)
        self.register_buffer('act_running_var', None)
        self.register_buffer('act_samples_seen', torch.tensor(0))
        self.activation_aware = False  # Flag to enable activation-aware mode
        
        # For tracking q parameter values
        self.register_buffer('q_history', torch.zeros(0))

    # Add a method to update activation statistics during forward pass
    def update_activation_stats(self, x, momentum=0.1):
        """Track activation statistics during forward passes"""
        if self.act_running_mean is None:
            # Initialize on first call with appropriate shape
            self.act_running_mean = torch.zeros(x.size(1) if x.dim() > 1 and self.channel_wise else 1, 
                                            device=x.device)
            self.act_running_var = torch.ones(x.size(1) if x.dim() > 1 and self.channel_wise else 1,
                                            device=x.device)
        
        # Update running statistics
        if x.dim() > 1 and self.channel_wise:
            # Calculate per-channel statistics
            dims = [0] + list(range(2, x.dim()))
            current_mean = x.mean(dim=dims)
            current_var = x.var(dim=dims, unbiased=False)
        else:
            # Calculate global statistics
            current_mean = x.mean()
            current_var = x.var(unbiased=False)
        
        # Update running stats with momentum
        if self.act_samples_seen == 0:
            self.act_running_mean.copy_(current_mean)
            self.act_running_var.copy_(current_var)
        else:
            self.act_running_mean.mul_(1 - momentum).add_(current_mean * momentum)
            self.act_running_var.mul_(1 - momentum).add_(current_var * momentum)
        
        self.act_samples_seen += 1
    
    def update_lookup_table(self):
        """Update the lookup table based on current q value"""
        for i in range(0, 2 * self.n_levels):
            # Value representation aligned with tubGEMM's 2-unary encoding
            # Each position in the table represents a specific number of pulses
            exponent = torch.floor(torch.tensor(i * self.q.item() / self.r))
            fraction = torch.remainder(torch.tensor(i * self.q.item()), self.r) / self.r
            
            # Powers of 2 are naturally aligned with tubGEMM's encoding
            val = (2.0 ** (-exponent)) * (2.0 ** (-fraction))
            self.lookup_table[i] = val
    
    def forward(self, x):
        """
        Quantize input tensor optimized for tubGEMM hardware
        """
        # Track activation statistics if enabled
        if self.training and self.activation_aware:
            self.update_activation_stats(x)
            
        if self.n_bits == 32:
            return x  # Skip quantization for full precision
        
        if not self.inited:
            # Auto-calibrate instead of raising error
            print(f"Auto-calibrating quantizer for shape {x.shape}")
            self.calibrate(x)
        
        # Store original tensor for debugging
        x_orig = x.clone()
        
        # Apply scale - handle different shapes, with safe guard against zero scale
        if self.channel_wise and x.dim() > 1:
            # Check if scale needs reshaping for per-channel scaling
            if self.scale.dim() == 1 and self.scale.shape[0] != x.shape[0]:
                # Re-calibrate with the current tensor shape
                print(f"Re-calibrating due to shape mismatch: {self.scale.shape[0]} vs {x.shape[0]}")
                self.calibrate(x)
            
            # Apply per-channel scaling by reshaping scale
            # Ensure scale has correct dimensionality
            if self.scale.dim() == 1:
                scale_view = self.scale.view(-1, *([1] * (x.dim() - 1)))
            else:
                # If scale is already a scalar, unsqueeze it
                scale_view = self.scale.view(*([1] * x.dim()))
            
            # SAFETY: Ensure scale is not zero - add small epsilon
            safe_scale = scale_view.clone()
            safe_scale[safe_scale < 1e-6] = 1e-6  # Minimum safe scale value
            x_scaled = x / safe_scale
        else:
            # Global scaling with safety
            safe_scale = self.scale.clone()
            safe_scale[safe_scale < 1e-6] = 1e-6  # Minimum safe scale value
            x_scaled = x / safe_scale
        
        # Debug print for scaling
        if not hasattr(self, 'debug_scaling'):
            self.debug_scaling = True
            print(f"SCALING DEBUG - scale: {safe_scale.mean().item():.8f}, input: {x.mean().item():.8f}, scaled: {x_scaled.mean().item():.8f}")
        
        # Skip processing for zeros - handle them separately
        zeros_mask = (x == 0)
        
        # Process only non-zero values
        x_nonzero = x_scaled.clone()
        
        # Calculate log2 representation
        eps = 1e-10  # Small value to avoid log(0)
        x_abs = torch.abs(x_nonzero).clamp(min=eps)
        x_log = -torch.log2(x_abs)# * self.r / self.q
        
        # Handle NaN values - replace with 0
        nan_mask = torch.isnan(x_log)
        if nan_mask.any():
            print(f"WARNING: NaN values detected in log calculation ({nan_mask.sum().item()} values). Replacing with 0.")
            x_log[nan_mask] = 0
        
        # Round to levels
        x_quant = torch.round(x_log)
        
        # Clamp to valid range
        x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
        
        # Convert quantized indices to actual values using lookup table
        # lookup_values = self.lookup_table[x_quant.long()]
        # lookup_values = 2.0 ** (-(x_quant * self.q / self.r))
        lookup_values = 2.0 ** (-x_quant)
        
        # Apply scaling back
        if self.channel_wise and x.dim() > 1:
            scale_view = self.scale.view(-1, *([1] * (x.dim() - 1)))
            reconstructed_values = lookup_values * scale_view
        else:
            reconstructed_values = lookup_values * self.scale
        
        # Apply signs from original input
        signs = torch.sign(x_nonzero)
        reconstructed_values = reconstructed_values * signs
        
        # Identify small values for sparsification
        # NOTE: We're now using x_scaled for the comparison, not x
        small_values_mask = torch.abs(x_scaled) < self.sparsity_threshold
        
        # Create output tensor starting with zeros
        output = torch.zeros_like(x)
        
        # Only copy non-small, non-zero values
        copy_mask = ~small_values_mask & ~zeros_mask
        output[copy_mask] = reconstructed_values[copy_mask]
        
        # Debug stats
        if not hasattr(self, 'debug_stats_printed'):
            self.debug_stats_printed = True
            zeros_count = zeros_mask.sum().item()
            sparsified_count = small_values_mask.sum().item()
            preserved_count = copy_mask.sum().item()
            total_count = x.numel()
            
            print(f"QUANTIZER STATS:")
            print(f"  Total values: {total_count}")
            print(f"  Original zeros: {zeros_count} ({zeros_count/total_count*100:.2f}%)")
            print(f"  Sparsified: {sparsified_count} ({sparsified_count/total_count*100:.2f}%)")
            print(f"  Preserved: {preserved_count} ({preserved_count/total_count*100:.2f}%)")
            print(f"  Output zeros: {(output == 0).sum().item()/total_count*100:.2f}%")
            print(f"  Output stats: min={output.min().item():.8f}, max={output.max().item():.8f}, mean={output.mean().item():.8f}")
            print(f"  Input stats: min={x_orig.min().item():.8f}, max={x_orig.max().item():.8f}, mean={x_orig.mean().item():.8f}")
        
        return output
    
    def calibrate(self, x, activation_data=None, percentile=99.9):
        """
        Calibrate the quantizer on a representative dataset
        
        Args:
            x: The tensor to calibrate on (weights or activations)
            activation_data: Optional activation statistics for activation-aware calibration
            percentile: Percentile for outlier protection
        """
        # Get maximum absolute value (with outlier protection using percentile)
        with torch.no_grad():
            # First determine if we're doing per-channel scaling
            do_channel_wise = self.channel_wise and x.dim() > 1
            
            if do_channel_wise:
                # Compute per-channel scaling factors
                reshaped_x = x.reshape(x.shape[0], -1)
                # Ensure there's at least one non-zero element in each channel
                has_nonzeros = (reshaped_x.abs().sum(dim=1) > 0)
                
                # Only compute for channels with non-zero values
                if has_nonzeros.any():
                    valid_channels = torch.where(has_nonzeros)[0]
                    valid_x = reshaped_x[valid_channels]
                    
                    # Initialize abs_max tensor with the correct shape
                    abs_max = torch.zeros(x.shape[0], device=x.device)
                    
                    # Compute percentile for valid channels
                    for i, channel_idx in enumerate(valid_channels):
                        channel_data = valid_x[i]
                        channel_abs = torch.abs(channel_data)
                        if channel_abs.numel() > 0:
                            abs_max[channel_idx] = torch.quantile(channel_abs, percentile/100)
                    
                    # For invalid channels, use a small default value
                    abs_max[~has_nonzeros] = 1e-4
                else:
                    # All zeros, use default small value
                    abs_max = torch.ones(x.shape[0], device=x.device) * 1e-4
                
                # Setup for per-channel scaling
                safe_scale = abs_max / (2**(self.n_bits-1))  # Per-channel scaling
                safe_scale = torch.clamp(safe_scale, min=1e-6)  # Ensure non-zero
                
                # Reshape scale parameter to match the size of abs_max
                if self.scale.shape != safe_scale.shape:
                    self.scale = nn.Parameter(torch.ones_like(safe_scale))
                
                # Update scale data
                self.scale.data = safe_scale
            else:
                # Compute global scaling factor, with safety checks
                if torch.all(x == 0):
                    # Handle all-zero tensors
                    abs_max = torch.tensor(1e-4, device=x.device)
                else:
                    # Normal case - non-zero tensor
                    abs_max = torch.quantile(torch.abs(x), percentile/100)
                    if abs_max == 0 or torch.isnan(abs_max):
                        # Safeguard against invalid values
                        abs_max = torch.tensor(1e-4, device=x.device)
                
                # Global scaling
                safe_scale = abs_max / (2**(self.n_bits-1))
                safe_scale = torch.clamp(safe_scale, min=1e-6)  # Ensure non-zero
                
                # Reset scale parameter to be a scalar
                if self.scale.dim() != 0 and self.scale.numel() != 1:
                    self.scale = nn.Parameter(torch.ones(1, device=x.device))
                
                # Update scale data
                self.scale.data = safe_scale
            
            # Optimize q parameter for the data distribution with activation awareness
            self._optimize_log_base(x, activation_data)
            
            # Update lookup table with optimized parameters
            self.update_lookup_table()
            
            # Set initialization flag
            self.inited = True
            
            # Debug output
            if self.scale.numel() == 1:
                scale_val = self.scale.item()
            else:
                scale_val = self.scale.mean().item()
            print(f"Calibration complete - scale={scale_val:.8f}, q={self.q.item():.6f}")
            
        return self
    
    def _optimize_log_base(self, x, activation_data=None):
        """
        Find optimal logarithmic base parameter (q) for given data distribution
        with optional activation awareness
        
        Args:
            x: The weight tensor to optimize for
            activation_data: Optional tensor or dict containing activation statistics
        """
        # Skip for very small tensors
        if x.numel() < 100:
            return
        
        # Sample data for optimization
        with torch.no_grad():
            # Take a subset of the data for efficiency
            if x.numel() > 10000:
                indices = torch.randperm(x.numel())[:10000]
                samples = x.flatten()[indices]
            else:
                samples = x.flatten()
            
            # Remove zeros to avoid log(0) issues
            samples = samples[samples != 0]
            if samples.numel() == 0:
                # If all values are zeros, keep the default q value
                return
            
            # Take log2 of absolute values
            log_samples = -torch.log2(torch.abs(samples).clamp(min=1e-10))

            coarse_q_value = 35.0
            self.q.data = torch.tensor([coarse_q_value])
    
            # Optional: Print information about the hardcoded value
            print(f"Using hardcoded q value: {coarse_q_value:.4f} for coarse quantization")
            
            # # Analyze distribution to find optimal q
            # q_values = torch.linspace(10, 40, 31)  # Range of possible q values
            # errors = []
            
            # # Test different q values
            # for q_val in q_values:
            #     # Quantize and dequantize with current q
            #     quantized = torch.round(log_samples * self.r / q_val)
            #     dequantized = 2.0 ** (-quantized * q_val / self.r)
                
            #     # Basic quantization error
            #     basic_error = F.mse_loss(dequantized, torch.abs(samples))
                
            #     # If we have activation information and activation-aware mode is enabled,
            #     # adjust the error calculation
            #     if activation_data is not None and self.activation_aware:
            #         # Extract activation statistics
            #         if isinstance(activation_data, dict):
            #             # If statistics were passed as a dictionary
            #             act_mean = activation_data.get('mean', 0)
            #             act_var = activation_data.get('var', 1)
            #         elif isinstance(activation_data, torch.Tensor):
            #             # If a tensor was passed
            #             act_mean = activation_data.mean().item()
            #             act_var = activation_data.var().item()
            #         elif self.act_running_mean is not None:
            #             # If we have our own running statistics
            #             act_mean = self.act_running_mean.mean().item()
            #             act_var = self.act_running_var.mean().item()
            #         else:
            #             # No valid activation data
            #             act_mean, act_var = 0, 1
                    
            #         # Calculate importance factor - weights that interact with
            #         # more active neurons should be quantized more precisely
            #         if isinstance(act_mean, torch.Tensor):
            #             act_mean = act_mean.item()
            #         if isinstance(act_var, torch.Tensor):
            #             act_var = act_var.item()
                    
            #         # Normalize to [0.8, 1.2] range to avoid extreme adjustments
            #         importance_factor = 1.0 + 0.2 * math.tanh(act_mean / math.sqrt(act_var + 1e-8))
                    
            #         # Adjust error based on activation importance
            #         adjusted_error = basic_error * importance_factor
            #         errors.append(adjusted_error.item())
            #     else:
            #         # Without activation info, just use the basic error
            #         errors.append(basic_error.item())
            
            # # Find q with minimum error
            # best_q_idx = torch.argmin(torch.tensor(errors))
            # best_q = q_values[best_q_idx]
            
            # # Update q parameter
            # self.q.data = torch.tensor([best_q])
            
            # # Track q values for analysis
            # new_history = torch.ones(self.q_history.shape[0] + 1, device=self.q_history.device)
            # new_history[:self.q_history.shape[0]] = self.q_history
            # new_history[-1] = best_q
            # self.q_history = new_history
            
            # # Print the optimized value - helpful for monitoring
            # print(f"Optimized q value: {best_q.item():.4f}")

    def get_q_history(self):
        """Return the history of optimized q values"""
        return self.q_history.cpu().numpy()