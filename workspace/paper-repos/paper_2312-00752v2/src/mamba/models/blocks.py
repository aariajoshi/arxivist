"""
Core Mamba blocks and SSM implementations.
Implements the Mamba Block, Input Projections, 1D Convolution, Selective SSM (S6),
Nonlinearity, and Output Projection (Section 3.4).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
    HAS_CUDA_OPS = True
except ImportError:
    HAS_CUDA_OPS = False

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6). (Section 3.2)
    Uses input-dependent parameters B, C, Delta.
    """
    def __init__(self, d_model: int, d_state: int, dt_rank: int | str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        if dt_rank == "auto":
            dt_rank = math.ceil(self.d_model / 16)
        self.dt_rank = int(dt_rank)
        
        # Eq. 3 in Section 3.2: Selection Mechanism
        self.x_proj = nn.Linear(self.d_model, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True)
        
        # A parameter
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Selective SSM.
        Args:
            x: Input tensor of shape [B, L, D]
        Returns:
            Output tensor of shape [B, L, D]
        """
        assert x.dim() == 3, f"Expected [B,L,D], got {x.shape}"
        B, L, D = x.shape
        
        # Eq 3: B_t = Linear(x_t), C_t = Linear(x_t), Delta_t = softplus(Parameter + Linear(x_t))
        x_proj = self.x_proj(x) # [B, L, dt_rank + 2*d_state]
        dt, B_mat, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # [B, L, D]
        
        # Reshape for scan
        x = x.transpose(1, 2) # [B, D, L]
        dt = dt.transpose(1, 2) # [B, D, L]
        B_mat = B_mat.transpose(1, 2) # [B, N, L]
        C = C.transpose(1, 2) # [B, N, L]
        A = -torch.exp(self.A_log.float()) # [D, N]
        
        if HAS_CUDA_OPS and x.is_cuda:
            # Use hardware-aware parallel algorithm with scan
            out = selective_scan_fn(
                x, dt, A, B_mat, C, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True
            )
        else:
            # Fallback pure PyTorch implementation (WARNING: slow)
            # ZOH Discretization and Linear Recurrence (Section 2)
            out = self._selective_scan_fallback(x, dt, A, B_mat, C, self.D, self.dt_proj.bias)
            
        return out.transpose(1, 2) # [B, L, D]
        
    def _selective_scan_fallback(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_mat: torch.Tensor, C: torch.Tensor, D: torch.Tensor, delta_bias: torch.Tensor) -> torch.Tensor:
        """Fallback for selective scan when custom CUDA kernels are not available."""
        # STUB: Pure PyTorch fallback for CPU/debugging.
        # This is very slow and uses a lot of memory.
        B_sz, D_sz, L = x.shape
        N = A.shape[1]
        
        dt = F.softplus(dt + delta_bias.view(1, -1, 1))
        
        dA = torch.exp(torch.einsum('bdl,dn->bdln', dt, A)) # [B, D, L, N]
        dB_x = torch.einsum('bdl,bnl,bdl->bdln', dt, B_mat, x) # [B, D, L, N]
        
        h = torch.zeros((B_sz, D_sz, N), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, :, t, :] * h + dB_x[:, :, t, :]
            y = torch.einsum('bdn,bn->bd', h, C[:, :, t])
            ys.append(y)
            
        y = torch.stack(ys, dim=2) # [B, D, L]
        y = y + x * D.view(1, -1, 1)
        return y


class MambaBlock(nn.Module):
    """
    Mamba Block (Section 3.4)
    Combines H3 and MLP blocks into a single block without attention.
    """
    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Input Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Selective SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state)
        
        # Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mamba Block.
        Args:
            x: Input tensor of shape [B, L, D]
        Returns:
            Output tensor of shape [B, L, D]
        """
        assert x.dim() == 3, f"Expected [B,L,D], got {x.shape}"
        
        # Input Projections -> [B, L, E * 2]
        x_proj = self.in_proj(x)
        x_proj, z = x_proj.split(self.d_inner, dim=-1)
        
        # 1D Convolution
        x_conv = x_proj.transpose(1, 2) # [B, E, L]
        if HAS_CUDA_OPS and x_conv.is_cuda:
            x_conv = causal_conv1d_fn(
                x_conv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation="silu"
            )
        else:
            x_conv = self.conv1d(x_conv)[..., :x.shape[1]]
            # Nonlinearity (Section 3.4) - SiLU/Swish
            x_conv = F.silu(x_conv)
            
        x_conv = x_conv.transpose(1, 2) # [B, L, E]
        
        # Selective SSM (S6)
        x_ssm = self.ssm(x_conv)
        
        # Nonlinearity on the other branch (gating)
        out = x_ssm * F.silu(z)
        
        # Output Projection
        out = self.out_proj(out)
        
        return out
