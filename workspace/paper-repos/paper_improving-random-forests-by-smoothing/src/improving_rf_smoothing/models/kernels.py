"""
Kernel functions for smoothing.
Implements the probability kernels described in Section 2.2 of the paper.
"""
import math
import torch
import torch.nn as nn

class GaussianKernel(nn.Module):
    """
    Gaussian Kernel.
    Paper reference: Section 2.2
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, z: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gaussian kernel k(z|x, Sigma).
        Eq: k_gs(z|x,Sigma) = prod_{j=1}^p (1/sqrt(2*pi*sigma_j^2)) * exp(-(z^(j)-x^(j))^2 / (2*sigma_j^2))
        
        Args:
            z: Target points [B, p] or [1, p]
            x: Source points [B, p]
            sigma: Standard deviations [..., p]
            
        Returns:
            Kernel values
        """
        # Eq in Section 2.2: Gaussian kernel
        diff = z - x
        exponent = -0.5 * (diff / sigma) ** 2
        norm = sigma * math.sqrt(2 * math.pi)
        return torch.prod(torch.exp(exponent) / norm, dim=-1)

    def cdf(self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Compute the CDF of the Gaussian kernel, used for integrating over leaf bounding boxes.
        
        Args:
            x: Evaluation points [..., p]
            loc: Means [..., p]
            scale: Standard deviations [..., p]
            
        Returns:
            CDF values [..., p] (factorized per dimension)
        """
        return 0.5 * (1 + torch.erf((x - loc) / (scale * math.sqrt(2))))

class HyperbolicSecantKernel(nn.Module):
    """
    Hyperbolic Secant Kernel.
    Paper reference: Section 2.2
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, z: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hyperbolic Secant kernel k(z|x, Sigma).
        Eq: k_hs(z|x,Sigma) = prod_{j=1}^p (1/(2*sigma_j)) * sech(pi*(z^(j)-x^(j)) / (2*sigma_j))
        
        Args:
            z: Target points [B, p]
            x: Source points [B, p]
            sigma: Scale parameters [..., p]
            
        Returns:
            Kernel values
        """
        # Eq in Section 2.2: Hyperbolic Secant kernel
        diff = z - x
        arg = (math.pi * diff) / (2 * sigma)
        sech = 1.0 / torch.cosh(arg)
        return torch.prod((1.0 / (2 * sigma)) * sech, dim=-1)

    def cdf(self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Compute the CDF of the Hyperbolic Secant kernel.
        CDF of sech(pi*x/2) is (2/pi) * arctan(exp(pi*x/2)).
        
        Args:
            x: Evaluation points [..., p]
            loc: Location parameters [..., p]
            scale: Scale parameters [..., p]
            
        Returns:
            CDF values [..., p] (factorized per dimension)
        """
        arg = (math.pi * (x - loc)) / (2 * scale)
        return (2.0 / math.pi) * torch.atan(torch.exp(arg))
