"""
Smoothed Random Forest containing smoothing logic and parameters.
"""
import torch
import torch.nn as nn
from typing import Any
from .random_forest import TreeEnsemble
from .kernels import GaussianKernel, HyperbolicSecantKernel

class SmoothedRandomForest(nn.Module):
    """
    Probabilistic Smoothing for Random Forests.
    Paper Reference: Section 2.2
    """
    def __init__(self, config: Any, tree_ensemble: TreeEnsemble):
        super().__init__()
        self.config = config
        self.tree_ensemble = tree_ensemble
        
        assert self.tree_ensemble.is_fitted, "TreeEnsemble must be fitted before initializing SmoothedRandomForest"
        
        self.num_trees = config.n_estimators
        self.num_features = self.tree_ensemble.num_features
        self.variant = config.variant
        
        # Setup Kernel
        if config.kernel.lower() == "gaussian":
            self.kernel = GaussianKernel()
        elif config.kernel.lower() in ["hyperbolicsecant", "hs", "hyperbolic_secant"]:
            self.kernel = HyperbolicSecantKernel()
        else:
            raise ValueError(f"Unknown kernel: {config.kernel}")
            
        # Initialize Smoothing Parameters (Sigma)
        # We model log_sigma to ensure positivity
        if self.variant == "EST-PD":
            self.log_sigma = nn.Parameter(torch.zeros(self.num_trees, self.num_features))
        elif self.variant == "EST":
            self.log_sigma = nn.Parameter(torch.zeros(self.num_trees, 1))
        elif self.variant == "STE-PD":
            self.log_sigma = nn.Parameter(torch.zeros(1, self.num_features))
        elif self.variant == "STE":
            self.log_sigma = nn.Parameter(torch.zeros(1, 1))
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
            
        # Initialize Re-calibration parameters
        self.beta_0 = nn.Parameter(torch.zeros(1))
        self.beta_1 = nn.Parameter(torch.ones(1))
        
    def get_sigma(self, tree_idx: int) -> torch.Tensor:
        """
        Get the standard deviation parameter for a specific tree.
        """
        sigma = torch.exp(self.log_sigma)
        if self.variant == "EST-PD" or self.variant == "EST":
            return sigma[tree_idx] # [p] or [1]
        else:
            return sigma[0] # [p] or [1]

    def compute_smoothed_tree(self, x: torch.Tensor, tree_idx: int) -> torch.Tensor:
        """
        Compute the smoothed prediction for a single tree.
        Eq in Section 2.2: \tilde{f}_t(x) = \sum_{leaf} value_{leaf} * P_{z~k}(z \in LeafBox | x, \Sigma_t)
        
        Args:
            x: Input tensor [B, p]
            tree_idx: Index of the tree
            
        Returns:
            Smoothed tree prediction [B]
        """
        # Get tree structures
        lb = self.tree_ensemble.leaf_lower_bounds[tree_idx].to(x.device) # [num_leaves, p]
        ub = self.tree_ensemble.leaf_upper_bounds[tree_idx].to(x.device) # [num_leaves, p]
        values = self.tree_ensemble.leaf_values[tree_idx].to(x.device)   # [num_leaves]
        
        sigma = self.get_sigma(tree_idx) # [p] or [1]
        
        B = x.shape[0]
        
        # Broadcast safely to avoid OOM
        x_exp = x.unsqueeze(1)
        lb_exp = lb.unsqueeze(0)
        ub_exp = ub.unsqueeze(0)
        
        if sigma.dim() == 1:
            sigma_exp = sigma.view(1, 1, -1)
        else:
            sigma_exp = sigma.view(1, 1, 1)
            
        # P(z \in [lb, ub]) = \prod_{j=1}^p (CDF(ub_j) - CDF(lb_j))
        cdf_ub = self.kernel.cdf(ub_exp, x_exp, sigma_exp)
        cdf_lb = self.kernel.cdf(lb_exp, x_exp, sigma_exp)
        
        prob_mass_dim = cdf_ub - cdf_lb # [B, num_leaves, p]
        prob_mass = torch.prod(prob_mass_dim, dim=2) # [B, num_leaves]
        
        f_tilde_t = torch.sum(prob_mass * values.unsqueeze(0), dim=1) # [B]
        
        return f_tilde_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Smoothed Random Forest.
        Eq in Section 2.2: \tilde{y}(x) = \beta_1 * \frac{1}{T} \sum_{t=1}^T \tilde{f}_t(x) + \beta_0
        
        Args:
            x: Input features [B, p]
            
        Returns:
            Smoothed predictions [B, 1]
        """
        assert x.dim() == 2, f"Expected [B, p], got {x.shape}"
        
        B = x.shape[0]
        f_tilde_sum = torch.zeros(B, device=x.device)
        
        # Process trees in loop to save memory
        for t in range(self.num_trees):
            f_tilde_t = self.compute_smoothed_tree(x, t)
            f_tilde_sum += f_tilde_t
            
        f_tilde = f_tilde_sum / self.num_trees
        
        y_tilde = self.beta_1 * f_tilde + self.beta_0
        return y_tilde.unsqueeze(-1)
