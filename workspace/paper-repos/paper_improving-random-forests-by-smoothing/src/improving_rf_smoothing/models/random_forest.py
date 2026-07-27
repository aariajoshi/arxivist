"""
Base Random Forest wrapper extracting tree structures.
"""
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

class TreeEnsemble(nn.Module):
    """
    Wraps a scikit-learn RandomForestRegressor and extracts tree structures for PyTorch.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_features=config.max_features,
            oob_score=True,
            random_state=42 # Will be set by seed util ideally
        )
        self.is_fitted = False
        
        # We will store the extracted tree bounding boxes and values here
        self.leaf_lower_bounds = [] # list of [num_leaves, p]
        self.leaf_upper_bounds = [] # list of [num_leaves, p]
        self.leaf_values = []       # list of [num_leaves]
        self.num_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the scikit-learn Random Forest.
        """
        self.rf.fit(X, y)
        self.is_fitted = True
        self.num_features = X.shape[1]
        self.extract_tree_structures()
        
    def _extract_single_tree(self, tree, num_features: int) -> Dict[str, torch.Tensor]:
        """
        Extract leaf bounding boxes for a single decision tree.
        """
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value.squeeze()
        
        # Initialize bounds [num_nodes, p]
        lower_bounds = np.full((n_nodes, num_features), -np.inf, dtype=np.float32)
        upper_bounds = np.full((n_nodes, num_features), np.inf, dtype=np.float32)
        
        # Traverse to compute bounds (DFS)
        stack = [(0, np.full(num_features, -np.inf), np.full(num_features, np.inf))]
        
        leaf_indices = []
        while len(stack) > 0:
            node_id, lb, ub = stack.pop()
            lower_bounds[node_id] = lb
            upper_bounds[node_id] = ub
            
            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                feat = feature[node_id]
                thresh = threshold[node_id]
                
                # Left child
                lb_left = lb.copy()
                ub_left = ub.copy()
                ub_left[feat] = min(ub_left[feat], thresh)
                stack.append((children_left[node_id], lb_left, ub_left))
                
                # Right child
                lb_right = lb.copy()
                ub_right = ub.copy()
                lb_right[feat] = max(lb_right[feat], thresh)
                stack.append((children_right[node_id], lb_right, ub_right))
            else:
                leaf_indices.append(node_id)
                
        # Filter for leaves only
        leaf_indices = np.array(leaf_indices)
        
        return {
            "lower_bounds": torch.tensor(lower_bounds[leaf_indices], dtype=torch.float32),
            "upper_bounds": torch.tensor(upper_bounds[leaf_indices], dtype=torch.float32),
            "values": torch.tensor(value[leaf_indices], dtype=torch.float32)
        }

    def extract_tree_structures(self) -> Dict[str, torch.Tensor]:
        """
        Extract tree structures into PyTorch tensors.
        """
        if not self.is_fitted:
            raise ValueError("Cannot extract tree structures before fitting.")
            
        self.leaf_lower_bounds = []
        self.leaf_upper_bounds = []
        self.leaf_values = []
        
        for estimator in self.rf.estimators_:
            tree_data = self._extract_single_tree(estimator.tree_, self.num_features)
            self.leaf_lower_bounds.append(tree_data["lower_bounds"])
            self.leaf_upper_bounds.append(tree_data["upper_bounds"])
            self.leaf_values.append(tree_data["values"])
            
        return {
            "lower": self.leaf_lower_bounds,
            "upper": self.leaf_upper_bounds,
            "values": self.leaf_values
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback forward pass using sklearn (for non-smoothed evaluation).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        device = x.device
        x_np = x.detach().cpu().numpy()
        preds = self.rf.predict(x_np)
        return torch.tensor(preds, dtype=torch.float32, device=device).unsqueeze(-1)
