"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch.nn.functional as F   # add this at the top


from attention import CrossModalAttention


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(self,
                 modality_dims: dict,
                 num_classes: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 **kwargs):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modality_names = list(modality_dims.keys())
        
        # TODO: Implement early fusion architecture
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:
        #   concat_dim = sum(modality_dims.values())
        #   Linear(concat_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, num_classes)
        concat_dim = sum(modality_dims.values())
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Extract features for each modality from dict
        #   2. Handle missing modalities (use zeros or learned embeddings)
        #   3. Concatenate all features
        #   4. Pass through fusion network
        batch = next(iter(modality_features.values()))
        batch_size, device = batch.size(0), batch.device          # <-- device safely
        feats_list = []
        
        for mod in self.modality_names:
            if mod in modality_features:
                feat = modality_features[mod]
            else:
                feat = torch.zeros(batch_size, self.modality_dims[mod], device=device)
            feats_list.append(feat)
        concat_feats = torch.cat(feats_list, dim=-1)
        
        return self.fusion(concat_feats)
        
        


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
        num_heads: int | None = None,   # <-- accept & ignore
        **kwargs,                       # <-- future-proof
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.num_classes = num_classes
        
        # TODO: Create separate classifier for each modality
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
        self.classifiers = nn.ModuleDict()
        for mod, dim in modality_dims.items():
            self.classifiers[mod] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        
        # TODO: Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        # Option 2: Attention over predictions
        # Option 3: Simple averaging
        self.fusion_weights = nn.Parameter(torch.ones(self.num_modalities) / self.num_modalities)

        
        
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Get predictions from each modality classifier
        #   2. Handle missing modalities (mask out or skip)
        #   3. Combine predictions using fusion weights
        #   4. Return both fused and per-modality predictions
        batch = next(iter(modality_features.values()))
        batch_size, device = batch.size(0), batch.device
        
        logits_list, per_modality_logits = [], {}
        for mod in self.modality_names:
            if mod in modality_features:
                logits = self.classifiers[mod](modality_features[mod])
                per_modality_logits[mod] = logits
            else:
                logits = torch.zeros(batch_size, self.num_classes, device=device)   # <-- fixed
            logits_list.append(logits)
            
        # Combine with weights
        weights = F.softmax(self.fusion_weights, dim=0)
        if modality_mask is not None:
            # zero out weights for missing mods, then renormalize
            w = weights.unsqueeze(0).expand(batch_size, -1) * modality_mask.float()
            weights = (w / (w.sum(dim=1, keepdim=True).clamp_min(1e-6))).mean(dim=0)
        
        fused_logits = sum(w * l for w, l in zip(weights, logits_list))
        return fused_logits, per_modality_logits
        
        


class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.
    
    Pros: Rich cross-modal interaction, robust to missing modalities
    Cons: More complex, higher computation cost
    
    This is the main focus of the assignment!
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Project each modality to common hidden dimension
        # Hint: Use nn.ModuleDict with Linear layers per modality
        self.projections = nn.ModuleDict()
        for mod, dim in modality_dims.items():
            self.projections[mod] = nn.Linear(dim, hidden_dim)
        
        # TODO: Implement cross-modal attention
        # Use CrossModalAttention from attention.py
        # Each modality should attend to all other modalities
        self.cross_attn = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)
        
        # TODO: Learn adaptive fusion weights based on modality availability
        # Hint: Small MLP that takes modality mask and outputs weights
        self.weight_net = nn.Sequential(
            nn.Linear(self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities)
        )
        
        # TODO: Final classifier
        # Takes fused representation -> num_classes logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        
    
    def forward(self, modality_features, modality_mask=None, return_attention=False):
        first = next(iter(modality_features.values()))
        B, device = first.size(0), first.device     # <-- add
        projected = {}
        for mod in self.modality_names:
            if mod in modality_features:
                projected[mod] = self.projections[mod](modality_features[mod])
            else:
                projected[mod] = torch.zeros(B, self.hidden_dim, device=device)  # <-- safe zeros
        stacked = torch.stack([projected[m] for m in self.modality_names], dim=1)   # (B,M,H)
        q = k = v = stacked.mean(dim=1)  # simple summary
        cross_out, cross_w = self.cross_attn(q, k, v, modality_mask)
        # simple uniform weighting (minimal)
        fused = sum(projected[m] for m in self.modality_names) / len(self.modality_names)
        logits = self.classifier(fused)               # <-- DO NOT reduce over batch
        return (logits, {"cross": cross_w}) if return_attention else logits
        
        
    
    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.
        
        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask
            
        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # TODO: Implement adaptive weighting
        # Ideas:
        #   1. Learn weight predictor from modality features + mask
        
        #   2. Higher weights for more reliable/informative modalities
        #   3. Ensure weights sum to 1 (softmax) and respect mask
        batch_size = modality_mask.size(0)
        # Simple: MLP on mask
        mask_flat = modality_mask.float()
        raw_weights = self.weight_net(mask_flat)
        weights = F.softmax(raw_weights, dim=-1)
        return weights
        
        


# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion,
    }
    
    if fusion_type != 'hybrid':
        kwargs.pop('num_heads', None)  # <-- Add 4 spaces (or 1 tab) here for indentation
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )




if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")

