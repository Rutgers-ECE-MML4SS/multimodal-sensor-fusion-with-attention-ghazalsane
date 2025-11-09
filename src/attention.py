"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

# --- NEW: Self-attention over modalities (sequence length = #modalities) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ModalitySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Transformer encoder-like structure (PreNorm)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Tiny FFN
        ffn_hidden = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    # --- inside attention.py ---

def _mha(self, tokens: torch.Tensor, mask: Optional[torch.Tensor]):
    """
    tokens: (B, M, H)
    mask:   (B, M)  1/0 or bool; True/1 means modality present
    returns: y (B,M,H), attn_w (B, num_heads, M, M)
    """
    B, M, _ = tokens.shape

    # Project to Q,K,V and reshape for multi-head
    q = self.q_proj(tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,M,dh)
    k = self.k_proj(tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,M,dh)
    v = self.v_proj(tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,M,dh)

    # Scaled dot-product attention
    attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B,H,M,M)

    # ---- FIXED MASK HANDLING ----
    if mask is not None:
        # accept float (0/1) or bool; convert to bool
        if mask.dtype is not torch.bool:
            mask = mask > 0.5
        # valid pairs where both query & key present
        valid_pairs = (mask[:, None, :, None] & mask[:, None, None, :])  # (B,1,M,M) -> broadcast over heads
        attn = attn.masked_fill(~valid_pairs, float("-inf"))

    attn_w = torch.softmax(attn, dim=-1)  # (B,H,M,M)
    attn_w = self.dropout(attn_w)

    y = (attn_w @ v)                              # (B,H,M,dh)
    y = y.transpose(1, 2).contiguous().view(B, M, self.hidden_dim)  # (B,M,H)
    y = self.out_proj(y)
    return y, attn_w


    def forward(self, tokens: torch.Tensor, modality_mask: Optional[torch.Tensor] = None):
        """
        tokens: (B, M, H)   (one token per modality)
        modality_mask: (B, M) with 1 for present, 0 for missing
        """
        # Block 1: MHA with PreNorm + residual
        y, attn_w = self._mha(self.ln1(tokens), modality_mask)
        tokens = tokens + self.dropout(y)

        # Block 2: FFN with PreNorm + residual
        y = self.ffn(self.ln2(tokens))
        tokens = tokens + y

        return tokens, attn_w  # tokens: (B,M,H), attn_w: (B,H,M,M)




class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.
    
    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # TODO: Implement multi-head attention projections
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim,   hidden_dim)
        self.v_proj = nn.Linear(key_dim,   hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Hint: Use nn.Linear for Q, K, V projections
        # Query from modality A, Key and Value from modality B
    @staticmethod
    def _to_3d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        # (B,D) -> (B,1,D), flag=True ; (B,L,D) -> as-is, flag=False
        if x.dim() == 2:
            return x.unsqueeze(1), True
        return x, False

    def _normalize_mask(
        self, mask: Optional[torch.Tensor], B: int, Lk: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Returns mask shaped (B,1,1,Lk) with True=keep, False=mask.
        Accepts:
          - None
          - (B,)                -> broadcast to Lk
          - (B, Lk)
          - (B, 1, 1, Lk)
        """
        if mask is None:
            return None
        m = mask
        if m.dtype != torch.bool:
            m = m.bool()
        if m.dim() == 1:
            m = m.view(B, 1, 1, 1).expand(B, 1, 1, Lk)
        elif m.dim() == 2:
            # (B, Lk) -> (B,1,1,Lk)
            assert m.size(1) == Lk, f"mask Lk={m.size(1)} != key length {Lk}"
            m = m.view(B, 1, 1, Lk)
        elif m.dim() == 4:
            # assume already (B,1,1,Lk)
            assert m.size(0) == B and m.size(-1) == Lk, "mask shape mismatch"
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")
        return m.to(device)
        
    
    def forward(
        self,
        query: torch.Tensor,   # (B,Dq) or (B,Lq,Dq)
        key:   torch.Tensor,   # (B,Dk) or (B,Lk,Dk)
        value: torch.Tensor,   # (B,Dk) or (B,Lv,Dk), expect Lk == Lv
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = query.size(0)
        device = query.device

        # Ensure 3D
        query, _ = self._to_3d(query)   # (B, Lq, Dq)
        key,   _ = self._to_3d(key)     # (B, Lk, Dk)
        value, _ = self._to_3d(value)   # (B, Lv, Dk)
        Lq, Lk, Lv = query.size(1), key.size(1), value.size(1)
        assert Lk == Lv, f"K length (Lk={Lk}) must equal V length (Lv={Lv})"

        # Projections -> (B, H, L*, dh)
        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key  ).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Lv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scores (B, H, Lq, Lk)
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        # Normalize/attach mask (True=keep). If your convention is True=mask, invert here.
        attn_mask = self._normalize_mask(mask, B, Lk, device)
        if attn_mask is not None:
            # attn_mask True=keep -> convert to additive mask
            # we mask where keep==False
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        # Weights & dropout
        weights = torch.softmax(scores, dim=-1)  # (B,H,Lq,Lk)
        if torch.isnan(weights).any():
            weights = torch.zeros_like(weights)
            weights[..., 0] = 1.0
        weights = self.dropout(weights)

        # Output (B,H,Lq,dh) -> (B,Lq,H*dh) -> proj -> (B,Lq,H*dh)
        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.hidden_dim)
        out = self.out_proj(out)

        # Return (B, hidden_dim) if Lq==1 to match your original API
        if Lq == 1:
            return out.squeeze(1), weights
        return out, weights


class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # TODO: Implement self-attention over temporal dimension
        self.qkv_proj = nn.Linear(feature_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Hint: Similar to CrossModalAttention but Q, K, V from same modality
        
        
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # TODO: Implement temporal self-attention
        # Steps:
        #   1. Project sequence to Q, K, V
        B, T, D = sequence.shape
        qkv = self.qkv_proj(sequence).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # (B, H, T, head_dim)
        
        #   2. Compute self-attention over sequence length
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        
        #   3. Apply mask for variable-length sequences
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(-1) == 0, float('-inf'))
        #   4. Return attended sequence and weights
        attn_weights = F.softmax(attn, dim=-1)
        # Guard for all -inf (point 5)
        if attn_weights.isnan().any():
            attn_weights = torch.zeros_like(attn_weights)
            attn_weights[:, :, :, 0] = 1.0  # Fallback to first
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ V).transpose(1, 2).reshape(B, T, self.hidden_dim)  # (B, T, hidden_dim)
        out = self.out_proj(out)
        return out, attn_weights
        
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        # TODO: Implement attention-based pooling
        
        # Option 1: Weighted average using mean attention weights
        weights = attention_weights.mean(dim=1).mean(dim=-1)  # (B, seq_len)
        weights = F.softmax(weights, dim=-1)
        pooled = (weights.unsqueeze(-1) * sequence).sum(dim=1)  # (B, hidden_dim)
        return pooled
        # Option 2: Learn pooling query vector
        # Option 3: Take output at special [CLS] token position
        
        


class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Create CrossModalAttention for each modality pair
        
        # Hint: Use nn.ModuleDict with keys like "video_to_audio"
        # For each pair (A, B), create attention A->B and B->A
        self.pairwise_attns = nn.ModuleDict()
        for i, mod_a in enumerate(self.modality_names):
            for j, mod_b in enumerate(self.modality_names):
                if i != j:
                    key = f"{mod_a}_to_{mod_b}"
                    self.pairwise_attns[key] = CrossModalAttention(
                        query_dim=modality_dims[mod_a],
                        key_dim=modality_dims[mod_b],
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout)
        
    
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        # TODO: Implement pairwise attention
        # Steps:
        #   1. For each modality pair (A, B):
        #      - Apply attention A->B (A attends to B)
        #      - Apply attention B->A (B attends to A)
        #   2. Aggregate attended features (options: sum, concat, gating)
        #   3. Handle missing modalities using mask
        #   4. Return attended features and attention maps for visualization
        attended = {mod: torch.zeros_like(modality_features[mod]) for mod in modality_features}
        attention_maps = {}
        for i, mod_a in enumerate(self.modality_names):
            for j, mod_b in enumerate(self.modality_names):
                if i != j:
                    key = f"{mod_a}_to_{mod_b}"
                    if mod_a in modality_features and mod_b in modality_features:
                        attended_a, weights = self.pairwise_attns[key](
                            query=modality_features[mod_a],
                            key=modality_features[mod_b],
                            value=modality_features[mod_b],
                            mask=modality_mask[:, j] if modality_mask is not None else None
                        )
                        # Project to hidden_dim for accumulation (point 8)
                        attended_a_proj = self.proj[mod_a](modality_features[mod_a]) if attended_a.size(-1) != self.hidden_dim else attended_a
                        attended[mod_a] += attended_a_proj
                        attention_maps[key] = weights
        return attended, attention_maps
                        
        
        


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Implement attention visualization
    # Create heatmap showing which modalities attend to which
    # Useful for understanding fusion behavior
    weights = attention_weights.mean(dim=0).detach().cpu().numpy()  # Avg heads
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='Blues', aspect='auto')
    plt.xlabel('Key Modalities')
    plt.ylabel('Query Modalities')
    plt.xticks(range(len(modality_names)), modality_names, rotation=45)
    plt.yticks(range(len(modality_names)), modality_names)
    plt.title('Cross-Modal Attention Heatmap')
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")

