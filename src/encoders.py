"""
Modality-Specific Encoders for Sensor Fusion

Implements lightweight encoders suitable for CPU training:
1. SequenceEncoder: For time-series data (IMU, audio, motion capture)
2. FrameEncoder: For frame-based data (video features)
3. SimpleMLPEncoder: For pre-extracted features
"""

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F



class SequenceEncoder(nn.Module):
    """
    Encoder for sequential/time-series sensor data.
    
    Options: 1D CNN, LSTM, GRU, or Transformer
    Output: Fixed-size embedding per sequence
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = 'lstm',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for RNN/Transformer
            output_dim: Output embedding dimension
            num_layers: Number of encoder layers
            encoder_type: One of ['lstm', 'gru', 'cnn', 'transformer']
            dropout: Dropout probability
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # TODO: Implement sequence encoder
        # Choose ONE of the following architectures:
        
        if encoder_type == 'lstm':
            # TODO: Implement LSTM encoder
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                               batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=False) 
            self.projection = nn.Linear(hidden_dim, output_dim)
            
            
        elif encoder_type == 'gru':
            # TODO: Implement GRU encoder
            self.rnn=nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=False)
            self.projection = nn.Linear(hidden_dim, output_dim)
            
        elif encoder_type == 'cnn':
            # TODO: Implement 1D CNN encoder
            # Stack of Conv1d -> BatchNorm -> ReLU -> Pool
            # Example:
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.projection = nn.Linear(hidden_dim, output_dim)
            
        elif encoder_type == 'transformer':
            # TODO: Implement Transformer encoder
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.projection = nn.Linear(hidden_dim, output_dim)
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        
    
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode variable-length sequences.
        
        Args:
            sequence: (batch_size, seq_len, input_dim) - input sequence
            lengths: Optional (batch_size,) - actual sequence lengths (for padding)
            
        Returns:
            encoding: (batch_size, output_dim) - fixed-size embedding
        """
        # TODO: Implement forward pass based on encoder_type
        # Handle variable-length sequences if lengths provided
        # Return fixed-size embedding via pooling or taking last hidden state
        if self.encoder_type in ['lstm', 'gru']:
            out, h = self.rnn(sequence)
            if isinstance(h, tuple):  # LSTM
                h_n = h[0]
            else:  # GRU
                h_n = h
            last_hidden = h_n[-1]  # (batch_size, hidden_dim)
            return self.projection(last_hidden)
        elif self.encoder_type == 'cnn':
            #CNN: transpose to (B, D, T), conv, pool
            x = sequence.transpose(1, 2)  # (B, input_dim, seq_len)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            pooled = self.pool(x).squeeze(-1)  # (B, hidden_dim)
            return self.projection(pooled)
        elif self.encoder_type == 'transformer':
            
            x = self.in_proj(sequence)
            out = self.transformer(x)
            pooled = out.mean(dim=1)  # (B, hidden_dim)
            return self.projection(pooled)
            
        
        


class FrameEncoder(nn.Module):
    """
    Encoder for frame-based data (e.g., video features).
    
    Aggregates frame-level features into video-level embedding.
    """
    
    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temporal_pooling: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Args:
            frame_dim: Dimension of per-frame features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temporal_pooling: How to pool frames ['average', 'max', 'attention']
            dropout: Dropout probability
        """
        super().__init__()
        self.temporal_pooling = temporal_pooling
        
        # TODO: Implement frame encoder
        # 1. Frame-level processing (optional MLP)
        # 2. Temporal aggregation (pooling or attention)
        self.frame_mlp = nn.Sequential(nn.Linear(frame_dim, hidden_dim),nn.ReLU(),nn.Dropout(dropout))
        
        if temporal_pooling == 'attention':
            # TODO: Implement attention-based pooling
            # Learn which frames are important
            self.attention = nn.Linear(hidden_dim, 1)
        elif temporal_pooling in ['average', 'max']:
            # Simple pooling, no learnable parameters needed
            pass
        else:
            raise ValueError(f"Unknown pooling: {temporal_pooling}")
        
        # TODO: Add projection layer
        self.projection = nn.Linear(hidden_dim, output_dim)
        
        
    
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of frames.
        
        Args:
            frames: (batch_size, num_frames, frame_dim) - frame features
            mask: Optional (batch_size, num_frames) - valid frame mask
            
        Returns:
            encoding: (batch_size, output_dim) - video-level embedding
        """
        # TODO: Implement forward pass
        # 1. Process frames (optional)
        h = self.frame_mlp(frames)
        # 2. Apply temporal pooling
        if self.temporal_pooling == 'attention':
            pooled = self.attention_pool(h, mask)
        elif self.temporal_pooling == 'average':
            if mask is None:
                pooled = h.mean(dim=1)
            else:
                m = mask.float().unsqueeze(-1)
                pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        elif self.temporal_pooling == 'max':
            if mask is not None:
                h = h.masked_fill(~mask.unsqueeze(-1).bool(), float('-inf'))
            pooled = h.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.temporal_pooling}")
        # 3. Project to output dimension
        return self.projection(pooled)
        
        
    
    def attention_pool(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool frames using learned attention weights.
        
        Args:
            frames: (batch_size, num_frames, frame_dim)
            mask: Optional (batch_size, num_frames) - valid frames
            
        Returns:
            pooled: (batch_size, frame_dim) - attended frame features
        """
        # TODO: Implement attention pooling
        
        # 1. Compute attention scores for each frame
        scores = self.attention(frames).squeeze(-1)  # (B, num_frames)
        # 2. Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        # 3. Softmax to get weights
        weights = F.softmax(scores, dim=1)  # (B, num_frames)
        # 4. Weighted sum of frames
        pooled = (weights.unsqueeze(-1) * frames).sum(dim=1)  # (B, frame_dim)
        return pooled
        
        


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for pre-extracted features.
    
    Use this when working with pre-computed features
    (e.g., ResNet features for images, MFCC for audio).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # TODO: Implement MLP encoder
        # Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output
        
        layers = []
        current_dim = input_dim
        
        # TODO: Add hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        
        
        # TODO: Add output layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        
        
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through MLP.
        
        Args:
            features: (batch_size, input_dim) - input features
            
        Returns:
            encoding: (batch_size, output_dim) - encoded features
        """
        # TODO: Implement forward pass
        # If 3D (B, T, D), average over T
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.encoder(features)
        
        


def build_encoder(
    modality: str,
    input_dim: int,
    output_dim: int,
    encoder_config: dict = None
) -> nn.Module:
    if encoder_config is None:
        encoder_config = {}

    # Drop problematic keys so we don't pass duplicates
    safe_cfg = {k: v for k, v in encoder_config.items()
                if k not in ['input_dim', 'output_dim', 'type']}
    print(f"DEBUG build_encoder: modality='{modality}', input_dim={input_dim}, encoder_config={encoder_config}, safe_cfg keys={list(safe_cfg.keys())}")

    if modality in ['video', 'frames']:
        print(f"DEBUG: Using FrameEncoder for {modality}")
        return FrameEncoder(
            frame_dim=input_dim,
            output_dim=output_dim,
            **safe_cfg
        )

    # Treat PAMAP2 streams as sequences
    if modality in ['imu', 'audio', 'mocap', 'accelerometer',
                    'imu_hand', 'imu_chest', 'imu_ankle', 'heart_rate']:
        enc_type = safe_cfg.pop('encoder_type', 'lstm')
        print(f"DEBUG: Using SequenceEncoder for {modality} with enc_type={enc_type}")
        return SequenceEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_type=enc_type,
            **safe_cfg
        )

    # Fallback
    
    print(f"DEBUG: FALLBACK to SimpleMLPEncoder for {modality} (this is unexpected!)")
    print(f"DEBUG: safe_cfg before ** = {safe_cfg}")  # This will show if 'input_dim' sneaked in
    return SimpleMLPEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        **safe_cfg
    )



if __name__ == '__main__':
    # Test encoders
    print("Testing encoders...")
    
    batch_size = 4
    seq_len = 100
    input_dim = 64
    output_dim = 128
    
    # Test SequenceEncoder
    print("\nTesting SequenceEncoder...")
    for enc_type in ['lstm', 'gru', 'cnn']:
        try:
            encoder = SequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_type=enc_type
            )
            
            sequence = torch.randn(batch_size, seq_len, input_dim)
            output = encoder(sequence)
            
            assert output.shape == (batch_size, output_dim)
            print(f"✓ {enc_type} encoder working! Output shape: {output.shape}")
            
        except NotImplementedError:
            print(f"✗ {enc_type} encoder not implemented yet")
        except Exception as e:
            print(f"✗ {enc_type} encoder error: {e}")
    
    # Test FrameEncoder
    print("\nTesting FrameEncoder...")
    try:
        num_frames = 30
        frame_dim = 512
        
        encoder = FrameEncoder(
            frame_dim=frame_dim,
            output_dim=output_dim,
            temporal_pooling='attention'
        )
        
        frames = torch.randn(batch_size, num_frames, frame_dim)
        output = encoder(frames)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ FrameEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ FrameEncoder not implemented yet")
    except Exception as e:
        print(f"✗ FrameEncoder error: {e}")
    
    # Test SimpleMLPEncoder
    print("\nTesting SimpleMLPEncoder...")
    try:
        encoder = SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        features = torch.randn(batch_size, input_dim)
        output = encoder(features)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ SimpleMLPEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ SimpleMLPEncoder not implemented yet")
    except Exception as e:
        print(f"✗ SimpleMLPEncoder error: {e}")

