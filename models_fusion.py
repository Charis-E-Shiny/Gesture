import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention module for multi-modal fusion"""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, kv_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.kv_dim = kv_dim or d_model  # Allow different input dim for key/value
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(self.kv_dim, d_model)  # Project kv_dim to d_model
        self.value = nn.Linear(self.kv_dim, d_model)  # Project kv_dim to d_model
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(context)
        
        return output


class CrossModalFusionTransformer(nn.Module):
    """Cross-modal fusion with concat and cross-attention (8 heads)"""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Concat the three modalities: hand, face, pose -> 3*d_model
        self.fusion_proj = nn.Linear(3 * d_model, d_model)
        
        # Cross-attention layers between modalities
        # When we concatenate two modalities, input is 2*d_model
        self.cross_attentions = nn.ModuleList([
            MultiHeadCrossAttention(d_model, num_heads, kv_dim=2*d_model) for _ in range(num_layers)
        ])
        
        # Self-attention for fused representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.1
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, hand: torch.Tensor, face: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand: (batch_size, seq_len, d_model)
            face: (batch_size, seq_len, d_model)
            pose: (batch_size, seq_len, d_model)
        
        Returns:
            fused: (batch_size, seq_len, d_model)
        """
        # Concatenate all modalities
        concat = torch.cat([hand, face, pose], dim=-1)
        fused = self.fusion_proj(concat)
        
        # Cross-attention between modalities
        for cross_attn in self.cross_attentions:
            # Apply cross-attention with each modality as query
            hand_update = cross_attn(hand, torch.cat([face, pose], dim=-1), 
                                    torch.cat([face, pose], dim=-1))
            face_update = cross_attn(face, torch.cat([hand, pose], dim=-1), 
                                    torch.cat([hand, pose], dim=-1))
            pose_update = cross_attn(pose, torch.cat([hand, face], dim=-1), 
                                    torch.cat([hand, face], dim=-1))
            
            # Update representations
            hand = hand + 0.1 * hand_update
            face = face + 0.1 * face_update
            pose = pose + 0.1 * pose_update
        
        # Final concatenation and projection
        concat = torch.cat([hand, face, pose], dim=-1)
        fused = self.fusion_proj(concat)
        
        # Self-attention on fused representation
        fused = self.self_attention(fused)
        fused = self.norm(fused)
        
        return fused
