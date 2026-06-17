import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HandEncoder(nn.Module):
    """Encoder for hand landmarks (21 landmarks -> 2 outputs)"""
    
    def __init__(self, input_dim: int = 42 * 3, output_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 2,
                 feedforward_dim: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Transformer encoder for hand landmarks
        self.embedding = nn.Linear(input_dim, output_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, output_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, 21*3)
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        return x


class FaceEncoder(nn.Module):
    """Encoder for face mesh points (468 points)"""
    
    def __init__(self, input_dim: int = 468 * 3, output_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 2,
                 feedforward_dim: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Linear(input_dim, output_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, output_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, 468*3)
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        return x


class PoseEncoder(nn.Module):
    """Encoder for pose landmarks (33 body landmarks)"""
    
    def __init__(self, input_dim: int = 33 * 3, output_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 2,
                 feedforward_dim: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Linear(input_dim, output_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, output_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, 33*3)
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        return x
