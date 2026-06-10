import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    """Conformer block: Conv + self-attention + causal masking"""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, kernel_size: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Feedforward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, 
                                        batch_first=True)
        self.norm_mha = nn.LayerNorm(d_model)
        self.dropout_mha = nn.Dropout(dropout)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, 2 * d_model, kernel_size, 
                     padding=(kernel_size - 1) // 2, groups=1),
            nn.GLU(dim=1),  # Gated Linear Unit
            nn.DepthwiseConv1d(d_model, d_model, kernel_size, 
                              padding=(kernel_size - 1) // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # Feedforward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: causal mask for streaming
        
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        # Residual feedforward 1
        res = x
        x = self.ff1(x)
        x = 0.5 * x + res
        
        # Residual multi-head attention (causal)
        res = x
        x_norm = self.norm_mha(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=mask)
        x = res + self.dropout_mha(attn_out)
        
        # Residual convolution (1D)
        res = x
        x_conv = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch_size, seq_len, d_model)
        x = res + x_conv
        
        # Residual feedforward 2
        res = x
        x = self.ff2(x)
        x = 0.5 * x + res
        
        x = self.final_norm(x)
        return x


class ConformerStreamingEncoder(nn.Module):
    """Conformer streaming encoder with causal masking"""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 4, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Linear(d_model, d_model)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            streaming: if True, apply causal masking
        
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        
        # Create causal mask if streaming
        if streaming:
            seq_len = x.shape[1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, 
                                               device=x.device, dtype=torch.bool), 
                                    diagonal=1)
        else:
            causal_mask = None
        
        for block in self.conformer_blocks:
            x = block(x, mask=causal_mask)
        
        x = self.norm(x)
        return x


class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution"""
    def __init__(self, channels: int, kernel_size: int, padding: int = 0, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                             stride=stride, groups=channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
