import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TransformerTranslator(nn.Module):
    """Seq2Seq Transformer for ASL gloss to English translation"""
    
    def __init__(self, src_vocab_size: int = 1000, tgt_vocab_size: int = 5000,
                 d_model: int = 512, num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_len, d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask_buffer', None)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(position * div_term)[:-1]
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), 
                         diagonal=1)
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (batch_size, src_len) - ASL gloss token IDs
            tgt: (batch_size, tgt_len) - English word token IDs
            src_mask: optional mask for source
            tgt_mask: optional causal mask for target
        
        Returns:
            logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Get sequence lengths
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]
        device = src.device
        
        # Source embedding + positional encoding
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        src_emb = src_emb + self.positional_encoding[:, :src_len, :].to(device)
        src_emb = self.dropout(src_emb)
        
        # Encoder
        encoder_output = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Target embedding + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = tgt_emb + self.positional_encoding[:, :tgt_len, :].to(device)
        tgt_emb = self.dropout(tgt_emb)
        
        # Create causal mask for target
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(tgt_len, device)
        
        # Decoder
        decoder_output = self.decoder(
            tgt_emb,
            encoder_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence (for inference)"""
        src_len = src.shape[1]
        device = src.device
        
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        src_emb = src_emb + self.positional_encoding[:, :src_len, :].to(device)
        src_emb = self.dropout(src_emb)
        
        encoder_output = self.encoder(src_emb)
        return encoder_output
    
    def decode_step(self, tgt_token: torch.Tensor, encoder_output: torch.Tensor,
                   step: int) -> torch.Tensor:
        """Decode one token at a time (for inference)"""
        device = tgt_token.device
        
        tgt_emb = self.tgt_embedding(tgt_token) * (self.d_model ** 0.5)
        tgt_emb = tgt_emb + self.positional_encoding[:, :1, :].to(device)
        
        # Causal mask for single step
        tgt_mask = None
        
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask=tgt_mask)
        logits = self.output_projection(decoder_output)
        
        return logits


class TranslationLoss(nn.Module):
    """Translation loss (cross-entropy + label smoothing)"""
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, tgt_len, vocab_size)
            targets: (batch_size, tgt_len)
        
        Returns:
            loss: scalar
        """
        # Reshape for cross-entropy loss
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        
        loss = self.loss_fn(logits, targets)
        return loss
