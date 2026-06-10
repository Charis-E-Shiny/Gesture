import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RNNTTransducer(nn.Module):
    """RNNT Transducer for ASL gloss recognition"""
    
    def __init__(self, input_dim: int = 512, vocab_size: int = 1000, 
                 hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Prediction network: LSTM that takes previous gloss tokens
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.prediction_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                       batch_first=True, dropout=0.1)
        
        # Joint network: combines encoder output and prediction
        self.joint_fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.joint_activation = nn.Tanh()
        self.joint_fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_output: torch.Tensor, gloss_tokens: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            encoder_output: (batch_size, seq_len, input_dim)
            gloss_tokens: (batch_size, target_len) - previous gloss tokens
            hidden_state: LSTM hidden state tuple
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden_state: updated LSTM hidden state
        """
        batch_size = encoder_output.shape[0]
        
        # Embedding gloss tokens
        embedded_tokens = self.embedding(gloss_tokens)
        
        # Prediction network forward pass
        pred_output, hidden_state = self.prediction_lstm(embedded_tokens, hidden_state)
        
        # Expand dimensions for joint computation
        # encoder_output: (batch_size, seq_len, input_dim)
        # pred_output: (batch_size, target_len, hidden_dim)
        
        # For simplicity, take the last prediction output and expand to match encoder seq_len
        last_pred = pred_output[:, -1:, :].expand(-1, encoder_output.shape[1], -1)
        
        # Joint network
        joint_input = torch.cat([encoder_output, last_pred], dim=-1)
        joint = self.joint_fc1(joint_input)
        joint = self.joint_activation(joint)
        logits = self.joint_fc2(joint)
        
        return logits, hidden_state
    
    def initialize_hidden_state(self, batch_size: int, device: torch.device
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state"""
        h = torch.zeros(self.prediction_lstm.num_layers, batch_size, self.hidden_dim,
                       device=device)
        c = torch.zeros(self.prediction_lstm.num_layers, batch_size, self.hidden_dim,
                       device=device)
        return h, c


class RNNTLoss(nn.Module):
    """RNNT Loss (simplified version using CTC-style loss)"""
    
    def __init__(self, blank_idx: int = 0):
        super().__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, target_len)
            input_lengths: (batch_size,)
            target_lengths: (batch_size,)
        
        Returns:
            loss: scalar
        """
        # Reshape logits for CTC loss
        logits = logits.transpose(0, 1)  # (seq_len, batch_size, vocab_size)
        log_probs = F.log_softmax(logits, dim=2)
        
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss
