import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict

from models_encoders import HandEncoder, FaceEncoder, PoseEncoder
from models_fusion import CrossModalFusionTransformer
from models_conformer import ConformerStreamingEncoder
from models_rnnt import RNNTTransducer, RNNTLoss
from models_translator import TransformerTranslator, TranslationLoss


class ASLRecognitionPipeline(nn.Module):
    """Complete ASL recognition pipeline"""
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 512,
                 english_vocab_size: int = 5000):
        super().__init__()
        self.vocab_size = vocab_size
        self.english_vocab_size = english_vocab_size
        self.hidden_dim = hidden_dim
        
        # Landmark dimensions
        hand_dim = 126  # 21 landmarks × 3 × 2 hands
        face_dim = 1404  # 468 landmarks × 3
        pose_dim = 99   # 33 landmarks × 3
        
        # Encoders for individual modalities
        self.hand_encoder = HandEncoder(input_dim=hand_dim, output_dim=hidden_dim)
        self.face_encoder = FaceEncoder(input_dim=face_dim, output_dim=hidden_dim)
        self.pose_encoder = PoseEncoder(input_dim=pose_dim, output_dim=hidden_dim)
        
        # Cross-modal fusion
        self.fusion = CrossModalFusionTransformer(d_model=hidden_dim, num_heads=8)
        
        # Streaming encoder
        self.conformer = ConformerStreamingEncoder(d_model=hidden_dim, num_heads=8)
        
        # RNNT for gloss recognition
        self.rnnt = RNNTTransducer(input_dim=hidden_dim, vocab_size=vocab_size)
        
        # Translator for ASL gloss to English
        self.translator = TransformerTranslator(
            src_vocab_size=vocab_size,
            tgt_vocab_size=english_vocab_size,
            d_model=hidden_dim
        )
        
        # Losses
        self.rnnt_loss = RNNTLoss(blank_idx=0)
        self.translation_loss = TranslationLoss(english_vocab_size)
    
    def forward(self, hand: torch.Tensor, face: torch.Tensor, pose: torch.Tensor,
                gloss_targets: Optional[torch.Tensor] = None,
                english_targets: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hand: (batch_size, seq_len, 126)
            face: (batch_size, seq_len, 1404)
            pose: (batch_size, seq_len, 99)
            gloss_targets: (batch_size, target_len) - for training
            english_targets: (batch_size, english_len) - for training
        
        Returns:
            output: dict with 'gloss_logits', 'english_logits', 'losses'
        """
        # Encode individual modalities
        hand_enc = self.hand_encoder(hand)
        face_enc = self.face_encoder(face)
        pose_enc = self.pose_encoder(pose)
        
        # Fuse modalities
        fused = self.fusion(hand_enc, face_enc, pose_enc)
        
        # Streaming conformer encoder
        conformer_out = self.conformer(fused, streaming=True)
        
        # RNNT for gloss recognition
        if gloss_targets is not None:
            gloss_logits, _ = self.rnnt(conformer_out, gloss_targets)
        else:
            # Inference: use empty gloss as input
            batch_size = conformer_out.shape[0]
            device = conformer_out.device
            empty_gloss = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            gloss_logits, _ = self.rnnt(conformer_out, empty_gloss)
        
        output = {
            'gloss_logits': gloss_logits,  # (batch_size, seq_len, vocab_size)
        }
        
        # Translation to English (if available)
        if english_targets is not None:
            # Predict gloss first
            gloss_pred = torch.argmax(gloss_logits, dim=-1)
            
            # Translate to English
            english_logits = self.translator(gloss_pred, english_targets)
            output['english_logits'] = english_logits
        
        # Compute losses if targets provided
        if gloss_targets is not None:
            batch_size = hand.shape[0]
            seq_len = gloss_logits.shape[1]
            
            # Prepare lengths for RNNT loss
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            target_lengths = torch.full((batch_size,), gloss_targets.shape[1], 
                                       dtype=torch.long)
            
            rnnt_loss = self.rnnt_loss(gloss_logits, gloss_targets, 
                                      input_lengths, target_lengths)
            output['rnnt_loss'] = rnnt_loss
        
        if english_targets is not None:
            trans_loss = self.translation_loss(output['english_logits'], 
                                              english_targets)
            output['translation_loss'] = trans_loss
        
        return output
    
    def recognize_gloss(self, hand: torch.Tensor, face: torch.Tensor, 
                       pose: torch.Tensor) -> torch.Tensor:
        """
        Recognize ASL gloss from landmarks
        
        Returns:
            gloss_ids: (batch_size, seq_len)
        """
        with torch.no_grad():
            output = self.forward(hand, face, pose)
            gloss_logits = output['gloss_logits']
            gloss_ids = torch.argmax(gloss_logits, dim=-1)
        
        return gloss_ids
    
    def translate_gloss(self, gloss_ids: torch.Tensor) -> torch.Tensor:
        """
        Translate ASL gloss to English
        
        Args:
            gloss_ids: (batch_size, seq_len)
        
        Returns:
            english_ids: (batch_size, english_len)
        """
        with torch.no_grad():
            # Encode gloss
            encoder_output = self.translator.encode(gloss_ids)
            
            # Greedy decoding
            batch_size = gloss_ids.shape[0]
            device = gloss_ids.device
            
            # Start token
            start_token = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
            english_seq = start_token.clone()
            
            # Generate tokens one by one
            for _ in range(100):  # Max length
                decoder_output = self.translator.decode_step(english_seq[:, -1:], 
                                                            encoder_output, 
                                                            english_seq.shape[1])
                next_token = torch.argmax(decoder_output, dim=-1)
                english_seq = torch.cat([english_seq, next_token], dim=1)
                
                # Stop if end token is generated
                if (next_token.squeeze() == 3).all():
                    break
        
        return english_seq
    
    def get_total_loss(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine all losses"""
        total_loss = 0.0
        
        if 'rnnt_loss' in output:
            total_loss = total_loss + output['rnnt_loss']
        
        if 'translation_loss' in output:
            total_loss = total_loss + output['translation_loss']
        
        return total_loss
