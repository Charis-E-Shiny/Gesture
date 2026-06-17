import torch
import torch.nn as nn

from pipeline import ASLRecognitionPipeline


class ASLLiveCaptionPrototype(nn.Module):
    """Small trainable wrapper around the requested ASL architecture."""

    def __init__(self, gloss_vocab_size: int, caption_vocab_size: int,
                 hidden_dim: int = 96, num_heads: int = 6):
        super().__init__()
        self.backbone = ASLRecognitionPipeline(
            vocab_size=gloss_vocab_size,
            hidden_dim=hidden_dim,
            english_vocab_size=max(caption_vocab_size, 4),
            num_heads=num_heads,
            encoder_layers=1,
            fusion_layers=1,
            conformer_layers=2,
        )
        self.gloss_classifier = nn.Linear(hidden_dim, gloss_vocab_size)
        self.caption_classifier = nn.Linear(hidden_dim, caption_vocab_size)

    def forward(self, hand, face, pose):
        encoded = self.backbone.encode_landmarks(hand, face, pose)
        pooled = encoded.mean(dim=1)
        return {
            "encoded": encoded,
            "gloss_logits": self.gloss_classifier(pooled),
            "caption_logits": self.caption_classifier(pooled),
        }

    @torch.no_grad()
    def predict(self, hand, face, pose):
        output = self.forward(hand, face, pose)
        gloss_id = output["gloss_logits"].argmax(dim=-1)
        caption_id = output["caption_logits"].argmax(dim=-1)
        return gloss_id, caption_id
