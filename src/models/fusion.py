"""
Cross-modal fusion layer (plan Section 4.3) — pulls the vibration-encoder
embedding and the acoustic-encoder embedding together when they come from
the same real fault event (i.e., a MaFaulDa sample, which alone has both
channels). Small MLP by design, same resource-conscious reasoning as encoder.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.joint = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, vib_emb: torch.Tensor, aud_emb: torch.Tensor) -> torch.Tensor:
        return self.joint(torch.cat([vib_emb, aud_emb], dim=-1))

    @staticmethod
    def paired_contrastive_loss(vib_emb: torch.Tensor, aud_emb: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """InfoNCE-style loss pulling matched (vib, aud) pairs from the same
        MaFaulDa fault event together, pushing mismatched pairs in the batch apart."""
        vib_n = F.normalize(vib_emb, dim=-1)
        aud_n = F.normalize(aud_emb, dim=-1)
        logits = vib_n @ aud_n.T / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)
