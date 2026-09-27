"""
Small CNN spectrogram encoder + domain-adversary head.

Deliberately shallow/narrow (4 conv layers, channels capped at 128) rather
than a deep ResNet — this is a 48-hour-hackathon-scale model meant to run
comfortably on a laptop CPU or a modest GPU, not a research-scale network.
If accuracy demands more capacity later, widen channels first, add depth
only if that's not enough — don't jump straight to a huge backbone.
"""

import torch
import torch.nn as nn

from .grl import grad_reverse


class SpectrogramEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.conv(spec)
        x = x.flatten(1)
        return self.proj(x)


class DomainAdversarialEncoder(nn.Module):
    """Wraps SpectrogramEncoder with a 'which dataset did this come from'
    classifier fed through the gradient-reversal layer, per plan Section 4.2."""

    def __init__(self, embedding_dim: int = 256, n_source_datasets: int = 6, grl_lambda: float = 1.0):
        super().__init__()
        self.encoder = SpectrogramEncoder(embedding_dim)
        self.domain_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, n_source_datasets)
        )
        self.grl_lambda = grl_lambda

    def forward(self, spec: torch.Tensor):
        emb = self.encoder(spec)
        domain_logits = self.domain_head(grad_reverse(emb, self.grl_lambda))
        return emb, domain_logits


from pathlib import Path
import torch
def load_encoder(ckpt_path: Path, embedding_dim: int = 128) -> SpectrogramEncoder:
    encoder = SpectrogramEncoder(embedding_dim)
    raw = torch.load(ckpt_path, map_location='cpu')
    state = raw['model'] if 'model' in raw else raw
    encoder_state = {k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder
