"""
Reusable pieces for the VGGish-backbone fine-tuning approach (see
scripts/train/train_vggish_finetune.py for the training entry point and
its full design rationale). Moved here from that script because
scripts/eval/evaluate_anomaly_only.py also needs them, and importing one
executable script from a sibling scripts/ subfolder is fragile — core
reusable classes belong in src/, per this project's own convention.
"""

import os
import sys

import torch
import torch.nn as nn

from .grl import grad_reverse


def _load_vggish_input_module():
    """VGGish's own preprocessing (mel-patch extraction) lives inside the
    torch.hub-cached repo, not as an installable package — load it from
    wherever torch.hub already cached it (triggered by a torch.hub.load
    call, which must run first)."""
    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, "harritaylor_torchvggish_master")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from torchvggish import vggish_input
    return vggish_input


class VGGishLastBlockDANN(nn.Module):
    """VGGish conv blocks 0-2 frozen, block 3 (last, 512-channel) unfrozen,
    VGGish's own FC head discarded in favor of a small new projection —
    see train_vggish_finetune.py's module docstring for why."""

    def __init__(self, vggish_full: nn.Module, n_domains: int, embedding_dim: int = 128):
        super().__init__()
        blocks = list(vggish_full.features.children())
        self.frozen_blocks = nn.Sequential(*blocks[:11])
        for p in self.frozen_blocks.parameters():
            p.requires_grad = False
        self.unfrozen_block = nn.Sequential(*blocks[11:])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, embedding_dim))
        self.fault_head = nn.Linear(embedding_dim, 2)
        self.domain_head = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(), nn.Linear(32, n_domains))

    def forward(self, x: torch.Tensor, grl_lambda: float):
        with torch.no_grad():
            mid = self.frozen_blocks(x)
        feats = self.unfrozen_block(mid)
        emb = self.head(feats)
        fault_logits = self.fault_head(emb)
        domain_logits = self.domain_head(grad_reverse(emb, grl_lambda))
        return fault_logits, domain_logits

    def trainable_parameters(self):
        return list(self.unfrozen_block.parameters()) + list(self.head.parameters()) + \
            list(self.fault_head.parameters()) + list(self.domain_head.parameters())
