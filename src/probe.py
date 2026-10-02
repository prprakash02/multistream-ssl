"""Frozen-encoder read-out heads used for downstream evaluation."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

POOL_STATS = {"mean": 1, "max": 1, "meanmax": 2, "meanmaxstd": 3}


def pool_features(z: Dict[str, torch.Tensor], streams: List[str],
                  how: str = "mean") -> torch.Tensor:
    """Concatenate pooled per-stream latents into ``[B, S * M * D]``; absent streams give zeros."""
    if how not in POOL_STATS:
        raise ValueError(f"unknown pooling '{how}'; expected one of {sorted(POOL_STATS)}")
    ref = next(iter(z.values()))
    B, _, D = ref.shape

    def _pool(v: torch.Tensor) -> torch.Tensor:
        stats = []
        if how in ("mean", "meanmax", "meanmaxstd"):
            stats.append(v.mean(dim=1))
        if how in ("max", "meanmax", "meanmaxstd"):
            stats.append(v.max(dim=1).values)
        if how == "meanmaxstd":
            stats.append(v.std(dim=1, unbiased=False))
        return torch.cat(stats, dim=-1)

    blocks = [_pool(z[m]) if m in z
              else torch.zeros(B, POOL_STATS[how] * D, device=ref.device, dtype=ref.dtype)
              for m in streams]
    return torch.cat(blocks, dim=-1)


class LinearProbe(nn.Module):
    """Batch-normalised linear classifier on pooled frozen features."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(feats))


class AttentiveProbe(nn.Module):
    """Learned per-stream temporal attention pool followed by a linear classifier."""

    def __init__(self, embed_dim: int, num_streams: int, num_classes: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_streams, embed_dim) * 0.02)
        self.bn = nn.BatchNorm1d(num_streams * embed_dim, affine=False)
        self.fc = nn.Linear(num_streams * embed_dim, num_classes)

    def forward(self, z: Dict[str, torch.Tensor], streams: List[str]) -> torch.Tensor:
        ref = next(iter(z.values()))
        B, _, D = ref.shape
        pooled = []
        for k, m in enumerate(streams):
            if m in z:
                weights = torch.softmax((z[m] * self.query[k].view(1, 1, D)).sum(-1), dim=1)
                pooled.append((weights.unsqueeze(-1) * z[m]).sum(1))
            else:
                pooled.append(torch.zeros(B, D, device=ref.device, dtype=ref.dtype))
        return self.fc(self.bn(torch.cat(pooled, dim=-1)))
