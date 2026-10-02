"""Per-stream temporal encoders mapping a clip ``[B, T, *raw]`` to latents ``[B, T, D]``.

Streams are encoded independently; all cross-stream interaction happens in the predictor.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class _TemporalContext(nn.Module):
    """Depthwise temporal convolution that contextualises each frame within its own stream."""

    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class _Base(nn.Module):
    """Shared output head: project to ``embed_dim`` and optionally mix over time."""

    def __init__(self, in_dim: int, embed_dim: int, temporal_context: bool = True):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.context = _TemporalContext(embed_dim) if temporal_context else nn.Identity()

    def _head(self, feats: torch.Tensor) -> torch.Tensor:
        return self.context(self.proj(feats))


class FeatureEncoder(_Base):
    """Precomputed per-frame features ``[B, T, F]``."""

    def __init__(self, in_dim: int, embed_dim: int, temporal_context: bool = True, **_):
        super().__init__(in_dim, embed_dim, temporal_context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


class MLPEncoder(_Base):
    """Per-frame MLP over a flat feature vector ``[B, T, F]``; leading axes are flattened."""

    def __init__(self, in_dim: int, embed_dim: int, hidden: int = 256,
                 temporal_context: bool = True, **_):
        super().__init__(embed_dim, embed_dim, temporal_context)
        self.body = _mlp([in_dim, hidden, embed_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        return self._head(self.body(x.reshape(b * t, -1)).reshape(b, t, -1))


class ConvEncoder(_Base):
    """Temporal 1D convolution over a channel stream ``[B, T, C]``."""

    def __init__(self, in_channels: int, embed_dim: int, hidden: int = 128, kernel: int = 5,
                 temporal_context: bool = True, **_):
        super().__init__(hidden, embed_dim, temporal_context)
        pad = kernel // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel, padding=pad), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self.tcn(x.transpose(1, 2)).transpose(1, 2))


class TransformerEncoder(_Base):
    """Temporal self-attention within one stream over features ``[B, T, F]``."""

    def __init__(self, in_dim: int, embed_dim: int, depth: int = 2, heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 temporal_context: bool = False, **_):
        super().__init__(in_dim, embed_dim, temporal_context)
        layer = nn.TransformerEncoderLayer(
            embed_dim, heads, int(embed_dim * mlp_ratio), dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(self.encoder(self.proj(x)))


class FrameEncoder(_Base):
    """Per-frame image backbone for frames ``[B, T, C, H, W]``."""

    def __init__(self, embed_dim: int, backbone: str = "resnet18", in_channels: int = 3,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 temporal_context: bool = True, **_):
        import torchvision

        net = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                  stride=old.stride, padding=old.padding, bias=False)
            if pretrained:
                with torch.no_grad():
                    net.conv1.weight.copy_(
                        old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad_(False)
        super().__init__(feat_dim, embed_dim, temporal_context)
        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        return self._head(self.backbone(x.reshape(b * t, *x.shape[2:])).reshape(b, t, -1))


ENCODERS = {
    "feature": FeatureEncoder,
    "mlp": MLPEncoder,
    "conv": ConvEncoder,
    "transformer": TransformerEncoder,
    "frame": FrameEncoder,
}


def build_encoder(spec: dict, embed_dim: int) -> nn.Module:
    """Build one encoder; ``spec['type']`` selects the class and the rest are keyword arguments."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in ENCODERS:
        raise KeyError(f"unknown encoder type '{kind}'; available: {sorted(ENCODERS)}")
    return ENCODERS[kind](embed_dim=embed_dim, **spec)


class StreamEncoders(nn.Module):
    """One encoder per stream, applied to whichever streams a batch carries."""

    def __init__(self, specs: Dict[str, dict], embed_dim: int):
        super().__init__()
        self.streams = list(specs.keys())
        self.embed_dim = embed_dim
        self.encoders = nn.ModuleDict({m: build_encoder(s, embed_dim) for m, s in specs.items()})

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.encoders[m](batch[m]) for m in self.streams if m in batch}
