"""Per-edge distributions over integer offsets and the soft alignment they induce.
For each ordered pair the score of a candidate is a time-averaged inner product between
projected latents, normalised onto the simplex over the candidate set.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Simplex projection that leaves exact zeros outside the selected candidates."""
    z = logits.transpose(dim, -1)
    z = z - z.max(dim=-1, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    k = z.shape[-1]
    rng = torch.arange(1, k + 1, device=z.device, dtype=z.dtype)
    cssv = z_sorted.cumsum(dim=-1) - 1.0
    support = (z_sorted - cssv / rng) > 0
    size = support.to(z.dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
    tau = cssv.gather(-1, size.long() - 1) / size
    return torch.clamp(z - tau, min=0.0).transpose(dim, -1)


def shift(z: torch.Tensor, offset: int, pad: str = "replicate") -> torch.Tensor:
    """Return ``z(t + offset)`` along time for ``z`` of shape ``[B, T, D]``."""
    T = z.shape[1]
    idx = torch.arange(T, device=z.device) + offset
    if pad == "replicate":
        return z.index_select(1, idx.clamp(0, T - 1))
    if pad == "zero":
        valid = ((idx >= 0) & (idx < T)).view(1, T, 1).to(z.dtype)
        return z.index_select(1, idx.clamp(0, T - 1)) * valid
    raise ValueError(f"unknown pad mode '{pad}'")


def soft_align(z_src: torch.Tensor, pi: torch.Tensor, candidates: torch.Tensor,
               pad: str = "replicate") -> torch.Tensor:
    """Marginalise the source trajectory over its offset distribution: ``sum_l pi(l) z(t + l)``."""
    out = torch.zeros_like(z_src)
    for k in range(candidates.shape[0]):
        out = out + pi[:, k].view(-1, 1, 1) * shift(z_src, int(candidates[k]), pad=pad)
    return out


class OffsetDistribution(nn.Module):
    """Correlation scores over the candidate offsets of every ordered pair, plus a per-edge bias."""

    def __init__(self, num_streams: int, max_offset: int = 4, embed_dim: int = 256,
                 proj_dim: int = 64, temperature: float = 0.1, readout: str = "sparsemax",
                 topk: int = 0, pad: str = "replicate") -> None:
        super().__init__()
        if readout not in {"sparsemax", "softmax"}:
            raise ValueError(f"unknown readout '{readout}'")
        self.M = num_streams
        self.L = int(max_offset)
        self.K = 2 * self.L + 1
        self.readout = readout
        self.pad = pad
        self.topk = int(topk)
        self.register_buffer("candidates", torch.arange(-self.L, self.L + 1).long())
        self.proj = nn.ModuleList([nn.Linear(embed_dim, proj_dim) for _ in range(num_streams)])
        self.log_temperature = nn.Parameter(torch.tensor(math.log(max(temperature, 1e-3))))
        # Zero at initialisation, so the response to an input shift is exact before training.
        self.bias = nn.Parameter(torch.zeros(num_streams, num_streams, self.K))

    def scores(self, latents: torch.Tensor) -> torch.Tensor:
        """Time-averaged inner products ``[B, M, M, K]`` between projected source and target."""
        B, M, T, _ = latents.shape
        phi = torch.stack(
            [F.normalize(self.proj[m](latents[:, m]), dim=-1) for m in range(M)], dim=1)
        flat = phi.reshape(B * M, T, -1)
        out = latents.new_zeros(B, M, M, self.K)
        for k in range(self.K):
            shifted = shift(flat, int(self.candidates[k]), pad=self.pad).reshape(B, M, T, -1)
            out[:, :, :, k] = torch.einsum("bstd,bytd->bsy", shifted, phi) / T
        return out

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return ``pi`` of shape ``[B, M, M, K]``; computed in fp32 for numerical stability."""
        device_type = latents.device.type
        latents = latents.float()
        if device_type in ("cuda", "cpu", "mps"):
            try:
                with torch.autocast(device_type=device_type, enabled=False):
                    return self._forward(latents)
            except (RuntimeError, ValueError):
                pass
        return self._forward(latents)

    def _forward(self, latents: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=1e-3)
        logits = self.scores(latents) / temperature + self.bias.unsqueeze(0)
        if self.readout == "sparsemax":
            return sparsemax(logits, dim=-1)
        return F.softmax(logits, dim=-1)

    def expectation(self, pi: torch.Tensor) -> torch.Tensor:
        """Expected offset ``E[l]`` per ordered pair, in frames, shape ``[..., M, M]``."""
        candidates = self.candidates.to(pi.dtype)
        if 0 < self.topk < self.K:
            values, index = pi.topk(self.topk, dim=-1)
            weights = values / values.sum(-1, keepdim=True).clamp_min(1e-8)
            return (weights * candidates[index]).sum(-1)
        return (pi * candidates.view(1, 1, 1, self.K)).sum(-1)

    def entropy(self, pi: torch.Tensor) -> torch.Tensor:
        """Mean entropy of the per-edge distributions."""
        return -(pi.clamp_min(1e-8).log() * pi).sum(-1).mean()
