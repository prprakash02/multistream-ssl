"""Objective terms: latent prediction, representation regularisation and cross-stream alignment."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def prediction_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None, kind: str = "cosine") -> torch.Tensor:
    """Per-frame distance between the prediction and the stop-gradient target ``[B, T, D]``."""
    if kind == "cosine":
        per = 2.0 - 2.0 * (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
    elif kind == "mse":
        per = ((pred - target) ** 2).mean(-1)
    elif kind == "smooth_l1":
        per = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
    else:
        raise ValueError(f"unknown loss kind '{kind}'")
    if mask is None:
        return per.mean()
    mask = mask.to(per.dtype)
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def variance_covariance(z: torch.Tensor, std_coeff: float = 1.0, cov_coeff: float = 0.04,
                        eps: float = 1e-4) -> torch.Tensor:
    """Variance and covariance penalty on embeddings ``[N, D]`` against a constant solution."""
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return std_coeff * F.relu(1.0 - std).mean() + cov_coeff * (off_diag ** 2).sum() / d


def spread(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Nearest-neighbour spread term on L2-normalised embeddings ``[N, D]``."""
    z = F.normalize(z, dim=-1, eps=eps)
    if z.shape[0] < 2:
        return z.new_zeros(())
    with torch.no_grad():
        similarity = z @ z.t()
        similarity.fill_diagonal_(-2.0)
        neighbour = similarity.argmax(dim=1)
    return -torch.log((z - z[neighbour]).norm(dim=-1).clamp_min(eps)).mean()


def cross_stream_infonce(pooled: dict[str, torch.Tensor], temperature: float = 0.1) -> torch.Tensor:
    """Clip-level contrastive term over co-indexed stream pairs, averaged over ordered pairs."""
    streams = [m for m, v in pooled.items() if v is not None]
    if len(streams) < 2:
        return next(iter(pooled.values())).new_zeros(())
    B = pooled[streams[0]].shape[0]
    if B < 2:
        return pooled[streams[0]].new_zeros(())
    g = {m: F.normalize(pooled[m], dim=-1, eps=1e-6) for m in streams}
    labels = torch.arange(B, device=pooled[streams[0]].device)
    total, count = pooled[streams[0]].new_zeros(()), 0
    for i, a in enumerate(streams):
        for b in streams[i + 1:]:
            logits = (g[a] @ g[b].t()) / max(temperature, 1e-3)
            total = total + 0.5 * (F.cross_entropy(logits, labels)
                                   + F.cross_entropy(logits.t(), labels))
            count += 1
    return total / max(count, 1)


class DINOHead(nn.Module):
    """Projection to an L2-normalised bottleneck followed by cosine logits over prototypes."""

    def __init__(self, in_dim: int, out_dim: int = 1024, hidden: int = 512,
                 bottleneck: int = 128, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(max(n_layers - 2, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, bottleneck)]
        self.mlp = nn.Sequential(*layers)
        self.prototypes = nn.Linear(bottleneck, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.prototypes.weight, dim=-1, eps=1e-6))


class DINOLoss(nn.Module):
    """Prototype cross-entropy with a centred and sharpened teacher."""

    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student = F.log_softmax(student_logits / self.student_temp, dim=-1)
        teacher = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1).detach()
        loss = -(teacher * student).sum(dim=-1).mean()
        if self.training:  # validation must not move the centre buffer
            self._update_center(teacher_logits)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)
