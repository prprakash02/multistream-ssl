"""Directional influence, per-pair transforms and the prediction of a stream's future latent."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import DINOHead, DINOLoss, prediction_loss
from .offsets import OffsetDistribution, shift, soft_align


class DirectionalInfluence(nn.Module):
    """Non-negative weight ``a_{j->i}`` per ordered pair, from a learned table and the context."""

    def __init__(self, num_streams: int, embed_dim: int = 256, hidden: int = 128,
                 normalize: str = "softmax") -> None:
        super().__init__()
        if normalize not in {"softmax", "softplus"}:
            raise ValueError(f"unknown normalisation '{normalize}'")
        self.M = num_streams
        self.normalize = normalize
        self.register_buffer("self_mask", torch.eye(num_streams).bool())
        self.table = nn.Parameter(torch.zeros(num_streams, num_streams))
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return ``a`` of shape ``[B, M_src, M_tgt]`` with the self edge masked out."""
        B = context.shape[0]
        src = context.unsqueeze(2).expand(B, self.M, self.M, -1)
        tgt = context.unsqueeze(1).expand(B, self.M, self.M, -1)
        logits = self.table.unsqueeze(0) + self.pair_mlp(torch.cat([src, tgt], dim=-1)).squeeze(-1)
        mask = self.self_mask.unsqueeze(0)
        if self.normalize == "softmax":
            a = F.softmax(logits.masked_fill(mask, float("-inf")), dim=1)
            return torch.nan_to_num(a, nan=0.0)
        return F.softplus(logits).masked_fill(mask, 0.0)


class PairTransform(nn.Module):
    """One map ``T_{j->i}`` per ordered pair into the target stream's predictive space."""

    def __init__(self, num_streams: int, embed_dim: int, kind: str = "mlp", hidden_ratio: int = 2):
        super().__init__()
        if kind not in {"linear", "mlp"}:
            raise ValueError(f"unknown transform kind '{kind}'")
        self.M = num_streams
        hidden = embed_dim * hidden_ratio

        def block():
            if kind == "linear":
                return nn.Linear(embed_dim, embed_dim)
            return nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, embed_dim))

        self.maps = nn.ModuleList([block() if s != t else nn.Identity()
                                   for s in range(num_streams) for t in range(num_streams)])

    def forward(self, src: int, tgt: int, x: torch.Tensor) -> torch.Tensor:
        return self.maps[src * self.M + tgt](x)


class Predictor(nn.Module):
    """Predicts ``z_i(t+h)`` from ``z_i(t)`` and the other streams marginalised over their edges."""

    def __init__(self, streams: List[str], embed_dim: int, max_offset: int = 4,
                 proj_dim: int = 64, temperature: float = 0.1, readout: str = "sparsemax",
                 topk: int = 0, normalize: str = "softmax", transform_kind: str = "mlp",
                 base_residual: bool = True, loss_kind: str = "cosine", pad: str = "replicate",
                 target_norm: str = "none", dino_coeff: float = 0.0,
                 dino_cfg: dict | None = None) -> None:
        super().__init__()
        self.streams = list(streams)
        self.M = len(streams)
        self.index = {m: k for k, m in enumerate(self.streams)}
        self.base_residual = base_residual
        self.loss_kind = loss_kind
        self.pad = pad
        self.target_norm = target_norm
        self.dino_coeff = float(dino_coeff)
        self.offsets = OffsetDistribution(
            self.M, max_offset=max_offset, embed_dim=embed_dim, proj_dim=proj_dim,
            temperature=temperature, readout=readout, topk=topk, pad=pad)
        self.influence = DirectionalInfluence(self.M, embed_dim=embed_dim, normalize=normalize)
        self.transform = PairTransform(self.M, embed_dim, kind=transform_kind)
        self.dino_head, self.dino = None, None
        if self.dino_coeff > 0:
            cfg = dino_cfg or {}
            out_dim = int(cfg.get("out_dim", 1024))
            self.dino_head = DINOHead(embed_dim, out_dim=out_dim,
                                      hidden=int(cfg.get("hidden", 512)),
                                      bottleneck=int(cfg.get("bottleneck", 128)))
            self.dino = DINOLoss(out_dim, teacher_temp=float(cfg.get("teacher_temp", 0.04)),
                                 student_temp=float(cfg.get("student_temp", 0.1)),
                                 center_momentum=float(cfg.get("center_momentum", 0.9)))

    def _stack(self, z: Dict[str, torch.Tensor]):
        """Return the pooled context ``[B, M, D]``, the padded latents ``[B, M, T, D]`` and
        the present streams."""
        present = [m for m in self.streams if m in z]
        ref = next(iter(z.values()))
        B, T, D = ref.shape
        context = ref.new_zeros(B, self.M, D)
        latents = ref.new_zeros(B, self.M, T, D)
        for m in present:
            context[:, self.index[m]] = z[m].mean(dim=1).detach()
            latents[:, self.index[m]] = z[m]
        return context, latents, present

    def edges(self, z: Dict[str, torch.Tensor]):
        """Return ``(pi, a, E[l])`` for a batch of online latents."""
        context, latents, _ = self._stack(z)
        pi = self.offsets(latents)
        return pi, self.influence(context), self.offsets.expectation(pi)

    def predict(self, z: Dict[str, torch.Tensor], target: str, pi: torch.Tensor,
                a: torch.Tensor, present: List[str]) -> torch.Tensor:
        """Assemble the prediction for one target stream, shape ``[B, T, D]``."""
        i = self.index[target]
        out = z[target] if self.base_residual else torch.zeros_like(z[target])
        for src in present:
            if src == target:
                continue
            j = self.index[src]
            aligned = soft_align(z[src], pi[:, j, i, :], self.offsets.candidates, pad=self.pad)
            out = out + a[:, j, i].view(-1, 1, 1) * self.transform(j, i, aligned)
        return out

    def forward(self, z_online: Dict[str, torch.Tensor], z_target: Dict[str, torch.Tensor],
                targets: List[str], horizon: int = 1) -> Dict[str, object]:
        """Return the predictive loss and the recovered edge quantities."""
        context, latents, present = self._stack(z_online)
        pi = self.offsets(latents)
        a = self.influence(context)
        ref = next(iter(z_online.values()))
        T = ref.shape[1]
        mask = torch.zeros(ref.shape[0], T, device=pi.device, dtype=torch.bool)
        mask[:, : max(T - horizon, 0)] = True

        total = torch.zeros((), device=pi.device, dtype=pi.dtype)
        prediction_total = torch.zeros_like(total)
        dino_total = torch.zeros_like(total)
        count = 0
        for tgt in targets:
            if tgt not in present:
                continue
            pred = self.predict(z_online, tgt, pi, a, present)
            future = shift(z_target[tgt].detach(), horizon, pad=self.pad)
            if self.target_norm == "layernorm":
                future = F.layer_norm(future, (future.shape[-1],))
            dino_term = pred.new_zeros(())
            if self.dino is not None and mask.any():
                flat = mask.reshape(-1)
                dim = pred.shape[-1]
                student = self.dino_head(pred.reshape(-1, dim))[flat]
                with torch.no_grad():
                    teacher = self.dino_head(future.reshape(-1, dim))[flat]
                dino_term = self.dino(student, teacher)
            regression = prediction_loss(pred, future, mask=mask, kind=self.loss_kind)
            total = total + regression + self.dino_coeff * dino_term
            prediction_total = prediction_total + regression
            dino_total = dino_total + dino_term
            count += 1

        denom = max(count, 1)
        return {
            "loss": total / denom,
            "prediction_loss": prediction_total / denom,
            "dino_loss": dino_total / denom,
            "pi": pi.detach(),
            "influence": a.detach(),
            "expected_offset": self.offsets.expectation(pi).detach(),
            "offset_entropy": self.offsets.entropy(pi).detach(),
        }
