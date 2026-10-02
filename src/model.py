"""The full self-supervised model: stream encoders, an EMA target and the pairwise predictor."""
from __future__ import annotations

import warnings
from typing import Dict, List

import torch
import torch.nn as nn

from .ema import EMA
from .encoders import StreamEncoders
from .losses import cross_stream_infonce, spread, variance_covariance
from .predictor import Predictor


class Model(nn.Module):
    """Optimised during pretraining and frozen for read-out; scales to any number of streams."""

    def __init__(self, cfg) -> None:
        super().__init__()
        model = cfg.model
        loss = model.get("loss", {})
        self.embed_dim = int(model.embed_dim)
        self.streams: List[str] = list(model.streams.keys())
        self.horizons = list(model.get("horizons", [1]))
        self.max_offset = int(model.get("max_offset", 4))
        if self.horizons and max(self.horizons) >= self.max_offset:
            warnings.warn("horizon is large relative to max_offset; some positive offsets "
                          "are unrepresentable", UserWarning, stacklevel=2)

        self.encoder = StreamEncoders(model.streams, self.embed_dim)
        self.predictor = Predictor(
            streams=self.streams,
            embed_dim=self.embed_dim,
            max_offset=self.max_offset,
            proj_dim=int(model.get("proj_dim", 64)),
            temperature=float(model.get("temperature", 0.1)),
            readout=model.get("readout", "sparsemax"),
            topk=int(model.get("topk", 0)),
            normalize=model.get("influence_normalize", "softmax"),
            transform_kind=model.get("transform_kind", "mlp"),
            base_residual=bool(model.get("base_residual", True)),
            loss_kind=loss.get("kind", "cosine"),
            pad=model.get("pad", "replicate"),
            target_norm=model.get("target_norm", "none"),
            dino_coeff=float(loss.get("dino_coeff", 0.0)),
            dino_cfg=dict(model.get("dino", {})),
        )

        ema = model.get("ema", {})
        self.ema = EMA(self.encoder, base_momentum=float(ema.get("base", 0.996)),
                       final_momentum=float(ema.get("final", 1.0)),
                       total_steps=ema.get("total_steps", None))

        self.vicreg_coeff = float(loss.get("vicreg_coeff", 0.0))
        self.spread_coeff = float(loss.get("spread_coeff", 0.0))
        self.align_coeff = float(loss.get("align_coeff", 0.0))
        self.align_temp = float(loss.get("align_temp", 0.1))
        # Auxiliary objective coefficient; always zero.
        self.aux_coeff = float(loss.get("aux_coeff", 0.0))
        if self.aux_coeff != 0.0:
            raise ValueError("aux_coeff is always zero")
        self.align_head = None
        if self.align_coeff > 0:
            proj = int(loss.get("align_proj_dim", 128))
            self.align_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim), nn.GELU(),
                nn.Linear(self.embed_dim, proj))

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if getattr(self, "ema", None) is not None:  # the target is held outside the module tree
            self.ema.target._apply(fn, *args, **kwargs)
        return self

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Online per-frame latents for every present stream."""
        return self.encoder(batch)

    @torch.no_grad()
    def encode_target(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stop-gradient latents from the EMA target."""
        self.ema.target.eval()
        return self.ema.target(batch)

    def forward(self, batch: Dict[str, torch.Tensor], targets: List[str],
                horizon: int = 1) -> Dict[str, object]:
        """Return the complete objective and the recovered edge quantities."""
        z_online = self.encode(batch)
        z_target = self.encode_target(batch)
        out = self.predictor(z_online, z_target, targets, horizon=horizon)
        loss = out["loss"]

        flat = torch.cat([z.reshape(-1, z.shape[-1]) for z in z_online.values()], dim=0)
        out["feature_std"] = flat.float().std(dim=0, unbiased=False).mean().detach()
        if self.vicreg_coeff > 0:
            loss = loss + self.vicreg_coeff * variance_covariance(flat)
        if self.spread_coeff > 0:
            loss = loss + self.spread_coeff * spread(flat)
        if self.align_head is not None:
            pooled = {m: self.align_head(z.mean(dim=1)) for m, z in z_online.items()}
            align = cross_stream_infonce(pooled, temperature=self.align_temp)
            out["align_loss"] = align.detach()
            loss = loss + self.align_coeff * align

        out["loss"] = loss
        return out

    @torch.no_grad()
    def update_target(self) -> float:
        """Advance the EMA target and report its momentum."""
        self.ema.update(self.encoder)
        return self.ema.momentum

    @torch.no_grad()
    def graph(self, batch: Dict[str, torch.Tensor]):
        """Batch-mean recovered edges: ``(pi, influence, E[l])`` over the stream pairs."""
        pi, influence, expected = self.predictor.edges(self.encode(batch))
        return pi.mean(0), influence.mean(0), expected.mean(0)

    def checkpoint(self) -> dict:
        """State of both the online model and its target."""
        return {"model": self.state_dict(), "ema": self.ema.state_dict()}

    def load_checkpoint(self, state: dict) -> None:
        """Restore a checkpoint written by :meth:`checkpoint`."""
        self.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
