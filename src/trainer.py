"""Pretraining loop: sampled targets, mixed precision, EMA updates and checkpointing."""
from __future__ import annotations

import csv
import math
import os
import random
from typing import List, Optional

import torch

from .data import move_batch


def cosine_schedule(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    """Drives pretraining for one configuration and writes checkpoints and a metrics log."""

    def __init__(self, cfg, model, train_loader, val_loader, device: str, run_dir: str) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        train = cfg.train
        optim = cfg.optim
        self.epochs = int(train.get("epochs", 100))
        self.grad_clip = float(train.get("grad_clip", 1.0))
        self.log_every = int(train.get("log_every", 50))
        self.targets_per_step = int(train.get("targets_per_step", 1))
        self.horizons = list(cfg.model.get("horizons", [1]))
        self.base_lr = float(optim.get("lr", 1e-3))
        self.min_lr = float(optim.get("min_lr", 1e-6))
        self.warmup_steps = int(optim.get("warmup_epochs", 5)) * max(len(train_loader), 1)
        schedule_epochs = int(optim.get("schedule_epochs", self.epochs))
        self.total_steps = schedule_epochs * max(len(train_loader), 1)

        self.use_amp = bool(train.get("amp", False)) and device != "cpu"
        self.amp_dtype = torch.bfloat16 if train.get("amp_dtype", "bfloat16") == "bfloat16" \
            else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.base_lr, weight_decay=float(optim.get("weight_decay", 0.05)),
            betas=tuple(optim.get("betas", [0.9, 0.95])))

        self.step = 0
        self.best_val = float("inf")
        self.metrics_path = os.path.join(run_dir, "metrics.csv")

    def _targets(self, present: List[str]) -> List[str]:
        return random.sample(present, min(self.targets_per_step, len(present)))

    def _set_lr(self) -> float:
        lr = cosine_schedule(self.step, self.warmup_steps, self.total_steps,
                             self.base_lr, self.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_epoch(self, epoch: int) -> dict:
        """Run one pass over the training loader."""
        self.model.train()
        totals = {"loss": 0.0, "prediction_loss": 0.0}
        seen = 0
        for i, batch in enumerate(self.train_loader):
            batch = move_batch(batch, self.device)
            present = list(batch["mods"].keys())
            horizon = random.choice(self.horizons)
            lr = self._set_lr()
            with torch.autocast(device_type=self.device.split(":")[0], dtype=self.amp_dtype,
                                enabled=self.use_amp):
                out = self.model(batch["mods"], targets=self._targets(present), horizon=horizon)
                loss = out["loss"]
            if not torch.isfinite(loss):  # skip the step rather than corrupt the weights
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.update_target()
            self.step += 1
            seen += 1
            totals["loss"] += float(loss.detach())
            totals["prediction_loss"] += float(out["prediction_loss"].detach())
            if self.log_every and i % self.log_every == 0:
                print(f"[epoch {epoch}] step {i}/{len(self.train_loader)} "
                      f"loss {float(loss.detach()):.4f} lr {lr:.2e}")
        return {k: v / max(seen, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """Mean objective over the validation loader, or ``None`` when there is no loader."""
        if self.val_loader is None:
            return None
        self.model.eval()
        total, seen = 0.0, 0
        for batch in self.val_loader:
            batch = move_batch(batch, self.device)
            present = list(batch["mods"].keys())
            out = self.model(batch["mods"], targets=present, horizon=self.horizons[0])
            total += float(out["loss"])
            seen += 1
        return total / max(seen, 1)

    def save(self, name: str, epoch: int) -> None:
        """Write a checkpoint with the optimiser state and the epoch reached."""
        torch.save({**self.model.checkpoint(), "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch, "step": self.step}, os.path.join(self.run_dir, name))

    def _log(self, row: dict) -> None:
        exists = os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def fit(self) -> float:
        """Train for the configured epochs and return the best validation objective."""
        for epoch in range(self.epochs):
            stats = self.train_epoch(epoch)
            val = self.validate()
            row = {"epoch": epoch, **stats, "val_loss": val if val is not None else ""}
            self._log(row)
            print(f"[epoch {epoch}] train {stats['loss']:.4f}"
                  + (f" val {val:.4f}" if val is not None else ""))
            self.save("last.pt", epoch)
            if val is not None and val < self.best_val:
                self.best_val = val
                self.save("best.pt", epoch)
        return self.best_val
