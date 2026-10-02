"""Gradient-free exponential-moving-average copy of the encoder used as the prediction target."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class EMA:
    """Holds the target network and updates it towards the online weights each step."""

    def __init__(self, online: nn.Module, base_momentum: float = 0.996,
                 final_momentum: float = 1.0, total_steps: int | None = None) -> None:
        self.target = copy.deepcopy(online)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.target.eval()
        self.base = base_momentum
        self.final = final_momentum
        self.total_steps = total_steps
        self._step = 0

    @property
    def momentum(self) -> float:
        if not self.total_steps:
            return self.base
        progress = min(self._step / self.total_steps, 1.0)
        return self.final - (self.final - self.base) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        """Move the target towards the online weights and copy its buffers."""
        m = self.momentum
        for target_param, online_param in zip(self.target.parameters(), online.parameters()):
            target_param.mul_(m).add_(online_param.detach(), alpha=1.0 - m)
        for target_buffer, online_buffer in zip(self.target.buffers(), online.buffers()):
            target_buffer.copy_(online_buffer)
        self._step += 1

    def state_dict(self) -> dict:
        return {"target": self.target.state_dict(), "step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self.target.load_state_dict(state["target"])
        self._step = state.get("step", 0)
