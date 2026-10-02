"""Minimal YAML configuration with attribute access and CLI overrides."""
from __future__ import annotations

import os
import random
from typing import Any, Sequence

import numpy as np
import torch
import yaml


class Config(dict):
    """Dict with attribute access; nested dicts are wrapped on read."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key] if key in self else default


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _set_path(cfg: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str, overrides: Sequence[str] = ()) -> Config:
    """Read a YAML file and apply ``key.path=value`` overrides."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    for item in overrides:
        key, _, value = item.partition("=")
        _set_path(raw, key.strip(), yaml.safe_load(value))
    return _wrap(raw)


def save_config(cfg: Config, path: str) -> None:
    """Write a configuration back to YAML."""
    with open(path, "w") as fh:
        yaml.safe_dump(_to_plain(cfg), fh, sort_keys=False)


def seed_everything(seed: int = 42) -> int:
    """Seed python, numpy and torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
