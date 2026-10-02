#!/usr/bin/env python
"""Self-supervised pretraining entry point.

    python -m pipeline.train --config configs/default.yaml --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config, seed_everything  # noqa: E402
from src.data import build_loader  # noqa: E402
from src.model import Model  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="self-supervised pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("-o", "--override", nargs="*", default=[], help="key.path=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.get("seed", 42)))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda unavailable, using cpu")
        device = "cpu"

    run_dir = args.run_dir or os.path.join(
        cfg.get("output_root", "outputs"), f"{cfg.data.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[run] {run_dir}")

    train_loader = build_loader(cfg, "train")
    val_loader = None
    if os.path.exists(os.path.join(cfg.data.root, cfg.data.name, "manifest_val.csv")):
        val_loader = build_loader(cfg, "val")

    trainer = Trainer(cfg, Model(cfg), train_loader, val_loader, device, run_dir)
    best = trainer.fit()
    print(f"[done] best validation objective {best:.4f}")


if __name__ == "__main__":
    main()
