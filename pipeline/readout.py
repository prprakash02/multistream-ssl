#!/usr/bin/env python
"""Frozen-encoder read-out: linear probe accuracy and the recovered per-edge quantities.

    python -m pipeline.readout --config configs/default.yaml --checkpoint outputs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, seed_everything  # noqa: E402
from src.data import build_loader, move_batch  # noqa: E402
from src.model import Model  # noqa: E402
from src.probe import LinearProbe, pool_features  # noqa: E402


@torch.no_grad()
def extract(model: Model, loader, device: str, pooling: str):
    """Pooled frozen features and labels for one split."""
    feats, labels = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        z = model.encode(batch["mods"])
        feats.append(pool_features(z, model.streams, how=pooling).float().cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(feats), torch.cat(labels)


def fit_probe(train_feats, train_labels, num_classes: int, epochs: int, lr: float, device: str):
    """Train a linear classifier on frozen features."""
    probe = LinearProbe(train_feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    feats, labels = train_feats.to(device), train_labels.to(device)
    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        criterion(probe(feats), labels).backward()
        optimizer.step()
    return probe


@torch.no_grad()
def recovered_edges(model: Model, loader, device: str):
    """Batch-mean expected offset and influence over the stream pairs."""
    batch = move_batch(next(iter(loader)), device)
    _, influence, expected = model.graph(batch["mods"])
    return expected.cpu(), influence.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="frozen-encoder read-out")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-o", "--override", nargs="*", default=[], help="key.path=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.get("seed", 42)))
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    model = Model(cfg).to(device)
    model.load_checkpoint(torch.load(args.checkpoint, map_location=device))
    model.eval()

    train_loader = build_loader(cfg, "train", shuffle=False)
    test_loader = build_loader(cfg, "test", shuffle=False)
    pooling = cfg.eval.get("pooling", "mean")
    train_feats, train_labels = extract(model, train_loader, device, pooling)
    test_feats, test_labels = extract(model, test_loader, device, pooling)

    probe = fit_probe(train_feats, train_labels, train_loader.dataset.num_classes,
                      int(cfg.eval.get("probe_epochs", 100)),
                      float(cfg.eval.get("probe_lr", 1e-3)), device)
    probe.eval()
    with torch.no_grad():
        accuracy = (probe(test_feats.to(device)).argmax(-1).cpu() == test_labels).float().mean()
    print(f"probe accuracy: {float(accuracy):.4f}")

    expected, influence = recovered_edges(model, test_loader, device)
    print(f"streams: {model.streams}")
    print(f"expected offset (frames, source -> target):\n{expected.numpy().round(3)}")
    print(f"influence (source -> target):\n{influence.numpy().round(3)}")


if __name__ == "__main__":
    main()
