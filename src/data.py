"""Stream-agnostic clip dataset, collation and loader construction.
The preprocessing stage writes ``<root>/<name>/`` with a meta file, split manifests and samples.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_meta(processed_dir: str) -> dict:
    """Read ``meta.json``: the common length ``T``, the class count and the stream table."""
    with open(os.path.join(processed_dir, "meta.json"), "r") as fh:
        return json.load(fh)


class ClipDataset(Dataset):
    """Clips stored as ``samples/<id>.npz`` with one ``arr_<stream>`` of shape ``[T, *raw]``
    per stream and an integer ``label``."""

    def __init__(self, processed_dir: str, split: str, streams: Optional[List[str]] = None) -> None:
        super().__init__()
        self.processed_dir = processed_dir
        self.meta = load_meta(processed_dir)
        available = list(self.meta["streams"].keys())
        self.streams = list(streams) if streams else available
        for m in self.streams:
            if m not in available:
                raise KeyError(f"stream '{m}' is not in {available}")
        self.T = int(self.meta["T"])
        self.num_classes = int(self.meta["num_classes"])
        manifest = os.path.join(processed_dir, f"manifest_{split}.csv")
        if not os.path.exists(manifest):
            raise FileNotFoundError(f"missing {manifest}; run the preprocessing stage first")
        self.manifest = pd.read_csv(manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(self.processed_dir, path)
        npz = np.load(path, allow_pickle=False)
        mods: Dict[str, torch.Tensor] = {}
        for m in self.streams:
            key = f"arr_{m}"
            if key in npz:  # a stream may be absent for an individual clip
                mods[m] = torch.from_numpy(np.asarray(npz[key]).astype(np.float32))
        return {"mods": mods, "label": int(row["label"]), "id": str(row["id"])}


def collate(items: List[dict]) -> dict:
    """Stack clips, zero-filling and masking any stream absent from an item."""
    union: List[str] = []
    for item in items:
        for m in item["mods"]:
            if m not in union:
                union.append(m)
    mods, avail = {}, {}
    for m in union:
        present = [item["mods"].get(m) for item in items]
        template = next(p for p in present if p is not None)
        mods[m] = torch.stack([p if p is not None else torch.zeros_like(template) for p in present])
        avail[m] = torch.tensor([p is not None for p in present], dtype=torch.bool)
    return {
        "mods": mods,
        "avail": avail,
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "id": [item["id"] for item in items],
    }


def move_batch(batch: dict, device) -> dict:
    """Move a collated batch to ``device``."""
    batch["mods"] = {m: v.to(device, non_blocking=True) for m, v in batch["mods"].items()}
    batch["label"] = batch["label"].to(device, non_blocking=True)
    return batch


def build_loader(cfg, split: str, shuffle: Optional[bool] = None) -> DataLoader:
    """Build a loader for one split from ``cfg.data``."""
    data = cfg.data
    processed_dir = os.path.join(data.root, data.name)
    dataset = ClipDataset(processed_dir, split, streams=list(cfg.model.streams.keys()))
    return DataLoader(
        dataset,
        batch_size=int(data.get("batch_size", 32)),
        shuffle=split == "train" if shuffle is None else shuffle,
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=True,
        drop_last=split == "train",
        collate_fn=collate,
    )
