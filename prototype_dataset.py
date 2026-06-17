import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedLandmarkDataset(Dataset):
    """Loads cached MediaPipe-style landmarks prepared from real datasets."""

    def __init__(self, manifest_path: str = "data/prototype_manifest.json",
                 split: str = "train"):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path} is missing. Run prepare_real_datasets.py first."
            )

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.root = self.manifest_path.parent
        self.samples: List[Dict] = [
            sample for sample in manifest["samples"] if sample["split"] == split
        ]
        self.gloss_to_id = manifest["gloss_to_id"]
        self.id_to_gloss = {int(v): k for k, v in self.gloss_to_id.items()}
        self.caption_to_id = manifest.get("caption_to_id", {})
        self.id_to_caption = {int(v): k for k, v in self.caption_to_id.items()}

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature_path = self.root / sample["feature_path"]
        data = np.load(feature_path)

        return {
            "hand": torch.from_numpy(data["hand"]).float(),
            "face": torch.from_numpy(data["face"]).float(),
            "pose": torch.from_numpy(data["pose"]).float(),
            "gloss_id": int(sample["gloss_id"]),
            "caption_id": int(sample.get("caption_id", 0)),
            "gloss": sample["gloss"],
            "caption": sample["caption"],
            "dataset": sample["dataset"],
            "source_path": sample.get("source_path", ""),
        }


class CachedLandmarkCollator:
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def _pad(self, tensors):
        max_len = max(t.shape[0] for t in tensors)
        width = tensors[0].shape[1]
        padded = []
        lengths = []
        for tensor in tensors:
            out = torch.full((max_len, width), self.pad_value, dtype=tensor.dtype)
            out[: tensor.shape[0]] = tensor
            padded.append(out)
            lengths.append(tensor.shape[0])
        return torch.stack(padded), torch.tensor(lengths, dtype=torch.long)

    def __call__(self, batch):
        hand, lengths = self._pad([item["hand"] for item in batch])
        face, _ = self._pad([item["face"] for item in batch])
        pose, _ = self._pad([item["pose"] for item in batch])

        return {
            "hand": hand,
            "face": face,
            "pose": pose,
            "lengths": lengths,
            "gloss_ids": torch.tensor([item["gloss_id"] for item in batch], dtype=torch.long),
            "caption_ids": torch.tensor([item["caption_id"] for item in batch], dtype=torch.long),
            "glosses": [item["gloss"] for item in batch],
            "captions": [item["caption"] for item in batch],
            "datasets": [item["dataset"] for item in batch],
        }
