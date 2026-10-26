"""Extract one sample observation from the SO-101 pick-place dataset.

Downloads the dataset (first time) and saves a single observation — the two camera
views and the 6-dim robot state — as small assets that the inference demo can load
without needing the full 300 MB dataset.

Usage:
    python scripts/extract_sample.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

DATASET_REPO = "ahmedsohail2003/so101-sim-pickplace"
EPISODE = 0
FRAME = 0  # first frame of the episode


def to_uint8_hwc(img_tensor) -> np.ndarray:
    """LeRobot returns float32 CHW images in [0, 1]; convert to uint8 HWC for PNG."""
    img = np.asarray(img_tensor)
    if img.dtype != np.uint8:
        img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    if img.shape[0] == 3:  # CHW -> HWC
        img = img.transpose(1, 2, 0)
    return img


def main() -> None:
    print(f"Loading dataset {DATASET_REPO} episode {EPISODE} ...")
    ds = LeRobotDataset(DATASET_REPO, episodes=[EPISODE], image_transforms=None)

    item = ds[FRAME]

    state = np.asarray(item["observation.state"], dtype=np.float32)
    front = to_uint8_hwc(item["observation.images.front"])
    wrist = to_uint8_hwc(item["observation.images.wrist"])
    task = str(item["task"])

    print(f"state shape {state.shape}, front {front.shape}, wrist {wrist.shape}")
    print(f"task = {task!r}")
    print(f"state = {state.tolist()}")

    Image.fromarray(front).save(ASSETS / "front.png")
    Image.fromarray(wrist).save(ASSETS / "wrist.png")
    np.save(ASSETS / "state.npy", state)

    meta = {
        "dataset_repo": DATASET_REPO,
        "episode": EPISODE,
        "frame": FRAME,
        "task": task,
        "state": state.tolist(),
    }
    (ASSETS / "sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved assets to {ASSETS}")


if __name__ == "__main__":
    main()
