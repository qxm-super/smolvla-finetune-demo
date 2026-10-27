"""Render one successful ACT rollout to an animated GIF for the README.

Loads the trained ACT checkpoint, runs a single closed-loop episode in the MuJoCo scene
(using the same box position that succeeds in ``eval_act.py`` episode 1), and writes a
downsampled side-by-side (front + wrist) GIF to ``assets/act_rollout.gif``.

Usage:
    python scripts/make_rollout_gif.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.so101_env import SO101PickPlace  # noqa: E402

CHECKPOINT = REPO_ROOT / "outputs" / "train" / "act_so101"
OUT = REPO_ROOT / "assets" / "act_rollout.gif"

FPS = 20
MAX_CHUNKS = 4
TARGET_FRAMES = 60  # downsampled GIF length
GIF_WIDTH = 400


def load_policy_and_processors(model_path: Path, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(model_path)).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def env_obs_to_model_obs(env_obs: dict) -> dict:
    state = np.asarray(env_obs["state"], dtype=np.float32)

    def hwc_to_chw(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.front": hwc_to_chw(env_obs["images"]["front"]),
        "observation.images.wrist": hwc_to_chw(env_obs["images"]["wrist"]),
    }


def side_by_side(env_obs: dict) -> np.ndarray:
    return np.concatenate([env_obs["images"]["front"], env_obs["images"]["wrist"]], axis=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy_and_processors(CHECKPOINT, device)
    env = SO101PickPlace(width=224, height=224, control_freq=FPS)

    # Same box position as eval_act.py episode 1 (known to succeed).
    rng = np.random.default_rng(0)
    box_xy = np.array([0.15, 0.0]) + rng.uniform(-0.02, 0.02, size=2)

    frames = []
    success = False
    env.reset(box_pos=box_xy)
    for _ in range(MAX_CHUNKS):
        with torch.no_grad():
            processed = pre(env_obs_to_model_obs(env.observe()))
            actions = post(policy.predict_action_chunk(processed))
        for a in actions[0].detach().cpu().numpy():
            info = env.step(a)
            frames.append(side_by_side(env.observe()))
            if info["success"]:
                success = True
                break
        if success:
            break
    env.close()

    print(f"captured {len(frames)} frames (success={success})")
    if not success:
        print("WARNING: episode did not succeed; gif will show a failed attempt")

    from PIL import Image

    step = max(1, len(frames) // TARGET_FRAMES)
    selected = frames[::step]
    h, w = selected[0].shape[:2]
    scale = GIF_WIDTH / w
    size = (GIF_WIDTH, int(h * scale))
    gif_frames = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in selected]
    gif_frames[0].save(
        OUT,
        save_all=True,
        append_images=gif_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    print(f"saved {OUT} ({len(gif_frames)} frames, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
