"""Evaluate the ACT policy trained in this scene.

Loads the ACT checkpoint (and its saved pre/post processors) from
``outputs/train/act_so101`` and runs closed-loop rollouts in the MuJoCo scene,
scoring success by whether the block ends up inside the tray. Box positions are
randomized per episode (same +-2 cm spread used during demo collection).

ACT predicts a 100-step chunk (chunk_size=100 @ 20 Hz = 5 s); each chunk is executed
open-loop, then the policy re-observes and predicts the next chunk.

Usage:
    python scripts/eval_act.py --episodes 20
    python scripts/eval_act.py --episodes 5 --render-video
"""

from __future__ import annotations

import argparse
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
OUTPUT_DIR = REPO_ROOT / "outputs" / "eval"

FPS = 20  # matches the demo dataset fps (collect_demos.py --fps 20)
MAX_CHUNKS = 4  # 4 chunks x 100 steps = 400 steps @ 20 Hz = 20 s


def load_policy_and_processors(model_path: Path, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.eval()
    n = sum(p.numel() for p in policy.parameters())
    print(f"Loaded ACT ({n:,} params).")

    # The checkpoint dir contains the pre/post processor configs saved by
    # train_act.py (with THIS dataset's normalization stats), so loading from
    # pretrained_path picks them up automatically.
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def env_obs_to_model_obs(env_obs: dict) -> dict:
    state = np.asarray(env_obs["state"], dtype=np.float32)

    def hwc_to_chw(img: np.ndarray) -> torch.Tensor:
        img = np.asarray(img, dtype=np.uint8)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.front": hwc_to_chw(env_obs["images"]["front"]),
        "observation.images.wrist": hwc_to_chw(env_obs["images"]["wrist"]),
    }


def render_side_by_side(env_obs: dict) -> np.ndarray:
    return np.concatenate([env_obs["images"]["front"], env_obs["images"]["wrist"]], axis=1)


def run_episode(env, policy, pre, post, device, record_video, box_pos):
    frames = []
    n_steps = 0
    env.reset(box_pos=box_pos)

    for _ in range(MAX_CHUNKS):
        env_obs = env.observe()
        model_obs = env_obs_to_model_obs(env_obs)

        with torch.no_grad():
            processed = pre(model_obs)
            actions = policy.predict_action_chunk(processed)
            actions = post(actions)
        chunk = actions[0].detach().cpu().numpy()  # (chunk_size, 6)

        for i in range(chunk.shape[0]):
            info = env.step(chunk[i])
            n_steps += 1
            if record_video:
                frames.append(render_side_by_side(env.observe()))
            if info["success"]:
                return True, n_steps, frames

    return False, n_steps, frames


def save_video(frames, fps, name):
    try:
        import cv2
    except ImportError:
        print("  (opencv not installed — skipping video)")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    out = OUTPUT_DIR / name
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  video saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    policy, pre, post = load_policy_and_processors(args.checkpoint, args.device)
    env = SO101PickPlace(width=224, height=224, control_freq=FPS)

    rng = np.random.default_rng(0)
    successes = 0
    total_steps = 0

    print(f"\nRolling out ACT for {args.episodes} episode(s) ...")
    for ep in range(args.episodes):
        box_xy = np.array([0.15, 0.0]) + rng.uniform(-0.02, 0.02, size=2)
        success, n_steps, frames = run_episode(
            env, policy, pre, post, args.device, args.render_video, box_xy
        )
        total_steps += n_steps
        successes += int(success)

        if args.render_video and frames:
            save_video(frames, FPS, f"act_episode_{ep}.mp4")

        print(f"  Ep {ep + 1:2d}: {'SUCCESS' if success else 'FAIL'} | steps={n_steps}")

    env.close()

    print("\n" + "=" * 44)
    print(f"Success rate: {successes}/{args.episodes} ({100 * successes / args.episodes:.0f}%)")
    print(f"Avg steps:    {total_steps / args.episodes:.0f}")


if __name__ == "__main__":
    main()
