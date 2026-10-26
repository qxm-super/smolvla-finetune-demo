"""Closed-loop SmolVLA rollout in a MuJoCo SO-ARM100 scene.

Runs the fine-tuned SmolVLA policy (``ahmedsohail2003/smolvla-so101-pickplace``)
inside a simulated SO-ARM100 pick-and-place task: the policy observes the two
cameras + 6-dim joint state, predicts a 50-step action chunk, and the chunk is
executed in physics. Episodes are recorded to video and scored for success.

Usage:
    python scripts/rollout.py --episodes 5
    python scripts/rollout.py --episodes 1 --render-video --task "..." --device cuda

Note: the MuJoCo scene is an *approximate* SO-ARM100 (different link lengths,
camera viewpoints and one elbow sign convention from the training robot), so the
transferred success rate is expected to be lower than the policy's native domain.
This script demonstrates the closed-loop VLA pipeline, not a faithful sim2sim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.so101_env import SO101PickPlace  # noqa: E402

MODEL_REPO = "ahmedsohail2003/smolvla-so101-pickplace"
MODEL_PATH = REPO_ROOT / "models" / "smolvla-so101-pickplace"
OUTPUT_DIR = REPO_ROOT / "outputs" / "rollout"

DEFAULT_TASK = "Pick up the red block and place it in the blue tray."
MAX_CHUNKS = 5  # episodes run at most 5 predictions of 50 steps (= 5 s @ 50 Hz)


def ensure_model(model_path: Path, repo_id: str) -> Path:
    """Download the model if missing, and patch its config for LeRobot 0.4.4."""
    if not (model_path / "model.safetensors").exists():
        print(f"Downloading {repo_id} ...")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, local_dir=str(model_path))

    config_path = model_path / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    changed = False
    if "pretrained_revision" in cfg:
        cfg.pop("pretrained_revision")
        changed = True
    if cfg.get("load_vlm_weights") is not False:
        cfg["load_vlm_weights"] = False
        changed = True
    if changed:
        config_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        print("Patched config.json for LeRobot 0.4.4 compatibility.")
    return model_path


def load_policy_and_processors(model_path: Path, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(model_path))
    policy.eval()
    print(f"Loaded SmolVLA policy ({sum(p.numel() for p in policy.parameters()):,} params).")

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def env_obs_to_model_obs(env_obs: dict, task: str) -> dict:
    """Convert the env observation into the dict SmolVLA's preprocessor expects."""
    state = np.asarray(env_obs["state"], dtype=np.float32)

    def hwc_to_chw(img: np.ndarray) -> torch.Tensor:
        img = np.asarray(img, dtype=np.uint8)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0, 1]

    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.front": hwc_to_chw(env_obs["images"]["front"]),
        "observation.images.wrist": hwc_to_chw(env_obs["images"]["wrist"]),
        "task": task,
    }


def render_side_by_side(env_obs: dict) -> np.ndarray:
    """Concatenate front | wrist views into one uint8 RGB frame for video."""
    front = env_obs["images"]["front"]
    wrist = env_obs["images"]["wrist"]
    return np.concatenate([front, wrist], axis=1)


def run_episode(env, policy, pre, post, task: str, device: str, record_video: bool):
    """Run one episode, returning (success, n_steps, frames)."""
    frames = []
    n_steps = 0
    env.reset()

    for _ in range(MAX_CHUNKS):
        env_obs = env.observe()
        model_obs = env_obs_to_model_obs(env_obs, task)

        with torch.no_grad():
            processed = pre(model_obs)
            actions = policy.predict_action_chunk(processed)
            actions = post(actions)
        chunk = actions[0].detach().cpu().numpy()  # (50, 6)

        for i in range(chunk.shape[0]):
            info = env.step(chunk[i])
            n_steps += 1
            if record_video:
                frames.append(render_side_by_side(env.observe()))
            if info["success"]:
                return True, n_steps, frames

    return False, n_steps, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--render-video", action="store_true",
                        help="save an mp4 per episode (requires opencv)")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    model_path = ensure_model(MODEL_PATH, MODEL_REPO)
    policy, pre, post = load_policy_and_processors(model_path, args.device)

    env = SO101PickPlace(width=224, height=224, control_freq=args.fps)

    successes = 0
    total_steps = 0

    print(f"\nRolling out SmolVLA for {args.episodes} episode(s) ...")
    for ep in range(args.episodes):
        success, n_steps, frames = run_episode(
            env, policy, pre, post, args.task, args.device, args.render_video
        )
        total_steps += n_steps
        successes += int(success)

        if args.render_video and frames:
            save_video(frames, args.fps, f"episode_{ep}.mp4")

        print(f"  Ep {ep + 1}: {'SUCCESS' if success else 'FAIL'} | steps={n_steps}")

    env.close()

    print("\n" + "=" * 44)
    print(f"Success rate: {successes}/{args.episodes} ({100 * successes / args.episodes:.0f}%)")
    print(f"Avg steps:    {total_steps / args.episodes:.0f}")


def save_video(frames: list, fps: int, name: str) -> None:
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


if __name__ == "__main__":
    main()
