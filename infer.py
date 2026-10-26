"""Language-conditioned action inference with SmolVLA (450M).

Loads the SmolVLA vision-language-action policy fine-tuned on the SO-ARM100
pick-and-place task, feeds it a natural-language instruction plus one camera
observation (two views + 6-dim joint state), and rolls out a 50-step action
trajectory in joint space.

The sample observation is loaded from `assets/` (extracted from the training
dataset by `scripts/extract_sample.py`), so the demo runs without the full
300 MB dataset. The model weights are downloaded from the Hugging Face Hub on
first run (or loaded from `models/` if already present).

Usage:
    python infer.py                                   # sample obs + default instruction
    python infer.py --task "Pick up the red block."   # try another instruction
    python infer.py --device cpu                      # no GPU
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
ASSETS = REPO_ROOT / "assets"
MODEL_REPO = "ahmedsohail2003/smolvla-so101-pickplace"
MODEL_PATH = REPO_ROOT / "models" / "smolvla-so101-pickplace"

DEFAULT_TASK = "Pick up the red block and place it in the blue tray."
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def ensure_model(model_path: Path, repo_id: str) -> Path:
    """Download the model if missing, and patch its config for LeRobot 0.4.4.

    The model was saved with LeRobot 0.6.0, whose `config.json` carries two fields
    that 0.4.4 rejects: `pretrained_revision` (unknown field) and
    `load_vlm_weights=True` (triggers a redundant 2 GB VLM download — the full VLM
    weights are already inside `model.safetensors`).
    """
    model_path = Path(model_path)
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


def load_policy(model_path: Path) -> SmolVLAPolicy:
    print(f"Loading SmolVLA policy from {model_path} ...")
    policy = SmolVLAPolicy.from_pretrained(str(model_path))
    policy.eval()
    print(f"  params: {sum(p.numel() for p in policy.parameters()):,}")
    return policy


def load_sample_observation(task: str) -> dict:
    state = np.load(ASSETS / "state.npy").astype(np.float32)

    def png_to_chw(path: str) -> torch.Tensor:
        img = np.array(Image.open(path).convert("RGB"))  # (H, W, 3) uint8, writable copy
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0, 1]

    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.front": png_to_chw(str(ASSETS / "front.png")),
        "observation.images.wrist": png_to_chw(str(ASSETS / "wrist.png")),
        "task": task,
    }


def visualize(task: str, actions: np.ndarray, out_path: Path) -> None:
    """Render the input observation and the predicted action trajectory."""
    front = np.array(Image.open(ASSETS / "front.png").convert("RGB"))
    wrist = np.array(Image.open(ASSETS / "wrist.png").convert("RGB"))

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])

    # Left: the observation the model was conditioned on.
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(np.concatenate([front, wrist], axis=1))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("Observation (front + wrist cameras)", fontsize=11)
    ax_img.text(
        0.5,
        -0.06,
        f'Instruction: "{task}"',
        ha="center",
        va="top",
        transform=ax_img.transAxes,
        fontsize=9,
        style="italic",
    )

    # Right: the 50-step predicted joint trajectory.
    ax = fig.add_subplot(gs[0, 1])
    steps = np.arange(actions.shape[0])
    for j, name in enumerate(JOINT_NAMES):
        ax.plot(steps, actions[:, j], marker="o", markersize=2, linewidth=1.2, label=name)
    ax.set_xlabel("Action step")
    ax.set_ylabel("Joint setpoint (rad)")
    ax.set_title(f"Predicted {actions.shape[0]}-step action chunk")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="device to run inference on (cuda|cpu)")
    parser.add_argument("--task", default=DEFAULT_TASK, help="language instruction")
    parser.add_argument(
        "--output",
        default=str(ASSETS / "inference_result.png"),
        help="where to write the visualization",
    )
    args = parser.parse_args()

    model_path = ensure_model(MODEL_PATH, MODEL_REPO)
    policy = load_policy(model_path)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    observation = load_sample_observation(args.task)

    with torch.no_grad():
        processed = pre(observation)
        actions = policy.predict_action_chunk(processed)
        actions = post(actions)

    actions_np = actions[0].detach().cpu().numpy()
    print(f"Predicted action chunk shape: {actions_np.shape}")
    print("First 3 action steps:")
    print(actions_np[:3])

    visualize(args.task, actions_np, Path(args.output))


if __name__ == "__main__":
    main()
