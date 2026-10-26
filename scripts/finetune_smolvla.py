"""Fine-tune SmolVLA (expert-only) on scripted demos collected in this MuJoCo scene.

Loads the cached ``smolvla-so101-pickplace`` checkpoint as the base, freezes the VLM
backbone, and trains only the action expert + state/action projections on the
demonstrations in ``datasets/smolvla_so101_demos``. The normalization stats are
re-derived from *this* dataset so the fine-tuned model matches the scene.

Usage:
    python scripts/finetune_smolvla.py --steps 2500 --batch-size 2
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the policy factory first: it registers the "smolvla" config subclass.
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

BASE_MODEL = REPO_ROOT / "models" / "smolvla-so101-pickplace"
DATASET_NAME = "smolvla_so101_demos"
DATASET_DIR = REPO_ROOT / "datasets" / DATASET_NAME
OUTPUT_DIR = REPO_ROOT / "outputs" / "train" / "smolvla_so101"

# Dataset key -> policy key (the policy's VLM expects camera1/camera2).
RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-freq", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # 1. Load the base policy config (load_vlm_weights is already False, so the VLM
    #    weights come from the checkpoint rather than a fresh 2 GB download).
    policy_cfg = PreTrainedConfig.from_pretrained(str(BASE_MODEL))
    policy_cfg.pretrained_path = BASE_MODEL
    policy_cfg.device = str(device)
    policy_cfg.push_to_hub = False
    policy_cfg.train_expert_only = True
    policy_cfg.train_state_proj = True
    policy_cfg.freeze_vision_encoder = True
    policy_cfg.load_vlm_weights = False

    # 2. Load the dataset with the policy's observation/action horizons.
    ds_meta = LeRobotDataset(DATASET_NAME, root=str(DATASET_DIR)).meta
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        DATASET_NAME,
        root=str(DATASET_DIR),
        delta_timestamps=delta_timestamps,
    )
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # 3. Build the policy (loads base weights) and freeze the VLM.
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta, rename_map=RENAME_MAP)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_trainable:,} trainable / {n_total:,} total params")

    # 4. Pre/post processors, re-normalized with THIS dataset's stats.
    stats = ds_meta.stats
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(BASE_MODEL),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "normalizer_processor": {
                "stats": stats,
                "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                "norm_map": policy_cfg.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": stats,
                "features": policy_cfg.output_features,
                "norm_map": policy_cfg.normalization_mapping,
            },
        },
    )

    # 5. Optimizer (trainable params only) + cosine schedule with warmup.
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
    )
    warmup = max(1, int(args.steps * 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Dataloader (single process; Windows-safe with num_workers=0).
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    def cycle(dl):
        while True:
            for batch in dl:
                yield batch

    dl_iter = cycle(dataloader)

    # 7. Training loop.
    policy.train()
    print(f"Training {args.steps} steps (batch_size={args.batch_size}, lr={args.lr}) ...")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = pre(next(dl_iter))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_freq == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"  step {step:5d}/{args.steps} | loss {loss.item():.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

    # 8. Save the fine-tuned checkpoint (policy + processors for eval).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {OUTPUT_DIR} ...")
    policy.save_pretrained(OUTPUT_DIR)
    pre.save_pretrained(OUTPUT_DIR)
    post.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
