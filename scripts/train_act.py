"""Train a LeRobot ACT policy on the scripted SO-101 demos collected in this scene.

Trains Action Chunking with Transformers (ACT) from scratch on
``datasets/smolvla_so101_demos`` using LeRobot's ACT implementation, then saves the
policy + pre/post processors so it can be rolled out closed-loop in the MuJoCo scene.

Usage:
    python scripts/train_act.py --steps 3000 --batch-size 8
    python scripts/train_act.py --steps 3000 --pretrained-backbone   # ImageNet ResNet18
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

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

DATASET_NAME = "smolvla_so101_demos"
DATASET_DIR = REPO_ROOT / "datasets" / DATASET_NAME
OUTPUT_DIR = REPO_ROOT / "outputs" / "train" / "act_so101"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="use ImageNet-pretrained ResNet18 (downloads ~45 MB on first run)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # 1. ACT config (defaults target bimanual ALOHA; only device/backbone override).
    cfg = ACTConfig()
    cfg.device = str(device)
    cfg.pretrained_backbone_weights = (
        "ResNet18_Weights.IMAGENET1K_V1" if args.pretrained_backbone else None
    )
    cfg.optimizer_lr = args.lr
    cfg.optimizer_lr_backbone = args.lr

    # 2. Load the dataset with ACT's observation/action horizons (chunk_size=100).
    ds_meta = LeRobotDataset(DATASET_NAME, root=str(DATASET_DIR)).meta
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(
        DATASET_NAME,
        root=str(DATASET_DIR),
        delta_timestamps=delta_timestamps,
    )
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # 3. Build the policy from scratch (make_policy infers input/output features) and
    #    its processors, normalizing with THIS dataset's stats.
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_total:,} params | inputs={list(cfg.input_features)}")

    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)

    # 4. Optimizer (backbone + transformer at the same lr, from scratch) + cosine warmup.
    param_groups = policy.get_optim_params()
    for group in param_groups:
        group["lr"] = args.lr
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4
    )
    warmup = max(1, int(args.steps * 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Dataloader (single process; Windows-safe with num_workers=0).
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

    # 6. Training loop (fp32; ACT is small enough that bf16 isn't needed and the VAE
    #    KL term is numerically sensitive).
    policy.train()
    print(f"Training {args.steps} steps (batch_size={args.batch_size}, lr={args.lr}) ...")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = pre(next(dl_iter))
        loss, loss_dict = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_freq == 0 or step == 1:
            elapsed = time.time() - t0
            l1 = loss_dict.get("l1_loss", float("nan"))
            kld = loss_dict.get("kld_loss", float("nan"))
            print(
                f"  step {step:5d}/{args.steps} | loss {loss.item():.4f} "
                f"(l1 {l1:.4f}, kld {kld:.4f}) | lr {optimizer.param_groups[0]['lr']:.2e} "
                f"| {elapsed:.1f}s"
            )

    # 7. Save the checkpoint (policy + processors for eval).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {OUTPUT_DIR} ...")
    policy.save_pretrained(OUTPUT_DIR)
    pre.save_pretrained(OUTPUT_DIR)
    post.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
