"""Collect scripted pick-and-place demonstrations as a LeRobot v3.0 dataset.

Runs a waypoint + Jacobian-IK expert controller in the MuJoCo SO-ARM100 scene and
records each step (6-dim rad state, 6-dim rad action, front + wrist cameras, task)
into a LeRobot dataset that SmolVLA can be fine-tuned on.

Usage:
    python scripts/collect_demos.py --smoke          # one episode, no dataset (sanity check)
    python scripts/collect_demos.py --episodes 20    # write a real dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.so101_env import SO101PickPlace, ARM_JOINTS  # noqa: E402

TASK = "Pick up the red block and place it in the blue tray."
DATASET_NAME = "smolvla_so101_demos"

# World-frame EE waypoints (z heights tuned for the scene in assets/so101_scene.xml).
SAFE_Z = 0.62       # above the workspace, safe for horizontal travel
PRE_GRASP_Z = 0.46  # just above the block
GRASP_Z = 0.434     # EE at block centre, finger pads wrap the block sides
PLACE_Z = 0.48      # inside the bowl rim, then open the gripper

JAW_OPEN = 1.4
JAW_CLOSED = 0.0

# Max EE displacement per control step (m). Rate-limiting keeps each demo smooth and
# long enough (> SmolVLA's chunk_size of 50) for the training sampler to find valid
# (observation, action-chunk) windows.
STEP_SIZE = 0.008

MAX_STEPS = 300


class ScriptedController:
    """Waypoint pick-and-place expert, outputting 6-dim rad actions.

    Joint order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, jaw],
    matching SmolVLA. IK is solved on a shadow MjData (scene convention) and the elbow
    is sign-flipped into the model convention at the boundary.
    """

    def __init__(self, env: SO101PickPlace):
        self.env = env
        self.model = env.model
        self.data = env.data
        self._ik_data = mujoco.MjData(self.model)

        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS
        ]
        self._dof_ids = [self.model.jnt_dofadr[jid] for jid in self._joint_ids]
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self._phase = 0
        self._phase_step = 0
        self._elbow_idx = ARM_JOINTS.index("elbow_flex")

    def reset(self) -> None:
        self._phase = 0
        self._phase_step = 0

    def select_action(self) -> np.ndarray:
        box = self.env.get_box_pos()
        bowl = self.env.get_bowl_pos()
        ee = self.env.get_ee_pos()

        if self._phase == 0:  # above block
            action = self._move([box[0], box[1], SAFE_Z], JAW_OPEN, ee, 0.03)
        elif self._phase == 1:  # pre-grasp
            action = self._move([box[0], box[1], PRE_GRASP_Z], JAW_OPEN, ee, 0.03)
        elif self._phase == 2:  # grasp
            action = self._move([box[0], box[1], GRASP_Z], JAW_OPEN, ee, 0.02)
        elif self._phase == 3:  # close gripper
            action = self._hold(JAW_CLOSED, wait_steps=25)
        elif self._phase == 4:  # lift
            action = self._move([box[0], box[1], SAFE_Z], JAW_CLOSED, ee, 0.03)
        elif self._phase == 5:  # above bowl
            action = self._move([bowl[0], bowl[1], SAFE_Z], JAW_CLOSED, ee, 0.03)
        elif self._phase == 6:  # into bowl
            action = self._move([bowl[0], bowl[1], PLACE_Z], JAW_CLOSED, ee, 0.02)
        elif self._phase == 7:  # open gripper
            action = self._hold(JAW_OPEN, wait_steps=15)
        elif self._phase == 8:  # retreat up
            action = self._move([bowl[0], bowl[1], SAFE_Z], JAW_OPEN, ee, 0.03)
        else:  # done — hold
            action = self._hold(JAW_OPEN, wait_steps=10**9)

        return action

    # ------------------------------------------------------------------
    def _move(self, target_pos, jaw, ee, threshold) -> np.ndarray:
        delta = target_pos - ee
        dist = float(np.linalg.norm(delta))
        # Rate-limit the EE toward the waypoint so the demo is smooth and long enough
        # (> chunk_size) for the training sampler.
        if dist > STEP_SIZE:
            interp_target = ee + delta / dist * STEP_SIZE
        else:
            interp_target = target_pos
        targets = self._ik_to(interp_target)
        self._phase_step += 1
        if dist < threshold or self._phase_step > 250:
            self._advance()
        return self._action(targets, jaw)

    def _hold(self, jaw, wait_steps: int) -> np.ndarray:
        targets = np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids]
        )
        self._phase_step += 1
        if self._phase_step > wait_steps:
            self._advance()
        return self._action(targets, jaw)

    def _advance(self) -> None:
        self._phase += 1
        self._phase_step = 0

    def _ik_to(self, target_pos) -> np.ndarray:
        """Damped-least-squares Jacobian IK on shadow data; returns 5 scene joints."""
        self._ik_data.qpos[:] = self.data.qpos[:]
        self._ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._ik_data)

        for _ in range(40):
            shadow_ee = self._ik_data.site_xpos[self._ee_site]
            error = target_pos - shadow_ee
            if np.linalg.norm(error) < 0.001:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self._ik_data, jacp, None, self._ee_site)
            J = jacp[:, self._dof_ids]
            lam = 0.05
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), error)

            for i, jid in enumerate(self._joint_ids):
                self._ik_data.qpos[self.model.jnt_qposadr[jid]] += dq[i] * 0.4
            mujoco.mj_forward(self.model, self._ik_data)

        return np.array(
            [self._ik_data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids]
        )

    def _action(self, scene_targets_5: np.ndarray, jaw: float) -> np.ndarray:
        """Build a 6-dim model-convention action from scene-convention joint targets."""
        action = np.zeros(6, dtype=np.float32)
        action[:5] = scene_targets_5
        action[self._elbow_idx] = -scene_targets_5[self._elbow_idx]  # elbow sign flip
        action[5] = jaw
        return action


def build_features() -> dict:
    names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.front": {
            "dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"],
        },
    }


def run_smoke() -> None:
    env = SO101PickPlace(width=224, height=224, control_freq=20)
    controller = ScriptedController(env)
    env.reset(seed=0)
    controller.reset()

    for step in range(MAX_STEPS):
        obs = env.observe()
        action = controller.select_action()
        info = env.step(action)
        if step % 25 == 0:
            print(
                f"  step {step:3d} | phase-ish ee={np.round(info['ee_pos'], 3)} "
                f"box={np.round(info['box_pos'], 3)} dist={info['ee_to_box_dist']:.3f}"
            )
        if info["success"]:
            print(f"  SUCCESS at step {step}")
            break
    else:
        print("  (reached max steps without success)")

    print("final info:", {k: (np.round(v, 3) if isinstance(v, np.ndarray) else v)
                          for k, v in info.items()})
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="one episode, no dataset written")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = REPO_ROOT / "datasets" / DATASET_NAME
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
        print(f"Removed existing dataset at {dataset_root}")

    dataset = LeRobotDataset.create(
        repo_id=DATASET_NAME,
        root=str(dataset_root),
        fps=args.fps,
        features=build_features(),
        robot_type="so101",
        use_videos=True,
    )

    env = SO101PickPlace(width=224, height=224, control_freq=args.fps)
    controller = ScriptedController(env)

    successes = 0
    rng = np.random.default_rng(0)

    print(f"Collecting {args.episodes} episodes at {args.fps} fps ...")
    for ep in range(args.episodes):
        # Randomize the block position for robustness (±2 cm in x/y).
        box_xy = np.array([0.15, 0.0]) + rng.uniform(-0.02, 0.02, size=2)
        env.reset(seed=ep, box_pos=box_xy)
        controller.reset()

        for step in range(MAX_STEPS):
            obs = env.observe()
            action = controller.select_action()
            dataset.add_frame({
                "observation.state": obs["state"].astype(np.float32),
                "action": action.astype(np.float32),
                "observation.images.front": obs["images"]["front"],
                "observation.images.wrist": obs["images"]["wrist"],
                "task": TASK,
            })
            info = env.step(action)
            if info["success"]:
                successes += 1
                break

        dataset.save_episode()
        print(f"  ep {ep + 1}: {'success' if info['success'] else 'FAIL'} "
              f"({step + 1} steps)")

    dataset.finalize()
    env.close()
    print(f"\nDone: {successes}/{args.episodes} successful episodes.")
    print(f"Dataset saved to {dataset_root}")


if __name__ == "__main__":
    main()
