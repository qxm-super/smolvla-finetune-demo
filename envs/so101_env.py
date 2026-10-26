"""MuJoCo SO-ARM100 pick-and-place environment matching SmolVLA's interface.

The SmolVLA policy (``ahmedsohail2003/smolvla-so101-pickplace``) was trained on a
6-dim *radian* joint vector ordered:

    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, jaw]

with ``jaw`` in radians too (0 = closed, ~1.5 = open). This environment wraps an
approximate SO-ARM100 MuJoCo scene (``assets/so101_scene.xml``) and exposes that
exact observation/action convention so the policy can be run closed-loop.

The scene is *not* a perfect replica of the robot the policy was trained on, so a
few boundaries are adapted here and documented:

* **Elbow sign.** The scene's ``elbow_flex`` joint folds negative (range
  ``[-pi, 0]``) while the real SO-ARM100 folds positive (home pose is ``+pi/2``).
  The elbow is therefore sign-flipped when translating between the two.
* **Gripper.** The policy's ``jaw`` is in radians; the scene's fingers are slide
  joints in meters (0 = closed, 0.04 = open). A linear map converts between them.
* **Cameras.** The policy's ``front``/``wrist`` views are mapped onto the scene's
  ``webcam`` (overhead) and ``arm_cam`` (wrist) cameras respectively.

These domain gaps mean the transferred policy is not expected to match its
training-time success rate; the point of this environment is to demonstrate the
full closed-loop VLA inference pipeline in physics.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
SCENE_XML = ASSETS_DIR / "so101_scene.xml"

# Joint order matches the SmolVLA dataset exactly.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# SmolVLA camera name -> scene camera name.
CAMERA_MAP = {"front": "webcam", "wrist": "arm_cam"}

# Gripper: policy jaw is rad (0 closed -> ~1.5 open), scene fingers are meters
# (0 closed -> 0.04 open).
JAW_OPEN_RAD = 1.5
FINGER_OPEN_M = 0.04
_JAW_TO_FINGER = FINGER_OPEN_M / JAW_OPEN_RAD
_FINGER_TO_JAW = JAW_OPEN_RAD / FINGER_OPEN_M

# Home pose (5 arm joints, rad, *scene convention*): arm raised out of the workspace
# so the scripted expert can reach the block. The elbow sign is flipped on the
# observation/action boundary, so this reads as ``+0.3`` in the model convention.
HOME_ARM_QPOS = np.array([0.0, 0.0, -0.3, -0.3, 0.0])
HOME_JAW_RAD = 1.4

# Grasp heuristics (mirrors the scene's snap-grasp design).
GRASP_EE_DIST = 0.06       # EE-to-box distance that can trigger a grasp (m)
GRASP_JAW_CLOSED = 0.675   # jaw (rad) below which the fingers are treated as closed
GRASP_JAW_CLOSING = 0.4    # commanded jaw (rad) below which the policy is "closing"
GRASP_JAW_OPENING = 0.8    # commanded jaw (rad) above which the policy is "opening"

# Success criterion (box inside bowl), matching the scene geometry.
BOWL_RADIUS = 0.05
BOWL_HEIGHT = 0.04


class SO101PickPlace:
    """Closed-loop SO-ARM100 pick-and-place in MuJoCo, SmolVLA-compatible."""

    def __init__(self, width: int = 224, height: int = 224, control_freq: int = 50):
        self.width = width
        self.height = height
        self.control_freq = control_freq

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        # 10 physics substeps per control step (scene timestep is 0.002 s).
        self._substeps = max(1, int(round(1.0 / control_freq / self.model.opt.timestep)))

        # MuJoCo IDs.
        self._arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS
        ]
        self._finger_left = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left"
        )
        self._box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._bowl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._box_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self._box_qpos_adr = self.model.jnt_qposadr[self._box_jnt]
        self._box_dof_adr = self.model.jnt_dofadr[self._box_jnt]

        self._ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Offscreen renderer (only for the two observation cameras).
        self._renderer = mujoco.Renderer(self.model, height, width)

        # Snap-grasp state.
        self._grasped = False
        self._grasp_offset = np.zeros(3)

        # Initial state captured after a forward pass.
        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        self._step_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, box_pos: np.ndarray | None = None) -> None:
        rng = np.random.default_rng(seed)

        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        self.data.ctrl[:] = 0.0

        # Arm to home pose.
        for jid, q in zip(self._arm_joint_ids, HOME_ARM_QPOS):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q

        # Gripper open.
        self._set_finger_qpos(self._jaw_to_finger(HOME_JAW_RAD))

        # Box at its default (or caller-specified) pose; orientation reset, velocity zeroed.
        box_xy = [0.15, 0.0] if box_pos is None else np.asarray(box_pos[:2], dtype=np.float64)
        self.data.qpos[self._box_qpos_adr : self._box_qpos_adr + 3] = [box_xy[0], box_xy[1], 0.44]
        self.data.qpos[self._box_qpos_adr + 3 : self._box_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[self._box_dof_adr : self._box_dof_adr + 6] = 0.0

        self._grasped = False
        self._grasp_offset = np.zeros(3)

        # Match actuator controls to the reset pose.
        for i, jid in enumerate(self._arm_joint_ids):
            self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        finger = self._jaw_to_finger(HOME_JAW_RAD)
        self.data.ctrl[5] = finger
        self.data.ctrl[6] = finger

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0

    def step(self, action: np.ndarray) -> dict:
        """Apply a 6-dim rad joint target and advance the simulation one step.

        ``action = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, jaw]``
        in radians (absolute targets).
        """
        action = np.asarray(action, dtype=np.float64).reshape(6)

        # Arm joints: direct targets (elbow sign-flipped on the boundary).
        for i, jid in enumerate(self._arm_joint_ids):
            target = action[i]
            if ARM_JOINTS[i] == "elbow_flex":
                target = -target
            lo, hi = self._ctrl_ranges[i]
            self.data.ctrl[i] = float(np.clip(target, lo, hi))

        # Jaw (rad) -> finger separation (m), clamped to the slide range.
        finger = float(np.clip(self._jaw_to_finger(action[5]), 0.0, FINGER_OPEN_M))
        self.data.ctrl[5] = finger
        self.data.ctrl[6] = finger

        self._update_grasp(action[5])

        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        return self._info()

    def observe(self) -> dict:
        """Return the SmolVLA-style observation (state + raw camera images)."""
        return {
            "state": self._state_rad(),
            "images": {
                name: self._render_camera(scene_cam)
                for name, scene_cam in CAMERA_MAP.items()
            },
        }

    def render(self, view: str = "front") -> np.ndarray:
        """Render one camera view as an ``(H, W, 3)`` uint8 RGB array."""
        return self._render_camera(CAMERA_MAP[view])

    def close(self) -> None:
        self._renderer.close()

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site].copy()

    def get_box_pos(self) -> np.ndarray:
        return self.data.xpos[self._box_body].copy()

    def get_bowl_pos(self) -> np.ndarray:
        return self.data.xpos[self._bowl_body].copy()

    # ------------------------------------------------------------------
    # State / info helpers
    # ------------------------------------------------------------------

    def _state_rad(self) -> np.ndarray:
        state = np.zeros(6, dtype=np.float64)
        for i, jid in enumerate(self._arm_joint_ids):
            q = self.data.qpos[self.model.jnt_qposadr[jid]]
            state[i] = -q if ARM_JOINTS[i] == "elbow_flex" else q
        state[5] = self._finger_to_jaw(self._finger_qpos())
        return state

    def _finger_qpos(self) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self._finger_left]])

    @staticmethod
    def _jaw_to_finger(jaw: float) -> float:
        return jaw * _JAW_TO_FINGER

    @staticmethod
    def _finger_to_jaw(finger: float) -> float:
        return finger * _FINGER_TO_JAW

    def _set_finger_qpos(self, finger: float) -> None:
        self.data.qpos[self.model.jnt_qposadr[self._finger_left]] = finger
        # The two fingers are coupled via an equality constraint, so setting the
        # left finger's qpos is enough for the visual state.
        right = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right")
        self.data.qpos[self.model.jnt_qposadr[right]] = finger

    def _render_camera(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()  # (H, W, 3) uint8 RGB

    def _update_grasp(self, jaw_action: float) -> None:
        """Kinematic snap-grasp: attach the box to the EE when the gripper closes."""
        ee_pos = self.data.site_xpos[self._ee_site].copy()
        box_pos = self.data.xpos[self._box_body].copy()
        dist = float(np.linalg.norm(ee_pos - box_pos))

        finger_qpos = self._finger_qpos()
        closing = jaw_action < GRASP_JAW_CLOSING
        opening = jaw_action > GRASP_JAW_OPENING

        if (
            not self._grasped
            and closing
            and finger_qpos < GRASP_JAW_CLOSED * _JAW_TO_FINGER
            and dist < GRASP_EE_DIST
        ):
            self._grasp_offset = box_pos - ee_pos
            self._grasped = True
        elif self._grasped and opening:
            self._grasped = False

        if self._grasped:
            new_box_pos = ee_pos + self._grasp_offset
            self.data.qpos[self._box_qpos_adr : self._box_qpos_adr + 3] = new_box_pos
            self.data.qvel[self._box_dof_adr : self._box_dof_adr + 6] = 0.0

    def _info(self) -> dict:
        box_pos = self.data.xpos[self._box_body].copy()
        bowl_pos = self.data.xpos[self._bowl_body].copy()
        ee_pos = self.data.site_xpos[self._ee_site].copy()

        horizontal_dist = float(np.linalg.norm(box_pos[:2] - bowl_pos[:2]))
        above_base = bool(box_pos[2] > bowl_pos[2] - 0.01)
        below_rim = bool(box_pos[2] < bowl_pos[2] + BOWL_HEIGHT)
        success = horizontal_dist < BOWL_RADIUS and above_base and below_rim

        return {
            "success": success,
            "box_pos": box_pos,
            "bowl_pos": bowl_pos,
            "ee_pos": ee_pos,
            "ee_to_box_dist": float(np.linalg.norm(ee_pos - box_pos)),
            "box_to_bowl_dist": horizontal_dist,
        }
