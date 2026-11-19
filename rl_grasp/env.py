"""Minimal closed-loop PyBullet grasp environment (Franka Panda + cylinder).

Scene and success criteria follow the project's pybullet_grasp.py:
fixed-base Panda, 55 g cylinder (r=26 mm, h=45 mm), top-down grasp,
success = lift >= 0.10 m with both finger contacts and no table support.

State-based observations (privileged sim state) and end-effector delta
actions keep this minimal version small and fast; swap in RGB-D later.
No gym/gymnasium dependency: plain reset()/step() API.
"""
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "robotic-grasping" / "results" / "rl-grasp"

ARM = tuple(range(7))
FINGERS = (9, 10)
TCP = 11
HOME = (0, -0.5, 0, -2.4, 0, 2.0, 0.8)
OBJ_RADIUS = 0.026
OBJ_HEIGHT = 0.045
OBJ_MASS = 0.055
MAX_OPENING = 0.08          # full gripper opening, m (Panda limit)
POS_SCALE = 0.02            # m per control step at |action| = 1
STEPS_PER_CTRL = 10         # physics steps per env step (240 Hz -> 24 Hz)
MAX_CTRL_STEPS = 200
LIFT_THRESHOLD = 0.10
OBJ_X_RANGE = (0.52, 0.60)
OBJ_Y_RANGE = (-0.04, 0.08)
OBS_DIM = 11
ACT_DIM = 4


class PandaGraspEnv:
    """ obs (11): tcp_xyz(3), opening_frac(1), obj_rel_xyz(3), lift(1),
                  left_contact(1), right_contact(1), on_support(1)
        act (4):  [dx, dy, dz, grip] in [-1, 1]; xyz scaled by POS_SCALE,
                  grip=+1 closes, grip=-1 fully opens.
        Orientation is fixed pointing down (cylinder is yaw-symmetric).
    """

    def __init__(self, gui=False, reward_mode="shaped", max_steps=MAX_CTRL_STEPS, seed=None):
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.client = self._connect(gui)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.down = p.getQuaternionFromEuler((math.pi, 0, 0))
        self.reset()

    def _connect(self, gui):
        return p.connect(p.GUI if gui else p.DIRECT)

    # -- world ----------------------------------------------------------
    def _build(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
        self.support = p.loadURDF("plane.urdf")
        x = float(self.rng.uniform(*OBJ_X_RANGE))
        y = float(self.rng.uniform(*OBJ_Y_RANGE))
        self.obj = self._spawn_object(x, y)
        p.changeDynamics(self.obj, -1, lateralFriction=1.2)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for j, q in enumerate(HOME):
            p.resetJointState(self.robot, j, q)
        for j in FINGERS:
            p.resetJointState(self.robot, j, MAX_OPENING / 2)
            p.changeDynamics(self.robot, j, lateralFriction=1.5)
        limits = [p.getJointInfo(self.robot, j)[8:10] for j in (*ARM, *FINGERS)]
        lo, hi = zip(*limits)
        self._ik = dict(lowerLimits=lo, upperLimits=hi,
                        jointRanges=[h - l for l, h in limits],
                        restPoses=(*HOME, 0.04, 0.04))
        for _ in range(60):
            p.stepSimulation()
        self.initial_z = p.getBasePositionAndOrientation(self.obj)[0][2]

    def _spawn_object(self, x, y):
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=OBJ_RADIUS, height=OBJ_HEIGHT)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=OBJ_RADIUS, length=OBJ_HEIGHT,
                                     rgbaColor=(0.95, 0.56, 0.09, 1))
        return p.createMultiBody(OBJ_MASS, collision, visual,
                                     (x, y, OBJ_HEIGHT / 2 + 0.005))

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._build()
        self.t = 0
        self.grasp_bonus_given = False
        self.prev_dist = self._dist()
        self.prev_lift = 0.0
        return self._obs()

    # -- state helpers ----------------------------------------------------
    def _tcp(self):
        return np.array(p.getLinkState(self.robot, TCP, computeForwardKinematics=True)[4])

    def _obj_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.obj)[0])

    def _contacts(self):
        fingers = {c[3] for c in p.getContactPoints(self.robot, self.obj)
                   if c[3] in FINGERS and c[9] > 0}
        on_support = any(c[9] > 0 for c in p.getContactPoints(self.obj, self.support))
        return FINGERS[0] in fingers, FINGERS[1] in fingers, on_support

    def _grasp_target(self):
        o = self._obj_pos()
        # Same finger-pad offset as pybullet_grasp.py (surface z - 26 mm).
        return np.array((o[0], o[1], o[2] + OBJ_HEIGHT / 2 - 0.026))

    def _dist(self):
        return float(np.linalg.norm(self._tcp() - self._grasp_target()))

    def _opening(self):
        return float(sum(p.getJointState(self.robot, j)[0] for j in FINGERS))

    def _obs(self):
        tcp, obj = self._tcp(), self._obj_pos()
        left, right, on_support = self._contacts()
        lift = obj[2] - self.initial_z
        return np.array([*tcp, self._opening() / MAX_OPENING,
                         *(obj - tcp), lift, float(left), float(right), float(on_support)],
                        dtype=np.float32)

    # -- control ----------------------------------------------------------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        tcp = self._tcp()
        target = tcp + a[:3] * POS_SCALE
        target[0] = np.clip(target[0], 0.35, 0.70)
        target[1] = np.clip(target[1], -0.30, 0.30)
        target[2] = np.clip(target[2], 0.005, 0.45)
        joints = p.calculateInverseKinematics(
            self.robot, TCP, target, self.down,
            maxNumIterations=60, residualThreshold=1e-4, **self._ik)
        p.setJointMotorControlArray(self.robot, ARM, p.POSITION_CONTROL,
                                    targetPositions=joints[:7], forces=[120] * 7)
        per_finger = MAX_OPENING * (1.0 - a[3]) / 4.0
        p.setJointMotorControlArray(self.robot, FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[per_finger] * 2, forces=[30, 30])
        for _ in range(STEPS_PER_CTRL):
            p.stepSimulation()
        self.t += 1

        obs = self._obs()
        left, right, on_support = self._contacts()
        lift = float(self._obj_pos()[2] - self.initial_z)
        grasped = left and right
        success = lift >= LIFT_THRESHOLD and grasped and not on_support
        dist = self._dist()
        if self.reward_mode == "sparse":
            reward = 1.0 if success else 0.0
        else:
            reward = 2.0 * (self.prev_dist - dist)          # reach progress
            if grasped and not self.grasp_bonus_given:
                reward += 1.0                               # first firm grasp
                self.grasp_bonus_given = True
            if grasped:
                reward += 20.0 * max(0.0, lift - self.prev_lift)  # lift progress
            if success:
                reward += 10.0
        self.prev_dist, self.prev_lift = dist, lift
        info = {"success": bool(success), "lift": lift, "grasped": grasped,
                "dist": dist, "steps": self.t}
        return obs, float(reward), bool(success), self.t >= self.max_steps, info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
