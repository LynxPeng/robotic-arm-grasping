"""Panda grasping from a segmented wrist depth camera."""
import math
from importlib.util import find_spec

import numpy as np
import pybullet as p
import pybullet_data

from env import (
    ARM, FINGERS, HOME, LIFT_THRESHOLD, MAX_OPENING, POS_SCALE, STEPS_PER_CTRL, TCP, PandaGraspEnv
)
from qt_rgb_env import OBJECTS

POINTS = 1024
IMAGE_SIZE = 96
NEAR, FAR, FOV = 0.025, 0.8, 70
ACT_DIM = 7
OBS_DIM = POINTS * 3 + 2
ROT_SCALE = math.radians(10)
POINT_SCALE = 0.25
GOAL_SCALE = 0.20
CAMERA_OFFSET = np.array((0.070, 0, -0.026))
EGL_SCENES_PER_CONNECTION = 64


def matrix(quat):
    return np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


def rotvec(quat):
    # q and -q encode the same rotation; take the shorter path.
    q = np.asarray(quat)
    axis, angle = p.getAxisAngleFromQuaternion(q if q[3] >= 0 else -q)
    return np.asarray(axis) * angle


def relative_pose(current, target):
    inv = p.invertTransform(*current)
    pos, quat = p.multiplyTransforms(*inv, *target)
    return np.concatenate((pos, rotvec(quat)))


def goal_label(current, target):
    return (relative_pose(current, target) /
            np.array([GOAL_SCALE] * 3 + [math.pi] * 3)).astype(np.float32)


class PointGraspEnv(PandaGraspEnv):
    def __init__(self, *, renderer='tiny', **kwargs):
        self.renderer = renderer
        self._recycle_egl = renderer == 'egl' and not kwargs.get('gui', False)
        self._egl_scenes = 0
        self.projection = p.computeProjectionMatrixFOV(FOV, 1, NEAR, FAR)
        self.requested_kind = None
        self.max_tilt = 0.0
        self.close_pose = None
        self._cached_goals = None
        self._cached_obj_pose = None
        self._cached_goal_pos = np.empty((0, 3), dtype=np.float64)
        self._cached_goal_quat = np.empty((0, 4), dtype=np.float64)
        super().__init__(reward_mode='sparse', **kwargs)

    def _connect(self, gui):
        client = super()._connect(gui)
        if self.renderer == 'egl':
            try:
                spec = find_spec('eglRenderer')
                if spec is None or not spec.origin:
                    raise RuntimeError('The Linux eglRenderer extension is not installed.')
                if p.loadPlugin(spec.origin, '_eglRendererPlugin', physicsClientId=client) < 0:
                    raise RuntimeError('PyBullet could not initialize eglRenderer.')
            except Exception:
                p.disconnect(client)
                raise
        return client

    def reset(self, seed=None, kind=None):
        self._point_history = []
        self._history_step = -1
        self._record_history = False
        if self._recycle_egl:
            # PyBullet retains EGL assets from rebuilt scenes until disconnect.
            if self._egl_scenes >= EGL_SCENES_PER_CONNECTION:
                self.close()
                self.client = self._connect(gui=False)
                p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
                self._egl_scenes = 0
            self._egl_scenes += 1
        self.requested_kind = kind
        self.max_tilt = 0.0
        self.close_pose = None
        self._cached_goals = None
        self._cached_obj_pose = None
        self.visible_before_contact = self.frames_before_contact = 0
        super().reset(seed)
        self._randomize_initial_pose()
        self.prev_dist = self._dist()
        self._record_history = True
        return self._obs()

    def _randomize_initial_pose(self):
        """Sample target-facing starts 0.2--0.5 m away, subject to reachability."""
        target = self._obj_pos()
        for attempt in range(256):
            distance = self.rng.uniform(0.2, 0.5)
            cosine = self.rng.uniform(0.5, 1.0)
            azimuth, roll = self.rng.uniform(-math.pi, math.pi, 2)
            sine = math.sqrt(1 - cosine * cosine)
            offset = np.array([sine * math.cos(azimuth), sine * math.sin(azimuth), cosine])
            position = target + distance * offset
            if not (0.35 <= position[0] <= 0.70 and -0.30 <= position[1] <= 0.30
                    and 0.10 <= position[2] <= 0.45):
                continue
            forward = -offset
            axis = np.cross([0., 0., 1.], forward)
            axis /= np.linalg.norm(axis)
            orientation = p.getQuaternionFromAxisAngle(axis, math.acos(forward[2]))
            _, orientation = p.multiplyTransforms(
                (0, 0, 0), orientation, (0, 0, 0), p.getQuaternionFromEuler((0, 0, roll)))
            for joint, value in zip(ARM, HOME):
                p.resetJointState(self.robot, joint, value, targetVelocity=0)
            joints = p.calculateInverseKinematics(
                self.robot, TCP, position, orientation, maxNumIterations=300,
                residualThreshold=1e-5, **self._ik)
            if any(not low <= value <= high for value, low, high in
                   zip(joints[:7], self._ik['lowerLimits'], self._ik['upperLimits'])):
                continue
            for joint, value in zip(ARM, joints):
                p.resetJointState(self.robot, joint, value, targetVelocity=0)
            actual, quat = self.pose()
            if np.linalg.norm(np.asarray(actual) - position) > 0.003:
                continue
            if abs(np.dot(quat, orientation)) < math.cos(math.radians(2) / 2):
                continue
            p.performCollisionDetection()
            if p.getClosestPoints(self.robot, self.obj, 0.005):
                continue
            if any(contact[3] >= 0 for contact in
                   p.getClosestPoints(self.robot, self.support, 0.005)):
                continue
            if not self._obs()[-1]:
                continue
            p.setJointMotorControlArray(self.robot, ARM, p.POSITION_CONTROL,
                                       targetPositions=joints[:7], forces=[120] * 7)
            p.setJointMotorControlArray(self.robot, FINGERS, p.POSITION_CONTROL,
                                       targetPositions=[MAX_OPENING / 2] * 2, forces=[30, 30])
            self.down = quat
            self.initial_pose_attempts = attempt + 1
            return
        raise RuntimeError('Could not sample a reachable, visible, collision-free initial pose.')

    def _spawn_object(self, x, y):
        self.kind = self.requested_kind or str(self.rng.choice(list(OBJECTS)))
        width, length, self.height = np.asarray(OBJECTS[self.kind]) * self.rng.uniform(0.92, 1.08)
        yaw = self.rng.uniform(-math.pi, math.pi)
        if self.kind == 'cylinder':
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=width / 2, height=self.height)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=width / 2, length=self.height,
                                         rgbaColor=(0.94, 0.57, 0.10, 1))
        else:
            half = (width / 2, length / 2, self.height / 2)
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=(0.94, 0.57, 0.10, 1))
        return p.createMultiBody(0.055, collision, visual, (x, y, self.height / 2 + 0.005),
                                 p.getQuaternionFromEuler((0, 0, yaw)))

    def pose(self):
        return p.getLinkState(self.robot, TCP, computeForwardKinematics=True)[4:6]

    def _obs(self):
        position, quat = self.pose()
        rotation = matrix(quat)
        eye = position + rotation @ CAMERA_OFFSET
        # Aim the offset camera inward so the target stays visible near contact.
        forward = rotation @ np.array((-0.35, 0, math.sqrt(1 - 0.35**2)))
        up = -rotation[:, 1]
        view = p.computeViewMatrix(eye, eye + forward, up)
        _, _, _, depth, mask = p.getCameraImage(
            IMAGE_SIZE, IMAGE_SIZE, view, self.projection,
            renderer=(p.ER_BULLET_HARDWARE_OPENGL if self.renderer == 'egl' else p.ER_TINY_RENDERER),
            physicsClientId=self.client)
        mask = np.asarray(mask).reshape(IMAGE_SIZE, IMAGE_SIZE)
        rows, cols = np.where((mask >= 0) & ((mask & 0xFFFFFF) == self.obj))
        points = np.zeros((POINTS, 3), dtype=np.float32)
        if rows.size:
            # Deterministic, evenly spaced samples from the visible pixels.
            count = min(POINTS, max(1, math.ceil(0.95 ** self.t * rows.size)))
            idx = np.linspace(0, rows.size - 1, count, dtype=np.int64)
            rows, cols = rows[idx], cols[idx]
            zbuf = np.asarray(depth).reshape(IMAGE_SIZE, IMAGE_SIZE)[rows, cols]
            z = FAR * NEAR / (FAR - (FAR - NEAR) * zbuf)
            scale = 2 * math.tan(math.radians(FOV / 2)) / IMAGE_SIZE
            offset = (IMAGE_SIZE - 1) / 2
            right = np.cross(forward, up)
            rays = (forward + (cols - offset)[:, None] * scale * right
                    + (offset - rows)[:, None] * scale * up)
            world = eye + z[:, None] * rays
            if self._record_history and self._history_step != self.t:
                self._point_history.append(world)
                self._history_step = self.t
            observed = (np.concatenate(self._point_history) if self._record_history
                        and self._point_history else world)
            idx = np.linspace(0, len(observed) - 1, POINTS, dtype=np.int64)
            points[:] = ((observed[idx] - position) @ rotation) / POINT_SCALE
        elif self._record_history and self._point_history:
            observed = np.concatenate(self._point_history)
            idx = np.linspace(0, len(observed) - 1, POINTS, dtype=np.int64)
            points[:] = ((observed[idx] - position) @ rotation) / POINT_SCALE
        return np.concatenate((points.ravel(), np.array(
            [self._opening() / MAX_OPENING, float(bool(rows.size))], dtype=np.float32)))

    def _current_grasp_goals(self):
        obj_pose = p.getBasePositionAndOrientation(self.obj)
        if self._cached_goals is None or self._cached_obj_pose != obj_pose:
            from ga_planner import grasp_poses
            self._cached_goals = grasp_poses(self)
            self._cached_obj_pose = obj_pose
            if self._cached_goals:
                self._cached_goal_pos = np.asarray([g[0] for g in self._cached_goals], dtype=np.float64)
                self._cached_goal_quat = np.asarray([g[1] for g in self._cached_goals], dtype=np.float64)
            else:
                self._cached_goal_pos = np.empty((0, 3), dtype=np.float64)
                self._cached_goal_quat = np.empty((0, 4), dtype=np.float64)
        return self._cached_goal_pos, self._cached_goal_quat

    def _should_retract(self, actual, action, trunc):
        pos = np.asarray(actual[0], dtype=np.float64)
        quat = np.asarray(actual[1], dtype=np.float64)
        goal_pos, goal_quat = self._current_grasp_goals()
        if len(goal_pos):
            pos_dists = np.linalg.norm(goal_pos - pos, axis=1)
            dots = np.clip(np.abs(goal_quat @ quat), 0.0, 1.0)
            rot_dists = 2.0 * np.arccos(dots)
            if np.any((pos_dists <= 0.009) & (rot_dists <= 0.15)):
                return True
            if np.linalg.norm(action[:3]) < 0.18 and np.any((pos_dists <= 0.014) & (rot_dists <= 0.22)):
                return True
        if trunc and np.linalg.norm(pos - self._obj_pos()) <= 0.10:
            return True
        return False

    def retract(self):
        """Scripted finger close and vertical lift (official GA-DDPG panda_scene.retract)."""
        self.close_pose = self.pose()
        close_pos = np.asarray(self.close_pose[0], dtype=np.float64)
        close_quat = self.close_pose[1]
        joints = p.calculateInverseKinematics(
            self.robot, TCP, close_pos, close_quat,
            maxNumIterations=60, residualThreshold=1e-4, **self._ik)
        p.setJointMotorControlArray(self.robot, ARM, p.POSITION_CONTROL,
                                    targetPositions=joints[:7], forces=[120] * 7)
        p.setJointMotorControlArray(self.robot, FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[0.0, 0.0], forces=[35, 35])
        for _ in range(12 * STEPS_PER_CTRL):
            p.stepSimulation()
        for step_idx in range(1, 11):
            lift_pos = close_pos + np.array([0.0, 0.0, 0.16 * (step_idx / 10.0)])
            lift_pos[2] = min(lift_pos[2], 0.45)
            joints = p.calculateInverseKinematics(
                self.robot, TCP, lift_pos, close_quat,
                maxNumIterations=60, residualThreshold=1e-4, **self._ik)
            p.setJointMotorControlArray(self.robot, ARM, p.POSITION_CONTROL,
                                        targetPositions=joints[:7], forces=[120] * 7)
            p.setJointMotorControlArray(self.robot, FINGERS, p.POSITION_CONTROL,
                                        targetPositions=[0.0, 0.0], forces=[35, 35])
            for _ in range(STEPS_PER_CTRL):
                p.stepSimulation()
        left, right, on_support = self._contacts()
        lift = float(self._obj_pos()[2] - self.initial_z)
        grasped = left and right
        success = lift >= LIFT_THRESHOLD and grasped and not on_support
        reward = 1.0 if success else 0.0
        return float(reward), bool(success), bool(grasped), lift

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        pose = self.pose()
        local_rotation = action[3:6] * ROT_SCALE
        angle = np.linalg.norm(local_rotation)
        delta = (p.getQuaternionFromAxisAngle(local_rotation / angle, angle)
                 if angle > 1e-8 else (0., 0., 0., 1.))
        _, self.down = p.multiplyTransforms((0, 0, 0), pose[1], (0, 0, 0), delta)
        translation = matrix(pose[1]) @ action[:3]
        # Policy only controls 6D approach with open fingers; retract() handles closing and lifting.
        obs, reward, term, trunc, info = super().step(np.r_[translation, -1.0])
        actual = self.pose()
        if self.close_pose is None:
            self.frames_before_contact += 1
            self.visible_before_contact += int(obs[-1])
        z_axis = matrix(actual[1])[:, 2]
        tilt = math.degrees(math.acos(np.clip(-z_axis[2], -1, 1)))
        if self.close_pose is None:
            self.max_tilt = max(self.max_tilt, tilt)
        if self._should_retract(actual, action, trunc):
            reward, success, grasped, lift = self.retract()
            term = True
            info.update(success=bool(success), grasped=bool(grasped), lift=lift)
        elif trunc:
            term = True
        grasp_tilt = None
        if self.close_pose is not None:
            close_z = matrix(self.close_pose[1])[:, 2]
            grasp_tilt = math.degrees(math.acos(np.clip(-close_z[2], -1, 1)))
        info.update(kind=self.kind, tilt_at_close_deg=grasp_tilt,
                    max_approach_tilt_deg=self.max_tilt,
                    visible_fraction=self.visible_before_contact / max(1, self.frames_before_contact))
        return obs, reward, term, trunc, info


class ScriptedExpert:
    """Generate candidate approach paths; PointGraspEnv.retract() closes and lifts."""

    def __init__(self, env, tilt_deg):
        pos, quat = p.getBasePositionAndOrientation(env.obj)
        yaw = p.getEulerFromQuaternion(quat)[2]
        if env.kind == 'cylinder':
            yaw = 0.0
        symmetry = math.pi / 2 if env.kind == 'cube' else math.pi
        yaw = (yaw + symmetry / 2) % symmetry - symmetry / 2
        _, target_quat = p.multiplyTransforms(
            (0, 0, 0), p.getQuaternionFromEuler((math.pi, 0, yaw)),
            (0, 0, 0), p.getQuaternionFromEuler((0, math.radians(tilt_deg), 0)))
        grasp = np.asarray(pos) + np.array((0, 0, env.height / 2 - 0.026))
        pregrasp = grasp - matrix(target_quat)[:, 2] * 0.12
        self.targets = [(pregrasp, target_quat), (grasp, target_quat)]
        self.phase = 0

    def action(self, pose):
        delta = relative_pose(pose, self.targets[self.phase])
        if self.phase == 0 and np.linalg.norm(delta[:3]) < 0.010 and np.linalg.norm(delta[3:]) < 0.14:
            self.phase = 1
        delta = relative_pose(pose, self.targets[self.phase])
        return np.clip(np.r_[delta[:3] / POS_SCALE, delta[3:] / ROT_SCALE, -1.],
                       -1, 1).astype(np.float32)
