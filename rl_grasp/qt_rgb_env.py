"""RGB observations and top-down yaw control for three Panda grasp objects."""

import math
import numpy as np
import pybullet as p

from env import PandaGraspEnv, TCP, MAX_OPENING, POS_SCALE

SIZE = 224
ACT_DIM = 5
AUX_DIM = 4
YAW_SCALE = math.radians(15)
OBJECTS = {
    'cylinder': (0.052, 0.052, 0.045),
    'cube': (0.047, 0.047, 0.045),
    'long_box': (0.105, 0.038, 0.045),
}


class RGBGraspEnv(PandaGraspEnv):
    def reset(self, seed=None, kind=None):
        self.requested_kind = kind
        self.yaw = 0.0
        self.view = p.computeViewMatrix((0.51, -0.40, 0.61), (0.55, 0.015, 0.085), (0, 0, 1))
        self.projection = p.computeProjectionMatrixFOV(45, 1, 0.05, 1.5)
        if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
            p.resetDebugVisualizerCamera(0.88, 48, -33, (0.49, 0.01, 0.15))
        return super().reset(seed)

    def _spawn_object(self, x, y):
        self.kind = self.requested_kind or str(self.rng.choice(list(OBJECTS)))
        yaw = self.rng.uniform(-math.pi, math.pi)
        dimensions = OBJECTS[self.kind]
        self.height = dimensions[2]
        color = (0.94, 0.57, 0.10, 1)
        if self.kind == 'cylinder':
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=dimensions[0] / 2, height=self.height)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=dimensions[0] / 2, length=self.height,
                                         rgbaColor=color)
        else:
            half = np.array(dimensions) / 2
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return p.createMultiBody(0.055, collision, visual, (x, y, self.height / 2 + 0.005),
                                 p.getQuaternionFromEuler((0, 0, yaw)))

    def _obs(self):
        position, rotation = p.getLinkState(self.robot, TCP, computeForwardKinematics=True)[4:6]
        yaw = p.getEulerFromQuaternion(rotation)[2]
        aux = np.array([self._opening() / MAX_OPENING, position[2] * 4,
                        math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        _, _, rgba, _, _ = p.getCameraImage(SIZE, SIZE, self.view, self.projection,
                                           renderer=p.ER_TINY_RENDERER,
                                           flags=p.ER_NO_SEGMENTATION_MASK)
        rgb = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8).reshape(SIZE, SIZE, 4)[:, :, :3])
        return rgb, aux

    def step(self, action):
        self.yaw = float(np.clip(self.yaw + np.clip(action[4], -1, 1) * YAW_SCALE,
                                 -math.pi / 2, math.pi / 2))
        self.down = p.getQuaternionFromEuler((math.pi, 0, self.yaw))
        obs, reward, term, trunc, info = super().step(action[:4])
        info['kind'] = self.kind
        return obs, reward, term, trunc, info


def expert(env, phase, wait):
    """Privileged script used only to collect the initial grasp transitions."""
    tcp = env._tcp()
    obj = env._obj_pos()
    grasp = env._grasp_target()
    object_rotation = p.getBasePositionAndOrientation(env.obj)[1]
    object_yaw = p.getEulerFromQuaternion(object_rotation)[2]
    symmetry = math.pi / 2 if env.kind == 'cube' else math.pi
    yaw = (object_yaw + symmetry / 2) % symmetry - symmetry / 2 if env.kind != 'cylinder' else 0.0
    # Freeze the yaw command during and after closing so contact does not turn into tracking.
    delta_yaw = (yaw - env.yaw) / YAW_SCALE if phase < 2 else 0.0
    if phase == 0:
        target = np.array((obj[0], obj[1], 0.20))
        grip = -1.
        if np.linalg.norm(tcp - target) < 0.016 and abs(delta_yaw) < 0.15:
            phase = 1
    elif phase == 1:
        target, grip = grasp, -1.
        if np.linalg.norm(tcp - target) < 0.008:
            phase, wait = 2, 0
    elif phase == 2:
        target, grip = grasp, 1.
        wait += 1
        if wait >= 12:
            phase = 3
    else:
        target, grip = np.array((obj[0], obj[1], 0.21)), 1.
    action = np.clip([*((target - tcp) / POS_SCALE), grip, delta_yaw], -1, 1)
    return action.astype(np.float32), phase, wait
