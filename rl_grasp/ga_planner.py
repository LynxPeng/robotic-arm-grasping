"""Real-time OMG-style grasp goal selection, nearest-goal supervision and trajectory optimizer.

Implements the three components required by Goal-Auxiliary DDPG (GA-DDPG):
* a per-object 6D grasp goal set ``G`` (from QDGset annotations when available,
  otherwise sampled analytically for primitives or mesh AABBs) with parallel-jaw
  180-deg symmetry;
* nearest-goal supervision ``g_t = argmin_{g in G} L_POSE(g, T_t)`` expressed as a
  relative pose for every online transition;
* a real-time receding-horizon trajectory optimizer starting from the live gripper
  pose ``T_t`` that selects an IK/collision-validated grasp goal and computes a
  smooth, collision-free approach velocity toward ``cur_goal`` (with gripper closing
  and vertical lifting executed automatically by ``PointGraspEnv.retract()``).
"""
import math

import numpy as np
import pybullet as p

from env import ARM, TCP, POS_SCALE
from ga_env import matrix, relative_pose, goal_label, ROT_SCALE

# Virtual gripper keypoints shared with the BC/aux pose loss in ga_model.Learner.
KEYPOINTS = np.array([
    [0., 0., 0.], [0., 0.04, 0.], [0., -0.04, 0.],
    [0., 0.04, 0.05], [0., -0.04, 0.05],
], dtype=np.float64)
HAND_TO_TCP = (0., 0., 0.105)


def _grasp_quat(yaw, tilt_deg):
    _, quat = p.multiplyTransforms(
        (0, 0, 0), p.getQuaternionFromEuler((math.pi, 0, yaw)),
        (0, 0, 0), p.getQuaternionFromEuler((0, math.radians(tilt_deg), 0)))
    return quat


def _mesh_grasps(env):
    grasps = getattr(getattr(env, 'asset', None), 'grasps', None)
    if grasps is None or not len(grasps):
        return None
    world_inertia = p.getBasePositionAndOrientation(env.obj)
    low, high = np.asarray(p.getAABB(env.obj))
    order = np.argsort(-np.asarray(grasps)[:, 7], kind='stable')
    poses = []
    for index in order:
        grasp = np.asarray(grasps[index])
        position, orientation = p.multiplyTransforms(
            *world_inertia, grasp[:3], grasp[3:7])
        position, orientation = p.multiplyTransforms(
            position, orientation, HAND_TO_TCP, (0, 0, 0, 1))
        forward = matrix(orientation)[:, 2]
        pregrasp = np.asarray(position) - 0.08 * forward
        if (forward[2] > -0.35 or position[2] < 0.008
                or np.any(position < low - 0.025) or np.any(position > high + 0.025)
                or not (0.36 < pregrasp[0] < 0.69 and abs(pregrasp[1]) < 0.28)
                or position[2] + 0.16 > 0.45):
            continue
        poses.append((np.asarray(position, dtype=np.float64), orientation))
        if len(poses) == 64:
            break
    return poses or None


def _with_symmetry(poses):
    """Include both parallel-jaw finger orientations (180-deg Z flip) for each grasp."""
    expanded = []
    for position, orientation in poses:
        pos = np.asarray(position, dtype=np.float64)
        expanded.append((pos, orientation))
        _, flipped = p.multiplyTransforms((0, 0, 0), orientation, (0, 0, 0), (0, 0, 1, 0))
        expanded.append((pos.copy(), flipped))
    return expanded


def grasp_poses(env):
    """World-frame candidate grasp poses ``(position, quaternion)`` for the object."""
    annotated = _mesh_grasps(env)
    if annotated:
        return _with_symmetry(annotated)
    pos, quat = p.getBasePositionAndOrientation(env.obj)
    pos = np.asarray(pos, dtype=np.float64)
    yaw = p.getEulerFromQuaternion(quat)[2]
    height = getattr(env, 'height', None)
    if height is None:
        low, high = np.asarray(p.getAABB(env.obj))
        height = float(high[2] - low[2])
        pos = np.array([pos[0], pos[1], (low[2] + high[2]) / 2])
    kind = getattr(env, 'kind', '')
    if kind == 'cylinder':
        base_yaw, symmetry = 0.0, math.pi
    else:
        symmetry = math.pi / 2 if kind == 'cube' else math.pi
        base_yaw = (yaw + symmetry / 2) % symmetry - symmetry / 2
    grasp = pos + np.array((0, 0, height / 2 - 0.026))
    poses = []
    for tilt in (0., 20., -20., 35., -35.):
        for dyaw in np.linspace(-symmetry / 2, symmetry / 2, 5, endpoint=False):
            poses.append((grasp.copy(), _grasp_quat(base_yaw + dyaw, tilt)))
    return _with_symmetry(poses)


def keypoint_distances(pose, goals):
    """L1 point-matching distance from ``pose`` to each goal pose (world frame)."""
    position, quat = pose
    rotation = matrix(quat)
    source = (rotation @ KEYPOINTS.T).T + np.asarray(position)
    distances = np.empty(len(goals), dtype=np.float64)
    for i, (goal_pos, goal_quat) in enumerate(goals):
        target = (matrix(goal_quat) @ KEYPOINTS.T).T + np.asarray(goal_pos)
        distances[i] = np.abs(target - source).sum()
    return distances


def nearest_goal(pose, goals):
    distances = keypoint_distances(pose, goals)
    index = int(np.argmin(distances))
    return goals[index], float(distances[index])


def _reachable_goal(env, goal, *, standoff=0.075):
    """Verify IK reachability and support/pregrasp clearance for a candidate grasp."""
    grasp_pos, quat = np.asarray(goal[0], dtype=np.float64), goal[1]
    approach = matrix(quat)[:, 2]
    pregrasp_pos = grasp_pos - standoff * approach
    lifted_pos = np.array([grasp_pos[0], grasp_pos[1], min(0.45, grasp_pos[2] + 0.16)])
    for idx, position in enumerate((pregrasp_pos, grasp_pos, lifted_pos)):
        joints = p.calculateInverseKinematics(
            env.robot, TCP, position, quat, maxNumIterations=160,
            residualThreshold=1e-5, **env._ik)[:7]
        if any(not lo <= value <= hi for value, lo, hi in
               zip(joints, env._ik['lowerLimits'], env._ik['upperLimits'])):
            return False
        for joint, value in zip(ARM, joints):
            p.resetJointState(env.robot, joint, value)
        actual = env.pose()
        if np.linalg.norm(np.asarray(actual[0]) - position) > 0.006:
            return False
        if abs(np.dot(actual[1], quat)) < math.cos(0.08 / 2):
            return False
        if any(c[3] >= 0 for c in p.getClosestPoints(env.robot, env.support, 0.001)):
            return False
        if idx == 0 and p.getClosestPoints(env.robot, env.obj, 0.001):
            return False
    return True


def _optimize_waypoint(pose, goal):
    """Compute the immediate OMG-optimized target pose along the approach corridor to ``goal``.

    Decomposes the displacement ``g_pos - p_curr`` into forward distance ``h`` along
    the grasp approach axis ``a`` and lateral offset ``r`` perpendicular to ``a``.
    Outside the corridor (``||r|| > 1.2 cm``), an adaptive standoff ``d_so < h``
    steers the gripper laterally into the approach corridor while strictly
    advancing forward along ``+a``; inside the corridor, ``d_so = 0`` drives
    monotonically into the grasp pose until ``PointGraspEnv.retract()`` triggers.
    """
    p_curr = np.asarray(pose[0], dtype=np.float64)
    grasp_pos, quat = np.asarray(goal[0], dtype=np.float64), goal[1]
    approach = matrix(quat)[:, 2]
    to_grasp = grasp_pos - p_curr
    along = float(np.dot(to_grasp, approach))
    radial_vec = to_grasp - along * approach
    radial = float(np.linalg.norm(radial_vec))

    if along < 0.012 and radial > 0.022:
        # Gripper is laterally beside the object and below pregrasp clearance:
        # arc outward/upward along -approach to enter the approach corridor safely.
        d_so = min(0.065, max(0.035, along + 0.020))
        target_pos = grasp_pos - d_so * approach - 0.35 * radial_vec
    elif radial > 0.012 and along > 0.015:
        # Smooth corridor entry: advance forward by at least max(1.5 cm, 0.35 * along)
        # while correcting 100% of radial offset toward the approach axis.
        d_so = min(0.080, max(0.0, along - max(0.015, 0.35 * along)))
        target_pos = grasp_pos - d_so * approach
    else:
        # Aligned inside approach corridor: drive straight to grasp pose.
        target_pos = grasp_pos.copy()

    target_pos[2] = max(target_pos[2], min(grasp_pos[2], 0.010))
    return target_pos, quat


def _command_from_waypoint(pose, waypoint):
    """Direction-preserving 7D action command toward ``waypoint`` with open gripper (-1)."""
    delta = relative_pose(pose, waypoint)
    v_pos = delta[:3] / POS_SCALE
    v_rot = delta[3:] / ROT_SCALE
    scale_pos = max(1.0, float(np.max(np.abs(v_pos))))
    scale_rot = max(1.0, float(np.max(np.abs(v_rot))))
    return np.clip(np.r_[v_pos / scale_pos, v_rot / scale_rot, -1.0],
                   -1.0, 1.0).astype(np.float32)


class Planner:
    """Nearest-goal supervision plus real-time OMG trajectory optimization from any pose."""

    def __init__(self, env, goals=None, *, screen_ik=True, max_candidates=16):
        self.env = env
        self.goals = list(goals) if goals is not None else grasp_poses(env)
        self.screen_ik = screen_ik
        self.max_candidates = max_candidates
        self.cur_goal = None
        self._cur_obj_pos = None

    def nearest_label(self, pose):
        if not self.goals:
            return np.zeros(6, np.float32), False
        goal, _ = nearest_goal(pose, self.goals)
        return goal_label(pose, goal), True

    def _goal_costs(self, pose):
        p_curr = np.asarray(pose[0], dtype=np.float64)
        kp_dists = keypoint_distances(pose, self.goals)
        costs = kp_dists.copy()
        for i, (g_pos, g_quat) in enumerate(self.goals):
            approach = matrix(g_quat)[:, 2]
            to_grasp = np.asarray(g_pos, dtype=np.float64) - p_curr
            along = float(np.dot(to_grasp, approach))
            radial = float(np.linalg.norm(to_grasp - along * approach))
            costs[i] += 1.2 * radial + (0.45 if along < -0.005 else 0.0)
        return costs

    def _select_goal(self, pose):
        if not self.goals:
            return False
        limit = (max(self.max_candidates, 64)
                 if getattr(self.env, 'asset', None) is not None else self.max_candidates)
        order = np.argsort(self._goal_costs(pose), kind='stable')[:limit]
        state = p.saveState() if self.screen_ik else None
        try:
            for index in order:
                goal = self.goals[int(index)]
                if self.screen_ik:
                    for standoff in (0.075, 0.050):
                        p.restoreState(stateId=state)
                        if _reachable_goal(self.env, goal, standoff=standoff):
                            self.cur_goal = goal
                            self._cur_obj_pos = self.env._obj_pos().copy()
                            return True
                else:
                    self.cur_goal = goal
                    self._cur_obj_pos = self.env._obj_pos().copy()
                    return True
            return False
        finally:
            if state is not None:
                p.restoreState(stateId=state)
                p.removeState(state)

    def _ensure_goal(self, pose):
        obj_pos = self.env._obj_pos()
        if self._cur_obj_pos is not None and np.linalg.norm(obj_pos - self._cur_obj_pos) > 0.012:
            self.goals = grasp_poses(self.env)
            self.cur_goal = None
        if self.cur_goal is not None:
            p_curr = np.asarray(pose[0], dtype=np.float64)
            g_pos, g_quat = self.cur_goal
            approach = matrix(g_quat)[:, 2]
            to_grasp = np.asarray(g_pos, dtype=np.float64) - p_curr
            along = float(np.dot(to_grasp, approach))
            if along >= -0.005:
                return True
        return self._select_goal(pose)

    def plan_from(self, pose):
        """Return ``(expert_action, goal_label, ok)`` from ``pose`` using real-time OMG optimization."""
        if not self.goals or not self._ensure_goal(pose):
            return None, None, False
        waypoint = _optimize_waypoint(pose, self.cur_goal)
        action = _command_from_waypoint(pose, waypoint)
        return action, goal_label(pose, self.cur_goal), True

    def rollout(self, obs):
        """Execute a full closed-loop OMG approach plan; PointGraspEnv.retract() closes and lifts."""
        rows = []
        info = {}
        while True:
            pose = self.env.pose()
            action, goal, ok = self.plan_from(pose)
            if not ok:
                return [], dict(success=False, steps=len(rows), visible_fraction=float(obs[-1]),
                                tilt_at_close_deg=None, planner_no_path=True)
            next_obs, reward, term, trunc, info = self.env.step(action)
            rows.append((obs, action, action, reward, next_obs, float(term),
                         goal, (1., 1.)))
            obs = next_obs
            if term or trunc:
                break
        if not info.get('success'):
            return [], info
        return rows, info
