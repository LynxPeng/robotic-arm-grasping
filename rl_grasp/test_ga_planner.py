"""CPU tests for the OMG-style goal set, nearest-goal supervision and planner."""
import unittest

import numpy as np
import pybullet as p

from ga_env import PointGraspEnv, ACT_DIM
from ga_planner import Planner, grasp_poses, nearest_goal, keypoint_distances


class PlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PointGraspEnv(renderer='tiny', max_steps=120)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _reset(self, seed):
        obs = self.env.reset(seed=seed)
        return obs, Planner(self.env)

    def test_goal_set_nonempty(self):
        self.env.reset(seed=11)
        goals = grasp_poses(self.env)
        self.assertGreater(len(goals), 0)
        for position, quat in goals:
            self.assertTrue(np.isfinite(position).all())
            self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=4)

    def test_nearest_goal_supervises_arbitrary_pose(self):
        _, planner = self._reset(12)
        # The core fix: even a failed/intermediate online state gets a goal label.
        for _ in range(5):
            action = np.random.default_rng(0).uniform(-1, 1, ACT_DIM).astype(np.float32)
            self.env.step(action)
            goal, has_goal = planner.nearest_label(self.env.pose())
            self.assertTrue(has_goal)
            self.assertEqual(goal.shape, (6,))
            self.assertTrue(np.isfinite(goal).all())

    def test_nearest_matches_keypoint_argmin(self):
        _, planner = self._reset(13)
        pose = self.env.pose()
        goal, distance = nearest_goal(pose, planner.goals)
        distances = keypoint_distances(pose, planner.goals)
        self.assertAlmostEqual(distance, float(distances.min()), places=6)
        self.assertIs(goal, planner.goals[int(np.argmin(distances))])

    def test_plan_from_valid_and_nondestructive(self):
        _, planner = self._reset(14)
        pose = self.env.pose()
        before = [p.getJointState(self.env.robot, j)[0] for j in range(7)]
        action, goal, ok = planner.plan_from(pose)
        after = [p.getJointState(self.env.robot, j)[0] for j in range(7)]
        np.testing.assert_allclose(before, after, atol=1e-6)
        if ok:
            self.assertEqual(action.shape, (ACT_DIM,))
            self.assertTrue((np.abs(action) <= 1.0 + 1e-6).all())
            self.assertEqual(goal.shape, (6,))
            self.assertTrue(np.isfinite(goal).all())

    def test_rollout_demo_rows(self):
        successes = 0
        for seed in (21, 22, 23, 24):
            obs, planner = self._reset(seed)
            rows, info = planner.rollout(obs)
            if info.get('success'):
                successes += 1
                self.assertGreater(len(rows), 0)
                for row in rows:
                    self.assertEqual(len(row), 8)
                    obs_, action, label, reward, next_obs, terminal, goal, masks = row
                    np.testing.assert_array_equal(action, label)
                    np.testing.assert_array_equal(masks, (1., 1.))
                    self.assertEqual(goal.shape, (6,))
        self.assertGreater(successes, 0, 'planner should produce at least one verified demo')


if __name__ == '__main__':
    unittest.main()
