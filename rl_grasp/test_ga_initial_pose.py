"""Check random starts without learning or changing model weights."""
import unittest

import numpy as np
import pybullet as p

from ga_env import PointGraspEnv, matrix


class InitialPoseTest(unittest.TestCase):
    def test_random_starts_and_repeatability(self):
        env = PointGraspEnv(seed=41)
        try:
            positions = []
            tilts = []
            for seed in range(30):
                obs = env.reset(seed, ('cube', 'cylinder', 'long_box')[seed % 3])
                position, quat = env.pose()
                positions.append(position)
                direction = env._obj_pos() - position
                distance = np.linalg.norm(direction)
                self.assertTrue(0.195 <= distance <= 0.505, distance)
                self.assertGreater(np.dot(matrix(quat)[:, 2], direction / distance), 0.99)
                self.assertEqual(obs[-1], 1)
                self.assertFalse(p.getClosestPoints(env.robot, env.obj, 0.0))
                tilts.append(np.degrees(np.arccos(-matrix(quat)[2, 2])))
            self.assertGreater(np.ptp(np.asarray(positions)[:, 2]), 0.1)
            self.assertGreater(np.ptp(tilts), 20)
            expected = env.reset(941, 'cube')
            env.reset(105, 'long_box')
            np.testing.assert_array_equal(env.reset(941, 'cube'), expected)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
