import unittest

import numpy as np
import torch

from ga_env import OBS_DIM, PointGraspEnv
from ga_model import Actor, Critic


class PointNetTest(unittest.TestCase):
    def test_forward_backward_and_parameter_count(self):
        torch.set_num_threads(1)
        actor, critic = Actor().cuda(), Critic().cuda()
        obs = torch.randn(2, OBS_DIM, device='cuda')
        action, goal = actor(obs)
        q1, q2, critic_goal = critic(obs, action)
        loss = action.square().mean() + goal.square().mean() + q1.mean() + q2.mean()
        loss = loss + critic_goal.square().mean()
        loss.backward()
        for model in (actor, critic):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        self.assertEqual(sum(p.numel() for m in (actor, critic) for p in m.parameters()), 1239829)

    def test_history_tracks_steps_and_resets(self):
        env = PointGraspEnv(seed=41)
        try:
            reference = env.reset(63, 'cube')
            self.assertEqual(len(env._point_history), 1)
            np.testing.assert_array_equal(env._obs(), reference)
            self.assertEqual(len(env._point_history), 1)
            env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
            self.assertEqual(len(env._point_history), 2)
            np.testing.assert_array_equal(env.reset(63, 'cube'), reference)
            self.assertEqual(len(env._point_history), 1)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
