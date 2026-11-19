"""BatchNorm state must follow the actor used by asynchronous inference."""
import unittest
from collections import defaultdict
import numpy as np
import torch
from ga_async import AsyncLearner
from ga_env import OBS_DIM
from ga_model import Learner


class BatchNormStateTest(unittest.TestCase):
    def test_native_encoder_and_targets(self):
        learner = Learner(torch.device('cuda'), 1e-4, .5)
        self.assertTrue(any(isinstance(m, torch.nn.BatchNorm2d)
                            for m in learner.actor.modules()))
        self.assertFalse(learner.actor_target.training)
        self.assertFalse(learner.critic_target.training)
        self.assertEqual(type(learner.actor.encoder.sa1).__name__, 'PointnetSAModule')

    def test_policy_copies_running_statistics(self):
        learner = Learner(torch.device('cuda'), 1e-4, .5)
        executor = AsyncLearner(torch, torch.device('cuda'), learner, None)
        with torch.no_grad():
            for name, buf in learner.actor.named_buffers():
                buf.fill_(3 if 'num_batches_tracked' in name else .5)
        learner.updates = 2
        executor.refresh_policy()
        for actual, expected in zip(executor.policy.buffers(), learner.actor.buffers()):
            torch.testing.assert_close(actual, expected)
        result = executor.predict(np.zeros((1, OBS_DIM), np.float32), defaultdict(float))
        self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
