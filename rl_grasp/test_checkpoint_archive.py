"""Check immutable step archives and complete training-state round trips."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ga_checkpoint import save_training


class CheckpointArchiveTest(unittest.TestCase):
    def test_step_archives_survive_latest_replacement(self):
        learner = SimpleNamespace(updates=5)
        for name in ('actor', 'critic', 'actor_target', 'critic_target'):
            setattr(learner, name, torch.nn.Linear(2, 1))
        for name in ('actor', 'critic'):
            net = getattr(learner, name)
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)
            net(torch.ones(1, 2)).sum().backward()
            opt.step()
            setattr(learner, name + '_opt', opt)
        replay = SimpleNamespace(size=2, cursor=2, fields=[np.ones((2, 3), np.float32)])
        rng = np.random.default_rng(41)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training-state.pt'
            def save(step, keep=True):
                save_training(path, torch, learner, torch.device('cpu'), rng, replay,
                              [], np.zeros((1, 2), np.float32), {}, step, 17, step + 20,
                              keep_steps=keep)
            save(20)
            first = path.with_name('training-state-step-000000020.pt')
            original = first.read_bytes()
            with torch.no_grad():
                learner.actor.weight.add_(1)
            save(40)
            self.assertEqual(first.read_bytes(), original)
            old = torch.load(first, weights_only=True)
            latest = torch.load(path, weights_only=True)
            self.assertEqual(old['steps'], 20)
            self.assertEqual(latest['steps'], 40)
            self.assertTrue(old['actor_opt']['state'])
            self.assertIn('numpy_rng', old)
            self.assertIn('torch_rng', old)
            self.assertEqual(len(old['replay']), 1)
            self.assertFalse(torch.equal(old['actor']['weight'], latest['actor']['weight']))
            self.assertEqual(path.read_bytes(), path.with_name('training-state-step-000000040.pt').read_bytes())
            save(60, keep=False)
            self.assertFalse(path.with_name('training-state-step-000000060.pt').exists())
            self.assertEqual(first.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
