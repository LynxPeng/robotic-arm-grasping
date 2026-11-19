"""Regression tests for independent dispatch and hindsight replay slots."""
from multiprocessing import Pipe
import unittest

import numpy as np

from ga_async import ingest
from ga_env import ACT_DIM, OBS_DIM
from ga_workers import PointPool
from train_ga import Replay


class AsyncSamplingTest(unittest.TestCase):
    def test_dispatch_budget_bounds_learning_backlog_without_barrier(self):
        from ga_async import dispatch_budget
        from types import SimpleNamespace
        args = SimpleNamespace(workers=10, update_every_env_steps=4, updates_per_batch=3)
        self.assertEqual(dispatch_budget(args, issued=26, updates=0, gate=1000), 0)
        self.assertEqual(dispatch_budget(args, issued=26, updates=1, gate=1000), 4)
        self.assertEqual(dispatch_budget(args, issued=23, updates=0, gate=24), 1)
        args.update_every_env_steps = None
        self.assertEqual(dispatch_budget(args, issued=100, updates=0, gate=1000), 0)

    def test_ready_worker_can_receive_new_action_while_other_is_pending(self):
        pool = PointPool.__new__(PointPool)
        pool.pipes = []
        workers = []
        for _ in range(2):
            parent, child = Pipe()
            self.addCleanup(parent.close)
            self.addCleanup(child.close)
            pool.pipes.append(parent)
            workers.append(child)
        workers[1].send((True, 'fast observation'))
        self.assertEqual(pool.receive_ready({0, 1}, timeout=0.5), [(1, 'fast observation')])
        pool.send('train_async', ['next action'], [1])
        self.assertEqual(workers[1].recv(), ('train_async', 'next action'))
        self.assertFalse(workers[0].poll())

    def test_goal_backfill_never_changes_an_overwritten_transition(self):
        replay = Replay(2)
        obs = np.zeros((2, OBS_DIM), np.float32)
        pending = [[], []]

        def reply(reward, end=None, labels=None):
            state = np.zeros(OBS_DIM, np.float32)
            row = (state, np.zeros(ACT_DIM, np.float32), np.zeros(ACT_DIM, np.float32),
                   reward, state, False, np.zeros(6, np.float32), (False, False))
            return state, reward, end, row, labels

        for step, (worker, result) in enumerate((
            (0, reply(0.)), (1, reply(1.)), (1, reply(2., {'success': False})),
            (0, reply(3., {'success': True}, np.array([[10.]*6, [30.]*6]))),
        )):
            ingest([(worker, result)], obs, pending, replay, step, {})
            self.assertEqual(replay.size, min(step+1, 2))
        np.testing.assert_array_equal(replay.reward, [2., 3.])
        np.testing.assert_array_equal(replay.masks[:, 1], [0., 1.])
        np.testing.assert_array_equal(replay.goal, [[0.]*6, [30.]*6])
        self.assertEqual(pending, [[], []])


if __name__ == '__main__':
    unittest.main()
