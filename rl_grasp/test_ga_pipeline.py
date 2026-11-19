"""CPU integration tests: 8-field replay, DAgger masks, online goals, planner demos."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ga_env import ACT_DIM, OBS_DIM
from ga_workers import PointPool
import train_ga


def _row(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    action = rng.uniform(-1, 1, ACT_DIM).astype(np.float32)
    label = action.copy()
    goal = rng.standard_normal(6).astype(np.float32)
    return (obs, action, label, 0.0, obs.copy(), 0.0, goal, (0., 1.))


class ReplayTest(unittest.TestCase):
    def test_roundtrip_eight_fields(self):
        replay = train_ga.Replay(8)
        rows = [_row(i) for i in range(5)]
        replay.append(rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'demos.npz'
            replay.save(path)
            loaded = train_ga.Replay.load(path)
        self.assertEqual(loaded.size, 5)
        self.assertEqual(len(loaded.fields), 8)
        np.testing.assert_array_equal(loaded.label[:5], replay.label[:5])
        np.testing.assert_array_equal(loaded.masks[:5], replay.masks[:5])

    def test_legacy_seven_field_load(self):
        rng = np.random.default_rng(1)
        size = 4
        obs = rng.standard_normal((size, OBS_DIM)).astype(np.float32)
        action = rng.uniform(-1, 1, (size, ACT_DIM)).astype(np.float32)
        arrays = [obs, action, np.zeros(size, np.float32), obs.copy(),
                  np.zeros(size, np.float32), np.zeros((size, 6), np.float32),
                  np.tile((1., 1.), (size, 1)).astype(np.float32)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.npz'
            np.savez(path, *arrays)
            loaded = train_ga.Replay.load(path)
        self.assertEqual(len(loaded.fields), 8)
        np.testing.assert_array_equal(loaded.label[:size], action)
        np.testing.assert_array_equal(loaded.action[:size], action)

    def test_batch_from_eight_tensors(self):
        import torch
        demos = train_ga.Replay(8)
        demos.append([_row(i) for i in range(6)])
        online = train_ga.Replay(8)
        online.append([_row(100 + i) for i in range(6)])
        batch = train_ga.batch_from(torch, torch.device('cpu'),
                                    np.random.default_rng(0), demos, online, 8)
        self.assertEqual(len(batch), 8)
        self.assertEqual(batch[2].shape, (8, ACT_DIM))  # label tensor


class AsyncIngestTest(unittest.TestCase):
    def test_ingest_eight_field_rows_and_backfill(self):
        from ga_async import ingest
        replay = train_ga.Replay(16)
        obs = np.zeros((2, OBS_DIM), np.float32)
        pending = [[], []]
        spans = {}
        row0 = _row(1)
        row1 = _row(2)
        replies = [(0, (row0[4], 0.0, None, row0, None)),
                   (1, (row1[4], 0.0, None, row1, None))]
        rewards, ended = ingest(replies, obs, pending, replay, 0, spans)
        self.assertEqual(replay.size, 2)
        self.assertEqual(len(rewards), 2)
        np.testing.assert_array_equal(replay.masks[:2, 1], (1., 1.))
        # Episode end on worker 0 relabels its pending transitions with hindsight goals.
        labels = np.stack([np.ones(6, np.float32)] * 2)
        end = dict(success=True)
        replies = [(0, (row0[4], 1.0, end, _row(3), labels))]
        ingest(replies, obs, pending, replay, 2, spans)
        self.assertEqual(replay.masks[0, 1], 1.)
        np.testing.assert_array_equal(replay.goal[0], np.ones(6, np.float32))


class WorkerPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = PointPool(2, 41, 60, renderer='tiny',
                             planner_cfg=dict(ratio=0.5, min_step=2, max_step=40,
                                              expert='planner', online_goals=True))

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_online_steps_have_goals_and_dagger(self):
        obs = self.pool.call('reset', [(700 + i, None) for i in range(2)], range(2))
        collected = []
        for _ in range(70):
            actions = [np.zeros(ACT_DIM, np.float32)] * 2
            replies = self.pool.call('train', actions, range(2))
            for next_obs, reward, info, rows in replies:
                if rows:
                    collected.extend(rows)
        self.assertGreater(len(collected), 0)
        has_goal = 0
        is_demo = 0
        for row in collected:
            self.assertEqual(len(row), 8)
            masks = row[7]
            has_goal += int(masks[1] > 0)
            is_demo += int(masks[0] > 0)
        # Every online transition gets nearest-goal supervision (the core fix).
        self.assertEqual(has_goal, len(collected))
        # DAgger relabels a fraction of steps as expert demonstrations.
        self.assertGreater(is_demo, 0)

    def test_planner_demo_command(self):
        success = 0
        for seed in (801, 802, 803):
            rows, info = self.pool.call('demo', [(seed, 'cylinder', 0.)], [0])[0]
            if info.get('success'):
                success += 1
                for row in rows:
                    self.assertEqual(len(row), 8)
                    np.testing.assert_array_equal(row[7], (1., 1.))
        self.assertGreater(success, 0)


if __name__ == '__main__':
    unittest.main()
