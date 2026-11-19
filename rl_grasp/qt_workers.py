"""CPU-only PyBullet workers, with one independent client per process."""

from multiprocessing import get_context

import numpy as np

from collect_demos import expert_action
from env import PandaGraspEnv


def run_worker(pipe, seed, horizon):
    rng = np.random.default_rng(seed)
    env = PandaGraspEnv(seed=seed, max_steps=horizon)
    obs = env._obs()
    phase = close_steps = 0
    try:
        pipe.send(obs)
        while True:
            command, value = pipe.recv()
            if command == 'close':
                break
            if command == 'reset':
                obs = env.reset(seed=value)
                phase = close_steps = 0
                pipe.send(obs)
                continue
            if command == 'expert':
                action, phase, close_steps = expert_action(env, phase, close_steps)
                if value:
                    action = np.clip(action + rng.normal(0, value, 4), -1, 1).astype(np.float32)
            else:
                action = value
            nxt, reward, term, trunc, info = env.step(action)
            done = term or trunc
            # Scale the existing shaping and favor reaching success without stalling.
            reward = reward * 0.1 - 0.008
            transition = np.concatenate((obs, action, nxt, [reward, done])).astype(np.float32)
            obs = nxt
            if done and command != 'eval':
                obs = env.reset()
                phase = close_steps = 0
            pipe.send((obs, transition, info if done else None))
    except EOFError:
        pass
    finally:
        env.close()
        pipe.close()


class Simulators:
    def __init__(self, count, seed, horizon):
        ctx = get_context('spawn')
        self.pipes, self.processes = [], []
        try:
            for index in range(count):
                parent, child = ctx.Pipe()
                worker = ctx.Process(target=run_worker, args=(child, seed+index, horizon), daemon=True)
                worker.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(worker)
            self.observations = np.stack([pipe.recv() for pipe in self.pipes])
        except BaseException:
            self.close()
            raise

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except OSError:
                pass
            pipe.close()
        for worker in self.processes:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join()

    def send(self, command, values):
        for pipe, value in zip(self.pipes, values, strict=True):
            pipe.send((command,value))

    def receive(self):
        observations, rows, ended = zip(*(pipe.recv() for pipe in self.pipes))
        self.observations = np.stack(observations)
        return np.stack(rows), [info for info in ended if info is not None]
