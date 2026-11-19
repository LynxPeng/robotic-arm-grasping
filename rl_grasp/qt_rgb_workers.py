"""Shared RGB observations for CPU simulation workers."""

from multiprocessing import get_context
import numpy as np

from qt_rgb_env import RGBGraspEnv, SIZE, ACT_DIM, expert


def worker(pipe, storage, index, seed, horizon):
    frame = np.frombuffer(storage, np.uint8).reshape(-1, SIZE, SIZE, 3)[index]
    rng = np.random.default_rng(seed)
    env = RGBGraspEnv(seed=seed, max_steps=horizon)
    phase = wait = 0

    def publish(obs, reward=0., ended=None, action=None, transition_kind=None):
        image, aux = obs
        np.copyto(frame, image)
        pipe.send((aux, reward, ended, action, transition_kind))

    try:
        publish(env._obs())
        while True:
            cmd, value = pipe.recv()
            if cmd == 'close':
                return
            if cmd == 'reset':
                publish(env.reset(seed=value[0], kind=value[1]))
                phase = wait = 0
                continue
            if cmd == 'expert':
                action, phase, wait = expert(env, phase, wait)
                if value:
                    action = np.clip(action + rng.normal(0, value, ACT_DIM), -1, 1).astype(np.float32)
            else:
                action = value
            transition_kind = env.kind
            obs, reward, term, trunc, info = env.step(action)
            ended = info if term or trunc else None
            if ended is not None and cmd != 'eval':
                obs = env.reset()
                phase = wait = 0
            publish(obs, reward * 0.1 - 0.008, ended, action, transition_kind)
    except EOFError:
        pass
    finally:
        env.close()
        pipe.close()


class RGBPool:
    def __init__(self, count, seed, horizon):
        ctx = get_context('spawn')
        self.frames_buffer = ctx.RawArray('B', count * SIZE * SIZE * 3)
        self.frames = np.frombuffer(self.frames_buffer, np.uint8).reshape(count, SIZE, SIZE, 3)
        self.pipes, self.processes = [], []
        try:
            for index in range(count):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=worker, args=(child, self.frames_buffer, index, seed+index,
                                                        horizon), daemon=True)
                proc.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(proc)
            self.aux = np.stack([pipe.recv()[0] for pipe in self.pipes])
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
        for proc in self.processes:
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join()

    def send(self, command, values):
        for pipe, value in zip(self.pipes, values, strict=True):
            pipe.send((command, value))

    def receive(self):
        aux, rewards, ended, actions, kinds = zip(*(pipe.recv() for pipe in self.pipes))
        self.aux = np.stack(aux)
        return np.array(rewards, dtype=np.float32), ended, np.stack(actions), kinds
