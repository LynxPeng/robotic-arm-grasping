"""Process-based PyBullet sampling. The parent owns the only CUDA context."""
from copy import deepcopy
from multiprocessing import get_context
from multiprocessing.connection import wait
from time import perf_counter, sleep
from traceback import format_exc

import numpy as np

from ga_env import PointGraspEnv, ScriptedExpert, goal_label
from ga_planner import Planner
from env import STEPS_PER_CTRL

DEFAULT_PLANNER = dict(ratio=0.5, min_step=5, max_step=18, expert='planner', online_goals=True)


def label_episode(transitions, close_pose):
    """Package per-step rows; successful episodes relabel goals with hindsight."""
    result = []
    for obs, action, label, reward, next_obs, terminal, goal, masks, pose in transitions:
        if close_pose is not None:
            goal = goal_label(pose, close_pose)
            masks = (masks[0], 1.)
        result.append((obs, action, label, reward, next_obs, terminal, goal, masks))
    return result


def worker(pipe, seed, horizon, gui, profile, renderer, resumable, assets=None, planner_cfg=None):
    env = None
    try:
        cfg = dict(DEFAULT_PLANNER, **(planner_cfg or {}))
        options = dict(seed=seed, max_steps=horizon, gui=gui, renderer=renderer)
        if assets is None:
            env = PointGraspEnv(**options)
        else:
            from qdg_env import QDGPointGraspEnv
            env = QDGPointGraspEnv(assets=assets, **options)
        if profile:
            # Only the dedicated benchmark worker process sees these wrappers.
            import pybullet as p
            spans = dict.fromkeys(('ik', 'physics', 'camera', 'observation'), 0.)

            def timed(key, call):
                def measure(*args, **kwargs):
                    start = perf_counter()
                    result = call(*args, **kwargs)
                    spans[key] += perf_counter() - start
                    return result
                return measure

            p.calculateInverseKinematics = timed('ik', p.calculateInverseKinematics)
            p.stepSimulation = timed('physics', p.stepSimulation)
            p.getCameraImage = timed('camera', p.getCameraImage)
            env._obs = timed('observation', env._obs)
        obs = env._obs()
        trajectory = []
        episode_rng = episode_kind = None
        episode_actions = []
        planner = None
        step_index = 0

        def reset(episode_seed=None, kind=None):
            nonlocal episode_rng, episode_kind, planner, step_index
            if resumable:
                source = env.rng if episode_seed is None else np.random.default_rng(episode_seed)
                episode_rng = deepcopy(source.bit_generator.state)
                episode_kind = kind
                episode_actions.clear()
            step_index = 0
            result = env.reset(episode_seed, kind)
            planner = Planner(env)
            return result

        def annotate(pose, executed_action, index):
            """Heuristic goal for every online step; DAgger expert label when queried."""
            if cfg['online_goals']:
                goal, has_goal = planner.nearest_label(pose)
            else:
                goal, has_goal = np.zeros(6, np.float32), False
            label = np.asarray(executed_action, np.float32)
            is_demo = 0.
            if (cfg['ratio'] > 0 and cfg['min_step'] <= index <= cfg['max_step']
                    and np.random.default_rng([seed, index]).random() < cfg['ratio']):
                expert_action, expert_goal, ok = planner.plan_from(pose)
                if ok:
                    label = np.asarray(expert_action, np.float32)
                    goal = np.asarray(expert_goal, np.float32)
                    is_demo, has_goal = 1., True
            return label, np.asarray(goal, np.float32), (is_demo, float(has_goal))

        def transition(action):
            pose = env.pose()
            next_obs, reward, term, trunc, info = env.step(action)
            return next_obs, reward, term, trunc, info, pose

        pipe.send((True, obs))
        while True:
            cmd, value = pipe.recv()
            if cmd == 'close':
                return
            if cmd == 'demo':
                trial_seed, kind, tilt = value
                obs = env.reset(trial_seed, kind)
                demo_planner = Planner(env)
                if cfg['expert'] == 'planner':
                    rows, info = demo_planner.rollout(obs)
                else:
                    expert = ScriptedExpert(env, tilt)
                    path = []
                    while True:
                        pose = env.pose()
                        action = expert.action(pose)
                        goal, has_goal = demo_planner.nearest_label(pose)
                        next_obs, reward, term, trunc, info = env.step(action)
                        path.append((obs, action, action, reward, next_obs, float(term),
                                     goal, (1., float(has_goal)), pose))
                        obs = next_obs
                        if term or trunc:
                            break
                    rows = label_episode(path, env.close_pose) if info['success'] else []
                reply = rows, info
            elif cmd == 'reset':
                obs = reset(*value)
                trajectory.clear()
                reply = obs
            elif cmd == 'snapshot' and resumable:
                reply = dict(rng=episode_rng, kind=episode_kind, actions=episode_actions)
            elif cmd == 'restore' and resumable:
                env.rng.bit_generator.state = value['rng']
                obs = reset(None, value['kind'])
                trajectory.clear()
                for items in value['actions']:
                    action = np.asarray(items, np.float32)
                    pose = env.pose()
                    label, goal, masks = annotate(pose, action, step_index)
                    step_index += 1
                    next_obs, reward, term, trunc, _, _ = transition(action)
                    if term or trunc:
                        raise ValueError('Checkpoint action history extends past the episode boundary')
                    trajectory.append((obs, action, label, reward, next_obs, float(term),
                                       goal, masks, pose))
                    episode_actions.append(items)
                    obs = next_obs
                reply = obs
            elif cmd in ('train', 'train_async', 'eval'):
                if profile:
                    for key in spans:
                        spans[key] = 0.
                    start = perf_counter()
                training = cmd != 'eval'
                if training:
                    pose = env.pose()
                    label, goal, masks = annotate(pose, value, step_index)
                    step_index += 1
                next_obs, reward, term, trunc, info, pose = transition(value)
                if profile:
                    step_done = perf_counter()
                    step_spans = spans.copy()
                done = term or trunc
                rows = []
                async_step = cmd == 'train_async'
                labels = None
                if training:
                    if resumable:
                        episode_actions.append(value.tolist())
                    trajectory.append((obs, value, label, reward, next_obs, float(term),
                                       goal, masks, pose))
                    if async_step:
                        rows = (obs, value, label, reward, next_obs, float(term), goal, masks)
                    if done:
                        close_pose = env.close_pose if info['success'] else None
                        if async_step:
                            if close_pose is not None:
                                labels = np.stack([goal_label(row[-1], close_pose) for row in trajectory])
                        else:
                            rows = label_episode(trajectory, close_pose)
                        trajectory.clear()
                resetting = done and training
                reset_start = perf_counter() if profile and resetting else None
                obs = reset() if resetting else next_obs
                reply = obs, reward, info if done else None, rows
                if async_step:
                    reply += (labels,)
                if profile and training:
                    ready = perf_counter()
                    reset_seconds = ready - reset_start if reset_start is not None else 0.
                    observe = step_spans.pop('observation')
                    step_spans['pointcloud'] = observe - step_spans['camera']
                    step_spans['other_control'] = step_done - start - sum(step_spans.values())
                    step_spans['episode_reset'] = reset_seconds
                    step_spans['bookkeeping'] = ready - step_done - reset_seconds
                    reply += ((ready, step_spans, resetting),)
                if gui:
                    sleep(STEPS_PER_CTRL / 240)
            else:
                raise ValueError(f'Unexpected worker command: {cmd}')
            pipe.send((True, reply))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        pipe.send((False, format_exc()))
    finally:
        if env is not None:
            env.close()
        pipe.close()


class PointPool:
    def __init__(self, count, seed, horizon, gui=False, *, profile=False, renderer='tiny', resumable=False,
                 assets=None, planner_cfg=None):
        self.pipes = []
        self.processes = []
        ctx = get_context('spawn')
        try:
            for i in range(count):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker, args=(child, seed + i, horizon, gui, profile,
                                                            renderer, resumable, assets, planner_cfg), daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
            self.receive(range(count))
        except BaseException:
            self.close()
            raise

    def send(self, cmd, values, indices):
        for i, value in zip(indices, values, strict=True):
            self.pipes[i].send((cmd, value))

    def receive(self, indices):
        results = []
        for i in indices:
            ok, payload = self.pipes[i].recv()
            if not ok:
                raise RuntimeError(f'PyBullet worker {i} failed:\n{payload}')
            results.append(payload)
        return results

    def receive_ready(self, indices, timeout=0., timing=None):
        available = {self.pipes[i]: i for i in indices}
        if not available:
            return []
        tick = perf_counter() if timing is not None else 0.
        ready = [available[pipe] for pipe in wait(available, timeout=timeout)]
        if timing is not None:
            timing['worker_wait_seconds' if timeout else 'worker_poll_s'] += perf_counter() - tick
            tick = perf_counter()
        replies = list(zip(ready, self.receive(ready), strict=True))
        if timing is not None:
            timing['worker_receive_s'] += perf_counter() - tick
        return replies

    def call(self, cmd, values, indices):
        self.send(cmd, values, indices)
        return self.receive(indices)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except OSError:
                pass
            pipe.close()
        for process in self.processes:
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join()
