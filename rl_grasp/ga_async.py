"""Asynchronous environment collection and bounded GPU learning submissions."""
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from time import perf_counter, sleep

import numpy as np


def dispatch_budget(args, issued, updates, gate):
    """Allow one worker wave plus four pending updates, without a worker barrier."""
    period = (args.update_every_env_steps if args.update_every_env_steps is not None
              else args.workers / args.updates_per_batch)
    limit = int((updates + 4) * period) + args.workers
    return max(0, min(gate, limit) - issued)


class AsyncLearner:
    """Run updates and action inference on different streams and actor weights."""

    def __init__(self, torch, device, learner, sample):
        self.torch, self.device, self.learner, self.sample = torch, device, learner, sample
        self.policy = deepcopy(learner.actor).eval().requires_grad_(False)
        self.policy_version = learner.updates // 2
        self.losses = None
        self.stream = self.event = None
        if device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=device)
            self.stream.wait_stream(torch.cuda.current_stream(device))
            self.event = torch.cuda.Event()

    def ready(self):
        return self.event is None or self.event.query()

    def refresh_policy(self):
        version = self.learner.updates // 2
        if version == self.policy_version or not self.ready():
            return
        with self.torch.no_grad():
            self.torch._foreach_copy_(list(self.policy.parameters()),
                                      list(self.learner.actor.parameters()))
            self.torch._foreach_copy_(list(self.policy.buffers()),
                                      list(self.learner.actor.buffers()))
        self.policy_version = version
        if self.stream is not None:
            # The next optimizer step can write the source after the copy completes.
            self.stream.wait_stream(self.torch.cuda.current_stream(self.device))

    def predict(self, obs, timing):
        tick = perf_counter()
        self.refresh_policy()
        timing['policy_refresh_s'] += perf_counter() - tick
        tick = perf_counter()
        with self.torch.no_grad():
            actions = self.policy(self.torch.as_tensor(obs, device=self.device))[0]
            timing['actor_submit_s'] += perf_counter() - tick
            tick = perf_counter()
            values = actions.cpu().numpy()
            timing['action_readback_s'] += perf_counter() - tick
            return values

    def submit(self, count, timing):
        if not count or not self.ready():
            return 0
        self.refresh_policy()
        context = self.torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with context:
            for _ in range(min(count, 4)):
                batch = self.sample(timing)
                tick = perf_counter()
                self.losses = self.learner.update(batch)
                timing['network_call_s'] += perf_counter() - tick
            if self.event is not None:
                self.event.record(self.stream)
        return min(count, 4)

    def finish(self):
        if self.stream is not None:
            self.stream.synchronize()

    def loss_metrics(self):
        if self.losses is None or not self.ready():
            return {}
        return dict(zip(self.losses, self.torch.stack(list(self.losses.values())).cpu().tolist()))


def ingest(replies, obs, pending_labels, replay, step, spans):
    """Store each new transition immediately and attach goals when known."""
    rows, ended, rewards, goals_to_apply = [], [], [], []
    for index, reply in replies:
        next_obs, reward, end_info, row, labels = reply[:5]
        obs[index] = next_obs
        pending_labels[index].append(step + len(rows))
        rows.append(row)
        rewards.append(reward)
        if end_info is not None:
            ended.append(end_info['success'])
            if labels is not None:
                goals_to_apply.extend(zip(pending_labels[index], labels, strict=True))
            pending_labels[index] = []
        if len(reply) == 6:
            _, timing, reset = reply[5]
            spans['sim_steps'] += 1
            spans['sim_resets'] += int(reset)
            for name, value in timing.items():
                spans[f'sim_{name}_s'] += value
    replay.append(rows)
    completed = step + len(rows)
    capacity = len(replay.obs)
    for position, label in goals_to_apply:
        if position >= completed - capacity:
            slot = position % capacity
            replay.goal[slot] = label
            replay.masks[slot, 1] = 1.
    return rewards, ended


def train_async(args, torch, device, learner, pool, obs, rng, replay, sample, monitor, log,
                online_start, deadline, start, stop, gate, pending_labels, on_boundary):
    executor = AsyncLearner(torch, device, learner, sample)
    available = set(range(args.workers))
    inflight = set()
    issued = step = start
    window_start = perf_counter()
    window_step, window_updates = start, learner.updates
    train_seconds = 0.
    rewards, outcomes = [], []
    times, spans = defaultdict(float), defaultdict(float)
    dispatches = submitted_actions = redispatches = 0
    idle = False
    timed_out = False
    while True:
        timed_out = timed_out or perf_counter() >= deadline
        replies = pool.receive_ready(inflight, timeout=0.001 if idle else 0., timing=times)
        for index, _ in replies:
            inflight.remove(index)
            available.add(index)
        if replies:
            tick = perf_counter()
            reward, ended = ingest(replies, obs, pending_labels, replay, step, spans)
            times['async_replay_ingest_s'] += perf_counter() - tick
            step += len(replies)
            rewards.extend(reward)
            outcomes.extend(ended)

        budget = dispatch_budget(args, issued, learner.updates, gate)
        indices = sorted(available)[:budget] if not timed_out else []
        if indices:
            if inflight:
                redispatches += 1
            tick = perf_counter()
            actions = executor.predict(obs[indices], times)
            noise = max(0.07, 0.20 - 0.13 * issued / args.steps)
            actions = np.clip(actions + rng.normal(0., noise, actions.shape), -1., 1.).astype(np.float32)
            send_start = perf_counter()
            pool.send('train_async', actions, indices)
            times['async_send_s'] += perf_counter() - send_start
            times['policy_seconds'] += perf_counter() - tick
            available.difference_update(indices)
            inflight.update(indices)
            issued += len(indices)
            dispatches += 1
            submitted_actions += len(indices)

        period = args.update_every_env_steps
        target_updates = step // period if period is not None else step * args.updates_per_batch // args.workers
        tick = perf_counter()
        submitted = executor.submit(target_updates-learner.updates, times)
        times['update_submit_seconds'] += perf_counter() - tick if submitted else 0.
        drained = not inflight and target_updates == learner.updates and (timed_out or step == gate)
        finished = drained and (timed_out or step == stop)
        if drained:
            executor.finish()

        if step-window_step >= args.log_every or drained:
            now = perf_counter()
            elapsed = now - window_start
            train_seconds += elapsed
            if rewards or learner.updates > window_updates:
                stats = dict(monitor.read()) if monitor is not None else {}
                if monitor is not None:
                    stats['torch_peak_allocated_mib'] = torch.cuda.max_memory_allocated()
                    stats['torch_peak_allocated_mib'] /= 2**20
                    stats['torch_peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
                if outcomes:
                    stats['rollout_success_rate'] = float(np.mean(outcomes))
                log('train', step=step, updates=learner.updates, graph_replays=learner.graph_replays,
                    learning_rate=learner.learning_rate,
                    online_window_start_s=window_start-online_start, online_window_end_s=now-online_start,
                    env_per_s=(step-window_step)/elapsed, updates_per_s=(learner.updates-window_updates)/elapsed,
                    step_reward_mean=float(np.mean(rewards)) if rewards else 0., episodes=len(outcomes),
                    replay_size=replay.size, async_dispatches=dispatches,
                    async_actions_per_dispatch=submitted_actions/max(1, dispatches),
                    async_redispatches_with_inflight=redispatches,
                    **times, **spans, **stats, **executor.loss_metrics())
            if drained:
                gate = on_boundary(step, finished)
            if finished:
                return step, train_seconds
            rewards.clear()
            outcomes.clear()
            times.clear()
            spans.clear()
            dispatches = submitted_actions = redispatches = 0
            window_step, window_updates = step, learner.updates
            window_start = perf_counter()

        idle = not (replies or indices or submitted or drained)
        if idle and not inflight:
            # A deadline or checkpoint can leave a final tranche of GPU work.
            sleep(0.0005)
