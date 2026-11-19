"""Run the local point-cloud, six-DoF grasp experiment and its controls."""
import argparse
import json
import math
import os
import subprocess
from contextlib import ExitStack
from collections import defaultdict
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, time

import numpy as np

from env import ROOT
from ga_checkpoint import load_training, save_training, signature
from ga_env import ACT_DIM, OBJECTS, OBS_DIM
from ga_workers import PointPool

PROJECT = 'ga-ddpg-mini'
TILTS = (0., -20., 20., -35., 35.)


class Replay:
    def __init__(self, capacity):
        self.obs = np.empty((capacity, OBS_DIM), np.float16)
        self.next_obs = np.empty_like(self.obs)
        self.action = np.empty((capacity, ACT_DIM), np.float32)
        self.label = np.empty((capacity, ACT_DIM), np.float32)
        self.reward = np.empty(capacity, np.float32)
        self.terminal = np.empty_like(self.reward)
        self.goal = np.empty((capacity, 6), np.float32)
        self.masks = np.empty((capacity, 2), np.float32)
        self.fields = (self.obs, self.action, self.label, self.reward, self.next_obs,
                       self.terminal, self.goal, self.masks)
        self.size = self.cursor = 0

    def append(self, rows):
        if not rows:
            return
        capacity = len(self.obs)
        rows = rows[-capacity:]
        idx = (self.cursor + np.arange(len(rows))) % capacity
        for field, values in zip(self.fields, zip(*rows), strict=True):
            field[idx] = values
        self.cursor = (self.cursor + len(rows)) % capacity
        self.size = min(capacity, self.size + len(rows))

    def save(self, path):
        np.savez(path, *(a[:self.size] for a in self.fields))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if data['arr_0'].shape[1:] != (OBS_DIM,):
                raise ValueError(f'Demonstration observation shape must be ({OBS_DIM},); regenerate demos for this encoder.')
            size = len(data['arr_0'])
            replay = cls(max(1, size))
            # Legacy archives store 7 fields without a BC label; demos used the
            # executed expert action as the label, so reconstruct label = action.
            legacy = len(data.files) == 7
            for i, field in enumerate(replay.fields):
                if legacy:
                    source = data['arr_1'] if i == 2 else data[f'arr_{i if i < 2 else i - 1}']
                else:
                    source = data[f'arr_{i}']
                field[:size] = source
            replay.size = size
        if not replay.size:
            raise ValueError(f'Empty demonstration archive: {path}')
        return replay


def batch_from(torch, device, rng, demos, online, size, timing=None):
    tick = perf_counter() if timing is not None else 0.
    demo_size = size if online is None or online.size < size else max(1, size // 4)
    a = rng.integers(demos.size, size=demo_size)
    b = rng.integers(online.size, size=size-demo_size) if demo_size < size else None
    if timing is not None:
        timing['replay_sample_s'] += perf_counter() - tick
    result = []
    for i, field in enumerate(demos.fields):
        if timing is not None:
            tick = perf_counter()
        sample = field[a] if b is None else np.concatenate((field[a], online.fields[i][b]))
        value = torch.from_numpy(sample)
        if timing is not None:
            timing['replay_sample_s'] += perf_counter() - tick
            tick = perf_counter()
        if device.type == 'cuda':
            value = value.pin_memory()
        if timing is not None:
            timing['pin_s'] += perf_counter() - tick
            tick = perf_counter()
        result.append(value.to(device, dtype=torch.float32, non_blocking=True))
        if timing is not None:
            timing['copy_submit_s'] += perf_counter() - tick
    return result


def gpu_metrics():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        util, mem = map(float, result.stdout.strip().split(','))
        return util, mem
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class GPUMonitor:
    """Sample device metrics concurrently with the training workload."""
    def __init__(self, online_start, sample_log=None):
        self.online_start = online_start
        self.sample_log = sample_log
        self.stop = Event()
        self.lock = Lock()
        self.samples = []
        self.thread = Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop.is_set():
            sample = gpu_metrics()
            if sample is not None:
                with self.lock:
                    self.samples.append(sample)
                if self.sample_log is not None:
                    util, memory = sample
                    row = dict(online_seconds=perf_counter()-self.online_start,
                               unix_seconds=time(), gpu_util_pct=util, gpu_device_mib=memory)
                    with self.sample_log.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row) + '\n')
            self.stop.wait(1)

    def read(self):
        with self.lock:
            samples, self.samples = self.samples, []
        if not samples:
            return {}
        utilization, memory = zip(*samples)
        return dict(gpu_util_pct_mean=float(np.mean(utilization)),
                    gpu_util_pct_peak=max(utilization),
                    gpu_device_memory_mib_peak=max(memory), gpu_samples=len(samples))

    def close(self):
        self.stop.set()
        self.thread.join()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('demo', 'bc', 'benchmark', 'td3', 'ga', 'eval'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--demos-from', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path, help='Continue a full training-state checkpoint')
    parser.add_argument('--save-every', type=int, default=0, help='Full checkpoint interval in environment steps')
    parser.add_argument('--keep-checkpoints', action='store_true',
                        help='Keep an immutable full checkpoint named by step at each save')
    parser.add_argument('--stop-after', type=int, help='Save and stop at this step without changing the training horizon')
    parser.add_argument('--renderer', choices=('tiny', 'egl'), default='tiny')
    parser.add_argument('--logger', choices=('trackio', 'tensorboard'), default='trackio')
    parser.add_argument('--tensorboard-dir', type=Path, default=Path('/root/tf-logs'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--async-envs', action='store_true',
                        help='Dispatch each ready environment independently of the others')
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--demo-trials', type=int, default=450)
    parser.add_argument('--bc-updates', type=int, default=2500)
    parser.add_argument('--bc-eval-every', type=int, default=3000,
                        help='Evaluate policy every N BC updates (0 to disable)')
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--updates-per-batch', type=int, default=3)
    parser.add_argument('--update-every-env-steps', type=int,
                        help='Schedule one update per N total environment steps across worker batches')
    parser.add_argument('--benchmark-seconds', type=float,
                        help='Stop a benchmark at the first completed worker batch after this deadline')
    parser.add_argument('--online-capacity', type=int, default=100000)
    parser.add_argument('--log-every', type=int, default=1200)
    parser.add_argument('--eval-every', type=int, default=12000)
    parser.add_argument('--eval-per-kind', type=int, default=20)
    parser.add_argument('--qdg-dataset', type=Path, help='Directory containing QDGset train and holdout manifests')
    parser.add_argument('--qdg-validation-objects', type=int, default=12,
                        help='Number of fixed training objects used for periodic validation')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--eval-seed', type=int, default=300001)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', type=float, nargs=3, metavar=('START', 'RAMP_END', 'DECAY_START'),
                        help='Ramp from START to --lr, hold to DECAY_START, then cosine decay to zero')
    parser.add_argument('--lr-milestones', type=int, nargs='+',
                        help='Update steps at which to multiply the learning rate by --lr-gamma')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                        help='Multiplicative factor for --lr-milestones (official GA-DDPG 0.5)')
    parser.add_argument('--lr-warmup-updates', type=int, default=0,
                        help='Linearly increase the online learning rate from 10%% to the target')
    parser.add_argument('--aux-weight', type=float, default=0.5)
    parser.add_argument('--dagger-ratio', type=float, default=0.5,
                        help='Fraction of eligible online steps relabelled by the OMG-style expert')
    parser.add_argument('--dagger-min-step', type=int, default=5)
    parser.add_argument('--dagger-max-step', type=int, default=18)
    parser.add_argument('--expert', choices=('planner', 'scripted'), default='planner',
                        help='Demonstration expert: OMG-style planner or legacy scripted waypoints')
    parser.add_argument('--no-online-goals', action='store_true',
                        help='Disable nearest-goal supervision on online transitions')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='MDP discount factor (official GA-DDPG 0.95)')
    parser.add_argument('--tau', type=float, default=1e-4,
                        help='Polyak target-network averaging (official repo 1e-4)')
    parser.add_argument('--ddpg-weight', type=float, default=0.2,
                        help='Actor RL mixing weight lambda in -lambda*Q + (1-lambda)*BC (official 0.1~0.2)')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Log host time by update stage')
    parser.add_argument('--profile-workers', action='store_true',
                        help='Break down simulation worker CPU time')
    parser.add_argument('--trace', action='store_true', help='Capture a short CUDA trace during benchmark')
    parser.add_argument('--cuda-graph', action='store_true', help='Replay the two TD3 update paths on CUDA')
    args = parser.parse_args()
    if args.lr_schedule:
        start, ramp_end, decay_start = args.lr_schedule
        period = args.update_every_env_steps
        if (not args.async_envs or (period is not None and period < 1) or args.lr_warmup_updates
                or not 0 < start <= args.lr < math.inf
                or not 0 < ramp_end <= decay_start < args.steps
                or (period is not None and any(step % period for step in (ramp_end, decay_start, args.steps)))):
            parser.error('--lr-schedule requires async sampling, aligned ramp/decay/end steps and no warmup')
    if args.lr_warmup_updates == 1 or args.lr_warmup_updates < 0:
        parser.error('--lr-warmup-updates must be zero or at least 2')
    for name in ('workers', 'horizon', 'demo_trials', 'bc_updates', 'steps', 'batch',
                 'updates_per_batch', 'online_capacity', 'log_every', 'eval_every', 'eval_per_kind',
                 'qdg_validation_objects'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if not 0 < args.gamma <= 1:
        parser.error('--gamma must be in (0, 1]')
    if not 0 < args.tau <= 1:
        parser.error('--tau must be in (0, 1]')
    if not 0 <= args.dagger_ratio <= 1:
        parser.error('--dagger-ratio must be in [0, 1]')
    if not 0 <= args.dagger_min_step <= args.dagger_max_step < args.horizon:
        parser.error('DAgger step window must lie within the episode horizon')
    if args.gui and (args.stage != 'eval' or args.workers != 1):
        parser.error('--gui requires eval --workers 1')
    if (args.profile or args.profile_workers or args.trace) and args.stage != 'benchmark':
        parser.error('profiling requires benchmark')
    if args.trace and (args.device != 'cuda' or args.steps < 16 * args.workers):
        parser.error('--trace requires CUDA and at least 16 worker batches')
    if args.cuda_graph and (args.device != 'cuda' or args.stage not in ('benchmark', 'td3', 'ga')):
        parser.error('--cuda-graph requires CUDA and an online benchmark, td3 or ga stage')
    if args.stage == 'eval' and not args.checkpoint:
        parser.error(f'{args.stage} requires --checkpoint or --resume')
    if args.stage in ('td3', 'ga') and not args.demos_from:
        parser.error(f'{args.stage} requires --demos-from')
    if args.qdg_dataset and args.stage == 'demo' and args.expert != 'planner':
        parser.error('QDGset demo collection in train_ga.py requires --expert planner')
    if args.qdg_dataset and args.stage not in ('demo', 'eval') and not args.demos_from:
        parser.error('Pass --demos-from when training on a QDGset dataset')
    if args.stage == 'ga' and args.aux_weight <= 0:
        parser.error('ga requires a positive --aux-weight')
    if args.async_envs and (args.stage not in ('benchmark', 'td3', 'ga') or args.trace):
        parser.error('--async-envs requires online training/benchmarking and cannot be combined with --trace')
    if args.save_every < 0 or (args.checkpoint and args.resume):
        parser.error('provide a nonnegative save interval and only one checkpoint for initialization')
    if args.resume or args.save_every or args.stop_after is not None:
        if args.stage not in ('benchmark', 'td3', 'ga') or not args.demos_from:
            parser.error('full checkpointing requires an online stage and --demos-from')
        if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
            parser.error('--stop-after must be between 1 and the training horizon')
        if not args.async_envs and (args.steps % args.workers or
                (args.stop_after is not None and args.stop_after % args.workers)):
            parser.error('synchronized checkpoint steps must align with worker batches')
    if args.trace and (args.resume or args.stop_after is not None):
        parser.error('--trace requires an uninterrupted run')
    if args.update_every_env_steps is not None and not (
            0 < args.update_every_env_steps <= min(args.steps, args.workers)):
        parser.error('--update-every-env-steps must be between 1 and the worker count and step budget')
    if args.benchmark_seconds is not None and (
            args.stage != 'benchmark' or not math.isfinite(args.benchmark_seconds)
            or args.benchmark_seconds <= 0):
        parser.error('--benchmark-seconds requires benchmark and a positive finite duration')
    return args


def evaluate(pool, count, per_kind, seed, predict, kinds=OBJECTS):
    trials = [(seed + 10000 * i + j, kind)
              for i, kind in enumerate(kinds) for j in range(per_kind)]
    results = []
    for start in range(0, len(trials), count):
        active = dict(enumerate(trials[start:start+count]))
        obs = pool.call('reset', active.values(), active)
        while active:
            actions = predict(np.stack(obs))
            replies = pool.call('eval', actions, active)
            waiting = {}
            obs = []
            for (idx, (trial_seed, kind)), (next_obs, _, info, _) in zip(
                    active.items(), replies, strict=True):
                if info is None:
                    waiting[idx] = (trial_seed, kind)
                    obs.append(next_obs)
                else:
                    results.append(dict(info, seed=trial_seed))
            active = waiting
    return results


def collect_demos(pool, args, log, kinds=OBJECTS):
    rng = np.random.default_rng(args.seed)
    demos = Replay(args.demo_trials * args.horizon)
    records = []
    start = perf_counter()
    if kinds == OBJECTS:
        conditions = [(kind, tilt) for kind in OBJECTS for tilt in TILTS]
        schedule = [conditions[i % len(conditions)] for i in range(args.demo_trials)]
        rng.shuffle(schedule)
    else:
        ordered = list(kinds)
        rng.shuffle(ordered)
        schedule = [(ordered[i % len(ordered)], TILTS[i % len(TILTS)]) for i in range(args.demo_trials)]
    for offset in range(0, len(schedule), args.workers):
        block = schedule[offset:offset+args.workers]
        trials = [(args.seed + 10000 + offset + j, kind, tilt)
                  for j, (kind, tilt) in enumerate(block)]
        for (seed, kind, tilt), (rows, info) in zip(
                trials, pool.call('demo', trials, range(len(trials))), strict=True):
            demos.append(rows)
            records.append(dict(info, seed=seed, requested_tilt_deg=tilt))
    successes = [row for row in records if row['success']]
    tilted = sum((r['tilt_at_close_deg'] or 0) >= 15 for r in successes)
    (args.out/'demo_trials.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    demos.save(args.out/'demos.npz')
    log('demo', trials=len(records), successes=len(successes), tilted_successes=tilted,
        saved_transitions=demos.size, env_per_s=sum(r['steps'] for r in records) / (perf_counter()-start),
        visible_fraction=float(np.mean([r['visible_fraction'] for r in records])))
    if args.stage != 'demo' and not demos.size:
        raise RuntimeError('No verified demonstration succeeded. Inspect demo_trials.json.')
    return demos


def main():
    args = arguments()
    train_assets = eval_assets = None
    if args.qdg_dataset:
        from qdg_env import dataset_signature, load_manifest
        args.qdg_sha256 = dataset_signature(args.qdg_dataset)
        if args.stage == 'eval':
            eval_assets = load_manifest(args.qdg_dataset, 'holdout')
        else:
            train_assets = load_manifest(args.qdg_dataset, 'train')
            keys = sorted(train_assets)
            chosen = np.random.default_rng(args.eval_seed).choice(
                keys, size=min(len(keys), args.qdg_validation_objects), replace=False)
            eval_assets = {key: train_assets[key] for key in chosen}
    if args.stage != 'demo':
        import torch
        from ga_model import Learner

        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the project environment.')
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('high')
        device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(vars(args), default=str))
    (args.out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    start = perf_counter()
    if args.logger == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter

        run = SummaryWriter(str(args.tensorboard_dir/PROJECT/args.out.name), flush_secs=5)
    else:
        os.environ['TRACKIO_DIR'] = str(ROOT/'robotic-grasping'/'results'/'trackio')
        import trackio

        run = trackio.init(project=PROJECT, name=args.out.name, group=args.stage,
                           config=config, embed=False)

    def _is_perf_metric(key):
        return (key.endswith(('_s', '_seconds', '_per_s'))
                or key.startswith(('async_', 'gpu_', 'torch_', 'sim_', 'online_window_'))
                or key in ('graph_replays', 'mean_steps'))

    def log(event, **metrics):
        elapsed = perf_counter() - start
        row = json.dumps(dict(event=event, seconds=elapsed, **metrics), allow_nan=False)
        with (args.out/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(row + '\n')
        print(row, flush=True)
        if args.logger == 'tensorboard':
            global_step = int(metrics.get('step', metrics.get('steps', metrics.get('update', 0))))
            for key, value in metrics.items():
                if key in ('step', 'steps', 'update'):
                    continue
                prefix = 'perf' if _is_perf_metric(key) else event
                tag = f'{prefix}/{event}_{key}' if prefix == 'perf' and event in ('bc', 'demo') else f'{prefix}/{key}'
                run.add_scalar(tag, value, global_step)
            run.add_scalar(f'perf/{event}_elapsed_seconds', elapsed, global_step)
            run.flush()
        else:
            run.log(dict({f'{event}/{k}': v for k, v in metrics.items()}, elapsed_seconds=elapsed))

    resumable = bool(args.resume or args.save_every or args.stop_after is not None)

    def make_pool(resources, count, seed, gui=False, evaluation=False):
        pool = PointPool(count, seed, args.horizon, gui=gui, profile=args.profile_workers,
                         renderer=args.renderer, resumable=resumable,
                         assets=eval_assets if evaluation else train_assets,
                         planner_cfg=dict(ratio=args.dagger_ratio, min_step=args.dagger_min_step,
                                          max_step=args.dagger_max_step, expert=args.expert,
                                          online_goals=not args.no_online_goals))
        resources.callback(pool.close)
        return pool

    try:
        with ExitStack() as resources:
            if args.stage == 'demo':
                kinds = tuple(train_assets) if train_assets is not None else OBJECTS
                collect_demos(make_pool(resources, args.workers, args.seed), args, log, kinds=kinds)
                return
            rng = np.random.default_rng(args.seed)
            learner = Learner(device, args.lr,
                              args.aux_weight if args.stage in ('ga', 'benchmark') else 0.,
                              cuda_graph=args.cuda_graph, lr_warmup_updates=args.lr_warmup_updates,
                              lr_schedule=((args.lr_schedule[0], args.lr, *args.lr_schedule[1:], args.steps)
                                           if args.lr_schedule else None),
                              env_steps_per_update=(args.update_every_env_steps
                                                    if args.update_every_env_steps is not None
                                                    else args.workers / args.updates_per_batch),
                              gamma=args.gamma, tau=args.tau, ddpg_weight=args.ddpg_weight,
                              lr_milestones=args.lr_milestones, lr_gamma=args.lr_gamma)
            if args.checkpoint:
                saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
                learner.actor.load_state_dict(saved['actor'])
                # BC provides the same actor initialization for both RL arms.
                if 'critic' in saved:
                    learner.critic.load_state_dict(saved['critic'])
                learner.critic_target.load_state_dict(learner.critic.state_dict())

            @torch.no_grad()
            def predict(obs, timing=None):
                tick = perf_counter() if timing is not None else 0.
                training = learner.actor.training
                learner.actor.eval()
                try:
                    actions = learner.actor(torch.as_tensor(obs, device=device))[0]
                finally:
                    learner.actor.train(training)
                if timing is not None:
                    timing['actor_submit_s'] += perf_counter() - tick
                    tick = perf_counter()
                values = actions.cpu().numpy()
                if timing is not None:
                    timing['action_readback_s'] += perf_counter() - tick
                return values

            def save(name, step):
                payload = dict(actor=learner.actor.state_dict(), steps=step, stage=args.stage)
                if args.stage in ('benchmark', 'td3', 'ga'):
                    payload['critic'] = learner.critic.state_dict()
                temporary = args.out/(name + '.tmp')
                torch.save(payload, temporary)
                temporary.replace(args.out/name)

            evaluator = None

            def validate(step, *, event='eval'):
                nonlocal evaluator
                kinds = tuple(eval_assets) if eval_assets is not None else OBJECTS
                eval_count = min(args.workers, 6, len(kinds)*args.eval_per_kind)
                if evaluator is None:
                    evaluator = make_pool(resources, eval_count, args.eval_seed, args.gui, evaluation=True)
                rows = evaluate(evaluator, eval_count, args.eval_per_kind, args.eval_seed, predict, kinds)
                prefix = 'bc-eval' if event == 'bc_eval' else 'eval'
                (args.out/f'{prefix}-{step}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                successes = sum(r['success'] for r in rows)
                tilted = sum(r['success'] and (r['tilt_at_close_deg'] or 0) >= 15 for r in rows)
                mean_reward = float(np.mean([r.get('reward', float(r['success'])) for r in rows]))
                mean_steps = float(np.mean([r.get('steps', 0) for r in rows]))
                breakdown = {f'{kind}_rate': sum(r['success'] for r in rows if r['kind'] == kind)
                             / args.eval_per_kind for kind in OBJECTS}
                if eval_assets is not None:
                    breakdown = {'objects': len(kinds),
                                 'objects_with_success': len({r['kind'] for r in rows if r['success']})}
                log(event, step=step, success_rate=successes/len(rows), successes=successes,
                    trials=len(rows), tilted_successes=tilted, mean_reward=mean_reward,
                    mean_steps=mean_steps,
                    visible_fraction=float(np.mean([r['visible_fraction'] for r in rows])),
                    **breakdown)
                return successes

            if args.stage == 'eval':
                validate(saved['steps'])
                return

            pool = None
            if args.demos_from:
                demos = Replay.load(args.demos_from)
            else:
                pool = make_pool(resources, args.workers, args.seed)
                demos = collect_demos(pool, args, log)
            if not (args.checkpoint or args.resume):
                started = perf_counter()
                bc_losses = []
                for step in range(1, args.bc_updates + 1):
                    loss = learner.pretrain(batch_from(torch, device, rng, demos, None, args.batch))
                    bc_losses.append(loss.item())
                    if step % 500 == 0 or step == args.bc_updates:
                        log('bc', update=step, loss=bc_losses[-1],
                            loss_avg=float(np.mean(bc_losses[-500:])),
                            updates_per_s=step/(perf_counter()-started))
                    if args.stage != 'bc' and args.bc_eval_every > 0 and step % args.bc_eval_every == 0:
                        validate(step, event='bc_eval')
            if args.stage == 'bc':
                save('bc.pt', args.bc_updates)
                validate(args.bc_updates)
                return

            learner.actor_target.load_state_dict(learner.actor.state_dict())
            if pool is None:
                pool = make_pool(resources, args.workers, args.seed)
            all_indices = range(args.workers)
            best = -1
            next_eval = args.eval_every
            start_step = 0
            stop_step = args.stop_after if args.stop_after is not None else args.steps
            online = Replay(args.online_capacity)
            pending_labels = [[] for _ in all_indices]
            checkpoint_config = signature(args, torch, device) if resumable else None
            if args.resume:
                resumed = load_training(args.resume, torch, learner, device, rng, online, checkpoint_config)
                start_step, best, next_eval = resumed['steps'], resumed['best'], resumed['next_eval']
                if start_step >= stop_step:
                    raise ValueError('Resume checkpoint already reached the requested stop step.')
                # Resume learning from fresh episodes. Historical transitions retain
                # their original next observations and goal-validity masks; new
                # episodes must never backfill goals into the interrupted episodes.
                obs = np.stack(pool.call('reset', [
                    (args.seed + 100000 + start_step * args.workers + i, None)
                    for i in all_indices], all_indices))
                log('resume', step=start_step, updates=learner.updates, replay_size=online.size,
                    environments_reinitialized=args.workers)
            else:
                obs = np.stack(pool.call('reset', [(args.seed + 100000 + i, None)
                                                  for i in all_indices], all_indices))
                if args.stage != 'benchmark':
                    best = validate(0)
                    save('best.pt', 0)
            initial_updates = learner.updates
            next_save = ((start_step // args.save_every + 1) * args.save_every) if args.save_every else stop_step
            online_start = perf_counter()
            monitor = (GPUMonitor(online_start, args.out/'gpu-samples.jsonl' if args.profile else None)
                       if device.type == 'cuda' else None)
            if monitor is not None:
                resources.callback(monitor.close)
                torch.cuda.reset_peak_memory_stats()
            deadline = (online_start + args.benchmark_seconds
                        if args.benchmark_seconds is not None else math.inf)

            def report_benchmark(step, train_seconds):
                log('benchmark', steps=step, updates=learner.updates,
                    graph_replays=learner.graph_replays, segment_steps=step-start_step,
                    env_per_s=(step-start_step)/train_seconds,
                    updates_per_s=(learner.updates-initial_updates)/train_seconds,
                    online_wall_s=perf_counter()-online_start)
                save('benchmark.pt', step)

            if args.async_envs:
                from ga_async import train_async

                def gate_step():
                    return min(stop_step, next_save if resumable else stop_step,
                               next_eval if args.stage != 'benchmark' else stop_step)

                def on_boundary(step, finished):
                    nonlocal best, next_eval, next_save
                    if args.stage != 'benchmark' and (step >= next_eval or step == args.steps):
                        score = validate(step)
                        save('latest.pt', step)
                        if score > best:
                            best = score
                            save('best.pt', step)
                        next_eval = (step // args.eval_every + 1) * args.eval_every
                    if resumable and (step >= next_save or finished):
                        workers = pool.call('snapshot', [None]*args.workers, all_indices)
                        for worker, ids in zip(workers, pending_labels, strict=True):
                            worker['replay_ids'] = ids
                        save_training(args.out/'training-state.pt', torch, learner, device, rng, online,
                                      workers, obs, checkpoint_config, step, best, next_eval,
                                      keep_steps=args.keep_checkpoints)
                        log('checkpoint', step=step, updates=learner.updates)
                        if args.save_every:
                            next_save = (step // args.save_every + 1) * args.save_every
                    return gate_step()

                def sample(timing):
                    return batch_from(torch, device, rng, demos, online, args.batch,
                                      timing if args.profile else None)

                step, train_seconds = train_async(
                    args, torch, device, learner, pool, obs, rng, online, sample, monitor, log,
                    online_start, deadline, start_step, stop_step, gate_step(), pending_labels, on_boundary)
                if args.stage == 'benchmark':
                    report_benchmark(step, train_seconds)
                return

            window_start = perf_counter()
            window_steps = window_updates = 0
            window_returns = []
            window_episodes = []
            policy_seconds = update_seconds = wait_seconds = 0.
            train_seconds = 0.
            timing = defaultdict(float) if args.profile else None
            worker_timing = defaultdict(float)
            profiler = None
            if args.trace:
                profiler = resources.enter_context(torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=min(100, (stop_step - start_step) // args.workers - 12),
                                                     warmup=4, active=8, repeat=1),
                    on_trace_ready=lambda p: p.export_chrome_trace(str(args.out/'cuda-trace.json'))))
            for prior in range(start_step, stop_step, args.workers):
                count = min(args.workers, stop_step - prior)
                step = prior + count
                period = args.update_every_env_steps
                batch_updates = (step // period - prior // period
                                 if period is not None else args.updates_per_batch)
                indices = range(count)
                tick = perf_counter()
                actions = predict(obs[:count], timing)
                noise = max(0.07, 0.20 - 0.13 * prior / args.steps)
                actions = np.clip(actions + rng.normal(0., noise, actions.shape), -1., 1.).astype(np.float32)
                pool.send('train', actions, indices)
                policy_seconds += perf_counter() - tick
                tick = perf_counter()
                # Run batched CUDA updates while the CPU processes simulate and render.
                for _ in range(batch_updates):
                    batch = batch_from(torch, device, rng, demos, online, args.batch, timing)
                    if timing is not None:
                        update_start = perf_counter()
                    if profiler is None:
                        losses = learner.update(batch)
                    else:
                        with torch.profiler.record_function('ga.network_update'):
                            losses = learner.update(batch)
                    if timing is not None:
                        timing['network_call_s'] += perf_counter() - update_start
                update_seconds += perf_counter() - tick
                tick = perf_counter()
                replies = pool.receive(indices)
                received = perf_counter()
                wait_seconds += received - tick
                if args.profile_workers:
                    timings = [row[4] for row in replies]
                    for _, spans, resetting in timings:
                        worker_timing['sim_steps'] += 1
                        worker_timing['sim_resets'] += int(resetting)
                        for key, seconds in spans.items():
                            worker_timing[f'sim_{key}_s'] += seconds
                    ready, tail_spans, reset_tail = max(timings, key=lambda row: row[0])
                    worker_timing['sim_batches'] += 1
                    worker_timing['sim_tail_reset_batches'] += int(reset_tail)
                    worker_timing['sim_after_tail_ready_s'] += max(0., received - ready)
                    for key, seconds in tail_spans.items():
                        worker_timing[f'sim_tail_{key}_s'] += seconds
                    replies = [row[:4] for row in replies]
                obs[:count] = np.stack([reply[0] for reply in replies])
                for _, reward, ended, rows in replies:
                    online.append(rows)
                    window_returns.append(reward)
                    if ended is not None:
                        window_episodes.append(ended['success'])
                budget_elapsed = perf_counter() >= deadline
                last_batch = step == stop_step or budget_elapsed
                window_steps += count
                window_updates += batch_updates
                should_eval = args.stage != 'benchmark' and (step >= next_eval or step == args.steps)
                should_save = resumable and (step >= next_save or last_batch)
                if window_steps >= args.log_every or should_eval or should_save or last_batch:
                    packed_losses = torch.stack(list(losses.values())).cpu().tolist()
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    elapsed = perf_counter() - window_start
                    train_seconds += elapsed
                    gpu_stats = monitor.read() if monitor is not None else {}
                    if monitor is not None:
                        gpu_stats['torch_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
                    episode_stats = {'episodes': len(window_episodes)}
                    if window_episodes:
                        episode_stats['rollout_success_rate'] = float(np.mean(window_episodes))
                    log('train', step=step, updates=learner.updates,
                        online_window_start_s=window_start-online_start,
                        online_window_end_s=window_start+elapsed-online_start,
                        graph_replays=learner.graph_replays,
                        env_per_s=window_steps/elapsed, updates_per_s=window_updates/elapsed,
                        step_reward_mean=float(np.mean(window_returns)), replay_size=online.size,
                        policy_seconds=policy_seconds, update_submit_seconds=update_seconds,
                        worker_wait_seconds=wait_seconds, **dict(zip(losses, packed_losses)),
                        **gpu_stats, **episode_stats, **(timing if timing is not None else {}),
                        **worker_timing)
                    if should_eval:
                        score = validate(step)
                        save('latest.pt', step)
                        if score > best:
                            best = score
                            save('best.pt', step)
                        next_eval = (step // args.eval_every + 1)*args.eval_every
                    if should_save:
                        workers = pool.call('snapshot', [None]*args.workers, all_indices)
                        save_training(args.out/'training-state.pt', torch, learner, device, rng, online,
                                      workers, obs, checkpoint_config, step, best, next_eval,
                                      keep_steps=args.keep_checkpoints)
                        log('checkpoint', step=step, updates=learner.updates)
                        if args.save_every:
                            next_save = (step // args.save_every + 1) * args.save_every
                    if monitor is not None:
                        monitor.read()
                    window_start = perf_counter()
                    window_steps = window_updates = 0
                    policy_seconds = update_seconds = wait_seconds = 0.
                    window_returns.clear()
                    window_episodes.clear()
                    if timing is not None:
                        timing.clear()
                    worker_timing.clear()
                if profiler is not None:
                    profiler.step()
                if budget_elapsed:
                    break
            if args.stage == 'benchmark':
                report_benchmark(step, train_seconds)
    finally:
        if args.logger == 'tensorboard':
            run.close()
        else:
            run.finish()


if __name__ == '__main__':
    main()
