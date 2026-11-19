"""Train a state-based QT-Opt-style Panda grasp policy, or evaluate a checkpoint."""

import argparse
import copy
import json
import math
import os
import subprocess
from threading import Event, Thread
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--eval-seed', type=int, default=100_001)
    parser.add_argument('--eval-episodes', type=int, default=48)
    parser.add_argument('--eval-every', type=int, default=24_000)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--demo-steps', type=int, default=24_000)
    parser.add_argument('--warmup-updates', type=int, default=1_200)
    parser.add_argument('--steps', type=int, default=120_000)
    parser.add_argument('--updates-per-tick', type=int, default=3)
    parser.add_argument('--target-population', type=int, default=64)
    parser.add_argument('--population', type=int, default=128)
    parser.add_argument('--demo-fraction', type=float, default=1/3)
    parser.add_argument('--scale-observations', action='store_true')
    parser.add_argument('--conservative-weight', type=float, default=0.0)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0003)
    args = parser.parse_args()
    if min(args.workers, args.horizon, args.batch, args.eval_episodes, args.eval_every,
           args.demo_steps, args.updates_per_tick) < 1 or min(args.population, args.target_population) < 8:
        parser.error('sizes must be positive and CEM populations must be at least eight')
    if min(args.steps, args.warmup_updates, args.conservative_weight) < 0 or args.batch < 3:
        parser.error('steps, warmup and conservative weight must be nonnegative, batch must be at least 3')
    if not 1/3 <= args.demo_fraction < 1:
        parser.error('demo fraction must be in [1/3, 1)')
    if args.gui and not args.checkpoint:
        parser.error('--gui requires --checkpoint')
    return args


def execute(args):
    # Independent physics processes do not need a BLAS thread pool each.
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

    import numpy as np
    import torch
    import torch.nn.functional as F

    from env import OBS_DIM, PandaGraspEnv, STEPS_PER_CTRL
    from qt_model import GraspQ, Replay, cem_action, td_target
    from qt_workers import Simulators

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run')
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    args.out.joinpath('config.json').write_text(json.dumps(vars(args), default=str, indent=2))

    def log(event, **data):
        record = dict(event=event, seconds=round(time.perf_counter() - started, 2), **data)
        line = json.dumps(record, allow_nan=False)
        with (args.out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
        print(line, flush=True)

    def act(model, obs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return cem_action(model, torch.from_numpy(obs).to(device), args.population).cpu().numpy()

    def evaluate(model, sim, step):
        results = []
        eval_start = time.perf_counter()
        # Preserve the training RNG trajectory across periodic evaluations.
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(args.eval_seed)
            if args.gui:
                env = PandaGraspEnv(gui=True, max_steps=args.horizon)
                try:
                    for episode in range(args.eval_episodes):
                        obs = env.reset(seed=args.eval_seed + episode)
                        for _ in range(args.horizon):
                            obs, _, term, trunc, info = env.step(act(model, obs[None])[0])
                            time.sleep(STEPS_PER_CTRL / 240)
                            if term or trunc:
                                results.append(dict(info, seed=args.eval_seed + episode))
                                break
                finally:
                    env.close()
            else:
                n = len(sim.pipes)
                for start in range(0, args.eval_episodes, n):
                    active = sim.pipes[:min(n, args.eval_episodes - start)]
                    seeds = {pipe: args.eval_seed + start + offset for offset, pipe in enumerate(active)}
                    for pipe in active:
                        pipe.send(('reset', seeds[pipe]))
                    obs = np.stack([pipe.recv() for pipe in active])
                    while active:
                        actions = act(model, obs)
                        for pipe, action in zip(active, actions, strict=True):
                            pipe.send(('eval', action))
                        continuing, next_obs = [], []
                        for pipe in active:
                            observation, _, done = pipe.recv()
                            if done is None:
                                continuing.append(pipe)
                                next_obs.append(observation)
                            else:
                                results.append(dict(done, seed=seeds[pipe]))
                        active = continuing
                        if active:
                            obs = np.stack(next_obs)
        successes = sum(result['success'] for result in results)
        (args.out / f'eval-{step}.json').write_text(json.dumps(results, indent=2))
        log('evaluation', steps=step, successes=successes, episodes=len(results),
            mean_lift_m=float(np.mean([r['lift'] for r in results])),
            mean_ep_steps=float(np.mean([r['steps'] for r in results])),
            eval_seconds=round(time.perf_counter()-eval_start, 2))
        return successes

    log('device', gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        args.horizon = checkpoint['horizon']
        args.population = checkpoint.get('population', args.population)
        model = GraspQ(OBS_DIM, checkpoint['width'], checkpoint.get('scale_observations', False)).to(device)
        model.load_state_dict(checkpoint['model'])
        sim = None if args.gui else Simulators(min(args.workers, args.eval_episodes), args.eval_seed, args.horizon)
        try:
            evaluate(model, sim, checkpoint['steps'])
        finally:
            if sim is not None:
                sim.close()
        return

    online = GraspQ(OBS_DIM, args.width, args.scale_observations).to(device)
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.AdamW(online.parameters(), lr=args.lr, fused=True, capturable=args.cuda_graph)
    sim = Simulators(args.workers, args.seed, args.horizon)
    replay = Replay(200_000, OBS_DIM * 2 + 6, device)
    demos = Replay(args.demo_steps + args.workers, OBS_DIM * 2 + 6, device)
    updates = 0
    last_loss = None

    static_rows = torch.empty((args.batch, OBS_DIM * 2 + 6), device=device)
    graph = None

    def gradient_step(rows):
        obs, action, nxt, reward, done = rows.split((OBS_DIM, 4, OBS_DIM, 1, 1), dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            with torch.no_grad():
                selected = cem_action(online, nxt, args.target_population, iterations=2)
                bootstrap = target(nxt, selected).float().clamp(-0.8, 1.5)
                y = td_target(reward.squeeze(1), done.squeeze(1).bool(), bootstrap, 0.985)
            q = online(obs, action).float()
            loss = F.smooth_l1_loss(q, y, beta=0.15)
            if args.conservative_weight:
                # The first third is always demonstration data. Penalize optimistic
                # scores on sampled alternatives at those same observations.
                n = args.batch // 3
                negatives = torch.rand((n, 16, 4), device=device) * 2 - 1
                near = (action[:n, None] + 0.5 * torch.randn_like(negatives)).clamp(-1, 1)
                candidates = torch.cat((negatives, near), dim=1)
                other_q = online(obs[:n], candidates).float()
                temperature = 0.05
                conservative = (temperature * other_q.div(temperature).logsumexp(1) - q[:n]).mean()
                loss = loss + args.conservative_weight * conservative
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(target.parameters()), list(online.parameters()), 0.007)
        return loss.detach()

    def update(use_online):
        nonlocal updates, last_loss, graph
        demo_count = min(args.batch - 1, int(args.batch * args.demo_fraction)) if use_online else args.batch
        rows = demos.sample(demo_count)
        if use_online:
            rows = torch.cat((rows, replay.sample(args.batch - demo_count)))
        if args.cuda_graph and updates >= 8:
            static_rows.copy_(rows)
            if graph is None:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    last_loss = gradient_step(static_rows)
                log('cuda_graph_captured', updates=updates)
            graph.replay()
        else:
            last_loss = gradient_step(rows)
        updates += 1

    def save(name, step, successes):
        path = args.out / name
        tmp = path.with_suffix('.tmp')
        torch.save(dict(model=online.state_dict(), width=args.width, horizon=args.horizon,
                        steps=step, updates=updates, eval_successes=successes, population=args.population,
                        scale_observations=args.scale_observations), tmp)
        tmp.replace(path)

    monitor_stop = Event()

    def monitor_gpu():
        query = ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                 '--format=csv,noheader,nounits', '--id=0']
        with (args.out / 'gpu.csv').open('w', encoding='utf-8', buffering=1) as f:
            f.write('seconds,gpu_util_pct,memory_util_pct,memory_used_mib,power_w\n')
            while not monitor_stop.is_set():
                try:
                    sample = subprocess.run(query, capture_output=True, text=True, timeout=3, check=True,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    f.write(f'{time.perf_counter()-started:.3f},' + sample.stdout.strip() + '\n')
                except (OSError, subprocess.SubprocessError):
                    return
                monitor_stop.wait(1)

    monitor = Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    evaluator = None
    try:
        successes = episodes = 0
        demo_started = time.perf_counter()
        for _ in range(math.ceil(args.demo_steps/args.workers)):
            sim.send('expert', [0.0 if i % 2 == 0 else 0.22 for i in range(args.workers)])
            rows, ended = sim.receive()
            demos.extend(torch.from_numpy(rows).to(device))
            successes += sum(r['success'] for r in ended)
            episodes += len(ended)
        log('demos', transitions=demos.count, successes=successes, episodes=episodes,
            transitions_per_second=round(demos.count/(time.perf_counter()-demo_started), 1))
        evaluator = Simulators(min(8, args.eval_episodes), args.eval_seed, args.horizon)
        evaluate(online, evaluator, 'untrained')
        for index in range(args.warmup_updates):
            update(False)
            if (index+1) % 250 == 0:
                log('warmup', updates=updates, loss=float(last_loss))
        best = evaluate(online, evaluator, 0)
        save('best.pt', 0, best)
        save('latest.pt', 0, best)
        next_eval = args.eval_every
        last_log = time.perf_counter()
        previous_step, previous_updates = 0, updates
        successes = episodes = 0
        for step in range(args.workers, args.steps + args.workers, args.workers):
            actions = act(online, sim.observations)
            epsilon = 0.45 - 0.3 * min(1.0, step / max(args.steps, 1))
            explore = rng.random(args.workers) < epsilon
            actions[explore] = np.clip(actions[explore] + rng.normal(0, 0.5, (explore.sum(), 4)), -1, 1)
            sim.send('step', actions)
            if replay.count >= args.batch:
                for _ in range(args.updates_per_tick):
                    update(True)
            rows, ended = sim.receive()
            replay.extend(torch.from_numpy(rows).to(device))
            successes += sum(r['success'] for r in ended)
            episodes += len(ended)
            if step - previous_step >= 4_000 or step >= next_eval or step >= args.steps:
                torch.cuda.synchronize()
                now = time.perf_counter()
                log('train', steps=step, updates=updates, episodes=episodes, successes=successes,
                    transitions_per_second=round((step-previous_step)/(now-last_log), 1),
                    updates_per_second=round((updates-previous_updates)/(now-last_log), 1),
                    loss=float(last_loss) if last_loss is not None else None,
                    cuda_peak_mib=round(torch.cuda.max_memory_allocated()/2**20))
                previous_step, previous_updates, last_log = step, updates, now
                successes = episodes = 0
            if step >= next_eval or step >= args.steps:
                score = evaluate(online, evaluator, step)
                save('latest.pt', step, score)
                if score > best:
                    best = score
                    save('best.pt', step, score)
                next_eval += args.eval_every
                last_log = time.perf_counter()
        log('complete', steps=step if args.steps else 0, updates=updates, best_eval_successes=best,
            eval_episodes=args.eval_episodes)
    finally:
        monitor_stop.set()
        monitor.join(timeout=4)
        sim.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    execute(parse_args())
