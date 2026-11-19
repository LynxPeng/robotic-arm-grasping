"""Train a 224x224 RGB QT-Opt-style grasp policy with continuous yaw."""

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--demos', type=int, default=9000)
    p.add_argument('--warmup', type=int, default=2400)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--updates-per-tick', type=int, default=5)
    p.add_argument('--eval-every', type=int, default=12000)
    p.add_argument('--reward-log-every', type=int, default=160,
                   help='Log statistics over this many environment steps (0 disables)')
    p.add_argument('--eval-per-kind', type=int, default=20)
    p.add_argument('--time-budget', type=float, default=0.,
                   help='Wall-clock seconds including validation (0 disables)')
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--eval-seed', type=int, default=300001)
    p.add_argument('--horizon', type=int, default=100)
    p.add_argument('--population', type=int, default=96)
    p.add_argument('--target-population', type=int, default=48)
    p.add_argument('--online-replay', type=int, default=14000)
    p.add_argument('--lr', type=float, default=0.00025)
    p.add_argument('--group', default='train')
    p.add_argument('--rank-mode', choices=('bounded', 'legacy'), default='bounded')
    p.add_argument('--skip-untrained', action='store_true')
    p.add_argument('--proposal', action='store_true')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--checkpoint', type=Path, help='Evaluate an existing policy')
    source.add_argument('--warm-start', type=Path, help='Initialize a new training run from model weights')
    p.add_argument('--proposal-only', action='store_true',
                   help='Evaluate the learned proposal directly for a CEM ablation')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--kind', choices=('cylinder', 'cube', 'long_box'))
    args = p.parse_args()
    if min(args.workers, args.batch, args.demos, args.online_replay, args.eval_every) < 2:
        p.error('workers, batch, demos, online replay and evaluation interval must be at least 2')
    if args.time_budget and args.time_budget <= 35:
        p.error('--time-budget must leave more than 35 seconds for validation')
    if args.reward_log_every < 0:
        p.error('--reward-log-every must be nonnegative')
    if args.proposal_only and not args.checkpoint:
        p.error('--proposal-only requires --checkpoint')
    if args.gui and not (args.checkpoint and args.kind):
        p.error('--gui requires --checkpoint and --kind')
    return args


def main(args):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    from torch.nn import functional as F
    from qt_rgb_env import SIZE, AUX_DIM, ACT_DIM, OBJECTS, RGBGraspEnv
    from qt_rgb_workers import RGBPool
    from qt_rgb_model import RGBQ, ImageReplay, select

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    if args.checkpoint or args.warm_start:
        saved = torch.load(args.checkpoint or args.warm_start,map_location=device,weights_only=True)
        args.proposal = saved.get('proposal',False)
        args.population = saved['population']
        args.horizon = saved['horizon']
    if args.proposal_only and not args.proposal:
        raise ValueError('the checkpoint does not contain a learned action proposal')
    explore_start, explore_end, explore_sigma = (
        (.18, .06, .25) if args.warm_start else (.35, .15, .42)
    )
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    config = dict(vars(args), image_size=SIZE, action_size=ACT_DIM, objects=OBJECTS,
                  initialization_steps=saved['steps'] if args.warm_start else None,
                  exploration=[explore_start, explore_end, explore_sigma],
                  device=torch.cuda.get_device_name(), proprio=['opening','height','sin_yaw','cos_yaw'])
    config = json.loads(json.dumps(config, default=str))
    (args.out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    trackio_dir = Path(__file__).resolve().parents[1] / 'robotic-grasping' / 'results' / 'trackio'
    os.environ['TRACKIO_DIR'] = str(trackio_dir)
    import trackio

    run = trackio.init(project='qt-opt-rgb', name=args.out.name, group=args.group,
                       config=config, embed=False)

    def log(event, *, echo=True, **data):
        elapsed = round(time.perf_counter()-start, 3)
        row = dict(event=event, seconds=elapsed, **data)
        text = json.dumps(row, allow_nan=False)
        with (args.out/'metrics.jsonl').open('a', encoding='utf-8') as fp:
            fp.write(text+'\n')
        if echo:
            print(text, flush=True)
        metrics = {f'{event}/{key}': value for key,value in data.items() if isinstance(value,(float,int))}
        run.log(dict(metrics, elapsed_seconds=elapsed))

    def gpu_metrics():
        command = ['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','--id=0']
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            utilization, memory, power = [float(x) for x in result.stdout.strip().split(',')]
            return dict(gpu_util_pct=utilization, gpu_mem_mib=memory, gpu_power_w=power)
        except (OSError, subprocess.SubprocessError, ValueError):
            return {}

    @torch.no_grad()
    def act(model, frames, aux):
        frame = torch.from_numpy(frames).to(device)
        proprio = torch.from_numpy(aux).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            feature = model.encode(frame, proprio)
            action = (model.proposal(feature) if args.proposal_only
                      else select(model, feature, args.population, 3))
        return action.float().cpu().numpy()

    def evaluate(model, pool, step):
        started = time.perf_counter()
        kinds = (args.kind,) if args.kind else tuple(OBJECTS)
        jobs = [(args.eval_seed + i*100000 + j, kind)
                for i,kind in enumerate(kinds) for j in range(args.eval_per_kind)]
        results = []
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(args.eval_seed)
            if args.gui:
                env = RGBGraspEnv(gui=True, max_steps=args.horizon)
                try:
                    for seed,kind in jobs:
                        frame, aux = env.reset(seed,kind)
                        for _ in range(args.horizon):
                            action = act(model, frame[None], aux[None])[0]
                            (frame,aux), _, term, trunc, info = env.step(action)
                            time.sleep(1/24)
                            if term or trunc:
                                results.append(dict(info,seed=seed))
                                break
                finally:
                    env.close()
            else:
                iterator = iter(jobs)
                active, obs = {}, {}
                for index,(pipe,job) in enumerate(zip(pool.pipes, iterator)):
                    active[index] = job
                    pipe.send(('reset',job))
                for index in active:
                    obs[index] = pool.pipes[index].recv()[0]
                while active:
                    indices = list(active)
                    actions = act(model, pool.frames[indices], np.stack([obs[i] for i in indices]))
                    for index,action in zip(indices,actions,strict=True):
                        pool.pipes[index].send(('eval',action))
                    resetting = []
                    for index in indices:
                        pipe = pool.pipes[index]
                        aux, _, ended, _, _ = pipe.recv()
                        if ended is None:
                            obs[index] = aux
                            continue
                        results.append(dict(ended,seed=active[index][0]))
                        job = next(iterator,None)
                        if job is None:
                            del active[index]
                        else:
                            active[index] = job
                            pipe.send(('reset',job))
                            resetting.append(index)
                    for index in resetting:
                        obs[index] = pool.pipes[index].recv()[0]
        assert len(results) == len(jobs)
        summary = {}
        for kind in kinds:
            records = [r for r in results if r['kind'] == kind]
            successes = sum(r['success'] for r in records)
            summary.update({f'{kind}_successes': successes, f'{kind}_episodes': len(records),
                            f'{kind}_rate': successes/len(records)})
        success = sum(r['success'] for r in results)
        log('eval', steps=step, successes=success, episodes=len(results),
            success_rate=success/len(results), eval_seconds=round(time.perf_counter()-started,3), **summary)
        (args.out/f'eval-{step}.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        return success

    online = RGBQ(proposal=args.proposal).to(device, memory_format=torch.channels_last)
    pool = evaluator = None
    try:
        if args.checkpoint:
            online.load_state_dict(saved['model'])
            if not args.gui:
                evaluator = RGBPool(min(args.workers, 12),args.eval_seed,args.horizon)
            evaluate(online,evaluator,saved['steps'])
            return

        if args.warm_start:
            online.load_state_dict(saved['model'])
        target = copy.deepcopy(online).requires_grad_(False)
        target2 = copy.deepcopy(online).requires_grad_(False)
        optimizer = torch.optim.AdamW(online.parameters(), lr=args.lr, fused=True)
        pool = RGBPool(args.workers,args.seed,args.horizon)
        demos = ImageReplay(math.ceil(args.demos/args.workers)*args.workers)
        replay = ImageReplay(args.online_replay)
        batch = args.batch
        buffers = [
            ([torch.empty((batch,SIZE,SIZE,3),dtype=torch.uint8,pin_memory=True) for _ in range(2)] +
             [torch.empty((batch,2*AUX_DIM+ACT_DIM+2),dtype=torch.float32,pin_memory=True)],
             torch.cuda.Event()) for _ in range(2)
        ]
        updates = 0
        last_loss = None

        def update(online_fraction):
            nonlocal updates,last_loss
            staged,event = buffers[updates%2]
            if updates >= len(buffers):
                event.synchronize()
            demo_count = round(batch*(1-online_fraction))
            demos.sample_into(0,demo_count,staged,rng)
            if demo_count < batch:
                replay.sample_into(demo_count,batch,staged,rng)
            img,nxt,metadata = [v.to(device,non_blocking=True) for v in staged]
            event.record()
            aux,a,next_aux,r,done = metadata.split((AUX_DIM,ACT_DIM,AUX_DIM,1,1),dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                with torch.no_grad():
                    f1 = target.encode(nxt,next_aux)
                    f2 = target2.encode(nxt,next_aux)
                    chosen = select(target,f1,args.target_population,2)
                    value = torch.minimum(target.score(f1,chosen),target2.score(f2,chosen)).float()
                    y = r[:,0] + 0.985*torch.where(done[:,0].bool(),0.,value.clamp(-0.8,1.5))
                f = online.encode(img,aux)
                q = online.score(f,a).float()
                td = F.smooth_l1_loss(q,y,beta=0.15)
                # Regularize the action ranking on demonstration states.
                n = min(demo_count,batch//2)
                uniform = torch.rand((n,12,ACT_DIM),device=device)*2-1
                nearby = (a[:n,None] + 0.55*torch.randn_like(uniform)).clamp(-1,1)
                alternatives = online.score(f[:n],torch.cat((uniform,nearby),1)).float()
                temperature = .05
                margin = temperature*torch.logsumexp(alternatives/temperature,dim=1)-q[:n]
                if args.rank_mode == 'bounded':
                    # Stop rewarding the model for pushing already-low alternatives further down.
                    margin = temperature*F.softplus(margin/temperature)
                rank = margin.mean()
                loss = td + .55*rank
                if online.proposal is not None:
                    # The learned center limits out-of-distribution CEM queries.
                    # The action actually executed still maximizes the critic within that region.
                    center = online.proposal(f[:demo_count]).float()
                    imitation = F.smooth_l1_loss(center,a[:demo_count],beta=.1)
                    loss = loss + 1.5*imitation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(),5.0)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(target.parameters()),list(online.parameters()),.005)
                torch._foreach_lerp_(list(target2.parameters()),list(online.parameters()),.001)
            last_loss = loss.detach()
            updates += 1

        def save(filename,step,score):
            path = args.out/filename
            temporary = path.with_suffix('.tmp')
            torch.save(dict(model=online.state_dict(), steps=step, eval_successes=score,
                            population=args.population,horizon=args.horizon,proposal=args.proposal,
                            warm_start_steps=saved['steps'] if args.warm_start else None),temporary)
            temporary.replace(path)

        demo_start = time.perf_counter()
        count = Counter()
        success = Counter()
        for _ in range(math.ceil(args.demos/args.workers)):
            before, aux = pool.frames.copy(),pool.aux
            pool.send('expert',[0 if i%2==0 else .20 for i in range(args.workers)])
            rewards, ended, action, _ = pool.receive()
            done = np.array([item is not None for item in ended])
            demos.append(before,aux,action,pool.frames,pool.aux,rewards,done)
            for item in ended:
                if item is not None:
                    count[item['kind']]+=1
                    success[item['kind']]+=int(item['success'])
        log('demo', transitions=demos.count, mean_steps_per_s=demos.count/(time.perf_counter()-demo_start),
            **{f'{kind}_episodes': count[kind] for kind in OBJECTS},
            **{f'{kind}_successes': success[kind] for kind in OBJECTS})

        evaluator = RGBPool(min(9,3*args.eval_per_kind),args.eval_seed,args.horizon)
        if not (args.skip_untrained or args.warm_start):
            evaluate(online,evaluator,-1)
        warm_start = time.perf_counter()
        for i in range(args.warmup):
            update(0.)
            if (i+1)%250==0 or i+1==args.warmup:
                torch.cuda.synchronize()
                log('warmup', updates=updates, loss=float(last_loss),
                    updates_per_s=updates/(time.perf_counter()-warm_start), **gpu_metrics())
        best = evaluate(online,evaluator,0)
        save('best.pt',0,best)
        save('latest.pt',0,best)
        # Start the online phase from freshly randomized scenes.
        pool.send('reset',[(None,None)]*args.workers)
        pool.aux = np.stack([pipe.recv()[0] for pipe in pool.pipes])
        last_time,last_step,last_updates = time.perf_counter(),0,updates
        last_eval = 0
        result = Counter()
        episodes = Counter()
        timing = Counter()
        reward_batches, reward_kinds = [], []
        last_reward_step = 0
        step = 0
        stop_reason = 'step_budget'
        for step in range(args.workers,args.steps+args.workers,args.workers):
            before, aux = pool.frames.copy(),pool.aux
            tick = time.perf_counter()
            action = act(online,before,aux)
            timing['action_seconds'] += time.perf_counter()-tick
            progress = min(1.,step/args.steps)
            mask = rng.random(args.workers) < explore_start + (explore_end-explore_start)*progress
            action[mask] = np.clip(action[mask]+rng.normal(0,explore_sigma,(mask.sum(),ACT_DIM)),-1,1)
            pool.send('step',action)
            tick = time.perf_counter()
            if replay.count >= batch:
                for _ in range(args.updates_per_tick):
                    update(.35)
            timing['update_submit_seconds'] += time.perf_counter()-tick
            tick = time.perf_counter()
            rewards,ended,executed,kinds = pool.receive()
            timing['sim_wait_seconds'] += time.perf_counter()-tick
            if args.reward_log_every:
                reward_batches.append(rewards)
                reward_kinds.extend(kinds)
                if step-last_reward_step >= args.reward_log_every or step >= args.steps:
                    window = np.concatenate(reward_batches)
                    kind_array = np.asarray(reward_kinds)
                    by_kind = {}
                    for kind in OBJECTS:
                        values = window[kind_array == kind]
                        by_kind[f'{kind}_count'] = len(values)
                        if len(values):
                            by_kind[f'{kind}_mean'] = float(values.mean())
                    log('step_reward', echo=False, steps=step, count=len(window),
                        mean=float(window.mean()), minimum=float(window.min()),
                        maximum=float(window.max()), positive_fraction=float((window>0).mean()),
                        **by_kind)
                    reward_batches.clear()
                    reward_kinds.clear()
                    last_reward_step = step
            done = np.array([item is not None for item in ended])
            # Auto-reset frames can serve as next images here because terminal targets are masked.
            replay.append(before,aux,executed,pool.frames,pool.aux,rewards,done)
            for item in ended:
                if item is not None:
                    episodes[item['kind']]+=1
                    result[item['kind']]+=int(item['success'])
            time_limit = bool(args.time_budget and time.perf_counter()-start >= args.time_budget-35)
            evaluate_now = step-last_eval>=args.eval_every or step>=args.steps or time_limit
            if step-last_step>=2000 or evaluate_now:
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-last_time
                log('train',steps=step,updates=updates,steps_per_s=(step-last_step)/elapsed,
                    updates_per_s=(updates-last_updates)/elapsed,
                    loss=float(last_loss) if last_loss is not None else 0.,
                    **{f'{kind}_episodes':episodes[kind] for kind in OBJECTS},
                    **{f'{kind}_successes':result[kind] for kind in OBJECTS},
                    **{key:round(value,3) for key,value in timing.items()}, **gpu_metrics())
                last_step,last_updates = step,updates
                result,episodes,timing = Counter(),Counter(),Counter()
                if evaluate_now:
                    score = evaluate(online,evaluator,step)
                    save('latest.pt',step,score)
                    if args.warm_start:
                        save(f'checkpoint-{step}.pt',step,score)
                    if score>best:
                        best=score
                        save('best.pt',step,score)
                    last_eval=step
                    if time_limit:
                        stop_reason = 'time_budget'
                        break
                last_time = time.perf_counter()
        log('complete',steps=step,updates=updates,best_successes=best,
            episodes=args.eval_per_kind*3, stop_reason=stop_reason)
    finally:
        for sim in (pool,evaluator):
            if sim is not None:
                sim.close()
        run.finish()


if __name__ == '__main__':
    main(options())
