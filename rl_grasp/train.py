"""Train SAC with demo replay on the PyBullet grasp env; log to Trackio.

Minimal HIL-SERL-style recipe: SAC + 50/50 demonstration/online replay
(RLPD), closed-loop end-effector control from the home pose to lifting the
object. --bc-pretrain jump-starts the actor by imitating demos before RL
(escapes the reach-only local optimum); --utd sets gradient updates per env
step; --resume continues from state.pt (saved at every eval and on SIGTERM).
"""
import argparse
import json
import os
import signal
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from env import ACT_DIM, OBS_DIM, PandaGraspEnv, RESULTS_DIR
from sac import SAC

ROOT = Path(__file__).resolve().parents[1]


def evaluate(env, agent, episodes, seed):
    rows = []
    for i in range(episodes):
        obs = env.reset(seed=seed + i)
        done, ep_r, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, deterministic=True))
            ep_r += r
            done = term or trunc
        rows.append((float(info["success"]), ep_r, info["lift"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=60_000)
    ap.add_argument("--demo-path", type=Path, default=RESULTS_DIR / "demos.npz")
    ap.add_argument("--demo-ratio", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=512)
    ap.add_argument("--utd", type=int, default=2, help="gradient updates per env step")
    ap.add_argument("--bc-pretrain", type=int, default=2000,
                    help="BC warmstart steps on demos before RL (0 disables)")
    ap.add_argument("--bc-coef", type=float, default=30.0,
                    help="fixed-weight BC anchor during RL (0 disables)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--reward", choices=("shaped", "sparse"), default="shaped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--trackio-dir", type=Path, default=None)
    ap.add_argument("--project", default="grconvnet-jacquard")
    ap.add_argument("--group", default="rl-sac")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", type=Path, default=None, help="state.pt to continue from")
    ap.add_argument("--no-trackio", action="store_true")
    args = ap.parse_args()

    if not args.demo_path.is_file():
        raise SystemExit(f"Demo file not found: {args.demo_path}\n"
                         f"Run collect_demos.py first.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_name = args.run_name or f"rl-sac-{datetime.now():%Y%m%d-%H%M%S-%f}"
    out = args.out_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    trackio_dir = args.trackio_dir or Path(os.environ.get(
        "TRACKIO_DIR", str(ROOT / "robotic-grasping" / "results" / "trackio")))
    os.environ["TRACKIO_DIR"] = str(trackio_dir)

    env = PandaGraspEnv(reward_mode=args.reward, seed=args.seed)
    agent = SAC(OBS_DIM, ACT_DIM, device=args.device, batch_size=args.batch_size,
                demo_ratio=args.demo_ratio, bc_coef=args.bc_coef, seed=args.seed)
    n_demo = agent.load_demos(args.demo_path)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "demo_transitions": n_demo,
                   "algorithm": "SAC + demo replay (HIL-SERL/RLPD style)",
                   "env": "PyBullet PandaGraspEnv", "torch": torch.__version__})
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run = None
    if not args.no_trackio:
        import trackio
        run = trackio.init(project=args.project, group=args.group, name=run_name,
                           embed=False, config=config)
    print(f"Run: {run_name}\nOutput: {out}\nDemos: {n_demo} transitions\n"
          f"Trackio dir: {trackio_dir}", flush=True)

    def log(metrics):
        if run is not None:
            run.log(metrics)

    def save_actor(tag):
        torch.save({"actor": agent.actor.state_dict(), "obs_dim": OBS_DIM,
                    "act_dim": ACT_DIM, "config": config}, out / f"actor_{tag}.pt")

    start_step, best_success = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=agent.device, weights_only=False)
        agent.actor.load_state_dict(ck["actor"])
        agent.critic.load_state_dict(ck["critic"])
        agent.critic_target.load_state_dict(ck["critic_target"])
        agent.actor_opt.load_state_dict(ck["actor_opt"])
        agent.critic_opt.load_state_dict(ck["critic_opt"])
        agent.alpha_opt.load_state_dict(ck["alpha_opt"])
        agent.log_alpha.data.copy_(ck["log_alpha"].to(agent.device))
        for k in ("obs", "act", "rew", "next_obs", "done", "size", "ptr"):
            setattr(agent.online, k, ck["online"][k])
        agent.updates = ck["updates"]
        start_step, best_success = ck["step"], ck["best_success"]
        rng.bit_generator.state = ck["rng"]
        agent.rng.bit_generator.state = ck["agent_rng"]
        print(f"Resumed from {args.resume} at step {start_step}", flush=True)
    elif args.bc_pretrain > 0:
        t0 = datetime.now()
        bc_loss = 0.0
        for _ in range(args.bc_pretrain):
            bc_loss = agent.bc_update(args.batch_size)
        print(f"BC pretrain: {args.bc_pretrain} steps, final MSE {bc_loss:.4f} "
              f"({(datetime.now() - t0).total_seconds():.1f}s)", flush=True)
        log({"bc/pretrain_loss": bc_loss})

    step = start_step

    def save_state():
        torch.save({
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
            "actor_opt": agent.actor_opt.state_dict(),
            "critic_opt": agent.critic_opt.state_dict(),
            "alpha_opt": agent.alpha_opt.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "online": {k: getattr(agent.online, k) for k in
                       ("obs", "act", "rew", "next_obs", "done", "size", "ptr")},
            "updates": agent.updates, "step": step, "best_success": best_success,
            "rng": rng.bit_generator.state, "agent_rng": agent.rng.bit_generator.state,
        }, out / "state.pt")

    stop = {"flag": False}

    def _on_term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)

    obs = env.reset()
    ep_r, ep_len = 0.0, 0
    try:
        while step < args.total_steps and not stop["flag"]:
            if step < args.learning_starts:
                action = rng.uniform(-1, 1, ACT_DIM).astype(np.float32)
            else:
                action = agent.act(obs)
            nxt, r, term, trunc, info = env.step(action)
            agent.online.add(obs, action, r, nxt, float(term))
            obs = nxt
            ep_r += r
            ep_len += 1
            step += 1
            if term or trunc:
                print(f"step {step:6d} | train ep reward {ep_r:7.2f} "
                      f"success {int(info['success'])} lift {info['lift']:.3f} "
                      f"len {ep_len}", flush=True)
                log({"env_steps": step, "train/episode_reward": ep_r,
                     "train/success": int(info["success"]),
                     "train/lift_m": info["lift"], "train/episode_steps": ep_len})
                obs = env.reset()
                ep_r, ep_len = 0.0, 0
            if step >= args.learning_starts:
                for _ in range(args.utd):
                    losses = agent.update()
                if agent.updates % 100 < args.utd:
                    log({"env_steps": step, "train/critic_loss": losses["critic_loss"],
                         "train/actor_loss": losses["actor_loss"],
                         "train/alpha": losses["alpha"],
                         "train/online_size": agent.online.size,
                         "train/demo_size": n_demo})
            if step % args.eval_every == 0 or step == args.total_steps:
                rows = evaluate(env, agent, args.eval_episodes, seed=90_000)
                rate = float(np.mean([r[0] for r in rows]))
                mean_r = float(np.mean([r[1] for r in rows]))
                mean_lift = float(np.mean([r[2] for r in rows]))
                print(f"step {step:6d} | EVAL success {rate:.2f} "
                      f"reward {mean_r:.2f} lift {mean_lift:.3f}", flush=True)
                log({"env_steps": step, "eval/success_rate": rate,
                     "eval/mean_reward": mean_r, "eval/mean_lift_m": mean_lift})
                if rate > best_success:
                    best_success = rate
                    save_actor("best")
                save_state()
                obs = env.reset()
                ep_r, ep_len = 0.0, 0
        save_actor("final")
        log({"env_steps": step, "final/best_eval_success": best_success})
        print(f"Done. Best eval success rate: {best_success:.2f}\n"
              f"Checkpoints: {out}", flush=True)
    finally:
        save_state()
        if run is not None:
            run.finish()
        env.close()


if __name__ == "__main__":
    main()