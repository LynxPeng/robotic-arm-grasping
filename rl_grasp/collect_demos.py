"""Collect scripted-expert demonstrations in the env's own action space.

Stands in for the human demonstrations of HIL-SERL: a closed-loop waypoint
controller (above object -> descend -> close -> lift) produces transitions
(obs, act, rew, next_obs, done) that are later replayed 50/50 with online
data during SAC updates (RLPD-style).
"""
import argparse
from pathlib import Path

import numpy as np

from env import PandaGraspEnv, POS_SCALE, RESULTS_DIR


def expert_action(env, phase, close_steps):
    tcp = env._tcp()
    obj = env._obj_pos()
    grasp = env._grasp_target()
    if phase == 0:                      # move above the object
        target = np.array((obj[0], obj[1], 0.22))
        grip = -1.0
        if np.linalg.norm(tcp[:2] - obj[:2]) < 0.012 and abs(tcp[2] - 0.22) < 0.02:
            phase = 1
    elif phase == 1:                    # descend to grasp height, open
        target = grasp
        grip = -1.0
        if np.linalg.norm(tcp - grasp) < 0.008:
            phase, close_steps = 2, 0
    elif phase == 2:                    # close fingers
        target = grasp
        grip = 1.0
        close_steps += 1
        if close_steps >= 15:
            phase = 3
    else:                               # lift while keeping grip closed
        target = np.array((obj[0], obj[1], grasp[2] + 0.16))
        grip = 1.0
    delta = np.clip((target - tcp) / POS_SCALE, -1, 1)
    return np.array([*delta, grip], dtype=np.float32), phase, close_steps


def collect(episodes, seed, out_path, gui=False, reward_mode="shaped"):
    env = PandaGraspEnv(gui=gui, reward_mode=reward_mode)
    obs_l, act_l, rew_l, next_l, done_l = [], [], [], [], []
    successes = 0
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        phase, close_steps, done, ep_r = 0, 0, False, 0.0
        info = {}
        while not done:
            a, phase, close_steps = expert_action(env, phase, close_steps)
            nxt, r, term, trunc, info = env.step(a)
            obs_l.append(obs)
            act_l.append(a)
            rew_l.append(r)
            next_l.append(nxt)
            done_l.append(float(term))  # truncation must still bootstrap
            obs = nxt
            ep_r += r
            done = term or trunc
        successes += int(info.get("success", False))
        print(f"demo {ep:02d}: success={info.get('success')} lift={info.get('lift', 0):.3f} "
              f"steps={info.get('steps')} reward={ep_r:.2f}", flush=True)
    env.close()
    data = dict(obs=np.asarray(obs_l, np.float32), act=np.asarray(act_l, np.float32),
                rew=np.asarray(rew_l, np.float32).reshape(-1, 1),
                next_obs=np.asarray(next_l, np.float32),
                done=np.asarray(done_l, np.float32).reshape(-1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    print(f"saved {len(obs_l)} transitions ({successes}/{episodes} successful demos) -> {out_path}",
          flush=True)
    return successes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "demos.npz")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--reward", choices=("shaped", "sparse"), default="shaped")
    args = ap.parse_args()
    collect(args.episodes, args.seed, args.out, gui=args.gui, reward_mode=args.reward)


if __name__ == "__main__":
    main()