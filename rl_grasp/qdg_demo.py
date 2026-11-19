"""Collect physics-verified GA-DDPG demonstrations from QDGset meshes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import file_digest
import json
import math
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import pybullet as p

from env import ARM, POS_SCALE, TCP
from ga_env import ROT_SCALE, matrix, relative_pose, goal_label
from ga_workers import label_episode
from train_ga import Replay
from qdg_env import QDGPointGraspEnv

IDENTITY = (0., 0., 0., 1.)
HAND_TO_TCP = (0., 0., 0.105)  # panda_grasptarget_hand in PyBullet's Panda URDF


def grasp_tcp(world_inertia, grasp, offset=HAND_TO_TCP):
    position, orientation = p.multiplyTransforms(
        *world_inertia, grasp[:3], grasp[3:7])
    return p.multiplyTransforms(position, orientation, offset, IDENTITY)


@dataclass
class ObjectAsset:
    key: str
    mesh: Path
    vertices: np.ndarray
    inertia_origin: np.ndarray
    mass: float
    grasps: np.ndarray

    @property
    def bounds(self):
        return np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))


def export_objects(archive, folder, count, seed, *, independent=False):
    import zarr

    folder.mkdir()
    rng = np.random.default_rng(seed)
    selected = []
    seen_meshes = set()
    with zarr.ZipStore(str(archive), mode='r') as store:
        datasets = zarr.open_group(store, mode='r')['datasets']
        groups = {}
        for source in sorted(datasets.group_keys()):
            group = datasets[source]
            keys = sorted(group.group_keys())
            print(f'{source}: {len(keys)} entries', flush=True)
            if 'augment' in source.lower() or (independent and source == 'graspnet'):
                continue
            keys = [key for key in keys if '_da_' not in key.lower()]
            rng.shuffle(keys)
            groups[source] = iter(keys)
        while groups and len(selected) < count:
            for source in tuple(groups):
                key = next(groups[source], None)
                if key is None:
                    del groups[source]
                    continue
                group = datasets[source][key]
                if 'grasps' not in group or 'obj' not in group:
                    continue
                grasps = np.asarray(group['grasps'], dtype=float)
                if grasps.ndim != 2 or grasps.shape[1] != 8:
                    continue
                finite = np.isfinite(grasps).all(axis=1)
                unit = np.abs(np.linalg.norm(grasps[:, 3:7], axis=1) - 1) < 0.01
                grasps = grasps[finite & unit & (grasps[:, 7] > 0)]
                if not len(grasps):
                    continue
                obj = group['obj']
                vertices = np.asarray(obj['vertices'], dtype=float)
                faces = np.asarray(obj['faces'], dtype=int)
                inertia = np.asarray(obj['inertia_origin'], dtype=float)
                mass = float(np.asarray(obj['mass']).item())
                if (not np.isfinite(vertices).all() or not np.isfinite(inertia).all()
                        or inertia.shape != (7,) or abs(np.linalg.norm(inertia[3:]) - 1) > .01
                        or not 0 < mass <= 2 or not len(faces) or faces.min() < 0
                        or faces.max() >= len(vertices)):
                    continue
                extent = np.ptp(vertices, axis=0)
                # Keep the pilot within the Panda's tabletop workspace.
                if not (.025 <= max(extent) <= .24 and min(extent) >= .008):
                    continue
                mesh = folder / f'{len(selected):02d}.obj'
                with mesh.open('w', encoding='ascii') as file:
                    np.savetxt(file, vertices, fmt='v %.9g %.9g %.9g')
                    np.savetxt(file, faces + 1, fmt='f %d %d %d')
                if independent:
                    with mesh.open('rb') as file:
                        digest = file_digest(file, 'sha256').digest()
                    if digest in seen_meshes:
                        mesh.unlink()
                        continue
                    seen_meshes.add(digest)
                selected.append(ObjectAsset(f'{source}/{key}', mesh, vertices, inertia, mass, grasps))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f'Only found {len(selected)} eligible objects, requested {count}')
    return selected


def candidates(env, offset, limit=64):
    world_inertia = p.getBasePositionAndOrientation(env.obj)
    low, high = np.asarray(p.getAABB(env.obj))
    order = np.argsort(-env.asset.grasps[:, 7], kind='stable')
    selected = []
    for index in order:
        grasp = env.asset.grasps[index]
        position, orientation = grasp_tcp(world_inertia, grasp, offset)
        forward = matrix(orientation)[:, 2]
        pregrasp = np.asarray(position) - .10 * forward
        if (forward[2] > -.35 or position[2] < .008
                or np.any(position < low - .025) or np.any(position > high + .025)
                or not (.36 < pregrasp[0] < .69 and abs(pregrasp[1]) < .28)
                or position[2] + .16 > .45):
            continue
        if any(np.linalg.norm(np.asarray(position) - other[1][0]) < .012
               and abs(np.dot(orientation, other[1][1])) > .98 for other in selected):
            continue
        selected.append((int(index), (position, orientation), float(grasp[7])))
        if len(selected) == limit:
            break
    return selected


def waypoints(start, target, screen_ik):
    grasp, orientation = target
    approach = matrix(orientation)[:, 2]
    pregrasp = np.asarray(grasp) - (.075 if screen_ik else .10) * approach
    # The wrist lies 10.5 cm behind the TCP, so rotating near the top of the
    # Cartesian workspace can put the wrist outside the Panda's reach.
    travel_z = (max(pregrasp[2] + .025, min(.33, start[0][2])) if screen_ik
                else min(.42, max(start[0][2], pregrasp[2] + .055)))
    return [
        ((start[0][0], start[0][1], travel_z), start[1]),
        ((pregrasp[0], pregrasp[1], travel_z), orientation),
        (pregrasp, orientation),
        target,
        target,
        (np.asarray(grasp) + [0, 0, .16], orientation),
    ]


def select_path(env, options, attempt, screen_ik):
    start = env.pose()
    ranked = options[min(attempt, len(options) - 1):] + options[:attempt]
    state = p.saveState() if screen_ik else None
    try:
        for index, (position, orientation), score in ranked:
            # Finger exchange preserves a parallel-jaw grasp.
            _, flipped = p.multiplyTransforms((0, 0, 0), orientation, (0, 0, 0), (0, 0, 1, 0))
            orientations = sorted((orientation, flipped),
                                  key=lambda q: -abs(np.dot(start[1], q)))
            for quat in orientations:
                targets = waypoints(start, (position, quat), screen_ik)
                if not screen_ik:
                    return index, score, targets
                p.restoreState(stateId=state)
                if reachable(env, targets):
                    return index, score, targets
        return None
    finally:
        if state is not None:
            p.restoreState(stateId=state)
            p.removeState(state)


def reachable(env, targets):
    for position, quat in (*targets[:4], targets[-1]):
        joints = p.calculateInverseKinematics(
            env.robot, TCP, position, quat, maxNumIterations=160,
            residualThreshold=1e-5, **env._ik)[:7]
        if any(not lo <= value <= hi for value, lo, hi in
               zip(joints, env._ik['lowerLimits'], env._ik['upperLimits'])):
            return False
        for joint, value in zip(ARM, joints):
            p.resetJointState(env.robot, joint, value)
        actual = env.pose()
        if np.linalg.norm(np.asarray(actual[0]) - position) > .006:
            return False
        if abs(np.dot(actual[1], quat)) < math.cos(.08 / 2):
            return False
        if any(c[3] >= 0 for c in p.getClosestPoints(env.robot, env.support, 0.001)):
            return False
    return True


def rollout(env, obs, targets):
    phase = wait = 0
    path = []
    peak_lift = 0.
    any_grasp = False
    while True:
        pose = env.pose()
        delta = relative_pose(pose, targets[phase])
        tolerance = .005 if phase == 3 else .009
        if phase < 4 and np.linalg.norm(delta[:3]) < tolerance and np.linalg.norm(delta[3:]) < .10:
            phase += 1
        elif phase == 4:
            wait += 1
            if wait >= 12:
                phase = 5
        delta = relative_pose(pose, targets[phase])
        action = np.clip(np.r_[delta[:3] / POS_SCALE, delta[3:] / ROT_SCALE,
                               1. if phase >= 4 else -1.], -1, 1).astype(np.float32)
        next_obs, reward, term, trunc, info = env.step(action)
        goal = goal_label(pose, targets[3])
        path.append((obs, action, action, reward, next_obs, float(term), goal, (1., 1.), pose))
        obs = next_obs
        peak_lift = max(peak_lift, info['lift'])
        any_grasp |= info['grasped']
        if term or trunc:
            error = relative_pose(env.pose(), targets[phase])
            info.update(phase=phase, peak_lift=peak_lift, any_bilateral_contact=any_grasp,
                        waypoint_position_error_m=float(np.linalg.norm(error[:3])),
                        waypoint_rotation_error_deg=math.degrees(np.linalg.norm(error[3:])))
            return label_episode(path, env.close_pose) if term else [], info


def test_object(job):
    asset, first_seed, attempts, horizon, tcp_offset, screen_ik = job
    records = []
    demos = []
    env = None
    try:
        for attempt in range(attempts):
            seed = first_seed + attempt
            record = dict(object=asset.key, seed=seed, success=False, steps=0)
            try:
                if env is None:
                    env = QDGPointGraspEnv(asset, seed=seed, max_steps=horizon)
                    obs = env._obs()
                else:
                    obs = env.reset(seed)
            except RuntimeError as exc:
                if str(exc) != 'Could not sample a reachable, visible, collision-free initial pose.':
                    raise
                record['failure'] = 'initial_pose'
                records.append(record)
                continue
            start = env.pose()
            record.update(start_position=start[0], start_quaternion=start[1],
                          initial_pose_attempts=env.initial_pose_attempts)
            options = candidates(env, (0., 0., tcp_offset))
            record['candidate_count'] = len(options)
            if not options:
                record['failure'] = 'no_tabletop_candidate'
                records.append(record)
                continue
            selected = select_path(env, options, attempt, screen_ik)
            if selected is None:
                record['failure'] = 'no_reachable_waypoints'
                records.append(record)
                continue
            index, score, targets = selected
            rows, info = rollout(env, obs, targets)
            record.update(info, grasp_index=index, grasp_score=score)
            if rows:
                demos.extend(rows)
            else:
                record['failure'] = ('waypoint_timeout' if info['phase'] < 4
                                     else 'lost_grasp' if info['any_bilateral_contact']
                                     else 'no_bilateral_contact')
            records.append(record)
        return records, demos
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--objects', type=int, default=24)
    parser.add_argument('--holdout', type=int, default=0)
    parser.add_argument('--candidate-objects', type=int,
                        help='Try additional training objects until --objects have successful demos.')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--horizon', type=int, default=160)
    parser.add_argument('--tcp-offset', type=float, default=HAND_TO_TCP[2])
    parser.add_argument('--screen-ik', action='store_true',
                        help='Use shorter approach waypoints and screen each candidate with Panda IK.')
    args = parser.parse_args()
    if min(args.objects, args.attempts, args.workers, args.horizon) < 1:
        parser.error('objects, attempts, workers and horizon must be positive')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    candidate_count = args.candidate_objects or args.objects
    if args.holdout < 0 or candidate_count < args.objects:
        parser.error('holdout must be nonnegative and candidate-objects must cover objects')
    assets = export_objects(args.archive, args.out / 'meshes', candidate_count + args.holdout,
                            args.seed, independent=bool(args.holdout))
    manifest = [dict(object=asset.key, mesh=asset.mesh.relative_to(args.out).as_posix(),
                     mass=asset.mass, inertia_origin=asset.inertia_origin.tolist(),
                     bounds=[asset.vertices.min(axis=0).tolist(), asset.vertices.max(axis=0).tolist()],
                     extent=np.ptp(asset.vertices, axis=0).tolist(),
                     grasp_candidates=len(asset.grasps)) for asset in assets]
    (args.out / 'objects.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.holdout:
        (args.out / 'holdout.json').write_text(json.dumps(manifest[:args.holdout], indent=2),
                                              encoding='utf-8')
    jobs = [(asset, args.seed + i * args.attempts, args.attempts, args.horizon, args.tcp_offset,
             args.screen_ik)
            for i, asset in enumerate(assets[args.holdout:])]
    records = []
    demos = Replay(args.objects * args.attempts * args.horizon)
    start = perf_counter()
    retained = set()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for first in range(0, len(jobs), 3 * args.workers):
            for trials, rows in pool.map(test_object, jobs[first:first + 3 * args.workers]):
                records.extend(trials)
                if rows and (not args.candidate_objects or len(retained) < args.objects):
                    retained.add(trials[0]['object'])
                    demos.append(rows)
                print(f'{trials[0]["object"]}: {sum(t["success"] for t in trials)}/{len(trials)} '
                      f'({len(retained)} retained)', flush=True)
            if args.candidate_objects and len(retained) >= args.objects:
                break
    if args.holdout:
        training = [item for item in manifest[args.holdout:] if item['object'] in retained]
        (args.out / 'train.json').write_text(json.dumps(training, indent=2), encoding='utf-8')
    summary = dict(objects=len({r['object'] for r in records}), attempts=len(records),
                   successes=sum(r['success'] for r in records),
                   objects_with_success=len({r['object'] for r in records if r['success']}),
                   failures=dict(Counter(r['failure'] for r in records if 'failure' in r)),
                   transitions=demos.size, retained_objects=len(retained), holdout_objects=args.holdout,
                   seed=args.seed, tcp_offset_m=args.tcp_offset,
                   horizon=args.horizon, attempts_per_object=args.attempts,
                   screen_ik=args.screen_ik, rollout_seconds=perf_counter() - start)
    (args.out / 'trials.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if demos.size:
        demos.save(args.out / 'demos.npz')
    print(json.dumps(summary, indent=2), flush=True)
    if args.candidate_objects and len(retained) != args.objects:
        raise RuntimeError(f'Only {len(retained)} objects yielded demos, expected {args.objects}')


if __name__ == '__main__':
    main()
