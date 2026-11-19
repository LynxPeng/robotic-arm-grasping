"""Atomic checkpoints at completed online rollout boundaries."""
from hashlib import file_digest
from importlib.metadata import version
import os
import sys


def signature(args, torch, device):
    with args.demos_from.open('rb') as stream:
        demos_hash = file_digest(stream, 'sha256').hexdigest()
    keys = ('stage', 'seed', 'workers', 'horizon', 'steps', 'batch', 'updates_per_batch',
            'online_capacity', 'lr', 'aux_weight', 'renderer', 'cuda_graph',
            'eval_every', 'eval_per_kind', 'eval_seed')
    settings = {key: getattr(args, key) for key in keys}
    if args.update_every_env_steps is not None:
        settings['update_every_env_steps'] = args.update_every_env_steps
    if args.async_envs:
        settings['async_envs'] = True
    if getattr(args, 'lr_warmup_updates', 0):
        settings['lr_warmup_updates'] = args.lr_warmup_updates
    if getattr(args, 'lr_schedule', None):
        settings['lr_schedule'] = args.lr_schedule
    settings['dagger_ratio'] = getattr(args, 'dagger_ratio', 0.)
    settings['dagger_min_step'] = getattr(args, 'dagger_min_step', 0)
    settings['dagger_max_step'] = getattr(args, 'dagger_max_step', 0)
    settings['expert'] = getattr(args, 'expert', 'scripted')
    settings['online_goals'] = not getattr(args, 'no_online_goals', False)
    settings['gamma'] = getattr(args, 'gamma', 0.99)
    settings['tau'] = getattr(args, 'tau', 0.005)
    settings['ddpg_weight'] = getattr(args, 'ddpg_weight', 0.2)
    if getattr(args, 'lr_milestones', None):
        settings['lr_milestones'] = list(args.lr_milestones)
        settings['lr_gamma'] = getattr(args, 'lr_gamma', 0.5)
    if getattr(args, 'qdg_sha256', None):
        settings['qdg_sha256'] = args.qdg_sha256
        settings['qdg_validation_objects'] = args.qdg_validation_objects
    return dict(architecture='pointnet2-cuda-bn-half-history-v3-omg-dagger', settings=settings, demos_sha256=demos_hash,
                torch=str(torch.__version__), platform=sys.platform, device=device.type,
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else '',
                pybullet=version('pybullet'))


def save_training(path, torch, learner, device, rng, replay, workers, obs,
                  config, step, best, next_eval, *, keep_steps=False):
    state = dict(version=1, config=config, steps=step, best=best, next_eval=next_eval,
                 updates=learner.updates, numpy_rng=rng.bit_generator.state,
                 torch_rng=torch.get_rng_state(), workers=workers, obs=torch.from_numpy(obs),
                 replay=[torch.from_numpy(a[:replay.size]) for a in replay.fields],
                 replay_cursor=replay.cursor)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        state['cuda_rng'] = torch.cuda.get_rng_state(device)
    for name in ('actor', 'critic', 'actor_target', 'critic_target', 'actor_opt', 'critic_opt'):
        state[name] = getattr(learner, name).state_dict()
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if keep_steps:
        # Atomic replacement of latest leaves each archived inode unchanged.
        archive = path.with_name(f'{path.stem}-step-{step:09d}{path.suffix}')
        os.link(path, archive)
    if hasattr(os, 'O_DIRECTORY'):
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load_training(path, torch, learner, device, rng, replay, expected):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != 1 or state['config'] != expected:
        raise ValueError('Resume requires matching training settings, runtime, renderer and demonstration data.')
    for name in ('actor', 'critic', 'actor_target', 'critic_target', 'actor_opt', 'critic_opt'):
        getattr(learner, name).load_state_dict(state.pop(name))
    stored = state.pop('replay')
    replay.size = len(stored[0])
    replay.cursor = state.pop('replay_cursor')
    for field, values in zip(replay.fields, stored, strict=True):
        field[:replay.size] = values.numpy()
    learner.updates = state['updates']
    rng.bit_generator.state = state.pop('numpy_rng')
    torch.set_rng_state(state.pop('torch_rng'))
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state.pop('cuda_rng'), device)
    return state
