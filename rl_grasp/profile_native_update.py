"""Bounded, in-memory profiling of a learner update on recorded grasp clouds."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from ga_model import Learner
from train_ga import Replay, batch_from

p = argparse.ArgumentParser()
p.add_argument('--demos', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(41)
device = torch.device('cuda')
learner = Learner(device, 1e-4, .5)
saved = torch.load(a.checkpoint, map_location=device, weights_only=True)
for name in ('actor', 'critic'):
    getattr(learner, name).load_state_dict(saved[name])
    getattr(learner, name+'_target').load_state_dict(saved[name])
replay = Replay.load(a.demos)
rng = np.random.default_rng(41)
batches = [batch_from(torch, device, rng, replay, None, 32) for _ in range(4)]
for i in range(16):
    learner.update(batches[i % 4])
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True) as prof:
    for batch in batches:
        learner.update(batch)
    torch.cuda.synchronize()
a.out.write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=80))
a.out.with_suffix('.shapes.txt').write_text(prof.key_averages(group_by_input_shape=True).table(
    sort_by='self_cuda_time_total', row_limit=35))
records = []
for e in prof.key_averages():
    if e.device_type == torch.autograd.DeviceType.CPU:
        records.append(dict(name=e.key, calls=e.count, self_cuda_us=e.self_device_time_total,
                            total_cuda_us=e.device_time_total))
a.out.with_suffix('.json').write_text(json.dumps(records, indent=2))
print(a.out)
