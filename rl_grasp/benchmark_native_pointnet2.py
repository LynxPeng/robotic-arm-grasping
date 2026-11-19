"""Synthetic learner throughput; excludes rendering, replay sampling and transfer."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from ga_env import OBS_DIM, ACT_DIM
from ga_model import Learner

p = argparse.ArgumentParser()
p.add_argument('--cuda-graph', action='store_true')
p.add_argument('--seconds', type=float, default=30)
p.add_argument('--batch', type=int, default=128)
p.add_argument('--profile', type=Path)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(41)
device = torch.device('cuda')
learner = Learner(device, 1e-4, .5, cuda_graph=a.cuda_graph)
torch.cuda.reset_peak_memory_stats()
b = a.batch
actions = torch.rand(b, ACT_DIM, device=device)*2-1
batch = (torch.randn(b, OBS_DIM, device=device), actions, actions.clone(),
         torch.zeros(b, device=device), torch.randn(b, OBS_DIM, device=device),
         torch.zeros(b, device=device), torch.zeros(b, 6, device=device),
         torch.ones(b, 2, device=device))
for _ in range(16):
    learner.update(batch)
torch.cuda.synchronize()
# Include capture allocations: reset after capture undercounts graph-private pools.
start = perf_counter()
count = 0
while perf_counter()-start < a.seconds:
    for _ in range(4):
        metrics = learner.update(batch)
    torch.cuda.synchronize()
    count += 4
elapsed = perf_counter()-start
assert all(torch.isfinite(v).all() for v in metrics.values())
print(json.dumps(dict(cuda_graph=a.cuda_graph, batch=b, updates=count, seconds=elapsed,
    updates_per_s=count/elapsed, milliseconds_per_update=1000*elapsed/count,
    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
    actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
    critic_parameters=sum(p.numel() for p in learner.critic.parameters()),
    encoder_parameters=sum(p.numel() for p in learner.actor.encoder.parameters()),
    graph_replays=learner.graph_replays)), flush=True)
if a.profile:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(4):
            learner.update(batch)
        torch.cuda.synchronize()
    a.profile.write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25))
