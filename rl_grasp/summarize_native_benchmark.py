"""Summarize bounded online windows, with device and allocator memory separate."""
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

resources = rows(root/'resources.jsonl')
result = {}
for mode, directory, micro in (
        ('batch128_eager', 'bounded-native-eager', 'micro-eager.json'),
        ('batch32_graph', 'bounded-native-b32-graph', 'micro-b32-graph.json')):
    run = root/directory
    metrics = rows(run/'metrics.jsonl')
    windows = [r for r in metrics if r['event'] == 'train']
    latter = [r for r in windows if r['online_window_start_s'] >= 90]
    seconds = sum(r['online_window_end_s']-r['online_window_start_s'] for r in latter)
    gpu = rows(run/'gpu-samples.jsonl')
    stable_gpu = [r for r in gpu if 90 <= r['online_seconds'] <= 180]
    online_start = gpu[0]['unix_seconds']-gpu[0]['online_seconds']
    cpu = [r for r in resources if online_start+90 <= r['unix_seconds'] <= online_start+180]
    totals = {key: sum(r.get(key, 0) for r in latter) for key in (
        'network_call_s', 'replay_sample_s', 'pin_s', 'copy_submit_s', 'policy_seconds',
        'action_readback_s', 'worker_wait_seconds', 'sim_steps', 'sim_camera_s',
        'sim_physics_s', 'sim_ik_s', 'sim_pointcloud_s', 'sim_episode_reset_s')}
    result[mode] = dict(
        micro=rows(root/micro)[-1],
        online=next(r for r in metrics if r['event']=='benchmark'),
        latter_window_seconds=seconds,
        latter_env_per_s=sum(r['env_per_s']*(r['online_window_end_s']-r['online_window_start_s'])
                            for r in latter)/seconds,
        latter_updates_per_s=sum(r['updates_per_s']*(r['online_window_end_s']-r['online_window_start_s'])
                                for r in latter)/seconds,
        latter_gpu_util_mean=float(np.mean([r['gpu_util_pct'] for r in stable_gpu])),
        latter_gpu_util_min=min(r['gpu_util_pct'] for r in stable_gpu),
        latter_gpu_util_max=max(r['gpu_util_pct'] for r in stable_gpu),
        gpu_device_peak_mib=max(r['gpu_device_mib'] for r in gpu),
        latter_gpu_memory_start_mib=stable_gpu[0]['gpu_device_mib'],
        latter_gpu_memory_end_mib=stable_gpu[-1]['gpu_device_mib'],
        torch_peak_allocated_mib=max(r['torch_peak_allocated_mib'] for r in windows),
        torch_peak_reserved_mib=max(r['torch_peak_reserved_mib'] for r in windows),
        latter_cpu_cores=float(np.mean([r['cpu_used_cores'] for r in cpu])),
        latter_cpu_quota_util_pct=float(np.mean([r['cpu_quota_util_pct'] for r in cpu])),
        latter_profile_s=totals,
        maximum_pending_updates=max(r['step']//4-r['updates'] for r in windows))
(root/'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
