"""Record GPU and cgroup CPU usage to TensorBoard during a bounded suite."""
import json
import subprocess
import sys
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

root = Path(sys.argv[1])
quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
cores = int(quota) / int(period)

def cpu_seconds():
    values = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    return int(values['usage_usec']) / 1e6

writer = SummaryWriter(f'/root/tf-logs/ga-ddpg-mini/{root.name}-resources', flush_secs=5)
with (root/'suite.log').open('w') as console:
    job = subprocess.Popen(['bash', 'rl_grasp/benchmark_native_suite.sh', str(root)],
                           stdout=console, stderr=subprocess.STDOUT)
    previous_time = time.monotonic()
    previous_cpu = cpu_seconds()
    while job.poll() is None:
        time.sleep(2)
        now, cpu = time.monotonic(), cpu_seconds()
        used = (cpu-previous_cpu)/(now-previous_time)
        row = dict(unix_seconds=time.time(), cpu_used_cores=used, cpu_quota_util_pct=used/cores*100)
        try:
            output = subprocess.check_output(['nvidia-smi', '--id=0',
                '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                text=True, timeout=3)
            gpu, memory = map(float, output.strip().split(','))
            row.update(gpu_util_pct=gpu, gpu_memory_mib=memory)
        except (subprocess.SubprocessError, ValueError):
            pass
        for key, value in row.items():
            if key != 'unix_seconds':
                writer.add_scalar('resources/'+key, value, int(now))
        writer.flush()
        with (root/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        previous_time, previous_cpu = now, cpu
writer.close()
(root/'exit-code.txt').write_text(str(job.returncode))
sys.exit(job.returncode)
