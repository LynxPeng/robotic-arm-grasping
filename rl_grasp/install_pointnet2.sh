#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
export MAX_JOBS=${MAX_JOBS:-3}
src=vendor/Pointnet2_PyTorch
if [ ! -d "$src/.git" ]; then
    git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git "$src"
fi
git -C "$src" checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626
.conda/bin/python - <<'PY'
from pathlib import Path
p = Path('vendor/Pointnet2_PyTorch/pointnet2_ops_lib/setup.py')
s = p.read_text()
s = s.replace('os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"',
              'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")')
p.write_text(s)
PY
.conda/bin/python -m pip install --no-build-isolation "$src/pointnet2_ops_lib"
