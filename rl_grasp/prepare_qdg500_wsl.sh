#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/robotic_arm"
prefix="$PWD/.conda"
py="$prefix/bin/python"
export PATH="$prefix/bin:/usr/lib/wsl/lib:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! "$py" -c 'import torch; assert torch.cuda.is_available()'; then
    "$py" -m pip install --force-reinstall torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
fi
"$py" -m pip install 'tensorboard==2.20.0' 'setuptools==80.10.2' wheel
if ! "$py" -c 'import torch; import pointnet2_ops._ext'; then
    export CUDA_HOME="$prefix" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=1
    export CC="$prefix/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$prefix/bin/x86_64-conda-linux-gnu-g++"
    export CPATH="$prefix/targets/x86_64-linux/include"
    for headers in "$prefix"/lib/python*/site-packages/nvidia/*/include; do
        if [[ -d "$headers" ]]; then
            CPATH="$CPATH:$headers"
        fi
    done
    src="$PWD/vendor/Pointnet2_PyTorch/pointnet2_ops_lib"
    sed -i 's/os.environ\["TORCH_CUDA_ARCH_LIST"\] = .*/os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")/' "$src/setup.py"
    "$py" -m pip install --no-build-isolation "$src"
fi
if ! "$py" -c 'import pybullet'; then
    MAMBA_ROOT_PREFIX="$PWD/.bootstrap/mamba" .bootstrap/bin/micromamba install -y -p "$prefix" -c conda-forge pybullet
fi
"$py" -c 'import torch, pybullet; import pointnet2_ops._ext; print(torch.cuda.get_device_name())'
