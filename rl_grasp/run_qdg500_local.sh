#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
runtime=${QDG_RUNTIME:-"$HOME/robotic_arm/.conda"}
py="$runtime/bin/python"
export PATH="$runtime/bin:/usr/lib/wsl/lib:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

results=robotic-grasping/results
dataset=${3:-"$results/qdg500-dataset"}
tag=${1:-qdg500-local}
resume=${2:-}
common=(ga --qdg-dataset "$dataset" --demos-from "$dataset/demos.npz"
        --workers 4 --batch 16 --lr 0.0003 --async-envs --cuda-graph
        --update-every-env-steps 4 --horizon 160 --steps 1000000
        --save-every 5000 --log-every 400 --eval-every 25000 --eval-per-kind 2
        --logger tensorboard --tensorboard-dir "$results/qdg500-tensorboard")

exec 9>"$results/qdg500-training.lock"
flock -n 9
if [[ -z "$resume" || "$resume" == "-" ]]; then
    bootstrap="$results/$tag-bootstrap"
    "$py" -u rl_grasp/train_ga.py "${common[@]}" --out "$bootstrap" --stop-after 320
    test -s "$bootstrap/training-state.pt"
    resume="$bootstrap/training-state.pt"
fi
"$py" -u rl_grasp/train_ga.py "${common[@]}" --out "$results/$tag" --resume "$resume"
