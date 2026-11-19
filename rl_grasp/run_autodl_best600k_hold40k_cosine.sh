#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/robotic_arm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run=qdg500-best600k-w12-b32-hold40k-cosine-250k-20261117
out="robotic-grasping/results/$run"
init=(--checkpoint robotic-grasping/results/qdg500-4090-w12-b32-lr3e4-nowarmup-20261116-2309/best.pt)
if [[ "${1:-}" == "resume" ]]; then
    init=(--resume "$out/training-state.pt")
    out="${out}-resume-$(date +%Y%m%d-%H%M%S)"
fi
exec .conda/bin/python -u rl_grasp/train_ga.py ga \
  --out "$out" "${init[@]}" \
  --demos-from robotic-grasping/results/qdg500-demo-w16-20261116-210216/demos.npz \
  --qdg-dataset robotic-grasping/results/qdg500-demo-w16-20261116-210216 \
  --workers 12 --batch 32 --lr 0.0001 --lr-schedule 0.0001 4 40000 \
  --horizon 160 --steps 250000 --online-capacity 100000 \
  --async-envs --cuda-graph --renderer egl --update-every-env-steps 4 \
  --seed 41 --eval-seed 300001 --qdg-validation-objects 12 --eval-per-kind 2 \
  --save-every 20000 --keep-checkpoints --eval-every 20000 --log-every 2000 \
  --logger tensorboard --tensorboard-dir /root/tf-logs