#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/robotic_arm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GA_SMOKE_ENCODER

qdg_ds="robotic-grasping/results/qdg500-demo-w16-20261116-210216"
demo_out="robotic-grasping/results/qdg500-omg-demo-retract-1500-20261118"
train_out="robotic-grasping/results/qdg500-omg-ga-v5-retract-b32-lr5e5-bc20k-20261118"

if [[ "${1:-}" == "resume" ]]; then
    exec .conda/bin/python -u rl_grasp/train_ga.py ga \
      --out "${train_out}-resume-$(date +%Y%m%d-%H%M%S)" \
      --resume "$train_out/training-state.pt" \
      --demos-from "$demo_out/demos.npz" \
      --qdg-dataset "$qdg_ds" \
      --workers 7 --batch 32 --updates-per-batch 10 \
      --gamma 0.95 --tau 0.0001 --ddpg-weight 0.2 \
      --lr 0.00005 --lr-milestones 80000 160000 240000 320000 --lr-gamma 0.5 --aux-weight 1.0 \
      --dagger-ratio 0.5 --dagger-min-step 3 --dagger-max-step 35 --expert planner \
      --bc-updates 20000 --bc-eval-every 4000 --horizon 50 --steps 280000 --online-capacity 100000 \
      --async-envs --cuda-graph --renderer egl --device cuda \
      --seed 41 --eval-seed 300001 --qdg-validation-objects 12 --eval-per-kind 2 \
      --save-every 14000 --keep-checkpoints --eval-every 14000 --log-every 1400 \
      --logger tensorboard --tensorboard-dir /root/tf-logs
fi

if [[ ! -f "$demo_out/demos.npz" ]]; then
    .conda/bin/python -u rl_grasp/train_ga.py demo \
      --out "$demo_out" \
      --qdg-dataset "$qdg_ds" \
      --workers 7 --demo-trials 1500 --horizon 50 \
      --renderer egl --expert planner --seed 41 \
      --logger tensorboard --tensorboard-dir /root/tf-logs
fi

exec .conda/bin/python -u rl_grasp/train_ga.py ga \
  --out "$train_out" \
  --demos-from "$demo_out/demos.npz" \
  --qdg-dataset "$qdg_ds" \
  --workers 7 --batch 32 --updates-per-batch 10 \
  --gamma 0.95 --tau 0.0001 --ddpg-weight 0.2 \
  --lr 0.00005 --lr-milestones 80000 160000 240000 320000 --lr-gamma 0.5 --aux-weight 1.0 \
  --dagger-ratio 0.5 --dagger-min-step 3 --dagger-max-step 35 --expert planner \
  --bc-updates 20000 --bc-eval-every 4000 --horizon 50 --steps 280000 --online-capacity 100000 \
  --async-envs --cuda-graph --renderer egl --device cuda \
  --seed 41 --eval-seed 300001 --qdg-validation-objects 12 --eval-per-kind 2 \
  --save-every 14000 --keep-checkpoints --eval-every 14000 --log-every 1400 \
  --logger tensorboard --tensorboard-dir /root/tf-logs
