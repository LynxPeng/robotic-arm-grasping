#!/usr/bin/env bash
# Bounded smoke benchmarks only; no convergence training.
set -Eeuo pipefail
root=$1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trainer=(.conda/bin/python -u rl_grasp/train_ga.py)
common=(--renderer egl --logger tensorboard --workers 10 --batch 32 --seed 41)
for mode in eager graph; do
    flags=()
    if [ "$mode" = graph ]; then flags+=(--cuda-graph); fi
    .conda/bin/python rl_grasp/benchmark_native_pointnet2.py "${flags[@]}" --batch 32 \
        > "$root/micro-b32-$mode.json" 2> "$root/micro-b32-$mode.stderr"
done
for mode in graph; do
    flags=()
    if [ "$mode" = graph ]; then flags+=(--cuda-graph); fi
    printf 'START %s %s\n' "$mode" "$(date -Is)"
    "${trainer[@]}" benchmark "${common[@]}" "${flags[@]}" --async-envs \
        --demos-from "$root/native-demos/demos.npz" --bc-updates 100 --lr .0001 \
        --update-every-env-steps 4 --steps 600000 --benchmark-seconds 180 \
        --online-capacity 30000 --log-every 400 --profile --profile-workers \
        --out "$root/bounded-native-b32-$mode" > "$root/bounded-b32-$mode.console.log" 2>&1
    printf 'END %s %s\n' "$mode" "$(date -Is)"
done
touch "$root/DONE"
