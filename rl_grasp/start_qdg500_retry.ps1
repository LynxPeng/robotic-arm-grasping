$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$results = 'robotic-grasping/results'
$tag = 'qdg500-local-w4-b16'
$dataset = "$results/qdg500-dataset"
if (!(Test-Path "$dataset/demos.npz") -or !(Test-Path "$dataset/train.json")) {
    $dataset = "$results/qdg500-dataset-retry"
}
$py = (Resolve-Path 'robotic-grasping/.conda/python.exe').Path
$wslRoot = '/mnt/d/Projects/DRLExp/robotic_arm'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$logs = Join-Path (Get-Location).Path "$results/qdg500-tensorboard"
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if (!(Get-NetTCPConnection -LocalPort 6006 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $py -ArgumentList @('-m','tensorboard.main','--logdir',$logs,'--host','127.0.0.1','--port','6006') -WindowStyle Hidden -RedirectStandardOutput "$results/qdg500-tensorboard.out.log" -RedirectStandardError "$results/qdg500-tensorboard.err.log"
}

if (!(Test-Path "$dataset/demos.npz") -or !(Test-Path "$dataset/train.json")) {
    & $py -u rl_grasp/qdg_demo.py data/qdgset/zarr_database.zip --out $dataset --objects 500 --holdout 50 --candidate-objects 1150 --attempts 4 --workers 4 --screen-ik *> "$results/$tag-demo.log"
    if ($LASTEXITCODE -ne 0) { throw 'QDGset demonstration preparation failed.' }
}

& wsl.exe -d Ubuntu -- bash "$wslRoot/rl_grasp/prepare_qdg500_wsl.sh" *> "$results/$tag-build.log"
if ($LASTEXITCODE -ne 0) { throw 'WSL training environment preparation failed.' }

& wsl.exe -d Ubuntu -- bash "$wslRoot/rl_grasp/run_qdg500_local.sh" $tag '-' $dataset *> "$results/$tag-train.log"
if ($LASTEXITCODE -ne 0) { throw 'GA-DDPG training command failed.' }
