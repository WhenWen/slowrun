#!/usr/bin/env bash
# Run as a step INSIDE the user's existing Tiger6 allocation, not a new job.
# srun --jobid=<holder> --overlap --ntasks=1 --cpus-per-task=4 --gres=gpu:10 \
#   bash experiments/professor_forcing/run_tiger.sh smoke
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SLURM_JOB_ID:?Run this inside a SLURM allocation}"
mode=${1:?expected baseline, pf, or smoke}
case "$mode" in baseline|pf|smoke) ;; *) exit 2 ;; esac
mkdir -p runs
exec 9>runs/tiger.lock
flock -n 9 || { echo 'Another Tiger experiment is active in this checkout'; exit 1; }
export CUDA_VISIBLE_DEVICES=${TIGER_GPU_IDS:-0,1,2,3,4,5,6,7}
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export TORCHINDUCTOR_COMPILE_THREADS=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HF_HOME="$PWD/.cache/huggingface"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export XDG_CACHE_HOME="$PWD/.cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor-tiger"
export TRITON_CACHE_DIR="$PWD/.cache/triton-tiger"
export CUDA_CACHE_PATH="$PWD/.cache/cuda-tiger"
mkdir -p "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
run_name="tiger-${mode}-$(date +%Y%m%d-%H%M%S)"
gpu_count=$(python -c 'import torch; print(torch.cuda.device_count())')
case "$gpu_count" in 1|2|4|8) ;; *) echo "Unsupported GPU count $gpu_count"; exit 1 ;; esac
hostname
date -Is
git rev-parse HEAD
echo "EXPLORATORY_HARDWARE GPUs=$gpu_count run=$run_name holder=$SLURM_JOB_ID"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE NCCL_NET=$NCCL_NET"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
python -c 'from prepare_data import verify_hash; verify_hash("fineweb_data/fineweb_train.pt"); verify_hash("fineweb_data/fineweb_val.pt")'
args=(--attention-backend fa2 --activation-checkpointing --device-batch-size 1 --run-name "$run_name"
      --logit-avg-dir "runs/$run_name/ensemble")
if [[ "$mode" != baseline ]]; then args+=(--pf-weight 0.02); fi
if [[ "$mode" == smoke ]]; then args+=(--max-steps "${TIGER_MAX_STEPS:-224}"); fi
torchrun --standalone --nproc_per_node="$gpu_count" train.py "${args[@]}"
echo "SLOWRUN_COMPLETED mode=$mode run=$run_name"
