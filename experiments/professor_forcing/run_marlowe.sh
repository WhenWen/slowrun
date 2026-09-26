#!/usr/bin/env bash
# Submit from the repository root with the guarded submit helper.
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000123
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=kaiyuew@stanford.edu
#SBATCH --output=runs/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mode=${1:?expected baseline, pf, smoke, or pair}
case "$mode" in
    pair)
        # Separate trainers/seeds/results, one physical node and shared compile cache.
        # Stop the comparison if either trainer fails; never retry automatically.
        bash "$0" baseline
        bash "$0" pf
        echo "SLOWRUN_PAIR_COMPLETED job=$SLURM_JOB_ID"
        exit 0
        ;;
    baseline|pf|smoke) ;;
    *) exit 2 ;;
esac
run_name="pf-${mode}-${SLURM_JOB_ID}"
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="/tmp/slowrun-${USER}-${SLURM_JOB_ID}/cuda"
export TORCHINDUCTOR_CACHE_DIR="/tmp/slowrun-${USER}-${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/slowrun-${USER}-${SLURM_JOB_ID}/triton"
mkdir -p "$TIKTOKEN_CACHE_DIR" "$HF_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "runs/$run_name"
hostname
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv
git rev-parse HEAD
python -c 'import torch; print("torch", torch.__version__); assert torch.cuda.device_count() == 8; assert torch.cuda.get_device_capability()[0] == 9'
python -c 'from prepare_data import verify_hash; verify_hash("fineweb_data/fineweb_train.pt"); verify_hash("fineweb_data/fineweb_val.pt")'
args=(--run-name "$run_name" --logit-avg-dir "runs/$run_name/ensemble")
if [[ "$mode" != baseline ]]; then args+=(--pf-weight 0.02); fi
if [[ "$mode" == smoke ]]; then args+=(--max-steps 224); fi
torchrun --standalone --nproc_per_node=8 train.py "${args[@]}"
echo "SLOWRUN_COMPLETED mode=$mode job=$SLURM_JOB_ID"
