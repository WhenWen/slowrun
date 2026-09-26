#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$TIKTOKEN_CACHE_DIR" fineweb_data
python -c 'import torch, tiktoken, datasets, kernels; print(torch.__version__); tiktoken.get_encoding("gpt2")'
if [[ -f fineweb_data/fineweb_train.pt && -f fineweb_data/fineweb_val.pt ]]; then
    python -c 'from prepare_data import verify_hash; verify_hash("fineweb_data/fineweb_train.pt"); verify_hash("fineweb_data/fineweb_val.pt")'
else
    python prepare_data.py
fi
echo SLOWRUN_DATA_READY
