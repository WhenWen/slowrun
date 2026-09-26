#!/usr/bin/env bash
set -euo pipefail
mode=${1:?expected baseline, pf, or smoke}
case "$mode" in baseline|pf|smoke) ;; *) exit 2 ;; esac
name="slowrun-pf-$mode"
mkdir -p runs
# A local lock and queue check prevent duplicate submission from this checkout.
exec 9>runs/submit.lock
flock -n 9 || { echo 'Submission already in progress'; exit 1; }
jobs=$(squeue -u "$USER" -h -o '%j')
if [[ "$jobs" == *"$name"* ]]; then
    echo "Refusing duplicate: $name already present in queue"
    exit 1
fi
bash -n experiments/professor_forcing/run_marlowe.sh
python -m py_compile train.py professor_forcing.py
sbatch --job-name="$name" experiments/professor_forcing/run_marlowe.sh "$mode"
squeue -u "$USER" -n "$name" -o '%.18i %.32j %.10P %.9T %.10M %.6D %R'
