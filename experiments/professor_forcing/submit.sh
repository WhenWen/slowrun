#!/usr/bin/env bash
set -euo pipefail
mode=${1:?expected baseline, pf, smoke, or pair}
case "$mode" in baseline|pf|smoke|pair) ;; *) exit 2 ;; esac
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
extra=()
if [[ "$mode" == smoke ]]; then extra+=(--time=00:30:00); fi
if [[ "$mode" == pair ]]; then
    dependency=${2:?pair requires the successful smoke dependency job ID}
    [[ "$dependency" =~ ^[0-9]+$ ]] || { echo 'Invalid dependency job ID'; exit 2; }
    dependency_info=$(scontrol show job -o "$dependency")
    [[ "$dependency_info" == *"JobName=slowrun-pf-smoke "* &&
       "$dependency_info" == *"UserId=$USER("* ]] || {
        echo "Dependency must be this user's slowrun-pf-smoke job"; exit 2;
    }
    extra+=(--time=03:00:00 --dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
fi
sbatch --job-name="$name" "${extra[@]}" experiments/professor_forcing/run_marlowe.sh "$mode"
squeue -u "$USER" -n "$name" -o '%.18i %.32j %.10P %.9T %.10M %.6D %R'
