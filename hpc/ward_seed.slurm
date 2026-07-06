#!/bin/bash
#SBATCH --job-name=ward-seed
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/faesayl/projects/ward/logs/%x_%A_%a.out
# full-9 mean incl cc/wm + seed-level CI: one array task per seed.
set -e
STAGE=/local/$USER/afnet3
mkdir -p "$STAGE"
( flock 9
  if [ ! -f "$STAGE/.ready" ]; then
    tar xzf ~/envs/afnet_env3.tar.gz -C "$STAGE"; touch "$STAGE/.ready"
  fi
) 9>/tmp/afnet_env3.lock
export PYTHONPATH="$STAGE/afnet_env/lib/python3.10/site-packages"
cd ~/projects/ward/code
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "[$(date +%T)] node $(hostname) task=$SLURM_ARRAY_TASK_ID"
/usr/bin/python3 -u -m experiments.run_robustness seedsweep "$SLURM_ARRAY_TASK_ID"
echo "[$(date +%T)] DONE"
