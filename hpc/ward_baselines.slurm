#!/bin/bash
#SBATCH --job-name=ward-baselines
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/home/faesayl/projects/ward/logs/%x_%j.out
# Same-protocol baselines: behavioral alignment vs first-43 no-align vs first-43+EA. Reuses afnet env.
set -e
STAGE=/local/$USER/afnet3
mkdir -p "$STAGE"
( flock 9
  if [ ! -f "$STAGE/.ready" ]; then tar xzf ~/envs/afnet_env3.tar.gz -C "$STAGE"; touch "$STAGE/.ready"; fi
) 9>/tmp/afnet_env3.lock
export PYTHONPATH="$STAGE/afnet_env/lib/python3.10/site-packages"
cd ~/projects/ward/code
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "[$(date +%T)] node $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
/usr/bin/python3 -u -m experiments.run_baselines
echo "[$(date +%T)] DONE"
