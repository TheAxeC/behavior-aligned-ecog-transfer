#!/bin/bash
#SBATCH --job-name=ward-anat-ablation
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/faesayl/projects/ward/logs/%x_%j.out
# Calibration-amount curve (cross-subject ρ vs minutes of target calibration). Reuses afnet env.
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
/usr/bin/python3 -u -m experiments.run_anat_ablation
echo "[$(date +%T)] DONE"
