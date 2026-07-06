#!/bin/bash
#SBATCH --job-name=ward-tcn-loso
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/faesayl/projects/ward/logs/%x_%A_%a.out
# TCN seq2seq decoder on the shared finger-aligned component space, LOSO sweep.
# One array task per config (hpc/run_loso_tcn.py GRID). Reuses afnet env (afnet code untouched).
set -e
STAGE=/local/$USER/afnet3
mkdir -p "$STAGE"
( flock 9
  if [ ! -f "$STAGE/.ready" ]; then
    echo "[$(date +%T)] extracting env to $STAGE ..."
    tar xzf ~/envs/afnet_env3.tar.gz -C "$STAGE"
    touch "$STAGE/.ready"
  fi
) 9>/tmp/afnet_env3.lock
export PYTHONPATH="$STAGE/afnet_env/lib/python3.10/site-packages"
cd ~/projects/ward/code
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "[$(date +%T)] node $(hostname) task=$SLURM_ARRAY_TASK_ID"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
/usr/bin/python3 -c "import torch; print('torch',torch.__version__,'cuda?',torch.cuda.is_available())"
/usr/bin/python3 -u -m experiments.run_loso_tcn "$SLURM_ARRAY_TASK_ID"
echo "[$(date +%T)] DONE"
