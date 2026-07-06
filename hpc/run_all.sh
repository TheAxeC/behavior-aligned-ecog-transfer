#!/bin/bash
# One command to reproduce every WARD table on the UTwente HPC. The eight jobs are INDEPENDENT (no
# dependencies): each stages the afnet env to node-local /local (flock-guarded) and writes one CSV to
# results/. This script submits them all and EXITS in seconds; SLURM runs them over the next hours and
# you can log out. When they finish, draw the four figures locally with `python paper.py` (no GPU).
#
#   bash hpc/run_all.sh        # submit all eight
#
# PREREQUISITES (built once; see README.md "Reproduce"):
#   - env tarball  ~/envs/afnet_env3.tar.gz     (staged to /local by each job)
#   - code synced to  ~/projects/ward/code      (re-rsync after any edit, then re-submit)
#   - data  ~/data/stanford_fingerflex/<c>/<c>_fingerflex.mat  (public; Stanford PURL zk881ps0522)
set -e

# Prerequisites that cannot be shipped. Fail LOUDLY here with the exact fix, never silently mid-job.
fail(){ echo "[run_all] MISSING PREREQUISITE: $1" >&2; echo "  fix: $2" >&2; exit 1; }
ENV_TARBALL="$HOME/envs/afnet_env3.tar.gz"
[ -f "$ENV_TARBALL" ] || fail "packed env $ENV_TARBALL" \
  "build the venv from requirements.txt on a login node (--no-cache-dir) and pack it to that path (see README 'Reproduce')"
STANFORD="$HOME/data/stanford_fingerflex"
ls "$STANFORD"/*/*_fingerflex.mat >/dev/null 2>&1 || fail "Stanford fingerflex .mat under $STANFORD/<subj>/<subj>_fingerflex.mat" \
  "download the public fingerflex.zip from the Stanford Digital Repository PURL zk881ps0522 (CC BY-SA) and extract per-subject into $STANFORD/"

mkdir -p "$HOME/projects/ward/logs"
cd "$(dirname "$0")/.."                 # -> code root (jobs cd to ~/projects/ward/code internally)
sb(){ printf '%-28s = %s\n' "$1" "$(sbatch --parsable "hpc/$1")"; }

echo "== headline + main tables =="
sb ward_tcn.slurm              # array 0-3: run_loso_tcn        -> results/ward_tcn_loso_<0-3>.csv  (config 2 = headline r 0.566, Table 2)
sb ward_baselines.slurm        #           run_baselines        -> results/ward_baselines.csv       (Table 3, alignment ablation)
sb ward_calib.slurm            #           run_calib_curve      -> results/ward_calib_curve.csv     (calibration curve)
sb ward_anat_ablation.slurm         #           run_anat_ablation  -> results/ward_anat_ablation.csv (anatomical-align + donor-count)

echo "== robustness (matched-count mean, controls, CI) =="
sb ward_full9ens.slurm    #           run_robustness full9ens  -> results/ward_full9ens.csv   (full-9 auditable mean 0.550)
sb ward_permute.slurm     #           run_robustness permute   -> results/ward_permute.csv    (shared-vs-permuted control, p=0.016)
sb ward_seed.slurm        # array 0-11: run_robustness seedsweep -> results/ward_seed<NN>.csv  (12-seed CI)
sb ward_donor.slurm       # array 1,2,4,6: run_robustness donorbars -> results/ward_donorbars_k<N>.csv (donor-count curve)

echo
echo "submitted all eight. watch: squeue -u $USER ; logs: ~/projects/ward/logs/ ; you can log out."
echo "when they finish, draw the figures locally (no GPU):  python paper.py   # -> results/interp/"
