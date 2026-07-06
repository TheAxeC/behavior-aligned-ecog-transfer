# Behavior-supervised alignment for cross-subject ECoG finger decoding

Code for *"Behavior-Supervised Alignment for Cross-Subject Neural Decoding across Non-Corresponding Electrode Grids"* (W. Ceyssens, A. Faes). Every ECoG implant places a different electrode grid over a different patch of cortex, so two subjects share no channel index and no geometry. The method bridges them with behavior: a per-subject spatial filter projects each grid onto shared axes defined by the decode target itself ("the cortical projection of finger f"), and a single shared temporal convolutional network (TCN), trained on donor subjects, decodes a new subject after a few minutes of calibration. On the Stanford fingerflex dataset (leave-one-subject-out, 7 subjects), minimal-calibration cross-subject mean Pearson r is 0.566, comparable to the CORTEG foundation model (0.554) while staying interpretable and pretraining-free.

## What the contribution is

The contribution is the **alignment**, not the decoder (a stock dilated-causal TCN). The shared axes are defined by behavior, so they mean the same thing across subjects whose grids share nothing; a same-protocol ablation shows that generic statistical alignment (Euclidean Alignment), and even a principled anatomical-coordinate alignment, cannot bridge non-corresponding grids, while behavior-supervised alignment can. Because each axis is a per-channel correlation between high-gamma activity and a finger's flexion, the shared space is a set of forward cortical maps one can plot and read against known motor somatotopy.

## Layout

This is a Python package; run modules from this directory (`python -m experiments.run_baselines`, etc.).

```
core/          the method (importable library)
  run_xsubject_hd.py   data loading, features (spectral bands + CAR), and the behavior-aligned
                       spatial filter `spatial_filters` (the alignment)
  seq_decoder.py       the shared TCN decoder + donor-pretrain / minimal-calibration training
  bttn_finger.py       the aligned spectro-spatial tensor builder (set_preset, aligned_TCB)
experiments/   the runners that produce the results + figures
  run_loso_tcn.py        LOSO TCN sweep         -> results/ward_tcn_loso_<0-3>.csv
  run_baselines.py       same-protocol ablation -> results/ward_baselines.csv
  run_calib_curve.py     calibration curve      -> results/ward_calib_curve.csv
  run_anat_ablation.py anatomical-align + donor-count -> results/ward_anat_ablation.csv
  run_robustness.py       robustness (full9ens | permute | seedsweep | donorbars) -> results/ward_*.csv
  plot_method.py, plot_method_detail.py, plot_interp.py, plot_calib.py   the figures -> results/interp/
hpc/           the SLURM scripts (.slurm) for the cluster runs
results/       generated CSVs + figures (regenerable; not in the deposit)
paper.py       rebuild the figures
```

The algorithm lives in `core/`: the alignment in `run_xsubject_hd.py::spatial_filters`, the decoder in `seq_decoder.py::TCN`.

## Reproduce

This is an HPC project: the result tables are GPU leave-one-subject-out sweeps (one SLURM job each on the UTwente cluster), and the four figures draw locally on any machine. Three steps.

**1. Set up (once).**
```bash
pip install -r requirements.txt          # numpy scipy scikit-learn torch matplotlib
```
- Build the env tarball `~/envs/afnet_env3.tar.gz` and sync this `code/` to `~/projects/ward/code` (each job stages the env to node-local `/local`; re-sync after any edit).
- Download the data (public): Stanford fingerflex library to `~/data/stanford_fingerflex/<c>/<c>_fingerflex.mat` (Stanford Digital Repository PURL `zk881ps0522`, CC BY-SA).

**2. Run every table (one command, on the cluster).**
```bash
bash hpc/run_all.sh        # submits all 8 GPU jobs, then exits; watch: squeue -u $USER
```
Each job writes one CSV to `results/`:

| SLURM job | runner | output CSV | manuscript |
|---|---|---|---|
| `ward_tcn.slurm` (array 0-3) | `run_loso_tcn <0-3>` | `ward_tcn_loso_<0-3>.csv` | Table 2; config 2 (`full`/R4/w128) = the headline r 0.566 |
| `ward_baselines.slurm` | `run_baselines` | `ward_baselines.csv` | Table 3 (alignment ablation) |
| `ward_calib.slurm` | `run_calib_curve` | `ward_calib_curve.csv` | calibration curve |
| `ward_anat_ablation.slurm` | `run_anat_ablation` | `ward_anat_ablation.csv` | anatomical-align + donor-count |
| `ward_full9ens.slurm` | `run_robustness full9ens` | `ward_full9ens.csv` | full-9 auditable mean (0.550, matched to CORTEG) |
| `ward_permute.slurm` | `run_robustness permute` | `ward_permute.csv` | shared-vs-permuted control (p=0.016) |
| `ward_seed.slurm` (array 0-11) | `run_robustness seedsweep <s>` | `ward_seed<NN>.csv` | 12-seed CI |
| `ward_donor.slurm` (array 1,2,4,6) | `run_robustness donorbars <k>` | `ward_donorbars_k<N>.csv` | donor-count curve |

The `run_loso_tcn` argument `<0-3>` picks one config from a 4-way sweep (`hd` vs `full` features, width 128 vs 192); config 2 (`full`/R4/width 128) is the headline, the other three justify that choice. The committed `results/ward_tcn_full_w128.csv` is exactly that config-2 output (renamed). GPU runs are nondeterministic: expect the means in-band (~0.56), not byte-exact. To run one job by hand instead of all: `python -m experiments.run_loso_tcn 2`, etc.

**3. Draw the figures (locally, no GPU).**
```bash
python paper.py        # all 4 figures -> results/interp/  (Figures 1-3 need ~/data/stanford_fingerflex)
```

| figure | script | reads |
|---|---|---|
| Figure 1, method overview | `experiments/plot_method.py` | `~/data/stanford_fingerflex` |
| Figure 2, method detail | `experiments/plot_method_detail.py` | `~/data/stanford_fingerflex` |
| Figure 3, finger spatial maps | `experiments/plot_interp.py` | `~/data/stanford_fingerflex` |
| Figure 4, calibration curve | `experiments/plot_calib.py` | `results/ward_calib_curve.csv` (no raw data) |

## Method

The alignment (`core/run_xsubject_hd.py::spatial_filters`): per finger f, the first filter is the l2-normalized per-channel correlation between high-gamma envelope and finger-f flexion (a forward/activation pattern); the remaining R-1 filters are the leading singular vectors of the correlation-magnitude-reweighted high-gamma covariance, sign-aligned to the correlation pattern. With R=4 this gives 5R=20 shared components. Features are spectral power in 24 bands spanning 2-300 Hz at a 100 Hz rate, common-average referenced. The decoder (`core/seq_decoder.py::TCN`) is a dilated causal TCN, dilations 1 to 64 (about a one-second receptive field), width 128, trained on the pooled donors and lightly fine-tuned on a few minutes of target calibration, averaged over three seeds. Evaluation is leave-one-subject-out, mean Pearson r over five fingers, with validation-split model selection (never test-set max).

License: MIT (`LICENSE`).
