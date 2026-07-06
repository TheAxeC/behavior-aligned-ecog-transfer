"""Rebuild the paper's artifacts.

Ward is an HPC project: the headline TABLES come from the SLURM jobs in hpc/ (each writes a CSV to
results/). This local entry point regenerates the FIGURES from the data + the committed result CSVs.
Run from this directory.

    python paper.py            # all 4 figures -> results/interp/ (Figures 1-3 need ~/data/, see README)

HPC tables (run on the cluster, partition main-gpu; one command submits all 8 jobs):
    bash hpc/run_all.sh    -> results/ward_*.csv  (headline Table 2 + Table 3 ablation + calibration curve
                              + the robustness: full-9 mean, permute control, seed CI, donor bars)
    (per-job breakdown + the runner/CSV/manuscript mapping are in the README "Reproduce" table)
"""
import subprocess
import sys

# (label, module, needs_raw_data) - the 4 manuscript figures and what regenerates each
FIGS = [
    ("Figure 1, method overview", "experiments.plot_method", True),
    ("Figure 2, method detail", "experiments.plot_method_detail", True),
    ("Figure 3, finger spatial maps (interpretability)", "experiments.plot_interp", True),
    ("Figure 4, calibration curve (from the committed CSV, no raw data needed)", "experiments.plot_calib", False),
]


def main():
    print("== figures -> results/interp/ (the first three need ~/data/stanford_fingerflex) ==", flush=True)
    for label, mod, _ in FIGS:
        print(f"\n-- {label} --", flush=True)
        subprocess.run([sys.executable, "-m", mod], check=True)
    print("\nDONE. The headline tables come from the hpc/*.slurm jobs (see this file's header).")


if __name__ == "__main__":
    main()
