# Optimal Information–Energy Tradeoffs Lie Below Operational Criticality

  
**Author:** Nalin Dhiman, IIT Mandi

---

## Table of Contents

1. [Overview and Scientific Claim](#1-overview-and-scientific-claim)
2. [Repository Layout](#2-repository-layout)
3. [Environment Setup](#3-environment-setup)
4. [⚠️ Path Configuration — Read This First](#4-️-path-configuration--read-this-first)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Stage 0: Smoke Tests (Fast Sanity Check)](#6-stage-0-smoke-tests-fast-sanity-check)
7. [Stage 1: Coarse Jc Sweep](#7-stage-1-coarse-jc-sweep)
8. [Stage 2: Refined Jc Sweep](#8-stage-2-refined-jc-sweep)
9. [Stage 3: Information–Energy Optimization Near Jc](#9-stage-3-informationenergy-optimization-near-jc)
10. [Stage 4: Figure Regeneration](#10-stage-4-figure-regeneration)
11. [Reviewer-Driven Robustness Analyses](#11-reviewer-driven-robustness-analyses)
    - [analysis_0.0.2 — Stability & Mechanism Diagnostics](#analysis_002--stability--mechanism-diagnostics)
    - [analysis_0.0.3 — Full Robustness Suite](#analysis_003--full-robustness-suite)
12. [Manuscript Figures Index](#12-manuscript-figures-index)
13. [Monitoring Long-Running Jobs](#13-monitoring-long-running-jobs)
14. [Reproducibility Guarantees](#14-reproducibility-guarantees)
15. [Known Caveats](#15-known-caveats)
16. [Citation](#16-citation)

---

## 1. Overview and Scientific Claim

This package provides the complete, reproducible simulation and analysis pipeline supporting the Physical Review E manuscript:

> **Optimal information–energy tradeoffs occupy a finite subcritical regime below operational criticality in a driven renewal network.**

Specifically, the manuscript argues — conservatively — that when a finite renewal/escape-rate population is jointly optimized for decoding performance and metabolic cost, the optimal coupling **J\*** consistently falls **below** the renormalized critical coupling **J\_c^{ren}(N)** measured from finite-size simulations. It does **not** claim a universal constant offset or a clean asymptotic scaling law.

The pipeline is built around two connected tasks:

1. **Renormalized criticality estimation**: Measure the operational critical coupling J_c^{ren}(N) from finite-size Monte Carlo simulations using susceptibility and correlation-time proxies.
2. **Information–energy Pareto optimization near criticality**: Optimize decoding performance under an energy penalty, and characterize how the optimal distance Δ\* = J_c − J\* depends on system size N and energy weight β_E.

The system can be run on a workstation or scaled to a cluster/DGX node.

---

## 2. Repository Layout

```
Optimal-information-energy/
│
├── README.md                          ← This file (comprehensive entry point)
├── README_submission.md               ← Short legacy readme (superseded)
├── reproduce_figures.sh               ← Helper noting figure source directories
│
├── manuscript/
│   ├── bibliography.bib               ← BibTeX references
│   ├── figures/                       ← All figures used in the manuscript and SI
│   │   ├── Fig1_objective_landscapes_clean.{pdf,png}
│   │   ├── Fig2_regime_summary_clean.{pdf,png}
│   │   ├── Fig3_mechanism_stability_clean.{pdf,png}
│   │   ├── Fig4_robustness_clean.{pdf,png}
│   │   ├── FigS1_critical_markers_clean.{pdf,png}
│   │   └── FigR_*.{pdf,png}           ← Reviewer response figures
│   └── tables/                        ← CSV tables from revision analyses
│
├── code_submission/
│   ├── make_fig1.py                   ← Standalone figure-generation scripts
│   ├── make_fig2.py
│   ├── make_fig3.py
│   ├── make_fig4.py
│   ├── make_figS1.py
│   ├── make_figR_betaC.py
│   ├── make_figR_decoder.py
│   ├── make_figR_filter.py
│   ├── make_figR_input_stats.py
│   ├── make_figR_matched.py
│   ├── make_figR_nonnormality.py
│   ├── make_figR_supracritical.py
│   ├── make_figR_uncertainty.py
│   │
│   └── renormalized_criticality_clean/   ← Core simulation + analysis pipeline
│       ├── README.md                     ← Legacy pipeline readme (merged here)
│       ├── configs/
│       │   └── base.yaml                 ← Default simulation hyperparameters
│       ├── env/
│       │   └── requirements.txt          ← Python dependencies
│       ├── src/                          ← Core library modules
│       │   ├── simulate_mc.py            ← Monte Carlo renewal simulation (Numba)
│       │   ├── criticality.py            ← Susceptibility + tau estimators
│       │   ├── stimulus.py               ← Ornstein–Uhlenbeck stimulus generator
│       │   ├── info_estimators.py        ← Cross-validated mutual information
│       │   ├── optimize.py               ← Objective & optimizer wrappers
│       │   ├── theory_hooks.py           ← Theoretical reference values
│       │   └── utils.py                  ← Shared utilities
│       ├── scripts/                      ← Executable pipeline scripts
│       │   ├── run_jc_grid.py            ← Per-(N,seed) Jc grid runner
│       │   ├── merge_jc_results.py       ← Aggregate + bootstrap Jc estimates
│       │   ├── launch_jc_coarse_parallel.sh  ← Parallel launcher (coarse)
│       │   ├── launch_jc_refine_parallel.sh  ← Parallel launcher (refined)
│       │   ├── run_opt_grid.py           ← Optimization sweep near Jc
│       │   └── make_figures.py           ← Automated figure regeneration
│       ├── results/                      ← Pre-computed result snapshots
│       │   ├── jc_refine_20260304_094704/
│       │   ├── opt_N2000_plateau_refine_20260318_115014/
│       │   ├── opt_N5000_plateau_precision_20260321_192439/
│       │   ├── opt_N8000_scale_20260319_150812/
│       │   ├── opt_N10000_scale_20260319_150818/
│       │   └── opt_N12000_scale_20260319_150824/
│       ├── analysis_0.0.2/              ← Reviewer-driven: stability diagnostics
│       │   ├── scripts/
│       │   │   ├── classify_plateaus.py
│       │   │   ├── marker_sensitivity_fixed.py
│       │   │   ├── mechanism_diagnostics_and_fix.py
│       │   │   └── tuned_branch_stability_clean.py
│       │   ├── figures/
│       │   └── tables/
│       └── analysis_0.0.3/             ← Reviewer-driven: full robustness suite
│           ├── scripts/
│           │   ├── step1_uncertainty.py
│           │   ├── step2_matched_protocol.py
│           │   ├── step3_filter_effect.py
│           │   ├── step4_betaC_fast.py
│           │   ├── step4_run_betaC_sweep.sh
│           │   ├── step5_nonnormality.py
│           │   ├── step6_decoder_robustness.py
│           │   ├── step7_tau_fast.py
│           │   ├── step7_extract.py
│           │   ├── step7_run_tau_sweep.sh
│           │   └── step8_supracritical_probe.py
│           ├── results_betaC/
│           ├── results_tauC/
│           ├── figures/
│           └── tables/
```

---

## 3. Environment Setup

**Python 3.9+ is required.** A virtual environment is strongly recommended.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
pip install -r env/requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical arrays and statistics |
| `pandas` | CSV result accumulation and processing |
| `numba` | JIT-compiled simulation (large speedup) |
| `matplotlib` | All figure generation |
| `pyyaml` | Config file parsing |
| `scikit-learn` | Cross-validated mutual information estimator |

> **Note on Numba:** The first run will be slow due to JIT compilation. Subsequent runs on the same machine will be significantly faster. This is expected behavior.

---

## 4. ⚠️ Path Configuration — Read This First

**All paths in this pipeline are user-specified at runtime. There are no hardcoded absolute paths in the source code.**

However, **you must always**:

### 4a. Set PYTHONPATH before running any script

Every script imports from `src/`. Without this, you will get `ModuleNotFoundError: No module named 'src'`.

```bash
# Navigate to the pipeline root first
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean

# Set PYTHONPATH to the current directory
export PYTHONPATH="$(pwd)"
```

Replace `/YOUR/PATH/TO/` with the **actual absolute path on your machine**.

### 4b. Always specify --outdir explicitly

Every script accepts `--outdir` to control where results are written. **Do not rely on defaults in shared or cluster environments.** Example:

```bash
# Good: explicit output directory
python scripts/run_jc_grid.py \
  --config configs/base.yaml \
  --outdir /YOUR/PATH/TO/results/jc_coarse_myrun

# Risky: relies on default outdir relative to cwd
python scripts/run_jc_grid.py --config configs/base.yaml
```

### 4c. When using pre-computed results

If you want to skip re-running the simulation and use the pre-packaged result snapshots (in `results/`), point `--jc_outdir` at the snapshot:

```bash
python scripts/run_opt_grid.py \
  --config configs/base.yaml \
  --jc_outdir /YOUR/PATH/TO/results/jc_refine_20260304_094704 \
  --outdir /YOUR/PATH/TO/results/opt_myrun \
  ...
```

---

## 5. Pipeline Overview

The pipeline has four sequential stages. Stages 1–2 estimate criticality; Stage 3 finds the optimal coupling. Stage 4 regenerates figures.

```
Stage 1: Coarse Jc sweep → Stage 2: Refined Jc sweep → Stage 3: Opt. near Jc → Stage 4: Figures
```

Each stage writes results incrementally (CSV rows appended during the run) to prevent data loss on cluster preemptions.

---

## 6. Stage 0: Smoke Tests (Fast Sanity Check)

Run these first to verify your environment is correct. They complete in minutes.

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

# Jc smoke test — runs a tiny grid and merges
python scripts/run_jc_grid.py \
  --config configs/base.yaml \
  --smoke_test \
  --outdir results/_smoke_jc

python scripts/merge_jc_results.py \
  --outdir results/_smoke_jc \
  --tau_quantile 0.9 \
  --stable_frac_threshold 0.6 \
  --bootstrap

# Optimization smoke test
python scripts/run_opt_grid.py \
  --config configs/base.yaml \
  --smoke_test \
  --outdir results/_smoke_opt
```

Expected output: CSVs in `results/_smoke_jc/` and `results/_smoke_opt/`, no errors.

---

## 7. Stage 1: Coarse Jc Sweep

Runs separate processes per (N, seed) to estimate the critical coupling across system sizes.

**Default sweep:** N ∈ {1000, 2000, 5000, 10000}, seeds ∈ {1, 2, 3}.

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

# Default launch
bash scripts/launch_jc_coarse_parallel.sh

# Or customize: choose N values, seeds, and parallelism
bash scripts/launch_jc_coarse_parallel.sh \
  --outdir results/jc_coarse_$(date +%Y%m%d_%H%M%S) \
  --Ns 1000,2000,5000 \
  --seeds 1,2,3 \
  --max_jobs 8
```

**Merge results after all jobs finish:**

```bash
python scripts/merge_jc_results.py \
  --outdir results/jc_coarse_YYYYMMDD_HHMMSS \
  --tau_quantile 0.9 \
  --stable_frac_threshold 0.67 \
  --bootstrap
```

**Outputs written to `--outdir`:**

| File | Description |
|---|---|
| `jc_curves_agg.csv` | Raw aggregated susceptibility/tau curves per (N, seed) |
| `jc_scaling_summary.csv` | Estimated Jc per N (used by Stage 2) |
| `curves_N*.png` | Diagnostic plots of the susceptibility/tau curves |
| `jc_scaling.png` | Finite-size scaling plot of Jc vs N |

---

## 8. Stage 2: Refined Jc Sweep

Uses the coarse `jc_scaling_summary.csv` to pick a center per N, then runs a denser grid in a narrow window around it. This is where the final Jc values used in the paper come from.

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

bash scripts/launch_jc_refine_parallel.sh \
  --coarse_outdir results/jc_coarse_YYYYMMDD_HHMMSS

# Full control over refinement window and simulation horizon:
bash scripts/launch_jc_refine_parallel.sh \
  --coarse_outdir results/jc_coarse_YYYYMMDD_HHMMSS \
  --seeds 1,2,3,4,5 \
  --half_width 0.20 \
  --refine_T 60 \
  --refine_dt 0.0002 \
  --max_jobs 8
```

**Merge:**

```bash
python scripts/merge_jc_results.py \
  --outdir results/jc_refine_YYYYMMDD_HHMMSS \
  --tau_quantile 0.9 \
  --stable_frac_threshold 0.6 \
  --bootstrap
```

> A pre-computed refined snapshot is included at `results/jc_refine_20260304_094704/` — you may use this directly in Stage 3 if you do not want to re-run Stages 1–2.

---

## 9. Stage 3: Information–Energy Optimization Near Jc

This sweeps a J-grid near the measured Jc, runs the optimizer for each (J, seed, β_E, τ_c) combination, and identifies J\* (the optimal coupling).

The run is designed to be **resumable**: pass `--resume` to continue interrupted runs without recomputing completed parameter combinations.

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

python -u scripts/run_opt_grid.py \
  --config configs/base.yaml \
  --jc_outdir results/jc_refine_20260304_094704 \
  --N 5000 \
  --J_center_source tau \
  --J_window 0.20 \
  --J_points 13 \
  --J_extra "0.00,-0.02,-0.01,0.01,0.02" \
  --seeds "1,2,3" \
  --tau_c_list "0.05" \
  --beta_E_list "0.01,0.1" \
  --T_opt 60 \
  --dt_opt 0.0002 \
  --n_restarts 3 \
  --maxiter 50 \
  --resume \
  --outdir results/opt_N5000_myrun
```

**Key flags:**

| Flag | Description |
|---|---|
| `--jc_outdir` | Path to a merged Jc result directory (Stages 1–2 output or pre-packaged) |
| `--N` | System size to optimize |
| `--J_center_source` | Which Jc proxy to center the grid on: `tau` or `chi` |
| `--J_window` | Total width of J sweep window (centered on Jc) |
| `--J_points` | Number of equally-spaced J values in the window |
| `--J_extra` | Comma-separated offsets from Jc to add (e.g., `0.00,-0.02`) |
| `--beta_E_list` | Comma-separated energy penalty weights to sweep |
| `--tau_c_list` | Comma-separated stimulus correlation times to sweep |
| `--T_opt` / `--dt_opt` | Simulation horizon and timestep for each optimization trial |
| `--n_restarts` | Number of random restarts per parameter combination |
| `--resume` | Skip already-completed (J, seed, …) combinations |

**Outputs written to `--outdir`:**

| File | Description |
|---|---|
| `opt_rows.csv` | Incremental per-trial results (written during run) |
| `opt_summary.csv` | Aggregated optimal J\* per (N, β_E, τ_c) |
| `objective_vs_J_*.png` | Objective landscape as a function of J |
| `objective_vs_Delta_*.png` | Objective landscape vs Δ = J_c − J |
| `Jstar_vs_betaE_*.png` | Optimal J\* vs energy penalty β_E |
| `Deltastar_vs_betaE_*.png` | Optimal gap Δ\* vs β_E |
| `run_manifest.json` | Config + git hash snapshot for reproducibility |

> **Computational cost:** Full optimization at N=5000, T=120, dt=10⁻⁴ is expensive (hours to days depending on hardware). Start with shorter `--T_opt` (e.g., 30–60) and larger `--dt_opt` (e.g., 0.0002), confirm qualitative behavior, then increase precision for the final run. Pre-computed results for N ∈ {2000, 5000, 8000, 10000, 12000} are included in `results/`.

---

## 10. Stage 4: Figure Regeneration

### Option A — Convenience script (auto-discovers latest results)

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

python scripts/make_figures.py --auto --outdir results/figs
```

### Option B — Per-figure scripts (recommended for selective regeneration)

All figure scripts live in `code_submission/`. Each can be run standalone and must know where to load data from. **Edit the `DATA_DIR` or equivalent variable at the top of each script to point to your results directory.**

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission
export PYTHONPATH="$(pwd)/renormalized_criticality_clean"

# Example: regenerate Fig 1
python make_fig1.py

# Example: regenerate reviewer figure for beta_C sensitivity
python make_figR_betaC.py
```

**Figure scripts available:**

| Script | Produces |
|---|---|
| `make_fig1.py` | Fig 1 — Objective landscapes across J |
| `make_fig2.py` | Fig 2 — Regime summary (Δ\* vs β_E) |
| `make_fig3.py` | Fig 3 — Mechanism stability analysis |
| `make_fig4.py` | Fig 4 — Robustness composite |
| `make_figS1.py` | Fig S1 — Critical marker sensitivity |
| `make_figR_betaC.py` | FigR — β_C sensitivity |
| `make_figR_decoder.py` | FigR — Decoder robustness |
| `make_figR_filter.py` | FigR — Rate filter effect |
| `make_figR_input_stats.py` | FigR — Input statistics robustness |
| `make_figR_matched.py` | FigR — Matched protocol control |
| `make_figR_nonnormality.py` | FigR — Non-normality diagnostics |
| `make_figR_supracritical.py` | FigR — Supracritical probe |
| `make_figR_uncertainty.py` | FigR — Objective uncertainty bands |

---

## 11. Reviewer-Driven Robustness Analyses

These analyses were added during revision to directly address referee criticisms. They are self-contained and do **not** require re-running Stages 1–3.

### analysis_0.0.2 — Stability & Mechanism Diagnostics

Located at: `code_submission/renormalized_criticality_clean/analysis_0.0.2/`

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

# Classify optimization plateaus (flat vs well-defined optima)
python analysis_0.0.2/scripts/classify_plateaus.py \
  --opt_dir results/opt_N5000_plateau_precision_20260321_192439

# Sensitivity of Jc markers to estimator parameters
python analysis_0.0.2/scripts/marker_sensitivity_fixed.py \
  --jc_dir results/jc_refine_20260304_094704

# Diagnostic plots for mechanism: does the optimizer find genuine minima?
python analysis_0.0.2/scripts/mechanism_diagnostics_and_fix.py \
  --opt_dir results/opt_N5000_plateau_precision_20260321_192439

# Clean analysis of tuned-branch stability boundaries
python analysis_0.0.2/scripts/tuned_branch_stability_clean.py \
  --opt_dir results/opt_N5000_plateau_precision_20260321_192439
```

Outputs (figures and tables) are written to `analysis_0.0.2/figures/` and `analysis_0.0.2/tables/` by default.

---

### analysis_0.0.3 — Full Robustness Suite

Located at: `code_submission/renormalized_criticality_clean/analysis_0.0.3/`

This is a **sequential 8-step** robustness audit run in order. Each step's outputs feed subsequent steps.

```bash
cd /YOUR/PATH/TO/Optimal-information-energy/code_submission/renormalized_criticality_clean
export PYTHONPATH="$(pwd)"

OPT_DIR="results/opt_N5000_plateau_precision_20260321_192439"
JC_DIR="results/jc_refine_20260304_094704"

# Step 1: Uncertainty quantification of objective curves
python analysis_0.0.3/scripts/step1_uncertainty.py \
  --opt_dir $OPT_DIR

# Step 2: Matched-protocol control (same J, different random seed ensemble)
python analysis_0.0.3/scripts/step2_matched_protocol.py \
  --opt_dir $OPT_DIR

# Step 3: Rate filter effect — impact of spike-rate filtering on objective
python analysis_0.0.3/scripts/step3_filter_effect.py \
  --opt_dir $OPT_DIR

# Step 4: beta_C sensitivity sweep (fast version)
python analysis_0.0.3/scripts/step4_betaC_fast.py \
  --opt_dir $OPT_DIR
# Or run the full sweep via shell:
bash analysis_0.0.3/scripts/step4_run_betaC_sweep.sh $OPT_DIR

# Step 5: Non-normality of tuned Jacobian diagnostics
python analysis_0.0.3/scripts/step5_nonnormality.py \
  --opt_dir $OPT_DIR

# Step 6: Decoder robustness (linear vs. nonlinear decoders)
python analysis_0.0.3/scripts/step6_decoder_robustness.py \
  --opt_dir $OPT_DIR

# Step 7: tau_c sweep (stimulus correlation time sensitivity)
python analysis_0.0.3/scripts/step7_tau_fast.py \
  --opt_dir $OPT_DIR
# Extract summary table:
python analysis_0.0.3/scripts/step7_extract.py \
  --outdir analysis_0.0.3/results_tauC
# Or run full sweep:
bash analysis_0.0.3/scripts/step7_run_tau_sweep.sh $OPT_DIR

# Step 8: Supracritical probe — does the optimizer erroneously explore J > Jc?
python analysis_0.0.3/scripts/step8_supracritical_probe.py \
  --opt_dir $OPT_DIR --jc_dir $JC_DIR
```

Result CSVs are written to `analysis_0.0.3/results_betaC/` and `analysis_0.0.3/results_tauC/`. Figures to `analysis_0.0.3/figures/`.

---

## 12. Manuscript Figures Index

All production figures (used directly in the submitted manuscript) are in `manuscript/figures/`. The `_clean` suffix denotes the final publication-quality version.

| Figure | File | Description |
|---|---|---|
| Fig 1 | `Fig1_objective_landscapes_clean.{pdf,png}` | Objective landscape F(J) across coupling values |
| Fig 2 | `Fig2_regime_summary_clean.{pdf,png}` | Regime summary: Δ\* = J_c − J\* vs β_E |
| Fig 3 | `Fig3_mechanism_stability_clean.{pdf,png}` | Mechanism and stability boundary analysis |
| Fig 4 | `Fig4_robustness_clean.{pdf,png}` | Robustness composite across N and τ_c |
| Fig S1 | `FigS1_critical_markers_clean.{pdf,png}` | Sensitivity of Jc markers to estimator parameters |
| FigR | `FigR_uncertainty_*_clean.{pdf,png}` | Uncertainty bands on objective curves |
| FigR | `FigR_matched_protocol_control_*_clean.{pdf,png}` | Matched-protocol control |
| FigR | `FigR_filter_effect_*_clean.{pdf,png}` | Spike-rate filter effect |
| FigR | `FigR_betaC_sensitivity_*_clean.{pdf,png}` | β_C sensitivity |
| FigR | `FigR_nonnormality_vs_J_*_clean.{pdf,png}` | Jacobian non-normality vs J |
| FigR | `FigR_decoder_robustness_clean.{pdf,png}` | Decoder robustness |
| FigR | `FigR_input_stats_robustness_clean.{pdf,png}` | Input statistics robustness |
| FigR | `FigR_supracritical_probe_clean.{pdf,png}` | Supracritical probe |

---

## 13. Monitoring Long-Running Jobs

When running on a cluster or DGX node, use these shell commands to check progress:

```bash
# Is the Jc grid runner still active?
pgrep -af "run_jc_grid.py" | head

# Is the optimization runner still active?
pgrep -af "run_opt_grid.py" | head

# Are output rows accumulating? (check last-modified timestamps)
ls -lh results/jc_coarse_*/rows_*.csv | tail -20

# Tail the optimization run log
tail -f results/opt_*/run.log

# Quick disk usage check
du -sh results/*/
```

---

## 14. Reproducibility Guarantees

Every run of `run_jc_grid.py` and `run_opt_grid.py` writes a `run_manifest.json` to the output directory containing:

- Full config (loaded from `base.yaml` plus any CLI overrides)
- Git commit hash (when run inside a git repository)
- Python version
- Package versions of key dependencies
- Timestamp of run start

Optimization results are appended to `opt_rows.csv` **incrementally during the run**, so partial results are not lost if the job is killed or preempted.

The pre-packaged `results/` directories contain the `run_manifest.json` files from the original runs used for the paper, so the exact conditions can be inspected without re-running.

---

## 15. Known Caveats

1. **Numba warm-up:** The first call to any JIT-compiled function in `src/simulate_mc.py` will be slow (30–120 seconds). This is normal. Do not kill the process prematurely.

2. **Memory at large N:** At N=10000–12000, each simulation trial holds O(N·T/dt) spike-count arrays in memory. Ensure at least 16 GB RAM (32 GB recommended) for full-precision runs.

3. **Mutual information estimator variance:** The cross-validated MI estimator (`src/info_estimators.py`) has high variance at small sample sizes. Use at least T=60 at N≥2000 for reliable estimates. Smoke tests with T=10 will show qualitative trends only.

4. **Paths on Windows:** The pipeline has only been tested on Linux/macOS. Windows users will need to replace `export PYTHONPATH=...` with `set PYTHONPATH=...` (cmd) or `$env:PYTHONPATH = ...` (PowerShell), and path separators may need adjusting in shell scripts.

5. **The `--resume` flag:** When resuming an interrupted optimization run, the script reads completed (J, seed, β_E, τ_c) combinations from the existing `opt_rows.csv`. If you change simulation parameters mid-run, do **not** use `--resume` — start a fresh `--outdir` to avoid mixing results from different settings.

---

## 16. Citation

If you use this code or data in your own research, please cite:

```
Nalin Dhiman,
"Optimal information–energy tradeoffs lie below operational criticality,"
IIT Mandi.
```

BibTeX source: [`manuscript/bibliography.bib`](manuscript/bibliography.bib)

---

*Questions or issues: please open a GitHub issue on this repository.*
