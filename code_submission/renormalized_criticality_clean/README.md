# Renormalized Criticality Pipeline

This repository is a **reproducible Monte Carlo + analysis pipeline** for interacting renewal / escape-rate populations.

The pipeline is built around two connected tasks:

1) **Renormalized criticality**: estimate an operational critical coupling \(J_c^{\mathrm{ren}}(N)\) from finite-size simulations (susceptibility + correlation-time proxies).

2) **Information–energy Pareto optimization** near criticality: optimize decoding performance under an energy penalty, and test how the optimal distance \(\Delta^* = J_c - J^*\) scales.

It is structured so you can run it on a workstation *or* scale it on a cluster / DGX node.

---

## Install (recommended)

From the repo root:

```bash
python -m pip install -r env/requirements.txt
```

Important dependencies:
- numpy, scipy, pandas, matplotlib
- numba (simulation acceleration)
- scikit-learn (for safe cross-validated information estimator)

---

## Always set PYTHONPATH

From the repo root:

```bash
export PYTHONPATH="$(pwd)"
```

This avoids `ModuleNotFoundError: No module named 'src'`.

---

## Smoke tests (fast sanity check)

### Jc smoke test
```bash
export PYTHONPATH="$(pwd)"
python scripts/run_jc_grid.py --config configs/base.yaml --smoke_test --outdir results/_smoke_jc
python scripts/merge_jc_results.py --outdir results/_smoke_jc --tau_quantile 0.9 --stable_frac_threshold 0.6 --bootstrap
```

### Optimization smoke test
```bash
export PYTHONPATH="$(pwd)"
python scripts/run_opt_grid.py --config configs/base.yaml --smoke_test --outdir results/_smoke_opt
```

---

## Stage 1: Coarse Jc sweep (parallel)

This launches separate processes per (N, seed). Defaults:
- \(N \in \{1000,2000,5000,10000\}\)
- seeds \(\in \{1,2,3\}\)

```bash
export PYTHONPATH="$(pwd)"
bash scripts/launch_jc_coarse_parallel.sh
```

Or customize:
```bash
bash scripts/launch_jc_coarse_parallel.sh --outdir results/jc_coarse_demo --Ns 1000,2000 --seeds 1,2,3 --max_jobs 8
```

When done, merge:
```bash
python scripts/merge_jc_results.py --outdir results/jc_coarse_demo --tau_quantile 0.9 --stable_frac_threshold 0.67 --bootstrap
```

Outputs:
- `jc_curves_agg.csv`, `jc_scaling_summary.csv`
- diagnostic plots `curves_N*.png`
- scaling plot `jc_scaling.png`

---

## Stage 2: Refined Jc sweep (parallel around measured centers)

Uses the coarse `jc_scaling_summary.csv` to pick a center per N, then runs a refined grid around it.

```bash
bash scripts/launch_jc_refine_parallel.sh --coarse_outdir results/jc_coarse_demo
```

You can control the refinement window and simulation horizon:
```bash
bash scripts/launch_jc_refine_parallel.sh \
  --coarse_outdir results/jc_coarse_demo \
  --seeds 1,2,3,4,5 \
  --half_width 0.20 \
  --refine_T 60 \
  --refine_dt 0.0002 \
  --max_jobs 8
```

Merge:
```bash
python scripts/merge_jc_results.py --outdir results/jc_refine_YYYYMMDD_HHMMSS --tau_quantile 0.9 --stable_frac_threshold 0.6 --bootstrap
```

---

## Stage 3: Information–energy optimization near Jc

This sweeps a J-grid near `Jc_used`, runs the optimizer, and writes:
- `opt_rows.csv` (incrementally, as it runs)
- `opt_summary.csv`
- plots: `objective_vs_J_*.png`, `objective_vs_Delta_*.png`, `Jstar_vs_betaE_*.png`, `Deltastar_vs_betaE_*.png`

Example (single N, use refined Jc):
```bash
export PYTHONPATH="$(pwd)"

python -u scripts/run_opt_grid.py \
  --config configs/base.yaml \
  --jc_outdir results/jc_refine_YYYYMMDD_HHMMSS \
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
  --outdir results/opt_prl_N5000_run1
```

**Reality check (be honest with yourself):**  
Full optimization at \(N=5000\) with \(T=120\) and \(dt=10^{-4}\) is expensive. Start with shorter `--T_opt` / larger `--dt_opt`, confirm qualitative behavior, then increase precision.

---

## Stage 4: Convenience figure regeneration

```bash
python scripts/make_figures.py --auto --outdir results/figs
```

---

## Monitoring long jobs (server)

Typical checks:
```bash
# is it still running?
pgrep -af "run_jc_grid.py" | head
pgrep -af "run_opt_grid.py" | head

# output accumulating?
ls -lh results/jc_coarse_*/rows_*.csv | tail
tail -f results/opt_prl_*/run.log
```

---

## Notes on reproducibility

- Every run writes a `run_manifest.json` with config + git hash (when available).
- Optimization results are appended to CSV *during* the run to prevent data loss.

