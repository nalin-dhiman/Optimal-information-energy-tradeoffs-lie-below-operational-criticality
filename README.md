# Stochastic renewal-network simulator

This repository contains the stochastic renewal-network simulator and reproducibility code used to study task performance under recurrent parameter reoptimization.

The network has stochastic escape spikes, voltage reset, refractory ages, a filtered population activity, and two recurrent feedback routes. Code includes the byte-preserved simulator, verified observation and diagnostic kernels, a causal physical-time ridge reader, fixed-network replay scripts, tests, and compact measured result tables.

Use Python 3.12 and install the recorded dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/validate_archival_kernel.py
python scripts/reproduce_key_results.py --config configs/smoke.json
python scripts/reproduce_figures.py
```

The default smoke example uses 80 neurons and checks code execution. Its generated scores are not the scientific benchmark. `configs/stage3_selected_reference.json` specifies the selected 2,000-neuron recurrent network and separately optimized true no-recurrence baseline. Replaying those fixed networks is optional and takes longer:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/reproduce_key_results.py \
  --config configs/stage3_selected_reference.json \
  --out local_outputs/fixed_reference --save-raw
```

Readers use independent training, validation and test trajectories. Validation selects ridge strength; test trajectories evaluate the frozen choices. True no recurrence sets both `J=0` and `theta_A=0`. A zero voltage coupling with nonzero population-hazard feedback remains recurrent.

Large raw simulation arrays and optimization checkpoints are intentionally excluded. `--save-raw` regenerates local arrays under the ignored output directory. Compact CSV records retain relative data identifiers and source hashes. The complete historical datasets are held in the project's local cold archive; this repository is sufficient for fixed-network simulation and summary regeneration without those arrays.

Model equations, units and limitations are in [docs/MODEL.md](docs/MODEL.md). See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for reproduction scope and [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for table definitions. The bounded experiments support useful recurrence and broad task equivalence after retuning; they establish no unique computational critical point or universal compensation law. The study is closed.
