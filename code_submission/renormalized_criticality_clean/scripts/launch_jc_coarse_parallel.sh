#!/usr/bin/env bash
set -euo pipefail


REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"


export PYTHONPATH="$REPO_DIR"


export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

MAX_JOBS="${MAX_JOBS:-10}"


OUTDIR=""
Ns="1000,2000,5000,10000"
SEEDS="1,2,3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2 ;;
    --Ns) Ns="$2"; shift 2 ;;             
    --seeds) SEEDS="$2"; shift 2 ;;        
    --max_jobs) MAX_JOBS="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 2 ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="results/jc_coarse_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTDIR/logs"

echo "Starting Coarse Sweep in $OUTDIR"
echo "REPO_DIR=$REPO_DIR"
echo "MAX_JOBS=$MAX_JOBS"
echo "Ns=$Ns"
echo "SEEDS=$SEEDS"

IFS=',' read -ra N_ARR <<< "$Ns"
IFS=',' read -ra S_ARR <<< "$SEEDS"

for N in "${N_ARR[@]}"; do
  for s in "${S_ARR[@]}"; do


    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
      sleep 2
    done

    log="$OUTDIR/logs/N${N}_seed${s}.log"

    nohup bash -lc "cd '$REPO_DIR' && PYTHONPATH='$REPO_DIR' nice -n 10 python -u scripts/run_jc_grid.py       --config configs/base.yaml       --outdir '$OUTDIR'       --N_list '$N'       --seed_list '$s'       --pass_mode coarse"       > "$log" 2>&1 &

    echo "Launched N=$N seed=$s -> $log"
  done
done

echo "Launched all jobs."
echo "To wait for all background jobs in this shell: wait"
echo "Merge after completion:"
echo "python scripts/merge_jc_results.py --outdir $OUTDIR --tau_quantile 0.9 --stable_frac_threshold 0.67 --bootstrap"
