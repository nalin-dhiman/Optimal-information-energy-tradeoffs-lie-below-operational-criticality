#!/usr/bin/env bash
set -euo pipefail



REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"


export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"


MAX_JOBS="${MAX_JOBS:-8}"
SEEDS="${SEEDS:-1,2,3,4,5}"
HALF_WIDTH="${HALF_WIDTH:-0.20}"    
REFINE_T="${REFINE_T:-60.0}"
REFINE_DT="${REFINE_DT:-0.0002}"

COARSE_OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coarse_outdir) COARSE_OUTDIR="$2"; shift 2 ;;
    --max_jobs) MAX_JOBS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;          
    --half_width) HALF_WIDTH="$2"; shift 2 ;;
    --refine_T) REFINE_T="$2"; shift 2 ;;
    --refine_dt) REFINE_DT="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 2 ;;
  esac
done

if [[ -z "$COARSE_OUTDIR" ]]; then
  echo "Usage: $0 --coarse_outdir <COARSE_OUTDIR_PATH> [--max_jobs 8] [--seeds 1,2,3,4,5] [--half_width 0.2] [--refine_T 60] [--refine_dt 0.0002]"
  exit 2
fi

SUMMARY_FILE="${COARSE_OUTDIR}/jc_scaling_summary.csv"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "Error: cannot find summary file: $SUMMARY_FILE"
  exit 2
fi

timestamp=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/jc_refine_${timestamp}"
mkdir -p "${OUTDIR}/logs"

echo "Starting Refine Sweep"
echo "REPO_DIR=$REPO_DIR"
echo "OUTDIR=$OUTDIR"
echo "SUMMARY_FILE=$SUMMARY_FILE"
echo "MAX_JOBS=$MAX_JOBS"
echo "SEEDS=$SEEDS"
echo "HALF_WIDTH=$HALF_WIDTH  REFINE_T=$REFINE_T  REFINE_DT=$REFINE_DT"


count_running() {
  pgrep -af "run_jc_grid.py" | grep -F "${OUTDIR}" | wc -l
}

CENTERS_CSV="${OUTDIR}/_centers.csv"

python - <<PY
import pandas as pd, numpy as np
df = pd.read_csv("${SUMMARY_FILE}")

def pick_center(r):
    # Prefer new merge output, then old merge output
    for k in ["Jc_tau_q_mean","Jc_tau_mean","Jc_chi_mean"]:
        if k in r and pd.notna(r[k]):
            return float(r[k])
    return np.nan

centers=[]
for _, r in df.iterrows():
    N=int(r["N"])
    c=pick_center(r)
    if np.isnan(c): 
        continue
    centers.append({"N":N,"center":c})

if not centers:
    raise SystemExit("No valid centers found in summary file.")

pd.DataFrame(centers).to_csv("${CENTERS_CSV}", index=False)
print("Wrote centers to", "${CENTERS_CSV}")
PY

IFS=',' read -ra SEED_ARR <<< "$SEEDS"


while IFS=, read -r N CENTER; do
  if [[ "$N" == "N" ]]; then
    continue
  fi


  JMIN=$(python - <<PY
c=float("${CENTER}")
hw=float("${HALF_WIDTH}")
print(max(0.0, c-hw))
PY
)
  JMAX=$(python - <<PY
c=float("${CENTER}")
hw=float("${HALF_WIDTH}")
print(c+hw)
PY
)

  for s in "${SEED_ARR[@]}"; do
    while [[ "$(count_running)" -ge "${MAX_JOBS}" ]]; do
      sleep 2
    done

    log="${OUTDIR}/logs/N${N}_seed${s}.log"

    nohup bash -lc "cd '$REPO_DIR' && PYTHONPATH='$REPO_DIR' nice -n 10 python -u scripts/run_jc_grid.py       --config configs/base.yaml       --outdir '$OUTDIR'       --N_list '$N'       --seed_list '$s'       --pass_mode refine       --refine_min '$JMIN'       --refine_max '$JMAX'       --refine_T '$REFINE_T'       --refine_dt '$REFINE_DT'"       > "${log}" 2>&1 &

    echo "Launched N=$N seed=$s refine=[${JMIN},${JMAX}] -> ${log}"
  done
done < "${CENTERS_CSV}"

N_CENTERS=$(python - <<PY
import pandas as pd
df=pd.read_csv("${CENTERS_CSV}")
print(len(df))
PY
)
EXPECTED=$(( N_CENTERS * ${#SEED_ARR[@]} ))

echo "All refine jobs launched."
echo "Expected rows files: $EXPECTED (centers=$N_CENTERS × seeds=${#SEED_ARR[@]})"
echo "Monitor: tail -f ${OUTDIR}/logs/N1000_seed1.log"
echo "Done when:"
echo "  pgrep -af '${OUTDIR}' | wc -l  -> 0"
echo "  ls ${OUTDIR}/rows_*.csv | wc -l -> $EXPECTED"
echo "Merge after completion:"
echo "python scripts/merge_jc_results.py --outdir ${OUTDIR} --tau_quantile 0.9 --stable_frac_threshold 0.6 --bootstrap"
