#!/bin/bash
set -e

BASE_DIR=""
cd $BASE_DIR
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

SCRIPT="python scripts/run_opt_grid.py"

for TAU_C in "0.02" "0.10"; do
    echo ">>> Running tau_c=${TAU_C} for N=5000"
    $SCRIPT --N 5000 --beta_C 0.005 --beta_E_list "0.0" --tau_c_list ${TAU_C} \
            --J_list "0.48,0.49,0.50,0.51,0.52,0.53" --seeds "1,2,3" \
            --T_opt 10.0 --dt_opt 0.0005 \
            --outdir "results_tauC/opt_N5000_tauC_${TAU_C}"
done

echo "tau_c sweep complete!"
