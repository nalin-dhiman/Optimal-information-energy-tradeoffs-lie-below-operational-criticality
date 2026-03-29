#!/bin/bash
set -e

BASE_DIR=""
cd $BASE_DIR
export PYTHONPATH=$BASE_DIR:$PYTHONPATH


SCRIPT="python scripts/run_opt_grid.py"

for BETA_C in "0.0" "0.01"; do
    echo ">>> Running beta_C=${BETA_C} for N=2000"
    $SCRIPT --N 2000 --beta_C ${BETA_C} --beta_E_list "0.0" --J_list "0.48,0.49,0.50,0.51,0.52" --seeds "1,2,3" --T_opt 10.0 --dt_opt 0.0005 --outdir "results_betaC/opt_N2000_betaC_${BETA_C}"

    echo ">>> Running beta_C=${BETA_C} for N=5000"
    $SCRIPT --N 5000 --beta_C ${BETA_C} --beta_E_list "0.0" --J_list "0.48,0.49,0.50,0.51,0.52,0.53" --seeds "1,2,3" --T_opt 10.0 --dt_opt 0.0005 --outdir "results_betaC/opt_N5000_betaC_${BETA_C}"

    echo ">>> Running beta_C=${BETA_C} for N=10000"
    $SCRIPT --N 10000 --beta_C ${BETA_C} --beta_E_list "0.0" --J_list "0.46,0.48,0.50,0.52,0.54,0.56" --seeds "1,2,3" --T_opt 10.0 --dt_opt 0.0005 --outdir "results_betaC/opt_N10000_betaC_${BETA_C}"
done

echo "beta_C sweep complete!"
