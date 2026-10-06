# Data format

`results_summary/selected_theta_by_J.csv` contains selected theta0/thetaV/thetaA, validation scores, optimizer seed, relative reader paths and a true-no-recurrence flag. Selection uses validation only.

`outer_test_blocks.csv` contains 186 measured rows: 30 retuned J values and one true-no-recurrence configuration, each with six independent blocks. MSE has squared input units; normalized MSE and R2 are dimensionless. Mean rate and x are per-neuron Hz. Relative raw/reader/prediction identifiers refer to canonical local archives; no NPZ is included here.

`outer_test_summary.csv` contains whole-block bootstrap intervals and means. `equivalence_by_J.csv` compares every recurrent J with the validation-selected J=0.26 using simultaneous intervals and the predefined +/-0.01 normalized-MSE tolerance. The span is sampled evidence, not a continuous theorem.

`stage2_prediction_metrics.csv` preserves first and pre-test augmented forecasts against constant, interpolation and empirical baselines. Group labels distinguish held-out couplings, mixture and combined conditions. MAE and RMSE are normalized-error prediction units.

`stage4_mechanism_predictions.csv` retains original and corrective rounds separately. The unchanged-response ablation is labeled post hoc and limits interpretation. Its errors are summarized in `stage4_mechanism_metrics.json`; an absolute-response improvement alone does not validate intervention-induced changes.

`cleanup_10Hz_gain.csv` and `cleanup_10Hz_contrasts.csv` describe 24 fresh whole blocks, three amplitudes and the final 10 Hz diagnostic. Gain is per-neuron Hz per model-voltage; complex convention is cosine minus i times sine. The smaller paired amplitude difference is consistent with sampling uncertainty; the simultaneous intervals do not certify 5% amplitude equivalence.

Locally generated outputs contain results.csv, reader_candidates.csv, failed_runs.csv and manifest.json. Optional raw NPZ files contain full input, activity, filter, mean voltage and eligible fraction, with one file per split/network/block. Never treat time rows as independent replicates.
