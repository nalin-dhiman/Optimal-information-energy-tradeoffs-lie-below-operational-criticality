# Reproducibility

Run the commands in README from the repository root. Requirements record the tested Python 3.12 environment. Numba fastmath and floating-point refractory ages can make exact bits platform dependent; record CPU, dependency versions and seeds when running elsewhere.

`validate_archival_kernel.py` compares untouched, observation-only and zero-probe kernels. `reproduce_key_results.py` defaults to a small deterministic example. It retains every ridge validation candidate, errors, seed identifiers, score units and configuration hash. `--save-raw` retains locally regenerated arrays; they belong outside Git.

The optional fixed-reference configuration reproduces selected networks and readers with original Stage3 split/seed conventions. It does not repeat the optimizer search or claim certified global optimality. The six independent blocks pair stimuli across network conditions. Training is 40 s, validation 20 s and outer test 40 s, after 4 s burn each. Network parameters are frozen in the config before replay.

`reproduce_figures.py` recomputes a summary table from the 186 immutable outer block rows and plots it with the true no-recurrence baseline. This is a data-based code-validation figure. Table SHA256 values are in `provenance/summary_manifest.json`; simulator SHA256 values are in `provenance/archival_source_hashes.json`.

Complete raw manifests, all optimizer outcomes, older source snapshots and failed prediction locks remain in separate local canonical archives. Compact public tables identify those data by relative path. Without the cold archive, users can regenerate the selected fixed networks but cannot claim to have inspected every historical raw output.

Parameter retuning produced practical equivalence across 21 sampled couplings spanning 0.60 J under a predefined 0.01 normalized-error tolerance. Prospective noise-pathway and compensation tests did not establish a successful general mechanism. No additional large scan is part of the release workflow.
