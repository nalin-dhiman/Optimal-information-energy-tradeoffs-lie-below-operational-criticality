# Model and physical clock

For bin n of duration d, voltage V_i,n, refractory age a_i,n and population filter x_n are pre-spike states. An age below tau_ref blocks spiking. Otherwise lambda_i,n = lambda0 exp(theta0 + theta_A x_n) exp(theta_V V_i,n), and an independent uniform draw produces a spike with probability 1-exp(-lambda_i,n d). A spike resets age to zero and optionally resets voltage to v_reset. Other ages increase by d.

With k_n spikes, A_n = k_n/(N d), in spikes/s/neuron. The stored filter is

    x_(n+1) = (1-d/tau_s) x_n + (d/tau_s) A_n.

After reset, all voltages, including refractory ones, update as

    V_i,n+1 = V_tilde_i,n + (d/tau_v)(u_n + J x_(n+1) - V_tilde_i,n)
              + sigma sqrt(2d/tau_v) xi_i,n.

The archival kernel preserves this order and its random-number stream. Initial voltage, age and filter are zero. Refractory eligibility therefore includes startup transients. Diagnostic voltage and log-hazard probes are zero in the baseline; tests compare that case bit-for-bit with the archival kernel.

Default benchmark parameters are N=2000, d=0.5 ms, tau_ref=5 ms, tau_s=20 ms, tau_v=10 ms, lambda0=100 Hz, private voltage-noise sigma=0.5, and reset voltage zero. Voltage and input have model units, not calibrated millivolts. J has model-voltage/per-neuron-Hz units, theta_V inverse model-voltage, theta_A inverse per-neuron-Hz, and theta0 is dimensionless. Activity is not heat or ATP consumption.

The input is the original Euler OU process. Its discrete stationary variance is sigma_u^2/(1-d/(2 tau_c)); measured target variance normalizes error. Four seconds of burn precede benchmark measurements.

The corrected reader causally filters x with 41 FIR taps, an 80 Hz cutoff and 10 ms group delay, then samples physical 5 ms rows. Current measured x plus 20 earlier rows span 100 ms. Target latency is an explicit parameter; the benchmark adds zero latency and never obtains a score clock from an estimated spectral cutoff. The stored x_n array corresponds to the code's post-spike x_(n+1), preceding the effect of u_n on voltage.

Means, scales and intercept are learned on training only; ridge alpha is selected on separate validation. Raw MSE has squared input units. Normalized MSE divides by test target variance and R2 equals one minus normalized MSE. No mutual-information rate is inferred from these scores.

True no recurrence requires J=0 and theta_A=0. Reset/age dynamics prevent treating a two-state no-reset closure as an exact neural theory. The benchmark is a bounded, finite-size reconstruction study, not a proof of a thermodynamic transition.
