# Coverage study: implementation and interpretation

## Design and execution

`coverage.r` contains the pilot grids, execution switches, study configuration,
and reporting calls. All reusable functions are in `helpers/coverage/`.
The default design uses n=500,1000,2000,3000,4000,5000, 100 independent evaluation
datasets per n, and an independent pilot per n. The two feature sets tune
separately on the same pilot. X1,X2,X3 contain signal; X4,X5 are noise.

The DGP has initial distribution (1/2,1/2,0), independent of the covariates,
and independent uniform censoring on [5,25]. All transitions are evaluated on
[0,4], where there is no censoring. True cumulative transition rates are
computed exactly from the piecewise-constant intensities. Pilot tuning uses
normalised OOB IBS over full observed follow-up, not only [0,4].

Forests have 500 trees, 70% subsampling without replacement, all covariates
available as split candidates, and 10 candidate thresholds. The pilot grid
crosses the supplied minimum node sizes with all five basic splitting rules.
The best OOB IBS setting is frozen for the evaluation replications. Every
observed event time is retained. The worker count is configurable in `coverage.r`.

## Node and forest variance

For transition j->k and profile x, let L_b(x) denote the prediction leaf in
tree b. The node-level Nelson–Aalen variance estimate is

\[
 Q_b(t\mid x)=\int_0^t \frac{dN_{jk,b,x}(u)}{Y_{j,b,x}(u)^2}.
\]

The code uses the actual observations in each prediction leaf. Under untied
events and subsampling without replacement, Q_b is the sum of the squared
Nelson–Aalen jumps. The simulator is checked for ties, and the native
`JFCppForestLeafVarianceMultistate` interface evaluates these curves.

The variance used for the forest estimate is

\[
 S_F(t\mid x)=\left[\frac mn+\frac{1-m/n}{B}\right]
              \frac1B\sum_{b=1}^B Q_b(t\mid x).
\]

For the default configuration, the multiplier is 0.7006. It is applied to
the average node variance, not to the average of its square roots. Mixed
leaves are retained. The asymptotic justification relies on the manuscript's
purity/regularity conditions; finite-sample pooling bias is evaluated by the
study and is not removed by this variance estimate.

## Stratified Nelson–Aalen benchmark

The benchmark pools observations with exactly the same included covariates:
16 profiles without noise and 96 profiles with noise. It uses ordinary
Nelson–Aalen estimates and raw variance increments dN/Y^2, including the
correct handling of repeated state visits. There is no forest scaling factor.
Empty strata have unavailable estimates. Nonempty zero-event strata have
zero estimated cumulative rate and zero variance. These cases are saved as
diagnostics rather than being silently removed from coverage denominators.

## Pointwise intervals and simultaneous bands

For the estimate A(t) and its raw variance estimate S(t), the nominal 95%
pointwise interval is A(t) +/- 1.959964 sqrt(S(t)). The simultaneous band is

\[
 A(t)\pm c_{0.95}\sqrt{S(4)},\qquad
 c_{0.95}=2.2414027273,
\]

where c is the 95% quantile of the supremum of absolute standard Brownian
motion on [0,1]. This critical value already accounts for both signs. The
terminal variance makes the simultaneous half-width constant in time before
lower-bound clipping. Lower endpoints are clipped at zero for both methods.

Simultaneous coverage is checked at 0, 4, every event time and the left limits
at positive event times. These checks suffice between jumps because the truth
is continuous and nondecreasing and the band bounds are step functions.
Coverage is simultaneous in time for each profile and transition, not jointly
across all profiles or transitions.

## Summaries

Each replication yields coverage indicators for each profile and transition.
Group summaries first average equally over profiles within each replication,
then average over replications. X1 groups contain 8 profiles without noise
and 48 with noise. Monte Carlo standard errors are the standard deviation
of the replication-level group averages divided by the square root of the
number of completed replications.

Bias and squared error are averaged over finite available estimates.
Group RMSE is the square root of the mean replication-level mean squared
error, evaluated separately at each reporting time. Unavailable intervals
count as failures for coverage; unavailable estimates are excluded from
width/bias/RMSE calculations and tracked by estimability diagnostics.

Selected-time tables use 1.2,2,3.3,4. Reporting-grid average coverage is the
arithmetic mean over the positive reporting times; it is not simultaneous
coverage. Curve illustrations show all JF estimates and the bands from
replication 1. With noise, illustrations fix X4=0,X5=1, whereas group tables
average over all relevant profiles.

## Saved outputs and resumption

The full run writes to `results/coverage/`. Configuration and software metadata,
pilot candidates and selections, datasets, per-replication curves, coverage
summaries, figures and LaTeX tables are saved separately. Compatible completed
pilot candidates and replication/noise arms are reused after interruption.
Increasing N extends the study without replacing completed replications.
Incompatible configurations are rejected. `reuse_from` is an optional helper
argument; it is not needed by the public study script.

The optional timing exercise in Section 6 of `coverage.r` uses its own
`results/coverage_timing_test/` directory. Enable only `run_timing_exercise`
to run it separately. See `COVERAGE_PAPER_OUTPUTS.md` for the reporting files
and `COVERAGE_GIT_OUTPUTS.md` for the output ignore policy.

## Checks

The small tests under `tests/` check the DGP, true cumulative rates, recurrent
risk sets, censoring, stratified Nelson–Aalen calculations, Brownian critical
value, intervals, between-jump coverage, independent node/forest variance
calculations, saved outputs, resumption and extending N. They validate the
implementation, not finite-sample nominal coverage.
