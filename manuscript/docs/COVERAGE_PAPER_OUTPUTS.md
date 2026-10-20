# Paper outputs for the averaged leaf variance study

Set `run_study <- FALSE` and `make_reports <- TRUE` in `coverage.r`, then run
`Rscript coverage.r` from `manuscript/`, or source
`helpers/coverage/coverage_paper_outputs.r` and call its reporting functions. Sourcing the
helper only defines functions. It never simulates, tunes, or fits a model.

The results directory is `results/coverage`. The reporting
functions check that `configuration.rds` specifies `scaled_mean_leaf_na` with
sampling without replacement. They reject the historical forest variance study.
The curve reader also checks the method and variance factor in each replicate's
diagnostic table. It uses the saved estimates and interval limits directly.

The variance used in these saved intervals is

\[
\widehat S_F(t\mid x)=\left[\frac mn+\frac{1-m/n}{B}\right]
\frac1B\sum_{b=1}^B Q_b(t\mid x),\qquad
Q_b(t\mid x)=\int_0^t\frac{dN_{jk,b,x}(u)}{Y_{j,b,x}(u)^2}.
\]

For the completed run, `B = 500`, `m/n = 0.7`, and the multiplier of the mean
leaf variance is `0.7006`. The reporting code does not rescale these variances
or recompute intervals using another estimator.

## Transition curve plots

```r
source("helpers/coverage/coverage_paper_outputs.r")
path <- "results/coverage"
plot_coverage_transition_curves(path, band_replicate = 1L)
```

One PDF is saved in `Plots/` for each transition, sample size, and noise setting:
48 PDFs for six sample sizes, four transitions, and two noise settings. Each PDF
has the 16 signal profiles in separate panels. For the noise setting, predictions
condition on `X4 = 0, X5 = 1`; these are individual conditional curves, not
averages across noise profiles.

- All 100 fitted JF curves are opaque grey, with no averaging or discarded runs.
- The true cumulative transition rate is bold black.
- Replicate 1's estimate is highlighted in blue. This index is fixed without
  choosing a favorable coverage outcome.
- That same replicate's pointwise limits are blue dashed lines, and its uniform
  limits are blue dotted lines. The limits are not empirical quantiles of the
  100 estimates, and are not confidence intervals for their Monte Carlo mean.
- Limits are drawn as lines, with no shaded ribbons. Legends identify all curves.
- Panels have borders, no gridlines or plot titles, readable ticks, and separate
  vertical scales so small cumulative rates remain visible.

The target is the **cumulative** transition rate, Lambda, rather than the
instantaneous intensity mu. The confidence intervals in the study concern
Lambda. At the 95% level the stored pointwise half-width is
`1.959964 * sqrt(S_F(t))`; the uniform half-width is
`2.2414027273 * sqrt(S_F(4))`. Lower limits are clipped at zero, as in the study.

By default, the plot reads the much smaller saved reporting-grid files. Estimates
and limits are drawn as right-continuous steps at these display times; jumps
between reporting times are not individually drawn. The exact true curve is
piecewise linear and its switch times are on the grid. For every original jump,
use `grid = "events"`; this reads the substantially larger full result files.
This display choice does not change any coverage calculation.

To make a smaller figure, pass a subset such as
`profiles = data.frame(X1 = c(0, 1), X2 = c(1, 4), X3 = c(0, 1))`.
The argument `n` selects sample sizes. A different `band_replicate` selects the
illustrated bands. The defaults always display all 100 estimates. The manifest
`transition_curve_plot_manifest.tsv` records the settings of the latest call.

## Coverage tables

```r
write_coverage_latex_tables(path, times = c(1.2, 2, 3.3, 4))
```

Twelve LaTeX tables and three companion TSV tables are saved in `Tables/`:

- Eight `coverage_pointwise_<transition>_<arm>.tex` files give coverage at
  `t = 1.2, 2, 3.3, 4`, separately for each transition and noise setting.
- Two `coverage_pointwise_grid_average_<arm>.tex` files give the arithmetic
  average of the pointwise coverage rates over the saved reporting times,
  excluding the deterministic anchor at zero. This is a grid average, not
  whole-curve coverage or a continuous-time integral.
- Two `coverage_uniform_<arm>.tex` files give the probability of covering the
  entire curve on `[0,4]`. They use the saved continuum coverage check, which
  includes left limits at all event times, rather than checking only the display
  grid or averaging the pointwise coverage of the uniform bands.

Every table has sample sizes in columns and includes JF and the fully stratified
Nelson--Aalen benchmark (NA), with Overall, `X1 = 0`, and `X1 = 1` groups. Numbers
are percentages, rounded to one decimal place by default. The TSV companions
retain numerical precision and the available diagnostics/Monte Carlo errors.

For tables, the noise setting averages over **all 96 profiles**, not just the
fixed-noise subset shown in the curve plots. Profiles have equal probability
under this DGP, so the arithmetic profile average is the population group
average. Unavailable intervals count as noncoverage; the denominator is all 100
completed datasets. The code checks full completion and cross-checks the grid
averages against the independently saved per-curve summaries. Pointwise times
must actually occur on the saved grid; no nearest-time substitution is made.

The tables need no additional LaTeX packages. Use, for example,
`\input{coverage_uniform_without_noise.tex}` after copying the file into the
paper directory. Their captions specify the coverage metric and denominator.

## Selected tuning parameters

```r
source("helpers/coverage/coverage_paper_outputs.r")
path <- "results/coverage"
write_coverage_tuning_latex_table(path)
```

This reads the saved `n*/pilot/*_selected.tsv` results and writes
`Tables/coverage_tuning.tex`, with one row per sample size and noise setting.
The columns are sample size, noise setting, selected splitting rule, minimum
node size, and the selected pilot's normalised OOB IBS. IBS is rounded to four
decimal places by default; use `digits` to change this. The companion
`Tables/coverage_tuning_selected.tsv` preserves the numerical precision.

Both feature sets use the same independent pilot dataset at each sample size;
their selected settings remain fixed throughout the evaluation replicates.
The exporter requires one successful saved selection for every configured
sample size and noise setting. It does not tune or fit models. Include the
table with `\input{coverage_tuning.tex}`; no extra LaTeX packages are required.

The `.gitignore` rules retain these PDFs, `.tex` files, and TSV summaries, while
the saved per-replicate inputs remain ignored. None of the implementation files
whose hashes protect study resumption is changed by this reporting addition.
