# Numerical studies

The study names follow the manuscript. Each study has **one R script**, with
numbered sections for settings, computation and reporting. All figures and
tables for a study are handled in that same script.

| Study | R script | Contents |
|---|---|---|
| Proportional hazards with a threshold interaction | [threshold_interaction.r](threshold_interaction.r) | Sample-size experiment, noise experiment, interaction Cox continuation, error plots and trajectory illustrations. |
| Prediction under non-proportional hazards | [non_proportional_hazards.r](non_proportional_hazards.r) | Sample-size experiment and absolute/relative IBS plots. |
| Proportional hazards: noise covariates and variable importance | [proportional_hazards.r](proportional_hazards.r) | Noise experiment, permutation VIMP experiment, IBS plots and LaTeX recovery tables. |
| Coverage study | [coverage.r](coverage.r) | Pilot tuning, JF and stratified Nelson–Aalen intervals, coverage summaries, plots, LaTeX tables and an optional timing exercise. |
| Biomedical applications | [biomedical_applications.r](biomedical_applications.r) | Separate sections for cphholter, bmt, ebmt3 and prova, followed by combined error plots. |

There are four simulation studies and one biomedical applications script.
The non-proportional-hazards prediction experiment is included alongside the
other studies. Reusable functions are in `helpers/`, including the shared
non-proportional-hazards simulator and the coverage inference functions in
`helpers/coverage/`. Existing small verification scripts are in `tests/`.

## Running the scripts

Use `manuscript/` as the working directory:

```sh
cd manuscript
Rscript check_environment.r
Rscript threshold_interaction.r
Rscript non_proportional_hazards.r
Rscript proportional_hazards.r
Rscript coverage.r
Rscript biomedical_applications.r
```

Each study can also be run with `source("<study>.r")` from an R session, or
by selecting its sections interactively. **Running an entire study script
starts its enabled computations**, which can take a long time. The switches
near the top select the sections to execute:

| Script | Computation switches | Reporting switches |
|---|---|---|
| `threshold_interaction.r` | `run_sample_size`, `run_noise` | `make_error_plots`, `make_trajectory_plots` |
| `non_proportional_hazards.r` | `run_sample_size` | `make_plots` |
| `proportional_hazards.r` | `run_noise`, `run_vimp` | `make_reports` |
| `coverage.r` | `run_study`, `run_timing_exercise` | `make_reports` |
| `biomedical_applications.r` | `run_analyses`, `run_tuning`, `datasets_to_run` | `make_comparison_plots` |

For reporting from completed results, set the computation switches to `FALSE`.
The threshold-interaction trajectory illustrations fit an illustrative forest
and comparison models, so also disable `make_trajectory_plots` if no model
fitting is wanted. The biomedical dataset sections retain their own model fits
and plots together; the final comparison section only reads score tables.
Exploratory biomedical tuning and the coverage timing exercise are disabled
by default.

Loading helper files only defines functions and loads packages. It does not
run an experiment or install packages.

## Packages

Install the standard dependencies separately:

```r
install.packages(c(
    "survival", "tictoc", "ggplot2", "dplyr", "glmnet", "data.table",
    "mstate", "tidyverse", "etm", "Matrix", "scales", "gtable"
))
```

Also install the authors' **JumpForests** and **AalenJohansen** packages.
These are dependencies, not bundled here. If their source directories are
available locally, use `R CMD INSTALL /path/to/package`.

`DEPENDENCIES.tsv` records the versions used to validate the initial copy
(R 4.6.1). The local JumpForests source was at commit
`471b41ec461b290fc5c98fc76386dc8056f0fb12`, with local changes to
`src/ForestSurvival.cpp` and `tests/spherical-vimp.R`. That commit alone is
therefore not an exact source lock for the installed package. Preserve the
package source and `sessionInfo()` when reproducing results.

Coverage requires the native `JFCppForestLeafVarianceMultistate` interface;
`check_environment.r` checks its availability. Transition-wise RSF uses the
survival-forest implementation in JumpForests.

## Study settings and outputs

### Proportional hazards with a threshold interaction

`threshold_interaction.r` starts all subjects in state 1 and has no censoring. Sample sizes
are 50, 100, 250, 500, 1000, 2000, 3000, 4000 and 5000, with 100 shared 80/20
splits at each size. The noise experiment uses n=1000 and noise counts 0–50.
The interaction Cox continuation reuses the original splits, noise matrices
and evaluation grids. Outputs are in `results/threshold_interaction/`.

JF and transition-wise RSF use logrank and the strict square-root node-size
heuristic. The sample-size experiment keeps all event times below n=2000
and uses 500 event times from n=2000 onward, matching the final saved study
configurations. Completed sample-size runs and interaction continuations can
be reused. The base noise experiment saves tables at completion rather than
checkpointing individual splits.

### Prediction under non-proportional hazards

`non_proportional_hazards.r` uses the piecewise intensity model shared with
coverage, but starts all subjects in state 1. Censoring is uniform on [5,25].
It compares JF and the additive and interaction Cox models over the same
sample-size range, using 100 shared splits and 250 event times. Outputs and
dataset caches are in `results/non_proportional_hazards/`.

### Proportional hazards: noise covariates and variable importance

`proportional_hazards.r` uses time-dependent baseline intensities with
log-linear covariate effects, initial distribution `(0.4,0.6,0)` and fixed
administrative censoring at 8. The noise experiment uses one n=1000 dataset
and 100 splits; the VIMP experiment generates 100 independent n=1000 datasets.
Held-out JF, Cox and Cox Lasso share each covariate permutation and condition
on observed starting states. The separate full-data OOB JF comparator uses
its internal permutation procedure and estimated initial distribution.
The VIMP experiment checkpoints completed run/noise combinations.
Outputs are in `results/proportional_hazards/`.

### Coverage study

All pilot grids, worker settings, execution and reporting calls are in
`coverage.r`. The copied configuration uses n=500, 1000, 2000, 3000, 4000 and
5000, N=100, 500 trees, 70% subsampling without replacement, and 12 workers.
Both noise arms tune separately on one shared pilot per n. Pilot OOB IBS uses
full follow-up and all event times. Coverage is evaluated on [0,4], with
initial distribution `(0.5,0.5,0)`.

The forest variance is the average prediction-leaf Nelson–Aalen variance
multiplied by `m/n + (1-m/n)/B`. Both pointwise and simultaneous bands are
implemented. The selected-time tables use 1.2, 2, 3.3 and 4. Results are saved
in `results/coverage/`; compatible completed work is reused on rerun. The
optional timing exercise uses a separate `results/coverage_timing_test/`.

Detailed descriptions are in [docs/COVERAGE_PAPER_OUTPUTS.md](docs/COVERAGE_PAPER_OUTPUTS.md)
and [docs/COVERAGE_STUDY_REPORT.md](docs/COVERAGE_STUDY_REPORT.md). These describe the configuration in `coverage.r` and the
`scaled_mean_leaf_na` variance estimator.

### Biomedical applications

All four datasets have sections in `biomedical_applications.r`. Select a
subset with `datasets_to_run`. The combined error plots require results for
all four datasets, so disable `make_comparison_plots` if those are unavailable.

All four CSV inputs are bundled in `data/biomedical_applications/`.
See [docs/BIOMEDICAL_DATA.md](docs/BIOMEDICAL_DATA.md) for their provenance and
column order. Outputs go in `results/biomedical_applications/<dataset>/`.
The dataset sections can overwrite their own output tables when rerun.

| Dataset | Subjects | Original covariates | Error-study minimum node size |
|---|---:|---:|---:|
| cphholter | 678 | 10 | 24 |
| bmt | 2009 | 4, excluding `team` | 41 |
| ebmt3 | 2204 | 4 | 42 |
| prova | 286 | 7 | 16 |

## Validation and provenance

The original project scripts and results remain unchanged. The reorganisation
preserves the copied statistical settings; it does not silently change the
DGPs or rerun results to reconcile differences in the paper. Specific
paper/code differences found while checking the study names are recorded in
[docs/PAPER_CODE_NOTES.md](docs/PAPER_CODE_NOTES.md).

`SOURCE_MANIFEST.tsv` identifies the source files, extracted line ranges
and source hashes (or a source description for renamed shared code). Study
helper functions use the same names as the study scripts.
The four biomedical datasets are bundled; simulated data, fitted forests and
results are generated on execution. `.gitignore` retains the four input CSVs,
paper tables, coverage summaries and tuning tables while excluding runtime objects.

To smoke-test all five entry points from fresh R sessions:

```sh
Rscript tests/smoke_manuscript.r
```

This checks the enabled sections with small fits, scoring, saved-table loading,
plotting and LaTeX export
in a temporary copy. It uses five-tree forests, two splits/replications,
smaller simulated samples and a shorter biomedical event grid. It retains
all four biomedical datasets. The production scripts, settings and results
are untouched. Successful temporary runs are deleted; failed runs are retained
in the system temporary directory for diagnosis. This checks execution, not full-study statistical
results or runtime.

The coverage checks can also be run independently:

```sh
Rscript tests/test_coverage_dgp.r
Rscript tests/test_coverage_inference.r
Rscript tests/test_coverage_study.r
```
