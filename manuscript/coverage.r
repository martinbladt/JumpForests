# Coverage study
# Run from manuscript/. Edit the switches below to run selected sections.

# 1. Helpers --------------------------------------------------------------
source("helpers/coverage/coverage_study.r")
source("helpers/coverage/coverage_paper_outputs.r")

# 2. Execution switches ---------------------------------------------------
run_study <- TRUE
make_reports <- TRUE
run_timing_exercise <- FALSE

# 3. Pilot grids and full-study configuration -----------------------------
coverage_min_node_sizes <- list(
    `500` = c(5L, 10L, 20L, 30L, 40L, 50L),
    `1000` = c(10L, 20L, 30L, 40L, 50L, 60L, 70L, 80L, 90L, 100L),
    `2000` = c(20L, 30L, 40L, 50L, 60L, 70L, 80L, 90L, 100L, 125L, 150L),
    '3000' = c(40L, 50L, 60L, 70L, 80L, 90L, 100L, 125L, 150L, 175L, 200L),
    '4000' = c(50L, 60L, 70L, 80L, 90L, 100L, 125L, 150L, 175L, 200L, 225L, 250L),
    '5000' = c(60L, 70L, 80L, 90L, 100L, 125L, 150L, 175L, 200L, 225L, 250L, 300L)
)
coverage_split_rules <- c("logrank", "gehan", "taroneware",
                          "approxlogrank", "petoprentice")
coverage_config <- coverage_study_config(
    min_node_sizes = coverage_min_node_sizes,
    split_rules = coverage_split_rules,
    n = c(500L, 1000L, 2000L, 3000L, 4000L, 5000L), N = 100L,
    ntrees = 500L, sample_rate = 0.7,
    end_time = 4, state2_start = 0,
    nworkers = 12L, seed = 20260930L
)
coverage_output_path <- "results/coverage"

# 4. Pilot tuning and coverage replications -------------------------------
if (run_study) {
    run_coverage_study(coverage_config, coverage_output_path)
}

# 5. Coverage plots, trajectory plots and LaTeX tables --------------------
if (make_reports) {
    # Requires a completed run; no models are fitted.
    plot_coverage_study(coverage_output_path)
    write_coverage_tuning_latex_table(coverage_output_path)
    plot_coverage_transition_curves(
        coverage_output_path, band_replicate = 1L,
        noise_profile = c(X4 = 0L, X5 = 1L), grid = "report"
    )
    write_coverage_latex_tables(
        coverage_output_path, times = c(1.2, 2, 3.3, 4)
    )
}

# 6. Optional separate timing exercise (not the full study) ---------------
if (run_timing_exercise) {
    test_path <- "results/coverage_timing_test"
    test_config <- coverage_study_config(
        min_node_sizes = c(5L, 20L),
        split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice"),
        n = c(500L, 1000L, 2000L), N = 5L, ntrees = 500L,
        sample_rate = 0.7, end_time = 4, state2_start = 0,
        nworkers = 10L, seed = 20260930L
    )
    test_timing <- system.time(run_coverage_study(test_config, test_path))
    coverage_write_table(data.frame(user_seconds = unname(test_timing["user.self"]),
                                   system_seconds = unname(test_timing["sys.self"]),
                                   elapsed_seconds = unname(test_timing["elapsed"])),
                         file.path(test_path, "test_run_timing.tsv"))
    read_arm_tables <- function(suffix) {
        files <- list.files(test_path, paste0("^(with_noise|without_noise)_", suffix, "\\.tsv$"),
                            recursive = TRUE, full.names = TRUE)
        data.table::rbindlist(lapply(files, data.table::fread))
    }
    selected <- read_arm_tables("selected")
    diagnostics <- read_arm_tables("diagnostics")
    pilot_candidates <- read_arm_tables("candidates")
    coverage_write_table(selected, file.path(test_path, "pilot_selected.tsv"))
    coverage_write_table(diagnostics, file.path(test_path, "fit_diagnostics.tsv"))
    timing_by_n_arm <- merge(
        pilot_candidates[, list(pilot_seconds = sum(elapsed_seconds), pilot_candidates = .N), by = c("n", "arm")],
        diagnostics[, list(replicate_seconds = sum(elapsed_seconds),
                           replicate_mean_seconds = mean(elapsed_seconds),
                           replicate_max_seconds = max(elapsed_seconds),
                           mean_censoring_fraction = mean(censoring_fraction)), by = c("n", "arm")],
        by = c("n", "arm"))
    coverage_write_table(timing_by_n_arm, file.path(test_path, "timing_by_n_arm.tsv"))
    overview <- data.table::fread(file.path(test_path, "simultaneous_by_group.tsv"))
    coverage_write_table(overview[overview$group == "overall", ],
                         file.path(test_path, "test_coverage_overview.tsv"))
    plot_coverage_study(test_path)
    message("Coverage test complete: ", normalizePath(test_path))
}
