# Proportional hazards: noise covariates and variable importance
# Run from manuscript/. Edit the switches below to run selected sections.

# 1. Helpers --------------------------------------------------------------
source("helpers/setup.r")

# 2. Settings -------------------------------------------------------------
run_noise <- TRUE
run_vimp <- TRUE
make_reports <- TRUE

# 3. Noise experiment -----------------------------------------------------
if (run_noise) {
    data <- simulate_proportional_hazards(1000L, seed = 2026L)
    manuscript_run_noise_study(data, "results/proportional_hazards/Data")
}

# 4. Variable-importance experiment ---------------------------------------
if (run_vimp) {
    manuscript_prepare_dirs("results/proportional_hazards/Data")
    run_proportional_hazards_vimp(
        n = 1000L, B = 100L, seed = 2026L,
        noise_counts = c(0L, 10L, 25L, 50L), num_permutations = 1L,
        num_event_times = 500L,
        path = "results/proportional_hazards/Data/vimp_rankings_sqrt_logrank.txt"
    )
}

# 5. IBS plots and LaTeX recovery tables ----------------------------------
if (make_reports) {
    manuscript_prepare_dirs(c("results/proportional_hazards/Plots", "results/proportional_hazards/Tables"))
    models <- c("jump_forest", "cox", "cox_lasso")
    errors <- setNames(lapply(models, function(model) {
        read.table(file.path("results/proportional_hazards/Data", paste0("ibs_errors_", model, ".txt")),
                   header = TRUE, check.names = FALSE)
    }), models)
    noise_counts <- seq.int(0L, 50L, by = 5L)
    plot_noise_ibs(errors, path = "results/proportional_hazards/Plots/IBS_noise_plot.pdf",
                       noise_counts = noise_counts)
    plot_noise_ibs_relative_to_JF(
        errors, path = "results/proportional_hazards/Plots/IBS_noise_relative_to_JF.pdf",
        noise_counts = noise_counts
    )
    plot_noise_ibs_panels(errors, path = "results/proportional_hazards/Plots/IBS_noise_panels.pdf",
                             noise_counts = noise_counts)
    write_proportional_hazards_vimp_tables(
        "results/proportional_hazards/Data/vimp_rankings_sqrt_logrank.txt"
    )
}
