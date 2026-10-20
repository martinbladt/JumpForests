# Proportional hazards with a threshold interaction
# Run from manuscript/. Edit the switches below to run selected sections.

# 1. Helpers --------------------------------------------------------------
source("helpers/setup.r")
source("helpers/threshold_interaction.r")

# 2. Settings -------------------------------------------------------------
run_sample_size <- TRUE
run_noise <- TRUE
make_error_plots <- TRUE
make_trajectory_plots <- TRUE
sample_sizes <- c(50L, 100L, 250L, 500L, 1000L, 2000L, 3000L, 4000L, 5000L)
B <- 100L
results_path <- "results/threshold_interaction/Data/sample_size"
plots_path <- "results/threshold_interaction/Plots/sample_size"

# 3. Sample-size experiment -----------------------------------------------
if (run_sample_size) {
    for (n in sample_sizes) {
        data <- load_threshold_interaction_data(n, file.path("results/threshold_interaction/Data", paste0("n", n)))
        # Match the study settings and saved configurations.
        run_threshold_interaction_harmonised_study(
            data, file.path(results_path, paste0("n", n)), B = B,
            num_event_times = if (n >= 2000L) 500L else 0L
        )
    }
}

# 4. Noise experiment and interaction Cox continuation --------------------
if (run_noise) {
    # The same seed/cached sample is used for the n=1000 sample-size study.
    data <- load_threshold_interaction_data(1000L, "results/threshold_interaction/Data/n1000")
    manuscript_run_noise_study(data, "results/threshold_interaction/Data")

    # Replays exactly the same 80/20 splits, noise matrices and evaluation grids.
    # This continuation preserves the JF, Cox and Cox Lasso tables.
    run_threshold_interaction_noise_interactions(data, "results/threshold_interaction/Data", seed = 2026L)
}

# 5. Absolute and relative error plots ------------------------------------
if (make_error_plots) {
    manuscript_prepare_dirs(plots_path)

    for (n in sample_sizes) {
        plot_model_errors_helper(file.path(results_path, paste0("n", n)),
                                 file.path(plots_path, paste0("n", n)))
    }
    plot_model_error_curves_helper(results_path, plots_path)
    for (metric in c("ibs", "ikl", "is")) {
        plot_relative_error_curves_helper(
            input_path = results_path, output_path = plots_path,
            error = metric, harmonised = TRUE
        )
    }
    plot_saved_threshold_interaction_noise_study()
}

# 6. Cumulative-rate and occupation-probability illustrations -------------
if (make_trajectory_plots) {
    manuscript_prepare_dirs("results/threshold_interaction/Plots/sample_size")

    # fix n, fit the best forest and plot relevant combinations of covariates for both the Cox model and the jump forest, 
    # and compare to the true cumulative transition intensities
    data_n1000 <- sim_threshold_interaction_data(1000)
    fitted_forest <- jfforest(MM ~ ., data = data_n1000$jump_data, feature_data = data_n1000$feature_data, 
                              min_node_size = floor(sqrt(nrow(data_n1000$feature_data))) + 1L,
                              splitrule = "logrank", seed = 2026, save_predictions = FALSE)

    test_points <- data.frame(
        X1 = c(0, 0, 1, 1),
        X2 = c(-0.25, 0.25, -0.25, 0.25),
        X3 = rep(0, 4)
    )

    hazard_plot <- plot_threshold_interaction_prediction_comparison(
        fitted_forest = fitted_forest,
        feature_data = data_n1000$feature_data,
        jump_data = data_n1000$jump_data,
        test_points = test_points,
        ncol = 4,
        nrow = 4,
        seed = 2026
    )

    hazard_plot
    ggsave(filename = "results/threshold_interaction/Plots/sample_size/transitionrates_threshold_interaction.pdf",
           plot = hazard_plot, width = 10, height = 8, units = "in")

    occupation_plot <- plot_threshold_interaction_prediction_comparison(
        fitted_forest = fitted_forest,
        feature_data = data_n1000$feature_data,
        jump_data = data_n1000$jump_data,
        test_points = test_points,
        seed = 2026,
        occupation_probabilities = TRUE
    )

    occupation_plot
    ggsave(
        filename = "results/threshold_interaction/Plots/sample_size/occupation_threshold_interaction.pdf",
        plot = occupation_plot, width = 9, height = 5.5, units = "in"
    )
}
