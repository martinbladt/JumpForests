# Prediction under non-proportional hazards
# Run from manuscript/. Edit the switches below to run selected sections.

# 1. Helpers --------------------------------------------------------------
source("helpers/setup.r")
source("helpers/non_proportional_hazards.r")

# 2. Settings -------------------------------------------------------------
run_sample_size <- TRUE
make_plots <- TRUE
non_proportional_hazards_sample_sizes <- c(50L, 100L, 250L, 500L, 1000L, 2000L, 3000L, 4000L, 5000L)
B <- 100L

# 3. Sample-size experiment -----------------------------------------------
if (run_sample_size) {
    for (n in non_proportional_hazards_sample_sizes) {
        run_non_proportional_hazards_prediction(n, B = B)
    }
}

# 4. Absolute and relative IBS plots --------------------------------------
if (make_plots) {
    plot_non_proportional_hazards_ibs_by_size(non_proportional_hazards_sample_sizes)
    plot_relative_error_curves_helper(
        input_path = "results/non_proportional_hazards/Data",
        output_path = "results/non_proportional_hazards/Plots",
        error = "ibs", harmonised = TRUE
    )
}
