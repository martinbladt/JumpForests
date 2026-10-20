# Run the public study scripts with manuscript/ as the working directory.
# This file loads definitions only; it does not fit models or install packages.
if (!file.exists("SOURCE_MANIFEST.tsv") || !file.exists("helpers/Helpers.r")) {
    stop("Set the working directory to manuscript/ before running these scripts.")
}

manuscript_prepare_dirs <- function(paths) {
    for (path in paths) dir.create(path, recursive = TRUE, showWarnings = FALSE)
    invisible(paths)
}

manuscript_prepare_real_dirs <- function(dataset) {
    root <- file.path("results/biomedical_applications", dataset)
    manuscript_prepare_dirs(c(
        file.path(root, c("Data", "Plots")),
        file.path(root, "Data", paste0("Noise", c(10, 25, 50))),
        file.path(root, "Plots", paste0("Noise", c(10, 25, 50)))
    ))
}

source("helpers/Helpers.r")
source("helpers/proportional_hazards.r")
source("helpers/noise_studies.r")
source("helpers/study_result_reports.r")
source("helpers/plot_noise_ibs_panels.r")

all_split_rules <- c("logrank", "gehan", "taroneware",
                     "approxlogrank", "petoprentice")

# Save the three original noise-study tables before the separate interaction
# continuation. A completed run can be reused; partial runs are not overwritten.
manuscript_run_noise_study <- function(data, path, B = 100L,
                                       max_noise = 50L, seed = 2026L) {
    manuscript_prepare_dirs(path)
    models <- c("jump_forest", "cox", "cox_lasso")
    files <- file.path(path, paste0("ibs_errors_", models, ".txt"))
    config_file <- file.path(path, "noise_study_configuration.rds")
    configuration <- list(data = data, B = B, max_noise = max_noise, seed = seed)
    if (any(file.exists(c(files, config_file)))) {
        if (!all(file.exists(c(files, config_file))) ||
            !identical(readRDS(config_file), configuration)) {
            stop("Incomplete or incompatible noise-study output in ", path,
                 ". Use a fresh directory or inspect the existing files.")
        }
        result <- setNames(lapply(files, utils::read.table,
                                  header = TRUE, check.names = FALSE), models)
        if (!all(vapply(result, function(x) {
            nrow(x) == B &&
                identical(names(x), paste0("noise_", 0:max_noise)) &&
                all(is.finite(as.matrix(x)))
        }, logical(1)))) stop("Invalid saved noise-study tables in ", path)
        message("Keeping completed noise-study tables in ", path)
        return(invisible(result))
    }
    result <- compute_noise_ibs(
        data$feature_data, data$jump_data,
        number_of_noise_variables = max_noise, B = B, seed = seed
    )
    for (model in models) {
        utils::write.table(result[[model]],
                          file.path(path, paste0("ibs_errors_", model, ".txt")),
                          sep = "\t", row.names = FALSE)
    }
    saveRDS(configuration, config_file)
    invisible(result)
}
