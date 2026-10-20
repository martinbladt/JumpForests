# Coverage of cumulative transition rates for the discrete non-proportional-hazards DGP.
# This file can be sourced independently; it never launches a study on source.
# All manuscript entry points and helpers use manuscript/ as the working directory.
coverage_source_directory <- normalizePath("helpers/coverage", mustWork = TRUE)
source(file.path(coverage_source_directory, "coverage_dgp.r"))
source(file.path(coverage_source_directory, "coverage_inference.r"))

# Both transient states have positive initial probability. The default interval
# is [0, end_time] for every transition; state2_start is only an optional
# subinterval override, retained for exploratory analyses of saved full curves.
coverage_study_config <- function(
    min_node_sizes, split_rules = c("logrank", "gehan", "taroneware",
                                    "approxlogrank", "petoprentice"),
    n = c(500L, 1000L, 2000L), N = 100L, ntrees = 500L,
    sample_rate = 0.7, end_time = 4, state2_start = 0,
    level = 0.95, report_times = seq(0, end_time, length.out = 251L),
    seed = 20260930L, nworkers = 10L, clip_lower = TRUE,
    parameters = non_proportional_hazards_parameters) {
    integer_values <- c(n, N, ntrees, seed, nworkers)
    if (any(!is.finite(integer_values)) || any(integer_values != floor(integer_values)) ||
        any(n < 2L) || N < 1L || ntrees < 2L || seed < 1L || nworkers < 1L) {
        stop("n, N, ntrees, seed and nworkers must be positive integers; ntrees >= 2")
    }
    if (anyDuplicated(n) || !(sample_rate > 0 && sample_rate < 1) ||
        !(state2_start >= 0 && state2_start < end_time) || !(level > 0 && level < 1)) {
        stop("Require distinct n, 0 < sample_rate < 1, 0 <= state2_start < end_time and 0 < level < 1")
    }
    if (!is.list(min_node_sizes)) {
        min_node_sizes <- setNames(rep(list(min_node_sizes), length(n)), as.character(n))
    }
    if (!all(as.character(n) %in% names(min_node_sizes))) {
        stop("Supply a minimum-node-size grid for every n")
    }
    min_node_sizes <- lapply(min_node_sizes[as.character(n)], function(z) {
        if (!length(z) || any(!is.finite(z)) || any(z < 1 | z != floor(z))) {
            stop("Minimum node sizes must be positive integers")
        }
        as.integer(sort(unique(z)))
    })
    basic_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    if (!length(split_rules) || any(!split_rules %in% basic_rules) || anyDuplicated(split_rules)) {
        stop("split_rules must be distinct members of the five basic rules")
    }
    if (any(!is.finite(report_times)) || any(report_times < 0 | report_times > end_time)) {
        stop("Reporting times must belong to [0, end_time]")
    }
    parameters <- utils::modifyList(non_proportional_hazards_parameters, parameters)
    if (end_time >= parameters$censor_upper) stop("end_time must precede the censoring endpoint")
    switches <- unlist(parameters[c("s1", "s2", "s3", "s4")], use.names = FALSE)
    report_times <- sort(unique(c(0, report_times, state2_start, end_time,
                                  switches[switches >= 0 & switches <= end_time])))
    list(schema_version = 3L, dgp = "coverage_initial_mixture",
         variance_estimator = "scaled_mean_leaf_na",
         initial_distribution = coverage_initial_distribution,
         n = as.integer(n), N = as.integer(N),
         min_node_sizes = min_node_sizes, split_rules = split_rules,
         ntrees = as.integer(ntrees), sample_rate = sample_rate,
         end_time = end_time, state2_start = state2_start, level = level,
         report_times = report_times, seed = as.integer(seed),
         nworkers = as.integer(nworkers), clip_lower = clip_lower,
         parameters = parameters, swr = FALSE, honest = FALSE,
         mtry = "all", nsplits = 10L, num_event_times = 0L)
}

coverage_save_rds <- function(object, path) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    staged <- tempfile(".coverage-", tmpdir = dirname(path))
    on.exit(unlink(staged))
    saveRDS(object, staged, compress = "gzip")
    if (!file.rename(staged, path)) stop("Could not save checkpoint: ", path)
    invisible(path)
}

coverage_write_table <- function(object, path) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    staged <- tempfile(".coverage-table-", tmpdir = dirname(path))
    on.exit(unlink(staged))
    data.table::fwrite(object, staged, sep = "\t", na = "NA")
    if (!file.rename(staged, path)) stop("Could not save table: ", path)
    invisible(path)
}

coverage_seed <- function(config, n, replicate = 0L, stage = 0L) {
    as.integer((as.double(config$seed) + 100000 * n + 100 * replicate + stage) %%
                   2147483646 + 1)
}

coverage_simulate <- function(n, seed, config) {
    do.call(simulate_coverage_data,
            c(list(n = n, seed = seed,
                   initial_distribution = config$initial_distribution), config$parameters))
}

coverage_check_no_ties <- function(data) {
    transitions <- unlist(lapply(data$jump_data, function(z) {
        z$times[-1L][diff(z$states) != 0]
    }), use.names = FALSE)
    if (anyDuplicated(transitions)) {
        stop("Tied original event times: recovering leaf NA variance from its jumps requires untied events")
    }
    invisible(TRUE)
}

coverage_fit <- function(data, feature_names, min_node_size, split_rule, config, seed) {
    JumpForests::jfforest(
        MM ~ ., data = data$jump_data, feature_data = data$feature_data[feature_names],
        splitrule = split_rule, min_node_size = min_node_size,
        mtry = length(feature_names), nsplits = config$nsplits,
        ntrees = config$ntrees, swr = FALSE, sample_rate = config$sample_rate,
        honest = FALSE, double_bootstrap = FALSE, num_event_times = 0L,
        # Compute OOB scores only for pilots; training predictions are unnecessary
        # for the coverage fits and consume substantial memory on the full grid.
        save_predictions = FALSE, seed = seed, nworkers = config$nworkers
    )
}

coverage_tune_pilot <- function(data, n, arm, config, path) {
    features <- if (arm == "with_noise") paste0("X", 1:5) else paste0("X", 1:3)
    ledger_path <- file.path(path, "pilot", paste0(arm, "_candidates.rds"))
    ledger <- if (file.exists(ledger_path)) readRDS(ledger_path) else data.frame()
    candidates <- expand.grid(min_node_size = config$min_node_sizes[[as.character(n)]],
                              split_rule = config$split_rules, stringsAsFactors = FALSE)
    for (candidate in seq_len(nrow(candidates))) {
        if (nrow(ledger) && candidate %in% ledger$candidate) next
        choice <- candidates[candidate, ]
        message("Pilot n=", n, " ", arm, " candidate ", candidate, "/", nrow(candidates),
                ": ", choice$split_rule, ", min_node_size=", choice$min_node_size)
        warnings <- character()
        start <- proc.time()[["elapsed"]]
        result <- tryCatch(withCallingHandlers({
            fit <- coverage_fit(data, features, choice$min_node_size, choice$split_rule,
                                config, coverage_seed(config, n, stage = 2L))
            error <- JumpForests::jfforest.error(fit)
            score <- error$normalised.IBS.error
            if (length(score) != 1L || !is.finite(score)) stop("Non-finite OOB IBS")
            list(score = score, nodes = fit$avg.num.terminal.nodes, status = "ok", error = "")
        }, warning = function(w) {
            warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
        }), error = function(e) list(score = NA_real_, nodes = NA_real_,
                                     status = "failed", error = conditionMessage(e)))
        row <- data.frame(n = n, arm = arm, candidate = candidate,
                          min_node_size = choice$min_node_size, split_rule = choice$split_rule,
                          oob_ibs = result$score, average_terminal_nodes = result$nodes,
                          elapsed_seconds = proc.time()[["elapsed"]] - start,
                          status = result$status, error = result$error,
                          warnings = paste(unique(warnings), collapse = " | "),
                          seed = coverage_seed(config, n, stage = 2L))
        ledger <- rbind(ledger, row)
        coverage_save_rds(ledger, ledger_path)
        coverage_write_table(ledger, sub("rds$", "tsv", ledger_path))
        rm(result)
        if (exists("fit", inherits = FALSE)) rm(fit)
        gc(verbose = FALSE)
    }
    # Also export a reused pilot ledger when no candidate needed fitting.
    coverage_write_table(ledger, sub("rds$", "tsv", ledger_path))
    good <- ledger[ledger$status == "ok" & is.finite(ledger$oob_ibs), , drop = FALSE]
    if (!nrow(good)) stop("All pilot candidates failed for n=", n, ", ", arm)
    # Deterministic tie breaking: node size first, then supplied splitting-rule order.
    best <- good[order(good$oob_ibs, good$min_node_size,
                       match(good$split_rule, config$split_rules)), ][1L, , drop = FALSE]
    coverage_write_table(best, file.path(path, "pilot", paste0(arm, "_selected.tsv")))
    best
}

coverage_forest_curves <- function(fit, profiles, feature_names, times) {
    native <- get0("JFCppForestLeafVarianceMultistate", asNamespace("JumpForests"),
                   inherits = FALSE)
    if (is.null(native)) {
        stop("JumpForests needs the coverage leaf-variance interface; restart R after installing it")
    }
    full_times <- fit$unique.event.times
    if (!length(full_times) || full_times[1L] != 0 || is.unsorted(full_times, strictly = TRUE)) {
        stop("Unexpected forest event grid")
    }
    index <- findInterval(times, full_times)
    if (any(index < 1L)) stop("Requested times precede the forest event grid")
    entries <- cbind(c(1L, 1L, 2L, 2L), c(2L, 3L, 1L, 3L))
    processed <- get("preprocess_data", asNamespace("JumpForests"))(
        profiles[feature_names], fit$categorical.levels)
    leaf <- native(fit, processed$data, seq_along(feature_names) - 1L,
                   processed$categorical, processed$unique_values, index, entries)
    labels <- list(time = as.character(times), transition = coverage_transition_names,
                   profile = profiles$profile_id)
    dimnames(leaf$A) <- dimnames(leaf$mean_leaf_Q) <- labels
    dimnames(leaf$leaf_ids) <- dimnames(leaf$leaf_sizes) <-
        list(tree = seq_len(fit$num.trees), profile = profiles$profile_id)
    factor <- coverage_leaf_variance_factor(fit$num.obs, fit$subsample.size, fit$num.trees)
    # These support-region diagnostics do not alter a leaf or filter a result.
    # They count covariate combinations routed to the same leaf, including
    # combinations absent from the realized training subsample.
    signal_keys <- .coverage_profile_keys(profiles, paste0("X", 1:3))
    mixed_signal <- matrix(FALSE, nrow(leaf$leaf_ids), ncol(leaf$leaf_ids))
    mixed_profile <- mixed_signal
    for (b in seq_len(nrow(leaf$leaf_ids))) {
        for (ids in split(seq_len(nrow(profiles)), leaf$leaf_ids[b, ])) {
            mixed_signal[b, ids] <- length(unique(signal_keys[ids])) > 1L
            mixed_profile[b, ids] <- length(ids) > 1L
        }
    }
    diagnostics <- data.frame(
        profile_id = profiles$profile_id,
        mean_leaf_size = colMeans(leaf$leaf_sizes),
        min_leaf_size = apply(leaf$leaf_sizes, 2L, min),
        max_leaf_size = apply(leaf$leaf_sizes, 2L, max),
        mixed_signal_leaf_fraction = colMeans(mixed_signal),
        mixed_profile_leaf_fraction = colMeans(mixed_profile))
    list(A = leaf$A, Q = factor * leaf$mean_leaf_Q,
         mean_leaf_Q = leaf$mean_leaf_Q, variance_factor = factor,
         variance_estimator = "scaled_mean_leaf_na", leaf_diagnostics = diagnostics,
         leaf_ids = leaf$leaf_ids, leaf_sizes = leaf$leaf_sizes)
}

coverage_result_tables <- function(curves, benchmark, data, profiles, times,
                                   n, replicate, arm, model, config) {
    point_tables <- vector("list", nrow(profiles) * 4L)
    curve_tables <- vector("list", length(point_tables))
    transition_names <- c("1->2", "1->3", "2->1", "2->3")
    critical <- coverage_brownian_sup_quantile(config$level)
    count <- 0L
    for (p in seq_len(nrow(profiles))) {
        truth <- coverage_true_cumulative(times, profiles[p, ], config$parameters)
        for (tr in seq_len(4L)) {
            count <- count + 1L
            origin <- if (tr <= 2L) 1L else 2L
            start <- if (origin == 1L) 0 else config$state2_start
            table <- coverage_interval_curve(
                times, curves$A[, tr, p], curves$Q[, tr, p], truth[, tr],
                start = start, end = config$end_time, level = config$level,
                sup_critical = critical, clip_lower = config$clip_lower)
            keep <- match(table$time, times)
            start_index <- match(start, times)
            table$n <- n
            table$replicate <- replicate
            table$arm <- arm
            table$model <- model
            table$profile_id <- profiles$profile_id[p]
            for (feature in paste0("X", 1:5)) {
                table[[feature]] <- if (feature %in% names(profiles)) profiles[[feature]][p] else NA_integer_
            }
            table$transition <- transition_names[tr]
            table$origin <- origin
            table$variance_estimator <- if (model == "JF") config$variance_estimator else "stratified_na"
            table$interval_start <- start
            table$interval_end <- config$end_time
            table$n_profile <- benchmark$n_profile[p]
            table$stratum_risk <- benchmark$risk[keep, origin, p]
            table$stratum_events <- benchmark$events[keep, tr, p] - benchmark$events[start_index, tr, p]
            table$empty_stratum <- table$n_profile == 0L
            table$zero_stratum_risk <- table$stratum_risk == 0L
            table$report_time <- table$time %in% config$report_times
            table$anchor <- table$time == start
            point_tables[[count]] <- table
            valid <- all(is.finite(table$estimate) & is.finite(table$variance))
            # Include left limits at every jump: checking only right-continuous
            # event-time values can miss failures immediately before a jump.
            simultaneous <- coverage_continuum_covered(
                table$time, table$estimate, table$true,
                table$simultaneous_lower, table$simultaneous_upper)
            pointwise_entire_curve <- coverage_continuum_covered(
                table$time, table$estimate, table$true,
                table$pointwise_lower, table$pointwise_upper)
            regular <- table$report_time & !table$anchor
            curve_tables[[count]] <- data.frame(
                n = n, replicate = replicate, arm = arm, model = model,
                variance_estimator = if (model == "JF") config$variance_estimator else "stratified_na",
                profile_id = profiles$profile_id[p], X1 = profiles$X1[p],
                transition = transition_names[tr], interval_start = start,
                interval_end = config$end_time, n_profile = benchmark$n_profile[p],
                estimable = valid, empty_stratum = benchmark$n_profile[p] == 0L,
                min_stratum_risk = min(table$stratum_risk),
                terminal_stratum_events = tail(table$stratum_events, 1L),
                terminal_zero_variance = isTRUE(tail(table$variance, 1L) == 0),
                simultaneous_covered = simultaneous,
                pointwise_entire_curve_covered = pointwise_entire_curve,
                mean_pointwise_covered = if (valid) mean(table$pointwise_covered[regular]) else NA_real_,
                mean_pointwise_width = if (valid) mean(table$pointwise_width[regular]) else NA_real_,
                mean_simultaneous_width = if (valid) mean(table$simultaneous_width[regular]) else NA_real_,
                terminal_bias = tail(table$bias, 1L),
                terminal_estimate = tail(table$estimate, 1L),
                terminal_variance = tail(table$variance, 1L))
        }
    }
    list(event_time_results = data.table::rbindlist(point_tables),
         curve_results = data.table::rbindlist(curve_tables))
}

run_coverage_study <- function(config, path, resume = TRUE, reuse_from = NULL) {
    if (!requireNamespace("JumpForests", quietly = TRUE) ||
        !requireNamespace("data.table", quietly = TRUE)) {
        stop("Install JumpForests and data.table before running the coverage study")
    }
    if (!exists("JFCppForestLeafVarianceMultistate", asNamespace("JumpForests"), inherits = FALSE)) {
        stop("Install the JumpForests coverage leaf-variance interface and restart R before running")
    }
    # Reuse only data and pilot ledgers from the same statistical design.
    # Old fitted averages do not contain the individual-tree variance curves.
    if (!is.null(reuse_from)) {
        previous <- readRDS(file.path(reuse_from, "configuration.rds"))
        if (!all(config$n %in% previous$config$n)) {
            stop("Reuse source must contain the requested sample sizes")
        }
        comparable_reuse <- function(z) {
            # Thread count changes execution only; trees have fixed per-tree
            # seeds. Allow a subset of completed sizes for validation runs.
            z[c("schema_version", "variance_estimator", "N", "nworkers")] <- NULL
            z$n <- config$n
            z$min_node_sizes <- z$min_node_sizes[as.character(config$n)]
            z[sort(names(z))]
        }
        if (!identical(comparable_reuse(previous$config), comparable_reuse(config))) {
            stop("Reuse source has a different DGP or fitting configuration")
        }
        if (normalizePath(path, mustWork = FALSE) == normalizePath(reuse_from)) {
            stop("Reuse source and new result directory must differ")
        }
    }
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
    config_path <- file.path(path, "configuration.rds")
    implementation <- list(
        package_version = as.character(utils::packageVersion("JumpForests")),
        package_dll_md5 = tools::md5sum(getLoadedDLLs()[["JumpForests"]][["path"]]),
        R_version = R.version.string,
        source_md5 = tools::md5sum(file.path(coverage_source_directory,
                                            c("coverage_study.r", "coverage_dgp.r", "coverage_inference.r"))))
    if (file.exists(config_path)) {
        if (!resume) stop("Output directory already contains a study; choose a new path")
        old <- readRDS(config_path)
        comparable <- function(z) { z$N <- NULL; z }
        if (!identical(comparable(old$config), comparable(config)) ||
            !identical(old$implementation, implementation)) {
            stop("Configuration or implementation differs from the saved study; choose a new path")
        }
        if (config$N < old$config$N) stop("N may be increased when resuming, but cannot be decreased")
    } else if (length(list.files(path, all.files = FALSE))) {
        stop("Refusing to initialise in a nonempty directory without a coverage configuration")
    }
    coverage_save_rds(list(config = config, implementation = implementation), config_path)
    if (!is.null(reuse_from)) {
        coverage_save_rds(list(path = normalizePath(reuse_from),
                              configuration = previous, reused = "datasets and pilot ledgers only"),
                          file.path(path, "reuse_source.rds"))
    }
    writeLines(capture.output(sessionInfo()), file.path(path, "session_info.txt"))
    arms <- c("without_noise", "with_noise")
    for (n in config$n) {
        n_path <- file.path(path, paste0("n", n))
        pilot_path <- file.path(n_path, "pilot", "dataset.rds")
        if (!is.null(reuse_from)) {
            for (name in c("dataset.rds", paste0(arms, "_candidates.rds"))) {
                target <- file.path(n_path, "pilot", name)
                source <- file.path(reuse_from, paste0("n", n), "pilot", name)
                if (!file.exists(target) && file.exists(source)) {
                    coverage_save_rds(readRDS(source), target)
                }
            }
        }
        if (!file.exists(pilot_path)) {
            pilot <- coverage_simulate(n, coverage_seed(config, n, stage = 1L), config)
            coverage_check_no_ties(pilot)
            coverage_save_rds(pilot, pilot_path)
        } else pilot <- readRDS(pilot_path)
        choices <- setNames(lapply(arms, function(arm) {
            coverage_tune_pilot(pilot, n, arm, config, n_path)
        }), arms)
        rm(pilot)
        for (replicate in seq_len(config$N)) {
            rep_path <- file.path(n_path, sprintf("replicate_%03d", replicate))
            data_path <- file.path(rep_path, "dataset.rds")
            if (!is.null(reuse_from) && !file.exists(data_path)) {
                source <- file.path(reuse_from, paste0("n", n),
                                    sprintf("replicate_%03d", replicate), "dataset.rds")
                if (file.exists(source)) coverage_save_rds(readRDS(source), data_path)
            }
            completed <- vapply(arms, function(arm) file.exists(file.path(rep_path, paste0(arm, ".rds"))), logical(1))
            if (all(completed)) next
            if (file.exists(data_path)) data <- readRDS(data_path) else {
                data <- coverage_simulate(n, coverage_seed(config, n, replicate, 3L), config)
                coverage_check_no_ties(data)
                coverage_save_rds(data, data_path)
            }
            events <- coverage_event_times(data)
            times <- sort(unique(c(config$report_times, events[events <= config$end_time])))
            for (arm in arms[!completed]) {
                message("Coverage n=", n, " replicate ", replicate, "/", config$N, " ", arm)
                start_clock <- proc.time()[["elapsed"]]
                features <- if (arm == "with_noise") paste0("X", 1:5) else paste0("X", 1:3)
                profiles <- coverage_profiles(arm == "with_noise")
                chosen <- choices[[arm]]
                warnings <- character()
                result <- tryCatch(withCallingHandlers({
                    benchmark <- coverage_stratified_na(data, profiles, features, times)
                    fit_start <- proc.time()[["elapsed"]]
                    fit <- coverage_fit(data, features, chosen$min_node_size, chosen$split_rule,
                                        config, coverage_seed(config, n, replicate, 4L))
                    fit_seconds <- proc.time()[["elapsed"]] - fit_start
                    actual_grid <- fit$unique.event.times
                    if (!isTRUE(all.equal(as.numeric(events), as.numeric(actual_grid), tolerance = 0))) {
                        stop("Forest changed the original event grid")
                    }
                    forest <- coverage_forest_curves(fit, profiles, features, times)
                    fit_diagnostics <- list(
                        ntrees = fit$num.trees, subsample_size = fit$subsample.size,
                        mtry = fit$mtry, average_terminal_nodes = fit$avg.num.terminal.nodes,
                        average_depth = fit$avg.tree.depth, num_event_times = length(actual_grid),
                        fit_seconds = fit_seconds, selected_min_node_size = chosen$min_node_size,
                        selected_split_rule = chosen$split_rule,
                        variance_estimator = config$variance_estimator,
                        leaf_variance_factor = forest$variance_factor,
                        initial_state_1_count = sum(vapply(data$jump_data,
                            function(z) z$states[1L] == 1L, logical(1))),
                        initial_state_2_count = sum(vapply(data$jump_data,
                            function(z) z$states[1L] == 2L, logical(1))))
                    rm(fit)
                    gc(verbose = FALSE)
                    forest_tables <- coverage_result_tables(forest, benchmark, data, profiles, times,
                                                            n, replicate, arm, "JF", config)
                    na_tables <- coverage_result_tables(benchmark, benchmark, data, profiles, times,
                                                        n, replicate, arm, "NA", config)
                    list(
                        schema_version = config$schema_version, n = n, replicate = replicate, arm = arm,
                        data_seed = coverage_seed(config, n, replicate, 3L),
                        forest_seed = coverage_seed(config, n, replicate, 4L), profiles = profiles,
                        times = times,
                        # Retain full cumulative curves from zero so any
                        # subinterval inside [0,T] can be
                        # investigated later without refitting either model.
                        cumulative_curves = list(JF = forest, "NA" = list(A = benchmark$A, Q = benchmark$Q)),
                        stratum_diagnostics = benchmark[c("n_profile", "risk", "events")],
                        diagnostics = fit_diagnostics,
                        censoring_fraction = mean(vapply(data$jump_data, function(z) tail(z$states, 1L) != 3L, logical(1))),
                        event_time_results = data.table::rbindlist(list(forest_tables$event_time_results, na_tables$event_time_results)),
                        curve_results = data.table::rbindlist(list(forest_tables$curve_results, na_tables$curve_results)))
                }, warning = function(w) {
                    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
                }), error = function(e) {
                    coverage_save_rds(list(n = n, replicate = replicate, arm = arm,
                                           message = conditionMessage(e), time = Sys.time()),
                                      file.path(rep_path, paste0(arm, "_failure.rds")))
                    stop("Coverage stopped at n=", n, ", replicate=", replicate, ", ", arm,
                         ": ", conditionMessage(e), ". Completed fits are checkpointed.", call. = FALSE)
                })
                result$warnings <- unique(warnings)
                result$elapsed_seconds <- proc.time()[["elapsed"]] - start_clock
                coverage_save_rds(result$event_time_results[result$event_time_results$report_time, ],
                                  file.path(rep_path, paste0(arm, "_report_grid.rds")))
                coverage_write_table(result$curve_results, file.path(rep_path, paste0(arm, "_curves.tsv")))
                coverage_write_table(cbind(
                    data.frame(n = n, replicate = replicate, arm = arm),
                    result$cumulative_curves$JF$leaf_diagnostics),
                    file.path(rep_path, paste0(arm, "_leaf_diagnostics.tsv")))
                coverage_write_table(as.data.frame(c(
                    list(n = n, replicate = replicate, arm = arm,
                         elapsed_seconds = result$elapsed_seconds,
                         censoring_fraction = result$censoring_fraction,
                         warnings = paste(result$warnings, collapse = " | ")),
                    result$diagnostics)),
                    file.path(rep_path, paste0(arm, "_diagnostics.tsv")))
                # The full result is the commit marker. Save it LAST so an
                # interruption cannot mark an arm complete without its tables.
                coverage_save_rds(result, file.path(rep_path, paste0(arm, ".rds")))
                message("  completed in ", round(result$elapsed_seconds, 1), " seconds")
                rm(result, benchmark, forest, forest_tables, na_tables)
                gc(verbose = FALSE)
            }
            rm(data)
        }
    }
    summarise_coverage_study(path)
    invisible(path)
}

# Summary tables use a common deterministic reporting grid. Event times differ
# between independently generated datasets; their full results are retained in
# each arm's RDS, while coverage across datasets is defined at common times.
summarise_coverage_study <- function(path) {
    saved <- readRDS(file.path(path, "configuration.rds"))
    config <- saved$config
    finite_mean <- function(z) if (any(is.finite(z))) mean(z[is.finite(z)]) else NA_real_
    finite_sum <- function(z) sum(z[is.finite(z)])
    ratio <- function(total, count) ifelse(count > 0, total / count, NA_real_)
    mcse <- function(z) if (length(z) > 1L) stats::sd(z) / sqrt(length(z)) else NA_real_
    sample_variance <- function(total, squared, count) {
        ifelse(count > 1, pmax(0, squared - total^2 / pmax(count, 1)) / (count - 1), NA_real_)
    }
    point_keys <- c("n", "arm", "model", "profile_id", "X1", "X2", "X3", "X4", "X5",
                    "transition", "interval_start", "interval_end", "time", "anchor")
    status_rows <- list()
    work <- list()
    for (n in config$n) {
        for (arm in c("without_noise", "with_noise")) {
            dirs <- file.path(path, paste0("n", n), sprintf("replicate_%03d", seq_len(config$N)))
            primary <- file.path(dirs, paste0(arm, ".rds"))
            grids <- file.path(dirs, paste0(arm, "_report_grid.rds"))
            curve_paths <- file.path(dirs, paste0(arm, "_curves.tsv"))
            primary_exists <- file.exists(primary)
            grid_exists <- file.exists(grids)
            curve_exists <- file.exists(curve_paths)
            complete <- primary_exists & grid_exists & curve_exists
            status_rows[[length(status_rows) + 1L]] <- data.frame(
                n = n, arm = arm, datasets_planned = config$N,
                datasets_completed = sum(primary_exists), datasets_summarised = sum(complete),
                report_grids_available = sum(primary_exists & grid_exists),
                curve_tables_available = sum(primary_exists & curve_exists),
                datasets_missing_outputs = sum(primary_exists & !complete),
                all_requested_complete = all(complete))
            if (any(complete)) {
                work[[length(work) + 1L]] <- list(n = n, arm = arm,
                    replicate = which(complete), grids = grids[complete], curves = curve_paths[complete])
            }
        }
    }
    completion <- data.table::rbindlist(status_rows)
    coverage_write_table(completion, file.path(path, "completion_status.tsv"))
    if (!length(work)) stop("No completed coverage results with all output files found")
    all_profiles <- all_groups <- all_curve_profiles <- all_curve_groups <- list()

    # Read just one completed arm at a time. Per-profile sufficient statistics
    # have a size determined by the reporting grid, independently of N. Group
    # summaries first average profiles within each independent dataset.
    for (part in work) {
        accumulator <- NULL
        group_rows <- vector("list", length(part$grids))
        curve_rows <- vector("list", length(part$grids))
        for (i in seq_along(part$grids)) {
            grid <- data.table::as.data.table(readRDS(part$grids[i]))
            if (any(grid$n != part$n) || any(grid$arm != part$arm) ||
                any(grid$replicate != part$replicate[i])) {
                stop("Checkpoint metadata disagrees with its path: ", part$grids[i])
            }
            if (anyDuplicated(grid[, ..point_keys])) {
                stop("Duplicate profile/time observations in ", part$grids[i])
            }
            sufficient <- grid[, list(
                completed = .N,
                estimable = sum(is.finite(estimate) & is.finite(variance)),
                covered = sum(pointwise_covered %in% TRUE),
                band_covered = sum(simultaneous_covered %in% TRUE),
                width_sum = finite_sum(pointwise_width), width_n = sum(is.finite(pointwise_width)),
                band_width_sum = finite_sum(simultaneous_width), band_width_n = sum(is.finite(simultaneous_width)),
                bias_sum = finite_sum(bias), bias_square_sum = finite_sum(bias^2), bias_n = sum(is.finite(bias)),
                estimate_sum = finite_sum(estimate), estimate_square_sum = finite_sum(estimate^2),
                estimate_n = sum(is.finite(estimate)),
                variance_sum = finite_sum(variance), variance_n = sum(is.finite(variance)),
                empty_sum = sum(empty_stratum %in% TRUE),
                zero_risk_sum = sum(zero_stratum_risk %in% TRUE),
                zero_variance_sum = sum(is.finite(variance) & variance == 0),
                risk_sum = finite_sum(stratum_risk), risk_n = sum(is.finite(stratum_risk))),
                by = point_keys]
            data.table::setorderv(sufficient, point_keys)
            if (is.null(accumulator)) {
                accumulator <- sufficient
            } else {
                if (!identical(accumulator[, ..point_keys], sufficient[, ..point_keys])) {
                    stop("Profile/time grids differ between completed datasets")
                }
                for (column in setdiff(names(sufficient), point_keys)) {
                    data.table::set(accumulator, j = column,
                                    value = accumulator[[column]] + sufficient[[column]])
                }
            }
            within_groups <- function(z) {
                z[, list(
                    coverage = mean(pointwise_covered %in% TRUE),
                    estimable_fraction = mean(is.finite(estimate) & is.finite(variance)),
                    band_pointwise_coverage = mean(simultaneous_covered %in% TRUE),
                    mean_width = finite_mean(pointwise_width), mean_bias = finite_mean(bias),
                    mean_squared_error = finite_mean(bias^2),
                    zero_variance_fraction = mean(is.finite(variance) & variance == 0)),
                    by = c("n", "replicate", "arm", "model", "group", "transition", "time", "anchor")]
            }
            grid[, group := paste0("X1=", X1)]
            by_x1 <- within_groups(grid)
            grid[, group := "overall"]
            group_rows[[i]] <- data.table::rbindlist(list(by_x1, within_groups(grid)))
            curve_rows[[i]] <- data.table::fread(part$curves[i])
            if (any(curve_rows[[i]]$n != part$n) || any(curve_rows[[i]]$arm != part$arm) ||
                any(curve_rows[[i]]$replicate != part$replicate[i])) {
                stop("Curve checkpoint metadata disagrees with its path: ", part$curves[i])
            }
            rm(grid, sufficient, by_x1)
        }
        profile_summary <- accumulator[, ..point_keys]
        profile_summary[, `:=`(
            datasets_completed = accumulator$completed, datasets_planned = config$N,
            datasets_estimable = accumulator$estimable,
            pointwise_covered_count = accumulator$covered,
            pointwise_coverage = accumulator$covered / accumulator$completed,
            pointwise_coverage_when_estimable = ratio(accumulator$covered, accumulator$estimable),
            band_pointwise_coverage = accumulator$band_covered / accumulator$completed,
            mean_pointwise_width = ratio(accumulator$width_sum, accumulator$width_n),
            mean_simultaneous_width = ratio(accumulator$band_width_sum, accumulator$band_width_n),
            bias = ratio(accumulator$bias_sum, accumulator$bias_n),
            rmse = sqrt(ratio(accumulator$bias_square_sum, accumulator$bias_n)),
            empirical_variance = sample_variance(accumulator$estimate_sum,
                                                accumulator$estimate_square_sum,
                                                accumulator$estimate_n),
            mean_estimated_variance = ratio(accumulator$variance_sum, accumulator$variance_n),
            empty_stratum_rate = accumulator$empty_sum / accumulator$completed,
            zero_stratum_risk_rate = accumulator$zero_risk_sum / accumulator$completed,
            zero_variance_rate = accumulator$zero_variance_sum / accumulator$completed,
            mean_stratum_risk = ratio(accumulator$risk_sum, accumulator$risk_n))]
        profile_summary[, coverage_mcse := sqrt(pointwise_coverage * (1 - pointwise_coverage) / datasets_completed)]
        within <- data.table::rbindlist(group_rows)
        group_summary <- within[, list(
            datasets_completed = .N, datasets_planned = config$N,
            pointwise_coverage = mean(coverage), coverage_mcse = mcse(coverage),
            estimable_fraction = mean(estimable_fraction),
            band_pointwise_coverage = mean(band_pointwise_coverage),
            mean_pointwise_width = finite_mean(mean_width), bias = finite_mean(mean_bias),
            rmse = sqrt(finite_mean(mean_squared_error)), zero_variance_fraction = mean(zero_variance_fraction)),
            by = c("n", "arm", "model", "group", "transition", "time", "anchor")]

        # These are one-row-per-profile summaries, much smaller than time grids.
        curves <- data.table::rbindlist(curve_rows)
        curve_profile <- curves[, list(
            datasets_completed = .N, datasets_planned = config$N,
            datasets_estimable = sum(estimable %in% TRUE),
            simultaneous_coverage = mean(simultaneous_covered %in% TRUE),
            simultaneous_coverage_when_estimable = if (any(estimable %in% TRUE))
                sum(simultaneous_covered %in% TRUE) / sum(estimable %in% TRUE) else NA_real_,
            pointwise_entire_curve_coverage = mean(pointwise_entire_curve_covered %in% TRUE),
            mean_pointwise_coverage = sum(mean_pointwise_covered, na.rm = TRUE) / .N,
            mean_pointwise_width = finite_mean(mean_pointwise_width),
            mean_simultaneous_width = finite_mean(mean_simultaneous_width),
            terminal_bias = finite_mean(terminal_bias), terminal_rmse = sqrt(finite_mean(terminal_bias^2)),
            terminal_empirical_variance = if (sum(is.finite(terminal_estimate)) > 1L)
                stats::var(terminal_estimate[is.finite(terminal_estimate)]) else NA_real_,
            terminal_mean_estimated_variance = finite_mean(terminal_variance),
            empty_stratum_rate = mean(empty_stratum %in% TRUE),
            zero_risk_anywhere_rate = mean(min_stratum_risk == 0, na.rm = TRUE),
            zero_event_rate = mean(terminal_stratum_events == 0, na.rm = TRUE),
            terminal_zero_variance_rate = mean(terminal_zero_variance %in% TRUE)),
            by = c("n", "arm", "model", "profile_id", "X1", "transition", "interval_start", "interval_end")]
        curve_profile[, simultaneous_coverage_mcse :=
            sqrt(simultaneous_coverage * (1 - simultaneous_coverage) / datasets_completed)]
        within_curve_groups <- function(z) {
            z[, list(
                simultaneous_coverage = mean(simultaneous_covered %in% TRUE),
                mean_pointwise_coverage = sum(mean_pointwise_covered, na.rm = TRUE) / .N,
                estimable_fraction = mean(estimable %in% TRUE),
                mean_pointwise_width = finite_mean(mean_pointwise_width),
                mean_simultaneous_width = finite_mean(mean_simultaneous_width),
                zero_risk_fraction = mean(min_stratum_risk == 0, na.rm = TRUE),
                zero_event_fraction = mean(terminal_stratum_events == 0, na.rm = TRUE)),
                by = c("n", "replicate", "arm", "model", "group", "transition", "interval_start", "interval_end")]
        }
        curves[, group := paste0("X1=", X1)]
        by_x1 <- within_curve_groups(curves)
        curves[, group := "overall"]
        curve_within <- data.table::rbindlist(list(by_x1, within_curve_groups(curves)))
        curve_group <- curve_within[, list(
            datasets_completed = .N, datasets_planned = config$N,
            simultaneous_coverage = mean(simultaneous_coverage),
            simultaneous_coverage_mcse = mcse(simultaneous_coverage),
            mean_pointwise_coverage = mean(mean_pointwise_coverage),
            estimable_fraction = mean(estimable_fraction),
            mean_pointwise_width = finite_mean(mean_pointwise_width),
            mean_simultaneous_width = finite_mean(mean_simultaneous_width),
            zero_risk_fraction = mean(zero_risk_fraction), zero_event_fraction = mean(zero_event_fraction)),
            by = c("n", "arm", "model", "group", "transition", "interval_start", "interval_end")]
        all_profiles[[length(all_profiles) + 1L]] <- profile_summary
        all_groups[[length(all_groups) + 1L]] <- group_summary
        all_curve_profiles[[length(all_curve_profiles) + 1L]] <- curve_profile
        all_curve_groups[[length(all_curve_groups) + 1L]] <- curve_group
        rm(accumulator, group_rows, curve_rows, within, curves, by_x1, curve_within)
        gc(verbose = FALSE)
    }
    result <- list(pointwise_by_profile = data.table::rbindlist(all_profiles),
                   pointwise_by_group = data.table::rbindlist(all_groups),
                   simultaneous_by_profile = data.table::rbindlist(all_curve_profiles),
                   simultaneous_by_group = data.table::rbindlist(all_curve_groups))
    for (name in names(result)) {
        coverage_write_table(result[[name]], file.path(path, paste0(name, ".tsv")))
    }
    invisible(result)
}
