# Systematic parameter study for the multi-state Fleming--Harrington rule
# ==========================================================================
#
# This is an empirical study, not a package unit test.  It answers four
# questions for the simple three-state model at the beginning of
# testing/test_multistate.r:
#
#   1. How does changing one of a1, b1, a2, or b2 affect held-out error?
#   2. Are there important interactions between the two transient states?
#   3. Do the exponents materially change tree size or depth?
#   4. Are apparent improvements stable across simulated data sets?
#
# Every parameter configuration in a replication uses the same training
# sample, independent test sample, and fitting seed.  Differences from the
# all-one baseline are therefore paired, which removes a large amount of
# irrelevant Monte Carlo variation.
#
# Full study:
#   Rscript testing/test_fh_parameter_effects.R
#
# Add the slower, selected-configuration forest robustness check:
#   FH_STUDY_RUN_FORESTS=true Rscript testing/test_fh_parameter_effects.R
#
# Fast smoke test:
#   FH_STUDY_QUICK=true Rscript testing/test_fh_parameter_effects.R
#
# Useful overrides include FH_STUDY_REPS, FH_STUDY_N_TRAIN,
# FH_STUDY_N_TEST, FH_STUDY_NSPLITS, FH_STUDY_RUN_FORESTS,
# FH_STUDY_FOREST_REPS, FH_STUDY_NTREES, and FH_STUDY_OUTPUT_DIR.


# Setup ---------------------------------------------------------------------

find_repository_root <- function() {
  file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_directory <- if (length(file_argument) > 0L) {
    dirname(normalizePath(sub("^--file=", "", file_argument[1L])))
  } else {
    character(0)
  }

  candidates <- unique(c(
    getwd(),
    dirname(getwd()),
    script_directory,
    dirname(script_directory)
  ))
  candidates <- candidates[nzchar(candidates)]
  is_repository <- vapply(
    candidates,
    function(path) file.exists(file.path(path, "DESCRIPTION")),
    logical(1)
  )
  if (!any(is_repository)) {
    stop("Run this script from the JumpForests repository or install JumpForests.")
  }
  normalizePath(candidates[which(is_repository)[1L]])
}

repository_root <- find_repository_root()

if (requireNamespace("devtools", quietly = TRUE)) {
  devtools::load_all(repository_root, quiet = TRUE)
} else {
  suppressPackageStartupMessages(library(JumpForests))
}

environment_flag <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) {
    return(default)
  }
  value <- tolower(value)
  if (value %in% c("true", "t", "yes", "y", "1")) {
    return(TRUE)
  }
  if (value %in% c("false", "f", "no", "n", "0")) {
    return(FALSE)
  }
  stop(name, " must be true or false.")
}

environment_integer <- function(name, default, minimum = 1L) {
  text_value <- Sys.getenv(name, unset = "")
  if (!nzchar(text_value)) {
    return(as.integer(default))
  }
  if (!grepl("^[0-9]+$", text_value)) {
    stop(name, " must be an integer at least ", minimum, ".")
  }
  numeric_value <- suppressWarnings(as.numeric(text_value))
  if (!is.finite(numeric_value) ||
      numeric_value < minimum ||
      numeric_value > .Machine$integer.max) {
    stop(name, " must be an integer at least ", minimum, ".")
  }
  as.integer(numeric_value)
}

quick_mode <- environment_flag("FH_STUDY_QUICK", FALSE)
default_output_directory <- file.path(
  repository_root, "testing", "fh_parameter_study_output"
)

study_settings <- list(
  quick = quick_mode,
  replications = environment_integer(
    "FH_STUDY_REPS", if (quick_mode) 2L else 30L
  ),
  n_train = environment_integer(
    "FH_STUDY_N_TRAIN", if (quick_mode) 250L else 1000L
  ),
  n_test = environment_integer(
    "FH_STUDY_N_TEST", if (quick_mode) 500L else 2000L
  ),
  min_node_size = environment_integer("FH_STUDY_MIN_NODE_SIZE", 20L),
  nsplits = environment_integer(
    "FH_STUDY_NSPLITS", if (quick_mode) 10L else 50L
  ),
  honest = environment_flag("FH_STUDY_HONEST", FALSE),
  base_seed = environment_integer("FH_STUDY_SEED", 2026L, minimum = 0L),
  run_forests = environment_flag("FH_STUDY_RUN_FORESTS", FALSE),
  forest_replications = environment_integer(
    "FH_STUDY_FOREST_REPS", if (quick_mode) 1L else 5L
  ),
  ntrees = environment_integer(
    "FH_STUDY_NTREES", if (quick_mode) 10L else 100L
  ),
  output_directory = Sys.getenv(
    "FH_STUDY_OUTPUT_DIR", unset = default_output_directory
  )
)

study_settings$forest_replications <- min(
  study_settings$forest_replications,
  study_settings$replications
)
if (!nzchar(study_settings$output_directory)) {
  stop("FH_STUDY_OUTPUT_DIR must not be empty.")
}


# Three-state data-generating process ---------------------------------------

# States 1 and 2 are transient and state 3 is absorbing.  X1 is the signal:
# it slows every transition through the factor 1 / (1 + X1 * time).  X2 is
# independent Gaussian noise.
three_state_jump_rate <- function(state, time, duration) {
  if (state == 1L) {
    2
  } else if (state == 2L) {
    3
  } else {
    0
  }
}

three_state_mark_distribution <- function(state, time, duration) {
  if (state == 1L) {
    c(0, 1 / 2, 1 / 2)
  } else if (state == 2L) {
    c(2 / 3, 0, 1 / 3)
  } else {
    c(0, 0, 1)
  }
}

simulate_three_state_sample <- function(n, seed) {
  stopifnot(length(n) == 1L, n >= 1L, length(seed) == 1L)
  set.seed(seed)

  signal <- runif(n)
  noise <- rnorm(n)
  censoring_times <- runif(n, min = 0, max = 5)
  paths <- vector("list", n)

  for (observation in seq_len(n)) {
    rates <- function(state, time, duration) {
      three_state_jump_rate(state, time, duration) /
        (1 + signal[observation] * time)
    }

    paths[[observation]] <- sim_path(
      i = sample(1:2, size = 1L),
      rates = rates,
      dists = three_state_mark_distribution,
      tn = censoring_times[observation],
      # The rates are decreasing in time, so 2 and 3 are valid global bounds.
      # testing/test_multistate.r used c(2*C, 3*C, 0), which is not a bound
      # when a censoring time C is below one and can bias the simulation.
      bs = c(2, 3, 0)
    )
  }

  features <- data.frame(X1 = signal, X2 = noise)
  observed_states <- sort(unique(unlist(lapply(paths, `[[`, "states"))))

  if (!identical(observed_states, 1:3)) {
    stop(
      "The simulated training/evaluation sample did not contain all three ",
      "states; increase its size or change the seed."
    )
  }
  if (length(paths) != nrow(features) ||
      any(!is.finite(as.matrix(features)))) {
    stop("The simulated paths and feature data are inconsistent.")
  }

  censored <- vapply(seq_len(n), function(observation) {
    isTRUE(all.equal(
      tail(paths[[observation]]$times, 1L),
      censoring_times[observation],
      tolerance = 1e-12
    ))
  }, logical(1))

  list(
    paths = paths,
    features = features,
    censoring_times = censoring_times,
    censoring_fraction = mean(censored)
  )
}


# Parameter configurations --------------------------------------------------

new_configuration <- function(
    id, label, family,
    a1, a2, b1, b2,
    oat_parameter = NA_character_, oat_level = NA_real_) {
  data.frame(
    config_id = id,
    label = label,
    family = family,
    a1 = a1,
    a2 = a2,
    a3 = 1,
    b1 = b1,
    b2 = b2,
    b3 = 1,
    oat_parameter = oat_parameter,
    oat_level = oat_level,
    stringsAsFactors = FALSE
  )
}

format_exponent <- function(value) {
  formatted <- format(value, trim = TRUE, scientific = FALSE)
  gsub("\\.", "p", formatted)
}

make_parameter_configurations <- function() {
  # The one-at-a-time study isolates the effect of each active coordinate.
  # A full Cartesian grid has 6^4 = 1296 configurations and obscures the
  # main effects, so a small set of interpretable interactions is added below.
  levels <- c(0, 0.5, 1, 2, 5, 10)
  configurations <- list(new_configuration(
    id = "baseline",
    label = "All-one baseline",
    family = "baseline",
    a1 = 1, a2 = 1, b1 = 1, b2 = 1
  ))

  coordinates <- c("a1", "b1", "a2", "b2")
  for (coordinate in coordinates) {
    for (level in levels[levels != 1]) {
      values <- c(a1 = 1, a2 = 1, b1 = 1, b2 = 1)
      values[coordinate] <- level
      configurations[[length(configurations) + 1L]] <- new_configuration(
        id = paste0("oat_", coordinate, "_", format_exponent(level)),
        label = paste0(coordinate, " = ", level),
        family = "one_at_a_time",
        a1 = values["a1"],
        a2 = values["a2"],
        b1 = values["b1"],
        b2 = values["b2"],
        oat_parameter = coordinate,
        oat_level = level
      )
    }
  }

  # These joint profiles test timing and state interactions that cannot be
  # inferred from one-at-a-time changes.
  joint_configurations <- list(
    new_configuration(
      "constant_active", "Constant weights in states 1 and 2", "joint",
      a1 = 0, a2 = 0, b1 = 0, b2 = 0
    ),
    new_configuration(
      "high_occupation_both", "High-occupation profile in both states", "joint",
      a1 = 2, a2 = 2, b1 = 0, b2 = 0
    ),
    new_configuration(
      "low_occupation_both", "Low-occupation profile in both states", "joint",
      a1 = 0, a2 = 0, b1 = 2, b2 = 2
    ),
    new_configuration(
      "central_both", "Moderately concentrated at p = 0.5", "joint",
      a1 = 2, a2 = 2, b1 = 2, b2 = 2
    ),
    new_configuration(
      "state1_low_state2_high",
      "State 1 low occupation; state 2 high occupation", "joint",
      a1 = 0, a2 = 2, b1 = 2, b2 = 0
    ),
    new_configuration(
      "state1_high_state2_low",
      "State 1 high occupation; state 2 low occupation", "joint",
      a1 = 2, a2 = 0, b1 = 0, b2 = 2
    ),
    new_configuration(
      "central_both_10", "Strong p = 0.5 concentration in both states", "joint",
      a1 = 10, a2 = 10, b1 = 10, b2 = 10
    ),
    new_configuration(
      "state1_central_10", "Strong p = 0.5 concentration in state 1", "legacy",
      a1 = 10, a2 = 1, b1 = 10, b2 = 1
    ),
    new_configuration(
      "state2_central_10", "Strong p = 0.5 concentration in state 2", "legacy",
      a1 = 1, a2 = 10, b1 = 1, b2 = 10
    ),
    new_configuration(
      "state1_central_100", "Extreme p = 0.5 concentration in state 1", "legacy",
      a1 = 100, a2 = 1, b1 = 100, b2 = 1
    ),
    new_configuration(
      "state2_central_100", "Extreme p = 0.5 concentration in state 2", "legacy",
      a1 = 1, a2 = 100, b1 = 1, b2 = 100
    )
  )

  configurations <- do.call(
    rbind,
    c(configurations, joint_configurations)
  )
  rownames(configurations) <- NULL

  exponent_columns <- c("a1", "a2", "a3", "b1", "b2", "b3")
  if (anyDuplicated(configurations$config_id) ||
      any(!is.finite(as.matrix(configurations[exponent_columns]))) ||
      any(as.matrix(configurations[exponent_columns]) < 0)) {
    stop("The Fleming--Harrington configuration table is invalid.")
  }
  if (any(configurations$a3 != 1) || any(configurations$b3 != 1)) {
    stop("State 3 must remain fixed in the parameter study.")
  }

  attr(configurations, "oat_levels") <- levels
  configurations
}

configuration_weights <- function(configuration) {
  list(
    a = as.numeric(configuration[c("a1", "a2", "a3")]),
    b = as.numeric(configuration[c("b1", "b2", "b3")])
  )
}


# Exact occupation probabilities and weight interpretation -----------------

# All transition intensities share the same time change.  For X1 = x,
#
#   u_x(t) = log(1 + x*t) / x,
#
# with limiting value t at x = 0.  Starting from (1/2, 1/2, 0), the exact
# conditional occupation probabilities are
#
#   p1 = (2/3) exp(-u) - (1/6) exp(-4u)
#   p2 = (1/3) exp(-u) + (1/6) exp(-4u)
#   p3 = 1 - exp(-u).
#
# For K_j(t) = p_j(t)^a_j * (1-p_j(t))^b_j, the interior maximum is
# p_j = a_j / (a_j + b_j).  Both transient occupations decrease from 0.5
# in this model, so increasing a tends to retain high-occupation/early events,
# whereas increasing b tends to retain lower-occupation/later events.  Equal,
# large exponents concentrate tightly near p = 0.5 and are therefore an early
# profile here.  These labels are model-specific: occupations need not be
# monotone in a general multi-state model.
#
# X1 has no effect at time zero and separates transition rates increasingly
# later in follow-up.  That favors some later emphasis.  Against this, risk
# sets, event counts, and censoring support all decline; state 2 is depleted
# especially quickly.  The useful exponents balance that growing signal
# contrast against growing variance.  Because every transition statistic is
# standardized, a and b change the time shape of its weight rather than acting
# as direct state-importance multipliers.
three_state_occupation <- function(time, x) {
  operational_time <- rep(time, length(x))
  positive_x <- x > 0
  operational_time[positive_x] <- log1p(x[positive_x] * time) / x[positive_x]

  exp_u <- exp(-operational_time)
  exp_4u <- exp(-4 * operational_time)
  cbind(
    p1 = (2 / 3) * exp_u - (1 / 6) * exp_4u,
    p2 = (1 / 3) * exp_u + (1 / 6) * exp_4u,
    p3 = 1 - exp_u
  )
}

pooled_three_state_occupation <- function(times, x_grid = seq(0, 1, length.out = 2001L)) {
  probabilities <- t(vapply(times, function(time) {
    colMeans(three_state_occupation(time, x_grid))
  }, numeric(3)))
  data.frame(time = times, probabilities, row.names = NULL)
}

make_weight_profile_table <- function() {
  occupation <- pooled_three_state_occupation(
    seq(0, 5, length.out = 501L)
  )
  profiles <- data.frame(
    profile = c(
      "constant", "default", "high occupation",
      "low occupation", "sharp p=0.5"
    ),
    a = c(0, 1, 2, 0, 10),
    b = c(0, 1, 0, 2, 10),
    stringsAsFactors = FALSE
  )

  result <- list()
  row_id <- 1L
  for (state in 1:2) {
    probabilities <- occupation[[paste0("p", state)]]
    for (profile_id in seq_len(nrow(profiles))) {
      raw_weight <- probabilities^profiles$a[profile_id] *
        (1 - probabilities)^profiles$b[profile_id]
      normalised_weight <- raw_weight / max(raw_weight)
      result[[row_id]] <- data.frame(
        time = occupation$time,
        state = state,
        profile = profiles$profile[profile_id],
        a = profiles$a[profile_id],
        b = profiles$b[profile_id],
        occupation_probability = probabilities,
        normalised_weight = normalised_weight
      )
      row_id <- row_id + 1L
    }
  }
  do.call(rbind, result)
}


# Fitting and scoring --------------------------------------------------------

tree_table <- function(tree) {
  getFromNamespace("getTreeTable", "JumpForests")(tree$Tree)
}

same_tree_structure <- function(first, second) {
  first_table <- tree_table(first)
  second_table <- tree_table(second)
  identical(first_table$left.daughters, second_table$left.daughters) &&
    identical(first_table$feature.IDs, second_table$feature.IDs) &&
    identical(first_table$thresholds, second_table$thresholds)
}

fit_and_score_tree <- function(
    configuration, training, test, replication, fit_seed, settings) {
  weights <- configuration_weights(configuration)
  start_time <- proc.time()[["elapsed"]]

  fitted <- jftree(
    MM ~ X1 + X2,
    data = training$paths,
    feature_data = training$features,
    splitrule = "flemingharrington",
    fh_weights_a = weights$a,
    fh_weights_b = weights$b,
    mtry = 2L,
    min_node_size = settings$min_node_size,
    nsplits = settings$nsplits,
    honest = settings$honest,
    seed = fit_seed
  )
  errors <- jftree.error(
    fitted,
    new_data = test$features,
    jump_data = test$paths
  )
  elapsed <- proc.time()[["elapsed"]] - start_time

  table <- tree_table(fitted)
  internal_nodes <- table$left.daughters != 0
  split_features <- fitted$feature.names[
    table$feature.IDs[internal_nodes] + 1L
  ]
  root_feature <- if (length(internal_nodes) > 0L && internal_nodes[1L]) {
    fitted$feature.names[table$feature.IDs[1L] + 1L]
  } else {
    NA_character_
  }

  data.frame(
    model = "tree",
    replication = replication,
    config_id = configuration$config_id,
    test_ibs = unname(errors$IBS.error),
    test_nibs = unname(errors$normalised.IBS.error),
    test_ikl = unname(errors$IKL.error),
    test_nikl = unname(errors$normalised.IKL.error),
    train_nibs = fitted$ibs.normalised,
    train_nikl = fitted$ikl.normalised,
    nodes = fitted$num.nodes,
    terminal_nodes = fitted$num.terminal.nodes,
    depth = fitted$tree.depth,
    root_is_signal = if (is.na(root_feature)) NA_real_ else {
      as.numeric(root_feature == "X1")
    },
    signal_split_fraction = if (length(split_features) == 0L) NA_real_ else {
      mean(split_features == "X1")
    },
    fit_and_score_seconds = elapsed,
    train_censoring_fraction = training$censoring_fraction,
    test_censoring_fraction = test$censoring_fraction
  )
}

fit_and_score_forest <- function(
    configuration, training, test, replication, fit_seed, settings) {
  weights <- configuration_weights(configuration)
  start_time <- proc.time()[["elapsed"]]

  fitted <- jfforest(
    MM ~ X1 + X2,
    data = training$paths,
    feature_data = training$features,
    splitrule = "flemingharrington",
    fh_weights_a = weights$a,
    fh_weights_b = weights$b,
    mtry = 2L,
    min_node_size = settings$min_node_size,
    nsplits = settings$nsplits,
    ntrees = settings$ntrees,
    honest = settings$honest,
    swr = FALSE,
    sample_rate = 0.7,
    seed = fit_seed,
    nworkers = 1L,
    save_predictions = FALSE
  )
  errors <- jfforest.error(
    fitted,
    new_data = test$features,
    jump_data = test$paths
  )
  elapsed <- proc.time()[["elapsed"]] - start_time

  data.frame(
    model = "forest",
    replication = replication,
    config_id = configuration$config_id,
    test_ibs = unname(errors$IBS.error),
    test_nibs = unname(errors$normalised.IBS.error),
    test_ikl = unname(errors$IKL.error),
    test_nikl = unname(errors$normalised.IKL.error),
    train_nibs = NA_real_,
    train_nikl = NA_real_,
    nodes = fitted$avg.num.nodes,
    terminal_nodes = fitted$avg.num.terminal.nodes,
    depth = fitted$avg.tree.depth,
    root_is_signal = NA_real_,
    signal_split_fraction = NA_real_,
    fit_and_score_seconds = elapsed,
    train_censoring_fraction = training$censoring_fraction,
    test_censoring_fraction = test$censoring_fraction
  )
}

check_absorbing_state_invariance <- function(training, test, fit_seed, settings) {
  common_arguments <- list(
    formula = MM ~ X1 + X2,
    data = training$paths,
    feature_data = training$features,
    splitrule = "flemingharrington",
    mtry = 2L,
    min_node_size = settings$min_node_size,
    nsplits = settings$nsplits,
    honest = settings$honest,
    seed = fit_seed
  )

  ordinary <- do.call(jftree, c(
    common_arguments,
    list(fh_weights_a = c(1, 1, 1), fh_weights_b = c(1, 1, 1))
  ))
  extreme_a3 <- do.call(jftree, c(
    common_arguments,
    list(fh_weights_a = c(1, 1, 100), fh_weights_b = c(1, 1, 1))
  ))
  extreme_b3 <- do.call(jftree, c(
    common_arguments,
    list(fh_weights_a = c(1, 1, 1), fh_weights_b = c(1, 1, 100))
  ))

  ordinary_error <- jftree.error(
    ordinary, new_data = test$features, jump_data = test$paths
  )
  extreme_a3_error <- jftree.error(
    extreme_a3, new_data = test$features, jump_data = test$paths
  )
  extreme_b3_error <- jftree.error(
    extreme_b3, new_data = test$features, jump_data = test$paths
  )

  a3_passed <- same_tree_structure(ordinary, extreme_a3) &&
    identical(ordinary$num.nodes, extreme_a3$num.nodes) &&
    isTRUE(all.equal(ordinary_error, extreme_a3_error, tolerance = 0))
  b3_passed <- same_tree_structure(ordinary, extreme_b3) &&
    identical(ordinary$num.nodes, extreme_b3$num.nodes) &&
    isTRUE(all.equal(ordinary_error, extreme_b3_error, tolerance = 0))

  if (!a3_passed || !b3_passed) {
    stop(
      "Changing a3 and b3 changed the fitted result, although state 3 is ",
      "absorbing and has no outgoing transition."
    )
  }

  data.frame(
    check = c("absorbing-state a3 is inert", "absorbing-state b3 is inert"),
    baseline_value = c(1, 1),
    comparison_value = c(100, 100),
    passed = c(a3_passed, b3_passed)
  )
}


# Paired Monte Carlo summaries ----------------------------------------------

safe_mean <- function(values) {
  if (all(is.na(values))) NA_real_ else mean(values, na.rm = TRUE)
}

safe_sd <- function(values) {
  values <- values[!is.na(values)]
  if (length(values) < 2L) NA_real_ else stats::sd(values)
}

summarise_values <- function(values) {
  number <- sum(!is.na(values))
  standard_deviation <- safe_sd(values)
  c(
    mean = safe_mean(values),
    sd = standard_deviation,
    mcse = if (number < 2L) NA_real_ else standard_deviation / sqrt(number)
  )
}

paired_statistics <- function(differences, error_metric = FALSE) {
  differences <- differences[!is.na(differences)]
  number <- length(differences)
  if (number == 0L) {
    return(c(
      mean_delta = NA_real_, lower_95 = NA_real_, upper_95 = NA_real_,
      win_rate = NA_real_, p_value = NA_real_
    ))
  }

  mean_delta <- mean(differences)
  if (number < 2L) {
    lower <- upper <- NA_real_
  } else {
    standard_deviation <- stats::sd(differences)
    standard_error <- standard_deviation / sqrt(number)
    margin <- stats::qt(0.975, df = number - 1L) * standard_error
    lower <- mean_delta - margin
    upper <- mean_delta + margin
  }

  p_value <- if (number < 2L) {
    NA_real_
  } else if (standard_deviation == 0) {
    if (mean_delta == 0) 1 else 0
  } else {
    2 * stats::pt(
      -abs(mean_delta / (standard_deviation / sqrt(number))),
      df = number - 1L
    )
  }

  c(
    mean_delta = mean_delta,
    lower_95 = lower,
    upper_95 = upper,
    win_rate = if (error_metric) mean(differences < 0) else NA_real_,
    p_value = p_value
  )
}

summarise_study_results <- function(raw_results, configurations) {
  metrics <- c(
    "test_ibs", "test_nibs", "test_ikl", "test_nikl",
    "train_nibs", "train_nikl", "nodes", "terminal_nodes", "depth",
    "root_is_signal", "signal_split_fraction", "fit_and_score_seconds"
  )
  paired_metrics <- c("test_nibs", "test_nikl", "terminal_nodes", "depth")
  baseline <- raw_results[
    raw_results$config_id == "baseline",
    c("replication", paired_metrics),
    drop = FALSE
  ]
  if (anyDuplicated(baseline$replication)) {
    stop("There must be exactly one baseline result per replication.")
  }

  result <- vector("list", nrow(configurations))
  for (configuration_id in seq_len(nrow(configurations))) {
    configuration <- configurations[configuration_id, , drop = FALSE]
    current <- raw_results[
      raw_results$config_id == configuration$config_id,
      ,
      drop = FALSE
    ]
    row <- configuration
    row$n_replications <- nrow(current)

    for (metric in metrics) {
      metric_summary <- summarise_values(current[[metric]])
      row[[paste0("mean_", metric)]] <- metric_summary["mean"]
      row[[paste0("sd_", metric)]] <- metric_summary["sd"]
      row[[paste0("mcse_", metric)]] <- metric_summary["mcse"]
    }

    paired <- merge(
      current[c("replication", paired_metrics)],
      baseline,
      by = "replication",
      suffixes = c("", "_baseline"),
      sort = TRUE
    )
    for (metric in paired_metrics) {
      differences <- paired[[metric]] - paired[[paste0(metric, "_baseline")]]
      paired_summary <- paired_statistics(
        differences,
        error_metric = metric %in% c("test_nibs", "test_nikl")
      )
      row[[paste0("mean_delta_", metric)]] <- paired_summary["mean_delta"]
      row[[paste0("lower_95_delta_", metric)]] <- paired_summary["lower_95"]
      row[[paste0("upper_95_delta_", metric)]] <- paired_summary["upper_95"]
      row[[paste0("paired_p_value_", metric)]] <- paired_summary["p_value"]
      if (metric %in% c("test_nibs", "test_nikl")) {
        row[[paste0("paired_win_rate_", metric)]] <- paired_summary["win_rate"]
      }
    }
    result[[configuration_id]] <- row
  }

  result <- do.call(rbind, result)
  rownames(result) <- NULL
  result$rank_test_nibs <- rank(result$mean_test_nibs, ties.method = "min")
  result$rank_test_nikl <- rank(result$mean_test_nikl, ties.method = "min")
  for (metric in c("test_nibs", "test_nikl")) {
    p_value_name <- paste0("paired_p_value_", metric)
    adjusted_name <- paste0("holm_p_value_", metric)
    result[[adjusted_name]] <- NA_real_
    comparisons <- result$config_id != "baseline" &
      !is.na(result[[p_value_name]])
    result[[adjusted_name]][comparisons] <- stats::p.adjust(
      result[[p_value_name]][comparisons],
      method = "holm"
    )
  }
  result
}

make_oat_summary <- function(summary, levels) {
  coordinates <- c("a1", "b1", "a2", "b2")
  baseline <- summary[summary$config_id == "baseline", , drop = FALSE]
  result <- list()

  for (coordinate in coordinates) {
    current <- summary[
      summary$family == "one_at_a_time" &
        summary$oat_parameter == coordinate,
      ,
      drop = FALSE
    ]
    baseline_copy <- baseline
    baseline_copy$oat_parameter <- coordinate
    baseline_copy$oat_level <- 1
    current <- rbind(current, baseline_copy)
    current <- current[match(levels, current$oat_level), , drop = FALSE]
    result[[coordinate]] <- current
  }

  do.call(rbind, result)
}


# Output and plots -----------------------------------------------------------

draw_interval_curve <- function(data, metric, lower, upper, title, y_label) {
  x <- log1p(data$oat_level)
  y <- data[[metric]]
  lower_values <- data[[lower]]
  upper_values <- data[[upper]]
  y_limits <- range(
    c(y, lower_values, upper_values, 0),
    finite = TRUE
  )
  if (diff(y_limits) == 0) {
    y_limits <- y_limits + c(-1, 1) * max(abs(y_limits), 1) * 0.05
  }

  plot(
    x, y,
    type = "b", pch = 19,
    xaxt = "n",
    xlab = "Exponent value",
    ylab = y_label,
    main = title,
    ylim = y_limits
  )
  axis(1, at = x, labels = data$oat_level)
  abline(h = 0, col = "grey60", lty = 2)
  valid_intervals <- is.finite(lower_values) & is.finite(upper_values)
  if (any(valid_intervals)) {
    interval_x <- x[valid_intervals]
    interval_lower <- lower_values[valid_intervals]
    interval_upper <- upper_values[valid_intervals]
    cap_width <- 0.015 * diff(range(x))
    segments(
      interval_x, interval_lower,
      interval_x, interval_upper
    )
    segments(
      interval_x - cap_width, interval_lower,
      interval_x + cap_width, interval_lower
    )
    segments(
      interval_x - cap_width, interval_upper,
      interval_x + cap_width, interval_upper
    )
  }
}

write_study_plots <- function(
    tree_summary, oat_summary, weight_profiles, output_file) {
  grDevices::pdf(output_file, width = 10, height = 8)
  old_parameters <- par(no.readonly = TRUE)
  on.exit({
    par(old_parameters)
    grDevices::dev.off()
  }, add = TRUE)

  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (coordinate in c("a1", "b1", "a2", "b2")) {
    draw_interval_curve(
      oat_summary[oat_summary$oat_parameter == coordinate, ],
      metric = "mean_delta_test_nibs",
      lower = "lower_95_delta_test_nibs",
      upper = "upper_95_delta_test_nibs",
      title = paste("Vary", coordinate),
      y_label = "Paired change in test NIBS"
    )
  }

  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (coordinate in c("a1", "b1", "a2", "b2")) {
    draw_interval_curve(
      oat_summary[oat_summary$oat_parameter == coordinate, ],
      metric = "mean_delta_test_nikl",
      lower = "lower_95_delta_test_nikl",
      upper = "upper_95_delta_test_nikl",
      title = paste("Vary", coordinate),
      y_label = "Paired change in test NIKL"
    )
  }

  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (coordinate in c("a1", "b1", "a2", "b2")) {
    draw_interval_curve(
      oat_summary[oat_summary$oat_parameter == coordinate, ],
      metric = "mean_delta_terminal_nodes",
      lower = "lower_95_delta_terminal_nodes",
      upper = "upper_95_delta_terminal_nodes",
      title = paste("Vary", coordinate),
      y_label = "Paired change in terminal nodes"
    )
  }

  par(mfrow = c(1, 1), mar = c(5, 5, 3, 1))
  colours <- ifelse(
    tree_summary$family == "baseline", "red",
    ifelse(tree_summary$family == "one_at_a_time", "grey50", "navy")
  )
  plot(
    tree_summary$mean_terminal_nodes,
    tree_summary$mean_test_nibs,
    pch = 19,
    col = colours,
    xlab = "Mean number of terminal nodes",
    ylab = "Mean held-out normalised IBS",
    main = "Prediction error versus tree complexity"
  )
  legend(
    "topright",
    legend = c("Baseline", "One-at-a-time", "Joint/legacy"),
    col = c("red", "grey50", "navy"),
    pch = 19,
    bty = "n"
  )
  labelled_rows <- unique(c(
    which(tree_summary$config_id == "baseline"),
    order(tree_summary$mean_test_nibs)[seq_len(min(3L, nrow(tree_summary)))]
  ))
  text(
    tree_summary$mean_terminal_nodes[labelled_rows],
    tree_summary$mean_test_nibs[labelled_rows],
    labels = tree_summary$config_id[labelled_rows],
    pos = 3,
    cex = 0.65
  )

  par(mfrow = c(1, 2), mar = c(5, 5, 3, 1))
  profile_names <- unique(weight_profiles$profile)
  profile_colours <- seq_along(profile_names)
  for (state in 1:2) {
    state_profiles <- weight_profiles[weight_profiles$state == state, ]
    plot(
      range(state_profiles$time),
      c(0, 1),
      type = "n",
      xlab = "Calendar time",
      ylab = "FH weight, normalised to maximum",
      main = paste("Root weight profiles: state", state)
    )
    for (profile_id in seq_along(profile_names)) {
      current <- state_profiles[
        state_profiles$profile == profile_names[profile_id],
      ]
      lines(
        current$time,
        current$normalised_weight,
        col = profile_colours[profile_id],
        lwd = 2
      )
    }
    if (state == 2L) {
      legend(
        "right",
        legend = profile_names,
        col = profile_colours,
        lwd = 2,
        cex = 0.8,
        bty = "n"
      )
    }
  }
}

print_study_report <- function(tree_summary, forest_summary = NULL) {
  ranked <- tree_summary[
    order(tree_summary$mean_test_nibs),
    c(
      "label", "mean_test_nibs", "mcse_test_nibs",
      "mean_delta_test_nibs", "lower_95_delta_test_nibs",
      "upper_95_delta_test_nibs", "holm_p_value_test_nibs",
      "mean_terminal_nodes", "mean_depth", "mean_root_is_signal"
    )
  ]

  cat("\nLowest mean held-out tree errors\n")
  print(utils::head(ranked, 10L), row.names = FALSE, digits = 5)

  cat(
    "\nNegative paired differences are improvements over the all-one baseline.",
    "\nThe confidence intervals quantify replication-to-replication uncertainty.",
    "\nThey are exploratory pointwise intervals; Holm p-values account for",
    "\nthe family of configuration-versus-baseline comparisons within each",
    "\nerror metric.",
    "\nComplexity is expected to move only indirectly because FH exponents do",
    "\nnot impose a pruning penalty or minimum split improvement.\n"
  )
  cat(
    "\nAcross configurations, mean terminal-node counts range from",
    sprintf("%.2f", min(tree_summary$mean_terminal_nodes)), "to",
    sprintf("%.2f", max(tree_summary$mean_terminal_nodes)), "and mean depths",
    "range from", sprintf("%.2f", min(tree_summary$mean_depth)), "to",
    sprintf("%.2f", max(tree_summary$mean_depth)), ".\n"
  )

  if (!is.null(forest_summary)) {
    forest_ranked <- forest_summary[
      order(forest_summary$mean_test_nibs),
      c(
        "label", "mean_test_nibs", "mcse_test_nibs",
        "mean_delta_test_nibs", "lower_95_delta_test_nibs",
        "upper_95_delta_test_nibs", "holm_p_value_test_nibs",
        "mean_terminal_nodes", "mean_depth"
      )
    ]
    cat("\nSelected exploratory forest comparison\n")
    print(forest_ranked, row.names = FALSE, digits = 5)
  }
}


# Main study ----------------------------------------------------------------

run_fh_parameter_study <- function(settings = study_settings) {
  configurations <- make_parameter_configurations()
  oat_levels <- attr(configurations, "oat_levels")

  forest_configuration_ids <- c(
    "baseline",
    "constant_active",
    "high_occupation_both",
    "low_occupation_both",
    "central_both",
    "state1_low_state2_high",
    "state1_high_state2_low",
    "central_both_10",
    "state1_central_10",
    "state2_central_10",
    "state1_central_100",
    "state2_central_100",
    "oat_a1_10",
    "oat_b1_10",
    "oat_a2_5",
    "oat_b2_10"
  )
  forest_configurations <- configurations[
    match(forest_configuration_ids, configurations$config_id),
    ,
    drop = FALSE
  ]

  tree_results <- vector(
    "list", settings$replications * nrow(configurations)
  )
  forest_results <- if (settings$run_forests) {
    vector(
      "list",
      settings$forest_replications * nrow(forest_configurations)
    )
  } else {
    list()
  }
  tree_result_id <- 1L
  forest_result_id <- 1L
  absorbing_state_check <- NULL
  largest_seed_offset <- 40000 + settings$replications
  if (settings$base_seed > .Machine$integer.max - largest_seed_offset) {
    stop(
      "FH_STUDY_SEED is too large for the deterministic replication offsets."
    )
  }

  cat(
    "Running", settings$replications, "paired tree replications over",
    nrow(configurations), "FH configurations.\n"
  )
  if (settings$run_forests) {
    cat(
      "Also running", settings$forest_replications,
      "replications of", nrow(forest_configurations),
      "selected", settings$ntrees, "-tree forests.\n"
    )
  }

  for (replication in seq_len(settings$replications)) {
    cat("Replication", replication, "of", settings$replications, "\n")
    training <- simulate_three_state_sample(
      settings$n_train,
      settings$base_seed + 10000L + replication
    )
    test <- simulate_three_state_sample(
      settings$n_test,
      settings$base_seed + 20000L + replication
    )
    tree_seed <- settings$base_seed + 30000L + replication
    forest_seed <- settings$base_seed + 40000L + replication

    if (replication == 1L) {
      absorbing_state_check <- check_absorbing_state_invariance(
        training, test, tree_seed, settings
      )
    }

    for (configuration_id in seq_len(nrow(configurations))) {
      tree_results[[tree_result_id]] <- fit_and_score_tree(
        configurations[configuration_id, , drop = FALSE],
        training,
        test,
        replication,
        tree_seed,
        settings
      )
      tree_result_id <- tree_result_id + 1L
    }

    if (settings$run_forests &&
        replication <= settings$forest_replications) {
      for (configuration_id in seq_len(nrow(forest_configurations))) {
        forest_results[[forest_result_id]] <- fit_and_score_forest(
          forest_configurations[configuration_id, , drop = FALSE],
          training,
          test,
          replication,
          forest_seed,
          settings
        )
        forest_result_id <- forest_result_id + 1L
      }
    }
  }

  tree_results <- do.call(rbind, tree_results)
  expected_tree_rows <- settings$replications * nrow(configurations)
  if (nrow(tree_results) != expected_tree_rows ||
      any(!is.finite(as.matrix(tree_results[c(
        "test_ibs", "test_nibs", "test_ikl", "test_nikl",
        "nodes", "terminal_nodes", "depth"
      )]))) ||
      any(tree_results$nodes != 2 * tree_results$terminal_nodes - 1) ||
      any(tree_results$nodes < 1) ||
      any(tree_results$depth < 0)) {
    stop("The tree study failed its result-shape or finiteness checks.")
  }

  tree_summary <- summarise_study_results(tree_results, configurations)
  oat_summary <- make_oat_summary(tree_summary, oat_levels)

  if (settings$run_forests) {
    forest_results <- do.call(rbind, forest_results)
    expected_forest_rows <- settings$forest_replications *
      nrow(forest_configurations)
    if (nrow(forest_results) != expected_forest_rows ||
        any(!is.finite(as.matrix(forest_results[c(
          "test_ibs", "test_nibs", "test_ikl", "test_nikl",
          "nodes", "terminal_nodes", "depth"
        )]))) ||
        any(abs(
          forest_results$nodes -
            (2 * forest_results$terminal_nodes - 1)
        ) > 1e-10)) {
      stop("The forest study failed its result-shape or finiteness checks.")
    }
    forest_summary <- summarise_study_results(
      forest_results, forest_configurations
    )
  } else {
    forest_results <- NULL
    forest_summary <- NULL
  }

  occupation_table <- pooled_three_state_occupation(
    c(0, 0.25, 0.5, 1, 2, 3, 5)
  )
  weight_profiles <- make_weight_profile_table()

  dir.create(settings$output_directory, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(
    configurations,
    file.path(settings$output_directory, "fh_parameter_configurations.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    tree_results,
    file.path(settings$output_directory, "fh_tree_results_raw.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    tree_summary,
    file.path(settings$output_directory, "fh_tree_results_summary.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    oat_summary,
    file.path(settings$output_directory, "fh_one_at_a_time_summary.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    absorbing_state_check,
    file.path(settings$output_directory, "fh_absorbing_state_check.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    occupation_table,
    file.path(settings$output_directory, "three_state_occupation.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    weight_profiles,
    file.path(settings$output_directory, "fh_weight_profiles.csv"),
    row.names = FALSE
  )
  if (settings$run_forests) {
    utils::write.csv(
      forest_results,
      file.path(settings$output_directory, "fh_forest_results_raw.csv"),
      row.names = FALSE
    )
    utils::write.csv(
      forest_summary,
      file.path(settings$output_directory, "fh_forest_results_summary.csv"),
      row.names = FALSE
    )
  }
  writeLines(
    capture.output(str(settings)),
    file.path(settings$output_directory, "study_settings.txt")
  )
  write_study_plots(
    tree_summary,
    oat_summary,
    weight_profiles,
    file.path(settings$output_directory, "fh_parameter_study.pdf")
  )

  print_study_report(tree_summary, forest_summary)
  cat("\nResults written to", settings$output_directory, "\n")

  invisible(list(
    settings = settings,
    configurations = configurations,
    tree_raw = tree_results,
    tree_summary = tree_summary,
    oat_summary = oat_summary,
    forest_raw = forest_results,
    forest_summary = forest_summary,
    absorbing_state_check = absorbing_state_check,
    occupation = occupation_table,
    weight_profiles = weight_profiles
  ))
}


if (sys.nframe() == 0L) {
  fh_parameter_study <- run_fh_parameter_study()
}
