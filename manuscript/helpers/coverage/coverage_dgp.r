# Shared non-proportional-hazards intensities and Nelson--Aalen benchmark. Only base R is required.
# simulate_non_proportional_hazards retains the sample-size study's all-state-1 initial law.
# simulate_coverage_data uses the coverage study's (1/2, 1/2, 0) initial law.

# The 1 -> 2 hazard changes at different times for the X1 groups. The 2 -> 3
# hazard has another pair of switches depending on X3. These time-dependent
# hazard ratios cannot be represented by a Cox model with fixed coefficients,
# even if it has the correct covariate interactions. X4 and X5 are noise.
# Independent C ~ Uniform(5, 25); the censoring fraction depends on the initial law.
simulate_non_proportional_hazards <- function(n, seed = 2026,
                            a1 = 0.52, b1 = 0.50, s1 = 1.2,
                            a2 = 0.02, b2 = 3.00, s2 = 3.3,
                            rate_13 = 0.02, rate_21 = 0.06,
                            a3 = 0.08, b3 = 0.70, s3 = 1.5,
                            a4 = 0.45, b4 = -0.40, s4 = 2.3,
                            censor_lower = 5, censor_upper = 25,
                            initial_distribution = c(1, 0, 0)) {
    if (!is.numeric(initial_distribution) || length(initial_distribution) != 3L ||
        any(!is.finite(initial_distribution)) || any(initial_distribution < 0) ||
        abs(sum(initial_distribution) - 1) > 1e-12) {
        stop("initial_distribution must contain three nonnegative probabilities summing to 1")
    }
    set.seed(seed)
    X1 <- rbinom(n, 1L, 1 / 2)
    X2 <- sample.int(4L, n, replace = TRUE)
    X3 <- rbinom(n, 1L, 1 / 2)
    X4 <- rbinom(n, 1L, 1 / 2)
    X5 <- sample.int(3L, n, replace = TRUE)
    feature_data <- data.frame(X1, X2, X3, X4, X5)
    censor_times <- runif(n, min = censor_lower, max = censor_upper)
    # Do not consume any extra random numbers for the original deterministic
    # initial law: existing sample-size simulations remain exactly reproducible.
    support <- which(initial_distribution > 0)
    initial_states <- if (length(support) == 1L) {
        rep.int(support, n)
    } else {
        sample.int(3L, n, replace = TRUE, prob = initial_distribution)
    }
    x2_effect <- c(-1, 0, 1, 4)
    # The 1 -> 2 rates are 0.52 before s1 and 0.02, 0.52, 1.02, or 2.52
    # afterward for X1 = 1. For X1 = 0 they jump from 0.02 to 3.02 at s2.

    # Rates are constant between the switch times. Simulate the next jump
    # exactly within each interval, then advance to the next switch if needed.
    jump_data <- vector("list", n)
    switch_times <- sort(unique(c(s1, s2, s3, s4, censor_upper)))
    for (i in seq_len(n)) {
        censor_time <- censor_times[i]
        time <- 0
        state <- initial_states[i]
        times <- 0
        states <- state
        while (time < censor_time && state != 3L) {
            next_change <- min(censor_time, switch_times[switch_times > time])
            if (state == 1L) {
                rate_12 <- if (X1[i] == 1L) {
                    a1 + b1 * (time >= s1) * x2_effect[X2[i]]
                } else {
                    a2 + b2 * (time >= s2)
                }
                rates <- c(rate_12, rate_13)
                destinations <- c(2L, 3L)
            } else {
                rate_23 <- if (X3[i] == 1L) {
                    a3 + b3 * (time >= s3)
                } else {
                    a4 + b4 * (time >= s4)
                }
                rates <- c(rate_21, rate_23)
                destinations <- c(1L, 3L)
            }
            waiting_time <- rexp(1L, rate = sum(rates))
            if (time + waiting_time < next_change) {
                time <- time + waiting_time
                state <- sample(destinations, 1L, prob = rates)
                times <- c(times, time)
                states <- c(states, state)
            } else {
                time <- next_change
            }
        }
        if (state != 3L) {
            times <- c(times, censor_time)
            states <- c(states, state)
        }
        jump_data[[i]] <- list(times = times, states = states)
    }
    list(jump_data = jump_data, feature_data = feature_data,
         censor_times = censor_times)
}

coverage_initial_distribution <- c(0.5, 0.5, 0)

# Initial state is drawn independently of X and C. Keep this entry point
# separate from the simulator used by the earlier sample-size/noise studies.
simulate_coverage_data <- function(
    n, seed = 2026, ..., initial_distribution = coverage_initial_distribution) {
    simulate_non_proportional_hazards(n = n, seed = seed, ...,
                    initial_distribution = initial_distribution)
}

non_proportional_hazards_parameters <- list(
    a1 = 0.52, b1 = 0.50, s1 = 1.2,
    a2 = 0.02, b2 = 3.00, s2 = 3.3,
    rate_13 = 0.02, rate_21 = 0.06,
    a3 = 0.08, b3 = 0.70, s3 = 1.5,
    a4 = 0.45, b4 = -0.40, s4 = 2.3,
    censor_lower = 5, censor_upper = 25
)

coverage_transition_names <- c("1->2", "1->3", "2->1", "2->3")

# Every covariate profile has the same probability within each arm. Profile IDs
# encode their actual values so they remain stable after subsetting or sorting.
coverage_profiles <- function(include_noise = TRUE) {
    if (!is.logical(include_noise) || length(include_noise) != 1L ||
        is.na(include_noise)) {
        stop("include_noise must be TRUE or FALSE")
    }
    levels <- list(X1 = 0:1, X2 = 1:4, X3 = 0:1)
    if (include_noise) levels <- c(levels, list(X4 = 0:1, X5 = 1:3))
    profiles <- do.call(expand.grid, c(levels, KEEP.OUT.ATTRS = FALSE))
    ids <- do.call(paste, c(lapply(names(profiles), function(name) {
        paste0(name, "=", profiles[[name]])
    }), sep = ";"))
    data.frame(profile_id = ids, profiles, check.names = FALSE)
}

# Exact cumulative rates from zero; these do not depend on the initial law.
# The coverage study now uses these full curves for all four transitions.
coverage_true_cumulative <- function(times, profile,
                                    parameters = non_proportional_hazards_parameters) {
    if (!is.numeric(times) || !length(times) ||
        any(!is.finite(times)) || any(times < 0)) {
        stop("times must be a nonempty vector of finite non-negative times")
    }
    if (!is.list(parameters)) stop("parameters must be a named list")
    unknown <- setdiff(names(parameters), names(non_proportional_hazards_parameters))
    if (length(unknown)) stop("Unknown DGP parameter: ", unknown[1L])
    parameters <- utils::modifyList(non_proportional_hazards_parameters, parameters)
    required <- c("X1", "X2", "X3")
    if (!all(required %in% names(profile))) {
        stop("profile must contain X1, X2, and X3")
    }
    values <- vapply(required, function(name) {
        value <- profile[[name]]
        if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
            stop("profile must describe exactly one numeric covariate profile")
        }
        as.numeric(value)
    }, numeric(1))
    if (!(values["X1"] %in% 0:1) || !(values["X2"] %in% 1:4) ||
        !(values["X3"] %in% 0:1)) {
        stop("profile has a covariate value outside the DGP support")
    }
    p <- parameters
    a12 <- if (values["X1"] == 1) {
        p$a1 * times + p$b1 * c(-1, 0, 1, 4)[values["X2"]] *
            pmax(times - p$s1, 0)
    } else {
        p$a2 * times + p$b2 * pmax(times - p$s2, 0)
    }
    a23 <- if (values["X3"] == 1) {
        p$a3 * times + p$b3 * pmax(times - p$s3, 0)
    } else {
        p$a4 * times + p$b4 * pmax(times - p$s4, 0)
    }
    result <- cbind(a12, p$rate_13 * times, p$rate_21 * times, a23)
    dimnames(result) <- list(as.character(times), coverage_transition_names)
    result
}

# Validate the trajectory representation and collect exact event/censoring
# records once. A repeated final state represents censoring, not a transition.
.coverage_path_records <- function(data) {
    paths <- data$jump_data
    if (!is.list(paths)) stop("data$jump_data must be a list of trajectories")
    records <- vector("list", length(paths))
    for (i in seq_along(paths)) {
        times <- paths[[i]]$times
        states <- paths[[i]]$states
        if (!is.numeric(times) || !is.numeric(states) || !length(times) ||
            length(times) != length(states) || any(!is.finite(times)) ||
            any(!is.finite(states)) || times[1L] != 0 ||
            any(diff(times) <= 0) || any(!states %in% 1:3)) {
            stop("Invalid times/states in trajectory ", i)
        }
        last <- length(times)
        next_ids <- if (last > 1L) seq.int(2L, last) else integer(0)
        from <- states[next_ids - 1L]
        to <- states[next_ids]
        changed <- from != to
        if (any(next_ids[!changed] != last) || any(from[changed] == 3L)) {
            stop("Only a final censoring record may repeat a state; state 3 ",
                 "must be absorbing (trajectory ", i, ")")
        }
        transitions <- if (any(changed)) {
            paste0(from[changed], "->", to[changed])
        } else character(0)
        if (any(!transitions %in% coverage_transition_names)) {
            stop("Unsupported transition in trajectory ", i)
        }
        records[[i]] <- list(
            initial = states[1L],
            event_times = times[next_ids[changed]],
            from = from[changed], to = to[changed],
            transition = match(transitions, coverage_transition_names),
            # This also handles observation ending immediately after a jump
            # into a transient state, without assuming unobserved follow-up.
            censor_time = if (states[last] != 3L) times[last] else numeric(0),
            censor_state = if (states[last] != 3L) states[last] else integer(0)
        )
    }
    records
}

coverage_event_times <- function(data) {
    records <- .coverage_path_records(data)
    sort(unique(c(0, unlist(lapply(records, `[[`, "event_times"),
                            use.names = FALSE))))
}

.coverage_profile_keys <- function(features, feature_names) {
    do.call(paste, c(unname(features[feature_names]), sep = "\r"))
}

# Fully stratified Nelson--Aalen estimates and raw sampling variance estimates.
# A/Q/events have dimensions [time, transition, profile]. risk is Y_j(t-),
# with dimensions [time, origin state (1,2), profile]. At time zero risk means
# the initial risk set. At a censoring/event time the subject is still at risk
# in its previous state; the change acts strictly after that time.
#
# Every jump is used before evaluation on `times`; requesting a sparse grid
# never combines several jumps before squaring. Empty strata have NA A/Q,
# zero risk/events, and n_profile=0. Nonempty strata with no events have A=Q=0.
coverage_stratified_na <- function(data, profiles, feature_names, times) {
    if (!is.data.frame(data$feature_data) || !is.data.frame(profiles)) {
        stop("data$feature_data and profiles must be data frames")
    }
    if (!is.character(feature_names) || !length(feature_names) ||
        anyDuplicated(feature_names) ||
        !all(feature_names %in% names(data$feature_data)) ||
        !all(feature_names %in% names(profiles))) {
        stop("feature_names must identify distinct columns in data and profiles")
    }
    if (!nrow(profiles) || !"profile_id" %in% names(profiles) ||
        anyNA(profiles$profile_id) || anyDuplicated(profiles$profile_id)) {
        stop("profiles must have nonempty, unique profile_id values")
    }
    if (anyNA(data$feature_data[feature_names]) ||
        anyNA(profiles[feature_names])) {
        stop("The stratification variables must not contain missing values")
    }
    if (!is.numeric(times) || !length(times) || any(!is.finite(times)) ||
        any(times < 0) || is.unsorted(times, strictly = TRUE)) {
        stop("times must be finite, non-negative and strictly increasing")
    }
    records <- .coverage_path_records(data)
    if (nrow(data$feature_data) != length(records)) {
        stop("The numbers of feature rows and trajectories differ")
    }
    keys <- .coverage_profile_keys(data$feature_data, feature_names)
    profile_keys <- .coverage_profile_keys(profiles, feature_names)
    indices <- split(seq_along(records), keys)
    nt <- length(times)
    np <- nrow(profiles)
    ids <- as.character(profiles$profile_id)
    labels <- list(time = as.character(times),
                   transition = coverage_transition_names, profile = ids)
    A <- array(NA_real_, c(nt, 4L, np), dimnames = labels)
    Q <- A
    events <- array(0, c(nt, 4L, np), dimnames = labels)
    risk <- array(0, c(nt, 2L, np), dimnames = list(
        time = as.character(times), origin_state = c("1", "2"), profile = ids
    ))
    n_profile <- setNames(integer(np), ids)
    for (k in seq_len(np)) {
        members <- indices[[profile_keys[k]]]
        n_profile[k] <- length(members)
        if (!length(members)) next
        selected <- records[members]
        flatten <- function(name) {
            unlist(lapply(selected, `[[`, name), use.names = FALSE)
        }
        initial <- flatten("initial")
        ev_times <- flatten("event_times")
        from <- flatten("from")
        to <- flatten("to")
        transition <- flatten("transition")
        censor_times <- flatten("censor_time")
        censor_states <- flatten("censor_state")
        change_times <- sort(unique(c(ev_times, censor_times)))
        number_changes <- length(change_times)
        initial_risk <- c(sum(initial == 1L), sum(initial == 2L))
        ev_ids <- match(ev_times, change_times)
        censor_ids <- match(censor_times, change_times)
        risk_after <- matrix(0, number_changes, 2L)
        for (j in 1:2) {
            changes <- tabulate(ev_ids[to == j], number_changes) -
                tabulate(ev_ids[from == j], number_changes) -
                tabulate(censor_ids[censor_states == j], number_changes)
            risk_after[, j] <- initial_risk[j] + cumsum(changes)
        }
        if (any(risk_after < 0)) stop("Negative risk set in profile ", ids[k])
        left_risk <- function(query_times) {
            prior <- findInterval(query_times, change_times, left.open = TRUE)
            rbind(initial_risk, risk_after)[prior + 1L, , drop = FALSE]
        }
        risk[, , k] <- left_risk(times)
        A[, , k] <- Q[, , k] <- 0
        if (!length(ev_times)) next
        event_grid <- sort(unique(ev_times))
        number_events <- length(event_grid)
        event_ids <- match(ev_times, event_grid)
        event_risk <- left_risk(event_grid)
        eval_ids <- findInterval(times, event_grid)
        source <- c(1L, 1L, 2L, 2L)
        for (tr in seq_len(4L)) {
            counts <- tabulate(event_ids[transition == tr], number_events)
            denominator <- event_risk[, source[tr]]
            if (any(counts > 0 & denominator == 0)) {
                stop("An event has an empty source-state risk set")
            }
            increment <- variance_increment <- numeric(number_events)
            observed <- counts > 0
            increment[observed] <- counts[observed] / denominator[observed]
            variance_increment[observed] <-
                counts[observed] / denominator[observed]^2
            A[, tr, k] <- c(0, cumsum(increment))[eval_ids + 1L]
            Q[, tr, k] <- c(0, cumsum(variance_increment))[eval_ids + 1L]
            events[, tr, k] <- c(0, cumsum(counts))[eval_ids + 1L]
        }
    }
    list(A = A, Q = Q, risk = risk, n_profile = n_profile,
         events = events, times = times, profiles = profiles,
         feature_names = feature_names)
}
