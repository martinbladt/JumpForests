#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)
library(tidyverse)

# Testing honesty across minimal node sizes for single regression trees
#-------------------------------------------------------------------------------------------------

# Set run_full_simulation <- FALSE before sourcing the file if only the helper functions are needed.
if (!exists("run_full_simulation")) {
    run_full_simulation <- FALSE
}

if (!exists("run_normality_simulation")) {
    run_normality_simulation <- TRUE
}

if (!exists("B")) {
    B <- 1000
}

if (!exists("K")) {
    K <- 1000
}

if (!exists("data_sizes")) {
    data_sizes <- c(100, 250, 500, 1000, 2000, 5000)
}

if (!exists("node_sizes")) {
    node_sizes <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
}

g <- function(x1, x2 = NULL) {
    if (is.null(x2)) {
        return(2 * x1[1] + 3 * sin(x1[2]))
    }

    2 * x1 + 3 * sin(x2)
}

simulate_data <- function(n) {
    X1 <- runif(n)
    X2 <- runif(n)
    X3 <- runif(n)  # noise
    X4 <- runif(n)  # noise
    Y <- g(X1, X2) + rnorm(n)
    data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3, X4 = X4)
}

run_min_node_size_simulation <- function(B, data_sizes, node_sizes) {
    num_data_sizes <- length(data_sizes)
    num_node_sizes <- length(node_sizes)
    num_rows <- B * num_data_sizes * num_node_sizes * 2

    simulation <- rep(0, num_rows)
    num_obs <- rep(0, num_rows)
    min_node_size <- rep(0, num_rows)
    type <- rep("", num_rows)
    mse <- rep(0, num_rows)
    row <- 1

    for (j in 1:num_data_sizes) {
        n <- data_sizes[j]

        for (b in 1:B) {
            cat("Data size", n, "simulation", b, "\n")
            train_data <- simulate_data(n)

            for (i in 1:num_node_sizes) {
                # dishonest tree
                tree <- jftree(Y ~ X1 + X2, data = train_data, min_node_size = node_sizes[i])
                simulation[row] <- b
                num_obs[row] <- n
                min_node_size[row] <- node_sizes[i]
                type[row] <- "Dishonest"
                mse[row] <- tree$mse.error
                row <- row + 1

                # honest tree
                tree_honest <- jftree(Y ~ X1 + X2, data = train_data, honest = TRUE, min_node_size = node_sizes[i])
                simulation[row] <- b
                num_obs[row] <- n
                min_node_size[row] <- node_sizes[i]
                type[row] <- "Honest"
                mse[row] <- tree_honest$mse.error
                row <- row + 1

                rm(tree, tree_honest)
            }

            gc()
        }
    }

    data.frame(simulation = simulation, num.obs = num_obs, min.node.size = min_node_size, type = type, mse = mse)
}

plot_min_node_size_mse <- function(df, node_sizes) {
    ggplot(data = df) +
        geom_line(aes(x = min.node.size, y = mse, colour = type, linetype = as.factor(num.obs),
                      group = interaction(type, num.obs)), linewidth = 1) +
        geom_point(aes(x = min.node.size, y = mse, colour = type, shape = type,
                       group = interaction(type, num.obs)), size = 1.5) +
        scale_x_continuous(breaks = node_sizes, trans = "log2") +
        xlab("Minimal node size") + ylab("Average MSE") + theme_bw() +
        scale_colour_manual(name = "Type", values = c("Dishonest" = "DarkBlue", "Honest" = "DarkGreen")) +
        scale_shape_manual(name = "Type", values = c("Dishonest" = 16, "Honest" = 17)) +
        scale_linetype_discrete(name = "Size of dataset") +
        theme(legend.position = "bottom")
}

# this actually perfoms the simulations
if (isTRUE(run_full_simulation)) {
    set.seed(2026)
    df_min_node_size_tree_errors <- run_min_node_size_simulation(B, data_sizes, node_sizes)

    if (!dir.exists("DecisionTreePlots")) {
        dir.create("DecisionTreePlots")
    }

    write.table(df_min_node_size_tree_errors, file = "DecisionTreePlots/df_min_node_size_tree_errors.txt", sep = "\t", row.names = FALSE)

    df_min_node_size_errors <- aggregate(
        mse ~ num.obs + min.node.size + type,
        data = df_min_node_size_tree_errors,
        FUN = mean
    )

    write.table(df_min_node_size_errors, file = "DecisionTreePlots/df_min_node_size_tree_errors_avg.txt", sep = "\t", row.names = FALSE)

    min_node_size_mse_plot <- plot_min_node_size_mse(df_min_node_size_errors, node_sizes)
    min_node_size_mse_plot
}

# do the below to plot
if (file.exists("DecisionTreePlots/df_min_node_size_tree_errors.txt") &&
    file.exists("DecisionTreePlots/df_min_node_size_tree_errors_avg.txt")) {
    # read in the data
    df_min_node_size_tree_errors <- read.table("DecisionTreePlots/df_min_node_size_tree_errors.txt", header = TRUE)
    head(df_min_node_size_tree_errors)
    dim(df_min_node_size_tree_errors)

    df_min_node_size_tree_errors_avg <- read.table("DecisionTreePlots/df_min_node_size_tree_errors_avg.txt", header = TRUE)
    head(df_min_node_size_tree_errors_avg)
    dim(df_min_node_size_tree_errors_avg)

    plot_min_node_size_mse(df_min_node_size_tree_errors_avg, node_sizes)
}

# Testing asymptotic normality
#-------------------------------------------------------------------------------------------------
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)
library(tidyverse)

if (!exists("normality_x")) {
    normality_x <- c(1/2, 1/2, 1/2, 1/2)
}

if (!exists("normality_alpha")) {
    normality_alpha <- 0.05
}

if (!exists("normality_min_node_size_powers")) {
    normality_min_node_size_powers <- c(1/2, 2/3, 4/5)
}

format_scaling_power <- function(power) {
    if (abs(power - 1/2) < 10^(-12)) {
        return("1/2")
    }
    if (abs(power - 2/3) < 10^(-12)) {
        return("2/3")
    }
    if (abs(power - 4/5) < 10^(-12)) {
        return("4/5")
    }

    as.character(power)
}

make_scaling_rows <- function(leaf_size, min_node_size, min_node_size_power) {
    data.frame(
        scaling = c("leaf.size^(1/2)", "k_n^(1/2)"),
        scaling.base = c("leaf.size", "k_n"),
        scaling.power = c(1/2, 1/2),
        scaling.factor = c(sqrt(leaf_size), sqrt(min_node_size)),
        min.node.size.power = min_node_size_power
    )
}

make_normality_test_rows <- function() {
    data.frame(
        test = c("Anderson-Darling", "Kolmogorov-Smirnov", "Shapiro-Wilk", "Cramer-von Mises"),
        statistic = rep(NA, 4),
        p.value = rep(NA, 4)
    )
}

standardise_values <- function(values) {
    (values - mean(values)) / sd(values)
}

anderson_darling_test <- function(values) {
    n <- length(values)
    z <- sort(standardise_values(values))
    p <- pnorm(z)
    p <- pmin(pmax(p, .Machine$double.xmin), 1 - .Machine$double.eps)
    i <- 1:n

    A2 <- -n - sum((2 * i - 1) * (log(p) + log(1 - rev(p)))) / n
    A2_star <- A2 * (1 + 0.75 / n + 2.25 / n^2)

    p_value <- if (A2_star < 0.2) {
        1 - exp(-13.436 + 101.14 * A2_star - 223.73 * A2_star^2)
    } else if (A2_star < 0.34) {
        1 - exp(-8.318 + 42.796 * A2_star - 59.938 * A2_star^2)
    } else if (A2_star < 0.6) {
        exp(0.9177 - 4.279 * A2_star - 1.38 * A2_star^2)
    } else if (A2_star < 13) {
        exp(1.2937 - 5.709 * A2_star + 0.0186 * A2_star^2)
    } else {
        0
    }

    data.frame(test = "Anderson-Darling", statistic = A2_star, p.value = pmin(pmax(p_value, 0), 1))
}

kolmogorov_smirnov_test <- function(values) {
    z <- standardise_values(values)
    result <- suppressWarnings(ks.test(z, "pnorm"))
    data.frame(test = "Kolmogorov-Smirnov", statistic = as.numeric(result$statistic), p.value = result$p.value)
}

shapiro_wilk_test <- function(values) {
    if (length(values) > 5000) {
        return(data.frame(test = "Shapiro-Wilk", statistic = NA, p.value = NA))
    }

    result <- shapiro.test(values)
    data.frame(test = "Shapiro-Wilk", statistic = as.numeric(result$statistic), p.value = result$p.value)
}

cramer_von_mises_test <- function(values) {
    n <- length(values)
    z <- sort(standardise_values(values))
    p <- pnorm(z)
    i <- 1:n

    W2 <- 1 / (12 * n) + sum((p - (2 * i - 1) / (2 * n))^2)
    W2_star <- W2 * (1 + 0.5 / n)

    p_value <- if (W2_star < 0.0275) {
        1 - exp(-13.953 + 775.5 * W2_star - 12542.61 * W2_star^2)
    } else if (W2_star < 0.051) {
        1 - exp(-5.903 + 179.546 * W2_star - 1515.29 * W2_star^2)
    } else if (W2_star < 0.092) {
        exp(0.886 - 31.62 * W2_star + 10.897 * W2_star^2)
    } else if (W2_star < 1.1) {
        exp(1.111 - 34.242 * W2_star + 12.832 * W2_star^2)
    } else {
        0
    }

    data.frame(test = "Cramer-von Mises", statistic = W2_star, p.value = pmin(pmax(p_value, 0), 1))
}

run_normality_tests <- function(values) {
    values <- values[is.finite(values)]
    if (length(values) < 3 || sd(values) == 0) {
        return(make_normality_test_rows())
    }

    rbind(
        anderson_darling_test(values),
        kolmogorov_smirnov_test(values),
        shapiro_wilk_test(values),
        cramer_von_mises_test(values)
    )
}

simulate_centered_scaled_tree_predictions <- function(B, n, x, min_node_size, min_node_size_powers) {
    new_data <- data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
    true_value <- g(x)

    results <- list()
    row <- 1

    for (b in 1:B) {
        cat("Data size", n, "normality simulation", b, "\n")
        train_data <- simulate_data(n)

        tree <- jftree(Y ~ X1 + X2, data = train_data, min_node_size = min_node_size)
        n_L <- getNodeSize(tree, x)
        prediction <- as.numeric(jftree.predict(tree, new_data = new_data))
        raw_error <- prediction - true_value
        scaling_rows <- make_scaling_rows(n_L, min_node_size, min_node_size_powers)
        results[[row]] <- data.frame(
            simulation = b,
            type = "Dishonest",
            leaf.size = n_L,
            prediction = prediction,
            raw.error = raw_error,
            scaling_rows,
            centered.scaled = scaling_rows$scaling.factor * raw_error
        )
        row <- row + 1

        tree_honest <- jftree(Y ~ X1 + X2, data = train_data, honest = TRUE, min_node_size = min_node_size)
        n_L_honest <- getNodeSize(tree_honest, x)
        prediction_honest <- as.numeric(jftree.predict(tree_honest, new_data = new_data))
        raw_error_honest <- prediction_honest - true_value
        scaling_rows_honest <- make_scaling_rows(n_L_honest, min_node_size, min_node_size_powers)
        results[[row]] <- data.frame(
            simulation = b,
            type = "Honest",
            leaf.size = n_L_honest,
            prediction = prediction_honest,
            raw.error = raw_error_honest,
            scaling_rows_honest,
            centered.scaled = scaling_rows_honest$scaling.factor * raw_error_honest
        )
        row <- row + 1

        rm(tree, tree_honest)
        gc()
    }

    do.call(rbind, results)
}

run_asymptotic_normality_study <- function(B, K, data_sizes, x, min_node_size_powers) {
    prediction_results <- list()
    test_results <- list()
    prediction_row <- 1
    test_row <- 1

    for (k in 1:K) {
        for (j in 1:length(data_sizes)) {
            n <- data_sizes[j]
            for (current_power in min_node_size_powers) {
                min_node_size <- floor(n^current_power)
                cat("Normality test batch", k, "on data of size", n, "with minimal node size", min_node_size, "\n")

                current_predictions <- simulate_centered_scaled_tree_predictions(B, n, x, min_node_size, current_power)
                current_predictions$batch <- k
                current_predictions$num.obs <- n
                current_predictions$min.node.size <- min_node_size
                current_predictions$min.node.size.power <- current_power
                prediction_results[[prediction_row]] <- current_predictions
                prediction_row <- prediction_row + 1

                for (current_type in c("Dishonest", "Honest")) {
                    for (current_scaling in unique(current_predictions$scaling)) {
                        current_subset <- current_predictions[current_predictions$type == current_type &
                                                                  current_predictions$scaling == current_scaling, ]
                        current_values <- current_subset$centered.scaled
                        current_tests <- run_normality_tests(current_values)
                        current_tests$batch <- k
                        current_tests$num.obs <- n
                        current_tests$min.node.size <- min_node_size
                        current_tests$min.node.size.power <- current_power
                        current_tests$type <- current_type
                        current_tests$scaling <- current_scaling
                        current_tests$scaling.base <- current_subset$scaling.base[1]
                        current_tests$scaling.power <- current_subset$scaling.power[1]
                        current_tests$B <- B
                        test_results[[test_row]] <- current_tests
                        test_row <- test_row + 1
                    }
                }
            }
        }
    }

    list(
        predictions = do.call(rbind, prediction_results),
        tests = do.call(rbind, test_results)
    )
}

summarise_normality_tests <- function(df, alpha = 0.05) {
    df$reject <- df$p.value < alpha
    aggregate(
        cbind(statistic, p.value, reject) ~ num.obs + min.node.size + min.node.size.power + type + scaling + scaling.base + scaling.power + test,
        data = df,
        FUN = function(x) mean(x, na.rm = TRUE)
    )
}

if (isTRUE(run_normality_simulation)) {
    set.seed(2026)

    if (!dir.exists("DecisionTreePlots")) {
        dir.create("DecisionTreePlots")
    }

    normality_results <- run_asymptotic_normality_study(B, K, data_sizes, normality_x, normality_min_node_size_powers)
    df_normality_tree_predictions <- normality_results$predictions
    df_normality_tree_tests <- normality_results$tests
    df_normality_tree_tests_summary <- summarise_normality_tests(df_normality_tree_tests, normality_alpha)

    write.table(df_normality_tree_predictions, file = "DecisionTreePlots/df_normality_tree_predictions.txt", sep = "\t", row.names = FALSE)
    write.table(df_normality_tree_tests, file = "DecisionTreePlots/df_normality_tree_tests.txt", sep = "\t", row.names = FALSE)
    write.table(df_normality_tree_tests_summary, file = "DecisionTreePlots/df_normality_tree_tests_summary.txt", sep = "\t", row.names = FALSE)

    head(df_normality_tree_tests)
    head(df_normality_tree_tests_summary)
}

# plan for 3/6:
# - ask codex to augment the code by adding additional scaled and centred values where N_L is replaced by powers of k_n
# - test the code for small values (e.g. B = 10 and K = 10)
# - run the code for B = 1000 and K = 1000 (will take several hours, run code during SPS and lunch break)


if (file.exists("DecisionTreePlots/df_normality_tree_predictions.txt") &&
    file.exists("DecisionTreePlots/df_normality_tree_tests.txt") &&
    file.exists("DecisionTreePlots/df_normality_tree_tests_summary.txt")) {
    # read in data after simulating
    df_normality_tree_predictions <- read.table("DecisionTreePlots/df_normality_tree_predictions.txt", header = TRUE)
    df_normality_tree_tests <- read.table("DecisionTreePlots/df_normality_tree_tests.txt", header = TRUE)
    df_normality_tree_tests_summary <- read.table("DecisionTreePlots/df_normality_tree_tests_summary.txt", header = TRUE)

    head(df_normality_tree_predictions)
    dim(df_normality_tree_predictions)
    head(df_normality_tree_tests)
    dim(df_normality_tree_tests)
    head(df_normality_tree_tests_summary)
    dim(df_normality_tree_tests_summary)
}
