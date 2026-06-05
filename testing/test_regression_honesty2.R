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
    run_normality_simulation <- FALSE
}

if (!exists("run_normality_postprocessing")) {
    run_normality_postprocessing <- TRUE
}

if (!exists("B")) {
    B <- 500
}

if (!exists("K")) {
    K <- 250
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

null_model_mse <- function() {
    mean_sin_x2 <- 1 - cos(1)
    mean_sin2_x2 <- 1/2 - sin(2)/4
    var_2x1 <- 4/12
    var_3sin_x2 <- 9 * (mean_sin2_x2 - mean_sin_x2^2)

    var_2x1 + var_3sin_x2 + 1
}

make_min_node_size_colours <- function(data_sizes) {
    data_sizes <- sort(unique(data_sizes))
    dishonest_colours <- colorRampPalette(c("#8DB4E2", "DarkBlue"))(length(data_sizes))
    honest_colours <- colorRampPalette(c("#78C679", "DarkGreen"))(length(data_sizes))

    c(
        setNames(dishonest_colours, paste("Dishonest, n =", data_sizes)),
        setNames(honest_colours, paste("Honest, n =", data_sizes))
    )
}

plot_min_node_size_mse <- function(df, node_sizes) {
    df$curve <- paste(df$type, ", n = ", df$num.obs, sep = "")
    df$curve <- factor(df$curve, levels = names(make_min_node_size_colours(df$num.obs)))

    df_null_model <- data.frame(mse = null_model_mse(), model = "Null model")

    ggplot(data = df) +
        geom_hline(data = df_null_model, aes(yintercept = mse, linetype = model),
                   colour = "black", linewidth = 0.8) +
        geom_line(aes(x = min.node.size, y = mse, colour = curve,
                      group = interaction(type, num.obs)), linewidth = 1) +
        geom_point(aes(x = min.node.size, y = mse, colour = curve, shape = type,
                       group = interaction(type, num.obs)), size = 1.5) +
        scale_x_continuous(breaks = node_sizes, trans = "log2") +
        xlab("Minimal node size") + ylab("Average MSE") + theme_bw() +
        scale_colour_manual(name = "Tree and data size", values = make_min_node_size_colours(df$num.obs)) +
        scale_shape_manual(name = "Type", values = c("Dishonest" = 16, "Honest" = 17)) +
        scale_linetype_manual(name = "", values = c("Null model" = "dashed")) +
        theme(
            legend.position = "bottom",
            legend.box = "vertical",
            legend.text = element_text(size = 8),
            legend.title = element_text(size = 9),
            legend.key.width = grid::unit(1.4, "lines"),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()
        ) +
        guides(
            linetype = guide_legend(order = 1),
            colour = guide_legend(order = 2, nrow = 3, byrow = TRUE),
            shape = "none"
        )
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
    ggsave("DecisionTreePlots/min_node_size_tree_mse_plot.png", min_node_size_mse_plot, width = 11, height = 6.5, units = "in")
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

    min_node_size_mse_plot <- plot_min_node_size_mse(df_min_node_size_tree_errors_avg, node_sizes)
    ggsave("DecisionTreePlots/min_node_size_tree_mse_plot.png", min_node_size_mse_plot, width = 11, height = 6.5, units = "in")
    min_node_size_mse_plot
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

if (!exists("normality_leaf_size_power")) {
    normality_leaf_size_power <- 1/2
}

if (!exists("normality_parallel")) {
    normality_parallel <- TRUE
}

if (!exists("normality_num_cores")) {
    normality_num_cores <- max(1, parallel::detectCores() - 1)
}

if (!exists("normality_seed")) {
    normality_seed <- 2026
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

make_scaling_rows <- function(leaf_size, min_node_size, min_node_size_power, include_leaf_size = TRUE) {
    scaling_rows <- data.frame(
        scaling = paste0("sqrt(k_n), k_n = n^(", format_scaling_power(min_node_size_power), ")"),
        scaling.base = "k_n",
        scaling.power = 1/2,
        scaling.factor = sqrt(min_node_size),
        min.node.size.power = min_node_size_power
    )

    if (include_leaf_size) {
        scaling_rows <- rbind(
            data.frame(
                scaling = "sqrt(leaf.size)",
                scaling.base = "leaf.size",
                scaling.power = 1/2,
                scaling.factor = sqrt(leaf_size),
                min.node.size.power = min_node_size_power
            ),
            scaling_rows
        )
    }

    scaling_rows
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

simulate_centered_scaled_tree_predictions <- function(B, n, x, min_node_size, min_node_size_power, include_leaf_size = TRUE) {
    new_data <- data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
    true_value <- g(x)

    results <- list()
    row <- 1

    for (b in 1:B) {
        #cat("Data size", n, "normality simulation", b, "\n")
        train_data <- simulate_data(n)

        tree <- jftree(Y ~ X1 + X2, data = train_data, min_node_size = min_node_size)
        n_L <- getNodeSize(tree, x)
        prediction <- as.numeric(jftree.predict(tree, new_data = new_data))
        raw_error <- prediction - true_value
        scaling_rows <- make_scaling_rows(n_L, min_node_size, min_node_size_power, include_leaf_size)
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
        scaling_rows_honest <- make_scaling_rows(n_L_honest, min_node_size, min_node_size_power, include_leaf_size)
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

run_asymptotic_normality_task <- function(task, B, x, leaf_size_power) {
    set.seed(task$seed)

    n <- task$num.obs
    current_power <- task$min.node.size.power
    min_node_size <- task$min.node.size
    include_leaf_size <- abs(current_power - leaf_size_power) < 10^(-12)
    cat("Normality test batch", task$batch, "on data of size", n, "with minimal node size", min_node_size, "\n")

    current_predictions <- simulate_centered_scaled_tree_predictions(B, n, x, min_node_size, current_power, include_leaf_size)
    current_predictions$batch <- task$batch
    current_predictions$num.obs <- n
    current_predictions$min.node.size <- min_node_size
    current_predictions$min.node.size.power <- current_power

    test_results <- list()
    test_row <- 1
    for (current_type in c("Dishonest", "Honest")) {
        for (current_scaling in unique(current_predictions$scaling)) {
            current_subset <- current_predictions[current_predictions$type == current_type &
                                                      current_predictions$scaling == current_scaling, ]
            current_values <- current_subset$centered.scaled
            current_tests <- run_normality_tests(current_values)
            current_tests$batch <- task$batch
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

    list(
        predictions = current_predictions,
        tests = do.call(rbind, test_results)
    )
}

run_asymptotic_normality_study <- function(B, K, data_sizes, x, min_node_size_powers, leaf_size_power = 1/2,
                                           parallel = TRUE, num_cores = 1, seed = 2026) {
    tasks <- expand.grid(
        batch = 1:K,
        num.obs = data_sizes,
        min.node.size.power = min_node_size_powers,
        KEEP.OUT.ATTRS = FALSE
    )
    tasks$min.node.size <- floor(tasks$num.obs^tasks$min.node.size.power)

    set.seed(seed)
    tasks$seed <- sample.int(.Machine$integer.max, nrow(tasks))

    use_parallel <- isTRUE(parallel) && num_cores > 1 && nrow(tasks) > 1 && .Platform$OS.type != "windows"
    if (use_parallel) {
        num_cores <- min(num_cores, nrow(tasks))
        cat("Running", nrow(tasks), "normality tasks in parallel using", num_cores, "cores\n")
        task_results <- parallel::mclapply(
            X = 1:nrow(tasks),
            FUN = function(i) run_asymptotic_normality_task(tasks[i, ], B, x, leaf_size_power),
            mc.cores = num_cores,
            mc.set.seed = FALSE,
            mc.preschedule = FALSE
        )
    } else {
        cat("Running", nrow(tasks), "normality tasks serially\n")
        task_results <- lapply(
            X = 1:nrow(tasks),
            FUN = function(i) run_asymptotic_normality_task(tasks[i, ], B, x, leaf_size_power)
        )
    }

    list(
        predictions = do.call(rbind, lapply(task_results, function(result) result$predictions)),
        tests = do.call(rbind, lapply(task_results, function(result) result$tests))
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
    set.seed(normality_seed)

    if (!dir.exists("DecisionTreePlots")) {
        dir.create("DecisionTreePlots")
    }

    normality_results <- run_asymptotic_normality_study(
        B,
        K,
        data_sizes,
        normality_x,
        normality_min_node_size_powers,
        normality_leaf_size_power,
        normality_parallel,
        normality_num_cores,
        normality_seed
    )
    df_normality_tree_predictions <- normality_results$predictions
    df_normality_tree_tests <- normality_results$tests
    df_normality_tree_tests_summary <- summarise_normality_tests(df_normality_tree_tests, normality_alpha)

    write.table(df_normality_tree_predictions, file = "DecisionTreePlots/df_normality_tree_predictions.txt", sep = "\t", row.names = FALSE)
    write.table(df_normality_tree_tests, file = "DecisionTreePlots/df_normality_tree_tests.txt", sep = "\t", row.names = FALSE)
    write.table(df_normality_tree_tests_summary, file = "DecisionTreePlots/df_normality_tree_tests_summary.txt", sep = "\t", row.names = FALSE)

    head(df_normality_tree_tests)
    head(df_normality_tree_tests_summary)
}

# important note: K = 10 and B = 500 takes 40 minutes, so K = 250 and B = 500 should take
# about 17 hours (it actually took 26)

normality_plot_theme <- function() {
    theme_bw() +
        theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            legend.position = "bottom"
        )
}

make_normality_file_name <- function(prefix, label, extension = "png") {
    safe_label <- gsub("[^A-Za-z0-9]+", "_", label)
    safe_label <- gsub("^_|_$", "", safe_label)
    file.path("DecisionTreePlots", paste0(prefix, "_", safe_label, ".", extension))
}

make_p_value_qq_data <- function(df) {
    df %>%
        filter(!is.na(p.value)) %>%
        group_by(test, scaling, num.obs, type) %>%
        arrange(p.value, .by_group = TRUE) %>%
        mutate(uniform.quantile = (row_number() - 0.5) / n()) %>%
        ungroup()
}

make_p_value_qq_plot <- function(df_qq, current_test) {
    ggplot(
        df_qq %>% filter(test == current_test),
        aes(x = uniform.quantile, y = p.value, colour = type)
    ) +
        geom_abline(intercept = 0, slope = 1, colour = "grey35", linewidth = 0.6) +
        geom_point(alpha = 0.55, size = 0.8) +
        facet_grid(scaling ~ num.obs) +
        scale_colour_manual(values = c("Honest" = "DarkGreen", "Dishonest" = "DarkBlue")) +
        labs(
            title = paste(current_test, "p-value QQ plot"),
            x = "Uniform quantiles",
            y = "Observed p-values",
            colour = "Tree"
        ) +
        normality_plot_theme()
}

make_rejection_rate_tables <- function(df, alpha = 0.05) {
    df_rejection_rates <- df %>%
        mutate(reject = p.value < alpha) %>%
        group_by(num.obs, min.node.size, min.node.size.power, scaling, scaling.base, scaling.power, test, type) %>%
        summarise(
            rejection.rate = mean(reject, na.rm = TRUE),
            num.batches = sum(!is.na(reject)),
            .groups = "drop"
        ) %>%
        arrange(scaling, num.obs, test, type)

    split(df_rejection_rates, df_rejection_rates$scaling)
}

make_normality_bias_summary <- function(df) {
    df %>%
        group_by(batch, num.obs, min.node.size, min.node.size.power, scaling, scaling.base, scaling.power, type) %>%
        summarise(batch.bias = mean(centered.scaled, na.rm = TRUE), .groups = "drop") %>%
        group_by(num.obs, min.node.size, min.node.size.power, scaling, scaling.base, scaling.power, type) %>%
        summarise(avg.bias = mean(batch.bias, na.rm = TRUE), .groups = "drop") %>%
        arrange(scaling, num.obs, type)
}

make_normality_bias_plot <- function(df_bias, current_scaling) {
    ggplot(
        df_bias %>% filter(scaling == current_scaling),
        aes(
            x = num.obs,
            y = avg.bias,
            colour = type,
            linetype = factor(min.node.size.power),
            group = interaction(type, min.node.size.power)
        )
    ) +
        geom_hline(yintercept = 0, colour = "grey35", linewidth = 0.5) +
        geom_line(linewidth = 0.9) +
        geom_point(size = 2) +
        scale_x_continuous(trans = "log10", breaks = sort(unique(df_bias$num.obs))) +
        scale_colour_manual(values = c("Honest" = "DarkGreen", "Dishonest" = "DarkBlue")) +
        scale_linetype_discrete(
            labels = function(x) vapply(as.numeric(x), format_scaling_power, character(1))
        ) +
        labs(
            title = paste("Bias for", current_scaling),
            x = "Size of dataset",
            y = "Average centered/scaled bias",
            colour = "Tree",
            linetype = "Power"
        ) +
        normality_plot_theme()
}

if (isTRUE(run_normality_postprocessing) &&
    file.exists("DecisionTreePlots/df_normality_tree_predictions.txt") &&
    file.exists("DecisionTreePlots/df_normality_tree_tests.txt") &&
    file.exists("DecisionTreePlots/df_normality_tree_tests_summary.txt")) {
    df_normality_tree_predictions <- read.table("DecisionTreePlots/df_normality_tree_predictions.txt", header = TRUE)
    df_normality_tree_tests <- read.table("DecisionTreePlots/df_normality_tree_tests.txt", header = TRUE)
    df_normality_tree_tests_summary <- read.table("DecisionTreePlots/df_normality_tree_tests_summary.txt", header = TRUE)

    df_normality_p_value_qq <- make_p_value_qq_data(df_normality_tree_tests)
    normality_p_value_qq_plots <- lapply(
        sort(unique(df_normality_p_value_qq$test)),
        function(current_test) make_p_value_qq_plot(df_normality_p_value_qq, current_test)
    )
    names(normality_p_value_qq_plots) <- sort(unique(df_normality_p_value_qq$test))

    for (current_test in names(normality_p_value_qq_plots)) {
        ggsave(
            filename = make_normality_file_name("normality_p_value_qq", current_test),
            plot = normality_p_value_qq_plots[[current_test]],
            width = 13,
            height = 9,
            units = "in"
        )
    }

    normality_rejection_rate_tables <- make_rejection_rate_tables(df_normality_tree_tests, normality_alpha)
    for (current_scaling in names(normality_rejection_rate_tables)) {
        write.table(
            normality_rejection_rate_tables[[current_scaling]],
            file = make_normality_file_name("normality_rejection_rates", current_scaling, "txt"),
            sep = "\t",
            row.names = FALSE
        )
    }

    normality_bias_summary <- make_normality_bias_summary(df_normality_tree_predictions)
    normality_bias_plots <- lapply(
        sort(unique(normality_bias_summary$scaling)),
        function(current_scaling) make_normality_bias_plot(normality_bias_summary, current_scaling)
    )
    names(normality_bias_plots) <- sort(unique(normality_bias_summary$scaling))

    for (current_scaling in names(normality_bias_plots)) {
        ggsave(
            filename = make_normality_file_name("normality_bias", current_scaling),
            plot = normality_bias_plots[[current_scaling]],
            width = 8,
            height = 5.5,
            units = "in"
        )
    }
}

# some comments on the QQ plots:
# Regarding the QQ plots, there is a very natural explanation for the behaviour in the two middle rows: For honest trees, it is almost
# impossible to split at all for low sample sizes with k_n equal to a high power of n, so here the behaviour is as a sample mean
# (which is of course asymptotically normal). But after that, when splits are possible, normality breaks down completely.

# Maybe only include the QQ-plots for sqrt(k_n) when k_n = sqrt(n) and sqrt(leaf size), since these are the most comparable ones. These
# are made separately. But all bias plots are relevant. We should discuss coloring etc. I included legends and titles for now only to
# make it easier for myself. When we have decided what should be included, I will remove them.

# read in tables with rejection rates
normality_rejection_rates_sqrt_k_n_k_n_n_1_2 <- read.table("DecisionTreePlots/normality_rejection_rates_sqrt_k_n_k_n_n_1_2.txt", header = TRUE)
normality_rejection_rates_sqrt_k_n_k_n_n_2_3 <- read.table("DecisionTreePlots/normality_rejection_rates_sqrt_k_n_k_n_n_2_3.txt", header = TRUE)
normality_rejection_rates_sqrt_k_n_k_n_n_4_5 <- read.table("DecisionTreePlots/normality_rejection_rates_sqrt_k_n_k_n_n_4_5.txt", header = TRUE)
normality_rejection_rates_sqrt_leaf_size <- read.table("DecisionTreePlots/normality_rejection_rates_sqrt_leaf_size.txt", header = TRUE)
