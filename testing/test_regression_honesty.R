#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

# Testing Assumption 1 + point prediction with and without honesty in general
#-------------------------------------------------------------------------------------------------

# this density on [0, 1] touches zero in exactly the point 2/3
f <- function(x) {
    9*(x - 2/3)^2
}

plot(f)

cdf <- function(x) {
    3*x^3 - 6*x^2 + 4*x
}

quantile_X <- function(p) {
  base <- 8 - 9*p
  cube_root <- sign(base) * abs(base)^(1/3)
  (2 - cube_root) / 3
}

set.seed(2026)
n <- 2000
U <- runif(n)
X <- quantile_X(U)
eps <- rnorm(n)

# the true regression functions
g <- function(x) {
    2 * x
}

Y <- g(X) + eps

test_data <- data.frame(Y = Y, X = X)

forest <- jfforest(Y ~ ., data = test_data, sample_rate = 0.4)
print_forest(forest)
forest$mse.error    # 1.037524
forest$R2           # 0.1639333

forest_honest <- jfforest(Y ~ ., data = test_data, honest = TRUE, double_bootstrap = TRUE, sample_rate = 0.8)
print_forest(forest_honest)
forest_honest$mse.error    # 1.018154
forest_honest$R2           # 0.1795427

# we use half the sample rate for the dishonest forest to get about the same average tree depth
# (15.54 for the dishonest forest and 15.558 for the honest)
# even with the same sample rate, there is about the same difference in error

jfforest.predict(forest, new_data = data.frame(X = 1/2))

# honesty with double bootstrap provides a slight improvement it seems,

# test varying sample sizes
test_values <- seq(0, 1, 0.01)
num_values <- length(test_values)
set.seed(2026)
sizes <- c(50, 100, 250, 500, 1000, 2500, 5000)
num_sizes <- length(sizes)

mse_sizes <- rep(0, num_sizes)
R2_sizes <- rep(0, num_sizes)
predicted <- matrix(rep(0, num_sizes * num_values), nrow = num_values)
mse_sizes_honest <- rep(0, num_sizes)
R2_sizes_honest <- rep(0, num_sizes)
predicted_honest <- matrix(rep(0, num_sizes * num_values), nrow = num_values)

for (i in 1:num_sizes) {
    U <- runif(sizes[i])
    X <- quantile_X(U)
    Y <- g(X) + rnorm(sizes[i])
    test_data <- data.frame(Y = Y, X = X)

    # dishonest forest
    current_forest <- jfforest(Y ~ ., data = data.frame(Y = Y, X = X))
    mse_sizes[i] <- current_forest$mse.error
    R2_sizes[i] <- current_forest$R2
    predicted[,i] <- jfforest.predict(current_forest, new_data = data.frame(X = test_values))

    # honest forest
    current_forest_honest <- jfforest(Y ~ ., data = data.frame(Y = Y, X = X), honest = TRUE, double_bootstrap = TRUE)
    mse_sizes_honest[i] <- current_forest_honest$mse.error
    R2_sizes_honest[i] <- current_forest_honest$R2
    predicted_honest[,i] <- jfforest.predict(current_forest_honest, new_data = data.frame(X = test_values))
}

# dishonest results
mse_sizes
R2_sizes

# honest results
mse_sizes_honest
R2_sizes_honest

test_values[67] # very close to 2/3, should expect wild behaviour here

predicted[67,]
predicted_honest[67,]
g(2/3)

library(tidyverse)
ggplot() +
    geom_line(aes(x = sizes, y = predicted[67,]), colour = "DarkBlue", linetype = 2) + 
    geom_line(aes(x = sizes, y = predicted_honest[67,]), colour = "DarkGreen", linetype = 1) +
    geom_abline(intercept = g(2/3), slope = 0, colour = "black") + theme_bw()

test_values[21]
predicted[21,]
predicted_honest[21,]
g(0.2)

ggplot() + 
    geom_line(aes(x = sizes, y = predicted[21,]), colour = "DarkBlue", linetype = 2) + 
    geom_line(aes(x = sizes, y = predicted_honest[21,]), colour = "DarkGreen", linetype = 1) +
    geom_abline(intercept = g(0.2), slope = 0, colour = "black") + theme_bw()

# probably impossible to infer anything here, should fit many forests to see the effect (see below)

# some plots of the regression functions
library(tidyverse)

# n = 5000
ggplot() + 
    geom_line(aes(x = test_values, y = predicted[,7]), colour = "DarkBlue", linetype = 2) + 
    geom_line(aes(x = test_values, y = predicted_honest[,7]), colour = "DarkGreen", linetype = 1) + 
    geom_function(fun = g, colour = "black", linewidth = 1) + theme_bw() + 
    xlab("Covariate value") + ylab("Predicted value")

# n = 500
ggplot() + 
    geom_line(aes(x = test_values, y = predicted[,4]), colour = "DarkBlue", linetype = 2) + 
    geom_line(aes(x = test_values, y = predicted_honest[,4]), colour = "DarkGreen", linetype = 1) + 
    geom_function(fun = g, colour = "black", linewidth = 1) + theme_bw() + 
    xlab("Covariate value") + ylab("Predicted value")

# honesty still performs better, even for the small values of n
# final point: predictions flatten out around 2/3 as is expected since trees cannot
# extrapolate outside the region of the data, and data is extremely sparse around 2/3

# try various minimal node sizes
# test varying sample sizes
set.seed(2026)
n <- 2000
U <- runif(n)
X <- quantile_X(U)
eps <- rnorm(n)

# the true regression functions
g <- function(x) {
    2 * x
}

Y <- g(X) + eps

test_data <- data.frame(Y = Y, X = X)

set.seed(2026)
node_sizes <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
num_sizes <- length(node_sizes)
n_forests <- 100

mse_sizes <- matrix(rep(0, num_sizes * n_forests), nrow = n_forests)
R2_sizes <- matrix(rep(0, num_sizes * n_forests), nrow = n_forests)
mse_sizes_honest <- matrix(rep(0, num_sizes * n_forests), nrow = n_forests)
R2_sizes_honest <- matrix(rep(0, num_sizes * n_forests), nrow = n_forests)

for (j in 1:n_forests) {
    print(j)
    for (i in 1:num_sizes) {
        # dishonest forest
        current_forest <- jfforest(Y ~ ., data = data.frame(Y = Y, X = X), min_node_size = node_sizes[i])
        mse_sizes[j, i] <- current_forest$mse.error
        R2_sizes[j, i] <- current_forest$R2
        rm(current_forest)
        gc()

        # honest forest
        current_forest_honest <- jfforest(Y ~ ., data = data.frame(Y = Y, X = X), honest = TRUE, double_bootstrap = TRUE, min_node_size = node_sizes[i])
        mse_sizes_honest[j, i] <- current_forest_honest$mse.error
        R2_sizes_honest[j, i] <- current_forest_honest$R2
        rm(current_forest_honest)
        gc()
    }
}

df_min_node_size_errors = cbind(data.frame(forest = 1:n_forests), as.data.frame(mse_sizes), as.data.frame(R2_sizes), as.data.frame(mse_sizes_honest), as.data.frame(R2_sizes_honest))

write.table(df_min_node_size_errors, file = "DecisionTreePlots/df_min_node_size_errors.txt", sep = "\t", row.names = FALSE)

# already saved, no need to run again
head(read.table("DecisionTreePlots/df_min_node_size_errors.txt"))

df_min_node_size_errors <- colMeans(as.data.frame(read.table("DecisionTreePlots/df_min_node_size_errors.txt", header = TRUE))[, -1])
node_sizes <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
mse_sizes <- as.numeric(df_min_node_size_errors[1:9])
R2_sizes <- as.numeric(df_min_node_size_errors[10:18])
mse_sizes_honest <- as.numeric(df_min_node_size_errors[19:27])
R2_sizes_honest <- as.numeric(df_min_node_size_errors[28:36])

library(tidyverse)
min_node_size_plot_mse <- ggplot() + 
    geom_line(aes(x = node_sizes, y = mse_sizes, colour = "MSE (Dishonest)", linetype = "MSE (Dishonest)")) + 
    geom_line(aes(x = node_sizes, y = mse_sizes_honest, colour = "MSE (Honest)", linetype = "MSE (Honest)")) + 
    scale_x_continuous(breaks = node_sizes, trans = "log2") +
    xlab("Minimal node size") + ylab("MSE") + theme_bw() + 
    scale_colour_manual(name = "Type", values = c("MSE (Dishonest)" = "DarkBlue", "MSE (Honest)" = "DarkGreen")) +
    scale_linetype_manual(name = "Type", values = c("MSE (Dishonest)" = 2, "MSE (Honest)" = 1)) +
    theme(legend.position = "bottom")
min_node_size_plot_mse

ggsave("DecisionTreePlots/min_node_size_plot_mse.png", units = "cm", width = 14, height = 10) 

min_node_size_plot_R2 <- ggplot() + 
    geom_line(aes(x = node_sizes, y = R2_sizes, colour = "Dishonest", linetype = "Dishonest")) + 
    geom_line(aes(x = node_sizes, y = R2_sizes_honest, colour = "Honest", linetype = "Honest")) + 
    scale_x_continuous(breaks = node_sizes, trans = "log2") +
    xlab("Minimal node size") + ylab(expression(R^2)) + theme_bw() + 
    scale_colour_manual(
    name = "Type",
    breaks = c("Dishonest", "Honest"),
    labels = c(expression(R^2 ~ "(Dishonest)"),
               expression(R^2 ~ "(Honest)")),
    values = c(
      "Dishonest" = "DarkBlue",
      "Honest" = "DarkGreen"
    )
  ) +
  scale_linetype_manual(
    name = "Type",
    breaks = c("Dishonest", "Honest"),
    labels = c(expression(R^2 ~ "(Dishonest)"),
               expression(R^2 ~ "(Honest)")),
    values = c(
      "Dishonest" = 2,
      "Honest" = 1
    )
  ) + theme(legend.position = "bottom")
min_node_size_plot_R2

ggsave("DecisionTreePlots/min_node_size_plot_R2.png", units = "cm", width = 14, height = 10) 

# now try to predict the regression function in its entirety across many runs (and also compute MSE and R^2)

set.seed(2026)
test_grid <- seq(0, 1, 0.05)
test_values <- data.frame(X = test_grid)
n_simulations <- 100
sizes <- c(50, 100, 250, 500, 1000, 2500, 5000)
num_sizes <- length(sizes)
num_test_values <- length(test_grid)

mse_sizes <- rep(0, num_sizes)
R2_sizes <- rep(0, num_sizes)
mse_sizes_honest <- rep(0, num_sizes)
R2_sizes_honest <- rep(0, num_sizes)
predictions <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)
predictions_honest <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)
var_sizes <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)
var_sizes_honest <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)

for (i in 1:num_sizes) {
    # define quantities for this size particularly
    mse_sizes_temp <- rep(0, n_simulations)
    R2_sizes_temp <- rep(0, n_simulations)
    mse_sizes_honest_temp <- rep(0, n_simulations)
    R2_sizes_honest_temp <- rep(0, n_simulations)
    predictions_temp <- matrix(rep(0, n_simulations * num_test_values), nrow = n_simulations)
    predictions_honest_temp <- matrix(rep(0, n_simulations * num_test_values), nrow = n_simulations)

    for (j in 1:n_simulations) {
        cat("Data size", sizes[i], "simulation", j, "\n")
        U <- runif(sizes[i])
        X <- quantile_X(U)
        Y <- g(X) + rnorm(sizes[i])
        test_data <- data.frame(Y = Y, X = X)

        # dishonest forest
        current_forest <- jfforest(Y ~ ., data = test_data, ntrees = 500)
        mse_sizes_temp[j] <- current_forest$mse.error
        R2_sizes_temp[j] <- current_forest$R2
        predictions_temp[j, ] <- jfforest.predict(current_forest, new_data = test_values)
        remove(current_forest)
        gc()

        # honest forest
        current_forest_honest <- jfforest(Y ~ ., data = test_data, honest = TRUE, double_bootstrap = TRUE, ntrees = 500)
        mse_sizes_honest_temp[j] <- current_forest_honest$mse.error
        R2_sizes_honest_temp[j] <- current_forest_honest$R2
        predictions_honest_temp[j, ] <- jfforest.predict(current_forest_honest, new_data = test_values)
        remove(current_forest_honest)
        gc()
    }

    # now average and save
    mse_sizes[i] <- mean(mse_sizes_temp)
    R2_sizes[i] <- mean(R2_sizes_temp)
    mse_sizes_honest[i] <- mean(mse_sizes_honest_temp)
    R2_sizes_honest[i] <- mean(R2_sizes_honest_temp)
    predictions[i, ] <- colMeans(predictions_temp)
    predictions_honest[i, ] <- colMeans(predictions_honest_temp)
    var_sizes[i, ] <- apply(predictions_temp, 2, var)
    var_sizes_honest[i, ] <- apply(predictions_honest_temp, 2, var)
}

df_data_size_errors <- data.frame(num.obs = sizes, mse = mse_sizes, R2 = R2_sizes, mse.honest = mse_sizes_honest, R2.honest = R2_sizes_honest)
df_data_size_variances <- cbind(data.frame(num.obs = sizes), as.data.frame(var_sizes), as.data.frame(var_sizes_honest))
df_data_size_predictions <- cbind(data.frame(num.obs = sizes), as.data.frame(predictions), as.data.frame(predictions_honest))

write.table(df_data_size_errors, file = "DecisionTreePlots/data_size_errors.txt", sep = "\t", row.names = FALSE)
write.table(df_data_size_variances, file = "DecisionTreePlots/df_data_size_variances.txt", sep = "\t", row.names = FALSE)
write.table(df_data_size_predictions, file = "DecisionTreePlots/df_data_size_predictions.txt", sep = "\t", row.names = FALSE)

# no need to run the above loop for plotting, simply read in the data here
df_data_size_errors <- as.data.frame(read.table("DecisionTreePlots/data_size_errors.txt", header = TRUE))

library(tidyverse)
n_mse_plot <- ggplot(data = df_data_size_errors) +
    geom_line(aes(x = num.obs, y = mse, colour = "MSE (Dishonest)", linetype = "MSE (Dishonest)")) +
    geom_line(aes(x = num.obs, y = mse.honest, colour = "MSE (Honest)", linetype = "MSE (Honest)")) +
    scale_x_continuous(breaks = df_data_size_errors$num.obs, trans = "log") +
    xlab("Size of dataset") + ylab("MSE") + theme_bw() +
    scale_colour_manual(name = "Type", values = c("MSE (Dishonest)" = "DarkBlue", "MSE (Honest)" = "DarkGreen")) +
    scale_linetype_manual(name = "Type", values = c("MSE (Dishonest)" = 2, "MSE (Honest)" = 1)) +
    theme(legend.position = "bottom")
n_mse_plot

ggsave("DecisionTreePlots/n_mse_plot.png", units = "cm", width = 14, height = 10) 

n_R2_plot <- ggplot(df_data_size_errors) + 
    geom_line(aes(x = num.obs, y = R2, colour = "Dishonest", linetype = "Dishonest")) + 
    geom_line(aes(x = num.obs, y = R2.honest, colour = "Honest", linetype = "Honest")) + 
    scale_x_continuous(breaks = df_data_size_errors$num.obs, trans = "log") +
    xlab("Size of dataset") + ylab(expression(R^2)) + theme_bw() + 
    scale_colour_manual(
    name = "Type",
    breaks = c("Dishonest", "Honest"),
    labels = c(expression(R^2 ~ "(Dishonest)"),
               expression(R^2 ~ "(Honest)")),
    values = c(
      "Dishonest" = "DarkBlue",
      "Honest" = "DarkGreen"
    )
  ) +
  scale_linetype_manual(
    name = "Type",
    breaks = c("Dishonest", "Honest"),
    labels = c(expression(R^2 ~ "(Dishonest)"),
               expression(R^2 ~ "(Honest)")),
    values = c(
      "Dishonest" = 2,
      "Honest" = 1
    )
  ) + theme(legend.position = "bottom")
n_R2_plot

ggsave("DecisionTreePlots/n_R2_plot.png", units = "cm", width = 14, height = 10) 

# now plot the predictions with specific focus on the x = 2/3 covariate
test_values <- seq(0, 1, 0.05)
num_test_values <- length(test_values)
dishonest_prediction_cols <- 2:(num_test_values + 1)
honest_prediction_cols <- (num_test_values + 2):(2 * num_test_values + 1)
test_values[14]    # closest to 2/3
g <- function(x) {
    2 * x
}

# two vectors of predictions per row (dishonest, then honest)
df_data_size_predictions <- as.data.frame(read.table("DecisionTreePlots/df_data_size_predictions.txt", header = TRUE))
head(df_data_size_predictions)
dim(df_data_size_predictions)
df_data_size_predictions[,1]

as.numeric(df_data_size_predictions[1, dishonest_prediction_cols])
test_values

library(tidyverse)
# row = 1: 50, row = 2: 100 etc. num_obs = (50  100  250  500 1000 2500 5000)
plot_regression_lines <- function(row) {
    ggplot() +
    geom_line(aes(x = test_values, y = as.numeric(df_data_size_predictions[row, dishonest_prediction_cols]), colour = "Dishonest", linetype = "Dishonest"), linewidth = 1.3) +
    geom_line(aes(x = test_values, y = as.numeric(df_data_size_predictions[row, honest_prediction_cols]), colour = "Honest", linetype = "Honest"), linewidth = 1.3) +
    geom_function(fun = g, linewidth = 1.3) + xlab("Covariate") + ylab("Regression function") + theme_bw() +
    scale_colour_manual(name = "Type", values = c("Dishonest" = "DarkBlue", "Honest" = "DarkGreen")) +
    scale_linetype_manual(name = "Type", values = c("Dishonest" = 2, "Honest" = 1)) + theme(legend.position = "bottom")
}

plot_regression_lines(7)

# now consider bias-variance decomposition plots
df_data_size_variances <- as.data.frame(read.table("DecisionTreePlots/df_data_size_variances.txt", header = TRUE))
variance_start_col <- if ("num.obs" %in% names(df_data_size_variances)) 2 else 1
dishonest_variance_cols <- variance_start_col:(variance_start_col + num_test_values - 1)
honest_variance_cols <- (variance_start_col + num_test_values):(variance_start_col + 2 * num_test_values - 1)

# compute the squared bias
biases_squared <- df_data_size_predictions[, dishonest_prediction_cols]
biases_squared_honest <- df_data_size_predictions[, honest_prediction_cols]
for (i in 1:nrow(biases_squared)) {
    biases_squared[i, ] <- (biases_squared[i, ] - g(test_values))^2
    biases_squared_honest[i, ] <- (biases_squared_honest[i, ] - g(test_values))^2
}

biases_squared
# note: variance of the noise is sigma^2 = 1
# bias-variance decomposition is MSE = bias^2 + var(f_hat) + sigma^2, so lowest possible MSE is 1

noise_variance <- 1
mse_plot <- function(row) {
    dishonest_variance <- as.numeric(df_data_size_variances[row, dishonest_variance_cols])
    dishonest_bias_squared <- as.numeric(biases_squared[row, ])
    dishonest_sum <- dishonest_bias_squared + dishonest_variance + noise_variance

    honest_variance <- as.numeric(df_data_size_variances[row, honest_variance_cols])
    honest_bias_squared <- as.numeric(biases_squared_honest[row, ])
    honest_sum <- honest_bias_squared + honest_variance + noise_variance

    ggplot() +
        geom_line(aes(x = test_values, y = dishonest_bias_squared, colour = "Dishonest", linetype = "Bias squared"), linewidth = 1.3) +
        geom_line(aes(x = test_values, y = dishonest_variance, colour = "Dishonest", linetype = "Variance"), linewidth = 1.3) +
        geom_line(aes(x = test_values, y = dishonest_sum, colour = "Dishonest", linetype = "Sum"), linewidth = 1.3) +
        geom_line(aes(x = test_values, y = honest_bias_squared, colour = "Honest", linetype = "Bias squared"), linewidth = 1.3) +
        geom_line(aes(x = test_values, y = honest_variance, colour = "Honest", linetype = "Variance"), linewidth = 1.3) +
        geom_line(aes(x = test_values, y = honest_sum, colour = "Honest", linetype = "Sum"), linewidth = 1.3) +
        scale_colour_manual(
            name = "Type",
            values = c("Dishonest" = "DarkBlue", "Honest" = "DarkGreen")
        ) +
        scale_linetype_manual(
            name = "Decomposition",
            values = c("Bias squared" = 1, "Variance" = 2, "Sum" = 3)
        ) +
        xlab("Covariate") + ylab("Error contribution") + theme_bw() +
        theme(legend.position = "bottom")
}

# maybe not that enlightening
mse_plot(3)

# just for good measure, try with another regression function
set.seed(2026)
n <- 2500
U <- runif(n)
X <- quantile_X(U)
eps <- rnorm(n)

# the true regression functions
g <- function(x) {
    3*sin(2* pi * x)
}

plot(g)

Y <- g(X) + eps

test_data <- data.frame(Y = Y, X = X)
forest <- jfforest(Y ~ ., data = test_data)
forest_honest <- jfforest(Y ~ ., data = test_data, honest = TRUE, double_bootstrap = TRUE)

forest$mse.error        # 1.074213 
forest$R2               # 0.6411498
forest_honest$mse.error # 1.006416
forest_honest$R2        # 0.6637979

library(tidyverse)
ggplot() + 
    geom_line(aes(x = test_values, y = jfforest.predict(forest, data.frame(X = test_values))), colour = "DarkBlue", linetype = 2) + 
    geom_line(aes(x = test_values, y = jfforest.predict(forest_honest, data.frame(X = test_values))), colour = "DarkGreen", linetype = 1) + 
    geom_function(fun = g, colour = "black", linewidth = 1) + theme_bw() + 
    xlab("Covariate value") + ylab("Predicted value")

# same conclusion as before, honesty with double bootstrap seems to work a lot better

# honesty MSE/R^2 compared to dishonesty for various datasets (should not include)
#-------------------------------------------------------------------------------------------------

data("BostonHousing", package = "mlbench")
data("CO2", package = "datasets")
data("Ozone", package = "mlbench")
Ozone <- na.omit(Ozone)
data("HousePrices", package = "AER")

# other potential datasets
data("CASchools", package = "AER")

set.seed(2026)
rfsrc(medv ~ ., data = BostonHousing)
forest_BostonHousing <- jfforest(medv ~ ., data = BostonHousing, min_node_size = 5)
forest_BostonHousing_honest <- jfforest(medv ~ ., data = BostonHousing, honest = TRUE, double_bootstrap = TRUE, min_node_size = 5)

print_forest(forest_BostonHousing_honest)

forest_BostonHousing$mse.error          # 12.99228
forest_BostonHousing_honest$mse.error   # 17.04466
forest_BostonHousing$R2                 # 0.8460987
forest_BostonHousing_honest$R2          # 0.7980958

# conclusion: honesty is slightly worse for BostonHousing

set.seed(2026)
rfsrc(uptake ~ ., data = CO2, min_node_size = 5)
# note: much better with min_node_size = 3
forest_CO2 <- jfforest(uptake ~ ., data = CO2, min_node_size = 3)
forest_CO2_honest <- jfforest(uptake ~ ., data = CO2, honest = TRUE, double_bootstrap = TRUE, min_node_size = 3)

print_forest(forest_CO2_honest)

forest_CO2$mse.error           # 11.20203
forest_CO2$R2                  # 0.9030625
forest_CO2_honest$mse.error    # 23.19975
forest_CO2_honest$R2           # 0.7992393

# conclusion: honesty is quite a lot worse for CO2, but this is also a very small dataset

set.seed(2026)
rfsrc(V4 ~ ., data = Ozone)
forest_Ozone <- jfforest(V4 ~ ., data = Ozone)
forest_Ozone_honest <- jfforest(V4 ~ ., data = Ozone, honest = TRUE, double_bootstrap = TRUE)

forest_Ozone$mse.error          # 17.64241
forest_Ozone$R2                 # 0.7356805
forest_Ozone_honest$mse.error   # 20.07164
forest_Ozone_honest$R2          # 0.6992856

# honesty is worse when the holdout set is not OOB

set.seed(2026)
rfsrc(price ~ ., data = HousePrices)
forest_HousePrices <- jfforest(price ~ ., data = HousePrices)
forest_HousePrices_honest <- jfforest(price ~ ., data = HousePrices, honest = TRUE, double_bootstrap = TRUE)

# rfsrc MSE:                              263183422.699777
# rfsrc R^2                               0.63089568
forest_HousePrices$mse.error            # 268634483
forest_HousePrices$R2                   # 0.6225595
forest_HousePrices_honest$mse.error     # 290063016
forest_HousePrices_honest$R2            # 0.5924517

# again honesty is worse when the holdout set is not OOB

# Testing asymptotic normality
#-------------------------------------------------------------------------------------------------
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

get_node_size <- function(tree, x) {
    getNodeSize(tree, x)
}

# here we need a density satisfying Assumption 1, so start with Unif[0, 1]^d for simplicity
g <- function(x) {
    2 * x[1] + 3 * sin(x[2])
}

# testing normality when the true data-generating distribution is known
n <- 2500
B <- 2000
x <- c(1/2, 1/2, 1/2, 1/2)
new_data = data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
result <- rep(0, B)
result_honest <- rep(0, B)
(k_n <- sqrt(n))

set.seed(2026)
for (b in 1:B) {
    # simulate data
    X1 <- runif(n)
    X2 <- runif(n)
    X3 <- runif(n)  # noise
    X4 <- runif(n)  # noise
    Y <- g(c(X1, X2, X3, X4)) + rnorm(n)
    train_data <- data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3, X4 = X4)

    cat("Fitting tree", b, "\n")
    tree <- jftree(Y ~ X1 + X2, data = train_data, min_node_size = k_n)
    tree_honest <- jftree(Y ~ X1 + X2, data = train_data, honest = TRUE, min_node_size = k_n)
    n_L <- get_node_size(tree, x)
    n_L_honest <- get_node_size(tree_honest, x)
    result[b] <- sqrt(n_L) * (jftree.predict(tree, new_data = new_data) - g(x))
    result_honest[b] <- sqrt(n_L_honest) * (jftree.predict(tree_honest, new_data = new_data) - g(x))
}

# results for when data is resampled in each loop
df_normality_resim <- data.frame(N = result, N.honest = result_honest)
write.table(df_normality_resim, file = "DecisionTreePlots/df_normality_resim.txt", sep = "\t", row.names = TRUE)

# plots
df_normality_resim <- read.table("DecisionTreePlots/df_normality_resim.txt", header = TRUE)
head(df_normality_resim)

qqnorm(df_normality_resim$N)
qqline(df_normality_resim$N)

qqnorm(df_normality_resim$N.honest)
qqline(df_normality_resim$N.honest)

# findings:
# for standard terminal node size (5), not asymptotically normal (around the true regression function)
# for k_n = sqrt(n), still not asymptotically normal

# maybe just an asymptotic bias? or it doesn't hold for a single tree?
# or it needs to be on the same dataset for each tree?

# try with same dataset for each tree
n <- 2500
B <- 2000
x <- c(1/2, 1/2, 1/2, 1/2)
new_data = data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
result <- rep(0, B)
result_honest <- rep(0, B)
(k_n <- sqrt(n))

# simulate data
set.seed(2026)
X1 <- runif(n)
X2 <- runif(n)
X3 <- runif(n)  # noise
X4 <- runif(n)  # noise
Y <- g(c(X1, X2, X3, X4)) + rnorm(n)
train_data <- data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3, X4 = X4)

for (b in 1:B) {
    cat("Fitting tree", b, "\n")
    tree <- jftree(Y ~ X1 + X2, data = train_data, min_node_size = k_n)
    tree_honest <- jftree(Y ~ X1 + X2, data = train_data, honest = TRUE, min_node_size = k_n)
    n_L <- get_node_size(tree, x)
    n_L_honest <- get_node_size(tree_honest, x)
    result[b] <- sqrt(n_L) * (jftree.predict(tree, new_data = new_data) - g(x))
    result_honest[b] <- sqrt(n_L_honest) * (jftree.predict(tree_honest, new_data = new_data) - g(x))
}

df_normality_same_data <- data.frame(N = result, N.honest = result_honest)
write.table(df_normality_same_data, file = "DecisionTreePlots/df_normality_same_data.txt", sep = "\t", row.names = TRUE)

# plots
df_normality_same_data <- read.table("DecisionTreePlots/df_normality_same_data.txt", header = TRUE)
head(df_normality_same_data)

qqnorm(df_normality_same_data$N)
qqline(df_normality_same_data$N)

# looks normal for honest trees, no? but definitely not centred around zero
qqnorm(df_normality_same_data$N.honest)
qqline(df_normality_same_data$N.honest)

hist(df_normality_same_data$N.honest, breaks = 50, prob = TRUE, xlab = "Honest samples", col = "gray")
lines(density(df_normality_same_data$N.honest), lwd = 2, col = "DarkBlue")
curve(dnorm(x, mean=mean(df_normality_same_data$N.honest), sd=sd(df_normality_same_data$N.honest)), 
      col="DarkGreen", lwd=2, add=TRUE)

# how does the bias change in n? 

data_sizes <- c(50, 100, 250, 500, 1000, 2000, 5000, 10000)
num_sizes <- length(data_sizes)
B <- 2000
x <- c(1/2, 1/2, 1/2, 1/2)
new_data = data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
result <- matrix(rep(0, B * num_sizes), nrow = B)
result_honest <- matrix(rep(0, B * num_sizes), nrow = B)

set.seed(2026)
for (j in 1:num_sizes) {
    n <- data_sizes[j]
    X1 <- runif(n)
    X2 <- runif(n)
    X3 <- runif(n)  # noise
    X4 <- runif(n)  # noise
    Y <- g(c(X1, X2, X3, X4)) + rnorm(n)
    train_data <- data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3, X4 = X4)
    k_n <- sqrt(n)
    
    # now fit B trees on this data
    for (b in 1:B) {
        cat("Fitting tree", b, "on data of size", n, "\n")
        tree <- jftree(Y ~ X1 + X2, data = train_data, min_node_size = k_n)
        tree_honest <- jftree(Y ~ X1 + X2, data = train_data, honest = TRUE, min_node_size = k_n)
        n_L <- get_node_size(tree, x)
        n_L_honest <- get_node_size(tree_honest, x)
        result[b, j] <- sqrt(n_L) * (jftree.predict(tree, new_data = new_data) - g(x))
        result_honest[b, j] <- sqrt(n_L_honest) * (jftree.predict(tree_honest, new_data = new_data) - g(x))
    }
}

head(result)
head(result_honest)

df_normality_data_sizes <- data.frame(N_50 = result[,1], N_50_honest = result_honest[,1],
                                      N_100 = result[,2], N_100_honest = result_honest[,2],
                                      N_250 = result[,3], N_250_honest = result_honest[,3],
                                      N_500 = result[,4], N_500_honest = result_honest[,4],
                                      N_1000 = result[,5], N_1000_honest = result_honest[,5],
                                      N_2000 = result[,6], N_2000_honest = result_honest[,6],
                                      N_5000 = result[,7], N_5000_honest = result_honest[,7],
                                      N_10000 = result[,8], N_10000_honest = result_honest[,8])

write.table(df_normality_data_sizes, file = "DecisionTreePlots/df_normality_data_sizes.txt", sep = "\t", row.names = TRUE)

# plots
df_normality_data_sizes <- read.table("DecisionTreePlots/df_normality_data_sizes.txt", header = TRUE)
head(df_normality_data_sizes)

# dishonest trees
qqnorm(df_normality_data_sizes$N_50)
qqline(df_normality_data_sizes$N_50)
qqnorm(df_normality_data_sizes$N_100)
qqline(df_normality_data_sizes$N_100)
qqnorm(df_normality_data_sizes$N_250)
qqline(df_normality_data_sizes$N_250)
# no

# honest trees
normal_plot <- function(vec) {
    hist(vec, breaks = 50, prob = TRUE, xlab = "Samples", col = "gray")
    lines(density(vec), lwd = 2, col = "DarkBlue")
    curve(dnorm(x, mean=mean(vec), sd=sd(vec)), 
        col="DarkGreen", lwd=2, add=TRUE)
}

qqnorm(df_normality_data_sizes$N_50_honest)
qqline(df_normality_data_sizes$N_50_honest)
normal_plot(df_normality_data_sizes$N_50_honest)
mean(df_normality_data_sizes$N_50_honest)   # 2.089832

qqnorm(df_normality_data_sizes$N_100_honest)
qqline(df_normality_data_sizes$N_100_honest)
normal_plot(df_normality_data_sizes$N_100_honest)
mean(df_normality_data_sizes$N_100_honest)  # -1.315116

qqnorm(df_normality_data_sizes$N_250_honest)
qqline(df_normality_data_sizes$N_250_honest)
normal_plot(df_normality_data_sizes$N_250_honest)
mean(df_normality_data_sizes$N_250_honest)  # -1.797804

qqnorm(df_normality_data_sizes$N_500_honest)
qqline(df_normality_data_sizes$N_500_honest)
normal_plot(df_normality_data_sizes$N_500_honest)
mean(df_normality_data_sizes$N_500_honest)  # -11.48705

# judging from the QQ plots, normality has not set in just yet

# better here
qqnorm(df_normality_data_sizes$N_1000_honest)
qqline(df_normality_data_sizes$N_1000_honest)
normal_plot(df_normality_data_sizes$N_1000_honest)
mean(df_normality_data_sizes$N_1000_honest)  # -0.6587944

# quite good here
qqnorm(df_normality_data_sizes$N_2000_honest)
qqline(df_normality_data_sizes$N_2000_honest)
normal_plot(df_normality_data_sizes$N_2000_honest)
mean(df_normality_data_sizes$N_2000_honest)  # -5.454237

qqnorm(df_normality_data_sizes$N_5000_honest)
qqline(df_normality_data_sizes$N_5000_honest)
normal_plot(df_normality_data_sizes$N_5000_honest)
mean(df_normality_data_sizes$N_5000_honest)  # 5.67846

# now it gets worse?
qqnorm(df_normality_data_sizes$N_10000_honest)
qqline(df_normality_data_sizes$N_10000_honest)
normal_plot(df_normality_data_sizes$N_10000_honest)
mean(df_normality_data_sizes$N_10000_honest)  # 12.59814

# try without honesty
qqnorm(df_normality_data_sizes$N_10000)
qqline(df_normality_data_sizes$N_10000)
normal_plot(df_normality_data_sizes$N_10000)
mean(df_normality_data_sizes$N_10000)  # 12.94885

# not sure if we can even conclude anything, honesty initially seemed somewhat promising,
# ignoring the growing bias, but since normality "fades" for larger n, I am not sure it
# is even worth pursuing further for single trees

# last ditch effort with whole forests: I expect that the effective sample size is on the
# order ntrees * k_n, and we choose the subsampling size s_n = n^(1 - 1/(1 + d/4))

devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

n <- 2000
num_forests <- 1000
x <- c(1/2, 1/2, 1/2, 1/2)
new_data = data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
result <- rep(0, B)
result_honest <- rep(0, B)
(k_n <- sqrt(n))
#(s_n <- sqrt(n))
#(sample_rate <- s_n/n)  # 0.02236068 (very very small, experiment with more reasonable sample sizes)
ntrees <- 500

# simulate data
set.seed(2026)
X1 <- runif(n)
X2 <- runif(n)
X3 <- runif(n)  # noise
X4 <- runif(n)  # noise
g <- function(x) {
    2 * x[1] + 3 * sin(x[2])
}
Y <- g(c(X1, X2, X3, X4)) + rnorm(n)
train_data <- data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3, X4 = X4)

pred <- rep(0, num_forests)
pred_honest <- rep(0, num_forests)

for (i in 1:num_forests) {
    cat("Fitting forest", i, "\n")
    forest <- jfforest(Y ~ X1 + X2, data = train_data, min_node_size = k_n, ntrees = ntrees, sample_rate = 0.2)
    pred[i] <- jfforest.predict(forest, new_data = new_data)

    forest_honest <- jfforest(Y ~ X1 + X2, data = train_data, min_node_size = k_n, ntrees = ntrees, 
                              sample_rate = 0.4, honest = TRUE, double_bootstrap = TRUE)
    pred_honest[i] <- jfforest.predict(forest_honest, new_data = new_data)

    # free memory
    rm(forest, forest_honest)
    gc()
}

normalised <- sqrt(k_n) * (pred - g(x))
normalised_honest <- sqrt(k_n) * (pred_honest - g(x))

df_forest_normality <- data.frame(N = normalised, N.honest = normalised_honest)
write.table(df_forest_normality, "DecisionTreePlots/df_forest_normality.txt", sep = "\t", row.names = TRUE)

df_forest_normality <- read.table("DecisionTreePlots/df_forest_normality.txt", header = TRUE)

normal_plot <- function(vec) {
    hist(vec, breaks = 30, prob = TRUE, xlab = "Samples", col = "gray")
    lines(density(vec), lwd = 2, col = "DarkBlue")
    curve(dnorm(x, mean=mean(vec), sd=sd(vec)), 
        col="DarkGreen", lwd=2, add=TRUE)
}

qqnorm(df_forest_normality$N)
qqline(df_forest_normality$N)
normal_plot(df_forest_normality$N)
mean(df_forest_normality$N)    # 3.788769
var(df_forest_normality$N)     # 0.002633281

qqnorm(df_forest_normality$N.honest)
qqline(df_forest_normality$N.honest)
normal_plot(df_forest_normality$N.honest)
mean(df_forest_normality$N.honest)     # 3.740768
var(df_forest_normality$N.honest)      # 0.001369785

# looks quite good, but how does it change with n?
data_sizes <- c(50, 100, 250, 500, 1000, 2000, 5000, 10000)
num_sizes <- length(data_sizes)
num_forests <- 1000
x <- c(1/2, 1/2, 1/2, 1/2)
new_data = data.frame(X1 = x[1], X2 = x[2], X3 = x[3], X4 = x[4])
pred <- matrix(rep(0, num_forests * num_sizes), nrow = num_forests)
pred_honest <- matrix(rep(0, num_forests * num_sizes), nrow = num_forests)

# this takes at least two hours to run
set.seed(2026)
for (j in 1:num_sizes) {
    n <- data_sizes[j]
    X1 <- runif(n)
    X2 <- runif(n)
    X3 <- runif(n)  # noise
    X4 <- runif(n)  # noise
    Y <- g(c(X1, X2, X3, X4)) + rnorm(n)
    train_data <- data.frame(Y = Y, X1 = X1, X2 = X2, X3 = X3, X4 = X4)
    k_n <- sqrt(n)
    
    # now fit B trees on this data
    for (i in 1:num_forests) {
        cat("Fitting forest", i, "on data of size", n, "\n")
        forest <- jfforest(Y ~ X1 + X2, data = train_data, min_node_size = k_n, sample_rate = 0.2, ntrees = ntrees)
        pred[i, j] <- jfforest.predict(forest, new_data = new_data)

        forest_honest <- jfforest(Y ~ X1 + X2, data = train_data, min_node_size = k_n, ntrees = ntrees, 
                              sample_rate = 0.4, honest = TRUE, double_bootstrap = TRUE)
        pred_honest[i, j] <- jfforest.predict(forest_honest, new_data = new_data)

        # free memory
        rm(forest, forest_honest)
        gc()
    }
}

df_forest_normality_sizes <- data.frame(forest = 1:num_forests, N.50 = pred[,1], N.100 = pred[,2], N.250 = pred[,3],
                                        N.500 = pred[,4], N.1000 = pred[,5], N.2000 = pred[,6], N.5000 = pred[,7],
                                        N.10000 = pred[,8], N.50.honest = pred_honest[,1], N.100.honest = pred_honest[,2],
                                        N.250.honest = pred_honest[,3], N.500.honest = pred_honest[,4], 
                                        N.1000.honest = pred_honest[,5], N.2000.honest = pred_honest[,6], N.5000.honest = pred_honest[,7],
                                        N.10000.honest = pred_honest[,8])

write.table(df_forest_normality_sizes, "DecisionTreePlots/df_forest_normality_sizes.png", sep = "\t", row.names = TRUE)

# plots
df_forest_normality_sizes <- read.table("DecisionTreePlots/df_forest_normality_sizes.png", header = TRUE)
head(df_forest_normality_sizes)

normalise <- function(vec, g_val, n) {
    sqrt(n) * (vec - g_val)
}

qq_plot <- function(pred_normalised) {
    qqnorm(pred_normalised)
    qqline(pred_normalised)
    print(mean(pred_normalised))
    print(var(pred_normalised))
}

# n = 50
pred_normalised_50 <- normalise(df_forest_normality_sizes$N.50, g(x), 50)
qq_plot(pred_normalised_50)
normal_plot(pred_normalised_50)
# mean = 0.935132
# var = 0.005425842

# n = 100
pred_normalised_100 <- normalise(df_forest_normality_sizes$N.100, g(x), 100)
qq_plot(pred_normalised_100)
normal_plot(pred_normalised_100)
# mean = -16.86048
# var = 0.01458967

# n = 250
pred_normalised_250 <- normalise(df_forest_normality_sizes$N.250, g(x), 250)
qq_plot(pred_normalised_250)
normal_plot(pred_normalised_250)
# mean = 8.245285
# var = 0.03343723

# n = 500 (becomes slightly less normal?)
pred_normalised_500 <- normalise(df_forest_normality_sizes$N.500, g(x), 500)
qq_plot(pred_normalised_500)
normal_plot(pred_normalised_500)
# mean = -12.18641
# var = 0.05482315

# n = 1000, okay becomes approximately normal again
pred_normalised_1000 <- normalise(df_forest_normality_sizes$N.1000, g(x), 1000)
qq_plot(pred_normalised_1000)
normal_plot(pred_normalised_1000)
# mean = -18.07773
# var = 0.09566616

# n = 2000
pred_normalised_2000 <- normalise(df_forest_normality_sizes$N.2000, g(x), 2000)
qq_plot(pred_normalised_2000)
normal_plot(pred_normalised_2000)
# mean = -45.189
# var = 0.1351436

# n = 5000, looks very nicely normal
pred_normalised_5000 <- normalise(df_forest_normality_sizes$N.5000, g(x), 5000)
qq_plot(pred_normalised_5000)
normal_plot(pred_normalised_5000)
# 68.69215
# 0.2029284

# n = 10000
pred_normalised_10000 <- normalise(df_forest_normality_sizes$N.10000, g(x), 10000)
qq_plot(pred_normalised_10000)
normal_plot(pred_normalised_10000)
# 5.770484
# 0.238531

# conclusion: the forest seems to become asymptotically normal, even for very small data samples, but
# the bias grows in n (except for n = 10000, but up until then a clear tendency), the variance also grows
# in n but slowly. overall the forest seems to be asymptotically normal, but not unbiased with the scaling
# factor sqrt(k_n)
