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
forest$mse.error    # 1.088126
forest$R2           # 0.1231571

forest_honest <- jfforest(Y ~ ., data = test_data, honest = TRUE, double_bootstrap = TRUE, sample_rate = 0.8)
print_forest(forest_honest)
forest_honest$mse.error    # 0.8168693
forest_honest$R2           # 0.3417433

# we use half the sample rate for the dishonest forest to get about the same average tree depth
# (15.54 for the dishonest forest and 15.558 for the honest)
# even with the same sample rate, there is about the same difference in error

jfforest.predict(forest, new_data = data.frame(X = 1/2))

# honesty with double bootstrap provides a significant improvement it seems,

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

# change column names!
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
test_values <- data.frame(X = seq(0, 1, 0.05))
n_forests <- 100
sizes <- c(50, 100, 250, 500, 1000, 2500, 5000)
num_sizes <- length(sizes)
num_test_values <- length(seq(0, 1, 0.05))

mse_sizes <- rep(0, num_sizes)
R2_sizes <- rep(0, num_sizes)
mse_sizes_honest <- rep(0, num_sizes)
R2_sizes_honest <- rep(0, num_sizes)
predictions <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)
predictions_honest <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)
var_sizes <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)
var_sizes_honest <- matrix(rep(0, num_sizes * num_test_values), nrow = num_sizes)

for (i in 1:num_sizes) {
    U <- runif(sizes[i])
    X <- quantile_X(U)
    Y <- g(X) + rnorm(sizes[i])
    test_data <- data.frame(Y = Y, X = X)

    # define quantities for this size particularly
    mse_sizes_temp <- rep(0, n_forests)
    R2_sizes_temp <- rep(0, n_forests)
    mse_sizes_honest_temp <- rep(0, n_forests)
    R2_sizes_honest_temp <- rep(0, n_forests)
    predictions_temp <- matrix(rep(0, n_forests * num_test_values), nrow = n_forests)
    predictions_honest_temp <- matrix(rep(0, n_forests * num_test_values), nrow = n_forests)

    # now compute predictions and error metrics across all forests
    for (j in 1:n_forests) {
        print(j)
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
df_data_size_variances <- cbind(as.data.frame(var_sizes), as.data.frame(var_sizes_honest))
df_data_size_predictions <- cbind(data.frame(num.obs = sizes), as.data.frame(predictions), as.data.frame(predictions_honest))
df_squared_biases <- 

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

# two vectors of predictions per row (dishonest, then honest)
df_data_size_predictions <- as.data.frame(read.table("DecisionTreePlots/df_data_size_predictions.txt", header = TRUE))
# TODO

# now consider bias-variance decomposition plots
df_data_size_variances <- as.data.frame(read.table("DecisionTreePlots/df_data_size_variances.txt", header = TRUE))

# compute the squared bias
biases_squared <- df_data_size_predictions[, 2:22]
biases_squared_honest <- df_data_size_predictions[, 23:43]
for (i in 1:nrow(biases)) {
    biases_squared[i, ] <- (biases_squared[i, ] - g(test_values$X))^2
    biases_squared_honest[i, ] <- (biases_squared_honest[i, ] - g(test_values$X))^2
}

# note: variance of the noise is sigma^2 = 1
# bias-variance decomposition is MSE = bias^2 + var(f_hat) + sigma^2, so lowest possible MSE is 1
# TODO

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
forest_honest$mse.error # 0.8437099
forest_honest$R2        # 0.7181514

library(tidyverse)
ggplot() + 
    geom_line(aes(x = test_values, y = jfforest.predict(forest, data.frame(X = test_values))), colour = "DarkBlue", linetype = 2) + 
    geom_line(aes(x = test_values, y = jfforest.predict(forest_honest, data.frame(X = test_values))), colour = "DarkGreen", linetype = 1) + 
    geom_function(fun = g, colour = "black", linewidth = 1) + theme_bw() + 
    xlab("Covariate value") + ylab("Predicted value")

# same conclusion as before, honesty with double bootstrap seems to work a lot better

# honesty MSE/R^2 compared to dishonesty for various datasets
#-------------------------------------------------------------------------------------------------

data("BostonHousing", package = "mlbench")
data("CO2", package = "datasets")
data("Ozone", package = "mlbench")
data("HousePrices", package = "AER")

# other potential datasets
data("CASchools", package = "AER")

Ozone <- na.omit(Ozone)

set.seed(2026)
rfsrc(medv ~ ., data = BostonHousing)
forest_BostonHousing <- jfforest(medv ~ ., data = BostonHousing, min_node_size = 5)
forest_BostonHousing_honest <- jfforest(medv ~ ., data = BostonHousing, honest = TRUE, double_bootstrap = TRUE, min_node_size = 5)

print_forest(forest_BostonHousing_honest)

forest_BostonHousing$mse.error          # 13.02692
forest_BostonHousing_honest$mse.error   # 14.33489
forest_BostonHousing$R2                 # 0.8456884
forest_BostonHousing_honest$R2          # 0.8301947

# conclusion: honesty is slightly worse for BostonHousing

set.seed(2026)
rfsrc(uptake ~ ., data = CO2, min_node_size = 5)
# note: much better with min_node_size = 3
forest_CO2 <- jfforest(uptake ~ ., data = CO2, min_node_size = 3)
forest_CO2_honest <- jfforest(uptake ~ ., data = CO2, honest = TRUE, double_bootstrap = TRUE, min_node_size = 3)

print_forest(forest_CO2_honest)

forest_CO2$mse.error           # 11.20203
forest_CO2$R2                  # 0.9030625
forest_CO2_honest$mse.error    # 17.93083
forest_CO2_honest$R2           # 0.8448343

# conclusion: honesty is quite a lot worse for CO2, but this is also a very small dataset

set.seed(2026)
rfsrc(V4 ~ ., data = Ozone)
forest_Ozone <- jfforest(V4 ~ ., data = Ozone)
forest_Ozone_honest <- jfforest(V4 ~ ., data = Ozone, honest = TRUE, double_bootstrap = TRUE)

forest_Ozone$mse.error          # 17.64241
forest_Ozone$R2                 # 0.7356805
forest_Ozone_honest$mse.error   # 16.98578
forest_Ozone_honest$R2          # 0.7455183

# honesty works slightly better for Ozone

set.seed(2026)
rfsrc(price ~ ., data = HousePrices)
forest_HousePrices <- jfforest(price ~ ., data = HousePrices)
forest_HousePrices_honest <- jfforest(price ~ ., data = HousePrices, honest = TRUE, double_bootstrap = TRUE)

# rfsrc MSE:                              263183422.699777
# rfsrc R^2                               0.63089568
forest_HousePrices$mse.error            # 268634483
forest_HousePrices$R2                   # 0.6225595
forest_HousePrices_honest$mse.error     # 248021766
forest_HousePrices_honest$R2            # 0.6515211

# overall conclusion: 50/50 whether it works better or not, so definitely worth
# to try both with and without honesty when applying the method in practice

# Testing asymptotic normality
#-------------------------------------------------------------------------------------------------

# here we need a density satisfying Assumption 1, so start with Unif[0, 1]^d for simplicity