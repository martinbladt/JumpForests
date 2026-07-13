#nolint start: line_length_linter
devtools::load_all()
{
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)
library(AalenJohansen)
require(survival)

# stole the example code from https://cran.r-project.org/web/packages/AalenJohansen/vignettes/AalenJohansen-vignette.html

# Markov model with independent censoring and covariates (a single Unif[0, 1] X)
jump_rate <- function(i, t, u){
  if(i == 1){
    2
  } else if(i == 2){
    3
  } else{
    0
  }
}

mark_dist <- function(i, s, v){
  if(i == 1){
    c(0, 1/2, 1/2)
  } else if(i == 2){
    c(2/3, 0, 1/3)
  } else{
    0
  }
}

lambda <- function(t, x){
  A <- matrix(c(2/(1+x*t)*mark_dist(1, t, 0), 3/(1+x*t)*mark_dist(2, t, 0), rep(0, 3)),
              nrow = 3, ncol = 3, byrow = TRUE)
  diag(A) <- -rowSums(A)
  A
}

set.seed(2026)

n <- 1000
X <- runif(n)   # signal
Y <- rnorm(n)
#Y <- rbinom(n, 3, 0.2)   # noise
c <- runif(n, 0, 5)

sim <- list()
for(i in 1:n){
  rates <- function(j, y, z){jump_rate(j, y, z)/(1+X[i]*y)}
  sim[[i]] <- sim_path(sample(1:2, 1), rates = rates, dists = mark_dist,
                       tn = c[i], bs = c(2*c[i], 3*c[i], 0))
  sim[[i]]$X <- X[i]
  sim[[i]]$Y <- Y[i]
}

sum(c == unlist(lapply(sim, FUN = function(z){tail(z$times, 1)}))) / n  #0.295
sim[1]

# now test the package
#test_data <- data.frame(X1 = X, X2 = Y)
#formula <- MM ~ X1 + X2
#formula[[3]]
#attr(terms(formula), "term.labels")
#which(names(test_data) %in% attr(terms(formula), "term.labels")) - 1

# test data
#sim
#test_data_functions_mm(sim, test_data, c(1, 2))
# conclusion: Data works precisely as intended, also for multi-states

# testing thinning of the unique event times
# (unique_event_times includes censoring in contrast to the functions below, just for testing purposes)
#unique_event_times <- sort(unique(unlist(lapply(sim, function(x) x$times))))
#test_unique_event_times(unique_event_times, 0.5)  # looks fine
# implement as an option

test_data <- data.frame(X1 = X, X2 = Y)
#test_data_functions_mm(sim, test_data, c(1, 2))
fitted_tree <- jftree(MM ~ X1 + X2, data = sim, feature_data = test_data, nsplits = 10, splitrule = "logrank", min_node_size = 20, honest = TRUE)
}
print_tree(fitted_tree, full = TRUE)
# to get exactly one split, just set seed to 2026 and n = 60 with nsplits = 2, 10

# still need more testing on IBS calculations
fitted_tree$ibs
fitted_tree$ibs.normalised

jftree.predict(fitted_tree)[[1]][1] # Nelson-Aalen
jftree.predict(fitted_tree)[[2]][1] # initial distribution
unlist(lapply(fitted_tree$init, function(z) sum(z)))
fitted_tree$censoring[1,]
#new_data <- data.frame(X2 = runif(10), X1 = rnorm(10))
new_data <- test_data
predicted <- jftree.predict(fitted_tree, new_data, compute_initial = TRUE, compute_censoring = FALSE)
predicted
predicted$predictions[[1]]
jftree.predict(fitted_tree)[[1]][[1]]
unlist(jftree.predict(fitted_tree)$predictions) - unlist(jftree.predict(fitted_tree, new_data = new_data, compute_initial = TRUE)$predictions) # all zeroes as it should be

predicted$censoring
predicted$predictions.init[[1]]
jftree.predict(fitted_tree)[[2]][[1]]

jftree.error(fitted_tree)
jftree.error(fitted_tree, new_data = test_data, jump_data = sim)

occupation_prob(init = fitted_tree$init[[1]], na = fitted_tree$predictions[[1]])
lapply(fitted_tree$init, function(z) sum(z))
lapply(occupation_prob(init = fitted_tree$init[[1]], na = fitted_tree$predictions[[1]]), function(z) sum(z))

# fit the forest (takes a couple of minutes when also saving predictions)
fitted_forest <- jfforest(MM ~ X1 + X2, data = sim, feature_data = test_data, ntrees = 100, min_node_size = 20, splitrule = "logrank", save_predictions = FALSE, honest = FALSE)
print_forest(fitted_forest)

jfforest.predict(fitted_forest)$predictions[[1]]
jfforest.predict(fitted_forest)$init[[1]]
jfforest.predict(fitted_forest, new_data = test_data, compute_initial = TRUE)$init
unlist(jfforest.predict(fitted_forest)$init) - unlist(jfforest.predict(fitted_forest, new_data = test_data, compute_initial = TRUE)$init) # all zeroes as it should be

jfforest.predict(fitted_forest)$predictions
jfforest.predict(fitted_forest, new_data = test_data, compute_initial = TRUE)$predictions
unlist(jfforest.predict(fitted_forest)$predictions) - unlist(jfforest.predict(fitted_forest, new_data = test_data, compute_initial = TRUE)$predictions)
# okay, the internal predictions are the same as predictions computed manually on the training data

#jfforest.predict(fitted_forest)  # warning: only call if the number of observations is not very large (otherwise it never finishes printing)
unlist(lapply(jfforest.predict(fitted_forest)[[2]], function(z) sum(z)))
occupation_prob(init = fitted_forest$init[[1]], na = fitted_forest$predictions[[1]])
predictions_new_data <- jfforest.predict(fitted_forest, new_data, compute_initial = TRUE)
occupation_prob(init = predictions_new_data$initial[[1]], na = predictions_new_data$predictions[[1]])
#new_data <- data.frame(X2 = runif(1), X1 = rnorm(1))
#new_data <- test_data[1,]

# testing the error metrics on a survival data set (veteran)
#-------------------------------------------------------------------------------------------------
{
devtools::load_all()
data(veteran, package = "randomForestSRC")
veteran$trt <- as.factor(veteran$trt)
veteran$celltype <- as.factor(veteran$celltype)
veteran$prior <- as.factor(veteran$prior)
# convert to multi-state dataset
jump_data_veteran <- lapply(seq_len(nrow(veteran)), function(i) {
  if (veteran$status[i] == 1) {
    list(times = c(0, veteran$time[i]), states = c(1L, 2L))   # important that the states are of type "integer"
  } else {
    list(times = c(0, veteran$time[i]), states = c(1L, 1L))
  }
})

veteran_features <- veteran[-c(3, 4)]
#test_data_functions_mm(jump_data_veteran, veteran_features, ncol(veteran_features))

veteran_tree_mm <- jftree(MM ~ ., data = jump_data_veteran, seed = 2026, feature_data = veteran_features, nsplits = 10, splitrule = "logrank", min_node_size = 10, honest = FALSE)
veteran_tree <- jftree(Surv(time, status) ~ ., veteran, seed = 2026, min_node_size = 10, honest = FALSE)

print_tree(veteran_tree_mm)
print_tree(veteran_tree)

veteran_tree_mm$ibs             # 56.07324
veteran_tree$ibs                # 57.51201
veteran_tree_mm$ibs.normalised  # 0.05612937
veteran_tree$ibs.normalised     # 0.05756958

# close enough

# equals 999, the last event time (as it should)
56.07324/0.05612937
57.51201/0.05756958
tail(veteran_tree_mm$unique.event.times)
tail(veteran_tree$unique.event.times)

# very little difference, but censoring is also extremely light for this dataset
veteran_tree_mm$censoring[1, ]
veteran_tree$censoring[1, ]
veteran_tree_mm$censoring[2, ]
veteran_tree$censoring[2, ]
veteran_tree_mm$censoring[5, ]
veteran_tree$censoring[5, ]

veteran_tree_mm$init
unlist(lapply(veteran_tree_mm$init, function(z) sum(z)))  # works!
}

# Study 1: Comparison to survival data for continuous distributions
#-------------------------------------------------------------------------------------------------

# plan for 18/5 and beyond

# 2) For moderate to heavy censoring, the KM estimators for censoring are quite different when using multi-state trees instead
#    of survival trees
# 4) Allow the user to input an initial distribution for multi-state error computation
# 6) Test errors against competing risks in randomForestSRC
# 8) Extend error computations to whole forests

# finished points
# 1) Fix initial value estimation (see veteran above), empirical studies show that this is very likely where the difference
#    in errors come from (big issue!)
# 3) Related to 2), inconsistent computation of KM, for survival we subtract the number of deaths, but we don't subtract the
#    total number of jumps for multi-states (changed, should no longer be a difference)
# 5) (Optional) Reconsider removing the 'censored only' event times. This is more natural. What went wrong with the error computation?
# 7) Write prediction functions for multi-state random forests (not just single trees)
# 9) There should be a function for survival forests which computes both OOB predictions and censoring predictions in one go instead of
#    using two different functions (that way we only need to determine the leaf once)
# 10) Repeat 9) for a single multi-state tree. Also do a thorough cleanup and remove old code used for prediction
# 11) Something goes wrong for categorical data (again...): the values are saved internally in a nonsensical way
# 12) Fix the subsampling printing error for honesty

devtools::load_all()
library(randomForestSRC)
set.seed(2026)
n <- 1000
X <- runif(n)
Y <- rnorm(n)                 # noise
Z <- 10 + rexp(n, 1/(2 + X))  # true survival time
R <- runif(n, 12, 20)         # censoring times
W <- pmin(Z, R)               # observed times
delta <- as.numeric(Z <= R)   # status indicators
1 - sum(delta)/n              # censoring rate

sim_data <- data.frame(time = W, status = delta, X1 = X, X2 = Y)
#test_data_functions(sim_data, c(1,2), c(3,4))

jump_data <- lapply(seq_len(nrow(sim_data)), function(i) {
  if (sim_data$status[i] == 1) {
    list(times = c(0, sim_data$time[i]), states = c(1L, 2L))   # important that the states are of type "integer"
  } else {
    list(times = c(0, sim_data$time[i]), states = c(1L, 1L))
  }
})

tree_survival <- jftree(Surv(time, status) ~ ., data = sim_data, min_node_size = 15, honest = FALSE)
tree_mm <- jftree(MM ~ ., data = jump_data, feature_data = sim_data[c(3, 4)], min_node_size = 15, honest = FALSE)

# results for n = 1000, seed = 2026
jftree.error(tree_survival) # IBS: 1.03857, normalised IBS: 0.0560871
tree_mm$ibs                 # IBS: 0.9436641
tree_mm$ibs.normalised      # normalised IBS: 0.05096178

tail(tree_survival$unique.event.times)
tail(tree_mm$unique.event.times)

unlist(lapply(tree_mm$init, function(z) sum(z)))

tree_mm$censoring[1,]
tree_survival$censoring[1,] # extreme difference for the first observation
tree_mm$censoring[2,]
tree_survival$censoring[2,]
tree_mm$censoring[3,]
tree_survival$censoring[3,] # pretty big difference
tree_mm$censoring[4,]
tree_survival$censoring[4,] # zero???
tree_mm$censoring[5,]
tree_survival$censoring[5,]

# may need to do some investigating here, but other matters are more pressing
# an obvious possible explanation for the difference is that censoring times are included in the unique event times for survival (maybe revert?)

# just a bonus comparison with randomForestSRC

forest_survival <- jfforest(Surv(time, status) ~ ., data = sim_data, min_node_size = 15, honest = TRUE)
forest_survival_src <- rfsrc(Surv(time, status) ~ ., data = sim_data)

print_forest(forest_survival) # C-error: 0.4609178, IBS: 1.187464, normalised IBS: 0.05969983
forest_survival_src           # C-error: 0.46226363, IBS: 1.20540655, normalised IBS: 0.06509696

# our survival forest provides significantly lower errors with honesty! C-error: 0.403807, IBS: 1.070356, normalised IBS: 0.0538122
# (just a bonus observation, important to investigate the effect of honesty somewhere, maybe a direct consequence of much larger OOB sample...)

# regain memory
rm('forest_survival', 'forest_survival_src')
gc()

# Study 2: Comparison to survival data for distributions with ties and categorical features
#-------------------------------------------------------------------------------------------------

{
devtools::load_all()
library(randomForestSRC)
set.seed(2026)
n <- 10000
X <- rbinom(n, 10, 0.5)
Y <- rnorm(n)                 # noise
Z <- 10 + rbinom(n, 80, 0.5)  # true survival time (relatively few unique values)
R <- 45 + rbinom(n, 20, 0.4)  # censoring times
W <- pmin(Z, R)               # observed times
delta <- as.numeric(Z <= R)   # status indicators
1 - sum(delta)/n              # censoring rate

sim_data <- data.frame(time = W, status = delta, X1 = X, X2 = Y)
sim_data$X1 <- as.factor(sim_data$X1)

typeof(sim_data$X1)
typeof(sim_data$X2)
test_data_functions(sim_data, c(1,2), c(3,4))

jump_data <- lapply(seq_len(nrow(sim_data)), function(i) {
  if (sim_data$status[i] == 1) {
    list(times = c(0, sim_data$time[i]), states = c(1L, 2L))   # important that the states are of type "integer"
  } else {
    list(times = c(0, sim_data$time[i]), states = c(1L, 1L))
  }
})

tree_survival <- jftree(Surv(time, status) ~ ., data = sim_data, min_node_size = 15, honest = FALSE)                # why did I not implement multi-threading for prediction for survival trees?
tree_mm <- jftree(MM ~ ., data = jump_data, feature_data = sim_data[c(3, 4)], min_node_size = 15, honest = FALSE)   # much faster for large n
}
print_tree(tree_mm)

# n = 1000 (quite different)
tree_survival$ibs             # 2.145489
tree_survival$ibs.normalised  # 0.03764016
tree_mm$ibs                   # 1.963569
tree_mm$ibs.normalised        # 0.03444858

# n = 10000 (more similar)
tree_survival$ibs             # 2.218344
tree_survival$ibs.normalised  # 0.0369724
tree_mm$ibs                   # 1.993813
tree_mm$ibs.normalised        # 0.03323021

# n = 25000
tree_survival$ibs             # 2.176473
tree_survival$ibs.normalised  # 0.03688937
tree_mm$ibs                   # 1.969716
tree_mm$ibs.normalised        # 0.03338501

tree_mm$censoring[1,]
tree_survival$censoring[1,]   # much lower
tree_mm$censoring[2,]
tree_survival$censoring[2,]   # much lower
tree_mm$censoring[3,]
tree_survival$censoring[3,]   # ditto

tree_mm$init  # at least this makes sense

# testing predictions
tree_mm$unique.event.times
row <- 121
jftree.predict(tree_mm)$predictions[[row]]
jftree.predict(tree_mm, new_data = sim_data[row, c(3,4)])[[1]]
unlist(jftree.predict(tree_mm)$predictions[[row]]) - unlist(jftree.predict(tree_mm, new_data = sim_data[row, c(3,4)])[[1]])
# seems to work for categorical data here (but X1 is also an 'integer', maybe 'factor' is the problem?)

sim_data$X1 <- sample(c("Yes", "No"), n, replace = TRUE)
typeof(sim_data$X1) # character
tree_survival <- jftree(Surv(time, status) ~ ., data = sim_data, min_node_size = 15, honest = FALSE)
tree_mm <- jftree(MM ~ ., data = jump_data, feature_data = sim_data[c(3, 4)], min_node_size = 15, honest = FALSE)

row <- 134
jftree.predict(tree_mm)$predictions[[row]]
jftree.predict(tree_mm, new_data = sim_data[row, c(3,4)])[[1]]
unlist(jftree.predict(tree_mm)$predictions[[row]]) - unlist(jftree.predict(tree_mm, new_data = sim_data[row, c(3,4)])[[1]])

# no issues in regards to Y, Y1, num_at_risk etc. here? so it has nothing to do with ties in the data I guess, the veteran dataset just acts weird

# reminder: should also compare to competing risks
#-------------------------------------------------------------------------------------------------

# comparison to CoAJ
#-------------------------------------------------------------------------------------------------
set.seed(2026)

# fit the forest (takes a couple of minutes when also saving predictions) (use data above!)
fitted_forest <- jfforest(MM ~ X1 + X2, data = sim, feature_data = test_data, ntrees = 100, min_node_size = 20, splitrule = "logrank", save_predictions = FALSE, honest = FALSE)
print_forest(fitted_forest)

x1 <- 0.2
x2 <- 0.6
x3 <- 0.8

new_data <- data.frame(X1 = c(x1, x2, x3), X2 = c(0, 0, 0))
forest_fit <- jfforest.predict(fitted_forest, new_data, compute_initial = TRUE)

fit1 <- aalen_johansen(sim, x = x1)
fit2 <- aalen_johansen(sim, x = x2)
fit3 <- aalen_johansen(sim, x = x3)

# compute the predictions from the conditional Aalen-Johansen estimator
v11 <- unlist(lapply(fit1$Lambda, FUN = function(L) L[2,1]))
v10 <- fit1$t
v21 <- unlist(lapply(fit2$Lambda, FUN = function(L) L[2,1]))
v20 <- fit2$t
v31 <- unlist(lapply(fit3$Lambda, FUN = function(L) L[2,1]))
v30 <- fit3$t
p1 <- unlist(lapply(fit1$p, FUN = function(L) L[2]))
P1 <- unlist(lapply(prodint(0, 5, 0.01, function(t){lambda(t, x = x1)}),
FUN = function(L) (c(1/2, 1/2, 0) %*% L)[2]))
p2 <- unlist(lapply(fit2$p, FUN = function(L) L[2]))
P2 <- unlist(lapply(prodint(0, 5, 0.01, function(t){lambda(t, x = x2)}),
FUN = function(L) (c(1/2, 1/2, 0) %*% L)[2]))

# compute the predictios from the random forest
times <- fitted_forest$unique.event.times
v11_forest <- unlist(lapply(forest_fit[[1]][[1]], FUN = function(L) L[2,1]))
v21_forest <- unlist(lapply(forest_fit[[1]][[2]], FUN = function(L) L[2,1]))
v31_forest <- unlist(lapply(forest_fit[[1]][[3]], FUN = function(L) L[2,1]))
#p1_forest <- unlist(lapply(aj, FUN = function(L) L[2]))
#P1 <- unlist(lapply(prodint(0, 5, 0.01, function(t){lambda(times, x = x1)}),
#FUN = function(L) (c(1/2, 1/2, 0) %*% L)[2]))

# plot options
par(mfrow = c(1, 2))
par(mar = c(2.5, 2.5, 1.5, 1.5))
options(vsc.dev.args = list(width = 1000, height = 600, res = 100))

# cumulative hazard for the transition from 2 to 1 (cAJ)
plot(v10, v11, type = "l", lty = 2, xlab = "", ylab = "", main = "Hazard (cAJ)", col = "red", xlim = c(0, 5), ylim = c(0, 4))
lines(v10, 2/x1*log(1+x1*v10), col = "red")
lines(v20, v21, lty = 2, col = "blue")
lines(v20, 2/x2*log(1+x2*v20), col = "blue")
lines(v30, v31, lty = 2, col = "darkgreen")
lines(v30, 2/x3*log(1+x3*v30), col = "darkgreen")

# occupation probability for state 2 (cAJ)
plot(v10, p1, type = "l", lty = 2, xlab = "", ylab = "", main = "Probability (cAJ)", col = "red")
lines(seq(0, 5, 0.01), P1, col = "red")
lines(v20, p2, lty = 2, col = "blue")
lines(seq(0, 5, 0.01), P2, col = "blue")

# cumulative hazard for the transition from 2 to 1 (RF)
plot(times, v11_forest, type = "l", lty = 2, xlab = "", ylab = "", main = "Hazard (RF)", col = "red", xlim = c(0, 5), ylim = c(0, 4))
lines(times, 2/x1*log(1+x1*times), col = "red")
lines(times, v21_forest, lty = 2, col = "blue")
lines(times, 2/x2*log(1+x2*times), col = "blue")
lines(times, v31_forest, lty = 2, col = "darkgreen")
lines(times, 2/x3*log(1+x3*times), col = "darkgreen")

# plot the predictions from the random forest


#fitted_tree <- jftree(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
#                      data = test_data, splitrule = "conserve", min_node_size = 2, nsplits = 2, seed = 2025)

#nolint_end


