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

n <- 100
X <- runif(n)   # signal
Y <- rnorm(n)
#Y <- rbinom(n, 3, 0.2)   # noise
c <- runif(n, 0, 5)

sim <- list()
for(i in 1:n){
  rates <- function(j, y, z){jump_rate(j, y, z)/(1+X[i]*y)}
  sim[[i]] <- sim_path(sample(1:2, 1), rates = rates, dists = mark_dist,
                       tn = c[i], bs = c(2*c[i], 3*c[i], 0))
  #sim[[i]]$X <- X[i]
  #sim[[i]]$Y <- Y[i]
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

test_data <- data.frame(X1 = X, X2 = as.character(Y))
test_data_functions_mm(sim, test_data, c(1, 2))
fitted_tree <- jftree(MM ~ X1 + X2, data = sim, feature_data = test_data, nsplits = 10, splitrule = "logrank", min_node_size = 20, honest = FALSE)
}
# to get exactly one split, just set seed to 2026 and n = 60 with nsplits = 2, 10

# these results do not make sense at the moment! there is a key decomposition computation that needs fixing in Data.cpp
fitted_tree$ibs
fitted_tree$ibs.normalised

jftree.predict(fitted_tree)[[1]][1] # Nelson-Aalen
jftree.predict(fitted_tree)[[2]][1] # initial distribution
lapply(fitted_tree$init, function(z) sum(z))
fitted_tree$censoring[1,]
new_data <- data.frame(X2 = runif(10), X1 = rnorm(10))
predicted <- jftree.predict(fitted_tree, new_data, compute_initial = TRUE, compute_censoring = TRUE)
predicted$predictions
predicted$censoring
predicted$predictions.init

occupation_prob(init = fitted_tree$init[[1]], na = fitted_tree$predictions[[1]])
lapply(fitted_tree$init, function(z) sum(z))
lapply(occupation_prob(init = fitted_tree$init[[1]], na = fitted_tree$predictions[[1]]), function(z) sum(z))

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
test_data_functions_mm(jump_data_veteran, veteran_features, ncol(veteran_features))

veteran_tree_mm <- jftree(MM ~ ., data = jump_data_veteran, seed = 2026, feature_data = veteran_features, nsplits = 10, splitrule = "logrank", min_node_size = 10, honest = FALSE)
veteran_tree <- jftree(Surv(time, status) ~ ., veteran, seed = 2026, min_node_size = 10, honest = FALSE)

veteran_tree_mm$ibs             # 90.56269
veteran_tree$ibs                # 54.02095  # big difference, investigate (maybe the at risk convention when computing the KM estimator for censoring?)
veteran_tree_mm$ibs.normalised  # 0.09065334
veteran_tree$ibs.normalised     # 0.05407503

# equals 999, the last event time (as it should)
90.56269/0.09065334
54.02095/0.05407503
tail(veteran_tree_mm$unique.event.times)
tail(veteran_tree$unique.event.times)

# a little difference, but censoring is also extremely light for this dataset
veteran_tree_mm$censoring[1, ]
veteran_tree$censoring[1, ]
veteran_tree_mm$censoring[2, ]
veteran_tree$censoring[2, ]
veteran_tree_mm$censoring[5, ]
veteran_tree$censoring[5, ]

veteran_tree_mm$init  # why are these not all c(1, 0) when honest == FALSE? (major problem)
}

# more comparisons to survival using simulated data
#-------------------------------------------------------------------------------------------------

# plan for 8/6 and beyond
# 1) Fix initial value estimation (see veteran above), empirical studies show that this is very likely where the difference
#    in errors come from (big issue!)
# 2) For moderate to heavy censoring, the KM estimators for censoring are quite different when using multi-state trees instead
#    of survival trees
# 3) Related to 2), inconsistent computation of KM, for survival we subtract the number of deaths, but we don't subtract the
#    total number of jumps for multi-states
# 4) Allow the user to input an initial distribution for multi-state error computation
# 5) (Optional) Reconsider removing the 'censored only' event times. This is more natural. What went wrong with the error computation?
# 6) Test errors against competing risks in randomForestSRC
# 7) Write prediction functions for multi-state random forests (not just single trees)
# 8) Extend error computations to whole forests

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

jftree.error(tree_survival) # IBS: 1.071071, normalised IBS: 0.05384817
tree_mm$ibs                 # IBS: 0.9889692
tree_mm$ibs.normalised      # normalised IBS: 0.05340845

18.51709  # last unique event time for the multi-state tree (total times: 858)
19.89058  # last unique event time for the survival tree (total times: 1000)
# reason for the difference: we do not remove the times that are censored (we used to, but I 'removed' this feature, something to do with censoring KM estimation)

unlist(lapply(tree_mm$init, function(z) sum(z)))  # hmmmm, works as intended here, so why not for veteran?

tree_mm$censoring[1,]
tree_survival$censoring[1,] # extreme difference for the first observation
tree_mm$censoring[2,]
tree_survival$censoring[2,]
tree_mm$censoring[3,]
tree_survival$censoring[3,] # pretty big difference
tree_mm$censoring[4,]
tree_survival$censoring[4,] # zero???
tree_mm$censoring[5,]
tree_survival$censoring[5,] # again zero

# may need to do some investigating here, but other matters are more pressing

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

# reminder: should also compare to competing risks
#-------------------------------------------------------------------------------------------------

# forest testing
#-------------------------------------------------------------------------------------------------

# fit the forest (takes a couple of minutes when also saving predictions)
fitted_forest <- jfforest(MM ~ X1 + X2, data = sim, feature_data = test_data, ntrees = 100, min_node_size = 20, splitrule = "logrank", save_predictions = FALSE, honest = FALSE)
print_forest(fitted_forest)

jfforest.predict(fitted_forest)[[2]][1:10]
lapply(jfforest.predict(fitted_forest)[[2]], function(z) sum(z))
occupation_prob(init = fitted_forest$init[[1]], na = fitted_forest$predictions[[1]])
predictions_new_data <- jfforest.predict(fitted_forest, new_data, compute_initial = TRUE)
occupation_prob(init = predictions_new_data$initial[[1]], na = predictions_new_data$predictions[[1]])
#new_data <- data.frame(X2 = runif(1), X1 = rnorm(1))
#new_data <- test_data[1,]

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
v11_forest <- unlist(lapply(forest_fit[[1]], FUN = function(L) L[2,1]))
v21_forest <- unlist(lapply(forest_fit[[2]], FUN = function(L) L[2,1]))
v31_forest <- unlist(lapply(forest_fit[[3]], FUN = function(L) L[2,1]))
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


