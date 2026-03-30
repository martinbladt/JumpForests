#nolint start: line_length_linter
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

n <- 200
X <- runif(n)   # signal
Y <- rnorm(n)   # noise
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

# the numbers at risk at time 0 should sum to the number of observations in the node but apparently this does not happen?? fix!
# note: this bug does not occur when we only have one split

# findings so far: 
# - the total number at risk is always consistent (thankfully), which explains why this bug was not found sooner, it is the total number
#   of observations in a node that changes when it shouldn't
# - it is only a problem when making a leaf, indicating that it is probably because one should the length of the HOLDOUT obs instead (try tomorrow)
# - it is ONLY an honesty issue, explaining why it is not a problem for the forest below

fitted_tree <- jftree(MM ~ X1 + X2, data = sim, feature_data = test_data, nsplits = 10, splitrule = "logrank", min_node_size = 20, honest = FALSE)
# to get exactly one split, just set seed to 2026 and n = 60 with nsplits = 2, 10

jftree.predict(fitted_tree)[[1]][1] # Nelson-Aalen
jftree.predict(fitted_tree)[[2]][1] # initial distribution (this doesn't sum to one, not an issue for forests, likely an index error somewhere in the background C++ code...)
lapply(fitted_tree$init, function(z) sum(z))
fitted_tree$censoring[1,]
new_data <- data.frame(X2 = runif(5), X1 = rnorm(5))
jftree.predict(fitted_tree, new_data, compute_initial = TRUE, compute_censoring = FALSE)
jftree.predict(fitted_tree, new_data, compute_initial = TRUE, compute_censoring = FALSE)$predictions.init
occupation_prob(init = fitted_tree$init[[1]], na = fitted_tree$predictions[[1]])
lapply(fitted_tree$init, function(z) sum(z))                                                                    # still some strange mistake here, see above
lapply(occupation_prob(init = fitted_tree$init[[1]], na = fitted_tree$predictions[[1]]), function(z) sum(z))

# fit the forest (takes a couple of minutes when also saving predictions)
fitted_forest <- jfforest(MM ~ X1 + X2, data = sim, feature_data = test_data, ntrees = 100, min_node_size = 100, splitrule = "logrank", save_predictions = FALSE, honest = FALSE)
print_forest(fitted_forest)

jfforest.predict(fitted_forest)[[2]][1:10]
lapply(jfforest.predict(fitted_forest)[[2]], function(z) sum(z))  # no problem with not summing to one for forests? even when all trees are trivial????
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


