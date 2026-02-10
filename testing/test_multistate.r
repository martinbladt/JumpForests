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

n <- 100
X <- runif(n)   # signal
Y <- rnorm(n)   # noise
c <- runif(n, 0, 5)

sim <- list()
for(i in 1:n){
  rates <- function(j, y, z){jump_rate(j, y, z)/(1+X[i]*y)}
  sim[[i]] <- sim_path(sample(1:2, 1), rates = rates, dists = mark_dist,
                       tn = c[i], bs = c(2*c[i], 3*c[i], 0))
}

sum(c == unlist(lapply(sim, FUN = function(z){tail(z$times, 1)}))) / n  #0.28
sim[1]

# now test the package
test_data <- data.frame(X1 = X, X2 = Y)
formula <- MM ~ X1 + X2
formula[[3]]
attr(terms(formula), "term.labels")
which(names(test_data) %in% attr(terms(formula), "term.labels")) - 1

# test data
#sim
test_data_functions_mm(sim, test_data, c(1, 2))
# conclusion: Data works precisely as intended, also for multi-states

# testing thinning of the unique event times
# (unique_event_times includes censoring in contrast to the functions below, just for testing purposes)
unique_event_times <- sort(unique(unlist(lapply(sim, function(x) x$times))))
test_unique_event_times(unique_event_times, 0.5)  # looks fine
# implement as an option

fitted_tree <- jftree(MM ~ X1 + X2, data = sim, feature_data = test_data, nsplits = 10, splitrule = "logrank", min_node_size = 20)
# to get exactly one split, just set seed to 2026 and n = 60 with nsplits = 2, 10

jftree.predict(fitted_tree)
new_data <- data.frame(X2 = runif(5), X1 = rnorm(5))
jftree.predict(fitted_tree, new_d)

# holy..., no errors?? Check thoroughly
fitted_forest <- jfforest(MM ~ X1 + X2, data = sim, feature_data = test_data, ntrees = 100, splitrule = "logrank")

jfforest.predict(fitted_forest)[[1]]
#new_data <- data.frame(X2 = runif(1), X1 = rnorm(1))
new_data <- test_data[1,]
jfforest.predict(fitted_forest, new_data)

# conditional Aalen-johansen
set.seed(2026)

n <- 10000
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

x1 <- 0.2
x2 <- 0.8

fit1 <- aalen_johansen(sim, x = x1)
fit2 <- aalen_johansen(sim, x = x2)

v11 <- unlist(lapply(fit1$Lambda, FUN = function(L) L[2,1]))
v10 <- fit1$t
v21 <- unlist(lapply(fit2$Lambda, FUN = function(L) L[2,1]))
v20 <- fit2$t
p1 <- unlist(lapply(fit1$p, FUN = function(L) L[2]))
P1 <- unlist(lapply(prodint(0, 5, 0.01, function(t){lambda(t, x = x1)}),
FUN = function(L) (c(1/2, 1/2, 0) %*% L)[2]))
p2 <- unlist(lapply(fit2$p, FUN = function(L) L[2]))
P2 <- unlist(lapply(prodint(0, 5, 0.01, function(t){lambda(t, x = x2)}),
FUN = function(L) (c(1/2, 1/2, 0) %*% L)[2]))

par(mfrow = c(1, 2))
par(mar = c(2.5, 2.5, 1.5, 1.5))

plot(v10, v11, type = "l", lty = 2, xlab = "", ylab = "", main = "Hazard", col = "red")
lines(v10, 2/x1*log(1+x1*v10), col = "red")
lines(v20, v21, lty = 2, col = "blue")
lines(v20, 2/x2*log(1+x2*v20), col = "blue")

plot(v10, p1, type = "l", lty = 2, xlab = "", ylab = "", main = "Probability", col = "red")
lines(seq(0, 5, 0.01), P1, col = "red")
lines(v20, p2, lty = 2, col = "blue")
lines(seq(0, 5, 0.01), P2, col = "blue")


#fitted_tree <- jftree(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
#                      data = test_data, splitrule = "conserve", min_node_size = 2, nsplits = 2, seed = 2025)

#nolint_end