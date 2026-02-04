#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)
library(AalenJohansen)
require(survival)

# stole the example code from https://cran.r-project.org/web/packages/AalenJohansen/vignettes/AalenJohansen-vignette.html
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

n <- 3
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

# interesting, it always succeeds the first time (and sometimes more than once) but then another memory error occurs
# it always finishes initialising the tree, so the error must lie in the multistate_tree pointer
# OR: we actually have to grow the tree to ensure proper garbage collection. that would explain why it always runs the first time
fitted_tree <- jftree(MM ~ X1 + X2, data = sim, feature_data = test_data)

#fitted_tree <- jftree(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
#                      data = test_data, splitrule = "conserve", min_node_size = 2, nsplits = 2, seed = 2025)

#nolint_end



