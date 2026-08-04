#nolint start: line_length_linter

devtools::load_all()
library(AalenJohansen)
library(survival)
library(eha)
#remotes::install_github("martinbladt/JumpPoisReg")
library(JumpPoisReg)
library(tictoc)
library(ggplot2)
library(dplyr)

# Helper functions for computing relevant quantities for numerical studies
#--------------------------------------------------------------------------------

# computes a list of the smooth product integral 
# (A: intensity matrix, s: starting time, t: end time, n: number of steps (h = (t - s)/n))
prodint <- function (A, s, t, n){
  x0 <- s
  y0 <- diag(nrow(A(s)))
  res <- list(y0)
  h <- (t - s)/n
  for (i in 1:n) {
    s1 <- h * y0 %*% A(x0)
    s2 <- h * (y0 + s1 / 2) %*% A (x0 + h / 2)
    s3 <- h * (y0 + s2 / 2) %*% A (x0 + h / 2)
    s4 <- h * (y0 + s3) %*% A(x0 + h)
    y0 <- y0 + s1/6 + s2/3 + s3 /3 + s4/6
    x0 <- x0 + h
    res[[i + 1]] <- y0
  }
  return(res)
}

# computes a list of the product integral from a provided Nelson-Aalen estimator
# (na: Nelson-Aalen estimator provided as a list)
prodint_from_data <- function(na) {
  y0 <- diag(nrow(na[[1]]))  # initialise with identity matrix
  res <- list(y0)
  for (i in 2:length(na)) {
    y0 <- y0 + y0 %*% (na[[i]] - na[[i - 1]])
    res[[i]] <- y0
  }
  return(res)
}

# computes a list of occupation probabilities from 0 to t
# (A: intensity matrix, t: end time, init: vector of initial probabilities, n: number of steps (h = t/n))
occprob <- function(A, t, init, n) {
  # fist compute the product integral
  res <- prodint(A, 0, t, n)
  # now multiply by initial probabilities
  return(lapply(res, function(z) init %*% z))
}
# computes a list of occupation probabilities from 0 to t based on a Nelson-Aalen estimator (a list)
# (na: a Nelson-Aalen estimator as a list, init: vector of initial probabilities)
occprob_from_data <- function(na, init) {
  # first compute the product integral
  res <- prodint_from_data(na)
  # now multiply by initial probabilities
  return(lapply(res, function(z) init %*% z))
}

# computes the matrix of transition probabilities in a semi-Markov illness-death model without reactivation
# (A: intensity matrix (a function of both time and duration), s: starting time, t: end time,
# u: initial duration, z: upper bound on the final duration, n: maximum number of
# subdivisions used by the numerical integrations)
transition_probs_semiMarkov <- function(A, s, t, u, z, n = 100L) {
    A_su <- A(s, u)

    integrate_scalar <- function(f, lower, upper) {
      if (lower >= upper) {
        return(0)
      }
      stats::integrate(
        function(x) vapply(x, f, numeric(1)),
        lower = lower,
        upper = upper,
        subdivisions = n,
        stop.on.error = TRUE
      )$value
    }

    mu01 <- function(time, duration) A(time, duration)[1L, 2L]
    mu02 <- function(time, duration) A(time, duration)[1L, 3L]
    mu12 <- function(time, duration) A(time, duration)[2L, 3L]

    # Survival in state 0 from s to end. The duration argument is included even
    # though mu01 and mu02 do not depend on it in the model considered here.
    S0 <- function(end) {
      exp(-integrate_scalar(function(time) {
          duration <- u + time - s
          mu01(time, duration) + mu02(time, duration)
        },
        s,
        end
      ))
    }

    # Survival in state 1, either from the initial state (entry = s and
    # initial_duration = u) or following a 0 -> 1 transition (initial_duration = 0).
    S1 <- function(entry, end, initial_duration = 0) {
      exp(-integrate_scalar(
        function(time) mu12(time, initial_duration + time - entry),
        entry,
        end
      ))
    }

    # Probability of being in state 1 at `end`, having started in state 0 at s.
    # If min_entry is supplied, only paths whose 0 -> 1 transition occurred at
    # or after that time are counted.
    P01 <- function(end, min_entry = s) {
      lower <- max(s, min_entry)
      integrate_scalar(
        function(entry) {
          S0(entry) * mu01(entry, u + entry - s) * S1(entry, end)
        },
        lower,
        end
      )
    }

    cutoff <- max(s, t - z)
    p00_t <- S0(t)
    p01_t <- P01(t)
    p11_t <- S1(s, t, u)

    # Death has final duration <= z exactly when it occurs in [cutoff, t].
    # Since death is absorbing, this probability equals the probability of being
    # alive at cutoff minus the probability of being alive at t.
    p02 <- S0(cutoff) + P01(cutoff) - p00_t - p01_t
    p12 <- S1(s, cutoff, u) - p11_t

    result <- matrix(0, nrow = 3L, ncol = 3L)
    result[1L, 1L] <- if (u + t - s <= z) p00_t else 0
    result[1L, 2L] <- P01(t, cutoff)
    result[1L, 3L] <- p02
    result[2L, 2L] <- if (u + t - s <= z) p11_t else 0
    result[2L, 3L] <- p12
    result[3L, 3L] <- as.numeric(u + t - s <= z)

    # Remove negligible negative values caused by subtracting nearly equal
    # probabilities in the death column.
    result[result < 0 & result > -sqrt(.Machine$double.eps)] <- 0
    result
}

# computes the vector of optimal theoretical expected Brier scores
# (occ_prob: list of occupation probabilities, c: weights)
optimal_brier <- function(c, occ_prob) {
    return(lapply(occ_prob, function(z) (z * (1 - z)) %*% c))
}

# computes the vector of optimal theoretical expected Kullback-Leibler scores
# (occ_prob: list of occupation probabilities, c: weights)
optimal_KL <- function(c, occ_prob) {
    # Kullback-Leibler error is only proper for uniform weights, otherwise the optimum has to be found
    summands <- lapply(occ_prob, function(z) c * z * log(c * z / as.numeric(z %*% c)))
    return(lapply(summands, function(z) -sum(z)))
}

# computes the vector of optimal theoretical expected spherical scores
# (occ_prob: list of occupation probabilities, c: weights)
optimal_spherical <- function(c, occ_prob) {
    return(lapply(occ_prob, function(z) 1 - sqrt((z * z) %*% c)))
}

# Helper functions for plotting
#--------------------------------------------------------------------------------



# Testing the helper functions
#--------------------------------------------------------------------------------

# test some Markov intensities

mu01 <- function(x) {
  if (x <= 65) (0.0004 + 10^(4.54 + 0.06*x - 10))
  else 0
}

mu10 <- function(x) {
  if (x <= 65) (2.0058 * exp(-0.117 * x))
  else 0
}

mu02 <- function(x) {
  0.0005 + 10^(5.88 + 0.038 * x - 10)
}

mu12 <- function(x) {
  if (x <= 65) 2 * mu02(x)
  else mu02(x)
}

Lambda <- function(t) {
  matrix(c(-mu01(t) - mu02(t), mu01(t), mu02(t),
           mu10(t), -mu10(t) - mu12(t), mu12(t),
           0, 0, 0), nrow = 3, byrow = TRUE)
}

# testing the errors
test <- occprob(Lambda, 100, c(1,0,0), 10^4)
w <- c(1/3, 1/3, 1/3)
optimal_brier(w, test)
optimal_brier(w, list(c(1/3, 1/3, 1/3))) # = 2/9 as it should be
optimal_spherical(w, test)
optimal_spherical(w, list(c(1/3, 1/3, 1/3))) # = 2/3 as it should be
optimal_KL(w, test)
optimal_KL(w, list(c(1/3, 1/3, 1/3))) # = log(3)/3 as it should be

