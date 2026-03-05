# A file containing basic functions

#' Compute the Nelson-Aalen estimator based on data in a node
#'
#' @param data A list of trajectory data for each individual.
#' @noRd
#'

nelson_aalen_legacy <- function(data) {
  n <- length(data)
  p <- max(unique(unlist(lapply(data, function(Z) Z$states)))) - 1

  # Extract sojourn times, cumulative sojourn times, and state transition
  # information
  R_times <- unlist(lapply(data, FUN = function(Z) tail(Z$times, 1)))
  t_pool <- unlist(lapply(data, FUN = function(Z) Z$times[-1]))

  # Create a vector specifying which jump times belong to which individual
  individuals <- c()
  for (i in 1:n){
    individuals <- c(individuals, rep(i, length(data[[i]]$times[-1])))
  }

  # Create a list of matrices specifying which jumps occur for each individual
  jumps_pool <- matrix(NA, 0, 2)
  for (i in 1:n){
    v <- data[[i]]$states
    jumps_pool <- rbind(jumps_pool, cbind(rev(rev(v)[-1]), v[-1]))
  }

  # Sort the times, individuals, and transitions by time
  order_of_times <- order(t_pool)
  ordered_times <- c(0,t_pool[order_of_times])
  ordered_individuals <- c(NA,individuals[order_of_times])
  ordered_jumps <- jumps_pool[order_of_times,]

  # Computes the list of matrices containing the number of jumps between every
  # state ordered according to time
  ordered_N <- rep(list(matrix(0, p + 1, p + 1)), nrow(ordered_jumps))
  for(i in 1:nrow(ordered_jumps)){
    ordered_N[[i]][ordered_jumps[i, ][1], ordered_jumps[i, ][2]] <- 1
  }
  ordered_N <- lapply(ordered_N, FUN = function(Z) Z - diag(diag(Z)))

  # Computes the number of jumps into each state subtracted the number of jumps
  # out of the state, again ordered according to time
  colsums_of_N <- lapply(ordered_N, function(N) colSums(N - t(N)))

  # Compute the cumulative jumps between every state
  cum_jumps <- list()
  cum_jumps[[1]] <- ordered_N[[1]]
  for (tm in 2:(length(ordered_times) - 1)) {
    cum_jumps[[tm]] <- cum_jumps[[tm - 1]] + ordered_N[[tm]]
  }

  # compute C, the censoring contribution in the estimator for I
  cens <- list()
  decisions <- ordered_times %in% R_times
  cens[[1]] <- colsums_of_N[[1]]
  if (decisions[2]){
    wch <- ordered_individuals[2]
    end_state <- as.numeric(1:(p + 1) == tail(data[[wch]]$states, 1))
    cens[[1]] <- cens[[1]] - end_state
  }
  for (tm in 2:(length(ordered_times) - 1)){
    cens[[tm]] <- cens[[tm - 1]] + colsums_of_N[[tm]]
    if (decisions[tm + 1]){
      wch <- ordered_individuals[tm + 1]
      end_state <- as.numeric(1:(p + 1) == tail(data[[wch]]$states, 1))
      cens[[tm]] <- cens[[tm]] - end_state
    }
  }

  # Extract initial status for each individual
  I_initial <-  lapply(data, FUN = function(Z) as.numeric(1:(p+1) == head(Z$states, 1)))

  # Compute initial rate for all individuals
  I0 <- Reduce("+", I_initial)

  # Compute rates over time
  It <- lapply(cens, FUN = function(N) I0 + N)

  # Compute increments for each time point
  increments <- list()
  increments[[1]] <- cum_jumps[[1]]
  for (i in 2:length(cum_jumps)){
    increments[[i]] <- cum_jumps[[i]] - cum_jumps[[i - 1]]
  }

  # Compute contribution of each time point
  contribution_first <- increments[[1]]/I0
  contribution_first[is.nan(contribution_first)] <- 0 # apply convention 0/0 = 0
  contributions <- mapply(FUN = function(a, b) {
    res <- b / a
    res[is.nan(res)] <- 0
    res
  }, It[-length(It)], increments[-1], SIMPLIFY = FALSE)
  
  # Compute cumulative sum of contributions
  cumsums <- list()
  cumsums[[1]] <- contribution_first
  for(i in 2:length(cum_jumps)){
    cumsums[[i]] <- contributions[[i-1]] + cumsums[[i - 1]]
  }

  # Final touch: adding the diagonal to the Nelson-Aalen estimator
  cumsums <- append(list(matrix(0, p + 1, p + 1)),
   lapply(cumsums, FUN = function(M){M_out <- M; diag(M_out) <- -rowSums(M); M_out}))

  return(list(Lambda = cumsums, N = cum_jumps, I0 = I0, It = It, t = ordered_times))
}

#' Compute the Aalen-Johansen estimator based on data in a node. Either data
#' has to be supplied or a list of Nelson-Aalen estimators should be given
#'
#' @param data A list of trajectory data for each individual.
#' @param na A list of Nelson-Aalen estimators
#' @noRd
#'

aalen_johansen_legacy <- function(data = null, na = null) {
  if(data == null & na == null) stop("Provide either data or ")
  if (na == null) {
    na <- nelson_aalen_legacy(data)
  }

  # Recompute contributions
  contributions <- list()
  contributions[[1]] <- na$Lambda[[1]]
  for (i in 2:length(na$Lambda)) {
    contributions[[i]] <- na$Lambda[[i]] - na$Lambda[[i - 1]]
  }

  # Compute the Aalen-Johansen estimator using difference equations
  aj <- list()
  aj[[1]] <- I0
  Delta <- na$Lambda[[1]]
  aj[[2]] <- aj[[1]] + as.vector(aj[[1]] %*% Delta) - aj[[1]] * rowSums(Delta)
  for (i in 2:length(na$Lambda)){
    Delta <- contributions[[i - 1]]
    aj[[i + 1]] <- aj[[i]] + as.vector(aj[[i]] %*% Delta) - aj[[i]] * rowSums((Delta))
  }

  return(list(p = aj))
}
