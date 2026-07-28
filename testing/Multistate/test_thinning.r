#nolint start: line_length_linter
devtools::load_all()
library(AalenJohansen)
library(tictoc) # for timing

set.seed(1)
jump_rate <- function(i, t, u) {
  if (i == 1) {
    0.1 + 0.002 * t
  } else if (i == 2) {
    ifelse(u < 4, 0.29, 0.09) + 0.001 * t
  } else {
    0
  }
}

mark_dist <- function(i, s, v) {
  if (i == 1) {
    c(0, 0.9, 0.1)
  } else if (i == 2) {
    c(0, 0, 1)
  } else {
    0
  }
}

n <- 5000
cens <- runif(n, 10, 40)
sim <- vector("list", n)
for (i in seq_len(n)) {
  sim[[i]] <- sim_path(
    as.integer(1),
    rates = jump_rate,
    dists = mark_dist,
    tn = cens[i],
    bs = c(0.1 + 0.002 * cens[i], 0.29 + 0.001 * cens[i], 0)
  )
}

# subsamples the observations that are in state 2 at time 10
landmark <- sim[unlist(lapply(sim, function(z) {
  any(z$times <= 10 & c(z$times[-1], Inf) > 10 & z$states == 2)
}))]
# computes the duration at time 10 in state 2
landmark <- lapply(landmark, function(z) {
  duration <- 10 - z$times[z$times <= 10 & c(z$times[-1], Inf) > 10 & z$states == 2]
  list(times = z$times, states = as.integer(z$states), X = duration)
})

# all event and censoring times pooled together
event_times <- sort(unique(unlist(lapply(landmark, function(z) z$times))))
length(event_times) # 1981

# testing the thinning functions (for n = 5000)
thinned <- test_unique_event_times(event_times, 209)
thinned$thinned_event_times
tail(thinned$thinned_event_times)
tail(event_times)
head(thinned$thinned_event_times)
head(event_times)
# at 209 thinned times, the final event time is not the same anymore when using averages

# testing in the semi-Markov model from test_semiMarkov.R
plot_dir <- if (dir.exists("Plots")) {
  "Plots"
} else if (dir.exists("../Plots")) {
  "../Plots"
} else if (dir.exists("../../Plots")) {
  "../../Plots"
} else {
  dir.create("Plots", showWarnings = FALSE)
  "Plots"
}
plot_file <- function(name) file.path(plot_dir, name)

u1 <- 1
u2 <- 5

P <- function(t, u) {
  exp(-(t - 10) * 0.09 - (t^2 - 100) * 0.0005 - pmax(0, pmin(t, 4 - (u - 10)) - 10) * 0.20)
}

aj_from_lambda <- function(lambda_list, a0 = diag(nrow(lambda_list[[1]]))) {
  out <- vector("list", length(lambda_list))
  out[[1]] <- a0
  if (length(lambda_list) >= 2) {
    for (i in 2:length(lambda_list)) {
      delta <- lambda_list[[i]] - lambda_list[[i - 1]]
      out[[i]] <- out[[i - 1]] + as.vector(out[[i - 1]] %*% delta) -
        out[[i - 1]] * rowSums(delta)
    }
  }
  out
}

extract_prob <- function(lambda_list, from, to) {
  aj <- aj_from_lambda(lambda_list)
  vapply(aj, function(M) M[from, to], numeric(1))
}

duration_features <- data.frame(duration = vapply(landmark, function(z) z$X, numeric(1)))
cond_forest <- jfforest(
  MM ~ duration,
  data = landmark,
  feature_data = duration_features,
  mtry = 1,
  min_node_size = 150,
  nsplits = 10,
  ntrees = 1000,  # originally 5000
  splitrule = "logrank",
  seed = 2026,
  nworkers = 0,
  save_predictions = FALSE
)
print_forest(cond_forest)

# 1740 unique event times (takes approximately 56 seconds on my machine)
length(cond_forest$unique.event.times)

cond_forest_thinned <- jfforest(
  MM ~ duration,
  data = landmark,
  feature_data = duration_features,
  mtry = 1,
  min_node_size = 150,
  nsplits = 10,
  ntrees = 1000,  # originally 5000
  splitrule = "logrank",
  seed = 2026,
  nworkers = 0,
  save_predictions = FALSE,
  num_event_times = 50
)
print_forest(cond_forest_thinned)
# num_event_times = 1000: 31 seconds
# num_event_times = 500: 16 seconds
# num_event_times = 250: 10 seconds
# num_event_times = 150: 6 seconds (the effect of thinning becomes a little bit clear)
# num_event_times = 100: 5 seconds (the effect of thinning becomes somewhat clear)
# num_event_times = 50: 3 seconds (the effect of thinning becomes clear)
# num_event_times = 10: <2 seconds (very degraded, but also only for illustration)

cond_forest$unique.event.times
#cond_forest$response.event.time.ids
#unlist(lapply(landmark, function(z) z$times))
#length(unlist(lapply(landmark, function(z) z$times)))
#length(cond_forest$response.event.time.ids)

cond_forest_thinned$unique.event.times
#cond_forest_thinned$response.event.time.ids
#unlist(lapply(landmark, function(z) z$times))
#length(unlist(lapply(landmark, function(z) z$times)))
#length(cond_forest_thinned$response.event.time.ids)

tail(cond_forest$unique.event.times)
tail(cond_forest_thinned$unique.event.times)
head(cond_forest$unique.event.times)
head(cond_forest_thinned$unique.event.times)

# compare the thinned and non-thinned forests
cond_pred <- jfforest.predict(cond_forest, data.frame(duration = c(u1, u2)))
cond_pred_thinned <- jfforest.predict(cond_forest_thinned, data.frame(duration = c(u1, u2)))
p_rf_u1 <- extract_prob(cond_pred[[1]], from = 2, to = 2)
p_rf_u2 <- extract_prob(cond_pred[[2]], from = 2, to = 2)
p_rf_u1_thinned <- extract_prob(cond_pred_thinned[[1]], from = 2, to = 2)
p_rf_u2_thinned <- extract_prob(cond_pred_thinned[[2]], from = 2, to = 2)

pdf(plot_file("p2_conditional_jumpforest_thinned50.pdf"), width = 6, height = 6)
plot(seq(10, 40, 0.1), P(seq(10, 40, 0.1), u1), type = "l", lty = 1, xlab = "", ylab = "",
     col = "#e74c3c", xlim = c(10, 40), lwd = 2)                            # true, u = u1
lines(seq(10, 40, 0.1), P(seq(10, 40, 0.1), u2), lwd = 2, col = "#3498DB")  # true, u = u2
lines(cond_forest$unique.event.times, p_rf_u1, lwd = 2, col = "#27AE60")    # forest w.o. thinning, u = u1
lines(cond_forest$unique.event.times, p_rf_u2, lwd = 2, col = "#F39C12")    # forest w.o. thinning, u = u2
lines(cond_forest_thinned$unique.event.times, p_rf_u1_thinned, lty = 2, lwd = 2, col = "#114125")    # forest w. thinning, u = u1
lines(cond_forest_thinned$unique.event.times, p_rf_u2_thinned, lty = 2, lwd = 2, col = "#db08f3")    # forest w. thinning, u = u2
legend("topright",
       legend = c("True, u = 1", "True, u = 5", "JumpForest, u = 1", "JumpForest, u = 5", "JumpForest (thinned), u = 1", "JumpForest (thinned), u = 5"),
       col = c("#e74c3c", "#3498DB", "#27AE60", "#F39C12", "#114125", "#db08f3"),
       lty = c(1, 1, 1, 1, 2, 2),
       lwd = c(2, 2, 2, 2, 2, 2),
       bty = "n",
       cex = 0.9,
       inset = c(0.05, 0.05))
title("JumpForest w. and w.o. thinning (1740 vs. 50 event times)")
dev.off()

#nolint_end