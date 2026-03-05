#nolint start: line_length_linter
devtools::load_all()
library(AalenJohansen)

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

n <- 10000
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

landmark <- sim[unlist(lapply(sim, function(z) {
  any(z$times <= 10 & c(z$times[-1], Inf) > 10 & z$states == 2)
}))]
landmark <- lapply(landmark, function(z) {
  duration <- 10 - z$times[z$times <= 10 & c(z$times[-1], Inf) > 10 & z$states == 2]
  list(times = z$times, states = as.integer(z$states), X = duration)
})

u1 <- 1
u2 <- 5
fit1 <- aalen_johansen(landmark, x = u1)
fit2 <- aalen_johansen(landmark, x = u2)
fit3 <- aalen_johansen(landmark)

v10 <- fit1$t
v20 <- fit2$t
v30 <- fit3$t
p1 <- unlist(lapply(fit1$p, function(L) L[2]))
p2 <- unlist(lapply(fit2$p, function(L) L[2]))
p3 <- unlist(lapply(fit3$p, function(L) L[2]))
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
  min_node_size = 50,
  nsplits = 10,
  ntrees = 200,
  splitrule = "logrank",
  seed = 2026,
  nworkers = 0,
  save_predictions = FALSE
)
cond_pred <- jfforest.predict(cond_forest, data.frame(duration = c(u1, u2)))
p_rf_u1 <- extract_prob(cond_pred[[1]], from = 2, to = 2)
p_rf_u2 <- extract_prob(cond_pred[[2]], from = 2, to = 2)

pdf(plot_file("p2_conditional_jumpforest.pdf"), width = 6, height = 6)
plot(v10, p1, type = "l", lty = 2, xlab = "", ylab = "",
     col = "#e74c3c", xlim = c(10, 40), lwd = 2)
lines(seq(10, 40, 0.1), P(seq(10, 40, 0.1), u1), lwd = 2, col = "#e74c3c")
lines(v20, p2, lty = 2, lwd = 2, col = "#3498DB")
lines(seq(10, 40, 0.1), P(seq(10, 40, 0.1), u2), lwd = 2, col = "#3498DB")
lines(v30, p3, lty = 3, lwd = 2)
lines(cond_forest$unique.event.times, p_rf_u1, lwd = 2, col = "#27AE60")
lines(cond_forest$unique.event.times, p_rf_u2, lwd = 2, col = "#F39C12")
legend("topright",
       legend = c("Landmark", "True, u = 1", "True, u = 5", "Conditional AJ, u = 1", "Conditional AJ, u = 5", "JumpForest, u = 1", "JumpForest, u = 5"),
       col = c("black", "#e74c3c", "#3498DB", "#e74c3c", "#3498DB", "#27AE60", "#F39C12"),
       lty = c(3, 1, 1, 2, 2, 1, 1),
       lwd = c(2, 2, 2, 2, 2, 2, 2),
       bty = "n",
       cex = 0.9,
       inset = c(0.05, 0.05))
dev.off()

#nolint end
