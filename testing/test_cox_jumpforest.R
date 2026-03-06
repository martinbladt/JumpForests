devtools::load_all()
library(survival)

# 1. Setup Parameters
set.seed(1)
n <- 1000
beta <- 0.8
lambda <- 0.01
rho <- 1.5

# 2. Generate Continuous Covariate and Survival Times
x <- rnorm(n, mean = 0, sd = 1)
u <- runif(n)
y_true <- (-log(u) / (lambda * exp(beta * x)))^(1 / rho)
censor <- rexp(n, rate = 1 / (10 * median(y_true)))
time <- pmin(y_true, censor)
status <- as.integer(y_true <= censor)

df <- data.frame(time = time, status = status, x = x)

# 3. Fit the Cox Proportional Hazards Model
fit <- coxph(Surv(time, status) ~ x, data = df)

# 4. Build two-state multi-state data for JumpForests and fit forest
mm_data <- lapply(seq_len(n), function(i) {
  end_state <- if (df$status[i] == 1) 2L else 1L
  list(times = c(0, df$time[i]), states = c(1L, end_state))
})
feature_data <- data.frame(x = df$x)

jf_fit <- jfforest(
  MM ~ x,
  data = mm_data,
  feature_data = feature_data,
  ntrees = 5000,
  mtry = 1,
  min_node_size = 100,
  splitrule = "logrank",
  seed = 1,
  nworkers = 0,
  save_predictions = FALSE
)

# helper to convert predicted Nelson-Aalen to AJ survival in state 1
aj_from_lambda <- function(lambda_list, a0 = diag(nrow(lambda_list[[1]]))) {
  out <- vector("list", length(lambda_list))
  out[[1]] <- a0
  if (length(lambda_list) >= 2) {
    for (i in 2:length(lambda_list)) {
      delta <- lambda_list[[i]] - lambda_list[[i - 1]]
      out[[i]] <- out[[i - 1]] + as.vector(out[[i - 1]] %*% delta) - out[[i - 1]] * rowSums(delta)
    }
  }
  out
}
extract_surv_state1 <- function(lambda_list) {
  aj <- aj_from_lambda(lambda_list)
  vapply(aj, function(M) M[1, 1], numeric(1))
}

# 5. Define Covariate Values to Visualize
x_vals <- quantile(df$x, probs = c(0.1, 0.5, 0.9))
jf_pred <- jfforest.predict(jf_fit, new_data = data.frame(x = as.numeric(x_vals)))

# 6. Create the Plot
plot(NULL, xlim = c(0, max(df$time)), ylim = c(0, 1),
     xlab = "", ylab = "",
     main = "")
grid(lty = "dotted", col = "gray80")

for (i in seq_along(x_vals)) {
  val <- x_vals[i]
  t_grid <- seq(0, max(df$time), length.out = 100)

  # TRUE survival
  true_s <- exp(-lambda * t_grid^rho * exp(beta * val))
  lines(t_grid, true_s, col = "#ED9912", lty = 1, lwd = 2)

  # Cox fitted survival
  surv_fit <- survfit(fit, newdata = data.frame(x = val))
  lines(surv_fit$time, surv_fit$surv, col = "gray80", lty = 3, lwd = 2)

  # JumpForest fitted survival (MM prediction)
  jf_surv <- extract_surv_state1(jf_pred[[i]])
  lines(jf_fit$unique.event.times, jf_surv, col = "#377EB8", lty = 2, lwd = 2)
}

# 7. Add Legend
legend("topright",
       legend = c("True", "Fitted (CoxPH)", "Fitted (JumpForest)"),
       col = c("#ED9912", "gray80", "#377EB8"),
       lty = c(1, 3, 2),
       lwd = 2, bty = "n", cex = 0.8)

# 8. Non-proportional hazards example (analogous workflow)
set.seed(1)
n_np <- 1000
tau <- 4
lambda1 <- 0.10
lambda2 <- 0.10
beta1 <- 1.2
beta2 <- -1.0

x_np <- rnorm(n_np, mean = 0, sd = 1)
e_np <- rexp(n_np, rate = 1)
rate1 <- lambda1 * exp(beta1 * x_np)
rate2 <- lambda2 * exp(beta2 * x_np)
H_tau <- rate1 * tau
y_np_true <- ifelse(e_np <= H_tau, e_np / rate1, tau + (e_np - H_tau) / rate2)
censor_np <- rexp(n_np, rate = 1 / (10 * median(y_np_true)))
time_np <- pmin(y_np_true, censor_np)
status_np <- as.integer(y_np_true <= censor_np)

df_np <- data.frame(time = time_np, status = status_np, x = x_np)
fit_np <- coxph(Surv(time, status) ~ x, data = df_np)

mm_data_np <- lapply(seq_len(n_np), function(i) {
  end_state <- if (df_np$status[i] == 1) 2L else 1L
  list(times = c(0, df_np$time[i]), states = c(1L, end_state))
})
feature_data_np <- data.frame(x = df_np$x)

jf_fit_np <- jfforest(
  MM ~ x,
  data = mm_data_np,
  feature_data = feature_data_np,
  ntrees = 5000,
  mtry = 1,
  min_node_size = 100,
  splitrule = "logrank",
  seed = 1,
  nworkers = 0,
  save_predictions = FALSE
)

x_vals_np <- quantile(df_np$x, probs = c(0.1, 0.5, 0.9))
jf_pred_np <- jfforest.predict(jf_fit_np, new_data = data.frame(x = as.numeric(x_vals_np)))

plot_max_np <- as.numeric(quantile(df_np$time, probs = 0.98))
plot(NULL, xlim = c(0, plot_max_np), ylim = c(0, 1),
     xlab = "", ylab = "",
     main = "")
grid(lty = "dotted", col = "gray80")

t_grid_np <- seq(0, plot_max_np, length.out = 120)
for (i in seq_along(x_vals_np)) {
  val <- x_vals_np[i]

  true_s_np <- ifelse(
    t_grid_np <= tau,
    exp(-lambda1 * exp(beta1 * val) * t_grid_np),
    exp(-lambda1 * exp(beta1 * val) * tau - lambda2 * exp(beta2 * val) * (t_grid_np - tau))
  )
  lines(t_grid_np, true_s_np, col = "#ED9912", lty = 1, lwd = 2)

  surv_fit_np <- survfit(fit_np, newdata = data.frame(x = val))
  lines(surv_fit_np$time, surv_fit_np$surv, col = "gray80", lty = 3, lwd = 2)

  jf_surv_np <- extract_surv_state1(jf_pred_np[[i]])
  lines(jf_fit_np$unique.event.times, jf_surv_np, col = "#377EB8", lty = 2, lwd = 2)
}

legend("topright",
       legend = c("True", "Fitted (CoxPH)", "Fitted (JumpForest)"),
       col = c("#ED9912", "gray80", "#377EB8"),
       lty = c(1, 3, 2),
       lwd = 2, bty = "n", cex = 0.8)
