# 02_fit_pooled_mnl.R
# Pooled (homogeneous) conditional logit / MNL baseline on the real camera CBC data.
# This is our simplest ground-truth benchmark: one set of part-worths for the whole sample.

camera <- readRDS("data/camera.rds")
n_alt  <- 5
attrs  <- colnames(camera[[1]]$X)
K      <- length(attrs)

# Negative log-likelihood for a pooled MNL across all respondents/tasks.
# Each task has n_alt alternatives; y indexes which alternative (1..n_alt) was chosen.
negLL <- function(beta, data) {
  ll <- 0
  for (resp in data) {
    X <- resp$X
    y <- resp$y
    n_task <- length(y)
    for (t in seq_len(n_task)) {
      rows <- ((t - 1) * n_alt + 1):(t * n_alt)
      Xt <- X[rows, , drop = FALSE]
      v  <- as.vector(Xt %*% beta)
      v  <- v - max(v)                 # numerical stability
      p  <- exp(v) / sum(exp(v))
      ll <- ll + log(p[y[t]])
    }
  }
  -ll
}

beta_start <- rep(0, K)
names(beta_start) <- attrs

fit <- optim(
  par     = beta_start,
  fn      = negLL,
  data    = camera,
  method  = "BFGS",
  hessian = TRUE,
  control = list(maxit = 500, reltol = 1e-10)
)

se       <- sqrt(diag(solve(fit$hessian)))
z        <- fit$par / se
p_val    <- 2 * pnorm(-abs(z))

results <- data.frame(
  attribute = attrs,
  estimate  = round(fit$par, 4),
  se        = round(se, 4),
  z         = round(z, 3),
  p_value   = round(p_val, 4)
)

cat("Convergence code (0 = success):", fit$convergence, "\n")
cat("Final negative log-likelihood:", round(fit$value, 2), "\n\n")
print(results)

saveRDS(list(fit = fit, results = results), "output/pooled_mnl_baseline.rds")
write.csv(results, "output/pooled_mnl_baseline.csv", row.names = FALSE)

# BIC for the pooled model, for comparison against latent-class fits
n_params_pooled <- length(fit$par)
n_resp <- length(camera)
bic_pooled <- 2 * fit$value + n_params_pooled * log(n_resp)
cat(sprintf("\nPooled MNL: logLik=%.2f, BIC=%.2f\n", -fit$value, bic_pooled))
