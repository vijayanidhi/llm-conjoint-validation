# 07_fit_synthetic_pooled_mnl.R
# Pooled MNL fit on the LLM SYNTHETIC respondent data, using the identical
# spec as 02_fit_pooled_mnl.R (real data), for direct comparison.

synthetic_camera <- readRDS("data/synthetic_camera.rds")
n_alt  <- 5
attrs  <- colnames(synthetic_camera[[1]]$X)
K      <- length(attrs)

negLL <- function(beta, data) {
  ll <- 0
  for (resp in data) {
    X <- resp$X; y <- resp$y; n_task <- length(y)
    for (t in seq_len(n_task)) {
      rows <- ((t - 1) * n_alt + 1):(t * n_alt)
      Xt <- X[rows, , drop = FALSE]
      v  <- as.vector(Xt %*% beta)
      v  <- v - max(v)
      p  <- exp(v) / sum(exp(v))
      ll <- ll + log(p[y[t]])
    }
  }
  -ll
}

beta_start <- rep(0, K); names(beta_start) <- attrs

fit <- optim(beta_start, negLL, data = synthetic_camera, method = "BFGS",
             hessian = TRUE, control = list(maxit = 500, reltol = 1e-10))

se <- sqrt(diag(solve(fit$hessian)))
z  <- fit$par / se
p_val <- 2 * pnorm(-abs(z))

results <- data.frame(attribute = attrs, estimate = round(fit$par, 4),
                       se = round(se, 4), z = round(z, 3), p_value = round(p_val, 4))

cat("Convergence code (0 = success):", fit$convergence, "\n")
cat("Final negative log-likelihood:", round(fit$value, 2), "\n\n")
print(results)

n_resp <- length(synthetic_camera)
bic <- 2 * fit$value + K * log(n_resp)
cat(sprintf("\nSynthetic pooled MNL: logLik=%.2f, BIC=%.2f\n", -fit$value, bic))

# outside-good (option 5, "none") choice rate check
all_y <- unlist(lapply(synthetic_camera, function(r) r$y))
cat("\nSynthetic data - proportion choosing each alternative:\n")
print(round(table(all_y) / length(all_y), 3))

saveRDS(list(fit = fit, results = results), "output/synthetic_pooled_mnl.rds")
write.csv(results, "output/synthetic_pooled_mnl.csv", row.names = FALSE)
