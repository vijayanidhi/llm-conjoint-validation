# 09_pooling_test.R
# Formal likelihood-ratio test of whether real and synthetic respondents can
# be treated as coming from the same preference structure ("pooled") or
# whether their part-worths are significantly different ("separate").
#
# H0 (pooled): one shared beta explains both real and synthetic choices
# H1 (separate): real and synthetic each get their own beta
# LR = 2 * (LL_separate - LL_pooled), df = K (extra params in H1)

real_camera <- readRDS("data/camera.rds")
synth_camera <- readRDS("data/synthetic_camera.rds")
n_alt <- 5
attrs <- colnames(real_camera[[1]]$X)
K <- length(attrs)

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

beta0 <- rep(0, K); names(beta0) <- attrs

# LL for real-only and synthetic-only (already fit in scripts 02 and 07, refit here for a clean self-contained script)
fit_real  <- optim(beta0, negLL, data = real_camera,  method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
fit_synth <- optim(beta0, negLL, data = synth_camera, method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
LL_separate <- -(fit_real$value + fit_synth$value)

# LL for pooled model: combine both datasets, fit ONE beta
combined <- c(real_camera, synth_camera)
fit_pooled <- optim(beta0, negLL, data = combined, method = "BFGS", control = list(maxit = 500, reltol = 1e-10))
LL_pooled <- -fit_pooled$value

LR_stat <- 2 * (LL_separate - LL_pooled)
df <- K   # separate model has 2K params vs pooled model's K params
p_value <- pchisq(LR_stat, df = df, lower.tail = FALSE)

cat(sprintf("LL (real-only):       %.2f\n", -fit_real$value))
cat(sprintf("LL (synthetic-only):  %.2f\n", -fit_synth$value))
cat(sprintf("LL (separate, summed):%.2f\n", LL_separate))
cat(sprintf("LL (pooled, shared beta): %.2f\n", LL_pooled))
cat(sprintf("\nLikelihood-ratio statistic: %.2f (df = %d)\n", LR_stat, df))
cat(sprintf("p-value: %s\n", format.pval(p_value, digits = 4)))

if (p_value < 0.001) {
  cat("\nResult: STRONGLY reject H0 (pooling). Real and synthetic respondents\n")
  cat("have significantly different preference structures (p < 0.001).\n")
} else if (p_value < 0.05) {
  cat("\nResult: Reject H0 (pooling) at the 5% level.\n")
} else {
  cat("\nResult: Fail to reject H0 -- no significant evidence that real and\n")
  cat("synthetic preferences differ.\n")
}

cat("\nNote: this test does not separately identify TASTE differences from\n")
cat("SCALE (noise-level) differences between the two populations -- both\n")
cat("contribute to rejecting pooling here. A Swait-Louviere style test that\n")
cat("explicitly estimates a relative scale parameter would isolate genuine\n")
cat("taste differences from response-consistency differences, and is a\n")
cat("natural refinement for the final writeup.\n")

saveRDS(list(LL_real = -fit_real$value, LL_synth = -fit_synth$value,
             LL_separate = LL_separate, LL_pooled = LL_pooled,
             LR_stat = LR_stat, df = df, p_value = p_value),
        "output/pooling_test.rds")
