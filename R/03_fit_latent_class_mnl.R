# 03_fit_latent_class_mnl.R (vectorized)
# Finite-mixture latent-class MNL (Kamakura & Russell 1989 style) via EM.
# Respondent-level class membership; segments have distinct part-worths.

camera <- readRDS("data/camera.rds")
n_alt  <- 5
attrs  <- colnames(camera[[1]]$X)
K      <- length(attrs)
N      <- length(camera)
n_task <- length(camera[[1]]$y)          # 16, constant across respondents

# Stack everything into flat matrices for vectorized ops.
# X_all: (N*n_task*n_alt) x K   |  y_all: one row index per task (which alt, 1..n_alt)
X_all   <- do.call(rbind, lapply(camera, function(r) r$X))
y_all   <- unlist(lapply(camera, function(r) r$y))                  # length N*n_task
resp_id <- rep(seq_len(N), each = n_task)                            # length N*n_task
task_id <- seq_len(N * n_task)

n_rows_total <- nrow(X_all)                       # N*n_task*n_alt
alt_task_id  <- rep(task_id, each = n_alt)         # which task each row belongs to
chosen_row   <- (task_id - 1) * n_alt + y_all      # row index of chosen alt per task

# Given beta (K-vector), returns vector of per-task log-likelihood (length N*n_task)
task_loglik_vec <- function(beta) {
  v <- as.vector(X_all %*% beta)
  m <- ave(v, alt_task_id, FUN = max)              # per-task max, for stability
  ev <- exp(v - m)
  denom <- ave(ev, alt_task_id, FUN = sum)
  logp <- (v - m) - log(denom)
  logp[chosen_row]
}

# Weighted negative log-lik and analytic gradient for a class, given respondent weights w (length N)
make_weighted_negLL <- function(w) {
  w_task <- w[resp_id]      # expand respondent weights to task level
  function(beta) {
    v <- as.vector(X_all %*% beta)
    m <- ave(v, alt_task_id, FUN = max)
    ev <- exp(v - m)
    denom <- ave(ev, alt_task_id, FUN = sum)
    p <- ev / denom                                  # choice probs, all rows
    logp_chosen <- (v[chosen_row] - m[chosen_row]) - log(denom[chosen_row])
    nll <- -sum(w_task * logp_chosen)

    # gradient: sum_task w * (X_chosen - sum_alt p*X_alt)
    Xp <- X_all * p                                   # weight each row by its choice prob
    # sum Xp over alternatives within each task -> expected X per task
    EX <- rowsum(Xp, alt_task_id)                      # n_task_total x K
    Xchosen <- X_all[chosen_row, , drop = FALSE]
    grad_per_task <- Xchosen - EX
    grad <- -colSums(w_task * grad_per_task)
    attr(nll, "gradient") <- grad
    nll
  }
}

fit_lc_mnl <- function(n_classes, n_restarts = 2, max_iter = 60, tol = 1e-5, seed = 1) {
  set.seed(seed)
  best <- NULL
  for (restart in seq_len(n_restarts)) {
    beta   <- matrix(rnorm(n_classes * K, sd = 0.3), nrow = n_classes)
    pi_cls <- rep(1 / n_classes, n_classes)
    ll_old <- -Inf

    for (iter in seq_len(max_iter)) {
      # E-step
      task_ll_mat <- matrix(NA, nrow = N * n_task, ncol = n_classes)
      for (c in 1:n_classes) task_ll_mat[, c] <- task_loglik_vec(beta[c, ])
      # sum to respondent level
      resp_ll_mat <- apply(task_ll_mat, 2, function(col) tapply(col, resp_id, sum))
      resp_ll_mat <- matrix(resp_ll_mat, nrow = N)

      log_w <- sweep(resp_ll_mat, 2, log(pi_cls), "+")
      mrow  <- apply(log_w, 1, max)
      denom <- mrow + log(rowSums(exp(log_w - mrow)))
      r_ic  <- exp(log_w - denom)
      ll_total <- sum(denom)

      if (abs(ll_total - ll_old) < tol) break
      ll_old <- ll_total

      pi_cls <- colMeans(r_ic)

      for (c in 1:n_classes) {
        negLL_fn <- make_weighted_negLL(r_ic[, c])
        opt <- optim(beta[c, ], negLL_fn,
                     gr = function(b) attr(negLL_fn(b), "gradient"),
                     method = "BFGS", control = list(maxit = 100))
        beta[c, ] <- opt$par
      }
    }

    n_params <- n_classes * K + (n_classes - 1)
    bic <- -2 * ll_total + n_params * log(N)
    cat(sprintf("  [%d classes | restart %d] logLik=%.2f, iters=%d, BIC=%.2f\n",
                n_classes, restart, ll_total, iter, bic))

    if (is.null(best) || ll_total > best$loglik) {
      best <- list(beta = beta, pi = pi_cls, loglik = ll_total, bic = bic,
                   r_ic = r_ic, n_classes = n_classes)
    }
  }
  best
}

results_by_k <- list()
for (k in 2:4) {
  cat(sprintf("Fitting %d-class latent class MNL...\n", k))
  t0 <- Sys.time()
  results_by_k[[as.character(k)]] <- fit_lc_mnl(k, n_restarts = 2, seed = 42 + k)
  cat("  time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "sec\n")
}

for (k in names(results_by_k)) {
  r <- results_by_k[[k]]
  cat(sprintf("\n=== %s classes: logLik=%.2f, BIC=%.2f, class shares=%s ===\n",
              k, r$loglik, r$bic, paste(round(r$pi, 3), collapse = ", ")))
  rownames(r$beta) <- paste0("class", seq_len(r$n_classes))
  colnames(r$beta) <- attrs
  print(round(r$beta, 3))
}

saveRDS(results_by_k, "output/latent_class_mnl.rds")
