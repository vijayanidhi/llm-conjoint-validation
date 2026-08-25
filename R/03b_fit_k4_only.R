camera <- readRDS("data/camera.rds")
n_alt  <- 5
attrs  <- colnames(camera[[1]]$X)
K      <- length(attrs)
N      <- length(camera)
n_task <- length(camera[[1]]$y)

X_all   <- do.call(rbind, lapply(camera, function(r) r$X))
y_all   <- unlist(lapply(camera, function(r) r$y))
resp_id <- rep(seq_len(N), each = n_task)
task_id <- seq_len(N * n_task)
alt_task_id <- rep(task_id, each = n_alt)
chosen_row  <- (task_id - 1) * n_alt + y_all

task_loglik_vec <- function(beta) {
  v <- as.vector(X_all %*% beta)
  m <- ave(v, alt_task_id, FUN = max)
  ev <- exp(v - m)
  denom <- ave(ev, alt_task_id, FUN = sum)
  logp <- (v - m) - log(denom)
  logp[chosen_row]
}

make_weighted_negLL <- function(w) {
  w_task <- w[resp_id]
  function(beta) {
    v <- as.vector(X_all %*% beta)
    m <- ave(v, alt_task_id, FUN = max)
    ev <- exp(v - m)
    denom <- ave(ev, alt_task_id, FUN = sum)
    p <- ev / denom
    logp_chosen <- (v[chosen_row] - m[chosen_row]) - log(denom[chosen_row])
    nll <- -sum(w_task * logp_chosen)
    Xp <- X_all * p
    EX <- rowsum(Xp, alt_task_id)
    Xchosen <- X_all[chosen_row, , drop = FALSE]
    grad_per_task <- Xchosen - EX
    grad <- -colSums(w_task * grad_per_task)
    attr(nll, "gradient") <- grad
    nll
  }
}

fit_lc_mnl <- function(n_classes, n_restarts = 1, max_iter = 200, tol = 1e-5, seed = 1) {
  set.seed(seed)
  best <- NULL
  for (restart in seq_len(n_restarts)) {
    beta   <- matrix(rnorm(n_classes * K, sd = 0.3), nrow = n_classes)
    pi_cls <- rep(1 / n_classes, n_classes)
    ll_old <- -Inf
    for (iter in seq_len(max_iter)) {
      task_ll_mat <- matrix(NA, nrow = N * n_task, ncol = n_classes)
      for (c in 1:n_classes) task_ll_mat[, c] <- task_loglik_vec(beta[c, ])
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

cat("Fitting 4-class latent class MNL (1 restart, background)...\n")
res4 <- fit_lc_mnl(4, n_restarts = 1, seed = 46)
rownames(res4$beta) <- paste0("class", 1:4)
colnames(res4$beta) <- attrs
cat(sprintf("\n4 classes: logLik=%.2f, BIC=%.2f, shares=%s\n",
            res4$loglik, res4$bic, paste(round(res4$pi,3), collapse=", ")))
print(round(res4$beta, 3))
saveRDS(res4, "output/latent_class_mnl_k4.rds")
