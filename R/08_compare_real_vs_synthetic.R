# 08_compare_real_vs_synthetic.R
real_res <- readRDS("output/pooled_mnl_baseline.rds")$results
syn_res  <- readRDS("output/synthetic_pooled_mnl.rds")$results

comp <- data.frame(
  attribute      = real_res$attribute,
  real_estimate  = real_res$estimate,
  synth_estimate = syn_res$estimate[match(real_res$attribute, syn_res$attribute)]
)
comp$diff <- round(comp$synth_estimate - comp$real_estimate, 3)

# Since raw scale differs (synthetic choices are much "sharper"/less noisy, so
# scale of all coefficients is inflated), also show each attribute's estimate
# relative to price -- this normalizes out the scale difference and lets us
# compare RELATIVE importance, which is more meaningful than raw magnitude.
comp$real_rel_to_price  <- round(comp$real_estimate  / abs(real_res$estimate[real_res$attribute == "price"]), 3)
comp$synth_rel_to_price <- round(comp$synth_estimate / abs(syn_res$estimate[syn_res$attribute == "price"]), 3)

cat("=== Real vs Synthetic part-worths (raw scale) ===\n")
print(comp[, c("attribute", "real_estimate", "synth_estimate", "diff")])

cat("\n=== Relative to price (scale-normalized comparison) ===\n")
print(comp[, c("attribute", "real_rel_to_price", "synth_rel_to_price")])

# Rank correlation: do real and synthetic agree on WHICH attributes matter most?
rank_cor <- cor(comp$real_estimate, comp$synth_estimate, method = "spearman")
cat(sprintf("\nSpearman rank correlation (real vs synthetic part-worths): %.3f\n", rank_cor))

write.csv(comp, "output/real_vs_synthetic_comparison.csv", row.names = FALSE)
