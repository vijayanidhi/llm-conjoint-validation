# 06_assemble_synthetic_camera.R
# Reassembles LLM synthetic-respondent JSON output (data/llm_responses/) back
# into the SAME list-of-lists {y, X} structure as bayesm::camera, so the
# exact same modeling scripts (02_fit_pooled_mnl.R, 03_fit_latent_class_mnl.R)
# can run on synthetic data with zero changes -- just point them at a
# different .rds file.

library(jsonlite)

camera <- readRDS("data/camera.rds")   # reuse real respondents' X designs
resp_files <- list.files("data/llm_responses", pattern = "synthetic_respondent_.*\\.json",
                          full.names = TRUE)

if (length(resp_files) == 0) {
  stop("No synthetic responses found in data/llm_responses/ -- run the Python pipeline first.")
}

synthetic_camera <- list()
for (f in resp_files) {
  resp <- fromJSON(f)
  rid  <- resp$respondent_id
  y    <- as.integer(resp$y)
  X    <- camera[[rid]]$X   # same design (choice sets) that real respondent rid saw
  synthetic_camera[[length(synthetic_camera) + 1]] <- list(y = y, X = X)
}

cat(sprintf("Assembled %d synthetic respondents into camera-format list.\n", length(synthetic_camera)))
saveRDS(synthetic_camera, "data/synthetic_camera.rds")
