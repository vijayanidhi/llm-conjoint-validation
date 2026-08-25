# 01_load_camera.R
# Load and inspect the bayesm camera CBC dataset (real human choice data, primary validation set)

library(bayesm)
data(camera)

n_resp   <- length(camera)
n_tasks  <- length(camera[[1]]$y)          # 16 per respondent
n_alt    <- nrow(camera[[1]]$X) / n_tasks  # 5 (4 cameras + outside option)
attrs    <- colnames(camera[[1]]$X)

cat(sprintf("Respondents: %d | Tasks/resp: %d | Alts/task: %d | Total tasks: %d\n",
            n_resp, n_tasks, n_alt, n_resp * n_tasks))
cat("Attributes:", paste(attrs, collapse = ", "), "\n")

saveRDS(camera, "data/camera.rds")
