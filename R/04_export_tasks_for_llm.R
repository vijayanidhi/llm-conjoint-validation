# 04_export_tasks_for_llm.R
# Converts the real camera CBC design (X matrices) into a clean JSON structure
# that the LLM pipeline can read. We reuse the REAL respondents' exact task
# designs (same attribute combinations they saw) so synthetic "respondents"
# face an apples-to-apples comparison -- only the decision-maker changes.

library(jsonlite)

camera <- readRDS("data/camera.rds")
attrs  <- colnames(camera[[1]]$X)   # canon sony nikon panasonic pixels zoom video swivel wifi price
n_alt  <- 5
n_task <- 16

# Turn one row of the X design matrix into a human-readable camera description
describe_alt <- function(row) {
  if (all(row == 0)) return(list(is_outside_good = TRUE, description = "None of these (no purchase)"))

  brand <- c("Canon","Sony","Nikon","Panasonic")[which(row[1:4] == 1)]
  if (length(brand) == 0) brand <- "Other brand"  # 5th baseline brand, if ever untagged

  feats <- c()
  if (row["pixels"]  == 1) feats <- c(feats, "higher pixel count")
  if (row["zoom"]    == 1) feats <- c(feats, "higher zoom")
  if (row["video"]   == 1) feats <- c(feats, "video capture capability")
  if (row["swivel"]  == 1) feats <- c(feats, "swivel display")
  if (row["wifi"]    == 1) feats <- c(feats, "wifi connectivity")

  price_usd <- row["price"] * 100   # price stored in hundreds of USD

  list(
    is_outside_good = FALSE,
    brand = brand,
    features = feats,
    price_usd = unname(price_usd),
    description = sprintf(
      "%s camera, $%.0f, with: %s",
      brand, price_usd, if (length(feats) == 0) "no additional features" else paste(feats, collapse = ", ")
    )
  )
}

# Build the full task list for ONE respondent's design (used as the template
# design for all synthetic respondents, OR loop over real respondents' own
# designs if you want each synthetic "twin" to face that respondent's exact tasks)
build_respondent_tasks <- function(resp) {
  X <- resp$X
  tasks <- vector("list", n_task)
  for (t in seq_len(n_task)) {
    rows <- ((t - 1) * n_alt + 1):(t * n_alt)
    alts <- lapply(rows, function(r) describe_alt(X[r, ]))
    tasks[[t]] <- list(task_number = t, alternatives = alts)
  }
  tasks
}

# Export: one JSON file per real respondent's design (so a synthetic "twin"
# can be run against the SAME 16 tasks that real respondent saw)
dir.create("data/llm_tasks", showWarnings = FALSE)
for (i in seq_along(camera)) {
  tasks <- build_respondent_tasks(camera[[i]])
  write_json(tasks, sprintf("data/llm_tasks/respondent_%03d_tasks.json", i),
             auto_unbox = TRUE, pretty = TRUE)
}

cat(sprintf("Exported %d respondent task files to data/llm_tasks/\n", length(camera)))

# Print one example so we can see what it looks like
cat("\n--- Example: respondent 1, task 1 ---\n")
print(build_respondent_tasks(camera[[1]])[[1]])
