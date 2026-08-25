camera <- readRDS("data/camera.rds")
all_y <- unlist(lapply(camera, function(r) r$y))
cat("Real data - proportion choosing each alternative (1-4=camera, 5=none):\n")
print(round(table(all_y) / length(all_y), 3))
