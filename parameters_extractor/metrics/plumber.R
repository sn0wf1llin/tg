path_r <- Sys.getenv("EXPONENTA_PROJECT_PATH")
library(plumber)
r <- plumb(paste0(path_r, 'parameters_extractor/metrics/api.R'))
r$run(port=8000)
