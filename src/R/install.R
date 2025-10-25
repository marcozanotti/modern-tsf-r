# Packages
pkgs <- c(
  "devtools", "remotes",
  "tidyverse", 
  "timetk", 
  "forecast", "prophet", "smooth", "thief",
  "glmnet", "earth", "kernlab", "kknn",
  "randomForest", "ranger", "xgboost", "bonsai", "lightgbm",
  "Cubist", "rules",
  "tidymodels", "modeltime", "modeltime.ensemble",
  "parallel", "doFuture", "tictoc",
  "reticulate"
)
install_and_load(pkgs)

# Install CatBoost from source for Linux
# devtools::install_url(
#   "https://github.com/catboost/catboost/releases/download/v1.0.0/catboost-R-Linux-1.0.0.tgz",
#   INSTALL_opts = c("--no-multiarch", "--no-test-load")
# )

# Install packages from GitHub
# remotes::install_github("business-science/modeltime.gluonts")
# remotes::install_github("business-science/modeltime.h2o")
