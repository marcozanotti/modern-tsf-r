# Modern Time Series Forecasting with R ----
# Marco Zanotti

# Lecture 2.5: Recursive Time Series Algorithms ----------------------------

# Goals:
# - Recursivity
# - Panel Recursivity



# Packages ----------------------------------------------------------------

source("src/R/utils.R")
source("src/R/install.R")



# RECURSIVITY - SINGLE TIME SERIES ----------------------------------------

?recursive()



# Data & Artifacts --------------------------------------------------------

email_tbl <- load_data("data/email", "email_prep", ext = ".parquet")


# * Data Preparation ------------------------------------------------------

# email data
email_tbl |> plot_time_series(ds, log1p(y), .smooth = FALSE)

email_prep_tbl <- email_tbl |>
  mutate(y = log_interval_vec(y, limit_lower = 0, offset = 1)) |>
  mutate(y = standardize_vec(y)) |>
  filter_by_time(.start_date = "2018-07-03") |>
  mutate(y_cleaned = ts_clean_vec(y, period = 7)) |>
  mutate(
    y = ifelse(
      ds |> between_time("2018-11-18", "2018-11-20"),
      y_cleaned,
      y
    )
  ) |>
  select(ds, y, promo)


# * Feature Engineering ---------------------------------------------------

# - Extend to Future Window
# - Add any lags to full dataset
# - Add any external regressors to full dataset

horizon <- 7 * 8
lags <- c(1, 2, 7, 14, 30)

data_prep_full_tbl <- email_prep_tbl |>
  future_frame(.data = _, ds, .length_out = horizon, .bind_data = TRUE) |>
  mutate(promo = ifelse(is.na(promo), 0, promo)) |>
  lag_transf() |>
  tk_augment_lags(y, .lags = c(horizon, 90))

data_prep_full_tbl |>
  pivot_longer(-ds) |>
  plot_time_series(ds, value, name, .smooth = FALSE)
data_prep_full_tbl |> slice_tail(n = horizon + 1)


# * Modelling & Forecast Data ---------------------------------------------

data_prep_tbl <- data_prep_full_tbl |>
  slice_head(n = nrow(data_prep_full_tbl) - horizon) |>
  drop_na()
forecast_tbl <- data_prep_full_tbl |> slice_tail(n = horizon)


# * Train / Test Sets -----------------------------------------------------

splits <- time_series_split(data_prep_tbl, assess = horizon, cumulative = TRUE)
splits |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(ds, y)


# * Recipes ---------------------------------------------------------------

# Recipe with calendar features, short-term and long-term dynamics
rcp_spec <- recipe(y ~ ., data = training(splits)) |>
  step_timeseries_signature(ds) |>
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) |>
  step_normalize(matches("(index.num)|(year)|(yday)")) |>
  step_dummy(all_nominal(), one_hot = TRUE) |>
  step_rm(ds)
rcp_spec |> prep() |> juice() |> glimpse()



# LINEAR REGRESSION -------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_lm <- linear_reg() |>
  set_engine("lm")


# * Workflows -------------------------------------------------------------

# LM - Non Recursive
wrkfl_fit_lm <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec) |>
  fit(training(splits))

# LM - Recursive
wrkfl_fit_lm_recursive <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    transform = lag_transf,
    train_tail = tail(training(splits), horizon)
  )



# ELASTIC NET -------------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_elanet <- linear_reg(
  mode = "regression",
  penalty = 0.01,
  mixture = 0.99
) |>
  set_engine("glmnet")


# * Workflows -------------------------------------------------------------

# ELASTIC NET - Recursive
wrkfl_fit_elanet_recursive <- workflow() |>
  add_model(model_spec_elanet) |>
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    transform = lag_transf,
    train_tail = tail(training(splits), horizon)
  )



# SVM ---------------------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_svm <- svm_rbf(
  mode = "regression",
  cost = 1,
  rbf_sigma = 0.01,
  margin = 0.1
) |>
  set_engine("kernlab")


# * Workflows -------------------------------------------------------------

# SVM - Recursive
wrkfl_fit_svm_recursive <- workflow() |>
  add_model(model_spec_svm) |>
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    transform = lag_transf,
    train_tail = tail(training(splits), horizon)
  )



# BOOSTING ----------------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_xgb <- boost_tree(
  mode = "regression",
  mtry = 25,
  trees = 1000,
  min_n = 2,
  tree_depth = 12,
  learn_rate = 0.3,
  loss_reduction = 0
) |>
  set_engine("xgboost")


# * Workflows -------------------------------------------------------------

# XGBOOST - Recursive
wrkfl_fit_xgb_recursive <- workflow() |>
  add_model(model_spec_xgb) |>
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    transform = lag_transf,
    train_tail = tail(training(splits), horizon)
  )



# ENSEMBLES ---------------------------------------------------------------

# * Workflows -------------------------------------------------------------

# ELANET - NON RECURSIVE!!!
wrkfl_fit_elanet <- workflow() |>
  add_model(model_spec_elanet) |>
  add_recipe(rcp_spec) |>
  fit(training(splits))

# SVM - NON RECURSIVE!!!
wrkfl_fit_svm <- workflow() |>
  add_model(model_spec_svm) |>
  add_recipe(rcp_spec) |>
  fit(training(splits))

# WEIGHTED ENSEMBLE - RECURSIVE
ensemble_fit_mean_recursive <- modeltime_table(
  wrkfl_fit_elanet,
  wrkfl_fit_svm
) |>
  ensemble_average(type = "mean") |>
  recursive(
    transform = lag_transf,
    train_tail = tail(training(splits), horizon)
  )



# Recursive Models' Performance -------------------------------------------

# * Evaluation
calibration_tbl <- modeltime_table(
  wrkfl_fit_lm,
  wrkfl_fit_lm_recursive,
  wrkfl_fit_elanet_recursive,
  wrkfl_fit_svm_recursive,
  wrkfl_fit_xgb_recursive,
  ensemble_fit_mean_recursive
) |>
  modeltime_calibrate(testing(splits))

calibration_tbl |>
  modeltime_accuracy()

calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue")

# * Refitting & Forecasting
refit_tbl <- calibration_tbl |>
  modeltime_refit(data = data_prep_tbl)

refit_tbl |>
  modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue")
