# Modern Time Series Forecasting with R ----
# Marco Zanotti

# Lecture 3.2: Recursive Time Series Algorithms ----------------------------

# Goals:
# - Panel Recursivity



# Packages ----------------------------------------------------------------

source("src/R/utils.R")
source("src/R/install.R")



# RECURSIVITY - PANEL TIME SERIES -----------------------------------------

?recursive()
?panel_tail()



# Data --------------------------------------------------------------------

m4_tbl <- load_data("data/m4", "m4_prep_sample", ext = ".parquet")


# * Feature Engineering ---------------------------------------------------

m4_tbl |> plot_time_series(ds, y, .facet_var = unique_id, .facet_ncol = 2)

horizon <- 24 * 2
lags <- c(1, 2, 6, 12, 24)
rolling_periods <- c(24, 48)

data_prep_full_tbl <- m4_tbl |>
  # Extend each time series
  extend_timeseries(.id_var = unique_id, .date_var = ds, .length_future = horizon) |>
  group_by(unique_id) |>
  # Add lags
  tk_augment_lags(y, .lags = lag_period) |>
  # Add rolling features
  tk_augment_slidify(
    y_lag24,
    mean,
    .period = rolling_periods,
    .align = "center",
    .partial = TRUE
  ) |>
  tk_augment_fourier(ds, .periods = c(12, 24, 36, 48), .K = 2) |>
  # Reformat Columns
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .)) |>
  ungroup()

data_prep_full_tbl |>
  select(-matches("(sin)|(cos)")) |>
  group_by(unique_id) |>
  pivot_longer(cols = -c(unique_id, ds)) |>
  plot_time_series(ds, value, .color_var = name, .facet_var = unique_id, .facet_ncol = 2, .smooth = FALSE)
data_prep_full_tbl |> group_by(unique_id) |> slice_tail(n = horizon + 1)


# * Modelling & Forecast Data ---------------------------------------------

data_prep_tbl <- data_prep_full_tbl |>
  drop_na() 
forecast_tbl <- data_prep_full_tbl |>
  group_by(unique_id) |>
  slice_tail(n = horizon) |> 
  ungroup()


# * Train / Test Sets -----------------------------------------------------

splits <- time_series_split(data_prep_tbl, assess = horizon, cumulative = TRUE)
splits |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(ds, y)


# * Recipes ---------------------------------------------------------------

# Baseline Recipe
# - Time Series Signature - Adds bulk time-based features
rcp_spec_sf <- recipe(y ~ ., data = training(splits)) |>
  update_role(unique_id, new_role = "id variable") |>   # make it not a predictor
  step_timeseries_signature(ds) |>
  step_rm(matches("(iso)|(xts)|(minute)|(second)|(year)|(quarter)|(month)|(half)|(mday)|(qday)|(yday)")) |>
  step_normalize(matches("(index.num)")) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>  # only predictors
  step_naomit(starts_with("lag_"))

rcp_spec_ml <- rcp_spec_sf |> step_rm(ds)

rcp_spec_sf |> prep() |> juice() |> glimpse()
rcp_spec_ml |> prep() |> juice() |> glimpse()



# KNN ---------------------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_knn <- nearest_neighbor(
  mode = "regression",
  neighbors = 50,
  dist_power = 10,
  weight_func = "optimal"
) |>
  set_engine("kknn")

# * Workflows -------------------------------------------------------------

# KNN - Recursive
wrkfl_fit_knn_recursive <- workflow() |>
  add_model(model_spec_knn) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits)) |>
  recursive(
    id = "unique_id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), unique_id, horizon)
  )



# RANDOM FOREST -----------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_rf <- rand_forest(
  mode = "regression",
  mtry = 25,
  trees = 1000,
  min_n = 25
) |>
  set_engine("ranger")

# * Workflows -------------------------------------------------------------

# RF - Recursive
wrkfl_fit_rf_recursive <- workflow() |>
  add_model(model_spec_rf) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits)) |>
  recursive(
    id = "unique_id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), unique_id, horizon)
  )



# CUBIST ------------------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_cubist <- cubist_rules(
  committees = 50,
  neighbors = 7,
  max_rules = 100
) |>
  set_engine("Cubist")

# * Workflows -------------------------------------------------------------

# CUBIST - Recursive
wrkfl_fit_cubist_recursive <- workflow() |>
  add_model(model_spec_cubist) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits)) |>
  recursive(
    id = "unique_id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), unique_id, horizon)
  )



# NEURAL NETWORK ----------------------------------------------------------

# * Engines ---------------------------------------------------------------

model_spec_nnet <- mlp(
  mode = "regression",
  hidden_units = 10,
  penalty = 1,
  epochs = 100
) |>
  set_engine("nnet")

# * Workflows -------------------------------------------------------------

# NNET - Recursive
wrkfl_fit_nnet_recursive <- workflow() |>
  add_model(model_spec_nnet) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits)) |>
  recursive(
    id = "unique_id",
    transform = lag_transf_grouped, 
    train_tail = panel_tail(training(splits), unique_id, horizon)
  )



# Recursive Models' Performance -------------------------------------------

# * Evaluation
calibration_tbl <- modeltime_table(
  wrkfl_fit_knn_recursive,
  wrkfl_fit_rf_recursive,
  wrkfl_fit_cubist_recursive,
  wrkfl_fit_nnet_recursive
) |>
  modeltime_calibrate(testing(splits), id = "unique_id")

calibration_tbl |>
  modeltime_accuracy(acc_by_id = TRUE) |>
  arrange(unique_id)

calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl, keep_data = TRUE) |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue", .facet_ncol = 2)

# * Refitting & Forecasting
refit_tbl <- calibration_tbl |> modeltime_refit(data = data_prep_tbl)

refit_tbl |>
  modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl, keep_data = TRUE) |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue", .facet_ncol = 2)

