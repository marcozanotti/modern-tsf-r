# Modern Time Series Forecasting with R ----

# Lecture 11: Panel Time Series Forecasting -------------------------------
# Marco Zanotti

# Goals:
# - Nested Forecasting
# - Nested Forecasting with many models
# - Global Modelling
# - Global Modelling with many models



# Packages ----------------------------------------------------------------

source("src/R/utils.R")
source("src/R/install.R")


# Data --------------------------------------------------------------------

m4_tbl <- load_data("data/m4", "m4_prep_sample", ext = ".parquet")


# * Feature Engineering ---------------------------------------------------

?extend_timeseries

m4_tbl |> plot_time_series(ds, y, .facet_var = unique_id, .facet_ncol = 2)

horizon <- 24 * 2
lag_period <- 24 * 2
rolling_periods <- c(24, 48, 96)

data_prep_full_tbl <- m4_tbl |>
  # Extend each time series
  extend_timeseries(.id_var = unique_id, .date_var = ds, .length_future = horizon) |>
  group_by(unique_id) |>
  # Add lags
  tk_augment_lags(y, .lags = lag_period) |>
  # Add rolling features
  tk_augment_slidify(
    y_lag48,
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



# NESTED FORECASTING ------------------------------------------------------

# Modeltime Nested Data Workflow ------------------------------------------

# * Modelling & Forecast Data ---------------------------------------------

?nest_timeseries

nested_data_tbl <- data_prep_full_tbl |>
  # Split into actual data & forecast data
  nest_timeseries(.id_var = unique_id, .length_future = horizon)

# from now on, we will work with list columns (that is nested data)


# * Train / Test Sets -----------------------------------------------------

?split_nested_timeseries
?extract_nested_train_split
?extract_nested_test_split

# splits <- time_series_split(data_prep_tbl, assess = horizon, cumulative = TRUE)
nested_data_tbl <- nested_data_tbl |>
  split_nested_timeseries(.length_test = horizon)

extract_nested_train_split(nested_data_tbl, .row_id = 1)
extract_nested_test_split(nested_data_tbl, .row_id = 1)


# * Recipes ---------------------------------------------------------------

# Baseline Recipe
# - Time Series Signature - Adds bulk time-based features
rcp_spec_sf <- recipe(y ~ ., data = extract_nested_train_split(nested_data_tbl)) |>
  step_timeseries_signature(ds) |>
  step_rm(matches("(iso)|(xts)|(minute)|(second)|(year)|(quarter)|(month)|(half)|(mday)|(qday)|(yday)")) |>
  step_normalize(matches("(index.num)")) |>
  step_dummy(all_nominal(), one_hot = TRUE) |>
  step_naomit(starts_with("lag_")) 

rcp_spec_ml <- rcp_spec_sf |> step_rm(ds)

rcp_spec_sf |> prep() |> juice() |> glimpse()
rcp_spec_ml |> prep() |> juice() |> glimpse()


# Modeltime Nested Modelling Workflow -------------------------------------

?linear_reg()
# - Baseline model for ML

# * Engines ---------------------------------------------------------------

model_spec_lm <- linear_reg() |> set_engine("lm")

# * Workflows -------------------------------------------------------------

wrkfl_fit_lm <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec_ml)

# * Calibration -----------------------------------------------------------

?modeltime_nested_fit
?control_nested_fit

nested_modeltime_tbl <- nested_data_tbl |>
  modeltime_nested_fit(
    model_list = list(wrkfl_fit_lm),
    control = control_nested_fit(verbose = TRUE, allow_par = FALSE)
  )

nested_modeltime_tbl # nested modeltime tables

# * Evaluation ------------------------------------------------------------

?extract_nested_test_accuracy
?extract_nested_test_forecast
?extract_nested_error_report

# Accuracy
nested_modeltime_tbl |>
  extract_nested_test_accuracy() |>
  table_modeltime_accuracy()

# Plotting
nested_modeltime_tbl |>
  extract_nested_test_forecast() |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.facet_ncol = 2)

# Error reporting
nested_modeltime_tbl |> extract_nested_error_report()

# Helper function to quickly calibrate, evaluate and plot (to use only with few time series)
nested_calibrate_evaluate_plot(
  nested_data_tbl,
  workflows = list(wrkfl_fit_lm),
  id_var = "unique_id",
  parallel = FALSE
)

# * Refitting -------------------------------------------------------------

?modeltime_nested_select_best
?modeltime_nested_refit

nested_modeltime_best_tbl <- nested_modeltime_tbl |>
  modeltime_nested_select_best(metric = "rmse")

# refit_tbl <- calibration_tbl |>
#   modeltime_refit(data = data_prep_tbl)
nested_best_refit_tbl <- nested_modeltime_best_tbl |>
  modeltime_nested_refit(control = control_refit(verbose = TRUE, allow_par = FALSE))

# Error reporting
nested_best_refit_tbl |> extract_nested_error_report()

# * Forecasting -----------------------------------------------------------

?modeltime_nested_forecast
?extract_nested_future_forecast

nested_best_refit_tbl |> extract_nested_future_forecast()

nested_forecast_tbl <- nested_best_refit_tbl |>
  modeltime_nested_forecast(
    control = control_nested_forecast(verbose = TRUE, allow_par = FALSE)
  )

nested_forecast_tbl |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.facet_ncol = 2)



# Nested Multiple Models Workflow -----------------------------------------

# Engines & Workflows -----------------------------------------------------

# * PROPHET XGBOOST -------------------------------------------------------

# PROPHET with boosting
model_spec_prophet_xgb <- prophet_boost(
  # PROPHET params
  changepoint_num = 25,
  changepoint_range = 0.8,
  seasonality_daily = FALSE,
  seasonality_weekly = FALSE,
  seasonality_yearly = FALSE,
  # XGBOOST params
  mtry = 0.75,
  min_n = 20,
  tree_depth = 3,
  learn_rate = 0.2,
  loss_reduction = 0.15,
  trees = 300
) |>
  set_engine(
    "prophet_xgboost",
    counts = FALSE
  )

set.seed(123)
wrkfl_fit_prophet_xgb <- workflow() |>
  add_model(model_spec_prophet_xgb) |>
  add_recipe(rcp_spec_sf |> step_rm(starts_with("lag")))

# * RANDOM FOREST ---------------------------------------------------------

model_spec_rf <- rand_forest(
  mode = "regression",
  mtry = 25,
  trees = 1000,
  min_n = 25
) |>
  set_engine("ranger")

set.seed(123)
wrkfl_fit_rf <- workflow() |>
  add_model(model_spec_rf) |>
  add_recipe(rcp_spec_ml)

# * NEURAL NETWORK --------------------------------------------------------

model_spec_nnetar <- nnetar_reg(
  non_seasonal_ar = 2,
  seasonal_ar = 1,
  hidden_units = 10,
  penalty = 10,
  num_networks = 10,
  epochs = 50
) |>
  set_engine("nnetar")

set.seed(123)
wrkfl_fit_nnetar <- workflow() |>
  add_model(model_spec_nnetar) |>
  add_recipe(rcp_spec_sf |> step_rm(starts_with("lag")))


# Calibration, Evaluation & Plotting --------------------------------------

# Setup Parallel Processing
# registerDoFuture()
# plan(strategy = multisession, workers = parallelly::availableCores())
# message("Number of parallel workers: ", nbrOfWorkers())
parallel_start(parallelly::availableCores())

set.seed(123)
nested_modeltime_tbl <- nested_data_tbl |>
  modeltime_nested_fit(
    model_list = list(
      wrkfl_fit_prophet_xgb,
      wrkfl_fit_rf,
      wrkfl_fit_nnetar
    ),
    control = control_nested_fit(verbose = TRUE, allow_par = TRUE)
  )

parallel_stop()
# plan(sequential)

# Accuracy
nested_modeltime_tbl |>
  extract_nested_test_accuracy() |>
  table_modeltime_accuracy()

# Plotting
nested_modeltime_tbl |>
  extract_nested_test_forecast() |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.facet_ncol = 2)

# Error reporting
nested_modeltime_tbl |> extract_nested_error_report()


# Refitting & Forecasting -------------------------------------------------

# Select best model for each time series
nested_modeltime_best_tbl <- nested_modeltime_tbl |>
  modeltime_nested_select_best(metric = "rmse")

parallel_start(parallelly::availableCores())

# Refitting
nested_best_refit_tbl <- nested_modeltime_best_tbl |>
  modeltime_nested_refit(
    control = control_refit(verbose = TRUE, allow_par = TRUE)
  )

parallel_stop()

# Error reporting
nested_best_refit_tbl |> extract_nested_error_report()

# Forecasting
nested_forecast_tbl <- nested_best_refit_tbl |>
  modeltime_nested_forecast(
    control = control_nested_forecast(verbose = TRUE, allow_par = FALSE)
  )

nested_forecast_tbl |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.facet_ncol = 2)



# GLOBAL MODELLING --------------------------------------------------------

# Modeltime Global Data Workflow ------------------------------------------

# * Modelling & Forecast Data ---------------------------------------------

data_prep_tbl <- data_prep_full_tbl |> drop_na()
forecast_tbl <- data_prep_full_tbl |> filter(is.na(y))

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



# Global Modelling Workflow -----------------------------------------------

?linear_reg()
# - Baseline model for ML

# * Engines ---------------------------------------------------------------

model_spec_lm <- linear_reg() |> set_engine("lm")

# * Workflows -------------------------------------------------------------

wrkfl_fit_lm <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits))

# * Calibration -----------------------------------------------------------

calibration_tbl <- modeltime_table(wrkfl_fit_lm) |>
  modeltime_calibrate(testing(splits), id = "unique_id")

# * Evaluation ------------------------------------------------------------

# Global accuracy
calibration_tbl |> modeltime_accuracy()

# Local accuracy
calibration_tbl |> modeltime_accuracy(acc_by_id = TRUE)

calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl, conf_by_id = TRUE) |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue", .facet_ncol = 2)

# * Refitting -------------------------------------------------------------

refit_tbl <- calibration_tbl |> modeltime_refit(data = data_prep_tbl)

# * Forecasting -----------------------------------------------------------

refit_tbl |>
  modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl, conf_by_id  = TRUE) |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue", .facet_ncol = 2)



# Global Multiple Models Workflow -----------------------------------------

# Engines & Workflows -----------------------------------------------------

# * ELASTIC NET -----------------------------------------------------------

model_spec_elanet <- linear_reg(
  mode = "regression",
  penalty = 0.01,
  mixture = 0.5
) |>
  set_engine("glmnet")

wrkfl_fit_elanet <- workflow() |>
  add_model(model_spec_elanet) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits))

# * SVM -------------------------------------------------------------------

model_spec_svm_rbf <- svm_rbf(
  mode = "regression",
  cost = 1,
  rbf_sigma = 0.01,
  margin = 0.1
) |>
  set_engine("kernlab")

set.seed(123)
wrkfl_fit_svm_rbf <- workflow() |>
  add_model(model_spec_svm_rbf) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits))

# * BOOSTING --------------------------------------------------------------

# XGBOOST
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

set.seed(123)
wrkfl_fit_xgb <- workflow() |>
  add_model(model_spec_xgb) |>
  add_recipe(rcp_spec_ml) |>
  fit(training(splits))


# Calibration, Evaluation & Plotting --------------------------------------

calibration_tbl <- modeltime_table(
  wrkfl_fit_lm,
  wrkfl_fit_elanet,
  wrkfl_fit_svm_rbf,
  wrkfl_fit_xgb
) |>
  modeltime_calibrate(testing(splits), id = "unique_id")

# Global accuracy
calibration_tbl |> modeltime_accuracy()

# Local accuracy
calibration_tbl |>
  modeltime_accuracy(acc_by_id = TRUE) |>
  arrange(unique_id)

calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl, conf_by_id = TRUE) |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_show = FALSE, .facet_ncol = 2)


# Refitting & Forecasting -------------------------------------------------

# Select Best Global Model
global_model_best <- calibration_tbl |> select_best_id(n = 1, metric = "rmse")

refit_global_tbl <- calibration_tbl |>
  filter(.model_id %in% global_model_best) |>
  modeltime_refit(data = data_prep_tbl)

refit_global_tbl |>
  modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl, conf_by_id  = TRUE) |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue", .facet_ncol = 2)


# Select Best Local Models
local_models_best <- calibration_tbl |>
  select_best_id(n = 1, metric = "rmse", by_id = TRUE, id_var = "unique_id")

refit_local_tbl <- calibration_tbl |>
  filter(.model_id %in% local_models_best[[".model_id"]]) |>
  modeltime_refit(data = data_prep_tbl)

ts_ids <- local_models_best[["unique_id"]]
forecast_local_tbl <- vector("list", length(ids))
for (i in seq_along(ts_ids)) {
  ts_id <- ts_ids[i]
  model_id <- local_models_best |> filter(unique_id == ts_id) |> pull(.model_id)
  forecast_local_tbl[[i]] <- refit_local_tbl |>
    filter(.model_id == model_id) |>
    modeltime_forecast(
      new_data = forecast_tbl |> filter(unique_id == ts_id),
      actual_data = data_prep_tbl |> filter(unique_id == ts_id),
      conf_by_id  = TRUE
    )
}
forecast_local_tbl <- bind_rows(forecast_local_tbl)

forecast_local_tbl |>
  group_by(unique_id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue", .facet_ncol = 2)

