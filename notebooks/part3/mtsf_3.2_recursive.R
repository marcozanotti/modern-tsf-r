# Modern Time Series Forecasting with R ----
# Marco Zanotti

# Lecture 3.2: Recursive Time Series Algorithms ----------------------------

# Goals:
# - Panel Recursivity



# Packages ----------------------------------------------------------------

source("src/R/utils.R")
source("src/R/packages.R")



# RECURSIVITY - PANEL TIME SERIES -----------------------------------------

?recursive()
?panel_tail()



# Data & Artifacts --------------------------------------------------------

email_tbl <- load_data("data/email", "email_prep", ext = ".parquet")


# * Data Preparation ------------------------------------------------------

email_tbl |> count(member_rating)

# email data by group
email_tbl <- email_tbl |>
  rename(id = member_rating) |>
  mutate(id = ifelse(id == 2, id, 1) |> as.factor()) |>
  group_by(id) |>
  summarise_by_time(ds, .by = "day", y = n()) |>
  pad_by_time(.pad_value = 0)

email_tbl |>
  plot_time_series(ds, log1p(y), .smooth = FALSE)

email_prep_tbl <- email_tbl |>
  mutate(y = log_interval_vec(y, limit_lower = 0, offset = 1)) |>
  mutate(y = standardize_vec(y)) |>
  filter_by_time(.start_date = "2018-07-05") |>
  mutate(y_trans_cleaned = ts_clean_vec(y, period = 7)) |>
  mutate(
    y = ifelse(
      ds |> between_time("2018-11-18", "2018-11-20"),
      y_trans_cleaned,
      y
    )
  ) |>
  select(-y, -y_trans_cleaned)

email_prep_tbl |>
  plot_time_series(ds, y, .smooth = FALSE)

# events data
events_daily_tbl <- events_tbl |>
  mutate(event_date = ymd_hms(event_date)) |>
  summarise_by_time(event_date, .by = "day", promo = n())


# * Feature Engineering ---------------------------------------------------

# - Extend to Future Window
# - Add any lags to full dataset
# - Add any external regressors to full dataset

horizon <- 7 * 8
lags <- c(1, 2, 7, 14, 30)

data_prep_full_tbl <- email_prep_tbl |>
  future_frame(.data = _, ds, .length_out = horizon, .bind_data = TRUE) |>
  lag_transf_grouped() |>
  group_by(id) |>
  tk_augment_lags(y, .lags = c(horizon, 90)) |>
  left_join(events_daily_tbl, by = c("ds" = "event_date")) |>
  mutate(promo = ifelse(is.na(promo), 0, promo))

data_prep_full_tbl |>
  pivot_longer(cols = -c(id, ds)) |>
  plot_time_series(ds, value, name, .smooth = FALSE)
data_prep_full_tbl |> slice_tail(n = horizon + 1)


# * Modelling & Forecast Data ---------------------------------------------

data_prep_tbl <- data_prep_full_tbl |>
  slice_head(n = nrow(data_prep_full_tbl) - horizon) |>
  drop_na() |>
  ungroup()
forecast_tbl <- data_prep_full_tbl |>
  slice_tail(n = horizon)


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
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    id = "id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), id, horizon)
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
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    id = "id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), id, horizon)
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
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    id = "id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), id, horizon)
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
  add_recipe(rcp_spec) |>
  fit(training(splits)) |>
  recursive(
    id = "id",
    transform = lag_transf_grouped,
    train_tail = panel_tail(training(splits), id, horizon)
  )



# Recursive Models' Performance -------------------------------------------

# * Evaluation
calibration_tbl <- modeltime_table(
  wrkfl_fit_knn_recursive,
  wrkfl_fit_rf_recursive,
  wrkfl_fit_cubist_recursive,
  wrkfl_fit_nnet_recursive
) |>
  modeltime_calibrate(testing(splits), id = "id")

calibration_tbl |>
  modeltime_accuracy(acc_by_id = TRUE) |>
  arrange(id)

calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl, keep_data = TRUE) |>
  group_by(id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue")

# * Refitting & Forecasting
refit_tbl <- calibration_tbl |>
  modeltime_refit(data = data_prep_tbl)

refit_tbl |>
  modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl, keep_data = TRUE) |>
  group_by(id) |>
  plot_modeltime_forecast(.conf_interval_fill = "lightblue")

