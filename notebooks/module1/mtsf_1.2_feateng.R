# Modern Time Series Forecasting with R ----
# Marco Zanotti

# Lecture 1.2: Features Engineering & Recipes -------------------------------

# Goals:
# - Learn advanced features engineering workflows & techniques
# - Learn how to use recipes

# Challenges:
# - Challenge 1 - Feature Engineering



# Packages ----------------------------------------------------------------

source("src/R/utils.R")
source("src/R/install.R")



# Data --------------------------------------------------------------------

email_tbl <- load_data("data/email", "email_prep", ext = ".parquet")




# Features Engineering ----------------------------------------------------

# Pre-processing Data

email_tbl |> plot_time_series(ds, log1p(y), .smooth = FALSE)

data_prep_tbl <- email_tbl |>
  # pre-processing
  mutate(y = log_interval_vec(y, limit_lower = 0, offset = 1)) |>
  mutate(y = standardize_vec(y)) |>
  # fix missing values at beginning of series
  filter_by_time(.start_date = "2018-07-03") |>
  # Cleaning
  mutate(y_trans_cleaned = ts_clean_vec(y, period = 7)) |>
  mutate(y = ifelse(
    ds |> between_time("2018-11-18", "2018-11-20"),
    y_trans_cleaned,
    y)
  ) |>
  select(-y, -y_trans_cleaned) |> 
  mutate(across(pageViews:sessions, .fns = log1p)) |>
  mutate(across(pageViews:sessions, .fns = standardize_vec))

data_prep_tbl |> plot_time_series(ds, y)
data_prep_tbl |>
  pivot_longer(cols = pageViews:sessions) |>
  plot_time_series(ds, .value = value, .facet_nrow = 3, .facet_vars = name)
data_prep_tbl |> plot_time_series(ds, promo, .smooth = FALSE)


# * Time-Based Features ---------------------------------------------------

# - tk_augment_timeseries_signature()

data_prep_signature_tbl <- data_prep_tbl |>
  tk_augment_timeseries_signature() |>
  select(
    -diff, -ends_with("iso"), -ends_with(".xts"), -contains("hour"),
    -contains("minute"), -contains("second"), -contains("am.pm")
  )
data_prep_signature_tbl |> glimpse()


# * Trend-Based Features --------------------------------------------------

# linear trend
data_prep_signature_tbl |> plot_time_series_regression(ds, .formula = y ~ index.num)

# nonlinear trend - basis splines
data_prep_signature_tbl |>
  plot_time_series_regression(
    ds,
    y ~ splines::bs(index.num, df = 3),
    .show_summary = TRUE
  )

# nonlinear trend - natural splines
data_prep_signature_tbl |>
  plot_time_series_regression(
    ds,
    y ~ splines::ns(index.num, knots = quantile(index.num, probs = c(0.25, 0.5))),
    .show_summary = TRUE
  )


# * Seasonal Features -----------------------------------------------------

# weekly seasonality
data_prep_signature_tbl |>
  plot_time_series_regression(ds, y ~ wday.lbl, .show_summary = TRUE)

# monthly seasonality
data_prep_signature_tbl |>
  plot_time_series_regression(ds, y ~ month.lbl, .show_summary = TRUE)


# * Interaction Features --------------------------------------------------

data_prep_signature_tbl |>
  plot_time_series_regression(ds, y ~ (as.factor(week2) * wday.lbl), .show_summary = TRUE)


# * Rolling Average Features ----------------------------------------------

# - tk_augment_slidify

data_prep_rolls_tbl <- data_prep_tbl |>
  tk_augment_slidify(
    y, mean,
    .period = c(7, 14, 30, 90),
    .align = "center",
    .partial = TRUE
  )
data_prep_rolls_tbl |> glimpse()

data_prep_rolls_tbl |> plot_time_series_regression(ds, y ~ ., .show_summary = TRUE)


# * Lag Features ----------------------------------------------------------

# tk_augment_lags

data_prep_tbl |>
  plot_acf_diagnostics(ds, y, .lags = 100)

data_prep_lags_tbl <- data_prep_tbl |>
  tk_augment_lags(y, .lags = c(1, 7, 14, 30, 90, 365)) |>
  select(-unique_id, -c(pageViews:promo)) |> 
  drop_na()
data_prep_lags_tbl |> glimpse()

data_prep_lags_tbl |> plot_time_series_regression(ds, y ~ ., .show_summary = TRUE)


# * Fourier Series Features -----------------------------------------------

# - tk_augment_fourier

data_prep_tbl |> plot_acf_diagnostics(ds, y, .lags = 100)

data_prep_fourier_tbl <- data_prep_tbl |>
  tk_augment_fourier(ds, .periods = c(1, 7, 14, 30, 90, 365), .K = 2) |> 
  select(-unique_id, -c(pageViews:promo))
data_prep_fourier_tbl |> glimpse()

data_prep_fourier_tbl |> plot_time_series_regression(ds, y ~ ., .show_summary = TRUE)


# * promo Data Features ---------------------------------------------------


data_prep_tbl |>
  plot_time_series(ds, y, .smooth = FALSE, .interactive = FALSE) +
  geom_point(
    aes(x = ds, y = y), color = "red",
    data = data_prep_tbl |> filter(promo == 1)
  )

data_prep_tbl |>
  select(ds, y, promo) |> 
  plot_time_series_regression(ds, y ~ ., .show_summary = TRUE)


# * External Regressor Features -------------------------------------------

data_prep_tbl |>
  plot_acf_diagnostics(
    ds, y,
    .lags = 100,
    .ccf_vars = pageViews:sessions,
    .show_ccf_vars_only = TRUE
  )

data_prep_google_tbl <- data_prep_tbl |>
  tk_augment_lags(pageViews:sessions, .lags = c(7, 42)) |>
  drop_na()
data_prep_google_tbl |> glimpse()

data_prep_google_tbl |>
  select(-unique_id, -promo) |> 
  plot_time_series_regression(ds, y ~ ., .show_summary = TRUE)



# Recipes -----------------------------------------------------------------

# Description of the steps to be applied to a data set to prepare it for analysis.
# based on steps
# may contain pre-processing steps
# may contain feature engineering steps

# Splitting Data
email_tbl |> tk_summary_diagnostics()

splits <- email_tbl |> 
  select(ds, y) |> 
  time_series_split(, date_var = ds, assess = 7 * 8, cumulative = TRUE)
splits |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(ds, y)

# Creating Recipe
recipe_spec_full <- recipe(y ~ ., data = training(splits)) |>
  
  # pre-processing steps
  step_log_interval(y, limit_lower = 0, offset = 1) |>
  step_normalize(y) |>
  step_filter(ds >= "2018-07-03") |>
  step_ts_clean(y, period = 7) |>

  # features engineering steps
  # time-based, trend and seasonal features
  step_timeseries_signature(ds) |>
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) |>
  step_normalize(matches("(index.num)|(year)|(yday)")) |>
  step_dummy(all_nominal(), one_hot = TRUE) |>
  step_ns(ends_with("index.num"), deg_free = 2) |>
  # interaction features
  step_interact(~ matches("week2") * matches("wday.lbl")) |>
  # rolling features
  step_slidify_augment(
    y, .f = mean, period = c(7, 14, 30, 90), align = "center", partial = TRUE
  ) |>
  # lag features
  # step_lag(y, lag = 56) |> # should remove NA's
  # fourier series features
  step_fourier(ds, period = c(7, 14, 30, 90), K = 2) |>
  step_rm(ds)

# Note: cannot add promo data or external regressors via recipe !!!!

recipe_spec_full
recipe_spec_full |> prep() |> juice() |> glimpse()

# Fitting on Recipe & Calibrating
model_spec_lm <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

workflow_fit_lm <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(recipe_spec_full) |>
  fit(training(splits))

calibration_tbl <- modeltime_table(workflow_fit_lm) |>
  modeltime_calibrate(new_data = testing(splits), quiet = FALSE)

calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = email_tbl) |>
  plot_modeltime_forecast()

calibration_tbl |> modeltime_accuracy()



# Features Engineering + Recipes Workflow -----------------------------------

# * Pre-processing Data ---------------------------------------------------

email_tbl |> glimpse()

# pre-processing target variable
email_prep_tbl <- email_tbl |>
  select(-unique_id) |> 
  mutate(y = log_interval_vec(y, limit_lower = 0, offset = 1)) |>
  mutate(y = standardize_vec(y)) |>
  filter_by_time(.start_date = "2018-07-03") |>
  mutate(y_cleaned = ts_clean_vec(y, period = 7)) |>
  mutate(y = ifelse(
    ds |> between_time("2018-11-18", "2018-11-20"),
    y_cleaned,
    y)
  ) |>
  select(-y_cleaned)

# save key parameters
limit_lower <- 0
limit_upper <- 3650.8
offset <- 1
std_mean <- -5.25529020756467
std_sd <- 1.1109817111334


# * Creating Features -----------------------------------------------------

# - Extend to Future Window
# - Add any lags to full dataset
# - Add any external regressors to full dataset

horizon <- 7 * 8
lag_period <- 7 * 8
rolling_periods <- c(30, 60, 90)

data_prep_full_tbl <- email_prep_tbl %>%
  # Add future window
  bind_rows(future_frame(.data = ., ds, .length_out = horizon)) |>
  # Add Autocorrelated Lags
  tk_augment_lags(y, .lags = lag_period) |>
  # Add rolling features
  tk_augment_slidify(
    y_lag56, mean, .period = rolling_periods, .align = "center", .partial = TRUE
  ) |>
  # Keep Promo
  select(-c(pageViews:sessions)) |>
  # Reformat Columns
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))

data_prep_full_tbl |>
  pivot_longer(-ds) |>
  plot_time_series(ds, value, name, .smooth = FALSE)


# * Separate into Modelling & Forecast Data -------------------------------

data_prep_full_tbl |> tail(horizon + 1)

data_prep_tbl <- data_prep_full_tbl |>
  slice_head(n = nrow(data_prep_full_tbl) - horizon)

forecast_tbl <- data_prep_full_tbl |>
  slice_tail(n = horizon) |> 
  mutate(promo = 0)


# * Train / Test Sets -----------------------------------------------------

splits <- time_series_split(data_prep_tbl, assess = horizon, cumulative = TRUE)

splits |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(ds, y)


# * Recipes ---------------------------------------------------------------

# Baseline Recipe
# - Time Series Signature - Adds bulk time-based features
# - Interaction: wday.lbl:week2
# - Fourier Features
rcp_spec <- recipe(y ~ ., data = training(splits)) |>
  # Time Series Signature
  step_timeseries_signature(ds) |>
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) |>
  step_normalize(matches("(index.num)|(year)|(yday)")) |>
  step_dummy(all_nominal(), one_hot = TRUE) |>
  # Interaction
  step_interact(~ matches("week2") * matches("wday.lbl")) |>
  # Fourier
  step_fourier(ds, period = c(7, 14, 30, 90, 365), K = 2)
rcp_spec |> prep() |> juice() |> glimpse()

# Spline Recipe
# - natural spline series on index.num
rcp_spec_spline <- rcp_spec |>
  step_ns(ends_with("index.num"), deg_free = 2) |>
  step_rm(ds) |>
  step_rm(starts_with("lag_"))
rcp_spec_spline |> prep() |> juice() |> glimpse()

# Lag Recipe
# - lags of y and rolls
rcp_spec_lag <- rcp_spec |>
  step_naomit(starts_with("lag_")) |>
  step_rm(ds)
rcp_spec_lag |> prep() |> juice() |> glimpse()


# * Model Engine Specification --------------------------------------------

model_spec_lm <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")


# * Model Workflows -------------------------------------------------------

# LM Spline Workflow
wrkfl_fit_lm_1_spline <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec_spline) |>
  fit(training(splits))
wrkfl_fit_lm_1_spline
wrkfl_fit_lm_1_spline |>
  extract_fit_parsnip() |>
  pluck("fit") |>
  summary()

# LM Lag Workflow
wrkfl_fit_lm_2_lag <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec_lag) |>
  fit(training(splits))
wrkfl_fit_lm_2_lag
wrkfl_fit_lm_2_lag |>
  extract_fit_parsnip() |>
  pluck("fit") |>
  summary()

# LM Spline Workflow
wrkfl_fit_lm_3 <- workflow() |>
  add_model(model_spec_lm) |>
  add_recipe(rcp_spec) |>
  fit(training(splits))
wrkfl_fit_lm_3
wrkfl_fit_lm_3 |>
  extract_fit_parsnip() |>
  pluck("fit") |>
  summary()


# * Modeltime -------------------------------------------------------------

# Calibration
calibration_tbl <- modeltime_table(
  wrkfl_fit_lm_1_spline,
  wrkfl_fit_lm_2_lag,
  wrkfl_fit_lm_3
) |>
  update_model_description(1, "LM - Spline Recipe") |>
  update_model_description(2, "LM - Lag Recipe") |>
  update_model_description(3, "LM - Base Recipe") |>
  modeltime_calibrate(new_data = testing(splits))
calibration_tbl

# Forecasting
calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl) |>
  plot_modeltime_forecast()

# Accuracy
calibration_tbl |> modeltime_accuracy()

# convert back to original scale
calibration_tbl |>
  modeltime_forecast(new_data = testing(splits), actual_data = data_prep_tbl) |>
  mutate(
    across(
      .value:.conf_hi,
      .fns = ~ standardize_inv_vec(x = ., mean = std_mean, sd = std_sd)
    )
  ) |>
  mutate(
    across(
      .value:.conf_hi,
      .fns = ~ log_interval_inv_vec(
        x = ., limit_lower = limit_lower, limit_upper = limit_upper, offset = offset
      )
    )
  ) |>
  plot_modeltime_forecast()

# Refitting (problem in refitting !!!!!)
# refit_tbl <- calibration_tbl |>
#   modeltime_refit(data = data_prep_tbl)

# refit_tbl |>
#   modeltime_forecast(new_data = data_prep_tbl, actual_data = data_prep_tbl) |>
#   plot_modeltime_forecast()

# refit_tbl |>
#   modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl) |>
#   plot_modeltime_forecast()


# refit_tbl |>
#   modeltime_forecast(new_data = data_prep_tbl, actual_data = data_prep_tbl) |>
#   mutate(
#     across(
#       .value:.conf_hi,
#       .fns = ~ standardize_inv_vec(x = ., mean = std_mean, sd = std_sd)
#     )
#   ) |>
#   mutate(
#     across(
#       .value:.conf_hi,
#       .fns = ~ log_interval_inv_vec(
#         x = ., limit_lower = limit_lower, limit_upper = limit_upper, offset = offset
#       )
#     )
#   ) |>
#   plot_modeltime_forecast()

# refit_tbl |>
#   modeltime_forecast(new_data = forecast_tbl, actual_data = data_prep_tbl) |>
#   mutate(
#     across(
#       .value:.conf_hi,
#       .fns = ~ standardize_inv_vec(x = ., mean = std_mean, sd = std_sd)
#     )
#   ) |>
#   mutate(
#     across(
#       .value:.conf_hi,
#       .fns = ~ log_interval_inv_vec(
#         x = ., limit_lower = limit_lower, limit_upper = limit_upper, offset = offset
#       )
#     )
#   ) |>
#   plot_modeltime_forecast()


# * Save Artifacts --------------------------------------------------------

feature_engineering_artifacts_list <- list(
  # Data
  data = list(
    "data_prep_tbl" = data_prep_tbl,
    "forecast_tbl" = forecast_tbl
  ),
  # Recipes
  recipes = list(
    "rcp_spec" = rcp_spec,
    "rcp_spec_spline" = rcp_spec_spline,
    "rcp_spec_lag" = rcp_spec_lag
  ),
  # Inversion Parameters
  standardize = list(
    std_mean = std_mean,
    std_sd   = std_sd
  ),
  log_interval = list(
    limit_lower = limit_lower,
    limit_upper = limit_upper,
    offset      = offset
  )
)

feature_engineering_artifacts_list |>
  write_rds("data/email/artifacts/feature_engineering_artifacts_list.rds")

