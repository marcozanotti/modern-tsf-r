# Modern Time Series Forecasting with R ----
# Marco Zanotti

# Lecture 1.1: Manipulation, Transformation & Visualization -----------------

# Goals:
# - Learn timetk data wrangling functionality
# - Commonly used time series transformations
# - Commonly used time series visualizations



# Packages ----------------------------------------------------------------

source("src/R/utils.R")
source("src/R/install.R")



# Data --------------------------------------------------------------------

email_tbl <- load_data("data/email", "email_prep", ext = ".parquet")



# Manipulation ------------------------------------------------------------

# * Summarize by Time -----------------------------------------------------

# - Apply commonly used aggregations
# - High-to-Low Frequency

# to weekly
email_tbl |>
  summarize_by_time(ds, .by = "week", y = n())

# to monthly
email_tbl |>
  summarize_by_time(ds, .by = "month", y = n())


# * Pad by Time -----------------------------------------------------------

# - Filling in time series gaps
# - Low-to-High Frequency (un-aggregating)

# fill daily gaps
email_tbl |>
  pad_by_time(.date_var = ds, .by = "day", .pad_value = 0, .start_date = "2018-06-01")

# weekly to daily
email_tbl |>
  pad_by_time(ds, .by = "day", .start_date = "2018-06") |>
  mutate_by_time(.by = "week", y = sum(y, na.rm = TRUE) / 7)


# * Filter by Time --------------------------------------------------------

# - Pare data down before modeling

email_tbl |>
  filter_by_time(.start_date = "2018-11-20")

email_tbl |>
  filter_by_time(.start_date = "2019-12", .end_date = "2019")

email_tbl |>
  filter_by_time(.start_date = "2019-12", .end_date = "2019-12-01" %+time% "4 weeks")


# * Mutate by Time --------------------------------------------------------

# - Get change from beginning/end of period

# first, last, mean, median by period
email_tbl |>
  mutate_by_time(
    .by = "1 week",
    y_mean = mean(y),
    y_median = median(y),
    y_max = max(y),
    y_min = min(y)
  )


# * Future Frame ----------------------------------------------------------

# - Forecasting helper

email_tbl |> future_frame(.length_out = 10)

# modelling example on date features
model_fit_lm <- lm(
  y ~ as.numeric(ds) + wday(ds, label = TRUE),
  data = email_tbl
)
future_tbl <- email_tbl |> future_frame(.date_var = ds, .length_out = "2 months")
predictions_vec <- predict(model_fit_lm, newdata = future_tbl) |> as.vector()

email_tbl |>
  select(ds, y) |>
  add_column(type = "actual") |>
  bind_rows(
    future_tbl |> mutate(y = predictions_vec, type = "prediction")
  ) |>
  plot_time_series(ds, y, type, .smooth = FALSE)



# Transformation ----------------------------------------------------------

# * Variance Reduction ----------------------------------------------------

# Log
email_tbl |> mutate(y = log(y))

# Log + 1
email_tbl |>
  mutate(y = log1p(y)) |>
  plot_time_series(ds, y)

# - inversion with exp() and expm1()

# Box-Cox
email_tbl |>
  mutate(y = box_cox_vec(y + 1, lambda = "auto")) |>
  plot_time_series(ds, y)

# - inversion with box_cox_inv_vec()


# * Range Reduction -------------------------------------------------------

# - Used in visualization to overlay series
# - Used in ML for models that are affected by feature magnitude (e.g. linear regression)

# Normalization Range (0,1)
email_tbl |> 
  mutate(y = normalize_vec(y)) |>
  plot_time_series(ds, y)

# Standardization
email_tbl |> 
  mutate(y = standardize_vec(y)) |>
  plot_time_series(ds, y)


# * Smoothing -------------------------------------------------------------

# - Identify trends and cycles
# - Clean seasonality

email_tbl |>
  mutate(
    y = log1p(y),
    y_smooth = smooth_vec(y, period = 24 * 7, degree = 0)
  ) |>
  pivot_longer(cols = c(y, y_smooth)) |>
  plot_time_series(ds, value, .color_var = name, .smooth = FALSE)


# * Rolling Averages ------------------------------------------------------

# - Common time series operations to visualize trend
# - A simple transformation that can help create improve features
# - Can help with outlier-effect reduction & trend detection
# - Note: Businesses often use a rolling average as a forecasting technique
# A rolling average forecast is usually sub-optimal (good opportunity for you!).

email_tbl |>
  mutate(
    y_roll = slidify_vec(
      y,
      .f = mean,
      .period = 24 * 7, # 6 months = 7 days * 24 weeks
      .align = "center",
      .partial = TRUE
    )
  ) |>
  pivot_longer(cols = c(y, y_roll)) |>
  plot_time_series(ds, log1p(value), .color_var = name, .smooth = FALSE)


# * Missing Values Imputation ---------------------------------------------

# - Imputation helps with filling gaps (if needed)

email_tbl |>
  mutate(y_na = ifelse(y == 0, NA, y)) |>
  mutate(y_imputed = ts_impute_vec(y_na, period = 7)) |>
  pivot_longer(cols = c(y_na, y_imputed)) |>
  plot_time_series(ds, log1p(value), .color_var = name, .smooth = FALSE)


# * Anomaly Cleaning ------------------------------------------------------

# - Outlier removal helps linear regression detect trend and reduces high leverage points
# WARNING: Make sure you check outliers against events
# - usually there is a reason for large values

# Anomaly detection
email_tbl |> plot_anomaly_diagnostics(ds, y)

email_tbl |> plot_anomaly_diagnostics(ds, log1p(y))

email_cleaned_tbl <- email_tbl |>
  mutate(
    y_log = log1p(y),
    y_cleaned = ts_clean_vec(y, period = 7),
    y_log_cleaned = ts_clean_vec(y_log, period = 7)
  )

email_cleaned_tbl |>
  pivot_longer(cols = c(y, y_log, y_cleaned, y_log_cleaned)) |>
  mutate(
    cleaned = ifelse(str_detect(name, "cleaned"), "cleaned", "level"),
    type = ifelse(str_detect(name, "log"), "log", "level")
  ) |>
  plot_time_series(ds, value, .color_var = cleaned, .facet_vars = type, .smooth = FALSE)

# without log
# outlier effect - before cleaning
email_cleaned_tbl |>
  plot_time_series_regression(
    ds,
    .formula = y ~ as.numeric(ds) +
      lubridate::wday(ds, label = TRUE) +
      lubridate::month(ds, label = TRUE),
    .show_summary = TRUE
  )

# outlier effect - after cleaning
email_cleaned_tbl |>
  plot_time_series_regression(
    ds,
    .formula = y_cleaned ~ as.numeric(ds) +
      lubridate::wday(ds, label = TRUE) +
      lubridate::month(ds, label = TRUE),
    .show_summary = TRUE
  )

# with log
# outlier effect - before cleaning
email_cleaned_tbl |>
  plot_time_series_regression(
    ds,
    .formula = y_log ~ as.numeric(ds) +
      lubridate::wday(ds, label = TRUE) +
      lubridate::month(ds, label = TRUE),
    .show_summary = TRUE
  )

# outlier effect - after cleaning
email_cleaned_tbl |>
  plot_time_series_regression(
    ds,
    .formula = y_log_cleaned ~ as.numeric(ds) +
      lubridate::wday(ds, label = TRUE) +
      lubridate::month(ds, label = TRUE),
    .show_summary = TRUE
  )


# * Lags & Differencing ---------------------------------------------------

# - Lags: Often used for feature engineering
# - Lags: Autocorrelation
# - MOST IMPORTANT: Can possibly use lagged variables in a model, if lags are correlated
# - Difference: Used to go from growth to change
# - Difference: Makes a series "stationary" (potentially)

# lags
email_tbl |> mutate(y_lag_1 = lag_vec(y, lag = 1))

email_tbl |> tk_augment_lags(.value = y, .lags = c(1, 2, 7, 14))

# differencing
email_tbl |>
  mutate(y_diff = diff_vec(y)) |>
  pivot_longer(cols = c(y, y_diff)) |>
  plot_time_series(ds, value, .color_var = name, .smooth = FALSE)


# * Fourier Transform ------------------------------------------------------

# - Useful for incorporating seasonality & autocorrelation
# - BENEFIT: Don't need a lag, just need a frequency (based on your time index)

# single fourier series
email_tbl |>
  mutate(sin14_k1 = fourier_vec(ds, period = 14, K = 1, type = "sin")) |>
  mutate(cos14_k1 = fourier_vec(ds, period = 14, K = 1, type = "cos")) |>
  select(-y) |>
  pivot_longer(matches("(cos)|(sin)")) |>
  plot_time_series(ds, value, .color_var = name, .smooth = FALSE)

# multiple fourier series
email_tbl |>
  tk_augment_fourier(ds, .periods = c(14, 30, 90, 365), .K = 2) |>
  select(ds, y, starts_with("ds_")) |>
  plot_time_series_regression(
    ds,
    .formula = log1p(y) ~ as.numeric(ds) + . - ds,
    .show_summary = TRUE
  )


# * Confined Interval -----------------------------------------------------

# - Transformation used to confine forecasts to a max/min interval

email_tbl |>
  plot_time_series(
    ds,
    log_interval_vec(y, limit_lower = 0, offset = 1)
  )



# Visualization -----------------------------------------------------------

# * Time Series Plot ------------------------------------------------------

email_tbl |> plot_time_series(ds, y, .smooth = FALSE)
email_tbl |> plot_time_series(ds, log1p(y)) # Log Transforms


# * Autocorrelation Function (ACF) Plot -----------------------------------

email_tbl |>
  plot_acf_diagnostics(ds, log1p(y), .lags = 10, .show_white_noise_bars = TRUE)

email_tbl |>
  plot_acf_diagnostics(ds, log1p(y), .lags = 500, .show_white_noise_bars = TRUE)


# * Cross-Correlation Function (CCF) Plot ---------------------------------

email_tbl |>
  drop_na() |>
  plot_acf_diagnostics(
    ds,
    y,
    .ccf_vars = pageViews:sessions,
    .lags = 100,
    .show_white_noise_bars = TRUE,
    .show_ccf_vars_only = TRUE,
    .facet_ncol = 3,
  )


# * Smoothing Plot --------------------------------------------------------

email_tbl |>
  plot_time_series(
    ds,
    log1p(y),
    .smooth_period = "90 days",
    .smooth_degree = 1
  )


# * Boxplots --------------------------------------------------------------

email_tbl |> plot_time_series_boxplot(ds, log1p(y), .period = "1 month")

email_tbl |>
  plot_time_series_boxplot(
    ds,
    log1p(y),
    .period = "1 month",
    .smooth = TRUE,
    .smooth_func = median, # change smoother
    .color_var = lubridate::year(ds)
  )


# * Seasonality Plot ------------------------------------------------------

email_tbl |> plot_seasonal_diagnostics(ds, log1p(y))


# * Decomposition Plot ----------------------------------------------------

email_tbl |> plot_stl_diagnostics(ds, log1p(y))


# * Anomaly Detection Plot ------------------------------------------------

email_tbl |> tk_anomaly_diagnostics(ds, y, .alpha = .01, .max_anomalies = .01)

email_tbl |> plot_anomaly_diagnostics(ds, y, .alpha = .01, .max_anomalies = .01)


# * Time Series Regression Plot -------------------------------------------

email_tbl |>
  plot_time_series_regression(
    ds,
    log1p(y) ~
      as.numeric(ds) + # linear trend
      lubridate::wday(ds, label = TRUE) + # week day calendar features
      lubridate::month(ds, label = TRUE), # month calendar features
    .show_summary = TRUE
  )

