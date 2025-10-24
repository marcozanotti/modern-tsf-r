# Modern Time Series Forecasting with R ----

# Lecture Extra: Anomaly Detection ----------------------------------------
# Marco Zanotti

# Goals:
# - Some anomaly detection algorithms
# - Ensemble approach



# Packages ----------------------------------------------------------------

source("R/utils.R")
source("R/packages.R")

library(forecast)
library(anomalize)
library(otsad)
library(anomaly)
library(tsoutliers)
library(stray)



# Data --------------------------------------------------------------------

email_tbl <- read_rds("data/email.rds")


# Pre-processing Data
email_tbl <- email_tbl |>
  summarise_by_time(ds, .by = "day", y = n()) |>
  pad_by_time(.pad_value = 0)

email_tbl |>
  plot_time_series(ds, log1p(y), .smooth = FALSE)

data_prep_tbl <- email_tbl |>
  # pre-processing
  mutate(y = log_interval_vec(y, limit_lower = 0, offset = 1)) |>
  mutate(y = standardize_vec(y)) |>
  # fix missing values at beginning of series
  filter_by_time(.start_date = "2018-07-03") |>
  select(-y)

data_prep_tbl |>
  plot_time_series(ds, y)



# Anomaly detection -------------------------------------------------------

data_prep_tbl |>
  tk_anomaly_diagnostics(.date_var = ds, .value = y)

data_prep_tbl |>
  plot_anomaly_diagnostics(ds, y)


# Functions
anomaly_detection <- function(
    data,
    dates,
    methods = c("forecast", "anomalize", "stray", "otsad", "anomaly"),
    score = TRUE,
    methods_ranking = NULL
) {

  # Initial set up of methods
  # to add a method:
  # 1) increase the num_tot_methods variable
  # 2) add the method name into available_methods variable
  # 3) add the method sub name to full_methods_names
  # 3) apply the method based on if-else conditionals
  data_tbl <- dplyr::tibble("datetime" = dates, "value" = data)
  n_tot_methods <- 10
  available_methods <- c(
    "forecast", "anomalize", "tsoutliers", "otsad",
    "otsad_knn", "anomaly", "stray"
  )
  full_methods_names <- c(
    "forecast", "anomalize", "tsoutliers",
    "otsad_cpp", "otsad_cpsd", "otsad_cpts",
    "otsad_knn",
    "anomaly_capa", "anomaly_scapa",
    "stray"
  )

  zero_vector <- vector("numeric", length(data))
  res_list <- purrr::map(seq_len(n_tot_methods), ~ rep(zero_vector, 1)) %>%
    purrr::set_names(full_methods_names)

  apply_methods <- intersect(methods, available_methods)

  # forecast
  if ("forecast" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method forecast...")
    out <- forecast::tsoutliers(x = data)$index
    res_list[["forecast"]][out] <- 1
  } else {
    res_list[["forecast"]] <- NULL
  }

  # anomalize
  if ("anomalize" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method anomalize...")
    out <- data_tbl %>%
      anomalize::time_decompose(target = "value", method = "twitter", message = FALSE) %>%
      anomalize::anomalize(target = "remainder", method = "gesd") %>%
      dplyr::pull("anomaly")
    res_list[["anomalize"]][which(out == "Yes")] <- 1
  } else {
    res_list[["anomalize"]] <- NULL
  }

  # tsoutliers
  if ("tsoutliers" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method tsoutliers...")
    data_ts <- stats::ts(data)
    out <- tsoutliers::tso(y = data_ts)$outliers$ind
    res_list[["tsoutliers"]][out] <- 1
  } else {
    res_list[["tsoutliers"]] <- NULL
  }

  # otsad
  if ("otsad" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method otsad...")
    n_train <- otsad::GetNumTrainingValues(length(data))
    # CpP
    out <- otsad::CpPewma(data, n_train)
    res_list[["otsad_cpp"]][which(out$is.anomaly == 1)] <- 1
    # CpSd
    out <- otsad::CpSdEwma(data, n_train)
    res_list[["otsad_cpsd"]][which(out$is.anomaly == 1)] <- 1
    # CpTs
    out <- otsad::CpTsSdEwma(data, n_train)
    res_list[["otsad_cpts"]][which(out$is.anomaly == 1)] <- 1
  } else {
    res_list[["otsad_cpp"]] <- NULL # CpP
    res_list[["otsad_cpsd"]] <- NULL # CpSd
    res_list[["otsad_cpts"]] <- NULL # CpTs
  }

  # otsad_knn
  if ("otsad_knn" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method otsad knn...")
    n_train <- otsad::GetNumTrainingValues(length(data))
    k_groups <- length(data) * 0.1 # 10% of data points taken into account
    out <- otsad::CpKnnCad(data, n_train, threshold = 0.95, k = k_groups)
    res_list[["otsad_knn"]][which(out$is.anomaly == 1)] <- 1
  } else {
    res_list[["otsad_knn"]] <- NULL
  }

  # anomaly
  if ("anomaly" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method anomaly...")
    out <- anomaly::point_anomalies(anomaly::capa.uv(data, transform = scale)) # base scale function as transform to avoid NaN problems
    res_list[["anomaly_capa"]][out[out$strength > 0, "location"]] <- 1
    # SCAPA
    out <- anomaly::point_anomalies(anomaly::scapa.uv(data, transform = scale)) # base scale function as transform to avoid NaN problems
    res_list[["anomaly_capa"]][out[out$strength > 0, "location"]] <- 1
  } else {
    res_list[["anomaly_capa"]] <- NULL # CAPA
    res_list[["anomaly_scapa"]] <- NULL # SCAPA
  }

  # stray
  if ("stray" %in% apply_methods) {
    logging::loginfo("Detecting anomalies through method stray...")
    n_train <- otsad::GetNumTrainingValues(length(data))
    k_groups <- length(data) * 0.1 # 10% of data points taken into account
    out <- stray::find_HDoutliers(
      data,
      k = k_groups,	knnsearchtype = "kd_tree",
      alpha = .05, p = .05, tn = n_train
    )$outliers
    res_list[["stray"]][out] <- 1
  } else {
    res_list[["stray"]] <- NULL
  }


  res_df <- dplyr::bind_cols(res_list)


  if (score) {

    score <- anomaly_score(res_df)
    if (is.null(methods_ranking)) {methods_ranking <- methods}
    ws <- rev(seq_along(res_df)) / sum(seq_along(res_df)) # compute weights as ranking
    res_df <- dplyr::select(res_df, dplyr::contains(methods_ranking)) # arrange columns
    score_w <- anomaly_score(res_df, ws)

    res_df$score <- score
    res_df$score_w <- score_w
    res_df <- dplyr::select(
      res_df,
      "score", dplyr::contains("score_w"), dplyr::everything()
    )

  }

  res_df <- dplyr::bind_cols(data_tbl, res_df)


  return(res_df)


}

anomaly_score <- function(anomaly_data, weights = NULL) {

  if (is.null(weights)) {
    logging::loginfo("Computing anomaly score...")
    score <- rowSums(anomaly_data) / ncol(anomaly_data)
  } else {
    logging::loginfo("Computing anomaly weighted score...")
    score <- rowSums(purrr::map2_df(anomaly_data, weights, ~ .x * .y))
  }
  return(score)

}


# Test functions
# data = data_prep_tbl$y
# dates = data_prep_tbl$ds

res_df <- anomaly_detection(
  data = data_prep_tbl$y,
  dates = data_prep_tbl$ds
)

res_df

res_df |>
  plot_time_series(.date_var = datetime, .value = score)
res_df |>
  ggplot(aes(x = datetime, y = value, col = score)) +
  geom_line()

