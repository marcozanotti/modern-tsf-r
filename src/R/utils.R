# Helper Functions

# Function to check packages already loaded into NAMESPACE
check_namespace <- function(pkgs) {

  pkgs_notloaded <- pkgs[!pkgs %in% loadedNamespaces()]
  if (length(pkgs_notloaded) == 0) {
    res <- NULL
  } else {
    res <- pkgs_notloaded
  }
  return(res)

}

# Function to install and load the specified packages
install_and_load <- function(pkgs, repos = getOption("repos")) {

  pkgs_inst <- pkgs[!pkgs %in% installed.packages()]

  if (length(pkgs_inst) == 0) {
    suppressMessages(lapply(pkgs, library, character.only = TRUE, quietly = TRUE))
    check_res <- check_namespace(pkgs)
    if (is.null(check_res)) {
      res <- "All packages correctly installed and loaded."
    } else {
      res <- paste0(
        "Problems loading packages ",
        paste0(check_res, collapse = ", "),
        "."
      )
    }

  } else {

    inst_res <- vector("character", length(pkgs_inst))

    for (i in seq_along(pkgs_inst)) {
      inst_res_tmp <- tryCatch(
        utils::install.packages(pkgs_inst[i], dependencies = TRUE, repos = repos, quiet = TRUE),
        error = function(e) e,
        warning = function(w) w
      )
      if (!is.null(inst_res_tmp)) {
        inst_res[i] <- inst_res_tmp$message
      }
    }

    pkgs_err <- pkgs_inst[!inst_res == ""]
    if (length(pkgs_err) == 0) {
      suppressMessages(lapply(pkgs, library, character.only = TRUE, quietly = TRUE))
      check_res <- check_namespace(pkgs)
      if (is.null(check_res)) {
        res <- "All packages correctly installed and loaded."
      } else {
        res <- paste0(
          "Problems loading packages ",
          paste0(check_res, collapse = ", "),
          "."
        )
      }
    } else {
      pkgs_noerr <- pkgs[!pkgs %in% pkgs_err]
      suppressMessages(lapply(pkgs_noerr, library, character.only = TRUE, quietly = TRUE))
      check_res <- check_namespace(pkgs_noerr)
      if (is.null(check_res)) {
        res <- paste0(
          "Problems installing packages ",
          paste0(pkgs_err, collapse = ", "),
          "."
        )
      } else {
        res <- c(
          paste0(
            "Problems installing packages ",
            paste0(pkgs_err, collapse = ", "),
            "."
          ),
          paste0(
            "Problems loading packages ",
            paste0(check_res, collapse = ", "),
            "."
          )
        )
      }
    }

  }

  message(toupper(
    paste0(
      "\n\n\n",
      "\n==================================================================",
      "\nResults:\n ",
      res,
      "\n=================================================================="
    )
  ))
  return(invisible(res))

}

# functions to load data
load_data <- function(file_path, file_name, ext = ".parquet") {

  full_path <- file.path(file_path, paste0(file_name, ext))

  if (ext == ".parquet") {
    res_df <- arrow::read_parquet(full_path, as_data_frame = TRUE)
  } else if (ext == ".csv") {
    res_df <- readr::read_csv(full_path, show_col_types = FALSE, progress = FALSE)
  } else {
    stop(sprintf("Unsupported file extension '%s'. Only .parquet and .csv are allowed.", ext))
  }

  return(res_df)

}

# function to save data
save_data <- function(df, file_path, file_name, ext = ".parquet", overwrite = TRUE, ...) {

  ext <- if (!startsWith(ext, ".")) paste0(".", ext) else ext
  if (!dir.exists(file_path)) dir.create(file_path, recursive = TRUE)

  full_path <- file.path(file_path, paste0(file_name, ext))

  if (file.exists(full_path) && !overwrite) {
    stop(sprintf("File '%s' already exists. Set overwrite = TRUE to replace it.", full_path))
  }

  if (ext == ".parquet") {
    arrow::write_parquet(df, sink = full_path, ...)
  } else if (ext == ".csv") {
    readr::write_csv(df, file = full_path, ...)
  } else {
    stop(sprintf("Unsupported file extension '%s'. Only .parquet and .csv are allowed.", ext))
  }

  return(invisible(full_path))

}

# function to prepare email data
prepare_email_data <- function(save = TRUE) {
  
  # subscribers
  subscribers <- load_data(
    file_path = "data/email/",
    file_name = "subscribers",
    ext = ".csv"
  )
  subscribers_daily_tbl <- subscribers |>
    summarise_by_time(.date_var = optin_time, .by = "day", y = n()) |>
    pad_by_time(.date_var = optin_time, .by = "day", .pad_value = 0) |> 
    mutate(unique_id = "email_subscribers") |> 
    rename(ds = optin_time) |> 
    select(unique_id, ds, y)

  # analytics
  analytics <- load_data(
    file_path = "data/email/",
    file_name = "analytics_hourly",
    ext = ".csv"
  )
  analytics_daily_tbl <- analytics |>
    mutate(ds = ymd_h(dateHour), .before = everything()) |>
    select(-dateHour) |>
    summarise_by_time(ds, .by = "day", across(pageViews:sessions, .fns = sum)) |> 
    mutate(ds = as.Date(ds))

  # events
  events <- load_data(
    file_path = "data/email/",
    file_name = "events",
    ext = ".csv"
  )
  events_daily_tbl <- events |>
    mutate(ds = as.Date(event_date)) |>
    summarise_by_time(ds, .by = "day", promo = n())

  # join all
  email_data <- subscribers_daily_tbl |>
    left_join(analytics_daily_tbl, by = "ds") |>
    left_join(events_daily_tbl, by = "ds") |>
    replace_na(list(promo = 0))

  if (save) {
    save_data(
      df = email_data,
      file_path = "data/email/",
      file_name = "email_prep",
      ext = ".parquet",
      overwrite = TRUE
    )
  }

  return(email_data)

}

# functions to prepare M4 data sample
prepare_m4_data <- function(n_sample = 8, seed = 123, save = TRUE) {

  is_not_na <- function(x) {
    return(!is.na(x))
  }

  train <- load_data(
    file_path = "data/m4/train/",
    file_name = "Hourly-train",
    ext = ".csv"
  ) |> 
    rename("unique_id" = "V1")
  
  series_len <- train |> is_not_na() |> rowSums()
  set.seed(seed)
  ids <- train[which(series_len == 961), ]$unique_id |> sample(size = n_sample)
  
  train <- train |>
    filter(unique_id %in% ids) |>
    pivot_longer(-unique_id, names_to = "ds", values_to = "y") |> 
    drop_na() |>
    mutate(ds = as.numeric(str_remove_all(ds, "V")) - 1) |> 
    mutate(ds = ymd_hms("2012-01-01 00:00:00") + hours(ds - 1)) |> 
    arrange(unique_id, ds)

  last_ds <- max(train$ds)

  test <- load_data(
    file_path = "data/m4/test/",
    file_name = "Hourly-test",
    ext = ".csv"
  ) |> 
    rename("unique_id" = "V1") |> 
    filter(unique_id %in% ids) |>
    pivot_longer(-unique_id, names_to = "ds", values_to = "y") |> 
    drop_na() |> 
    mutate(ds = as.numeric(str_remove_all(ds, "V")) - 1) |> 
    mutate(ds = ymd_hms(last_ds) + hours(ds)) |> 
    arrange(unique_id, ds)

  m4_sample <- bind_rows(train, test) |> arrange(unique_id, ds)

  if (save) {
    save_data(
      df = m4_sample,
      file_path = "data/m4/",
      file_name = "m4_prep_sample",
      ext = ".parquet",
      overwrite = TRUE
    )
  }

  return(m4_sample)

}

# Function to calibrate models, evaluate their accuracy and plot results
calibrate_evaluate_plot <- function(..., type = "testing", updated_desc = NULL) {

  if (type == "testing") {
    new_data <- testing(splits)
  } else {
    new_data <- training(splits) %>% drop_na()
  }

  calibration_tbl <- modeltime_table(...)

  if (!is.null(updated_desc)) {
    for (i in seq_along(updated_desc)) {
      calibration_tbl <- calibration_tbl %>%
        update_model_description(.model_id = i, .new_model_desc = updated_desc[i])
    }
  }

  calibration_tbl <- calibration_tbl %>%
    modeltime_calibrate(new_data)

  print(calibration_tbl %>% modeltime_accuracy())

  print(
    calibration_tbl %>%
      modeltime_forecast(new_data = new_data, actual_data = data_prep_tbl) %>%
      plot_modeltime_forecast(.conf_interval_show = FALSE)
  )

  return(invisible(calibration_tbl))

}


# Function to extract the .model_id of the "best" model according to a metric
select_best_id <- function(calibration, n = 1, metric = "rmse", by_id = FALSE, id_var = NULL) {

  model_best_id <- calibration %>%
    modeltime_accuracy(acc_by_id = by_id)

  if (by_id) {
    if (is.null(id_var)) {
      stop("Specify the id variable name.")
    }
    model_best_id <- model_best_id %>%
      group_by(!!rlang::sym(id_var))
  }

  if (metric == "rsq") {
    model_best_id <- model_best_id %>%
      slice_max(!!rlang::sym(metric), n = n) %>%
      pull(.model_id)
  } else {
    model_best_id <- model_best_id %>%
      slice_min(!!rlang::sym(metric), n = n) %>%
      pull(.model_id)
  }

  return(model_best_id)

}


# Function to add lags
lag_transf <- function(data){
  data_lags <- data %>%
    tk_augment_lags(optins_trans, .lags = lags)
  return(data_lags)
}


# Function to add lags by group
lag_transf_grouped <- function(data){
  data_lags <- data %>%
    group_by(id) %>%
    tk_augment_lags(optins_trans, .lags = lags) %>%
    ungroup()
  return(data_lags)
}


# Function to calibrate models, evaluate their accuracy and plot results on nested data
nested_calibrate_evaluate_plot <- function(nested_data, workflows, id_var, parallel = FALSE) {

  nested_calibration_tbl <- nested_data %>%
    modeltime_nested_fit(
      model_list = workflows,
      control = control_nested_fit(
        verbose   = TRUE,
        allow_par = parallel
      )
    )

  print(nested_calibration_tbl %>% extract_nested_test_accuracy())

  print(
    nested_calibration_tbl %>%
      extract_nested_test_forecast() %>%
      group_by(!!rlang::sym(id_var)) %>%
      plot_modeltime_forecast(.conf_interval_show = FALSE)
  )

  return(invisible(nested_calibration_tbl))

}


# Function to convert back to original scale
std_logint_inv_vec <- function(x, mean, sd, limit_lower, limit_upper, offset) {

  res <- standardize_inv_vec(x, mean, sd) |>
    log_interval_inv_vec(limit_lower, limit_upper, offset)
  return(res)

}
