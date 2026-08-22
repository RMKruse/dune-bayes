# BAMLSS fixture generator for dune-bayes issues #107 and #175.
# Script version: issue-0175-response-standardization-v1
# R version pinned for fixture generation: R 4.4.x
# bamlss package version pinned for fixture generation: bamlss 1.2-5
# Required package versions: yaml 2.3.10, jsonlite 2.0.0, reticulate 1.39.0
# Output schema: predictions.csv has dataset, observation, log_density, cdf,
# q05, q50, q95, and response predictive draws sample_0001...sample_N.

suppressPackageStartupMessages({
  library(bamlss)
  library(jsonlite)
  library(reticulate)
  library(yaml)
})

SCRIPT_VERSION <- "issue-0175-response-standardization-v1"
EPS <- 1e-6

usage <- function() {
  cat(
    paste(
      "Usage:",
      "Rscript experiments/uci_benchmark/bamlss/run.R",
      "--config experiments/uci_benchmark/config.yaml",
      "--dataset autompg --smoke --output-dir experiments/uci_benchmark/fixtures/bamlss",
      "[--seed 10701] [--n-iter 4000] [--burnin 1000] [--thin 5]",
      "[--predictive-samples 500]",
      sep = "\n"
    ),
    "\n"
  )
}

parse_args <- function(argv) {
  args <- list(
    seed = 10701L,
    smoke = FALSE,
    n_iter = 4000L,
    burnin = 1000L,
    thin = 5L,
    predictive_samples = 500L
  )
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--help" || key == "-h") {
      usage()
      quit(status = 0)
    } else if (key == "--smoke") {
      args$smoke <- TRUE
      i <- i + 1L
      next
    }
    if (i == length(argv)) {
      stop("Missing value for ", key, call. = FALSE)
    }
    value <- argv[[i + 1L]]
    if (key == "--config") {
      args$config <- value
    } else if (key == "--dataset") {
      args$dataset <- value
    } else if (key == "--output-dir") {
      args$output_dir <- value
    } else if (key == "--seed") {
      args$seed <- as.integer(value)
    } else if (key == "--n-iter") {
      args$n_iter <- as.integer(value)
    } else if (key == "--burnin") {
      args$burnin <- as.integer(value)
    } else if (key == "--thin") {
      args$thin <- as.integer(value)
    } else if (key == "--predictive-samples") {
      args$predictive_samples <- as.integer(value)
    } else {
      stop("Unknown argument: ", key, call. = FALSE)
    }
    i <- i + 2L
  }
  for (required in c("config", "dataset", "output_dir")) {
    if (is.null(args[[required]])) {
      stop("Missing required argument --", required, call. = FALSE)
    }
  }
  args
}

resolve_path <- function(path, base_dir) {
  if (grepl("^(/|[A-Za-z]:)", path)) {
    return(normalizePath(path, mustWork = FALSE))
  }
  normalizePath(file.path(base_dir, path), mustWork = FALSE)
}

cache_key <- function(dataset_name, smoke) {
  if (isTRUE(smoke)) {
    paste0(dataset_name, "-smoke")
  } else {
    dataset_name
  }
}

log_mean_exp <- function(values) {
  pivot <- max(values)
  pivot + log(mean(exp(values - pivot)))
}

as_draw_matrix <- function(value, n_obs, label) {
  matrix_value <- as.matrix(value)
  if (ncol(matrix_value) == n_obs) {
    return(matrix_value)
  }
  if (nrow(matrix_value) == n_obs) {
    return(t(matrix_value))
  }
  stop(label, " predictions have shape ", paste(dim(matrix_value), collapse = "x"),
       ", expected one dimension to equal ", n_obs, call. = FALSE)
}

mean_term <- function(feature, train) {
  unique_count <- length(unique(train[[feature]]))
  if (unique_count <= 4L) {
    return(feature)
  }
  # mgcv refuses a spline basis with k larger than the number of unique
  # covariate values; smoke fixtures are intentionally tiny.
  basis_rank <- min(10L, unique_count)
  sprintf("s(%s, k = %d)", feature, basis_rank)
}

main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  args <- parse_args(argv)
  set.seed(args$seed)

  config_path <- normalizePath(args$config, mustWork = TRUE)
  experiment_dir <- dirname(config_path)
  config <- yaml::read_yaml(config_path)
  dataset <- NULL
  for (candidate in config$datasets) {
    if (identical(candidate$name, args$dataset)) {
      dataset <- candidate
      break
    }
  }
  if (is.null(dataset)) {
    stop("Unknown dataset in config: ", args$dataset, call. = FALSE)
  }
  if (!identical(dataset$family, "normal")) {
    stop(
      "The committed BAMLSS fixture script currently supports normal datasets; got ",
      dataset$family,
      call. = FALSE
    )
  }

  key <- cache_key(dataset$name, args$smoke)
  cache_path <- file.path(
    resolve_path(config$data$cache_dir, experiment_dir),
    paste0(key, ".csv")
  )
  split_path <- file.path(
    resolve_path(config$data$split_dir, experiment_dir),
    paste0(key, ".npz")
  )
  frame <- read.csv(cache_path, check.names = TRUE)
  response <- make.names(dataset$response)
  numpy <- reticulate::import("numpy", convert = FALSE)
  split <- numpy$load(split_path)
  train_indices <- as.integer(reticulate::py_to_r(split[["train_indices"]])) + 1L
  test_indices <- as.integer(reticulate::py_to_r(split[["test_indices"]])) + 1L
  train <- frame[train_indices, , drop = FALSE]
  test <- frame[test_indices, , drop = FALSE]

  response_loc <- mean(train[[response]])
  response_scale <- max(
    sqrt(mean((train[[response]] - response_loc)^2)),
    EPS
  )
  train[[response]] <- (train[[response]] - response_loc) / response_scale

  feature_names <- setdiff(names(train), response)
  mean_terms <- vapply(feature_names, mean_term, character(1), train = train)
  formula_mu <- as.formula(paste(response, "~", paste(mean_terms, collapse = " + ")))
  formula_sigma <- as.formula("~ 1")
  fit <- bamlss::bamlss(
    list(mu = formula_mu, sigma = formula_sigma),
    family = "gaussian",
    data = train,
    n.iter = args$n_iter,
    burnin = args$burnin,
    thin = args$thin
  )

  parameters <- predict(fit, newdata = test, type = "parameter", FUN = identity)
  n_obs <- nrow(test)
  mu <- as_draw_matrix(parameters$mu, n_obs, "mu")
  sigma <- as_draw_matrix(parameters$sigma, n_obs, "sigma")
  mu <- mu * response_scale + response_loc
  sigma <- sigma * response_scale
  draw_count <- nrow(mu)
  if (draw_count < 1L) {
    stop("BAMLSS returned no posterior parameter draws.", call. = FALSE)
  }

  y <- test[[response]]
  posterior_draws <- sample(seq_len(draw_count), args$predictive_samples, replace = TRUE)
  response_samples <- matrix(NA_real_, nrow = args$predictive_samples, ncol = n_obs)
  for (i in seq_len(args$predictive_samples)) {
    draw <- posterior_draws[[i]]
    response_samples[i, ] <- rnorm(n_obs, mean = mu[draw, ], sd = sigma[draw, ])
  }

  output <- data.frame(
    dataset = dataset$name,
    observation = seq_len(n_obs) - 1L,
    log_density = vapply(
      seq_len(n_obs),
      function(i) log_mean_exp(dnorm(y[[i]], mean = mu[, i], sd = sigma[, i], log = TRUE)),
      numeric(1)
    ),
    cdf = vapply(
      seq_len(n_obs),
      function(i) mean(pnorm(y[[i]], mean = mu[, i], sd = sigma[, i])),
      numeric(1)
    ),
    q05 = apply(response_samples, 2L, quantile, probs = 0.05, names = FALSE),
    q50 = apply(response_samples, 2L, quantile, probs = 0.50, names = FALSE),
    q95 = apply(response_samples, 2L, quantile, probs = 0.95, names = FALSE)
  )
  sample_columns <- as.data.frame(t(response_samples))
  names(sample_columns) <- sprintf("sample_%04d", seq_len(args$predictive_samples))
  output <- cbind(output, sample_columns)

  dataset_dir <- file.path(resolve_path(args$output_dir, getwd()), dataset$name)
  dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(output, file.path(dataset_dir, "predictions.csv"), row.names = FALSE)
  jsonlite::write_json(
    list(
      script_version = SCRIPT_VERSION,
      seed = args$seed,
      date = as.character(Sys.Date()),
      dataset = dataset$name,
      family = dataset$family,
      cache_path = cache_path,
      split_path = split_path,
      n_train = nrow(train),
      n_test = nrow(test),
      predictive_samples = args$predictive_samples,
      response_transform = list(
        method = "standard",
        fit_partition = "train",
        n_fit = nrow(train),
        loc = response_loc,
        scale = response_scale
      ),
      r_version = R.version.string,
      package_versions = list(
        bamlss = as.character(utils::packageVersion("bamlss")),
        yaml = as.character(utils::packageVersion("yaml")),
        jsonlite = as.character(utils::packageVersion("jsonlite")),
        reticulate = as.character(utils::packageVersion("reticulate"))
      ),
      session_info = capture.output(utils::sessionInfo())
    ),
    file.path(dataset_dir, "provenance.json"),
    auto_unbox = TRUE,
    # Preserve enough digits for the Python boundary's float32 EPS check.
    digits = 17,
    pretty = TRUE
  )
}

main()
