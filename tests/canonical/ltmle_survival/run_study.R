suppressPackageStartupMessages(library(ltmle))
options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args[[1]]), stringsAsFactors = FALSE, check.names = FALSE)
truths <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
output <- args[[3]]

scenario <- "censored_survival_curve"
dynamic_label <- "treat then continue if l2 positive"
z <- qnorm(0.975)

assigned <- function(frame, label, horizon) {
  a1 <- if (label == "never") 0 else 1
  if (horizon == 1) return(matrix(rep(a1, nrow(frame)), ncol = 1))
  a2 <- if (label == "never") {
    rep(0, nrow(frame))
  } else if (label == "always") {
    rep(1, nrow(frame))
  } else {
    as.numeric(!is.na(frame$L2) & frame$L2 > 0)
  }
  cbind(rep(a1, nrow(frame)), a2)
}

mechanism <- function(frame, abar, horizon) {
  p_a1 <- plogis(0.3 * frame$W1 - 0.4 * frame$W2)
  p_c1 <- plogis(2.2 + 0.3 * frame$W1 - 0.3 * abar[, 1])
  if (horizon == 1) return(cbind(p_a1, p_c1))
  l2 <- ifelse(is.na(frame$L2), 0, frame$L2)
  p_a2 <- plogis(0.5 * l2 + 0.6 * abar[, 1] - 0.2 * frame$W2)
  p_c2 <- plogis(2.4 + 0.2 * l2)
  cbind(p_a1, p_c1, p_a2, p_c2)
}

fit_regimen <- function(frame, label, horizon) {
  abar <- assigned(frame, label, horizon)
  g <- mechanism(frame, abar, horizon)
  if (horizon == 1) {
    data <- frame[c("W1", "W2", "A1", "C1", "Y1")]
    arguments <- list(
      data = data,
      Anodes = "A1",
      Cnodes = "C1",
      Ynodes = "Y1",
      survivalOutcome = TRUE,
      Qform = c(Y1 = "Q.kplus1 ~ W1 + W2")
    )
  } else {
    data <- frame[c("W1", "W2", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")]
    arguments <- list(
      data = data,
      Anodes = c("A1", "A2"),
      Cnodes = c("C1", "C2"),
      Lnodes = "L2",
      Ynodes = c("Y1", "Y2"),
      survivalOutcome = TRUE,
      Qform = c(Y1 = "Q.kplus1 ~ W1 + W2", Y2 = "Q.kplus1 ~ W1 + W2 + L2")
    )
  }
  arguments <- c(
    arguments,
    list(
      gform = g,
      abar = abar,
      gbounds = c(1e-8, 1),
      SL.library = "glm",
      stratify = TRUE,
      variance.method = "ic"
    )
  )
  targeted <- withCallingHandlers(
    do.call(ltmle, arguments),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      stop(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
  if (any(abs(targeted$cum.g - targeted$cum.g.unbounded) > 1e-12, na.rm = TRUE)) {
    stop(sprintf("%s at horizon %d activated a cumulative g bound", label, horizon))
  }
  first_q <- targeted$fit$Q[[1]]
  initial <- if (inherits(first_q, "no.Y.variation")) {
    unname(first_q$Y.value)
  } else {
    coefficients <- first_q[, "Estimate"]
    design <- model.matrix(~ W1 + W2, data = frame)
    unname(mean(plogis(design[, names(coefficients), drop = FALSE] %*% coefficients)))
  }
  list(
    estimate = unname(targeted$estimates[["tmle"]]),
    initial = initial,
    ic = as.numeric(targeted$IC$tmle)
  )
}

truth_for <- function(replicate, estimand) {
  selected <- truths$replicate == replicate & truths$scenario == scenario & truths$estimand == estimand
  if (sum(selected) != 1) stop(sprintf("truth join found %d rows", sum(selected)))
  truths$truth[selected]
}

row_for <- function(replicate, name, estimate, initial, ic, n) {
  truth <- truth_for(replicate, name)
  standard_error <- sd(ic) / sqrt(n)
  low <- estimate - z * standard_error
  high <- estimate + z * standard_error
  data.frame(
    implementation = "ltmle",
    scenario = scenario,
    replicate = replicate,
    n = n,
    estimand = name,
    truth = truth,
    estimate = estimate,
    inference_estimate = estimate,
    std_error = standard_error,
    ci_lower = low,
    ci_upper = high,
    inference_scale = "identity",
    covered = as.integer(low <= truth && truth <= high),
    initial_estimate = initial,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  fits <- list(
    "never @ t=1" = fit_regimen(frame, "never", 1),
    "always @ t=1" = fit_regimen(frame, "always", 1),
    "never @ t=2" = fit_regimen(frame, "never", 2),
    "always @ t=2" = fit_regimen(frame, "always", 2),
    "dynamic @ t=2" = fit_regimen(frame, dynamic_label, 2)
  )
  labels <- list(
    c("never", "1", "never @ t=1"),
    c("never", "2", "never @ t=2"),
    c("always", "1", "always @ t=1"),
    c("always", "2", "always @ t=2"),
    c(dynamic_label, "2", "dynamic @ t=2")
  )
  rows <- lapply(labels, function(spec) {
    fit <- fits[[spec[[3]]]]
    name <- sprintf("risk_regimen[%s @ t=%s]", spec[[1]], spec[[2]])
    row_for(replicate, name, fit$estimate, fit$initial, fit$ic, nrow(frame))
  })
  comparisons <- list(
    c("always", "1", "always @ t=1", "never @ t=1"),
    c("always", "2", "always @ t=2", "never @ t=2"),
    c(dynamic_label, "2", "dynamic @ t=2", "never @ t=2")
  )
  for (spec in comparisons) {
    left <- fits[[spec[[3]]]]
    right <- fits[[spec[[4]]]]
    name <- sprintf("ate_regimen[%s vs never @ t=%s]", spec[[1]], spec[[2]])
    rows[[length(rows) + 1]] <- row_for(
      replicate,
      name,
      left$estimate - right$estimate,
      left$initial - right$initial,
      left$ic - right$ic,
      nrow(frame)
    )
  }
  do.call(rbind, rows)
}

groups <- split(samples, samples$replicate)
expected <- length(groups)
rm(samples)
invisible(gc())

fit_group <- function(index) {
  frame <- groups[[index]]
  tryCatch(
    {
      result <- fit_one(frame)
      if (index %% 10 == 0) cat(sprintf("completed %d/%d samples\n", index, expected))
      result
    },
    error = function(condition) {
      structure(
        list(index = index, replicate = frame$replicate[[1]], message = conditionMessage(condition)),
        class = "study_error"
      )
    }
  )
}

requested <- suppressWarnings(as.integer(Sys.getenv("CLEVERLY_R_CORES", "")))
cores <- if (is.na(requested) || requested < 1L) max(1L, parallel::detectCores()) else requested
cat(sprintf("fitting %d samples on %d cores\n", expected, cores))
results <- parallel::mclapply(seq_along(groups), fit_group, mc.cores = cores, mc.preschedule = FALSE)
failed <- vapply(results, inherits, logical(1), what = "study_error")
if (any(failed)) {
  messages <- vapply(
    results[failed],
    function(error) sprintf("replicate %s (group %s): %s", error$replicate, error$index, error$message),
    character(1)
  )
  stop(paste(messages, collapse = "\n"))
}
if (!all(vapply(results, is.data.frame, logical(1)))) stop("an R worker returned no result")
result <- do.call(rbind, results)
if (length(unique(result$replicate)) != expected) stop("the R run dropped replications")
write.csv(result, output, row.names = FALSE, na = "")
cat("ltmle ", as.character(packageVersion("ltmle")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
