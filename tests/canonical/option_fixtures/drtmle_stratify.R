suppressPackageStartupMessages(library(drtmle))
options(digits = 17)

# ---- option fixtures shared: begin ----
# The four scripts in tests/canonical/option_fixtures carry this block byte for byte;
# tests/unit/test_option_fixtures.py compares the copies.  Change it in all four.
#
# It writes the fixture as JSON with base R alone (the pinned images need not hold jsonlite) and
# records where the fixture came from.  A length-one vector is written as a scalar and a longer
# one as an array; a missing or non-finite number is written as null.  Floats carry 17
# significant digits, so the file round-trips exactly.

fixture_arguments <- function(usage) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 1L) stop(usage)
  args[[1]]
}

fixture_string <- function(value) {
  if (is.na(value)) "null" else encodeString(value, quote = "\"")
}

fixture_number <- function(value) {
  if (is.na(value) || !is.finite(value)) {
    "null"
  } else if (value == round(value) && abs(value) < 1e15) {
    sprintf("%.0f", value)
  } else {
    sprintf("%.17g", value)
  }
}

fixture_json <- function(x, indent = 0L) {
  pad <- strrep("  ", indent)
  inner <- strrep("  ", indent + 1L)
  if (is.null(x)) {
    return("null")
  }
  if (is.list(x)) {
    if (length(x) == 0L) {
      return(if (is.null(names(x))) "[]" else "{}")
    }
    parts <- vapply(x, fixture_json, character(1), indent = indent + 1L, USE.NAMES = FALSE)
    if (!is.null(names(x))) {
      keys <- vapply(names(x), fixture_string, character(1), USE.NAMES = FALSE)
      return(paste0("{\n", paste0(inner, keys, ": ", parts, collapse = ",\n"), "\n", pad, "}"))
    }
    return(paste0("[\n", paste0(inner, parts, collapse = ",\n"), "\n", pad, "]"))
  }
  if (is.character(x)) {
    values <- vapply(x, fixture_string, character(1), USE.NAMES = FALSE)
  } else if (is.logical(x)) {
    values <- ifelse(is.na(x), "null", ifelse(x, "true", "false"))
  } else if (is.numeric(x)) {
    values <- vapply(as.numeric(x), fixture_number, character(1), USE.NAMES = FALSE)
  } else {
    stop(sprintf("cannot write a %s to JSON", class(x)[[1]]))
  }
  if (length(values) == 1L) values else paste0("[", paste(values, collapse = ", "), "]")
}

fixture_provenance <- function(script, packages) {
  # The checksum is the script's own, so a JSON file says which script wrote it.  The queue
  # command also passes CLEVERLY_SCRIPT_SHA256, used when tools::sha256sum is absent.
  file_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  path <- if (length(file_argument) > 0L) sub("^--file=", "", file_argument[[1]]) else NA_character_
  digest <- Sys.getenv("CLEVERLY_SCRIPT_SHA256", unset = NA_character_)
  if (!is.na(path) && exists("sha256sum", envir = asNamespace("tools"), inherits = FALSE)) {
    digest <- unname(tools::sha256sum(path))
  }
  list(
    script = script,
    script_sha256 = digest,
    image_id = Sys.getenv("CLEVERLY_IMAGE_ID", unset = NA_character_),
    r_version = R.version.string,
    packages = lapply(
      stats::setNames(nm = packages),
      function(name) as.character(utils::packageVersion(name))
    )
  )
}

fixture_write <- function(payload, path) {
  # Raw bytes, so the file ends in LF whatever the platform's text-mode habits are.
  text <- paste0(fixture_json(payload), "\n")
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(charToRaw(text), connection)
  invisible(path)
}
# ---- option fixtures shared: end ----

# X22 part (f), the comparator side.  `drtmle(stratify = TRUE)` fits the outcome regression on the
# rows with A = a for each arm a, and `stratify = FALSE` fits one regression on [A, W].  The
# script exports the sample and, for each setting, the initial outcome regression at both arms,
# the propensity at both arms and the TMLE estimate.
#
# Sources read at drtmle 1.1.2, commit 538a3a2 (`R/drtmle.R`, `R/estimate.R`):
#   drtmle(Y, A, W, DeltaA, DeltaY, a_0, family, stratify, SL_Q, SL_g, glm_Q, glm_g, glm_Qr,
#     glm_gr, guard, reduction, returnModels, returnNuisance, cvFolds, maxIter, tolIC, tolg, ...)
#   estimateQ, stratified GLM: `glm(trainY[include2 & include] ~ <glm_Q>, data = trainW[...])` for
#     each arm in a_0, predicted on every row.
#   estimateQ, pooled GLM: `glm(Y ~ <glm_Q>, data = data.frame(Y, A, trainW))`, predicted at
#     A = a for each arm.  The pooled formula must name A, or the fit ignores the treatment.
#   The result holds `nuisance_aiptw_c$Qn` and `$gn` (a list over a_0, in a_0 order), and
#   `tmle$est` (a vector over a_0).  `guard = NULL` runs no extra targeting, so `drtmle$est` and
#   `tmle$est` are the same numbers.

output <- fixture_arguments("usage: drtmle_stratify.R OUTPUT.json")
provenance <- fixture_provenance("drtmle_stratify.R", "drtmle")

SEED <- 20261017L
N <- 400L
TOLG <- 0.01

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)
W1 <- rnorm(N)
W2 <- rnorm(N)
A <- rbinom(N, 1L, plogis(0.3 * W1 - 0.2 * W2))
# The effect of A changes sign with W2, so an additive-in-A regression cannot follow the law and
# the stratified fit differs from the pooled one.
Y <- rbinom(N, 1L, plogis(-0.2 + 0.4 * W1 - 0.3 * W2 + A * (0.6 + 1.2 * W2)))
W <- data.frame(W1 = W1, W2 = W2)

numbers <- function(x) as.numeric(unlist(x, use.names = FALSE))

fit_setting <- function(stratify) {
  fit <- drtmle::drtmle(
    Y = Y, A = A, W = W,
    a_0 = c(0, 1),
    family = binomial(),
    stratify = stratify,
    glm_Q = if (stratify) "W1 + W2" else "A + W1 + W2",
    glm_g = "W1 + W2",
    guard = NULL,
    cvFolds = 1,
    tolg = TOLG,
    returnNuisance = TRUE
  )
  initial <- fit$nuisance_aiptw_c
  g_arms <- lapply(initial$gn, numbers)
  # The truncation at tolg must not bind, or the propensity differs from the fitted glm.
  stopifnot(all(unlist(g_arms) > TOLG))
  list(
    stratify = stratify,
    formula_q = if (stratify) "W1 + W2" else "A + W1 + W2",
    qn_arm0 = numbers(initial$Qn[[1]]),
    qn_arm1 = numbers(initial$Qn[[2]]),
    gn_arm0 = g_arms[[1]],
    gn_arm1 = g_arms[[2]],
    gcomp = numbers(fit$gcomp$est),
    tmle = numbers(fit$tmle$est),
    drtmle = numbers(fit$drtmle$est)
  )
}

payload <- list(
  fixture = "drtmle_stratify",
  schema = 1L,
  provenance = provenance,
  seed = SEED,
  n = N,
  tolg = TOLG,
  arm_levels = c(0, 1),
  data = list(Y = as.numeric(Y), A = as.numeric(A), W1 = W1, W2 = W2),
  settings = list(fit_setting(TRUE), fit_setting(FALSE))
)
fixture_write(payload, output)
