suppressPackageStartupMessages(library(ctmle))
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

# X22 part (g), the comparator side.  `ctmleDiscrete(preOrder = TRUE, patience = 2)` stops the
# cross-validated scoring of the pre-ordered path when the count of scores worse than the best so
# far reaches 2; `patience = FALSE` scores the whole path.  Both runs get the same sample, the same
# initial outcome regression Q and the same selection folds.  The script exports the sample, the
# folds, Q, and for each run the selected index, the estimate and the cross-validated score of
# every candidate (null where the stop left the candidate unscored).
#
# Sources read at ctmle 0.1.2, commit 18de559 (`R/ctmle_discrete.R`, `R/functions_discrete.R`):
#   ctmleDiscrete(Y, A, W, Wg = W, Q = NULL, preOrder = FALSE, order = NULL, patience = FALSE,
#     Qbounds, cvQinit, Qform, SL.library, alpha, family, gbound, like_type, fluctuation, verbose,
#     detailed, PEN, V, folds, stopFactor)
#   With preOrder = TRUE, `Wg <- Wg[, order]`, and `cv(..., patience, preOrder, folds)` scores the
#   candidates "A ~ 1", then one more covariate per pass.  `cv.res$penlikelihood` is the score
#   (Inf where unscored) and `cv.res$best_k` the last position of the smallest finite score.
#   `detailed = TRUE` keeps `cv.res` and `candidates` in the value.
#   `folds` is a list of V index vectors; `V` must equal `length(folds)`.

output <- fixture_arguments("usage: ctmle_patience.R OUTPUT.json")
provenance <- fixture_provenance("ctmle_patience.R", "ctmle")

SEED <- 20261018L
N <- 600L
P <- 6L
V <- 5L
PATIENCE <- 2L

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)
Wmatrix <- matrix(rnorm(N * P), nrow = N, dimnames = list(NULL, paste0("W", seq_len(P))))
A <- rbinom(N, 1L, plogis(0.9 * Wmatrix[, 1]))
# W1 confounds; W2 to W6 are noise.  The initial outcome regression leaves W1 out, so adding W1
# to the treatment model helps and the noise covariates after it do not.
Y <- rbinom(N, 1L, plogis(-0.2 + 0.5 * A + 0.8 * Wmatrix[, 1]))
W <- as.data.frame(Wmatrix)
frame <- data.frame(Y = Y, A = A, W)

Q_COVARIATES <- c("W2", "W3", "W4", "W5", "W6")
q_fit <- glm(reformulate(c("A", Q_COVARIATES), response = "Y"), data = frame, family = binomial())
q0 <- as.numeric(predict(q_fit, newdata = transform(frame, A = 0), type = "response"))
q1 <- as.numeric(predict(q_fit, newdata = transform(frame, A = 1), type = "response"))

assignment <- sample(rep(seq_len(V) - 1L, length.out = N))
folds <- lapply(seq_len(V) - 1L, function(k) which(assignment == k))
stopifnot(
  sum(lengths(folds)) == N,
  all(vapply(folds, function(rows) length(unique(A[rows])) == 2L, logical(1)))
)

run <- function(patience) {
  fit <- ctmle::ctmleDiscrete(
    Y = frame$Y, A = frame$A, W = W, Wg = W, Q = cbind(q0, q1),
    preOrder = TRUE, order = seq_len(P), patience = patience,
    family = "binomial", like_type = "loglike", gbound = 0.025, PEN = FALSE,
    V = V, folds = folds, detailed = TRUE
  )
  score <- as.numeric(fit$cv.res$penlikelihood)
  list(
    patience = if (isFALSE(patience)) NULL else patience,
    best_k = as.integer(fit$best_k),
    est = as.numeric(fit$est),
    cv_score = score,
    scored = sum(is.finite(score)),
    candidates = length(score),
    stopped_early = sum(is.finite(score)) < length(score),
    terms = as.character(fit$candidates$terms)
  )
}

payload <- list(
  fixture = "ctmle_patience",
  schema = 1L,
  provenance = provenance,
  seed = SEED,
  n = N,
  patience = PATIENCE,
  data = c(list(Y = as.numeric(Y), A = as.numeric(A)), as.list(W)),
  covariates = colnames(W),
  q_covariates = Q_COVARIATES,
  fold = as.integer(assignment),
  q_arm0 = q0,
  q_arm1 = q1,
  runs = list(whole_path = run(FALSE), patient = run(PATIENCE))
)
fixture_write(payload, output)
