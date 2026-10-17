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

# X22 part (h), the comparator side.  `Wg` is the covariate set of the treatment mechanism and `W`
# the set of the outcome regression (here only through the initial Q).  The script runs
# `ctmleDiscrete(preOrder = TRUE)` twice on one sample, once with `Wg = W[, c(1, 3)]` and once with
# `Wg = W`, and exports the selected index, the estimate, the cross-validated score of every
# candidate and the covariates the path added.  W2 drives the treatment strongly and lies outside
# the restricted set, so the two runs differ.
#
# Sources read at ctmle 0.1.2, commit 18de559 (`R/ctmle_discrete.R`, `R/functions_discrete.R`):
#   ctmleDiscrete(Y, A, W, Wg = W, Q, preOrder, order, patience, family, gbound, like_type, PEN,
#     V, folds, detailed, ...).  With preOrder = TRUE the value `Wg[, order]` is the candidate pool:
#   `order` indexes the columns of `Wg`, not of `W`.  Restricting to c(1, 3) and ordering c(1, 2)
#   adds W1 and then W3.
#   `cv.res$penlikelihood` is the score per candidate and `cv.res$best_k` the selected position;
#   `candidates$terms` lists the covariates the path added.

output <- fixture_arguments("usage: ctmle_wg.R OUTPUT.json")
provenance <- fixture_provenance("ctmle_wg.R", "ctmle")

SEED <- 20261019L
N <- 600L
P <- 4L
V <- 5L
WG_COLUMNS <- c(1L, 3L)

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)
Wmatrix <- matrix(rnorm(N * P), nrow = N, dimnames = list(NULL, paste0("W", seq_len(P))))
A <- rbinom(N, 1L, plogis(0.8 * Wmatrix[, 1] + 1.1 * Wmatrix[, 2] + 0.3 * Wmatrix[, 3]))
Y <- rbinom(N, 1L, plogis(-0.2 + 0.5 * A + 0.8 * Wmatrix[, 1] + 0.6 * Wmatrix[, 2] + 0.4 * Wmatrix[, 3]))
W <- as.data.frame(Wmatrix)
frame <- data.frame(Y = Y, A = A, W)

# An initial outcome regression that omits W2, which drives both treatment and outcome, so adding
# W2 to the treatment mechanism has something to correct.
Q_COVARIATES <- c("W1", "W3", "W4")
q_fit <- glm(reformulate(c("A", Q_COVARIATES), response = "Y"), data = frame, family = binomial())
q0 <- as.numeric(predict(q_fit, newdata = transform(frame, A = 0), type = "response"))
q1 <- as.numeric(predict(q_fit, newdata = transform(frame, A = 1), type = "response"))

assignment <- sample(rep(seq_len(V) - 1L, length.out = N))
folds <- lapply(seq_len(V) - 1L, function(k) which(assignment == k))
stopifnot(
  sum(lengths(folds)) == N,
  all(vapply(folds, function(rows) length(unique(A[rows])) == 2L, logical(1)))
)

run <- function(wg, order) {
  fit <- ctmle::ctmleDiscrete(
    Y = frame$Y, A = frame$A, W = W, Wg = wg, Q = cbind(q0, q1),
    preOrder = TRUE, order = order,
    family = "binomial", like_type = "loglike", gbound = 0.025, PEN = FALSE,
    V = V, folds = folds, detailed = TRUE
  )
  score <- as.numeric(fit$cv.res$penlikelihood)
  list(
    pool = colnames(wg),
    order = as.integer(order),
    best_k = as.integer(fit$best_k),
    est = as.numeric(fit$est),
    cv_score = score,
    terms = as.character(fit$candidates$terms)
  )
}

payload <- list(
  fixture = "ctmle_wg",
  schema = 1L,
  provenance = provenance,
  seed = SEED,
  n = N,
  data = c(list(Y = as.numeric(Y), A = as.numeric(A)), as.list(W)),
  covariates = colnames(W),
  q_covariates = Q_COVARIATES,
  wg_columns = WG_COLUMNS,
  fold = as.integer(assignment),
  q_arm0 = q0,
  q_arm1 = q1,
  runs = list(
    restricted = run(W[, WG_COLUMNS], seq_along(WG_COLUMNS)),
    whole_pool = run(W, seq_len(P))
  )
)
fixture_write(payload, output)
