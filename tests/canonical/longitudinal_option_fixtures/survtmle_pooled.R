suppressPackageStartupMessages(library(survtmle))
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

# X33 part (j), the comparator side.  `hazard_pooling = "pooled"` fits one censoring regression
# over every at-risk (unit, node) row with the node's time as a column, as survtmle's
# `glm.ctime` reads `t`.  The script fits survtmle twice on one sample, first with one
# intercept and slope set per visit (which `hazard_pooling = "node"` equals), then with one
# linear visit term, and prints each fit's censoring hazard at every unit and visit.
#
# Sources read at survtmle 1.1.1 (CRAN; the cran/survtmle mirror, tag 1.1.1):
#   makeDataList.R: one long row per unit and visit up to its observed time; C = 1 on the row
#     of a censoring.  A unit censored at visit s is observed event-free through s.
#   censoring_estimate.R, estimateCensoring: fits `C ~ <glm.ctime>` on the long rows except an
#     event row before t0 and a censored row at t0, and predicts it at t - 1, with G_dC = 1 at
#     t = 1.  So visit s's censoring model is the retention model of node s + 1.
# The law has no event at the last visit, which is t0.  survtmle's rule keeps an event row at
# t0 with C = 0; with no such row, its fitted rows are the at-risk rows of visits 1 to 3, the
# rows `hazard_pooling` stacks for nodes 2 to 4.  A per-visit design gives a row at t0 no
# column, so it is the per-node fit on any law; a linear visit term needs the law's condition.

output <- fixture_arguments("usage: survtmle_pooled.R OUTPUT.json")
provenance <- fixture_provenance("survtmle_pooled.R", "survtmle")

SEED <- 20263310L
N <- 500L
K <- 4L

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)
W1 <- rbinom(N, 1L, 0.5)
W2 <- rbinom(N, 1L, 0.5)
trt <- rbinom(N, 1L, plogis(-0.2 + 0.5 * W1 - 0.4 * W2))
ftime <- rep(K, N)
ftype <- rep(0L, N)
open <- rep(TRUE, N)
for (t in seq_len(K - 1L)) {
  event <- open & rbinom(N, 1L, plogis(-2.4 + 0.5 * trt + 0.6 * W1 - 0.4 * W2 + 0.1 * t)) == 1L
  ftime[event] <- t
  ftype[event] <- 1L
  open <- open & !event
  # A strong time trend in the dropout, so a pooled linear-t fit differs from the per-visit one.
  dropped <- open & rbinom(N, 1L, plogis(-3.2 + 0.3 * trt + 0.4 * W1 + 0.5 * t)) == 1L
  ftime[dropped] <- t
  open <- open & !dropped
}
adjust <- data.frame(W1 = W1, W2 = W2)
# makeDataList numbers the long rows by the rank of the observed time, so the per-visit
# formula's visit k is visit k only when every visit 1..K is an observed time.
stopifnot(identical(sort(unique(as.integer(ftime))), seq_len(K)))

per_visit <- paste(
  "-1 +",
  paste(vapply(seq_len(K - 1L), function(k) {
    # A numeric indicator: a logical I(t == k) enters model.matrix as a two-level factor, and
    # without an intercept its FALSE level adds aliased columns.
    sprintf("I(as.numeric(t == %d)) + I(as.numeric(t == %d)):trt + I(as.numeric(t == %d)):W1 + I(as.numeric(t == %d)):W2", k, k, k, k)
  }, character(1)), collapse = " + ")
)
FORMULAS <- list(node = per_visit, pooled = "trt + W1 + W2 + t")

hazard_at <- function(model, visit) {
  as.numeric(stats::predict(
    model,
    newdata = data.frame(trt = trt, W1 = W1, W2 = W2, t = visit),
    type = "response"
  ))
}

# The law has no event at t0 by design (see above), so survtmle notes that t0 lies past the last
# event, and its node-t0 outcome regression, whose outcome is zero on every row, is rank
# deficient (speedglm and its predict say so in two texts).  The glm notice of fitted
# probabilities at 0 or 1 is muffled too.  Any other warning stops the script: no stage
# records it here.
EXPECTED <- c(
  "t0 larger than last observed endpoint",
  "fitted probabilities numerically 0 or 1",
  "prediction from a rank-deficient fit may be misleading",
  "the matrix is either rank-deficient or not positive definite"
)

expected_only <- function(condition) {
  text <- conditionMessage(condition)
  if (any(vapply(EXPECTED, grepl, logical(1), x = text, fixed = TRUE))) {
    invokeRestart("muffleWarning")
  }
  stop(sprintf("unexpected survtmle warning: %s", text), call. = FALSE)
}

fit_setting <- function(name) {
  fitted <- withCallingHandlers(
    survtmle::survtmle(
      ftime = ftime, ftype = ftype, trt = trt, adjustVars = adjust, t0 = K,
      method = "mean", glm.trt = "W1 + W2", glm.ftime = "trt*(W1 + W2)",
      glm.ctime = FORMULAS[[name]], returnModels = TRUE, verbose = FALSE
    ),
    warning = expected_only
  )
  model <- fitted$ctimeMod
  list(
    hazard_pooling = name,
    formula = FORMULAS[[name]],
    coefficients = as.list(stats::coef(model)),
    rank = model$rank,
    # Visit s's censoring hazard at each unit's (trt, W1, W2): the retention of node s + 1 is
    # one minus it.
    censoring_hazard = withCallingHandlers(
      lapply(seq_len(K - 1L), function(visit) hazard_at(model, visit)),
      warning = expected_only
    ),
    estimate = as.numeric(fitted$est[, 1]),
    estimate_rows = rownames(fitted$est)
  )
}

payload <- list(
  fixture = "survtmle_pooled",
  schema = 1L,
  provenance = provenance,
  seed = SEED,
  n = N,
  visits = K,
  data = list(W1 = as.numeric(W1), W2 = as.numeric(W2), A = as.numeric(trt),
              time = as.numeric(ftime), event = as.numeric(ftype)),
  settings = list(fit_setting("node"), fit_setting("pooled"))
)
fixture_write(payload, output)
