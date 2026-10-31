suppressPackageStartupMessages(library(lmtp))
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

# X33 part (d), the comparator side.  `ratio_trim = q` replaces every per-node density ratio by
# min(r, Q_q), with Q_q R's type-7 quantile over all n x T entries, zeros and ties included,
# as lmtp's `.trim` does.  The script prints two things: lmtp's trim on a fixed matrix with
# zeros and ties, and the ratio matrices of one lmtp fit at `.trim = 1` and `.trim = q`, so the
# trim is read where lmtp applies it, after the fold recombination.
#
# Sources read at lmtp 1.5.4, commit f04a2b47f46debc515ce4ae778e05ebfde922c44:
#   utils.R: trim <- function(x, trim) pmin(x, quantile(x, trim, na.rm = TRUE))
#   density_ratios.R, cf_density_ratios: `ans$density_ratios <- trim(ans$density_ratios,
#     control$.trim)` on the recombined n x tau matrix of per-node ratios.
#   estimators.R, lmtp_tmle: returns that matrix as `density_ratios`.

output <- fixture_arguments("usage: lmtp_trim.R OUTPUT.json")
provenance <- fixture_provenance("lmtp_trim.R", "lmtp")

SEED <- 20263304L
QUANTILES <- c(0.5, 0.8, 0.9, 0.975)
TRIM <- 0.9
N <- 400L
TAU <- 3L
trim <- getFromNamespace("trim", "lmtp")

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)

# A fixed matrix: a fifth of the entries zero, the rest drawn from four values, so every
# quantile above lands on a tie or between two tied runs.
matrix_values <- matrix(
  sample(c(0, 0.5, 1.25, 2, 4), 60L * TAU, replace = TRUE, prob = c(0.2, 0.2, 0.3, 0.2, 0.1)),
  nrow = 60L
)
fixed <- lapply(QUANTILES, function(q) {
  list(
    quantile = q,
    threshold = as.numeric(stats::quantile(matrix_values, q, na.rm = TRUE)),
    trimmed = as.numeric(trim(matrix_values, q))
  )
})

# A three-node law with right censoring: W, then L_t, A_t and the retention C_t at each node.
W <- rbinom(N, 1L, 0.5)
frame <- data.frame(W = W)
alive <- rep(TRUE, N)
previous <- rep(0, N)
for (t in seq_len(TAU)) {
  L <- rnorm(N, 0.3 * W + 0.4 * previous)
  A <- rbinom(N, 1L, plogis(-0.2 + 0.6 * W + 0.5 * L - 0.4 * previous))
  stay <- rbinom(N, 1L, plogis(2.2 - 0.3 * W - 0.2 * L))
  frame[[sprintf("L%d", t)]] <- ifelse(alive, L, NA)
  frame[[sprintf("A%d", t)]] <- ifelse(alive, A, NA)
  frame[[sprintf("C%d", t)]] <- ifelse(alive, stay, 0L)
  alive <- alive & stay == 1L
  previous <- A
}
frame$Y <- ifelse(alive, rbinom(N, 1L, plogis(-0.5 + 0.4 * W + 0.3 * frame$L3 + 0.5 * frame$A3)), NA)

fit <- function(trim_level) {
  set.seed(SEED + 1L)
  lmtp::lmtp_tmle(
    data = frame,
    trt = sprintf("A%d", seq_len(TAU)),
    outcome = "Y",
    baseline = "W",
    time_vary = lapply(seq_len(TAU), function(t) sprintf("L%d", t)),
    cens = sprintf("C%d", seq_len(TAU)),
    shift = lmtp::static_binary_on,
    outcome_type = "binomial",
    learners_outcome = "SL.glm",
    learners_trt = "SL.glm",
    folds = 1,
    control = lmtp::lmtp_control(.trim = trim_level, .learners_outcome_folds = 2,
                                 .learners_trt_folds = 2)
  )
}
untrimmed <- fit(1)
trimmed <- fit(TRIM)

columns <- function(m) lapply(seq_len(ncol(m)), function(j) as.numeric(m[, j]))

payload <- list(
  fixture = "lmtp_trim",
  schema = 1L,
  provenance = provenance,
  seed = SEED,
  fixed = list(n = nrow(matrix_values), tau = TAU, columns = columns(matrix_values), cases = fixed),
  fit = list(
    n = N,
    tau = TAU,
    trim = TRIM,
    data = lapply(as.list(frame), as.numeric),
    untrimmed = columns(untrimmed$density_ratios),
    trimmed = columns(trimmed$density_ratios),
    estimate_untrimmed = untrimmed$estimate@x,
    estimate_trimmed = trimmed$estimate@x
  )
)
fixture_write(payload, output)
