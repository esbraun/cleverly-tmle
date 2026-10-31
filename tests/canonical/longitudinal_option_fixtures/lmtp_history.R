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

# X33 part (e), the comparator side.  `markov_order = k` keeps, at node t, the baseline W, the
# covariates L_{t-k}..L_t and the treatments A_{t-k-1}..A_{t-1} in the mechanism design, as
# lmtp's `k` does.  The script prints lmtp's column sets for every node and order.
#
# Source read at lmtp 1.5.4, commit f04a2b47f46debc515ce4ae778e05ebfde922c44 (`R/Vars.R`):
#   LmtpVars$new(W, L, A, C, D, Y, outcome_type, tau, k)$history(var, t)
#   history("A", t) is the treatment and censoring design at node t (density_ratios.R reads
#     `history("A", time)` and adds A_t and C_t).
#   history("L", t + 1) is the outcome-regression design at node t (tmle.R and sdr.R read
#     `history("L", time + 1)`).  lmtp fits it on every at-risk row, so it holds A_t and the
#     earlier treatments of the window; LTMLE fits the followers and leaves out a static node's
#     treatment, which is constant among them.

output <- fixture_arguments("usage: lmtp_history.R OUTPUT.json")
provenance <- fixture_provenance("lmtp_history.R", "lmtp")

TAU <- 4L
ORDERS <- list(0, 1, 2, Inf)
LmtpVars <- getFromNamespace("LmtpVars", "lmtp")

variables <- function(k) {
  LmtpVars$new(
    W = "W",
    L = lapply(seq_len(TAU), function(t) sprintf("L%d", t)),
    A = sprintf("A%d", seq_len(TAU)),
    C = sprintf("C%d", seq_len(TAU)),
    D = NULL,
    Y = "Y",
    outcome_type = "binomial",
    tau = TAU,
    k = k
  )
}

order_label <- function(k) if (is.infinite(k)) "Inf" else sprintf("%d", k)

orders <- lapply(ORDERS, function(k) {
  vars <- variables(k)
  nodes <- lapply(seq_len(TAU), function(t) {
    list(
      node = t,
      mechanism = as.list(vars$history("A", t)),
      outcome = as.list(vars$history("L", t + 1L))
    )
  })
  list(order = order_label(k), nodes = nodes)
})

payload <- list(
  fixture = "lmtp_history",
  schema = 1L,
  provenance = provenance,
  tau = TAU,
  orders = orders
)
fixture_write(payload, output)
