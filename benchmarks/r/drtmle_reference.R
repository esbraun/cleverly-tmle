#!/usr/bin/env Rscript
#
# Piece F3: the bounded differential run against the published `drtmle` R package.
#
# This is a **diagnostic**, and what it is for is stated in `docs/roadmap.md`'s piece F and
# in `CLAUDE.md`: a divergence it finds is a *question*, adjudicated against Benkeser et al.,
# `docs/drtmle/theorem-concordance.md`, the exact-law identities and the remainder
# decomposition -- never settled by which side R is on.  Changing this package to match R is
# stop-ship 17.  Nothing here is a release criterion and nothing here runs in any test tier.
#
# It records the same step vocabulary `benchmarks/drtmle_trace.py` records, on the same frozen
# fixture, from the same initial `Qn` and `gn`, so that `benchmarks/drtmle_r_compare.py` can
# align the two trajectories and name the **earliest** divergence.
#
# Two design decisions carry the whole thing and each is the one F2 already had to make.
#
# **The package's own loop runs; nothing is re-implemented.**  The internals are replaced in
# the `drtmle` namespace by wrappers that call the originals and record either side of them,
# and restored in `on.exit` -- the R idiom for exactly what `TracingDRTMLE` does in Python.  A
# replay of the loop written here would be a second implementation, and a first-divergence
# hunt whose instrument is a re-implementation finds the instrument.  That the wrapping does
# not move the fit is *checked* rather than argued: `--verify` refits with the wrappers off and
# compares `psi` and `se` to `VERIFY_TOLERANCE`.
#
# **Arrays leave here as raw little-endian float64, not as text.**  F2's own record says why:
# written at 17 significant digits and read back with a fast parser, the fixture's `w1` came
# back short by one unit in the last place on 65 of 200 rows, at 2.2e-16 -- precisely the size
# of difference this run would find between two implementations and mis-classify as a learner
# difference.  A binary blob has no parser to be inexact.  The index beside it is scalars only,
# where CSV is safe.

suppressWarnings(suppressMessages(library(drtmle)))

# `psi` and `se` from a wrapped fit against an unwrapped one.  Not a tolerance on the
# comparison -- a tolerance on the *instrument*, which should be exact and is checked at a bar
# that would catch a wrapper that dropped a step rather than one that reordered a sum.
VERIFY_TOLERANCE <- 1e-12

args <- commandArgs(trailingOnly = TRUE)
arg_of <- function(flag, default = NULL) {
  hit <- match(flag, args)
  if (is.na(hit)) {
    if (is.null(default)) stop(sprintf("%s is required", flag))
    return(default)
  }
  args[[hit + 1L]]
}

fixture_path <- arg_of("--fixture", "benchmarks/fixtures/drtmle_trace_v1.csv")
manifest_path <- arg_of("--manifest", "benchmarks/fixtures/drtmle_trace_v1.json")
out_dir <- arg_of("--out", "benchmarks/results/r-trace")
qsteps <- as.integer(arg_of("--qsteps", "2"))
max_iter <- as.integer(arg_of("--max-iter", "3"))
# `drtmle`'s own default is `1/length(Y)`; `-1` here means "leave it at the package's default"
# so that a record taken without this flag is the package as shipped. Every record now writes
# the realised value into `meta.csv` -- before this, no committed record said what bar produced
# it, which is the one thing a stopping-rule comparison cannot leave implicit.
tol_ic <- as.numeric(arg_of("--tol-ic", "-1"))
# The per-step state is what a first-divergence hunt reads and what a *ladder* does not: at a
# tight `tolIC` R can run a hundred rounds, and the full export is ~3 MB gzipped a rung against
# a few KB for the scalars. The blocks, the steps and the summary are the whole of what a
# stopping-bar reading needs.
blocks_only <- "--blocks-only" %in% args
verify <- !("--no-verify" %in% args)

stopifnot(qsteps %in% c(1L, 2L))

# ------------------------------------------------------------------ the fixture

# The digest is checked before a number is read, and the failure is fatal.  Every trace
# already taken, here and in Python, is against these bytes; a silently regenerated fixture
# would make two runs incomparable while both looked fine.  `digest` is one apt package and
# one CRAN package; a `sha256sum` shell-out would be one more thing to be portable.
sha256_of <- function(path) {
  if (requireNamespace("digest", quietly = TRUE)) {
    return(digest::digest(file = path, algo = "sha256"))
  }
  out <- system2("sha256sum", shQuote(path), stdout = TRUE)
  sub(" .*$", "", out)
}

manifest_text <- paste(readLines(manifest_path, warn = FALSE), collapse = "")
declared <- sub('^.*"sha256"[^"]*"([0-9a-f]{64})".*$', "\\1", manifest_text)
observed <- sha256_of(fixture_path)
if (!identical(declared, observed)) {
  stop(sprintf(
    "fixture digest mismatch: manifest says %s, the file is %s. Every trace already taken is\nagainst the manifest's bytes -- regenerate the Python side too, or restore the file.",
    declared, observed
  ))
}

tolg_of <- function(text) {
  block <- sub('^.*"g_bounds"[^\\[]*\\[([^]]*)\\].*$', "\\1", text)
  as.numeric(strsplit(gsub("[[:space:]]", "", block), ",")[[1]])
}
g_bounds <- tolg_of(manifest_text)
# `drtmle` takes one lower bound where this package takes a pair.  The fixture's truncation is
# slack on every row by construction -- F2 reports `clipped = 0` -- so the upper bound has
# nothing to bind on here and the pair collapses to R's scalar without a convention being
# chosen.  A fixture that turns clipping on is a second fixture and a different comparison.
tolg <- g_bounds[[1]]

fixture <- read.csv(fixture_path, colClasses = "numeric")
n <- nrow(fixture)
Y <- fixture$y
A <- fixture$a
W <- data.frame(w1 = fixture$w1, w2 = fixture$w2)
folds <- fixture$fold

# The arms, in the order Python reports them.  `drtmle`'s `a_0` orders every list it returns,
# so this is the axis the comparison aligns on and it is written once.
a_0 <- c(1, 0)
# `Qn` is a list over `a_0` and `gn` is a list of arm probabilities in the same order.  These
# are the fixture's committed columns: the *identical* initial nuisances the Python trace
# starts from, which is what makes a divergence downstream about the construction.
Qn_init <- list(fixture$qn1, fixture$qn0)
gn_init <- list(fixture$gn, 1 - fixture$gn)

# ------------------------------------------------------------------ the recorder

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

steps <- list()
arrays <- list()
blocks <- list()
offset <- 0L
# Gzipped, because this export is **committed** and read back with no R installed. `gzfile`
# and Python's `gzip` agree on the container; the bytes inside it are still raw little-endian
# float64, so nothing acquires a parser to be inexact.
blob <- gzfile(file.path(out_dir, "arrays.f64.gz"), open = "wb")

emit <- function(step, field, arm, values) {
  values <- as.double(values)
  writeBin(values, blob, size = 8, endian = "little")
  arrays[[length(arrays) + 1L]] <<- data.frame(
    step = step, field = field, arm = arm,
    offset = offset, length = length(values), stringsAsFactors = FALSE
  )
  offset <<- offset + length(values)
}

# The state, exactly the six arrays `benchmarks/drtmle_trace.py`'s `State` carries, named the
# way *this package* names them rather than the way R does.  R's `grn1` is the signed quotient
# and its `grn2` is the probability, which is the other way round from the paper and from this
# package -- the roadmap's R3 discussion records that, and mapping it here rather than in the
# comparison is what keeps one statement of it.
state <- new.env(parent = emptyenv())
state$q <- lapply(Qn_init, identity)
state$g <- lapply(gn_init, identity)
state$qr <- lapply(a_0, function(a) rep(NA_real_, n))
state$gr1 <- lapply(a_0, function(a) rep(NA_real_, n)) # R's grn2: the probability
state$gr2 <- lapply(a_0, function(a) rep(NA_real_, n)) # R's grn1: the signed quotient

phase <- "prime"
round_no <- 0L

record <- function(equation, epsilon = rep(NA_real_, length(a_0)),
                   converged = NA, note = "") {
  index <- length(steps) + 1L
  if (!blocks_only) {
    for (i in seq_along(a_0)) {
      emit(index, "q", a_0[[i]], state$q[[i]])
      emit(index, "g", a_0[[i]], state$g[[i]])
      emit(index, "qr", a_0[[i]], state$qr[[i]])
      emit(index, "gr1", a_0[[i]], state$gr1[[i]])
      emit(index, "gr2", a_0[[i]], state$gr2[[i]])
    }
  }
  steps[[index]] <<- data.frame(
    step = index, phase = phase, round = round_no, equation = equation,
    epsilon_1 = epsilon[[1]], epsilon_0 = epsilon[[2]],
    converged = converged, note = note, stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------------ the wrappers

ns <- asNamespace("drtmle")
originals <- list()
wrap <- function(name, wrapper) {
  originals[[name]] <<- get(name, envir = ns)
  utils::assignInNamespace(name, wrapper, ns = "drtmle")
}
restore <- function() {
  for (name in names(originals)) utils::assignInNamespace(name, originals[[name]], ns = "drtmle")
}

# **The committed fold column, reached through the package's own supported branch.**
#
# Installed on **every** fit, traced or not, and that is not tidiness. It is a configuration
# wrapper rather than a recording one: R's own `make_validRows` draws a *random* split when it
# is handed a count, so a "plain" fit without this would differ from a traced one by its folds
# and not by the tracing. The verify step caught exactly that -- `worst |traced - plain| =
# 0.00342` against a bar of 1e-12 -- which is what an instrument check is for, and it is
# recorded here rather than quietly repaired.
install_folds <- function() {
  wrap("make_validRows", function(cvFolds, n, ...) {
    stopifnot(n == length(folds))
    originals$make_validRows(folds, n = n, ...)
  })
}

# **The three influence-curve blocks, and their means.**
#
# `docs/roadmap.md`'s F3 row asks for a comparison "over arrays, **scores** and coefficients",
# and the scores are these: `eval_Dstar`'s mean is equation (8)'s empirical mean, since `psi_t`
# is the mean of `Qn_a` and so `mean(Q - psi)` is identically zero; `eval_Dstar_g`'s is
# equation (9)'s and `eval_Dstar_Q`'s is equation (10)'s. The *arrays* are the correction
# blocks the reported curve subtracts -- this package's `D*_g` and `D*_Q`.
#
# **One asymmetry is recorded rather than smoothed over.** Inside R's loop `eval_Dstar_Q` is
# handed `gn = gn`, the **initial** mechanism, while `eval_Dstar_g` is handed `gn = gnStar`,
# the targeted one. Whether that is deliberate is a question for the derivation, and it is the
# class of thing roadmap item 20 turned on here: a block evaluated at one mechanism while the
# equation was solved at another leaves the curve uncentred exactly where the two differ. So
# `at_targeted_g` travels on every block row, comparing the argument against the current state
# rather than trusting the call site.
block_call <- 0L
record_block <- function(name, per_arm, gn_argument) {
  index <- length(steps)
  # A monotone counter beside the step index, because a step index is **not** unique here:
  # `eval_Dstar` is called inside the loop and again after it, and both land at whatever step
  # count the loop happened to leave. The exit scores are "the last call per block", and that
  # sentence needs an order the step index cannot supply.
  block_call <<- block_call + 1L
  targeted <- isTRUE(all.equal(
    lapply(gn_argument, as.numeric), lapply(state$g, as.numeric),
    tolerance = 0
  ))
  for (i in seq_along(a_0)) {
    values <- as.numeric(per_arm[[i]])
    if (!blocks_only) emit(index, name, a_0[[i]], values)
    blocks[[length(blocks) + 1L]] <<- data.frame(
      call = block_call, step = index, round = round_no, phase = phase,
      block = name, arm = a_0[[i]],
      mean = mean(values), sd = stats::sd(values), at_targeted_g = targeted,
      stringsAsFactors = FALSE
    )
  }
}

install_blocks <- function() {
  wrap("eval_Dstar", function(...) {
    out <- originals$eval_Dstar(...)
    record_block("D", out, list(...)$gn)
    out
  })
  wrap("eval_Dstar_g", function(...) {
    out <- originals$eval_Dstar_g(...)
    record_block("D_g", out, list(...)$gn)
    out
  })
  wrap("eval_Dstar_Q", function(...) {
    out <- originals$eval_Dstar_Q(...)
    record_block("D_Q", out, list(...)$gn)
    out
  })
}

install_wrappers <- function() {
  wrap("fluctuateG", function(...) {
    out <- originals$fluctuateG(...)
    state$g <- lapply(out, function(x) unlist(x$est))
    # A round *opens* with equation (9) under R's order, exactly as it does under this
    # package's `"cleverly"`.  Numbering here rather than after the fact -- unlike the Python
    # recorder, this stream has one shape and no second update order to cover.
    round_no <<- round_no + 1L
    phase <<- "round"
    record("9", epsilon = sapply(out, function(x) as.numeric(x$eps)[[1]]))
    out
  })
  wrap("fluctuateQ1", function(...) {
    out <- originals$fluctuateQ1(...)
    state$q <- lapply(out, function(x) unlist(x[[1]]))
    record("8", epsilon = sapply(out, function(x) as.numeric(x$eps)[[1]]))
    out
  })
  wrap("fluctuateQ2", function(...) {
    out <- originals$fluctuateQ2(...)
    state$q <- lapply(out, function(x) unlist(x[[1]]))
    record("10", epsilon = sapply(out, function(x) as.numeric(x$eps)[[1]]))
    out
  })
  wrap("fluctuateQ", function(...) {
    out <- originals$fluctuateQ(...)
    state$q <- lapply(out, function(x) unlist(x[[1]]))
    # `Qsteps = 1` solves (8) and (10) as one two-column submodel.  This package's closing
    # pass has a four-column solve it calls `joint`; the two are not the same object, and the
    # comparison keeps them apart rather than aligning them by name.
    record("joint", epsilon = sapply(out, function(x) as.numeric(x$eps)[[1]]))
    out
  })
  wrap("estimateQrn", function(...) {
    out <- originals$estimateQrn(...)
    out
  })
  wrap("estimategrn", function(...) {
    out <- originals$estimategrn(...)
    out
  })
}

# The two refits are recorded where their *predictions are assembled*, not where a learner
# returned: `estimateQrn` and `estimategrn` are called once per fold, so a wrapper on them
# records a third of an array.  `reorder_list` is the function that stitches the folds back
# into a full-length vector, which is the first point at which the state exists.
install_reorder <- function() {
  wrap("reorder_list", function(...) {
    out <- originals$reorder_list(...)
    dots <- list(...)
    if (isTRUE(dots$grn_ind)) {
      state$gr2 <- lapply(out, function(x) unlist(x$grn1)) # R's grn1 -> this package's gr2
      state$gr1 <- lapply(out, function(x) unlist(x$grn2)) # R's grn2 -> this package's gr1
      record("refit", note = "gr")
    } else {
      state$qr <- lapply(out, function(x) unlist(x))
      record("refit", note = "qr")
    }
    out
  })
}

# ------------------------------------------------------------------ the run

fit_once <- function(traced) {
  install_folds()
  if (traced) {
    install_wrappers()
    install_reorder()
    install_blocks()
  }
  on.exit(restore(), add = TRUE)
  drtmle::drtmle(
    Y = Y, A = A, W = W, a_0 = a_0,
    Qn = Qn_init, gn = gn_init,
    glm_Qr = "gn", glm_gr = "Qn",
    guard = c("Q", "g"),
    reduction = "univariate",
    maxIter = max_iter, Qsteps = qsteps, tolg = tolg,
    tolIC = if (tol_ic > 0) tol_ic else 1 / length(Y),
    cvFolds = length(unique(folds)), se_cv = "none",
    returnModels = FALSE, use_future = FALSE
  )
}

# ------------------------------------------------------ the first reduced fit, on its own
#
# F3's own stopping rule is *"stop immediately if the trace inputs or the first reduced fits
# do not agree"*, and that gate cannot be read off either trajectory: R primes its loop with a
# `Qr` refit and this package primes with an equation-(8) solve, so the first reduction each
# stream records is taken at a different outcome regression.  The comparable quantity is the
# reduction at the **initial** pair, which is neither side's first recorded step, so it is
# computed here on purpose and exported on its own.  Python's counterpart is the reduced set
# its alternation is *handed* -- `trace.steps[0].before`.
reference_reduction <- function() {
  validRows <- lapply(sort(unique(folds)), function(f) which(folds == f))
  qr <- vector("list", length(a_0))
  gr1 <- vector("list", length(a_0))
  gr2 <- vector("list", length(a_0))
  for (i in seq_along(a_0)) {
    qr[[i]] <- rep(NA_real_, n)
    gr1[[i]] <- rep(NA_real_, n)
    gr2[[i]] <- rep(NA_real_, n)
  }
  for (rows in validRows) {
    q_out <- originals$estimateQrn(
      Y = Y, A = A, W = W, DeltaA = rep(1, n), DeltaY = rep(1, n),
      Qn = Qn_init, gn = gn_init, glm_Qr = "gn", SL_Qr = NULL,
      family = stats::gaussian(), a_0 = a_0, returnModels = FALSE, validRows = rows
    )
    g_out <- originals$estimategrn(
      Y = Y, A = A, W = W, DeltaA = rep(1, n), DeltaY = rep(1, n),
      Qn = Qn_init, gn = gn_init, SL_gr = NULL, tolg = tolg, glm_gr = "Qn",
      a_0 = a_0, reduction = "univariate", returnModels = FALSE, validRows = rows
    )
    for (i in seq_along(a_0)) {
      qr[[i]][rows] <- as.numeric(q_out$est[[i]])
      # R's `grn1` is the signed quotient and its `grn2` is the probability -- the other way
      # round from the paper and from this package. Mapped once, here.
      gr2[[i]][rows] <- as.numeric(g_out$est[[i]]$grn1)
      gr1[[i]][rows] <- as.numeric(g_out$est[[i]]$grn2)
    }
  }
  list(qr = qr, gr1 = gr1, gr2 = gr2)
}

originals$estimateQrn <- get("estimateQrn", envir = ns)
originals$estimategrn <- get("estimategrn", envir = ns)
reference <- reference_reduction()
phase <- "reference"
state$qr <- reference$qr
state$gr1 <- reference$gr1
state$gr2 <- reference$gr2
record("reference", note = "the reduction at the initial (Qn, gn), neither side's first step")
state$qr <- lapply(a_0, function(a) rep(NA_real_, n))
state$gr1 <- lapply(a_0, function(a) rep(NA_real_, n))
state$gr2 <- lapply(a_0, function(a) rep(NA_real_, n))
phase <- "prime"

traced_fit <- fit_once(TRUE)
steps_frame <- do.call(rbind, steps)
arrays_frame <- do.call(rbind, arrays)
blocks_frame <- do.call(rbind, blocks)
close(blob)

verified <- NA_real_
if (verify) {
  plain <- fit_once(FALSE)
  verified <- max(
    abs(traced_fit$drtmle$est - plain$drtmle$est),
    abs(traced_fit$drtmle$cov - plain$drtmle$cov)
  )
  if (!is.finite(verified) || verified > VERIFY_TOLERANCE) {
    stop(sprintf(
      "the wrappers moved the fit: worst |traced - plain| = %.3g against %.3g. The instrument\nis not allowed to change what it measures; nothing downstream of this is readable.",
      verified, VERIFY_TOLERANCE
    ))
  }
}

# `est` is one row per `a_0`; the contrast is what this package reports as `ate` and it is
# formed here rather than in the comparison so that one side owns the sign.
est <- as.numeric(traced_fit$drtmle$est)
cov <- traced_fit$drtmle$cov
summary_frame <- data.frame(
  estimand = c("ey1", "ey0", "ate"),
  psi = c(est[[1]], est[[2]], est[[1]] - est[[2]]),
  se = c(
    sqrt(cov[1, 1]), sqrt(cov[2, 2]),
    sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
  ),
  stringsAsFactors = FALSE
)

meta_frame <- data.frame(
  key = c(
    "qsteps", "max_iter", "tol_ic", "tolg", "n", "n_folds", "arms",
    "rounds", "capped", "blocks_only", "verify_residual", "package_version"
  ),
  value = c(
    qsteps, max_iter, if (tol_ic > 0) tol_ic else 1 / n, tolg, n, length(unique(folds)),
    paste(a_0, collapse = "|"), max(steps_frame$round),
    as.integer(max(steps_frame$round) >= max_iter), as.integer(blocks_only),
    verified, as.character(utils::packageVersion("drtmle"))
  ),
  stringsAsFactors = FALSE
)

write.csv(steps_frame, file.path(out_dir, "steps.csv"), row.names = FALSE)
write.csv(blocks_frame, file.path(out_dir, "blocks.csv"), row.names = FALSE)
write.csv(arrays_frame, file.path(out_dir, "arrays.csv"), row.names = FALSE)
write.csv(summary_frame, file.path(out_dir, "summary.csv"), row.names = FALSE)
write.csv(meta_frame, file.path(out_dir, "meta.csv"), row.names = FALSE)

# The inputs, re-emitted from what R actually read, so the comparison can gate on the two
# sides agreeing *before* it reads a single trajectory.  F3's own stopping rule: if the trace
# inputs or the first reduced fits do not agree, repair the diagnostic rather than interpret
# anything downstream of it.
inputs <- gzfile(file.path(out_dir, "inputs.f64.gz"), open = "wb")
for (column in c("w1", "w2", "a", "y", "fold", "weight", "qn1", "qn0", "gn")) {
  writeBin(as.double(fixture[[column]]), inputs, size = 8, endian = "little")
}
close(inputs)

# **The record describes itself.**  This export is committed and read back with no R installed,
# so every file it contains carries a SHA-256 the Python side checks before it reads a number --
# the same discipline the input fixture has, for the same reason: a silently regenerated record
# would make two comparisons incomparable while both looked fine.
manifest_lines <- c("file,sha256,bytes")
for (name in sort(list.files(out_dir))) {
  if (name == "manifest.csv") next
  path <- file.path(out_dir, name)
  manifest_lines <- c(manifest_lines, sprintf(
    "%s,%s,%d", name, sha256_of(path), file.info(path)$size
  ))
}
writeLines(manifest_lines, file.path(out_dir, "manifest.csv"))

cat(sprintf(
  "drtmle %s  Qsteps=%d  tolIC=%.3g  rounds=%d/%d  steps=%d  psi[ate]=%.10f  se[ate]=%.10f  verify=%.3g\n",
  utils::packageVersion("drtmle"), qsteps, if (tol_ic > 0) tol_ic else 1 / n,
  max(steps_frame$round), max_iter, nrow(steps_frame),
  summary_frame$psi[[3]], summary_frame$se[[3]], verified
))
