# A pinned adapter for npcausal 0.1.0 at 56a5ac117a29258b67b94874be662a171b5131f7.
#
# `npcausal::ipsi` draws its sample splits inside its body (R/ipsi.R line 114) and takes no
# fold argument.  A paired study needs the exact realized Python assignment, so
# `ipsi_with_folds` builds a copy of the pinned function whose first change is that statement:
# `s <- sample(...)` becomes `s <- folds`.  Precedent: `lmtp_crossfit_adapter.R` replaces only
# `task$folds`.
#
# The second change is memory only.  The pinned multiplier bootstrap (R/ipsi.R lines 228-231)
# holds all 10,000 sign vectors as one n x 10,000 matrix: 7.7 GB at n = 96,000, about 19 GB
# at its peak, which a 16 GB Docker VM kills.  The copy draws each column when it reads it.
# `rbinom(n * nbs)` and `nbs` successive `rbinom(n)` calls consume one RNG stream in the same
# order, and each column enters the same expression, so `maxvals` and `calpha` are identical.
# `bootstrap_stream_check()` asserts that identity before a study starts.  Every other
# statement of the body is the pinned one, which the adapter checks.  Estimation stays in the
# pinned code.
#
# The adapter also holds the saturated learner both sides fit (`SL.cellmeans`), the long design
# `ipsi` reads, and the inverse of the recycling defect of `return_ifvals = TRUE`.  Every check
# here stays outside a `study_stage`, so a failed check is fatal rather than one group's
# failure (RM39a2).

# The statements the copy replaces, as the pinned source writes them, and their replacements.
# Compared as language objects, so the check does not depend on how a given R version deparses
# them.
IPSI_FOLD_DRAW <- quote(s <- sample(rep(seq_len(nsplits), ceiling(n / nsplits))[seq_len(n)]))
IPSI_BOOTSTRAP_MATRIX <- quote(
  mult <- matrix(2 * rbinom(n * nbs, 1, 0.5) - 1, nrow = n, ncol = nbs)
)
IPSI_BOOTSTRAP_MAX <- quote(
  maxvals <- sapply(seq_len(nbs), function(col) {
    max(abs(apply(mult[, col] * ifvals2, 2, sum) / sqrt(n)))
  })
)
IPSI_REPLACEMENTS <- list(
  list(pinned = IPSI_FOLD_DRAW, copy = quote(s <- folds)),
  list(pinned = IPSI_BOOTSTRAP_MATRIX, copy = quote(mult <- NULL)),
  list(
    pinned = IPSI_BOOTSTRAP_MAX,
    copy = quote(
      maxvals <- sapply(seq_len(nbs), function(col) {
        column <- 2 * rbinom(n, 1, 0.5) - 1
        max(abs(apply(column * ifvals2, 2, sum) / sqrt(n)))
      })
    )
  )
)

# The identity the streamed bootstrap rests on, on a small seeded problem: the pinned matrix
# form and the streamed form give identical maxima.
bootstrap_stream_check <- function(n = 997L, nbs = 50L, k = 3L, seed = 20261241L) {
  set.seed(seed)
  ifvals2 <- matrix(stats::rnorm(n * k), nrow = n)
  set.seed(seed + 1L)
  mult <- matrix(2 * rbinom(n * nbs, 1, 0.5) - 1, nrow = n, ncol = nbs)
  pinned <- sapply(seq_len(nbs), function(col) {
    max(abs(apply(mult[, col] * ifvals2, 2, sum) / sqrt(n)))
  })
  set.seed(seed + 1L)
  streamed <- sapply(seq_len(nbs), function(col) {
    column <- 2 * rbinom(n, 1, 0.5) - 1
    max(abs(apply(column * ifvals2, 2, sum) / sqrt(n)))
  })
  if (!identical(pinned, streamed)) {
    stop("the streamed multiplier bootstrap does not reproduce the pinned one on this R")
  }
  invisible(TRUE)
}

ipsi_with_folds <- function(original = npcausal::ipsi, namespace = asNamespace("npcausal")) {
  bootstrap_stream_check()
  pinned <- body(original)
  statements <- as.list(pinned)
  replaced <- statements
  positions <- integer(length(IPSI_REPLACEMENTS))
  for (index in seq_along(IPSI_REPLACEMENTS)) {
    target <- IPSI_REPLACEMENTS[[index]]$pinned
    hits <- which(vapply(statements, function(statement) identical(statement, target), logical(1)))
    if (length(hits) != 1L) {
      stop(sprintf(
        "npcausal::ipsi holds the pinned statement `%s` %d times, not once",
        deparse(target)[1L], length(hits)
      ))
    }
    positions[index] <- hits
    replaced[[hits]] <- IPSI_REPLACEMENTS[[index]]$copy
  }
  copy <- original
  formals(copy) <- c(formals(original), alist(folds = ))
  body(copy) <- as.call(replaced)
  environment(copy) <- namespace
  # The whole body, with the pinned statements restored, is the pinned body.
  restored <- as.list(body(copy))
  for (index in seq_along(IPSI_REPLACEMENTS)) {
    restored[[positions[index]]] <- IPSI_REPLACEMENTS[[index]]$pinned
  }
  if (!identical(deparse(as.call(restored)), deparse(pinned))) {
    stop("the fold-injected copy of npcausal::ipsi differs from the pinned body elsewhere")
  }
  copy
}

# ---------------------------------------------------------------------------- the learner

cell_keys <- function(X) {
  do.call(paste, c(lapply(names(X), function(name) as.character(X[[name]])), sep = "|"))
}

# Saturated cell means: the mean of `Y` within each distinct design row.  An unseen row takes
# the training mean, which is what `tests.discrete_law_longitudinal.CellMeans` returns.  The
# treatment design carries a `block` column (the node), and `ipsi` fits one pooled treatment
# model over every node.  There an unseen row takes the training mean of its own block, which is
# what a per-node `CellMeans` fit returns (`X19-plan.md` 6.1).  `ipsi` hands the same library to
# the treatment and the outcome fits, so one wrapper serves both, keyed on that column.
SL.cellmeans <- function(Y, X, newX, family, obsWeights, ...) {
  if (any(obsWeights != 1)) stop("SL.cellmeans is fitted unweighted in this study")
  keys <- cell_keys(X)
  means <- tapply(Y, keys, mean)
  core <- list(keys = names(means), means = as.numeric(means), default = mean(Y))
  if ("block" %in% names(X)) {
    blocks <- tapply(Y, X$block, mean)
    core$blocks <- setNames(as.numeric(blocks), names(blocks))
  }
  fit <- structure(list(object = core), class = "SL.cellmeans")
  list(pred = predict.SL.cellmeans(fit, newX), fit = fit)
}

predict.SL.cellmeans <- function(object, newdata, ...) {
  object <- object$object
  found <- match(cell_keys(newdata), object$keys)
  out <- object$means[found]
  unseen <- is.na(found)
  if (any(unseen)) {
    if (!is.null(object$blocks)) {
      fallback <- object$blocks[as.character(newdata$block[unseen])]
      if (anyNA(fallback)) stop("a treatment block has no training row")
      out[unseen] <- fallback
    } else {
      out[unseen] <- object$default
    }
  }
  out
}

# ------------------------------------------------------------------------ the long design

# The history at node t of a wide frame with baseline W, covariates L2..LT and arms A1..AT:
# W, A1, L2, ..., A_{t-1}, L_t, padded with zeros to the length at the last node.  The cell
# partition is the one the Python fit's node-t design induces.
ipsi_history <- function(frame, t, n_times) {
  columns <- "W"
  if (t > 1) {
    for (s in 2:t) columns <- c(columns, paste0("A", s - 1), paste0("L", s))
  }
  width <- 2L * n_times - 1L
  values <- as.matrix(frame[, columns, drop = FALSE])
  padded <- cbind(values, matrix(0, nrow(frame), width - length(columns)))
  colnames(padded) <- paste0("h", seq_len(width))
  padded
}

ipsi_long <- function(frame, n_times) {
  n <- nrow(frame)
  id <- rep(seq_len(n), each = n_times)
  time <- rep(seq_len(n_times), times = n)
  pieces <- lapply(seq_len(n_times), function(t) ipsi_history(frame, t, n_times))
  history <- matrix(0, n * n_times, ncol(pieces[[1]]), dimnames = list(NULL, colnames(pieces[[1]])))
  arms <- numeric(n * n_times)
  for (t in seq_len(n_times)) {
    rows <- which(time == t)
    history[rows, ] <- pieces[[t]]
    arms[rows] <- frame[[paste0("A", t)]]
  }
  list(
    y = frame$Y,
    a = arms,
    x.trt = data.frame(block = time, history),
    x.out = data.frame(history),
    time = time,
    id = id
  )
}

# ------------------------------------------------------------------------- the un-recycle

# `return_ifvals = TRUE` returns `ifvals - est.eff` (R/ipsi.R line 255), and R recycles the
# length-k vector down the n x k matrix column-major.  Entry (i, j) had
# `est.eff[((j - 1) * n + i - 1) %% k + 1]` subtracted.  This adds that back and subtracts
# `est.eff[j]`, which is the centred influence value the published standard error reads.
ipsi_centred_ifvals <- function(fit) {
  returned <- as.matrix(fit$ifvals)
  n <- nrow(returned)
  k <- ncol(returned)
  estimates <- fit$res.ptwise$est
  index <- matrix(((rep(seq_len(k), each = n) - 1) * n + rep(seq_len(n), k) - 1) %% k + 1, n, k)
  raw <- returned + matrix(estimates[index], n, k)
  for (j in seq_len(k)) {
    published <- fit$res.ptwise$se[[j]]
    if (abs(sd(raw[, j]) - published) > 1e-12 * max(1, published)) {
      stop("the un-recycled influence values disagree with npcausal's published standard error")
    }
  }
  sweep(raw, 2, estimates)
}

# One full-grid call per fold assignment.  `print(c(i, j))` of the `sl` path is captured.
ipsi_arguments <- function(frame, deltas, folds, n_times) {
  # The arguments of one `ipsi` call over the whole plan grid.  A runner passes them to
  # `do.call(ipsi_copy, .)` inside its stage, so the stage holds the comparator call only.
  long <- ipsi_long(frame, n_times)
  list(
    y = long$y, a = long$a, x.trt = long$x.trt, x.out = long$x.out,
    time = long$time, id = long$id, delta.seq = deltas, nsplits = length(unique(folds)),
    progress_bar = FALSE, return_ifvals = TRUE, fit = "sl", sl.lib = "SL.cellmeans",
    folds = folds
  )
}

ipsi_grid_result <- function(fit) {
  list(
    estimate = fit$res.ptwise$est,
    sigma = fit$res.ptwise$se,
    calpha = as.numeric(fit$calpha),
    ic = ipsi_centred_ifvals(fit)
  )
}

ipsi_fit_grid <- function(frame, deltas, folds, n_times, ipsi_copy) {
  arguments <- ipsi_arguments(frame, deltas, folds, n_times)
  fit <- NULL
  invisible(capture.output(fit <- do.call(ipsi_copy, arguments)))
  ipsi_grid_result(fit)
}
