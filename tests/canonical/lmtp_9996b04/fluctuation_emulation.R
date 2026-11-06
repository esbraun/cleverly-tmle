# The X59 lmtp_tmle fit of one deterministic plan, emulated without lmtp: lmtp 9996b04's
# estimate_tmle loop (R/tmle.R) transcribed for one fold, no N or D nodes and unit weights,
# driving the study's own learner (lmtp_horizon_learners.R) and Q check
# (long_horizon_lmtp_adapter.R).  It reproduced the smoke's categorical_k5_static `high`
# failure exactly: 346,904 of 346,904 follower cells outside Q_BOUNDS after a node-5 epsilon of
# 3.4e15.  Evidence for UNPAIRED_PLANS in canonical_long_horizon_categorical_ltmle.py; not a
# declared source, and no run reads it.  SAMPLE.csv is one replicate's comparator input
# (reference_sample(...).frame written as CSV).
#
#   Rscript tests/canonical/lmtp_9996b04/fluctuation_emulation.R SAMPLE.csv PLAN
args <- commandArgs(TRUE); plan <- args[[2]]
study_env <- function(name, default = "") { v <- Sys.getenv(name); if (nzchar(v)) v else default }
study_stage <- function(stage, expr) expr
fix <- file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))), "..")
learn <- readLines(file.path(fix, "lmtp_horizon_learners.R"))
learn <- learn[seq_len(grep("^local\\(\\{", learn)[[1]] - 1L)]   # drop the lmtp namespace hook
eval(parse(text = learn), globalenv())
adapter <- readLines(file.path(fix, "long_horizon_lmtp_adapter.R"))
eval(parse(text = adapter), globalenv())
frame <- as.data.frame(data.table::fread(args[[1]]))
frame <- horizon_typed(frame)
K <- as.integer(unique(frame$design__K)); m <- 3L; nodes <- 1:K
trt <- paste0("A", nodes); cens <- paste0("C", nodes)
ratios <- as.matrix(frame[horizon_columns(frame, "ratio", plan, nodes)])
horizon_learner_state(ratios, integer(0), m, K, 0L, folds = as.integer(frame$fold))
frame$..unit <- seq_len(nrow(frame))
followers <- horizon_followers(!is.na(as.matrix(frame[trt])), ratios)
natural <- frame[c("W", paste0("L", 2:K), trt, cens, "Y", "..unit")]
for (c in cens) natural[[c]] <- ifelse(is.na(natural[[c]]), 0, natural[[c]])   # fix_censoring_ind
natural$..i..lmtp_id <- natural$..unit
shifted <- natural
for (t in nodes) shifted[[trt[[t]]]] <- frame[[sprintf("shift__%s__%d__1", plan, t)]]
bound <- function(x, p = 1e-05) pmax(pmin(x, 1 - p), p)
observed <- function(d, t) if (t == 0) rep(TRUE, nrow(d)) else d[[cens[[t]]]] == 1
weights <- t(apply(ratios, 1, cumprod))
n <- nrow(natural)
pn <- matrix(nrow = n, ncol = K + 1); ps <- pn; init <- pn
ps[, K + 1] <- natural$Y
for (time in rev(nodes)) {
  i <- observed(natural, time)
  history <- c("W", if (time > 1) paste0("L", 2:time), trt[seq_len(time)], cens[seq_len(time - 1)])
  vars <- c("..i..lmtp_id", history, "Y")
  fit <- horizon_fit(natural[i, vars], "Y", "..i..lmtp_id", "quasibinomial", stratified = TRUE)
  ip <- observed(natural, time - 1)
  us <- natural[ip, c("..i..lmtp_id", history)]; us[, trt[[time]]] <- shifted[ip, trt[[time]]]
  pn[ip, time] <- predict(fit, natural[ip, ], 1e-05)
  ps[ip, time] <- predict(fit, us, 1e-05)
  init[ip, time] <- pn[ip, time]
  iv <- i
  fl <- suppressWarnings(glm(ps[iv, time + 1] ~ offset(qlogis(pn[iv, time])), weights = weights[iv, time], family = "binomial"))
  cat(sprintf("node %d: fitted rows %d, coef %s, eps %.4g, y class %s\n", time, fit$rows,
              paste(sprintf("%.3g", fit$coefficient), collapse = " "), coef(fl), class(ps[iv, time + 1])))
  natural[ip, "Y"] <- bound(plogis(qlogis(ps[ip, time]) + coef(fl)))
  pn[ip, time] <- bound(plogis(qlogis(pn[ip, time]) + coef(fl)))
  ps[ip, time] <- bound(plogis(qlogis(ps[ip, time]) + coef(fl)))
}
count <- function(p) { v <- p[cbind(followers, followers)]; v <- v[!is.na(v)]; c(cells = length(v), outside = sum(v < horizon_q_bounds[[1]] | v > horizon_q_bounds[[2]]), lo = min(v), hi = max(v)) }
print(rbind(targeted = count(cbind(pn[, nodes], ps[, nodes])), initial = count(cbind(init[, nodes], init[, nodes]))))
