# The node-K fluctuation of lmtp 9996b04's cf_tmle (R/tmle.R, `fluc`), raw versus rescaled
# weights, on the inputs fluctuation_scan.py writes.  Evidence for the unpaired plans of
# long-horizon-categorical-ltmle; not a declared source, and no run reads it.
#
# lmtp fits glm(y ~ offset(qlogis(Q)), weights = cumulative ratios, family = binomial) at R's
# default start, mu = (w y + 0.5) / (w + 1).  The scan fits the intercept-only model at the
# offset zero, once on the raw cumulative ratios and once on them divided by their mean.  The
# maximum likelihood estimate does not depend on the weights' scale, so a raw fit far from the
# rescaled one is glm.fit diverging from its start, not a property of the data.
#
#   Rscript tests/canonical/lmtp_9996b04/fluctuation_scan.R INPUT_DIR OUTPUT.csv

args <- commandArgs(TRUE)
rows <- list()
for (path in sort(list.files(args[[1]], pattern = "\\.csv$", full.names = TRUE))) {
  frame <- as.data.frame(data.table::fread(path))
  K <- unique(frame$design__K)
  plans <- unique(sub("^ratio__(.*)__[0-9]+$", "\\1", grep("^ratio__", names(frame), value = TRUE)))
  for (replicate in unique(frame$replicate)) {
    one <- frame[frame$replicate == replicate, ]
    for (plan in plans) {
      ratios <- as.matrix(one[sprintf("ratio__%s__%d", plan, seq_len(K))])
      weight <- t(apply(ratios, 1, cumprod))[, K]
      kept <- !is.na(one[[sprintf("C%d", K)]]) & one[[sprintf("C%d", K)]] == 1 & weight > 0
      y <- one$Y[kept]
      w <- weight[kept]
      fit <- function(scale) {
        coef(suppressWarnings(glm(y ~ 1, weights = w / scale, family = "binomial")))[[1]]
      }
      raw <- fit(1)
      rescaled <- fit(mean(w))
      rows[[length(rows) + 1L]] <- data.frame(
        scenario = unique(one$scenario), plan = plan, replicate = replicate, rows = length(y),
        max_weight = max(w), raw = raw, rescaled = rescaled,
        diverged = !is.finite(raw) || abs(raw - rescaled) > 1e-6 * max(1, abs(rescaled))
      )
    }
  }
}
write.csv(do.call(rbind, rows), args[[2]], row.names = FALSE)
