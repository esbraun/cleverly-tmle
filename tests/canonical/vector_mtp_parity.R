# The stacked-classifier parity check of vector-mtp's unsupplied pair, run before the declared
# run.  It reads draws that tests/canonical/vector_mtp_parity.py wrote (natural columns and the
# shift__ columns of each plan of the ordered law), fits lmtp's own density ratios with
# cf_density_ratios and learners_trt = "SL.glm" through the shared cross-fit adapter's task, and
# writes them beside the draw for the Python side to compare with cleverly's node_ratio.
#
# usage: vector_mtp_parity.R DRAWS.csv OUTPUT.csv

suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_vector_adapter.R")
source("/fixture/study_harness.R")
options(digits = 17)

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 2) stop("usage: vector_mtp_parity.R DRAWS.csv OUTPUT.csv")
draws <- read.csv(arguments[[1]], stringsAsFactors = FALSE, check.names = FALSE)
trt <- list(c("F", "D1", "D2"))
variables <- c("F", "D1", "D2", "Y", "W1", "W2")
out <- list()
for (draw in sort(unique(draws$draw))) {
  frame <- draws[draws$draw == draw, ]
  frame <- frame[order(frame$row), ]
  for (plan in c("natural", "ordered", "full")) {
    natural <- frame[, variables]
    shifted <- vector_shifted(frame, plan, trt)[, variables]
    task <- lmtp_internal("LmtpTask")$new(
      data = natural, shifted = shifted, A = trt, Y = "Y", L = list(NULL), W = c("W1", "W2"),
      C = NULL, D = NULL, k = Inf, id = NULL, outcome_type = "binomial", bounds = NULL,
      folds = 1, weights = NULL
    )
    task$folds <- fold_list(rep(0L, nrow(frame)))
    density <- lmtp_internal("cf_density_ratios")(
      task, "SL.glm", FALSE,
      lmtp::lmtp_control(.trim = 1, .learners_trt_folds = 2, .return_full_fits = TRUE),
      function(...) invisible(NULL)
    )
    out[[length(out) + 1]] <- data.frame(
      draw = draw, row = frame$row, plan = plan, lmtp_ratio = as.numeric(density$density_ratios[, 1])
    )
  }
}
write.csv(do.call(rbind, out), arguments[[2]], row.names = FALSE)
