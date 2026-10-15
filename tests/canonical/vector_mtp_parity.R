# The parity check of vector-mtp's unsupplied pair, run before the declared run.  It reads the
# draws and the grid that tests/canonical/vector_mtp_parity.py wrote.  For each draw and each
# plan of the pair it fits lmtp's own ratios through the vector adapter, with mtp = TRUE and the
# plan's log-ratio classifier SL.logratio.<plan>, for that plan and for the reference, as the
# runner does.  It also evaluates the R transcription of the ratio on the grid.
#
# usage: vector_mtp_parity.R DIR (reads DIR/draws.csv and DIR/grid.csv; writes DIR/lmtp.csv and
# DIR/grid_lmtp.csv)

suppressPackageStartupMessages(library(lmtp))
future::plan(future::sequential)
source("/fixture/lmtp_vector_adapter.R")
source("/fixture/vector_mtp_logratio.R")
source("/fixture/study_harness.R")
options(digits = 17)

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1) stop("usage: vector_mtp_parity.R DIR")
directory <- arguments[[1]]
draws <- read.csv(file.path(directory, "draws.csv"), stringsAsFactors = FALSE, check.names = FALSE)
grid <- read.csv(file.path(directory, "grid.csv"), stringsAsFactors = FALSE, check.names = FALSE)

grid_out <- grid[, c("plan", "F", "D1", "D2", "W1", "W2")]
grid_out$ratio <- NA_real_
for (plan in unique(grid$plan)) {
  rows <- grid$plan == plan
  grid_out$ratio[rows] <- ordered_ratio(
    plan, grid$F[rows], grid$D1[rows], grid$D2[rows], grid$W1[rows], grid$W2[rows]
  )
}
grid_out$log_ratio <- log(pmax(grid_out$ratio, VECTOR_MTP$floor))
write.csv(grid_out, file.path(directory, "grid_lmtp.csv"), row.names = FALSE)

trt <- list(c("F", "D1", "D2"))
control <- lmtp::lmtp_control(
  .trim = 1, .learners_outcome_folds = 2, .learners_trt_folds = 2, .return_full_fits = TRUE
)
fits <- c(ordered = "SL.logratio.ordered", full = "SL.logratio.full")
out <- list()
for (draw in sort(unique(draws$draw))) {
  frame <- draws[draws$draw == draw, ]
  frame <- frame[order(frame$row), ]
  frame$fold <- 0L
  for (fit in names(fits)) {
    for (plan in c("natural_course", fit)) {
      frames <- lmtp_vector_frames(frame, plan, trt, "Y", c("W1", "W2"), list(NULL))
      task <- vector_task(
        frames$natural, frames$shifted, trt, "Y", c("W1", "W2"), list(NULL), frame$fold
      )
      density <- lmtp_vector_density_ratios(task, fits[[fit]], control)
      out[[length(out) + 1]] <- data.frame(
        draw = draw, row = frame$row, fit = fit, plan = plan,
        lmtp_ratio = as.numeric(density$density_ratios[, 1])
      )
    }
  }
}
write.csv(do.call(rbind, out), file.path(directory, "lmtp.csv"), row.names = FALSE)
