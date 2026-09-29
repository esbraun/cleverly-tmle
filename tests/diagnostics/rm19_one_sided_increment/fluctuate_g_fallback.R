# R drtmle 1.1.2's fluctuateG on a constructed case, for the RM19 transcription test.
#
# Run in the pinned image. The input needs the columns A, g0, g1, qr0 and qr1, and the output
# file carries them, so the committed fixture regenerates in place:
#   docker run --rm -v "<this directory>:/work" cleverly-drtmle-reference:538a3a2 \
#     /work/fluctuate_g_fallback.R /work/fluctuate-g-fallback.csv /work/fluctuate-g-fallback.csv
#
# The case separates arm 1, so both glm attempts of fluctuateG fail there and R predicts with
# the failed retry's fitted values (fluctuate.R:118-125). Arm 0 converges on the first attempt.
args <- commandArgs(trailingOnly = TRUE)
case <- read.csv(args[1])
n <- nrow(case)
out <- drtmle:::fluctuateG(
  Y = rep(0, n), A = case$A, W = NULL, DeltaY = rep(1, n), DeltaA = rep(1, n),
  a_0 = c(0, 1), gn = list(case$g0, case$g1), Qrn = list(case$qr0, case$qr1), tolg = 0.01
)
full <- function(x) sprintf("%.17g", x)
result <- data.frame(
  A = full(case$A), g0 = full(case$g0), g1 = full(case$g1),
  qr0 = full(case$qr0), qr1 = full(case$qr1),
  r_g0 = full(out[[1]]$est), r_g1 = full(out[[2]]$est),
  r_eps0 = full(out[[1]]$eps), r_eps1 = full(out[[2]]$eps)
)
write.csv(result, args[2], row.names = FALSE, quote = FALSE)
cat("drtmle", as.character(packageVersion("drtmle")), "\n")
