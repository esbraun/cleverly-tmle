suppressPackageStartupMessages(library(tmle))
source("/fixture/study_harness.R")
source("/fixture/tmle_point_adapter.R")
options(digits = 17)

args <- study_arguments("usage: probe_native_result2.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
samples <- read.csv(gzfile(args$samples), stringsAsFactors = FALSE)
truths <- read.csv(args$truths, stringsAsFactors = FALSE)
frame <- subset(samples, replicate == 0L)

q_z0 <- cbind(frame$qn_z0_a0, frame$qn_z0_a1)
q_z1 <- cbind(frame$qn_z1_a0, frame$qn_z1_a1)
p_z1 <- cbind(frame$pzn_a0, frame$pzn_a1)
p_delta1 <- cbind(frame$pin_a0, frame$pin_a1, frame$pin_a0, frame$pin_a1)
fit <- tmle::tmle(
  Y = frame$Y,
  A = frame$A,
  W = frame["W"],
  Z = frame$Z,
  Delta = frame$Delta,
  Q = q_z0,
  Q.Z1 = q_z1,
  g1W = frame$gn1,
  pZ1 = p_z1,
  pDelta1 = p_delta1,
  family = "binomial",
  fluctuation = "logistic",
  Qbounds = c(0.001, 0.999),
  gbound = c(0.01, 0.99),
  cvQinit = FALSE,
  evalATT = FALSE,
  verbose = FALSE
)

# Deliberately preserve the upstream defect: result two receives exact Q.Z1
# counterfactuals, but tmle 2.1.1 constructed its observed QAW offset from Q.
rows <- tmle_point_rows(
  fit[[2L]],
  q_z1,
  rep(1, nrow(frame)),
  truths,
  "binary_cde_z1_mar",
  0L,
  implementation = "tmle-r-cde-native-result2"
)
write.csv(rows, args$output, row.names = FALSE, quote = TRUE)
