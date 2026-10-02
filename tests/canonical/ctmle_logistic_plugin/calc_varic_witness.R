suppressPackageStartupMessages(library(ctmle))
options(digits = 17)

# R ctmle's own calc_varIC (R/functions.R, lines 39 to 60, at 18de559), called on the
# inputs a cleverly selector fit used: the targeted outcome regression, the bounded
# propensity of the selected candidate and that candidate's covariates.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: calc_varic_witness.R INPUTS.csv OUTPUT.csv")
frame <- read.csv(args[[1]], stringsAsFactors = FALSE)
selected <- grep("^W", names(frame), value = TRUE)
Q <- cbind(QAW = frame$QAW, Q0W = frame$Q0W, Q1W = frame$Q1W)
h <- frame$A / frame$g1W - (1 - frame$A) / (1 - frame$g1W)
variances <- ctmle:::calc_varIC(
  Y = frame$Y,
  Q = Q,
  h = h,
  A = frame$A,
  W = as.matrix(frame[selected]),
  g1W = frame$g1W,
  ICg = TRUE
)
write.csv(
  data.frame(
    n = nrow(frame),
    var_dstar = variances[[1]],
    var_ic = variances[[2]],
    ctmle_version = as.character(packageVersion("ctmle"))
  ),
  args[[2]],
  row.names = FALSE
)
