library(ltmle)
options(digits = 17)

fixture_dir <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(fixture_dir)) fixture_dir <- "."

make_common <- function() {
  W <- rep(c(0, 1), each = 12)
  A1 <- rep(c(rep(1, 10), 0, 0), 2)
  A2 <- rep(c(rep(1, 8), 0, 0, 1, 0), 2)
  C1 <- C2 <- rep(1, length(W))
  data.frame(
    W = W,
    A1 = A1,
    C1 = C1,
    A2 = A2,
    C2 = C2,
    g_A1 = ifelse(W == 0, 0.5, 0.9),
    g_C1 = rep(0.8, length(W)),
    g_A2 = ifelse(W == 0, 0.5, 0.9),
    g_C2 = rep(0.8, length(W))
  )
}

g_columns <- c("g_A1", "g_C1", "g_A2", "g_C2")

longitudinal <- make_common()
longitudinal$L2 <- as.numeric(xor(longitudinal$W == 1, longitudinal$A1 == 1))
longitudinal$Y <- c(
  0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
  0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1
)
longitudinal <- longitudinal[c(
  "W", "A1", "C1", "L2", "A2", "C2", "Y", g_columns
)]
longitudinal_fit <- ltmle(
  longitudinal[c("W", "A1", "C1", "L2", "A2", "C2", "Y")],
  Anodes = c("A1", "A2"),
  Cnodes = c("C1", "C2"),
  Lnodes = "L2",
  Ynodes = "Y",
  survivalOutcome = FALSE,
  Qform = c(L2 = "Q.kplus1 ~ 1", Y = "Q.kplus1 ~ 1"),
  gform = as.matrix(longitudinal[g_columns]),
  abar = c(1, 1),
  gbounds = c(0.2, 0.99),
  SL.library = "glm",
  stratify = TRUE,
  variance.method = "ic"
)

# A distinct end-of-study witness for the censoring masks.  Rows are one-indexed:
# 4 and 16 follow A1=1 but leave at C1; 8 and 20 follow (A1,A2)=(1,1) but leave
# at C2.  Values after censoring are absent, as they would be in an observed panel.
censored <- longitudinal
censored$C1[c(4, 16)] <- 0
censored[c(4, 16), c("L2", "A2", "C2", "Y")] <- NA
censored$C2[c(8, 20)] <- 0
censored$Y[c(8, 20)] <- NA
censored_fit <- ltmle(
  censored[c("W", "A1", "C1", "L2", "A2", "C2", "Y")],
  Anodes = c("A1", "A2"),
  Cnodes = c("C1", "C2"),
  Lnodes = "L2",
  Ynodes = "Y",
  survivalOutcome = FALSE,
  Qform = c(L2 = "Q.kplus1 ~ 1", Y = "Q.kplus1 ~ 1"),
  gform = as.matrix(censored[g_columns]),
  abar = c(1, 1),
  gbounds = c(0.2, 0.99),
  SL.library = "glm",
  stratify = TRUE,
  variance.method = "ic"
)

survival <- make_common()
survival$Y1 <- c(
  1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
)
survival$Y2_event <- c(
  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0
)
survival$Y2 <- pmax(survival$Y1, survival$Y2_event)
# Both implementations condition later nodes on still being at risk.  Write the absent
# post-event treatment/censoring nodes explicitly instead of relying on ltmle's cleaner.
survival$A2[survival$Y1 == 1] <- NA
survival$C2[survival$Y1 == 1] <- NA
survival <- survival[c(
  "W", "A1", "C1", "Y1", "A2", "C2", "Y2", "Y2_event", g_columns
)]
survival_fit <- ltmle(
  survival[c("W", "A1", "C1", "Y1", "A2", "C2", "Y2")],
  Anodes = c("A1", "A2"),
  Cnodes = c("C1", "C2"),
  Ynodes = c("Y1", "Y2"),
  survivalOutcome = TRUE,
  Qform = c(Y1 = "Q.kplus1 ~ 1", Y2 = "Q.kplus1 ~ 1"),
  gform = as.matrix(survival[g_columns]),
  abar = c(1, 1),
  gbounds = c(0.2, 0.99),
  SL.library = "glm",
  stratify = TRUE,
  variance.method = "ic"
)

write.csv(longitudinal, file.path(fixture_dir, "longitudinal.csv"), row.names = FALSE)
write.csv(censored, file.path(fixture_dir, "censored.csv"), row.names = FALSE)
write.csv(survival, file.path(fixture_dir, "survival.csv"), row.names = FALSE)

write_result <- function(fit, variant) {
  ic <- fit$IC$tmle
  epsilon <- vapply(fit$fit$Qstar, function(node) unname(coef(node)[1]), numeric(1))
  data.frame(
    variant = variant,
    row = seq_along(ic),
    estimate = rep(unname(fit$estimates["tmle"]), length(ic)),
    influence_curve = as.numeric(ic),
    cumulative_g_t1 = fit$cum.g[, 2],
    cumulative_g_t2 = fit$cum.g[, 4],
    epsilon_t1 = rep(epsilon[1], length(ic)),
    epsilon_t2 = rep(epsilon[2], length(ic))
  )
}
expected <- rbind(
  write_result(longitudinal_fit, "longitudinal"),
  write_result(censored_fit, "censored"),
  write_result(survival_fit, "survival")
)
write.csv(expected, file.path(fixture_dir, "reference.csv"), row.names = FALSE)

cat("ltmle ", as.character(packageVersion("ltmle")), "\n", sep = "")
cat("R ", R.version.string, "\n", sep = "")
cat("longitudinal estimate ", unname(longitudinal_fit$estimates["tmle"]), "\n", sep = "")
cat("censored estimate ", unname(censored_fit$estimates["tmle"]), "\n", sep = "")
cat("survival estimate ", unname(survival_fit$estimates["tmle"]), "\n", sep = "")
