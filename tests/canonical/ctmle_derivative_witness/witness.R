suppressPackageStartupMessages({
  library(ctmle)
  library(glmnet)
})
options(digits = 17)

# The R witness of C-TMLE0, R ctmle 0.1.2 at 18de559, ctmletype = 3.
#
# Usage: witness.R INPUTS.csv PENALTIES.csv OUT_DIR
#
# 1. glmnet at the two declared penalties gives g1W and g1WPrev (block A). A second run at a
#    tighter threshold gives the solver's self-gap.
# 2. type3() is a copy of R/ctmle_glmnet.R lines 100-112 and 133-174. Two edits only: the glm
#    control is an argument, and the function returns its intermediate arrays. Its default
#    pass must equal ctmleGlmnet(ctmletype = 3) bit for bit.
# 3. A second pass of the copy at glm.control(epsilon = 1e-14, maxit = 100) is the reference
#    the package is compared with.
# 4. ctmleGeneral(ctmletype = 3) must equal ctmleGlmnet(ctmletype = 3) bit for bit, and
#    fluctuation = "linear" must equal "logistic" bit for bit (type 3 ignores it).
# 5. Steps 2-4 also run for block B (a partner that crosses gbound) and block C (a mechanism
#    clipped at both bounds).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: witness.R INPUTS.csv PENALTIES.csv OUT_DIR")
frame <- read.csv(args[[1]])
declared <- read.csv(args[[2]])
out <- args[[3]]
gbound <- declared$gbound
alpha <- declared$alpha
lam <- declared$lam
lam_plus <- declared$lam_plus
cross_factor <- declared$cross_factor

W <- as.matrix(frame[, paste0("W", 1:6)])
A <- frame$A
path <- sort(c(lam, lam_plus), decreasing = TRUE)

lasso_pair <- function(thresh) {
  fit <- glmnet(W, factor(A), family = "binomial", lambda = path, thresh = thresh, maxit = 1e7)
  cbind(
    as.vector(predict(fit, newx = W, s = lam, type = "response")),
    as.vector(predict(fit, newx = W, s = lam_plus, type = "response"))
  )
}
pair <- lasso_pair(1e-14)
tight <- lasso_pair(1e-18)
g1W <- pair[, 1]
g1WPrev <- pair[, 2]

# ctmle_glmnet.R lines 100-112 (the initial Q and the outcome scaling), then lines 133-174
# (type 3), at 18de559. `bound` is tmle::bound, which ctmle reaches through Depends.
type3 <- function(Y, A, Q, g1W, g1WPrev, gbound, alpha, control = glm.control()) {
  Qbounds <- range(Y)
  QAW <- (1 - A) * Q[, 1] + A * Q[, 2]
  Q <- cbind(QAW, Q0W = Q[, 1], Q1W = Q[, 2])
  Q <- bound(Q, Qbounds)
  Ystar <- bound(Y, Qbounds)
  ab <- range(Ystar, na.rm = TRUE)
  Ystar[is.na(Ystar)] <- 0
  Ystar <- (Ystar - ab[1])/diff(ab)
  Q <- (Q - ab[1])/diff(ab)
  Qbounds <- c(alpha, 1 - alpha)
  Q <- bound(Q, Qbounds)
  Q <- qlogis(Q)
  Qinitial <- Q

  g <- g1W
  Y <- Ystar
  X <- data.frame(A)

  g1W.total1 <- bound(g1WPrev, c(gbound, 1-gbound))
  g0W.total1 <- 1 - bound(g1WPrev, c(gbound, 1-gbound))
  g1W.total <- bound(g, c(gbound, 1-gbound))
  g0W.total <- 1 - bound(g, c(gbound, 1-gbound))

  H1W <- X[,1]/g1W.total
  H0W <- (1 - X[,1])/g0W.total

  ddg1<-(g1W.total-g1W.total1)
  zeros <- sum(ddg1 == 0)
  ddg1[which(ddg1==0)]<-1e-10
  ddg0<-(g0W.total-g0W.total1)
  zeros <- zeros + sum(ddg0 == 0)
  ddg0[which(ddg0==0)]<-1e-10

  ddH1W <- (X[,1]/(g1W.total^2)) * ddg1
  ddH0W <- ((1 - X[,1])/(g0W.total^2)) * ddg0

  suppressWarnings(epsilon<- coef(glm(Y ~ -1 +
                                        offset(Q[, "QAW"])+
                                        H0W+H1W+
                                        ddH0W + ddH1W, family = "binomial", control = control)))

  Qstar <- Q + cbind((epsilon[1] * H0W+ epsilon[2] * H1W),
                     epsilon[1]/g0W.total, epsilon[2]/g1W.total)+
    cbind((epsilon[3] * ddH0W+ epsilon[4] * ddH1W),
          (epsilon[3]/(g0W.total^2))*ddg0, (epsilon[4]/(g1W.total^2))*ddg1)
  Qstar <- plogis(Qstar)

  est  <- (mean(Qstar[,"Q1W"]) - mean(Qstar[,"Q0W"]))*diff(ab)

  Dstar <-  (A/g1W.total - (1 - A)/(1-g1W.total))  * (Y - Qstar[, "QAW"]) +
    Qstar[, "Q1W"] - Qstar[, "Q0W"] -  (mean(Qstar[,"Q1W"]) - mean(Qstar[,"Q0W"]))

  var.psi <- var(Dstar)/length(Y) * (diff(ab)^2)
  CI <- est + c(-1.96,1.96) * sqrt(var.psi)
  list(
    Ystar = Ystar, Qinitial = Qinitial, epsilon = epsilon, Qstar = Qstar, Dstar = Dstar,
    est = est, var.psi = var.psi, CI = CI, ab = ab, zeros = zeros
  )
}

run_block <- function(name, Y, Q, g1W, g1WPrev, family) {
  verbatim <- type3(Y, A, Q, g1W, g1WPrev, gbound, alpha)
  strict <- type3(Y, A, Q, g1W, g1WPrev, gbound, alpha, glm.control(epsilon = 1e-14, maxit = 100))
  package <- ctmleGlmnet(
    Y = Y, A = A, W = data.frame(W), Q = Q, ctmletype = 3, g1W = g1W, g1WPrev = g1WPrev,
    alpha = alpha, family = family, gbound = gbound
  )
  general <- ctmleGeneral(
    Y = Y, A = A, W = data.frame(W), Q = Q, ctmletype = 3, gn_candidates = NULL,
    g1W = g1W, g1WPrev = g1WPrev, alpha = alpha, family = family, gbound = gbound
  )
  linear <- ctmleGlmnet(
    Y = Y, A = A, W = data.frame(W), Q = Q, ctmletype = 3, g1W = g1W, g1WPrev = g1WPrev,
    alpha = alpha, family = family, gbound = gbound, fluctuation = "linear"
  )
  for (other in list(package, general, linear)) {
    stopifnot(
      identical(other$est, verbatim$est),
      identical(other$var.psi, verbatim$var.psi),
      identical(other$CI, verbatim$CI)
    )
  }
  rows <- data.frame(
    Y = Y, Ystar = verbatim$Ystar, A = A, g1W = g1W, g1WPrev = g1WPrev,
    QAW = verbatim$Qinitial[, "QAW"], Q0W = verbatim$Qinitial[, "Q0W"],
    Q1W = verbatim$Qinitial[, "Q1W"],
    Qstar_AW = strict$Qstar[, "QAW"], Qstar_0W = strict$Qstar[, "Q0W"],
    Qstar_1W = strict$Qstar[, "Q1W"], Dstar = strict$Dstar,
    verbatim_Qstar_AW = verbatim$Qstar[, "QAW"], verbatim_Qstar_0W = verbatim$Qstar[, "Q0W"],
    verbatim_Qstar_1W = verbatim$Qstar[, "Q1W"]
  )
  write.csv(rows, file.path(out, paste0("rows-", name, ".csv")), row.names = FALSE)
  data.frame(
    block = name, n = length(Y), ab_low = verbatim$ab[1], ab_high = verbatim$ab[2],
    H0W = strict$epsilon[["H0W"]], H1W = strict$epsilon[["H1W"]],
    ddH0W = strict$epsilon[["ddH0W"]], ddH1W = strict$epsilon[["ddH1W"]],
    est = strict$est, var_psi = strict$var.psi,
    verbatim_H0W = verbatim$epsilon[["H0W"]], verbatim_H1W = verbatim$epsilon[["H1W"]],
    verbatim_ddH0W = verbatim$epsilon[["ddH0W"]], verbatim_ddH1W = verbatim$epsilon[["ddH1W"]],
    verbatim_est = verbatim$est, verbatim_var_psi = verbatim$var.psi,
    package_est = package$est, package_var_psi = package$var.psi,
    package_ci_low = package$CI[1], package_ci_high = package$CI[2],
    package_pvalue = package$pvalue, zero_ddg = verbatim$zeros
  )
}

binary_Q <- cbind(frame$Q0W_binary, frame$Q1W_binary)
continuous_Q <- cbind(frame$Q0W_continuous, frame$Q1W_continuous)
crossing <- plogis(cross_factor * qlogis(g1WPrev))
summary <- rbind(
  run_block("A-binary", frame$Y_binary, binary_Q, g1W, g1WPrev, "binomial"),
  run_block("A-continuous", frame$Y_continuous, continuous_Q, g1W, g1WPrev, "gaussian"),
  run_block("B", frame$Y_binary, binary_Q, g1W, crossing, "binomial"),
  run_block("C", frame$Y_binary, binary_Q, frame$gM, frame$gMPrev, "binomial")
)
write.csv(summary, file.path(out, "summary.csv"), row.names = FALSE)
write.csv(
  data.frame(g1W = g1W, g1WPrev = g1WPrev, g1W_tight = tight[, 1], g1WPrev_tight = tight[, 2]),
  file.path(out, "lasso.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    ctmle_version = as.character(packageVersion("ctmle")),
    glmnet_version = as.character(packageVersion("glmnet")),
    tmle_version = as.character(packageVersion("tmle")),
    bound_source = environmentName(environment(bound)),
    glmnet_fdev = glmnet.control()$fdev,
    glmnet_devmax = glmnet.control()$devmax,
    self_gap = max(abs(tight - pair))
  ),
  file.path(out, "versions.csv"),
  row.names = FALSE
)
