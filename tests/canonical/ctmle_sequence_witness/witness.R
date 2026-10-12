suppressPackageStartupMessages({
  library(ctmle)
  library(glmnet)
})
options(digits = 17)

# R `ctmle` 0.1.2 at 18de559 and its `glmnet` on the fixture inputs.  Every output is a CSV in
# OUTPUT_DIR.  `regenerate.py` reads them, and `tests/unit/test_ctmle_sequence_witness.py`
# compares them with the package.  Line numbers cite the `ctmle` sources at 18de559.
#
# The same adapter setting as the registered study: the default `glmnet` path runs to
# `lambda_max * 1e-4` instead of stopping early on the deviance (plan 3.5, fact 3).
glmnet.control(fdev = 0, devmax = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: witness.R INPUT_DIR OUTPUT_DIR")
input_dir <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

frame <- read.csv(file.path(input_dir, "inputs.csv"), stringsAsFactors = FALSE)
grid <- read.csv(file.path(input_dir, "grid.csv"))$lambda
names_w <- grep("^W[0-9]+$", names(frame), value = TRUE)
x <- as.matrix(frame[names_w])
n <- nrow(frame)
K <- length(grid)
A <- frame$A
Y <- frame$Y
A3 <- frame$A3
w <- frame$w
Q <- cbind(frame$Q0W, frame$Q1W)
folds <- lapply(sort(unique(frame$fold)), function(f) which(frame$fold == f))
GBOUND <- 0.025

write_frame <- function(value, name) {
  write.csv(value, file.path(output_dir, name), row.names = FALSE)
}

facts <- list()
fact <- function(key, value) facts[[key]] <<- value

wide <- function(predictions, prefix = "g") {
  out <- as.data.frame(matrix(predictions, nrow = n))
  names(out) <- sprintf("%s%02d", prefix, seq_len(ncol(out)))
  out
}

coefficients <- function(fit) {
  beta <- as.matrix(coef(fit))
  out <- data.frame(lambda = fit$lambda, a0 = beta[1, ])
  for (j in seq_along(names_w)) out[[names_w[[j]]]] <- beta[j + 1, ]
  out
}

# ---- 1. The solver at the fixture grid, with a tighter run for the self-gap ---------------
tight <- glmnet(x, A, family = "binomial", lambda = grid, thresh = 1e-14, maxit = 1e7)
tighter <- glmnet(x, A, family = "binomial", lambda = grid, thresh = 1e-16, maxit = 1e8)
g_tight <- predict(tight, newx = x, type = "response")
g_tighter <- predict(tighter, newx = x, type = "response")
fact("self_gap_binary", max(abs(g_tight - g_tighter)))
write_frame(wide(g_tight), "g_tight.csv")
write_frame(coefficients(tight), "coef_tight.csv")

weighted <- glmnet(
  x, A, family = "binomial", weights = w, lambda = grid, thresh = 1e-14, maxit = 1e7
)
weighted_tighter <- glmnet(
  x, A, family = "binomial", weights = w, lambda = grid, thresh = 1e-16, maxit = 1e8
)
g_weighted <- predict(weighted, newx = x, type = "response")
fact(
  "self_gap_weighted",
  max(abs(g_weighted - predict(weighted_tighter, newx = x, type = "response")))
)
write_frame(wide(g_weighted), "g_weighted.csv")
write_frame(coefficients(weighted), "coef_weighted.csv")

# The default `type.multinomial` is "ungrouped": the symmetric objective, one lasso penalty
# for each class coefficient.
multinomial <- glmnet(
  x, factor(A3), family = "multinomial", lambda = grid, thresh = 1e-14, maxit = 1e7
)
multinomial_tighter <- glmnet(
  x, factor(A3), family = "multinomial", lambda = grid, thresh = 1e-16, maxit = 1e8
)
p_multinomial <- predict(multinomial, newx = x, type = "response")
fact(
  "self_gap_multinomial",
  max(abs(p_multinomial - predict(multinomial_tighter, newx = x, type = "response")))
)
classes <- dimnames(p_multinomial)[[2]]
multinomial_wide <- list()
for (k in seq_len(K)) {
  for (c in seq_along(classes)) {
    multinomial_wide[[sprintf("k%02d_a%s", k, classes[[c]])]] <- p_multinomial[, c, k]
  }
}
write_frame(as.data.frame(multinomial_wide), "g_multinomial.csv")
multinomial_coef <- do.call(rbind, lapply(seq_along(classes), function(c) {
  beta <- as.matrix(coef(multinomial)[[classes[[c]]]])
  out <- data.frame(lambda = multinomial$lambda, class = classes[[c]], a0 = beta[1, ])
  for (j in seq_along(names_w)) out[[names_w[[j]]]] <- beta[j + 1, ]
  out
}))
write_frame(multinomial_coef, "coef_multinomial.csv")

# ---- 2. The first value of the default path ---------------------------------------------
fact("lambda_max", glmnet(x, A, family = "binomial")$lambda[[1]])
fact("lambda_max_weighted", glmnet(x, A, family = "binomial", weights = w)$lambda[[1]])

# ---- 3. R's own internal arrays ---------------------------------------------------------
# The construction and the evaluation fit the default path and read it at each penalty with
# `predict(s=)` (`R/functions_glmnet.R`, lines 22-25 and 135-138).
X <- data.frame(A, frame[names_w])
lassox <- model.matrix(as.formula(paste("A ~", paste(names_w, collapse = " + "))), X)[, -1]
default_path <- glmnet(x = lassox, y = as.factor(A), family = "binomial")
fact("default_path_min_lambda", min(default_path$lambda))
fact("default_path_length", length(default_path$lambda))
g_internal <- predict(default_path, newx = lassox, s = grid, type = "response")
write_frame(wide(g_internal), "g_internal.csv")
g_glm <- fitted(glm(A ~ ., data = data.frame(A = A, x), family = binomial()))
write_frame(data.frame(g = g_glm), "g_glm.csv")

# ---- 4. ctmleGeneral on G = cbind(g_internal, g_glm) ------------------------------------
G <- unname(cbind(g_internal, g_glm))
# `cv_general` redraws the folds with `sample` (`R/functions_general.R`, line 287), so the
# seed fixes its draw.  The comparison reads the analysis-sample path only.
set.seed(20261612)
general <- ctmleGeneral(
  Y = Y, A = A, W = data.frame(x), Q = Q, ctmletype = 1,
  gn_candidates = G, gn_candidates_cv = G, family = "binomial", like_type = "loglik",
  gbound = GBOUND, PEN = FALSE, detailed = TRUE, folds = folds, V = length(folds)
)
accepted_columns <- general$candidates$candidates$gns
same <- function(a, b) isTRUE(all(as.vector(a) == as.vector(b)))
general_accepted <- apply(accepted_columns, 2, function(column) {
  hits <- which(apply(G, 2, function(candidate) same(candidate, column)))
  if (length(hits) != 1L) stop("an accepted ctmleGeneral column matches no single candidate")
  hits
})
write_frame(
  data.frame(
    index = seq_len(ncol(G)),
    est = general$candidates$results.all$est,
    likelihood = general$candidates$results.all$likelihood,
    accepted = as.integer(seq_len(ncol(G)) %in% general_accepted)
  ),
  "general_path.csv"
)
fact("general_best_k", general$best_k)

# ---- 5. ctmleGlmnet type 1 on the fixture grid and folds -------------------------------
# Type 1 has no branch for a selected last penalty (plan 1.4, quirk 7).  Any error here
# exits with status 10, and `regenerate.py` records the message and rejects the seed.
set.seed(20261612)
lasso <- tryCatch(
  ctmleGlmnet(
    Y = Y, A = A, W = data.frame(x), Q = Q, lambdas = grid, ctmletype = 1,
    folds = folds, V = length(folds), alpha = 0.995, family = "binomial",
    gbound = GBOUND, like_type = "loglik", fluctuation = "logistic", PEN = FALSE,
    detailed = TRUE
  ),
  error = function(condition) {
    writeLines(conditionMessage(condition), file.path(output_dir, "ctmle_glmnet_error.txt"))
    quit(save = "no", status = 10)
  }
)
best_k <- lasso$best_k
write_frame(
  data.frame(
    index = seq_len(K),
    lambda = grid,
    est = lasso$candidates$results.all$est,
    likelihood = lasso$candidates$results.all$likelihood,
    accepted = as.integer(grid %in% lasso$candidates$candidates$lambdas)
  ),
  "glmnet_path.csv"
)
fact("glmnet_best_k", best_k)
fact("glmnet_est", lasso$est)
fact("glmnet_var_psi", lasso$var.psi)

# The explicit-lambda fit of the final step at R's best_k (`R/ctmle_glmnet.R`, lines
# 270-281), copied verbatim, so the package's final step can run on the same columns.
lambdas_remain <- grid[grid < grid[best_k]]
glmnet_remain <- glmnet(x = lassox, y = as.factor(A), family = "binomial", lambda = lambdas_remain)
gn_remain <- predict(glmnet_remain, newx = lassox, type = "response")
write_frame(wide(gn_remain, "f"), "g_final_cols.csv")

fact("glmnet_version", as.character(packageVersion("glmnet")))
fact("ctmle_version", as.character(packageVersion("ctmle")))
fact("SuperLearner_version", as.character(packageVersion("SuperLearner")))
fact("R_version", paste(R.version$major, R.version$minor, sep = "."))
write_frame(
  data.frame(key = names(facts), value = vapply(facts, function(v) format(v, digits = 17), "")),
  "facts.csv"
)
