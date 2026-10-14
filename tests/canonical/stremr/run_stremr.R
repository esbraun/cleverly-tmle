# stremr survNPMSM, directIPW and fit_GCOMP on long.csv, for the sequential baselines.
# Run in the image built from the Dockerfile beside this file (see README.md).
suppressPackageStartupMessages({
  library(stremr)
  library(data.table)
})
options(digits = 17)

fixture_dir <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(fixture_dir)) fixture_dir <- "."

long <- fread(file.path(fixture_dir, "long.csv"))
long[, `:=`(A.set1 = 1L, A.set0 = 0L)]
# stratifyQ_by_rule fits the node-t regression on the rows that follow the rule at t alone, so
# the node-2 rows mix both first arms.  Interacting every covariate with Aprev splits that fit
# into one fit per first arm.  The arm that follows the plan is cleverly's node-2 fit.
interactions <- c("WxAprev", "LxAprev", "LprevxAprev")
long[, `:=`(WxAprev = W * Aprev, LxAprev = L * Aprev, LprevxAprev = Lprev * Aprev)]
data <- importData(
  long,
  ID = "ID", t = "t", covars = c("W", "L", "Lprev", "Aprev", interactions),
  CENS = "C", TRT = "A", OUTCOME = "Y"
)
# One treatment model per time, on the history that cleverly's design holds at that node.
data <- fitPropensity(
  data,
  gform_CENS = "C ~ W",
  gform_TRT = "A ~ W + L + Lprev + Aprev",
  stratify_TRT = list(A = c("t == 0L", "t == 1L"))
)
# The outcome regressions read the covariates and no current arm column, as cleverly's outcome
# design does at a label node.  Each is fitted on the rows that follow the plan at its node.
qforms <- c(
  "Qkplus1 ~ W + L + Lprev",
  paste("Qkplus1 ~ W + L + Lprev + Aprev +", paste(interactions, collapse = " + "))
)

rows <- list()
for (plan in c("always", "never")) {
  column <- if (plan == "always") "A.set1" else "A.set0"
  weights <- getIPWeights(data, intervened_TRT = column, rule_name = plan)
  hazard <- survNPMSM(weights, data)$estimates
  direct <- directIPW(weights, data)$estimates
  plug_in <- fit_GCOMP(
    data, tvals = c(0, 1), intervened_TRT = column, Qforms = qforms,
    stratifyQ_by_rule = TRUE
  )$estimates
  # Base indexing: inside a data.table `[`, a bare `time` names the column, not the loop index.
  at <- function(table, column, node) {
    value <- table[[column]][table[["time"]] == node]
    stopifnot(length(value) == 1L)
    value
  }
  for (node in c(0, 1)) {
    rows[[length(rows) + 1]] <- data.frame(
      plan = plan,
      horizon = node + 1,
      estimator = c("survNPMSM", "directIPW", "GCOMP"),
      risk = c(
        1 - at(hazard, "St.NPMSM", node),
        1 - at(direct, "St.directIPW", node),
        1 - at(plug_in, "St.GCOMP", node)
      ),
      max_weight = max(weights[["cum.IPAW"]])
    )
  }
}
write.csv(do.call(rbind, rows), file.path(fixture_dir, "reference.csv"), row.names = FALSE)
writeLines(
  c(
    paste("R", R.version$major, R.version$minor, sep = " "),
    paste("stremr", as.character(packageVersion("stremr"))),
    paste("data.table", as.character(packageVersion("data.table"))),
    paste("speedglm", as.character(packageVersion("speedglm")))
  ),
  file.path(fixture_dir, "versions.txt")
)
