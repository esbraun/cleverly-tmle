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
data <- importData(
  long,
  ID = "ID", t = "t", covars = c("W", "L", "Lprev", "Aprev"),
  CENS = "C", TRT = "A", OUTCOME = "Y"
)
# One treatment model per time, on the history that cleverly's design holds at that node.
data <- fitPropensity(
  data,
  gform_CENS = "C ~ W",
  gform_TRT = "A ~ W + L + Lprev + Aprev",
  stratify_TRT = list(A = c("t == 0L", "t == 1L"))
)
# The outcome regressions read the covariates and no arm column, as cleverly's outcome design
# does at a label node.  Each is fitted on the followers of the plan (stratifyQ_by_rule).
qforms <- rep.int("Qkplus1 ~ W + L + Lprev", 2)

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
  for (time in c(0, 1)) {
    rows[[length(rows) + 1]] <- data.frame(
      plan = plan,
      horizon = time + 1,
      estimator = c("survNPMSM", "directIPW", "GCOMP"),
      risk = c(
        1 - hazard[time == hazard$time, ][["St.NPMSM"]],
        1 - direct[time == direct$time, ][["St.directIPW"]],
        1 - plug_in[time == plug_in$time, ][["St.GCOMP"]]
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
