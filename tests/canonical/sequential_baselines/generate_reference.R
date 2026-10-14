# W9: R ltmle 1.3-0 g-computation and IPTW points on data.csv, for the sequential baselines.
# Run in the pinned image of tests/canonical/ltmle (see README.md beside this file).
library(ltmle)
options(digits = 17)

fixture_dir <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(fixture_dir)) fixture_dir <- "."

raw <- read.csv(file.path(fixture_dir, "data.csv"))
panel <- data.frame(
  W1 = raw$W1,
  W2 = raw$W2,
  A1 = raw$A1,
  C1 = BinaryToCensoring(is.uncensored = raw$C1),
  L2 = raw$L2,
  A2 = raw$A2,
  C2 = BinaryToCensoring(is.uncensored = raw$C2),
  Y = raw$Y
)

# Every regression reads the full history.  Among the followers of a static regime the arm
# columns are constant, and glm drops them as aliased, as cleverly's GLM does.
qform <- c(
  L2 = "Q.kplus1 ~ W1 + W2 + A1",
  Y = "Q.kplus1 ~ W1 + W2 + A1 + L2 + A2"
)
gform <- c(
  "A1 ~ W1 + W2",
  "C1 ~ W1 + W2 + A1",
  "A2 ~ W1 + W2 + A1 + L2",
  "C2 ~ W1 + W2 + A1 + L2 + A2"
)

n <- nrow(panel)
second <- ifelse(is.na(raw$L2), 0, as.numeric(raw$L2 > 0))
regimes <- list(
  always = matrix(1, n, 2),
  never = matrix(0, n, 2),
  rule = cbind(as.numeric(raw$W1 > 0), second)
)

fit <- function(abar, gcomp, stratify) {
  ltmle(
    panel,
    Anodes = c("A1", "A2"),
    Cnodes = c("C1", "C2"),
    Lnodes = "L2",
    Ynodes = "Y",
    survivalOutcome = FALSE,
    Qform = qform,
    gform = gform,
    abar = abar,
    gbounds = c(0.01, 1),
    SL.library = "glm",
    stratify = stratify,
    gcomp = gcomp,
    estimate.time = FALSE,
    variance.method = "ic"
  )
}

rows <- list()
for (name in names(regimes)) {
  plug_in <- fit(regimes[[name]], gcomp = TRUE, stratify = TRUE)
  weighted <- fit(regimes[[name]], gcomp = FALSE, stratify = FALSE)
  rows[[length(rows) + 1]] <- data.frame(
    regime = name,
    estimator = c("gcomp", "iptw"),
    ltmle_name = c(names(plug_in$estimates)[1], "iptw"),
    estimate = c(unname(plug_in$estimates[1]), unname(weighted$estimates["iptw"]))
  )
}
reference <- do.call(rbind, rows)
write.csv(reference, file.path(fixture_dir, "reference.csv"), row.names = FALSE)
writeLines(
  c(
    paste("R", R.version$major, R.version$minor, sep = " "),
    paste("ltmle", as.character(packageVersion("ltmle")))
  ),
  file.path(fixture_dir, "versions.txt")
)
