# Plan cell P of X41: R ltmle 1.3-0 g-computation and IPTW points, per replicate, on laws D2B
# and G.  A skeleton: the Python side that writes the samples and reads the output is not
# registered yet (README.md beside this file).  It runs in the pinned image of
# tests/canonical/ltmle.
#
# The two calls are those of the W9 fixture after commit 7f6eb50c.  The outcome regressions
# read the covariates and no arm column, as cleverly's `outcome_design` does at a label or rule
# node.  The mechanisms read the full history.
suppressPackageStartupMessages(library(ltmle))
source("/fixture/study_harness.R")
options(digits = 17)

paths <- study_arguments("usage: run_study.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
truths <- read.csv(paths$truths, stringsAsFactors = FALSE, check.names = FALSE)

# The declared Yrange of law G: `ContinuousLaw.q_bounds()` in
# tests/studies/sequential_baselines_laws.py.  D2B has a binary outcome.
G_YRANGE <- c(-3.025001, 4.165)

LAWS <- list(
  D2B = list(
    columns = c("W", "L1", "A1", "C1", "L2", "A2", "C2", "Y"),
    anodes = c("A1", "A2"),
    cnodes = c("C1", "C2"),
    # L1 precedes the first arm, so ltmle reads it as a baseline column.
    lnodes = "L2",
    qform = c(L2 = "Q.kplus1 ~ W + L1", Y = "Q.kplus1 ~ W + L1 + L2"),
    gform = c(
      "A1 ~ W + L1",
      "C1 ~ W + L1 + A1",
      "A2 ~ W + L1 + A1 + L2",
      "C2 ~ W + L1 + A1 + L2 + A2"
    ),
    yrange = NULL,
    regimes = function(frame) {
      n <- nrow(frame)
      l2 <- ifelse(is.na(frame$L2), 0, frame$L2)
      list(always = matrix(1, n, 2), never = matrix(0, n, 2), rule = cbind(frame$L1, l2))
    }
  ),
  G = list(
    columns = c(
      "W", "V", "L1", "A1", "C1", "L2", "A2", "C2", "L3", "A3", "C3", "Y"
    ),
    anodes = c("A1", "A2", "A3"),
    cnodes = c("C1", "C2", "C3"),
    lnodes = c("L2", "L3"),
    qform = c(
      L2 = "Q.kplus1 ~ W + V + L1",
      L3 = "Q.kplus1 ~ W + V + L1 + L2",
      Y = "Q.kplus1 ~ W + V + L1 + L2 + L3"
    ),
    gform = c(
      "A1 ~ W + V + L1",
      "C1 ~ W + V + L1 + A1",
      "A2 ~ W + V + L1 + A1 + L2",
      "C2 ~ W + V + L1 + A1 + L2 + A2",
      "A3 ~ W + V + L1 + A1 + L2 + A2 + L3",
      "C3 ~ W + V + L1 + A1 + L2 + A2 + L3 + A3"
    ),
    yrange = G_YRANGE,
    regimes = function(frame) {
      n <- nrow(frame)
      list(
        treat_all = matrix(1, n, 3),
        treat_none = matrix(0, n, 3),
        rule = matrix(frame$V, n, 3)
      )
    }
  )
)

ltmle_point <- function(law, panel, abar, gcomp, stratify) {
  arguments <- list(
    data = panel,
    Anodes = law$anodes,
    Cnodes = law$cnodes,
    Lnodes = law$lnodes,
    Ynodes = "Y",
    survivalOutcome = FALSE,
    Qform = law$qform,
    gform = law$gform,
    abar = abar,
    gbounds = c(0.01, 1),
    SL.library = "glm",
    stratify = stratify,
    gcomp = gcomp,
    estimate.time = FALSE,
    variance.method = "ic"
  )
  if (!is.null(law$yrange)) arguments$Yrange <- law$yrange
  withCallingHandlers(
    study_stage("ltmle", do.call(ltmle, arguments)),
    warning = function(condition) {
      if (grepl("Cnodes|censoring", conditionMessage(condition), ignore.case = TRUE)) {
        invokeRestart("muffleWarning")
      }
      study_fatal_error(sprintf("unexpected ltmle warning: %s", conditionMessage(condition)))
    }
  )
}

fit_one <- function(frame) {
  replicate <- frame$replicate[[1]]
  scenario <- frame$scenario[[1]]
  law <- LAWS[[frame$law[[1]]]]
  panel <- frame[law$columns]
  for (node in law$cnodes) {
    panel[[node]] <- BinaryToCensoring(is.uncensored = panel[[node]])
  }
  rows <- list()
  for (name in names(law$regimes(frame))) {
    abar <- law$regimes(frame)[[name]]
    plug_in <- ltmle_point(law, panel, abar, gcomp = TRUE, stratify = TRUE)
    weighted <- ltmle_point(law, panel, abar, gcomp = FALSE, stratify = FALSE)
    rows[[length(rows) + 1L]] <- data.frame(
      implementation = "ltmle-r-sequential-baselines",
      scenario = scenario,
      replicate = replicate,
      regime = name,
      estimator = c("gcomp", "iptw"),
      estimate = c(unname(plug_in$estimates[1]), unname(weighted$estimates["iptw"])),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

expected <- length(unique(truths$replicate))
# One replicate of G at n = 2,000 is 2,000 lines, so the chunk holds 32 replicates.
results <- study_stream(paths$samples, fit_one, expected, chunk_lines = 64000L)
study_collect(
  results,
  expected = expected,
  output = paths$output,
  versions = study_version("ltmle"),
  key = c("scenario", "replicate", "regime", "estimator")
)
