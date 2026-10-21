# The lmtp 1.5.4 adapter of the repeated-outcome SDR study (`longitudinal-repeated-sdr`, X32b
# plan K12).
#
# `lmtp_sdr_adapter.R` is sourced unchanged: the gated hash of `longitudinal-sdr` covers it, and
# that file already supplies `lmtp_sdr_with_folds`, the shifted values, the per-node ratios and
# the Python fold assignment.  This file adds one learner, `SL.cellmeans.outcomes`, which
# `run_ensemble` dispatches to in the same way as the saturated `SL.cellmeans` of that adapter.
#
# The pairing learner.  cleverly fits `CellMeansOn("outcomes", CellMeans())` (the Python class
# `CellMeansOnOutcomes`): cell means keyed on the earlier outcomes Y_{<t} among the plan's
# followers through t.  lmtp fits one pooled regression per node on every row uncensored at t,
# with the arm in its design.  This learner keys that regression on the design columns named
# `^Y[0-9]+$` (the earlier outcomes) and `^A[0-9]+$` (the arms through t).  At the prediction
# point the arm is the plan's, A_{<=t} = d_{<=t}(Y_{<t}), so the lmtp cell is the set of
# followers with that Y_{<t}, which is the cleverly cell.  The key must hold Y_{<t}: a rule
# follower is defined by Y_{<t} and A_{<=t} together.  W and L are left out of the key on
# purpose.  The outcome regression is then wrong and the correction is nonzero in every
# replicate, which is what makes the pairing test something (plan X32b section 5.3).
#
# A prediction in a cell no training row occupies is NA, as in `lmtp_cellmeans_learner`.  lmtp
# predicts every row and masks the rows it does not read, so an NA on a row it does read may
# reach the estimate and fail the group, while an NA on a masked row leaves no trace.  The
# runner therefore checks the cells before each fit (`check_cells` in
# tests/canonical/sdr_repeated/run_study.R) and refuses, inside a stage, a group that has an
# unseen cell.  The primary n makes that rare (`expected_unseen_cell_fits` in
# tests/studies/canonical_sdr_repeated.py).
#
# The key is read from the columns lmtp hands the learner, not from a list of names, so a
# column lmtp adds that matches neither pattern stays out of the key.  The outcome column `y`
# is removed before the patterns apply, because the final-node outcome Y_h matches `^Y[0-9]+$`.

source("/fixture/lmtp_sdr_adapter.R")

lmtp_cellmeans_outcomes_learner <- function(data, y, learners, outcome_type, id, folds) {
  features <- setdiff(names(data), c(id, y))
  features <- features[grepl("^Y[0-9]+$", features) | grepl("^A[0-9]+$", features)]
  if (!any(grepl("^A[0-9]+$", features))) {
    # Without the arm the cell is not the lmtp counterpart of the cleverly cell.
    study_fatal_error(sprintf(
      "SL.cellmeans.outcomes found no arm column among %s", paste(names(data), collapse = ", ")
    ))
  }
  key <- do.call(paste, c(lapply(data[features], as.character), sep = "\r"))
  means <- tapply(data[[y]], key, mean)
  structure(list(means = means, features = features), class = "lmtp_cellmeans")
}

local({
  namespace <- asNamespace("lmtp")
  # The binding now holds the dispatcher `lmtp_sdr_adapter.R` installed, so the saturated
  # learner stays reachable behind this one.
  previous <- get("run_ensemble", envir = namespace)
  dispatch <- function(data, y, learners, outcome_type, id, folds) {
    if (identical(learners, "SL.cellmeans.outcomes")) {
      return(lmtp_cellmeans_outcomes_learner(data, y, learners, outcome_type, id, folds))
    }
    previous(data, y, learners, outcome_type, id, folds)
  }
  unlockBinding("run_ensemble", namespace)
  assign("run_ensemble", dispatch, envir = namespace)
  lockBinding("run_ensemble", namespace)
})
