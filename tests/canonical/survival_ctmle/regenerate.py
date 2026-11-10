"""Regenerate the survival-ctmle evidence (X91 part B) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.survival_ctmle` and its properties
module, fixed before any declared draw from the committed pilot
(``tests/studies/survival_ctmle_pilot.csv.gz``: 1,600 primary replicates by the shared rule).

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X91 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red bias row (gate (a)) of the risk of arm 1 or the risk difference is a defect suspect of
  the selection or its targeting: the outcome model is correct in every gated cell, so the
  fluctuation solves the score whatever the selected treatment and censoring models.
  Diagnose it against ``tests/unit/test_survival_ctmle.py``, then against the full-:math:`g`
  arm's estimates on the same rows (``estimate[cleverly-hazard-ltmle-full-g]``): the arm shares
  the hazard engine, so a bias both arms show points at the engine or the law, and one the
  subject alone shows points at the selection.  The pilot read every gated standardized bias
  inside 0.07 against the margin of 0.25.
* A red row of gate (b) (``rmse-comparison.csv``) is a finding about the selection, not a defect
  of the estimator: the pilot read the upper bound of the RMSE ratio at 0.58 against the
  ceiling of 1.  With no defect found it takes the route above, and the claim that the
  selection improves on the full treatment model at OR 15 is withheld.
* A fit-failure share of the subject above the failure bound (gate (c)) is a defect suspect;
  the pilot lost no subject fit.  A raise of a hazard ``LTMLE`` arm is counted in
  ``comparator_failures`` and decides nothing.
* Coverage, the SE ratio, the reported cells, the risk of arm 0 and the selection readings are
  reported and cannot fail the run (F18).  The pilot's plug-in coverage reads 0.77 to 0.95 in
  the gated cells, the working-mechanism diagnostic the design states.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import survival_ctmle, survival_ctmle_properties

if __name__ == "__main__":
    run(survival_ctmle, survival_ctmle_properties, here=Path(__file__).resolve().parent)
