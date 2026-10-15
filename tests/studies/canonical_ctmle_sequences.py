"""Registered evidence for the lasso and learners C-TMLE sequences, ``canonical-ctmle-sequences``.

The subject is ``CTMLE(strategy="lasso")`` and ``CTMLE(strategy="learners")``.  The comparator
is R ``ctmle`` 0.1.2 at ``18de559``: ``ctmleGlmnet(ctmletype = 1)`` for the lasso path, and
``build_gn_seq`` with ``ctmleGeneral(ctmletype = 1)`` for the declared candidates.  The image
is built from the unchanged ``tests/canonical/ctmle_selector`` context, and the runner
``tests/canonical/ctmle_sequences/run_ctmle_sequences.R`` is mounted beside the shared harness.
The design is rev 2.2 of the X16 plan, sections 3.4 to 3.4.3.

**The laws** are two binary laws of one family on ``W1`` to ``W4``, one per scenario:
:func:`x16_lasso_law` and :func:`x16_learners_law`, with the exact ATE from
:func:`~tests.studies.canonical_tmle.truth_for`.  ``W1`` is an instrument, ``W3`` a
confounder and an effect modifier.  The outcome GLM of both sides is main terms, so it omits
the ``A W3`` term: it is misspecified, and each candidate's TMLE limit depends on how much of
``W3`` its mechanism carries.  ``tests/unit/test_ctmle_sequences_design.py`` checks that the
candidate limits spread, that the plug-in is biased, and the share of ``g`` outside the bounds.
This record draws its own datasets through :func:`~tests.studies.evidence.seeds.draw_replicate`.

**The penalty grids** are read from ``tests/canonical/ctmle_sequences/lambdas.csv``, which both
sides read.  The file holds one grid per scenario, 20 penalties each, by :func:`penalty_design`
on that scenario's law: log-spaced from ``top`` down to ``top / 100``, with ``top`` equal to
:data:`TOP_FRACTION` times the 1st percentile of ``lambda_max`` over :data:`DESIGN_DRAWS`
design draws at the primary ``n``.  :func:`learner_lambdas` are entries
:data:`LEARNER_ENTRIES` of the ``binary_learners`` grid.

**binary_lasso** fits ``strategy="lasso", lambdas=STUDY_LAMBDAS["binary_lasso"] + (0,)`` in
sample, with five unstratified selection folds that :func:`assert_unstratified_selection`
reads back.  R fits ``ctmleGlmnet`` on the same rows with the same split, carried in the
``selection_fold`` column, and runs its own type-1 final step from the package's
cross-validated index, carried in the ``package_selected`` column.

**binary_learners** pairs on R's own split.  ``make_r_folds.R`` draws ``folds_R`` for every
replicate from the seed R sets before that replicate, and the result is committed as
``r_folds.csv.gz``.  The package fits ``strategy="learners"`` with ``selection_split=folds_R``
and one :class:`~cleverly.estimators.PropensityArrays` pair for each of
:func:`learner_lambdas`.  The terminal is the main-terms logistic regression.

**Gates.**  Two kinds.  The *truth gates* are the framework's bias, coverage and SE-ratio rows
of each implementation; only the package's bias rows carry a pass claim.  The *replay gates*
compare what both sides compute from identical inputs: R writes its outcome and propensity
coefficients, and :mod:`tests.studies.ctmle_sequence_replay` rebuilds R's arrays and runs the
package's construction and final step on them.  :func:`scientific_failures` reports a red
replay gate.  The paired comparison of the selected estimates is descriptive
(``paired_role="descriptive"``): quirk 2 of R ``ctmle`` drives R's cross-validation to the
last candidate, so the two selected estimates are not the same estimator.

**The R runs are pending** (``pending_reference``, Study queue item 9).  The record runs the
Python phase and the property study, and it gates the package's truth rows, the structural
invariants and the property cells.  The replay gates, the selection files and the paired
comparison wait for the change that registers the R runs.

**Structural invariants** of every package fit fail the run: the path length, the labels,
``selected == argmin(cv_risk)`` at the first minimum, and ``final is None`` exactly when the
lasso fit selected its last index (:func:`check_structure`).

**Red cells.**  The record publishes under ``reporting``, declared before the run.
:data:`RED_CELL_OWNERS` names the owner of each kind of red row, and :data:`EXPECTED_RED`
the red set the declaration expects.  The ledger and the roadmap's
"Red-cell owners" table list a row only after the run shows it red, because
``tests/unit/test_red_cell_ledger.py`` refuses an owner with no red row.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly._typing import EstimandName
from cleverly.datasets import DGP
from cleverly.estimators import CTMLE, PropensityArrays
from cleverly.learners._l1_logistic import L1Logistic, default_penalty_grid
from cleverly.utils.bounds import expit
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ctmle_selector import (
    CTMLE_COMMIT,
    R_BASE_IMAGE,
    assert_unstratified_selection,
)
from tests.studies.canonical_cvtmle import STRATIFY_FOLDS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 800
PRIMARY_N = 2000
SEED = 20261601
G_BOUNDS = (0.025, 0.975)
SELECTION_FOLDS = 5
COVARIATES = ("W1", "W2", "W3", "W4")
IMPLEMENTATION = "cleverly-ctmle-sequences"
REFERENCE = "r-ctmle"
SCENARIOS = ("binary_lasso", "binary_learners")

#: Where the comparator's inputs live.  Both files are written in the declaration step and
#: committed with it, so this module imports without them and only a fit reads them.
ARTIFACTS = ROOT / "tests" / "canonical" / "ctmle_sequences"
LAMBDAS_FILE = ARTIFACTS / "lambdas.csv"
R_FOLDS_FILE = ARTIFACTS / "r_folds.csv.gz"

#: The grid rule.  ``top`` is this fraction of the 1st percentile of ``lambda_max`` over the
#: design draws, and the grid falls log-linearly from ``top`` to ``top * GRID_RATIO``.
TOP_FRACTION = 0.8
TOP_PERCENTILE = 1.0
GRID_RATIO = 0.01
GRID_SIZE = 20
DESIGN_DRAWS = 50
#: The label of the design draws' seed stream, so they share no sample with any replicate.
DESIGN_STREAM = "lambda-design"
#: The 1-based entries of the ``binary_learners`` grid the declared candidates use, least
#: adaptive first.  On :func:`x16_learners_law` entries 2 and 4 leave most of ``W3`` out, and
#: entries 16 and 19 carry it (plan 3.4.1, N2).
LEARNER_ENTRIES = (2, 4, 16, 19)
#: The number of candidates of each scenario's path: the grid and the penalty 0, or the
#: declared entries and the terminal.
PATH_LENGTHS = {"binary_lasso": GRID_SIZE + 1, "binary_learners": len(LEARNER_ENTRIES) + 1}

#: The terminal and outcome learner of both scenarios: main-terms logistic, unpenalized.
LOGISTIC_C = 1e6
LOGISTIC_MAX_ITER = 2000

SCENARIO_ESTIMANDS: Mapping[str, tuple[EstimandName, ...]] = {
    "binary_lasso": ("ate",),
    "binary_learners": ("ate",),
}

PROPERTY_CELLS = {
    "double_robustness": (
        "lasso_outcome_correct",
        "lasso_treatment_correct",
        "learners_outcome_correct",
        "learners_treatment_correct",
        "both_wrong_lasso",
        "both_wrong_learners",
    ),
    "selector_necessity": ("lasso_collaborative", "learners_collaborative", "empty_control"),
    "multi_arm": ("lasso_treatment_correct", "empty_control"),
}

#: The ``property/cell`` keys of the design test's ``DECLARED_GATES`` whose bias is a stand-in
#: from another study and not a measurement of this study's cell.  A stand-in cannot fail on
#: this study's code, so ``tests/canonical/ctmle_sequences/regenerate.py`` refuses a full run
#: while this set names a key.  It is empty: the property pilot of
#: ``tests/canonical/ctmle_sequences/power_probe.py`` measured the four cells that held one.
STAND_INS: frozenset[str] = frozenset()

#: The owner of each kind of red row, declared before the run (plan 3.4).  F18 owns the
#: inference the selector paths withhold: every coverage or standard-error-ratio red, of the
#: truth rows and of the property cells, and every ``selector_necessity`` red.
#: ``X16-sequence-reading`` reads a red package bias row, a red replay gate, or a red
#: double-robustness positive bias cell first, against quirks 2, 4 and 8 of the X16 plan, the
#: replay flags and ``r-selection.csv``.  The paired rows are descriptive and state no verdict.
#:
#: ``truth/binary_lasso/coverage`` moved from F18 after the first run.  The final step reports
#: the penalty 0 on 771 of 800 fits, and plain TMLE on the same samples covers 0.9225 with an
#: SE ratio of 0.913 (``tests/diagnostics/x16_lasso_coverage/``).  The shortfall is the plug-in
#: variance under near-positivity, which plain TMLE shares, and not the selection that F18
#: owns.
RED_CELL_OWNERS: Mapping[str, str] = {
    "truth/binary_lasso/coverage": "X16-plugin-variance",
    "truth/coverage": "F18",
    "truth/se_ratio": "F18",
    "truth/bias": "X16-sequence-reading",
    "replay": "X16-sequence-reading",
    "property/selector_necessity": "F18",
    "property/double_robustness/positive/bias": "X16-sequence-reading",
    "property/double_robustness/positive/coverage": "F18",
    "property/double_robustness/positive/se_ratio": "F18",
    "property/multi_arm/positive/coverage": "F18",
    "property/multi_arm/positive/se_ratio": "F18",
}

#: The red rows this declaration expects (plan 3.4.3), by the ledger's key.  The
#: ``binary_lasso`` coverage row is expected red: the plan's probe read a coverage of 0.899 and
#: an SE ratio of 0.881.  ``selector_necessity/learners_collaborative`` is a declared expected
#: red on its bias clause (design review S3): the binary selector precedent is 0.217 SD, so the
#: clause passes with probability about 0.05.  The RMSE bound is the necessity family's
#: discriminating gate.  The red bias clause fails the family's joint clause, so the family's
#: other two cells are red by that clause only.  Every other row is expected green.
EXPECTED_RED_TRUTH: tuple[str, ...] = (f"{IMPLEMENTATION}/binary_lasso/ate",)
EXPECTED_RED_PROPERTIES: tuple[str, ...] = (
    "selector_necessity/learners_collaborative",
    "selector_necessity/lasso_collaborative",
    "selector_necessity/empty_control",
)
#: The expected red rows that pass their own rule and are red by the family's joint clause only.
EXPECTED_JOINT_CLAUSE_RED: tuple[str, ...] = (
    "selector_necessity/lasso_collaborative",
    "selector_necessity/empty_control",
)
EXPECTED_RED = EXPECTED_RED_TRUTH + EXPECTED_RED_PROPERTIES

#: The side artefacts.  ``r-selection.csv`` holds one row per R fit beside the package's own
#: selection; ``r-selection-summary.csv`` the counts and the descriptive non-adapter pairing;
#: ``replay.csv`` and ``replay-summary.csv`` the replay gates (plan 3.4.2).
SELECTION_FILE = "r-selection.csv"
SELECTION_SUMMARY_FILE = "r-selection-summary.csv"
REPLAY_FILE = "replay.csv"
REPLAY_SUMMARY_FILE = "replay-summary.csv"
#: Every side artefact reads R's rows, so a run with the comparator pending writes none.  The
#: change that registers the R runs declares these as the record's ``extra_artifacts`` again.
REFERENCE_SIDE_ARTIFACTS = (
    SELECTION_FILE,
    SELECTION_SUMMARY_FILE,
    REPLAY_FILE,
    REPLAY_SUMMARY_FILE,
)
#: Why the R runs are pending, and the queue row that owns them.  The owner's rule: a method
#: merges once its Python studies pass, and the comparator study follows in a later change.
PENDING_REASON = (
    "Study queue item 9: the R ctmle runs need Docker and the pinned reference image, and "
    "a later change registers them"
)
#: The property module's side artefact: the selection of every lasso and learners property fit.
PROPERTY_SELECTION_FILE = "property-selection.csv"
#: The minimum of the default ``glmnet`` path on the training rows of R's cross-validation
#: fold ``v``, the rows outside the package's selection fold ``v - 1``.  ``predict(s = 0)``
#: clamps to it while the construction inside that fold scores its tries.
FOLD_PATH_COLUMNS = tuple(f"path_min_lambda_fold_{v}" for v in range(1, SELECTION_FOLDS + 1))
#: The vector columns of the R row: each holds its values printed with ``%.17g`` and joined by
#: ``;``, or is missing where the scenario or the start has none.
VECTOR_COLUMNS = (
    "q_coef",
    "g_coef",
    "r_accepted",
    "cand_est",
    "cand_loss",
    "final_r_coef",
    "final_r_loss",
    "final_pkg_coef",
    "final_pkg_loss",
)
#: The columns the R runner writes after the published fourteen, read by
#: :func:`extra_artifacts` before the driver drops them.  The first nine are rev 2.1's.
REFERENCE_COLUMNS = (
    "best_k",
    "n_candidates",
    "adapter_used",
    "path_min_lambda",
    *FOLD_PATH_COLUMNS,
    "adapter_error_matched",
    "rebuild_best_k",
    "rebuild_agrees",
    "q_coef",
    "g_coef",
    "r_accepted",
    "cand_est",
    "cand_loss",
    "final_r_start",
    "final_r_coef",
    "final_r_loss",
    "final_r_index",
    "final_r_est",
    "final_pkg_start",
    "final_pkg_coef",
    "final_pkg_loss",
    "final_pkg_index",
    "final_pkg_est",
)
#: The columns of ``r-selection.csv`` that R writes, as :data:`REFERENCE_COLUMNS` minus the
#: vectors the replay reads, with ``r_accepted`` kept.
SELECTION_REFERENCE_COLUMNS = (
    "best_k",
    "n_candidates",
    "adapter_used",
    "adapter_error_matched",
    "rebuild_best_k",
    "rebuild_agrees",
    "path_min_lambda",
    *FOLD_PATH_COLUMNS,
    "r_accepted",
)
#: The package's own selection, carried on its replicate rows to :func:`extra_artifacts`.
#: Every index is 1-based, as R's ``best_k`` is.  ``package_final`` is missing when no final
#: step ran, and ``package_accepted`` joins the accepted indices with ``;``.
PACKAGE_COLUMNS = ("package_selected", "package_final", "package_accepted")
#: The quirk-7 error of ``ctmleGlmnet`` type 1 at a selected last penalty (plan 3.5, fact 6).
QUIRK_7_TEXT = "length of 'dimnames' [1] not equal to array extent"
#: The confidence level of the descriptive paired interval.
SUMMARY_CONFIDENCE = 0.99

STUDY = StudyRecord(
    name="C-TMLE candidate sequences",
    slug="canonical-ctmle-sequences",
    artifacts=ARTIFACTS,
    document="docs/technical-reference/method-evidence/c-tmle-candidate-sequences.md",
    anchor="c-tmle-candidate-sequences",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    # The R ctmle runs are pending.  The change that registers them moves REFERENCE to
    # ``reference``, declares ``paired_role="descriptive"`` (quirk 2 drives R's
    # cross-validation to the last candidate, so the selected estimates of the two sides are
    # not one estimator, and the replay gates carry the comparison), and declares
    # REFERENCE_SIDE_ARTIFACTS as ``extra_artifacts``.
    pending_reference=REFERENCE,
    pending_reason=PENDING_REASON,
    # Reporting, declared before the run.  The selector-necessity cells of both selector
    # studies are red under F18.  The complete result is published, and RED_CELL_OWNERS names
    # who reads each red row.
    publication_policy="reporting",
    property_artifacts=(PROPERTY_SELECTION_FILE,),
    modules=(
        "tests/studies/canonical_ctmle_sequences.py",
        "tests/studies/ctmle_sequence_properties.py",
        "tests/studies/ctmle_sequence_replay.py",
        "tests/canonical/ctmle_sequence_witness/harness.py",
        "tests/studies/canonical_ctmle_selector.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/bounded_cv_laws.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ctmle_sequences",
    properties_module="tests.studies.ctmle_sequence_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "ctmle_commit": CTMLE_COMMIT,
    "ctmle_version": "0.1.2",
    "r_base_image": R_BASE_IMAGE,
    "glmnet_control": "glmnet.control(fdev = 0, devmax = 1)",
    "binary_lasso": (
        "ctmleGlmnet(Y, A, W, Q = cbind(q0, q1), lambdas = c(STUDY_LAMBDAS$binary_lasso, 0), "
        "ctmletype = 1, family = 'binomial', like_type = 'loglik', gbound = 0.025, PEN = FALSE, "
        "V = 5, folds = the package's selection folds, detailed = TRUE); std_error = "
        "sqrt(var.psi) of the type-1 final step"
    ),
    "binary_learners": (
        "build_gn_seq(A, W, SL.library = SL.cv1lasso..SL.cv4lasso at LEARNER_LAMBDAS and "
        "SL.glm, folds = folds_R), then set.seed(s_r) and ctmleGeneral(ctmletype = 1, "
        "gn_candidates, gn_candidates_cv, family = 'binomial', like_type = 'loglik', "
        "gbound = 0.025, PEN = FALSE, V = 5, folds = folds_R, detailed = TRUE); "
        "std_error = sqrt(varDstar_origscale[best_k] / n)"
    ),
    "type_1_adapter": (
        "ctmleGlmnet type 1 has no branch for a selected last penalty (quirk 7). The runner "
        "catches an error of ctmleGlmnet only when its message holds the quirk-7 text "
        f"{QUIRK_7_TEXT!r} and the rebuilt best_k is the last penalty; it then reports "
        "results.all$est[best_k] with varDstar_origscale[best_k] / n. Any other error is a "
        "comparator failure. rebuild_type1 also runs on every replicate where ctmleGlmnet "
        "completes, and the runner stops unless its best_k and results.all$est are identical"
    ),
    "replay_outputs": (
        "coef(glm), coef(gbw, s = lambdas) of the full-sample default path, coef of each "
        "build_gn_seq full fit, the accepted candidates, results.all est and likelihood, and "
        "the type-1 final step from R's best_k and from the package's selected index "
        "(ctmle_glmnet.R:248-327); every coefficient set reproduces R's arrays to 1e-12"
    ),
    "outcome_regression": "glm(Y ~ A + W1 + W2 + W3 + W4, family = binomial())",
    "red_cell_owners": dict(RED_CELL_OWNERS),
}

CONFIGURATION = {
    "cross_fit": False,
    "simultaneous_intervals": False,
    "selection_folds": SELECTION_FOLDS,
    "selection_inner_folds": 2,
    "penalty": False,
    "g_bounds": list(G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "laws": (
        "x16_lasso_law: g = expit(-1.0 + 1.8 W1 + 0.8 W2 + 0.5 W3), Qbar = expit(-0.5 + 0.4 a "
        "+ 0.8 W2 - 0.5 W3 + 2.0 a W3 + 0.5 W4); x16_learners_law: g = expit(-1.2 + 1.2 W1 + "
        "0.8 W2 + 1.0 W3), Qbar = expit(-0.5 + 0.4 a + 0.8 W2 - 1.5 W3 + 3.5 a W3 + 0.5 W4); "
        "W1..W4 independent N(0, 1), binary A and Y"
    ),
    "outcome_learner": (
        f"LogisticRegression(C={LOGISTIC_C:g}) on A, W1, W2, W3, W4: main terms, which omit "
        "the A W3 term of both laws"
    ),
    "lambda_grid": (
        f"{GRID_SIZE} log-spaced penalties per scenario from top to top * {GRID_RATIO:g}, top "
        f"= {TOP_FRACTION} x the {TOP_PERCENTILE:g}st percentile of lambda_max over "
        f"{DESIGN_DRAWS} design draws of that scenario's law at n = {PRIMARY_N}; read from "
        "lambdas.csv; binary_lasso appends the penalty 0"
    ),
    "learner_lambdas": (
        f"binary_learners grid entries {', '.join(str(entry) for entry in LEARNER_ENTRIES)}"
    ),
    "binary_lasso_folds": (
        f"unstratified {SELECTION_FOLDS}-fold selection assignments drawn from the "
        "estimator's own seed and asserted; R reads the same assignment"
    ),
    "binary_learners_folds": (
        "folds_R from make_r_folds.R, committed as r_folds.csv.gz; selection_split = "
        "folds_R - 1; PropensityArrays cv columns fitted out of fold on folds_R"
    ),
    "declared_differences": (
        "the package refits the outcome inside each selection fold; R refits epsilon on every "
        "row inside each fold (quirk 2) and refits glmnet on every row in its evaluation "
        "(quirk 3); R clamps the penalty 0 to the minimum of the path it reads, the training "
        "rows' path while a construction scores its tries and the full-sample path elsewhere "
        "(quirk 4, recorded per fold in r-selection.csv); R's construction criterion always "
        "adds var(D*), whatever PEN is, so with PEN = FALSE R's construction is penalized and "
        "the package's (penalty=False) is not, and R's stop can bind where the package's "
        "cannot; inside each cross-validation fold R scales that construction penalty by "
        "n_train / n (quirk 8); for binary_learners the package refits its terminal under the "
        "split, while R reads SL.glm's cv column"
    ),
    "paired_role": (
        "descriptive: quirk 2 drives R's cross-validation to the last candidate, so the "
        "selected estimates are not compared under a verdict"
    ),
    "replay_gates": (
        "per scenario, on every replicate without a near_tie or clip flag: accepted indices "
        "identical; each candidate estimate and loss within 1e-6; for each final-step start, "
        "the final index identical and the estimate and every option loss within 1e-6; at "
        "least 400 unflagged replicates, and for binary_lasso at least 100 final-step "
        "comparisons. The package side uses R's arrays, R's initial Q, penalty=True, "
        "loss='loglik', alpha=0.995"
    ),
    "expected_adapter_rate": (
        "close to every binary_lasso replicate: on the rev 2.1 law R chose the last penalty on "
        "10 of 10 probe replicates (one-sided 95% Clopper-Pearson lower bound 0.74), because "
        "quirk 2 drives "
        "R's cross-validated loss toward the end. About 0 of 800 replicates are expected to be "
        "compared without the adapter (95% upper bound about 207). The rate is published and "
        "not gated"
    ),
    "inference": (
        "package: plug-in D* diagnostic (plugin_std_error); R: var.psi of the type-1 final "
        "step for binary_lasso and varDstar_origscale[best_k] / n for binary_learners"
    ),
    "failure_rule": (
        "a failed or dropped replication, a non-finite estimate, an R error outside the type-1 "
        "adapter, a quirk-5 separation violation, a rebuild disagreement, a coefficient "
        "reconstruction mismatch, or a broken structural invariant of a package fit fails "
        "the run"
    ),
    "red_cell_owners": dict(RED_CELL_OWNERS),
    "expected_red": list(EXPECTED_RED),
    "comparison_scope": "binary ATE, two arms, complete outcomes, in-sample nuisances",
}


# --------------------------------------------------------------------------- the laws


def _law(
    name: str,
    propensity_coef: tuple[float, float, float, float],
    outcome_coef: tuple[float, float, float, float, float, float],
) -> DGP:
    b0, b1, b2, b3 = propensity_coef
    c0, ca, c2, c3, cia, c4 = outcome_coef

    def propensity(w: np.ndarray) -> np.ndarray:
        return np.asarray(expit(b0 + b1 * w[:, 0] + b2 * w[:, 1] + b3 * w[:, 2]), dtype=float)

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        linear = c0 + ca * a + c2 * w[:, 1] + c3 * w[:, 2] + cia * a * w[:, 2] + c4 * w[:, 3]
        return np.asarray(expit(linear), dtype=float)

    return DGP(
        name=name,
        n_latent=len(COVARIATES),
        covariate_names=COVARIATES,
        propensity=propensity,
        outcome_mean=outcome_mean,
        family="binomial",
    )


@functools.cache
def x16_lasso_law() -> DGP:
    """The ``binary_lasso`` law: a weak, late confounder (plan 3.4.1).

    ``g(W) = expit(-1.0 + 1.8 W1 + 0.8 W2 + 0.5 W3)`` and
    ``Qbar(a, W) = expit(-0.5 + 0.4 a + 0.8 W2 - 0.5 W3 + 2.0 a W3 + 0.5 W4)``.  ``W3`` enters
    the lasso path late, so the cross-validation stops early on many fits and the final step
    carries the estimate back to the unpenalized end.

    Returns
    -------
    DGP
        The law.
    """
    return _law("x16_lasso", (-1.0, 1.8, 0.8, 0.5), (-0.5, 0.4, 0.8, -0.5, 2.0, 0.5))


@functools.cache
def x16_learners_law() -> DGP:
    """The ``binary_learners`` law: a strong confounder and a gap in the declared entries.

    ``g(W) = expit(-1.2 + 1.2 W1 + 0.8 W2 + 1.0 W3)`` and
    ``Qbar(a, W) = expit(-0.5 + 0.4 a + 0.8 W2 - 1.5 W3 + 3.5 a W3 + 0.5 W4)``.  The early
    declared entries leave much of ``W3`` out, and the late ones carry it, so the
    cross-validation can reject the biased candidates (plan 3.4.1).

    Returns
    -------
    DGP
        The law.
    """
    return _law("x16_learners", (-1.2, 1.2, 0.8, 1.0), (-0.5, 0.4, 0.8, -1.5, 3.5, 0.5))


def scenario_law(scenario: str) -> DGP:
    """The law one scenario samples.

    Parameters
    ----------
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.

    Returns
    -------
    DGP
        :func:`x16_lasso_law` or :func:`x16_learners_law`.
    """
    if scenario == "binary_lasso":
        return x16_lasso_law()
    if scenario == "binary_learners":
        return x16_learners_law()
    raise KeyError(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw directly from a supplied seed for the manifest-seed audit."""
    from tests.studies.canonical_tmle import truth_for

    dgp = scenario_law(scenario)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


# --------------------------------------------------------------------------- the grid


def penalty_design(
    sample: Any,
    n: int,
    *,
    size: int,
    covariates: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    """The declared grid rule: ``size`` log-spaced penalties from ``top`` to ``top / 100``.

    ``top`` is :data:`TOP_FRACTION` times the :data:`TOP_PERCENTILE` percentile of
    ``lambda_max`` over :data:`DESIGN_DRAWS` draws.  ``lambda_max`` is the first value of
    :func:`~cleverly.learners._l1_logistic.default_penalty_grid`, the smallest penalty at which
    every coefficient is zero.  Each draw takes its seed from this record's
    :data:`DESIGN_STREAM` stream under ``label``, so no draw is a replicate.

    Parameters
    ----------
    sample : callable
        ``sample(n, seed) -> frame`` for the law the grid is for.
    n : int
        The sample size of every draw.
    size : int
        The number of penalties.
    covariates : tuple of str
        The covariate columns the penalized mechanism reads.
    label : str
        The name of the law, which keeps two laws' design draws apart.

    Returns
    -------
    tuple of float
        The penalties, strictly decreasing and positive.
    """
    tops = []
    for draw in range(DESIGN_DRAWS):
        frame = sample(n, stream_seed(STUDY, DESIGN_STREAM, label, n, draw))
        x = frame.loc[:, list(covariates)].to_numpy(dtype=float)
        tops.append(float(default_penalty_grid(x, frame["A"].to_numpy())[0]))
    top = TOP_FRACTION * float(np.percentile(tops, TOP_PERCENTILE))
    grid = top * np.exp(np.linspace(0.0, math.log(GRID_RATIO), size))
    return tuple(float(value) for value in grid)


def primary_grid(scenario: str) -> tuple[float, ...]:
    """One scenario's grid, recomputed by the rule that wrote ``lambdas.csv``.

    Parameters
    ----------
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.

    Returns
    -------
    tuple of float
        :data:`GRID_SIZE` penalties.
    """
    law = scenario_law(scenario)

    def sample(n: int, seed: int) -> pd.DataFrame:
        return draw_from_seed(scenario, n, seed)[0]

    return penalty_design(sample, PRIMARY_N, size=GRID_SIZE, covariates=COVARIATES, label=law.name)


def _missing(path: Path, what: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{path.relative_to(ROOT).as_posix()} is missing. It holds {what}, which the "
        "declaration step writes and commits; see tests/canonical/ctmle_sequences/README.md"
    )


@functools.cache
def _grid_table() -> dict[str, tuple[float, ...]]:
    if not LAMBDAS_FILE.exists():
        raise _missing(LAMBDAS_FILE, "the declared penalty grids")
    table = pd.read_csv(LAMBDAS_FILE, float_precision="round_trip")
    if list(table.columns) != ["scenario", "entry", "lambda"]:
        raise ValueError("lambdas.csv must hold the columns scenario, entry, lambda")
    if set(table["scenario"]) != set(SCENARIOS):
        raise ValueError(f"lambdas.csv must hold one grid for each of {SCENARIOS}")
    out: dict[str, tuple[float, ...]] = {}
    for scenario in SCENARIOS:
        rows = table.loc[table["scenario"] == scenario]
        if len(rows) != GRID_SIZE or not np.array_equal(rows["entry"], np.arange(1, GRID_SIZE + 1)):
            raise ValueError(f"lambdas.csv must hold entries 1 to {GRID_SIZE} for {scenario}")
        values = rows["lambda"].to_numpy(dtype=float)
        if not (np.all(values > 0.0) and np.all(np.diff(values) < 0.0)):
            raise ValueError("lambdas.csv must hold positive, strictly decreasing penalties")
        out[scenario] = tuple(float(value) for value in values)
    return out


def study_lambdas(scenario: str) -> tuple[float, ...]:
    """``STUDY_LAMBDAS[scenario]``: one committed grid, without the appended penalty 0.

    Parameters
    ----------
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.

    Returns
    -------
    tuple of float
        The :data:`GRID_SIZE` penalties of that scenario in ``lambdas.csv``, in file order.

    Raises
    ------
    FileNotFoundError
        When ``lambdas.csv`` has not been written yet.
    ValueError
        When the file breaks the grid rule's shape.
    """
    return _grid_table()[scenario]


def learner_lambdas() -> tuple[float, ...]:
    """``LEARNER_LAMBDAS``: entries :data:`LEARNER_ENTRIES` of the ``binary_learners`` grid.

    Returns
    -------
    tuple of float
        One penalty per declared candidate, least adaptive first.
    """
    grid = study_lambdas("binary_learners")
    return tuple(grid[entry - 1] for entry in LEARNER_ENTRIES)


def __getattr__(name: str) -> Any:
    # The grids keep the plan's names, and read the committed file only when used.
    if name == "STUDY_LAMBDAS":
        return {scenario: study_lambdas(scenario) for scenario in SCENARIOS}
    if name == "LEARNER_LAMBDAS":
        return learner_lambdas()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.cache
def _r_folds_table() -> dict[int, np.ndarray]:
    if not R_FOLDS_FILE.exists():
        raise _missing(R_FOLDS_FILE, "R's selection folds for binary_learners")
    table = pd.read_csv(R_FOLDS_FILE)
    if list(table.columns) != ["replicate", "row", "fold"]:
        raise ValueError("r_folds.csv.gz must hold the columns replicate, row, fold")
    out: dict[int, np.ndarray] = {}
    for replicate, group in table.groupby("replicate", sort=True):
        ordered = group.sort_values("row")
        if not np.array_equal(ordered["row"], np.arange(len(ordered))):
            raise ValueError(f"r_folds.csv.gz replicate {replicate} does not list rows 0 to n-1")
        out[int(replicate)] = ordered["fold"].to_numpy(dtype=np.int64)
    return out


def r_folds(replicate: int, n: int) -> np.ndarray:
    """R's fold of each row of ``binary_learners`` replicate ``replicate``, as 1 to 5.

    Parameters
    ----------
    replicate : int
        The replicate index.
    n : int
        The sample size, which the committed folds must match.

    Returns
    -------
    ndarray of int
        One fold label per row, in sample order.

    Raises
    ------
    FileNotFoundError
        When ``r_folds.csv.gz`` has not been written yet.
    ValueError
        When the committed folds do not cover this replicate at this ``n``.
    """
    table = _r_folds_table()
    if replicate not in table:
        raise ValueError(f"r_folds.csv.gz has no replicate {replicate}")
    folds = table[replicate]
    if folds.size != n:
        raise ValueError(
            f"r_folds.csv.gz holds {folds.size} rows for replicate {replicate}, not n = {n}. "
            f"The folds are drawn at the declared n = {PRIMARY_N}"
        )
    if set(np.unique(folds).tolist()) != set(range(1, SELECTION_FOLDS + 1)):
        raise ValueError(f"r_folds.csv.gz replicate {replicate} does not use folds 1 to 5")
    return folds


# --------------------------------------------------------------------------- the fits


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, solver="lbfgs")


def _settings() -> dict[str, Any]:
    """The estimator keywords both scenarios share."""
    return {
        "outcome_learner": _logistic(),
        "treatment_learner": _logistic(),
        "cross_fit": False,
        "selection_inner_folds": 2,
        "penalty": False,
        "estimands": ("ate",),
        "ctmle_estimand": "ate",
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        # The outcome is binary, so no ``q_bounds``: the scaler is already the identity.
        "stratify_folds": STRATIFY_FOLDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }


def propensity_arrays(frame: pd.DataFrame, folds: np.ndarray) -> list[PropensityArrays]:
    """The ``binary_learners`` candidates: one ``(full, cv)`` pair per learner penalty.

    ``full`` is :class:`~cleverly.learners._l1_logistic.L1Logistic` fitted on every row.
    ``cv`` predicts the rows of each fold from a fit on the other four, so each row's ``cv``
    value comes from a fit that never read that row.

    Parameters
    ----------
    frame : DataFrame
        The replicate's sample.
    folds : ndarray of int
        R's fold of each row, 1 to 5.

    Returns
    -------
    list of PropensityArrays
        One pair per entry of :func:`learner_lambdas`, least adaptive first.
    """
    x = frame.loc[:, list(COVARIATES)].to_numpy(dtype=float)
    a = frame["A"].to_numpy()
    pairs = []
    for penalty in learner_lambdas():
        full = L1Logistic(penalty=penalty).fit(x, a).predict_proba(x)[:, 1]
        cv = np.full(len(frame), np.nan)
        for fold in range(1, SELECTION_FOLDS + 1):
            held = folds == fold
            model = L1Logistic(penalty=penalty).fit(x[~held], a[~held])
            cv[held] = model.predict_proba(x[held])[:, 1]
        pairs.append(PropensityArrays(full=full, cv=cv))
    return pairs


def estimator(scenario: str, frame: pd.DataFrame, split: np.ndarray | None = None) -> CTMLE:
    """The unfitted subject of one scenario on one sample.

    Parameters
    ----------
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.
    frame : DataFrame
        The replicate's sample, which the learners arrays are fitted on.
    split : ndarray of int or None
        R's folds, 1 to 5, for ``binary_learners``.

    Returns
    -------
    CTMLE
        The estimator, with the study's settings.
    """
    if scenario == "binary_lasso":
        return CTMLE(
            strategy="lasso",
            lambdas=(*study_lambdas(scenario), 0.0),
            selection_folds=SELECTION_FOLDS,
            **_settings(),
        )
    if scenario == "binary_learners":
        if split is None:
            raise ValueError("binary_learners needs R's folds")
        return CTMLE(
            strategy="learners",
            treatment_candidates=propensity_arrays(frame, split),
            selection_split=(split - 1).tolist(),
            **_settings(),
        )
    raise KeyError(scenario)


def check_structure(result: Any, scenario: str) -> None:
    """Refuse a fit that breaks a structural invariant of its scenario (plan 3.4).

    Each one is definitional, so a broken one is a defect and fails the run:

    * the path holds :data:`PATH_LENGTHS` candidates, with the declared labels;
    * ``selected`` is the first minimum of ``cv_risk``;
    * a lasso fit runs its final step exactly when ``selected`` is not the last index, and a
      learners fit never runs one.

    Parameters
    ----------
    result : Any
        The single-parameter result.
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.

    Raises
    ------
    RuntimeError
        Naming the broken invariant.
    """
    selection = result.extra["ctmle"]
    expected = PATH_LENGTHS[scenario]
    problems = []
    if len(selection.path) != expected or len(selection.cv_risk) != expected:
        problems.append(f"the path holds {len(selection.path)} candidates, not {expected}")
    labels = _declared_labels(scenario)
    if tuple(selection.labels) != labels:
        problems.append(f"the labels are {selection.labels}, not {labels}")
    cv_risk = np.asarray(selection.cv_risk, dtype=float)
    if selection.selected != int(np.argmin(cv_risk)):
        problems.append(
            f"selected is {selection.selected}, but the first minimum of cv_risk is "
            f"{int(np.argmin(cv_risk))}"
        )
    last = expected - 1
    if scenario == "binary_lasso":
        if (selection.final is None) != (selection.selected == last):
            problems.append(
                f"the final step is {selection.final} with selected {selection.selected}; it "
                "runs exactly when selected is not the last index"
            )
    elif selection.final is not None:
        problems.append(f"a learners fit reports a final step, {selection.final}")
    if problems:
        raise RuntimeError(f"{scenario}: a structural invariant fails: " + "; ".join(problems))


def _declared_labels(scenario: str) -> tuple[str, ...]:
    if scenario == "binary_lasso":
        penalties = (*study_lambdas(scenario), 0.0)
        return tuple("lambda=0" if value == 0.0 else f"lambda={value:.4g}" for value in penalties)
    entries = tuple(f"arrays[{index}]" for index in range(len(LEARNER_ENTRIES)))
    return (*entries, "LogisticRegression[terminal]")


def fit_cleverly(
    frame: pd.DataFrame, scenario: str, replicate: int, *, folds: np.ndarray | None = None
) -> Any:
    """Fit one scenario's subject on one replicate, and check its structure.

    Parameters
    ----------
    frame : DataFrame
        The replicate's sample.
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.
    replicate : int
        The replicate index, which selects R's folds for ``binary_learners``.
    folds : ndarray of int or None
        R's folds, 1 to 5.  ``None`` reads them from ``r_folds.csv.gz``.

    Returns
    -------
    Any
        The single-parameter result.
    """
    split = None
    if scenario == "binary_learners":
        split = r_folds(replicate, len(frame)) if folds is None else np.asarray(folds)
    result = (
        estimator(scenario, frame, split)
        .fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES))
        .single()
    )
    if scenario == "binary_lasso":
        assert_unstratified_selection(result)
    else:
        assert split is not None
        selection = result.extra["ctmle"].folds
        if not np.array_equal(selection.assignment, split - 1):
            raise RuntimeError("the learners fit did not score its candidates on R's folds")
    check_structure(result, scenario)
    return result


def accepted_indices(n_steps: tuple[int, ...]) -> tuple[int, ...]:
    """The accepted indices, from 0, of a path built with ``penalty=False``.

    Without the penalty no try scores above the initial fit it fluctuates, so the stop rule
    cannot bind (plan 1.3).  Every block of equal ``n_steps`` then ends at an accepted index.

    Parameters
    ----------
    n_steps : tuple of int
        ``CTMLESelection.n_steps``.

    Returns
    -------
    tuple of int
        The last index of each block, in order.

    Raises
    ------
    RuntimeError
        When the blocks do not count up from 1 in steps of 1.
    """
    steps = np.asarray(n_steps, dtype=int)
    ends = [
        index
        for index in range(steps.size)
        if index == steps.size - 1 or steps[index + 1] != steps[index]
    ]
    if not np.array_equal(steps[ends], np.arange(1, len(ends) + 1)):
        raise RuntimeError(f"n_steps {tuple(n_steps)} do not count accepted blocks from 1")
    return tuple(ends)


def package_selection(result: Any) -> dict[str, Any]:
    """:data:`PACKAGE_COLUMNS` of one fit, 1-based.

    Parameters
    ----------
    result : Any
        The single-parameter result.

    Returns
    -------
    dict
        ``package_selected``, ``package_final`` (NaN when no final step ran) and
        ``package_accepted``.
    """
    selection = result.extra["ctmle"]
    return {
        "package_selected": int(selection.selected) + 1,
        "package_final": math.nan if selection.final is None else int(selection.final) + 1,
        "package_accepted": ";".join(
            str(index + 1) for index in accepted_indices(tuple(selection.n_steps))
        ),
    }


def _rows_from_result(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    estimate = result["ate"]
    reference = float(truth["ate"])
    std_error, low, high = reported_inference(estimate)
    psi = float(estimate.psi)
    if not math.isfinite(psi):
        raise RuntimeError(f"{scenario} replicate {replicate} gave a non-finite estimate")
    return [
        {
            "implementation": IMPLEMENTATION,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": "ate",
            "truth": reference,
            "estimate": psi,
            "inference_estimate": psi,
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
            **package_selection(result),
        }
    ]


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario, replicate)
    rows = _rows_from_result(result, truth, scenario, replicate, len(frame))
    return [{column: row[column] for column in REPLICATE_COLUMNS} for row in rows]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit_cleverly(frame, scenario, replicate)
    sample = frame.copy()
    # The split the subject scored its candidates on, taken off the fit.  For binary_lasso R
    # selects against it.  For binary_learners it is folds_R - 1, and R redraws folds_R from
    # its own seed and checks the draw against r_folds.csv.gz.  Both columns need the fit, so
    # a failed fit leaves its replicate without a sample, and the driver passes no
    # ``reference_input``.
    sample.insert(0, "package_selected", int(result.extra["ctmle"].selected) + 1)
    sample.insert(0, "selection_fold", result.extra["ctmle"].folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return sample, truth_row, _rows_from_result(result, truth, scenario, replicate, len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, n)
            for scenario in STUDY.scenarios
            for replicate in range(replicates)
        ],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes.kept])
    rows = pd.DataFrame([row for _, _, records in outcomes.kept for row in records])
    # The package's selection travels on its rows to ``extra_artifacts``; the driver keeps
    # only REPLICATE_COLUMNS in the published replicate file.
    return replication.result(
        outcomes,
        rows.loc[:, [*REPLICATE_COLUMNS, *PACKAGE_COLUMNS]],
        samples=samples,
        truths=truths,
    )


# --------------------------------------------------------------------------- side artefacts


def selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per R fit, with the package's own selection beside R's.

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows of both implementations, with the reference's extra
        columns and the package's :data:`PACKAGE_COLUMNS`.

    Returns
    -------
    DataFrame
        ``scenario``, ``replicate``, :data:`SELECTION_REFERENCE_COLUMNS` and
        :data:`PACKAGE_COLUMNS`, sorted.
    """
    reference = rows.loc[rows["implementation"] == REFERENCE]
    missing = [column for column in REFERENCE_COLUMNS if column not in reference.columns]
    if missing:
        raise RuntimeError(f"the R rows carry no {missing} columns for {SELECTION_FILE}")
    subject = rows.loc[rows["implementation"] == IMPLEMENTATION]
    missing = [column for column in PACKAGE_COLUMNS if column not in subject.columns]
    if missing:
        raise RuntimeError(f"the package rows carry no {missing} columns for {SELECTION_FILE}")
    out = reference.loc[:, ["scenario", "replicate", *SELECTION_REFERENCE_COLUMNS]].copy()
    for column in ("best_k", "n_candidates", "adapter_used"):
        out[column] = out[column].astype(int)
    if out.duplicated(["scenario", "replicate"]).any():
        raise RuntimeError("the R rows hold a replicate twice")
    learners = out["scenario"] == "binary_learners"
    if (out.loc[learners, "adapter_used"] != 0).any():
        raise RuntimeError("the type-1 adapter applies to binary_lasso only")
    lasso = out.loc[~learners]
    if (lasso["adapter_used"] != lasso["adapter_error_matched"]).any():
        raise RuntimeError("an adapter replicate carries no matched quirk-7 error")
    if (lasso.loc[lasso["adapter_used"] == 0, "rebuild_agrees"] != 1).any():
        raise RuntimeError("a completed ctmleGlmnet fit disagrees with its rebuild")
    package = subject.loc[:, ["scenario", "replicate", *PACKAGE_COLUMNS]]
    out = out.merge(package, on=["scenario", "replicate"], how="left", validate="one_to_one")
    return out.sort_values(["scenario", "replicate"], ignore_index=True)


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan


def selection_summary(rows: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """The adapter counts, and the descriptive pairing of the replicates R fit directly.

    One row per scenario.  The replicates where R needed the type-1 adapter are left out of
    the pairing, so it reads only fits that ``ctmleGlmnet`` completed itself.  The interval is
    a Student interval at :data:`SUMMARY_CONFIDENCE` on the paired estimate difference,
    ``cleverly`` minus R, conditional on ``ctmleGlmnet`` completing without the adapter.  It
    is descriptive and states no verdict.  With no compared replicate every statistic is
    NaN, and the page prints "no replicate compared" (``claims.citation``).

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows.
    selection : DataFrame
        :func:`selection_rows` of the same rows.

    Returns
    -------
    DataFrame
        One row per scenario.
    """
    from scipy.stats import t as student

    out = []
    for scenario in SCENARIOS:
        chosen = selection.loc[selection["scenario"] == scenario]
        kept = chosen.loc[chosen["adapter_used"] == 0, "replicate"]
        subset = rows.loc[(rows["scenario"] == scenario) & rows["replicate"].isin(kept)]
        # When R needed the adapter for every replicate, nothing is compared and the columns
        # of the pivot are empty.  The summary then reports zero compared replicates.
        paired = subset.pivot(index="replicate", columns="implementation", values="estimate")
        paired = paired.reindex(columns=[IMPLEMENTATION, REFERENCE]).dropna()
        difference = (paired[IMPLEMENTATION] - paired[REFERENCE]).to_numpy(dtype=float)
        count = int(difference.size)
        mean = float(np.mean(difference)) if count else math.nan
        sd = float(np.std(difference, ddof=1)) if count > 1 else math.nan
        half = (
            float(student.ppf(0.5 + SUMMARY_CONFIDENCE / 2.0, count - 1)) * sd / math.sqrt(count)
            if count > 1
            else math.nan
        )
        truth = subset.groupby("replicate")["truth"].first().reindex(paired.index)
        out.append(
            {
                "scenario": scenario,
                "replicates": len(chosen),
                "adapter_replicates": int((chosen["adapter_used"] != 0).sum()),
                "r_selected_last": int((chosen["best_k"] == PATH_LENGTHS[scenario]).sum()),
                "package_selected_last": int(
                    (chosen["package_selected"] == PATH_LENGTHS[scenario]).sum()
                ),
                "compared_replicates": count,
                "confidence_level": SUMMARY_CONFIDENCE,
                "mean_difference": mean,
                "sd_difference": sd,
                "ci_lower": mean - half,
                "ci_upper": mean + half,
                "max_abs_difference": float(np.max(np.abs(difference))) if count else math.nan,
                "rmse_cleverly": _rmse(paired[IMPLEMENTATION] - truth),
                "rmse_reference": _rmse(paired[REFERENCE] - truth),
            }
        )
    return pd.DataFrame(out)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The selection files and the replay files, from the rows' extra columns.

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows of both implementations.

    Returns
    -------
    dict
        Each side artefact by its file name.  It is empty while the comparator is pending.
    """
    from tests.studies import ctmle_sequence_replay as replay

    if STUDY.comparator_state == "pending":
        # No R row exists to read.  The record declares no side artefact while it is pending.
        if (rows["implementation"] == REFERENCE).any():
            raise RuntimeError(f"the comparator {REFERENCE} is pending, but its rows are present")
        return {}
    selection = selection_rows(rows)
    compared = replay.replay_rows(rows)
    return {
        SELECTION_FILE: selection,
        SELECTION_SUMMARY_FILE: selection_summary(rows, selection),
        REPLAY_FILE: compared,
        REPLAY_SUMMARY_FILE: replay.replay_summary(compared),
    }


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The red replay gates, which the regeneration reports under ``reporting``.

    Parameters
    ----------
    artifacts : Mapping
        The side artefacts by file name.

    Returns
    -------
    dict
        ``{"replay gate": rows}``: the ``replay-summary.csv`` rows whose gate failed.  It is
        empty while the comparator is pending.
    """
    if STUDY.comparator_state == "pending":
        # The replay gates read R's arrays, so they wait for the R runs.
        return {}
    summary = artifacts[REPLAY_SUMMARY_FILE]
    return {"replay gate": summary.loc[~summary["passed"].astype(bool)]}
