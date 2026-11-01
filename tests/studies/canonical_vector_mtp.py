"""Registered evidence for vector modified treatment policies (``vector-mtp``).

The subject is a modified treatment policy on a vector treatment: a node with a binary flag
and one or two doses, read by the joint induced density (``ratio="density"``) or by the stacked
classifier (``ratio="classifier"``), in ``TMLE`` with ``policies=`` and in ``LTMLE``.  The laws,
their truths and efficiency bounds, the plans and the oracle learners live in
:mod:`tests.studies.vector_mtp_common`.  The comparator is R ``lmtp`` 1.5.4.  Four primary
scenarios, 1,000 replications at n = 2,000 each:

==========================================  ===================================================
scenario                                    subject and pairing
==========================================  ===================================================
``point_vector_mixed``                      the point law, ``TMLE``, ``A = (F, D)``.  Plans
                                            ``natural course``, ``dose up`` and ``flag
                                            scaled``.  The ratio is the oracle induced density's
                                            (``PolicySet.policy_ratio``), supplied to ``lmtp``
                                            with both shifted components.  Margin-read
``point_vector_categorical``                the categorical law, ``TMLE``, ``A = (A1, A2)``,
                                            both binary.  Plans ``natural course``, ``a2 off``
                                            and ``a1 on``.  The ratio is the discrete formula on the
                                            exact joint probabilities, supplied.  Margin-read
``longitudinal_vector_mixed``               the two-node law, ``LTMLE``, nodes ``(F1, D1)`` and
                                            ``(F2, D2)``, one fold.  The point plans at both
                                            nodes.  Each node's ratio (``node_ratio``) and
                                            shifted components are supplied.  Exact
``longitudinal_vector_ordered_classifier``  the ordered law as one ``LTMLE`` node
                                            ``(F, D1, D2)``.  Plans ``natural course``,
                                            ``ordered`` and ``full``.  Neither ratio is
                                            supplied: each side fits its own stacked classifier,
                                            the correctly specified logistic regression on the
                                            plan's log ratio, one plan per fit.  Read by the
                                            parity rule (below)
==========================================  ===================================================

**Learners.**  Each density-route fit reads the oracle: :class:`~tests.studies.vector_mtp_common.
Oracle` returns the exact flag probability and the exact pooled-hazard probabilities of each
dose at :func:`~tests.studies.oracle_density_bins.oracle_bins` bins (320 at n = 2,000).  The
categorical scenario reads :class:`~tests.studies.vector_mtp_common.OracleCells`, the exact joint
probabilities.  The outcome regressions are the correctly specified logistic models: main terms
plus the squared dose (point), the squared second-node dose (two-node final regression), the
squared doses (ordered), and the saturated joint levels (categorical).  The two-node first
regression is least squares on ``[W, F1, D1]`` (``SL.glm``'s Gaussian fit on the same columns).
The unsupplied pair's classifier is
:class:`~tests.studies.vector_mtp_common.OracleLogOdds`: an unpenalized logistic regression of
the stack label on one feature, the plan's true log ratio, which the learner computes from each
stacked row's own columns.  It is the correctly specified model of the stacked odds.  A
main-terms classifier is misspecified for a ratio that jumps at the knee and at the caps (60
draws: SE ratios 0.68 and 0.74, coverage 0.85 and 0.87), so it would publish a known-invalid
interval.  The pair fits ``(natural course, ordered)`` and ``(natural course, full)``
separately (:data:`CLASSIFIER_PLANS`), each with its own one-feature learner, because one
learner serves every regimen of a fit.  The reference's mean is read off the ``ordered`` fit,
and each contrast off its own fit.  Under ``natural course`` the two stacked copies are equal,
so either learner fits ``u = 1/2`` and the ratio 1.

**The comparator.**  ``tests/canonical/lmtp_vector_adapter.R`` defines ``lmtp_vector_tmle``,
which builds an ``LmtpTask`` whose treatment nodes are vectors (``trt = list(c("F", "D"))``
and ``list(c("F1", "D1"), c("F2", "D2"))``), writes every shifted component from the columns
``shift__<plan>__<node>__<component>``, and calls ``cf_tmle`` and ``theta_dr`` with the
supplied ratio ``ratio__<plan>__<node>``, ``.trim = 1`` and ``k = Inf``.  It never calls
``cf_density_ratios``.  The shared adapters are sourced and stay byte-identical.  The runner
gives ``lmtp`` the same outcome designs: ``SL.glm`` with the squared doses, and with the
product ``A1 * A2`` on the categorical law, so its design spans the three joint-level
indicators this package fits.  Every categorical component stays a binary numeric column.  The
ordered classifier scenario calls ``lmtp_vector_classifier_tmle`` on the same vector task.  It
calls ``cf_density_ratios`` with ``mtp = TRUE``: with ``mtp = FALSE`` ``lmtp`` 1.5.4 multiplies
each ratio by the indicator that no component moved (``R/density_ratios.R:58-61``), which is
zero on every row a plan moves.  Its learner is ``SL.logratio.<plan>``
(``tests/canonical/vector_mtp_logratio.R``), the R twin of ``OracleLogOdds``, one plan per fit
as above.  It then calls ``cf_tmle``, the plug-in and ``theta_dr`` with the classifier fits,
with the ``1e-5`` prediction screen of ``lmtp_vector_tmle``.  Both sides refuse a fit whose
classifier probability reaches 0.999 (:func:`screen_classifier`), where ``lmtp`` clips the
ratio's denominator and this package does not.

**Pairing reading, declared before any run.**

* ``longitudinal_vector_mixed`` is an exactness witness.  At one fold the designs, the ratios,
  the learners and the intercept fluctuation weighted by the cumulative ratio match, as in
  ``longitudinal-mtp``'s ``mtp_continuous``.  A pre-declaration R smoke requires every paired
  difference below :data:`SMOKE_GATE`, with no prediction at ``lmtp``'s bound of ``1e-5``.
* ``point_vector_mixed`` and ``point_vector_categorical`` are read under the default margins.
  ``TMLE`` with ``policies=`` targets with the covariate submodel over the observed and policy
  values, and ``lmtp`` with the ratio-weighted intercept, as in ``policy-point-mtp``.
* ``longitudinal_vector_ordered_classifier`` is read by the parity rule.  The two sides fit
  one model on one design, and the remaining construction is ``lmtp``'s stacking: its row
  grouping, its prediction bound and its ratio ``pred / (1 - min(pred, 0.999))``.
  ``tests/canonical/vector_mtp_parity.R`` with ``tests/canonical/vector_mtp_parity.py`` fits
  both on three draws and compares every ratio, and checks the R transcription of the log ratio
  on a grid at ``rtol = 1e-12``.  It also requires that the fraction of rows with a positive
  ``lmtp`` ratio equal the package's.  Under ``mtp = TRUE`` both fractions are 1, because
  neither classifier predicts 0, so the witness compares 1 with 1.  Under ``mtp = FALSE``
  ``lmtp``'s fraction falls to the share of unmoved rows, and the witness fails.  The local
  thread runs it before the declared run.  Three outcomes are declared now:

  =====================================================  =====================================
  parity reading                                         consequence
  =====================================================  =====================================
  every ratio gap below :data:`SMOKE_GATE`               exactness witness under
                                                         :data:`SMOKE_GATE`, by a declaration
                                                         amendment before the run
  every relative gap below ``PARITY_TOLERANCE``          margin-read, only after the cause is
  (``1e-4``), with a diagnosed and named numerical       diagnosed, named and recorded in the
  cause such as the IRLS stopping rule                   amendment
  any other reading, a failed transcription or a         the declaration is blocked until the
  failed witness                                         cause is found and fixed
  =====================================================  =====================================

  *Amendment, 2026-10-18, before the declared run.*  The parity check ran at ``main``
  ``d063842c`` and printed ``exact``.  The R transcription of the log ratio showed no ratio or
  feature mismatch over 4,896 grid points (largest gap ``6.39e-14``).  The largest ratio gap
  was ``7.1e-10`` on the ordered plan and ``1.66e-11`` on the full plan.  The positive-ratio
  fraction was 1 on both sides.  The ordered classifier scenario is therefore an exactness
  witness under :data:`SMOKE_GATE`, the first row of the table.

**Linear remainder.**  Every interval this study publishes is a Wald interval on the
efficient influence function.  No interval is exposed to a non-linear remainder of the policy
itself: each plan is a known function of observed values, so the target parameter needs no
extra nuisance.  The remaining second-order terms are the products of the ratio and outcome
errors, which the oracle density makes ``o(n^(-1/2))``.

**Near-zero comparisons.**  The exactness witness compares paired differences against
:data:`SMOKE_GATE` = ``1e-6``, the named absolute floor.  The truth quadrature is checked at
``1e-10`` (:mod:`tests.studies.vector_mtp_common`).  The preimage-cut control's ratio is
checked against the closed form without the flag-moving term, to the binning (median gap below
``1e-3``), in ``tests/unit/test_vector_mtp_design.py``.

**Properties.**  :mod:`tests.studies.vector_mtp_properties` declares every family, its sizes
and the design numbers behind them.

**Red-cell reading, declared before any run.**  Publication policy is ``reporting``, and the
declared reading below is the merge gate, as in ``longitudinal-mtp``.  The fast tier then gates
the artifacts' integrity, and the pull-request review gates the reading.  Every red cell is
diagnosed first, by the rule of ``docs/development/method-benchmarking.md``.  A genuine problem
(an algorithm defect, an inconsistent estimator, or a test or design bug) blocks the merge.  It
is fixed, and the study re-runs under a fresh declaration that states the change.  A red
``interval_calibration`` positive is read as a finite-sample red only if all three hold:

1. its SE-ratio 99% interval contains 1,
2. its coverage interval is consistent with 0.95, and
3. the diagnosis finds no defect.

Such a cell is published red with the owner row ``vector-mtp-calibration``, and it does not
block the merge.  Any other red calibration positive is a defect suspect and blocks the merge.
This holds for every truth row of ``longitudinal_vector_ordered_classifier`` too: its
classifier is correctly specified, so no exception is declared for it.  A red paired row of a
margin-read scenario is read as the construction difference above and gets the owner row
``vector-mtp-pairing``.  A red exactness-witness row or a red primary truth row blocks the
merge and gets a diagnosis.  A red double-robustness or necessity row is a defect suspect and
blocks the merge until its diagnosis ends.  If the diagnosis finds no defect, the row is
published red with the owner row ``vector-mtp-robustness``.  A replication that
raises is never redrawn.  No change may make a test less meaningful or easier to pass.

**Page limits, declared before any run.**  The study has no size ladder.  The oracle bins grow
as ``n^(2/3)`` (:func:`~tests.studies.oracle_density_bins.oracle_bins`), and the
``longitudinal-mtp`` ladder carries the binned ratio's rate, so this study claims behaviour at
n = 2,000 only.  The classifier route is read with the correctly specified one-feature
classifier, so the study says nothing about a learned classifier's rate.  Of the density
route's terms, the necessity pairs see the preimage sum, the Jacobian and the flag map's read
of the natural ``D1``.  The ``D2`` density's and the ``D2`` map's reads of the natural ``D1``
move the outcome-wrong positives by 0.42 and -0.19 SD only, so the fast tier
(``tests/unit/test_vector_density_mutations.py``) is their witness.

**Budget.**  Measured on this container (4 cores, one BLAS thread per worker, four workers) by
the design probes that ``tests/unit/test_vector_mtp_design.py`` records.  Each figure is the
mean seconds of one fit under that load, times the declared replications.

==============================  ==============================  ==========  =================
phase                           fits                            CPU-hours   peak RSS a worker
==============================  ==============================  ==========  =================
Python primary                  4 scenarios x 1,000 (7.7 s a    2.1         0.28 GiB
                                replication)
``lmtp`` reference              4 scenarios x 1,000             not         see below
                                                                measured
``interval_calibration``        point density 24,000 x 3.1 s;   69.4        0.28 GiB
                                two-node density 9,600 x 5.9 s;
                                ordered density 19,200 x 6.0 s;
                                classifiers 3 x 10,000 x 0.1
                                to 0.2 s
``double_robustness``           12 cells x 2,000 (78.1 s a set  43.4        0.28 GiB
                                of 12)
three necessity families        3 pairs x 1,000 (10.5, 11.7     9.4         0.28 GiB
                                and 11.6 s a pair)
==============================  ==============================  ==========  =================

The Python phases take about 124 CPU-hours: 122 for the property families and 2.1 for the
primary fits.  The property phase reserves
:func:`~tests.studies.vector_mtp_properties.fit_bytes` (0.5 GiB) per worker, so
:func:`~tests.parallel.memory_capped_workers` admits 12 workers on a 16 GiB host.  The wall
time is the CPU-hours over the workers, because the fits are independent:

=======  ===========================
workers  wall time, Python phases
=======  ===========================
3        about 41 hours
4        about 31 hours
8        about 15.5 hours
12       about 10.5 hours
=======  ===========================

The declared run is ``--jobs 12 --reference-jobs 3``.  ``generate_property_rows`` is one
:func:`~tests.studies.evidence.replication.fit_properties` call, which keeps a checkpoint per
finished key inside the regeneration driver.  A run that fits each fit set of
:data:`~tests.studies.vector_mtp_properties.FIT_SETS` in replicate chunks outside the driver
gives the same rows when no fit fails: each key's rows depend on the key alone, and the
chunks are assembled in ``FIT_SETS`` order before ``calibration_controls``, the column order
and the sort.  At a budget of two replications per fit set the assembled rows and summary
equal the driver's property phase, byte for byte.

No ``R`` with ``lmtp`` runs in this container, so the reference phase is measured by the R smoke
before the declared run.  Each replication makes 13 ``lmtp`` calls.  Each of the other three
scenarios fits its three regimens.  The unsupplied pair fits ``natural course`` and ``ordered``
with ``SL.logratio.ordered``, and ``natural course`` and ``full`` with ``SL.logratio.full``.
The smoke runs 20 replications at ``--reference-jobs J`` in wall time ``W`` seconds.  The
reference phase then needs about ``1000 W J / 20 / 3600`` CPU-hours.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression

from cleverly.estimators import TMLE
from cleverly.learners.density import bin_edges
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.regimen import _resolve_mtp_node
from tests.parallel import STUDY_JOBS
from tests.studies import vector_mtp_common as common
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.oracle_density_bins import oracle_bins

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20263101
RESAMPLING_SEED = 2026310101
#: The pre-declaration smoke gate on the exactness witness.
SMOKE_GATE = 1e-6
#: Bounds wide enough that no cumulative probability of these laws reaches them.
G_BOUNDS = (0.001, 1.0)
REFERENCE = "natural course"

POINT_MIXED = "point_vector_mixed"
POINT_CATEGORICAL = "point_vector_categorical"
LONG_MIXED = "longitudinal_vector_mixed"
ORDERED_CLASSIFIER = "longitudinal_vector_ordered_classifier"

#: The law of each scenario, and whether ``LTMLE`` reads it.
LAWS: dict[str, str] = {
    POINT_MIXED: "point",
    POINT_CATEGORICAL: "categorical",
    LONG_MIXED: "long",
    ORDERED_CLASSIFIER: "ordered",
}
LONGITUDINAL = frozenset({LONG_MIXED, ORDERED_CLASSIFIER})
LABELS: dict[str, tuple[str, ...]] = {
    POINT_MIXED: common.POINT_LABELS,
    POINT_CATEGORICAL: common.CATEGORICAL_LABELS,
    LONG_MIXED: common.LONG_LABELS,
    ORDERED_CLASSIFIER: common.ORDERED_LABELS,
}

#: The treatment columns of each law, node by node.
NODES: dict[str, tuple[tuple[str, ...], ...]] = {
    "point": (("F", "D"),),
    "categorical": (("A1", "A2"),),
    "long": (("F1", "D1"), ("F2", "D2")),
    "ordered": (("F", "D1", "D2"),),
}
DOSES: dict[str, tuple[str, ...]] = {
    "point": ("D",),
    "categorical": (),
    "long": ("D1", "D2"),
    "ordered": ("D1", "D2"),
}


def names(scenario: str, labels: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The published estimands of a scenario: each plan's mean and its contrast."""
    kind = "regimen" if scenario in LONGITUDINAL else "policy"
    chosen = LABELS[scenario] if labels is None else labels
    return tuple(f"ey_{kind}[{label}]" for label in chosen) + tuple(
        f"ate_{kind}[{label} vs {REFERENCE}]" for label in chosen if label != REFERENCE
    )


ESTIMANDS: dict[str, tuple[str, ...]] = {scenario: names(scenario) for scenario in LAWS}


def truth_of(scenario: str, labels: tuple[str, ...] | None = None) -> dict[str, float]:
    """The exact value of every estimand of a scenario."""
    kind = "regimen" if scenario in LONGITUDINAL else "policy"
    chosen = LABELS[scenario] if labels is None else labels
    truth = common.TRUTHS[LAWS[scenario]]
    means = {label: truth(label) for label in chosen}
    out = {f"ey_{kind}[{label}]": value for label, value in means.items()}
    for label, value in means.items():
        if label != REFERENCE:
            out[f"ate_{kind}[{label} vs {REFERENCE}]"] = value - truth(REFERENCE)
    return out


#: Every published property cell, by family (:mod:`tests.studies.vector_mtp_properties`).
CALIBRATION_LABELS = (
    "point_vector_density",
    "point_vector_classifier",
    "longitudinal_vector_density",
    "longitudinal_vector_classifier",
    "longitudinal_vector_ordered_density",
    "longitudinal_vector_ordered_classifier",
)
ROBUSTNESS_LABELS = (
    "point_vector_ordered_density",
    "longitudinal_vector_density",
    "longitudinal_vector_ordered_density",
)
ROBUSTNESS_CONFIGURATIONS = ("both_correct", "outcome_correct", "outcome_wrong", "both_wrong")
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "double_robustness": tuple(
        f"{label}__{configuration}"
        for label in ROBUSTNESS_LABELS
        for configuration in ROBUSTNESS_CONFIGURATIONS
    ),
    "preimage_necessity": (
        "flag_scaled__declared_preimage",
        "flag_scaled__preimage_cut_control",
    ),
    "jacobian_necessity": (
        "ordered_full__declared_jacobian",
        "ordered_full__jacobian_drop_control",
    ),
    "image_necessity": (
        "ordered_crossed__declared_image",
        "ordered_crossed__shifted_read_control",
    ),
}

STUDY = StudyRecord(
    name="vector modified treatment policies",
    slug="vector-mtp",
    artifacts=ROOT / "tests" / "canonical" / "vector_mtp",
    document="docs/technical-reference/method-evidence/vector-modified-treatment-policies.md",
    anchor="vector-modified-treatment-policies",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-vector-mtp",
    reference="lmtp",
    comparison="matching implementation",
    # RM39b2 rev 3 section 6: the primary bound each side's next run applies.
    failure_bound={"cleverly-vector-mtp": 0.01, "lmtp": 0.01},
    modules=(
        "tests/studies/canonical_vector_mtp.py",
        "tests/studies/vector_mtp_properties.py",
        "tests/studies/vector_mtp_common.py",
        "tests/studies/oracle_density_bins.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_policy_adapter.R",
        "tests/canonical/lmtp_vector_adapter.R",
        "tests/canonical/vector_mtp_logratio.R",
        "tests/canonical/vector_mtp_runner.R",
    ),
    runner_module="tests.studies.canonical_vector_mtp",
    properties_module="tests.studies.vector_mtp_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_tmle and theta_dr on an LmtpTask with vector treatment nodes, every shifted "
        "component and the per-node ratio this package computed written beside the replicate "
        "data (no lmtp density ratio fitted); the ordered classifier scenario fits lmtp's own "
        "stacked classifier through cf_density_ratios with mtp = TRUE, one plan per fit, each "
        "an unpenalized glm on that plan's log ratio computed inside the learner "
        "(SL.logratio.<plan>)"
    ),
    "reference_adapter": "tests/canonical/lmtp_vector_adapter.R",
    "reference_learners": "tests/canonical/vector_mtp_logratio.R",
    "pairing": {
        LONG_MIXED: "exact under SMOKE_GATE",
        POINT_MIXED: "default margins: the point fluctuations differ by construction",
        POINT_CATEGORICAL: "default margins: the point fluctuations differ by construction",
        ORDERED_CLASSIFIER: (
            "exact under SMOKE_GATE: the parity check read every ratio gap below SMOKE_GATE "
            "(amendment 2026-10-18)"
        ),
    },
}

CONFIGURATION = {
    "construction": "ordinary, one fold",
    "density_bins": "ceil(320 (n / 2000)^(2/3)): 320 at n = 2,000, index-only oracle design",
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": {
        POINT_MIXED: "SquaredLogistic(squared=(D,))",
        POINT_CATEGORICAL: "QuasiBinomialGLM on the joint-level indicators",
        LONG_MIXED: "SquaredLogistic(squared=(D2,)); first node LinearRegression",
        ORDERED_CLASSIFIER: "SquaredLogistic(squared=(D1, D2))",
    },
    "treatment_learner": {
        POINT_MIXED: "Oracle (flag probability and pooled-hazard bins)",
        POINT_CATEGORICAL: "OracleCells (exact joint probabilities)",
        LONG_MIXED: "Oracle (flag probability and pooled-hazard bins)",
        ORDERED_CLASSIFIER: (
            "OracleLogOdds per plan: an unpenalized logistic regression of the stack label on "
            "the plan's true log ratio, computed inside the learner from each stacked row"
        ),
    },
    "plans": {scenario: list(labels) for scenario, labels in LABELS.items()},
    "reference": REFERENCE,
}


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def edges_of(frame: pd.DataFrame, law: str) -> tuple[tuple[float, ...], ...]:
    """The whole-sample bin edges each dose's density reads, at :func:`oracle_bins`."""
    bins = oracle_bins(len(frame))
    return tuple(
        tuple(float(v) for v in bin_edges(frame[dose].to_numpy(dtype=float), bins))
        for dose in DOSES[law]
    )


# ---------------------------------------------------------------------------- learners


class OracleCells(BaseEstimator):
    """The exact joint probabilities of the categorical law's four cells, given ``[W1, W2]``.

    The classes are the joint levels in the package's order: ``(0, 0)``, ``(0, 1)``,
    ``(1, 0)`` and ``(1, 1)``.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleCells:
        del X, y, sample_weight
        self.classes_ = np.arange(4)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.shape[1] != 2:
            raise ValueError(f"unexpected categorical design width {x.shape[1]}")
        w1, w2 = x[:, 0], x[:, 1]
        return np.column_stack(
            [
                common._cat_joint(a1, a2, w1, w2)
                for a1, a2 in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
            ]
        )


#: Each law's correctly specified outcome regression in the design each estimator hands it:
#: ``TMLE`` puts the treatment first, ``LTMLE`` puts the history first.
def outcome_learner(law: str, longitudinal: bool) -> Any:
    if law == "categorical":
        return QuasiBinomialGLM()
    if law == "point":
        return common.SquaredLogistic(squared=(1,))
    if law == "ordered":
        return common.SquaredLogistic(squared=(3, 4) if longitudinal else (1, 2))
    if law == "long":
        return common.SquaredLogistic(squared=(5,))
    raise KeyError(law)


def learners(
    law: str,
    frame: pd.DataFrame,
    *,
    longitudinal: bool,
    outcome: str = "correct",
    mechanism: str = "oracle",
    label: str = "",
    pseudo: str = "least_squares",
) -> tuple[Any, Any, Any]:
    """``(outcome, pseudo, treatment)`` learners of one nuisance configuration.

    ``outcome`` is ``"correct"`` or ``"wrong"`` (the prior, and the mean at the first node).
    ``mechanism`` is ``"oracle"``, ``"marginal"`` (the misspecified oracle), ``"log_odds"``
    (:class:`~tests.studies.vector_mtp_common.OracleLogOdds` of ``label``) or ``"cells"``
    (:class:`OracleCells`).  ``pseudo`` is ``"least_squares"`` (the comparator's ``SL.glm``)
    or ``"rich"`` (:class:`~tests.studies.vector_mtp_common.RichPseudo`).
    """
    if outcome == "correct":
        q: Any = outcome_learner(law, longitudinal)
        first: Any = common.RichPseudo() if pseudo == "rich" else LinearRegression()
    elif outcome == "wrong":
        q, first = DummyClassifier(strategy="prior"), DummyRegressor(strategy="mean")
    else:
        raise ValueError(outcome)
    if mechanism in {"oracle", "marginal"}:
        g: Any = common.Oracle(law, edges_of(frame, law), marginal=mechanism == "marginal")
    elif mechanism == "log_odds":
        g = common.OracleLogOdds(law, label)
    elif mechanism == "cells":
        g = OracleCells()
    else:
        raise ValueError(mechanism)
    return q, first, g


# ---------------------------------------------------------------------------- fits


def point_policies(law: str, labels: tuple[str, ...]) -> tuple[Any, ...]:
    if law == "point":
        return common.point_policies(labels)
    if law == "ordered":
        return common.ordered_policies(labels)
    if law == "categorical":
        return common.categorical_policies(labels)
    raise KeyError(law)


def regimens(law: str, labels: tuple[str, ...]) -> dict[str, Any]:
    if law == "long":
        return common.long_regimens(labels)
    if law == "ordered":
        return common.ordered_regimens(labels)
    raise KeyError(law)


def fit_point(
    frame: pd.DataFrame,
    law: str,
    labels: tuple[str, ...],
    *,
    ratio: Literal["density", "classifier"] = "density",
    **configuration: str,
) -> Any:
    """``TMLE`` with ``policies=`` on a point law, under one nuisance configuration."""
    q, _, g = learners(law, frame, longitudinal=False, **configuration)
    treatment = list(NODES[law][0])
    return (
        TMLE(
            policies=point_policies(law, labels),
            outcome_learner=clone(q),
            treatment_learner=clone(g),
            cross_fit=False,
            simultaneous=False,
            density_bins=oracle_bins(len(frame)),
            ratio=ratio,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment=treatment,
            covariates=["W1", "W2"],
            continuous_treatment=list(DOSES[law]),
        )
        .single()
    )


def fit_longitudinal(
    frame: pd.DataFrame,
    law: str,
    labels: tuple[str, ...],
    *,
    ratio: Literal["density", "classifier"] = "density",
    **configuration: str,
) -> Any:
    """One-fold ``LTMLE`` on the two-node law or the ordered law as one node."""
    q, first, g = learners(law, frame, longitudinal=True, **configuration)
    if law == "long":
        baseline, time_varying = ["W"], [[], ["L2"]]
    else:
        baseline, time_varying = ["W1", "W2"], None
    return LTMLE(
        regimens(law, labels),
        reference=REFERENCE,
        outcome_learner=clone(q),
        pseudo_learner=clone(first),
        treatment_learner=clone(g),
        n_folds=1,
        g_bounds=G_BOUNDS,
        density_bins=oracle_bins(len(frame)),
        ratio=ratio,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=[list(node) for node in NODES[law]],  # type: ignore[misc]
        baseline=baseline,
        time_varying=time_varying,
        continuous_treatment=list(DOSES[law]),
    )


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The primary fit of a scenario."""
    law = LAWS[scenario]
    labels = LABELS[scenario]
    if scenario == POINT_MIXED:
        return fit_point(frame, law, labels)
    if scenario == POINT_CATEGORICAL:
        return fit_point(frame, law, labels, mechanism="cells")
    if scenario == LONG_MIXED:
        return fit_longitudinal(frame, law, labels)
    if scenario == ORDERED_CLASSIFIER:
        return PlanFits(
            {
                plan: screen_classifier(
                    fit_longitudinal(
                        frame,
                        law,
                        (REFERENCE, plan),
                        ratio="classifier",
                        mechanism="log_odds",
                        label=plan,
                    )
                )
                for plan in CLASSIFIER_PLANS
            }
        )
    raise KeyError(scenario)


#: The plans of the unsupplied pair.  Each is fitted with the reference in its own ``LTMLE``
#: fit, with :class:`~tests.studies.vector_mtp_common.OracleLogOdds` on that plan's log ratio.
#: The reference's mean is read off the first plan's fit, and each contrast off its own fit.
CLASSIFIER_PLANS = ("ordered", "full")
#: The ratio at which a fitted classifier probability reaches ``lmtp``'s clip at 0.999.
CLASSIFIER_RATIO_SCREEN = 0.999 / (1.0 - 0.999)


def screen_classifier(result: Any) -> Any:
    """Refuse a fit whose classifier probability reaches 0.999 at any row.

    ``lmtp`` reads ``pred / (1 - min(pred, 0.999))`` and this package ``u / (1 - u)`` below
    ``1 - 1e-12``, so the two ratios agree only below 0.999.  The closed-form ratio's largest
    value over 2,000,000 draws is 35.0 (``u = 0.972``).  The largest fitted ratio over the
    1,000-draw design probe of the pair is 49.4 (``u = 0.980``).  The screen sits at 999, so it
    must never bind.
    """
    for label, fitted in result.fits.items():
        if fitted.node_ratio is not None and np.max(fitted.node_ratio) >= CLASSIFIER_RATIO_SCREEN:
            raise RuntimeError(f"a fitted classifier probability of {label!r} reached 0.999")
    return result


class PlanFits:
    """The unsupplied pair's fits, one per plan, read as one result.

    Parameters
    ----------
    by_plan : dict
        Each plan of :data:`CLASSIFIER_PLANS` mapped to its ``LTMLE`` result over
        ``(natural course, plan)``.
    """

    def __init__(self, by_plan: dict[str, Any]) -> None:
        self.by_plan = by_plan
        first = by_plan[CLASSIFIER_PLANS[0]]
        self.data = first.data
        self.fits = {REFERENCE: first.fits[REFERENCE]} | {
            plan: result.fits[plan] for plan, result in by_plan.items()
        }

    def __getitem__(self, name: str) -> Any:
        plan = name[name.index("[") + 1 :].split(" vs ")[0].rstrip("]")
        return self.by_plan[CLASSIFIER_PLANS[0] if plan == REFERENCE else plan][name]


# ---------------------------------------------------------------------------- rows


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.SAMPLERS[LAWS[scenario]](n, seed), truth_of(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def initial_estimates(result: Any, scenario: str) -> dict[str, float]:
    """The untargeted plug-in of every estimand of a fit."""
    labels = LABELS[scenario]
    if scenario in LONGITUDINAL:
        means = {label: _initial_regimen(result.fits[label]) for label in labels}
    else:
        nuisance = result.nuisance
        means = {
            label: float(
                np.mean(nuisance.scaler.unscale_levels(nuisance.outcome.arms[float(index)]))
            )
            for index, label in enumerate(labels)
        }
    kind = "regimen" if scenario in LONGITUDINAL else "policy"
    out = {f"ey_{kind}[{label}]": value for label, value in means.items()}
    for label in labels:
        if label != REFERENCE:
            out[f"ate_{kind}[{label} vs {REFERENCE}]"] = means[label] - means[REFERENCE]
    return out


def _initial_regimen(regimen_fit: Any) -> float:
    """The untargeted plug-in of a regimen: the first node's initial carried value."""
    first = regimen_fit.steps[0]
    if first.initial_by_arm is None:
        return float(np.mean(first.initial))
    return float(np.mean(np.sum(regimen_fit.policy[0] * first.initial_by_arm, axis=1)))


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    initials = initial_estimates(result, scenario)
    rows = []
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": initials[name],
            }
        )
    return rows


def pairing_columns(result: Any, scenario: str) -> dict[str, np.ndarray]:
    """The shifted components and the per-node ratios this fit used, for the comparator.

    ``shift__<plan>__<node>__<component>`` holds the value the plan assigns to a component,
    and ``ratio__<plan>__<node>`` the ratio at the node: ``PolicySet.policy_ratio`` of a point
    fit, or :attr:`~cleverly.longitudinal.RegimenFit.node_ratio` of a longitudinal one.  The
    ordered classifier scenario writes its ratio too.  ``lmtp`` does not read it there, and the
    parity check does.
    """
    law = LAWS[scenario]
    labels = LABELS[scenario]
    out: dict[str, np.ndarray] = {}
    if scenario in LONGITUDINAL:
        plans = regimens(law, labels)
        for label in labels:
            fitted = result.fits[label]
            assert fitted.node_ratio is not None
            for time, columns in enumerate(NODES[law], start=1):
                _, values, _ = _resolve_mtp_node(plans[label], result.data, time)
                assert values is not None and len(values) == 1
                assigned = np.asarray(values[0], dtype=float)
                for k, column in enumerate(columns):
                    out[f"shift__{slug(label)}__{time}__{column}"] = assigned[:, k]
                out[f"ratio__{slug(label)}__{time}"] = np.asarray(
                    fitted.node_ratio[:, time - 1], dtype=float
                )
        return out
    evaluated = result.nuisance.policies
    columns = NODES[law][0]
    for index, label in enumerate(labels):
        shifted = np.asarray(evaluated.shifted[:, index], dtype=float)
        if law == "categorical":
            # The joint code of each row, decoded to its components.
            levels = np.asarray(result.data.treatment_levels, dtype=float)
            assigned = levels[np.rint(shifted).astype(int)]
        else:
            assigned = shifted
        for k, column in enumerate(columns):
            out[f"shift__{slug(label)}__1__{column}"] = assigned[:, k]
        out[f"ratio__{slug(label)}__1"] = np.asarray(evaluated.policy_ratio[:, index], dtype=float)
    return out


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit(frame, scenario)
    sample = frame.copy()
    for name, values in pairing_columns(result, scenario).items():
        sample[name] = values
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", np.zeros(len(sample), dtype=int))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, result_rows(result, truth, scenario, replicate, len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    # Replicate-major, so every scenario of one replicate is contiguous in the sample archive
    # and the comparator's runner reads them as one group.  The laws differ in their columns,
    # so the archive is the union of the columns, with the absent ones empty.
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in LAWS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


__all__ = ["STUDY", "draw_and_fit", "fit"]
