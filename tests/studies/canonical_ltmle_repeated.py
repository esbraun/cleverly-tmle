"""Canonical repeated-outcome longitudinal TMLE evidence study (X32a).

The paired comparison gives cleverly and R ``ltmle`` the same realized censored panels of
:func:`cleverly.datasets.make_longitudinal_repeated`, the same three plans, the generating
treatment and censoring probabilities, and quasibinomial sequential regressions.  cleverly
fits the panel once with ``outcome=Repeated(["Y1", "Y2", "Y3"])``.  R fits the prefix
through ``Y_h`` once per horizon ``h`` with ``survivalOutcome = FALSE``, so its earlier
``Y_s`` are covariates, and its ``Qform`` at each block is the main-terms regression on
the history cleverly regresses on at that node.  The independent property study lives in
``ltmle_repeated_properties``.

Declared before any run
-----------------------

* Primary cells: ``ey_regimen`` for ``never``, ``always`` and :data:`RULE_LABEL` at
  ``h = 1, 2, 3``, and ``ate_regimen`` of the other two plans against ``never``.  The rule
  equals ``always`` at the first node, so its ``t = 1`` mean and contrast are exact
  duplicates of ``always``'s.  A fast test checks that identity, and the two duplicates are
  not primary cells.  Thirteen cells remain.
* Both sides receive the generating mechanisms (:class:`KnownRepeatedMechanism` here, a
  numeric ``gform`` in R) and ``g_bounds = G_BOUNDS``.  Both fit in sample (``n_folds = 1``)
  at ``n = 2,000``.
* The law has depth two.  ``W`` takes three values, and ``A_t``, ``C_t`` and ``Y_t`` read
  ``Y_{t-2}``.  The mechanisms therefore have three design widths, and
  :class:`KnownRepeatedMechanism` builds each index from the generator's coefficient tables
  by column name.
* ``PRIMARY_REPLICATES`` comes from the design rule in
  ``tests/unit/test_ltmle_repeated_method_study.py``: the smallest multiple of 400 such that
  (a) a cell whose true standardized bias equals the paired-difference margin, 0.15, passes
  the bias gate with probability at least 0.90, (b) the coverage-floor gate passes with
  probability at least 0.99 per cell at coverage 0.93, and (c) the product of the pass
  probabilities over the 26 coverage tests (13 cells for each implementation) is at least
  0.95 when each cell sits at the one-sided 80% lower bound of its pilot coverage.  The rule
  and the pilot that sized it are recorded in ``PRIMARY_PILOT`` below.
* The cleverly rows publish ``initial_estimate = nan``.  A single-fold fit exposes no
  untargeted plug-in: ``steps[0].initial`` is the first node's regression of the already
  targeted chain, a different quantity from the R column.  R keeps ``initial = gcomp``, the
  sequential regression with no fluctuation, and the parity rule in
  ``tests/unit/test_ltmle_repeated_method_study.py`` reads R's targeting move as its
  denominator.
* The paired comparison reads the registered :class:`Margins` defaults.
* Publication policy ``gated``.  The registry takes one policy per study, and the declared
  reading gates the paired cells and the positive property cells.  A red cell is routed by
  the rule in ``ltmle_repeated_properties``.  A red cell is diagnosed and fixed before a
  fresh declaration on fresh seeds.  No red cell is deferred.

Declaration history
-------------------

* First declaration (5728c3ce, seed ``20263201``).  Its rows informed the revisions that
  ``ltmle_repeated_properties`` records.
* Second declaration (seed ``20263202``).  It re-sized the calibration arms and the N1
  control.  A third run of the same declaration (52cb8596) added the ``gcomp`` initial
  estimate on the R side.  Every gate was green.
* Fourth declaration (:data:`SEED`, ``20263204``).  A design review of the third run found
  seven defects, which the redesign fixes before any new draw:

  1. The paired gate cannot tell a plug-in from a targeted fit, because the untargeted
     sequential regression is nearly unbiased on that law.  The study now carries a
     ``targeting_necessity`` family and a parity rule on R's own targeting move.
  2. The law was first-order Markov, so no check read the history of ``Q`` beyond one node.
     The law now has depth two, and ``full_history_necessity`` separates a full-history
     regression from a Markov one.
  3. Binary ``W`` saturated the first-node regression, so the first fluctuation was zero.
     ``W`` takes three values.
  4. The primary coverage gate passed with probability about 0.56 at 1,600 replications,
     because the rule's followers were rare and the weights large.  Positivity is raised
     and the sizing rule includes coverage.
  5. The 28 calibration cells passed jointly with probability about 0.93, and the continuous
     arm was never piloted at ``n = 8,000``.  The sizing rule is the family-wise product
     and every arm is piloted.
  6. The reversed-rule and both-wrong controls did not read their own limits.  Both now
     read an exact limit, and every contrast is at least 2.5 influence-curve spreads from
     zero at the primary ``n``.
  7. The page made false statements about the initial estimates, the unread margins, and
     positivity.  They are corrected with this declaration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator

from cleverly.datasets import REPEATED_RULE_LABEL, make_longitudinal_repeated
from cleverly.datasets.longitudinal import _REPEATED_A, _REPEATED_A1, _REPEATED_C, _REPEATED_C1
from cleverly.longitudinal import LTMLE, DynamicRegimen, Repeated
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import (
    G_BOUNDS,
    LTMLE_SOURCE_COMMIT,
    LTMLE_TARBALL_SHA256,
    LTMLE_VERSION,
    R_BASE_IMAGE,
    QuasiBinomialGLM,
    regimen_rows,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 3_200
PRIMARY_N = 2_000
SEED = 20263204
RESAMPLING_SEED = 2026320401
SCENARIO = "censored_repeated_outcome"
HORIZONS = (1, 2, 3)
TIMES = 3

RULE_LABEL = REPEATED_RULE_LABEL


def treat_after(previous: str) -> Any:
    """The rule's node: treat exactly when the outcome ``previous`` was positive.

    A function of the history frame a rule is handed, which holds ``Y_{t-1}`` because a
    repeated outcome puts it there.
    """

    def node(history: Any) -> np.ndarray:
        return (np.asarray(history[previous], dtype=float) > 0.0).astype(int)

    return node


REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    RULE_LABEL: DynamicRegimen(
        RULE_LABEL, (1, treat_after("Y1"), treat_after("Y2")), rule_kind="known"
    ),
}
REFERENCE = "never"

#: The fit's roles.  ``Y_s`` is prepended to block ``s + 1`` by the container.
COLUMNS: dict[str, Any] = {
    "outcome": Repeated([f"Y{t}" for t in range(1, TIMES + 1)]),
    "treatment": [f"A{t}" for t in range(1, TIMES + 1)],
    "baseline": ["W"],
    "time_varying": [[], ["L2"], ["L3"]],
    "censoring": [f"C{t}" for t in range(1, TIMES + 1)],
}

MEAN_NAMES = (
    *(f"ey_regimen[never @ t={h}]" for h in HORIZONS),
    *(f"ey_regimen[always @ t={h}]" for h in HORIZONS),
    *(f"ey_regimen[{RULE_LABEL} @ t={h}]" for h in HORIZONS[1:]),
)
CONTRAST_NAMES = (
    *(f"ate_regimen[always vs never @ t={h}]" for h in HORIZONS),
    *(f"ate_regimen[{RULE_LABEL} vs never @ t={h}]" for h in HORIZONS[1:]),
)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)

#: The rule's exact duplicates of ``always`` at the first node, checked by a fast test.
DUPLICATES = (
    f"ey_regimen[{RULE_LABEL} @ t=1]",
    f"ate_regimen[{RULE_LABEL} vs never @ t=1]",
)

#: Property-cell labels of the reported parameters at the last horizon.  Every property
#: family reads these.
GATED_LABELS = ("never_t3", "always_t3", "rule_t3", "always_vs_never_t3", "rule_vs_never_t3")

#: The earlier horizons.  The C1 calibration arm gates these beside :data:`GATED_LABELS`.
EARLIER_LABELS = (
    "never_t1",
    "never_t2",
    "always_t1",
    "always_t2",
    "rule_t2",
    "always_vs_never_t1",
    "always_vs_never_t2",
    "rule_vs_never_t2",
)


def label_estimand(label: str) -> str:
    """The estimand a property label names."""
    stem, horizon = label.rsplit("_t", 1)
    plan = {"never": "never", "always": "always", "rule": RULE_LABEL}
    if "_vs_" in stem:
        left, right = stem.split("_vs_")
        return f"ate_regimen[{plan[left]} vs {plan[right]} @ t={horizon}]"
    return f"ey_regimen[{plan[stem]} @ t={horizon}]"


def _property_cells() -> dict[str, tuple[str, ...]]:
    """The property cells ``ltmle_repeated_properties`` publishes, by family."""
    calibration = (
        "estimated_mechanisms",
        "crossfit_glm",
        "crossfit_boosted",
        "continuous_gaussian",
    )
    return {
        "interval_calibration": (
            *(f"{label}__{arm}" for label in GATED_LABELS for arm in calibration),
            *(f"{label}__estimated_mechanisms" for label in EARLIER_LABELS),
            *(
                f"{label}__{control}"
                for label in GATED_LABELS
                for control in ("shrunken_se_control", "noise_control")
            ),
        ),
        "small_sample_calibration": tuple(
            f"{label}__{arm}"
            for label in GATED_LABELS
            for arm in ("estimated_mechanisms", "continuous_gaussian")
        ),
        "double_robustness": (
            *(f"{label}__mechanism_correct" for label in GATED_LABELS),
            # The both-wrong limit is the follower mean given ``W``, exact for a static plan.
            *(f"{label}__both_wrong" for label in GATED_LABELS if not label.startswith("rule")),
        ),
        "simultaneous_coverage": tuple(
            f"all_reported__{kind}" for kind in ("simultaneous_band", "pointwise_joint_control")
        ),
        "rule_necessity": tuple(
            f"{label}__{arm}"
            for label in ("rule_t2", "rule_t3")
            for arm in ("declared", "reversed_rule")
        ),
        "repeated_outcome_necessity": tuple(
            f"{label}__{arm}"
            for label in ("never_t2", "never_t3", "always_t2", "always_t3", "rule_t2", "rule_t3")
            for arm in ("repeated", "absorbing_reading")
        ),
        "outcome_history_necessity": tuple(
            f"{label}__{arm}"
            for label in ("never_t3", "always_t3", "always_vs_never_t3")
            for arm in ("repeated", "history_free")
        ),
    }


STUDY = StudyRecord(
    name="repeated-outcome longitudinal TMLE",
    slug="longitudinal-repeated-outcome",
    outcome_kind="repeated",
    artifacts=ROOT / "tests" / "canonical" / "ltmle_repeated",
    document="docs/technical-reference/method-evidence/repeated-outcome-longitudinal-tmle.md",
    anchor="repeated-outcome-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="ltmle",
    modules=(
        "tests/studies/canonical_ltmle_repeated.py",
        "tests/studies/ltmle_repeated_properties.py",
        "tests/studies/ltmle_repeated_law.py",
        # ``QuasiBinomialGLM``, ``G_BOUNDS`` and ``regimen_rows`` come from there.
        "tests/studies/canonical_ltmle.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/datasets/longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_ltmle_repeated",
    properties_module="tests.studies.ltmle_repeated_properties",
    property_cells=_property_cells(),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

REFERENCE_METADATA = {
    "ltmle_version": LTMLE_VERSION,
    "ltmle_source_commit": LTMLE_SOURCE_COMMIT,
    "ltmle_tarball_sha256": LTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "ltmle per horizon on the prefix through Y_h, survivalOutcome=FALSE, main-terms Qform, "
        "numeric gform; the initial estimate is a second call with gcomp=TRUE. The cleverly "
        "rows publish no initial estimate"
    ),
}

CONFIGURATION = {
    "construction": "ordinary",
    "outcome_kind": "repeated",
    "horizon_mode": "all_prefixes",
    "r_survival_outcome": False,
    "cross_fit": False,
    "simultaneous_intervals": False,
    "variance_method": "ic",
    "stratify": True,
    "g_bounds": list(G_BOUNDS),
    "horizons": list(HORIZONS),
    "regimens": list(REGIMENS),
    "q_formulas": [
        "Q.kplus1 ~ W",
        "Q.kplus1 ~ W + Y1 + L2",
        "Q.kplus1 ~ W + Y1 + L2 + Y2 + L3",
    ],
}


#: Each known-mechanism design, by ``(kind, width)``: the generator's coefficient table, the
#: column of the design that supplies each of the table's terms, and the terms that read a
#: node before the first (zero by construction, so no column holds them).  The columns are
#: the layout :meth:`~cleverly.longitudinal.LongitudinalData.history_design` builds.
_MECHANISM_DESIGNS: dict[
    tuple[str, int], tuple[dict[str, float], dict[str, int], frozenset[str]]
] = {
    ("treatment", 1): (_REPEATED_A1, {"w": 0}, frozenset()),
    ("treatment", 4): (_REPEATED_A, {"w": 0, "y": 1, "l": 2, "a": 3}, frozenset({"y2"})),
    ("treatment", 7): (_REPEATED_A, {"w": 0, "y2": 1, "y": 3, "l": 4, "a": 6}, frozenset()),
    ("censoring", 2): (_REPEATED_C1, {"w": 0, "a": 1}, frozenset()),
    ("censoring", 5): (_REPEATED_C, {"y": 1, "l": 2, "a": 4}, frozenset({"y2"})),
    ("censoring", 8): (_REPEATED_C, {"y2": 1, "y": 3, "l": 4, "a": 7}, frozenset()),
}


class KnownRepeatedMechanism(BaseEstimator):
    """The generating treatment or censoring probabilities, keyed by design width.

    :meth:`~cleverly.longitudinal.LongitudinalData.history_design` builds a mechanism's
    design at node ``t`` as ``covariate_history(t)``, ``[W, Y1, L2, ..., Y_{t-1}, L_t]``,
    followed by one column per earlier arm, plus the current arm for a censoring model.  The
    widths are therefore distinct at every node of this panel:

    =====  ===================================  =====================================
    node   treatment design                     censoring design
    =====  ===================================  =====================================
    1      ``[W]``                              ``[W, A1]``
    2      ``[W, Y1, L2, A1]``                  ``[W, Y1, L2, A1, A2]``
    3      ``[W, Y1, L2, Y2, L3, A1, A2]``      ``[W, Y1, L2, Y2, L3, A1, A2, A3]``
    =====  ===================================  =====================================

    The index at each width is the generator's own coefficient table
    (``_REPEATED_A1``, ``_REPEATED_A``, ``_REPEATED_C1``, ``_REPEATED_C``) applied to the
    column that table's term names.  At node 3 the previous outcome ``y`` is ``Y2`` and the
    second-order outcome ``y2`` is ``Y1``.  At node 2 the second-order term reads ``Y_0 = 0``
    and contributes nothing.  A width this does not recognise raises.  A permutation within a
    width would not, so ``tests/unit/test_ltmle_repeated_method_study.py`` checks every node
    against the generating probabilities and checks that a column swap moves them.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownRepeatedMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        key = (self.kind, matrix.shape[1])
        if (
            key not in _MECHANISM_DESIGNS
        ):  # pragma: no cover - a changed design is a contract failure
            raise ValueError(f"unexpected {self.kind} mechanism design {matrix.shape}")
        table, columns, zero_terms = _MECHANISM_DESIGNS[key]
        read = set(table) - {"intercept"}
        if (
            read != set(columns) | zero_terms
        ):  # pragma: no cover - a changed table is a contract failure
            raise ValueError(f"{key} does not cover the terms {sorted(read)}")
        index = np.full(len(matrix), table["intercept"])
        for term, column in columns.items():
            index = index + table[term] * matrix[:, column]
        probability = expit(index)
        return np.column_stack([1.0 - probability, probability])


def truths() -> dict[str, float]:
    """Every primary parameter, from the generator's enumeration."""
    _, truth = make_longitudinal_repeated(n=1, seed=0, backend="pandas")
    return {name: float(truth[name]) for name in ESTIMANDS}


TRUTH = truths()


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, _ = make_longitudinal_repeated(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownRepeatedMechanism("treatment"),
        censoring_learner=KnownRepeatedMechanism("censoring"),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **COLUMNS)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    result = fit_cleverly(frame)
    # No initial estimate: a single-fold fit exposes no untargeted plug-in (see the module
    # docstring), so the column is ``nan`` for the subject, as in the point-treatment studies.
    return regimen_rows(
        STUDY,
        result,
        truth,
        dict.fromkeys(ESTIMANDS, math.nan),
        ESTIMANDS,
        scenario,
        replicate,
        n=len(frame),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's sample and truth rows, which need no fit, and no subject rows."""
    scenario, replicate, _ = key
    frame, truth = drawn
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths_rows, []


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truths_rows, _ = _reference_input(key, drawn)
    return sample, truths_rows, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    outcomes = replication.fit_primary(
        [(SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths_frame = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths_frame
    )
