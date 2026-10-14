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
* ``PRIMARY_REPLICATES`` comes from the design rule in
  ``tests/unit/test_ltmle_repeated_method_study.py``: the smallest multiple of 400 at which a
  cell whose true standardized bias equals the paired-difference margin, 0.15, passes the
  bias gate with probability at least 0.90.  The rule gives 1,600.
* The paired comparison reads the registered :class:`Margins` defaults.
* Publication policy ``gated``.  The registry takes one policy per study, and the declared
  reading gates the paired cells and the positive property cells.  A red cell is routed by
  the rule in ``ltmle_repeated_properties``.  A red cell that the rule reads as
  finite-sample re-registers the study under ``reporting`` with the owner
  ``X32-finite-sample``, and the run is not repeated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator

from cleverly.datasets import REPEATED_RULE_LABEL, make_longitudinal_repeated
from cleverly.longitudinal import LTMLE, DynamicRegimen, Repeated
from cleverly.utils.parallel import map_parallel
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
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20263201
RESAMPLING_SEED = 2026320101
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

#: Property-cell labels of the reported parameters at the last horizon.  The SE-ratio rule
#: gates these and only these.
GATED_LABELS = ("never_t3", "always_t3", "rule_t3", "always_vs_never_t3", "rule_vs_never_t3")

#: The earlier horizons.  Their calibration is reported under ``horizon_calibration``.
REPORTED_LABELS = (
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
            *(
                f"{label}__{control}"
                for label in GATED_LABELS
                for control in ("shrunken_se_control", "noise_control")
            ),
        ),
        "horizon_calibration": tuple(f"{label}__estimated_mechanisms" for label in REPORTED_LABELS),
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

    A width this does not recognise raises.  A permutation within a width would not, so
    ``tests/unit/test_ltmle_repeated_method_study.py`` checks every node against the
    generating probabilities and checks that a column swap moves them.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownRepeatedMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        width = matrix.shape[1]
        column = matrix.T
        if self.kind == "treatment" and width == 1:
            index = -0.2 + 0.6 * column[0]
        elif self.kind == "treatment" and width == 4:
            index = -0.2 + 0.4 * column[0] + 0.5 * column[2] - 1.4 * column[1] + 0.6 * column[3]
        elif self.kind == "treatment" and width == 7:
            index = -0.2 + 0.4 * column[0] + 0.5 * column[4] - 1.4 * column[3] + 0.6 * column[6]
        elif self.kind == "censoring" and width == 2:
            index = 2.0 + 0.3 * column[0] - 0.3 * column[1]
        elif self.kind == "censoring" and width == 5:
            index = 2.2 + 0.3 * column[2] - 1.0 * column[1] + 0.2 * column[4]
        elif self.kind == "censoring" and width == 8:
            index = 2.2 + 0.3 * column[4] - 1.0 * column[3] + 0.2 * column[7]
        else:  # pragma: no cover - a changed longitudinal design is a study-contract failure
            raise ValueError(f"unexpected {self.kind} mechanism design {matrix.shape}")
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


def horizon_initials(
    result: Any, mean_names: Sequence[str], contrast_names: Sequence[str]
) -> dict[str, float]:
    """The untargeted plug-in behind each mean at a horizon, and each contrast of two.

    ``fits[f"{label} @ t={h}"].steps[0].initial`` is the first node's regression of the
    already targeted second node, which is what ``initial_estimate`` means on both sides.
    """
    means: dict[str, float] = {}
    for name in mean_names:
        inner = name[len("ey_regimen[") : -1]
        label, horizon = inner.rsplit(" @ t=", 1)
        means[name] = float(np.mean(result.fits[f"{label} @ t={int(horizon)}"].steps[0].initial))
    for name in contrast_names:
        comparison, horizon = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        means[name] = (
            means[f"ey_regimen[{left} @ t={horizon}]"] - means[f"ey_regimen[{right} @ t={horizon}]"]
        )
    return means


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    result = fit_cleverly(frame)
    return regimen_rows(
        STUDY,
        result,
        truth,
        horizon_initials(result, MEAN_NAMES, CONTRAST_NAMES),
        ESTIMANDS,
        scenario,
        replicate,
        n=len(frame),
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths_rows, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths_frame = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths_frame, estimates.loc[:, list(REPLICATE_COLUMNS)]
