"""Registered multi-arm selector C-TMLE evidence with an explicit empty R comparison.

The fit is not cross-fitted, and it still draws a split: the selector scores its candidate
path over five *selection* folds.  That split reads the treatment unless the fit says
otherwise, which makes a candidate's cross-validated loss a function of the rows it is
scored on.  :func:`fit_cleverly` therefore passes ``stratify_folds="none"`` and
:func:`assert_unstratified_selection` reads the realized partition back off every result.

The binary selector row states the same rule in its own module rather than sharing this
one.  A shared helper would put every module the other study reaches into this study's
manifest, and the two rows would then record each other's sources as their own provenance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from cleverly.learners.crossfit import RANDOM_PARTITION_GENERATOR
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 800
PRIMARY_N = 1500
SEED = 20260829
SELECTION_FOLDS = 5

#: What every split this study draws is held to.  ``"none"`` is passed explicitly rather
#: than left to the estimator's default, so this study's declaration does not move when that
#: default does.
STRATIFY_FOLDS = "none"
SCENARIOS = {
    "multi_arm_selector_greedy": multi_arm_common.ALL_ESTIMANDS,
    "multi_arm_selector_ordered": multi_arm_common.ALL_ESTIMANDS,
    "multi_arm_selector_discrete": multi_arm_common.ALL_ESTIMANDS,
}

STUDY = StudyRecord(
    name="selector-based multi-arm C-TMLE",
    slug="canonical-multi-arm-ctmle-selector",
    artifacts=ROOT / "tests" / "canonical" / "multi_arm_ctmle_selector",
    document="docs/technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md",
    anchor="selector-based-multi-arm-c-tmle",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261003,
    margins=Margins(),
    implementation="cleverly-multi-arm-ctmle-selector",
    reference=None,
    publication_policy="reporting",
    modules=(
        "tests/studies/multi_arm_common.py",
        "tests/studies/canonical_multi_arm_ctmle_selector.py",
        "tests/studies/multi_arm_ctmle_selector_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_multi_arm_ctmle_selector",
    properties_module="tests.studies.multi_arm_ctmle_selector_properties",
    property_cells={
        "selector_necessity": ("greedy", "ordered", "discrete", "empty_control"),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
    },
)

REFERENCE_METADATA: dict[str, str] = {}
CONFIGURATION = {
    "strategies": ["greedy", "ordered", "discrete"],
    "outcome_family": "binomial",
    "treatment_levels": list(multi_arm_common.LABELS),
    "reference": multi_arm_common.REFERENCE,
    "cross_fit": False,
    "selection_folds": SELECTION_FOLDS,
    "selection_inner_folds": 2,
    "penalty": False,
    "stratify_folds": STRATIFY_FOLDS,
    "q_bounds": (
        "none anywhere in this row; the outcome is binary on the primary law and on every "
        "property law, so the scaler is already the identity"
    ),
    "folds": (
        f"unstratified {SELECTION_FOLDS}-fold selection assignments drawn from the "
        "estimator's own seed; no outer cross-fitting on the primary, and unstratified "
        "five-fold outer assignments in the property cells"
    ),
    "comparison_scope": "none; R ctmle 0.1.2 is binary-treatment only",
}


def draw_from_seed(scenario: str, n: int, seed: int):  # type: ignore[no-untyped-def]
    return multi_arm_common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int):  # type: ignore[no-untyped-def]
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _strategy(scenario: str) -> tuple[str, dict[str, Any]]:
    covariates = ("W1", "W2", "W3")
    if scenario.endswith("ordered"):
        return "ordered", {"ordering": covariates}
    if scenario.endswith("discrete"):
        return "discrete", {"candidates": ((), ("W1",), ("W1", "W2"), covariates)}
    return "greedy", {}


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    strategy, options = _strategy(scenario)
    result = (
        CTMLE(
            strategy=strategy,
            outcome_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            cross_fit=False,
            selection_folds=SELECTION_FOLDS,
            selection_inner_folds=2,
            penalty=False,
            estimands=("ey", "ate", "rr", "or"),
            ctmle_estimand="ate",
            reference=multi_arm_common.REFERENCE,
            simultaneous=False,
            g_bounds=multi_arm_common.G_BOUNDS,
            # The outcome is binary here, so no ``q_bounds``: the scaler is already the
            # identity and ``TMLE._scaler`` refuses a second declaration of it.  The split
            # is the declaration this fit needs.
            stratify_folds=STRATIFY_FOLDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )
    assert_unstratified_selection(result)
    return result


def assert_unstratified_selection(result: Any) -> None:
    """Refuse a fit whose selection split was not the declared unstratified one.

    ``stratify_folds="none"`` is a keyword the caller passes, and the realized partition is
    what the fit did with it.  A balanced split carries no origin at all, because it comes
    from scikit-learn rather than from the package's own generator, so the absence of one is
    itself the finding.  Reading it back is what makes the published ``folds`` line in
    ``CONFIGURATION`` a checked statement rather than a description.

    Parameters
    ----------
    result : Any
        A fitted single-parameter C-TMLE result carrying a selector path.

    Raises
    ------
    RuntimeError
        When the selection folds were not drawn by
        :func:`~cleverly.learners.random_partition` as a plain v-fold split.
    """
    origin = result.extra["ctmle"].folds.origin
    if origin is None:
        raise RuntimeError(
            "the selection folds carry no generator record, which is what a split balanced "
            "on the treatment or the outcome leaves behind. This study draws them with "
            f"stratify_folds={STRATIFY_FOLDS!r}"
        )
    if origin.generator != RANDOM_PARTITION_GENERATOR or origin.scheme != "vfold":
        raise RuntimeError(
            f"the selection folds came from generator={origin.generator!r} as "
            f"scheme={origin.scheme!r}, and this study's rows are evidence about a plain "
            f"v-fold split of iid rows drawn by {RANDOM_PARTITION_GENERATOR!r}"
        )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: dict[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return multi_arm_common.cleverly_rows(STUDY, fit_cleverly, frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    rows = multi_arm_common.draw_and_fit(
        STUDY,
        fit_cleverly,
        replicates=replicates,
        n=n,
        n_jobs=n_jobs,
        include_samples=False,
    )
    assert isinstance(rows, pd.DataFrame)
    return rows
