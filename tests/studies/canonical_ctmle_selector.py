"""Canonical selector-based C-TMLE evidence against R ``ctmle``.

The comparison is deliberately bounded to the common unpenalized construction.  Cleverly's
published penalty is validated independently because it follows the paper equation rather than
the implementation-specific adjustment in R ``ctmle``.

The fit is not cross-fitted, and it still draws a split: the selector scores its candidate
path over five *selection* folds.  That split reads the treatment unless the fit says
otherwise, which makes a candidate's cross-validated loss a function of the rows it is
scored on.  :func:`fit_cleverly` therefore passes ``stratify_folds="none"`` and
:func:`assert_unstratified_selection` reads the realized partition back off every result.
R selects against the same assignment, carried over in the ``selection_fold`` column, so
the two sides keep comparing one split rather than two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly._typing import EstimandName
from cleverly.estimators import CTMLE
from cleverly.learners.crossfit import RANDOM_PARTITION_GENERATOR
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import Q_BOUNDS, STRATIFY_FOLDS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

CTMLE_COMMIT = "18de559f47dc1286617350a0668391e80e1dbf7c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 800
PRIMARY_N = 2000
SEED = 20240822
G_BOUNDS = (0.025, 0.975)
SELECTION_FOLDS = 5

SCENARIO_ESTIMANDS: Mapping[str, tuple[EstimandName, ...]] = {
    "binary_greedy": ("ate",),
    "binary_ordered": ("ate",),
    "binary_discrete": ("ate",),
}

PROPERTY_CELLS = {
    "double_robustness": (
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
    ),
    "selector_necessity": ("collaborative", "empty_control"),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "interval_calibration": ("correctly_specified",),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
}

STUDY = StudyRecord(
    name="selector-based point-treatment C-TMLE",
    slug="canonical-ctmle-selector",
    artifacts=ROOT / "tests" / "canonical" / "ctmle_selector",
    document="docs/technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md",
    anchor="selector-based-point-treatment-c-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-ctmle-selector",
    reference="r-ctmle",
    # Reporting rather than gated, declared before the run that measured the red cells.
    # Two property cells fail on the bounded laws under unstratified folds, and every
    # margin here is interval-shaped, so a larger budget makes a verdict easier: raising
    # one after seeing a failure would buy a pass rather than earn it.  The complete
    # result is published instead, both red cells and their intervals, and the evidence
    # page and the validation grid name them.  ``multi-arm-ctmle-selector`` already
    # publishes the same two families red under the same policy.
    publication_policy="reporting",
    modules=(
        "tests/studies/canonical_ctmle_selector.py",
        "tests/studies/ctmle_selector_properties.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/bounded_cv_laws.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ctmle_selector",
    properties_module="tests.studies.ctmle_selector_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "ctmle_commit": CTMLE_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "cross_fit": False,
    "simultaneous_intervals": False,
    "selection_folds": SELECTION_FOLDS,
    "selection_inner_folds": 2,
    "penalty": False,
    "g_bounds": list(G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "q_bounds": (
        "none on the binary primary law, whose scaler is already the identity; "
        f"declared {list(Q_BOUNDS)} on the bounded property laws, whose outcome is a "
        "proportion"
    ),
    "folds": (
        f"unstratified {SELECTION_FOLDS}-fold selection assignments drawn from the "
        "estimator's own seed; no outer cross-fitting"
    ),
    "comparison_scope": "binary ATE; continuous outcomes are assessed independently",
}


#: All three scenarios are the same binary law.  What differs between them is the selector
#: strategy, not the process -- the distinct names exist so each strategy draws its own
#: ``replicate_seed`` stream rather than three verdicts being one draw reported three times.
LAW = "binary"


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw directly from a supplied seed for the manifest-seed audit."""
    from tests.studies.canonical_tmle import scenario_dgp, truth_for

    del scenario
    dgp = scenario_dgp(LAW)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


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
            estimands=("ate",),
            ctmle_estimand="ate",
            simultaneous=False,
            g_bounds=G_BOUNDS,
            # The outcome is binary here, so no ``q_bounds``: the scaler is already the
            # identity and :meth:`~cleverly.estimators.TMLE._scaler` refuses a second
            # declaration of it.  The split is the declaration this fit needs.
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
    from scikit-learn rather than from the package's own generator, so the absence of one
    is itself the finding.  Reading it back is what makes the published ``folds`` line in
    ``CONFIGURATION`` a checked statement rather than a description, and this row needs
    that: the same assignment is handed to R, and a split that quietly started reading the
    treatment again would move both sides together and leave every paired gate green.

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
            "the selection folds carry no generator record, which is what a split "
            "balanced on the treatment or the outcome leaves behind. This study draws "
            f"them with stratify_folds={STRATIFY_FOLDS!r}"
        )
    if origin.generator != RANDOM_PARTITION_GENERATOR or origin.scheme != "vfold":
        raise RuntimeError(
            f"the selection folds came from generator={origin.generator!r} as "
            f"scheme={origin.scheme!r}, and this study's rows are evidence about a plain "
            f"v-fold split of iid rows drawn by {RANDOM_PARTITION_GENERATOR!r}"
        )


def _rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    n: int,
) -> list[dict[str, Any]]:
    estimate = result["ate"]
    reference = float(truth["ate"])
    low, high = estimate.ci
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": "ate",
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
        }
    ]


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return _rows_from_result(fit_cleverly(frame, scenario), truth, scenario, replicate, len(frame))


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    sample = frame.copy()
    # The split this fit actually scored its candidates over, taken off the fit rather
    # than rebuilt here from the same rule.  R selects against the same partition, so a
    # reconstruction that silently stopped matching would move the reference's answer
    # while every gate in the study kept passing.
    sample.insert(0, "selection_fold", result.extra["ctmle"].folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return sample, truth_row, _rows_from_result(result, truth, scenario, replicate, len(frame))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, records in outcomes for row in records])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]
