"""A point vector treatment with a dose, declared with ``delta=``, ``intermediate=`` or ``strata=``.

A vector treatment adds its component columns to the missingness, intermediate and outcome
designs, and a policy fit reads each mechanism at the values the policy assigns.  Nothing
else changes: the clever covariate divides the policy ratio by the missingness and
intermediate mechanisms, as on a scalar dose (``scope-and-refusals.md``, the policy rows),
and baseline strata give one score block per stratum.

The oracle is ``tests/discrete_law_vector_coarsened.py``: the point mixed law of
``tests/discrete_law_policy_point.py`` with a recorded-outcome indicator and an intermediate,
both of which read both components.  With saturated learners every exact fit equals the
functional and its complex-step Gateaux derivative, on both ratio routes.  W8 misspecifies
the outcome regression, so the targeting step carries the mechanisms at the assigned values.
The mutation controls break one line each and ask a witness to move by more than
``CAUGHT``.  The measured deviations at the time of writing are in each docstring.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import tests.discrete_law_vector_coarsened as law
from cleverly.data import CausalData
from cleverly.estimators import TMLE, _nuisance
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from .. import longitudinal_mtp as mtp
from ._source_mutation import mutate
from .test_point_vector_treatment import joint

#: The mixed laws put every dose on a grid, so R-b4 would refuse each fit; the
#: fixture turns that refusal off and leaves every other check on (X40 step H).
pytestmark = [
    pytest.mark.xdist_group("point_vector_coarsened"),
    pytest.mark.usefixtures("exact_grid_law"),
]

EXACT = 1e-10
CAUGHT = 1e-4
LABELS = ("natural", "dose up", "flag gated", "dose gated", "random")
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
#: The module, which the function ``cleverly.estimators.tmle`` shadows as an attribute.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: Every composition the witnesses run, as ``CausalData.from_frame`` keywords.
COMPOSITIONS: dict[str, dict[str, Any]] = {
    "delta": {"delta": "Delta"},
    "delta + intermediate": {"delta": "Delta", "intermediate": "Z"},
    "delta + strata": {"delta": "Delta", "strata": "W"},
    "delta + intermediate + strata": {"delta": "Delta", "intermediate": "Z", "strata": "W"},
}


def coarse_data(**declared: Any) -> CausalData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(
            law.frame(),
            outcome="Y",
            treatment=["F", "D"],
            covariates=["W"],
            continuous_treatment=["D"],
            **declared,
        )


def coarse_fit(
    ratio: str, declared: dict[str, Any], labels: Sequence[str] = LABELS, **overrides: Any
) -> Any:
    """A saturated fit of the coarsened law: a result set keyed by the intermediate level."""
    settings: dict[str, Any] = {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellMeans() if ratio == "classifier" else CellProbabilities(),
        "missingness_learner": CellMeans(),
        "intermediate_learner": CellMeans(),
        "cross_fit": False,
        "ratio": ratio,
        "simultaneous": False,
        "g_bounds": NO_TRUNCATION,
        "nuisance_bound": 1e-8,
        **overrides,
    }
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return TMLE(policies=[joint(label) for label in labels], **settings).fit(
            coarse_data(**declared)
        )


def _names(label: str, strata: bool) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = [(f"ey_policy[{label}]", None)]
    if strata:
        out.extend((f"ey_policy[{label}][W={float(w)}]", w) for w in (0, 1))
    return out


def deviation(results: Any, declared: dict[str, Any], labels: Sequence[str] = LABELS) -> float:
    """The largest error of an estimate or of its influence curve against the oracle."""
    rows = law.first_row_of()
    worst = 0.0
    for level, result in results.items():
        oracle_level = None if level is None else int(level)
        for label in labels:
            for name, stratum in _names(label, "strata" in declared):
                estimate = result.estimates[name]
                oracle = f"ey_policy[{label}]"
                worst = max(worst, abs(estimate.psi - law.truth(oracle, oracle_level, stratum)))
                curve = np.asarray(estimate.influence_curve)[rows]
                eif = law.eif(oracle, oracle_level, stratum)
                worst = max(worst, float(np.max(np.abs(curve - eif))))
    return worst


def targeted_error(ratio: str, declared: dict[str, Any]) -> float:
    """W8: the largest error of a targeted estimate with a misspecified outcome regression."""
    from sklearn.linear_model import LogisticRegression

    results = coarse_fit(ratio, declared, outcome_learner=LogisticRegression())
    worst = 0.0
    for level, result in results.items():
        oracle_level = None if level is None else int(level)
        for label in LABELS:
            for name, stratum in _names(label, "strata" in declared):
                truth = law.truth(f"ey_policy[{label}]", oracle_level, stratum)
                worst = max(worst, abs(result.psi(name) - truth))
    return worst


# ------------------------------------------------------------------ witnesses


def test_the_law_sees_both_components_and_every_composition() -> None:
    """Both mechanisms move with each component; levels and complete cases differ."""
    margins = law.check_nondegenerate()
    assert min(margins.values()) > 1e-2


@pytest.mark.parametrize("ratio", ["classifier", "density"])
@pytest.mark.parametrize("composition", sorted(COMPOSITIONS))
def test_every_policy_mean_and_influence_curve_is_exact(ratio: str, composition: str) -> None:
    """Estimate and influence curve equal the oracle, per level and per stratum.

    Measured: at most 3e-14 on the classifier route and 3e-11 on the density route.
    """
    declared = COMPOSITIONS[composition]
    assert deviation(coarse_fit(ratio, declared), declared) < EXACT


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_w8_targeting_reaches_the_truth_with_a_misspecified_outcome(ratio: str) -> None:
    """The fluctuation moves and lands on every truth, so each mechanism enters it rightly."""
    declared = COMPOSITIONS["delta + intermediate + strata"]
    assert targeted_error(ratio, declared) < EXACT


def test_the_fluctuation_moves_on_the_w8_fit() -> None:
    from sklearn.linear_model import LogisticRegression

    declared = COMPOSITIONS["delta + intermediate"]
    results = coarse_fit("classifier", declared, outcome_learner=LogisticRegression())
    for result in results.values():
        epsilons = [abs(float(np.max(np.abs(f.epsilon)))) for f in result.fluctuations.values()]
        assert max(epsilons) > 1e-3


def test_the_complete_case_reading_is_not_the_parameter() -> None:
    """A fit that dropped the rows with no outcome would land on another number."""
    results = coarse_fit("classifier", COMPOSITIONS["delta"])
    estimate = results[None].psi("ey_policy[flag gated]")
    assert abs(estimate - law.complete_case_truth("ey_policy[flag gated]")) > 1e-2


# ------------------------------------------------------------------ mutation controls


def _mutate_fit(monkeypatch: pytest.MonkeyPatch, edits: list[tuple[str, str]]) -> None:
    mutate(monkeypatch, _nuisance, _nuisance, "fit_nuisances", edits)
    # The estimator holds its own reference to the function.
    monkeypatch.setattr(tmle_module, "fit_nuisances", _nuisance.fit_nuisances)


def _caught(declared: dict[str, Any]) -> float:
    gap = max(
        deviation(coarse_fit("classifier", declared), declared),
        targeted_error("classifier", declared),
    )
    assert gap > CAUGHT
    return gap


def test_mc1_a_missingness_design_without_the_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response model drops the flag column.  Measured: exact fit 3.0 (curve), W8 0.027."""
    _mutate_fit(
        monkeypatch,
        [
            (
                "            missingness_design,\n            data.observed.astype(float),",
                "            missingness_design[:, 1:],\n            data.observed.astype(float),",
            ),
            (
                "predict_designs=_mechanism_designs(data, arms, policy_set, missingness_design),",
                "predict_designs={key: value[:, 1:] for key, value in _mechanism_designs("
                "data, arms, policy_set, missingness_design).items()},",
            ),
        ],
    )
    _caught(COMPOSITIONS["delta"])


def test_mc2_an_intermediate_design_without_the_dose_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intermediate model drops the dose column.  Measured: exact fit 14 (curve), W8 0.024."""
    _mutate_fit(
        monkeypatch,
        [
            (
                "            intermediate_design,\n            data.intermediate,",
                "            np.delete(intermediate_design, 1, axis=1),\n"
                "            data.intermediate,",
            ),
            (
                "predict_designs=_mechanism_designs(data, arms, policy_set, intermediate_design),",
                "predict_designs={key: np.delete(value, 1, axis=1) for key, value in "
                "_mechanism_designs(data, arms, policy_set, intermediate_design).items()},",
            ),
        ],
    )
    _caught(COMPOSITIONS["delta + intermediate"])


def test_mc3_mechanisms_read_at_the_observed_values_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each mechanism is read at the observed treatment, not the assigned one.

    Measured: exact fit 7e-15 (the reported curve reads the observed block), W8 0.050.
    """
    mutate(
        monkeypatch,
        _nuisance,
        _nuisance,
        "_mechanism_designs",
        [
            (
                "designs[_arm_key(code)] = data.counterfactual_design(policy_set.shifted[:, index])",
                "designs[_arm_key(code)] = design",
            )
        ],
    )
    _caught(COMPOSITIONS["delta + intermediate"])


def test_the_unmutated_controls_are_exact() -> None:
    for name in ("delta", "delta + intermediate"):
        declared = COMPOSITIONS[name]
        assert deviation(coarse_fit("classifier", declared), declared) < EXACT
        assert targeted_error("classifier", declared) < EXACT
