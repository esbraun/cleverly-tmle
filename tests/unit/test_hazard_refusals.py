"""Every hazard and isotonic refusal, pinned by type and message before any learner (X14 H1-H11).

Each fit passes learners that raise on ``fit``, so a refusal that reached a learner would
fail with ``AssertionError`` rather than the refusal under test.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, MethodConfigurationError, RefusalReason
from cleverly.interventions import ModifiedPolicy, Shift
from cleverly.longitudinal import IPSI, LTMLE, SDR, SequentialGComputation, SequentialIPW
from cleverly.methods import (
    GComputationMethod,
    IPSIMethod,
    IPWMethod,
    SDRMethod,
    TMLEMethod,
)
from cleverly.msm import MSM
from tests.discrete_law_point_survival import CellMeans, end_of_study_law, survival_law


class _Refuse(CellMeans):
    def fit(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("a learner ran")


def _refusing() -> dict[str, object]:
    return {
        "outcome_learner": _Refuse(),
        "treatment_learner": _Refuse(),
        "censoring_learner": _Refuse(),
    }


def _msm() -> MSM:
    def design(label: object, horizon: int, frame: object) -> np.ndarray:
        return np.ones((len(frame), 1))  # type: ignore[arg-type]

    return MSM(design=design, terms=("intercept",), design_kind="known")


LAW = survival_law()


def _assert_gap(error: CapabilityError) -> None:
    assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert error.refusal.roadmap == "X70"


def test_h1_a_treatment_at_several_nodes_is_refused() -> None:
    message = (
        "survival_model='hazard' is built for one baseline treatment held over every node, "
        "and this design declares a treatment at 3 nodes. Hazard targeting for a treatment "
        "at several nodes is not built. X70 in docs/roadmap.md tracks this. LTMLE with "
        "survival_model='iterated_mean' estimates this curve, and isotonic=True makes it "
        "monotone. If one baseline treatment is held over every node, declare treatment='A'."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE({"a1": 1}, n_folds=1, survival_model="hazard", **_refusing()).fit(
            LAW.frame(), **LAW.fit_columns(held=False)
        )
    _assert_gap(raised.value)


def test_h2_a_covariate_after_baseline_is_refused() -> None:
    law = survival_law(with_l2=True)
    message = (
        "survival_model='hazard' models the event and censoring hazards given the baseline "
        "treatment and the baseline covariates. This fit declares covariates measured after "
        "baseline (L2), and hazard targeting with such covariates is not built: it needs a "
        "model and a fluctuation of each later covariate. X70 in docs/roadmap.md tracks this. "
        "LTMLE with survival_model='iterated_mean' adjusts for them, and isotonic=True makes "
        "its curve monotone."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE({"a1": 1}, n_folds=1, survival_model="hazard", **_refusing()).fit(
            law.frame(), **law.fit_columns()
        )
    _assert_gap(raised.value)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"survival_model": "hazard"},
            "survival_model='hazard' needs an event indicator at every node, and this fit "
            "declares one end-of-study outcome. A hazard is the chance of the event at a node "
            "among the units still at risk, so an outcome measured once has none.",
        ),
        (
            {"isotonic": True, "pseudo_learner": _Refuse()},
            "isotonic=True makes a curve over the horizons monotone, and an end-of-study fit "
            "reports one horizon.",
        ),
    ],
)
def test_h3_an_end_of_study_outcome_is_refused(settings: dict[str, Any], message: str) -> None:
    law = end_of_study_law()
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE({"a1": 1}, n_folds=1, **settings, **_refusing()).fit(law.frame(), **law.fit_columns())
    assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_h4_a_one_step_iterated_mean_fit_is_refused() -> None:
    message = (
        "targeting='one_step' requires survival_model='hazard'. The one-step walk moves one "
        "set of hazards toward the score equations of every horizon together (Cai and van der "
        "Laan 2020). A one-step update of the iterated-mean recursion is not built. Set "
        "survival_model='hazard' on a survival design with one baseline treatment, or keep "
        "targeting='iterative' and set isotonic=True for a monotone curve"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        LTMLE({"a1": 1}, targeting="one_step", **_refusing())


@pytest.mark.parametrize(
    ("settings", "option"),
    [
        ({"survival_model": "hazard"}, "survival_model='hazard'"),
        ({"isotonic": True}, "isotonic=True"),
    ],
)
def test_h5_a_working_model_is_refused(settings: dict[str, Any], option: str) -> None:
    message = (
        f"msm= with {option} is not built. A working model reports coefficients over regimen "
        "and horizon cells. X70 in docs/roadmap.md tracks this. Fit msm= with "
        "survival_model='iterated_mean' and isotonic=False, or report the curve and take "
        "contrasts."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE({"a1": 1}, msm=_msm(), **settings, **_refusing())
    _assert_gap(raised.value)


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_r_d1_a_ratio_trim_on_a_hazard_fit_is_refused(targeting: str) -> None:
    """The hazard path has no trim, so a fit that stamped one would report an untrimmed curve.

    Before the refusal, ``ratio_trim=0.5`` on ``make_point_survival(n=300, seed=3)`` gave a
    curve bit-identical to ``ratio_trim=None`` and still stamped the trimmed-ratio status.
    """
    message = (
        "ratio_trim= with survival_model='hazard' is not built. lmtp's trim acts on the density "
        "ratios of the iterated-mean recursion, and the hazard targeting has no trim of its "
        "clever covariates. X79 in docs/roadmap.md tracks this. Fit ratio_trim= with "
        "survival_model='iterated_mean', or fit survival_model='hazard' without ratio_trim= "
        "and bound the treatment mechanism with g_bounds=."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"a1": 1},
            survival_model="hazard",
            targeting=targeting,
            ratio_trim=0.5,
            **_refusing(),
        )
    assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert raised.value.refusal.roadmap == "X79"
    # The iterated-mean recursion keeps its trim.
    assert LTMLE({"a1": 1}, ratio_trim=0.5, **_refusing()).ratio_trim == 0.5


def test_h6_a_pseudo_learner_on_a_hazard_fit_is_refused() -> None:
    message = (
        "pseudo_learner= is not used with survival_model='hazard'. A hazard fit regresses the "
        "event indicator at each node and fits no pseudo-outcome. Pass the hazard learner as "
        "outcome_learner="
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        LTMLE({"a1": 1}, survival_model="hazard", pseudo_learner=_Refuse(), **_refusing())


def test_h7_a_modified_treatment_policy_is_refused() -> None:
    def up(a: Any, h: Any) -> Any:
        return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)

    policy = ModifiedPolicy("up", apply=up, policy_kind="known")
    message = (
        "regimen 'up' is a modified treatment policy, and survival_model='hazard' does not "
        "support one. Its plug-in evaluates the event and censoring hazards at the shifted "
        "dose, and that construction is not built. X70 in docs/roadmap.md tracks this. Use "
        "survival_model='iterated_mean', which supports a modified treatment policy at "
        "baseline."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE({"up": policy}, n_folds=1, survival_model="hazard", **_refusing()).fit(
            LAW.frame(), **LAW.fit_columns()
        )
    _assert_gap(raised.value)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"targeting": "newton"}, "targeting must be 'iterative' or 'one_step'; got 'newton'"),
        ({"step_size": 0.0}, "step_size must be a positive finite number; got 0.0"),
        ({"step_size": float("inf")}, "step_size must be a positive finite number; got inf"),
        ({"step_size": True}, "step_size must be a positive finite number; got True"),
        (
            {"survival_model": "cox"},
            "survival_model must be 'iterated_mean' or 'hazard'; got 'cox'",
        ),
        ({"isotonic": 1}, "isotonic must be True or False; got 1"),
    ],
)
def test_h8_an_unknown_setting_is_refused(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        LTMLE({"a1": 1}, **settings, **_refusing())


def test_h8_a_step_size_under_iterative_targeting_is_accepted() -> None:
    LTMLE({"a1": 1}, step_size=0.5, **_refusing())


def test_h9_q_bounds_on_a_hazard_fit_is_the_binary_outcome_refusal() -> None:
    with pytest.raises(ValueError, match="q_bounds does not apply to a binary outcome"):
        LTMLE(
            {"a1": 1}, n_folds=1, survival_model="hazard", q_bounds=(0.01, 0.99), **_refusing()
        ).fit(LAW.frame(), **LAW.fit_columns())


@pytest.mark.parametrize("option", [{"survival_model": "hazard"}, {"isotonic": True}])
def test_h10_a_point_treatment_functional_refuses_the_curve_options(
    option: dict[str, object],
) -> None:
    message = (
        "survival_model= and isotonic= apply to a survival curve over time nodes, and a "
        "point-treatment functional has no curve. Remove them, or declare a longitudinal or "
        "TimeToEvent design"
    )
    with pytest.raises(MethodConfigurationError, match=re.escape(message)):
        TMLEMethod().with_overrides(**option).estimator_kwargs()


@pytest.mark.parametrize(
    ("method", "name"),
    [
        (SDRMethod, "SDR"),
        (IPSIMethod, "IPSI"),
        (GComputationMethod, "SequentialGComputation"),
        (IPWMethod, "SequentialIPW"),
    ],
)
@pytest.mark.parametrize(
    "option",
    [
        {"survival_model": "hazard"},
        {"targeting": "one_step"},
        {"step_size": 0.01},
        {"isotonic": True},
    ],
)
def test_h12_an_untargeted_longitudinal_method_refuses_the_curve_options(
    method: type[TMLEMethod], name: str, option: dict[str, object]
) -> None:
    message = (
        "survival_model=, a hazard targeting= algorithm, step_size= and isotonic= are settings "
        f"of the targeted fit, method='tmle'. The {name} estimator fits no hazard, and no "
        "registered study measures its projected curve. Fit method='tmle', or remove them"
    )
    with pytest.raises(MethodConfigurationError, match=re.escape(message)):
        method().with_overrides(**option).estimator_kwargs(longitudinal=True)


@pytest.mark.parametrize("engine", [SDR, IPSI, SequentialGComputation, SequentialIPW])
def test_h12_the_untargeted_engines_take_no_curve_option(engine: type) -> None:
    for option in ("survival_model", "step_size", "isotonic"):
        with pytest.raises(TypeError, match=re.escape(f"unexpected keyword argument {option!r}")):
            engine({"a1": 1}, **{option: None})
    assert "survival_model" not in SDRMethod().estimator_kwargs(longitudinal=True)
    assert "isotonic" not in IPSIMethod().estimator_kwargs(longitudinal=True)


def test_h11_a_continuous_baseline_treatment_is_refused() -> None:
    rng = np.random.default_rng(0)
    n = 200
    w = rng.integers(0, 2, n).astype(float)
    frame = pd.DataFrame({"W": w, "D": rng.uniform(0.0, 2.0, n)})
    for k in (1, 2):
        frame[f"C{k}"] = 1.0
        frame[f"Y{k}"] = 0.0
    message = (
        "survival_model='hazard' needs a categorical baseline treatment, and treatment 'D' "
        "is continuous. The hazard plug-in would need the treatment density at every dose, "
        "and that construction is not built. X70 in docs/roadmap.md tracks this. Use "
        "survival_model='iterated_mean'."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE({"up": Shift(0.5, cap=None)}, n_folds=1, survival_model="hazard", **_refusing()).fit(
            frame,
            treatment="D",
            continuous_treatment=["D"],
            outcome=["Y1", "Y2"],
            baseline=["W"],
            censoring=["C1", "C2"],
        )
    _assert_gap(raised.value)
