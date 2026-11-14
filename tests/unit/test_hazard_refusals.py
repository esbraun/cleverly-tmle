"""Every hazard and isotonic refusal, pinned by type and message before any learner.

X14 shipped H1 to H13.  X70 narrowed H5 and added H14 (part (d)); part (c) closed H7 and H11,
whose tests now check that the same designs fit, and added H16, H19, H20, H24 and H25.  Part
(a) closed H1, whose test now checks that a treatment at several nodes fits, and added H15, H17,
H18, H21, H22, H23 and H26 (H17 is pinned in ``tests/unit/test_hazard_multinode.py``).  Part
(b1) narrowed H2 to a design with a treatment at several nodes and H6 to a design without
covariates after baseline; its own refusals H28 to H31 are pinned in
``tests/unit/test_hazard_time_varying.py``.

Each fit passes learners that raise on ``fit``, so a refusal that reached a learner would
fail with ``AssertionError`` rather than the refusal under test.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.assessment import assessment_capabilities
from cleverly.exceptions import CapabilityError, MethodConfigurationError, RefusalReason
from cleverly.interventions import DoseQuantile, ModifiedPolicy, Shift
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    Separable,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.longitudinal.estimator import longitudinal_truncation_curve
from cleverly.methods import (
    GComputationMethod,
    IPSIMethod,
    IPWMethod,
    SDRMethod,
    TMLEMethod,
)
from cleverly.msm import MSM
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    end_of_study_law,
    survival_law,
)


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


def test_h1_a_treatment_at_several_nodes_now_fits() -> None:
    """X70 part (a) closed H1: a static plan at three treatment nodes is fitted."""
    result = LTMLE(
        {"a1": (1, 1, 1)},
        n_folds=1,
        survival_model="hazard",
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(LAW.frame(), **LAW.fit_columns(held=False))
    assert all(fit.hazard is not None and fit.hazard.converged for fit in result.fits.values())


def test_h2_a_covariate_after_baseline_beside_several_treatment_nodes_is_refused() -> None:
    """X70 part (b1) narrowed H2 to a design with a treatment at several nodes.

    Part (a) fits a treatment at several nodes with baseline covariates only; with a covariate
    measured after baseline the design stays refused until part (b2).
    """
    law = survival_law(with_l2=True)
    message = (
        "survival_model='hazard' on a design with a treatment at several nodes models the event "
        "and censoring hazards given the treatment history and the baseline covariates. This "
        "fit declares covariates measured after baseline (L2), and hazard targeting with such "
        "covariates on a design with a treatment at several nodes is not built. X70 in "
        "docs/roadmap.md tracks this. LTMLE with survival_model='iterated_mean' adjusts for "
        "them, and isotonic=True makes its curve monotone. If one baseline treatment is held "
        "over every node, declare treatment='A'."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"a1": 1}, n_folds=1, survival_model="hazard", targeting="one_step", **_refusing()
        ).fit(law.frame(), **law.fit_columns(held=False))
    _assert_gap(raised.value)


def test_h2_a_covariate_after_baseline_on_a_held_design_now_fits() -> None:
    """X70 part (b1): a held design with a covariate after baseline is fitted (one step)."""
    law = survival_law(with_l2=True)
    result = LTMLE(
        {"a1": 1},
        n_folds=1,
        survival_model="hazard",
        targeting="one_step",
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(law.frame(), **law.fit_columns())
    assert result.fits["a1 @ t=3"].hazard.nodes


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
        "survival_model='hazard' on a survival design, or keep "
        "targeting='iterative' and set isotonic=True for a monotone curve"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        LTMLE({"a1": 1}, targeting="one_step", **_refusing())


@pytest.mark.parametrize("survival_model", ["iterated_mean", "hazard"])
def test_h5_a_working_model_with_the_isotonic_projection_is_refused(survival_model: str) -> None:
    message = (
        "msm= with isotonic=True is not built. The isotonic projection moves the levels of "
        "one curve and leaves their influence curves as they were, and a working model fitted "
        "to projected levels has no influence curve this package derives. Set isotonic=False. "
        "survival_model='iterated_mean' takes msm= on every design, and survival_model='hazard' "
        "with targeting='one_step' takes it where the design admits a hazard fit and projects "
        "monotone curves."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"a1": 1},
            msm=_msm(),
            isotonic=True,
            survival_model=survival_model,
            targeting="one_step" if survival_model == "hazard" else "iterative",
            **_refusing(),
        )
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert raised.value.refusal.roadmap is None


def test_h14_a_working_model_on_the_per_horizon_algorithm_is_refused() -> None:
    message = (
        "msm= with survival_model='hazard' needs targeting='one_step'. A working model's "
        "cells span the horizons, and the per-horizon algorithm restarts from the initial "
        "hazards at each horizon, so no one set of hazards solves the coefficient scores. "
        "Set targeting='one_step'."
    )
    with pytest.raises(MethodConfigurationError, match=re.escape(message)) as raised:
        LTMLE({"a1": 1}, msm=_msm(), survival_model="hazard", **_refusing())
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_r_d1_a_ratio_trim_on_a_hazard_fit_is_refused(targeting: str) -> None:
    """The hazard path has no trim, so a fit that stamped one would report an untrimmed curve.

    Before the refusal, ``ratio_trim=0.5`` on ``make_point_survival(n=300, seed=3)`` gave a
    curve bit-identical to ``ratio_trim=None`` and still stamped the trimmed-ratio status.
    """
    message = (
        "ratio_trim= with survival_model='hazard' is not built. lmtp's trim acts on the density "
        "ratios of the iterated-mean recursion, and the hazard targeting has no trim of its "
        "clever covariates. X80 in docs/roadmap.md tracks this. Fit ratio_trim= with "
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
    assert raised.value.refusal.roadmap == "X80"
    # The iterated-mean recursion keeps its trim.
    assert LTMLE({"a1": 1}, ratio_trim=0.5, **_refusing()).ratio_trim == 0.5


def test_h6_a_pseudo_learner_on_a_hazard_fit_without_later_covariates_is_refused() -> None:
    """X70 part (b1) moved H6 to the fit: with covariates after baseline it fits the continuation."""
    message = (
        "pseudo_learner= is not used with survival_model='hazard' on a design with baseline "
        "covariates only. Such a fit regresses the event indicator at each node and fits no "
        "pseudo-outcome. Pass the hazard learner as outcome_learner="
    )
    estimator = LTMLE(
        {"a1": 1}, n_folds=1, survival_model="hazard", pseudo_learner=_Refuse(), **_refusing()
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        estimator.fit(LAW.frame(), **LAW.fit_columns())


def test_h7_a_modified_treatment_policy_now_fits() -> None:
    """X70 part (c) closed H7: a policy on the categorical baseline treatment is fitted."""
    result = LTMLE(
        {"up": _up()},
        n_folds=1,
        survival_model="hazard",
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(LAW.frame(), **LAW.fit_columns())
    assert all(fit.hazard is not None and fit.hazard.converged for fit in result.fits.values())


def _up() -> ModifiedPolicy:
    def up(a: Any, h: Any) -> Any:
        return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)

    return ModifiedPolicy("up", apply=up, policy_kind="known")


def test_h16_a_tuned_cap_on_a_hazard_fit_is_refused() -> None:
    frame, columns = _dose_frame()
    message = (
        "regimen 'up' tunes its cap from the treatment law, and survival_model='hazard' does "
        "not build the tuner's term in the influence curve. Use survival_model='iterated_mean', "
        "which builds it."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"up": Shift(0.5, cap=DoseQuantile(0.9), target="population")},
            n_folds=1,
            survival_model="hazard",
            **_refusing(),
        ).fit(frame, **columns)
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert raised.value.refusal.roadmap is None


def test_h19_a_working_model_beside_a_policy_on_a_hazard_fit_is_refused() -> None:
    message = (
        "msm= with survival_model='hazard' cannot take the modified treatment policy 'up'. Each "
        "such policy is targeted by its own weighted walk, and a coefficient score sums cells "
        "of every regimen, so no one walk solves it. Fit msm= with "
        "survival_model='iterated_mean', which takes a modified treatment policy cell."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"a0": 0, "up": _up()},
            n_folds=1,
            msm=_msm(),
            survival_model="hazard",
            targeting="one_step",
            **_refusing(),
        ).fit(LAW.frame(), **LAW.fit_columns())
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_h20_a_separable_regimen_beside_a_policy_is_refused() -> None:
    law = survival_law(causes=2)
    message = (
        "a Separable regimen and the modified treatment policy 'up' cannot share one fit. A "
        "separable fit targets every regimen of the fit on the two hazard factors, and a "
        "modified treatment policy needs its own weighted walk. Fit the separable regimens and "
        "the modified treatment policy in two LTMLE calls."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"up": _up(), "sep": Separable(arms={"death": 0, "relapse": 1}, first="death")},
            n_folds=1,
            survival_model="hazard",
            targeting="one_step",
            **_refusing(),
        ).fit(law.frame(), **law.fit_columns())
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_h24_pooled_hazards_beside_a_policy_are_refused() -> None:
    message = (
        "hazard_pooling='pooled' cannot take the modified treatment policy 'up'. The pooled "
        "hazard fit predicts every regimen at its plan's treatment values and builds no design "
        "at a shifted dose. Set hazard_pooling='node', which predicts each shifted dose from "
        "its own design."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        LTMLE(
            {"a0": 0, "up": _up()},
            n_folds=1,
            survival_model="hazard",
            hazard_pooling="pooled",
            **_refusing(),
        ).fit(LAW.frame(), **LAW.fit_columns())
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    # An iterated-mean fit pools its censoring regressions beside a policy and reaches them.
    with pytest.raises(AssertionError, match="a learner ran"):
        LTMLE({"up": _up()}, n_folds=1, hazard_pooling="pooled", **_refusing()).fit(
            LAW.frame(), **LAW.fit_columns()
        )


def test_h25_a_truncation_replay_of_a_policy_walk_is_refused() -> None:
    result = LTMLE(
        {"a0": 0, "up": _up()},
        n_folds=1,
        survival_model="hazard",
        targeting="one_step",
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(LAW.frame(), **LAW.fit_columns())
    message = (
        "the truncation replay re-targets the retained initial hazards of one shared walk, and "
        "regimen 'up' was targeted by its own weighted walk, whose slot hazards and loss "
        "weights the replay does not rebuild. Refit with each g_bounds= value, or replay a fit "
        "without a modified treatment policy."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        result.diagnostics.truncation_curve(bounds=[0.05])
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    # The row reads unavailable before any call, with the record the replay raises.
    (row,) = (
        item for item in assessment_capabilities(result) if item.operation == "truncation_curve"
    )
    assert not row.available
    with pytest.raises(CapabilityError, match=re.escape(message)) as replayed:
        longitudinal_truncation_curve(result, [0.05])
    assert row.refusal == replayed.value.refusal
    # The witness: the same fit without the policy replays, and its row reads available.
    static = LTMLE(
        {"a0": 0, "a1": 1},
        n_folds=1,
        survival_model="hazard",
        targeting="one_step",
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(LAW.frame(), **LAW.fit_columns())
    (row,) = (
        item for item in assessment_capabilities(static) if item.operation == "truncation_curve"
    )
    assert row.available
    assert len(static.diagnostics.truncation_curve(bounds=[0.05]).to_frame())


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


def _dose_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(0)
    n = 1_000
    frame = pd.DataFrame({"W": rng.integers(0, 2, n).astype(float), "D": rng.uniform(0, 2, n)})
    for k in (1, 2):
        frame[f"C{k}"] = 1.0
        frame[f"Y{k}"] = (rng.uniform(size=n) < 0.2).astype(float)
    first = frame["Y1"] == 1.0
    frame.loc[first, ["C2", "Y2"]] = np.nan
    columns: dict[str, Any] = {
        "treatment": "D",
        "continuous_treatment": ["D"],
        "outcome": ["Y1", "Y2"],
        "baseline": ["W"],
        "censoring": ["C1", "C2"],
    }
    return frame, columns


def test_h11_a_continuous_baseline_treatment_now_fits() -> None:
    """X70 part (c) closed H11: a policy on a continuous baseline dose is fitted."""
    frame, columns = _dose_frame()
    result = LTMLE(
        {"up": Shift(0.5, cap=None)},
        n_folds=1,
        survival_model="hazard",
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        censoring_learner=CellMeans(),
    ).fit(frame, **columns)
    estimates = [result.estimates[f"risk_regimen[up @ t={t}]"].psi for t in (1, 2)]
    assert 0.0 < estimates[0] <= estimates[1] < 1.0


# ------------------------------------------------- a treatment at several nodes (X70 (a))


def _assert_redirection(error: CapabilityError) -> None:
    assert error.refusal.reason is RefusalReason.API_REDIRECTION
    assert error.refusal.roadmap is None


def _multinode(regimens: dict[str, Any], **settings: Any) -> LTMLE:
    return LTMLE(regimens, n_folds=1, survival_model="hazard", **settings, **_refusing())


H15 = (
    "regimen 'up' holds a modified treatment policy at node 2, after the first decision. The "
    "policy reads the treatment each unit would have received at that node under the earlier "
    "intervention, so the hazard plug-in would need the law of that natural treatment as a "
    "targeted factor. Use survival_model='iterated_mean', which supports a modified treatment "
    "policy at every node."
)
H21 = (
    "regimen 'up' holds a modified treatment policy at the first of 3 decisions. "
    "survival_model='hazard' targets a modified treatment policy on one baseline treatment "
    "held over every node, and this design has a treatment at each node. Use "
    "survival_model='iterated_mean'."
)
H22 = (
    "survival_model='hazard' on a design with a treatment at several nodes needs a categorical "
    "treatment at every node, and node 2 ('A2') is continuous. Use "
    "survival_model='iterated_mean'."
)


def test_h15_a_policy_after_the_first_node_is_refused() -> None:
    with pytest.raises(CapabilityError, match=re.escape(H15)) as raised:
        _multinode({"up": (0, _up(), 0)}).fit(LAW.frame(), **LAW.fit_columns(held=False))
    _assert_redirection(raised.value)


def test_h21_a_policy_at_the_first_of_several_nodes_is_refused() -> None:
    with pytest.raises(CapabilityError, match=re.escape(H21)) as raised:
        _multinode({"up": (_up(), 0, 0)}).fit(LAW.frame(), **LAW.fit_columns(held=False))
    _assert_redirection(raised.value)


@pytest.mark.parametrize("first", ["categorical", "policy"])
def test_h22_a_continuous_later_node_is_refused_first(first: str) -> None:
    """H22 comes before H21 and H15: a continuous later node can only hold a policy."""
    node_1: Any = 0 if first == "categorical" else _up()
    with pytest.raises(CapabilityError, match=re.escape(H22)) as raised:
        _multinode({"up": (node_1, Shift(0.5, cap=None), 0)}).fit(
            LAW.frame(), **LAW.fit_columns(held=False), continuous_treatment=["A2"]
        )
    _assert_redirection(raised.value)


def test_h23_a_markov_order_on_several_nodes_is_refused() -> None:
    from cleverly.longitudinal import LongitudinalData

    data = LongitudinalData.from_frame(LAW.frame(), **LAW.fit_columns(held=False))
    message = (
        "markov_order= restricts each node's hazard to the last nodes of the history, and "
        "survival_model='hazard' on a design with a treatment at several nodes builds its paths "
        "and their hazards on the whole treatment history. Its influence curve under that "
        "restriction is not derived here. Use survival_model='iterated_mean' with "
        "markov_order=, or drop markov_order=."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        _multinode({"a1": (1, 1, 1)}).fit(data.with_markov_order(1))
    _assert_redirection(raised.value)


def test_h26_a_known_tilt_after_the_first_node_is_refused() -> None:
    """A tilt after node 1 needs a declared mechanism there, which H26 refuses first."""
    from cleverly.interventions import Incremental
    from cleverly.longitudinal import DynamicRegimen

    frame = LAW.frame()
    for level in range(3):
        frame[f"p2_{level}"] = 1.0 / 3.0
    known = {"A2": {level: f"p2_{level}" for level in range(3)}}
    message = (
        "treatment_probabilities= declares the treatment probabilities of 'A2' at node 2, one "
        "value per row at the observed treatment history. survival_model='hazard' on a design "
        "with a treatment at several nodes weights the hazards of every treatment path a "
        "regimen follows by the mechanism along that path's own treatment history, which a "
        "declared value does not give. Only the treatment probabilities of the first node, "
        "which follow no earlier treatment, can be declared on this fit. Leave this factor to "
        "be estimated by its learner, or use survival_model='iterated_mean', which reads "
        "declared factors at the observed history. To declare the treatment probabilities of "
        "the first node alone, pass the mapping form treatment_probabilities={'A1': {level: "
        "column or array, ...}}; an array over every node is refused even when its later "
        "columns are missing."
    )
    tilt = DynamicRegimen("tilt", (1, Incremental({0: 1.0, 1: 2.0, 2: 1.0}), 1))
    for regimens in ({"tilt": tilt}, {"always": (1, 1, 1)}):
        with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
            _multinode(regimens).fit(
                frame, **LAW.fit_columns(held=False), treatment_probabilities=known
            )
        _assert_redirection(raised.value)


def test_h26_declared_retention_at_the_first_node_is_refused() -> None:
    frame = LAW.frame()
    columns = LAW.fit_columns(held=False)
    first = columns["censoring"][0]
    retention = {first: np.full(len(frame), 0.9)}
    message = (
        f"censoring_probabilities= declares the retention probabilities of {first!r} at node 1"
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        _multinode({"always": (1, 1, 1)}).fit(frame, **columns, censoring_probabilities=retention)
    _assert_redirection(raised.value)


def test_h18_a_separable_regimen_on_several_nodes_is_refused() -> None:
    law = survival_law(causes=2)
    message = (
        "a Separable regimen is built for one baseline treatment held over every node, and this "
        "design declares a treatment at 3 nodes. A separable effect splits that one treatment "
        "into the components that act on each cause. Declare one baseline treatment held over "
        "every node, as treatment='A', for separable effects. A treatment at several nodes "
        "takes static regimens, known rules and known policies."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        _multinode(
            {"a0": (0, 0, 0), "sep": Separable(arms={"death": 0, "relapse": 1}, first="death")},
            targeting="one_step",
        ).fit(law.frame(), **law.fit_columns(held=False))
    _assert_redirection(raised.value)


def test_h18_pooled_hazards_on_several_nodes_are_refused() -> None:
    message = (
        "hazard_pooling='pooled' fits one regression over every node, so every node needs the "
        "same design. This design has a treatment decision at each of its 3 nodes, so its node "
        "designs differ. Use hazard_pooling='node'."
    )
    with pytest.raises(CapabilityError, match=re.escape(message)) as raised:
        _multinode({"a0": (0, 0, 0)}, hazard_pooling="pooled").fit(
            LAW.frame(), **LAW.fit_columns(held=False)
        )
    assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION


@pytest.mark.parametrize("estimator", ["ctmle", "drtmle"])
def test_h18_the_collaborative_and_drift_survival_fits_refuse_several_nodes(
    estimator: str,
) -> None:
    from cleverly.longitudinal.collaborative import SurvivalCTMLE
    from cleverly.longitudinal.drift import SurvivalDRTMLE

    cls = SurvivalCTMLE if estimator == "ctmle" else SurvivalDRTMLE
    name = cls.__name__
    with pytest.raises(CapabilityError, match=re.escape(f"{name} fits one baseline treatment")):
        settings = {"outcome_learner": _Refuse()}
        if estimator == "drtmle":
            settings = _refusing()
        cls({"a0": (0, 0, 0), "a1": (1, 1, 1)}, n_folds=1, **settings).fit(
            LAW.frame(), **LAW.fit_columns(held=False)
        )
