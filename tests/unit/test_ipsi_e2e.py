r"""T6, T12, T13, T15 and T17: what an IPSI fit refuses, and how it composes.

T13 pins each refusal whole, before any learner runs: R1 (``LTMLE`` with an estimated
mechanism at a tilt node), R2 (``SDR`` with a tilt), R3 (a tilt node with three levels), R4
(a tilt beside a modified treatment policy), R5 (a fluctuation setting), R6 (a ratio of a
level outside its scale's domain), R7 (a link projection that does not converge) and R8 (a
point-treatment container).  Each refusal has a control that the same surface answers when
its condition is absent.  The range witness fits a constant regression of 5 or -3 with a
censoring learner that predicts no dropout, so a level leaves ``[0, 1]``.

T6 and T12 run the few-cluster status and the bands.  T15 runs ``CausalStudy``: an
incremental estimand on a longitudinal design fits through ``ipsi``, and ``tmle`` and
``sdr`` report R1 and R2.  With every tilt node's mechanism declared, ``tmle`` fits the
tilt as a known policy.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    IncrementalEffect,
    IncrementalMean,
    IPSIMethod,
    LongitudinalTreatment,
    PointTreatment,
    RegimeContrast,
    Targeting,
)
from cleverly.data import CausalData
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    ParameterSpaceWarning,
)
from cleverly.interventions import Incremental, Shift
from cleverly.longitudinal import IPSI, LTMLE, SDR, DynamicRegimen, ipsi
from cleverly.longitudinal.estimator import IPSI_REFUSED_SETTINGS
from cleverly.methods import (
    IPSI_NO_FLUCTUATION,
    IPSI_POINT_REFUSAL,
    LTMLE_TILT_REFUSAL,
    SDR_TILT_REFUSAL,
)
from cleverly.msm import MSM

from .. import discrete_law_longitudinal as binary
from .. import discrete_law_longitudinal_ipsi as law
from ._inference_status_support import (
    LONGITUDINAL_NODES,
    assert_keeps_inference,
    assert_withholds,
    cluster_labels,
    longitudinal_learners,
)

pytestmark = pytest.mark.xdist_group("ipsi_e2e")

TILTS = [Incremental(1.0), Incremental(2.0)]
MEANS = ("ey_regimen[odds x2]", "ey_regimen[natural course]")

_STUDY_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}
_STUDY_SETTINGS: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "pseudo_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "censoring_learner": LogisticRegression(max_iter=1000),
    "n_folds": 2,
    "random_state": 17,
    "simultaneous": False,
}


class NeverFit:
    """A learner that fails the test if a refusal comes after the nuisance fit."""

    def fit(self, *args: Any, **kwargs: Any) -> NeverFit:
        raise AssertionError("a learner ran before the refusal")

    def predict(self, x: Any) -> Any:
        raise AssertionError("a learner ran before the refusal")

    def predict_proba(self, x: Any) -> Any:
        raise AssertionError("a learner ran before the refusal")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


def _never() -> dict[str, Any]:
    return {
        "outcome_learner": NeverFit(),
        "pseudo_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
        "n_folds": 1,
        "simultaneous": False,
    }


def _quiet(build: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(*args, **kwargs)


@pytest.fixture(scope="module")
def panel() -> Any:
    frame, _ = make_longitudinal(n=400, seed=0)
    return frame


@pytest.fixture(scope="module")
def ordinary(panel: Any) -> Any:
    """Every level inside ``(0, 1)``: the control of each R6 and range-warning witness."""
    return _quiet(
        IPSI(TILTS, **longitudinal_learners()).fit, panel, outcome="Y", **LONGITUDINAL_NODES
    )


def _constant_fit(panel: Any, constant: float, **settings: Any) -> Any:
    """Every regression predicts ``constant``; no unit drops out, by the censoring learner."""
    return IPSI(
        settings.pop("regimens", TILTS),
        outcome_learner=DummyRegressor(strategy="constant", constant=constant),
        pseudo_learner=DummyRegressor(strategy="constant", constant=constant),
        treatment_learner=DummyClassifier(strategy="prior"),
        censoring_learner=DummyClassifier(strategy="constant", constant=1),
        n_folds=1,
        simultaneous=False,
        **settings,
    ).fit(panel, outcome="Y", **LONGITUDINAL_NODES)


@pytest.fixture(scope="module")
def above(panel: Any) -> Any:
    """Both levels are near 1.5: above 1 and positive."""
    return _quiet(_constant_fit, panel, 5.0)


@pytest.fixture(scope="module")
def below(panel: Any) -> Any:
    """``ey_regimen[natural course]`` is -0.11: below 0."""
    return _quiet(_constant_fit, panel, -3.0)


# ------------------------------------------------------------------ T13: R1 and R2


def test_r1_ltmle_refuses_an_estimated_tilt_before_any_learner(panel: Any) -> None:
    with pytest.raises(CapabilityError) as raised:
        LTMLE({"x2": Incremental(2.0)}, **_never()).fit(panel, outcome="Y", **LONGITUDINAL_NODES)
    assert str(raised.value) == LTMLE_TILT_REFUSAL


def test_r1_ltmle_refuses_the_incremental_keyword() -> None:
    with pytest.raises(TypeError) as raised:
        LTMLE({"always": 1}, incremental=[Incremental(2.0)])
    assert LTMLE_TILT_REFUSAL in str(raised.value)


def test_r2_sdr_refuses_a_tilt_before_any_learner(panel: Any) -> None:
    with pytest.raises(CapabilityError) as raised:
        SDR({"x2": Incremental(2.0)}, **_never()).fit(panel, outcome="Y", **LONGITUDINAL_NODES)
    assert str(raised.value) == SDR_TILT_REFUSAL


def test_r2_sdr_refuses_a_tilt_beside_a_label(panel: Any) -> None:
    plan = DynamicRegimen("tilt then always", (Incremental(2.0), 1))
    with pytest.raises(CapabilityError) as raised:
        SDR([plan], **_never()).fit(panel, outcome="Y", **LONGITUDINAL_NODES)
    assert str(raised.value) == SDR_TILT_REFUSAL


# ------------------------------------------------------------------ T13: R3 and R4


def _three_level_panel(panel: Any) -> Any:
    frame = panel.copy()
    a2 = frame["A2"].to_numpy(dtype=float).copy()
    observed = np.flatnonzero(~np.isnan(a2))
    a2[observed[::7]] = 2.0
    return frame.assign(A2=a2)


def test_r3_refuses_a_tilt_node_with_three_levels(panel: Any) -> None:
    with pytest.raises(DataError) as raised:
        IPSI(TILTS, **_never()).fit(_three_level_panel(panel), outcome="Y", **LONGITUDINAL_NODES)
    assert str(raised.value) == (
        "an incremental propensity-score intervention tilts the *odds* of treatment, which "
        "names two arms; node 'A2' has 3 ([0.0, 1.0, 2.0]). Kennedy's binary odds multiplier "
        "alone does not select a multinomial policy. This API has no declared multi-arm tilt; "
        "one odds per contrast or a cost-based reference policy defines a different "
        "intervention with its own influence function."
    )


def test_r3_leaves_a_label_plan_on_three_levels_to_the_learners(panel: Any) -> None:
    """The control: a label plan on the same node is not refused, so the first learner runs."""
    with pytest.raises(AssertionError, match="a learner ran before the refusal"):
        IPSI({"always": 1}, **_never()).fit(
            _three_level_panel(panel), outcome="Y", **LONGITUDINAL_NODES
        )


def test_r4_refuses_a_tilt_beside_a_modified_treatment_policy(panel: Any) -> None:
    plan = DynamicRegimen("tilt then shift", (Incremental(2.0), Shift(0.0, cap=None)))
    with pytest.raises(CapabilityError) as raised:
        IPSI([plan], **_never()).fit(panel, outcome="Y", **LONGITUDINAL_NODES)
    assert str(raised.value) == (
        "regimen 'tilt then shift' sets node 'A2' by a modified treatment policy (Shift). "
        "IPSI estimates no density ratio for a modified treatment policy or a risk-ratio "
        "tilt, so a plan that mixes them with an odds tilt has no estimator here. Fit plans "
        "of those nodes alone with LTMLE or SDR"
    )


# ------------------------------------------------------------------ T13: R5 and R8


@pytest.mark.parametrize("setting", IPSI_REFUSED_SETTINGS)
def test_r5_refuses_each_fluctuation_setting(setting: str) -> None:
    with pytest.raises(TypeError) as raised:
        IPSI(TILTS, **{setting: 0.1})
    assert str(raised.value) == IPSI_NO_FLUCTUATION


def test_r5_leaves_an_unknown_keyword_to_the_shared_refusal() -> None:
    with pytest.raises(TypeError, match=r"^IPSI got an unexpected keyword argument 'bogus'"):
        IPSI(TILTS, bogus=1)


def test_r8_refuses_a_point_treatment_container() -> None:
    frame, _ = make_linear_ate(n=60, seed=3)
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
    with pytest.raises(TypeError) as raised:
        IPSI(TILTS, **_never()).fit(data)
    assert str(raised.value) == IPSI_POINT_REFUSAL


# ------------------------------------------------------------------ T13 and T17: R6 and range


def test_the_witness_levels_leave_the_outcome_range(above: Any, below: Any) -> None:
    assert above["ey_regimen[odds x2]"].psi > 1.0
    assert below["ey_regimen[natural course]"].psi < 0.0
    assert 0.0 < below["ey_regimen[odds x2]"].psi < 1.0


def test_r6_refuses_the_odds_scale_above_one(above: Any) -> None:
    value = above["ey_regimen[odds x2]"].psi
    with pytest.raises(CapabilityError) as raised:
        above.ratio(*MEANS, kind="or")
    assert str(raised.value) == (
        f"ey_regimen[odds x2] is {value:.4g}, outside (0, 1). Kennedy's estimator averages "
        "influence-function values and can leave the outcome range (Kennedy 2019, Section "
        "4.1); the odds scale needs a level in (0, 1). Report the difference scale"
    )


def test_a_positive_level_above_one_keeps_its_risk_ratio(above: Any) -> None:
    ratio = above.ratio(*MEANS, kind="rr")
    expected = above[MEANS[0]].psi / above[MEANS[1]].psi
    assert ratio.psi == pytest.approx(expected, rel=1e-12)


def test_r6_refuses_the_log_scale_below_zero(below: Any) -> None:
    value = below["ey_regimen[natural course]"].psi
    with pytest.raises(CapabilityError) as raised:
        below.ratio(*MEANS, kind="rr")
    assert str(raised.value) == (
        f"ey_regimen[natural course] is {value:.4g}, at or below 0. Kennedy's estimator "
        "averages influence-function values and can leave the outcome range (Kennedy 2019, "
        "Section 4.1); the log scale needs a positive level. Report the difference scale"
    )


def test_r6_does_not_refuse_levels_inside_the_range(ordinary: Any) -> None:
    for kind in ("rr", "or"):
        assert np.isfinite(ordinary.ratio(*MEANS, kind=kind).psi)


def test_the_range_warning_names_each_level_outside(panel: Any) -> None:
    with pytest.warns(ParameterSpaceWarning) as caught:
        _constant_fit(panel, -3.0)
    messages = [str(item.message) for item in caught if item.category is ParameterSpaceWarning]
    assert len(messages) == 1
    assert messages[0].startswith("ey_regimen[natural course] = -0.11: outside the outcome range")
    assert "ey_regimen[odds x2]" not in messages[0]
    assert "Kennedy's estimator averages influence-function values" in messages[0]


def test_no_range_warning_inside_the_range(panel: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterSpaceWarning)
        IPSI(TILTS, **longitudinal_learners()).fit(panel, outcome="Y", **LONGITUDINAL_NODES)


# ------------------------------------------------------------------ T13: R7


def _log_delta_msm(link: str) -> MSM:
    deltas = {"natural course": 1.0, "odds x2": 2.0, "odds x0.5": 0.5}

    def build(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.column_stack([np.ones(len(frame)), np.full(len(frame), np.log(deltas[label]))])

    return MSM(design=build, terms=("1", "log_delta"), link=link, design_kind="known")


THREE_TILTS = [Incremental(1.0), Incremental(2.0), Incremental(0.5)]


def test_r7_refuses_a_logit_projection_that_does_not_converge(panel: Any) -> None:
    with pytest.raises(CapabilityError) as raised:
        _quiet(_constant_fit, panel, 5.0, regimens=THREE_TILTS, msm=_log_delta_msm("logit"))
    assert str(raised.value) == (
        "the working model's logit projection of the IPSI estimates did not converge. A "
        "one-step estimate can leave [0, 1], so a logit link can have no finite solution. Use "
        "the identity link"
    )


def test_the_identity_projection_of_the_same_levels_is_reported(panel: Any) -> None:
    fit = _quiet(_constant_fit, panel, 5.0, regimens=THREE_TILTS, msm=_log_delta_msm("identity"))
    assert set(fit.estimates) == {"msm_regimen[1]", "msm_regimen[log_delta]"}
    assert all(np.isfinite(estimate.psi) for estimate in fit.estimates.values())


# ------------------------------------------------------------------ T6: clusters

N_CLUSTER_ROWS = 400


def _clustered(panel: Any, k: int | None, n_folds: int) -> Any:
    frame, roles = panel, {}
    if k is not None:
        frame = panel.assign(cluster=cluster_labels(N_CLUSTER_ROWS, k))
        roles["id"] = "cluster"
    return _quiet(
        IPSI(TILTS, **longitudinal_learners(n_folds=n_folds)).fit,
        frame,
        outcome="Y",
        **LONGITUDINAL_NODES,
        **roles,
    )


@pytest.mark.parametrize("n_folds", [1, 5])
def test_nineteen_clusters_withhold_the_interval(panel: Any, n_folds: int) -> None:
    result = _clustered(panel, 19, n_folds)
    assert result.config.estimator == "ipsi"
    assert_withholds(result, "few_cluster_plugin")


@pytest.mark.parametrize("n_folds", [1, 5])
def test_twenty_clusters_keep_the_interval_on_a_t_reference(panel: Any, n_folds: int) -> None:
    result = _clustered(panel, 20, n_folds)
    assert_keeps_inference(result)
    assert {estimate.reference_df for estimate in result.estimates.values()} == {18}


def test_the_cluster_labels_reach_the_variance_and_not_the_estimate(panel: Any) -> None:
    plain, clustered = _clustered(panel, None, 1), _clustered(panel, 20, 1)
    for name, estimate in clustered.estimates.items():
        assert estimate.psi == plain[name].psi
        assert abs(estimate.std_error - plain[name].std_error) > 1e-4
        sums = np.bincount(
            cluster_labels(N_CLUSTER_ROWS, 20), weights=clustered.influence_curves[name]
        )
        variance = 20 / 19 * float(np.sum((sums - sums.mean()) ** 2)) / N_CLUSTER_ROWS**2
        assert estimate.plugin_std_error == pytest.approx(np.sqrt(variance), rel=1e-6)


def test_grouped_folds_keep_each_cluster_in_one_fold(panel: Any) -> None:
    result = _clustered(panel, 20, 5)
    labels = cluster_labels(N_CLUSTER_ROWS, 20)
    for cluster in np.unique(labels):
        assert np.unique(result.folds.assignment[labels == cluster]).size == 1


# ------------------------------------------------------------------ T12 and T13: bands, bootstrap


def test_the_bootstrap_runs_and_reports_no_percentile_inference(panel: Any) -> None:
    fit = _quiet(
        IPSI(TILTS, **longitudinal_learners(n_bootstrap=2)).fit,
        panel,
        outcome="Y",
        **LONGITUDINAL_NODES,
    )
    assert fit.bootstrap is not None and fit.bootstrap.n_requested == 2
    for estimate in fit.estimates.values():
        assert estimate.bootstrap is not None
        assert estimate.bootstrap.inferential is False


def test_the_simultaneous_bands_cover_every_declared_plan(panel: Any) -> None:
    fit = _quiet(
        IPSI(THREE_TILTS, **longitudinal_learners(simultaneous=True)).fit,
        panel,
        outcome="Y",
        **LONGITUDINAL_NODES,
    )
    bands = fit.simultaneous
    assert bands is not None
    assert bands.critical_value > 1.959
    assert set(bands.bands) == set(fit.estimates)
    for name, (low, high) in bands.bands.items():
        pointwise = fit[name].ci
        assert low <= pointwise[0] and pointwise[1] <= high


def test_the_ipsi_function_is_the_class_fit(panel: Any) -> None:
    by_function = _quiet(
        ipsi, panel, regimens=TILTS, outcome="Y", **LONGITUDINAL_NODES, **longitudinal_learners()
    )
    by_class = _quiet(
        IPSI(TILTS, **longitudinal_learners()).fit, panel, outcome="Y", **LONGITUDINAL_NODES
    )
    assert by_function.config.estimator == "ipsi"
    for name, estimate in by_class.estimates.items():
        assert by_function[name].psi == estimate.psi


# ------------------------------------------------------------------ T15: CausalStudy


def _study(panel: Any, **design: Any) -> CausalStudy:
    return CausalStudy(panel, design=LongitudinalTreatment(**_STUDY_COLUMNS, **design))


def test_an_incremental_effect_on_a_longitudinal_design_lists_ipsi(panel: Any) -> None:
    effect = _study(panel).identify(IncrementalEffect(TILTS))
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["ipsi"].available and methods["ipsi"].reason is None
    assert not methods["tmle"].available
    assert methods["tmle"].reason == LTMLE_TILT_REFUSAL
    assert not methods["sdr"].available
    assert methods["sdr"].reason == SDR_TILT_REFUSAL
    with pytest.raises(CapabilityError) as raised:
        effect.estimate(method="tmle")
    assert LTMLE_TILT_REFUSAL in str(raised.value)


def test_the_identification_states_kennedy_and_kim_assumptions(panel: Any) -> None:
    identification = _study(panel).identify(IncrementalMean(TILTS)).identification
    text = " ".join(identification.assumptions)
    assert "no treatment positivity at a tilt node" in text
    assert "Kennedy 2019, Theorem 1" in text
    assert "Assumptions A2-M and A3" in text
    assert "not doubly robust" in identification.dr_condition
    assert identification.references == ("Kennedy (2019)", "Kim, Kennedy & Naimi (2021)")


def test_a_regimen_estimand_without_a_tilt_does_not_list_ipsi(panel: Any) -> None:
    effect = _study(panel).identify(RegimeContrast({"always": 1, "never": 0}, reference="never"))
    methods = {record.name: record for record in effect.available_methods()}
    assert not methods["ipsi"].available
    assert methods["tmle"].available and methods["sdr"].available


def test_a_point_functional_does_not_list_ipsi() -> None:
    frame, _ = make_linear_ate(n=120, seed=19)
    effect = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    ).identify(ATE())
    methods = {record.name: record for record in effect.available_methods()}
    assert not methods["ipsi"].available
    assert methods["ipsi"].reason == IPSI_POINT_REFUSAL


@pytest.mark.parametrize("estimand", [IncrementalEffect, IncrementalMean])
def test_causal_study_method_ipsi_is_the_direct_fit(panel: Any, estimand: Any) -> None:
    workflow = _quiet(
        _study(panel).identify(estimand(TILTS)).estimate, method="ipsi", **_STUDY_SETTINGS
    )
    direct = _quiet(
        IPSI(TILTS, **_STUDY_SETTINGS).fit,
        panel,
        **{key: list(value) for key, value in _STUDY_COLUMNS.items() if key != "outcome"},
        outcome="Y",
    )
    assert workflow.config.estimator == "ipsi"
    assert workflow.fitted_method == "ipsi"
    prefix = "ate_regimen[" if estimand is IncrementalEffect else "ey_regimen["
    assert workflow.estimates and all(name.startswith(prefix) for name in workflow.estimates)
    for name, estimate in workflow.estimates.items():
        assert estimate.psi == direct[name].psi
        assert estimate.std_error == direct[name].std_error
        key = workflow.parameter_keys[name]
        assert key.estimand == estimand(TILTS).name


def test_an_incremental_estimand_refuses_two_tilts_of_one_name(panel: Any) -> None:
    with pytest.raises(DataError, match="two tilts share the name 'odds x2'"):
        _study(panel).identify(IncrementalMean([Incremental(2.0), Incremental(2.0)]))


def test_an_incremental_estimand_refuses_a_regime(panel: Any) -> None:
    with pytest.raises(CapabilityError, match=r"IncrementalMean\.interventions"):
        _study(panel).identify(IncrementalMean([Incremental(2.0), 1]))


# ------------------------------------------------------------------ T15: known mechanism


def _known_tilt_frame() -> pd.DataFrame:
    frame = law.tilt_frame()
    declared = np.column_stack(
        [
            law.TILT_TABLES["G1"][frame["W"].to_numpy(dtype=int)],
            np.where(
                frame["C1"].to_numpy(dtype=float) == 1.0,
                law.TILT_TABLES["G2"][
                    frame["W"].to_numpy(dtype=int),
                    frame["A1"].to_numpy(dtype=int),
                    np.nan_to_num(frame["L2"].to_numpy(dtype=float)).astype(int),
                ],
                np.nan,
            ),
        ]
    )
    return frame.assign(
        g1_1=declared[:, 0],
        g1_0=1.0 - declared[:, 0],
        g2_1=declared[:, 1],
        g2_0=1.0 - declared[:, 1],
    )


_KNOWN_DESIGN: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W",),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
    "treatment_probabilities": {
        "A1": {0: "g1_0", 1: "g1_1"},
        "A2": {0: "g2_0", 1: "g2_1"},
    },
}
_CELL_SETTINGS: dict[str, Any] = {
    "outcome_learner": binary.CellMeans(),
    "pseudo_learner": binary.CellMeans(),
    "treatment_learner": binary.CellMeans(),
    "censoring_learner": binary.CellMeans(),
    "cross_fit": False,
    "g_bounds": (1e-8, 1.0),
    "simultaneous": False,
}


def test_a_declared_mechanism_lets_tmle_fit_the_tilt_as_a_known_policy() -> None:
    frame = _known_tilt_frame()
    study = CausalStudy(frame, design=LongitudinalTreatment(**_KNOWN_DESIGN))
    effect = study.identify(IncrementalMean([Incremental(1.0), Incremental(2.0)]))
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].available and methods["ipsi"].available
    assert not methods["sdr"].available
    by_tmle = _quiet(effect.estimate, method="tmle", **_CELL_SETTINGS)
    by_ipsi = _quiet(effect.estimate, method="ipsi", **_CELL_SETTINGS)
    assert by_tmle.config.estimator == "tmle"
    truth = law.functional_binary(
        law.TILT_PROBS, (law.tilt(2.0), law.tilt(2.0)), base=law.TILT_PROBS, known=(1, 2)
    )
    for fit in (by_tmle, by_ipsi):
        assert fit["ey_regimen[odds x2]"].psi == pytest.approx(float(truth), abs=1e-10)


def test_an_estimated_mechanism_keeps_tmle_unavailable_on_the_same_frame() -> None:
    """The control: the same frame with no declaration reports R1."""
    design = {
        key: value for key, value in _KNOWN_DESIGN.items() if key != "treatment_probabilities"
    }
    effect = CausalStudy(_known_tilt_frame(), design=LongitudinalTreatment(**design)).identify(
        IncrementalMean([Incremental(2.0)])
    )
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].reason == LTMLE_TILT_REFUSAL


# ------------------------------------------------------------------ IPSIMethod


def test_ipsi_method_refuses_a_point_translation() -> None:
    with pytest.raises(MethodConfigurationError) as raised:
        IPSIMethod().estimator_kwargs()
    assert str(raised.value) == IPSI_POINT_REFUSAL


@pytest.mark.parametrize(
    ("field", "value"), [("submodel_alpha", 0.9), ("max_iter", 3), ("tol", 1e-6)]
)
def test_ipsi_method_refuses_a_fluctuation_setting(field: str, value: Any) -> None:
    method = IPSIMethod(targeting=Targeting(**{field: value}))
    with pytest.raises(MethodConfigurationError) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert str(raised.value) == IPSI_NO_FLUCTUATION


def test_ipsi_method_translates_to_engine_settings_without_a_fluctuation() -> None:
    kwargs = IPSIMethod().estimator_kwargs(longitudinal=True)
    assert not set(IPSI_REFUSED_SETTINGS) & kwargs.keys()
    IPSI(TILTS, **kwargs)
