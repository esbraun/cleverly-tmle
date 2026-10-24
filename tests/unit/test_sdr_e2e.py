r"""T8 and T9: what an SDR fit refuses, and how it composes with the rest of the package.

T8 pins each refusal whole.  R2 refuses the fluctuation settings by name.  R3 refuses a log or
odds functional of a level outside that scale's domain.  R4 refuses ``method="sdr"`` on a
point functional before an engine is built.  A logit or log working model whose projection of
the transforms does not converge raises.  Each refusal has a control that the same surface
answers when its condition is absent.  The witness fit regresses every node on the constant
2 or -2 with prior mechanisms, so a transform mean leaves ``[0, 1]`` and the range warning
fires.

T9 runs the shipped compositions end to end: the few-cluster status at 19 and 20 clusters in
sample and at five grouped folds, the truncation replay, the bootstrap, the simultaneous bands,
``CausalStudy(method="sdr")``, the report text, persistence, and :func:`sdr`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    ATE,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    RegimeContrast,
    SDRMethod,
    Targeting,
)
from cleverly.assessment import SDR_SCORE_EQUATIONS_REFUSAL
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.exceptions import (
    CapabilityError,
    MethodConfigurationError,
    ParameterSpaceWarning,
    RefusalReason,
)
from cleverly.longitudinal import LTMLE, SDR, sdr
from cleverly.longitudinal.estimator import SDR_REFUSED_SETTINGS
from cleverly.methods import SDR_POINT_REFUSAL, sdr_no_fluctuation_refusal

from .. import discrete_law_longitudinal as law
from ..longitudinal_sdr import COLUMNS, sample
from ..studies.canonical_ltmle import declared_regimens
from . import test_influence_gateaux_longitudinal_msm as msm_tests
from ._inference_status_support import (
    LONGITUDINAL_NODES,
    ROUTES,
    assert_keeps_inference,
    assert_round_trips,
    assert_shows_interval,
    cluster_labels,
    longitudinal_learners,
)
from .test_sequential_design import COLUMNS as PANEL_COLUMNS
from .test_sequential_design import multivalue_panel

pytestmark = pytest.mark.xdist_group("sdr_e2e")

PAIR = {"always": 1, "never": 0}
MEANS = ("ey_regimen[always]", "ey_regimen[never]")

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


def _constant_fit(constant: float, **settings: Any) -> Any:
    """Every regression predicts ``constant``; every mechanism is the class share."""
    regimens = settings.pop(
        "regimens", declared_regimens({label: law.REGIMEN_SPEC[label] for label in PAIR})
    )
    return SDR(
        regimens,
        outcome_learner=DummyRegressor(strategy="constant", constant=constant),
        pseudo_learner=DummyRegressor(strategy="constant", constant=constant),
        treatment_learner=DummyClassifier(strategy="prior"),
        censoring_learner=DummyClassifier(strategy="prior"),
        n_folds=1,
        simultaneous=False,
        **settings,
    ).fit(law.frame(), **COLUMNS)


def _quiet(build: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(*args, **kwargs)


@pytest.fixture(scope="module")
def above() -> Any:
    """``ey_regimen[always]`` is 1.26: above 1 and positive."""
    return _quiet(_constant_fit, 2.0)


@pytest.fixture(scope="module")
def below() -> Any:
    """``ey_regimen[always]`` is -0.63: below 0."""
    return _quiet(_constant_fit, -2.0)


@pytest.fixture(scope="module")
def panel() -> Any:
    frame, _ = make_longitudinal(n=400, seed=0)
    return frame


@pytest.fixture(scope="module")
def ordinary(panel: Any) -> Any:
    """Every level inside ``(0, 1)``: the control of each R3 and range-warning witness."""
    return _quiet(
        SDR(PAIR, reference="never", **longitudinal_learners()).fit,
        panel,
        outcome="Y",
        **LONGITUDINAL_NODES,
    )


# ------------------------------------------------------------------ T8: R2


@pytest.mark.parametrize("setting", SDR_REFUSED_SETTINGS)
def test_r2_refuses_each_fluctuation_setting_by_name(setting: str) -> None:
    with pytest.raises(MethodConfigurationError) as raised:
        SDR(PAIR, **{setting: 0.1})
    assert str(raised.value) == sdr_no_fluctuation_refusal(setting).message()
    assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_r2_leaves_an_unknown_keyword_to_the_shared_refusal() -> None:
    with pytest.raises(TypeError, match=r"^SDR got an unexpected keyword argument 'bogus'"):
        SDR(PAIR, bogus=1)


def test_ltmle_still_takes_the_fluctuation_settings() -> None:
    fitter = LTMLE(PAIR, alpha=0.9, max_iter=3, tol=1e-6)
    assert (fitter.alpha, fitter.max_iter, fitter.tol) == (0.9, 3, 1e-6)


# ------------------------------------------------------------------ T8: R3 and the warning


def test_the_witness_levels_leave_the_outcome_range(above: Any, below: Any) -> None:
    assert above["ey_regimen[always]"].psi > 1.0
    assert below["ey_regimen[always]"].psi < 0.0
    assert 0.0 < above["ey_regimen[never]"].psi < 1.0


def test_r3_refuses_the_odds_scale_above_one(above: Any) -> None:
    value = above["ey_regimen[always]"].psi
    with pytest.raises(CapabilityError) as raised:
        above.ratio(*MEANS, kind="or")
    assert str(raised.value) == (
        f"ey_regimen[always] is {value:.4g}, outside (0, 1). An SDR estimate is an average of "
        "transformed outcomes and can leave the outcome range (Díaz et al. 2023, Section "
        "5.3); the odds scale needs a level in (0, 1). Report the difference scale."
    )


def test_a_positive_level_above_one_keeps_its_risk_ratio(above: Any) -> None:
    """The log scale needs only a positive level, so R3 does not refuse ``rr`` here."""
    ratio = above.ratio(*MEANS, kind="rr")
    expected = above["ey_regimen[always]"].psi / above["ey_regimen[never]"].psi
    assert ratio.psi == pytest.approx(expected, rel=1e-12)


def test_r3_refuses_the_log_scale_below_zero(below: Any) -> None:
    value = below["ey_regimen[always]"].psi
    with pytest.raises(CapabilityError) as raised:
        below.ratio(*MEANS, kind="rr")
    assert str(raised.value) == (
        f"ey_regimen[always] is {value:.4g}, at or below 0. An SDR estimate is an average of "
        "transformed outcomes and can leave the outcome range (Díaz et al. 2023, Section "
        "5.3); the log scale needs a positive level. Report the difference scale."
    )


def test_r3_does_not_refuse_levels_inside_the_range(ordinary: Any) -> None:
    for kind in ("rr", "or"):
        assert np.isfinite(ordinary.ratio(*MEANS, kind=kind).psi)


@pytest.mark.parametrize("constant", [2.0, -2.0])
def test_the_range_warning_names_the_level(constant: float) -> None:
    with pytest.warns(ParameterSpaceWarning, match=r"^ey_regimen\[always\] = ") as caught:
        _constant_fit(constant)
    messages = [str(item.message) for item in caught if item.category is ParameterSpaceWarning]
    assert len(messages) == 1
    assert "outside the outcome range [0, 1]" in messages[0]
    assert "ey_regimen[never]" not in messages[0]


def test_no_range_warning_inside_the_range(panel: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterSpaceWarning)
        SDR(PAIR, reference="never", **longitudinal_learners()).fit(
            panel, outcome="Y", **LONGITUDINAL_NODES
        )


def test_ltmle_never_warns_on_the_parameter_space() -> None:
    """LTMLE clips its regressions, so the same learners keep its levels in [0, 1]."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LTMLE(
            declared_regimens({label: law.REGIMEN_SPEC[label] for label in PAIR}),
            outcome_learner=DummyRegressor(strategy="constant", constant=2.0),
            pseudo_learner=DummyRegressor(strategy="constant", constant=2.0),
            treatment_learner=DummyClassifier(strategy="prior"),
            censoring_learner=DummyClassifier(strategy="prior"),
            n_folds=1,
            simultaneous=False,
        ).fit(law.frame(), **COLUMNS)
    assert not [item for item in caught if item.category is ParameterSpaceWarning]


# ------------------------------------------------------------------ T8: the working model


def _msm_fit(constant: float, family: str) -> Any:
    return _quiet(
        _constant_fit,
        constant,
        regimens=declared_regimens(law.REGIMEN_SPEC),
        msm=msm_tests.declared(family),
    )


@pytest.mark.parametrize(("constant", "family"), [(3.0, "logit"), (5.0, "log")])
def test_a_projection_that_does_not_converge_is_refused(constant: float, family: str) -> None:
    with pytest.raises(CapabilityError) as raised:
        _msm_fit(constant, family)
    assert str(raised.value) == (
        f"the working model's {family} projection of the SDR transforms did not converge. An "
        "SDR transform can leave [0, 1] (Díaz et al. 2023, Section 5.3), so a "
        f"{family} link can have no finite solution. Use the identity link, or fit "
        "LTMLE(msm=...)."
    )


def test_the_identity_projection_of_the_same_transforms_is_reported() -> None:
    """The control: the same transforms, under the identity link, have a solution."""
    fit = _msm_fit(5.0, "identity")
    assert all(np.isfinite(estimate.psi) for estimate in fit.estimates.values())


def test_a_converging_log_projection_is_reported() -> None:
    """At the constant 3 the log projection converges while the logit one does not."""
    fit = _msm_fit(3.0, "log")
    assert all(np.isfinite(estimate.psi) for estimate in fit.estimates.values())


# ------------------------------------------------------------------ T8: R4 and SDRMethod


def test_r4_lists_sdr_as_unavailable_on_a_point_functional() -> None:
    frame, _ = make_linear_ate(n=120, seed=19)
    effect = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    ).identify(ATE())
    methods = {record.name: record for record in effect.available_methods()}
    assert not methods["sdr"].available
    assert methods["sdr"].refusal == SDR_POINT_REFUSAL
    assert methods["tmle"].available
    with pytest.raises(CapabilityError) as raised:
        effect.estimate(method="sdr")
    assert SDR_POINT_REFUSAL.message() in str(raised.value)


def test_r4_lists_sdr_as_available_on_a_longitudinal_functional(panel: Any) -> None:
    effect = CausalStudy(panel, design=LongitudinalTreatment(**_STUDY_COLUMNS)).identify(
        RegimeContrast(PAIR, reference="never")
    )
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["sdr"].available and methods["sdr"].reason is None


def test_sdr_method_refuses_a_point_translation() -> None:
    with pytest.raises(MethodConfigurationError) as raised:
        SDRMethod().estimator_kwargs()
    assert str(raised.value) == SDR_POINT_REFUSAL.message()


@pytest.mark.parametrize(
    ("field", "value", "engine"),
    [("submodel_alpha", 0.9, "alpha"), ("max_iter", 3, "max_iter"), ("tol", 1e-6, "tol")],
)
def test_sdr_method_refuses_a_fluctuation_setting(field: str, value: Any, engine: str) -> None:
    method = SDRMethod(targeting=Targeting(**{field: value}))
    with pytest.raises(MethodConfigurationError) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert str(raised.value) == sdr_no_fluctuation_refusal(engine).message()


def test_sdr_method_translates_to_engine_settings_without_a_fluctuation() -> None:
    kwargs = SDRMethod().estimator_kwargs(longitudinal=True)
    assert not set(SDR_REFUSED_SETTINGS) & kwargs.keys()
    SDR(PAIR, **kwargs)


# ------------------------------------------------------------------ T8: sensitivity


def test_every_sensitivity_analysis_refuses_as_it_does_for_ltmle(ordinary: Any, panel: Any) -> None:
    """The shipped longitudinal refusals name a longitudinal fit, not one estimator."""
    ltmle = _quiet(
        LTMLE(PAIR, reference="never", **longitudinal_learners()).fit,
        panel,
        outcome="Y",
        **LONGITUDINAL_NODES,
    )
    shipped = {item.operation: item for item in ltmle.sensitivity.capabilities}
    reported = {item.operation: item for item in ordinary.sensitivity.capabilities}
    assert reported.keys() == shipped.keys()
    for name, item in reported.items():
        assert item.status == shipped[name].status
        assert item.status.value == "unavailable"
        assert item.reason == shipped[name].reason
        assert "LTMLE" not in (item.reason or "")


# ------------------------------------------------------------------ T9: clusters


N_CLUSTER_ROWS = 400


def _clustered(k: int | None, n_folds: int) -> Any:
    frame = multivalue_panel(n=N_CLUSTER_ROWS, seed=43)
    roles: dict[str, Any] = {}
    if k is not None:
        frame = frame.assign(cluster=cluster_labels(N_CLUSTER_ROWS, k))
        roles["id"] = "cluster"
    return _quiet(
        SDR(PAIR, reference="never", **longitudinal_learners(n_folds=n_folds)).fit,
        frame,
        **PANEL_COLUMNS,
        **roles,
    )


@pytest.mark.parametrize("n_folds", [1, 5])
def test_nineteen_clusters_show_a_labelled_t_interval(n_folds: int) -> None:
    result = _clustered(19, n_folds)
    assert result.config.estimator == "sdr"
    assert_shows_interval(result, "few_cluster_plugin", 17)


@pytest.mark.parametrize("n_folds", [1, 5])
def test_twenty_clusters_keep_the_interval_on_a_t_reference(n_folds: int) -> None:
    result = _clustered(20, n_folds)
    assert_keeps_inference(result)
    assert {estimate.reference_df for estimate in result.estimates.values()} == {18}


def test_the_cluster_labels_reach_the_variance_and_not_the_estimate() -> None:
    """In sample the labels move only the spread: the curve is summed by cluster."""
    plain, clustered = _clustered(None, 1), _clustered(20, 1)
    for name, estimate in clustered.estimates.items():
        assert estimate.psi == plain[name].psi
        assert abs(estimate.std_error - plain[name].std_error) > 1e-4
        curve = clustered.influence_curves[name]
        sums = np.bincount(cluster_labels(N_CLUSTER_ROWS, 20), weights=curve)
        # The cluster-summed variance, with the J / (J - 1) small-sample factor.
        variance = 20 / 19 * float(np.sum((sums - sums.mean()) ** 2)) / N_CLUSTER_ROWS**2
        assert estimate.plugin_std_error == pytest.approx(np.sqrt(variance), rel=1e-6)


def test_grouped_folds_keep_each_cluster_in_one_fold() -> None:
    result = _clustered(20, 5)
    labels = cluster_labels(N_CLUSTER_ROWS, 20)
    for cluster in np.unique(labels):
        assert np.unique(result.folds.assignment[labels == cluster]).size == 1


# ------------------------------------------------------------------ T9: truncation replay


def _cell_fit(frame: Any, n_folds: int, g_bounds: tuple[float, float]) -> Any:
    return _quiet(
        SDR(
            declared_regimens({label: law.REGIMEN_SPEC[label] for label in PAIR}),
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=law.CellMeans(),
            censoring_learner=law.CellMeans(),
            n_folds=n_folds,
            random_state=4,
            g_bounds=g_bounds,
            simultaneous=False,
        ).fit,
        frame,
        **COLUMNS,
    )


@pytest.fixture(scope="module")
def replay_frame() -> Any:
    return sample(600, 3)


@pytest.mark.parametrize("n_folds", [1, 5])
def test_the_replay_recipe_names_the_sdr_recursion(replay_frame: Any, n_folds: int) -> None:
    fit = _cell_fit(replay_frame, n_folds, (0.01, 1.0))
    assert fit.replay_recipe is not None
    assert fit.replay_recipe.estimator == "sdr"


@pytest.mark.parametrize("n_folds", [1, 5])
def test_the_replay_at_the_fitted_bound_is_the_fit(replay_frame: Any, n_folds: int) -> None:
    fit = _cell_fit(replay_frame, n_folds, (0.01, 1.0))
    curve = fit.diagnostics.truncation_curve([fit.config.g_bounds]).to_dict("list")
    assert all(curve["is_fitted_bound"])
    assert curve["psi"] == curve["fitted_psi"]
    assert all(delta == 0.0 for delta in curve["delta_from_fitted"])


@pytest.mark.parametrize("n_folds", [1, 5])
def test_the_replay_at_another_bound_is_a_fit_at_that_bound(
    replay_frame: Any, n_folds: int
) -> None:
    """The replay reruns the SDR recursion at the new bound.

    In sample, saturated regressions leave residuals that sum to zero in each cell, so no
    bound moves the estimate.  At five folds the new bound moves it, which is the witness
    that the replay reads the bound.
    """
    fit = _cell_fit(replay_frame, n_folds, (0.01, 1.0))
    direct = _cell_fit(replay_frame, n_folds, (0.2, 1.0))
    curve = fit.diagnostics.truncation_curve([(0.2, 1.0)]).to_dict("list")
    replayed = dict(zip(curve["estimand"], curve["psi"], strict=True))
    for name, estimate in direct.estimates.items():
        assert replayed[name] == pytest.approx(estimate.psi, abs=1e-12)
        if n_folds > 1:
            assert abs(replayed[name] - fit[name].psi) > 1e-4


# ------------------------------------------------------------------ T9: bootstrap and bands


def test_the_bootstrap_runs_and_reports_no_percentile_inference(panel: Any) -> None:
    """LTMLE's in-sample end-of-study kind is licensed; SDR's same design is not."""
    settings = longitudinal_learners(n_bootstrap=2)
    fits = {
        engine.__name__: _quiet(
            engine(PAIR, reference="never", **settings).fit,
            panel,
            outcome="Y",
            **LONGITUDINAL_NODES,
        )
        for engine in (SDR, LTMLE)
    }
    assert fits["SDR"].bootstrap is not None
    assert fits["SDR"].bootstrap.n_requested == 2
    for name, estimate in fits["SDR"].estimates.items():
        assert estimate.bootstrap is not None
        assert estimate.bootstrap.diagnostic is False
        assert estimate.bootstrap.support.label == "experimental"
        assert fits["LTMLE"].estimates[name].bootstrap.diagnostic is False


def test_the_simultaneous_bands_cover_every_estimate(panel: Any) -> None:
    fit = _quiet(
        SDR(PAIR, reference="never", **longitudinal_learners(simultaneous=True)).fit,
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


# ------------------------------------------------------------------ T9: the workflow


def test_causal_study_method_sdr_is_the_direct_fit(panel: Any) -> None:
    workflow = _quiet(
        CausalStudy(panel, design=LongitudinalTreatment(**_STUDY_COLUMNS))
        .identify(RegimeContrast(PAIR, reference="never"))
        .estimate,
        method="sdr",
        **_STUDY_SETTINGS,
    )
    direct = _quiet(
        SDR(PAIR, reference="never", **_STUDY_SETTINGS).fit,
        panel,
        **{key: list(value) for key, value in _STUDY_COLUMNS.items() if key != "outcome"},
        outcome="Y",
    )
    assert workflow.config.estimator == "sdr"
    assert workflow.fitted_method == "sdr"
    for name, estimate in workflow.estimates.items():
        assert estimate.psi == direct[name].psi
        assert estimate.std_error == direct[name].std_error


def test_the_sdr_function_is_the_class_fit(panel: Any) -> None:
    by_function = _quiet(
        sdr,
        panel,
        regimens=PAIR,
        reference="never",
        outcome="Y",
        **LONGITUDINAL_NODES,
        **longitudinal_learners(),
    )
    by_class = _quiet(
        SDR(PAIR, reference="never", **longitudinal_learners()).fit,
        panel,
        outcome="Y",
        **LONGITUDINAL_NODES,
    )
    assert by_function.config.estimator == "sdr"
    for name, estimate in by_class.estimates.items():
        assert by_function[name].psi == estimate.psi


# ------------------------------------------------------------------ T9: reports


def test_the_summary_names_the_estimator(ordinary: Any) -> None:
    summary = ordinary.summary()
    assert summary.heading.strip() == "Longitudinal estimate (SDR)"
    header = dict(summary.header)
    assert header["Method"] == "sequentially doubly robust estimator (SDR)"
    assert (header["Time points"], header["Rows"]) == ("2", "400")
    full = ordinary.summary(detail="full")
    assert (
        full.section("Settings")
        .lines[0]
        .endswith("on each partial-path probability of treatment and staying under observation.")
    )
    assert "Diaz, Williams, Hoffman and Schenck" in " ".join(full.section("Method details").lines)
    assert "LTMLE" not in summary.heading


def test_the_diagnostics_report_no_score_equation(ordinary: Any) -> None:
    items = {item.name: item for item in ordinary.diagnostics.run_all().items}
    score = items["score_equations"]
    assert score.status.value == "not_applicable"
    assert score.detail == SDR_SCORE_EQUATIONS_REFUSAL.message()
    assert items["support"].status.value == "completed"


def test_the_support_report_carries_the_sdr_node_columns(ordinary: Any) -> None:
    frame = ordinary.diagnostics.support().to_frame()
    sdr_columns = {
        "target_outside",
        "target_min",
        "target_max",
        "prediction_outside",
        "bound_binding",
    }
    assert sdr_columns <= set(frame.columns)
    assert not {"epsilon", "converged"} & set(frame.columns)
    # The node-1 transform leaves [0, 1] on this panel; the node-2 target is the outcome.
    first = frame[frame["time"] == 1]
    assert (first["target_outside"] > 0.0).all()
    assert (frame[frame["time"] == 2]["target_outside"] == 0.0).all()


# ------------------------------------------------------------------ T9: persistence


@pytest.mark.parametrize("route", ROUTES)
def test_an_sdr_fit_round_trips(route: str) -> None:
    result = _clustered(20, 5)
    restored = assert_round_trips(result, "influence_curve", route)
    assert restored.config.estimator == "sdr"
    assert restored.fitted_method == "sdr"
    assert restored.replay_recipe.estimator == "sdr"
    assert len(restored.fold_mechanisms) == len(result.fold_mechanisms) == 5
    assert restored.summary() == result.summary()


def test_save_and_load_keep_the_sdr_fit(ordinary: Any, tmp_path: Any) -> None:
    path = tmp_path / "sdr.cleverly"
    ordinary.save(path)
    loaded = cleverly.load(path)
    assert loaded.config.estimator == "sdr"
    for name, estimate in ordinary.estimates.items():
        assert loaded[name].psi == estimate.psi


# ------------------------------------------------------------------ T8: a continuous outcome


def _continuous(panel: Any, constant: float | None) -> Any:
    """A continuous end-of-study outcome; ``constant`` replaces both regressions."""
    frame = panel.assign(Y=panel["Y"] * 3.0 + panel["W1"])
    learners = longitudinal_learners()
    if constant is not None:
        learners["outcome_learner"] = DummyRegressor(strategy="constant", constant=constant)
        learners["pseudo_learner"] = DummyRegressor(strategy="constant", constant=constant)
    return SDR(PAIR, reference="never", **learners).fit(frame, outcome="Y", **LONGITUDINAL_NODES)


def test_a_continuous_level_outside_its_bounds_warns_with_those_bounds(panel: Any) -> None:
    """With no declared ``q_bounds`` the range is the derived one: the observed range padded."""
    with pytest.warns(ParameterSpaceWarning) as caught:
        fit = _continuous(panel, 20.0)
    assert fit.data.family == "gaussian" and fit.config.q_bounds is None
    observed = (panel["Y"] * 3.0 + panel["W1"]).dropna()
    width = float(observed.max() - observed.min())
    low, high = fit.scaler.lower, fit.scaler.upper
    assert (low, high) == pytest.approx(
        (observed.min() - 0.1 * width, observed.max() + 0.1 * width), abs=1e-12
    )
    (message,) = [str(item.message) for item in caught if item.category is ParameterSpaceWarning]
    assert message.startswith("ey_regimen[never] = ")
    assert f"outside the outcome range [{low:.4g}, {high:.4g}]" in message
    assert fit["ey_regimen[never]"].psi < low


def test_a_continuous_level_inside_its_bounds_does_not_warn(panel: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterSpaceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        _continuous(panel, None)
