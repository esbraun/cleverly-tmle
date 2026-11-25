"""Engine options the method objects carry: the route reaches the engine, or refuses first.

Each routed option is tested at a non-default value, because a default value cannot show that
the translation reached the engine. The parity tests compare a study fit with the direct engine
fit on the same data and seed, with a witness that the option moved something, so a route that
dropped the key would fail. The refusal tests use a learner whose ``fit`` raises, so each
refusal is shown to fire before any learner.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    CollaborativeTMLEMethod,
    DRTMLEMethod,
    EPMethod,
    GComputationMethod,
    Inference,
    IPSIMethod,
    IPWMethod,
    LongitudinalTreatment,
    ModelSpec,
    ParametricGFormulaMethod,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
    SDRMethod,
    SparseInfluenceWarning,
    Targeting,
    TimeToEvent,
    TMLEMethod,
)
from cleverly import methods as methods_module
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators.tmle import TMLE
from cleverly.exceptions import CapabilityError, CleverlyError, MethodConfigurationError
from cleverly.interventions import Incremental
from cleverly.learners import GLM
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
    SurvivalCTMLE,
    SurvivalDRTMLE,
)
from cleverly.methods import ENGINE_ONLY, SHORTCUTS
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from cleverly.survival.data import continuous_kwargs
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_pricing_common as common
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.unit.test_continuous_survival_memory import _law

ROLES: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
REGIMENS = {"always": 1, "never": 0}


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }


def _longitudinal_effect(cluster: bool = False) -> Any:
    frame, _ = make_longitudinal(n=400, seed=12)
    roles = dict(ROLES)
    if cluster:
        frame["site"] = np.arange(len(frame)) // 20
        roles["cluster"] = "site"
    study = CausalStudy(frame, design=LongitudinalTreatment(**roles))
    return study, study.identify(RegimeMean(REGIMENS))


def _continuous_effect(horizons: tuple[float, ...] = (0.2, 1.0)) -> Any:
    t, d, a, w = _law(300, 1)
    frame = pd.DataFrame({"time": t, "event": d, "Z": a, "W": w[:, 0]})
    design = TimeToEvent(
        time="time", event="event", treatment="Z", baseline=["W"], grid="continuous"
    )
    study = CausalStudy(frame, design=design)
    return study, study.identify(
        RegimeContrast({"t": 1, "c": 0}, reference="c", horizons=list(horizons))
    )


def _point_effect() -> Any:
    frame, _ = make_linear_ate(n=200, seed=0)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return CausalStudy(frame, design=design).identify(ATE())


# ------------------------------------------------------------------ the groups


class TestTheGroups:
    def test_every_new_field_has_its_same_name_shortcut(self) -> None:
        for group, name in (
            ("inference", "bootstrap_interval"),
            ("inference", "variance"),
            ("inference", "cluster_variance"),
            ("runtime", "max_memory_bytes"),
            ("models", "hazard_library"),
            ("models", "censoring_library"),
            ("targeting", "norm"),
        ):
            assert SHORTCUTS[group][name] == name

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: Inference(bootstrap_interval="student"), "bootstrap_interval must be"),
            (lambda: Inference(variance="sandwich"), "variance must be"),
            (lambda: Inference(cluster_variance="bootstrap"), "cluster_variance must be"),
            (lambda: Inference(bootstrap_interval="bca"), "n_bootstrap=0 runs none"),
            (
                lambda: Inference(cluster_variance="jackknife", n_bootstrap=20),
                "both replace the influence-curve variance",
            ),
            (
                lambda: Inference(cluster_variance="jackknife", variance="policy_weighted"),
                "both replace the influence-curve variance",
            ),
            (lambda: Targeting(norm="sup"), "norm must be"),
            (lambda: ModelSpec(hazard_library=()), "at least one hazard learner"),
            (lambda: ModelSpec(censoring_library="cox"), "sequence of hazard learners"),
            (lambda: Runtime(max_memory_bytes=0), "max_memory_bytes must be"),
        ],
    )
    def test_a_value_outside_its_domain_is_a_cleverly_error(
        self, build: Callable[[], Any], match: str
    ) -> None:
        with pytest.raises(CleverlyError, match=match):
            build()

    def test_a_library_is_stored_as_a_tuple(self) -> None:
        models = ModelSpec(hazard_library=[CoxHazard("main")])
        assert isinstance(models.hazard_library, tuple)


# ------------------------------------------------------------------ routing


@pytest.mark.parametrize(
    ("option", "value", "extra"),
    [
        ("bootstrap_interval", "bca", {"n_bootstrap": 5, "cross_fit": False}),
        ("variance", "policy_weighted", {"cross_fit": False}),
        ("cluster_variance", "jackknife", {}),
        ("max_memory_bytes", 2**40, {}),
    ],
)
def test_a_longitudinal_option_reaches_ltmle(
    option: str, value: Any, extra: dict[str, Any]
) -> None:
    method = TMLEMethod().with_overrides(**{option: value}, **extra)
    assert method.estimator_kwargs(longitudinal=True)[option] == value
    _, effect = _longitudinal_effect()
    assert getattr(effect._longitudinal_engine(method), option) == value


@pytest.mark.parametrize(
    ("method_type", "option", "value", "extra"),
    [
        (SDRMethod, "bootstrap_interval", "bca", {"n_bootstrap": 5}),
        (SDRMethod, "variance", "policy_weighted", {"cross_fit": False}),
        (SDRMethod, "max_memory_bytes", 2**40, {}),
        (IPSIMethod, "bootstrap_interval", "bca", {"n_bootstrap": 5}),
        (IPSIMethod, "max_memory_bytes", 2**40, {}),
    ],
)
def test_an_option_reaches_sdr_and_ipsi(
    method_type: type[TMLEMethod], option: str, value: Any, extra: dict[str, Any]
) -> None:
    method = method_type().with_overrides(**{option: value}, **extra)
    kwargs = method.estimator_kwargs(longitudinal=True)
    assert kwargs[option] == value
    engine = (SDR if method_type is SDRMethod else IPSI)(
        {"x2": Incremental(2.0)} if method_type is IPSIMethod else REGIMENS, **kwargs
    )
    assert getattr(engine, option) == value


def test_the_jackknife_reaches_the_point_engine_and_meets_its_refusal() -> None:
    method = TMLEMethod().with_overrides(cluster_variance="jackknife")
    assert method.estimator_kwargs()["cluster_variance"] == "jackknife"
    with pytest.raises(CapabilityError, match="few-cluster-floor-tmle"):
        _point_effect()._point_engine(method)


def test_the_continuous_settings_reach_the_engine() -> None:
    hazards, censoring = (CoxHazard("main"),), (CoxHazard("treatment"),)
    method = TMLEMethod().with_overrides(
        hazard_library=hazards,
        censoring_library=censoring,
        norm="euclidean",
        variance="influence_curve",
        n_bootstrap=5,
        bootstrap_interval="bca",
        max_memory_bytes=2**40,
    )
    _, effect = _continuous_effect()
    engine = effect._continuous_engine(method)
    assert engine.hazard_library == hazards and engine.censoring_library == censoring
    assert (engine.norm, engine.variance, engine.bootstrap_interval) == (
        "euclidean",
        "influence_curve",
        "bca",
    )
    assert engine.max_memory_bytes == 2**40
    default = effect._continuous_engine(TMLEMethod())
    assert (default.norm, default.variance) == ("covariance", "switched")


def test_control_a_translation_that_drops_a_key_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: the routing assertion fails when the translation loses the key."""
    original = TMLEMethod.estimator_kwargs

    def dropping(self: TMLEMethod, *, longitudinal: bool = False) -> dict[str, Any]:
        kwargs = original(self, longitudinal=longitudinal)
        kwargs.pop("bootstrap_interval", None)
        return kwargs

    monkeypatch.setattr(TMLEMethod, "estimator_kwargs", dropping)
    with pytest.raises((KeyError, AssertionError)):
        test_a_longitudinal_option_reaches_ltmle(
            "bootstrap_interval", "bca", {"n_bootstrap": 5, "cross_fit": False}
        )


def test_policy_weighted_reads_no_band_unless_one_is_asked_for() -> None:
    method = TMLEMethod().with_overrides(variance="policy_weighted", cross_fit=False)
    assert method.estimator_kwargs(longitudinal=True)["simultaneous"] is False
    asked = method.with_overrides(simultaneous=True)
    assert asked.estimator_kwargs(longitudinal=True)["simultaneous"] is True
    _, effect = _longitudinal_effect()
    with pytest.raises(CapabilityError, match="variance='policy_weighted'"):
        effect._longitudinal_engine(asked)


# ------------------------------------------------------------------ parity with the engine


def _same(study_result: Any, direct: Any) -> None:
    assert study_result.estimates
    for name, estimate in study_result.estimates.items():
        other = direct.estimates[name]
        assert estimate.psi == other.psi
        assert estimate.std_error == other.std_error
        assert estimate.variance_source == other.variance_source
        assert estimate.support.label == other.support.label
        assert estimate.support.row == other.support.row
        if estimate.bootstrap is not None:
            assert estimate.bootstrap.interval == other.bootstrap.interval
            assert estimate.bootstrap.ci == other.bootstrap.ci


class TestParity:
    def test_bca_matches_the_engine_and_differs_from_the_percentile_interval(self) -> None:
        study, effect = _longitudinal_effect()
        settings = {**_learners(), "random_state": 0, "simultaneous": False, "n_bootstrap": 19}
        routed = _quiet(
            lambda: effect.estimate(cross_fit=False, bootstrap_interval="bca", **settings)
        )
        percentile = _quiet(lambda: effect.estimate(cross_fit=False, **settings))
        direct = _quiet(
            lambda: LTMLE(REGIMENS, n_folds=1, bootstrap_interval="bca", **settings).fit(study.data)
        )
        _same(routed, direct)
        moved = [
            name
            for name, estimate in routed.estimates.items()
            if estimate.bootstrap.interval == "bca"
            and percentile.estimates[name].bootstrap.interval == "percentile"
            and np.max(
                np.abs(np.subtract(estimate.bootstrap.ci, percentile.estimates[name].bootstrap.ci))
            )
            > 1e-6
        ]
        assert moved

    def test_policy_weighted_matches_the_engine_and_lifts_an_estimate(self) -> None:
        tau = 2
        frame = common.sample(600, 11, tau)
        study = CausalStudy(frame, design=LongitudinalTreatment(outcome="Y", **pricing.roles(tau)))
        effect = study.identify(
            RegimeContrast(common.regimens(pricing.LABELS, tau), reference="natural_prices")
        )
        settings = {
            "outcome_learner": QuasiBinomialGLM(),
            "pseudo_learner": LinearRegression(),
            "treatment_learner": common.OracleRatio(),
            "g_bounds": pricing.G_BOUNDS,
            "ratio": "classifier",
            "random_state": 0,
        }
        routed = _quiet(
            lambda: effect.estimate(cross_fit=False, variance="policy_weighted", **settings)
        )
        direct = _quiet(
            lambda: LTMLE(
                common.regimens(pricing.LABELS, tau),
                reference="natural_prices",
                n_folds=1,
                simultaneous=False,
                variance="policy_weighted",
                **settings,
            ).fit(frame, outcome="Y", **pricing.roles(tau))
        )
        _same(routed, direct)
        assert any(e.variance_source == "policy_weighted" for e in routed.estimates.values())

    def test_the_jackknife_matches_the_engine_and_moves_the_standard_error(self) -> None:
        study, effect = _longitudinal_effect(cluster=True)
        settings = {**_learners(), "random_state": 5, "simultaneous": False}
        routed = _quiet(
            lambda: effect.estimate(cross_fit=False, cluster_variance="jackknife", **settings)
        )
        default = _quiet(lambda: effect.estimate(cross_fit=False, **settings))
        direct = _quiet(
            lambda: LTMLE(REGIMENS, n_folds=1, cluster_variance="jackknife", **settings).fit(
                study.data
            )
        )
        _same(routed, direct)
        assert all(
            abs(e.std_error - default.estimates[n].std_error) > 1e-6
            for n, e in routed.estimates.items()
        )

    def test_the_continuous_settings_match_the_engine_and_each_one_moves_the_fit(self) -> None:
        study, effect = _continuous_effect()
        base = {"random_state": 0, "cross_fit": False, "n_multiplier": 500}
        default = _quiet(lambda: effect.estimate(**base))

        def routed(**changes: Any) -> Any:
            return _quiet(lambda: effect.estimate(**base, **changes))

        def moved(result: Any) -> bool:
            return any(
                abs(e.psi - default.estimates[n].psi) > 1e-9 for n, e in result.estimates.items()
            )

        assert moved(routed(norm="euclidean"))
        assert moved(routed(hazard_library=(CoxHazard("treatment"),)))
        # At t = 0.2 the influence-curve variance is sparse, so the default switched variance
        # raises a standard error there; the routed influence-curve variance raises none.
        assert any(
            e.switched is not None
            and e.switched.predictable_variance > e.switched.influence_variance
            for e in default.estimates.values()
        )
        ic = routed(variance="influence_curve")
        assert any(
            e.std_error < default.estimates[n].std_error - 1e-9 for n, e in ic.estimates.items()
        )
        assert all(e.switched is None for e in ic.estimates.values())
        settings = {
            "hazard_library": (CoxHazard("main"),),
            "censoring_library": (CoxHazard("treatment"),),
            "norm": "euclidean",
            "variance": "influence_curve",
        }
        both = routed(**settings)
        direct = _quiet(
            lambda: ContinuousSurvivalTMLE(
                (0.2, 1.0),
                arms={"t": 1, "c": 0},
                reference="c",
                cross_fit=False,
                random_state=0,
                n_multiplier=500,
                simultaneous=True,
                **settings,
            ).fit_data(study.data)
        )
        _same(both, direct)


# ------------------------------------------------------------------ refusals


def _refuses(call: Callable[[], Any], error: type[Exception], match: str) -> CleverlyError:
    with pytest.raises(error, match=match) as caught:
        call()
    assert isinstance(caught.value, CleverlyError)
    return caught.value


class TestRefusals:
    def test_the_point_route_refuses_what_its_engine_lacks(self) -> None:
        effect = _point_effect()
        untouched = {
            "outcome_learner": LinearRegression(),
            "treatment_learner": LogisticRegression(max_iter=1000),
        }
        for overrides, match, reason in (
            ({"variance": "influence_curve"}, "takes no variance choice", "api_redirection"),
            (
                {"n_bootstrap": 5, "bootstrap_interval": "bca"},
                "percentile interval only",
                "api_redirection",
            ),
            ({"max_memory_bytes": 2**30}, "runs no memory guard", "invalid_declaration"),
            ({"norm": "euclidean"}, "fits no universal submodel", "invalid_declaration"),
        ):
            error = _refuses(
                lambda o=overrides: effect.estimate(**o, **untouched),
                MethodConfigurationError,
                match,
            )
            assert error.refusal.reason == reason

    def test_ep_refuses_the_budget_through_the_point_translation(self) -> None:
        method = EPMethod().with_overrides(max_memory_bytes=2**30)
        _refuses(method.estimator_kwargs, MethodConfigurationError, "runs no memory guard")

    def test_a_binned_fit_refuses_the_continuous_settings(self) -> None:
        method = TMLEMethod().with_overrides(hazard_library=(CoxHazard("main"),))
        _refuses(
            lambda: method.estimator_kwargs(longitudinal=True),
            MethodConfigurationError,
            r"fits no universal submodel: \['hazard_library'\]",
        )

    def test_the_variance_values_belong_to_their_route(self) -> None:
        _refuses(
            lambda: (
                TMLEMethod().with_overrides(variance="switched").estimator_kwargs(longitudinal=True)
            ),
            MethodConfigurationError,
            "A binned longitudinal fit takes",
        )
        _refuses(
            lambda: continuous_kwargs(TMLEMethod().with_overrides(variance="policy_weighted")),
            MethodConfigurationError,
            "contrasts two static arms",
        )

    def test_control_mapping_switched_to_the_default_would_be_caught(self) -> None:
        """Mutation control: the test asserts the raise, not the value the engine gets."""
        method = TMLEMethod().with_overrides(variance="switched")
        with pytest.raises(MethodConfigurationError):
            method.estimator_kwargs(longitudinal=True)
        assert method.inference.variance == "switched"

    def test_policy_weighted_with_cross_fitting_names_the_method_spelling(self) -> None:
        _, effect = _longitudinal_effect()
        error = _refuses(
            lambda: effect.estimate(variance="policy_weighted", outcome_learner=LinearRegression()),
            MethodConfigurationError,
            "fold-wise variance regressions",
        )
        assert "cross_fit=False" in str(error.refusal.alternative)

    @pytest.mark.parametrize("method_type", [SDRMethod, IPSIMethod])
    def test_sdr_and_ipsi_refuse_the_jackknife_at_construction(
        self, method_type: type[TMLEMethod]
    ) -> None:
        error = _refuses(
            lambda: method_type(inference=Inference(cluster_variance="jackknife")),
            MethodConfigurationError,
            "cluster_variance='jackknife' refits every nuisance",
        )
        assert error.refusal.reason == "inference_gap"

    def test_ipsi_refuses_the_policy_weighted_variance_on_both_layers(self) -> None:
        _refuses(
            lambda: IPSIMethod(inference=Inference(variance="policy_weighted")),
            MethodConfigurationError,
            "adds a term for the estimated treatment mechanism",
        )
        _refuses(
            lambda: IPSI({"x2": Incremental(2.0)}, variance="policy_weighted"),
            CapabilityError,
            "adds a term for the estimated treatment mechanism",
        )

    @pytest.mark.parametrize(
        "method_type", [GComputationMethod, IPWMethod, ParametricGFormulaMethod]
    )
    @pytest.mark.parametrize(
        ("inference", "runtime", "match"),
        [
            (Inference(cluster_variance="jackknife"), Runtime(), "ships for LTMLE"),
            (Inference(variance="policy_weighted"), Runtime(), "would ignore the setting"),
            (
                Inference(n_bootstrap=5, bootstrap_interval="bca"),
                Runtime(),
                "bootstrap standard deviation only",
            ),
            (Inference(), Runtime(max_memory_bytes=2**30), "runs no memory guard"),
        ],
    )
    def test_each_baseline_refuses_at_construction(
        self,
        method_type: type[TMLEMethod],
        inference: Inference,
        runtime: Runtime,
        match: str,
    ) -> None:
        _refuses(
            lambda: method_type(inference=inference, runtime=runtime),
            MethodConfigurationError,
            match,
        )

    def test_control_the_g_formula_runs_the_shared_construction_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: without the ``super()`` call the construction check is skipped,
        and the translation still refuses."""
        monkeypatch.setattr(
            ParametricGFormulaMethod,
            "__post_init__",
            lambda self: None,
        )
        method = ParametricGFormulaMethod(inference=Inference(variance="policy_weighted"))
        _refuses(
            lambda: method.estimator_kwargs(longitudinal=True),
            MethodConfigurationError,
            "would ignore the setting",
        )

    @pytest.mark.parametrize(
        ("engine", "overrides", "error", "match"),
        [
            (
                SequentialGComputation,
                {"variance": "policy_weighted"},
                CapabilityError,
                "would ignore",
            ),
            (SequentialIPW, {"variance": "policy_weighted"}, CapabilityError, "would ignore"),
            (ParametricGFormula, {"variance": "policy_weighted"}, CapabilityError, "would ignore"),
            (
                SequentialGComputation,
                {"n_bootstrap": 5, "bootstrap_interval": "bca"},
                CapabilityError,
                "bootstrap standard deviation only",
            ),
            (
                SurvivalCTMLE,
                {"max_memory_bytes": 2**30},
                MethodConfigurationError,
                "SurvivalCTMLE runs",
            ),
            (
                SurvivalDRTMLE,
                {"max_memory_bytes": 2**30},
                MethodConfigurationError,
                "SurvivalDRTMLE runs",
            ),
        ],
    )
    def test_an_engine_refuses_a_setting_it_would_ignore(
        self, engine: type, overrides: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        _refuses(lambda: engine(REGIMENS, **overrides), error, match)

    @pytest.mark.parametrize("method_type", [CollaborativeTMLEMethod, DRTMLEMethod])
    def test_the_survival_methods_refuse_the_budget(self, method_type: type[TMLEMethod]) -> None:
        method = method_type().with_overrides(max_memory_bytes=2**30)
        _refuses(
            lambda: method.estimator_kwargs(longitudinal=True),
            MethodConfigurationError,
            "runs no memory guard",
        )

    def test_an_engine_only_setting_raises_its_own_refusal(self) -> None:
        error = _refuses(
            lambda: TMLEMethod().with_overrides(restricted_horizons=(1.0,)),
            MethodConfigurationError,
            "declares targets, not a method setting",
        )
        assert error.refusal.reason == "api_redirection"
        assert "result.rmst(d, tau)" in str(error.refusal.alternative)
        error = _refuses(
            lambda: TMLEMethod().with_overrides(tilt="factual"),
            MethodConfigurationError,
            "mutation control",
        )
        assert error.refusal.reason == "invalid_declaration"

    def test_the_restricted_mean_redirect_runs(self) -> None:
        study, effect = _continuous_effect(horizons=(0.5, 1.0))
        method = TMLEMethod().with_overrides(random_state=0, cross_fit=False, n_multiplier=500)
        direct = _quiet(
            lambda: ContinuousSurvivalTMLE(
                (0.5, 1.0),
                restricted_horizons=(1.0,),
                arms=effect.functional.interventions,
                reference=effect.functional.reference,
                **continuous_kwargs(method),
            ).fit_data(study.data)
        )
        assert np.isfinite(direct.rmst(effect.functional.reference, 1.0).psi)

    def test_the_curve_keeps_its_own_budget(self) -> None:
        """The curve's own ``max_memory_bytes`` keyword never meets the point route refusal."""
        from tests.unit.test_targeting_curves import _route_one

        assert _quiet(lambda: _route_one(max_memory_bytes=2**40)) is not None


def test_an_ipsi_bca_fit_reads_experimental() -> None:
    frame, _ = make_longitudinal(n=200, seed=0, censoring=False)
    result = _quiet(
        lambda: IPSI(
            [Incremental(2.0)],
            n_folds=1,
            n_bootstrap=5,
            bootstrap_interval="bca",
            simultaneous=False,
            random_state=0,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )
    )
    estimate = result.estimates["ey_regimen[odds x2]"]
    assert estimate.bootstrap.interval == "bca"
    assert estimate.support.label == "experimental"
    assert "bootstrap_interval" in inspect.signature(IPSI).parameters


# ------------------------------------------------------------------ the sparse advice


def _sparse_effect() -> Any:
    frame = law_.draw(3, 2, 600, 9, law_.L_R)
    design = LongitudinalTreatment(**law_.roles(3, 2), **law_.probability_columns(3, 2))
    study = CausalStudy(frame, design=design)
    return study.identify(RegimeMean(plans_.regimens(("low", "high", "respond"), 3, 2)))


SPARSE = {
    "outcome_learner": GLM(),
    "pseudo_learner": GLM(),
    "g_bounds": (1e-8, 1.0),
    "random_state": 0,
}


def test_the_sparse_advice_names_a_call_that_runs_on_the_study_route() -> None:
    effect = _sparse_effect()
    with pytest.warns(SparseInfluenceWarning) as record:
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("always", category=SparseInfluenceWarning)
        effect.estimate("tmle", cross_fit=False, **SPARSE)
    text = " ".join(
        str(w.message) for w in record if issubclass(w.category, SparseInfluenceWarning)
    )
    assert "n_bootstrap=199, bootstrap_interval='bca'" in text
    assert "Inference(n_bootstrap=199, bootstrap_interval='bca') on a method object" in text
    # The advice, at 19 draws for speed, runs as flat keywords and as a method object, and the
    # refit no longer warns.
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=GLM(), pseudo_learner=GLM()),
        targeting=Targeting(g_bounds=(1e-8, 1.0)),
        inference=Inference(n_bootstrap=19, bootstrap_interval="bca"),
        runtime=Runtime(random_state=0),
    ).with_overrides(cross_fit=False)
    for refit in (
        lambda: effect.estimate(
            "tmle", cross_fit=False, n_bootstrap=19, bootstrap_interval="bca", **SPARSE
        ),
        lambda: effect.estimate(method),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", SparseInfluenceWarning)
            result = refit()
        assert next(iter(result.estimates.values())).bootstrap.interval == "bca"


def test_control_without_the_shortcut_the_advised_refit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: the advice is only as good as the shortcut it names."""
    shortcuts = {group: dict(fields) for group, fields in SHORTCUTS.items()}
    del shortcuts["inference"]["bootstrap_interval"]
    monkeypatch.setattr(methods_module, "SHORTCUTS", shortcuts)
    with pytest.raises(MethodConfigurationError, match="unsupported tmle option"):
        _sparse_effect().estimate(
            "tmle", cross_fit=False, n_bootstrap=19, bootstrap_interval="bca", **SPARSE
        )


# ------------------------------------------------------------------ engine reach


#: The engine arguments a study sets from the design or the estimand, not from the method.
_STUDY_SET = {
    "regimens",
    "reference",
    "horizons",
    "msm",
    "arms",
    "causes",
    "estimands",
    "interventions",
    "learned_rule",
    "policies",
    "incremental",
    "effect_modification",
    "family",
}


@pytest.mark.parametrize(
    ("engine", "translated"),
    [
        (TMLE, lambda: TMLEMethod().estimator_kwargs()),
        (LTMLE, lambda: TMLEMethod().estimator_kwargs(longitudinal=True)),
        (SDR, lambda: SDRMethod().estimator_kwargs(longitudinal=True)),
        (IPSI, lambda: IPSIMethod().estimator_kwargs(longitudinal=True)),
        (ContinuousSurvivalTMLE, lambda: continuous_kwargs(TMLEMethod())),
    ],
)
def test_every_engine_parameter_is_reached_set_by_the_study_or_listed(
    engine: type, translated: Callable[[], dict[str, Any]]
) -> None:
    """The drift that left options engine-only cannot return unnoticed."""
    parameters = {
        name
        for name, parameter in inspect.signature(engine).parameters.items()
        if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }
    # The continuous translation leaves None-valued settings to the engine's defaults.
    reached = (
        set(translated()) | {"variance", "norm"}
        if engine is ContinuousSurvivalTMLE
        else set(translated())
    )
    unreached = sorted(parameters - reached - _STUDY_SET - set(ENGINE_ONLY))
    assert unreached == []
