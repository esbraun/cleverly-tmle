"""The support, status, summary and persistence of the K-arm and weighted per-arm stack (X58b2).

X58 plan, 13. The predicate admits any arm-coded treatment, observation weights, and the
targets ``ey_obs``, ``par`` and ``paf``. Such a fit matches no support row, so its estimates
are labelled ``"experimental"``. TMLE and CTMLE refuse a
:class:`~cleverly.learners.ParametricOutcome` whose ``arm_columns`` misread the treatment
block, wherever the outcome learner holds it (Q8). The witnesses of the stack itself live in
``test_oat_multi_arm_stack.py`` and ``test_oat_population_weighted.py``.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import cleverly
from cleverly.estimators import CTMLE, TMLE, _oat_stack
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators.ctmle import _held_estimates, per_arm_parametric_support
from cleverly.exceptions import (
    CapabilityError,
    DataWarning,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.learners import ParametricMechanism, ParametricOutcome, SuperLearner
from tests.studies import oat_per_arm_laws as laws
from tests.unit._inference_status_support import ROUTES, assert_round_trips, restore

pytestmark = pytest.mark.xdist_group("oat_per_arm_k_arm_support")

tmle_module = importlib.import_module("cleverly.estimators.tmle")

COVARIATES = ["W1", "W2", "W3"]

FOUR_ARM = laws.OatLaw(
    name="four_arm_active",
    c=(-0.4, 0.0, 0.4, 0.2),
    u=((0.7, 0.7), (0.7, -0.7), (-0.8, 0.5), (0.4, 0.3)),
    v=((0.0, 0.0), (0.6, 0.7), (-0.3, -0.7), (0.4, -0.4)),
)
"""A four-arm law with every arm well represented (the smallest ``g`` stays near 0.1)."""


def _method_text(result: Any) -> str:
    """The Method details of a full summary, as one string."""
    return " ".join(result.summary(detail="full").section("Method details").lines)


def _options(**overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(arm_columns=2),
        "treatment_learner": ParametricMechanism(),
        "cross_fit": False,
        "simultaneous": False,
        "estimands": ("ey", "ate"),
        "g_bounds": laws.G_BOUNDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    options.update(overrides)
    return options


def _fit(
    frame: pd.DataFrame,
    weights: str | None = None,
    *,
    weights_estimated: bool = False,
    **overrides: Any,
) -> Any:
    roles: dict[str, Any] = {"weights": weights}
    if weights_estimated:
        roles["weights_estimated"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**_options(**overrides))
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, **roles)
            .single()
        )


def _prepared(estimator: Any, frame: pd.DataFrame, **roles: Any) -> Any:
    return estimator._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        delta=None,
        weights=roles.pop("weights", None),
        id=None,
        intermediate=None,
        strata=None,
        **roles,
    )


def _three_arm(n: int = 1500, seed: int = 111) -> pd.DataFrame:
    return laws.THREE_ARM_ACTIVE.sample(n, seed)[0]


def _binary(n: int = 1500, seed: int = 113) -> pd.DataFrame:
    return laws.BINARY_ACTIVE.sample(n, seed)[0]


def _weighted(n: int = 1500, seed: int = 115) -> pd.DataFrame:
    return laws.BINARY_WEIGHTED.sample(n, seed)[0]


def _failing(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    original = _oat_stack._arm_failure

    def mutant(stack: Any, rconds: dict[str, float], j: int) -> str | None:
        return "forced" if j == arm else original(stack, rconds, j)

    monkeypatch.setattr(_oat_stack, "_arm_failure", mutant)


@pytest.fixture(scope="module")
def three_arm() -> Any:
    return _fit(_three_arm())


# --------------------------------------------------------------------------- K arms


class TestTheArms:
    def test_three_arms_are_admitted_with_their_detail(self) -> None:
        estimator = CTMLE(**_options())
        verdict = per_arm_parametric_support(estimator, _prepared(estimator, _three_arm()))
        assert verdict.admitted
        assert verdict.detail.endswith("on a 3-arm treatment")

    def test_the_fit_records_three_stacked_arms(self, three_arm: Any) -> None:
        record = three_arm.extra["ctmle"]
        assert record.stacked_arms == 3
        assert {estimate.inference for estimate in three_arm.estimates.values()} == {
            "influence_curve"
        }

    def test_a_three_arm_fit_is_supported(self, three_arm: Any) -> None:
        for estimate in three_arm.estimates.values():
            assert estimate.support.label == "supported"
            assert np.isfinite(estimate.std_error)

    def test_four_arms_are_admitted_and_experimental(self) -> None:
        result = _fit(
            FOUR_ARM.sample(2000, 117)[0], outcome_learner=ParametricOutcome(arm_columns=3)
        )
        assert result.extra["ctmle"].stacked_arms == 4
        assert set(result.estimates) == {
            "ey[0]",
            "ey[1]",
            "ey[2]",
            "ey[3]",
            "ate[1 vs 0]",
            "ate[2 vs 0]",
            "ate[3 vs 0]",
        }
        for estimate in result.estimates.values():
            assert estimate.inference == "influence_curve"
            assert estimate.support.label == "experimental"

    def test_the_default_estimands_are_judged_as_resolved(self) -> None:
        """``estimands=None`` resolves to ``ate``, ``ey``, ``rr`` and ``or`` on three arms."""
        result = _fit(_three_arm(), estimands=None)
        assert {name.split("[")[0] for name in result.estimates} == {"ate", "ey", "rr", "or"}
        assert {estimate.inference for estimate in result.estimates.values()} == {"influence_curve"}

    def test_binary_default_estimands_are_refused_before_the_predicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A binary ``estimands=None`` holds ``att``, which CTMLE refuses at its own check."""
        calls: list[int] = []
        original = ctmle_module.per_arm_parametric_support

        def recording(estimator: Any, data: Any) -> Any:
            calls.append(1)
            return original(estimator, data)

        monkeypatch.setattr(ctmle_module, "per_arm_parametric_support", recording)
        with pytest.raises(CapabilityError, match="ATT and ATC clever"):
            _fit(_binary(), estimands=None, outcome_learner=ParametricOutcome())
        assert calls == []

    def test_a_retarget_to_att_keeps_the_generated_design_status(self) -> None:
        estimator = CTMLE(**_options())
        data = _prepared(estimator, _three_arm())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(data).single()
            out, _ = estimator.retarget(data, fit.nuisance, estimands=("att",))
        assert {estimate.inference for estimate in out.values()} == {"generated_design_plugin"}


# --------------------------------------------------------------------------- Q8


def _mismatched_learners() -> dict[str, Any]:
    return {
        "bare": ParametricOutcome(arm_columns=1),
        "library": SuperLearner([LogisticRegression(), ParametricOutcome(arm_columns=1)]),
        "pipeline": Pipeline([("model", ParametricOutcome(arm_columns=1))]),
    }


def _tmle(frame: pd.DataFrame, learner: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(
            outcome_learner=learner,
            treatment_learner=LogisticRegression(),
            estimands=("ey",),
            cross_fit=False,
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)


class TestTheArmColumnPreflight:
    """Q8: a ``ParametricOutcome`` whose arm columns misread the treatment block is refused."""

    @pytest.mark.parametrize("where", ["bare", "library", "pipeline"])
    def test_tmle_refuses_it_wherever_it_is(self, where: str) -> None:
        with pytest.raises(MethodConfigurationError, match="arm_columns=2") as raised:
            _tmle(_three_arm(), _mismatched_learners()[where])
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    @pytest.mark.parametrize("where", ["bare", "library", "pipeline"])
    def test_ctmle_refuses_it_wherever_it_is(self, where: str) -> None:
        with pytest.raises(MethodConfigurationError, match="does not match the treatment"):
            _fit(_three_arm(), outcome_learner=_mismatched_learners()[where])

    def test_a_binary_treatment_refuses_two_columns(self) -> None:
        with pytest.raises(MethodConfigurationError, match="which has 1 column"):
            _tmle(_binary(), ParametricOutcome(arm_columns=2))

    def test_a_matching_declaration_passes(self) -> None:
        result = _tmle(
            _three_arm(), SuperLearner([LogisticRegression(), ParametricOutcome(arm_columns=2)])
        )
        assert len(result.single().estimates) == 3

    def test_mutation_no_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the preflight the mismatched library fits silently."""
        monkeypatch.setattr(
            tmle_module.TMLE, "_refuse_mismatched_arm_columns", lambda self, data: None
        )
        result = _tmle(_three_arm(), _mismatched_learners()["library"])
        assert len(result.single().estimates) == 3


def _vector_frame(*, dose: bool, n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.integers(0, 2, n)
    a2 = rng.normal(size=n) if dose else rng.integers(0, 2, n)
    logit = 0.3 * a1 - 0.2 * a2 + 0.5 * w
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    second = a2 if dose else a2.astype(str)
    return pd.DataFrame({"W1": w, "A1": a1.astype(str), "A2": second, "Y": y})


def _vector_tmle(frame: pd.DataFrame, arm_columns: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(
            outcome_learner=ParametricOutcome(arm_columns=arm_columns),
            treatment_learner=LogisticRegression(),
            estimands=("ey",),
            simultaneous=False,
            cross_fit=False,
            random_state=0,
        ).fit(frame, outcome="Y", treatment=["A1", "A2"], covariates=["W1"])


class TestTheVectorTreatmentWidth:
    """The width is the treatment block the outcome design really has."""

    def test_a_two_by_two_categorical_vector_has_three_joint_columns(self) -> None:
        frame = _vector_frame(dose=False)
        result = _vector_tmle(frame, 3)
        assert len(result.single().estimates) == 4

    def test_a_two_by_two_categorical_vector_refuses_two_columns(self) -> None:
        with pytest.raises(MethodConfigurationError, match=r"which has 3 column") as raised:
            _vector_tmle(_vector_frame(dose=False), 2)
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "arm_columns=3" in str(raised.value.refusal.alternative)

    def test_a_dose_vector_has_one_column_per_component(self) -> None:
        """A vector with a dose reads its component blocks: 1 label column and 1 dose."""
        estimator = TMLE(
            outcome_learner=ParametricOutcome(arm_columns=2), treatment_learner=LogisticRegression()
        )
        data = estimator._prepare(
            _vector_frame(dose=True),
            outcome="Y",
            treatment=["A1", "A2"],
            covariates=["W1"],
            continuous_treatment=["A2"],
            delta=None,
            weights=None,
            id=None,
            intermediate=None,
            strata=None,
        )
        assert data.treatment_kind == "vector"
        estimator._refuse_mismatched_arm_columns(data)
        estimator.outcome_learner = ParametricOutcome(arm_columns=3)
        with pytest.raises(MethodConfigurationError, match=r"which has 2 column"):
            estimator._refuse_mismatched_arm_columns(data)


# --------------------------------------------------------------------------- the targets


class TestThePopulationTargets:
    def test_ey_obs_par_and_paf_take_the_stacked_curve(self) -> None:
        result = _fit(_three_arm(), estimands=("ey", "ey_obs", "par", "paf"))
        assert {"ey_obs", "par[0]", "paf[0]"} <= set(result.estimates)
        for estimate in result.estimates.values():
            assert estimate.inference == "influence_curve"
            assert np.isfinite(estimate.std_error)


# --------------------------------------------------------------------------- the band


class TestTheBand:
    def test_the_band_is_built_and_supported(self) -> None:
        result = _fit(_three_arm(), simultaneous=True)
        assert result.simultaneous is not None
        assert set(result.simultaneous.bands) == set(result.estimates)
        assert result.simultaneous.support.label == "supported"

    def test_one_held_estimate_builds_no_band_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _failing(monkeypatch, 1)
        result = _fit(
            _binary(),
            outcome_learner=ParametricOutcome(),
            estimands=("ey0", "ey1", "ate"),
            simultaneous=True,
        )
        assert result.simultaneous is None
        assert result["ey0"].inference == "influence_curve"
        settings = result.summary(detail="full").section("Settings").lines
        assert (
            "No simultaneous band: fewer than two rows have an interval. Left out: ate, ey1."
            in (settings)
        )

    def test_a_sound_fit_without_a_band_request_says_nothing(self) -> None:
        result = _fit(_binary(), outcome_learner=ParametricOutcome(), estimands=("ey0", "ate"))
        assert "No simultaneous band" not in result.summary(detail="full")


# --------------------------------------------------------------------------- the held set


class TestTheHeldSet:
    def test_an_unrecorded_estimate_reads_every_arm(self) -> None:
        estimates = {"ey[0]": None, "ey[1]": None, "custom": None}
        reads = {"ey[0]": frozenset({0.0}), "ey[1]": frozenset({1.0})}
        held = _held_estimates(estimates, reads, {1.0: "forced"})
        assert held == frozenset({"ey[0]"})

    def test_an_estimate_that_reads_no_arm_is_held(self) -> None:
        estimates = {"ey_obs": None, "ey[1]": None}
        reads = {"ey_obs": frozenset(), "ey[1]": frozenset({1.0})}
        assert _held_estimates(estimates, reads, {1.0: "forced"}) == frozenset({"ey_obs"})


# --------------------------------------------------------------------------- weights


class TestTheWeights:
    def test_the_summary_states_the_weights(self) -> None:
        fixed = _fit(_weighted(), "wt", outcome_learner=ParametricOutcome())
        text = _method_text(fixed)
        assert "with the data weights in every equation" in text
        assert "which are declared estimated" not in text
        estimated = _fit(
            _weighted(), "wt", outcome_learner=ParametricOutcome(), weights_estimated=True
        )
        assert "The interval conditions on the weights, which are declared estimated" in (
            _method_text(estimated)
        )

    def test_constant_weights_equal_the_unweighted_stack(self) -> None:
        frame = _binary().assign(one=3.0)
        weighted = _fit(frame, "one", outcome_learner=ParametricOutcome())
        plain = _fit(frame, outcome_learner=ParametricOutcome())
        for name, estimate in plain.estimates.items():
            assert weighted[name].psi == pytest.approx(estimate.psi, rel=1e-10, abs=1e-12)
            assert weighted[name].std_error == pytest.approx(estimate.std_error, rel=1e-9)


# --------------------------------------------------------------------------- persistence


class TestPersistence:
    @pytest.mark.parametrize("route", ROUTES)
    def test_a_three_arm_fit_round_trips(self, three_arm: Any, route: str) -> None:
        restored = assert_round_trips(three_arm, "influence_curve", route)
        assert restored.extra["ctmle"].stacked_arms == 3
        assert restored.summary() == three_arm.summary()

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_failed_arm_round_trips(self, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
        _failing(monkeypatch, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DataWarning)
            result = _fit(_three_arm())
        restored = restore(result, route)
        statuses = {name: estimate.inference for name, estimate in result.estimates.items()}
        assert statuses["ey[2]"] == "stack_failure_plugin"
        assert statuses["ey[1]"] == "influence_curve"
        assert {name: e.inference for name, e in restored.estimates.items()} == statuses
        assert restored.summary() == result.summary()

    @pytest.mark.parametrize(
        ("estimated", "status"),
        [(False, "influence_curve"), (True, "estimated_weight_plugin")],
    )
    def test_a_weighted_fit_round_trips(self, estimated: bool, status: str) -> None:
        result = _fit(
            _weighted(), "wt", outcome_learner=ParametricOutcome(), weights_estimated=estimated
        )
        for route in ROUTES:
            restored = assert_round_trips(result, status, route)
            assert restored.summary() == result.summary()

    def test_save_and_load_keep_the_band(self, tmp_path: Any) -> None:
        result = _fit(_three_arm(), simultaneous=True)
        loaded = cleverly.load(result.save(tmp_path / "fit.pkl"))
        assert loaded.simultaneous is not None
        assert loaded.simultaneous.bands == result.simultaneous.bands
