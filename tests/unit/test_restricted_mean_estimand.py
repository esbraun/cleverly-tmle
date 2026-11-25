"""Restricted means declared on the estimand: ``restricted_horizons=`` on RegimeMean/RegimeContrast.

On a binned grid the targeted fit stores each regimen's restricted mean survival time (RMST)
and restricted mean time lost (RMTL) to each cause, read from its targeted risks at every grid
time below the restriction time ``tau``. By linearity that reading is the TMLE of the
restricted mean. With exact times the continuous-time walk targets them
(``tests/unit/test_method_engine_options.py`` checks that route against the engine).

Every component that can vanish at the truth has a nonzero witness or a mutation control: the
grid spacings (an uneven grid, and a shifted-spacing mutation), the sign of ``versus`` (a law
with a treatment effect, and a reversed-sign mutation), the cause index (two causes with
different incidences, and a swapped-cause mutation), the eliminated mask, the node index in
the keys, the refit template and the truncation replay.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import CausalStudy, PointTreatment, RegimeContrast, RegimeMean, TimeToEvent
from cleverly.datasets import make_linear_ate, make_longitudinal, make_longitudinal_competing
from cleverly.exceptions import CapabilityError, CleverlyError, DataError, MethodConfigurationError
from cleverly.longitudinal import LTMLE, SDR, SurvivalDRTMLE
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.estimator import LongitudinalResult, _SequentialEstimator
from cleverly.methods import ENGINE_ONLY, TMLEMethod
from tests import discrete_law_longitudinal as end_law
from tests.studies import canonical_point_survival as point_study
from tests.studies import survival_grid_law as grid_law

UNEVEN = (1.0, 2.5, 4.0, 7.0)
UNEVEN_FIVE = (1.0, 2.5, 4.0, 7.0, 9.0)
PLANS = {"one": 1, "zero": 0}
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _expit(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=float)


def _frame(grid: tuple[float, ...], n: int = 500, seed: int = 0, causes: int = 1) -> pd.DataFrame:
    """Event and censoring times on ``grid``, with a treatment effect on every cause.

    Treatment lowers the hazard of cause 1 and raises that of cause 2, so the incidences of
    the two causes differ and the RMST differs between the arms.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, _expit(0.4 * w))
    time = np.full(n, grid[-1] + 1.0)
    event = np.zeros(n, dtype=int)
    open_ = np.ones(n, dtype=bool)
    for k, g in enumerate(grid):
        first = _expit(-1.7 + 0.5 * w - 0.7 * a + 0.1 * k)
        second = _expit(-2.6 + 0.3 * w + 0.6 * a) if causes == 2 else np.zeros(n)
        draw = rng.random(n)
        one = open_ & (draw < first)
        two = open_ & ~one & (draw < first + second)
        for mask, code in ((one, 1), (two, 2)):
            time[mask], event[mask] = g, code
        open_ &= ~(one | two)
        censored = open_ & (rng.random(n) < 0.04)
        time[censored] = g
        open_ &= ~censored
    return pd.DataFrame({"time": time, "event": event, "A": a, "W": w})


def _design(grid: tuple[float, ...], **settings: Any) -> TimeToEvent:
    return TimeToEvent(
        time="time", event="event", treatment="A", baseline=("W",), grid=grid, **settings
    )


LEARNERS: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "pseudo_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "censoring_learner": LogisticRegression(max_iter=1000),
    "competing_learner": LogisticRegression(max_iter=1000),
}


def _fit_settings(**settings: Any) -> dict[str, Any]:
    return {**LEARNERS, "cross_fit": False, "random_state": 0, "simultaneous": False, **settings}


def _estimate(
    estimand: Any, grid: tuple[float, ...] = UNEVEN, n_causes: int = 1, **settings: Any
) -> Any:
    design_settings = {key: settings.pop(key) for key in ("causes", "eliminate") if key in settings}
    frame = _frame(grid, causes=n_causes)
    study = CausalStudy(frame, design=_design(grid, **design_settings))
    return _quiet(lambda: study.estimate(estimand, **_fit_settings(**settings)))


def _engine(**settings: Any) -> LTMLE:
    configured: dict[str, Any] = {
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 0,
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }
    return LTMLE(PLANS, reference="zero", **{**configured, **settings})


def _container(grid: tuple[float, ...] = UNEVEN) -> Any:
    return CausalStudy(_frame(grid), design=_design(grid)).data


def _risk(result: Any, regimen: str, node: int, cause: str | None = None) -> str:
    """The reported risk name of ``regimen`` at node ``node``, read from the index."""
    for name, key in (result.parameter_index or {}).items():
        if key == (regimen, cause, node) and result.estimates[name].scale == "level":
            return name
    raise KeyError((regimen, cause, node))


def _from_risks(result: Any, name: str) -> Any:
    """A stored restricted mean computed again from the risks, outside the stored lookup."""
    unstored = replace(result, restricted_index={})
    regimen, versus, cause, node, measure = result.restricted_index[name]
    tau = unstored._restriction_time(node)
    if measure == "rmst":
        return unstored.rmst(regimen, tau, versus=versus)
    return unstored.rmtl(regimen, tau, cause, versus=versus)


@pytest.fixture(scope="module")
def uneven_mean() -> Any:
    return _estimate(RegimeMean(PLANS, reference="zero", restricted_horizons=(7.0,)))


@pytest.fixture(scope="module")
def uneven_contrast() -> Any:
    return _estimate(RegimeContrast(PLANS, reference="zero", restricted_horizons=(7.0,)))


# ------------------------------------------------------------------ 1. the spacing


def _linearity_holds(result: Any) -> bool:
    """The stored RMST is 7 - 1.5 F(1) - 1.5 F(2.5) - 3 F(4), value and curve."""
    weights = {1: 1.5, 2: 1.5, 3: 3.0}
    risks = {node: result.estimates[_risk(result, "one", node)] for node in weights}
    stored = result.estimates["rmst_regimen[one @ t=4]"]
    value = 7.0 - sum(weights[node] * risks[node].psi for node in weights)
    curve = -sum(weights[node] * risks[node].influence_curve for node in weights)
    return bool(
        abs(stored.psi - value) < 1e-12
        and np.allclose(stored.influence_curve, curve, atol=1e-12, rtol=0)
    )


class TestTheSpacing:
    def test_the_stored_rmst_weights_each_risk_by_its_gap(self, uneven_mean: Any) -> None:
        assert _linearity_holds(uneven_mean)
        assert uneven_mean.restricted_index["rmst_regimen[one @ t=4]"] == (
            "one",
            None,
            None,
            4,
            "rmst",
        )

    def test_a_shifted_spacing_fails_the_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: the gaps ``g_k - g_{k-1}`` for ``k = 1..m-1``, shifted by one node."""
        original = estimator_module.restriction

        def shifted(horizon: Any, n_times: int, grid: Any, what: str) -> Any:
            node, _ = original(horizon, n_times, grid, what)
            values = (0.0, *grid)
            return node, tuple(values[k] - values[k - 1] for k in range(1, node))

        monkeypatch.setattr(estimator_module, "restriction", shifted)
        mutated = _estimate(RegimeMean(PLANS, reference="zero", restricted_horizons=(7.0,)))
        assert not _linearity_holds(mutated)


# ------------------------------------------------------------------ 2. the exact law


def _grid_fit(**settings: Any) -> Any:
    return _quiet(
        lambda: LTMLE(
            {label: list(plan) for label, plan in grid_law.REGIMENS.items()},
            reference="never",
            outcome_learner=end_law.CellMeans(),
            pseudo_learner=end_law.CellMeans(),
            treatment_learner=end_law.CellMeans(),
            censoring_learner=end_law.CellMeans(),
            n_folds=1,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
            restricted_horizons=(3, 5),
            **settings,
        ).fit(grid_law.weighted_frame(), weights="w", **grid_law.fit_columns())
    )


@pytest.fixture(scope="module")
def exact_fit() -> Any:
    return _grid_fit()


def _reversed_versus(original: Callable[..., Any]) -> Callable[..., Any]:
    """Mutation: the second regimen enters a difference with the first one's sign."""

    def mutated(self: Any, key: Any, spacings: Any) -> Any:
        regimen, versus, cause, node, measure = key
        name, weights, constant, scale = original(self, key, spacings)
        if versus is None:
            return name, weights, constant, scale
        _, first, _, _ = original(self, (regimen, None, cause, node, measure), spacings)
        _, second, _, _ = original(self, (versus, None, cause, node, measure), spacings)
        both = {key: first.get(key, 0.0) + second.get(key, 0.0) for key in {*first, *second}}
        return name, both, 0.0, scale

    return mutated


class TestTheExactLaw:
    @pytest.mark.parametrize("regimen", list(grid_law.REGIMENS))
    @pytest.mark.parametrize("tau", [3, 5])
    def test_each_stored_level_is_the_enumeration_truth(
        self, exact_fit: Any, regimen: str, tau: int
    ) -> None:
        stored = exact_fit.estimates[f"rmst_regimen[{regimen} @ t={tau}]"]
        assert stored.psi == pytest.approx(grid_law.rmst_truth(regimen, tau), abs=1e-10)
        lost = exact_fit.estimates[f"rmtl_regimen[{regimen} @ t={tau}]"]
        assert lost.psi == pytest.approx(grid_law.rmtl_truth(regimen, tau), abs=1e-10)

    @pytest.mark.parametrize("regimen", ["always", "early"])
    def test_each_stored_difference_is_the_difference_of_truths(
        self, exact_fit: Any, regimen: str
    ) -> None:
        rmst = exact_fit.estimates[f"rmst_regimen[{regimen} vs never @ t=5]"]
        rmtl = exact_fit.estimates[f"rmtl_regimen[{regimen} vs never @ t=5]"]
        effect = grid_law.rmst_truth(regimen, 5) - grid_law.rmst_truth("never", 5)
        # The nonzero witness: the law moves the RMST, so a sign error cannot vanish.
        assert abs(effect) > 0.05
        assert rmst.psi == pytest.approx(effect, abs=1e-10)
        lost = grid_law.rmtl_truth(regimen, 5) - grid_law.rmtl_truth("never", 5)
        assert rmtl.psi == pytest.approx(lost, abs=1e-10)

    def test_a_reversed_versus_sign_fails_by_far_more_than_the_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            LongitudinalResult,
            "_restricted_weights",
            _reversed_versus(LongitudinalResult._restricted_weights),
        )
        mutated = _grid_fit()
        effect = grid_law.rmst_truth("always", 5) - grid_law.rmst_truth("never", 5)
        lost = grid_law.rmtl_truth("always", 5) - grid_law.rmtl_truth("never", 5)
        assert abs(mutated["rmst_regimen[always vs never @ t=5]"].psi - effect) > 1e-9
        assert abs(mutated["rmtl_regimen[always vs never @ t=5]"].psi - lost) > 1e-9


# ------------------------------------------------------------------ 3. and 4. causes


@pytest.fixture(scope="module")
def competing() -> Any:
    return _estimate(
        RegimeMean(PLANS, reference="zero", restricted_horizons=(7.0,)),
        n_causes=2,
        causes={1: "first", 2: "second"},
    )


def _per_cause_holds(result: Any) -> bool:
    weights = {1: 1.5, 2: 1.5, 3: 3.0}
    for cause in ("first", "second"):
        stored = result.rmtl("one", 7.0, cause)
        value = sum(
            weight * result.estimates[_risk(result, "one", node, cause)].psi
            for node, weight in weights.items()
        )
        if abs(stored.psi - value) > 1e-12:
            return False
    return True


class TestCauses:
    def test_the_restricted_means_add_up_to_tau(self, competing: Any) -> None:
        total = competing.rmst("one", 7.0).psi + sum(
            competing.rmtl("one", 7.0, cause).psi for cause in ("first", "second")
        )
        assert total == pytest.approx(7.0, abs=1e-12)

    def test_each_rmtl_sums_its_own_cause(self, competing: Any) -> None:
        first = competing.rmtl("one", 7.0, "first").psi
        second = competing.rmtl("one", 7.0, "second").psi
        # The witness: the two causes' incidences differ by construction.
        assert abs(first - second) > 0.03
        assert _per_cause_holds(competing)

    def test_swapped_cause_labels_fail_the_check(self, competing: Any) -> None:
        """Mutation: the stored index files each cause's RMTL under the other cause."""
        swap = {"first": "second", "second": "first"}
        index = {
            name: (key[0], key[1], swap.get(key[2], key[2]), key[3], key[4])
            for name, key in competing.restricted_index.items()
        }
        assert not _per_cause_holds(replace(competing, restricted_index=index))

    def test_the_eliminated_cause_leaves_the_rmst(self) -> None:
        frame, _ = make_longitudinal_competing(n=400, seed=5)
        columns = {
            "outcome": {"relapse": ("R1", "R2"), "death": ("D1", "D2")},
            "treatment": ["A1", "A2"],
            "baseline": ["W1", "W2"],
            "time_varying": [[], ["L2"]],
            "censoring": ["C1", "C2"],
        }

        def fit(**extra: Any) -> Any:
            return _quiet(
                lambda: LTMLE(
                    {"always": 1, "never": 0},
                    reference="never",
                    n_folds=1,
                    simultaneous=False,
                    outcome_learner=LinearRegression(),
                    pseudo_learner=LinearRegression(),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    censoring_learner=LogisticRegression(max_iter=1000),
                    competing_learner=LogisticRegression(max_iter=1000),
                    restricted_horizons=(3,),
                ).fit(frame, **columns, **extra)
            )

        eliminated, kept = fit(eliminate=["death"]), fit()
        name = "rmst_regimen[always @ t=3]"
        relapse = [_risk(eliminated, "always", node) for node in (1, 2)]
        assert eliminated[name].psi == pytest.approx(
            3.0 - sum(eliminated[risk].psi for risk in relapse), abs=1e-12
        )
        # The witness that the eliminated mask reaches the sum.
        assert abs(eliminated[name].psi - kept[name].psi) > 1e-3


# ------------------------------------------------------------------ 5. and 6. stored values


class TestStoredValues:
    def test_the_contrast_is_the_difference_of_levels(
        self, uneven_mean: Any, uneven_contrast: Any
    ) -> None:
        difference = uneven_contrast.estimates["rmst_regimen[one vs zero @ t=4]"]
        one = uneven_mean.estimates["rmst_regimen[one @ t=4]"]
        zero = uneven_mean.estimates["rmst_regimen[zero @ t=4]"]
        assert difference.psi == pytest.approx(one.psi - zero.psi, abs=1e-12)
        np.testing.assert_allclose(
            difference.influence_curve,
            one.influence_curve - zero.influence_curve,
            atol=1e-12,
            rtol=0,
        )

    @pytest.mark.parametrize(
        "settings",
        [{}, {"n_folds": 2, "repeats": 2}, {"n_bootstrap": 5}],
        ids=["single", "repeats", "bootstrap"],
    )
    def test_each_stored_estimate_equals_its_own_sum_at_two_taus(
        self, settings: dict[str, Any]
    ) -> None:
        result = _quiet(
            lambda: _engine(restricted_horizons=(4.0, 7.0), **settings).fit(_container())
        )
        assert len(result.restricted_index) == 2 * 2 * 2 + 2 * 2
        for name in result.restricted_index:
            stored, computed = result.estimates[name], _from_risks(result, name)
            assert stored.psi == computed.psi
            assert stored.std_error == computed.std_error
            if result.n_repeats > 1:
                for repeat in result.repeats:
                    assert name in repeat.estimates
            if "n_bootstrap" in settings:
                assert stored.bootstrap is not None
                assert stored.bootstrap.ci == computed.bootstrap.ci
        if "n_bootstrap" in settings:
            draws = result.bootstrap.draws
            for name, (constant, weights) in result.restricted_weights.items():
                expected = constant + sum(
                    weight * np.asarray(draws[risk]) for risk, weight in weights.items()
                )
                np.testing.assert_allclose(draws[name], expected, atol=1e-12, rtol=0)

    def test_a_restriction_time_is_found_however_it_is_spelled(self) -> None:
        uneven = _quiet(lambda: _engine(restricted_horizons=(7.0,)).fit(_container()))
        stored = uneven.estimates["rmst_regimen[one vs zero @ t=4]"]
        assert uneven.rmst("one", 7, versus="zero") is stored
        assert uneven.rmst("one", 7.0, versus="zero") is stored
        unit = _quiet(
            lambda: _engine(restricted_horizons=(3,)).fit(_container((1.0, 2.0, 3.0, 4.0)))
        )
        level = unit.estimates["rmst_regimen[one @ t=3]"]
        assert unit.rmst("one", 3) is level
        assert unit.rmst("one", 3.0) is level
        assert unit.rmst("one", np.int64(3)) is level


# ------------------------------------------------------------------ 7. horizons and keys


class TestHorizonsAndKeys:
    def test_the_fit_requests_every_grid_time_below_tau(self) -> None:
        frame = _frame(UNEVEN_FIVE)
        study = CausalStudy(frame, design=_design(UNEVEN_FIVE))
        estimand = RegimeMean(PLANS, horizons=(2.5,), restricted_horizons=(7.0,))
        effect = study.identify(estimand)
        assert effect._restricted_fit_horizons() == (1.0, 2.5, 4.0)
        result = _quiet(lambda: effect.estimate(**_fit_settings()))
        assert sorted(result.estimates) == sorted(
            [
                "risk_regimen[one @ t=2]",
                "risk_regimen[zero @ t=2]",
                "rmst_regimen[one @ t=4]",
                "rmst_regimen[zero @ t=4]",
                "rmtl_regimen[one @ t=4]",
                "rmtl_regimen[zero @ t=4]",
            ]
        )
        assert set(result.parameter_keys) == set(result.estimates)

    @staticmethod
    def _keys_agree(result: Any) -> bool:
        rmst = result.parameter_keys["rmst_regimen[one @ t=4]"]
        risk = result.parameter_keys["risk_regimen[one @ t=4]"]
        if not rmst.horizon == risk.horizon == 4:
            return False
        frame = result.to_frame().set_index("estimand")
        return bool(
            rmst.horizon == risk.horizon == 4
            and rmst.measure == "rmst"
            and risk.measure is None
            and frame.loc["rmst_regimen[one @ t=4]", "time"] == 7.0
            and frame.loc["risk_regimen[one @ t=4]", "time"] == 7.0
        )

    def test_a_restricted_key_carries_the_node_index(self) -> None:
        estimand = RegimeMean(PLANS, reference="zero", restricted_horizons=(7.0,))
        assert self._keys_agree(_estimate(estimand, grid=UNEVEN_FIVE))

    def test_a_key_that_stores_tau_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: the restricted mean is filed under the grid time, not its node."""
        original = _SequentialEstimator._with_restricted

        def by_time(self: Any, result: Any) -> Any:
            stored = original(self, result)
            index = {
                name: (*key[:3], int(stored._restriction_time(key[3])), key[4])
                for name, key in stored.restricted_index.items()
            }
            return replace(stored, restricted_index=index)

        monkeypatch.setattr(_SequentialEstimator, "_with_restricted", by_time)
        estimand = RegimeMean(PLANS, reference="zero", restricted_horizons=(7.0,))
        assert not self._keys_agree(_estimate(estimand, grid=UNEVEN_FIVE))

    @pytest.mark.parametrize(
        ("settings", "moves"),
        [
            (
                {"survival_model": "hazard", "targeting": "one_step", "pseudo_learner": None},
                True,
            ),
            ({}, False),
        ],
        ids=["one_step_hazard", "iterated_mean"],
    )
    def test_the_extra_times_move_a_declared_risk_on_the_one_step_walk_only(
        self, settings: dict[str, Any], moves: bool
    ) -> None:
        def risk(estimand: Any) -> float:
            result = _estimate(estimand, grid=UNEVEN_FIVE, **settings)
            return float(result.estimates["risk_regimen[one @ t=4]"].psi)

        plain = risk(RegimeMean(PLANS, horizons=(7.0,)))
        restricted = risk(RegimeMean(PLANS, horizons=(7.0,), restricted_horizons=(9.0,)))
        if moves:
            assert plain != restricted
            assert restricted == pytest.approx(plain, abs=1e-3)
        else:
            assert plain == restricted


# ------------------------------------------------------------------ 8. refits and replays


class TestRefitsAndReplays:
    def test_a_refutation_refits_the_stored_name(self, uneven_contrast: Any) -> None:
        refuted = _quiet(
            lambda: cleverly.validation.refute(
                uneven_contrast,
                estimand="rmst_regimen[one vs zero @ t=4]",
                tests=("subset",),
                n_replicates=2,
                random_state=0,
            )
        )
        assert refuted.estimand == "rmst_regimen[one vs zero @ t=4]"

    def test_a_refit_without_the_field_loses_the_name(
        self, uneven_contrast: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation (a): the refit template drops ``restricted_horizons``."""
        original = _SequentialEstimator._refit_template

        def dropped(self: Any) -> Any:
            template = original(self)
            template.restricted_horizons = ()
            return template

        monkeypatch.setattr(_SequentialEstimator, "_refit_template", dropped)
        result = _estimate(RegimeContrast(PLANS, reference="zero", restricted_horizons=(7.0,)))
        with pytest.raises(KeyError):
            _quiet(
                lambda: cleverly.validation.refute(
                    result,
                    estimand="rmst_regimen[one vs zero @ t=4]",
                    tests=("subset",),
                    n_replicates=2,
                    random_state=0,
                )
            )

    def test_each_truncation_row_carries_the_restricted_mean(self, uneven_contrast: Any) -> None:
        name = "rmst_regimen[one vs zero @ t=4]"
        curve = _quiet(
            lambda: estimator_module.longitudinal_truncation_curve(uneven_contrast, [0.01, 0.05])
        )
        rows = curve[curve["estimand"] == name]
        assert len(rows) == 2
        fitted = rows[rows["is_fitted_bound"]]
        assert float(fitted["psi"].iloc[0]) == uneven_contrast.estimates[name].psi
        draw = uneven_contrast.repeats[0]
        replay = estimator_module._refit_bound(
            uneven_contrast, draw, uneven_contrast.config.g_bounds
        )
        assert estimator_module._fitted_replay_matches(uneven_contrast, draw, replay)

    def test_a_replay_that_skips_the_rebuild_raises(
        self, uneven_contrast: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation (b): the replay omits the restricted means."""
        monkeypatch.setattr(estimator_module, "_restricted_estimates", lambda result, risks: {})
        with pytest.raises(RuntimeError, match="omitted reported parameters"):
            estimator_module._refit_bound(
                uneven_contrast, uneven_contrast.repeats[0], uneven_contrast.config.g_bounds
            )


# ------------------------------------------------------------------ 9. support


def _g14(estimand: Any) -> Any:
    frame, _ = point_study.draw_from_seed("survival", 400, 0)
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=("W1", "W2"),
        grid=(1.0, 2.0, 3.0, 4.0, 5.0),
    )
    return _quiet(
        lambda: CausalStudy(frame, design=design).estimate(
            estimand,
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            **point_study.primary_learners(),
        )
    )


@pytest.fixture(scope="module")
def g14_contrast() -> Any:
    return _g14(RegimeContrast(point_study.REGIMENS, reference="arm0", restricted_horizons=(5,)))


@pytest.fixture(scope="module")
def g14_engine() -> Any:
    frame, _ = point_study.draw_from_seed("survival", 400, 0)
    return _quiet(lambda: point_study.fit("survival", frame))


class TestSupport:
    def test_the_measured_difference_reads_supported(self, g14_contrast: Any) -> None:
        for name in ("rmst_regimen[arm1 vs arm0 @ t=5]", "rmtl_regimen[arm1 vs arm0 @ t=5]"):
            support = g14_contrast.estimates[name].support
            assert support.label == "supported", support.reason
            assert support.row == "G14.survival"

    def test_the_unmeasured_level_reads_experimental(self) -> None:
        result = _g14(RegimeMean(point_study.REGIMENS, reference="arm0", restricted_horizons=(5,)))
        support = result.estimates["rmst_regimen[arm1 @ t=5]"].support
        assert support.label == "experimental"
        assert "Row G14.survival licenses no restricted mean for this estimate." in support.reason

    def test_the_engine_route_reads_supported_and_a_ratio_does_not(self, g14_engine: Any) -> None:
        assert g14_engine.rmst("arm1", 5, versus="arm0").support.label == "supported"
        ratio = g14_engine.ratio("risk_regimen[arm1 @ t=5]", "risk_regimen[arm0 @ t=5]")
        assert ratio.support.label == "experimental"

    def test_without_the_kind_the_difference_reads_experimental(
        self, g14_engine: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the row loses ``restricted_mean_difference``."""
        from cleverly import _support

        rows = _support.registered_rows()
        mutated = tuple(
            replace(row, kinds=row.kinds - {"restricted_mean_difference"})
            if row.id == "G14.survival"
            else row
            for row in rows
        )
        monkeypatch.setattr(_support, "registered_rows", lambda: mutated)
        assert g14_engine.rmst("arm1", 5, versus="arm0").support.label == "experimental"

    @pytest.mark.parametrize(("folds", "row"), [(1, "G11.nodes4"), (5, "G12.nodes4")])
    def test_the_four_node_rows_license_level_and_difference(self, folds: int, row: str) -> None:
        from tests.studies import full_refit_bootstrap_properties as boot

        label = "derived_survival" if folds == 1 else "derived_crossfit"
        group = boot.GROUPS[label][0]
        frame = boot._sample(label, group, 0)
        result = _quiet(lambda: boot._fit(label, group, 0, frame))
        for estimate in (
            result.rmst("always", 5),
            result.rmst("always", 5, versus="never"),
        ):
            assert estimate.support.label == "supported", estimate.support.reason
            assert estimate.support.row == row

    @pytest.mark.parametrize(
        "case",
        ["four_node_held", "competing_rmtl", "hazard", "sdr"],
    )
    def test_every_other_binned_fit_reads_experimental(self, case: str) -> None:
        if case == "competing_rmtl":
            result = _estimate(
                RegimeContrast(PLANS, reference="zero", restricted_horizons=(7.0,)),
                n_causes=2,
                causes={1: "first", 2: "second"},
            )
            name = "rmtl_regimen[one vs zero, first @ t=4]"
        else:
            settings: dict[str, Any] = {
                "four_node_held": {},
                "hazard": {"survival_model": "hazard", "pseudo_learner": None},
                "sdr": {"method": "sdr"},
            }[case]
            result = _estimate(
                RegimeContrast(PLANS, reference="zero", restricted_horizons=(7.0,)), **settings
            )
            name = "rmst_regimen[one vs zero @ t=4]"
        support = result.estimates[name].support
        assert support.label == "experimental"
        assert "Row " in support.reason or "row" in support.reason


# ------------------------------------------------------------------ 11. refusals


def _raising() -> Any:
    class Raising(LinearRegression):
        def fit(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a learner was fitted")

    return Raising()


def _refusal(call: Callable[[], Any], error: type[Exception], match: str) -> Any:
    with pytest.raises(error, match=match) as caught:
        call()
    return caught.value


class TestRefusals:
    def test_rg1_a_point_treatment_design(self) -> None:
        frame, _ = make_linear_ate(n=100, seed=0)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
        )
        error = _refusal(
            lambda: study.identify(RegimeMean((0, 1), restricted_horizons=(2,))),
            CapabilityError,
            r"RegimeMean\(restricted_horizons=\.\.\.\) asks for restricted means of a time to "
            "an event, and a point-treatment design has one outcome and no follow-up time; "
            "declare TimeToEvent to ask that question",
        )
        assert error.refusal.reason == "invalid_declaration"

    def test_rg2_an_end_of_study_outcome(self) -> None:
        frame, _ = make_longitudinal(n=100, seed=0)
        design = cleverly.LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        error = _refusal(
            lambda: CausalStudy(frame, design=design).identify(
                RegimeMean({"always": 1, "never": 0}, restricted_horizons=(2,))
            ),
            CapabilityError,
            r"restricted_horizons= sums a survival curve, and this design reports one "
            r"end-of-study mean of 'Y'\.",
        )
        assert error.refusal.reason == "different_estimand"
        assert error.refusal.alternative == "Declare an outcome sequence to estimate a curve."

    def test_rg2_a_repeated_outcome(self) -> None:
        frame, _ = make_longitudinal(n=100, seed=0)
        frame["Y1"] = np.where(frame["C1"] == 1, frame["A1"], np.nan)
        design = cleverly.LongitudinalTreatment(
            outcome=cleverly.Repeated(["Y1", "Y"]),
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        error = _refusal(
            lambda: CausalStudy(frame, design=design).identify(
                RegimeMean({"always": 1, "never": 0}, restricted_horizons=(2,))
            ),
            CapabilityError,
            "restricted_horizons= sums a survival curve, and this design reports the mean of a "
            "repeated outcome",
        )
        assert error.refusal.reason == "different_estimand"
        assert error.refusal.alternative == estimator_module.REPEATED_ALTERNATIVE

    @pytest.mark.parametrize(
        ("grid", "tau", "match"),
        [
            (UNEVEN, 5.0, r"restricted horizon 5\.0 is not a reported grid time"),
            (UNEVEN, 1.0, r"restricted horizon 1\.0 is not a reported grid time"),
            (
                (1.0, 2.0, 3.0, 4.0),
                6,
                "restricted horizon must be an integer from 2 to 5; RMST up to t=1 is the "
                "constant 1",
            ),
        ],
    )
    def test_rg3_a_time_the_grid_cannot_take(
        self, grid: tuple[float, ...], tau: float, match: str
    ) -> None:
        study = CausalStudy(_frame(grid), design=_design(grid))
        _refusal(
            lambda: study.identify(RegimeMean(PLANS, restricted_horizons=(tau,))),
            DataError,
            match,
        )

    @pytest.mark.parametrize(
        ("times", "match"),
        [
            ((-1.0,), "restricted_horizons are positive finite restriction times"),
            ((np.inf,), "restricted_horizons are positive finite restriction times"),
            ((2.0, 1.0), "restricted_horizons are strictly increasing"),
        ],
    )
    def test_rg4_restriction_times_are_checked_at_declaration(
        self, times: tuple[float, ...], match: str
    ) -> None:
        for estimand in (RegimeMean, RegimeContrast):
            _refusal(
                lambda estimand=estimand: estimand(PLANS, restricted_horizons=times),
                DataError,
                match,
            )

    def test_rg5_rg6_rg7_the_method_table(self) -> None:
        study = CausalStudy(_frame(UNEVEN), design=_design(UNEVEN))
        plain = study.identify(RegimeContrast(PLANS, reference="zero")).available_methods()
        effect = study.identify(RegimeContrast(PLANS, reference="zero", restricted_horizons=(7.0,)))
        restricted = effect.available_methods()
        before = {row.name: row for row in plain}
        after = {row.name: row for row in restricted}
        assert {name for name, row in after.items() if not row.available} == {
            name for name, row in before.items() if not row.available
        } | {"drtmle", "gcomp", "ipw", "parametric_gformula"}
        for name, row in before.items():
            if not row.available:
                assert after[name].refusal == row.refusal
        drtmle = after["drtmle"].refusal
        assert drtmle.reason == "inference_gap"
        assert drtmle.condition == (
            "SurvivalDRTMLE reports no restricted mean. Its drift correction regresses residuals "
            "on scalar indices of the initial fits (Díaz 2019, Section 5), a pattern whose "
            "remainder has not been audited for any new target."
        )
        assert drtmle.alternative == "Fit method='tmle'."
        for name in ("gcomp", "ipw", "parametric_gformula"):
            reason = after[name].refusal
            assert reason.reason == "api_redirection"
            assert reason.condition.startswith(
                "restricted_horizons= adds restricted means to the targeted fits: LTMLE, SDR "
                "and the collaborative TMLE."
            )
            assert reason.alternative == "Fit method='tmle' or method='sdr'."
        for method in ("drtmle", "gcomp", "ipw", "parametric_gformula"):
            with pytest.raises(CapabilityError):
                effect.estimate(method=method, outcome_learner=_raising(), cross_fit=False)

    def test_rg5_the_drift_engine_refuses_at_construction(self) -> None:
        error = _refusal(
            lambda: SurvivalDRTMLE({"a1": 1, "a0": 0}, restricted_horizons=(3,)),
            CapabilityError,
            "SurvivalDRTMLE reports no restricted mean",
        )
        assert error.refusal.reason == "inference_gap"
        assert error.refusal.roadmap == "X62"

    def test_rg8_and_13_a_level_read_from_a_contrast_result(self) -> None:
        result = _estimate(RegimeContrast(PLANS, reference="zero"))
        error = _refusal(
            lambda: result.rmst("one", 7.0),
            CapabilityError,
            "this result reports contrasts against 'zero' only, so it holds no risk curve of "
            "'one' to sum.",
        )
        assert error.refusal.reason == "api_redirection"
        assert error.refusal.alternative == (
            "Read result.rmst(regimen, tau, versus='zero') from RegimeContrast(..., "
            "restricted_horizons=(tau,)), or declare RegimeMean(..., restricted_horizons=(tau,)) "
            "for each regimen's own restricted mean."
        )

    def test_rg9_an_undeclared_time_the_result_cannot_sum(self) -> None:
        result = _estimate(RegimeMean(PLANS, horizons=(2.5,)))
        _refusal(
            lambda: result.rmst("one", 7.0),
            ValueError,
            r"Refit with horizons=None or include \[1, 3\], or declare "
            r"restricted_horizons=\(tau,\) on the estimand",
        )

    def test_the_engine_refuses_a_tau_whose_times_it_does_not_fit(self) -> None:
        _refusal(
            lambda: _engine(horizons=(4.0,), restricted_horizons=(7.0,)).fit(_container()),
            MethodConfigurationError,
            r"horizons= omits \[1\.0, 2\.5\]",
        )

    def test_r10_the_method_object_names_the_estimand(self) -> None:
        error = _refusal(
            lambda: TMLEMethod().with_overrides(restricted_horizons=(1.0,)),
            MethodConfigurationError,
            "restricted_horizons= declares restricted means, which are targets of the "
            "estimand, not a method setting.",
        )
        assert error.refusal == ENGINE_ONLY["restricted_horizons"]

    def test_sdr_and_the_collaborative_fit_take_the_field(self) -> None:
        sdr = _quiet(
            lambda: SDR(
                PLANS,
                reference="zero",
                n_folds=1,
                simultaneous=False,
                outcome_learner=LinearRegression(),
                pseudo_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
                restricted_horizons=(7.0,),
            ).fit(_container())
        )
        assert "rmst_regimen[one vs zero @ t=4]" in sdr.estimates
        collaborative = _estimate(
            RegimeContrast(PLANS, reference="zero", restricted_horizons=(7.0,)),
            method="collaborative_tmle",
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=None,
            treatment_learner=None,
            censoring_learner=None,
            competing_learner=None,
        )
        assert "rmst_regimen[one vs zero @ t=4]" in collaborative.estimates


# ------------------------------------------------------------------ 12. the band


class TestTheBand:
    @staticmethod
    def _band(restricted: tuple[float, ...]) -> Any:
        return _estimate(
            RegimeMean(PLANS, reference="zero", restricted_horizons=restricted),
            simultaneous=True,
            n_multiplier=500,
        ).simultaneous

    def test_a_restricted_mean_never_joins_the_band(self) -> None:
        plain, restricted = self._band(()), self._band((7.0,))
        assert set(restricted.bands) == set(plain.bands)
        assert not any(name.startswith(("rmst", "rmtl")) for name in restricted.bands)
        assert restricted.critical_value == plain.critical_value
        assert restricted.support == plain.support

    def test_letting_the_restricted_means_in_moves_the_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the band narrows over the restricted means too."""
        import cleverly.study as study_module

        original = study_module._narrow_bands

        def widened(result: Any, retained: Any, method: Any) -> Any:
            family = {
                **retained,
                **{name: result.estimates[name] for name in result.restricted_index},
            }
            family = {name: e for name, e in family.items() if e.scale == "level"}
            return original(result, family, method)

        plain = self._band(())
        monkeypatch.setattr(study_module, "_narrow_bands", widened)
        mutated = self._band((7.0,))
        assert set(mutated.bands) != set(plain.bands)


# ------------------------------------------------------------------ 14. to 16.


class TestTheResult:
    def test_a_saved_result_keeps_the_restricted_means(
        self, uneven_contrast: Any, tmp_path: Any
    ) -> None:
        path = tmp_path / "restricted.joblib"
        uneven_contrast.save(path)
        loaded = cleverly.load(path)
        assert loaded.restricted_horizons == uneven_contrast.restricted_horizons
        assert loaded.restricted_index == uneven_contrast.restricted_index
        assert loaded.restricted_weights == uneven_contrast.restricted_weights
        assert loaded.parameter_keys == uneven_contrast.parameter_keys
        for name, estimate in uneven_contrast.estimates.items():
            assert loaded.estimates[name].psi == estimate.psi

    def test_an_evalue_refuses_a_stored_restricted_mean(self, uneven_mean: Any) -> None:
        from cleverly.sensitivity import evalue

        with pytest.raises(CleverlyError, match="reads the risk ratio of two reported risks"):
            evalue(uneven_mean, "rmst_regimen[one @ t=4]")
        with pytest.raises(CleverlyError, match="are not risks this fit reports"):
            evalue(uneven_mean, ("rmst_regimen[one @ t=4]", "rmst_regimen[zero @ t=4]"))
        assert not set(uneven_mean.restricted_index) & set(uneven_mean.parameter_index)

    def test_the_score_rows_are_the_risks_equations(self) -> None:
        """A restricted mean adds no equation: its score is the risks' scores combined."""
        container = _container()
        plain = _quiet(lambda: _engine().fit(container))
        restricted = _quiet(lambda: _engine(restricted_horizons=(7.0,)).fit(container))
        before = plain.diagnostics.score_equations()
        after = restricted.diagnostics.score_equations()
        assert after.rows == before.rows
        assert after.passed


# ------------------------------------------------------------------ plots and the summary


class TestPlotsAndSummary:
    @staticmethod
    def _times(figure: Any) -> list[list[float]]:
        return [
            [float(x) for x in line.get_xdata()]
            for axis in figure.axes
            for line in axis.get_lines()
            if line.get_gid() == "cleverly-guide"
        ]

    def test_the_curve_draws_every_grid_time_of_an_engine_result(self) -> None:
        from cleverly.plotting import curve, forest

        result = _quiet(lambda: _engine(restricted_horizons=(7.0,)).fit(_container()))
        assert self._times(curve(result, measure="rmst")) == [[2.5, 4.0, 7.0]] * 3
        labels = {
            text.get_text() for axis in forest(result).axes for text in axis.get_yticklabels()
        }
        assert any("rmst" in label for label in labels)

    def test_the_curve_draws_the_stored_means_alone_where_risks_are_missing(
        self, uneven_contrast: Any
    ) -> None:
        from cleverly.plotting import curve

        declared = _estimate(
            RegimeMean(PLANS, horizons=(2.5,), restricted_horizons=(7.0,)), grid=UNEVEN_FIVE
        )
        assert self._times(curve(declared, measure="rmst")) == [[7.0], [7.0]]
        # The contrast family draws its stored differences in one panel.
        figure = curve(uneven_contrast, measure="rmtl")
        assert len(figure.axes) == 1
        assert self._times(figure) == [[7.0]]

    def test_the_summary_says_what_each_restricted_mean_is(self, uneven_contrast: Any) -> None:
        from cleverly.longitudinal._summary import meaning

        assert meaning(uneven_contrast, "rmst_regimen[one vs zero @ t=4]") == (
            "mean event-free time up to time 7 if every unit followed plan one, minus that "
            "mean under plan zero"
        )


def test_each_draw_of_a_repeated_fit_stores_its_own_restricted_means() -> None:
    """Each cross-fitting draw holds every restricted mean as the sum of its own risks.

    The repeat registry of ``test_longitudinal_repeats`` names this test as the witness of
    ``restricted_horizons=`` and of the result's ``restricted_*`` fields.
    """
    result = _quiet(
        lambda: _engine(restricted_horizons=(7.0,), n_folds=2, repeats=2).fit(_container())
    )
    assert result.n_repeats == 2
    assert result.restricted_horizons == (7.0,)
    assert set(result.restricted_index) == set(result.restricted_weights)
    for name, (constant, weights) in result.restricted_weights.items():
        draws = []
        for repeat in result.repeats:
            expected = constant + sum(
                weight * repeat.estimates[risk].psi for risk, weight in weights.items()
            )
            assert repeat.estimates[name].psi == pytest.approx(expected, abs=1e-12)
            draws.append(repeat.estimates[name].psi)
        # The two draws differ, so a pooled value copied into each draw would fail above.
        assert draws[0] != draws[1]
