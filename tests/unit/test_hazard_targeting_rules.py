"""The targeting rules on the hazard walk of ``survival_model="hazard"`` (X33).

``stopping=`` and ``iterate=`` act on both hazard algorithms, and ``step_bound=`` acts on the
per-horizon fit.  The one-step walk takes its steps in ``step_size``, so ``LTMLE`` refuses a step
bound beside it as a redirection (R-c4), before any data or learner.

The witnesses run the wrong hazard learner of ``tests/unit/test_hazard_survival_law.py`` with the
exact mechanism, so the targeting has steps to take.  A declared rule stops at the first iterate
that meets it, and its thresholds are the ones a hand computation from the reported influence
curves gives.  ``iterate="best"`` on the per-horizon fit reports the iterate that a run capped at
that iterate reports, and on the one-step walk it reports the last accepted step.  The defaults,
written out, give the shipped fit bit for bit.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import MethodConfigurationError, RefusalReason
from cleverly.fluctuation._score import StopContext
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.hazard import walk_curve
from cleverly.methods import TMLEMethod
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, PointLaw, static
from tests.unit.test_hazard_survival_law import ArmOnly, _internals

LAW = PointLaw("survival", 2)
REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
ARMS = {"a0": 0, "a1": 1, "a2": 2}
ALGORITHMS = ("iterative", "one_step")

#: The pinned text of R-c4 beside the one-step hazard walk.
ONE_STEP_MESSAGE = (
    "targeting='one_step' with survival_model='hazard' already walks the universal "
    "least-favorable submodel in steps of step_size. step_bound= bounds the steps of "
    "targeting='iterative'."
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


def _fit(learner: Any = None, **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            REGIMENS,
            n_folds=1,
            survival_model="hazard",
            step_size=2e-2,
            outcome_learner=learner if learner is not None else CellMeans(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
            **settings,
        ).fit(LAW.frame(), **LAW.fit_columns())


def _records(result: Any) -> dict[int, Any]:
    """One record per horizon of a per-horizon fit, or the one shared record of a walk."""
    return {fit.horizon: fit.hazard for fit in result.fits.values()}


def _psi(result: Any) -> dict[str, float]:
    return {name: estimate.psi for name, estimate in result.estimates.items()}


def _truth(fit: Any) -> float:
    return LAW.functional(LAW.probs, static(ARMS[fit.regimen.label]), fit.horizon, 1)


# ----------------------------------------------------------------------------- R-c4


def test_r_c4_a_step_bound_beside_the_one_step_walk_is_a_redirection() -> None:
    with pytest.raises(MethodConfigurationError, match=re.escape(ONE_STEP_MESSAGE)) as raised:
        LTMLE(
            REGIMENS, survival_model="hazard", targeting="one_step", step_bound=0.5, **_refusing()
        )
    refusal = raised.value.refusal
    assert refusal.reason is RefusalReason.API_REDIRECTION
    assert refusal.roadmap is None
    assert refusal.alternative == "step_bound= bounds the steps of targeting='iterative'."


def test_r_c4_the_refusal_precedes_the_data_and_every_learner() -> None:
    estimator = None
    with pytest.raises(MethodConfigurationError):
        estimator = LTMLE(
            REGIMENS, survival_model="hazard", targeting="one_step", step_bound=1.0, **_refusing()
        )
    assert estimator is None


def test_a_malformed_rule_is_still_its_own_error() -> None:
    """Validity comes first: a bad value is the R-a1, R-b1 or R-c1 error."""
    with pytest.raises(ValueError, match="stopping must be"):
        LTMLE(REGIMENS, survival_model="hazard", stopping="bogus", **_refusing())
    with pytest.raises(ValueError, match="iterate must be"):
        LTMLE(REGIMENS, survival_model="hazard", iterate="middle", **_refusing())
    with pytest.raises(ValueError, match="step_bound must be a positive finite number"):
        LTMLE(REGIMENS, survival_model="hazard", targeting="one_step", step_bound=-1.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"iterate": "best"},
        {"stopping": "scaled_var"},
        {"stopping": "sample_size", "iterate": "best"},
        {"step_bound": 0.5},
        {"step_bound": 0.5, "step_search": False},
    ],
)
def test_the_rules_are_accepted_at_construction(settings: dict[str, Any]) -> None:
    LTMLE(REGIMENS, survival_model="hazard", **settings, **_refusing())


@pytest.mark.parametrize("settings", [{"iterate": "best"}, {"stopping": "sample_size"}])
def test_the_one_step_walk_accepts_iterate_and_stopping(settings: dict[str, Any]) -> None:
    LTMLE(REGIMENS, survival_model="hazard", targeting="one_step", **settings, **_refusing())


def test_the_method_route_passes_a_rule_through() -> None:
    method = TMLEMethod().with_overrides(survival_model="hazard", stopping="sample_size")
    kwargs = method.estimator_kwargs(longitudinal=True)
    assert kwargs["stopping"] == "sample_size"
    LTMLE(REGIMENS, **{**kwargs, **_refusing()})


# ------------------------------------------------------------------- the defaults


def test_the_written_out_defaults_are_bit_identical_to_the_omitted_ones() -> None:
    omitted = _fit()
    written = _fit(stopping="relative", iterate="last", step_bound=None, step_search=True)
    assert list(omitted.estimates) == list(written.estimates)
    for name in omitted.estimates:
        assert np.array_equal(omitted.estimates[name].psi, written.estimates[name].psi)
        assert np.array_equal(
            omitted.estimates[name].influence_curve, written.estimates[name].influence_curve
        )
        assert omitted.estimates[name].variance == written.estimates[name].variance


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_a_default_record_holds_no_rule(targeting: str) -> None:
    for record in _records(_fit(ArmOnly(), targeting=targeting)).values():
        assert record.stopping == "relative"
        assert record.thresholds is None
        assert record.rule_ratio is None
        assert record.best_iterate is None


def test_the_hazard_walk_calls_no_point_treatment_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hazard solvers reuse the rule functions of the targeting modules, and no solver.

    Every fluctuation solver of the targeting modules raises when called, and a hazard fit
    under each rule still runs.
    """
    from cleverly.estimators import targeting as targeting_module
    from cleverly.fluctuation import iterative as iterative_module
    from cleverly.longitudinal import sequential as sequential_module

    def tripwire(*args: object, **kwargs: object) -> object:
        raise AssertionError("a targeting solver ran on the hazard walk")

    for module, names in (
        (
            iterative_module,
            ("solve_fluctuation", "_solve_fluctuation_ruled", "_solve_fluctuation_bounded"),
        ),
        (targeting_module, ("solve_submodel",)),
        (sequential_module, ("solve_fluctuation", "_fluctuate_node")),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, tripwire)
    assert _fit().estimates
    assert _fit(ArmOnly(), stopping="sample_size", iterate="best", step_bound=0.5).estimates
    assert _fit(ArmOnly(), targeting="one_step", stopping="scaled_var").estimates
    assert {"rule_ratio_at", "BestIterate", "StopContext"} <= set(vars(hazard_module))


# --------------------------------------------------------------------- stopping=


@pytest.mark.parametrize("rule", ["sample_size", "scaled_var"])
def test_a_rule_stops_the_per_horizon_fit_at_its_first_iterate_that_meets_it(rule: str) -> None:
    plain = _records(_fit(ArmOnly()))
    ruled = _records(_fit(ArmOnly(), stopping=rule))
    for horizon, record in ruled.items():
        assert record.stopping == rule
        assert record.converged
        assert record.rule_ratio is not None and record.rule_ratio <= 1.0
        assert 1 <= record.n_iter < plain[horizon].n_iter
    # One iteration fewer misses the rule at every horizon.
    shorter = _records(
        _fit(ArmOnly(), stopping=rule, max_iter=min(r.n_iter for r in ruled.values()) - 1)
    )
    for horizon, record in shorter.items():
        if record.n_iter < ruled[horizon].n_iter:
            assert not record.converged
            assert record.rule_ratio > 1.0


@pytest.mark.parametrize("rule", ["sample_size", "scaled_var"])
def test_a_rule_stops_the_one_step_walk_at_its_first_step_that_meets_it(rule: str) -> None:
    plain = _fit(ArmOnly(), targeting="one_step")
    ruled = _fit(ArmOnly(), targeting="one_step", stopping=rule)
    record = next(iter(ruled.fits.values())).hazard
    assert record.converged
    assert record.rule_ratio is not None and record.rule_ratio <= 1.0
    assert 1 <= record.n_iter < next(iter(plain.fits.values())).hazard.n_iter
    # The same walk, capped one step short, misses the rule.
    inputs, start = _internals(ruled)
    horizons = ruled.config.horizons
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, short = walk_curve(
            start,
            inputs,
            horizons,
            record.targets,
            step_size=2e-2,
            tol=record.tol,
            max_steps=record.n_iter - 1,
            stopping=rule,
            stop=StopContext(None, 1.0),
        )
    assert short.rule_ratio is not None and short.rule_ratio > 1.0
    assert not short.converged
    # The rule moves the estimate: the walk stops before the score is solved to tol.
    assert max(abs(ruled[n].psi - plain[n].psi) for n in plain.estimates) > 1e-8


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_the_thresholds_are_the_hand_computation_from_the_influence_curves(
    targeting: str,
) -> None:
    for rule in ("sample_size", "scaled_var"):
        result = _fit(ArmOnly(), targeting=targeting, stopping=rule)
        n = LAW.n
        for record in {id(f.hazard): f.hazard for f in result.fits.values()}.values():
            curves = np.array(
                [
                    result.fits[
                        next(
                            key
                            for key, fit in result.fits.items()
                            if (fit.regimen.label, fit.cause, fit.horizon) == target
                        )
                    ].influence_curve_scaled
                    for target in record.targets
                ]
            )
            if rule == "sample_size":
                expected = np.full(len(record.targets), 1.0 / n)
            else:
                expected = np.std(curves, axis=1, ddof=1) / (np.sqrt(n) * min(np.log(n), 10.0))
            np.testing.assert_allclose(record.thresholds, expected, rtol=1e-12, atol=0.0)
            statistic = np.abs(record.score) / expected
            assert record.rule_ratio == pytest.approx(float(np.max(statistic)), rel=1e-12)


def test_the_summary_prints_the_residual_line_of_a_declared_rule() -> None:
    text = _fit(ArmOnly(), stopping="scaled_var").summary()
    assert "stopping rule scaled_var" in text


# ---------------------------------------------------------------------- iterate=


def test_best_on_the_per_horizon_fit_is_the_fit_capped_at_its_iterate() -> None:
    """A bounded step without a search overshoots, so the best iterate is not the last."""
    settings: dict[str, Any] = {"step_bound": 0.05, "step_search": False}
    best = _fit(ArmOnly(), iterate="best", **settings)
    records = _records(best)
    assert any(record.best_iterate < record.n_iter for record in records.values())
    for horizon, record in records.items():
        assert 1 <= record.best_iterate <= record.n_iter
        capped = _fit(ArmOnly(), max_iter=record.best_iterate, **settings)
        for key, fit in best.fits.items():
            if fit.horizon != horizon:
                continue
            reference = capped.fits[key]
            assert fit.psi_scaled == reference.psi_scaled
            assert np.array_equal(fit.influence_curve_scaled, reference.influence_curve_scaled)
            assert fit.hazard.converged == reference.hazard.converged


def test_best_on_the_one_step_walk_is_the_last_accepted_step() -> None:
    last = _fit(ArmOnly(), targeting="one_step")
    best = _fit(ArmOnly(), targeting="one_step", iterate="best")
    record = next(iter(best.fits.values())).hazard
    assert record.best_iterate == record.n_iter >= 1
    assert _psi(best) == _psi(last)


# -------------------------------------------------------------------- step_bound=


@pytest.mark.parametrize("search", [True, False])
def test_a_step_bound_changes_the_per_horizon_walk(search: bool) -> None:
    plain = _records(_fit(ArmOnly()))
    bounded = _fit(ArmOnly(), step_bound=0.5, step_search=search, max_iter=100)
    for horizon, record in _records(bounded).items():
        assert record.n_iter != plain[horizon].n_iter
    if search:
        # The line search reaches the root, and the exact mechanism returns the truth.
        for fit in bounded.fits.values():
            assert fit.hazard.converged
            assert fit.psi_scaled == pytest.approx(_truth(fit), abs=1e-8)


def test_a_step_bound_is_replayed() -> None:
    from cleverly.longitudinal.estimator import _fitted_replay_matches, _refit_bound

    result = _fit(ArmOnly(), step_bound=0.5, stopping="sample_size", iterate="best")
    replay = _refit_bound(result, result.repeats[0], result.config.g_bounds)
    assert _fitted_replay_matches(result, result.repeats[0], replay)


# ------------------------------------------------------------------------- NaN score


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_a_nan_relative_score_stops_the_default_rule_at_once(
    targeting: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped loop read ``while trace[-1] > tol``, which a NaN score leaves at once.

    ``not met()`` with ``met = trace[-1] <= tol`` would instead run to ``max_iter``. The default
    rule keeps the shipped exit, so a NaN score takes no step.
    """
    result = _fit(ArmOnly(), targeting=targeting)
    inputs, start = _internals(result)
    record = next(iter(result.fits.values())).hazard
    assert record.n_iter >= 1
    monkeypatch.setattr(hazard_module, "relative_score", lambda *args: float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if targeting == "iterative":
            horizon = record.targets[0][2]
            _, fit = hazard_module.solve_per_horizon(
                start, inputs, horizon, record.targets, max_iter=5, tol=record.tol
            )
        else:
            _, fit = walk_curve(
                start,
                inputs,
                result.config.horizons,
                record.targets,
                step_size=2e-2,
                tol=record.tol,
                max_steps=5,
            )
    assert fit.n_iter == 0
    assert not fit.converged
