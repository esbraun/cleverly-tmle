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

import dataclasses
import itertools
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
from tests.unit._source_mutation import mutate
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
    summary = _fit(ArmOnly(), stopping="scaled_var").summary()
    assert "Under the declared stopping rule" in summary.note("stopping_rule").text
    assert "stopping_rule" not in _fit(ArmOnly()).summary().note_codes


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
    # Measured: the Newton walk takes 4 and 3 iterations, the searched walk 6 and 12 within
    # the default budget, and the fixed step runs out at 100.
    plain = _records(_fit(ArmOnly()))
    budget: dict[str, Any] = {} if search else {"max_iter": 100}
    bounded = _fit(ArmOnly(), step_bound=0.5, step_search=search, **budget)
    for horizon, record in _records(bounded).items():
        assert record.n_iter != plain[horizon].n_iter
    if search:
        # The line search reaches the root, and the exact mechanism returns the truth.
        for fit in bounded.fits.values():
            assert fit.hazard.converged
            assert fit.psi_scaled == pytest.approx(_truth(fit), abs=1e-8)


#: The length line of ``_bounded_hazard_step``, the anchor of its mutations.
HAZARD_STEP = "length = _line_step(line.slope, length, loss=line.loss, mass=line.mass())"
#: The step search the hazard walk shipped before: SciPy's bounded Brent method on loss values.
HAZARD_LOSS_SEARCH = (
    "from scipy.optimize import minimize_scalar\n"
    "\n"
    "        found = minimize_scalar(\n"
    '            line.loss, bounds=(0.0, length), method="bounded",\n'
    '            options={"xatol": 1.4901161193847656e-08},\n'
    "        )\n"
    "        length = float(found.x)"
)


def _hazard_step(monkeypatch: pytest.MonkeyPatch, new: str) -> None:
    mutate(monkeypatch, hazard_module, hazard_module, "_bounded_hazard_step", [(HAZARD_STEP, new)])


def _assert_contracts(result: Any) -> None:
    """Every horizon converges within 15 iterations, contracting below ``1e-6`` without a rise."""
    for horizon, record in _records(result).items():
        assert record.converged and record.failure is None, horizon
        assert record.n_iter <= 15, horizon
        trace = np.asarray(record.trace)
        tail = trace[int(np.argmax(trace < 1e-6)) :]
        assert tail[0] < 1e-6 and len(tail) >= 2, horizon
        assert np.all(np.diff(tail) < 0.0), horizon
        decades = (np.log10(tail[0]) - np.log10(tail[-1])) / (len(tail) - 1)
        assert decades >= 0.08, horizon


def test_a_searched_walk_reaches_tol_within_the_default_budget() -> None:
    """The bounded walk contracts to ``tol=1e-10`` within the default ``max_iter=20``.

    Measured: horizons 1 and 2 converge in 6 and 12 iterations.  From the first iterate below
    ``1e-6`` the trace falls at every step (smallest cut 3.1 times) and by 1.64 and 0.79
    decades per iteration, so the 0.08 gate has a margin of ten.  The strict decrease catches a
    walk that wanders at a floor even when one iterate happens to land under ``tol``.
    """
    _assert_contracts(_fit(ArmOnly(), step_bound=0.5))


def test_the_bounded_loss_search_fails_the_default_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hazard_step(monkeypatch, HAZARD_LOSS_SEARCH)
    with pytest.raises(AssertionError):
        _assert_contracts(_fit(ArmOnly(), step_bound=0.5))


def test_the_hazard_step_checks_descent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hazard step passes its loss and a positive absolute mass to the descent check.

    The clip can make the hazard line loss non-unimodal, so the step must not end above where it
    started.  With the mass set to 0 the check trips on the round-off rise of the last steps and
    halves them.  The one-cause walk above still meets its count and rate gates (horizon 1 in 9
    iterations instead of 6).  The two-cause window gate of
    :func:`test_a_two_cause_walk_converges_linearly` does fail (a cut of 4.1 against 6.5, from
    8.4e-10 to 2.1e-10), but that margin is too thin for a control.  So this spy is the witness,
    and ``test_step_bound.py`` tests the check itself on a line with a high valley.
    """
    calls: list[tuple[bool, float]] = []
    shipped = hazard_module._line_step

    def spy(slope: Any, delta: float, *, loss: Any = None, mass: float = 0.0) -> float:
        calls.append((loss is not None, mass))
        return float(shipped(slope, delta, loss=loss, mass=mass))

    monkeypatch.setattr(hazard_module, "_line_step", spy)
    _fit(ArmOnly(), step_bound=0.5)
    assert len(calls) >= 18
    assert all(passed and np.isfinite(mass) and mass > 0.0 for passed, mass in calls)


def _two_cause_walk() -> Any:
    law = PointLaw("survival", 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            REGIMENS,
            n_folds=1,
            survival_model="hazard",
            outcome_learner=ArmOnly(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
        ).fit(law.frame(), **law.fit_columns())
    inputs, state = _internals(result)
    targets = tuple((regimen, None, 2) for regimen in REGIMENS for _ in range(2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, fit = hazard_module.solve_per_horizon(
            state, inputs, 2, targets, max_iter=100, tol=1e-10, step_bound=0.5
        )
    return fit


def _assert_windows_fall(fit: Any) -> None:
    trace = np.asarray(fit.trace)
    maxima = [float(trace[i : i + 10].max()) for i in range(0, len(trace), 10)]
    assert len(maxima) >= 3
    for before, after in itertools.pairwise(maxima):
        assert after <= before / 6.5


def test_a_two_cause_walk_converges_linearly() -> None:
    """Six targets: the walk zig-zags, and converges at a linear rate without a stall.

    Steepest descent with an exact line search contracts by a factor set by the spread of the
    curvatures across the targets, so the relative score rises inside a zig-zag.  The gate reads
    the maxima of windows of 10 iterations: each must be at most a 6.5th of the one before.
    Measured: converged in 43 iterations, window cuts 70, 846, 65 and 91 (a margin of ten).  A
    walk at a floor, whose window maxima stay flat, fails.
    """
    fit = _two_cause_walk()
    assert fit.converged and fit.failure is None and fit.n_iter <= 100
    _assert_windows_fall(fit)


def test_the_bounded_loss_search_stalls_the_two_cause_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured: unconverged at 100, with window cuts of 0.7 to 3.4 after the fourth window.
    _hazard_step(monkeypatch, HAZARD_LOSS_SEARCH)
    fit = _two_cause_walk()
    assert not fit.converged
    with pytest.raises(AssertionError):
        _assert_windows_fall(fit)


def test_a_searched_walk_that_returns_to_a_visited_state_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The logits alternate between two states, as in a round-off cycle at the floor.

    The searched step is a function of the logits, so the walk stops at the first revisit.  With
    the revisit exit removed, the same walk runs to ``max_iter``.
    """
    result = _fit(ArmOnly(), step_bound=0.5)
    inputs, start = _internals(result)
    record = next(iter(result.fits.values())).hazard
    shifted = dataclasses.replace(start, logits=start.logits + 1e-3)

    def walk() -> Any:
        states = itertools.cycle([shifted, start])
        monkeypatch.setattr(hazard_module, "_bounded_hazard_step", lambda *args: next(states))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, fit = hazard_module.solve_per_horizon(
                start,
                inputs,
                record.targets[0][2],
                record.targets,
                max_iter=30,
                tol=1e-10,
                step_bound=0.5,
            )
        return fit

    fit = walk()
    # As for a zero-length step: the walk stopped short of ``max_iter`` without moving on.
    assert fit.n_iter == 2 and not fit.converged
    assert fit.failure == "line_search_exhausted"
    mutate(
        monkeypatch,
        hazard_module,
        hazard_module,
        "solve_per_horizon",
        [("visited.seen(state.logits)", "False")],
    )
    assert walk().n_iter == 30


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
