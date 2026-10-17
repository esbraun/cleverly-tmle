"""Hazard-based survival targeting, checked exactly on a finite law (X14 E1-E8, E10, E11).

``tests/discrete_law_point_survival.py`` realises each law exactly.  A saturated hazard fit is
the oracle, so the estimate equals the g-formula and the influence curve equals its Gateaux
derivative.  A hazard learner that reads the arm only is wrong, and with the exact mechanism
the targeted estimate must still equal the truth (double robustness), while the initial
plug-in misses it.  Every check runs both algorithms: ``targeting="iterative"`` targets each
horizon on its own, and ``targeting="one_step"`` walks one set of hazards over the whole curve.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import WeightingWarning
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.estimator import (
    _fitted_replay_matches,
    _refit_bound,
    longitudinal_truncation_curve,
)
from cleverly.longitudinal.hazard import (
    SIGMA_FLOOR,
    WALK_LOSS_TOLERANCE,
    HazardState,
    hazard_inputs,
    influence,
    solve_per_horizon,
    walk_curve,
)
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    PointLaw,
    known,
    static,
    survival_law,
)

REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
ALGORITHMS = ("iterative", "one_step")
#: Bit-exact identities hold on the platform that computed them.  CI runs on Linux, where a
#: different BLAS can move the last bits, so CI compares at a relative 1e-12.
EXACT = os.environ.get("CI") is None
FLOOR = 1e-12
#: ``q(a | w)``, indexed ``[w, a]``.
POLICY = np.array([[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]])
#: The one-cause law has three nodes.  The two-cause law has two, so that it has 2,048
#: rows rather than 32,768: a one-step walk over two causes takes several hundred steps.
LAWS = {"one cause": survival_law(), "two causes": PointLaw("survival", 2, causes=2)}
#: The two-node one-cause law, for the checks that need no third node.
SMALL = PointLaw("survival", 2)
#: The first one-step step.  Larger than the default 1e-3 so that the walk on the
#: 32,768-row law takes about a hundred steps rather than a thousand; the walk halves it
#: whenever the score norm would rise, so the identities do not depend on it.
STEP = 2e-2


class ArmOnly(CellMeans):
    """A wrong hazard learner: the cell mean given the arm and the history, without ``W``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ArmOnly:
        super().fit(np.asarray(X)[:, 1:], y, sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(np.asarray(X)[:, 1:])

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(np.asarray(X)[:, 1:])


def _policy(frame: Any) -> np.ndarray:
    return POLICY[np.asarray(frame["W"], dtype=int)]


STOCHASTIC = Stochastic(_policy, "q", density_kind="known")
POINT_MASS = Stochastic(
    lambda frame: np.tile([0.0, 1.0, 0.0], (len(frame), 1)), "one", density_kind="known"
)


def _fit(
    law: PointLaw,
    targeting: str,
    *,
    learner: Any = None,
    regimens: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    **columns: Any,
) -> Any:
    return LTMLE(
        regimens or REGIMENS,
        n_folds=1,
        survival_model="hazard",
        targeting=targeting,
        step_size=STEP,
        outcome_learner=learner if learner is not None else CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(law.frame() if frame is None else frame, **law.fit_columns(), **columns)


def _horizons(law: PointLaw) -> tuple[int, ...]:
    return tuple(range(1, law.n_times + 1))


def _causes(law: PointLaw) -> tuple[tuple[str | None, int], ...]:
    return ((None, 1),) if law.causes == 1 else (("relapse", 1), ("death", 2))


def _name(label: str, cause: str | None, horizon: int) -> str:
    if cause is None:
        return f"risk_regimen[{label} @ t={horizon}]"
    return f"cif_regimen[{label}, {cause} @ t={horizon}]"


def _same(left: Any, right: Any) -> None:
    if EXACT:
        assert np.array_equal(np.asarray(left), np.asarray(right))
    else:
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=FLOOR)


def _internals(result: Any) -> tuple[Any, HazardState]:
    """The targeting inputs and the initial state a fit started from."""
    recipe = result.replay_recipe
    cumulative = [
        result.mechanism.cumulative(result.data, level, result.config.g_bounds)
        for level in recipe.hazard.levels
    ]
    inputs = hazard_inputs(result.data, recipe.plans, cumulative, alpha=recipe.alpha)
    return inputs, HazardState.from_hazards(
        result.initial_hazards.hazards, inputs.bounds, inputs.empty
    )


@functools.cache
def _wrong(name: str, targeting: str) -> Any:
    """The fit of law ``name`` with the wrong hazard learner, shared across the module."""
    return _fit(LAWS[name], targeting, learner=ArmOnly())


@pytest.fixture(scope="module", params=[(law, a) for law in LAWS for a in ALGORITHMS])
def saturated(request: pytest.FixtureRequest) -> tuple[PointLaw, Any]:
    name, targeting = request.param
    return LAWS[name], _fit(LAWS[name], targeting)


@pytest.fixture(scope="module", params=[(law, a) for law in LAWS for a in ALGORITHMS])
def misspecified(request: pytest.FixtureRequest) -> tuple[PointLaw, Any]:
    name, targeting = request.param
    return LAWS[name], _wrong(name, targeting)


# --------------------------------------------------------------------------- E1, E2


def test_e1_the_saturated_estimate_is_the_g_formula(saturated: tuple[PointLaw, Any]) -> None:
    law, result = saturated
    for label, arm in REGIMENS.items():
        for cause, code in _causes(law):
            for horizon in _horizons(law):
                truth = law.functional(law.probs, static(arm), horizon, code)
                psi = result[_name(label, cause, horizon)].psi
                assert psi == pytest.approx(truth, rel=1e-12, abs=FLOOR)


def test_e2_the_influence_curve_is_the_gateaux_derivative(
    saturated: tuple[PointLaw, Any],
) -> None:
    law, result = saturated
    rows = law.first_row_of()
    for label, arm in REGIMENS.items():
        for cause, code in _causes(law):
            for horizon in _horizons(law):
                np.testing.assert_allclose(
                    result[_name(label, cause, horizon)].influence_curve[rows],
                    law.eif(static(arm), horizon, code),
                    rtol=1e-10,
                    atol=1e-12,
                )


# ------------------------------------------------------------------------------ E3


def test_e3_targeting_repairs_a_wrong_hazard_with_the_exact_mechanism(
    misspecified: tuple[PointLaw, Any],
) -> None:
    law, result = misspecified
    worst_initial = 0.0
    for key, fit in result.fits.items():
        record = fit.hazard
        arm = REGIMENS[fit.regimen.label]
        code = 1 if fit.cause in (None, "relapse") else 2
        truth = law.functional(law.probs, static(arm), fit.horizon, code)
        assert fit.psi_scaled == pytest.approx(truth, abs=1e-8), key
        index = record.targets.index((fit.regimen.label, fit.cause, fit.horizon))
        worst_initial = max(worst_initial, abs(float(record.initial_levels[index]) - truth))
    assert worst_initial > 1e-2


# ------------------------------------------------------------------------------ E4


def test_e4_every_target_score_is_solved(misspecified: tuple[PointLaw, Any]) -> None:
    _, result = misspecified
    rows = result.diagnostics.score_equations().rows
    assert len(rows) == len(result.fits)
    for row in rows:
        assert row.passed
        assert row.relative_score <= 1e-10
    for fit in result.fits.values():
        record = fit.hazard
        assert record.converged
        if record.algorithm == "per_horizon":
            # Each record solves the targets of its own horizon only.
            assert {target[2] for target in record.targets} == {fit.horizon}


# ------------------------------------------------------------------------------ E5


@pytest.mark.parametrize("name", list(LAWS))
def test_e5_one_step_reports_every_level_from_one_set_of_hazards(name: str) -> None:
    law = LAWS[name]
    result = _wrong(name, "one_step")
    records = {id(fit.hazard) for fit in result.fits.values()}
    assert len(records) == 1
    inputs, start = _internals(result)
    record = next(iter(result.fits.values())).hazard
    state, _ = walk_curve(start, inputs, _horizons(law), record.targets, step_size=STEP, tol=1e-10)
    levels = state.levels(inputs.q, inputs.weights)
    reported = {
        (fit.regimen.label, fit.cause, fit.horizon): fit.psi_scaled for fit in result.fits.values()
    }
    for r, label in enumerate(REGIMENS):
        for j, (cause, _) in enumerate(_causes(law)):
            for horizon in _horizons(law):
                _same(levels[r, j, horizon - 1], reported[(label, cause, horizon)])


@pytest.mark.parametrize("name", list(LAWS))
def test_e5_a_per_horizon_state_reproduces_its_own_horizon_only(name: str) -> None:
    law = LAWS[name]
    result = _wrong(name, "iterative")
    inputs, start = _internals(result)
    reported = {
        (fit.regimen.label, fit.cause, fit.horizon): fit.psi_scaled for fit in result.fits.values()
    }
    for horizon in _horizons(law):
        targets = next(f.hazard.targets for f in result.fits.values() if f.horizon == horizon)
        state, _ = solve_per_horizon(start, inputs, horizon, targets, max_iter=20, tol=1e-10)
        levels = state.levels(inputs.q, inputs.weights)
        elsewhere = 0.0
        for r, label in enumerate(REGIMENS):
            for j, (cause, _) in enumerate(_causes(law)):
                _same(levels[r, j, horizon - 1], reported[(label, cause, horizon)])
                for other in _horizons(law):
                    if other != horizon:
                        diff = abs(levels[r, j, other - 1] - reported[(label, cause, other)])
                        elsewhere = max(elsewhere, diff)
        assert elsewhere > 1e-6


# ------------------------------------------------------------------------------ E6


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_e6_the_causes_and_survival_add_to_one(targeting: str) -> None:
    law = LAWS["two causes"]
    result = _wrong("two causes", targeting)
    inputs, start = _internals(result)
    if targeting == "one_step":
        record = next(iter(result.fits.values())).hazard
        state = walk_curve(
            start, inputs, _horizons(law), record.targets, step_size=STEP, tol=1e-10
        )[0]
        states = dict.fromkeys(_horizons(law), state)
    else:
        states = {}
        for horizon in _horizons(law):
            targets = next(f.hazard.targets for f in result.fits.values() if f.horizon == horizon)
            states[horizon] = solve_per_horizon(
                start, inputs, horizon, targets, max_iter=20, tol=1e-10
            )[0]
    for horizon, state in states.items():
        levels = state.levels(inputs.q, inputs.weights)[:, :, horizon - 1]
        for r in range(len(REGIMENS)):
            mixture = sum(
                inputs.q[r, :, level] * state.survival[:, horizon - 1, level]
                for level in range(inputs.n_levels)
            )
            survival = float(np.average(mixture, weights=inputs.weights))
            np.testing.assert_allclose(levels[r].sum(), 1.0 - survival, rtol=1e-12)


# ------------------------------------------------------------------------------ E7


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_e7_a_point_mass_policy_is_the_static_arm(targeting: str) -> None:
    law = SMALL
    policy = _fit(law, targeting, learner=ArmOnly(), regimens={"one": POINT_MASS})
    arm = _fit(law, targeting, learner=ArmOnly(), regimens={"one": 1})
    for name in arm.estimates:
        assert policy[name].psi == arm[name].psi
        assert np.array_equal(policy[name].influence_curve, arm[name].influence_curve)


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_e7_a_stochastic_policy_is_the_g_formula_and_doubly_robust(targeting: str) -> None:
    law = SMALL
    rows = law.first_row_of()
    saturated = _fit(law, targeting, regimens={"q": STOCHASTIC})
    wrong = _fit(law, targeting, learner=ArmOnly(), regimens={"q": STOCHASTIC})
    for horizon in _horizons(law):
        name = _name("q", None, horizon)
        truth = law.functional(law.probs, known(POLICY), horizon)
        assert saturated[name].psi == pytest.approx(truth, rel=1e-12, abs=FLOOR)
        np.testing.assert_allclose(
            saturated[name].influence_curve[rows],
            law.eif(known(POLICY), horizon),
            rtol=1e-10,
            atol=1e-12,
        )
        assert wrong[name].psi == pytest.approx(truth, abs=1e-8)


# ------------------------------------------------------------------------------ E8


def _weighted(law: PointLaw) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """The law's frame with integer unit weights, and the frame with each row repeated."""
    frame = law.frame()
    weights = 1 + frame["W"].to_numpy(dtype=int) + (frame["A"].to_numpy() == 2).astype(int)
    weighted = frame.assign(wt=weights.astype(float))
    repeated = frame.loc[np.repeat(frame.index.to_numpy(), weights)].reset_index(drop=True)
    return weighted, repeated, weights


@pytest.mark.parametrize("targeting", ALGORITHMS)
@pytest.mark.parametrize("name", ["one cause", "two causes"])
def test_e8_a_weighted_fit_is_the_fit_on_repeated_rows(name: str, targeting: str) -> None:
    """Integer weights are row repetitions: the weighted law passes E1, E2 and E3.

    Saturated, the weighted fit equals the fit on repeated rows, and its influence curve
    is the repeated fit's scaled by the normalised weight.  With the wrong hazard learner
    the weighted estimate equals the truth of the tilted law, which is the saturated value.
    The wrong-hazard influence curves are not compared: the targeted hazards depend on the
    path, and a one-step walk under weights follows a different metric.
    """
    law = SMALL if name == "one cause" else LAWS[name]
    weighted, repeated, weights = _weighted(law)
    first = np.concatenate([[0], np.cumsum(weights)[:-1]])
    scale = weights / weights.mean()
    with pytest.warns(WeightingWarning):
        exact = _fit(law, targeting, frame=weighted, weights="wt")
    truth = _fit(law, targeting, frame=repeated)
    with pytest.warns(WeightingWarning):
        wrong = _fit(law, targeting, learner=ArmOnly(), frame=weighted, weights="wt")
    for key in truth.estimates:
        assert exact[key].psi == pytest.approx(truth[key].psi, rel=1e-12, abs=FLOOR)
        np.testing.assert_allclose(
            exact[key].influence_curve,
            scale * truth[key].influence_curve[first],
            rtol=1e-10,
            atol=1e-12,
        )
        assert wrong[key].psi == pytest.approx(truth[key].psi, abs=1e-8)


# ----------------------------------------------------------------------------- E10


@pytest.mark.parametrize("targeting", ALGORITHMS)
def test_e10_a_replay_at_the_fitted_bound_is_bit_identical(targeting: str) -> None:
    result = _wrong("two causes", targeting)
    replay = _refit_bound(result, result.replay_recipe, result.config.g_bounds)
    assert _fitted_replay_matches(result, replay)
    curve = longitudinal_truncation_curve(result, [result.config.g_bounds, 0.3])
    fitted = curve[curve["is_fitted_bound"]]
    assert (fitted["delta_from_fitted"] == 0.0).all()


# ----------------------------------------------------------------------------- E11


@pytest.mark.parametrize("name", list(LAWS))
def test_e11_the_walk_trace(name: str) -> None:
    law = LAWS[name]
    result = _wrong(name, "one_step")
    record = next(iter(result.fits.values())).hazard
    norms = np.asarray(record.norm_trace)
    assert np.all(np.diff(norms) <= 0.0)
    losses = np.asarray(record.walk_loss_trace)
    assert np.all(np.diff(losses) <= WALK_LOSS_TOLERANCE)
    assert len(record.trace) == len(norms) == record.n_iter + 1
    assert record.relative_score == pytest.approx(record.trace[-1], rel=0, abs=0)
    inputs, start = _internals(result)
    _, curves = influence(start, inputs, _horizons(law))
    sigma = np.maximum(np.mean(curves**2, axis=-1), SIGMA_FLOOR).reshape(-1)
    assert np.array_equal(record.sigma, sigma)


# ---------------------------------------------------------------- the MLE fallback


@pytest.mark.parametrize("name", list(LAWS))
def test_the_maximum_likelihood_fallback_alone_repairs_a_wrong_hazard(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-horizon targeting with every secant step refused runs Moore's MLE step only.

    The secant step never fails on these laws, so this forces the fallback path and asks
    it for E3 and E4.
    """
    monkeypatch.setattr(hazard_module, "_secant_step", lambda *args, **kwargs: None)
    law = LAWS[name]
    result = _fit(law, "iterative", learner=ArmOnly())
    for key, fit in result.fits.items():
        assert fit.hazard.converged, key
        assert fit.hazard.relative_score <= 1e-10, key
        arm = REGIMENS[fit.regimen.label]
        code = 1 if fit.cause in (None, "relapse") else 2
        truth = law.functional(law.probs, static(arm), fit.horizon, code)
        assert fit.psi_scaled == pytest.approx(truth, abs=1e-8), key
