"""``Targeting(step_bound=)``: the ``tmle3`` constrained step, its refusals and its plumbing.

``step_bound=delta`` replaces each Newton step of the targeting loop by a step along the
normalised score ``d = s / ||s||``, at most ``delta`` long.  With ``step_search=True`` the length
``t*`` minimises the weighted mean quasi-loss along ``d`` over ``[0, delta]``, and with
``step_search=False`` it is ``delta``.  One score column makes ``d = sign(s)``.

The reference below is written from the ``tmle3`` ``fit_submodel`` description with ``numpy`` and
``scipy`` only, so the solver's helper is never compared with itself.  The mutation controls break
one line of ``_bounded_step`` each and ask the same reference comparison to fail.  A ``t*`` that is
not tiny is compared at ``1e-7``: the line search stops within ``xatol`` of ``1.5e-8``, and a step
below about ``1e-5`` sits at the noise of that stop.
"""

from __future__ import annotations

import importlib
import re
import warnings
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import SDRMethod, Targeting, TMLEMethod
from cleverly._refusal import RefusalReason
from cleverly.datasets import (
    make_binary_outcome,
    make_instrument,
    make_linear_ate,
    make_longitudinal,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.estimators.targeting import TargetingSpec, solve_submodel
from cleverly.exceptions import ConvergenceWarning, MethodConfigurationError
from cleverly.fluctuation import (
    InitialFit,
    mean_submodel,
    solve_fluctuation,
)
from cleverly.fluctuation import iterative as iterative_module
from cleverly.fluctuation._score import (
    LINEAR_RULE_REFUSAL,
    STEP_BOUND_ONE_STEP_REFUSAL,
    STEP_SEARCH_REFUSAL,
    StopContext,
    check_step_bound,
)
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR
from cleverly.longitudinal.sequential import _fluctuate_node
from cleverly.methods import IPSI_NO_FLUCTUATION, IPSIMethod, sdr_no_fluctuation_refusal
from cleverly.utils.bounds import expit, logit
from tests.conftest import SELECTOR_CONFIGS, linear_in_sample
from tests.unit import test_stopping_rule_engines as engines
from tests.unit import test_stopping_rule_longitudinal as longitudinal
from tests.unit._source_mutation import mutate

targeting_module = importlib.import_module("cleverly.estimators.targeting")
estimator_module = importlib.import_module("cleverly.longitudinal.estimator")

LO, HI = -3.0, 9.0
SPAN = HI - LO
XATOL = 1.4901161193847656e-08
ALPHA = 0.9995

#: The plan's pinned refusal texts (plan I-2.8).
R_C1 = (
    "step_bound must be a positive finite number; got {value!r}. "
    "step_bound=None takes the full Newton step"
)
R_C2 = (
    "algorithm='one_step' already walks the universal least-favorable submodel in steps of "
    "step_size. step_bound= bounds the steps of algorithm='iterative'."
)
R_C3 = "step_search= chooses how a bounded step is taken, so it needs step_bound="


# ------------------------------------------------------------------------------- the reference


def reference_walk(
    y: np.ndarray,
    p0: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    delta: float,
    search: bool,
    steps: int,
    *,
    mask: np.ndarray | None = None,
    fit_x: np.ndarray | None = None,
    fit_w: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """The ``tmle3`` constrained walk, one record per step, from first principles.

    ``s`` is ``mean(w h (Y - Q*))`` over all rows, zero where the outcome is unobserved.  The
    direction is ``s / ||s||_2``.  The length is the minimiser of the weighted mean quasi-loss
    along the direction by a bounded SciPy search with ``xatol = sqrt(eps)``, or ``delta``.  The
    loss reads the fit's own covariate ``fit_x`` and weights ``fit_w``, which differ from
    ``(x, w)`` only in the weighted form (``target_weights``).
    """
    n = y.shape[0]
    seen = np.ones(n, dtype=bool) if mask is None else mask
    cov = x if fit_x is None else fit_x
    wt = w if fit_w is None else fit_w
    p = np.clip(p0, 1.0 - ALPHA, ALPHA)
    epsilon = np.zeros(x.shape[1])
    records: list[dict[str, Any]] = []
    for _ in range(steps):
        residual = np.where(seen, y - p, 0.0)
        score = np.mean(w[:, None] * x * residual[:, None], axis=0)
        norm = float(np.sqrt(np.sum(score**2)))
        if norm == 0.0:
            break
        direction = score / norm
        covariate = cov @ direction

        def risk(t: float, p: np.ndarray = p, covariate: np.ndarray = covariate) -> float:
            q = expit(logit(p) + t * covariate)
            loss = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
            return float(np.sum(wt[seen] * loss[seen]) / np.sum(wt[seen]))

        if search:
            found = minimize_scalar(
                risk, bounds=(0.0, delta), method="bounded", options={"xatol": XATOL}
            )
            length = float(found.x)
        else:
            length = delta
        epsilon = epsilon + length * direction
        p = np.clip(expit(logit(p) + length * covariate), 1.0 - ALPHA, ALPHA)
        records.append(
            {"t": length, "d": direction, "score": score, "epsilon": epsilon.copy(), "p": p.copy()}
        )
    return records


def setting(seed: int = 3, n_units: int = 60, per_unit: int = 4) -> dict[str, Any]:
    """Clustered continuous outcome on ``[LO, HI]``, rescaled to ``[0, 1]``, a poor initial fit."""
    rng = np.random.default_rng(seed)
    n = n_units * per_unit
    shared = np.repeat(rng.normal(size=n_units), per_unit)
    w = rng.normal(size=n) + shared
    g1 = expit(0.5 * w)
    a = rng.binomial(1, g1).astype(float)
    raw = 2 + 1.5 * a + w + rng.normal(size=n) + np.repeat(rng.normal(size=n_units), per_unit)
    return {
        "n": n,
        "units": np.repeat(np.arange(n_units), per_unit),
        "g1": g1,
        "a": a,
        "y": np.clip((raw - LO) / SPAN, 0.01, 0.99),
        "initial": InitialFit(np.full(n, 0.4), {0.0: np.full(n, 0.4), 1.0: np.full(n, 0.4)}),
        "ones": np.ones(n),
    }


def walk_solver(s: dict[str, Any], steps: int, **kwargs: Any) -> Any:
    submodel = mean_submodel(s["a"], s["g1"])
    return solve_fluctuation(
        s["y"], s["initial"], submodel, s["ones"], None, max_iter=steps, warn=False, **kwargs
    )


def deviation(s: dict[str, Any], delta: float, search: bool, steps: int = 4) -> float:
    """Largest gap between the solver and the reference, in ``epsilon`` and in the fitted ``Q*``."""
    submodel = mean_submodel(s["a"], s["g1"])
    walk = reference_walk(
        s["y"], s["initial"].observed, submodel.observed, s["ones"], delta, search, steps
    )
    worst = 0.0
    for k, record in enumerate(walk, start=1):
        fit = walk_solver(s, k, step_bound=delta, step_search=search)
        worst = max(
            worst,
            float(np.max(np.abs(fit.epsilon - record["epsilon"]))),
            float(np.max(np.abs(fit.targeted.observed - record["p"]))),
        )
    return worst


DELTA = 0.15


# ----------------------------------------------------------------------------- the exact steps


class TestEachStepAgainstTheReference:
    def test_the_searched_walk_takes_the_reference_step_at_every_iteration(self) -> None:
        s = setting()
        assert deviation(s, DELTA, True, steps=6) < 1e-7

    def test_the_fixed_walk_takes_the_whole_bound_at_every_iteration(self) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        walk = reference_walk(
            s["y"], s["initial"].observed, submodel.observed, s["ones"], 0.1, False, 5
        )
        assert deviation(s, 0.1, False, steps=5) < 1e-12
        previous = np.zeros(2)
        for record in walk:
            # Each step has length delta exactly and points along the normalised score.
            np.testing.assert_allclose(
                np.linalg.norm(record["epsilon"] - previous), 0.1, atol=1e-15
            )
            np.testing.assert_allclose(
                record["epsilon"] - previous,
                0.1 * record["score"] / np.linalg.norm(record["score"]),
                atol=1e-15,
            )
            previous = record["epsilon"]

    def test_the_witnesses_the_exact_tests_sit_above(self) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        newton = walk_solver(s, 1)
        walk = reference_walk(
            s["y"], s["initial"].observed, submodel.observed, s["ones"], DELTA, True, 4
        )
        lengths = [record["t"] for record in walk]
        # (1) The first bounded step is below the Newton step, and it binds.
        assert lengths[0] == pytest.approx(DELTA, abs=1e-6)
        assert float(np.linalg.norm(newton.epsilon)) > DELTA + 0.05
        # (2) A later step is interior: the search, not the bound, chose it.
        assert 1e-2 < lengths[1] < DELTA - 0.05
        # (3) The loss falls at every accepted step, so the sign of the direction is right.
        losses = []
        p = np.clip(s["initial"].observed, 1 - ALPHA, ALPHA)
        for record in [{"p": p}, *walk]:
            q = record["p"]
            losses.append(float(-np.mean(s["y"] * np.log(q) + (1 - s["y"]) * np.log(1 - q))))
        assert all(b < a for a, b in pairwise(losses))

    def test_a_walk_cut_by_max_iter_moves_the_estimate_off_the_newton_fit(self) -> None:
        s = setting()
        newton = walk_solver(s, 20)
        cut = walk_solver(s, 2, step_bound=0.01)
        assert not cut.converged and cut.failure == "max_iter_reached" and cut.n_iter == 2
        assert float(np.max(np.abs(cut.targeted.observed - newton.targeted.observed))) > 1e-3
        assert float(np.max(np.abs(cut.epsilon - newton.epsilon))) > 0.15

    def test_an_unbounded_walk_is_the_shipped_solve(self) -> None:
        s = setting()
        plain = walk_solver(s, 20)
        named = walk_solver(s, 20, step_bound=None, step_search=True)
        np.testing.assert_array_equal(plain.epsilon, named.epsilon)
        np.testing.assert_array_equal(plain.targeted.observed, named.targeted.observed)
        assert plain.trace == named.trace and plain.n_iter == named.n_iter == 1

    def test_a_bounded_walk_reaches_the_relative_tolerance_when_the_search_is_on(self) -> None:
        s = setting()
        fit = walk_solver(s, 100, step_bound=DELTA)
        assert fit.converged and fit.failure is None and fit.n_iter < 100
        assert fit.relative_score_norm <= 1e-10
        # It lands on the Newton solution of the same score equation.
        newton = walk_solver(s, 20)
        np.testing.assert_allclose(fit.epsilon, newton.epsilon, atol=2e-6)

    def test_a_fixed_step_walk_oscillates_about_the_root_and_never_converges(self) -> None:
        s = setting()
        with pytest.warns(ConvergenceWarning, match=r"step_bound=0\.1 limits each iteration"):
            fit = solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                max_iter=60,
                step_bound=0.1,
                step_search=False,
            )
        assert not fit.converged and fit.n_iter == 60
        # The score stalls at the size of one step; no iterate lands inside the tolerance.
        assert min(fit.trace) > 1e-6

    def test_a_score_that_is_exactly_zero_takes_no_step(self) -> None:
        n = 40
        y = np.full(n, 0.4)
        initial = InitialFit(np.full(n, 0.4), {0.0: np.full(n, 0.4), 1.0: np.full(n, 0.4)})
        submodel = mean_submodel(np.tile([0.0, 1.0], n // 2), np.full(n, 0.5))
        fit = solve_fluctuation(y, initial, submodel, np.ones(n), None, step_bound=0.1)
        assert fit.n_iter == 0 and fit.converged and fit.failure is None
        np.testing.assert_array_equal(fit.epsilon, np.zeros(2))
        assert fit.trace == (0.0,)


# ----------------------------------------------------------------------- mutation controls


def edit(
    monkeypatch: pytest.MonkeyPatch, old: str, new: str, *, owner: Any = iterative_module
) -> None:
    mutate(monkeypatch, iterative_module, owner, "_bounded_step", [(old, new)])


class TestMutationControls:
    """Each control passes the clean comparison first, then fails it with one line broken."""

    @pytest.fixture
    def clean(self) -> dict[str, Any]:
        s = setting()
        assert deviation(s, DELTA, True) < 1e-7
        assert deviation(s, 0.1, False) < 1e-12
        return s

    def test_the_unnormalised_score_direction_fails(
        self, monkeypatch: pytest.MonkeyPatch, clean: dict[str, Any]
    ) -> None:
        edit(monkeypatch, "direction = score / np.linalg.norm(score)", "direction = score")
        assert deviation(clean, 0.1, False) > 1e-3
        assert deviation(clean, DELTA, True) > 1e-3

    def test_the_reversed_direction_fails(
        self, monkeypatch: pytest.MonkeyPatch, clean: dict[str, Any]
    ) -> None:
        edit(
            monkeypatch,
            "direction = score / np.linalg.norm(score)",
            "direction = -score / np.linalg.norm(score)",
        )
        assert deviation(clean, 0.1, False) > 1e-2
        assert deviation(clean, DELTA, True) > 1e-2

    def test_a_loose_line_search_fails(
        self, monkeypatch: pytest.MonkeyPatch, clean: dict[str, Any]
    ) -> None:
        edit(monkeypatch, '"xatol": _STEP_SEARCH_XATOL', '"xatol": 1e-3')
        assert deviation(clean, DELTA, True) > 1e-6

    def test_an_upper_bound_of_twice_the_step_bound_fails(
        self, monkeypatch: pytest.MonkeyPatch, clean: dict[str, Any]
    ) -> None:
        edit(monkeypatch, "bounds=(0.0, delta)", "bounds=(0.0, 2.0 * delta)")
        assert deviation(clean, DELTA, True) > 0.05

    def test_the_newton_direction_fails_on_two_columns(
        self, monkeypatch: pytest.MonkeyPatch, clean: dict[str, Any]
    ) -> None:
        edit(
            monkeypatch,
            "direction = score / np.linalg.norm(score)",
            "p0 = expit(offset)\n"
            "    hessian = x.T @ (x * (weights * p0 * (1.0 - p0))[:, None])\n"
            "    direction = np.linalg.solve(hessian, score)\n"
            "    direction = direction / np.linalg.norm(direction)",
        )
        assert deviation(clean, 0.1, False) > 1e-3

    def test_a_search_that_reads_the_wrong_weights_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = setting()
        s["ones"] = 0.5 + np.arange(s["n"]) / s["n"]
        assert deviation(s, DELTA, True) < 1e-7
        edit(
            monkeypatch,
            "-quasi_loglik(y, expit(offset + t * covariate), weights) / total",
            "-quasi_loglik(y, expit(offset + t * covariate), np.ones_like(weights)) / total",
        )
        assert deviation(s, DELTA, True) > 1e-8


# ---------------------------------------------------------------- the weighted and observed rows


class TestWeightsAndMissingOutcomes:
    def test_weights_and_an_unobserved_mask_follow_the_reference(self) -> None:
        s = setting()
        rng = np.random.default_rng(8)
        weights = rng.uniform(0.4, 1.8, size=s["n"])
        weights /= weights.mean()
        observed = rng.random(s["n"]) > 0.25
        submodel = mean_submodel(s["a"], s["g1"])
        walk = reference_walk(
            s["y"], s["initial"].observed, submodel.observed, weights, 0.1, True, 4, mask=observed
        )
        for k, record in enumerate(walk, start=1):
            fit = solve_fluctuation(
                s["y"],
                s["initial"],
                submodel,
                weights,
                observed,
                max_iter=k,
                warn=False,
                step_bound=0.1,
            )
            np.testing.assert_allclose(fit.epsilon, record["epsilon"], atol=1e-7)
            np.testing.assert_allclose(fit.targeted.observed, record["p"], atol=1e-7)

    @pytest.mark.parametrize("search", [True, False])
    def test_the_weighted_form_follows_the_unweighted_score_in_its_own_coordinates(
        self, search: bool
    ) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        magnitude = np.abs(submodel.observed).sum(axis=1)
        walk = reference_walk(
            s["y"],
            s["initial"].observed,
            submodel.observed,
            s["ones"],
            0.1,
            search,
            4,
            fit_x=np.sign(submodel.observed),
            fit_w=s["ones"] * magnitude,
        )
        for k, record in enumerate(walk, start=1):
            fit = walk_solver(s, k, step_bound=0.1, step_search=search, target_weights=True)
            np.testing.assert_allclose(fit.epsilon, record["epsilon"], atol=1e-7)
            np.testing.assert_allclose(fit.targeted.observed, record["p"], atol=1e-7)

    def test_carried_fits_move_by_the_same_bounded_steps(self) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        carried = InitialFit(
            np.full(s["n"], 0.3), {0.0: np.full(s["n"], 0.3), 1.0: np.full(s["n"], 0.3)}
        )
        fit = solve_fluctuation(
            s["y"],
            s["initial"],
            submodel,
            s["ones"],
            None,
            carry=(carried,),
            max_iter=3,
            warn=False,
            step_bound=0.1,
            step_search=False,
        )
        moved = expit(
            logit(np.clip(carried.observed, 1 - ALPHA, ALPHA)) + submodel.observed @ fit.epsilon
        )
        # Three steps of a shrunk logistic walk: the carried fit is the endpoint, not a refit.
        np.testing.assert_allclose(fit.carried[0].observed, moved, atol=1e-12)


# --------------------------------------------------------------------------- one column nodes


class TestTheLongitudinalNode:
    @staticmethod
    def node(**overrides: Any) -> dict[str, Any]:
        node = longitudinal.crafted_node()
        return {**node, **overrides}

    @pytest.mark.parametrize("search", [True, False])
    def test_the_node_steps_along_the_sign_of_its_score(self, search: bool) -> None:
        base = self.node()
        y, p0, w = base["pseudo_outcome"], base["initial"], base["loss_weights"]
        x = np.ones((y.shape[0], 1))
        walk = reference_walk(y, p0, x, w, 0.05, search, 6)
        for k, record in enumerate(walk, start=1):
            fit = _fluctuate_node(**{**base, "max_iter": k}, step_bound=0.05, step_search=search)
            np.testing.assert_allclose(fit.epsilon, record["epsilon"], atol=1e-7)
            # One column: the direction is the sign of the score, a unit step.
            assert record["d"][0] == np.sign(record["score"][0])
        assert any(record["t"] == pytest.approx(0.05, abs=1e-6) for record in walk)

    def test_the_default_node_is_untouched_by_a_none_bound(self) -> None:
        base = self.node()
        plain = _fluctuate_node(**base)
        named = _fluctuate_node(**base, step_bound=None, step_search=True)
        np.testing.assert_array_equal(plain.epsilon, named.epsilon)
        assert plain.n_iter == named.n_iter

    def test_a_constant_node_is_not_solved_under_a_bound(self) -> None:
        base = self.node()
        n = base["pseudo_outcome"].shape[0]
        base["pseudo_outcome"] = np.zeros(n)
        base["initial"] = np.zeros(n)
        fit = _fluctuate_node(**base, step_bound=0.05)
        assert fit.converged and fit.n_iter == 0 and fit.stopping == "relative"
        assert fit.thresholds is None and fit.rule_ratio is None
        ruled = _fluctuate_node(
            **base, step_bound=0.05, stopping="sample_size", stop=StopContext(None, 1.0)
        )
        assert ruled.converged and ruled.rule_ratio == 0.0

    def test_a_declared_rule_with_a_bound_still_needs_its_context(self) -> None:
        with pytest.raises(ValueError, match="StopContext"):
            _fluctuate_node(**self.node(), step_bound=0.05, stopping="sample_size")


def every_node(result: Any) -> list[tuple[str, Any]]:
    return longitudinal.every_node(result)


def assert_node_walks(result: Any, delta: float, search: bool) -> None:
    """Every per-regimen node's ``epsilon`` is the reference walk from the node's own state."""
    nodes = every_node(result)
    assert nodes
    moved = 0
    for key, step in nodes:
        fluctuation = step.fluctuation
        trained = np.asarray(step.trained_on, dtype=bool)
        y = np.asarray(step.pseudo_outcome, dtype=float)
        w = np.asarray(result.data.weights, dtype=float) * np.asarray(step.clever, dtype=float)
        walk = reference_walk(
            y,
            np.asarray(step.initial, dtype=float),
            np.ones((y.shape[0], 1)),
            w,
            delta,
            search,
            fluctuation.n_iter,
            mask=trained,
        )
        if fluctuation.n_iter and walk:
            np.testing.assert_allclose(
                fluctuation.epsilon, walk[-1]["epsilon"], atol=2e-7, err_msg=key
            )
            moved += 1
    assert moved > 0


BOUND = 0.05


def ltmle(**settings: Any) -> LTMLE:
    return LTMLE(
        longitudinal.PLANS,
        step_bound=BOUND,
        max_iter=8,
        **longitudinal.learners(),
        **settings,
    )


def bounded_ltmle(*args: Any, **kwargs: Any) -> LTMLE:
    """``LTMLE`` with the module's bound, for the fit builders of the stopping-rule tests."""
    kwargs.pop("stopping", None)
    return LTMLE(*args, step_bound=BOUND, max_iter=8, **kwargs)


def fitted(estimator: LTMLE, frame: Any, **columns: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return estimator.fit(frame, outcome="Y", **longitudinal.NODES, **columns)


class TestEveryPerRegimenEngine:
    @pytest.mark.parametrize("search", [True, False])
    @pytest.mark.parametrize("n_folds", [1, 3])
    def test_every_node_of_an_end_of_study_fit_follows_the_reference(
        self, search: bool, n_folds: int
    ) -> None:
        frame, _ = make_longitudinal(n=300, seed=12)
        result = fitted(ltmle(n_folds=n_folds, step_search=search), frame)
        assert_node_walks(result, BOUND, search)

    def test_a_survival_fit(self) -> None:
        frame, _ = longitudinal.make_longitudinal_survival(n=300, seed=1)
        estimator = ltmle()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = estimator.fit(frame, **longitudinal.SURVIVAL)
        assert_node_walks(result, BOUND, True)

    def test_a_competing_risks_fit(self) -> None:
        frame, _ = longitudinal.make_longitudinal_competing(n=300, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = ltmle(n_folds=3).fit(frame, **longitudinal.COMPETING)
        assert_node_walks(result, BOUND, True)

    def test_a_clustered_fit_under_a_declared_rule_and_a_bound(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12, cluster_size=8)
        result = fitted(ltmle(stopping="scaled_var"), frame, id="id")
        longitudinal.assert_hand_match(result, "scaled_var")
        assert_node_walks(result, BOUND, True)

    def test_every_supported_composition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Static, dynamic, policy, MTP, continuous, clustered, weighted, held, known, folds."""
        monkeypatch.setattr(longitudinal, "LTMLE", bounded_ltmle)
        for composition in longitudinal.COMPOSITIONS.values():
            frame = longitudinal.repeated_module.panel(
                continuous=composition.continuous, held=composition.held
            )
            result = longitudinal.end_of_study_panel("relative", composition, frame)
            assert_node_walks(result, BOUND, True)

    def test_a_repeated_outcome_walks_every_horizon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(longitudinal, "LTMLE", bounded_ltmle)
        frame = longitudinal.repeated_module.panel()
        for name in ("static", "static_folds"):
            result = longitudinal.repeated_panel("relative", longitudinal.COMPOSITIONS[name], frame)
            assert result.data.is_repeated
            assert {fit.horizon for fit in result.fits.values()} == {1, 2, 3}
            assert_node_walks(result, BOUND, True)

    def test_a_long_panel_walks_each_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(longitudinal, "LTMLE", bounded_ltmle)
        result = longitudinal.long_panel_fit("relative")
        assert result.data.panel is not None
        assert_node_walks(result, BOUND, True)

    def test_the_working_model_walks_the_stacked_node_with_every_term(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        original = iterative_module._bounded_step

        def spy(x: Any, *args: Any) -> Any:
            seen.append(int(x.shape[1]))
            return original(x, *args)

        monkeypatch.setattr(iterative_module, "_bounded_step", spy)
        estimator = LTMLE(
            longitudinal.PLANS,
            msm=longitudinal.msm_module.DOSE,
            n_folds=1,
            simultaneous=False,
            step_bound=BOUND,
            outcome_learner=LogisticRegression(max_iter=1000, random_state=0),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=0),
            censoring_learner=LogisticRegression(max_iter=1000, random_state=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit(
                longitudinal.msm_module.panel(120),
                outcome="Y",
                treatment=("A1", "A2"),
                baseline=("W1", "W2"),
                time_varying=(("L1",), ("L2",)),
                censoring=("C1", "C2"),
            )
        (fit,) = result.msm_fits
        assert any(width == len(fit.model.terms) > 1 for width in seen)
        assert np.all(np.isfinite([e.psi for e in result.estimates.values()]))


# ------------------------------------------------------------------------------ the refusals


class TestRefusals:
    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), "0.1", True, [0.1]])
    def test_r_c1_a_bound_that_is_not_positive_and_finite(self, value: Any) -> None:
        expected = re.escape(R_C1.format(value=value))
        for build in (
            lambda: Targeting(step_bound=value),
            lambda: TargetingSpec(step_bound=value),
            lambda: TMLE(**linear_in_sample(step_bound=value)),
            lambda: LTMLE(longitudinal.PLANS, step_bound=value),
        ):
            with pytest.raises(ValueError, match=f"^{expected}$"):
                build()

    def test_r_c2_the_bound_beside_the_one_step_algorithm(self) -> None:
        for build in (
            lambda: Targeting(algorithm="one_step", step_bound=0.1),
            lambda: TargetingSpec(targeting="one_step", step_bound=0.1),
            lambda: TMLE(**linear_in_sample(targeting="one_step", step_bound=0.1)),
        ):
            with pytest.raises(MethodConfigurationError) as raised:
                build()
            assert str(raised.value) == R_C2 == STEP_BOUND_ONE_STEP_REFUSAL.message()
            assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION

    def test_r_c3_the_search_without_a_bound(self) -> None:
        for build in (
            lambda: Targeting(step_search=False),
            lambda: TargetingSpec(step_search=False),
            lambda: TMLE(**linear_in_sample(step_search=False)),
            lambda: LTMLE(longitudinal.PLANS, step_search=False),
        ):
            with pytest.raises(MethodConfigurationError) as raised:
                build()
            assert str(raised.value) == R_C3 == STEP_SEARCH_REFUSAL.message()
            assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_the_search_flag_is_a_bool(self) -> None:
        with pytest.raises(ValueError, match=r"step_search must be True or False; got 'no'"):
            Targeting(step_bound=0.1, step_search="no")  # type: ignore[arg-type]

    def test_r_abc1_the_bound_beside_the_linear_fluctuation(self) -> None:
        for build in (
            lambda: Targeting(fluctuation="linear", step_bound=0.1),
            lambda: TargetingSpec(fluctuation="linear", step_bound=0.1),
            lambda: TMLE(**linear_in_sample(fluctuation="linear", step_bound=0.1)),
        ):
            with pytest.raises(MethodConfigurationError) as raised:
                build()
            assert str(raised.value) == LINEAR_RULE_REFUSAL.message()
            assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_the_solver_refuses_a_bound_with_the_linear_fluctuation(self) -> None:
        s = setting()
        with pytest.raises(MethodConfigurationError) as raised:
            solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                kind="linear",
                step_bound=0.1,
            )
        assert str(raised.value) == LINEAR_RULE_REFUSAL.message()

    def test_the_most_specific_refusal_comes_first(self) -> None:
        with pytest.raises(ValueError, match="step_bound must be"):
            check_step_bound(0.0, False, "linear", "one_step")
        with pytest.raises(MethodConfigurationError) as raised:
            check_step_bound(None, False, "linear", "one_step")
        assert raised.value.refusal is STEP_SEARCH_REFUSAL
        with pytest.raises(MethodConfigurationError) as raised:
            check_step_bound(0.1, True, "linear", "one_step")
        assert raised.value.refusal is STEP_BOUND_ONE_STEP_REFUSAL
        check_step_bound(None, True, "linear", "one_step")
        check_step_bound(np.float64(0.1), np.True_, "logistic", "iterative")
        check_step_bound(1, True, "logistic", "iterative")

    def test_r_abc2_sdr_refuses_a_step_bound_and_a_search_by_name(self) -> None:
        for name, value in (("step_bound", 0.1), ("step_search", False)):
            with pytest.raises(MethodConfigurationError) as raised:
                SDR(longitudinal.PLANS, **{name: value})
            assert str(raised.value) == sdr_no_fluctuation_refusal(name).message()
            assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
            method = SDRMethod(
                targeting=Targeting(
                    **(
                        {"step_bound": 0.1}
                        if name == "step_bound"
                        else {"step_bound": 0.1, "step_search": False}
                    )
                )
            )
            with pytest.raises(MethodConfigurationError) as raised:
                method.estimator_kwargs(longitudinal=True)
            assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert {"step_bound", "step_search"} <= set(estimator_module.SDR_REFUSED_SETTINGS)
        kwargs = SDRMethod().estimator_kwargs(longitudinal=True)
        assert "step_bound" not in kwargs and "step_search" not in kwargs

    def test_r_abc3_ipsi_refuses_a_step_bound_and_a_search(self) -> None:
        for name, value in (("step_bound", 0.1), ("step_search", False)):
            with pytest.raises(MethodConfigurationError) as raised:
                IPSI([Incremental(2.0)], **{name: value})
            assert str(raised.value) == IPSI_NO_FLUCTUATION.message()
            assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        method = IPSIMethod(targeting=Targeting(step_bound=0.1))
        with pytest.raises(MethodConfigurationError) as raised:
            method.estimator_kwargs(longitudinal=True)
        assert str(raised.value) == IPSI_NO_FLUCTUATION.message()
        assert {"step_bound", "step_search"} <= set(estimator_module.IPSI_REFUSED_SETTINGS)
        assert "step_bound" not in IPSIMethod().estimator_kwargs(longitudinal=True)


# ------------------------------------------------------------- is ``relative`` reachable?


class TestRelativeReachability:
    """The measurement behind the ``Targeting(step_bound=)`` docstring.

    On six arm-mean fits (a continuous and a binary outcome, bounds 0.01, 0.05 and 0.25) the
    bounded walk with the search on reaches ``tol=1e-10`` with a sufficient iteration budget.
    The count sits at the round-off floor, so it moves with the platform. Some fits finish
    within the default budget, and others need more iterations. With the search off the walk
    lands within one bound of the root and stays there, so ``relative`` is never reached.
    """

    @staticmethod
    def fit(frame: Any, **settings: Any) -> Any:
        estimator = TMLE(**linear_in_sample(**settings))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return estimator.fit(frame, outcome="Y", treatment="A").single().fluctuations["mean"]

    @pytest.fixture(params=["continuous", "binary"])
    def frame(self, request: pytest.FixtureRequest) -> Any:
        if request.param == "continuous":
            return make_linear_ate(n=400, seed=2)[0]
        return make_binary_outcome(n=400, seed=17)[0]

    @pytest.mark.parametrize("delta", [0.01, 0.05, 0.25])
    def test_the_search_reaches_the_tolerance_with_a_sufficient_iteration_budget(
        self, frame: Any, delta: float
    ) -> None:
        default = self.fit(frame, step_bound=delta)
        assert np.isfinite(default.relative_score_norm)
        assert default.converged == (default.relative_score_norm <= 1e-10)
        long = self.fit(frame, step_bound=delta, max_iter=1000)
        assert long.converged and long.relative_score_norm <= 1e-10
        if default.converged:
            assert default.n_iter <= 20 and default.failure is None
            np.testing.assert_array_equal(default.epsilon, long.epsilon)
            np.testing.assert_array_equal(default.targeted.observed, long.targeted.observed)
            return
        assert default.n_iter == 20
        # The linear learner on a binary outcome pins over 1% of the predictions at the shrink
        # bounds, and the shipped classifier names that before the iteration cap.
        assert default.failure in {"max_iter_reached", "bounds_pinned"}
        assert default.failure == "max_iter_reached" or frame["Y"].nunique() == 2
        # The unconverged walk is close, but not at the tolerance.
        assert 1e-10 < default.relative_score_norm < 1e-7
        assert long.n_iter > default.n_iter

    def test_an_exhausted_walk_recovers_with_a_larger_iteration_budget(self) -> None:
        frame = make_linear_ate(n=400, seed=2)[0]
        short = self.fit(frame, step_bound=0.01, max_iter=1)
        assert short.n_iter == 1 and not short.converged
        assert short.failure == "max_iter_reached"
        assert np.isfinite(short.relative_score_norm) and short.relative_score_norm > 1e-10
        recovered = self.fit(frame, step_bound=0.01, max_iter=1000)
        assert recovered.converged and recovered.failure is None
        assert recovered.n_iter > short.n_iter and recovered.relative_score_norm <= 1e-10

    @pytest.mark.parametrize("delta", [0.01, 0.05, 0.25])
    def test_the_fixed_step_never_reaches_the_tolerance(self, frame: Any, delta: float) -> None:
        fixed = self.fit(frame, step_bound=delta, step_search=False, max_iter=300)
        assert fixed.n_iter == 300 and not fixed.converged
        assert fixed.relative_score_norm > 1e-4

    def test_a_declared_rule_stops_a_bounded_walk_in_a_few_iterations(self, frame: Any) -> None:
        for rule in ("sample_size", "scaled_var"):
            fit = self.fit(frame, step_bound=0.05, stopping=rule)
            assert fit.converged and fit.n_iter <= 5
            assert fit.rule_ratio is not None and fit.rule_ratio <= 1.0

    def test_the_bounded_walk_tests_first_and_the_newton_path_steps_first(self) -> None:
        # The first-step row of the declared differences in `point-treatment-tmle.md`.  The
        # initial fit of this frame already meets `scaled_var`: the bounded walk takes no step
        # and the Newton path takes one, as `tmle3_Update$update` does.
        frame = make_linear_ate(n=400, seed=2)[0]
        bounded = self.fit(frame, step_bound=0.05, stopping="scaled_var")
        newton = self.fit(frame, stopping="scaled_var")
        assert bounded.n_iter == 0 and bounded.converged
        np.testing.assert_array_equal(bounded.epsilon, np.zeros_like(bounded.epsilon))
        assert newton.n_iter == 1 and newton.converged


# ------------------------------------------------------------------------------ the warnings


class TestTheWarning:
    def test_a_walk_that_misses_the_shipped_tolerance_names_the_bound(self) -> None:
        s = setting()
        with pytest.warns(
            ConvergenceWarning,
            match=(
                r"targeting step did not drive the relative score below 1e-10 after 2 "
                r"iteration\(s\) \(relative score = .*step_bound=0\.01 limits each iteration"
            ),
        ):
            solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                max_iter=2,
                step_bound=0.01,
            )

    def test_a_walk_that_misses_a_declared_rule_uses_the_pinned_rule_text(self) -> None:
        s = setting()
        with pytest.warns(
            ConvergenceWarning,
            match=(
                r"targeting step did not meet the sample_size stopping rule after 2 "
                r"iteration\(s\) \(largest score over its threshold = .*step_bound=0\.01 limits"
            ),
        ):
            solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                max_iter=2,
                step_bound=0.01,
                stopping="sample_size",
                stop=StopContext(s["units"], SPAN),
            )

    def test_a_walk_that_meets_its_rule_is_silent(self) -> None:
        s = setting()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                max_iter=500,
                step_bound=0.15,
            )
        assert fit.converged


# ------------------------------------------------------------------------------ the plumbing


class TestPlumbing:
    def test_every_surface_defaults_to_no_bound(self) -> None:
        assert Targeting().step_bound is None and Targeting().step_search is True
        assert TargetingSpec().step_bound is None and TargetingSpec().step_search is True
        estimator = TMLE(**linear_in_sample())
        assert estimator.step_bound is None and estimator.step_search is True
        spec = estimator.targeting_spec()
        assert spec.step_bound is None and spec.step_search is True
        engine = LTMLE(longitudinal.PLANS)
        assert engine.step_bound is None and engine.step_search is True

    def test_the_shortcuts_route_to_the_targeting_group(self) -> None:
        method = TMLEMethod().with_overrides(step_bound=0.05, step_search=False)
        assert method.targeting.step_bound == 0.05 and method.targeting.step_search is False
        point = method.estimator_kwargs()
        assert point["step_bound"] == 0.05 and point["step_search"] is False
        engine = method.estimator_kwargs(longitudinal=True)
        assert engine["step_bound"] == 0.05 and engine["step_search"] is False
        defaults = TMLEMethod().estimator_kwargs()
        assert defaults["step_bound"] is None and defaults["step_search"] is True
        assert LTMLE(longitudinal.PLANS, **engine).step_bound == 0.05

    def test_the_spec_reaches_the_solver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        searched = TargetingSpec(step_bound=0.3, step_search=False, max_iter=1)
        fit = solve_submodel(
            s["y"],
            s["initial"],
            submodel,
            s["ones"],
            np.ones(s["n"], dtype=bool),
            searched,
            warn=False,
        )
        walk = reference_walk(
            s["y"], s["initial"].observed, submodel.observed, s["ones"], 0.3, False, 1
        )
        np.testing.assert_allclose(fit.epsilon, walk[0]["epsilon"], atol=1e-15)
        plain = solve_submodel(
            s["y"],
            s["initial"],
            submodel,
            s["ones"],
            np.ones(s["n"], dtype=bool),
            TargetingSpec(max_iter=1),
            warn=False,
        )
        assert float(np.max(np.abs(plain.epsilon - fit.epsilon))) > 0.05
        # Dropping either field on the way to the solver is a different fit.
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_solve_submodel_ruled",
            [("step_search=spec.step_search,", "step_search=True,")],
        )
        broken = solve_submodel(
            s["y"],
            s["initial"],
            submodel,
            s["ones"],
            np.ones(s["n"], dtype=bool),
            searched,
            warn=False,
        )
        assert float(np.max(np.abs(broken.epsilon - walk[0]["epsilon"]))) > 1e-3

    def test_dropping_the_bound_on_the_way_to_the_solver_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        spec = TargetingSpec(step_bound=0.05, max_iter=1)
        observed = np.ones(s["n"], dtype=bool)
        bounded = solve_submodel(
            s["y"], s["initial"], submodel, s["ones"], observed, spec, warn=False
        )
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_solve_submodel_ruled",
            [("step_bound=spec.step_bound,", "step_bound=None,")],
        )
        broken = solve_submodel(
            s["y"], s["initial"], submodel, s["ones"], observed, spec, warn=False
        )
        assert float(np.linalg.norm(bounded.epsilon)) == pytest.approx(0.05, abs=1e-6)
        assert float(np.linalg.norm(broken.epsilon)) > 0.2

    def test_a_default_fit_equals_a_fit_that_names_the_defaults(self) -> None:
        frame = make_linear_ate(n=300, seed=2)[0]
        plain = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        named = (
            TMLE(**linear_in_sample(step_bound=None, step_search=True))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        for key, estimate in plain.estimates.items():
            assert estimate.psi == named.estimates[key].psi
            np.testing.assert_array_equal(
                estimate.influence_curve, named.estimates[key].influence_curve
            )
        for group, fluctuation in plain.fluctuations.items():
            np.testing.assert_array_equal(fluctuation.epsilon, named.fluctuations[group].epsilon)

    def test_a_ltmle_default_fit_equals_a_fit_that_names_the_defaults(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
        plain = fitted(LTMLE(longitudinal.PLANS, **longitudinal.learners()), frame)
        named = fitted(
            LTMLE(longitudinal.PLANS, step_bound=None, step_search=True, **longitudinal.learners()),
            frame,
        )
        for key, estimate in plain.estimates.items():
            assert estimate.psi == named.estimates[key].psi
            np.testing.assert_array_equal(
                estimate.influence_curve, named.estimates[key].influence_curve
            )

    def test_the_replay_recipe_and_a_truncation_replay_repeat_the_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
        result = fitted(ltmle(step_search=False), frame)
        assert (
            result.replay_recipe.step_bound == BOUND and result.replay_recipe.step_search is False
        )
        seen: list[tuple[Any, Any]] = []
        original = estimator_module._run_recursion

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append((kwargs["step_bound"], kwargs["step_search"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(estimator_module, "_run_recursion", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = result.diagnostics.truncation_curve([result.config.g_bounds, (0.05, 1.0)])
        assert len(curve) >= 2 and seen
        assert all(pair == (BOUND, False) for pair in seen)

    def test_a_cluster_bootstrap_refits_under_the_bound(self) -> None:
        frame, _ = make_longitudinal(n=160, seed=4, cluster_size=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ltmle(n_folds=1, n_bootstrap=4).fit(
                frame, outcome="Y", **longitudinal.NODES, id="id"
            )
        assert result.estimates

    def test_a_point_result_round_trips_the_bound(self) -> None:
        frame = make_linear_ate(n=200, seed=2)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = (
                TMLE(**linear_in_sample(step_bound=0.05, step_search=False))
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
        restored = loads(dumps(result))
        spec = restored.config.targeting_spec
        assert spec.step_bound == 0.05 and spec.step_search is False


# --------------------------------------------------------------------- every point group fits


class TestEveryPointGroupFitsUnderABound:
    @pytest.mark.parametrize("search", [True, False])
    @pytest.mark.parametrize("kind", sorted(kind for kind in engines.KINDS if kind != "one-step"))
    def test_the_fit_runs_and_walks_the_bounded_step(
        self, kind: str, search: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, columns, settings, _ = engines.KINDS[kind]()
        lengths: list[float] = []
        original = iterative_module._bounded_step

        def spy(*args: Any) -> Any:
            step, length = original(*args)
            lengths.append(length)
            return step, length

        monkeypatch.setattr(iterative_module, "_bounded_step", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator = TMLE(**linear_in_sample(step_bound=0.05, step_search=search, **settings))
            result = estimator.fit(frame, outcome="Y", treatment="A", **columns).single()
        assert lengths, kind
        assert max(lengths) <= 0.05 + 1e-12
        if not search:
            assert all(length == 0.05 for length in lengths)
        assert all(np.isfinite(e.psi) for e in result.estimates.values())
        for fluctuation in result.fluctuations.values():
            assert fluctuation.method == "iterative" and fluctuation.stopping == "relative"

    def test_a_bounded_walk_under_a_declared_rule_stops_at_the_hand_ratio(self) -> None:
        s = setting()
        submodel = mean_submodel(s["a"], s["g1"])
        observed = np.ones(s["n"], dtype=bool)
        stop = StopContext(s["units"], SPAN)

        def at(max_iter: int) -> Any:
            spec = TargetingSpec(step_bound=0.01, stopping="sample_size", max_iter=max_iter)
            return solve_submodel(
                s["y"], s["initial"], submodel, s["ones"], observed, spec, warn=False, stop=stop
            )

        def hand(fit: Any) -> float:
            arms = {
                column: fit.targeted.arms[level] for level, column in submodel.arm_columns.items()
            }
            return engines.hand_state(
                s["y"],
                fit.targeted.observed,
                submodel.observed,
                s["ones"],
                observed,
                units=s["units"],
                span=SPAN,
                rule="sample_size",
                arm_curves=arms,
            )[1]

        final = at(500)
        assert final.converged and final.rule_ratio is not None and final.stopping == "sample_size"
        assert final.rule_ratio == pytest.approx(hand(final), rel=1e-9) and final.rule_ratio <= 1.0
        assert 2 < final.n_iter < 500
        # The walk stops at the first iterate that meets the rule: every earlier one misses it.
        for k in range(1, final.n_iter):
            earlier = at(k)
            assert not earlier.converged and hand(earlier) > 1.0
            assert earlier.rule_ratio == pytest.approx(hand(earlier), rel=1e-9)
        # The shipped ruler could not have stopped it: the relative score is far above tol.
        assert final.relative_score_norm > 1e3 * 1e-10

    @pytest.mark.parametrize("rule", ["relative", "sample_size"])
    def test_fold_targeted_fits_walk_each_fold(self, rule: str) -> None:
        rng = np.random.default_rng(3)
        n_units, per_unit = 48, 8
        n = n_units * per_unit
        w = rng.normal(size=n) + np.repeat(rng.normal(size=n_units), per_unit)
        a = rng.binomial(1, expit(0.4 * w)).astype(float)
        y = np.clip(0.4 + 0.1 * a + 0.1 * np.tanh(w) + 0.1 * rng.normal(size=n), 0.01, 0.99)
        frame = pd.DataFrame(
            {
                "Y": y,
                "A": a,
                "W1": w,
                "W2": rng.normal(size=n),
                "cluster": np.repeat(np.arange(n_units), per_unit),
            }
        )
        settings = {"stopping": rule} if rule != "relative" else {}
        estimator = TMLE(
            **linear_in_sample(
                cross_fit=True,
                n_folds=4,
                learner_folds=4,
                q_bounds=(0.0, 1.0),
                targeting_scheme="fold",
                step_bound=0.02,
                **settings,
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fluctuation = (
                estimator.fit(frame, outcome="Y", treatment="A", id="cluster")
                .single()
                .fluctuations["mean"]
            )
        assert len(fluctuation.folds) == 4
        if rule == "relative":
            assert all(record.n_iter > 0 for record in fluctuation.folds)
        else:
            # A fold whose initial fit already meets the rule takes no step.
            assert all(record.converged for record in fluctuation.folds)

    @pytest.mark.parametrize("strategy", ["greedy", "ordered", "discrete"])
    def test_ctmle_candidates_walk_the_bounded_step(self, strategy: str) -> None:
        frame, _ = make_instrument(n=300, seed=44)
        estimator = CTMLE(
            strategy=strategy,  # type: ignore[arg-type]
            **linear_in_sample(estimands=("ate",), step_bound=0.05),
            **SELECTOR_CONFIGS[strategy],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = estimator.fit(frame, outcome="Y", treatment="A").single()
        assert result.ctmle_selection is not None
        assert np.isfinite(result.estimates["ate"].psi)

    def test_the_one_step_group_refuses_the_bound(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            TMLE(**linear_in_sample(targeting="one_step", step_bound=0.05))
        assert str(raised.value) == R_C2

    def test_a_truncation_curve_a_retarget_and_a_bootstrap_run_under_the_bound(self) -> None:
        frame = engines._clustered()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = (
                TMLE(**linear_in_sample(step_bound=0.05, max_iter=40, n_bootstrap=4))
                .fit(frame, outcome="Y", treatment="A", id="cluster")
                .single()
            )
            curve = result.diagnostics.truncation_curve([0.01, 0.1])
            estimates, fluctuations = result.estimator.retarget(
                result.data, result.nuisance, estimands=("ate",)
            )
        assert np.all(np.isfinite(curve["psi"].to_numpy()))
        assert estimates["ate"].psi == result.estimates["ate"].psi
        assert fluctuations["mean"].n_iter == result.fluctuations["mean"].n_iter
        assert result.bootstrap is not None and result.bootstrap.n_failed == 0


# ------------------------------------------------------- the alternating exits, the plan's regime


def _bounded_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Swap the one-step walk of the G3b fixtures for the bounded step of the same size."""
    translated = {
        key: value for key, value in kwargs.items() if key not in {"targeting", "step_size"}
    }
    if kwargs.get("targeting") == "one_step":
        translated["step_bound"] = engines.STEP
    return translated


@pytest.fixture
def bounded_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engines, "TMLE", lambda **kw: TMLE(**_bounded_kwargs(kw)))
    monkeypatch.setattr(engines, "DRTMLE", lambda **kw: DRTMLE(**_bounded_kwargs(kw)))


@pytest.mark.usefixtures("bounded_engines")
class TestTheAlternatingExitsUnderTheBoundedWalk:
    """The plan's regime: ``step_bound=0.01`` with ``stopping="sample_size"``, so the shipped
    ruler cannot exit and the rule ratio read at each pair must."""

    @pytest.mark.parametrize(("order", "guard"), engines.ORDERS_AND_GUARDS)
    def test_drtmle_rounds_read_the_hand_ratio(
        self, monkeypatch: pytest.MonkeyPatch, order: str, guard: tuple[str, ...]
    ) -> None:
        assert engines.drtmle_problems(monkeypatch, order, guard) == []

    def test_the_missing_outcome_loop_reads_the_hand_ratio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert engines.missing_problems(monkeypatch) == []

    def test_point_ipsi_reads_the_solvers_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert engines.ipsi_problems(monkeypatch) == []

    def test_the_fixtures_walk_the_bounded_step_and_not_the_one_step_walk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lengths: list[float] = []
        original = iterative_module._bounded_step

        def spy(*args: Any) -> Any:
            step, length = original(*args)
            lengths.append(length)
            return step, length

        monkeypatch.setattr(iterative_module, "_bounded_step", spy)
        result = (
            engines._drtmle("drtmle", ("Q",)).fit(engines._complete(), outcome="Y", treatment="A")
        ).single()
        assert lengths and max(lengths) <= engines.STEP + 1e-12
        assert result.fluctuations["mean"].method == "iterative"
