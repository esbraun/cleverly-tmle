"""``Targeting(iterate="best")``: the ``tmle3`` ``use_best`` rule, its refusals and its plumbing.

``iterate="best"`` reports, of the iterates ``k >= 1`` of a targeting walk, the one whose raw score
vector has the smallest 2-norm, the first of any tie.  The initial fit is never a candidate.  The
walk still stops by its rule, ``Fluctuation.best_iterate`` records ``k*``, ``trace`` and
``n_iter`` describe the whole walk, and ``converged`` is judged at the reported iterate.  On the
one-step walk the norm never rises, so the best iterate is the last, and a tie takes the later.

The reference below is the ``reference_walk`` of ``test_step_bound.py``, written from the ``tmle3``
``fit_submodel`` description with ``numpy`` and ``scipy`` only.  A fixed-length step lands within
one bound of the root and oscillates around it, so the best iterate differs from the last.  The
mutation controls break one line each (report ``k* + 1``, rank by the relative statistic, offer
iterate 0, ``<=`` for ``<``, first of ties on the one-step walk) and ask the same comparison to
fail.
"""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import SDRMethod, Targeting, TMLEMethod
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.estimators.targeting import TargetingSpec, solve_submodel
from cleverly.exceptions import ConvergenceWarning, MethodConfigurationError, RefusalReason
from cleverly.fluctuation import InitialFit, mean_submodel, solve_fluctuation
from cleverly.fluctuation import _score as score_module
from cleverly.fluctuation import iterative as iterative_module
from cleverly.fluctuation import one_step as one_step_module
from cleverly.fluctuation._score import (
    ITERATES,
    LINEAR_RULE_REFUSAL,
    BestIterate,
    StopContext,
    check_iterate,
    iterate_norm,
    score_scale,
)
from cleverly.fluctuation.one_step import solve_one_step
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR
from cleverly.longitudinal.sequential import _fluctuate_node
from cleverly.methods import IPSI_NO_FLUCTUATION, IPSIMethod, sdr_no_fluctuation_refusal
from tests.conftest import linear_in_sample
from tests.unit import test_step_bound as bounded
from tests.unit import test_stopping_rule_engines as engines
from tests.unit import test_stopping_rule_longitudinal as longitudinal
from tests.unit._source_mutation import mutate

targeting_module = importlib.import_module("cleverly.estimators.targeting")
sequential_module = importlib.import_module("cleverly.longitudinal.sequential")
estimator_module = importlib.import_module("cleverly.longitudinal.estimator")
msm_module = importlib.import_module("cleverly.longitudinal.msm")

#: The plan's pinned refusal text (plan I-2.8, R-a1).
R_A1 = "iterate must be 'last' or 'best'; got {value!r}"

#: ``(seed, delta, steps)`` of fixed-length walks whose best iterate is not the last one.  Seed 8
#: at ``delta = 0.3`` is the one where the 2-norm and the relative statistic rank the iterates
#: differently (see ``TestTheRankingIsTheTwoNorm``).
WALKS = [(3, 0.05, 12), (3, 0.1, 12), (3, 0.3, 12), (8, 0.3, 12)]


# ------------------------------------------------------------------------------ the reference


def iterates(s: dict[str, Any], delta: float, steps: int) -> dict[str, Any]:
    """Score, ``epsilon`` and ``Q*`` of iterates ``0 .. steps`` of the fixed-length reference walk.

    ``reference_walk`` records the score *before* each step, so asking it for ``steps + 1`` records
    gives the score of every iterate up to ``steps`` in its ``score`` entries.
    """
    submodel = mean_submodel(s["a"], s["g1"])
    walk = bounded.reference_walk(
        s["y"], s["initial"].observed, submodel.observed, s["ones"], delta, False, steps + 1
    )
    scale = score_scale(submodel.observed, s["ones"], np.ones(s["n"], dtype=bool))
    scores = [record["score"] for record in walk[: steps + 1]]
    return {
        "score": scores,
        "norm": [float(np.linalg.norm(x)) for x in scores],
        "relative": [float(np.max(np.abs(x) / scale)) for x in scores],
        "epsilon": [np.zeros(2), *(record["epsilon"] for record in walk[:steps])],
        "q": [np.clip(s["initial"].observed, 1 - bounded.ALPHA, bounded.ALPHA)]
        + [record["p"] for record in walk[:steps]],
    }


def kstar(norms: list[float]) -> int:
    """The index of the smallest norm among iterates ``k >= 1``; ``argmin`` takes the first tie."""
    return 1 + int(np.argmin(norms[1:]))


def best_problems(s: dict[str, Any], delta: float, steps: int, **kwargs: Any) -> list[str]:
    """Compare the solver's best iterate with the reference replay of ``k*`` steps."""
    ref = iterates(s, delta, steps)
    k = kstar(ref["norm"])
    fit = bounded.walk_solver(
        s, steps, step_bound=delta, step_search=False, iterate="best", **kwargs
    )
    problems: list[str] = []
    if fit.best_iterate != k:
        problems.append(f"best_iterate {fit.best_iterate} != {k}")
    if not np.allclose(fit.epsilon, ref["epsilon"][k], atol=1e-12, rtol=0):
        problems.append("epsilon is not the replay of k* steps")
    if not np.allclose(fit.targeted.observed, ref["q"][k], atol=1e-12, rtol=0):
        problems.append("Q* is not the replay of k* steps")
    if not np.allclose(fit.score, ref["score"][k], atol=1e-12, rtol=0):
        problems.append("the score is not the score of iterate k*")
    if fit.n_iter != steps or len(fit.trace) != steps + 1:
        problems.append("n_iter and trace must describe the whole walk")
    elif fit.relative_score_norm != pytest.approx(fit.trace[k], rel=1e-9):
        problems.append("the reported relative score is not trace[k*]")
    return problems


# ---------------------------------------------------------------------- the exact replay


class TestTheBestIterateIsAReplayOfKStarSteps:
    @pytest.mark.parametrize(("seed", "delta", "steps"), WALKS)
    def test_the_reported_iterate_is_the_first_smallest_norm(
        self, seed: int, delta: float, steps: int
    ) -> None:
        assert best_problems(bounded.setting(seed), delta, steps) == []

    @pytest.mark.parametrize(("seed", "delta", "steps"), WALKS)
    def test_the_witnesses_the_exact_test_sits_above(
        self, seed: int, delta: float, steps: int
    ) -> None:
        s = bounded.setting(seed)
        ref = iterates(s, delta, steps)
        k = kstar(ref["norm"])
        # (1) k* is not the last iterate, and it is not the first when the walk is long.
        assert k < steps
        # (2) The estimates differ from the last iterate's by more than the floor.
        last = bounded.walk_solver(s, steps, step_bound=delta, step_search=False)
        best = bounded.walk_solver(s, steps, step_bound=delta, step_search=False, iterate="best")
        assert float(np.max(np.abs(best.targeted.observed - last.targeted.observed))) > 1e-6
        assert float(np.max(np.abs(best.epsilon - last.epsilon))) > 1e-6
        # (3) The reported norm is the smallest of the walk's accepted iterates k >= 1.
        assert float(np.linalg.norm(best.score)) == pytest.approx(min(ref["norm"][1:]), rel=1e-12)
        assert float(np.linalg.norm(best.score)) < float(np.linalg.norm(last.score))

    def test_the_last_iterate_is_reported_by_default_and_best_iterate_is_none(self) -> None:
        s = bounded.setting()
        fit = bounded.walk_solver(s, 12, step_bound=0.1, step_search=False)
        assert fit.best_iterate is None
        named = bounded.walk_solver(s, 12, step_bound=0.1, step_search=False, iterate="last")
        np.testing.assert_array_equal(fit.epsilon, named.epsilon)
        assert named.best_iterate is None

    def test_the_trace_is_complete_and_the_walk_still_runs_to_its_cap(self) -> None:
        s = bounded.setting()
        fit = bounded.walk_solver(s, 12, step_bound=0.1, step_search=False, iterate="best")
        last = bounded.walk_solver(s, 12, step_bound=0.1, step_search=False)
        assert fit.trace == last.trace and len(fit.trace) == 13 and fit.n_iter == 12
        assert fit.best_iterate == 2 and fit.trace[2] == min(fit.trace[1:])

    def test_the_initial_fit_is_never_the_reported_iterate(self) -> None:
        """The initial fit solves the score equation and every step overshoots it."""
        s = bounded.setting()
        newton = bounded.walk_solver(s, 20)
        start = {**s, "initial": newton.targeted}
        ref = iterates(start, 0.3, 6)
        assert ref["norm"][0] < 1e-12 < min(ref["norm"][1:])
        assert best_problems(start, 0.3, 6, tol=1e-18) == []
        fit = bounded.walk_solver(
            start, 6, step_bound=0.3, step_search=False, iterate="best", tol=1e-18
        )
        assert fit.best_iterate is not None and fit.best_iterate >= 1
        assert float(np.linalg.norm(fit.epsilon)) > 1e-6
        assert float(np.linalg.norm(fit.score)) > float(ref["norm"][0])

    def test_a_walk_that_takes_no_step_reports_the_initial_fit_as_iterate_zero(self) -> None:
        n = 40
        y = np.full(n, 0.4)
        initial = InitialFit(np.full(n, 0.4), {0.0: np.full(n, 0.4), 1.0: np.full(n, 0.4)})
        submodel = mean_submodel(np.tile([0.0, 1.0], n // 2), np.full(n, 0.5))
        fit = solve_fluctuation(
            y, initial, submodel, np.ones(n), None, step_bound=0.1, iterate="best"
        )
        assert fit.n_iter == 0 and fit.converged and fit.best_iterate == 0
        np.testing.assert_array_equal(fit.epsilon, np.zeros(2))

    def test_a_searched_walk_that_descends_reports_its_last_iterate(self) -> None:
        s = bounded.setting()
        last = bounded.walk_solver(s, 100, step_bound=bounded.DELTA)
        best = bounded.walk_solver(s, 100, step_bound=bounded.DELTA, iterate="best")
        assert best.best_iterate is not None
        # The norm falls to the tolerance, so a later iterate never has a larger one by much.
        assert float(np.linalg.norm(best.score)) <= float(np.linalg.norm(last.score)) * (1 + 1e-9)
        assert best.n_iter == last.n_iter and best.trace == last.trace


# --------------------------------------------------------------------- converged at the reported


class TestConvergedIsJudgedAtTheReportedIterate:
    """The rule stops the walk at iterate ``B``; the 2-norm picks an earlier ``A``."""

    SEED, DELTA, A, B = 8, 0.3, 1, 3

    def fixture(self) -> tuple[dict[str, Any], float, dict[str, Any]]:
        s = bounded.setting(self.SEED)
        ref = iterates(s, self.DELTA, 8)
        rel = ref["relative"]
        assert kstar(ref["norm"][: self.B + 1]) == self.A
        assert rel[self.B] < min(rel[1 : self.B]) < rel[self.A] + 1e-12
        tol = 0.5 * (rel[self.B] + min(rel[1 : self.B]))
        return s, tol, ref

    def test_the_last_iterate_meets_the_rule_and_the_best_does_not(self) -> None:
        s, tol, ref = self.fixture()
        last = bounded.walk_solver(s, 8, step_bound=self.DELTA, step_search=False, tol=tol)
        assert last.converged and last.n_iter == self.B and last.failure is None
        with pytest.warns(ConvergenceWarning, match=r"relative score"):
            best = solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                max_iter=8,
                tol=tol,
                step_bound=self.DELTA,
                step_search=False,
                iterate="best",
            )
        # The walk stopped at the same iterate; the report is iterate A, where the rule is missed.
        assert best.n_iter == self.B and len(best.trace) == self.B + 1
        assert best.best_iterate == self.A
        assert not best.converged and best.failure is not None
        assert best.relative_score_norm == pytest.approx(ref["relative"][self.A], rel=1e-9)
        assert best.relative_score_norm > tol > last.relative_score_norm
        np.testing.assert_allclose(best.epsilon, ref["epsilon"][self.A], atol=1e-12)

    def test_a_declared_rule_judges_the_reported_iterate_by_its_own_ratio(self) -> None:
        s = bounded.setting(3)
        submodel = mean_submodel(s["a"], s["g1"])

        def walk(span: float, **kwargs: Any) -> Any:
            return solve_fluctuation(
                s["y"],
                s["initial"],
                submodel,
                s["ones"],
                None,
                max_iter=kwargs.pop("max_iter", 12),
                warn=False,
                step_bound=0.05,
                step_search=False,
                stopping="sample_size",
                stop=StopContext(s["units"], span),
                **kwargs,
            )

        # A rule that is never met in 12 steps: the ratio belongs to the reported iterate.
        best = walk(bounded.SPAN, iterate="best")
        assert best.best_iterate == 4 and best.n_iter == 12 and not best.converged
        assert best.rule_ratio is not None and best.rule_ratio > 1.0
        prefix = walk(bounded.SPAN, max_iter=4)
        assert best.rule_ratio == prefix.rule_ratio
        np.testing.assert_array_equal(best.thresholds, prefix.thresholds)
        np.testing.assert_array_equal(best.epsilon, prefix.epsilon)
        arms = {column: best.targeted.arms[level] for level, column in submodel.arm_columns.items()}
        _, hand = engines.hand_state(
            s["y"],
            best.targeted.observed,
            submodel.observed,
            s["ones"],
            np.ones(s["n"], dtype=bool),
            units=s["units"],
            span=bounded.SPAN,
            rule="sample_size",
            arm_curves=arms,
        )
        assert best.rule_ratio == pytest.approx(hand, rel=1e-9)
        # The last iterate has a larger ratio than the reported one, so the choice is visible.
        last = walk(bounded.SPAN)
        assert last.rule_ratio is not None and last.rule_ratio > best.rule_ratio
        # A rule the initial fit already meets takes no step, and the report is iterate 0.
        met = walk(bounded.SPAN * 1e-3, iterate="best")
        assert met.converged and met.best_iterate == met.n_iter == 0


# ---------------------------------------------------------------------------- the newton walk


class TestTheNewtonWalk:
    def test_a_walk_whose_norm_falls_reports_the_last_iterate_bit_for_bit(self) -> None:
        s = bounded.setting()
        last = bounded.walk_solver(s, 20)
        best = bounded.walk_solver(s, 20, iterate="best")
        assert best.best_iterate == best.n_iter == last.n_iter
        np.testing.assert_array_equal(best.epsilon, last.epsilon)
        np.testing.assert_array_equal(best.targeted.observed, last.targeted.observed)
        np.testing.assert_array_equal(best.score, last.score)
        assert best.trace == last.trace and best.converged == last.converged

    def test_a_node_with_a_shrink_that_binds_walks_three_newton_iterates(self) -> None:
        node = longitudinal.crafted_node()
        last = _fluctuate_node(**node)
        best = _fluctuate_node(**node, iterate="best")
        assert last.n_iter == 3 and best.n_iter == 3 and best.best_iterate == 3
        np.testing.assert_array_equal(best.epsilon, last.epsilon)
        assert last.best_iterate is None

    def test_carried_fits_and_the_newton_detail_follow_the_reported_iterate(self) -> None:
        s = bounded.setting()
        submodel = mean_submodel(s["a"], s["g1"])
        carry = (
            InitialFit(
                np.full(s["n"], 0.3), {0.0: np.full(s["n"], 0.3), 1.0: np.full(s["n"], 0.3)}
            ),
        )
        kwargs: dict[str, Any] = {
            "max_iter": 9,
            "warn": False,
            "step_bound": 0.1,
            "step_search": False,
        }
        best = solve_fluctuation(
            s["y"], s["initial"], submodel, s["ones"], None, carry=carry, iterate="best", **kwargs
        )
        assert best.best_iterate is not None and best.best_iterate < best.n_iter
        prefix = solve_fluctuation(
            s["y"],
            s["initial"],
            submodel,
            s["ones"],
            None,
            carry=carry,
            **{**kwargs, "max_iter": best.best_iterate},
        )
        np.testing.assert_array_equal(best.epsilon, prefix.epsilon)
        np.testing.assert_array_equal(best.carried[0].observed, prefix.carried[0].observed)
        assert best.loglik == prefix.loglik
        assert best.hessian_condition == prefix.hessian_condition

    def test_the_linear_fluctuation_has_no_iterates(self) -> None:
        s = bounded.setting()
        with pytest.raises(MethodConfigurationError):
            solve_fluctuation(
                s["y"],
                s["initial"],
                mean_submodel(s["a"], s["g1"]),
                s["ones"],
                None,
                kind="linear",
                iterate="best",
            )


# -------------------------------------------------------------------------------- one step


def scripted_scores(monkeypatch: pytest.MonkeyPatch, module: Any, vectors: list[Any]) -> None:
    """Make ``module.score_columns`` return ``vectors`` in call order, then the last one forever."""
    calls = iter(vectors)
    state = {"last": np.asarray(vectors[-1], dtype=float)}

    def fake(*args: Any, **kwargs: Any) -> np.ndarray:
        state["last"] = np.asarray(next(calls, state["last"]), dtype=float)
        return state["last"]

    monkeypatch.setattr(module, "score_columns", fake)


def one_step_fit(*, iterate: str, steps: int = 2, **kwargs: Any) -> Any:
    s = bounded.setting()
    return solve_one_step(
        s["y"],
        s["initial"],
        mean_submodel(s["a"], s["g1"]),
        s["ones"],
        None,
        max_steps=steps,
        step_size=1e-3,
        warn=False,
        iterate=iterate,
        **kwargs,
    )


#: Two accepted steps of equal norm: ``||(3, 4)|| = ||(4, 3)|| = 5``.  The first call is the
#: initial score, then one candidate per step.
EQUAL_NORM_STEPS = [(3.0, 4.0), (4.0, 3.0), (3.0, 4.0)]


class TestTheOneStepWalk:
    def test_a_real_walk_reports_its_last_iterate_and_the_fit_is_the_default_fit(self) -> None:
        s = bounded.setting()
        kwargs: dict[str, Any] = {"max_steps": 400, "step_size": 5e-3, "tol": 1e-3, "warn": False}
        args = (s["y"], s["initial"], mean_submodel(s["a"], s["g1"]), s["ones"], None)
        last = solve_one_step(*args, **kwargs)
        best = solve_one_step(*args, iterate="best", **kwargs)
        assert best.best_iterate == best.n_iter == last.n_iter > 10
        np.testing.assert_array_equal(best.epsilon, last.epsilon)
        np.testing.assert_array_equal(best.targeted.observed, last.targeted.observed)
        assert best.trace == last.trace and best.converged == last.converged
        assert last.best_iterate is None and best.method == "one_step"

    def test_a_walk_that_takes_no_step_reports_iterate_zero(self) -> None:
        s = bounded.setting()
        fit = solve_one_step(
            s["y"],
            s["initial"],
            mean_submodel(s["a"], s["g1"]),
            s["ones"],
            None,
            tol=10.0,
            iterate="best",
        )
        assert fit.n_iter == 0 and fit.best_iterate == 0 and fit.converged

    def test_a_step_of_equal_norm_takes_the_later_iterate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripted_scores(monkeypatch, one_step_module, EQUAL_NORM_STEPS)
        fit = one_step_fit(iterate="best")
        assert fit.n_iter == 2
        # Iterates 1 and 2 both have norm 5, and the later one is reported.
        assert fit.best_iterate == 2
        np.testing.assert_allclose(fit.score, [3.0, 4.0])

    def test_the_default_walk_reports_the_same_iterate_without_a_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripted_scores(monkeypatch, one_step_module, EQUAL_NORM_STEPS)
        fit = one_step_fit(iterate="last")
        assert fit.n_iter == 2 and fit.best_iterate is None
        np.testing.assert_allclose(fit.score, [3.0, 4.0])

    def test_a_declared_rule_composes_with_the_best_iterate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = bounded.setting()
        stop = StopContext(s["units"], bounded.SPAN)
        last = one_step_fit(iterate="last", steps=30, stopping="sample_size", stop=stop)
        best = one_step_fit(iterate="best", steps=30, stopping="sample_size", stop=stop)
        assert best.best_iterate == best.n_iter == last.n_iter
        assert best.rule_ratio == last.rule_ratio and best.stopping == "sample_size"
        np.testing.assert_array_equal(best.thresholds, last.thresholds)
        np.testing.assert_array_equal(best.epsilon, last.epsilon)

    def test_the_tied_norm_fixture_is_an_equal_norm_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scripted vectors have exactly equal norms, so ``<`` and ``<=`` differ on them."""
        norms = [iterate_norm(np.array(v)) for v in EQUAL_NORM_STEPS]
        assert norms[0] == norms[1] == norms[2] == 5.0


class TestTheSnapshot:
    def test_a_strictly_smaller_norm_replaces_the_snapshot_and_a_tie_keeps_the_first(self) -> None:
        best = BestIterate()
        for k, norm in enumerate([4.0, 2.0, 2.0, 3.0, 2.0], start=1):
            best.offer(k, norm, (k,))
        assert best.k == 2 and best.state == (2,) and best.norm == 2.0

    def test_the_one_step_snapshot_lets_a_tie_take_the_later_iterate(self) -> None:
        best = BestIterate(last_on_ties=True)
        for k, norm in enumerate([4.0, 2.0, 2.0, 3.0, 2.0], start=1):
            best.offer(k, norm, (k,))
        assert best.k == 5

    def test_an_empty_snapshot_has_no_state(self) -> None:
        best = BestIterate()
        assert best.k is None and best.state is None
        best.offer(1, float("nan"), (1,))
        assert best.k is None

    def test_the_ranking_statistic_is_the_two_norm_of_the_raw_score(self) -> None:
        assert iterate_norm(np.array([3.0, -4.0])) == 5.0
        assert iterate_norm(np.zeros(3)) == 0.0


# ------------------------------------------------------------------ the ranking is the 2-norm


class TestTheRankingIsTheTwoNorm:
    """Seed 8 at ``delta = 0.3``: iterate 1 has the smaller norm, iterate 3 the smaller relative."""

    def test_the_two_statistics_rank_the_iterates_differently(self) -> None:
        ref = iterates(bounded.setting(8), 0.3, 12)
        by_norm = kstar(ref["norm"])
        by_relative = 1 + int(np.argmin(ref["relative"][1:]))
        assert by_norm == 1 and by_relative != by_norm

    def test_the_solver_reports_the_two_norm_winner(self) -> None:
        s = bounded.setting(8)
        fit = bounded.walk_solver(s, 12, step_bound=0.3, step_search=False, iterate="best")
        assert fit.best_iterate == 1
        assert fit.trace[1] > min(fit.trace[2:])


# ------------------------------------------------------------------------- mutation controls


class TestMutationControls:
    """Each control passes the exact comparison first, then fails it with one line broken."""

    @pytest.fixture
    def clean(self) -> None:
        for seed, delta, steps in WALKS:
            assert best_problems(bounded.setting(seed), delta, steps) == []
        start = {
            **bounded.setting(),
            "initial": bounded.walk_solver(bounded.setting(), 20).targeted,
        }
        assert best_problems(start, 0.3, 6, tol=1e-18) == []

    @staticmethod
    def broken() -> list[str]:
        problems: list[str] = []
        for seed, delta, steps in WALKS:
            problems += best_problems(bounded.setting(seed), delta, steps)
        start = {
            **bounded.setting(),
            "initial": bounded.walk_solver(bounded.setting(), 20).targeted,
        }
        problems += best_problems(start, 0.3, 6, tol=1e-18)
        return problems

    def test_reporting_the_iterate_after_the_best_fails(
        self, clean: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class After(BestIterate):
            """Snapshots the state one offer after the winning norm: iterate ``k* + 1``."""

            def __init__(self, last_on_ties: bool = False) -> None:
                super().__init__(last_on_ties)
                self.take_next = False

            def offer(self, k: int, norm: float, state: tuple[Any, ...]) -> None:
                if self.take_next:
                    self.k, self.state, self.take_next = k, state, False
                    return
                before = self.norm
                super().offer(k, norm, state)
                self.take_next = self.norm < before

        monkeypatch.setattr(iterative_module, "BestIterate", After)
        assert self.broken()

    def test_ranking_by_the_relative_statistic_fails(
        self, clean: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            iterative_module,
            iterative_module,
            "_solve_fluctuation_bounded",
            [("iterate_norm(score),", "trace[-1],")],
        )
        assert self.broken()

    def test_offering_the_initial_fit_fails(
        self, clean: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            iterative_module,
            iterative_module,
            "_solve_fluctuation_bounded",
            [
                (
                    'best = BestIterate() if iterate == "best" else None\n',
                    'best = BestIterate() if iterate == "best" else None\n'
                    "    if best is not None:\n"
                    "        best.offer(\n"
                    "            0,\n"
                    "            iterate_norm(score_before),\n"
                    "            (epsilon, current, carried, score_before, thresholds, ratio),\n"
                    "        )\n",
                )
            ],
        )
        assert any("best_iterate" in problem for problem in self.broken())

    def test_less_or_equal_in_place_of_less_fails(
        self, clean: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            score_module,
            BestIterate,
            "offer",
            [
                (
                    "better = norm <= self.norm if self.last_on_ties else norm < self.norm",
                    "better = norm <= self.norm",
                )
            ],
        )
        # No tie occurs on the real walks, so the scripted equal-norm walk carries this control.
        scripted = pytest.MonkeyPatch()
        try:
            scripted_scores(scripted, iterative_module, [(3.0, 4.0), (4.0, 3.0), (3.0, 4.0)])
            s = bounded.setting()
            fit = bounded.walk_solver(
                s, 2, step_bound=0.1, step_search=False, iterate="best", tol=1e-30
            )
        finally:
            scripted.undo()
        assert fit.best_iterate == 2

    def test_first_of_ties_on_the_one_step_walk_fails(
        self, clean: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            one_step_module,
            one_step_module,
            "_solve_one_step_ruled",
            [("BestIterate(last_on_ties=True)", "BestIterate(last_on_ties=False)")],
        )
        scripted_scores(monkeypatch, one_step_module, EQUAL_NORM_STEPS)
        assert one_step_fit(iterate="best").best_iterate == 1  # the control reports iterate 1

    def test_the_clean_scripted_walk_reports_the_first_of_tied_newton_style_iterates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripted_scores(monkeypatch, iterative_module, [(3.0, 4.0), (4.0, 3.0), (3.0, 4.0)])
        s = bounded.setting()
        fit = bounded.walk_solver(
            s, 2, step_bound=0.1, step_search=False, iterate="best", tol=1e-30
        )
        assert fit.best_iterate == 1
        np.testing.assert_allclose(fit.score, [4.0, 3.0])


# --------------------------------------------------------------------------- the plumbing


class TestPlumbing:
    def test_every_surface_defaults_to_the_last_iterate(self) -> None:
        assert Targeting().iterate == "last" and TargetingSpec().iterate == "last"
        estimator = TMLE(**linear_in_sample())
        assert estimator.iterate == "last" and estimator.targeting_spec().iterate == "last"
        assert LTMLE(longitudinal.PLANS).iterate == "last"
        assert ITERATES == ("last", "best")

    def test_the_shortcuts_route_to_the_targeting_group(self) -> None:
        method = TMLEMethod().with_overrides(iterate="best")
        assert method.targeting.iterate == "best"
        assert method.estimator_kwargs()["iterate"] == "best"
        engine = method.estimator_kwargs(longitudinal=True)
        assert engine["iterate"] == "best"
        assert TMLEMethod().estimator_kwargs()["iterate"] == "last"
        assert LTMLE(longitudinal.PLANS, **engine).iterate == "best"

    def test_the_spec_reaches_the_solver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = bounded.setting()
        submodel = mean_submodel(s["a"], s["g1"])
        observed = np.ones(s["n"], dtype=bool)
        spec = TargetingSpec(step_bound=0.1, step_search=False, max_iter=12, iterate="best")

        def solve() -> Any:
            return solve_submodel(
                s["y"], s["initial"], submodel, s["ones"], observed, spec, warn=False
            )

        assert solve().best_iterate == 2
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_solve_submodel_ruled",
            [("iterate=spec.iterate,\n    )\n", "iterate='last',\n    )\n")],
        )
        assert solve().best_iterate is None

    def test_the_one_step_spec_reaches_the_solver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = bounded.setting()
        submodel = mean_submodel(s["a"], s["g1"])
        spec = TargetingSpec(
            targeting="one_step", max_iter=20, step_size=5e-3, tol=1e-2, iterate="best"
        )
        fit = solve_submodel(
            s["y"], s["initial"], submodel, s["ones"], np.ones(s["n"], dtype=bool), spec, warn=False
        )
        assert fit.method == "one_step" and fit.best_iterate == fit.n_iter

    def test_a_default_fit_equals_a_fit_that_names_the_default(self) -> None:
        frame = make_linear_ate(n=300, seed=2)[0]
        plain = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        named = (
            TMLE(**linear_in_sample(iterate="last")).fit(frame, outcome="Y", treatment="A").single()
        )
        for key, estimate in plain.estimates.items():
            assert estimate.psi == named.estimates[key].psi
            np.testing.assert_array_equal(
                estimate.influence_curve, named.estimates[key].influence_curve
            )
        for group, fluctuation in plain.fluctuations.items():
            assert fluctuation.best_iterate is None
            np.testing.assert_array_equal(fluctuation.epsilon, named.fluctuations[group].epsilon)

    def test_a_best_fit_with_a_converging_walk_equals_the_default_fit(self) -> None:
        frame = make_linear_ate(n=300, seed=2)[0]
        plain = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        best = (
            TMLE(**linear_in_sample(iterate="best")).fit(frame, outcome="Y", treatment="A").single()
        )
        for key, estimate in plain.estimates.items():
            assert estimate.psi == best.estimates[key].psi
        for group, fluctuation in best.fluctuations.items():
            assert fluctuation.best_iterate == fluctuation.n_iter >= 1
            np.testing.assert_array_equal(fluctuation.epsilon, plain.fluctuations[group].epsilon)

    def test_a_ltmle_default_fit_equals_a_fit_that_names_the_default(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
        plain = bounded.fitted(LTMLE(longitudinal.PLANS, **longitudinal.learners()), frame)
        named = bounded.fitted(
            LTMLE(longitudinal.PLANS, iterate="last", **longitudinal.learners()), frame
        )
        for key, estimate in plain.estimates.items():
            assert estimate.psi == named.estimates[key].psi
        for _, step in longitudinal.every_node(plain):
            assert step.fluctuation.best_iterate is None

    def test_the_replay_recipe_and_a_truncation_replay_repeat_the_iterate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
        result = bounded.fitted(bounded.ltmle(step_search=False, iterate="best"), frame)
        assert result.replay_recipe.iterate == "best"
        seen: list[str] = []
        original = estimator_module._run_recursion

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["iterate"])
            return original(*args, **kwargs)

        monkeypatch.setattr(estimator_module, "_run_recursion", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = result.diagnostics.truncation_curve([result.config.g_bounds, (0.05, 1.0)])
        assert len(curve) >= 2 and seen and set(seen) == {"best"}

    def test_a_point_result_round_trips_the_iterate_and_the_best_index(self) -> None:
        frame = make_linear_ate(n=200, seed=2)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = (
                TMLE(**linear_in_sample(step_bound=0.05, step_search=False, iterate="best"))
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
        restored = loads(dumps(result))
        assert restored.config.targeting_spec.iterate == "best"
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.best_iterate is not None and fluctuation.best_iterate >= 1
        assert restored.fluctuations["mean"].best_iterate == fluctuation.best_iterate
        np.testing.assert_array_equal(restored.fluctuations["mean"].epsilon, fluctuation.epsilon)

    def test_a_longitudinal_result_round_trips_the_best_index(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
        result = bounded.fitted(bounded.ltmle(step_search=False, iterate="best"), frame)
        restored = loads(dumps(result))
        before = [step.fluctuation.best_iterate for _, step in longitudinal.every_node(result)]
        after = [step.fluctuation.best_iterate for _, step in longitudinal.every_node(restored)]
        assert before == after and any(k is not None for k in before)
        assert restored.replay_recipe.iterate == "best"


# ----------------------------------------------------------------------------- the refusals


class TestRefusals:
    @pytest.mark.parametrize("value", ["first", "Best", None, 1, True, ["best"]])
    def test_r_a1_an_iterate_that_does_not_exist(self, value: Any) -> None:
        expected = re.escape(R_A1.format(value=value))
        for build in (
            lambda: Targeting(iterate=value),
            lambda: TargetingSpec(iterate=value),
            lambda: TMLE(**linear_in_sample(iterate=value)),
            lambda: LTMLE(longitudinal.PLANS, iterate=value),
            lambda: check_iterate(value, "logistic"),
        ):
            with pytest.raises(ValueError, match=f"^{expected}$"):
                build()

    def test_r_abc1_the_best_iterate_beside_the_linear_fluctuation(self) -> None:
        for build in (
            lambda: Targeting(fluctuation="linear", iterate="best"),
            lambda: TargetingSpec(fluctuation="linear", iterate="best"),
            lambda: TMLE(**linear_in_sample(fluctuation="linear", iterate="best")),
        ):
            with pytest.raises(MethodConfigurationError) as raised:
                build()
            assert str(raised.value) == LINEAR_RULE_REFUSAL.message()
            assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_the_default_iterate_is_allowed_beside_the_linear_fluctuation(self) -> None:
        assert Targeting(fluctuation="linear", iterate="last").iterate == "last"

    def test_r_abc2_sdr_refuses_a_best_iterate_by_name(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            SDR(longitudinal.PLANS, iterate="best")
        assert str(raised.value) == sdr_no_fluctuation_refusal("iterate").message()
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        method = SDRMethod(targeting=Targeting(iterate="best"))
        with pytest.raises(MethodConfigurationError) as via_method:
            method.estimator_kwargs(longitudinal=True)
        assert str(via_method.value) == sdr_no_fluctuation_refusal("iterate").message()
        assert "iterate" in estimator_module.SDR_REFUSED_SETTINGS
        assert "iterate" not in SDRMethod().estimator_kwargs(longitudinal=True)

    def test_r_abc3_ipsi_refuses_a_best_iterate(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            IPSI([Incremental(2.0)], iterate="best")
        assert str(raised.value) == IPSI_NO_FLUCTUATION.message()
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        method = IPSIMethod(targeting=Targeting(iterate="best"))
        with pytest.raises(MethodConfigurationError) as via_method:
            method.estimator_kwargs(longitudinal=True)
        assert str(via_method.value) == IPSI_NO_FLUCTUATION.message()
        assert "iterate" in estimator_module.IPSI_REFUSED_SETTINGS
        assert "iterate" not in IPSIMethod().estimator_kwargs(longitudinal=True)


# --------------------------------------------------------------------- every point group


def prefix_problems(
    calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]], original: Any, cap: str = "max_iter"
) -> list[str]:
    """Each best solve equals the default solve cut at ``k*`` iterations (the walk is a prefix)."""
    problems: list[str] = []
    for args, kwargs, fit in calls:
        if kwargs.get("iterate") != "best":
            continue
        k = fit.best_iterate
        if k is None or not 0 <= k <= fit.n_iter:
            problems.append(f"best_iterate {k} outside 0..{fit.n_iter}")
            continue
        if k == 0:
            continue
        prefix = original(*args, **{**kwargs, "iterate": "last", cap: k, "warn": False})
        if not np.array_equal(prefix.epsilon, fit.epsilon):
            problems.append(f"epsilon is not the replay of {k} iterations")
        if not np.array_equal(prefix.targeted.observed, fit.targeted.observed):
            problems.append(f"Q* is not the replay of {k} iterations")
        if not np.array_equal(prefix.score, fit.score):
            problems.append(f"the score is not the score of iteration {k}")
    return problems


class TestEveryPointGroupFitsUnderTheBestIterate:
    @pytest.mark.parametrize("kind", sorted(engines.KINDS))
    def test_the_fit_runs_and_each_solve_reports_the_replay_of_its_best_iterate(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, columns, settings, _ = engines.KINDS[kind]()
        calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
        original = {
            "solve_fluctuation": targeting_module.solve_fluctuation,
            "solve_one_step": targeting_module.solve_one_step,
        }

        def spy(name: str) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                fit = original[name](*args, **kwargs)
                calls.append((args, kwargs, fit))
                return fit

            return wrapped

        monkeypatch.setattr(targeting_module, "solve_fluctuation", spy("solve_fluctuation"))
        monkeypatch.setattr(targeting_module, "solve_one_step", spy("solve_one_step"))
        bound = (
            {}
            if settings.get("targeting") == "one_step"
            else {"step_bound": 0.05, "step_search": False}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator = TMLE(**linear_in_sample(iterate="best", **bound, **settings))
            result = estimator.fit(frame, outcome="Y", treatment="A", **columns).single()
        assert calls, kind
        assert all(np.isfinite(e.psi) for e in result.estimates.values())
        for _, kwargs, fit in calls:
            assert kwargs["iterate"] == "best" and fit.best_iterate is not None, kind
        solver = (
            "solve_one_step" if settings.get("targeting") == "one_step" else "solve_fluctuation"
        )
        cap = "max_steps" if solver == "solve_one_step" else "max_iter"
        assert prefix_problems(calls, original[solver], cap) == [], kind

    def test_a_bounded_fixed_walk_reports_an_iterate_before_the_last_somewhere(self) -> None:
        frame, columns, _, _ = engines.KINDS["arm-means"]()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = (
                TMLE(
                    **linear_in_sample(
                        iterate="best", step_bound=0.05, step_search=False, max_iter=15
                    )
                )
                .fit(frame, outcome="Y", treatment="A", **columns)
                .single()
            )
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.best_iterate is not None
        assert 1 <= fluctuation.best_iterate < fluctuation.n_iter == 15
        assert fluctuation.trace[fluctuation.best_iterate] == min(fluctuation.trace[1:])

    def test_fold_targeted_fits_record_the_best_iterate_of_each_fold(self) -> None:
        rng = np.random.default_rng(3)
        n_units, per_unit = 48, 8
        n = n_units * per_unit
        w = rng.normal(size=n) + np.repeat(rng.normal(size=n_units), per_unit)
        a = rng.binomial(1, bounded.expit(0.4 * w)).astype(float)
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
        estimator = TMLE(
            **linear_in_sample(
                cross_fit=True,
                n_folds=3,
                learner_folds=3,
                q_bounds=(0.0, 1.0),
                targeting_scheme="fold",
                step_bound=0.05,
                step_search=False,
                max_iter=10,
                iterate="best",
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fluctuation = (
                estimator.fit(frame, outcome="Y", treatment="A", id="cluster")
                .single()
                .fluctuations["mean"]
            )
        assert len(fluctuation.folds) == 3
        assert all(record.best_iterate is not None for record in fluctuation.folds)
        assert fluctuation.best_iterate is None  # the pooled record aggregates independent solves


@pytest.fixture
def best_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """The alternating-exit fixtures of ``test_step_bound.py`` with the best iterate switched on."""

    def translate(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**bounded._bounded_kwargs(kwargs), "iterate": "best", "step_search": False}

    monkeypatch.setattr(engines, "TMLE", lambda **kw: TMLE(**translate(kw)))
    monkeypatch.setattr(engines, "DRTMLE", lambda **kw: DRTMLE(**translate(kw)))


#: A fixed-length walk cannot meet the rule, so the Q-guard fit exits at its stall rule.  The helper
#: that audits the rounds states that exit as a problem; the audit of every round's ratio and of
#: the replayed stall rule still runs and must pass.
STALL_EXIT = "the Q-guard fit exited at 'stall'"


@pytest.mark.usefixtures("best_engines")
class TestTheAlternatingLoopsApplyTheBestIteratePerInnerSolve:
    """Every round's exit reads the ratio of the reported iterate, at the pair it judges."""

    @pytest.mark.parametrize(("order", "guard"), engines.ORDERS_AND_GUARDS)
    def test_drtmle_rounds_read_the_hand_ratio(
        self, monkeypatch: pytest.MonkeyPatch, order: str, guard: tuple[str, ...]
    ) -> None:
        problems = engines.drtmle_problems(monkeypatch, order, guard)
        assert [p for p in problems if not p.startswith(STALL_EXIT)] == []

    def test_the_missing_outcome_loop_reads_the_hand_ratio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert engines.missing_problems(monkeypatch) == []

    def test_point_ipsi_reads_the_solvers_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert engines.ipsi_problems(monkeypatch) == []

    @staticmethod
    def inner_fits(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        fits: list[Any] = []
        original = targeting_module.solve_submodel

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            fit = original(*args, **kwargs)
            fits.append(fit)
            return fit

        monkeypatch.setattr(targeting_module, "solve_submodel", wrapped)
        return fits

    def test_every_inner_solve_of_a_drtmle_round_reports_a_best_iterate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fits = self.inner_fits(monkeypatch)
        engines._drtmle("drtmle", ("Q",)).fit(
            engines._complete(), outcome="Y", treatment="A"
        ).single()
        assert len(fits) > 3
        assert all(fit.best_iterate is not None for fit in fits)
        # The oscillating inner walk reports an iterate before its last in some solve.
        assert any(fit.best_iterate < fit.n_iter for fit in fits)

    def test_every_inner_solve_of_a_point_ipsi_round_reports_a_best_iterate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fits = self.inner_fits(monkeypatch)
        estimator = TMLE(
            **linear_in_sample(
                stopping="sample_size",
                incremental=[Incremental(2.0)],
                step_bound=engines.STEP,
                step_search=False,
                iterate="best",
            )
        )
        estimator.fit(engines._ipsi_frame(), outcome="Y", treatment="A").single()
        assert len(fits) >= 2
        assert all(fit.best_iterate is not None for fit in fits)


# ---------------------------------------------------------------------------- longitudinal


def assert_node_best(result: Any, delta: float, search: bool) -> int:
    """Each node reports iterate ``k*`` of the reference walk from its own state.

    Returns how many nodes reported an iterate before their last.
    """
    nodes = longitudinal.every_node(result)
    assert nodes
    earlier = 0
    for key, step in nodes:
        fluctuation = step.fluctuation
        if not fluctuation.n_iter:
            continue
        trained = np.asarray(step.trained_on, dtype=bool)
        y = np.asarray(step.pseudo_outcome, dtype=float)
        w = np.asarray(result.data.weights, dtype=float) * np.asarray(step.clever, dtype=float)
        walk = bounded.reference_walk(
            y,
            np.asarray(step.initial, dtype=float),
            np.ones((y.shape[0], 1)),
            w,
            delta,
            search,
            fluctuation.n_iter + 1,
            mask=trained,
        )
        norms = [
            float(np.linalg.norm(record["score"])) for record in walk[: fluctuation.n_iter + 1]
        ]
        k = kstar(norms)
        assert fluctuation.best_iterate == k, key
        np.testing.assert_allclose(
            fluctuation.epsilon, walk[k - 1]["epsilon"], atol=2e-7, err_msg=key
        )
        earlier += int(k < fluctuation.n_iter)
    return earlier


class TestTheLongitudinalNode:
    def test_the_node_reports_the_best_iterate_of_its_reference_walk(self) -> None:
        base = longitudinal.crafted_node()
        y, p0, w = base["pseudo_outcome"], base["initial"], base["loss_weights"]
        x = np.ones((y.shape[0], 1))
        fit = _fluctuate_node(
            **{**base, "max_iter": 12}, step_bound=0.4, step_search=False, iterate="best"
        )
        walk = bounded.reference_walk(y, p0, x, w, 0.4, False, 13)
        norms = [float(np.linalg.norm(record["score"])) for record in walk]
        k = kstar(norms)
        assert fit.best_iterate == k
        np.testing.assert_allclose(fit.epsilon, walk[k - 1]["epsilon"], atol=1e-12)
        assert k < 12 and fit.n_iter == 12

    def test_a_constant_node_takes_no_step_and_records_iterate_zero(self) -> None:
        base = longitudinal.crafted_node()
        n = base["pseudo_outcome"].shape[0]
        base["pseudo_outcome"] = np.zeros(n)
        base["initial"] = np.zeros(n)
        fit = _fluctuate_node(**base, iterate="best")
        assert fit.converged and fit.n_iter == 0 and fit.best_iterate == 0
        assert _fluctuate_node(**base).best_iterate is None

    def test_the_default_node_is_untouched_by_the_last_iterate(self) -> None:
        base = longitudinal.crafted_node()
        plain = _fluctuate_node(**base)
        named = _fluctuate_node(**base, iterate="last")
        np.testing.assert_array_equal(plain.epsilon, named.epsilon)
        assert plain.n_iter == named.n_iter and named.best_iterate is None


class TestEveryPerRegimenEngine:
    @pytest.mark.parametrize("n_folds", [1, 3])
    def test_every_node_of_an_end_of_study_fit_reports_its_best_iterate(self, n_folds: int) -> None:
        frame, _ = make_longitudinal(n=300, seed=12)
        estimator = bounded.ltmle(n_folds=n_folds, step_search=False, iterate="best")
        result = bounded.fitted(estimator, frame)
        assert assert_node_best(result, bounded.BOUND, False) >= 1

    def test_a_survival_fit(self) -> None:
        frame, _ = longitudinal.make_longitudinal_survival(n=300, seed=1)
        estimator = bounded.ltmle(step_search=False, iterate="best")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = estimator.fit(frame, **longitudinal.SURVIVAL)
        assert_node_best(result, bounded.BOUND, False)

    def test_a_competing_risks_fit(self) -> None:
        frame, _ = longitudinal.make_longitudinal_competing(n=300, seed=1)
        estimator = bounded.ltmle(n_folds=3, step_search=False, iterate="best")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = estimator.fit(frame, **longitudinal.COMPETING)
        assert_node_best(result, bounded.BOUND, False)

    def test_the_best_iterate_composes_with_a_declared_rule_and_clusters(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12, cluster_size=8)
        estimator = bounded.ltmle(stopping="scaled_var", step_search=False, iterate="best")
        result = bounded.fitted(estimator, frame, id="id")
        longitudinal.assert_hand_match(result, "scaled_var")
        for _, step in longitudinal.every_node(result):
            assert step.fluctuation.best_iterate is not None

    def test_a_repeated_outcome_and_a_long_panel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def best_ltmle(*args: Any, **kwargs: Any) -> LTMLE:
            kwargs.pop("stopping", None)
            return LTMLE(
                *args, step_bound=0.05, step_search=False, iterate="best", max_iter=8, **kwargs
            )

        monkeypatch.setattr(longitudinal, "LTMLE", best_ltmle)
        frame = longitudinal.repeated_module.panel()
        result = longitudinal.repeated_panel("relative", longitudinal.COMPOSITIONS["static"], frame)
        assert result.data.is_repeated
        assert assert_node_best(result, 0.05, False) >= 0
        panel = longitudinal.long_panel_fit("relative")
        assert panel.data.panel is not None
        assert_node_best(panel, 0.05, False)

    def test_the_working_model_reports_the_best_iterate_of_its_stacked_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
        original = msm_module.solve_fluctuation

        def spy(*args: Any, **kwargs: Any) -> Any:
            fit = original(*args, **kwargs)
            calls.append((args, kwargs, fit))
            return fit

        monkeypatch.setattr(msm_module, "solve_fluctuation", spy)
        estimator = LTMLE(
            longitudinal.PLANS,
            msm=longitudinal.msm_module.DOSE,
            n_folds=1,
            simultaneous=False,
            step_bound=0.05,
            step_search=False,
            iterate="best",
            max_iter=8,
            outcome_learner=LogisticRegression(max_iter=1000, random_state=0),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=0),
            censoring_learner=LogisticRegression(max_iter=1000, random_state=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(
                longitudinal.msm_module.panel(120),
                outcome="Y",
                treatment=("A1", "A2"),
                baseline=("W1", "W2"),
                time_varying=(("L1",), ("L2",)),
                censoring=("C1", "C2"),
            )
        assert calls
        assert all(kwargs["iterate"] == "best" for _, kwargs, _ in calls)
        assert prefix_problems(calls, original) == []
        assert any(
            fit.best_iterate is not None and fit.best_iterate < fit.n_iter for *_, fit in calls
        )
