"""Exact-law tests of the continuous-time longitudinal TMLE (X6 part (e), T-e1 to T-e16b, T-e24).

Laws PL and PL2 (``tests/discrete_law_continuous_longitudinal.py``) enter by weight, so the
weighted empirical law equals the law. Under saturated learners every estimate equals the
law's g-formula and its influence curve the Gateaux derivative (T-e1). A defect of a term, a
weight or a clever covariate shows under wrong intensities and continuation (``nol``, which
drop ``L``) with the right mechanism, where the walk must repair the estimate; a defect of the
initial slot values shows under saturated intensities with a mechanism that omits ``L``
(``gnol``). Each mutation below is checked to break the exact assertion, and keeps a control
on which its term vanishes.

The walk's smallest step (``MIN_STEP = 1e-8``) bounds how far a ``nol`` walk can push the
score below its starting value, so a correct ``nol`` fit reaches the truth to about 1e-9; the
estimates are checked to ``ATOL = 1e-8`` and every mutation misses by more than 1e-6.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly.survival import _longitudinal_nuisance as nuisance
from cleverly.survival import _longitudinal_targeting as T
from cleverly.survival import longitudinal
from cleverly.survival._longitudinal_nuisance import ContinuationModel
from tests.discrete_law_continuous_longitudinal import (
    point_law,
    point_law_2,
    static,
    treat_when_l,
)
from tests.unit._continuous_longitudinal_support import exact_errors, rule_when_l

pytestmark = pytest.mark.xdist_group("continuous_longitudinal_exact")

#: The engine's soft outcome, read by the chain check even under its mutation.
_SOFT_OUTCOME = T.soft_outcome
#: The exact assertion's tolerance on an estimate.
ATOL = 1e-8
#: Every mutation must miss by more than this.
MISS = 1e-6

STATIC = {"always": (1, static(1)), "never": (0, static(0))}


def _laws() -> list[Any]:
    return [point_law(), point_law_2()]


# ---------------------------------------------------------------------- T-e1


@pytest.mark.parametrize("law", _laws(), ids=["PL", "PL2"])
def test_saturated_fit_equals_g_formula_and_gateaux(monkeypatch: Any, law: Any) -> None:
    """T-e1: every risk equals the g-formula and its influence curve the Gateaux derivative."""
    errors = exact_errors(monkeypatch, law, STATIC)
    assert errors["psi"] < 1e-10
    assert errors["ic"] < 1e-10


def test_rule_regime_equals_g_formula(monkeypatch: Any) -> None:
    """T-e1, T-e15: a rule read at each decision visit (treat when the current L is 1)."""
    regimes = {"rule": (rule_when_l, treat_when_l), "never": (0, static(0))}
    errors = exact_errors(monkeypatch, point_law(), regimes)
    assert errors["psi"] < 1e-10 and errors["ic"] < 1e-10


@pytest.mark.parametrize("law", _laws(), ids=["PL", "PL2"])
def test_walk_repairs_wrong_intensities(monkeypatch: Any, law: Any) -> None:
    """The base of the mutations: under ``nol`` learners the walk reaches the truth."""
    assert exact_errors(monkeypatch, law, STATIC, learners="nol")["psi"] < ATOL


def test_walk_needs_no_step_at_wrong_mechanism(monkeypatch: Any) -> None:
    """Under ``gnol`` the saturated intensities already give the truth."""
    assert exact_errors(monkeypatch, point_law(), STATIC, learners="gnol")["psi"] < 1e-10


# ---------------------------------------------------------------------- mutations


def _scores_with(monkeypatch: Any, change: Any) -> None:
    """Patch the score so it reads ``change(problem, state)`` in place of the node's inputs."""
    original = T.node_scores

    def scores(problem: Any, state: Any, parts: Any = None) -> Any:
        parts = T.node_curves(problem, state) if parts is None else parts
        mutated_problem, mutated_state = change(problem, state)
        return original(mutated_problem, mutated_state, parts)

    monkeypatch.setattr(T, "node_scores", scores)


def _drop_exit(index: Any) -> Any:
    def change(problem: Any, state: Any) -> Any:
        e = index(problem, state)
        if e is None:
            return problem, state
        exits = state.exits.copy()
        events = problem.events.copy()
        exits[e] = 0.0
        events[e] = False
        return replace(problem, events=events), T.NodeState(exits, state.continuation)

    return change


def test_visit_term(monkeypatch: Any) -> None:
    """T-e2a: dropping the visit-intensity term breaks PL and PL2, not PL with certain visits."""
    for law in _laws():
        with monkeypatch.context() as patch:
            _scores_with(patch, _drop_exit(lambda p, s: -1 if p.has_visit else None))
            assert exact_errors(patch, law, STATIC)["ic"] > MISS
            assert exact_errors(patch, law, STATIC, learners="nol")["psi"] > MISS
    # Control: with certain visits the guard makes Z = v_v where the visit empties the segment.
    _scores_with(monkeypatch, _drop_exit(lambda p, s: -1 if p.has_visit else None))
    control = exact_errors(monkeypatch, point_law(certain_visits=True), STATIC)
    assert control["psi"] < 1e-10 and control["ic"] < 1e-10


def _no_covariate_term(problem: Any, state: Any) -> Any:
    return replace(problem, visit_at=np.full_like(problem.visit_at, -1)), state


def test_covariate_term(monkeypatch: Any) -> None:
    """T-e2b: dropping the covariate term breaks PL and PL2, not PL where L moves nothing."""
    for law in _laws():
        with monkeypatch.context() as patch:
            _scores_with(patch, _no_covariate_term)
            assert exact_errors(patch, law, STATIC)["ic"] > MISS
            assert exact_errors(patch, law, STATIC, learners="nol")["psi"] > MISS
    _scores_with(monkeypatch, _no_covariate_term)
    control = exact_errors(monkeypatch, point_law(l_moves=False), STATIC)
    assert control["ic"] < 1e-10


def test_death_terms(monkeypatch: Any) -> None:
    """T-e2c: dropping cause 2's term, or conditioning on S(u-)(1 - dL_l), breaks PL."""
    law = point_law()
    with monkeypatch.context() as patch:
        _scores_with(patch, _drop_exit(lambda p, s: 1))
        assert exact_errors(patch, law, STATIC)["ic"] > MISS
    original = T.node_scores

    def conditioned(problem: Any, state: Any, parts: Any = None) -> Any:
        parts = T.node_curves(problem, state) if parts is None else parts
        out = original(problem, state, parts)
        exits, Y = state.exits, problem.at_risk
        before = np.concatenate([np.ones((exits.shape[1], 1)), parts.survival[:, :-1]], axis=1)
        h = problem.slot_weight * problem.follow[:, None]
        J = exits.shape[0] - int(problem.has_visit)
        for p in range(problem.n_targets):
            for l in range(J):
                if l == problem.cause[p]:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    other = parts.values[p] * parts.survival / (before * (1.0 - exits[l]))
                other = np.where(np.isfinite(other), other, parts.values[p])
                residual = problem.events[l] - Y * exits[l]
                out[:, p] += (h * (parts.values[p] - other) * residual).sum(axis=1)
        return out

    monkeypatch.setattr(T, "node_scores", conditioned)
    assert exact_errors(monkeypatch, law, STATIC)["ic"] > MISS


def _visit_terms(monkeypatch: Any, law: Any) -> list[Any]:
    """Each visit node's per-row visit term at its initial fit, sum_u h c_v (dN_v - Y dE_v)."""
    terms: list[Any] = []
    original_walk = T.walk_node

    def walk(problem: Any, initial: Any, **kw: Any) -> Any:
        if problem.has_visit:
            parts = T.node_curves(problem, initial)
            residual = problem.events[-1] - problem.at_risk * initial.exits[-1]
            h = problem.slot_weight * problem.follow[:, None]
            coefficient = parts.visit_value - parts.values
            terms.append(np.einsum("cm,pcm->cp", h * residual, coefficient))
        return original_walk(problem, initial, **kw)

    monkeypatch.setattr(T, "walk_node", walk)
    exact_errors(monkeypatch, law, STATIC)
    return terms


def test_survival_guard(monkeypatch: Any) -> None:
    """T-e2d: with certain visits the guard Z = v_v makes the visit term zero row by row.

    The total of a row that exits at u is the same for any Z there, so the estimate and its
    curve cannot see the guard; the visit term alone can.
    """
    law = point_law(certain_visits=True)
    with monkeypatch.context() as patch:
        terms = _visit_terms(patch, law)
    assert terms and all(np.max(np.abs(term)) < 1e-12 for term in terms)
    original = T.node_curves

    def unguarded(problem: Any, state: Any) -> Any:
        parts = original(problem, state)
        parts.values = np.where(parts.survival[None] > 0, parts.values, 0.0)
        return parts

    monkeypatch.setattr(T, "node_curves", unguarded)
    assert max(np.max(np.abs(term)) for term in _visit_terms(monkeypatch, law)) > MISS


def test_death_before_visit(monkeypatch: Any) -> None:
    """T-e4: the visit at a death time needs survival of the death; the other order fails."""
    monkeypatch.setattr(longitudinal, "visit_exit", lambda deaths, visit: visit)
    assert exact_errors(monkeypatch, point_law(), STATIC)["psi"] > 1e-3


def test_treatment_factor_at_every_visit(monkeypatch: Any) -> None:
    """T-e6: the baseline treatment factor only (pi = 1 at later visits) breaks ``nol``."""
    original = longitudinal._Fit._fit_cap

    def baseline_only(self: Any, cap: int) -> Any:
        fits = original(self, cap)
        later = self.data.visit_index >= 1
        for p1 in fits.p1:
            p1[later] = 1.0
        return fits

    monkeypatch.setattr(longitudinal._Fit, "_fit_cap", baseline_only)
    regimes = {"always": (1, static(1)), "never": (0, static(0))}
    errors = exact_errors(monkeypatch, point_law(), regimes, learners="nol")
    assert errors["psi"] > MISS


def test_censoring_weight(monkeypatch: Any) -> None:
    """T-e7: S_c(u) in place of S_c(u-), or censoring before the visit, breaks PL."""
    law = point_law(censor_reads_a=True)
    with monkeypatch.context() as patch:
        patch.setattr(longitudinal, "censoring_before", lambda retention: retention)
        assert exact_errors(patch, law, STATIC, learners="nol")["psi"] > MISS
    # No tie shift: censoring at a visit time is read with the segment before the visit.
    monkeypatch.setattr(nuisance, "shift", lambda data: 0.0)
    assert exact_errors(monkeypatch, law, STATIC, learners="nol")["psi"] > MISS


def test_regime_slot(monkeypatch: Any) -> None:
    """T-e8: the intensities read at the observed A(s_k) break a ``gnol`` fit."""
    original = nuisance.design

    def observed(data: Any, coded: Any, visits: Any, slot: Any = None) -> Any:
        return original(data, coded, visits)

    monkeypatch.setattr(nuisance, "design", observed)
    assert exact_errors(monkeypatch, point_law(), STATIC, learners="gnol")["psi"] > MISS


def test_continuation_at_counterfactual_visit_times(monkeypatch: Any) -> None:
    """T-e9: Z read only at the observed next visit, or the continuation at s_k, breaks PL2."""
    law = point_law_2()
    original_curves = T.node_curves

    def observed_only(problem: Any, state: Any) -> Any:
        lam = state.continuation
        if lam is not None:
            keep = np.zeros(lam.shape[1:3], dtype=bool)
            seen = np.flatnonzero(problem.visit_at >= 0)
            keep[seen, problem.visit_at[seen]] = True
            lam = np.where(keep[None, :, :, None], lam, 0.0)
        return original_curves(problem, T.NodeState(state.exits, lam))

    with monkeypatch.context() as patch:
        patch.setattr(T, "node_curves", observed_only)
        assert exact_errors(patch, law, STATIC)["ic"] > MISS
        assert exact_errors(patch, point_law(), STATIC)["ic"] > MISS
    original_hazards = ContinuationModel.hazards

    def at_visit(self: Any, x: Any, u: Any) -> Any:
        return original_hazards(self, x, x[:, -1])

    monkeypatch.setattr(ContinuationModel, "hazards", at_visit)
    assert exact_errors(monkeypatch, law, STATIC, learners="gnol")["psi"] > MISS


def _chain(monkeypatch: Any) -> list[tuple[Any, Any, Any]]:
    """Per continuation fit: its labels, the later node's targeted and initial soft outcome."""
    walked: list[tuple[Any, Any, Any]] = []
    out: list[tuple[Any, Any, Any]] = []
    original_walk, original_soft = T.walk_node, _SOFT_OUTCOME
    original_fit = ContinuationModel.fit.__func__

    def walk(problem: Any, initial: Any, **kw: Any) -> Any:
        record = original_walk(problem, initial, **kw)
        walked.append((problem, initial, record.state))
        return record

    def fit(cls: Any, learner: Any, x: Any, u: Any, labels: Any, *args: Any) -> Any:
        problem, initial, final = walked[-1]
        out.append((labels, original_soft(problem, final), original_soft(problem, initial)))
        return original_fit(cls, learner, x, u, labels, *args)

    monkeypatch.setattr(T, "walk_node", walk)
    monkeypatch.setattr(ContinuationModel, "fit", classmethod(fit))
    exact_errors(monkeypatch, point_law_2(), STATIC, learners="nol")
    return out


def test_soft_outcome_chain(monkeypatch: Any) -> None:
    """T-e9b: node k's continuation trains on node k + 1's targeted soft outcome.

    The estimate cannot see this link: node k tilts its own continuation, which solves the
    covariate term whatever the labels, so the check reads the labels themselves.
    """
    with monkeypatch.context() as patch:
        chain = _chain(patch)
    assert all(np.allclose(labels, targeted, rtol=0, atol=1e-14) for labels, targeted, _ in chain)
    # Witness: the walk moves the soft outcome, so labels from the initial fit differ.
    assert any(np.max(np.abs(targeted - initial)) > MISS for _, targeted, initial in chain)
    initials: dict[int, Any] = {}
    original_walk, original_soft = T.walk_node, T.soft_outcome

    def walk(problem: Any, initial: Any, **kw: Any) -> Any:
        initials[id(problem)] = initial
        return original_walk(problem, initial, **kw)

    monkeypatch.setattr(T, "walk_node", walk)
    monkeypatch.setattr(
        T, "soft_outcome", lambda problem, state: original_soft(problem, initials[id(problem)])
    )
    chain = _chain(monkeypatch)
    assert not all(
        np.allclose(labels, targeted, rtol=0, atol=1e-14) for labels, targeted, _ in chain
    )


def test_second_visit_node(monkeypatch: Any) -> None:
    """T-e9c: node 2's score part is nonzero; stopping the pass at node 1 breaks PL2."""
    parts: list[Any] = []
    original_walk = T.walk_node

    def walk(problem: Any, initial: Any, **kw: Any) -> Any:
        record = original_walk(problem, initial, **kw)
        parts.append((problem, T.node_rows(problem, initial)[0]))
        return record

    with monkeypatch.context() as patch:
        patch.setattr(T, "walk_node", walk)
        exact_errors(patch, point_law_2(), STATIC, learners="nol")
    # Node K* = 2 is the one node without a visit exit.
    deepest = [rows for problem, rows in parts if not problem.has_visit]
    assert deepest and all(np.max(np.abs(rows)) > 1e-6 for rows in deepest)
    original_cap = nuisance.visit_cap

    def capped(*args: Any, **kwargs: Any) -> Any:
        cap, fold, count = original_cap(*args, **kwargs)
        return min(cap, 1), fold, count

    monkeypatch.setattr(nuisance, "visit_cap", capped)
    assert exact_errors(monkeypatch, point_law_2(), STATIC)["psi"] > MISS


def test_earlier_treatment_in_design(monkeypatch: Any) -> None:
    """T-e10: dropping A(s_{k-1}) from the design breaks a ``gnol`` fit on PL, whose deaths read it."""
    original = nuisance.design

    def no_previous(data: Any, coded: Any, visits: Any, slot: Any = None) -> Any:
        X = original(data, coded, visits, slot=slot)
        X[:, 3] = 0.0
        return X

    monkeypatch.setattr(nuisance, "design", no_previous)
    assert exact_errors(monkeypatch, point_law(), STATIC, learners="gnol")["psi"] > MISS


def test_weight_floor(monkeypatch: Any) -> None:
    """T-e11: a binding floor changes the influence curve on the rare cell's units only."""
    law = point_law(rare_cell=True)
    from tests.unit._continuous_longitudinal_support import law_data, saturated

    curves = {}
    for floor in (1e-6, 1 / 32):
        estimator = saturated(monkeypatch, law.targets, {"always": 1, "never": 0}, g_bounds=floor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit_data(law_data(law))
        name = "cif_regimen[always, 1 @ t=2]"
        curves[floor] = (result[name].psi, result[name].influence_curve, result.record)
    psi_free, ic_free, free = curves[1e-6]
    psi_floor, ic_floor, bound = curves[1 / 32]
    assert free.bounded_share == 0.0 and bound.bounded_share > 0.0
    # The floor leaves the plug-in (no step at the saturated fit) and moves the rare cell only.
    assert abs(psi_free - psi_floor) < 1e-12
    rare = np.array(
        [tokens[0][1] == 1 and tokens[1][2] == 1 and tokens[2][2] == 1 for tokens, _ in law.paths]
    )
    assert np.max(np.abs(ic_free - ic_floor)[~rare]) < 1e-12
    assert np.max(np.abs(ic_free - ic_floor)[rare]) > 1e-3


def test_rule_read_at_decision_visits(monkeypatch: Any) -> None:
    """T-e15: the rule read with baseline L, or at a non-decision visit, breaks PL."""
    regimes = {"rule": (rule_when_l, treat_when_l), "never": (0, static(0))}
    original = longitudinal.rule_frame

    def baseline(data: Any, visits: Any) -> Any:
        frame = original(data, visits)
        first = data.first_visit[data.unit[visits]]
        frame["L"] = data.covariates[first, 0]
        return frame

    with monkeypatch.context() as patch:
        patch.setattr(longitudinal, "rule_frame", baseline)
        assert exact_errors(patch, point_law(), regimes)["psi"] > MISS
    # Control: a law with a non-decision visit; the rule is not read there.
    law = point_law(non_decision=True)
    assert exact_errors(monkeypatch, law, regimes)["psi"] < 1e-10
    from cleverly.survival import _longitudinal_data

    original_from = _longitudinal_data.ContinuousLongitudinalData.from_frame.__func__

    def every_decision(cls: Any, data: Any, **kw: Any) -> Any:
        kw["decision"] = None
        return original_from(cls, data, **kw)

    monkeypatch.setattr(
        _longitudinal_data.ContinuousLongitudinalData, "from_frame", classmethod(every_decision)
    )
    assert exact_errors(monkeypatch, law, regimes)["psi"] > MISS


def test_non_followers_score_nothing(monkeypatch: Any) -> None:
    """T-e16: keeping the exits of rows off the regime breaks ``nol``."""
    _scores_with(
        monkeypatch,
        lambda problem, state: (
            replace(problem, follow=np.ones_like(problem.follow)),
            state,
        ),
    )
    assert exact_errors(monkeypatch, point_law(), STATIC, learners="nol")["psi"] > MISS


@pytest.mark.parametrize("law", _laws(), ids=["PL", "PL2"])
def test_slot_weight_tilt(monkeypatch: Any, law: Any) -> None:
    """T-e16b: tilting with the factual weight h in place of the slot weight misses by 1e-3."""
    original = T.node_direction

    def factual(problem: Any, state: Any, v: Any, parts: Any = None) -> Any:
        weight = problem.slot_weight * problem.follow[:, None]
        return original(replace(problem, slot_weight=weight), state, v, parts)

    monkeypatch.setattr(T, "node_direction", factual)
    assert exact_errors(monkeypatch, law, STATIC, learners="nol")["psi"] > 1e-3


# ---------------------------------------------------------------------- T-e12, T-e24


def test_node_cutoffs_bound_the_total(monkeypatch: Any) -> None:
    """T-e12: with each node walked to s_n / N, every target's total score is at most s_n."""
    from tests.unit._continuous_longitudinal_support import law_data, saturated

    law = point_law_2()

    def fit(**kw: Any) -> Any:
        estimator = saturated(
            monkeypatch,
            law.targets,
            {"always": 1, "never": 0},
            tol=0.02,
            max_iter=4000,
            hazard_library=[nuisance_free_hazard()],
            visit_library=[nuisance_free_hazard()],
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return estimator.fit_data(law_data(law))

    result = fit()
    assert max(result.record.standardized_score) <= 0.02
    monkeypatch.setattr(longitudinal, "node_cutoff", lambda cutoff, nodes: cutoff)
    assert max(fit().record.standardized_score) > 0.02


def nuisance_free_hazard() -> Any:
    """A hazard learner that drops L, as the ``nol`` setting does."""
    from tests.unit._continuous_longitudinal_support import L_COLUMN, CellHazard

    return CellHazard((L_COLUMN,))


def test_target_past_the_cap(monkeypatch: Any) -> None:
    """T-e24: with K* = 1 on PL2 the fit targets the regime at baseline and visit 1 only."""
    monkeypatch.setattr(nuisance, "VISIT_CAP_MIN_FOLLOWERS", 120)
    errors = exact_errors(monkeypatch, point_law_2(), STATIC, cap=1)
    assert errors["psi"] < 1e-10 and errors["ic"] < 1e-10
    # Against the regime at every visit, the capped fit misses.
    assert exact_errors(monkeypatch, point_law_2(), STATIC)["psi"] > MISS
    # Mutation: censor at visit K* + 1 (no visit intensity past the cap) in place of the merge.
    original = nuisance.segments

    def censored(data: Any, cap: int, last: float) -> Any:
        seg = original(data, cap + 1, last)
        keep = seg.k <= cap
        past = (seg.k == cap) & (seg.next_visit >= 0)
        return nuisance.Segments(
            unit=seg.unit[keep],
            visit=seg.visit[keep],
            k=seg.k[keep],
            entry=seg.entry[keep],
            stop=seg.stop[keep],
            exit=np.where(past, 0, seg.exit)[keep],
            next_visit=np.where(past, -1, seg.next_visit)[keep],
            cap=cap,
        )

    monkeypatch.setattr(nuisance, "segments", censored)
    assert exact_errors(monkeypatch, point_law_2(), STATIC, cap=1)["psi"] > MISS
