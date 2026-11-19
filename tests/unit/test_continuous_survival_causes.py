r"""The continuous-time TMLE with one cause and with three (X6 part (b), plan rev 5, 4.2).

T-b1 and T-b2 on an exact law of :math:`J` causes, built here for any :math:`J`: a binary
covariate :math:`W`, a binary arm, event nodes at times 1 and 2 whose cause hazards depend on
:math:`(A, W)`, and censoring at time 1.5 that depends on :math:`(A, W)`.  Every probability is a
multiple of 1/2048, so the sample of 2,048 rows realises the law exactly: the law's own hazards
are the saturated fit, the g-formula is a quotient of cell masses, and the efficient influence
function is the complex-step Gateaux derivative of that quotient.  The hazards are 1/8 to 1/2 per
node, so the squared term of the multinomial variance is far from zero.  With several causes the
all-cause hazard varies with :math:`W`, so a fit that ignores :math:`W` has the wrong survival and
the clever covariate's term shared by every cause, :math:`-\sum_j H_j / S`, does not vanish from
its score.

Each witness is exact; each mutation drops one cause-indexed term from the computation it feeds
and fails where that term is nonzero (CLAUDE.md, "Scientific changes").  With one cause there is
no competing cause: the tests check that nothing reads an empty sum and that the estimand is the
risk of the one event type.
"""

from __future__ import annotations

import warnings
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, _targeting
from cleverly.survival import result as result_module
from cleverly.survival._targeting import (
    Problem,
    curves,
    evaluate,
    point_weights,
    predictable_covariance,
    walk,
)
from cleverly.survival.result import MEASURED_DEFAULTS, continuous_survival_support

#: Relative tolerance of the exact identities: rounding only.
EXACT = 1e-10
#: A mutation must move some entry by more than this, relative to the largest entry.
MUTATION = 1e-6
HORIZONS = (1.0, 2.0)
GRID = np.array([1.0, 1.5, 2.0])
CENSOR = 1.5
END = 2.5


def _pw(w: int) -> Fraction:
    return Fraction(1, 2)


def _pa(a: int, w: int) -> Fraction:
    """P(A = a | W = w): one quarter or one half treated."""
    p1 = Fraction(1, 4) if w == 0 else Fraction(1, 2)
    return p1 if a == 1 else 1 - p1


def _hazard(J: int, l: int, node: int, a: int, w: int) -> Fraction:
    """The cause-``l`` hazard at event node ``node``: 1/8 to 4/8, varying with A, W and l.

    With three causes ``l`` runs over every residue mod 3, so without the cause-1 term at
    ``w = 1`` the all-cause hazard would be 3/4 in every cell.  That term moves it with the
    covariate.  One cause keeps the plain law, whose pooled-arm fit then misses every target.
    """
    bump = 1 if J > 1 and l == 0 and w == 1 else 0
    return Fraction(1 + (l + node + a + 2 * w) % 3 + bump, 8)


def _censoring(a: int, w: int) -> Fraction:
    return Fraction(1, 4) if (a + w) % 2 == 0 else Fraction(1, 2)


@cache
def _support(J: int) -> tuple[tuple[tuple[int, int, float, int], ...], tuple[Fraction, ...]]:
    """Every observed point ``(w, a, time, cause)`` and its probability."""
    points: list[tuple[int, int, float, int]] = []
    masses: list[Fraction] = []
    for w in (0, 1):
        for a in (0, 1):
            base = _pw(w) * _pa(a, w)
            h1 = [_hazard(J, l, 1, a, w) for l in range(J)]
            h2 = [_hazard(J, l, 2, a, w) for l in range(J)]
            s1 = 1 - sum(h1)
            c = _censoring(a, w)
            for l in range(J):
                points.append((w, a, 1.0, l + 1))
                masses.append(base * h1[l])
            points.append((w, a, CENSOR, 0))
            masses.append(base * s1 * c)
            for l in range(J):
                points.append((w, a, 2.0, l + 1))
                masses.append(base * s1 * (1 - c) * h2[l])
            points.append((w, a, END, 0))
            masses.append(base * s1 * (1 - c) * (1 - sum(h2)))
    assert sum(masses) == 1
    return tuple(points), tuple(masses)


@cache
def _sample(J: int) -> tuple[np.ndarray, ...]:
    """The exact sample: each point repeated by its count, and the first row of each point."""
    points, masses = _support(J)
    counts = [int(m * 2048) for m in masses]
    assert all(Fraction(c, 2048) == m for c, m in zip(counts, masses, strict=True))
    rows = np.repeat(np.array(points, dtype=float), counts, axis=0)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return rows, first[np.array(counts) > 0], np.array([c > 0 for c in counts])


def _functional(J: int, probs: np.ndarray, a: int, j: int, k: int) -> Any:
    """:math:`\\sum_w P(w) F_j(t_k \\mid a, w)` from point masses ``probs`` (complex allowed)."""
    points, _ = _support(J)
    cov = np.array([p[0] for p in points])
    arm = np.array([p[1] for p in points])
    time = np.array([p[2] for p in points])
    cause = np.array([p[3] for p in points])
    total = probs.sum()
    out = 0
    for w in (0, 1):
        cell = (cov == w) & (arm == a)
        survival, incidence = 1.0, 0.0
        for u in (g for g in GRID if g <= HORIZONS[k]):
            at_risk = probs[cell & (time >= u)].sum()
            jumps = [probs[cell & (time == u) & (cause == l + 1)].sum() / at_risk for l in range(J)]
            incidence = incidence + survival * jumps[j]
            survival = survival * (1 - sum(jumps))
        out = out + probs[cov == w].sum() / total * incidence
    return out


@cache
def _gateaux(J: int, a: int, j: int, k: int) -> np.ndarray:
    _, masses = _support(J)
    base = np.array([float(m) for m in masses])
    out = []
    for point in range(base.size):
        direction = np.zeros(base.size, dtype=complex)
        direction[point] = 1.0
        perturbed = (1 - 1e-30j) * base + 1e-30j * direction
        out.append(np.imag(_functional(J, perturbed, a, j, k)) / 1e-30)
    return np.array(out)


def _truth(J: int, a: int, j: int, k: int) -> float:
    _, masses = _support(J)
    return float(np.real(_functional(J, np.array([float(m) for m in masses]), a, j, k)))


def _targets(J: int) -> list[tuple[int, int, int]]:
    """``(a, j, k)`` in the engine's order ``p = (a J + j) K + k``."""
    return [(a, j, k) for a in (0, 1) for j in range(J) for k in range(len(HORIZONS))]


@cache
def _problem(J: int, *, wrong: bool = False) -> tuple[Problem, np.ndarray]:
    """The exact-law problem with the law's own hazards, or with hazards that ignore ``W``."""
    rows, _, _ = _sample(J)
    cov, arm, time = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2]
    cause = rows[:, 3].astype(int)
    n, M = time.size, GRID.size
    at_risk = (time[:, None] >= GRID[None, :]).astype(float)
    events = np.stack(
        [((time[:, None] == GRID[None, :]) & (cause[:, None] == l + 1)) for l in range(J)]
    )
    events = events.astype(float)
    dlam = np.zeros((2, J, n, M))
    weight = np.zeros((2, n, M))
    for a in (0, 1):
        for w in (0, 1):
            unit = cov == w
            for node, m in ((1, 0), (2, 2)):
                for l in range(J):
                    if wrong:
                        # The pooled hazard of arm a: the law's events over its at-risk mass.
                        cell = arm == a
                        value = events[l, cell, m].sum() / at_risk[cell, m].sum()
                    else:
                        value = float(_hazard(J, l, node, a, w))
                    dlam[a, l, unit, m] = value
            retention = np.array([1.0, 1.0, 1.0 - float(_censoring(a, w))])
            weight[a, unit] = 1.0 / (float(_pa(a, w)) * retention)[None, :]
    index = np.searchsorted(GRID, HORIZONS, side="right") - 1
    return Problem(arm, at_risk, events, weight, point_weights(index, M)), dlam


def _ic_error(J: int, problem: Problem, dlam: np.ndarray) -> float:
    _, first, kept = _sample(J)
    evaluation = evaluate(problem, dlam)
    return max(
        float(np.max(np.abs(evaluation.influence[first, p] - _gateaux(J, a, j, k)[kept])))
        for p, (a, j, k) in enumerate(_targets(J))
    )


def _bias(J: int, psi: np.ndarray) -> np.ndarray:
    return np.array([psi[p] - _truth(J, a, j, k) for p, (a, j, k) in enumerate(_targets(J))])


# ----------------------------------------------------------------------- T-b1, T-b2


@pytest.mark.parametrize("J", [1, 3])
def test_the_saturated_fit_is_the_g_formula_and_needs_no_step(J: int) -> None:
    problem, dlam = _problem(J)
    result = walk(problem, dlam, cutoff=1e-12)
    assert result.steps == 0
    np.testing.assert_allclose(_bias(J, result.evaluation.psi), 0.0, atol=1e-14)
    np.testing.assert_allclose(result.evaluation.score, 0.0, atol=1e-14)
    # Every truth is far from zero: no cause is empty.
    assert min(_truth(J, *target) for target in _targets(J)) > 0.05


@pytest.mark.parametrize("J", [1, 3])
def test_the_influence_curve_is_the_gateaux_derivative(J: int) -> None:
    assert _ic_error(J, *_problem(J)) < 1e-12


def test_reading_the_targets_in_cause_major_order_breaks_the_three_cause_curve() -> None:
    # Mutation of the target order: (j K + k) 2 + a in place of (a J + j) K + k.  With three
    # causes the two orders differ at almost every position, so the check above reads the order.
    J = 3
    problem, dlam = _problem(J)
    _, first, kept = _sample(J)
    influence = evaluate(problem, dlam).influence
    K = len(HORIZONS)
    moved = [
        float(np.max(np.abs(influence[first, (j * K + k) * 2 + a] - _gateaux(J, a, j, k)[kept])))
        for a, j, k in _targets(J)
        if (j * K + k) * 2 + a != (a * J + j) * K + k
    ]
    assert moved and min(moved) > 1e-3


@pytest.mark.parametrize("J", [1, 3])
def test_targeting_wrong_hazards_reaches_the_truth(J: int) -> None:
    problem, dlam = _problem(J, wrong=True)
    result = walk(problem, dlam, cutoff=1e-7, max_steps=20_000)
    initial = np.abs(_bias(J, result.initial.psi))
    # Nonzero witness: the hazards that ignore W miss most targets, every cause included.
    assert (initial > 1e-3).sum() >= len(initial) // 2
    for j in range(J):
        rows = [p for p, (_, jj, _) in enumerate(_targets(J)) if jj == j]
        assert np.max(initial[rows]) > 1e-3, j
    assert result.converged
    assert np.max(np.abs(_bias(J, result.evaluation.psi))) < 1e-3 * np.max(initial)
    # The incidences of every cause and the survival add to one after the walk.
    survival, incidence = curves(result.dlam)
    np.testing.assert_allclose(survival + incidence.sum(axis=1), 1.0, atol=1e-13)


def test_the_three_cause_all_cause_hazard_varies_with_the_covariate() -> None:
    # Nonzero witness for the shared term of the direction: the pooled-arm fit's survival is
    # wrong only if the all-cause hazard moves with W.
    totals = {
        sum(_hazard(3, l, node, a, w) for l in range(3))
        for node in (1, 2)
        for a in (0, 1)
        for w in (0, 1)
    }
    assert totals == {Fraction(3, 4), Fraction(7, 8)}


def _path_derivative(J: int, direction: Any) -> tuple[float, float]:
    r"""The log-likelihood's path derivative along ``direction`` and the score norm it equals.

    The direction is that of the walk, :math:`\Sigma^{-1} P_n D / \lVert P_n D \rVert`, on the
    pooled-arm fit; the multinomial likelihood of each cell's jumps is read at the received arm
    (``tests/unit/test_continuous_survival.py``, check 4).
    """
    problem, dlam = _problem(J, wrong=True)
    evaluation = evaluate(problem, dlam)
    sigma = np.cov(evaluation.influence, rowvar=False, bias=True)
    metric = _targeting._metric(sigma, "covariance")
    x = evaluation.score
    length = float(np.sqrt(x @ metric @ x))
    path = direction(problem, evaluation, metric @ x / length, dlam)
    factual = np.stack([(problem.arm == a) for a in (0, 1)])[:, None, :, None]

    def loglik(step: float) -> float:
        jumps = (_targeting.tilt_jumps(dlam, step * path) * factual).sum(axis=0)
        none = 1.0 - jumps.sum(axis=0)
        stay = problem.at_risk - problem.events.sum(axis=0)
        return (
            float(
                np.sum(problem.events * np.log(np.where(problem.events > 0, jumps, 1.0)))
                + np.sum(stay * np.log(none))
            )
            / problem.n
        )

    h = 1e-6
    return (loglik(h) - loglik(-h)) / (2 * h), length


@pytest.mark.parametrize("J", [1, 3])
def test_the_path_derivative_of_the_log_likelihood_is_the_score_norm(J: int) -> None:
    derivative, length = _path_derivative(J, _targeting._direction)
    assert derivative == pytest.approx(length, rel=1e-6)


def test_a_shared_term_without_the_third_cause_leaves_the_score_norm() -> None:
    # Mutation: the term every cause shares, -sum_j H_j / S, summed over the first two causes.
    # The direction is then not the score's, and the path derivative misses its norm.
    def two_cause_shared(
        problem: Problem, evaluation: Any, v: np.ndarray, dlam: np.ndarray
    ) -> np.ndarray:
        J, Q = problem.n_causes, problem.n_rows
        assert problem.expand is None
        out = np.zeros((2, J, *problem.at_risk.shape))
        for a in (0, 1):
            own = v[a * J * Q : (a + 1) * J * Q].reshape(J, Q) @ problem.target_weights
            own = np.cumsum(own[:, ::-1], axis=1)[:, ::-1]
            shared = sum(
                _targeting.conditional_risk(evaluation.survival[a], dlam[a, j], own[j][None, :])
                for j in range(2)
            )
            for l in range(J):
                out[a, l] = problem.weight[a] * (own[l][None, :] - shared)
        return out

    shipped, length = _path_derivative(3, _targeting._direction)
    mutated, _ = _path_derivative(3, two_cause_shared)
    assert shipped == pytest.approx(length, rel=1e-6)
    assert abs(mutated - length) > 1e3 * MUTATION * length


def test_freezing_the_third_cause_keeps_the_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mutation: drop cause 3 from the tilt.  Its hazards stay at the wrong fit, and the walk
    # cannot reach the truth (every cause's risk depends on cause 3 through the survival).
    J = 3
    problem, dlam = _problem(J, wrong=True)
    original = _targeting._direction

    def frozen(problem: Problem, evaluation: Any, v: np.ndarray, dlam: np.ndarray) -> np.ndarray:
        out = original(problem, evaluation, v, dlam)
        out[:, 2] = 0.0
        return out

    monkeypatch.setattr(_targeting, "_direction", frozen)
    result = walk(problem, dlam, cutoff=1e-9, max_steps=2_000)
    assert np.max(np.abs(_bias(J, result.evaluation.psi))) > 1e-3


def test_a_survival_that_leaves_out_the_third_cause_breaks_the_sum_and_the_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mutation: the product limit over the first two causes only.  The incidences and the
    # survival no longer add to one, and the influence curve leaves the Gateaux derivative.
    J = 3
    problem, dlam = _problem(J)

    def two_causes(jumps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        survival = np.cumprod(1.0 - jumps[:, :2].sum(axis=1), axis=-1)
        incidence = np.cumsum(_targeting._before(survival)[:, None] * jumps, axis=-1)
        return survival, incidence

    monkeypatch.setattr(_targeting, "curves", two_causes)
    survival, incidence = two_causes(dlam)
    assert np.max(np.abs(survival + incidence.sum(axis=1) - 1.0)) > 1e-2
    assert _ic_error(J, problem, dlam) > 1e-3


@pytest.mark.parametrize("J", [1, 3])
def test_tilting_only_the_received_arm_keeps_the_bias(J: int) -> None:
    # The counterfactual arm's tilt vanishes in no score; with it masked the walk solves the
    # factual scores and stays biased (the factual-tilt control of the study, exactly).
    problem, dlam = _problem(J, wrong=True)
    result = walk(problem, dlam, cutoff=1e-9, max_steps=20_000, tilt="factual")
    initial = np.abs(_bias(J, result.initial.psi))
    assert np.max(np.abs(_bias(J, result.evaluation.psi))[initial > 1e-3]) > 1e-3


# ------------------------------------------------- the predictable variation, any J


def _moments(problem: Problem, dlam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evaluation = evaluate(problem, dlam)
    curves_ = evaluation.influence
    return predictable_covariance(problem, evaluation, dlam), curves_.T @ curves_ / curves_.shape[0]


def _gap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)) / np.max(np.abs(right)))


@pytest.mark.parametrize("J", [1, 3])
def test_the_predictable_variance_is_the_population_second_moment(J: int) -> None:
    pv, moments = _moments(*_problem(J))
    assert _gap(pv, moments) < EXACT
    # Witness: the cross-target block is far from zero.
    assert np.max(np.abs(pv - np.diag(np.diag(pv)))) > 0.1 * np.max(np.diag(pv))


def test_dropping_the_square_breaks_the_one_cause_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    # With one cause V = c^2 dLambda (1 - dLambda): the square is the (1 - dLambda) factor, which
    # the law's jumps of 1/8 to 3/8 keep far from one.
    def no_square(clever: np.ndarray, dlam: np.ndarray, rho: np.ndarray) -> np.ndarray:
        P = clever.shape[1]
        flat = (clever[0] * np.sqrt(rho * dlam[0])[None]).reshape(P, -1)
        return flat @ flat.T

    monkeypatch.setattr(_targeting, "_multinomial_moment", no_square)
    assert _gap(*_moments(*_problem(1))) > MUTATION


def test_leaving_the_third_cause_out_of_the_cross_term_breaks_the_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The negative cross terms of V involve all three causes; summing the first moment over two
    # of them breaks the identity.
    def two_cause_first(clever: np.ndarray, dlam: np.ndarray, rho: np.ndarray) -> np.ndarray:
        J, P = clever.shape[:2]
        out = np.zeros((P, P))
        first = np.zeros(clever.shape[1:])
        for l in range(J):
            flat = (clever[l] * np.sqrt(rho * dlam[l])[None]).reshape(P, -1)
            out += flat @ flat.T
            if l < 2:
                first += clever[l] * dlam[l][None]
        flat = (first * np.sqrt(rho)[None]).reshape(P, -1)
        return out - flat @ flat.T

    monkeypatch.setattr(_targeting, "_multinomial_moment", two_cause_first)
    assert _gap(*_moments(*_problem(3))) > MUTATION


# ------------------------------------------------------------------ the engine, any J


def _frame(J: int, n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.binomial(1, 1.0 / (1.0 + np.exp(-x)))
    rates = np.stack([np.exp(-1.5 + 0.3 * j + 0.4 * x - 0.4 * z) for j in range(J)])
    total = rates.sum(axis=0)
    event_time = rng.exponential(1.0 / total)
    cumulative = np.cumsum(rates / total, axis=0)
    cause = 1 + (rng.uniform(size=n)[None, :] > cumulative[:-1]).sum(axis=0)
    censor = rng.exponential(1.0 / 0.15, size=n)
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, censor),
            "event": np.where(event_time <= censor, cause, 0),
            "Z": z,
            "X": x,
        }
    )


@cache
def _fit(J: int) -> Any:
    estimator = ContinuousSurvivalTMLE(
        (0.5, 1.0, 2.0),
        arms={"t": 1, "c": 0},
        reference="c",
        causes={j: f"c{j}" for j in range(1, J + 1)},
        cross_fit=False,
        hazard_library=(CoxHazard("main"),),
        censoring_library=(CoxHazard("main"),),
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(_frame(J), time="time", event="event", treatment="Z", baseline=["X"])


@pytest.mark.parametrize("J", [1, 3])
def test_a_fit_names_one_risk_per_arm_cause_and_time(J: int) -> None:
    result = _fit(J)
    assert result.causes == tuple(f"c{j}" for j in range(1, J + 1))
    levels = [name for name in result.estimates if name.startswith(("risk_", "cif_"))]
    differences = [name for name in result.estimates if name.startswith("ate_")]
    assert len(levels) == 2 * J * 3 and len(differences) == J * 3
    # One event type is a risk of the event, with no cause in its name; with three causes each
    # incidence names its cause.
    head = "risk_regimen[t @ t=1]" if J == 1 else "cif_regimen[t, c3 @ t=1]"
    assert head in result.estimates
    # The survival scale is one minus the risk only with one cause; with three it refuses.
    if J == 1:
        survival = result.curve(scale="survival")
        risk = result.curve()
        level = risk["estimand"].str.startswith("risk_").to_numpy()
        expected = np.where(level, 1.0 - risk["psi"], -risk["psi"])
        np.testing.assert_allclose(survival["psi"], expected, atol=1e-12)
    else:
        with pytest.raises(Exception, match="not event-free survival"):
            result.curve(scale="survival")


@pytest.mark.parametrize("J", [1, 3])
def test_the_evalue_note_names_a_competing_cause_only_when_there_is_one(J: int) -> None:
    from cleverly.sensitivity import evalue

    result = _fit(J)
    pair = (
        ("risk_regimen[t @ t=1]", "risk_regimen[c @ t=1]")
        if J == 1
        else ("cif_regimen[t, c3 @ t=1]", "cif_regimen[c, c3 @ t=1]")
    )
    note = evalue(result, pair).note
    assert ("competing cause can prevent" in note) is (J > 1)
    assert ("one minus the event-free survival" in note) is (J == 1)


@pytest.mark.parametrize("J", [1, 3, 4])
def test_a_fit_with_other_than_two_causes_reads_the_causes_reason(J: int) -> None:
    # The fit has 300 units: one cause and three state the causes study's own size floor, and
    # four causes state that no study measures them.  The sentence names the number of causes,
    # so it reaches the engine's count rather than a sentence every small fit prints.
    result = _fit(J)
    noun = {1: "one cause", 3: "three causes", 4: "four causes"}[J]
    for estimate in result.estimates.values():
        support = estimate.support
        assert support.label == "experimental"
        assert f"The fit has {noun}" in support.reason
        if J == 4:
            assert support.limits[-1].startswith("No study measures four causes")
        else:
            assert support.limits[-1].startswith("The continuous-survival-causes study measures")


@pytest.mark.parametrize(
    ("causes", "measured", "label", "phrase"),
    [
        (1, frozenset({1, 2, 3}), "supported", "continuous-survival-causes study measures"),
        (3, frozenset({1, 2, 3}), "supported", "continuous-survival-causes study measures"),
        (3, frozenset({1, 2}), "experimental", "does not show every gated row"),
        (1, frozenset({2, 3}), "experimental", "does not show every gated row"),
        (4, frozenset({1, 2, 3}), "experimental", "no study measures more than three"),
        (2, frozenset({1, 2, 3}), "supported", "continuous-time-survival study measures"),
    ],
)
def test_the_causes_condition_of_the_support_rule(
    monkeypatch: pytest.MonkeyPatch,
    causes: int,
    measured: frozenset[int],
    label: str,
    phrase: str,
) -> None:
    monkeypatch.setattr(result_module, "MEASURED_CAUSES", measured)
    support = continuous_survival_support(
        scale="difference",
        event_share=0.6,
        n=1_000,
        causes=causes,
        cross_fitted=True,
        default_nuisances=True,
        settings=MEASURED_DEFAULTS,
        variance_rel_sd=0.05,
    )
    assert support.label == label
    assert phrase in support.reason
    if label == "experimental":
        assert support.evidence == ("X6",)
    limit = support.limits[-1]
    if causes == 2:
        assert not any("continuous-survival-causes" in limit for limit in support.limits)
    elif causes > 3:
        assert limit.startswith("No study measures four causes")
    else:
        # The limit names the study and the law of this number of causes only, and the gated
        # times: the first quartile is published without a verdict.
        law = "cause-2 hazard removed" if causes == 1 else "third hazard"
        other = "third hazard" if causes == 1 else "cause-2 hazard removed"
        assert limit.startswith("The continuous-survival-causes study measures")
        assert law in limit and other not in limit
        assert "gated at the median and upper quartile" in limit
        assert ("does not show every gated row" in limit) is (causes not in measured)
    if label == "supported" and causes != 2:
        assert support.evidence == ("continuous-survival-causes", "continuous-survival-variance")


def test_the_shipped_measured_causes_are_one_to_three() -> None:
    # The design test ties this set to the committed record; with it, a one- or three-cause fit
    # at the study's size reads supported and four causes do not.
    assert frozenset({1, 2, 3}) == result_module.MEASURED_CAUSES
    common: dict[str, Any] = {
        "scale": "level",
        "event_share": 0.6,
        "cross_fitted": True,
        "default_nuisances": True,
        "settings": MEASURED_DEFAULTS,
        "variance_rel_sd": 0.05,
    }
    for causes in (1, 3):
        assert continuous_survival_support(n=1_000, causes=causes, **common).label == "supported"
        assert continuous_survival_support(n=999, causes=causes, **common).label == "experimental"
    assert continuous_survival_support(n=10_000, causes=4, **common).label == "experimental"


def test_a_measured_number_of_causes_keeps_its_own_size_floor_and_no_restricted_horizon() -> None:
    # Nonzero witnesses of the two conditions a measured one-cause fit still meets: the
    # causes study measures n = 1,000 (the 500 floor was measured with two causes), and the
    # functionals study measures restricted horizons with two causes only.
    common: dict[str, Any] = {
        "event_share": 0.6,
        "cross_fitted": True,
        "default_nuisances": True,
        "settings": MEASURED_DEFAULTS,
        "variance_rel_sd": 0.05,
    }
    small = continuous_survival_support(scale="level", n=600, causes=1, **common)
    assert small.label == "experimental" and "n = 1,000 only" in small.reason
    # Below 500 units the causes floor's sentence names the 500-unit floor too, so the variance
    # study's size sentence is not printed beside it.
    tiny = continuous_survival_support(scale="level", n=400, causes=3, **common)
    assert tiny.label == "experimental" and "n = 1,000 only" in tiny.reason
    assert "down to n = 500" not in tiny.reason
    assert (
        "down to n = 500"
        in continuous_survival_support(scale="level", n=400, causes=2, **common).reason
    )
    two = continuous_survival_support(scale="level", n=600, causes=2, **common)
    assert two.label == "supported"
    restricted = continuous_survival_support(
        scale="level", n=1_000, causes=3, restricted=True, restricted_horizons=1, **common
    )
    assert restricted.label == "experimental"
    assert "restricted horizons with two causes only" in restricted.reason
