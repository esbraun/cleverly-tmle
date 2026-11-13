"""Targeting curves: TOC, cumulative gain, AUTOC and Qini of a priority score (X92 PR-5).

The checks of the X92 PR-5 plan, section 9. Each exact check carries the witness that keeps it
from passing vacuously, or a deliberate mutation of the code that must make it fail:

- an exact finite-support law whose empirical law is the law, with exact nuisances, against
  hand-computed truths, and four mutations of the rank rule;
- the AUTOC and Qini weights against the double sums of Yadlowsky et al. (2.9), and the
  tie-averaged weights against all 720 orderings of a six-row example;
- the TMLE identity (the substitution estimate equals the targeted-score form) with a
  misspecified outcome regression, and two mutations of the fluctuation;
- the basis reduction, with the unreduced fluctuation as its mutation;
- the range of a binary outcome, with the one-step score as its witness;
- observation weights as duplication;
- the half-sample bootstrap over all 70 half-samples of eight rows, its clusters and seeds,
  and the fixed-cut mutation;
- the band, the paired comparison and its unpaired mutation;
- the refusals, the support labels, the summary, the frame and the determinism.
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly import _pooled_targeting as pt
from cleverly import targeting_curves as tc
from cleverly.bases import Linear, Modifiers
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.inference.bootstrap import studentized_critical_value
from cleverly.study import (
    SCORE_IN_SAMPLE_REFUSAL,
    SCORE_INPUTS_REFUSAL,
    SCORE_KIND_REFUSAL,
)
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_oat import SaturatedCategorical

#: Below this absolute size a difference counts as zero in the exactness checks.
EXACT_FLOOR = 1e-12
#: A mutation must move some reported number by more than this.
MUTATION_FLOOR = 1e-6


def _quiet(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)

    return wrapped


# ----------------------------------------------------------------- the exact finite-support law

PER_CELL = 40
CELLS = [(w1, w2) for w1 in range(5) for w2 in range(2)]


def _tau(w1: Any, w2: Any) -> Any:
    return 1.0 + 0.6 * w1 - 0.5 * w2 + 0.3 * (np.asarray(w1) == 2)


def _law_frame(weights: bool = False) -> pd.DataFrame:
    """Forty rows per ``(W1, W2)`` cell, treated share ``g(W)`` exactly, and outcome noise
    that sums to zero within every ``(A, W)`` cell, so the saturated fits are the oracle."""
    rows = []
    for w1, w2 in CELLS:
        g = (1 + (w1 + w2) % 3) / 4
        treated = int(g * PER_CELL)
        for i in range(PER_CELL):
            a = 1 if i < treated else 0
            within = i if a else i - treated
            noise = 0.25 if within % 2 == 0 else -0.25
            y = 0.5 * w2 + 0.2 * w1 + a * float(_tau(w1, w2)) + noise
            rows.append((y, a, w1, w2, 1 + (i % 3) if weights else 1))
    return pd.DataFrame(rows, columns=["Y", "A", "W1", "W2", "w"])


def _exact_fit(frame: pd.DataFrame, weights: str | None = None) -> tuple[Any, Any]:
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], weights=weights
    )
    fit = (
        TMLE(
            outcome_learner=CellMeans(),
            treatment_learner=SaturatedCategorical(),
            cross_fit=False,
            estimands=("ate",),
            reference=0.0,
            q_bounds=(-3.0, 8.0),
            random_state=0,
        )
        .fit(data)
        .single()
    )
    return fit, data


@functools.cache
def _exact() -> tuple[Any, Any, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = _law_frame()
        fit, data = _exact_fit(frame)
    return fit, data, frame["W1"].to_numpy(float)


def _exact_curve() -> Any:
    fit, data, score = _exact()
    return tc.targeting_fit(
        fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=20, random_state=0
    )


def _truths() -> dict[tuple[str, float], float]:
    """The law's quantities, written from the definitions with no rank weights.

    The score ``S = W1`` puts mass 1/5 on each of its values; ``tau(u)`` is the mean effect
    at rank fraction ``u`` (descending ``S``), constant on each fifth.  ``TOC(q)`` is
    ``(1/q) int_0^q tau(u) du - ATE``.  The areas are the ``n``-point forms of (2.8):
    ``AUTOC = (1/n) sum_j TOC(j/n)`` and ``Qini = (1/n) sum_j (j/n) TOC(j/n)``.
    """
    levels = [float(np.mean([_tau(w1, w2) for w2 in (0, 1)])) for w1 in (4, 3, 2, 1, 0)]
    ate = float(np.mean(levels))

    def integral(q: float) -> float:
        whole = math.floor(q / 0.2 + 1e-12)
        value = sum(levels[:whole]) * 0.2
        if whole < 5:
            value += levels[whole] * (q - 0.2 * whole)
        return value

    def toc(q: float) -> float:
        return integral(q) / q - ate

    n = PER_CELL * len(CELLS)
    out = {("ate", 1.0): ate}
    for q in tc.DEFAULT_GRID:
        out[("toc", q)] = toc(q)
        out[("gain", q)] = integral(q)
    previous = 0.0
    for edge in (0.2, 0.4, 0.6, 0.8, 1.0):
        out[("group_effect", edge)] = (integral(edge) - integral(previous)) / (edge - previous)
        previous = edge
    out[("autoc", 1.0)] = sum(toc(j / n) for j in range(1, n + 1)) / n
    out[("qini", 1.0)] = sum((j / n) * toc(j / n) for j in range(1, n + 1)) / n
    return out


def _reported(curve: Any) -> dict[tuple[str, float], float]:
    out = {}
    for quantity in ("toc", "gain", "group_effect", "autoc", "qini"):
        for row in curve.rows(quantity):
            out[(quantity, float(row["q"]))] = float(row["estimate"])
    return out


def _largest_gap(curve: Any) -> float:
    truths = _truths()
    return max(abs(value - truths[key]) for key, value in _reported(curve).items())


@_quiet
def test_the_exact_law_gives_the_hand_computed_truths() -> None:
    curve = _exact_curve()
    truths = _truths()
    reported = _reported(curve)
    assert len(reported) == 10 + 10 + 5 + 2
    for key, value in reported.items():
        assert value == pytest.approx(truths[key], rel=1e-12, abs=EXACT_FLOOR), key
    assert curve.ate == pytest.approx(truths[("ate", 1.0)], rel=1e-12)
    # Witnesses: the effect at the cut differs from the average at two grid points, a tied
    # block straddles q = 0.3, and both areas are far from 0.
    assert abs(truths[("toc", 0.1)]) > 0.5 and abs(truths[("toc", 0.5)]) > 0.1
    (row,) = [r for r in curve.rows("toc") if r["q"] == 0.3]
    assert row["tied_cut"] and 0.0 < row["tie_share"] < 1.0
    assert truths[("autoc", 1.0)] > 0.3 and truths[("qini", 1.0)] > 0.1


def _edges_read_tied(curve: Any) -> bool:
    """On the exact law, ``S = W1`` has five blocks of 80 rows, so the cuts at q = 0.2, 0.4,
    0.6 and 0.8 fall exactly on block edges, where the limit is not normal (D3)."""
    for row in curve.rows("toc"):
        if row["q"] in (0.2, 0.4, 0.6, 0.8):
            if not (row["tied_cut"] and math.isnan(row["tie_share"])):
                return False
            if "edge of a tied block" not in row["status_reason"]:
                return False
    return True


@_quiet
def test_a_cut_at_the_edge_of_a_tied_block_reads_tied() -> None:
    curve = _exact_curve()
    assert _edges_read_tied(curve)
    assert all(row["status"] == "experimental" for row in curve.rows("toc") if row["q"] == 0.4)


@_quiet
def test_reading_only_cuts_inside_a_block_misses_the_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: the strict-inside rule leaves every edge cut reading untied."""

    def strict(layout: Any, mass: Any, grid: Any) -> dict[float, Any]:
        order = layout.order
        m = tc._block_sums(mass[order], layout.starts)
        counts = np.diff(np.append(layout.starts, order.size)).astype(float)
        b = np.cumsum(m)
        out = {}
        for q in grid:
            share = tc._shares(m, b - m, b[-1:], q, "average")
            inside = np.flatnonzero((share > 0.0) & (share < 1.0))
            tied = bool(inside.size and counts[inside[0]] > 1)
            out[q] = tc._CutFact(
                float(np.sum(counts * share)),
                tied,
                float(share[inside[0]]) if tied else float("nan"),
            )
        return out

    monkeypatch.setattr(tc, "_cut_facts", strict)
    assert not _edges_read_tied(_exact_curve())


def _facts(score: np.ndarray, mass: np.ndarray | None = None) -> dict[float, Any]:
    weights = np.ones(score.size) if mass is None else mass
    return tc._cut_facts(tc._layout(score), weights, (0.3,))


@pytest.mark.parametrize(
    ("score", "where"),
    [
        # n = 999: 300 single rows above a tied block of 699; the cut 299.7 falls 0.3 of a row
        # inside the last single row, beside the atom.
        (np.r_[np.arange(300, 0, -1.0), np.zeros(699)], "above the block"),
        # n = 1,237: a tied block of 371 rows, then single rows; the cut 371.1 falls in the
        # first single row below the atom.
        (np.r_[np.full(371, 1_000.0), np.arange(866, 0, -1.0)], "below the block"),
    ],
)
def test_a_cut_inside_a_single_row_beside_a_tied_block_reads_tied(
    score: np.ndarray, where: str
) -> None:
    fact = _facts(score)[0.3]
    assert fact.tied and math.isnan(fact.share) and fact.code == "tied_edge", where


def test_a_cut_inside_a_single_row_away_from_ties_reads_untied() -> None:
    """The witness: the neighbour rule does not flag a cut among single rows."""
    score = np.r_[np.arange(999, 0, -1.0)]
    assert not _facts(score)[0.3].tied


def test_a_zero_weight_block_above_the_cut_does_not_hide_its_edge() -> None:
    """A zero-mass row at the top shifts a count of the blocks above the cut by one.

    The 20 weighted single rows above the cut at q = 0.2 sit on the edge of an 80-row block;
    counting the zero-mass row's block would put the edge between two single rows instead.
    """
    score = np.r_[100.0, np.arange(50.0, 30.0, -1.0), np.zeros(80)]
    mass = np.r_[0.0, np.ones(100)]
    mass = mass / mass.mean()
    fact = tc._cut_facts(tc._layout(score), mass, (0.2,))[0.2]
    assert fact.tied and fact.code == "tied_edge"


@pytest.mark.parametrize("mutation", ["ascending", "at_or_above", "h_k", "uncentred_qini"])
@_quiet
def test_each_rank_rule_mutation_moves_a_truth(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mutation == "ascending":
        monkeypatch.setattr(tc, "_order", lambda s: np.argsort(np.asarray(s), kind="stable"))
    elif mutation == "at_or_above":
        shares = tc._shares
        monkeypatch.setattr(
            tc, "_shares", lambda m, a, total, q, rule: shares(m, a, total, q, "at_or_above")
        )
    elif mutation == "h_k":
        weight = tc._autoc_block_weight
        monkeypatch.setattr(
            tc, "_autoc_block_weight", lambda a, b, m, total: weight(a + 1.0, b + 1.0, m, total)
        )
    else:
        weight = tc._qini_block_weight
        monkeypatch.setattr(
            tc, "_qini_block_weight", lambda a, b, m, total: weight(a, b, m, total) + 0.5
        )
    assert _largest_gap(_exact_curve()) > MUTATION_FLOOR


# ----------------------------------------------------------------- the weight identities


@pytest.mark.parametrize("n", [1, 2, 7, 50])
def test_the_area_weights_are_the_double_sums(n: int) -> None:
    weights = tc.block_weights(np.ones(n), tc.DEFAULT_GRID)
    k = np.arange(1, n + 1)
    autoc = np.array([sum(1.0 / j for j in range(i, n + 1)) - 1.0 for i in k])
    qini = (n + 1 - 2 * k) / (2.0 * n)
    np.testing.assert_allclose(weights[("autoc", 1.0)], autoc, rtol=1e-12, atol=EXACT_FLOOR)
    np.testing.assert_allclose(weights[("qini", 1.0)], qini, rtol=1e-12, atol=EXACT_FLOOR)


def test_tied_weights_average_every_ordering_of_six_rows() -> None:
    score = np.array([3.0, 2.0, 3.0, 2.0, 1.0, 2.0])
    grid = (0.3, 0.5, 1.0)
    layout = tc._layout(score)
    m = tc._block_sums(np.ones(6), layout.starts)
    tied = tc.block_weights(m, grid)
    keys = [("toc", 0.3), ("toc", 0.5), ("gain", 0.5), ("autoc", 1.0), ("qini", 1.0)]
    totals = {key: np.zeros(6) for key in keys}
    for tiebreak in itertools.permutations(range(6)):
        order = np.lexsort((np.array(tiebreak), -score))
        untied = tc.block_weights(np.ones(6), grid)
        for key in keys:
            totals[key][order] += untied[key]
    for key in keys:
        np.testing.assert_allclose(
            tied[key][layout.block_of_row], totals[key] / 720.0, rtol=1e-12, atol=EXACT_FLOOR
        )


def test_a_weighted_row_takes_the_mean_weight_of_its_duplicates() -> None:
    """Weights as duplication at the rank level, in the masses' own units.

    Mutation control: positions counted by rows instead of by mass move the weights.
    """
    masses = np.array([2.0, 1.0, 3.0])
    weighted = tc.block_weights(masses, (0.5, 1.0))
    expanded = tc.block_weights(np.ones(6), (0.5, 1.0))
    owner = np.repeat(np.arange(3), masses.astype(int))
    for key in [("toc", 0.5), ("autoc", 1.0), ("qini", 1.0)]:
        means = np.array([expanded[key][owner == i].mean() for i in range(3)])
        np.testing.assert_allclose(weighted[key], means, rtol=1e-12)
    by_rows = tc.block_weights(np.ones(3), (0.5, 1.0))
    assert np.max(np.abs(by_rows[("autoc", 1.0)] - weighted[("autoc", 1.0)])) > MUTATION_FLOOR


@_quiet
def test_integer_weights_give_the_duplicated_frame() -> None:
    """Integer weights against the frame with each row repeated, on the exact law.

    TOC, gain, group effects and Qini are ratios of masses and match to rounding. AUTOC's
    harmonic weights read the masses normalised to mean one, which is the duplicated frame
    only in raw-mass units; the rank-level test above checks that case.
    """
    frame = _law_frame(weights=True)
    fit, data = _exact_fit(frame, weights="w")
    score = frame["W1"].to_numpy(float)
    weighted = tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=5)
    repeated = frame.loc[frame.index.repeat(frame["w"])].reset_index(drop=True)
    fit_r, data_r = _exact_fit(repeated)
    duplicated = tc.targeting_fit(
        fit_r, data_r, {"S": repeated["W1"].to_numpy(float)}, pair=(1.0, 0.0), n_half_samples=5
    )
    left, right = _reported(weighted), _reported(duplicated)
    for key, value in left.items():
        if key[0] != "autoc":
            assert value == pytest.approx(right[key], rel=1e-12, abs=EXACT_FLOOR), key
    assert "observation weights" in weighted.status_reason


# ----------------------------------------------------------------- the TMLE identity


@functools.cache
def _misspecified() -> tuple[Any, Any, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 1000
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, expit(1.5 * w[:, 0]))
    y = w[:, 1] + a * (1 + w[:, 0]) ** 2 + 0.3 * rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(),
                estimands=("ate",),
                reference=0.0,
                q_bounds=(-6.0, 8.0),
                g_bounds=(0.25, 0.75),
                tol=1e-10,
                random_state=0,
            )
            .fit(data)
            .single()
        )
    return fit, data, w


def _identity_gap(curve: Any, **mutation: Any) -> float:
    points = tc.control_values(curve, n_half_samples=2, random_state=0, **mutation)
    rows = curve.rows("toc") + curve.rows("gain") + curve.rows("autoc") + curve.rows("qini")
    return max(
        abs(row["estimate"] - points[curve.score][(row["quantity"], row["q"])][0]) for row in rows
    )


def _continuous_curve() -> Any:
    fit, data, w = _misspecified()
    score = w[:, 0] + 0.1 * w[:, 1]
    return tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=20)


@_quiet
def test_each_estimate_equals_its_targeted_score_form() -> None:
    curve = _continuous_curve()
    assert _identity_gap(curve) < 1e-8
    # Witness: the initial fit is misspecified, so targeting moved it.
    assert _identity_gap(curve, scores="one_step") > 1e-4
    toc = curve.rows("toc")[2]
    gain = curve.rows("gain")[2]
    assert toc["estimate"] == pytest.approx(gain["estimate"] / toc["q"] - curve.ate, rel=1e-12)
    assert curve.span["score_norm"] < 1e-8


@_quiet
def test_dropping_the_autoc_column_breaks_its_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = tc._fluctuation_keys
    monkeypatch.setattr(tc, "_fluctuation_keys", lambda grid: keys(grid)[:-2] + keys(grid)[-1:])
    curve = _continuous_curve()
    point = tc.control_values(curve, n_half_samples=2)[curve.score][("autoc", 1.0)][0]
    assert abs(curve.area("autoc")["estimate"] - point) > MUTATION_FLOOR


@_quiet
def test_an_untruncated_mechanism_in_the_clever_covariate_breaks_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    propensity = pt.clever_propensity
    monkeypatch.setattr(
        pt, "clever_propensity", lambda nuisance, bounds: propensity(nuisance, (1e-6, 1 - 1e-6))
    )
    assert _identity_gap(_continuous_curve()) > MUTATION_FLOOR


@_quiet
def test_the_reduced_basis_solves_every_column_of_a_discrete_score() -> None:
    fit, data, w = _misspecified()
    score = np.floor(2.0 * (w[:, 0] + 1.0))
    curve = tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=5)
    assert curve.span["columns"] == 12.0 and curve.span["rank"] == 4.0
    assert curve.span["out_of_span"] < tc.TARGETING_SPAN_RTOL
    assert _identity_gap(curve) < 1e-8


@_quiet
def test_skipping_the_reduction_fails_as_a_singular_fluctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, data, w = _misspecified()
    score = np.floor(2.0 * (w[:, 0] + 1.0))
    monkeypatch.setattr(tc, "span_basis", lambda columns: (columns, 0.0))
    with pytest.raises(DataError, match="singular_hessian"):
        tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=5)


@_quiet
def test_a_binary_outcome_keeps_its_targeted_means_in_range() -> None:
    rng = np.random.default_rng(4)
    n = 1000
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, expit(-4.0 * w[:, 0]))
    p = np.where(a == 1, expit(-2 + 4 * (w[:, 0] > 0.6)), expit(-1 + w[:, 1]))
    frame = pd.DataFrame({"Y": rng.binomial(1, p), "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
    fit = (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            estimands=("ate",),
            reference=0.0,
            g_bounds=(0.01, 0.99),
            random_state=0,
        )
        .fit(data)
        .single()
    )
    curve = tc.targeting_fit(fit, data, {"S": w[:, 0]}, pair=(1.0, 0.0), n_half_samples=5)
    top = curve.rows("gain")[0]["estimate"] / 0.1
    assert -1.0 <= top <= 1.0
    assert abs(curve.rows("toc")[0]["estimate"]) <= 2.0 * (1.0 - 0.1)
    one_step = tc.control_values(curve, n_half_samples=2, scores="one_step")["S"]
    assert abs(one_step[("gain", 0.1)][0] / 0.1) > 1.0


# ----------------------------------------------------------------- the half-sample bootstrap


def _evaluation(
    score: np.ndarray, gamma: np.ndarray, units: np.ndarray | None = None, **scores: np.ndarray
) -> Any:
    n = score.size
    return tc.TargetingEvaluation(
        scores={"S": score, **scores},
        targeted=gamma,
        one_step=gamma,
        mass=np.ones(n),
        units=np.arange(n) if units is None else units,
        grid=tc.DEFAULT_GRID,
        edges=(0.5, 1.0),
    )


def test_all_seventy_half_samples_give_the_finite_population_variance() -> None:
    rng = np.random.default_rng(0)
    gamma = rng.normal(size=8)
    evaluation = _evaluation(rng.normal(size=8), gamma)
    masks = np.zeros((70, 8))
    for row, chosen in enumerate(itertools.combinations(range(8), 4)):
        masks[row, list(chosen)] = 1.0
    draws = tc.half_sample_draws(evaluation, n_half_samples=70, random_state=0, masks=masks)
    ate = draws["S"][("ate", 1.0)]
    variance = np.var(gamma) / 4.0 * (8 - 4) / (8 - 1)
    assert np.var(ate) == pytest.approx(variance, rel=1e-12)


def test_clustered_draws_keep_half_the_clusters_whole() -> None:
    units = np.repeat(np.arange(5), 3)
    inverse, mask = tc._masks(units, 50, np.random.default_rng(1))
    assert np.all(mask.sum(axis=1) == 2)
    rows = mask[:, inverse]
    for cluster in range(5):
        block = rows[:, units == cluster]
        assert np.all(block == block[:, :1])


def test_seeded_draws_reproduce_and_differ_across_seeds() -> None:
    rng = np.random.default_rng(2)
    evaluation = _evaluation(rng.normal(size=60), rng.normal(size=60))
    one = tc.half_sample_draws(evaluation, n_half_samples=30, random_state=7)
    two = tc.half_sample_draws(evaluation, n_half_samples=30, random_state=7)
    three = tc.half_sample_draws(evaluation, n_half_samples=30, random_state=8)
    np.testing.assert_array_equal(one["S"][("toc", 0.5)], two["S"][("toc", 0.5)])
    assert not np.array_equal(one["S"][("toc", 0.5)], three["S"][("toc", 0.5)])


def test_the_fast_draws_equal_the_rank_weights_recomputed_per_draw() -> None:
    rng = np.random.default_rng(5)
    n = 40
    score = np.round(rng.normal(size=n), 1)  # ties
    gamma = rng.normal(size=n)
    evaluation = _evaluation(score, gamma)
    inverse, masks = tc._masks(evaluation.units, 25, np.random.default_rng(9))
    fast = tc.half_sample_draws(evaluation, n_half_samples=25, random_state=0, masks=masks)["S"]
    for draw in range(25):
        kept = masks[draw][inverse] > 0
        reference = tc._point_values(
            tc._layout(score[kept]),
            gamma[kept],
            np.ones(int(kept.sum())),
            evaluation.grid,
            evaluation.edges,
        )
        for key, value in reference.items():
            assert fast[key][draw] == pytest.approx(value, rel=1e-10, abs=EXACT_FLOOR), key


def test_a_fixed_full_sample_cut_changes_the_spread() -> None:
    """Mutation control: the cut recomputed per draw carries the variance term at the cut."""
    n = 400
    score = np.arange(n, dtype=float)
    gamma = np.where((score >= 180) & (score < 220), 10.0, 0.0)
    evaluation = _evaluation(score, gamma)
    shipped = tc.half_sample_draws(evaluation, n_half_samples=400, random_state=0)
    fixed = tc.half_sample_draws(evaluation, n_half_samples=400, random_state=0, cut="fixed")
    se = np.std(shipped["S"][("toc", 0.5)], ddof=1)
    se_fixed = np.std(fixed["S"][("toc", 0.5)], ddof=1)
    assert abs(se_fixed / se - 1.0) > 0.1


# ----------------------------------------------------------------- band and comparison


@functools.cache
def _paired_curve() -> Any:
    fit, data, w = _misspecified()
    scores = {"S": w[:, 0] + 0.1 * w[:, 1], "R": w[:, 0] + 0.5 * w[:, 1]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tc.targeting_fit(
            fit, data, scores, pair=(1.0, 0.0), n_half_samples=200, random_state=3
        )


def test_the_band_is_the_sup_t_band_of_the_stored_draws() -> None:
    curve = _paired_curve()
    draws = tc.half_sample_draws(curve.evaluation, n_half_samples=200, random_state=3)["S"]
    below = [q for q in curve.grid if q < 1.0]
    rows = curve.rows("toc")
    matrix = np.column_stack([draws[("toc", q)] for q in below])
    se = np.std(matrix, axis=0, ddof=1)
    estimates = np.array([row["estimate"] for row in rows[:-1]])
    critical, _ = studentized_critical_value(
        estimates, matrix, np.broadcast_to(se, matrix.shape), curve.alpha
    )
    assert curve.critical_value == pytest.approx(critical, rel=1e-12)
    for row, spread in zip(rows[:-1], se, strict=True):
        assert row["std_error"] == pytest.approx(spread, rel=1e-12)
        assert row["band_lower"] <= row["lower"] and row["upper"] <= row["band_upper"]
    assert math.isnan(rows[-1]["band_lower"]) and math.isnan(rows[-1]["std_error"])
    assert rows[-1]["estimate"] == pytest.approx(0.0, abs=EXACT_FLOOR)


@_quiet
def test_no_heterogeneity_withholds_the_band() -> None:
    """A constant effect gives every TOC draw exactly 0, so the band has no spread to scale."""
    frame = _law_frame()
    frame["Y"] = frame["Y"] - frame["A"] * _tau(frame["W1"], frame["W2"]) + frame["A"]
    fit, data = _exact_fit(frame)
    curve = tc.targeting_fit(
        fit, data, {"S": frame["W1"].to_numpy(float)}, pair=(1.0, 0.0), n_half_samples=20
    )
    assert math.isnan(curve.critical_value)
    assert all(math.isnan(row["band_lower"]) for row in curve.rows("toc"))


def test_the_difference_is_paired_on_the_same_draws() -> None:
    curve = _paired_curve()
    draws = tc.half_sample_draws(curve.evaluation, n_half_samples=200, random_state=3)
    for quantity, q in [("toc", 0.2), ("autoc", 1.0), ("qini", 1.0)]:
        (row,) = [r for r in curve.rows(quantity, "S - R") if r["q"] == q]
        difference = draws["S"][(quantity, q)] - draws["R"][(quantity, q)]
        assert row["std_error"] == pytest.approx(np.std(difference, ddof=1), rel=1e-12)


def test_independent_draws_change_the_difference_spread() -> None:
    """Mutation control: the second score drawn from its own stream widens the difference."""
    curve = _paired_curve()
    evaluation = curve.evaluation
    paired = tc.half_sample_draws(evaluation, n_half_samples=200, random_state=3)
    unpaired = tc.half_sample_draws(evaluation, n_half_samples=200, random_state=3, paired=False)
    key = ("autoc", 1.0)
    se_paired = np.std(paired["S"][key] - paired["R"][key], ddof=1)
    se_unpaired = np.std(unpaired["S"][key] - unpaired["R"][key], ddof=1)
    assert se_unpaired > 2.0 * se_paired


# ----------------------------------------------------------------- refusals and labels


def _study_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1, w2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = w2 + a * (1 + w1) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "S": w1, "R": w1 + 0.5 * w2})


def _effect(frame: pd.DataFrame | None = None, model: Any = None) -> Any:
    study = CausalStudy(
        _study_frame() if frame is None else frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
    )
    return study.identify(ConditionalContrast(Modifiers(["W1"]), model=model))


LEARNERS: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(),
    "outcome_by_arm": True,
    "q_bounds": (-5.0, 6.0),
}


def _route_one(**kwargs: Any) -> Any:
    arguments = {
        "computed_from": ["W1"],
        "score_kind": "external",
        "n_half_samples": 20,
        "random_state": 0,
        **LEARNERS,
        **kwargs,
    }
    return _effect().targeting_curve("S", **arguments)


@_quiet
def test_route_one_refuses_what_it_cannot_read() -> None:
    from cleverly import ATE

    study = CausalStudy(
        _study_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
    )
    with pytest.raises(CapabilityError, match="ConditionalContrast"):
        study.identify(ATE()).targeting_curve("S", computed_from=["W1"], score_kind="external")
    with pytest.raises(CapabilityError) as caught:
        _route_one(repeats=2)
    assert caught.value.refusal is tc.TARGETING_REPEATS_REFUSAL
    with pytest.raises(CapabilityError) as caught:
        _route_one(score_kind=None)
    assert caught.value.refusal is SCORE_KIND_REFUSAL
    with pytest.raises(DataError, match="same column"):
        _route_one(compare="S", compare_computed_from=["W1"])
    with pytest.raises(CapabilityError) as missing_inputs:
        _route_one(compare="R")
    with pytest.raises(CapabilityError) as plain:
        _route_one(computed_from=None)
    assert missing_inputs.value.refusal is plain.value.refusal is SCORE_INPUTS_REFUSAL


@_quiet
def test_route_one_refuses_an_in_sample_compare_column() -> None:
    effect = _effect(model=Linear(["W1"]))
    fit = effect.estimate(**LEARNERS, random_state=0)
    frame = _study_frame()
    frame["C"] = fit.effect_curve().predict(frame[["W1"]])
    effect = _effect(frame, model=Linear(["W1"]))
    effect.estimate(**LEARNERS, random_state=0).effect_curve()
    with pytest.raises(CapabilityError) as caught:
        effect.targeting_curve(
            "S",
            computed_from=["W1"],
            score_kind="external",
            compare="C",
            compare_computed_from=["W1"],
            n_half_samples=5,
            **LEARNERS,
        )
    assert caught.value.refusal is SCORE_IN_SAMPLE_REFUSAL


@pytest.mark.parametrize("split", [1, 2, 0.0, 1.0, "cross_fit", True])
def test_route_two_refuses_a_fold_average(split: Any) -> None:
    curve = _split_curve_fit()
    with pytest.raises(CapabilityError) as caught:
        curve.diagnostics.targeting_curve(split=split)
    assert caught.value.refusal is tc.TARGETING_SPLIT_REFUSAL
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION


@functools.cache
def _split_curve_fit() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effect = _effect(_study_frame(1000), model=Linear(["W1"]))
        return effect.estimate(**LEARNERS, random_state=0).effect_curve()


@_quiet
def test_route_two_scores_held_out_rows_and_is_deterministic() -> None:
    curve = _split_curve_fit()
    one = curve.diagnostics.targeting_curve(n_half_samples=20, random_state=4)
    two = curve.diagnostics.targeting_curve(n_half_samples=20, random_state=4)
    assert one.route == "split" and one.n_train == 500 and one.n_eval == 500
    pd.testing.assert_frame_equal(one.to_frame(), two.to_frame())
    assert one.score == curve.prediction_column
    # The score is the training refit's prediction, which the curve keeps; it is not the
    # curve fitted on every row.
    refit = one.score_curve
    assert refit is not None and refit is not curve
    probe = pd.DataFrame({"W1": np.linspace(-1.0, 1.0, 7)})
    assert not np.allclose(refit.predict(probe), curve.predict(probe))
    with pytest.raises(CapabilityError, match="engine"):
        from dataclasses import replace

        replace(curve, identified_effect=None).diagnostics.targeting_curve(
            compare="R", compare_computed_from=["W1", "W2"], score_kind="external"
        )


@_quiet
def test_route_two_compares_a_study_column_on_the_same_rows() -> None:
    curve = _split_curve_fit()
    paired = curve.diagnostics.targeting_curve(
        compare="R",
        compare_computed_from=["W1", "W2"],
        score_kind="external",
        n_half_samples=20,
        random_state=4,
    )
    assert paired.compare == "R"
    assert paired.rows("autoc", f"{paired.score} - R")


@_quiet
def test_a_log_ratio_curve_is_a_score_on_route_two() -> None:
    rng = np.random.default_rng(1)
    n = 600
    w = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    frame = pd.DataFrame(
        {"Y": rng.binomial(1, expit(-0.5 + a * w)), "A": a, "W1": w, "W2": rng.normal(size=n)}
    )
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    effect = study.identify(
        ConditionalContrast(Modifiers(["W1"]), scale="log_ratio", model=Linear(["W1"]))
    )
    fit = effect.estimate(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        random_state=0,
    )
    targeting = fit.effect_curve().diagnostics.targeting_curve(n_half_samples=10, random_state=0)
    assert targeting.route == "split" and np.isfinite(targeting.area("autoc")["estimate"])


@pytest.mark.parametrize(
    ("grid", "groups", "match"),
    [
        ((0.0, 0.5), 5, "lie in"),
        ((0.5, 1.2), 5, "lie in"),
        ((0.5, 0.2), 5, "increasing"),
        ((0.25, 0.5), (0.3,), "grid"),
        (None, True, "groups"),
    ],
)
def test_the_grid_and_group_rules(grid: Any, groups: Any, match: str) -> None:
    fit, data, w = _misspecified()
    with pytest.raises(DataError, match=match):
        tc.targeting_fit(
            fit, data, {"S": w[:, 0]}, pair=(1.0, 0.0), grid=grid, groups=groups, n_half_samples=2
        )


@_quiet
def test_the_fluctuation_settings_it_cannot_carry_are_refused() -> None:
    fit, data, w = _misspecified()
    from dataclasses import replace

    spec = fit.config.targeting_spec
    for change, words in [
        ({"target_weights": True}, "target_weights"),
        ({"stopping": "scaled_var"}, "stopping"),
    ]:
        config = replace(fit.config, targeting_spec=replace(spec, **change))
        mutated = replace(fit, config=config)
        with pytest.raises(CapabilityError, match=words) as caught:
            tc.targeting_fit(mutated, data, {"S": w[:, 0]}, pair=(1.0, 0.0))
        assert caught.value.refusal.roadmap == "X92"


class _NeverFitted(LinearRegression):
    """A learner that fails the test if anything fits it."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner was fitted before the refusal")


@pytest.mark.parametrize(
    ("change", "words"),
    [({"target_weights": True}, "target_weights"), ({"stopping": "scaled_var"}, "stopping")],
)
def test_route_one_refuses_its_fluctuation_settings_before_any_learner(
    change: dict[str, Any], words: str
) -> None:
    with pytest.raises(CapabilityError, match=words):
        _route_one(**{**change, "outcome_learner": _NeverFitted()})


@pytest.mark.parametrize(
    ("change", "words"),
    [({"target_weights": True}, "target_weights"), ({"stopping": "scaled_var"}, "stopping")],
)
def test_route_two_refuses_its_fluctuation_settings_before_any_learner(
    change: dict[str, Any], words: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    curve = _split_curve_fit()
    spec = TMLE.targeting_spec
    monkeypatch.setattr(TMLE, "targeting_spec", lambda self: replace(spec(self), **change))

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner was fitted before the refusal")

    monkeypatch.setattr(LinearRegression, "fit", never)
    monkeypatch.setattr(LogisticRegression, "fit", never)
    with pytest.raises(CapabilityError, match=words):
        curve.diagnostics.targeting_curve(n_half_samples=10, random_state=0)


_DROPPED_SETTINGS = [
    ({"targeting_scheme": "fold"}, "TARGETING_SCHEME_REFUSAL"),
    ({"cv_evaluation": True}, "TARGETING_SCHEME_REFUSAL"),
    ({"n_bootstrap": 20}, "TARGETING_BOOTSTRAP_REFUSAL"),
]

#: The roadmap item each dropped setting's refusal names: the fold settings are an
#: implementation gap tracked by X92, the refit bootstrap an inference gap that names none.
_DROPPED_ROADMAP = {"TARGETING_SCHEME_REFUSAL": "X92", "TARGETING_BOOTSTRAP_REFUSAL": None}


@pytest.mark.parametrize(("change", "refusal"), _DROPPED_SETTINGS)
def test_route_one_refuses_settings_the_curve_would_drop_before_any_learner(
    change: dict[str, Any], refusal: str
) -> None:
    with pytest.raises(CapabilityError) as caught:
        _route_one(**{**change, "outcome_learner": _NeverFitted()})
    assert caught.value.refusal is getattr(tc, refusal)
    assert caught.value.refusal.roadmap == _DROPPED_ROADMAP[refusal]


@pytest.mark.parametrize(("change", "refusal"), _DROPPED_SETTINGS)
def test_route_two_refuses_settings_the_curve_would_drop_before_any_learner(
    change: dict[str, Any], refusal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cleverly import _curve_assessment as ca

    curve = _split_curve_fit()
    kwargs = ca._ate_kwargs
    monkeypatch.setattr(ca, "_ate_kwargs", lambda *a, **k: {**kwargs(*a, **k), **change})

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner was fitted before the refusal")

    monkeypatch.setattr(LinearRegression, "fit", never)
    monkeypatch.setattr(LogisticRegression, "fit", never)
    with pytest.raises(CapabilityError) as caught:
        curve.diagnostics.targeting_curve(n_half_samples=10, random_state=0)
    assert caught.value.refusal is getattr(tc, refusal)


@_quiet
def test_an_outcome_on_a_tiny_scale_keeps_its_intervals() -> None:
    """The zero-spread guard is relative: the same law with Y in units of 1e-13 keeps every
    interval, at 1e-13 times the SEs of the unscaled fit (an absolute 1e-12 would withhold
    them all)."""
    frame = _study_frame()
    small = frame.assign(Y=frame["Y"] * 1e-13)
    arguments = {"computed_from": ["W1"], "score_kind": "external", "n_half_samples": 20}
    plain = _effect(frame).targeting_curve("S", **arguments, random_state=0, **LEARNERS)
    learners = {**LEARNERS, "q_bounds": (-5e-13, 6e-13)}
    scaled = _effect(small).targeting_curve("S", **arguments, random_state=0, **learners)
    for one, other in zip(plain.rows("toc"), scaled.rows("toc"), strict=True):
        if one["q"] < 1.0:
            assert other["std_error"] < 1e-12
            assert other["std_error"] == pytest.approx(1e-13 * one["std_error"], rel=1e-4)
    assert np.isfinite(scaled.critical_value)


@_quiet
def test_a_constant_score_withholds_the_intervals_its_rounding_would_give(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every TOC of a score that ties every row is 0, and its draws differ only by rounding.

    The mutation control removes the zero-spread guard: the same rows then carry finite
    intervals of width near 1e-16.
    """
    fit, data, _ = _misspecified()
    score = np.zeros(data.n)
    curve = tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=20)
    rows = [row for row in curve.rows("toc") if row["q"] < 1.0]
    assert all(math.isnan(row["std_error"]) for row in rows)
    assert all("do not vary" in row["status_reason"] for row in rows)
    assert math.isnan(curve.critical_value)
    monkeypatch.setattr(tc, "_ZERO_SPREAD", 0.0)
    mutated = tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=20)
    widths = [row["upper"] - row["lower"] for row in mutated.rows("toc") if row["q"] < 1.0]
    assert any(math.isfinite(width) and width < 1e-10 for width in widths)


def _clustered_curve(n_clusters: int) -> Any:
    frame = _study_frame(800, seed=4)
    frame["C"] = np.arange(800) % n_clusters
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), cluster="C"),
    )
    effect = study.identify(ConditionalContrast(Modifiers(["W1"])))
    return effect.targeting_curve(
        "S",
        computed_from=["W1"],
        score_kind="external",
        n_half_samples=20,
        random_state=0,
        **LEARNERS,
    )


@_quiet
def test_fewer_than_forty_clusters_withhold_every_interval_and_the_band() -> None:
    few = _clustered_curve(20)
    estimates = few.to_frame()
    shown = estimates[~((estimates["quantity"] == "toc") & (estimates["q"] == 1.0))]
    assert shown["std_error"].isna().all()
    assert shown["status_reason"].str.contains("fewer than 40 clusters").all()
    assert math.isnan(few.critical_value)
    # The witness: at 50 clusters the same fit carries its intervals and band.
    enough = _clustered_curve(50)
    assert np.isfinite(enough.area("autoc")["std_error"])
    assert np.isfinite(enough.critical_value)


@_quiet
def test_a_known_mechanism_is_read_from_the_data_and_divides_the_clever_covariate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _study_frame(1200, seed=5)
    known = 0.3 + 0.4 * (frame["W2"] > 0)
    frame["A"] = np.random.default_rng(6).binomial(1, known)
    frame["p1"], frame["p0"] = known, 1.0 - known
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2"),
            treatment_probabilities={0: "p0", 1: "p1"},
        ),
    )
    effect = study.identify(ConditionalContrast(Modifiers(["W1"])))
    seen: list[np.ndarray] = []
    clever = pt.clever_propensity

    def spy(nuisance: Any, g_bounds: tuple[float, float]) -> Any:
        seen.append(clever(nuisance, g_bounds))
        return seen[-1]

    monkeypatch.setattr(pt, "clever_propensity", spy)
    status = tc.targeting_status
    declared: list[bool] = []

    def status_spy(**kwargs: Any) -> Any:
        declared.append(kwargs["known_mechanism"])
        return status(**kwargs)

    monkeypatch.setattr(tc, "targeting_status", status_spy)
    arguments = {
        "outcome_learner": LinearRegression(),
        "outcome_by_arm": True,
        "q_bounds": (-5.0, 6.0),
    }
    curve = effect.targeting_curve(
        "S",
        computed_from=["W1"],
        score_kind="external",
        n_half_samples=20,
        random_state=0,
        **arguments,
    )
    (g,) = seen
    assert g.shape == (1200, 2)
    assert np.allclose(g[:, 1], known.to_numpy(), atol=1e-12)
    assert np.allclose(g[:, 0], 1.0 - known.to_numpy(), atol=1e-12)
    # The configuration reads cell K's: a known mechanism, which the registered study measures.
    assert declared == [True]
    assert tc.targeting_status(
        route="external",
        outcome="continuous",
        known_mechanism=True,
        weighted=False,
        clustered=False,
        missing=False,
        arms=2,
        cross_fitted=True,
    ) == ("supported", ())
    # 1,200 rows put 120 in the top tenth, below the 200 the study calibrates at.
    first = curve.rows("toc")[0]
    assert first["status"] == "experimental" and "fewer than 200 rows" in first["status_reason"]


@_quiet
def test_few_top_rows_withhold_the_interval_and_band_without_refusing() -> None:
    rng = np.random.default_rng(0)
    frame = _study_frame(300)
    frame["S"] = frame["W1"] + 1e-3 * rng.normal(size=300)
    effect = _effect(frame)
    curve = effect.targeting_curve(
        "S", computed_from=["W1"], score_kind="external", n_half_samples=20, **LEARNERS
    )
    first, second = curve.rows("toc")[:2]
    assert first["rows_in_top"] == 30 and math.isnan(first["std_error"])
    assert "fewer than 50 rows" in first["status_reason"]
    assert math.isnan(curve.critical_value)
    assert second["rows_in_top"] == 60 and np.isfinite(second["std_error"])
    assert "fewer than 200 rows" in second["status_reason"]
    assert second["status"] == "experimental"
    text = str(curve.summary())
    assert "0.1*" in text


def test_an_area_on_a_continuous_score_with_atoms_is_experimental() -> None:
    fit, data, w = _misspecified()
    score = w[:, 0].copy()
    score[:30] = 0.0  # an atom of 3% of the rows, among many distinct values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = tc.targeting_fit(fit, data, {"S": score}, pair=(1.0, 0.0), n_half_samples=5)
    assert "Appendix A.2" in curve.area("autoc")["status_reason"]
    plain = _continuous_curve()
    assert "Appendix A.2" not in plain.area("autoc")["status_reason"]


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"weighted": True}, "weights"),
        ({"clustered": True}, "clusters"),
        ({"missing": True}, "missing"),
        ({"arms": 3}, "multi_arm"),
        ({"cross_fitted": False}, "not_cross_fitted"),
        ({"outcome": "binary", "route": "split"}, "unmeasured"),
        ({"known_mechanism": True, "outcome": "binary"}, "unmeasured"),
    ],
)
def test_each_departure_reads_experimental(changes: dict[str, Any], code: str) -> None:
    base = {
        "route": "external",
        "outcome": "continuous",
        "known_mechanism": False,
        "weighted": False,
        "clustered": False,
        "missing": False,
        "arms": 2,
        "cross_fitted": True,
    }
    assert tc.targeting_status(**base) == ("supported", ())
    status, codes = tc.targeting_status(**{**base, **changes})
    assert status == "experimental" and code in codes


_MEASURED_BASE: dict[str, Any] = {
    "known_mechanism": False,
    "weighted": False,
    "clustered": False,
    "missing": False,
    "arms": 2,
    "cross_fitted": True,
}


@pytest.mark.parametrize(
    ("route", "outcome", "known"),
    [
        ("external", "continuous", False),
        ("external", "binary", False),
        ("external", "continuous", True),
        ("split", "continuous", False),
    ],
)
def test_each_configuration_the_study_measures_reads_supported(
    route: str, outcome: str, known: bool
) -> None:
    """Cells H, B, K and S of the registered study, each with the composition it measured."""
    arguments = {**_MEASURED_BASE, "known_mechanism": known}
    assert tc.targeting_status(route=route, outcome=outcome, **arguments) == ("supported", ())


@pytest.mark.parametrize(
    ("route", "outcome", "known"),
    [
        ("split", "binary", False),
        ("split", "continuous", True),
        ("external", "binary", True),
        ("split", "binary", True),
    ],
)
def test_every_configuration_outside_the_study_reads_unmeasured(
    route: str, outcome: str, known: bool
) -> None:
    """The control: the four (route, outcome, mechanism) keys no cell measures."""
    arguments = {**_MEASURED_BASE, "known_mechanism": known}
    assert (route, outcome, "known" if known else "estimated") not in tc.TARGETING_MEASURED
    assert tc.targeting_status(route=route, outcome=outcome, **arguments) == (
        "experimental",
        ("unmeasured",),
    )


def _in_scope_curve(*, n: int, binary: bool = False) -> Any:
    """An external pair of scores on a cross-fitted ATE fit, as cell H (or B) measures them."""
    rng = np.random.default_rng(11)
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, expit(0.5 * w[:, 0]))
    if binary:
        y = rng.binomial(1, expit(-0.2 + 0.4 * w[:, 1] + a * (0.3 + 1.2 * w[:, 0]))).astype(float)
    else:
        y = w[:, 1] + a * (1 + w[:, 0]) + 0.5 * rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    frame["S"] = frame["W1"] + 0.5 * frame["W2"]
    frame["R"] = frame["W1"] + 1.5 * frame["W2"]
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    effect = study.identify(ConditionalContrast(Modifiers(["W1"])))
    learners = {
        "outcome_learner": LogisticRegression() if binary else LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "outcome_by_arm": True,
    }
    if not binary:
        learners["q_bounds"] = (-8.0, 8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.targeting_curve(
            "S",
            computed_from=["W1", "W2"],
            score_kind="external",
            compare="R",
            compare_computed_from=["W1", "W2"],
            n_half_samples=20,
            random_state=0,
            **learners,
        )


def test_a_measured_curve_reads_supported_where_its_rows_are_measured() -> None:
    """At n = 2,000 every row of cell H's configuration, the paired rows included, reads
    supported: the top tenth holds the 200 rows the study calibrates at."""
    curve = _in_scope_curve(n=2_000)
    frame = curve.estimates
    assert curve.status == "supported", curve.status_reason
    assert set(frame["status"]) == {"supported"}
    assert {"S", "R", "S - R"} <= set(frame["score"])


def test_a_grid_point_below_the_measured_top_rows_reads_experimental() -> None:
    """The control in size: at n = 1,500 the top tenth holds 150 rows, so TOC(0.1) and every
    row that reads it is experimental while TOC(0.2) stays supported."""
    curve = _in_scope_curve(n=1_500)
    first, second = curve.rows("toc")[:2]
    assert first["rows_in_top"] == 150 and first["status"] == "experimental"
    assert "fewer than 200 rows" in first["status_reason"]
    assert second["status"] == "supported"
    assert curve.status == "experimental"


def test_a_paired_difference_outside_cell_c_reads_experimental() -> None:
    """The control in configuration: a binary outcome is measured (cells B, U, D) for one score,
    but the paired difference only on cell H's configuration."""
    curve = _in_scope_curve(n=2_000, binary=True)
    frame = curve.estimates
    rows = list(zip(frame["score"], frame["status"], frame["status_reason"], strict=True))
    single = [(status, reason) for score, status, reason in rows if score != "S - R"]
    paired = [(status, reason) for score, status, reason in rows if score == "S - R"]
    assert {status for status, _ in single} == {"supported"}, {reason for _, reason in single}
    assert paired and {status for status, _ in paired} == {"experimental"}
    assert all("paired difference" in reason for _, reason in paired)


def test_the_curve_status_is_the_worst_row_status() -> None:
    curve = _paired_curve()
    statuses = set(curve.estimates["status"])
    assert curve.status == ("experimental" if "experimental" in statuses else "supported")
    # A 1,000-row fit puts 100 rows in the top tenth, below the 200 the study measures.
    assert curve.status == "experimental" and "fewer than 200 rows" in curve.status_reason


# ----------------------------------------------------------------- summary and frame


def test_the_frame_holds_every_column_in_pandas_and_polars() -> None:
    curve = _paired_curve()
    frame = curve.to_frame()
    assert list(frame.columns) == list(tc.COLUMNS)
    assert set(frame["quantity"]) == {"toc", "gain", "group_effect", "autoc", "qini"}
    assert set(frame["score"]) == {"S", "R", "S - R"}
    from dataclasses import replace

    polars = replace(curve, backend="polars").to_frame()
    assert type(polars).__module__.startswith("polars")
    assert polars.shape == frame.shape


def test_the_summary_states_the_question_and_its_notes() -> None:
    curve = _paired_curve()
    summary = curve.summary()
    header = dict(summary.header)
    assert header["Question"] == "how much treating by S first gains over treating at random"
    assert header["Compared with"] == "R" and header["Rows"] == "1,000"
    assert {"targeting_external", "targeting_band", "targeting_experimental"} <= set(
        summary.note_codes
    )
    full = curve.summary(detail="full")
    comparison = full.section("Comparison with R").lines
    assert any(line.startswith("AUTOC of S - R: ") for line in comparison)
    assert full.section("P-values against 0").lines
    assert repr(curve).startswith("TargetingCurve('S'")
