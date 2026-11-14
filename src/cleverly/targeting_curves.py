r"""Targeting curves of a priority score: the TOC, the cumulative gain, the AUTOC and the Qini.

A priority score :math:`S = s(W)` ranks units for treatment.  With :math:`u(W)` the rank
position of a unit in the population ranking by decreasing :math:`S`, ties broken by an
independent uniform draw, and :math:`\tau(W) = E[Y^t - Y^r \mid W]`:

- the targeting operator characteristic :math:`TOC(q) = E[\tau \mid u \le q] - ATE`, on a
  declared grid of treated fractions :math:`q`, with :math:`TOC(1) = 0`;
- the cumulative gain :math:`G(q) = q\,E[\tau \mid u \le q]`, whose random-targeting line is
  :math:`q\,ATE`, so :math:`G(q) - q\,ATE = q\,TOC(q)`;
- the two areas :math:`AUTOC = \int_0^1 TOC(q)\,dq` and
  :math:`Qini = \int_0^1 q\,TOC(q)\,dq`;
- group effects :math:`E[\tau \mid q_{k-1} < u \le q_k]` between declared grid edges.

Every quantity is on the difference scale, whatever the scale of the score (Yadlowsky,
Fleming, Shah, Brunskill and Wager, arXiv:2111.07966v2, Definition 2 and Examples 2 and 3).

**Estimator.**  Each quantity is a weighted mean :math:`N^{-1}\sum_i w_i c(i)\,\Gamma_i` of a
doubly robust score with rank weights :math:`c` that depend on the score and the observation
weights only (:func:`block_weights`).  The estimator is a TMLE: one pooled logistic fluctuation
of the ATE fit's cross-fitted outcome regression along a clever covariate with one column per
direction of the span of the weight columns, so that every reported value is a substitution
estimate :math:`N^{-1}\sum_i w_i c(i)\,(\bar Q^*_t - \bar Q^*_r)(W_i)` and equals the
paper's estimator (2.8), (2.9) with the targeted scores :math:`\Gamma^*` in place of
:math:`\Gamma`, up to the solver's tolerance.

**Inference.**  The half-sample bootstrap of Yadlowsky et al. (Lemma 4, Corollary 5): each
draw keeps half the units without replacement, recomputes the ranks, the cuts and the tie
blocks on its rows, and evaluates every quantity from :math:`\Gamma^*` held fixed; the
standard error is the SD of the draws with no rescaling, and the interval is the Wald interval.
The band over the grid is the studentized sup-t band of the same draws.  The areas are the
published result; the points at a fixed ``q``, the band, the tie rule at an interior cut and
the targeted draws are natural extensions (``docs/technical-reference/conditional-effects.md``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
from scipy.special import digamma

from ._pooled_targeting import TARGETING_SPAN_RTOL, solve_pooled, span_basis
from ._typing import FloatArray, IntArray
from .data.known_mechanism import divides_by_declaration
from .exceptions import CapabilityError, DataError, Refusal, RefusalReason
from .inference.delta import reference_quantile, two_sided_pvalue
from .utils.frames import frame_from_dict

__all__ = [
    "DEFAULT_GRID",
    "TARGETING_ATOM_SHARE",
    "TARGETING_BOOTSTRAP_REFUSAL",
    "TARGETING_MEASURED",
    "TARGETING_MEASURED_TOP_ROWS",
    "TARGETING_MIN_CLUSTERS",
    "TARGETING_MIN_TOP_ROWS",
    "TARGETING_PAIRED_MEASURED",
    "TARGETING_REPEATS_REFUSAL",
    "TARGETING_SCHEME_REFUSAL",
    "TARGETING_SPAN_RTOL",
    "TARGETING_SPLIT_REFUSAL",
    "TargetingCurve",
    "TargetingEvaluation",
    "block_weights",
    "check_targeting_settings",
    "check_targeting_spec",
    "half_sample_draws",
    "targeting_fit",
    "targeting_status",
]

#: Why a targeting curve fits one cross-fitting split (X92 PR-5 plan, section 7).
TARGETING_REPEATS_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "The curve ranks one set of scores from one cross-fitting split; a median over repeated "
    "splits of a whole curve has no reviewed joint law, and no study measures it.",
    alternative="Fit with repeats=1.",
)

#: Why route 2 takes a fraction only (X92 PR-5 plan, section 3).
TARGETING_SPLIT_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "The average over folds of fold-specific targeting curves is a property of the learning "
    "algorithm. Its only source (Imai and Li 2023, Section 4) covers completely randomized "
    "experiments with difference-in-means scores, and no result covers doubly robust scores "
    "from estimated nuisances.",
    alternative="Pass split=0.5.",
)

#: Why a targeting curve refuses fold-wise targeting and fold evaluation of its ATE fit.
TARGETING_SCHEME_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "A targeting curve fits one pooled fluctuation over every evaluation row from the "
    "cross-fitted nuisances, so fold-wise targeting or fold evaluation of the ATE fit would be "
    "ignored.",
    alternative="Fit with targeting_scheme='pooled' and cv_evaluation=False.",
    roadmap="X92",
)

#: Why a targeting curve refuses a refit bootstrap of its ATE fit.
TARGETING_BOOTSTRAP_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "A targeting curve's intervals come from its half-sample bootstrap; a refit bootstrap of "
    "the ATE fit would be computed and ignored.",
    alternative="Fit with n_bootstrap=0; n_half_samples= sets the curve's own draws.",
)

#: The default grid of treated fractions, as grf's ``rank_average_treatment_effect``.
DEFAULT_GRID: Final[tuple[float, ...]] = tuple(round(0.1 * k, 1) for k in range(1, 11))

#: A grid point whose top fraction holds fewer rows reports its point with no interval and no
#: band.  A declared guard with no source: a half-sample of 50 top rows holds about 25.
TARGETING_MIN_TOP_ROWS: Final = 50

#: The fewest top rows at which the ``targeting-curves`` study calibrates an interval
#: (``q = 0.1`` at ``n = 2,000``, cells H, B, U and S); a grid point with fewer reads
#: experimental.  Cell P's ``n = 1,000`` measures the band's power only, which says nothing
#: about an interval's coverage.
TARGETING_MEASURED_TOP_ROWS: Final = 200

#: A score has atoms when one tie block holds at least this share of the evaluation rows.  A
#: declared guard with no source.
TARGETING_ATOM_SHARE: Final = 0.01

#: Below this many clusters the curve reports points only.  The number is borrowed from
#: :data:`cleverly._inference_status.FEW_CLUSTER_THRESHOLD`, which keeps its own meaning (the
#: Student-t reference); this guard has no source of its own.
TARGETING_MIN_CLUSTERS: Final = 40

#: Below this size relative to the largest targeted score a standard error counts as zero
#: (``TOC(1)`` and a degenerate draw): rounding in a mean of the scores is of that order.
_ZERO_SPREAD: Final = 1e-10

#: The configurations the registered ``targeting-curves`` study measures, each with the cell
#: that measures it, keyed by ``(route, outcome, mechanism)``.  A configuration outside this
#: table, or any departure from binary treatment, iid unweighted rows, complete outcomes, one
#: split and cross-fitted nuisances, reads experimental.
TARGETING_MEASURED: Final[Mapping[tuple[str, str, str], str]] = {
    ("external", "continuous", "estimated"): "H, C, N, P",
    ("external", "binary", "estimated"): "B, U, D",
    ("external", "continuous", "known"): "K",
    ("split", "continuous", "estimated"): "S",
}

#: The one configuration whose paired difference of two scores the study measures (cell C, on
#: cell H's fits).  A difference row of any other configuration reads experimental.
TARGETING_PAIRED_MEASURED: Final[tuple[str, str, str]] = ("external", "continuous", "estimated")

Quantity = Literal["toc", "gain", "group_effect", "autoc", "qini"]
TieRule = Literal["average", "at_or_above"]

_ORDER: Final[dict[str, int]] = {"supported": 0, "experimental": 1}

#: The reasons a row or a curve reads experimental, in the order they are listed.
REASONS: Final[Mapping[str, str]] = {
    "unmeasured": "no validation study of these curves measures this configuration",
    "unmeasured_pair": (
        "the paired difference of two scores is measured on an external score of a continuous "
        "outcome with an estimated mechanism only"
    ),
    "weights": "observation weights: the weighted rank positions follow grf's code, with no study",
    "clusters": "clustered rows: the half-samples draw whole clusters, with no study",
    "missing": "missing outcomes: the scores divide by the response mechanism, with no study",
    "multi_arm": "a pair of arms of a multi-arm treatment, with no study",
    "not_cross_fitted": "the ATE fit is not cross-fitted, which no study measures",
    "few_rows": "fewer than {floor} rows in the top fraction, which the study does not measure",
    "tied_cut": (
        "the cut falls inside a tied block of the score: the interval is valid when the "
        "population cut falls strictly inside the mass of one score value, not at its edge, and "
        "the data cannot tell the two apart"
    ),
    "tied_edge": (
        "the cut falls at the edge of a tied block of the score, where the limit is not normal"
    ),
    "atoms": (
        "the tie rule for a continuous score with atoms rests on the source's unproved "
        "statement (Yadlowsky et al., Appendix A.2)"
    ),
}

#: Why an interval is withheld, by cause.
WITHHELD: Final[Mapping[str, str]] = {
    "top_rows": "no interval: fewer than {guard} rows in the top fraction",
    "few_clusters": "no interval: fewer than {guard} clusters for the half-sample bootstrap",
    "anchor": "no interval: TOC(1) is 0 by definition",
    "degenerate": "no interval: the half-sample draws do not vary, so the spread is rounding",
}


# ------------------------------------------------------------------ the rank weights


@dataclass(frozen=True)
class _Layout:
    """The rows in decreasing score order and their tie blocks.

    ``order[k]`` is the row at rank ``k``; ``starts`` are the ranks where each block of equal
    score begins; ``block_of_row[i]`` is the block of row ``i``.
    """

    order: IntArray
    starts: IntArray
    block_of_row: IntArray

    @property
    def n_blocks(self) -> int:
        return int(self.starts.size)


def _order(score: FloatArray) -> IntArray:
    """The rows in decreasing score order (stable)."""
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")


def _layout(score: FloatArray) -> _Layout:
    s = np.asarray(score, dtype=float).reshape(-1)
    order = _order(s)
    ordered = s[order]
    changes = np.flatnonzero(np.diff(ordered) != 0.0) + 1
    starts = np.concatenate([[0], changes]).astype(np.int64)
    block_sorted = np.zeros(s.size, dtype=np.int64)
    block_sorted[changes] = 1
    block_sorted = np.cumsum(block_sorted)
    block_of_row = np.empty(s.size, dtype=np.int64)
    block_of_row[order] = block_sorted
    return _Layout(order=order, starts=starts, block_of_row=block_of_row)


def _block_sums(values_sorted: FloatArray, starts: IntArray) -> FloatArray:
    """Sum ``values_sorted`` (``(..., n)``, in rank order) within each block."""
    return np.add.reduceat(values_sorted, starts, axis=-1)


def _harmonic(x: FloatArray) -> FloatArray:
    """``H_x = psi(x + 1) + gamma``, the harmonic number at a real argument."""
    return digamma(np.asarray(x, dtype=float) + 1.0) + np.euler_gamma


def _autoc_block_weight(
    a: FloatArray, b: FloatArray, m: FloatArray, total: FloatArray
) -> FloatArray:
    r"""The AUTOC weight of a block of mass ``m`` occupying positions ``(a, b]`` of ``total``.

    With unit masses the weight of rank :math:`k` is :math:`H_n - H_{k-1} - 1`, from
    (2.9) with :math:`\alpha \equiv 1`; a block takes the mean over its ranks,
    :math:`H_n - (F(b) - F(a)) / m - 1` with :math:`F(x) = x H_x - x`, which is exact for
    integer masses and extends to real ones.
    """

    def f(x: FloatArray) -> FloatArray:
        return x * _harmonic(x) - x

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(m > 0.0, (f(b) - f(a)) / np.where(m > 0.0, m, 1.0), 0.0)
    return _harmonic(total) - mean - 1.0


def _qini_block_weight(
    a: FloatArray, b: FloatArray, m: FloatArray, total: FloatArray
) -> FloatArray:
    r"""The Qini weight :math:`(N - a - b) / (2N)` of a block at positions ``(a, b]``.

    With unit masses this is :math:`(n + 1 - 2k) / (2n)` at rank :math:`k`, from (2.9) with
    :math:`\alpha(u) = u`, and a block of ranks ``k_first`` to ``k_last`` takes
    :math:`(n + 1 - (k_first + k_last)) / (2n)`.
    """
    del m
    return (total - a - b) / (2.0 * total)


def _shares(
    m: FloatArray, a: FloatArray, total: FloatArray, q: float, tie_rule: TieRule
) -> FloatArray:
    """Each block's share in the top fraction ``q``: 1 above the cut, 0 below, fractional at it."""
    cut = q * total
    if tie_rule == "at_or_above":
        return np.where(a < cut * (1.0 - 1e-12), 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(m > 0.0, (cut - a) / np.where(m > 0.0, m, 1.0), 0.0)
    share = np.clip(share, 0.0, 1.0)
    share = np.where(share < 1e-12, 0.0, share)
    return np.where(share > 1.0 - 1e-12, 1.0, share)


def block_weights(
    m: FloatArray,
    grid: Sequence[float],
    edges: Sequence[float] = (),
    *,
    tie_rule: TieRule = "average",
    shares: Mapping[float, FloatArray] | None = None,
) -> dict[tuple[str, float], FloatArray]:
    r"""The rank weight of every reported quantity at every tie block.

    ``m`` holds the block masses in decreasing score order, ``(..., n_blocks)``; a leading axis
    is a stack of half-samples.  Each weight multiplies the block's sum of
    :math:`w_i \Gamma_i`, and the quantity is that sum over blocks divided by the total mass.

    Parameters
    ----------
    m : ndarray
        Block masses, ``(..., n_blocks)``.
    grid : sequence of float
        Treated fractions in ``(0, 1]``.
    edges : sequence of float
        Upper edges of the group effects, on the grid; the first group starts at 0.
    tie_rule : {"average", "at_or_above"}
        ``"average"`` gives the block at the cut its fractional share; ``"at_or_above"``
        puts the whole block above the cut, the rule a mutation control reads.
    shares : mapping of float to ndarray or None
        Shares fixed in advance (a fixed-cut control); ``None`` computes them from ``m``.

    Returns
    -------
    dict
        Keyed by ``(quantity, q)``: ``("toc", q)``, ``("gain", q)``, ``("ate", 1.0)``,
        ``("autoc", 1.0)``, ``("qini", 1.0)`` and ``("group_effect", edge)``.
    """
    m = np.asarray(m, dtype=float)
    b = np.cumsum(m, axis=-1)
    a = b - m
    total = b[..., -1:]
    out: dict[tuple[str, float], FloatArray] = {("ate", 1.0): np.ones_like(m)}
    member: dict[float, FloatArray] = {0.0: np.zeros_like(m)}
    for q in sorted({*grid, *edges}):
        share = (
            _shares(m, a, total, q, tie_rule)
            if shares is None
            else np.broadcast_to(shares[q], m.shape)
        )
        top = np.sum(m * share, axis=-1, keepdims=True) / total[..., :1]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[("toc", q)] = share / top - 1.0
        out[("gain", q)] = q * share / top
        member[q] = q * share / top
    previous = 0.0
    for edge in edges:
        out[("group_effect", edge)] = (member[edge] - member[previous]) / (edge - previous)
        previous = edge
    out[("autoc", 1.0)] = _autoc_block_weight(a, b, m, total)
    out[("qini", 1.0)] = _qini_block_weight(a, b, m, total)
    return out


# ------------------------------------------------------------------ the evaluation


@dataclass(frozen=True)
class TargetingEvaluation:
    """The evaluation rows a targeting curve's half-sample draws read.

    Parameters
    ----------
    scores : dict of str to ndarray
        Each ranked score at each evaluation row, by name; the first is the primary score.
    targeted : ndarray
        The targeted doubly robust score :math:`\\Gamma^*` of each row, on the outcome scale.
    one_step : ndarray
        The one-step score :math:`\\Gamma` from the initial outcome regression.
    mass : ndarray
        Each row's observation weight, normalised to mean one.
    units : ndarray
        The independent unit of each row: its cluster, or the row itself.
    grid : tuple of float
        The treated fractions.
    edges : tuple of float
        The group-effect edges.
    """

    scores: dict[str, FloatArray] = field(repr=False)
    targeted: FloatArray = field(repr=False)
    one_step: FloatArray = field(repr=False)
    mass: FloatArray = field(repr=False)
    units: IntArray = field(repr=False)
    grid: tuple[float, ...]
    edges: tuple[float, ...]

    def layout(self, name: str) -> _Layout:
        """The rank order and tie blocks of one score."""
        return _layout(self.scores[name])


def _point_values(
    layout: _Layout,
    values: FloatArray,
    mass: FloatArray,
    grid: Sequence[float],
    edges: Sequence[float],
    tie_rule: TieRule = "average",
) -> dict[tuple[str, float], float]:
    """Every quantity as ``sum_b c_b sum_{i in b} w_i v_i / N`` on the full sample."""
    order = layout.order
    m = _block_sums(mass[order], layout.starts)
    sums = _block_sums(mass[order] * values[order], layout.starts)
    weights = block_weights(m, grid, edges, tie_rule=tie_rule)
    total = float(m.sum())
    return {key: float(np.sum(c * sums) / total) for key, c in weights.items()}


def _row_columns(
    layout: _Layout,
    mass: FloatArray,
    grid: Sequence[float],
    keys: Sequence[tuple[str, float]],
) -> FloatArray:
    """The weight of each quantity in ``keys`` at each row, in row order, ``(n, len(keys))``."""
    m = _block_sums(mass[layout.order], layout.starts)
    weights = block_weights(m, grid)
    return np.column_stack([weights[key][layout.block_of_row] for key in keys])


def _masks(units: IntArray, n_draws: int, rng: np.random.Generator) -> tuple[IntArray, FloatArray]:
    """``n_draws`` half-samples of the units, as a row-to-unit index and a unit mask."""
    labels, inverse = np.unique(units, return_inverse=True)
    count = labels.size
    half = count // 2
    keys = rng.random((n_draws, count))
    chosen = np.argsort(keys, axis=1)[:, :half]
    mask = np.zeros((n_draws, count))
    np.put_along_axis(mask, chosen, 1.0, axis=1)
    return inverse.astype(np.int64), mask


def half_sample_draws(
    evaluation: TargetingEvaluation,
    *,
    n_half_samples: int,
    random_state: int | None,
    scores: Literal["targeted", "one_step"] = "targeted",
    cut: Literal["quantile", "fixed"] = "quantile",
    ranks: Literal["recompute", "fixed"] = "recompute",
    tie_rule: TieRule = "average",
    paired: bool = True,
    masks: FloatArray | None = None,
    chunk: int | None = None,
) -> dict[str, dict[tuple[str, float], FloatArray]]:
    """Each quantity of each score in each half-sample, ``(n_half_samples,)`` per key.

    The draws keep :math:`\\Gamma^*` fixed and recompute the ranks, the cuts and the tie blocks
    on the rows each draw keeps.  The other values of the keyword arguments are deliberate
    mutations the unit tests and the study's controls read: the one-step scores, the
    full-sample cut, the full-sample rank weights, the whole tied block at the cut, and
    independent draws for every score after the first.

    Parameters
    ----------
    evaluation : TargetingEvaluation
        The evaluation rows.
    n_half_samples : int
        Draws.
    random_state : int or None
        Seed of the draws.
    scores : {"targeted", "one_step"}
        Which doubly robust score the draws average.
    cut : {"quantile", "fixed"}
        ``"fixed"`` keeps each block's full-sample share in the top fraction and divides by
        the draw's own mass above it.
    ranks : {"recompute", "fixed"}
        ``"fixed"`` applies the full-sample rank weights to every draw.
    tie_rule : {"average", "at_or_above"}
        As :func:`block_weights`.
    paired : bool
        ``False`` draws every score after the first from its own stream.
    masks : ndarray or None
        ``(D, n_units)`` explicit unit masks in the order of ``numpy.unique(units)``, in place of
        random draws.
    chunk : int or None
        Draws per vectorised block; ``None`` sizes it to about four million cells.

    Returns
    -------
    dict
        By score name, a mapping from ``(quantity, q)`` to the draws.
    """
    gamma = evaluation.targeted if scores == "targeted" else evaluation.one_step
    mass = evaluation.mass
    rng = np.random.default_rng(random_state)
    inverse = np.unique(evaluation.units, return_inverse=True)[1].astype(np.int64)
    shipped = cut == "quantile" and ranks == "recompute" and tie_rule == "average"
    if masks is None:
        _, unit_masks = _masks(evaluation.units, n_half_samples, rng)
    else:
        unit_masks = np.asarray(masks, dtype=float)
    out: dict[str, dict[tuple[str, float], FloatArray]] = {}
    for index, (name, _score) in enumerate(evaluation.scores.items()):
        layout = evaluation.layout(name)
        order = layout.order
        draws_masks = unit_masks
        if not paired and index > 0 and masks is None:
            stream = np.random.default_rng([0 if random_state is None else random_state, index])
            _, draws_masks = _masks(evaluation.units, n_half_samples, stream)
        full_m = _block_sums(mass[order], layout.starts)
        fixed_shares = None
        fixed_weights = None
        if cut == "fixed":
            b = np.cumsum(full_m)
            fixed_shares = {
                q: _shares(full_m, b - full_m, b[-1:], q, tie_rule)
                for q in sorted({*evaluation.grid, *evaluation.edges})
            }
        if ranks == "fixed":
            fixed_weights = block_weights(
                full_m, evaluation.grid, evaluation.edges, tie_rule=tie_rule
            )
        size = chunk or max(1, int(4_000_000 // max(1, gamma.size)))
        pieces: dict[tuple[str, float], list[FloatArray]] = {}
        for start in range(0, draws_masks.shape[0], size):
            row_mask = draws_masks[start : start + size][:, inverse]
            weighted = row_mask[:, order] * mass[order]
            m = _block_sums(weighted, layout.starts)
            sums = _block_sums(weighted * gamma[order], layout.starts)
            total = m.sum(axis=1)
            if shipped:
                for key, values in _shipped_draws(
                    m, sums, evaluation.grid, evaluation.edges
                ).items():
                    pieces.setdefault(key, []).append(values)
                continue
            weights = (
                fixed_weights
                if fixed_weights is not None
                else block_weights(
                    m, evaluation.grid, evaluation.edges, tie_rule=tie_rule, shares=fixed_shares
                )
            )
            for key, c in weights.items():
                pieces.setdefault(key, []).append(np.sum(c * sums, axis=-1) / total)
        out[name] = {key: np.concatenate(parts) for key, parts in pieces.items()}
    return out


def _shipped_draws(
    m: FloatArray, sums: FloatArray, grid: Sequence[float], edges: Sequence[float]
) -> dict[tuple[str, float], FloatArray]:
    """The shipped rule's quantities from the block masses and sums of a stack of draws.

    The same arithmetic as :func:`block_weights`, without forming a share for every block at
    every grid point: the top fraction's sum is the cumulative sum through the block before
    the cut plus the cut block's fractional share.  A unit test checks it against the general
    form.
    """
    upper = np.cumsum(m, axis=1)
    running = np.cumsum(sums, axis=1)
    total = upper[:, -1]
    ate = running[:, -1] / total
    out: dict[tuple[str, float], FloatArray] = {("ate", 1.0): ate}
    rows = np.arange(m.shape[0])
    gain: dict[float, FloatArray] = {0.0: np.zeros_like(total)}
    for q in sorted({*grid, *edges}):
        cut = q * total
        index = np.minimum(np.sum(upper < cut[:, None] * (1.0 - 1e-12), axis=1), m.shape[1] - 1)
        before = np.where(index > 0, upper[rows, index - 1], 0.0)
        summed = np.where(index > 0, running[rows, index - 1], 0.0)
        block = m[rows, index]
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.clip(np.where(block > 0.0, (cut - before) / block, 0.0), 0.0, 1.0)
        top = summed + share * sums[rows, index]
        gain[q] = top / total
        out[("gain", q)] = gain[q]
        out[("toc", q)] = top / cut - ate
    previous = 0.0
    for edge in edges:
        out[("group_effect", edge)] = (gain[edge] - gain[previous]) / (edge - previous)
        previous = edge
    b = upper
    a = upper - m
    column = total[:, None]
    out[("autoc", 1.0)] = np.sum(_autoc_block_weight(a, b, m, column) * sums, axis=1) / total
    out[("qini", 1.0)] = np.sum(_qini_block_weight(a, b, m, column) * sums, axis=1) / total
    return out


# ------------------------------------------------------------------ the status


def targeting_status(
    *,
    route: str,
    outcome: str,
    known_mechanism: bool,
    weighted: bool,
    clustered: bool,
    missing: bool,
    arms: int,
    cross_fitted: bool,
) -> tuple[str, tuple[str, ...]]:
    """The configuration's support label and its reasons.

    A curve's configuration reads ``supported`` when it is one of :data:`TARGETING_MEASURED`
    and departs from the measured composition in nothing.  Otherwise it reads
    ``experimental``, with every reason that applies.

    Parameters
    ----------
    route : {"external", "split"}
        Where the score came from.
    outcome : {"binary", "continuous"}
        The outcome type.
    known_mechanism : bool
        Whether the treatment mechanism was declared known.
    weighted : bool
        Whether the rows carry observation weights.
    clustered : bool
        Whether the rows are clustered.
    missing : bool
        Whether some outcomes are missing.
    arms : int
        The number of treatment arms.
    cross_fitted : bool
        Whether the ATE fit cross-fitted its nuisances.

    Returns
    -------
    tuple
        ``"supported"`` or ``"experimental"``, and the reason codes (keys of
        :data:`REASONS`).
    """
    reasons: list[str] = []
    if _configuration(route, outcome, known_mechanism) not in TARGETING_MEASURED:
        reasons.append("unmeasured")
    for flag, code in (
        (weighted, "weights"),
        (clustered, "clusters"),
        (missing, "missing"),
        (arms > 2, "multi_arm"),
        (not cross_fitted, "not_cross_fitted"),
    ):
        if flag:
            reasons.append(code)
    return ("experimental" if reasons else "supported"), tuple(reasons)


def _configuration(route: str, outcome: str, known_mechanism: bool) -> tuple[str, str, str]:
    return (route, outcome, "known" if known_mechanism else "estimated")


def _reason_text(code: str) -> str:
    return REASONS[code].format(floor=TARGETING_MEASURED_TOP_ROWS)


def _worst(statuses: Sequence[str]) -> str:
    return max(statuses, key=lambda status: _ORDER[status]) if statuses else "supported"


# ------------------------------------------------------------------ the result


#: The columns of :attr:`TargetingCurve.estimates`, in order.
COLUMNS: Final[tuple[str, ...]] = (
    "score",
    "quantity",
    "q",
    "estimate",
    "std_error",
    "lower",
    "upper",
    "band_lower",
    "band_upper",
    "p_value",
    "rows_in_top",
    "tied_cut",
    "tie_share",
    "mean_score",
    "status",
    "status_reason",
)


@dataclass(frozen=True)
class TargetingCurve:
    """How much treating by a priority score first gains over treating at random.

    Parameters
    ----------
    score : str
        The ranked score's name.
    compare : str or None
        The second score of a paired comparison, or ``None``.
    route : {"external", "split"}
        ``"external"``: a study column fitted on other rows. ``"split"``: a curve refitted on
        a declared training split and evaluated on the other rows.
    n_train : int or None
        Rows of the training part on the split route; ``None`` on the external route.
    n_eval : int
        Evaluation rows.
    grid : tuple of float
        The treated fractions.
    estimates : dict of str to list
        One entry per column of :data:`COLUMNS`, one row per estimate: the TOC and the gain at
        each grid point, the group effects, the AUTOC and the Qini, for each score and, with a
        comparison, their differences under the name ``"<score> - <compare>"``.
    ate : float
        The average effect, read from the fluctuation's own constant column, so that each TOC
        equals its top-fraction mean minus this value exactly.
    critical_value : float
        The sup-t critical value of the band over the grid points below 1, or ``nan`` when the
        band is withheld.
    n_half_samples : int
        Half-sample draws.
    alpha : float
        Intervals hold at ``1 - alpha``; the band jointly.
    status : {"supported", "experimental"}
        The worst status of the rows.
    status_reason : str
        Every reason that applies, or the empty string.
    backend : str or None
        The dataframe backend of the study's data.
    span : dict of str to float
        The fluctuation's basis: ``columns`` (weight columns), ``rank`` (basis directions),
        ``out_of_span`` (the largest relative component of a column outside the basis) and
        ``score_norm`` (the largest score equation after targeting).
    evaluation : TargetingEvaluation
        The evaluation rows the draws read.
    fit : TMLEResult
        The ATE fit on the evaluation rows whose nuisances the curve targets.
    score_curve : EffectCurveResult or None
        On the split route, the curve refit on the training rows whose predictions are the
        score; ``None`` on the external route.
    """

    score: str
    compare: str | None
    route: str
    n_train: int | None
    n_eval: int
    grid: tuple[float, ...]
    estimates: dict[str, list[Any]] = field(repr=False)
    ate: float
    critical_value: float
    n_half_samples: int
    alpha: float
    status: str
    status_reason: str
    backend: str | None = None
    span: dict[str, float] = field(default_factory=dict, repr=False)
    evaluation: TargetingEvaluation | None = field(default=None, repr=False, compare=False)
    fit: Any = field(default=None, repr=False, compare=False)
    score_curve: Any = field(default=None, repr=False, compare=False)

    def rows(self, quantity: str, score: str | None = None) -> list[dict[str, Any]]:
        """The rows of one quantity of one score, in grid order.

        Parameters
        ----------
        quantity : {"toc", "gain", "group_effect", "autoc", "qini"}
            The quantity.
        score : str or None
            The score's name, or ``"<score> - <compare>"``; the primary score when omitted.

        Returns
        -------
        list of dict
            One mapping per row, keyed by :data:`COLUMNS`.
        """
        name = self.score if score is None else score
        n = len(self.estimates["score"])
        out = []
        for index in range(n):
            if (
                self.estimates["score"][index] == name
                and self.estimates["quantity"][index] == quantity
            ):
                out.append({column: self.estimates[column][index] for column in COLUMNS})
        return out

    def area(self, quantity: str = "autoc", score: str | None = None) -> dict[str, Any]:
        """The row of one area.

        Parameters
        ----------
        quantity : {"autoc", "qini"}
            The area.
        score : str or None
            As :meth:`rows`.

        Returns
        -------
        dict
            The row, keyed by :data:`COLUMNS`.
        """
        (row,) = self.rows(quantity, score)
        return row

    def to_frame(self) -> Any:
        """The estimates, one row each, in the study's backend.

        Returns
        -------
        dataframe
            The columns of :data:`COLUMNS`.
        """
        return frame_from_dict(
            {column: self.estimates[column] for column in COLUMNS}, backend=self.backend
        )

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the curve.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the gain, the group effects, the differences and the p-values.

        Returns
        -------
        Summary
            The report.
        """
        from ._summary._targeting import targeting_curve_summary

        return targeting_curve_summary(self, detail=detail)

    def plot(self, view: str = "toc", **kwargs: Any) -> Any:
        """Draw the curve with :func:`cleverly.plotting.targeting`.

        Parameters
        ----------
        view : {"toc", "gain", "difference"}
            Which view to draw.
        **kwargs : Any
            Passed to :func:`cleverly.plotting.targeting`.

        Returns
        -------
        Figure
            The figure drawn on.
        """
        from .plotting import targeting

        return targeting(self, view=view, **kwargs)

    def _repr_html_(self) -> str:
        return str(self.summary().as_html())

    def __repr__(self) -> str:
        autoc = self.area("autoc")["estimate"]
        return (
            f"TargetingCurve({self.score!r}, route={self.route!r}, n_eval={self.n_eval}, "
            f"autoc={autoc:.4g}, status={self.status!r})"
        )


# ------------------------------------------------------------------ the fit


def check_grid(grid: Sequence[float] | None) -> tuple[float, ...]:
    """A declared grid: finite, strictly increasing, in ``(0, 1]``, ending at 1.

    Parameters
    ----------
    grid : sequence of float or None
        The treated fractions; ``None`` takes :data:`DEFAULT_GRID`.

    Returns
    -------
    tuple of float
        The grid, with 1 appended when it does not end there.

    Raises
    ------
    DataError
        For a point outside ``(0, 1]``, a repeated or unsorted grid, or an empty one.
    """
    if grid is None:
        return DEFAULT_GRID
    values = [float(q) for q in grid]
    if not values or not all(math.isfinite(q) and 0.0 < q <= 1.0 for q in values):
        raise DataError(f"every grid point must lie in (0, 1]; got {values}")
    if any(b <= a for a, b in itertools.pairwise(values)):
        raise DataError(f"the grid must be strictly increasing; got {values}")
    if values[-1] != 1.0:
        values.append(1.0)
    return tuple(values)


def check_groups(
    groups: int | Sequence[float] | None, grid: tuple[float, ...]
) -> tuple[float, ...]:
    """The upper edges of the group effects, each on the grid, ending at 1.

    Parameters
    ----------
    groups : int, sequence of float or None
        A number of equal groups, or their upper edges; ``None`` or 0 for no groups.
    grid : tuple of float
        The grid.

    Returns
    -------
    tuple of float
        The edges.

    Raises
    ------
    DataError
        When an edge is not a grid point.
    """
    if isinstance(groups, bool):
        raise DataError(f"groups must be a number of groups or their edges; got {groups}")
    if groups is None or (isinstance(groups, int) and groups == 0):
        return ()
    if isinstance(groups, int):
        if groups < 1:
            raise DataError(f"groups must be a positive number of groups; got {groups}")
        edges = [round(k / groups, 12) for k in range(1, groups + 1)]
    else:
        edges = [float(edge) for edge in groups]
        if not edges or edges[-1] != 1.0:
            edges.append(1.0)
    on_grid = [edge for edge in edges if not any(math.isclose(edge, q, abs_tol=1e-9) for q in grid)]
    if on_grid:
        raise DataError(
            f"every group edge must be a grid point, so that each group effect is a difference "
            f"of two gains on the grid; {on_grid} are not in {list(grid)}"
        )
    if any(b <= a for a, b in itertools.pairwise(edges)):
        raise DataError(f"group edges must be strictly increasing; got {edges}")
    snapped = [next(q for q in grid if math.isclose(edge, q, abs_tol=1e-9)) for edge in edges]
    return tuple(snapped)


def _fluctuation_keys(grid: Sequence[float]) -> list[tuple[str, float]]:
    """The weight columns of one score the fluctuation targets: TOC below 1 and the two areas.

    The gains and the group effects are linear combinations of these and the shared constant.
    """
    return [*(("toc", q) for q in grid if q < 1.0), ("autoc", 1.0), ("qini", 1.0)]


@dataclass(frozen=True)
class _Targeted:
    tau: FloatArray
    targeted: FloatArray
    one_step: FloatArray
    span: dict[str, float]


def _target(
    fit: Any,
    data: Any,
    layouts: Mapping[str, _Layout],
    grid: tuple[float, ...],
    pair: tuple[float, float],
) -> _Targeted:
    """One pooled fluctuation along the span of every score's weight columns."""
    mass = np.asarray(data.weights, dtype=float)
    n = int(mass.size)
    keys = _fluctuation_keys(grid)
    blocks = [np.ones((n, 1))]
    for layout in layouts.values():
        blocks.append(_row_columns(layout, mass, grid, keys))
    columns = np.column_stack(blocks)
    basis, out_of_span = span_basis(columns)
    pooled = solve_pooled(fit, data, basis, -basis, pair, what="the targeting curve")
    q_t, q_r, y = pooled.q_t, pooled.q_r, pooled.y
    gamma_star = (q_t - q_r) + pooled.weight_t * (y - q_t) - pooled.weight_r * (y - q_r)
    span = {
        "columns": float(columns.shape[1]),
        "rank": float(basis.shape[1]),
        "out_of_span": out_of_span,
        "score_norm": pooled.score_norm,
    }
    return _Targeted(q_t - q_r, gamma_star, pooled.d_t - pooled.d_r, span)


def _spread(draws: FloatArray) -> float:
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def check_targeting_spec(spec: Any) -> None:
    """Refuse an ATE fit's targeting settings that no targeting curve can solve under.

    Both routes call it on the ATE estimator before it fits, and :func:`targeting_fit` again
    on the fit it receives.

    Parameters
    ----------
    spec : TargetingSpec
        The estimator's ``targeting_spec()``.

    Raises
    ------
    CapabilityError
        For ``target_weights=True`` or a declared stopping rule.
    """
    if spec.target_weights:
        raise CapabilityError(
            "a targeting curve fluctuates along several weight columns at once, and every row "
            "loads several of them, which the weighted form of the fluctuation cannot carry.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Fit with target_weights=False.",
            roadmap="X92",
        )
    if spec.stopping != "relative":
        raise CapabilityError(
            "a targeting curve solves its fluctuation under the shipped relative stopping test; "
            "a declared stopping rule needs a stopping context no targeting curve builds.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Fit with stopping='relative'.",
            roadmap="X92",
        )


def check_targeting_settings(estimator: Any) -> None:
    """Refuse an ATE estimator's settings that a targeting curve would silently drop.

    Both routes call it before the ATE fit: the targeting settings of
    :func:`check_targeting_spec`, then fold-wise targeting, fold evaluation and a refit
    bootstrap, none of which the pooled fluctuation and the half-sample bootstrap read.

    Parameters
    ----------
    estimator : TMLE
        The ATE estimator, before it fits.

    Raises
    ------
    CapabilityError
        For ``target_weights=True``, a declared stopping rule, ``targeting_scheme="fold"``,
        ``cv_evaluation=True`` or ``n_bootstrap`` above 0.
    """
    check_targeting_spec(estimator.targeting_spec())
    if estimator.targeting_scheme != "pooled" or bool(estimator.cv_evaluation):
        raise CapabilityError(refusal=TARGETING_SCHEME_REFUSAL)
    if int(estimator.n_bootstrap) > 0:
        raise CapabilityError(refusal=TARGETING_BOOTSTRAP_REFUSAL)


def targeting_fit(
    fit: Any,
    data: Any,
    scores: Mapping[str, FloatArray],
    *,
    pair: tuple[float, float],
    grid: Sequence[float] | None = None,
    groups: int | Sequence[float] | None = 5,
    n_half_samples: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = None,
    route: str = "external",
    n_train: int | None = None,
    score_curve: Any = None,
) -> TargetingCurve:
    """Fit the targeting curve of one score, or two, from an ATE fit on the evaluation rows.

    Parameters
    ----------
    fit : TMLEResult
        The single ATE fit on the evaluation rows.
    data : CausalData
        The evaluation rows, as the fit prepared them.
    scores : mapping of str to ndarray
        The primary score, and optionally a second one to compare, by name.
    pair : tuple of float
        The arm codes ``(t, r)``.
    grid : sequence of float or None
        Treated fractions; :data:`DEFAULT_GRID` when omitted.
    groups : int, sequence of float or None
        Group effects: a number of equal groups, or their upper edges on the grid.
    n_half_samples : int
        Half-sample draws.
    alpha : float
        Intervals hold at ``1 - alpha``.
    random_state : int or None
        Seed of the draws.
    route : {"external", "split"}
        Where the score came from.
    n_train : int or None
        Training rows on the split route.
    score_curve : EffectCurveResult or None
        The training refit whose predictions are the score, on the split route.

    Returns
    -------
    TargetingCurve
        The curve.

    Raises
    ------
    CapabilityError
        For ``target_weights=True`` or a declared stopping rule on the ATE fit.
    DataError
        For a score with a non-finite value, a grid or group declaration off its rules, or a
        targeting step that does not converge.
    """
    check_targeting_spec(fit.config.targeting_spec)
    if int(n_half_samples) < 2:
        raise DataError(f"n_half_samples must be at least 2; got {n_half_samples}")
    if not 0.0 < float(alpha) < 1.0:
        raise DataError(f"alpha must lie in (0, 1); got {alpha}")
    grid_ = check_grid(grid)
    edges = check_groups(groups, grid_)
    arrays = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in scores.items()}
    for name, values in arrays.items():
        if values.shape[0] != data.n or not np.all(np.isfinite(values)):
            raise DataError(f"the score {name!r} must hold one finite value per evaluation row")
    layouts = {name: _layout(values) for name, values in arrays.items()}
    targeted = _target(fit, data, layouts, grid_, pair)
    mass = np.asarray(data.weights, dtype=float)
    units = (
        np.arange(data.n, dtype=np.int64)
        if data.cluster is None
        else np.unique(np.asarray(data.cluster), return_inverse=True)[1].astype(np.int64)
    )
    evaluation = TargetingEvaluation(
        scores=arrays,
        targeted=targeted.targeted,
        one_step=targeted.one_step,
        mass=mass,
        units=units,
        grid=grid_,
        edges=edges,
    )
    draws = half_sample_draws(
        evaluation, n_half_samples=int(n_half_samples), random_state=random_state
    )
    n_units = int(np.unique(units).size)
    clustered = data.cluster is not None
    few_clusters = clustered and n_units < TARGETING_MIN_CLUSTERS
    outcome = "binary" if getattr(data, "family", "") == "binomial" else "continuous"
    known_mechanism = divides_by_declaration(data)
    config_status, config_codes = targeting_status(
        route=route,
        outcome=outcome,
        known_mechanism=known_mechanism,
        weighted=bool(np.any(np.abs(mass - 1.0) > 1e-12)),
        clustered=clustered,
        missing=bool(not np.all(np.asarray(data.observed, dtype=bool))),
        arms=len(tuple(fit.nuisance.arms)),
        cross_fitted=bool(getattr(fit.config, "cross_fit", True)),
    )
    z = reference_quantile(float(alpha))
    builder = _TargetingRows(
        z=z,
        alpha=float(alpha),
        config_codes=config_codes,
        few_clusters=few_clusters,
        scale=float(np.max(np.abs(targeted.targeted))) if targeted.targeted.size else 1.0,
    )
    names = list(arrays)
    plug_ins: dict[str, dict[tuple[str, float], float]] = {}
    critical = float("nan")
    for name in names:
        layout = layouts[name]
        plug_ins[name] = _point_values(layout, targeted.tau, mass, grid_, edges)
        facts = _cut_facts(layout, mass, grid_)
        atoms = _has_atoms(layout, data.n)
        critical_name = builder.score_rows(
            name,
            plug_ins[name],
            draws[name],
            facts,
            atoms,
            grid_,
            edges,
            arrays[name],
            layout,
            mass,
        )
        if name == names[0]:
            critical = critical_name
    if len(names) == 2:
        paired = _configuration(route, outcome, known_mechanism) == TARGETING_PAIRED_MEASURED
        builder.difference_rows(
            names, plug_ins, draws, grid_, pair_codes=() if paired else ("unmeasured_pair",)
        )
    ate = plug_ins[names[0]][("ate", 1.0)]
    statuses = list(builder.columns["status"])
    status = _worst([*statuses, config_status])
    reasons = list(
        dict.fromkeys([*(_reason_text(code) for code in config_codes), *builder.reasons])
    )
    return TargetingCurve(
        score=names[0],
        compare=names[1] if len(names) == 2 else None,
        route=route,
        n_train=n_train,
        n_eval=int(data.n),
        grid=grid_,
        estimates={column: list(builder.columns[column]) for column in COLUMNS},
        ate=float(ate),
        critical_value=critical,
        n_half_samples=int(n_half_samples),
        alpha=float(alpha),
        status=status,
        status_reason="; ".join(reasons),
        backend=getattr(data, "backend", None),
        span=targeted.span,
        evaluation=evaluation,
        fit=fit,
        score_curve=score_curve,
    )


@dataclass(frozen=True)
class _CutFact:
    """``tied`` is set when the cut falls inside a tied block (``share`` is its share above the
    cut) or exactly at the edge of one (``share`` is NaN): either way the score has an atom at
    the cut, which D1 excludes."""

    rows_in_top: float
    tied: bool
    share: float

    @property
    def code(self) -> str:
        """The row reason of a tied cut: inside a block, or at its edge."""
        return "tied_cut" if math.isfinite(self.share) else "tied_edge"


def _cut_facts(layout: _Layout, mass: FloatArray, grid: Sequence[float]) -> dict[float, _CutFact]:
    """Per grid point: the rows in the top fraction, and whether the cut falls in a tied block."""
    order = layout.order
    m = _block_sums(mass[order], layout.starts)
    counts = np.diff(np.append(layout.starts, order.size)).astype(float)
    b = np.cumsum(m)
    a = b - m
    total = b[-1:]
    out = {}
    for q in grid:
        share = _shares(m, a, total, q, "average")
        rows = float(np.sum(counts * share))
        inside = np.flatnonzero((share > 0.0) & (share < 1.0))
        if inside.size:
            block = int(inside[0])
            if counts[block] > 1:
                out[q] = _CutFact(rows, True, float(share[block]))
                continue
            # A cut a fraction of a row inside a single row beside a tied block: the sample
            # quantile is that block's edge to within one row, an atom of the score at the cut.
            edge = any(counts[j] > 1 for j in _weighted_neighbours(m, block - 1, block + 1))
            out[q] = _CutFact(rows, edge, float("nan"))
            continue
        # The cut sits on the boundary between the last weighted block wholly above it and the
        # first weighted block wholly below; a tied block on either side is an atom at the cut.
        above = np.flatnonzero((share >= 1.0) & (m > 0.0))
        last = int(above[-1]) if above.size else -1
        edge = q < 1.0 and any(counts[j] > 1 for j in _weighted_neighbours(m, last, last + 1))
        out[q] = _CutFact(rows, edge, float("nan"))
    return out


def _weighted_neighbours(m: FloatArray, before: int, after: int) -> list[int]:
    """The nearest block with positive mass at or before ``before``, and at or after ``after``.

    A block of zero-weight rows carries no mass, so it is not a neighbour of the cut.
    """
    out = []
    j = before
    while j >= 0 and m[j] <= 0.0:
        j -= 1
    if j >= 0:
        out.append(j)
    j = after
    while j < m.size and m[j] <= 0.0:
        j += 1
    if j < m.size:
        out.append(j)
    return out


def _has_atoms(layout: _Layout, n: int) -> bool:
    """A continuous score with atoms: a block of at least 1% of rows and over n/10 values."""
    counts = np.diff(np.append(layout.starts, layout.order.size))
    return bool(counts.max() >= TARGETING_ATOM_SHARE * n and layout.n_blocks > n / 10.0)


class _TargetingRows:
    """The estimates table, built row by row."""

    def __init__(
        self,
        *,
        z: float,
        alpha: float,
        config_codes: tuple[str, ...],
        few_clusters: bool,
        scale: float = 1.0,
    ) -> None:
        # A spread at or below this is rounding in means of scores of size ``scale``.
        self.zero_spread = _ZERO_SPREAD * max(float(scale), np.finfo(float).tiny)
        self.z = z
        self.alpha = alpha
        self.config_codes = config_codes
        self.few_clusters = few_clusters
        self.columns: dict[str, list[Any]] = {column: [] for column in COLUMNS}
        self.reasons: list[str] = []
        self.toc_status: dict[str, dict[float, tuple[str, bool]]] = {}

    def add(
        self,
        *,
        score: str,
        quantity: str,
        q: float,
        estimate: float,
        draws: FloatArray | None,
        codes: Sequence[str] = (),
        withheld: str | None = None,
        rows_in_top: float = float("nan"),
        tied: bool | None = None,
        tie_share: float = float("nan"),
        mean_score: float = float("nan"),
        band: tuple[float, float] = (float("nan"), float("nan")),
        p_value: bool = False,
    ) -> tuple[str, bool]:
        se = float("nan") if draws is None or withheld else _spread(draws)
        if withheld is None and self.few_clusters and draws is not None:
            withheld = WITHHELD["few_clusters"].format(guard=TARGETING_MIN_CLUSTERS)
        if withheld is None and math.isfinite(se) and se <= self.zero_spread:
            # A score that ties every row draws one value up to rounding: an interval of
            # width 1e-16 would claim a precision nothing measured.
            withheld = WITHHELD["degenerate"]
        if withheld is not None:
            se = float("nan")
        low = estimate - self.z * se if math.isfinite(se) else float("nan")
        high = estimate + self.z * se if math.isfinite(se) else float("nan")
        p = (
            two_sided_pvalue(estimate, se)
            if p_value and math.isfinite(se) and se > self.zero_spread
            else float("nan")
        )
        row_codes = list(dict.fromkeys(codes))
        texts = [_reason_text(code) for code in row_codes]
        if withheld is not None:
            texts.append(withheld)
        texts += [_reason_text(code) for code in self.config_codes]
        status = "experimental" if (row_codes or self.config_codes or withheld) else "supported"
        if withheld is not None and withheld.startswith("no interval: TOC(1)"):
            status = "experimental" if (row_codes or self.config_codes) else "supported"
        values = {
            "score": score,
            "quantity": quantity,
            "q": float(q),
            "estimate": float(estimate),
            "std_error": se,
            "lower": low,
            "upper": high,
            "band_lower": band[0],
            "band_upper": band[1],
            "p_value": p,
            "rows_in_top": float(rows_in_top),
            "tied_cut": bool(tied) if tied is not None else False,
            "tie_share": float(tie_share),
            "mean_score": float(mean_score),
            "status": status,
            "status_reason": "; ".join(dict.fromkeys(texts)),
        }
        for column in COLUMNS:
            self.columns[column].append(values[column])
        for text in [_reason_text(code) for code in row_codes]:
            if text not in self.reasons:
                self.reasons.append(text)
        if (
            withheld is not None
            and not withheld.startswith("no interval: TOC(1)")
            and withheld not in self.reasons
        ):
            self.reasons.append(withheld)
        return status, withheld is not None

    def score_rows(
        self,
        name: str,
        values: Mapping[tuple[str, float], float],
        draws: Mapping[tuple[str, float], FloatArray],
        facts: Mapping[float, _CutFact],
        atoms: bool,
        grid: Sequence[float],
        edges: Sequence[float],
        score: FloatArray,
        layout: _Layout,
        mass: FloatArray,
    ) -> float:
        point_state: dict[float, tuple[str, bool, tuple[str, ...]]] = {}
        below = [q for q in grid if q < 1.0]
        for q in grid:
            fact = facts[q]
            point_codes: list[str] = []
            withheld = None
            if q < 1.0:
                if fact.rows_in_top < TARGETING_MIN_TOP_ROWS:
                    withheld = WITHHELD["top_rows"].format(guard=TARGETING_MIN_TOP_ROWS)
                elif fact.rows_in_top < TARGETING_MEASURED_TOP_ROWS:
                    point_codes.append("few_rows")
                if fact.tied:
                    point_codes.append(fact.code)
            point_state[q] = ("", withheld is not None, tuple(point_codes))
        # The band over the grid points below 1, from the same draws.
        se = np.array([_spread(draws[("toc", q)]) for q in below])
        band_withheld = (
            self.few_clusters
            or any(point_state[q][1] for q in below)
            or not np.all(np.isfinite(se))
            or bool(np.any(se <= self.zero_spread))
        )
        critical = float("nan")
        if below and not band_withheld:
            from .inference.bootstrap import studentized_critical_value

            matrix = np.column_stack([draws[("toc", q)] for q in below])
            estimates = np.array([values[("toc", q)] for q in below])
            critical, _ = studentized_critical_value(
                estimates, matrix, np.broadcast_to(se, matrix.shape), self.alpha
            )
        for q in grid:
            fact = facts[q]
            _, withheld_flag, codes = point_state[q]
            if q < 1.0:
                withheld = (
                    WITHHELD["top_rows"].format(guard=TARGETING_MIN_TOP_ROWS)
                    if withheld_flag
                    else None
                )
                j = below.index(q)
                band = (
                    (values[("toc", q)] - critical * se[j], values[("toc", q)] + critical * se[j])
                    if math.isfinite(critical)
                    else (float("nan"), float("nan"))
                )
                self.add(
                    score=name,
                    quantity="toc",
                    q=q,
                    estimate=values[("toc", q)],
                    draws=draws[("toc", q)],
                    codes=codes,
                    withheld=withheld,
                    rows_in_top=fact.rows_in_top,
                    tied=fact.tied,
                    tie_share=fact.share,
                    band=band,
                )
            else:
                self.add(
                    score=name,
                    quantity="toc",
                    q=q,
                    estimate=0.0,
                    draws=None,
                    withheld=WITHHELD["anchor"],
                    rows_in_top=fact.rows_in_top,
                    tied=False,
                )
        for q in grid:
            fact = facts[q]
            _, withheld_flag, codes = point_state[q]
            self.add(
                score=name,
                quantity="gain",
                q=q,
                estimate=values[("gain", q)] if q < 1.0 else values[("ate", 1.0)],
                draws=draws[("gain", q)] if q < 1.0 else draws[("ate", 1.0)],
                codes=codes,
                withheld=(
                    WITHHELD["top_rows"].format(guard=TARGETING_MIN_TOP_ROWS)
                    if withheld_flag
                    else None
                ),
                rows_in_top=fact.rows_in_top,
                tied=fact.tied if q < 1.0 else False,
                tie_share=fact.share,
            )
        previous = 0.0
        order = layout.order
        m = _block_sums(mass[order], layout.starts)
        sums_score = _block_sums(mass[order] * score[order], layout.starts)
        weights = block_weights(m, grid, edges)
        for edge in edges:
            involved = [q for q in (previous, edge) if 0.0 < q < 1.0]
            codes = tuple(code for q in involved for code in point_state[q][2])
            withheld = (
                WITHHELD["top_rows"].format(guard=TARGETING_MIN_TOP_ROWS)
                if any(point_state[q][1] for q in involved)
                else None
            )
            group_weight = weights[("group_effect", edge)]
            mean_score = float(np.sum(group_weight * sums_score) / np.sum(group_weight * m))
            self.add(
                score=name,
                quantity="group_effect",
                q=edge,
                estimate=values[("group_effect", edge)],
                draws=draws[("group_effect", edge)],
                codes=codes,
                withheld=withheld,
                mean_score=mean_score,
            )
            previous = edge
        for quantity in ("autoc", "qini"):
            self.add(
                score=name,
                quantity=quantity,
                q=1.0,
                estimate=values[(quantity, 1.0)],
                draws=draws[(quantity, 1.0)],
                codes=("atoms",) if atoms else (),
                p_value=True,
            )
        self.toc_status[name] = {
            q: (
                "experimental" if point_state[q][2] else "supported",
                point_state[q][1],
            )
            for q in grid
        }
        return critical

    def difference_rows(
        self,
        names: Sequence[str],
        values: Mapping[str, Mapping[tuple[str, float], float]],
        draws: Mapping[str, Mapping[tuple[str, float], FloatArray]],
        grid: Sequence[float],
        *,
        pair_codes: tuple[str, ...] = (),
    ) -> None:
        first, second = names
        label = f"{first} - {second}"
        for q in grid:
            if q >= 1.0:
                continue
            states = [self.toc_status[name][q] for name in names]
            codes: list[str] = []
            for name in names:
                for index in range(len(self.columns["score"])):
                    if (
                        self.columns["score"][index] == name
                        and self.columns["quantity"][index] == "toc"
                        and self.columns["q"][index] == q
                    ):
                        if self.columns["tied_cut"][index]:
                            share = self.columns["tie_share"][index]
                            codes.append("tied_cut" if math.isfinite(share) else "tied_edge")
                        if (
                            TARGETING_MIN_TOP_ROWS
                            <= self.columns["rows_in_top"][index]
                            < TARGETING_MEASURED_TOP_ROWS
                        ):
                            codes.append("few_rows")
            withheld = (
                WITHHELD["top_rows"].format(guard=TARGETING_MIN_TOP_ROWS)
                if any(state[1] for state in states)
                else None
            )
            self.add(
                score=label,
                quantity="toc",
                q=q,
                estimate=values[first][("toc", q)] - values[second][("toc", q)],
                draws=draws[first][("toc", q)] - draws[second][("toc", q)],
                codes=[*codes, *pair_codes],
                withheld=withheld,
                p_value=True,
            )
        for quantity in ("autoc", "qini"):
            codes = []
            for name in names:
                for index in range(len(self.columns["score"])):
                    if (
                        self.columns["score"][index] == name
                        and self.columns["quantity"][index] == quantity
                        and self.columns["status_reason"][index].startswith(_reason_text("atoms"))
                    ):
                        codes.append("atoms")
            self.add(
                score=label,
                quantity=quantity,
                q=1.0,
                estimate=values[first][(quantity, 1.0)] - values[second][(quantity, 1.0)],
                draws=draws[first][(quantity, 1.0)] - draws[second][(quantity, 1.0)],
                codes=[*codes, *pair_codes],
                p_value=True,
            )


def control_values(
    curve: TargetingCurve,
    *,
    n_half_samples: int | None = None,
    random_state: int | None = None,
    **mutation: Any,
) -> dict[str, dict[tuple[str, float], tuple[float, float]]]:
    """Each quantity's point and half-sample SE under a deliberate mutation of the draws.

    For the study's controls and the unit tests' mutation checks: the keyword arguments are
    those of :func:`half_sample_draws`.  The point is the targeted-score form
    :math:`N^{-1}\\sum w c \\Gamma^*` under the same rule, which equals the reported
    substitution estimate when the rule is the shipped one.

    Parameters
    ----------
    curve : TargetingCurve
        A fitted curve.
    n_half_samples : int or None
        Draws; the curve's own count when omitted.
    random_state : int or None
        Seed of the draws.
    **mutation : Any
        ``scores``, ``cut``, ``ranks``, ``tie_rule`` and ``paired``.

    Returns
    -------
    dict
        By score name, ``(quantity, q)`` to ``(point, standard error)``.
    """
    evaluation = curve.evaluation
    if evaluation is None:
        raise DataError("this curve keeps no evaluation rows")
    draws = half_sample_draws(
        evaluation,
        n_half_samples=n_half_samples or curve.n_half_samples,
        random_state=random_state,
        **mutation,
    )
    tie_rule = mutation.get("tie_rule", "average")
    out: dict[str, dict[tuple[str, float], tuple[float, float]]] = {}
    for name in evaluation.scores:
        points = _point_values(
            evaluation.layout(name),
            evaluation.targeted
            if mutation.get("scores", "targeted") == "targeted"
            else evaluation.one_step,
            evaluation.mass,
            evaluation.grid,
            evaluation.edges,
            tie_rule=tie_rule,
        )
        out[name] = {key: (points[key], _spread(values)) for key, values in draws[name].items()}
    return out
