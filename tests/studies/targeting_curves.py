r"""Declaration and driver of the ``targeting-curves`` study (X92 PR-5 plan, section 10).

The study measures the TMLE-targeted targeting curve of
:meth:`cleverly.study.IdentifiedEffect.targeting_curve` and of
``curve.diagnostics.targeting_curve()``: interval calibration of the TOC at fixed treated
fractions, the AUTOC, the Qini and two group effects; the joint coverage, size and power of the
sup-t band; the paired difference of two scores; a binary outcome, a known mechanism with a
misspecified outcome regression, an atomic score and a declared split; and controls that must
fail.  Everything a run reads is fixed here before it starts:

- the laws, as samplers with the plan's coefficients (:func:`sample`);
- the exact truths (:func:`truth`), by Gauss--Legendre quadrature with the score's indicator
  integrated in closed form along ``W3``, and on cell ``S`` by a fixed scrambled Sobol set over
  the fitted curve of each replicate (:func:`split_truth`);
- the fit groups (:data:`GROUPS`), each a law, ``n``, ``R`` and the cells its fits fill, with
  the controls that share its fits;
- the design numbers (:func:`design_numbers`): each calibration row's pass probability, the
  family-wise rate, the band's power and the size of cell ``P``'s slope, and each control's
  displacement and detection probability, computed from the influence functions of the rank
  functionals (:func:`influence_sd`) rather than copied;
- the study record (:data:`STUDY`) with its margins, failure bound and publication policy.

The run is ``python -m tests.canonical.targeting_curves.regenerate`` (the declared form of
:mod:`tests.canonical.declared_run`); its docstring holds the red-cell reading.  The pilot
(:mod:`tests.studies.targeting_curves_pilot`) runs before the declaration is final and checks
what the design cannot compute in closed form: the in-sample control's displacement, the
seconds per fit, and the kurtosis and SE spread the sizing assumes.

Deviations from the plan's table, recorded here so the run reads them:

- the split curve of cell ``S`` is ``BSpline(["W1", "W2", "W3"], n_interior_knots=5)``: one
  intercept and eight terms per modifier, 25 terms, the nearest to the plan's 24 an additive
  B-spline basis of three modifiers can have;
- a calibration row reads the bootstrap interval the curve reports, ``estimate +- z * SE``,
  with ``SE`` the SD of the 2,000 half-sample draws.
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy import optimize, stats
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly import targeting_curves as tc
from cleverly.bases import BSpline, Modifiers
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "targeting-curves"

#: The primary seed and the resampling seed of the summaries.
SEED = 20261107
RESAMPLING_SEED = 20261108

#: Half-sample draws of every fit (plan 5.1).
N_HALF_SAMPLES = 2_000

#: The two-sided level of every interval.
CRITICAL = float(stats.norm.ppf(0.975))

#: Declared outcome bounds of every continuous fit, wide enough that no draw reaches them.
Q_BOUNDS = (-12.0, 12.0)

#: Gauss--Legendre nodes per piece of the exact truths.
QUADRATURE_NODES = 40

#: Points of the scrambled Sobol set of cell ``S``'s truth (plan: ``2^20``).
SOBOL_POINTS = 2**20
SOBOL_SEED = 20261109

#: The score coefficients of ``W3``: ``S = W1 + c W3``.
C_PRIMARY = 0.5
C_SECOND = 1.5

GRID = tc.DEFAULT_GRID


# ---------------------------------------------------------------------------- the laws


def propensity(w: np.ndarray) -> np.ndarray:
    """``g(W) = expit(0.5 W1 - 0.5 W2 + 0.25 W3)``, every law's mechanism."""
    return expit(0.5 * w[:, 0] - 0.5 * w[:, 1] + 0.25 * w[:, 2])


def binary_mean(w: np.ndarray, a: Any) -> np.ndarray:
    """``Qbar(a, W) = expit(-0.2 + 0.4 W2 + a (0.3 + 1.2 W1))``, the binary laws' outcome."""
    return expit(-0.2 + 0.4 * w[:, 1] + np.asarray(a) * (0.3 + 1.2 * w[:, 0]))


@dataclass(frozen=True)
class Law:
    """One law: the outcome, the effect, the noise and the score.

    Parameters
    ----------
    label : str
        The law's key.
    binary : bool
        Whether the outcome is binary (:func:`binary_mean`), else continuous.
    effect : str
        ``"linear"``: ``tau = 1 + beta W1``; ``"interaction"``: ``tau = 1 + W1 + W1 W3``;
        ``"constant"``: ``tau = 1``.  A binary law's effect is its arm difference.
    sigma : float
        The noise SD of a continuous law.
    beta : float
        The slope of a ``"linear"`` effect.
    score : str
        ``"linear"``: ``W1 + c W3``; ``"w3"``: ``W3``; ``"atoms"``: ``round(2 W1) / 2``.
    c : float
        The ``W3`` coefficient of a linear score.
    """

    label: str
    binary: bool
    effect: str = "linear"
    sigma: float = 0.25
    beta: float = 1.0
    score: str = "linear"
    c: float = C_PRIMARY

    def tau(self, w: np.ndarray) -> np.ndarray:
        """The conditional effect at ``w``."""
        if self.binary:
            return binary_mean(w, 1) - binary_mean(w, 0)
        if self.effect == "constant":
            return np.ones(len(w))
        if self.effect == "interaction":
            return 1.0 + w[:, 0] + w[:, 0] * w[:, 2]
        return 1.0 + self.beta * w[:, 0]

    def base(self, w: np.ndarray) -> np.ndarray:
        """The control-arm mean of a continuous law."""
        return 0.5 * w[:, 0] - 0.5 * w[:, 1] + w[:, 2]

    def score_of(self, w: np.ndarray) -> np.ndarray:
        """The score at ``w``."""
        if self.score == "w3":
            return w[:, 2].copy()
        if self.score == "atoms":
            return np.round(2.0 * w[:, 0]) / 2.0
        return w[:, 0] + self.c * w[:, 2]

    def gamma_variance(self, w: np.ndarray) -> np.ndarray:
        """``Var(Gamma | W)`` of the efficient score at ``w``."""
        g = propensity(w)
        if self.binary:
            q1, q0 = binary_mean(w, 1), binary_mean(w, 0)
            return q1 * (1 - q1) / g + q0 * (1 - q0) / (1 - g)
        return self.sigma**2 * (1.0 / g + 1.0 / (1.0 - g))


LAWS: Mapping[str, Law] = {
    "H": Law("H", binary=False),
    "B": Law("B", binary=True),
    "K": Law("K", binary=False, effect="interaction"),
    "U": Law("U", binary=True, score="w3"),
    "N": Law("N", binary=False, effect="constant", sigma=0.5),
    "D": Law("D", binary=True, score="atoms"),
}


def power_law(beta: float) -> Law:
    """Cell ``P``'s law at slope ``beta``."""
    return Law("P", binary=False, sigma=1.0, beta=beta)


def sample(law: Law, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n`` rows of ``law``: ``Y``, ``A``, ``W1``-``W3``, the score ``S``, a second score
    ``S_B = W1 + 1.5 W3`` and the mechanism ``g`` (cell ``K``'s known treatment probability)."""
    w = rng.uniform(-1.0, 1.0, (n, 3))
    g = propensity(w)
    a = rng.binomial(1, g)
    if law.binary:
        y = rng.binomial(1, binary_mean(w, a)).astype(float)
    else:
        y = law.base(w) + a * law.tau(w) + law.sigma * rng.normal(size=n)
    return pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W1": w[:, 0],
            "W2": w[:, 1],
            "W3": w[:, 2],
            "S": law.score_of(w),
            "S_B": w[:, 0] + C_SECOND * w[:, 2],
            "p1": g,
            "p0": 1.0 - g,
        }
    )


# --------------------------------------------------------------------- the exact truths


def _nodes(low: float, high: float, k: int = QUADRATURE_NODES) -> tuple[np.ndarray, np.ndarray]:
    x, weight = leggauss(k)
    half = 0.5 * (high - low)
    return low + half * (x + 1.0), half * weight


def _w2_nodes() -> tuple[np.ndarray, np.ndarray]:
    x, weight = _nodes(-1.0, 1.0)
    return x, weight / 2.0


def _above(law: Law, s: float, f: Callable[[np.ndarray], np.ndarray]) -> float:
    """``E[f(W) 1{score >= s}]`` exactly, for a linear or ``W3`` score.

    The indicator is integrated in closed form along ``W3``: the inner integral runs from
    ``max(-1, (s - W1) / c)`` to 1, and the outer ``W1`` integral breaks where that bound hits
    ``-1`` or ``1``, so every piece is smooth and Gauss--Legendre is exact to rounding.
    """
    w2, m2 = _w2_nodes()
    if law.score == "w3":
        breaks = [-1.0, 1.0]
        c = 0.0
    else:
        c = law.c
        breaks = sorted({-1.0, 1.0, *[b for b in (s - c, s + c) if -1.0 < b < 1.0]})
    x, base = leggauss(QUADRATURE_NODES)
    total = 0.0
    for left, right in itertools.pairwise(breaks):
        w1, m1 = _nodes(left, right)
        low = np.full(len(w1), s) if law.score == "w3" else (s - w1) / c
        low = np.clip(low, -1.0, 1.0)
        half = 0.5 * (1.0 - low)
        w3 = low[:, None] + half[:, None] * (x[None, :] + 1.0)
        m3 = half[:, None] * base[None, :]
        k1, k2, k3 = len(w1), len(w2), len(x)
        w = np.empty((k1, k2, k3, 3))
        w[..., 0] = w1[:, None, None]
        w[..., 1] = w2[None, :, None]
        w[..., 2] = w3[:, None, :]
        weight = (m1[:, None, None] / 2.0) * m2[None, :, None] * (m3[:, None, :] / 2.0)
        total += float(np.sum(weight.reshape(-1) * f(w.reshape(-1, 3))))
    return total


def _box(f: Callable[[np.ndarray], np.ndarray], w1: tuple[float, float]) -> float:
    """``E[f(W) 1{W1 in w1}]`` exactly, by tensor Gauss--Legendre."""
    x1, m1 = _nodes(*w1)
    x2, m2 = _w2_nodes()
    x3, m3 = _w2_nodes()
    grid = np.array(np.meshgrid(x1, x2, x3, indexing="ij")).reshape(3, -1).T
    weight = np.einsum("i,j,k->ijk", m1 / 2.0, m2, m3).reshape(-1)
    return float(np.sum(weight * f(grid)))


#: The atoms of ``round(2 W1) / 2`` in decreasing score order: ``W1`` interval and value.
ATOMS = (
    ((0.75, 1.0), 1.0),
    ((0.25, 0.75), 0.5),
    ((-0.25, 0.25), 0.0),
    ((-0.75, -0.25), -0.5),
    ((-1.0, -0.75), -1.0),
)


class Curve:
    """The population curve of one law: ``G(u) = E[tau 1{U <= u}]`` and what it determines.

    ``U`` is the top fraction of the score; at an atom the block is split in proportion, as
    the estimator's tie rule does (D3).
    """

    def __init__(self, law: Law) -> None:
        self.law = law
        self.ate = _box(law.tau, (-1.0, 1.0))
        if law.score == "atoms":
            self.blocks = [
                (_box(lambda w: np.ones(len(w)), span), _box(law.tau, span)) for span, _ in ATOMS
            ]

    def _cut(self, u: float) -> float:
        def excess(s: float) -> float:
            return _above(self.law, s, lambda w: np.ones(len(w))) - u

        span = 1.0 if self.law.score == "w3" else 1.0 + self.law.c
        return float(optimize.brentq(excess, -span, span, xtol=1e-14, rtol=1e-14))

    def gain(self, u: float) -> float:
        """``G(u)``."""
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return self.ate
        if self.law.score == "atoms":
            total, used = 0.0, 0.0
            for mass, mean in self.blocks:
                take = min(mass, u - used)
                total += mean * take / mass
                used += take
                if used >= u - 1e-15:
                    break
            return total
        return _above(self.law, self._cut(u), self.law.tau)

    def toc(self, q: float) -> float:
        """``TOC(q) = G(q) / q - ATE``."""
        return self.gain(q) / q - self.ate

    def _breaks(self) -> list[float]:
        if self.law.score == "atoms":
            return [0.0, *np.cumsum([mass for mass, _ in self.blocks]).tolist()]
        if self.law.score == "w3":
            return [0.0, 1.0]
        c = self.law.c
        inner = [_above(self.law, s, lambda w: np.ones(len(w))) for s in (1.0 - c, c - 1.0)]
        return sorted({0.0, 1.0, *[u for u in inner if 0.0 < u < 1.0]})

    def area(self, kind: str) -> float:
        """``AUTOC = int_0^1 TOC(u) du`` or ``Qini = int_0^1 u TOC(u) du``."""
        total = 0.0
        breaks = self._breaks()
        for left, right in itertools.pairwise(breaks):
            u, m = _nodes(left, right, 24)
            values = np.array([self.toc(float(x)) for x in u])
            total += float(np.sum(m * (values if kind == "autoc" else u * values)))
        return total

    def group(self, low: float, high: float) -> float:
        """The mean effect of the ranks in ``(low, high]``."""
        return (self.gain(high) - self.gain(low)) / (high - low)


@functools.cache
def _truth(law: Law) -> tuple[tuple[str, float], ...]:
    curve = Curve(law)
    out = {row_key("toc", q): curve.toc(q) for q in GRID if q < 1.0}
    out[row_key("autoc")] = curve.area("autoc")
    out[row_key("qini")] = curve.area("qini")
    out[row_key("group_effect", 0.2)] = curve.group(0.0, 0.2)
    out[row_key("group_effect", 1.0)] = curve.group(0.8, 1.0)
    out["ate"] = curve.ate
    return tuple(out.items())


def truth(law: Law) -> dict[str, float]:
    """Every reported quantity's truth under ``law``, keyed as :func:`row_key` names them.

    Cached per law: a law's truth takes a few seconds of quadrature and every replicate reads it.
    """
    return dict(_truth(law))


def row_key(quantity: str, q: float = 1.0, score: str = "S") -> str:
    """The name of one reported quantity of one score: ``toc[0.5]``, ``autoc``, ``S - S_B:qini``."""
    name = quantity if quantity in ("autoc", "qini") else f"{quantity}[{q:g}]"
    return name if score == "S" else f"{score}:{name}"


#: The paired difference's score name, as :class:`~cleverly.targeting_curves.TargetingCurve`
#: names it.
DIFFERENCE = "S - S_B"

#: The quantities of cell ``C``: TOC differences at 0.1 and 0.5 and the two area differences.
PAIRED = tuple(
    row_key(quantity, q, DIFFERENCE)
    for quantity, q in (("toc", 0.1), ("toc", 0.5), ("autoc", 1.0), ("qini", 1.0))
)


@functools.cache
def paired_truth() -> dict[str, float]:
    """Cell ``C``'s truths: the TOC and areas of ``S`` minus those of ``S_B``, on ``H``'s law."""
    primary = truth(LAWS["H"])
    second = truth(Law("H_B", binary=False, c=C_SECOND))
    return {name: primary[name.split(":")[1]] - second[name.split(":")[1]] for name in PAIRED}


def _sobol() -> np.ndarray:
    return 2.0 * stats.qmc.Sobol(3, scramble=True, seed=SOBOL_SEED).random(SOBOL_POINTS) - 1.0


def split_truth(
    score: Callable[[pd.DataFrame], np.ndarray], *, points: np.ndarray | None = None
) -> dict[str, float]:
    """Cell ``S``'s truths for one fitted curve, on the shared Sobol set.

    ``score`` maps a frame of ``W1``-``W3`` to the curve's predictions.  The Sobol set stands
    for ``U(-1, 1)^3``; the TOC, AUTOC and Qini are the rank functionals of :data:`GRID` over
    it, with the AUTOC and Qini read at every point's rank (the ``2^20``-point form of (2.8)).
    """
    w = _sobol() if points is None else points
    law = LAWS["H"]
    chunk = 2**18
    values = np.concatenate(
        [
            np.asarray(
                score(pd.DataFrame(w[start : start + chunk], columns=["W1", "W2", "W3"])),
                dtype=float,
            )
            for start in range(0, len(w), chunk)
        ]
    )
    tau = law.tau(w)
    order = np.argsort(-values, kind="stable")
    m = len(w)
    weights = tc.block_weights(np.ones(m), GRID)
    ranked = tau[order]
    out = {
        row_key("toc", q): float(np.sum(weights[("toc", q)] * ranked) / m) for q in GRID if q < 1.0
    }
    out[row_key("autoc")] = float(np.sum(weights[("autoc", 1.0)] * ranked) / m)
    out[row_key("qini")] = float(np.sum(weights[("qini", 1.0)] * ranked) / m)
    return out


# ------------------------------------------------------------------ the groups and cells

CALIBRATION = "interval_calibration"
BIAS = "bias"
JOINT = "simultaneous_coverage"
TYPE_I = "type_i_error"
POWER = "power"

POSITIVE = "correctly_specified"
#: Cell ``K``'s positive kind: its outcome regression is misspecified on purpose and its
#: mechanism is known, so ``correctly_specified`` would be a false claim.
KNOWN_MECHANISM = "known_mechanism"
#: The kinds of a calibration positive.
POSITIVE_KINDS = (POSITIVE, KNOWN_MECHANISM)

#: Replications of every group but ``P`` (plan section 10, sizing).
R_MAIN = 3_600
#: Replications of the band-power group.
R_POWER = 800

#: The side of cell ``K``'s untargeted-draws control on AUTOC: the one-step draws' SD falls
#: 20 to 25% short of the targeted draws' (:func:`k_untargeted_ratio`, which the design test
#: recomputes).  The control reads AUTOC only: at TOC(0.5) and Qini the ratio is 1 to within
#: 2%, inside the band, so no control row there could leave it.
K_UNTARGETED_SIDE = "shrunken"


@dataclass(frozen=True)
class Group:
    """One declared fit group: a law, a size, its replications, and the rows its fits fill.

    Parameters
    ----------
    label : str
        The group key, which the seed stream and the cell names read.
    law : str
        A key of :data:`LAWS`, or ``"P"``.
    n : int
        Rows per replication (evaluation rows; cell ``S`` draws twice as many).
    replicates : int
        Replications in the declared run.
    rows : tuple of str
        The quantities read as calibration positives (:func:`row_key` names).
    controls : Mapping of str to tuple of str
        Each control the group's fits compute, and the quantities it reads.
    band : str or None
        ``"coverage"``, ``"type_i"`` or ``"power"``: the band row the group fills.
    seconds : float
        Single-thread seconds per replication, for the cost: measured on the smoke run with
        its controls (cell ``S`` includes its per-replicate Sobol truth and the in-sample
        refit); the pilot remeasures them.
    positive : str
        The kind of its calibration positives: :data:`POSITIVE`, or :data:`KNOWN_MECHANISM`
        for cell ``K``.
    """

    label: str
    law: str
    n: int
    replicates: int
    rows: tuple[str, ...]
    controls: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    band: str | None = None
    seconds: float = 0.6
    positive: str = POSITIVE


#: The quantities of the TOC at three fractions and the two areas.
CORE = (row_key("toc", 0.1), row_key("toc", 0.5), row_key("toc", 0.9), "autoc", "qini")

#: The groups of section 10, in run order.  ``C`` is ``H``'s second score on the same fits.
GROUPS: tuple[Group, ...] = (
    Group(
        "H",
        "H",
        2_000,
        R_MAIN,
        (*CORE, row_key("group_effect", 0.2), row_key("group_effect", 1.0), *PAIRED),
        controls={
            "fixed_cut": (row_key("toc", 0.5), row_key("toc", 0.9)),
            "fixed_ranks": ("autoc",),
            "pointwise_joint": ("band",),
            "unpaired": (row_key("toc", 0.1, DIFFERENCE), row_key("autoc", score=DIFFERENCE)),
        },
        band="coverage",
        seconds=2.3,
    ),
    Group("B", "B", 2_000, R_MAIN, CORE, seconds=0.8),
    Group(
        "K",
        "K",
        2_000,
        R_MAIN,
        (row_key("toc", 0.5), "autoc", "qini"),
        controls={"untargeted": ("autoc",)},
        seconds=1.0,
        positive=KNOWN_MECHANISM,
    ),
    Group("P", "P", 1_000, R_POWER, (), band="power", seconds=0.4),
    Group(
        "U",
        "U",
        2_000,
        R_MAIN,
        (row_key("toc", 0.1), row_key("toc", 0.5), "autoc", "qini"),
        band="type_i",
        seconds=0.7,
    ),
    Group("N", "N", 2_000, R_MAIN, ("autoc", "qini"), band="type_i", seconds=0.7),
    Group(
        "D",
        "D",
        2_000,
        R_MAIN,
        (row_key("toc", 0.2), row_key("toc", 0.5), "autoc", "qini"),
        controls={"whole_block": (row_key("toc", 0.2),)},
        seconds=0.4,
    ),
    Group(
        "S",
        "H",
        2_000,
        R_MAIN,
        CORE,
        controls={"in_sample": (row_key("toc", 0.1), "autoc")},
        seconds=8.0,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}

#: What each control computes from its group's fit, and the rule it must fail.
CONTROL_CLAIMS: Mapping[str, str] = {
    "fixed_cut": "the draws reuse the full-sample cut: the SE-ratio interval must lie above the "
    "band (the fixed-threshold variance exceeds the quantile one at q = 0.5 and 0.9)",
    "fixed_ranks": "the draws reuse the full-sample rank weights: the SE-ratio interval must lie "
    "outside the band on the side design_numbers computes",
    "pointwise_joint": "the pointwise intervals read jointly over the grid: joint coverage must "
    "lie below nominal",
    "unpaired": "the second score drawn from its own stream: the SE-ratio interval of the "
    "difference must lie above the band",
    "untargeted": "the draws average the one-step scores while the TMLE is reported: the "
    "SE-ratio interval must lie below the band (k_untargeted_ratio computes the side)",
    "whole_block": "the rule S >= cut puts the whole tied block in the top fraction: the bias "
    "must lie wholly outside +-0.25 of its own SD",
    "in_sample": "the curve fitted on the evaluation rows themselves: the bias must lie wholly "
    "outside +-0.25 of its own SD",
}


def _label(name: str) -> str:
    return name.replace("[", "_").replace("]", "").replace(" - ", "_minus_").replace(":", "_")


def calibration_cell(group: str, name: str, kind: str = POSITIVE) -> str:
    """The ``interval_calibration`` cell of one quantity of one group."""
    return f"{group}_{_label(name)}__{kind}"


def is_positive(cell: str) -> bool:
    """Whether a calibration cell is a positive (of either :data:`POSITIVE_KINDS`)."""
    return cell.rsplit("__", 1)[-1] in POSITIVE_KINDS


def se_control_kind(control: str) -> str | None:
    """The calibration-control kind a SE control's rows answer to, or ``None`` before the pilot."""
    if control in ("fixed_cut", "unpaired"):
        return "inflated_se_control"
    if control == "fixed_ranks":
        return FIXED_RANKS_KIND
    if control == "untargeted":
        return f"{K_UNTARGETED_SIDE}_se_control"
    return None


#: The side the fixed-rank AUTOC control leaves the band on, from :func:`influence_sd` on
#: ``H`` (``design_numbers()["controls"]["H_autoc__fixed_ranks"]["ratio"]``; the unit test
#: recomputes it).
FIXED_RANKS_KIND = "inflated_se_control"


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells: dict[str, list[str]] = {CALIBRATION: [], BIAS: [], JOINT: [], TYPE_I: [], POWER: []}
    cells[CALIBRATION] += [
        calibration_cell(group.label, name, group.positive) for name in group.rows
    ]
    for control, names in group.controls.items():
        if control in ("whole_block", "in_sample"):
            cells[BIAS] += [f"{group.label}_{_label(name)}__{control}" for name in names]
        elif control == "pointwise_joint":
            cells[JOINT].append(f"{group.label}_band__{control}")
        else:
            kind = se_control_kind(control)
            if kind is not None:
                cells[CALIBRATION] += [
                    calibration_cell(group.label, f"{control}_{name}", kind) for name in names
                ]
    if group.band == "coverage":
        cells[JOINT].append(f"{group.label}_band__{POSITIVE}")
    elif group.band == "type_i":
        cells[TYPE_I].append(f"{group.label}_band__null")
    elif group.band == "power":
        cells[POWER].append(f"{group.label}_band__slope")
    return {family: tuple(names) for family, names in cells.items() if names}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every declared property cell, by family, in group order."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, names in group_cells(group).items():
            out.setdefault(family, []).extend(names)
    return {family: tuple(names) for family, names in out.items()}


PROPERTY_CELLS = property_cells()


# ---------------------------------------------------------------------------- the fits


def spline(binary: bool) -> Any:
    """The correctly specified spline GLM of an arm: cubic splines, then a GLM."""
    final = LogisticRegression(max_iter=2_000) if binary else LinearRegression()
    return make_pipeline(SplineTransformer(n_knots=5, degree=3), final)


def learners(group: Group, law: Law) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``group``."""
    if group.label == "K":
        settings: dict[str, Any] = {"outcome_learner": LinearRegression(), "outcome_by_arm": True}
    else:
        settings = {
            "outcome_learner": spline(law.binary),
            "treatment_learner": LogisticRegression(max_iter=2_000),
            "outcome_by_arm": True,
        }
    if not law.binary:
        settings["q_bounds"] = Q_BOUNDS
    return settings


#: The split curve of cell ``S``.
SPLIT_MODIFIERS = Modifiers(
    ["W1", "W2", "W3"], domains={"W1": (-1.0, 1.0), "W2": (-1.0, 1.0), "W3": (-1.0, 1.0)}
)
SPLIT_BASIS = BSpline(["W1", "W2", "W3"], n_interior_knots=5)
#: The basis cell ``S`` moves to, on record and before the run, if the pilot's in-sample
#: displacement falls short of detection 0.90 on :data:`SPLIT_BASIS`.  It has 34 terms.
SPLIT_FALLBACK_BASIS = BSpline(["W1", "W2", "W3"], n_interior_knots=8)


def law_of(group: Group, slope: float | None = None) -> Law:
    """The law of ``group``; cell ``P`` reads its declared slope."""
    if group.law == "P":
        return power_law(P_SLOPE if slope is None else slope)
    return LAWS[group.law]


def _study(group: Group, frame: pd.DataFrame) -> CausalStudy:
    roles: dict[str, Any] = {}
    if group.label == "K":
        roles["treatment_probabilities"] = {0: "p0", 1: "p1"}
    return CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), **roles),
    )


def _band(curve: Any, truths: Mapping[str, float], score: str) -> dict[str, float]:
    """The band's reading: whether it covers every grid truth, and whether it excludes 0."""
    rows = [row for row in curve.rows("toc", score) if row["q"] < 1.0]
    critical = float(curve.critical_value)
    covered = all(
        row["estimate"] - critical * row["std_error"]
        <= truths[row_key("toc", row["q"])]
        <= row["estimate"] + critical * row["std_error"]
        for row in rows
    )
    pointwise = all(
        row["lower"] <= truths[row_key("toc", row["q"])] <= row["upper"] for row in rows
    )
    rejected = any(row["band_lower"] > 0.0 or row["band_upper"] < 0.0 for row in rows)
    return {
        "covered": float(covered),
        "pointwise_covered": float(pointwise),
        "rejected": float(rejected),
        "statistic": max(abs(row["estimate"]) / row["std_error"] for row in rows),
        "critical": critical,
    }


def _rows_of(curve: Any, score: str) -> dict[str, dict[str, float]]:
    out = {}
    for quantity in ("toc", "gain", "group_effect", "autoc", "qini"):
        for row in curve.rows(quantity, score):
            name = row_key(quantity, row["q"], "S" if score == curve.score else score)
            out[name] = {"estimate": row["estimate"], "std_error": row["std_error"]}
    return out


#: Half-sample draws of each SE control.  A control needs only to leave the band, which the
#: design's ratios (1.24 and above) do by a wide margin; 500 draws put the Monte Carlo CV of
#: its SE near 3%, which :func:`se_control_detection` carries.
CONTROL_HALF_SAMPLES = 500

#: The draws' mutation of each SE control (:func:`cleverly.targeting_curves.half_sample_draws`).
SE_MUTATIONS: Mapping[str, Mapping[str, Any]] = {
    "fixed_cut": {"cut": "fixed"},
    "fixed_ranks": {"ranks": "fixed"},
    "unpaired": {"paired": False},
    "untargeted": {"scores": "one_step"},
}


def _control_ses(
    curve: Any, control: str, names: tuple[str, ...], random_state: int
) -> dict[str, float]:
    """Each named quantity's SE under one control's mutation, from one set of draws.

    A name of the paired difference (``S - S_B:...``) reads the SD of the per-draw differences
    of the two scores' draws, which ``paired=False`` draws from separate streams.
    """
    draws = tc.half_sample_draws(
        curve.evaluation,
        n_half_samples=CONTROL_HALF_SAMPLES,
        random_state=random_state,
        **SE_MUTATIONS[control],
    )
    out = {}
    for name in names:
        key = _parse(name)
        if name.startswith(f"{DIFFERENCE}:"):
            values = draws["S"][key] - draws["S_B"][key]
        else:
            values = draws["S"][key]
        out[name] = float(np.std(values, ddof=1))
    return out


def _parse(name: str) -> tuple[str, float]:
    name = name.split(":")[-1]
    if "[" not in name:
        return name, 1.0
    quantity, q = name.rstrip("]").split("[")
    return quantity, float(q)


def fit_group(
    group: Group, frame: pd.DataFrame, *, random_state: int, core: bool = False
) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    ``core=True`` fits the score alone and reads its rows only: no second score, band or
    controls.  The primary phase reads that.

    Returns
    -------
    dict
        ``"rows"``: each reported quantity's estimate and SE by :func:`row_key` name;
        ``"controls"``: each control's value by quantity (an SE, or a point for a bias
        control); ``"band"``: the band's reading; ``"truths"``: cell ``S``'s per-replicate
        truths and its in-sample control's.
    """
    law = law_of(group)
    settings = learners(group, law)
    out: dict[str, Any] = {"controls": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if group.label == "S":
            effect = _study(group, frame).identify(
                ConditionalContrast(SPLIT_MODIFIERS, model=SPLIT_BASIS)
            )
            fitted = effect.estimate(**settings, random_state=random_state).effect_curve()
            curve = fitted.diagnostics.targeting_curve(
                n_half_samples=N_HALF_SAMPLES, random_state=random_state
            )
            out["truths"] = split_truth(curve.score_curve.predict)
            out["rows"] = _rows_of(curve, curve.score)
            out["curve_status"] = curve.status
            evaluation = frame.loc[_evaluation_rows(curve, frame)].reset_index(drop=True)
            out["controls"]["in_sample"] = _in_sample(group, evaluation, random_state)
            return out
        effect = _study(group, frame).identify(ConditionalContrast(Modifiers(["W1"])))
        extra: dict[str, Any] = {}
        if group.label == "H" and not core:
            extra = {"compare": "S_B", "compare_computed_from": ["W1", "W3"]}
        curve = effect.targeting_curve(
            "S",
            computed_from=["W1", "W3"],
            score_kind="external",
            n_half_samples=N_HALF_SAMPLES,
            random_state=random_state,
            **extra,
            **settings,
        )
    out["rows"] = _rows_of(curve, "S")
    out["curve_status"] = curve.status
    if core:
        return out
    if group.label == "H":
        out["rows"].update(_rows_of(curve, DIFFERENCE))
    if group.band is not None:
        out["band"] = _band(curve, truth(law), "S")
    for control, names in group.controls.items():
        if control in SE_MUTATIONS:
            out["controls"][control] = _control_ses(curve, control, names, random_state)
        elif control == "whole_block":
            points = tc.control_values(curve, n_half_samples=2, tie_rule="at_or_above")
            out["controls"][control] = {name: points["S"][_parse(name)][0] for name in names}
    return out


def _evaluation_rows(curve: Any, frame: pd.DataFrame) -> np.ndarray:
    """The rows of ``frame`` a split curve evaluated on, matched by the score's inputs."""
    evaluation = curve.fit.data
    covariates = np.asarray(evaluation.covariates, dtype=float)
    keys = {tuple(np.round(row, 12)) for row in covariates}
    own = np.round(frame[["W1", "W2", "W3"]].to_numpy(float), 12)
    return np.array([tuple(row) in keys for row in own])


def _in_sample(group: Group, evaluation: pd.DataFrame, random_state: int) -> dict[str, float]:
    """The in-sample control: the curve fitted on the evaluation rows ranks them itself."""
    law = law_of(group)
    effect = _study(group, evaluation).identify(
        ConditionalContrast(SPLIT_MODIFIERS, model=SPLIT_BASIS)
    )
    fitted = effect.estimate(**learners(group, law), random_state=random_state).effect_curve()
    score = np.asarray(fitted.predict(evaluation[["W1", "W2", "W3"]]), dtype=float)
    data = fitted.fit.data
    ate = _ate_fit(group, law, data, random_state)
    curve = tc.targeting_fit(ate, ate.data, {"S": score}, pair=(1.0, 0.0), n_half_samples=2)
    truths = split_truth(fitted.predict)
    out = {}
    for name in GROUP_BY_LABEL["S"].controls["in_sample"]:
        quantity, q = _parse(name)
        (row,) = [r for r in curve.rows(quantity) if r["q"] == q]
        out[name] = float(row["estimate"])
        out[f"truth:{name}"] = truths[name]
    return out


def _ate_fit(group: Group, law: Law, data: Any, random_state: int) -> Any:
    from cleverly.estimators import TMLE

    settings = learners(group, law)
    return (
        TMLE(**settings, estimands=("ate",), reference=0.0, random_state=random_state)
        .fit(data)
        .single()
    )


# ------------------------------------------------------------ the primary phase

#: The primary phase is cell ``H``'s five core rows, from a fit of the score alone.
PRIMARY_GROUP = "H"
PRIMARY_REPLICATES = R_MAIN
PRIMARY_N = 2_000
#: Single-thread seconds of one primary fit (the core-only fit of :func:`fit_group`).
PRIMARY_SECONDS = 0.6
SCENARIOS = {PRIMARY_GROUP: CORE}

STUDY = StudyRecord(
    name="Targeting curves (TOC, cumulative gain, AUTOC, Qini)",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "targeting_curves",
    document="docs/technical-reference/method-evidence/targeting-curves.md",
    anchor="targeting-curves",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-targeting-curves",
    reference=None,
    failure_bound={"cleverly-targeting-curves": 0.01},
    modules=(
        "tests/studies/targeting_curves.py",
        "tests/studies/targeting_curves_properties.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.targeting_curves",
    properties_module="tests.studies.targeting_curves_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    group.label: f"law {group.law}, n={group.n}, R={group.replicates}" for group in GROUPS
}


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths.

    Cell ``S`` draws twice its evaluation size; its truths depend on the fitted curve and are
    read from the fit (:func:`split_truth`), so the returned mapping is empty.
    """
    group = GROUP_BY_LABEL[group_label]
    law = law_of(group)
    rows = 2 * n if group.label == "S" else n
    frame = sample(law, rows, np.random.default_rng(seed))
    if group.label == "S":
        return frame, {}
    truths = truth(law)
    if group.label == "H":
        truths.update(paired_truth())
    return frame, truths


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One primary row per core quantity of cell ``H``."""
    group = GROUP_BY_LABEL[scenario]
    fitted = fit_group(group, frame, random_state=replicate, core=True)["rows"]
    rows = []
    for name in SCENARIOS[scenario]:
        estimate, se = fitted[name]["estimate"], fitted[name]["std_error"]
        reference = float(truths[name])
        low, high = estimate - CRITICAL * se, estimate + CRITICAL * se
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": estimate,
                "inference_estimate": estimate,
                "std_error": se,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truths = drawn
    return primary_rows(frame, truths, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


# ---------------------------------------------------------- the design numbers


@dataclass(frozen=True)
class LawGrid:
    """A scrambled Sobol set over ``U(-1, 1)^3`` carrying what the influence functions read.

    ``u`` is each point's top fraction of the score among the points, which stands for the
    population ``U`` to within ``1 / len(w)``.  A continuous score is ranked with no ties (they
    have probability 0, and a lattice would create them); an atomic score's ties are averaged,
    so a block shares one ``u``.
    """

    w: np.ndarray
    tau: np.ndarray
    variance: np.ndarray
    u: np.ndarray
    blocks: np.ndarray


#: Seed of the design's Sobol set, apart from cell ``S``'s truth set.
DESIGN_SOBOL_SEED = 20261110


def law_grid(law: Law, *, points: int = 2**18) -> LawGrid:
    """``points`` scrambled Sobol points of ``law`` (a power of 2), the same for every law."""
    unit = stats.qmc.Sobol(3, scramble=True, seed=DESIGN_SOBOL_SEED).random(points)
    w = 2.0 * unit - 1.0
    score = law.score_of(w)
    method = "average" if law.score == "atoms" else "ordinal"
    ranks = stats.rankdata(-score, method=method)
    _, blocks = np.unique(-score, return_inverse=True)
    return LawGrid(w, law.tau(w), law.gamma_variance(w), (ranks - 0.5) / len(w), blocks)


def _weight(
    quantity: str, q: float, u: np.ndarray
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """``c(u)`` of a quantity and its derivative's measure, as ``(c, dc)`` on a fine ``u`` grid."""
    if quantity == "toc":
        return (u <= q) / q - 1.0, lambda v: np.zeros_like(v)
    if quantity == "autoc":
        return -np.log(u) - 1.0, lambda v: -1.0 / v
    return (1.0 - 2.0 * u) / 2.0, lambda v: -np.ones_like(v)


def influence(
    grid: LawGrid, quantity: str, q: float = 1.0, *, mutation: str | None = None, bins: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """``(a, h)``: the influence function of a rank functional is ``a(U) (Gamma - tau) + h(W)``.

    ``a`` multiplies the noise of the score, whose conditional variance is ``grid.variance``;
    ``h`` is the part that varies with ``W``.  For a weight ``c(u)``,
    ``h = tau c(U) - theta + int_U^1 m c' du - int_0^1 m c' u du`` with ``m(u) = E[tau | U = u]``
    (the rank term of Theorem 3).  ``mutation="fixed_cut"`` replaces the TOC's quantile term
    by the fixed-threshold mean; ``mutation="fixed_ranks"`` drops the rank term.
    """
    u, tau = grid.u, grid.tau
    c, dc = _weight(quantity, q, u)
    theta = float(np.mean(tau * c))
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, bins - 1)
    m_bins = np.bincount(index, weights=tau, minlength=bins) / np.maximum(
        np.bincount(index, minlength=bins), 1
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    if quantity == "toc":
        m_q = float(np.interp(q, centres, m_bins))
        top = u <= q
        if mutation == "fixed_cut":
            mean_top = float(np.mean(tau[top]))
            rank = -(mean_top / q) * (top - q)
        else:
            rank = -(m_q / q) * (top - q)
    else:
        integrand = m_bins * dc(centres) / bins
        tail = np.cumsum(integrand[::-1])[::-1]
        rank = tail[index] - float(np.sum(integrand * centres))
    if mutation == "fixed_ranks":
        rank = np.zeros_like(rank)
    return c, tau * c - theta + rank


def influence_sd(
    grid: LawGrid, quantity: str, q: float = 1.0, *, mutation: str | None = None
) -> float:
    """The SD of one rank functional's influence function on ``grid``."""
    a, h = influence(grid, quantity, q, mutation=mutation)
    return math.sqrt(float(np.mean(a**2 * grid.variance) + np.var(h)))


def _covariance(grid: LawGrid, keys: list[tuple[str, float]]) -> np.ndarray:
    parts = [influence(grid, quantity, q) for quantity, q in keys]
    k = len(parts)
    out = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            a_i, h_i = parts[i]
            a_j, h_j = parts[j]
            out[i, j] = float(
                np.mean(a_i * a_j * grid.variance)
                + np.mean((h_i - h_i.mean()) * (h_j - h_j.mean()))
            )
    return out


def band_power(beta: float, n: int, *, draws: int = 200_000, seed: int = 7) -> float:
    """The exact-law power of the sup-t band of cell ``P`` at slope ``beta``.

    The TOC estimates over the grid points below 1 are normal with the covariance of their
    influence functions over ``n``; the band rejects when some ``|T_j|`` exceeds the 0.95
    quantile of ``max_j |Z_j|`` under the correlation of the ``T_j``.
    """
    law = power_law(beta)
    grid = law_grid(law, points=2**16)
    keys = [("toc", q) for q in GRID if q < 1.0]
    cov = _covariance(grid, keys)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    curve = Curve(law)
    shift = np.array([curve.toc(q) for _, q in keys]) / (sd / math.sqrt(n))
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(keys)), corr, size=draws)
    critical = float(np.quantile(np.max(np.abs(z), axis=1), 0.95))
    return float(np.mean(np.max(np.abs(z + shift), axis=1) > critical))


def solve_power_slope(target: float = 0.90, n: int = 1_000) -> float:
    """The slope ``beta`` at which cell ``P``'s band has exact-law power ``target``."""
    return float(optimize.brentq(lambda b: band_power(b, n) - target, 0.05, 1.0, xtol=1e-3))


#: Cell ``P``'s slope: the band's exact-law power is 0.90 at ``n = 1,000`` (:func:`solve_power_slope`;
#: the design test recomputes it).
P_SLOPE = 0.46


def calibration_row_pass(
    replicates: int, ratio: float = 1.0, *, se_cv: float = 0.1, kurtosis: float = 3.0
) -> float:
    """P(a calibration row passes), as ``conditional_projection.calibration_row_pass``."""
    from tests.studies.conditional_projection import calibration_row_pass as shared

    return shared(replicates, ratio, se_cv=se_cv, kurtosis=kurtosis)


def se_control_detection(ratio: float, replicates: int, *, se_cv: float = 0.1) -> float:
    """P(an SE control's 99% SE-ratio interval lies wholly outside the band, on its side)."""
    low, high = Margins().calibration_se_ratio
    z = float(stats.norm.ppf(0.995))
    sd = ratio * math.sqrt(se_cv**2 / replicates + 1.0 / (2.0 * replicates))
    if ratio >= 1.0:
        return float(stats.norm.sf((high + z * sd - ratio) / sd))
    return float(stats.norm.cdf((low - z * sd - ratio) / sd))


def bias_pass_probability(standardized: float, replicates: int, *, control: bool) -> float:
    """As ``conditional_projection.bias_pass_probability``."""
    from tests.studies.conditional_projection import bias_pass_probability as shared

    return shared(standardized, replicates, control=control)


def type_i_pass_probability(size: float, replicates: int) -> float:
    """As ``conditional_projection.type_i_pass_probability``."""
    from tests.studies.conditional_projection import type_i_pass_probability as shared

    return shared(size, replicates)


def pointwise_joint_coverage(*, draws: int = 400_000, seed: int = 11) -> float:
    """The exact-law joint coverage of cell ``H``'s pointwise 95% intervals over the grid.

    The TOC estimates at the grid points below 1 are normal with the covariance of their
    influence functions (:func:`_covariance`); the pointwise intervals cover jointly when every
    ``|Z_j|`` is at most :data:`CRITICAL`.  The Monte Carlo SE is about 0.0007.
    """
    grid = law_grid(LAWS["H"])
    cov = _covariance(grid, [("toc", q) for q in GRID if q < 1.0])
    sd = np.sqrt(np.diag(cov))
    z = np.random.default_rng(seed).multivariate_normal(
        np.zeros(len(sd)), cov / np.outer(sd, sd), size=draws
    )
    return float(np.mean(np.max(np.abs(z), axis=1) <= CRITICAL))


def joint_control_detection(coverage: float, replicates: int) -> float:
    """P(the pointwise control's 99% joint-coverage interval lies wholly below 0.95)."""
    z = float(stats.norm.ppf(0.995))
    sd = math.sqrt(coverage * (1.0 - coverage) / replicates)
    return float(stats.norm.cdf((0.95 - z * sd - coverage) / sd))


#: The size and seed of :func:`k_untargeted_ratio`'s one fit.
K_RATIO_N = 20_000
K_RATIO_SEED = 20261111


@functools.cache
def k_untargeted_ratio() -> dict[str, float]:
    """The SD ratio of cell ``K``'s one-step draws to its targeted draws, by quantity.

    One fit of cell ``K``'s law at :data:`K_RATIO_N` with its learners (a main-terms outcome
    regression and the known mechanism), 1,000 half-samples of each.  A ratio has no closed
    form: it is the shrinkage of the fluctuation, which depends on how the main-terms fit is
    wrong.  At this size its Monte Carlo SE is about 0.02.
    """
    group = GROUP_BY_LABEL["K"]
    law = law_of(group)
    frame = sample(law, K_RATIO_N, np.random.default_rng(K_RATIO_SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effect = _study(group, frame).identify(ConditionalContrast(Modifiers(["W1"])))
        curve = effect.targeting_curve(
            "S",
            computed_from=["W1", "W3"],
            score_kind="external",
            n_half_samples=1_000,
            random_state=0,
            **learners(group, law),
        )
    targeted = tc.control_values(curve, n_half_samples=1_000, random_state=0)["S"]
    one_step = tc.control_values(curve, n_half_samples=1_000, random_state=0, scores="one_step")[
        "S"
    ]
    return {name: one_step[_parse(name)][1] / targeted[_parse(name)][1] for name in group.rows}


def whole_block_shift() -> tuple[float, float]:
    """``(shift, SD)`` of cell ``D``'s whole-tied-block control at ``q = 0.2``.

    Under ``S >= cut`` the top fraction at 0.2 is the top two atoms, 0.375 of the mass; the
    control's limit is that fraction's TOC, and its SD is the fixed-threshold mean's.
    """
    law = LAWS["D"]
    curve = Curve(law)
    limit = curve.gain(0.375) / 0.375 - curve.ate
    grid = law_grid(law)
    top = grid.u <= 0.375
    a = top / 0.375 - 1.0
    mean_top = float(np.mean(grid.tau[top]))
    h = grid.tau * a - (mean_top - curve.ate) - (mean_top / 0.375) * (top - 0.375)
    sd = math.sqrt(float(np.mean(a**2 * grid.variance) + np.var(h)))
    return limit - curve.toc(0.2), sd


def design_numbers() -> dict[str, Any]:
    """Every computed number the declaration fixes before the run.

    Returns
    -------
    dict
        ``calibration``: the per-row pass probability at ratio 1 and 0.97 and the family-wise
        rate; ``controls``: each SE control's ratio and each bias control's standardized shift,
        with its detection probability, and the pointwise control's joint coverage;
        ``reported``: the ratios that carry no control row because they are close to 1 (the
        fixed-cut ratio at ``q = 0.1``, and cell ``K``'s untargeted ratio at TOC(0.5) and
        Qini); ``power``: cell ``P``'s exact-law band power at the declared slope and its pass
        probability at 0.90 and 0.87; ``type_i_pass``; ``cost``: single-thread hours of the
        declared run (the property groups and the primary phase) and of the pilot.
    """
    from tests.studies.evidence.property_verdicts import power_cell_pass_probability

    positives = [cell for cell in PROPERTY_CELLS[CALIBRATION] if is_positive(cell)]
    per_row = calibration_row_pass(R_MAIN, 1.0)
    h = law_grid(LAWS["H"])
    controls: dict[str, dict[str, float]] = {}
    reported = {
        "H_toc_0.1__fixed_cut_ratio": influence_sd(h, "toc", 0.1, mutation="fixed_cut")
        / influence_sd(h, "toc", 0.1)
    }
    for q in (0.5, 0.9):
        ratio = influence_sd(h, "toc", q, mutation="fixed_cut") / influence_sd(h, "toc", q)
        controls[f"H_toc_{q:g}__fixed_cut"] = {
            "ratio": ratio,
            "detection": se_control_detection(ratio, R_MAIN),
        }
    ratio = influence_sd(h, "autoc", mutation="fixed_ranks") / influence_sd(h, "autoc")
    controls["H_autoc__fixed_ranks"] = {
        "ratio": ratio,
        "detection": se_control_detection(ratio, R_MAIN),
    }
    second = law_grid(Law("H_B", binary=False, c=C_SECOND))
    for quantity, q in (("toc", 0.1), ("autoc", 1.0)):
        a_1, h_1 = influence(h, quantity, q)
        a_2, h_2 = influence(second, quantity, q)
        paired = math.sqrt(float(np.mean((a_1 - a_2) ** 2 * h.variance) + np.var(h_1 - h_2)))
        unpaired = math.sqrt(
            influence_sd(h, quantity, q) ** 2 + influence_sd(second, quantity, q) ** 2
        )
        controls[f"H_{quantity}_{q:g}__unpaired"] = {
            "ratio": unpaired / paired,
            "detection": se_control_detection(unpaired / paired, R_MAIN),
        }
    joint = pointwise_joint_coverage()
    controls["H_band__pointwise_joint"] = {
        "coverage": joint,
        "detection": joint_control_detection(joint, R_MAIN),
    }
    untargeted = k_untargeted_ratio()
    controls["K_autoc__untargeted"] = {
        "ratio": untargeted["autoc"],
        "detection": se_control_detection(untargeted["autoc"], R_MAIN),
    }
    for name in (row_key("toc", 0.5), "qini"):
        reported[f"K_{_label(name)}__untargeted_ratio"] = untargeted[name]
    shift, sd = whole_block_shift()
    standardized = shift / (sd / math.sqrt(GROUP_BY_LABEL["D"].n))
    controls["D_toc_0.2__whole_block"] = {
        "standardized": standardized,
        "detection": bias_pass_probability(standardized, R_MAIN, control=True),
    }
    power = band_power(P_SLOPE, GROUP_BY_LABEL["P"].n)
    cpu = sum(group.replicates * group.seconds for group in GROUPS) / 3600.0
    cpu += PRIMARY_REPLICATES * PRIMARY_SECONDS / 3600.0
    pilot = PILOT_REPLICATES * sum(group.seconds for group in GROUPS) / 3600.0
    return {
        "calibration": {
            "rows": len(positives),
            "pass_at_1": per_row,
            "pass_at_0.97": calibration_row_pass(R_MAIN, 0.97),
            "family": per_row ** len(positives),
            "pass_at_1_r3200": calibration_row_pass(3_200, 1.0),
        },
        "controls": controls,
        "reported": reported,
        "power": {
            "slope": P_SLOPE,
            "exact_law_power": power,
            "pass_at_0.90": power_cell_pass_probability(0.90, R_POWER),
            "pass_at_0.87": power_cell_pass_probability(0.87, R_POWER),
        },
        "type_i_pass": type_i_pass_probability(0.05, R_MAIN),
        "cost": {"cpu_hours": cpu, "wall_hours_4": cpu / 4.0, "pilot_cpu_hours": pilot},
    }


#: Replications of each group in the pilot.
PILOT_REPLICATES = 800

#: The pilot's committed summary (:mod:`tests.studies.targeting_curves_pilot`): a pilot on its
#: own seed stream, read by the design test, not a result of the run.
PILOT_CSV = ROOT / "tests" / "studies" / "targeting_curves_pilot.csv"


def pilot_rows() -> dict[tuple[str, str, str], dict[str, float]]:
    """The pilot's summary row of each ``(group, kind, quantity)``; empty before the pilot runs.

    ``kind`` is ``correctly_specified`` for a calibration row, the control's name for a
    control, and ``band`` for the band's rates.
    """
    if not PILOT_CSV.exists():
        return {}
    frame = pd.read_csv(PILOT_CSV)
    return {
        (str(row["group"]), str(row["kind"]), str(row["name"])): row
        for row in frame.to_dict("records")
    }
