r"""The pilots of ``dose-response-npcausal`` (X93 S2), and the rule that reads the declaration off them.

Every pilot draw takes ``stream_seed(STUDY, "pilot", "primary", r)``, a stream no declared cell
reads, at the declared ``n``.  Two pilots fit those draws.

* This module's pilot fits each draw with the subject at every pair of
  :data:`~tests.studies.dose_response_npcausal.PILOT_CURVE` (every declared dose at every
  candidate bandwidth) and with :func:`ctseff_sketch`, a Python transcription of
  ``npcausal::ctseff`` at commit ``56a5ac1``, at every candidate bandwidth.  It also refits the
  first :data:`REFLECTED_DRAWS` draws with the dose reflected, :math:`A \mapsto 5 - A`, which
  is the witness for where the subject's standard-error over-report comes from.  It runs where
  R cannot: ``python -m tests.studies.dose_response_npcausal_pilot --jobs N``.
* The R pilot (:mod:`tests.canonical.dose_response_npcausal.pilot`) fits the same draws with
  the real ``ctseff``, through the declared runner, in the declared image, and commits
  :data:`R_OUTPUT`, :data:`R_FAILURES` and :data:`R_METADATA`.

:func:`declaration_from_pilot` reads the bandwidths and the replicate count off the subject's
rows and the reference's rows: the R pilot's once it is committed, the transcription's until
then.  The study module's :data:`~tests.studies.dose_response_npcausal.BANDWIDTHS` and
:data:`~tests.studies.dose_response_npcausal.PRIMARY_REPLICATES` must equal its output, which
the design test checks.

The transcription follows ``ctseff`` line by line, including its leave-one-out bandwidth risk,
which runs even when ``bw.seq`` holds one value.  It differs in three declared places:

* ``smooth.spline`` is :func:`scipy.interpolate.make_smoothing_spline` with its GCV penalty and
  a knot at every grid point, where R places about 60 knots.  The outcome spline smooths a curve
  that is exactly linear in the dose, which both reproduce, so only the marginal-density spline
  can differ;
* ``density`` is an exact Gaussian kernel sum with R's ``bw.nrd0`` bandwidth, on R's 512-point
  grid from ``min - 3 bw`` to ``max + 3 bw``, where R bins and convolves by FFT;
* ``SuperLearner`` with its one learner is the ordinary least-squares fit of the same design.

``KernSmooth::locpoly`` is transcribed from its Fortran: linear binning onto 401 points with
``truncate = TRUE`` (``rlbin``), Gaussian weights cut at four bandwidths (``locpol``), and the
2x2 LINPACK solve (``dgefa``/``dgesl``), whose division by a zero pivot gives ``NaN`` at a node
whose window holds one bin or none.  ``approx`` drops such a node, returns ``NA`` outside the
remaining nodes, and an ``NA`` risk leaves ``which.min`` empty, so ``ctseff`` errors.  The
transcription raises :class:`CtseffRefusal` there, and the pilot records the call as a failure.
It keeps ``ctseff``'s pairing of the two density grids: the density of the standardized
residuals is ``approx(density(all)$x, density(observed)$y, xout = all)``.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.stats import beta, binom, norm, t

from cleverly.utils.parallel import map_parallel
from tests.studies import dose_response_npcausal as study
from tests.studies import dose_response_npcausal_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).parent
OUTPUT = HERE / "dose_response_npcausal_pilot.csv.gz"
#: The R pilot's rows, failure records and run metadata, written by
#: :mod:`tests.canonical.dose_response_npcausal.pilot`.
R_OUTPUT = HERE / "dose_response_npcausal_pilot_r.csv.gz"
R_FAILURES = HERE / "dose_response_npcausal_pilot_r_failures.csv"
R_METADATA = HERE / "dose_response_npcausal_pilot_r.json"
DRAWS = 1_000
REFLECTED_DRAWS = 200
SKETCH = "ctseff-sketch"
REFLECTED = "cleverly-reflected"
#: The reflection :math:`a \mapsto 5 - a` maps :math:`D = [-1.5, 6.5]` and the doses to
#: themselves.
MIRROR = 5.0

#: ``KernSmooth::locpoly``'s grid size and Gaussian cut (``tau``), in bandwidths.
LOCPOLY_POINTS = 401
LOCPOLY_TAU = 4.0

COLUMNS = (
    "replicate",
    "implementation",
    "estimand",
    "dose",
    "bandwidth",
    "truth",
    "estimate",
    "initial_estimate",
    "std_error",
    "covered",
    "efficient_sd",
    "seconds",
    "refusal",
)


def draw(replicate: int) -> pd.DataFrame:
    """Pilot draw ``replicate``, the sample both pilots fit."""
    return law.sample(study.PRIMARY_N, stream_seed(study.STUDY, "pilot", "primary", replicate))


# ------------------------------------------------------------------------- the transcription


class CtseffRefusal(RuntimeError):
    """A call where ``ctseff`` would raise; the message names the step."""


def _interactions(columns: np.ndarray) -> np.ndarray:
    """``[1, x, x_i x_j (i < j)]``: R's ``glm(Y ~ .^2)`` design."""
    p = columns.shape[1]
    products = [columns[:, i] * columns[:, j] for i in range(p) for j in range(i + 1, p)]
    return np.column_stack([np.ones(len(columns)), columns, *products])


def _ols(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _nrd0(x: np.ndarray) -> float:
    spread = min(float(np.std(x, ddof=1)), float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34)
    return 0.9 * spread * len(x) ** -0.2


def _density_grid(x: np.ndarray) -> tuple[np.ndarray, float]:
    bw = _nrd0(x)
    return np.linspace(x.min() - 3.0 * bw, x.max() + 3.0 * bw, 512), bw


def _kde(points: np.ndarray, data: np.ndarray, bw: float) -> np.ndarray:
    out = np.empty(len(points))
    for start in range(0, len(points), 64):
        block = points[start : start + 64]
        out[start : start + 64] = norm.pdf((block[:, None] - data[None, :]) / bw).mean(axis=1) / bw
    return out


def r_seq(low: float, high: float, points: int) -> np.ndarray:
    """R's ``seq(low, high, length.out = points)``: ``low + i (high - low)/(points - 1)``, then ``high``."""
    step = (high - low) / (points - 1)
    return np.concatenate([[low], low + np.arange(1, points - 1) * step, [high]])


def approx(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """R's ``approx(x, y, xout)``: linear, ``rule = 1``, ``na.rm = TRUE``, on increasing ``x``.

    A ``NaN`` node is dropped, and a point outside the remaining nodes is ``NaN`` (R's ``NA``).
    A point on a node returns the node's value, as R's ``approx1`` does.
    """
    keep = ~np.isnan(y)
    x, y = x[keep], y[keep]
    xout = np.asarray(xout, dtype=float)
    out = np.full(xout.shape, np.nan)
    if len(x) < 2:
        return out
    inside = (xout >= x[0]) & (xout <= x[-1])
    v = xout[inside]
    j = np.clip(np.searchsorted(x, v, side="right"), 1, len(x) - 1)
    i = j - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        value = y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
    value = np.where(v == x[j], y[j], np.where(v == x[i], y[i], value))
    out[inside] = value
    return out


def _rlbin(
    x: np.ndarray, y: np.ndarray, low: float, high: float, points: int
) -> tuple[np.ndarray, np.ndarray]:
    """KernSmooth's ``rlbin`` with ``trun = 1``: linear binning, a point past the last node dropped."""
    counts = [0.0] * points
    sums = [0.0] * points
    delta = (high - low) / (points - 1)
    for xi, yi in zip(x.tolist(), y.tolist(), strict=True):
        lxi = ((xi - low) / delta) + 1.0
        li = int(lxi)
        rem = lxi - li
        if 1 <= li < points:
            counts[li - 1] += 1.0 - rem
            counts[li] += rem
            sums[li - 1] += (1.0 - rem) * yi
            sums[li] += rem * yi
    return np.array(counts), np.array(sums)


def locpoly(x: np.ndarray, y: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """``KernSmooth::locpoly(x, y, bandwidth = bandwidth)``: degree 1, Gaussian, 401 points."""
    low, high = float(np.min(x)), float(np.max(x))
    points = LOCPOLY_POINTS
    delta = (high - low) / (points - 1)
    counts, sums = _rlbin(x, y, low, high, points)
    reach = min(math.floor(LOCPOLY_TAU * bandwidth / delta), points)
    if reach == 0:
        raise CtseffRefusal("locpoly: binning grid too coarse for current (small) bandwidth")
    offsets = np.arange(-reach, reach + 1)
    weights = np.exp(-(((delta * np.abs(offsets)) / bandwidth) ** 2) / 2.0)
    s0, s1, s2 = (np.zeros(points) for _ in range(3))
    t0, t1 = np.zeros(points), np.zeros(points)
    # locpol: every bin k with a nonzero count adds to the nodes j within the cut, with the
    # moment factor (delta (k - j))^i built by repeated multiplication.
    for k in (int(index) for index in np.flatnonzero(counts != 0.0)):
        j = np.arange(max(0, k - reach), min(points, k + reach + 1))
        kernel = weights[k - j + reach]
        first = (1.0 * delta) * (k - j)
        second = (first * delta) * (k - j)
        s0[j] += counts[k] * kernel
        t0[j] += sums[k] * kernel
        s1[j] += counts[k] * kernel * first
        t1[j] += sums[k] * kernel * first
        s2[j] += counts[k] * kernel * second
    return r_seq(low, high, points), _linpack_intercept(s0, s1, s2, t0, t1)


def _linpack_intercept(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, t0: np.ndarray, t1: np.ndarray
) -> np.ndarray:
    """The first component of ``[[s0, s1], [s1, s2]] b = [t0, t1]`` by LINPACK ``dgefa``/``dgesl``.

    Partial pivoting on the first column; a zero pivot skips the elimination, and the back
    substitution then divides by zero, as the Fortran does.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        swap = np.abs(s1) > np.abs(s0)
        a11 = np.where(swap, s1, s0)
        a21 = np.where(swap, s0, s1)
        a12 = np.where(swap, s2, s1)
        a22 = np.where(swap, s1, s2)
        b1 = np.where(swap, t1, t0)
        b2 = np.where(swap, t0, t1)
        zero = a11 == 0.0
        m21 = np.where(zero, a21, (-1.0 / a11) * a21)
        a22 = np.where(zero, a22, a22 + a12 * m21)
        b2 = np.where(zero, b2, b2 + b1 * m21)
        x2 = b2 / a22
        x1 = (b1 + (-x2) * a12) / a11
    return np.asarray(x1, dtype=float)


@dataclass(frozen=True)
class CtseffInputs:
    """What every bandwidth's ``ctseff`` call shares: the data, the grid and the pseudo-outcome."""

    a: np.ndarray
    grid: np.ndarray
    pseudo: np.ndarray
    centred: np.ndarray
    marginal: np.ndarray


def ctseff_inputs(frame: pd.DataFrame) -> CtseffInputs:
    """``ctseff``'s nuisances and pseudo-outcome, which do not depend on the bandwidth."""
    w = frame[list(study.COVARIATES)].to_numpy()
    a = frame["A"].to_numpy()
    y = frame["Y"].to_numpy()
    n = len(a)
    grid = study.npcausal_grid(a)
    k = len(grid)
    design_w = _interactions(w)
    pi = design_w @ _ols(design_w, a)
    pi2 = design_w @ _ols(design_w, (a - pi) ** 2)
    if np.any(pi2 <= 0.0):
        raise CtseffRefusal("density: the fitted dose variance is not positive")
    observed = _interactions(np.column_stack([w, a]))
    coefficients = _ols(observed, y)
    mu = observed @ coefficients
    mu_grid = np.column_stack(
        [_interactions(np.column_stack([w, np.full(n, value)])) @ coefficients for value in grid]
    )
    scale = np.sqrt(pi2)
    standard = np.concatenate(
        [(a - pi) / scale, ((grid[None, :] - pi[:, None]) / scale[:, None]).T.ravel()]
    )
    x_all, _ = _density_grid(standard)
    x_obs, bw_obs = _density_grid(standard[:n])
    y_obs = _kde(x_obs, standard[:n], bw_obs)
    pi_hat = approx(x_all, y_obs, standard)
    pi_obs = pi_hat[:n] / scale
    pi_grid = (pi_hat[n:].reshape(k, n).T) / scale[:, None]
    marginal = pi_grid.mean(axis=0)
    m_grid = mu_grid.mean(axis=0)
    varpi = make_smoothing_spline(grid, marginal)(a)
    m_obs = make_smoothing_spline(grid, m_grid)(a)
    pseudo = (y - mu) / (pi_obs / varpi) + m_obs
    if not np.all(np.isfinite(pseudo)):
        raise CtseffRefusal("pseudo-outcome: a density value outside the density grid")
    return CtseffInputs(a, grid, pseudo, mu_grid - m_grid[None, :], marginal)


def _hatvals(inputs: CtseffInputs, bandwidth: float) -> np.ndarray:
    """``ctseff``'s ``hatvals``: the local-linear leverage on the grid, interpolated to the data."""
    a, grid = inputs.a, inputs.grid
    u = (a[None, :] - grid[:, None]) / bandwidth
    kern = norm.pdf(u) / bandwidth
    k0, k1, k2 = kern.mean(axis=1), (u * kern).mean(axis=1), (u * u * kern).mean(axis=1)
    leverage = k2 * (norm.pdf(0.0) / bandwidth) / (k0 * k2 - k1**2) / len(a)
    return approx(grid, leverage, a)


def knife_edge(a: np.ndarray, bandwidth: float) -> bool:
    """Whether R's outcome for this call hinges on the last bit of the highest dose's bin.

    ``rlbin`` drops the highest dose when ``(max - min) / delta`` rounds to exactly 400, and
    keeps it in the last two bins otherwise.  When the second-highest dose lies outside the
    last node's window, the dropped case leaves that window empty (``NaN``, an ``NA`` risk and
    an error) and the kept case gives a finite fit.  One unit in the last place of the
    highest dose, from parsing the CSV or from R's arithmetic, flips it: the R pilot's draw 11
    at bandwidth 1 failed with "'bandwidth' must be a scalar or an array of length
    'gridsize'" where the transcription fitted.  The transcription cannot settle such a call.
    """
    ordered = np.sort(np.asarray(a, dtype=float))
    low, high = float(ordered[0]), float(ordered[-1])
    points = LOCPOLY_POINTS
    delta = (high - low) / (points - 1)
    reach = min(math.floor(LOCPOLY_TAU * bandwidth / delta), points)
    second = int(((float(ordered[-2]) - low) / delta) + 1.0)
    return second + 1 < points - reach


def leave_one_out_risk(inputs: CtseffInputs, bandwidth: float) -> float:
    """``ctseff``'s ``risk.fn(bandwidth)``; ``NaN`` where R's is ``NA``."""
    grid, fit = locpoly(inputs.a, inputs.pseudo, bandwidth)
    smoothed = approx(grid, fit, inputs.a)
    return float(np.mean(((inputs.pseudo - smoothed) / (1.0 - _hatvals(inputs, bandwidth))) ** 2))


def ctseff_fit(inputs: CtseffInputs, bandwidth: float, doses: Sequence[float]) -> pd.DataFrame:
    """One ``ctseff(..., bw.seq = bandwidth)`` call at ``doses``; raises :class:`CtseffRefusal`."""
    if math.isnan(leave_one_out_risk(inputs, bandwidth)):
        raise CtseffRefusal(
            "risk.fn: the leave-one-out risk is NA, so which.min(risk.est) is empty and "
            "locpoly receives a zero-length bandwidth"
        )
    a, grid, pseudo = inputs.a, inputs.grid, inputs.pseudo
    n = len(a)
    nodes, fit = locpoly(a, pseudo, bandwidth)
    estimates = approx(nodes, fit, np.asarray(doses, dtype=float))
    steps = np.diff(grid)
    rows = []
    for dose, estimate in zip(doses, estimates, strict=True):
        if not math.isfinite(float(estimate)):
            raise CtseffRefusal(f"est: no finite estimate at dose {dose}")
        u = (a - dose) / bandwidth
        weight = norm.pdf(u) / bandwidth
        design = np.column_stack([np.ones(n), u])
        coefficients = np.linalg.solve(
            design.T @ (design * weight[:, None]), design.T @ (weight * pseudo)
        )
        dh = np.array(
            [
                [weight.mean(), (weight * u).mean()],
                [(weight * u).mean(), (weight * u * u).mean()],
            ]
        )
        g = (grid - dose) / bandwidth
        kernel_grid = norm.pdf(g) / bandwidth
        integrand1 = kernel_grid[None, :] * inputs.centred * inputs.marginal[None, :]
        integrand2 = g[None, :] * integrand1
        int1 = (steps[None, :] * (integrand1[:, 1:] + integrand1[:, :-1]) / 2.0).sum(axis=1)
        int2 = (steps[None, :] * (integrand2[:, 1:] + integrand2[:, :-1]) / 2.0).sum(axis=1)
        residual = pseudo - coefficients[0] - coefficients[1] * u
        scores = np.linalg.solve(
            dh, np.vstack([weight * residual + int1, u * weight * residual + int2])
        )
        se = float(np.sqrt(np.cov(scores)[0, 0]))
        rows.append({"dose": dose, "estimate": float(estimate), "se": se / np.sqrt(n)})
    return pd.DataFrame(rows)


def ctseff_sketch(frame: pd.DataFrame, bandwidth: float, doses: Sequence[float]) -> pd.DataFrame:
    """``ctseff``'s estimate and standard error at ``doses``, transcribed (module docstring)."""
    return ctseff_fit(ctseff_inputs(frame), bandwidth, doses)


# -------------------------------------------------------------------------------- the pilot


def _name(h: float, a0: float) -> str:
    return next(
        name for name, (dose, width, _) in study.PILOT_PAIRS.items() if dose == a0 and width == h
    )


def _subject_rows(
    frame: pd.DataFrame, replicate: int, implementation: str, truth_of: Callable[[float], float]
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, initial = study.fit_with_initials(frame, curve=study.PILOT_CURVE)
    except Exception as error:  # a refusal is a pilot finding, recorded with its text
        return [
            {
                "replicate": replicate,
                "implementation": implementation,
                "refusal": f"{type(error).__name__}: {error}",
            }
        ]
    seconds = time.perf_counter() - start
    rows = []
    for name, (a0, h, _) in study.PILOT_PAIRS.items():
        estimate = result[name]
        low, high = estimate.ci
        truth = truth_of(a0)
        # The reflected law at a0 is the original law at MIRROR - a0; the kernel is symmetric.
        efficient_a0 = MIRROR - a0 if implementation == REFLECTED else a0
        rows.append(
            {
                "replicate": replicate,
                "implementation": implementation,
                "estimand": name,
                "dose": a0,
                "bandwidth": h,
                "truth": truth,
                "estimate": float(estimate.psi),
                "initial_estimate": float(initial[name]),
                "std_error": float(estimate.std_error),
                "covered": int(low <= truth <= high),
                "efficient_sd": law.efficiency_sd(h, efficient_a0, study.KERNEL)
                / math.sqrt(study.PRIMARY_N),
                "seconds": seconds,
                "refusal": "",
            }
        )
    return rows


def sketch_rows(frame: pd.DataFrame, replicate: int) -> list[dict[str, Any]]:
    """The transcription's rows at every candidate bandwidth; a refused call is one row."""
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    try:
        inputs: CtseffInputs | None = ctseff_inputs(frame)
        shared, problem = time.perf_counter() - start, ""
    except CtseffRefusal as error:
        inputs, shared, problem = None, 0.0, str(error)
    for h in study.CANDIDATE_BANDWIDTHS:
        start = time.perf_counter()
        try:
            if inputs is None:
                raise CtseffRefusal(problem)
            fitted = ctseff_fit(inputs, study.NPCAUSAL_BANDWIDTHS[h], study.DOSES)
        except CtseffRefusal as error:
            rows.append(
                {
                    "replicate": replicate,
                    "implementation": SKETCH,
                    "bandwidth": h,
                    "refusal": str(error),
                }
            )
            continue
        seconds = shared + time.perf_counter() - start
        for row in fitted.itertuples(index=False):
            truth = float(law.curve(row.dose))
            low, high = row.estimate - 1.96 * row.se, row.estimate + 1.96 * row.se
            rows.append(
                {
                    "replicate": replicate,
                    "implementation": SKETCH,
                    "estimand": _name(h, row.dose),
                    "dose": row.dose,
                    "bandwidth": h,
                    "truth": truth,
                    "estimate": float(row.estimate),
                    "std_error": float(row.se),
                    "covered": int(low <= truth <= high),
                    "seconds": seconds,
                    "refusal": "",
                }
            )
    return rows


def _one(replicate: int) -> list[dict[str, Any]]:
    frame = draw(replicate)
    rows = _subject_rows(frame, replicate, study.IMPLEMENTATION, lambda a0: float(law.curve(a0)))
    rows += sketch_rows(frame, replicate)
    if replicate < REFLECTED_DRAWS:
        mirrored = frame.assign(A=MIRROR - frame["A"])
        rows += _subject_rows(
            mirrored, replicate, REFLECTED, lambda a0: float(law.curve(MIRROR - a0))
        )
    return rows


def run(n_jobs: int, draws: int = DRAWS) -> pd.DataFrame:
    """Fit ``draws`` pilot draws and write every row to :data:`OUTPUT`."""
    outcomes = map_parallel(_one, [(r,) for r in range(draws)], n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows], columns=list(COLUMNS))
    frame.to_csv(OUTPUT, index=False)
    return frame


def read(path: Path) -> pd.DataFrame:
    """A committed pilot table, every float as written."""
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])


# --------------------------------------------------------------------------- the declaration

#: The one-sided Clopper-Pearson level of a pilot failure rate's upper bound.
FAILURE_LEVEL = 0.95
#: The largest chance the declaration accepts that the reference's failures exceed the
#: declared bound ``floor(FAILURE_BOUND * R)`` at the declared count, at that upper bound.
FAILURE_EXCEED = 0.01
#: The joint pass probability the paired legs are sized to, over the cells outside the
#: pre-registered red set.
PAIRED_PASS = 0.95
#: The largest count the paired-leg rule may ask for: about 95 R CPU-hours at two ``ctseff``
#: calls of 21.4 CPU-s (the local probe) per replicate.  Past it the straddle reading of the regeneration
#: docstring governs a calibration red at doses 1.5 and 2.5 instead.
PAIRED_CAP = 8_000
#: Bootstrap resamples for the pilot spread of each paired-leg statistic.
PAIRED_BOOTSTRAP = 400


@dataclass(frozen=True)
class PilotRows:
    """One implementation's pilot rows at the candidate pairs, and the draws each call lost.

    Parameters
    ----------
    rows : pandas.DataFrame
        One row per draw and estimand: ``replicate``, ``estimand``, ``estimate``,
        ``std_error``, ``covered`` and ``truth``.
    failed : Mapping[float, frozenset[int]]
        The draws whose call at each candidate bandwidth failed.  A subject fit covers every
        bandwidth, so its lost draws repeat under each key.
    draws : int
        The pilot draws fitted.
    source : str
        The implementation the rows came from.
    """

    rows: pd.DataFrame
    failed: Mapping[float, frozenset[int]]
    draws: int
    source: str


@dataclass(frozen=True)
class Declaration:
    """What the pilot rule declares, and every reading it took.

    Parameters
    ----------
    bandwidths : tuple[float, float]
        The selected consecutive pair of candidate bandwidths.
    replicates : int
        The declared count, the largest of the three readings.
    assumed : int
        The shared rule with the subject's pilot rows for both implementations.
    own : int
        The shared rule with each implementation's own pilot rows.
    paired : int or None
        The paired-leg count, or ``None`` past :data:`PAIRED_CAP` (the straddle reading).
    reference_failures : int
        Pilot draws where either selected call of the reference failed.
    failure_exceed : float
        The chance the reference exceeds its failure bound at ``replicates``, at the pilot
        rate's upper bound.
    source : str
        ``"npcausal"`` for the R pilot, :data:`SKETCH` for the transcription.
    """

    bandwidths: tuple[float, float]
    replicates: int
    assumed: int
    own: int
    paired: int | None
    reference_failures: int
    failure_exceed: float
    source: str


def failure_upper(failures: int, draws: int) -> float:
    """The one-sided :data:`FAILURE_LEVEL` Clopper-Pearson upper bound on a failure rate."""
    if failures >= draws:
        return 1.0
    return float(beta.ppf(FAILURE_LEVEL, failures + 1, draws - failures))


def failure_exceed(failures: int, draws: int, replicates: int) -> float:
    """``P(Binomial(replicates, upper) > floor(FAILURE_BOUND * replicates))``."""
    allowed = math.floor(study.FAILURE_BOUND * replicates)
    return float(binom.sf(allowed, replicates, failure_upper(failures, draws)))


def _wide(rows: pd.DataFrame, names: Sequence[str], column: str) -> pd.DataFrame:
    return rows.pivot(index="replicate", columns="estimand", values=column)[list(names)]


def _reading(
    rows: pd.DataFrame, names: Sequence[str], prefix: str
) -> tuple[dict[str, int], sizing.PilotGroup]:
    hits = _wide(rows, names, "covered")
    error = _wide(rows.assign(error=rows["estimate"] - rows["truth"]), names, "error")
    standardized = error / error.std(ddof=1)
    group = sizing.PilotGroup(
        standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(standardized)
    )
    return {f"{prefix}/{name}": int(hits[name].sum()) for name in names}, group


def _primary(
    covered: Mapping[str, int], draws: int, implementations: int, groups: list[sizing.PilotGroup]
) -> int:
    share = sizing.PRIMARY_CLAUSE_SHARE
    return sizing.primary_replicates(
        study.STUDY.margins,
        covered,
        draws,
        implementations=implementations,
        coverage_pass=share,
        bias_groups=groups,
        bias_pass=share,
    )


@dataclass(frozen=True)
class PairedLegs:
    """One cell's paired statistics on the pilot and their spread at the pilot's size.

    Parameters
    ----------
    rmse_ratio, coverage_difference, calibration_excess : float
        The full-pilot values of the three non-inferiority statistics.
    rmse_sd, coverage_sd, calibration_sd : float
        Their bootstrap standard deviations at the pilot's size.
    difference_mean, difference_sd, pooled_sd : float
        The paired mean difference, its spread per replicate, and the pooled spread the
        similarity margin is a share of.
    draws : int
        The paired pilot draws.
    """

    rmse_ratio: float
    coverage_difference: float
    calibration_excess: float
    rmse_sd: float
    coverage_sd: float
    calibration_sd: float
    difference_mean: float
    difference_sd: float
    pooled_sd: float
    draws: int

    def pass_probability(self, replicates: int, *, preregistered: bool = False) -> float:
        """The normal-approximation chance every leg passes at ``replicates`` replicates.

        ``preregistered`` leaves out the RMSE and calibration legs, which the pre-registered
        red signature at :data:`~tests.studies.dose_response_npcausal.EXPECTED_RED_DOSE` admits.
        """
        margins = study.STUDY.margins
        z = float(norm.ppf(margins.confidence_level))
        scale = math.sqrt(self.draws / replicates)
        rmse = self.rmse_sd * scale
        coverage = self.coverage_sd * scale
        probability = 1.0
        if not preregistered:
            probability *= float(
                norm.cdf((margins.rmse_noninferiority - self.rmse_ratio - z * rmse) / rmse)
            )
        probability *= float(
            norm.cdf(
                (self.coverage_difference - margins.coverage_noninferiority - z * coverage)
                / coverage
            )
        )
        if not preregistered:
            spread = self.calibration_sd * scale
            if z * spread > margins.calibration_noninferiority:
                return 0.0
            probability *= float(
                norm.cdf(
                    (margins.calibration_noninferiority - self.calibration_excess - z * spread)
                    / spread
                )
            )
        quantile = float(t.ppf(0.5 + margins.confidence_level / 2.0, replicates - 1))
        se = self.difference_sd / math.sqrt(replicates)
        reach = margins.paired_difference * self.pooled_sd - quantile * se
        if reach <= 0.0:
            return 0.0
        probability *= float(
            norm.cdf((reach - self.difference_mean) / se)
            - norm.cdf((-reach - self.difference_mean) / se)
        )
        return probability


def paired_legs(subject: pd.DataFrame, reference: pd.DataFrame, name: str) -> PairedLegs:
    """One cell's :class:`PairedLegs` from the draws both implementations fitted."""
    left = subject.loc[subject["estimand"] == name].set_index("replicate")
    right = reference.loc[reference["estimand"] == name].set_index("replicate")
    shared = left.index.intersection(right.index)
    left, right = left.loc[shared], right.loc[shared]
    truth = float(left["truth"].iloc[0])
    columns = {
        "es": left["estimate"].to_numpy(),
        "er": right["estimate"].to_numpy(),
        "ss": left["std_error"].to_numpy(),
        "sr": right["std_error"].to_numpy(),
        "cs": left["covered"].to_numpy(dtype=float),
        "cr": right["covered"].to_numpy(dtype=float),
    }

    def statistics(index: np.ndarray) -> np.ndarray:
        es, er = columns["es"][index], columns["er"][index]
        rmse = math.sqrt(np.mean((es - truth) ** 2) / np.mean((er - truth) ** 2))
        coverage = float(np.mean(columns["cs"][index] - columns["cr"][index]))
        calibration = abs(columns["ss"][index].mean() / es.std(ddof=1) - 1.0) - abs(
            columns["sr"][index].mean() / er.std(ddof=1) - 1.0
        )
        return np.array([rmse, coverage, calibration])

    count = len(shared)
    rng = np.random.default_rng(stream_seed(study.STUDY, "pilot", "paired", name))
    resampled = np.array(
        [statistics(rng.integers(0, count, count)) for _ in range(PAIRED_BOOTSTRAP)]
    )
    full = statistics(np.arange(count))
    difference = columns["es"] - columns["er"]
    spread = resampled.std(axis=0, ddof=1)
    return PairedLegs(
        rmse_ratio=float(full[0]),
        coverage_difference=float(full[1]),
        calibration_excess=float(full[2]),
        rmse_sd=float(spread[0]),
        coverage_sd=float(spread[1]),
        calibration_sd=float(spread[2]),
        difference_mean=float(difference.mean()),
        difference_sd=float(difference.std(ddof=1)),
        pooled_sd=float(math.sqrt(0.5 * (columns["es"].var(ddof=1) + columns["er"].var(ddof=1)))),
        draws=count,
    )


def paired_replicates(
    subject: pd.DataFrame, reference: pd.DataFrame, names: Sequence[str]
) -> int | None:
    """The smallest multiple of 400 at which every paired leg passes with probability 0.95.

    The probability is the product over cells and legs of each leg's normal-approximation pass
    chance (:meth:`PairedLegs.pass_probability`), the RMSE and calibration legs left out at
    the pre-registered red dose.  ``None`` when :data:`PAIRED_CAP` does not reach it.
    """
    legs = {name: paired_legs(subject, reference, name) for name in names}
    replicates = sizing.BUDGET_STEP
    while replicates <= PAIRED_CAP:
        product = 1.0
        for name, cell in legs.items():
            dose = study.PILOT_PAIRS[name][0]
            product *= cell.pass_probability(
                replicates, preregistered=dose == study.EXPECTED_RED_DOSE
            )
        if product >= PAIRED_PASS:
            return replicates
        replicates += sizing.BUDGET_STEP
    return None


def subject_rows(frame: pd.DataFrame | None = None) -> PilotRows:
    """The subject's rows from this module's pilot."""
    frame = read(OUTPUT) if frame is None else frame
    rows = frame.loc[frame["implementation"] == study.IMPLEMENTATION]
    lost = frozenset(int(r) for r in rows.loc[rows["refusal"].fillna("") != "", "replicate"])
    return PilotRows(
        rows.loc[rows["refusal"].fillna("") == ""],
        dict.fromkeys(study.CANDIDATE_BANDWIDTHS, lost),
        int(frame["replicate"].nunique()),
        study.IMPLEMENTATION,
    )


@functools.cache
def knife_edges(draws: int = DRAWS) -> dict[float, frozenset[int]]:
    """The pilot draws whose call at each candidate is a :func:`knife_edge`."""
    doses = [draw(r)["A"].to_numpy() for r in range(draws)]
    return {
        h: frozenset(r for r, a in enumerate(doses) if knife_edge(a, study.NPCAUSAL_BANDWIDTHS[h]))
        for h in study.CANDIDATE_BANDWIDTHS
    }


def sketch_reference(frame: pd.DataFrame | None = None) -> PilotRows:
    """The transcription's rows from this module's pilot, standing in for ``ctseff``.

    A :func:`knife_edge` call counts as failed, since R may fail it whatever the
    transcription returned; its rows leave the reading with it.
    """
    frame = read(OUTPUT) if frame is None else frame
    rows = frame.loc[frame["implementation"] == SKETCH]
    refused = rows.loc[rows["refusal"].fillna("") != ""]
    edges = knife_edges(int(frame["replicate"].nunique()))
    failed = {
        h: frozenset(int(r) for r in refused.loc[refused["bandwidth"] == h, "replicate"]) | edges[h]
        for h in study.CANDIDATE_BANDWIDTHS
    }
    return PilotRows(
        rows.loc[rows["refusal"].fillna("") == ""],
        failed,
        int(frame["replicate"].nunique()),
        SKETCH,
    )


def r_reference() -> PilotRows:
    """The R pilot's ``ctseff`` rows and failure records."""
    rows = read(R_OUTPUT)
    failures = (
        read(R_FAILURES)
        if R_FAILURES.exists()
        else pd.DataFrame(columns=["replicate", "bandwidth"])
    )
    failed = {
        h: frozenset(int(r) for r in failures.loc[failures["bandwidth"] == h, "replicate"])
        for h in study.CANDIDATE_BANDWIDTHS
    }
    metadata = json.loads(R_METADATA.read_text(encoding="utf-8"))
    return PilotRows(rows, failed, int(metadata["draws"]), study.REFERENCE)


def declaration_from_pilot(
    subject: PilotRows | None = None, reference: PilotRows | None = None
) -> Declaration:
    """The declared bandwidths and replicate count, read off the pilots.

    For each consecutive pair of candidate bandwidths in order, the count is the largest of
    the shared rule with the subject's rows for both implementations (``assumed``), the shared
    rule with each implementation's own rows on the draws both fitted (``own``), and
    :func:`paired_replicates`.  The first pair whose reference and subject failures keep
    :func:`failure_exceed` at or below :data:`FAILURE_EXCEED` at that count is declared.  The
    reference is the committed R pilot when it exists and the transcription otherwise.
    """
    subject = subject_rows() if subject is None else subject
    if reference is None:
        reference = r_reference() if R_OUTPUT.exists() else sketch_reference()
    candidates = study.CANDIDATE_BANDWIDTHS
    for low, high in itertools.pairwise(candidates):
        names = [name for name, (_, h, _) in study.PILOT_PAIRS.items() if h in (low, high)]
        lost = reference.failed[low] | reference.failed[high]
        subject_lost = subject.failed[low] | subject.failed[high]
        subject_cells = subject.rows.loc[subject.rows["estimand"].isin(names)]
        reference_cells = reference.rows.loc[
            reference.rows["estimand"].isin(names) & ~reference.rows["replicate"].isin(lost)
        ]
        shared = set(subject_cells["replicate"]) & set(reference_cells["replicate"])
        covered, subject_group = _reading(subject_cells, names, study.IMPLEMENTATION)
        assumed = _primary(covered, subject.draws - len(subject_lost), 2, [subject_group])
        both_subject = subject_cells.loc[subject_cells["replicate"].isin(shared)]
        both_reference = reference_cells.loc[reference_cells["replicate"].isin(shared)]
        own_subject, own_subject_group = _reading(both_subject, names, study.IMPLEMENTATION)
        own_reference, own_reference_group = _reading(both_reference, names, study.REFERENCE)
        own = _primary(
            {**own_subject, **own_reference},
            len(shared),
            1,
            [own_subject_group, own_reference_group],
        )
        paired = paired_replicates(both_subject, both_reference, names)
        replicates = max(assumed, own, paired or 0)
        exceed = max(
            failure_exceed(len(lost), reference.draws, replicates),
            failure_exceed(len(subject_lost), subject.draws, replicates),
        )
        if exceed <= FAILURE_EXCEED:
            return Declaration(
                (low, high), replicates, assumed, own, paired, len(lost), exceed, reference.source
            )
    raise RuntimeError(
        "no consecutive pair of candidate bandwidths keeps the reference inside its failure "
        "bound; the design needs a dose law with bounded support (review finding B1, option b)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=DRAWS)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.draws)
