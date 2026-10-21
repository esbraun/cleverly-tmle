"""Sizing rules and a resumable runner shared by ``mtp-boundaries``, ``mtp-pricing`` and
``mtp-tuning``.

**The arm-pooled calibration rule.**  A calibration positive passes when the 99% bootstrap
interval of its SE ratio lies inside ``[0.93, 1.07]`` and its coverage interval lies inside
``[0.92, 0.98]``.  The SE-ratio half binds.  With ``R`` replications the estimated ratio is
about ``N(rho, sigma_R^2)``, ``sigma_R = c / sqrt(2 R)``, and the interval is
``rho_hat +/- 2.576 sigma_R``.  :data:`C_FIT` is fitted so that a perfectly calibrated cell
passes with probability 0.95 at ``R = 2,400``, the pass probability the beta brief records.

The arms' true ratios are unknown before the run, so a pilot of ``P`` draws per arm estimates
them.  The predictive model is ``rho_j = m + e + u_j``: ``m`` is the observed pooled pilot
mean, ``e ~ N(0, sigma_P^2 / arms)`` is the error of that pooled mean shared by every arm,
and ``u_j ~ N(0, tau^2)`` is each arm's own departure.  ``sigma_P = c / sqrt(2 P)``, and
``tau`` sits at its 80% upper bound from the chi-square of the arm means.  The family-wise
pass probability is ``E_e[prod_j P_j(e)]``, computed by Gauss-Hermite quadrature with 80
nodes.  ``R`` is the smallest multiple of 400 up to 40,000 with family-wise probability at
least 0.95 and per-cell probability at least 0.90.  The model is the reviewer's
(``reviews/x40/rev2/pass_prob_lib.py`` in the beta plans); ``tests/unit/test_mtp_boundaries_
design.py`` reproduces its recorded outputs.

**The per-arm screen.**  Before pooling, each arm's pilot SE ratio must satisfy
``|rho_hat_j - 1| <= 2.576 sigma_P + 0.01`` (:func:`screen_half_width`).  An arm outside the
band is diagnosed alone before ``R`` is set from the others.

**Parts.**  The pilots fit through :func:`run_in_parts`, which fits a list of keys in chunks,
writes each finished chunk to its own compressed CSV, and skips a chunk whose file exists, so a
run that a container restart stops resumes where it stopped.  The assembled rows do not depend
on the chunking.  A study's own property loop runs through ``fit_properties``, which
checkpoints each key (:func:`property_checkpoint`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from cleverly.utils.parallel import map_parallel

#: The two-sided 99% critical value of the SE-ratio interval.
Z = float(norm.ppf(0.995))
#: The half-width of the calibration SE-ratio band, ``[0.93, 1.07]``.
HALF = 0.07
#: The smallest and largest ``R`` the rule considers, and its step.
R_STEP = 400
R_MAX = 40_000
#: The family-wise and per-cell pass probabilities the rule requires.
FAMILY_TARGET = 0.95
CELL_TARGET = 0.90
#: The pilot mean's distance from 1 at which a family is a defect suspect.
OFF_MEAN = 0.015
#: The slack the per-arm screen adds to ``2.576 sigma_P``.
SCREEN_SLACK = 0.01
#: The learner screen's band on the pilot SE ratio (``mtp-pricing``).
LEARNER_BAND = (0.95, 1.05)

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def cell_pass(rho: Any, replicates: int, c: float) -> np.ndarray:
    """The probability that a cell with true SE ratio ``rho`` passes at ``replicates``."""
    rho = np.asarray(rho, dtype=float)
    sigma = c / np.sqrt(2.0 * replicates)
    slack = HALF - Z * sigma
    if slack <= 0:
        return np.zeros_like(rho)
    return np.asarray(
        norm.cdf((slack - (rho - 1.0)) / sigma) - norm.cdf((-slack - (rho - 1.0)) / sigma)
    )


def _calibrate_c() -> float:
    low, high = 0.9, 1.3
    for _ in range(60):
        middle = 0.5 * (low + high)
        if cell_pass(np.array([1.0]), 2400, middle)[0] > 0.95:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


#: The spread constant of the SE-ratio estimate, ``1.0692``.
C_FIT = _calibrate_c()


def pilot_sigma(pilot: int, c: float = C_FIT) -> float:
    """``sigma_P``: the standard deviation of one arm's pilot SE ratio at ``pilot`` draws."""
    return float(c / np.sqrt(2.0 * pilot))


def tau_upper(pilot: int, arms: int, c: float = C_FIT) -> float:
    """The 80% upper bound of the between-arm spread ``tau`` when the arms agree."""
    sigma = pilot_sigma(pilot, c)
    factor = chi2.median(arms - 1) / chi2.ppf(0.2, arms - 1)
    return float(sigma * np.sqrt(max(factor - 1.0, 0.0)))


def pass_probabilities(
    replicates: int, mean: float, shared: float, tau: float, arms: int, c: float = C_FIT
) -> tuple[float, float]:
    """``(per-cell, family-wise)`` pass probability under the arm-pooled model.

    ``shared`` is the variance of the pooled pilot mean, ``sigma_P^2 / arms``.
    """
    per_e = []
    for e in np.sqrt(shared) * _GH_X:
        rho = mean + e + tau * _GH_X
        per_e.append(np.sum(_GH_W * cell_pass(rho, replicates, c)))
    values = np.asarray(per_e)
    return float(np.sum(_GH_W * values)), float(np.sum(_GH_W * values**arms))


def smallest_replicates(
    mean: float, pilot: int, arms: int, *, tau: float | None = None, c: float = C_FIT
) -> tuple[int | None, float, float]:
    """``(R, per-cell, family-wise)``: the smallest ``R`` the rule accepts, or ``None``.

    ``tau`` defaults to :func:`tau_upper`.  Where no ``R`` up to :data:`R_MAX` is accepted the
    probabilities are those at :data:`R_MAX`.
    """
    sigma = pilot_sigma(pilot, c)
    spread = tau_upper(pilot, arms, c) if tau is None else tau
    for replicates in range(R_STEP, R_MAX + 1, R_STEP):
        cell, family = pass_probabilities(replicates, mean, sigma**2 / arms, spread, arms, c)
        if family >= FAMILY_TARGET and cell >= CELL_TARGET:
            return replicates, cell, family
    return None, *pass_probabilities(R_MAX, mean, sigma**2 / arms, spread, arms, c)


def tau_from_pilot(ratios: Sequence[float], pilot: int, c: float = C_FIT) -> float:
    """The 80% upper bound of ``tau`` from observed pilot SE ratios, one per arm.

    The arm means' sum of squares about their mean, over ``sigma_P^2``, is chi-square with
    ``arms - 1`` degrees of freedom plus the ``tau^2`` excess.  The bound solves
    ``(sum of squares) / (sigma_P^2 + tau^2) = chi2.ppf(0.2, arms - 1)``.
    """
    values = np.asarray(ratios, dtype=float)
    sigma = pilot_sigma(pilot, c)
    squares = float(np.sum((values - values.mean()) ** 2))
    bound = squares / chi2.ppf(0.2, values.size - 1) - sigma**2
    return float(np.sqrt(max(bound, 0.0)))


def replicates_from_pilot(ratios: Sequence[float], pilot: int, c: float = C_FIT) -> dict[str, Any]:
    """The declared rule applied to a pilot: the screen, the contingency and ``R``.

    Returns the screened arms, the pooled mean of the arms inside the band, whether the
    family is a defect suspect, and ``(R, per-cell, family-wise)`` from the arms inside the
    band.  The predictive ``tau`` is the larger of :func:`tau_upper` (agreeing arms) and
    :func:`tau_from_pilot` (the observed spread), so a spread pilot never lowers ``R``.
    """
    values = np.asarray(ratios, dtype=float)
    half = screen_half_width(pilot, c)
    inside = np.abs(values - 1.0) <= half
    kept = values[inside]
    if kept.size < 2:
        return {"screened": inside.tolist(), "mean": float("nan"), "suspect": True, "R": None}
    mean = float(kept.mean())
    spread = max(tau_upper(pilot, kept.size, c), tau_from_pilot(kept, pilot, c))
    replicates, cell, family = smallest_replicates(mean, pilot, kept.size, tau=spread, c=c)
    return {
        "screened": inside.tolist(),
        "mean": mean,
        "tau": spread,
        "suspect": bool(abs(mean - 1.0) >= OFF_MEAN or replicates is None),
        "R": replicates,
        "per_cell": cell,
        "family_wise": family,
    }


def screen_half_width(pilot: int, c: float = C_FIT) -> float:
    """The per-arm screen's half-width, ``2.576 sigma_P + 0.01``."""
    return Z * pilot_sigma(pilot, c) + SCREEN_SLACK


def learner_screen_pass(pilot: int, c: float = C_FIT) -> float:
    """The probability that a calibrated candidate's pilot SE ratio lands in the band."""
    sigma = pilot_sigma(pilot, c)
    return float(2.0 * norm.cdf((LEARNER_BAND[1] - 1.0) / sigma) - 1.0)


def necessity_power(
    standardized_bias: float, replicates: int, margin: float = 0.25, z: float = Z
) -> float:
    """The probability that a control's bias is established outside the margin.

    ``bias_discriminated`` needs the 99% interval of the standardized bias to clear
    ``margin``.  The standardized bias has standard error about ``1 / sqrt(R)``, so the
    probability is ``Phi(sqrt(R) (|delta| - margin) - z)``.
    """
    return float(norm.cdf(np.sqrt(replicates) * (abs(standardized_bias) - margin) - z))


def equivalence_power(
    standardized_bias: float, replicates: int, margin: float = 0.25, z: float = Z
) -> float:
    """The probability that a positive arm's bias is established inside the margin."""
    root = np.sqrt(replicates)
    upper = norm.cdf(root * (margin - standardized_bias) - z)
    lower = norm.cdf(root * (-margin - standardized_bias) + z)
    return float(max(upper - lower, 0.0))


#: The paired draws a pair pilot needs before it can separate a powered pair from an
#: unpowered one: the MC SE of a standardized bias is then about ``1 / sqrt(100) = 0.1``
#: own SD, against the 0.12 own SD between the margin 0.25 and the 0.372 that power 0.90 needs
#: at R = 1,000 (design review M2).
PAIR_PILOT_DRAWS = 100


def pair_reading(
    declared: Any, control: Any, truth: Any, replicates: int, threshold: float = 0.10
) -> dict[str, float]:
    """A pair pilot read on the scales the verdicts read: each arm's own empirical SD.

    ``bias_discriminated`` standardizes a control's bias by the control's own SD, and
    ``bias_equivalent`` the positive's by its own; ``necessity_verdicts`` reads the paired
    displacement in the positive arm's SD.  ``declared`` and ``control`` are the two arms'
    estimates on the same draws, and ``truth`` their truths.
    """
    declared = np.asarray(declared, dtype=float)
    control = np.asarray(control, dtype=float)
    truth = np.asarray(truth, dtype=float)
    k = declared.size
    own = float(np.std(declared, ddof=1))
    control_sd = float(np.std(control, ddof=1))
    positive = float(np.mean(declared - truth) / own)
    negative = float(np.mean(control - truth) / control_sd)
    moved = control - declared
    displacement = float(abs(np.mean(moved)) / own)
    return {
        "draws": float(k),
        "positive_standardized_bias": positive,
        "positive_mc_se": float(1.0 / np.sqrt(k)),
        "equivalence_power": equivalence_power(positive, replicates),
        "control_standardized_bias": negative,
        "control_mc_se": float(1.0 / np.sqrt(k)),
        "necessity_power": necessity_power(negative, replicates),
        "displacement": displacement,
        "displacement_mc_se": float(np.std(moved, ddof=1) / np.sqrt(k) / own),
        "displacement_clears": float(displacement >= threshold),
    }


# ---------------------------------------------------------------------------- the gate


def gating_findings(
    record: Any,
    cells: Sequence[str],
    truth_implementations: Sequence[str] = (),
    calibration: str = "interval_calibration",
) -> list[str]:
    """Every red gating verdict of ``record``'s committed tables that no owner may hold.

    The record publishes under ``reporting``, so the regeneration does not refuse a red row;
    this is the mechanical gate (design review M7).  ``cells`` are ``property/cell`` keys, and
    every truth row of each of ``truth_implementations`` gates too.  A red gating row passes
    only as a finite-sample red under the beta brief's three-condition reading: it is a
    calibration positive or a truth row, the 99% interval of its SE ratio contains 1, its
    coverage interval contains 0.95, and :data:`~tests.studies.evidence.red_cells.OWNERS`
    names its owner (the owner row records the diagnosis that found no defect).  A red control
    or pair arm is a defect suspect and never passes.  Without committed tables there is
    nothing to read, and the result is empty.

    Parameters
    ----------
    record : StudyRecord
        The study.
    cells : sequence of str
        The gating property cells, as ``property/cell``.
    truth_implementations : sequence of str
        The implementations whose truth rows gate.
    calibration : str
        The property family whose ``positive`` rows may be finite-sample reds.

    Returns
    -------
    list of str
        One line per inadmissible red row.
    """
    from tests.studies.evidence import red_cells

    if not record.artifact(red_cells.FRAMES["properties"]).exists():
        return []
    tables = red_cells.frames(record)
    gating = set(cells)
    out: list[str] = []

    def finite_sample(row: Any) -> bool:
        return bool(
            row.se_ratio_ci_lower <= 1.0 <= row.se_ratio_ci_upper
            and row.coverage_ci_lower <= 0.95 <= row.coverage_ci_upper
        )

    def judge(key: Any, row: Any, may_be_finite: bool) -> None:
        owner = red_cells.OWNERS.get(key)
        if not may_be_finite:
            out.append(f"{key.key} is a red control or pair arm: a defect suspect")
        elif not finite_sample(row):
            out.append(f"{key.key} is red outside the finite-sample reading")
        elif owner is None:
            out.append(f"{key.key} is a finite-sample red with no owner row")

    seen: set[str] = set()
    for row in tables["properties"].itertuples():
        name = f"{row.property}/{row.cell}"
        if name not in gating:
            continue
        seen.add(name)
        if bool(row.passed) and bool(row.property_passed):
            continue
        key = red_cells.RedKey(record.slug, "property", name)
        judge(key, row, row.property == calibration and row.role == "positive")
    out += [f"{name} is a gating cell the committed table does not hold" for name in gating - seen]
    for row in tables["performance"].itertuples():
        if row.implementation not in truth_implementations or bool(row.passed):
            continue
        name = f"{row.implementation}/{row.scenario}/{row.estimand}"
        judge(red_cells.RedKey(record.slug, "truth", name), row, True)
    return sorted(out)


# ---------------------------------------------------------------------------- parts


def property_checkpoint(default: Path | None = None) -> Path | Literal["ambient"] | None:
    """Where a study's property loop keeps its finished keys, so a stopped run resumes.

    Under ``tests/canonical/regenerate.py`` the loop runs inside the driver's checkpoint
    context, whose scratch survives a container restart, and the answer is ``"ambient"``.
    Outside it a caller names the directory, or the environment variable
    ``CLEVERLY_STUDY_PARTS`` does, and without either nothing is kept.
    """
    from tests.studies.evidence.replication import checkpoint_root

    if checkpoint_root() is not None:
        return "ambient"
    if default is not None:
        return default
    named = os.environ.get("CLEVERLY_STUDY_PARTS")
    return Path(named) if named else None


def run_in_parts(
    fit_one: Callable[..., list[dict[str, Any]]],
    keys: Sequence[tuple[Hashable, ...]],
    *,
    label: str,
    root: Path | None,
    chunk: int = 50,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit every key, ``chunk`` keys to a part, and resume from the parts on disk.

    ``fit_one(*key)`` returns a list of row dictionaries.  Each finished part is written to
    ``root / label / part-<index>.csv.gz`` through a temporary file and an atomic rename, so
    a part on disk is always complete.  With ``root=None`` nothing is written.
    """
    frames: list[pd.DataFrame] = []
    directory = None if root is None else Path(root) / label
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(keys), chunk)):
        batch = list(keys[start : start + chunk])
        path = None if directory is None else directory / f"part-{index:05d}.csv.gz"
        if path is not None and path.exists():
            frames.append(pd.read_csv(path, float_precision="round_trip"))
            continue
        outcomes = map_parallel(fit_one, batch, n_jobs=n_jobs)
        part = pd.DataFrame([row for rows in outcomes for row in rows])
        if path is not None:
            temporary = path.with_suffix(".tmp")
            part.to_csv(temporary, index=False, compression="gzip", float_format="%.17g")
            os.replace(temporary, path)
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------- exactness

#: The columns of ``exactness.csv``: one row per exactness-witness estimand.
EXACTNESS_COLUMNS = (
    "scenario",
    "estimand",
    "pairs",
    "max_abs_estimate_gap",
    "max_abs_se_gap",
    "gate",
    "passed",
)

#: The audit name the regeneration driver and the failure ledger read.
EXACTNESS_AUDIT = "exactness witness"


def exactness_table(
    rows: pd.DataFrame,
    *,
    subject: str,
    reference: str,
    scenarios: Sequence[str],
    gate: float,
) -> pd.DataFrame:
    """The largest paired gap of each exactness-witness estimand, judged against ``gate``.

    A witness pairs the subject and the reference on one replication's sample, with the same
    ratios and shifted values on both sides, so their estimates and standard errors agree to
    numerical precision.  A row passes only when every paired estimate and every paired
    standard error differ by less than ``gate``.  A replication that either side lost is not
    a pair, and the failure accounting counts it.  An estimand with no pairs reads red.
    """
    keys = ["scenario", "replicate", "estimand"]
    chosen = rows.loc[rows["scenario"].isin(scenarios)]
    sides = {
        name: chosen.loc[chosen["implementation"] == name, [*keys, "estimate", "std_error"]]
        for name in (subject, reference)
    }
    paired = sides[subject].merge(sides[reference], on=keys, suffixes=("_s", "_r"))
    paired["estimate_gap"] = (paired["estimate_s"] - paired["estimate_r"]).abs()
    paired["se_gap"] = (paired["std_error_s"] - paired["std_error_r"]).abs()
    estimands = chosen[["scenario", "estimand"]].drop_duplicates()
    summary = (
        paired.groupby(["scenario", "estimand"])
        .agg(
            pairs=("replicate", "size"),
            max_abs_estimate_gap=("estimate_gap", "max"),
            max_abs_se_gap=("se_gap", "max"),
        )
        .reset_index()
    )
    table = estimands.merge(summary, on=["scenario", "estimand"], how="left")
    table["pairs"] = table["pairs"].fillna(0).astype(int)
    table["gate"] = gate
    table["passed"] = (
        (table["pairs"] > 0)
        & (table["max_abs_estimate_gap"] < gate)
        & (table["max_abs_se_gap"] < gate)
    )
    return table.loc[:, list(EXACTNESS_COLUMNS)].sort_values(
        ["scenario", "estimand"], ignore_index=True
    )


def exactness_failures(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The ``scientific_failures`` frame of a study with exactness witnesses."""
    return {EXACTNESS_AUDIT: table.loc[~table["passed"].astype(bool)]}
