"""Property families of ``mtp-tuning``: the tuned-cap cells of law Q.

Every cell reads ``ey_regimen[tuned_cap]`` under ``Shift(1.0, cap=c(h))`` at both nodes of law Q
(:mod:`tests.studies.canonical_mtp_tuning`), at n = 2,000.

============================================  ===============================================
calibration positive                          construction and truth
============================================  ===============================================
``population_dose_quantile_tau2``             one fold, ``DoseQuantile``, oracle density;
                                              :math:`\\Psi_0`
``population_quantile_regression_tau2``       one fold, ``LinearQuantileRegression``; :math:`\\Psi_0`
``population_dose_quantile_crossfit_tau2``    five folds, oracle density, ratio at
                                              :math:`\\theta_n`; :math:`\\Psi_0`
``population_dose_quantile_crossfit_``        five folds, the declared classifier at
``classifier_tau2``                           :math:`\\theta^{(-k)}`; :math:`\\Psi_0`
``fold_average_dose_quantile_tau2``           five folds, fold-average target; the mean of
                                              :math:`\\Phi(\\theta^{(-k)})`, per replication
``fold_average_quantile_regression_tau2``     the same with ``LinearQuantileRegression``
============================================  ===============================================

Each positive has the shared ``shrunken_se_control`` and ``noise_control``, and one derived
control of its own, at no extra fit:

* ``term_dropped_control`` (population cells): the same estimate with the standard error of
  the curve without :math:`\\sum_t \\partial_t\\Phi' \\mathrm{IF}_t`.  The term is load-bearing,
  so the SE ratio falls below the band.  Expected ratio 0.612 (dose quantile) and 0.832
  (regression), from ``lawQ.log``.
* ``population_truth_control`` (fold-average cells): the same rows scored on :math:`\\Psi_0`.
  The fold-average interval does not cover the population target, so the target must be
  declared.  Same reading.

Both read ``se_ratio_leaves_band(..., "below")``
(:data:`~tests.studies.evidence.property_verdicts.SE_CONTROL_SIDES`).  The sizes and the
pilot are ``mtp-pricing``'s: the six arms enter the 18-arm pooled rule
(:func:`tests.studies.mtp_pricing_properties.pooled_replicates`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests import law_q_tuned as lawq
from tests.law_q_tuned import sample
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_mtp_tuning as study
from tests.studies import mtp_pricing_properties as pricing
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.oracle_density_bins import fit_bytes as one_fit_bytes

STUDY = study.STUDY
CALIBRATION_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
FOLDS = 5
PILOT_DRAWS = 2_000

#: Each calibration label: ``(tuner kind, target, folds, ratio route)``.
CELLS: dict[str, tuple[str, str, int, str]] = {
    "population_dose_quantile_tau2": ("dose_quantile", "population", 1, "density"),
    "population_quantile_regression_tau2": ("quantile_regression", "population", 1, "density"),
    "population_dose_quantile_crossfit_tau2": ("dose_quantile", "population", FOLDS, "density"),
    "population_dose_quantile_crossfit_classifier_tau2": (
        "dose_quantile",
        "population",
        FOLDS,
        "classifier",
    ),
    "fold_average_dose_quantile_tau2": ("dose_quantile", "fold_average", FOLDS, "density"),
    "fold_average_quantile_regression_tau2": (
        "quantile_regression",
        "fold_average",
        FOLDS,
        "density",
    ),
}
assert tuple(CELLS) == study.CALIBRATION_LABELS
#: Provisional: the expected ``R`` of the 18-arm pooled rule at a pilot mean of 1.005.
CALIBRATION_REPLICATES: dict[str, int] = dict.fromkeys(CELLS, pricing.S2_REPLICATES)
#: The expected SE ratio of each derived control (``lawQ.log``), which the design test
#: checks against the oracle's curves.
EXPECTED_CONTROL_RATIO = {"dose_quantile": 0.612, "quantile_regression": 0.832}

#: The noise scale of each ``noise_control``: ``mtp-pricing``'s ``NOISE_FACTOR`` times the arm's
#: own pilot SD times sqrt(n) (design review B2), the pilot's ``noise_sd`` (2,000 draws,
#: 2026-10-15).
NOISE_SD: dict[str, float] = {
    "population_dose_quantile_tau2": 0.528,
    "population_quantile_regression_tau2": 0.437,
    "population_dose_quantile_crossfit_tau2": 0.53,
    "population_dose_quantile_crossfit_classifier_tau2": 0.531,
    "fold_average_dose_quantile_tau2": 0.326,
    "fold_average_quantile_regression_tau2": 0.367,
}

#: The cells that gate the merge (design review M7): all six positives and their three derived
#: controls each, every published property cell of the record.
GATING_CELLS: tuple[str, ...] = tuple(
    f"{family}/{cell}" for family, cells in study.PROPERTY_CELLS.items() for cell in cells
)
ESTIMAND = study.ESTIMAND
#: The name a fit reports the fold-average target under.
FOLD_ESTIMAND = f"ey_regimen[{study.LABEL}, fold average]"


def estimand(target: str) -> str:
    return ESTIMAND if target == "population" else FOLD_ESTIMAND


def calibration_fit(label: str, frame: pd.DataFrame) -> Any:
    kind, target, folds, route = CELLS[label]
    learner = study.classifier() if route == "classifier" else None
    return study.fit(frame, kind, target, n_folds=folds, ratio=route, treatment_learner=learner)


def term_dropped_se(result: Any) -> float:
    """The reported standard error rescaled to the curve without the tuning term."""
    fitted = result.fits[study.LABEL]
    curve = np.asarray(fitted.influence_curve_scaled, dtype=float)
    term = np.asarray(fitted.tuning_term, dtype=float)
    return float(result[ESTIMAND].std_error * np.std(curve - term) / np.std(curve))


# ---------------------------------------------------------------------------- kernel bias

#: The declared ceiling on the relative kernel bias at n = 2,000 (X40 plan section 2.5, m5).
KERNEL_BIAS_LIMIT = 0.01
_HERMITE = np.polynomial.hermite_e.hermegauss(60)
_Z, _ZW = _HERMITE[0], _HERMITE[1] / np.sum(_HERMITE[1])


def _smoothed(density: Any, at: np.ndarray, h: float, *args: np.ndarray) -> np.ndarray:
    """``(g * phi_h)(at) = E[g(at - h Z)]``, the expectation of a normal-kernel estimate."""
    return sum(w * density(at - h * z, *args) for z, w in zip(_Z, _ZW, strict=True))


def _histories(law: Any, node: int) -> tuple[dict[str, np.ndarray], np.ndarray, Any]:
    """The observed history of ``node`` on its quadrature grid, its mass and its density."""
    h, mass = law._history(node)
    if node == 1:
        return h, mass, (lambda a: law.g1(a, h["W"][..., None]))
    return h, mass, (lambda a: law.g2(a, h["W"][..., None], h["A1"][..., None], h["L2"][..., None]))


def _gap_scale(law: Any, node: int) -> float:
    """Silverman's scale ``0.9 min(sd, IQR / 1.34)`` of ``A_t - c_t(H_t)`` on the law.

    The scale of ``A_t - e_t`` is the same, since ``e_t = c_t - 1``.  For a dose quantile it
    is the scale of ``A_t`` itself.
    """
    from scipy import optimize

    h, mass, density = _histories(law, node)
    c = law.cap(node, h)
    x, wts = lawq._segments(*lawq.A_RANGE, c[:, None])
    g = density(x) * wts
    gap = x - c[:, None]
    mean = float(np.sum(mass[:, None] * g * gap))
    sd = float(np.sqrt(np.sum(mass[:, None] * g * (gap - mean) ** 2)))
    cdf = law.cdf1 if node == 1 else law.cdf2
    args = (h["W"],) if node == 1 else (h["W"], h["A1"], h["L2"])

    def quantile(q: float) -> float:
        return float(
            optimize.brentq(lambda t: np.sum(mass * cdf(c + t, *args)) - q, -8.0, 8.0, xtol=1e-12)
        )

    iqr = quantile(0.75) - quantile(0.25)
    return 0.9 * min(sd, iqr / 1.34)


def _relative(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(estimate - truth)) / np.max(np.abs(truth)))


def kernel_derivative(law: Any, node: int, h: float) -> np.ndarray:
    """The expectation of :func:`~cleverly.interventions.tuning.cap_derivative` at bandwidth
    ``h``: :meth:`~tests.law_q_tuned.LawQ.derivative` with ``g_t(e_t)`` smoothed by the kernel."""
    w, fw = law.w_grid
    if node == 1:
        hist = {"W": w}
        c1 = law.cap(1, hist)
        e1 = c1 - lawq.DELTA
        jump = _smoothed(law.g1, e1, h, w) * (law.q1(w, c1) - law.q1(w, e1))
        return np.sum((fw * jump)[:, None] * law.gradient(1, hist), axis=0)
    c1 = law.cap(1, {"W": w})
    a1, wa = lawq._segments(*lawq.A_RANGE, (c1 - lawq.DELTA)[:, None])
    b = np.where(a1 + lawq.DELTA <= c1[:, None], a1 + lawq.DELTA, a1)
    ww = np.broadcast_to(w[:, None], a1.shape)
    mass = fw[:, None] * law.g1(a1, w[:, None]) * wa
    total = 0.0
    for l2 in (0.0, 1.0):
        ll = np.full(a1.shape, l2)
        p = law.p_l2(ww, b)
        weight = mass * (p if l2 else 1.0 - p)
        hist = {"W": ww, "A1": b, "L2": ll}
        c2 = law.cap(2, hist)
        e2 = c2 - lawq.DELTA
        smoothed = _smoothed(law.g2, e2, h, ww, b, ll)
        jump = smoothed * (law.m(ww, b, ll, c2) - law.m(ww, b, ll, e2))
        total = total + np.sum((weight * jump)[..., None] * law.gradient(2, hist), axis=(0, 1))
    return np.asarray(total)


def kernel_biases(kind: str, rule: str, n: int = CALIBRATION_N) -> dict[str, float]:
    """The relative kernel bias of each kernel quantity of the tuned cap at ``n``, by quadrature.

    Each bandwidth is :func:`~cleverly.interventions.tuning.kernel_bandwidth`'s, with the law's
    own Silverman scale in place of the sample's: ``n^(-1/5)`` for ``f_t(theta_t)`` (dose
    quantile) and ``dPhi_t``, ``n^(-1/3)`` for Powell's ``J_t`` (regression), halved under
    ``"undersmoothed"``.  The keys are ``f1``, ``f2`` or ``J1``, ``J2``, and ``dPhi1``,
    ``dPhi2``; a vector or matrix reads its largest error over its largest entry.
    """
    law = study.law(kind)
    half = 0.5 if rule == "undersmoothed" else 1.0
    out: dict[str, float] = {}
    for node in (1, 2):
        scale = _gap_scale(law, node)
        h = half * scale * n ** (-0.2)
        out[f"dPhi{node}"] = _relative(kernel_derivative(law, node, h), law.derivative(node))
        tuner = law.tuners[node - 1]
        if not tuner.features:
            theta = float(law.theta[node - 1][0])
            f = law.marginal_density(node, theta)
            smoothed = sum(
                w * law.marginal_density(node, theta - h * z) for z, w in zip(_Z, _ZW, strict=True)
            )
            out[f"f{node}"] = abs(smoothed / f - 1.0)
            continue
        hj = half * scale * n ** (-1.0 / 3.0)
        hist, mass = law._history(node)
        x = law._features(node, tuner.features, hist)
        fit = x @ law.theta[node - 1]
        cdf = law.cdf1 if node == 1 else law.cdf2
        dens = law.g1 if node == 1 else law.g2
        args = (hist["W"],) if node == 1 else (hist["W"], hist["A1"], hist["L2"])
        box = (cdf(fit + hj, *args) - cdf(fit - hj, *args)) / (2.0 * hj)
        expected = (x * (mass * box)[:, None]).T @ x
        truth = (x * (mass * dens(fit, *args))[:, None]).T @ x
        out[f"J{node}"] = _relative(expected, truth)
    return out


def declared_bandwidth(kind: str, n: int = CALIBRATION_N) -> str | None:
    """The plan's rule (section 2.5): Silverman's bandwidth if every relative kernel bias at
    ``n`` is at most :data:`KERNEL_BIAS_LIMIT`, else the undersmoothed one if it then is,
    else ``None``: the kind's cells are then declared ``reporting`` before the run."""
    for rule in ("silverman", "undersmoothed"):
        if max(kernel_biases(kind, rule, n).values()) <= KERNEL_BIAS_LIMIT:
            return rule
    return None


class DeclaredLaw:
    """Law Q under one tuner kind and target.

    Parameters
    ----------
    kind : str
        ``"dose_quantile"`` or ``"quantile_regression"``.
    target : str
        ``"population"`` or ``"fold_average"``.
    """

    def __init__(self, kind: str, target: str) -> None:
        self.kind = kind
        self.target = target
        self.name = f"Q_{kind}"

    def truth(self) -> dict[str, float]:
        """:math:`\\Psi_0`; a fold-average cell's rows carry their own truth."""
        return {estimand(self.target): study.population_truth(self.kind)}


FIT_SETS: tuple[tuple[str, str, int, int], ...] = tuple(
    ("interval_calibration", label, CALIBRATION_N, CALIBRATION_REPLICATES[label]) for label in CELLS
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _pilot_seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "diagnostic", "pilot", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    cells = []
    for label, (kind, target, _, _) in CELLS.items():
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
            (study.EXTRA_CONTROLS[label], "control"),
        ):
            cells.append(
                PropertyCell(
                    property="interval_calibration",
                    cell=f"{label}__{cell}",
                    dgp=DeclaredLaw(kind, target),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=CALIBRATION_N,
                    replicates=CALIBRATION_REPLICATES[label],
                    seed=_seed("interval_calibration", label, 0),
                    role=role,
                    estimand=estimand(target),
                )
            )
    return tuple(cells)


#: The cells whose committed truth varies by replication: the positive and its two shared
#: controls on a fold-average target, whose rows carry the mean of the fold maps' values.
#: ``population_truth_control`` is scored on the constant :math:`\\Psi_0` and is not here.
REPLICATE_TRUTH_CELLS = frozenset(
    f"{label}__{cell}"
    for label, (_, target, _, _) in CELLS.items()
    if target == "fold_average"
    for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
)


def replicate_truth(cell: str, replicate: int) -> float:
    """The exact truth one replication of a :data:`REPLICATE_TRUTH_CELLS` cell publishes.

    Redraws the replication's sample from the cell's own stream and refits it, then reads
    the truth through the code that writes the ``truth`` column of the rows.

    Parameters
    ----------
    cell : str
        A member of :data:`REPLICATE_TRUTH_CELLS`.
    replicate : int
        The replication index.
    """
    if cell not in REPLICATE_TRUTH_CELLS:
        raise ValueError(f"{cell} has a truth that does not vary by replication")
    label = cell.split("__", 1)[0]
    kind, _, _, _ = CELLS[label]
    frame = sample(CALIBRATION_N, _seed("interval_calibration", label, replicate))
    return _replicate_truth(kind, "fold_average", calibration_fit(label, frame))


def _replicate_truth(kind: str, target: str, result: Any) -> float:
    """The truth a replication's positive row carries (the one definition both paths read)."""
    population = study.population_truth(kind)
    return population if target == "population" else study.fold_average_truth(result, kind)


def _fit_rows(
    family: str, label: str, replicate: int, n: int, requested: int, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """The positive row and the derived control row of one replication, fitted on ``frame``."""
    kind, target, _, _ = CELLS[label]
    result = calibration_fit(label, frame)
    estimate = result[estimand(target)]
    population = study.population_truth(kind)
    truth = _replicate_truth(kind, target, result)
    shared: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}
    positive = replicate_row(
        property_name=family,
        cell=f"{label}__correctly_specified",
        role="positive",
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
        **shared,
    )
    control = control_row(
        property_name=family,
        cell=f"{label}__{study.EXTRA_CONTROLS[label]}",
        truth=population,
        estimate=float(estimate.psi),
        standard_error=(
            term_dropped_se(result) if target == "population" else float(estimate.std_error)
        ),
        critical=CRITICAL,
        **shared,
    )
    return [positive, control]


def fit_set_rows(
    family: str, label: str, replicate: int, n: int, requested: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """The positive row and the derived control row of one replication."""
    frame = sample(n, _seed(family, label, replicate) if seed is None else seed)
    return _fit_rows(family, label, replicate, n, requested, frame)


#: A property key: ``(family, label, replicate, n, requested)``.
_Key = tuple[str, str, int, int, int]

#: Each positive's derived controls, which the keys name and :func:`calibration_controls` writes
#: after the loop.  The cell's own extra control comes from the worker.
DERIVED_CELLS: dict[str, tuple[str, ...]] = calibration_derived(tuple(CELLS))


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    family, label, replicate, n, _ = key
    return sample(n, _seed(family, label, replicate))


def _fit(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _fit_rows(*key, frame)


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    family, label, *_ = key
    positive = f"{family}/{label}__correctly_specified"
    return [positive, f"{family}/{label}__{study.EXTRA_CONTROLS[label]}", *DERIVED_CELLS[positive]]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def _seed_of(key: _Key) -> int:
    family, label, replicate, *_ = key
    return _seed(family, label, replicate)


def fit_bytes(n: int) -> int:
    return max(2 * one_fit_bytes(n), 2**30)


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6).  Up to ``floor(0.01 R)`` lost
#: replicates are judged by the worst case of each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set.

    The keys of every fit set run in one :func:`~tests.studies.evidence.replication.
    fit_properties` loop.  A finished key is kept under
    :func:`~tests.studies.mtp_study_tools.property_checkpoint` (the driver's checkpoint
    scratch, else ``parts``), so a stopped run resumes.  A ``budget`` run keeps nothing,
    because a key's row carries ``requested``.
    """
    jobs = memory_capped_workers(n_jobs, fit_bytes(CALIBRATION_N))
    keys: list[_Key] = []
    for family, label, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        keys += [(family, label, r, n, requested) for r in range(requested)]
    outcomes = fit_properties(
        keys,
        _fit,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        seed_of=_seed_of,
        checkpoint=tools.property_checkpoint(parts) if budget is None else None,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CELLS),
                efficiency_bounds=NOISE_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes no
#: efficiency band: the noise control and the two derived controls of each cell read the SE ratio.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)


def _pilot_rows(
    family: str, label: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    rows = fit_set_rows(
        family, label, replicate, n, requested, seed=_pilot_seed(family, label, replicate)
    )
    seconds = time.perf_counter() - started
    for row in rows:
        row["seconds"] = seconds / len(rows)
    return rows


def calibration_pilot(
    parts: Path, *, draws: int = PILOT_DRAWS, n_jobs: int = 1, labels: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The six arms' calibration pilot, read with the row's own truth.

    The classifier arm's pilot is also its screen under the rule of
    :mod:`tests.studies.mtp_pricing_properties`: if its SE ratio leaves
    :data:`~tests.studies.mtp_study_tools.LEARNER_BAND`, the cell is declared ``reporting``
    with an owner before the run.
    """
    out: dict[str, Any] = {"arms": {}}
    for label in labels or tuple(CELLS):
        keys = [("interval_calibration", label, r, CALIBRATION_N, draws) for r in range(draws)]
        rows = tools.run_in_parts(
            _pilot_rows, keys, label=f"pilot__{label}", root=parts, n_jobs=n_jobs
        )
        own = rows.loc[rows["cell"] == f"{label}__correctly_specified"]
        dropped = rows.loc[rows["cell"] == f"{label}__{study.EXTRA_CONTROLS[label]}"]
        error = own["estimate"] - own["truth"]
        spread = float(error.std(ddof=1))
        out["arms"][label] = {
            "draws": len(own),
            "noise_sd": float(pricing.NOISE_FACTOR * spread * np.sqrt(CALIBRATION_N)),
            "se_ratio": float(own["std_error"].mean() / spread),
            "standardized_bias": float(error.mean() / spread),
            "coverage": float(own["covered"].mean()),
            "control_se_ratio": float(
                dropped["std_error"].mean()
                / float((dropped["estimate"] - dropped["truth"]).std(ddof=1))
            ),
            "seconds": float(rows["seconds"].sum() / len(own)),
        }
    return out


#: Per-fit seconds and peak memory from the one-replicate smoke (warm process, one core,
#: n = 2,000, 2026-10-15), and the CPU-hours they imply.  The fold-average seconds hold the
#: fold-average truth's quadrature (0.3 s).
_CALIBRATION_SECONDS = {
    "population_dose_quantile_tau2": 0.66,
    "population_quantile_regression_tau2": 0.99,
    "population_dose_quantile_crossfit_tau2": 1.74,
    "population_dose_quantile_crossfit_classifier_tau2": 2.61,
    "fold_average_dose_quantile_tau2": 1.86,
    "fold_average_quantile_regression_tau2": 3.24,
}
_PRIMARY_SECONDS = 1.38  # the two scenarios of one replicate
_HOUR = 3600.0


def _calibration_hours(replicates: int) -> float:
    return sum(_CALIBRATION_SECONDS.values()) * replicates / _HOUR


BUDGET: dict[str, Any] = {
    "seconds": {"primary_replicate": _PRIMARY_SECONDS, "calibration_fit": _CALIBRATION_SECONDS},
    "peak_gib": 0.27,
    "cpu_hours": {
        "calibration_pilot": _calibration_hours(PILOT_DRAWS),
        "calibration": {r: _calibration_hours(r) for r in (6_000, 6_400, 8_000)},
        "primary": study.PRIMARY_REPLICATES * _PRIMARY_SECONDS / _HOUR,
    },
}
BUDGET["cpu_hours"]["python_total"] = (
    BUDGET["cpu_hours"]["calibration_pilot"]
    + BUDGET["cpu_hours"]["calibration"][6_400]
    + BUDGET["cpu_hours"]["primary"]
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot", choices=("calibration",))
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--labels", nargs="*", default=())
    arguments = parser.parse_args(argv)
    result = calibration_pilot(
        arguments.parts,
        draws=arguments.draws or PILOT_DRAWS,
        n_jobs=arguments.jobs,
        labels=tuple(arguments.labels),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    (arguments.parts / f"{arguments.pilot}-pilot.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
