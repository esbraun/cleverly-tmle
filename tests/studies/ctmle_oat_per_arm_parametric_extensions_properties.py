"""Repeated-sampling properties of the ``K``-arm, weighted and population-target stacked curve (X58b2).

Every fitted cell fits ``CTMLE(strategy="oat", oat_design="per_arm")`` with
:class:`~cleverly.learners.ParametricOutcome` and :class:`~cleverly.learners.ParametricMechanism`,
which the support predicate admits, so each estimate reports the exact stacked curve.  The laws,
fits, cells and budgets are declared in
:mod:`tests.studies.canonical_ctmle_oat_per_arm_parametric_extensions`.

=========================  ===============================================================
family                     cells
=========================  ===============================================================
``interval_calibration``   ``<group>__<estimate>__corrected``: SE ratio and coverage of the
                           gated estimates of each group.  ``three_arm`` (in sample,
                           ``reference="1"``): ``ey``, ``ate``, ``log_rr``, ``log_or``,
                           ``par[1]``, ``paf[1]`` and ``ey_obs``.  ``three_arm_shared``: ``ey``
                           and ``ate`` on the strong shared prognosis.  ``binary_population``:
                           ``par``, ``paf`` and ``ey_obs``.  ``three_arm_crossfit``
                           (cross-fitted, ``reference="0"``): ``ey`` and ``ate``.  ``weighted``:
                           ``ey0``, ``ey1``, ``ate``, ``log_rr``, ``log_or``, ``par``, ``paf``
                           and ``ey_obs``.  ``weighted_crossfit``: ``ey0``, ``ey1`` and ``ate``.
                           ``composition`` (weighted, cross-fitted, ``reference="1"``): ``ey``
                           and ``ate``
``interval_calibration``   the controls.  ``<label>__shrunken_se_control`` and
                           ``<label>__noise_control`` are derived from each group's contrast row.
                           ``<group>__<estimate>__shipped_curve__shrunken_se_control`` reports
                           the shipped adaptive-propensity standard error of the same fit.
                           ``three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control``
                           reports ``sqrt(se_0^2 + se_1^2)``.  Each passes by leaving the
                           calibration band on its declared side.
``simultaneous_coverage``  ``<label>__simultaneous_band``: the max-t multiplier band over the
                           declared band set.  ``<label>__pointwise_joint_control`` reads the
                           pointwise intervals of the same fit jointly.
                           ``<label>__shipped_curve_band_control`` builds the band from the
                           shipped curves of the same estimates.  A control passes when the
                           upper end of its joint-coverage interval is below 0.95
``generated_design``       ``three_arm_active__oracle_design__<contrast>`` (``Qbar_0`` pinned,
                           shipped curve) against ``three_arm_active__estimated__<contrast>``
                           (the stacked curve), both cross-fitted, ``reference="0"``, one sample
                           per pair
``type_i_error``           ``three_arm_null__ate_0_vs_1`` and ``three_arm_null__ate_2_vs_1`` on
                           the three-arm sharp null, ``n = 1,000``
=========================  ===============================================================

**Gated ``ey_obs``.**  The curve of ``ey_obs`` is the shipped empirical one.  The support rows
``C11.binary``, ``C11.k_arms`` and ``C11.weighted`` license its interval, so each group that fits
it gates a ``<group>__ey_obs__corrected`` cell (design review m2).

**Reported, not gated.**  The summary carries two quantities that no verdict reads.

* ``covariance_mc``, ``covariance_reported`` and their ratio on the ``ate[0 vs 1]`` row of
  ``three_arm_shared``.
* ``se_ratio_deficit_lower`` and ``se_ratio_deficit_upper``: the paired deficit of the
  generated-design pair, by contrast.

A replicate that raises is not redrawn.  :func:`group_rows` fits through
:func:`~tests.studies.evidence.replication.fit_properties`, so one failed fit costs the cells
that its key writes one replicate each, and the run continues.  An admitted fit with an estimate
whose status is not ``"influence_curve"`` raises in ``fit_cleverly`` and counts the same way, so
a band cell loses a replicate when any estimate of its band set fails.  :data:`FAILURE_BOUND` lets a
cell lose ``floor(0.01 R)`` replicates, and the accounting judges each clause by its worst case
over the lost values.  A lost contrast row also costs the two controls derived from it
(:data:`DERIVED_CELLS`).  Each finished replicate is checkpointed under the driver, so a killed run
resumes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.exceptions import DataError
from cleverly.inference.multiplier import simultaneous_bands
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ctmle_oat_per_arm_parametric_extensions import (
    BUDGETS,
    CALIBRATION_N,
    CONTROL_LABELS,
    GENERATED_CONTRASTS,
    GENERATED_LABEL,
    GROUPS,
    JOINT,
    SHRUNKEN_SE_FACTOR,
    STUDY,
    Band,
    Cell,
    Group,
    cells_of,
    control_bounds,
    fit_cleverly,
    sample,
    slug,
    truth,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    bootstrap_draw_blocks,
    control_row,
    se_ratio_deficit_interval,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows

#: The share of a cell's requested replicates that the accounting lets a run lose, declared for
#: the study's property run (RM39b2).  Up to ``floor(0.01 R)`` lost replicates are judged by
#: the worst case of each clause; one more is ``bound exceeded``.  The runner's
#: ``STUDY.failure_bound`` carries the same value.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The replicate columns a row carries beside the shared ones.
EXTRA_COLUMNS = (
    "arm0_estimate",
    "arm1_estimate",
    "reported_cov01",
)
#: The summary columns no verdict reads.
SUMMARY_COLUMNS = (
    "se_ratio_deficit_lower",
    "se_ratio_deficit_upper",
    "covariance_mc",
    "covariance_reported",
    "covariance_ratio",
    "covariance_ratio_ci_lower",
    "covariance_ratio_ci_upper",
)

#: The cell whose row carries the arm-mean covariance report.
COVARIANCE_CELL = "three_arm_shared__ate_0_vs_1__corrected"


# ------------------------------------------------------------------------------ one replicate

#: ``(group, replicate)``, or ``(group, replicate, requested)`` in the property loop.
Key = tuple[Any, ...]


def _sample_seed(key: Key) -> int:
    return stream_seed(STUDY, "property_sample", key[0], key[1])


def _draw(key: Key) -> pd.DataFrame:
    group = GROUPS[key[0]]
    return sample(group.law, group.n, _sample_seed(key))


def _group_of(key: Key) -> str:
    return key[0]


def _replicate_of(key: Key) -> int:
    return key[1]


def shipped_std_error(result: Any, token: str) -> float:
    """The standard error of the shipped adaptive-propensity curve of ``token``.

    Each draw stores the standard error of its uncorrected curve (``RepeatFit.inference``), on
    the scale of the estimand.  The rule is the median over draws of
    ``sigma_r^2 + (psi_r - psi_med)^2`` of :func:`~cleverly.inference.median_estimates`; with the
    shipped ``sigma_r`` it gives the shipped counterpart, and with one draw it is that draw's
    shipped standard error.  Every fit of this study makes one draw, and the weighted fits store
    the standard error of the weighted shipped curve (plan 13.4, DW6).
    """
    draws = result.repeats
    shipped = np.array([draw.inference[token].shipped_se for draw in draws], dtype=float)
    if np.isnan(shipped).any():
        raise DataError("a draw stored no shipped standard error, so the stack did not run")
    points = np.array([draw.psi[token] for draw in draws], dtype=float)
    return float(np.sqrt(np.median(shipped**2 + (points - np.median(points)) ** 2)))


def _estimate(result: Any, token: str) -> tuple[float, float]:
    """``(value, standard error)`` of ``token``; a ratio on the log scale."""
    name = token.removeprefix("log_")
    estimate = result[name]
    value = float(estimate.log_psi) if token != name else float(estimate.psi)
    return value, float(estimate.plugin_std_error)


def _measure_cell(cell: Cell, result: Any) -> dict[str, Any]:
    """What one cell reads off its fit: no truth, which the driver adds."""
    value, se = _estimate(result, cell.estimand)
    if cell.recipe == "shipped_curve":
        se = shipped_std_error(result, cell.estimand)
    elif cell.recipe == "cross_term_dropped":
        # ``ate[0 vs 1]`` is ``ey[0] - ey[1]``: the standard error without the covariance term.
        first, second = (part.strip() for part in cell.estimand[4:-1].split(" vs "))
        se = float(
            np.hypot(
                result[f"ey[{first}]"].plugin_std_error, result[f"ey[{second}]"].plugin_std_error
            )
        )
    elif cell.recipe != "corrected":
        raise KeyError(cell.recipe)
    measured: dict[str, Any] = {"cell": cell.name, "estimate": value, "std_error": se}
    if cell.report == "covariance":
        curves = result.influence_curves
        measured.update(
            arm0_estimate=float(result["ey[0]"].psi),
            arm1_estimate=float(result["ey[1]"].psi),
            reported_cov01=float(np.cov(curves["ey[0]"], curves["ey[1]"])[0, 1] / result.n),
        )
    return measured


def shipped_banded(result: Any) -> Any:
    """A plug-in fit with the band its shipped curves give on the same estimates.

    The shipped per-arm fit supplies no inference, so it builds no band.  This function calls the
    public :func:`~cleverly.inference.multiplier.simultaneous_bands` on the fit's own estimates
    relabelled ``"influence_curve"``, which the function requires.  The curves, the standard
    errors and the multiplier arguments are those of the fit, and the band equals the one the fit
    would build if it supplied inference.  The point estimates do not read the status.
    """
    estimator = result.estimator
    diagnostic = {
        name: dataclasses.replace(estimate, inference="influence_curve")
        for name, estimate in result.estimates.items()
    }
    bands = simultaneous_bands(
        diagnostic,
        alpha=estimator.alpha_sig,
        n_replicates=estimator.n_multiplier,
        kind=estimator.multiplier_kind,
        random_state=estimator.random_state,
        cluster=result.data.cluster,
    )
    return dataclasses.replace(result, simultaneous=bands)


def _joint_rows(
    group: Group, replicate: int, results: dict[str, Any], requested: int
) -> list[dict[str, Any]]:
    """The joint rows of every band of the group: the band, then each control it declares."""
    out: list[dict[str, Any]] = []
    for band in group.bands:
        truths = {name: truth(group.law, name) for name in band.names}
        shared = {
            "label": band.label,
            "replicate": replicate,
            "n": group.n,
            "requested": requested,
            "pointwise_critical": CRITICAL,
        }
        band_row, pointwise_row = joint_coverage_rows(
            results[band.fit], truths, band.names, **shared
        )
        out.append(band_row)
        if band.pointwise:
            out.append(pointwise_row)
        if band.shipped is not None:
            shipped_row, _ = joint_coverage_rows(
                shipped_banded(results[band.shipped]), truths, band.names, **shared
            )
            shipped_row["cell"] = f"{band.label}__shipped_curve_band_control"
            shipped_row["role"] = "control"
            out.append(shipped_row)
    return out


def _measure(key: Key, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit every configuration of the key's group on one sample and read every fitted cell."""
    group = GROUPS[key[0]]
    results = {fit.name: fit_cleverly(frame, fit) for fit in group.fits}
    measured = [
        _measure_cell(cell, results[cell.fit])
        for cell in cells_of(group.name)
        if cell.family != JOINT
    ]
    return measured, results


def _rows(
    group: Group, replicate: int, measured: list[dict[str, Any]], requested: int
) -> list[dict[str, Any]]:
    """The rows of one replicate: each measurement with its truth, interval and role."""
    by_name = {cell.name: cell for cell in cells_of(group.name)}
    rows = []
    for item in measured:
        cell = by_name[item["cell"]]
        reference = truth(cell.truth, cell.estimand)
        row = control_row(
            property_name=cell.family,
            cell=cell.name,
            replicate=replicate,
            n=group.n,
            requested=requested,
            truth=reference,
            estimate=item["estimate"],
            standard_error=item["std_error"],
            critical=CRITICAL,
            role=cell.role,
        )
        for column in EXTRA_COLUMNS:
            row[column] = item.get(column, np.nan)
        rows.append(row)
    return rows


#: The derived controls of each base cell, as ``property/cell`` names: a lost contrast fit also
#: costs the ``shrunken_se_control`` and the ``noise_control`` of its row.
DERIVED_CELLS = calibration_derived(CONTROL_LABELS, positive_suffix="corrected")


def _cells_of(key: Key) -> list[str]:
    """The ``property/cell`` names that one fit writes, its derived controls included."""
    names = [f"{cell.family}/{cell.name}" for cell in cells_of(key[0])]
    for base, controls in DERIVED_CELLS.items():
        if base in names:
            names.extend(controls)
    return names


def _replication(key: Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Fit one key's sample and return the rows of its cells: ``(group, replicate, requested)``."""
    group = GROUPS[key[0]]
    measured, results = _measure(key, frame)
    return [
        *_rows(group, key[1], measured, key[2]),
        *_joint_rows(group, key[1], results, key[2]),
    ]


def group_rows(group: Group, *, n_jobs: int, replicates: int | None = None) -> pd.DataFrame:
    """Fit ``group`` for ``replicates`` (default its budget) and return its rows.

    The loop runs through :func:`~tests.studies.evidence.replication.fit_properties`, under the
    label ``properties/<group>``.  Under the regeneration driver, each finished replicate is
    checkpointed, and a rerun fits only the replicates without a checkpoint.  A failed fit costs
    the cells of its key one replicate each, and the driver records it.  Outside the driver, a
    failed fit raises ``RuntimeError``.
    """
    requested = group.replicates if replicates is None else replicates
    # The truth of each cell is a cached quadrature: compute it here, so that an error in it
    # stops the run before any fit and each worker reads the cache.
    for cell in cells_of(group.name):
        if cell.family != JOINT:
            truth(cell.truth, cell.estimand)
    outcomes = fit_properties(
        [(group.name, replicate, requested) for replicate in range(requested)],
        _replication,
        cells_of=_cells_of,
        cell_of=_group_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        label=f"properties/{group.name}",
        draw=_draw,
        derived=DERIVED_CELLS,
        seed_of=_sample_seed,
    )
    return pd.DataFrame([row for rows in outcomes for row in rows])


# ------------------------------------------------------------------------------- the whole run


def _controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The derived controls: the shrunken and noise arms of each group's contrast row."""
    return calibration_controls(
        rows,
        STUDY,
        labels=CONTROL_LABELS,
        efficiency_bounds=control_bounds(),
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
        positive_suffix="corrected",
    )


def _assemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = pd.concat(frames, ignore_index=True)
    controls = _controls(rows)
    out = pd.concat([rows, controls], ignore_index=True)
    return out.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Every replication of every group, then the derived controls."""
    return _assemble([group_rows(group, n_jobs=n_jobs) for group in GROUPS.values()])


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 20) -> pd.DataFrame:
    """The first ``replicates`` replications of every group, for a disposable smoke run."""
    return _assemble(
        [
            group_rows(group, n_jobs=n_jobs, replicates=min(replicates, group.replicates))
            for group in GROUPS.values()
        ]
    )


# ------------------------------------------------------------------------------ the verdicts


def _generated_design_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The oracle and the estimated arm must both calibrate on each contrast.

    The oracle arm pins ``Qbar_0``, so the shipped curve has no generated-design term.  The
    estimated arm reads the stacked curve.  Both are cross-fitted.  The pair is the empirical
    test that the stacked curve restores the term the shipped curve omits.  The verdict is joint
    over the four cells, two contrasts of two arms, and the paired deficit of each contrast is
    reported.
    """
    generated = rows.loc[rows["property"] == "generated_design"]
    margins = STUDY.margins
    verdicts = {}
    for contrast in GENERATED_CONTRASTS:
        cells = {
            kind: f"{GENERATED_LABEL}__{kind}__{slug(contrast)}"
            for kind in ("oracle_design", "estimated")
        }
        arms = {
            kind: generated.loc[generated["cell"] == name].sort_values("replicate")
            for kind, name in cells.items()
        }
        deficit = se_ratio_deficit_interval(
            arms["estimated"],
            arms["oracle_design"],
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "generated_design", GENERATED_LABEL, slug(contrast), "deficit"),
        )
        for kind, arm in arms.items():
            interval = se_ratio_interval(
                arm,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "generated_design", *cells[kind].split("__")),
            )
            mask = (summary["property"] == "generated_design") & (summary["cell"] == cells[kind])
            row = summary.loc[mask].iloc[0]
            coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
            verdicts[cells[kind]] = bool(
                interval.within(*margins.calibration_se_ratio)
                and coverage.within(*margins.calibration_coverage)
            )
            summary.loc[mask, "se_ratio_ci_lower"] = interval.low
            summary.loc[mask, "se_ratio_ci_upper"] = interval.high
            summary.loc[mask, "se_ratio_deficit_lower"] = deficit.low
            summary.loc[mask, "se_ratio_deficit_upper"] = deficit.high
            summary.loc[mask, "passed"] = verdicts[cells[kind]]
    summary.loc[summary["property"] == "generated_design", "property_passed"] = all(
        verdicts.values()
    )


def _covariance_stats(values: np.ndarray) -> float:
    """``mean(reported) / cov(arm0, arm1)`` of rows ``(arm0, arm1, reported)``."""
    return float(values[:, 2].mean() / np.cov(values[:, 0], values[:, 1])[0, 1])


def _report_covariance(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The Monte Carlo covariance of the arm means against the mean reported covariance."""
    group = rows.loc[
        (rows["property"] == "interval_calibration") & (rows["cell"] == COVARIANCE_CELL)
    ]
    values = group[["arm0_estimate", "arm1_estimate", "reported_cov01"]].to_numpy(dtype=float)
    margins = STUDY.margins
    rng = np.random.default_rng(stream_seed(STUDY, "covariance", COVARIANCE_CELL))
    draws = []
    for block in bootstrap_draw_blocks(values, replicates=margins.bootstrap_replicates, rng=rng):
        centred = block[:, :, :2] - block[:, :, :2].mean(axis=1, keepdims=True)
        covariance = (centred[:, :, 0] * centred[:, :, 1]).sum(axis=1) / (block.shape[1] - 1)
        draws.append(block[:, :, 2].mean(axis=1) / covariance)
    tail = (1.0 - margins.confidence_level) / 2.0
    low, high = np.quantile(np.concatenate(draws), [tail, 1.0 - tail])
    mask = (summary["property"] == "interval_calibration") & (summary["cell"] == COVARIANCE_CELL)
    summary.loc[mask, "covariance_mc"] = float(np.cov(values[:, 0], values[:, 1])[0, 1])
    summary.loc[mask, "covariance_reported"] = float(values[:, 2].mean())
    summary.loc[mask, "covariance_ratio"] = _covariance_stats(values)
    summary.loc[mask, "covariance_ratio_ci_lower"] = float(low)
    summary.loc[mask, "covariance_ratio_ci_upper"] = float(high)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the generated-design pairs, the bands and the reported quantities."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=SUMMARY_COLUMNS, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix="corrected")
    _generated_design_verdicts(summary, rows)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    _report_covariance(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rule for the one verdict the module writes outside a shared
#: function.  The four generated-design cells share one verdict, so the group is the family.
PROPERTY_RULES = (
    property_verdicts.generated_design_rule(group="family", shared="_generated_design_verdicts"),
)

__all__ = [
    "BUDGETS",
    "DERIVED_CELLS",
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "Band",
    "generate_property_rows",
    "generate_smoke_property_rows",
    "group_rows",
    "shipped_banded",
    "shipped_std_error",
    "summarize_properties",
]
