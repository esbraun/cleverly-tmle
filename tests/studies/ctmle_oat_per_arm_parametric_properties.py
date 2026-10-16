"""Repeated-sampling properties of the stacked per-arm C-TMLE curve.

Every cell fits ``CTMLE(strategy="oat", oat_design="per_arm", cross_fit=False)`` with
:class:`~cleverly.learners.ParametricOutcome` and :class:`~cleverly.learners.ParametricMechanism`,
which the support predicate admits, so each estimate reports the corrected stacked curve.  The
laws, fits, cells and budgets are declared in
:mod:`tests.studies.canonical_ctmle_oat_per_arm_parametric`.

=========================  ===============================================================
family                     cells
=========================  ===============================================================
``interval_calibration``   ``<label>__<estimand>__corrected``: SE ratio and coverage of
                           ``ey0``, ``ey1`` and ``ate`` on ``binary_active`` at
                           ``n = 1,500`` and on the ladder ``n_500``, ``n_2000`` and
                           ``n_8000``; of ``ate``, ``log_rr`` and ``log_or`` on the
                           shared-prognosis law; of ``ey0``, ``ey1`` and ``ate`` on the
                           U-shaped law with the cubic and the linear basis (``kappa_cubic``,
                           ``kappa_linear``), the steep law with a binding bound
                           (``bounds_active``), ``g_bounds="auto"`` (``auto_bounds``), the
                           bounded outcome (``bounded_outcome``), and ``binary_active``
                           without ``W1`` in the outcome features, against the limit
                           ``psi*`` (``pseudo_true``)
``interval_calibration``   the controls.  ``binary_active__ate__shrunken_se_control`` and
                           ``binary_active__ate__noise_control`` are derived from the
                           primary ``ate`` rows.  ``binary_active__ate__shipped_curve__
                           shrunken_se_control`` reports the shipped adaptive-propensity
                           standard error of the same fits, and ``shared_prognosis__ate__
                           cross_term_dropped__inflated_se_control`` the ATE standard error
                           without the cross-arm covariance.  Each control passes by leaving
                           the calibration band on its declared side.
``generated_design``       ``binary_active__oracle_design`` (``Qbar_0`` pinned, shipped curve)
                           against ``binary_active__estimated`` (the stacked curve), one
                           sample per pair
``type_i_error``           ``sharp_null`` on the sharp-null twin, ``n = 1,000``
``power``                  ``alternative`` on ``binary_active`` at ``n = 800``
=========================  ===============================================================

The ladder and the joint cells sit in ``interval_calibration`` because the harness classifies
the families it knows, and the shared calibration verdict reads one estimand per cell.

**Reported, not gated.**  The summary carries four quantities that no verdict reads.

* ``psi0_coverage``: the coverage of ``psi_0`` by the pseudo-true cells, expected 0.30 to 0.56.
* ``sqrt_n_empirical_sd``, ``sqrt_n_reported_se`` and ``quadrature_sd`` on the ladder cells.
* ``covariance_mc``, ``covariance_reported`` and their ratio, with a resampling interval, on the
  ATE row of the shared-prognosis cell.  The ATE variance contains twice the covariance, so
  the gated row reads it.
* ``clipped_share`` on the steep cell.

A replicate that raises is not redrawn.  :func:`group_rows` fits through
:func:`~tests.studies.evidence.replication.fit_properties`, so one failed fit costs the cells
that its key writes one replicate each, and the run continues.  An admitted fit whose stack
status is not ``"influence_curve"`` raises in ``fit_cleverly`` and counts the same way.  :data:`FAILURE_BOUND` lets a cell lose ``floor(0.01 R)`` replicates,
and the accounting judges each clause by its worst case over the lost values.  A lost
``binary_active`` fit also costs the two controls derived from its ``ate`` row
(:data:`DERIVED_CELLS`).  Each finished replicate is checkpointed under the driver, so a killed
run resumes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators._oat_stack import fitted_stack, stack_report
from cleverly.exceptions import DataError
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ctmle_oat_per_arm_parametric import (
    BUDGETS,
    CALIBRATION_N,
    CELLS,
    CONTROL_LABELS,
    GROUPS,
    SHRUNKEN_SE_FACTOR,
    STUDY,
    Cell,
    Group,
    cells_of,
    curve_sd,
    fit_cleverly,
    sample,
    truth,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval, clopper_pearson
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
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

#: The share of a cell's requested replicates that the accounting lets a run lose, declared for
#: the study's next property run (RM39b2).  Up to ``floor(0.01 R)`` lost replicates are judged by
#: the worst case of each clause; one more is ``bound exceeded``.  The runner's
#: ``STUDY.failure_bound`` carries the same value.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The replicate columns a row carries beside the shared ones.
EXTRA_COLUMNS = (
    "psi0_covered",
    "arm0_estimate",
    "arm1_estimate",
    "reported_cov01",
    "clipped_share",
)
#: The summary columns no verdict reads.
SUMMARY_COLUMNS = (
    "se_ratio_deficit_lower",
    "se_ratio_deficit_upper",
    "psi0_coverage",
    "psi0_coverage_ci_lower",
    "psi0_coverage_ci_upper",
    "covariance_mc",
    "covariance_reported",
    "covariance_ratio",
    "covariance_ratio_ci_lower",
    "covariance_ratio_ci_upper",
    "sqrt_n_empirical_sd",
    "sqrt_n_reported_se",
    "quadrature_sd",
    "clipped_share",
)

GENERATED_LABEL = "binary_active"
COVARIANCE_CELL = "shared_prognosis__ate__corrected"
#: The estimate names of the log-scale ratio tokens.
RATIO_ESTIMATES = {"log_rr": "rr", "log_or": "or"}


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


def shipped_ate_std_error(stack: Any, report: Any) -> float:
    """The standard error of the shipped adaptive-propensity curve of the ATE.

    The shipped curve of an arm is the corrected curve plus the correction the estimate builder
    subtracts, both on the fitted scale.
    """
    if report.curves is None or report.failures:
        raise DataError("the stacked curve failed, so the shipped curve has no comparison")
    shipped = {
        arm: report.curves.curves[:, j] + report.corrections[arm]
        for j, arm in enumerate(sorted(report.corrections))
    }
    ate = (shipped[1.0] - shipped[0.0]) * stack.scale
    return float(np.std(ate, ddof=1) / np.sqrt(len(ate)))


def _estimate(result: Any, token: str) -> tuple[float, float]:
    """``(value, standard error)`` of ``token``; a ratio on the log scale."""
    estimate = result[RATIO_ESTIMATES.get(token, token)]
    value = float(estimate.log_psi) if token in RATIO_ESTIMATES else float(estimate.psi)
    return value, float(estimate.plugin_std_error)


def _measure_cell(cell: Cell, result: Any, parts: dict[str, Any]) -> dict[str, Any]:
    """What one cell reads off its fit: no truth, which the driver adds."""
    value, se = _estimate(result, cell.estimand)
    if cell.recipe == "shipped_curve":
        se = shipped_ate_std_error(*parts["stack"])
    elif cell.recipe == "cross_term_dropped":
        se = float(np.hypot(result["ey0"].plugin_std_error, result["ey1"].plugin_std_error))
    elif cell.recipe != "corrected":
        raise KeyError(cell.recipe)
    measured: dict[str, Any] = {"cell": cell.name, "estimate": value, "std_error": se}
    if cell.report == "covariance":
        curves = result.influence_curves
        measured.update(
            arm0_estimate=float(result["ey0"].psi),
            arm1_estimate=float(result["ey1"].psi),
            reported_cov01=float(np.cov(curves["ey0"], curves["ey1"])[0, 1] / result.n),
        )
    if cell.report == "clipped":
        _, report = parts["stack"]
        measured["clipped_share"] = float(np.mean(report.curves.clipped_share["g_bounds"]))
    return measured


def _measure(key: Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Fit every configuration of the key's group on one sample and read every cell."""
    group = GROUPS[key[0]]
    results = {fit.name: fit_cleverly(frame, fit) for fit in group.fits}
    stacks: dict[str, tuple[Any, Any]] = {}
    out = []
    for cell in cells_of(group.name):
        result = results[cell.fit]
        parts: dict[str, Any] = {}
        if cell.recipe == "shipped_curve" or cell.report == "clipped":
            if cell.fit not in stacks:
                stack = fitted_stack(result)
                arms = tuple(float(arm) for arm in result.data.arm_codes)
                stacks[cell.fit] = (stack, stack_report(stack, arms))
            parts["stack"] = stacks[cell.fit]
        out.append(_measure_cell(cell, result, parts))
    return out


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
        if cell.report == "psi0":
            alternative = truth("binary_active", cell.estimand)
            row["psi0_covered"] = int(
                abs(item["estimate"] - alternative) <= CRITICAL * item["std_error"]
            )
        rows.append(row)
    return rows


#: The derived controls of each base cell, as ``property/cell`` names: a lost ``binary_active``
#: fit also costs the ``shrunken_se_control`` and the ``noise_control`` of its ``ate`` row.
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
    return _rows(GROUPS[key[0]], key[1], _measure(key, frame), key[2])


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
        truth(cell.truth, cell.estimand)
        if cell.report == "psi0":
            truth("binary_active", cell.estimand)
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
    """The derived controls: the shrunken and noise arms of the primary ``ate`` row."""
    bounds = curve_sd()
    return calibration_controls(
        rows,
        STUDY,
        labels=CONTROL_LABELS,
        efficiency_bounds={label: bounds[label.split("__", 1)[1]] for label in CONTROL_LABELS},
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
    """The oracle and the estimated arm must both calibrate; the paired deficit is reported.

    The oracle arm pins ``Qbar_0``, so the shipped curve has no generated-design term.  The
    estimated arm reads the stacked curve.  The pair is the empirical test that the stacked
    curve restores the term the shipped curve omits.
    """
    generated = rows.loc[rows["property"] == "generated_design"]
    margins = STUDY.margins
    arms = {
        kind: generated.loc[generated["cell"] == f"{GENERATED_LABEL}__{kind}"].sort_values(
            "replicate"
        )
        for kind in ("oracle_design", "estimated")
    }
    deficit = se_ratio_deficit_interval(
        arms["estimated"],
        arms["oracle_design"],
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", GENERATED_LABEL, "deficit"),
    )
    verdicts = {}
    for kind, arm in arms.items():
        interval = se_ratio_interval(
            arm,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "generated_design", GENERATED_LABEL, kind),
        )
        mask = (summary["property"] == "generated_design") & (
            summary["cell"] == f"{GENERATED_LABEL}__{kind}"
        )
        row = summary.loc[mask].iloc[0]
        coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
        verdicts[kind] = bool(
            interval.within(*margins.calibration_se_ratio)
            and coverage.within(*margins.calibration_coverage)
        )
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "se_ratio_deficit_lower"] = deficit.low
        summary.loc[mask, "se_ratio_deficit_upper"] = deficit.high
        summary.loc[mask, "passed"] = verdicts[kind]
    summary.loc[summary["property"] == "generated_design", "property_passed"] = all(
        verdicts.values()
    )


def _report_psi0(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The coverage of ``psi_0`` by the pseudo-true cells, which target ``psi*``."""
    margins = STUDY.margins
    for cell in (c for c in CELLS if c.report == "psi0"):
        group = rows.loc[(rows["property"] == cell.family) & (rows["cell"] == cell.name)]
        covered = group["psi0_covered"].to_numpy(dtype=float)
        interval = clopper_pearson(
            int(covered.sum()), len(covered), confidence_level=margins.confidence_level
        )
        mask = (summary["property"] == cell.family) & (summary["cell"] == cell.name)
        summary.loc[mask, "psi0_coverage"] = float(covered.mean())
        summary.loc[mask, "psi0_coverage_ci_lower"] = interval.low
        summary.loc[mask, "psi0_coverage_ci_upper"] = interval.high


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


def _report_root_n(summary: pd.DataFrame) -> None:
    """``sqrt(n)`` times the empirical SD and the mean reported SE, beside the quadrature SD."""
    bounds = curve_sd()
    ladder = (summary["property"] == "interval_calibration") & summary["cell"].str.startswith("n_")
    for index in summary.index[ladder]:
        estimand = str(summary.loc[index, "cell"]).split("__")[1]
        scale = float(np.sqrt(summary.loc[index, "n"]))
        summary.loc[index, "sqrt_n_empirical_sd"] = scale * summary.loc[index, "empirical_se"]
        summary.loc[index, "sqrt_n_reported_se"] = scale * summary.loc[index, "mean_std_error"]
        summary.loc[index, "quadrature_sd"] = bounds[estimand]


def _report_clipped(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The mean share of rows at a declared ``g_bounds`` limit."""
    for cell in (c for c in CELLS if c.report == "clipped"):
        group = rows.loc[(rows["property"] == cell.family) & (rows["cell"] == cell.name)]
        mask = (summary["property"] == cell.family) & (summary["cell"] == cell.name)
        summary.loc[mask, "clipped_share"] = float(group["clipped_share"].mean())


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the generated-design pair, and the reported quantities."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=SUMMARY_COLUMNS, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix="corrected")
    _generated_design_verdicts(summary, rows)
    _report_psi0(summary, rows)
    _report_covariance(summary, rows)
    _report_root_n(summary)
    _report_clipped(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rule for the one verdict the module writes outside a shared
#: function.  The oracle-design and estimated cells share one verdict, so the group is the family.
PROPERTY_RULES = (
    property_verdicts.generated_design_rule(group="family", shared="_generated_design_verdicts"),
)


__all__ = [
    "BUDGETS",
    "DERIVED_CELLS",
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "generate_smoke_property_rows",
    "group_rows",
    "shipped_ate_std_error",
    "summarize_properties",
]
