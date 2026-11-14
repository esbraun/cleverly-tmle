r"""The ``fold_variance_calibration`` family of ``few-cluster-fold-variance``.

Each key draws one sample of a cell of :data:`tests.studies.few_cluster_fold_variance.CELLS`,
fits it twice, and writes seven rows, one per arm
(:func:`tests.studies.few_cluster_fold_variance.draw_arms`). Beside the shared replicate
columns a row records ``withheld`` (the estimate withheld its interval, so every arm reads a
miss in that arm) and ``reference_df``. A key whose fit raises is lost in all seven cells.

The statistic of the family is the **variance ratio** over the shown fits of a cell,

.. math::

    \widehat{\mathrm{VR}} = \overline{\hat\sigma^2}\,/\,\widehat{\mathrm{Var}}(\hat\psi),

the mean squared SE over the ``ddof=1`` variance of the estimates, with a 99% percentile
bootstrap interval that resamples replicates jointly (:func:`variance_ratio_interval`). The
gated rules (plan X54 rev 3.1, sections 2.4 and 5.3):

=========================================  ==========================================================
rule                                       rows and threshold
=========================================  ==========================================================
``variance ratio in band``                 every ``t_reference`` row: the 99% interval inside
                                           :math:`[0.93^2, 1.07^2] = [0.8649, 1.1449]`
``shown coverage in band``                 every ``t_reference`` row: the exact 99% interval of
                                           coverage over the shown fits inside
                                           ``margins.calibration_coverage`` (0.92, 0.98)
``paired iid/cluster se ratio under its    every ``iid_t_control`` row: the 99% upper end of the mean
ceiling``                                  IID SE over the mean cluster SE of the same fits below
                                           0.80 (clause 1 of the IID control)
``deflated variance ratio below band``     every ``deflated_control`` row: the 99% upper end of its
                                           variance ratio below 0.8649
``pooled-centred variance ratio above      the ``pooled_centered`` rows of the cells in
band``                                     :data:`~tests.studies.few_cluster_fold_variance.POOLED_CENTERED_GATED`:
                                           the 99% lower end of its variance ratio above 1.1449
=========================================  ==========================================================

Every other row is ``diagnostic`` and states no verdict. Every row publishes the shown and
withheld counts, the shown-fit coverage with its exact interval, the variance ratio with its
interval, the shown-fit SE ratio with its interval, the reference degrees of freedom
:math:`k = J - V` and :math:`c_k = E\sqrt{\chi^2_k/k}`, the SE ratio of an exactly calibrated
variance under normality, and the over-coverage reading (the shown lower end above 0.99).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_fold_variance as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval, clopper_pearson, percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    bootstrap_draw_blocks,
    paired_se_ratio_interval,
    se_ratio_interval,
)
from tests.studies.evidence.property_failures import (
    SUPPORT,
    Clause,
    Context,
    FamilyRule,
    Judged,
    above_judge,
    band_judge,
    below_judge,
    cells,
    paired_se_ratio_clause,
    se_ratio_sets,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY

#: The variance-ratio band: the calibration SE-ratio band squared, :math:`[0.8649, 1.1449]`.
VARIANCE_BAND: Final[tuple[float, float]] = (
    STUDY.margins.calibration_se_ratio[0] ** 2,
    STUDY.margins.calibration_se_ratio[1] ** 2,
)
#: The two-sided band of the shown-fit coverage interval.
COVERAGE_BAND: Final[tuple[float, float]] = STUDY.margins.calibration_coverage
#: The ceiling of the IID control's paired SE ratio (clause 1 of the IID control).
IID_CONTROL_CEILING: Final[float] = property_verdicts.FEW_CLUSTER_IID_CONTROL_CEILING
#: The over-coverage reading's threshold.
OVER_COVERAGE: Final[float] = STUDY.margins.over_coverage_ceiling

#: The columns a row carries beside :data:`REPLICATE_COLUMNS`.
EXTRA_COLUMNS: Final[tuple[str, ...]] = ("withheld", "reference_df")

#: The readings :func:`summarize_properties` adds to every row.
READING_COLUMNS: Final[tuple[str, ...]] = (
    "shown_draws",
    "withheld_draws",
    "shown_coverage",
    "shown_ci_lower",
    "shown_ci_upper",
    "shown_above_099",
    "variance_ratio",
    "variance_ratio_ci_lower",
    "variance_ratio_ci_upper",
    "shown_se_ratio",
    "shown_se_ratio_ci_lower",
    "shown_se_ratio_ci_upper",
    "reference_k",
    "jensen_factor",
    "paired_se_ratio",
    "paired_se_ratio_ci_lower",
    "paired_se_ratio_ci_upper",
)


def _cell(cell: str, arm: str) -> str:
    return study.cell_name(cell, arm)


POSITIVE_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(cell.name, "t_reference") for cell in study.CELLS
)
IID_CELLS: Final[tuple[str, ...]] = tuple(_cell(cell.name, "iid_t_control") for cell in study.CELLS)
DEFLATED_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(cell.name, "deflated_control") for cell in study.CELLS
)
POOLED_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(name, "pooled_centered") for name in sorted(study.POOLED_CENTERED_GATED)
)

#: The positive and IID-control arms of each cell, the pairs clause 1 of the IID control reads.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    FAMILY: tuple(
        (_cell(cell.name, "t_reference"), _cell(cell.name, "iid_t_control")) for cell in study.CELLS
    )
}


def _split(name: str) -> tuple[str, str]:
    """``(cell prefix, arm)`` of a published cell name."""
    prefix, arm = name.rsplit("__", 1)
    return prefix, arm


def _role(name: str) -> str:
    return study.arm_role(*_split(name))


# ------------------------------------------------------------------------------- statistics


def variance_ratio(estimates: np.ndarray, std_errors: np.ndarray) -> float:
    """``mean(se**2) / var(estimate, ddof=1)``; ``nan`` below two draws or at zero spread."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size < 2:
        return float("nan")
    spread = float(np.var(estimates, ddof=1))
    if spread <= 0.0:
        return float("nan")
    return float(np.mean(np.square(np.asarray(std_errors, dtype=float))) / spread)


def variance_ratio_interval(
    estimates: np.ndarray,
    std_errors: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> Interval:
    """The percentile bootstrap interval of :func:`variance_ratio`.

    Each bootstrap draw resamples whole replicates, so a draw keeps each fit's estimate with
    its own SE, and the numerator and the denominator read the same resampled replicates.
    """
    values = np.column_stack(
        [np.asarray(estimates, dtype=float), np.asarray(std_errors, dtype=float)]
    )
    if len(values) < 2:
        return Interval(float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ratios = np.concatenate(
        [
            np.mean(np.square(block[:, :, 1]), axis=1) / np.var(block[:, :, 0], axis=1, ddof=1)
            for block in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
        ]
    )
    return percentile_interval(ratios, confidence_level=confidence_level)


def variance_seed(record: Any, cell: str) -> int:
    """The verdict's own resampling seed of one cell's variance ratio."""
    return int(stream_seed(record, FAMILY, cell, "variance_ratio"))


# ------------------------------------------------------------------------------------- rows


def _seed(cell: str, replicate: int) -> int:
    return int(stream_seed(STUDY, "property_sample", FAMILY, cell, replicate))


def _draw(key: tuple[str, int, int]) -> pd.DataFrame:
    name, replicate, _ = key
    cell = study.CELL_BY_NAME[name]
    return study.draw(cell.law, cell.clusters, np.random.default_rng(_seed(name, replicate)))


def requested(cell: str, budget: int | None = None) -> int:
    """A cell's requested replications, capped at ``budget`` for a pre-run check."""
    declared = study.PROPERTY_REPLICATES[cell]
    return declared if budget is None else min(budget, declared)


def key_rows(
    name: str, replicate: int, entry: Mapping[str, Mapping[str, Any]], truth: float, wanted: int
) -> list[dict[str, Any]]:
    """The seven rows of one key, off :func:`~tests.studies.few_cluster_fold_variance.draw_arms`."""
    cell = study.CELL_BY_NAME[name]
    rows = []
    for arm in study.ARMS:
        published = _cell(name, arm)
        reading = entry[arm]
        rows.append(
            {
                "property": FAMILY,
                "cell": published,
                "role": _role(published),
                "replicate": replicate,
                "n": cell.n,
                "requested_replicates": wanted,
                "failed_replicates": 0,
                "truth": float(truth),
                "estimate": float(reading["psi"]),
                "std_error": float(reading["std_error"]),
                "covered": int(reading["covered"]),
                "rejected": int(reading["rejected"]),
                "withheld": int(bool(reading["withheld"])),
                "reference_df": int(reading["reference_df"]),
            }
        )
    return rows


def _fit_key(key: tuple[str, int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    name, replicate, wanted = key
    cell = study.CELL_BY_NAME[name]
    truth = study.law_truth(cell.law)[study.TARGETS[cell.law]]
    return key_rows(name, replicate, study.draw_arms(frame, cell, truth), truth, wanted)


def _cells_of(key: tuple[str, int, int]) -> list[str]:
    """Every arm of the key's cell reads its one fit."""
    return [f"{FAMILY}/{_cell(key[0], arm)}" for arm in study.ARMS]


def _cell_of(key: tuple[str, int, int]) -> str:
    return f"{FAMILY}/{_cell(key[0], 't_reference')}"


def _replicate_of(key: tuple[str, int, int]) -> int:
    return key[1]


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its role, size and requested replicates."""
    out = []
    for cell in study.CELLS:
        for arm in study.ARMS:
            name = _cell(cell.name, arm)
            out.append(
                PropertyCell(
                    property=FAMILY,
                    cell=name,
                    dgp=cell,
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=cell.n,
                    replicates=study.PROPERTY_REPLICATES[cell.name],
                    seed=_seed(cell.name, 0),
                    role=_role(name),
                    estimand=study.TARGETS[cell.law],
                )
            )
    return tuple(out)


def synthetic_columns(cell: Any, rng: np.random.Generator) -> dict[str, Any]:
    """The extra columns of a synthetic row for the property differential."""
    del rng
    return {"withheld": 0, "reference_df": int(cell.dgp.df)}


#: The share of a cell's requested replications that the accounting lets a run lose. The plan
#: asked for 0.10; the harness caps a bound at 0.05
#: (:data:`~tests.studies.evidence.property_failures.MAXIMUM_BOUND`). The pilot lost no fit.
FAILURE_BOUND = 0.05

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pre-run check."""
    keys = [
        (cell.name, replicate, requested(cell.name, budget))
        for cell in study.CELLS
        for replicate in range(requested(cell.name, budget))
    ]
    outcomes = fit_properties(
        keys,
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *EXTRA_COLUMNS]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------------- readings


def _shown(group: pd.DataFrame) -> pd.DataFrame:
    return group.loc[group["withheld"].to_numpy(dtype=float) < 0.5]


def fold_variance_readings(summary: pd.DataFrame, rows: pd.DataFrame, record: Any) -> None:
    """Fill :data:`READING_COLUMNS` on every row of the family."""
    margins = record.margins
    level = margins.confidence_level
    family = rows.loc[rows["property"] == FAMILY]
    groups = dict(tuple(family.groupby("cell", sort=False)))
    for index in summary.index[summary["property"] == FAMILY]:
        name = str(summary.loc[index, "cell"])
        prefix, _ = _split(name)
        cell = study.CELL_BY_NAME[prefix]
        group = groups[name]
        shown = _shown(group)
        hits, trials = int(shown["covered"].sum()), len(shown)
        values: dict[str, float] = {
            "shown_draws": float(trials),
            "withheld_draws": float(len(group) - trials),
            "reference_k": float(cell.df),
            "jensen_factor": study.jensen_factor(cell.df),
        }
        if trials:
            exact = clopper_pearson(hits, trials, confidence_level=level)
            values.update(
                shown_coverage=hits / trials,
                shown_ci_lower=exact.low,
                shown_ci_upper=exact.high,
                shown_above_099=float(exact.low > OVER_COVERAGE),
            )
        estimates = shown["estimate"].to_numpy(dtype=float)
        std_errors = shown["std_error"].to_numpy(dtype=float)
        values["variance_ratio"] = variance_ratio(estimates, std_errors)
        ratio = variance_ratio_interval(
            estimates,
            std_errors,
            replicates=margins.bootstrap_replicates,
            confidence_level=level,
            seed=variance_seed(record, name),
        )
        values["variance_ratio_ci_lower"], values["variance_ratio_ci_upper"] = ratio.low, ratio.high
        if trials >= 2:
            spread = float(np.std(estimates, ddof=1))
            values["shown_se_ratio"] = float(np.mean(std_errors) / spread) if spread > 0 else np.nan
            se_ratio = se_ratio_interval(
                shown,
                replicates=margins.bootstrap_replicates,
                confidence_level=level,
                seed=stream_seed(record, FAMILY, name, "se_ratio"),
                truth_varies=False,
            )
            values["shown_se_ratio_ci_lower"] = se_ratio.low
            values["shown_se_ratio_ci_upper"] = se_ratio.high
        if name in IID_CELLS:
            paired = paired_se_ratio_interval(
                group,
                groups[_cell(prefix, "t_reference")],
                replicates=margins.bootstrap_replicates,
                confidence_level=level,
                seed=stream_seed(record, FAMILY, name),
            )
            values["paired_se_ratio"] = paired.ratio
            values["paired_se_ratio_ci_lower"] = paired.interval.low
            values["paired_se_ratio_ci_upper"] = paired.interval.high
        for column, value in values.items():
            summary.loc[index, column] = value


def fold_variance_verdicts(summary: pd.DataFrame) -> None:
    """The declared verdicts of the family, read off its summary columns."""
    low, high = VARIANCE_BAND
    family = summary["property"] == FAMILY
    for index in summary.index[family]:
        row = summary.loc[index]
        name = str(row["cell"])
        if name in POSITIVE_CELLS:
            passed = bool(
                low <= row["variance_ratio_ci_lower"]
                and row["variance_ratio_ci_upper"] <= high
                and COVERAGE_BAND[0] <= row["shown_ci_lower"]
                and row["shown_ci_upper"] <= COVERAGE_BAND[1]
            )
        elif name in IID_CELLS:
            passed = bool(row["paired_se_ratio_ci_upper"] < IID_CONTROL_CEILING)
        elif name in DEFLATED_CELLS:
            passed = bool(row["variance_ratio_ci_upper"] < low)
        elif name in POOLED_CELLS:
            passed = bool(row["variance_ratio_ci_lower"] > high)
        else:
            passed = True
        summary.loc[index, "passed"] = passed
        summary.loc[index, "property_passed"] = passed


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the family's readings and its declared verdicts."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=READING_COLUMNS, rate_labels=()
    )
    fold_variance_readings(summary, rows, STUDY)
    fold_variance_verdicts(summary)
    return finish(summary, rates)


# ------------------------------------------------------------------------------ declared rules


def _variance_clause(name: str, judge: Any, applies: Any) -> Clause:
    """A variance-ratio interval clause over the shown fits, with the S5 candidate set.

    A lost draw is either withheld, so it leaves the shown fits unchanged, or shown at an
    extreme of the survivors: the SE at the survivor maximum with the estimates at their mean,
    and the SE at the survivor minimum with the estimates at each vertex split
    (:func:`~tests.studies.evidence.property_failures.se_ratio_sets`). The mean squared SE is
    monotone in its fills and the variance convex in its own, so the set holds both extremes of
    the point ratio, as it does for the SE ratio.
    """

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(float(row["variance_ratio_ci_lower"]), float(row["variance_ratio_ci_upper"]))
        return judge(ctx, row, iv)

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        shown = _shown(ctx.cell(FAMILY, str(row["cell"])))
        margins = ctx.margins

        def interval(frame: pd.DataFrame) -> Interval:
            return variance_ratio_interval(
                frame["estimate"].to_numpy(dtype=float),
                frame["std_error"].to_numpy(dtype=float),
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=variance_seed(ctx.record, str(row["cell"])),
            )

        out = [("every lost draw withheld", judge(ctx, row, interval(shown)))]
        for label, frame in se_ratio_sets(shown, ctx.k(row), truth_varies=False):
            out.append((label, judge(ctx, row, interval(frame))))
        return out

    return Clause(name, "S5", applies, read, candidates)


def _coverage_read(ctx: Context, row: pd.Series) -> Judged:
    iv = Interval(float(row["shown_ci_lower"]), float(row["shown_ci_upper"]))
    return band_judge(lambda _: COVERAGE_BAND)(ctx, row, iv)


def _coverage_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """A lost draw is withheld or shown; a shown one is covered or not: every split."""
    trials = int(row["shown_draws"])
    hits = round(float(row["shown_coverage"]) * trials) if trials else 0
    level = ctx.margins.confidence_level
    judge = band_judge(lambda _: COVERAGE_BAND)
    out = []
    k = ctx.k(row)
    for shown in range(k + 1):
        for covered in range(shown + 1):
            if trials + shown == 0:
                continue
            iv = clopper_pearson(hits + covered, trials + shown, confidence_level=level)
            out.append((f"{shown} shown, {covered} covered", judge(ctx, row, iv)))
    return out


#: The failure accounting's declared rules: the verdicts of :func:`fold_variance_verdicts`.
PROPERTY_RULES = (
    FamilyRule(
        families=frozenset({FAMILY}),
        shared="summarize_properties",
        clauses=(
            _variance_clause(
                "variance ratio in band",
                band_judge(lambda _: VARIANCE_BAND),
                cells(*POSITIVE_CELLS),
            ),
            Clause(
                "shown coverage in band",
                "S2",
                cells(*POSITIVE_CELLS),
                _coverage_read,
                _coverage_candidates,
                imputation=SUPPORT,
            ),
            paired_se_ratio_clause(
                "paired iid/cluster se ratio under its ceiling",
                judge=below_judge(lambda _: IID_CONTROL_CEILING),
                seed=lambda ctx, row: int(stream_seed(ctx.record, FAMILY, str(row["cell"]))),
                applies=cells(*IID_CELLS),
            ),
            _variance_clause(
                "deflated variance ratio below band",
                below_judge(lambda _: VARIANCE_BAND[0]),
                cells(*DEFLATED_CELLS),
            ),
            _variance_clause(
                "pooled-centred variance ratio above band",
                above_judge(lambda _: VARIANCE_BAND[1]),
                cells(*POOLED_CELLS),
            ),
        ),
    ),
)


__all__: Sequence[str] = (
    "PROPERTY_RULES",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
    "variance_ratio",
    "variance_ratio_interval",
)
