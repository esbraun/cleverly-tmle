r"""The ``adaptive_learners`` family of ``few-cluster-adaptive-learners``: rows, readings, rules.

Each key fits one fit of one draw (:data:`tests.studies.few_cluster_adaptive_learners.POINT_FITS`
and ``LTMLE_FITS``), so a fit that raises costs that fit's cells one replicate and no other
fit's. The fits of one draw share its sample, drawn outside the fit from
``stream_seed(record, "property_sample", "<law>/J<count>", replicate)``. Beside the shared
replicate columns a row records ``withheld`` (the estimate withheld its interval; every arm of
the fit then reads a miss) and ``sl_single`` (every outcome Super Learner of the fit put at
least 0.999 of its weight on one candidate).

The declared rules (plan X54 rev 3.2, sections 3.6 and 5.4, review findings B3 and N3):

=================================  ==========================================================
rule                               rows and threshold
=================================  ==========================================================
``coverage floor`` (G1)            every positive: the exact 99% lower end of coverage at
                                   least 0.90, a withheld fit a miss and a lost fit a miss in
                                   the worst case of the failure accounting
``bias equivalent`` (G2)           every positive: the 99% bias interval inside
                                   :math:`\pm 0.25` empirical SD
``variance ratio lower end``       every positive: the 99% percentile-bootstrap lower end of
                                   :math:`\overline{\hat\sigma^2}/\widehat{\mathrm{Var}}(\hat\psi)`
                                   over shown fits at least :math:`0.8649 = 0.93^2`
``shown coverage lower end``       every positive: the exact 99% lower end of shown-only
                                   coverage at least 0.92
``paired non-inferiority``         every L1 positive: the exact 99% lower end of the paired
                                   all-draw coverage difference to its ``glm_reference`` arm
                                   above :math:`-0.02`
``paired iid/cluster se ratio``    every ``iid_t_control``: the two-clause rule of the
and ``paired positive se ratio``   registered few-cluster studies
``must fail``                      ``interpolating_control`` and ``stacked_control``: the exact
                                   99% upper end of all-draw coverage below 0.90
=================================  ==========================================================

A positive row's ``passed`` is the conjunction of its five clauses (four off L1). The caveat
of the docs goes for a configuration only where its positive row passes, its IID control
passes and its must-fail control (:data:`~tests.studies.few_cluster_adaptive_learners.MUST_FAIL_OF`)
passes: the reading ``caveat_removed``. Every positive row also publishes the shown-only
coverage, the withheld share, the variance ratio and its interval, the median and largest SE
in empirical-SD units, the Super Learner single-candidate share and the over-coverage reading.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cleverly.inference import influence_variance
from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_adaptive_learners as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson, percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    bootstrap_draw_blocks,
    control_row,
    paired_se_ratio_interval,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_failures import (
    RANGE,
    SUPPORT,
    Clause,
    Context,
    FamilyRule,
    Judged,
    below_judge,
    bias_clause,
    ceiling_judge,
    floor_judge,
    paired_positive_se_clause,
    paired_se_ratio_clause,
    proportion_clause,
    roles,
)
from tests.studies.evidence.property_verdicts import (
    FEW_CLUSTER_COLUMNS,
    FEW_CLUSTER_IID_CONTROL_CEILING,
    apply_shared_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY
PROPERTY_REPLICATES: Final[int] = study.PROPERTY_REPLICATES

#: The variance-ratio floor: an SE ratio of 0.93, squared (plan 5.4, B3).
VARIANCE_RATIO_FLOOR: Final[float] = 0.93**2
#: The shown-only coverage floor, the lower leg of the calibration band (plan 5.4, N3).
SHOWN_COVERAGE_FLOOR: Final[float] = 0.92
#: The paired non-inferiority margin on all-draw coverage against the GLM baseline (B3).
NONINFERIORITY_MARGIN: Final[float] = -0.02
#: The upper end below which a must-fail control's all-draw coverage must lie.
MUST_FAIL_CEILING: Final[float] = 0.90
#: The over-coverage reading's threshold.
OVER_COVERAGE: Final[float] = 0.99
#: The weight share at which a Super Learner counts as putting all its weight on one candidate.
SINGLE_CANDIDATE: Final[float] = 0.999

#: The columns a row carries beside :data:`REPLICATE_COLUMNS`.
EXTRA_COLUMNS: Final[tuple[str, ...]] = ("withheld", "sl_single")

#: The readings :func:`adaptive_readings` adds to the summary.
READING_COLUMNS: Final[tuple[str, ...]] = (
    *FEW_CLUSTER_COLUMNS,
    "clusters",
    "withheld_draws",
    "shown_draws",
    "shown_coverage",
    "shown_ci_lower",
    "shown_ci_upper",
    "variance_ratio",
    "variance_ratio_ci_lower",
    "variance_ratio_ci_upper",
    "median_se_over_sd",
    "max_se_over_sd",
    "sl_single_share",
    "noninferiority_difference",
    "noninferiority_lower",
    "g1_met",
    "variance_ratio_met",
    "shown_coverage_met",
    "noninferiority_met",
    "over_coverage",
    "caveat_removed",
)


def parse(cell: str) -> tuple[str, str, int, str]:
    """``(fit, law, clusters, arm)`` of a published cell name."""
    fit, law, count, arm = cell.split("__")
    return fit, law, int(count.removeprefix("j")), arm


def _role(fit: str, arm: str) -> str:
    return dict(study.ARMS[fit])[arm]


def noninferiority_gated(cell: str) -> bool:
    """Whether a positive cell carries the paired non-inferiority clause: L1 only."""
    fit, law, _, arm = parse(cell)
    return law == "equal10" and fit in study.ADAPTIVE_FITS and arm == "t_reference"


def glm_cell(cell: str) -> str:
    """The ``glm_reference`` cell paired with a default-library cell."""
    fit, law, clusters, _ = parse(cell)
    return study.cell_name(study.GLM_OF[fit], law, clusters, "glm_reference")


def must_fail_cell(cell: str) -> str:
    """The must-fail control the caveat of a positive cell reads."""
    fit, law, clusters, _ = parse(cell)
    control = study.MUST_FAIL_OF[fit]
    if fit == "ltmle_crossfit":
        law, clusters = "equal10", 20
    return study.cell_name(control, law, clusters, control)


def iid_cell(cell: str) -> str:
    """The IID control of a default-library cell."""
    fit, law, clusters, _ = parse(cell)
    return study.cell_name(fit, law, clusters, "iid_t_control")


# ------------------------------------------------------------------------------- statistics


def variance_ratio_interval(
    estimates: np.ndarray,
    std_errors: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    r""":math:`\overline{\hat\sigma^2}/\widehat{\mathrm{Var}}(\hat\psi)` and its percentile interval.

    The bootstrap resamples replicates jointly, each draw carrying its estimate and its standard
    error, so the interval reads the dependence between them (plan 4.3).
    """
    estimates = np.asarray(estimates, dtype=float)
    std_errors = np.asarray(std_errors, dtype=float)
    if estimates.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    point = float(np.mean(std_errors**2) / np.var(estimates, ddof=1))
    values = np.column_stack([estimates, std_errors**2])
    rng = np.random.default_rng(seed)
    ratios = []
    for block in bootstrap_draw_blocks(values, replicates=replicates, rng=rng):
        ratios.append(block[:, :, 1].mean(axis=1) / block[:, :, 0].var(axis=1, ddof=1))
    interval = percentile_interval(np.concatenate(ratios), confidence_level=confidence_level)
    return point, interval.low, interval.high


def mcnemar_lower(gained: int, lost: int, trials: int, level: float = 0.99) -> float:
    """The lower end of a paired coverage difference, exact on the discordant draws.

    ``gained`` draws cover in the first arm only and ``lost`` in the second only. The share
    :math:`\\theta` of ``gained`` among the discordant draws has a Clopper-Pearson interval, and
    the difference is :math:`(2\\theta - 1)(\\text{gained} + \\text{lost}) / \\text{trials}`.
    """
    discordant = gained + lost
    if trials == 0:
        return float("nan")
    if discordant == 0:
        return 0.0
    low = clopper_pearson(gained, discordant, confidence_level=level).low
    return (2.0 * low - 1.0) * discordant / trials


def paired_counts(positive: pd.DataFrame, glm: pd.DataFrame) -> dict[str, int]:
    """Concordance of all-draw coverage on the replicates both arms kept."""
    joined = positive[["replicate", "covered"]].merge(
        glm[["replicate", "covered"]], on="replicate", suffixes=("", "_glm")
    )
    mine = joined["covered"].to_numpy() > 0
    theirs = joined["covered_glm"].to_numpy() > 0
    return {
        "trials": len(joined),
        "gained": int(np.sum(mine & ~theirs)),
        "lost": int(np.sum(~mine & theirs)),
    }


def lost_counts(positive: pd.DataFrame, glm: pd.DataFrame, requested: int) -> dict[str, int]:
    """:func:`paired_counts` over every requested replicate, at the worst fill of a lost fit.

    A lost default-library fit reads a miss and a lost GLM fit a cover (the S7 candidate), so
    every lost fit can only move the difference against the default library.
    """
    mine = dict(zip(positive["replicate"], positive["covered"] > 0, strict=True))
    theirs = dict(zip(glm["replicate"], glm["covered"] > 0, strict=True))
    gained = lost = 0
    for replicate in range(requested):
        a = bool(mine.get(replicate, False))
        b = bool(theirs.get(replicate, True))
        gained += int(a and not b)
        lost += int(b and not a)
    return {"trials": requested, "gained": gained, "lost": lost}


# ------------------------------------------------------------------------------------- rows


def _seed(law: str, clusters: int, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", f"{law}/J{clusters}", replicate)


def t_critical(df: int | None) -> float:
    """The two-sided t quantile, or the normal one for ``None``."""
    from scipy.stats import norm

    level = 1.0 - STUDY.margins.alpha / 2.0
    return float(norm.ppf(level) if df is None else student_t.ppf(level, df))


#: The ``p`` of candidate (B): ``max(2, d)`` with ``d = 2`` baseline covariates in every fit.
WANG_P: Final[int] = 2


def fit_rows(
    fit: str,
    law: str,
    clusters: int,
    replicate: int,
    fitted: Any,
    requested: int,
) -> list[dict[str, Any]]:
    """The rows of one fit of one draw, one per arm."""
    estimate = fitted.estimate
    truth = study.law_truth(law)
    n = study.CLUSTER_SIZE[law] * clusters
    shown = bool(estimate.supplies_inference)
    common: dict[str, Any] = {
        "property_name": FAMILY,
        "replicate": replicate,
        "requested": requested,
        "n": n,
        "truth": truth,
    }
    extra = {"withheld": int(not shown), "sl_single": float(fitted.sl_single)}
    rows: list[dict[str, Any]] = []
    for arm, role in study.ARMS[fit]:
        cell = study.cell_name(fit, law, clusters, arm)
        if arm in {"t_reference", "glm_reference", "interpolating_control", "stacked_control"}:
            row = replicate_row(
                cell=cell, role=role, estimate=estimate, alpha=STUDY.margins.alpha, **common
            )
        elif arm == "iid_t_control":
            row = control_row(
                cell=cell,
                estimate=float(estimate.psi),
                standard_error=float(np.sqrt(influence_variance(estimate.influence_curve))),
                critical=t_critical(estimate.reference_df),
                **common,
            )
        elif arm == "wang_inflated":
            row = control_row(
                cell=cell,
                estimate=float(estimate.psi),
                standard_error=float(estimate.plugin_std_error)
                * float(np.sqrt(clusters / (clusters - WANG_P))),
                critical=t_critical(clusters - WANG_P),
                role=role,
                **common,
            )
        else:  # pragma: no cover - ARMS names every arm
            raise KeyError(arm)
        if not shown:
            # A withheld interval is a miss in every arm of the fit (plan 3.6).
            row["covered"] = 0
            row["rejected"] = 0
        rows.append({**row, **extra})
    return rows


#: One key per fit of a draw: ``(law, J, replicate, fit, requested)``.
_Key = tuple[str, int, int, str, int]


def _draw(key: _Key) -> pd.DataFrame:
    law, clusters, replicate = key[:3]
    return study.draw(law, clusters, _seed(law, clusters, replicate))


def _cells_of(key: _Key) -> list[str]:
    law, clusters, _, fit, _ = key
    return [f"{FAMILY}/{study.cell_name(fit, law, clusters, arm)}" for arm, _ in study.ARMS[fit]]


def _cell_of(key: _Key) -> str:
    law, clusters, _, fit, _ = key
    return f"{FAMILY}/{study.cell_name(fit, law, clusters, study.ARMS[fit][0][0])}"


def _replicate_of(key: _Key) -> int:
    return key[2]


def _key_rows(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    law, clusters, replicate, fit, requested = key
    return fit_rows(fit, law, clusters, replicate, study.fit_one(fit, frame), requested)


def keys(replicates: int) -> list[_Key]:
    """Every key of a run of ``replicates`` per cell, the expensive stacked fits first."""
    out = [
        (law, clusters, replicate, fit, replicates)
        for law, clusters in study.DRAW_CELLS
        for replicate in range(replicates)
        for fit in study.fits_of(law)
    ]
    weight = {"tmle_stacked": 0, "ltmle_crossfit": 0}
    return sorted(out, key=lambda key: (weight.get(key[3], 1), -key[1]))


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its law, its role and the root of its stream."""
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=study.cell_name(fit, law, clusters, arm),
            dgp=law,
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=study.CLUSTER_SIZE[law] * clusters,
            replicates=PROPERTY_REPLICATES,
            seed=_seed(law, clusters, 0),
            role=role,
            estimand="ate" if fit not in study.LTMLE_FITS else "ate_regimen[always vs never]",
        )
        for law, clusters in study.DRAW_CELLS
        for fit in study.fits_of(law)
        for arm, role in study.ARMS[fit]
    )


#: The arms of one fit, which pair each IID control with its positive arm.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    FAMILY: tuple(
        tuple(study.cell_name(fit, law, clusters, arm) for arm, _ in study.ARMS[fit])
        for law, clusters in study.DRAW_CELLS
        for fit in study.fits_of(law)
    )
}

#: The share of a cell's requested replicates the accounting lets a run lose. The plan asked
#: for 0.10 (Q3); the harness caps a bound at 0.05
#: (:data:`~tests.studies.evidence.property_failures.MAXIMUM_BOUND`). Every lost fit is a miss
#: in the worst case of G1, so the bound cannot help a positive cell pass.
FAILURE_BOUND = 0.05

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replicate; ``budget`` caps each cell for a pre-run check."""
    requested = PROPERTY_REPLICATES if budget is None else min(budget, PROPERTY_REPLICATES)
    outcomes = fit_properties(
        keys(requested),
        _key_rows,
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


def synthetic_columns(cell: Any, rng: np.random.Generator) -> dict[str, Any]:
    """The extra columns of a synthetic row for the property differential."""
    return {"withheld": int(rng.uniform() < 0.01), "sl_single": float(rng.uniform() < 0.3)}


# ---------------------------------------------------------------------------------- readings


def _cell_rows(rows: pd.DataFrame, cell: str) -> pd.DataFrame:
    return rows.loc[(rows["property"] == FAMILY) & (rows["cell"] == cell)]


def _shown(group: pd.DataFrame) -> pd.DataFrame:
    return group.loc[group["withheld"].to_numpy(dtype=float) < 0.5]


def adaptive_readings(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Fill :data:`READING_COLUMNS` on every row of the family."""
    margins = STUDY.margins
    level = margins.confidence_level
    family = summary.index[summary["property"] == FAMILY]
    groups = dict(tuple(rows.loc[rows["property"] == FAMILY].groupby("cell", sort=False)))
    empty = rows.iloc[0:0]

    def _cell_rows(_: pd.DataFrame, name: str) -> pd.DataFrame:
        return groups.get(name, empty)

    for index in family:
        cell = str(summary.loc[index, "cell"])
        fit, law, clusters, arm = parse(cell)
        group = _cell_rows(rows, cell)
        shown = _shown(group)
        values: dict[str, float] = {
            "clusters": float(clusters),
            "withheld_draws": float(len(group) - len(shown)),
            "shown_draws": float(len(shown)),
        }
        hits = int((shown["covered"] > 0).sum())
        if len(shown):
            interval = clopper_pearson(hits, len(shown), confidence_level=level)
            values["shown_coverage"] = hits / len(shown)
            values["shown_ci_lower"], values["shown_ci_upper"] = interval.low, interval.high
        ratio, low, high = variance_ratio_interval(
            shown["estimate"].to_numpy(),
            shown["std_error"].to_numpy(),
            replicates=margins.bootstrap_replicates,
            confidence_level=level,
            seed=stream_seed(STUDY, "variance_ratio", FAMILY, cell),
        )
        values["variance_ratio"] = ratio
        values["variance_ratio_ci_lower"], values["variance_ratio_ci_upper"] = low, high
        spread = float(shown["estimate"].std(ddof=1)) if len(shown) > 1 else float("nan")
        se = shown["std_error"].to_numpy(dtype=float)
        values["median_se_over_sd"] = float(np.median(se) / spread) if se.size else np.nan
        values["max_se_over_sd"] = float(np.max(se) / spread) if se.size else np.nan
        single = group["sl_single"].to_numpy(dtype=float)
        single = single[np.isfinite(single)]
        values["sl_single_share"] = float(single.mean()) if single.size else np.nan
        values["over_coverage"] = float(summary.loc[index, "coverage_ci_lower"] > OVER_COVERAGE)
        if arm == "t_reference":
            values["g1_met"] = float(
                summary.loc[index, "coverage_ci_lower"] >= margins.coverage_floor
            )
            values["variance_ratio_met"] = float(low >= VARIANCE_RATIO_FLOOR)
            values["shown_coverage_met"] = float(
                values.get("shown_ci_lower", -1.0) >= SHOWN_COVERAGE_FLOOR
            )
            counts = paired_counts(group, _cell_rows(rows, glm_cell(cell)))
            if counts["trials"]:
                values["noninferiority_difference"] = (counts["gained"] - counts["lost"]) / counts[
                    "trials"
                ]
                values["noninferiority_lower"] = mcnemar_lower(
                    counts["gained"], counts["lost"], counts["trials"], level
                )
            values["noninferiority_met"] = float(
                values.get("noninferiority_lower", -1.0) > NONINFERIORITY_MARGIN
            )
        if arm == "iid_t_control":
            positive = study.cell_name(fit, law, clusters, "t_reference")
            positive_rows = _cell_rows(rows, positive)
            paired = paired_se_ratio_interval(
                group,
                positive_rows,
                replicates=margins.bootstrap_replicates,
                confidence_level=level,
                seed=stream_seed(STUDY, FAMILY, cell),
            )
            positive_ratio = se_ratio_interval(
                positive_rows,
                replicates=margins.bootstrap_replicates,
                confidence_level=level,
                seed=stream_seed(STUDY, FAMILY, positive),
                truth_varies=False,
            )
            values["paired_se_ratio"] = paired.ratio
            values["paired_se_ratio_ci_lower"] = paired.interval.low
            values["paired_se_ratio_ci_upper"] = paired.interval.high
            values["paired_positive_se_ratio_ci_upper"] = positive_ratio.high
        for column, value in values.items():
            summary.loc[index, column] = value
    # The SE-ratio interval of every cell, which the IID control's second clause reads.
    for index in family:
        cell = str(summary.loc[index, "cell"])
        ratio_interval = se_ratio_interval(
            _cell_rows(rows, cell),
            replicates=margins.bootstrap_replicates,
            confidence_level=level,
            seed=stream_seed(STUDY, FAMILY, cell),
            truth_varies=False,
        )
        summary.loc[index, "se_ratio_ci_lower"] = ratio_interval.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio_interval.high


def adaptive_verdicts(summary: pd.DataFrame) -> None:
    """The declared verdicts of the family, read off its summary columns, and the caveat reading.

    A positive row passes on G1, G2, the variance-ratio and shown-coverage lower ends and, on
    L1, the paired non-inferiority clause. An IID control passes on both of its clauses, and a
    must-fail control when its all-draw coverage upper end is below 0.90. A diagnostic row
    states no verdict. ``caveat_removed`` holds on a positive row that passes with its IID
    control and its must-fail control.
    """
    margins = STUDY.margins
    family = summary["property"] == FAMILY
    by_cell = {str(summary.loc[i, "cell"]): i for i in summary.index[family]}
    for index in summary.index[family]:
        row = summary.loc[index]
        cell = str(row["cell"])
        role = str(row["role"])
        _, _, _, arm = parse(cell)
        if role == "positive":
            passed = (
                bool(row["coverage_ci_lower"] >= margins.coverage_floor)
                and bool(row["bias_equivalent"])
                and bool(row["variance_ratio_ci_lower"] >= VARIANCE_RATIO_FLOOR)
                and bool(row["shown_ci_lower"] >= SHOWN_COVERAGE_FLOOR)
            )
            if noninferiority_gated(cell):
                passed = passed and bool(row["noninferiority_lower"] > NONINFERIORITY_MARGIN)
        elif arm == "iid_t_control":
            passed = bool(
                row["paired_se_ratio_ci_upper"] < FEW_CLUSTER_IID_CONTROL_CEILING
                and row["paired_positive_se_ratio_ci_upper"] <= margins.se_ratio_sanity[1]
            )
        elif role == "control":
            passed = bool(row["coverage_ci_upper"] < MUST_FAIL_CEILING)
        else:
            passed = True
        summary.loc[index, "passed"] = passed
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        if str(summary.loc[index, "role"]) != "positive":
            continue
        controls = (by_cell[iid_cell(cell)], by_cell[must_fail_cell(cell)])
        summary.loc[index, "caveat_removed"] = float(
            bool(summary.loc[index, "passed"])
            and all(bool(summary.loc[i, "passed"]) for i in controls)
        )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the family's readings and its declared verdicts."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=READING_COLUMNS, rate_labels=()
    )
    adaptive_readings(summary, rows)
    adaptive_verdicts(summary)
    for index in summary.index[summary["property"] == FAMILY]:
        cell = str(summary.loc[index, "cell"])
        group = next(g for g in PROPERTY_GROUPS[FAMILY] if cell in g)
        members = summary.loc[(summary["property"] == FAMILY) & summary["cell"].isin(group)]
        judged = members.loc[members["role"] != property_verdicts.DIAGNOSTIC_ROLE, "passed"]
        summary.loc[index, "property_passed"] = (
            bool(judged.all()) if str(summary.loc[index, "role"]) != "diagnostic" else True
        )
    return finish(summary, rates)


# ------------------------------------------------------------------------------ declared rules


def _is(*arms: str) -> Any:
    def applies(ctx: Context, row: pd.Series) -> bool:
        return parse(str(row["cell"]))[3] in arms and str(row["role"]) != "diagnostic"

    return applies


def _vr_read(ctx: Context, row: pd.Series) -> Judged:
    low = float(row["variance_ratio_ci_lower"])
    return Judged(low - VARIANCE_RATIO_FLOOR, bool(low >= VARIANCE_RATIO_FLOOR))


def _fill_points(k: int, cap: int = 11) -> Iterable[int]:
    if k <= cap:
        return range(k + 1)
    return sorted({round(x) for x in np.linspace(0, k, cap)})


def _vr_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """The lost SEs at the survivor minimum, the lost estimates split between the extremes."""
    group = _shown(ctx.cell(FAMILY, str(row["cell"])))
    estimates = group["estimate"].to_numpy(dtype=float)
    errors = group["std_error"].to_numpy(dtype=float)
    k = ctx.k(row)
    out = []
    for j in _fill_points(k):
        fills = np.r_[np.full(j, estimates.max()), np.full(k - j, estimates.min())]
        _, low, _ = variance_ratio_interval(
            np.r_[estimates, fills],
            np.r_[errors, np.full(k, errors.min())],
            replicates=ctx.margins.bootstrap_replicates,
            confidence_level=ctx.margins.confidence_level,
            seed=stream_seed(STUDY, "variance_ratio", FAMILY, str(row["cell"])),
        )
        out.append(
            (
                f"j={j} at max, SE at min",
                Judged(low - VARIANCE_RATIO_FLOOR, bool(low >= VARIANCE_RATIO_FLOOR)),
            )
        )
    return out


def _sc_read(ctx: Context, row: pd.Series) -> Judged:
    low = float(row["shown_ci_lower"])
    return Judged(low - SHOWN_COVERAGE_FLOOR, bool(low >= SHOWN_COVERAGE_FLOOR))


def _sc_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """A lost fit is a shown draw that covers or misses: ``j`` of ``k`` cover."""
    trials = int(row["shown_draws"])
    hits = round(float(row["shown_coverage"]) * trials)
    k = ctx.k(row)
    level = ctx.margins.confidence_level
    out = []
    for j in range(k + 1):
        low = clopper_pearson(hits + j, trials + k, confidence_level=level).low
        out.append(
            (
                f"j={j} covered",
                Judged(low - SHOWN_COVERAGE_FLOOR, bool(low >= SHOWN_COVERAGE_FLOOR)),
            )
        )
    return out


def _ni_read(ctx: Context, row: pd.Series) -> Judged:
    value = float(row["noninferiority_lower"])
    return Judged(value - NONINFERIORITY_MARGIN, bool(value > NONINFERIORITY_MARGIN))


def _ni_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """A lost default-library fit at ``covered = 0``, a lost GLM fit at ``covered = 1`` (S7)."""
    cell = str(row["cell"])
    positive = ctx.cell(FAMILY, cell)
    glm = ctx.cell(FAMILY, glm_cell(cell))
    counts = lost_counts(positive, glm, int(row["requested_replicates"]))
    value = mcnemar_lower(
        counts["gained"], counts["lost"], counts["trials"], ctx.margins.confidence_level
    )
    return [
        (
            "lost default fit a miss, lost GLM fit a cover",
            Judged(value - NONINFERIORITY_MARGIN, bool(value > NONINFERIORITY_MARGIN)),
        )
    ]


def _ni_reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
    return [(FAMILY, glm_cell(str(row["cell"])))]


def _ni_applies(ctx: Context, row: pd.Series) -> bool:
    return noninferiority_gated(str(row["cell"]))


def _iid_seed(ctx: Context, row: pd.Series) -> int:
    return stream_seed(STUDY, FAMILY, str(row["cell"]))


#: The failure accounting's declared rules: the verdicts of :func:`adaptive_verdicts`.
PROPERTY_RULES = (
    FamilyRule(
        families=frozenset({FAMILY}),
        shared="summarize_properties",
        group="pair",
        clauses=(
            proportion_clause(
                "coverage floor",
                statistic="S2",
                judge=floor_judge(lambda ctx: ctx.margins.coverage_floor),
                applies=roles("positive"),
            ),
            bias_clause("bias equivalent", kind="equivalent", applies=roles("positive")),
            Clause(
                "variance ratio lower end",
                "S5",
                roles("positive"),
                _vr_read,
                _vr_candidates,
                imputation=RANGE,
            ),
            Clause(
                "shown coverage lower end",
                "S2",
                roles("positive"),
                _sc_read,
                _sc_candidates,
                imputation=SUPPORT,
            ),
            Clause(
                "paired non-inferiority",
                "S7",
                _ni_applies,
                _ni_read,
                _ni_candidates,
                imputation=SUPPORT,
                reads=_ni_reads,
            ),
            paired_se_ratio_clause(
                "paired iid/cluster se ratio under its ceiling",
                judge=below_judge(lambda ctx: FEW_CLUSTER_IID_CONTROL_CEILING),
                seed=_iid_seed,
                applies=_is("iid_t_control"),
            ),
            paired_positive_se_clause(
                "paired positive se ratio at most the sanity ceiling",
                judge=ceiling_judge(lambda ctx: ctx.margins.se_ratio_sanity[1]),
                seed=_iid_seed,
                applies=_is("iid_t_control"),
                truth_varies=lambda ctx: False,
            ),
            proportion_clause(
                "must fail",
                statistic="S2",
                judge=below_judge(lambda ctx: MUST_FAIL_CEILING),
                applies=_is("interpolating_control", "stacked_control"),
            ),
        ),
    ),
)


def caveat_outcomes(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Each positive configuration's clause readings and caveat, keyed by its cell.

    The input of the declared consequences of plan section 5.8 at registration.
    """
    out: dict[str, dict[str, Any]] = {}
    family = summary.loc[(summary["property"] == FAMILY) & (summary["role"] == "positive")]
    for _, row in family.iterrows():
        out[str(row["cell"])] = {
            "g1": bool(row["g1_met"]),
            "g2": bool(row["bias_equivalent"]),
            "variance_ratio": bool(row["variance_ratio_met"]),
            "shown_coverage": bool(row["shown_coverage_met"]),
            "noninferiority": bool(row["noninferiority_met"])
            if noninferiority_gated(str(row["cell"]))
            else None,
            "caveat_removed": bool(row["caveat_removed"]),
        }
    return out


__all__: Sequence[str] = (
    "PROPERTY_RULES",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
)
