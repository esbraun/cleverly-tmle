r"""The ``boundary_node`` family of ``boundary-node-ltmle``: rows, readings and declared rules.

Each key draws one sample of a cell of :data:`tests.studies.boundary_node_ltmle.CELLS`, fits it
once, and writes nine rows: three targets by three arms
(:func:`tests.studies.boundary_node_ltmle.draw_arms`). Beside the shared replicate columns a
row records ``firing`` (node 4 of the target's firing regimen flagged), ``multi_flag``
(firing, and another node flagged), ``supplemented`` (the estimate added the supplement at any
node) and ``withheld`` (the estimate withheld its interval, so every arm reads a miss). A key whose fit raises is lost in all nine cells, and the declared
rules read it as a miss wherever it could fall.

The gates are the statistics the supplement changes (plan rev 3.1):

==========================  =====================================================================
rule                        rows and threshold
==========================  =====================================================================
``bias``                    every ``supplemented`` row: G2, the 99% bias interval inside
                            :math:`\pm 0.25` SD
``firing floor``            every ``supplemented`` row with 600 or more expected firing draws at
                            R = 4,000 (exact, from the law): the exact 99% lower end of coverage
                            on the firing draws at least 0.90
``zero-cell coverage``      ``risk[always]`` of both zero variants, where every draw fires: G1
                            with the exact 99% lower end at least 0.92
``paired gain``             the ``plugin`` row of law A, ``always``, :math:`n = 200`: on firing
                            draws the exact McNemar lower end of supplemented minus plug-in
                            coverage above 0 (clause (i))
``plug-in loses coverage``  the same row: Newcombe's 99% interval of plug-in coverage on firing
                            minus non-firing draws wholly below 0 (clause (ii))
==========================  =====================================================================

Every other row is ``diagnostic`` and publishes its readings without a verdict. Every row
publishes the all-draw coverage and its exact interval, the firing, quiet and multi-flag
coverages, and the SE ratio on the firing draws with its 99% bootstrap interval. A quiet draw
is neither firing nor supplemented, so it shows the ordinary interval; ``rare_event_limit``
marks a row whose quiet coverage has an exact lower end below 0.90, the shortfall the
declaration assigns to F31's rare-event limit. A ``supplemented`` row also publishes the median
ratio of its half-width to the ``oracle_node`` half-width on firing draws, with
``firing_above_099``, the over-coverage reading (the firing lower end above 0.99).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import boundary_node_laws as laws
from tests.studies import boundary_node_ltmle as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_failures import (
    SUPPORT,
    Clause,
    Context,
    FamilyRule,
    Judged,
    bias_clause,
    cells,
    floor_judge,
    proportion_clause,
    roles,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY
PROPERTY_REPLICATES: Final[int] = study.PROPERTY_REPLICATES

#: The exact lower end that a gated firing coverage must reach (plan 5.5).
FIRING_FLOOR: Final[float] = 0.90
#: The exact lower end of the zero cells' all-draw coverage of ``risk[always]`` (plan 2.5(5)).
ZERO_FLOOR: Final[float] = 0.92
#: The over-coverage reading's threshold.
OVER_COVERAGE: Final[float] = 0.99
#: The bootstrap resamples of the firing SE ratio's interval, a reading.
SE_RATIO_RESAMPLES: Final[int] = 1_000

#: The columns a row carries beside :data:`REPLICATE_COLUMNS`.
EXTRA_COLUMNS: Final[tuple[str, ...]] = ("firing", "multi_flag", "supplemented", "withheld")

#: The readings :func:`summarize_properties` adds to every row.
READING_COLUMNS: Final[tuple[str, ...]] = (
    "expected_firing",
    "firing_gated",
    "firing_draws",
    "firing_coverage",
    "firing_ci_lower",
    "firing_ci_upper",
    "firing_above_099",
    "firing_se_ratio",
    "firing_se_ratio_ci_lower",
    "firing_se_ratio_ci_upper",
    "quiet_draws",
    "quiet_coverage",
    "quiet_ci_lower",
    "quiet_ci_upper",
    "rare_event_limit",
    "multi_flag_draws",
    "multi_flag_coverage",
    "withheld_draws",
    "all_draw_floor_met",
    "width_ratio_to_oracle",
    "mcnemar_gained",
    "mcnemar_lost",
    "mcnemar_lower",
    "newcombe_lower",
    "newcombe_upper",
)


def _cell(cell: str, target: str, arm: str) -> str:
    return study.cell_name(cell, target, arm)


#: The positive cells whose ``firing floor`` is gated: 600 or more expected firing draws.
FIRING_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(cell.name, target, "supplemented")
    for cell in study.CELLS
    for target in study.TARGETS
    if study.firing_gated(cell, target)
)
#: The zero cells' ``risk[always]`` rows, gated on all-draw coverage.
ZERO_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(cell.name, "always", "supplemented") for cell in study.CELLS if cell.zero
)
#: The control row of ``paired_attributable`` and the positive row it is paired with.
CONTROL_CELL: Final[str] = _cell(*study.PAIRED_CONTROL, "plugin")
PAIRED_POSITIVE: Final[str] = _cell(*study.PAIRED_CONTROL, "supplemented")


def _role(name: str) -> str:
    cell, target, arm = name.split("__")
    return study.arm_role(cell, target, arm)


# ------------------------------------------------------------------------------- statistics


def mcnemar_lower(gained: int, lost: int, trials: int, level: float = 0.99) -> float:
    """The lower end of the paired difference in coverage, exact on the discordant draws.

    The discordant draws are ``gained`` (supplemented covers, plug-in misses) and ``lost``;
    the share :math:`\\theta` of ``gained`` among them has a Clopper-Pearson interval, and the
    difference is :math:`(2\\theta - 1)(\\text{gained} + \\text{lost}) / \\text{trials}`.
    """
    discordant = gained + lost
    if discordant == 0 or trials == 0:
        return 0.0
    low = clopper_pearson(gained, discordant, confidence_level=level).low
    return (2.0 * low - 1.0) * discordant / trials


def newcombe_interval(
    x1: int, n1: int, x2: int, n2: int, level: float = 0.99
) -> tuple[float, float]:
    """Newcombe's hybrid score interval of :math:`p_1 - p_2` (two independent binomials)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))

    def wilson(x: int, n: int) -> tuple[float, float, float]:
        p = x / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return p, centre - half, centre + half

    p1, l1, u1 = wilson(x1, n1)
    p2, l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return (
        d - float(np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
        d + float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)),
    )


# ------------------------------------------------------------------------------------- rows


def _draw(key: tuple[str, int]) -> pd.DataFrame:
    name, replicate = key
    cell = study.CELL_BY_NAME[name]
    seed = stream_seed(STUDY, "property_sample", FAMILY, name, replicate)
    return laws.law_of(cell.variant).draw(
        cell.n, np.random.default_rng(seed), cluster_size=cell.cluster_size
    )


def _requested(budget: int | None) -> int:
    return PROPERTY_REPLICATES if budget is None else min(budget, PROPERTY_REPLICATES)


def key_rows(
    name: str, replicate: int, entries: Sequence[Mapping[str, Any]], requested: int
) -> list[dict[str, Any]]:
    """The nine rows of one key, off :func:`~tests.studies.boundary_node_ltmle.draw_arms`."""
    cell = study.CELL_BY_NAME[name]
    rows = []
    for entry in entries:
        for arm in study.ARMS:
            published = _cell(name, entry["target"], arm)
            reading = entry[arm]
            rows.append(
                {
                    "property": FAMILY,
                    "cell": published,
                    "role": _role(published),
                    "replicate": replicate,
                    "n": cell.n,
                    "requested_replicates": requested,
                    "failed_replicates": 0,
                    "truth": float(entry["truth"]),
                    "estimate": float(entry["psi"]),
                    "std_error": float(reading["std_error"]),
                    "covered": int(reading["covered"]),
                    "rejected": int(reading["rejected"]),
                    "firing": int(bool(entry["firing"])),
                    "multi_flag": int(bool(entry["multi_flag"])),
                    "supplemented": int(bool(entry["supplemented_here"])),
                    "withheld": int(bool(entry["withheld"])),
                }
            )
    return rows


def _fit_key(requested: int) -> Any:
    def fit_one(key: tuple[str, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
        name, replicate = key
        cell = study.CELL_BY_NAME[name]
        truth = laws.law_of(cell.variant).truths()
        return key_rows(name, replicate, study.draw_arms(frame, cell, truth), requested)

    return fit_one


def _cells_of(key: tuple[str, int]) -> list[str]:
    """Every target and arm of the key's cell reads its one fit."""
    return [
        f"{FAMILY}/{_cell(key[0], target, arm)}" for target in study.TARGETS for arm in study.ARMS
    ]


def _cell_of(key: tuple[str, int]) -> str:
    return f"{FAMILY}/{key[0]}"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


def declared_cells() -> tuple[Any, ...]:
    """Every published cell, with its role, size and requested replicates."""
    from tests.studies.evidence.properties import PropertyCell

    out = []
    for cell in study.CELLS:
        for target in study.TARGETS:
            for arm in study.ARMS:
                name = _cell(cell.name, target, arm)
                out.append(
                    PropertyCell(
                        property=FAMILY,
                        cell=name,
                        dgp=laws.law_of(cell.variant),
                        outcome_learner=lambda: None,
                        treatment_learner=lambda: None,
                        n=cell.n,
                        replicates=PROPERTY_REPLICATES,
                        seed=stream_seed(STUDY, "property_sample", FAMILY, cell.name),
                        role=_role(name),
                        estimand=study.TARGETS[target],
                    )
                )
    return tuple(out)


def synthetic_columns(cell: Any, rng: np.random.Generator) -> dict[str, Any]:
    """The extra columns of a synthetic row for the property differential."""
    firing = int(rng.uniform() < 0.5)
    return {
        "firing": firing,
        "multi_flag": int(firing and rng.uniform() < 0.1),
        "supplemented": int(firing or rng.uniform() < 0.05),
        "withheld": 0,
    }


def synthetic_centre(cell: Any, columns: Mapping[str, Any], centre: float, scale: float) -> float:
    """A synthetic row's centre: the control misses on its firing draws and covers elsewhere.

    The paired control's clauses read the plug-in's coverage on firing against non-firing
    draws, so its synthetic rows lose coverage on the firing draws only, as the law does.
    """
    if cell.role == "control":
        return centre if columns["firing"] else 0.0
    return centre


#: The share of a cell's requested replications that the accounting lets a run lose. The plan
#: asked for 0.10; the harness caps a bound at 0.05
#: (:data:`~tests.studies.evidence.property_failures.MAXIMUM_BOUND`). The pilot lost 6 of
#: 2,000 law A draws at n = 100 and 3 of 2,000 cross-fitted ones, each because no unit followed
#: ``always`` through node 4 in the sample or in an outer training fold, so the worst cell
#: expects about 12 of its 200 allowed losses. The study GLM keeps its last iterate on a
#: numerically separated pseudo-outcome, so its IRLS loses no draw. Every lost draw is read as a
#: miss wherever it can fall.
FAILURE_BOUND = 0.05

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pre-run check."""
    requested = _requested(budget)
    outcomes = fit_properties(
        [(cell.name, replicate) for cell in study.CELLS for replicate in range(requested)],
        _fit_key(requested),
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


def _flag(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype=float) > 0.5


def _proportion(hits: int, trials: int, level: float) -> tuple[float, float, float]:
    if trials == 0:
        return (float("nan"), float("nan"), float("nan"))
    interval = clopper_pearson(hits, trials, confidence_level=level)
    return hits / trials, interval.low, interval.high


def _paired(rows: pd.DataFrame, control: str, positive: str) -> pd.DataFrame:
    """The control's rows beside the positive's ``covered``, joined on the replicate."""
    left = rows.loc[(rows["property"] == FAMILY) & (rows["cell"] == control)]
    right = rows.loc[(rows["property"] == FAMILY) & (rows["cell"] == positive)]
    return left[["replicate", "covered", "firing"]].merge(
        right[["replicate", "covered"]], on="replicate", suffixes=("", "_positive")
    )


def paired_counts(rows: pd.DataFrame) -> dict[str, int]:
    """The paired control's counts: firing draws, gains, losses, plug-in hits by group."""
    pair = _paired(rows, CONTROL_CELL, PAIRED_POSITIVE)
    fire = _flag(pair["firing"])
    plug = pair["covered"].to_numpy() > 0
    sup = pair["covered_positive"].to_numpy() > 0
    return {
        "firing": int(fire.sum()),
        "quiet": int((~fire).sum()),
        "gained": int(np.sum(fire & sup & ~plug)),
        "lost": int(np.sum(fire & ~sup & plug)),
        "plug_firing": int(np.sum(fire & plug)),
        "plug_quiet": int(np.sum(~fire & plug)),
    }


def firing_se_ratio(
    group: pd.DataFrame, mask: np.ndarray, level: float, seed: int
) -> tuple[float, float, float]:
    """Mean reported SE over the spread of the errors on ``mask``, with a bootstrap interval."""
    errors = (group["estimate"] - group["truth"]).to_numpy(dtype=float)[mask]
    reported = group["std_error"].to_numpy(dtype=float)[mask]
    keep = np.isfinite(errors) & np.isfinite(reported)
    errors, reported = errors[keep], reported[keep]
    if errors.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, errors.size, (SE_RATIO_RESAMPLES, errors.size))
    ratios = reported[picks].mean(axis=1) / errors[picks].std(axis=1, ddof=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(ratios, [tail, 1.0 - tail])
    return float(reported.mean() / errors.std(ddof=1)), float(low), float(high)


def boundary_node_readings(summary: pd.DataFrame, rows: pd.DataFrame, level: float) -> None:
    """Fill :data:`READING_COLUMNS` on every row of the family."""
    family = rows.loc[rows["property"] == FAMILY]
    groups = dict(tuple(family.groupby("cell", sort=False)))
    for index in summary.index[summary["property"] == FAMILY]:
        name = str(summary.loc[index, "cell"])
        cell_name, target, arm = name.split("__")
        cell = study.CELL_BY_NAME[cell_name]
        group = groups[name]
        fire = _flag(group["firing"])
        quiet = ~fire & ~_flag(group["supplemented"])
        covered = group["covered"].to_numpy() > 0
        multi = _flag(group["multi_flag"])
        values: dict[str, float] = {
            "expected_firing": study.expected_firing(cell, target),
            "firing_gated": float(name in FIRING_CELLS),
            "firing_draws": float(fire.sum()),
            "quiet_draws": float(quiet.sum()),
            "multi_flag_draws": float(multi.sum()),
            "multi_flag_coverage": float(covered[multi].mean()) if multi.any() else np.nan,
            "withheld_draws": float(_flag(group["withheld"]).sum()),
            "all_draw_floor_met": float(summary.loc[index, "coverage_ci_lower"] >= FIRING_FLOOR),
        }
        for prefix, mask in (("firing", fire), ("quiet", quiet)):
            rate, low, high = _proportion(int(covered[mask].sum()), int(mask.sum()), level)
            values[f"{prefix}_coverage"] = rate
            values[f"{prefix}_ci_lower"] = low
            values[f"{prefix}_ci_upper"] = high
        values["firing_above_099"] = float(values["firing_ci_lower"] > OVER_COVERAGE)
        values["rare_event_limit"] = float(values["quiet_ci_lower"] < FIRING_FLOOR)
        seed = stream_seed(STUDY, "firing_se_ratio", FAMILY, name)
        (
            values["firing_se_ratio"],
            values["firing_se_ratio_ci_lower"],
            values["firing_se_ratio_ci_upper"],
        ) = firing_se_ratio(group, fire, level, seed)
        if arm == "supplemented":
            oracle = groups[_cell(cell_name, target, "oracle_node")]
            joined = group[["replicate", "std_error", "firing"]].merge(
                oracle[["replicate", "std_error"]], on="replicate", suffixes=("", "_oracle")
            )
            firing = joined.loc[_flag(joined["firing"])]
            values["width_ratio_to_oracle"] = (
                float(np.median(firing["std_error"] / firing["std_error_oracle"]))
                if len(firing)
                else np.nan
            )
        if name == CONTROL_CELL:
            counts = paired_counts(rows)
            values["mcnemar_gained"] = float(counts["gained"])
            values["mcnemar_lost"] = float(counts["lost"])
            values["mcnemar_lower"] = mcnemar_lower(
                counts["gained"], counts["lost"], counts["firing"], level
            )
            values["newcombe_lower"], values["newcombe_upper"] = newcombe_interval(
                counts["plug_firing"],
                counts["firing"],
                counts["plug_quiet"],
                counts["quiet"],
                level,
            )
        for column, value in values.items():
            summary.loc[index, column] = value


def boundary_node_verdicts(summary: pd.DataFrame) -> None:
    """The declared verdicts of the family, read off its summary columns.

    A ``supplemented`` row passes on G2, on its firing floor where gated, and on the zero-cell
    floor where declared. The control passes when both paired clauses hold. A diagnostic row
    states no verdict.
    """
    family = summary["property"] == FAMILY
    for index in summary.index[family]:
        row = summary.loc[index]
        name, role = str(row["cell"]), str(row["role"])
        if role == "positive":
            passed = bool(row["bias_equivalent"])
            if name in FIRING_CELLS:
                passed = passed and bool(row["firing_ci_lower"] >= FIRING_FLOOR)
            if name in ZERO_CELLS:
                passed = passed and bool(row["coverage_ci_lower"] >= ZERO_FLOOR)
        elif role == "control":
            passed = bool(row["mcnemar_lower"] > 0.0) and bool(row["newcombe_upper"] < 0.0)
        else:
            passed = True
        summary.loc[index, "passed"] = passed


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the family's readings and its declared verdicts."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=READING_COLUMNS, rate_labels=()
    )
    boundary_node_readings(summary, rows, STUDY.margins.confidence_level)
    boundary_node_verdicts(summary)
    return finish(summary, rates)


# ------------------------------------------------------------------------------ declared rules


def _firing_read(ctx: Context, row: pd.Series) -> Judged:
    low = float(row["firing_ci_lower"])
    return Judged(low - FIRING_FLOOR, bool(low >= FIRING_FLOOR))


def _firing_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """Each lost draw is a firing miss or a non-firing draw: ``j`` of ``k`` fire."""
    trials = int(row["firing_draws"])
    hits = round(float(row["firing_coverage"]) * trials) if trials else 0
    level = ctx.margins.confidence_level
    out = []
    for j in range(ctx.k(row) + 1):
        low = clopper_pearson(hits, trials + j, confidence_level=level).low if trials + j else 0.0
        out.append((f"j={j} firing misses", Judged(low - FIRING_FLOOR, bool(low >= FIRING_FLOOR))))
    return out


def _control_counts(ctx: Context) -> dict[str, int]:
    return paired_counts(ctx.rows)


def _gain_read(ctx: Context, row: pd.Series) -> Judged:
    value = float(row["mcnemar_lower"])
    return Judged(value, bool(value > 0.0))


def _gain_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """A lost draw is a concordant miss in both arms: ``j`` of ``k`` fire."""
    counts = _control_counts(ctx)
    level = ctx.margins.confidence_level
    out = []
    for j in range(ctx.k(row) + 1):
        value = mcnemar_lower(counts["gained"], counts["lost"], counts["firing"] + j, level)
        out.append((f"j={j} firing", Judged(value, bool(value > 0.0))))
    return out


def _loss_read(ctx: Context, row: pd.Series) -> Judged:
    high = float(row["newcombe_upper"])
    return Judged(-high, bool(high < 0.0))


def _loss_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """A lost draw is a plug-in miss: ``j`` of ``k`` among the firing draws, the rest not."""
    counts = _control_counts(ctx)
    level = ctx.margins.confidence_level
    k = ctx.k(row)
    out = []
    for j in range(k + 1):
        _, high = newcombe_interval(
            counts["plug_firing"],
            counts["firing"] + j,
            counts["plug_quiet"],
            counts["quiet"] + k - j,
            level,
        )
        out.append((f"j={j} firing", Judged(-high, bool(high < 0.0))))
    return out


def _paired_reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
    return [(FAMILY, PAIRED_POSITIVE)]


#: The failure accounting's declared rules: the verdicts of :func:`boundary_node_verdicts`.
PROPERTY_RULES = (
    FamilyRule(
        families=frozenset({FAMILY}),
        shared="summarize_properties",
        clauses=(
            bias_clause("bias", kind="equivalent", applies=roles("positive")),
            Clause(
                "firing floor",
                "S2",
                cells(*FIRING_CELLS),
                _firing_read,
                _firing_candidates,
                imputation=SUPPORT,
            ),
            proportion_clause(
                "zero-cell coverage",
                statistic="S2",
                judge=floor_judge(lambda ctx: ZERO_FLOOR),
                applies=cells(*ZERO_CELLS),
            ),
            Clause(
                "paired gain",
                "S7",
                cells(CONTROL_CELL),
                _gain_read,
                _gain_candidates,
                imputation=SUPPORT,
                reads=_paired_reads,
            ),
            Clause(
                "plug-in loses coverage",
                "S2",
                cells(CONTROL_CELL),
                _loss_read,
                _loss_candidates,
                imputation=SUPPORT,
            ),
        ),
    ),
)
