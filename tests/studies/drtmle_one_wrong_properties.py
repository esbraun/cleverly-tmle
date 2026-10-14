"""Property cells of the DR-TMLE one-wrong witness study (RM40 plan section 5).

Two families, fixed before any registered run.

``one_wrong_inference`` answers to the rules of
:func:`~tests.studies.evidence.property_verdicts.one_wrong_inference_verdicts`, selected by the
last ``__`` segment of each cell:

==========================================  =========  ======  =====  ============  ====
cell                                        role       n       R      estimands     rule
==========================================  =========  ======  =====  ============  ====
``q_wrong_g_glm__ate__conservative``        positive   4,000   4,000  ate           C
``q_wrong_g_glm__{ey1,ey0}__valid``         positive   4,000   4,000  arms          V
``q_wrong_g_known__{ate,ey1,ey0}__valid``   positive   4,000   4,000  all           V
``q_wrong_g_glm_cf__ate__conservative``     positive   4,000   3,000  ate           C
``q_wrong_g_glm_cf__{ey1,ey0}__valid``      positive   4,000   3,000  arms          V
``q_wrong_g_slow_n<n>__ate__first_order``   control    ladder  1,000  ate           F
``q_wrong_g_slow__ate__growth``             control    ladder         ate           G
``q_slow_g_wrong_n<n>__ate__first_order``   control    ladder  1,000  ate           F
``q_slow_g_wrong__ate__growth``             control    ladder         ate           G
``mar_q_wrong_pi_slow__{ey1,ey0}__first_order``  control  4,000  1,000  arms        F
==========================================  =========  ======  =====  ============  ====

The ladder is n = 1,000, 4,000 and 16,000.  ``q_wrong_g_known`` reads the same draws as
``q_wrong_g_glm``, so the pair is a paired attribution.

``one_wrong_reported`` publishes the arms of the ladders and law M's ATE with the diagnostic
role: each row states its bias, SE ratio and coverage, and no verdict.

Every row also carries the fit's remainder proxy, the projected standard error of the
parametric-mechanism cells, the outer-loop exit and the seconds the fit took.  The summary
publishes them as reported columns: ``remainder_proxy_over_sd``, ``projected_se_ratio``,
``nonconverged_fits``, ``seconds_per_fit``, and on the ``q_wrong_g_glm`` rows the paired
spread ratio against ``q_wrong_g_known``.  None of them decides a verdict.

**Red-row reading, declared before the run (RM40 plan 5.9).**  Rule C departs from the brief's
calibration reading, because a conservative cell's target is not nominal.  Its predicted
coverage is ``2 Phi(1.96 rho) - 1``: :data:`PLANNING_COVERAGE` at the planning values
:data:`PLANNING_RATIO`.

* A red C cell whose SE-ratio interval reaches 1.07 is finite-sample only when its 99% SE-ratio
  interval contains the planning value, its 99% coverage interval contains the planning
  coverage, and the diagnosis (projected SE ratio, exits) finds no defect.  It then goes to
  ``reporting`` with the owner :data:`RED_CELL_OWNER`.  Otherwise it is a defect suspect.
* A C or V cell with coverage below the floor or a bias outside the margin is a defect
  suspect: diagnose, fix, and re-run under a fresh declaration.
* A red V cell on ``q_wrong_g_known`` is a defect suspect, because H2 is exact.
* A red F or G cell is a defect suspect in the design or the estimator, diagnosed with the
  remainder proxy.
* Any other calibration-shaped row follows the brief's rule unchanged.

No replications are ever added, and an unfixed defect never ends as an owner row.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import drtmle_one_wrong as study
from tests.studies.drtmle_one_wrong import STUDY
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    paired_spread_ratio_interval,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    ONE_WRONG_COLUMNS,
    ONE_WRONG_FAMILY,
    apply_shared_verdicts,
    finish,
    one_wrong_inference_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

REPORTED_FAMILY = "one_wrong_reported"
RED_CELL_OWNER = "RM40-conservative-resolution"

PAIRED_N = 4_000
PAIRED_REPLICATES = 4_000
CROSS_FIT_N = 4_000
CROSS_FIT_REPLICATES = 3_000
LADDER = (1_000, 4_000, 16_000)
LADDER_REPLICATES = 1_000
LAW_M_N = 4_000
LAW_M_REPLICATES = 1_000

#: The paired settings, in the order a replication fits them.
PAIRED = ("q_wrong_g_glm", "q_wrong_g_known")
CROSS_FIT = "q_wrong_g_glm_cf"
LADDERS = {"q_wrong_g_slow": LADDER, "q_slow_g_wrong": LADDER}
LAW_M = "mar_q_wrong_pi_slow"

#: The sign of each first-order control's bias, which the design test computes from the
#: population conditioning term and pins equal to this declaration.
PREDICTED_SIGNS = {
    "q_wrong_g_slow__ate": -1,
    "q_slow_g_wrong__ate": -1,
    "mar_q_wrong_pi_slow__ey1": -1,
    "mar_q_wrong_pi_slow__ey0": -1,
}

#: Rule C's planning SE ratios: in sample, the one-sided lower 90% bound of the probe's
#: pooled ratio ``1.1441 +/- 0.0148`` (``reviews/rm40/pool_se_ratio.py``); cross-fitted, the
#: lower 99% bound of its R = 400 probe ratio 1.231.  The 1.185 projection at the outcome
#: GLM's limit lies 2.8 standard errors above the pooled 1.1441; that gap is an open
#: diagnostic, which the projected SE ratio column reads.
POOLED_RATIO = 1.1441
POOLED_RATIO_SE = 0.0148
PLANNING_RATIO = {
    "q_wrong_g_glm": POOLED_RATIO - 1.2816 * POOLED_RATIO_SE,
    "q_wrong_g_glm_cf": 1.138,
}
#: Each C cell's predicted coverage, ``2 Phi(1.96 rho) - 1`` at its planning ratio.
PLANNING_COVERAGE = {"q_wrong_g_glm": 0.972, "q_wrong_g_glm_cf": 0.974}

SEEDS = {
    "paired": "q_wrong_paired",
    CROSS_FIT: CROSS_FIT,
    **{label: label for label in LADDERS},
    LAW_M: LAW_M,
}

TRUTH = study.truth()

#: The diagnostic columns each property row carries beside the shared schema.
DIAGNOSTIC_COLUMNS = (
    "remainder_proxy",
    "projected_std_error",
    "converged",
    "rounds",
    "exit_reason",
    "seconds",
)
#: The reported columns each summary row carries.
REPORTED_COLUMNS = (
    "remainder_proxy_over_sd",
    "projected_se_ratio",
    "nonconverged_fits",
    "seconds_per_fit",
    "paired_sd_ratio",
    "paired_sd_ratio_ci_lower",
    "paired_sd_ratio_ci_upper",
)


def _sample(label: str, replicate: int, n: int, law: str) -> pd.DataFrame:
    seed = stream_seed(STUDY, "property_sample", SEEDS[label], n, replicate)
    return study.draw_law_m(n, seed) if law == "M" else study.draw_law_l(n, seed)


def _rows(
    setting: str,
    frame: pd.DataFrame,
    *,
    replicate: int,
    requested: int,
    cells: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Fit one setting and write one row per ``estimand -> (cell, family)``."""
    result, record = study.fit_record(setting, frame)
    rows = []
    for estimand, (cell, family) in cells.items():
        role = DIAGNOSTIC_ROLE if family == REPORTED_FAMILY else _role(cell)
        row = replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            replicate=replicate,
            n=len(frame),
            requested=requested,
            truth=TRUTH[estimand],
            estimate=result[estimand],
            alpha=STUDY.margins.alpha,
        )
        row.update(
            remainder_proxy=record.remainder[estimand],
            projected_std_error=record.projected_std_error[estimand],
            converged=int(record.converged),
            rounds=record.rounds,
            exit_reason=record.exit_reason,
            seconds=record.seconds,
        )
        rows.append(row)
    return rows


def _role(cell: str) -> str:
    kind = cell.rsplit("__", 1)[-1]
    return "positive" if kind in {"conservative", "valid"} else "control"


def _positive_cells(setting: str) -> dict[str, tuple[str, str]]:
    first = "valid" if setting == "q_wrong_g_known" else "conservative"
    return {
        "ate": (f"{setting}__ate__{first}", ONE_WRONG_FAMILY),
        "ey1": (f"{setting}__ey1__valid", ONE_WRONG_FAMILY),
        "ey0": (f"{setting}__ey0__valid", ONE_WRONG_FAMILY),
    }


def _ladder_cells(label: str, n: int) -> dict[str, tuple[str, str]]:
    return {
        "ate": (f"{label}_n{n}__ate__first_order", ONE_WRONG_FAMILY),
        "ey1": (f"{label}_n{n}__ey1__reported", REPORTED_FAMILY),
        "ey0": (f"{label}_n{n}__ey0__reported", REPORTED_FAMILY),
    }


LAW_M_CELLS = {
    "ey1": (f"{LAW_M}__ey1__first_order", ONE_WRONG_FAMILY),
    "ey0": (f"{LAW_M}__ey0__first_order", ONE_WRONG_FAMILY),
    "ate": (f"{LAW_M}__ate__reported", REPORTED_FAMILY),
}


_Key = tuple[tuple[str, str, int, int, int]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    block, label, n, replicate, _ = key[0]
    if block == "paired":
        return _sample("paired", replicate, n, "L")
    if block == "cross_fit":
        return _sample(CROSS_FIT, replicate, n, "L")
    if block == "ladder":
        return _sample(label, replicate, n, "L")
    return _sample(LAW_M, replicate, n, "M")


def _block_cells(block: str, label: str, n: int) -> dict[str, tuple[str, str]]:
    """The ``estimand -> (cell, family)`` maps of a block, one per fitted setting."""
    if block == "paired":
        return {
            f"{setting}/{estimand}": value
            for setting in PAIRED
            for estimand, value in _positive_cells(setting).items()
        }
    if block == "cross_fit":
        return _positive_cells(CROSS_FIT)
    if block == "ladder":
        return _ladder_cells(label, n)
    return LAW_M_CELLS


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows fill."""
    block, label, n, _, _ = key[0]
    return [f"{family}/{cell}" for cell, family in _block_cells(block, label, n).values()]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[0][3]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    block, label, n, replicate, requested = key[0]
    if block == "paired":
        return [
            row
            for setting in PAIRED
            for row in _rows(
                setting,
                frame,
                replicate=replicate,
                requested=requested,
                cells=_positive_cells(setting),
            )
        ]
    if block == "cross_fit":
        return _rows(
            CROSS_FIT,
            frame,
            replicate=replicate,
            requested=requested,
            cells=_positive_cells(CROSS_FIT),
        )
    if block == "ladder":
        return _rows(
            label, frame, replicate=replicate, requested=requested, cells=_ladder_cells(label, n)
        )
    return _rows(LAW_M, frame, replicate=replicate, requested=requested, cells=LAW_M_CELLS)


def payloads(budget: int | None = None) -> list[_Key]:
    """Every replication of the study, the largest fits first.

    ``budget`` caps each block's replications for a smoke run, which then requests what it
    runs.
    """

    def capped(count: int) -> int:
        return count if budget is None else min(budget, count)

    blocks: list[tuple[str, str, int, int]] = [
        *(
            ("ladder", label, n, LADDER_REPLICATES)
            for n in sorted(LADDER, reverse=True)
            for label in LADDERS
        ),
        ("cross_fit", CROSS_FIT, CROSS_FIT_N, CROSS_FIT_REPLICATES),
        ("paired", "paired", PAIRED_N, PAIRED_REPLICATES),
        ("law_m", LAW_M, LAW_M_N, LAW_M_REPLICATES),
    ]
    return [
        ((block, label, n, replicate, capped(count)),)
        for block, label, n, count in blocks
        for replicate in range(capped(count))
    ]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries.

    Not named ``declared_cells``: the shared evidence tests read that name as a tuple of
    :class:`~tests.studies.evidence.properties.PropertyCell` objects with a ``dgp``.
    """
    cells: dict[str, tuple[str, str]] = {}
    for setting in (*PAIRED, CROSS_FIT):
        cells.update(
            {cell: (family, name) for name, (cell, family) in _positive_cells(setting).items()}
        )
    for label, sizes in LADDERS.items():
        for n in sizes:
            cells.update(
                {cell: (family, name) for name, (cell, family) in _ladder_cells(label, n).items()}
            )
    cells.update({cell: (family, name) for name, (cell, family) in LAW_M_CELLS.items()})
    return tuple(
        (family, cell, name, TRUTH[name]) for cell, (family, name) in sorted(cells.items())
    )


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each block for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        _replication,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    return rows.loc[:, [*REPLICATE_COLUMNS, *DIAGNOSTIC_COLUMNS]].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def _reported(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Fill the reported columns.  None of them decides a verdict."""
    for index in summary.index[summary["cell"].isin(rows["cell"].unique())]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[rows["cell"] == cell]
        spread = float(group["estimate"].std(ddof=1))
        summary.loc[index, "remainder_proxy_over_sd"] = (
            float(group["remainder_proxy"].mean()) / spread
        )
        summary.loc[index, "projected_se_ratio"] = (
            float(group["projected_std_error"].mean()) / spread
        )
        summary.loc[index, "nonconverged_fits"] = int((group["converged"] == 0).sum())
        summary.loc[index, "seconds_per_fit"] = float(group["seconds"].mean())
        if cell.startswith("q_wrong_g_glm__"):
            twin = cell.replace("q_wrong_g_glm__", "q_wrong_g_known__").replace(
                "__conservative", "__valid"
            )
            paired = paired_spread_ratio_interval(
                group,
                rows.loc[rows["cell"] == twin],
                replicates=STUDY.margins.bootstrap_replicates,
                confidence_level=STUDY.margins.confidence_level,
                seed=stream_seed(STUDY, "paired_sd_ratio", cell),
            )
            summary.loc[index, "paired_sd_ratio"] = paired.ratio
            summary.loc[index, "paired_sd_ratio_ci_lower"] = paired.interval.low
            summary.loc[index, "paired_sd_ratio_ci_upper"] = paired.interval.high


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The property summary: the shared columns, the family's verdicts, the reported columns."""
    shared = rows.loc[:, list(REPLICATE_COLUMNS)]
    summary, rates = apply_shared_verdicts(
        shared,
        STUDY,
        extra_columns=(*ONE_WRONG_COLUMNS, *REPORTED_COLUMNS),
        rate_labels=(),
    )
    rates.extend(
        one_wrong_inference_verdicts(
            summary, shared, STUDY, predicted_signs=PREDICTED_SIGNS, ladders=LADDERS
        )
    )
    summary.loc[summary["property"] == REPORTED_FAMILY, "passed"] = True
    _reported(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).
PROPERTY_RULES = (
    property_verdicts.one_wrong_rule(predicted_signs=PREDICTED_SIGNS, ladders=LADDERS),
)
