"""Repeated-sampling accuracy of complete-data DR-TMLE on a declared known mechanism.

One family, fixed before any registered run.  On one draw of the two-arm law of
:mod:`tests.studies.known_mechanism_law` per replication, n = 2,000 and 1,000 replications,
the ATE is fitted at each of the four guards with the wrong outcome regression.  Each cell
answers to the accuracy rule of :mod:`tests.studies.known_mechanism_properties`: the bias
margin, the coverage floor and the SE-ratio sanity band together.  A replication that raises
is never redrawn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.canonical_known_mechanism_drtmle import STUDY, fit_cleverly
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

ACCURACY = "known_mechanism_accuracy"
ACCURACY_N = 2_000
ACCURACY_REPLICATES = 1_000
#: Cell suffix -> the guard it fits.
GUARD_CELLS: dict[str, tuple[str, ...]] = {
    "none": (),
    "Q": ("Q",),
    "g": ("g",),
    "Qg": ("Q", "g"),
}
TRUTH = law.truth(2)["ate"]


def _draw(replicate: int) -> Any:
    seed = stream_seed(STUDY, "property_sample", ACCURACY, "paired", replicate)
    return law.sample(ACCURACY_N, seed)


def _replication(replicate: int, frame: Any) -> list[dict[str, Any]]:
    rows = []
    for suffix, guard in GUARD_CELLS.items():
        result = fit_cleverly(frame, guard)
        rows.append(
            replicate_row(
                property_name=ACCURACY,
                cell=f"ate__drtmle_known__{suffix}",
                role="positive",
                replicate=replicate,
                n=ACCURACY_N,
                requested=ACCURACY_REPLICATES,
                truth=TRUTH,
                estimate=result["ate"],
                alpha=STUDY.margins.alpha,
            )
        )
    return rows


def _cells_of(key: tuple[int]) -> list[str]:
    """Every guard cell reads the same draw, so a key fills all of them."""
    return [f"{ACCURACY}/ate__drtmle_known__{suffix}" for suffix in GUARD_CELLS]


def _cell_of(key: tuple[int]) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: tuple[int]) -> int:
    return key[0]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries.

    Not named ``declared_cells``: the shared evidence tests read that name as a tuple of
    :class:`~tests.studies.evidence.properties.PropertyCell` objects with a ``dgp``.
    """
    return tuple((ACCURACY, f"ate__drtmle_known__{suffix}", "ate", TRUTH) for suffix in GUARD_CELLS)


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    replicates = ACCURACY_REPLICATES if budget is None else min(budget, ACCURACY_REPLICATES)
    outcomes = fit_properties(
        [(r,) for r in range(replicates)],
        splat_drawn(_replication),
        cells_of=_cells_of,
        draw=splat_draw(_draw),
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    if budget is not None:
        # A capped smoke run requests what it runs, so the summary reads it as complete.
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    margins = STUDY.margins
    family = summary["property"] == ACCURACY
    summary.loc[family, "passed"] = (
        summary.loc[family, "bias_equivalent"].astype(bool)
        & (summary.loc[family, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[family, "se_ratio"].between(*margins.se_ratio_sanity)
    )
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  Every cell is positive.
PROPERTY_RULES = (property_verdicts.accuracy_rule(ACCURACY, shared="summarize_properties"),)
