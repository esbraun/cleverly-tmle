"""The held-out cell of ``band-t-reference``, published as its own ``reporting`` study (Fix-2).

The strata law at n = 1,000 (plan 5.2) was not used to choose the rule or the floor: planning
draws on a disjoint stream fixed only its n.  It is declared at risk.  On planning draws about
2% of its fits read a constant arm cell, which no band rule can repair, and the predicted pass
probability of its gated reading is about 0.76 at R = 5,200.  The cell licenses no row and is
outside the sizing family, so its verdict must block nothing in ``band-t-reference``.  It
therefore lives in this companion study under ``publication_policy="reporting"``: a red verdict
is published with an owner, as a reporting study's red rows are.

The cell, its two paired controls, the diagnostic df arms and the floor-2 and floor-10 arms
read through the same row module as ``band-t-reference``
(:func:`tests.studies.evidence.simultaneous.t_band_rows`), at that study's declared R, on this
study's own seed streams.  The primary scenario is ``band-t-reference``'s shipped default fit
with ``band_reference="t"``, drawn on this study's seeds.

Declared reading (plan 5.2): green, the rule holds out of sample; red with the shown-band
reading (fits with no constant cell, ``t-band-replicates.csv``) inside (0.92, 0.98), the
shortfall is the refusal rate, reported with the measured constant-cell share; red with that
reading also outside, a finding against the rule (D5).  The floor arms move D4 only as plan 5.2
states.

Nothing here has run.  The study is not in :func:`~tests.studies.evidence.registry.registered`
until its run commits artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import band_t_reference as main_study
from tests.studies.band_t_reference import HELD_OUT_CELL, Declaration, cell_names
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.evidence.simultaneous import FAMILY
from tests.studies.point_study_helpers import primary_rows

SLUG = "band-t-reference-held-out"
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 1_000
SEED = 20263623
RESAMPLING_SEED = 20263624
SCENARIO = main_study.SCENARIO
ESTIMANDS = main_study.ESTIMANDS
PILOT_STREAM = "band-t-reference-held-out/pilot"
PILOT_PATH = ROOT / "tests" / "studies" / "band_t_reference_held_out_pilot.csv"
CELLS = (HELD_OUT_CELL,)
#: The held-out cell runs at ``band-t-reference``'s declared R; its pass probability there is
#: stated beside it (:func:`held_out_pass`).
REPLICATES = main_study.REPLICATES
SMOKE_LABELS = (HELD_OUT_CELL.label,)
PROPERTY_CELLS = {FAMILY: cell_names(HELD_OUT_CELL)}

STUDY = StudyRecord(
    name="held-out cell of the per-estimate t reference band",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "band_t_reference_held_out",
    document="docs/technical-reference/method-evidence/band-t-reference.md",
    anchor="band-t-reference-held-out",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-band-t-reference-held-out",
    reference=None,
    failure_bound={"cleverly-band-t-reference-held-out": 0.01},
    modules=tuple(
        sorted(
            {
                *main_study.MODULES,
                "tests/studies/band_t_reference_held_out.py",
                "tests/studies/band_t_reference_held_out_properties.py",
            }
        )
    ),
    runner_module="tests.studies.band_t_reference_held_out",
    properties_module="tests.studies.band_t_reference_held_out_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    property_artifacts=("t-band-replicates.csv",),
)

DECLARATION = Declaration(
    record=STUDY,
    cells=CELLS,
    replicates=REPLICATES,
    pilot_stream=PILOT_STREAM,
    pilot_path=PILOT_PATH,
    smoke_labels=SMOKE_LABELS,
)

CONFIGURATION = {
    "primary_fit": main_study.CONFIGURATION["primary_fit"],
    "primary_law": main_study.CONFIGURATION["primary_law"],
    "band_rule": main_study.CONFIGURATION["band_rule"],
    "cell": {"source": HELD_OUT_CELL.source, "n": HELD_OUT_CELL.n, "rows": ()},
    "replicates": f"band-t-reference's declared R ({REPLICATES} before the pilot)",
}


def held_out_pass(replicates: int, coverage: float, pilot_replicates: int) -> float:
    """The held-out cell's unpooled pass probability at ``replicates``, stated beside it."""
    return main_study.family_pass([coverage], [pilot_replicates], replicates, pooled=False)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample of the canonical binary law from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return main_study.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one replication, under this study's implementation name."""
    result = main_study.fit_cleverly(frame)
    if result.simultaneous is None or result.simultaneous.reference != "t":
        raise AssertionError("the primary fit did not build a t band")
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=SCENARIO,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    _, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(SCENARIO, index, n) for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])
