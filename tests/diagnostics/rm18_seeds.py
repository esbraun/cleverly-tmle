"""The declared fresh seeds of every RM18 follow-up part, in the declared assignment order.

RM18 of ``docs/roadmap.md`` judges a fresh seed against every seed that an RM18 part draws on
the same registered record, in the order its "seed assignment order" table gives.  One part's
seeds therefore depend on the parts before it, so the assignment is computed here, once, for
every part, at every part's declared budget.  A smoke run takes a prefix of the same seeds.

The parts import this module; it imports them inside :func:`weighted_seeds` alone, to read their
labels, because their labels live beside the rest of their declarations.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from tests.diagnostics import rm18_shared as shared
from tests.studies import drtmle_properties as binary
from tests.studies import multi_arm_ctmle_selector_properties as selector
from tests.studies import multi_arm_drtmle_properties as multi
from tests.studies.canonical_drtmle import STUDY as BINARY
from tests.studies.canonical_multi_arm_ctmle_selector import STUDY as SELECTOR
from tests.studies.canonical_multi_arm_drtmle import STUDY as MULTI
from tests.studies.canonical_weighted_ltmle import STUDY as ORDINARY
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.property_verdicts import CONTRACTION_FAMILY
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: BD-P: the throwaway pilot record and the step 2 record, with their fixed budgets.
PILOT_RECORD = replace(CROSSFIT, seed=stream_seed(CROSSFIT, "rm18", "boundary-pilot"))
PAIRED_RECORD = replace(CROSSFIT, seed=stream_seed(CROSSFIT, "rm18", "boundary-paired"))
PILOT_REPLICATES = 800
BUDGET_CAP = 20_000

#: The outer-rung budget Design SL declares, and its rungs, which the multi-arm set holds.
SL_OUTER_REPLICATES = 73_000
SL_OUTER_SIZES = (2_000, 8_000)


def primary_seeds(record: StudyRecord, replicates: int) -> list[int]:
    """The primary sample seeds of ``replicates`` draws of ``record``, in replicate order."""
    return [
        replicate_seed(record, scenario, index)
        for scenario in record.scenarios
        for index in range(replicates)
    ]


def paired_seeds() -> tuple[list[int], list[int]]:
    """The BD-P pilot seeds and the step 2 seeds up to the cap, fixed by their records.

    A BD-P seed that equals a registered seed, or repeats another BD-P seed, stops the part.
    """
    pilot = primary_seeds(PILOT_RECORD, PILOT_REPLICATES)
    paired = primary_seeds(PAIRED_RECORD, BUDGET_CAP)
    registered = shared.weighted_registered_seeds(CROSSFIT)
    everything = [*pilot, *paired]
    if len(set(everything)) != len(everything) or not registered.isdisjoint(everything):
        raise RuntimeError("a BD-P seed repeats a registered or another BD-P seed")
    return pilot, paired


@functools.cache
def weighted_seeds() -> dict[str, tuple[int, ...]]:
    """The fresh seeds of FW-A and BD-3 on the cross-fitted record, and of OW-A, OW-B and OW-C
    on the ordinary one, each in its label order."""
    from tests.diagnostics.rm18_boundary import run as bd
    from tests.diagnostics.rm18_fixed_weights import run as fw
    from tests.diagnostics.rm18_ordinary_weighted import run as ow

    out: dict[str, tuple[int, ...]] = {}
    pilot, paired = paired_seeds()
    taken = shared.weighted_registered_seeds(CROSSFIT) | set(pilot) | set(paired)
    for part, labels in (("FW-A", fw.labels()), ("BD-3", bd.control_labels())):
        out[part] = tuple(shared.fresh_seeds(CROSSFIT, labels, taken))
    taken = shared.weighted_registered_seeds(ORDINARY)
    for part, labels in (
        ("OW-A", ow.calibration_labels()),
        ("OW-B", ow.null_labels()),
        ("OW-C", ow.family_labels()),
    ):
        out[part] = tuple(shared.fresh_seeds(ORDINARY, labels, taken))
    return out


# ------------------------------------------------------------------- the coverage studies

#: The coverage-study cells of BD-1 and BD-2, by record, in the declared assignment order:
#: the part, the property, the cell and the declared budget.
COVERAGE_CELLS: dict[str, tuple[tuple[str, str, str, int], ...]] = {
    MULTI.slug: (
        ("BD-1", "double_robust_contraction", "outcome_correct_n4000", 6_000),
        ("BD-1", "root_n_and_efficiency", "n_500", 6_000),
        ("BD-2", "interval_calibration", "correctly_specified", 17_000),
    ),
    SELECTOR.slug: (("BD-1", "root_n_and_efficiency", "n_500", 6_000),),
}
COVERAGE_RECORDS = {MULTI.slug: (MULTI, multi), SELECTOR.slug: (SELECTOR, selector)}


def _states(seed: int, replicates: int) -> list[int]:
    return [int(state) for state in np.random.SeedSequence(seed).generate_state(replicates)]


def coverage_registered(record: StudyRecord, module: Any) -> set[int]:
    """Every replicate seed a registered coverage-study module draws, SL's budget included."""
    states = set(primary_seeds(record, record.replicates))
    for cell in module.cells():
        count = cell.replicates
        if module is multi and cell.property == CONTRACTION_FAMILY and cell.n in SL_OUTER_SIZES:
            count = max(count, SL_OUTER_REPLICATES)
        states.update(_states(cell.seed, count))
    return states


@functools.cache
def coverage_seeds() -> dict[tuple[str, str, str], int]:
    """The ``CoverageStudy`` seed of each re-read multi-arm or selector cell.

    A seed moves as a whole, to the ``"retry"`` label, when any of its replicate seeds is taken
    or repeats another.  Each assigned cell's replicate seeds join the taken set.
    """
    out: dict[tuple[str, str, str], int] = {}
    for slug, cells in COVERAGE_CELLS.items():
        record, module = COVERAGE_RECORDS[slug]
        taken = coverage_registered(record, module)
        for _, property_name, cell, declared in cells:
            labels = ("rm18", "boundary", property_name, cell)
            seed, counter = stream_seed(record, *labels), 0
            while True:
                states = _states(seed, declared)
                if len(set(states)) == declared and taken.isdisjoint(states):
                    break
                counter += 1
                seed = stream_seed(record, *labels, "retry", counter)
            taken.update(states)
            out[slug, property_name, cell] = seed
    return out


# --------------------------------------------------------------------------- the binary rung

BINARY_CELL = ("double_robust_contraction", "treatment_correct_n1500")
BINARY_REPLICATES = 6_000


def cells(part: str) -> list[tuple[StudyRecord, Any, str, str, int]]:
    """The re-read cells of BD-1 or BD-2: record, property module, property, cell and budget."""
    out = [
        (*COVERAGE_RECORDS[slug], property_name, cell, declared)
        for slug, entries in COVERAGE_CELLS.items()
        for owner, property_name, cell, declared in entries
        if owner == part
    ]
    if part == "BD-1":
        out.append((BINARY, binary, *BINARY_CELL, BINARY_REPLICATES))
    return out


def binary_registered() -> set[int]:
    seeds = set(primary_seeds(BINARY, BINARY.replicates))
    seeds.update(
        stream_seed(BINARY, "property", cell.property, cell.cell, str(index + cell.seed_offset))
        for cell in binary.cells()
        for index in range(cell.replicates)
    )
    return seeds


def binary_labels() -> list[tuple[Any, ...]]:
    return [("rm18", "boundary", *BINARY_CELL, index) for index in range(BINARY_REPLICATES)]


@functools.cache
def binary_seeds() -> tuple[int, ...]:
    return tuple(shared.fresh_seeds(BINARY, binary_labels(), binary_registered()))


def every_part() -> dict[str, dict[str, Sequence[int]]]:
    """Every fresh sample seed of every part, by record and part, for the disjointness test.

    A coverage cell contributes its replicate seeds.  The collision rule reads one record at a
    time, so two parts on different records may share a seed.
    """
    out: dict[str, dict[str, Sequence[int]]] = {}
    for part, seeds in weighted_seeds().items():
        record = CROSSFIT.slug if part in {"FW-A", "BD-3"} else ORDINARY.slug
        out.setdefault(record, {})[part] = seeds
    pilot, paired = paired_seeds()
    out[CROSSFIT.slug].update({"BD-P pilot": pilot, "BD-P step 2": paired})
    out[BINARY.slug] = {"BD-1": binary_seeds()}
    declared = {(slug, p, c): r for slug, cells in COVERAGE_CELLS.items() for _, p, c, r in cells}
    for (slug, property_name, cell), seed in coverage_seeds().items():
        out.setdefault(slug, {})[f"{property_name}/{cell}"] = _states(
            seed, declared[slug, property_name, cell]
        )
    return out
