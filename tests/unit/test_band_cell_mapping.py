"""The mapping from support rows to the band cells that license their ``band`` kind.

``cleverly._support_rows.BAND_CELLS`` names, for every row that licenses ``band``, the committed
``simultaneous_coverage/<setting>__simultaneous_band`` cells whose fits fall under the row's key
(band-finite-sample plan, section 4.5). These checks tie the mapping to the rows and to the
committed artifacts: a row cannot gain the band without a cell, a mapped cell must be a positive
row of a study the row cites, and a band cell of a cited study cannot license nothing silently.
The mutation controls at the end delete or redirect one entry and require a check to fail.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import pytest

from cleverly._support import DEFAULT_BAND_OPTION, SupportRow
from cleverly._support_rows import (
    BAND_CELLS,
    BAND_OPTION_DRAWS,
    DEFAULT_BAND_DRAWS,
    LONGITUDINAL_ROWS,
    POINT_ROWS,
    UNMAPPED_BAND_CELLS,
)
from tests.studies.evidence.registry import ROOT, registered

ROWS: dict[str, SupportRow] = {row.id: row for row in (*POINT_ROWS, *LONGITUDINAL_ROWS)}
BAND_SUFFIX = "__simultaneous_band"
PHASE = "simultaneous_coverage"


def band_rows(rows: Mapping[str, SupportRow] = ROWS) -> set[str]:
    """The ids of the rows that license ``band``."""
    return {row_id for row_id, row in rows.items() if "band" in row.kinds}


@cache
def directories() -> dict[str, str]:
    """The study directory of each registered slug."""
    return {record.slug: record.artifacts.name for record in registered()}


@cache
def positive_band_cells(directory: str) -> dict[str, bool]:
    """``"<directory>:<property>/<cell>"`` to ``passed`` for each positive band row of a study."""
    record = next(r for r in registered() if r.artifacts.name == directory)
    path = record.artifacts / "properties.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            f"{directory}:{row['property']}/{row['cell']}": row["passed"] == "True"
            for row in csv.DictReader(handle)
            if row["role"] == "positive"
            and row["property"] == PHASE
            and row["cell"].endswith(BAND_SUFFIX)
        }


def unmapped_rows(mapping: Mapping[str, tuple[str, ...]]) -> set[str]:
    """Band rows with no mapped cell, and mapped ids that are not band rows."""
    return band_rows() ^ {row_id for row_id, cells in mapping.items() if cells}


def missing_cells(mapping: Mapping[str, tuple[str, ...]]) -> list[tuple[str, str]]:
    """``(row, cell)`` pairs whose cell is not a positive band row of a study the row cites."""
    missing = []
    for row_id, cells in mapping.items():
        cited = {directories()[slug] for slug in ROWS[row_id].evidence}
        for cell in cells:
            directory = cell.split(":", 1)[0]
            if directory not in cited or cell not in positive_band_cells(directory):
                missing.append((row_id, cell))
    return missing


def rows_that_move(
    passed: Mapping[str, bool], mapping: Mapping[str, tuple[str, ...]] = BAND_CELLS
) -> set[str]:
    """The PR 2 rule: a row's band option moves only when every cell mapped to it passes."""
    return {
        row_id
        for row_id, cells in mapping.items()
        if cells and all(passed.get(cell, False) for cell in cells)
    }


class TestTheMapping:
    def test_every_band_row_has_a_cell_and_every_entry_is_a_band_row(self) -> None:
        assert unmapped_rows(BAND_CELLS) == set()

    def test_every_mapped_cell_is_a_positive_band_row_of_a_cited_study(self) -> None:
        assert missing_cells(BAND_CELLS) == []

    def test_a_cell_is_written_as_directory_phase_and_setting(self) -> None:
        for row_id, cells in BAND_CELLS.items():
            assert len(set(cells)) == len(cells), row_id
            for cell in cells:
                directory, name = cell.split(":", 1)
                assert name.startswith(f"{PHASE}/") and name.endswith(BAND_SUFFIX), cell
                assert directory in directories().values(), cell

    def test_every_band_cell_of_a_cited_study_is_mapped_or_named_unmapped(self) -> None:
        cited = {directories()[slug] for row_id in band_rows() for slug in ROWS[row_id].evidence}
        mapped = {cell for cells in BAND_CELLS.values() for cell in cells}
        published = {cell for directory in cited for cell in positive_band_cells(directory)}
        assert published - mapped - set(UNMAPPED_BAND_CELLS) == set()

    def test_an_unmapped_cell_is_published_has_a_reason_and_licenses_nothing(self) -> None:
        mapped = {cell for cells in BAND_CELLS.values() for cell in cells}
        for cell, reason in UNMAPPED_BAND_CELLS.items():
            directory = cell.split(":", 1)[0]
            assert cell in positive_band_cells(directory), cell
            assert cell not in mapped, cell
            assert reason.strip().endswith("."), cell


class TestTheMoveRule:
    """The PR 2 rule, on a fixture of verdicts rather than on committed artifacts."""

    MAPPING: Mapping[str, tuple[str, ...]] = {
        "A": ("s:simultaneous_coverage/a__simultaneous_band",),
        "B": (
            "s:simultaneous_coverage/b1__simultaneous_band",
            "s:simultaneous_coverage/b2__simultaneous_band",
        ),
        "C": ("s:simultaneous_coverage/c__simultaneous_band",),
    }

    def test_a_row_moves_only_when_every_mapped_cell_passes(self) -> None:
        passed = {
            "s:simultaneous_coverage/a__simultaneous_band": True,
            "s:simultaneous_coverage/b1__simultaneous_band": True,
            "s:simultaneous_coverage/b2__simultaneous_band": False,
        }
        # C has no verdict, which never counts as a pass.
        assert rows_that_move(passed, self.MAPPING) == {"A"}

    def test_every_cell_passing_moves_every_row(self) -> None:
        passed = {cell: True for cells in self.MAPPING.values() for cell in cells}
        assert rows_that_move(passed, self.MAPPING) == set(self.MAPPING)

    def test_the_committed_verdicts_move_every_row_and_the_rows_list_the_default(self) -> None:
        """Wave 2 regenerated every mapped cell with the default band, and every one passes.

        So every band row lists the default option alone, with no option shortfall.  A red
        mapped cell after a regeneration fails here until its row keeps a shortfall.
        """
        passed = {
            cell: verdict
            for directory in set(directories().values())
            for cell, verdict in positive_band_cells(directory).items()
        }
        assert rows_that_move(passed) == set(BAND_CELLS)
        for row_id in BAND_CELLS:
            assert ROWS[row_id].band_options == (DEFAULT_BAND_OPTION,), row_id
            assert ROWS[row_id].band_option_shortfalls == (), row_id

    def test_each_row_reads_the_draws_its_cells_measured(self) -> None:
        """1,000 draws where the study pins ``N_MULTIPLIER = 1_000``; the default elsewhere."""
        plan = json.loads((ROOT / "tests" / "canonical" / "band-rerun-plan.json").read_text())
        pinned = {
            Path(entry["artifact_directory"]).name
            for entry in plan["studies"].values()
            if any("N_MULTIPLIER = 1_000" in pin for pin in entry["pinned_draws"])
        }
        assert pinned == {"drtmle_composite", "drtmle_mar_multi_arm"}
        assert plan["default_band"]["n_replicates"] == DEFAULT_BAND_DRAWS
        for row_id, cells in BAND_CELLS.items():
            measured = min(
                1_000 if cell.split(":", 1)[0] in pinned else DEFAULT_BAND_DRAWS for cell in cells
            )
            assert ROWS[row_id].band_option_draws == ((DEFAULT_BAND_OPTION, measured),), row_id
            assert BAND_OPTION_DRAWS.get(row_id, DEFAULT_BAND_DRAWS) == measured, row_id


class TestMutationControls:
    def test_deleting_one_entry_fails_the_row_check(self) -> None:
        row_id = next(iter(BAND_CELLS))
        mutated = {key: cells for key, cells in BAND_CELLS.items() if key != row_id}
        assert unmapped_rows(mutated) == {row_id}

    def test_an_entry_for_a_row_without_the_band_fails_the_row_check(self) -> None:
        unbanded = next(row_id for row_id in ROWS if row_id not in band_rows())
        mutated = {**BAND_CELLS, unbanded: next(iter(BAND_CELLS.values()))}
        assert unmapped_rows(mutated) == {unbanded}

    @pytest.mark.parametrize(
        "cell",
        [
            "longitudinal_sdr:simultaneous_coverage/missing__simultaneous_band",
            "default_bands:simultaneous_coverage/default_fit__simultaneous_band",
        ],
        ids=["missing cell", "uncited study"],
    )
    def test_pointing_an_entry_elsewhere_fails_the_cell_check(self, cell: str) -> None:
        mutated = {**BAND_CELLS, "G23.cross_fitted": (cell,)}
        assert missing_cells(mutated) == [("G23.cross_fitted", cell)]
