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
import re
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
    SHORTFALLS,
    T_BAND_CELLS,
    UNMAPPED_BAND_CELLS,
)
from tests.studies.evidence.registry import ROOT, registered

ROWS: dict[str, SupportRow] = {row.id: row for row in (*POINT_ROWS, *LONGITUDINAL_ROWS)}
BAND_SUFFIX = "__simultaneous_band"
PHASE = "simultaneous_coverage"
T_OPTION = ("mammen", "logit", "t")
T_SUFFIX = "__t_band"
T_STUDY = "band_t_reference"
T_SLUG = "band-t-reference"


def band_rows(rows: Mapping[str, SupportRow] = ROWS) -> set[str]:
    """The ids of the rows that license ``band``."""
    return {row_id for row_id, row in rows.items() if "band" in row.kinds}


@cache
def directories() -> dict[str, str]:
    """The study directory of each registered slug."""
    return {record.slug: record.artifacts.name for record in registered()}


@cache
def property_rows(directory: str) -> tuple[dict[str, str], ...]:
    """The committed ``properties.csv`` rows of a registered study, by its directory."""
    record = next(r for r in registered() if r.artifacts.name == directory)
    path = record.artifacts / "properties.csv"
    if not path.is_file():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


@cache
def positive_band_cells(directory: str, suffix: str = BAND_SUFFIX) -> dict[str, bool]:
    """``"<directory>:<property>/<cell>"`` to ``passed`` for each positive band row of a study.

    ``suffix`` is ``"__simultaneous_band"`` for a default band cell and ``"__t_band"`` for a
    ``band-t-reference`` cell.
    """
    return {
        f"{directory}:{row['property']}/{row['cell']}": row["passed"] == "True"
        for row in property_rows(directory)
        if row["role"] == "positive" and row["property"] == PHASE and row["cell"].endswith(suffix)
    }


def unmapped_rows(
    mapping: Mapping[str, tuple[str, ...]],
    t_mapping: Mapping[str, tuple[str, ...]] = T_BAND_CELLS,
) -> set[str]:
    """Band rows with no mapped cell, and mapped ids that are not band rows.

    A row's ``band`` kind may rest on :data:`T_BAND_CELLS` alone (Fix-2 plan, test 9); the
    ``t`` band checks below hold those entries to their own rule.
    """
    mapped = {row_id for table in (mapping, t_mapping) for row_id, cells in table.items() if cells}
    return band_rows() ^ mapped


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
        mapped = {
            cell for table in (BAND_CELLS, T_BAND_CELLS) for c in table.values() for cell in c
        }
        for cell, reason in UNMAPPED_BAND_CELLS.items():
            directory = cell.split(":", 1)[0]
            suffix = T_SUFFIX if cell.endswith(T_SUFFIX) else BAND_SUFFIX
            assert cell in positive_band_cells(directory, suffix), cell
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

        So every row in :data:`BAND_CELLS` lists the default option, with no option shortfall,
        and lists the ``t`` triple besides only when :data:`T_BAND_CELLS` maps it (Fix-2 plan,
        test 9).  A red mapped cell after a regeneration fails here until its row keeps a
        shortfall.
        """
        passed = {
            cell: verdict
            for directory in set(directories().values())
            for cell, verdict in positive_band_cells(directory).items()
        }
        assert rows_that_move(passed) == set(BAND_CELLS)
        for row_id in BAND_CELLS:
            listed = (DEFAULT_BAND_OPTION, *((T_OPTION,) if row_id in T_BAND_CELLS else ()))
            assert ROWS[row_id].band_options == listed, row_id
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
            t_draws = ((T_OPTION, DEFAULT_BAND_DRAWS),) if row_id in T_BAND_CELLS else ()
            assert ROWS[row_id].band_option_draws == (
                (DEFAULT_BAND_OPTION, measured),
                *t_draws,
            ), row_id
            assert BAND_OPTION_DRAWS.get(row_id, DEFAULT_BAND_DRAWS) == measured, row_id
        # Every band-t-reference cell refits its shape with the default 10,000 draws.
        for row_id in set(T_BAND_CELLS) - set(BAND_CELLS):
            assert ROWS[row_id].band_option_draws == ((T_OPTION, DEFAULT_BAND_DRAWS),), row_id


class TestMutationControls:
    def test_deleting_one_entry_fails_the_row_check(self) -> None:
        row_id = next(iter(BAND_CELLS))
        mutated = {key: cells for key, cells in BAND_CELLS.items() if key != row_id}
        assert unmapped_rows(mutated) == {row_id}

    def test_deleting_a_t_only_entry_fails_the_row_check(self) -> None:
        row_id = next(key for key in T_BAND_CELLS if key not in BAND_CELLS)
        mutated = {key: cells for key, cells in T_BAND_CELLS.items() if key != row_id}
        assert unmapped_rows(BAND_CELLS, mutated) == {row_id}

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


# ------------------------------------------------------------------ the t band (Fix-2 plan 4.9)


def t_band_rows(rows: Mapping[str, SupportRow] = ROWS) -> set[str]:
    """The ids of the rows that list the ``t`` triple."""
    return {row_id for row_id, row in rows.items() if T_OPTION in row.band_options}


def t_mapping_problems(
    mapping: Mapping[str, tuple[str, ...]],
    rows: Mapping[str, SupportRow],
    passed: Mapping[str, bool],
    controls_passed: Mapping[str, bool],
) -> list[str]:
    """Every way ``mapping`` breaks the licence rule, as short sentences.

    A row lists the triple exactly when it has an entry; each cell is written by ``_t_band``,
    is a positive ``__t_band`` row of band-t-reference that passed, and its gated paired
    control passed (the pointwise control; the reference gain is reported and passes, plan rev
    3.3); a row only in this mapping lists the ``t`` triple and not the default; a ``t`` row
    licenses ``band`` and cites band-t-reference.
    """
    problems = []
    for row_id in t_band_rows(rows) ^ {row_id for row_id, cells in mapping.items() if cells}:
        problems.append(f"{row_id}: lists the t triple without cells, or cells without it")
    for row_id, cells in mapping.items():
        row = rows[row_id]
        if "band" not in row.kinds:
            problems.append(f"{row_id}: maps t cells but licenses no band")
        if T_SLUG not in row.evidence:
            problems.append(f"{row_id}: maps t cells but does not cite {T_SLUG}")
        if row_id not in BAND_CELLS and DEFAULT_BAND_OPTION in row.band_options:
            problems.append(f"{row_id}: only t cells, but lists the default band")
        for cell in cells:
            directory, name = cell.split(":", 1)
            if directory != T_STUDY or not (
                name.startswith(f"{PHASE}/") and name.endswith(T_SUFFIX)
            ):
                problems.append(f"{row_id}: {cell} is not written by _t_band")
            if not passed.get(cell, False):
                problems.append(f"{row_id}: {cell} did not pass")
            if not controls_passed.get(cell, False):
                problems.append(f"{row_id}: a gated control of {cell} did not pass")
    return problems


def t_controls_passed(directory: str = T_STUDY) -> dict[str, bool]:
    """Each positive ``t`` cell to whether every paired control of its label passed."""
    rows = property_rows(directory)
    out = {}
    for cell in positive_band_cells(directory, T_SUFFIX):
        label = cell.split("/", 1)[1].removesuffix(T_SUFFIX)
        controls = [
            row
            for row in rows
            if row["property"] == PHASE
            and row["cell"].startswith(f"{label}__")
            and row["role"] in {"control", "reference_control"}
        ]
        out[cell] = bool(controls) and all(row["passed"] == "True" for row in controls)
    return out


class TestTheTBandMapping:
    def test_the_committed_mapping_and_rows_agree(self) -> None:
        passed = positive_band_cells(T_STUDY, T_SUFFIX)
        assert t_mapping_problems(T_BAND_CELLS, ROWS, passed, t_controls_passed()) == []
        assert t_band_rows() == set(T_BAND_CELLS)

    def test_the_mapping_is_the_rows_the_study_declared(self) -> None:
        """Each licensing cell's declared rows (``band_t_reference.CELLS``) are its entries."""
        from tests.studies.band_t_reference import CELLS

        declared: dict[str, set[str]] = {}
        for cell in CELLS:
            for row_id in cell.rows:
                declared.setdefault(row_id, set()).add(f"{T_STUDY}:{PHASE}/{cell.label}{T_SUFFIX}")
        assert {row_id: set(cells) for row_id, cells in T_BAND_CELLS.items()} == declared

    def test_every_positive_t_cell_is_mapped_or_named_unmapped(self) -> None:
        mapped = {cell for cells in T_BAND_CELLS.values() for cell in cells}
        for directory in (T_STUDY, "band_t_reference_held_out"):
            published = set(positive_band_cells(directory, T_SUFFIX))
            assert published, directory
            assert published - mapped - set(UNMAPPED_BAND_CELLS) == set(), directory

    def test_the_held_out_cell_licenses_no_row(self) -> None:
        cell = "band_t_reference_held_out:simultaneous_coverage/strata_n1000__t_band"
        assert cell in UNMAPPED_BAND_CELLS
        assert all(cell not in cells for cells in T_BAND_CELLS.values())

    def test_a_t_band_cell_cannot_sit_in_the_default_mapping(self) -> None:
        for cells in BAND_CELLS.values():
            for cell in cells:
                assert not cell.endswith(T_SUFFIX), cell

    def test_t14_licenses_the_default_band_from_the_committed_cross_fitted_cell(self) -> None:
        cell = "tmle3_stratified:simultaneous_coverage/crossfit_strata__simultaneous_band"
        assert BAND_CELLS["T14"] == (cell,)
        assert positive_band_cells("tmle3_stratified")[cell] is True
        unmapped = "tmle3_stratified:simultaneous_coverage/strata__simultaneous_band"
        assert positive_band_cells("tmle3_stratified")[unmapped] is False
        assert unmapped in UNMAPPED_BAND_CELLS


class TestTheTBandMutationControls:
    """The committed mapping and rows, with every mapped cell and control passing.

    Each control changes one thing about T30, which holds two cells, and the check must fail.
    """

    CELLS = (
        f"{T_STUDY}:{PHASE}/cde_z0{T_SUFFIX}",
        f"{T_STUDY}:{PHASE}/cde_z0_n8000{T_SUFFIX}",
    )

    def fixture(self) -> tuple[dict, dict, dict, dict]:
        mapping = dict(T_BAND_CELLS)
        passed = {cell: True for cells in mapping.values() for cell in cells}
        return mapping, dict(ROWS), passed, dict(passed)

    def test_the_fixture_is_clean(self) -> None:
        mapping, _, _, _ = self.fixture()
        assert mapping["T30"] == self.CELLS
        assert t_mapping_problems(*self.fixture()) == []

    def test_deleting_an_entry_fails(self) -> None:
        mapping, rows, passed, controls = self.fixture()
        del mapping["T30"]
        assert t_mapping_problems(mapping, rows, passed, controls) != []

    def test_redirecting_an_entry_fails(self) -> None:
        mapping, rows, passed, controls = self.fixture()
        mapping["T30"] = (f"{T_STUDY}:{PHASE}/cde_z0__simultaneous_band",)
        assert t_mapping_problems(mapping, rows, passed, controls) != []

    def test_one_failed_cell_fails(self) -> None:
        mapping, rows, passed, controls = self.fixture()
        passed[self.CELLS[0]] = False
        assert t_mapping_problems(mapping, rows, passed, controls) != []

    def test_one_failed_control_clause_fails(self) -> None:
        mapping, rows, passed, controls = self.fixture()
        controls[self.CELLS[1]] = False
        assert t_mapping_problems(mapping, rows, passed, controls) != []

    def test_listing_the_default_on_a_t_only_row_fails(self) -> None:
        from dataclasses import replace

        mapping, rows, passed, controls = self.fixture()
        rows["T30"] = replace(rows["T30"], band_options=(DEFAULT_BAND_OPTION, T_OPTION))
        assert t_mapping_problems(mapping, rows, passed, controls) != []

    def test_a_t_row_that_does_not_cite_the_study_fails(self) -> None:
        from dataclasses import replace

        mapping, rows, passed, controls = self.fixture()
        evidence = tuple(slug for slug in rows["T30"].evidence if slug != T_SLUG)
        rows["T30"] = replace(rows["T30"], evidence=evidence)
        assert t_mapping_problems(mapping, rows, passed, controls) != []


def band_shortfall_problems(
    shortfalls: Mapping[tuple[str, str], str] = SHORTFALLS,
    unmapped: Mapping[str, str] = UNMAPPED_BAND_CELLS,
) -> list[str]:
    """The band shortfalls that are not a default-band reading.

    ``band_support`` lets a band shortfall bind the default triple only, so every
    ``(row, "band")`` sentence must state a red reading of the default band, and the row must
    name a default-band cell in ``UNMAPPED_BAND_CELLS`` that carries the reading.
    """
    problems = []
    for (row_id, kind), text in shortfalls.items():
        if kind != "band":
            continue
        if "default band" not in text:
            problems.append(f"{row_id}: the shortfall does not say 'default band'")
        named = re.compile(rf"\brow {re.escape(row_id)}\b")
        cells = [
            cell
            for cell, reason in unmapped.items()
            if cell.endswith(BAND_SUFFIX)
            and named.search(reason)
            and "default band" in reason
            and "band shortfall" in reason
        ]
        if not cells:
            problems.append(f"{row_id}: no unmapped default-band cell carries the shortfall")
    return problems


class TestBandShortfallsReadTheDefaultBand:
    def test_every_band_shortfall_is_a_default_band_reading(self) -> None:
        assert [key for key in SHORTFALLS if key[1] == "band"] != []
        assert band_shortfall_problems() == []

    def test_a_shortfall_that_does_not_name_the_default_band_fails(self) -> None:
        shortfalls = dict(SHORTFALLS)
        shortfalls[("T13", "band")] = "Row T13 is red for its band: joint coverage 0.92 to 0.94."
        assert band_shortfall_problems(shortfalls) == [
            "T13: the shortfall does not say 'default band'"
        ]

    def test_a_shortfall_without_its_default_band_cell_fails(self) -> None:
        unmapped = {
            cell: reason
            for cell, reason in UNMAPPED_BAND_CELLS.items()
            if not re.search(r"\brow T13\b", reason)
        }
        assert band_shortfall_problems(unmapped=unmapped) == [
            "T13: no unmapped default-band cell carries the shortfall"
        ]

    def test_a_t_band_cell_does_not_carry_a_shortfall(self) -> None:
        unmapped = {
            cell.replace(BAND_SUFFIX, T_SUFFIX) if re.search(r"\brow T30\b", reason) else cell: (
                reason
            )
            for cell, reason in UNMAPPED_BAND_CELLS.items()
        }
        assert band_shortfall_problems(unmapped=unmapped) == [
            "T30: no unmapped default-band cell carries the shortfall"
        ]
