"""The prepared support rows of the long-horizon studies: cases land on rows, kinds follow gates."""

from __future__ import annotations

import pytest

from cleverly._support import SupportRow, classify_key
from cleverly._support_rows import LONGITUDINAL_ROWS
from tests import long_horizon_design as D
from tests.studies.evidence.registry import registered
from tests.studies.long_horizon_support import CASES, MEMBERS, META, ROWS, licensed_kinds
from tests.studies.support_cases import RowCase


def _id(case: RowCase) -> str:
    return f"{case.row}-{case.cell}"


def test_row_ids_are_unique_and_clear_of_the_registered_rows() -> None:
    ids = [row.id for row in ROWS]
    assert len(ids) == len(set(ids)) == 17
    assert not set(ids) & {row.id for row in LONGITUDINAL_ROWS}


def test_every_scenario_of_the_three_studies_belongs_to_exactly_one_row() -> None:
    covered = [member for members in MEMBERS.values() for member in members]
    assert len(covered) == len(set(covered))
    declared = {(study, s.name) for study in D.STUDIES for s in D.STUDIES[study].SCENARIOS}
    assert set(covered) == declared


def test_every_row_has_a_case_for_each_scenario_and_each_band() -> None:
    by_row: dict[str, list[str]] = {}
    for case in CASES:
        by_row.setdefault(case.row, []).append(case.cell)
    for row in ROWS:
        expected = [name for _, name in MEMBERS[row.id]]
        if "bootstrap_band" in row.kinds:
            expected.append("k5__simultaneous_band")
        assert by_row[row.id][: len(expected)] == expected, row.id
        # A row that no primary scenario covers stands on property-cell cases.
        assert len(by_row[row.id]) > len(expected) or expected, row.id


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_case_classifies_to_its_own_row(case: RowCase) -> None:
    observed = case.build()
    result = classify_key(observed.key, observed.learners, rows=ROWS)
    assert result.row is not None, (case.row, result.nearest)
    assert result.row.id == case.row
    assert case.slug in result.row.evidence


def test_the_declared_kinds_are_the_kinds_the_gates_license() -> None:
    for row in ROWS:
        assert row.kinds == licensed_kinds(row.id), row.id
        assert "band" not in row.kinds, row.id


def test_the_kind_rule_has_a_nonzero_witness_for_each_kind() -> None:
    """Each licensed kind is absent from some row, so a wrong rule changes a row."""
    kinds = [licensed_kinds(row.id) for row in ROWS]
    for kind in ("wald", "bootstrap_bca", "bootstrap_band"):
        assert any(kind in k for k in kinds), kind
    assert any("bootstrap_bca" not in k for k in kinds)
    assert any("bootstrap_band" not in k for k in kinds)
    bands = {row.id for row in ROWS if "bootstrap_band" in row.kinds}
    assert bands == {"X39.tmle.binary_k5", "X39.tmle.categorical_k5"}
    # The cross-fitted fits and the MSM fit run no bootstrap, so they license none.
    for row in ROWS:
        setting = META[row.id]
        if setting.folds > 1 or setting.msm:
            assert not {"bootstrap_bca", "bootstrap_band"} & row.kinds, row.id


def test_no_two_prepared_rows_match_one_fit() -> None:
    for case in CASES:
        observed = case.build()
        assert sum(r.matches(observed.key, observed.learners) for r in ROWS) == 1, case.row


def _all_registered() -> bool:
    slugs = {record.slug for record in registered()}
    return {row.evidence[0] for row in ROWS} <= slugs


def test_the_rows_move_into_the_package_once_the_studies_register() -> None:
    """The post-run gate: skips until the studies register, then fails until the move is done."""
    if not _all_registered():
        pytest.skip("the long-horizon studies have not registered: the declared run is pending")
    from cleverly._support import registered_rows

    table: tuple[SupportRow, ...] = registered_rows()
    missing = [row.id for row in ROWS if row.id not in {r.id for r in table}]
    assert not missing, (
        f"the studies are registered: move {missing} into _support_rows.py, their cases into "
        "support_cases.py, and delete tests/studies/long_horizon_support.py"
    )
