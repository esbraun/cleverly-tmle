"""The support rows of the long-horizon studies: cases land on rows, kinds follow the gates.

The rows ``G46`` live in ``cleverly._support_rows``, and their cases in
``tests/studies/support_cases.py``. ``tests/unit/test_support_rows.py`` classifies every case onto
its own row. This module checks what that module cannot: that the three studies' scenarios and
gated rows license exactly the kinds a row declares.

**Kinds.**  Every row licenses ``point``. It licenses ``wald`` when the study gates a Wald row of
a member scenario or an ``interval_calibration`` cell of the row's setting. It licenses
``bootstrap_bca`` when the study gates a BCa row of a member scenario. The design gates BCa only
where the Wald row is not (``n < 2 n_kappa``). A row licenses ``band`` when the study gates the
Wald-curve band of its setting, and ``bootstrap_band`` when it gates the studentized band.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cleverly._support import Kind, SupportRow, registered_rows
from cleverly._support_rows import LONGITUDINAL_ROWS
from tests import long_horizon_design as D
from tests.studies import long_horizon_primary as primary
from tests.studies.evidence.registry import registered
from tests.studies.support_cases import CASES, LONG_HORIZON_MEMBERS, RowCase

ROWS = tuple(row for row in LONGITUDINAL_ROWS if row.id.startswith("G46."))
OTHER_ROWS = tuple(row for row in LONGITUDINAL_ROWS if not row.id.startswith("G46."))
LONG_CASES = tuple(case for case in CASES if case.row.startswith("G46."))
STUDY_OF_SLUG = {
    "long-horizon-ltmle": "S1",
    "long-horizon-categorical-ltmle": "S2",
    "long-horizon-sdr": "S3",
}


@dataclass(frozen=True)
class Setting:
    """The estimator setting of one row, read from its key."""

    study: str
    K: int
    m: int
    folds: int
    msm: bool
    learned: bool


def setting(row: SupportRow) -> Setting:
    """Read the setting from the key the row declares."""
    key = row.key
    return Setting(
        study=STUDY_OF_SLUG[row.evidence[0]],
        K=len(key.treatment),
        m=int(key.treatment[0]),
        folds=5 if key.splitting == ("cross_fitted",) else 1,
        msm="msm" in key.family,
        learned="A1" not in key.qualifiers,
    )


def licensed_kinds(row: SupportRow) -> frozenset[Kind]:
    """The kinds the committed gate lists license for ``row``."""
    s = setting(row)
    members = {scenario for _, scenario in LONG_HORIZON_MEMBERS[row.id]}
    gated = [(name, estimand) for study, name, estimand in D.GATED_ROWS if study == s.study]
    in_members = [(name, estimand) for name, estimand in gated if name in members]
    kinds: set[Kind] = {"point"}
    calibrated = not s.learned and any(
        cell.study == s.study
        and (cell.quantity.K, cell.quantity.m, cell.folds, cell.msm) == (s.K, s.m, s.folds, s.msm)
        for cell in D.calibration_cells()
    )
    if calibrated or any(not estimand.endswith(primary.BCA_SUFFIX) for _, estimand in in_members):
        kinds.add("wald")
    if any(estimand.endswith(primary.BCA_SUFFIX) for _, estimand in in_members):
        kinds.add("bootstrap_bca")
    if (
        s.study in D.BAND_STUDIES
        and s.folds == 1
        and not s.msm
        and not s.learned
        and s.K == 5
        and D.band_config(s.study)[0].m == s.m
    ):
        kinds.add("bootstrap_band" if s.study in D.BOOT_BAND_STUDIES else "band")
    return frozenset(kinds)


def _id(case: RowCase) -> str:
    return f"{case.row}-{case.cell}"


def test_row_ids_are_unique_and_clear_of_the_other_rows() -> None:
    ids = [row.id for row in ROWS]
    assert len(ids) == len(set(ids)) == 17
    assert not set(ids) & {row.id for row in OTHER_ROWS}
    assert {row.id for row in ROWS} == set(LONG_HORIZON_MEMBERS)


def test_every_scenario_of_the_three_studies_belongs_to_exactly_one_row() -> None:
    covered = [member for members in LONG_HORIZON_MEMBERS.values() for member in members]
    assert len(covered) == len(set(covered))
    declared = {(study, s.name) for study in D.STUDIES for s in D.STUDIES[study].SCENARIOS}
    assert set(covered) == declared


def test_every_row_has_a_case_for_each_scenario_and_each_band() -> None:
    by_row: dict[str, list[str]] = {}
    for case in LONG_CASES:
        by_row.setdefault(case.row, []).append(case.cell)
    for row in ROWS:
        expected = [name for _, name in LONG_HORIZON_MEMBERS[row.id]]
        if row.kinds & {"band", "bootstrap_band"}:
            expected.append("k5__simultaneous_band")
        assert by_row[row.id][: len(expected)] == expected, row.id
        # A row that no primary scenario covers stands on property-cell cases.
        assert len(by_row[row.id]) > len(expected) or expected, row.id


def test_the_declared_kinds_are_the_kinds_the_gates_license() -> None:
    for row in ROWS:
        assert row.kinds == licensed_kinds(row), row.id
        # S2 gates the bootstrap band and reports the Wald-curve band; S1 gates the Wald-curve one.
        assert ("band" in row.kinds) == (row.id == "G46.tmle.binary_k5"), row.id


def test_the_kind_rule_has_a_nonzero_witness_for_each_kind() -> None:
    """Each licensed kind is absent from some row, so a wrong rule changes a row."""
    kinds = [licensed_kinds(row) for row in ROWS]
    for kind in ("wald", "bootstrap_bca", "band", "bootstrap_band"):
        assert any(kind in k for k in kinds), kind
    assert any("bootstrap_bca" not in k for k in kinds)
    assert any("bootstrap_band" not in k for k in kinds)
    assert any("band" not in k for k in kinds)
    assert {row.id for row in ROWS if "band" in row.kinds} == {"G46.tmle.binary_k5"}
    assert {row.id for row in ROWS if "bootstrap_band" in row.kinds} == {"G46.tmle.categorical_k5"}
    # The cross-fitted fits and the MSM fit run no bootstrap, so they license none.
    for row in ROWS:
        s = setting(row)
        if s.folds > 1 or s.msm:
            assert not {"bootstrap_bca", "bootstrap_band"} & row.kinds, row.id
    # The MSM row gates both Wald coefficients (n = 96,000).
    row = next(row for row in ROWS if row.id == "G46.tmle.binary_k5_msm")
    assert "wald" in licensed_kinds(row)


@pytest.mark.parametrize("case", LONG_CASES, ids=_id)
def test_no_other_row_matches_the_fit_of_a_case(case: RowCase) -> None:
    observed = case.build()
    assert sum(r.matches(observed.key, observed.learners) for r in LONGITUDINAL_ROWS) == 1  # type: ignore[union-attr]


def test_the_rows_are_in_the_package_table_and_their_studies_are_registered() -> None:
    table = {row.id for row in registered_rows()}
    assert {row.id for row in ROWS} <= table
    slugs = {record.slug for record in registered()}
    assert {row.evidence[0] for row in ROWS} <= slugs
