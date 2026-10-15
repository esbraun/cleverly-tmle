"""The point-treatment support rows and the cases that stand for them.

Each row of ``cleverly._support_rows`` names the studies that cover it. Each case of
``tests/studies/support_cases.py`` builds one fit of that configuration, and the case must
classify onto its own row. A case that lands on another row fails, so a row cannot pass on
the evidence of a different study. The controls at the end change one setting of a case and
require the classification to move.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from cleverly._support import (
    KEY_AXES,
    SupportRow,
    classify,
    classify_key,
    point_key,
    registered_rows,
)
from cleverly._support_rows import POINT_ROWS, merge_rows, rows_over
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.support_cases import CASES, RowCase, _build, frame

ROWS = {row.id: row for row in POINT_ROWS}
KINDS = {"point", "wald", "wald_t", "band", "bootstrap", "derived", "test"}


def classified(case: RowCase) -> Any:
    estimator, data, estimands = case.build()
    return classify(estimator, data, estimands)


def case_id(case: RowCase) -> str:
    return f"{case.row}-{case.slug}-{case.cell}"


def non_family(row: SupportRow) -> tuple[Any, ...]:
    axes = row.key.axes()
    return tuple(axes[name] for name in KEY_AXES if name != "family")


class TestRegistry:
    def test_the_registered_rows_are_the_point_rows(self) -> None:
        assert registered_rows() == POINT_ROWS

    def test_row_ids_are_unique(self) -> None:
        assert len(ROWS) == len(POINT_ROWS)

    def test_every_row_has_a_case_and_every_case_names_a_row(self) -> None:
        with_cases = {case.row for case in CASES}
        assert set(ROWS) - with_cases == set()
        assert with_cases - set(ROWS) == set()

    def test_no_two_rows_cover_the_same_fit(self) -> None:
        """Two rows with equal axes and a shared family would both match one fit."""
        for index, first in enumerate(POINT_ROWS):
            for second in POINT_ROWS[index + 1 :]:
                overlap = non_family(first) == non_family(second) and (
                    first.key.family & second.key.family
                )
                assert not overlap, (first.id, second.id)

    def test_every_row_licenses_the_point_estimate_and_known_kinds(self) -> None:
        for row in POINT_ROWS:
            assert "point" in row.kinds, row.id
            assert row.kinds <= KINDS, row.id
            assert row.evidence and row.cells, row.id

    def test_c_tmle_rows_license_the_point_only(self) -> None:
        for row in POINT_ROWS:
            if row.key.engine[0] == "collaborative_tmle":
                assert row.kinds == {"point"}, row.id

    def test_every_evidence_slug_is_a_registered_study(self) -> None:
        slugs = {record.slug for record in registered()}
        for row in POINT_ROWS:
            assert set(row.evidence) <= slugs, row.id


class TestCases:
    @pytest.mark.parametrize("case", CASES, ids=case_id)
    def test_a_case_classifies_to_its_own_row(self, case: RowCase) -> None:
        result = classified(case)
        assert result.row is not None, (case.row, result.nearest)
        assert result.row.id == case.row
        assert case.slug in result.row.evidence

    def test_each_row_has_a_case_for_each_evidence_slug(self) -> None:
        cased = {(case.row, case.slug) for case in CASES}
        for row in POINT_ROWS:
            for slug in row.evidence:
                assert (row.id, slug) in cased, (row.id, slug)

    def test_each_case_slug_is_evidence_of_its_row(self) -> None:
        for case in CASES:
            assert case.slug in ROWS[case.row].evidence, (case.row, case.slug)

    def test_each_source_names_a_line_of_an_existing_file(self) -> None:
        for case in CASES:
            name, _, line = case.source.rpartition(":")
            path = ROOT / name
            assert path.is_file(), case.source
            assert 1 <= int(line) <= len(path.read_text(encoding="utf-8").splitlines()), case.source

    def test_the_cases_cover_the_point_studies_the_rows_name(self) -> None:
        assert {case.slug for case in CASES} == {
            slug for row in POINT_ROWS for slug in row.evidence
        }

    def test_the_point_studies_without_a_row(self) -> None:
        """The registered studies that no point row names are longitudinal, exempt, or listed here."""
        named = {slug for row in POINT_ROWS for slug in row.evidence}
        point_unmapped = {
            record.slug
            for record in registered()
            if record.slug not in named and not _is_longitudinal_or_exempt(record.slug)
        }
        assert point_unmapped == {"canonical-ctmle-sequences", "default-simultaneous-bands"}


def _is_longitudinal_or_exempt(slug: str) -> bool:
    exempt = {
        "omitted-variable-bound-se",
        "calibration-slope-warning",
        "learned-rule-cvtmle-boundary",
    }
    longitudinal = ("ltmle", "longitudinal", "sdr", "survival", "sequential", "node-mechanisms")
    return slug in exempt or any(part in slug for part in longitudinal)


class TestControls:
    """Each control changes one setting and must move the classification."""

    @staticmethod
    def case(row: str, slug: str | None = None) -> RowCase:
        return next(c for c in CASES if c.row == row and (slug is None or c.slug == slug))

    def test_a_case_named_for_another_row_fails(self) -> None:
        wrong = replace(self.case("T01.binary", "canonical-tmle"), row="T02.binary")
        assert classified(wrong).row.id != wrong.row  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("row", "attribute", "value"),
        [
            ("T01.binary", "cross_fit", True),
            ("T01.binary", "fluctuation", "linear"),
            ("T02.binary", "cv_evaluation", True),
            ("T05.binary", "repeats", 1),
            ("T06", "target_weights", True),
            ("D01", "guard", ("Q",)),
            ("D08", "randomized", False),
            ("C03", "oat_design", "per_arm"),
        ],
    )
    def test_one_changed_setting_leaves_the_row(self, row: str, attribute: str, value: Any) -> None:
        case = self.case(row)
        estimator, data, estimands = case.build()
        assert classify(estimator, data, estimands).row.id == row  # type: ignore[union-attr]
        changed = copy.copy(estimator)
        setattr(changed, attribute, value)
        moved = classify(changed, data, estimands)
        assert moved.row is None or moved.row.id != row

    def test_a_cluster_count_below_the_floor_leaves_the_row(self) -> None:
        estimator, _, _ = self.case("T11.in_sample").build()
        _, data, _ = _build(estimator, frame(clusters=8, n=240), ("ate",))
        small = classify(estimator, data, ("ate",))
        assert small.row is None
        assert small.nearest is not None and "dependence" in small.nearest[1]

    def test_a_family_the_row_lacks_leaves_the_row(self) -> None:
        estimator, data, _ = self.case("T06").build()
        assert classify(estimator, data, ("ey1", "att")).row is None

    def test_a_row_covers_a_fit_of_a_subset_of_its_families(self) -> None:
        estimator, data, _ = self.case("T01.binary").build()
        assert classify(estimator, data, ("ate",)).row.id == "T01.binary"  # type: ignore[union-attr]


class TestExpansion:
    def test_rows_over_makes_one_row_per_value(self) -> None:
        base = ROWS["T05.binary"]
        rows = rows_over(base, "dependence", {"a": "iid", "b": "t"})
        assert [row.id for row in rows] == ["T05.binary.a", "T05.binary.b"]
        assert rows[1].key.dependence == "t"
        assert rows[0].key == base.key

    def test_rows_over_refuses_the_family_axis(self) -> None:
        with pytest.raises(ValueError, match="cannot expand"):
            rows_over(ROWS["T06"], "family", {"a": frozenset({"P1"})})

    def test_merge_rows_refuses_rows_that_differ_beyond_the_family(self) -> None:
        with pytest.raises(ValueError, match="differ on more than the family"):
            merge_rows("X", ROWS["T01.binary"], ROWS["T02.binary"])

    def test_two_rows_with_one_key_are_a_match_error(self) -> None:
        estimator, data, estimands = TestControls.case("T06").build()
        key, learners = point_key(estimator, data, estimands)
        twin = replace(ROWS["T06"], id="TWIN")
        with pytest.raises(RuntimeError, match="all match one fit"):
            classify_key(key, learners, rows=[ROWS["T06"], twin])


def test_the_counts() -> None:
    kinds = Counter(row.id[0] for row in POINT_ROWS)
    assert kinds["T"] > 0 and kinds["D"] > 0 and kinds["C"] > 0
