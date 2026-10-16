"""The support rows and the cases that stand for them.

Each row of ``cleverly._support_rows`` names the studies that cover it. Each case of
``tests/studies/support_cases.py`` builds one fit of that configuration, or runs a study's own
fit to its classifier, and the case must classify onto its own row. A case that lands on another row fails, so a row cannot pass on
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
    REGIMEN_KINDS,
    SupportRow,
    classify,
    classify_key,
    point_key,
    regimen_kinds,
    registered_rows,
)
from cleverly._support_rows import (
    EXEMPT_STUDIES,
    FAMILY_THEORY,
    LONGITUDINAL_ROWS,
    OPTION_ALLOWLIST,
    POINT_ROWS,
    merge_rows,
    rows_over,
)
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.support_cases import (
    CASES,
    Observed,
    RowCase,
    _build,
    classification,
    frame,
)

ALL_ROWS = (*POINT_ROWS, *LONGITUDINAL_ROWS)
ROWS = {row.id: row for row in ALL_ROWS}
KINDS = {
    "point",
    "wald",
    "wald_t",
    "band",
    "bootstrap",
    "bootstrap_bca",
    "bootstrap_band",
    "derived",
}
OUTCOME_KINDS = {"end_of_study", "survival", "competing", "repeated"}


def classified(case: RowCase) -> Any:
    return classification(case)


def case_id(case: RowCase) -> str:
    return f"{case.row}-{case.slug}-{case.cell}"


def non_family(row: SupportRow) -> tuple[Any, ...]:
    axes = row.key.axes()
    return tuple(axes[name] for name in KEY_AXES if name != "family")


class TestRegistry:
    def test_the_registered_rows_are_the_point_and_longitudinal_rows(self) -> None:
        assert registered_rows() == ALL_ROWS

    def test_row_ids_are_unique(self) -> None:
        assert len(ROWS) == len(ALL_ROWS)

    def test_every_row_has_a_case_and_every_case_names_a_row(self) -> None:
        with_cases = {case.row for case in CASES}
        assert set(ROWS) - with_cases == set()
        assert with_cases - set(ROWS) == set()

    def test_no_two_point_rows_cover_the_same_fit(self) -> None:
        """Two rows with equal axes and a shared family would both match one fit."""
        for index, first in enumerate(POINT_ROWS):
            for second in POINT_ROWS[index + 1 :]:
                overlap = non_family(first) == non_family(second) and (
                    first.key.family & second.key.family
                )
                assert not overlap, (first.id, second.id)

    def test_no_two_longitudinal_rows_cover_the_same_fit(self) -> None:
        """A longitudinal fit holds one outcome kind, one reporting token and some regimen kinds."""
        for index, first in enumerate(LONGITUDINAL_ROWS):
            for second in LONGITUDINAL_ROWS[index + 1 :]:
                if non_family(first) != non_family(second):
                    continue
                shared_fixed = {
                    token
                    for token in (first.key.family & second.key.family)
                    if token in OUTCOME_KINDS | {"msm", "regimens"}
                }
                same_shape = len(shared_fixed) == 2
                regimens = (
                    (first.key.family & second.key.family) - OUTCOME_KINDS - {"msm", "regimens"}
                )
                assert not (same_shape and regimens), (first.id, second.id)

    def test_every_row_licenses_the_point_estimate_and_known_kinds(self) -> None:
        for row in ALL_ROWS:
            assert "point" in row.kinds, row.id
            assert row.kinds <= KINDS, row.id
            assert row.evidence and row.cells, row.id

    def test_c_tmle_rows_license_the_point_only_unless_the_interval_is_supplied(self) -> None:
        # C-TMLE0 (derivative) and the stacked per-arm curve supply an interval. Every other
        # collaborative path withholds it.
        supplied = ("derivative", "parametric_stack")
        for row in ALL_ROWS:
            if row.key.engine[0] == "collaborative_tmle":
                if any(part in row.key.engine for part in supplied):
                    assert "wald" in row.kinds, row.id
                else:
                    assert row.kinds == {"point"}, row.id

    def test_every_evidence_slug_is_a_registered_study(self) -> None:
        slugs = {record.slug for record in registered()}
        for row in ALL_ROWS:
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
        for row in ALL_ROWS:
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

    def test_the_cases_cover_the_studies_the_rows_name(self) -> None:
        assert {case.slug for case in CASES} == {slug for row in ALL_ROWS for slug in row.evidence}

    def test_every_registered_study_is_row_evidence_or_exempt(self) -> None:
        named = {slug for row in ALL_ROWS for slug in row.evidence}
        slugs = {record.slug for record in registered()}
        assert slugs - named - set(EXEMPT_STUDIES) == set()

    def test_an_exempt_study_is_registered_named_by_no_row_and_has_a_reason(self) -> None:
        named = {slug for row in ALL_ROWS for slug in row.evidence}
        slugs = {record.slug for record in registered()}
        assert set(EXEMPT_STUDIES) <= slugs
        assert not set(EXEMPT_STUDIES) & named
        assert all(reason.strip().endswith(".") for reason in EXEMPT_STUDIES.values())

    def test_a_synthetic_case_naming_another_studys_row_fails_the_evidence_check(self) -> None:
        wrong = replace(CASES[0], slug="canonical-ltmle")
        assert wrong.slug not in ROWS[wrong.row].evidence


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
            ("T22", "ratio", "classifier"),
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


class TestLongitudinalControls:
    """Each control changes one axis of a captured longitudinal key and must leave the row."""

    @staticmethod
    def observed(row: str, slug: str | None = None) -> Observed:
        case = next(c for c in CASES if c.row == row and (slug is None or c.slug == slug))
        built = case.build()
        assert isinstance(built, Observed)
        return built

    @pytest.mark.parametrize(
        ("row", "axis", "value"),
        [
            ("G01", "splitting", ("cross_fitted",)),
            ("G01", "weighting", "fixed"),
            ("G01", "dependence", "clustered"),
            ("G01", "dependence", "below_floor"),
            ("G01", "treatment", ("held", "2", "2")),
            ("G01", "treatment", ("2", "2", "2")),
            ("G01", "outcome", ("gaussian", "unbounded")),
            ("G01", "missing", ("complete",)),
            ("G01", "qualifiers", frozenset({"A1"})),
            ("G01", "engine", ("sdr",)),
            ("G01", "family", frozenset({"end_of_study", "regimens", "mtp"})),
            ("G01", "family", frozenset({"survival", "regimens", "static"})),
            ("G21", "qualifiers", frozenset({"A1", "A2"})),
            ("G28.d2.ipw.hajek", "options", (("multiplier_kind", "rademacher"),)),
            ("G10.continuous", "treatment", ("continuous", "2")),
            ("G35.density", "treatment", ("continuous", "continuous")),
            ("G36.classifier", "treatment", ("mixed", "mixed")),
        ],
    )
    def test_one_changed_axis_leaves_the_row(self, row: str, axis: str, value: Any) -> None:
        built = self.observed(row)
        assert classify_key(built.key, built.learners).row.id == row  # type: ignore[union-attr]
        moved = classify_key(replace(built.key, **{axis: value}), built.learners)
        assert moved.row is None or moved.row.id != row

    def test_a_longitudinal_case_with_no_row_names_its_nearest_row(self) -> None:
        built = self.observed("G01")
        result = classify_key(replace(built.key, weighting="estimated"), built.learners)
        assert result.row is None
        assert result.nearest is not None and result.nearest[1] == ("weighting",)

    def test_normalization_separates_the_two_ipw_rows(self) -> None:
        hajek = self.observed("G28.d2.ipw.hajek")
        ht = self.observed("G28.d2.ipw.horvitz_thompson")
        assert hajek.key != ht.key
        assert classify_key(ht.key, ht.learners).row.id == "G28.d2.ipw.horvitz_thompson"  # type: ignore[union-attr]

    @staticmethod
    def without_option(key: Any, name: str) -> Any:
        return replace(key, options=tuple(item for item in key.options if item[0] != name))

    def test_a_hazard_fit_never_matches_an_iterated_mean_row(self) -> None:
        hazard = self.observed("G30.survival", "point-survival-hazard")
        assert dict(hazard.key.options)["survival_model"] == "hazard"
        assert classify_key(hazard.key, hazard.learners).row.id == "G30.survival"  # type: ignore[union-attr]
        # The same design read as an iterated-mean fit is the registered G14 shape.
        mean = replace(
            hazard.key,
            options=tuple(
                (name, "iterated_mean" if name == "survival_model" else value)
                for name, value in hazard.key.options
            ),
        )
        assert classify_key(mean, hazard.learners).row.id == "G14.survival"  # type: ignore[union-attr]

    def test_a_hazard_fit_with_no_row_is_experimental(self) -> None:
        # A hazard fit with estimated weights has no registered row, and the iterated-mean row
        # of the same design must not absorb it.
        hazard = self.observed("G30.survival", "point-survival-hazard")
        result = classify_key(replace(hazard.key, weighting="estimated"), hazard.learners)
        assert result.row is None
        mean = self.observed("G14.survival")
        moved = replace(
            mean.key,
            options=tuple(
                (name, "hazard" if name == "survival_model" else value)
                for name, value in mean.key.options
            ),
            weighting="estimated",
        )
        assert classify_key(moved, mean.learners).row is None

    def test_control_without_the_survival_model_option_the_hazard_row_collides(self) -> None:
        # Mutation control: the registered rows with the option removed from every key put
        # a hazard fit on two rows, which the classifier refuses.
        rows = [
            replace(row, key=self.without_option(row.key, "survival_model"))
            if row.key.kind == "longitudinal"
            else row
            for row in ALL_ROWS
        ]
        hazard = self.observed("G30.survival", "point-survival-hazard")
        with pytest.raises(RuntimeError):
            classify_key(
                self.without_option(hazard.key, "survival_model"), hazard.learners, rows=rows
            )

    def test_a_held_design_is_not_covered_by_a_free_one(self) -> None:
        held = self.observed("G14.end_of_study")
        free = self.observed("G01")
        assert held.key.treatment[0] == "held"
        assert classify_key(held.key, held.learners).row.id == "G14.end_of_study"  # type: ignore[union-attr]
        assert classify_key(free.key, free.learners).row.id == "G01"  # type: ignore[union-attr]

    def test_a_static_and_a_stochastic_categorical_fit_share_one_row(self) -> None:
        stochastic = self.observed("G08.in_sample", "stochastic-categorical-ltmle")
        static = self.observed("G08.in_sample", "canonical-categorical-ltmle")
        assert "stochastic" in stochastic.key.family and "stochastic" not in static.key.family
        for built in (stochastic, static):
            assert classify_key(built.key, built.learners).row.id == "G08.in_sample"  # type: ignore[union-attr]


class TestRegimenKinds:
    @staticmethod
    def plan(*, mtp: bool = False, tilt: bool = False, policy: bool = False) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(has_mtp=mtp, has_tilt=tilt, has_policy=policy)

    def test_each_kind_has_a_plan_that_makes_it(self) -> None:
        make = self.plan
        assert regimen_kinds([make()]) == {"static"}
        assert regimen_kinds([make(policy=True)]) == {"stochastic"}
        assert regimen_kinds([make(mtp=True, policy=True)]) == {"mtp"}
        assert regimen_kinds([make(tilt=True, policy=True)]) == {"tilt"}

    def test_mixed_plans_hold_every_kind(self) -> None:
        make = self.plan
        assert regimen_kinds([make(), make(policy=True), make(mtp=True)]) == {
            "static",
            "stochastic",
            "mtp",
        }

    def test_the_kinds_are_the_declared_ones(self) -> None:
        make = self.plan
        assert regimen_kinds([make(), make(mtp=True), make(tilt=True), make(policy=True)]) <= set(
            REGIMEN_KINDS
        )


class TestTheoryAndOptions:
    def test_every_family_a_row_holds_has_a_theory_anchor(self) -> None:
        families = {family for row in ALL_ROWS for family in row.key.family}
        assert families - set(FAMILY_THEORY) == set()

    def test_a_family_without_an_anchor_is_found(self) -> None:
        families = {family for row in ALL_ROWS for family in row.key.family}
        trimmed = {name: anchor for name, anchor in FAMILY_THEORY.items() if name != "survival"}
        assert families - set(trimmed) == {"survival"}

    def test_each_anchor_row_is_a_row_of_the_evidence_page(self) -> None:
        text = (ROOT / "docs" / "technical-reference" / "evidence.md").read_text(encoding="utf-8")
        for family, anchor in FAMILY_THEORY.items():
            if anchor.row:
                assert f"| {anchor.row} |" in text, family

    def test_each_anchor_names_existing_instruments(self) -> None:
        for family, anchor in FAMILY_THEORY.items():
            assert anchor.instruments, family
            for path in anchor.instruments:
                assert (ROOT / path).is_file(), (family, path)

    def test_the_families_with_no_evidence_page_row_are_listed(self) -> None:
        """The evidence page lacks a row for the longitudinal MTP and the longitudinal tilt."""
        open_rows = {family for family, anchor in FAMILY_THEORY.items() if not anchor.row}
        assert open_rows == {"mtp", "tilt"}

    def test_each_allowlisted_option_is_stated_in_its_cited_file(self) -> None:
        for name, cited in OPTION_ALLOWLIST.items():
            path = ROOT / cited
            assert path.is_file(), (name, cited)
            assert name in path.read_text(encoding="utf-8"), (name, cited)

    def test_no_allowlisted_option_is_a_key_option(self) -> None:
        keyed = {name for row in ALL_ROWS for name, _ in row.key.options}
        assert keyed & set(OPTION_ALLOWLIST) == set()

    def test_the_allowlist_holds_the_declared_options(self) -> None:
        declared = {"g_bounds", "n_folds", "learner_folds", "random_state", "n_jobs", "run_id"}
        assert declared <= set(OPTION_ALLOWLIST)


def test_the_counts() -> None:
    kinds = Counter(row.id[0] for row in ALL_ROWS)
    assert kinds["T"] > 0 and kinds["D"] > 0 and kinds["C"] > 0 and kinds["G"] > 0
