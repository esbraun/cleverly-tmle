"""The B4 draft: the kind of each positive red row, its default decision and its candidate rows.

The synthetic rows below stand for the committed ledger, so each rule of plan section 6.2 has a
case that fails when the rule is wrong. One test reads the committed ledger and checks that the
draft holds one entry per positive red row.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from cleverly._support import registered_rows
from tests.studies.evidence import support_dispositions as sd
from tests.studies.evidence.red_cells import RedKey, RedRow


def positive(
    slug: str,
    kind: str,
    key: str,
    *,
    owner: str = "",
    role: str = "positive",
    measured: str = "coverage 0.90",
    legs: tuple[str, ...] = (),
) -> sd.PositiveRow:
    red = RedRow(RedKey(slug, kind, key), role, "its own rule", measured)
    return sd.PositiveRow(red, owner, legs)


ROWS = [
    ("A1", frozenset({"point", "wald", "band"}), ("study-a",)),
    ("A2", frozenset({"point", "wald_t"}), ("study-a", "study-b")),
    ("A3", frozenset({"point", "bootstrap"}), ("study-a",)),
    ("B1", frozenset({"point"}), ("study-b",)),
]


class TestKinds:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("interval_calibration/ate__correctly_specified", ("wald",)),
            ("interval_calibration/boot_ate__correctly_specified", ("bootstrap",)),
            ("few_cluster_reference/j20", ("wald_t", "wald")),
            ("clustered_inference/cluster_robust", ("wald_t", "wald")),
            ("simultaneous_coverage/band", ("band",)),
            ("root_n_and_efficiency/n_500", ("point", "wald")),
            ("double_robustness/ate", ("point",)),
            ("double_robust_contraction/ate", ("point",)),
            ("sequential_double_robustness/ate", ("point",)),
            ("targeting_necessity/ate", ("point",)),
            ("selector_necessity/greedy", ("point",)),
            ("generated_design/ate", ("point",)),
            ("type_i_error/sharp_null", ("test",)),
            ("power/alternative", ("test",)),
            ("unknown_family/cell", ()),
        ],
    )
    def test_a_property_family_maps_to_its_kinds(self, key: str, expected: tuple[str, ...]) -> None:
        assert sd.kinds_of(positive("study-a", "property", key)) == expected

    def test_a_t_reference_study_reads_its_wald_cell_as_wald_t(self) -> None:
        row = positive("few-cluster-cross-fitted-ltmle", "property", "interval_calibration/j20")
        assert sd.kinds_of(row) == ("wald_t",)

    def test_a_paired_row_is_a_wald_row(self) -> None:
        assert sd.kinds_of(positive("study-a", "paired", "s/ate", role="paired")) == ("wald",)

    @pytest.mark.parametrize(
        ("legs", "expected"),
        [
            (("bias_equivalent",), ("point",)),
            (("coverage_valid",), ("wald",)),
            (("se_calibrated",), ("wald",)),
            (("bias_equivalent", "coverage_valid"), ("point", "wald")),
            ((), ("point", "wald")),
        ],
    )
    def test_a_truth_row_takes_its_kinds_from_the_failing_leg(
        self, legs: tuple[str, ...], expected: tuple[str, ...]
    ) -> None:
        row = positive("study-a", "truth", "cleverly-x/s/ate", role="cleverly-x", legs=legs)
        assert sd.kinds_of(row) == expected


class TestDefaults:
    def draft(self, *rows: sd.PositiveRow) -> list[sd.DraftEntry]:
        return sd.draft_entries(rows, ROWS)

    def test_f18_and_f19_default_to_a_withheld_status_through_q1(self) -> None:
        for owner in ("F18", "F19"):
            (entry,) = self.draft(
                positive("study-b", "property", "double_robustness/x", owner=owner)
            )
            assert (entry.disposition, entry.flag) == ("withheld (status)", "Q1")
            assert entry.candidates == ("A2", "B1")

    def test_a_finite_sample_owner_defaults_to_a_stated_limit_with_the_measured_numbers(
        self,
    ) -> None:
        (entry,) = self.draft(
            positive(
                "study-a",
                "property",
                "interval_calibration/x",
                owner="X13-finite-sample",
                measured="coverage 0.931 to 0.944",
            )
        )
        assert entry.disposition == "supported with a stated limit"
        assert entry.limit == "X13-finite-sample: coverage 0.931 to 0.944"
        assert entry.flag == ""

    def test_the_x20_crossfit_cell_is_experimental_and_the_clustered_cell_is_flagged(self) -> None:
        crossfit, clustered = self.draft(
            positive(
                "study-a",
                "property",
                "interval_calibration/boot_ey_crossfit",
                owner="X20-bootstrap",
            ),
            positive(
                "study-a",
                "property",
                "interval_calibration/boot_ey_clustered",
                owner="X20-bootstrap",
            ),
        )
        assert (crossfit.disposition, crossfit.flag) == ("experimental", "")
        assert crossfit.kinds == ("bootstrap",)
        assert clustered.disposition == "experimental"
        assert "0.912 to 0.934" in clustered.flag

    @pytest.mark.parametrize(
        "owner",
        ["X8-logit-small-stratum", "X8-drtmle-one-sided-bias", "RM18-one-sided-bias", "F28", "F27"],
    )
    def test_the_flagged_experimental_owners_are_flagged(self, owner: str) -> None:
        (entry,) = self.draft(
            positive("study-a", "property", "interval_calibration/x", owner=owner)
        )
        assert entry.disposition == "experimental"
        assert entry.flag.startswith(owner)

    def test_an_owner_with_no_default_is_flagged(self) -> None:
        (entry,) = self.draft(
            positive("study-a", "property", "interval_calibration/x", owner="X41-negative-control")
        )
        assert entry.disposition == "experimental"
        assert "no default for owner X41-negative-control" in entry.flag

    def test_a_family_with_no_kind_rule_is_flagged(self) -> None:
        (entry,) = self.draft(positive("study-a", "property", "other/x", owner="F28"))
        assert "no kind rule" in entry.flag


class TestCandidates:
    def test_a_candidate_names_the_study_and_lists_every_kind(self) -> None:
        (entry,) = sd.draft_entries(
            [
                positive(
                    "study-a", "property", "simultaneous_coverage/band", owner="band-finite-sample"
                )
            ],
            ROWS,
        )
        assert entry.candidates == ("A1",)

    def test_a_study_whose_rows_lack_the_kind_has_no_candidate_and_a_flag(self) -> None:
        (entry,) = sd.draft_entries(
            [
                positive(
                    "study-b", "property", "simultaneous_coverage/band", owner="band-finite-sample"
                )
            ],
            ROWS,
        )
        assert entry.candidates == ()
        assert "no row of the study lists these kinds" in entry.flag

    def test_a_study_with_no_row_has_no_candidate(self) -> None:
        (entry,) = sd.draft_entries([positive("study-c", "paired", "s/ate", role="paired")], ROWS)
        assert entry.candidates == ()


class TestRender:
    def test_the_draft_marks_flagged_lines_and_counts_the_entries(self) -> None:
        entries = sd.draft_entries(
            [
                positive(
                    "study-a", "property", "interval_calibration/x", owner="X13-finite-sample"
                ),
                positive("study-a", "property", "interval_calibration/y", owner="F28"),
            ],
            ROWS,
        )
        text = sd.render_draft(entries)
        assert "Entries: 2. Flagged: 1." in text
        assert text.count("FLAG:") == 1
        assert "- supported with a stated limit: 1" in text

    def test_a_pipe_in_a_cell_does_not_split_the_table_row(self) -> None:
        entries = sd.draft_entries(
            [positive("study-a", "property", "interval_calibration/x|y", owner="F28")], ROWS
        )
        line = next(row for row in sd.render_draft(entries).splitlines() if "study-a" in row)
        assert line.count("|") == 9


class TestCommittedTable:
    def test_the_draft_has_one_entry_per_positive_red_row_of_the_ledger(self) -> None:
        from cleverly._support import registered_rows
        from tests.studies.evidence.red_cells import frames, red_rows
        from tests.studies.evidence.registry import registered

        expected = 0
        for record in registered():
            tables = frames(record)
            for red in red_rows(record, tables):
                kind = red.key.kind
                expected += int(
                    (kind == "property" and red.role == "positive")
                    or (kind == "truth" and red.role.startswith("cleverly"))
                    or kind == "paired"
                )
        positives = sd.positive_red_rows()
        entries = sd.draft_entries(
            positives, [(row.id, row.kinds, row.evidence) for row in registered_rows()]
        )
        assert len(entries) == expected == len({entry.key for entry in entries})
        assert all(entry.kinds for entry in entries)


class TestCommandLine:
    def test_the_help_text_names_the_draft_option(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "tests.studies.evidence.support_dispositions", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0
        assert "--draft" in done.stdout and "--out" in done.stdout

    def test_the_draft_option_is_required(self) -> None:
        with pytest.raises(SystemExit) as raised:
            sd.build_parser().parse_args([])
        assert raised.value.code == 2

    def test_main_writes_the_draft_to_a_file(self, tmp_path: Path) -> None:
        out = tmp_path / "b4.md"
        assert sd.main(["--draft", "--out", str(out)]) == 0
        assert out.read_text(encoding="utf-8").startswith("# B4 draft")


def problems(
    dispositions: Mapping[RedKey, sd.Disposition],
    decisions: Mapping[RedKey, str],
    positive_keys: set[RedKey],
) -> list[str]:
    """What is wrong with a table of decisions, as one text per problem (plan section 6.3)."""
    rows = {row.id: row for row in registered_rows()}
    found: list[str] = []
    for key in positive_keys - set(dispositions):
        found.append(f"no decision for {key}")
    for key in set(dispositions) - positive_keys:
        found.append(f"{key} is no positive red row")
    for key, entry in dispositions.items():
        row = rows.get(entry.row)
        if row is None:
            found.append(f"{key}: row {entry.row} does not exist")
            continue
        if entry.disposition == "withheld (status)":
            if row.key.engine[0] != "collaborative_tmle":
                found.append(f"{key}: row {entry.row} does not hold the withheld engine")
        elif sd.ROW_STUDY.get(key.slug, key.slug) not in row.evidence:
            found.append(f"{key}: row {entry.row} does not list the study")
        if entry.disposition == "supported with a stated limit":
            if not set(entry.kinds) <= row.kinds:
                found.append(f"{key}: row {entry.row} lists no {entry.kinds}")
            if not entry.limit or entry.limit not in row.limits:
                found.append(f"{key}: the row note does not print the limit")
        elif entry.limit:
            found.append(f"{key}: only a stated limit carries a limit text")
        if entry.disposition == "experimental":
            for kind in entry.kinds:
                shortfall = any(k == kind for k, _ in row.shortfalls)
                if kind in row.kinds and not shortfall and key not in sd.NO_SHORTFALL:
                    found.append(f"{key}: row {entry.row} labels no {kind} fit")
        if entry.flag and not decisions.get(key):
            found.append(f"{key}: the flagged line has no decision")
    for key in decisions:
        if not dispositions[key].flag:
            found.append(f"{key}: a decision for an unflagged line")
    return found


class TestCommittedDecisions:
    @pytest.fixture(scope="class")
    def keys(self) -> set[RedKey]:
        return {positive.red.key for positive in sd.positive_red_rows()}

    def test_the_committed_table_has_no_problem(self, keys: set[RedKey]) -> None:
        assert problems(sd.DISPOSITIONS, sd.DECISIONS, keys) == []

    def test_dropping_an_entry_fails_the_check(self, keys: set[RedKey]) -> None:
        table = dict(sd.DISPOSITIONS)
        table.pop(next(iter(table)))
        assert any("no decision for" in text for text in problems(table, sd.DECISIONS, keys))

    def test_a_control_role_row_fails_the_check(self, keys: set[RedKey]) -> None:
        extra = RedKey("canonical-tmle", "property", "type_i_error/a_control_cell")
        table = {**sd.DISPOSITIONS, extra: next(iter(sd.DISPOSITIONS.values()))}
        assert any("is no positive red row" in text for text in problems(table, sd.DECISIONS, keys))

    def test_blanking_a_decision_fails_the_check(self, keys: set[RedKey]) -> None:
        decisions = dict(sd.DECISIONS)
        decisions[next(iter(decisions))] = ""
        assert any("has no decision" in text for text in problems(sd.DISPOSITIONS, decisions, keys))

    def test_a_limit_that_the_row_does_not_state_fails_the_check(self, keys: set[RedKey]) -> None:
        key = next(k for k, d in sd.DISPOSITIONS.items() if d.disposition.startswith("supported"))
        table = {
            **sd.DISPOSITIONS,
            key: replace(sd.DISPOSITIONS[key], row="T01.binary", kinds=("band",)),
        }
        assert any("lists no" in text for text in problems(table, sd.DECISIONS, keys))

    def test_a_licensed_experimental_cell_needs_a_shortfall_or_a_reason(
        self, keys: set[RedKey]
    ) -> None:
        key = next(k for k in sd.NO_SHORTFALL)
        without = {k: v for k, v in sd.NO_SHORTFALL.items() if k != key}
        saved = sd.NO_SHORTFALL
        sd.NO_SHORTFALL = without  # type: ignore[misc]
        try:
            assert any(
                "labels no" in text for text in problems(sd.DISPOSITIONS, sd.DECISIONS, keys)
            )
        finally:
            sd.NO_SHORTFALL = saved  # type: ignore[misc]

    def test_every_shortfall_makes_the_fit_of_its_row_experimental(self) -> None:
        from cleverly._support import Classification, support_for

        for row in registered_rows():
            for kind, text in row.shortfalls:
                classification = Classification(row.key, {}, row, None, ())
                support = support_for(
                    classification, "influence_curve", 5 if kind == "wald_t" else None, kind=kind
                )
                assert support.label == "experimental"
                assert text in support.reason

    def test_a_row_without_its_shortfall_licenses_the_fit(self) -> None:
        """The control of the test above: removing the shortfall removes the label."""
        from cleverly._support import Classification, support_for

        row = next(r for r in registered_rows() if r.id == "T16.logit_binary")
        classification = Classification(row.key, {}, replace(row, shortfalls=()), None, ())
        assert support_for(classification, "influence_curve").label == "supported"
        with_shortfall = Classification(row.key, {}, row, None, ())
        assert support_for(with_shortfall, "influence_curve").label == "experimental"
