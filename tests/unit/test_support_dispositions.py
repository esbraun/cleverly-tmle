"""The B4 draft: the kind of each positive red row, its default decision and its candidate rows.

The synthetic rows below stand for the committed ledger, so each rule of plan section 6.2 has a
case that fails when the rule is wrong. One test reads the committed ledger and checks that the
draft holds one entry per positive red row.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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
    def test_the_decision_tables_are_empty_until_the_owner_answers(self) -> None:
        assert dict(sd.DISPOSITIONS) == {}
        assert dict(sd.DECISIONS) == {}

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
