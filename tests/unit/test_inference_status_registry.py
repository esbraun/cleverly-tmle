"""The one table of labelled statuses agrees with the type, the docs and the roadmap.

:data:`cleverly._inference_status.STATUS_RECORDS` holds every text the package prints
for a labelled status, and every consumer reads it there. A record shows or withholds its
interval. A new status is therefore one Literal member and one record. These tests pin the
four places that must move with it: the Literal, the status table in
``docs/technical-reference/inference.md``, the roadmap item each reason names, and the
two-tier precedence a fit with several statuses resolves by.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import get_args

import pytest

from cleverly import _inference_status
from cleverly._inference_status import (
    STATUS_RECORDS,
    InferenceStatus,
    StatusRecord,
    precedent_status,
    status_record,
)
from cleverly.inference import influence

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PAGE = ROOT / "docs" / "technical-reference" / "inference.md"
ROADMAP = ROOT / "docs" / "roadmap.md"

#: The precedence order, written out. A fit that meets more than one status takes the first one
#: here. The order is the row order of the status table in ``inference.md``: the two collaborative
#: statuses, the estimated weight of a guarded DR-TMLE fit, and then the cluster status.
PRECEDENCE = (
    "working_mechanism_plugin",
    "generated_design_plugin",
    "estimated_weight_plugin",
    "adaptive_baseline_plugin",
    "no_cluster_variance_plugin",
    "few_cluster_plugin",
    "constant_node_plugin",
    "stack_failure_plugin",
)

#: The statuses whose interval is shown under the experimental label. Eric's answer of
#: 2026-10-15 keeps the other labelled statuses point-only.
SHOWN = ("estimated_weight_plugin", "few_cluster_plugin")
WITHHELD = tuple(status for status in PRECEDENCE if status not in SHOWN)


def _status_table() -> list[list[str]]:
    """The rows of the status table under ``## Inference status``, cells stripped."""
    text = INFERENCE_PAGE.read_text(encoding="utf-8")
    section = text.split("## Inference status", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith('| `"')]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def _documented_statuses() -> list[str]:
    return [re.fullmatch(r'`"([a-z_]+)"`', row[0]).group(1) for row in _status_table()]  # type: ignore[union-attr]


class TestTheTableMatchesTheType:
    def test_every_non_inferential_member_has_one_record(self) -> None:
        assert set(STATUS_RECORDS) == set(get_args(InferenceStatus)) - {"influence_curve"}

    def test_influence_curve_has_no_record(self) -> None:
        with pytest.raises(KeyError):
            status_record("influence_curve")

    def test_the_documented_names_are_the_leaf_objects(self) -> None:
        """``cleverly.inference.influence`` re-exports the type and the predicate."""
        assert influence.InferenceStatus is InferenceStatus
        assert influence.supplies_inference is _inference_status.supplies_inference

    def test_the_table_is_read_only(self) -> None:
        assert isinstance(STATUS_RECORDS, MappingProxyType)
        with pytest.raises(TypeError):
            STATUS_RECORDS["new_status"] = STATUS_RECORDS[PRECEDENCE[0]]  # type: ignore[index]


class TestEachRecordIsComplete:
    @pytest.mark.parametrize("status", list(STATUS_RECORDS))
    def test_every_text_is_present(self, status: str) -> None:
        record = STATUS_RECORDS[status]
        assert isinstance(record, StatusRecord)
        for text in (
            record.reason,
            record.assessment_note,
            record.summary_label,
            record.bootstrap_note,
            record.diagnostic_noun,
            record.reopened_by,
        ):
            assert text.strip() == text and text

    @pytest.mark.parametrize("status", list(STATUS_RECORDS))
    def test_the_reason_is_whole_sentences(self, status: str) -> None:
        """Every refusal prints the reason after a full stop, so it opens a sentence."""
        reason = STATUS_RECORDS[status].reason
        assert reason[0].isupper() and reason.endswith(".")

    @pytest.mark.parametrize("status", WITHHELD)
    def test_the_reason_withholds_all_three_accessors(self, status: str) -> None:
        reason = STATUS_RECORDS[status].reason
        assert "no confidence interval, no p-value and no standard error" in reason
        assert "plugin_std_error and plugin_interval" in reason

    @pytest.mark.parametrize("status", SHOWN)
    def test_a_shown_reason_states_the_shortfall_and_withholds_nothing(self, status: str) -> None:
        record = STATUS_RECORDS[status]
        assert record.interval == "shown"
        assert record.shown_reason
        assert record.shown_reason[0].isupper() and record.shown_reason.endswith(".")
        assert "no confidence interval" not in record.shown_reason
        assert f"{record.reopened_by} in docs/roadmap.md" in record.shown_reason
        assert "no confidence interval" not in record.reason

    @pytest.mark.parametrize("status", WITHHELD)
    def test_a_withheld_record_states_no_shown_reason(self, status: str) -> None:
        record = STATUS_RECORDS[status]
        assert record.interval == "withheld" and record.shown_reason is None

    def test_a_record_that_disagrees_with_its_interval_is_refused(self) -> None:
        template = STATUS_RECORDS[WITHHELD[0]]
        with pytest.raises(ValueError, match="shown_reason"):
            replace(template, interval="shown")
        with pytest.raises(ValueError, match="shown_reason"):
            replace(template, shown_reason="Some words.")

    @pytest.mark.parametrize("status", list(STATUS_RECORDS))
    def test_the_reason_names_its_roadmap_item_and_the_anchor_exists(self, status: str) -> None:
        record = STATUS_RECORDS[status]
        assert f"{record.reopened_by} in docs/roadmap.md" in record.reason
        assert f"{record.reopened_by} in the roadmap" in record.assessment_note
        heading = re.compile(rf"^#{{2,4}} {re.escape(record.reopened_by)}\. ", re.MULTILINE)
        assert heading.search(ROADMAP.read_text(encoding="utf-8")), record.reopened_by

    def test_the_summary_labels_are_distinct(self) -> None:
        labels = [record.summary_label for record in STATUS_RECORDS.values()]
        assert len(set(labels)) == len(labels)
        # No label is the inferential header, so a diagnostic column never reads as one.
        assert not {"std_err", "se", "standard error"} & set(labels)

    def test_the_diagnostic_nouns_are_distinct(self) -> None:
        nouns = [record.diagnostic_noun for record in STATUS_RECORDS.values()]
        assert len(set(nouns)) == len(nouns)


class TestTheDocumentedTable:
    """``docs/technical-reference/inference.md#inference-status`` has one row per status."""

    def test_the_page_lists_every_member_once(self) -> None:
        documented = _documented_statuses()
        assert sorted(documented) == sorted(get_args(InferenceStatus))

    def test_the_page_lists_the_statuses_in_precedence_order(self) -> None:
        documented = _documented_statuses()
        assert documented[0] == "influence_curve"
        assert tuple(documented[1:]) == tuple(STATUS_RECORDS)

    def test_each_row_shows_the_label_and_the_reopen_item_of_its_record(self) -> None:
        rows = dict(zip(_documented_statuses(), _status_table(), strict=True))
        for status, record in STATUS_RECORDS.items():
            cells = " | ".join(rows[status])
            # A shown status keeps the ordinary column. A withheld one names its own label.
            label = "std_err" if record.interval == "shown" else record.summary_label
            assert f"`{label}`" in cells
            assert f"[{record.reopened_by}](" in cells


class TestThePrecedence:
    def test_the_order_is_pinned(self) -> None:
        assert tuple(STATUS_RECORDS) == PRECEDENCE

    def test_the_shown_statuses_are_the_two_the_owner_chose(self) -> None:
        shown = tuple(s for s, r in STATUS_RECORDS.items() if r.interval == "shown")
        assert shown == SHOWN

    def test_two_statuses_of_one_tier_resolve_to_the_earlier_in_the_table(self) -> None:
        for tier in (WITHHELD, SHOWN):
            for first, second in combinations(tier, 2):
                assert precedent_status([second, first]) == first
                assert precedent_status([first, "influence_curve", second]) == first

    def test_a_withheld_status_wins_over_a_shown_one_wherever_it_sits(self) -> None:
        """The two tiers, with a nonzero witness for each order of the table.

        ``no_cluster_variance_plugin`` sits before ``few_cluster_plugin``, and
        ``constant_node_plugin`` sits after it. Both beat the shown status, so a single
        tier in table order would fail the second pair.
        """
        for withheld in WITHHELD:
            for shown in SHOWN:
                assert precedent_status([shown, withheld]) == withheld
                assert precedent_status([withheld, shown]) == withheld

    def test_a_single_tier_in_table_order_fails_the_witness(self) -> None:
        """Deliberate mutation: the pre-tier rule returns the shown status for a late withheld one."""

        def one_tier(statuses: list[str]) -> str:
            applied = {s for s in statuses if s != "influence_curve"}
            return next((s for s in STATUS_RECORDS if s in applied), "influence_curve")

        assert one_tier(["few_cluster_plugin", "constant_node_plugin"]) == "few_cluster_plugin"
        assert precedent_status(["few_cluster_plugin", "constant_node_plugin"]) == (
            "constant_node_plugin"
        )

    def test_supplies_inference_reads_the_interval_of_the_record(self) -> None:
        assert _inference_status.supplies_inference("influence_curve")
        for status in SHOWN:
            assert _inference_status.supplies_inference(status)
        for status in WITHHELD:
            assert not _inference_status.supplies_inference(status)
        assert not _inference_status.is_labelled("influence_curve")
        assert all(_inference_status.is_labelled(s) for s in PRECEDENCE)

    def test_no_non_inferential_status_resolves_to_influence_curve(self) -> None:
        assert precedent_status([]) == "influence_curve"
        assert precedent_status(["influence_curve", "influence_curve"]) == "influence_curve"
        for status in STATUS_RECORDS:
            assert precedent_status([status]) == status

    def test_an_unknown_status_is_refused(self) -> None:
        with pytest.raises(KeyError):
            precedent_status(["influence_curve", "no_such_status"])

    def test_the_resolution_reads_the_table_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The nonzero witness: with one real status, the pair loop above has no pair.

        A two-entry table in the reverse of alphabetical order shows the resolution
        follows insertion order rather than the order of the arguments or of the names.
        """
        record = STATUS_RECORDS[PRECEDENCE[0]]
        table: Mapping[str, StatusRecord] = MappingProxyType(
            {"second_by_name": record, "first_by_name": record}
        )
        monkeypatch.setattr(_inference_status, "STATUS_RECORDS", table)
        assert precedent_status(["first_by_name", "second_by_name"]) == "second_by_name"
        assert precedent_status(["first_by_name"]) == "first_by_name"


class TestTheFewClusterThreshold:
    """The two constants hold the threshold and the floor, and the text is formatted from them."""

    def test_the_threshold_is_the_sourced_count(self) -> None:
        # Nugent et al. (2024), Section 2.2: "fewer than 40 clusters randomized (N < 40)".
        assert _inference_status.FEW_CLUSTER_THRESHOLD == 40

    def test_the_t_reference_floor_leaves_one_degree_of_freedom(self) -> None:
        # J - 2 >= 1 needs J >= 3. Below it, no_cluster_variance_plugin withholds the interval.
        assert _inference_status.MINIMUM_T_REFERENCE_CLUSTERS == 3

    def test_the_floor_is_the_smallest_measured_count(self) -> None:
        # clustered-few-cluster-tmle measures 10, 20 and 30 clusters.
        from tests.studies.clustered_few_cluster_tmle import CLUSTER_COUNTS

        assert _inference_status.MINIMUM_INTERVAL_CLUSTERS == min(CLUSTER_COUNTS) == 10

    def test_the_longitudinal_floor_is_the_smallest_measured_ltmle_count(self) -> None:
        from tests.studies.clustered_few_cluster_tmle import fits_at

        measured = [count for count in (10, 20, 30) if "ltmle_in_sample" in fits_at(count)]
        assert _inference_status.MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS == min(measured) == 20

    def test_the_reason_states_the_floor_and_the_threshold_it_applies(self) -> None:
        threshold = _inference_status.FEW_CLUSTER_THRESHOLD
        floor = _inference_status.MINIMUM_INTERVAL_CLUSTERS
        reason = STATUS_RECORDS["few_cluster_plugin"].reason
        t_floor = _inference_status.MINIMUM_T_REFERENCE_CLUSTERS
        assert (
            f"from {t_floor} clusters with positive weight mass up to one below {floor}" in reason
        )
        longitudinal = _inference_status.MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS
        assert f"one below {longitudinal} on a longitudinal fit" in reason
        assert "no registered study measures an interval below those counts" in reason
        assert (
            f"fewer than {t_floor} clusters" in STATUS_RECORDS["no_cluster_variance_plugin"].reason
        )
        assert f"fewer than {threshold}" in _inference_status.T_REFERENCE_NOTE
        assert f"fewer than {threshold}" in _inference_status.NO_T_REFERENCE_BANDS

    @pytest.mark.parametrize(
        ("literal", "constant"),
        [
            ("40", "FEW_CLUSTER_THRESHOLD: Final[int] = 40"),
            ("10", "MINIMUM_INTERVAL_CLUSTERS: Final[int] = 10"),
        ],
    )
    def test_no_count_is_written_by_hand_in_the_module(self, literal: str, constant: str) -> None:
        """Each literal appears once, on its constant, so the text cannot drift from it."""
        source = Path(_inference_status.__file__).read_text(encoding="utf-8")
        code = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        written = [line for line in code if re.search(rf"\b{literal}\b", line)]
        assert written == [constant]
