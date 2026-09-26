"""The one table of non-inferential statuses agrees with the type, the docs and the roadmap.

:data:`cleverly._inference_status.NON_INFERENTIAL` holds every text the package prints
for a status that supplies no inference, and every consumer reads it there. A new status
is therefore one Literal member and one record. These tests pin the four places that
must move with it: the Literal, the status table in
``docs/technical-reference/inference.md``, the roadmap item each reason names, and the
precedence order a fit with several statuses resolves by.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import get_args

import pytest

from cleverly import _inference_status
from cleverly._inference_status import (
    NON_INFERENTIAL,
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
#: here. The order is the one the roadmap's RM20 precedence table gives: the two collaborative
#: statuses, the estimated weight of a guarded DR-TMLE fit, and then the two cluster statuses.
PRECEDENCE = (
    "working_mechanism_plugin",
    "generated_design_plugin",
    "estimated_weight_plugin",
    "unequal_cluster_plugin",
    "few_cluster_plugin",
)


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
        assert set(NON_INFERENTIAL) == set(get_args(InferenceStatus)) - {"influence_curve"}

    def test_influence_curve_has_no_record(self) -> None:
        with pytest.raises(KeyError):
            status_record("influence_curve")

    def test_the_documented_names_are_the_leaf_objects(self) -> None:
        """``cleverly.inference.influence`` re-exports the type and the predicate."""
        assert influence.InferenceStatus is InferenceStatus
        assert influence.supplies_inference is _inference_status.supplies_inference

    def test_the_table_is_read_only(self) -> None:
        assert isinstance(NON_INFERENTIAL, MappingProxyType)
        with pytest.raises(TypeError):
            NON_INFERENTIAL["new_status"] = NON_INFERENTIAL[PRECEDENCE[0]]  # type: ignore[index]


class TestEachRecordIsComplete:
    @pytest.mark.parametrize("status", list(NON_INFERENTIAL))
    def test_every_text_is_present(self, status: str) -> None:
        record = NON_INFERENTIAL[status]
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

    @pytest.mark.parametrize("status", list(NON_INFERENTIAL))
    def test_the_reason_is_whole_sentences(self, status: str) -> None:
        """Every refusal prints the reason after a full stop, so it opens a sentence."""
        reason = NON_INFERENTIAL[status].reason
        assert reason[0].isupper() and reason.endswith(".")

    @pytest.mark.parametrize("status", list(NON_INFERENTIAL))
    def test_the_reason_withholds_all_three_accessors(self, status: str) -> None:
        reason = NON_INFERENTIAL[status].reason
        assert "no confidence interval, no p-value and no standard error" in reason
        assert "plugin_std_error and plugin_interval" in reason

    @pytest.mark.parametrize("status", list(NON_INFERENTIAL))
    def test_the_reason_names_its_roadmap_item_and_the_anchor_exists(self, status: str) -> None:
        record = NON_INFERENTIAL[status]
        assert f"{record.reopened_by} in docs/roadmap.md" in record.reason
        assert f"{record.reopened_by} in the roadmap" in record.assessment_note
        heading = re.compile(rf"^#{{2,4}} {re.escape(record.reopened_by)}\. ", re.MULTILINE)
        assert heading.search(ROADMAP.read_text(encoding="utf-8")), record.reopened_by

    def test_the_summary_labels_are_distinct(self) -> None:
        labels = [record.summary_label for record in NON_INFERENTIAL.values()]
        assert len(set(labels)) == len(labels)
        # No label is the inferential header, so a diagnostic column never reads as one.
        assert not {"std_err", "se", "standard error"} & set(labels)

    def test_the_diagnostic_nouns_are_distinct(self) -> None:
        nouns = [record.diagnostic_noun for record in NON_INFERENTIAL.values()]
        assert len(set(nouns)) == len(nouns)


class TestTheDocumentedTable:
    """``docs/technical-reference/inference.md#inference-status`` has one row per status."""

    def test_the_page_lists_every_member_once(self) -> None:
        documented = _documented_statuses()
        assert sorted(documented) == sorted(get_args(InferenceStatus))

    def test_the_page_lists_the_statuses_in_precedence_order(self) -> None:
        documented = _documented_statuses()
        assert documented[0] == "influence_curve"
        assert tuple(documented[1:]) == tuple(NON_INFERENTIAL)

    def test_each_row_shows_the_label_and_the_reopen_item_of_its_record(self) -> None:
        rows = dict(zip(_documented_statuses(), _status_table(), strict=True))
        for status, record in NON_INFERENTIAL.items():
            cells = " | ".join(rows[status])
            assert f"`{record.summary_label}`" in cells
            assert f"[{record.reopened_by}](" in cells


class TestThePrecedence:
    def test_the_order_is_pinned(self) -> None:
        assert tuple(NON_INFERENTIAL) == PRECEDENCE

    def test_every_pair_resolves_to_the_earlier_status(self) -> None:
        for first, second in combinations(NON_INFERENTIAL, 2):
            assert precedent_status([second, first]) == first
            assert precedent_status([first, "influence_curve", second]) == first

    def test_no_non_inferential_status_resolves_to_influence_curve(self) -> None:
        assert precedent_status([]) == "influence_curve"
        assert precedent_status(["influence_curve", "influence_curve"]) == "influence_curve"
        for status in NON_INFERENTIAL:
            assert precedent_status([status]) == status

    def test_an_unknown_status_is_refused(self) -> None:
        with pytest.raises(KeyError):
            precedent_status(["influence_curve", "no_such_status"])

    def test_the_resolution_reads_the_table_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The nonzero witness: with one real status, the pair loop above has no pair.

        A two-entry table in the reverse of alphabetical order shows the resolution
        follows insertion order rather than the order of the arguments or of the names.
        """
        record = NON_INFERENTIAL[PRECEDENCE[0]]
        table: Mapping[str, StatusRecord] = MappingProxyType(
            {"second_by_name": record, "first_by_name": record}
        )
        monkeypatch.setattr(_inference_status, "NON_INFERENTIAL", table)
        assert precedent_status(["first_by_name", "second_by_name"]) == "second_by_name"
        assert precedent_status(["first_by_name"]) == "first_by_name"


class TestTheFewClusterThreshold:
    """The one constant holds the threshold, and the text is formatted from it."""

    def test_the_threshold_is_the_sourced_count(self) -> None:
        # Nugent et al. (2024), Section 2.2: "fewer than 40 clusters randomized (N < 40)".
        assert _inference_status.FEW_CLUSTER_THRESHOLD == 40

    def test_the_reason_states_the_threshold_it_applies(self) -> None:
        threshold = _inference_status.FEW_CLUSTER_THRESHOLD
        reason = NON_INFERENTIAL["few_cluster_plugin"].reason
        assert (
            f"when it has fewer than {threshold} clusters with positive weight mass, "
            "or when one baseline stratum"
        ) in reason
        assert f"below {threshold} clusters" in reason
        assert f"at least {threshold} of them" in NON_INFERENTIAL["unequal_cluster_plugin"].reason

    def test_no_threshold_is_written_by_hand_in_the_module(self) -> None:
        """The literal appears once, on the constant, so the text cannot drift from it."""
        source = Path(_inference_status.__file__).read_text(encoding="utf-8")
        code = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        written = [line for line in code if re.search(r"\b40\b", line)]
        assert written == ["FEW_CLUSTER_THRESHOLD: Final[int] = 40"]
