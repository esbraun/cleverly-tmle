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
#: here. The order is the row order of the status table in ``inference.md``: the two collaborative
#: statuses, the estimated weight of a guarded DR-TMLE fit, and then the cluster status.
PRECEDENCE = (
    "working_mechanism_plugin",
    "generated_design_plugin",
    "estimated_weight_plugin",
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
    """The two constants hold the threshold and the floor, and the text is formatted from them."""

    def test_the_threshold_is_the_sourced_count(self) -> None:
        # Nugent et al. (2024), Section 2.2: "fewer than 40 clusters randomized (N < 40)".
        assert _inference_status.FEW_CLUSTER_THRESHOLD == 40

    def test_the_floor_is_the_smallest_measured_count(self) -> None:
        # clustered-few-cluster-tmle measures 10, 20 and 30 clusters.
        from tests.studies.clustered_few_cluster_tmle import CLUSTER_COUNTS

        assert _inference_status.MINIMUM_INTERVAL_CLUSTERS == min(CLUSTER_COUNTS) == 10

    def test_the_reason_states_the_floor_and_the_threshold_it_applies(self) -> None:
        threshold = _inference_status.FEW_CLUSTER_THRESHOLD
        floor = _inference_status.MINIMUM_INTERVAL_CLUSTERS
        reason = NON_INFERENTIAL["few_cluster_plugin"].reason
        cross_fitted = _inference_status.MINIMUM_CROSS_FITTED_LONGITUDINAL_CLUSTERS
        assert (
            "when it reads fewer clusters with positive weight mass, in the fit or in one "
            "reported baseline stratum, than the smallest count its registered study measures: "
            f"{floor} for TMLE, DR-TMLE and in-sample LTMLE, and {cross_fitted} for "
            "cross-fitted LTMLE"
        ) in reason
        assert f"From the floor to {threshold - 1} such clusters" in reason
        assert "No registered study measures an interval below the floor of its fit" in reason
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
