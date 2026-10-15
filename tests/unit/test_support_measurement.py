"""The phase-1 aggregator reads record files and counts them, on a synthetic record root.

The root below holds one source of each category and one record of each kind. Each test states
which counts the aggregator must give and which record it must leave out.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.studies import support_measurement as sm


def key(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "point",
        "engine": ["tmle"],
        "family": ["P1"],
        "outcome": ["binomial"],
        "missing": ["complete"],
        "treatment": ["binary"],
        "weighting": "none",
        "splitting": ["in_sample"],
        "dependence": "iid",
        "qualifiers": [],
        "options": [["multiplier_kind", "rademacher"]],
    }
    return base | overrides


SUPPORTED = key()
EXPERIMENTAL = key(splitting=["stacked"])
CLUSTERED = key(dependence="below_floor")
CTMLE = key(engine=["collaborative_tmle", "oat", "per_arm"])


def classification(group_key: dict[str, Any], label: str, **extra: Any) -> dict[str, Any]:
    return {
        "record": "classification",
        "source": "TMLE._fit_single",
        "key": group_key,
        "learners": {},
        "label": label,
        "row": "T01.binary" if label == "supported" else None,
        "nearest": None if label == "supported" else ["T02.binary", ["engine"]],
        "conditions": [],
    } | extra


def outcome(group_key: dict[str, Any], statuses: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "record": "fit_outcome",
        "source": "TMLE._fit_single",
        "key": group_key,
        "row": "T01.binary",
        "label": "supported",
        "statuses": statuses,
        "supplies_inference": [s == "influence_curve" for s in statuses],
        "clustered": False,
        "bootstrap_attached": False,
        "bootstrap_inferential_today": False,
        "band_built_today": False,
        "n_estimates": 1,
    } | extra


def write(root: Path, group: str, pid: int, records: list[Any]) -> None:
    folder = root / group
    folder.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    (folder / f"{pid}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write(
        tmp_path,
        "fast",
        11,
        [
            classification(SUPPORTED, "supported"),
            classification(SUPPORTED, "supported"),
            classification(EXPERIMENTAL, "experimental", conditions=["F21"]),
            outcome(SUPPORTED, ["influence_curve"]),
            outcome(EXPERIMENTAL, ["estimated_weight_plugin"]),
            outcome(EXPERIMENTAL, ["estimated_weight_plugin", "working_mechanism_plugin"]),
            outcome(CLUSTERED, ["few_cluster_plugin"]),
            outcome(
                SUPPORTED,
                ["influence_curve"],
                bootstrap_attached=True,
                bootstrap_inferential_today=True,
                supplies_inference=[True],
            ),
            outcome(
                EXPERIMENTAL,
                ["influence_curve"],
                bootstrap_attached=True,
                bootstrap_inferential_today=False,
                supplies_inference=[True],
                row="T01.binary",
            ),
            outcome(CTMLE, ["influence_curve"], bootstrap_attached=True, supplies_inference=[True]),
            outcome(SUPPORTED, ["influence_curve"], band_built_today=True),
            outcome(EXPERIMENTAL, ["influence_curve"], band_built_today=True, row="T09"),
            {
                "record": "refusal_site",
                "site": "arm.fold_targeting",
                "code": "X",
                "roadmap": "F1",
                "flagged": True,
            },
            {
                "record": "refusal_site",
                "site": "drtmle.weight_tilted_missing_outcome",
                "code": "X",
                "roadmap": "F2",
                "flagged": False,
            },
            {
                "record": "availability_site",
                "site": "study.ctmle_clustered",
                "code": "X",
                "roadmap": "F3",
                "flagged": True,
            },
            {
                "record": "classification_error",
                "source": "TMLE._fit_single",
                "error": "KeyError('x')",
            },
            "{not json",
        ],
    )
    write(
        tmp_path,
        "notebooks/one",
        12,
        [
            classification(SUPPORTED, "supported"),
            {
                "record": "refusal_site",
                "site": "arm.fold_targeting",
                "code": "X",
                "roadmap": "F1",
                "flagged": True,
            },
        ],
    )
    write(
        tmp_path,
        "studies/default_bands",
        13,
        [classification(SUPPORTED, "supported"), classification(EXPERIMENTAL, "experimental")],
    )
    write(tmp_path, "studies/unknown_directory", 14, [classification(SUPPORTED, "supported")])
    return tmp_path


ROWS = {
    "T01.binary": frozenset({"point", "wald", "bootstrap"}),
    "T09": frozenset({"point", "wald", "band"}),
}


def measured(root: Path, **overrides: Any) -> sm.Measurement:
    settings: dict[str, Any] = {
        "rows": ROWS,
        "cases": {"default-simultaneous-bands": {sm._key_text(SUPPORTED)}},
        "directories": {"default_bands": "default-simultaneous-bands"},
        "registered_slugs": ["default-simultaneous-bands", "canonical-tmle"],
    }
    return sm.measure(sm.read_records(root), **(settings | overrides))


class TestReading:
    def test_each_record_carries_the_directory_of_its_file(self, root: Path) -> None:
        groups = {record["group"] for record in sm.read_records(root)}
        assert groups == {
            "fast",
            "notebooks/one",
            "studies/default_bands",
            "studies/unknown_directory",
        }

    def test_a_line_that_is_not_json_is_counted_and_skipped(self, root: Path) -> None:
        kinds = [record["record"] for record in sm.read_records(root)]
        assert kinds.count("malformed") == 1


class TestFitsByLabel:
    def test_calls_and_distinct_keys_by_source_and_label(self, root: Path) -> None:
        fits = measured(root).fits
        assert fits["fast"]["supported"] == {"calls": 2, "keys": 1}
        assert fits["fast"]["experimental"] == {"calls": 1, "keys": 1}
        assert fits["notebooks/one"]["supported"] == {"calls": 1, "keys": 1}

    def test_the_fit_total_counts_classification_records_only(self, root: Path) -> None:
        totals = measured(root).totals
        assert totals["fits"] == 2 + 1 + 1 + 2 + 1
        assert totals["fit_outcome"] == 9

    def test_an_experimental_key_names_its_nearest_row_and_condition(self, root: Path) -> None:
        entry = measured(root).experimental[0]
        assert entry["nearest"] == ["T02.binary", ["engine"]]
        assert entry["conditions"] == ["F21"]
        assert entry["calls"] == {"fast": 1, "studies": 1}


class TestAvailability:
    def test_an_availability_record_is_listed_and_in_no_fit_count(self, root: Path) -> None:
        result = measured(root)
        assert result.availability == {"study.ctmle_clustered": 1}
        assert result.totals["availability_site"] == 1
        assert all(site["site"] != "study.ctmle_clustered" for site in result.reach)
        assert result.totals["fits"] == 7

    def test_a_classifier_error_is_reported(self, root: Path) -> None:
        errors = measured(root).errors
        assert [e["error"] for e in errors] == ["KeyError('x')"]


class TestGroupAReach:
    def test_flagged_sites_come_first_with_calls_by_category(self, root: Path) -> None:
        reach = measured(root).reach
        assert [entry["site"] for entry in reach] == [
            "arm.fold_targeting",
            "drtmle.weight_tilted_missing_outcome",
        ]
        assert reach[0]["flagged"] is True
        assert reach[0]["calls"] == {"fast": 1, "notebooks": 1}


class TestWithheldToShown:
    def test_a_default_shown_status_moves_and_a_mixed_fit_waits_for_the_alternative(
        self, root: Path
    ) -> None:
        status = measured(root).status
        assert status["estimated_weight_plugin"]["calls"] == 2
        assert status["few_cluster_plugin"]["calls"] == 1
        assert status["working_mechanism_plugin"]["calls"] == 1
        moves = status["(fits whose every status moves)"]
        # Default: the lone estimated-weight fit and the lone few-cluster fit.
        assert moves["calls"] == 2
        # Alternative: those two and the mixed fit of a shown and a withheld-today status.
        assert moves["alternative_calls"] == 3


class TestBootstrapAndBand:
    def test_each_section_3_4_row_counts_its_fits(self, root: Path) -> None:
        flips = measured(root).flips
        assert flips["licensed bootstrap: inferential and supported"]["calls"] == 1
        assert (
            flips["diagnostic today at a supplying status: inferential and experimental"]["calls"]
            == 1
        )
        assert flips["C-TMLE bootstrap: stays diagnostic"]["calls"] == 1
        assert flips["band built, shape not in a row: experimental"]["calls"] == 1

    def test_a_band_in_a_row_that_lists_it_is_supported(self, root: Path) -> None:
        flips = measured(root).flips
        assert flips["band built, row lists band: supported"] == {"calls": 1, "keys": 1}

    def test_a_default_shown_status_without_a_band_is_a_candidate(self, root: Path) -> None:
        flips = measured(root).flips
        label = "no band today, status shown by default: a band is built when requested"
        assert flips[label]["calls"] == 3


class TestStudies:
    def test_a_runner_key_without_a_case_or_with_an_experimental_label_is_listed(
        self, root: Path
    ) -> None:
        studies = measured(root).studies
        by_key = {(entry["study"], entry["label"]): entry for entry in studies}
        assert ("default-simultaneous-bands", "experimental") in by_key
        assert by_key["default-simultaneous-bands", "experimental"]["matches_a_case"] is False
        # The supported key of the same study equals a case key, so it is not listed.
        assert ("default-simultaneous-bands", "supported") not in by_key

    def test_a_directory_with_no_slug_stands_for_itself(self, root: Path) -> None:
        studies = measured(root).studies
        assert any(
            entry["study"] == "unknown_directory" and not entry["matches_a_case"]
            for entry in studies
        )

    def test_a_registered_study_with_no_record_is_listed(self, root: Path) -> None:
        assert measured(root).silent_studies == ["canonical-tmle"]

    def test_without_cases_every_key_matches(self, root: Path) -> None:
        studies = measured(root, cases=None).studies
        assert all(entry["matches_a_case"] for entry in studies)


class TestDirectories:
    def test_a_regeneration_script_maps_to_the_study_whose_runner_it_names(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "alpha"
        folder.mkdir()
        (folder / "regenerate.py").write_text(
            "from tests.studies import canonical_alpha, alpha_properties\n", encoding="utf-8"
        )
        other = tmp_path / "beta"
        other.mkdir()
        (other / "regenerate.py").write_text("import nothing_registered\n", encoding="utf-8")
        found = sm.canonical_directories(tmp_path, {"alpha-study": "tests.studies.canonical_alpha"})
        assert found == {"alpha": "alpha-study"}

    def test_the_repository_directories_map_to_registered_slugs(self) -> None:
        found = sm.canonical_directories()
        assert found["default_bands"] == "default-simultaneous-bands"


class TestCommandLine:
    def test_the_help_text_names_every_option(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "tests.studies.support_measurement", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0
        for option in ("--records", "--out", "--json", "--skip-cases"):
            assert option in done.stdout

    def test_main_writes_the_tables_and_leaves_the_records_alone(
        self, root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        before = {p: p.read_bytes() for p in root.rglob("*.jsonl")}
        out = tmp_path_factory.mktemp("out") / "measurement.md"
        status = sm.main(
            ["--records", str(root), "--out", str(out), "--json", str(out.with_suffix(".json"))],
            case_source=lambda: {},
        )
        assert status == 0
        text = out.read_text(encoding="utf-8")
        for heading in (
            "## Fits by label",
            "## Experimental keys",
            "## Withheld to shown",
            "## Bootstrap and band flips",
            "## Group-A and flagged reach",
            "## Availability records",
            "## Studies",
        ):
            assert heading in text
        assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["totals"]
        assert {p: p.read_bytes() for p in root.rglob("*.jsonl")} == before

    def test_a_missing_record_root_is_an_error(self, tmp_path: Path) -> None:
        assert sm.main(["--records", str(tmp_path / "absent")], case_source=lambda: {}) == 2
