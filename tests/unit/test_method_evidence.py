"""Every registered method-evidence study, against its committed artefacts.

Parametrized over :func:`~tests.studies.evidence.registry.registered` rather than written
per study: a method that declares a :class:`~tests.studies.evidence.registry.StudyRecord`
inherits the completeness checks, the recomputation, the published verdicts and the negative
controls without a line of new test code.

The split between this module and ``tests/e2e/test_method_evidence_slow.py`` is cost, not
trust.  Everything deterministic -- the summaries, the Student and exact binomial intervals,
and the verdicts those endpoints imply -- is recomputed here from the replication rows.  The
resampling bounds and the full re-execution of the estimator over every replication are the
same checks at a price the fast tier cannot pay, and live there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from tests.documents import pipe_table
from tests.studies.evidence.claims import describe, load, matches, value
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.inference import clopper_pearson, student_interval
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.schema import truth_on_inference_scale, validate_replicates

STUDIES = registered()
IDS = [study.slug for study in STUDIES]

#: Mutation controls only need a verdict to flip, not a published endpoint, so they run the
#: resampling at a fraction of the published budget.
CONTROL_BOOTSTRAP = 2_000


@pytest.fixture(params=STUDIES, ids=IDS)
def study(request: pytest.FixtureRequest) -> StudyRecord:
    return request.param


@pytest.fixture
def rows(study: StudyRecord) -> pd.DataFrame:
    return pd.read_csv(study.artifact("replicates.csv.gz"))


def _cheap(study: StudyRecord) -> StudyRecord:
    margins = dataclasses.replace(study.margins, bootstrap_replicates=CONTROL_BOOTSTRAP)
    return dataclasses.replace(study, margins=margins)


class TestArtifacts:
    def test_the_manifest_hashes_every_published_result(self, study: StudyRecord) -> None:
        manifest = json.loads(study.artifact("manifest.json").read_text(encoding="utf-8"))
        assert set(manifest["sha256"]) == {
            "replicates.csv.gz",
            "summary.csv",
            "equivalence.csv",
            "performance-tests.csv",
            "property-replicates.csv.gz",
            "properties.csv",
        }
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(study.artifact(name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["reference_sha256"].items():
            assert hashlib.sha256(study.artifact(name).read_bytes()).hexdigest() == digest
        configuration = manifest["configuration"]
        assert configuration["replicates"] == study.replicates
        assert configuration["n"] == study.n
        assert configuration["seed"] == study.seed
        assert configuration["margins"] == study.margins.as_json()

    def test_the_subject_side_is_identified_rather_than_described(self, study: StudyRecord) -> None:
        """``"working tree"`` names no run; a version and a revision name one.

        The manifest is not asserted equal to the working tree -- re-execution is what keeps
        the artefacts honest, and a hash gate here would only mean rewriting the manifest for
        a comment change.  What it must do is identify the run well enough to reproduce it.
        """
        manifest = json.loads(study.artifact("manifest.json").read_text(encoding="utf-8"))
        subject = manifest["generated_with"]["subject"]
        assert subject["implementation"] == study.implementation
        assert re.fullmatch(r"\d+\.\d+.*", subject["cleverly_version"]), subject
        assert re.fullmatch(r"[0-9a-f]{40}", subject["cleverly_commit"]), subject
        for package in ("python", "numpy", "scipy", "pandas", "scikit_learn"):
            assert subject[package], f"{package} version missing from the manifest"
        assert set(manifest["study_module_sha256"]) == set(study.modules)
        for module in study.modules:
            assert (ROOT / module).exists(), f"{module} is named by the manifest but is gone"

    def test_the_replication_file_satisfies_the_shared_contract(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        validate_replicates(rows, record=study)

    def test_published_summaries_are_recomputed_from_the_replication_rows(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        published = pd.read_csv(study.artifact("summary.csv"))
        pd.testing.assert_frame_equal(
            published, summarize(rows), check_exact=False, rtol=1e-12, atol=1e-12
        )

    @pytest.mark.parametrize("name", ["summary", "equivalence", "performance", "properties"])
    def test_reader_facing_tables_are_not_empty(self, study: StudyRecord, name: str) -> None:
        assert not load(study)[name].empty


class TestPublishedVerdicts:
    """The deterministic half of the published verdicts, recomputed from the rows."""

    def test_the_bias_and_coverage_intervals_are_the_rows_own(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        published = pd.read_csv(study.artifact("performance-tests.csv")).set_index(
            ["implementation", "scenario", "estimand"]
        )
        for key, group in rows.groupby(["implementation", "scenario", "estimand"], sort=True):
            record = published.loc[key]
            inference = group["inference_estimate"].to_numpy(dtype=float)
            truth = truth_on_inference_scale(
                str(key[2]),
                float(group["truth"].iloc[0]),
                str(group["inference_scale"].iloc[0]),
            )
            bias = student_interval(
                inference - truth, confidence_level=study.margins.confidence_level
            )
            coverage = clopper_pearson(
                int(group["covered"].sum()),
                len(group),
                confidence_level=study.margins.confidence_level,
            )
            assert bias.low == pytest.approx(record["bias_ci_lower"], rel=1e-12)
            assert bias.high == pytest.approx(record["bias_ci_upper"], rel=1e-12)
            assert coverage.low == pytest.approx(record["coverage_ci_lower"], rel=1e-12)
            assert coverage.high == pytest.approx(record["coverage_ci_upper"], rel=1e-12)

    def test_every_verdict_is_the_consequence_of_its_recorded_endpoints(
        self, study: StudyRecord
    ) -> None:
        """No verdict column may say something its own interval does not."""
        published = pd.read_csv(study.artifact("performance-tests.csv"))
        expected_bias = (published["bias_ci_lower"] >= -published["bias_margin"]) & (
            published["bias_ci_upper"] <= published["bias_margin"]
        )
        assert published["bias_equivalent"].equals(expected_bias)
        assert published["coverage_valid"].equals(
            published["coverage_ci_lower"] >= published["coverage_floor"]
        )
        assert published["se_calibrated"].equals(
            (published["se_ratio_ci_lower"] >= published["se_ratio_margin_lower"])
            & (published["se_ratio_ci_upper"] <= published["se_ratio_margin_upper"])
        )
        assert published["passed"].equals(
            published["bias_equivalent"] & published["coverage_valid"] & published["se_calibrated"]
        )

    def test_each_implementation_independently_performs_against_truth(
        self, study: StudyRecord
    ) -> None:
        published = pd.read_csv(study.artifact("performance-tests.csv"))
        assert set(published["implementation"]) == set(study.implementations)
        assert (published["confidence_level"] == study.margins.confidence_level).all()
        assert published["passed"].all(), published.loc[~published["passed"]].to_string()

    def test_the_subject_is_similar_to_and_no_worse_than_the_reference(
        self, study: StudyRecord
    ) -> None:
        published = pd.read_csv(study.artifact("equivalence.csv"))
        assert published["dropped_replications"].eq(0).all()
        assert published["paired_similarity"].all(), published.loc[~published["paired_similarity"]]
        assert published["subject_not_inferior"].all(), published.loc[
            ~published["subject_not_inferior"]
        ]
        assert published["passed"].equals(
            published["paired_similarity"] & published["subject_not_inferior"]
        ), "the published verdict is not the two claims the document makes"

    def test_the_reference_is_reported_on_its_own_terms(self, study: StudyRecord) -> None:
        """A reference that degrades is a reference finding, not a subject failure."""
        published = pd.read_csv(study.artifact("equivalence.csv"))
        assert published["reference_valid"].all(), published.loc[~published["reference_valid"]]
        assert published["subject_valid"].all(), published.loc[~published["subject_valid"]]

    def test_paper_property_verdicts_are_recomputed_from_the_replication_rows(
        self, study: StudyRecord
    ) -> None:
        from tests.studies.canonical_properties import summarize_properties

        rows = pd.read_csv(study.artifact("property-replicates.csv.gz"))
        published = pd.read_csv(study.artifact("properties.csv"))
        pd.testing.assert_frame_equal(
            published,
            summarize_properties(rows),
            check_exact=False,
            check_dtype=False,
            rtol=1e-9,
            atol=1e-9,
        )
        assert published["passed"].all(), published.loc[~published["passed"]].to_string()


class TestNegativeControls:
    """Corrupt one implementation and require exactly that one to fail."""

    @pytest.fixture
    def cell(self, study: StudyRecord, rows: pd.DataFrame) -> pd.DataFrame:
        scenario = next(iter(study.scenarios))
        estimand = study.scenarios[scenario][0]
        return rows.loc[(rows["scenario"] == scenario) & (rows["estimand"] == estimand)].copy()

    @staticmethod
    def _shift_bias(frame: pd.DataFrame, mask: pd.Series) -> None:
        spread = float(frame.loc[mask, "inference_estimate"].std(ddof=1))
        frame.loc[mask, "inference_estimate"] += spread

    @staticmethod
    def _lose_coverage(frame: pd.DataFrame, mask: pd.Series) -> None:
        frame.loc[mask, "covered"] = 0

    @staticmethod
    def _inflate_standard_errors(frame: pd.DataFrame, mask: pd.Series) -> None:
        frame.loc[mask, "std_error"] *= 2.0

    @pytest.mark.parametrize("target", [0, 1], ids=["subject", "reference"])
    @pytest.mark.parametrize("label", ["bias", "coverage", "standard error"])
    def test_a_corrupted_implementation_fails_alone(
        self, study: StudyRecord, cell: pd.DataFrame, target: int, label: str
    ) -> None:
        mutations = {
            "bias": self._shift_bias,
            "coverage": self._lose_coverage,
            "standard error": self._inflate_standard_errors,
        }
        implementations = study.implementations
        implementation = implementations[target]
        other = implementations[1 - target]

        mutated = cell.copy()
        mutations[label](mutated, mutated["implementation"] == implementation)
        verdicts = independent_performance_tests(mutated, record=_cheap(study), n_jobs=1).set_index(
            "implementation"
        )
        assert not bool(verdicts.loc[implementation, "passed"]), (
            f"a corrupted {label} for {implementation} was accepted"
        )
        assert bool(verdicts.loc[other, "passed"]), (
            f"corrupting {implementation}'s {label} implicated {other}"
        )

    def test_a_material_subject_regression_fails_similarity_and_noninferiority(
        self, study: StudyRecord, cell: pd.DataFrame
    ) -> None:
        cheap = _cheap(study)
        mutated = cell.copy()
        mask = mutated["implementation"] == study.implementation
        shift = 4.0 * float(mutated.loc[mask, "estimate"].std(ddof=1))
        mutated.loc[mask, "estimate"] += shift
        mutated.loc[mask, "covered"] = 0
        summaries = summarize(mutated)
        performance = independent_performance_tests(mutated, record=cheap, n_jobs=1)
        verdict = equivalence(mutated, summaries, performance, record=cheap, n_jobs=1).iloc[0]
        assert not bool(verdict["paired_similarity"])
        assert not bool(verdict["subject_not_inferior"])
        assert not bool(verdict["passed"])

    def test_a_reference_regression_leaves_the_subject_standing(
        self, study: StudyRecord, cell: pd.DataFrame
    ) -> None:
        """The asymmetry the document promises, exhibited rather than described."""
        cheap = _cheap(study)
        mutated = cell.copy()
        mask = mutated["implementation"] == study.reference
        mutated.loc[mask, "covered"] = 0
        summaries = summarize(mutated)
        performance = independent_performance_tests(mutated, record=cheap, n_jobs=1)
        verdict = equivalence(mutated, summaries, performance, record=cheap, n_jobs=1).iloc[0]
        assert not bool(verdict["reference_valid"])
        assert bool(verdict["subject_valid"])
        assert bool(verdict["subject_not_inferior"]), (
            "a reference that covers nothing cannot make the subject inferior"
        )


class TestTheStudyStillMeasuresTheCode:
    """The artefacts are evidence about ``cleverly`` only if ``cleverly`` still produces them."""

    @pytest.mark.parametrize("replicate", [0, 1])
    def test_refitting_a_committed_replication_reproduces_its_row(
        self, study: StudyRecord, rows: pd.DataFrame, replicate: int
    ) -> None:
        from tests.studies.canonical_tmle import cleverly_rows, draw_scenario

        for scenario in study.scenarios:
            frame, truth = draw_scenario(scenario, study.n, replicate)
            refitted = pd.DataFrame(cleverly_rows(frame, truth, scenario, replicate))
            published = rows.loc[
                (rows["implementation"] == study.implementation)
                & (rows["scenario"] == scenario)
                & (rows["replicate"] == replicate)
            ]
            merged = published.merge(refitted, on="estimand", suffixes=("_published", "_refitted"))
            assert len(merged) == len(published) == len(study.scenarios[scenario])
            for column in ("estimate", "std_error", "ci_lower", "ci_upper"):
                # Four orders of magnitude tighter than the narrowest margin any verdict
                # uses, and loose enough for the last bits of a different BLAS.
                assert merged[f"{column}_refitted"].to_numpy() == pytest.approx(
                    merged[f"{column}_published"].to_numpy(), rel=1e-6, abs=1e-9
                ), f"{scenario} replicate {replicate} no longer reproduces its {column}"

    def test_the_reference_moved_its_estimates_off_the_plug_in(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        """Agreement cannot be explained by neither implementation targeting anything.

        ``tmle3`` reports the pre-targeting plug-in beside the targeted estimate, so at least
        one target has to have moved: an exact-agreement check goes blind precisely where the
        fluctuation is zero.
        """
        reference = rows.loc[(rows["implementation"] == study.reference) & (rows["replicate"] == 0)]
        moved = (reference["estimate"] - reference["initial_estimate"]).abs()
        assert moved.max() > 1e-3, moved.describe()

    def test_native_interval_scales_are_recorded_instead_of_forced_to_match(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        for estimand in study.incomparable_se:
            scales = (
                rows.loc[rows["estimand"] == estimand]
                .groupby("implementation")["inference_scale"]
                .nunique()
            )
            assert (scales == 1).all()
            reported = set(
                rows.loc[rows["estimand"] == estimand, "inference_scale"].drop_duplicates()
            )
            assert len(reported) > 1, (
                f"{estimand} is declared incomparable but both implementations report the "
                f"same scale {reported}, so the exemption is not earned"
            )

    def test_the_ratio_estimands_report_on_the_log_scale(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        for estimand in ("rr", "or"):
            if estimand not in study.estimands:
                continue
            assert set(rows.loc[rows["estimand"] == estimand, "inference_scale"]) == {"log"}
            selected = rows.loc[rows["estimand"] == estimand]
            assert np.allclose(
                selected["inference_estimate"], np.log(selected["estimate"]), rtol=1e-9
            )


GRID = ROOT / "docs" / "evidence.md"

#: The grid's header, in order.  The first study's row split the old single "paper-property
#: study" column in two: 34 of the tests it was credited with are per-implementation
#: performance tests, half of them measuring the R reference, and none of them is a property
#: from the TMLE paper.
GRID_COLUMNS = (
    "method",
    "estimands and intervals",
    "independent performance vs truth",
    "cross-implementation study",
    "paper-property study",
    "limitations",
)

#: Which cell carries which pair of counts, and which quantities they must equal.
COUNTED = {
    "independent performance vs truth": ("independent_tests_passed", "independent_tests_total"),
    "cross-implementation study": ("paired_tests_passed", "paired_tests_total"),
    "paper-property study": ("property_cells_passed", "property_cells_total"),
}

MEASURED_COLUMNS = ("quantity", "value", "source")

LINK = re.compile(r"\]\(([^)\s]+)\)")
COUNT = re.compile(r"(\d+)/(\d+)")


def _grid() -> dict[str, dict[str, str]]:
    rows = {row["method"]: row for row in pipe_table(GRID, GRID_COLUMNS)}
    assert len(rows) == len(pipe_table(GRID, GRID_COLUMNS)), "the grid has a duplicate method"
    return rows


class TestTheMethodEvidenceGrid:
    """``docs/evidence.md``'s method grid against the register and the committed results.

    The target table above it in the same document says of itself that it is a gate and not a
    note.  The grid arrived as a note: nothing read it, its counts were typed, and one of them
    counted the wrong study.  These tests are what make the two halves of that document the
    same kind of object, and they are written against the register rather than against this
    row, so the second method to be added inherits them.
    """

    def test_every_registered_study_has_a_row_and_every_row_a_study(self) -> None:
        rows = set(_grid())
        studies = {study.name for study in STUDIES}
        assert rows == studies, (
            f"rows with no registered study {sorted(rows - studies)}, studies with no row "
            f"{sorted(studies - rows)}. A study whose results are committed but unrowed is one "
            f"no reader is routed to"
        )

    def test_the_row_points_at_the_registered_studys_document_and_anchor(
        self, study: StudyRecord
    ) -> None:
        row = _grid()[study.name]
        targets = [target for cell in row.values() for target in LINK.findall(cell)]
        assert targets, (
            "the row links to nothing, so it stands in for the run rather than citing it"
        )
        anchors = set()
        for target in targets:
            path, _, anchor = target.partition("#")
            resolved = (GRID.parent / path).resolve()
            assert resolved == study.document_path.resolve(), (
                f"{study.slug}'s row links to {path}, not to its registered document "
                f"{study.document}"
            )
            anchors.add(anchor)
        assert study.anchor in anchors, (
            f"no cell links to {study.slug}'s registered section #{study.anchor}"
        )

    @pytest.mark.parametrize("column", sorted(COUNTED))
    def test_every_count_is_derived_from_the_committed_results(
        self, study: StudyRecord, column: str
    ) -> None:
        """The 34/34 that counted a different study is why this is not read by eye."""
        cell = _grid()[study.name][column]
        found = COUNT.findall(cell)
        assert len(found) == 1, f"{column!r} should carry exactly one count, found {found}"
        passed_name, total_name = COUNTED[column]
        data = load(study)
        passed, total = (int(part) for part in found[0])
        assert total == int(value(study, total_name, data)), (
            f"{column!r} claims {total} tests; {total_name} is "
            f"{int(value(study, total_name, data))}"
        )
        assert passed == int(value(study, passed_name, data)), (
            f"{column!r} claims {passed} passed; {passed_name} is "
            f"{int(value(study, passed_name, data))}"
        )

    def test_no_cell_is_left_to_the_reader(self, study: StudyRecord) -> None:
        blank = [
            column
            for column, cell in _grid()[study.name].items()
            if len(cell) < 20 or cell in {"-", "--", "—", ""}
        ]
        assert blank == [], (
            f"{study.slug}'s row says nothing in {blank}. The limitations column in particular: "
            f"every study goes blind somewhere, so an empty one has not been thought about"
        )


class TestTheQuotedMeasurements:
    """Every number the study document prints, against the artefacts it printed them from."""

    def test_every_quoted_value_is_the_rounding_of_the_computed_one(
        self, study: StudyRecord
    ) -> None:
        data = load(study)
        wrong = []
        for row in pipe_table(study.document_path, MEASURED_COLUMNS):
            name = row["quantity"].strip("`")
            computed = value(study, name, data)
            if not matches(row["value"], computed):
                wrong.append(f"{name}: {describe(computed, row['value'])}")
        assert wrong == [], (
            "the document quotes values its own results do not produce:\n  " + "\n  ".join(wrong)
        )

    def test_the_table_reaches_every_family_of_result(self, study: StudyRecord) -> None:
        """A measured table that quoted one artefact would leave the rest unchecked prose."""
        quoted = {
            row["quantity"].strip("`") for row in pipe_table(study.document_path, MEASURED_COLUMNS)
        }
        assert len(quoted) >= 8, f"only {len(quoted)} quantities are gated: {sorted(quoted)}"
        families = {_family(name) for name in quoted}
        missing = {"performance", "equivalence", "properties"} - families
        assert missing == set(), f"nothing in the measured table comes from {sorted(missing)}"


#: Which artefact an aggregate quantity summarises, for the coverage check above.
_FAMILY = {
    "independent_tests": "performance",
    "subject_tests": "performance",
    "min_coverage": "performance",
    "max_se_ratio": "performance",
    "min_se_ratio": "performance",
    "max_standardized_bias": "performance",
    "paired_tests": "equivalence",
    "max_rmse_ratio": "equivalence",
    "min_coverage_difference": "equivalence",
    "max_calibration_excess": "equivalence",
    "max_margin_utilization": "equivalence",
    "property_cells": "properties",
}


def _family(name: str) -> str:
    """Which artefact a quantity name comes from -- the reference form says so directly."""
    if "[" in name:
        return name.split("[", 1)[0]
    for prefix, family in _FAMILY.items():
        if name.startswith(prefix):
            return family
    return "other"
