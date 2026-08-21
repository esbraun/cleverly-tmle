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
from tests.studies import canonical_tmle
from tests.studies.evidence.claims import load, value
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.document import RESULTS_START, generated_results, render_results
from tests.studies.evidence.inference import clopper_pearson, student_interval
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.schema import truth_on_inference_scale, validate_replicates
from tests.studies.evidence.seeds import replicate_seed

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
        if name == "equivalence" and study.reference is None:
            assert load(study)[name].empty
            return
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
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
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
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
        published = pd.read_csv(study.artifact("equivalence.csv"))
        assert published["reference_valid"].all(), published.loc[~published["reference_valid"]]
        assert published["subject_valid"].all(), published.loc[~published["subject_valid"]]

    def test_paper_property_verdicts_are_recomputed_from_the_replication_rows(
        self, study: StudyRecord
    ) -> None:
        rows = pd.read_csv(study.artifact("property-replicates.csv.gz"))
        published = pd.read_csv(study.artifact("properties.csv"))
        pd.testing.assert_frame_equal(
            published,
            study.properties().summarize_properties(rows),
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

    @pytest.mark.parametrize("label", ["bias", "coverage", "standard error"])
    def test_a_corrupted_implementation_fails_alone(
        self, study: StudyRecord, cell: pd.DataFrame, label: str
    ) -> None:
        mutations = {
            "bias": self._shift_bias,
            "coverage": self._lose_coverage,
            "standard error": self._inflate_standard_errors,
        }
        for implementation in study.implementations:
            mutated = cell.copy()
            mutations[label](mutated, mutated["implementation"] == implementation)
            verdicts = independent_performance_tests(
                mutated, record=_cheap(study), n_jobs=1
            ).set_index("implementation")
            assert not bool(verdicts.loc[implementation, "passed"]), (
                f"a corrupted {label} for {implementation} was accepted"
            )
            for other in set(study.implementations) - {implementation}:
                assert bool(verdicts.loc[other, "passed"]), (
                    f"corrupting {implementation}'s {label} implicated {other}"
                )

    def test_understated_standard_errors_fail_the_calibration_cell_alone(
        self, study: StudyRecord
    ) -> None:
        """The failure the primary study's one-sided coverage floor lets through.

        A reported standard error uniformly 10% too small leaves true coverage at 0.922, which
        clears a 0.90 floor at 1,600 replications about two times in three, and sits inside the
        0.80--1.20 sanity band that :class:`~tests.studies.evidence.registry.Margins` refuses to
        let anyone tighten -- the band is a screen behind the coverage gate by construction.
        The calibration cell is the gate that catches it, so this mutation is what shows the
        cell is load bearing rather than decorative.
        """
        rows = pd.read_csv(study.artifact("property-replicates.csv.gz"))
        mutated = rows.copy()
        mask = mutated["property"] == "interval_calibration"
        assert mask.any(), "the study declares no calibration cell to corrupt"
        mutated.loc[mask, "std_error"] *= 0.90
        summary = study.properties().summarize_properties(mutated).set_index(["property", "cell"])
        published = study.properties().summarize_properties(rows).set_index(["property", "cell"])
        assert not bool(summary.loc[("interval_calibration", "correctly_specified"), "passed"])
        untouched = summary.index.drop(("interval_calibration", "correctly_specified"))
        assert summary.loc[untouched, "passed"].equals(published.loc[untouched, "passed"]), (
            "corrupting the calibration cell moved a verdict somewhere else"
        )

    def test_a_material_subject_regression_fails_similarity_and_noninferiority(
        self, study: StudyRecord, cell: pd.DataFrame
    ) -> None:
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
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
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
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
        runner = study.runner()
        for scenario in study.scenarios:
            frame, truth = runner.draw_scenario(scenario, study.n, replicate)
            refitted = pd.DataFrame(runner.cleverly_rows(frame, truth, scenario, replicate))
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

    def test_each_study_draws_from_the_seed_it_publishes(self, study: StudyRecord) -> None:
        """The manifest's ``seed`` has to be the seed the samples actually came from.

        It was not.  Two studies imported the ordinary-TMLE runner's ready-made
        ``draw_scenario``, which closes over *its* record, so three rows sampled from one seed
        while each published a different one -- and neither of the two could be reproduced
        from what it published.  Nothing failed, because every other check in this module
        reads the committed rows and the runner through the same wrong door.
        """
        runner = study.runner()
        for scenario in study.scenarios:
            drawn, _ = runner.draw_scenario(scenario, study.n, 0)
            # Rebuilt from the published seed through the law's own sampler rather than
            # through the runner again, which is what makes this a check on the seed and not
            # a restatement of whatever the runner did.
            seed = replicate_seed(study, scenario, 0)
            dgp = canonical_tmle.scenario_dgp(scenario)
            if scenario == "continuous":
                expected, _ = canonical_tmle.sample_continuous(dgp, study.n, seed)
            else:
                expected, _ = dgp.sample(study.n, seed=seed, backend="pandas")
            pd.testing.assert_frame_equal(drawn, expected)

    def test_the_registered_studies_do_not_share_their_samples(self) -> None:
        """Three rows drawn from one seed are one experiment reported three times."""
        for scenario in STUDIES[0].scenarios:
            frames = [
                other.runner().draw_scenario(scenario, 200, 0)[0]
                for other in STUDIES
                if scenario in other.scenarios
            ]
            for index, frame in enumerate(frames):
                for other in frames[index + 1 :]:
                    assert not frame.equals(other), (
                        f"two registered studies draw the identical {scenario} sample, so their "
                        f"results are one draw's luck reported twice"
                    )

    def test_the_reference_moved_its_estimates_off_the_plug_in(
        self, study: StudyRecord, rows: pd.DataFrame
    ) -> None:
        """Agreement cannot be explained by neither implementation targeting anything.

        ``tmle3`` reports the pre-targeting plug-in beside the targeted estimate, so at least
        one target has to have moved: an exact-agreement check goes blind precisely where the
        fluctuation is zero.
        """
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
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


GRID = ROOT / "docs" / "methodology.md"

#: The grid's header, in order.  The first study's row split the old single "paper-property
#: study" column in two: 34 of the tests it was credited with are per-implementation
#: performance tests, half of them measuring the R reference, and none of them is a property
#: from the TMLE paper.
GRID_COLUMNS = (
    "method",
    "estimands and intervals",
    "performance vs truth",
    "cross-implementation",
    "scientific properties",
    "limitations",
)

#: Which cell carries which pair of counts, and which quantities they must equal.
COUNTED = {
    "performance vs truth": ("independent_tests_passed", "independent_tests_total"),
    "cross-implementation": ("paired_tests_passed", "paired_tests_total"),
    "scientific properties": ("property_cells_passed", "property_cells_total"),
}

LINK = re.compile(r"\]\(([^)\s]+)\)")
COUNT = re.compile(r"(\d+)/(\d+)")


def _grid() -> dict[str, dict[str, str]]:
    parsed = pipe_table(GRID, GRID_COLUMNS)
    rows = {}
    for row in parsed:
        label = row["method"].split("](", 1)[0].removeprefix("[")
        rows[label] = row
    assert len(rows) == len(parsed), "the grid has a duplicate method"
    return rows


class TestTheMethodEvidenceGrid:
    """The Technical appendix's method grid against the register and committed results."""

    def test_every_registered_study_has_a_row_and_every_row_a_study(self) -> None:
        rows = set(_grid())
        studies = {study.name for study in STUDIES}
        assert rows == studies, (
            f"rows with no registered study {sorted(rows - studies)}, studies with no row "
            f"{sorted(studies - rows)}. A study whose results are committed but unrowed is one "
            f"no reader is routed to"
        )

    def test_the_method_name_points_at_the_registered_studys_page(self, study: StudyRecord) -> None:
        row = _grid()[study.name]
        targets = LINK.findall(row["method"])
        assert len(targets) == 1, (
            f"{study.slug}'s method name should have one page link, found {targets}"
        )
        path, _, anchor = targets[0].partition("#")
        resolved = (GRID.parent / path).resolve()
        assert resolved == study.document_path.resolve(), (
            f"{study.slug}'s method links to {path}, not its registered page {study.document}"
        )
        assert anchor == "", "a dedicated study page should be linked at its page root"

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


class TestThePublishedStudyPages:
    """The dedicated pages are complete, current renderings of the committed artifacts."""

    def test_the_generated_result_block_is_current(self, study: StudyRecord) -> None:
        document = study.document_path.read_text(encoding="utf-8")
        assert generated_results(document) == render_results(study)

    def test_every_registered_property_is_explained(self, study: StudyRecord) -> None:
        document = study.document_path.read_text(encoding="utf-8")
        explanation = document.partition(RESULTS_START)[0]

        def normalized(text: str) -> str:
            return "".join(character for character in text.casefold() if character.isalnum())

        explained = normalized(explanation)
        missing = [name for name in study.property_cells if normalized(name) not in explained]
        assert missing == [], f"{study.slug} does not explain property tests {missing}"

    def test_the_page_has_no_unregistered_result_rows(self, study: StudyRecord) -> None:
        data = load(study)
        expected = {
            "performance": sum(
                len(estimands) * len(study.implementations)
                for estimands in study.scenarios.values()
            ),
            "equivalence": 0 if study.reference is None else study.cells,
            "properties": sum(len(cells) for cells in study.property_cells.values()),
        }
        actual = {family: len(data[family]) for family in expected}
        assert actual == expected
