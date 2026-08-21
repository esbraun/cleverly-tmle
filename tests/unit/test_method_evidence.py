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
from tests.studies import canonical_properties, canonical_tmle, cvtmle_properties
from tests.studies.evidence.claims import (
    describe,
    load,
    matches,
    quantities,
    thresholds,
    value,
)
from tests.studies.evidence.comparison import equivalence
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
        assert published["property_passed"].all(), published.loc[
            ~published["property_passed"]
        ].to_string()

    def test_no_property_row_publishes_another_row_s_verdict(self, study: StudyRecord) -> None:
        """The gate the performance table had and this one did not.

        ``crossfit_overfitting`` computed one scalar from three statements -- two per-row and
        one paired -- and broadcast it across the property.  So ``in_sample_control``, a
        deliberately in-sample fit with 0.65 coverage and an SE ratio of 0.58, published
        ``passed=True`` beside the *cross-fit* arm's rule.  It was not wrong to pass; its own
        rule is that it must understate its spread, and it does.  Nothing published said so,
        which is the defect: a reader could not tell which claim the verdict answered.

        Every row's ``passed`` must therefore follow from that row's own endpoints, and the
        clause that really is about the pair lives in ``property_passed``.
        """
        published = pd.read_csv(study.artifact("properties.csv"))
        control = published["role"] == "control"
        robustness = published["property"] == "double_robustness"

        def verdicts(mask: pd.Series, column: str) -> list[bool]:
            # ``bias_equivalent`` reads back as object, not bool: the rate rows leave it
            # blank, so the column is mixed and ``Series.equals`` would compare dtypes.
            return [bool(value) for value in published.loc[mask, column]]

        assert verdicts(robustness & ~control, "passed") == verdicts(
            robustness & ~control, "bias_equivalent"
        )
        assert verdicts(robustness & control, "passed") == verdicts(
            robustness & control, "bias_discriminated"
        )

        overfitting = published.loc[published["property"] == "crossfit_overfitting"]
        if overfitting.empty:
            pytest.skip("study declares no cross-fit overfitting cells")
        margins = study.margins
        for row in overfitting.itertuples():
            if row.role == "control":
                expected = row.se_ratio_ci_upper <= cvtmle_properties.OVERFIT_SE_CONTROL_CEILING
            else:
                expected = (
                    row.se_ratio_ci_lower >= cvtmle_properties.OVERFIT_SE_FLOOR
                    and row.se_ratio_ci_upper <= margins.se_ratio_sanity[1]
                )
            assert bool(row.passed) is bool(expected), (
                f"{row.cell} publishes passed={row.passed} against its own endpoints"
            )
        # The paired clause belongs to neither row, so it is the one thing both share.
        assert overfitting["property_passed"].nunique() == 1
        assert bool(overfitting["property_passed"].iloc[0]) is bool(
            overfitting["passed"].all()
            and overfitting["coverage_gain_ci_lower"].iloc[0]
            >= cvtmle_properties.OVERFIT_COVERAGE_GAIN
        )


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
            if hasattr(runner, "draw_from_seed"):
                expected, _ = runner.draw_from_seed(scenario, study.n, seed)
            else:
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
        for row in pipe_table(study.document_path, MEASURED_COLUMNS, section=study.anchor):
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
            row["quantity"].strip("`")
            for row in pipe_table(study.document_path, MEASURED_COLUMNS, section=study.anchor)
        }
        assert len(quoted) >= 8, f"only {len(quoted)} quantities are gated: {sorted(quoted)}"
        families = {_family(name) for name in quoted}
        required = {"performance", "properties"}
        if study.reference is not None:
            required.add("equivalence")
        missing = required - families
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


#: Names that summarise no single artefact family: the study's own configuration, and the
#: descriptive counts read off ``summary.csv``.  Listed so that a new aggregate cannot be
#: added without either a family or a deliberate exemption.
_CONFIGURATION_QUANTITIES = frozenset({"replicates", "n"})


def _family(name: str) -> str:
    """Which artefact a quantity name comes from -- the reference form says so directly."""
    if "[" in name:
        return name.split("[", 1)[0]
    # Longest prefix first.  ``min_coverage`` is a prefix of ``min_coverage_difference``, and
    # first-match-wins on insertion order filed every equivalence bound under ``performance``.
    for prefix in sorted(_FAMILY, key=len, reverse=True):
        if name.startswith(prefix):
            return _FAMILY[prefix]
    return "other"


def _exempt(name: str) -> bool:
    """Configuration, a descriptive count of ``summary.csv`` rows, or a declared threshold.

    A ``margin:`` name resolves against the study's *declaration* rather than its artefacts,
    so it belongs to no artefact family by construction and must not be asked to count
    towards artefact coverage.  It is still gated: the quoted-value test resolves it like any
    other name, which is the whole point of naming a threshold instead of retyping it.
    """
    return (
        name in _CONFIGURATION_QUANTITIES
        or name.startswith("margin:")
        or name.endswith("summary_cells")
        or "cells_with_" in name
    )


class TestTheQuantityVocabulary:
    """The name-to-artefact map, which one gate reads and nothing else checks."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("min_coverage", "performance"),
            ("min_coverage_ci_lower", "performance"),
            ("min_coverage_difference_lower", "equivalence"),
            ("properties[power/alternative]:rejection_rate", "properties"),
        ],
    )
    def test_a_longer_prefix_wins_over_a_shorter_one(self, name: str, expected: str) -> None:
        assert _family(name) == expected

    def test_every_declared_threshold_is_the_constant_the_study_applies(
        self, study: StudyRecord
    ) -> None:
        """The vocabulary must resolve to the declaration, not to a second copy of it.

        Before this, the rules were prose.  Moving ``MINIMUM_POWER`` or ``OVERFIT_SE_FLOOR``
        left a page asserting a threshold no study had applied, and the only gate over the
        published rules compared the renderer's literal against the page's -- the same
        literal on both sides.
        """
        declared = thresholds(study)
        margins = study.margins
        assert declared["margin:coverage_floor"] == margins.coverage_floor
        assert declared["margin:se_ratio_sanity_upper"] == margins.se_ratio_sanity[1]
        assert declared["margin:type_i_ceiling"] == margins.alpha + margins.type_i_margin
        assert declared["margin:minimum_power"] == canonical_properties.MINIMUM_POWER
        assert declared["margin:root_n_slope_lower"] == (
            canonical_properties.ROOT_N_SLOPE - canonical_properties.ROOT_N_SLOPE_MARGIN
        )
        if "crossfit_overfitting" in study.property_cells:
            assert declared["margin:overfit_se_floor"] == cvtmle_properties.OVERFIT_SE_FLOOR
            assert (
                declared["margin:overfit_control_ceiling"]
                == cvtmle_properties.OVERFIT_SE_CONTROL_CEILING
            )
        if "selector_necessity" in study.property_cells:
            selector = study.properties()
            assert declared["margin:selector_rmse_ratio"] == selector.SELECTOR_RMSE_RATIO
        # And every one of them resolves through the same entry point a document quotes.
        for name, expected in declared.items():
            assert value(study, name) == expected

    def test_every_declared_threshold_is_published_in_the_study_s_own_table(
        self, study: StudyRecord
    ) -> None:
        """A rule the reader cannot see the number for is a rule they cannot check.

        Quoting the thresholds by name is what puts them under
        ``test_every_quoted_value_is_the_rounding_of_the_computed_one`` and under
        ``document.fill``, so moving a constant moves the published page instead of leaving
        it asserting a rule the study never applied.  Requiring every declared threshold to
        appear stops a new one from being added and quietly never shown.
        """
        quoted = {
            row["quantity"].strip("`")
            for row in pipe_table(study.document_path, MEASURED_COLUMNS, section=study.anchor)
        }
        missing = sorted(set(thresholds(study)) - quoted)
        assert missing == [], (
            f"{study.slug} declares {missing} but its section quotes none of them, so the "
            f"page states rules whose numbers nothing checks"
        )

    def test_every_declared_quantity_resolves_to_an_artefact_or_is_exempt(
        self, study: StudyRecord
    ) -> None:
        """A quantity with no family is invisible to the table-coverage gate below."""
        unresolved = [
            name for name in quantities(study) if _family(name) == "other" and not _exempt(name)
        ]
        assert unresolved == [], (
            f"{sorted(unresolved)} resolve to no artefact family, so quoting one of them would "
            f"count towards nothing in test_the_table_reaches_every_family_of_result"
        )
