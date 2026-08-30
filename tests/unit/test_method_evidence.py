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
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.documents import pipe_table
from tests.studies.evidence import descriptions, property_verdicts
from tests.studies.evidence.claims import load, quantities, thresholds, value
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.document import (
    _CLOSE,
    _OPEN,
    ACCURACY_COLUMNS,
    AGREEMENT_COLUMNS,
    GENERATED,
    PROPERTY_COLUMNS,
    _section,
    render,
)
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

#: Where a result-neutral edit to a hashed reference source is declared, instead of being hidden
#: by rewriting the manifest that recorded the bytes which actually ran.
REVISIONS = ROOT / "tests" / "canonical" / "provenance-revisions.md"
REVISION_COLUMNS = ("source", "recorded", "current", "judgement")


def _revisions() -> dict[tuple[str, str], str]:
    """``(source, hash a manifest recorded) -> hash the file carries now``, from the ledger.

    Keyed on the *recorded* hash rather than on the source alone, so one source may be judged
    more than once as different studies are regenerated at different times.  The shared
    ``drtmle`` context is the case: one study records the edited Dockerfile because it was
    regenerated after the edit, and the other still records the Dockerfile that built its image.
    """
    table: dict[tuple[str, str], str] = {}
    for row in pipe_table(REVISIONS, REVISION_COLUMNS):
        key = (row["source"].strip("`"), row["recorded"].strip("`"))
        assert key not in table, f"{key} carries two judgements; a source has one or none"
        assert row["judgement"].startswith("result-neutral:"), (
            f"{key} is declared without saying what kind of claim it makes. A judgement that "
            f"is not 'result-neutral: <reason>' is a regeneration, not a declaration"
        )
        table[key] = row["current"].strip("`")
    return table


def test_no_provenance_revision_outlives_the_manifest_it_explains() -> None:
    """The ledger refuses a stale row, on the terms ``accepted_reference_failure`` already sets.

    A declaration explains a difference between one recorded hash and the working tree.  Once
    every study that recorded that hash has been regenerated, there is no difference left and
    the row explains nothing; carried anyway, it would read as a standing exemption for the
    source rather than as a judgement about one edit.
    """
    recorded: dict[str, set[str]] = {}
    for study in STUDIES:
        manifest = json.loads(study.artifact("manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["reference_sha256"].items():
            recorded.setdefault(name, set()).add(digest)
    for (name, digest), current in _revisions().items():
        assert digest in recorded.get(name, set()), (
            f"no manifest records {name} at {digest[:12]} any more, so this row explains "
            f"nothing. Remove it rather than carrying a stale exception"
        )
        source = ROOT / name
        assert source.exists(), f"{name} is judged by the ledger but is gone"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == current, (
            f"{name} changed again after it was judged; re-read the edit and restate the "
            f"judgement against the bytes that are committed now"
        )


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
        } | set(study.extra_artifacts)
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(study.artifact(name).read_bytes()).hexdigest() == digest
        # Reference sources are keyed from the repository root rather than from the study
        # directory, because two studies may share one Docker context and one adapter.
        for name, digest in manifest["reference_sha256"].items():
            source = ROOT / name
            assert source.exists(), f"{name} is hashed by the manifest but is gone"
            current = hashlib.sha256(source.read_bytes()).hexdigest()
            if current == digest:
                continue
            # A moved reference source is one of two things, and the gate has to make the
            # maintainer say which.  A result-determining edit means the artefacts are stale
            # and the study is regenerated.  A result-neutral edit is declared in the ledger.
            # Editing `digest` in the manifest is the third option, and it is the one that
            # leaves the manifest claiming that bytes which never ran produced the result.
            declared = _revisions().get((name, digest))
            assert declared is not None, (
                f"{name} no longer hashes to what {study.slug} recorded. Regenerate the study, "
                f"or record the judgement in {REVISIONS.name} rather than editing the manifest"
            )
            assert declared == current, (
                f"{name} changed again after its judgement in {REVISIONS.name}, which explains "
                f"{declared[:12]} and not {current[:12]}"
            )
        configuration = manifest["configuration"]
        assert configuration["replicates"] == study.replicates
        assert configuration["n"] == study.n
        assert configuration["seed"] == study.seed
        assert configuration["publication_policy"] == study.publication_policy
        assert configuration["margins"] == study.margins.as_json()
        if study.accepted_reference_failure:
            assert configuration["accepted_reference_failure"] == study.accepted_reference_failure
        else:
            assert "accepted_reference_failure" not in configuration

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
        if study.publication_policy == "gated":
            failures = published.loc[~published["passed"]]
            if study.accepted_reference_failure:
                failures = failures.loc[failures["implementation"] != study.reference]
            assert failures.empty, failures.to_string()

    def test_the_subject_is_similar_to_and_no_worse_than_the_reference(
        self, study: StudyRecord
    ) -> None:
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
        published = pd.read_csv(study.artifact("equivalence.csv"))
        assert published["dropped_replications"].eq(0).all()
        assert published["passed"].equals(
            (published["paired_similarity"] & published["subject_not_inferior"])
            | published["coverage_superior"]
        ), "the published verdict is not an equivalence or superiority conclusion"
        expected = np.select(
            [
                published["coverage_superior"],
                published["paired_similarity"] & published["subject_not_inferior"],
                # A cell whose calibration bound is wider than the calibration margin could
                # not have concluded non-inferiority at any true excess, so its failure is a
                # statement about the design and is named apart from an unsettled one.
                ~published["calibration_resolved"],
            ],
            ["superior", "equivalent", "underpowered"],
            default="inconclusive",
        )
        assert published["comparison_conclusion"].tolist() == expected.tolist()
        assert published["calibration_resolved"].equals(
            ~published["se_comparable"]
            | (
                published["calibration_excess_resolution"]
                <= published["calibration_noninferiority_margin"]
            )
        ), "the resolution flag is not the published width against the published margin"
        if study.publication_policy == "gated":
            assert published["passed"].all(), published.loc[~published["passed"]]

    def test_the_reference_is_reported_on_its_own_terms(self, study: StudyRecord) -> None:
        """A reference that degrades is a reference finding, not a subject failure.

        The subject is required to be valid either way.  The reference is required to be
        valid unless the study declared in advance that it is not, which is the whole content
        of ``accepted_reference_failure``: a comparator can be worth comparing against on
        agreement and non-inferiority while failing its own truth gates, and the alternative
        the framework offered was deleting it.  The declaration is checked against the rows,
        so a study cannot carry a reason for a failure it no longer has.
        """
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
        published = pd.read_csv(study.artifact("equivalence.csv"))
        if study.publication_policy == "reporting":
            return
        assert published["subject_valid"].all(), published.loc[~published["subject_valid"]]
        if not study.accepted_reference_failure:
            assert published["reference_valid"].all(), published.loc[~published["reference_valid"]]
            return
        assert not published["reference_valid"].all(), (
            f"{study.slug} declares an accepted reference failure but every reference row "
            f"is valid; the declaration is stale"
        )
        assert published["passed"].all(), (
            "an accepted reference failure must not relax the subject's own paired verdict"
        )

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
        if study.publication_policy == "gated":
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

        def verdicts(mask: pd.Series, column: str) -> list[bool]:
            # ``bias_equivalent`` reads back as object, not bool: the rate rows leave it
            # blank, so the column is mixed and ``Series.equals`` would compare dtypes.
            return [bool(value) for value in published.loc[mask, column]]

        families = set(study.property_cells)
        unclassified = sorted(families - BIAS_GATED_PROPERTIES - ENDPOINT_GATED_PROPERTIES)
        assert unclassified == [], (
            f"{unclassified} declare cells but no per-row rule this test knows how to check. "
            f"A family arriving without one is how a control came to publish the positive "
            f"arm's verdict; classify it above rather than leaving it ungated"
        )
        for family in sorted(families & BIAS_GATED_PROPERTIES):
            rows = published["property"] == family
            assert verdicts(rows & ~control, "passed") == verdicts(
                rows & ~control, "bias_equivalent"
            ), f"{family}: a positive row's verdict is not its own bias-equivalence endpoint"
            assert verdicts(rows & control, "passed") == verdicts(
                rows & control, "bias_discriminated"
            ), f"{family}: a control's verdict is not its own discrimination endpoint"

        # Two clauses, and the second is the one a bias-gated rule cannot state.  Each row
        # keeps its own direction on the bias, and every row of the family must also have
        # reported an error on the scale of its own spread -- otherwise the fit collapsed and
        # the bias beside it is evidence about the collapse, not about the union model.
        low, high = property_verdicts.UNION_MODEL_SE_BAND
        union = published.loc[published["property"] == "double_robustness"]
        for row in union.itertuples():
            endpoint = row.bias_discriminated if row.role == "control" else row.bias_equivalent
            expected = bool(endpoint) and low <= row.se_ratio <= high
            assert bool(row.passed) is expected, (
                f"{row.cell} publishes passed={row.passed} against its own union-model "
                f"endpoints: bias {endpoint}, SE ratio {row.se_ratio}"
            )

        root_n = published.loc[published["property"] == "root_n_and_efficiency"]
        for row in root_n.itertuples():
            expected = (
                # Resolved on either side, which is the rule ``apply_shared_verdicts`` applies:
                # a control size established as sub-nominal is a published limitation and one
                # established as adequate is an improvement.  Only a straddling interval says
                # nothing, and only that fails.
                (
                    row.coverage_ci_upper < 1.0 - study.margins.alpha
                    or row.coverage_ci_lower >= study.margins.coverage_floor
                )
                if row.role == "control"
                else (
                    bool(row.bias_equivalent)
                    and row.coverage_ci_lower >= study.margins.coverage_floor
                    and study.margins.se_ratio_sanity[0]
                    <= row.se_ratio
                    <= study.margins.se_ratio_sanity[1]
                )
            )
            assert bool(row.passed) is bool(expected), (
                f"{row.cell} publishes passed={row.passed} against its own root-n endpoint"
            )

        # A ladder rung answers to coverage and a rate row to its fitted slope, and the two
        # roles answer in opposite directions.  Checked here rather than left to the family's
        # classification above: listing a family as endpoint-gated says only that its rule is
        # not bias equivalence, and a family with no check of its own is the state the
        # ``unclassified`` assertion exists to prevent one step earlier.
        contraction = published.loc[published["property"] == "double_robust_contraction"]
        for row in contraction.itertuples():
            if str(row.cell).startswith("rate_"):
                contracts = row.slope_ci_upper < 0.0
                expected = (not contracts) if row.role == "control" else contracts
            elif row.role == "control":
                expected = row.coverage_ci_upper < study.margins.coverage_floor
            else:
                expected = row.coverage_ci_lower >= study.margins.coverage_floor
            assert bool(row.passed) is bool(expected), (
                f"{row.cell} publishes passed={row.passed} against its own contraction endpoint"
            )

        calibration = published.loc[published["property"] == "interval_calibration"]
        if (
            "efficiency_empirical_ci_lower" in calibration.columns
            and calibration["efficiency_empirical_ci_lower"].notna().any()
        ):
            properties = study.properties()
            for row in calibration.itertuples():
                kind = row.cell.split("__", 1)[1]
                if kind == "correctly_specified":
                    expected = (
                        study.margins.calibration_se_ratio[0]
                        <= row.se_ratio_ci_lower
                        <= row.se_ratio_ci_upper
                        <= study.margins.calibration_se_ratio[1]
                        and study.margins.calibration_coverage[0]
                        <= row.coverage_ci_lower
                        <= row.coverage_ci_upper
                        <= study.margins.calibration_coverage[1]
                        and properties.EFFICIENCY_RATIO_BAND[0]
                        <= row.efficiency_empirical_ci_lower
                        <= row.efficiency_empirical_ci_upper
                        <= properties.EFFICIENCY_RATIO_BAND[1]
                        and properties.EFFICIENCY_RATIO_BAND[0]
                        <= row.efficiency_reported_ci_lower
                        <= row.efficiency_reported_ci_upper
                        <= properties.EFFICIENCY_RATIO_BAND[1]
                    )
                elif kind == "shrunken_se_control":
                    expected = row.se_ratio_ci_upper < study.margins.calibration_se_ratio[0]
                else:
                    expected = (
                        row.efficiency_empirical_ci_lower > properties.EFFICIENCY_RATIO_BAND[1]
                    )
                assert bool(row.passed) is bool(expected), (
                    f"{row.cell} publishes passed={row.passed} against its calibration endpoint"
                )

        corrected = published.loc[published["property"] == "corrected_mar_inference"]
        for row in corrected.itertuples():
            expected = (
                bool(row.bias_discriminated)
                if row.role == "control"
                else (
                    bool(row.bias_equivalent)
                    and row.coverage_ci_lower >= study.margins.coverage_floor
                    and study.margins.se_ratio_sanity[0]
                    <= row.se_ratio
                    <= study.margins.se_ratio_sanity[1]
                )
            )
            assert bool(row.passed) is bool(expected), (
                f"{row.cell} publishes passed={row.passed} against its corrected-MAR endpoint"
            )

        correction = published.loc[published["property"] == "correction_necessity"]
        if not correction.empty:
            properties = study.properties()
            initial = correction.loc[
                correction["cell"].str.endswith("__initial_score_control")
            ].iloc[0]
            for row in correction.itertuples():
                expected = (
                    row.bias_ci_upper <= properties.CORRECTION_SCORE_RATIO * initial.bias_ci_lower
                    if row.cell.endswith("__closed_score")
                    else row.bias_ci_lower >= properties.UNCORRECTED_SCORE_FLOOR
                )
                assert bool(row.passed) is bool(expected), (
                    f"{row.cell} publishes passed={row.passed} against its correction-score endpoint"
                )

        necessity = published.loc[published["property"] == "selector_necessity"]
        if not necessity.empty:
            # The RMSE comparison belongs to neither row, so it is the one thing both share.
            assert necessity["property_passed"].nunique() == 1
            assert bool(necessity["property_passed"].iloc[0]) is bool(
                necessity["passed"].all()
                and necessity["rmse_ratio"].iloc[0] <= study.properties().SELECTOR_RMSE_RATIO
            )

        targeting = published.loc[published["property"] == "targeting_necessity"]
        if not targeting.empty:
            # The displacement is a statement about the *pair* -- how far the fluctuation moved
            # the estimate -- so like the selector's RMSE ratio it belongs to neither row and
            # is the one thing both carry.  Without it, an estimator whose targeting step did
            # nothing would pass: the untargeted arm would sit on the truth beside the targeted
            # one, and both rows' own bias endpoints would be satisfied.
            assert targeting["property_passed"].nunique() == 1
            assert bool(targeting["property_passed"].iloc[0]) is bool(
                targeting["passed"].all()
                and targeting["targeting_displacement"].iloc[0]
                >= study.properties().TARGETING_DISPLACEMENT
            )

        missingness = published.loc[published["property"] == "missingness_necessity"]
        if not missingness.empty:
            assert missingness["property_passed"].nunique() == 1
            assert bool(missingness["property_passed"].iloc[0]) is bool(
                missingness["passed"].all()
                and missingness["missingness_displacement"].iloc[0]
                >= study.properties().MISSINGNESS_DISPLACEMENT
            )

        projection = published.loc[published["property"] == "projection_necessity"]
        if not projection.empty:
            assert projection["property_passed"].nunique() == 1
            assert bool(projection["property_passed"].iloc[0]) is bool(
                projection["passed"].all()
                and projection["projection_displacement"].iloc[0]
                >= study.properties().PROJECTION_DISPLACEMENT
            )

        weighting = published.loc[published["property"] == "weight_necessity"]
        if not weighting.empty:
            control = weighting.loc[weighting["role"] == "control"]
            assert len(control) == 1
            assert bool(control["alternative_bias_equivalent"].iloc[0])
            assert weighting["property_passed"].nunique() == 1
            assert bool(weighting["property_passed"].iloc[0]) is bool(
                weighting["passed"].all()
                and control["alternative_bias_equivalent"].all()
                and weighting["necessity_displacement"].iloc[0]
                >= study.properties().WEIGHT_DISPLACEMENT
            )

        learner_weighting = published.loc[published["property"] == "learner_weight_necessity"]
        if not learner_weighting.empty:
            control = learner_weighting.loc[learner_weighting["role"] == "control"]
            assert len(control) == 1
            assert bool(control["alternative_bias_equivalent"].iloc[0])
            assert learner_weighting["property_passed"].nunique() == 1
            assert bool(learner_weighting["property_passed"].iloc[0]) is bool(
                learner_weighting["passed"].all()
                and control["alternative_bias_equivalent"].all()
                and learner_weighting["necessity_displacement"].iloc[0]
                >= study.properties().WEIGHT_DISPLACEMENT
            )

        recursion = published.loc[
            published["property"].isin(
                {"survival_recursion_necessity", "competing_risk_recursion_necessity"}
            )
        ]
        if not recursion.empty:
            # The same shape as targeting above, and needed for the same reason: each row's own
            # bias endpoint is satisfied by a recursion that does nothing, because the
            # survivor-only arm would then be the estimate.  The joint clause is the
            # displacement, and it was ungated when the family arrived -- classifying the family
            # as bias-gated checks the rows and says nothing about the claim over the pair.
            assert recursion["property_passed"].nunique() == 1
            assert bool(recursion["property_passed"].iloc[0]) is bool(
                recursion["passed"].all()
                and recursion["recursion_displacement"].iloc[0]
                >= study.properties().RECURSION_DISPLACEMENT
            )

        design = published.loc[published["property"] == "generated_design"]
        if not design.empty:
            margins = study.margins
            for row in design.itertuples():
                if row.cell == "oracle_design":
                    expected = (
                        margins.calibration_se_ratio[0]
                        <= row.se_ratio_ci_lower
                        <= row.se_ratio_ci_upper
                        <= margins.calibration_se_ratio[1]
                    )
                else:
                    expected = (
                        row.se_ratio_deficit_upper <= -study.properties().GENERATED_DESIGN_DEFICIT
                    )
                assert bool(row.passed) is bool(expected), (
                    f"{row.cell} publishes passed={row.passed} against its own endpoints"
                )
            assert design["property_passed"].nunique() == 1
            assert bool(design["property_passed"].iloc[0]) is bool(design["passed"].all())

        clustered = published.loc[published["property"] == "clustered_inference"]
        if not clustered.empty:
            margins = study.margins
            for row in clustered.itertuples():
                if row.cell == "cluster_robust":
                    expected = (
                        margins.calibration_se_ratio[0]
                        <= row.se_ratio_ci_lower
                        <= row.se_ratio_ci_upper
                        <= margins.calibration_se_ratio[1]
                        and margins.calibration_coverage[0]
                        <= row.coverage_ci_lower
                        <= row.coverage_ci_upper
                        <= margins.calibration_coverage[1]
                    )
                else:
                    expected = (
                        row.se_ratio_ci_upper <= property_verdicts.CLUSTER_ROBUST_CONTROL_SE_CEILING
                    )
                assert bool(row.passed) is bool(expected), (
                    f"{row.cell} publishes passed={row.passed} against its clustered-inference endpoints"
                )
            assert clustered["property_passed"].nunique() == 1
            assert bool(clustered["property_passed"].iloc[0]) is bool(
                clustered["passed"].all()
                and clustered["coverage_gain_ci_lower"].iloc[0]
                >= property_verdicts.CLUSTERED_COVERAGE_GAIN
            )

        overfitting = published.loc[published["property"] == "crossfit_overfitting"]
        if overfitting.empty:
            pytest.skip("study declares no cross-fit overfitting cells")
        margins = study.margins
        for row in overfitting.itertuples():
            if row.role == "control":
                expected = row.se_ratio_ci_upper <= property_verdicts.OVERFIT_SE_CONTROL_CEILING
            else:
                expected = (
                    row.se_ratio_ci_lower >= property_verdicts.OVERFIT_SE_FLOOR
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
            >= property_verdicts.OVERFIT_COVERAGE_GAIN
        )


#: Property families whose per-row verdict is the bias claim read in both directions: a
#: positive row's equivalence interval inside the margin, a control's outside it.
BIAS_GATED_PROPERTIES = frozenset(
    {
        "categorical_probability_necessity",
        "mechanism_requirement",
        "cap_necessity",
        "density_necessity",
        "learner_weight_necessity",
        "mar_robustness",
        "missingness_necessity",
        "robustness_contract",
        "selector_necessity",
        "competing_risk_recursion_necessity",
        "survival_recursion_necessity",
        "targeting_necessity",
        "weight_necessity",
        "projection_necessity",
        "ratio_necessity",
        "rule_necessity",
    }
)

#: Families whose rows answer to other endpoints -- coverage, an SE ratio, a rejection rate,
#: a fitted slope.  Listed rather than inferred so that a family added to a study has to be
#: classified here before its verdicts count.
ENDPOINT_GATED_PROPERTIES = frozenset(
    {
        "clustered_inference",
        "crossfit_overfitting",
        "corrected_mar_inference",
        "correction_necessity",
        # Deliberately *not* bias-gated, though its cells carry a bias.  Its ladder rungs
        # answer to coverage and its rate rows to a fitted slope, because the level claim
        # about the bias is ``double_robustness``'s and repeating it at three more sizes
        # would publish one red cell four times without adding a statement.
        "double_robust_contraction",
        # Bias in both directions *and* the union-model SE screen, so it is no longer the
        # one-endpoint family this list's siblings are.  Listing it as bias-gated is what let
        # a control with a reported error 87.6 times its empirical spread publish a pass: the
        # gate asserted ``passed == bias_discriminated`` exactly, so the framework held the
        # gap in place rather than merely missing it.
        "double_robustness",
        "generated_design",
        "interval_calibration",
        "natural_course_identity",
        "power",
        "root_n_and_efficiency",
        "root_n_rate",
        "static_reduction",
        "treatment_score_necessity",
        "type_i_error",
    }
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
        published = study.properties().summarize_properties(rows).set_index(["property", "cell"])
        family = (
            "interval_calibration"
            if "interval_calibration" in study.property_cells
            else "clustered_inference"
        )
        calibration = published.loc[published.index.get_level_values(0) == family]
        positive_cells = set(
            calibration.loc[calibration["role"] == "positive"].index.get_level_values(1)
        )
        mutated = rows.copy()
        mask = (mutated["property"] == family) & mutated["cell"].isin(positive_cells)
        assert mask.any(), "the study declares no calibration cell to corrupt"
        mutated.loc[mask, "std_error"] *= 0.90
        summary = study.properties().summarize_properties(mutated).set_index(["property", "cell"])
        changed = [(family, cell) for cell in positive_cells]
        assert not summary.loc[changed, "passed"].any()
        untouched = summary.index.drop(changed)
        assert summary.loc[untouched, "passed"].equals(published.loc[untouched, "passed"]), (
            "corrupting the calibration cell moved a verdict somewhere else"
        )

    def test_a_collapsed_reported_error_fails_the_union_model_cells(
        self, study: StudyRecord
    ) -> None:
        """The failure a bias-only union-model verdict let through.

        A ``double_robustness`` cell fits at least one nuisance wrong on purpose, so nothing
        in the run guarantees the fit stayed well posed.  A univariate guard regression handed
        a constant single regressor reported a standard error 87.6 times its empirical spread,
        and the ``both_wrong`` control passed on its bias endpoint alone, because that endpoint
        is standardized by the *empirical* spread and never reads what the fit reported.

        :data:`~tests.studies.evidence.property_verdicts.UNION_MODEL_SE_BAND` is the screen,
        and it binds on no committed cell -- the family spans 0.61 to 2.31, which is the union
        model behaving as the theory allows.  A rule nothing can fail reads exactly like a rule
        nothing has broken, so this mutation is what makes it load bearing.

        Scaling ``std_error`` moves the reported error and nothing else: ``covered`` and
        ``rejected`` travel on the committed rows, and the bias endpoints are computed from the
        estimates.  So the mutation isolates the clause, and the untouched assertion below
        shows it reaches no other verdict.
        """
        if "double_robustness" not in study.property_cells:
            pytest.skip("the study declares no union-model cells")
        rows = pd.read_csv(study.artifact("property-replicates.csv.gz"))
        published = study.properties().summarize_properties(rows).set_index(["property", "cell"])
        union = published.loc[published.index.get_level_values(0) == "double_robustness"]
        # At least one, not all of them: two registered rows publish a red union-model cell
        # under reporting policy, and a mutation whose cells were already failing would show
        # nothing.
        assert union["passed"].any(), "no union-model cell currently passes, so nothing can flip"

        mutated = rows.copy()
        mask = mutated["property"] == "double_robustness"
        # Two orders of magnitude, which is the scale the collapse actually reached.  Every
        # committed cell sits below 2.31, so this clears the band's upper limit from every one
        # of them without depending on where any single cell started.
        mutated.loc[mask, "std_error"] *= 100.0
        summary = study.properties().summarize_properties(mutated).set_index(["property", "cell"])
        changed = list(union.index)
        assert not summary.loc[changed, "passed"].any(), (
            "a reported error 100 times the empirical spread still publishes a union-model pass"
        )
        assert summary.loc[changed, "bias_equivalent"].equals(
            published.loc[changed, "bias_equivalent"]
        ), "the mutation moved a bias endpoint, so it does not isolate the SE screen"
        untouched = summary.index.drop(changed)
        assert summary.loc[untouched, "passed"].equals(published.loc[untouched, "passed"]), (
            "corrupting the union-model cells moved a verdict somewhere else"
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
            # ``n`` and ``initial_estimate`` are published columns like any other, and this
            # gate used to compare neither.  A shared row builder that unified either one --
            # ``len(frame)`` against ``result.n``, or a plug-in against ``math.nan`` -- would
            # have moved a published number with nothing anywhere to notice.  ``nan_ok``
            # because ``canonical_tmle``, ``canonical_cvtmle``, ``canonical_ctmle_oat`` and
            # ``canonical_ctmle_selector`` write ``math.nan`` into ``initial_estimate``
            # deliberately: they report no plug-in.  Six published rows carry it, because
            # the two fold-evaluated and repeated rows read the cross-fitted builder.
            for column in (
                "n",
                "estimate",
                "std_error",
                "ci_lower",
                "ci_upper",
                "initial_estimate",
            ):
                # Four orders of magnitude tighter than the narrowest margin any verdict
                # uses, and loose enough for the last bits of a different BLAS.
                assert merged[f"{column}_refitted"].to_numpy() == pytest.approx(
                    merged[f"{column}_published"].to_numpy(), rel=1e-6, abs=1e-9, nan_ok=True
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
            # Through the runner, never through a law this test picked: a fallback here
            # would audit a study against *this module's* scenarios rather than its own,
            # and a study whose laws differ would pass by being redirected.
            expected, _ = runner.draw_from_seed(scenario, study.n, seed)
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
        """A floor: somewhere in the file, the reference's fluctuation was not a no-op.

        ``tmle3`` reports the pre-targeting plug-in beside the targeted estimate, so at least
        one target has to have moved: an exact-agreement check goes blind precisely where the
        fluctuation is zero.

        **What this is not** is a per-replication guarantee, and it cannot be made into one.
        Targeting is legitimately inert for whole cells of a registered study: a C-TMLE
        selector that chooses the empty path does not fluctuate at all, and the median
        ``|estimate - initial| / std_error`` is exactly ``0`` for ``canonical-ctmle-selector``
        and for ``canonical-tmle``'s least-moved estimand.  Requiring a fixed replication to
        move, at any absolute or relative threshold, fails those studies for behaving
        correctly.  It is also blind to *how far*: the longitudinal report's whole file clears
        this by a factor of two.

        So the strength of each study's witness is published rather than gated here --
        ``max_targeting_displacement`` and ``median_targeting_displacement`` put it in the
        measured table, and a study whose comparison cannot separate an untargeted plug-in
        carries a ``targeting_necessity`` family that can.

        **A one-step comparator has no plug-in to move off.**  ``npcausal``'s ``ipsi`` adds the
        correction to an average of influence values and never forms an untargeted estimate, so
        there is no second number for its runner to publish.  Writing the estimate twice would
        answer this check with a tautology, so the runner writes nothing and this check requires
        the study to carry the family that can separate the two instead.  The distinction is the
        point: an absent column says the quantity does not exist, and a repeated one would say it
        exists and did not move.
        """
        if study.reference is None:
            pytest.skip("study declares no comparison implementation")
        reference = rows.loc[rows["implementation"] == study.reference]
        if reference["initial_estimate"].isna().all():
            assert "targeting_necessity" in study.property_cells, (
                f"{study.slug} publishes no plug-in for {study.reference} and declares no "
                f"targeting_necessity family, so nothing separates targeting from a plug-in"
            )
            pytest.skip(f"{study.reference} publishes no untargeted plug-in")
        assert reference["initial_estimate"].notna().all(), (
            "the reference publishes a plug-in for some replications and not others"
        )
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


GRID = ROOT / "docs" / "technical-reference" / "method-evidence" / "validation-grid.md"

#: The grid's header, in order.  The first study's row split a single "paper-property study"
#: column in two: 34 of the tests it was credited with are per-implementation performance
#: tests, half of them measuring the R reference, and none of them is a property from the TMLE
#: paper.  Every column name says whose implementation its cell is about, because the grid's
#: subject is ``cleverly`` and a reader has to be able to see that without opening a study.
GRID_COLUMNS = (
    "method in `cleverly`",
    "canonical implementation compared",
    "estimands validated",
    "accuracy vs known truth",
    "`cleverly` vs canonical",
    "theory properties",
    "study design and headline results",
    "limitations",
)

#: Which cell carries which pair of counts, and which quantities they must equal.
COUNTED = {
    "accuracy vs known truth": ("independent_tests_passed", "independent_tests_total"),
    "`cleverly` vs canonical": ("paired_tests_passed", "paired_tests_total"),
    "theory properties": ("property_cells_passed", "property_cells_total"),
}

#: Quantities the headline cell must publish, and the label each is written behind.  Checked
#: by containment of the rendered value, so an edited number fails rather than drifting.
HEADLINE = {
    "worst standardized bias": "max_standardized_bias",
    "lowest coverage": "min_coverage",
}

MEASURED_COLUMNS = ("quantity", "value", "source")

LINK = re.compile(r"\]\(([^)\s]+)\)")
COUNT = re.compile(r"(\d+)/(\d+)")


def _grid() -> dict[str, dict[str, str]]:
    rows = {row["method in `cleverly`"]: row for row in pipe_table(GRID, GRID_COLUMNS)}
    assert len(rows) == len(pipe_table(GRID, GRID_COLUMNS)), "the grid has a duplicate method"
    return rows


class TestTheMethodEvidenceGrid:
    """The implementation validation grid against the register and committed results.

    The grid arrived as a note: nothing read it, its counts were typed, and one of them counted
    the wrong study.  These tests are what make it a gate instead, and they are written against
    the register rather than against any one row, so the next method to be added inherits them.
    """

    def test_every_registered_study_has_a_row_and_every_row_a_study(self) -> None:
        rows = set(_grid())
        studies = {study.name for study in STUDIES}
        assert rows == studies, (
            f"rows with no registered study {sorted(rows - studies)}, studies with no row "
            f"{sorted(studies - rows)}. A study whose results are committed but unrowed is one "
            f"no reader is routed to"
        )

    def test_the_row_points_at_the_registered_studys_document(self, study: StudyRecord) -> None:
        """Every link in a row reaches that study's own page, and the row links somewhere.

        This replaced an anchor check.  Each study used to be one ``##`` section of a shared
        document, so "the row cites its own section" needed the fragment to be read.  A study
        is now its own page, which makes the path alone the stronger statement: a row that
        reached the wrong study would fail here whether or not it carried a fragment.
        """
        row = _grid()[study.name]
        targets = [target for cell in row.values() for target in LINK.findall(cell)]
        assert targets, (
            "the row links to nothing, so it stands in for the run rather than citing it"
        )
        for target in targets:
            path, _, _ = target.partition("#")
            resolved = (GRID.parent / path).resolve()
            assert resolved == study.document_path.resolve(), (
                f"{study.slug}'s row links to {path}, not to its registered document "
                f"{study.document}"
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

    def test_confidence_level_is_described_as_a_bound_not_a_pass_rate(
        self, study: StudyRecord
    ) -> None:
        for column in ("accuracy vs known truth", "`cleverly` vs canonical"):
            cell = _grid()[study.name][column]
            if "0/0" in cell:
                continue
            assert "using 99% confidence bounds" in cell
            assert "pass at 99%" not in cell

    def test_efficiency_claim_requires_an_independent_bound(self, study: StudyRecord) -> None:
        properties = load(study)["properties"]
        columns = [name for name in properties if name.startswith("efficiency_")]
        has_bound = bool(columns) and bool(properties[columns].notna().any().any())
        claim = "efficien" in _grid()[study.name]["theory properties"].casefold()
        assert claim == has_bound, (
            f"{study.slug}'s grid efficiency claim is {claim}, but its committed properties "
            f"carry an independent efficiency comparison={has_bound}"
        )

    def test_the_canonical_implementation_cell_names_the_registered_reference(
        self, study: StudyRecord
    ) -> None:
        """The cell has to identify the comparator by its pin, not by its name.

        A package name alone dates: ``tmle3`` has meant several different implementations.  What
        makes the row reproducible is the commit the container actually built, so every pin the
        manifest recorded must appear in the cell -- and a study with no comparator has to say so
        rather than leaving the column to be read as an oversight.
        """
        cell = _grid()[study.name]["canonical implementation compared"]
        reference = json.loads(study.artifact("manifest.json").read_text(encoding="utf-8"))[
            "generated_with"
        ].get("reference")
        if study.reference is None:
            assert reference is None, "the record claims no comparator but the manifest names one"
            assert "none claimed" in cell.casefold(), (
                f"{study.slug} has no comparator; its cell must say so rather than being vague"
            )
            return
        assert reference is not None, "the record names a comparator the manifest does not"
        pins = {
            key: str(recorded)
            for key, recorded in reference.items()
            if key.endswith("_commit") or key.endswith("_version")
        }
        assert pins, f"{study.slug}'s manifest records no comparator pin to check the cell against"
        missing = [
            f"{key}={recorded}"
            for key, recorded in pins.items()
            if (recorded[:7] if key.endswith("_commit") else recorded) not in cell
        ]
        assert missing == [], (
            f"{study.slug}'s cell does not name {missing}. A reader cannot rebuild the comparison "
            f"from a package name alone"
        )

    def test_the_headline_results_are_the_committed_ones(self, study: StudyRecord) -> None:
        """The one grid column carrying measured numbers rather than verdicts."""
        cell = _grid()[study.name]["study design and headline results"]
        data = load(study)
        missing = [
            f"{label} {render(value(study, quantity, data))}"
            for label, quantity in HEADLINE.items()
            if f"{label} {render(value(study, quantity, data))}" not in cell
        ]
        for shown in (f"{study.replicates:,}", f"{study.n:,}"):
            if shown not in cell:
                missing.append(shown)
        assert missing == [], (
            f"{study.slug}'s headline cell does not publish {missing}. Every number here is "
            f"derived, so an edited one is a failure rather than a stale reading"
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

    def test_every_quoted_value_is_the_one_the_generator_writes(self, study: StudyRecord) -> None:
        """Not merely *a* rounding of the computed value, but the one ``fill`` would write.

        ``claims.matches`` accepts any correct rounding at the precision printed.  That is the
        right rule for reading a published table and the wrong one for gating a generated one,
        and the difference is not academic.  Seven digits typed into this study's bound rows
        satisfied it while ``python -m tests.studies.evidence.document`` rewrote the same rows
        to four.  The gate was green, the committed document was not what the generator
        produces, and the next person to run the generator got a diff nobody asked for.
        Precision is the generator's decision alone now, and a study whose claim needs a
        longer one declares it in ``quoted_decimals`` rather than typing it into the table.
        """
        data = load(study)
        wrong = []
        quoted = set()
        for row in pipe_table(study.document_path, MEASURED_COLUMNS, section=study.anchor):
            name = row["quantity"].strip("`")
            quoted.add(name)
            written = render(value(study, name, data), study.quoted_decimals.get(name))
            if row["value"] != written:
                wrong.append(f"{name}: quoted {row['value']}, generator writes {written}")
        assert wrong == [], (
            "the measured table is not what `python -m tests.studies.evidence.document` "
            "writes:\n  " + "\n  ".join(wrong)
        )
        stale = sorted(set(study.quoted_decimals) - quoted)
        assert stale == [], (
            f"{study.slug} declares a quoted precision for {stale}, which its measured table "
            f"does not quote. A renamed quantity would silently drop the precision its claim "
            f"needs and leave this document passing"
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
    "max_targeting_displacement": "replicates",
    "median_targeting_displacement": "replicates",
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

    A ``bound:`` name is exempt for the same reason and gated the same way.  It is an exact
    efficiency bound computed from the study's declared law, so no artefact produced it, and
    a study that publishes one still has to quote it in its measured table.
    """
    return (
        name in _CONFIGURATION_QUANTITIES
        or name.startswith("margin:")
        or name.startswith("bound:")
        or name.endswith("summary_cells")
        or "cells_with_" in name
    )


def _published_block(study: StudyRecord, name: str) -> list[str]:
    """The lines a study section publishes between one pair of generated sentinels."""
    document = study.document_path
    lines = document.read_text(encoding="utf-8").splitlines()
    start, stop = _section(lines, study.anchor, document)
    opening = _OPEN.format(name=name)
    head = next((index for index in range(start, stop) if lines[index].strip() == opening), None)
    assert head is not None, f"{study.slug}'s section carries no {opening}"
    tail = next((index for index in range(head + 1, stop) if lines[index].strip() == _CLOSE), None)
    assert tail is not None, f"{opening} is never closed"
    return lines[head + 1 : tail]


def _agreement_result(row: Any) -> str:
    """How one committed paired verdict must read in the published table.

    Three renderings for three states.  Bold means "a claim came out false" everywhere in
    these documents, so an ``"underpowered"`` cell -- which made no claim -- is italic
    instead.  Written here rather than imported from the renderer on purpose: this check
    exists to disagree with the renderer, and sharing its function would make the two agree
    by construction.
    """
    conclusion = str(row.comparison_conclusion)
    if bool(row.passed):
        return conclusion
    return f"*{conclusion}*" if conclusion == "underpowered" else f"**{conclusion}**"


class TestThePublishedTestTables:
    """One documentation row per committed test, against the results it was rendered from.

    The measured-values table gates the numbers a study's prose quotes.  These gate the tables
    themselves.  Before them, a reader who wanted to know *which* tests a 34/34 counted had to
    open a CSV on GitHub; the count was published and the tests were not, so a row could pass its
    own arithmetic while saying nothing about what was actually run.
    """

    @pytest.mark.parametrize("name", sorted(GENERATED))
    def test_the_published_table_is_the_one_its_results_render(
        self, study: StudyRecord, name: str
    ) -> None:
        """Nothing between the sentinels is typed, so nothing between them can go stale."""
        if name == "agreement" and study.reference is None:
            pytest.skip("study declares no comparator")
        published = _published_block(study, name)
        rendered = GENERATED[name](study, load(study))
        assert published == rendered, (
            f"{study.slug}'s {name} table is not what its artefacts render. Run "
            f"`python -m tests.studies.evidence.document` rather than editing it by hand"
        )

    def test_generated_efficiency_language_requires_an_independent_bound(
        self, study: StudyRecord
    ) -> None:
        properties = load(study)["properties"]
        columns = [name for name in properties if name.startswith("efficiency_")]
        has_bound = bool(columns) and bool(properties[columns].notna().any().any())
        rows = pipe_table(study.document_path, PROPERTY_COLUMNS, section=study.anchor)
        published = "\n".join(
            row[column]
            for row in rows
            for column in ("what was tested", "what must hold", "measured")
        ).casefold()
        assert ("efficien" in published) == has_bound

    def test_a_study_without_a_comparator_publishes_no_agreement_table(
        self, study: StudyRecord
    ) -> None:
        """The absence has to be stated, because an empty table reads like a missing one."""
        if study.reference is not None:
            pytest.skip("study declares a comparator")
        assert study.artifact("equivalence.csv").exists()
        lines = study.document_path.read_text(encoding="utf-8").splitlines()
        start, stop = _section(lines, study.anchor, study.document_path)
        section = "\n".join(lines[start:stop])
        assert _OPEN.format(name="agreement") not in section, (
            f"{study.slug} has no comparator but its section carries an agreement block"
        )
        assert "no canonical implementation is compared" in section.casefold(), (
            f"{study.slug} must say in its own section that no comparator is claimed"
        )

    @pytest.mark.parametrize(
        ("name", "columns", "artifact", "keys"),
        [
            (
                "accuracy",
                ACCURACY_COLUMNS,
                "performance",
                ("scenario", "estimand", "implementation"),
            ),
            ("agreement", AGREEMENT_COLUMNS, "equivalence", ("scenario", "estimand")),
            ("properties", PROPERTY_COLUMNS, "properties", ("property", "cell")),
        ],
    )
    def test_every_published_result_is_the_committed_verdict(
        self,
        study: StudyRecord,
        name: str,
        columns: tuple[str, ...],
        artifact: str,
        keys: tuple[str, ...],
    ) -> None:
        """Read out of the document and compared with the CSV, not with the renderer.

        The test above would pass a renderer that printed every verdict as a pass, because it
        compares the document with that same renderer's output.  This one goes to the committed
        column instead, so the two checks fail for different reasons.
        """
        if name == "agreement" and study.reference is None:
            pytest.skip("study declares no comparator")
        rows = pipe_table(study.document_path, columns, section=study.anchor)
        frame = load(study)[artifact]
        assert len(rows) == len(frame), (
            f"{study.slug}'s {name} table publishes {len(rows)} rows against {len(frame)} "
            f"committed tests"
        )
        published = [row["result"] for row in rows]
        expected = (
            [_agreement_result(row) for row in frame.itertuples()]
            if name == "agreement"
            else ["pass" if bool(passed) else "**fail**" for passed in frame["passed"]]
        )
        assert sorted(published) == sorted(expected), (
            f"{study.slug}'s {name} results are not the committed ones"
        )

    def test_every_key_a_result_file_uses_has_a_description(self, study: StudyRecord) -> None:
        """The one hand-written column, required to cover every key that can reach a table."""
        data = load(study)
        undescribed: list[str] = []
        for frame, resolvers in (
            (
                data["performance"],
                (
                    ("implementation", descriptions.implementation),
                    ("scenario", descriptions.scenario),
                    ("estimand", descriptions.estimand),
                ),
            ),
            (
                data["equivalence"],
                (("scenario", descriptions.scenario), ("estimand", descriptions.estimand)),
            ),
        ):
            for column, resolve in resolvers:
                for key in sorted(set(frame[column])):
                    try:
                        resolve(str(key))
                    except descriptions.Undescribed as absent:
                        undescribed.append(str(absent))
        for family, cell, role in sorted(
            {
                (str(row.property), str(row.cell), str(row.role))
                for row in data["properties"].itertuples()
            }
        ):
            try:
                descriptions.claim(family)
            except descriptions.Undescribed as absent:
                undescribed.append(str(absent))
            try:
                descriptions.cell(family, cell, role=role)
            except descriptions.Undescribed as absent:
                undescribed.append(str(absent))
        assert undescribed == [], (
            f"{study.slug} publishes rows whose meaning is undeclared:\n  "
            + "\n  ".join(sorted(set(undescribed)))
            + "\nAdd them to tests/studies/evidence/descriptions.py"
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
        assert declared["margin:minimum_power"] == property_verdicts.MINIMUM_POWER
        assert declared["margin:root_n_slope_lower"] == (
            property_verdicts.ROOT_N_SLOPE - property_verdicts.ROOT_N_SLOPE_MARGIN
        )
        if "crossfit_overfitting" in study.property_cells:
            assert declared["margin:overfit_se_floor"] == property_verdicts.OVERFIT_SE_FLOOR
            assert (
                declared["margin:overfit_control_ceiling"]
                == property_verdicts.OVERFIT_SE_CONTROL_CEILING
            )
            assert (
                declared["margin:overfit_coverage_gain"] == property_verdicts.OVERFIT_COVERAGE_GAIN
            )
        if "clustered_inference" in study.property_cells:
            assert (
                declared["margin:iid_control_se_ceiling"]
                == property_verdicts.CLUSTER_ROBUST_CONTROL_SE_CEILING
            )
            assert (
                declared["margin:clustered_coverage_gain"]
                == property_verdicts.CLUSTERED_COVERAGE_GAIN
            )
        if "double_robustness" in study.property_cells:
            low, high = property_verdicts.UNION_MODEL_SE_BAND
            assert declared["margin:union_model_se_lower"] == low
            assert declared["margin:union_model_se_upper"] == high
        if "generated_design" in study.property_cells:
            assert (
                declared["margin:generated_design_deficit"]
                == study.properties().GENERATED_DESIGN_DEFICIT
            )
        if "selector_necessity" in study.property_cells:
            selector = study.properties()
            assert declared["margin:selector_rmse_ratio"] == selector.SELECTOR_RMSE_RATIO
        if "targeting_necessity" in study.property_cells:
            assert (
                declared["margin:targeting_displacement"]
                == study.properties().TARGETING_DISPLACEMENT
            )
        if "weight_necessity" in study.property_cells:
            assert declared["margin:weight_displacement"] == study.properties().WEIGHT_DISPLACEMENT
        if "learner_weight_necessity" in study.property_cells:
            assert (
                declared["margin:learner_weight_displacement"]
                == study.properties().WEIGHT_DISPLACEMENT
            )
        if "projection_necessity" in study.property_cells:
            assert (
                declared["margin:projection_displacement"]
                == study.properties().PROJECTION_DISPLACEMENT
            )
        if (
            "survival_recursion_necessity" in study.property_cells
            or "competing_risk_recursion_necessity" in study.property_cells
        ):
            assert (
                declared["margin:recursion_displacement"]
                == study.properties().RECURSION_DISPLACEMENT
            )
        if any(
            cell.endswith("noise_control")
            for cell in study.property_cells.get("interval_calibration", ())
        ):
            efficiency = study.properties()
            assert declared["margin:shrunken_se_factor"] == efficiency.SHRUNKEN_SE_FACTOR
            if study.calibration_efficiency_ratio:
                low, high = efficiency.EFFICIENCY_RATIO_BAND
                assert declared["margin:efficiency_ratio_lower"] == low
                assert declared["margin:efficiency_ratio_upper"] == high
        for estimand, deviation in study.efficiency_bounds.items():
            assert declared[f"bound:{estimand}_standard_error"] == deviation / math.sqrt(study.n)
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
