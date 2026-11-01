"""The declared comparator phase of ``structural-nested-mean`` (X44, Study queue row 43b).

The comparator is EconML 0.17.0 ``DynamicDML``.  These tests check the declaration without
EconML: the cells it pairs, the task table the runner reads, the computed limits of the
Markov-state fit, the descriptive table, the predictive replicate count, and the registry rules
for a reference that reports a subset of the study's cells.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import betabinom, norm

from tests.studies import canonical_structural_nested_mean as study
from tests.studies import structural_nested_mean_laws as laws
from tests.studies.evidence.registry import ROOT, Margins
from tests.studies.evidence.schema import REPLICATE_COLUMNS, validate_replicates

EXACT = 1e-10
#: The cell floor and the family floor of the predictive rule.
CELL_FLOOR = 0.90
FAMILY_FLOOR = 0.95
#: The count of the committed subject run, before the comparator raised it.
SUBJECT_REPLICATES = 1_000


class TestCells:
    def test_the_paired_cells_are_declared_cells_without_counterfactual_means(self) -> None:
        for scenario, names in study.REFERENCE_SCENARIOS.items():
            assert set(names) <= set(study.SCENARIOS[scenario])
            assert not any(name.startswith("ey") for name in names)
        assert set(study.COMPARATOR_PAIRED) == set(study.REFERENCE_SCENARIOS)

    def test_a_blip_on_a_past_treatment_or_on_one_period_only_is_not_paired(self) -> None:
        # ``h`` and ``p`` have a blip feature that ``DynamicDML``'s shared ``X`` cannot state.
        assert "blip[A2:1.0:A1]" in laws.setup("H").estimands
        assert "blip[A2:1.0:A1]" not in study.MARKOV_ESTIMANDS["h"]
        assert {"h", "p"}.isdisjoint(study.COMPARATOR_PAIRED)

    def test_the_registered_record_reports_the_paired_cells(self) -> None:
        record = study.COMPARATOR_STUDY
        assert (record.reference, record.pending_reference) == ("econml", None)
        assert record.reported("econml") == study.REFERENCE_SCENARIOS
        assert record.reported(study.IMPLEMENTATION) == study.SCENARIOS
        assert record.paired_cells() == {
            (scenario, name)
            for scenario, names in study.REFERENCE_SCENARIOS.items()
            for name in names
        }
        assert set(record.extra_artifacts) == {
            study.MARKOV_REPLICATES,
            study.MARKOV_SUMMARY,
            study.MARKOV_FAILURES,
            study.EXACTNESS_FILE,
        }

    def test_the_hook_binds_the_driver_keywords(self) -> None:
        parameters = inspect.signature(study.reference_artifacts).parameters
        assert set(parameters) == {
            "reference",
            "here",
            "samples",
            "truths_path",
            "reference_results",
            "output",
            "cores",
        }
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())


class TestTasks:
    def test_the_task_table_carries_the_exact_truths_and_settings(self) -> None:
        table = study.comparator_tasks()
        assert tuple(table.columns) == study.TASK_COLUMNS
        for row in table.itertuples(index=False):
            name = study.SETUP_OF[row.scenario]
            assert row.truth == laws.truth(name, row.estimand)
            assert row.nodes == laws.setup(name).law.n_nodes
            if row.variant == "paired":
                assert row.order == study.INTERACTIONS.get(name, study.DEFAULT_INTERACTIONS)
                assert row.features == ""
        markov = table.loc[table["variant"] == "markov"]
        assert set(markov["scenario"]) == set(study.COMPARATOR_MARKOV)

    def test_a_contrast_carries_its_regimens(self) -> None:
        table = study.comparator_tasks().set_index(["scenario", "variant", "estimand"])
        row = table.loc[("c", "paired", "ate_regimen[dose_high vs dose0]")]
        assert (row["t1"], row["t0"]) == ("1.0;-0.5", "0.0;0.0")

    def test_the_sample_carries_the_subjects_folds(self) -> None:
        frame, _ = study.draw_scenario("s2", 400, 0)
        sample = study.comparator_sample("s2", 0, frame)
        result = study.fit("S2", frame, random_state=study.COMPARATOR_FOLD_SEED)
        np.testing.assert_array_equal(sample["fold"], np.asarray(result.folds.assignment))
        assert list(sample.columns[:2]) == ["scenario", "replicate"]
        assert study.comparator_sample("n", 0, frame).empty


class TestLimits:
    @pytest.mark.parametrize("scenario", ["s2", "k"])
    def test_the_markov_state_suffices_on_the_markov_laws(self, scenario: str) -> None:
        name = study.SETUP_OF[scenario]
        for estimand in study.MARKOV_ESTIMANDS[scenario]:
            limit = laws.dynamic_dml_limit(name, estimand)
            assert limit == pytest.approx(laws.truth(name, estimand), abs=EXACT)

    @pytest.mark.parametrize("scenario", ["s2", "s3", "k", "c", "m"])
    def test_the_history_state_returns_every_paired_truth(self, scenario: str) -> None:
        name = study.SETUP_OF[scenario]
        for estimand in study.REFERENCE_SCENARIOS[scenario]:
            limit = laws.dynamic_dml_limit(name, estimand, state="history")
            assert limit == pytest.approx(laws.truth(name, estimand), abs=EXACT)

    @pytest.mark.parametrize(
        ("scenario", "estimand", "gap"),
        [
            ("s3", "blip[A3:1.0:intercept]", 0.238),
            ("m", "blip[A2:1.0:intercept]", 0.688),
            ("p", "blip[A1:1.0:intercept]", 0.790),
            ("h", "blip[A1:1.0:intercept]", 0.301),
        ],
    )
    def test_the_markov_state_misses_on_a_law_it_does_not_hold(
        self, scenario: str, estimand: str, gap: float
    ) -> None:
        # A nonzero witness: a limit that silently returned the truth would fail here.
        name = study.SETUP_OF[scenario]
        features = study.MARKOV_FEATURES.get(scenario, ())
        limit = laws.dynamic_dml_limit(name, estimand, features=features)
        assert limit - laws.truth(name, estimand) == pytest.approx(gap, abs=5e-4)


class TestDescriptiveTable:
    def _rows(self, shift: float, spread: float = 0.05, replicates: int = 400) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        out = []
        for scenario, names in study.MARKOV_ESTIMANDS.items():
            name = study.SETUP_OF[scenario]
            features = study.MARKOV_FEATURES.get(scenario, ())
            for estimand in names:
                limit = laws.dynamic_dml_limit(name, estimand, features=features)
                truth = laws.truth(name, estimand)
                for replicate in range(replicates):
                    estimate = limit + shift * spread + spread * rng.normal()
                    out.append(
                        {
                            "implementation": "econml",
                            "scenario": scenario,
                            "replicate": replicate,
                            "n": 2000,
                            "estimand": estimand,
                            "truth": truth,
                            "estimate": estimate,
                            "inference_estimate": estimate,
                            "std_error": spread,
                            "ci_lower": estimate - 1.96 * spread,
                            "ci_upper": estimate + 1.96 * spread,
                            "inference_scale": "identity",
                            "covered": int(abs(estimate - truth) <= 1.96 * spread),
                            "initial_estimate": math.nan,
                        }
                    )
        return pd.DataFrame(out, columns=list(REPLICATE_COLUMNS))

    def test_a_fit_at_its_limit_agrees_and_one_off_it_does_not(self) -> None:
        empty = pd.DataFrame(columns=list(study.MARKOV_FAILURE_COLUMNS))
        at = study.markov_summary(self._rows(0.0), empty)
        assert at["agrees_with_limit"].all()
        assert (at["replicates"] == 400).all()
        off = study.markov_summary(self._rows(0.6), empty)
        assert not off["agrees_with_limit"].any()

    def test_the_gap_column_is_the_declared_difference(self) -> None:
        empty = pd.DataFrame(columns=list(study.MARKOV_FAILURE_COLUMNS))
        table = study.markov_summary(self._rows(0.0), empty).set_index(["scenario", "estimand"])
        assert table.loc[("s2", "blip[A1:1.0:intercept]"), "gap"] == pytest.approx(0.0, abs=EXACT)
        assert table.loc[("m", "blip[A2:1.0:intercept]"), "gap"] == pytest.approx(0.688, abs=5e-4)
        # The fit centres on its limit, so a law with a gap covers its truth rarely.
        assert table.loc[("m", "blip[A2:1.0:intercept]"), "coverage"] < 0.01
        assert table.loc[("m", "blip[A2:1.0:intercept]"), "limit_coverage"] > 0.9


def coverage_leg_pass_probability(
    pilot: tuple[int, int, str], replicates: int, margin: float = Margins().coverage_noninferiority
) -> float:
    """The predictive chance that the paired coverage leg of one cell passes.

    Every discordant pilot pair had one side covering, so the leg reads the share ``p`` of such
    pairs.  ``p`` has the Jeffreys posterior of the pilot count, and the study's count is
    beta-binomial.  The leg passes when the one-sided 99% lower bound of the coverage difference
    (subject minus reference) clears ``margin``, in its normal form.
    """
    draws, discordant, side = pilot
    count = np.arange(replicates + 1)
    weight = betabinom.pmf(count, replicates, discordant + 0.5, draws - discordant + 0.5)
    share = count / replicates
    difference = share if side == "subject" else -share
    half = norm.ppf(Margins().confidence_level) * np.sqrt(share * (1.0 - share) / replicates)
    return float(weight[difference - half >= margin].sum())


class TestReplicates:
    def test_the_declared_count_is_the_predictive_rule(self) -> None:
        def passes(replicates: int) -> bool:
            cells = [
                coverage_leg_pass_probability(pilot, replicates)
                for pilot in study.COMPARATOR_PILOT.values()
            ]
            return min(cells) >= CELL_FLOOR and math.prod(cells) >= FAMILY_FLOOR

        declared = study.COMPARATOR_REPLICATES
        assert declared % 200 == 0 and declared >= SUBJECT_REPLICATES
        assert passes(declared)
        assert not any(passes(r) for r in range(SUBJECT_REPLICATES, declared, 200)), (
            "a smaller count passes the rule"
        )

    def test_every_primary_scenario_runs_at_the_comparators_count(self) -> None:
        assert study.COMPARATOR_REPLICATES == study.PRIMARY_REPLICATES
        assert study.STUDY.replicates == study.COMPARATOR_REPLICATES
        assert study.STUDY is study.COMPARATOR_STUDY
        assert study.PENDING_STUDY.replicates == study.COMPARATOR_REPLICATES

    def test_the_subject_side_cell_never_binds(self) -> None:
        pilot = study.COMPARATOR_PILOT["ate_regimen[r12 vs r00]"]
        assert coverage_leg_pass_probability(pilot, SUBJECT_REPLICATES) == pytest.approx(1.0)


class TestRegistry:
    def test_a_reference_subset_needs_a_reference_and_declared_cells(self) -> None:
        base = study.PENDING_STUDY
        with pytest.raises(ValueError, match="needs a reference"):
            dataclasses.replace(
                base,
                pending_reference=None,
                pending_reason="",
                comparison=None,
                failure_bound={study.IMPLEMENTATION: 0.01},
                reference_scenarios={"s2": ("blip[A1:1.0:intercept]",)},
            )
        with pytest.raises(ValueError, match="does not declare"):
            dataclasses.replace(base, reference_scenarios={"s2": ("ey_regimen[r99]",)})
        with pytest.raises(ValueError, match="no estimand"):
            dataclasses.replace(base, reference_scenarios={"s2": ()})

    def _rows(self) -> pd.DataFrame:
        record = study.COMPARATOR_STUDY
        out = []
        for implementation in record.implementations:
            for scenario, names in record.reported(implementation).items():
                for name in names:
                    truth = laws.truth(study.SETUP_OF[scenario], name)
                    for replicate in range(record.replicates):
                        out.append(
                            {
                                "implementation": implementation,
                                "scenario": scenario,
                                "replicate": replicate,
                                "n": record.n,
                                "estimand": name,
                                "truth": truth,
                                "estimate": truth,
                                "inference_estimate": truth,
                                "std_error": 0.1,
                                "ci_lower": truth - 0.2,
                                "ci_upper": truth + 0.2,
                                "inference_scale": "identity",
                                "covered": 1,
                                "initial_estimate": truth + 0.5,
                            }
                        )
        return pd.DataFrame(out, columns=list(REPLICATE_COLUMNS))

    def test_the_schema_reads_the_reference_subset_and_refuses_a_lost_cell(self) -> None:
        record = dataclasses.replace(study.COMPARATOR_STUDY, replicates=2)
        rows = self._rows().loc[lambda frame: frame["replicate"] < 2]
        validate_replicates(rows, record=record)
        lost = rows.loc[
            ~(
                (rows["implementation"] == "econml")
                & (rows["estimand"] == "ate_regimen[r21 vs r00]")
            )
        ]
        with pytest.raises(ValueError, match="lacks"):
            validate_replicates(lost, record=record)
        extra = rows.loc[
            (rows["implementation"] == study.IMPLEMENTATION) & (rows["estimand"] == "ey_reference")
        ].assign(implementation="econml")
        with pytest.raises(ValueError, match="beyond"):
            validate_replicates(pd.concat([rows, extra], ignore_index=True), record=record)


class TestComparatorPin:
    """The manifest's comparator pin is the Dockerfile's base image and the lock file's hash."""

    folder = ROOT / "tests" / "canonical" / "econml_dynamic_dml"

    def test_the_pin_is_read_back_from_the_dockerfile_and_the_lock_file(self) -> None:
        pin = study.REFERENCE_METADATA
        dockerfile = (self.folder / "Dockerfile").read_text()
        base = re.search(r"^FROM (\S+)$", dockerfile, flags=re.MULTILINE)
        assert base is not None
        assert pin["python_base_image"] == base.group(1)
        assert re.fullmatch(r"python:[\w.-]+@sha256:[0-9a-f]{64}", pin["python_base_image"])
        assert f"econml.__version__ == '{pin['econml_version']}'" in dockerfile
        lock = (self.folder / "requirements.txt").read_bytes()
        assert pin["requirements_sha256"] == hashlib.sha256(lock).hexdigest()
        assert re.search(
            rf"^econml=={re.escape(pin['econml_version'])} ", lock.decode(), flags=re.MULTILINE
        )

    def test_the_driver_reads_the_pin_from_the_study_module(self) -> None:
        assert any(key.endswith("_version") for key in study.REFERENCE_METADATA)


class TestExactness:
    """``exactness.csv`` gates the paired estimates and publishes the SE ratios."""

    def _rows(self) -> pd.DataFrame:
        record = study.STUDY
        out = []
        for implementation in record.implementations:
            ratio = 1.0 if implementation == study.IMPLEMENTATION else 1.02
            for scenario, names in record.reported(implementation).items():
                for name in names:
                    truth = laws.truth(study.SETUP_OF[scenario], name)
                    for replicate in range(3):
                        estimate = truth + 0.01 * (replicate - 1)
                        out.append(
                            {
                                "implementation": implementation,
                                "scenario": scenario,
                                "replicate": replicate,
                                "n": record.n,
                                "estimand": name,
                                "truth": truth,
                                "estimate": estimate,
                                "inference_estimate": estimate,
                                "std_error": 0.1 * ratio,
                                "ci_lower": estimate - 0.2,
                                "ci_upper": estimate + 0.2,
                                "inference_scale": "identity",
                                "covered": 1,
                                "initial_estimate": truth + 0.5,
                            }
                        )
        return pd.DataFrame(out, columns=list(REPLICATE_COLUMNS))

    @staticmethod
    def _table(rows: pd.DataFrame) -> pd.DataFrame:
        return study.extra_artifacts(rows)["exactness.csv"]

    def test_the_artifact_is_declared_and_has_one_row_per_paired_cell(self) -> None:
        assert "exactness.csv" in study.COMPARATOR_STUDY.extra_artifacts
        table = self._table(self._rows())
        assert tuple(table.columns) == study.EXACTNESS_COLUMNS
        assert len(table) == len(study.STUDY.paired_cells()) == 21
        assert set(zip(table["scenario"], table["estimand"], strict=True)) == set(
            study.STUDY.paired_cells()
        )
        assert (table["paired_replicates"] == 3).all()
        assert table["passed"].all()
        assert (table["gate"] == 1e-9).all()
        assert table["mean_se_ratio"].to_numpy() == pytest.approx(1.02)
        assert study.scientific_failures({"exactness.csv": table})[study.EXACTNESS_AUDIT].empty

    def test_a_point_moved_by_one_millionth_turns_its_row_red(self) -> None:
        rows = self._rows()
        scenario, estimand = "k", "ate_regimen[r21 vs r00]"
        hit = (
            (rows["implementation"] == "econml")
            & (rows["scenario"] == scenario)
            & (rows["estimand"] == estimand)
            & (rows["replicate"] == 1)
        )
        rows.loc[hit, "estimate"] += 1e-6
        table = self._table(rows)
        red = table.loc[~table["passed"]]
        assert list(zip(red["scenario"], red["estimand"], strict=True)) == [(scenario, estimand)]
        assert float(red["max_estimate_gap"].iloc[0]) == pytest.approx(1e-6, rel=1e-3)
        assert len(study.scientific_failures({"exactness.csv": table})[study.EXACTNESS_AUDIT]) == 1

    def test_a_standard_error_scaled_by_three_percent_leaves_its_row_green(self) -> None:
        rows = self._rows()
        scenario, estimand = "k", "ate_regimen[r21 vs r00]"
        hit = (
            (rows["implementation"] == "econml")
            & (rows["scenario"] == scenario)
            & (rows["estimand"] == estimand)
        )
        rows.loc[hit, "std_error"] *= 1.03
        table = self._table(rows)
        cell = table.loc[(table["scenario"] == scenario) & (table["estimand"] == estimand)]
        assert bool(cell["passed"].iloc[0])
        assert table["passed"].all()
        assert float(cell["mean_se_ratio"].iloc[0]) == pytest.approx(1.02 * 1.03)

    def test_a_cell_with_no_pairs_reads_red(self) -> None:
        rows = self._rows()
        rows = rows.loc[~((rows["implementation"] == "econml") & (rows["scenario"] == "c"))]
        table = self._table(rows)
        lost = table.loc[table["scenario"] == "c"]
        assert (lost["paired_replicates"] == 0).all() and not lost["passed"].any()

    def test_a_lost_paired_fit_counts_in_the_failure_accounting(self) -> None:
        table = self._table(self._rows())
        columns = [
            "implementation",
            "group_id",
            "scenarios",
            "replicates",
            "attempt",
            "class",
            "stage",
            "call",
            "message",
            "seed",
        ]

        def record(implementation: str, scenarios: str, replicates: str) -> list[object]:
            return [implementation, "g", scenarios, replicates, 1, "fitting_failure", "fit"]

        lost = pd.DataFrame(
            [
                # Both sides lost replicate 4 of ``k``: one lost pair.
                [*record("econml", "k", "4"), "dynamic_dml(...)", "singular", 1],
                [*record(study.IMPLEMENTATION, "k", "4"), "fit(...)", "singular", 1],
                # The subject alone lost replicate 7 of ``h``, which no paired cell reads.
                [*record(study.IMPLEMENTATION, "h", "7"), "fit(...)", "singular", 2],
            ],
            columns=columns,
        )
        audit = study.account_failures(lost, {"exactness.csv": table}).set_index("leg")
        assert set(audit.index) == {study.EXACTNESS_AUDIT, study.MARKOV_AUDIT}
        assert audit.loc[study.EXACTNESS_AUDIT, "failed"] == 1
        assert not audit.loc[study.EXACTNESS_AUDIT, "passed"]
        assert audit.loc[study.MARKOV_AUDIT, "passed"]
        clean = study.account_failures(lost.iloc[:0], {"exactness.csv": table})
        assert clean["passed"].all()
