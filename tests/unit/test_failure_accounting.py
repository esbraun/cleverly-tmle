"""The accounting of recorded comparator failures (RM39a2).

A reference fit that fails leaves its replicate out of every cell it would have entered.  These
tests pin the four things that make that safe to publish: the missing rows must be exactly the
recorded ones, a cell may lose at most ``floor(bound * R)`` replicates, each verdict leg must
survive its least favourable fill of the lost values, and a red accounting row reaches the
ledger as a red cell with an owner.  Each gate has a mutation control that fails when the gate
is removed or inverted.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence import claims, failures, red_cells
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.manifest import write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import Margins, StudyRecord
from tests.studies.evidence.schema import validate_replicates

REFERENCE = "R ref"
SUBJECT = "cleverly"
TRUTH = 1.0
SPREAD = 0.1


def _record(*, replicates: int = 200, bound: float = 0.05, **changes: object) -> StudyRecord:
    values: dict[str, object] = {
        "name": "failure test",
        "slug": "failure-test",
        "artifacts": Path("."),
        "document": "test.md",
        "anchor": "test",
        "scenarios": {"law": ("ate",)},
        "replicates": replicates,
        "n": 500,
        "seed": 11,
        "reference": REFERENCE,
        "failure_bound": {REFERENCE: bound} if bound else {},
        "margins": Margins(bootstrap_replicates=400),
    }
    values.update(changes)
    return StudyRecord(**values)  # type: ignore[arg-type]


def _rows(
    record: StudyRecord,
    failed: tuple[int, ...] = (),
    *,
    reference_bias: float = 0.0,
    seed: int = 3,
) -> pd.DataFrame:
    """Calibrated, unbiased subject rows; reference rows with ``failed`` left out."""
    rng = np.random.default_rng(seed)
    out = []
    for scenario, estimands in record.scenarios.items():
        for estimand in estimands:
            # Two implementations of one estimator: nearly the same estimate on each sample.
            shared = rng.normal(size=record.replicates)
            draws = {
                SUBJECT: shared,
                REFERENCE: shared + 0.05 * rng.normal(size=record.replicates),
            }
            for implementation, noise in draws.items():
                noise = (noise - noise.mean()) / noise.std(ddof=1)
                shift = reference_bias if implementation == REFERENCE else 0.0
                for replicate in range(record.replicates):
                    if implementation == REFERENCE and replicate in failed:
                        continue
                    estimate = TRUTH + shift * SPREAD + SPREAD * float(noise[replicate])
                    low, high = estimate - 2.576 * SPREAD, estimate + 2.576 * SPREAD
                    out.append(
                        {
                            "implementation": implementation,
                            "scenario": scenario,
                            "replicate": replicate,
                            "n": record.n,
                            "estimand": estimand,
                            "truth": TRUTH,
                            "estimate": estimate,
                            "inference_estimate": estimate,
                            "std_error": SPREAD,
                            "ci_lower": low,
                            "ci_upper": high,
                            "inference_scale": "identity",
                            "covered": bool(low <= TRUTH <= high),
                            "initial_estimate": estimate,
                        }
                    )
    return pd.DataFrame(out)


def _failures(*groups: tuple[str, str], implementation: str = REFERENCE) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "implementation": implementation,
                "group_id": f"r{index:09d}",
                "scenarios": scenarios,
                "replicates": replicates,
                "attempt": 1,
                "class": "fitting_failure",
                "stage": "fit",
                "call": "drtmle(...)",
                "message": "singular",
                "seed": 1,
            }
            for index, (scenarios, replicates) in enumerate(groups)
        ],
        columns=list(failures.FAILURE_COLUMNS),
    )


class TestDeclaration:
    def test_the_allowance_is_the_floor_of_bound_times_replicates(self) -> None:
        assert failures.allowed(_record(replicates=300, bound=0.01), REFERENCE) == 3
        assert failures.allowed(_record(replicates=99, bound=0.05), REFERENCE) == 4
        assert failures.allowed(_record(bound=0.0), REFERENCE) == 0
        assert failures.allowed(_record(), SUBJECT) == 0

    def test_a_bound_outside_its_range_or_on_an_unknown_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 0.05\]"):
            _record(bound=0.06)
        with pytest.raises(ValueError, match="undeclared"):
            _record(failure_bound={"nobody": 0.01})

    def test_the_manifest_records_a_declared_bound_only(self, tmp_path: Path) -> None:
        import json

        for record, expected in ((_record(), {REFERENCE: 0.05}), (_record(bound=0.0), None)):
            path = tmp_path / "manifest.json"
            write_manifest(path, record, [])
            configuration = json.loads(path.read_text())["configuration"]
            assert configuration.get("failure_bound") == expected


class TestReconciliation:
    def test_a_record_without_a_scenario_column_costs_every_scenario(self) -> None:
        record = _record(scenarios={"a": ("ate",), "b": ("ate", "att")})
        lost = failures.expand(_failures(("*", "4;5")), record)
        assert len(lost) == 6
        assert set(lost["scenario"]) == {"a", "b"}

    def test_an_unknown_scenario_and_a_double_record_are_refused(self) -> None:
        record = _record()
        with pytest.raises(ValueError, match="unknown"):
            failures.expand(_failures(("other", "1")), record)
        with pytest.raises(ValueError, match="two failure records"):
            failures.expand(_failures(("law", "1"), ("law", "1;2")), record)

    def test_recorded_missing_rows_validate_and_unrecorded_ones_do_not(self) -> None:
        record = _record()
        rows = _rows(record, failed=(3, 7))
        validate_replicates(rows, record=record, failures=_failures(("law", "3;7")))
        with pytest.raises(ValueError, match="do not cover"):
            validate_replicates(rows, record=record, failures=_failures(("law", "3")))
        with pytest.raises(ValueError, match="both survived and failed"):
            validate_replicates(rows, record=record, failures=_failures(("law", "3;7;8")))
        with pytest.raises(ValueError):
            validate_replicates(rows, record=record)

    def test_a_subject_failure_waits_for_the_python_phase(self) -> None:
        record = _record()
        with pytest.raises(ValueError, match="RM39b"):
            failures.reconcile(
                _rows(record), _failures(("law", "1"), implementation=SUBJECT), record=record
            )

    def test_a_cell_with_no_survivor_is_refused(self) -> None:
        record = _record(replicates=3)
        rows = _rows(record, failed=(0, 1, 2))
        with pytest.raises(ValueError, match="every replicate failed"):
            failures.reconcile(rows, _failures(("law", "0;1;2")), record=record)


def _truth(record: StudyRecord, rows: pd.DataFrame, lost: str) -> pd.DataFrame:
    return failures.truth_accounting(rows, _failures(("law", lost)), record=record, n_jobs=1)


class TestTruthAccounting:
    def test_a_few_failures_of_a_sound_reference_stay_within_the_bound(self) -> None:
        record = _record()
        frame = _truth(record, _rows(record, failed=(1, 2)), "1;2")
        assert frame["gate_passed"].all()
        assert set(frame["reason"]) == {failures.WITHIN_BOUND}
        assert set(frame["leg"]) == {"failure bound", "bias", "coverage", "se ratio"}

    def test_more_failures_than_the_bound_allows_are_red(self) -> None:
        record = _record(bound=0.01)
        lost = tuple(range(3))
        frame = _truth(record, _rows(record, failed=lost), "0;1;2")
        assert not frame["gate_passed"].any()
        assert set(frame["reason"]) == {failures.BOUND_EXCEEDED}

    def test_a_reference_near_its_bias_margin_fails_its_worst_case(self) -> None:
        """Mutation control: the complete cases pass, the least favourable fill does not."""
        record = _record()
        lost = tuple(range(10))
        rows = _rows(record, failed=lost, reference_bias=0.03)
        frame = _truth(record, rows, ";".join(str(value) for value in lost))
        bias = frame.loc[frame["leg"] == "bias"].iloc[0]
        assert bias["complete_case"] > 0 > bias["worst_case"]
        assert not bias["leg_passed"]
        assert set(frame["reason"]) == {failures.WORST_CASE_FAILS}

    def test_the_corrected_verdict_is_red_and_removing_the_gate_turns_it_green(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _record(replicates=400, bound=0.005)
        rows = _rows(record, failed=(0, 1, 2))
        lost = _failures(("law", "0;1;2"))
        performance = independent_performance_tests(rows, record=record, n_jobs=1)
        accounting = failures.truth_accounting(rows, lost, record=record, n_jobs=1)
        reference = performance["implementation"] == REFERENCE
        assert performance.loc[reference, "passed"].all()
        corrected = failures.correct_performance(performance, accounting)
        assert not corrected.loc[reference, "passed"].any()
        assert corrected.loc[~reference, "passed"].all()
        monkeypatch.setattr(failures, "_gate", lambda *_: (True, failures.WITHIN_BOUND))
        ungated = failures.truth_accounting(rows, lost, record=record, n_jobs=1)
        assert failures.correct_performance(performance, ungated)["passed"].all()


class TestPairedAccounting:
    def _paired(
        self, record: StudyRecord, lost: tuple[int, ...], **changes: object
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rows = _rows(record, failed=lost, **changes)  # type: ignore[arg-type]
        recorded = _failures(("law", ";".join(str(value) for value in lost)))
        summaries = summarize(rows, truth_varies=False)
        performance = independent_performance_tests(rows, record=record, n_jobs=1)
        accounting = failures.truth_accounting(rows, recorded, record=record, n_jobs=1)
        performance = failures.correct_performance(performance, accounting)
        paired = equivalence(rows, summaries, performance, record=record, n_jobs=1)
        paired_accounting = failures.paired_accounting(rows, recorded, record=record, n_jobs=1)
        return paired, paired_accounting, failures.correct_paired(paired, paired_accounting)

    def test_the_pairing_drops_exactly_the_recorded_replicates(self) -> None:
        record = _record()
        paired, accounting, _ = self._paired(record, (4, 9))
        assert paired["dropped_replications"].tolist() == [2]
        assert set(accounting["leg"]) == {
            "failure bound",
            "similarity",
            "rmse ratio",
            "coverage difference",
            "coverage superiority",
            "calibration excess",
        }
        witness = accounting.iloc[0]
        assert math.isfinite(witness["witness_failed_error"])
        assert math.isfinite(witness["witness_succeeded_error"])

    def test_a_cell_over_its_bound_reads_unaccounted(self) -> None:
        record = _record(bound=0.005)
        paired, accounting, corrected = self._paired(record, (4, 9))
        assert paired["passed"].all()
        assert not accounting["gate_passed"].any()
        assert corrected["comparison_conclusion"].tolist() == [failures.UNACCOUNTED]
        assert not corrected["passed"].any()

    def test_the_similarity_leg_fails_when_the_lost_differences_could_move_it(self) -> None:
        """Mutation control: a margin the survivors clear is not cleared at the vertices."""
        record = _record(margins=Margins(bootstrap_replicates=400, paired_difference=0.014))
        lost = tuple(range(10))
        _, accounting, _ = self._paired(record, lost)
        similarity = accounting.loc[accounting["leg"] == "similarity"].iloc[0]
        assert similarity["complete_case"] >= 0 > similarity["worst_case"]
        assert similarity["imputation"].startswith("j=")


class TestCandidateSets:
    """Mutation controls for the fills each leg tries (plan section 9).

    A fill vanishes when the survivors already sit at it, so each test either builds data
    whose verdict turns on the one fill under test, or captures the arrays a leg evaluates
    and compares them with the set built here from the survivors alone.
    """

    LOST = tuple(range(190, 200))

    def _reference(self, record: StudyRecord) -> pd.DataFrame:
        rows = _rows(record, failed=self.LOST)
        return rows.loc[rows["implementation"] == REFERENCE].sort_values("replicate")

    def _wide(self, record: StudyRecord) -> pd.DataFrame:
        rows = _rows(record, failed=self.LOST)
        return rows.pivot(
            index="replicate",
            columns="implementation",
            values=["estimate", "covered", "inference_estimate", "std_error"],
        ).sort_index()

    def test_truth_coverage_fails_when_every_lost_replicate_misses(self) -> None:
        """Only the all-missed fill fails here; a fill that imputes no miss passes."""
        from tests.studies.evidence.inference import clopper_pearson

        record = _record()
        group = self._reference(record).copy()
        k, survivors = len(self.LOST), len(group)
        floor = record.margins.coverage_floor
        confidence = record.margins.confidence_level

        def low(covered: int, total: int) -> float:
            return clopper_pearson(covered, total, confidence_level=confidence).low - floor

        count = next(
            c for c in range(survivors + 1) if low(c, survivors) >= 0 and low(c, survivors + k) < 0
        )
        assert low(count + k, survivors + k) >= 0, "the all-covered fill must pass"
        group["covered"] = [True] * count + [False] * (survivors - count)

        (_, coverage, _) = failures.truth_legs(group, k, record=record, seed=1)
        assert coverage.complete_case >= 0 > coverage.worst_case
        assert coverage.imputation == "j=0 covered"
        assert coverage.worst_case == pytest.approx(low(count, survivors + k))

    def test_the_se_ratio_band_tries_the_upper_and_every_lower_edge_fill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _record()
        group = self._reference(record)
        k = len(self.LOST)
        spread = group["inference_estimate"].to_numpy(dtype=float)
        se = group["std_error"].to_numpy(dtype=float).copy()
        se[:5] = SPREAD / 2  # SE minimum and maximum differ.
        group = group.assign(std_error=se)
        seen: list[bytes] = []
        real = failures._se_ratio

        def spy(spread_values: np.ndarray, se_values: np.ndarray, **options: object) -> np.ndarray:
            seen.append(spread_values.tobytes() + b"|" + se_values.tobytes())
            return real(spread_values, se_values, **options)  # type: ignore[arg-type]

        monkeypatch.setattr(failures, "_se_ratio", spy)
        failures.truth_legs(group, k, record=record, seed=1)

        def key(values: np.ndarray, std_errors: np.ndarray) -> bytes:
            return values.tobytes() + b"|" + std_errors.tobytes()

        def tail(value: float, count: int) -> np.ndarray:
            return np.full(count, value)

        expected = {
            key(spread, se),
            key(
                np.concatenate([spread, tail(spread.mean(), k)]),
                np.concatenate([se, tail(se.max(), k)]),
            ),
        }
        for j in range(k + 1):
            values = np.concatenate([spread, tail(spread.min(), j), tail(spread.max(), k - j)])
            expected.add(key(values, np.concatenate([se, tail(se.min(), k)])))
        assert set(seen) == expected

    def test_the_rmse_and_coverage_legs_fill_the_reference_at_its_best_case(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lost reference estimates sit at the truth and the lost intervals cover it."""
        record = _record()
        wide = self._wide(record)
        k = len(self.LOST)
        captured: list[dict[str, np.ndarray]] = []
        real = failures.bootstrap

        def spy(data: dict[str, np.ndarray], *args: object, **options: object) -> object:
            captured.append({name: np.asarray(values) for name, values in data.items()})
            return real(data, *args, **options)  # type: ignore[arg-type]

        monkeypatch.setattr(failures, "bootstrap", spy)
        failures.paired_legs(wide, self.LOST, record=record, truth=TRUTH, comparable=False, seed=1)
        best = next(data for data in captured if len(data["reference_estimate"]) == 200)
        assert np.all(best["reference_estimate"][-k:] == TRUTH)
        assert np.all(best["reference_covered"][-k:] == 1.0)
        survivors = wide.drop(index=list(self.LOST))
        np.testing.assert_array_equal(
            best["reference_estimate"][:-k], survivors[("estimate", REFERENCE)].to_numpy(float)
        )
        np.testing.assert_array_equal(
            best["subject_estimate"],
            np.concatenate(
                [
                    survivors[("estimate", SUBJECT)].to_numpy(float),
                    wide.loc[list(self.LOST), ("estimate", SUBJECT)].to_numpy(float),
                ]
            ),
        )

    def test_the_calibration_leg_tries_every_se_fill_with_every_estimate_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nine SE and estimate pairs per split, the root strictly inside the SE range."""
        record = _record()
        wide = self._wide(record).copy()
        survivors_index = [r for r in wide.index if r not in self.LOST]
        se = np.linspace(0.05, 0.15, len(survivors_index))
        wide.loc[survivors_index, ("std_error", REFERENCE)] = se
        k = len(self.LOST)
        seen: set[bytes] = set()

        def spy(
            subject_se: np.ndarray,
            subject_inference: np.ndarray,
            reference_se: np.ndarray,
            reference_inference: np.ndarray,
            **options: object,
        ) -> float:
            seen.add(reference_se.tobytes() + b"|" + reference_inference.tobytes())
            return 0.0

        monkeypatch.setattr(failures, "_calibration_slack", spy)
        failures.paired_legs(wide, self.LOST, record=record, truth=TRUTH, comparable=True, seed=1)

        estimates = wide.loc[survivors_index, ("inference_estimate", REFERENCE)].to_numpy(float)
        sets = [np.concatenate([estimates, np.full(k, estimates.mean())])]
        sets += [
            np.concatenate(
                [estimates, np.full(j, estimates.min()), np.full(k - j, estimates.max())]
            )
            for j in range(k + 1)
        ]
        expected = {se.tobytes() + b"|" + estimates.tobytes()}
        roots_inside = 0
        for values in sets:
            root = (np.std(values, ddof=1) * (len(se) + k) - se.sum()) / k
            roots_inside += int(se.min() < root < se.max())
            for fill in (se.min(), se.max(), min(max(root, se.min()), se.max())):
                expected.add(
                    np.concatenate([se, np.full(k, fill)]).tobytes() + b"|" + values.tobytes()
                )
        assert roots_inside, "the control needs a root that neither edge fill reproduces"
        assert seen == expected


def _leg(name: str, complete: float, worst: float) -> failures.Leg:
    strict = name == "coverage superiority"
    return failures.Leg(name, complete, worst, "", worst > 0 if strict else worst >= 0)


def _paired(similarity: tuple[float, float], superiority: tuple[float, float]) -> list:
    return [
        _leg("similarity", *similarity),
        _leg("rmse ratio", 0.1, 0.05),
        _leg("coverage difference", 0.1, 0.05),
        _leg("coverage superiority", *superiority),
        _leg("calibration excess", 0.1, 0.05),
    ]


class TestRoutes:
    """A paired cell passes by equivalence or by superiority; each route is accounted alone."""

    def test_a_superior_cell_that_loses_superiority_reads_equivalent(self) -> None:
        gate = failures._paired_gate(1, 5, _paired((0.1, 0.05), (0.02, -0.01)))
        assert gate == (True, failures.SUPERIORITY_LOST)
        assert failures.accounted_conclusion("superior", equivalent=True, gate=gate) == (
            "equivalent",
            True,
        )
        assert failures.accounted_conclusion("superior", equivalent=False, gate=gate) == (
            failures.UNACCOUNTED,
            False,
        )
        assert failures.accounted_conclusion("equivalent", equivalent=True, gate=gate) == (
            "equivalent",
            True,
        )

    def test_a_superior_cell_that_is_not_similar_is_not_a_false_red(self) -> None:
        """Similarity never held, so its worst case decides nothing."""
        gate = failures._paired_gate(1, 5, _paired((-0.1, -0.2), (0.02, 0.01)))
        assert gate == (True, failures.WITHIN_BOUND)
        assert failures.accounted_conclusion("superior", equivalent=False, gate=gate) == (
            "superior",
            True,
        )

    def test_an_equivalent_cell_that_loses_equivalence_is_unaccounted(self) -> None:
        gate = failures._paired_gate(1, 5, _paired((0.1, -0.05), (0.02, 0.01)))
        assert gate == (True, failures.EQUIVALENCE_LOST)
        assert failures.accounted_conclusion("equivalent", equivalent=True, gate=gate) == (
            failures.UNACCOUNTED,
            False,
        )
        assert failures.accounted_conclusion("superior", equivalent=True, gate=gate) == (
            "superior",
            True,
        )

    def test_no_surviving_route_is_red_and_names_whether_the_failures_decided_it(self) -> None:
        lost = failures._paired_gate(1, 5, _paired((0.1, -0.05), (0.0, -0.01)))
        assert lost == (False, failures.WORST_CASE_FAILS)
        assert failures.accounted_conclusion("equivalent", equivalent=True, gate=lost) == (
            failures.UNACCOUNTED,
            False,
        )
        own = failures._paired_gate(1, 5, _paired((-0.1, -0.2), (-0.01, -0.02)))
        assert own == (False, failures.COMPLETE_CASE_FAILS)
        assert failures.accounted_conclusion("inconclusive", equivalent=False, gate=own) == (
            "inconclusive",
            False,
        )

    def test_a_truth_leg_red_on_its_complete_cases_is_not_charged_to_the_failures(self) -> None:
        record = _record()
        lost = tuple(range(3))
        rows = _rows(record, failed=lost, reference_bias=0.6)
        frame = _truth(record, rows, ";".join(str(value) for value in lost))
        bias = frame.loc[frame["leg"] == "bias"].iloc[0]
        assert bias["complete_case"] < 0
        assert set(frame["reason"]) == {failures.COMPLETE_CASE_FAILS}


class TestHooks:
    def test_a_hook_without_account_failures_is_not_accountable(self) -> None:
        record = _record()
        study = SimpleNamespace(scientific_failures=lambda frames: {})
        frame = failures.hook_accounting(study, _failures(("law", "1")), {}, record=record)
        assert frame["reason"].tolist() == [failures.NOT_ACCOUNTABLE]
        assert not frame["gate_passed"].any()

    def test_no_failure_means_no_hook_row(self) -> None:
        study = SimpleNamespace(scientific_failures=lambda frames: {})
        assert failures.hook_accounting(study, None, {}, record=_record()).empty

    def test_a_missing_reference_fit_is_a_failing_audit_row(self) -> None:
        record = _record()
        gates = {"score audit": pd.DataFrame(), "solver": pd.DataFrame()}
        study = SimpleNamespace(
            scientific_failures=lambda frames: gates,
            account_failures=lambda lost, frames: failures.audits(
                gates, missing={"score audit": failures.missing_fits(lost, record)}
            ),
        )
        frame = failures.hook_accounting(study, _failures(("law", "1;2")), {}, record=record)
        assert frame.set_index("leg")["complete_case"].to_dict() == {
            "score audit": 2.0,
            "solver": 0.0,
        }
        assert frame.set_index("leg")["gate_passed"].to_dict() == {
            "score audit": False,
            "solver": True,
        }

    def test_a_probe_counts_a_failed_replication_as_covered_and_nothing_else(self) -> None:
        record = _record(replicates=4)
        probe = pd.DataFrame(
            {"scenario": "law", "replicate": [0, 2, 3], "passed": [True, True, True]}
        )
        audits = failures.probe_audits(
            probe,
            _failures(("law", "1")),
            record=record,
            scenarios=("law",),
            coverage="coverage",
            audit="probe",
        )
        assert audits["passed"].all()
        gap = failures.probe_audits(
            probe.iloc[[0, 2]],
            _failures(("law", "1")),
            record=record,
            scenarios=("law",),
            coverage="coverage",
            audit="probe",
        )
        assert gap.set_index("leg")["passed"].to_dict() == {"coverage": False, "probe": True}
        # The last replication is gone: the probed and failed ones still form 0..k-1.
        last = failures.probe_audits(
            probe.iloc[[0, 1]],
            _failures(("law", "1")),
            record=record,
            scenarios=("law",),
            coverage="coverage",
            audit="probe",
        )
        assert last.set_index("leg")["passed"].to_dict() == {"coverage": False, "probe": True}

    def test_the_probe_reads_the_surviving_replications_only(self, tmp_path: Path) -> None:
        samples = tmp_path / "samples.csv.gz"
        values = [0.1 + 0.2, 1 / 3, 2 / 3, 0.7, 1e-17]
        pd.DataFrame({"scenario": "law", "replicate": [0, 0, 1, 1, 2], "y": values}).to_csv(
            samples, index=False
        )
        results = tmp_path / "reference.csv"
        pd.DataFrame({"scenario": ["law", "law"], "replicate": [0, 2]}).to_csv(results, index=False)
        with failures.surviving_samples(samples, results) as path:
            assert path != samples and path.parent == samples.parent
            kept = gzip.decompress(path.read_bytes()).decode().splitlines()
            source = gzip.decompress(samples.read_bytes()).decode().splitlines()
            # The kept lines are the source's own bytes, not a reparse.
            assert kept == [source[0], source[1], source[2], source[5]]
        assert not path.exists()

    def test_without_a_failure_the_probe_reads_the_samples_themselves(self, tmp_path: Path) -> None:
        samples = tmp_path / "samples.csv.gz"
        pd.DataFrame({"scenario": "law", "replicate": [0, 1], "y": [0.1, 0.2]}).to_csv(
            samples, index=False
        )
        results = tmp_path / "reference.csv"
        pd.DataFrame({"scenario": ["law", "law"], "replicate": [0, 1]}).to_csv(results, index=False)
        with failures.surviving_samples(samples, results) as path:
            assert path == samples
        assert samples.exists()


class TestLedger:
    def _tables(self, record: StudyRecord) -> dict[str, pd.DataFrame]:
        rows = _rows(record, failed=(4, 9))
        recorded = _failures(("law", "4;9"))
        summaries = summarize(rows, truth_varies=False)
        performance = independent_performance_tests(rows, record=record, n_jobs=1)
        accounting = failures.truth_accounting(rows, recorded, record=record, n_jobs=1)
        performance = failures.correct_performance(performance, accounting)
        paired = equivalence(rows, summaries, performance, record=record, n_jobs=1)
        paired_accounting = failures.paired_accounting(rows, recorded, record=record, n_jobs=1)
        hook = pd.DataFrame(
            [
                {
                    **paired_accounting.iloc[0].to_dict(),
                    "table": "hook",
                    "scenario": "",
                    "estimand": "",
                    "leg": "score audit",
                    "gate_passed": False,
                    "reason": failures.WORST_CASE_FAILS,
                }
            ]
        )
        return {
            "performance": performance,
            "equivalence": failures.correct_paired(paired, paired_accounting),
            "properties": pd.DataFrame(
                columns=["property", "cell", "role", "passed", "property_passed"]
            ),
            "accounting": pd.concat([accounting, paired_accounting, hook], ignore_index=True),
        }

    def test_an_unaccounted_row_recomputes_and_needs_an_owner(self) -> None:
        record = _record(bound=0.005, publication_policy="reporting")
        tables = self._tables(record)
        rows = red_cells.red_rows(record, tables)
        kinds = {row.key.kind: row for row in rows}
        assert kinds["paired"].fails_by.startswith(failures.UNACCOUNTED)
        assert "failures 2 of 200" in kinds["paired"].measured
        assert kinds["truth"].fails_by == f"failure accounting: {failures.BOUND_EXCEEDED}"
        assert kinds["accounting"].key.key == "hook/score audit"
        found = red_cells.findings(rows, {}, {record.slug: "reporting"}, [])
        assert len(found) == len(rows)
        assert all("has no owner" in line for line in found)

    def test_an_unaccountable_hook_row_reaches_the_ledger(self) -> None:
        """A hook with no ``account_failures`` counts no rows; the ledger still reads it."""
        record = _record(publication_policy="reporting")
        study = SimpleNamespace(scientific_failures=lambda frames: {})
        hook = failures.hook_accounting(study, _failures(("law", "1")), {}, record=record)
        tables = self._tables(record)
        accounting = tables["accounting"]
        tables["accounting"] = pd.concat(
            [accounting.loc[accounting["table"] != "hook"], hook], ignore_index=True
        )
        (row,) = [row for row in red_cells.red_rows(record, tables) if row.key.kind == "accounting"]
        assert row.fails_by == f"failure accounting: {failures.NOT_ACCOUNTABLE}"
        assert row.measured == "failing rows not counted, failures 1 of 200"

    def test_a_hand_edited_conclusion_is_refused(self) -> None:
        record = _record(bound=0.005, publication_policy="reporting")
        tables = self._tables(record)
        tables["equivalence"] = tables["equivalence"].assign(comparison_conclusion="equivalent")
        with pytest.raises(ValueError, match="legs conclude 'unaccounted'"):
            red_cells.red_rows(record, tables)


def test_the_score_audit_pairs_around_a_recorded_reference_failure() -> None:
    record = _record(replicates=3, extra_artifacts=(claims.FIT_DIAGNOSTICS_FILE,))
    diagnostics = pd.DataFrame(
        {
            "implementation": [SUBJECT] * 3 + [REFERENCE] * 2,
            "scenario": "law",
            "replicate": [0, 1, 2, 0, 2],
            "score_passed": True,
        }
    )
    data = {claims.FIT_DIAGNOSTICS: diagnostics, claims.FAILURES: _failures(("law", "1"))}
    audit = claims._score_audit_aggregates(record)
    assert audit["score_audited_fits"](data) == 3.0
    assert claims._aggregates(record)["reference_failures"](data) == 1.0
    assert claims._aggregates(record)["failure_rate"](data) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="not paired"):
        audit["score_audited_fits"]({claims.FIT_DIAGNOSTICS: diagnostics})
