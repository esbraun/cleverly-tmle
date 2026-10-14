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
    subject_failed: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Calibrated, unbiased subject rows; reference rows with ``failed`` left out.

    The subject rows of ``subject_failed`` are left out as well.
    """
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
                    if replicate in (failed if implementation == REFERENCE else subject_failed):
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


def _failures(
    *groups: tuple[str, str], implementation: str = REFERENCE, kind: str = "fitting_failure"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "implementation": implementation,
                "group_id": f"r{index:09d}",
                "scenarios": scenarios,
                "replicates": replicates,
                "attempt": 1,
                "class": kind,
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

    def test_a_subject_failure_covers_the_subjects_missing_rows(self) -> None:
        record = _record()
        rows = _rows(record, subject_failed=(1, 6))
        failures.reconcile(rows, _failures(("law", "1;6"), implementation=SUBJECT), record=record)
        with pytest.raises(ValueError, match="do not cover"):
            failures.reconcile(rows, _failures(("law", "1"), implementation=SUBJECT), record=record)

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


def _joined(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _replicates(values: tuple[int, ...]) -> str:
    return ";".join(str(value) for value in values)


class TestSubjectFailures:
    """The subject's failed replicates take its worst case and its own bound (RM39b).

    Each leg test is a mutation control: the subject's complete cases pass and its least
    favourable fill does not, so a fill at the subject's best case would turn it green.
    """

    LOST = tuple(range(190, 200))

    def _accounting(
        self,
        record: StudyRecord,
        *,
        subject_failed: tuple[int, ...] = (),
        reference_failed: tuple[int, ...] = (),
        recorded: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rows = _rows(record, failed=reference_failed, subject_failed=subject_failed)
        if recorded is None:
            parts = []
            if subject_failed:
                parts.append(
                    _failures(("law", _replicates(subject_failed)), implementation=SUBJECT)
                )
            if reference_failed:
                parts.append(_failures(("law", _replicates(reference_failed))))
            recorded = _joined(*parts)
        failures.reconcile(rows, recorded, record=record)
        return failures.paired_accounting(rows, recorded, record=record, n_jobs=1)

    def _subject_record(self, **changes: object) -> StudyRecord:
        return _record(failure_bound={REFERENCE: 0.05, SUBJECT: 0.05}, **changes)

    def test_the_rmse_leg_fills_the_subject_at_its_farthest_survivor(self) -> None:
        accounting = self._accounting(self._subject_record(), subject_failed=self.LOST)
        rmse = accounting.loc[accounting["leg"] == "rmse ratio"].iloc[0]
        assert rmse["complete_case"] >= 0 > rmse["worst_case"]
        assert rmse["imputation"] == "subject estimate at farthest"

    def test_the_coverage_leg_counts_each_failed_subject_replicate_as_a_miss(self) -> None:
        accounting = self._accounting(self._subject_record(), subject_failed=self.LOST)
        coverage = accounting.loc[accounting["leg"] == "coverage difference"].iloc[0]
        assert coverage["complete_case"] >= 0 > coverage["worst_case"]
        assert coverage["imputation"] == "subject missed"

    def test_the_calibration_leg_fills_the_subject_at_a_spread_extreme(self) -> None:
        """Mutation control s3: the complete cases pass, the spread-extreme fill does not."""
        accounting = self._accounting(self._subject_record(), subject_failed=self.LOST)
        calibration = accounting.loc[accounting["leg"] == "calibration excess"].iloc[0]
        assert calibration["complete_case"] >= 0 > calibration["worst_case"]
        assert calibration["imputation"].startswith("subject SE at min")

    def test_mixed_failures_keep_each_replicate_s_two_sides_aligned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control s5: each replicate both sides fitted stays paired with itself.

        The fills reuse survivor values, so the check reads the pairs of the shared replicates:
        each one must appear in every full-length bootstrap input, and each side's real value
        of a replicate only the other side lost must appear too.
        """
        record = self._subject_record()
        subject_lost, reference_lost = (3, 4, 5), (7, 8, 9, 10)
        rows = _rows(record, failed=reference_lost, subject_failed=subject_lost)
        wide = (
            rows.pivot(index="replicate", columns="implementation", values="estimate")
            .sort_index()
            .reindex(range(record.replicates))
        )
        shared = wide.dropna()
        captured: list[dict[str, np.ndarray]] = []
        real = failures.bootstrap

        def spy(data: dict[str, np.ndarray], *args: object, **options: object) -> object:
            captured.append({name: np.asarray(values) for name, values in data.items()})
            return real(data, *args, **options)  # type: ignore[arg-type]

        monkeypatch.setattr(failures, "bootstrap", spy)
        self._accounting(record, subject_failed=subject_lost, reference_failed=reference_lost)
        full = [
            data
            for data in captured
            if len(data.get("subject_estimate", ())) == record.replicates
            and "reference_estimate" in data
        ]
        assert full
        for data in full:
            pairs = set(zip(data["subject_estimate"], data["reference_estimate"], strict=True))
            assert set(zip(shared[SUBJECT], shared[REFERENCE], strict=True)) <= pairs
            assert set(wide.loc[list(subject_lost), REFERENCE]) <= set(data["reference_estimate"])
            assert set(wide.loc[list(reference_lost), SUBJECT]) <= set(data["subject_estimate"])

    def test_the_subject_worst_case_fills_reach_the_bootstrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = self._subject_record()
        rows = _rows(record, subject_failed=self.LOST)
        wide = rows.pivot(
            index="replicate",
            columns="implementation",
            values=["estimate", "covered", "inference_estimate", "std_error"],
        ).sort_index()
        captured: list[dict[str, np.ndarray]] = []
        real = failures.bootstrap

        def spy(data: dict[str, np.ndarray], *args: object, **options: object) -> object:
            captured.append({name: np.asarray(values) for name, values in data.items()})
            return real(data, *args, **options)  # type: ignore[arg-type]

        monkeypatch.setattr(failures, "bootstrap", spy)
        failures.paired_legs(
            wide,
            (),
            record=record,
            truth=TRUTH,
            comparable=False,
            seed=1,
            subject_failed=self.LOST,
        )
        worst = next(data for data in captured if len(data["subject_estimate"]) == 200)
        k = len(self.LOST)
        survivors = wide.drop(index=list(self.LOST))[("estimate", SUBJECT)].to_numpy(float)
        farthest = survivors[np.argmax(np.abs(survivors - TRUTH))]
        assert np.all(worst["subject_estimate"][-k:] == farthest)
        assert np.all(worst["subject_covered"][-k:] == 0.0)
        np.testing.assert_array_equal(worst["subject_estimate"][:-k], survivors)
        # The reference keeps its real values on the replicates only the subject lost.
        np.testing.assert_array_equal(
            worst["reference_estimate"][-k:],
            wide.loc[list(self.LOST), ("estimate", REFERENCE)].to_numpy(float),
        )

    def test_the_calibration_leg_crosses_the_two_sides_candidate_sets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subject (k + 2) candidates times reference 3 * (k + 2), the complete case aside."""
        record = self._subject_record()
        subject_lost, reference_lost = (3, 4, 5), (7, 8)
        rows = _rows(record, failed=reference_lost, subject_failed=subject_lost)
        wide = (
            rows.pivot(
                index="replicate",
                columns="implementation",
                values=["estimate", "covered", "inference_estimate", "std_error"],
            )
            .sort_index()
            .reindex(range(record.replicates))
        )
        survivors = [r for r in wide.index if r not in subject_lost]
        wide.loc[survivors, ("std_error", SUBJECT)] = np.linspace(0.05, 0.15, len(survivors))
        pairs: set[bytes] = set()
        subject_sets: set[bytes] = set()
        reference_sets: set[bytes] = set()

        def spy(
            subject_se: np.ndarray,
            subject_inference: np.ndarray,
            reference_se: np.ndarray,
            reference_inference: np.ndarray,
            **options: object,
        ) -> float:
            if len(subject_se) == record.replicates:
                one = subject_se.tobytes() + subject_inference.tobytes()
                other = reference_se.tobytes() + reference_inference.tobytes()
                subject_sets.add(one)
                reference_sets.add(other)
                pairs.add(one + b"|" + other)
            return 0.0

        monkeypatch.setattr(failures, "_calibration_slack", spy)
        legs, _ = failures.paired_legs(
            wide,
            reference_lost,
            record=record,
            truth=TRUTH,
            comparable=True,
            seed=1,
            subject_failed=subject_lost,
        )
        assert len(subject_sets) == len(subject_lost) + 2
        assert len(pairs) == len(subject_sets) * len(reference_sets)
        assert len(reference_sets) <= 3 * (len(reference_lost) + 2)
        calibration = legs[-1]
        assert calibration.name == "calibration excess"

    def test_both_sides_failing_drops_the_union_and_writes_each_bound(self) -> None:
        accounting = self._accounting(
            self._subject_record(), subject_failed=(2, 5), reference_failed=(1, 2)
        )
        assert accounting["dropped_replications"].unique().tolist() == [3]
        assert (
            accounting["subject_failed"].unique().tolist(),
            accounting["reference_failed"].unique().tolist(),
        ) == ([2], [2])
        bounds = accounting.loc[accounting["leg"].str.endswith("failure bound")]
        assert dict(zip(bounds["leg"], bounds["complete_case"], strict=True)) == {
            "failure bound": 2.0,
            "subject failure bound": 2.0,
        }
        witness = accounting.iloc[0]
        for column in (
            "witness_failed_error",
            "witness_succeeded_error",
            "witness_reference_error_on_subject_failed",
            "witness_reference_error_on_shared",
        ):
            assert math.isfinite(witness[column])

    def test_a_subject_over_its_own_bound_is_red_on_the_paired_cell(self) -> None:
        """Mutation control: the reference's bound alone would pass this cell."""
        record = _record(failure_bound={REFERENCE: 0.05, SUBJECT: 0.005})
        accounting = self._accounting(record, subject_failed=(4, 9))
        assert set(accounting["reason"]) == {failures.BOUND_EXCEEDED}
        bound = accounting.loc[accounting["leg"] == "subject failure bound"].iloc[0]
        assert (bound["complete_case"], bound["worst_case"], bound["leg_passed"]) == (
            2.0,
            1.0,
            False,
        )
        reference = accounting.loc[accounting["leg"] == "failure bound"].iloc[0]
        assert reference["leg_passed"]

    def test_a_subject_truth_cell_takes_its_own_legs_and_bound(self) -> None:
        record = self._subject_record()
        rows = _rows(record, subject_failed=(1, 2))
        recorded = _failures(("law", "1;2"), implementation=SUBJECT)
        frame = failures.truth_accounting(rows, recorded, record=record, n_jobs=1)
        assert set(frame["implementation"]) == {SUBJECT}
        assert set(frame["leg"]) == {"failure bound", "bias", "coverage", "se ratio"}
        assert frame["gate_passed"].all()
        undeclared = failures.truth_accounting(rows, recorded, record=_record(), n_jobs=1)
        assert set(undeclared["reason"]) == {failures.BOUND_EXCEEDED}

    def test_an_induced_reference_loss_counts_against_the_subject(self) -> None:
        """The reference loses the replicate, and the subject's bound pays for it."""
        lost = (4, 9)
        tight = _record(failure_bound={REFERENCE: 0.005, SUBJECT: 0.05})
        recorded = _joined(
            _failures(("law", _replicates(lost)), implementation=SUBJECT),
            _failures(("law", _replicates(lost)), kind=failures.PAIRED_INPUT_MISSING),
        )
        rows = _rows(tight, failed=lost, subject_failed=lost)
        truth = failures.truth_accounting(rows, recorded, record=tight, n_jobs=1)
        reference = truth.loc[truth["implementation"] == REFERENCE]
        assert set(reference["leg"]) == {"failure bound", "bias", "coverage", "se ratio"}
        bound = reference.loc[reference["leg"] == "failure bound"].iloc[0]
        assert bound["complete_case"] == 0.0 and bound["leg_passed"]
        paired = self._accounting(
            tight, subject_failed=lost, reference_failed=lost, recorded=recorded
        )
        assert failures.BOUND_EXCEEDED not in set(paired["reason"])
        bounds = paired.loc[paired["leg"].str.endswith("failure bound")]
        assert bounds[["leg", "complete_case", "leg_passed"]].values.tolist() == [
            ["failure bound", 0.0, True],
            ["subject failure bound", 2.0, True],
        ]
        assert (paired["subject_failed"].iloc[0], paired["reference_failed"].iloc[0]) == (2, 0)
        # Mutation control: read as the reference's own failure, the same loss is over its bound.
        own = _joined(
            _failures(("law", _replicates(lost)), implementation=SUBJECT),
            _failures(("law", _replicates(lost))),
        )
        relabelled = self._accounting(
            tight, subject_failed=lost, reference_failed=lost, recorded=own
        )
        assert set(relabelled["reason"]) == {failures.BOUND_EXCEEDED}


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


class TestSubjectHooks:
    """The hook, probe and claims reads split each side's failures (RM39b M1)."""

    def _study(self, record: StudyRecord, gates: dict[str, pd.DataFrame]) -> SimpleNamespace:
        subject, reference = record.implementation, str(record.reference)
        return SimpleNamespace(
            scientific_failures=lambda frames: gates,
            account_failures=lambda lost, frames: failures.audits(
                gates,
                missing={
                    "score audit": failures.missing_fits(lost, record, subject)
                    + failures.missing_fits(lost, record, reference),
                    "solver": failures.missing_fits(lost, record, subject),
                },
            ),
        )

    def test_a_hook_row_names_the_side_and_its_bound(self) -> None:
        record = _record(failure_bound={REFERENCE: 0.05, SUBJECT: 0.01})
        study = self._study(record, {"score audit": pd.DataFrame(), "solver": pd.DataFrame()})
        subject_only = failures.hook_accounting(
            study, _failures(("law", "1"), implementation=SUBJECT), {}, record=record
        )
        assert set(subject_only["implementation"]) == {SUBJECT}
        assert set(subject_only["allowed"]) == {2}
        assert subject_only.set_index("leg")["gate_passed"].to_dict() == {
            "score audit": False,
            "solver": False,
        }
        both = failures.hook_accounting(
            study,
            _joined(_failures(("law", "1"), implementation=SUBJECT), _failures(("law", "1;3"))),
            {},
            record=record,
        )
        assert set(both["implementation"]) == {f"{SUBJECT}+{REFERENCE}"}
        assert both["allowed"].isna().all()
        row = both.set_index("leg").loc["score audit"]
        assert (row["subject_failed"], row["reference_failed"], row["complete_case"]) == (1, 2, 3.0)

    def test_an_induced_reference_record_is_the_subjects_on_the_hook_row(self) -> None:
        record = _record(failure_bound={REFERENCE: 0.05, SUBJECT: 0.05})
        study = self._study(record, {"score audit": pd.DataFrame(), "solver": pd.DataFrame()})
        frame = failures.hook_accounting(
            study,
            _joined(
                _failures(("law", "4"), implementation=SUBJECT),
                _failures(("law", "4"), kind=failures.PAIRED_INPUT_MISSING),
            ),
            {},
            record=record,
        )
        assert set(frame["implementation"]) == {SUBJECT}
        # Both fits are missing, so the shared-bar audit counts two.
        assert frame.set_index("leg")["complete_case"].to_dict() == {
            "score audit": 2.0,
            "solver": 1.0,
        }

    def test_a_probe_keeps_a_replication_only_the_subject_lost(self) -> None:
        record = _record(replicates=4)
        probe = pd.DataFrame({"scenario": "law", "replicate": [0, 1, 2, 3], "passed": True})
        audits = failures.probe_audits(
            probe,
            _failures(("law", "1"), implementation=SUBJECT),
            record=record,
            scenarios=("law",),
            coverage="coverage",
            audit="probe",
        )
        assert audits["passed"].all()
        # Mutation control: read as the reference's, the same record is a probe overlap.
        overlap = failures.probe_audits(
            probe,
            _failures(("law", "1")),
            record=record,
            scenarios=("law",),
            coverage="coverage",
            audit="probe",
        )
        assert overlap.set_index("leg")["passed"].to_dict() == {"coverage": False, "probe": True}

    def test_the_claims_count_each_side_and_the_score_audit_pairs_around_both(
        self, tmp_path: Path
    ) -> None:
        record = _record(
            replicates=4, extra_artifacts=(claims.FIT_DIAGNOSTICS_FILE,), artifacts=tmp_path
        )
        diagnostics = pd.DataFrame(
            {
                "implementation": [SUBJECT] * 3 + [REFERENCE] * 3,
                "scenario": "law",
                "replicate": [0, 2, 3, 0, 1, 3],
                "score_passed": True,
            }
        )
        diagnostics.to_csv(record.artifact(claims.FIT_DIAGNOSTICS_FILE), index=False)
        recorded = _joined(_failures(("law", "1"), implementation=SUBJECT), _failures(("law", "2")))
        data = {claims.FIT_DIAGNOSTICS: diagnostics, claims.FAILURES: recorded}
        aggregates = claims._aggregates(record)
        assert aggregates["subject_failures"](data) == 1.0
        assert aggregates["reference_failures"](data) == 1.0
        # A reference fit the subject's loss left without an input is the subject's failure.
        induced = _joined(recorded, _failures(("law", "1"), kind=failures.PAIRED_INPUT_MISSING))
        with_induced = {**data, claims.FAILURES: induced}
        assert aggregates["reference_failures"](with_induced) == 1.0
        assert aggregates["failure_rate"](with_induced) == 0.25
        assert claims._score_audit_aggregates(record)["score_audited_fits"](data) == 3.0
        # Mutation control: without the subject's record the pairing does not close.
        alone = {**data, claims.FAILURES: _failures(("law", "2"))}
        with pytest.raises(ValueError, match="not paired"):
            claims._score_audit_aggregates(record)["score_audited_fits"](alone)

    def test_a_subject_failure_reaches_the_ledger_with_an_owner_row(self) -> None:
        record = _record(failure_bound={REFERENCE: 0.05}, publication_policy="reporting")
        lost = (4, 9)
        rows = _rows(record, subject_failed=lost)
        recorded = _failures(("law", "4;9"), implementation=SUBJECT)
        summaries = summarize(rows, truth_varies=False)
        performance = independent_performance_tests(rows, record=record, n_jobs=1)
        accounting = failures.truth_accounting(rows, recorded, record=record, n_jobs=1)
        performance = failures.correct_performance(performance, accounting)
        paired = equivalence(rows, summaries, performance, record=record, n_jobs=1)
        paired_accounting = failures.paired_accounting(rows, recorded, record=record, n_jobs=1)
        tables = {
            "performance": performance,
            "equivalence": failures.correct_paired(paired, paired_accounting),
            "properties": pd.DataFrame(
                columns=["property", "cell", "role", "passed", "property_passed"]
            ),
            "accounting": pd.concat([accounting, paired_accounting], ignore_index=True),
        }
        red = red_cells.red_rows(record, tables)
        kinds = {row.key.kind: row for row in red}
        # The subject declared no bound, so its truth cell and the paired cell are red.
        assert kinds["truth"].role == SUBJECT
        assert kinds["truth"].fails_by == f"failure accounting: {failures.BOUND_EXCEEDED}"
        assert kinds["paired"].fails_by.startswith(failures.UNACCOUNTED)
        found = red_cells.findings(red, {}, {record.slug: "reporting"}, [])
        assert len(found) == len(red)
        assert all("has no owner" in line for line in found)


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


def test_the_score_audit_pairs_around_a_recorded_reference_failure(tmp_path: Path) -> None:
    record = _record(
        replicates=3, extra_artifacts=(claims.FIT_DIAGNOSTICS_FILE,), artifacts=tmp_path
    )
    diagnostics = pd.DataFrame(
        {
            "implementation": [SUBJECT] * 3 + [REFERENCE] * 2,
            "scenario": "law",
            "replicate": [0, 1, 2, 0, 2],
            "score_passed": True,
        }
    )
    # The aggregates choose the score audit from the committed file's header.
    diagnostics.to_csv(record.artifact(claims.FIT_DIAGNOSTICS_FILE), index=False)
    data = {claims.FIT_DIAGNOSTICS: diagnostics, claims.FAILURES: _failures(("law", "1"))}
    audit = claims._score_audit_aggregates(record)
    assert audit["score_audited_fits"](data) == 3.0
    assert claims._aggregates(record)["reference_failures"](data) == 1.0
    assert claims._aggregates(record)["failure_rate"](data) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="not paired"):
        audit["score_audited_fits"]({claims.FIT_DIAGNOSTICS: diagnostics})


class TestPrimaryBound:
    """RM39b2: each side's primary bound is 0.01, declared for the study's next run."""

    def test_every_registered_study_declares_each_side_s_bound(self) -> None:
        from tests.studies.evidence.registry import registered

        for study in registered():
            assert dict(study.failure_bound) == dict.fromkeys(study.implementations, 0.01), (
                study.slug
            )

    def test_one_failure_reaches_the_legs_at_0_01_and_exceeds_the_bound_at_0(self) -> None:
        """A failed replicate inside the allowance is judged by its worst case."""
        within = _truth(_record(bound=0.01), _rows(_record(bound=0.01), failed=(4,)), "4")
        assert set(within["reason"]) == {failures.WITHIN_BOUND}
        exceeded = _truth(_record(bound=0.0), _rows(_record(bound=0.0), failed=(4,)), "4")
        assert set(exceeded["reason"]) == {failures.BOUND_EXCEEDED}

    def test_a_manifest_agrees_with_the_record_by_its_key_or_by_no_failure(self) -> None:
        record = _record(bound=0.01)
        stated = {"failure_bound": {REFERENCE: 0.01}}
        assert failures.bound_consistent(stated, record, failures_recorded=True)
        assert failures.bound_consistent({}, record, failures_recorded=False)
        # Mutation controls: a different stated bound, and failures decided under no key.
        other = {"failure_bound": {REFERENCE: 0.02}}
        assert not failures.bound_consistent(other, record, failures_recorded=False)
        assert not failures.bound_consistent({}, record, failures_recorded=True)

    @staticmethod
    def _committed(tmp_path: Path, bounds: dict[str, float] | None, *groups: pd.DataFrame) -> Path:
        import json

        configuration = {} if bounds is None else {"failure_bound": bounds}
        (tmp_path / "manifest.json").write_text(json.dumps({"configuration": configuration}))
        if groups:
            pd.concat(groups, ignore_index=True).to_csv(tmp_path / failures.FAILURES, index=False)
        return tmp_path

    def test_a_bound_is_never_raised_after_a_run_that_lost_replicates(self, tmp_path: Path) -> None:
        record = _record(bound=0.01)
        here = self._committed(tmp_path, None, _failures(("law", "3")))
        assert failures.raised_after_failures(here, record) == [REFERENCE]
        assert failures.raised_after_failures(here, _record(bound=0.0)) == []
        self._committed(tmp_path, {REFERENCE: 0.01})
        assert failures.raised_after_failures(here, record) == []
        assert failures.raised_after_failures(here, _record(bound=0.005)) == []

    def test_a_paired_input_record_counts_against_the_subject(self, tmp_path: Path) -> None:
        record = _record(failure_bound={REFERENCE: 0.0, SUBJECT: 0.01})
        missing = _failures(("law", "3"), kind=failures.PAIRED_INPUT_MISSING)
        here = self._committed(tmp_path, None, missing)
        assert failures.raised_after_failures(here, record) == [SUBJECT]

    def test_a_run_without_failures_may_take_any_declared_bound(self, tmp_path: Path) -> None:
        assert failures.raised_after_failures(self._committed(tmp_path, None), _record()) == []

    def test_the_claims_quote_the_bound_the_committed_run_applied(self, tmp_path: Path) -> None:
        """A bound declared for the next run is not quoted for a run made without it."""
        record = _record(bound=0.01, artifacts=tmp_path)
        self._committed(tmp_path, None)
        declared = claims.thresholds(record)
        assert "margin:reference_failure_bound" not in declared
        assert "margin:reference_failures_allowed" not in declared
        self._committed(tmp_path, {REFERENCE: 0.01})
        declared = claims.thresholds(record)
        assert declared["margin:reference_failure_bound"] == 0.01
        assert declared["margin:reference_failures_allowed"] == 2.0
        assert "margin:subject_failure_bound" not in declared
