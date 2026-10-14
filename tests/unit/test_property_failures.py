"""RM39 part b2: a recorded property failure costs its cells one replicate, and nothing else."""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence import property_failures, property_verdicts
from tests.studies.evidence.failures import FAILURE_COLUMNS
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    coverage_gain_interval,
    paired,
    paired_displacement,
    paired_spread_ratio_interval,
    require_accounted,
    se_ratio_deficit_interval,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import Recording


def _record_row(scenarios: str, replicate: int, *, group: str = "properties") -> dict:
    cell = scenarios.split(";")[0].split("/")[1]
    return {
        "implementation": "subject",
        "group_id": f"{group}/{cell}/r{replicate:09d}",
        "scenarios": scenarios,
        "replicates": str(replicate),
        "attempt": 1,
        "class": "fitting_failure",
        "stage": "cleverly fit",
        "call": "numpy.linalg:inv",
        "message": "LinAlgError: Singular matrix",
        "seed": "1",
    }


def _records(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(FAILURE_COLUMNS))


def _cell(
    family: str,
    cell: str,
    *,
    requested: int = 40,
    lost: tuple[int, ...] = (),
    seed: int = 0,
    role: str = "positive",
    failed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [replicate for replicate in range(requested) if replicate not in lost]
    estimates = rng.normal(0.0, 1.0, size=requested)[ids]
    return pd.DataFrame(
        {
            "property": family,
            "cell": cell,
            "role": role,
            "replicate": ids,
            "n": 500,
            "requested_replicates": requested,
            "failed_replicates": len(lost) if failed is None else failed,
            "truth": 0.0,
            "estimate": estimates,
            "std_error": 1.0 + 0.1 * rng.random(requested)[ids],
            "covered": (np.abs(estimates) < 1.96).astype(int),
            "rejected": (np.abs(estimates) > 1.96).astype(int),
        }
    ).loc[:, list(REPLICATE_COLUMNS)]


class TestRecordsAndStamps:
    def test_frame_sorts_the_run_s_records_and_is_none_without_one(self) -> None:
        recorded = Recording("subject")
        assert property_failures.frame(recorded) is None
        recorded.frames.append(_records(_record_row("p/b", 3), _record_row("p/a", 9)))
        published = property_failures.frame(recorded)
        assert published is not None
        assert list(published.columns) == list(FAILURE_COLUMNS)
        assert list(published["group_id"]) == sorted(published["group_id"])

    def test_without_a_record_the_rows_come_back_unchanged(self) -> None:
        rows = _cell("p", "a")
        assert property_failures.stamp(rows, None) is rows
        assert property_failures.stamp(rows, _records()) is rows

    def test_a_multi_cell_record_stamps_each_cell_once(self) -> None:
        rows = pd.concat(
            [
                _cell("p", "a", lost=(3,), failed=0),
                _cell("p", "b", lost=(3, 5), failed=0),
                _cell("q", "c"),
            ],
            ignore_index=True,
        )
        records = _records(_record_row("p/a;p/b", 3), _record_row("p/b", 5))
        stamped = property_failures.stamp(rows, records)
        failed = stamped.groupby(["property", "cell"])["failed_replicates"].first().to_dict()
        assert failed == {("p", "a"): 1, ("p", "b"): 2, ("q", "c"): 0}
        assert stamped["failed_replicates"].dtype == rows["failed_replicates"].dtype
        property_failures.reconcile(stamped, records)

    def test_two_records_of_one_cell_replicate_are_refused(self) -> None:
        with pytest.raises(ValueError, match="two property failure records"):
            property_failures.expand(_records(_record_row("p/a", 3), _record_row("p/a;p/b", 3)))


class TestReconcile:
    def test_a_short_cell_without_a_record_is_refused(self) -> None:
        with pytest.raises(ValueError, match="do not cover"):
            property_failures.reconcile(_cell("p", "a", lost=(4,), failed=0), None)

    def test_a_record_of_a_replicate_that_survived_is_refused(self) -> None:
        records = _records(_record_row("p/a", 4))
        rows = property_failures.stamp(_cell("p", "a"), records)
        with pytest.raises(ValueError, match="both survived and failed"):
            property_failures.reconcile(rows, records)

    def test_a_record_of_an_unknown_cell_is_refused(self) -> None:
        records = _records(_record_row("p/a;p/ghost", 4))
        rows = property_failures.stamp(_cell("p", "a", lost=(4,)), records)
        with pytest.raises(ValueError, match="p/ghost: a record names a cell with no surviving"):
            property_failures.reconcile(rows, records)

    def test_a_stamp_that_disagrees_with_the_records_is_refused(self) -> None:
        records = _records(_record_row("p/a", 4))
        rows = _cell("p", "a", lost=(4,), failed=2)
        with pytest.raises(ValueError, match="stamped 2 failures, recorded 1"):
            property_failures.reconcile(rows, records)

    def test_a_derived_control_loses_its_base_s_replicates(self) -> None:
        derived = {"p/base": ("p/base_control",)}
        rows = pd.concat(
            [_cell("p", "base", lost=(4,)), _cell("p", "base_control", lost=(4,))],
            ignore_index=True,
        )
        property_failures.reconcile(
            rows, _records(_record_row("p/base;p/base_control", 4)), derived=derived
        )
        # A deliberate mutation: the control's record names another replicate.
        rows = pd.concat(
            [_cell("p", "base", lost=(4,)), _cell("p", "base_control", lost=(5,))],
            ignore_index=True,
        )
        records = _records(_record_row("p/base", 4), _record_row("p/base_control", 5))
        with pytest.raises(ValueError, match="lost other replicates than its base"):
            property_failures.reconcile(rows, records, derived=derived)


class TestPairedHelpers:
    """p13: arms that lost different replicates read the replicates both kept."""

    @staticmethod
    def _arms() -> tuple[pd.DataFrame, pd.DataFrame]:
        left = _cell("fam", "left", lost=(2,), seed=1)
        right = _cell("fam", "right", lost=(7,), seed=2, role="control")
        return left, right

    def test_paired_reads_the_intersection_in_each_arm_s_order(self) -> None:
        left, right = self._arms()
        kept_left, kept_right = paired(left.iloc[::-1], right)
        assert set(kept_left["replicate"]) == set(range(40)) - {2, 7}
        assert list(kept_left["replicate"]) == sorted(kept_left["replicate"], reverse=True)
        assert list(kept_right["replicate"]) == sorted(kept_right["replicate"])

    def test_a_misalignment_no_stamp_explains_still_refuses(self) -> None:
        left, right = self._arms()
        with pytest.raises(ValueError, match="do not explain"):
            paired(left, right.assign(failed_replicates=0))
        with pytest.raises(ValueError, match="not paired"):
            paired(left.drop(columns="failed_replicates"), right)
        with pytest.raises(ValueError, match="request 40 and 41"):
            paired(left, _cell("fam", "right", requested=41, lost=(7,)))

    def test_each_helper_equals_its_value_on_the_shared_replicates(self) -> None:
        left, right = self._arms()
        shared = set(range(40)) - {2, 7}
        complete_left = left.loc[left["replicate"].isin(shared)].assign(failed_replicates=0)
        complete_right = right.loc[right["replicate"].isin(shared)].assign(failed_replicates=0)
        complete_left["requested_replicates"] = len(shared)
        complete_right["requested_replicates"] = len(shared)
        # Renumber the shared ids so the complete-case arms are complete.
        renumber = {old: new for new, old in enumerate(sorted(shared))}
        for arm in (complete_left, complete_right):
            arm["replicate"] = arm["replicate"].map(renumber)
        rows = pd.concat([left, right], ignore_index=True)
        complete_rows = pd.concat([complete_left, complete_right], ignore_index=True)
        assert paired_displacement(rows, "fam", "left", "right") == paired_displacement(
            complete_rows, "fam", "left", "right"
        )
        options = {"replicates": 200, "confidence_level": 0.99, "seed": 5}
        assert se_ratio_deficit_interval(left, right, **options) == se_ratio_deficit_interval(
            complete_left, complete_right, **options
        )
        assert paired_spread_ratio_interval(left, right, **options) == paired_spread_ratio_interval(
            complete_left, complete_right, **options
        )
        assert coverage_gain_interval(left, right, **options) == coverage_gain_interval(
            complete_left, complete_right, **options
        )


@functools.cache
def _committed(slug_module: str) -> tuple[StudyRecord, pd.DataFrame]:
    import importlib

    study = importlib.import_module(f"tests.studies.{slug_module}").STUDY
    return study, pd.read_csv(study.artifact("property-replicates.csv.gz"))


def _lose(rows: pd.DataFrame, losses: dict[str, int]) -> pd.DataFrame:
    """Drop one replicate from each named ``property/cell`` and stamp the loss."""
    records = _records(*(_record_row(name, replicate) for name, replicate in losses.items()))
    kept = rows
    for name, replicate in losses.items():
        family, cell = name.split("/")
        kept = kept.loc[
            ~(
                (kept["property"] == family)
                & (kept["cell"] == cell)
                & (kept["replicate"] == replicate)
            )
        ]
    stamped = property_failures.stamp(kept.reset_index(drop=True), records)
    property_failures.reconcile(stamped, records)
    return stamped


@pytest.mark.parametrize(
    ("module", "losses"),
    [
        (
            "canonical_ctmle_selector",
            {"selector_necessity/collaborative": 3, "selector_necessity/empty_control": 11},
        ),
        (
            "canonical_multi_arm_ctmle_selector",
            {"selector_necessity/greedy": 3, "selector_necessity/empty_control": 11},
        ),
    ],
)
def test_a_selector_pair_that_lost_different_replicates_still_summarizes(
    module: str, losses: dict[str, int]
) -> None:
    """p13 and F3: the two module-local pairing sites read the replicates both arms kept."""
    study, rows = _committed(module)
    summary = study.properties().summarize_properties(_lose(rows, losses))
    necessity = summary.loc[summary["property"] == "selector_necessity"]
    assert np.isfinite(necessity["rmse_ratio"]).all()
    # Without the stamp the same gap still stops the summary.
    unstamped = _lose(rows, losses).assign(failed_replicates=0)
    with pytest.raises(ValueError, match="do not explain"):
        study.properties().summarize_properties(unstamped)


class TestVerdictBudget:
    """F1: a contraction rung keeps its ids below the verdict budget and never refills."""

    family = property_verdicts.CONTRACTION_FAMILY

    def _ladder(self, *, budget: int, lost: tuple[int, ...]) -> pd.DataFrame:
        return pd.concat(
            [
                _cell(self.family, "outcome_correct_n500", requested=budget, lost=lost),
                _cell(self.family, "outcome_correct_n2000", requested=20),
            ],
            ignore_index=True,
        )

    def test_p14_a_lost_id_below_the_budget_takes_no_later_replicate(self) -> None:
        rows = property_verdicts._at_verdict_budget(self._ladder(budget=60, lost=(5,)), 20)
        rung = rows.loc[rows["cell"] == "outcome_correct_n500"]
        assert list(rung["replicate"]) == [replicate for replicate in range(20) if replicate != 5]
        assert rung["failed_replicates"].unique().tolist() == [1]
        assert rung["requested_replicates"].unique().tolist() == [20]
        require_accounted(rows)

    def test_p14b_a_lost_id_above_the_budget_leaves_the_rung_complete(self) -> None:
        rows = property_verdicts._at_verdict_budget(self._ladder(budget=60, lost=(45,)), 20)
        rung = rows.loc[rows["cell"] == "outcome_correct_n500"]
        assert list(rung["replicate"]) == list(range(20))
        assert rung["failed_replicates"].unique().tolist() == [0]
        require_accounted(rows)

    def test_p14c_a_rung_at_the_budget_that_lost_an_id_is_not_refused(self) -> None:
        rows = property_verdicts._at_verdict_budget(self._ladder(budget=20, lost=(9,)), 20)
        rung = rows.loc[rows["cell"] == "outcome_correct_n500"]
        assert len(rung) == 19
        assert rung["failed_replicates"].unique().tolist() == [1]

    def test_a_rung_requested_below_the_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requested 19 replications"):
            property_verdicts._at_verdict_budget(self._ladder(budget=19, lost=()), 20)

    def test_without_a_failure_the_rows_are_the_first_budget_replicates(self) -> None:
        ladder = self._ladder(budget=60, lost=())
        rows = property_verdicts._at_verdict_budget(ladder, 20)
        expected = (
            ladder.loc[ladder["cell"] == "outcome_correct_n500"]
            .sort_values("replicate")
            .head(20)
            .assign(requested_replicates=20)
        )
        pd.testing.assert_frame_equal(
            rows.loc[rows["cell"] == "outcome_correct_n500"].reset_index(drop=True),
            expected.reset_index(drop=True),
        )


def _calibration_record() -> StudyRecord:
    return StudyRecord(
        name="x",
        slug="x",
        artifacts=Path("."),
        document="x.md",
        anchor="x",
        scenarios={"law": ("ate",)},
        replicates=10,
        n=50,
        seed=1,
    )


class TestCalibrationNoise:
    """p16 and F2: a lost base replicate leaves every other ``noise_control`` row unchanged."""

    @staticmethod
    def _controls(base: pd.DataFrame) -> pd.DataFrame:
        return property_verdicts.calibration_controls(
            base,
            _calibration_record(),
            labels=("z0",),
            efficiency_bounds={"z0": 2.0},
            calibration_n=400,
            shrunken_se_factor=0.8,
            critical=1.96,
        )

    def test_dropping_a_base_replicate_moves_no_other_noise_row(self) -> None:
        complete = _cell("interval_calibration", "z0__correctly_specified")
        lost = _cell("interval_calibration", "z0__correctly_specified", lost=(6,))
        noise = {
            name: frame.loc[frame["cell"] == "z0__noise_control"].set_index("replicate")
            for name, frame in (
                ("complete", self._controls(complete)),
                ("lost", self._controls(lost)),
            )
        }
        assert 6 not in noise["lost"].index
        pd.testing.assert_frame_equal(
            noise["lost"].drop(columns="failed_replicates"),
            noise["complete"].drop(index=6).drop(columns="failed_replicates"),
        )

    def test_the_noise_equals_a_draw_sized_to_complete_rows(self) -> None:
        base = _cell("interval_calibration", "z0__correctly_specified")
        noise = self._controls(base)
        noisy = noise.loc[noise["cell"] == "z0__noise_control", "estimate"].to_numpy()
        rng = np.random.default_rng(
            property_verdicts.stream_seed(_calibration_record(), "efficiency_noise", "z0")
        )
        expected = base["estimate"].to_numpy() + rng.normal(scale=2.0 / np.sqrt(400), size=40)
        np.testing.assert_array_equal(noisy, expected)

    @pytest.mark.parametrize("mutation", ["unsorted", "duplicate", "out of range"])
    def test_unsorted_or_duplicate_base_ids_are_refused(self, mutation: str) -> None:
        base = _cell("interval_calibration", "z0__correctly_specified")
        if mutation == "unsorted":
            base = base.iloc[::-1]
        elif mutation == "duplicate":
            base.loc[3, "replicate"] = 2
        else:
            base.loc[39, "replicate"] = 40
        with pytest.raises(ValueError, match="unique, ascending replicate ids"):
            self._controls(base)


def test_a_rate_row_publishes_the_sum_of_its_rungs_stamps() -> None:
    """F15: ``fitted_rate_row`` counts every rung's recorded failures, 0 without one."""
    record = dataclasses.replace(
        _calibration_record(),
        margins=dataclasses.replace(_calibration_record().margins, bootstrap_replicates=50),
    )
    rungs = [
        _cell("root_n_and_efficiency", f"n_{size}", lost=lost, seed=size)
        for size, lost in ((500, (1,)), (2000, ()), (8000, (3, 4)))
    ]
    for rung, size in zip(rungs, (500, 2000, 8000), strict=True):
        rung["n"] = size
    ladder = pd.concat(rungs, ignore_index=True)
    row = property_verdicts.fitted_rate_row(
        ladder,
        record,
        ["property", "cell", "failed_replicates"],
        ladder_property="root_n_and_efficiency",
        property_name="root_n_rate",
        cell="empirical_sd",
        role="positive",
        statistic="spread",
        seed_labels=("root_n_rate", "empirical_sd"),
        verdict=lambda fitted: True,
    )
    assert row["failed_replicates"] == 3
    complete = ladder.assign(failed_replicates=0)
    assert (
        property_verdicts.fitted_rate_row(
            complete,
            record,
            ["property", "cell", "failed_replicates"],
            ladder_property="root_n_and_efficiency",
            property_name="root_n_rate",
            cell="empirical_sd",
            role="positive",
            statistic="spread",
            seed_labels=("root_n_rate", "empirical_sd"),
            verdict=lambda fitted: True,
        )["failed_replicates"]
        == 0
    )
