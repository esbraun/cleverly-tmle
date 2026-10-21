"""RM39 part b2: a recorded property failure costs its cells one replicate, and nothing else."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.data import CausalData
from tests.studies.evidence import property_failures, property_verdicts
from tests.studies.evidence.failures import (
    BOUND_EXCEEDED,
    FAILURE_COLUMNS,
    NOT_ACCOUNTABLE,
    WITHIN_BOUND,
    WORST_CASE_FAILS,
    filled,
    vertex,
)
from tests.studies.evidence.inference import clopper_pearson, student_interval
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    coverage_gain_interval,
    largest_standardized_value,
    mean_difference_interval,
    paired,
    paired_displacement,
    paired_spread_ratio_interval,
    require_accounted,
    se_ratio_deficit_interval,
    se_ratio_interval,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import DRAW_NOT_SEPARABLE, Recording
from tests.studies.evidence.seeds import stream_seed


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
    return study, read_rows(study.artifact("property-replicates.parquet"))


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


def test_p13_the_projection_ratio_refuses_an_unmatched_replicate_with_no_stamp() -> None:
    """M3, G10: ``projection_sd_ratio`` pairs through ``paired``, so an unexplained gap stops it."""
    from tests.studies.stochastic_categorical_ltmle_properties import projection_sd_ratio

    rows = pd.concat(
        [
            _cell("randomizer_projection", "mix__integrated", lost=(4,), seed=1),
            _cell("randomizer_projection", "mix__recorded_randomizer", seed=2),
        ],
        ignore_index=True,
    )
    options = {"replicates": 50, "level": 0.99, "seed": 3}
    point, _ = projection_sd_ratio(rows, **options)
    kept = rows.loc[rows["replicate"] != 4]
    expected = float(
        np.std(kept.loc[kept["cell"] == "mix__integrated", "estimate"], ddof=1)
        / np.std(kept.loc[kept["cell"] == "mix__recorded_randomizer", "estimate"], ddof=1)
    )
    assert point == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="do not explain"):
        projection_sd_ratio(rows.assign(failed_replicates=0), **options)


@pytest.mark.parametrize(
    ("family", "label", "positive"),
    [
        ("clustered_calibration", "clusters_200", "known_ratio"),
        ("weighted_calibration", "selection_weighted", "correctly_specified"),
    ],
)
def test_p16_a_lost_sdr_base_replicate_moves_no_other_noise_row(
    family: str, label: str, positive: str
) -> None:
    """H3, G9: ``longitudinal_sdr`` indexes its noise by replicate."""
    from tests.studies.longitudinal_sdr_properties import _derived_arms

    complete = _cell(family, f"{label}__{positive}")
    lost = _cell(family, f"{label}__{positive}", lost=(6,))
    noise = {
        name: _derived_arms(frame, family, label, positive)[1].set_index("replicate")
        for name, frame in (("complete", complete), ("lost", lost))
    }
    assert 6 not in noise["lost"].index
    pd.testing.assert_frame_equal(
        noise["lost"].drop(columns="failed_replicates"),
        noise["complete"].drop(index=6).drop(columns="failed_replicates"),
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


# ---- the accounting (RM39b2 section 12: leg controls p1-p12 and p15) -------------------------

#: A property module that declares the largest bound a module may declare.
BOUNDED = SimpleNamespace(FAILURE_BOUND=property_failures.MAXIMUM_BOUND)


def _study(bootstrap: int = 200) -> StudyRecord:
    base = _calibration_record()
    return dataclasses.replace(
        base, margins=dataclasses.replace(base.margins, bootstrap_replicates=bootstrap)
    )


def _signs(size: int) -> np.ndarray:
    """``+1, -1, +1, ...``: a spread of one whose resampled spread barely moves."""
    return np.where(np.arange(size) % 2 == 0, 1.0, -1.0)


def _hits(count: int, size: int) -> np.ndarray:
    return (np.arange(size) < count).astype(int)


def _arm(
    family: str,
    cell: str,
    role: str,
    *,
    requested: int,
    lost: Sequence[int] = (),
    estimates: np.ndarray,
    std_errors: float | np.ndarray = 1.0,
    covered: np.ndarray | None = None,
    rejected: np.ndarray | None = None,
    n: int = 500,
) -> pd.DataFrame:
    """One cell's surviving rows: ``requested`` minus ``lost``, stamped with ``len(lost)``."""
    gone = set(lost)
    ids = [replicate for replicate in range(requested) if replicate not in gone]
    estimates = np.asarray(estimates, dtype=float)
    assert len(estimates) == len(ids)
    errors = np.broadcast_to(np.asarray(std_errors, dtype=float), estimates.shape)
    return pd.DataFrame(
        {
            "property": family,
            "cell": cell,
            "role": role,
            "replicate": ids,
            "n": n,
            "requested_replicates": requested,
            "failed_replicates": len(gone),
            "truth": 0.0,
            "estimate": estimates,
            "std_error": errors,
            "covered": (np.abs(estimates) <= 1.96 * errors).astype(int)
            if covered is None
            else covered,
            "rejected": (np.abs(estimates) > 1.96 * errors).astype(int)
            if rejected is None
            else rejected,
        }
    ).loc[:, list(REPLICATE_COLUMNS)]


def _gap_records(rows: pd.DataFrame) -> pd.DataFrame:
    """One record per replicate each cell lost: the records that explain the rows' gaps."""
    records = []
    for (family, cell), group in rows.groupby(["property", "cell"], sort=True):
        requested = int(group["requested_replicates"].iloc[0])
        for replicate in sorted(set(range(requested)) - set(group["replicate"])):
            records.append(_record_row(f"{family}/{cell}", replicate))
    return _records(*records)


def _shared(rows: pd.DataFrame, study: StudyRecord, **options: Any) -> pd.DataFrame:
    summary, rates = property_verdicts.apply_shared_verdicts(rows, study, **options)
    return property_verdicts.finish(summary, rates)


@dataclasses.dataclass
class Control:
    """A leg control: rows that lost replicates, their complete-case summary, the leg."""

    rows: pd.DataFrame
    summary: pd.DataFrame
    study: StudyRecord
    family: str
    cell: str
    leg: str
    rules: Sequence[property_failures.FamilyRule] = property_verdicts.FAMILY_RULES
    groups: Mapping[str, Sequence[Sequence[str]]] = dataclasses.field(default_factory=dict)


def _module(control: Control) -> SimpleNamespace:
    """A property module at the largest bound, with the control's declared groups."""
    return SimpleNamespace(
        FAILURE_BOUND=property_failures.MAXIMUM_BOUND, PROPERTY_GROUPS=control.groups
    )


def _account(
    control: Control, *, rules: Sequence[property_failures.FamilyRule] | None = None
) -> pd.DataFrame:
    records = _gap_records(control.rows)
    property_failures.reconcile(control.rows, records)
    return property_failures.property_accounting(
        control.rows,
        records,
        control.summary,
        record=control.study,
        module=_module(control),
        rules=control.rules if rules is None else rules,
    )


def _held(
    rules: Sequence[property_failures.FamilyRule], family: str, leg: str
) -> tuple[property_failures.FamilyRule, ...]:
    """The deliberate mutation: one leg's candidates replaced by its complete case."""

    def complete(clause: property_failures.Clause) -> property_failures.Clause:
        return dataclasses.replace(
            clause, candidates=lambda ctx, row: [("complete case", clause.read(ctx, row))]
        )

    return tuple(
        dataclasses.replace(
            rule,
            clauses=tuple(
                complete(clause) if clause.name == leg else clause for clause in rule.clauses
            ),
        )
        if family in rule.families
        else rule
        for rule in rules
    )


def _of(frame: pd.DataFrame, family: str, cell: str) -> pd.DataFrame:
    return frame.loc[(frame["property"] == family) & (frame["cell"] == cell)]


def _every(step: int, count: int) -> list[int]:
    """``count`` lost ids spread across the run: ``0, step, 2 * step, ...``."""
    return list(range(0, step * count, step))


def _context(control: Control) -> property_failures.Context:
    return property_failures.Context(
        control.study,
        control.rows,
        control.summary,
        property_failures.counts(_gap_records(control.rows)),
        control.groups,
    )


def _rule(family: str) -> property_failures.FamilyRule:
    return next(rule for rule in property_verdicts.FAMILY_RULES if family in rule.families)


def _row(control: Control) -> pd.Series:
    return _of(control.summary, control.family, control.cell).iloc[0]


# Each builder returns a control that lost replicates (``lose=True``) at most the bound allows,
# or the same cell complete (``lose=False``), whose candidates must equal its complete case.


def _p1_calibration_coverage(lose: bool = True) -> Control:
    """S2: coverage 3730 of 4000 sits in its band; 200 lost misses leave it."""
    study = _study()
    size = 4000 if lose else 4200
    rows = _arm(
        "interval_calibration",
        "z0__correctly_specified",
        "positive",
        requested=4200,
        lost=_every(21, 200) if lose else (),
        estimates=_signs(size),
        covered=_hits(3730, size),
    )
    return Control(
        rows,
        _shared(rows, study, rate_labels=()),
        study,
        "interval_calibration",
        "z0__correctly_specified",
        "coverage in calibration band",
    )


def _p2_bias_held_scale(lose: bool = True) -> Control:
    """S1: two outliers at +-10 make the balanced vertex split wide."""
    study = _study()
    estimates = _signs(380 if lose else 400)
    estimates[0], estimates[1] = 10.0, -10.0
    rows = _arm(
        "double_robustness",
        "outcome_correct",
        "positive",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=estimates,
        std_errors=float(np.std(estimates, ddof=1)),
    )
    return Control(
        rows,
        _shared(rows, study, rate_labels=()),
        study,
        "double_robustness",
        "outcome_correct",
        "bias equivalent",
    )


def _p3_bias_discriminated(lose: bool = True) -> Control:
    """S1: a control bias of 0.41 standard deviations clears the 0.25 margin; fills at the
    survivor minimum pull it back inside."""
    study = _study()
    estimates = _signs(380 if lose else 400) + 0.41
    rows = _arm(
        "double_robustness",
        "both_wrong",
        "control",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=estimates,
        std_errors=float(np.std(estimates, ddof=1)),
    )
    return Control(
        rows,
        _shared(rows, study, rate_labels=()),
        study,
        "double_robustness",
        "both_wrong",
        "bias discriminated",
    )


def _p4_calibration_se_ratio(lose: bool = True) -> Control:
    """S5: an SE ratio near 1.06; SEs at the survivor maximum push it through 1.07."""
    study = _study()
    size = 4160 if lose else 4200
    std_errors = np.where(np.arange(size) < int(0.2 * size), 1.3, 1.0)
    rows = _arm(
        "interval_calibration",
        "z0__correctly_specified",
        "positive",
        requested=4200,
        lost=_every(100, 40) if lose else (),
        estimates=_signs(size),
        std_errors=std_errors,
        covered=_hits(int(0.955 * size), size),
    )
    return Control(
        rows,
        _shared(rows, study, rate_labels=()),
        study,
        "interval_calibration",
        "z0__correctly_specified",
        "se ratio in calibration band",
    )


def _p5_resolve(lose: bool = True) -> Control:
    """S2: a root-n control of 362 covered of 380 resolves at ``j = 0`` and ``j = k``."""
    study = _study()
    size = 380 if lose else 400
    rows = _arm(
        "root_n_and_efficiency",
        "n_250",
        "control",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=_signs(size),
        covered=_hits(362, size),
        n=250,
    )
    return Control(
        rows,
        _shared(rows, study, rate_labels=()),
        study,
        "root_n_and_efficiency",
        "n_250",
        "coverage resolves",
    )


def _p6_rate(lose: bool = True) -> Control:
    """S6: a spread contracting at ``n ** -0.56`` sits inside the root-n band; the lost
    rungs' extremes push it out below."""
    from scipy.stats import norm

    study = _study()
    size = 380 if lose else 400
    quantiles = norm.ppf((np.arange(size) + 0.5) / size)
    quantiles = quantiles[np.random.default_rng(0).permutation(size)]
    rungs = [
        _arm(
            "root_n_and_efficiency",
            f"n_{n}",
            "positive",
            requested=400,
            lost=_every(20, 20) if lose else (),
            estimates=quantiles * n**-0.56,
            std_errors=n**-0.56,
            n=n,
        )
        for n in (250, 1000, 4000)
    ]
    rows = pd.concat(rungs, ignore_index=True)
    return Control(
        rows, _shared(rows, study), study, "root_n_rate", "empirical_sd", "slope in root-n band"
    )


def _p7_coverage_gain(
    lose: bool = True, *, across_keys: bool = False, control_loses: bool = True
) -> Control:
    """S7: a gain of 0.17 over 4000 paired draws; 200 lost differences at -1 cost 0.057.

    ``across_keys`` loses 100 different replicates in each arm, which the gain reads as 200
    lost differences.  ``control_loses=False`` leaves the control arm complete.
    """
    study = _study()
    if not lose:
        lost = {"cross_fit": [], "in_sample_control": []}
    elif across_keys:
        lost = {
            "cross_fit": _every(21, 100),
            "in_sample_control": [replicate + 10 for replicate in _every(21, 100)],
        }
    else:
        lost = {
            "cross_fit": _every(21, 200),
            "in_sample_control": _every(21, 200) if control_loses else [],
        }
    gone = set(lost["cross_fit"]) | set(lost["in_sample_control"])
    # The misses sit on ids both arms kept, so the paired complete case is a gain of 0.17.
    shared = [replicate for replicate in range(4200) if replicate not in gone]
    arms = []
    for cell, role, error, share in (
        ("cross_fit", "positive", 1.0, 0.05),
        ("in_sample_control", "control", 0.5, 0.22),
    ):
        ids = [replicate for replicate in range(4200) if replicate not in set(lost[cell])]
        missed = set(shared[: round(share * len(shared))])
        arms.append(
            _arm(
                "crossfit_overfitting",
                cell,
                role,
                requested=4200,
                lost=lost[cell],
                estimates=_signs(len(ids)),
                std_errors=error,
                covered=np.array([int(replicate not in missed) for replicate in ids]),
            )
        )
    rows = pd.concat(arms, ignore_index=True)
    summary, rates = property_verdicts.apply_shared_verdicts(
        rows,
        study,
        rate_labels=(),
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
    )
    property_verdicts.crossfit_overfitting_verdicts(summary, rows, study, positive_cell="cross_fit")
    return Control(
        rows,
        property_verdicts.finish(summary, rates),
        study,
        "crossfit_overfitting",
        "cross_fit",
        "coverage gain",
    )


# p9, p10 and p33: S9 and S10 run as synthetic families whose summary row a study would
# compute with the same helper on its survivors.


def _synthetic(
    family: str,
    rows: pd.DataFrame,
    subject: dict[str, Any],
    clause: property_failures.Clause,
) -> tuple[pd.DataFrame, tuple[property_failures.FamilyRule, ...]]:
    summary = pd.DataFrame(
        [
            {
                "property": family,
                "cell": cell,
                "role": "positive" if cell == "subject" else property_verdicts.DIAGNOSTIC_ROLE,
                "replicates": int((rows["cell"] == cell).sum()),
                "failed_replicates": int(
                    rows.loc[rows["cell"] == cell, "failed_replicates"].iloc[0]
                ),
                **(subject if cell == "subject" else {}),
            }
            for cell in ("subject", *sorted(set(rows["cell"]) - {"subject"}))
        ]
    )
    rule = property_failures.FamilyRule(
        families=frozenset({family}), shared="synthetic", clauses=(clause,)
    )
    context = property_failures.Context(_study(), rows, summary)
    passed = clause.read(context, summary.iloc[0]).passed
    summary["passed"] = [passed] + [True] * (len(summary) - 1)
    summary["property_passed"] = summary["passed"]
    return summary, (rule,)


def _p9_spread_ratio(lose: bool = True) -> Control:
    """S9: a spread ratio near one under a 1.06 cap; the numerator's widest split and the
    denominator's mean fills push it over."""
    study = _study()
    size = 380 if lose else 400
    lost = _every(20, 20) if lose else ()
    numerator = _signs(size)
    numerator[0], numerator[1] = 2.0, -2.0
    rows = pd.concat(
        [
            _arm(
                "synthetic_spread",
                "subject",
                "positive",
                requested=400,
                lost=lost,
                estimates=numerator,
            ),
            _arm(
                "synthetic_spread",
                "reference",
                "diagnostic",
                requested=400,
                lost=lost,
                estimates=_signs(size),
            ),
        ],
        ignore_index=True,
    )
    seed = 11
    ratio = paired_spread_ratio_interval(
        rows.loc[rows["cell"] == "subject"],
        rows.loc[rows["cell"] == "reference"],
        replicates=study.margins.bootstrap_replicates,
        confidence_level=study.margins.confidence_level,
        seed=seed,
    )
    clause = property_failures.spread_ratio_clause(
        "spread ratio under its cap",
        cells=("subject", "reference"),
        side="upper",
        judge=property_failures.ceiling_judge(lambda ctx: 1.06),
        seed=lambda ctx, row: seed,
        applies=property_failures.roles("positive"),
    )
    summary, rules = _synthetic(
        "synthetic_spread",
        rows,
        {"spread_ratio_ci_lower": ratio.interval.low, "spread_ratio_ci_upper": ratio.interval.high},
        clause,
    )
    return Control(rows, summary, study, "synthetic_spread", "subject", clause.name, rules)


def _p10_extra_mean(lose: bool = True) -> Control:
    """S10: a share of 0.49 under a 0.5 cap; 20 lost rows at the support bound 1 exceed it."""
    study = _study()
    size = 380 if lose else 400
    rows = _arm(
        "synthetic_share",
        "subject",
        "positive",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=_signs(size),
    )
    rows["share"] = (np.arange(size) < round(0.49 * size)).astype(float)
    clause = property_failures.extra_mean_clause(
        "share under its cap",
        value=lambda group: group["share"].to_numpy(dtype=float),
        summary_column="mean_share",
        side="cap",
        limit=0.5,
        applies=property_failures.roles("positive"),
        support=(0.0, 1.0),
    )
    summary, rules = _synthetic(
        "synthetic_share", rows, {"mean_share": float(rows["share"].mean())}, clause
    )
    return Control(rows, summary, study, "synthetic_share", "subject", clause.name, rules)


# ---- rev 3 statistic types (RM39b2-plan-rev3 section 9) ---------------------------------------


def _summary_of(
    family: str,
    rows: pd.DataFrame,
    rules: Sequence[property_failures.FamilyRule],
    *,
    columns: Mapping[str, Mapping[str, Any]],
    study: StudyRecord,
    groups: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> pd.DataFrame:
    """A synthetic family's summary: each cell's published columns, its rule's verdicts.

    The verdicts are the rule's complete case, which is what a study that computes the same
    statistic on its survivors publishes.
    """
    records = []
    for cell, group in rows.loc[rows["property"] == family].groupby("cell", sort=False):
        records.append(
            {
                "property": family,
                "cell": cell,
                "role": str(group["role"].iloc[0]),
                "replicates": len(group),
                "failed_replicates": int(group["failed_replicates"].iloc[0]),
                **columns.get(str(cell), {}),
            }
        )
    summary = pd.DataFrame(records)
    context = property_failures.Context(study, rows, summary, {}, groups or {})
    rule = property_failures.rule_for(family, rules=rules)
    assert rule is not None
    verdicts = property_failures.rule_verdicts(rule, summary, context)
    return summary.merge(verdicts, on=["property", "cell"], how="left", validate="one_to_one")


def _one_rule(family: str, *clauses: property_failures.Clause, group: str = "row") -> tuple:
    return (
        property_failures.FamilyRule(
            families=frozenset({family}), shared="synthetic", clauses=clauses, group=group
        ),
    )


def _bias_columns(rows: pd.DataFrame, study: StudyRecord) -> dict[str, float]:
    errors = (rows["estimate"] - rows["truth"]).to_numpy(dtype=float)
    interval = student_interval(errors, confidence_level=study.margins.confidence_level)
    return {
        "bias_ci_lower": interval.low,
        "bias_ci_upper": interval.high,
        "empirical_se": float(np.std(errors, ddof=1)),
    }


def _p18_fixed_floor(lose: bool = True) -> Control:
    """S1f: a both-wrong control's bias clears a fixed floor of 0.28 at 0.287; 20 fills at the
    survivor minimum leave it at 0.241."""
    study = _study()
    family = "synthetic_corrected_mar_inference"
    rows = _arm(
        family,
        "both_wrong",
        "control",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=_signs(380 if lose else 400) + 0.42,
    )
    clause = property_failures.bias_clause(
        "bias clears the both-wrong floor",
        kind="beyond",
        margin=lambda ctx, row: 0.28,
        strict=False,
        statistic="S1f",
        applies=property_failures.roles("control"),
    )
    rules = _one_rule(family, clause)
    summary = _summary_of(
        family, rows, rules, columns={"both_wrong": _bias_columns(rows, study)}, study=study
    )
    return Control(rows, summary, study, family, "both_wrong", clause.name, rules)


def _p32_held_margin(lose: bool = True) -> Control:
    """S1: an SDR control clears 0.25 times its held ``empirical_se``; a split pulls it back."""
    study = _study()
    family = "synthetic_sequential_double_robustness"
    rows = _arm(
        family,
        "p_b_control",
        "control",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=_signs(380 if lose else 400) + 0.42,
    )
    clause = property_failures.bias_clause(
        "bias beyond the control floor",
        kind="beyond",
        margin=lambda ctx, row: 0.25 * float(row["empirical_se"]),
        applies=property_failures.roles("control"),
    )
    rules = _one_rule(family, clause)
    summary = _summary_of(
        family, rows, rules, columns={"p_b_control": _bias_columns(rows, study)}, study=study
    )
    return Control(rows, summary, study, family, "p_b_control", clause.name, rules)


def _p33_floor_mean(lose: bool = True) -> Control:
    """S10 on an unbounded value: a mean of 0.505 above a 0.5 floor; 20 rows at the survivor
    minimum 0.305 take it to 0.495."""
    study = _study()
    family = "synthetic_floor"
    size = 380 if lose else 400
    rows = _arm(
        family,
        "subject",
        "positive",
        requested=400,
        lost=_every(20, 20) if lose else (),
        estimates=_signs(size),
    )
    rows["overlap"] = 0.505 + 0.2 * _signs(size)
    clause = property_failures.extra_mean_clause(
        "mean overlap above its floor",
        value=lambda group: group["overlap"].to_numpy(dtype=float),
        summary_column="mean_overlap",
        side="floor",
        limit=0.5,
        applies=property_failures.roles("positive"),
    )
    rules = _one_rule(family, clause)
    summary = _summary_of(
        family,
        rows,
        rules,
        columns={"subject": {"mean_overlap": float(rows["overlap"].mean())}},
        study=study,
    )
    return Control(rows, summary, study, family, "subject", clause.name, rules)


DISPLACEMENT = "synthetic_necessity"


def _displacement(
    *,
    shift: float,
    scale: float,
    threshold: float,
    left_lost: Sequence[int],
    right_lost: Sequence[int],
) -> Control:
    study = _study()
    left = _arm(
        DISPLACEMENT,
        "a__targeted",
        "positive",
        requested=400,
        lost=left_lost,
        estimates=_signs(400 - len(left_lost)),
    )
    right = _arm(
        DISPLACEMENT,
        "a__untargeted",
        "control",
        requested=400,
        lost=right_lost,
        estimates=scale * _signs(400 - len(right_lost)) + shift,
    )
    rows = pd.concat([left, right], ignore_index=True)
    clause = property_failures.displacement_clause(
        "displacement at the threshold",
        column="displacement",
        labels=("a",),
        arms=("targeted", "untargeted"),
        threshold=threshold,
    )
    rules = _one_rule(DISPLACEMENT, clause, group="family")
    value = paired_displacement(rows, DISPLACEMENT, "a__targeted", "a__untargeted")
    summary = _summary_of(
        DISPLACEMENT,
        rows,
        rules,
        columns={cell: {"displacement": value} for cell in ("a__targeted", "a__untargeted")},
        study=study,
    )
    return Control(rows, summary, study, DISPLACEMENT, "a__untargeted", clause.name, rules)


def _p20_same_key(lose: bool = True) -> Control:
    """S12: a displacement of 0.58 over a 0.5 threshold; both arms lose the same 20 keys, and
    their means at the nearer ends close it to 0.48."""
    lost = _every(20, 20) if lose else []
    return _displacement(shift=0.58, scale=1.0, threshold=0.5, left_lost=lost, right_lost=lost)


def _p20_cross_key(lose: bool = True) -> Control:
    """S12: as the same-key case, with the arms losing different keys."""
    left = _every(20, 20) if lose else []
    right = [replicate + 1 for replicate in left]
    return _displacement(shift=0.58, scale=1.0, threshold=0.5, left_lost=left, right_lost=right)


def _p20_straddle(lose: bool = True) -> Control:
    """S12: a wide control arm; the attainable difference of means holds zero, so ``D = 0``."""
    lost = _every(20, 20) if lose else []
    return _displacement(shift=0.28, scale=5.0, threshold=0.25, left_lost=lost, right_lost=lost)


SELECTOR = "synthetic_selector_necessity"


def _rmse_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> float:
    left, right = paired(numerator, denominator)
    errors = [(arm["estimate"] - arm["truth"]).to_numpy(dtype=float) for arm in (left, right)]
    return float(np.sqrt(np.mean(errors[0] ** 2) / np.mean(errors[1] ** 2)))


def _selector(*, denominator: np.ndarray | None = None, lose: bool = True) -> Control:
    study = _study()
    lost = _every(20, 20) if lose else []
    numerator = 0.4 * _signs(400 - len(lost))
    numerator[0] = 2.0
    rows = pd.concat(
        [
            _arm(
                SELECTOR,
                "collaborative",
                "positive",
                requested=400,
                lost=lost,
                estimates=numerator,
            ),
            _arm(
                SELECTOR,
                "empty_control",
                "control",
                requested=400,
                lost=lost,
                estimates=_signs(400 - len(lost)) if denominator is None else denominator,
            ),
        ],
        ignore_index=True,
    )
    clause = property_failures.rmse_ratio_clause(
        "rmse ratio under its threshold",
        paths=[("collaborative", "empty_control")],
        statistic=_rmse_ratio,
        threshold=0.5,
    )
    rules = _one_rule(SELECTOR, clause, group="family")
    ratio = _rmse_ratio(
        rows.loc[rows["cell"] == "collaborative"], rows.loc[rows["cell"] == "empty_control"]
    )
    summary = _summary_of(
        SELECTOR,
        rows,
        rules,
        columns={"collaborative": {"rmse_ratio": ratio}, "empty_control": {"rmse_ratio": np.nan}},
        study=study,
    )
    return Control(rows, summary, study, SELECTOR, "collaborative", clause.name, rules)


def _p21_numerator(lose: bool = True) -> Control:
    """S13: an RMSE ratio of 0.41 under 0.5; the numerator's lost errors at its farther end
    (2.0) raise it past the threshold."""
    return _selector(lose=lose)


SINGLE = "synthetic_single_robustness"

#: p22's small cell: 37 errors near 0.5 and one at -8, searched so that the least bound over
#: the box sits at an interior equal fill, away from every split and from the mean fill.
_SMALL_ERRORS = np.concatenate([np.random.default_rng(1).normal(0.5, 0.5, 37), [-8.0]])


def _p22_bias_bound(lose: bool = True) -> Control:
    """S14: ``B_large = 0.67`` sits below ``B_small``; the small cell's least bound, 0.610 at
    an interior equal fill, falls under it."""
    study = _study()
    small = _SMALL_ERRORS if lose else np.concatenate([_SMALL_ERRORS, [0.5, 0.5]])
    rows = pd.concat(
        [
            _arm(
                SINGLE,
                "small",
                "positive",
                requested=40,
                lost=(5, 25) if lose else (),
                estimates=small,
                n=1000,
            ),
            _arm(
                SINGLE, "large", "positive", requested=40, estimates=0.51 + 0.5 * _signs(40), n=4000
            ),
        ],
        ignore_index=True,
    )
    clause = property_failures.bias_bound_clause(
        "large bound below the small",
        small="small",
        large="large",
        kind="below small",
        column="absolute_bias_bound",
    )
    rules = _one_rule(SINGLE, clause, group="family")
    summary = _summary_of(
        SINGLE,
        rows,
        rules,
        columns={
            cell: {
                "absolute_bias_bound": property_failures.error_bias_bound(
                    rows.loc[rows["cell"] == cell, "estimate"].to_numpy(dtype=float)
                )
            }
            for cell in ("small", "large")
        },
        study=study,
    )
    return Control(rows, summary, study, SINGLE, "small", clause.name, rules)


def _p24_contraction(lose: bool = True) -> Control:
    """S6 ``bias``: biases of 0.08, 0.04 and 0.02 contract.  The smallest rung holds one error
    at -2, so a fill of its 20 lost rows at -1.52 brings its mean to zero, and the slope no
    longer falls."""
    study = _study()
    family = property_verdicts.CONTRACTION_FAMILY
    rungs = []
    for n, bias in ((500, 0.08), (2000, 0.04), (8000, 0.02)):
        lost = _every(20, 20) if lose and n == 500 else []
        size = 400 - len(lost)
        errors = 0.05 * _signs(size)
        if n == 500:
            errors[-1] = -2.0
        errors += bias - float(np.mean(errors))
        rungs.append(
            _arm(
                family,
                f"outcome_correct_n{n}",
                "positive",
                requested=400,
                lost=lost,
                estimates=errors,
                n=n,
            )
        )
    rows = pd.concat(rungs, ignore_index=True)
    columns = [
        "property",
        "cell",
        "role",
        "replicates",
        "failed_replicates",
        "slope",
        "slope_ci_lower",
        "slope_ci_upper",
        "passed",
        "property_passed",
    ]
    summary = pd.DataFrame(
        property_verdicts.contraction_rates(rows, study, columns, scenarios=("outcome_correct",))
    )
    summary["property_passed"] = summary["passed"]
    return Control(rows, summary, study, family, "rate_outcome_correct", "bias contracts")


def _module_rules(module: Any) -> tuple[property_failures.FamilyRule, ...]:
    """A module's declared rules ahead of the shared ones, as :func:`rule_for` resolves them."""
    return (*module.PROPERTY_RULES, *property_verdicts.FAMILY_RULES)


def _p25_pair(lose: bool = True) -> Control:
    """``group = pair`` with ``mar_arm_indexed_cvtmle``'s declared label pairs: only the pair
    that lost reads red."""
    from tests.studies import mar_arm_indexed_cvtmle_properties as module

    study = _study()
    (pairs,) = module.PROPERTY_GROUPS.values()
    frames = []
    for index, (positive, control) in enumerate(pairs):
        arms = _p7_coverage_gain(lose=lose and index == 0, control_loses=False).rows.copy()
        arms["cell"] = arms["cell"].map({"cross_fit": positive, "in_sample_control": control})
        frames.append(arms)
    rows = pd.concat(frames, ignore_index=True)
    summaries = []
    for positive, control in pairs:
        arms = rows.loc[rows["cell"].isin([positive, control])]
        summary, rates = property_verdicts.apply_shared_verdicts(
            arms,
            study,
            rate_labels=(),
            extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        )
        property_verdicts.crossfit_overfitting_verdicts(
            summary, arms, study, positive_cell=positive, control_cell=control
        )
        summaries.append(property_verdicts.finish(summary, rates))
    return Control(
        rows,
        pd.concat(summaries, ignore_index=True),
        study,
        "crossfit_overfitting",
        pairs[0][0],
        "coverage gain",
        property_verdicts.FAMILY_RULES,
        module.PROPERTY_GROUPS,
    )


def _p26_cap(lose: bool = True) -> Control:
    """S10 through ``full_refit_bootstrap``'s own rule: a failed-bootstrap share of 0.0095 under
    the 0.01 cap; 20 lost rows at share 1 break it, which the shared calibration rule alone
    never reads (G1)."""
    from tests.studies import full_refit_bootstrap_properties as module

    study = _study()
    size = 4180 if lose else 4200
    rows = _arm(
        "interval_calibration",
        "z0__correctly_specified",
        "positive",
        requested=4200,
        lost=_every(200, 20) if lose else (),
        estimates=_signs(size),
        covered=_hits(round(0.95 * size), size),
    )
    rows["bootstrap_requested"] = 100
    rows["bootstrap_failed"] = (np.arange(size) < round(0.95 * size)).astype(int)
    summary = _shared(rows, study, rate_labels=())
    summary["bootstrap_failed_fraction"] = float(module.failed_bootstrap_share(rows).mean())
    return Control(
        rows,
        summary,
        study,
        "interval_calibration",
        "z0__correctly_specified",
        "failed bootstrap share under the cap",
        _module_rules(module),
    )


PROJECTION = "randomizer_projection"


def _p27_projection(lose: bool = True) -> Control:
    """S9 hook through ``stochastic_categorical_ltmle``'s own rule: the integrated arm's spread
    sits under the recorded one's; the numerator's widest split pushes the verdict's one-sided
    bound over one.  Twelve lost rows keep the bias legs green."""
    from tests.studies import stochastic_categorical_ltmle_properties as module

    study = _study()
    size = 388 if lose else 400
    lost = _every(33, 12) if lose else ()
    integrated = 0.9 * _signs(size)
    integrated[0], integrated[1] = 2.5, -2.5
    rows = pd.concat(
        [
            _arm(
                PROJECTION,
                "mix__integrated",
                "positive",
                requested=400,
                lost=lost,
                estimates=integrated,
            ),
            _arm(
                PROJECTION,
                "mix__recorded_randomizer",
                "control",
                requested=400,
                lost=lost,
                estimates=_signs(size),
            ),
        ],
        ignore_index=True,
    )
    rules = _module_rules(module)
    _, upper = module.projection_sd_ratio(
        rows,
        replicates=study.margins.bootstrap_replicates,
        level=module.PROJECTION_LEVEL,
        seed=stream_seed(study, PROJECTION, "sd_ratio"),
    )
    columns = {}
    for cell, group in rows.groupby("cell"):
        errors = (group["estimate"] - group["truth"]).to_numpy(dtype=float)
        columns[str(cell)] = {
            **_bias_columns(group, study),
            "bias_margin": study.margins.standardized_bias * float(np.std(errors, ddof=1)),
            "projection_sd_ratio_ci_upper": upper,
        }
    summary = _summary_of(PROJECTION, rows, rules, columns=columns, study=study)
    return Control(
        rows,
        summary,
        study,
        PROJECTION,
        "mix__integrated",
        "projection spread ratio under one",
        rules,
    )


GENERATED = property_verdicts.GENERATED_DESIGN_FAMILY


def _p31_label(lose: bool = True) -> Control:
    """``group = label`` with ``ctmle_oat_per_arm``'s declared label groups: a loss in the first
    label's estimated arm breaks its coverage band; the other label keeps its joint verdict."""
    from tests.studies import ctmle_oat_per_arm_properties as module

    study = _study()
    groups = module.PROPERTY_GROUPS
    cells = [cell for members in groups[GENERATED] for cell in members]
    losing = groups[GENERATED][0][1]
    arms = []
    for cell in cells:
        loses = lose and cell == losing
        size = 4000 if loses else 4200
        arms.append(
            _arm(
                GENERATED,
                cell,
                "positive",
                requested=4200,
                lost=_every(21, 200) if loses else (),
                estimates=_signs(size),
                covered=_hits(3730 if loses else round(0.95 * size), size),
            )
        )
    rows = pd.concat(arms, ignore_index=True)
    rules = _module_rules(module)
    level = study.margins.confidence_level
    columns = {}
    for cell, group in rows.groupby("cell"):
        coverage = clopper_pearson(int(group["covered"].sum()), len(group), confidence_level=level)
        ratio = se_ratio_interval(
            group,
            replicates=study.margins.bootstrap_replicates,
            confidence_level=level,
            seed=stream_seed(study, GENERATED, *str(cell).split("__")),
        )
        columns[str(cell)] = {
            "coverage": float(group["covered"].mean()),
            "coverage_ci_lower": coverage.low,
            "coverage_ci_upper": coverage.high,
            "se_ratio_ci_lower": ratio.low,
            "se_ratio_ci_upper": ratio.high,
        }
    summary = _summary_of(GENERATED, rows, rules, columns=columns, study=study, groups=groups)
    return Control(
        rows, summary, study, GENERATED, losing, "coverage in calibration band", rules, groups
    )


def _p6_interior(lose: bool = True) -> Control:
    """S6 (G3): a root-n ladder whose smallest rung lost 20 of 400; the slope interval leaves
    its band under an interior vertex split alone.  The exponent is tuned to put the band's
    edge between that split and every rev 2 candidate."""
    study = _study()
    rungs = []
    for n in (250, 1000, 4000):
        lost = _every(20, 20) if lose and n == 250 else []
        size = 400 - len(lost)
        draws = np.random.default_rng(730 + n).standard_t(5, size)
        rungs.append(
            _arm(
                "root_n_and_efficiency",
                f"n_{n}",
                "positive",
                requested=400,
                lost=lost,
                estimates=draws * n**-0.49265,
                std_errors=n**-0.49265,
                n=n,
            )
        )
    rows = pd.concat(rungs, ignore_index=True)
    return Control(
        rows, _shared(rows, study), study, "root_n_rate", "empirical_sd", "slope in root-n band"
    )


LEG_CONTROLS_REV3 = {
    "p18": _p18_fixed_floor,
    "p20 same key": _p20_same_key,
    "p20 cross key": _p20_cross_key,
    "p20 straddle": _p20_straddle,
    "p21": _p21_numerator,
    "p22": _p22_bias_bound,
    "p24": _p24_contraction,
    "p25": _p25_pair,
    "p26": _p26_cap,
    "p27": _p27_projection,
    "p31": _p31_label,
    "p32": _p32_held_margin,
    "p33": _p33_floor_mean,
    "p6 interior": _p6_interior,
}


LEG_CONTROLS = {
    "p1": _p1_calibration_coverage,
    "p2": _p2_bias_held_scale,
    "p3": _p3_bias_discriminated,
    "p4": _p4_calibration_se_ratio,
    "p5": _p5_resolve,
    "p6": _p6_rate,
    "p7": _p7_coverage_gain,
    "p7 across keys": functools.partial(_p7_coverage_gain, across_keys=True),
    "p9": _p9_spread_ratio,
    "p10": _p10_extra_mean,
    **LEG_CONTROLS_REV3,
}


@functools.cache
def _control(name: str) -> Control:
    return LEG_CONTROLS[name]()


@functools.cache
def _accounted(name: str) -> pd.DataFrame:
    return _account(_control(name))


class TestLegControls:
    """F7: each leg control fails its worst case inside the bound, and only through its leg."""

    @pytest.mark.parametrize("name", list(LEG_CONTROLS))
    def test_the_complete_case_is_green(self, name: str) -> None:
        row = _row(_control(name))
        assert bool(row["passed"]) and bool(row["property_passed"])
        assert int(row["failed_replicates"]) > 0

    @pytest.mark.parametrize("name", list(LEG_CONTROLS))
    def test_the_worst_case_fails_through_the_leg_inside_the_bound(self, name: str) -> None:
        control = _control(name)
        cell = _of(_accounted(name), control.family, control.cell)
        assert not cell.empty
        assert (cell["failed"] <= cell["allowed"]).all()
        assert set(cell["reason"]) == {WORST_CASE_FAILS}
        assert set(cell.loc[~cell["leg_passed"].astype(bool), "leg"]) == {control.leg}
        leg = cell.loc[cell["leg"] == control.leg].iloc[0]
        assert leg["worst_case"] < 0.0 <= leg["complete_case"]

    @pytest.mark.parametrize("name", list(LEG_CONTROLS))
    def test_holding_the_leg_at_its_complete_case_turns_the_cell_green(self, name: str) -> None:
        control = _control(name)
        mutated = _account(control, rules=_held(control.rules, control.family, control.leg))
        cell = _of(mutated, control.family, control.cell)
        assert cell["gate_passed"].astype(bool).all()
        assert set(cell["reason"]) == {WITHIN_BOUND}

    @pytest.mark.parametrize("name", list(LEG_CONTROLS))
    def test_the_corrected_summary_sets_the_leg_s_target_false(self, name: str) -> None:
        control = _control(name)
        rule = next(rule for rule in control.rules if control.family in rule.families)
        target = rule.clause(control.leg).target
        corrected = _row(
            dataclasses.replace(
                control,
                summary=property_failures.correct_properties(
                    control.summary, _accounted(name), module=_module(control), rules=control.rules
                ),
            )
        )
        assert not bool(corrected["property_passed"])
        assert bool(corrected["passed"]) is (target != "passed")

    @pytest.mark.parametrize("name", list(LEG_CONTROLS))
    def test_with_nothing_lost_every_candidate_is_the_complete_case(self, name: str) -> None:
        """The candidates rebuild the verdict's own statistic: at ``k = 0`` they equal it."""
        control = LEG_CONTROLS[name](lose=False)
        context = _context(control)
        rule = next(rule for rule in control.rules if control.family in rule.families)
        row = _row(control)
        assert context.k(row) == 0
        for clause in rule.clauses:
            if not clause.applies(context, row):
                continue
            complete = clause.read(context, row)
            for label, candidate in clause.candidates(context, row):
                assert candidate.passed == complete.passed, (clause.name, label)
                assert candidate.slack == pytest.approx(complete.slack, abs=1e-9), (
                    clause.name,
                    label,
                )


def test_p2_the_balanced_split_passes_only_under_a_rescaled_margin() -> None:
    """S1 holds the margin at the complete-case scale; a fill may not widen it."""
    control = _control("p2")
    row = _row(control)
    errors = control.rows["estimate"].to_numpy(dtype=float)
    balanced = vertex(errors, 20, 10)
    interval = student_interval(balanced, confidence_level=control.study.margins.confidence_level)
    rescaled = control.study.margins.standardized_bias * float(np.std(balanced, ddof=1))
    assert interval.within(-rescaled, rescaled)
    assert not interval.within(-float(row["bias_margin"]), float(row["bias_margin"]))
    judged = dict(
        _rule("double_robustness").clause("bias equivalent").candidates(_context(control), row)
    )
    assert not judged["j=10 at min"].passed


def test_p3_a_candidate_discriminated_on_the_other_side_fails() -> None:
    """m14: the side is the complete-case bias's.  A clause witness at ``k = 200``, which no
    bound allows, because a 5% fill cannot move a discriminated bias across zero."""
    estimates = _signs(200) + 0.5
    estimates[0] = -40.0
    rows = _arm(
        "double_robustness",
        "both_wrong",
        "control",
        requested=400,
        lost=range(200, 400),
        estimates=estimates,
        std_errors=float(np.std(estimates, ddof=1)),
    )
    study = _study()
    summary = _shared(rows, study, rate_labels=())
    row = summary.iloc[0]
    assert float(row["bias"]) > 0.0
    clause = _rule("double_robustness").clause("bias discriminated")
    context = property_failures.Context(
        study, rows, summary, {("double_robustness", "both_wrong"): 200}
    )
    judged = dict(clause.candidates(context, row))
    lowest = student_interval(
        vertex(rows["estimate"].to_numpy(dtype=float), 200, 200),
        confidence_level=study.margins.confidence_level,
    )
    assert lowest.high < -float(row["bias_margin"])
    assert not judged["j=200 at min"].passed


def test_p5_the_resolve_rule_fails_between_its_holding_ends() -> None:
    """S2 enumerates every split: the ends resolve and an intermediate one straddles."""
    control = _control("p5")
    judged = dict(
        _rule("root_n_and_efficiency")
        .clause("coverage resolves")
        .candidates(_context(control), _row(control))
    )
    assert judged["j=0 covered"].passed and judged["j=20 covered"].passed
    assert not all(value.passed for value in judged.values())


class TestNoRule:
    """A family with no declared rule reads ``failures not accountable`` (rev 3 section 7.3)."""

    def test_a_cell_that_passes_its_complete_case_is_red(self) -> None:
        family, cell = "undeclared_family", "subject"
        rows = _arm(family, cell, "positive", requested=400, lost=(3,), estimates=_signs(399))
        summary = pd.DataFrame(
            [
                {
                    "property": family,
                    "cell": name,
                    "role": "positive",
                    "replicates": replicates,
                    "failed_replicates": failed,
                    "passed": True,
                    "property_passed": True,
                }
                for name, replicates, failed in ((cell, 399, 1), ("untouched", 400, 0))
            ]
        )
        records = _gap_records(rows)
        accounting = property_failures.property_accounting(
            rows, records, summary, record=_study(), module=BOUNDED
        )
        assert set(accounting["reason"]) == {NOT_ACCOUNTABLE}
        assert list(accounting["leg"]) == [
            property_failures.BOUND_LEG,
            property_failures.NOT_ACCOUNTABLE_LEG,
            property_failures.WITNESS_LEG,
        ]
        corrected = property_failures.correct_properties(summary, accounting)
        assert list(corrected["passed"]) == [False, True]
        assert not corrected["property_passed"].any()


class TestBound:
    """p12: ``floor(b * R)`` failures pass the bound, one more exceeds it, per cell."""

    @staticmethod
    def _power(lost: dict[str, int]) -> Control:
        study = _study()
        cells = []
        for cell, requested in (("large", 400), ("small", 100)):
            size = requested - lost[cell]
            cells.append(
                _arm(
                    "power",
                    cell,
                    "positive",
                    requested=requested,
                    lost=_every(3, lost[cell]),
                    estimates=_signs(size),
                    rejected=np.ones(size, dtype=int),
                )
            )
        rows = pd.concat(cells, ignore_index=True)
        return Control(rows, _shared(rows, study, rate_labels=()), study, "power", "large", "")

    def test_the_allowance_passes_and_one_more_exceeds_it(self) -> None:
        within = _account(self._power({"large": 20, "small": 5}))
        assert set(within["reason"]) == {WITHIN_BOUND}
        assert within.groupby("cell")["allowed"].first().to_dict() == {"large": 20, "small": 5}
        exceeded = _account(self._power({"large": 21, "small": 6}))
        assert set(exceeded["reason"]) == {BOUND_EXCEEDED}
        assert not exceeded["gate_passed"].any()

    def test_a_zero_bound_allows_nothing(self) -> None:
        control = self._power({"large": 1, "small": 0})
        records = _gap_records(control.rows)
        accounting = property_failures.property_accounting(
            control.rows, records, control.summary, record=control.study, module=None
        )
        assert set(accounting["reason"]) == {BOUND_EXCEEDED}
        assert set(accounting["cell"]) == {"large"}

    def test_a_bound_outside_the_range_is_refused(self) -> None:
        control = self._power({"large": 1, "small": 0})
        with pytest.raises(ValueError, match="outside"):
            property_failures.property_accounting(
                control.rows,
                _gap_records(control.rows),
                control.summary,
                record=control.study,
                module=SimpleNamespace(FAILURE_BOUND=0.06),
            )

    def test_without_a_record_there_is_no_accounting_and_no_correction(self) -> None:
        control = self._power({"large": 0, "small": 0})
        accounting = property_failures.property_accounting(
            control.rows, None, control.summary, record=control.study, module=BOUNDED
        )
        assert accounting.empty
        assert list(accounting.columns) == list(property_failures.PROPERTY_ACCOUNTING_COLUMNS)
        assert property_failures.correct_properties(control.summary, accounting) is control.summary


def test_a_group_joint_family_accounts_its_untouched_cells() -> None:
    """The coverage gain is the pair's: a loss in one arm turns both arms' joint verdict red."""
    control = _p7_coverage_gain(control_loses=False)
    accounting = _account(control)
    assert set(accounting["cell"]) == {"cross_fit", "in_sample_control"}
    untouched = _of(accounting, "crossfit_overfitting", "in_sample_control")
    assert set(untouched["failed"]) == {0}
    assert set(untouched["reason"]) == {WORST_CASE_FAILS}
    corrected = property_failures.correct_properties(control.summary, accounting)
    family = corrected.loc[corrected["property"] == "crossfit_overfitting"]
    assert family["passed"].all()
    assert not family["property_passed"].any()


class TestDifferential:
    """p15 and F10: the differential comparison catches a dropped conjunct or a moved threshold."""

    @staticmethod
    @functools.cache
    def _ladder() -> tuple[StudyRecord, pd.DataFrame]:
        control = _p6_rate(lose=False)
        return control.study, control.rows

    def _disagreements(
        self, rules: Sequence[property_failures.FamilyRule], kind: str
    ) -> pd.DataFrame:
        study, rows = self._ladder()
        moved = property_failures.perturbed(rows, kind, alpha=study.margins.alpha)
        summary = _shared(moved, study)
        context = property_failures.Context(study, moved, summary)
        return property_failures.differential(summary, context, rules=rules)

    @staticmethod
    def _mutated(
        family: str, clauses: Sequence[property_failures.Clause]
    ) -> tuple[property_failures.FamilyRule, ...]:
        return tuple(
            dataclasses.replace(rule, clauses=tuple(clauses)) if family in rule.families else rule
            for rule in property_verdicts.FAMILY_RULES
        )

    @pytest.mark.parametrize("kind", property_failures.PERTURBATIONS)
    def test_the_shared_rules_agree_with_the_shared_verdicts(self, kind: str) -> None:
        assert self._disagreements(property_verdicts.FAMILY_RULES, kind).empty

    def test_the_perturbations_move_verdicts(self) -> None:
        study, rows = self._ladder()
        red = {
            kind: int((~_shared(property_failures.perturbed(rows, kind), study)["passed"]).sum())
            for kind in property_failures.PERTURBATIONS
        }
        assert red["none"] == 0
        assert all(red[kind] > 0 for kind in ("se_low", "se_high", "bias")), red

    def test_a_dropped_conjunct_disagrees(self) -> None:
        rule = _rule("root_n_and_efficiency")
        rules = self._mutated(
            "root_n_and_efficiency",
            [clause for clause in rule.clauses if clause.name != "se ratio sanity"],
        )
        wrong = self._disagreements(rules, "se_high")
        assert set(wrong["property"]) == {"root_n_and_efficiency"}

    def test_a_moved_threshold_disagrees(self) -> None:
        rule = _rule("root_n_and_efficiency")
        moved = property_failures.proportion_clause(
            "coverage floor",
            statistic="S2",
            judge=property_failures.floor_judge(lambda ctx: 0.97),
            applies=property_failures.roles("positive"),
        )
        rules = self._mutated(
            "root_n_and_efficiency",
            [moved if clause.name == "coverage floor" else clause for clause in rule.clauses],
        )
        wrong = self._disagreements(rules, "none")
        assert set(wrong["property"]) == {"root_n_and_efficiency"}

    def test_a_module_rule_takes_precedence_over_the_shared_one(self) -> None:
        """D3'a: a family resolves to the module's rule first, so its clauses are compared."""
        rule = _rule("root_n_and_efficiency")
        dropped = dataclasses.replace(rule, clauses=rule.clauses[:1])
        study, rows = self._ladder()
        moved = property_failures.perturbed(rows, "se_high")
        summary = _shared(moved, study)
        context = property_failures.Context(study, moved, summary)
        assert property_failures.differential(summary, context).empty
        module = SimpleNamespace(PROPERTY_RULES=(dropped,))
        assert property_failures.rule_for("root_n_and_efficiency", module) is dropped
        wrong = property_failures.differential(summary, context, module=module)
        assert set(wrong["property"]) == {"root_n_and_efficiency"}

    @pytest.mark.parametrize("kind", property_failures.binding_perturbations())
    def test_the_shared_rules_agree_on_each_graded_perturbation(self, kind: str) -> None:
        assert self._disagreements(property_verdicts.FAMILY_RULES, kind).empty

    def test_the_graded_perturbations_are_deterministic(self) -> None:
        _, rows = self._ladder()
        for kind in ("mixed/3", "mixed/4", "coverage/0.94", "two_point/1.06/1.05"):
            pd.testing.assert_frame_equal(
                property_failures.perturbed(rows, kind), property_failures.perturbed(rows, kind)
            )

    def test_coverage_and_rejection_reach_their_share(self) -> None:
        _, rows = self._ladder()
        covered = property_failures.perturbed(rows, "coverage/0.8")
        shares = covered.groupby(["property", "cell"])["covered"].mean()
        assert np.allclose(shares, 0.8, atol=1.0 / 380)
        pd.testing.assert_series_equal(covered["rejected"], rows["rejected"])
        rejected = property_failures.perturbed(rows, "rejection/0.1")
        shares = rejected.groupby(["property", "cell"])["rejected"].mean()
        assert np.allclose(shares, 0.1, atol=1.0 / 380)

    def test_a_role_kind_moves_that_role_alone(self) -> None:
        """M1: the role kinds of the binding search leave the other role's rows unchanged."""
        _, rows = self._ladder()
        first = sorted(set(rows["cell"]))[0]
        rows = rows.assign(role=np.where(rows["cell"] == first, "control", rows["role"]))
        control = (rows["role"] == "control").to_numpy()
        assert control.any() and (~control).any()
        keys = [rows["property"], rows["cell"]]
        for kind in ("centre/control/0", "spread_role/control/4", "se_role/control/2"):
            moved = property_failures.perturbed(rows, kind)
            pd.testing.assert_frame_equal(moved.loc[~control], rows.loc[~control])
            assert not moved.loc[control].equals(rows.loc[control]), kind
        centred = property_failures.perturbed(rows, "centre/control/0")
        errors = centred["estimate"] - centred["truth"]
        assert np.allclose(errors[control].groupby([k[control] for k in keys]).mean(), 0.0)
        flipped = property_failures.perturbed(rows, "coverage_role/control/0.5")
        pd.testing.assert_series_equal(flipped["covered"][~control], rows["covered"][~control])
        shares = flipped.loc[control].groupby(["property", "cell"])["covered"].mean()
        assert np.allclose(shares, 0.5, atol=1.0 / 380)

    def test_a_composed_kind_applies_its_parts_in_order(self) -> None:
        _, rows = self._ladder()
        composed = property_failures.perturbed(rows, "se/1.03+coverage/0.9")
        stepwise = property_failures.perturbed(
            property_failures.perturbed(rows, "se/1.03"), "coverage/0.9"
        )
        pd.testing.assert_frame_equal(composed, stepwise)
        assert set(property_failures.targeted_perturbations()).isdisjoint(
            property_failures.binding_perturbations()
        )
        with pytest.raises(ValueError, match="unknown perturbation"):
            property_failures.perturbed(rows, "se/x")

    def test_an_extra_column_is_scaled_and_shifted_in_one_cell(self) -> None:
        _, rows = self._ladder()
        rows = rows.assign(bootstrap_failed=0)
        kinds = property_failures.extra_perturbations(rows)
        cells = rows.groupby(["property", "cell"]).ngroups
        assert len(kinds) == 2 * cells
        shifted = property_failures.perturbed(rows, kinds[1])
        _, column, family, cell, how = kinds[1].split("|")
        assert (column, how) == ("bootstrap_failed", "shift")
        inside = (shifted["property"] == family) & (shifted["cell"] == cell)
        assert (shifted.loc[inside, column] == property_failures.EXTRA_SHIFT).all()
        assert (shifted.loc[~inside, column] == 0).all()


def test_p34_a_module_that_writes_a_verdict_without_rules_is_reported() -> None:
    """P34: the verdict-writer check reports a module whose write no declared rule restates."""
    sources = {
        "tests.studies.toy_properties": (
            "from tests.studies import toy_helper\n"
            "def summarize_properties(summary):\n"
            "    summary.loc[summary['role'] == 'positive', 'passed'] = True\n"
            "    return summary\n"
        ),
        "tests.studies.toy_helper": "def helper(frame):\n    return frame.assign(property_passed=1)\n",
        "tests.studies.toy_clean": "def summarize_properties(summary):\n    return summary\n",
    }
    writes = property_failures.verdict_writes(sources["tests.studies.toy_properties"])
    assert writes == [3]
    undeclared = SimpleNamespace(__name__="tests.studies.toy_properties")
    found = property_failures.undeclared_verdict_writers(undeclared, source=sources.get)
    assert found == ["tests.studies.toy_helper", "tests.studies.toy_properties"]
    declared = SimpleNamespace(__name__="tests.studies.toy_properties", PROPERTY_RULES=())
    assert property_failures.undeclared_verdict_writers(declared, source=sources.get) == []
    clean = SimpleNamespace(__name__="tests.studies.toy_clean")
    assert property_failures.undeclared_verdict_writers(clean, source=sources.get) == []


def test_p35_a_clause_no_perturbation_moves_is_reported_unbound() -> None:
    """P35: a clause that holds on every perturbation decides nothing, so it is unbound."""
    control = _p6_rate(lose=False)
    rule = _rule("root_n_and_efficiency")
    inert = property_failures.proportion_clause(
        "coverage at least zero",
        statistic="S2",
        judge=property_failures.floor_judge(lambda ctx: 0.0),
        applies=property_failures.roles("positive"),
    )
    rules = tuple(
        dataclasses.replace(shared, clauses=(*shared.clauses, inert)) if shared is rule else shared
        for shared in property_verdicts.FAMILY_RULES
    )
    used: set[tuple[str, str]] = set()
    decisive: set[tuple[str, str]] = set()
    for kind in (*property_failures.PERTURBATIONS, *property_failures.binding_perturbations()):
        moved = property_failures.perturbed(control.rows, kind, alpha=control.study.margins.alpha)
        summary = _shared(moved, control.study)
        context = property_failures.Context(control.study, moved, summary)
        applying, deciding = property_failures.clause_binding(summary, context, rules=rules)
        used |= applying
        decisive |= deciding
    unbound = property_failures.unbound(used, decisive)
    assert ("root_n_and_efficiency", "coverage at least zero") in unbound
    # The mutation control: the real floor the inert clause sits beside is bound.
    assert ("root_n_and_efficiency", "coverage floor") in decisive


# ---- what each rev 3 control shows beyond its leg ------------------------------------------


def _leg(name: str) -> pd.Series:
    control = _control(name)
    cell = _of(_accounted(name), control.family, control.cell)
    return cell.loc[cell["leg"] == control.leg].iloc[0]


def _candidates(control: Control) -> dict[str, property_failures.Judged]:
    rule = next(rule for rule in control.rules if control.family in rule.families)
    clause = rule.clause(control.leg)
    assert clause.candidates is not None
    return dict(clause.candidates(_context(control), _row(control)))


def test_p20_an_interval_that_holds_zero_gives_no_displacement() -> None:
    """S12: ``D_min = 0`` when the attainable difference of means holds zero."""
    leg = _leg("p20 straddle")
    assert leg["worst_case"] == pytest.approx(-0.25)
    low, high, _ = property_failures.displacement_bounds(
        *(
            _control("p20 straddle").rows.loc[lambda f, c=cell: f["cell"] == c]
            for cell in ("a__targeted", "a__untargeted")
        ),
        truth_varies=False,
    )
    assert low < 0.0 < high


def test_p20_the_cross_key_arms_read_every_survivor() -> None:
    """S12 across keys: each arm's mean reads its own survivors, not only the shared keys."""
    same, cross = _leg("p20 same key"), _leg("p20 cross key")
    assert same["worst_case"] < 0.0 and cross["worst_case"] < 0.0
    assert same["worst_case"] != pytest.approx(cross["worst_case"], abs=1e-12)


def test_p21_truth_outside_the_denominator_s_range_is_conservative() -> None:
    """S13: a denominator whose errors all lie in [0.5, 1.5] is filled at truth, which no
    completion in its range reaches, so the leg is more adverse than any range completion."""
    denominator = 1.0 + 0.5 * _signs(380)
    control = _selector(denominator=denominator)
    ((_, judged),) = _candidates(control).items()
    numerator = control.rows.loc[control.rows["cell"] == "collaborative"]
    lost = sorted(set(range(400)) - set(numerator["replicate"]))

    def completed(frame: pd.DataFrame, fill: float) -> pd.DataFrame:
        filled_rows = pd.DataFrame(
            {
                "replicate": lost,
                "estimate": fill,
                "truth": 0.0,
                "failed_replicates": 0,
                "requested_replicates": 400,
            }
        )
        whole = pd.concat([frame.assign(failed_replicates=0), filled_rows], ignore_index=True)
        return whole.sort_values("replicate", ignore_index=True)

    denominator_rows = control.rows.loc[control.rows["cell"] == "empty_control"]
    in_range = max(
        _rmse_ratio(completed(numerator, 2.0), completed(denominator_rows, fill))
        for fill in np.linspace(0.5, 1.5, 21)
    )
    assert 0.5 - judged.slack > in_range


def test_p22_a_split_search_and_a_mean_fill_both_miss_the_least_small_bound() -> None:
    """S14: the small cell's least bound sits at an interior equal fill."""
    control = _control("p22")
    errors = control.rows.loc[control.rows["cell"] == "small", "estimate"].to_numpy(dtype=float)
    least, centre = property_failures._equal_fill_bound(errors, 2)
    splits = min(property_failures.error_bias_bound(vertex(errors, 2, j)) for j in range(3))
    at_mean = property_failures.error_bias_bound(
        np.concatenate([errors, np.full(2, float(np.mean(errors)))])
    )
    large = float(_of(control.summary, SINGLE, "large")["absolute_bias_bound"].iloc[0])
    assert errors.min() < centre < errors.max() and not np.isclose(centre, errors.mean())
    # The large bound lies between the least small bound and what either search finds, so the
    # clause fails only under the equal-fill search.
    assert least < large < min(splits, at_mean)
    assert _leg("p22")["worst_case"] == pytest.approx(least - large)


def test_p24_the_fill_nearest_a_zero_mean_breaks_the_contraction() -> None:
    judged = _candidates(_control("p24"))
    zero = [label for label in judged if "nearest a zero mean" in label]
    assert zero and all(not judged[label].passed for label in zero)


def test_p25_only_the_pair_that_lost_turns_red() -> None:
    control = _control("p25")
    (pairs,) = control.groups.values()
    corrected = property_failures.correct_properties(
        control.summary, _accounted("p25"), module=_module(control), rules=control.rules
    ).set_index("cell")
    assert not corrected.loc[list(pairs[0]), "property_passed"].any()
    others = [cell for pair in pairs[1:] for cell in pair]
    assert len(others) >= 2 and corrected.loc[others, "property_passed"].all()
    assert set(_accounted("p25")["cell"]) == set(pairs[0])


def test_p26_the_shared_calibration_rule_alone_stays_green() -> None:
    """G1: without the module's cap clause the same loss reads within the bound."""
    control = _control("p26")
    shared = _account(control, rules=property_verdicts.FAMILY_RULES)
    assert set(shared["reason"]) == {WITHIN_BOUND}
    assert set(_accounted("p26")["reason"]) == {WORST_CASE_FAILS}
    assert _leg("p26")["imputation"].startswith(property_failures.SUPPORT)


def test_p27_the_hook_reads_the_verdict_s_own_statistic() -> None:
    """G6: ``projection_sd_ratio`` and ``paired_spread_ratio_interval`` differ on these rows."""
    from tests.studies import stochastic_categorical_ltmle_properties as module

    control = _control("p27")
    rows = control.rows
    seed = stream_seed(control.study, PROJECTION, "sd_ratio")
    _, upper = module.projection_sd_ratio(
        rows,
        replicates=control.study.margins.bootstrap_replicates,
        level=module.PROJECTION_LEVEL,
        seed=seed,
    )
    default = paired_spread_ratio_interval(
        rows.loc[rows["cell"] == "mix__integrated"],
        rows.loc[rows["cell"] == "mix__recorded_randomizer"],
        replicates=control.study.margins.bootstrap_replicates,
        confidence_level=control.study.margins.confidence_level,
        seed=seed,
    )
    assert default.interval.high != pytest.approx(upper, rel=1e-6)


def test_p28_a_module_rule_s_conjunct_turns_the_cell_red_and_the_shared_rule_does_not() -> None:
    """D3'a precedence on a real module: ``full_refit_bootstrap``'s cap is read only through
    the module's own rule.  The same cell resolved to the shared rule is green."""
    from tests.studies import full_refit_bootstrap_properties as module

    control = _control("p26")
    declared = SimpleNamespace(
        FAILURE_BOUND=property_failures.MAXIMUM_BOUND, PROPERTY_RULES=module.PROPERTY_RULES
    )
    assert property_failures.rule_for("interval_calibration", declared) is module.PROPERTY_RULES[0]
    assert property_failures.rule_for("interval_calibration") is _rule("interval_calibration")
    records = _gap_records(control.rows)
    resolved = property_failures.property_accounting(
        control.rows, records, control.summary, record=control.study, module=declared
    )
    shared = property_failures.property_accounting(
        control.rows, records, control.summary, record=control.study, module=BOUNDED
    )
    assert set(resolved["reason"]) == {WORST_CASE_FAILS}
    assert set(shared["reason"]) == {WITHIN_BOUND}
    corrected = property_failures.correct_properties(control.summary, resolved, module=declared)
    assert not corrected["passed"].astype(bool).any()


def test_p31_only_the_label_that_lost_turns_red() -> None:
    control = _control("p31")
    (losing, kept) = control.groups[GENERATED]
    corrected = property_failures.correct_properties(
        control.summary, _accounted("p31"), module=_module(control), rules=control.rules
    ).set_index("cell")
    assert not corrected.loc[list(losing), "property_passed"].any()
    assert corrected.loc[losing[0], "passed"] and not corrected.loc[losing[1], "passed"]
    assert corrected.loc[list(kept), "property_passed"].all()


def test_a_label_rule_needs_a_declaration() -> None:
    control = _control("p31")
    with pytest.raises(ValueError, match="declares no label groups"):
        _account(dataclasses.replace(control, groups={}))


def test_p7_the_aligned_fill_never_takes_an_infeasible_difference() -> None:
    """G5: across keys, a lost control arm whose positive survived covered has a difference of
    0 or 1, so the old fill at -1 is infeasible.  The aligned fill reads 0 there, and still
    breaks the gain through the lost positive arms whose controls survived covered."""
    control = _p7_coverage_gain(across_keys=True)
    rows = control.rows
    positive = rows.loc[rows["cell"] == "cross_fit"].set_index("replicate")["covered"]
    negative = rows.loc[rows["cell"] == "in_sample_control"].set_index("replicate")["covered"]
    control_lost = sorted(set(range(4200)) - set(negative.index))
    positive_lost = sorted(set(range(4200)) - set(positive.index))
    infeasible = [replicate for replicate in control_lost if positive.get(replicate) == 1]
    assert infeasible, "no control-lost key whose positive survived covered"
    judged = _candidates(dataclasses.replace(control, leg="coverage gain"))
    assert not any(value.passed for value in judged.values())
    # The rev 2 fill set every lost difference to -1: strictly more adverse than any completion.
    kept = sorted(set(positive.index) & set(negative.index))
    differences = (positive.loc[kept] - negative.loc[kept]).to_numpy(dtype=float)
    lost = len(set(control_lost) | set(positive_lost))
    old = np.concatenate([differences, -np.ones(lost)])
    aligned = np.concatenate(
        [
            differences,
            [
                positive.get(replicate, 0.0) - negative.get(replicate, 1.0)
                for replicate in sorted(set(control_lost) | set(positive_lost))
            ],
        ]
    )
    assert old.mean() < aligned.mean()


class TestNotAccountable:
    """p11 (R3-1): an identity claim or the fit's own numerical success, with a lost
    replicate, reads ``failures not accountable`` at any bound."""

    @staticmethod
    def _identity(family: str) -> Control:
        study = _study()
        rows = _arm(family, "subject", "positive", requested=400, lost=(3,), estimates=_signs(399))
        clause = property_failures.unaccountable_clause(
            "identity within its tolerance",
            statistic="sup-norm identity",
            read=lambda ctx, row: property_failures.Judged(math.nan, bool(row["identity"])),
            applies=property_failures.every,
        )
        rules = _one_rule(family, clause)
        summary = _summary_of(
            family, rows, rules, columns={"subject": {"identity": True}}, study=study
        )
        return Control(rows, summary, study, family, "subject", clause.name, rules)

    @pytest.mark.parametrize("family", ["natural_course_identity", "static_reduction"])
    def test_an_identity_cell_that_lost_a_replicate_is_red(self, family: str) -> None:
        control = self._identity(family)
        assert bool(_row(control)["passed"])
        accounting = _account(control)
        assert set(accounting["reason"]) == {NOT_ACCOUNTABLE}
        legs = accounting.loc[accounting["leg"] == property_failures.NOT_ACCOUNTABLE_LEG]
        assert list(legs["imputation"]) == ["identity within its tolerance"]
        corrected = property_failures.correct_properties(
            control.summary, accounting, module=_module(control), rules=control.rules
        )
        assert not corrected["passed"].any()

    @staticmethod
    def _correction(lost: dict[str, Sequence[int]]) -> Control:
        """``correction_necessity``: an initial score control (S1f, accountable) and a closed
        score row that reads the initial row's endpoint (not accountable)."""
        study = _study()
        family = "correction_necessity"
        rows = pd.concat(
            [
                _arm(
                    family,
                    "ate__initial_score_control",
                    "control",
                    requested=400,
                    lost=lost.get("initial", ()),
                    estimates=_signs(400 - len(lost.get("initial", ()))) + 2.0,
                ),
                _arm(
                    family,
                    "ate__closed_score",
                    "positive",
                    requested=400,
                    lost=lost.get("closed", ()),
                    estimates=0.01 * _signs(400 - len(lost.get("closed", ()))),
                ),
            ],
            ignore_index=True,
        )
        initial = property_failures.bias_clause(
            "initial score above its floor",
            kind="beyond",
            side=lambda ctx, row: 1,
            margin=lambda ctx, row: 1.0,
            strict=False,
            statistic="S1f",
            applies=property_failures.cells("ate__initial_score_control"),
        )

        def closes(ctx: property_failures.Context, row: pd.Series) -> property_failures.Judged:
            summary = ctx.summary.set_index("cell")
            ceiling = 0.1 * float(summary.loc["ate__initial_score_control", "bias_ci_lower"])
            return property_failures.Judged(
                ceiling - float(row["bias_ci_upper"]), float(row["bias_ci_upper"]) <= ceiling
            )

        closed = property_failures.unaccountable_clause(
            "closed score under the ratio",
            statistic="numerical success",
            read=closes,
            applies=property_failures.cells("ate__closed_score"),
            reads=lambda ctx, row: [(family, "ate__initial_score_control")],
        )
        rules = _one_rule(family, initial, closed)
        columns = {
            cell: _bias_columns(rows.loc[rows["cell"] == cell], study)
            for cell in ("ate__initial_score_control", "ate__closed_score")
        }
        summary = _summary_of(family, rows, rules, columns=columns, study=study)
        assert summary["passed"].all()
        return Control(rows, summary, study, family, "ate__closed_score", closed.name, rules)

    def test_a_lost_closed_score_replicate_is_red_and_the_initial_row_is_not_read(self) -> None:
        accounting = _account(self._correction({"closed": (3,)}))
        assert set(accounting["cell"]) == {"ate__closed_score"}
        assert set(accounting["reason"]) == {NOT_ACCOUNTABLE}

    def test_a_lost_initial_replicate_leaves_the_initial_row_accountable(self) -> None:
        accounting = _account(self._correction({"initial": (3,)}))
        initial = _of(accounting, "correction_necessity", "ate__initial_score_control")
        closed = _of(accounting, "correction_necessity", "ate__closed_score")
        assert set(initial["reason"]) == {WITHIN_BOUND}
        assert "initial score above its floor" in set(initial["leg"])
        # The closed row reads the initial row's endpoint, so it is red too (R3-1).
        assert set(closed["reason"]) == {NOT_ACCOUNTABLE}
        assert set(closed["failed"]) == {0}


class TestWitness:
    """p37 (D3'f): each lost replicate's witness, placed among the cell's survivors."""

    @staticmethod
    def _accounting(witnesses: pd.DataFrame | None) -> pd.DataFrame:
        control = _control("p2")
        return property_failures.property_accounting(
            control.rows,
            _gap_records(control.rows),
            control.summary,
            record=control.study,
            module=_module(control),
            witnesses=witnesses,
        )

    def test_a_lost_replicate_s_witness_carries_its_survivor_quantile(self) -> None:
        values = np.arange(400, dtype=float)
        witnesses = pd.DataFrame(
            {
                "property": "double_robustness",
                "cell": "outcome_correct",
                "replicate": range(400),
                "statistic": "mean outcome",
                "value": values,
            }
        )
        accounting = self._accounting(witnesses)
        rows = accounting.loc[accounting["leg"] == property_failures.WITNESS_LEG]
        lost = _every(20, 20)
        assert list(rows["complete_case"]) == [float(replicate) for replicate in lost]
        survivors = np.array([v for v in values if int(v) not in set(lost)])
        expected = [float(np.mean(survivors < value)) for value in rows["complete_case"]]
        assert list(rows["worst_case"]) == pytest.approx(expected)
        assert rows["imputation"].iloc[1] == "replicate 20: mean outcome"
        assert rows["leg_passed"].all()
        # Reported, never gated: the cell's gate is its legs'.
        assert set(rows["reason"]) == {WORST_CASE_FAILS}

    def test_a_draw_that_does_not_separate_reports_nan_with_its_reason(self) -> None:
        witnesses = pd.DataFrame(
            {
                "property": "double_robustness",
                "cell": "outcome_correct",
                "replicate": range(400),
                "statistic": DRAW_NOT_SEPARABLE,
                "value": np.nan,
            }
        )
        rows = self._accounting(witnesses)
        rows = rows.loc[rows["leg"] == property_failures.WITNESS_LEG]
        assert rows["complete_case"].isna().all() and rows["worst_case"].isna().all()
        assert rows["imputation"].str.endswith(DRAW_NOT_SEPARABLE).all()

    def test_mid_rank_ties(self) -> None:
        assert property_failures._mid_rank(1.0, np.array([0.0, 1.0, 1.0, 2.0])) == 0.5


class TestSampleWitness:
    """D3'f: the declared witness reads the draw alone, through any container a study draws."""

    def test_a_binary_column_reads_its_minority_share(self) -> None:
        treatment = (np.arange(200) < 10).astype(float)
        p = 10 / 200
        expected = (1 - p) / np.std(treatment, ddof=1)
        assert largest_standardized_value(pd.DataFrame({"A": treatment})) == pytest.approx(expected)
        # The witness separates draws: a nearly empty arm reads larger than a balanced one.
        balanced = (np.arange(200) < 100).astype(float)
        assert largest_standardized_value(pd.DataFrame({"A": balanced})) < expected

    def test_an_outlier_in_any_container_is_found(self) -> None:
        rng = np.random.default_rng(3)
        covariates = rng.normal(size=(300, 2))
        covariates[7, 1] = 12.0
        data = CausalData.from_arrays(
            rng.normal(size=300), (np.arange(300) % 2).astype(float), covariates
        )
        frame = pd.DataFrame(covariates, columns=["x0", "x1"])
        column = covariates[:, 1]
        expected = float(np.max(np.abs(column - column.mean())) / np.std(column, ddof=1))
        for drawn in (data, (data, {"ate": 1.0}), frame, {"frame": frame}, [covariates]):
            assert largest_standardized_value(drawn) == pytest.approx(expected)

    def test_a_draw_with_no_spread_reads_nan(self) -> None:
        assert np.isnan(largest_standardized_value(pd.DataFrame({"A": np.ones(5)})))
        assert np.isnan(largest_standardized_value(("text", 3.0)))
        assert math.isnan(property_failures._mid_rank(math.nan, np.array([0.0])))


def test_p38_each_type_s_legs_carry_its_label() -> None:
    """D3'e: bounded types fill over their support, the rest rest on the range assumption."""
    expected = {
        "p1": property_failures.SUPPORT,  # S2
        "p2": property_failures.RANGE,  # S1
        "p4": property_failures.RANGE,  # S5
        "p5": property_failures.SUPPORT,  # S2
        "p6": property_failures.RANGE,  # S6
        "p7": property_failures.SUPPORT,  # S7
        "p9": property_failures.RANGE,  # S9
        "p10": property_failures.SUPPORT,  # S10 on a share
        "p18": property_failures.RANGE,  # S1f
        "p20 same key": property_failures.RANGE,  # S12
        "p21": property_failures.RANGE,  # S13
        "p22": property_failures.RANGE,  # S14
        "p33": property_failures.RANGE,  # S10 on an unbounded value
    }
    for name, label in expected.items():
        assert _leg(name)["imputation"].startswith(f"{label}: "), name
    rule = property_verdicts.FAMILY_RULES
    for item in rule:
        for clause in item.clauses:
            bounded = clause.statistic in {"S2", "S3", "S7"}
            assert (clause.imputation == property_failures.SUPPORT) is bounded, clause.name


GROWTH = "synthetic_one_wrong"


def _t_draws(ctx: property_failures.Context, values: np.ndarray, seed: int) -> np.ndarray:
    """Bootstrap draws of a rung's ``mean / sd``, the verdict's per-rung statistic."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(ctx.margins.bootstrap_replicates, len(values)))
    sampled = values[index]
    return sampled.mean(axis=1) / sampled.std(axis=1, ddof=1)


def _growth_clause(*, equal_fill: bool = True) -> property_failures.Clause:
    return property_failures.growth_clause(
        "growth above zero",
        rungs=lambda ctx, row: (("small", 1000), ("large", 8000)),
        sign=lambda ctx, row: -1.0,
        seed=lambda ctx, row, size: size,
        draws=_t_draws,
        judge=property_failures.above_judge(lambda ctx: 0.0),
        applies=property_failures.cells("growth"),
        equal_fill=equal_fill,
    )


def _growth(
    small: np.ndarray,
    large: np.ndarray,
    *,
    lost_small: Sequence[int],
    lost_large: Sequence[int],
    bootstrap: int = 2000,
    equal_fill: bool = True,
) -> Control:
    """A first-order bias that grows with ``n``: ``G = -(t_large - t_small)`` above zero."""
    study = _study(bootstrap)
    requested_small = len(small) + len(lost_small)
    requested_large = len(large) + len(lost_large)
    rows = pd.concat(
        [
            _arm(
                GROWTH,
                "small",
                "control",
                requested=requested_small,
                lost=lost_small,
                estimates=small,
                n=1000,
            ),
            _arm(
                GROWTH,
                "large",
                "control",
                requested=requested_large,
                lost=lost_large,
                estimates=large,
                n=8000,
            ),
        ],
        ignore_index=True,
    )
    clause = _growth_clause(equal_fill=equal_fill)
    context = property_failures.Context(study, rows, pd.DataFrame())
    draws = [_t_draws(context, values, size) for values, size in ((small, 1000), (large, 8000))]
    interval = property_failures.percentile_interval(
        -(draws[1] - draws[0]), confidence_level=study.margins.confidence_level
    )
    growth = {
        "property": GROWTH,
        "cell": "growth",
        "role": "control",
        "replicates": np.nan,
        "failed_replicates": np.nan,
        "growth_ci_lower": interval.low,
        "growth_ci_upper": interval.high,
    }
    rules = _one_rule(GROWTH, clause)
    summary = pd.DataFrame([growth])
    summary["passed"] = bool(interval.low > 0.0)
    summary["property_passed"] = summary["passed"]
    rule = rules[0]
    return Control(rows, summary, study, GROWTH, "growth", clause.name, (rule,))


def test_p6_only_an_interior_split_leaves_the_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """G3: the rev 2 set (the point-widest split and the mean fill) stays in the band, an
    interior split leaves it, and the point slope stays in the band under every fill."""
    control = _control("p6 interior")
    values = control.rows.loc[control.rows["cell"] == "n_250", "estimate"].to_numpy(dtype=float)
    widest = max(range(21), key=lambda j: float(np.std(vertex(values, 20, j), ddof=1)))
    pruned = _candidates(control)
    monkeypatch.setattr(
        property_failures, "prune", lambda options, weight, direction: list(options)
    )
    every = _candidates(control)
    failing = {label for label, judged in every.items() if not judged.passed}
    assert failing and failing <= set(pruned)
    splits = {int(label.split("j=")[1].split()[0]) for label in failing}
    assert all(0 < j < 20 and j != widest for j in splits), (splits, widest)
    assert every[f"n_250 j={widest} at min"].passed and every["n_250 at survivor mean"].passed
    sizes = np.array([250.0, 1000.0, 4000.0])
    others = [
        float(
            np.log(np.std(control.rows.loc[control.rows["cell"] == f"n_{n}", "estimate"], ddof=1))
        )
        for n in (1000, 4000)
    ]
    for _, option in property_failures.rung_options(values, 20, "spread", 0.0):
        slope = property_failures.log_slope(
            sizes, np.array([float(np.log(np.std(option, ddof=1))), *others])
        )
        assert -0.625 < float(slope) < -0.375


#: p23's rungs: standardized normal quantiles, permuted, shifted to a chosen ``t``.
def _quantiles(size: int, t: float, seed: int) -> np.ndarray:
    from scipy.stats import norm

    values = norm.ppf((np.arange(size) + 0.5) / size)
    values = values[np.random.default_rng(seed).permutation(size)]
    return (values - values.mean()) / values.std(ddof=1) + t


def _growth_options_l(control: Control, cell: str, size: int) -> list[tuple[str, np.ndarray]]:
    values = control.rows.loc[control.rows["cell"] == cell, "estimate"].to_numpy(dtype=float)
    k = 1000 - len(values)
    context = _context(control)
    return [
        (label, _t_draws(context, option, size))
        for label, option in property_failures.growth_options(values, k)
    ]


def _lower(small: np.ndarray, large: np.ndarray) -> float:
    return property_failures.percentile_interval(-(large - small), confidence_level=0.99).low


def test_p23_only_the_equal_fill_breaks_the_growth() -> None:
    """S15 (H1): at ``t = -0.88`` every vertex is less extreme than the complete case and the
    equal fill near -2.0 is more extreme, so only the equal fill turns ``L > 0`` red."""
    small = _quantiles(950, -0.88, 0)
    control = _growth(small, _quantiles(4000, -1.035, 1), lost_small=_every(20, 50), lost_large=[])
    assert bool(_row(control)["passed"])
    complete = property_failures.standardized_bias(small)
    options = property_failures.growth_options(small, 50)
    vertices = [property_failures.standardized_bias(v) for label, v in options if "j=" in label]
    least = [property_failures.standardized_bias(v) for label, v in options if "least" in label]
    assert min(vertices) > complete > least[0]
    context = _context(control)
    large = _t_draws(
        context,
        control.rows.loc[control.rows["cell"] == "large", "estimate"].to_numpy(dtype=float),
        8000,
    )
    lower = {
        label: _lower(draws, large) for label, draws in _growth_options_l(control, "small", 1000)
    }
    assert all(value > 0.0 for label, value in lower.items() if "j=" in label)
    assert any(value <= 0.0 for label, value in lower.items() if "least" in label)
    accounting = _account(control)
    assert set(accounting["reason"]) == {WORST_CASE_FAILS}
    # The p30 mutation: without the equal fills the leg reads green.
    mutated = _account(control, rules=_one_rule(GROWTH, _growth_clause(equal_fill=False)))
    assert set(mutated["reason"]) == {WITHIN_BOUND}


def test_p23_only_the_product_of_both_rungs_breaks_the_growth() -> None:
    """S15: each rung at its worst with the other at its most favourable fill keeps ``L > 0``;
    both at their worst do not.  The candidate set is the product across the rungs."""
    control = _growth(
        _quantiles(950, -0.3, 0),
        _quantiles(950, -0.5, 1),
        lost_small=_every(20, 50),
        lost_large=_every(20, 50),
    )
    assert bool(_row(control)["passed"])
    small = _growth_options_l(control, "small", 1000)
    large = _growth_options_l(control, "large", 8000)
    grid = np.array([[_lower(s, l) for _, l in large] for _, s in small])
    assert grid.min() <= 0.0
    assert grid.min(axis=0).max() > 0.0 and grid.min(axis=1).max() > 0.0
    judged = _candidates(control)
    failing = [label for label, value in judged.items() if not value.passed]
    assert failing and all("small" in label and "large" in label for label in failing)


# ---- p30: brute-force exactness of every type labelled exact (M7) --------------------------

#: Random completions per case.  Each case also tries every vertex of its box.
COMPLETIONS = 10_000


def _box(rng: np.random.Generator, values: np.ndarray, k: int) -> list[np.ndarray]:
    """Completions of ``values`` with ``k`` fills in the survivors' range: every vertex, then
    :data:`COMPLETIONS` uniform draws."""
    low, high = float(np.min(values)), float(np.max(values))
    corners = [np.array(corner, dtype=float) for corner in itertools.product((low, high), repeat=k)]
    draws = list(rng.uniform(low, high, size=(COMPLETIONS, k)))
    return [np.concatenate([values, fill]) for fill in [*corners, *draws]]


def _exact_cell(seed: int, k: int, *, shift: float = 0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(shift, 1.0, size=30 - k)


def _point_extremes(options: Sequence[np.ndarray], statistic) -> tuple[float, float]:
    values = [float(statistic(option)) for option in options]
    return min(values), max(values)


CASES = [(seed, k) for seed in (1, 2, 3) for k in (1, 2, 3)]

#: How far the box's least slack may sit above the worst candidate.  A candidate is a
#: completion in the box, so for an exact type the box attains the worst case.
ATTAINED = 1e-9


def _sound_and_attained(slacks: Sequence[float], worst: float) -> None:
    """No completion is more adverse than the worst candidate, and one is as adverse."""
    assert min(slacks) >= worst - 1e-12
    assert min(slacks) <= worst + ATTAINED


def _pairs(rng: np.random.Generator, left: np.ndarray, right: np.ndarray, k: int) -> list:
    """Completions of two cells: the diagonal of their boxes, and every pair of corners."""
    lefts, rights = _box(rng, left, k), _box(rng, right, k)
    corners = itertools.product(lefts[: 2**k], rights[: 2**k])
    return [*zip(lefts, rights, strict=True), *corners]


class TestExactness:
    """p30: for each type labelled exact, no completion in the box is more adverse than the
    worst candidate, and one completion is as adverse, at ``R = 30`` and ``k = 1..3``.

    Two cells enumerate every pair of their corners.  Where a candidate fills inside the range,
    such as an equal fill, its completion joins the box."""

    @staticmethod
    def _cell_control(errors: np.ndarray, k: int, clause, columns, *, role="positive") -> Control:
        study = _study()
        family = "synthetic_exact"
        rows = _arm(family, "subject", role, requested=30, lost=range(30 - k, 30), estimates=errors)
        rules = _one_rule(family, clause)
        summary = _summary_of(family, rows, rules, columns={"subject": columns}, study=study)
        return Control(rows, summary, study, family, "subject", clause.name, rules)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    @pytest.mark.parametrize("kind", ["equivalent", "beyond"])
    def test_s1(self, seed: int, k: int, kind: str) -> None:
        errors = _exact_cell(seed, k, shift=0.6 if kind == "beyond" else 0.0)
        study = _study()
        clause = property_failures.bias_clause(
            "bias", kind=kind, margin=lambda ctx, row: 0.3, applies=property_failures.every
        )
        control = self._cell_control(
            errors,
            k,
            clause,
            _bias_columns(_arm("x", "y", "positive", requested=30 - k, estimates=errors), study),
        )
        worst = min(judged.slack for judged in _candidates(control).values())
        side = 1 if np.mean(errors) >= 0 else -1
        level = study.margins.confidence_level
        rng = np.random.default_rng(seed)
        slacks = []
        for completed in _box(rng, errors, k):
            interval = student_interval(completed, confidence_level=level)
            if kind == "equivalent":
                slacks.append(0.3 - max(-interval.low, interval.high))
            else:
                slacks.append(interval.low - 0.3 if side > 0 else -0.3 - interval.high)
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    @pytest.mark.parametrize("statistic", ["S2", "S3"])
    def test_s2_s3(self, seed: int, k: int, statistic: str) -> None:
        rng = np.random.default_rng(seed)
        hits = (rng.random(30 - k) < 0.85).astype(int)
        column, prefix = ("covered", "coverage") if statistic == "S2" else ("rejected", "rejection")
        level = _study().margins.confidence_level
        interval = clopper_pearson(int(hits.sum()), 30 - k, confidence_level=level)
        rate_column = "coverage" if statistic == "S2" else "rejection_rate"
        clause = property_failures.proportion_clause(
            "rate",
            statistic=statistic,
            judge=property_failures.band_judge(lambda ctx: (0.7, 0.95)),
            applies=property_failures.every,
        )
        study = _study()
        family = "synthetic_exact"
        rows = _arm(
            family,
            "subject",
            "positive",
            requested=30,
            lost=range(30 - k, 30),
            estimates=_signs(30 - k),
            **{column: hits},
        )
        rules = _one_rule(family, clause)
        summary = _summary_of(
            family,
            rows,
            rules,
            study=study,
            columns={
                "subject": {
                    rate_column: float(hits.mean()),
                    f"{prefix}_ci_lower": interval.low,
                    f"{prefix}_ci_upper": interval.high,
                }
            },
        )
        control = Control(rows, summary, study, family, "subject", "rate", rules)
        worst = min(judged.slack for judged in _candidates(control).values())
        slacks = []
        for fill in itertools.product((0, 1), repeat=k):
            whole = clopper_pearson(int(hits.sum()) + sum(fill), 30, confidence_level=level)
            slacks.append(min(whole.low - 0.7, 0.95 - whole.high))
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    def test_s4(self, seed: int, k: int) -> None:
        rng = np.random.default_rng(seed)
        errors = _exact_cell(seed, k)
        std_errors = rng.uniform(0.8, 1.2, size=30 - k)
        study = _study()
        clause = property_failures.se_point_clause(
            "ratio", band=lambda ctx: (0.8, 1.2), applies=property_failures.every
        )
        family = "synthetic_exact"
        rows = _arm(
            family,
            "subject",
            "positive",
            requested=30,
            lost=range(30 - k, 30),
            estimates=errors,
            std_errors=std_errors,
        )
        ratio = float(std_errors.mean() / np.std(errors, ddof=1))
        rules = _one_rule(family, clause)
        summary = _summary_of(
            family, rows, rules, study=study, columns={"subject": {"se_ratio": ratio}}
        )
        control = Control(rows, summary, study, family, "subject", "ratio", rules)
        worst = min(judged.slack for judged in _candidates(control).values())
        # The ratio is extreme at independent ends of the two boxes, so every corner pair runs.
        # The SE-ratio set's high end fills the SEs at their maximum and the errors at their
        # mean, inside the box: pair that completion too.
        at_mean = np.concatenate([errors, np.full(k, float(np.mean(errors)))])
        at_max = np.concatenate([std_errors, np.full(k, float(np.max(std_errors)))])
        slacks = []
        for completed, ses in [*_pairs(rng, errors, std_errors, k), (at_mean, at_max)]:
            value = float(ses.mean() / np.std(completed, ddof=1))
            slacks.append(min(value - 0.8, 1.2 - value))
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    def test_s7(self, seed: int, k: int) -> None:
        rng = np.random.default_rng(seed)
        positive_lost = list(range(30 - k, 30))
        control_lost = list(range(k))
        positive_hits = (rng.random(30 - k) < 0.9).astype(int)
        control_hits = (rng.random(30 - k) < 0.5).astype(int)
        study = _study()
        family = "synthetic_exact_pair"
        rows = pd.concat(
            [
                _arm(
                    family,
                    "positive",
                    "positive",
                    requested=30,
                    lost=positive_lost,
                    estimates=_signs(30 - k),
                    covered=positive_hits,
                ),
                _arm(
                    family,
                    "control",
                    "control",
                    requested=30,
                    lost=control_lost,
                    estimates=_signs(30 - k),
                    covered=control_hits,
                ),
            ],
            ignore_index=True,
        )
        clause = property_failures.coverage_gain_clause(
            "gain", floor=0.2, seed=lambda ctx, row: 9, applies=property_failures.every
        )
        kept_p, kept_c = paired(
            rows.loc[rows["cell"] == "positive"], rows.loc[rows["cell"] == "control"]
        )
        interval = mean_difference_interval(
            kept_p["covered"].to_numpy(float) - kept_c["covered"].to_numpy(float),
            replicates=study.margins.bootstrap_replicates,
            confidence_level=study.margins.confidence_level,
            seed=9,
        )
        rules = _one_rule(family, clause, group="pair")
        summary = _summary_of(
            family,
            rows,
            rules,
            study=study,
            columns={
                cell: {
                    "coverage_gain_ci_lower": interval.low,
                    "coverage_gain_ci_upper": interval.high,
                }
                for cell in ("positive", "control")
            },
        )
        control = Control(rows, summary, study, family, "positive", "gain", rules)
        worst = min(judged.slack for judged in _candidates(control).values())
        positive = dict(zip(range(30 - k), positive_hits, strict=True))
        negative = dict(zip(range(k, 30), control_hits, strict=True))
        slacks = []
        for fills in itertools.product((0, 1), repeat=2 * k):
            p = {**positive, **dict(zip(positive_lost, fills[:k], strict=True))}
            c = {**negative, **dict(zip(control_lost, fills[k:], strict=True))}
            order = [*range(30 - k), *positive_lost]
            differences = np.array([p[i] - c[i] for i in order], dtype=float)
            whole = mean_difference_interval(
                differences,
                replicates=study.margins.bootstrap_replicates,
                confidence_level=study.margins.confidence_level,
                seed=9,
            )
            slacks.append(whole.low - 0.2)
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    def test_s12_under_the_held_scale(self, seed: int, k: int) -> None:
        rng = np.random.default_rng(seed)
        left = _exact_cell(seed, k)
        right = rng.normal(0.8, 1.0, size=30 - k)
        rows = pd.concat(
            [
                _arm(
                    DISPLACEMENT,
                    "a__targeted",
                    "positive",
                    requested=30,
                    lost=range(30 - k, 30),
                    estimates=left,
                ),
                _arm(
                    DISPLACEMENT,
                    "a__untargeted",
                    "control",
                    requested=30,
                    lost=range(k),
                    estimates=right,
                ),
            ],
            ignore_index=True,
        )
        clause = property_failures.displacement_clause(
            "d",
            column="displacement",
            labels=("a",),
            arms=("targeted", "untargeted"),
            threshold=0.5,
        )
        rules = _one_rule(DISPLACEMENT, clause, group="family")
        value = paired_displacement(rows, DISPLACEMENT, "a__targeted", "a__untargeted")
        summary = _summary_of(
            DISPLACEMENT,
            rows,
            rules,
            study=_study(),
            columns={cell: {"displacement": value} for cell in ("a__targeted", "a__untargeted")},
        )
        control = Control(rows, summary, _study(), DISPLACEMENT, "a__targeted", "d", rules)
        worst = min(judged.slack for judged in _candidates(control).values())
        kept, _ = paired(
            rows.loc[rows["cell"] == "a__targeted"], rows.loc[rows["cell"] == "a__untargeted"]
        )
        held = float(np.std(kept.sort_values("replicate")["estimate"], ddof=1))
        slacks = [
            abs(float(np.mean(completed_right) - np.mean(completed_left))) / held - 0.5
            for completed_left, completed_right in _pairs(rng, left, right, k)
        ]
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    def test_s13_with_truth_in_the_denominator_s_range(self, seed: int, k: int) -> None:
        rng = np.random.default_rng(seed)
        numerator = rng.normal(0.0, 0.6, size=30 - k)
        denominator = rng.normal(0.0, 1.0, size=30 - k)
        rows = pd.concat(
            [
                _arm(
                    SELECTOR,
                    "collaborative",
                    "positive",
                    requested=30,
                    lost=range(30 - k, 30),
                    estimates=numerator,
                ),
                _arm(
                    SELECTOR,
                    "empty_control",
                    "control",
                    requested=30,
                    lost=range(30 - k, 30),
                    estimates=denominator,
                ),
            ],
            ignore_index=True,
        )
        clause = property_failures.rmse_ratio_clause(
            "r", paths=[("collaborative", "empty_control")], statistic=_rmse_ratio, threshold=0.5
        )
        rules = _one_rule(SELECTOR, clause, group="family")
        ratio = _rmse_ratio(
            rows.loc[rows["cell"] == "collaborative"], rows.loc[rows["cell"] == "empty_control"]
        )
        summary = _summary_of(
            SELECTOR,
            rows,
            rules,
            study=_study(),
            columns={
                "collaborative": {"rmse_ratio": ratio},
                "empty_control": {"rmse_ratio": np.nan},
            },
        )
        control = Control(rows, summary, _study(), SELECTOR, "collaborative", "r", rules)
        worst = min(judged.slack for judged in _candidates(control).values())
        assert denominator.min() <= 0.0 <= denominator.max()
        # The denominator is least with its fills at the truth, 0, inside its range.
        at_truth = np.concatenate([denominator, np.zeros(k)])
        tops = _box(rng, numerator, k)[: 2**k]
        slacks = [
            0.5 - float(np.sqrt(np.mean(top**2) / np.mean(bottom**2)))
            for top, bottom in [
                *_pairs(rng, numerator, denominator, k),
                *((top, at_truth) for top in tops),
            ]
        ]
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    @pytest.mark.parametrize("kind", ["tolerance", "below small"])
    def test_s14(self, seed: int, k: int, kind: str) -> None:
        rng = np.random.default_rng(seed)
        small = rng.normal(0.3, 1.0, size=30 - k)
        large = rng.normal(0.1, 0.5, size=30 - k)
        rows = pd.concat(
            [
                _arm(SINGLE, "small", "positive", requested=30, lost=range(k), estimates=small),
                _arm(
                    SINGLE,
                    "large",
                    "positive",
                    requested=30,
                    lost=range(30 - k, 30),
                    estimates=large,
                ),
            ],
            ignore_index=True,
        )
        clause = property_failures.bias_bound_clause(
            "b",
            small="small",
            large="large",
            kind=kind,
            column="bound",
            tolerance=0.5 if kind == "tolerance" else None,
        )
        rules = _one_rule(SINGLE, clause, group="family")
        bound = property_failures.error_bias_bound
        summary = _summary_of(
            SINGLE,
            rows,
            rules,
            study=_study(),
            columns={"small": {"bound": bound(small)}, "large": {"bound": bound(large)}},
        )
        control = Control(rows, summary, _study(), SINGLE, "large", "b", rules)
        worst = min(judged.slack for judged in _candidates(control).values())
        # The small size's bound is least at equal fills, inside its range: pair that too.
        _, centre = property_failures._equal_fill_bound(small, k)
        equal = np.concatenate([small, np.full(k, centre)])
        tops = _box(rng, large, k)[: 2**k]
        slacks = [
            (0.5 if kind == "tolerance" else bound(bottom)) - bound(top)
            for top, bottom in [*_pairs(rng, large, small, k), *((top, equal) for top in tops)]
        ]
        _sound_and_attained(slacks, worst)

    @pytest.mark.parametrize(("seed", "k"), CASES)
    @pytest.mark.parametrize("statistic", ["spread", "bias"])
    def test_s6_point(self, seed: int, k: int, statistic: str) -> None:
        values = _exact_cell(seed, k, shift=0.4)

        def point(option: np.ndarray) -> float:
            if statistic == "spread":
                return float(np.log(np.std(option, ddof=1)))
            return float(np.log(max(abs(np.mean(option)), np.finfo(float).tiny)))

        low, high = _point_extremes(
            [option for _, option in property_failures.rung_options(values, k, statistic, 0.0)],
            point,
        )
        for completed in _box(np.random.default_rng(seed), values, k):
            assert low - 1e-12 <= point(completed) <= high + 1e-12

    @pytest.mark.parametrize(("seed", "k"), CASES)
    def test_s9_point(self, seed: int, k: int) -> None:
        rng = np.random.default_rng(seed)
        numerator = _exact_cell(seed, k)
        denominator = rng.normal(0.0, 1.0, size=30 - k)
        worst = max(
            float(np.std(vertex(numerator, k, j), ddof=1))
            / float(np.std(filled(denominator, k, float(np.mean(denominator))), ddof=1))
            for j in range(k + 1)
        )
        for top, bottom in zip(_box(rng, numerator, k), _box(rng, denominator, k), strict=True):
            assert float(np.std(top, ddof=1) / np.std(bottom, ddof=1)) <= worst + 1e-12

    @pytest.mark.parametrize(("seed", "k"), CASES)
    def test_s15_point(self, seed: int, k: int) -> None:
        assert not _s15_violations(seed, k, equal_fill=True)

    def test_the_s15_test_fails_without_the_equal_fill(self) -> None:
        """The deliberate mutation of M7: drop the equal fills and a completion beats the set."""
        assert _s15_violations(5, 3, equal_fill=False)


def _s15_violations(seed: int, k: int, *, equal_fill: bool) -> list[float]:
    """Completions whose ``mean / sd`` lies outside the S15 candidates' range, at ``t`` near
    -0.9, where the extreme is interior (H1)."""
    values = np.random.default_rng(seed).normal(size=30 - k)
    values = (values - values.mean()) / values.std(ddof=1) - 0.9
    low, high = _point_extremes(
        [
            option
            for _, option in property_failures.growth_options(values, k, equal_fill=equal_fill)
        ],
        property_failures.standardized_bias,
    )
    return [
        t
        for completed in _box(np.random.default_rng(seed), values, k)
        if not low - 1e-12 <= (t := property_failures.standardized_bias(completed)) <= high + 1e-12
    ]


class TestSummaryRows:
    """A module's ``SUMMARY_ROWS`` puts the accounting's candidates on its summary's scale."""

    @staticmethod
    def _accounting(module: Any) -> pd.DataFrame:
        from tests.studies import ctmle_oat_per_arm_strata_missing_properties as properties
        from tests.studies.evidence.registry import registered

        (study,) = [record for record in registered() if record.slug == properties.STUDY.slug]
        rows = read_rows(study.artifact("property-replicates.parquet"))
        configuration = properties.CONFIGURATION_BY_NAME["missing_outcome_crossfit"]
        names = ";".join(f"{configuration.family}/{cell}" for cell in configuration.cells)
        records = _records(_record_row(names, 7, group="properties/missing_outcome_crossfit"))
        lost = rows["cell"].isin(configuration.cells) & (rows["replicate"] == 7)
        kept = property_failures.stamp(rows.loc[~lost], records)
        summary = properties.summarize_properties(kept)
        accounting = property_failures.property_accounting(
            kept, records, summary, record=study, module=module
        )
        return accounting.loc[
            accounting["cell"].isin(
                ["rr__missing_outcome_crossfit", "or__missing_outcome_crossfit"]
            )
            & (accounting["leg"] == "se ratio in union band")
        ]

    def test_a_ratio_cell_s_se_candidates_read_the_estimate_scale(self) -> None:
        """One lost replicate of 800 moves the SE-ratio slack by order ``1 / R``."""
        from tests.studies import ctmle_oat_per_arm_strata_missing_properties as properties

        legs = self._accounting(properties)
        assert len(legs) == 2
        shift = (legs["complete_case"] - legs["worst_case"]).abs()
        assert (shift < 0.02).all(), legs.to_string()

    def test_without_the_declaration_the_candidates_read_the_log_scale(self) -> None:
        """Mutation control: the candidates then compare ``SE(log psi)`` with the spread of psi."""
        from tests.studies import ctmle_oat_per_arm_strata_missing_properties as properties

        public = {name: value for name, value in vars(properties).items() if name.isupper()}
        public.pop("SUMMARY_ROWS")
        legs = self._accounting(SimpleNamespace(**public))
        shift = (legs["complete_case"] - legs["worst_case"]).abs()
        assert (shift > 0.25).all(), legs.to_string()

    def test_a_module_without_the_declaration_reads_its_rows_unchanged(self) -> None:
        rows = _cell("power", "large")
        assert property_failures.summary_rows(SimpleNamespace(), rows) is rows
        assert property_failures.summary_rows(None, rows) is rows
