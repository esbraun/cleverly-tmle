"""RM39 part b2: a recorded property failure costs its cells one replicate, and nothing else."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence import property_failures, property_verdicts
from tests.studies.evidence.failures import (
    BOUND_EXCEEDED,
    FAILURE_COLUMNS,
    NOT_ACCOUNTABLE,
    WITHIN_BOUND,
    WORST_CASE_FAILS,
    vertex,
)
from tests.studies.evidence.inference import student_interval
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
        module=BOUNDED,
        rules=control.rules if rules is None else rules,
        local=frozenset(),
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
        {"efficiency_band": None},
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


# p8-p10: S8, S9 and S10 have no shared family yet, so each runs as a synthetic family whose
# summary row a study would compute with the same helper on its survivors.


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


def _p8_deficit(lose: bool = True) -> Control:
    """S8: a subject 0.05 below its reference; the subject's fills raise its ratio and the
    reference's widest split lowers its own, and the deficit crosses zero."""
    study = _study()
    size = 380 if lose else 400
    lost = _every(20, 20) if lose else ()
    reference = _signs(size)
    reference[0], reference[1] = 1.5, -1.5
    rows = pd.concat(
        [
            _arm(
                "synthetic_deficit",
                "subject",
                "positive",
                requested=400,
                lost=lost,
                estimates=_signs(size),
                std_errors=0.9,
            ),
            _arm(
                "synthetic_deficit",
                "reference",
                "diagnostic",
                requested=400,
                lost=lost,
                estimates=reference,
                std_errors=0.95,
            ),
        ],
        ignore_index=True,
    )
    seed = 7
    interval = se_ratio_deficit_interval(
        rows.loc[rows["cell"] == "subject"],
        rows.loc[rows["cell"] == "reference"],
        replicates=study.margins.bootstrap_replicates,
        confidence_level=study.margins.confidence_level,
        seed=seed,
    )
    clause = property_failures.deficit_clause(
        "deficit below zero",
        cells=("subject", "reference"),
        judge=property_failures.below_judge(lambda ctx: 0.0),
        seed=lambda ctx, row: seed,
        applies=property_failures.roles("positive"),
    )
    summary, rules = _synthetic(
        "synthetic_deficit",
        rows,
        {"se_ratio_deficit_ci_lower": interval.low, "se_ratio_deficit_ci_upper": interval.high},
        clause,
    )
    return Control(rows, summary, study, "synthetic_deficit", "subject", clause.name, rules)


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
    """S10: a share of 0.49 under a 0.5 cap; 20 lost rows at the survivor maximum exceed it."""
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
        column="share",
        summary_column="mean_share",
        side="cap",
        limit=0.5,
        applies=property_failures.roles("positive"),
    )
    summary, rules = _synthetic(
        "synthetic_share", rows, {"mean_share": float(rows["share"].mean())}, clause
    )
    return Control(rows, summary, study, "synthetic_share", "subject", clause.name, rules)


LEG_CONTROLS = {
    "p1": _p1_calibration_coverage,
    "p2": _p2_bias_held_scale,
    "p3": _p3_bias_discriminated,
    "p4": _p4_calibration_se_ratio,
    "p5": _p5_resolve,
    "p6": _p6_rate,
    "p7": _p7_coverage_gain,
    "p7 across keys": functools.partial(_p7_coverage_gain, across_keys=True),
    "p8": _p8_deficit,
    "p9": _p9_spread_ratio,
    "p10": _p10_extra_mean,
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
                    control.summary, _accounted(name), rules=control.rules
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


class TestNotAccountable:
    """p11: a family no shared rule computes reads ``failures not accountable``."""

    @staticmethod
    def _summary(family: str, cell: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = _arm(family, cell, "positive", requested=400, lost=(3,), estimates=_signs(399))
        summary = pd.DataFrame(
            [
                {
                    "property": family,
                    "cell": cell,
                    "role": "positive",
                    "replicates": 399,
                    "failed_replicates": 1,
                    "passed": True,
                    "property_passed": True,
                },
                {
                    "property": family,
                    "cell": "untouched",
                    "role": "positive",
                    "replicates": 400,
                    "failed_replicates": 0,
                    "passed": True,
                    "property_passed": True,
                },
            ]
        )
        return rows, summary

    @pytest.mark.parametrize(
        ("family", "local"),
        [
            ("targeting_necessity", frozenset()),
            ("double_robustness", frozenset({"double_robustness"})),
        ],
    )
    def test_a_cell_that_passes_its_complete_case_is_red(
        self, family: str, local: frozenset[str]
    ) -> None:
        rows, summary = self._summary(family, "subject")
        records = _gap_records(rows)
        accounting = property_failures.property_accounting(
            rows, records, summary, record=_study(), module=BOUNDED, local=local
        )
        assert set(accounting["reason"]) == {NOT_ACCOUNTABLE}
        assert list(accounting["leg"]) == [
            property_failures.BOUND_LEG,
            property_failures.NOT_ACCOUNTABLE_LEG,
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
        return property_failures.differential(rules, summary, context)

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

    def test_a_study_local_family_is_not_compared(self) -> None:
        rule = _rule("root_n_and_efficiency")
        rules = self._mutated("root_n_and_efficiency", rule.clauses[:1])
        study, rows = self._ladder()
        moved = property_failures.perturbed(rows, "se_high")
        summary = _shared(moved, study)
        context = property_failures.Context(study, moved, summary)
        assert not property_failures.differential(rules, summary, context).empty
        assert property_failures.differential(
            rules, summary, context, local=frozenset({"root_n_and_efficiency"})
        ).empty
