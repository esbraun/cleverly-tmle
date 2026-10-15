"""The design of the repeated-outcome SDR study, checked before any run.

Pilots P1 and P3 and the cross-fit and booster arms of P2 are recorded.  The in-sample and
continuous arms of P2 are pending a re-pilot at the doubled size, so the pilot state stays
``"placeholder"``, and these tests state that.
They assert that each declared size equals the declared rule applied to the pilot tables,
that the declared run refuses to start, and that every control the study declares separates from
the truth by a nonzero amount.  They also check the pending lmtp comparator declaration and the
pairing columns that the later change reads.  The framework tests in
``tests/unit/test_method_evidence.py`` recompute committed verdicts when artifacts exist.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.base import clone

from tests.canonical import property_differential
from tests.canonical.sdr_repeated import pilot, regenerate
from tests.studies import canonical_longitudinal_sdr as sdr_pairing
from tests.studies import canonical_sdr_repeated as study
from tests.studies import ltmle_repeated_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies import sdr_repeated_properties as properties
from tests.studies.canonical_ltmle_repeated import DUPLICATES, RULE_LABEL
from tests.studies.evidence import property_failures, property_verdicts, replication
from tests.studies.evidence.registry import registered
from tests.unit.test_method_evidence import UNCOMMITTED_MODULES
from tests.unit.test_study_provenance import _required_from

pytestmark = pytest.mark.filterwarnings("ignore")

#: The fast sample size of a wiring check.
SMALL_N = 1_000


def _gap_in_se(limit: float, truth: float, name: str, n: int) -> float:
    """The distance of a limit from the truth in standard errors of the efficient curve."""
    return (limit - truth) / (properties.EFFICIENCY_SD[name] / np.sqrt(n))


def _truth(name: str) -> float:
    return properties.TRUTH[properties.label_estimand(name)]


# ---------------------------------------------------------------------------- the record


def test_the_record_declares_a_pending_lmtp_comparator() -> None:
    """The record is not in ``registered()`` yet: registration needs the committed artifacts."""
    record = study.STUDY
    assert record.slug == "longitudinal-repeated-sdr"
    assert record.reference is None
    assert record.pending_reference == "lmtp"
    assert record.comparator_state == "pending"
    assert record not in registered()
    # The reason names the change and the queue row that own the runs.
    assert "X32b PR C" in record.pending_reason
    assert "TRACKER Study queue" in record.pending_reason


def test_the_study_seeds_are_unique() -> None:
    seeds = [record.seed for record in registered()]
    assert study.STUDY.seed not in seeds
    resampling = [record.resampling_seed for record in registered()]
    assert study.STUDY.resampling_seed not in resampling


def test_the_study_declares_thirteen_estimands_in_both_scenarios() -> None:
    assert len(study.LABELS) == len(study.ESTIMANDS) == 13
    assert set(study.SCENARIOS) == {"repeated_folds_1", "repeated_folds_5"}
    assert study.SCENARIO_FOLDS == {"repeated_folds_1": 1, "repeated_folds_5": 5}
    assert set(study.STUDY.scenarios) == set(study.SCENARIOS)
    assert all(tuple(cells) == study.ESTIMANDS for cells in study.STUDY.scenarios.values())
    # The rule equals ``always`` at the first node: the two duplicates are not primary cells.
    assert not set(DUPLICATES) & set(study.ESTIMANDS)
    assert set(DUPLICATES) <= set(properties.TRUTH)
    for duplicate, original in zip(
        DUPLICATES,
        ("ey_regimen[always @ t=1]", "ate_regimen[always vs never @ t=1]"),
        strict=True,
    ):
        assert properties.TRUTH[duplicate] == pytest.approx(properties.TRUTH[original], abs=1e-12)


def test_the_study_record_declares_the_cells_each_family_emits() -> None:
    sizes = {family: len(names) for family, names in study.STUDY.property_cells.items()}
    assert sizes == {
        "interval_calibration": 54,
        "small_sample_calibration": 10,
        "double_robustness": 15,
        "correction_necessity": 10,
        "simultaneous_coverage": 2,
        "rule_necessity": 4,
        "repeated_outcome_necessity": 12,
        "outcome_history_necessity": 6,
        "full_history_necessity": 10,
    }
    assert sum(sizes.values()) == 123


def test_the_module_list_holds_every_study_specific_import() -> None:
    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))
    assert "tests/studies/repeated_sizing.py" in study.STUDY.modules


def test_the_driver_declares_the_pending_reference_and_its_files_exist() -> None:
    """The manifest of a pending study records the declared runner bytes."""
    reference = regenerate.REFERENCE
    for path in reference.files(Path(regenerate.__file__).resolve().parent):
        assert path.exists(), path


def test_the_regeneration_driver_skips_the_properties_on_request() -> None:
    from tests.canonical import regenerate as framework

    source = inspect.getsource(framework._arguments)
    assert "--skip-properties" in source


# ---------------------------------------------------------------------------- the truths


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("plan", ["never", "always", law.RULE])
def test_the_sdr_limit_is_the_truth_whatever_the_regression_when_the_mechanisms_are_true(
    plan: str, horizon: int
) -> None:
    truth = law.BINARY.counterfactual(plan, horizon)
    for key in ("baseline", "outcomes", "full"):
        limit = law.BINARY.sdr_limit(plan, horizon, [key] * horizon, ["true"] * horizon)
        assert limit == pytest.approx(truth, abs=1e-10)


def test_the_declared_truth_table_matches_the_law() -> None:
    for name in study.LABELS:
        plan, reference, horizon = properties.label_parts(name)
        value = law.BINARY.counterfactual(plan, horizon)
        if reference is not None:
            value -= law.BINARY.counterfactual(reference, horizon)
        assert _truth(name) == pytest.approx(value, abs=1e-10)
        # The canonical module reads the same truth.
        assert study.TRUTH[properties.label_estimand(name)] == pytest.approx(value, abs=1e-10)


# ---------------------------------------------------------------------------- the controls


def test_the_pairing_plug_in_misses_the_truth_in_eleven_of_thirteen_cells() -> None:
    """Plan L3: a fit that skipped the correction would fail the bias gate in eleven cells."""
    gaps = {
        name: _gap_in_se(
            properties._sdr_value(name, "y_only", "true", plug=True),
            _truth(name),
            name,
            study.PRIMARY_N,
        )
        for name in study.LABELS
    }
    pinned = {
        "never_t1": -0.57,
        "never_t2": -1.24,
        "never_t3": -1.61,
        "always_t1": 0.52,
        "always_t2": 1.00,
        "always_t3": 1.21,
        "always_vs_never_t1": 0.76,
        "always_vs_never_t2": 1.60,
        "always_vs_never_t3": 2.02,
        "rule_vs_never_t2": 0.87,
        "rule_vs_never_t3": 1.02,
        "rule_t2": -0.15,
        "rule_t3": -0.24,
    }
    assert set(gaps) == set(pinned)
    for name, value in pinned.items():
        assert gaps[name] == pytest.approx(value, abs=0.006), name
    # Every gap but the rule's two means exceeds 0.25 EIF-SE, which is the bias margin.
    small = {name for name, gap in gaps.items() if abs(gap) <= 0.25}
    assert small == {"rule_t2", "rule_t3"}


def test_the_both_wrong_limits_are_nonzero_on_both_sides_at_the_declared_size() -> None:
    gaps = {
        name: _gap_in_se(properties.BOTH_WRONG_LIMIT[name], _truth(name), name, properties.N)
        for name in study.GATED_LABELS
    }
    assert gaps["never_t3"] == pytest.approx(-1.82, abs=0.01)
    assert gaps["always_t3"] == pytest.approx(1.03, abs=0.01)
    assert gaps["rule_t3"] == pytest.approx(-0.76, abs=0.01)
    assert gaps["always_vs_never_t3"] == pytest.approx(2.07, abs=0.01)
    assert gaps["rule_vs_never_t3"] == pytest.approx(0.82, abs=0.01)
    assert min(gaps.values()) < -1.7 < 1.9 < max(gaps.values())
    assert min(abs(gap) for gap in gaps.values()) > 0.7


def test_the_uncorrected_plug_in_limit_is_nonzero_and_differs_from_both_wrong_for_the_rule() -> (
    None
):
    gaps = {
        name: _gap_in_se(properties.PLUG_IN_LIMIT[name], _truth(name), name, properties.N)
        for name in study.GATED_LABELS
    }
    assert gaps["never_t3"] == pytest.approx(-1.82, abs=0.01)
    assert gaps["always_t3"] == pytest.approx(1.03, abs=0.01)
    assert gaps["rule_t3"] == pytest.approx(-0.79, abs=0.01)
    assert gaps["always_vs_never_t3"] == pytest.approx(2.07, abs=0.01)
    assert gaps["rule_vs_never_t3"] == pytest.approx(0.80, abs=0.01)
    # A static plan's correction is inert under intercept mechanisms, so the two limits agree.
    # The rule's does not, because its ratio reads ``Y_{t-1}``.  This is the nonzero witness.
    for name in ("never_t3", "always_t3", "always_vs_never_t3"):
        assert properties.PLUG_IN_LIMIT[name] == pytest.approx(
            properties.BOTH_WRONG_LIMIT[name], abs=1e-10
        )
    for name in ("rule_t3", "rule_vs_never_t3"):
        assert abs(properties.PLUG_IN_LIMIT[name] - properties.BOTH_WRONG_LIMIT[name]) > 5e-4


def test_the_markov_limit_separates_from_the_truth_on_both_sides() -> None:
    gaps = {
        name: _gap_in_se(
            properties.MARKOV_LIMIT[name], _truth(name), name, properties.FULL_HISTORY_N
        )
        for name in study.GATED_LABELS
    }
    assert gaps["never_t3"] == pytest.approx(-3.22, abs=0.02)
    assert gaps["always_t3"] == pytest.approx(1.93, abs=0.02)
    assert gaps["rule_t3"] == pytest.approx(-2.14, abs=0.02)
    assert gaps["always_vs_never_t3"] == pytest.approx(3.73, abs=0.02)
    # The smallest N3 gap, and the plan's pinned value.
    assert gaps["rule_vs_never_t3"] == pytest.approx(0.91, abs=0.01)
    assert min(gaps.values()) < 0 < max(gaps.values())


def test_the_absorbing_reading_separates_from_the_truth_at_the_declared_size() -> None:
    gaps = [
        abs(
            _gap_in_se(properties.ABSORBING_LIMIT[name], _truth(name), name, properties.ABSORBING_N)
        )
        for name in properties.ABSORBING_LABELS
    ]
    assert min(gaps) == pytest.approx(11.88, abs=0.02)
    assert max(gaps) == pytest.approx(16.17, abs=0.02)


def test_the_history_free_and_reversed_limits_are_nonzero_on_both_sides() -> None:
    history = {
        name: _gap_in_se(
            properties.HISTORY_FREE_LIMIT[name], _truth(name), name, properties.HISTORY_FREE_N
        )
        for name in properties.HISTORY_LABELS
    }
    assert history["never_t3"] == pytest.approx(-1.42, abs=0.01)
    assert history["always_t3"] == pytest.approx(0.73, abs=0.01)
    assert history["always_vs_never_t3"] == pytest.approx(1.56, abs=0.01)
    # The control fits the reversed rule, so its gap reads in the reversed rule's own spread:
    # -3.58 and -2.35 (plan section 5.3).  In the declared rule's spread, which is the unit of
    # the paired displacement, the same gaps are -3.39 and -2.14.  The plan's pair is the one
    # the discrimination check needs, and the second pair is the displacement's unit.
    for name, own, declared in (("rule_t2", -3.58, -3.39), ("rule_t3", -2.35, -2.14)):
        gap = properties.REVERSED_TRUTH[name] - _truth(name)
        scale = np.sqrt(properties.N)
        assert gap / (properties.REVERSED_EFFICIENCY_SD[name] / scale) == pytest.approx(
            own, abs=0.01
        )
        assert gap / (properties.EFFICIENCY_SD[name] / scale) == pytest.approx(declared, abs=0.01)


def test_the_longhand_uncorrected_plug_in_reads_the_plug_in_limit_on_a_large_sample() -> None:
    frame = properties.sample("in_sample", seed=3, n=300_000)
    for name in study.GATED_LABELS:
        value = properties.uncorrected(frame, name)
        spread = properties.EFFICIENCY_SD[name] / np.sqrt(len(frame))
        assert abs(value - properties.PLUG_IN_LIMIT[name]) < 4.0 * spread
        # Nonzero witness: the plug-in is not the truth, so a function that returned the truth
        # would fail the line above.
        assert abs(value - _truth(name)) > 4.0 * spread


@pytest.mark.parametrize("configuration", ["both_wrong", "markov_history", "mechanism_correct"])
def test_a_large_sample_fit_reaches_its_declared_limit_and_not_the_truth(
    configuration: str,
) -> None:
    frame = properties.sample("in_sample", seed=17, n=120_000)
    result = properties.fit(frame, configuration)
    targets = {
        "both_wrong": properties.BOTH_WRONG_LIMIT,
        "markov_history": properties.MARKOV_LIMIT,
        "mechanism_correct": {name: _truth(name) for name in study.GATED_LABELS},
    }[configuration]
    separated = 0
    for name, limit in targets.items():
        fitted = result[properties.label_estimand(name)]
        assert abs(fitted.psi - limit) < 3.5 * fitted.std_error, name
        separated += int(abs(fitted.psi - _truth(name)) > 3.5 * fitted.std_error)
    if configuration != "mechanism_correct":
        # Nonzero witness: a fit that converged to the truth would pass every line above.
        assert separated >= 1


def test_the_pairing_learner_reads_the_earlier_outcomes_only() -> None:
    assert study.CellMeansOnOutcomes.columns("outcomes", 1) == []
    assert study.CellMeansOnOutcomes.columns("outcomes", 3) == [1]
    assert study.CellMeansOnOutcomes.columns("outcomes", 5) == [1, 3]
    rng = np.random.default_rng(0)
    design = rng.integers(0, 2, size=(400, 5)).astype(float)
    y = (design[:, 1] + design[:, 3] + rng.normal(0, 0.5, 400) > 1.0).astype(float)
    learner = clone(study.CellMeansOnOutcomes(properties.CellMeans())).fit(design, y)
    shifted = design.copy()
    shifted[:, [0, 2, 4]] = 1.0 - shifted[:, [0, 2, 4]]
    assert np.allclose(learner.predict(design), learner.predict(shifted))
    # Nonzero witness: the two kept columns do move the prediction.
    moved = design.copy()
    moved[:, 1] = 1.0 - moved[:, 1]
    assert not np.allclose(learner.predict(design), learner.predict(moved))
    # At the first node the key is empty and every row gets the mean.
    first = clone(study.CellMeansOnOutcomes(properties.CellMeans())).fit(design[:, :1], y)
    assert np.allclose(first.predict(design[:, :1]), np.mean(y))


def test_the_baseline_and_booster_learners_are_the_declared_ones() -> None:
    q, pseudo, _, _ = properties._learners("mechanism_correct")
    assert q.kind == pseudo.kind == "baseline"
    q, _, g, _ = properties._learners("both_wrong")
    assert q.kind == "baseline"
    assert type(g).__name__ == "DummyClassifier"
    q, _, g, _ = properties._learners("outcome_correct")
    assert type(g).__name__ == "BaselineOnly"
    q, _, _, _ = properties._learners("markov_history")
    assert q.kind == "markov"
    q, pseudo, _, _ = properties._learners(properties.BOOSTER_ARM)
    assert type(q).__name__ == "HistGradientBoostingClassifier"
    assert type(pseudo).__name__ == "HistGradientBoostingRegressor"


@pytest.mark.parametrize("horizon", [2, 3])
def test_the_control_fits_report_the_declared_estimand_names(horizon: int) -> None:
    frame = properties.sample("in_sample", seed=5, n=SMALL_N)
    absorbing = properties.absorbing_fit(frame, horizon)
    history_free = properties.history_free_fit(frame, horizon)
    for result, labels in (
        (absorbing, properties.ABSORBING_LABELS),
        (history_free, properties.HISTORY_LABELS),
    ):
        for name in labels:
            if properties.label_parts(name)[2] != horizon and result is absorbing:
                continue
            fitted = result[properties._end_of_study_name(name)]
            assert np.isfinite(fitted.psi)
            assert fitted.std_error > 0.0


def test_the_booster_arm_is_gated_one_sided_and_the_other_arms_are_two_sided() -> None:
    assert properties.ONE_SIDED_ARMS == (properties.BOOSTER_ARM,)
    cells = study.STUDY.property_cells["interval_calibration"]
    assert [c for c in cells if c.endswith(f"__{properties.BOOSTER_ARM}")] == [
        f"{name}__{properties.BOOSTER_ARM}" for name in study.GATED_LABELS
    ]
    # The booster fits at five folds, like the cross-fit arm.
    assert properties.fit_n("interval_calibration", properties.BOOSTER_ARM) == 8_000


def _booster_summary(
    se_low: float, se_high: float, cover_low: float, cover_high: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "property": ["interval_calibration"],
            "cell": [f"never_t3__{properties.BOOSTER_ARM}"],
            "passed": [True],
            "two_sided_passed": [np.nan],
            "se_ratio_ci_lower": [se_low],
            "se_ratio_ci_upper": [se_high],
            "coverage_ci_lower": [cover_low],
            "coverage_ci_upper": [cover_high],
        }
    )


@pytest.mark.parametrize(
    ("interval", "one_sided", "two_sided"),
    [
        # Conservative: above the two-sided band, so only the one-sided rule passes.
        ((1.09, 1.14, 0.965, 0.975), True, False),
        # Inside both bands.
        ((0.99, 1.03, 0.94, 0.96), True, True),
        # The SE ratio's lower bound is below 0.93: anti-conservative.
        ((0.90, 0.95, 0.94, 0.96), False, False),
        # The coverage interval's lower end is below 0.92.
        ((0.99, 1.03, 0.91, 0.94), False, False),
    ],
)
def test_the_one_sided_rule_reads_the_lower_ends_and_reports_the_two_sided_result(
    interval: tuple[float, float, float, float], one_sided: bool, two_sided: bool
) -> None:
    summary = _booster_summary(*interval)
    # ``apply_shared_verdicts`` wrote the two-sided verdict into ``passed``.
    margins = properties.MARGINS
    summary.loc[0, "passed"] = bool(
        interval[0] >= margins.calibration_se_ratio[0]
        and interval[1] <= margins.calibration_se_ratio[1]
        and interval[2] >= margins.calibration_coverage[0]
        and interval[3] <= margins.calibration_coverage[1]
    )
    properties.one_sided_calibration_verdicts(summary)
    assert bool(summary.loc[0, "passed"]) is one_sided
    assert bool(summary.loc[0, "two_sided_passed"]) is two_sided


def test_the_one_sided_rule_leaves_the_other_arms_alone() -> None:
    summary = _booster_summary(1.09, 1.14, 0.965, 0.975)
    summary.loc[0, "cell"] = "never_t3__crossfit"
    summary.loc[0, "passed"] = False
    properties.one_sided_calibration_verdicts(summary)
    assert bool(summary.loc[0, "passed"]) is False
    assert np.isnan(summary.loc[0, "two_sided_passed"])


# ---------------------------------------------------------------------------- the pairing columns


def test_the_pairing_helpers_equal_the_x32a_copies() -> None:
    """The helpers are a copy of ``canonical_longitudinal_sdr``'s, applied to the repeated fit."""
    frame, _ = study.draw_scenario("repeated_folds_5", SMALL_N, 0)
    result = study.fit_cleverly(frame, folds=5)
    for plan in study.plans_of(result):
        ratios = study.supplied_ratios(result, plan)
        np.testing.assert_array_equal(ratios, sdr_pairing.supplied_ratios(result, plan))
        study.check_ratios(result, plan, ratios)
        sdr_pairing.check_ratios(result, plan, ratios)
        # Nonzero witness: a plan has followers, so some ratio is positive.
        assert (ratios > 0.0).any()
    assert study.slug("treat then treat after an outcome") == sdr_pairing.slug(
        "treat then treat after an outcome"
    )


def test_a_wrong_ratio_is_refused() -> None:
    """A mutation control: ratios that are not the weights the engine reads are refused."""
    frame, _ = study.draw_scenario("repeated_folds_1", SMALL_N, 1)
    result = study.fit_cleverly(frame, folds=1)
    plan = study.plans_of(result)[1]
    ratios = study.supplied_ratios(result, plan)
    broken = ratios.copy()
    followers = np.flatnonzero(broken[:, 1] > 0.0)
    broken[followers[0], 1] *= 1.01
    with pytest.raises(AssertionError, match="not the weights"):
        study.check_ratios(result, plan, broken)


def test_the_pairing_columns_hold_the_ratio_and_the_arm_of_every_plan() -> None:
    frame, _ = study.draw_scenario("repeated_folds_1", SMALL_N, 2)
    result = study.fit_cleverly(frame, folds=1)
    columns = study.pairing_columns(result)
    assert len(columns) == 3 * 3 * 2
    for node in (1, 2, 3):
        # A unit that leaves the study carries no arm, and its ratio is zero.
        observed = frame[f"A{node}"].notna().to_numpy()
        never = np.asarray(columns[f"shift__never__{node}__1"], dtype=float)
        always = np.asarray(columns[f"shift__always__{node}__1"], dtype=float)
        assert np.all(never[observed] == 0.0)
        assert np.all(always[observed] == 1.0)
        # Nonzero witness: some unit is censored, so the check above reads a strict subset.
        assert observed.mean() < 1.0 or node == 1
    rule = study.slug(RULE_LABEL)
    seen = frame["A2"].notna().to_numpy()
    arms = np.asarray(columns[f"shift__{rule}__2__1"], dtype=float)
    previous = frame["Y1"].to_numpy()
    assert np.array_equal(arms[seen], (previous[seen] > 0).astype(float))
    # The ratio is the plan's density over the known mechanism at the observed arm.
    observed = frame["A1"].to_numpy() == 1
    assert np.all(columns["ratio__never__1"][observed] == 0.0)
    stayed = frame["C1"].to_numpy() == 1
    assert (columns["ratio__never__1"][~observed & stayed] > 0.0).all()
    assert np.all(columns["ratio__never__1"][~stayed] == 0.0)


def test_the_pairing_columns_do_not_depend_on_the_fold() -> None:
    """The mechanisms are known, so the ratios are fold free and the training-fold recursion
    of the cross-fitted fit reads the same ratios as ``lmtp::cf_sdr``."""
    frame, _ = study.draw_scenario("repeated_folds_5", SMALL_N, 3)
    one = study.pairing_columns(study.fit_cleverly(frame, folds=1))
    five = study.pairing_columns(study.fit_cleverly(frame, folds=5))
    assert set(one) == set(five)
    for name in one:
        np.testing.assert_array_equal(np.asarray(one[name]), np.asarray(five[name]))


def test_the_correction_is_nonzero_at_both_fold_counts() -> None:
    """The pairing learner is wrong, so the estimate differs from the first-node plug-in."""
    frame, _ = study.draw_scenario("repeated_folds_5", 4_000, 4)
    for folds in (1, 5):
        result = study.fit_cleverly(frame, folds=folds)
        initial = study.initials(result)
        moved = [
            abs(float(result[name].psi) - initial[name])
            for name in study.ESTIMANDS
            if name.endswith("@ t=3]")
        ]
        assert min(moved) > 1e-4
        assert max(moved) > 1e-3


# ---------------------------------------------------------------------------- the sizes


def test_the_pilot_tables_are_recorded_except_the_pending_p2_arms() -> None:
    """P1, P3 and the cross-fit and booster arms of P2 are pinned.  Two P2 arms are pending."""
    assert study.PILOT_STATE == "placeholder"
    assert set(properties.CALIBRATION_PILOT_PENDING_ARMS) == {"in_sample", "continuous"}
    # P1 and P3 are recorded: no cell holds the uniform planning value.
    assert len(set(properties.PRIMARY_PILOT_RATIOS.values())) > 20
    assert properties.PRIMARY_PILOT_RATIOS["repeated_folds_5/rule_t3"] == 1.036
    assert properties.PRIMARY_PILOT_COVERED["repeated_folds_1/never_t1"] == 1_901
    assert properties.PRIMARY_PILOT_N == study.PRIMARY_N
    assert len(set(properties.PROPERTY_PILOT_BIASES.values())) > 40
    assert (
        properties.PROPERTY_PILOT_COVERED["full_history_necessity/never_t3__markov_history"] == 221
    )
    # P2: the cross-fit and booster arms are recorded.
    recorded = [
        cell
        for cell in properties.CALIBRATION_CELLS
        if cell.rsplit("__", 1)[1] not in properties.CALIBRATION_PILOT_PENDING_ARMS
        and not cell.endswith("__in_sample")
    ]
    assert len(recorded) == 10
    assert properties.CALIBRATION_PILOT_RATIOS["rule_t3__crossfit"] == 0.9815
    assert properties.CALIBRATION_PILOT_RATIOS["rule_t3__crossfit_boosted"] == 1.0036
    assert properties.CALIBRATION_PILOT_COVERED["rule_t3__crossfit"] == 7_520
    # P2: the pending arms hold uniform planning values, not the 8,000 rows, which no longer
    # describe a declared size.
    pending = [
        cell
        for cell in properties.CALIBRATION_CELLS
        if cell.endswith(("__in_sample", "__continuous"))
    ]
    assert len(pending) == 13 + 5
    assert {properties.CALIBRATION_PILOT_RATIOS[cell] for cell in pending} == {1.0}
    assert {properties.CALIBRATION_PILOT_BIASES[cell] for cell in pending} == {0.0}
    assert {properties.CALIBRATION_PILOT_COVERED[cell] for cell in pending} == {7_600}
    assert properties.CALIBRATION_PILOT_JOINT_COVERAGE == 0.95
    # The pending arms name the size the re-pilot must draw.
    assert properties.CALIBRATION_PILOT_N["in_sample"] == 32_000
    assert properties.CALIBRATION_PILOT_N["continuous"] == 32_000
    assert properties.CALIBRATION_PILOT_N["crossfit"] == 8_000
    assert properties.CALIBRATION_PILOT_N[properties.BOOSTER_ARM] == 8_000


def test_the_in_sample_and_continuous_sizes_are_the_base_doubled_twice() -> None:
    """Pilot P2 at 8,000 showed an in-sample variance shortfall, and the n doubled twice."""
    assert properties.IN_SAMPLE_N_BASE == 8_000
    assert properties.IN_SAMPLE_N_DOUBLINGS == 2
    assert properties.IN_SAMPLE_N == properties.IN_SAMPLE_N_BASE * 2**2 == 32_000
    assert properties.CONTINUOUS_N_BASE == 8_000
    assert properties.CONTINUOUS_N_DOUBLINGS == 2
    assert properties.CONTINUOUS_N == properties.CONTINUOUS_N_BASE * 2**2 == 32_000
    # The cross-fit and booster arms stay at the calibration size.
    assert properties.CALIBRATION_N == 8_000
    assert properties.fit_n("interval_calibration", "in_sample") == 32_000
    assert properties.fit_n("interval_calibration", "continuous") == 32_000
    assert properties.fit_n("interval_calibration", "crossfit") == 8_000
    assert properties.fit_n("interval_calibration", properties.BOOSTER_ARM) == 8_000
    # The C4 joint band reads the C1 fits, and the derived controls read their size.
    joint = [c for c in properties.declared_cells() if c.property == "simultaneous_coverage"]
    assert {c.n for c in joint if c.cell.startswith(properties.JOINT_LABEL)} == {32_000}


def test_each_declared_size_is_the_rule_applied_to_the_placeholder_pilots() -> None:
    derived = properties.declared_sizes()
    assert derived["primary_replicates"] == study.PRIMARY_REPLICATES
    assert derived["calibration_replicates"] == properties.CALIBRATION_REPLICATES
    assert derived["calibration_budget_met"] == properties.CALIBRATION_BUDGET_MET
    assert derived["double_robustness"] == properties.DOUBLE_ROBUST_REPLICATES
    assert derived["correction_necessity"] == properties.CORRECTION_REPLICATES
    assert derived["rule_necessity"] == properties.RULE_REPLICATES
    assert derived["repeated_outcome_necessity"] == properties.ABSORBING_REPLICATES
    assert derived["outcome_history_necessity"] == properties.HISTORY_FREE_REPLICATES
    assert derived["full_history_necessity"] == properties.FULL_HISTORY_REPLICATES
    # No rule fires on placeholder tables, so no size has doubled.
    assert not any(derived[key] for key in ("primary_n_rule", "c6_n_rule", "n1_n_rule"))
    assert properties.BOOSTER_REPLICATES == properties.CALIBRATION_REPLICATES
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES
    assert study.STUDY.n == study.PRIMARY_N
    # Each ``n`` is its base size times two to the recorded number of doublings.  A rule that
    # fires moves the doubling count, so these lines hold at every pilot state.
    assert study.PRIMARY_N == study.PRIMARY_N_BASE * 2**study.PRIMARY_N_DOUBLINGS
    assert properties.CONTINUOUS_N == (
        properties.CONTINUOUS_N_BASE * 2**properties.CONTINUOUS_N_DOUBLINGS
    )
    assert properties.IN_SAMPLE_N == (
        properties.IN_SAMPLE_N_BASE * 2**properties.IN_SAMPLE_N_DOUBLINGS
    )
    assert properties.ABSORBING_N == (
        properties.ABSORBING_N_BASE * 2**properties.ABSORBING_N_DOUBLINGS
    )
    for doublings in (
        study.PRIMARY_N_DOUBLINGS,
        properties.IN_SAMPLE_N_DOUBLINGS,
        properties.CONTINUOUS_N_DOUBLINGS,
        properties.ABSORBING_N_DOUBLINGS,
    ):
        assert doublings in (0, 1, 2)
    # The recorded P1 and P3 pins fire no rule, so those sizes have not doubled.  The pending
    # P2 arms doubled twice on the declared route, from the 8,000-draw pilot.
    assert study.PRIMARY_N_DOUBLINGS == properties.ABSORBING_N_DOUBLINGS == 0
    assert properties.IN_SAMPLE_N_DOUBLINGS == properties.CONTINUOUS_N_DOUBLINGS == 2
    assert properties.CALIBRATION_N == 8_000


def test_the_property_instrument_has_six_families() -> None:
    sizes = properties.property_replicates(
        properties.PROPERTY_PILOT_BIASES, properties.PROPERTY_PILOT_REPLICATES
    )
    assert set(sizes) == set(properties.PROPERTY_FAMILIES)
    assert len(properties.PROPERTY_FAMILIES) == 6
    assert sizing.family_share(6) == pytest.approx(1.0 - 0.049 / 3.0 / 6.0)
    assert sizing.family_share(6) == pytest.approx(0.997278, abs=1e-6)


def test_the_union_bound_allocation_sums_to_the_study_failure_probability() -> None:
    d = 0.049 / 3.0
    assert pytest.approx(d) == sizing.SIZED_FAIL
    assert pytest.approx(0.05) == sizing.DISCRIMINATION_FAIL + 3.0 * sizing.SIZED_FAIL
    assert pytest.approx(0.991833, abs=1e-6) == sizing.PRIMARY_CLAUSE_SHARE
    assert sizing.SDR_CALIBRATION_TERMS == 5
    assert pytest.approx(1.0 - d / 5.0) == properties.C6_SHARE
    # The C6 rule of this study reads the five-term share, which is lower than the IPSI one.
    assert properties.C6_SHARE > sizing.C6_SHARE


def test_the_pilot_rules_applied_to_the_placeholder_tables_give_the_declared_sizes() -> None:
    primary = (
        properties.PRIMARY_PILOT_COVERED,
        properties.PRIMARY_PILOT_BIASES,
        properties.PRIMARY_PILOT_REPLICATES,
    )
    verdicts = properties.apply_pilot_rules(
        primary=primary,
        calibration={
            "ratios": properties.CALIBRATION_PILOT_RATIOS,
            "biases": properties.CALIBRATION_PILOT_BIASES,
            "replicates": properties.CALIBRATION_PILOT_REPLICATES,
            "joint_coverage": properties.CALIBRATION_PILOT_JOINT_COVERAGE,
        },
        properties=(properties.PROPERTY_PILOT_BIASES, properties.PROPERTY_PILOT_REPLICATES),
    )
    assert verdicts["primary_replicates"] == study.PRIMARY_REPLICATES
    assert verdicts["calibration_replicates"] == properties.CALIBRATION_REPLICATES
    assert verdicts["property_replicates"]["double_robustness"] == (
        properties.DOUBLE_ROBUST_REPLICATES
    )
    assert not verdicts["primary_n_rule_fires"]
    assert not verdicts["c6_n_rule_fires"]
    assert not verdicts["n1_n_rule_fires"]


def test_the_calibration_rule_reaches_the_cap_when_the_c6_ratio_is_off() -> None:
    """Plan WA: C6 ratios of 0.95 at 8,000 draws put the rule at the cap, and 0.96 does not."""
    for ratio, expected in ((0.95, (50_000, False)), (0.96, (21_200, True))):
        ratios = {
            cell: ratio if cell.endswith("__continuous") else 1.0
            for cell in properties.CALIBRATION_PILOT_RATIOS
        }
        size = properties.calibration_size(
            ratios,
            properties.CALIBRATION_PILOT_BIASES,
            properties.CALIBRATION_PILOT_REPLICATES,
            properties.CALIBRATION_PILOT_JOINT_COVERAGE,
        )
        assert tuple(size) == expected


def test_the_n_rules_fire_on_a_biased_pilot_and_not_on_an_unbiased_one() -> None:
    biased = dict.fromkeys(properties.PRIMARY_PILOT_BIASES, 0.45)
    assert properties.primary_n_rule_fires(biased, properties.PRIMARY_PILOT_REPLICATES)
    calibration = dict.fromkeys(properties.CALIBRATION_PILOT_BIASES, 0.0)
    assert not properties.c6_rule_fires(calibration, properties.CALIBRATION_PILOT_REPLICATES)
    calibration.update(
        {cell: 0.6 for cell in calibration if cell.endswith("__continuous")},
    )
    assert properties.c6_rule_fires(calibration, properties.CALIBRATION_PILOT_REPLICATES)
    families = dict(properties.PROPERTY_PILOT_BIASES)
    assert not properties.property_n_rule_fires(
        families, properties.PROPERTY_PILOT_REPLICATES, "repeated_outcome_necessity"
    )
    for cell in families:
        if cell.startswith("repeated_outcome_necessity/"):
            families[cell] = 0.5
    assert properties.property_n_rule_fires(
        families, properties.PROPERTY_PILOT_REPLICATES, "repeated_outcome_necessity"
    )


def test_the_primary_size_is_the_smallest_with_a_small_expected_number_of_unseen_cells() -> None:
    base = study.minimum_primary_n(study.PRIMARY_REPLICATES)
    assert base * 2**study.PRIMARY_N_DOUBLINGS == study.PRIMARY_N
    assert base == study.PRIMARY_N_BASE
    # The plan's pins for the smallest cell and the expected counts.
    assert study.cell_masses().min() == pytest.approx(0.0016, abs=1e-4)
    assert study.expected_unseen_cell_fits(2_400, 11_000) == pytest.approx(0.018, abs=5e-4)
    assert study.expected_unseen_cell_fits(2_400, 12_000) == pytest.approx(0.005, abs=5e-4)
    assert study.expected_unseen_cell_fits(3_600, 12_000) == pytest.approx(0.0075, abs=5e-4)
    assert study.expected_unseen_cell_fits(5_200, 12_000) == pytest.approx(0.0108, abs=5e-4)
    assert study.minimum_primary_n(3_600) == 12_000
    assert study.minimum_primary_n(5_200) == 13_000
    # The ``n`` rule may double the size twice, and the count then still holds.
    for n in (24_000, 48_000):
        assert study.expected_unseen_cell_fits(study.PRIMARY_REPLICATES, n) < 1e-6


def _arm_spread(family: str, cell: str, n: int) -> float:
    """The spread of a cell's estimate: the pilot's once recorded, else the law's own."""
    return properties.PROPERTY_PILOT_SPREADS[f"{family}/{cell}"]


def _joint_control_probability() -> float:
    """P(the pointwise joint control's 99% interval lies below the nominal rate).

    The control's exact limit is the joint coverage of the pointwise intervals under the law's
    correlation of the influence curves.  The two duplicate parameters of the fifteen reported
    equal ``always``'s, so the thirteen labels carry the same limit.
    """
    correlation = properties.label_correlations(study.LABELS)
    draws = np.random.default_rng(0).multivariate_normal(
        np.zeros(len(study.LABELS)), correlation, size=200_000
    )
    critical = float(norm.ppf(1.0 - properties.MARGINS.alpha / 2.0))
    limit = float(np.mean(np.all(np.abs(draws) <= critical, axis=1)))
    replicates = properties.CALIBRATION_REPLICATES
    z = float(norm.ppf(0.5 + properties.MARGINS.confidence_level / 2.0))
    se = np.sqrt(limit * (1.0 - limit) / replicates)
    nominal = 1.0 - properties.MARGINS.alpha
    return float(norm.cdf((nominal - limit) / se - z))


def _replicates(family: str) -> int:
    return {
        "double_robustness": properties.DOUBLE_ROBUST_REPLICATES,
        "correction_necessity": properties.CORRECTION_REPLICATES,
        "rule_necessity": properties.RULE_REPLICATES,
        "repeated_outcome_necessity": properties.ABSORBING_REPLICATES,
        "outcome_history_necessity": properties.HISTORY_FREE_REPLICATES,
        "full_history_necessity": properties.FULL_HISTORY_REPLICATES,
    }[family]


def _discrimination_probabilities() -> dict[str, float]:
    """The pass probability of every discrimination check, at the declared sizes.

    The checks are: every control that must miss its truth (26), every paired displacement of a
    necessity family (21), the two derived calibration controls of each of thirteen labels (26)
    and the C4 pointwise control (1).  That is 74.
    """
    margins = properties.MARGINS
    out: dict[str, float] = {}
    for family, (arm, limits) in properties.CONTROL_LIMITS.items():
        positive_arm = properties.property_cells()[family][0].split("__", 1)[1]
        for name, limit in limits.items():
            cell = f"{name}__{arm}"
            n = properties.family_n(family)
            spread = _arm_spread(family, cell, n)
            gap = (limit - _truth(name)) / spread
            out[f"miss/{family}/{cell}"] = sizing.bias_discriminated_probability(
                margins, gap, _replicates(family)
            )
            if family == "double_robustness":
                continue
            # The paired displacement is the mean difference over the positive arm's spread.
            # Its standard deviation is at most the sum of the two spreads over the positive's.
            positive = _arm_spread(family, f"{name}__{positive_arm}", n)
            noise = (spread + positive) / positive
            displaced = (limit - _truth(name)) / positive
            threshold = (
                properties.CORRECTION_DISPLACEMENT
                if family == "correction_necessity"
                else properties.NECESSITY_DISPLACEMENT
            )
            out[f"displacement/{family}/{cell}"] = float(
                norm.cdf(np.sqrt(_replicates(family)) * (abs(displaced) - threshold) / noise)
            )
    for name in study.LABELS:
        ratio = properties.CALIBRATION_PILOT_RATIOS[f"{name}__in_sample"]
        for control, factor in (
            ("shrunken_se_control", properties.SHRUNKEN_SE_FACTOR),
            ("noise_control", 2.0**-0.5),
        ):
            out[f"interval_calibration/{name}__{control}"] = sizing.calibration_control_probability(
                margins, properties.CALIBRATION_REPLICATES, factor * ratio
            )
    out["simultaneous_coverage/pointwise_joint_control"] = _joint_control_probability()
    return out


def test_every_control_is_discriminated_at_the_declared_sizes() -> None:
    """The union bound gives each of 74 checks the pass probability 1 - 0.001 / 74.

    The law gaps stand in for the pilot spreads while ``PILOT_STATE`` is "placeholder".  The
    test reads the pilot spreads and ratios from the recorded tables once the state changes.
    """
    probabilities = _discrimination_probabilities()
    floor = sizing.discrimination_floor(74)
    assert floor == pytest.approx(1.0 - 0.001 / 74)
    kinds = [key.split("/", 1)[0] for key in probabilities]
    assert kinds.count("miss") == 26
    assert kinds.count("displacement") == 21
    assert kinds.count("interval_calibration") == 26
    assert kinds.count("simultaneous_coverage") == 1
    assert len(probabilities) == 74
    below = {key: value for key, value in probabilities.items() if value < floor}
    assert not below, below


def test_the_discrimination_check_fails_a_control_that_sits_on_the_truth() -> None:
    """A mutation control: a gap of zero must not clear the floor, or the check is vacuous."""
    floor = sizing.discrimination_floor(74)
    probability = sizing.bias_discriminated_probability(
        properties.MARGINS, 0.0, properties.DOUBLE_ROBUST_REPLICATES
    )
    assert probability < floor
    assert (
        sizing.calibration_control_probability(
            properties.MARGINS, properties.CALIBRATION_REPLICATES, 1.0
        )
        < floor
    )
    # A gap at the margin is not discriminated either.
    assert (
        sizing.bias_discriminated_probability(
            properties.MARGINS,
            properties.MARGINS.standardized_bias,
            properties.DOUBLE_ROBUST_REPLICATES,
        )
        < floor
    )


def test_each_pilot_table_names_its_size_and_its_replicates() -> None:
    """A recorded table must come from the declared pilot, at the current declared sizes."""
    assert properties.PRIMARY_PILOT_N > 0
    assert set(properties.CALIBRATION_PILOT_N) == set(properties.PILOT_UNITS["P2"])
    assert set(properties.PROPERTY_PILOT_N) == set(properties.PILOT_UNITS["P3"])
    assert set(properties.PILOT_UNITS["P1"]) == set(study.SCENARIOS)
    if study.PILOT_STATE != "recorded":
        assert study.PILOT_STATE == "placeholder"
        return
    assert properties.PRIMARY_PILOT_REPLICATES == 2_000
    assert properties.CALIBRATION_PILOT_REPLICATES == 8_000
    assert properties.PROPERTY_PILOT_REPLICATES == 2_000
    for pilot_name, recorded in (
        ("P1", dict.fromkeys(properties.PILOT_UNITS["P1"], properties.PRIMARY_PILOT_N)),
        ("P2", properties.CALIBRATION_PILOT_N),
        ("P3", properties.PROPERTY_PILOT_N),
    ):
        for unit, size in recorded.items():
            assert size == properties.pilot_declared_n(pilot_name, unit)
            assert size == properties.pilot_recorded_n(pilot_name, unit)


def test_the_pins_cover_every_cell_the_pilots_report() -> None:
    primary = {f"{s}/{n}" for s in study.SCENARIOS for n in study.LABELS}
    assert set(properties.PRIMARY_PILOT_RATIOS) == primary == set(properties.PRIMARY_PILOT_COVERED)
    cells = set(properties.CALIBRATION_CELLS)
    assert set(properties.CALIBRATION_PILOT_COVERED) == cells
    assert set(properties.CALIBRATION_PILOT_SPREADS) == cells
    assert len(cells) == 13 + 5 + 5 + 5
    declared = {key for names in properties.property_cells().values() for key in names}
    assert len(declared) == 57
    for table in (
        properties.PROPERTY_PILOT_BIASES,
        properties.PROPERTY_PILOT_RATIOS,
        properties.PROPERTY_PILOT_COVERED,
        properties.PROPERTY_PILOT_SPREADS,
    ):
        assert set(table) == declared


def test_the_c6_arm_has_its_own_declared_size() -> None:
    assert properties.fit_n("interval_calibration", "continuous") == properties.CONTINUOUS_N
    assert properties.fit_n("interval_calibration", "crossfit") == properties.CALIBRATION_N
    payload_sizes = {
        payload[0][1]: payload[0][5]
        for payload in properties._payloads()
        if payload[0][0] == "interval_calibration"
    }
    assert payload_sizes["continuous"] == properties.CONTINUOUS_N
    assert payload_sizes[properties.BOOSTER_ARM] == properties.CALIBRATION_N


def test_doubling_the_c6_size_leaves_the_other_arms_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(properties, "CONTINUOUS_N", 16_000)
    assert properties.fit_n("interval_calibration", "continuous") == 16_000
    assert properties.fit_n("interval_calibration", "in_sample") == properties.IN_SAMPLE_N
    assert properties.pilot_declared_n("P2", "continuous") == 16_000
    assert properties.pilot_declared_n("P2", "crossfit") == 8_000
    assert properties.pilot_declared_n("P2", properties.BOOSTER_ARM) == 8_000


def test_doubling_the_in_sample_size_leaves_the_other_arms_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(properties, "IN_SAMPLE_N", 16_000)
    assert properties.fit_n("interval_calibration", "in_sample") == 16_000
    assert properties.pilot_declared_n("P2", "in_sample") == 16_000
    assert properties.pilot_declared_n("P2", "crossfit") == 8_000
    assert properties.pilot_declared_n("P2", properties.BOOSTER_ARM) == 8_000
    assert properties.pilot_declared_n("P2", "continuous") == properties.CONTINUOUS_N


def test_doubling_the_n1_size_leaves_the_other_families_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(properties, "ABSORBING_N", 16_000)
    assert properties.family_n("repeated_outcome_necessity") == 16_000
    assert properties.pilot_declared_n("P3", "repeated_outcome_necessity") == 16_000
    assert properties.pilot_declared_n("P3", "full_history_necessity") == 8_000
    assert properties.pilot_declared_n("P3", "outcome_history_necessity") == 2_000


def test_the_declared_primary_run_is_refused_while_the_pilots_are_placeholders() -> None:
    with pytest.raises(study.PlaceholderPilotsError, match="placeholder pilot tables"):
        study.draw_and_fit(replicates=study.PRIMARY_REPLICATES, n=study.PRIMARY_N, n_jobs=1)


def test_the_declared_property_run_is_refused_while_the_pilots_are_placeholders() -> None:
    with pytest.raises(study.PlaceholderPilotsError, match="placeholder pilot tables"):
        properties.generate_property_rows(n_jobs=1)


# ---------------------------------------------------------------------------- the drivers


def test_draw_and_fit_returns_a_phase_result_without_samples() -> None:
    result = study.draw_and_fit(replicates=1, n=SMALL_N, n_jobs=1)
    assert isinstance(result, replication.PhaseResult)
    # PR B commits no samples: the study is pending and PR C re-runs the Python side.
    assert result.samples is None
    rows = result.rows
    assert set(rows["scenario"]) == set(study.SCENARIOS)
    for scenario in study.SCENARIOS:
        assert set(rows.loc[rows["scenario"] == scenario, "estimand"]) == set(study.ESTIMANDS)
    assert result.truths is not None
    assert set(result.truths["estimand"]) == set(study.ESTIMANDS)
    # The initial estimate is the plug-in, and it differs from the estimate.
    assert (rows["initial_estimate"].notna()).all()
    assert (rows["initial_estimate"] != rows["estimate"]).all()


def test_the_pairing_sample_holds_the_fold_and_numeric_pairing_columns() -> None:
    for scenario, folds in study.SCENARIO_FOLDS.items():
        frame, _ = study.draw_scenario(scenario, SMALL_N, 0)
        result = study.fit_cleverly(frame, folds=folds)
        sample = study.pairing_sample(frame, result, scenario, 0)
        assert len(sample) == SMALL_N
        expected = {
            f"{kind}__{plan}__{node}" + ("__1" if kind == "shift" else "")
            for kind in ("ratio", "shift")
            for plan in ("never", "always", study.slug(RULE_LABEL))
            for node in (1, 2, 3)
        }
        assert expected <= set(sample.columns)
        assert {"scenario", "replicate", "fold", "row"} <= set(sample.columns)
        assert sample["fold"].nunique() == folds
        # No object column: a shift is an int8 arm code.
        shifts = sample.filter(like="shift__")
        assert (shifts.dtypes == np.int8).all()
        assert not (sample.dtypes.astype(str) == "object").any()


def test_the_primary_driver_returns_a_small_constant_payload_for_each_replicate() -> None:
    """The declared run holds 4,800 payloads in the parent, so each must stay small.

    A returned sample of 12,000 rows held 12 MB, and 2,400 replicates of it held 29 GB.  The
    payload must not grow with ``n``.
    """
    import pickle

    sizes = {}
    for n in (SMALL_N, 3 * SMALL_N):
        key = ("repeated_folds_5", 0, n)
        payload = study._replicate(key, study._draw(key))
        sizes[n] = len(pickle.dumps(payload))
    assert sizes[SMALL_N] < 60_000
    assert sizes[3 * SMALL_N] < 1.1 * sizes[SMALL_N]
    # 2 scenarios x 5,200 replicates at the largest size the rules allow stay below 1 GB.
    assert sizes[3 * SMALL_N] * 2 * 5_200 < 1e9


def test_a_missing_plan_value_on_an_observed_unit_is_refused() -> None:
    """A mutation control for the shift: NaN is legitimate only after the unit left."""
    frame, _ = study.draw_scenario("repeated_folds_1", SMALL_N, 5)
    result = study.fit_cleverly(frame, folds=1)
    plan = study.plans_of(result)[0]
    plan.values[0, 1] = np.nan  # unit 0 is still observed at node 2 unless it was censored
    stayed = np.asarray(result.data.uncensored, dtype=bool)[:, 0]
    if stayed[0]:
        with pytest.raises(AssertionError, match="still observed"):
            study.shifted_values(result, plan, 2)
    else:  # pragma: no cover - seed dependent
        pytest.skip("unit 0 is censored")


def test_the_fit_sets_emit_exactly_the_declared_cells(
    small_declared: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_sets = tuple((family, configuration, 2) for family, configuration, _ in properties.FIT_SETS)
    keys = properties._payloads(fit_sets)
    outcomes = [properties._replication(payload) for (payload,) in keys]
    # ``fit_properties`` stops the run unless each key's rows name exactly its ``_cells_of``
    # without the derived calibration controls, so the declaration is checked fit set by fit set.
    controls = {name for names in properties.DERIVED_CELLS.values() for name in names}
    for key, emitted_rows in zip(keys, outcomes, strict=True):
        named = {f"{row['property']}/{row['cell']}" for row in emitted_rows}
        cells = properties._cells_of(key)
        assert len(cells) == len(set(cells))
        assert set(cells) - controls == named, key
        assert set(cells) & controls == {
            control for name in named for control in properties.DERIVED_CELLS.get(name, ())
        }
    rows = properties.assemble_property_rows(outcomes)
    # The property loop runs the same keys through ``fit_properties`` with the draw split, and
    # publishes the same rows.
    monkeypatch.setattr(properties, "_payloads", lambda: keys)
    monkeypatch.setattr(properties, "require_recorded_pilots", lambda what: None)
    with replication.recording("subject") as recorded:
        looped = properties.generate_property_rows(n_jobs=1)
    assert recorded.failures is None
    pd.testing.assert_frame_equal(looped, rows)
    emitted: dict[str, set[str]] = {}
    for family, cell in set(zip(rows["property"], rows["cell"], strict=True)):
        emitted.setdefault(family, set()).add(cell)
    assert {family: set(cells) for family, cells in study.STUDY.property_cells.items()} == emitted
    roles = rows.groupby("property")["role"].agg(set)
    assert roles["small_sample_calibration"] == {"diagnostic"}
    assert roles["double_robustness"] == {"positive", "control"}


def test_the_property_verdicts_publish_the_two_sided_column_for_the_booster_cells() -> None:
    """A wiring check on a few replications: the verdict code runs on the declared cells."""
    fit_sets = tuple(
        item
        for item in properties.FIT_SETS
        if item[0] in {"interval_calibration", "simultaneous_coverage"}
    )
    outcomes = [
        properties._replication((family, configuration, replicate, 8, 31 + replicate, SMALL_N))
        for family, configuration, _ in fit_sets
        for replicate in range(8)
    ]
    rows = properties.assemble_property_rows(outcomes)
    summary = properties.summarize_properties(rows)
    assert "two_sided_passed" in summary.columns
    boosted = summary["cell"].str.endswith(f"__{properties.BOOSTER_ARM}")
    assert boosted.sum() == 5
    assert summary.loc[boosted, "two_sided_passed"].notna().all()
    assert summary.loc[~boosted, "two_sided_passed"].isna().all()


# ---------------------------------------------------------------------------- the pilot driver


@pytest.fixture
def small_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare a small size everywhere, so a pilot unit fits in a fraction of a second."""
    for name in (
        "PRIMARY_N",
        "CALIBRATION_N",
        "IN_SAMPLE_N",
        "CONTINUOUS_N",
        "ABSORBING_N",
        "HISTORY_FREE_N",
        "FULL_HISTORY_N",
        "N",
    ):
        monkeypatch.setattr(properties, name, SMALL_N)


def test_the_pilot_task_lists_cover_every_unit() -> None:
    sizes = {unit: properties.pilot_declared_n("P3", unit) for unit in properties.PILOT_UNITS["P3"]}
    tasks = pilot.task_list("P3", 3, sizes)
    assert len(tasks) == 3 * 6
    assert {task[2] for task in tasks} == {
        properties.N,
        properties.ABSORBING_N,
        properties.HISTORY_FREE_N,
        properties.FULL_HISTORY_N,
    }
    assert pilot.resolve_n("P2", ["in_sample", "continuous"], None, {}) == {
        "in_sample": properties.IN_SAMPLE_N,
        "continuous": properties.CONTINUOUS_N,
    }
    assert pilot.resolve_n("P2", ["in_sample", "continuous"], 100, {"continuous": 200}) == {
        "in_sample": 100,
        "continuous": 200,
    }
    assert pilot.DEFAULT_REPLICATES == {"P1": 2_000, "P2": 8_000, "P3": 2_000}


@pytest.mark.usefixtures("small_declared")
def test_the_pilot_driver_writes_parts_records_the_plan_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "p1"
    written = pilot.run("P1", out, jobs=1, chunk=2, replicates=2)
    assert written == 2
    assert len(list(out.glob("part-*.json"))) == 2
    assert not list(out.glob("*.tmp"))
    assert capsys.readouterr().out.count("[P1] part") == 2
    plan = json.loads((out / "plan.json").read_text())
    assert plan["resolved_n"] == dict.fromkeys(study.SCENARIOS, SMALL_N)
    assert set(plan["modules"]) == set(pilot.study_files())
    assert "src/cleverly/longitudinal/sdr.py" in plan["modules"]
    assert plan["head"]
    assert pilot.run("P1", out, jobs=1, chunk=2, replicates=2) == 0
    assert "skipped" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pilot.run("P1", out, jobs=1, chunk=2, replicates=3)
    with pytest.raises(SystemExit):
        pilot.run("P1", out, jobs=1, chunk=2, replicates=2, n=SMALL_N + 1)
    _, by_unit = pilot.load_parts("P1", out)
    assert {unit: len(results) for unit, results in by_unit.items()} == dict.fromkeys(
        study.SCENARIOS, 2
    )
    row = by_unit["repeated_folds_1"][0][0]
    assert row["n"] == SMALL_N


def test_a_part_that_exists_is_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "part-00000000.json"
    assert pilot._write_once(path, [1])
    assert not pilot._write_once(path, [2])
    assert json.loads(path.read_text()) == [1]
    assert not list(tmp_path.glob("*.tmp"))


def test_two_launches_write_their_temporary_files_under_different_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "part-00000000.json"
    names = []
    for pid in (101, 202):
        monkeypatch.setattr(pilot.os, "getpid", lambda pid=pid: pid)
        names.append(path.with_name(f"{path.name}.{pilot.os.getpid()}.tmp").name)
    assert names[0] != names[1]
    assert pilot._write_once(path, [1])


@pytest.mark.usefixtures("small_declared")
def test_the_pilot_driver_runs_the_tail_down_to_an_index(tmp_path: Path) -> None:
    out = tmp_path / "p3"
    pilot.run("P3", out, jobs=1, chunk=1, replicates=1, start=2, descending=True)
    assert [p.name for p in sorted(out.glob("part-*.json"))] == [
        f"part-{index:08d}.json" for index in range(2, 6)
    ]
    with pytest.raises(SystemExit, match="missing"):
        pilot.load_parts("P3", out)


@pytest.mark.usefixtures("small_declared")
def test_a_unit_drawn_at_an_old_size_is_refused_until_it_is_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The N1 path: the n rule doubles ABSORBING_N, only N1 is piloted again, and the
    assembly reads the old directory for the other families."""
    first = tmp_path / "p3"
    pilot.run("P3", first, jobs=1, chunk=6, replicates=2)
    chosen, sizes = pilot.collect("P3", [first])
    assert sizes["repeated_outcome_necessity"] == SMALL_N
    monkeypatch.setattr(properties, "ABSORBING_N", 2 * SMALL_N)
    with pytest.raises(SystemExit, match="no directory supplies"):
        pilot.collect("P3", [first])
    assert "skipped repeated_outcome_necessity" in capsys.readouterr().out
    second = tmp_path / "p3-n1"
    pilot.run("P3", second, jobs=1, chunk=6, replicates=2, units=["repeated_outcome_necessity"])
    chosen, sizes = pilot.collect("P3", [first, second])
    assert sizes == {
        "double_robustness": SMALL_N,
        "correction_necessity": SMALL_N,
        "rule_necessity": SMALL_N,
        "repeated_outcome_necessity": 2 * SMALL_N,
        "outcome_history_necessity": SMALL_N,
        "full_history_necessity": SMALL_N,
    }
    rows = pilot._flatten("P3", [r for unit in properties.PILOT_UNITS["P3"] for r in chosen[unit]])
    table = properties.property_cell_table(rows)
    assert set(table.index) == {k for names in properties.property_cells().values() for k in names}
    sizes_found = properties.pilot_row_sizes("P3", rows)
    assert sizes_found["repeated_outcome_necessity"] == {2 * SMALL_N}
    # The directory order does not matter for a unit that only one directory supplies.
    assert pilot.collect("P3", [second, first])[1] == sizes


@pytest.mark.usefixtures("small_declared")
def test_the_p2_rerun_of_two_arms_combines_with_the_old_crossfit_and_booster_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The in-sample and continuous arms were drawn at the old size and are refused."""
    first = tmp_path / "p2"
    pilot.run("P2", first, jobs=1, chunk=8, replicates=1)
    for name in ("IN_SAMPLE_N", "CONTINUOUS_N"):
        monkeypatch.setattr(properties, name, 2 * SMALL_N)
    with pytest.raises(SystemExit, match="no directory supplies"):
        pilot.collect("P2", [first])
    out = capsys.readouterr().out
    assert "skipped in_sample" in out
    assert "skipped continuous" in out
    assert "skipped crossfit " not in out
    second = tmp_path / "p2-doubled"
    pilot.run("P2", second, jobs=1, chunk=8, replicates=1, units=["in_sample", "continuous"])
    _, sizes = pilot.collect("P2", [first, second])
    assert sizes == {
        "in_sample": 2 * SMALL_N,
        "crossfit": SMALL_N,
        "continuous": 2 * SMALL_N,
        properties.BOOSTER_ARM: SMALL_N,
    }


@pytest.mark.usefixtures("small_declared")
def test_rows_at_the_wrong_size_stop_the_assembly(tmp_path: Path) -> None:
    out = tmp_path / "p1"
    pilot.run("P1", out, jobs=1, chunk=2, replicates=1)
    for part in out.glob("part-*.json"):
        results = json.loads(part.read_text())
        for result in results:
            for row in result:
                row["n"] = SMALL_N + 5
        part.write_text(json.dumps(results))
    with pytest.raises(SystemExit, match="the rows of"):
        pilot.collect("P1", [out])


@pytest.mark.usefixtures("small_declared")
def test_a_plan_without_resolved_sizes_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "p1"
    pilot.run("P1", out, jobs=1, chunk=2, replicates=1)
    legacy = {"pilot": "P1", "replicates": 1, "n": None, "chunk": 2, "tasks": 2}
    (out / "plan.json").write_text(json.dumps(legacy))
    with pytest.raises(SystemExit, match="resolved sample sizes"):
        pilot.load_parts("P1", out)


@pytest.mark.usefixtures("small_declared")
@pytest.mark.usefixtures("small_declared")
def test_assemble_warns_loudly_when_the_recorded_hashes_differ_from_the_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "p1"
    pilot.run("P1", out, jobs=1, chunk=2, replicates=1)
    capsys.readouterr()
    pilot.collect("P1", [out])
    assert "WARNING" not in capsys.readouterr().out
    plan = json.loads((out / "plan.json").read_text())
    plan["modules"]["src/cleverly/longitudinal/sdr.py"] = "0" * 64
    plan["modules"]["src/cleverly/longitudinal/removed.py"] = "1" * 64
    plan["head"] = "0" * 40
    (out / "plan.json").write_text(json.dumps(plan))
    pilot.collect("P1", [out])
    text = capsys.readouterr().out
    assert text.count("WARNING") >= 2
    assert "differs: src/cleverly/longitudinal/sdr.py" in text
    assert "differs: src/cleverly/longitudinal/removed.py" in text
    assert "0" * 40 in text


def test_the_p2_arms_run_one_at_a_time(tmp_path: Path) -> None:
    out = tmp_path / "p2-booster"
    pilot.run("P2", out, jobs=1, chunk=2, replicates=2, units=[properties.BOOSTER_ARM])
    plan = json.loads((out / "plan.json").read_text())
    assert plan["units"] == [properties.BOOSTER_ARM]
    _, by_unit = pilot.load_parts("P2", out)
    cells = {row["cell"] for row in by_unit[properties.BOOSTER_ARM][0]}
    assert cells == {f"{name}__{properties.BOOSTER_ARM}" for name in study.GATED_LABELS}


@pytest.mark.usefixtures("small_declared")
def test_the_in_sample_pilot_arm_carries_the_joint_band(tmp_path: Path) -> None:
    out = tmp_path / "p2-in-sample"
    pilot.run("P2", out, jobs=1, chunk=2, replicates=2, units=["in_sample"])
    _, by_unit = pilot.load_parts("P2", out)
    cells = {row["cell"] for row in by_unit["in_sample"][0]}
    assert f"{properties.JOINT_LABEL}__simultaneous_band" in cells
    assert f"{properties.JOINT_LABEL}__pointwise_joint_control" in cells
    assert len([cell for cell in cells if cell.endswith("__in_sample")]) == 13


def test_the_pilot_summaries_read_synthetic_rows() -> None:
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "replicate": r,
                "label": name,
                "estimate": _truth(name) + rng.normal(0, 0.01),
                "std_error": 0.01,
                "covered": int(rng.random() < 0.95),
                "n": 1_000,
            }
            for scenario in study.SCENARIOS
            for name in study.LABELS
            for r in range(40)
        ]
    )
    covered, biases, replicates = properties.summarize_primary_pilot(rows)
    assert replicates == 40
    assert set(covered) == {f"{s}/{n}" for s in study.SCENARIOS for n in study.LABELS}
    assert max(abs(value) for value in biases.values()) > 0.0
    table = properties.primary_cell_table(rows)
    assert list(table.columns) == ["draws", "covered", "ratio", "bias", "spread"]
    assert (table["ratio"] > 0.5).all()
    assert properties.pilot_row_sizes("P1", rows) == {
        scenario: {1_000} for scenario in study.SCENARIOS
    }


# ---------------------------------------------------------------------------- the failure contract
#
# RM39b2 (#304) checks the contract of a registered study: the bound, the declared rules and the
# clause-binding witnesses.  This study is not registered yet, so those checks skip it.  The tests
# below hold it to the same contract until the registering change.


def test_the_property_module_declares_the_failure_bound_and_the_witness() -> None:
    assert properties.FAILURE_BOUND == 0.01
    assert 0.0 <= properties.FAILURE_BOUND <= property_failures.MAXIMUM_BOUND
    assert isinstance(properties.PROPERTY_WITNESS, replication.Witness)


def test_the_record_declares_the_primary_bound_of_each_side() -> None:
    sides = {
        study.STUDY.implementation,
        *filter(None, [study.STUDY.reference, study.STUDY.pending_reference]),
    }
    assert dict(study.STUDY.failure_bound) == dict.fromkeys(sides, 0.01)


def test_every_verdict_family_resolves_to_a_declared_rule() -> None:
    families = {
        cell.property
        for cell in properties.declared_cells()
        if cell.role != property_verdicts.DIAGNOSTIC_ROLE
    }
    missing = [f for f in sorted(families) if property_failures.rule_for(f, properties) is None]
    assert missing == []
    assert property_failures.undeclared_verdict_writers(properties) == []
    own = {family for rule in properties.PROPERTY_RULES for family in rule.families}
    assert own <= families


def test_the_necessity_rules_read_the_declarations_the_verdicts_read() -> None:
    for necessity in properties.NECESSITIES:
        rule = property_failures.rule_for(necessity.family, properties)
        assert rule is not None
        clause = rule.clause("least displacement over labels")
        assert clause.statistic == "S12"
        assert rule.group == "family"


def test_the_failure_differential_covers_the_unregistered_module() -> None:
    """Without a committed run, the job reads the module's synthetic rows (``SYNTHETIC_MODULES``)."""
    assert properties.__name__ in property_differential.SYNTHETIC_MODULES
    assert properties.__name__ in UNCOMMITTED_MODULES


def test_each_clause_the_module_uses_has_a_witness_in_the_binding_table() -> None:
    subject = properties.__name__
    judged = property_differential.judge(subject, "none")
    assert judged.wrong.empty, judged.wrong.to_string()
    bound = property_differential._pairs(property_differential.read_bindings(), subject)
    reasoned = property_differential._pairs(property_differential.read_reasons(), subject)
    assert sorted(judged.used - bound - reasoned) == []
    witnesses = property_differential.read_bindings()
    named = witnesses.loc[witnesses["subject"] == subject]
    assert not named.empty
    for _, family, clause, kind in named.itertuples(index=False, name=None):
        assert (family, clause) in property_differential.judge(subject, kind).decisive, (
            family,
            clause,
            kind,
        )
