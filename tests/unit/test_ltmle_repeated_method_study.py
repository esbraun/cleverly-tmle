"""The design of the repeated-outcome longitudinal study, checked before any run.

Each number the study declares before its run is derived here from the exact law in
:mod:`tests.studies.ltmle_repeated_law`: the truths, the control limits, the replication
budgets, and the power of every control.  The study is declared and not run, so nothing here
reads a committed artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from cleverly.datasets import REPEATED_RULE_LABEL, make_longitudinal_repeated, repeated_truth
from cleverly.datasets.longitudinal import _REPEATED_C, _REPEATED_C1
from cleverly.longitudinal import LongitudinalData
from tests.studies import canonical_ltmle_repeated as study
from tests.studies import ltmle_repeated_law as law
from tests.studies import ltmle_repeated_properties as properties
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.registry import registered

PLANS = {"never": "never", "always": "always", REPEATED_RULE_LABEL: law.RULE}
CONFIDENCE = study.STUDY.margins.confidence_level
Z = float(norm.ppf(0.5 + CONFIDENCE / 2.0))
MARGIN = study.STUDY.margins.standardized_bias


def _parse(name: str) -> tuple[str, int, str | None]:
    """``(plan, horizon, reference)`` of a fit's estimand name, in the law's plan names."""
    if name.startswith("ate_regimen["):
        comparison, horizon = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        return PLANS[left], int(horizon), PLANS[right]
    plan, horizon = name[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
    return PLANS[plan], int(horizon), None


def _sd(label: str) -> float:
    """The efficiency bound's standard deviation of a property label at the study's ``n``."""
    plan, horizon, reference = _parse(study.label_estimand(label))
    return law.BINARY.eif_sd(plan, horizon, reference=reference) / np.sqrt(properties.N)


def _discriminated(gap: float, scale: float, replicates: int) -> float:
    """P(the control's 99% bias interval clears ``MARGIN`` spreads), at standardized ``gap``."""
    shift = abs(gap) / scale
    return float(norm.cdf(np.sqrt(replicates) * (shift - MARGIN) - Z))


def _equivalent(bias: float, replicates: int) -> float:
    """P(an arm's 99% bias interval lies inside ``MARGIN`` spreads), at standardized ``bias``."""
    root = np.sqrt(replicates)
    return float(norm.cdf(root * (MARGIN - bias) - Z) - norm.cdf(-root * (MARGIN + bias) + Z))


# ---------------------------------------------------------------------------- the record


def test_the_record_is_registered_with_its_own_seeds() -> None:
    record = study.STUDY
    assert record in registered()
    others = [other for other in registered() if other.slug != record.slug]
    taken = {other.seed for other in others} | {other.resampling_seed for other in others}
    assert record.seed not in taken
    assert record.resampling_seed not in taken
    assert record.seed != record.resampling_seed
    assert (record.replicates, record.n) == (1_600, 2_000)
    assert record.publication_policy == "gated"
    assert record.artifacts.name == "ltmle_repeated"


def test_the_primary_study_reports_every_unique_parameter_once() -> None:
    assert len(set(study.ESTIMANDS)) == len(study.ESTIMANDS) == 13
    assert set(study.TRUTH) == set(study.ESTIMANDS)
    assert not set(study.DUPLICATES) & set(study.ESTIMANDS)
    gated = {study.label_estimand(label) for label in study.GATED_LABELS}
    reported = {study.label_estimand(label) for label in study.REPORTED_LABELS}
    assert gated | reported == set(study.ESTIMANDS)
    assert not gated & reported


# ---------------------------------------------------------------------------- the law


@pytest.mark.parametrize("plan", ["never", "always", REPEATED_RULE_LABEL])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_the_truth_agrees_three_ways(plan: str, horizon: int) -> None:
    """The generator's enumeration, the forward composition and the observed g-formula."""
    enumerated = repeated_truth(plan, horizon)
    assert law.BINARY.counterfactual(PLANS[plan], horizon) == pytest.approx(enumerated, abs=1e-12)
    assert law.BINARY.g_formula(PLANS[plan], horizon) == pytest.approx(enumerated, abs=1e-12)


def test_the_reversed_rule_is_identified_and_far_from_the_rule() -> None:
    for horizon in (2, 3):
        reversed_value = law.BINARY.counterfactual("reversed", horizon)
        assert law.BINARY.g_formula("reversed", horizon) == pytest.approx(reversed_value, abs=1e-12)
        assert reversed_value - repeated_truth(REPEATED_RULE_LABEL, horizon) > 0.09


def test_the_continuous_truths_are_identified() -> None:
    for plan in ("never", "always", law.RULE):
        assert law.CONTINUOUS.g_formula(plan, 3) == pytest.approx(
            law.CONTINUOUS.counterfactual(plan, 3), abs=1e-12
        )
    assert properties.CONTINUOUS_TRUTH["always_vs_never_t3"] > 0.05


def test_censoring_reads_the_history_and_never_the_current_outcome() -> None:
    """``C_t`` reads ``L_t``, ``Y_{t-1}`` and ``A_t`` only, and each term is nonzero."""
    assert set(_REPEATED_C1) == {"intercept", "w", "a"}
    assert set(_REPEATED_C) == {"intercept", "l", "y", "a"}
    assert all(value != 0.0 for value in (*_REPEATED_C1.values(), *_REPEATED_C.values()))
    # On the paths, staying at a node is independent of that node's outcome given H_t, A_t.
    paths = law.BINARY.paths
    for time in (2, 3):
        keys = [*law.BINARY.history(time), f"A{time}", f"Y{time}"]
        weights = paths["p"]
        stayed = weights * (paths[f"C{time}"] == 1.0)
        rate = (
            stayed.groupby([paths[k] for k in keys]).sum()
            / weights.groupby([paths[k] for k in keys]).sum()
        )
        by_outcome = rate.unstack(f"Y{time}")
        np.testing.assert_allclose(by_outcome[0.0], by_outcome[1.0], atol=1e-12)


def test_every_plan_keeps_positivity_through_the_last_node() -> None:
    for plan in ("never", "always", law.RULE):
        followed = law.BINARY.followed(plan, 3)
        weights = law.BINARY.cumulative(plan, 3)[followed]
        assert float(np.min(weights)) > 0.01, plan
        assert float(np.sum(law.BINARY.paths["p"].to_numpy()[followed])) > 0.04, plan


def _data(frame: pd.DataFrame) -> LongitudinalData:
    return LongitudinalData.from_frame(frame, **study.COLUMNS)


def test_the_known_mechanisms_read_the_declared_history_design() -> None:
    frame, _ = make_longitudinal_repeated(n=800, seed=11, backend="pandas")
    data = _data(frame)
    treatment = study.KnownRepeatedMechanism("treatment").fit(None, None)
    censoring = study.KnownRepeatedMechanism("censoring").fit(None, None)
    filled = frame.fillna(0.0)
    w = filled["W"].to_numpy()
    for time in (1, 2, 3):
        y_prev = filled[f"Y{time - 1}"].to_numpy() if time > 1 else np.zeros(len(frame))
        a_prev = filled[f"A{time - 1}"].to_numpy() if time > 1 else np.zeros(len(frame))
        l_t = filled[f"L{time}"].to_numpy() if time > 1 else np.zeros(len(frame))
        a_t = filled[f"A{time}"].to_numpy()
        history = data.history_design(time)
        assert history.shape[1] == {1: 1, 2: 4, 3: 7}[time]
        expected_g = np.asarray(law.BINARY.treatment(time, w, l_t, y_prev, a_prev))
        current = data.history_design(time, include_current=True)
        assert current.shape[1] == {1: 2, 2: 5, 3: 8}[time]
        expected_c = np.asarray(law.BINARY.stay(time, w, l_t, y_prev, a_t))
        at_risk = frame[f"A{time}"].notna().to_numpy()
        np.testing.assert_allclose(
            treatment.predict_proba(history)[at_risk, 1], expected_g[at_risk], atol=1e-12
        )
        np.testing.assert_allclose(
            censoring.predict_proba(current)[at_risk, 1], expected_c[at_risk], atol=1e-12
        )


def test_a_column_swap_moves_the_known_mechanisms() -> None:
    """The deliberate mutation: the width check alone cannot see a permuted design."""
    frame, _ = make_longitudinal_repeated(n=800, seed=12, backend="pandas")
    data = _data(frame)
    for kind, include in (("treatment", False), ("censoring", True)):
        model = study.KnownRepeatedMechanism(kind).fit(None, None)
        for time in (2, 3):
            design = data.history_design(time, include_current=include)
            # ``Y_{t-1}`` sits just before ``L_t`` in the covariate history.
            outcome = 2 * (time - 1) - 1
            swapped = design.copy()
            swapped[:, [outcome, outcome + 1]] = swapped[:, [outcome + 1, outcome]]
            moved = np.abs(model.predict_proba(design)[:, 1] - model.predict_proba(swapped)[:, 1])
            assert float(np.max(moved)) > 0.1, (kind, time)


def test_the_rule_duplicates_always_at_the_first_node() -> None:
    assert repeated_truth(REPEATED_RULE_LABEL, 1) == repeated_truth("always", 1)
    frame, _ = make_longitudinal_repeated(n=1_000, seed=13, backend="pandas")
    result = study.fit_cleverly(frame)
    assert set(result.estimates) == set(study.ESTIMANDS) | set(study.DUPLICATES)
    left, right = study.DUPLICATES
    assert float(result[left].psi) == pytest.approx(
        float(result["ey_regimen[always @ t=1]"].psi), abs=1e-12
    )
    assert float(result[right].psi) == pytest.approx(
        float(result["ate_regimen[always vs never @ t=1]"].psi), abs=1e-12
    )


# ---------------------------------------------------------------------------- the controls


def test_each_control_limit_sits_away_from_the_truth() -> None:
    truth = properties.TRUTH
    for label in properties.ABSORBING_LABELS:
        gap = properties.ABSORBING_LIMIT[label] - truth[study.label_estimand(label)]
        assert gap >= 0.05, label
    for label in properties.HISTORY_LABELS:
        gap = properties.HISTORY_FREE_LIMIT[label] - truth[study.label_estimand(label)]
        assert abs(gap) >= 0.02, label
    for label in properties.STATIC_GATED:
        gap = properties.BOTH_WRONG_LIMIT[label] - truth[study.label_estimand(label)]
        assert abs(gap) >= 0.02, label


def test_the_history_free_reading_is_refused_for_a_rule() -> None:
    with pytest.raises(ValueError, match="rule"):
        law.BINARY.g_formula(law.RULE, 3, history="free")


def _control_designs() -> list[tuple[str, str, float, float, int]]:
    """``(family, label, gap, scale, replicates)`` for every control the study gates."""
    truth = properties.TRUTH
    designs = []
    for label in properties.ABSORBING_LABELS:
        gap = properties.ABSORBING_LIMIT[label] - truth[study.label_estimand(label)]
        designs.append(
            ("repeated_outcome_necessity", label, gap, _sd(label), properties.NECESSITY_REPLICATES)
        )
    for label in properties.HISTORY_LABELS:
        gap = properties.HISTORY_FREE_LIMIT[label] - truth[study.label_estimand(label)]
        designs.append(
            ("outcome_history_necessity", label, gap, _sd(label), properties.NECESSITY_REPLICATES)
        )
    for label in properties.STATIC_GATED:
        gap = properties.BOTH_WRONG_LIMIT[label] - truth[study.label_estimand(label)]
        designs.append(
            ("double_robustness", label, gap, _sd(label), properties.DOUBLE_ROBUST_REPLICATES)
        )
    for label in properties.RULE_LABELS:
        horizon = int(label.rsplit("_t", 1)[1])
        gap = law.BINARY.counterfactual("reversed", horizon) - repeated_truth(
            REPEATED_RULE_LABEL, horizon
        )
        designs.append(("rule_necessity", label, gap, _sd(label), properties.NECESSITY_REPLICATES))
    return designs


@pytest.mark.parametrize(
    ("family", "label", "gap", "scale", "replicates"),
    _control_designs(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_control_is_powered(
    family: str, label: str, gap: float, scale: float, replicates: int
) -> None:
    """The control's spread is approximated by the target plan's efficiency bound."""
    assert _discriminated(gap, scale, replicates) >= 0.99
    # The paired displacement, in the positive arm's spread, clears the declared threshold.
    assert abs(gap) / scale >= 2 * properties.NECESSITY_DISPLACEMENT
    # The positive arm, and a control that converges to its own limit, read as equivalent.
    assert _equivalent(0.0, replicates) >= 0.99


def test_the_smallest_control_gap_is_the_one_the_design_reports() -> None:
    shifts = {
        (family, label): abs(gap) / scale for family, label, gap, scale, _ in _control_designs()
    }
    weakest = min(shifts, key=shifts.__getitem__)
    assert weakest == ("double_robustness", "always_t3")
    assert shifts[weakest] == pytest.approx(0.56, abs=0.02)


# ---------------------------------------------------------------------------- the budgets


def test_the_calibration_budget_passes_at_the_piloted_ratio() -> None:
    """The pilot puts ``rule_t3`` at about 0.986 at ``n = 8,000``; 3,200 is sized for it."""
    assert properties.CALIBRATION_N == 8_000
    assert properties.CALIBRATION_REPLICATES == 3_200
    assert properties.calibration_pass_probability(3_200, ratio=0.986) >= 0.96
    assert properties.calibration_pass_probability(2_800, ratio=0.986) < 0.97
    assert properties.calibration_pass_probability(3_200) >= 0.99


def test_the_small_sample_arms_are_reported_not_gated() -> None:
    """The pilot's ``n = 2,000`` in-sample shortfall publishes with no verdict."""
    assert properties.N == 2_000
    assert properties.family_n("small_sample_calibration") == properties.N
    assert properties.family_n("interval_calibration") == properties.CALIBRATION_N
    assert "small_sample_calibration" in properties.REPORTED_CALIBRATION


def test_the_primary_budget_tolerates_a_small_second_order_bias() -> None:
    """A cell whose true standardized bias is 0.15 passes with probability at least 0.90."""
    assert study.PRIMARY_REPLICATES == 1_600
    assert _equivalent(0.15, 1_600) >= 0.90
    assert _equivalent(0.15, 1_200) < 0.90


def test_the_shrunken_error_control_leaves_the_band() -> None:
    low, _ = study.STUDY.margins.calibration_se_ratio
    spread = properties.SHRUNKEN_SE_FACTOR / np.sqrt(2.0 * (properties.CALIBRATION_REPLICATES - 1))
    upper = properties.SHRUNKEN_SE_FACTOR + Z * spread
    assert float(norm.cdf((low - upper) / spread)) >= 0.99


def _joint_draws() -> np.ndarray:
    names = sorted(set(study.ESTIMANDS) | set(study.DUPLICATES))
    curves = []
    for name in names:
        plan, horizon, reference = _parse(name)
        curves.append(law.BINARY.eif(plan, horizon, reference=reference))
    matrix = np.array(curves)
    weights = law.BINARY.paths["p"].to_numpy()
    covariance = (matrix * weights) @ matrix.T
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    rng = np.random.default_rng(2026320199)
    return rng.multivariate_normal(np.zeros(len(names)), correlation, size=200_000, method="eigh")


def test_the_simultaneous_band_and_its_control_are_powered() -> None:
    draws = _joint_draws()
    critical = float(norm.ppf(1.0 - study.STUDY.margins.alpha / 2.0))
    pointwise = float(np.mean(np.all(np.abs(draws) <= critical, axis=1)))
    assert pointwise < 0.85
    replicates = properties.CALIBRATION_REPLICATES
    counts = np.arange(replicates + 1)

    def passing(rule: Any, coverage: float) -> float:
        inside = np.array(
            [
                rule(clopper_pearson(int(count), replicates, confidence_level=CONFIDENCE))
                for count in counts
            ]
        )
        return float(np.sum(binom.pmf(counts[inside], replicates, coverage)))

    low, high = study.STUDY.margins.calibration_coverage
    band = passing(lambda interval: interval.low >= low and interval.high <= high, 0.95)
    control = passing(lambda interval: interval.high < 1.0 - study.STUDY.margins.alpha, pointwise)
    assert band >= 0.99
    assert control >= 0.99


def test_one_replication_of_each_fit_set_emits_exactly_the_declared_cells() -> None:
    fit_sets = tuple((family, configuration, 1) for family, configuration, _ in properties.FIT_SETS)
    rows = pd.DataFrame(
        [
            row
            for (payload,) in properties._payloads(fit_sets)
            for row in properties._replication(payload)
        ]
    )
    controls = properties.calibration_controls(
        rows,
        study.STUDY,
        labels=study.GATED_LABELS,
        efficiency_bounds=properties.EFFICIENCY_SD,
        calibration_n=properties.CALIBRATION_N,
        shrunken_se_factor=properties.SHRUNKEN_SE_FACTOR,
        critical=properties.CRITICAL,
        positive_suffix="estimated_mechanisms",
    )
    rows = pd.concat([rows, controls], ignore_index=True)
    emitted = {
        (str(family), str(cell))
        for family, cell in rows[["property", "cell"]].itertuples(index=False)
    }
    declared = {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }
    assert emitted == declared
    estimates = rows["estimate"].to_numpy(dtype=float)
    assert np.isfinite(estimates).all()
    # Every point estimate is a mean or a contrast of means of a variable in [0, 1].  The
    # noise control adds noise, and a joint-coverage row records an indicator.
    pointwise = ~rows["cell"].str.endswith("noise_control") & (
        rows["property"] != "simultaneous_coverage"
    )
    assert (np.abs(estimates[pointwise.to_numpy()]) <= 1.0).all()


def test_the_declared_truths_are_finite_and_cover_every_label() -> None:
    truths = properties.declared_truths()
    assert set(truths["continuous"]) == set(study.GATED_LABELS)
    assert set(truths["absorbing"]) == set(properties.ABSORBING_LABELS)
    assert set(truths["history_free"]) == set(properties.HISTORY_LABELS)
    assert set(truths["both_wrong"]) == set(properties.STATIC_GATED)
    for table in truths.values():
        assert all(np.isfinite(value) for value in table.values())


def test_the_rule_reads_only_the_earlier_outcome() -> None:
    history = pd.DataFrame({"Y1": [0.0, 1.0, 0.0], "L2": [1.0, 0.0, 0.0]})
    np.testing.assert_array_equal(study.treat_after("Y1")(history), [0, 1, 0])
