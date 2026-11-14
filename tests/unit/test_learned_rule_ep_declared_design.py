"""The design of ``learned-rule-ep-declared`` (P1 plan rev 2.2, section 5.2), without a run.

The declared sieve's limit is computed by quadrature on the RM30 grid; the calibration ``R``
needs the 1,400-draw pilots per cell, which have not run.  The harness, the paired limit fit,
the streams, the summary and the cost gate are checked here on small fits (design review
B3, B4, S3, S4, S6, S7).
"""

from __future__ import annotations

import math
from functools import cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import cleverly.estimators.ep as ep_module
from tests.studies import _learned_rule_law as base
from tests.studies import learned_rule_ep_declared as study
from tests.studies import learned_rule_ep_declared_pilot as pilot
from tests.studies import learned_rule_ep_declared_properties as properties
from tests.studies import repeated_sizing
from tests.studies.evidence.property_verdicts import (
    COVERAGE_CONTROLS,
    DIAGNOSTIC_ROLE,
    SE_CONTROL_SIDES,
)


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


@cache
def small_fit() -> tuple[pd.DataFrame, int, study.Measured]:
    """One verified ``nonexc__S`` fit at n = 2,000 on the smoke stream."""
    cell = study.calibration_cell("nonexc", "S")
    frame = base.draw("non_exceptional", 2_000, study.sample_seed(cell, 0, "smoke"))
    seed = study.fold_seed(cell, 0, "smoke")
    return frame, seed, study.measured(frame, "S", "non_exceptional", seed, check=True)


def test_two_gated_arms_and_a_diagnostic_l_optimal_cell() -> None:
    # Design review S4: optimal__L states no verdict; the gated arms are S (3) and L (2).
    assert numbers()["arms"] == {"__S": 3, "__L": 2}
    assert study.DIAGNOSTIC_CELLS == ("optimal__L",)
    assert study.PROPERTY_CELLS["interval_calibration"][-2:] == study.CONTROL_CELLS
    assert properties.replicates_of("optimal__L") == study.DIAGNOSTIC_REPLICATES


def test_every_declared_cell_kind_is_one_the_shared_verdict_reads() -> None:
    """Design review B1: the control names carry the shared verdict's control kinds."""
    for cell in study.CONTROL_CELLS:
        kind = cell.rsplit("__", 1)[-1]
        assert kind in SE_CONTROL_SIDES or kind in COVERAGE_CONTROLS, cell
    for cell in study.POSITIVE_CELLS + study.DIAGNOSTIC_CELLS:
        assert cell.rsplit("__", 1)[-1] in study.CURVES


def test_the_declared_sieve_has_a_fixed_size_and_the_learner_a_cv_size() -> None:
    sieve = study.declared_rule("S")
    learner = study.declared_rule("L")
    assert not sieve.sieve.chosen_by_cv
    assert sieve.sieve.size == study.SIEVE_SIZE == int(study.SIEVE_FAMILY.max_size)
    assert learner.sieve is None
    assert learner.learner.get_params()["max_depth"] == 2


def test_the_sieve_limit_rule_is_near_d0_at_the_optimal_cell() -> None:
    # Plan 5.2: the limit rule differs from d0 by a regret below 0.01 SE at n = 16,000.
    optimal = numbers()["optimal_known_g"]
    se = optimal["efficiency_sd"] / math.sqrt(study.CALIBRATION_N)
    assert 0.0 <= optimal["regret"] < 0.01 * se


def test_each_law_has_one_sieve_threshold_per_level() -> None:
    for law in study.LAWS:
        thresholds = numbers()[law]["thresholds"]
        assert all(len(points) == 1 for points in thresholds.values())
        assert numbers()[law]["regret"] >= 0.0


def test_the_primary_cells_name_their_curve_and_law() -> None:
    assert study.scenario_of("S_nonexc_2k") == ("S", "non_exceptional")
    assert study.scenario_of("L_misspec_2k") == ("L", "misspecified_limit")
    assert set(study.STUDY.scenarios) == set(study.PRIMARY_CELLS)


def test_the_primaries_pass_together_with_probability_090() -> None:
    """Design review S6: four primaries at 2,000 replicates and coverage 0.930."""
    single = repeated_sizing.coverage_floor_pass_probability(
        study.STUDY.margins, study.PRIMARY_REPLICATES, 0.930
    )
    assert single >= 0.98
    assert single ** len(study.PRIMARY_CELLS) >= 0.90


def test_the_l_budget_passes_at_the_plan_ratio() -> None:
    """Design review S3: ``L`` at 7,200 passes a correct interval with SE ratio 0.97."""
    margins = study.STUDY.margins
    assert float(repeated_sizing.calibration_pass(margins, study.BUDGETED_R["L"], 0.97)) >= 0.99
    assert float(repeated_sizing.calibration_pass(margins, study.BUDGETED_R["S"], 1.0)) >= 0.99


def test_each_stream_has_its_own_samples_and_folds_per_cell() -> None:
    """Design review S2: the pilot and smoke draw no declared seed, and cells share none."""
    seeds: dict[int, tuple[str, ...]] = {}
    for stream in study.STREAMS:
        for cell in study.PROPERTY_CELLS["interval_calibration"]:
            for replicate in range(3):
                for kind, seed in (
                    ("sample", study.sample_seed(cell, replicate, stream)),
                    ("fold", study.fold_seed(cell, replicate, stream)),
                ):
                    assert seed not in seeds, (stream, cell, replicate, kind, seeds.get(seed))
                    seeds[seed] = (stream, cell, str(replicate), kind)


@pytest.mark.xdist_group("learned_rule_ep_declared_fit")
def test_the_harness_integrates_the_fit_own_rules_and_matches_the_refit() -> None:
    """Design review B4: the recorded rules are the refitted rules, row for row."""
    _, _, fitted = small_fit()
    assert fitted.verified
    assert fitted.truth.rows_checked == fitted.result.data.n
    assert len(fitted.truth.fold_values) == base.N_FOLDS
    assert np.isfinite(fitted.truth.truth) and fitted.truth.oracle_se > 0.0


@pytest.mark.xdist_group("learned_rule_ep_declared_fit")
def test_a_perturbed_recorded_rule_stops_the_harness() -> None:
    """Design review B4: the witness that the comparison sees a wrong fold rule."""
    frame, seed, _ = small_fit()
    with study.recorded_fold_rules() as records:
        result = base.fit(
            frame, base.outcome_learner(), seed, rule=study.declared_rule("S", random_state=seed)
        )
    rules = [study._treat(records[fold]) for fold in sorted(records)]
    study.verify(result, rules)
    flipped = list(rules)
    flipped[3] = lambda covariates, rule=rules[3]: ~rule(covariates)
    with pytest.raises(base.HarnessMismatch, match="fold 3"):
        study.verify(result, flipped)
    with pytest.raises(base.HarnessMismatch, match="fold 3"):
        study.integrate(result, flipped, "non_exceptional")


@pytest.mark.xdist_group("learned_rule_ep_declared_fit")
def test_the_paired_limit_fit_shares_the_folds_and_is_finite() -> None:
    """Design review B3: the ``S`` cell's limit-rule fit, which the S red reading reads."""
    frame, seed, fitted = small_fit()
    paired = study.paired_fit(frame, "non_exceptional", seed, fitted.result, optimal=False)
    assert np.isfinite(paired.estimate) and paired.std_error > 0.0
    assert paired.truth == study.limit_value("non_exceptional")[0]


def test_unit_weights_change_no_histogram_booster_fit() -> None:
    """Design review B4: dropping unit weights leaves the declared learner's fit unchanged."""
    rng = np.random.default_rng(7)
    features = pd.DataFrame({"W1": rng.uniform(-1, 1, 4_000), "W2": rng.integers(0, 2, 4_000)})
    target = np.sin(3 * features["W1"]) + 0.5 * features["W2"] + rng.normal(0, 1, 4_000)
    learner = study.declared_rule("L").learner
    assert isinstance(learner, HistGradientBoostingRegressor)
    plain = HistGradientBoostingRegressor(**learner.get_params()).fit(features, target)
    unit = HistGradientBoostingRegressor(**learner.get_params()).fit(
        features, target, sample_weight=np.ones(len(target))
    )
    assert np.array_equal(plain.predict(features), unit.predict(features))


def test_the_final_stage_passes_no_weights_only_when_they_are_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def spy(learner: Any, features: Any, target: Any, weights: Any) -> Any:
        seen.append(weights)
        return learner

    monkeypatch.setattr(ep_module, "fit_learner", spy)
    debiased = SimpleNamespace(contrast=np.linspace(-1.0, 1.0, 10))
    values = {"W1": np.linspace(0.0, 1.0, 10)}
    learner = study.declared_rule("L").learner
    for weights in (np.ones(10), np.linspace(0.5, 1.5, 10)):
        ep_module.final_stage(
            debiased,  # type: ignore[arg-type]
            np.ones((10, 1)),
            weights,
            "difference",
            learner=learner,
            values=values,
        )
    assert seen[0] is None
    assert np.array_equal(seen[1], np.linspace(0.5, 1.5, 10))


def synthetic_rows(seed: int = 0, replicates: int = 60) -> pd.DataFrame:
    """Rows of every declared cell with the published columns, drawn at a correct interval."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell in study.PROPERTY_CELLS["interval_calibration"]:
        if cell in study.CONTROL_CELLS:
            role = "control"
        elif cell in study.DIAGNOSTIC_CELLS:
            role = DIAGNOSTIC_ROLE
        else:
            role = "positive"
        error = rng.normal(0.0, 1.0, replicates)
        limit = rng.normal(0.0, 1.0, replicates)
        # The shrunken control and the diagnostic cell would both be red if read as positives.
        se = 0.7 if cell in (study.SHRUNKEN_CONTROL, *study.DIAGNOSTIC_CELLS) else 1.0
        has_limit = cell.endswith("__S") or cell == study.SHRUNKEN_CONTROL
        rows += [
            {
                "property": "interval_calibration",
                "cell": cell,
                "role": role,
                "replicate": r,
                "n": study.CALIBRATION_N,
                "requested_replicates": replicates,
                "failed_replicates": 0,
                "truth": 0.0,
                "estimate": float(error[r]),
                "std_error": se,
                "covered": int(abs(error[r]) <= 1.96 * se),
                "rejected": int(abs(error[r]) > 1.96 * se),
                "oracle_se": 1.0,
                "rule_rows_checked": study.CALIBRATION_N,
                "solver_warnings": 0,
                "regret": 0.01,
                "harness_verified": int(r % study.VERIFY_EVERY == 0),
                "limit_estimate": float(limit[r]) if has_limit else np.nan,
                "limit_std_error": 1.0 if has_limit else np.nan,
                "limit_truth": 0.0 if has_limit else np.nan,
            }
            for r in range(replicates)
        ]
    return pd.DataFrame(rows)


def test_the_summary_reads_every_cell_and_its_readings() -> None:
    """Design review B1: the declared summary on rows of every cell, readings included."""
    summary = properties.summarize_properties(synthetic_rows()).set_index("cell")
    assert set(summary.index) == set(study.PROPERTY_CELLS["interval_calibration"])
    diagnostic = summary.loc["optimal__L"]
    assert bool(diagnostic["passed"])
    assert np.isfinite(diagnostic["se_ratio_ci_lower"]) and np.isfinite(diagnostic["regret_mean"])
    for cell in ("nonexc__S", "misspec__S", "optimal__S"):
        assert np.isfinite(summary.loc[cell, "limit_se_ratio"]), cell
    for cell in ("nonexc__S", "misspec__S", "nonexc__L", "misspec__L"):
        assert np.isfinite(summary.loc[cell, "oracle_se_ratio"]), cell
    assert np.isnan(summary.loc["optimal__S", "oracle_se_ratio"])
    assert np.isnan(summary.loc["nonexc__L", "limit_se_ratio"])


def test_the_smoke_cost_stays_inside_the_cost_gate() -> None:
    """Design review S7: measured seconds, and the smoke read against 1.5 times them."""
    timings = pd.read_csv(study.SMOKE_TIMINGS)
    assert set(timings["group"]) == set(study.PROPERTY_CELLS["interval_calibration"]) - {
        study.SHRUNKEN_CONTROL
    }
    for row in timings.itertuples(index=False):
        assert row.seconds_per_fit <= study.COST_GATE * study.seconds_of(row.group), row.group
    assert numbers()["cpu_hours"] == pytest.approx(study.run_cpu_hours())


def test_the_declaration_names_every_file_a_run_writes() -> None:
    """Design review S7: the run's files are declared in code, not only in the plan."""
    assert set(study.RUN_FILES).isdisjoint(study.CONDITIONAL_RUN_FILES)
    assert {"properties.csv", "property-replicates.parquet", "manifest.json"} <= set(
        study.RUN_FILES
    )
    assert "tests/studies/learned_rule_ep_declared_properties.py" in study.STUDY.modules


def test_the_pilot_rule_sizes_each_arm_and_the_budget_covers_it() -> None:
    """Design review S2: each arm's R comes from its pilot through X32a's rule."""
    flat = pilot.summarize(synthetic_rows(replicates=200))
    assert set(flat["cell"]) == set(study.POSITIVE_CELLS)
    assert pilot.STREAM == "pilot" and study.verified(study.VERIFY_EVERY + 1, pilot.STREAM)
    flat["se_ratio"] = 1.0
    flat["replicates"] = pilot.REPLICATES
    computed = pilot.declared_replicates(flat)
    assert set(computed) == set(study.CURVES)
    assert all(computed[curve] <= study.BUDGETED_R[curve] for curve in study.CURVES)
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the pilot has not run; the declaration commit adds it (plan 5.2)")
    computed = pilot.declared_replicates(pd.read_csv(pilot.PILOT_CSV))
    assert all(computed[curve] <= study.BUDGETED_R[curve] for curve in study.CURVES)
