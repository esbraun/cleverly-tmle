"""The declaration of ``continuous-survival-variance`` (X6 plan rev 5, section 5.1, "if S")."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.inference import make_estimate
from cleverly.survival import result as survival_result
from tests.studies import continuous_survival_law as base
from tests.studies import continuous_survival_variance as study
from tests.studies import continuous_survival_variance_law as laws
from tests.studies import continuous_survival_variance_pilot as pilot
from tests.studies import continuous_survival_variance_readings as readings
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import replicate_seed

TESTS = Path(__file__).resolve().parents[1]
PILOT = pilot.read()


def test_the_declaration_is_the_plans() -> None:
    assert study.PRIMARY_REPLICATES == study.STUDY.replicates == 3_600
    assert study.REPLICATE_CAP == 7_200
    assert study.PILOT_REPLICATES == 500
    assert study.MARGINS.standardized_bias == 0.2
    assert study.STUDY.reference is None
    assert study.CELLS == {
        "v1_rare_cause2": ("rare_cause2", 1_000, True),
        "v2_heavy_weights": ("heavy_weights", 1_000, True),
        "v3_half_n": ("base", 500, True),
        "v4_in_sample": ("base", 1_000, False),
        "v7_quadratic": ("quadratic", 1_000, True),
    }
    assert study.STUDY.sizes(study.PRIMARY_N)["v3_half_n"] == 500
    assert laws.QUADRATIC == 0.3 and laws.RARE_INTERCEPT == -2.6
    assert laws.HEAVY_OMEGA == (2.0, -1.0, 1.5)
    assert readings.THRESHOLD == 0.125
    assert readings.SHRUNKEN_SE_FACTOR == 0.85


def test_each_cell_gates_twelve_and_reads_six() -> None:
    for cell, (law, _, _) in study.CELLS.items():
        gated, early = laws.gated(law), laws.first_quartile(law)
        assert len(gated) == 12 and len(early) == 6, cell
        assert not set(gated) & set(early)
        assert study.ESTIMANDS[cell] == gated
        assert all(f"t={laws.LAWS[law].times[0]:g}]" in name for name in early)


def test_the_seeds_are_used_nowhere_else_in_tests() -> None:
    mine = {str(study.SEED), str(study.RESAMPLING_SEED)}
    for path in TESTS.rglob("*.py"):
        if path.name in {"continuous_survival_variance.py", Path(__file__).name}:
            continue
        numbers = set(re.findall(r"\d{7,}", path.read_text(encoding="utf-8").replace("_", "")))
        assert not mine & numbers, path


def test_the_base_law_is_the_registered_studys() -> None:
    truth = laws.truth("base")
    for name, value in base.TRUTH.items():
        assert truth[name] == pytest.approx(value, abs=1e-10)


def test_each_changed_law_moves_what_it_declares() -> None:
    reference = laws.truth("base")
    rare, heavy = laws.truth("rare_cause2"), laws.truth("heavy_weights")
    quadratic = laws.truth("quadratic")
    median = base.TIMES[1]
    # The propensity leaves every risk alone and changes the efficient SD.
    for name, value in reference.items():
        assert heavy[name] == pytest.approx(value, abs=1e-12)
    name = base.level_name("control", "cause1", median)
    assert laws.efficiency("heavy_weights")[name] > 1.05 * laws.efficiency("base")[name]
    # The rarer cause 2 and the quadratic cause-1 term move the risks they touch.
    rare_name = base.level_name("treated", "cause2", laws.RARE_TIMES[1])
    assert rare[rare_name] < 0.8 * laws.LAWS["base"].risk(1, 2, laws.RARE_TIMES[1])
    # The quadratic term nearly averages out of the marginal risk (it is centred at E X3^2 = 1),
    # so its witness is the conditional hazard, which no model linear in X3 can hold.
    one = np.array(1.0)
    h1, _ = laws.LAWS["quadratic"].rates(one * 0, one * 0, np.array([0.0, 1.0, 2.0]), 1)
    h0, _ = laws.LAWS["base"].rates(one * 0, one * 0, np.array([0.0, 1.0, 2.0]), 1)
    assert np.log(h1 / h0) == pytest.approx([-0.3, 0.0, 0.9])
    name = base.level_name("treated", "cause1", median)
    assert quadratic[name] != pytest.approx(reference[name], abs=1e-4)


def test_the_rare_cause_times_are_its_event_time_quartiles() -> None:
    quartiles = laws.LAWS["rare_cause2"].event_time_quartiles(n=400_000)
    assert quartiles == pytest.approx(laws.RARE_TIMES, abs=0.02)


def test_the_at_target_count_carries_the_plans_row_pass() -> None:
    row = float(
        sizing.calibration_pass(study.MARGINS, study.PRIMARY_REPLICATES, np.array([1.0]), 0.95)[0]
    )
    assert row == pytest.approx(0.99923, abs=2e-4)
    assert (
        sizing.calibration_control_probability(
            study.MARGINS, study.PRIMARY_REPLICATES, readings.SHRUNKEN_SE_FACTOR
        )
        > 0.999
    )


def test_the_floor_constants_are_pending_until_the_run() -> None:
    committed = study.STUDY.artifacts.joinpath(study.FLOORS_FILE).exists()
    if not committed:
        assert survival_result.MEASURED_REL_SD is None
        assert survival_result.MEASURED_REL_SD_EARLY is None
        return
    values = readings.floor_values(pd.read_csv(study.STUDY.artifact(study.FLOORS_FILE)))
    assert values["r_prim"] == survival_result.MEASURED_REL_SD


# ---------------------------------------------------------------------- the readings


@pytest.mark.parametrize(
    ("passed", "expected"),
    [
        ([True, True, True], 0.3),
        ([True, False, True], 0.1),
        ([False, True, True], 0.0),
    ],
)
def test_the_rel_sd_floor_is_a_monotone_cap(passed: list[bool], expected: float) -> None:
    # Rungs out of order, so the cap must sort them; a red middle rung caps below itself even
    # though the higher rung passes.  The mutation (the largest passing rung) reads 0.3 in the
    # second and third cases.
    rungs = [0.3, 0.2, 0.1]
    assert readings.rel_sd_floor(rungs, passed[::-1]) == expected
    mutation = max((r for r, p in zip(rungs, passed[::-1], strict=True) if p), default=0.0)
    if not all(passed):
        assert mutation != expected


def test_the_shipped_interval_is_the_estimates() -> None:
    # The engine's own construction of a risk and a difference under the switched rule, with
    # the PV above the IC so the reported standard error is the PV's.
    rng = np.random.default_rng(0)
    curve = rng.normal(0.0, 0.5, size=400)
    cases = (
        (0.3, "level", (0.0, 1.0), True),
        (0.02, "level", (0.0, 1.0), True),
        (-0.05, "difference", (-1.0, 1.0), False),
    )
    for psi, scale, bounds, level in cases:
        estimate = make_estimate(
            "x",
            psi,
            curve,
            n=curve.size,
            scale=scale,  # type: ignore[arg-type]
            rule_widened=False,
            bounds=bounds,
            covariance_rule="switched",
            predictable_variance=4.0 * float(np.var(curve, ddof=1)) / curve.size,
        )
        mine = readings.shipped_interval(
            np.array([psi]), np.array([estimate.std_error]), np.array([level])
        )
        assert tuple(mine[0]) == pytest.approx(estimate.ci, abs=1e-12)


def _synthetic(se_scale: float = 1.0, switched_scale: float = 1.0) -> pd.DataFrame:
    """Rows in the study's columns: calibrated IC, S at ``switched_scale`` on the early rows."""
    rng = np.random.default_rng(3)
    records = []
    for cell, (law, n, _) in study.CELLS.items():
        truth = laws.truth(law)
        early = laws.first_quartile(law)
        for replicate in range(300):
            extra: dict[str, float] = {}
            for index, name in enumerate(early):
                sd = 0.01
                value = truth[name] + rng.normal(0.0, sd)
                extra.update(
                    {
                        f"early{index}_estimate": value,
                        f"early{index}_std_error": sd * switched_scale * se_scale,
                        f"early{index}_ic_std_error": sd * se_scale,
                        f"early{index}_pv_std_error": sd * switched_scale,
                        f"early{index}_pv_nosq_std_error": sd,
                        f"early{index}_variance_rel_sd": 0.2,
                        f"early{index}_oracle_std_error": sd,
                    }
                )
            for name in study.ESTIMANDS[cell]:
                sd = 0.02
                value = truth[name] + rng.normal(0.0, sd)
                records.append(
                    {
                        "scenario": cell,
                        "replicate": replicate,
                        "n": n,
                        "estimand": name,
                        "truth": truth[name],
                        "estimate": value,
                        "std_error": sd * se_scale,
                        "ic_std_error": sd * se_scale,
                        "pv_std_error": sd,
                        "pv_nosq_std_error": sd,
                        "variance_rel_sd": 0.05,
                        "oracle_std_error": sd,
                        **extra,
                    }
                )
    return pd.DataFrame(records)


def test_the_readings_run_on_rows_like_the_studys() -> None:
    # Early IC 30% low, S calibrated there: the benefit clause must find a row, the harm clause
    # none, and the controls must behave as declared.
    rows = _synthetic(se_scale=0.7, switched_scale=1.0 / 0.7)
    candidates = readings.candidate_readings(readings.long_rows(rows), replicates=400)
    assert set(candidates["candidate"]) == set(readings.CANDIDATES)
    assert len(candidates) == len(readings.CANDIDATES) * 18 * len(study.CELLS)
    rule = readings.ship_rule(candidates).set_index("candidate")
    assert bool(rule.loc["s", "benefit_pass"])
    assert not bool(rule.loc["s_never", "benefit_pass"])
    assert bool(rule.loc["s_shrunk", "instrument_ok"])
    table = readings.floors(candidates)
    assert set(table["rule"]) == {"primary", "calibration"}
    assert set(table["default"]) == {readings.floor_values(table)["default"]}


def test_the_pilot_draws_from_its_own_stream() -> None:
    for cell in study.CELLS:
        primary = {replicate_seed(study.STUDY, cell, r) for r in range(study.PILOT_REPLICATES)}
        seeds = {pilot.pilot_seed(cell, r) for r in range(study.PILOT_REPLICATES)}
        assert len(seeds) == study.PILOT_REPLICATES
        assert not seeds & primary


def test_a_declared_run_waits_for_the_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.canonical.continuous_survival_variance import regenerate

    monkeypatch.setattr(sys, "argv", ["regenerate", "--output", "/tmp/x"])
    if PILOT.empty:
        with pytest.raises(SystemExit, match="run the pilot first"):
            regenerate.refuse_undeclared()
    else:
        assert regenerate.pilot_refusals() == []
    monkeypatch.setattr(sys, "argv", ["regenerate", "--replicates", "2", "--output", "/tmp/x"])
    regenerate.refuse_undeclared()


def test_the_committed_readings_follow_from_the_committed_rows() -> None:
    # The ship rule and the floors are recomputed from the committed candidate readings, and the
    # variance rows carry the published replicates' estimates and standard errors.
    if not study.STUDY.artifacts.joinpath(study.CANDIDATES_FILE).exists():
        pytest.skip("the study has not run")
    candidates = pd.read_csv(study.STUDY.artifact(study.CANDIDATES_FILE))
    rule = pd.read_csv(study.STUDY.artifact(study.SHIP_FILE))
    pd.testing.assert_frame_equal(
        readings.ship_rule(candidates).fillna(""), rule.fillna(""), check_dtype=False
    )
    floors = pd.read_csv(study.STUDY.artifact(study.FLOORS_FILE))
    pd.testing.assert_frame_equal(readings.floors(candidates), floors, check_dtype=False)
    long = pd.read_csv(study.STUDY.artifact(study.VARIANCE_ROWS_FILE))
    published = pd.read_parquet(study.STUDY.artifact("replicates.parquet"))
    keys = ["scenario", "replicate", "estimand"]
    joined = published.merge(long, on=keys, suffixes=("", "_long"))
    assert len(joined) == len(published)
    assert np.allclose(joined["estimate"], joined["estimate_long"])
    assert np.allclose(joined["std_error"], joined["std_error_long"])
