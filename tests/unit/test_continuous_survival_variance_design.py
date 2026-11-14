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
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES
    assert study.REPLICATE_CAP == 7_200
    assert study.PILOT_REPLICATES == 1_000
    assert study.EXTENSION_REPLICATES == 2_000
    assert study.MARGINS.standardized_bias == 0.2
    assert study.STUDY.reference is None
    assert study.ALL_CELLS == {
        "v1_rare_cause2": ("rare_cause2", 1_000, True),
        "v2_heavy_weights": ("heavy_weights", 1_000, True),
        "v3_half_n": ("base", 500, True),
        "v4_in_sample": ("base", 1_000, False),
        "v7_quadratic": ("quadratic", 1_000, True),
    }
    assert ("v7_quadratic" in study.CELLS) == study.V7_GATED
    assert set(study.CELLS) <= set(study.ALL_CELLS)
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
    # 3,600 is the at-target count of the 30 first-quartile rows (the family passes with 0.977);
    # one row passes with the plan's 0.99923 and the shrunk-SE control fails with certainty.
    assert pilot.at_target_count() == 3_600
    row = float(sizing.calibration_pass(study.MARGINS, 3_600, np.array([1.0]), 0.95)[0])
    assert row == pytest.approx(0.99923, abs=2e-4)
    assert row**30 == pytest.approx(0.977, abs=1e-3)
    assert row**30 >= sizing.FAMILY_WISE_PASS
    assert (
        sizing.calibration_control_probability(study.MARGINS, 3_600, readings.SHRUNKEN_SE_FACTOR)
        > 0.999
    )


def test_the_count_is_the_shared_rule_on_the_committed_pilot() -> None:
    # Before the pilot is committed the record holds the placeholder, which regenerate.py
    # refuses; once it is committed the count must be the rule's and at most the cap.
    if PILOT.empty:
        assert study.PRIMARY_REPLICATES == study.PENDING_REPLICATES
        return
    count = pilot.primary_count(PILOT)
    assert count == study.PRIMARY_REPLICATES <= study.REPLICATE_CAP
    assert count >= pilot.at_target_count()


def test_the_floor_constants_are_pending_until_the_run() -> None:
    committed = study.STUDY.artifacts.joinpath(study.FLOORS_FILE).exists()
    if not committed:
        assert survival_result.MEASURED_REL_SD is None
        assert survival_result.MEASURED_REL_SD_EARLY is None
        assert survival_result.MEASURED_EVENT_SHARE[0] == 0.4
        assert survival_result.MEASURED_N == 1_000
        return
    values = readings.floor_values(pd.read_csv(study.STUDY.artifact(study.FLOORS_FILE)))
    assert values["r_prim"] == survival_result.MEASURED_REL_SD
    assert values["share_floor"] == survival_result.MEASURED_EVENT_SHARE[0]
    assert values["r_cal"] == survival_result.MEASURED_REL_SD_EARLY
    assert values["measured_n"] == survival_result.MEASURED_N


def test_the_default_variance_is_the_ship_rules() -> None:
    # The engine default and the measured default are "switched" iff the committed ship rule
    # ships S.  Before the run the branch holds "switched", and PR A does not merge until the
    # rule passes (regenerate.py, merge condition).
    import inspect

    from cleverly.survival import ContinuousSurvivalTMLE

    default = inspect.signature(ContinuousSurvivalTMLE).parameters["variance"].default
    assert default == survival_result.MEASURED_DEFAULTS["variance"]
    if not study.STUDY.artifacts.joinpath(study.SHIP_FILE).exists():
        assert default == "switched"
        return
    rule = pd.read_csv(study.STUDY.artifact(study.SHIP_FILE))
    assert (default == "switched") == readings.ships(rule)


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
    assert {"s_shrunk_primary", "ic_sparse_calibration"} <= set(rule.index)
    # The synthetic IC is 30% low at every early row, so no cell's sparsest row passes.
    assert int(rule.loc["ic_sparse_calibration", "harm_rows"]) == 0
    assert candidates[["gain_low", "gain_high"]].notna().all().all()
    pv = candidates.loc[candidates["candidate"].isin(readings.PV_CANDIDATES)]
    assert (pv["matched_low"] <= pv["matched_high"]).all()
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
    # Every instrument check holds; otherwise the run is void (regenerate.py).
    assert readings.instruments_ok(rule), rule.loc[~rule["instrument_ok"].astype(bool)]
    long = pd.read_csv(study.STUDY.artifact(study.VARIANCE_ROWS_FILE))
    # The point columns of the candidate readings follow from the variance rows.
    for (scenario, estimand), group in long.groupby(["scenario", "estimand"]):
        errors = readings.candidate_errors(group)
        estimate = group["estimate"].to_numpy()
        truth = float(group["truth"].iloc[0])
        level = group["level"].to_numpy(dtype=bool)
        chosen = candidates.loc[
            (candidates["scenario"] == scenario) & (candidates["estimand"] == estimand)
        ].set_index("candidate")
        spread = float(np.std(estimate, ddof=1))
        for name, se in errors.items():
            interval = readings.shipped_interval(estimate, se, level)
            coverage = float(np.mean((interval[:, 0] <= truth) & (truth <= interval[:, 1])))
            assert chosen.loc[name, "coverage"] == pytest.approx(coverage, abs=1e-12)
            assert chosen.loc[name, "se_ratio"] == pytest.approx(se.mean() / spread, rel=1e-9)
            k_row = se.mean() / errors["ic"].mean()
            assert chosen.loc[name, "k_row"] == pytest.approx(k_row, rel=1e-9)
    published = pd.read_parquet(study.STUDY.artifact("replicates.parquet"))
    keys = ["scenario", "replicate", "estimand"]
    joined = published.merge(long, on=keys, suffixes=("", "_long"))
    assert len(joined) == len(published)
    assert np.allclose(joined["estimate"], joined["estimate_long"])
    assert np.allclose(joined["std_error"], joined["std_error_long"])


def _pilot_rows(
    *, early_ratio: float = 1.0, failures: int = 0, plug_in_bias: float = 1.0
) -> pd.DataFrame:
    """Pilot rows like the driver's: calibrated and unbiased, with chosen departures."""
    rng = np.random.default_rng(5)
    records = []
    for cell, (law, n, _) in study.ALL_CELLS.items():
        truth = laws.truth(law)
        early = set(laws.first_quartile(law))
        coefficient = laws.LAWS[law].quadratic
        # Each target's errors standardized to mean 0 and SD exactly 0.01, so the pilot's SE
        # ratio is exactly 1 where the SE is 0.01 and no plug-in check fires by noise.
        noise = rng.normal(size=(study.PILOT_REPLICATES, len(truth)))
        noise = (noise - noise.mean(axis=0)) / noise.std(axis=0, ddof=1)
        for replicate in range(study.PILOT_REPLICATES):
            head = {"label": cell, "replicate": replicate, "n": n, "quadratic": coefficient}
            if replicate < failures:
                records.append({**head, "refusal": "DataError"})
                continue
            for column, (name, value) in enumerate(truth.items()):
                sd = 0.01
                error = sd * noise[replicate, column]
                se = sd * (early_ratio if name in early else 1.0)
                records.append(
                    {
                        **head,
                        "estimand": name,
                        "early": name in early,
                        "truth": value,
                        "estimate": value + error,
                        "std_error": se,
                        "ic_std_error": se,
                        "pv_std_error": se,
                        "variance_rel_sd": 0.1,
                        "initial_estimate": value + error + plug_in_bias * sd,
                        "refusal": "",
                        "seconds": 1.0,
                    }
                )
    return pd.DataFrame(records)


def test_the_pilot_readings_run_on_rows_like_the_drivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.canonical.continuous_survival_variance import regenerate

    rows = _pilot_rows()
    count = pilot.primary_count(rows)
    # Clause (a) at the 0.2 margin binds: 6,000 at 60 calibrated rows of 1,000 draws.
    assert count == 6_000
    assert pilot.v7_halving(rows) is False
    assert pilot.v7_witness(rows) > pilot.WITNESS_BIAS
    assert set(pilot.failure_shares(rows).values()) == {0.0}
    assert pilot.extension_cells(rows, count) == ()
    monkeypatch.setattr(pilot, "read", lambda: rows)
    assert regenerate.pilot_refusals() == [
        f"the pilot sets {count} primary replicates and the record declares "
        f"{study.PRIMARY_REPLICATES}"
    ]
    monkeypatch.setattr(study, "PRIMARY_REPLICATES", count)
    assert regenerate.pilot_refusals() == []
    # Each departure is refused by name.
    departures = {
        "extend its pilot to 2000 draws": _pilot_rows(early_ratio=1.06),
        "lost 0.020 of its pilot fits": _pilot_rows(failures=20),
        "set V7_GATED = False": _pilot_rows(plug_in_bias=0.0),
    }
    for message, frame in departures.items():
        monkeypatch.setattr(pilot, "read", lambda frame=frame: frame)
        assert any(message in reason for reason in regenerate.pilot_refusals()), message
    # A V7 pilot drawn at another coefficient is not read as this law's.
    stale = rows.assign(quadratic=np.where(rows["label"] == "v7_quadratic", 0.15, 0.0))
    assert pilot.v7_witness(stale) is None and pilot.v7_halving(stale) is None
