"""The declarations of the two continuous-survival-bootstrap records (X6 plan rev 5, section 5.5)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studies import continuous_survival_bootstrap as study
from tests.studies import continuous_survival_bootstrap_cross_fitted as cross_fitted
from tests.studies import continuous_survival_bootstrap_pilot as pilot
from tests.studies import continuous_survival_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import replicate_seed

TESTS = Path(__file__).resolve().parents[1]
PILOT = pilot.read()


def test_the_declaration_is_the_plans() -> None:
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES
    assert study.N_BOOTSTRAP == 500 and study.PRIMARY_N == 1_000
    assert study.PILOT_REPLICATES == 400 and study.PILOT_BOOTSTRAP == 200
    assert study.REPLICATE_CAP == 7_600 and study.FAILURE_BOUND == 0.01
    assert study.SHRUNKEN_SE_FACTOR == 0.85
    assert study.MARGINS.standardized_bias == 0.2
    assert study.STUDY.reference is None and study.STUDY.property_cells == {}
    assert study.CELLS == {"b1_in_sample": False}
    assert study.STUDY.reported_scenarios == frozenset()
    assert study.STUDY.publication_policy == "gated"


def test_the_cross_fitted_fit_is_a_reported_record_of_its_own() -> None:
    record = cross_fitted.STUDY
    assert cross_fitted.CELLS == {"b2_cross_fitted": True}
    assert record.replicates == cross_fitted.PRIMARY_REPLICATES == 1_000
    assert cross_fitted.PILOT_REPLICATES == 50 and record.n == study.PRIMARY_N
    assert record.slug != study.STUDY.slug and record.artifacts != study.STUDY.artifacts
    assert record.implementation != study.STUDY.implementation
    # Reported now and after any run: every cell reported, the bound published only, and the
    # policy publishes red verdicts rather than refusing on them.
    assert record.reported_scenarios == frozenset(record.scenarios)
    assert record.reported_bounds_gate is False and record.publication_policy == "reporting"
    assert set(record.modules) >= set(study.STUDY.modules)
    assert (record.artifacts / "regenerate.py").exists()
    assert (study.STUDY.artifacts / "regenerate.py").exists()


def test_the_gated_rows_are_the_median_and_upper_quartile() -> None:
    assert law.TIMES == (0.46, 1.12, 2.32)
    expected = {
        *(
            law.level_name(arm, cause, t)
            for arm in law.ARMS
            for cause in law.CAUSES.values()
            for t in law.TIMES
        ),
        *(law.difference_name(cause, t) for cause in law.CAUSES.values() for t in law.TIMES),
    }
    assert set(study.ALL_ROWS) == expected and len(study.ALL_ROWS) == 18
    assert study.ROWS == law.GATED and len(study.ROWS) == 12
    assert f"@ t={law.TIMES[0]:g}]" not in " ".join(study.ROWS)
    assert set(study.EARLY) == expected - set(study.ROWS) and len(study.EARLY) == 6
    assert study.ESTIMANDS == {study.GATED_CELL: study.ROWS}
    assert study.EARLY_FILE in study.STUDY.extra_artifacts
    assert cross_fitted.ESTIMANDS == {cross_fitted.CELL: study.ALL_ROWS}


@pytest.mark.parametrize("module", [study, cross_fitted], ids=["in_sample", "cross_fitted"])
def test_the_seeds_are_used_nowhere_else_in_tests(module: object) -> None:
    mine = {str(module.SEED), str(module.RESAMPLING_SEED)}  # type: ignore[attr-defined]
    own = Path(module.__file__).name  # type: ignore[attr-defined]
    assert len(mine) == 2
    for path in TESTS.rglob("*.py"):
        if path.name in {own, Path(__file__).name}:
            continue
        numbers = set(re.findall(r"\d{7,}", path.read_text(encoding="utf-8").replace("_", "")))
        assert not mine & numbers, path
    assert not {str(study.SEED), str(study.RESAMPLING_SEED)} & {
        str(cross_fitted.SEED),
        str(cross_fitted.RESAMPLING_SEED),
    }


def test_the_streams_are_disjoint() -> None:
    seen: set[int] = set()
    streams = {
        **{
            f"primary/{module.STUDY.slug}/{cell}": [
                replicate_seed(module.STUDY, cell, r) for r in range(module.PRIMARY_REPLICATES)
            ]
            for module in (study, cross_fitted)
            for cell in module.CELLS
        },
        **{
            f"pilot/{kind}": [pilot.pilot_seed(kind, r) for r in range(count)]
            for kind, (_, count) in pilot.KINDS.items()
        },
    }
    for name, seeds in streams.items():
        assert len(set(seeds)) == len(seeds), name
        assert not set(seeds) & seen, name
        seen |= set(seeds)


def test_the_count_is_the_family_wise_rule_at_the_design_point() -> None:
    count = pilot.at_target_count()
    assert count == study.PRIMARY_REPLICATES == 3_200
    one = float(sizing.calibration_pass(study.MARGINS, count, np.array([1.0]), 0.95)[0])
    assert one ** len(study.ROWS) >= sizing.FAMILY_WISE_PASS
    less = float(sizing.calibration_pass(study.MARGINS, count - 400, np.array([1.0]), 0.95)[0])
    assert less ** len(study.ROWS) < sizing.FAMILY_WISE_PASS
    # The 12 gated rows and all 18 reach the same count.
    assert len(study.ROWS) == 12 and pilot.at_target_count(18) == count
    assert sizing.calibration_control_probability(study.MARGINS, count, 0.85) > 0.999


def test_the_committed_pilot_holds_both_cells() -> None:
    assert not PILOT.empty
    assert pilot.draws(PILOT, "in_sample") >= 100
    assert pilot.draws(PILOT, "cross_fitted") >= 1
    table = pilot.coverage_table(PILOT)
    assert set(table["estimand"]) == set(study.ALL_ROWS)
    checks = pilot.calibration_checks(PILOT, study.PRIMARY_REPLICATES)
    assert set(checks["estimand"]) == set(study.ROWS)
    budget = pilot.budget_cpu_hours(PILOT)
    assert set(budget) == {study.GATED_CELL, cross_fitted.CELL}
    # The cross-fitted record runs at 1,000, not the in-sample record's 3,200.
    per_fit = pilot.seconds(PILOT)["cross_fitted_per_fit"]
    assert np.isclose(budget[cross_fitted.CELL], 1_000 * 501 * per_fit / 3_600)


def test_the_failure_reading_follows_the_pilots_lost_fits() -> None:
    allowance = int(study.FAILURE_BOUND * study.PRIMARY_REPLICATES)
    failed, held = pilot.failed_draws(PILOT, "in_sample"), pilot.draws(PILOT, "in_sample")
    probability = pilot.predicted_exceedance(failed, held, study.PRIMARY_REPLICATES, allowance)
    named = any(r.startswith("in_sample lost ") for r in pilot.failure_refusal(PILOT, "in_sample"))
    assert named == (probability > pilot.FAILURE_EXCEEDANCE)
    # The cross-fitted record's bound gates nothing, so its reading never refuses its run.
    assert pilot.failure_exceedance(PILOT, "cross_fitted") is not None
    assert pilot.failure_refusal(PILOT, "cross_fitted") == []
    # Witness: no lost fit in 400 draws proceeds, and 2 in 100 (the committed pilot's
    # in-sample reading, from withheld bootstraps) does not.
    assert pilot.predicted_exceedance(0, 400, 3_200, allowance) < pilot.FAILURE_EXCEEDANCE
    assert pilot.predicted_exceedance(2, 100, 3_200, allowance) > pilot.FAILURE_EXCEEDANCE


def test_a_declared_run_waits_for_the_declared_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.canonical.continuous_survival_bootstrap import regenerate

    monkeypatch.setattr(sys, "argv", ["regenerate", "--output", "/tmp/x"])
    reasons = regenerate.pilot_refusals()
    if pilot.draws(PILOT, "in_sample") < study.PILOT_REPLICATES:
        assert any("short of the declared 400; extend it with --resume" in r for r in reasons)
    with pytest.raises(SystemExit):
        monkeypatch.setattr(pilot, "read", pd.DataFrame)
        regenerate.refuse_undeclared()
    assert regenerate.pilot_refusals() == [
        f"{pilot.OUTPUT.name} is not committed; run the pilot first"
    ]
    monkeypatch.setattr(sys, "argv", ["regenerate", "--replicates", "2", "--output", "/tmp/x"])
    regenerate.refuse_undeclared()


def test_the_cross_fitted_run_waits_for_its_own_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.canonical.continuous_survival_bootstrap_cross_fitted import regenerate

    monkeypatch.setattr(sys, "argv", ["regenerate", "--output", "/tmp/x"])
    reasons = regenerate.pilot_refusals()
    short = pilot.draws(PILOT, "cross_fitted") < cross_fitted.PILOT_REPLICATES
    assert any("short of the declared 50" in r for r in reasons) == short
    assert not any("in_sample" in r for r in reasons)
    monkeypatch.setattr(pilot, "read", pd.DataFrame)
    with pytest.raises(SystemExit):
        regenerate.refuse_undeclared()


# ------------------------------------------------------------ the readings on synthetic rows


def _rows(*, b1_ratio: float = 1.0, fits: int = 3_200, seed: int = 4) -> pd.DataFrame:
    """Primary rows like the driver's: unbiased, with the bootstrap SE at ``b1_ratio``."""
    rng = np.random.default_rng(seed)
    records = []
    for cell in (study.GATED_CELL, cross_fitted.CELL):
        ratio = b1_ratio
        for estimand in study.ROWS[:2]:
            truth = float(law.TRUTH[estimand])
            noise = rng.normal(size=fits) * 0.01
            estimate = truth + noise
            half = 1.96 * 0.01 * ratio
            for replicate in range(fits):
                value = float(estimate[replicate])
                records.append(
                    {
                        "scenario": cell,
                        "replicate": replicate,
                        "estimand": estimand,
                        "truth": truth,
                        "estimate": value,
                        "std_error": 0.01 * ratio,
                        "ci_lower": value - half,
                        "ci_upper": value + half,
                        "bca_lower": value - half,
                        "bca_upper": value + half,
                        "wald_std_error": 0.01,
                        "wald_lower": value - 0.0196,
                        "wald_upper": value + 0.0196,
                    }
                )
    return pd.DataFrame(records)


def test_the_calibration_rule_and_its_control_read_the_rows() -> None:
    table = study.calibration(_rows().query("scenario == @study.GATED_CELL"))
    gated = table.loc[table["gated"]]
    assert set(gated["scenario"]) == {study.GATED_CELL}
    assert set(gated["variant"]) == {"percentile", "shrunken_se_control"}
    assert gated["passed"].all()
    shrunk = table.loc[table["variant"] == "shrunken_se_control", "se_ratio"]
    assert np.allclose(shrunk, 0.85 * table.loc[table["variant"] == "percentile", "se_ratio"])
    assert study.scientific_failures({study.CALIBRATION_FILE: table})[study.CALIBRATION_AUDIT].empty
    # Nonzero witness: a bootstrap SE 8% low fails the calibration band of the gated record, and
    # the cross-fitted record gives the same rows no verdict.
    rows = _rows(b1_ratio=0.92)
    low = study.calibration(rows.query("scenario == @study.GATED_CELL"))
    failed = study.scientific_failures({study.CALIBRATION_FILE: low})[study.CALIBRATION_AUDIT]
    assert set(failed["variant"]) == {"percentile"} and len(failed) == 2
    reported = cross_fitted.calibration(rows.query("scenario == @cross_fitted.CELL"))
    assert not reported["gated"].any() and reported["passed"].all()
    assert len(reported) == len(low)
    assert cross_fitted.scientific_failures({study.CALIBRATION_FILE: reported}) == {}
    # A bootstrap SE 15% high leaves the shrunk control inside the band: the instrument fails.
    high = study.calibration(_rows(b1_ratio=1.15).query("scenario == @study.GATED_CELL"))
    failed = study.scientific_failures({study.CALIBRATION_FILE: high})[study.CALIBRATION_AUDIT]
    assert set(failed["variant"]) == {"percentile", "shrunken_se_control"}


def test_the_warp_speed_screen_reads_the_pooled_quantiles() -> None:
    rng = np.random.default_rng(1)
    name = study.ROWS[0]
    truth = float(law.TRUTH[name])
    estimate = truth + 0.01 * rng.normal(size=4_000)
    draw = estimate + 0.01 * rng.normal(size=4_000)
    rows = pd.DataFrame(
        {
            "kind": "warp",
            "replicate": range(4_000),
            "estimand": name,
            "truth": truth,
            "estimate": estimate,
            "bootstrap_draw": draw,
            "refusal": "",
        }
    )
    (coverage,) = pilot.warp_coverage(rows)["coverage"]
    assert abs(coverage - 0.95) < 0.015
    # Witness: a bootstrap half as spread as the sampling law covers far less.
    rows["bootstrap_draw"] = estimate + 0.005 * rng.normal(size=4_000)
    (narrow,) = pilot.warp_coverage(rows)["coverage"]
    assert narrow < 0.75


def test_the_first_quartile_publishes_without_a_verdict() -> None:
    rng = np.random.default_rng(3)
    fits = 400
    frame: dict[str, object] = {"scenario": study.GATED_CELL, "replicate": range(fits)}
    for index, name in enumerate(study.EARLY):
        estimate = float(law.TRUTH[name]) + 0.01 * rng.normal(size=fits)
        frame |= {
            f"early{index}_estimate": estimate,
            f"early{index}_bootstrap_std_error": 0.009,
            f"early{index}_percentile_covered": 1.0,
            f"early{index}_bca_covered": 0.0,
            f"early{index}_wald_std_error": 0.01,
            f"early{index}_wald_covered": 1.0,
        }
    table = study.early_time(pd.DataFrame(frame))
    assert list(table["estimand"]) == list(study.EARLY)
    assert "passed" not in table and "gated" not in table
    assert np.allclose(table["bootstrap_se_ratio"] / table["wald_se_ratio"], 0.9)
    assert (table["bca_coverage"] == 0.0).all() and (table["percentile_coverage"] == 1.0).all()
    assert set(study.EARLY_COLUMNS) <= set(study.EXTRA_COLUMNS)
    assert not set(study.EARLY_COLUMNS) & set(cross_fitted.EXTRA_COLUMNS)
