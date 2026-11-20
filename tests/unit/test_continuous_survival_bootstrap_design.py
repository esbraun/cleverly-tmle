"""The declaration of ``continuous-survival-bootstrap`` (X6 plan rev 5, section 5.5)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studies import continuous_survival_bootstrap as study
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
    assert study.CELLS == {"b1_in_sample": False, "b2_cross_fitted": True}
    assert study.STUDY.reported_scenarios == {"b2_cross_fitted"}
    assert study.GATED_CELL not in study.STUDY.reported_scenarios


def test_every_risk_and_difference_at_the_three_quartiles_is_a_row() -> None:
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
    assert set(study.ROWS) == expected and len(study.ROWS) == 18
    assert all(rows == study.ROWS for rows in study.ESTIMANDS.values())


def test_the_seeds_are_used_nowhere_else_in_tests() -> None:
    mine = {str(study.SEED), str(study.RESAMPLING_SEED)}
    for path in TESTS.rglob("*.py"):
        if path.name in {"continuous_survival_bootstrap.py", Path(__file__).name}:
            continue
        numbers = set(re.findall(r"\d{7,}", path.read_text(encoding="utf-8").replace("_", "")))
        assert not mine & numbers, path


def test_the_streams_are_disjoint() -> None:
    seen: set[int] = set()
    streams = {
        **{
            f"primary/{cell}": [
                replicate_seed(study.STUDY, cell, r) for r in range(study.PRIMARY_REPLICATES)
            ]
            for cell in study.CELLS
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
    # Plan 5.5's twelve rows reach the same count, and a design ratio of 0.98 asks for 5,600.
    assert pilot.at_target_count(12) == count
    assert sizing.calibration_control_probability(study.MARGINS, count, 0.85) > 0.999


def test_the_committed_pilot_holds_both_cells() -> None:
    assert not PILOT.empty
    assert pilot.draws(PILOT, "in_sample") >= 100
    assert pilot.draws(PILOT, "cross_fitted") >= 1
    table = pilot.coverage_table(PILOT)
    assert set(table["estimand"]) == set(study.ROWS)
    assert set(pilot.budget_cpu_hours(PILOT, study.PRIMARY_REPLICATES)) == set(study.CELLS)


def test_the_failure_reading_follows_the_pilots_lost_fits() -> None:
    allowance = int(study.FAILURE_BOUND * study.PRIMARY_REPLICATES)
    refusals = pilot.failure_refusals(PILOT, study.PRIMARY_REPLICATES)
    for kind in ("in_sample", "cross_fitted"):
        failed, held = pilot.failed_draws(PILOT, kind), pilot.draws(PILOT, kind)
        probability = pilot.predicted_exceedance(failed, held, study.PRIMARY_REPLICATES, allowance)
        named = any(reason.startswith(f"{kind} lost ") for reason in refusals)
        assert named == (probability > pilot.FAILURE_EXCEEDANCE), kind
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


# ------------------------------------------------------------ the readings on synthetic rows


def _rows(*, b1_ratio: float = 1.0, fits: int = 3_200, seed: int = 4) -> pd.DataFrame:
    """Primary rows like the driver's: unbiased, with the bootstrap SE at ``b1_ratio``."""
    rng = np.random.default_rng(seed)
    records = []
    for cell in study.CELLS:
        ratio = b1_ratio if cell == study.GATED_CELL else 1.0
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
    table = study.calibration(_rows())
    gated = table.loc[table["gated"]]
    assert set(gated["scenario"]) == {study.GATED_CELL}
    assert set(gated["variant"]) == {"percentile", "shrunken_se_control"}
    assert gated["passed"].all()
    shrunk = table.loc[table["variant"] == "shrunken_se_control", "se_ratio"]
    assert np.allclose(shrunk, 0.85 * table.loc[table["variant"] == "percentile", "se_ratio"])
    assert study.scientific_failures({study.CALIBRATION_FILE: table})[study.CALIBRATION_AUDIT].empty
    # Nonzero witness: a bootstrap SE 8% low fails the calibration band in b1 only, and the
    # reported cell carries no verdict whatever it reads.
    low = study.calibration(_rows(b1_ratio=0.92))
    failed = study.scientific_failures({study.CALIBRATION_FILE: low})[study.CALIBRATION_AUDIT]
    assert set(failed["variant"]) == {"percentile"} and len(failed) == 2
    assert low.loc[low["scenario"] == "b2_cross_fitted", "passed"].all()
    # A bootstrap SE 15% high leaves the shrunk control inside the band: the instrument fails.
    high = study.calibration(_rows(b1_ratio=1.15))
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
