"""The declared constants of the ``learned-rule-ep`` study, recomputed or pinned (P1 plan 5.5).

Each number here was fixed before any declared run. The EP limit rules, their values, their
regret against ``d0`` and the efficiency bounds are recomputed by quadrature on the RM30 grid.
The sizing pilot (:mod:`tests.studies.learned_rule_ep_pilot`, committed as
``tests/studies/learned_rule_ep_pilot.csv``) ran on its own seed and streams, and the plan's
sizing rule and every gate's assurance (its pass probability averaged over the pilot's
uncertainty) are recomputed from it. The study's seeds are checked against every other
learned-rule study and the pilot. No check draws a sample.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.canonical import learned_rule_run as run
from tests.studies import _learned_rule_ep_law as law
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_optimal_law as optimal_law
from tests.studies import learned_rule_ep as study
from tests.studies import learned_rule_ep_pilot as pilot
from tests.studies import learned_rule_ep_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The assurance every gate must reach (plan rev 2.5, B2: the N2 rule's target).
TARGET = 0.90

# ------------------------------------------------------------------------------ the limits


def test_the_efficiency_bounds_are_the_limit_rules_sds() -> None:
    for name in law.LAWS:
        _, variance = law.limit_value(name)
        assert math.sqrt(variance) == pytest.approx(study.EFFICIENCY_BOUNDS[name], abs=2e-6)
    variance = optimal_law.law_limit("non_exceptional")["variance"]
    assert math.sqrt(variance) == pytest.approx(
        study.EFFICIENCY_BOUNDS["optimal_known_g"], abs=2e-6
    )


def test_the_limit_rule_crosses_once_with_no_mass_at_its_threshold() -> None:
    """C3 on the EP limit: one crossing per ``W2``, and a share near 0 linear in ``epsilon``."""
    expected = {
        "non_exceptional": (-0.079741, -0.120259),
        "misspecified_limit": (0.290044, 0.266244),
    }
    for name, (low, high) in expected.items():
        crossings = law.thresholds(name)
        assert crossings[0.0] == pytest.approx((low,), abs=1e-5)
        assert crossings[1.0] == pytest.approx((high,), abs=1e-5)
        coarse, fine = law.near_threshold(name, 0.01), law.near_threshold(name, 0.001)
        assert 0.0 < fine < coarse < 0.05
        assert fine == pytest.approx(coarse / 10, rel=0.3)


def test_the_limit_rules_regret_against_d0_is_below_any_budget() -> None:
    """C3f for the optimal value: the additive limit rule gives up 0.002 SE at n = 4,000."""
    assert law.disagreement("non_exceptional") == pytest.approx(0.010125, abs=1e-6)
    regret = law.optimal_regret("non_exceptional")
    assert regret == pytest.approx(2.5454e-5, rel=1e-3)
    se = study.EFFICIENCY_BOUNDS["optimal_known_g"] / math.sqrt(properties.CALIBRATION_N)
    assert regret / se < 0.01
    truth = optimal_law.law_limit("non_exceptional")["truth"]
    assert law.limit_value("non_exceptional")[0] == pytest.approx(truth - regret, abs=1e-12)


def test_the_limit_is_the_additive_projection_of_the_effect() -> None:
    """``E[tau | W1] + E[tau | W2] - E[tau]``: its residual is orthogonal to both margins."""
    for name in law.LAWS:
        scores = law.limit_scores(name)
        effect = law._effect(name)
        mass = law._mass()
        residual = {w2: effect[w2] - scores[w2] for w2 in base.W2_LEVELS}
        np.testing.assert_allclose(
            np.mean([residual[w2] for w2 in base.W2_LEVELS], axis=0), 0.0, atol=1e-12
        )
        for w2 in base.W2_LEVELS:
            assert abs(float(np.sum(mass * residual[w2]))) < 1e-12


# ------------------------------------------------------------------------------- the pilot


@cache
def pilot_rows() -> pd.DataFrame:
    return pd.read_csv(pilot.PILOT_CSV).set_index("cell")


def _row(cell: str) -> Any:
    return pilot_rows().loc[cell]


def test_the_pilot_ran_every_declared_cell_in_full() -> None:
    rows = pilot_rows()
    assert set(rows.index) == set(pilot.CELLS)
    for cell, (_, _, n, replicates) in pilot.CELLS.items():
        assert int(rows.loc[cell, "replicates"]) == replicates
        assert int(rows.loc[cell, "n"]) == n
    assert pilot.CELLS["non_exceptional__declared"][3] == 2_000


def test_the_calibration_count_follows_the_plans_rule() -> None:
    """R from the ``non_exceptional`` pilot ratio minus one pilot SE (plan 5.5)."""
    row = _row("non_exceptional__declared")
    sigma = tools.pilot_sigma(int(row["replicates"]))
    assert 0.016 < sigma < 0.018
    replicates, admitted = properties.sizing_reading(float(row["se_ratio"]), sigma)
    assert replicates == properties.CALIBRATION_REPLICATES
    assert admitted is properties.T49_ADMITTED


def _se_pass(cell: str, replicates: int) -> tuple[float, float]:
    """The SE-ratio gate's pass probability at the pilot ratio, and its assurance.

    The assurance is :func:`tests.studies.mtp_study_tools.per_arm_pass`: the pass probability
    averaged over the pilot ratio's normal error.
    """
    row = _row(cell)
    ratio = float(row["se_ratio"])
    point = float(tools.cell_pass(ratio, replicates, tools.C_FIT))
    (assured,) = tools.per_arm_pass((ratio,), int(row["replicates"]), replicates)
    return point, float(assured)


def _coverage_law(cell: str, points: int = 400) -> np.ndarray:
    """The pilot coverage's Jeffreys posterior, as ``points`` equal-mass quantiles."""
    row = _row(cell)
    m = int(row["replicates"])
    covered = round(float(row["coverage"]) * m)
    mass = (np.arange(points) + 0.5) / points
    return np.asarray(stats.beta.ppf(mass, covered + 0.5, m - covered + 0.5))


def _interval_pass(coverage: np.ndarray, replicates: int, accept: Any) -> float:
    """``P(accept(Clopper-Pearson interval))`` at ``replicates``, averaged over ``coverage``."""
    counts = np.arange(replicates + 1)
    level = study.STUDY.margins.confidence_level
    tail = (1.0 - level) / 2.0
    low = np.where(counts > 0, stats.beta.ppf(tail, counts, replicates - counts + 1), 0.0)
    high = np.where(
        counts < replicates, stats.beta.ppf(1.0 - tail, counts + 1, replicates - counts), 1.0
    )
    ok = accept(low, high).astype(float)
    pmf = stats.binom.pmf(counts[None, :], replicates, np.atleast_1d(coverage)[:, None])
    return float(np.mean(pmf @ ok))


def calibration_pass(cell: str, replicates: int) -> dict[str, float]:
    """A gated cell's pass probability at its pilot values and its assurance.

    The two gates are read as independent: the cell's number is their product.
    """
    band = study.STUDY.margins.calibration_coverage

    def inside(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return (lo > band[0]) & (hi < band[1])

    se_point, se_assured = _se_pass(cell, replicates)
    coverage = float(_row(cell)["coverage"])
    cov_point = _interval_pass(np.array([coverage]), replicates, inside)
    cov_assured = _interval_pass(_coverage_law(cell), replicates, inside)
    return {"point": se_point * cov_point, "assurance": se_assured * cov_assured}


GATED = ("non_exceptional__declared", "misspecified_limit__declared", "optimal_known_g__declared")

#: Each gated cell's assurance at R = 7,200, recomputed here: below the 0.9 target for every
#: cell, because the pilot's own error is as large as the margin it is read against.  With the
#: ``non_exceptional`` ratio minus one SE below 0.96 the plan's rule already keeps the ``T49``
#: rows experimental; the cells run as declared and are never downgraded (plan 5.5).
ASSURANCE = {
    "non_exceptional__declared": 0.737,
    "misspecified_limit__declared": 0.806,
    "optimal_known_g__declared": 0.851,
}


def test_every_calibration_cell_passes_at_its_pilot_values() -> None:
    """At R = 7,200 each gated cell passes at its pilot SE ratio and coverage (plan 5.5)."""
    replicates = properties.CALIBRATION_REPLICATES
    for cell in GATED:
        reading = calibration_pass(cell, replicates)
        assert reading["point"] >= 0.95, (cell, reading)
        assert reading["assurance"] == pytest.approx(ASSURANCE[cell], abs=0.01), (cell, reading)
        assert reading["assurance"] < TARGET
    assert not properties.T49_ADMITTED


def test_the_fold_locality_control_resolves_below_the_band() -> None:
    """The control's 99% coverage interval at its declared count lies below 0.92."""
    cell = "non_exceptional__validation_rule_control"
    floor = study.STUDY.margins.calibration_coverage[0]
    assurance = _interval_pass(
        _coverage_law(cell), properties.CONTROL_REPLICATES, lambda lo, hi: hi < floor
    )
    assert assurance >= 0.999
    assert float(_row(cell)["coverage"]) < 0.6


def test_the_primary_budget_resolves_the_framework_defaults() -> None:
    """At 2,000 replicates of n = 2,000: the coverage floor, the SE screen and the bias margin.

    The floor passes with probability 0.99 at the pilot coverage 0.930 but its assurance over
    the 300-draw pilot is 0.77: a red floor with the n = 4,000 cells green reads as the
    finite-sample rule-learner deficit (the red-cell readings of ``regenerate.py``).
    """
    cell = "non_exceptional__primary"
    margins = study.STUDY.margins
    replicates = study.PRIMARY_REPLICATES

    def floor(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return lo >= margins.coverage_floor

    point = _interval_pass(np.array([float(_row(cell)["coverage"])]), replicates, floor)
    assert point >= 0.98
    assert _interval_pass(_coverage_law(cell), replicates, floor) == pytest.approx(0.77, abs=0.01)
    ratio = float(_row(cell)["se_ratio"])
    sigma = tools.pilot_sigma(int(_row(cell)["replicates"]))
    low, high = margins.se_ratio_sanity
    assert low + 3 * sigma < ratio < high - 3 * sigma
    bias = float(_row(cell)["standardized_bias"])
    spread = 1.0 / math.sqrt(int(_row(cell)["replicates"]))
    assert abs(bias) + 3 * spread < margins.standardized_bias
    assert tools.equivalence_power(bias, replicates, margins.standardized_bias) > 0.99


def test_the_pilot_cost_bounds_the_declared_run() -> None:
    """The declared run's CPU hours from the pilot's seconds per fit, harness refit included.

    The pilot replays each fold rule; the run refits it, which costs about one more fit. A
    gated or diagnostic replicate also fits the plug-in rule (0.2 s) and its harness.
    """
    seconds = {cell: float(_row(cell)["seconds"]) for cell in pilot.CELLS}
    big = seconds["non_exceptional__declared"]
    small = seconds["non_exceptional__primary"]
    calibration = properties.CALIBRATION_REPLICATES
    hours = (
        2 * calibration * 2 * big
        + calibration * 2 * seconds["optimal_known_g__declared"]
        + 2 * study.PRIMARY_REPLICATES * 2 * small
        + 2 * properties.DIAGNOSTIC_REPLICATES * 2 * small * 0.8
        + properties.EXCEPTIONAL_REPLICATES * 2 * small
        + properties.CONTROL_REPLICATES * 2 * small
    ) / 3600
    assert hours < 96, hours  # one run inside 24 hours on 4 cores


# ------------------------------------------------------------------------- the declaration


def test_the_cells_and_their_sizes() -> None:
    cells = study.PROPERTY_CELLS
    assert set(cells) == {
        "interval_calibration",
        "small_sample_diagnostic",
        "exceptional_law_report",
    }
    draws = {(d.family, d.label): d for d in properties.PROPERTY_DRAWS}
    for name in law.LAWS:
        draw = draws["interval_calibration", f"{name}__declared"]
        assert draw.replicates == properties.CALIBRATION_REPLICATES
        assert properties.size_of(draw.family, draw.label) == 4_000
        diagnostic = draws["small_sample_diagnostic", f"{name}__n1000"]
        assert diagnostic.replicates == 2_000
        assert properties.size_of(diagnostic.family, diagnostic.label) == 1_000
    control = draws["interval_calibration", "non_exceptional__validation_rule_control"]
    assert (control.replicates, properties.size_of(control.family, control.label)) == (500, 2_000)
    exceptional = draws["exceptional_law_report", "exceptional__n2000"]
    assert (exceptional.replicates, exceptional.law) == (2_400, "exceptional")
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (2_000, 2_000)
    assert law.rule().modifiers.columns == ("W1", "W2")


def test_the_study_is_declared_not_registered() -> None:
    """No support row cites it, so every EP-rule fit reads experimental until the run lands."""
    assert study.STUDY.slug not in {record.slug for record in registered()}
    for module in (
        "src/cleverly/estimators/ep.py",
        "src/cleverly/interventions/_rule_learners.py",
        "tests/studies/_learned_rule_optimal_law.py",
    ):
        assert module in study.STUDY.modules


def test_the_red_cell_readings_are_declared_in_the_driver() -> None:
    from tests.canonical.learned_rule_ep import regenerate

    text = regenerate.__doc__ or ""
    assert "Red-cell readings, declared before any run" in text
    for cell in ("__declared", "optimal_known_g__declared", "validation_rule_control"):
        assert cell in text


def test_the_exceptional_reading_resolves_at_every_count() -> None:
    """At 2,400 replicates the two readings meet: no count reads :data:`UNRESOLVED`.

    The 99% interval's upper end is below 0.95 up to 2,251 covered (a coverage of 0.938) and
    its lower end is at or above 0.90 from 2,252 on, so the cell reads one way or the other.
    """
    replicates = properties.EXCEPTIONAL_REPLICATES
    readings = [properties.exceptional_reading(c, replicates)[0] for c in range(replicates + 1)]
    assert properties.UNRESOLVED not in readings
    assert readings.index(properties.NO_UNDER_COVERAGE) == 2_252
    assert set(readings[2_252:]) == {properties.NO_UNDER_COVERAGE}


# ------------------------------------------------------------------------------- the seeds


def _seeds(record: Any) -> dict[str, set[int]]:
    suffixes = properties.suffixes(record)
    sample = base.declared_sample_seeds(
        record, study.PRIMARY_REPLICATES, properties.PROPERTY_DRAWS, suffixes
    )
    fold = {
        stream_seed(record, "fold_partition", "primary", s, r)
        for s in record.scenarios
        for r in range(study.PRIMARY_REPLICATES)
    }
    for draw in properties.PROPERTY_DRAWS:
        fold |= {
            stream_seed(
                record,
                "fold_partition",
                draw.family,
                draw.label,
                r,
                *suffixes[draw.family, draw.label],
            )
            for r in range(draw.replicates)
        }
    return {"sample": sample, "fold": fold}


def _pilot_seeds() -> set[int]:
    return {
        stream_seed(pilot.PILOT_RECORD, stream, cell, r)
        for stream in (pilot.STREAM, pilot.FOLD_STREAM)
        for cell, (_, _, _, replicates) in pilot.CELLS.items()
        for r in range(replicates)
    }


def test_no_seed_is_shared_with_another_learned_rule_study_or_the_pilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = study.STUDY
    assert (record.seed, record.resampling_seed) == (20261481, 20261482)
    assert all(suffix == () for suffix in properties.suffixes(record).values())
    mine = _seeds(record)
    size = 2 * study.PRIMARY_REPLICATES + sum(d.replicates for d in properties.PROPERTY_DRAWS)
    assert len(mine["sample"]) == len(mine["fold"]) == size
    assert not mine["sample"] & mine["fold"]
    everyone = run.declared_sample_seeds()
    assert mine["sample"] <= everyone
    original = base.declared_sample_seeds

    def without_this_study(target: Any, *args: Any) -> set[int]:
        return set() if target.slug == record.slug else original(target, *args)

    monkeypatch.setattr(base, "declared_sample_seeds", without_this_study)
    others = run.declared_sample_seeds()
    assert not mine["sample"] & others
    seeds = _pilot_seeds()
    assert len(seeds) == 2 * sum(cell[3] for cell in pilot.CELLS.values())
    assert not seeds & everyone
    assert not seeds & mine["fold"]
    assert replicate_seed(record, "non_exceptional", 0) not in seeds
