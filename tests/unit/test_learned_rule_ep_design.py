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
from tests.studies import learned_rule_ep_size_pilot as size_pilot
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
        "non_exceptional": (-0.077024, -0.116253),
        "misspecified_limit": (0.284089, 0.260469),
    }
    for name, (low, high) in expected.items():
        crossings = law.thresholds(name)
        assert crossings[0.0] == pytest.approx((low,), abs=1e-5)
        assert crossings[1.0] == pytest.approx((high,), abs=1e-5)
        coarse, fine = law.near_threshold(name, 0.01), law.near_threshold(name, 0.001)
        assert 0.0 < fine < coarse < 0.05
        assert fine == pytest.approx(coarse / 10, rel=0.3)


def test_the_limit_rules_regret_against_d0_is_below_any_budget() -> None:
    """C3f for the optimal value: the limit rule gives up 0.005 SE at n = 16,000."""
    assert law.disagreement("non_exceptional") == pytest.approx(0.00975, abs=1e-6)
    regret = law.optimal_regret("non_exceptional")
    assert regret == pytest.approx(2.4259e-5, rel=1e-3)
    se = study.EFFICIENCY_BOUNDS["optimal_known_g"] / math.sqrt(properties.CALIBRATION_N)
    assert regret / se < 0.01
    truth = optimal_law.law_limit("non_exceptional")["truth"]
    assert law.limit_value("non_exceptional")[0] == pytest.approx(truth - regret, abs=1e-12)


def test_the_limit_is_the_projection_onto_the_largest_default_member() -> None:
    """The residual of ``tau`` is orthogonal, under ``P_0``, to every column of the member.

    The member is the default family's largest (six cosines of ``W1`` and the indicator of
    ``W2``), the limit of the size step over the finite family.  The size pilot's additive rule
    is the infinite-size limit, within 0.017 of it.
    """
    from cleverly.bases import Sieve

    assert Sieve.default(law.MODIFIERS).basis.max_size == 6
    mass = law._mass() / len(base.W2_LEVELS)
    for name in law.LAWS:
        scores = law.limit_scores(name)
        effect = law._effect(name)
        design = np.vstack([law._member_design(w2) for w2 in base.W2_LEVELS])
        residual = np.concatenate([effect[w2] - scores[w2] for w2 in base.W2_LEVELS])
        weight = np.concatenate([mass for _ in base.W2_LEVELS])
        assert design.shape[1] >= 7
        np.testing.assert_allclose(design.T @ (weight * residual), 0.0, atol=1e-12)
        gap = max(
            float(np.max(np.abs(scores[w2] - law.additive_scores(name)[w2])))
            for w2 in base.W2_LEVELS
        )
        assert 0.005 < gap < 0.017


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


@cache
def size_rows() -> pd.DataFrame:
    return pd.read_csv(size_pilot.PILOT_CSV).set_index("cell")


#: Each gated cell and its pilot at the calibration size (plan rev 2.6).
SIZED = {
    "non_exceptional__declared": "non_exceptional__n16000",
    "misspecified_limit__declared": "misspecified_limit__n16000",
    "optimal_known_g__declared": "optimal_known_g__n16000",
}


def _size_row(cell: str) -> Any:
    return size_rows().loc[cell]


def test_the_size_pilot_ran_every_cell_in_full() -> None:
    rows = size_rows()
    assert set(rows.index) == set(size_pilot.CELLS)
    for cell, (_, _, n, replicates) in size_pilot.CELLS.items():
        assert int(rows.loc[cell, "replicates"]) == replicates
        assert int(rows.loc[cell, "n"]) == n
    assert set(SIZED.values()) <= set(rows.index)
    for pilot_cell in SIZED.values():
        assert int(_size_row(pilot_cell)["n"]) == properties.CALIBRATION_N


def test_the_excess_variance_of_rule_learning_shrinks_with_n() -> None:
    """At n = 4,000 rule learning adds 6.5% to the error variance; at 16,000 it adds about none.

    The n = 4,000 excess reads the oracle-SE ratio, ``ratio ** -2 - 1``; the n = 16,000 excess
    reads the paired variance ratio against the limit-rule fit on the same draws, whose 95% upper
    end lies below the n = 4,000 point excess.
    """
    four = float(_row("non_exceptional__declared")["oracle_ratio"]) ** -2 - 1
    assert four == pytest.approx(0.065, abs=0.002)
    sixteen = _size_row("non_exceptional__n16000")
    upper = float(sixteen["variance_ratio"]) - 1 + 1.96 * float(sixteen["variance_ratio_sd"])
    assert upper < four
    assert float(sixteen["error_correlation"]) > 0.95


def test_the_calibration_count_follows_the_plans_rule() -> None:
    """The plan's rule on the ``non_exceptional`` ratio minus one pilot SE at n = 16,000 (5.5)."""
    row = _size_row("non_exceptional__n16000")
    sigma = tools.pilot_sigma(int(row["replicates"]))
    assert 0.019 < sigma < 0.021
    replicates, admitted = properties.sizing_reading(float(row["se_ratio"]), sigma)
    assert (replicates, admitted) == (4_800, True)
    assert admitted is properties.T49_ADMITTED
    # the assurance test raises 4,800 to the plan's other count
    assert properties.CALIBRATION_REPLICATES == 7_200
    # the first pilot, at n = 4,000, put the rows out of reach (rev 2.5's declaration)
    first = _row("non_exceptional__declared")
    old = properties.sizing_reading(
        float(first["se_ratio"]), tools.pilot_sigma(int(first["replicates"]))
    )
    assert old == (7_200, False)


def _assured(ratio: float, sigma: float, replicates: int) -> float:
    """The SE-ratio gate's pass probability averaged over a normal law of the true ratio."""
    passes = tools.cell_pass(ratio + sigma * tools._GH_X, replicates, tools.C_FIT)
    return float(np.sum(tools._GH_W * passes))


def _coverage_law(cell: str, points: int = 400, rows: Any = None) -> np.ndarray:
    """The pilot coverage's Jeffreys posterior, as ``points`` equal-mass quantiles."""
    row = _row(cell) if rows is None else rows
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


def pooled_limit_ratio() -> tuple[float, float]:
    """The limit-rule fits' SE ratio pooled over the three gated pilot cells, with its SD.

    Each cell's ratio carries the pilot error of :func:`tools.pilot_sigma`, so the pool is the
    plain mean and its SD that error over the square root of three.
    """
    rows = [_size_row(SIZED[cell]) for cell in SIZED]
    ratios = np.array([float(row["limit_se_ratio"]) for row in rows])
    sigmas = np.array([tools.pilot_sigma(int(row["replicates"])) for row in rows])
    weights = sigmas**-2
    return float(np.sum(weights * ratios) / np.sum(weights)), float(np.sum(weights) ** -0.5)


def calibration_assurance(cell: str, replicates: int) -> dict[str, Any]:
    """A gated cell's assurance at ``replicates`` from its pilot at the calibration size.

    ``unpaired`` reads the SE ratio with the pilot error of :func:`tools.pilot_sigma`.
    ``paired`` reads the SE ratio as the limit-rule fit's ratio times the paired ratio
    (learned over limit fit, on the same draws, with its bootstrap SD), and takes the limit-rule
    fit's ratio to be exactly 1.  Theory does not give that at ``misspecified_limit``, where the
    outcome regression is wrong and the fixed-rule variance is conservative.  ``pooled`` takes
    the limit-rule fit's ratio from :func:`pooled_limit_ratio` instead, one draw shared by the
    three cells.  The coverage gate reads the pilot coverage's Jeffreys posterior.  The gates
    of a cell are read as independent: a cell's number is their product.  ``pooled_nodes``
    holds the cell's pooled number at each quadrature node of the shared limit ratio, for the
    family reading.
    """
    row = _size_row(SIZED[cell])
    band = study.STUDY.margins.calibration_coverage

    def inside(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return (lo > band[0]) & (hi < band[1])

    coverage = _interval_pass(_coverage_law(cell, rows=row), replicates, inside)
    unpaired = _assured(
        float(row["se_ratio"]), tools.pilot_sigma(int(row["replicates"])), replicates
    )
    ratio, sd = float(row["paired_se_ratio"]), float(row["paired_se_ratio_sd"])
    paired = _assured(ratio, sd, replicates)
    centre, spread = pooled_limit_ratio()
    nodes = np.array(
        [
            np.sum(
                tools._GH_W
                * tools.cell_pass(
                    (centre + spread * x) * (ratio + sd * tools._GH_X), replicates, tools.C_FIT
                )
            )
            for x in tools._GH_X
        ]
    )
    return {
        "coverage": coverage,
        "unpaired": unpaired * coverage,
        "paired": paired * coverage,
        "pooled": float(np.sum(tools._GH_W * nodes)) * coverage,
        "pooled_nodes": nodes * coverage,
    }


def family_assurance(replicates: int) -> dict[str, float]:
    """The chance that every gated cell passes, on each reading of :func:`calibration_assurance`.

    The unpaired and paired readings multiply the cells' numbers.  The pooled reading
    multiplies them at each node of the shared limit ratio and then averages, because the
    three cells see the same limit-rule ratio.
    """
    readings = [calibration_assurance(cell, replicates) for cell in GATED]
    nodes = np.prod([reading["pooled_nodes"] for reading in readings], axis=0)
    return {
        "unpaired": float(np.prod([reading["unpaired"] for reading in readings])),
        "paired": float(np.prod([reading["paired"] for reading in readings])),
        "pooled": float(np.sum(tools._GH_W * nodes)),
    }


GATED = tuple(SIZED)

#: Each gated cell's assurance at the declared R = 7,200, recomputed here.
ASSURANCE = {
    "non_exceptional__declared": {
        "coverage": 1.000,
        "unpaired": 0.965,
        "paired": 1.000,
        "pooled": 0.973,
    },
    "misspecified_limit__declared": {
        "coverage": 0.972,
        "unpaired": 0.917,
        "paired": 0.972,
        "pooled": 0.965,
    },
    "optimal_known_g__declared": {
        "coverage": 1.000,
        "unpaired": 0.953,
        "paired": 0.996,
        "pooled": 0.996,
    },
}

#: The family readings at R = 7,200: the chance that no gated cell reads red under a correct
#: method.  Plan rev 2.7 records them; R stays at the rule's 7,200.
FAMILY = {"unpaired": 0.84, "paired": 0.97, "pooled": 0.94}


def test_every_calibration_cell_is_assured_at_the_declared_count() -> None:
    """Every gated cell reaches the 0.9 target (rev 2.5 B2) on every reading at R = 7,200.

    The plan's rule reads 4,800 from the ``non_exceptional`` pilot, but at 4,800 the
    ``misspecified_limit`` cell's unpaired assurance is 0.85, so the count is the plan's other
    count, 7,200 (rev 2.6), which reaches the per-cell target without the paired reading's
    assumption.  Family-wise the unpaired reading is 0.84: about a 16% chance of some gated
    red under a correct method, which the red-cell readings of ``regenerate.py`` absorb.  The
    pooled-limit reading gives 0.94 and the paired one, which takes the limit-rule fit's ratio
    to be 1, gives 0.97.
    """
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates == 7_200
    for cell in GATED:
        reading = calibration_assurance(cell, replicates)
        for key, value in ASSURANCE[cell].items():
            assert reading[key] == pytest.approx(value, abs=0.01), (cell, reading)
        assert min(reading["unpaired"], reading["paired"], reading["pooled"]) >= TARGET
    family = family_assurance(replicates)
    for key, value in FAMILY.items():
        assert family[key] == pytest.approx(value, abs=0.01), family
    centre, spread = pooled_limit_ratio()
    assert centre == pytest.approx(1.011, abs=0.002)
    assert spread == pytest.approx(0.0117, abs=0.0005)
    smaller = calibration_assurance("misspecified_limit__declared", 4_800)
    assert smaller["unpaired"] == pytest.approx(0.849, abs=0.01)
    assert smaller["unpaired"] < TARGET


def test_the_unpaired_family_reaches_090_only_above_the_rule_counts() -> None:
    """The unpaired family reaches 0.90 at about R = 9,600, above the rule's 4,800 and 7,200.

    Plan rev 2.7 records this and keeps R at 7,200: the rule allows no other count.
    """
    assert family_assurance(9_600)["unpaired"] == pytest.approx(0.90, abs=0.015)
    assert family_assurance(7_200)["unpaired"] < 0.88


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
    the 300-draw pilot is 0.77: a red floor with the n = 16,000 cells green reads as the
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
    """The declared run's CPU hours from the pilots' seconds per fit, harness refit included.

    The pilots replay each fold rule; the run refits it, which costs about one more fit.  A
    gated or diagnostic replicate also fits the plug-in rule and its harness, which the
    limit-rule fit's seconds stand in for; a calibration replicate also fits the limit-rule
    paired fit (rev 2.7), about 1.5 CPU-hours in all.  The calibration cells read the size
    pilot, the others the first pilot.
    """
    sized = {cell: _size_row(SIZED[cell]) for cell in GATED}
    paired = properties.CALIBRATION_REPLICATES * sum(
        float(row["limit_seconds"]) for row in sized.values()
    )
    assert 1.2 < paired / 3600 < 1.8
    calibration = (
        properties.CALIBRATION_REPLICATES
        * sum(2 * float(row["seconds"]) + float(row["limit_seconds"]) for row in sized.values())
        + paired
    )
    small = float(_row("non_exceptional__primary")["seconds"])
    others = (
        2 * study.PRIMARY_REPLICATES * 2 * small
        + 2 * properties.DIAGNOSTIC_REPLICATES * 2 * small * 0.8
        + properties.EXCEPTIONAL_REPLICATES * 2 * small
        + properties.CONTROL_REPLICATES * 2 * small
    )
    hours = (calibration + others) / 3600
    assert 100 < hours < 120, hours  # one resumable run: about 29 hours on 4 cores, 7 on 16


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
        assert properties.size_of(draw.family, draw.label) == 16_000
        diagnostic = draws["small_sample_diagnostic", f"{name}__n1000"]
        assert diagnostic.replicates == 2_000
        assert properties.size_of(diagnostic.family, diagnostic.label) == 1_000
    control = draws["interval_calibration", "non_exceptional__validation_rule_control"]
    assert (control.replicates, properties.size_of(control.family, control.label)) == (500, 2_000)
    exceptional = draws["exceptional_law_report", "exceptional__n2000"]
    assert (exceptional.replicates, exceptional.law) == (2_400, "exceptional")
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (2_000, 2_000)
    assert law.rule().modifiers.columns == ("W1", "W2")


def test_the_manifest_configuration_names_the_declared_sizes() -> None:
    """The run copies ``CONFIGURATION`` into ``manifest.json``: it must say what runs."""
    fit = study.CONFIGURATION["fit"]
    assert f"interval_calibration n = {properties.CALIBRATION_N}" in fit
    assert f"primary n = {study.PRIMARY_N}" in fit
    assert f"small_sample_diagnostic n = {properties.DIAGNOSTIC_N}" in fit


def test_the_study_is_registered_and_its_rows_carry_the_declared_limits() -> None:
    """The run landed, so rows ``T51`` and ``T51.optimal`` (the plan's ``T49``) cite it.

    Each carries :data:`T49_LIMITS` whatever the run showed (plan rev 2.7, S5).
    """
    from cleverly._support_rows import POINT_ROWS

    assert study.STUDY.slug in {record.slug for record in registered()}
    assert properties.T49_ADMITTED
    rows = {row.id: row for row in POINT_ROWS if study.STUDY.slug in row.evidence}
    assert set(rows) == {"T51", "T51.optimal"}
    sentence = properties.T49_LIMITS[0].upper() + properties.T49_LIMITS[1:]
    for row in rows.values():
        assert any(limit.startswith(sentence) for limit in row.limits), row.id
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
    assert "limit_std_error" in text
    assert "T49_LIMITS" in text


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
    """The sample and fold seeds of both pilots."""
    cells = [(pilot.STREAM, pilot.FOLD_STREAM, pilot.CELLS)]
    cells.append((size_pilot.STREAM, size_pilot.FOLD_STREAM, size_pilot.CELLS))
    return {
        stream_seed(pilot.PILOT_RECORD, stream, cell, r)
        for sample, fold, table in cells
        for stream in (sample, fold)
        for cell, (_, _, _, replicates) in table.items()
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
    cells = (*pilot.CELLS.values(), *size_pilot.CELLS.values())
    assert len(seeds) == 2 * sum(cell[3] for cell in cells)
    assert not seeds & everyone
    assert not seeds & mine["fold"]
    assert replicate_seed(record, "non_exceptional", 0) not in seeds


def test_the_t49_limits_name_the_calibration_size_and_the_first_pilot() -> None:
    """Every ``T49`` row's limits name n = 16,000 and the n = 4,000 pilot's deficit (rev 2.7)."""
    first = _row("non_exceptional__declared")
    limits = properties.T49_LIMITS
    assert f"n = {properties.CALIBRATION_N:,}" in limits
    assert f"n = {int(first['n']):,}" in limits
    assert f"{float(first['se_ratio']):.3f}" in limits
    assert f"{float(first['coverage']):.4f}" in limits


def test_every_calibration_row_carries_its_limit_rule_fit() -> None:
    """The paired limit-rule fit's columns are published on every row (rev 2.7, S2)."""
    assert set(properties.LIMIT_COLUMNS) <= set(properties.ROW_COLUMNS)
    assert set(properties.LIMIT_COLUMNS) == {"limit_estimate", "limit_std_error", "limit_truth"}


def test_the_exceptional_reading_is_published() -> None:
    """The summary writes the declared reading on the exceptional row, and smoke off budget."""
    rows = pd.DataFrame(
        {
            "property": [study.EXCEPTIONAL_FAMILY] * 4 + ["interval_calibration"],
            "cell": [study.EXCEPTIONAL_CELL] * 4 + ["non_exceptional__declared"],
            "covered": [1, 1, 1, 0, 1],
        }
    )
    summary = rows.drop_duplicates(["property", "cell"])[["property", "cell"]].reset_index(
        drop=True
    )
    properties.exceptional(summary, rows)
    texts = summary.set_index("cell")["reading"]
    assert texts[study.EXCEPTIONAL_CELL] == properties.SMOKE
    assert pd.isna(texts["non_exceptional__declared"])
    full = pd.DataFrame(
        {
            "property": study.EXCEPTIONAL_FAMILY,
            "cell": study.EXCEPTIONAL_CELL,
            "covered": [1] * 2_100 + [0] * (properties.EXCEPTIONAL_REPLICATES - 2_100),
        }
    )
    properties.exceptional(summary, full)
    assert summary.set_index("cell")["reading"][study.EXCEPTIONAL_CELL] == properties.UNDER_COVERS
