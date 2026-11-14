"""The Fix-6 sizing rule: joint draws of one fit's rows, shared planning draws, set-aside rows.

The rule in :mod:`tests.studies.drtmle_foldwise_sizing` draws the bias, coverage and SE-ratio
rows of one fit jointly from the pilot's influence covariance, lets units of one configuration
share a planning draw, sets aside only coverage and SE-ratio rows no acceptance reads and only
below :data:`SET_ASIDE_THRESHOLD` at the ceiling, and sizes the contraction ladder
predictively.  The sizing pilot continues its streams with ``--start``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.fix6_foldwise_sizing import run as pilot
from tests.studies import multi_arm_drtmle_foldwise as unit_m
from tests.studies import stratified_drtmle_foldwise as unit_s
from tests.studies.drtmle_foldwise_sizing import (
    SET_ASIDE_CLAUSES,
    SET_ASIDE_THRESHOLD,
    LadderPlan,
    Pilot,
    Rule,
    control_budget,
    control_fails_pass,
    ladder_resolution_pass,
    pilot_from_rows,
    predictive_outer_budget,
    read_planning,
    second_order_pool,
    set_aside_rows,
    single,
    without,
)
from tests.studies.evidence.seeds import stream_seed


def synthetic_rows(
    replicates: int = 4_000, *, bias: float = 0.1, se: float = 1.0, seed: int = 7
) -> pd.DataFrame:
    """Two rows of one fit, errors correlated 0.6, a Wald interval at the reported SE."""
    rng = np.random.default_rng(seed)
    errors = rng.multivariate_normal([bias, 0.0], [[1.0, 0.6], [0.6, 1.0]], replicates)
    out = []
    for r in range(replicates):
        for index, name in enumerate(("a", "b")):
            error = float(errors[r, index])
            reported = se * float(np.exp(0.05 * rng.standard_normal()))
            out.append(
                {
                    "replicate": r,
                    "estimand": name,
                    "truth": 0.0,
                    "estimate": error,
                    "std_error": reported,
                    "covered": int(abs(error) <= 1.959964 * reported),
                }
            )
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- one fit, jointly
def test_the_influence_covariance_carries_every_statistic_of_one_fit() -> None:
    fitted = pilot_from_rows(synthetic_rows(), source="synthetic")
    assert fitted.rows == ("a", "b") and fitted.calibrated
    covariance = np.asarray(fitted.influence)
    assert covariance.shape == (6, 6)
    a_bias, a_cover, a_ratio = (fitted.index(s, "a") for s in ("bias", "coverage", "se_ratio"))
    # The bias influence has unit variance and the rows' errors correlate as drawn.
    assert covariance[a_bias, a_bias] == pytest.approx(1.0, abs=0.08)
    assert covariance[a_bias, fitted.index("bias", "b")] == pytest.approx(0.6, abs=0.06)
    # The coverage influence has the Bernoulli variance.
    c = fitted.coverage[0]
    assert covariance[a_cover, a_cover] == pytest.approx(c * (1 - c), rel=0.02)
    # A draw with a larger spread lowers both the coverage and the SE ratio: they move together,
    # which drawing the rows independently would miss.
    assert covariance[a_cover, a_ratio] > 0.01
    assert fitted.se_ratio[0] == pytest.approx(1.0, abs=0.05)


def test_a_bias_only_pilot_carries_only_the_bias_block() -> None:
    fitted = pilot_from_rows(synthetic_rows(500), source="s", calibration=False)
    assert not fitted.calibrated and np.shape(fitted.influence) == (2, 2)
    with pytest.raises(KeyError):
        fitted.index("coverage", "a")


def test_the_planning_draws_follow_the_influence_covariance() -> None:
    fitted = pilot_from_rows(synthetic_rows(), source="synthetic")
    rule = Rule(pilots={"p": fitted}, units={"u": (("p",), 400)}, seed=3, draws=40_000)
    draws = rule.planning_draws["p"]
    empirical = np.cov(draws, rowvar=False) * fitted.replicates
    assert np.allclose(empirical, fitted.covariance, atol=0.03)
    assert np.allclose(draws.mean(axis=0), fitted.point, atol=0.002)


def test_without_restricts_every_block() -> None:
    fitted = pilot_from_rows(synthetic_rows(500), source="s")
    kept = without(fitted, ["a"])
    assert kept.rows == ("b",)
    covariance = np.asarray(fitted.influence)
    wide = [fitted.index(s, "b") for s in ("bias", "coverage", "se_ratio")]
    assert np.allclose(np.asarray(kept.influence), covariance[np.ix_(wide, wide)])


# ------------------------------------------------------------------- one configuration
def test_two_pilots_of_one_configuration_share_their_planning_draw() -> None:
    fitted = pilot_from_rows(synthetic_rows(400), source="s")
    cell = single("cell of the same fit", 400, "a", fitted.bias[0])
    rule = Rule(
        pilots={"primary": fitted, "cell": cell},
        units={"primary": (("primary",), 400), "cells": (("cell",), 400)},
        seed=5,
        shares={"cell": "primary"},
    )
    planned = rule.planning_draws
    assert np.array_equal(planned["cell"][:, 0], planned["primary"][:, fitted.index("bias", "a")])
    # Each draws its own study noise.
    primary_bias = rule.row_clauses("primary", 400, ("bias",))[("a", "bias")]
    cell_bias = rule.row_clauses("cell", 400, ("bias",))[("a", "bias")]
    assert not np.array_equal(primary_bias, cell_bias)
    unshared = dataclasses.replace(rule, shares={})
    assert not np.array_equal(unshared.planning_draws["cell"][:, 0], planned["cell"][:, 0])
    with pytest.raises(ValueError, match="does not plan"):
        Rule(
            pilots={"primary": fitted, "cell": single("x", 400, "z", 0.0)},
            units={"u": (("primary", "cell"), 400)},
            seed=1,
            shares={"cell": "primary"},
        )


# ------------------------------------------------------------------------ rows set aside
def test_the_set_aside_threshold_and_clauses_are_declared() -> None:
    assert SET_ASIDE_THRESHOLD == 0.80
    assert SET_ASIDE_CLAUSES == ("se_ratio", "coverage")


def test_only_unprotected_calibration_rows_below_the_threshold_are_set_aside() -> None:
    low = pilot_from_rows(synthetic_rows(400, se=0.83), source="under-covered")
    rule = Rule(pilots={"p": low}, units={"u": (("p",), 400)}, seed=11)
    aside, chosen = set_aside_rows(rule)
    assert chosen and all(clause in SET_ASIDE_CLAUSES for _, _, clause, _ in chosen)
    assert all(value < SET_ASIDE_THRESHOLD for *_, value in chosen)
    assert aside.unit_pass("u", 8_000) > rule.unit_pass("u", 8_000)
    protected = dataclasses.replace(rule, protected=frozenset({("p", "a"), ("p", "b")}))
    assert set_aside_rows(protected)[1] == ()
    with pytest.raises(ValueError, match="never set aside"):
        dataclasses.replace(rule, set_aside=frozenset({("p", "a", "bias")}))
    with pytest.raises(ValueError, match="acceptance"):
        dataclasses.replace(protected, set_aside=frozenset({("p", "a", "coverage")}))
    # A row with a pass at the ceiling at or above the threshold stays.
    clean = pilot_from_rows(synthetic_rows(400), source="clean")
    assert set_aside_rows(Rule(pilots={"p": clean}, units={"u": (("p",), 400)}, seed=11))[1] == ()


def test_the_records_name_the_rows_an_acceptance_reads() -> None:
    from tests.studies import drtmle_foldwise_n1500 as unit_b

    assert unit_b.rule().protected == {
        ("cell_outcome_correct", "ate"),
        ("cell_treatment_correct", "ate"),
    }
    assert unit_m.rule().protected == frozenset()
    assert unit_s.rule().protected == frozenset()


# ------------------------------------------------------------------------------ the ladder
PLAN = LadderPlan("synthetic", ("m", "v"), 8_000, (0.4, 0.4), ((1.0, 0.5), (0.5, 1.0)), 600)
SIZES = (2_000, 4_000, 8_000)


def test_the_ladder_resolution_pass_grows_with_the_outer_budget() -> None:
    values = [
        ladder_resolution_pass(PLAN, SIZES, (r, 800, r), target=0.25, level=0.99, seed=1)
        for r in (400, 1_200, 3_200)
    ]
    assert values == sorted(values) and values[-1] > 0.95
    budget, value, sized = predictive_outer_budget(PLAN, SIZES, 800, target=0.25, level=0.99)
    assert sized and value >= 0.95 and budget % 400 == 0
    small = dataclasses.replace(PLAN, standardized=(0.05, 0.05))
    assert predictive_outer_budget(small, SIZES, 800, target=0.25, level=0.99)[2] is False


def test_a_first_order_control_fails_to_contract_at_the_floor() -> None:
    strong = dataclasses.replace(PLAN, size=2_000, standardized=(3.0, 3.0))
    assert control_fails_pass(strong, SIZES, 400, first_order=-0.5, level=0.99) > 0.99
    assert control_budget(strong, SIZES, first_order=-0.5, level=0.99)[::2] == (400, True)


def test_the_second_order_pool_carries_the_lower_rung_up_by_inverse_variance() -> None:
    low = dataclasses.replace(PLAN, size=2_000, standardized=(0.8, 0.4), replicates=600)
    top = dataclasses.replace(PLAN, standardized=(0.4, 0.4), replicates=600)
    pooled = second_order_pool(top, low)
    # Half the lower rung's value, with variance (1/4)/600, so four times the top's weight.
    assert pooled.standardized == (pytest.approx(0.4), pytest.approx(0.24))
    assert pooled.replicates == 3_000
    assert (pooled.size, pooled.correlation, pooled.labels) == (8_000, top.correlation, top.labels)
    with pytest.raises(ValueError, match="same labels"):
        second_order_pool(top, dataclasses.replace(low, labels=("m", "w")))


# ------------------------------------------------------------------------------- the pilot
def test_the_pilot_continues_its_streams() -> None:
    assert "b_both_correct" in pilot.GROUPS
    module, n = pilot.GROUPS["b_both_correct"]
    assert n == 1_500 and module.PRIMARY_SCENARIOS == ("both_correct",)
    # The ladder groups keep the entropy their first replications drew with.
    retired = dataclasses.replace(unit_s.STUDY, resampling_seed=pilot.LADDER_PILOT_ENTROPY)
    assert pilot.seed("ladder_tc_8000", 250) == stream_seed(
        retired, "sizing_pilot", "ladder_tc_8000", 250
    )
    assert pilot.seed("s_primary", 250) == stream_seed(
        unit_s.STUDY, "sizing_pilot", "s_primary", 250
    )


def test_the_pilot_starts_at_its_index_and_merges_earlier_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    pilot.main(["--output", str(first), "--group", "m_outcome_correct", "--replicates", "3"])
    second = tmp_path / "second.csv"
    pilot.main(
        [
            "--output",
            str(second),
            "--group",
            "m_outcome_correct",
            "--start",
            "3",
            "--replicates",
            "1",
            "--rows",
            str(tmp_path / "first-rows.csv.gz"),
            "--planning",
            str(tmp_path / "planning.json"),
        ]
    )
    rows = pd.read_csv(tmp_path / "second-rows.csv.gz")
    assert list(rows["replicate"]) == [3]
    summary = pd.read_csv(second)
    assert int(summary["replicates"].iloc[0]) == 4
    plans, seconds = read_planning(tmp_path / "planning.json")
    (cell,) = plans.values()
    assert set(plans) == set(seconds) == {"m_outcome_correct"}
    assert isinstance(cell, Pilot) and cell.replicates == 4 and cell.rows == ("outcome_correct",)
    with pytest.raises(SystemExit, match="repeat"):
        pilot.main(
            [
                "--output",
                str(tmp_path / "third.csv"),
                "--replicates",
                "0",
                "--rows",
                str(tmp_path / "first-rows.csv.gz"),
                "--rows",
                str(tmp_path / "first-rows.csv.gz"),
            ]
        )
