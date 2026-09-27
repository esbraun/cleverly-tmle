"""The RM18 FW-A diagnostic: seeds, bounds, the reading rule and its mutations, the harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_boundary.run import control_payloads
from tests.diagnostics.rm18_fixed_weights import run as fw
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import ratio_draws, ratio_intervals
from tests.studies.evidence.seeds import stream_seed

ROWS, VALIDATION, READING = shared.part_paths(fw.HERE, fw.PART)


def test_the_bounds_are_the_exact_selected_and_unweighted_bounds() -> None:
    unweighted = np.sqrt(np.sum(law.PROBS * np.square(law.eif(shared.STATIC))))
    assert fw.BOUNDS["U"] == pytest.approx(unweighted, rel=1e-12)
    assert fw.BOUNDS["W"] == weighted.efficiency_sd(shared.STATIC)
    # A nonzero witness: the weights move the bound, so the two arms read different targets.
    assert abs(fw.BOUNDS["W"] - fw.BOUNDS["U"]) > 0.4


def test_the_declared_registered_interval_is_the_committed_one() -> None:
    published = shared.read_rows(CROSSFIT.artifact("properties.csv")).set_index(
        ["property", "cell"]
    )
    row = published.loc[fw.CELL]
    committed = (
        row["efficiency_empirical_ratio"],
        row["efficiency_empirical_ci_lower"],
        row["efficiency_empirical_ci_upper"],
    )
    assert tuple(round(float(value), 6) for value in committed) == fw.REGISTERED_EFFICIENCY


def test_fresh_seeds_miss_every_registered_seed_and_each_other() -> None:
    seeds = [payload[-1] for payload in fw.payloads(fw.REPLICATES)]
    assert len(seeds) == len(set(seeds)) == 2 * 3 * fw.REPLICATES
    assert set(seeds).isdisjoint(shared.weighted_registered_seeds(CROSSFIT))
    # BD-3 draws the same selected law on the same record, so its draws must not repeat these.
    assert set(seeds).isdisjoint(payload[5] for payload in control_payloads(2_655))
    # The label is the declared one wherever no collision moved it.
    first = fw.payloads(1)[0]
    assert first[-1] == stream_seed(CROSSFIT, "rm18", fw.DESIGN, "W", 2_000, 0)


def test_the_collision_rule_moves_a_registered_seed_and_keeps_the_rest() -> None:
    label = ("rm18", fw.DESIGN, "W", 2_000, 0)
    natural = stream_seed(CROSSFIT, *label)
    moved = shared.fresh_seeds(CROSSFIT, [label], {natural})[0]
    assert moved == stream_seed(CROSSFIT, *label, "retry", 1)
    assert shared.fresh_seeds(CROSSFIT, [label, label], set()) == [
        natural,
        stream_seed(CROSSFIT, *label, "retry", 1),
    ]


def test_ratio_draws_are_the_draws_behind_ratio_intervals() -> None:
    rng = np.random.default_rng(3)
    group = pd.DataFrame(
        {"estimate": rng.normal(size=60), "std_error": 1.0 + 0.1 * rng.random(60), "n": 2_000}
    )
    draws = ratio_draws(group, replicates=500, seed=11, bound=2.0)
    intervals = ratio_intervals(group, replicates=500, seed=11, confidence_level=0.99, bound=2.0)
    for name, interval in intervals.items():
        assert shared.interval_of(draws[name]) == interval


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (Interval(0.975, 1.025), fw.CONTRACTING),
        (Interval(1.031, 1.06), fw.PERSISTENT),
        (Interval(1.01, 1.04), fw.UNRESOLVED),
        (Interval(0.95, 0.99), fw.UNRESOLVED),
        (Interval(0.96, 1.02), fw.UNRESOLVED),
    ],
)
def test_the_reading_rule_and_each_boundary(interval: Interval, expected: str) -> None:
    assert fw.reading_label(interval) == expected


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (Interval(0.001, 0.02), fw.WEIGHTS_ADD),
        (Interval(-0.02, -0.001), fw.REVERSE),
        (Interval(-0.01, 0.01), fw.NO_WEIGHT_EXCESS),
    ],
)
def test_the_attribution_rule(interval: Interval, expected: str) -> None:
    assert fw.attribution(interval) == expected


def _synthetic(ratio_at_top: float) -> pd.DataFrame:
    """Ladder rows whose W arm at 32,000 has empirical efficiency near ``ratio_at_top``."""
    rng = np.random.default_rng(7)
    frames = []
    for arm in fw.ARMS:
        for n in fw.SIZES:
            read = (arm, n) == ("W", fw.READ_SIZE)
            spread = (ratio_at_top if read else 1.0) * fw.BOUNDS[arm] / np.sqrt(n)
            draws = rng.normal(size=10_000 if read else 50)
            estimate = draws / draws.std(ddof=1) * spread
            frames.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "n": n,
                        "estimate": estimate,
                        "std_error": fw.BOUNDS[arm] / np.sqrt(n),
                        "covered": 1,
                        "rejected": 0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _holding() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "part": fw.PART,
                "check": "refit rows",
                "compared": 1,
                "largest_difference": 0.0,
                "result": shared.HOLDS,
            }
        ]
    )


def _reading(table: pd.DataFrame) -> str:
    return str(table.loc[table["statistic"] == "reading", "result"].iloc[0])


def test_the_table_reads_the_top_rung_and_a_mutation_moves_it() -> None:
    assert _reading(fw.reading_table(_synthetic(1.0), _holding())) == fw.CONTRACTING
    assert _reading(fw.reading_table(_synthetic(1.10), _holding())) == fw.PERSISTENT


def test_a_failed_harness_check_publishes_no_reading() -> None:
    failed = _holding().assign(result=shared.FAILS)
    table = fw.reading_table(_synthetic(1.0), failed)
    assert _reading(table) == shared.NOT_VALIDATED
    assert set(table["statistic"]) == {"largest scaled difference", "reading"}


def test_one_registered_calibration_replicate_reproduces() -> None:
    validation = fw.validate(1, 1)
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()


def test_one_fresh_ladder_replicate_is_the_declared_fit() -> None:
    payload = fw.payloads(1)[-1]
    row = shared.ladder_replicate(payload)
    frame = weighted.sample(
        law.PROBS, payload[5], payload[-1], selection=np.ones_like(weighted.SELECTION)
    )
    assert (frame["obs_weight"] == 1.0).all()
    result = weighted.fit(frame, "both_correct", cross_fit=True)
    assert row["arm"] == "U" and row["n"] == 32_000
    assert row["estimate"] == float(result[shared.STATIC].psi)


@pytest.mark.skipif(not ROWS.exists(), reason="FW-A has not run")
def test_the_committed_reading_follows_from_the_committed_rows() -> None:
    rebuilt = fw.reading_table(shared.read_rows(ROWS), shared.read_rows(VALIDATION))
    pd.testing.assert_frame_equal(rebuilt, shared.read_rows(READING), check_dtype=False, rtol=1e-12)


@pytest.mark.skipif(not ROWS.exists(), reason="FW-A has not run")
def test_one_committed_fresh_row_refits_and_every_seed_is_declared() -> None:
    rows = shared.read_rows(ROWS)
    payloads = {(p[4], p[5], p[6]): p for p in fw.payloads(fw.REPLICATES)}
    declared = [
        payloads[arm, n, index][-1]
        for arm, n, index in zip(rows["arm"], rows["n"], rows["replicate"], strict=True)
    ]
    assert rows["seed"].tolist() == declared
    first = rows.iloc[0]
    refit = shared.ladder_replicate(
        payloads[first["arm"], int(first["n"]), int(first["replicate"])]
    )
    for column in ("estimate", "std_error"):
        assert shared.scaled_difference(refit[column], first[column]) <= shared.TOLERANCE
