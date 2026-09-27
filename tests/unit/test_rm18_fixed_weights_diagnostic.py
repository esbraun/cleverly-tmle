"""The RM18 FW-A diagnostic, and the run guard every RM18 diagnostic shares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_fixed_weights import run as fw
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.manifest import UNKNOWN
from tests.studies.evidence.properties import ratio_draws, ratio_intervals
from tests.studies.evidence.seeds import stream_seed

ROWS, VALIDATION, READING = shared.part_paths(fw.HERE, fw.PART)
#: The declared registered interval: "The recomputed summary must give 1.060724 (1.019495 to
#: 1.100601)".
DECLARED_EFFICIENCY = (1.060724, 1.019495, 1.100601)


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
    assert tuple(round(float(value), 6) for value in committed) == DECLARED_EFFICIENCY


def test_the_seeds_follow_the_declared_labels_and_the_collision_rule() -> None:
    seeds = rm18_seeds.weighted_seeds()[fw.PART]
    assert len(seeds) == len(set(seeds)) == 2 * 3 * fw.REPLICATES
    moved = [
        label
        for label, seed in zip(fw.labels(), seeds, strict=True)
        if seed != stream_seed(CROSSFIT, *label)
    ]
    # The review found that ('U', 2000, 4828) equals a BD-P step 2 seed; the rule moves it.
    assert ("rm18", fw.DESIGN, "U", 2_000, 4_828) in moved
    assert len(moved) < 5
    # A smoke run takes the first draws of each rung, with the declared seeds.
    smoke = fw.payloads(2)
    assert [(p[4], p[5], p[6]) for p in smoke[:2]] == [("W", 2_000, 0), ("W", 2_000, 1)]
    assert smoke[0][-1] == seeds[0] and smoke[2][-1] == seeds[fw.REPLICATES]


def test_the_collision_rule_moves_a_taken_seed_and_records_each_assignment() -> None:
    label = ("rm18", fw.DESIGN, "W", 2_000, 0)
    natural = stream_seed(CROSSFIT, *label)
    taken = {natural}
    assert shared.fresh_seeds(CROSSFIT, [label], taken) == [
        stream_seed(CROSSFIT, *label, "retry", 1)
    ]
    assert taken == {natural, stream_seed(CROSSFIT, *label, "retry", 1)}
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


def _synthetic(ratio_at_top: float, short: bool = False) -> pd.DataFrame:
    """Ladder rows at the declared budget whose W arm at the read size has empirical efficiency
    exactly ``ratio_at_top``.  ``short`` drops one row from the first rung."""
    rng = np.random.default_rng(7)
    frames = []
    for arm in fw.ARMS:
        for n in fw.SIZES:
            read = (arm, n) == ("W", fw.READ_SIZE)
            spread = (ratio_at_top if read else 1.0) * fw.BOUNDS[arm] / np.sqrt(n)
            count = fw.REPLICATES - (1 if short and not frames else 0)
            draws = rng.normal(size=count)
            frames.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "n": n,
                        "estimate": draws / draws.std(ddof=1) * spread,
                        "std_error": fw.BOUNDS[arm] / np.sqrt(n),
                        "covered": 1,
                        "rejected": 0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _holding() -> pd.DataFrame:
    return shared.validation_frame([shared.validation_row(fw.PART, "refit rows", (True, 0.0, 1))])


def _reading(table: pd.DataFrame) -> str:
    return str(table.loc[table["statistic"] == "reading", "result"].iloc[0])


def test_the_table_reads_the_top_rung_and_a_mutation_moves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One rung per arm, at a budget whose interval can resolve the 0.03 window.
    monkeypatch.setattr(fw, "SIZES", (fw.READ_SIZE,))
    monkeypatch.setattr(fw, "REPLICATES", 10_000)
    assert _reading(fw.reading_table(_synthetic(1.0), _holding())) == fw.CONTRACTING
    assert _reading(fw.reading_table(_synthetic(1.10), _holding())) == fw.PERSISTENT


def test_a_short_rung_makes_every_label_a_smoke_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw, "REPLICATES", 50)
    table = fw.reading_table(_synthetic(1.0), _holding())
    assert shared.SMOKE not in set(table["result"])
    table = fw.reading_table(_synthetic(1.0, short=True), _holding())
    labelled = table.loc[table["statistic"].isin(["reading", "Delta(n)"]), "result"]
    assert set(labelled) == {shared.SMOKE}


def test_a_failed_harness_check_publishes_no_reading() -> None:
    failed = _holding().assign(result=shared.FAILS)
    table = fw.reading_table(_synthetic(1.0), failed)
    assert _reading(table) == shared.NOT_VALIDATED
    assert set(table["statistic"]) == {"largest scaled difference", "reading"}


def test_a_non_finite_fresh_row_stops_the_part() -> None:
    rows = pd.DataFrame({"estimate": [0.1, np.nan], "std_error": [0.01, 0.01]})
    with pytest.raises(RuntimeError, match="failed fit stops the part"):
        shared.require_finite(rows)
    shared.require_finite(rows.iloc[:1])


def test_one_registered_calibration_replicate_reproduces() -> None:
    validation = shared.validation_frame(
        shared.validate_weighted(
            CROSSFIT,
            cross_fit=True,
            part=fw.PART,
            payload=("interval_calibration", "correctly_specified"),
            cells=[fw.CELL],
            cap=1,
            jobs=1,
        )
    )
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()


def test_one_fresh_ladder_replicate_is_the_declared_fit() -> None:
    payload = next(p for p in fw.payloads(1) if (p[4], p[5]) == ("U", 32_000))
    row = shared.ladder_replicate(payload)
    frame = weighted.sample(
        law.PROBS, 32_000, payload[-1], selection=np.ones_like(weighted.SELECTION)
    )
    assert (frame["obs_weight"] == 1.0).all()
    result = weighted.fit(frame, "both_correct", cross_fit=True)
    assert row["estimate"] == float(result[shared.STATIC].psi)


# ------------------------------------------------------------------------ the run guard (R6)


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, ...], str]:
    """A fake ``git`` whose answers a test sets: clean, pushed, and holding every file."""
    answers: dict[tuple[str, ...], str] = {
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): "abc",
        ("rev-parse", "@{u}"): "abc",
    }
    monkeypatch.setattr(shared, "_git", lambda *arguments: answers.get(arguments, ""))
    return answers


def test_a_clean_pushed_run_from_this_tree_is_not_refused(git: dict[tuple[str, ...], str]) -> None:
    assert shared.refusals(smoke=False, pushed=["tests/diagnostics/rm18_boundary/pilot.csv"]) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({("status", "--porcelain"): " M x.py"}, "the tree has changes"),
        ({("rev-parse", "@{u}"): "def"}, "is not its pushed upstream"),
        ({("rev-parse", "HEAD"): UNKNOWN}, "is not its pushed upstream"),
        (
            {("cat-file", "-e", "@{u}:pilot.csv"): UNKNOWN},
            "pilot.csv is not in the pushed upstream",
        ),
    ],
)
def test_each_refusal_fires(
    git: dict[tuple[str, ...], str], change: dict[tuple[str, ...], str], reason: str
) -> None:
    git.update(change)
    refused = shared.refusals(smoke=False, pushed=["pilot.csv"])
    assert len(refused) == 1 and reason in refused[0], refused
    assert shared.refusals(smoke=True, pushed=["pilot.csv"]) == []


def test_a_library_from_another_tree_is_refused(
    git: dict[tuple[str, ...], str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shared, "ROOT", tmp_path)
    refused = shared.refusals(smoke=False)
    assert len(refused) == 1 and "cleverly is imported from" in refused[0]


# ------------------------------------------------------------------------ the committed run


def _ran() -> bool:
    """The declared set, not the file: a declared part whose record is missing fails."""
    return fw.PART in shared.RAN


def test_the_part_has_run() -> None:
    assert _ran()


@pytest.mark.skipif(not _ran(), reason="FW-A has not run")
def test_the_committed_rows_meet_the_declared_budget() -> None:
    rows = shared.read_rows(ROWS)
    assert rows.groupby(["arm", "n"]).size().eq(fw.REPLICATES).all()
    assert len(rows.groupby(["arm", "n"])) == len(fw.ARMS) * len(fw.SIZES)


@pytest.mark.skipif(not _ran(), reason="FW-A has not run")
def test_the_committed_reading_follows_from_the_committed_rows() -> None:
    rebuilt = fw.table(fw.PART, fw.HERE, 1)
    pd.testing.assert_frame_equal(
        shared.as_committed(rebuilt), shared.read_rows(READING), check_dtype=False, rtol=1e-12
    )
    assert shared.SMOKE not in set(rebuilt["result"])


@pytest.mark.skipif(not _ran(), reason="FW-A has not run")
def test_one_committed_fresh_row_refits_and_every_seed_is_declared() -> None:
    rows = shared.read_rows(ROWS)
    payloads = {(p[4], p[5], p[6]): p for p in fw.payloads()}
    declared = [
        payloads[key][-1] for key in zip(rows["arm"], rows["n"], rows["replicate"], strict=True)
    ]
    assert rows["seed"].tolist() == declared
    first = rows.iloc[0]
    refit = shared.ladder_replicate(
        payloads[first["arm"], int(first["n"]), int(first["replicate"])]
    )
    for column in ("estimate", "std_error"):
        assert shared.scaled_difference(refit[column], first[column]) <= shared.TOLERANCE
