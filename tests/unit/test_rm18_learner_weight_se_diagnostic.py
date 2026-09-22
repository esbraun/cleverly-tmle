"""The RM18 learner-weight standard-error diagnostic: selection, reading rule and mutations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.rm18_learner_weight_se.refit import (
    COMPARISON,
    CONTRAST_REGIMENS,
    FLOOR_NOT_FIT,
    HERE,
    NOT_THE_FLOOR,
    NOT_VALIDATED,
    POSITIVITY,
    REFIT_COLUMNS,
    REPRODUCTION_TOLERANCE,
    SELECTED,
    UNRESOLVED,
    payloads,
    predictions,
    reading,
    reading_table,
    refit_all,
    reproduced,
)
from tests.diagnostics.rm18_learner_weight_se.select import (
    ARMS,
    DECLARED_COMPARISON,
    DECLARED_SELECTED,
    comparison,
    learner_weight_rows,
    selected,
)

REGIMENS = (*CONTRAST_REGIMENS, "treat_if_l2")


@pytest.fixture(scope="module")
def committed() -> pd.DataFrame:
    return learner_weight_rows()


def test_selection_equals_the_committed_exploded_set(committed: pd.DataFrame) -> None:
    exploded = committed.loc[committed["std_error"] > 1.0]
    # The declaration's own statement: the same replicates explode in both arms.
    by_arm = [set(exploded.loc[exploded["cell"] == arm, "replicate"].astype(int)) for arm in ARMS]
    assert by_arm[0] == by_arm[1] == set(DECLARED_SELECTED)
    assert selected(committed) == DECLARED_SELECTED
    assert comparison(committed, DECLARED_SELECTED) == DECLARED_COMPARISON
    assert committed.loc[committed["std_error"] <= 1.0, "std_error"].max() < 1.0


def test_comparison_skips_selected_replicates(committed: pd.DataFrame) -> None:
    assert comparison(committed, (0, 5)) == (*range(1, 5), *range(6, 15))


def test_payloads_are_the_study_payloads_of_both_arms() -> None:
    chosen = payloads((*DECLARED_SELECTED, *DECLARED_COMPARISON))
    assert len(chosen) == 2 * (len(DECLARED_SELECTED) + len(DECLARED_COMPARISON))
    assert {payload[1] for payload in chosen} == {arm.removeprefix("static__") for arm in ARMS}
    # Both arms share one sample seed per replicate: the pair is a paired control.
    seeds: dict[int, set[int]] = {}
    for payload in chosen:
        seeds.setdefault(payload[2], set()).add(payload[5])
    assert all(len(values) == 1 for values in seeds.values())


def _frame() -> pd.DataFrame:
    """A synthetic refit table on which every prediction holds and the control passes."""
    records = []
    for group, replicates in ((SELECTED, (17, 20)), (COMPARISON, (0, 1))):
        for replicate in replicates:
            for cell in ARMS:
                for regimen in REGIMENS:
                    floored = 1 if group == SELECTED and regimen == "always" else 0
                    records.append(
                        {
                            "group": group,
                            "replicate": replicate,
                            "cell": cell,
                            "regimen": regimen,
                            "committed_estimate": 0.2,
                            "refit_estimate": 0.2,
                            "estimate_relative_difference": 0.0,
                            "committed_std_error": 30.0 if group == SELECTED else 0.1,
                            "refit_std_error": 30.0 if group == SELECTED else 0.1,
                            "std_error_relative_difference": 0.0,
                            "floored_share": 0.99 if group == SELECTED else 0.0,
                            "followers": 50,
                            "floored_followers": floored,
                            "zero_prefix_followers": floored,
                            "min_follower_prefix": 0.0 if floored else 0.01,
                            "max_weight": 1e8 if floored else 150.0,
                            "effective_n": 1.0 if floored else 35.0,
                        }
                    )
    return pd.DataFrame(records, columns=list(REFIT_COLUMNS))


def _at(frame: pd.DataFrame, group: str, replicate: int, regimen: str | None = None) -> pd.Series:
    mask = (frame["group"] == group) & (frame["replicate"] == replicate)
    if regimen is not None:
        mask &= frame["regimen"] == regimen
    return mask


def _control_miss(frame: pd.DataFrame) -> None:
    frame.loc[_at(frame, COMPARISON, 1), "std_error_relative_difference"] = 2e-9


def _no_floor_in_one_arm(frame: pd.DataFrame) -> None:
    mask = _at(frame, SELECTED, 20, "always") & (frame["cell"] == ARMS[1])
    frame.loc[mask, ["floored_followers", "zero_prefix_followers"]] = 0


def _floor_only_outside_the_contrast(frame: pd.DataFrame) -> None:
    _no_floor_in_one_arm(frame)
    mask = _at(frame, SELECTED, 20, "treat_if_l2") & (frame["cell"] == ARMS[1])
    frame.loc[mask, ["floored_followers", "zero_prefix_followers"]] = 1


def _small_share(frame: pd.DataFrame) -> None:
    frame.loc[_at(frame, SELECTED, 17), "floored_share"] = 0.89


def _nonzero_floored_prefix(frame: pd.DataFrame) -> None:
    frame.loc[_at(frame, SELECTED, 17, "always"), "zero_prefix_followers"] = 0


def _comparison_floor(frame: pd.DataFrame) -> None:
    frame.loc[
        _at(frame, COMPARISON, 0, "never"), ["floored_followers", "zero_prefix_followers"]
    ] = 1


def _comparison_heavy_weight(frame: pd.DataFrame) -> None:
    frame.loc[_at(frame, COMPARISON, 0, "never"), "max_weight"] = 2_000.0


def _unchanged(frame: pd.DataFrame) -> None:
    del frame


@pytest.mark.parametrize(
    ("mutate", "expected", "failing"),
    [
        pytest.param(_unchanged, POSITIVITY, set(), id="every-prediction-holds"),
        pytest.param(_control_miss, NOT_VALIDATED, set(), id="control-miss"),
        pytest.param(_no_floor_in_one_arm, NOT_THE_FLOOR, {"P1"}, id="no-floor"),
        pytest.param(
            _floor_only_outside_the_contrast, NOT_THE_FLOOR, {"P1"}, id="floor-off-contrast"
        ),
        pytest.param(_small_share, UNRESOLVED, {"P3"}, id="share-below-0.9"),
        pytest.param(_nonzero_floored_prefix, FLOOR_NOT_FIT, {"P4"}, id="nonzero-prefix"),
        # P2 and P5 are reported and never read.
        pytest.param(_comparison_floor, POSITIVITY, {"P2"}, id="comparison-floor"),
        pytest.param(_comparison_heavy_weight, POSITIVITY, {"P5"}, id="weight-at-2000"),
    ],
)
def test_reading_follows_the_declared_rule(
    mutate: Callable[[pd.DataFrame], None], expected: str, failing: set[str]
) -> None:
    frame = _frame()
    mutate(frame)
    read = predictions(frame)
    control = reproduced(frame)
    assert reading(control, read) == expected
    if control:
        assert {name for name, (holds, _) in read.items() if not holds} == failing
    table = reading_table(frame).set_index("item")["result"]
    assert table["reading"] == expected
    if not control:
        assert set(table.drop(["reproduction control", "reading"])) == {"not read"}


def test_p5_reads_every_regimen_and_p2_reads_the_contrast() -> None:
    """P5's declared scope is every regimen; P2's read is the contrast regimens.

    A heavy weight in the ``treat_if_l2`` fit fails the declared P5 and leaves the
    supplementary contrast-only row holding.  A floored ``treat_if_l2`` follower leaves the read
    P2 holding and fails the supplementary every-regimen row.  Neither moves the reading.
    """
    frame = _frame()
    frame.loc[_at(frame, COMPARISON, 0, "treat_if_l2"), "max_weight"] = 5_000.0
    table = reading_table(frame).set_index("item")["result"]
    assert table["P5"] == "fails"
    assert table["P5 over the contrast regimens, not read"] == "holds"
    assert table["reading"] == POSITIVITY

    frame = _frame()
    mask = _at(frame, COMPARISON, 1, "treat_if_l2")
    frame.loc[mask, ["floored_followers", "zero_prefix_followers"]] = 1
    table = reading_table(frame).set_index("item")["result"]
    assert table["P2"] == "holds"
    assert table["P2 over every fitted regimen, not read"] == "fails"
    assert table["reading"] == POSITIVITY


#: The statistics a refit records, which a fresh refit must give again.
MECHANISM = (
    "refit_estimate",
    "refit_std_error",
    "floored_share",
    "followers",
    "floored_followers",
    "zero_prefix_followers",
    "min_follower_prefix",
    "max_weight",
    "effective_n",
)


def test_one_declared_refit_reproduces_its_recorded_rows() -> None:
    """Refit selected replicate 17 in one arm, and compare every recorded regimen row.

    The refit goes through :func:`refit_all`, so it also checks that a payload reaches the pool
    whole.  One cross-fitted fit takes well under a second.
    """
    replicate = DECLARED_SELECTED[0]
    call = payloads((replicate,))[0]
    cell = f"static__{call[1]}"
    records = refit_all([call], jobs=1)
    assert {(record["replicate"], record["cell"]) for record in records} == {(replicate, cell)}
    fresh = pd.DataFrame(records).set_index("regimen").sort_index()
    recorded = pd.read_csv(HERE / "refit.csv")
    recorded = (
        recorded.loc[(recorded["replicate"] == replicate) & (recorded["cell"] == cell)]
        .set_index("regimen")
        .sort_index()
    )
    assert list(fresh.index) == list(recorded.index) == sorted(REGIMENS)
    # The selected replicate has a floored follower, so the check is not one of zeros.
    assert recorded.loc["always", "floored_followers"] >= 1
    for column in MECHANISM:
        np.testing.assert_allclose(
            fresh[column].to_numpy(dtype=float),
            recorded[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=0.0,
            err_msg=column,
        )


RECORDED_REFIT = HERE / "refit.csv"
RECORDED_READING = HERE / "reading.csv"


@pytest.fixture(scope="module")
def recorded() -> pd.DataFrame:
    return pd.read_csv(RECORDED_REFIT)


def test_recorded_refit_covers_the_declared_sets(recorded: pd.DataFrame) -> None:
    assert list(recorded.columns) == list(REFIT_COLUMNS)
    groups = recorded.groupby("group")["replicate"].agg(lambda values: tuple(sorted(set(values))))
    assert groups[SELECTED] == DECLARED_SELECTED
    assert groups[COMPARISON] == DECLARED_COMPARISON
    assert not recorded.duplicated(["replicate", "cell", "regimen"]).any()
    assert set(recorded.groupby(["replicate", "cell"])["regimen"].agg(frozenset)) == {
        frozenset(REGIMENS)
    }


def test_recorded_refit_reproduces_the_committed_rows(
    recorded: pd.DataFrame, committed: pd.DataFrame
) -> None:
    merged = recorded.merge(committed, on=["replicate", "cell"], validate="many_to_one")
    assert len(merged) == len(recorded)
    # The committed columns are the artifact's values.  pandas' default CSV parser can move
    # the last bit of a value on each read, so the check allows a few units of rounding.
    for column in ("estimate", "std_error"):
        np.testing.assert_allclose(
            merged[f"committed_{column}"], merged[column], rtol=1e-14, atol=0.0
        )
    for refit, published in (("refit_estimate", "estimate"), ("refit_std_error", "std_error")):
        relative = (merged[refit] - merged[published]).abs() / merged[published].abs()
        assert (relative <= REPRODUCTION_TOLERANCE).all(), refit
    assert reproduced(recorded)


def test_recorded_reading_follows_from_the_recorded_statistics(recorded: pd.DataFrame) -> None:
    published = pd.read_csv(RECORDED_READING, keep_default_na=False)
    rebuilt = reading_table(recorded).fillna("")
    assert published.to_dict("records") == rebuilt.to_dict("records")


def _without_floor(frame: pd.DataFrame) -> None:
    mask = (frame["replicate"] == DECLARED_SELECTED[0]) & (frame["cell"] == ARMS[0])
    frame.loc[mask, ["floored_followers", "zero_prefix_followers"]] = 0


def _off_committed(frame: pd.DataFrame) -> None:
    mask = frame["replicate"] == DECLARED_COMPARISON[0]
    frame.loc[mask, "std_error_relative_difference"] = 1e-8


def _share_below(frame: pd.DataFrame) -> None:
    frame.loc[frame["replicate"] == DECLARED_SELECTED[-1], "floored_share"] = 0.5


def _prefix_above_zero(frame: pd.DataFrame) -> None:
    mask = (frame["replicate"] == DECLARED_SELECTED[0]) & (frame["floored_followers"] > 0)
    frame.loc[mask, "zero_prefix_followers"] -= 1


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(_without_floor, NOT_THE_FLOOR, id="no-floor"),
        pytest.param(_off_committed, NOT_VALIDATED, id="control-miss"),
        pytest.param(_share_below, UNRESOLVED, id="share"),
        pytest.param(_prefix_above_zero, FLOOR_NOT_FIT, id="nonzero-prefix"),
    ],
)
def test_mutating_the_recorded_statistics_changes_the_reading(
    recorded: pd.DataFrame, mutate: Callable[[pd.DataFrame], None], expected: str
) -> None:
    before = reading_table(recorded).set_index("item")["result"]["reading"]
    frame = recorded.copy()
    mutate(frame)
    after = reading_table(frame).set_index("item")["result"]["reading"]
    assert (before, after) == (POSITIVITY, expected)
