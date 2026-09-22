"""The RM18 learner-weight standard-error diagnostic: selection, reading rule and mutations."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from tests.diagnostics.rm18_learner_weight_se.refit import (
    COMPARISON,
    CONTRAST_REGIMENS,
    FLOOR_NOT_FIT,
    NOT_THE_FLOOR,
    NOT_VALIDATED,
    POSITIVITY,
    REFIT_COLUMNS,
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


def test_supplementary_rows_read_every_regimen() -> None:
    frame = _frame()
    mask = _at(frame, COMPARISON, 0, "treat_if_l2")
    frame.loc[mask, "max_weight"] = 5_000.0
    table = reading_table(frame).set_index("item")["result"]
    assert table["P5"] == "holds"
    assert table["P5 over every fitted regimen, not read"] == "fails"
    assert table["reading"] == POSITIVITY


def test_refit_all_passes_each_payload_whole() -> None:
    # Replicate 100 is outside both sets.  One cross-fitted fit costs well under a second.
    call = payloads((100,))[0]
    records = refit_all([call], jobs=1)
    assert {record["regimen"] for record in records} == set(REGIMENS)
    assert {(record["replicate"], record["cell"]) for record in records} == {
        (100, f"static__{call[1]}")
    }
