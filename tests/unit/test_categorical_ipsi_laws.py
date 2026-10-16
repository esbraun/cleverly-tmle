r"""The three-level longitudinal laws realise their tables exactly, and no term can vanish.

The module :mod:`tests.discrete_law_longitudinal_categorical_incremental` asserts its nonzero
rules at import.  These tests restate them, so a change of a table that breaks a rule fails
here with the name of the rule, and they check that each sample holds its law's counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import discrete_law_longitudinal_categorical_incremental as law

pytestmark = pytest.mark.xdist_group("categorical_ipsi_laws")

MARGINS = law.nonzero_margins()


def test_every_nonzero_rule_has_a_margin_of_at_least_one() -> None:
    failing = {name: value for name, value in MARGINS.items() if value < 1.0}
    assert failing == {}


@pytest.mark.parametrize(
    "family",
    [
        "end/T_a",
        "end/T_b",
        "end/per_node",
        "uncensored/per_node",
        "survival h2/per_node",
        "survival h2/static_then_tilt",
        "competing relapse h2/T_a",
        "competing death h1/T_a",
    ],
)
def test_each_family_states_each_kind_of_rule(family: str) -> None:
    names = [name for name in MARGINS if name.startswith(family + ":")]
    assert any("moves the mean" in name for name in names), family
    assert any("share of the variance" in name for name in names), family
    assert any("coefficient of level" in name for name in names), family


def test_every_level_of_every_tilt_node_has_a_coefficient_witness() -> None:
    """A coefficient per level, so a mechanism term that vanishes on one level is seen."""
    for node in (1, 2):
        for level in range(law.K):
            assert f"end/T_a: node {node} coefficient of level {level}" in MARGINS


def test_the_dropout_rules_are_stated_for_every_tilt_plan() -> None:
    for plan in ("T_a", "T_b", "per_node"):
        assert MARGINS[f"end/{plan}: complete case differs"] >= 1.0
        assert MARGINS[f"end/{plan}: complete-case curve differs"] >= 1.0


def test_a_factor_of_a_named_level_moves_the_mean() -> None:
    names = [name for name in MARGINS if name.endswith("moves the mean") and "factor of" in name]
    assert len(names) >= 8


def test_the_samples_hold_their_laws_counts() -> None:
    for frame, counts in (
        (law.end_frame(), law.COUNTS),
        (law.uncensored_frame(), law.UNCENSORED_COUNTS),
        (law.survival_frame(), law.SURVIVAL_COUNTS),
        (law.competing_frame(), law.COMPETING_COUNTS),
    ):
        assert len(frame) == counts.sum()
        assert np.all(counts >= 0)
    assert law.COUNTS.sum() == law.N
    assert law.SURVIVAL_COUNTS.sum() == law.SURVIVAL_N
    assert law.COMPETING_COUNTS.sum() == law.COMPETING_N


def test_the_raw_arm_order_is_not_the_sorted_order() -> None:
    """A tilt resolved by position instead of by label would land on another level."""
    assert tuple(sorted(law.ARM_LABELS)) != law.ARM_LABELS
