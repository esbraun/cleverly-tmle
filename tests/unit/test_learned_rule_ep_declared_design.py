"""The design of ``learned-rule-ep-declared`` (P1 plan rev 2.1, section 5.2), without a run.

The declared sieve's limit is computed by quadrature on the RM30 grid; the calibration ``R``
needs the 1,400-draw pilots per cell, which have not run.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

from tests.studies import learned_rule_ep_declared as study


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


def test_two_arms_of_three_laws() -> None:
    assert numbers()["arms"] == {"__S": 3, "__L": 3}
    assert study.PROPERTY_CELLS["interval_calibration"][-2:] == ("S_shrunk", "L_fold_leak")


def test_the_declared_sieve_has_a_fixed_size_and_the_learner_a_cv_size() -> None:
    sieve = study.declared_rule("S")
    learner = study.declared_rule("L")
    assert not sieve.sieve.chosen_by_cv
    assert sieve.sieve.size == study.SIEVE_SIZE == int(study.SIEVE_FAMILY.max_size)
    assert learner.sieve is None
    assert learner.learner.get_params()["max_depth"] == 2


def test_the_sieve_limit_rule_is_near_d0_at_the_optimal_cell() -> None:
    # Plan 5.2: the limit rule differs from d0 by a regret below 0.01 SE at n = 16,000.
    optimal = numbers()["optimal_known_g"]
    se = optimal["efficiency_sd"] / math.sqrt(study.CALIBRATION_N)
    assert 0.0 <= optimal["regret"] < 0.01 * se


def test_each_law_has_one_sieve_threshold_per_level() -> None:
    for law in study.LAWS:
        thresholds = numbers()[law]["thresholds"]
        assert all(len(points) == 1 for points in thresholds.values())
        assert numbers()[law]["regret"] >= 0.0


def test_the_primary_cells_name_their_curve_and_law() -> None:
    assert study.scenario_of("S_nonexc_2k") == ("S", "non_exceptional")
    assert study.scenario_of("L_misspec_2k") == ("L", "misspecified_limit")
    assert set(study.STUDY.scenarios) == set(study.PRIMARY_CELLS)
