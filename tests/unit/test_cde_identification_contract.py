"""Independent finite-law checks for the controlled-direct-effect MAR contract."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from cleverly import CausalStudy, ControlledDirectEffect, PointTreatment, study
from cleverly.targets import TARGETS, builtin
from cleverly.targets.builtin import (
    COMPLETE_OUTCOME_PREFIX,
    CONDITIONING_EVENT_PREFIX,
    CONSISTENCY_PREFIX,
    INTERMEDIATE_CAVEAT_PREFIX,
    MISSINGNESS_CAVEAT_PREFIX,
    NO_CONFOUNDING_PREFIX,
    REFERENCE_ONLY_POSITIVITY_PREFIX,
    TWO_ARM_POSITIVITY_PREFIX,
)
from tests import discrete_law_cde


def _conditional_mean(cells: np.ndarray, outcome: int, given: dict[int, int]) -> float:
    """Return an exact empirical conditional mean from an equiprobable support table."""
    selected = np.ones(len(cells), dtype=bool)
    for column, value in given.items():
        selected &= cells[:, column] == value
    return float(np.mean(cells[selected, outcome]))


def _assert_pairwise_response_independence(cells: np.ndarray) -> None:
    """Assert Y and Z are each marginally independent of Delta."""
    for response in (0, 1):
        assert _conditional_mean(cells, 0, {2: response}) == 0.5
        assert _conditional_mean(cells, 1, {2: response}) == 0.5


def test_pairwise_missingness_claims_do_not_identify_the_cde_regression() -> None:
    """The XOR law satisfies both old clauses but violates MAR within Z."""
    # Columns are (Y, Z, Delta), with Delta = Y XOR Z. All four cells are equiprobable.
    xor = np.asarray(((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)), dtype=int)
    _assert_pairwise_response_independence(xor)
    for intermediate in (0, 1):
        assert _conditional_mean(xor, 0, {1: intermediate}) == 0.5
        observed = _conditional_mean(xor, 0, {1: intermediate, 2: 1})
        assert observed == float(1 - intermediate)
        assert observed != 0.5


def test_conditional_mar_control_identifies_the_observed_regression() -> None:
    """Three independent fair bits satisfy pairwise and conditional MAR."""
    independent = np.asarray(tuple(itertools.product((0, 1), repeat=3)), dtype=int)
    _assert_pairwise_response_independence(independent)
    for intermediate in (0, 1):
        assert _conditional_mean(independent, 0, {1: intermediate}) == 0.5
        for response in (0, 1):
            assert _conditional_mean(independent, 0, {1: intermediate, 2: response}) == 0.5


def test_identified_cde_states_both_response_restrictions() -> None:
    effect = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=0.0))

    assumptions = effect.identification.assumptions
    assert any("Y is independent of Delta given (A, Z, W)" in item for item in assumptions)
    assert any("Delta is independent of Z given (A, W)" in item for item in assumptions)


#: Which registered target owns each assumption prefix ``cleverly.study`` matches on.
#:
#: :class:`~cleverly.targets.Identification` says its assumptions are written "for a
#: reader, not for a parser", and the study nevertheless finds the sentence a narrower
#: design replaces by its opening words. Rewording one of those openings used to cancel
#: the substitution silently: no exception, no test failure, and a multi-arm summary kept
#: the binary positivity sentence forever. It now fails here.
PREFIX_OWNERS = (
    (CONSISTENCY_PREFIX, "ate"),
    (NO_CONFOUNDING_PREFIX, "ate"),
    (TWO_ARM_POSITIVITY_PREFIX, "ate"),
    (CONDITIONING_EVENT_PREFIX, "att"),
    (REFERENCE_ONLY_POSITIVITY_PREFIX, "att"),
    (MISSINGNESS_CAVEAT_PREFIX, "ey_shift"),
    (INTERMEDIATE_CAVEAT_PREFIX, "ey_shift"),
    (COMPLETE_OUTCOME_PREFIX, "ey_obs"),
)


@pytest.mark.parametrize(
    ("prefix", "owner"), PREFIX_OWNERS, ids=[owner + ":" + p for p, owner in PREFIX_OWNERS]
)
def test_each_assumption_prefix_matches_one_assumption_of_the_target_that_owns_it(
    prefix: str, owner: str
) -> None:
    assumptions = TARGETS[owner].identification.assumptions
    matched = [item for item in assumptions if item.startswith(prefix)]
    assert len(matched) == 1, (
        f"{prefix!r} matches {len(matched)} assumptions of {owner!r}; the study's "
        "substitution keys on this opening, so a reword has to move the constant too"
    )


def test_every_exported_prefix_constant_is_seated_in_the_ownership_table() -> None:
    """A new prefix has to declare which target owns it, or this list is not a contract."""
    exported = {
        name: getattr(builtin, name) for name in builtin.__all__ if name.endswith("_PREFIX")
    }
    assert set(exported.values()) == {prefix for prefix, _ in PREFIX_OWNERS}


def test_the_study_reaches_the_prefixes_through_the_module_that_owns_them() -> None:
    """The study imports the constants; it no longer repeats the sentences as literals."""
    source = Path(study.__file__).read_text(encoding="utf-8")
    for prefix, _ in PREFIX_OWNERS:
        assert f'"{prefix}"' not in source, (
            f"{prefix!r} is spelled out in cleverly/study.py; import the constant from "
            "cleverly.targets.builtin so a reword cannot reach one end and not the other"
        )
