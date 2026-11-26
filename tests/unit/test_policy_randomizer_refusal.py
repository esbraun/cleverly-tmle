"""A randomized policy needs a ``Randomizer``: finitely many values of known probability.

The estimator integrates the randomizer out as a finite mixture of branch means, so a
continuous draw is refused when the policy is built (X64), and the two-point discretization
that the refusal names is accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import ModifiedPolicy, Piece, Randomizer

_RAISE = (Piece(-np.inf, np.inf, lambda a, h: a + 0.5, lambda b, h: b - 0.5, lambda b, h: 1.0),)
_KEEP = (Piece(-np.inf, np.inf),)


def test_a_continuous_randomizer_is_refused_when_the_policy_is_built() -> None:
    with pytest.raises(CapabilityError) as caught:
        ModifiedPolicy(
            "raise at random",
            pieces={"raise": _RAISE, "keep": _KEEP},
            randomizer=np.random.default_rng(0).uniform,  # type: ignore[arg-type]
            policy_kind="known",
        )
    error = caught.value
    assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert "draws a continuous randomizer" in str(error)
    assert "Randomizer((True, False), (delta, 1 - delta))" in str(error)
    assert error.refusal.roadmap == "X64"


def test_the_two_point_randomizer_the_refusal_names_is_accepted() -> None:
    policy = ModifiedPolicy(
        "raise at random",
        pieces={True: _RAISE, False: _KEEP},
        randomizer=Randomizer((True, False), (0.25, 0.75)),
        policy_kind="known",
    )
    assert policy.randomizer is not None
    assert policy.randomizer.probabilities == (0.25, 0.75)
