"""The :class:`~cleverly.interventions.Intervention` objects matching :data:`law.REGIMES`.

The oracle in :mod:`tests.discrete_law` declares the regimes as densities, because the
parameter name it keys on carries the regime's *label*.  This module builds the library
objects that must reproduce them.  Keeping the two apart is what makes the comparison a
check rather than a restatement: the oracle never constructs an
:class:`~cleverly.interventions.Intervention`, and these never state a functional.

``tests/unit/test_regimes.py`` asserts the densities agree, which is the join between
them; every other regime test may then use either side freely.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cleverly.interventions import Rule, Static, Stochastic
from tests import discrete_law as law


def _levels(frame: Any) -> np.ndarray:
    """``W`` as an integer index into the law's three covariate values."""
    return np.rint(np.asarray(frame["W"], dtype=float)).astype(int)


def interventions() -> tuple[Any, ...]:
    """The three regimes, in the order their codes follow: never, rule, tilt.

    One of each kind on purpose -- static, deterministic-and-``W``-dependent, and
    stochastic-everywhere -- so that code which mixes over the arms is distinguishable
    from code which picks a column.
    """
    return (
        Static(0.0, name="never"),
        # d(w) = 0 at w = 1 and 1 elsewhere: the rule has to *look* at W, or the
        # comparison against a static regime proves nothing.
        Rule(lambda frame: np.where(_levels(frame) == 1, 0, 1), name="rule"),
        Stochastic(lambda frame: law.REGIMES["tilt"][_levels(frame)], name="tilt"),
    )
