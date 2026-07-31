"""The library-side incremental interventions the oracle law's ``IPSI_DELTAS`` describes.

Kept out of ``tests/discrete_law.py`` for the reason :mod:`tests.regimes` is: the oracle
must state the estimand without constructing anything the library provides, and
``tests/unit/test_oracle_independence.py`` enforces that by parsing the law modules.  The
join between the two sides -- that the density this builds really is the one the oracle
differentiates -- is asserted in ``tests/unit/test_influence_gateaux_ipsi.py``, and for the
missing-outcome law in ``tests/unit/test_influence_gateaux_ipsi_mar.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cleverly.interventions import Incremental

from . import discrete_law as law


def interventions(deltas: Mapping[str, float] | None = None) -> tuple[Any, ...]:
    """One tilt per entry of ``deltas``, in that order.

    Defaults to :data:`tests.discrete_law.IPSI_DELTAS`.  The argument exists so that
    :mod:`tests.discrete_law_mar`, which restates the multipliers rather than importing
    them, can be built from *its* declaration rather than from the parent law's -- if the
    two ever drift apart, the tilts a fit declares should follow the law being checked,
    and the join test should be what fails.
    """
    values = law.IPSI_DELTAS if deltas is None else deltas
    return tuple(Incremental(delta, name=name) for name, delta in values.items())
