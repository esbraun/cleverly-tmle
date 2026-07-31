"""The library-side incremental interventions the oracle law's ``IPSI_DELTAS`` describes.

Kept out of ``tests/discrete_law.py`` for the reason :mod:`tests.regimes` is: the oracle
must state the estimand without constructing anything the library provides, and
``tests/unit/test_oracle_independence.py`` enforces that by parsing the law modules.  The
join between the two sides -- that the density this builds really is the one the oracle
differentiates -- is asserted in ``tests/unit/test_influence_gateaux_ipsi.py``.
"""

from __future__ import annotations

from typing import Any

from cleverly.interventions import Incremental

from . import discrete_law as law


def interventions() -> tuple[Any, ...]:
    """One tilt per entry of :data:`tests.discrete_law.IPSI_DELTAS`, in that order."""
    return tuple(Incremental(delta, name=name) for name, delta in law.IPSI_DELTAS.items())
