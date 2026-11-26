"""The cluster counts at which ``few-cluster-adaptive-learners`` measured the default library.

The counts are the study's own (``COUNTS`` in ``tests/studies/few_cluster_adaptive_learners.py``,
held equal by ``tests/unit/test_adaptive_nuisance_support.py``), not a policy floor, so a change
of an interval floor never moves the measured ground or the sentence that states it. They sit in
a module of their own because :mod:`cleverly._inference_status`, which states them in its
t-reference note, writes no cluster count by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

__all__ = ["ADAPTIVE_MEASURED_COUNTS"]

#: The positive-mass cluster counts of each measured fit kind: stacked ``TMLE`` (``"point"``) on
#: two equal-size laws, and cross-fitted ``LTMLE`` (``"longitudinal"``) on one.
ADAPTIVE_MEASURED_COUNTS: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {"point": (10, 20, 40), "longitudinal": (20,)}
)
