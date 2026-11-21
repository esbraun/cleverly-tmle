"""The point-reading fallback of the P1 PR 2 studies (plan rev 3, 5.0 rule 4 and 5.5).

When no ``R`` up to :data:`~tests.studies.repeated_sizing.MAXIMUM_REPLICATES` meets X32a's rule
after the ``pilot_2`` extension, a declaration may read every calibration row at its pooled pilot
ratio (``tau = 0``, no pilot noise) and take the smallest multiple of
:data:`~tests.studies.repeated_sizing.BUDGET_STEP` at which the product of the rows' pass
probabilities over the whole study is at least the family target.  The target is 0.95 for both
``projection-evidence`` and ``score-assessment-compositions`` (chosen 2026-11-20).

Reading pilot points as truth is weaker than X32a at the same target, so the fallback has a
two-sided precondition (R5): every row whose pilot 99% interval excludes 1, below or above, must
have had the targeted diagnosis of plan 5.5 with no defect found, and only rows whose point lies
within :data:`ABSORB_BAND` may be absorbed.  :func:`point_reading_replicates` refuses otherwise,
so a declaration cannot buy a pass for a row that is a defect suspect.  The study-specific
:mod:`tests.studies.repeated_sizing` is read, not changed.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
from scipy import stats

from tests.studies import repeated_sizing
from tests.studies.evidence.registry import Margins

#: The pilot points the fallback may absorb (plan 5.0 rule 4).
ABSORB_BAND = (0.97, 1.03)
#: The level of the pilot interval the precondition reads.
PILOT_INTERVAL_LEVEL = 0.99


def excludes_one(ratio: float, standard_error: float) -> bool:
    """Whether the pilot's 99% normal interval of a ratio excludes 1, on either side."""
    z = float(stats.norm.ppf(0.5 + PILOT_INTERVAL_LEVEL / 2.0))
    return bool(abs(ratio - 1.0) > z * standard_error)


def precondition_failures(
    ratios: Mapping[str, float],
    standard_errors: Mapping[str, float],
    diagnosed: Collection[str],
) -> list[str]:
    """The rows the fallback may not absorb, each with its reason; empty when it may run.

    A row fails when its 99% interval excludes 1 and it is not in ``diagnosed`` (the rows whose
    targeted diagnosis found no defect), or when its point lies outside :data:`ABSORB_BAND`.
    """
    failures = []
    low, high = ABSORB_BAND
    for cell, ratio in ratios.items():
        if not low <= ratio <= high:
            failures.append(f"{cell}: pilot ratio {ratio:.4f} outside [{low}, {high}]")
        elif excludes_one(ratio, standard_errors[cell]) and cell not in diagnosed:
            failures.append(f"{cell}: pilot 99% interval excludes 1 and has no diagnosis")
    return failures


def point_reading_pass(margins: Margins, ratios: Mapping[str, float], replicates: int) -> float:
    """The product of the rows' calibration pass probabilities at their pilot points."""
    values = np.array(list(ratios.values()), dtype=float)
    return float(np.prod(repeated_sizing.calibration_pass(margins, replicates, values)))


def point_reading_replicates(
    margins: Margins,
    ratios: Mapping[str, float],
    standard_errors: Mapping[str, float],
    *,
    target: float,
    diagnosed: Collection[str] = (),
) -> int:
    """The smallest budget step at which :func:`point_reading_pass` reaches ``target``.

    Raises
    ------
    ValueError
        When the precondition fails (:func:`precondition_failures`), or no step up to
        :data:`~tests.studies.repeated_sizing.MAXIMUM_REPLICATES` reaches ``target``.
    """
    failures = precondition_failures(ratios, standard_errors, diagnosed)
    if failures:
        raise ValueError("the point-reading fallback may not run: " + "; ".join(failures))
    replicates = repeated_sizing.BUDGET_STEP
    while point_reading_pass(margins, ratios, replicates) < target:
        replicates += repeated_sizing.BUDGET_STEP
        if replicates > repeated_sizing.MAXIMUM_REPLICATES:
            raise ValueError("no budget up to 50,000 reaches the point-reading target")
    return replicates


__all__ = [
    "ABSORB_BAND",
    "excludes_one",
    "point_reading_pass",
    "point_reading_replicates",
    "precondition_failures",
]
