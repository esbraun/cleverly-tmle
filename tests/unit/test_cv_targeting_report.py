"""What ``result.cv_targeting.pooled`` reports, and when the object exists.

``CVTargeting.pooled`` is the whole-sample plug-in at the fitted fluctuation. Under
``cv_evaluation=True`` that fluctuation minimises the fold-reweighted loss, which scales
each validation fold by ``n / (V * sum(w_fold))`` (``TMLE._validation_weights``). The
report therefore equals the default stacked (Levy) report only when every fold has
equal weight mass. These tests pin both sides of that condition with one fixed split, so
the only thing that differs between the two samples is the fold sizes.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly import SplitPlan
from cleverly.datasets import make_binary_outcome
from cleverly.learners import random_partition
from tests.conftest import fast_tmle

LINEAR = ("ate", "ey1", "ey0")


def _fits(n: int) -> tuple[Any, Any]:
    """The default stacked fit and the fold-evaluated fit on the same two-fold split."""
    frame = make_binary_outcome(n=n, seed=3)[0]
    # A two-fold draw fixes the fold sizes: 50 and 50 at n = 100, 51 and 50 at n = 101.
    # The seed is named because the reweighting gap below is a property of this split:
    # every two-fold draw has these sizes, and they move the fluctuation by different
    # amounts. This one moves it furthest from the stacked report.
    plan = SplitPlan.from_folds([random_partition(n, 2, seed=5)])
    settings: dict[str, Any] = {"estimands": LINEAR, "n_folds": 2, "split_plan": plan}
    stacked = fast_tmle(**settings).fit(frame, outcome="Y", treatment="A").single()
    evaluated = (
        fast_tmle(**settings, cv_evaluation=True).fit(frame, outcome="Y", treatment="A").single()
    )
    return stacked, evaluated


def test_a_default_stacked_fit_has_no_cv_targeting() -> None:
    stacked, evaluated = _fits(100)

    assert stacked.cv_targeting is None
    assert evaluated.cv_targeting is not None


def test_pooled_equals_the_stacked_report_at_equal_fold_sizes() -> None:
    stacked, evaluated = _fits(100)
    detail = evaluated.cv_targeting

    assert detail.fold_sizes == (50, 50)
    for name in LINEAR:
        assert detail.pooled[name].psi == pytest.approx(stacked[name].psi, abs=1e-10)


def test_pooled_departs_from_the_stacked_report_at_unequal_fold_sizes() -> None:
    """The deliberate witness: one extra row moves the reweighted fluctuation.

    At 51 and 50 rows the reweighting factors are ``101 / 102`` and ``101 / 100``. If
    ``pooled`` were the stacked report, this gap would be zero, as it is at 50 and 50.
    """
    stacked, evaluated = _fits(101)
    detail = evaluated.cv_targeting

    assert detail.fold_sizes == (51, 50)
    gap = abs(detail.pooled["ate"].psi - stacked["ate"].psi)
    assert gap > 1e-3, "the fold-reweighting gap vanished at unequal fold sizes"
