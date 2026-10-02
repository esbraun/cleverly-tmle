r"""The variance of the learned-rule value at unequal folds.

The reported estimate is the ``1/V`` average of fold plug-ins, and its variance is

.. math::

    V^{-2} \sum_v n_v^{-2} \sum_{i \in v} D_i^2 ,

with each fold's curve :math:`D_i` centred at its own estimate
(``cross_validated_variance``).  At equal folds it equals van der Laan and Luedtke's
:math:`\sigma_n^2 / n`, so only unequal folds can see a change of the variance: the
fixture draws 203 rows into five folds.  The controls show that three wrong forms differ
from the reported variance by far more than the tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.unit import _learned_rule_support as support


@pytest.fixture(scope="module")
def planted() -> Any:
    return support.planted()


@pytest.fixture(scope="module")
def raw(planted: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    """The fold-centred curve ``D_i``, recovered from the reported, rescaled one."""
    tests = [np.asarray(test) for _, test in planted.nuisance.folds]
    n = planted.data.n
    curve = np.asarray(planted.estimates[support.NAME].influence_curve, dtype=float).copy()
    for test in tests:
        curve[test] *= len(tests) * test.size / n
    return curve, tests


def _cross_validated(curve: np.ndarray, tests: list[np.ndarray]) -> float:
    return float(
        np.sum([np.sum(curve[test] ** 2) / test.size**2 for test in tests]) / len(tests) ** 2
    )


def test_the_variance_is_the_cross_validated_form(planted: Any, raw: Any) -> None:
    curve, tests = raw
    variance = planted.estimates[support.NAME].variance
    assert variance == pytest.approx(_cross_validated(curve, tests), rel=1e-12, abs=0.0)


def test_control_the_pooled_form_differs(planted: Any, raw: Any) -> None:
    """``mean(D^2) / n`` ignores the fold sizes."""
    curve, _ = raw
    variance = planted.estimates[support.NAME].variance
    pooled = float(np.mean(curve**2)) / curve.size
    assert abs(pooled / variance - 1.0) > 1e-4


def test_control_the_empirical_variance_differs(planted: Any, raw: Any) -> None:
    """``var(D) / n`` also recentres at the pooled mean."""
    curve, _ = raw
    variance = planted.estimates[support.NAME].variance
    assert abs(float(np.var(curve)) / curve.size / variance - 1.0) > 1e-4


def test_control_recentring_within_folds_differs(planted: Any, raw: Any) -> None:
    """The fold means are not zero, so recentring each fold moves the variance."""
    curve, tests = raw
    variance = planted.estimates[support.NAME].variance
    centred = curve.copy()
    for test in tests:
        centred[test] -= np.mean(curve[test])
    assert abs(_cross_validated(centred, tests) / variance - 1.0) > 1e-4
