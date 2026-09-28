r"""The influence curve of the learned-rule value, row by row (roadmap row RM30).

Montoya, van der Laan, Skeem and Petersen (2023), Section 4.2, centre each fold's working
curve at that fold's own estimate:

.. math::

    D_i = H_i \{Y_i - \bar Q^*_{n,v}(A_i, W_i)\} + \bar Q^*_{n,v}(d_i, W_i) - \psi^*_{nv},
    \qquad i \in v .

The reported curve is :math:`D_i` scaled by :math:`n / (V n_v)`, so its full-sample mean
represents the ``1/V`` average.  One pooled update need not make any one fold's mean zero,
and the fixture shows that it does not.  The controls show that the longhand check sees a
dropped residual term and a curve centred at the pooled estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.unit import _learned_rule_support as support


@pytest.fixture(scope="module")
def planted() -> Any:
    """A misspecified outcome regression at unequal folds: n = 203, V = 5."""
    return support.fit(support.law_frame(203, 1), outcome_learner=support.misspecified_learner())


@pytest.fixture(scope="module")
def longhand(planted: Any) -> dict[str, Any]:
    pieces = support.validation_pieces(planted)
    n, tests = planted.data.n, pieces["tests"]
    residual = pieces["h"] * (pieces["y"] - pieces["q_star_observed"])
    plug_in = pieces["q_star_rule"]
    raw = np.empty(n)
    for test in tests:
        raw[test] = residual[test] + plug_in[test] - np.mean(plug_in[test])
    scale = np.empty(n)
    for test in tests:
        scale[test] = n / (len(tests) * test.size)
    return {**pieces, "raw": raw, "scale": scale, "residual": residual, "plug_in": plug_in}


def test_the_curve_is_the_longhand_curve_row_by_row(planted: Any, longhand: dict[str, Any]) -> None:
    reported = np.asarray(planted.estimates[support.NAME].influence_curve, dtype=float)
    np.testing.assert_allclose(reported, longhand["scale"] * longhand["raw"], atol=1e-12, rtol=0)


def test_each_fold_is_centred_at_its_own_estimate(longhand: dict[str, Any]) -> None:
    """The fold means are nonzero, and only their ``1/V`` average is zero."""
    means = np.array([np.mean(longhand["raw"][test]) for test in longhand["tests"]])
    assert float(np.max(np.abs(means))) > 1e-4
    assert abs(float(np.mean(means))) < 1e-10


def test_control_a_dropped_residual_term_is_seen(planted: Any, longhand: dict[str, Any]) -> None:
    reported = np.asarray(planted.estimates[support.NAME].influence_curve, dtype=float)
    dropped = longhand["raw"] - longhand["residual"]
    assert float(np.max(np.abs(reported - longhand["scale"] * dropped))) > 1e-3


def test_control_a_curve_centred_at_the_pooled_estimate_is_seen(
    planted: Any, longhand: dict[str, Any]
) -> None:
    reported = np.asarray(planted.estimates[support.NAME].influence_curve, dtype=float)
    pooled = longhand["residual"] + longhand["plug_in"] - planted.estimates[support.NAME].psi
    assert float(np.max(np.abs(reported - longhand["scale"] * pooled))) > 1e-4
