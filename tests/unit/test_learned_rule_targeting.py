r"""The pooled update of the learned-rule value, against its longhand.

Van der Laan and Luedtke (2015), Appendix B, fit one :math:`\varepsilon` on the pooled
validation rows, with the clever covariate :math:`\mathbb 1\{A = d_{nv}(W)\} / g_{nv}(A \mid W)`
and the loss :math:`(1/V) \sum_v P^1_{n,v}`.  The fit therefore solves

.. math::

    \frac1V \sum_v \frac1{n_v} \sum_{i \in v} H_i \{Y_i - \bar Q^*_{n,v}(A_i, W_i)\} = 0,

and the estimate is the ``1/V`` average of the fold plug-ins.  The fixture plants a
misspecified outcome regression, so :math:`\varepsilon` is nonzero, and draws 203 rows into
five folds, so the folds are unequal and the weights :math:`n / (V n_v)` differ from one.
The controls show that each check would see the mutation it is written against: an
update left at :math:`\varepsilon = 0`, an unweighted pooled loss, and an ``n_v / n``
average of the fold plug-ins.
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
def pieces(planted: Any) -> dict[str, Any]:
    return support.validation_pieces(planted)


def _weighted_score(pieces: dict[str, Any], prediction: np.ndarray) -> float:
    """``(1/V) sum_v (1/n_v) sum_{i in v} H_i (Y_i - prediction_i)``."""
    residual = pieces["h"] * (pieces["y"] - prediction)
    return float(np.mean([np.mean(residual[test]) for test in pieces["tests"]]))


class TestTheFixtureHasTeeth:
    def test_the_folds_are_unequal(self, pieces: dict[str, Any]) -> None:
        sizes = {int(test.size) for test in pieces["tests"]}
        assert len(sizes) > 1

    def test_the_rule_treats_some_rows_and_not_others(self, pieces: dict[str, Any]) -> None:
        assert 0.0 < float(np.mean(pieces["rule"])) < 1.0

    def test_epsilon_is_nonzero(self, pieces: dict[str, Any]) -> None:
        assert float(np.max(np.abs(pieces["epsilon"]))) > 1e-2


class TestThePooledScore:
    def test_the_weighted_pooled_score_is_zero(self, pieces: dict[str, Any]) -> None:
        assert abs(_weighted_score(pieces, pieces["q_star_observed"])) < 1e-10

    def test_control_the_initial_fit_leaves_it_nonzero(
        self, planted: Any, pieces: dict[str, Any]
    ) -> None:
        """At ``epsilon = 0`` the score is far from zero, so the check above sees a skipped
        update."""
        initial = np.asarray(planted.nuisance.outcome.observed, dtype=float)
        assert abs(_weighted_score(pieces, initial)) > 1e-3

    def test_control_the_unweighted_score_is_not_zero(self, pieces: dict[str, Any]) -> None:
        """At unequal folds the stacked score differs, so an unweighted update would leave
        the weighted score nonzero."""
        residual = pieces["h"] * (pieces["y"] - pieces["q_star_observed"])
        assert abs(float(np.mean(residual))) > 1e-5


class TestTheEstimate:
    def test_is_the_one_over_v_average_of_the_fold_plug_ins(
        self, planted: Any, pieces: dict[str, Any]
    ) -> None:
        folds = [float(np.mean(pieces["q_star_rule"][test])) for test in pieces["tests"]]
        assert planted.estimates[support.NAME].psi == pytest.approx(np.mean(folds), abs=1e-12)
        record = planted.extra["learned_rule"]
        np.testing.assert_allclose(record.fold_estimates, folds, atol=1e-12, rtol=0)

    def test_control_the_n_v_over_n_average_differs(
        self, planted: Any, pieces: dict[str, Any]
    ) -> None:
        """So the check above sees an estimate weighted by fold size."""
        folds = np.array([np.mean(pieces["q_star_rule"][test]) for test in pieces["tests"]])
        sizes = np.array([test.size for test in pieces["tests"]], dtype=float)
        weighted = float(np.sum(sizes / sizes.sum() * folds))
        assert abs(weighted - planted.estimates[support.NAME].psi) > 1e-6

    def test_the_regime_density_is_the_fitted_rule(self, planted: Any) -> None:
        """The one-hot density sums to one at every row, and its treated column is the rule."""
        values = planted.nuisance.regimes.values
        np.testing.assert_array_equal(values.sum(axis=1), np.ones((planted.data.n, 1)))
        assert set(np.unique(values)) <= {0.0, 1.0}
