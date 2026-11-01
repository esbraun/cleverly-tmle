"""Unit witnesses for the truth harness of the ``learned-rule-compositions`` study.

The study's verdicts read the error ``estimate - truth`` of each row, so a wrong truth, or a wrong
row-level SE in the clustered control, would fail or pass a cell for the wrong reason. These
checks recompute each piece another way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from tests.studies import _learned_rule_compositions_law as law
from tests.studies import _learned_rule_law as base


class TestTheClusteredRegression:
    @pytest.mark.parametrize(
        ("a", "w1", "w2"), [(0.0, -0.7, 0.0), (1.0, 0.2, 1.0), (1.0, 0.9, 0.0)]
    )
    def test_the_hermite_marginal_matches_adaptive_quadrature(
        self, a: float, w1: float, w2: float
    ) -> None:
        eta = 0.2 + 0.5 * w1 - 0.3 * w2 + a * float(base.blip("non_exceptional", np.array([w1]))[0])
        scale = law.SIGMA_U * np.sqrt(2 * np.pi)

        def density(u: float) -> float:
            return float(np.exp(-0.5 * (u / law.SIGMA_U) ** 2) / scale)

        expected, _ = quad(lambda u: expit(eta + u) * density(u), -np.inf, np.inf, epsabs=1e-13)
        got = float(law.qbar0("clustered", a, np.array([w1]), w2)[0])
        assert got == pytest.approx(expected, abs=1e-12)

    def test_no_cluster_effect_gives_the_rm30_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(law, "SIGMA_U", 0.0)
        for a in (0.0, 1.0):
            np.testing.assert_allclose(
                law.qbar0("clustered", a, base.GRID, 1.0),
                base.qbar0(a, base.GRID, 1.0, "non_exceptional"),
                atol=1e-14,
                rtol=0,
            )


class TestTheWeightedValue:
    def _rule(self) -> dict[float, np.ndarray]:
        return {w2: 0.1 * (1 + w2) < base.GRID for w2 in base.W2_LEVELS}

    def test_the_weighted_value_is_the_weighted_mean(self) -> None:
        rule = self._rule()
        value, _ = law.rule_value(rule, "weighted")
        numerator = denominator = 0.0
        for w2 in base.W2_LEVELS:
            q = law.qbar0("weighted", rule[w2].astype(float), base.GRID, w2)
            w = law.weight(base.GRID)
            numerator += np.trapezoid(w * q, base.GRID)
            denominator += np.trapezoid(w, base.GRID)
        assert value == pytest.approx(numerator / denominator, abs=1e-14)
        # The witness: the weight moves the value.
        unweighted, _ = law.rule_value(rule, "fold_targeted")
        assert abs(value - unweighted) > 1e-2

    def test_a_unit_weight_gives_the_rm30_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rule = self._rule()
        monkeypatch.setattr(law, "weight", lambda w1: np.ones_like(np.asarray(w1, dtype=float)))
        value, variance = law.rule_value(rule, "weighted")
        expected, expected_variance = base.rule_value(rule, "misspecified_limit")
        assert value == pytest.approx(expected, abs=1e-12)
        assert variance == pytest.approx(expected_variance, rel=1e-10)


def _fit(scenario: str, n: int) -> Any:
    frame = law.draw(scenario, n, 7)
    return law.fit(frame, scenario, base.outcome_learner(), 3)


class TestTheHarness:
    @pytest.mark.parametrize(("scenario", "n"), [("weighted", 600), ("clustered", 600)])
    def test_every_validation_row_is_checked(self, scenario: str, n: int) -> None:
        result = _fit(scenario, n)
        truth = law.harness(result, base.outcome_learner(), scenario)
        assert truth.rows_checked == n
        assert len(truth.fold_values) == law.N_FOLDS

    def test_an_unweighted_refit_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _fit("weighted", 2_000)
        original = law.fit_on_rows

        def unweighted(learner: Any, design: Any, outcome: Any, weights: Any, *rest: Any) -> Any:
            return original(learner, design, outcome, np.ones_like(weights), *rest)

        monkeypatch.setattr(law, "fit_on_rows", unweighted)
        with pytest.raises(base.HarnessMismatch):
            law.harness(result, base.outcome_learner(), "weighted")


def test_the_row_variance_se_is_the_se_without_clusters() -> None:
    """Clusters of one row split like rows, so the control's SE is the fit without ``id=``.

    The clustered fit itself reports the centred cluster form, which differs, so the check also
    shows that the helper does not read the fit's own SE.
    """
    frame = law.draw("clustered", 600, 11).assign(cluster=np.arange(600))
    clustered = law.fit(frame, "clustered", base.outcome_learner(), 5)
    rows = (
        law.estimator("clustered", base.outcome_learner(), 5)
        .fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
        .single()
    )
    assert [list(t) for _, t in clustered.nuisance.folds] == [
        list(t) for _, t in rows.nuisance.folds
    ]
    row_se = float(rows.estimates[law.ESTIMAND].std_error)
    assert law.row_variance_se(clustered) == pytest.approx(row_se, rel=1e-10)
    assert float(clustered.estimates[law.ESTIMAND].std_error) != pytest.approx(row_se, rel=1e-6)
