"""The pre-run design test of ``marginal-bound-limit-coverage`` (X82 SH).

``X82-H-plan.md`` section 8 declares these checks before the run:

* the quadrature truth has converged, agrees with a Monte Carlo draw of the law, and gives the
  ATE at ``Gamma = 1``;
* the sharp set at ``Gamma = 1.5`` is not degenerate, and the binary cell's probabilities cross
  the kink of the closed form on a set of positive mass;
* the wrong-gamma control can fail: on one fitted draw, the ``Gamma = 1.25`` interval misses
  the ``Gamma = 1.5`` end with expected coverage below 0.5;
* each cell records the declared GLM configuration, and its rows are finite;
* one property draw gives every declared cell.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit

from tests.studies import marginal_bound_coverage as study
from tests.studies import marginal_bound_coverage_properties as properties
from tests.studies import marginal_bound_law as law

#: The quadrature truth may move by this much when the node count doubles.
CONVERGENCE_TOLERANCE = 1e-6

#: Monte Carlo draws of the law for the agreement check, and its tolerance in MC SEs.
MONTE_CARLO_DRAWS = 400_000
MONTE_CARLO_SES = 4.0


@pytest.mark.parametrize("cell", law.CELLS)
class TestTheTruth:
    def test_the_quadrature_has_converged(self, cell: str) -> None:
        coarse = law.sharp_set(cell, study.GAMMA)
        fine = law.sharp_set(cell, study.GAMMA, nodes=2 * law.NODES)
        assert max(abs(a - b) for a, b in zip(coarse, fine, strict=True)) < CONVERGENCE_TOLERANCE

    def test_the_quadrature_agrees_with_monte_carlo(self, cell: str) -> None:
        rng = np.random.default_rng(20262323)
        w = rng.standard_normal((MONTE_CARLO_DRAWS, len(law.COVARIATES)))
        e1 = expit(w @ law.A_COEF)
        means = law._arm_means(cell, w @ law.Y_COEF)
        for arm, upper in ((1, True), (0, False), (1, False), (0, True)):
            share = e1 if arm == 1 else 1.0 - e1
            mean = means[arm]
            values = share * mean + (1.0 - share) * law._rho(cell, mean, study.GAMMA, upper)
            error = values.std(ddof=1) / math.sqrt(MONTE_CARLO_DRAWS)
            exact = law.arm_bound(cell, arm, study.GAMMA, upper)
            assert abs(values.mean() - exact) < MONTE_CARLO_SES * error, (arm, upper)

    def test_gamma_one_is_the_ate(self, cell: str) -> None:
        lower, upper = law.sharp_set(cell, 1.0)
        assert lower == pytest.approx(upper, abs=1e-12)
        if cell == "SH1":
            assert lower == pytest.approx(law.GAUSSIAN_EFFECT, abs=1e-9)

    def test_the_set_is_not_degenerate(self, cell: str) -> None:
        truth = study.truths(cell)
        assert truth["ate_upper"] - truth["ate_lower"] > 0.1
        assert truth == study.truths(cell, study.GAMMA)


def test_the_binary_cell_crosses_the_kink() -> None:
    """The closed form switches branch at ``p = 1 / (1 + Gamma)``; both branches carry mass."""
    rng = np.random.default_rng(5)
    w = rng.standard_normal((200_000, len(law.COVARIATES)))
    kink = 1.0 / (1.0 + study.GAMMA)
    for p in law._arm_means("SH2", w @ law.Y_COEF):
        below = float(np.mean(p < kink))
        assert 0.1 < below < 0.9


def test_the_gaussian_shift_is_the_sharp_bound() -> None:
    """``m + sigma k`` is the dual value ``q + E[phi(Y - q)]`` at the tau-quantile."""
    gamma = study.GAMMA
    tau = gamma / (1.0 + gamma)
    q = float(stats.norm.ppf(tau))
    excess = float(stats.norm.pdf(q)) - q * (1.0 - tau)
    deficit = excess + q
    dual = q + gamma * excess - deficit / gamma
    assert dual == pytest.approx(law.gaussian_shift(gamma), rel=1e-12)


@pytest.fixture(scope="module")
def first_draws() -> dict[str, object]:
    return {cell: study.draw_scenario(cell, study.PRIMARY_N, 0) for cell in law.CELLS}


@pytest.mark.parametrize("cell", law.CELLS)
def test_the_wrong_gamma_control_can_fail(cell: str, first_draws: dict[str, object]) -> None:
    frame, truth = first_draws[cell]  # type: ignore[misc]
    result = study.bounds(cell, frame)
    control = study.end_values(result, study.CONTROL_GAMMA)
    for name, (_, se) in control.items():
        shift = abs(truth[name] - study.truths(cell, study.CONTROL_GAMMA)[name])
        expected = stats.norm.cdf(study.TWO_SIDED - shift / se) - stats.norm.cdf(
            -study.TWO_SIDED - shift / se
        )
        assert expected < 0.5, (cell, name, expected)


def test_each_cell_records_its_glm_configuration() -> None:
    cells = study.cell_configurations()
    expected = {(cell, row) for cell, rows in study.SCENARIOS.items() for row in rows}
    assert set(zip(cells["scenario"], cells["parameter"], strict=True)) == expected
    assert set(cells["method"]) == {"TMLE"}
    assert set(cells["cross_fitting"]) == {"folds=5"}
    assert set(cells["n"]) == {study.PRIMARY_N}
    families = dict(zip(cells["scenario"], cells["outcome_family"], strict=True))
    assert families == {"SH1": "gaussian", "SH2": "binomial"}


def test_one_replicate_of_every_cell_gives_finite_rows() -> None:
    phase = study.draw_and_fit(replicates=1, n=study.PRIMARY_N, n_jobs=1)
    rows = phase.rows
    declared = {(cell, row) for cell, rows_of in study.SCENARIOS.items() for row in rows_of}
    assert set(zip(rows["scenario"], rows["estimand"], strict=True)) == declared
    assert np.isfinite(rows["estimate"]).all()
    assert (rows["std_error"] > 0).all()


def test_one_property_draw_gives_every_declared_cell() -> None:
    rows = [
        row
        for cell in law.CELLS
        for row in properties._replicate(cell, 0, properties._draw(cell, 0))
    ]
    assert {row["cell"] for row in rows} == set(study.PROPERTY_CELLS["interval_calibration"])
    assert all(math.isfinite(row["estimate"]) for row in rows)
