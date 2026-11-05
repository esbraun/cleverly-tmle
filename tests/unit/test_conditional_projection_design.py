"""The declaration of the ``conditional-projection`` study, before its run.

The truths are checked against an independent Monte Carlo estimate and against the closed
form where one exists, the quadrature error is shown to sit far below any margin, and one
cheap replication runs through the declared fit.  The run itself is not started here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from tests.studies import conditional_projection as study


def test_the_truth_matches_an_independent_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(-1.0, 1.0, (400_000, 3))
    tau = expit(study.outcome_logit(w, 1)) - expit(study.outcome_logit(w, 0))
    phi = np.column_stack([np.ones(len(w)), w[:, 0], w[:, 2]])
    estimate = np.linalg.solve(phi.T @ phi, phi.T @ tau)
    truth = study.truth("observational")
    np.testing.assert_allclose(
        [truth[f"cate_projection[{t}]"] for t in study.TERMS], estimate, atol=3e-3
    )


def test_the_heterogeneity_law_has_its_closed_form_truth() -> None:
    truth = study.truth("hetero_trial", slope=0.10)
    np.testing.assert_allclose(
        [truth[f"cate_projection[{t}]"] for t in study.TERMS], [0.2, 0.1, 0.0], atol=1e-12
    )


@pytest.mark.parametrize(("law", "scale"), [("observational", "difference"), ("crr", "log_ratio")])
def test_the_quadrature_error_is_negligible(law: str, scale: str) -> None:
    assert study.quadrature_error(law, scale) < 1e-10


def test_the_projection_is_genuine() -> None:
    """The CATE is not linear in W3, so the W3 slope is a projection, not a coefficient."""
    w, mass = study._grid()
    q_t, q_r = study.arm_means("observational", w)
    phi = np.column_stack([np.ones(len(w)), w[:, 0], w[:, 2]])
    beta = study.projection(phi, mass, q_t, q_r, "difference")
    residual = q_t - q_r - phi @ beta
    assert float(np.sqrt(mass @ residual**2)) > 0.01


def test_the_cells_are_the_declared_ones() -> None:
    names = [cell.name for cell in study.CELLS]
    assert len(names) == len(set(names))
    assert {cell.law for cell in study.CELLS} <= set(study.LAWS)
    assert set(study.SMOKE_CELLS) <= set(names)
    assert study.CELL_BY_NAME["power_4k"].replicates == 1_000


def test_one_cheap_replication_runs() -> None:
    cell = study.CELL_BY_NAME["obs_1k"]
    frame = study.sample(cell.law, 400, np.random.default_rng([study.SMOKE_SEED, 0]))
    estimates = study.fit_cell(cell, frame)
    assert set(estimates) == {f"cate_projection[{t}]" for t in study.TERMS}
    assert all(np.isfinite(value).all() for value in estimates.values())
