"""The RM18 CD diagnostic: the exact bound, the analytic ratio, the reading rule, the harness."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_comparator_density import run as cd
from tests.studies.evidence.inference import Interval

ROWS, VALIDATION, READING = shared.part_paths(cd.HERE, cd.PART)


def _hermite(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for an expectation over one standard normal variable."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(points)
    return nodes, weights / np.sqrt(2.0 * np.pi)


def test_the_exact_bound_matches_gauss_hermite_quadrature() -> None:
    """``Var(D)`` over ``(W1, W2, W3, A)`` by a tensor product of Gauss-Hermite rules."""
    nodes, weights = _hermite(40)
    grid = np.meshgrid(nodes, nodes, nodes, nodes, indexing="ij")
    mass = np.einsum("i,j,k,l->ijkl", weights, weights, weights, weights).reshape(-1)
    latent = np.column_stack([axis.reshape(-1) for axis in grid[:3]])
    noise = grid[3].reshape(-1)
    mean = cd.DGP.dose_mean(latent)
    dose = mean + cd.DGP.dose_scale * noise
    ratio = np.exp(cd.DELTA * (dose - mean) / cd.DGP.dose_scale**2 - cd.DELTA**2 / 2.0)
    shift = cd.DGP.outcome_mean(latent, dose + cd.DELTA) - cd.DGP.outcome_mean(latent, dose)
    # E[(h - 1)^2 (Y - Q)^2 | A, W] = (h - 1)^2 noise^2, and the cross term vanishes.
    second = np.sum(mass * ((ratio - 1.0) ** 2 * cd.DGP.noise_scale**2 + shift**2))
    first = np.sum(mass * shift)
    assert math.sqrt(second - first**2) == pytest.approx(cd.SIGMA, abs=1e-6)
    assert pytest.approx(0.223973, abs=5e-7) == cd.SIGMA


def test_the_analytic_ratio_is_the_normal_tilt() -> None:
    covariates = np.random.default_rng(2).normal(size=(50, 3))
    density = cd.AnalyticDensity(covariates)
    dose = density.mean + np.random.default_rng(3).normal(size=50)
    ratio = density.density_at(dose - cd.DELTA) / density.density_at(dose)
    expected = np.exp(cd.DELTA * (dose - density.mean) - cd.DELTA**2 / 2.0)
    np.testing.assert_allclose(ratio, expected, rtol=1e-12)
    # The mutation: the opposite shift is a different ratio.
    wrong = density.density_at(dose + cd.DELTA) / density.density_at(dose)
    assert not np.allclose(wrong, expected, rtol=1e-3)


def test_the_committed_pair_reproduces_the_registered_excess() -> None:
    committed = cd.committed_rows()
    rows = committed.loc[committed["estimand"] == cd.ESTIMAND]
    bound = cd.excess(rows, "Cb", "L")
    assert shared.scaled_difference(bound["upper"], cd.REGISTERED_EXCESS[0]) <= shared.TOLERANCE
    assert (
        shared.scaled_difference(bound["resolution"], cd.REGISTERED_EXCESS[1]) <= shared.TOLERANCE
    )


def test_one_refit_reproduces_its_committed_rows() -> None:
    fitted = pd.DataFrame(cd.refit(0))
    validation = cd.validate(fitted, cap=1)
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()
    analytic = fitted.loc[fitted["implementation"] == cd.ANALYTIC].iloc[0]
    binned = fitted.loc[
        (fitted["implementation"] == cd.BINNED) & (fitted["estimand"] == cd.ESTIMAND)
    ].iloc[0]
    # Retargeting moves the estimate: the analytic arm is not the binned fit relabelled.
    assert analytic["estimate"] != binned["estimate"]


@pytest.mark.parametrize(
    ("binned", "analytic", "bound", "expected"),
    [
        (Interval(0.01, 0.05), Interval(-0.01, 0.02), 0.05, cd.DENSITY),
        (Interval(0.01, 0.05), Interval(-0.03, -0.01), 0.04, cd.DENSITY),
        (Interval(0.01, 0.05), Interval(-0.01, 0.02), 0.051, cd.UNRESOLVED),
        (Interval(0.01, 0.05), Interval(0.001, 0.02), 0.04, cd.TARGETING),
        (Interval(-0.05, -0.01), Interval(-0.01, 0.02), 0.04, cd.COMPARATOR_READING),
        (Interval(-0.01, 0.05), Interval(-0.01, 0.02), 0.04, cd.UNRESOLVED),
    ],
)
def test_the_reading_rule_and_its_mutations(
    binned: Interval, analytic: Interval, bound: float, expected: str
) -> None:
    assert cd.reading_label(binned, analytic, bound) == expected


def _arms(inflation: float) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    scale = cd.SIGMA / math.sqrt(cd.N)
    frames = []
    for arm, implementation in cd.ARMS.items():
        errors = scale * (1.0 + 0.02 * rng.normal(size=300))
        frames.append(
            pd.DataFrame(
                {
                    "implementation": implementation,
                    "replicate": np.arange(300),
                    "truth": 0.3,
                    "estimate": 0.3 + scale * rng.normal(size=300),
                    "inference_estimate": 0.0,
                    "std_error": errors * (inflation if arm == "Cb" else 1.0),
                    "covered": 1,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_an_inflated_binned_standard_error_moves_the_first_difference() -> None:
    assert cd.d_intervals(_arms(1.0))["Cb", "L"].contains(0.0)
    assert cd.d_intervals(_arms(1.08))["Cb", "L"].low > 0.0


@pytest.mark.skipif(not ROWS.exists(), reason="CD has not run")
def test_the_committed_reading_follows_from_the_committed_rows() -> None:
    rebuilt = cd.reading_table(shared.read_rows(ROWS), shared.read_rows(VALIDATION))
    pd.testing.assert_frame_equal(rebuilt, shared.read_rows(READING), check_dtype=False, rtol=1e-12)


@pytest.mark.skipif(not ROWS.exists(), reason="CD has not run")
def test_one_committed_analytic_row_retargets_again() -> None:
    rows = shared.read_rows(ROWS)
    committed = rows.loc[(rows["implementation"] == cd.ANALYTIC) & (rows["replicate"] == 0)].iloc[0]
    fresh = next(row for row in cd.refit(0) if row["implementation"] == cd.ANALYTIC)
    for column in ("estimate", "std_error"):
        assert shared.scaled_difference(fresh[column], committed[column]) <= shared.TOLERANCE
