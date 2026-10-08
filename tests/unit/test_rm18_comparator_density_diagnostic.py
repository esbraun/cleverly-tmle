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
    upper, resolution = cd.registered_excess()
    assert shared.scaled_difference(bound["upper"], upper) <= shared.TOLERANCE
    assert shared.scaled_difference(bound["resolution"], resolution) <= shared.TOLERANCE
    # The declared values, as the roadmap prints them.
    assert (round(upper, 6), round(resolution, 6)) == (0.064762, 0.043925)


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


def _arms(
    inflation: float = 1.08, analytic_spread: float = 1.0, comparator: float = 1.0, count: int = 800
) -> pd.DataFrame:
    """Three arms on the same draws, as the registered pairing has them.

    Each arm reports an SE equal to the spread of the estimates, times ``inflation`` for Cb and
    ``comparator`` for L.  Ca's estimates spread ``analytic_spread`` times as far.
    """
    rng = np.random.default_rng(9)
    scale = cd.SIGMA / math.sqrt(cd.N)
    noise = rng.normal(size=count)
    noise = noise / noise.std(ddof=1) * scale
    reported = scale * (1.0 + 0.01 * rng.normal(size=count))
    factors = {"Cb": (inflation, 1.0), "Ca": (1.0, analytic_spread), "L": (comparator, 1.0)}
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "implementation": implementation,
                    "replicate": np.arange(count),
                    "truth": 0.3,
                    "estimate": 0.3 + noise * factors[arm][1],
                    "inference_estimate": 0.3 + noise * factors[arm][1],
                    "std_error": reported * factors[arm][0],
                    "covered": 1,
                }
            )
            for arm, implementation in cd.ARMS.items()
        ],
        ignore_index=True,
    )


def _holding() -> pd.DataFrame:
    return shared.validation_frame(
        [shared.validation_row(cd.PART, "Cb refit rows", (True, 0.0, 1))]
    )


def _reading(table: pd.DataFrame) -> str:
    return str(table.loc[table["statistic"] == "reading", "result"].iloc[0])


def test_the_table_reads_the_ca_l_bound_and_mutations_move_it() -> None:
    # Cb's SE is 8% too wide, so its own excess over L is above 0.05. The reading still names
    # the density, which it can only do by reading the (Ca, L) bound.
    assert _reading(cd.reading_table(_arms(), _holding())) == cd.DENSITY
    # Ca's spread is off while its SE is not: D_Ca - D_L still covers 0, and the (Ca, L) bound
    # leaves the margin, so the density reading fails.
    assert _reading(cd.reading_table(_arms(analytic_spread=1.10), _holding())) == cd.UNRESOLVED
    assert (
        _reading(cd.reading_table(_arms(inflation=1.0, comparator=1.08), _holding()))
        == cd.COMPARATOR_READING
    )
    assert _reading(cd.reading_table(_arms(count=799), _holding())) == shared.SMOKE


def test_an_inflated_binned_standard_error_moves_the_first_difference() -> None:
    assert cd.d_intervals(_arms(1.0))["Cb", "L"].contains(0.0)
    assert cd.d_intervals(_arms(1.08))["Cb", "L"].low > 0.0


def test_the_committed_reading_follows_from_the_committed_rows() -> None:
    rows = shared.read_rows(ROWS)
    assert (
        rows.groupby("implementation").size().eq(800).all()
        and rows["implementation"].nunique() == 3
    )
    rebuilt = cd.table(cd.PART, cd.HERE, 1)
    pd.testing.assert_frame_equal(
        shared.as_committed(rebuilt), shared.read_rows(READING), check_dtype=False, rtol=1e-12
    )
    assert shared.SMOKE not in set(rebuilt["result"])


def test_one_committed_analytic_row_retargets_again() -> None:
    rows = shared.read_rows(ROWS)
    committed = rows.loc[(rows["implementation"] == cd.ANALYTIC) & (rows["replicate"] == 0)].iloc[0]
    fresh = next(row for row in cd.refit(0) if row["implementation"] == cd.ANALYTIC)
    for column in ("estimate", "std_error"):
        assert shared.scaled_difference(fresh[column], committed[column]) <= shared.TOLERANCE
