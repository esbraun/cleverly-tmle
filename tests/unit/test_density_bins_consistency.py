"""The default bin count grows with n, so the binned density ratio is consistent.

A modified treatment policy reads the ratio ``g(a - delta) / g(a)`` from a binned density.  At a
fixed bin count the ratio's error in the tail bins does not shrink as ``n`` grows: the
``longitudinal-mtp`` study found it.  The witness here gives the binned density its exact bin
probabilities, so the only error left is the binning, and checks that the tail error shrinks
from ``n = 1,000`` to ``n = 100,000`` under the default rule.  The mutation holds the count at
20 and the witness fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from cleverly.learners import density as density_module
from cleverly.learners.density import ConditionalDensity, bin_edges, default_density_bins

DELTA = 0.5
#: Tail points of the standard normal dose between its 95% and 99% quantiles.
TAIL = np.linspace(norm.ppf(0.95), norm.ppf(0.99), 50)


def _tail_error(n: int, seed: int = 20261071) -> float:
    """Mean |binned ratio - true ratio| on the tail points, with exact bin probabilities."""
    dose = np.random.default_rng(seed).standard_normal(n)
    edges = bin_edges(dose, density_module.default_density_bins(n))
    probabilities = np.diff(norm.cdf(edges))
    probabilities = probabilities / probabilities.sum()
    rows = np.tile(probabilities, (TAIL.size, 1))
    density = ConditionalDensity(rows, edges)
    binned = density.density_at(TAIL - DELTA) / density.density_at(TAIL)
    true = norm.pdf(TAIL - DELTA) / norm.pdf(TAIL)
    return float(np.mean(np.abs(binned - true)))


def test_the_default_count_follows_the_declared_rule() -> None:
    assert default_density_bins(500) == 20
    assert default_density_bins(1_000) == 20
    assert default_density_bins(8_000) == 40
    assert default_density_bins(1_000_000) == 200
    counts = [default_density_bins(n) for n in (10**3, 10**4, 10**5, 10**6)]
    assert counts == sorted(counts) and counts[-1] > counts[0]


def test_the_tail_error_shrinks_with_n() -> None:
    small, large = _tail_error(1_000), _tail_error(100_000)
    assert large < 0.5 * small


def test_a_fixed_count_leaves_the_tail_error_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: 20 bins at every n.  The witness above then fails."""
    monkeypatch.setattr(density_module, "default_density_bins", lambda n: 20)
    small, large = _tail_error(1_000), _tail_error(100_000)
    assert large > 0.7 * small


def test_an_explicit_count_is_honoured() -> None:
    """``density_bins=`` set by a user is passed through unchanged."""
    from sklearn.linear_model import LogisticRegression

    from cleverly.learners import Folds
    from cleverly.learners.density import fit_conditional_density

    rng = np.random.default_rng(3)
    w = rng.standard_normal((400, 1))
    a = w[:, 0] + rng.standard_normal(400)
    folds = Folds.single(400)
    fitted, _ = fit_conditional_density(
        LogisticRegression(max_iter=500), w, a, np.ones(400), folds, n_bins=7
    )
    assert fitted.edges.size == 8
    default, _ = fit_conditional_density(
        LogisticRegression(max_iter=500), w, a, np.ones(400), folds
    )
    assert default.edges.size == default_density_bins(400) + 1
