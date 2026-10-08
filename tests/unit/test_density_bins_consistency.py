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

from cleverly.exceptions import MethodConfigurationError
from cleverly.learners import density as density_module
from cleverly.learners.density import (
    ConditionalDensity,
    bin_edges,
    default_density_bins,
    hazard_design_bytes,
)

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


def _dose_fit(n: int, **settings: object) -> object:
    from sklearn.linear_model import LogisticRegression

    from cleverly.learners import Folds
    from cleverly.learners.density import fit_conditional_density

    rng = np.random.default_rng(5)
    w = rng.standard_normal((n, 1))
    a = w[:, 0] + rng.standard_normal(n)
    return fit_conditional_density(
        LogisticRegression(max_iter=500), w, a, np.ones(n), Folds.single(n), **settings
    )


def test_the_default_grows_above_one_thousand_rows_through_the_fit() -> None:
    """The mutation: ``n_bins = 20`` in ``fit_conditional_density`` fails both checks."""
    _, diagnostics = _dose_fit(2_000)
    assert diagnostics.n_bins == default_density_bins(2_000) == 26


def test_the_default_grows_above_one_thousand_rows_through_a_tmle_fit() -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from cleverly.estimators import TMLE
    from cleverly.interventions import Shift

    rng = np.random.default_rng(0)
    n = 2_000
    w = rng.standard_normal(n)
    a = w + rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(a + w) / 3.0))).astype(float)
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    result = (
        TMLE(
            policies=[Shift(0.5, cap=float(a.max()))],
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(max_iter=500),
            n_folds=2,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )
    assert result.nuisance.diagnostics["density"].n_bins == 26


def test_a_design_larger_than_the_free_memory_is_refused_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal names the design's size and a bin count whose design fits."""
    monkeypatch.setattr(density_module, "available_memory", lambda: 4 * 1024**2)
    with pytest.raises(MethodConfigurationError, match=r"needs 0\.\d GiB .* Pass density_bins="):
        _dose_fit(2_000)
    monkeypatch.setattr(density_module, "available_memory", lambda: None)
    _, diagnostics = _dose_fit(2_000)
    assert diagnostics.n_bins == 26


def test_the_size_estimate_matches_the_design_the_fit_builds() -> None:
    """Equal-mass bins: the estimate is within 5% of the exact table, and grows as n^(5/3)."""
    rng = np.random.default_rng(6)
    n, k = 4_000, default_density_bins(4_000)
    a = rng.standard_normal(n)
    bins = np.clip(np.digitize(a, bin_edges(a, k)) - 1, 0, k - 1)
    design, _, _ = density_module._long_expansion(rng.standard_normal((n, 2)), bins, k - 1)
    assert hazard_design_bytes(n, k, 2) == pytest.approx(design.nbytes, rel=0.05)
    ratio = hazard_design_bytes(80_000, default_density_bins(80_000), 2) / hazard_design_bytes(
        10_000, default_density_bins(10_000), 2
    )
    assert ratio == pytest.approx(8.0 ** (5.0 / 3.0), rel=0.15)
