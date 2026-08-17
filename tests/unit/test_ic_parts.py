"""The influence curve, split into the two terms it is made of.

``H(A,W){Y - Q*}`` and ``Q*(a,W) - Psi`` answer different questions, and the
estimator sums them in one expression.  Keeping them apart is what lets a diagnostic
say *why* an influence curve has a heavy tail: the residual term is inflated by
positivity (one unit with a huge inverse-propensity weight), the plug-in term by
genuine outcome heterogeneity, and only the first of those is helped by truncation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.datasets import GENERATORS
from cleverly.estimators import TMLE
from cleverly.estimators.targeting import build_submodel
from tests.conftest import binary_mean_parts, binary_means


def _pieces(result):  # type: ignore[no-untyped-def]
    submodel = build_submodel(
        result.data,
        result.nuisance,
        "mean",
        bounds=result.config.g_bounds,
        nuisance_bound=result.config.missingness_bound,
    )
    scaled = result.nuisance.scaler.scale(result.data.outcome)
    targeted = result.fluctuations["mean"].targeted
    return scaled, targeted, submodel


@pytest.fixture(scope="module")
def result():
    frame, _ = GENERATORS["nonlinear_ate"](n=400, seed=5)
    covariates = [c for c in frame.columns if c.startswith("W")]
    return (
        TMLE(outcome_learner="glm", treatment_learner="glm", n_folds=4, random_state=7)
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def test_the_parts_sum_to_the_whole(result) -> None:  # type: ignore[no-untyped-def]
    """Agreement to rounding, not bit-for-bit: the sum is bracketed differently."""
    scaled, targeted, submodel = _pieces(result)
    weights, observed = result.data.weights, result.data.observed
    _, ic_one, _, ic_zero = binary_means(scaled, targeted, submodel, weights, observed)
    parts_one, parts_zero = binary_mean_parts(scaled, targeted, submodel, weights, observed)
    np.testing.assert_allclose(parts_one.total, ic_one, rtol=0, atol=1e-15)
    np.testing.assert_allclose(parts_zero.total, ic_zero, rtol=0, atol=1e-15)


def test_the_shares_add_up_and_are_informative(result) -> None:  # type: ignore[no-untyped-def]
    scaled, targeted, submodel = _pieces(result)
    parts, _ = binary_mean_parts(
        scaled, targeted, submodel, result.data.weights, result.data.observed
    )
    shares = parts.shares()
    assert set(shares) == {"residual", "plugin"}
    assert all(value >= 0.0 for value in shares.values())
    assert max(shares.values()) < 1.5
    # Both terms carry real weight in an ordinary fit; if one were ~0 the decomposition
    # would be telling us nothing.  That was the stated claim and nothing above asserts
    # it -- a split of residual=1.0, plugin=0.0 passes every line of it.  Measured here
    # at 0.79 and 0.11, so a floor of 0.05 is well clear of both while still failing a
    # term that has collapsed.
    assert min(shares.values()) > 0.05


def test_the_residual_term_grows_as_truncation_is_loosened() -> None:
    """The claim the decomposition exists to support, on a controlled comparison.

    Comparing two *datasets* would not isolate overlap -- they differ in outcome
    scale and heterogeneity too, and the residual share can move either way. Holding
    the data and the nuisance fits fixed and only loosening the propensity bound
    changes exactly one thing: how large the inverse-propensity weights are allowed
    to get. The residual term must grow with them.
    """
    frame, _ = GENERATORS["weak_overlap"](n=400, seed=5)
    covariates = [c for c in frame.columns if c.startswith("W")]
    fit = (
        TMLE(outcome_learner="glm", treatment_learner="glm", n_folds=4, random_state=7)
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )

    def residual_share(bound: float) -> float:
        submodel = build_submodel(
            fit.data,
            fit.nuisance,
            "mean",
            bounds=(bound, 1.0 - bound),
            nuisance_bound=fit.config.missingness_bound,
        )
        parts, _ = binary_mean_parts(
            fit.nuisance.scaler.scale(fit.data.outcome),
            fit.fluctuations["mean"].targeted,
            submodel,
            fit.data.weights,
            fit.data.observed,
        )
        return parts.shares()["residual"]

    tight, loose = residual_share(0.10), residual_share(0.001)
    assert loose > tight, (
        f"loosening the propensity bound from 0.10 to 0.001 did not increase the "
        f"residual term's share of the variance ({tight:.3f} -> {loose:.3f}); either "
        "the decomposition is mislabelled or the clever covariate is not responding "
        "to truncation"
    )
