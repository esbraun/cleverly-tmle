"""The stratum-conditional influence curves of ``TMLE(strata=)``, against their analytic form.

A stratum parameter is the marginal parameter of the law given ``V = s``.  Its gradient in the
full model is ``I(V = s) D_s / P(V = s)``, where ``D_s`` is the marginal efficient influence
function computed inside the stratum.  For the ATT and the ATC, ``D_s`` divides by the
within-stratum arm share ``P(A = a | V = s)``.

The fit below starts from a deliberately wrong outcome regression, so every fluctuation moves
and the curves are evaluated at targeted predictions that differ from the initial ones.  The
analytic curves are written here from the targeted predictions the fit reports, the known
propensity and the sample's own stratum and arm shares.  Two deliberate mutations of that
analytic form must each fail: the marginal arm share in place of the within-stratum one, and
the stratum mass replaced by one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin

from tests.studies import stratified_law as law
from tests.studies.canonical_stratified_tmle import fit_cleverly

N = 3_000
SEED = 20261013
STEMS = ("ey[1]", "ey[0]", "ate", "att", "atc", "par")


class KnownTreatment(BaseEstimator, ClassifierMixin):
    """The law's propensity, read from the ``(W, V)`` treatment design."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> KnownTreatment:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        p = law.propensity(values[:, 0], values[:, 1])
        return np.column_stack([1.0 - p, p])


class PerturbedOutcome(BaseEstimator, ClassifierMixin):
    """The law's outcome regression, shifted on the logit scale so targeting has work to do."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> PerturbedOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        a, w, v = values[:, 0], values[:, 1], values[:, 2]
        p = expit(logit(law.outcome(a, w, v)) + 0.35 + 0.3 * a - 0.25 * v)
        return np.column_stack([1.0 - p, p])


@pytest.fixture(scope="module")
def fitted() -> tuple[Any, pd.DataFrame]:
    frame = law.sample(N, SEED)
    result = fit_cleverly(
        frame,
        estimands=("ey", "ate", "att", "atc", "par"),
        outcome_learner=PerturbedOutcome(),
        treatment_learner=KnownTreatment(),
    )
    return result, frame


def analytic(
    result: Any,
    frame: pd.DataFrame,
    stem: str,
    stratum: int,
    *,
    share: str = "stratum",
    mass: str = "empirical",
) -> tuple[float, np.ndarray]:
    """The analytic stratum parameter and its full-sample curve.

    ``share="marginal"`` and ``mass="one"`` are the two deliberate mutations.
    """
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    v = frame["V"].to_numpy()
    g = law.propensity(frame["W"].to_numpy(dtype=float), v.astype(float))
    inside = v == stratum
    weight = float(inside.mean()) if mass == "empirical" else 1.0

    mean = result.fluctuations["mean"].targeted
    q1, q0 = mean.arms[1.0], mean.arms[0.0]
    psi1 = float(q1[inside].mean())
    psi0 = float(q0[inside].mean())
    d1 = a / g * (y - q1) + q1 - psi1
    d0 = (1.0 - a) / (1.0 - g) * (y - q0) + q0 - psi0

    def treated_share(arm: np.ndarray) -> float:
        return float(arm[inside].mean() if share == "stratum" else arm.mean())

    if stem == "ey[1]":
        value, curve = psi1, d1
    elif stem == "ey[0]":
        value, curve = psi0, d0
    elif stem == "ate":
        value, curve = psi1 - psi0, d1 - d0
    elif stem == "att":
        targeted = result.fluctuations["att"].targeted
        t1, t0 = targeted.arms[1.0], targeted.arms[0.0]
        p = treated_share(a)
        value = float(np.mean((a * (t1 - t0))[inside])) / p
        curve = (
            a / p * (y - t1)
            - (1.0 - a) * g / (p * (1.0 - g)) * (y - t0)
            + a / p * (t1 - t0 - value)
        )
    elif stem == "atc":
        targeted = result.fluctuations["atc"].targeted
        t1, t0 = targeted.arms[1.0], targeted.arms[0.0]
        u = treated_share(1.0 - a)
        value = float(np.mean(((1.0 - a) * (t1 - t0))[inside])) / u
        curve = (
            a * (1.0 - g) / (u * g) * (y - t1)
            - (1.0 - a) / u * (y - t0)
            + (1.0 - a) / u * (t1 - t0 - value)
        )
    elif stem == "par":
        observed = float(y[inside].mean())
        value = observed - psi0
        curve = (y - observed) - d0
    else:  # pragma: no cover - declaration guard
        raise KeyError(stem)
    return value, np.where(inside, curve / weight, 0.0)


def test_every_fluctuation_moves(fitted: tuple[Any, pd.DataFrame]) -> None:
    """The nonzero witness: no targeted prediction equals the initial one."""
    result, _ = fitted
    for group in ("mean", "att", "atc"):
        epsilon = np.asarray(result.fluctuations[group].epsilon)
        assert epsilon.shape == (len(law.STRATA) * (2 if group == "mean" else 1),)
        assert np.min(np.abs(epsilon)) > 1e-3, (group, epsilon)


@pytest.mark.parametrize("stratum", law.STRATA)
@pytest.mark.parametrize("stem", STEMS)
def test_each_stratum_curve_is_the_analytic_gradient(
    fitted: tuple[Any, pd.DataFrame], stem: str, stratum: int
) -> None:
    result, frame = fitted
    estimate = result[law.stratum_name(stem, stratum)]
    value, curve = analytic(result, frame, stem, stratum)
    assert estimate.psi == pytest.approx(value, abs=1e-12)
    np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("stratum", law.STRATA)
@pytest.mark.parametrize("stem", ("att", "atc"))
def test_the_marginal_arm_share_is_detected(
    fitted: tuple[Any, pd.DataFrame], stem: str, stratum: int
) -> None:
    """Mutation control: the marginal share in the ATT or ATC block fails the comparison."""
    result, frame = fitted
    estimate = result[law.stratum_name(stem, stratum)]
    value, curve = analytic(result, frame, stem, stratum, share="marginal")
    moved = max(abs(estimate.psi - value), float(np.max(np.abs(estimate.influence_curve - curve))))
    assert moved > 1e-3


@pytest.mark.parametrize("stratum", law.STRATA)
@pytest.mark.parametrize("stem", STEMS)
def test_a_unit_stratum_mass_is_detected(
    fitted: tuple[Any, pd.DataFrame], stem: str, stratum: int
) -> None:
    """Mutation control: dropping the ``1 / P_n(V = s)`` embedding fails the comparison."""
    result, frame = fitted
    estimate = result[law.stratum_name(stem, stratum)]
    _, curve = analytic(result, frame, stem, stratum, mass="one")
    assert float(np.max(np.abs(estimate.influence_curve - curve))) > 1e-3


def test_the_marginal_estimate_is_the_stratum_mixture(fitted: tuple[Any, pd.DataFrame]) -> None:
    """The marginal ATE is the empirical mixture of the three stratum ATEs."""
    result, frame = fitted
    v = frame["V"].to_numpy()
    mixture = sum(
        float(np.mean(v == stratum)) * result[law.stratum_name("ate", stratum)].psi
        for stratum in law.STRATA
    )
    assert result["ate"].psi == pytest.approx(mixture, abs=1e-12)
