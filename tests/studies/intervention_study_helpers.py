"""Shared finite-law helpers for the intervention-family evidence studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from tests import discrete_law as law

INTERVENTION_CALIBRATION_REPLICATES = 4_000


def saturated_discrete_outcome() -> Pipeline:
    """The correctly specified outcome regression for the shared finite-support law.

    The design a learner receives is ``[A, W]``, and ``W`` carries the three covariate levels
    as the numbers 0, 1 and 2.  A logistic regression reading that column as a number states
    that the second level sits exactly halfway between the first and the third on the logit
    scale, which this law does not do.  One parameter per cell says nothing about the levels
    and is what the R comparators fit, so the comparison measures targeting rather than two
    unrelated regression pipelines.

    The misspecification this replaces was not visible in bias, because the studies supply the
    treatment mechanism exactly and a targeted estimator with a correct mechanism stays
    consistent whatever the outcome model says.  It was visible in the *influence curve*: the
    numeric-column fit inflated the standard error of every intervention-versus-natural-course
    contrast by five to six percent against the exact-law efficient influence curve, while the
    saturated fit reproduces it to within one.  A conservative interval is not a wrong answer,
    which is why only a comparison against an implementation that saturates its own design
    found it.

    Returns
    -------
    sklearn.pipeline.Pipeline
        One-hot indicators for ``A`` and ``W``, their two-way interactions, and a
        near-unregularised logistic regression.  The identically-zero product of two levels of
        the same variable is collinear and contributes nothing; ``C=1e6`` keeps the fit off the
        boundary without moving it.
    """
    return make_pipeline(
        OneHotEncoder(drop="first", sparse_output=False),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        LogisticRegression(C=1e6, max_iter=2_000),
    )


def sample_discrete(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows from a declared ``P(W, A, Y)`` array."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(law.SUPPORT), size=n, p=np.asarray(probs).reshape(-1))
    values = np.asarray(law.SUPPORT, dtype=float)[cells]
    return pd.DataFrame({"W": values[:, 0], "A": values[:, 1], "Y": values[:, 2]})


def truths(probs: np.ndarray, estimands: Sequence[str]) -> dict[str, float]:
    """Evaluate the independent finite-law oracle for each estimand."""
    return {name: float(law.functional(probs, name)) for name in estimands}


def probabilities(q: np.ndarray, *, g: np.ndarray = law.G) -> np.ndarray:
    """Build ``P(W, A, Y)`` from the shared baseline law and supplied nuisances."""
    out = np.empty_like(law.PROBS)
    for w, a, y in law.SUPPORT:
        arm = g[w] if a == 1 else 1.0 - g[w]
        outcome = q[w, a] if y == 1 else 1.0 - q[w, a]
        out[w, a, y] = law.P_W[w] * arm * outcome
    return out


def initial_regime_means(result: Any) -> Mapping[str, float]:
    """Project the untargeted arm regressions through the declared regime densities."""
    regimes = result.nuisance.regimes
    if regimes is None:  # pragma: no cover - a study contract guard
        raise AssertionError("a regime study fit did not retain its evaluated regimes")
    arms = np.column_stack(
        [
            result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[arm])
            for arm in result.data.arm_codes
        ]
    )
    weights = np.asarray(result.data.weights, dtype=float)
    means: dict[str, float] = {}
    for code, label in regimes.labels.items():
        mixture = np.einsum("ij,ij->i", regimes.column(code), arms)
        means[label] = float(np.average(mixture, weights=weights))
    return means


def initial_regime_estimates(result: Any) -> dict[str, float]:
    """Return untargeted means and contrasts under their public parameter names."""
    means = dict(initial_regime_means(result))
    reference = result.nuisance.regimes.label(result.nuisance.regimes.reference)
    out = {f"ey_regime[{label}]": value for label, value in means.items()}
    for label, value in means.items():
        if label != reference:
            out[f"ate_regime[{label} vs {reference}]"] = value - means[reference]
    return out


def incremental_estimates(
    result: Any,
    *,
    targeted_outcome: bool = False,
    targeted_mechanism: bool = False,
) -> dict[str, float]:
    """Project one incremental fit at a declared targeting boundary.

    The two switches expose the two load-bearing halves of incremental targeting without
    duplicating its projection arithmetic in the primary and property studies.
    """
    tilts = result.nuisance.incremental
    if tilts is None:  # pragma: no cover - a study contract guard
        raise AssertionError("an incremental study fit did not retain its evaluated tilts")
    fluctuation = result.fluctuations["ipsi"]
    outcome = fluctuation.targeted if targeted_outcome else result.nuisance.outcome
    if targeted_mechanism:
        if fluctuation.mechanism is None:  # pragma: no cover - a study contract guard
            raise AssertionError("an incremental fit did not target its treatment mechanism")
        tilts = tilts.at(fluctuation.mechanism.propensity)
    arms = np.column_stack(
        [result.nuisance.scaler.unscale_levels(outcome.arms[arm]) for arm in result.data.arm_codes]
    )
    mixtures = np.einsum("ikr,ik->ir", tilts.values, arms)
    weights = np.asarray(result.data.weights, dtype=float)
    means = {
        label: float(np.average(mixtures[:, int(code)], weights=weights))
        for code, label in tilts.labels.items()
    }
    reference = tilts.label(tilts.reference)
    out = {f"ey_ipsi[{label}]": value for label, value in means.items()}
    for label, value in means.items():
        if label != reference:
            out[f"ate_ipsi[{label} vs {reference}]"] = value - means[reference]
    return out


def primary_rows(
    *,
    result: Any,
    reference: Mapping[str, float],
    implementation: str,
    scenario: str,
    replicate: int,
    initials: Mapping[str, float],
    estimands: Sequence[str],
) -> list[dict[str, Any]]:
    """Convert one fit to the registered primary-replication schema."""
    rows: list[dict[str, Any]] = []
    for name in estimands:
        estimate = result[name]
        low, high = estimate.ci
        truth = float(reference[name])
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.data.n,
                "estimand": name,
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": float(initials[name]),
            }
        )
    return rows


def efficiency_sd(probs: np.ndarray, estimand: str) -> float:
    """Standard deviation of the oracle influence curve on the finite law."""
    base = np.asarray(probs, dtype=complex)
    step = 1e-30
    curve = np.empty(len(law.SUPPORT))
    for point, support in enumerate(law.SUPPORT):
        mass = np.zeros_like(base)
        mass[support] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        curve[point] = np.imag(law.functional(perturbed, estimand)) / step
    return float(np.sqrt(np.sum(np.asarray(probs).reshape(-1) * curve**2)))
