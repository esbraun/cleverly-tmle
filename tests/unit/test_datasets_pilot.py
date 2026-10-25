"""The randomized-pilot law: the exact truth, the recorded offer chances, and the draw.

:func:`cleverly.datasets.randomized_pilot_truth` integrates the success probability by quadrature.
Two Monte Carlo checks hold it: the success probability averaged over drawn covariates (no outcome
noise), and the Horvitz-Thompson means of the drawn outcomes under the recorded chances, which also
check the outcome and offer draws. Each tolerance is four Monte Carlo standard errors.

An exact-law check is blind to a term that vanishes at the truth, so two deliberately wrong laws
must fail the same comparison: one that drops the high-risk part of the offer's effect, and one
that drops the curvature in ``discharge_risk``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import PILOT_OFFER_CHANCES, make_randomized_pilot, randomized_pilot_truth
from cleverly.datasets import pilot as law
from cleverly.utils.bounds import expit

MONTE_CARLO_N = 1_000_000


def _oracle_means(n: int, seed: int) -> dict[str, tuple[float, float]]:
    """Each arm mean of the success probability over drawn covariates, with its standard error."""
    rng = np.random.default_rng(seed)
    risk, utilization = rng.standard_normal((2, n))
    out = {}
    for arm in (0, 1):
        values = law._success_probability(arm, risk, utilization)
        out[f"ey{arm}"] = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(n)))
    difference = law._success_probability(1, risk, utilization) - law._success_probability(
        0, risk, utilization
    )
    out["ate"] = (float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(n)))
    return out


def test_the_truth_matches_the_success_probability_over_drawn_covariates() -> None:
    truth = randomized_pilot_truth()
    for key, (mean, error) in _oracle_means(MONTE_CARLO_N, 20261025).items():
        assert abs(truth[key] - mean) < 4.0 * error, key


def test_the_truth_matches_the_weighted_means_of_drawn_outcomes() -> None:
    frame, truth = make_randomized_pilot(MONTE_CARLO_N, seed=20261026)
    offer = frame["transition_navigation"].to_numpy()
    success = frame["transition_success"].to_numpy()
    terms = {
        "ey1": offer * success / frame["chance_offer"].to_numpy(),
        "ey0": (1.0 - offer) * success / frame["chance_no_offer"].to_numpy(),
    }
    terms["ate"] = terms["ey1"] - terms["ey0"]
    for key, values in terms.items():
        error = values.std(ddof=1) / np.sqrt(MONTE_CARLO_N)
        assert abs(truth[key] - values.mean()) < 4.0 * error, key


def _no_high_risk_gain(offer: Any, risk: Any, utilization: Any) -> Any:
    r = np.asarray(risk, dtype=float)
    high = (r > 0.0).astype(float)
    return expit(-0.2 + 0.4 * offer - 1.2 * high - 0.3 * r + 0.5 * r**2 - 0.5 * utilization)


def _no_curvature(offer: Any, risk: Any, utilization: Any) -> Any:
    r = np.asarray(risk, dtype=float)
    high = (r > 0.0).astype(float)
    return expit(-0.2 + offer * (0.4 + 0.6 * high) - 1.2 * high - 0.3 * r - 0.5 * utilization)


@pytest.mark.parametrize("wrong", [_no_high_risk_gain, _no_curvature])
def test_a_wrong_law_fails_the_monte_carlo_comparison(
    wrong: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = _oracle_means(MONTE_CARLO_N, 20261025)
    monkeypatch.setattr(law, "_success_probability", wrong)
    mutated = randomized_pilot_truth()
    misses = [abs(mutated[key] - mean) > 4.0 * error for key, (mean, error) in observed.items()]
    assert any(misses)


def test_the_draw_records_the_design_chances() -> None:
    frame, _ = make_randomized_pilot(4_000, seed=5)
    high = frame["high_risk"].to_numpy()
    assert np.array_equal(high, (frame["discharge_risk"].to_numpy() > 0.0).astype(float))
    chance = np.where(high == 1.0, PILOT_OFFER_CHANCES[1], PILOT_OFFER_CHANCES[0])
    assert np.array_equal(frame["chance_offer"].to_numpy(), chance)
    assert np.allclose(frame["chance_no_offer"].to_numpy() + chance, 1.0)
    for group, expected in PILOT_OFFER_CHANCES.items():
        offered = frame["transition_navigation"].to_numpy()[high == group]
        assert abs(offered.mean() - expected) < 4.0 * np.sqrt(
            expected * (1 - expected) / offered.size
        )


def test_the_same_seed_gives_the_same_rows() -> None:
    first, _ = make_randomized_pilot(200, seed=11)
    second, _ = make_randomized_pilot(200, seed=11)
    assert first.equals(second)
