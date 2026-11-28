"""The simulated navigation offer with exact event times (``cleverly.datasets.exact_times``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly.datasets import (
    EXACT_ARMS,
    EXACT_CAUSES,
    exact_readmission_protocol,
    exact_readmission_truth,
    make_exact_readmission,
)
from cleverly.datasets import exact_times as exact_module

HORIZONS = (1.5, 3.0)
TRUTH = exact_readmission_truth(HORIZONS, restricted_horizons=(3.0,))


def _inline_draw(n: int, seed: int) -> pd.DataFrame:
    """The construction ``docs/examples/plots.ipynb`` used inline before this module existed."""
    event_rng = np.random.default_rng(seed)
    risk_score = event_rng.normal(size=n)
    offered = event_rng.binomial(1, 1 / (1 + np.exp(-risk_score)))
    relapse_rate = np.exp(-1.5 + 0.5 * risk_score - offered)
    death_rate = np.exp(-2.0 - 0.5 * risk_score - 0.5 * offered)
    event_time = event_rng.exponential(1 / (relapse_rate + death_rate))
    cause = np.where(event_rng.uniform(size=n) < relapse_rate / (relapse_rate + death_rate), 1, 2)
    dropout = event_rng.exponential(1 / 0.15, size=n)
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, dropout),
            "event": np.where(event_time <= dropout, cause, 0),
            "offered": offered,
            "risk_score": risk_score,
        }
    )


@pytest.mark.parametrize("seed", [3, 2026])
def test_the_draw_is_the_inline_plots_construction_bit_for_bit(seed: int) -> None:
    frame, _ = make_exact_readmission(1_000, seed=seed, backend="pandas")
    pd.testing.assert_frame_equal(frame, _inline_draw(1_000, seed), check_exact=True)


def _potential_outcomes(arm: int, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Event times and causes with the offer set to ``arm`` and no one leaving follow-up."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=draws)
    h1, h2 = np.exp(-1.5 + 0.5 * x - arm), np.exp(-2.0 - 0.5 * x - 0.5 * arm)
    event_time = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=draws) < h1 / (h1 + h2), 1, 2)
    return event_time, cause


def test_every_truth_matches_a_monte_carlo_draw_of_the_potential_event_times() -> None:
    draws = 2_000_000
    for arm, code in EXACT_ARMS.items():
        event_time, cause = _potential_outcomes(code, draws, seed=10 + code)
        for cause_code, name in EXACT_CAUSES.items():
            for t in HORIZONS:
                hit = (cause == cause_code) & (event_time <= t)
                se = hit.std() / np.sqrt(draws)
                assert abs(hit.mean() - TRUTH[f"cif_regimen[{arm}, {name} @ t={t:g}]"]) < 4 * se
            lost = np.where(cause == cause_code, np.clip(3.0 - event_time, 0.0, None), 0.0)
            se = lost.std() / np.sqrt(draws)
            assert abs(lost.mean() - TRUTH[f"rmtl_regimen[{arm}, {name} @ t=3]"]) < 4 * se
        free = np.minimum(event_time, 3.0)
        se = free.std() / np.sqrt(draws)
        assert abs(free.mean() - TRUTH[f"rmst_regimen[{arm} @ t=3]"]) < 4 * se


def test_the_death_free_risk_matches_a_monte_carlo_draw() -> None:
    hypothetical = exact_readmission_truth(HORIZONS, death_prevented=True)
    draws = 2_000_000
    rng = np.random.default_rng(5)
    for arm, code in EXACT_ARMS.items():
        x = rng.normal(size=draws)
        readmission_time = rng.exponential(1 / np.exp(-1.5 + 0.5 * x - code))
        for t in HORIZONS:
            hit = readmission_time <= t
            se = hit.std() / np.sqrt(draws)
            assert abs(hit.mean() - hypothetical[f"risk_regimen[{arm} @ t={t:g}]"]) < 4 * se
    for t in HORIZONS:
        treated = hypothetical[f"risk_regimen[offered @ t={t:g}]"]
        control = hypothetical[f"risk_regimen[not offered @ t={t:g}]"]
        assert hypothetical[f"ate_regimen[offered vs not offered @ t={t:g}]"] == treated - control
        assert hypothetical[f"rr_regimen[offered vs not offered @ t={t:g}]"] == treated / control
        # Removing death removes a competing event, so readmission can only become more likely.
        for arm in EXACT_ARMS:
            assert (
                hypothetical[f"risk_regimen[{arm} @ t={t:g}]"]
                > TRUTH[f"cif_regimen[{arm}, readmission @ t={t:g}]"] + 0.005
            )


def test_the_quadrature_equals_the_closed_form_at_a_fixed_risk_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the risk score held at 0 the average is the conditional quantity itself.
    monkeypatch.setattr(exact_module, "_average", lambda conditional: conditional(0.0))
    truth = exact_readmission_truth((2.0,), restricted_horizons=(2.0,))
    h1, h2 = np.exp(-1.5), np.exp(-2.0)
    h = h1 + h2
    assert truth["cif_regimen[not offered, readmission @ t=2]"] == pytest.approx(
        h1 / h * (1 - np.exp(-2 * h)), abs=1e-12
    )
    assert truth["rmtl_regimen[not offered, death @ t=2]"] == pytest.approx(
        h2 / h * (2 - (1 - np.exp(-2 * h)) / h), abs=1e-12
    )
    assert truth["rmst_regimen[not offered @ t=2]"] == pytest.approx(
        (1 - np.exp(-2 * h)) / h, abs=1e-12
    )


def test_the_quadrature_averages_over_the_standard_normal_risk_score() -> None:
    # E exp(X / 2) = exp(1 / 8) for X ~ N(0, 1): a witness the fixed-score test cannot see.
    assert exact_module._average(lambda x: float(np.exp(0.5 * x))) == pytest.approx(
        np.exp(0.125), abs=1e-10
    )


def test_the_contrasts_are_built_from_the_risks() -> None:
    for name in EXACT_CAUSES.values():
        for t in HORIZONS:
            treated = TRUTH[f"cif_regimen[offered, {name} @ t={t:g}]"]
            control = TRUTH[f"cif_regimen[not offered, {name} @ t={t:g}]"]
            assert TRUTH[f"ate_regimen[offered vs not offered, {name} @ t={t:g}]"] == (
                treated - control
            )
            assert TRUTH[f"rr_regimen[offered vs not offered, {name} @ t={t:g}]"] == (
                treated / control
            )
    for arm in EXACT_ARMS:
        lost = sum(TRUTH[f"rmtl_regimen[{arm}, {name} @ t=3]"] for name in EXACT_CAUSES.values())
        assert TRUTH[f"rmst_regimen[{arm} @ t=3]"] == pytest.approx(3.0 - lost, abs=1e-14)


def test_swapping_the_two_hazards_moves_the_untreated_readmission_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under the offer the two hazards are mirror images in the risk score, so a swap of the two
    # causes leaves the offered arm unchanged; the arm without the offer witnesses it.
    original = exact_module._rates

    def swapped(arm: float, x: object) -> tuple[object, object]:
        h1, h2 = original(arm, x)
        return h2, h1

    monkeypatch.setattr(exact_module, "_rates", swapped)
    moved = exact_readmission_truth((3.0,))
    assert (
        abs(
            moved["cif_regimen[not offered, readmission @ t=3]"]
            - TRUTH["cif_regimen[not offered, readmission @ t=3]"]
        )
        > 0.05
    )


def test_the_truth_returned_with_a_draw_is_the_truth_at_its_times() -> None:
    _, truth = make_exact_readmission(50, seed=0, horizons=(1.0, 2.0), restricted_horizons=(2.0,))
    assert truth == exact_readmission_truth((1.0, 2.0), restricted_horizons=(2.0,))


@pytest.mark.parametrize(
    ("horizons", "restricted"),
    [
        ((), ()),
        ((0.0, 1.0), ()),
        ((2.0, 1.0), ()),
        ((1.0, 1.0), ()),
        ((1.0, 2.0), (1.5,)),
        ((1.0, float("inf")), ()),
        ((float("nan"),), ()),
    ],
)
def test_malformed_times_are_refused(
    horizons: tuple[float, ...], restricted: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        exact_readmission_truth(horizons, restricted_horizons=restricted)


def test_the_death_free_truth_has_no_restricted_means() -> None:
    with pytest.raises(ValueError, match="restricted"):
        exact_readmission_truth((3.0,), restricted_horizons=(3.0,), death_prevented=True)


def test_a_draw_needs_a_patient() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_exact_readmission(0)


def test_the_protocol_names_both_plans_and_both_events() -> None:
    protocol = exact_readmission_protocol()
    assert len(protocol.treatment_strategies) == len(EXACT_ARMS)
    assert "death before any readmission" in protocol.outcome
