"""The team-run discharge program: exact truth, observed layout, and the team component.

:func:`cleverly.datasets.team_readmission_truth` enumerates ``lives_alone`` and integrates
``discharge_risk`` by Gauss--Hermite.  A Monte Carlo draw of the intervened law, team component
included, checks it plan by plan.  Exact-law checks are blind to a term that vanishes at the
truth, so two wrong truths must fail the same comparison: one that drops the readmissions before
day 30, and one that leaves the call out of the day-90 risk.

The team component must leave the truth alone and still correlate teammates.  The tests check
both, and a draw with the component switched off is the control that shows the correlation comes
from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import (
    make_team_readmission,
    team_readmission_protocol,
    team_readmission_truth,
)
from cleverly.datasets import readmission as law

#: Draws in the Monte Carlo comparison, in teams of 40 that share one sign.
MONTE_CARLO_N = 1_000_000
TEAM = 40
#: The floor added to four Monte Carlo standard errors: the quadrature error of the truth.
QUADRATURE_FLOOR = 1e-9

PLANS = {"always": (1.0, 1.0), "never": (0.0, 0.0), "offer_only": (1.0, 0.0)}


def _intervened(offer: float, call: float, n: int, seed: int, delta: float) -> np.ndarray:
    """Draw the day-90 readmission of ``n`` patients, every one on the plan ``(offer, call)``.

    Teams of 40, so the team component enters as it does in the generator.
    """
    rng = np.random.default_rng(seed)
    alone = rng.binomial(1, law._LAW["lives_alone"], n).astype(float)
    risk = rng.standard_normal(n)
    early = rng.binomial(1, law._early(alone, risk, offer)).astype(float)
    late = law._late(alone, risk, offer, call)
    team = np.arange(n) // TEAM
    shared = rng.choice((-1.0, 1.0), size=int(team[-1]) + 1)[team]
    mean = late + delta * shared * np.minimum(late, 1.0 - late)
    return np.where(early == 1.0, 1.0, rng.binomial(1, mean).astype(float))


def _tolerance(draws: np.ndarray) -> float:
    """Four standard errors of the plan mean, from team means: teammates share their sign."""
    teams = draws.reshape(-1, TEAM).mean(axis=1)
    return 4.0 * teams.std(ddof=1) / np.sqrt(teams.size) + QUADRATURE_FLOOR


@pytest.mark.parametrize("plan", list(PLANS))
def test_the_truth_matches_a_monte_carlo_draw_of_the_intervened_law(plan: str) -> None:
    offer, call = PLANS[plan]
    draws = _intervened(offer, call, MONTE_CARLO_N, seed=11, delta=law.TEAM_DELTA)
    tolerance = _tolerance(draws)
    assert abs(draws.mean() - team_readmission_truth(offer, call)) < tolerance


def _without_early(offer: float, call: float) -> float:
    """A wrong truth: the day-90 risk of the patients not readmitted by day 30, read for all."""
    points, weights = np.polynomial.hermite_e.hermegauss(64)
    weights = weights / np.sqrt(2.0 * np.pi)
    share = law._LAW["lives_alone"]
    return sum(
        mass * float(np.sum(weights * law._late(alone, points, offer, call)))
        for alone, mass in ((1.0, share), (0.0, 1.0 - share))
    )


def _without_call(offer: float, call: float) -> float:
    """A wrong truth: the day-90 risk that ignores the call."""
    del call
    return team_readmission_truth(offer, 0.0)


@pytest.mark.parametrize("wrong", [_without_early, _without_call])
def test_a_wrong_truth_fails_the_same_comparison(wrong: Any) -> None:
    offer, call = PLANS["always"]
    draws = _intervened(offer, call, MONTE_CARLO_N, seed=11, delta=law.TEAM_DELTA)
    assert abs(draws.mean() - wrong(offer, call)) > 10.0 * _tolerance(draws)


def test_the_quadrature_has_converged() -> None:
    for offer, call in PLANS.values():
        assert team_readmission_truth(offer, call, nodes=96) == pytest.approx(
            team_readmission_truth(offer, call), abs=1e-12
        )


def test_the_truth_dictionary_uses_the_names_a_fit_reports() -> None:
    _, truth = make_team_readmission(n_teams=2, team_size=5, seed=0)
    assert set(truth) == {
        "ey_regimen[always]",
        "ey_regimen[never]",
        "ey_regimen[offer_only]",
        "ate_regimen[always vs never]",
    }
    assert truth["ate_regimen[always vs never]"] == pytest.approx(
        team_readmission_truth(1.0, 1.0) - team_readmission_truth(0.0, 0.0)
    )
    # The lesson of the tutorial: the plan lowers readmission, and the call adds to the offer.
    assert (
        truth["ey_regimen[always]"] < truth["ey_regimen[offer_only]"] < truth["ey_regimen[never]"]
    )


def test_the_observed_layout_is_monotone_and_ordered_in_time() -> None:
    frame, _ = make_team_readmission(n_teams=50, team_size=40, seed=3)
    assert list(frame.columns) == [
        "team",
        "lives_alone",
        "discharge_risk",
        "offer",
        "followed_30",
        "readmitted_30",
        "call",
        "followed_90",
        "readmitted_90",
    ]
    assert frame.groupby("team").size().eq(40).all()
    lost_30 = frame["followed_30"].eq(0)
    later = ["readmitted_30", "call", "followed_90", "readmitted_90"]
    assert frame.loc[lost_30, later].isna().all().all()
    assert frame.loc[~lost_30, later[:3]].notna().all().all()
    lost_90 = frame["followed_90"].eq(0)
    assert frame.loc[lost_90, "readmitted_90"].isna().all()
    early = frame["readmitted_30"].eq(1)
    # A patient readmitted by day 30 stays in follow-up and counts as readmitted by day 90.
    assert frame.loc[early, "followed_90"].eq(1).all()
    assert frame.loc[early, "readmitted_90"].eq(1).all()
    assert early.any() and lost_30.any() and lost_90.any()


def _team_correlation(frame: Any) -> float:
    """The correlation of two patients' day-90 residuals within a team and an offer arm."""
    observed = frame[frame["readmitted_90"].notna() & frame["readmitted_30"].eq(0)]
    fitted = law._late(
        observed["lives_alone"], observed["discharge_risk"], observed["offer"], observed["call"]
    )
    residual = (observed["readmitted_90"] - fitted).to_numpy()
    groups = observed.groupby(["team", "offer"]).indices.values()
    pairs = [
        residual[a] * residual[b]
        for rows in groups
        for i, a in enumerate(rows)
        for b in rows[i + 1 :]
    ]
    return float(np.mean(pairs) / np.mean(residual**2))


def test_the_team_component_correlates_teammates(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _ = make_team_readmission(n_teams=200, team_size=40, seed=5)
    shared = _team_correlation(frame)
    monkeypatch.setattr(law, "TEAM_DELTA", 0.0)
    control, _ = make_team_readmission(n_teams=200, team_size=40, seed=5)
    independent = _team_correlation(control)
    assert shared > 0.05
    assert abs(independent) < 0.02


def _late_residuals(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Day-90 outcome minus the late risk, for the patients it is drawn for, and their teams."""
    observed = frame[frame["readmitted_90"].notna() & frame["readmitted_30"].eq(0)]
    fitted = law._late(
        observed["lives_alone"], observed["discharge_risk"], observed["offer"], observed["call"]
    )
    return (observed["readmitted_90"] - fitted).to_numpy(), observed["team"].to_numpy()


def _team_mean_shift(frame: Any) -> tuple[float, float]:
    """The mean late residual and its standard error from team sums (team as the unit)."""
    residual, team = _late_residuals(frame)
    sums = np.bincount(team.astype(int), weights=residual)
    counts = np.bincount(team.astype(int))
    mean = sums.sum() / counts.sum()
    centred = sums - mean * counts
    return mean, float(np.sqrt(np.sum(centred**2)) / counts.sum())


def test_the_generated_outcome_has_the_late_risk_as_its_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator's own draw, not a copy of the law, keeps the day-90 mean at the late risk.

    Censoring and the call read only earlier columns, so among the patients observed at day 90
    and not readmitted by day 30 the team component must average zero.  Signs of 0 and 1, with a
    nonzero mean, are the mutation control: they must move the mean.
    """
    frame, _ = make_team_readmission(n_teams=500, team_size=40, seed=13)
    mean, error = _team_mean_shift(frame)
    assert abs(mean) < 4.0 * error
    monkeypatch.setattr(law, "_TEAM_SIGNS", (0.0, 1.0))
    shifted, _ = make_team_readmission(n_teams=500, team_size=40, seed=13)
    moved, moved_error = _team_mean_shift(shifted)
    assert moved > 10.0 * moved_error


def test_the_team_component_reads_no_earlier_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping every team's sign changes the day-90 outcome and no earlier column."""
    frame, _ = make_team_readmission(n_teams=20, team_size=10, seed=8)
    monkeypatch.setattr(law, "_TEAM_SIGNS", (1.0, -1.0))
    flipped, _ = make_team_readmission(n_teams=20, team_size=10, seed=8)
    before = list(frame.columns[:-1])
    assert frame[before].equals(flipped[before])
    assert not frame["readmitted_90"].equals(flipped["readmitted_90"])


def test_bad_sizes_are_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_team_readmission(n_teams=0)


def test_the_protocol_names_the_two_decisions_and_the_horizon() -> None:
    protocol = team_readmission_protocol()
    assert protocol.horizon == "90 days after discharge"
    assert "day-30" in protocol.treatment_strategies[0]
    assert any("composite" in rule for rule in protocol.intercurrent_event_handling)
