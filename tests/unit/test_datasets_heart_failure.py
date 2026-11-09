"""The heart-failure water pill: exact separable truth, the observed layout, and the event order.

:func:`cleverly.datasets.heart_failure_truth` sums the separable law of Stensrud et al. (2022),
identification (8) with death first, over the eight covariate cells.  A Monte Carlo draw of the
world where the fluid part and the potassium part of the dose are set apart checks it.  Exact-law
checks are blind to a term that vanishes at the truth, so a wrong truth that reads both parts at
one dose must fail the same comparison at unequal doses.

A large draw of :func:`cleverly.datasets.make_heart_failure`, binned onto the checks, gives
empirical cell hazards; plugged into the same identification they must recover the truth, which
checks the day binning and that leaving follow-up depends only on the recorded columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly.datasets import heart_failure as law
from cleverly.datasets import heart_failure_protocol, heart_failure_truth, make_heart_failure
from cleverly.datasets.heart_failure import HEART_FAILURE_DAYS, dose_probabilities

#: Draws in each Monte Carlo comparison.
MONTE_CARLO_N = 400_000
#: Dose pairs ``(fluid part, potassium part)`` of the Monte Carlo comparison.
PAIRS = ((0, 0), (2, 2), (2, 0), (0, 2), (1, 2))


def _set_apart(readmission_dose: int, death_dose: int, n: int, seed: int) -> np.ndarray:
    """``(n,)`` first-event codes by day 90 when every patient gets the two parts at two doses.

    Nobody leaves follow-up.  Death is drawn first in each month, at the potassium part's dose;
    readmission among those alive, at the fluid part's dose.  This is the structural world of
    the generator, simulated rather than summed.
    """
    rng = np.random.default_rng(seed)
    older = rng.binomial(1, law._P_OLDER, n)
    prior = rng.integers(0, len(law._PRIOR), n)
    code = np.zeros(n, dtype=int)
    for month in range(1, len(HEART_FAILURE_DAYS) + 1):
        at_risk = code == 0
        died = at_risk & (rng.uniform(size=n) < law._death(month, death_dose, older, prior))
        readmitted = (
            at_risk
            & ~died
            & (rng.uniform(size=n) < law._readmission(month, readmission_dose, older, prior))
        )
        code[died] = 2
        code[readmitted] = 1
    return code


@pytest.mark.parametrize("pair", PAIRS)
def test_the_truth_matches_a_monte_carlo_draw_of_the_parts_set_apart(pair: tuple[int, int]) -> None:
    codes = _set_apart(*pair, n=MONTE_CARLO_N, seed=sum(pair) + 10 * pair[0])
    truth = heart_failure_truth(*pair).set_index("day").loc[90]
    for code, cause in ((1, "readmission"), (2, "death")):
        share = (codes == code).mean()
        tolerance = 4.0 * np.sqrt(share * (1.0 - share) / MONTE_CARLO_N)
        assert abs(share - truth[cause]) < tolerance


def test_a_truth_that_reads_both_parts_at_one_dose_fails_the_comparison() -> None:
    # Deliberate mutation: reading the potassium part at the fluid part's dose gives the static
    # risk of the high dose, which the draw with the parts set apart must reject.
    codes = _set_apart(2, 0, n=MONTE_CARLO_N, seed=3)
    wrong = heart_failure_truth(2, 2).set_index("day").loc[90]
    for code, cause in ((1, "readmission"), (2, "death")):
        share = (codes == code).mean()
        tolerance = 4.0 * np.sqrt(share * (1.0 - share) / MONTE_CARLO_N)
        assert abs(share - wrong[cause]) > 10 * tolerance


def test_the_two_event_orders_agree_at_equal_doses_and_differ_otherwise() -> None:
    for dose in (0, 1, 2):
        death_first = heart_failure_truth(dose, dose)
        readmission_first = heart_failure_truth(dose, dose, first="readmission")
        pd.testing.assert_frame_equal(death_first, readmission_first, rtol=1e-12, atol=1e-14)
    gap = heart_failure_truth(2, 0, first="readmission") - heart_failure_truth(2, 0)
    assert (gap["readmission"].abs() > 0.01).all()
    assert (gap["death"].abs() > 0.005).all()


def _binned_cell_hazards(
    frame: pd.DataFrame,
) -> dict[tuple[int, int, int, int], tuple[float, float]]:
    """Empirical ``(lambda_D, h_Y)`` by ``(older, prior, dose, month)`` from the binned days."""
    month = (
        np.searchsorted(np.asarray(HEART_FAILURE_DAYS), frame["day"].to_numpy(), side="left") + 1
    )
    event = frame["event"].to_numpy()
    hazards = {}
    keys = frame[["older", "prior_admissions", "dose"]].to_numpy(dtype=int)
    for k in range(1, len(HEART_FAILURE_DAYS) + 1):
        # At risk in month k: binned into month k or later.  A patient who left after the check
        # that ends month k has that check's day, event 0, and was at risk through month k.
        at_risk = month >= k
        died = at_risk & (month == k) & (event == 2)
        readmitted = at_risk & (month == k) & (event == 1)
        for cell in {tuple(row) for row in keys}:
            rows = at_risk & (keys == cell).all(axis=1)
            alive = rows & ~died
            hazards[(*cell, k)] = (died[rows].mean(), readmitted[alive].mean())
    return hazards


def test_a_large_draw_recovers_the_truth_from_its_cell_hazards() -> None:
    frame, truth = make_heart_failure(n=400_000, seed=5)
    hazards = _binned_cell_hazards(frame)
    weights = frame.groupby(["older", "prior_admissions"]).size() / len(frame)
    for readmission_dose, death_dose in PAIRS:
        risk = np.zeros(2)
        for (older, prior), mass in weights.items():
            survival, running = 1.0, np.zeros(2)
            for k in range(1, len(HEART_FAILURE_DAYS) + 1):
                lam_d = hazards[(int(older), int(prior), death_dose, k)][0]
                lam_y = hazards[(int(older), int(prior), readmission_dose, k)][1] * (1 - lam_d)
                running += survival * np.array([lam_y, lam_d])
                survival *= 1 - lam_y - lam_d
            risk += mass * running
        row = truth.query(
            f"readmission_dose == {readmission_dose} and death_dose == {death_dose} and day == 90"
        ).iloc[0]
        assert risk[0] == pytest.approx(row["readmission"], abs=0.008)
        assert risk[1] == pytest.approx(row["death"], abs=0.008)


def test_the_layout_codes_and_days() -> None:
    frame, truth = make_heart_failure(n=5_000, seed=1)
    assert list(frame.columns) == ["older", "prior_admissions", "dose", "day", "event"]
    assert set(frame["event"].unique()) == {0, 1, 2}
    assert set(frame["dose"].unique()) == {0.0, 1.0, 2.0}
    censored = frame.loc[frame["event"].eq(0), "day"]
    assert set(censored.unique()) <= {30.0, 60.0, 90.0}
    events = frame.loc[frame["event"].gt(0), "day"]
    assert events.between(1, 90).all() and (events == events.round()).all()
    assert len(truth) == 27
    static = truth.query("readmission_dose == death_dose")
    for dose, rows in static.groupby("readmission_dose"):
        expected = heart_failure_truth(int(dose), int(dose))
        np.testing.assert_allclose(rows["readmission"], expected["readmission"], rtol=1e-12)
    np.testing.assert_allclose(dose_probabilities([0, 1], [3, 0]).sum(axis=1), 1.0)


def test_the_truth_refuses_an_unknown_dose_or_order() -> None:
    with pytest.raises(ValueError, match="readmission_dose"):
        heart_failure_truth(3, 0)
    with pytest.raises(ValueError, match="first"):
        heart_failure_truth(0, 0, first="censoring")


def test_the_protocol_names_the_three_strategies() -> None:
    protocol = heart_failure_protocol()
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 3
    assert "competing" in protocol.intercurrent_event_handling[0]
