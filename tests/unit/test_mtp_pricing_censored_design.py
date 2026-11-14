"""The design of ``mtp-pricing-censored``: law P-C and the numbers its declaration reads.

Law P-C (:mod:`tests.studies.mtp_pricing_censored_common`) is law P at three nodes with a dropout
after each price.  The checks below hold the facts the study's reading leans on: the dropout
changes no column of law P's record, the dropout is at random given the history so law P's
truth is the parameter, the retention bounds are as declared, the numpy own curve that fixes the
tail size is ``cleverly``'s recursion, and the calibration cells' replications follow from the
sizing rule.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from tests.studies import canonical_mtp_pricing_censored as study
from tests.studies import mtp_pricing_censored_common as law
from tests.studies import mtp_pricing_censored_own_curve as own
from tests.studies import mtp_pricing_censored_properties as properties
from tests.studies import mtp_pricing_common as law_p
from tests.studies import mtp_study_tools as tools
from tests.studies import mtp_tail_size_common as tail
from tests.studies.evidence.inference import clopper_pearson

#: The largest difference from ``cleverly``, in standard deviations of the curve.
AGREEMENT = 1e-8


def test_the_dropout_changes_no_column_of_law_p() -> None:
    frame = law.sample(5_000, 17)
    complete = law_p.sample(5_000, 17, law.TAU)
    for name in complete.columns:
        observed = frame[name].notna()
        np.testing.assert_array_equal(frame.loc[observed, name], complete.loc[observed, name])
    # The first node's record is never removed, and monotone dropout removes every later node.
    assert frame[["W1", "L1", "A1", "C1"]].notna().all().all()
    for t in range(1, law.TAU + 1):
        left = frame[f"C{t}"] == 0
        later = [f"C{s}" for s in range(t + 1, law.TAU + 1)] + ["Y", "R"]
        assert frame.loc[left, later].isna().all().all()
        assert left.sum() > 0


def test_inverse_retention_weighting_recovers_law_p_truth() -> None:
    """The dropout is at random given the history and the dose, so law P's truth stands."""
    n = 1_000_000
    frame = law.sample(n, 2026771777)
    weighted = law.ipcw_natural_mean(frame)
    truth, truth_se = law_p.TRUTHS[law.TAU]["ey[natural_prices]"]
    retained = frame["C3"] == 1
    assert 0.10 < 1.0 - retained.mean() < 0.17
    # The Monte Carlo SE of the weighted mean, from its own draws.
    weight = np.ones(n)
    for t in range(1, law.TAU + 1):
        l = np.nan_to_num(frame[f"L{t}"].to_numpy())
        gap = np.nan_to_num(frame[f"A{t}"].to_numpy()) - law_p.centre(l)
        weight = weight / law.retention(frame["W1"], l, gap)
    terms = np.where(retained, weight * np.nan_to_num(frame["Y"].to_numpy()), 0.0)
    se = float(np.std(terms) / np.sqrt(n))
    assert abs(weighted - truth) < 4.0 * np.hypot(se, truth_se)


def test_the_retention_is_bounded_away_from_zero() -> None:
    assert law.minimum_retention() == pytest.approx(0.5866, abs=1e-4)
    assert law.minimum_retention() ** law.TAU > 0.2
    grid = np.linspace(-law_p.W_BOUND, law_p.W_BOUND, 11)
    for l in (0.0, 1.0):
        for gap in np.linspace(-2.0, 2.0, 9):
            assert (law.retention(grid, l, gap) >= law.minimum_retention() - 1e-12).all()


def test_the_retention_logit_is_a_main_terms_logistic_in_the_history() -> None:
    """The declared censoring learner is correctly specified: logit linear in W1, L_t, A_t."""
    frame = law.sample(200_000, 23)
    model = law.censoring_learner().fit(frame[["W1", "L1", "A1"]], frame["C1"])
    intercept = law.ALPHA - law.BETA_GAP * 6.0
    expected = [law.BETA_W, law.BETA_L - 2.0 * law.BETA_GAP, law.BETA_GAP]
    np.testing.assert_allclose(model.coef_[0], expected, atol=0.08)
    assert model.intercept_[0] == pytest.approx(intercept, abs=0.4)


@pytest.mark.parametrize(
    ("estimator", "learners"),
    [("ltmle", "primary"), ("sdr", "primary"), ("ltmle", "property"), ("sdr", "property")],
)
def test_the_numpy_censored_fit_is_cleverlys(estimator: str, learners: str) -> None:
    """With ``cleverly``'s learned retention, the two recursions meet to rounding."""
    frame = law.sample(2_000, 31)
    settings = {}
    if learners == "property":
        settings = {"outcome_learner": law_p.FinalOutcome(), "pseudo_learner": law_p.SplinePseudo()}
    alpha = study.PRIMARY_ALPHA if estimator == "ltmle" and learners == "primary" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.fit(frame, estimator=estimator, alpha=alpha, **settings)
    cumulative = np.asarray(result.fits[study.REFERENCE].cumulative_unbounded, dtype=float)
    steps, previous = [], np.ones(len(frame))
    for t in range(law.TAU):
        safe = np.where(previous > 0.0, previous, 1.0)
        steps.append(np.where(previous > 0.0, cumulative[:, t] / safe, 1.0))
        previous = cumulative[:, t]
    cols = own.columns(frame)
    curve = own.OwnCurve(law.TAU, estimator, learners, study.LABELS, retention=lambda _: steps).fit(
        cols
    )  # type: ignore[arg-type]
    mine = curve.quantities(cols)
    for label in study.LABELS:
        estimate = result[f"ey_regimen[{label}]"]
        theirs = np.asarray(estimate.influence_curve, dtype=float)
        assert np.max(np.abs(theirs - mine[f"ey[{label}]"])) / np.std(theirs) < AGREEMENT
        assert abs(float(estimate.psi) - curve.plans[label].psi) < AGREEMENT
    # The learned retention is close to the true one, the curve's limit.
    true = own.true_retention(cols)
    for t in range(1, law.TAU + 1):
        alive = own.alive(cols, t - 1)
        assert np.max(np.abs(steps[t - 1][alive] - true[t - 1][alive])) < 0.25


def test_the_calibration_cells_are_sized_from_the_extended_pilots() -> None:
    """The fallback target fails the pilots' ratios; the per-arm rule over them sets R."""
    target = properties.target_replicates()
    assert target["R"] == properties.TARGET_REPLICATES == 2_800
    for ratio in properties.PILOT_SE_RATIOS.values():
        assert tools.cell_pass(ratio, properties.TARGET_REPLICATES, tools.C_FIT) < 0.90
    sizing = properties.calibration_replicates()
    assert sizing["R"] == properties.SIZED_REPLICATES == 10_800
    assert min(sizing["per_cell"]) >= tools.CELL_TARGET
    assert sizing["family_wise"] >= tools.FAMILY_TARGET
    assert set(properties.CALIBRATION_REPLICATES.values()) == {properties.SIZED_REPLICATES}


def test_the_primary_coverage_floor_passes_at_nominal_coverage() -> None:
    """At 1,000 replications a row covering at 0.95 clears the coverage floor almost surely."""
    margins = study.STUDY.margins
    reps = study.PRIMARY_REPLICATES
    passing = [
        k
        for k in range(reps + 1)
        if clopper_pearson(k, reps, confidence_level=margins.confidence_level).low
        >= margins.coverage_floor
    ]
    assert binom.sf(min(passing) - 1, reps, 0.95) > 0.999


def test_the_study_samples_record_the_joint_ratio() -> None:
    """The comparator columns are the policy ratio over the learned retention, zero after leaving."""
    key = ("censored_sdr_tau3", 0, 500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample, _, _ = study._replicate(key, study._draw(key))
    assert isinstance(sample, pd.DataFrame)
    for t in range(1, law.TAU + 1):
        column = sample[f"ratio__natural_prices__{t}"]
        stayed = (sample[[f"C{s}" for s in range(1, t + 1)]] == 1).all(axis=1)
        assert (column[~stayed] == 0.0).all()
        # Under the natural course the joint ratio is the inverse retention, above one.
        assert (column[stayed] > 1.0).all()


TAIL = Path(__file__).resolve().parents[1] / "diagnostics" / "x77_censored_tail"


def test_the_tail_sizes_and_the_study_n_follow_from_the_committed_moments() -> None:
    frame = pd.read_csv(TAIL / "law_pc_moments.csv")
    assert set(zip(frame.learners, frame.estimator, strict=True)) == set(own.LAW_PC_FITS)
    assert (frame.draws == own.LAW_PC_DRAWS).all()
    assert (frame.kappa_lower <= frame.kappa).all()
    assert (frame.kappa <= frame.kappa_upper).all()
    for row in frame.itertuples():
        assert row.n_kappa == tail.tail_size(row.kappa_upper)
    quantity = {
        "ey_regimen[natural_prices]": "ey[natural_prices]",
        f"ey_regimen[{study.PLAN}]": f"ey[{study.PLAN}]",
        f"ate_regimen[{study.PLAN} vs {study.REFERENCE}]": f"ate[{study.PLAN}]",
    }
    indexed = frame.set_index(["learners", "estimator", "quantity"])["n_kappa"]
    for name, key in quantity.items():
        rows = [indexed[learners, estimator, key] for learners, estimator in own.LAW_PC_FITS]
        assert study.TAIL_SIZES[name] == max(rows)
    assert max(2_000, tail.gated_size(max(study.TAIL_SIZES.values()))) == study.PRIMARY_N
    sizes = {f"{s}/{name}": study.PRIMARY_N for s in study.SCENARIOS for name in study.TAIL_SIZES}
    tails = {f"{s}/{name}": n for s in study.SCENARIOS for name, n in study.TAIL_SIZES.items()}
    gated, below, missing = tail.classify(sizes, tails)
    assert gated == study.WALD_GATED_CELLS
    assert below == study.BELOW_TAIL_SIZE
    assert not missing and not study.TAIL_SIZE_NOT_COMPUTED


def test_the_pair_pilots_clear_the_power_rule() -> None:
    for name, reading in properties.PAIR_PILOT_READINGS.items():
        assert name in properties.PAIR_SETS
        assert reading["equivalence_power"] >= 0.90
        assert reading["necessity_power"] >= 0.90
        assert reading["displacement"] > properties.NECESSITY_DISPLACEMENT
