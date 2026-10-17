"""The declared design of ``survival-node-bounds`` (X33 part (i), ``X22-plan.md`` II-3.4).

Every number here was fixed before any registered run.  The exact-law checks call no sampler:
they read :mod:`tests.studies.survival_node_bounds_law` and fits on its 80 weighted records.
The pilot numbers are records, not checks.  Each pilot drew from a ``SeedSequence`` label that
no declared stream reads (``X33-study-design.md`` beside the plan names the scripts).

* **n.**  The declared rule takes the smallest n in {1,000, 2,000, 4,000} with at least 40
  expected events per arm by visit 5.  Arm 0 has 30.8 at 1,000 and 61.6 at 2,000.
* **The factor.**  The declared factor 1.5 drops to 1.1 when the unbounded mean fit leaves the
  1.5 bounds on fewer than half of the draws.  The pilot rate is :data:`EXCEEDANCE_PILOT`.
* **R.**  The smallest multiple of 200 at which the arm-pooled predictive rule on the
  calibration pilot (:data:`CALIBRATION_PILOT`) gives the four gated calibration positives a
  family-wise pass probability of at least 0.95.  It is 7,200, also a multiple of 400.  At
  nominal behaviour 2,800 suffices.
* **N.**  The control's exact-law limit at :data:`NECESSITY_N` clears the margin, the positive
  arm's limit is zero, and the root pilot (:data:`ROOT_PILOT`) records no draw whose lowered
  bound sat below the follower mean of the pseudo-outcome.
* **W.**  The gated exceedance share is :data:`WITNESS_PILOT`, two Monte Carlo errors above one
  half.  At n = 4,000 it falls below one half (:data:`WITNESS_PILOT_4000`).

**Open blocker.**  At n = 2,000 an arm can have no event at a visit.  The shipped fit then
refuses the standard errors that read that node, and the driver does not record that refusal,
so the run stops.  The declared calibration stream reaches such a draw at replicate 1,245
(:data:`REFUSED_CALIBRATION_REPLICATES`, a data-only scan).  The pre-queue review decides n.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from tests.studies import canonical_survival_node_bounds as study
from tests.studies import survival_node_bounds_law as law
from tests.studies import survival_node_bounds_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence.registry import Margins, registered
from tests.studies.evidence.seeds import stream_seed
from tests.studies.mtp_study_tools import (
    CELL_TARGET,
    FAMILY_TARGET,
    necessity_power,
    pass_probabilities,
    pilot_sigma,
    replicates_from_pilot,
    tau_from_pilot,
    tau_upper,
)

pytestmark = pytest.mark.xdist_group("survival_node_bounds_design")

Z = float(norm.ppf(0.995))

#: A record, not a check: the share of 800 pilot draws at n = 2,000 on which the unbounded fit
#: leaves the valid bounds of each factor (``pilot_w.py``, ``SeedSequence([3333, n, draw])``).
EXCEEDANCE_PILOT = {
    ("mean_model", 1.5): 0.0512,
    ("mean_model", 1.1): 0.5175,
    ("hazard_model", 1.5): 0.0025,
    ("hazard_model", 1.1): 0.4288,
}
#: A record, not a check: the share of the same draws on which the bounded and unbounded
#: estimates differ by more than ``1e-4`` on some reported parameter.
MOVED_PILOT = {
    ("mean_model", 1.5): 0.3875,
    ("mean_model", 1.1): 0.6787,
    ("hazard_model", 1.5): 0.0037,
    ("hazard_model", 1.1): 0.0612,
}
#: A record, not a check: the calibration pilot (``pilot_k.py``, ``SeedSequence([3336, draw])``)
#: at the declared factor.  It drew 4,000 samples.  One (draw 1,626) has no arm-0 event at visit
#: 3, and the shipped fit refuses its standard errors, so 3,999 are read.
CALIBRATION_PILOT_DRAWS = 3_999
#: The SE ratios of the four gated positives, in ``properties.CALIBRATION_LABELS`` order.
CALIBRATION_PILOT = (0.990024, 1.00948, 0.989353, 1.008846)
#: The W shares of the gated mean model on the same draws.
WITNESS_PILOT = {"exceeds_bound": 0.5164, "estimate_moved": 0.6832}
#: A record, not a check: the same pilot at n = 4,000 (2,000 draws,
#: ``SeedSequence([3336, 4000, draw])``).
WITNESS_PILOT_4000 = {"exceeds_bound": 0.3755, "estimate_moved": 0.5795}
#: A record, not a check: the root pilot of N's control at ``NECESSITY_N`` (``pilot_n.py``,
#: ``SeedSequence([3334, n, draw])``, 600 draws).  The gap is the lowered bound minus the
#: clever-covariate-weighted follower mean of the pseudo-outcome.  ``worst_z`` is the smallest
#: mean gap over its standard deviation, at arm 1, horizon 5, node 5.
ROOT_PILOT = {"draws": 600, "negative_gaps": 0, "unconverged": 0, "worst_z": 3.71}
#: A record, not a check: the positive arm's mean error at ``NECESSITY_N`` over the same 600
#: draws, in its empirical standard deviations, and its Monte Carlo standard error.
NECESSITY_POSITIVE_BIAS = (0.086, 0.041)
NECESSITY_N = properties.NECESSITY_N
#: The control's exact-law limit in the positive arm's standard deviations at ``NECESSITY_N``.
NECESSITY_LIMIT = 0.5849
#: A record, not a check: a data-only scan of the declared calibration stream
#: (``stream_seed(STUDY, "property_sample", "calibration", r)``, r < 8,000).  No fit was read.
#: In each of these replicates one arm has no event at one visit: ``replicate -> (arm, visit)``.
#: The primary streams of both models hold no such draw below 8,000.
REFUSED_CALIBRATION_REPLICATES = {1245: (0, 3), 7589: (0, 2), 7906: (0, 2)}


def _weighted_fit(model: str, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.fit(model, law.exact_frame(), weights="weight", **kwargs)


# ------------------------------------------------------------------ declared numbers


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_N, study.SEED, study.RESAMPLING_SEED) == (2_000, 20263391, 2026339101)
    assert study.FAILURE_BOUND == properties.FAILURE_BOUND == 0.01
    assert record.pending_reference == "survtmle"
    assert "#302" in str(record.pending_reason)
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.WITNESS_SHARE, properties.WITNESS_MOVEMENT) == (0.5, 1e-4)
    assert properties.WITNESS_GATED == (study.MEAN_MODEL,)
    assert properties.UNCONVERGED_SHARE == 0.01
    assert (properties.NECESSITY_N, properties.NECESSITY_SHARE) == (16_000, 0.5)
    assert properties.NECESSITY_DISPLACEMENT == 0.25


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {study.SEED, study.RESAMPLING_SEED}
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
        if seed is not None
    }
    assert not mine & others


def test_the_two_cell_maps_agree() -> None:
    assert study.STUDY.property_cells == properties.PROPERTY_CELLS
    keys = properties.smoke_keys(1, 0)
    named = {name for key in keys for name in properties.cells_of(key)}
    declared = {
        f"{family}/{cell}" for family, cells in study.PROPERTY_CELLS.items() for cell in cells
    }
    assert named == declared


# ------------------------------------------------------------------ the law


def test_the_truths_by_the_observed_hazards() -> None:
    """The second route: the identified hazards of the exact observed law, then the product."""
    frame = law.exact_frame()
    for horizon in law.HORIZONS:
        risks = {}
        for arm in law.ARMS:
            total = 0.0
            for w1, w2 in law.CELLS:
                cell = frame.loc[(frame["W1"] == w1) & (frame["W2"] == w2)]
                own = cell.loc[cell["A"] == arm]
                survive = 1.0
                for t in range(1, horizon + 1):
                    at_risk = own.loc[own["time"] >= t, "weight"].sum()
                    events = own.loc[(own["time"] == t) & (own["event"] == 1), "weight"].sum()
                    survive *= 1.0 - events / at_risk
                total += float(cell["weight"].sum()) * (1.0 - survive)
            risks[arm] = total
            name = law.risk_name("arm1" if arm else "arm0", horizon)
            assert risks[arm] == pytest.approx(law.TRUTH[name], abs=1e-14)
        contrast = law.TRUTH[law.contrast_name(horizon)]
        assert risks[1] - risks[0] == pytest.approx(contrast, abs=1e-14)
    assert law.TRUTH[law.risk_name("arm1", 5)] == pytest.approx(0.133619, abs=5e-7)
    assert law.TRUTH[law.contrast_name(5)] == pytest.approx(0.049509, abs=5e-7)


def test_the_efficiency_bounds() -> None:
    expected = {
        law.risk_name("arm1", 3): 0.3457,
        law.risk_name("arm0", 3): 0.3470,
        law.contrast_name(3): 0.4890,
        law.risk_name("arm1", 5): 0.4579,
        law.risk_name("arm0", 5): 0.4653,
        law.contrast_name(5): 0.6509,
    }
    for name, value in expected.items():
        assert law.EFFICIENCY_SD[name] == pytest.approx(value, abs=5e-5), name


def test_the_primary_n_is_the_smallest_with_forty_events_per_arm() -> None:
    passing = [n for n in (1_000, 2_000, 4_000) if min(law.expected_events(n).values()) >= 40]
    assert passing[0] == study.PRIMARY_N
    events = law.expected_events(study.PRIMARY_N)
    assert events[0] == pytest.approx(61.6, abs=0.05)
    assert events[1] == pytest.approx(150.6, abs=0.05)


@pytest.mark.parametrize("model", study.MODELS)
def test_the_exact_law_fit_returns_the_truth(model: str) -> None:
    """The declared nuisance models are correct up to the unsaturated additive scaled logit."""
    result = _weighted_fit(model)
    for name in study.ESTIMANDS[model]:
        standardized = abs(result.psi(name) - law.TRUTH[name]) * np.sqrt(study.PRIMARY_N)
        assert standardized / law.EFFICIENCY_SD[name] < 1e-3, name


def _no_event_probability(n: int) -> float:
    """The chance that some arm has no event at some visit in a sample of ``n``."""
    keep = 1.0
    for arm in law.ARMS:
        for visit in range(1, law.K + 1):
            mass = sum(
                probability
                for probability, a, _, _, history in law.records()
                if a == arm and history.event == 1 and history.time == visit
            )
            keep *= 1.0 - (1.0 - mass) ** n
    return 1.0 - keep


def test_the_declared_calibration_stream_holds_refused_draws() -> None:
    """The open blocker: at n = 2,000 the calibration stream reaches a refused draw."""
    assert _no_event_probability(study.PRIMARY_N) == pytest.approx(3.03e-5, rel=0.01)
    assert _no_event_probability(4_000) < 1e-9
    for replicate, (arm, visit) in REFUSED_CALIBRATION_REPLICATES.items():
        seed = stream_seed(study.STUDY, "property_sample", "calibration", replicate)
        frame = law.draw(study.PRIMARY_N, seed)
        events = (frame["A"] == arm) & (frame["time"] == visit) & (frame["event"] == 1)
        assert not events.any()
    assert min(REFUSED_CALIBRATION_REPLICATES) < study.REPLICATES


# ------------------------------------------------------------------ bounds


@pytest.mark.parametrize("factor", (law.DECLARED_FACTOR, law.REDUCED_FACTOR))
def test_the_valid_bounds_hold_the_truth(factor: float) -> None:
    mean = law.mean_bounds(factor)
    hazard = law.hazard_bounds(factor)
    for t in range(1, law.K + 1):
        for a in law.ARMS:
            for cell in law.CELLS:
                hazard_value = float(law.hazard(t, a, *cell))
                assert hazard[t][0] <= hazard_value < hazard[t][1]
                for horizon in law.BOUND_HORIZONS:
                    if horizon >= t:
                        value = law.iterated_mean(t, horizon, a, *cell)
                        assert mean[t][0] <= value < mean[t][1]


def test_the_factor_rule() -> None:
    declared = EXCEEDANCE_PILOT[(study.MEAN_MODEL, law.DECLARED_FACTOR)]
    expected = law.REDUCED_FACTOR if declared < 0.5 else law.DECLARED_FACTOR
    assert study.BOUND_FACTOR == expected == 1.1
    np.testing.assert_allclose(
        [u for _, u in study.MEAN_BOUNDS.values()],
        [0.21933, 0.18685, 0.14963, 0.10682, 0.05737],
        atol=5e-6,
    )


# ------------------------------------------------------------------ R


def _arm_pooled_replicates(step: int) -> int:
    pilot = CALIBRATION_PILOT_DRAWS
    arms = len(CALIBRATION_PILOT)
    mean = float(np.mean(CALIBRATION_PILOT))
    shared = pilot_sigma(pilot) ** 2 / arms
    tau = max(tau_upper(pilot, arms), tau_from_pilot(CALIBRATION_PILOT, pilot))
    for replicates in range(step, 40_001, step):
        cell, family = pass_probabilities(replicates, mean, shared, tau, arms)
        if family >= FAMILY_TARGET and cell >= CELL_TARGET:
            return replicates
    raise AssertionError("no R up to 40,000")


def test_the_replicates_follow_the_arm_pooled_rule() -> None:
    assert len(CALIBRATION_PILOT) == len(properties.CALIBRATION_LABELS) == 4
    rule = replicates_from_pilot(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)
    assert rule["screened"] == [True] * 4
    assert not rule["suspect"]
    assert rule["R"] == 7_200
    assert _arm_pooled_replicates(200) == _arm_pooled_replicates(400) == 7_200
    assert study.REPLICATES == study.PRIMARY_REPLICATES == properties.CALIBRATION_REPLICATES
    assert study.REPLICATES == properties.NECESSITY_REPLICATES == 7_200


def test_the_nominal_calibration_rule() -> None:
    """At nominal behaviour R = 2,800 suffices; the arm-pooled rule asks for more."""

    def family(replicates: int) -> float:
        spread = np.sqrt(1.0 / (2.0 * (replicates - 1)))
        low, high = properties.EFFICIENCY_RATIO_BAND
        efficiency = norm.cdf((high - 1 - Z * spread) / spread) - norm.cdf(
            (low - 1 + Z * spread) / spread
        )
        cell = calibration_pass_probability(replicates, true_se_ratio=1.0) * efficiency
        return float(cell**4)

    passing = [r for r in range(200, 6_001, 200) if family(r) >= 0.95]
    assert passing[0] == 2_800
    assert family(study.REPLICATES) > 0.999


def test_the_witness_is_marginal_at_the_declared_n() -> None:
    """W's exceedance share sits two Monte Carlo errors above one half at n = 2,000."""
    rate = WITNESS_PILOT["exceeds_bound"]
    replicates = study.REPLICATES
    point = norm.cdf((rate - 0.5) / np.sqrt(rate * (1 - rate) / replicates))
    spread = np.sqrt(rate * (1 - rate) * (1 / CALIBRATION_PILOT_DRAWS + 1 / replicates))
    assert point > 0.99
    assert 0.94 < norm.cdf((rate - 0.5) / spread) < 0.96
    assert WITNESS_PILOT["estimate_moved"] > 0.65
    assert WITNESS_PILOT_4000["exceeds_bound"] < properties.WITNESS_SHARE


# ------------------------------------------------------------------ N


def _necessity_sd() -> float:
    """The positive arm's standard deviation at ``NECESSITY_N``: the intercept-only curve."""
    return law.efficiency_sd(properties.F5_ARM1, intercept_only=True) / np.sqrt(NECESSITY_N)


def test_the_excluding_bound_excludes_the_truth_and_admits_a_root() -> None:
    bounds = properties.NECESSITY_BOUNDS
    for t in properties.NECESSITY_NODES:
        upper = bounds[t][1]
        assert properties.at_risk_mean(t, 1) < upper < study.MEAN_BOUNDS[t][1]
        assert any(law.iterated_mean(t, 5, 1, *cell) > upper for cell in law.CELLS), t


def test_the_necessity_limits() -> None:
    """The positive arm's exact-law limit is the truth; the control's clears the margin."""
    sd = _necessity_sd()
    positive = _weighted_fit(study.MEAN_MODEL, **properties.necessity_learners())
    control = _weighted_fit(
        study.MEAN_MODEL, bounds=properties.NECESSITY_BOUNDS, **properties.necessity_learners()
    )
    truth = law.TRUTH[properties.F5_ARM1]
    assert abs(positive.psi(properties.F5_ARM1) - truth) / sd < 1e-3
    limit = (control.psi(properties.F5_ARM1) - truth) / sd
    assert limit == pytest.approx(NECESSITY_LIMIT, abs=5e-3)
    assert necessity_power(limit, properties.NECESSITY_REPLICATES) >= 0.999


def test_the_root_pilot_and_the_positive_arm() -> None:
    assert ROOT_PILOT["negative_gaps"] == ROOT_PILOT["unconverged"] == 0
    # A normal tail at the worst node keeps the unconverged share far below the 1% rule.
    assert norm.sf(ROOT_PILOT["worst_z"]) < properties.UNCONVERGED_SHARE / 50
    bias, error = NECESSITY_POSITIVE_BIAS
    root = np.sqrt(properties.NECESSITY_REPLICATES)
    assert norm.cdf(root * (0.25 - (bias + 2.0 * error)) - Z) >= 0.99
