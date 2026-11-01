"""The declared design of ``survival-node-bounds`` (X33 part (i), ``X22-plan.md`` II-3.4).

Every number here was fixed before any registered run.  The exact-law checks call no sampler:
they read :mod:`tests.studies.survival_node_bounds_law` and fits on its 80 weighted records.
The pilot numbers are records, not checks.  Each pilot drew from a ``SeedSequence`` label that
no declared stream reads (``X33-study-design.md`` beside the plan names the scripts).

**The declaration** (this module, the study modules and the design note are one commit):

* **Law.**  The declared law with two changes.  **D1**: the declaration names no marginal for
  ``W``, and the study sets ``P(W1 = 1) = 0.3`` and ``P(W2 = 1) = 0.5``.  **D9**: the
  treatment intercept is -0.5, not 0.3.  Together they balance the arms and thin the binding
  cell ``(a = 1, W1 = 1, W2 = 0)``, so that no node of either arm is in the
  ``constant_node_plugin`` regime at the declared n.
* **n.**  The smallest n in {1,000, 2,000, 4,000} with at least 40 expected events per arm by
  visit 5, and an exact union bound of at most 1e-6 on the chance that some arm has no event at
  some visit.  The 40-event rule alone picks 1,000 (:func:`test_the_primary_n_follows_the_rule`).
  The union bound at 1,000 is 2.0e-3 and at 2,000 is 5.3e-7, so n = 2,000.
* **Failure bound.**  0.01 for each implementation, as the shipped studies declare.  A cell may
  lose ``floor(0.01 R)`` replicates, and the accounting fills each lost value at its adverse
  support bound.  A draw with no event for an arm at a visit reports ``constant_node_plugin``,
  and the study translates it into a ``DataError``, which the driver records as a lost
  replicate.  Conditioning on that event would change any statistic in [0, 1] by at most its
  probability, so the recorded loss is stricter than conditioning.
* **The factor.**  The declared factor 1.5 drops to 1.1 when the unbounded mean fit leaves the
  1.5 bounds on fewer than half of the draws.  The pilot rate is :data:`EXCEEDANCE_PILOT`.
* **R.**  The smallest multiple of 400 at which the family-wise product over every gated cell
  is at least 0.95: K by the arm-pooled predictive rule on :data:`CALIBRATION_PILOT`, W by its
  predictive pass probability, and N.  The K rule alone gives the same value.  The pre-queue
  review's pilot gave 6,400 and the pilot under the committed law code gave 7,200.  The
  pilots are not pooled, and the larger value, 7,200, is declared.
* **N.**  The control lowers every node (D6) at n = 16,000 (D7).  Its exact-law limit clears
  the margin, the positive arm's limit is zero, and the root pilot (:data:`ROOT_PILOT`)
  records no draw whose lowered bound sat below the follower mean of the pseudo-outcome.
* **W.**  The gated exceedance and moved shares are :data:`WITNESS_PILOT`, several Monte Carlo
  errors above one half.  The moved share reads ``F(5 | 1)`` and ``RD(5)`` (D5).
* **D10.**  Node 1 retains every unit.  The container declares that factor known, as
  ``survtmle`` sets ``g_dC = 1`` at the first time, so the pooled censoring model is correctly
  specified.  No P-cell routes a convention gap; the comparator is pending.

======  ==========================================================================
id      deviation from the declaration
======  ==========================================================================
D1      ``P(W1 = 1) = 0.3``, ``P(W2 = 1) = 0.5``: the declaration names no marginal
D2      the reported cells form the diagnostic family ``reported_calibration``
D3      the hazard-model W cell is diagnostic: its moved share is about 0.1
D4      the 1% unconverged rule is a cell verdict over a ``converged`` column
D5      W exceedance reads targeted predictions (mean) and initial hazards (hazard); the
        moved share reads ``F(5 | 1)`` and ``RD(5)``
D6      N lowers every node, share 0.5
D7      N at n = 16,000
D8      one R for primary, K, W, C3 and N
D9      the treatment intercept is -0.5
D10     node 1 retention declared known; no convention gap is routed
======  ==========================================================================

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from cleverly.exceptions import DataError
from cleverly.validation._failures import recordable
from tests.studies import canonical_survival_node_bounds as study
from tests.studies import survival_node_bounds_law as law
from tests.studies import survival_node_bounds_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered
from tests.studies.mtp_study_tools import (
    CELL_TARGET,
    FAMILY_TARGET,
    equivalence_power,
    necessity_power,
    pass_probabilities,
    pilot_sigma,
    replicates_from_pilot,
    tau_from_pilot,
    tau_upper,
)

RUNNER = ROOT / "tests" / "canonical" / "survival_node_bounds" / "run_study.R"

pytestmark = pytest.mark.xdist_group("survival_node_bounds_design")

Z = float(norm.ppf(0.995))

#: The pilot labels (``SeedSequence([label, n, draw])``); no declared stream reads them.  The
#: declared streams spawn from ``SEED`` (``stream_seed``, ``replicate_seed``).
PILOT_LABELS = {"exceedance": 7101, "calibration": 7102, "necessity": 7103}

#: A record, not a check: the share of 1,200 pilot draws at n = 2,000 on which the unbounded fit
#: leaves the valid bounds of each factor (``x33_pilot_w.py``).
EXCEEDANCE_PILOT: dict[tuple[str, float], float] = {
    ("mean_model", 1.5): 0.1125,
    ("mean_model", 1.1): 0.5892,
    ("hazard_model", 1.5): 0.0233,
    ("hazard_model", 1.1): 0.4700,
}
#: A record, not a check: the calibration pilot (``x33_pilot_k.py``) at the declared factor.
#: It drew 4,000 samples at n = 2,000.
CALIBRATION_PILOT_DRAWS = 4_000
#: The SE ratios of the four gated positives, in ``properties.CALIBRATION_LABELS`` order.
CALIBRATION_PILOT = (0.9962, 0.9915, 0.9959, 0.9915)
#: The W shares of the gated mean model on the same draws: the exceedance share and the share
#: of ``F(5 | 1)`` and ``RD(5)`` that moved by more than ``1e-4``.
WITNESS_PILOT = {"exceeds_bound": 0.5923, "estimate_moved": 0.6518}
#: A record, not a check: the root pilot of N's control at ``NECESSITY_N`` (``x33_pilot_n.py``,
#: 600 draws).  The gap is the lowered bound minus the clever-covariate-weighted follower mean
#: of the pseudo-outcome.  ``worst_z`` is the smallest mean gap over its standard deviation.
ROOT_PILOT = {"draws": 600, "negative_gaps": 0, "unconverged": 0, "worst_z": 3.85}
#: A record, not a check: the positive arm's mean error at ``NECESSITY_N`` over the same draws,
#: in its empirical standard deviations, and its Monte Carlo standard error.
NECESSITY_POSITIVE_BIAS = (0.027, 0.041)
NECESSITY_N = properties.NECESSITY_N
#: The control's exact-law limit in the positive arm's standard deviations at ``NECESSITY_N``.
NECESSITY_LIMIT = 0.604


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
    assert record.failure_bound == {"cleverly": 0.01, "survtmle": 0.01}
    assert record.pending_reference == "survtmle"
    assert "#302" in str(record.pending_reason)
    assert "TODO" in str(record.pending_reason)
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.WITNESS_SHARE, properties.WITNESS_MOVEMENT) == (0.5, 1e-4)
    assert properties.WITNESS_GATED == (study.MEAN_MODEL,)
    assert properties.UNCONVERGED_SHARE == 0.01
    assert (properties.NECESSITY_N, properties.NECESSITY_SHARE) == (16_000, 0.5)
    assert properties.NECESSITY_DISPLACEMENT == 0.25
    assert (law.P_W1, law.P_W2, law.G_INTERCEPT) == (0.3, 0.5, -0.5)


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
    assert {(c.property, c.cell) for c in properties.declared_cells()} == {
        tuple(name.split("/", 1)) for name in declared
    }


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
    expected = {
        law.risk_name("arm1", 5): 0.119298,
        law.risk_name("arm0", 5): 0.074794,
        law.contrast_name(5): 0.044504,
        law.risk_name("arm1", 3): 0.066562,
        law.risk_name("arm0", 3): 0.041193,
    }
    for name, value in expected.items():
        assert law.TRUTH[name] == pytest.approx(value, abs=5e-7), name


def test_the_efficiency_bounds() -> None:
    expected = {
        law.risk_name("arm1", 3): 0.4119,
        law.risk_name("arm0", 3): 0.2602,
        law.contrast_name(3): 0.4865,
        law.risk_name("arm1", 5): 0.5481,
        law.risk_name("arm0", 5): 0.3501,
        law.contrast_name(5): 0.6488,
    }
    for name, value in expected.items():
        assert law.EFFICIENCY_SD[name] == pytest.approx(value, abs=5e-5), name


def _event_cells(n: float) -> dict[tuple[int, int], float]:
    """The expected events of each ``(arm, visit)`` cell in a sample of ``n``."""
    mass: dict[tuple[int, int], float] = {}
    for probability, a, _, _, history in law.records():
        if history.event == 1:
            key = (a, history.time)
            mass[key] = mass.get(key, 0.0) + probability
    return {key: n * value for key, value in mass.items()}


def test_the_law_balances_the_arms_and_keeps_positivity() -> None:
    """D1 and D9: no (arm, visit) is thin, and no unit is near the treatment boundary."""
    events = _event_cells(study.PRIMARY_N)
    assert len(events) == 10
    low = min(events, key=lambda key: events[key])
    assert low == (0, 1)
    assert events[low] == pytest.approx(15.28, abs=0.01)
    assert min(law.g(1, *cell) for cell in law.CELLS) == pytest.approx(0.289, abs=1e-3)
    assert min(law.g(0, *cell) for cell in law.CELLS) == pytest.approx(0.5, abs=1e-3)
    total = law.expected_events(study.PRIMARY_N)
    assert total[0] == pytest.approx(85.6, abs=0.05)
    assert total[1] == pytest.approx(86.8, abs=0.05)


def test_the_primary_n_follows_the_rule() -> None:
    """At least 40 expected events per arm by visit 5, and a union bound of at most 1e-6."""
    forty = [n for n in (1_000, 2_000, 4_000) if min(law.expected_events(n).values()) >= 40]
    assert forty[0] == 1_000
    passing = [n for n in forty if law.no_event_bound(n) <= 1e-6]
    assert passing[0] == study.PRIMARY_N
    assert law.no_event_bound(1_000) == pytest.approx(2.0e-3, rel=0.01)
    assert law.no_event_bound(2_000) == pytest.approx(5.3e-7, rel=0.01)
    assert law.no_event_bound(4_000) < 1e-13


def test_the_union_bound_matches_the_exact_law() -> None:
    """The bound adds ``(1 - m)^n`` over the ten cells; it is the cell-level miss probability."""
    events = _event_cells(1.0)
    n = study.PRIMARY_N
    assert law.no_event_bound(n) == pytest.approx(sum((1.0 - m) ** n for m in events.values()))
    # Over a ~21,600-draw run the chance of any such draw stays near 1%.
    assert 21_600 * law.no_event_bound(n) < 0.012


@pytest.mark.parametrize("model", study.MODELS)
def test_the_exact_law_fit_returns_the_truth(model: str) -> None:
    """The declared nuisance models are correct up to the unsaturated additive scaled logit."""
    result = _weighted_fit(model)
    for name in study.ESTIMANDS[model]:
        standardized = abs(result.psi(name) - law.TRUTH[name]) * np.sqrt(study.PRIMARY_N)
        assert standardized / law.EFFICIENCY_SD[name] < 1e-3, name


# ------------------------------------------------------------------ refusals


def _without_events(arm: int, visit: int) -> Any:
    frame = law.draw(study.PRIMARY_N, 20263391)
    keep = ~((frame["A"] == arm) & (frame["time"] == visit) & (frame["event"] == 1))
    return frame.loc[keep].reset_index(drop=True)


@pytest.mark.parametrize("model", study.MODELS)
def test_a_node_with_no_event_is_a_recordable_failure(model: str) -> None:
    """The fit translates ``constant_node_plugin`` into a ``DataError`` the driver records."""
    frame = _without_events(0, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DataError, match="constant_node_plugin") as caught:
            study.fit(model, frame)
        # The same draw through the primary runner's replicate: it raises the same error.
        truth = dict(law.TRUTH)
        with pytest.raises(DataError, match="constant_node_plugin"):
            study._replicate((model, 0, study.PRIMARY_N), (frame, truth))
    assert recordable(caught.value)


def test_a_draw_with_events_everywhere_is_not_refused() -> None:
    frame = law.draw(study.PRIMARY_N, 20263391)
    events = frame.loc[frame["event"] == 1].groupby(["A", "time"]).size()
    assert len(events) == 10
    for model in study.MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.fit(model, frame)
        assert {result[name].inference for name in study.ESTIMANDS[model]} == {"influence_curve"}


# ------------------------------------------------------------------ the R runner


def _runner_vector(name: str) -> list[float]:
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(rf"^{name} <- c\((?P<body>[^)]*)\)", text, re.M)
    assert match is not None, name
    return [float(item) for item in match.group("body").replace("\n", " ").split(",")]


def test_the_runner_carries_the_python_bounds() -> None:
    """The survtmle runner is written by hand from the Python bounds; this reads them back."""
    text = RUNNER.read_text(encoding="utf-8")
    assert f"bound_factor <- {study.BOUND_FACTOR}" in text
    mean = _runner_vector("mean_upper")
    hazard = _runner_vector("hazard_upper")
    np.testing.assert_allclose(mean, [u for _, u in study.MEAN_BOUNDS.values()], rtol=1e-15)
    np.testing.assert_allclose(hazard, [u for _, u in study.HAZARD_BOUNDS.values()], rtol=1e-15)
    assert "reported <- c(3, 5)" in text
    assert "method = methods[[scenario]]" in text
    assert 'mean_model = "mean", hazard_model = "hazard"' in text
    assert 'ctime_formula <- "trt + W1 + W2 + t"' in text
    assert 'hazard_model = "trt*(W1 + W2) + t"' in text
    names = set(study.ESTIMANDS[study.MEAN_MODEL])
    assert {law.risk_name("arm1", 5), law.contrast_name(3)} <= names
    assert 'sprintf("risk_regimen[%s @ t=%d]", arm, horizon)' in text
    assert "ate_regimen[arm1 vs %s @ t=%d]" in text


# ------------------------------------------------------------------ D10


def test_node_one_retention_is_declared_known() -> None:
    """No unit is censored before node 1, so the pooled censoring fit reads nodes 2 to 5."""
    data = law.container(law.draw(500, 1))
    known = data.known_mechanisms
    assert known is not None
    first, *rest = known.censoring
    assert np.array_equal(np.asarray(first), np.ones(500))
    assert all(item is None for item in rest)
    assert len(rest) == law.K - 1
    assert known.treatment == ()


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


def _predictive_share(replicates: int, rate: float, draws: int) -> float:
    """The probability that a share of ``replicates`` draws reaches one half, over the pilot."""
    spread = np.sqrt(rate * (1.0 - rate) * (1.0 / draws + 1.0 / replicates))
    return float(norm.cdf((rate - properties.WITNESS_SHARE) / spread))


def _positive_equivalence(replicates: int) -> float:
    """The positive arm of N is equivalent to zero, predictive over its pilot error."""
    bias, error = NECESSITY_POSITIVE_BIAS
    grid = norm.ppf(np.linspace(0.005, 0.995, 199))
    return float(np.mean([equivalence_power(bias + g * error, replicates) for g in grid]))


def _calibration_pass(replicates: int) -> tuple[float, float]:
    """The per-cell and the family-wise pass probability of the four calibration positives."""
    pilot = CALIBRATION_PILOT_DRAWS
    arms = len(CALIBRATION_PILOT)
    mean = float(np.mean(CALIBRATION_PILOT))
    shared = pilot_sigma(pilot) ** 2 / arms
    tau = max(tau_upper(pilot, arms), tau_from_pilot(CALIBRATION_PILOT, pilot))
    cell, family = pass_probabilities(replicates, mean, shared, tau, arms)
    return float(cell), float(family)


def _family_wise(replicates: int) -> float:
    """The product over every gated cell: K, W and N."""
    witness = _predictive_share(
        replicates, WITNESS_PILOT["exceeds_bound"], CALIBRATION_PILOT_DRAWS
    ) * _predictive_share(replicates, WITNESS_PILOT["estimate_moved"], CALIBRATION_PILOT_DRAWS)
    necessity = necessity_power(NECESSITY_LIMIT, replicates) * _positive_equivalence(replicates)
    return _calibration_pass(replicates)[1] * witness * necessity


def test_the_replicates_follow_the_family_wise_product() -> None:
    """R is the smallest multiple of 400 at which the product over every gated cell is 0.95."""
    assert len(CALIBRATION_PILOT) == len(properties.CALIBRATION_LABELS) == 4
    first = next(r for r in range(400, 40_001, 400) if _family_wise(r) >= FAMILY_TARGET)
    assert first == study.REPLICATES == 7_200
    assert _family_wise(6_800) < FAMILY_TARGET <= _family_wise(7_200)
    rule = replicates_from_pilot(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)
    assert rule["screened"] == [True] * 4
    assert not rule["suspect"]
    assert rule["R"] == 7_200
    assert study.REPLICATES == study.PRIMARY_REPLICATES == properties.CALIBRATION_REPLICATES
    assert study.REPLICATES == properties.NECESSITY_REPLICATES
    assert study.REPLICATES % 400 == 0


def test_the_nominal_calibration_rule() -> None:
    """At nominal behaviour R = 2,800 suffices; the pilot-based rule asks for more."""

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


def test_the_witness_clears_one_half_by_several_errors() -> None:
    """W's gated shares sit several Monte Carlo errors above one half at n = 2,000."""
    draws = CALIBRATION_PILOT_DRAWS
    for key in ("exceeds_bound", "estimate_moved"):
        rate = WITNESS_PILOT[key]
        error = np.sqrt(rate * (1.0 - rate) / draws)
        assert (rate - properties.WITNESS_SHARE) / error > 5.0, key
        assert _predictive_share(study.REPLICATES, rate, draws) > 0.99, key
    assert _family_wise(study.REPLICATES) >= FAMILY_TARGET
    assert _calibration_pass(study.REPLICATES)[0] >= CELL_TARGET


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


def test_the_hazard_witness_reads_leaving_the_bounds_not_the_clip() -> None:
    """``hazard_exceedance`` reads ``[l, u]``, and the fit clips into a narrower interval.

    So the page and the helper describe the hazard share as leaving the bounds. Neither may
    call it the clipped share.
    """
    page = (ROOT / "docs/technical-reference/method-evidence/survival-node-bounds.md").read_text()
    prose = " ".join(page.split())
    assert "clip the initial hazards" not in prose
    assert "The initial hazards of the `hazard_model` leave the bounds on" in prose
    doc = " ".join((study.hazard_exceedance.__doc__ or "").split())
    assert "clips exactly these" not in doc
    assert "a hazard can stay inside the bounds and still be clipped" in doc
