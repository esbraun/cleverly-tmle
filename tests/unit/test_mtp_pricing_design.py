"""The pre-run design of ``mtp-pricing`` and ``mtp-tuning``: truths, shares, sizing and controls.

Every number the two studies declare before their pilots is recomputed here: law P's
quadrature truth at ``tau = 2`` against a fresh Monte Carlo, the pinned Monte Carlo truths at
``tau = 3`` and 5, the binding shares of the two plans (``X40-plan.md`` section 5.3), the
cumulative ratio at ``tau = 5``, the exact ratio of the primary rows, the 18-arm pooled rule
and the learner screen, the classifier-split control's leak, and on law Q the derived
controls' expected ratios and the node-1 derivative the declared outcome learner recovers
(review m3).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests import law_q_tuned as lawq
from tests.studies import canonical_mtp_pricing as study
from tests.studies import canonical_mtp_tuning as tuning
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_pricing_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies import mtp_tuning_properties as tuning_properties
from tests.studies.evidence.registry import ROOT, registered

pytestmark = pytest.mark.xdist_group("mtp_pricing_design")

UP, DOWN, NATURAL = "price_up", "price_down", "natural_prices"


# ---------------------------------------------------------------------------- truths


def test_the_quadrature_truths_are_stable_and_the_plans_move_the_mean() -> None:
    means = {label: common.quadrature_truth(label) for label in common.LABELS}
    assert means[UP] < means[NATURAL] - 0.05
    assert means[DOWN] > means[NATURAL] + 0.05
    assert common.truth(2, f"ate[{UP}]") == pytest.approx(means[UP] - means[NATURAL])


def test_the_monte_carlo_agrees_with_the_quadrature_at_tau_two() -> None:
    drawn = common.monte_carlo_truth(2, draws=2_000_000)
    for label in common.LABELS:
        for kind, revenue in (("ey", False), ("er", True)):
            value, se = drawn[f"{kind}[{label}]"]
            assert abs(value - common.quadrature_truth(label, revenue)) < 4.0 * se


@pytest.mark.parametrize("tau", [3, 5])
def test_the_pinned_truths_recompute_on_fresh_draws(tau: int) -> None:
    pinned = common.TRUTHS[tau]
    assert set(pinned) == {
        *(f"{kind}[{label}]" for kind in ("ey", "er") for label in common.LABELS),
        *(f"ate[{label}]" for label in common.LABELS[1:]),
    }
    # X12 6.12: the Monte Carlo SE sits below 0.1 of the bias margin, 0.25 of the oracle
    # cell's spread at n = 2,000 (properties.NOISE_SD / sqrt(2,000)).
    spread = properties.NOISE_SD[f"up_oracle_tau{tau}"] / np.sqrt(2_000)
    for key, (_, se) in pinned.items():
        if key.startswith("ey"):
            assert se < 0.1 * 0.25 * spread
    fresh = common.simulate(1_000_000, 4_242 + tau, tau)
    for label, (p, _) in fresh.items():
        value, se = pinned[f"ey[{label}]"]
        mc = p.std(ddof=1) / np.sqrt(p.size)
        assert abs(p.mean() - value) < 5.0 * np.hypot(se, mc)


# ---------------------------------------------------------------------------- shares


@pytest.fixture(scope="module")
def observed() -> object:
    return common.sample(1_000_000, 5, 5)


def test_both_boundaries_bind_at_every_node(observed: object) -> None:
    """Section 5.3: ``up`` binds 0.195 to 0.204, ``down`` 0.176 to 0.189; assert >= 0.15."""
    for t in range(1, 6):
        l, a = observed[f"L{t}"].to_numpy(), observed[f"A{t}"].to_numpy()
        up = float(np.mean(a >= common.upper(l) / common.UP))
        down = float(np.mean(a < common.lower(l) / common.DOWN))
        assert up >= 0.15 and down >= 0.15
        assert 0.193 <= up <= 0.206
        assert 0.172 <= down <= 0.191


def test_the_cumulative_ratio_at_five_nodes(observed: object) -> None:
    """The cumulative ratio keeps mean one and reaches tens at five nodes; no truncation."""
    w1 = observed["W1"].to_numpy()
    cumulative, gap = np.ones(len(observed)), np.zeros(len(observed))
    quantiles = {}
    for t in range(1, 6):
        l, a = observed[f"L{t}"].to_numpy(), observed[f"A{t}"].to_numpy()
        cumulative = cumulative * common.ratio(UP, a, common.price_mean(w1, l, gap), l)
        gap = a - common.centre(l)
        assert abs(cumulative.mean() - 1.0) < 0.02
        quantiles[t] = np.quantile(cumulative, [0.5, 0.99, 0.999])
    assert quantiles[5][2] > 20.0
    assert quantiles[1][2] < 4.0


# ---------------------------------------------------------------------------- the fits


def test_the_primary_ratio_is_the_closed_form() -> None:
    frame = common.sample(1_000, 3, 3)
    result = study.fit(frame, 3)
    w1 = frame["W1"].to_numpy()
    for label in (UP, DOWN):
        gap = np.zeros(len(frame))
        for t in range(1, 4):
            l, a = frame[f"L{t}"].to_numpy(), frame[f"A{t}"].to_numpy()
            exact = common.ratio(label, a, common.price_mean(w1, l, gap), l)
            np.testing.assert_allclose(
                result.fits[label].node_ratio[:, t - 1], exact, rtol=0, atol=1e-12
            )
            gap = a - common.centre(l)


def test_the_row_fold_control_leaks_and_restores() -> None:
    from cleverly.learners import density_ratio
    from cleverly.longitudinal import sequential

    seen: list[np.ndarray] = []
    fit_learner = density_ratio.fit_learner

    def record(learner, design, target, weight, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(np.asarray(kwargs["groups"]))
        return fit_learner(learner, design, target, weight, **kwargs)

    frame = common.sample(600, 7, 2)
    density_ratio.fit_learner = record
    try:
        with properties.row_folds():
            study.fit(
                frame,
                2,
                (NATURAL, UP),
                n_folds=5,
                **properties.wrong_learners(),
                **properties.ratio_settings("classifier", 2, "quadratic_logistic"),
            )
    finally:
        density_ratio.fit_learner = fit_learner
    assert sequential.classifier_ratio is density_ratio.classifier_ratio
    # Each training stack holds units with one block only: the leak the control needs.
    one_block = [np.mean(np.bincount(units)[np.unique(units)] == 1) for units in seen]
    assert seen and min(one_block) > 0.2


# ---------------------------------------------------------------------------- sizing


@pytest.mark.parametrize(
    ("arms", "pilot", "expected"),
    [
        (18, 2_000, (6_000, 6_400, 8_000)),
        (16, 2_000, (6_000, 6_800, 8_400)),
        (18, 3_000, (5_200, 5_600, 6_800)),
    ],
)
def test_the_pooled_rule_reproduces_the_rev3_probe(
    arms: int, pilot: int, expected: tuple[int, int, int]
) -> None:
    got = tuple(tools.smallest_replicates(m, pilot, arms)[0] for m in (1.000, 1.005, 1.010))
    assert got == expected


def test_the_pooled_rule_reads_all_eighteen_arms() -> None:
    assert len(properties.CELLS) + len(tuning_properties.CELLS) == properties.POOLED_ARMS
    with pytest.raises(ValueError, match="18 arms"):
        properties.pooled_replicates([1.0] * 12)
    assert properties.pooled_replicates([1.0] * 18)["R"] is not None


def test_the_learner_screen_keeps_a_calibrated_learner() -> None:
    assert tools.pilot_sigma(properties.SCREEN_DRAWS) == pytest.approx(0.0169, abs=1e-4)
    assert tools.learner_screen_pass(properties.SCREEN_DRAWS) > 0.99
    arms = {
        "a": {"se_ratio": 1.02, "standardized_bias": 0.05},
        "b": {"se_ratio": 1.00, "standardized_bias": 0.08},
        "c": {"se_ratio": 1.08, "standardized_bias": 0.00},
    }
    assert properties.screen_choice(arms) == "a"
    assert properties.screen_choice({"c": arms["c"]}) is None


def test_the_declared_sizes() -> None:
    assert set(properties.CELLS) == set(study.CALIBRATION_LABELS)
    assert all(r % 400 == 0 for r in properties.CALIBRATION_REPLICATES.values())
    assert properties.CLASSIFIER_SPLIT_PILOT_DRAWS == 400
    assert properties.PAIR_REPLICATES == 1_000
    # The pairs' smoke biases: ratio of one at 0.10 and 0.14 against the 0.25 margin.
    assert tools.necessity_power(4.6, properties.PAIR_REPLICATES) > 0.99


def test_the_seeds_are_new() -> None:
    mine = {study.STUDY.slug, tuning.STUDY.slug}
    others = {
        seed
        for record in registered()
        if record.slug not in mine
        for seed in (record.seed, record.resampling_seed)
        if seed is not None
    }
    seeds = {study.SEED, study.RESAMPLING_SEED, tuning.SEED, tuning.RESAMPLING_SEED}
    assert len(seeds) == 4
    assert not seeds & (others | {common.TRUTH_SEED})


def test_the_records() -> None:
    from tests.canonical.mtp_pricing.regenerate import REFERENCE

    assert study.STUDY.comparator_state == "pending"
    assert study.STUDY.pending_reference == "lmtp"
    assert tuning.STUDY.comparator_state == "none"
    assert tuning.STUDY.truth_varies_by_replicate
    runner = (ROOT / "tests" / "canonical" / REFERENCE.runner).read_text(encoding="utf-8")
    estimators = re.search(r"estimators <- c\((.*?)\n\)", runner, re.S)
    horizons = re.search(r"horizons <- c\((.*?)\n\)", runner, re.S)
    assert estimators is not None and horizons is not None
    assert dict(re.findall(r'(\w+) = "(\w+)"', estimators.group(1))) == {
        scenario: spec[1] for scenario, spec in study.SCENARIOS.items()
    }
    assert {k: int(v) for k, v in re.findall(r"(\w+) = (\d+)L", horizons.group(1))} == {
        scenario: spec[0] for scenario, spec in study.SCENARIOS.items()
    }


@pytest.mark.parametrize("module", [properties, tuning_properties])
def test_the_budget_is_recorded(module: object) -> None:
    budget = module.BUDGET  # type: ignore[attr-defined]
    assert {"seconds", "cpu_hours", "peak_gib"} <= set(budget)
    assert budget["cpu_hours"]["python_total"] > 0


# ---------------------------------------------------------------------------- law Q


@pytest.fixture(scope="module")
def law_q_fits() -> dict[str, list[object]]:
    """Population-target fits at n = 10,000: two under the quantile, one under regression."""
    return {
        "dose_quantile": [
            tuning.fit(lawq.sample(10_000, 900 + seed), "dose_quantile", "population")
            for seed in range(2)
        ],
        "quantile_regression": [
            tuning.fit(lawq.sample(10_000, 900), "quantile_regression", "population")
        ],
    }


@pytest.mark.parametrize("kind", ["dose_quantile", "quantile_regression"])
def test_the_derived_controls_expected_ratio(
    kind: str, law_q_fits: dict[str, list[object]]
) -> None:
    """``lawQ.log``: the curve without the tuning term reports 0.612 (quantile) and 0.832
    (regression) of the full curve's SE.  At n = 10,000 the fitted ratio lies within 0.03."""
    ratios = [
        tuning_properties.term_dropped_se(result) / result[tuning.ESTIMAND].std_error
        for result in law_q_fits[kind]
    ]
    assert np.mean(ratios) == pytest.approx(
        tuning_properties.EXPECTED_CONTROL_RATIO[kind], abs=0.03
    )


@pytest.mark.parametrize("kind", ["dose_quantile", "quantile_regression"])
def test_the_declared_learner_recovers_the_node_one_derivative(
    kind: str, law_q_fits: dict[str, list[object]]
) -> None:
    """Review m3: node 1's kernel derivative, read through the declared ``SplineFirst``
    outcome regression, recovers the oracle's :math:`\\partial_1\\Phi` (``LawQ.derivative``).

    The kernel estimate converges at ``n^(-2/5)``.  At n = 10,000 the mean estimate lies
    within 10% of the truth's largest coordinate; the probe read 0.0357 against 0.0357
    (quantile) and (0.0431, -0.0047) against (0.0420, -0.0060) (regression).
    """
    truth = np.asarray(tuning.law(kind).derivative(1), dtype=float)
    estimates = np.mean(
        [np.ravel(result.fits[tuning.LABEL].tuning_derivatives[0]) for result in law_q_fits[kind]],
        axis=0,
    )
    assert np.max(np.abs(estimates - truth)) < 0.10 * np.max(np.abs(truth))
