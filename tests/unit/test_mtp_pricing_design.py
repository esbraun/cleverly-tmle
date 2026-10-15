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
import pandas as pd
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
    assert set(properties.CELLS) >= properties.REPORTING_LABELS
    assert set(properties.POOLED_LABELS) == {
        *(set(properties.CELLS) - properties.REPORTING_LABELS),
        *tuning_properties.CELLS,
    }
    assert 18 - len(properties.REPORTING_LABELS) == properties.POOLED_ARMS
    with pytest.raises(ValueError, match="arms of POOLED_LABELS"):
        properties.pooled_replicates(dict.fromkeys(properties.CELLS, 1.0))
    assert properties.pooled_replicates(dict.fromkeys(properties.POOLED_LABELS, 1.0))["R"]


def test_both_s2_records_read_one_r() -> None:
    """Design review m2: one pooled rule gives one ``R`` to every gating arm of both records."""
    gating = {
        label: r
        for label, r in properties.CALIBRATION_REPLICATES.items()
        if label not in properties.REPORTING_LABELS
    }
    assert set(gating.values()) == {properties.S2_REPLICATES}
    assert set(tuning_properties.CALIBRATION_REPLICATES.values()) == {properties.S2_REPLICATES}
    for label in properties.REPORTING_LABELS:
        assert properties.CALIBRATION_REPLICATES[label] == properties.PILOT_DRAWS


def test_each_noise_control_reads_its_own_arm() -> None:
    """Design review B2: the noise is sized from each arm's own SD, so its SE ratio is
    ``rho / sqrt(1 + k^2)`` with ``k = NOISE_FACTOR``, far below the band's lower end."""
    for module in (properties, tuning_properties):
        assert set(module.NOISE_SD) == set(module.CELLS)
        assert all(sd > 0 for sd in module.NOISE_SD.values())
    k = properties.NOISE_FACTOR
    assert k >= 0.47
    # The control's expected ratio at rho = 1, against the 0.93 band end, at 1,000 MC SEs.
    assert 1.0 / np.sqrt(1.0 + k**2) + 3.0 * tools.pilot_sigma(properties.S2_REPLICATES) < 0.93
    rows = pd.DataFrame(
        {
            "estimate": [0.1, 0.3, 0.2, 0.4],
            "truth": [0.25] * 4,
            "std_error": [0.1] * 4,
            "n": [2_000] * 4,
            "covered": [True] * 4,
            "seconds": [1.0] * 4,
        }
    )
    summary = properties.arm_summary(rows)
    spread = rows["estimate"].std(ddof=1)
    assert summary["noise_sd"] == pytest.approx(k * spread * np.sqrt(2_000))


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


# ---------------------------------------------------------------------------- the gate


@pytest.mark.parametrize(
    ("module", "record", "expected"),
    [(properties, study, 9 * 3 + 2 + 4), (tuning_properties, tuning, 6 * 4)],
    ids=["mtp-pricing", "mtp-tuning"],
)
def test_the_gating_cells_are_published_and_hold_no_inadmissible_red(
    module: object, record: object, expected: int
) -> None:
    """Design review M7: both S2 records publish under ``reporting``; this is their gate.

    ``mtp-pricing`` gates its nine non-density positives, the three density positives while
    the screen leaves them out of ``REPORTING_LABELS``, and every registered pair arm.
    """
    gating = module.GATING_CELLS  # type: ignore[attr-defined]
    cells = record.PROPERTY_CELLS  # type: ignore[attr-defined]
    published = {f"{family}/{cell}" for family, names in cells.items() for cell in names}
    assert set(gating) <= published
    assert not any(cell.startswith("reporting/") for cell in gating)
    density = 3 * 3 if module is properties and not properties.REPORTING_LABELS else 0
    assert len(gating) == expected + density
    assert tools.gating_findings(record.STUDY, gating) == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------- the learners


def _node_one_hazard(
    frame: object, bins: int, learner: type = common.SplineHazard
) -> tuple[object, np.ndarray]:
    """A hazard fitted on node 1's pooled bin rows, as the density layer stacks them."""
    from cleverly.learners.density import bin_edges

    a = frame["A1"].to_numpy()  # type: ignore[index]
    history = frame[["W1", "L1"]].to_numpy()  # type: ignore[index]
    edges = bin_edges(a, bins)
    index = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, bins - 1)
    rows = np.repeat(np.arange(len(a)), index + 1)
    k = np.concatenate([np.arange(i + 1) for i in index])
    design = np.column_stack([history[rows], k])
    target = (k == index[rows]).astype(float)
    return learner((tuple(edges),)).fit(design, target), edges


class _PooledPosition(common.SplineHazard):
    """The mutation M1 removed: the bin's position on the pooled grid, with no fixed hazard
    below the row's support and none at its top."""

    def _parts(self, X):  # type: ignore[no-untyped-def]
        history, _, _, _ = super()._parts(X)
        index = np.rint(np.asarray(X, dtype=float)[:, -1]).astype(int)
        edges = np.asarray(self.edges[0], dtype=float)
        mid = 0.5 * (edges[index] + edges[index + 1])
        position = (mid - edges[0]) / (edges[-1] - edges[0])
        return history, position, np.zeros(len(index), dtype=bool), index == edges.size - 2


def _outside_mass(hazard: object, edges: np.ndarray, l: float) -> tuple[float, float]:
    bins = edges.size - 1
    rows = np.column_stack([np.full(bins, 0.3), np.full(bins, l), np.arange(bins)])
    h = hazard.predict_proba(rows)[:, 1]  # type: ignore[attr-defined]
    mass = h * np.concatenate([[1.0], np.cumprod(1.0 - h)[:-1]])
    outside = (edges[1:] <= common.lower(l)) | (edges[:-1] >= common.upper(l))
    return float(mass.sum()), float(mass[outside].sum())


def test_the_declared_density_reads_the_declared_support() -> None:
    """Design review M1: on the pooled grid ``[4, 10]`` a bin means different doses at
    ``L = 0`` (support ``[4, 8]``) and ``L = 1`` (``[6, 10]``).  The declared hazard reads the
    support-relative position and fixes the hazards the support implies, so its density puts
    no mass outside the row's support.  The pooled-position mutation puts 0.06% to 0.9% there
    (three draws at n = 2,000, 2026-10-15), which this test requires to exceed 1e-4."""
    frame = common.sample(2_000, 61, 1)
    declared, edges = _node_one_hazard(frame, 64)
    pooled, _ = _node_one_hazard(frame, 64, _PooledPosition)
    for l in (0.0, 1.0):
        total, outside = _outside_mass(declared, edges, l)
        assert total == pytest.approx(1.0, abs=1e-9)
        assert outside < 1e-9
    assert max(_outside_mass(pooled, edges, l)[1] for l in (0.0, 1.0)) > 1e-4


def test_the_declared_quadratic_logistic_converges() -> None:
    """Design review M8: on the standardized design L-BFGS stops well inside its cap at every
    node of a ``tau = 3`` classifier fit at n = 2,000."""
    iterations: list[int] = []
    fit = common.QuadraticLogistic.fit

    def record(self, X, y, sample_weight=None):  # type: ignore[no-untyped-def]
        out = fit(self, X, y, sample_weight)
        iterations.append(out.n_iter_)
        return out

    common.QuadraticLogistic.fit = record  # type: ignore[method-assign]
    try:
        properties.calibration_fit(
            "up_learned_classifier_tau3", common.sample(2_000, 62, 3), "quadratic_logistic"
        )
    finally:
        common.QuadraticLogistic.fit = fit  # type: ignore[method-assign]
    assert len(iterations) >= 3
    assert max(iterations) < 1_000


@pytest.mark.parametrize("kind", ["dose_quantile", "quantile_regression"])
def test_the_kernel_bias_rule_picks_the_declared_bandwidth(kind: str) -> None:
    """Design review M6 (X40 plan section 2.5, m5): at n = 2,000 the relative kernel bias of
    ``f_t(theta_t)`` or ``J_t``, and of ``dPhi_t``, by quadrature, is at most 1% at the
    declared bandwidth, and the declared bandwidth is the rule's first that meets it."""
    rule = tuning_properties.declared_bandwidth(kind)
    assert rule is not None
    assert tuning.BANDWIDTH[kind] == rule
    assert all(cap.bandwidth == rule for cap in tuning.caps(kind))
    biases = tuning_properties.kernel_biases(kind, rule)
    assert max(biases.values()) <= tuning_properties.KERNEL_BIAS_LIMIT
    if kind == "dose_quantile":
        silverman = tuning_properties.kernel_biases(kind, "silverman")
        assert silverman["f1"] == pytest.approx(0.0186, abs=5e-4)
        assert silverman["f2"] == pytest.approx(0.0179, abs=5e-4)


def test_the_kernel_expectation_tends_to_the_derivative() -> None:
    """A witness for :func:`kernel_derivative`: its bias shrinks with the bandwidth, about as
    ``h^2``, and it reads the analytic derivative as ``h -> 0``."""
    law = tuning.law("dose_quantile")
    truth = law.derivative(2)
    errors = [
        np.max(np.abs(tuning_properties.kernel_derivative(law, 2, h) - truth))
        for h in (0.4, 0.2, 0.1)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.25)
    assert errors[2] < 1e-2 * np.max(np.abs(truth))


def test_the_s2_r_is_the_pooled_rule_read_off_the_published_pilots() -> None:
    """Sizing on the declared pilot readings of both S2 records, the reporting arms left out."""
    assert set(properties.PILOT_SE_RATIOS) == set(properties.POOLED_LABELS)
    reading = properties.pooled_replicates(properties.PILOT_SE_RATIOS)
    assert all(reading["screened"]) and not reading["suspect"]
    assert reading["R"] == properties.S2_REPLICATES
    assert reading["family_wise"] >= 0.95 and reading["per_cell"] >= 0.90


def test_the_registered_pairs_and_their_sizes() -> None:
    """``classifier_split`` took its declared exit; ``wrong_tau5`` moved to n = 4,000."""
    assert "classifier_split" not in study.PAIRED_CELLS
    assert all("classifier_split" not in families for _, families in properties.PAIR_SETS.values())
    assert properties.PAIR_SIZES == {"dr_tau3": 2_000, "wrong_tau3": 2_000, "wrong_tau5": 4_000}
    sizes = {
        (cell.property, cell.cell): cell.n
        for cell in properties.declared_cells()
        if cell.property == "ratio_necessity"
    }
    assert sizes[("ratio_necessity", "up_tau5__learned_classifier")] == 4_000
    assert sizes[("ratio_necessity", "up_tau3__learned_classifier")] == 2_000
