"""The pre-declaration design of the long-horizon studies, checked against the exact law.

Every number here comes from the law with no fit to a sample: the tail criterion, the law of
each quantity, every cell and scenario n, every gating scenario, each control's exact bias and
sampling SD with its displacement, and the closed-form sizing floors.  The scans over the whole
law grid and the sample pilots run in ``tests/long_horizon_pilots.py``.
"""

from __future__ import annotations

import collections
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from tests import discrete_law_longitudinal_horizon as H
from tests import long_horizon_design as D
from tests import long_horizon_sampling_sd as S
from tests.studies import long_horizon_common as C

#: Section 5.1: ``n_kappa`` of ``D*`` on ``L_R`` and on the fallback law, per failing quantity.
FAILING = {
    D.Quantity(4, 3, "high"): (336_944, 106_415, 8.3e-4),
    D.Quantity(4, 3, "high", "low"): (273_587, 121_829, 6.3e-4),
    D.Quantity(5, 3, "high"): (4_222_406, 63_602, 1.0e-3),
    D.Quantity(5, 3, "high", "low"): (3_754_107, 123_271, 4.1e-4),
    D.Quantity(5, 3, "switch"): (1_646_022, 119_056, 4.8e-4),
    D.Quantity(5, 3, "switch", "low"): (1_362_592, 68_013, 4.4e-4),
    D.Quantity(5, 3, "mix"): (173_126, 92_984, 8.0e-6),
    D.Quantity(5, 3, "taper"): (246_603, 74_816, 1.5e-5),
    D.Quantity(5, 3, "taper", "mix"): (173_469, 93_227, 8.0e-6),
}


def _quantities() -> list[D.Quantity]:
    """Every quantity a primary row or a gated calibration cell reads."""
    seen = {row.quantity: None for row in D.primary_rows()}
    seen.update({cell.quantity: None for cell in D.calibration_cells() if not cell.msm})
    return list(seen)


# ----------------------------------------------------------------------------- section 5.1
def test_the_quantities_that_fail_on_the_realistic_law_are_the_declared_ones() -> None:
    failing = {q for q in _quantities() if not D.tail(q, H.L_R).meets()}
    assert failing == set(D.FALLBACKS) == set(FAILING)
    for q, (on_realistic, _, _) in FAILING.items():
        assert D.tail(q, H.L_R).n_star == on_realistic


def test_each_fallback_law_meets_the_criterion_for_its_quantity() -> None:
    for q, (_, there, support) in FAILING.items():
        reading = D.tail(q, D.FALLBACKS[q])
        assert reading.meets(), q
        assert reading.n_star == there, q
        assert reading.n_inf <= reading.n_star, q
        assert reading.support == pytest.approx(support, rel=0.06), q


def test_the_glm_limit_curve_moves_no_quantity_to_another_law() -> None:
    """``D_inf`` is lighter-tailed than ``D*`` for every quantity on its own law."""
    for q in _quantities():
        reading = D.tail(q, D.law_of(q))
        assert reading.n_inf <= reading.n_star, q
        assert reading.meets(), q


def test_every_primary_row_has_one_gating_scenario_and_each_scenario_its_n() -> None:
    gated: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    ungated = []
    for row in D.primary_rows():
        scenario = D.gating(row)
        if scenario is None:
            ungated.append(row)
            continue
        gated[(row.study, scenario.name)].append(D.tail(row.quantity, scenario.law).n_kappa)
    # The `mix` mean at K = 5, m = 3 needs (1, 0.05, 3.5); the cross-fitted scenarios run on L_R
    # only, so their `mix` mean reports.
    assert {(row.study, str(row.quantity), row.folds) for row in ungated} == {
        ("S2", "mix K=5 m=3", 5),
        ("S3", "mix K=5 m=3", 5),
    }
    for study in D.STUDIES:
        for scenario in D.scenarios(study):
            if scenario.msm:
                need = max(
                    D.msm_tail(5, scenario.law, t).n_kappa for t in ("intercept", "duration")
                )
                assert scenario.n == D.round_n(need) == 48_000
                continue
            assert scenario.n == D.round_n(max(gated[(study, scenario.name)])), scenario.name


def test_the_policy_scenario_of_the_sdr_study_runs_at_its_gating_n() -> None:
    """``taper_vs_low`` (42,420) sets ``sdr_policy_k5`` at 44,000."""
    q = D.Quantity(5, 3, "taper", "low")
    assert D.tail(q, H.L_R).n_kappa == 42_420
    assert next(s for s in D.scenarios("S3") if s.name == "sdr_policy_k5").n == 44_000


def test_every_calibration_cell_runs_at_its_quantitys_n_on_its_law() -> None:
    cells = D.calibration_cells()
    assert collections.Counter(cell.study for cell in cells) == {"S1": 16, "S2": 17, "S3": 32}
    for cell in cells:
        reading = cell.tail()
        assert cell.n == D.round_n(reading.n_kappa), cell.name
        assert reading.support >= D.SUPPORT_FLOOR, cell.name
        if not cell.msm:
            assert cell.law == D.law_of(cell.quantity), cell.name
    smallest = min(cell.tail().support for cell in cells)
    assert smallest == pytest.approx(4.1e-6, rel=0.03)


# ----------------------------------------------------------------------------- section 5.3
def _positive_for(control: D.Control) -> D.Positive:
    """The control's gated positive: same study, family, K, m and law."""
    c = control.config
    matches = [
        p
        for p in D.positives()
        if p.study == control.study
        and p.family == control.family
        and (p.config.K, p.config.m, p.config.law) == (c.K, c.m, c.law)
        and (p.config.first == c.target or p.config.first == c.first or c.target == "")
    ]
    assert matches, control.cell
    return matches[0]


@pytest.mark.parametrize(
    "control", D.controls(), ids=lambda c: f"{c.study}-{c.family}-{c.cell}-{c.config.m}"
)
def test_each_control_reaches_three_sampling_sds_at_the_smallest_n_the_rule_admits(
    control: D.Control,
) -> None:
    reading = control.config.exact()
    assert reading.mean_function < 1e-12
    assert reading.bias == pytest.approx(control.bias, abs=6e-5)
    assert reading.sigma == pytest.approx(control.sigma, rel=1e-3)
    assert reading.d(control.n) >= D.D_FLOOR
    cap = D.NMAX_Q14 if control.n > D.NMAX else D.NMAX
    assert control.n == D.smallest_n(reading, _positive_for(control).n, cap)


@pytest.mark.parametrize(
    "positive",
    [p for p in D.positives() if p.family != "sequential_double_robustness"],
    ids=lambda p: f"{p.study}-{p.family}-{p.cell}-{p.config.m}-K{p.config.K}",
)
def test_each_positive_is_exact_and_runs_at_its_own_n(positive: D.Positive) -> None:
    """Exact bias zero (H2); ``n' = max(cell n, n_kappa of max(kappa_{D*}, kappa_curve))``."""
    c = positive.config
    frame = c.mechanism_frame(D.frame(c.K, c.m, c.law), population=False)
    settings = {"kind": c.kind, "learners": c.node_learners()}
    estimate = S.Sequential(frame, c.K, c.m, c.first, c.law, **settings).fit().psi
    if c.second:
        estimate -= S.Sequential(frame, c.K, c.m, c.second, c.law, **settings).fit().psi
    assert abs(estimate - c.truth) < 1e-12
    assert positive.n >= D.round_n(c.positive_tail())
    assert positive.n == D.round_n(c.positive_tail()) or positive.family == "double_robustness"


def test_the_multiply_robust_patterns_are_exact_and_share_one_n() -> None:
    """Both patterns read zero bias; their n is the larger tail of the two sampling curves."""
    patterns = [p for p in D.positives() if p.family == "sequential_double_robustness"]
    tails = []
    for positive in patterns:
        reading = positive.config.exact()
        assert abs(reading.bias) < 1e-12
        tails.append(S.n_kappa(reading.kappa))
    assert {p.n for p in patterns} == {D.round_n(max(tails))} == {56_000}


def test_the_monte_carlo_check_draws_the_eleven_declared_controls() -> None:
    drawn = [c for c in D.controls() if c.monte_carlo]
    assert len(drawn) == 11
    assert len({D.Stream("mc", c.cell, c.config, c.n, 400).ident for c in drawn}) == 11


def test_the_targeting_controls_are_the_untargeted_fit_with_the_positives_rung() -> None:
    controls = [c for c in D.controls() if c.family == "targeting_necessity"]
    positives = [p for p in D.positives() if p.family == "targeting_necessity"]
    assert len(controls) == len(positives) == 12
    for control, positive in zip(controls, positives, strict=True):
        assert control.config.estimator == "plugin"
        assert control.config.learners == positive.config.learners
        reading = control.config.exact()
        # The targeting move is minus the control's bias on the frame.
        assert reading.estimate - reading.truth == pytest.approx(control.bias, abs=6e-5)


# ----------------------------------------------------------------------------- sizing floors
def test_f1_needs_the_declared_replicates() -> None:
    assert D.f1_replicates(16) == D.f1_replicates(17) == 3_200
    assert D.f1_replicates(32) == 3_600
    assert D.calibration_pass(1.0, 3_200) ** 16 == pytest.approx(0.960, abs=2e-3)
    assert D.calibration_pass(1.0, 3_600) ** 32 == pytest.approx(0.975, abs=2e-3)


@pytest.mark.parametrize(
    ("replicates", "largest", "at_015"),
    [(1_000, 0.095, 0.72), (1_200, 0.108, 0.81), (2_000, 0.140, 0.97), (3_000, 0.160, 0.998)],
)
def test_the_bias_ladder(replicates: int, largest: float, at_015: float) -> None:
    """The plan prints 0.96 at R = 2,000 and bias 0.15 SD; the closed form reads 0.971."""
    root = brentq(lambda bias: D.bias_pass(bias, replicates) - 0.99, 0.0, 0.25)
    assert root == pytest.approx(largest, abs=6e-4)
    assert D.bias_pass(0.15, replicates) == pytest.approx(at_015, abs=0.006)


def test_a_truth_row_at_the_threshold_passes_with_probability_099() -> None:
    assert D.truth_row_pass(0.9365, 0.0, 1_200) >= 0.99
    assert D.bias_pass(0.108, 1_200) >= 0.99


def test_the_power_cell_reaches_090_on_both_curves() -> None:
    """On ``D*`` 48,000 suffices; on ``D_inf`` power reads 0.899 there, so the cell runs at 52,000."""
    contrast = H.contrast(5, 2, "coin", "coinlast")
    weight = D.weight(5, 2, H.L_R)
    limit = D.glm_curve(5, 2, "coin", H.L_R) - D.glm_curve(5, 2, "coinlast", H.L_R)
    sigma_inf = S.sd(weight, limit)
    assert D.round_n(D.power_n(contrast["psi"], contrast["sigma"])) == 48_000

    def power(n: int, sigma: float) -> float:
        return float(norm.cdf(abs(contrast["psi"]) * math.sqrt(n) / sigma - norm.ppf(0.975)))

    assert power(48_000, sigma_inf) < 0.90
    assert D.round_n(D.power_n(contrast["psi"], sigma_inf)) == D.POWER_N
    assert power(D.POWER_N, sigma_inf) >= 0.90


def test_the_inflated_se_control_leaves_the_band() -> None:
    """``sqrt((v_a + v_b) / Var(D_a - D_b))`` of the paired policy contrast on ``L_R``."""
    half = norm.ppf(0.995) * D.SE_RATIO_SCALE * math.sqrt(400 / 6_000)
    for K, m in ((3, 2), (4, 2), (5, 2), (3, 3), (4, 3), (5, 3)):
        policy, last = ("coin", "coinlast") if m == 2 else ("mix", "mixlast")
        inflation = H.contrast(K, m, policy, last)["inflation"]
        assert 1.36 <= inflation <= 1.401
        assert inflation - half > 1.07


# ----------------------------------------------------------------------------- oracle curve
#: Section 5.3: the oracle-curve SE ratio and coverage of ``D*`` for the three-level static top
#: plan at K = 5 on ``L_R``, with the draws the plan used (10,000 to 20,000).
SUPPORT_LOSS = {4_000: (0.428, 0.947), 16_000: (0.585, 0.937), 64_000: (0.764, 0.930)}
#: This test draws 2,000 per n; the SE-ratio tolerance is three Monte Carlo SDs of that.
ORACLE_DRAWS = 2_000


def test_the_support_loss_predictions_are_the_declared_oracle_readings() -> None:
    """The reporting tripwire reads [prediction - 0.10, prediction + 0.10] and +-0.02 on coverage.

    The plan prints SE ratio 0.428, 0.585, 0.764 and coverage 0.947, 0.937, 0.930.  At 2,000
    draws this test reads 0.434, 0.569, 0.792 and 0.946, 0.943, 0.932 (seed ``PILOT_SEED``):
    every reading is inside three Monte Carlo SDs of the plan's number, so the prediction the
    pages print holds, and the short standard error is a property of the law.
    """
    weight = D.weight(5, 3, H.L_R)
    curve = D.eif(5, 3, "high", H.L_R)
    tolerance = 3 / math.sqrt(2 * ORACLE_DRAWS)
    for n, (ratio, coverage) in SUPPORT_LOSS.items():
        reading = D.oracle_reading(weight, curve, n, ORACLE_DRAWS)
        assert reading.se_ratio == pytest.approx(ratio, abs=tolerance)
        assert reading.coverage == pytest.approx(coverage, abs=0.02)
        assert reading.se_ratio < 0.80


@pytest.mark.parametrize(
    "q", [D.Quantity(4, 2, "always", "never"), D.Quantity(4, 3, "respond", "low")], ids=str
)
def test_the_root_n_control_rung_passes_under_the_shipped_rule(q: D.Quantity) -> None:
    """The rung at 4,000 resolves with probability above 0.99 on the oracle curve of ``D_inf``.

    Rule (``property_verdicts.py``, the ``small`` branch of the efficiency summary)::

        summary.loc[small, "passed"] = (
            summary.loc[small, "coverage_ci_upper"] < 1.0 - margins.alpha
        ) | (summary.loc[small, "coverage_ci_lower"] >= margins.coverage_floor)

    with ``alpha = 0.05`` and ``coverage_floor = 0.90``.  The rung is below the regime
    (``n_kappa`` 5,320 and 5,183) and the oracle coverage there is near nominal, so the lower
    limit at R = 1,000 clears 0.90 and the rung passes with probability 1.0.  The rung at 12,000
    lies above the regime and is a positive; its oracle SE ratio reads inside (0.80, 1.20).
    """
    tail = D.tail(q, H.L_R)
    assert tail.n_star in (5_320, 5_183)
    assert 4_000 < tail.n_kappa < 12_000
    weight = D.weight(q.K, q.m, H.L_R)
    limit = D.limit_curve(q, H.L_R)
    control = D.oracle_reading(weight, limit, 4_000, ORACLE_DRAWS)
    assert 0.90 < control.coverage < 0.97
    assert D.control_rung_pass(control.coverage, 1_000) >= 0.99
    positive = D.oracle_reading(weight, limit, 12_000, ORACLE_DRAWS)
    assert 0.80 < positive.se_ratio < 1.20
    assert positive.coverage > 0.9394


def test_the_rung_rule_cannot_fail_at_1000_replicates_but_can_at_100() -> None:
    """Reading: the 99 percent interval is narrower than the gap between 0.90 and 0.95.

    At R = 1,000 the interval has width at most 0.042, so it cannot reach 0.95 and fall under
    0.90: the rule passes with probability above 0.999 at every true coverage from 0.85 to 0.99.
    At R = 100 the interval is wider than 0.05 and the rule can fail, so the rule has teeth.
    """
    grid = np.arange(0.85, 0.995, 0.01)
    assert min(D.control_rung_pass(float(c), 1_000) for c in grid) > 0.999
    assert D.control_rung_pass(0.93, 100) < 0.9


def test_the_noise_control_leaves_the_band_in_every_calibration_cell() -> None:
    """``(1 + sigma_eff^2 / sigma_inf^2)^(-1/2)`` plus the half-width h stays below 0.93 at R = 6,000.

    Recorded over the 65 gated cells: 0.709 to 0.719 (``sigma_inf / sigma_eff`` 1.005 to 1.033),
    far below the 2.15 that the plan's limit allows.
    """
    half = norm.ppf(0.995) * D.SE_RATIO_SCALE * math.sqrt(400 / 6_000)
    cells = D.calibration_cells()
    assert len(cells) == 65
    for cell in cells:
        sigma_eff, sigma_inf = D.calibration_sigmas(cell)
        assert sigma_inf < 2.15 * sigma_eff, cell.name
        assert D.noise_control_ratio(sigma_eff, sigma_inf) + half < 0.93, cell.name


# ----------------------------------------------------------------------------- the pilots
def test_every_pilot_stream_has_its_own_checkpoint_cell_and_seeds() -> None:
    streams = D.streams()
    assert len({s.ident for s in streams}) == len(streams)
    counts = collections.Counter(s.family for s in streams)
    assert counts["mc"] == 11
    assert counts["f2"] == len(D.calibration_cells()) == 65
    assert counts["overfit"] == 2
    first = streams[0]
    assert D.stream_seed(first.ident, 0) == D.stream_seed(first.ident, 0)
    assert D.stream_seed(first.ident, 0) != D.stream_seed(first.ident, 1)
    assert all(s.cost() > 0 for s in streams)


@pytest.mark.parametrize("kind", ["constant", "confounder_blind", "wblind"])
def test_a_samples_wrong_mechanism_is_the_laws(kind: str) -> None:
    """On the weighted frame, the history lookup a sample uses equals the mass-weighted one."""
    K, m = 3, 3
    frame = D.frame(K, m, H.L_R)
    config = D.Config("LTMLE", K, m, H.L_R, "respond", mechanism=kind)
    looked_up = config.mechanism_frame(frame, population=True)
    direct = config.mechanism_frame(frame, population=False)
    columns = [f"g{t}_{k}" for t in range(1, K + 1) for k in range(m)]
    columns += [f"pc{t}" for t in range(1, K + 1)]
    np.testing.assert_allclose(
        looked_up[columns].to_numpy(), direct[columns].to_numpy(), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "config",
    [
        D.Config("plugin", 3, 2, H.L_R, "respond", learners="int"),
        D.Config("LTMLE", 3, 3, H.L_R, "respond", "low", learners="int", mechanism="constant"),
        D.Config("LTMLE", 3, 2, H.L_R, "respond", "never", learners="offset"),
        D.Config(
            "SDR",
            3,
            3,
            H.L_R,
            "mix",
            learners="int",
            mechanism="complement",
            complement_nodes=(1, 2, 3),
        ),
        D.Config("LTMLE", 3, 2, H.L_R, "mix_reversed", target="coin"),
    ],
    ids=["plugin", "contrast", "offset", "complement", "policy"],
)
def test_a_sample_fit_of_each_configuration_kind(config: D.Config) -> None:
    """A fit of one draw lies within five sampling SDs of the exact estimator's value."""
    n = 4_000
    sample = H.draw(config.K, config.m, n, 7, config.law)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = config.fit_sample(sample)
    exact = config.exact()
    assert abs(out["estimate"] - exact.estimate) < 5 * exact.sigma / math.sqrt(n)


def test_a_sample_fit_reads_the_exact_estimator_on_the_frame() -> None:
    """``fit_sample`` on the weighted frame returns the exact reading's estimate."""
    config = D.Config("LTMLE", 3, 3, H.L_R, "respond", "low", learners="int", mechanism="constant")
    frame = D.frame(3, 3, H.L_R)
    data = config.mechanism_frame(frame, population=True)
    result = C.fit(data, 3, 3, ("low", "respond"), weights="w", outcome_learner=C.InterceptOnly())
    assert float(result["ate_regimen[respond vs low]"].psi) == pytest.approx(
        config.exact().estimate, abs=1e-11
    )
    assert isinstance(data, pd.DataFrame)


# ----------------------------------------------------------------------------- rev 4.2: gates
def test_the_gated_rows_are_the_rows_in_their_gating_scenario_only() -> None:
    rows = D.GATED_ROWS
    assert len(rows) == len(set(rows)) == 120
    by_study = collections.Counter(study for study, _, _ in rows)
    assert by_study == {"S1": 28, "S2": 43, "S3": 49}
    for row in D.primary_rows():
        scenario = D.gating(row)
        q = row.quantity
        name = f"ey_regimen[{q.first}]" if not q.second else f"ate_regimen[{q.first} vs {q.second}]"
        if scenario is not None:
            assert (row.study, scenario.name, name) in rows
    # The `taper` mean reports in `sdr_policy_k5` (n_kappa 246,603 against 44,000) and gates in
    # its own scenario; the same holds for the `mix` mean at K = 5 in the other two arms.
    assert ("S3", "sdr_policy_k5", "ey_regimen[taper]") not in rows
    assert ("S3", "sdr_policy_k5_taper", "ey_regimen[taper]") in rows
    assert ("S3", "sdr_policy_k5", "ate_regimen[taper vs low]") in rows
    assert ("S2", "categorical_k5", "ey_regimen[mix]") not in rows
    assert ("S2", "categorical_k5_mix", "ey_regimen[mix]") in rows
    assert ("S2", "categorical_k5_crossfit", "ey_regimen[mix]") not in rows
    # Both duration coefficients of the S1 MSM scenario are gated.
    assert {name for _, scenario, name in rows if scenario == "binary_k5_msm"} == {
        "msm_regimen[(intercept)]",
        "msm_regimen[duration]",
    }


def test_the_gated_cells_are_every_positive_and_control_with_a_verdict() -> None:
    from tests.studies import long_horizon_properties as properties
    from tests.studies.evidence.property_verdicts import CALIBRATION_CONTROL_KINDS
    from tests.studies.evidence.simultaneous import joint_cells

    cells = D.GATED_CELLS
    assert len(cells) == len(set(cells)) == 294
    for study in D.STUDIES:
        declared: set[tuple[str, str, str]] = set()
        for fit_set in properties.fit_sets(study):
            for arm in fit_set.arms:
                if arm.role == "diagnostic":
                    continue
                names = (
                    list(joint_cells(arm.cell))
                    if arm.read == "band"
                    else [arm.cell, *([arm.inflated] if arm.inflated else [])]
                )
                family = "simultaneous_coverage" if arm.read == "band" else fit_set.family
                declared.update((study, family, name) for name in names)
                if fit_set.family == "interval_calibration":
                    label = arm.cell.removesuffix("__known_mechanism")
                    declared.update(
                        (study, family, f"{label}__{kind}") for kind in CALIBRATION_CONTROL_KINDS
                    )
        label = properties.rate_label(study)
        declared.update(
            (study, "root_n_rate", f"{label}__{kind}") for kind in properties.RATE_KINDS
        )
        assert {c for c in cells if c[0] == study} == declared, study
    # Reporting cells have no verdict and are not gated.
    assert not [c for c in cells if c[1] in ("support_loss", "learned_mechanism")]
    assert not [c for c in cells if "n4000" in c[2] and c[1] == "sequential_double_robustness"]


def _summaries(study: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Passing summaries holding exactly the study's gated rows and cells."""
    performance = pd.DataFrame(
        [
            {
                "implementation": "subject",
                "scenario": scenario,
                "estimand": estimand,
                "passed": True,
            }
            for tag, scenario, estimand in D.GATED_ROWS
            if tag == study
        ]
    )
    properties = pd.DataFrame(
        [
            {"property": family, "cell": cell, "passed": True}
            for tag, family, cell in D.GATED_CELLS
            if tag == study
        ]
    )
    return performance, properties


def test_red_gated_names_a_red_truth_row_a_red_cell_and_an_absent_one() -> None:
    performance, properties = _summaries("S2")
    assert D.red_gated("S2", "subject", performance, properties) == []
    performance.loc[3, "passed"] = False
    properties.loc[7, "passed"] = False
    properties = properties.drop(index=11)
    # A reporting cell that is red gates nothing.
    properties = pd.concat(
        [properties, pd.DataFrame([{"property": "support_loss", "cell": "x", "passed": False}])]
    )
    red = D.red_gated("S2", "subject", performance, properties)
    assert len(red) == 3
    assert sum(line.startswith("truth row") and line.endswith("red") for line in red) == 1
    assert sum(line.startswith("property cell") and line.endswith("red") for line in red) == 1
    assert sum(line.endswith("absent") for line in red) == 1
    # Another study's gated cells are not this study's.
    assert D.red_gated("S1", "subject", *_summaries("S1")) == []


@pytest.mark.parametrize("study", ["S1", "S2", "S3"])
def test_no_gated_truth_row_or_property_cell_is_red_on_the_committed_summaries(study: str) -> None:
    """Plan 5.6: the studies register under ``reporting``, and these members block the merge.

    The summaries are the study's committed ``performance-tests.csv`` and ``properties.csv``.  The
    test skips until the study commits them, and runs from the day it does.
    """
    record = D.STUDIES[study].STUDY
    performance_file = record.artifact("performance-tests.csv")
    properties_file = record.artifact("properties.csv")
    missing = [f.name for f in (performance_file, properties_file) if not f.exists()]
    if missing:
        pytest.skip(f"{record.slug} has not committed {missing} yet: the declared run is pending")
    red = D.red_gated(
        study,
        record.implementation,
        pd.read_csv(performance_file),
        pd.read_csv(properties_file),
    )
    assert not red, "\n".join(red)


# ----------------------------------------------------------------------------- rev 4.2: F2
def test_f2_uses_each_arms_measured_coverage() -> None:
    """Review minor 2: the Wald model ``2 Phi(1.96 rho) - 1`` reads high for the cloud arms."""
    model = 2 * norm.cdf(1.96 * 0.967) - 1
    assert model == pytest.approx(0.9415, abs=5e-4)
    assert D.calibration_pass(0.967, 6_000, coverage=model) == pytest.approx(
        D.calibration_pass(0.967, 6_000)
    )
    # The measured 0.9353 (S2 `policy_vs_ref` K = 4) is 0.6 points lower, and passes less often.
    assert D.calibration_pass(0.967, 6_000, coverage=0.9353) < D.calibration_pass(0.967, 6_000)
    ratios, variances = [0.99, 0.98, 1.0], [1e-4] * 3
    wald = D.f2_family_pass(ratios, variances, 4_000)
    measured = D.f2_family_pass(
        ratios, variances, 4_000, coverages=[0.9353] * 3, coverage_errors=[0.004] * 3
    )
    assert measured < wald
    # Measured coverage at the model value reproduces the Wald F2 to its Monte Carlo error.
    at_model = D.f2_family_pass(
        ratios,
        variances,
        4_000,
        coverages=[2 * norm.cdf(1.96 * r) - 1 for r in ratios],
        coverage_errors=[1e-9] * 3,
    )
    assert at_model == pytest.approx(wald, abs=0.02)
    with pytest.raises(ValueError, match="one value per arm"):
        D.f2_family_pass(ratios, variances, 4_000, coverages=[0.95])
