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
