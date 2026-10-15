"""The law-side values that the X32b witnesses and studies compare with, pinned.

Two exact laws, ``BINARY`` and ``CONTINUOUS`` of ``tests/discrete_law_repeated.py`` (``T = 2``),
realise each cell probability exactly in ``N = 2 * 4**7`` rows.  A fit with a saturated learner
therefore equals the population value that these laws state.  The values below are the exact
witnesses E1, E2, E2b, E3, E5 and E6 and the mutations N8' and N9, as the plan declares them.

The ``T = 3`` law of ``tests/studies/ltmle_repeated_law.py`` states the truths, influence
curves and control limits of the studies.  A control limit is in units of the efficient
influence curve's standard error at a sample size: ``gap / (sd / sqrt(n))``.  Each value has a
source in the plan that names the probe that computed it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests import discrete_law_repeated as laws
from tests.studies import ltmle_repeated_law as law3

PLANS = ("never", "always", "treat_if_high")
FULL_HISTORY = ["W", "y1lev", "L2"]
W_ONLY = ["W"]
LAWS = (laws.BINARY, laws.CONTINUOUS)
IDS = [law.name for law in LAWS]

#: ``E2``: the ``W``-only plug-in misses the truth at ``h = 2`` by these amounts.
PLUG_IN_GAP = {
    "binary": (+0.114063, -0.105031, -0.002363),
    "continuous": (-0.021510, +0.028745, -0.006857),
}
#: ``E2b``: the doubly robust estimator that is not sequentially so misses at ``P_mix``.
NON_SEQUENTIAL_GAP = {
    "binary": (-0.001563, -0.001815, -0.001363),
    "continuous": (-0.000297, -0.005093, +0.001192),
}
#: ``N8'``: scaling the correction by 0.9 moves the ``E2`` estimate at ``h = 2``.
N8_MOVE = {
    "binary": (+0.011406, -0.010503, -0.000236),
    "continuous": (-0.002151, +0.002874, -0.000686),
}
#: ``N9``: dropping the censoring factor moves the ``E2`` estimate at ``h = 2``.
N9_MOVE = {
    "binary": (+0.088281, -0.084856, +0.001071),
    "continuous": (-0.014978, +0.022700, -0.006666),
}
#: ``E6``: with ``omega := 1`` the ``W``-only one-step misses the truth at ``h = 2``.
OMEGA_GAP = {
    "binary": {0.5: +0.041320, 2.0: -0.023484},
    "continuous": {0.5: -0.003991, 2.0: +0.007983},
}


def _truth(law: laws.RepeatedLaw, plan: str, horizon: int) -> float:
    return law.truth[f"ey_regimen[{plan} @ t={horizon}]"]


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_e1_e2_a_saturated_or_w_only_regression_with_known_ratios_is_exact(
    law: laws.RepeatedLaw,
) -> None:
    """Both estimates equal the truth at every horizon: the correction carries the whole gap."""
    for plan in PLANS:
        for horizon in (1, 2):
            truth = _truth(law, plan, horizon)
            saturated, _ = law.sdr_population(plan, horizon, key_1=W_ONLY, key_2=FULL_HISTORY)
            w_only, _ = law.sdr_population(plan, horizon, key_1=W_ONLY, key_2=W_ONLY)
            assert saturated == pytest.approx(truth, abs=1e-12), (plan, horizon)
            assert w_only == pytest.approx(truth, abs=1e-12), (plan, horizon)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_e2_the_w_only_plug_in_misses_the_truth_by_a_declared_nonzero_gap(
    law: laws.RepeatedLaw,
) -> None:
    for plan, expected in zip(PLANS, PLUG_IN_GAP[law.name], strict=True):
        plug_in, _ = law.sdr_population(plan, 2, key_1=W_ONLY, key_2=W_ONLY, scale=0.0)
        gap = plug_in - _truth(law, plan, 2)
        assert gap == pytest.approx(expected, abs=6e-7), plan
        assert abs(gap) > 1e-3, plan
        # At h = 1 the node-1 regression on W is saturated, so the gap is zero by construction.
        first, _ = law.sdr_population(plan, 1, key_1=W_ONLY, key_2=W_ONLY, scale=0.0)
        assert first == pytest.approx(_truth(law, plan, 1), abs=1e-12)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_e2b_sequential_double_robustness_at_a_mixed_pattern(law: laws.RepeatedLaw) -> None:
    """Node 2 wrong with the true ratio and node 1 right with a wrong ratio: still exact.

    The non-sequential estimator, which regresses node 1 on ``m_2`` and not on ``phi_2``,
    misses by a declared nonzero amount at the same nuisances.
    """
    options: dict[str, Any] = {"key_1": W_ONLY, "key_2": W_ONLY, "ratio_1": "intercept"}
    for plan, expected in zip(PLANS, NON_SEQUENTIAL_GAP[law.name], strict=True):
        sequential, _ = law.sdr_population(plan, 2, **options)
        control, _ = law.sdr_population(plan, 2, sequential=False, **options)
        assert sequential == pytest.approx(_truth(law, plan, 2), abs=1e-12), plan
        assert control - _truth(law, plan, 2) == pytest.approx(expected, abs=6e-7), plan
        assert abs(expected) > 1e-4


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_n8_the_scaled_correction_moves_the_estimate_and_every_curve(
    law: laws.RepeatedLaw,
) -> None:
    for plan, expected in zip(PLANS, N8_MOVE[law.name], strict=True):
        base, base_curve = law.sdr_population(plan, 2, key_1=W_ONLY, key_2=W_ONLY)
        moved, moved_curve = law.sdr_population(plan, 2, key_1=W_ONLY, key_2=W_ONLY, scale=0.9)
        assert moved - base == pytest.approx(expected, abs=6e-7), plan
        for horizon in (1, 2):
            _, left = law.sdr_population(plan, horizon, key_1=W_ONLY, key_2=W_ONLY)
            _, right = law.sdr_population(plan, horizon, key_1=W_ONLY, key_2=W_ONLY, scale=0.9)
            assert np.max(np.abs(left - right)) >= 0.13, (plan, horizon)
        assert np.max(np.abs(moved_curve - base_curve)) >= 1.0


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_n9_dropping_the_censoring_factor_moves_the_sdr_estimate_and_curve(
    law: laws.RepeatedLaw,
) -> None:
    for plan, expected in zip(PLANS, N9_MOVE[law.name], strict=True):
        base, _ = law.sdr_population(plan, 2, key_1=W_ONLY, key_2=W_ONLY)
        moved, _ = law.sdr_population(plan, 2, key_1=W_ONLY, key_2=W_ONLY, omega=False)
        assert moved - base == pytest.approx(expected, abs=6e-7), plan
        for horizon in (1, 2):
            _, left = law.sdr_population(plan, horizon, key_1=W_ONLY, key_2=FULL_HISTORY)
            _, right = law.sdr_population(
                plan, horizon, key_1=W_ONLY, key_2=FULL_HISTORY, omega=False
            )
            assert np.max(np.abs(left - right)) >= 0.33, (plan, horizon)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_e5_e6_the_ipsi_one_step_is_exact_under_single_robustness(law: laws.RepeatedLaw) -> None:
    """Saturated ``m``, or ``W``-only ``m`` with the true mechanisms, gives the tilt truth."""
    for delta in (0.5, 2.0):
        for horizon in (1, 2):
            truth = law.tilt_counterfactual_value(delta, horizon)
            for key in ("full", "W"):
                estimate, curve = law.ipsi_population(delta, horizon, key=key)
                assert estimate == pytest.approx(truth, abs=1e-12), (delta, horizon, key)
                assert abs(float(np.sum(law.probs * curve))) < 1e-12


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_e6_n9_without_the_dropout_weights_the_w_only_one_step_misses(
    law: laws.RepeatedLaw,
) -> None:
    """The precondition of E6: ``omega := 1`` leaves a nonzero gap at ``h = 2``."""
    for delta, expected in OMEGA_GAP[law.name].items():
        truth = law.tilt_counterfactual_value(delta, 2)
        estimate, _ = law.ipsi_population(delta, 2, key="W", omega=False)
        assert estimate - truth == pytest.approx(expected, abs=6e-7), delta
        assert abs(expected) > 1e-3


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_n9_dropping_the_dropout_weights_leaves_the_estimate_and_moves_the_curve(
    law: laws.RepeatedLaw,
) -> None:
    """Single robustness in ``omega``: the saturated estimate stays, and the curve moves."""
    for delta in (0.5, 2.0):
        for horizon in (1, 2):
            base, curve = law.ipsi_population(delta, horizon, key="full")
            moved, moved_curve = law.ipsi_population(delta, horizon, key="full", omega=False)
            assert moved == pytest.approx(base, abs=1e-12)
            assert np.max(np.abs(moved_curve - curve)) >= 0.5, (delta, horizon)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_the_known_mechanisms_are_the_conditionals_of_the_cells(law: laws.RepeatedLaw) -> None:
    """The tables ``G1``, ``C1``, ``G2``, ``C2`` equal the ratios of the cell masses."""
    g1, c1, g2, c2 = law.known_mechanisms
    frame = law.support_frame
    for position in range(len(frame)):
        row = frame.iloc[position]
        w, a1, high = int(row.W), int(row.A1), int(row.high)
        pi_1 = law._mass(law.probs, w=w, a1=1) / law._mass(law.probs, w=w)
        assert g1[position] == pytest.approx(pi_1, abs=1e-14)
        stay_1 = law._mass(law.probs, w=w, a1=a1, c1=1) / law._mass(law.probs, w=w, a1=a1)
        assert c1[position] == pytest.approx(stay_1, abs=1e-14)
        if row.C1 != 1:
            continue
        pattern = {"w": w, "a1": a1, "c1": 1, "l2": int(row.L2)}
        pattern["y1"] = int(row.y1lev)
        pi_2 = law._mass(law.probs, a2=1, **pattern) / law._mass(law.probs, **pattern)
        assert g2[position] == pytest.approx(pi_2, abs=1e-14)
        assert high == law.high(int(row.y1lev))
        if row.A2 >= 0:
            stay_2 = law._mass(law.probs, c2=1, a2=int(row.A2), **pattern) / law._mass(
                law.probs, a2=int(row.A2), **pattern
            )
            assert c2[position] == pytest.approx(stay_2, abs=1e-14)


# ---------------------------------------------------------------- the X32a law (T = 3)

BINARY = law3.BINARY
RULE = law3.RULE
PLAN_NAMES = ("never", "always", RULE)


def _units(gap: float, sd: float, n: int) -> float:
    return gap / (sd / np.sqrt(n))


def test_the_ipsi_truths_and_influence_curve_spreads() -> None:
    """Section 5.1: the truth and the SD of the efficient influence curve of each tilt."""
    truths = {
        0.5: (0.1756, 0.2253, 0.3085),
        1.0: (0.1594, 0.1999, 0.2712),
        2.0: (0.1452, 0.1735, 0.2348),
    }
    spreads = {
        0.5: (0.432, 0.497, 0.574),
        1.0: (0.381, 0.422, 0.476),
        2.0: (0.374, 0.420, 0.499),
    }
    for delta in truths:
        for horizon, (truth, spread) in enumerate(
            zip(truths[delta], spreads[delta], strict=True), 1
        ):
            composed = BINARY.tilt_counterfactual(delta, horizon)
            assert BINARY.tilt_functional(delta, horizon) == pytest.approx(composed, abs=1e-14)
            assert BINARY.tilt_functional(delta, horizon, mechanisms="true") == pytest.approx(
                composed, abs=1e-14
            )
            assert BINARY.ipsi_limit(delta, horizon) == pytest.approx(composed, abs=1e-14)
            assert composed == pytest.approx(truth, abs=5e-5)
            assert BINARY.ipsi_eif_sd(delta, horizon) == pytest.approx(spread, abs=5e-4)
            curve = BINARY.ipsi_eif(delta, horizon)
            assert abs(float(np.sum(BINARY.paths["p"].to_numpy() * curve))) < 1e-14


def test_the_mechanism_term_carries_one_to_three_percent_of_the_variance() -> None:
    """Dropping it moves the curve: the share is small and nonzero at the last node."""
    shares = BINARY.ipsi_mechanism_share(0.5, 3)
    assert shares == pytest.approx([0.0024, 0.0010, 0.0266], abs=5e-5)
    assert all(
        0.0 < share < 0.04
        for delta in (0.5, 1.0, 2.0)
        for share in BINARY.ipsi_mechanism_share(delta, 3)
    )


def test_the_tilt_contrasts_are_wide_in_standard_errors() -> None:
    """Section 5.1: 5.26 to 6.67 EIF-SE at ``n = 2,000`` against the natural course."""
    expected = {
        (1, 2.0): -5.26,
        (1, 0.5): 5.28,
        (2, 2.0): -6.42,
        (2, 0.5): 5.74,
        (3, 2.0): -6.51,
        (3, 0.5): 6.26,
    }
    for (horizon, delta), value in expected.items():
        gap = BINARY.tilt_functional(delta, horizon) - BINARY.tilt_functional(1.0, horizon)
        sd = BINARY.ipsi_eif_sd(delta, horizon, reference=1.0)
        assert _units(gap, sd, 2000) == pytest.approx(value, abs=6e-3), (horizon, delta)


def test_the_horizons_of_one_tilt_are_correlated() -> None:
    curves = {h: BINARY.ipsi_eif(2.0, h) for h in (1, 2, 3)}
    weights = BINARY.paths["p"].to_numpy()

    def correlation(first: int, second: int) -> float:
        a, b = curves[first], curves[second]
        return float(
            np.sum(weights * a * b) / np.sqrt(np.sum(weights * a * a) * np.sum(weights * b * b))
        )

    assert correlation(1, 2) == pytest.approx(0.0069, abs=5e-5)
    assert correlation(1, 3) == pytest.approx(0.2489, abs=5e-5)
    assert correlation(2, 3) == pytest.approx(0.0013, abs=5e-5)


def test_l2_ipsi_one_step_necessity_limits() -> None:
    """The ``one_step_necessity`` control: the tilt plug-in with ``W``-only ``m``."""
    expected = {
        (2, 0.5): (-0.47, -1.22, 0.02),
        (2, 2.0): (0.41, 0.88, 0.02),
        (3, 0.5): (-2.23, -4.40, -0.23),
        (3, 2.0): (1.94, 4.29, -0.23),
    }
    for (horizon, delta), (mean, contrast, natural) in expected.items():
        truth = BINARY.tilt_functional(delta, horizon)
        plug = BINARY.ipsi_plug_in_limit(delta, horizon)
        natural_plug = BINARY.ipsi_plug_in_limit(1.0, horizon)
        natural_truth = BINARY.tilt_functional(1.0, horizon)
        sd = BINARY.ipsi_eif_sd(delta, horizon)
        sd_contrast = BINARY.ipsi_eif_sd(delta, horizon, reference=1.0)
        assert _units(plug - truth, sd, 2000) == pytest.approx(mean, abs=6e-3)
        gap = (plug - natural_plug) - (truth - natural_truth)
        assert _units(gap, sd_contrast, 2000) == pytest.approx(contrast, abs=6e-3)
        natural_sd = BINARY.ipsi_eif_sd(1.0, horizon)
        assert _units(natural_plug - natural_truth, natural_sd, 2000) == pytest.approx(
            natural, abs=6e-3
        )


def test_l2_ipsi_dropout_ignored_limits() -> None:
    """``omega := 1``: single robustness in ``omega`` with a consistent ``m`` and not without."""
    expected = {
        (0.5, 2): -0.03,
        (0.5, 3): -0.48,
        (1.0, 2): 0.02,
        (1.0, 3): -0.23,
        (2.0, 2): 0.06,
        (2.0, 3): 0.08,
    }
    for (delta, horizon), value in expected.items():
        truth = BINARY.tilt_functional(delta, horizon)
        sd = BINARY.ipsi_eif_sd(delta, horizon)
        w_only = BINARY.ipsi_limit(delta, horizon, m_kind="baseline", omega_one=True)
        assert _units(w_only - truth, sd, 2000) == pytest.approx(value, abs=6e-3)
        full = BINARY.ipsi_limit(delta, horizon, m_kind="full", omega_one=True)
        assert full == pytest.approx(truth, abs=1e-12)
        w_correct = BINARY.ipsi_limit(delta, horizon, m_kind="baseline")
        assert w_correct == pytest.approx(truth, abs=1e-12)


def test_e7_the_ipsi_one_step_is_not_doubly_robust() -> None:
    """A correct ``m`` with ``W``-only or Markov ``pi`` and ``omega`` leaves a bias."""
    truth = BINARY.tilt_functional(0.5, 3)
    assert BINARY.ipsi_limit(0.5, 3, pi_kind="baseline") - truth == pytest.approx(0.00275, abs=5e-6)
    assert BINARY.ipsi_limit(0.5, 3, pi_kind="markov") - truth == pytest.approx(0.00252, abs=5e-6)
    truth = BINARY.tilt_functional(2.0, 3)
    assert BINARY.ipsi_limit(2.0, 3, pi_kind="baseline") - truth == pytest.approx(
        -0.00228, abs=5e-6
    )
    # The control: a correct pi with any m is exact, so the bias comes from pi and not from m.
    assert BINARY.ipsi_limit(0.5, 3, m_kind="baseline") == pytest.approx(
        BINARY.tilt_functional(0.5, 3), abs=1e-12
    )


def test_l2_ipsi_absorbing_and_history_free_gaps() -> None:
    """N1: 22.8 to 36.1 EIF-SE at ``n = 8,000``.  N2: -1.79 and +0.88 at ``h = 3``."""
    expected = {
        (0.5, 2): 22.8,
        (0.5, 3): 30.2,
        (1.0, 2): 25.9,
        (1.0, 3): 36.1,
        (2.0, 2): 25.2,
        (2.0, 3): 33.5,
    }
    for (delta, horizon), value in expected.items():
        gap = BINARY.tilt_functional(delta, horizon, absorbing=True) - BINARY.tilt_functional(
            delta, horizon
        )
        assert _units(gap, BINARY.ipsi_eif_sd(delta, horizon), 8000) == pytest.approx(
            value, abs=0.06
        )
    for delta, value in ((0.5, -1.79), (2.0, 0.88)):
        gap = BINARY.tilt_functional(delta, 3, history="free") - BINARY.tilt_functional(delta, 3)
        assert _units(gap, BINARY.ipsi_eif_sd(delta, 3), 8000) == pytest.approx(value, abs=6e-3)


def _sdr_row(limit: Any, n: int) -> list[float]:
    """The gap of a limit in EIF-SE at ``n`` for the five gated labels, in report order."""
    value = {plan: limit(plan) for plan in PLAN_NAMES}
    truth = {plan: BINARY.counterfactual(plan, 3) for plan in PLAN_NAMES}
    row = [_units(value[p] - truth[p], BINARY.eif_sd(p, 3), n) for p in PLAN_NAMES]
    for plan in ("always", RULE):
        gap = (value[plan] - value["never"]) - (truth[plan] - truth["never"])
        row.append(_units(gap, BINARY.eif_sd(plan, 3, reference="never"), n))
    return row


def test_l2_sdr_control_limits_at_the_third_horizon() -> None:
    """C2 ``both_wrong``, ``correction_necessity``, N3 and the exact positive limits."""
    every = lambda kind: (kind,) * 3  # noqa: E731
    both_wrong = _sdr_row(
        lambda p: BINARY.sdr_limit(p, 3, every("baseline"), every("intercept")), 2000
    )
    assert both_wrong == pytest.approx([-1.82, 1.03, -0.76, 2.07, 0.82], abs=6e-3)
    plug_in = _sdr_row(lambda p: BINARY.sdr_plug_in_limit(p, 3, "baseline"), 2000)
    assert plug_in == pytest.approx([-1.82, 1.03, -0.79, 2.07, 0.80], abs=6e-3)
    markov = lambda p: BINARY.sdr_limit(p, 3, every("markov"), every("intercept"))  # noqa: E731
    assert _sdr_row(markov, 2000) == pytest.approx([-1.61, 0.97, -1.07, 1.86, 0.46], abs=6e-3)
    assert _sdr_row(markov, 8000)[4] == pytest.approx(0.91, abs=6e-3)
    for q, g in (
        (every("full"), every("true")),
        (every("full"), every("intercept")),
        (every("baseline"), every("true")),
    ):
        assert _sdr_row(lambda p, q=q, g=g: BINARY.sdr_limit(p, 3, q, g), 2000) == pytest.approx(
            [0.0] * 5, abs=1e-9
        )


def test_l2_sdr_rule_necessity_and_history_free_gaps() -> None:
    """C7 against the reversed rule, and N2 against the history-free g-formula."""
    for horizon, value in ((2, -3.58), (3, -2.35)):
        gap = BINARY.counterfactual("reversed", horizon) - BINARY.counterfactual(RULE, horizon)
        assert _units(gap, BINARY.eif_sd("reversed", horizon), 2000) == pytest.approx(
            value, abs=6e-3
        )
    free = {plan: BINARY.g_formula(plan, 3, history="free") for plan in ("never", "always")}
    truth = {plan: BINARY.counterfactual(plan, 3) for plan in ("never", "always")}
    assert _units(free["never"] - truth["never"], BINARY.eif_sd("never", 3), 2000) == pytest.approx(
        -1.42, abs=6e-3
    )
    assert _units(
        free["always"] - truth["always"], BINARY.eif_sd("always", 3), 2000
    ) == pytest.approx(0.73, abs=6e-3)
    contrast = (free["always"] - free["never"]) - (truth["always"] - truth["never"])
    sd = BINARY.eif_sd("always", 3, reference="never")
    assert _units(contrast, sd, 2000) == pytest.approx(1.56, abs=6e-3)


def test_l2_sdr_absorbing_gaps() -> None:
    """N1: +11.9 to +16.2 EIF-SE at ``n = 8,000`` for the six labels of the family."""
    values = [
        _units(
            BINARY.g_formula(plan, horizon, absorbing=True) - BINARY.counterfactual(plan, horizon),
            BINARY.eif_sd(plan, horizon),
            8000,
        )
        for horizon in (2, 3)
        for plan in PLAN_NAMES
    ]
    assert min(values) == pytest.approx(11.9, abs=0.06)
    assert max(values) == pytest.approx(16.2, abs=0.06)


def test_l3_the_primary_plug_in_gap_of_the_declared_pairing_key() -> None:
    """11 of 13 cells miss by more than 0.25 EIF-SE at ``n = 12,000``; the rule's means do not."""
    expected = {
        "never_t1": -0.57,
        "always_t1": 0.52,
        "always_vs_never_t1": 0.76,
        "never_t2": -1.24,
        "always_t2": 1.00,
        "rule_t2": -0.15,
        "always_vs_never_t2": 1.60,
        "rule_vs_never_t2": 0.87,
        "never_t3": -1.61,
        "always_t3": 1.21,
        "rule_t3": -0.24,
        "always_vs_never_t3": 2.02,
        "rule_vs_never_t3": 1.02,
    }
    observed: dict[str, float] = {}
    for horizon in (1, 2, 3):
        plug = {p: BINARY.sdr_plug_in_limit(p, horizon, "y_only") for p in PLAN_NAMES}
        truth = {p: BINARY.counterfactual(p, horizon) for p in PLAN_NAMES}
        for plan in PLAN_NAMES:
            if plan == RULE and horizon == 1:
                continue
            name = f"{'rule' if plan == RULE else plan}_t{horizon}"
            observed[name] = _units(plug[plan] - truth[plan], BINARY.eif_sd(plan, horizon), 12000)
        for plan in ("always", RULE):
            if plan == RULE and horizon == 1:
                continue
            gap = (plug[plan] - plug["never"]) - (truth[plan] - truth["never"])
            sd = BINARY.eif_sd(plan, horizon, reference="never")
            name = f"{'rule' if plan == RULE else plan}_vs_never_t{horizon}"
            observed[name] = _units(gap, sd, 12000)
    assert observed == pytest.approx(expected, abs=6e-3)
    necessity = [name for name, value in observed.items() if abs(value) > 0.25]
    assert len(necessity) == 11
    assert {name for name in observed if name not in necessity} == {"rule_t2", "rule_t3"}
