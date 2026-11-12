"""The design numbers of ``small-sample-wald``, recomputed from the study and law modules.

Fix-3 plan rev 4, sections 5.2.2 and 5.4: every number the planning scripts printed
(``rows/Fix-3/exploratory/heldout_design.out``, ``rev3_sizing.out``, ``rev4_sizing.out``,
``empty_cell_prob.out`` and review 3's ``design_checks.py`` and ``rev4_checks.py``) is
recomputed here from :mod:`tests.studies.small_sample_laws` and
:mod:`tests.studies.small_sample_wald`.  Where rev 4 supersedes rev 3 (H2's empty design cells)
both are recomputed, under their own withheld shares.  The two numbers the planning scripts
read off simulation parquets that were not kept (the median fitted df of part 1 and the
leverage smoke) are not recomputed.

Each expected value is the printed one, compared at half a unit of its last printed digit.
No estimator is fitted on a held-out law: the fit plumbing tests below use frames drawn from
L1, which the owner studies already fit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.exceptions import DataError
from cleverly.inference import Transform, wald_ci
from tests.studies import small_sample_laws as laws
from tests.studies import small_sample_wald as study
from tests.studies import small_sample_wald_properties as props
from tests.studies import stratified_alternating_law as alternating
from tests.studies import stratified_law as l1


def printed(value: float, expected: float, digits: int) -> bool:
    """``value`` prints as ``expected`` at ``digits`` decimals, within half a unit."""
    return abs(value - expected) <= 0.5 * 10.0**-digits + 1e-12


SHORT = {"ey[1][V=2]": "ey1", "ey[0][V=2]": "ey0", "ate[V=2]": "ate"}


# ------------------------------------------------------------- heldout_design.out, part 1

#: law, estimand, n, kappa, nu_pop as printed.
PART_ONE = (
    ("strat_in", "ey[1][V=2]", 2000, 46.2, 17.7),
    ("strat_in", "ey[0][V=2]", 2000, 8.2, 111.2),
    ("strat_in", "ate[V=2]", 2000, 11.8, 74.0),
    ("l1000", "ey[1][V=2]", 1000, 46.2, 8.8),
    ("l1000", "ey[0][V=2]", 1000, 8.2, 55.6),
    ("l1000", "ate[V=2]", 1000, 11.8, 37.0),
    ("l500", "ey[1][V=2]", 500, 46.2, 4.4),
    ("l500", "ey[0][V=2]", 500, 8.2, 27.8),
    ("l500", "ate[V=2]", 500, 11.8, 18.5),
    ("logitmsm", "msm[a][V=2]", 2000, 27.9, 29.7),
    ("logitmsm", "msm[W][V=2]", 2000, 18.4, 45.9),
    ("logitmsm", "msm[(intercept)][V=2]", 2000, 10.9, 81.1),
    ("fresh_rare", "ey[1][V=2]", 2000, 93.4, 6.5),
    ("fresh_rare", "ey[0][V=2]", 2000, 35.1, 17.6),
    ("fresh_rare", "ate[V=2]", 2000, 25.7, 24.3),
    ("fresh_lognormal06", "ey[1][V=2]", 2000, 71.8, 5.7),
    ("fresh_lognormal06", "ate[V=2]", 2000, 48.8, 8.4),
    ("fresh_lognormal", "ey[1][V=2]", 2000, 3631.0, 0.1),
    ("fresh_lognormal", "ate[V=2]", 2000, 1905.7, 0.2),
)


def _part_one(law: str, estimand: str, n: int) -> tuple[float, float]:
    if law in ("strat_in", "l1000", "l500"):
        kappa = laws.l1_pt(2, SHORT[estimand])[1]
        return kappa, laws.nu_pop(kappa, n * float(l1.P_V[2]))
    if law == "logitmsm":
        term = estimand[4:].split("]", 1)[0]
        _, kappa, share, _ = laws.msm_kappa(alternating.PROBS, 2, term)
        return kappa, laws.nu_pop(kappa, n * share)
    if law == "fresh_rare":
        kappa = laws.rare_pt(SHORT[estimand])[1]
        return kappa, laws.nu_pop(kappa, n * 0.15)
    kappa = laws.continuous_pt(law.removeprefix("fresh_"), SHORT[estimand])[1]
    return kappa, laws.nu_pop(kappa, n * 0.1)


@pytest.mark.parametrize(("law", "estimand", "n", "kappa", "nu"), PART_ONE)
def test_part_one_population_df(law: str, estimand: str, n: int, kappa: float, nu: float) -> None:
    got_kappa, got_nu = _part_one(law, estimand, n)
    assert printed(got_kappa, kappa, 1)
    assert printed(got_nu, nu, 1)


def test_part_one_student_has_no_fourth_moment() -> None:
    for estimand in ("ey1", "ate"):
        _, kappa, _ = laws.continuous_pt("student", estimand)
        assert kappa == math.inf
        assert laws.nu_pop(kappa, 200) == 0.0 or math.isinf(kappa)


def test_part_one_constant_cells_and_ladder() -> None:
    assert printed(laws.l1_const(2000, 2, 1), 0.00012, 5)
    assert laws.l1_const(2000, 2, 0) == pytest.approx(4.13e-18, rel=2e-3)
    assert printed(laws.rare_const(2000, 1), 0.01540, 5)
    assert printed(laws.rare_const(2000, 0), 0.00018, 5)
    # empty_cell_prob.out
    for n, expected in ((500, 0.1038), (1000, 0.0108), (2000, 0.0001), (8000, 0.0000)):
        assert printed(laws.l1_const(n, 2, 1), expected, 4)
    assert printed(laws.rare_const(2000, 1), 0.0154, 4)


def test_rare_truths_monte_carlo_against_quadrature() -> None:
    monte_carlo = laws.diagnosis_truths("rare")
    quadrature = laws.rare_quadrature_truths()
    for name, value in quadrature.items():
        # About two Monte Carlo SEs at 4,000,000 draws.
        assert monte_carlo[name] == pytest.approx(value, abs=6e-5)


# ------------------------------------------------------------- heldout_design.out, part 2


def test_h1_design() -> None:
    pw, g, _, _ = laws.h1_cells()
    n = 2000
    assert printed(float(np.sum(pw * g)), 0.0423, 4)
    assert printed(n * float(np.sum(pw * g)), 84.6, 1)
    assert printed(float(g.min()), 0.0323, 4)
    expected = {
        "ey1": (0.891320, 202.6, 19.8, 0.0356, 3.05, (0.0, 1.0)),
        "ey0": (0.439144, 1.1, 38265.7, 0.0113, 38.74, (0.0, 1.0)),
        "ate": (0.452176, 166.5, 24.2, 0.0373, 14.67, (-1.0, 1.0)),
    }
    for estimand, (truth, kappa, nu, se, distance, (low, high)) in expected.items():
        psi, got_kappa, sd = laws.h1_pt(estimand)
        got_se = sd / math.sqrt(n)
        assert printed(psi, truth, 6)
        assert printed(got_kappa, kappa, 1)
        assert printed(laws.nu_pop(got_kappa, n), nu, 1)
        assert printed(got_se, se, 4)
        assert printed(min(psi - low, high - psi) / got_se, distance, 2)
    treated, untreated = laws.h1_const(n)
    assert treated == pytest.approx(1.92e-4, rel=3e-3)
    assert untreated < 5e-3 * 1e-2
    assert laws.either(treated, untreated) == pytest.approx(1.92e-4, rel=3e-3)
    assert laws.h1_truths() == pytest.approx(
        {"ey[1]": 0.891320, "ey[0]": 0.439144, "ate": 0.452176}, abs=5e-7
    )


def test_h2_design() -> None:
    n = 2000
    _, _, arms = laws.h2_cells()
    assert np.allclose(np.round(1 - arms[1][2], 3), [0.214, 0.269, 0.332])
    assert np.allclose(np.round(1 - arms[0][2], 3), [0.31, 0.378, 0.45])
    expected = {"ey1": (3.651050, 17.1, 14.9), "ey0": (2.055211, 18.8, 13.5)}
    expected["ate"] = (1.595838, 10.8, 24.4)
    for estimand, (truth, kappa, nu) in expected.items():
        psi, got_kappa, _ = laws.h2_pt(estimand)
        assert printed(psi, truth, 6)
        assert printed(got_kappa, kappa, 1)
        assert printed(laws.nu_pop(got_kappa, n * float(laws.H2_PV[2])), nu, 1)
    m2, m3, m4 = arms[1][1]
    assert np.allclose(np.round(m3 / m2**1.5, 2), [1.63, 1.68, 1.77])
    assert np.allclose(np.round(m4 / m2**2, 2), [6.83, 6.98, 7.27])
    assert laws.h2_const(n, 1) == pytest.approx(4.38e-16, rel=3e-3)
    assert laws.h2_truths() == pytest.approx(
        {"ey[1][V=2]": 3.651050, "ey[0][V=2]": 2.055211, "ate[V=2]": 1.595838}, abs=5e-7
    )


def test_h2_empty_design_cells() -> None:
    """Review 3's ``design_checks.py``: P(some stratum-2 cell is empty) and its count."""
    cells = laws.h2_cell_probabilities()
    assert len(cells) == 6
    assert printed(2000 * cells[(1, 0)], 6.5, 1)
    assert (1 - cells[(1, 0)]) ** 2000 == pytest.approx(0.00156, abs=5e-6)
    assert printed(laws.h2_empty_cell(2000), 0.0016, 4)
    assert printed(2000 * laws.h2_empty_cell(2000), 3.1, 1)


def test_h3_design() -> None:
    n = 2000
    probs = laws.h3_probs()
    assert printed(float(probs.sum()), 1.0, 6)
    expected = {
        "(intercept)": (0.6, 52.2, 11.7, 14.384),
        "a": (0.9, 18.9, 33.6, 14.953),
        "W": (-0.4, 70.4, 8.6, 10.101),
    }
    for term, (truth, kappa, nu, sd) in expected.items():
        psi, got_kappa, share, got_sd = laws.msm_kappa(probs, 2, term)
        assert printed(psi, truth, 6)
        assert printed(got_kappa, kappa, 1)
        assert printed(laws.nu_pop(got_kappa, n * share), nu, 1)
        assert printed(got_sd, sd, 3)
    assert laws.h3_truths() == pytest.approx(
        {"msm[(intercept)][V=2]": 0.6, "msm[a][V=2]": 0.9, "msm[W][V=2]": -0.4}, abs=1e-9
    )
    support = np.asarray(alternating.SUPPORT)
    for arm, share, constant in ((1, 0.0307, 1.09e-08), (0, 0.1193, 2.62e-51)):
        inside = (support[:, 0] == 2) & (support[:, 2] == arm)
        assert printed(float(probs[inside].sum()), share, 4)
        assert laws.l1_const(n, 2, arm, probs) == pytest.approx(constant, rel=5e-3)
    w = np.arange(3.0)
    assert np.allclose(np.round(laws.h3_q(1, w, 2), 3), [0.818, 0.75, 0.668])
    assert np.allclose(np.round(laws.h3_q(0, w, 2), 3), [0.646, 0.55, 0.45])
    assert np.allclose(np.round(laws.h3_g(w, 2), 3), [0.1, 0.168, 0.269])


def test_declared_withheld_shares_match_the_laws() -> None:
    for cell, declared in study.DECLARED_WITHHELD.items():
        group = study.GROUP_OF_CELL[cell]
        estimand = study.CELL_BY_NAME[cell].estimand
        share = laws.constant_cell_share(group.law, group.n, estimand)
        if group.law == "h2":
            share = laws.either(share, laws.h2_empty_cell(group.n))
        if math.isnan(share):
            # The continuous and identity-MSM laws have no constant cell.
            assert declared == 0.0
            continue
        assert share == pytest.approx(declared, rel=0.06, abs=1e-6), cell
    assert set(study.DECLARED_WITHHELD) == set(study.GATED_CELLS)
    assert len(study.GATED_CELLS) == 14
    # rare_ate inherits arm 1's constant cell, which is why it is reported (plan 5.2.1).  The
    # sizing script entered 0.0157, a rounding up of this share, to price it.
    share = laws.constant_cell_share("rare", 2000, "ate[V=2]")
    assert printed(share, 0.0156, 4)
    assert share <= 0.0157


# ------------------------------------------------------------- rev3 and rev4 sizing

#: rev4_sizing.out: per-cell pass probability at calibration, and at the predicted reading.
PER_CELL = {
    "boundary_mean": (0.99808, 0.99988),
    "boundary_mean_n8000": (0.99822, 0.99822),
    "logit_msm_v2_a": (0.99808, 0.99982),
    "identity_msm_v2_w": (0.99822, 0.99988),
    "rare_ey0": (0.99800, 0.99977),
    "student_ey1": (0.99822, 0.99986),
    "student_ate": (0.99822, 0.99366),
    "lognormal06_ey1": (0.99822, 0.94266),
    "ho_sparse_treat_ey1": (0.99799, 0.99799),
    "ho_sparse_treat_ate": (0.99799, 0.99799),
    "ho_hurdle_ey1": (0.99524, 0.99524),
    "ho_hurdle_ate": (0.99524, 0.99524),
    "ho_logit_msm_a": (0.99822, 0.99822),
    "ho_logit_msm_w": (0.99822, 0.99822),
}

#: Rev 3 sized H2 with no withheld share; rev 4 (review 3, B-1) at 0.0016.
REV3_WITHHELD = {**study.DECLARED_WITHHELD, "ho_hurdle_ey1": 0.0, "ho_hurdle_ate": 0.0}


def test_per_cell_pass_probabilities() -> None:
    for cell, (calibrated, predicted) in PER_CELL.items():
        share = study.DECLARED_WITHHELD[cell]
        assert printed(study.two_leg_pass(0.95, withheld=share), calibrated, 5), cell
        coverage = study.PREDICTED_COVERAGE.get(cell, 0.95)
        assert printed(study.two_leg_pass(coverage, withheld=share), predicted, 5), cell
    # rev3_sizing.out: H2 at no withheld share.
    assert printed(study.two_leg_pass(0.95, withheld=0.0), 0.99822, 5)


def test_cell_products() -> None:
    def product(shares: dict[str, float], predicted: bool) -> float:
        out = 1.0
        for cell, share in shares.items():
            coverage = study.PREDICTED_COVERAGE.get(cell, 0.95) if predicted else 0.95
            out *= study.two_leg_pass(coverage, withheld=share)
        return out

    assert printed(product(study.DECLARED_WITHHELD, False), 0.9686, 4)
    assert printed(product(study.DECLARED_WITHHELD, True), 0.9184, 4)
    assert printed(product(REV3_WITHHELD, False), 0.9745, 4)
    assert printed(product(REV3_WITHHELD, True), 0.9239, 4)


def test_family_pass_probabilities() -> None:
    assert printed(study.family_pass(predicted=False), 0.9523, 4)
    predicted = study.family_pass(predicted=True)
    assert printed(predicted, 0.9061, 4)
    assert printed(1.0 - predicted, 0.0939, 4)
    # rev3_sizing.out's FINAL line, before review 3's empty design cells.
    assert printed(study.family_pass(predicted=False, withheld=REV3_WITHHELD), 0.9580, 4)
    assert printed(study.family_pass(predicted=True, withheld=REV3_WITHHELD), 0.9116, 4)


def test_reported_and_rejected_designs() -> None:
    assert printed(study.two_leg_pass(0.9556, withheld=0.0157), 0.8440, 4)
    assert printed(study.two_leg_pass(0.970, withheld=1.2e-4), 0.6211, 4)
    assert printed(study.two_leg_pass(0.9533, withheld=1.2e-4), 0.9998, 4)
    assert printed(study.lower_leg_pass(0.95), 0.9982, 4)
    assert printed(study.two_leg_pass(0.95, replicates=1600), 0.98872, 5)
    assert printed(study.two_leg_pass(0.95, replicates=2400), 0.99973, 5)
    assert printed(1.0 - study.two_leg_pass(0.9435), 0.0573, 4)


def test_audit_bounds() -> None:
    """rev4_sizing.out, A6 and A10: the t's coverage gain and the unseen low-df share."""
    for nu, gain in ((5, 0.0398), (10, 0.0241), (30, 0.0089), (100, 0.0027), (250, 0.0011)):
        assert printed(study.coverage_gain(nu), gain, 4)
    bounds = {20: (0.233, 0.0104, 0.0120), 50: (0.101, 0.0051, 0.0067)}
    bounds[100] = (0.052, 0.0032, 0.0048)
    for refits, (share, at_250, at_100) in bounds.items():
        unseen = study.unseen_share(refits)
        assert printed(unseen, share, 3)
        assert printed(study.coverage_gain(250) + unseen * study.coverage_gain(5), at_250, 4)
        assert printed(study.coverage_gain(100) + unseen * study.coverage_gain(5), at_100, 4)
    z = stats.norm.ppf(0.975)
    for nu, value in ((250, 1.216), (50, 1.24), (20, 1.286)):
        assert printed((stats.t.ppf(0.975, nu) / z - 1) * nu, value, 3)


def test_paired_gain_power() -> None:
    """Review 3's ``rev4_checks.py``: the exact power and the family at that power."""
    power = study.paired_gain_power()
    assert printed(power, 0.9896, 4)
    assert printed(study.family_pass(predicted=False, gain_power=power), 0.9548, 4)
    assert study.gain_passes(8, 0) and not study.gain_passes(7, 0)
    assert stats.poisson.sf(7, 21) == pytest.approx(study.STUDENT_GAIN_POWER, abs=5e-5)


# ------------------------------------------------------------- the rule and the triggers


@pytest.mark.parametrize(
    ("psi", "se", "bounds", "family", "nu"),
    [
        (0.93, 0.03, (0.0, 1.0), "binomial", 12.0),
        (0.45, 0.04, (-1.0, 1.0), "binomial", 30.0),
        (0.9, 0.2, None, "binomial", 8.0),
        (3.6, 0.5, None, "gaussian", 14.9),
        (3.6, 0.5, None, "gaussian", 2.1),
        (3.6, 0.5, None, "gaussian", None),
    ],
)
def test_rule_interval_matches_its_vector_twin(
    psi: float, se: float, bounds: tuple[float, float] | None, family: str, nu: float | None
) -> None:
    low, high, scale = study.rule_interval(psi, se, bounds=bounds, outcome_family=family, nu=nu)
    df = math.inf if family != "gaussian" or nu is None else max(nu, study.DF_CAP)
    a, b = bounds if bounds is not None else (math.nan, math.nan)
    v_low, v_high = study.vector_interval(
        np.array([psi]),
        np.array([se]),
        low=np.array([a]),
        high=np.array([b]),
        df=np.array([df]),
        logit=bounds is not None,
    )
    assert scale == ("range_logit" if bounds is not None else "identity")
    assert (low, high) == pytest.approx((v_low[0], v_high[0]), rel=1e-12)
    if bounds is None:
        assert (low, high) == pytest.approx(
            wald_ci(psi, se, study.ALPHA, df=None if df == math.inf else df), rel=1e-12
        )
    else:
        transform = Transform.range_logit(*bounds)
        centre = transform.forward(psi)
        half = stats.norm.ppf(0.975) * se * transform.slope(psi)
        assert low == pytest.approx(transform.inverse(centre - half), rel=1e-12)


def test_rule_interval_withholds_on_or_outside_bounds() -> None:
    for psi in (1.0, 1.2, 0.0):
        low, high, _ = study.rule_interval(
            psi, 0.1, bounds=(0.0, 1.0), outcome_family="binomial", nu=10.0
        )
        assert math.isnan(low) and math.isnan(high)


def test_rule_vectors_read_the_rows() -> None:
    frame = pd.DataFrame(
        {
            "estimate": [0.93, 3.6, 0.9],
            "std_error": [0.03, 0.5, 0.2],
            "truth": [0.95, 2.0, 1.0],
            "nu": [12.0, 3.0, 8.0],
            "kish": [20.0, 4.0, 9.0],
            "bound_low": [0.0, math.nan, math.nan],
            "bound_high": [1.0, math.nan, math.nan],
            "outcome_family": ["binomial", "gaussian", "binomial"],
            "withheld": ["", "", "constant_cell"],
        }
    )
    low, high = props.rule_vectors(frame, alpha=study.ALPHA)
    for index, row in frame.iterrows():
        bounds = None if math.isnan(row["bound_low"]) else (row["bound_low"], row["bound_high"])
        expected = study.rule_interval(
            row["estimate"],
            row["std_error"],
            bounds=bounds,
            outcome_family=row["outcome_family"],
            nu=row["nu"],
        )
        assert (low[index], high[index]) == pytest.approx(expected[:2], rel=1e-12)
    # The capped t floors the df at 5; a withheld interval never covers.
    kish_low, _ = props.candidate_interval(frame, "kish_t")
    assert kish_low[1] == pytest.approx(3.6 - stats.t.ppf(0.975, 5.0) * 0.5, rel=1e-12)
    assert props._covers(frame, low, high).tolist() == [True, False, False]


def test_withheld_trigger() -> None:
    class Stub:
        def __init__(self, psi: float, bounds: tuple[float, float] | None) -> None:
            self.psi, self.bounds, self.transform = psi, bounds, None
            self.scale, self.n, self.n_clusters = "level", 4, 4

    frame = pd.DataFrame(
        {"V": [2, 2, 2, 2, 0, 0], "A": [1, 1, 0, 0, 1, 1], "Y": [1, 1, 0, 1, 0, 1]}
    )
    # Arm 1 of stratum 2 holds only ones: ey[1][V=2] and ate[V=2] read it, ey[0][V=2] does not.
    assert study.withheld_trigger(frame, "ey[1][V=2]", Stub(1.0, (0.0, 1.0))) == "constant_cell"
    assert study.withheld_trigger(frame, "ate[V=2]", Stub(0.5, (-1.0, 1.0))) == "constant_cell"
    assert study.withheld_trigger(frame, "ey[0][V=2]", Stub(0.5, (0.0, 1.0))) == ""
    assert study.withheld_trigger(frame, "ey[0][V=2]", Stub(1.0, (0.0, 1.0))) == "out_of_range"
    observed = frame.assign(Delta=[1, 0, 1, 1, 1, 1], Y=[1, 0, 0, 1, 0, 1])
    assert study.withheld_trigger(observed, "ey[1][V=2]", Stub(0.5, (0.0, 1.0))) == (
        "constant_cell"
    )
    assert study.arms_read("ate[low vs high]") == ("low", "high")
    assert study.arms_read("ey[high]") == ("high",)
    assert study.arms_read("msm[a][V=2]") == (0, 1)


def test_cells_and_controls_are_declared() -> None:
    names = [cell.name for group in study.GROUPS for cell in group.cells]
    assert len(names) == len(set(names))
    for family, cells in study.PROPERTY_CELLS.items():
        assert cells, family
    assert set(study.PROPERTY_CELLS[study.CONTROL]) == {
        f"{cell}__{control}"
        for cell in study.GATED_CELLS
        for control in study.controls_of(study.GROUP_OF_CELL[cell], study.CELL_BY_NAME[cell])
    }
    assert set(study.PAIRED_GAIN_CELLS) <= set(study.GATED_CELLS)
    assert study.WITNESS_SOURCE in study.CELL_BY_NAME
    held_out = {cell for cell in study.GATED_CELLS if cell.startswith("ho_")}
    assert {study.GROUP_OF_CELL[cell].law for cell in held_out} == set(study.HELDOUT_LAWS)
    assert {falsifier[2] for falsifier in props.FALSIFIERS} <= held_out


def test_every_cell_has_a_truth() -> None:
    for group in study.GROUPS:
        truth = study.truths(group.law)
        for cell in group.cells:
            assert math.isfinite(float(truth[cell.estimand])), cell.name


# ------------------------------------------------------------- the H2 outcome model


def test_cell_means_refuses_an_empty_cell() -> None:
    design = np.array([[a, w, v] for a in (0, 1) for w in range(3) for v in range(3)] * 2)
    target = np.arange(len(design), dtype=float)
    model = laws.CellMeans().fit(design, target)
    assert model.predict(design[:18]) == pytest.approx((target[:18] + target[18:]) / 2)
    with pytest.raises(DataError, match=laws.EMPTY_DESIGN_CELL):
        laws.CellMeans().fit(design[1:18], target[1:18])
    weights = np.ones(len(design))
    weights[[0, 18]] = 0.0
    with pytest.raises(DataError, match=laws.EMPTY_DESIGN_CELL):
        laws.CellMeans().fit(design, target, sample_weight=weights)


def test_held_out_fit_plumbing_on_an_owner_law() -> None:
    """The H1 and H2 fits on an L1 frame, which the owner studies already fit."""
    frame = l1.sample(2000, 20261112)
    plain = laws.h1_fit(frame[["W", "A", "Y"]])
    assert {"ey[1]", "ey[0]", "ate"} <= set(plain.estimates)
    stratified = laws.h2_fit(frame.assign(Y=frame["Y"].astype(float)))
    estimate = stratified["ey[1][V=2]"]
    inside = frame.loc[frame["V"] == 2]
    cells = inside.groupby(["W", "A"])["Y"].mean().unstack()
    weights = inside["W"].value_counts(normalize=True).sort_index()
    plug_in = float((cells[1] * weights).sum())
    assert float(estimate.psi) == pytest.approx(plug_in, abs=0.02)
