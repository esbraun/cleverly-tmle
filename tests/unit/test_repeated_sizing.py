"""The shared sizing rules of the repeated-outcome studies: the allocation and the ``n`` rule.

The rules live in :mod:`tests.studies.repeated_sizing`.  X32a binds them to its margins, and its
own tests (``test_ltmle_repeated_method_study.py``) pin its sizes.  These tests cover what X32b
adds (rev 2.1 of its plan):

* the union-bound allocation (WA): the failure shares sum to 0.05, the pilot sizes are pinned,
  the calibration cap returns 50,000 and marks it, and every discrimination check holds its
  share at its planning size;
* the rules, each against the value the design probes recorded;
* the declared ``n`` rules (WN): each is silent for an exactly unbiased study, and each fires
  under a common bias and under a bias in one cell.

The probes are ``reviews/x32/x32b-rev2-scripts/`` (``n_rule.py``, ``allocation.py``,
``calibration_cap*.py``, ``c6_coverage_rule.py``, ``rev21_rules.py``).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import beta, norm

from tests.studies import canonical_ltmle_repeated as study
from tests.studies import ltmle_repeated_law as law
from tests.studies import ltmle_repeated_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.inference import coverage_for_se_ratio

MARGINS = study.STUDY.margins


# ---------------------------------------------------------------- the studies' label correlations

_IPSI_LABEL = re.compile(r"ipsi_([\d.]+)(?:_vs_([\d.]+))?_t(\d)$")


def ipsi_label(delta: float, horizon: int, reference: float | None = None) -> str:
    """The label of the tilt ``delta`` at ``horizon``, or of its contrast with ``reference``."""
    if reference is None:
        return f"ipsi_{delta}_t{horizon}"
    return f"ipsi_{delta}_vs_{reference}_t{horizon}"


@functools.cache
def _ipsi_curve(model: str, delta: float, horizon: int, reference: float | None) -> np.ndarray:
    laws = {"binary": law.BINARY, "continuous": law.CONTINUOUS}
    return laws[model].ipsi_eif(delta, horizon, reference=reference)


def ipsi_correlations(model: law.RepeatedOutcomeLaw) -> sizing.Correlations:
    """The correlation of the Kim influence curves of ``ipsi_*`` labels under ``model``."""

    def correlations(labels: Sequence[str]) -> np.ndarray:
        curves = []
        for label in labels:
            match = _IPSI_LABEL.match(label)
            assert match is not None, label
            delta, reference, horizon = match.groups()
            curves.append(
                _ipsi_curve(
                    model.name, float(delta), int(horizon), float(reference) if reference else None
                )
            )
        return sizing.influence_correlations(curves, model.paths["p"].to_numpy())

    return correlations


IPSI = ipsi_correlations(law.BINARY)
IPSI_CONTINUOUS = ipsi_correlations(law.CONTINUOUS)
#: ``SDR`` and ``LTMLE`` share the efficient influence curve, so X32a's correlations serve.


@functools.cache
def _sdr_matrix(labels: tuple[str, ...]) -> np.ndarray:
    return properties.label_correlations(labels)


def SDR(labels: Sequence[str]) -> np.ndarray:
    """X32a's label correlations, cached: each call otherwise rebuilds every influence curve."""
    return _sdr_matrix(tuple(labels))


IPSI_PRIMARY = [ipsi_label(d, h) for d in (0.5, 1.0, 2.0) for h in (1, 2, 3)] + [
    ipsi_label(d, h, 1.0) for d in (0.5, 2.0) for h in (1, 2, 3)
]
IPSI_THIRD = [ipsi_label(d, 3) for d in (0.5, 1.0, 2.0)] + [
    ipsi_label(d, 3, 1.0) for d in (0.5, 2.0)
]
SDR_PRIMARY = [
    "never_t1",
    "never_t2",
    "never_t3",
    "always_t1",
    "always_t2",
    "always_t3",
    "rule_t2",
    "rule_t3",
    "always_vs_never_t1",
    "always_vs_never_t2",
    "always_vs_never_t3",
    "rule_vs_never_t2",
    "rule_vs_never_t3",
]
SDR_THIRD = ["never_t3", "always_t3", "rule_t3", "always_vs_never_t3", "rule_vs_never_t3"]


def _cells(group: str, labels: Sequence[str], *arms: str) -> list[str]:
    """``group/label__arm`` for each label and arm: a positive cell and its control are two arms."""
    return [f"{group}/{label}__{arm}" for label in labels for arm in arms]


# ---------------------------------------------------------------- WN: the n rules


@dataclass(frozen=True)
class Rule:
    """A declared ``n`` rule: the groups of cells it reads and how it decides."""

    name: str
    cells: dict[str, list[str]]
    correlations: dict[str, sizing.Correlations]
    pilot: int
    #: Decides from the pilot groups of every group of the study.
    fires: Callable[[dict[str, sizing.PilotGroup]], bool]
    #: The groups that carry the bias in a power case.
    biased: tuple[str, ...]
    #: The rule without the outlier test (rev 2), the control of the one-cell case.
    pooled_only: Callable[[dict[str, sizing.PilotGroup]], bool] | None = None


def _pooled_only(
    judged: Callable[[dict[str, sizing.PilotGroup]], list[sizing.PilotGroup]],
    others: Callable[[dict[str, sizing.PilotGroup]], list[sizing.PilotGroup]],
    share: float,
) -> Callable[[dict[str, sizing.PilotGroup]], bool]:
    """Rev 2's reading: the pooled probability with no outlier test."""

    def fires(groups: dict[str, sizing.PilotGroup]) -> bool:
        pooled = judged(groups)
        tau = sizing.heterogeneity_bound(pooled + others(groups))
        gate = sizing._bias_gate(MARGINS, sizing.N_RULE_REFERENCE_REPLICATES)
        return sizing.pooled_pass_probability(pooled, tau, gate) < share

    return fires


def _primary_rule(name: str, labels: list[str], correlations: sizing.Correlations) -> Rule:
    """Two independent scenarios, one group each.  The rule reads both."""
    share = sizing.PRIMARY_CLAUSE_SHARE

    def judged(groups: dict[str, sizing.PilotGroup]) -> list[sizing.PilotGroup]:
        return list(groups.values())

    return Rule(
        name,
        {
            "scenario_1": _cells("scenario_1", labels, "cleverly"),
            "scenario_2": _cells("scenario_2", labels, "cleverly"),
        },
        {"scenario_1": correlations, "scenario_2": correlations},
        sizing.PILOT_DRAWS,
        lambda groups: sizing.n_rule_fires(MARGINS, judged(groups), share),
        ("scenario_1", "scenario_2"),
        _pooled_only(judged, lambda groups: [], share),
    )


def _family_rule(
    name: str, families: dict[str, list[str]], focus: str, correlations: sizing.Correlations
) -> Rule:
    """A property family is judged alone, and every other family of the study feeds the spread."""
    share = sizing.family_share(len(families))

    def others(groups: dict[str, sizing.PilotGroup]) -> list[sizing.PilotGroup]:
        return [group for family, group in groups.items() if family != focus]

    return Rule(
        name,
        {
            family: _cells(family, labels, "positive", "control")
            for family, labels in families.items()
        },
        dict.fromkeys(families, correlations),
        sizing.PILOT_DRAWS,
        lambda groups: sizing.n_rule_fires(MARGINS, [groups[focus]], share, extra=others(groups)),
        (focus,),
        _pooled_only(lambda groups: [groups[focus]], others, share),
    )


def _c6_rule() -> Rule:
    """``IPSI`` C6 reads the coverage half at the cap, with the spread of every calibration arm."""
    arms = {"C1": IPSI_PRIMARY, "C3": IPSI_THIRD, "C6": IPSI_THIRD}

    def others(groups: dict[str, sizing.PilotGroup]) -> list[sizing.PilotGroup]:
        return [group for arm, group in groups.items() if arm != "C6"]

    return Rule(
        "ipsi_c6",
        {arm: _cells(arm, labels, "calibration") for arm, labels in arms.items()},
        {"C1": IPSI, "C3": IPSI, "C6": IPSI_CONTINUOUS},
        sizing.CALIBRATION_PILOT_DRAWS,
        lambda groups: sizing.c6_rule_fires(MARGINS, groups["C6"], extra=others(groups)),
        ("C6",),
    )


IPSI_FAMILIES = {
    "single_robustness": [ipsi_label(d, 3) for d in (0.5, 2.0)]
    + [ipsi_label(d, 3, 1.0) for d in (0.5, 2.0)],
    "repeated_outcome_necessity": [ipsi_label(d, h) for d in (0.5, 1.0, 2.0) for h in (2, 3)],
    "outcome_history_necessity": [ipsi_label(d, 3) for d in (0.5, 2.0)],
}
SDR_FAMILIES = {
    "double_robustness": SDR_THIRD,
    "correction_necessity": SDR_THIRD,
    "rule_necessity": ["rule_t2", "rule_t3"],
    "repeated_outcome_necessity": [
        f"{p}_t{h}" for p in ("never", "always", "rule") for h in (2, 3)
    ],
    "outcome_history_necessity": ["never_t3", "always_t3", "always_vs_never_t3"],
    "full_history_necessity": SDR_THIRD,
}

RULES = [
    _primary_rule("ipsi_primary", IPSI_PRIMARY, IPSI),
    _primary_rule("sdr_primary", SDR_PRIMARY, SDR),
    _family_rule("ipsi_n1", IPSI_FAMILIES, "repeated_outcome_necessity", IPSI),
    _family_rule("ipsi_n2", IPSI_FAMILIES, "outcome_history_necessity", IPSI),
    _family_rule("sdr_n1", SDR_FAMILIES, "repeated_outcome_necessity", SDR),
]
C6 = _c6_rule()
BIAS_RULES = RULES  # the rules keyed to the 0.25 bias margin
RULE_IDS = [rule.name for rule in RULES]


def _fire_count(
    rule: Rule,
    decide: Callable[[dict[str, sizing.PilotGroup]], bool],
    *,
    bias: float,
    cell: int | None = None,
    sims: int,
    pilot: int | None = None,
    seed: int = 20261014,
) -> tuple[int, int]:
    """Fires over ``sims`` seeded pilots.

    Every cell of ``rule.biased`` has the true standardized bias ``bias``, or with ``cell`` set
    only that cell of the first biased group has it.  The pilot bias of a cell is its true bias
    plus noise with covariance ``correlation / pilot``, which :func:`sizing.property_bias_groups`
    assigns to the group.
    """
    pilot = rule.pilot if pilot is None else pilot
    rng = np.random.default_rng(seed)
    chol = {}
    for group, names in rule.cells.items():
        corr = rule.correlations[group](sizing.cell_labels(names))
        chol[group] = np.linalg.cholesky(corr + 1e-12 * np.eye(len(corr)))
    fires = 0
    for _ in range(sims):
        groups: dict[str, sizing.PilotGroup] = {}
        for group, names in rule.cells.items():
            truth = np.zeros(len(names))
            if group in rule.biased:
                if cell is None:
                    truth[:] = bias
                elif group == rule.biased[0]:
                    truth[cell] = bias
            draw = truth + chol[group] @ rng.standard_normal(len(names)) / np.sqrt(pilot)
            groups.update(
                sizing.property_bias_groups(
                    dict(zip(names, draw, strict=True)), pilot, rule.correlations[group]
                )
            )
        fires += bool(decide(groups))
    return fires, sims


def _upper_bound(fires: int, sims: int) -> float:
    """The one-sided 95% Clopper-Pearson upper bound of a firing rate."""
    return float(beta.ppf(0.95, fires + 1, sims - fires))


@pytest.mark.parametrize("rule", [*RULES, C6], ids=[*RULE_IDS, C6.name])
def test_wn_each_declared_n_rule_is_silent_at_zero_bias(rule: Rule) -> None:
    """The rate at which an exactly unbiased study doubles ``n`` is at most 0.05 (H1).

    0 fires in 2,000 pilots bound the rate at 1.5e-3, and 2 fires at 3.1e-3.  Rev 1's rule,
    the 80% bound of any standardized bias above 0.10, fired with probability 0.986.
    """
    fires, sims = _fire_count(rule, rule.fires, bias=0.0, sims=2_000)
    assert _upper_bound(fires, sims) <= 0.05


@pytest.mark.parametrize("rule", RULES, ids=RULE_IDS)
def test_wn_each_declared_n_rule_fires_under_a_common_bias_of_0_15(rule: Rule) -> None:
    """At the paired-difference margin the rule fires with probability at least 0.95 (observed
    0.991 to 1.000 in the probes), so a silent rule is not a rule that never fires."""
    fires, sims = _fire_count(rule, rule.fires, bias=0.15, sims=300)
    assert fires / sims >= 0.95


@pytest.mark.parametrize("rule", RULES, ids=RULE_IDS)
def test_wn_each_declared_n_rule_fires_on_one_cell_at_0_20(rule: Rule) -> None:
    """One cell at 0.20 with the rest unbiased fires the rule with probability at least 0.95
    (observed 0.999 to 1.000).  That cell fails its gate at 2,400 replications with P = 0.55."""
    fires, sims = _fire_count(rule, rule.fires, bias=0.20, cell=0, sims=300)
    assert fires / sims >= 0.95


@pytest.mark.parametrize("rule", RULES[:2] + RULES[4:], ids=RULE_IDS[:2] + RULE_IDS[4:])
def test_wn_the_pooled_reading_alone_misses_the_one_cell_case(rule: Rule) -> None:
    """The control for the outlier test: rev 2's pooled rule fires on one cell at 0.20 with
    probability 0.15 to 0.33, so the power case above fails without the test."""
    assert rule.pooled_only is not None
    fires, sims = _fire_count(rule, rule.pooled_only, bias=0.20, cell=0, sims=300)
    assert fires / sims < 0.6


def test_wn_the_c6_rule_fires_under_a_common_bias_of_0_50() -> None:
    """C6 has no bias gate, so the rule reads the coverage half at the cap.

    A bias of 0.50 lowers the Wald coverage to 0.921.  At 50,000 replications the exact interval
    lies inside the band [0.92, 0.98] only from a coverage of 0.923, so no number of
    replications cures it (observed 1.000).  A bias of 0.30 leaves the coverage inside
    the band: the rule stays quiet there, although the bias rule of rev 2 fired on 0.12.
    """
    fires, sims = _fire_count(C6, C6.fires, bias=0.50, sims=300)
    assert fires / sims >= 0.95
    fires, sims = _fire_count(C6, C6.fires, bias=0.30, sims=300)
    assert _upper_bound(fires, sims) <= 0.05
    fires, sims = _fire_count(C6, C6.fires, bias=0.12, sims=300)
    assert _upper_bound(fires, sims) <= 0.05


def test_wn_the_c6_coverage_half_passes_at_the_cap_up_to_a_bias_near_0_48() -> None:
    """The coverage half at 50,000 replications passes with P = 0.5 at a bias of 0.481."""
    half = sizing.coverage_half_probability(
        MARGINS, sizing.CALIBRATION_CAP, np.array([0.0, 0.4, 0.481, 0.55])
    )
    assert half[0] == pytest.approx(1.0, abs=1e-9)
    assert half[1] > 0.9999
    assert half[2] == pytest.approx(0.5, abs=0.01)
    assert half[3] < 1e-3
    wald = sizing.wald_coverage(MARGINS, np.ones(2), np.array([0.481, 0.0]))
    assert wald[0] == pytest.approx(0.9231, abs=5e-4)
    assert wald[1] == pytest.approx(0.95, abs=5e-4)


def test_wn_the_statistic_is_the_robust_probability_at_the_reference_size() -> None:
    """``n_rule_statistic`` is the robust probability with the 2,400-replication bias cell."""
    corr = SDR(SDR_THIRD)
    group = sizing.PilotGroup(np.full(5, 0.02), corr / 2_000)
    gate = sizing._bias_gate(MARGINS, sizing.N_RULE_REFERENCE_REPLICATES)
    expected, outliers = sizing.robust_probability([group], gate)
    assert outliers == 0
    assert sizing.n_rule_statistic(MARGINS, [group]) == pytest.approx(expected, abs=1e-15)
    assert sizing.N_RULE_REFERENCE_REPLICATES == 2_400
    # With no outlier the robust reading is X32a's pooled reading at the 80% bound.
    tau = sizing.heterogeneity_bound([group])
    assert expected == pytest.approx(sizing.pooled_pass_probability([group], tau, gate), abs=1e-15)
    # The statistic falls as the pilot bias rises, and the rule fires at the share.
    biased = sizing.PilotGroup(np.full(5, 0.15), corr / 2_000)
    assert sizing.n_rule_statistic(MARGINS, [biased]) < sizing.n_rule_statistic(MARGINS, [group])
    assert not sizing.n_rule_fires(MARGINS, [group], sizing.PRIMARY_CLAUSE_SHARE)
    assert sizing.n_rule_fires(MARGINS, [biased], sizing.PRIMARY_CLAUSE_SHARE)
    # ``extra`` feeds the spread only: a wide group elsewhere in the study widens the bound.
    wide = sizing.PilotGroup(np.array([-0.3, 0.3, -0.3, 0.3, 0.0]), corr / 2_000)
    assert sizing.n_rule_statistic(MARGINS, [group], extra=[wide]) < (
        sizing.n_rule_statistic(MARGINS, [group])
    )


def test_wn_the_outlier_test_reads_one_cell_alone() -> None:
    """A noise-free pilot with one cell at 0.20 has one outlier, and a flat pilot has none.

    The outlier leaves the pool and is read alone, so the probability falls to that cell's own
    failure probability (0.55 at 2,400 replications).  The pooled reading of the same pilot
    spreads the cell over the group and keeps the probability above the share.
    """
    corr = SDR(SDR_PRIMARY)
    size = len(corr)
    gate = sizing._bias_gate(MARGINS, sizing.N_RULE_REFERENCE_REPLICATES)
    bias = np.zeros(size)
    bias[0] = 0.20
    judged = [
        sizing.PilotGroup(bias, corr / 2_000),
        sizing.PilotGroup(np.zeros(size), corr / 2_000),
    ]
    probability, outliers = sizing.robust_probability(judged, gate)
    assert outliers == 1
    own_failure = 1.0 - float(gate(np.array([0.20]))[0])
    assert own_failure == pytest.approx(0.55, abs=0.02)
    assert probability < sizing.PRIMARY_CLAUSE_SHARE
    tau = sizing.heterogeneity_bound(judged)
    pooled = sizing.pooled_pass_probability(judged, tau, gate)
    assert pooled > probability
    flat = [sizing.PilotGroup(np.zeros(size), corr / 2_000)] * 2
    assert sizing.robust_probability(flat, gate)[1] == 0
    # Groups of fewer than three cells are not tested.
    pair = sizing.PilotGroup(np.array([0.0, 0.3]), SDR(["rule_t2", "rule_t3"]) / 2_000)
    assert sizing.robust_probability([pair], gate)[1] == 0
    # The same pilot with every cell at 0.20 is a common bias, not an outlier.
    common = [sizing.PilotGroup(np.full(size, 0.20), corr / 2_000)]
    assert sizing.robust_probability(common, gate)[1] == 0


# ---------------------------------------------------------------- WA: the allocation


def test_wa_the_failure_shares_sum_to_the_study_failure_probability() -> None:
    """0.001 for discrimination and ``d = 0.049 / 3`` for each of three sized instruments make
    0.05, by the union bound, which holds under any dependence."""
    d = sizing.SIZED_FAIL
    assert sizing.DISCRIMINATION_FAIL == 0.001
    assert d == pytest.approx(0.016333, abs=5e-7)
    assert sizing.DISCRIMINATION_FAIL + 3 * d == pytest.approx(sizing.STUDY_FAIL, abs=1e-15)
    # The primary splits d between its two clauses.
    assert pytest.approx(0.991833, abs=5e-7) == sizing.PRIMARY_CLAUSE_SHARE
    assert pytest.approx(d, abs=1e-15) == 2 * (1.0 - sizing.PRIMARY_CLAUSE_SHARE)
    # The property families split d among themselves, and the calibration terms split it.
    assert sizing.family_share(6) == pytest.approx(0.997278, abs=5e-7)
    assert sizing.family_share(3) == pytest.approx(0.994556, abs=5e-7)
    for families in (3, 6):
        assert families * (1.0 - sizing.family_share(families)) == pytest.approx(d, abs=1e-15)
    assert pytest.approx(0.995917, abs=5e-7) == sizing.C6_SHARE
    assert (sizing.IPSI_CALIBRATION_TERMS, sizing.SDR_CALIBRATION_TERMS) == (4, 5)
    assert pytest.approx(d, abs=1e-15) == sizing.IPSI_CALIBRATION_TERMS * (1.0 - sizing.C6_SHARE)
    # Rev 1 sized each instrument to 0.95; on X32a's pinned numbers that product is 0.869.
    assert 0.95 * 0.9523 * 0.961 < 0.87
    # The discrimination share is split over k checks.
    assert sizing.discrimination_floor(74) == pytest.approx(1.0 - 0.001 / 74)
    assert sizing.discrimination_floor(55) == pytest.approx(1.0 - 0.001 / 55)


def test_wa_the_pilot_sizes_are_pinned() -> None:
    """The calibration pilots P2 and P5 draw 8,000 per arm and every other pilot 2,000.

    The 8,000-draw pilot reaches the calibration cap less often than the 4,000-draw pilot in
    every row of the plan's table, and it costs less where the truth lies 0.02 to 0.04 from one
    (``calibration_cap.py``).
    """
    assert sizing.CALIBRATION_PILOT_DRAWS == 8_000
    assert sizing.PILOT_DRAWS == 2_000
    assert sizing.CALIBRATION_CAP == 50_000


IPSI_ARMS = {"C1": IPSI_PRIMARY, "C3": IPSI_THIRD, "C6": IPSI_THIRD}
IPSI_ARM_CORRELATIONS = {"C1": IPSI, "C3": IPSI, "C6": IPSI_CONTINUOUS}


def _c6_pilot(value: float) -> dict[str, float]:
    """Pilot SE ratios: C1 and C3 at one, every C6 cell at ``value``."""
    ratios = {}
    for arm, labels in IPSI_ARMS.items():
        for label in labels:
            ratios[f"{label}__{arm}"] = value if arm == "C6" else 1.0
    return ratios


@pytest.mark.parametrize(
    ("ratio", "replicates", "met"),
    [(0.97, 10_800, True), (0.96, 23_200, True), (0.95, 50_000, False), (0.94, 50_000, False)],
)
def test_wa_the_calibration_rule_marks_the_cap_and_does_not_diagnose_it(
    ratio: float, replicates: int, met: bool
) -> None:
    """With every C6 pilot ratio pinned and the other arms at one (8,000 draws per arm), the
    rule returns the size and whether it meets the budget (``calibration_cap_pinned.py``).

    At the cap the gate decides: a C6 cell at a true ratio of 0.96 fails its gate with
    probability 1.4e-13 and one at 0.95 with 2.2e-5, so the cap is not a calibration failure.
    """
    size = sizing.calibration_size(
        MARGINS,
        _c6_pilot(ratio),
        sizing.CALIBRATION_PILOT_DRAWS,
        IPSI_ARM_CORRELATIONS,
    )
    assert size.replicates == replicates
    assert size.budget_met is met
    if not met:
        assert size.replicates == sizing.CALIBRATION_CAP


def test_wa_a_ratio_inside_the_band_still_passes_the_gate_at_the_cap() -> None:
    """Why the cap leaves the verdict to the gate: the gate passes a true ratio of 0.96."""
    for ratio, failure in ((0.96, 1.4e-13), (0.95, 2.2e-5)):
        cell = float(sizing.calibration_pass(MARGINS, sizing.CALIBRATION_CAP, np.asarray(ratio)))
        assert 1.0 - cell == pytest.approx(failure, rel=0.1)
    # Outside the claim: a cell at 0.94 fails with P = 0.215, and five such cells exceed one.
    edge = float(sizing.calibration_pass(MARGINS, sizing.CALIBRATION_CAP, np.asarray(0.94)))
    assert 1.0 - edge == pytest.approx(0.215, abs=0.005)


@functools.cache
def _planning() -> dict[str, int]:
    """The planning sizes of the property families: the rule at the declared share on P4."""
    return sizing.property_bias_replicates(
        MARGINS,
        properties.PROPERTY_PILOT_BIASES,
        properties.PROPERTY_PILOT_REPLICATES,
        properties.label_correlations,
        per_family=sizing.family_share(6),
    )


def _units(gap: float, sd: float, n: int) -> float:
    """A gap in spreads of the estimator at ``n`` rows: the EIF-SE of the plan's section 5.2."""
    return gap / (sd / np.sqrt(n))


# Discrimination checks, ``(family, smallest gap in EIF-SE, checks, replications)`` per study.
# The gaps come from the laws, and the counts are the plan's.


def _sdr_gaps() -> list[tuple[str, float, int, int]]:
    binary = law.BINARY
    rule = law.RULE
    plans = ("never", "always", rule)

    def row(limit: Callable[[str], float], n: int) -> list[float]:
        value = {plan: limit(plan) for plan in plans}
        truth = {plan: binary.counterfactual(plan, 3) for plan in plans}
        out = [_units(value[p] - truth[p], binary.eif_sd(p, 3), n) for p in plans]
        for plan in ("always", rule):
            gap = (value[plan] - value["never"]) - (truth[plan] - truth["never"])
            out.append(_units(gap, binary.eif_sd(plan, 3, reference="never"), n))
        return out

    every = lambda kind: (kind,) * 3  # noqa: E731
    both_wrong = row(lambda p: binary.sdr_limit(p, 3, every("baseline"), every("intercept")), 2_000)
    plug_in = row(lambda p: binary.sdr_plug_in_limit(p, 3, "baseline"), 2_000)
    markov = row(lambda p: binary.sdr_limit(p, 3, every("markov"), every("intercept")), 8_000)
    reversed_rule = [
        _units(
            binary.counterfactual("reversed", h) - binary.counterfactual(rule, h),
            binary.eif_sd("reversed", h),
            2_000,
        )
        for h in (2, 3)
    ]
    free = {p: binary.g_formula(p, 3, history="free") for p in ("never", "always")}
    truth = {p: binary.counterfactual(p, 3) for p in ("never", "always")}
    history_free = [
        _units(free[p] - truth[p], binary.eif_sd(p, 3), 2_000) for p in ("never", "always")
    ] + [
        _units(
            (free["always"] - free["never"]) - (truth["always"] - truth["never"]),
            binary.eif_sd("always", 3, reference="never"),
            2_000,
        )
    ]
    absorbing = [
        _units(
            binary.g_formula(p, h, absorbing=True) - binary.counterfactual(p, h),
            binary.eif_sd(p, h),
            8_000,
        )
        for h in (2, 3)
        for p in plans
    ]
    smallest = lambda values: min(abs(v) for v in values)  # noqa: E731
    return [
        ("C2 both_wrong", smallest(both_wrong), 5, _planning()["double_robustness"]),
        ("correction_necessity", smallest(plug_in), 10, _planning()["targeting_necessity"]),
        ("C7 reversed rule", smallest(reversed_rule), 4, _planning()["rule_necessity"]),
        ("N1 absorbing", smallest(absorbing), 12, _planning()["repeated_outcome_necessity"]),
        ("N2 history-free", smallest(history_free), 6, _planning()["outcome_history_necessity"]),
        ("N3 Markov", smallest(markov), 10, _planning()["full_history_necessity"]),
    ]


def _ipsi_gaps() -> list[tuple[str, float, int, int]]:
    binary = law.BINARY
    single = []
    for delta in (0.5, 2.0):
        truth = binary.tilt_functional(delta, 3)
        plug = binary.ipsi_plug_in_limit(delta, 3)
        single.append(_units(plug - truth, binary.ipsi_eif_sd(delta, 3), 2_000))
        gap = (plug - binary.ipsi_plug_in_limit(1.0, 3)) - (truth - binary.tilt_functional(1.0, 3))
        single.append(_units(gap, binary.ipsi_eif_sd(delta, 3, reference=1.0), 2_000))
    absorbing = [
        _units(
            binary.tilt_functional(d, h, absorbing=True) - binary.tilt_functional(d, h),
            binary.ipsi_eif_sd(d, h),
            8_000,
        )
        for d in (0.5, 1.0, 2.0)
        for h in (2, 3)
    ]
    history_free = [
        _units(
            binary.tilt_functional(d, 3, history="free") - binary.tilt_functional(d, 3),
            binary.ipsi_eif_sd(d, 3),
            8_000,
        )
        for d in (0.5, 2.0)
    ]
    smallest = lambda values: min(abs(v) for v in values)  # noqa: E731
    return [
        ("single_robustness", smallest(single), 8, _planning()["double_robustness"]),
        ("N1 absorbing", smallest(absorbing), 12, _planning()["repeated_outcome_necessity"]),
        ("N2 history-free", smallest(history_free), 4, _planning()["outcome_history_necessity"]),
    ]


#: The derived calibration controls: the shrunken SE (ratio 0.70 of the true SE) and the noise
#: arm (ratio ``1 / sqrt 2``), each at the calibration size.  ``SDR`` has 26 and ``IPSI`` 30.
CONTROL_RATIOS = (properties.SHRUNKEN_SE_FACTOR, 2**-0.5)
CALIBRATION_R = 8_000
CHECK_COUNTS = {"SDR": (74, 26), "IPSI": (55, 30)}


@pytest.mark.parametrize("study_name", ["SDR", "IPSI"])
def test_wa_every_discrimination_check_holds_its_share_at_its_size(study_name: str) -> None:
    """Each of ``k`` checks passes with probability at least ``1 - 0.001 / k``.

    ``k`` is 74 for ``SDR`` and 55 for ``IPSI``: the sampled bias checks of the families below,
    the derived calibration controls, and the C4 pointwise control.  The planning size of a
    family is the size the property rule gives its X32a counterpart at the declared share.  The
    smallest gap is 0.73 EIF-SE (``SDR`` N2 ``always_t3`` at ``n = 2,000``).
    """
    rows = _sdr_gaps() if study_name == "SDR" else _ipsi_gaps()
    total, controls = CHECK_COUNTS[study_name]
    assert sum(count for _, _, count, _ in rows) + controls + 1 == total
    floor = sizing.discrimination_floor(total)
    assert floor == pytest.approx(1.0 - 0.001 / total)
    smallest = min(gap for _, gap, _, _ in rows)
    for family, gap, _, replicates in rows:
        probability = sizing.bias_discriminated_probability(MARGINS, gap, replicates)
        assert probability >= floor, (study_name, family, gap, replicates)
    for ratio in CONTROL_RATIOS:
        assert sizing.calibration_control_probability(MARGINS, CALIBRATION_R, ratio) >= floor
    # The binding gap is the plan's 0.73 (SDR) and 0.88 (IPSI).
    assert smallest == pytest.approx(0.73 if study_name == "SDR" else 0.88, abs=0.006)


def test_wa_the_discrimination_check_would_miss_a_gap_inside_the_margin() -> None:
    """The control for the check above: a gap below the margin of 0.25 does not clear it."""
    for replicates in (1_200, 2_800):
        assert sizing.bias_discriminated_probability(MARGINS, 0.20, replicates) < 0.01
        assert sizing.bias_discriminated_probability(MARGINS, 0.73, replicates) > 0.99999
    # A derived calibration control with the SE ratio inside the band is not discriminated.
    assert sizing.calibration_control_probability(MARGINS, CALIBRATION_R, 0.95) < 0.01


# ---------------------------------------------------------------- the rules against the probes

P2 = properties.CALIBRATION_PILOT_RATIOS
P2_WITHOUT_BOOSTED = {k: v for k, v in P2.items() if not k.endswith("crossfit_boosted")}
P2_N = properties.CALIBRATION_PILOT_REPLICATES


def test_the_primary_rule_at_the_clause_share_matches_the_probe() -> None:
    """X32a's P1 gives 2,400 at both shares, and a low pilot coverage raises the size
    (``allocation.py``: 26 cells and 2 implementations, 30 cells and 1)."""
    p1 = (properties.PRIMARY_PILOT_COVERED, properties.PRIMARY_PILOT_REPLICATES)
    assert sizing.primary_replicates(MARGINS, *p1) == 2_400
    assert sizing.primary_replicates(MARGINS, *p1, coverage_pass=sizing.PRIMARY_CLAUSE_SHARE) == (
        2_400
    )
    expected = {
        0.95: [2_400, 2_400, 2_400, 2_400],
        0.945: [2_800, 2_400, 2_800, 2_400],
        0.94: [4_000, 2_400, 4_000, 2_400],
        0.935: [6_400, 3_600, 6_000, 3_200],
        0.93: [11_200, 5_200, 10_400, 4_800],
    }
    for coverage, sizes in expected.items():
        observed = []
        for cells, implementations in ((26, 2), (30, 1)):
            for pilot in (400, 2_000):
                covered = {f"c{i}": round(coverage * pilot) for i in range(cells)}
                observed.append(
                    sizing.primary_replicates(
                        MARGINS,
                        covered,
                        pilot,
                        implementations=implementations,
                        coverage_pass=sizing.PRIMARY_CLAUSE_SHARE,
                    )
                )
        assert observed == sizes, coverage


def test_clause_d_of_the_primary_rule_reads_the_robust_probability() -> None:
    """Clause (d): the robust probability that every bias gate passes is at least the share.

    An unbiased pilot leaves the size at the pilot-free floor of 2,400.  A pilot with a common
    bias of 0.15 in every cell raises it.  So does a pilot with one cell at 0.15 and the rest
    unbiased: the outlier test reads that cell alone, and it fails its gate at 2,400
    replications with probability 0.0101, more than the clause's 0.0082.  A cell at 0.20
    cannot be cured by replications, because the pilot noise alone puts the true bias past the
    margin with probability 0.01.  The rule then asks for a larger ``n``.
    """
    covered = {f"c{i}": 1_900 for i in range(26)}
    corr = SDR(SDR_PRIMARY)

    def groups(bias: float, cell: int | None = None) -> list[sizing.PilotGroup]:
        values = np.full(len(corr), 0.0 if cell is not None else bias)
        if cell is not None:
            values[cell] = bias
        return [
            sizing.PilotGroup(values.copy(), corr / 2_000),
            sizing.PilotGroup(np.zeros(len(corr)), corr / 2_000),
        ]

    def size(pilot: list[sizing.PilotGroup] | None) -> int:
        return sizing.primary_replicates(
            MARGINS,
            covered,
            2_000,
            coverage_pass=sizing.PRIMARY_CLAUSE_SHARE,
            bias_groups=pilot,
            bias_pass=sizing.PRIMARY_CLAUSE_SHARE,
        )

    assert size(None) == 2_400
    assert size(groups(0.0)) == 2_400
    common = size(groups(0.15))
    assert common > 2_400
    one_cell = size(groups(0.15, cell=0))
    assert one_cell > 2_400
    # The size meets the share, and the step below it does not.
    pilot = groups(0.15, cell=0)

    def robust(replicates: int) -> float:
        return sizing.robust_probability(pilot, sizing._bias_gate(MARGINS, replicates))[0]

    assert robust(one_cell) >= sizing.PRIMARY_CLAUSE_SHARE
    assert robust(one_cell - sizing.BUDGET_STEP) < sizing.PRIMARY_CLAUSE_SHARE
    with pytest.raises(ValueError, match="primary rule"):
        size(groups(0.20, cell=0))


def test_the_calibration_rule_matches_the_probe() -> None:
    """P2 under the union bound: 7,200 for ``SDR`` (C5 one-sided, C4 joint band), 8,800 for
    ``IPSI`` (C5 reported, C4 joint band) and 11,600 for ``SDR`` if C5 were two-sided
    (``allocation.py``)."""
    correlations = properties.label_correlations
    assert properties.calibration_replicates(P2, P2_N) == 8_000
    sdr = sizing.calibration_size(
        MARGINS,
        P2,
        P2_N,
        correlations,
        one_sided_arms=("crossfit_boosted",),
        joint_coverage=0.95,
    )
    assert sdr == (7_200, True)
    ipsi = sizing.calibration_size(
        MARGINS, P2_WITHOUT_BOOSTED, P2_N, correlations, joint_coverage=0.95
    )
    assert ipsi == (8_800, True)
    two_sided = sizing.calibration_size(MARGINS, P2, P2_N, correlations, joint_coverage=0.95)
    assert two_sided == (11_600, True)
    # C4 enters as one more term: a joint coverage near the band's end costs replications.
    near_edge = sizing.calibration_size(
        MARGINS, P2_WITHOUT_BOOSTED, P2_N, correlations, joint_coverage=0.93
    )
    assert near_edge.replicates > ipsi.replicates


def test_the_calibration_failure_bound_is_the_union_over_the_terms() -> None:
    """The bound is the sum of the arms' failure probabilities and the C4 term."""
    correlations = properties.label_correlations
    groups = properties.calibration_groups(P2, P2_N)
    tau = sizing.heterogeneity_bound(groups)
    names = [name for arm in sizing.calibration_arms(P2).values() for name in arm]
    replicates = 7_200

    def arm_failure(group: sizing.PilotGroup, one_sided: bool) -> float:
        def cell(ratio: np.ndarray) -> np.ndarray:
            return sizing.calibration_pass(MARGINS, replicates, ratio, one_sided=one_sided)

        return 1.0 - sizing.pooled_pass_probability([group], tau, cell)

    arms = list(sizing.calibration_arms(P2))
    expected = sum(
        arm_failure(group, arm == "crossfit_boosted")
        for arm, group in zip(arms, groups, strict=True)
    )
    expected += 1.0 - sizing.joint_band_pass_probability(MARGINS, replicates, 0.95)
    bound = sizing.calibration_failure_bound(
        MARGINS,
        replicates,
        groups,
        tau,
        names,
        one_sided_arms=("crossfit_boosted",),
        joint_coverage=0.95,
    )
    assert bound == pytest.approx(expected, abs=1e-15)
    assert bound <= sizing.SIZED_FAIL
    # The union bound is at least the failure of any one term.
    assert bound >= 1.0 - sizing.joint_band_pass_probability(MARGINS, replicates, 0.95)
    assert correlations is properties.label_correlations


def test_the_property_rule_at_the_declared_share_matches_the_probe() -> None:
    """P4 at the share ``1 - d / 6`` for each of six families (``allocation.py``)."""
    sizes = sizing.property_bias_replicates(
        MARGINS,
        properties.PROPERTY_PILOT_BIASES,
        properties.PROPERTY_PILOT_REPLICATES,
        properties.label_correlations,
        per_family=sizing.family_share(6),
    )
    assert sizes == _planning()
    assert sizes == {
        "double_robustness": 1_800,
        "full_history_necessity": 1_600,
        "outcome_history_necessity": 1_600,
        "repeated_outcome_necessity": 4_000,
        "rule_necessity": 3_200,
        "targeting_necessity": 2_200,
    }
    product = sizing.property_bias_pass_probability(
        MARGINS,
        properties.PROPERTY_PILOT_BIASES,
        properties.PROPERTY_PILOT_REPLICATES,
        sizes,
        properties.label_correlations,
    )
    assert product == pytest.approx(0.9871, abs=5e-5)
    assert 1.0 - product <= sizing.SIZED_FAIL


def test_the_one_sided_c5_cell_differs_from_the_two_sided_cell_above_the_band() -> None:
    """C5 reads the lower end of the band only (the ``SDR`` boosted arm is conservative).

    At the X32a boosted arm's mean ratio 1.0162 and 8,000 replications, the one-sided cell
    passes with probability 1.000000 and the two-sided cell with 0.999981.  The nonzero
    witness is a ratio above the band: the two-sided cell fails it and the one-sided cell
    passes.
    """
    ratio = np.asarray(1.0162)
    one = sizing.calibration_pass(MARGINS, 8_000, ratio, one_sided=True)
    two = sizing.calibration_pass(MARGINS, 8_000, ratio)
    assert float(one) == pytest.approx(1.0, abs=5e-7)
    assert float(two) == pytest.approx(0.999981, abs=5e-7)
    above = np.asarray(1.12)
    assert float(sizing.calibration_pass(MARGINS, 8_000, above)) < 1e-6
    assert float(sizing.calibration_pass(MARGINS, 8_000, above, one_sided=True)) > 0.99
    # Below the band both cells fail.
    below = np.asarray(0.88)
    assert float(sizing.calibration_pass(MARGINS, 8_000, below, one_sided=True)) < 1e-6


def test_the_coverage_half_reads_the_pilot_bias() -> None:
    r"""The Wald coverage at ``(r, b)`` is :math:`\Phi(1.96 r - b) - \Phi(-1.96 r - b)`.

    At zero bias it is X32a's :math:`2\Phi(1.96 r) - 1`.  A bias of 0.3 spreads moves the
    coverage of a calibrated cell from 0.950 to 0.940, and the cell then passes less often.
    """
    critical = float(norm.ppf(0.975))
    unbiased = sizing.wald_coverage(MARGINS, np.asarray(1.0), 0.0)
    assert float(unbiased) == pytest.approx(2 * norm.cdf(critical) - 1, abs=1e-15)
    assert float(unbiased) == pytest.approx(
        coverage_for_se_ratio(1.0, alpha=MARGINS.alpha), abs=1e-12
    )
    moved = sizing.wald_coverage(MARGINS, np.asarray(1.0), 0.3)
    assert float(moved) == pytest.approx(norm.cdf(critical - 0.3) - norm.cdf(-critical - 0.3))
    assert float(moved) == pytest.approx(0.9396, abs=5e-5)
    ratio = np.asarray(1.0)
    at_zero = float(sizing.calibration_pass(MARGINS, 4_000, ratio))
    assert at_zero == pytest.approx(float(properties._calibration_pass(4_000, ratio)), abs=0)
    assert float(sizing.calibration_pass(MARGINS, 4_000, ratio, bias=0.3)) < at_zero
    # A bias of 0.9 puts the coverage at 0.89, below the band, so the cell fails.
    assert float(sizing.calibration_pass(MARGINS, 4_000, ratio, bias=0.9)) < 1e-3
    # The bias reaches the rule: a pilot with a bias at every cell needs more replications.
    biases = dict.fromkeys(P2, 0.3)
    base = sizing.calibration_size(
        MARGINS,
        P2,
        P2_N,
        properties.label_correlations,
        one_sided_arms=("crossfit_boosted",),
    )
    biased = sizing.calibration_size(
        MARGINS,
        P2,
        P2_N,
        properties.label_correlations,
        one_sided_arms=("crossfit_boosted",),
        biases=biases,
    )
    assert biased.replicates > base.replicates


def test_the_c4_joint_band_is_the_exact_interval_inside_the_band() -> None:
    """The cell passes when the exact interval of the joint coverage lies inside [0.92, 0.98]."""
    accepted = sizing.accepted_counts(MARGINS, 8_000)
    assert accepted is not None
    assert sizing.joint_band_pass_probability(MARGINS, 8_000, 0.95) == pytest.approx(1.0, abs=1e-12)
    assert sizing.joint_band_pass_probability(MARGINS, 8_000, 0.915) == pytest.approx(
        1.3e-5, rel=0.02
    )
    # Coverage on the edge of the band passes about as often as the covered count is accepted.
    assert 0.001 < sizing.joint_band_pass_probability(MARGINS, 8_000, 0.92) < 0.01
    assert sizing.joint_band_pass_probability(MARGINS, 100, 0.95) == 0.0


def test_the_x32a_binding_reproduces_the_shared_rules() -> None:
    """X32a's functions are the shared rules at its margins and correlations."""
    assert properties.bias_equivalent_probability(0.15, 2_400) == pytest.approx(
        float(sizing.bias_equivalent_probability(MARGINS, 0.15, 2_400)), abs=0
    )
    groups = properties.calibration_groups(P2, P2_N)
    shared = sizing.calibration_groups(P2, P2_N, properties.label_correlations)
    assert len(groups) == len(shared) == 4
    for left, right in zip(groups, shared, strict=True):
        np.testing.assert_array_equal(left.values, right.values)
        np.testing.assert_array_equal(left.noise, right.noise)
    # A mapping of correlation functions gives each arm its own.
    mapped = sizing.calibration_groups(
        P2,
        P2_N,
        dict.fromkeys(
            ("estimated_mechanisms", "crossfit_glm", "crossfit_boosted", "continuous_gaussian"),
            properties.label_correlations,
        ),
    )
    for left, right in zip(shared, mapped, strict=True):
        np.testing.assert_array_equal(left.noise, right.noise)
