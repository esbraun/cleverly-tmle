"""The declared design of ``hazard-time-varying-multinode`` (plan X70 section 5, (b2)).

Every check here reads law (M) of :mod:`tests.studies.hazard_time_varying_multinode_law` and the
declared constants, before any pilot or run: the truths against an independent forward sum, the
exact-sample fits, positivity, the efficiency bounds, every robustness limit and every witness
of a component that could vanish at the truth, the sizes that need no pilot, the record and the
seeds.  The readings that need a committed pilot (the calibration, primary and wrong-learner
counts, the band, the influence-curve gap, the agreement report) read
``tests/canonical/hazard_time_varying_multinode/pilot.csv.gz``, as in
``tests/unit/test_hazard_time_varying_study_design.py``.  A pilot group with some labels
committed fails, naming the rest; a group with none committed skips.  The calibration screen
is centred on each arm's expected SE ratio from an exact simulation of the law, and the large
fit's influence-curve SD is read against each arm's exact kurtosis (plan X70 rev 3.8 and its
addendum).

Each gap below is in efficiency SEs at the arm's size: the exact limit minus the truth over
:math:`\\sigma / \\sqrt{n}`, with :math:`\\sigma` the SD of the efficient influence function, at
:math:`n = 2{,}000`, or :math:`n = 8{,}000` for an untargeted ``outcome_correct`` arm.  An exact limit is the fit of the law as data, every support point once and
weighted by its mass.
"""

from __future__ import annotations

import itertools
import json
import re
import warnings
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import chisquare, norm, poisson

from cleverly._support_rows import LONGITUDINAL_ROWS
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal import hazard_backward as backward
from cleverly.utils.bounds import DEFAULT_LTMLE_G_BOUNDS
from tests.studies import canonical_hazard_time_varying_multinode as study
from tests.studies import hazard_time_varying_multinode_law as law
from tests.studies import hazard_time_varying_multinode_pilot as pilot
from tests.studies import hazard_time_varying_multinode_properties as properties
from tests.studies import mtp_study_tools
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed
from tests.unit.test_hazard_time_varying import mutated_panel
from tests.unit.test_hazard_time_varying_multinode import (
    _held_slot_at_later_nodes,
    _level_plan_designs,
    _observed_slot,
    _policy_at_level_paths,
    patched,
)

MARGINS = study.STUDY.margins
CONTRAST = study.CONTRAST_LABEL
N = properties.DOUBLE_ROBUST_N
#: The fit's hazard clip: every logit lies in ``[logit(1 - alpha), logit(alpha)]``.
HAZARD_ALPHA = 0.9995

#: The untargeted exact-law gap of each wrong-learner configuration's contrast and policy
#: arms.  ``mechanism_correct`` is the targeting-necessity control (floor 3 SEs);
#: ``continuation_wrong`` keeps the hazard right, so its plug-in misses through the continuation
#: alone (floor 1 SE, (b1)'s +1.369 SE witness).
PLUG_IN_GAPS = {
    ("multinode_contrast", "mechanism_correct"): 7.676,
    ("multinode_policy", "mechanism_correct"): 6.165,
    ("multinode_contrast", "continuation_wrong"): 0.796,
    ("multinode_policy", "continuation_wrong"): 0.585,
}
#: The one declared shortfall (plan X70 rev 3.7): the covariate term's witness, the untargeted
#: ``continuation_wrong`` contrast, against its 1 SE floor.  At ``T = 3`` only node 2's
#: continuation reads a later ``L``.  Its arm discriminates it with probability 1 at spreads up
#: to 1.5 SEs, and the 4.4 witness
#: ``test_a_wrong_hazard_and_continuation_with_the_right_mechanism_is_exact`` carries the term
#: on a law built for it.
DECLARED_SHORTFALL = {("multinode_contrast", "continuation_wrong"): 0.796}
#: The targeted both-wrong control's gap on the contrast.
BOTH_WRONG_GAP = 7.026
#: The baseline-only fit's gap on the contrast: always and never fitted with ``L`` dropped,
#: which (a)'s engine fits.  ``L_k`` confounds the later treatments, so it misses.
BASELINE_ONLY_GAP = 7.028

#: Each slot mutation of the unit tests, as a context manager.
MUTATIONS: dict[str, Callable[[], AbstractContextManager[Any]]] = {
    # every node of the prediction design set to the slot's level (the level plan of (a))
    "level_plan": lambda: patched(hazard_module, "_current_level_designs", _level_plan_designs),
    # at k >= 2 the held branch: one slot, the hazard at the observed A_1
    "held_slot": lambda: patched(backward, "node_slots", _held_slot_at_later_nodes()),
    # the policy evaluated with every earlier treatment set to the slot's level
    "policy_prefix": lambda: patched(backward, "node_slots", _policy_at_level_paths()),
    # the observed A_k in place of the policy mixture at every policy node
    "observed_slot": lambda: mutated_panel(_observed_slot),
}
#: ``policy_risk`` is the policy's risk at the last node, which the policy-mixture witness reads.
POLICY_RISK = "policy_risk"
#: The exact limit of each mutation on the ``outcome_correct`` fit, targeted and untargeted, by
#: ``(mutation, arm, targeted)``: a gated arm of the study reads each (the untargeted ones are
#: the ``__outcome_correct_untargeted`` arms, in SEs at their n = 8,000, twice the gap at 2,000).  At a static or rule node every follower's
#: observed ``A_k`` is the slot, so the walk fits the slot's hazard to the followers' own events
#: and repairs a slot defect under any mechanism: the rule's targeted gaps are small, and its
#: untargeted ones are not.  A defect of the policy's prefix or slots changes the mixture that
#: defines the target, so the walk cannot repair it.
WITNESS_GAPS: dict[tuple[str, str, bool], float] = {
    ("level_plan", "multinode_rule", True): 0.112,
    ("level_plan", "multinode_rule", False): -8.372,
    ("level_plan", "multinode_policy", True): 0.857,
    ("level_plan", "multinode_policy", False): -1.028,
    ("held_slot", "multinode_rule", True): -0.190,
    ("held_slot", "multinode_rule", False): 2.986,
    ("held_slot", "multinode_policy", True): 1.325,
    ("held_slot", "multinode_policy", False): 3.902,
    ("policy_prefix", "multinode_policy", True): 1.143,
    ("policy_prefix", "multinode_policy", False): 2.286,
    ("observed_slot", "multinode_policy", True): 3.497,
    ("observed_slot", POLICY_RISK, True): 4.359,
}
#: The plan's floor of each component's witness: 3 SEs, or, where no coefficient inside the
#: positivity and hazard limits reaches it, the largest gap reached and at least 1 SE.  With the
#: untargeted arms at n = 8,000 the held slot reaches 3 (3.90); the policy's prefix, at 2.29,
#: keeps the fallback.
COMPONENT_FLOORS = {
    "level_plan": 3.0,
    "held_slot": 3.0,
    "policy_prefix": 1.0,
    "observed_slot": 3.0,
}
#: The measured witnesses under 1 SE.  Each component's largest witness meets its floor, so
#: none of these is a shortfall: a component shows on another arm.
BELOW_FALLBACK = frozenset(
    {
        ("level_plan", "multinode_rule", True),
        ("level_plan", "multinode_policy", True),
        ("held_slot", "multinode_rule", True),
    }
)
#: The replicates of a robustness arm, which every witness above is read at: the targeted arms'
#: configurations and the untargeted arms each run at least this many.
ARM_REPLICATES = 1_200


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int = N) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _policy_risk() -> law.Target:
    return study.target_of(study.risk_name("policy", study.LAST))


def _gap(result: Any, arm: str, n: int = N) -> float:
    """The exact-limit gap of one arm (a cell label or :data:`POLICY_RISK`) in SEs at ``n``."""
    if arm == POLICY_RISK:
        aim = _policy_risk()
        estimate = float(result[study.risk_name("policy", study.LAST)].psi)
        return (estimate - aim.value(law.LAW)) / (law.efficiency_sd(law.LAW, aim) / np.sqrt(n))
    name = properties.ESTIMAND[arm]
    return (float(result[name].psi) - properties.TRUTHS[arm]) / _scale(arm, n)


def _witness_n(targeted: bool) -> int:
    """The n of the arm a witness is read on: the untargeted arms run at n = 8,000."""
    return N if targeted else properties.UNTARGETED_N


@pytest.fixture(scope="module")
def exact_fits() -> dict[tuple[str, bool], Any]:
    """The exact-sample fit of every configuration, targeted and, where read, untargeted."""
    fits = {
        (configuration, True): _quiet(study.exact_fit, configuration)
        for configuration in study.CONFIGURATIONS
    }
    for configuration in ("both_correct", "mechanism_correct", "continuation_wrong"):
        fits[(configuration, False)] = _quiet(study.exact_fit, configuration, targeted=False)
    return fits


@pytest.fixture(scope="module")
def mutated_fits() -> dict[tuple[str, bool], Any]:
    """The ``outcome_correct`` exact-sample fit under each slot mutation, targeted or not."""
    out = {}
    for name, mutation in MUTATIONS.items():
        for targeted in (True, False):
            with mutation():
                out[(name, targeted)] = _quiet(
                    study.exact_fit, "outcome_correct", targeted=targeted
                )
    return out


# ---------------------------------------------------------------------------- independent routes


def _forward_truth(density: law.Density, horizon: int) -> float:
    """``psi_r(t)`` by a forward sum over every path of covariates and treatments.

    Apart from the law module's recursion: each path of ``(A_1, L_2, A_2, ..., L_t, A_t)``
    carries its probability under the regimen given ``W``, and along it the risk accumulates
    as survival times the hazard at each node.  Only the coefficients and the regimen are read.
    """
    c = law.COEFFICIENTS
    total = 0.0
    for w1, w2 in law.STRATA:
        for arms in itertools.product((0, 1), repeat=horizon):
            for path in itertools.product((1, -1), repeat=horizon - 1):
                weight, risk, survival = 1.0, 0.0, 1.0
                previous_arm, previous_l = 0, None
                for node in range(1, horizon + 1):
                    l = None if node == 1 else path[node - 2]
                    if node > 1:
                        l0, l1, la, ll = c.covariate
                        eta = l0 + l1 * w1 + la * previous_arm + ll * float(previous_l == 1)
                        p = float(expit(eta))
                        weight *= p if l == 1 else 1.0 - p
                    q = density(node, w1, previous_arm, l)
                    a = arms[node - 1]
                    weight *= q if a == 1 else 1.0 - q
                    h0, ht, h1, h2, hp, ha, hl = c.hazard
                    hazard = float(
                        expit(
                            h0
                            + ht * (node - 1)
                            + h1 * w1
                            + h2 * w2
                            + hp * previous_arm
                            + ha * a
                            + hl * float(l == 1)
                        )
                    )
                    risk += survival * hazard
                    survival *= 1.0 - hazard
                    previous_arm, previous_l = a, l
                total += law.MASS * weight * risk
    return total


# ---------------------------------------------------------------------------- truths


class TestTheTruths:
    def test_the_backward_truth_is_the_forward_sum(self) -> None:
        for label, density in law.REGIMENS.items():
            for horizon in range(1, law.LAW.n_times + 1):
                assert law.LAW.truth(density, horizon) == pytest.approx(
                    _forward_truth(density, horizon), abs=1e-14
                ), (label, horizon)

    def test_the_exact_sample_fit_returns_every_truth(
        self, exact_fits: dict[tuple[str, bool], Any]
    ) -> None:
        result = exact_fits[("both_correct", True)]
        for name, value in study.TRUTH["multinode"].items():
            assert result[name].psi == pytest.approx(value, abs=study.EXACT), name

    def test_the_support_is_the_law(self) -> None:
        assert law.LAW.probs.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(law.LAW.probs > 0.0)
        assert len(set(law.LAW.support)) == len(law.LAW.support)

    def test_a_large_draw_matches_the_support_jointly(self) -> None:
        """The support-point frequencies of 200,000 draws against the exact probabilities.

        Each unit is mapped to the support point it realises, and a chi-square test over the
        points with expected count at least 5 (the rest pooled) must not reject at 1e-4.
        """
        n = 200_000
        frame = law.LAW.draw(n, stream_seed(study.STUDY, "design_joint", "multinode", 0))
        counts = Counter(law._keys(law.LAW, frame))
        assert set(counts) <= set(law.LAW.support)
        expected = law.LAW.probs * n
        observed = np.array([counts.get(point, 0) for point in law.LAW.support], dtype=float)
        large = expected >= 5.0
        pooled_expected = np.append(expected[large], expected[~large].sum())
        pooled_observed = np.append(observed[large], observed[~large].sum())
        assert chisquare(pooled_observed, pooled_expected).pvalue > 1e-4

    def test_the_declared_truths(self) -> None:
        np.testing.assert_allclose(
            [properties.TRUTHS[f"multinode_{arm}"] for arm in study.ARMS],
            [-0.727560, -0.319157, -0.336952, 0.128960],
            atol=5e-6,
        )
        for arm in study.ARMS:
            assert properties.TRUTHS[f"crossfit_{arm}"] == properties.TRUTHS[f"multinode_{arm}"]
        assert study.TRUTH["multinode"] == study.truths()

    def test_the_curves_rise(self) -> None:
        for density in law.REGIMENS.values():
            values = [law.LAW.truth(density, t) for t in range(1, law.LAW.n_times + 1)]
            assert np.all(np.diff(values) > 0.0)

    def test_the_contrast_is_far_from_zero(self) -> None:
        """The current-level slot's witness: always against never at the last node is at least
        5 SEs from zero, so a slot that read one level for every regimen would miss it."""
        assert properties.TRUTHS[CONTRAST] / _scale(CONTRAST) == pytest.approx(-31.272, abs=1e-3)


class TestThePositivity:
    def test_the_bounds_never_bind(self) -> None:
        """Every ``g_k`` lies in ``[0.15, 0.85]``, the smallest cumulative mechanism over every
        path of treatments and covariates stays above the default bound, and every hazard stays
        at least twice the clip's ``1 - alpha``."""
        reading = law.positivity(law.LAW)
        assert reading["g_min"] >= 0.15 and reading["g_max"] <= 0.85
        assert reading["cumulative_min"] > DEFAULT_LTMLE_G_BOUNDS[0]
        assert reading["hazard_min"] >= 2.0 * (1.0 - HAZARD_ALPHA)
        assert reading["hazard_max"] <= HAZARD_ALPHA
        expected = {
            "g_min": 0.19782,
            "g_max": 0.78583,
            "hazard_min": 0.00333,
            "retention_min": 0.72112,
            "cumulative_min": 0.01356,
        }
        for key, value in expected.items():
            assert reading[key] == pytest.approx(value, abs=1e-5), key

    def test_every_regimen_keeps_its_followers(self) -> None:
        masses = {
            label: min(law.follower_mass(law.LAW, density, k) for k in (1, 2, 3))
            for label, density in law.REGIMENS.items()
        }
        assert min(masses.values()) >= 0.05, masses
        assert masses["never"] == pytest.approx(0.0724, abs=1e-4)


# ---------------------------------------------------------------------------- held cells


def _cell_masses() -> dict[tuple[str, int, int], float]:
    """The mass of an event at node ``k`` among a regimen's followers with ``A_k = a``.

    Keyed ``(regimen, k, a)``, over every follower retained at ``k``: the cells the node panel
    keys by the node's treatment, and holds when no unit of the cell has the event.  A cell no
    follower reaches is not held (``never`` has no follower at level 1).
    """
    masses: dict[tuple[str, int, int], float] = {}
    for label, density in law.REGIMENS.items():
        for point, mass in zip(law.LAW.support, law.LAW.probs, strict=True):
            w1, previous_arm = point[0], 0
            for node in range(1, law.LAW.n_times + 1):
                a = law.LAW.value(point, "A", node)
                if a is None:
                    break
                q = density(node, w1, previous_arm, law.LAW.value(point, "L", node))
                if (q if a == 1 else 1.0 - q) == 0.0:
                    break
                key = (label, node, a)
                retained = law.LAW.value(point, "C", node) == 1
                event = retained and law.LAW.value(point, "Y", node) == 1
                masses[key] = masses.get(key, 0.0) + (float(mass) if event else 0.0)
                if not retained or event:
                    break
                previous_arm = a
    return masses


def expected_held(n: int) -> float:
    """The expected count of held cells in one draw of ``n`` rows.

    A cell of mass :math:`v` holds no event with probability :math:`(1 - v)^n`; the sum over
    every cell bounds the chance that a draw holds one.  A continuation cell cannot be held:
    its soft labels come from logistic hazards and are positive.
    """
    return float(sum((1.0 - mass) ** n for mass in _cell_masses().values()))


def held_allowance(draws: int) -> int:
    """The held draws a set may drop: the failure bound's ``floor(0.01 R)``, one for a smoke."""
    strict = int(np.floor(properties.FAILURE_BOUND * draws))
    return max(1, strict) if draws < 100 else strict


def held_units() -> list[tuple[str, int, int]]:
    """Every set of draws the study and its pilot fit: ``(what, n, draws)``.

    A key that runs several fits of one draw counts once, since a held cell depends on the
    draw alone.
    """
    units = [
        (f"{family}/{stream}", n, replicates)
        for family, stream, _, n, replicates, _ in properties.FIT_SETS
    ]
    units.append(("primary/multinode", study.PRIMARY_N, study.PRIMARY_REPLICATES))
    units += [
        (f"pilot/{label}", pilot.pilot_n(label), draws)
        for label, (_, draws) in pilot.PILOTS.items()
        if label != "ic_gap"
    ]
    sizes = Counter((fit, n) for fit, n, _ in pilot.IC_GAP_JOBS)
    units += [(f"pilot/ic_gap/{fit}/{n}", n, draws) for (fit, n), draws in sizes.items()]
    return units


class TestTheHeldCells:
    def test_the_held_fit_is_a_recorded_lost_replicate(self) -> None:
        """At 60 rows a held cell is all but certain; it raises a recordable ``DataError``."""
        from cleverly.exceptions import DataError
        from cleverly.validation._failures import recordable

        assert expected_held(60) > 0.5
        with pytest.raises(study.HeldCellError) as caught:
            _quiet(study.fit_recorded, "multinode", law.LAW.draw(60, 0))
        assert isinstance(caught.value, DataError)
        assert recordable(caught.value)

    def test_the_pilot_writes_a_held_draw_as_held(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pilot, "PILOT_N", 60)
        rows = [_quiet(pilot._one, ("calibration_multinode", r)) for r in range(3)]
        held = [found for found in rows if len(found) == 1 and found[0]["held"] == 1]
        assert held, "no small draw held a cell"
        assert all(found[0]["refusal"] == "" for found in held)

    def test_the_expected_held_fits_stay_within_each_allowance(self) -> None:
        """Each set's expected held count is ``R`` times :func:`expected_held` at its n; its
        excess over the allowance is a Poisson tail, and the union of the tails stays below
        1e-3."""
        per_fit = {n: expected_held(n) for _, n, _ in held_units()}
        tails = {
            what: float(poisson.sf(held_allowance(draws), draws * per_fit[n]))
            for what, n, draws in held_units()
        }
        assert sum(tails.values()) < 1e-3, tails
        total = sum(draws * per_fit[n] for _, n, draws in held_units())
        assert total < 0.1, total


# ---------------------------------------------------------------------------- bounds


class TestTheEfficiencyBounds:
    def test_the_exact_fit_influence_curve_is_the_efficient_one(
        self, exact_fits: dict[tuple[str, bool], Any]
    ) -> None:
        """At the true nuisances the fit's influence curve is the efficient influence function:
        on the exact sample each row's curve is the support point's value times its weight over
        the mean weight."""
        result = exact_fits[("both_correct", True)]
        scale = law.LAW.probs * law.LAW.probs.size
        for label, name in properties.COEFFICIENTS["multinode"]:
            curve = np.asarray(result[name].influence_curve)
            np.testing.assert_allclose(
                curve, properties.target(label).curve(law.LAW) * scale, atol=1e-5
            )

    def test_the_efficient_influence_function_is_centred(self) -> None:
        for name in study.estimand_names():
            curve = study.target_of(name).curve(law.LAW)
            assert float(np.sum(law.LAW.probs * curve)) == pytest.approx(0.0, abs=1e-12), name

    def test_the_declared_bounds(self) -> None:
        np.testing.assert_allclose(
            [properties.EFFICIENCY_SD[f"multinode_{arm}"] for arm in study.ARMS],
            [1.04046, 1.01905, 0.59720, 0.74827],
            atol=5e-5,
        )

    def test_the_vectorized_curve_is_the_support_curve(self) -> None:
        frame = law.LAW.exact_frame().drop(columns="wt")
        for label, _ in properties.COEFFICIENTS["multinode"]:
            aim = properties.target(label)
            np.testing.assert_allclose(
                law.LAW.frame_curves(aim, frame), aim.curve(law.LAW), atol=1e-14
            )


# ---------------------------------------------------------------------------- the learners


class TestTheLearners:
    def test_the_window_keeps_w_the_last_l_and_the_last_arms(self) -> None:
        design = np.array(
            [[0.0, 2.0, 1.0, -1.0, 1.0, 0.0, 1.0], [1.0, 3.0, -1.0, 1.0, 0.0, 1.0, 0.0]]
        )
        assert list(law.Window(law.NewtonLogistic(), arms=2)._kept(design)) == [0, 1, 3, 5, 6]
        assert list(law.Window(law.NewtonLogistic(), arms=1)._kept(design)) == [0, 1, 3, 6]
        dropped = law.Window(law.NewtonLogistic(), arms=2, drop_l=True)._kept(design)
        assert list(dropped) == [0, 1, 5, 6]
        first = np.array([[0.0, 2.0, 1.0], [1.0, 3.0, 0.0]])
        assert list(law.Window(law.NewtonLogistic(), arms=2)._kept(first)) == [0, 1, 2]

    def test_the_newton_logistic_is_the_same_model(self) -> None:
        """It and the quasi-Newton learner fit one unpenalised model; Newton solves it further."""
        rng = np.random.default_rng(0)
        design = rng.normal(size=(400, 3))
        outcome = rng.random(400) < expit(design @ np.array([0.5, -1.0, 0.3]))
        newton = law.NewtonLogistic().fit(design, outcome)
        quasi = law.Logistic().fit(design, outcome)
        np.testing.assert_allclose(
            newton.predict_proba(design), quasi.predict_proba(design), atol=1e-6
        )

    def test_the_untargeted_pass_of_the_correct_learners_is_the_truth(
        self, exact_fits: dict[tuple[str, bool], Any]
    ) -> None:
        """With the walks skipped only the initial fits remain, so the plug-in reaches every
        truth on the exact sample only when the hazard and the continuation are correct."""
        result = exact_fits[("both_correct", False)]
        for name, value in study.TRUTH["multinode"].items():
            assert result[name].psi == pytest.approx(value, abs=study.EXACT), name


# ---------------------------------------------------------------------------- robustness


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(
        self, exact_fits: dict[tuple[str, bool], Any]
    ) -> None:
        """Every estimate of every configuration but ``both_wrong`` reaches its truth."""
        for configuration in study.CONFIGURATIONS:
            if configuration == "both_wrong":
                continue
            result = exact_fits[(configuration, True)]
            for name, value in study.TRUTH["multinode"].items():
                assert result[name].psi == pytest.approx(value, abs=study.EXACT), (
                    configuration,
                    name,
                )

    def test_the_untargeted_gaps(self, exact_fits: dict[tuple[str, bool], Any]) -> None:
        """Each wrong-learner configuration's untargeted plug-in misses by the declared gap.

        ``mechanism_correct`` misses by 7.7 SEs, above the necessity control's floor of 3.
        ``continuation_wrong`` keeps the hazard correct, so the hazard term of the walk is zero
        and the covariate term carries its whole correction: 0.80 SE on the contrast, below the
        plan's 1 SE floor (at most 0.96 at the coefficients searched that keep the contrast
        5 SEs from zero; at ``T = 3`` only node 2's continuation reads a later ``L``).  The declaration states the shortfall; the 4.4
        witness ``test_a_wrong_hazard_and_continuation_with_the_right_mechanism_is_exact``
        carries the covariate term on a law built for it.
        """
        for (label, configuration), recorded in PLUG_IN_GAPS.items():
            gap = _gap(exact_fits[(configuration, False)], label)
            assert gap == pytest.approx(recorded, abs=1e-3), (label, configuration)
        assert PLUG_IN_GAPS[(CONTRAST, "mechanism_correct")] >= 3.0
        assert PLUG_IN_GAPS[(CONTRAST, "continuation_wrong")] < 1.0
        # The wrong-learner configurations whose mechanism is correct: the walk repairs them.
        assert {configuration for _, configuration in PLUG_IN_GAPS} == {
            configuration
            for configuration in properties.WRONG_LEARNER_ARMS
            if study.CONFIGURATIONS[configuration][2]
        }

    def test_the_both_wrong_control_is_discriminated(
        self, exact_fits: dict[tuple[str, bool], Any]
    ) -> None:
        """The both-wrong control (mechanism without ``L_k``, hazard and continuation without
        ``L``) misses by 7.0 SEs, which its fits discriminate at a spread of 1 and 1.5 SEs."""
        gap = _gap(exact_fits[("both_wrong", True)], CONTRAST)
        assert gap == pytest.approx(BOTH_WRONG_GAP, abs=1e-3)
        replicates = properties.configuration_replicates("both_wrong")
        for spread in (1.0, 1.5):
            probability = sizing.bias_discriminated_probability(MARGINS, gap / spread, replicates)
            assert probability >= sizing.discrimination_floor(2)


# ---------------------------------------------------------------------------- witnesses


class TestTheWitnesses:
    def test_the_slot_witnesses(self, mutated_fits: dict[tuple[str, bool], Any]) -> None:
        """Each slot mutation's exact limit on the ``outcome_correct`` fit, against its floor.

        Each component's largest witness meets the plan's floor: the level plan on the rule's
        untargeted arm (-8.37 SEs at n = 8,000), the held slot on the policy's untargeted arm
        (3.90), the policy's prefix (2.29 untargeted, 1.14 targeted) and the policy mixture (3.50
        on the targeted contrast).  At n = 2,000 the untargeted gaps were half these.
        """
        for (mutation, arm, targeted), recorded in WITNESS_GAPS.items():
            gap = _gap(mutated_fits[(mutation, targeted)], arm, _witness_n(targeted))
            assert gap == pytest.approx(recorded, abs=1e-3), (mutation, arm, targeted)
        for component, floor in COMPONENT_FLOORS.items():
            largest = max(
                abs(gap)
                for (mutation, arm, _), gap in WITNESS_GAPS.items()
                if mutation == component and arm != POLICY_RISK
            )
            assert largest >= floor, (component, largest)
        below = {key for key, gap in WITNESS_GAPS.items() if abs(gap) < 1.0}
        assert below == BELOW_FALLBACK

    def test_each_component_is_discriminated(self) -> None:
        """Each component's largest witness on a declared arm clears the bias margin at the
        arm's 1,200 replicates, with the two-check floor, at spreads of 1 and 1.5 SEs."""
        assert properties.UNTARGETED_REPLICATES >= ARM_REPLICATES
        assert min(map(properties.configuration_replicates, study.CONFIGURATIONS)) >= ARM_REPLICATES
        for component in COMPONENT_FLOORS:
            largest = max(
                abs(gap)
                for (mutation, arm, _), gap in WITNESS_GAPS.items()
                if mutation == component and arm != POLICY_RISK
            )
            for spread in (1.0, 1.5):
                probability = sizing.bias_discriminated_probability(
                    MARGINS, largest / spread, ARM_REPLICATES
                )
                assert probability >= sizing.discrimination_floor(2), (component, spread)

    def test_the_declared_shortfall_is_discriminated(
        self, exact_fits: dict[tuple[str, bool], Any]
    ) -> None:
        """The one witness under its floor, the untargeted ``continuation_wrong`` contrast at
        0.80 SE, is still discriminated at spreads up to 1.5 SEs."""
        for (label, configuration), recorded in DECLARED_SHORTFALL.items():
            gap = _gap(exact_fits[(configuration, False)], label)
            assert gap == pytest.approx(recorded, abs=1e-3)
            assert abs(gap) < 1.0
            replicates = properties.configuration_replicates(configuration)
            for spread in (1.0, 1.25, 1.5):
                probability = sizing.bias_discriminated_probability(
                    MARGINS, abs(gap) / spread, replicates
                )
                assert probability >= sizing.discrimination_floor(2), spread

    def test_the_untargeted_positive_limits_are_the_truth(
        self, mutated_fits: dict[tuple[str, bool], Any]
    ) -> None:
        """The untargeted robustness arms read the backward plug-in of the ``outcome_correct``
        fit, whose hazard and continuation are correct: on the exact sample it is the truth."""
        result = _quiet(study.exact_fit, "outcome_correct", targeted=False)
        assert {f"{label}__{c}" for label, c in study.UNTARGETED_ARMS} == {
            "multinode_rule__outcome_correct",
            "multinode_policy__outcome_correct",
        }
        for name, value in study.TRUTH["multinode"].items():
            assert result[name].psi == pytest.approx(value, abs=study.EXACT), name

    def test_a_static_arm_holds_one_slot(self, mutated_fits: dict[tuple[str, bool], Any]) -> None:
        """The observed-slot and prefix mutations touch policy nodes only: a static arm or the
        rule holds one slot, so their limits are the truth under the mutation."""
        for mutation in ("observed_slot", "policy_prefix"):
            for targeted in (True, False):
                for arm in ("multinode_contrast", "multinode_rule", "multinode_risk"):
                    gap = _gap(mutated_fits[(mutation, targeted)], arm)
                    assert gap == pytest.approx(0.0, abs=1e-5)

    def test_the_baseline_only_fit_misses(self) -> None:
        """``L_k`` confounds the later treatments: always and never fitted on the same law with
        ``L`` dropped, which (a)'s engine fits, miss the contrast by at least 3 SEs."""
        columns = law.LAW.columns(weights="wt")
        columns["time_varying"] = [[] for _ in range(law.LAW.n_times)]
        frame = law.LAW.exact_frame().drop(columns=["L2", "L3"])
        learners = study.learners("both_correct")
        del learners["pseudo_learner"]
        static = {label: study.regimens()[label] for label in ("always", "never")}
        result = _quiet(
            LTMLE(
                static,
                reference=study.REFERENCE,
                n_folds=1,
                simultaneous=False,
                survival_model="hazard",
                targeting=study.ALGORITHM,
                tol=study.TOL,
                random_state=0,
                **learners,
            ).fit,
            frame,
            **columns,
        )
        gap = _gap(result, CONTRAST)
        assert gap == pytest.approx(BASELINE_ONLY_GAP, abs=1e-3)
        assert gap >= 3.0


# ---------------------------------------------------------------------------- the pilot


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()


def _pilot(label: str) -> pd.DataFrame:
    return PILOT.loc[PILOT["label"] == label] if not PILOT.empty else PILOT


#: The pilot label groups whose readings are read together.  A group with some labels committed
#: and others not fails, naming the missing labels; a group with none committed skips.
#: ``untargeted`` is the n = 8,000 label that plan X70 rev 3.8 added after the first eight.
PILOT_GROUPS = {
    "calibration": tuple(f"calibration_{fit}" for fit in study.FITS),
    "wrong learner": tuple(pilot.WRONG_LEARNER_LABELS),
    "untargeted": (pilot.UNTARGETED_LABEL,),
}


def _group(name: str) -> None:
    """Fail when only some labels of a pilot group are committed; skip when none is."""
    labels = PILOT_GROUPS[name]
    missing = [label for label in labels if _pilot(label).empty]
    if len(missing) == len(labels):
        pytest.skip(f"the {name} pilot has not been committed")
    if missing:
        pytest.fail(
            f"only part of the {name} pilot is committed; missing {missing}. Its readings are "
            "read together, so commit the rest before reading any"
        )


def _kept(label: str) -> pd.DataFrame:
    """One pilot label's rows without its held draws, which the declared run would lose.

    No draw may be refused, every requested draw is kept or held, and the held draws stay
    within :func:`held_allowance`.  The caller has checked the label's group.
    """
    rows = _pilot(label)
    assert not rows.empty, label
    assert rows["refusal"].fillna("").eq("").all(), label
    draws = pilot.PILOTS[label][1]
    assert rows["replicate"].nunique() == draws, label
    held = rows.loc[rows["held"] == 1, "replicate"]
    assert held.nunique() <= held_allowance(draws), (label, held.nunique())
    return rows.loc[rows["held"] == 0]


#: The pass probability each positive bias arm needs (design review d).
POSITIVE_PASS = 0.99
#: A wrong-learner reading that needs more replicates than this stops the declaration for a
#: diagnosis (plan X70 rev 2.7).
RULE_CAP = 4_000
#: Every positive arm whose verdict reads a bias interval, with its replicates.
POSITIVE_ARMS: dict[str, int] = {
    cell.cell: cell.replicates
    for cell in properties.declared_cells()
    if cell.role == "positive"
    and (
        cell.property in ("double_robustness", "targeting_necessity")
        or (cell.property == "root_n_and_efficiency" and cell.n != min(properties.RATE_SIZES))
    )
}


def _tolerated_bias(replicates: int) -> float:
    """The largest |standardized bias| at which a positive arm passes with ``POSITIVE_PASS``."""
    z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    return MARGINS.standardized_bias - (z + float(norm.ppf(POSITIVE_PASS))) / np.sqrt(replicates)


def _pilot_rule(errors: np.ndarray) -> tuple[float, int]:
    """The rule of plan X70 rev 2.7 on one arm's pilot errors: ``(reading, replicates)``.

    The reading is the |standardized bias| plus one of its standard errors; the replicates are
    the larger of :data:`~tests.studies.hazard_time_varying_multinode_properties.
    DOUBLE_ROBUST_REPLICATES` and the smallest multiple of 400 that passes with
    :data:`POSITIVE_PASS` at the reading.  A reading no count passes returns one step past
    :data:`RULE_CAP`.
    """
    standardized = abs(errors.mean()) / errors.std(ddof=1) + 1.0 / np.sqrt(errors.size)
    needed = int(properties.DOUBLE_ROBUST_REPLICATES)
    while sizing.bias_equivalent_probability(MARGINS, standardized, needed) < POSITIVE_PASS:
        needed += 400
        if needed > RULE_CAP:
            break
    return float(standardized), needed


def _wrong_learner_errors(
    configuration: str, label: str, column: str = "wrong_targeted"
) -> np.ndarray:
    """One wrong-learner configuration's pilot errors on one cell label at n = 2,000."""
    _group("wrong learner")
    rows = _kept(properties.WRONG_LEARNER_ARMS[configuration])
    values = rows.loc[rows["estimand"] == properties.ESTIMAND[label], column]
    return values.to_numpy(dtype=float) - properties.TRUTHS[label]


#: The pass probability the joint band's pilot reading needs at the declared count: plan X70
#: section 5 states 0.95 for (b2), above (b1)'s 0.90.
JOINT_PASS = 0.95


def band_pass_probability(coverage: float, replicates: int, level: float = 0.99) -> float:
    """The chance that the declared joint cell's exact interval lies inside the band.

    At true joint coverage ``coverage`` the cell's covered count over ``replicates`` is
    binomial, and the cell passes when the two-sided Clopper-Pearson interval at ``level`` lies
    inside :attr:`Margins.calibration_coverage`, the rule
    :func:`~tests.studies.evidence.property_verdicts.simultaneous_coverage_verdicts` reads.
    """
    from scipy.stats import beta as beta_law
    from scipy.stats import binom

    hits = np.arange(replicates + 1)
    tail = (1.0 - level) / 2.0
    with np.errstate(invalid="ignore"):
        low = np.where(hits > 0, beta_law.ppf(tail, hits, replicates - hits + 1), 0.0)
        high = np.where(
            hits < replicates, beta_law.ppf(1.0 - tail, hits + 1, replicates - hits), 1.0
        )
    lower, upper = MARGINS.calibration_coverage
    inside = (low >= lower) & (high <= upper)
    return float(binom.pmf(hits[inside], replicates, coverage).sum())


class TestTheWrongLearnerPilots:
    @pytest.mark.parametrize("arm", list(PLUG_IN_GAPS), ids=lambda arm: "-".join(arm))
    def test_the_pilot_plug_in_varies(self, arm: tuple[str, str]) -> None:
        """Each wrong-learner label's plug-in varies and centres on its exact-sample limit."""
        label, configuration = arm
        values = _wrong_learner_errors(configuration, label, "wrong_plug_in")
        assert np.var(values, ddof=1) > 0.0
        offset = PLUG_IN_GAPS[arm] * _scale(label, pilot.PILOT_N)
        error = np.std(values, ddof=1) / np.sqrt(values.size)
        assert abs(values.mean() - offset) < 3.0 * error + 1e-3 * _scale(label, pilot.PILOT_N)

    @pytest.mark.parametrize("label", list(pilot.WRONG_LEARNER_LABELS))
    def test_the_wrong_learner_fits_converge(self, label: str) -> None:
        """At most 1% of a wrong-learner label's kept draws fail to converge."""
        _group("wrong learner")
        rows = _kept(label).drop_duplicates("replicate")
        share = float((rows["converged"] == 0).mean())
        assert share <= properties.NONCONVERGED_SUSPECT_SHARE, (label, share)
        assert rows["unexpected_warnings"].sum() == 0, label

    def test_the_controls_are_discriminated_at_the_pilot(self) -> None:
        """The both-wrong control and the untargeted necessity control, read at the pilot.

        Each control's |standardized bias| minus one of its standard errors clears the margin
        at its configuration's replicates with the two-check floor.
        """
        controls = {
            "both_wrong": _wrong_learner_errors("both_wrong", CONTRAST),
            "mechanism_correct": _wrong_learner_errors(
                "mechanism_correct", CONTRAST, "wrong_plug_in"
            ),
        }
        for configuration, errors in controls.items():
            gap = abs(errors.mean()) / errors.std(ddof=1) - 1.0 / np.sqrt(errors.size)
            replicates = properties.configuration_replicates(configuration)
            probability = sizing.bias_discriminated_probability(MARGINS, gap, replicates)
            assert probability >= sizing.discrimination_floor(2), (configuration, gap)


# ---------------------------------------------------------------------------- sizes


class TestTheSizes:
    def test_the_power_cell_has_power(self) -> None:
        assert properties.POWER_REPLICATES == properties.CALIBRATION_REPLICATES
        power = design_power(
            properties.TRUTHS[CONTRAST], properties.EFFICIENCY_SD[CONTRAST], properties.POWER_N
        )
        assert power >= 0.80
        assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99

    def test_the_derived_controls_leave_the_band(self) -> None:
        replicates = properties.CALIBRATION_REPLICATES
        assert (
            sizing.calibration_control_probability(
                MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR
            )
            > 0.999
        )
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        noise = np.sqrt(2.0)
        spread = noise / np.sqrt(2.0 * (replicates - 1))
        upper = properties.EFFICIENCY_RATIO_BAND[1]
        assert norm.cdf((noise - z * spread - upper) / spread) > 0.999

    def test_the_untargeted_control_is_discriminated(self) -> None:
        replicates = properties.configuration_replicates(properties.NECESSITY_CONFIGURATION)
        for spread in (1.0, 1.5):
            probability = sizing.bias_discriminated_probability(
                MARGINS, PLUG_IN_GAPS[(CONTRAST, "mechanism_correct")] / spread, replicates
            )
            assert probability >= sizing.discrimination_floor(2)

    def test_every_positive_arm_passes_at_its_exact_limit(self) -> None:
        """Each positive bias arm passes with probability at least 0.99 at standardized bias 0."""
        for cell in properties.declared_cells():
            if cell.role != "positive" or cell.property not in (
                "double_robustness",
                "targeting_necessity",
            ):
                continue
            probability = sizing.bias_equivalent_probability(MARGINS, 0.0, cell.replicates)
            assert probability >= 0.99, (cell.cell, probability)

    def test_every_gated_primary_cell_has_a_thousand_draws(self) -> None:
        for cell in properties.declared_cells():
            if cell.property in ("double_robustness", "targeting_necessity", "power"):
                assert cell.replicates >= 1_000, cell.cell
        assert study.PRIMARY_REPLICATES >= 1_000

    @staticmethod
    def _calibration_rows(fit_label: str) -> pd.DataFrame:
        """One calibration label's kept rows; fails while only some labels are committed."""
        _group("calibration")
        assert pilot.PILOTS[f"calibration_{fit_label}"][1] == sizing.CALIBRATION_PILOT_DRAWS
        return _kept(f"calibration_{fit_label}")

    def _calibration_ratios(self) -> dict[str, float]:
        """Each calibration arm's pilot mean SE over the SD of its estimates."""
        ratios = {}
        for label in properties.CALIBRATION_LABELS:
            rows = self._calibration_rows(properties.FIT_OF[label])
            arm = rows.loc[rows["estimand"] == properties.ESTIMAND[label]]
            assert len(arm) == rows["replicate"].nunique()
            ratios[label] = float(arm["std_error"].mean() / arm["estimate"].std(ddof=1))
        return ratios

    def test_the_calibration_screen_reads_the_expected_ratio(self) -> None:
        """The defect screen, centred on each arm's expected SE ratio (plan X70 rev 3.8 addendum).

        A heavy-tailed influence curve reads :math:`E \\hat\\sigma` below :math:`\\sigma`:
        :func:`~tests.studies.hazard_time_varying_multinode_properties.expected_se_ratio`, an
        exact simulation of the law with no fit, gives 0.978 for the policy arms
        (:math:`\\kappa = 599`) and 0.987 to 0.994 for the rest.  The rest of the policy arms'
        pilot deficit is the estimates' SD above the bound, a second-order excess the screen
        counts as departure.  Each arm's pilot ratio must lie within the screen's half-width of
        its centre, and the arms' mean departure within the off-mean bound, as the pooled rule
        asks of a departure from one.
        """
        ratios = self._calibration_ratios()
        departures = {
            label: ratio - properties.expected_se_ratio(label, properties.CALIBRATION_N)
            for label, ratio in ratios.items()
        }
        half = mtp_study_tools.screen_half_width(sizing.CALIBRATION_PILOT_DRAWS)
        assert all(abs(value) <= half for value in departures.values()), departures
        assert abs(np.mean(list(departures.values()))) < mtp_study_tools.OFF_MEAN, departures

    def test_the_expected_ratio_is_simulated_exactly(self) -> None:
        """The simulated centre against its first-order cross-check and the kurtosis.

        The first-order formula overstates the deficit, most where the curve is most
        heavy-tailed: by 0.015 on the policy arms and by 0.003 or less on the rest.
        """
        assert properties.EIF_KURTOSIS["multinode_policy"] == pytest.approx(599.35, abs=0.01)
        n = properties.CALIBRATION_N
        for label in properties.CALIBRATION_LABELS:
            exact = properties.expected_se_ratio(label, n)
            first = properties.predicted_se_ratio(label, n)
            assert first < exact < 1.0, label
            limit = 0.02 if properties.EIF_KURTOSIS[label] > 500 else 0.004
            assert exact - first < limit, (label, exact, first)
        assert properties.expected_se_ratio("multinode_policy", n) == pytest.approx(
            0.978, abs=0.003
        )
        assert properties.expected_se_ratio(
            "multinode_contrast", n
        ) == properties.expected_se_ratio("crossfit_contrast", n)

    def test_the_calibration_rms_ratio_is_one(self) -> None:
        """Root mean square SE over the estimates' SD, within the screen's half-width of one.

        The rule, declared before this reading was written (plan X70 rev 3.8 addendum): it
        reads :math:`E[SE^2] = \\mathrm{Var}\\,\\hat\\psi` with no approximation and no
        kurtosis, so it is centred on one.  Its tolerance is the screen's,
        :math:`2.576\\,\\sigma_P + 0.01` with :math:`\\sigma_P` the model's SE of an 8,000-draw
        ratio (0.0085), which bounds the bootstrap Monte Carlo SE of this ratio on every arm of
        the committed pilot (0.0078 to 0.0083); the 0.01 allows the second-order excess of the
        estimates' SD over the bound.
        """
        half = mtp_study_tools.screen_half_width(sizing.CALIBRATION_PILOT_DRAWS)
        for label in properties.CALIBRATION_LABELS:
            rows = self._calibration_rows(properties.FIT_OF[label])
            arm = rows.loc[rows["estimand"] == properties.ESTIMAND[label]]
            se = arm["std_error"].to_numpy(dtype=float)
            ratio = float(np.sqrt(np.mean(se**2)) / arm["estimate"].std(ddof=1))
            assert abs(ratio - 1.0) <= half, (label, ratio)

    def test_a_red_calibration_row_is_read_against_its_expected_ratio(self) -> None:
        """The finite-sample route of a red calibration row, declared before the run.

        Its 99% SE-ratio interval must contain the arm's expected ratio, not 1
        (``finite_sample_red``, named in ``regenerate.py``): an interval about the policy
        arms' expected 0.978 that excludes 1 takes the route, and one that contains 1 but not
        0.978 does not.
        """
        policy = properties.expected_se_ratio("multinode_policy", properties.CALIBRATION_N)
        assert properties.finite_sample_red("multinode_policy", policy - 0.016, policy + 0.016)
        assert policy + 0.016 < 1.0
        assert not properties.finite_sample_red("multinode_policy", policy + 0.005, 1.01)
        text = (ROOT / "tests/canonical/hazard_time_varying_multinode/regenerate.py").read_text(
            encoding="utf-8"
        )
        assert "finite_sample_red" in text
        assert "contains 1," not in text

    def test_the_calibration_replicates_follow_the_pilot(self) -> None:
        """The per-arm rule at the pilot's own ratios (plan X70 rev 3.8): each arm is sized at
        its reading, not pooled toward one."""
        ratios = self._calibration_ratios()
        reading = mtp_study_tools.per_arm_replicates(
            list(ratios.values()), sizing.CALIBRATION_PILOT_DRAWS
        )
        assert reading["R"] == properties.CALIBRATION_REPLICATES
        assert min(reading["per_cell"]) >= mtp_study_tools.CELL_TARGET
        assert reading["family_wise"] >= mtp_study_tools.FAMILY_TARGET

    @pytest.mark.parametrize(
        "configuration",
        [c for c in properties.WRONG_LEARNER_ARMS if c != "both_wrong"],
    )
    def test_the_wrong_learner_arms_are_sized_by_the_pilot_rule(self, configuration: str) -> None:
        """Each configuration with a wrong learner takes the pre-committed size (plan X70 rev
        2.7): the largest need among its targeted positive arms, by :func:`_pilot_rule` at its
        800-draw pilot, with no cap; a need above 4,000 stops the declaration for a diagnosis.
        The targeted necessity arm reads the ``mechanism_correct`` contrast fits."""
        needs = {}
        for label in properties.CONFIGURATION_ARMS[configuration]:
            reading, needed = _pilot_rule(_wrong_learner_errors(configuration, label))
            assert needed <= RULE_CAP, (label, reading, needed)
            needs[label] = needed
        assert properties.WRONG_LEARNER_REPLICATES[configuration] == max(needs.values()), needs
        for label in properties.CONFIGURATION_ARMS[configuration]:
            assert POSITIVE_ARMS[f"{label}__{configuration}"] == max(needs.values())
        if configuration == properties.NECESSITY_CONFIGURATION:
            assert POSITIVE_ARMS[f"{CONTRAST}__targeted"] == max(needs.values())

    def test_the_control_configuration_keeps_the_floor(self) -> None:
        assert properties.WRONG_LEARNER_REPLICATES["both_wrong"] == (
            properties.DOUBLE_ROBUST_REPLICATES
        )

    def test_the_untargeted_arms_are_sized_by_their_pilot(self) -> None:
        """The untargeted ``outcome_correct`` arms at n = 8,000, by :func:`_pilot_rule` at the
        ``untargeted_outcome_correct_n8000`` pilot (plan X70 rev 3.8)."""
        _group("untargeted")
        rows = _kept(pilot.UNTARGETED_LABEL)
        assert (rows["n"] == properties.UNTARGETED_N).all()
        needs = {}
        for label, configuration in properties.UNTARGETED_ARMS:
            assert configuration == "outcome_correct"
            arm = rows.loc[rows["estimand"] == properties.ESTIMAND[label]]
            errors = arm["wrong_plug_in"].to_numpy(dtype=float) - properties.TRUTHS[label]
            reading, needed = _pilot_rule(errors)
            assert needed <= RULE_CAP, (label, reading, needed)
            needs[label] = needed
        assert max(needs.values()) == properties.UNTARGETED_REPLICATES, needs
        for label, configuration in properties.UNTARGETED_ARMS:
            cell = f"{label}__{configuration}{properties.UNTARGETED_SUFFIX}"
            assert POSITIVE_ARMS[cell] == properties.UNTARGETED_REPLICATES

    def test_the_positive_arms_follow_the_calibration_pilot(self) -> None:
        """The ``both_correct`` arms and the claimed rungs pass at the calibration pilot's
        reading of their cell label."""
        rows = self._calibration_rows("multinode")
        for cell_label, name in properties.COEFFICIENTS["multinode"]:
            arm = rows.loc[rows["estimand"] == name]
            error = (arm["estimate"] - arm["truth"]).to_numpy(dtype=float)
            standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
            for cell, replicates in POSITIVE_ARMS.items():
                if cell.startswith(f"{cell_label}__") and cell.endswith(
                    ("__both_correct", "__n_2000", "__n_8000")
                ):
                    probability = sizing.bias_equivalent_probability(
                        MARGINS, standardized, replicates
                    )
                    assert probability >= POSITIVE_PASS, (cell, probability)

    def test_the_band_pass_probability(self) -> None:
        """The joint-band rule passes a calibrated band and stops at a band at its edge."""
        replicates = properties.CALIBRATION_REPLICATES
        assert band_pass_probability(0.95, replicates) > 0.999
        assert band_pass_probability(0.915, replicates) < 0.01
        assert band_pass_probability(0.925, replicates) < JOINT_PASS

    @pytest.mark.parametrize("joint", study.JOINT_LABELS)
    def test_the_joint_band_pilot_meets_the_band(self, joint: str) -> None:
        """The declared joint cell passes with probability :data:`JOINT_PASS` at the pilot's
        coverage +/- 1 SE.

        The ``multinode`` calibration pilot fits with ``simultaneous=True``, as the declared
        calibration fits do, so it reads the band the ``simultaneous_coverage`` cells gate, over
        all 21 estimates.  The rule was fixed before the pilot was read.  A failure stops the
        declaration for a diagnosis: the band is not widened, and the count is not raised after
        the reading.
        """
        rows = self._calibration_rows(properties.JOINT_FIT[joint])
        band = rows.loc[rows["estimand"] == "joint_band", "covered"].to_numpy(dtype=float)
        pointwise = rows.loc[rows["estimand"] == "joint_pointwise", "covered"]
        assert band.size == rows["replicate"].nunique() == pointwise.size
        coverage = float(band.mean())
        error = float(np.sqrt(coverage * (1.0 - coverage) / band.size))
        replicates = properties.CALIBRATION_REPLICATES
        passes = {
            edge: band_pass_probability(edge, replicates)
            for edge in (coverage - error, coverage + error)
        }
        if min(passes.values()) < JOINT_PASS:
            pytest.fail(
                f"{joint}: at the pilot's joint coverage {coverage:.4f} +/- {error:.4f} the "
                f"declared cell at {replicates} replicates passes with probability "
                f"{min(passes.values()):.3f}, below {JOINT_PASS}; the declaration stops for a "
                "diagnosis, and the band is not widened"
            )

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the ``multinode``
        calibration pilot."""
        covered: dict[str, int] = {}
        groups = []
        for scenario in study.SCENARIOS:
            rows = self._calibration_rows(scenario)
            rows = rows.loc[~rows["estimand"].isin(pilot.JOINT_ROWS)]
            assert int(rows["n"].iloc[0]) == study.PRIMARY_N
            names = list(study.ESTIMANDS[scenario])
            hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
            covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
            error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
                index="replicate", columns="estimand", values="error"
            )[names]
            standardized = error / error.std(ddof=1)
            groups.append(
                sizing.PilotGroup(
                    standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
                )
            )
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                sizing.CALIBRATION_PILOT_DRAWS,
                implementations=1,
                coverage_pass=share,
                bias_groups=groups,
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )

    @pytest.mark.parametrize("label", [*pilot.PILOTS])
    def test_every_pilot_label_converged(self, label: str) -> None:
        """Every kept fit of a committed label converged with no unexpected warning."""
        if _pilot(label).empty:
            pytest.skip(f"{label} has not been committed")
        rows = _pilot(label)
        rows = rows.loc[rows["held"] == 0]
        assert (rows["converged"] == 1).all(), label
        assert rows["unexpected_warnings"].sum() == 0, label


SMOKE = (
    json.loads(study.SMOKE_OUTPUT.read_text(encoding="utf-8"))
    if study.SMOKE_OUTPUT.exists()
    else None
)


class TestTheSmokeGate:
    def test_the_cost_table(self) -> None:
        """One cost per smoke cell (plan X70 rev 3.7): ``mechanism_correct`` at 2.5 s, the other
        configurations at 1.3 to 1.5 s, the cross-fitted label at 1.8 s."""
        assert set(study.COST_SECONDS) == set(study.SMOKE_CELLS)
        assert study.COST_SECONDS["multinode/mechanism_correct"] == 2.5
        assert study.COST_FACTOR == 2.0
        kinds = {kind for kind, *_ in study.SMOKE_CELLS.values()}
        assert kinds == {"hazard", "untargeted", "iterated", "isotonic"}
        sizes = {n for _, _, _, n in study.SMOKE_CELLS.values()}
        assert sizes == set(study.RATE_SIZES)
        configurations = {c for kind, _, c, _ in study.SMOKE_CELLS.values() if kind == "hazard"}
        assert configurations == set(study.CONFIGURATIONS)

    def test_each_smoke_cell_is_a_declared_fit(self) -> None:
        """Each smoke cell's ``(kind, fit label, configuration, n)`` is a fit the run makes, and
        every kind of fit the run makes has a smoke cell."""
        declared = set()
        for family, stream, configuration, n, _, fit_label in properties.FIT_SETS:
            if family == properties.DIFFERENCE_FAMILY:
                declared |= {
                    (kind, fit_label, configuration, n) for kind in ("iterated", "isotonic")
                }
            elif stream == properties.UNTARGETED_STREAM:
                declared.add(("untargeted", fit_label, configuration, n))
            else:
                declared.add(("hazard", fit_label, configuration, n))
                if family == "double_robustness" and configuration == (
                    properties.NECESSITY_CONFIGURATION
                ):
                    declared.add(("untargeted", fit_label, configuration, n))
        cells = set(study.SMOKE_CELLS.values())
        assert cells <= declared, cells - declared
        assert {cell[:3] for cell in cells} == {cell[:3] for cell in declared}
        assert study.SMOKE_CELLS["untargeted/outcome_correct"][3] == properties.UNTARGETED_N

    def test_the_committed_smoke_cleared(self) -> None:
        """``smoke.json``, written by ``python -m
        tests.studies.canonical_hazard_time_varying_multinode``, holds every cell at the
        declared replicates, and every fit met the gate: converged, every node score at most
        ``tol``, no held cell, no failure, and at most twice its cell's cost."""
        assert SMOKE is not None, "the smoke has not been committed"
        assert SMOKE["replicates"] == study.SMOKE_REPLICATES
        assert set(SMOKE["cells"]) == set(study.SMOKE_CELLS)
        for name, cell in SMOKE["cells"].items():
            assert len(cell["runs"]) == study.SMOKE_REPLICATES, name
            for run in cell["runs"]:
                assert study.smoke_cleared(name, run), (name, run)
                assert run["seconds"] <= study.COST_FACTOR * study.COST_SECONDS[name], name
        assert SMOKE["cleared"] is True

    @pytest.mark.parametrize("seconds", [1.0, 2.7])
    def test_a_slow_fit_fails_the_gate(
        self, seconds: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def cell(name: str, replicate: int) -> dict[str, Any]:
            del name, replicate
            return {
                "held": False,
                "failure": "",
                "seconds": seconds,
                "converged": True,
                "largest_node_score": 0.0,
                "walk_steps": 1,
                "unexpected_warnings": [],
            }

        monkeypatch.setattr(study, "_smoke_cell", cell)
        name = "multinode/both_correct"
        record = study.SMOKE_GATE(replicates=2, cells=(name,), output=None)
        assert record["cleared"] is (seconds <= study.COST_FACTOR * study.COST_SECONDS[name])

    def test_a_stalled_walk_fails_the_gate(self) -> None:
        run = {"held": False, "failure": "", "seconds": 0.1, "converged": True}
        assert study.smoke_cleared("multinode/both_correct", {**run, "largest_node_score": 0.0})
        assert not study.smoke_cleared(
            "multinode/both_correct", {**run, "largest_node_score": 10 * study.TOL}
        )
        assert not study.smoke_cleared("multinode/both_correct", {**run, "converged": False})


# ---------------------------------------------------------------------------- cells and seeds


def _pilot_seeds() -> set[int]:
    """Every module-level ``PILOT_SEED`` under ``tests/``."""
    pattern = re.compile(r"^PILOT_SEED(?:\s*:\s*int)?\s*=\s*([0-9_]+)\s*$", re.MULTILINE)
    seeds = set()
    for path in Path(ROOT, "tests").rglob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            seeds.add(int(match.group(1).replace("_", "")))
    return seeds


class TestTheCells:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_declares_every_cell_the_module_writes(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
        }
        written = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
        filled = {
            name
            for key in properties._payloads(properties.FIT_SETS, 1)
            for name in properties.cells_of(key)
        }
        rate = {name for name in declared if name.startswith("root_n_rate/")}
        assert written | rate == declared
        assert filled | rate == declared

    def test_the_study_is_registered(self) -> None:
        assert study.STUDY in registered()
        assert study.STUDY.publication_policy == "gated"

    def test_the_seeds_are_distinct(self) -> None:
        from tests.studies import canonical_hazard_multinode, canonical_hazard_time_varying

        records = {record.slug: record for record in registered()}
        for sibling in (canonical_hazard_time_varying.STUDY, canonical_hazard_multinode.STUDY):
            records.setdefault(sibling.slug, sibling)
        others = [record for slug, record in records.items() if slug != study.STUDY.slug]
        taken = {record.seed for record in others} | {record.resampling_seed for record in others}
        assert study.SEED not in taken | _pilot_seeds()
        assert study.RESAMPLING_SEED not in taken | _pilot_seeds()
        seeds = [replicate_seed(study.STUDY, "multinode", r) for r in range(50)]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, stream, r)
            for family, stream, *_ in properties.FIT_SETS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", "ic_gap", fit, n, r)
            for fit, n, r in pilot.IC_GAP_JOBS
        ]
        seeds.append(properties.EXPECTED_RATIO_SEED)
        assert properties.EXPECTED_RATIO_SEED not in taken | _pilot_seeds()
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)


# ---------------------------------------------------------------------------- the IC gap


def _ic_gaps(fit_label: str) -> dict[str, tuple[list[float], float]]:
    rows = _pilot("ic_gap")
    if rows.empty:
        pytest.skip("the ic_gap pilot has not been committed")
    assert rows["refusal"].fillna("").eq("").all()
    assert (rows["held"] == 0).all(), "an ic_gap fit held a cell; rerun the label"
    small_n, large_n = properties.IC_GAP_SIZES
    rows = rows.loc[rows["fit_label"] == fit_label]
    out: dict[str, tuple[list[float], float]] = {}
    for label, _ in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == label]
        small = arm.loc[arm["n"] == small_n, "estimate"].to_numpy(dtype=float)
        large = arm.loc[arm["n"] == large_n, "estimate"].to_numpy(dtype=float)
        assert small.size == properties.IC_GAP_DRAWS and large.size == 1, (fit_label, label)
        out[label] = (list(small), float(large[0]))
    return out


def _fit_key(fit_label: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The support key of one fit label's fit, read where the estimator classifies it."""
    from cleverly import _support
    from cleverly.longitudinal import estimator as estimator_module

    keys: list[Any] = []
    original = estimator_module.classify_longitudinal_fit

    def spy(estimator: Any, data: Any, **settings: Any) -> Any:
        names = (
            "outcome_kind",
            "cross_fitted",
            "known_factors",
            "known_nodes",
            "has_msm",
            "regimen_kinds",
            "tuned_cap",
            "separable_first",
        )
        key, _ = _support.longitudinal_key(
            estimator, data, **{name: settings[name] for name in names if name in settings}
        )
        keys.append(key)
        return original(estimator, data, **settings)

    monkeypatch.setattr(estimator_module, "classify_longitudinal_fit", spy)
    result = _quiet(study.fit, fit_label, law.LAW.draw(600, 1))
    return keys[-1], result


class TestTheSupportRows:
    @pytest.mark.parametrize("row", study.support_rows(), ids=lambda row: row.id)
    def test_each_row_covers_its_fit_and_licenses_it(
        self, row: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key, result = _fit_key(study.G63_MULTINODE_FITS[row.id], monkeypatch)
        assert key.is_covered_by(row.key), key.differing_axes(row.key)
        assert dict(row.key.options)["later_covariates"] is True
        assert row.evidence == (study.STUDY.slug,)
        assert row.cells == study.G63_MULTINODE_CELLS[row.id]
        assert set(row.cells) <= {*study.CALIBRATION_LABELS, *study.JOINT_LABELS}
        supports = {(e.support.label, e.support.row) for e in result.estimates.values()}
        assert supports == {("supported", row.id)}

    def test_no_other_row_covers_these_fits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``later_covariates`` option keeps every fit off the baseline multi-node rows.

        Without it the fits differ from ``G62.three_node`` in the targeting algorithm alone (the
        ``G62`` regimen rows name ``targeting="iterative"``), and from ``G63.held`` in the
        treatment axis, so no other row covers either fit, marked or not.
        """
        from dataclasses import replace

        for row_id, fit_label in study.G63_MULTINODE_FITS.items():
            key, _ = _fit_key(fit_label, monkeypatch)
            covering = {row.id for row in LONGITUDINAL_ROWS if key.is_covered_by(row.key)}
            assert covering == {row_id}
            options = tuple(
                (name, False if name == "later_covariates" else value)
                for name, value in key.options
            )
            unmarked = replace(key, options=options)
            assert not any(unmarked.is_covered_by(row.key) for row in study.support_rows())

    def test_the_rows_say_what_the_key_cannot(self) -> None:
        """The banded row licenses the gated band, and every row limits its covariate law."""
        from cleverly._support_rows import _LONG_BAND, _LONG_WALD, BAND_CELLS

        for row in study.support_rows():
            kinds = _LONG_BAND if row.id in study.G63_MULTINODE_BANDED else _LONG_WALD
            assert frozenset(row.kinds) == kinds, row.id
            assert (row.id in BAND_CELLS) == (row.id in study.G63_MULTINODE_BANDED), row.id
            text = " ".join(row.limits)
            assert "binary" in text and "Markov" in text and "Regimens:" in text, row.id
        banded = {study.G63_MULTINODE_FITS[row_id] for row_id in study.G63_MULTINODE_BANDED}
        assert {label.removesuffix("_curve") for label in study.JOINT_LABELS} == banded


class TestTheInfluenceCurveGap:
    @pytest.mark.parametrize("fit_label", list(study.FITS))
    def test_a_fit_meets_the_declared_bound(self, fit_label: str) -> None:
        small_n, large_n = properties.IC_GAP_SIZES
        for label, (small, large) in _ic_gaps(fit_label).items():
            median = float(np.median(small))
            assert median <= properties.IC_GAP_BOUND[small_n], (label, small)
            assert large <= properties.IC_GAP_BOUND[large_n], (label, large)
            assert large <= properties.IC_GAP_FALL * median, (label, median, large)

    @pytest.mark.parametrize("fit_label", list(study.FITS))
    def test_a_large_fit_reports_the_bound(self, fit_label: str) -> None:
        """At n = 40,000 each fit's influence-curve SD over the bound lies within three of its
        sampling SDs of one (plan X70 rev 3.8): ``large_fit_tolerance``, from the arm's exact
        kurtosis, 0.18 for the policy arms and 0.09 to 0.12 for the rest."""
        rows = _pilot("ic_gap")
        if rows.empty:
            pytest.skip("the ic_gap pilot has not been committed")
        rows = rows.loc[(rows["fit_label"] == fit_label) & (rows["n"] == pilot.BOUND_N)]
        assert len(rows) == len(properties.COEFFICIENTS[fit_label])
        for label, ratio in zip(rows["estimand"], rows["estimate"], strict=True):
            tolerance = properties.large_fit_tolerance(label, pilot.BOUND_N)
            assert ratio == pytest.approx(1.0, abs=tolerance), (label, tolerance)


# ---------------------------------------------------------------------------- the agreement


def _paired_ratios(
    rows: pd.DataFrame, fit_label: str, column: str = "iterated_estimate"
) -> dict[str, float]:
    ratios = {}
    for label, name in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == name]
        difference = arm["estimate"] - arm[column]
        ratios[label] = float(difference.std(ddof=1)) / _scale(label, pilot.PILOT_N)
    return ratios


@pytest.mark.parametrize("fit_label", study.DIFFERENCE_FITS)
def test_the_agreement_pilot_is_reported(fit_label: str) -> None:
    """The ``iterated_mean_difference`` readings, reported with no floor."""
    if _pilot(f"fits_differ_{fit_label}").empty:
        pytest.skip("the pilot has not been committed")
    rows = _kept(f"fits_differ_{fit_label}")
    decreases = rows.loc[rows["estimand"] == "largest_decrease", "estimate"]
    ratios = _paired_ratios(rows, fit_label)
    ratios |= {
        f"{label} (isotonic)": value
        for label, value in _paired_ratios(rows, fit_label, "isotonic_estimate").items()
    }
    reading = properties.fits_differ_reading(ratios, int((decreases > 0.0).sum()), len(decreases))
    assert all(np.isfinite(value) and value > 0.0 for value in reading["ratios"].values())
    print(reading["text"])


def test_a_partial_pilot_group_fails_and_names_the_missing_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with some labels committed fails, naming the rest; with none it skips."""
    module = __import__(__name__, fromlist=["PILOT"])
    partial = pd.DataFrame({"label": ["calibration_multinode", "wrong_learner_outcome_correct"]})
    monkeypatch.setattr(module, "PILOT", partial)
    for name, missing in (
        ("calibration", "calibration_crossfit"),
        ("wrong learner", "wrong_learner_mechanism_correct"),
    ):
        with pytest.raises(pytest.fail.Exception, match=missing):
            _group(name)
    with pytest.raises(pytest.skip.Exception):
        _group("untargeted")
    monkeypatch.setattr(module, "PILOT", pd.DataFrame({"label": ["ic_gap"]}))
    with pytest.raises(pytest.skip.Exception):
        _group("calibration")
