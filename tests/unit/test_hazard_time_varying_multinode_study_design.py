"""The declared design of ``hazard-time-varying-multinode`` (plan X70 section 5, (b2)).

Every check here reads law (M) of :mod:`tests.studies.hazard_time_varying_multinode_law` and the
declared constants, before any pilot or run: the truths against an independent forward sum, the
exact-sample fits, positivity, the efficiency bounds, every robustness limit and every witness
of a component that could vanish at the truth, the sizes that need no pilot, the record and the
seeds.  The readings that need a committed pilot (the calibration and wrong-learner counts, the
band, the influence-curve gap) are not written yet: the pilot has not run, and a test that
waits for it would skip.  They are added with the pilot commit, as in
``tests/unit/test_hazard_time_varying_study_design.py``.

Each gap below is in efficiency SEs at the arm's size, :math:`n = 2{,}000`: the exact limit
minus the truth over :math:`\\sigma / \\sqrt{n}`, with :math:`\\sigma` the SD of the efficient
influence function.  An exact limit is the fit of the law as data, every support point once and
weighted by its mass.
"""

from __future__ import annotations

import itertools
import re
import warnings
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
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
    ("multinode_policy", "mechanism_correct"): 5.551,
    ("multinode_contrast", "continuation_wrong"): 0.796,
    ("multinode_policy", "continuation_wrong"): 0.383,
}
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
#: The exact limit of each mutation under ``outcome_correct`` (a correct mechanism repairs a
#: slot defect), by ``(mutation, arm)``.  Plan X70 asks 3 SEs of the rule and policy arms, and
#: when no coefficient inside the positivity and hazard limits reaches it, the largest gap
#: reached, at least 1 SE.  The coefficients were searched over ``h_P``, ``h_A``, ``b_L`` and
#: ``c_L`` within those limits before any pilot; these are the gaps at the declared ones.
WITNESS_GAPS: dict[tuple[str, str], float] = {
    ("level_plan", "multinode_rule"): 0.112,
    ("level_plan", "multinode_policy"): 1.626,
    ("held_slot", "multinode_rule"): -0.190,
    ("held_slot", "multinode_policy"): -0.496,
    ("policy_prefix", "multinode_policy"): 0.150,
    ("observed_slot", "multinode_policy"): 2.157,
    ("observed_slot", POLICY_RISK): 3.073,
}
#: The witnesses below the plan's 1 SE fallback at the declared coefficients.  The rule's
#: followers hold ``A_{k-1} != A_k`` only when ``L`` changes between nodes, which the Markov
#: covariate makes rare (at most 0.35 SE at any coefficient searched), and the policy reads
#: ``A_{k-1}`` through the plan's fixed 0.2 term (at most 0.4 SE).  The policy's held slot
#: reached 1.3 SEs only at a baseline hazard that held a cell in 7% of the ladder's n = 1,000
#: fits.  The 4.4 mutations of ``tests/unit/test_hazard_time_varying_multinode.py`` carry
#: these components; the declaration states it before any run.
BELOW_FALLBACK = frozenset(
    {
        ("level_plan", "multinode_rule"),
        ("held_slot", "multinode_rule"),
        ("held_slot", "multinode_policy"),
        ("policy_prefix", "multinode_policy"),
    }
)


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int = N) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _policy_risk() -> law.Target:
    return study.target_of(study.risk_name("policy", study.LAST))


def _gap(result: Any, arm: str) -> float:
    """The exact-limit gap of one arm (a cell label or :data:`POLICY_RISK`) in SEs."""
    if arm == POLICY_RISK:
        aim = _policy_risk()
        estimate = float(result[study.risk_name("policy", study.LAST)].psi)
        return (estimate - aim.value(law.LAW)) / (law.efficiency_sd(law.LAW, aim) / np.sqrt(N))
    name = properties.ESTIMAND[arm]
    return (float(result[name].psi) - properties.TRUTHS[arm]) / _scale(arm)


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
def mutated_fits() -> dict[str, Any]:
    """The ``outcome_correct`` exact-sample fit under each slot mutation."""
    out = {}
    for name, mutation in MUTATIONS.items():
        with mutation():
            out[name] = _quiet(study.exact_fit, "outcome_correct")
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
            [-0.727560, -0.319157, -0.322477, 0.128960],
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
        (f"pilot/{label}", pilot.PILOT_N, draws)
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
            [1.04046, 1.01905, 0.66815, 0.74827],
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
    def test_the_slot_witnesses(self, mutated_fits: dict[str, Any]) -> None:
        """Each slot mutation's exact limit under ``outcome_correct``, against the plan's floor.

        The policy-mixture witness (the observed slot at nodes 2 and 3, on the policy's risk)
        meets 3 SEs; on the policy contrast it and the level plan meet the 1 SE fallback.
        :data:`BELOW_FALLBACK` names the four witnesses below 1 SE; each component keeps its
        4.4 mutation.
        """
        for (mutation, arm), recorded in WITNESS_GAPS.items():
            gap = _gap(mutated_fits[mutation], arm)
            assert gap == pytest.approx(recorded, abs=1e-3), (mutation, arm)
        assert abs(WITNESS_GAPS[("observed_slot", POLICY_RISK)]) >= 3.0
        below = {key for key, gap in WITNESS_GAPS.items() if abs(gap) < 1.0}
        assert below == BELOW_FALLBACK
        policy = [gap for (_, arm), gap in WITNESS_GAPS.items() if arm == "multinode_policy"]
        assert max(abs(gap) for gap in policy) >= 1.0

    def test_a_static_arm_holds_one_slot(self, mutated_fits: dict[str, Any]) -> None:
        """The observed-slot and prefix mutations touch policy nodes only: a static arm or the
        rule holds one slot, so their limits are the truth under the mutation."""
        for mutation in ("observed_slot", "policy_prefix"):
            for arm in ("multinode_contrast", "multinode_rule", "multinode_risk"):
                assert _gap(mutated_fits[mutation], arm) == pytest.approx(0.0, abs=1e-5)

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

    def test_the_calibration_replicates_at_a_calibrated_pilot(self) -> None:
        """The provisional count: the arm-pooled rule at an 8,000-draw pilot reading a ratio of
        one on the six arms.  The committed pilot replaces it."""
        assert len(properties.CALIBRATION_LABELS) == 6
        replicates, cell, family = mtp_study_tools.smallest_replicates(
            1.0, sizing.CALIBRATION_PILOT_DRAWS, len(properties.CALIBRATION_LABELS)
        )
        assert replicates == properties.CALIBRATION_REPLICATES
        assert cell >= mtp_study_tools.CELL_TARGET and family >= mtp_study_tools.FAMILY_TARGET
        assert all(
            draws == sizing.CALIBRATION_PILOT_DRAWS
            for label, (_, draws) in pilot.PILOTS.items()
            if label.startswith("calibration_")
        )

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


class TestTheSmokeGate:
    def test_the_cost_table(self) -> None:
        """Plan X70's (b2) cost inputs: 1.3 s in sample and 1.8 s cross-fitted at n = 2,000."""
        assert study.COST_SECONDS == {"multinode": 1.3, "crossfit": 1.8}
        assert study.COST_FACTOR == 2.0
        assert set(study.COST_SECONDS) == set(study.FITS)

    @pytest.mark.parametrize("seconds", [1.0, 2.7])
    def test_a_slow_fit_fails_the_gate(
        self, seconds: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def cell(fit_label: str, configuration: str, replicate: int) -> dict[str, Any]:
            del fit_label, configuration, replicate
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
        record = study.SMOKE_GATE(replicates=2, cells=(("multinode", "both_correct"),), output=None)
        assert record["cleared"] is (seconds <= study.COST_FACTOR * study.COST_SECONDS["multinode"])


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

    def test_the_study_waits_for_its_run(self) -> None:
        """The record registers with its passing declared run, and its rows land with it."""
        assert study.STUDY not in registered()
        assert study.STUDY.publication_policy == "gated"
        rows = {row.id for row in LONGITUDINAL_ROWS}
        assert "G63.held.in_sample" in rows
        assert not rows & set(study.G63_MULTINODE_CELLS)
        licensed = {label for labels in study.G63_MULTINODE_CELLS.values() for label in labels}
        assert licensed <= {*study.CALIBRATION_LABELS, *study.JOINT_LABELS}

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
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)
