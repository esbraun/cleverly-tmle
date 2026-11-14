r"""Compositions of the value of the optimal rule (X11 (f3), plan rev 2, PR 2).

``estimands=["ey_optimal_rule"]`` (and ``OptimalRuleValue``) takes known or estimated
weights, baseline strata, known regimes beside it (``interventions=``, ``reference=``), the
in-sample fit and an intermediate held at a level.  The estimator is the learned-rule fit with
the same setting, and only the target changes:

- weights: the optimal value of the population the weights describe,
  :math:`\Psi^w_{d_0^w}(P_w) = E_{P_w}\bar Q_w(d_0^w(W), W)` with
  :math:`d_0^w = \arg\max_a \bar Q_w(a, \cdot)`;
- strata: :math:`E_0[\bar Q_0(d_0(W), W) \mid S = s]` per stratum, and the marginal
  :math:`\Psi_{d_0}`;
- known regimes: :math:`\Psi_{d_0} - \Psi_{d_k}`, reported as ``ate_optimal_rule[...]``;
- in sample: :math:`\Psi_{d_0}` from the in-sample TMLE of the full-sample rule (JCI Thm 5);
- an intermediate at level :math:`z`: :math:`\Psi^z_{d_0^z} = E_0\bar Q_0(d_0^z(W), z, W)`.

The witnesses are the plan's (section 4, PR 2):

- W-f3-id: each composition's point, variance, interval, curve and record equal the learned-rule
  fit's with the same setting bit for bit, at a nonzero fluctuation; only the names, the
  target and the family of the key change.
- W-f3w: on the weighted law of the study the quadrature gap
  :math:`\Psi^w_{d_0^w} - \Psi_{d_0}` is 0.0537; under the weight :math:`1 + A Y` the weighted
  optimum :math:`d_0^w` differs from :math:`d_0` on a declared share, its value gap under
  :math:`P_w` exceeds a declared floor, and the fit's fold rules follow :math:`d_0^w`.
- W-f3s: the two strata's optimal values differ by a declared margin, and their share mixture
  is the marginal optimal value.
- W-f3c: the contrast's oracle is the optimal value minus the known regime's, each fold's curve
  is its Gateaux derivative, and the learned-rule contrast and the flipped sign fail.
- W-f3z: at each level of the intermediate law the value of :math:`d_0^z` exceeds the value of
  the total-effect rule :math:`\arg\max_a \sum_z q_0(z \mid a, w)\bar Q_0(a, z, w)` by a
  declared floor.
- W-f3n: the band refusal and the default-band skip for the optimal contrast, the stratified fit,
  and the declaration refusals of a name of the other target.
- W-f3i: the in-sample optimal fit equals the in-sample learned-rule fit, and an adaptive learner
  keeps the in-sample warning.

The sensitivity analyses of an optimal-value fit keep their refusal in this PR; the last class
pins it for every admitted composition.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    LearnedRuleValue,
    OptimalRuleValue,
    PointTreatment,
    TMLEMethod,
)
from cleverly.estimators import TMLE
from cleverly.estimators._strata import FOLD_EVALUATED_STRATA_TARGETS
from cleverly.exceptions import CapabilityError, InSampleLearnerWarning, RefusalReason
from cleverly.inference import simultaneous_bands
from cleverly.inference.multiplier import KNOWN_REGIME_BAND_REFUSAL
from cleverly.interventions import LearnedRule, Static
from cleverly.interventions.learned import (
    CONTRAST_TARGETS,
    OPTIMAL_CONTRAST_TARGET,
    OPTIMAL_RULE_SENSITIVITY_REFUSAL,
    OPTIMAL_RULE_TARGET,
)
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.sensitivity.omitted_variable import benchmark_refusal, fit_wide_bound_refusal
from cleverly.targets import TARGETS
from tests import discrete_law as exact
from tests.conftest import OracleOutcome, OracleTreatment
from tests.studies import _learned_rule_intermediate_law as zlaw
from tests.studies import _learned_rule_law as base
from tests.unit import _learned_rule_support as support
from tests.unit.test_learned_rule_intermediate import cde_frame, fit_cde
from tests.unit.test_learned_rule_intermediate import outcome_learner as cde_outcome_learner
from tests.unit.test_learned_rule_optimal import EXACT_COUNTS, EXACT_PROBS, WRONG_Q, _exact_law

FIT: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
LEARNED = support.NAME
OPTIMAL = "ey_optimal_rule[learned rule]"

#: W-f3w: the weight of the study's ``weighted`` cell, ``1 + 1.5 * 1{W1 > 0}``, and the gap
#: ``Psi^w_{d0^w}(P_w) - Psi_{d0}(P0)`` the plan computes (``rows/X11/scripts/gap.py``).
WEIGHTED_GAP = 0.0537
#: W-f3w: the weight ``1 + c A Y`` that moves the optimum, its ``c``, the floor of the share of
#: ``W`` on which ``d0^w`` differs from ``d0`` (0.347 by quadrature) and the floor of the value
#: gap ``Psi^w_{d0^w} - Psi^w_{d0}`` under ``P_w`` (0.0277).
JOINT_C = 1.0
JOINT_SHARE_FLOOR = 0.3
JOINT_GAP_FLOOR = 0.025
#: W-f3s: the floor of the difference of the two strata's optimal values (0.0670).
STRATA_MARGIN = 0.05
#: W-f3z: the floors of the value gap between the level optimum and the total-effect rule at
#: each level (0.00676 and 0.00426 by quadrature), and of the share on which they differ
#: (0.166 and 0.134).
LEVEL_GAP_FLOORS = {0.0: 0.005, 1.0: 0.003}
LEVEL_SHARE_FLOOR = 0.1


def _flat(text: Any) -> str:
    return " ".join(str(text).split())


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _optimal_name(name: str) -> str:
    """The name the optimal-value fit reports for the learned-rule fit's ``name``."""
    return name.replace("ey_learned_rule", "ey_optimal_rule").replace(
        "ate_learned_rule", "ate_optimal_rule"
    )


# ===================================================================== the quadrature

#: The trapezoid mass of each grid point of ``W1`` for one level of ``W2`` (density 1/2,
#: probability 1/2).
MASS = np.full(base.GRID.size, base.GRID[1] - base.GRID[0])
MASS[[0, -1]] /= 2.0
MASS *= 0.25


def _qbar(a: float, w1: np.ndarray, w2: float) -> np.ndarray:
    return base.qbar0(a, w1, w2, "non_exceptional")


def weighted_values(
    weight: Callable[[float, float, np.ndarray, float], Any],
) -> dict[str, float]:
    r"""The optimal values of the tilt :math:`dP_w = w\,dP/E[w]` of the ``non_exceptional`` law.

    Under :math:`P_w` the outcome regression is
    :math:`\bar Q_w(a, W) = E[wY \mid a, W] / E[w \mid a, W]` and the density of ``W`` is
    proportional to :math:`E[w \mid W] = \sum_a g_0(a \mid W) E[w \mid a, W]`.  ``weighted`` is
    :math:`\Psi^w_{d_0^w}(P_w)`, ``at_d0`` the value of the unweighted optimum :math:`d_0` under
    :math:`P_w`, ``unweighted`` :math:`\Psi_{d_0}(P_0)`, and ``share`` the :math:`P_0`-mass of
    ``W`` on which :math:`d_0^w \ne d_0`.
    """
    weighted = at_d0 = mass = unweighted = share = 0.0
    for w2 in base.W2_LEVELS:
        treated = base.g0(base.GRID, w2)
        tilted, total = {}, np.zeros(base.GRID.size)
        for a in (0.0, 1.0):
            q = _qbar(a, base.GRID, w2)
            top = weight(a, 1.0, base.GRID, w2) * q
            bottom = top + weight(a, 0.0, base.GRID, w2) * (1.0 - q)
            tilted[a] = top / bottom
            total = total + (treated if a else 1.0 - treated) * bottom
        d0w = tilted[1.0] > tilted[0.0]
        d0 = _qbar(1.0, base.GRID, w2) > _qbar(0.0, base.GRID, w2)
        weighted += float(np.sum(MASS * total * np.where(d0w, tilted[1.0], tilted[0.0])))
        at_d0 += float(np.sum(MASS * total * np.where(d0, tilted[1.0], tilted[0.0])))
        mass += float(np.sum(MASS * total))
        unweighted += float(
            np.sum(MASS * np.where(d0, _qbar(1.0, base.GRID, w2), _qbar(0.0, base.GRID, w2)))
        )
        share += float(np.sum(MASS * (d0w != d0)))
    return {
        "weighted": weighted / mass,
        "at_d0": at_d0 / mass,
        "unweighted": unweighted,
        "share": share,
    }


def stratum_values() -> dict[float, float]:
    """:math:`E_0[\\bar Q_0(d_0(W), W) \\mid W_2 = s]` for each level of the stratum ``W2``."""
    out = {}
    for w2 in base.W2_LEVELS:
        d0 = _qbar(1.0, base.GRID, w2) > _qbar(0.0, base.GRID, w2)
        value = np.where(d0, _qbar(1.0, base.GRID, w2), _qbar(0.0, base.GRID, w2))
        out[w2] = float(np.sum(MASS * value) / np.sum(MASS))
    return out


def level_values(z: float) -> dict[str, float]:
    r"""The level-:math:`z` value of :math:`d_0^z` and of the total-effect rule, by quadrature.

    On the law of :mod:`tests.studies._learned_rule_intermediate_law`,
    :math:`d_0^z(w) = \arg\max_a \bar Q_0(a, z, w)` and
    :math:`d_{\rm blind}(w) = \arg\max_a \sum_{z'} q_0(z' \mid a, w)\bar Q_0(a, z', w)`, the
    optimal rule of the total effect.  Both are evaluated at level ``z``:
    :math:`E_0\bar Q_0(d(W), z, W)`.
    """
    optimal = blind = share = 0.0
    for w2 in zlaw.W2_LEVELS:
        dz = zlaw.qbar0(1.0, z, base.GRID, w2) > zlaw.qbar0(0.0, z, base.GRID, w2)

        def total(a: float, w2: float = w2) -> np.ndarray:
            return sum(
                zlaw.q0(level, a, base.GRID) * zlaw.qbar0(a, level, base.GRID, w2)
                for level in zlaw.LEVELS
            )

        db = total(1.0) > total(0.0)
        on = zlaw.qbar0(1.0, z, base.GRID, w2), zlaw.qbar0(0.0, z, base.GRID, w2)
        optimal += float(np.sum(MASS * np.where(dz, *on)))
        blind += float(np.sum(MASS * np.where(db, *on)))
        share += float(np.sum(MASS * (dz != db)))
    return {"optimal": optimal, "blind": blind, "share": share}


# ===================================================================== W-f3-id


def _weighted_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    return support.law_frame(n, seed).assign(w=lambda f: 1.0 + 1.5 * (f["W1"] > 0))


def _engine_pair(
    frame: pd.DataFrame | None = None, fit: dict[str, Any] | None = None, **overrides: Any
) -> tuple[Any, Any]:
    """The learned-rule fit and the optimal-value fit of one setting, on one draw."""
    frame = support.law_frame(203, 1) if frame is None else frame
    fit = {} if fit is None else fit
    settings = {"outcome_learner": support.misspecified_learner(), **overrides}
    return _quiet(
        lambda: tuple(
            support.learned_tmle(**settings, **extra).fit(frame, **FIT, **fit).single()
            for extra in ({}, {"estimands": [OPTIMAL_RULE_TARGET]})
        )
    )


def _level_pair(z: float) -> tuple[Any, Any]:
    learned = fit_cde()[z]
    optimal = fit_cde(estimands=[OPTIMAL_RULE_TARGET])[z]
    return learned, optimal


def _cde_study(frame: pd.DataFrame | None = None) -> CausalStudy:
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), intermediate="Z")
    return CausalStudy(cde_frame() if frame is None else frame, design=design)


def _method_level_pair(z: float) -> tuple[Any, Any]:
    options = {
        "outcome_learner": cde_outcome_learner(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "intermediate_learner": LogisticRegression(max_iter=1000),
        "n_folds": 5,
        "cv_evaluation": True,
        "simultaneous": False,
        "random_state": 0,
    }
    study = _cde_study()
    return _quiet(
        lambda: tuple(
            study.identify(estimand(intermediate=z)).estimate(**options)
            for estimand in (LearnedRuleValue, OptimalRuleValue)
        )
    )


#: Each admitted composition, as a pair of fits on one draw: the learned-rule fit and the
#: optimal-value fit with the same setting.
PAIRS: dict[str, Callable[[], tuple[Any, Any]]] = {
    "weights": lambda: _engine_pair(_weighted_frame(203, 1), {"weights": "w"}),
    "estimated_weights": lambda: _engine_pair(
        _weighted_frame(203, 1), {"weights": "w", "weights_estimated": True}
    ),
    "strata": lambda: _engine_pair(support.law_frame(400, 2), {"strata": ["W2"]}),
    "strata_stacked": lambda: _engine_pair(
        support.law_frame(400, 2), {"strata": ["W2"]}, cv_evaluation=False
    ),
    "known_regimes": lambda: _engine_pair(interventions=[Static(1), Static(0)]),
    "known_regimes_stacked": lambda: _engine_pair(
        interventions=[Static(1), Static(0)], cv_evaluation=False
    ),
    "reference": lambda: _engine_pair(interventions=[Static(1)], reference="learned rule"),
    "known_regimes_strata": lambda: _engine_pair(
        support.law_frame(400, 2), {"strata": ["W2"]}, interventions=[Static(1)]
    ),
    "in_sample": lambda: _engine_pair(cross_fit=False, cv_evaluation=False),
    "intermediate_0": lambda: _level_pair(0.0),
    "intermediate_1": lambda: _level_pair(1.0),
    "method_intermediate_0": lambda: _method_level_pair(0.0),
    "method_intermediate_1": lambda: _method_level_pair(1.0),
}


@cache
def _pairs(name: str) -> tuple[Any, Any]:
    return PAIRS[name]()


class TestWf3IdIdentity:
    """W-f3-id: every composition is the learned-rule fit with the same setting, renamed."""

    @pytest.mark.parametrize("name", list(PAIRS))
    def test_every_estimate_is_bit_for_bit(self, name: str) -> None:
        learned, optimal = _pairs(name)
        assert list(optimal.estimates) == [_optimal_name(key) for key in learned.estimates]
        for key, a in learned.estimates.items():
            b = optimal.estimates[_optimal_name(key)]
            assert a.psi == b.psi, key
            assert a.variance == b.variance, key
            assert a.ci == b.ci, key
            np.testing.assert_array_equal(a.influence_curve, b.influence_curve)
        record_a, record_b = learned.extra["learned_rule"], optimal.extra["learned_rule"]
        assert record_b == replace(record_a, estimand=OPTIMAL_RULE_TARGET)

    @pytest.mark.parametrize("name", list(PAIRS))
    def test_only_the_family_of_the_key_changes(self, name: str) -> None:
        learned, optimal = _pairs(name)
        key_a, key_b = learned.config.support_key, optimal.config.support_key
        assert key_a.family == frozenset({"P11"})
        assert key_b.family == frozenset({"P12"})
        assert replace(key_b, family=key_a.family) == key_a

    @pytest.mark.parametrize("name", list(PAIRS))
    def test_every_composition_reads_experimental(self, name: str) -> None:
        """No study registers these keys yet, so no row covers any of them."""
        _, optimal = _pairs(name)
        for estimate in optimal.estimates.values():
            assert estimate.support.label == "experimental"
            assert estimate.support.row is None

    @pytest.mark.parametrize(
        "name", ["weights", "strata", "known_regimes", "in_sample", "intermediate_0"]
    )
    def test_the_fluctuation_moves(self, name: str) -> None:
        """The nonzero witness: the identity is not that of an untargeted plug-in."""
        _, optimal = _pairs(name)
        assert float(np.max(np.abs(optimal.fluctuations["regime"].epsilon))) > 1e-3

    def test_the_contrast_is_the_difference_of_its_columns(self) -> None:
        _, optimal = _pairs("known_regimes")
        for known in ("always 1", "always 0"):
            contrast = optimal.estimates[f"ate_optimal_rule[learned rule vs {known}]"]
            value = optimal.estimates[OPTIMAL]
            regime = optimal.estimates[f"ey_known_regime[{known}]"]
            assert contrast.psi == pytest.approx(value.psi - regime.psi, abs=1e-14)
            np.testing.assert_allclose(
                contrast.influence_curve,
                value.influence_curve - regime.influence_curve,
                atol=1e-14,
                rtol=0,
            )

    def test_reference_flips_the_sign(self) -> None:
        """The mutation of the sign: ``reference=`` naming the learned rule reverses it."""
        _, flipped = _pairs("reference")
        _, default = _engine_pair(interventions=[Static(1)])
        forward = default.estimates["ate_optimal_rule[learned rule vs always 1]"]
        backward = flipped.estimates["ate_optimal_rule[always 1 vs learned rule]"]
        assert backward.psi == -forward.psi
        assert abs(forward.psi) > 1e-3

    def test_the_stratum_estimates_mix_to_the_marginal(self) -> None:
        _, optimal = _pairs("strata")
        frame = support.law_frame(400, 2)
        share = float(np.mean(frame["W2"] == 1.0))
        mixture = (1.0 - share) * optimal.estimates[f"{OPTIMAL}[W2=0.0]"].psi + share * (
            optimal.estimates[f"{OPTIMAL}[W2=1.0]"].psi
        )
        assert mixture == pytest.approx(optimal.estimates[OPTIMAL].psi, abs=1e-12)

    def test_the_contrast_keys_known_regimes(self) -> None:
        _, optimal = _pairs("known_regimes")
        assert "known_regimes" in optimal.config.support_key.qualifiers


# ===================================================================== W-f3w


class TestWf3wTheWeightedOptimum:
    """W-f3w: the weighted optimal value is another number, and a joint weight moves the rule."""

    def test_the_study_weight_moves_the_value_and_not_the_rule(self) -> None:
        values = weighted_values(lambda a, y, w1, w2: 1.0 + 1.5 * (w1 > 0))
        assert values["share"] == 0.0
        assert values["weighted"] == values["at_d0"]
        assert values["weighted"] - values["unweighted"] == pytest.approx(WEIGHTED_GAP, abs=5e-4)

    def test_a_weight_joint_in_a_and_y_moves_the_optimum(self) -> None:
        values = weighted_values(lambda a, y, w1, w2: 1.0 + JOINT_C * a * y)
        assert values["share"] > JOINT_SHARE_FLOOR, values
        assert values["weighted"] - values["at_d0"] > JOINT_GAP_FLOOR, values

    @pytest.mark.parametrize(
        "weight",
        [
            lambda a, y, w1, w2: 1.0 + JOINT_C * a + 0.0 * w1,
            lambda a, y, w1, w2: 1.0 + JOINT_C * y * (w1 > 0),
        ],
        ids=["treatment_alone", "covariate_and_outcome"],
    )
    def test_control_a_weight_not_joint_in_a_and_y_keeps_the_rule(self, weight: Any) -> None:
        """A weight in ``A`` alone keeps ``Qbar``; one in ``(W, Y)`` keeps the blip's sign."""
        values = weighted_values(weight)
        assert values["share"] == 0.0
        assert values["weighted"] == values["at_d0"]

    def test_the_fit_learns_the_weighted_optimum(self) -> None:
        """At n = 3,000 the fold rules treat where ``d0^w`` does and ``d0`` does not."""
        frame = support.law_frame(3_000, 11).assign(w=lambda f: 1.0 + JOINT_C * f["A"] * f["Y"])
        band = (frame["W1"] > -0.7) & (frame["W1"] < -0.2)
        estimator = support.learned_tmle(estimands=[OPTIMAL_RULE_TARGET])
        weighted = _quiet(lambda: estimator.fit(frame, weights="w", **FIT).single())
        unweighted = _quiet(lambda: estimator.fit(frame, **FIT).single())
        for result, low, high in ((weighted, 0.9, 1.0), (unweighted, 0.0, 0.1)):
            rules = np.asarray(result.nuisance.regimes.fold_rules)
            treated = float(np.mean(rules[band.to_numpy()] == 1))
            assert low <= treated <= high, treated

    def test_the_summary_names_the_weighted_population(self) -> None:
        _, optimal = _pairs("weights")
        line = optimal.extra["learned_rule"].describe()
        assert "the value of the optimal rule of the weighted population (weights=)" in line
        assert "depend on the treatment and the outcome together" in line


# ===================================================================== W-f3s


class TestWf3sTheStrata:
    """W-f3s: stratum optimal values differ, and their share mixture is the marginal one."""

    def test_the_stratum_values_differ(self) -> None:
        values = stratum_values()
        assert abs(values[0.0] - values[1.0]) > STRATA_MARGIN, values

    def test_the_share_mixture_is_the_marginal_value(self) -> None:
        values = stratum_values()
        marginal = weighted_values(lambda a, y, w1, w2: 1.0 + 0.0 * w1)["unweighted"]
        assert 0.5 * values[0.0] + 0.5 * values[1.0] == pytest.approx(marginal, abs=1e-12)

    def test_the_ten_fold_stratified_fit_runs(self) -> None:
        """The study cell's setting: ``strata=W2``, ten folds, no X28 refusal."""
        result = _quiet(
            lambda: support.fit(
                support.law_frame(2_000, 4),
                estimands=[OPTIMAL_RULE_TARGET],
                n_folds=10,
                outcome_learner=support.outcome_learner(),
            )
        )
        assert list(result.estimates) == [OPTIMAL]
        stratified = _quiet(
            lambda: (
                support.learned_tmle(estimands=[OPTIMAL_RULE_TARGET], n_folds=10)
                .fit(support.law_frame(2_000, 4), strata=["W2"], **FIT)
                .single()
            )
        )
        assert set(stratified.estimates) == {OPTIMAL, f"{OPTIMAL}[W2=0.0]", f"{OPTIMAL}[W2=1.0]"}

    def test_the_summary_names_the_stratum_target(self) -> None:
        _, optimal = _pairs("strata")
        assert "Each stratum of W2 targets the optimal value within the stratum" in (
            optimal.extra["learned_rule"].describe()
        )

    def test_the_targets_are_fold_evaluated_stratum_targets(self) -> None:
        assert {OPTIMAL_RULE_TARGET, OPTIMAL_CONTRAST_TARGET} <= FOLD_EVALUATED_STRATA_TARGETS


# ===================================================================== W-f3c

OPTIMAL_CONTRASTS = exact.PER_ARM_NAMES["ate_optimal_rule"]
EXACT_OPTIMAL = f"ey_optimal_rule[{exact.LEARNED_RULE_LABEL}]"


@cache
def _exact_contrast_fit() -> Any:
    return (
        TMLE(
            learned_rule=LearnedRule(),
            estimands=[OPTIMAL_RULE_TARGET],
            interventions=[Static(1), Static(0)],
            outcome_learner=OracleOutcome(exact.DiscreteLaw(_exact_law(WRONG_Q))),
            treatment_learner=OracleTreatment(exact.DiscreteLaw(EXACT_PROBS)),
            cross_fit=True,
            cv_evaluation=True,
            n_folds=4,
            simultaneous=False,
            random_state=3,
        )
        .fit(exact.frame(EXACT_COUNTS), outcome="Y", treatment="A")
        .single()
    )


def _exact_fold_curves(name: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each fold: the support point of each row, the fold's unscaled curve, the fold law."""
    result = _exact_contrast_fit()
    n = result.data.n
    cells = np.repeat(
        np.arange(len(exact.SUPPORT)), [EXACT_COUNTS[w, a, y] for w, a, y in exact.SUPPORT]
    )
    w = np.rint(result.data.covariates[:, 0]).astype(int)
    targeted = result.fluctuations["regime"].targeted
    q_star = np.array(
        [
            [float(targeted.arms[float(a)][np.flatnonzero(w == cell)[0]]) for a in (0, 1)]
            for cell in range(3)
        ]
    )
    g = EXACT_PROBS.sum(axis=2)[:, 1] / EXACT_PROBS.sum(axis=(1, 2))
    arm = np.stack([1.0 - g, g], axis=1)
    outcome = np.stack([1.0 - q_star, q_star], axis=2)
    curve = np.asarray(result.estimates[name].influence_curve, dtype=float)
    folds = [np.asarray(test) for _, test in result.nuisance.folds]
    pieces = []
    for test in folds:
        share = np.bincount(w[test], minlength=3) / test.size
        probs = share[:, None, None] * arm[:, :, None] * outcome
        pieces.append((cells[test], curve[test] * len(folds) * test.size / n, probs))
    return pieces


class TestWf3cTheContrast:
    """W-f3c: the contrast's truth, its curve, and the mutations of name and sign."""

    def test_the_truth_is_the_optimal_value_minus_the_known_value(self) -> None:
        for contrast, known in zip(
            OPTIMAL_CONTRASTS, exact.PER_ARM_NAMES["ey_known_regime"], strict=True
        ):
            assert exact.TRUTH[contrast] == pytest.approx(
                exact.TRUTH[EXACT_OPTIMAL] - exact.TRUTH[known], abs=1e-15
            )
            np.testing.assert_allclose(
                exact.eif(contrast), exact.eif(EXACT_OPTIMAL) - exact.eif(known), atol=1e-14
            )

    def test_control_the_learned_rule_contrast_and_the_flipped_sign_fail(self) -> None:
        """At the base law every blip is positive, so ``d0`` treats everyone while the fixed
        rule of the learned-rule oracle does not treat ``w = 1``: the two contrasts differ."""
        for optimal, learned in zip(
            OPTIMAL_CONTRASTS, exact.PER_ARM_NAMES["ate_learned_rule"], strict=True
        ):
            assert abs(exact.TRUTH[optimal] - exact.TRUTH[learned]) > 1e-2
        nonzero = exact.TRUTH[OPTIMAL_CONTRASTS[1]]
        assert abs(nonzero - (-nonzero)) > 1e-2

    def test_the_fit_reports_every_oracle_name(self) -> None:
        names = set(_exact_contrast_fit().estimates)
        assert names == {EXACT_OPTIMAL, *exact.PER_ARM_NAMES["ey_known_regime"], *OPTIMAL_CONTRASTS}
        assert float(np.min(np.abs(_exact_contrast_fit().fluctuations["regime"].epsilon))) > 1e-2

    @pytest.mark.parametrize("name", OPTIMAL_CONTRASTS)
    def test_each_fold_curve_is_the_gateaux_derivative(self, name: str) -> None:
        for cells, raw, probs in _exact_fold_curves(name):
            for point in np.unique(cells):
                expected = exact.gateaux(name, int(point), probs=probs)
                np.testing.assert_allclose(raw[cells == point], expected, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("name", OPTIMAL_CONTRASTS)
    def test_control_a_contrast_without_its_known_term_is_caught(self, name: str) -> None:
        gaps = [
            np.max(
                np.abs(raw[cells == point] - exact.gateaux(EXACT_OPTIMAL, int(point), probs=probs))
            )
            for cells, raw, probs in _exact_fold_curves(name)
            for point in np.unique(cells)
        ]
        assert float(np.max(gaps)) > 1e-2

    def test_the_record_of_the_contrast_carries_the_regret(self) -> None:
        record = TARGETS[OPTIMAL_CONTRAST_TARGET].identification
        text = " ".join(record.assumptions)
        assert "the regret of the contrast" in text
        assert "positivity for each known regime" in text
        assert "joint asymptotic linearity" in text
        assert TARGETS[OPTIMAL_CONTRAST_TARGET].in_default_set is False

    def test_the_summary_names_the_contrast_target(self) -> None:
        _, optimal = _pairs("known_regimes")
        line = optimal.extra["learned_rule"].describe()
        assert "Each contrast ate_optimal_rule[...] targets the optimal value minus" in line


# ===================================================================== W-f3z


class TestWf3zTheLevelOptimum:
    """W-f3z: at each level the optimum differs from the total-effect rule by a value gap."""

    @pytest.mark.parametrize("z", zlaw.LEVELS)
    def test_the_level_optimum_beats_the_total_effect_rule(self, z: float) -> None:
        values = level_values(z)
        assert values["optimal"] - values["blind"] > LEVEL_GAP_FLOORS[z], values
        assert values["share"] > LEVEL_SHARE_FLOOR, values

    @pytest.mark.parametrize("z", zlaw.LEVELS)
    def test_the_level_optimum_is_the_best_rule(self, z: float) -> None:
        """No rule of the grid does better at level ``z``: the other level's rule is worse."""
        other = 1.0 - z
        value = 0.0
        for w2 in zlaw.W2_LEVELS:
            rule = zlaw.qbar0(1.0, other, base.GRID, w2) > zlaw.qbar0(0.0, other, base.GRID, w2)
            on = zlaw.qbar0(1.0, z, base.GRID, w2), zlaw.qbar0(0.0, z, base.GRID, w2)
            value += float(np.sum(MASS * np.where(rule, *on)))
        assert level_values(z)["optimal"] > value

    def test_the_summary_names_the_level(self) -> None:
        _, optimal = _pairs("intermediate_1")
        line = optimal.extra["learned_rule"].describe()
        assert "with the intermediate held at Z = 1" in line
        assert "not the best pair of treatment and intermediate" in line

    def test_the_identified_record_holds_the_intermediate_assumptions(self) -> None:
        effect = _cde_study().identify(OptimalRuleValue(intermediate=0.0))
        text = " ".join(effect.identification.assumptions)
        assert "intermediate positivity at Z = 0" in text
        assert "no observation weights" not in text
        assert effect.functional.intermediate == 0.0
        assert effect.functional.target == OPTIMAL_RULE_TARGET


# ===================================================================== W-f3n


def _known_optimal(**overrides: Any) -> Any:
    settings = {"estimands": [OPTIMAL_RULE_TARGET], "interventions": [Static(1)], **overrides}
    return _quiet(lambda: support.fit(**settings))


def _refused(call: Callable[[], Any]) -> CapabilityError:
    with pytest.raises(CapabilityError) as caught:
        call()
    return caught.value


class TestWf3nNamesAndWiring:
    """W-f3n: the band, the names a fit reports and the names it refuses."""

    def test_the_contrast_is_a_contrast_target(self) -> None:
        assert OPTIMAL_CONTRAST_TARGET in CONTRAST_TARGETS

    def test_the_default_band_is_skipped(self) -> None:
        estimator = support.learned_tmle(
            estimands=[OPTIMAL_RULE_TARGET], interventions=[Static(1)], simultaneous=True
        )
        assert estimator.simultaneous
        result = _quiet(lambda: estimator.fit(support.law_frame(), **FIT).single())
        assert result.simultaneous is None

    def test_control_without_known_regimes_the_band_runs(self) -> None:
        """Nonzero witness: the stratified optimal fit alone builds its band."""
        result = _quiet(
            lambda: (
                support.learned_tmle(estimands=[OPTIMAL_RULE_TARGET], simultaneous=True)
                .fit(support.law_frame(), strata=["W2"], **FIT)
                .single()
            )
        )
        assert result.simultaneous is not None

    @pytest.mark.parametrize(
        "names",
        [
            [OPTIMAL, "ey_known_regime[always 1]", "ate_optimal_rule[learned rule vs always 1]"],
            [OPTIMAL, "ate_optimal_rule[learned rule vs always 1]"],
            ["ate_optimal_rule[learned rule vs always 1]"],
        ],
        ids=["all", "value_and_contrast", "contrast_alone"],
    )
    def test_simultaneous_bands_refuses_the_optimal_contrast(self, names: list[str]) -> None:
        estimates = _known_optimal().estimates
        caught = _refused(lambda: simultaneous_bands({name: estimates[name] for name in names}))
        assert caught.refusal == KNOWN_REGIME_BAND_REFUSAL
        assert "X11 (c2)" in str(caught)
        assert "ate_optimal_rule" in str(caught)

    def test_one_name_alone_reports_every_contrast_and_an_explicit_list_its_own(self) -> None:
        both = _known_optimal()
        assert list(both.estimates) == [
            OPTIMAL,
            "ey_known_regime[always 1]",
            "ate_optimal_rule[learned rule vs always 1]",
        ]
        explicit = _known_optimal(estimands=[OPTIMAL_RULE_TARGET, OPTIMAL_CONTRAST_TARGET])
        assert list(explicit.estimates) == [OPTIMAL, "ate_optimal_rule[learned rule vs always 1]"]

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            (
                {
                    "estimands": [OPTIMAL_RULE_TARGET, "ate_learned_rule"],
                    "interventions": [Static(1)],
                },
                "the contrast of the optimal value with a known regime is ate_optimal_rule",
            ),
            (
                {"estimands": [OPTIMAL_RULE_TARGET, "ey_learned_rule"]},
                "One fit reports one of the two targets",
            ),
            (
                {"estimands": [OPTIMAL_CONTRAST_TARGET], "interventions": [Static(1)]},
                "it needs ey_optimal_rule among the estimands",
            ),
            (
                {
                    "estimands": ["ey_learned_rule", OPTIMAL_CONTRAST_TARGET],
                    "interventions": [Static(1)],
                },
                "it needs ey_optimal_rule among the estimands",
            ),
            (
                {"estimands": [OPTIMAL_RULE_TARGET, OPTIMAL_CONTRAST_TARGET]},
                "this fit declares none",
            ),
            (
                {"estimands": [OPTIMAL_RULE_TARGET], "reference": "learned rule"},
                "this fit declares none",
            ),
        ],
        ids=[
            "learned_contrast",
            "learned_value",
            "contrast_without_value",
            "contrast_beside_learned",
            "contrast_without_regimes",
            "reference_without_regimes",
        ],
    )
    def test_the_declaration_refusals(self, overrides: dict[str, Any], fragment: str) -> None:
        caught = _refused(lambda: support.learned_tmle(**overrides))
        assert caught.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert fragment in _flat(caught)

    def test_a_retarget_to_the_other_family_refuses(self) -> None:
        optimal = _known_optimal()
        caught = _refused(
            lambda: optimal.estimator.retarget(
                optimal.data, optimal.nuisance, estimands=("ate_learned_rule",)
            )
        )
        assert caught.refusal.reason is RefusalReason.INVALID_DECLARATION
        learned = _quiet(lambda: support.fit(interventions=[Static(1)]))
        caught = _refused(
            lambda: learned.estimator.retarget(
                learned.data, learned.nuisance, estimands=(OPTIMAL_CONTRAST_TARGET,)
            )
        )
        assert "it needs ey_optimal_rule among the estimands" in _flat(caught)

    def test_the_strata_with_clusters_keep_x28(self) -> None:
        frame = support.law_frame().assign(cluster=np.repeat(np.arange(100), 4))
        caught = _refused(
            lambda: support.learned_tmle(estimands=[OPTIMAL_RULE_TARGET]).fit(
                frame, strata=["W2"], id="cluster", **FIT
            )
        )
        assert caught.refusal.roadmap == "X28"

    def test_a_rule_learner_with_an_intermediate_keeps_j2(self) -> None:
        from sklearn.linear_model import LinearRegression

        caught = _refused(
            lambda: _cde_study().identify(
                OptimalRuleValue(
                    rule=LearnedRule(loss="blip", learner=LinearRegression()), intermediate=0.0
                )
            )
        )
        assert "X11 (j2)" in str(caught)


# ===================================================================== W-f3i


class TestWf3iInSample:
    """W-f3i: the in-sample optimal fit, its warning and its summary."""

    def test_the_in_sample_fit_is_the_learned_rule_fit(self) -> None:
        learned, optimal = _pairs("in_sample")
        assert optimal.extra["learned_rule"].scheme == "in_sample"
        assert optimal.estimates[OPTIMAL].psi == learned.estimates[LEARNED].psi

    def test_an_adaptive_learner_keeps_the_in_sample_warning(self) -> None:
        estimator = support.learned_tmle(
            estimands=[OPTIMAL_RULE_TARGET],
            cross_fit=False,
            cv_evaluation=False,
            outcome_learner=RandomForestClassifier(n_estimators=20, random_state=0),
        )
        with pytest.warns(InSampleLearnerWarning):
            result = estimator.fit(support.law_frame(), **FIT).single()
        assert result.estimates[OPTIMAL].support.label == "experimental"

    def test_the_summary_names_the_in_sample_estimator(self) -> None:
        _, optimal = _pairs("in_sample")
        line = optimal.extra["learned_rule"].describe()
        assert "the value of the one rule learned on every row" in line
        assert "condition C5a" in line
        assert "The fold rules assign" not in line

    def test_the_method_layer_runs_the_in_sample_fit(self) -> None:
        study = CausalStudy(
            support.law_frame(),
            design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
        )
        options = {
            "outcome_learner": support.outcome_learner(),
            "treatment_learner": support.treatment_learner(),
            "simultaneous": False,
            "random_state": 0,
        }
        method = TMLEMethod(cross_fitting=CrossFitting(enabled=False))
        learned, optimal = _quiet(
            lambda: tuple(
                study.identify(estimand()).estimate(method, **options)
                for estimand in (LearnedRuleValue, OptimalRuleValue)
            )
        )
        assert optimal.estimates[OPTIMAL].psi == learned.estimates[LEARNED].psi


# =========================================================== the method layer


class TestTheMethodLayer:
    """``OptimalRuleValue`` identifies and estimates on weighted and stratified designs."""

    @pytest.mark.parametrize(
        "extra", [{"weights": "w"}, {"strata": ("W2",)}], ids=["weights", "strata"]
    )
    def test_the_study_estimates_the_learned_rule_estimate(self, extra: dict[str, Any]) -> None:
        design = PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES, **extra)
        study = CausalStudy(_weighted_frame(), design=design)
        method = TMLEMethod(cross_fitting=CrossFitting(fold_evaluation=True))
        options = {
            "outcome_learner": support.outcome_learner(),
            "treatment_learner": support.treatment_learner(),
            "simultaneous": False,
            "random_state": 0,
        }
        learned, optimal = _quiet(
            lambda: tuple(
                study.identify(estimand()).estimate(method, **options)
                for estimand in (LearnedRuleValue, OptimalRuleValue)
            )
        )
        assert [_optimal_name(name) for name in learned.estimates] == list(optimal.estimates)
        for name, estimate in learned.estimates.items():
            assert optimal.estimates[_optimal_name(name)].psi == estimate.psi

    def test_the_identified_record_describes_each_composition(self) -> None:
        study = CausalStudy(
            support.law_frame(),
            design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
        )
        text = " ".join(study.identify(OptimalRuleValue()).identification.assumptions)
        for fragment in ("d0^w", "E0[Qbar0(d0(W), W) | S = s]", "d0^z", "condition C5a"):
            assert fragment in text, fragment
        assert "no observation weights, baseline strata" not in text


# ======================================================= the sensitivity analyses


#: Every admitted composition that the sensitivity analyses are asked about.
SENSITIVITY_COMPOSITIONS = (
    "weights",
    "estimated_weights",
    "strata",
    "known_regimes",
    "in_sample",
    "intermediate_0",
    "method_intermediate_1",
)


class TestTheSensitivityAnalysesStillRefuse:
    """The omitted-variable bound and ``simulated_confounding`` keep the optimal refusal."""

    @pytest.mark.parametrize("name", SENSITIVITY_COMPOSITIONS)
    def test_the_bound_refuses_every_composition(self, name: str) -> None:
        _, optimal = _pairs(name)
        assert fit_wide_bound_refusal(optimal) is OPTIMAL_RULE_SENSITIVITY_REFUSAL
        assert benchmark_refusal(optimal, ["W2"]) is OPTIMAL_RULE_SENSITIVITY_REFUSAL

    @pytest.mark.parametrize("name", SENSITIVITY_COMPOSITIONS)
    def test_simulated_confounding_refuses_every_name(self, name: str) -> None:
        _, optimal = _pairs(name)
        for estimand in optimal.estimates:
            refusal = simulated_confounding_refusal(optimal, estimand)
            assert refusal is OPTIMAL_RULE_SENSITIVITY_REFUSAL, estimand
