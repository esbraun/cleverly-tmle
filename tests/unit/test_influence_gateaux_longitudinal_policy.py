"""Known stochastic policies at a longitudinal node, against an independent g-formula.

The oracle is :mod:`tests.discrete_law_longitudinal_policy`: the parameter under each plan
written as ratios of finite-support masses, and its complex-step Gateaux derivative.  A
saturated learner on a sample that realises the law exactly makes every comparison exact,
so the point estimate and the influence curve must equal the oracle to rounding.

Exact-law checks are blind to a term that vanishes at the truth.  So the module also holds
nonzero witnesses for every policy-specific term (the ratio numerator, the policy-weighted
marginal, the earlier arm in the policy frame, the level order, the support mask), and one
``monkeypatch`` mutation per term that must move the estimate or the curve by more than
``1e-4``.  A misspecified initial fit (T5b) makes the fluctuation coefficients nonzero, so
the targeting itself has a witness: the per-arm update, the carried marginal and the
numerator in the loss weight.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache, partial
from typing import Any

import numpy as np
import pytest
from scipy.optimize import brentq
from sklearn.dummy import DummyRegressor

from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData, sequential
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import regimen as regimen_module
from cleverly.longitudinal.regimen import resolve_plans, resolve_regimens
from cleverly.provenance import fingerprint_array
from cleverly.utils.bounds import bound, expit, logit, shrink_probabilities

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_multivalue as law
from .. import discrete_law_longitudinal_policy as policy
from .. import discrete_law_survival as survival
from .. import longitudinal_policies as policies
from ..studies.canonical_ltmle import QuasiBinomialGLM

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
ROWS = law.first_row_of()
#: How far a mutation must move the estimate or the curve to count as caught.
CAUGHT = 1e-4


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": law.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    return settings


def fit_plans(labels: tuple[str, ...] = tuple(policy.PLANS), **overrides: Any) -> Any:
    """A saturated fit of ``labels`` on the exact 512-row law, the first as reference."""
    frame = overrides.pop("frame", None)
    settings = _settings(reference=labels[0], **overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(policies.regimens(labels), **settings).fit(
            law.frame() if frame is None else frame, **COLUMNS
        )


@cache
def eif(name: str) -> np.ndarray:
    return policy.eif_policy(policy.PROBS, name)


def deviation(result: Any, labels: tuple[str, ...]) -> float:
    """The largest error of any reported estimate or curve against the oracle."""
    worst = 0.0
    for name in policy.names(labels):
        worst = max(worst, abs(result.psi(name) - policy.TRUTH[name]))
        worst = max(worst, float(np.max(np.abs(result.influence_curves[name][ROWS] - eif(name)))))
    return worst


@pytest.fixture(scope="module")
def fit() -> Any:
    return fit_plans()


# ------------------------------------------------------------------ T1, T2


def test_every_reported_parameter_has_an_oracle(fit: Any) -> None:
    assert set(fit) == set(policy.NAMES)


@pytest.mark.parametrize("name", policy.NAMES)
def test_point_estimate_is_the_policy_g_formula(fit: Any, name: str) -> None:
    """T1: mixed plans in both orders, and a partial-support policy."""
    assert fit.psi(name) == pytest.approx(policy.TRUTH[name], abs=1e-12)


@pytest.mark.parametrize("name", policy.NAMES)
def test_influence_curve_is_the_gateaux_derivative(fit: Any, name: str) -> None:
    """T2: the integrated curve at every support point."""
    np.testing.assert_allclose(fit.influence_curves[name][ROWS], eif(name), atol=1e-10, rtol=0)


def test_the_partial_policy_drops_exactly_the_rows_it_cannot_draw(fit: Any) -> None:
    """T2, structure: a zero-probability observed arm leaves the plan, and only that arm."""
    data = fit.data
    plan = fit.fits["partial"]
    second = plan.steps[1]
    code = np.nan_to_num(data.treatment[:, 1], nan=0.0).astype(int)
    density = plan.policy[1][np.arange(data.n), code]
    excluded = second.at_risk & (density == 0.0)
    assert excluded.any()
    np.testing.assert_array_equal(second.trained_on, second.at_risk & (density > 0.0))
    assert not np.any(second.clever[excluded])
    assert np.all(second.clever[second.trained_on] > 0.0)


# ------------------------------------------------------------------ T3 witnesses


def test_the_policies_are_not_degenerate() -> None:
    for table in (policy.POLICY1, policy.POLICY2):
        assert np.max(table) <= 0.5
        np.testing.assert_allclose(table.sum(axis=-1), 1.0, atol=0, rtol=0)


def test_the_policies_differ_from_the_mechanism() -> None:
    """A policy equal to ``g`` makes the ratio one and hides the numerator."""
    for table, mechanism in ((policy.POLICY1, law.G1), (policy.POLICY2, law.G2)):
        assert np.max(np.abs(table / mechanism - 1.0)) >= 0.5
        rows = table.reshape(-1, 3)
        assert np.mean(np.any(rows != mechanism.reshape(-1, 3), axis=1)) >= 0.5


def test_the_second_policy_reads_the_earlier_arm() -> None:
    assert any(
        not np.array_equal(policy.POLICY2[w, a1, l2], policy.POLICY2[w, b1, l2])
        for w in range(2)
        for l2 in range(2)
        for a1 in range(3)
        for b1 in range(3)
    )


def test_a_label_permutation_moves_the_estimand() -> None:
    order = [1, 2, 0]
    permuted = (("policy", policy.POLICY1[:, order]), ("policy", policy.POLICY2[..., order]))
    moved = policy.functional_policy(policy.PROBS, permuted)
    assert abs(moved - policy.TRUTH["ey_regimen[mix]"]) > 0.01


@pytest.mark.parametrize("label", ["mix", "taper"])
def test_each_policy_mean_is_no_deterministic_mean(label: str) -> None:
    psi = policy.TRUTH[f"ey_regimen[{label}]"]
    for mean in (*policy.deterministic_means().values(), policy.observed_mean()):
        assert abs(psi - mean) > 0.01


def test_the_ratio_numerator_and_the_marginal_are_nonzero_terms(fit: Any) -> None:
    mix = fit.fits["mix"]
    for step in mix.steps:
        rows = step.trained_on
        selected = 1.0 / mix.cumulative[rows, step.time - 1]
        assert np.max(np.abs(step.clever[rows] - selected)) > 0.1
        assert np.max(np.abs(step.value[step.at_risk] - step.targeted[step.at_risk])) > 0.01


# ------------------------------------------------------------------ T4 mutations


@contextmanager
def _patched(target: Any, name: str, replacement: Any) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        setattr(target, name, original)


def _wrap_step(change: Callable[[Any], Any]) -> Callable[..., Any]:
    original = sequential._targeted_step

    def mutated(*args: Any, **kwargs: Any) -> Any:
        return change(original(*args, **kwargs))

    return mutated


def _numerator_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """M1: the curve and the loss weight read ``1 / g``, the selected-column control."""
    original = sequential._clever_covariate

    def mutated(at_risk: Any, trained_on: Any, cumulative: Any, time: int, numerator: Any = None):  # type: ignore[no-untyped-def]
        return original(at_risk, trained_on, cumulative, time)

    monkeypatch.setattr(sequential, "_clever_covariate", mutated)


def _marginal_dropped_at(time: int) -> Callable[[pytest.MonkeyPatch], None]:
    """M2: node ``time`` carries the observed-arm prediction instead of the marginal."""

    def apply(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sequential,
            "_targeted_step",
            _wrap_step(
                lambda step: dataclasses.replace(step, marginal=None) if step.time == time else step
            ),
        )

    return apply


def _numerator_not_cumulative(monkeypatch: pytest.MonkeyPatch) -> None:
    """M3: the numerator is the current node's density, not the running product."""

    def mutated(self: Any, data: Any) -> Any:
        if not self.has_policy:
            return None
        return self.intervention_density(data)

    monkeypatch.setattr(regimen_module.Plan, "cumulative_numerator", mutated)


def _residual_reads_marginal(monkeypatch: pytest.MonkeyPatch) -> None:
    """M4: the residual reads the marginal instead of the observed-arm prediction."""
    monkeypatch.setattr(
        sequential,
        "_targeted_step",
        _wrap_step(
            lambda step: (
                step if step.marginal is None else dataclasses.replace(step, targeted=step.marginal)
            )
        ),
    )


def _frame_without_earlier_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """M5: the policy frame hands every row the first level as its earlier arm."""
    original = LongitudinalData.policy_frame

    def mutated(self: Any, time: int) -> Any:
        frame = original(self, time)
        for name in self.treatment_names[: time - 1]:
            frame[name] = self.treatment_levels[0][0]
        return frame

    monkeypatch.setattr(LongitudinalData, "policy_frame", mutated)


def _columns_not_reordered(monkeypatch: pytest.MonkeyPatch) -> None:
    """M6: a policy frame's columns are read in their own order, not by label."""
    monkeypatch.setattr(
        regimen_module,
        "_frame_by_level",
        lambda frame, levels, *, label, name: np.asarray(frame, dtype=float),
    )


def _numerator_at_the_reference_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """M7: the ratio numerator reads the policy at level 0, not at the residual's arm."""

    def mutated(self: Any, data: Any) -> Any:
        if not self.has_policy:
            return None
        columns = []
        for time in range(1, data.n_times + 1):
            if self.is_policy_node(time):
                present = ~np.isnan(data.treatment[:, time - 1])
                columns.append(np.where(present, self.policy_at(time)[:, 0], 0.0))
            else:
                columns.append(regimen_module._node_numerator(self, data, time))
        return np.cumprod(np.column_stack(columns), axis=1)

    monkeypatch.setattr(regimen_module.Plan, "cumulative_numerator", mutated)


MUTATIONS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], tuple[str, ...]]] = {
    "M1 numerator dropped": (_numerator_dropped, ("low", "mix")),
    "M2 marginal dropped at node 2": (_marginal_dropped_at(2), ("low", "mix")),
    "M2 marginal dropped at node 1": (_marginal_dropped_at(1), ("low", "mix")),
    "M3 numerator not cumulative": (_numerator_not_cumulative, ("low", "mix")),
    "M4 residual reads the marginal": (_residual_reads_marginal, ("low", "mix")),
    "M5 frame without the earlier arm": (_frame_without_earlier_arms, ("low", "mix")),
    "M6 columns not reordered by label": (_columns_not_reordered, ("low", "mix")),
    "M7 numerator at the wrong arm": (_numerator_at_the_reference_arm, ("low", "partial")),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_each_mutation_moves_the_estimate_or_the_curve(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply, labels = MUTATIONS[mutation]
    assert deviation(fit_plans(labels), labels) < 1e-10
    apply(monkeypatch)
    assert deviation(fit_plans(labels), labels) > CAUGHT


# ------------------------------------------------------------------ T5 remainder


def test_the_second_order_remainder_is_nonzero_and_quadratic() -> None:
    name = "ey_regimen[mix]"
    direction = np.zeros_like(policy.PROBS)
    direction[8] = 1.0
    direction -= policy.PROBS

    def remainder(epsilon: float) -> float:
        moved = policy.PROBS + epsilon * direction
        return float(
            policy.functional(moved, name)
            - policy.functional(policy.PROBS, name)
            + np.dot(policy.PROBS, policy.eif_policy(moved, name))
        )

    coarse = remainder(0.0002)
    fine = remainder(0.0001)
    assert abs(fine) > 1e-8
    assert coarse / fine == pytest.approx(4.0, rel=0.03)


# ------------------------------------------------------------------ T5b targeting witness

ALPHA = 0.9995


def _misspecified() -> Any:
    return fit_plans(("mix",), outcome_learner=QuasiBinomialGLM(), pseudo_learner=DummyRegressor())


def _longhand_epsilon(step: Any, weights: np.ndarray) -> float:
    rows = step.trained_on
    offset = logit(shrink_probabilities(step.initial[rows], ALPHA))
    outcome = step.pseudo_outcome[rows]
    weight = (weights * step.clever)[rows]

    def score(epsilon: float) -> float:
        return float(np.sum(weight * (outcome - expit(offset + epsilon))))

    return float(brentq(score, -5.0, 5.0, xtol=1e-14))


def _targeting_errors(result: Any) -> dict[str, float]:
    """How far the misspecified fit is from the targeting it claims to have done."""
    mix = result.fits["mix"]
    weights = result.data.weights
    errors = {"epsilon": 0.0, "marginal": 0.0, "observed": 0.0}
    for step in mix.steps:
        epsilon = float(step.fluctuation.epsilon[0])
        errors["epsilon"] = max(errors["epsilon"], abs(epsilon - _longhand_epsilon(step, weights)))
        rows = step.at_risk
        moved = bound(
            expit(logit(shrink_probabilities(step.initial_by_arm[rows], ALPHA)) + epsilon),
            1.0 - ALPHA,
            ALPHA,
        )
        longhand = np.sum(mix.policy[step.time - 1][rows] * moved, axis=1)
        errors["marginal"] = max(
            errors["marginal"], float(np.max(np.abs(step.value[rows] - longhand)))
        )
        codes = mix.assignment[rows, step.time - 1].astype(int)
        selected = step.targeted_by_arm[rows][np.arange(rows.sum()), codes]
        errors["observed"] = max(
            errors["observed"], float(np.max(np.abs(step.targeted[rows] - selected)))
        )
    errors["score"] = abs(float(np.mean(result.influence_curves["ey_regimen[mix]"])))
    return errors


def test_the_fluctuation_moves_every_arm_and_solves_the_ratio_score() -> None:
    """T5b: nonzero coefficients, a longhand solve, the per-arm update, ``P_n D* = 0``."""
    result = _misspecified()
    for step in result.fits["mix"].steps:
        assert abs(float(step.fluctuation.epsilon[0])) > 0.05
    errors = _targeting_errors(result)
    assert errors["epsilon"] < 1e-10
    assert errors["marginal"] < 1e-12
    assert errors["observed"] == 0.0
    assert errors["score"] < 1e-10


def test_m8_a_marginal_of_the_initial_fit_fails_the_targeting_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M8: the node carries the policy mean of the *initial* per-arm predictions."""
    original = sequential._targeted_step

    def mutated(node: Any, fluctuation: Any, *, regression_target: Any) -> Any:
        step = original(node, fluctuation, regression_target=regression_target)
        if node.policy is None:
            return step
        carried = np.where(node.at_risk, np.sum(node.policy * node.initial_by_arm, axis=1), 0.5)
        return dataclasses.replace(step, marginal=carried)

    monkeypatch.setattr(sequential, "_targeted_step", mutated)
    assert _targeting_errors(_misspecified())["marginal"] > CAUGHT


def test_m9_a_loss_weight_without_the_numerator_leaves_the_score_unsolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M9: the fluctuation is weighted by ``w / g`` while the curve keeps the ratio."""
    original = sequential._clever_covariate

    def mutated(at_risk: Any, trained_on: Any, cumulative: Any, time: int, numerator: Any = None):  # type: ignore[no-untyped-def]
        counterfactual, _ = original(at_risk, trained_on, cumulative, time)
        _, clever = original(at_risk, trained_on, cumulative, time, numerator=numerator)
        return counterfactual, clever

    monkeypatch.setattr(sequential, "_clever_covariate", mutated)
    errors = _targeting_errors(_misspecified())
    assert max(errors["score"], errors["epsilon"]) > CAUGHT


# ------------------------------------------------------------------ T6 point masses


def _one_hot(codes: Callable[[Any], np.ndarray], levels: tuple[str, ...]) -> Callable[[Any], Any]:
    def density(frame: Any) -> np.ndarray:
        chosen = codes(frame)
        return np.column_stack([(chosen == level).astype(float) for level in levels])

    return density


SORTED = ("high", "low", "standard")


def _first_rule(frame: Any) -> np.ndarray:
    return np.where(np.asarray(frame["W"]) == 0, "high", "low")


def _second_rule(frame: Any) -> np.ndarray:
    return np.where(np.asarray(frame["L2"]) == 1, "high", "low")


def _point_mass_regimen() -> DynamicRegimen:
    return DynamicRegimen(
        "plan",
        (
            Stochastic(_one_hot(_first_rule, SORTED), "p1", density_kind="known"),
            Stochastic(_one_hot(_second_rule, SORTED), "p2", density_kind="known"),
        ),
    )


def _rule_regimen() -> DynamicRegimen:
    return DynamicRegimen("plan", (_first_rule, _second_rule), rule_kind="known")


def _same_steps(left: Any, right: Any) -> None:
    for a, b in zip(left.steps, right.steps, strict=True):
        for item in dataclasses.fields(a):
            if item.name == "fluctuation":
                np.testing.assert_array_equal(a.fluctuation.epsilon, b.fluctuation.epsilon)
                continue
            first, second = getattr(a, item.name), getattr(b, item.name)
            if isinstance(first, np.ndarray):
                assert np.array_equal(first, second), item.name
            else:
                assert first == second, item.name


@pytest.mark.parametrize("folds", [1, 2])
def test_a_one_hot_policy_is_its_rule_bit_for_bit(folds: int) -> None:
    """T6a: a collapsed point mass runs the rule's code, at one and at two folds."""

    def run(plan: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE({"plan": plan}, **_settings(n_folds=folds, random_state=3)).fit(
                law.frame(), **COLUMNS
            )

    stochastic, rule = run(_point_mass_regimen()), run(_rule_regimen())
    name = "ey_regimen[plan]"
    assert stochastic.psi(name) == rule.psi(name)
    assert np.array_equal(stochastic.influence_curves[name], rule.influence_curves[name])
    left, right = stochastic.fits["plan"], rule.fits["plan"]
    assert np.array_equal(left.cumulative, right.cumulative)
    assert left.policy == () and left.cumulative_numerator is None
    _same_steps(left, right)
    assert stochastic.config.plan_fingerprints == rule.config.plan_fingerprints
    assert stochastic.config.policy_point_mass_nodes == (("plan", (1, 2)),)


def test_a_one_hot_survival_policy_is_its_rule_bit_for_bit() -> None:
    def run(plan: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE(
                {"plan": plan},
                outcome_learner=survival.CellMeans(),
                pseudo_learner=survival.CellMeans(),
                treatment_learner=survival.CellMeans(),
                censoring_learner=survival.CellMeans(),
                n_folds=1,
                g_bounds=NO_TRUNCATION,
                simultaneous=False,
            ).fit(survival.frame(), **SURVIVAL_COLUMNS)

    point = DynamicRegimen(
        "plan",
        (
            Stochastic(lambda h: np.tile([0.0, 1.0], (len(h), 1)), "p1", density_kind="known"),
            Stochastic(
                lambda h: np.column_stack([1.0 - np.asarray(h["L2"]), np.asarray(h["L2"])]),
                "p2",
                density_kind="known",
            ),
        ),
    )
    rule = DynamicRegimen("plan", (1, lambda h: h["L2"]), rule_kind="known")
    stochastic, deterministic = run(point), run(rule)
    for name in stochastic.estimates:
        assert np.array_equal(
            stochastic.influence_curves[name], deterministic.influence_curves[name]
        )
        assert stochastic.psi(name) == deterministic.psi(name)


def test_a_density_one_ulp_off_one_hot_does_not_collapse() -> None:
    data = LongitudinalData.from_frame(law.frame(), **COLUMNS)

    def almost(frame: Any) -> np.ndarray:
        values = np.tile([0.9999999, 1e-7, 0.0], (len(frame), 1))
        return values

    regimens = resolve_regimens(
        {"near": DynamicRegimen("near", (Stochastic(almost, "n", density_kind="known"), "low"))},
        2,
    )
    (plan,) = resolve_plans(regimens, data)
    assert plan.has_policy and plan.point_mass_nodes == ()


@contextmanager
def _no_collapse() -> Iterator[None]:
    original = estimator_module.resolve_plans
    with _patched(
        estimator_module,
        "resolve_plans",
        partial(original, _collapse_point_masses=False),
    ):
        yield


def _constant_point_mass() -> DynamicRegimen:
    return DynamicRegimen(
        "plan",
        (
            Stochastic(_one_hot(_first_rule, SORTED), "p1", density_kind="known"),
            Stochastic(
                _one_hot(lambda frame: np.full(len(frame), "high"), SORTED),
                "p2",
                density_kind="known",
            ),
        ),
    )


@pytest.mark.parametrize("covariate", [False, True], ids=["constant", "reads_L2"])
def test_the_policy_path_on_a_point_mass_is_the_rule_on_the_exact_law(covariate: bool) -> None:
    """T6b (i): with collapse off, the policy machinery reproduces the rule exactly.

    A saturated learner partitions by distinct design row, so the extra arm block of a
    policy node changes no prediction on the exact law.
    """
    point = _point_mass_regimen() if covariate else _constant_point_mass()
    rule = (
        _rule_regimen()
        if covariate
        else DynamicRegimen("plan", (_first_rule, "high"), rule_kind="known")
    )
    with _no_collapse():
        stochastic = LTMLE({"plan": point}, **_settings()).fit(law.frame(), **COLUMNS)
    assert stochastic.fits["plan"].policy != ()
    deterministic = LTMLE({"plan": rule}, **_settings()).fit(law.frame(), **COLUMNS)
    name = "ey_regimen[plan]"
    assert stochastic.psi(name) == pytest.approx(deterministic.psi(name), abs=1e-12)
    np.testing.assert_allclose(
        stochastic.influence_curves[name], deterministic.influence_curves[name], atol=1e-12
    )


def test_the_policy_path_on_a_constant_point_mass_is_the_rule_under_a_glm() -> None:
    """T6b (ii): under a GLM only the constant point mass is compared.

    A covariate-dependent arm block is a nonlinear function of the design, so a GLM fit with
    it differs from one without it, the argument of ``history_design``'s docstring.  A
    constant block is collinear with nothing a GLM can use on the rows that remain on the
    plan, so the two fits agree.
    """
    glm = {"outcome_learner": QuasiBinomialGLM(), "pseudo_learner": QuasiBinomialGLM()}
    with _no_collapse():
        stochastic = LTMLE({"plan": _constant_point_mass()}, **_settings(**glm)).fit(
            law.frame(), **COLUMNS
        )
    rule = DynamicRegimen("plan", (_first_rule, "high"), rule_kind="known")
    deterministic = LTMLE({"plan": rule}, **_settings(**glm)).fit(law.frame(), **COLUMNS)
    name = "ey_regimen[plan]"
    assert stochastic.psi(name) == pytest.approx(deterministic.psi(name), abs=1e-8)


# ------------------------------------------------------------------ T7 recorded randomizer


def test_the_integrated_curve_is_the_projection_of_the_recorded_one(fit: Any) -> None:
    """T7: Theorem 3 as stated on the augmented law, against the integrated estimator."""
    recorded = recorded_fit()
    integrated_name = "ey_regimen[mix]"
    recorded_name = "ey_regimen[mix (recorded)]"
    truth = policy.TRUTH[integrated_name]
    assert fit.psi(integrated_name) == pytest.approx(truth, abs=1e-12)
    assert recorded.psi(recorded_name) == pytest.approx(truth, abs=1e-12)
    pairs = policy.RANDOMIZER_LEVELS**2
    recorded_ic = recorded.influence_curves[recorded_name].reshape(-1, pairs)
    integrated_ic = fit.influence_curves[integrated_name]
    np.testing.assert_allclose(recorded_ic.mean(axis=1), integrated_ic, atol=1e-10, rtol=0)
    assert np.mean(integrated_ic**2) < 0.99 * np.mean(recorded_ic**2)


@cache
def recorded_fit() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE({"mix (recorded)": policies.recorded_regimen()}, **_settings()).fit(
            policy.augmented_frame(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[["E1"], ["L2", "E2"]],
        )


# ------------------------------------------------------------------ T9 survival, competing

SURVIVAL_COLUMNS: dict[str, Any] = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "censoring": ["C1", "C2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}


def _survival_fit(module: Any, outcome: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"never": 0, "draw": policies.survival_regimen()},
            outcome_learner=module.CellMeans(),
            pseudo_learner=module.CellMeans(),
            treatment_learner=module.CellMeans(),
            censoring_learner=module.CellMeans(),
            n_folds=1,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
        ).fit(module.frame(), **{**SURVIVAL_COLUMNS, "outcome": outcome})


def _survival_deviation() -> float:
    result = _survival_fit(survival, ["Y1", "Y2"])
    rows = survival.first_row_of()
    worst = 0.0
    for horizon in survival.HORIZONS:
        name = f"risk_regimen[draw @ t={horizon}]"
        function = partial(policy.functional_policy_survival, horizon=horizon)
        worst = max(worst, abs(result.psi(name) - function(survival.PROBS)))
        reported = result.influence_curves[name][rows]
        worst = max(worst, float(np.max(np.abs(reported - policy.eif(function, survival.PROBS)))))
    return worst


def test_a_survival_policy_is_the_policy_cumulative_risk() -> None:
    """T9: both horizons, with censoring at both nodes."""
    assert _survival_deviation() < 1e-10


def test_dropping_the_censoring_factor_fails_the_survival_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def without_censoring(self: Any, data: Any, plan: Any, bounds: Any) -> Any:
        lower, upper = bounds
        running = np.ones(data.n)
        raw, bounded = [], []
        for time in range(1, data.n_times + 1):
            running = running * self.treatment[time - 1][plan.label]
            raw.append(running)
            bounded.append(bound(running, lower, upper))
        return np.column_stack(raw), np.column_stack(bounded)

    monkeypatch.setattr(sequential.Mechanism, "cumulative_with_unbounded", without_censoring)
    assert _survival_deviation() > CAUGHT


@pytest.mark.parametrize("cause", competing.CAUSES)
def test_a_competing_risk_policy_is_the_policy_incidence(cause: str) -> None:
    result = _survival_fit(competing, competing.outcome_columns())
    rows = competing.first_row_of()
    for horizon in competing.HORIZONS:
        name = f"cif_regimen[draw, {cause} @ t={horizon}]"
        function = partial(policy.functional_policy_competing, cause=cause, horizon=horizon)
        assert result.psi(name) == pytest.approx(function(competing.PROBS), abs=1e-12)
        np.testing.assert_allclose(
            result.influence_curves[name][rows],
            policy.eif(function, competing.PROBS),
            atol=1e-10,
            rtol=0,
        )


# ------------------------------------------------------------------ T10 weights


def _weight(point: tuple[int, ...]) -> float:
    _, a1, l2, a2, y = point
    return 1.0 + 0.5 * (a1 == 1) + 0.3 * l2 + 0.4 * (a2 == 2) + 0.8 * y


CELL_WEIGHTS = np.array([_weight(point) for point in law.SUPPORT])


def _tilted(name: str, probs: Any) -> Any:
    tilted = probs * CELL_WEIGHTS / np.sum(probs * CELL_WEIGHTS)
    return policy.functional(tilted, name)


@pytest.mark.parametrize("name", policy.names(("low", "mix", "taper")))
def test_a_weighted_policy_fit_is_the_tilted_policy_parameter(name: str) -> None:
    """T10: every nuisance and score is weighted, so the fit answers for ``P_w``."""
    labels = ("low", "mix", "taper")
    rows = np.repeat(np.arange(len(law.SUPPORT)), law.COUNTS)
    frame = law.frame().assign(w=CELL_WEIGHTS[rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(policies.regimens(labels), **_settings(reference="low")).fit(
            frame, weights="w", **COLUMNS
        )
    function = partial(_tilted, name)
    assert abs(function(policy.PROBS) - policy.TRUTH[name]) > 1e-3
    assert result.psi(name) == pytest.approx(function(policy.PROBS), abs=1e-12)
    np.testing.assert_allclose(
        result.influence_curves[name][ROWS], policy.eif(function, policy.PROBS), atol=1e-10
    )


def test_fingerprint_helper_is_the_values_digest_without_a_policy() -> None:
    data = LongitudinalData.from_frame(law.frame(), **COLUMNS)
    (plan,) = resolve_plans(resolve_regimens({"low": "low"}, 2), data)
    assert estimator_module._plan_fingerprint(plan) == fingerprint_array(plan.values)
