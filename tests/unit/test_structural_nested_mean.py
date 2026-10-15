"""The structural nested mean estimator: exact laws, curves, mutations and refusals.

Each exact law of :mod:`tests.discrete_law_structural` is a finite table with the exact
probabilities as weights.  A saturated learner then returns the population conditional means,
so the fit returns the population solution.  The truths come from an independent g-formula
enumeration.

* W1 compares the coefficients, the reference mean, the regimen levels and the contrasts with
  the truth on laws S2, S3, K, C, H and P.
* W3 differentiates each second-stage estimator in the observation weights, with the nuisances
  held fixed, and compares the derivative with the influence curve.
* W4 gives the population limit of the structural fit on two laws where the declared features
  are too few, in closed form.
* W5 fits law M, where the second treatment reads a covariate the period state omits.
* M1 to M7 each break one line of the estimator and require a witness to fail with the limit
  that the broken line implies.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import cleverly
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.inference.cluster import influence_covariance
from cleverly.inference.multiplier import simultaneous_bands
from cleverly.interventions import Shift, Stochastic
from cleverly.learners import GLM
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import (
    LTMLE,
    DynamicRegimen,
    LongitudinalData,
    Repeated,
    StructuralNestedMean,
    StructuralNestedMeanResult,
    snmm,
)
from cleverly.longitudinal import structural as module

from .. import discrete_law_structural as laws
from ._source_mutation import mutate

#: Saturated learners for the exact laws: every product of the binary and few-valued columns.
OUTCOME = GLM(family="gaussian", interactions=8, tol=1e-13)
TREATMENT = GLM(interactions=8, tol=1e-13)
EXACT = 1e-9


def _fit(
    law: laws.Law,
    features: dict[Any, Any] | None = None,
    regimens: Any = None,
    **settings: Any,
) -> StructuralNestedMeanResult:
    settings = {
        "outcome_learner": OUTCOME,
        "treatment_learner": TREATMENT,
        "n_folds": 1,
        "simultaneous": False,
        **settings,
    }
    return StructuralNestedMean(regimens, blip_features=features, **settings).fit(
        laws.frame(law),
        outcome="Y",
        treatment=law.treatments,
        baseline=law.baseline,
        time_varying=law.time_varying,
        weights="p",
        family="gaussian",
        continuous_treatment=list(law.doses),
    )


def _regimens(law: laws.Law) -> dict[str, list[float]]:
    return {
        "r" + "".join(f"{value:g}" for value in assignment).replace("-", "m"): list(assignment)
        for assignment in laws.static_assignments(law)
    }


#: The coefficients each exact law declares, node by node, as ``(features, rows)``.  A row of
#: ``rows`` lists one coefficient per feature for one coding component.  Written by hand from
#: the structural equations of each law and checked against the g-formula below.
TRUTH: dict[str, dict[str, Any]] = {
    "S2": {"features": None, "psi": [[[1.28]], [[0.5]]]},
    "S3": {"features": None, "psi": [[[1.15]], [[0.95]], [[0.6]]]},
    "K": {
        "features": None,
        "psi": [
            [[1.0 + 0.7 * laws.K_SHIFT[1], 0.4 + 0.7 * laws.K_SHIFT[2]]],
            [[laws.K_EFFECT_2[1], laws.K_EFFECT_2[2]]],
        ],
    },
    "C": {"features": None, "psi": [[[0.97]], [[0.6]]]},
    "H": {
        "features": {1: ["W"], 2: ["W", "A1"]},
        "psi": [[[1.14], [0.5]], [[0.5], [0.4], [0.6]]],
    },
    "P": {"features": {2: ["A1"]}, "psi": [[[1.28]], [[0.5], [1.5]]]},
}


def _phi_of(features: dict[Any, Any] | None, node: int) -> Callable[[dict[str, float]], np.ndarray]:
    terms = [] if not features else list(features.get(node, []))

    def phi(history: dict[str, float]) -> np.ndarray:
        return np.array([1.0, *(history[name] for name in terms)])

    return phi


def _coding_of(law: laws.Law, node: int) -> Callable[[float], np.ndarray]:
    treatment = law.treatments[node - 1]
    if treatment in law.doses:
        return lambda level: np.array([level])
    levels = sorted({path[treatment] for path, _ in laws.paths(law)})
    return lambda level: np.array([1.0 * (level == other) for other in levels[1:]])


def _levels_of(law: laws.Law, node: int) -> list[float]:
    treatment = law.treatments[node - 1]
    return sorted({path[treatment] for path, _ in laws.paths(law)})


@pytest.mark.parametrize("name", list(TRUTH))
def test_the_declared_coefficients_are_the_blips_of_the_law(name: str) -> None:
    """The hand-written truth table satisfies the structural model, by the g-formula."""
    law = laws.LAWS[name]()
    declared = TRUTH[name]
    for node in range(1, law.n_nodes + 1):
        coefficients = np.array(declared["psi"][node - 1])
        gap = laws.structural_blip_errors(
            law,
            node,
            _levels_of(law, node),
            _phi_of(declared["features"], node),
            coefficients,
            _coding_of(law, node),
        )
        assert gap < EXACT, (name, node, gap)


def _blip_vector(result: StructuralNestedMeanResult, node: int) -> np.ndarray:
    return np.asarray(result.blip(node))


@pytest.mark.parametrize("name", list(TRUTH))
def test_w1_exact_law_returns_the_truth(name: str) -> None:
    """W1: coefficients, E[Y(0)], every regimen level and every contrast equal the g-formula."""
    law = laws.LAWS[name]()
    declared = TRUTH[name]
    regimens = _regimens(law)
    first = next(iter(regimens))
    result = _fit(law, declared["features"], regimens, reference=first)
    for node in range(1, law.n_nodes + 1):
        assert np.allclose(
            _blip_vector(result, node), np.array(declared["psi"][node - 1]), atol=EXACT
        )
    assert abs(result.psi("ey_reference") - laws.reference_mean(law)) < EXACT
    truth = {label: laws.regimen_mean(law, values) for label, values in regimens.items()}
    for label, value in truth.items():
        assert abs(result.psi(f"ey_regimen[{label}]") - value) < EXACT, label
        if label != first:
            name_ = f"ate_regimen[{label} vs {first}]"
            assert abs(result.psi(name_) - (value - truth[first])) < EXACT, name_
    assert result.inference_status == "influence_curve"


def test_w1_levels_are_named_by_the_users_labels() -> None:
    law = laws.law_k()
    result = _fit(law, None, {"a": [1.0, 2.0], "b": [0.0, 0.0]})
    assert [name for name in result if name.startswith("blip")] == [
        "blip[A1:1.0:intercept]",
        "blip[A1:2.0:intercept]",
        "blip[A2:1.0:intercept]",
        "blip[A2:2.0:intercept]",
    ]
    assert "ate_regimen[b vs a]" in result


def test_w1_static_dose_level_at_an_interior_dose() -> None:
    law = laws.law_c()
    result = _fit(law, None, {"inner": [0.5, 0.25], "zero": [0.0, 0.0]})
    assert abs(result.psi("ey_regimen[inner]") - laws.regimen_mean(law, [0.5, 0.25])) < EXACT


def _expected_mean(law: laws.Law, node: int, target: Callable[[dict[str, float]], float]) -> Any:
    return laws.conditional_mean(law, node, target)


def _context(law: laws.Law, features: dict[Any, Any] | None) -> tuple[Any, Any, Any, Any]:
    """The prepared data, the node layout, the nuisances and the residual stage of a law."""
    frame = laws.frame(law)
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=law.treatments,
        baseline=law.baseline,
        time_varying=law.time_varying,
        weights="p",
        family="gaussian",
        continuous_treatment=list(law.doses),
    )
    nodes = module._parse_blip_features(data, features)
    learners = module._Learners(OUTCOME, TREATMENT, learner_folds=5, random_state=0)
    nuisances = module._fit_nuisances(data, nodes, learners, Folds.single(data.n), 1)
    return data, nodes, nuisances, module._build_stage(data, nodes, nuisances)


@pytest.mark.parametrize("name", ["H", "P", "K"])
def test_w1_product_nuisances_equal_the_conditional_means_of_the_law(name: str) -> None:
    """Every fitted E[Z_j | H_t] equals the g-formula-free conditional mean of Z_j.

    The product nuisance of a feature that reads an earlier treatment is the mean of the
    product, which is not the product of the means.  The independent value is a lookup over
    the cells of the law.
    """
    law = laws.LAWS[name]()
    features = TRUTH[name]["features"]
    _, nodes, nuisances, _ = _context(law, features)
    frame = laws.frame(law)
    for (t, j), fitted in nuisances.m.items():
        node = nodes[j - 1]
        coding = _coding_of(law, j)
        for k, feature in enumerate(node.features):
            for l in range(node.d):

                def z(
                    path: dict[str, float],
                    feature: Any = feature,
                    l: int = l,
                    j: int = j,
                    coding: Any = coding,
                ) -> float:
                    value = 1.0
                    for factor in feature.factors:
                        if factor.kind == "baseline":
                            value *= path[law.baseline[factor.index]]
                        elif factor.kind == "covariate":
                            value *= path[law.time_varying[factor.node - 1][factor.index]]
                        else:
                            value *= _coding_of(law, factor.node)(
                                path[law.treatments[factor.node - 1]]
                            )[factor.index]
                    return value * coding(path[law.treatments[j - 1]])[l]

                lookup = laws.conditional_mean(law, t, z)
                expected = np.array(
                    [lookup({n: row[n] for n in frame.columns}) for _, row in frame.iterrows()]
                )
                assert np.max(np.abs(fitted[:, k * node.d + l] - expected)) < EXACT, (t, j, k, l)


def test_w1_collinear_features_are_refused() -> None:
    """A feature the history determines at the node adds no treatment variation: refuse."""
    law = laws.law_s2()
    with pytest.raises(CapabilityError) as raised:
        _fit(law, {2: ["A1", ("A1", "A1")]})
    assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION


# ----------------------------------------------------------------------------- W5, law M


def test_w5_the_full_history_removes_the_markov_restriction() -> None:
    """Law M: ``A2`` reads ``S1``, which the period state omits.  The fit still equals the truth."""
    law = laws.law_m()
    result = _fit(law, None, {"all": [1.0, 1.0], "none": [0.0, 0.0], "late": [0.0, 1.0]})
    for label, assignment in {"all": [1, 1], "none": [0, 0], "late": [0, 1]}.items():
        assert abs(result.psi(f"ey_regimen[{label}]") - laws.regimen_mean(law, assignment)) < EXACT
    assert abs(result.psi("blip[A2:1.0:intercept]") - 0.5) < EXACT
    assert abs(result.psi("blip[A1:1.0:intercept]") - 1.28) < EXACT


# --------------------------------------------------------------------------- W3, curves


def _functionals(
    data: Any,
    nodes: Any,
    stage: Any,
    regimens: dict[str, list[float]],
    reference: str,
) -> Callable[[np.ndarray], tuple[dict[str, float], dict[str, np.ndarray]]]:
    inputs = {
        label: module._regimen_inputs(data, nodes, codes) for label, codes in regimens.items()
    }

    def evaluate(weights: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        solution = module._solve(stage, weights)
        values: dict[str, float] = {}
        curves: dict[str, np.ndarray] = {}
        stacked = np.concatenate(solution.psi)
        for index in range(stacked.size):
            values[f"psi[{index}]"] = float(stacked[index])
            curves[f"psi[{index}]"] = solution.psi_curve[:, index]
        values["mu0"] = solution.mu0
        curves["mu0"] = solution.mu0_curve
        levels = {label: module._level_curve(solution, inputs[label]) for label in regimens}
        for label, (value, curve) in levels.items():
            values[f"level[{label}]"] = value
            curves[f"level[{label}]"] = curve
            if label != reference:
                values[f"contrast[{label}]"] = value - levels[reference][0]
                curves[f"contrast[{label}]"] = curve - levels[reference][1]
        return values, curves

    return evaluate


def _curve_gaps(
    law: laws.Law,
    features: dict[Any, Any] | None,
    regimens: dict[str, list[float]],
    step: float = 1e-6,
) -> dict[str, float]:
    """The largest gap, over rows, between ``n dT/dw_i`` and the curve of row ``i``."""
    data, nodes, _, stage = _context(law, features)
    reference = next(iter(regimens))
    evaluate = _functionals(data, nodes, stage, regimens, reference)
    weights = np.asarray(data.weights, dtype=float)
    values, curves = evaluate(weights)
    gaps = dict.fromkeys(values, 0.0)
    for row in range(data.n):
        up, down = weights.copy(), weights.copy()
        up[row] += step
        down[row] -= step
        high, _ = evaluate(up)
        low, _ = evaluate(down)
        for key in values:
            derivative = data.n * (high[key] - low[key]) / (2.0 * step)
            gaps[key] = max(gaps[key], abs(derivative - curves[key][row] / weights[row]))
    return gaps


W3_CASES: dict[str, dict[str, list[float]]] = {
    "H": {"r00": [0.0, 0.0], "r11": [1.0, 1.0], "r10": [1.0, 0.0], "r01": [0.0, 1.0]},
    "K": {"r00": [0.0, 0.0], "r12": [1.0, 2.0], "r21": [2.0, 1.0], "r22": [2.0, 2.0]},
}


@pytest.mark.parametrize("name", list(W3_CASES))
def test_w3_curves_are_the_derivative_of_the_estimator_in_the_weights(name: str) -> None:
    """W3: the curve of every parameter is n times the derivative in its row's weight."""
    gaps = _curve_gaps(laws.LAWS[name](), TRUTH[name]["features"], W3_CASES[name])
    assert max(gaps.values()) < 1e-6, {k: v for k, v in gaps.items() if v >= 1e-6}


def test_w3_the_check_is_not_vacuous() -> None:
    """The comparison would see a curve that is off by a nonzero term."""
    law = laws.law_h()
    data, nodes, _, stage = _context(law, TRUTH["H"]["features"])
    evaluate = _functionals(data, nodes, stage, W3_CASES["H"], "r00")
    _, curves = evaluate(np.asarray(data.weights, dtype=float))
    assert np.max(np.abs(curves["level[r11]"])) > 0.1
    assert np.max(np.abs(curves["contrast[r11]"] - curves["level[r11]"])) > 1e-3


# --------------------------------------------------------------------------- W4, limits


def _scalar_limit(law: laws.Law, node: int, treatment: str) -> tuple[float, float]:
    """The ratio ``E[Z (Y - q)] / E[Z^2]`` at the last node, with ``Z = A - E[A | H]``.

    Returns the limit and ``E[Z^2]``.  At the last node ``Y - q = beta(H) Z``, so the limit is
    a variance-weighted average of the true blip.
    """
    mean_a = laws.conditional_mean(law, node, lambda p: p[treatment])
    mean_y = laws.conditional_mean(law, node, lambda p: law.outcome(p))
    numerator = laws.moment(
        law, lambda p: (p[treatment] - mean_a(p)) * (law.outcome(p) - mean_y(p))
    )
    denominator = laws.moment(law, lambda p: (p[treatment] - mean_a(p)) ** 2)
    return numerator / denominator, denominator


def test_w4_baseline_only_features_on_law_p_have_a_closed_form_limit() -> None:
    """The node-2 coefficient is a variance-weighted blip: 0.5 + 1.5 E[A1 v] / E[v]."""
    law = laws.law_p()
    result = _fit(law, None, {"r01": [0.0, 1.0], "r00": [0.0, 0.0]}, reference="r00")
    mean_a = laws.conditional_mean(law, 2, lambda p: p["A2"])
    weight_total = laws.moment(law, lambda p: mean_a(p) * (1.0 - mean_a(p)))
    weighted_a1 = laws.moment(law, lambda p: p["A1"] * mean_a(p) * (1.0 - mean_a(p)))
    limit = 0.5 + 1.5 * weighted_a1 / weight_total
    assert abs(result.psi("blip[A2:1.0:intercept]") - limit) < EXACT
    assert abs(result.psi("ate_regimen[r01 vs r00]") - limit) < EXACT
    # The regimen truth is 0.5.  The structural contrast misses it by the closed-form amount.
    assert abs(laws.regimen_mean(law, [0, 1]) - laws.regimen_mean(law, [0, 0]) - 0.5) < EXACT
    assert abs(result.psi("ate_regimen[r01 vs r00]") - 0.5) > 0.05
    assert abs(limit - _scalar_limit(law, 2, "A2")[0]) < EXACT


def test_w4_an_undeclared_covariate_blip_has_a_closed_form_limit() -> None:
    """Law L with phi = 1: the node-2 coefficient is 0.5 + 0.8 E[S2 v] / E[v]."""
    law = laws.law_l()
    result = _fit(law, None, {"r01": [0.0, 1.0], "r00": [0.0, 0.0]}, reference="r00")
    mean_a = laws.conditional_mean(law, 2, lambda p: p["A2"])
    weight_total = laws.moment(law, lambda p: mean_a(p) * (1.0 - mean_a(p)))
    weighted_s2 = laws.moment(law, lambda p: p["S2"] * mean_a(p) * (1.0 - mean_a(p)))
    limit = 0.5 + laws.L_SLOPE * weighted_s2 / weight_total
    assert abs(result.psi("blip[A2:1.0:intercept]") - limit) < EXACT
    truth = laws.regimen_mean(law, [0, 1]) - laws.regimen_mean(law, [0, 0])
    assert abs(result.psi("ate_regimen[r01 vs r00]") - truth) > 0.05


def test_w4_the_declared_covariate_blip_is_recovered() -> None:
    """Law L with phi_2 = (1, S2): the blips and E[Y(0)] equal the truth."""
    law = laws.law_l()
    result = _fit(law, {2: ["S2"]})
    assert abs(result.psi("blip[A2:1.0:intercept]") - 0.5) < EXACT
    assert abs(result.psi("blip[A2:1.0:S2]") - laws.L_SLOPE) < EXACT
    assert abs(result.psi("blip[A1:1.0:intercept]") - 1.28) < EXACT
    assert abs(result.psi("ey_reference") - laws.reference_mean(law)) < EXACT


# ------------------------------------------------------------------ mutation controls


def test_m1_dropping_the_later_blip_adjustment_misses_the_truth_by_its_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1: node t then regresses the unadjusted residual.  Law S3 has an A1 -> A2 -> A3 path."""
    law = laws.law_s3()
    declared = np.array([1.15, 0.95, 0.6])
    clean = _fit(law)
    mutate(
        monkeypatch,
        module,
        module,
        "_g_estimation",
        [("adjusted = adjusted - z_res[(t, j)] @ psi[j - 1]", "pass")],
    )
    broken = _fit(law)
    got = np.array([broken.psi(f"blip[A{t}:1.0:intercept]") for t in (1, 2, 3)])
    limits = [_scalar_limit(law, t, f"A{t}")[0] for t in (1, 2, 3)]
    assert np.allclose(got, limits, atol=EXACT)
    assert np.max(np.abs(got - declared)) > 0.02
    assert abs(got[2] - declared[2]) < EXACT, "the last node has no later blip to drop"
    assert np.allclose(
        [clean.psi(f"blip[A{t}:1.0:intercept]") for t in (1, 2, 3)], declared, atol=EXACT
    )


def test_m2_a_period_state_treatment_nuisance_misses_by_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2: ``E[A2 | W, S2, A1]`` in place of ``E[A2 | H_2]``.  Law M confounds ``A2`` by ``S1``."""
    law = laws.law_m()
    real = module._treatment_design

    def period_state(data: Any, time: int) -> np.ndarray:
        design = real(data, time)
        # The history design at node 2 is [W, S1, S2, A1].  The state omits S1.
        return design[:, [0, 2, 3]] if time == 2 else design

    monkeypatch.setattr(module, "_treatment_design", period_state)
    broken = _fit(law)
    mean_full = laws.conditional_mean(law, 2, lambda p: p["A2"])
    mean_state = laws.treatment_state_mean(law, 2, ["W", "S2", "A1"])
    variance = laws.moment(law, lambda p: (p["A2"] - mean_full(p)) ** 2)
    shortfall = laws.moment(law, lambda p: (mean_full(p) - mean_state(p)) ** 2)
    limit = 0.5 * variance / (variance + shortfall)
    assert shortfall > 1e-3
    assert abs(broken.psi("blip[A2:1.0:intercept]") - limit) < EXACT
    assert abs(limit - 0.5) > 0.01


def test_m3_the_wrong_reference_moves_the_level_blips_and_the_reference_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3: a one-hot against the last level.  Regimen levels are invariant to the reference."""
    law = laws.law_k()
    regimens = {"r12": [1.0, 2.0], "r00": [0.0, 0.0]}
    clean = _fit(law, None, regimens)
    mutate(
        monkeypatch,
        module,
        module,
        "_encode",
        [
            (
                "return arm_indicators(codes, n_levels)",
                "return arm_indicators(n_levels - 1 - np.asarray(codes), n_levels)",
            )
        ],
    )
    broken = _fit(law, None, regimens)
    assert abs(broken.psi("ey_regimen[r12]") - clean.psi("ey_regimen[r12]")) < EXACT
    assert abs(broken.psi("ey_regimen[r00]") - clean.psi("ey_regimen[r00]")) < EXACT
    assert abs(broken.psi("ey_reference") - clean.psi("ey_reference")) > 0.05
    assert abs(broken.psi("blip[A1:1.0:intercept]") - clean.psi("blip[A1:1.0:intercept]")) > 0.05


def test_m4_transposed_jacobian_blocks_fail_the_curve_check_on_a_vector_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M4: the off-diagonal blocks transposed, as the EconML covariance does for a vector node."""
    assert max(_curve_gaps(laws.law_k(), None, W3_CASES["K"]).values()) < 1e-6
    mutate(
        monkeypatch,
        module,
        module,
        "_jacobian",
        [
            (
                "block = (own * w[:, None]).T @ z_res[(t, j)] / n",
                "block = ((own * w[:, None]).T @ z_res[(t, j)] / n)"
                ".T if j > t else (own * w[:, None]).T @ z_res[(t, j)] / n",
            )
        ],
    )
    broken = _curve_gaps(laws.law_k(), None, W3_CASES["K"])
    assert max(broken.values()) > 1e-3


def test_m4_a_scalar_law_cannot_see_the_transpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """With scalar nodes the blocks are 1 x 1, so the transpose changes nothing."""
    mutate(
        monkeypatch,
        module,
        module,
        "_jacobian",
        [
            (
                "block = (own * w[:, None]).T @ z_res[(t, j)] / n",
                "block = ((own * w[:, None]).T @ z_res[(t, j)] / n)"
                ".T if j > t else (own * w[:, None]).T @ z_res[(t, j)] / n",
            )
        ],
    )
    cases = {"r00": [0.0, 0.0], "r11": [1.0, 1.0]}
    assert max(_curve_gaps(laws.law_s2(), None, cases).values()) < 1e-6


def test_m5_dropping_the_feature_mean_term_fails_the_contrast_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M5: the ``w (phi - phibar)`` term of the level curve, which is nonzero on law H."""
    mutate(
        monkeypatch,
        module,
        module,
        "_level_curve",
        [("curve = curve + (w[:, None] * (phi - mean_phi)) @ (blip @ coding)", "pass")],
    )
    gaps = _curve_gaps(laws.law_h(), TRUTH["H"]["features"], W3_CASES["H"])
    assert gaps["contrast[r11]"] > 1e-3
    assert gaps["level[r11]"] > 1e-3
    assert gaps["psi[0]"] < 1e-6, "the coefficient curves do not read the level curve"


def test_m6_dropping_the_mean_of_z_derivative_fails_the_level_curve_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6: the ``E_n[Z]`` derivative of the reference mean cancels in a contrast."""
    mutate(
        monkeypatch,
        module,
        module,
        "_reference_curve",
        [("curve = w * (adjusted - mu0) - psi_curve @ mean_z", "curve = w * (adjusted - mu0)")],
    )
    gaps = _curve_gaps(laws.law_h(), TRUTH["H"]["features"], W3_CASES["H"])
    assert gaps["mu0"] > 1e-3
    assert gaps["level[r11]"] > 1e-3
    assert gaps["contrast[r11]"] < 1e-6, "the term cancels between the two levels"


def test_m7_the_treatment_mean_in_place_of_the_product_nuisance_fails_on_law_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M7: ``phi * E[c(A_j) | H_t]`` in place of ``E[phi * c(A_j) | H_t]``."""
    law = laws.law_p()
    features = TRUTH["P"]["features"]
    clean = _fit(law, features)
    mutate(
        monkeypatch,
        module,
        module,
        "_fit_nuisances",
        [
            ("undetermined * codings[j][:, l],", "codings[j][:, l],"),
            (
                "mean[:, k * node.d + l] = determined * cache[key]",
                "mean[:, k * node.d + l] = determined * undetermined * cache[key]",
            ),
        ],
    )
    broken = _fit(law, features)
    assert abs(clean.psi("blip[A1:1.0:intercept]") - 1.28) < EXACT
    assert abs(broken.psi("blip[A1:1.0:intercept]") - 1.28) > 0.01


# ------------------------------------------------------------------ labels and files


def _simulated(n: int = 600, seed: int = 0, clusters: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    s1 = 0.5 * w + rng.normal(size=n)
    a1 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * s1)))
    s2 = 0.5 * s1 + 0.8 * a1 + rng.normal(size=n)
    a2 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * s2 - 0.3 * a1)))
    y = a1 + 0.5 * a2 + 0.7 * s2 + 0.3 * w + rng.normal(size=n)
    frame = pd.DataFrame({"W": w, "S1": s1, "A1": a1, "S2": s2, "A2": a2, "Y": y})
    if clusters is not None:
        frame["id"] = rng.integers(0, clusters, size=n)
    return frame


SIM = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [["S1"], ["S2"]],
}
GLMS = {
    "outcome_learner": GLM(family="gaussian"),
    "treatment_learner": GLM(family="binomial"),
    "random_state": 0,
    "simultaneous": False,
}
REGIMENS = {"always": [1, 1], "never": [0, 0]}


def test_w6_the_structural_label_survives_save_and_load(tmp_path: Any) -> None:
    result = snmm(_simulated(), regimens=REGIMENS, n_folds=2, **GLMS, **SIM)
    frame = result.to_frame()
    assert set(frame["model"]) == {module.SNMM_LABEL}
    assert len(frame["model"]) == len(result)
    assert module.SNMM_LABEL in result.summary()
    path = tmp_path / "snmm.joblib"
    result.save(path)
    loaded = cleverly.load(path)
    assert isinstance(loaded, StructuralNestedMeanResult)
    assert loaded.label == module.SNMM_LABEL
    assert set(loaded.to_frame()["model"]) == {module.SNMM_LABEL}
    assert module.SNMM_LABEL in loaded.summary()
    assert loaded.psi("ey_regimen[always]") == result.psi("ey_regimen[always]")
    assert loaded.inference_status == result.inference_status
    assert list(loaded.kinds.values()) == list(result.kinds.values())


def test_status_follows_the_fold_count_the_learners_and_the_weights() -> None:
    frame = _simulated()
    cross = snmm(frame, regimens=REGIMENS, n_folds=2, **GLMS, **SIM)
    assert cross.inference_status == "influence_curve"
    insample = snmm(frame, regimens=REGIMENS, n_folds=1, **GLMS, **SIM)
    assert insample.inference_status == "influence_curve"
    from sklearn.linear_model import LinearRegression, LogisticRegression

    with pytest.warns(UserWarning, match="adaptive_baseline_plugin"):
        adaptive = snmm(
            frame,
            regimens=REGIMENS,
            n_folds=1,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            random_state=0,
            simultaneous=False,
            **SIM,
        )
    assert adaptive.inference_status == "adaptive_baseline_plugin"
    assert not adaptive["ey_regimen[always]"].supplies_inference
    cross_adaptive = snmm(
        frame,
        regimens=REGIMENS,
        n_folds=2,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        random_state=0,
        simultaneous=False,
        **SIM,
    )
    assert cross_adaptive.inference_status == "influence_curve"
    weighted = frame.assign(w=np.random.default_rng(1).uniform(0.5, 1.5, len(frame)))
    with pytest.warns(UserWarning, match="estimated_weight_plugin"):
        estimated = snmm(
            weighted,
            regimens=REGIMENS,
            n_folds=2,
            weights="w",
            weights_estimated=True,
            **GLMS,
            **SIM,
        )
    assert estimated.inference_status == "estimated_weight_plugin"
    with pytest.warns(UserWarning, match="few_cluster_plugin"):
        few = snmm(_simulated(clusters=8), regimens=REGIMENS, n_folds=2, id="id", **GLMS, **SIM)
    assert few.inference_status == "few_cluster_plugin"


@pytest.mark.filterwarnings("ignore:.*look.* like counts of identical units")
def test_weights_and_clusters() -> None:
    """Integer weights equal replicated rows for the point estimate; whole clusters share a fold."""
    frame = _simulated(300)
    counts = np.random.default_rng(2).integers(1, 4, size=len(frame))
    weighted = snmm(
        frame.assign(w=counts.astype(float)),
        regimens=REGIMENS,
        n_folds=1,
        weights="w",
        **GLMS,
        **SIM,
    )
    replicated = snmm(
        frame.loc[frame.index.repeat(counts)], regimens=REGIMENS, n_folds=1, **GLMS, **SIM
    )
    for name in (
        "blip[A1:1:intercept]",
        "blip[A2:1:intercept]",
        "ey_regimen[always]",
        "ey_reference",
    ):
        assert abs(weighted.psi(name) - replicated.psi(name)) < 1e-9, name
    clustered = snmm(
        _simulated(400, clusters=40), regimens=REGIMENS, n_folds=4, id="id", **GLMS, **SIM
    )
    assignment = clustered.folds.assignment
    labels = clustered.data.cluster
    for label in np.unique(labels):
        assert len(set(assignment[labels == label])) == 1
    assert clustered.inference_status == "influence_curve"
    unit = snmm(
        _simulated(400, clusters=None).assign(id=np.arange(400)),
        regimens=REGIMENS,
        n_folds=1,
        id="id",
        **GLMS,
        **SIM,
    )
    plain = snmm(_simulated(400), regimens=REGIMENS, n_folds=1, **GLMS, **SIM)
    assert unit["ey_regimen[always]"].std_error == pytest.approx(
        plain["ey_regimen[always]"].std_error, rel=1e-9
    )


def _clustered(n_clusters: int = 40, size: int = 10, seed: int = 3) -> pd.DataFrame:
    """Clusters of ``size`` rows that share a random intercept of the outcome.

    The intercept has standard deviation 2 against a unit residual, so the intraclass
    correlation of the outcome is 0.8.  The standard error of a regimen level then grows by a
    factor above 1.5 when the fit sums the curve within a cluster.
    """
    rng = np.random.default_rng(seed)
    n = n_clusters * size
    ident = np.repeat(np.arange(n_clusters), size)
    shared = 2.0 * rng.normal(size=n_clusters)[ident]
    w = rng.normal(size=n)
    s1 = 0.5 * w + rng.normal(size=n)
    a1 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * s1)))
    s2 = 0.5 * s1 + 0.8 * a1 + rng.normal(size=n)
    a2 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * s2 - 0.3 * a1)))
    y = a1 + 0.5 * a2 + 0.7 * s2 + 0.3 * w + shared + rng.normal(size=n)
    return pd.DataFrame({"W": w, "S1": s1, "A1": a1, "S2": s2, "A2": a2, "Y": y, "id": ident})


def _cluster_standard_error(curve: np.ndarray, cluster: np.ndarray) -> float:
    """The cluster-robust standard error of a mean of ``curve``, by explicit loops."""
    labels = np.unique(cluster)
    sums = np.array([curve[cluster == label].sum() for label in labels])
    centered = sums - sums.mean()
    n_clusters = len(labels)
    variance = (centered @ centered) / (n_clusters - 1)
    return float(np.sqrt(n_clusters * variance) / len(curve))


def _clustered_fits() -> tuple[StructuralNestedMeanResult, StructuralNestedMeanResult]:
    frame = _clustered()
    settings = {**GLMS, "n_folds": 1, "n_multiplier": 400, "simultaneous": True}
    return (
        snmm(frame, regimens=REGIMENS, id="id", **settings, **SIM),
        snmm(frame, regimens=REGIMENS, **settings, **SIM),
    )


def test_w8_the_variance_sums_the_curve_within_a_cluster() -> None:
    """W8: every standard error is the cluster-summed variance of its own curve.

    The curve is the same with and without ``id=``.  Only the variance differs, and only a
    design with a shared cluster effect separates the two.  A fit with one unit per cluster
    equals the unclustered fit whether or not the code sums within a cluster.
    """
    clustered, plain = _clustered_fits()
    labels = clustered.data.cluster
    assert clustered.inference_status == "influence_curve"
    for name in clustered:
        curve = clustered[name].influence_curve
        assert np.array_equal(curve, plain[name].influence_curve), name
        assert clustered[name].std_error == pytest.approx(
            _cluster_standard_error(curve, labels), rel=1e-9
        ), name
    # The levels carry the shared effect, so the unclustered variance is far too small.
    for name in ("ey_reference", "ey_regimen[always]", "ey_regimen[never]"):
        assert clustered[name].std_error > 1.5 * plain[name].std_error, name


def test_w8_the_bands_and_the_joint_covariance_read_the_clusters() -> None:
    clustered, plain = _clustered_fits()
    names = [name for name, kind in clustered.kinds.items() if kind in ("level", "contrast")]
    chosen = {name: clustered[name] for name in names}
    settings = {"alpha": 0.05, "n_replicates": 400, "kind": "rademacher", "random_state": 0}
    with_clusters = simultaneous_bands(chosen, cluster=clustered.data.cluster, **settings)
    without = simultaneous_bands(chosen, cluster=None, **settings)
    assert clustered.simultaneous is not None
    assert clustered.simultaneous.critical_value == pytest.approx(
        with_clusters.critical_value, rel=1e-12
    )
    # The two critical values differ by a stated margin, so the comparison can fail.
    assert abs(with_clusters.critical_value - without.critical_value) > 0.01
    curves = np.column_stack([clustered[name].influence_curve for name in names])
    assert np.allclose(
        clustered.covariance(names), influence_covariance(curves, clustered.data.cluster)
    )
    assert np.linalg.norm(clustered.covariance(names) - plain.covariance(names)) > 0.05


def test_w8_dropping_the_cluster_from_the_variance_fails_the_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: ``cluster=None`` in the estimates or in the bands is seen by the W8 witnesses."""
    mutate(
        monkeypatch,
        module,
        StructuralNestedMean,
        "_estimates",
        [("cluster=data.cluster,", "cluster=None,")],
    )
    mutate(
        monkeypatch,
        module,
        StructuralNestedMean,
        "_bands",
        [("cluster=data.cluster,", "cluster=None,")],
    )
    with pytest.raises(AssertionError):
        test_w8_the_variance_sums_the_curve_within_a_cluster()
    with pytest.raises(AssertionError):
        test_w8_the_bands_and_the_joint_covariance_read_the_clusters()


# ------------------------------------------------------------------------- refusals


def _refused(
    reason: RefusalReason,
    call: Callable[[], Any],
    *,
    roadmap: str | None = None,
) -> CapabilityError:
    with pytest.raises(CapabilityError) as raised:
        call()
    assert raised.value.refusal.reason is reason
    if roadmap is not None:
        assert raised.value.refusal.roadmap == roadmap
    return raised.value


def _censored() -> pd.DataFrame:
    frame = _simulated()
    frame["C1"] = 1
    frame["C2"] = 1
    return frame


def test_w7_refusals_carry_their_reason_codes() -> None:
    frame = _simulated()
    redirect = RefusalReason.API_REDIRECTION
    invalid = RefusalReason.INVALID_DECLARATION
    fit = lambda **kw: StructuralNestedMean(REGIMENS, **GLMS, **kw.pop("settings", {})).fit(  # noqa: E731
        kw.pop("data", frame), **{**SIM, **kw}
    )
    _refused(redirect, lambda: fit(data=_censored(), censoring=["C1", "C2"]))
    _refused(
        redirect,
        lambda: StructuralNestedMean(
            DynamicRegimen("rule", (lambda h: h["W"] > 0, 1), rule_kind="known"), **GLMS
        ).fit(frame, **SIM),
    )
    from cleverly.interventions import Incremental

    _refused(
        redirect,
        lambda: StructuralNestedMean({"tilt": Incremental(2.0)}, **GLMS).fit(frame, **SIM),
    )
    _refused(redirect, lambda: StructuralNestedMean(REGIMENS, msm=object(), **GLMS))
    for name in ("pseudo_learner", "censoring_learner", "horizons"):
        _refused(invalid, lambda name=name: StructuralNestedMean(REGIMENS, **{name: 1}))
    _refused(invalid, lambda: fit(settings={"blip_features": {1: ["Nope"]}}))
    _refused(invalid, lambda: fit(settings={"blip_features": {1: ["A1"]}}))
    _refused(invalid, lambda: fit(settings={"blip_features": {1: ["S2"]}}))
    _refused(invalid, lambda: fit(settings={"blip_features": {3: ["W"]}}))
    with pytest.raises(DataError, match="levels"):
        StructuralNestedMean({"far": [5, 5]}, **GLMS).fit(frame, **SIM)


def test_w7_a_regimen_dose_outside_the_observed_range_is_refused() -> None:
    law = laws.law_c()
    _refused(
        RefusalReason.INVALID_DECLARATION,
        lambda: _fit(law, None, {"out": [3.0, 0.0], "zero": [0.0, 0.0]}),
    )


def test_w7_a_dose_range_without_the_reference_is_refused() -> None:
    frame = _simulated().assign(D1=lambda f: f["W"].abs() + 1.0)
    _refused(
        RefusalReason.INVALID_DECLARATION,
        lambda: StructuralNestedMean(None, **GLMS).fit(
            frame,
            outcome="Y",
            treatment=["D1", "A2"],
            baseline=["W"],
            time_varying=[["S1"], ["S2"]],
            continuous_treatment=["D1"],
        ),
    )


def test_w7_a_time_varying_blip_feature_allows_blips_only() -> None:
    frame = _simulated()
    features = {2: ["S2"]}
    clean = StructuralNestedMean(None, blip_features=features, **GLMS).fit(frame, **SIM)
    assert "blip[A2:1:S2]" in clean and "ey_reference" in clean
    _refused(
        RefusalReason.IMPLEMENTATION_GAP,
        lambda: StructuralNestedMean(REGIMENS, blip_features=features, **GLMS).fit(frame, **SIM),
        roadmap="X72",
    )
    _refused(
        RefusalReason.IMPLEMENTATION_GAP,
        lambda: clean.check_model(),
        roadmap="X72",
    )


def test_w7_a_singular_block_is_refused() -> None:
    frame = _simulated().assign(A2=1)
    _refused(
        RefusalReason.INVALID_DECLARATION,
        lambda: StructuralNestedMean(REGIMENS, **GLMS).fit(frame, **SIM),
    )


def test_w7_survival_and_repeated_outcomes_are_redirected() -> None:
    frame = _simulated().assign(E1=0, E2=0)
    _refused(
        RefusalReason.API_REDIRECTION,
        lambda: StructuralNestedMean(REGIMENS, **GLMS).fit(
            frame,
            outcome=["E1", "E2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[["S1"], ["S2"]],
        ),
    )


def test_w7_every_redirected_design_carries_its_reason_code() -> None:
    """Each design the estimator redirects raises ``API_REDIRECTION`` before any learner."""
    redirect = RefusalReason.API_REDIRECTION
    frame = _simulated().assign(Y2=lambda f: f["Y"], E1=0, E2=0, D1=0, D2=0)
    fit = lambda **kw: StructuralNestedMean(REGIMENS, **GLMS).fit(  # noqa: E731
        kw.pop("data", frame), **{**SIM, **kw}
    )
    _refused(redirect, lambda: fit(outcome=Repeated(["Y", "Y2"])))
    density = Stochastic(lambda h: np.full((len(h), 2), 0.5), "q", density_kind="known")
    for plan in (
        DynamicRegimen("stochastic", (density, 1)),
        DynamicRegimen("shift", (Shift(0.0, cap=None), 1)),
    ):
        _refused(
            redirect,
            lambda plan=plan: StructuralNestedMean({plan.label: plan}, **GLMS).fit(frame, **SIM),
        )
    _refused(redirect, lambda: fit(outcome={"event": ["E1", "E2"], "competing": ["D1", "D2"]}))
    data = LongitudinalData.from_frame(frame, **SIM)
    declared = data.with_known_mechanisms(treatment_probabilities=np.full((len(frame), 2), 0.5))
    _refused(redirect, lambda: StructuralNestedMean(REGIMENS, **GLMS).fit(declared))
    # Only a survival design holds one decision over the nodes, and the survival refusal comes
    # first, so the held refusal is checked on the design check itself.
    _refused(
        redirect,
        lambda: StructuralNestedMean._refuse_design(dataclasses.replace(data, treatment_held=True)),
    )


def test_w7_a_reference_with_no_regimen_is_refused() -> None:
    _refused(
        RefusalReason.INVALID_DECLARATION,
        lambda: StructuralNestedMean(None, reference="never", **GLMS).fit(_simulated(), **SIM),
    )


def test_w7_check_model_refusals_carry_their_reason_codes() -> None:
    """A continuous node, a fit with no interval and a comparison with no interval refuse."""
    dose = _fit(laws.law_c(), None, {"inner": [0.5, 0.25], "zero": [0.0, 0.0]})
    _refused(RefusalReason.INVALID_DECLARATION, dose.check_model)
    from sklearn.linear_model import LinearRegression, LogisticRegression

    with pytest.warns(UserWarning, match="adaptive_baseline_plugin"):
        adaptive = snmm(
            _simulated(),
            regimens=REGIMENS,
            n_folds=1,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            random_state=0,
            simultaneous=False,
            **SIM,
        )
    _refused(RefusalReason.INFERENCE_GAP, adaptive.check_model)
    result = snmm(_simulated(), regimens=REGIMENS, n_folds=1, **GLMS, **SIM)

    class NoInterval(LTMLE):
        """A nonparametric fit whose estimates carry a non-inferential status."""

        def fit(self, *args: Any, **kwargs: Any) -> Any:
            fit = super().fit(*args, **kwargs)
            return dataclasses.replace(
                fit,
                estimates={
                    name: dataclasses.replace(estimate, inference="constant_node_plugin")
                    for name, estimate in fit.estimates.items()
                },
            )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "LTMLE", NoInterval)
        _refused(RefusalReason.INFERENCE_GAP, result.check_model)


def _extreme_law() -> laws.Law:
    """Node 1 has a cell, ``W = 1`` with mass 0.4, whose treatment probability is 0.995."""
    variables = (
        laws.Var("W", "baseline", 0, lambda p: laws.bern(0.4)),
        laws.Var("S1", "covariate", 1, lambda p: laws.bern(0.5)),
        laws.Var("A1", "treatment", 1, lambda p: laws.bern(0.995 if p["W"] else 0.5)),
        laws.Var("S2", "covariate", 2, lambda p: laws.bern(0.3 + 0.2 * p["A1"])),
        laws.Var("A2", "treatment", 2, lambda p: laws.bern(0.5)),
    )
    return laws.Law("E", variables, lambda p: p["A1"] + 0.5 * p["A2"] + 0.3 * p["S2"])


def test_the_extreme_share_is_the_mass_of_the_cells_near_zero_or_one() -> None:
    """The share is 0.4 at the default margin, 0 at a margin below the cell, and nan for a dose."""
    law = _extreme_law()
    flagged = _fit(law).diagnostics.extreme_share
    assert flagged[0] == pytest.approx(0.4, abs=1e-9)
    assert flagged[1] == pytest.approx(0.0, abs=1e-9)
    narrow = _fit(law, positivity_margin=0.001).diagnostics.extreme_share
    assert narrow[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(_fit(laws.law_c()).diagnostics.extreme_share[0])


def test_a_regression_learner_fits_a_dose_node_with_a_binary_outcome() -> None:
    """A classifier cannot fit a dose, so ``regression_learner`` takes the targets off {0, 1}."""
    rng = np.random.default_rng(0)
    n = 800
    w = rng.normal(size=n)
    d1 = rng.normal(size=n) + 0.3 * w
    s2 = 0.5 * d1 + rng.normal(size=n)
    a2 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * s2)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * d1 + 0.5 * a2 + 0.3 * w))))
    frame = pd.DataFrame({"W": w, "D1": d1, "S2": s2, "A2": a2, "Y": y})
    design = {
        "outcome": "Y",
        "treatment": ["D1", "A2"],
        "baseline": ["W"],
        "time_varying": [[], ["S2"]],
        "continuous_treatment": ["D1"],
    }
    classifiers = {
        "outcome_learner": GLM(family="binomial"),
        "treatment_learner": GLM(family="binomial"),
        "n_folds": 2,
        "random_state": 0,
        "simultaneous": False,
    }
    with pytest.raises(DataError, match="binomial target"):
        StructuralNestedMean(None, **classifiers).fit(frame, **design)
    fitted = StructuralNestedMean(
        None, regression_learner=GLM(family="gaussian"), **classifiers
    ).fit(frame, **design)
    assert fitted.inference_status == "influence_curve"
    assert np.isfinite(fitted.psi("ey_reference"))
    assert {"blip[D1:dose:intercept]", "blip[A2:1:intercept]"} <= set(fitted)


# ------------------------------------------------------------------------ model check


def test_check_model_scope_and_rank_rule() -> None:
    frame = _simulated(500)
    result = snmm(frame, regimens=REGIMENS, n_folds=1, **GLMS, **SIM)
    check = result.check_model()
    assert check.tested and check.df == 1
    assert np.isfinite(check.statistic) and 0.0 <= check.pvalue <= 1.0
    assert check.names == ("ate_regimen[never vs always]",)
    assert "Structural model check" in check.summary()
    single = snmm(frame, regimens={"only": [1, 1]}, n_folds=1, **GLMS, **SIM)
    _refused(RefusalReason.INVALID_DECLARATION, single.check_model)
    cross = snmm(frame, regimens=REGIMENS, n_folds=2, **GLMS, **SIM)
    _refused(RefusalReason.INVALID_DECLARATION, cross.check_model)
    assert cross.check_model(q_bounds=(-30.0, 30.0)).tested
    strict = result.check_model(rank_floor=10.0)
    assert not strict.tested and "difference covariance" in strict.summary()


def test_check_model_uses_the_folds_of_the_structural_fit() -> None:
    """The nonparametric fit draws no split of its own: the shared folds are the fit's."""
    # Without a seed each fit would draw a different split, so only sharing makes them equal.
    result = snmm(
        _simulated(500), regimens=REGIMENS, n_folds=3, **{**GLMS, "random_state": None}, **SIM
    )
    check = result.check_model(q_bounds=(-30.0, 30.0))
    assert np.array_equal(check.ltmle.folds.assignment, result.folds.assignment)
    assert isinstance(check.ltmle, cleverly.longitudinal.LongitudinalResult)
    assert check.ltmle.config.n_folds == 3
    assert isinstance(LTMLE, type)


def test_w4_the_structural_contrasts_on_law_h_match_and_on_law_l_miss() -> None:
    """With the declared features the contrast of law H is the truth.

    On law L with ``phi = 1`` it differs by a closed-form population amount.
    """
    law = laws.law_h()
    regimens = {"r00": [0.0, 0.0], "r11": [1.0, 1.0], "r10": [1.0, 0.0]}
    declared = _fit(law, TRUTH["H"]["features"], regimens)
    truth = laws.regimen_mean(law, [1, 1]) - laws.regimen_mean(law, [0, 0])
    assert abs(declared.psi("ate_regimen[r11 vs r00]") - truth) < EXACT
    law = laws.law_l()
    wrong = _fit(law, None, {"r00": [0.0, 0.0], "r01": [0.0, 1.0]}, reference="r00")
    gap = wrong.psi("ate_regimen[r01 vs r00]") - (
        laws.regimen_mean(law, [0, 1]) - laws.regimen_mean(law, [0, 0])
    )
    assert abs(gap) > 0.05


def _law_l_limits() -> tuple[float, float]:
    """The population limits of ``psi_2`` and ``psi_1`` on law L with ``phi = 1``.

    Node 2 gives ``0.5 + slope * E[S2 v] / E[v]``.  Node 1 solves its equation with that
    limit in place of the later blip: ``psi_1 = (E[Z Y~] - psi_2 E[Z Z_2]) / E[Z^2]`` for the
    residual treatment ``Z`` of node 1.
    """
    law = laws.law_l()
    mean_a2 = laws.conditional_mean(law, 2, lambda p: p["A2"])
    weight = laws.moment(law, lambda p: mean_a2(p) * (1.0 - mean_a2(p)))
    weighted_s2 = laws.moment(law, lambda p: p["S2"] * mean_a2(p) * (1.0 - mean_a2(p)))
    psi_2 = 0.5 + laws.L_SLOPE * weighted_s2 / weight
    first_a1 = laws.conditional_mean(law, 1, lambda p: p["A1"])
    first_a2 = laws.conditional_mean(law, 1, lambda p: p["A2"])
    first_y = laws.conditional_mean(law, 1, lambda p: law.outcome(p))
    variance = laws.moment(law, lambda p: (p["A1"] - first_a1(p)) ** 2)
    cross_y = laws.moment(law, lambda p: (p["A1"] - first_a1(p)) * (law.outcome(p) - first_y(p)))
    cross_a = laws.moment(law, lambda p: (p["A1"] - first_a1(p)) * (p["A2"] - first_a2(p)))
    return psi_2, (cross_y - psi_2 * cross_a) / variance


def test_check_model_statistic_on_exact_law_l() -> None:
    """The check reports the closed-form structural gap against the g-formula truth.

    Law L with ``phi = 1`` has a blip that varies with an undeclared covariate.  The
    saturated nonparametric contrasts equal the g-formula truth, and the structural
    contrasts equal the closed-form limits, so the difference is nonzero and known.
    """
    law = laws.law_l()
    regimens = {"r00": [0.0, 0.0], "r01": [0.0, 1.0], "r10": [1.0, 0.0], "r11": [1.0, 1.0]}
    check = _fit(law, None, regimens, reference="r00").check_model()
    assert check.names == (
        "ate_regimen[r01 vs r00]",
        "ate_regimen[r10 vs r00]",
        "ate_regimen[r11 vs r00]",
    )
    truth = np.array(
        [
            laws.regimen_mean(law, a) - laws.regimen_mean(law, [0, 0])
            for a in ([0, 1], [1, 0], [1, 1])
        ]
    )
    psi_2, psi_1 = _law_l_limits()
    limits = np.array([psi_2, psi_1, psi_1 + psi_2])
    assert np.allclose(check.nonparametric, truth, atol=EXACT)
    assert np.allclose(check.structural, limits, atol=EXACT)
    assert np.allclose(check.difference, limits - truth, atol=EXACT)
    assert np.min(np.abs(check.difference)) > 0.01, "every contrast carries a nonzero gap"
    assert check.df == 3 and check.tested
    gap = check.difference
    assert check.statistic == pytest.approx(gap @ np.linalg.solve(check.covariance, gap))
    assert check.pvalue == pytest.approx(stats.chi2.sf(check.statistic, df=3), rel=1e-12)


FOUR_REGIMENS = {
    "always": [1, 1],
    "never": [0, 0],
    "first": [1, 0],
    "second": [0, 1],
}


def _check_pieces(check: Any, result: StructuralNestedMeanResult, cluster: Any) -> dict[str, Any]:
    """The difference curve, covariance and statistic of a check, rebuilt from the curves."""
    difference = np.column_stack(
        [result[name].influence_curve - check.ltmle[name].influence_curve for name in check.names]
    )
    structural_only = np.column_stack([result[name].influence_curve for name in check.names])
    gap = check.structural - check.nonparametric
    covariance = influence_covariance(difference, cluster)
    return {
        "covariance": covariance,
        "structural_only": influence_covariance(structural_only, cluster),
        "statistic": float(gap @ np.linalg.solve(covariance, gap)),
    }


@pytest.mark.parametrize("clustered", [False, True], ids=["iid", "clustered"])
def test_check_model_covariance_degrees_of_freedom_and_clusters(clustered: bool) -> None:
    """The check covariance is that of the difference of the two curves, at the cluster level.

    The structural-only covariance, a one-degree-of-freedom tail and an unclustered
    covariance each differ from the shipped values by a stated margin.
    """
    frame = _clustered() if clustered else _simulated(500)
    result = snmm(
        frame,
        regimens=FOUR_REGIMENS,
        reference="never",
        n_folds=1,
        id="id" if clustered else None,
        **GLMS,
        **SIM,
    )
    check = result.check_model()
    cluster = result.data.cluster
    assert (cluster is not None) is clustered
    assert check.df == 3 and check.tested
    pieces = _check_pieces(check, result, cluster)
    assert np.allclose(check.covariance, pieces["covariance"], rtol=1e-10)
    assert check.statistic == pytest.approx(pieces["statistic"], rel=1e-8)
    assert check.pvalue == pytest.approx(stats.chi2.sf(check.statistic, df=3), rel=1e-10)
    # The statistic is large enough that a wrong degrees of freedom moves the tail.
    assert abs(stats.chi2.sf(check.statistic, df=1) - check.pvalue) > 0.01
    norm = np.linalg.norm
    assert norm(check.covariance - pieces["structural_only"]) > 0.1 * norm(check.covariance)
    if clustered:
        unclustered = _check_pieces(check, result, None)
        assert norm(check.covariance - unclustered["covariance"]) > 0.1 * norm(check.covariance)


def test_the_one_call_form_equals_the_class() -> None:
    frame = _simulated(200)
    kwargs = {**GLMS, "n_folds": 2}
    by_class = StructuralNestedMean(REGIMENS, **kwargs).fit(frame, **SIM)
    by_call = snmm(frame, regimens=REGIMENS, **kwargs, **SIM)
    assert list(by_class) == list(by_call)
    assert by_class.psi("ey_regimen[always]") == by_call.psi("ey_regimen[always]")


def test_the_summary_prints_the_condition_and_the_bands() -> None:
    result = snmm(
        _simulated(300), regimens=REGIMENS, n_folds=2, **{**GLMS, "simultaneous": True}, **SIM
    )
    text = result.summary()
    assert "simultaneous" in text and "valid under the linear structural nested mean model" in text
    assert warnings is not None
