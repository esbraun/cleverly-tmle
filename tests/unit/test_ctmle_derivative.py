r"""C-TMLE0: the derivative columns and the update seam.

The seam :func:`cleverly.estimators._ctmle_derivative.derivative_update` fits the clever
covariates ``h_a`` and their derivatives ``dh_a = h_a (g_a - g+_a) / g_a`` in one solve.
Its derivative scores vanish at the truth only when the outcome regression is right, so
every witness here runs on a law whose outcome regression is wrong, where the plain TMLE
leaves the derivative scores far from zero.

* An independent transcription of R ``ctmle_glmnet.R`` lines 133-166 (an unnormalized
  four-column logistic fit with an offset) gives the same coefficients and targeted fit.
* W1: each arm's derivative score is solved, and the plain TMLE's is not (W1b missing
  outcomes, W1c the combined ATE pair, W1d fixed weights, W1e the linear fluctuation).
* W2: the estimate moves away from the plain TMLE's.
* W3: rescaling a derivative column changes nothing.
* W4: the counterfactual derivative block moves the predictions of the other arm's rows.
* W5: three arms need one derivative column each.
* The refusals of the seam.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.special import expit, logit

from cleverly.estimators._ctmle_derivative import (
    DERIVATIVE_ONE_STEP_REFUSAL,
    DERIVATIVE_TARGET_WEIGHTS_REFUSAL,
    derivative_ratio,
    derivative_update,
)
from cleverly.estimators.targeting import TargetingSpec, solve_submodel
from cleverly.exceptions import CapabilityError
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import Submodel, derivative_columns, mean_submodel

BOUNDS = (0.025, 0.975)
SPEC = TargetingSpec(alpha=0.995)


def _bounded(g: np.ndarray) -> np.ndarray:
    return np.clip(g, *BOUNDS)


def _law(n: int = 1500, seed: int = 0, *, arms: int = 2, missing: bool = False) -> dict[str, Any]:
    """A law with a main-terms mechanism and an initial outcome regression that omits W2.

    The omitted confounder keeps the derivative scores away from zero before the update.
    The partner halves the mechanism's logits, as a larger lasso penalty would.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3))
    index = 0.5 * w[:, 0] - 0.4 * w[:, 1] + 0.3 * w[:, 2]
    if arms == 2:
        g1 = expit(index)
        g = np.column_stack([1.0 - g1, g1])
        partner1 = expit(0.5 * index)
        partner = np.column_stack([1.0 - partner1, partner1])
    else:
        logits = np.column_stack([np.zeros(n), index, -0.7 * index + 0.5 * w[:, 2]])
        g = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        partner = np.exp(0.5 * logits) / np.exp(0.5 * logits).sum(axis=1, keepdims=True)
    codes = np.arange(arms, dtype=float)
    a = np.array([rng.choice(codes, p=row) for row in g])
    y = rng.binomial(1, expit(-0.3 + 0.5 * a + 0.6 * w[:, 0] + 0.8 * w[:, 1])).astype(float)

    def misspecified(level: Any) -> np.ndarray:
        # The initial regression leaves W2 out, so it is wrong for every arm.
        return np.asarray(expit(-0.3 + 0.5 * level + 0.6 * w[:, 0]), dtype=float)

    initial = InitialFit(misspecified(a), {float(level): misspecified(level) for level in codes})
    pi = None
    observed = np.ones(n, dtype=bool)
    if missing:
        pi_value = expit(1.2 + 0.6 * w[:, 0] - 0.3 * a)
        pi = np.column_stack([expit(1.2 + 0.6 * w[:, 0] - 0.3 * level) for level in codes])
        observed = rng.uniform(size=n) < pi_value
    return {
        "a": a,
        "y": y,
        "w": w,
        "g": _bounded(g),
        "partner": _bounded(partner),
        "initial": initial,
        "pi": pi,
        "observed": observed,
        "arms": tuple(float(level) for level in codes),
        "weights": np.ones(n),
    }


def _base(law: dict[str, Any]) -> Any:
    return mean_submodel(law["a"], law["g"], arms=law["arms"], missingness=law["pi"])


def _update(law: dict[str, Any], spec: TargetingSpec = SPEC, **overrides: Any) -> Any:
    arguments = {
        "scaled": law["y"],
        "initial": law["initial"],
        "base": _base(law),
        "mechanism": law["g"],
        "partner": law["partner"],
        "treatment": law["a"],
        "observed": law["observed"],
        "weights": law["weights"],
        "spec": spec,
    }
    arguments.update(overrides)
    return derivative_update(**arguments)


def _derivative_scores(law: dict[str, Any], targeted: InitialFit) -> np.ndarray:
    """``P_n w dh_a (Y - Q)`` per arm on the normalized columns, from first principles."""
    a, y, mask = law["a"], law["y"], law["observed"]
    ratio = (law["g"] - law["partner"]) / law["g"]
    pi = np.ones_like(law["g"]) if law["pi"] is None else law["pi"]
    scores = []
    for j, level in enumerate(law["arms"]):
        column = (a == level) / (law["g"][:, j] * pi[:, j]) * ratio[:, j]
        rows = (a == level) & mask
        column = column / np.sqrt(np.mean(column[rows] ** 2))
        residual = np.where(mask, y - targeted.observed, 0.0)
        scores.append(np.mean(law["weights"] * column * residual))
    return np.array(scores)


def _plain(law: dict[str, Any], spec: TargetingSpec = SPEC) -> Any:
    return solve_submodel(
        law["y"], law["initial"], _base(law), law["weights"], law["observed"], spec, warn=False
    )


def _estimate(targeted: InitialFit, weights: np.ndarray) -> float:
    return float(
        np.average(targeted.arms[1.0], weights=weights)
        - np.average(targeted.arms[0.0], weights=weights)
    )


@pytest.fixture(scope="module")
def binary() -> dict[str, Any]:
    return _law(seed=1)


class TestRTranscription:
    """The seam against an independent transcription of R ``ctmle`` type 3."""

    def test_coefficients_and_targeted_fit_match_the_four_column_glm(
        self, binary: dict[str, Any]
    ) -> None:
        law = binary
        a, y = law["a"], law["y"]
        g1, g1_prev = law["g"][:, 1], law["partner"][:, 1]
        g0, g0_prev = 1.0 - g1, 1.0 - g1_prev
        # ctmle_glmnet.R:138-148: H1W, H0W, ddH1W, ddH0W.
        h1, h0 = a / g1, (1 - a) / g0
        dd1, dd0 = a / g1**2 * (g1 - g1_prev), (1 - a) / g0**2 * (g0 - g0_prev)
        x = np.column_stack([h0, h1, dd0, dd1])
        offset = logit(law["initial"].observed)
        beta = np.zeros(4)
        for _ in range(100):
            p = expit(offset + x @ beta)
            gradient = x.T @ (y - p)
            hessian = x.T @ (x * (p * (1 - p))[:, None])
            step = np.linalg.solve(hessian, gradient)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-14:
                break
        q1 = expit(
            logit(law["initial"].arms[1.0]) + beta[1] / g1 + beta[3] / g1**2 * (g1 - g1_prev)
        )
        q0 = expit(
            logit(law["initial"].arms[0.0]) + beta[0] / g0 + beta[2] / g0**2 * (g0 - g0_prev)
        )

        result = _update(law)

        # The package shrinks predictions into (0.005, 0.995) after each step and R does
        # not (quirk 3), so the comparison is exact only when no prediction reaches it.
        for values in (q0, q1, result.targeted.observed):
            assert values.min() > 0.005 and values.max() < 0.995
        assert result.names == ("h0", "h1", "dh0", "dh1")
        np.testing.assert_allclose(result.epsilon, beta, rtol=1e-8)
        np.testing.assert_allclose(result.targeted.arms[1.0], q1, atol=1e-10)
        np.testing.assert_allclose(result.targeted.arms[0.0], q0, atol=1e-10)
        np.testing.assert_allclose(
            _estimate(result.targeted, law["weights"]), np.mean(q1 - q0), atol=1e-10
        )


class TestScoresAreSolved:
    """W1 and its variants: the derivative scores vanish only under the update."""

    def test_binary(self, binary: dict[str, Any]) -> None:
        result = _update(binary)
        after = _derivative_scores(binary, result.targeted)
        before = _derivative_scores(binary, _plain(binary).targeted)
        assert np.max(np.abs(after)) <= 1e-8
        assert np.min(np.abs(before)) > 1e-2
        np.testing.assert_allclose(result.score_after, after, atol=1e-12)

    def test_missing_outcomes_divide_the_column_by_pi(self) -> None:
        law = _law(missing=True, seed=1)
        assert not law["observed"].all()
        result = _update(law)
        after = _derivative_scores(law, result.targeted)
        assert np.max(np.abs(after)) <= 1e-8
        # The column without 1/pi is a different direction, and its score is left over.
        without = dict(law, pi=np.ones_like(law["g"]))
        assert np.max(np.abs(_derivative_scores(without, result.targeted))) > 1e-3

    def test_the_combined_ate_pair_is_solved(self, binary: dict[str, Any]) -> None:
        law = binary
        result = _update(law)
        g1, g1_plus = law["g"][:, 1], law["partner"][:, 1]
        a = law["a"]
        residual = law["y"] - result.targeted.observed
        ate = a / g1 - (1 - a) / (1 - g1)
        # The JCvdL direction d/dh C(G_h) = -G'/lG^2, with G' the finite difference.
        lg = np.where(a == 1, g1, 1 - g1)
        derivative = (g1_plus - g1) * (2 * a - 1) / lg**2
        for column in (ate, derivative):
            assert abs(np.mean(column * residual)) / np.mean(np.abs(column)) <= 1e-8

    def test_fixed_weights(self) -> None:
        law = _law(seed=2)
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.3, 2.0, size=law["a"].shape[0])
        law["weights"] = weights / weights.mean()
        result = _update(law)
        assert np.max(np.abs(_derivative_scores(law, result.targeted))) <= 1e-8
        assert np.min(np.abs(_derivative_scores(law, _plain(law).targeted))) > 1e-2

    @pytest.mark.filterwarnings("ignore:the linear fluctuation produced")
    def test_linear_fluctuation(self, binary: dict[str, Any]) -> None:
        spec = replace(SPEC, fluctuation="linear")
        result = _update(binary, spec)
        assert result.fluctuation.method == "linear"
        assert np.max(np.abs(_derivative_scores(binary, result.targeted))) <= 1e-8


def test_w2_the_estimate_moves_away_from_the_plain_tmle(binary: dict[str, Any]) -> None:
    derivative = _estimate(_update(binary).targeted, binary["weights"])
    plain = _estimate(_plain(binary).targeted, binary["weights"])
    assert abs(derivative - plain) > 1e-3


@pytest.mark.parametrize("factors", [(-1.0, 1.0), (1.0, 0.5), (3.0, 3.0), (0.5, -1.0)])
def test_w3_rescaling_a_derivative_column_changes_nothing(
    binary: dict[str, Any], factors: tuple[float, float]
) -> None:
    law = binary
    reference = _update(law)
    ratio = derivative_ratio(law["g"], law["partner"])
    submodel, _ = derivative_columns(_base(law), ratio, law["a"], law["observed"])
    multiplier = np.array([1.0, 1.0, *factors])
    rescaled = submodel.map_arms(lambda values: values * multiplier)
    fluctuation = solve_submodel(
        law["y"], law["initial"], rescaled, law["weights"], law["observed"], SPEC, warn=False
    )
    for level in law["arms"]:
        np.testing.assert_allclose(
            fluctuation.targeted.arms[level], reference.targeted.arms[level], atol=1e-9
        )


def test_w4_the_counterfactual_block_moves_the_other_arms_rows(binary: dict[str, Any]) -> None:
    law = binary
    result = _update(law)
    ratio = derivative_ratio(law["g"], law["partner"])
    submodel, _ = derivative_columns(_base(law), ratio, law["a"], law["observed"])
    epsilon = result.fluctuation.epsilon
    for j, level in enumerate(law["arms"]):
        others = law["a"] != level
        moved = epsilon[2 + j] * submodel.arms[level][others, 2 + j]
        assert np.max(np.abs(moved)) > 1e-4


def test_w5_three_arms_need_one_derivative_column_each() -> None:
    law = _law(arms=3, seed=3)
    result = _update(law)
    assert result.names == ("h0", "h1", "h2", "dh0", "dh1", "dh2")
    assert np.max(np.abs(_derivative_scores(law, result.targeted))) <= 1e-8
    # One shared derivative column, the sum of the three, leaves an arm unsolved.
    ratio = derivative_ratio(law["g"], law["partner"])
    submodel, _ = derivative_columns(_base(law), ratio, law["a"], law["observed"])

    def pooled(values: np.ndarray) -> np.ndarray:
        return np.column_stack([values[:, :3], values[:, 3:].sum(axis=1)])

    shared = Submodel(
        pooled(submodel.observed),
        {level: pooled(values) for level, values in submodel.arms.items()},
        ("h0", "h1", "h2", "dh"),
        "mean",
        dict(submodel.arm_columns),
    )
    fluctuation = solve_submodel(
        law["y"], law["initial"], shared, law["weights"], law["observed"], SPEC, warn=False
    )
    assert np.max(np.abs(_derivative_scores(law, fluctuation.targeted))) > 1e-2


class TestRefusals:
    def test_a_partner_equal_to_the_mechanism(self, binary: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="derivative column is zero"):
            _update(binary, partner=binary["g"].copy())

    def test_a_partner_equal_after_the_bound_on_one_arm(self, binary: dict[str, Any]) -> None:
        law = binary
        partner = law["partner"].copy()
        # Arm 0's rows take the mechanism itself; arm 1's keep the partner.  Off the simplex
        # on purpose: each column is read only on its own arm's rows.
        partner[law["a"] == 0.0] = law["g"][law["a"] == 0.0]
        with pytest.raises(ValueError, match="treatment 0 and an observed outcome, so that"):
            _update(law, partner=partner)

    def test_a_partner_off_by_a_constant_factor(self, binary: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="by a constant factor"):
            _update(binary, partner=binary["g"] * 0.9)

    def test_a_solve_that_does_not_converge(self, binary: dict[str, Any]) -> None:
        # Each outer iteration runs a full inner Newton solve, so only a tolerance no
        # floating-point score can meet forces the failure.
        spec = replace(SPEC, max_iter=1, tol=0.0)
        with pytest.raises(ValueError, match=r"did not converge in 1 iterations"):
            _update(binary, spec)

    def test_one_step_targeting(self, binary: dict[str, Any]) -> None:
        with pytest.raises(CapabilityError, match=re.escape(DERIVATIVE_ONE_STEP_REFUSAL.message())):
            _update(binary, replace(SPEC, targeting="one_step"))

    def test_target_weights(self, binary: dict[str, Any]) -> None:
        with pytest.raises(
            CapabilityError, match=re.escape(DERIVATIVE_TARGET_WEIGHTS_REFUSAL.message())
        ):
            _update(binary, replace(SPEC, target_weights=True))

    def test_derivative_columns_extend_only_the_mean_submodel(self, binary: dict[str, Any]) -> None:
        base = _base(binary)
        contrast = replace(base, group="att", arm_columns={})
        with pytest.raises(ValueError, match="one-column-per-arm 'mean' submodel"):
            derivative_columns(contrast, np.ones((binary["a"].shape[0], 2)), binary["a"])
