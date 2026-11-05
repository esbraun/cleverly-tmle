r"""Is the EP learner's debiasing exact, and does each of its components matter?

The EP learner (``cleverly.estimators.ep``) adjusts the outcome regressions by one pooled
offset-logistic regression and fits the curve to the adjusted contrast (P1 plan 2.5).  Its
arithmetic is checked here on finite-support laws held as frequency-weighted populations:
each support point is one row whose weight is its probability, and the outcome is its
conditional mean.  The logistic score is linear in the outcome, so these rows solve the
population score equation, and the estimator's arithmetic equals the population value.

``W = (V, W2)`` with ``V`` in ``{0, 1, 2}`` and ``W2`` in ``{0, 1}``.  The curve is indexed by
``V`` alone, so ``W2`` is marginalised and the propensity varies inside each ``V`` cell: with a
saturated basis in ``W`` every weighting would solve the same equations, and the ``1 / pi``
weight would be invisible.  The laws (plan 4.1):

- **L-disc**: the law itself;
- **L-wrong**: the outcome regression is off by a function of ``W2`` that no feature spans,
  and the propensity is exact (the debiasing has something to repair);
- **L-wrong-pi**: the outcome regression is exact and the propensity is wrong;
- **L-miss**: L-wrong with missing outcomes, ``Pi(a, W)`` in ``[0.6, 0.9]``;
- **L-three**: L-wrong with a third arm and the declared pair ``(2, 0)``.

Each witness of plan 4.2 pairs with a deliberate mutation, patched into the module, that must
fail it.  The end-to-end tests at the bottom run the estimator through the study layer at the
oracle learners on ``tests.discrete_law`` and check the arrays the fit reads off the nuisance
fit: the ``1 / pi`` and ``1 / Pi`` weights, the outcome scale ``b - a`` and the clip.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.special import expit, logit

from cleverly.estimators import ep

# ---------------------------------------------------------------- the population laws

P_W = np.array([[0.20, 0.15], [0.10, 0.20], [0.25, 0.10]])  # [v, w2]
PI_1 = np.array([[0.30, 0.65], [0.55, 0.35], [0.70, 0.40]])  # P(A = 1 | W)
MU = np.array(  # [a, v, w2]
    [
        [[0.30, 0.45], [0.25, 0.50], [0.40, 0.35]],
        [[0.55, 0.60], [0.60, 0.85], [0.50, 0.75]],
        [[0.70, 0.40], [0.65, 0.80], [0.20, 0.55]],
    ]
)
#: The outcome-regression error of L-wrong: a function of ``W2`` the features do not span.
DELTA = np.array(  # [a, v, w2]
    [
        [[0.10, -0.15], [0.10, -0.15], [0.10, -0.15]],
        [[-0.12, 0.08], [-0.12, 0.08], [-0.12, 0.08]],
        [[-0.15, 0.12], [-0.15, 0.12], [-0.15, 0.12]],
    ]
)
PI_OBSERVED = np.array(  # [a, v, w2]
    [
        [[0.60, 0.90], [0.80, 0.65], [0.70, 0.85]],
        [[0.85, 0.60], [0.70, 0.90], [0.65, 0.75]],
        [[0.75, 0.80], [0.60, 0.70], [0.90, 0.65]],
    ]
)
THIRD_ARM = 0.2


@dataclass(frozen=True)
class Population:
    """The rows of one law, with the nuisances the fit is handed."""

    v: np.ndarray
    a: np.ndarray
    mass: np.ndarray
    y: np.ndarray
    observed: np.ndarray
    density: np.ndarray  # the supplied pi_n(A | W) Pi_n(A, W) at each row
    propensity: np.ndarray  # the supplied pi_n(A | W) alone
    mu_n: np.ndarray  # [a, row] the supplied outcome regression
    true_tau: np.ndarray  # mu_0(t, W) - mu_0(r, W) at each row
    true_t: np.ndarray
    true_r: np.ndarray
    pair: tuple[float, float]


def arm_probabilities(three: bool) -> np.ndarray:
    """``P(A = a | W)``, indexed ``[a, v, w2]``."""
    if not three:
        return np.stack([1.0 - PI_1, PI_1])
    rest = 1.0 - THIRD_ARM
    return np.stack([rest * (1.0 - PI_1), rest * PI_1, np.full_like(PI_1, THIRD_ARM)])


def population(
    *,
    wrong_mu: bool = False,
    wrong_pi: bool = False,
    missing: bool = False,
    three: bool = False,
) -> Population:
    arms = 3 if three else 2
    pair = (2.0, 0.0) if three else (1.0, 0.0)
    probs = arm_probabilities(three)
    supplied = probs
    if wrong_pi:
        tilted = np.clip(PI_1 + np.array([[0.25, -0.20], [-0.15, 0.25], [-0.30, 0.20]]), 0.1, 0.9)
        supplied = np.stack([1.0 - tilted, tilted])
    mu_n = MU[:arms] + (DELTA[:arms] if wrong_mu else 0.0)
    rows: dict[str, list[float]] = {key: [] for key in ("v", "a", "mass", "y", "o", "d", "g")}
    index: list[tuple[int, int]] = []
    for v, w2, a in itertools.product(range(3), range(2), range(arms)):
        seen = PI_OBSERVED[a, v, w2] if missing else 1.0
        for delta in (1, 0) if missing else (1,):
            share = seen if delta else 1.0 - seen
            rows["v"].append(v)
            rows["a"].append(a)
            rows["mass"].append(P_W[v, w2] * probs[a, v, w2] * share)
            rows["y"].append(MU[a, v, w2] if delta else 0.0)
            rows["o"].append(float(delta))
            rows["d"].append(supplied[a, v, w2] * (PI_OBSERVED[a, v, w2] if missing else 1.0))
            rows["g"].append(supplied[a, v, w2])
            index.append((v, w2))
    cells = tuple(np.array(index).T)
    t, r = int(pair[0]), int(pair[1])
    return Population(
        v=np.array(rows["v"], dtype=float),
        a=np.array(rows["a"], dtype=float),
        mass=np.array(rows["mass"]),
        y=np.array(rows["y"]),
        observed=np.array(rows["o"]),
        density=np.array(rows["d"]),
        propensity=np.array(rows["g"]),
        mu_n=np.stack([mu_n[arm][cells] for arm in range(arms)]),
        true_tau=MU[t][cells] - MU[r][cells],
        true_t=MU[t][cells],
        true_r=MU[r][cells],
        pair=pair,
    )


def linear(v: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(v), v])


def indicators(v: np.ndarray) -> np.ndarray:
    return np.column_stack([(v == level).astype(float) for level in range(3)])


def run(law: Population, design: np.ndarray, scale: str, *, weight: Any = None) -> Any:
    t, r = int(law.pair[0]), int(law.pair[1])
    debiased = ep.debias(
        y=law.y,
        treatment=law.a,
        observed=law.observed,
        initial_observed=law.mu_n[law.a.astype(int), np.arange(law.a.size)],
        initial_t=law.mu_n[t],
        initial_r=law.mu_n[r],
        weight=law.mass / law.density if weight is None else weight,
        design=design,
        pair=law.pair,
        scale=scale,
    )
    beta, _ = ep.final_stage(debiased, design, law.mass, scale)
    return debiased, beta


def blp(law: Population, design: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """The population least-squares projection of ``contrast`` on ``design``."""
    root = np.sqrt(law.mass)
    return np.linalg.lstsq(design * root[:, None], contrast * root, rcond=None)[0]


def log_crr_by_cell(law: Population) -> np.ndarray:
    """``log E[mu_0(t, W) | V] / E[mu_0(r, W) | V]``, the saturated log-CRR projection."""
    out = []
    for level in range(3):
        at = law.v == level
        out.append(
            np.log(np.sum(law.mass[at] * law.true_t[at]) / np.sum(law.mass[at] * law.true_r[at]))
        )
    return np.array(out)


def truth(law: Population) -> np.ndarray:
    return blp(law, linear(law.v), law.true_tau)


def t_learner(law: Population) -> np.ndarray:
    t, r = int(law.pair[0]), int(law.pair[1])
    return blp(law, linear(law.v), law.mu_n[t] - law.mu_n[r])


#: The arithmetic runs one Newton solve to convergence; this is the window on its result.
EXACT = 1e-10


# ---------------------------------------------------------------- the laws themselves


class TestTheLawsHaveSomethingToShow:
    def test_the_t_learner_is_off_at_the_wrong_regression(self) -> None:
        for law in (population(wrong_mu=True), population(wrong_mu=True, three=True)):
            assert np.max(np.abs(t_learner(law) - truth(law))) > 0.02

    def test_the_propensity_varies_inside_each_modifier_cell(self) -> None:
        assert np.min(np.ptp(PI_1, axis=1)) > 0.2

    def test_the_projection_is_not_saturated(self) -> None:
        law = population()
        residual = law.true_tau - linear(law.v) @ truth(law)
        assert np.max(np.abs(residual)) > 0.05


# ---------------------------------------------------------------- double robustness


class TestTheDebiasingIsExact:
    @pytest.mark.parametrize(
        "law",
        [
            pytest.param(population(), id="L-disc"),
            pytest.param(population(wrong_mu=True), id="L-wrong"),
            pytest.param(population(wrong_pi=True), id="L-wrong-pi"),
            pytest.param(population(wrong_mu=True, missing=True), id="L-miss"),
            pytest.param(population(wrong_mu=True, three=True), id="L-three"),
        ],
    )
    def test_the_cate_projection_is_the_truth(self, law: Population) -> None:
        _, beta = run(law, linear(law.v), "difference")
        np.testing.assert_allclose(beta, truth(law), atol=EXACT, rtol=0)

    def test_the_debiasing_coefficient_is_nonzero_at_the_wrong_regression(self) -> None:
        debiased, _ = run(population(wrong_mu=True), linear(population().v), "difference")
        assert np.min(np.abs(debiased.coefficients)) > 1e-3

    def test_the_debiasing_coefficient_is_zero_at_the_right_regression(self) -> None:
        """L-wrong-pi: the score is zero at ``b = 0`` whatever the weights."""
        law = population(wrong_pi=True)
        debiased, _ = run(law, linear(law.v), "difference")
        np.testing.assert_allclose(debiased.coefficients, 0.0, atol=EXACT)

    def test_the_score_is_nonzero_before_and_solved_after(self) -> None:
        law = population(wrong_mu=True)
        debiased, _ = run(law, linear(law.v), "difference")
        assert np.min(np.abs(debiased.score_before)) > 1e-3
        assert np.max(np.abs(debiased.score_after)) < EXACT

    def test_the_contrast_matches_the_cate_at_every_cell_when_saturated(self) -> None:
        """L-disc with an asymmetric blip and ``Indicators(V)``: per-cell exact."""
        law = population(wrong_mu=True)
        _, beta = run(law, indicators(law.v), "difference")
        np.testing.assert_allclose(beta, blp(law, indicators(law.v), law.true_tau), atol=EXACT)
        assert np.ptp(beta) > 0.05


class TestTheLogRatio:
    @pytest.mark.parametrize(
        "law",
        [
            pytest.param(population(wrong_mu=True), id="L-wrong"),
            pytest.param(population(wrong_mu=True, missing=True), id="L-miss"),
            pytest.param(population(wrong_mu=True, three=True), id="L-three"),
        ],
    )
    def test_the_saturated_log_crr_is_the_truth(self, law: Population) -> None:
        debiased, beta = run(law, indicators(law.v), "log_ratio")
        np.testing.assert_allclose(beta, log_crr_by_cell(law), atol=EXACT, rtol=0)
        assert np.max(np.abs(debiased.score_after)) < EXACT
        assert debiased.coefficients.size == 6

    def test_the_t_learner_log_crr_is_off(self) -> None:
        law = population(wrong_mu=True)
        plug_in = []
        for level in range(3):
            at = law.v == level
            plug_in.append(
                np.log(
                    np.sum(law.mass[at] * law.mu_n[1][at]) / np.sum(law.mass[at] * law.mu_n[0][at])
                )
            )
        assert np.max(np.abs(np.array(plug_in) - log_crr_by_cell(law))) > 0.02

    def test_a_linear_basis_leaves_a_gap_on_the_log_ratio(self) -> None:
        """Plan 2.5: at a fixed unsaturated size the two-block debiasing is not exact.

        The log-ratio projection onto ``Linear(V)`` solves ``E[phi {(mu_t + mu_r) expit(eta)
        - mu_t}] = 0``; the EP fit solves the arm-wise marginal equations instead, and its
        final stage reads ``mu*``, which still carries the regression's error inside each
        ``V`` cell.  The gap is reported, not hidden.
        """
        law = population(wrong_mu=True)
        _, beta = run(law, linear(law.v), "log_ratio")
        exact = _log_crr_projection(law, linear(law.v))
        gap = np.max(np.abs(beta - exact))
        assert 1e-6 < gap < 0.05

    def test_the_debiased_arms_stay_inside_the_unit_interval(self) -> None:
        law = population(wrong_mu=True, missing=True)
        for scale in ("difference", "log_ratio"):
            debiased, _ = run(law, indicators(law.v), scale)
            for arm in (debiased.targeted_t, debiased.targeted_r):
                assert np.all((arm > 0.0) & (arm < 1.0))


def _log_crr_projection(law: Population, design: np.ndarray) -> np.ndarray:
    """Newton on the population equation of the log-CRR projection (plan 2.1)."""
    beta = np.zeros(design.shape[1])
    total = law.true_t + law.true_r
    for _ in range(100):
        p = expit(design @ beta)
        score = design.T @ (law.mass * (total * p - law.true_t))
        hessian = (design * (law.mass * total * p * (1 - p))[:, None]).T @ design
        beta = beta - np.linalg.solve(hessian, score)
    return beta


# ---------------------------------------------------------------- the mutations


class TestEachComponentMatters:
    """Plan 4.2: each mutation, patched into the module, breaks its witness."""

    def error(self, law: Population, scale: str = "difference") -> float:
        _, beta = run(law, linear(law.v), scale)
        return float(np.max(np.abs(beta - truth(law))))

    def test_the_targeting_sign(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)
        increments = ep._arm_increments
        monkeypatch.setattr(
            ep, "_arm_increments", lambda *args: tuple(-x for x in increments(*args))
        )
        baseline = float(np.max(np.abs(t_learner(law) - truth(law))))
        assert self.error(law) > 1.5 * baseline

    def test_a_dropped_basis_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)
        features = ep.debias_features

        def without_the_slope(*args: Any) -> np.ndarray:
            x = features(*args).copy()
            x[:, 1] = 0.0
            return x

        monkeypatch.setattr(ep, "debias_features", without_the_slope)
        debiased, _ = run(law, linear(law.v), "difference")
        x = features(linear(law.v), law.a, law.pair, "difference")
        after = x.T @ (law.mass / law.density * (law.y - _fitted_observed(law, debiased)))
        assert abs(after[1]) > 1e-3
        assert self.error(law) > 1e-3

    def test_the_contrast_direction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)
        monkeypatch.setattr(
            ep.Debiasing, "contrast", property(lambda self: self.targeted_r - self.targeted_t)
        )
        _, beta = run(law, linear(law.v), "difference")
        np.testing.assert_allclose(beta, -truth(law), atol=EXACT)
        assert self.error(law) > 0.1

    def test_the_third_arm_carries_no_feature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True, three=True)
        assert self.error(law) < EXACT

        def one_against_the_rest(design: Any, treatment: Any, pair: Any, scale: str) -> Any:
            at_t = (np.asarray(treatment) == pair[0]).astype(float)
            return np.asarray(design) * (2.0 * at_t - 1.0)[:, None]

        monkeypatch.setattr(ep, "debias_features", one_against_the_rest)
        assert self.error(law) > 1e-3

    def test_the_offset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)
        newton = ep.logistic_newton
        monkeypatch.setattr(
            ep,
            "logistic_newton",
            lambda z, y, w, *, label, offset=None: newton(z, y, w, label=label),
        )
        assert self.error(law) > 1e-3

    def test_the_two_a_minus_one_factor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)

        def unsigned(design: Any, treatment: Any, pair: Any, scale: str) -> Any:
            a = np.asarray(treatment)
            inside = ((a == pair[0]) | (a == pair[1])).astype(float)
            return np.asarray(design) * inside[:, None]

        monkeypatch.setattr(ep, "debias_features", unsigned)
        debiased, _ = run(law, linear(law.v), "difference")
        fitted = _fitted_observed(law, debiased, unsigned)
        after = _signed_features(law).T @ (law.mass / law.density * (law.y - fitted))
        assert np.max(np.abs(after)) > 1e-3
        assert self.error(law) > 1e-3

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_the_inverse_propensity_weight(self, scale: str) -> None:
        """Drop ``1 / pi`` and keep the mass: both scales move off the truth.

        On the log ratio this is the only witness of the propensity weight: the ep-crr
        rescue cell does not see it (design review B1 of PR-b).
        """
        law = population(wrong_mu=True)
        design, exact = (
            (linear(law.v), truth(law))
            if scale == "difference"
            else (indicators(law.v), log_crr_by_cell(law))
        )
        _, right = run(law, design, scale)
        np.testing.assert_allclose(right, exact, atol=EXACT)
        _, beta = run(law, design, scale, weight=law.mass)
        assert np.max(np.abs(beta - exact)) > 1e-3

    def test_the_inverse_observation_weight(self) -> None:
        """L-miss: keep ``Delta`` and ``1 / pi``, drop ``1 / Pi``; both scales move."""
        law = population(wrong_mu=True, missing=True)
        for scale, design, exact in (
            ("difference", linear(law.v), truth(law)),
            ("log_ratio", indicators(law.v), log_crr_by_cell(law)),
        ):
            _, beta = run(law, design, scale, weight=law.mass / law.propensity)
            assert np.max(np.abs(beta - exact)) > 1e-3

    def test_the_log_ratio_second_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)
        features = ep.debias_features

        def one_block(*args: Any) -> np.ndarray:
            x = features(*args).copy()
            x[:, 3:] = 0.0
            return x

        monkeypatch.setattr(ep, "debias_features", one_block)
        debiased, beta = run(law, indicators(law.v), "log_ratio")
        assert np.max(np.abs(beta - log_crr_by_cell(law))) > 1e-3
        x = features(indicators(law.v), law.a, law.pair, "log_ratio")
        fitted = _fitted_observed(law, debiased, scale="log_ratio", design=indicators(law.v))
        after = x.T @ (law.mass / law.density * (law.y - fitted))
        assert np.max(np.abs(after[3:])) > 1e-3

    def test_the_log_ratio_final_stage_weight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        law = population(wrong_mu=True)
        newton = ep.logistic_newton

        def unweighted(z: Any, y: Any, w: Any, *, label: str, offset: Any = None) -> Any:
            if label == "EP log-ratio final stage":
                w = law.mass
            return newton(z, y, w, label=label, offset=offset)

        monkeypatch.setattr(ep, "logistic_newton", unweighted)
        _, beta = run(law, indicators(law.v), "log_ratio")
        assert np.max(np.abs(beta - log_crr_by_cell(law))) > 1e-3


def _signed_features(law: Population) -> np.ndarray:
    return (
        linear(law.v)
        * ((law.a == law.pair[0]).astype(float) - (law.a == law.pair[1]).astype(float))[:, None]
    )


def _fitted_observed(
    law: Population,
    debiased: Any,
    features: Any = None,
    *,
    scale: str = "difference",
    design: Any = None,
) -> np.ndarray:
    """``mu*(A_i, W_i)`` at each row's observed arm, as the debiasing fit it."""
    design = linear(law.v) if design is None else design
    build = ep.debias_features if features is None else features
    x = build(design, law.a, law.pair, scale)
    offset = logit(law.mu_n[law.a.astype(int), np.arange(law.a.size)])
    return expit(offset + x @ debiased.coefficients)


# ---------------------------------------------------------------- scores on random laws


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("scale", ["difference", "log_ratio"])
def test_the_score_is_solved_on_random_laws(seed: int, scale: str) -> None:
    """Plan 4.3: after debiasing every declared score is below ``1e-8 (1 + |scale|)``."""
    rng = np.random.default_rng(seed)
    n = 400
    v = rng.uniform(-1, 1, n)
    a = rng.integers(0, 3, n).astype(float)
    design = np.column_stack([np.ones(n), v, np.cos(np.pi * v)])
    mu = rng.uniform(0.05, 0.95, (3, n))
    debiased = ep.debias(
        y=rng.uniform(0, 1, n),
        treatment=a,
        observed=(rng.uniform(size=n) < 0.8).astype(float),
        initial_observed=mu[a.astype(int), np.arange(n)],
        initial_t=mu[2],
        initial_r=mu[0],
        weight=rng.uniform(1.0, 6.0, n),
        design=design,
        pair=(2.0, 0.0),
        scale=scale,
    )
    assert np.max(np.abs(debiased.score_after)) < 1e-8 * (1 + 1)
    for arm in (debiased.targeted_t, debiased.targeted_r):
        assert np.all((arm > 0.0) & (arm < 1.0))


def test_the_risk_is_the_one_step_risk_of_the_curve() -> None:
    """The CATE one-step risk is ``E[(theta - tau)^2] - E[tau^2]`` at exact pseudo-outcomes."""
    law = population()
    theta = linear(law.v) @ truth(law)
    risk = ep.one_step_risk(theta, law.true_t, law.true_r, "difference", law.mass)
    expected = np.sum(law.mass * ((theta - law.true_tau) ** 2 - law.true_tau**2)) / np.sum(law.mass)
    assert risk == pytest.approx(expected, abs=1e-14)
    # The log ratio's loss is minimised at the per-cell log CRR by a saturated curve.
    saturated = indicators(law.v) @ log_crr_by_cell(law)
    best = ep.one_step_risk(saturated, law.true_t, law.true_r, "log_ratio", law.mass)
    for shift in (-0.05, 0.05):
        moved = ep.one_step_risk(saturated + shift, law.true_t, law.true_r, "log_ratio", law.mass)
        assert moved > best
