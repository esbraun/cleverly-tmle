"""X82 part H: marginal sensitivity model bounds (``cleverly.sensitivity.marginal``).

The exact checks run on a finite law expanded into rows in proportion to its probabilities, so
a sample mean over the rows is the population expectation. The sharp value of each arm bound is
the linear program of the model, solved by ``scipy.optimize.linprog``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import marginal as marginal_module
from cleverly.sensitivity.marginal import (
    MSM_BOUND_INSTRUMENT,
    _binary_side,
    _continuous_side,
    marginal_bound_refusal,
    marginal_sensitivity_bounds,
)
from cleverly.validation import _interpretation as interpretation
from cleverly.validation._interpretation import (
    DESCRIPTIVE_SENTENCE,
    CalibrationScope,
    VerdictEvidence,
)
from tests.conftest import linear_drtmle

pytestmark = pytest.mark.xdist_group("marginal_sensitivity")


# ---------------------------------------------------------------------------------------
# The finite law


#: P(W = w), P(A = 1 | W = w), and the law of Y given (A, W) on the support SUPPORT.
W_PROBABILITY = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
PROPENSITY = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
SUPPORT = (0.0, 1.0, 2.0, 4.0, 7.0)
#: Y | A = a, W = w, one row of probabilities over SUPPORT per (a, w).
OUTCOME_LAW = {
    (1, 0): (Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(2, 10), Fraction(2, 10)),
    (1, 1): (Fraction(3, 10), Fraction(1, 10), Fraction(1, 10), Fraction(4, 10), Fraction(1, 10)),
    (1, 2): (Fraction(2, 10), Fraction(2, 10), Fraction(2, 10), Fraction(2, 10), Fraction(2, 10)),
    (0, 0): (Fraction(4, 10), Fraction(3, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)),
    (0, 1): (Fraction(1, 10), Fraction(4, 10), Fraction(2, 10), Fraction(2, 10), Fraction(1, 10)),
    (0, 2): (Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(3, 10), Fraction(4, 10)),
}
#: Binary outcome law P(Y = 1 | A = a, W = w).
BINARY_LAW = {
    (1, 0): Fraction(1, 5),
    (1, 1): Fraction(3, 5),
    (1, 2): Fraction(9, 10),
    (0, 0): Fraction(1, 10),
    (0, 1): Fraction(2, 5),
    (0, 2): Fraction(7, 10),
}


def lp_sharp(values: np.ndarray, probabilities: np.ndarray, gamma: float, upper: bool) -> float:
    """sup (or inf) of E[w Y] over Gamma^{-1} <= w <= Gamma with E[w] = 1."""
    sign = -1.0 if upper else 1.0
    solved = linprog(
        sign * values * probabilities,
        A_eq=probabilities[None, :],
        b_eq=[1.0],
        bounds=[(1.0 / gamma, gamma)] * len(values),
        method="highs",
    )
    assert solved.success
    return float(sign * solved.fun)


def dual_value(values: np.ndarray, probabilities: np.ndarray, gamma: float, q: float) -> float:
    """q + E[phi_Gamma(Y - q)] at one q."""
    residual = values - q
    phi = np.where(residual > 0, gamma, 1.0 / gamma) * residual
    return float(q + np.sum(probabilities * phi))


def quantile(values: np.ndarray, probabilities: np.ndarray, tau: float) -> float:
    """The smallest support point whose cumulative probability reaches tau."""
    cumulative = np.cumsum(probabilities)
    return float(values[np.searchsorted(cumulative, tau - 1e-12)])


def expand(binary: bool) -> dict[str, np.ndarray]:
    """Rows in exact proportion to the finite law (common denominator 800)."""
    w_rows, a_rows, y_rows = [], [], []
    total = 800
    for w, (pw, pa) in enumerate(zip(W_PROBABILITY, PROPENSITY, strict=True)):
        for a, share in ((1, pa), (0, 1 - pa)):
            cell = pw * share
            if binary:
                p = BINARY_LAW[(a, w)]
                for y, py in ((1.0, p), (0.0, 1 - p)):
                    count = cell * py * total
                    assert count.denominator == 1
                    w_rows += [w] * int(count)
                    a_rows += [a] * int(count)
                    y_rows += [y] * int(count)
            else:
                for y, py in zip(SUPPORT, OUTCOME_LAW[(a, w)], strict=True):
                    count = cell * py * total
                    assert count.denominator == 1
                    w_rows += [w] * int(count)
                    a_rows += [a] * int(count)
                    y_rows += [y] * int(count)
    return {"w": np.asarray(w_rows), "a": np.asarray(a_rows, float), "y": np.asarray(y_rows)}


def law_mean(a: int, w: int, binary: bool) -> float:
    if binary:
        return float(BINARY_LAW[(a, w)])
    return float(sum(y * float(p) for y, p in zip(SUPPORT, OUTCOME_LAW[(a, w)], strict=True)))


def law_values(a: int, w: int, binary: bool) -> tuple[np.ndarray, np.ndarray]:
    if binary:
        p = float(BINARY_LAW[(a, w)])
        return np.array([0.0, 1.0]), np.array([1.0 - p, p])
    return np.asarray(SUPPORT), np.asarray([float(p) for p in OUTCOME_LAW[(a, w)]])


def sharp_arm_bound(arm: int, gamma: float, upper: bool, binary: bool) -> float:
    """E[1{A = arm} Y + 1{A != arm} rho(W)] with rho the LP optimum."""
    total = 0.0
    for w, (pw, pa) in enumerate(zip(W_PROBABILITY, PROPENSITY, strict=True)):
        share = float(pa) if arm == 1 else 1.0 - float(pa)
        values, probabilities = law_values(arm, w, binary)
        rho = lp_sharp(values, probabilities, gamma, upper)
        total += float(pw) * (share * law_mean(arm, w, binary) + (1.0 - share) * rho)
    return total


# ---------------------------------------------------------------------------------------
# The linear program and its dual


@pytest.mark.parametrize("gamma", [1.3, 2.0, 4.0])
@pytest.mark.parametrize("upper", [True, False])
def test_the_dual_at_the_tau_quantile_equals_the_linear_program(gamma: float, upper: bool) -> None:
    tau = gamma / (1 + gamma) if upper else 1 / (1 + gamma)
    gaps = []
    for key in OUTCOME_LAW:
        values, probabilities = law_values(key[0], key[1], binary=False)
        sharp = lp_sharp(values, probabilities, gamma, upper)
        q = quantile(values, probabilities, tau)
        if upper:
            dual = dual_value(values, probabilities, gamma, q)
        else:
            dual = -dual_value(-values, probabilities, gamma, -q)
        assert dual == pytest.approx(sharp, rel=1e-10, abs=1e-12)
        # Mutation control: the other side's quantile is still valid but misses the
        # optimum on some cell. A discrete law can put both quantiles on one atom.
        wrong = quantile(values, probabilities, 1 - tau)
        if upper:
            gap = dual_value(values, probabilities, gamma, wrong) - sharp
        else:
            gap = sharp + dual_value(-values, probabilities, gamma, -wrong)
        assert gap >= -1e-12
        gaps.append(gap)
    assert max(gaps) > 1e-6


def test_weak_duality_any_quantile_gives_a_valid_bound() -> None:
    values, probabilities = law_values(1, 0, binary=False)
    sharp = lp_sharp(values, probabilities, 2.0, upper=True)
    median = quantile(values, probabilities, 0.5)
    assert dual_value(values, probabilities, 2.0, median) > sharp + 1e-6
    for q in np.linspace(-1.0, 8.0, 37):
        assert dual_value(values, probabilities, 2.0, q) >= sharp - 1e-12


@pytest.mark.parametrize("gamma", [1.25, 2.0, 5.0])
def test_the_binary_closed_form_is_the_linear_program(gamma: float) -> None:
    for p in np.linspace(0.01, 0.99, 25):
        values, probabilities = np.array([0.0, 1.0]), np.array([1 - p, p])
        upper = min(gamma * p, 1 - (1 - p) / gamma)
        lower = max(1 - gamma + gamma * p, p / gamma)
        assert upper == pytest.approx(lp_sharp(values, probabilities, gamma, True), abs=1e-10)
        assert lower == pytest.approx(lp_sharp(values, probabilities, gamma, False), abs=1e-10)
    # Mutation control: the odds-ratio form of another sensitivity model is not this bound.
    share = 0.3
    odds_form = gamma * share / (gamma * share + 1 - share)
    assert odds_form != pytest.approx(min(gamma * share, 1 - (1 - share) / gamma), abs=1e-6)


# ---------------------------------------------------------------------------------------
# The estimator at the population


def population_inputs(arm: int, binary: bool) -> dict[str, np.ndarray]:
    rows = expand(binary)
    e1 = np.asarray([float(PROPENSITY[w]) for w in rows["w"]])
    e = e1 if arm == 1 else 1.0 - e1
    mu = np.asarray([law_mean(arm, int(w), binary) for w in rows["w"]])
    treated = (rows["a"] == arm).astype(float)
    return {"y": rows["y"], "w": rows["w"], "treated": treated, "e": e, "mu": mu}


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("gamma", [1.5, 3.0])
def test_the_binary_side_reproduces_the_sharp_bound(arm: int, upper: bool, gamma: float) -> None:
    inputs = population_inputs(arm, binary=True)
    side = _binary_side(
        inputs["y"], inputs["treated"], inputs["e"], inputs["mu"], gamma, upper=upper
    )
    assert side.value == pytest.approx(sharp_arm_bound(arm, gamma, upper, True), rel=1e-12)


def test_the_binary_side_is_doubly_valid() -> None:
    """Exact mu and a wrong e: sharp. Exact e and a wrong mu: still a valid upper bound."""
    gamma = 2.0
    inputs = population_inputs(1, binary=True)
    sharp = sharp_arm_bound(1, gamma, True, True)
    wrong_e = np.clip(inputs["e"] + 0.15 * (inputs["w"] - 1), 0.05, 0.95)
    side = _binary_side(inputs["y"], inputs["treated"], wrong_e, inputs["mu"], gamma, True)
    assert side.value == pytest.approx(sharp, rel=1e-12)
    wrong_mu = np.full_like(inputs["mu"], 0.5)
    side = _binary_side(inputs["y"], inputs["treated"], inputs["e"], wrong_mu, gamma, True)
    assert side.value >= sharp - 1e-12
    # Mutation control: with both nuisances wrong the bound is no longer the sharp value,
    # so the exact checks above see each nuisance.
    side = _binary_side(inputs["y"], inputs["treated"], wrong_e, wrong_mu, gamma, True)
    assert abs(side.value - sharp) > 1e-3


class Oracle:
    """A regressor that ignores its training rows and returns a declared function of W."""

    def __init__(self, table: dict[int, float]) -> None:
        self.table = table

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"table": self.table}

    def set_params(self, **params: Any) -> Oracle:
        self.table = params["table"]
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> Oracle:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self.table[int(w)] for w in x[:, 0]])


def oracle_tables(arm: int, gamma: float, upper: bool) -> tuple[dict[int, float], dict[int, float]]:
    """The tau-quantile of sign * Y and the conditional mean of its tail pseudo-outcome."""
    tau = gamma / (1 + gamma)
    sign = 1.0 if upper else -1.0
    quantiles, tails = {}, {}
    for w in range(3):
        values, probabilities = law_values(arm, w, binary=False)
        signed = sign * values
        order = np.argsort(signed)
        q = quantile(signed[order], probabilities[order], tau)
        quantiles[w] = q
        tails[w] = float(np.sum(probabilities * (q + np.maximum(signed - q, 0.0) / (1 - tau))))
    return quantiles, tails


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("upper", [True, False])
def test_the_continuous_side_reproduces_the_sharp_bound(arm: int, upper: bool) -> None:
    gamma = 2.0
    inputs = population_inputs(arm, binary=False)
    quantiles, tails = oracle_tables(arm, gamma, upper)
    assignment = np.arange(inputs["y"].shape[0]) % 5
    side = _continuous_side(
        inputs["w"][:, None].astype(float),
        inputs["y"],
        inputs["treated"],
        inputs["e"],
        inputs["mu"],
        gamma,
        upper,
        assignment,
        5,
        lambda tau: Oracle(quantiles),
        Oracle(tails),
        0,
    )
    assert side.value == pytest.approx(sharp_arm_bound(arm, gamma, upper, False), rel=1e-12)
    # Mutation control: the other side's quantile moves the value off the sharp bound.
    wrong, _ = oracle_tables(arm, gamma, not upper)
    moved = _continuous_side(
        inputs["w"][:, None].astype(float),
        inputs["y"],
        inputs["treated"],
        inputs["e"],
        inputs["mu"],
        gamma,
        upper,
        assignment,
        5,
        lambda tau: Oracle({w: -q for w, q in wrong.items()}),
        Oracle(tails),
        0,
    )
    assert abs(moved.value - side.value) > 1e-3


def test_the_correction_weight_is_the_odds_of_the_other_arm() -> None:
    """Mutation control: weight 1/e in place of (1 - e)/e breaks the exact value."""
    gamma = 2.0
    inputs = population_inputs(1, binary=True)
    wrong_mu = np.full_like(inputs["mu"], 0.5)
    y, t, e = inputs["y"], inputs["treated"], inputs["e"]
    tau = gamma / (1 + gamma)
    q = (wrong_mu > 1 - tau).astype(float)
    rho = np.minimum(1 - 1 / gamma + wrong_mu / gamma, gamma * wrong_mu)
    r = q + np.where(y - q > 0, gamma, 1 / gamma) * (y - q)
    shipped = _binary_side(y, t, e, wrong_mu, gamma, True).value
    mutated = float(np.mean(t * y + (1 - t) * rho + t / e * (r - rho)))
    exact = float(np.mean(t * y + (1 - t) * rho + t * (1 - e) / e * (r - rho)))
    assert shipped == pytest.approx(exact, rel=1e-14)
    assert abs(mutated - exact) > 1e-3


def test_the_tail_regression_trains_on_quantiles_fitted_outside_its_fold() -> None:
    """Every quantile fit that feeds a training pseudo-outcome excludes the rows it predicts."""
    calls: list[tuple[frozenset[int], frozenset[int]]] = []

    class Recording(LinearRegression):
        def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
            self._train = frozenset(np.round(x[:, 1]).astype(int).tolist())
            return super().fit(x[:, :1], y)

        def predict(self, x: Any) -> Any:
            calls.append((self._train, frozenset(np.round(x[:, 1]).astype(int).tolist())))
            return super().predict(x[:, :1])

    rng = np.random.default_rng(3)
    n = 200
    x = np.column_stack([rng.normal(size=n), np.arange(n)])
    treated = rng.binomial(1, 0.5, size=n).astype(float)
    y = x[:, 0] + rng.normal(size=n)
    assignment = np.arange(n) % 4
    _continuous_side(
        x,
        y,
        treated,
        np.full(n, 0.5),
        x[:, 0],
        2.0,
        True,
        assignment,
        4,
        lambda tau: Recording(),
        LinearRegression(),
        0,
    )
    assert calls
    for train, predicted in calls:
        assert not train & predicted


# ---------------------------------------------------------------------------------------
# Fits


def gaussian_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-w[:, 0])))
    y = w[:, 0] + a + rng.normal(size=n)
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": a, "Y": y})


def binary_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-w[:, 0])))
    y = rng.binomial(1, 1 / (1 + np.exp(-(w[:, 0] + a - 0.5))))
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": a, "Y": y})


def gaussian_fit(frame: pd.DataFrame | None = None, **options: Any) -> Any:
    frame = gaussian_frame() if frame is None else frame
    columns = {key: options.pop(key) for key in ("id", "weights") if key in options}
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
            q_bounds=(-12.0, 12.0),
            n_folds=5,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], **columns)
        .single()
    )


def binary_fit(frame: pd.DataFrame | None = None, **options: Any) -> Any:
    frame = binary_frame() if frame is None else frame
    return (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
            n_folds=5,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )


@pytest.fixture(scope="module")
def gaussian() -> Any:
    return gaussian_fit()


@pytest.fixture(scope="module")
def binary() -> Any:
    return binary_fit()


def aipw(result: Any, arm: float) -> float:
    nuisance = result.repeats[0].nuisance
    g = nuisance.propensity.bounded(result.config.g_bounds)
    index = list(nuisance.propensity.arms).index(arm)
    mu = nuisance.scaler.unscale_levels(np.asarray(nuisance.outcome.arms[arm]))
    treated = (np.asarray(result.data.treatment) == arm).astype(float)
    y = np.asarray(result.data.outcome, dtype=float)
    return float(np.mean(mu + treated / g[:, index] * (y - mu)))


@pytest.mark.parametrize("which", ["gaussian", "binary"])
def test_gamma_one_is_the_cross_fitted_aipw_estimate(which: str, request: Any) -> None:
    fit = request.getfixturevalue(which)
    bounds = marginal_sensitivity_bounds(fit, 1.0)
    row = bounds.rows[0]
    expected = aipw(fit, 1.0) - aipw(fit, 0.0)
    assert row.lower == pytest.approx(expected, rel=1e-12)
    assert row.upper == pytest.approx(expected, rel=1e-12)
    arm = marginal_sensitivity_bounds(fit, 1.0, estimand="ey1").rows[0]
    assert arm.lower == pytest.approx(aipw(fit, 1.0), rel=1e-12)


@pytest.mark.parametrize("which", ["gaussian", "binary"])
def test_the_bounds_widen_with_gamma(which: str, request: Any) -> None:
    fit = request.getfixturevalue(which)
    bounds = marginal_sensitivity_bounds(fit, (1.0, 1.5, 2.5))
    for before, after in zip(bounds.rows, bounds.rows[1:], strict=False):
        assert after.lower < before.lower
        assert after.upper > before.upper
    for row in bounds.rows:
        assert row.lower_limit < row.lower <= row.upper < row.upper_limit
    assert bounds.null == 0.0
    frame = bounds.to_frame()
    assert list(frame["gamma"]) == [1.0, 1.5, 2.5]


def test_the_interval_reads_each_side_curve(binary: Any) -> None:
    bounds = marginal_sensitivity_bounds(binary, 2.0, level=0.9)
    row = bounds.rows[0]
    z = 1.6448536269514722
    assert row.lower_limit == pytest.approx(row.lower - z * row.lower_std_error, rel=1e-12)
    assert row.upper_limit == pytest.approx(row.upper + z * row.upper_std_error, rel=1e-12)
    # The ATE side curve is the difference of the arm side curves.
    data = binary.data
    nuisance = binary.repeats[0].nuisance
    g = nuisance.propensity.bounded(binary.config.g_bounds)
    y = np.asarray(data.outcome, dtype=float)
    a = np.asarray(data.treatment, dtype=float)
    up1 = _binary_side(y, a, g[:, 1], nuisance.outcome.arms[1.0], 2.0, True)
    low0 = _binary_side(y, 1 - a, g[:, 0], nuisance.outcome.arms[0.0], 2.0, False)
    assert row.upper == pytest.approx(up1.value - low0.value, rel=1e-12)
    se = np.sqrt(influence_variance(up1.curve - low0.curve))
    assert row.upper_std_error == pytest.approx(se, rel=1e-12)


def test_the_breakdown_reads_the_grid(binary: Any) -> None:
    bounds = marginal_sensitivity_bounds(binary, (1.0, 1.5, 2.0, 3.0, 6.0))
    assert bounds.limit_breakdown is not None
    assert bounds.breakdown is not None
    assert bounds.limit_breakdown <= bounds.breakdown
    arm = marginal_sensitivity_bounds(binary, (1.0, 2.0), estimand="ey1")
    assert arm.null is None and arm.breakdown is None


def test_a_drtmle_fit_is_bounded() -> None:
    fit = (
        DRTMLE(
            estimands=("ate", "ey1", "ey0"),
            **linear_drtmle(n_folds=5, outcome_learner=LogisticRegression(max_iter=1000)),
        )
        .fit(binary_frame(), outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )
    bounds = marginal_sensitivity_bounds(fit, (1.0, 2.0))
    assert bounds.rows[1].lower < bounds.rows[0].lower


def test_clusters_enter_the_variance() -> None:
    frame = gaussian_frame(n=400, seed=5)
    rng = np.random.default_rng(6)
    cluster = np.repeat(np.arange(40), 10)
    shock = rng.normal(scale=2.0, size=40)[cluster]
    frame = frame.assign(cluster=cluster, Y=frame["Y"] + shock)
    clustered = gaussian_fit(frame, id="cluster")
    plain = gaussian_fit(frame)
    with_clusters = marginal_sensitivity_bounds(clustered, 1.5, estimand="ey1").rows[0]
    without = marginal_sensitivity_bounds(plain, 1.5, estimand="ey1").rows[0]
    # Mutation control: the i.i.d. variance misses the shared cluster shock.
    assert with_clusters.upper_std_error > 1.2 * without.upper_std_error
    reading = marginal_sensitivity_bounds(clustered, 1.5).limit_reading
    assert reading is not None


def test_repeats_take_the_median_rule() -> None:
    fit = binary_fit(repeats=3, simultaneous=False)
    bounds = marginal_sensitivity_bounds(fit, 2.0, estimand="ey1")
    values, variances = [], []
    data = fit.data
    y = np.asarray(data.outcome, dtype=float)
    a = np.asarray(data.treatment, dtype=float)
    for repeat in fit.repeats:
        nuisance = repeat.nuisance
        g = nuisance.propensity.bounded(fit.config.g_bounds)
        side = _binary_side(y, a, g[:, 1], nuisance.outcome.arms[1.0], 2.0, True)
        values.append(side.value)
        variances.append(influence_variance(side.curve))
    centre = float(np.median(values))
    spread = float(np.median(np.asarray(variances) + (np.asarray(values) - centre) ** 2))
    assert bounds.rows[0].upper == pytest.approx(centre, rel=1e-12)
    assert bounds.rows[0].upper_std_error == pytest.approx(np.sqrt(spread), rel=1e-12)
    assert len(set(np.round(values, 12))) > 1


def test_the_default_learners_select_by_held_out_loss(gaussian: Any) -> None:
    bounds = marginal_sensitivity_bounds(gaussian, 2.0)
    assert set(bounds.quantile_learner) <= {"linear quantile", "boosted quantile"}
    assert set(bounds.pseudo_outcome_learner) <= {"linear", "boosted"}
    assert bounds.quantile_learner and bounds.pseudo_outcome_learner


def test_invalid_arguments_raise(binary: Any) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        marginal_sensitivity_bounds(binary, 0.5)
    with pytest.raises(ValueError, match="estimand"):
        marginal_sensitivity_bounds(binary, 2.0, estimand="att")
    with pytest.raises(ValueError, match="level"):
        marginal_sensitivity_bounds(binary, 2.0, level=1.0)


# ---------------------------------------------------------------------------------------
# The limit reading


def covering_row(rule: str) -> VerdictEvidence:
    return VerdictEvidence(
        instrument=MSM_BOUND_INSTRUMENT,
        rule=rule,
        rung="L2",
        theory="test row",
        scope=CalibrationScope.unrestricted(),
        sentence="covered by the test row",
        exact=True,
    )


def test_every_limit_is_descriptive_on_the_shipped_registry(gaussian: Any, binary: Any) -> None:
    assert not [
        row for row in interpretation.VERDICT_EVIDENCE if row.instrument == MSM_BOUND_INSTRUMENT
    ]
    for fit in (gaussian, binary):
        bounds = marginal_sensitivity_bounds(fit, 1.5)
        assert bounds.limit_descriptive
        assert bounds.limit_reading is not None
        assert bounds.limit_reading.sentence.startswith(DESCRIPTIVE_SENTENCE)
        assert f"confidence limit: {DESCRIPTIVE_SENTENCE}" in bounds.summary()


def test_a_registered_row_licenses_exactly_its_rule(
    monkeypatch: pytest.MonkeyPatch, gaussian: Any, binary: Any
) -> None:
    monkeypatch.setattr(
        interpretation,
        "VERDICT_EVIDENCE",
        (*interpretation.VERDICT_EVIDENCE, covering_row("binary")),
    )
    assert not marginal_sensitivity_bounds(binary, 1.5).limit_descriptive
    assert marginal_sensitivity_bounds(gaussian, 1.5).limit_descriptive


# ---------------------------------------------------------------------------------------
# Refusals


class Raising(LinearRegression):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a refused request fitted a learner")


def refused_with(result: Any, reason: str, owner: str | None) -> None:
    refusal = marginal_bound_refusal(result)
    assert refusal is not None
    assert refusal.reason.value == reason
    assert refusal.roadmap == owner
    with pytest.raises(CapabilityError):
        marginal_sensitivity_bounds(
            result, 2.0, quantile_learner=lambda tau: Raising(), pseudo_outcome_learner=Raising()
        )


def test_a_collaborative_fit_is_refused() -> None:
    fit = (
        CTMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
            estimands=("ate", "ey1", "ey0"),
        )
        .fit(binary_frame(), outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )
    refused_with(fit, "implementation_gap", "X82")


def test_a_multi_arm_fit_is_refused() -> None:
    frame = binary_frame()
    rng = np.random.default_rng(2)
    frame = frame.assign(A=rng.integers(0, 3, size=len(frame)))
    fit = binary_fit(frame)
    refused_with(fit, "implementation_gap", "X82")


def test_a_weighted_fit_is_refused() -> None:
    frame = gaussian_frame().assign(wt=np.linspace(0.5, 1.5, 400))
    refused_with(gaussian_fit(frame, weights="wt"), "implementation_gap", "X82")


def test_a_fit_without_influence_curve_inference_is_refused() -> None:
    frame = gaussian_frame().assign(cluster=np.repeat([0, 1], 200))
    fit = gaussian_fit(frame, id="cluster", cross_fit=False)
    assert fit.inference_status != "influence_curve"
    refusal = marginal_bound_refusal(fit)
    assert refusal is not None
    refused_with(fit, "inference_gap", refusal.roadmap)


@pytest.mark.parametrize(("family", "owner"), [("longitudinal", "F16"), ("structural", None)])
def test_other_families_are_refused(family: str, owner: str) -> None:
    class Stub:
        assessment_family = family

    refusal = marginal_bound_refusal(Stub())
    assert refusal is not None and refusal.roadmap == owner


def test_the_module_names_one_owner() -> None:
    assert marginal_module._OWNER == "X87"
    assert marginal_module._PART_H2 == "X82"
