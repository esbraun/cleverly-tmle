r"""Randomized missing-outcome DR-TMLE above two arms.

Díaz and van der Laan (2017) state their construction for one arm indicator. Above two arms
the fit applies it to each indicator ``1{A = a}`` and stacks the per-arm estimators. No
fluctuation parameter is shared across arms: the outcome, observation and treatment tilts are
all per arm.

Four kinds of evidence, as ``docs/technical-reference/dr-tmle/theorem.md`` sets out:

* **exact law** (:mod:`tests.discrete_law_mar_multi`). At the oracle nuisances every
  correction vanishes, so the curve is the EIF. That check is blind to the corrections, so
  two drift laws make each block nonzero while the estimate stays exact;
* **mutation controls** on a live finite-sample fit. On the exact frame every
  ``W``-measurable covariate solves the treatment equation at the oracle mechanism, so a
  treatment-side mutation cannot fail there. A finite sample moves every tilt;
* **an independent one-indicator reference**: a test-local implementation of Steps 1-5
  (pages 18-20) on ``(W, 1{A=a}, Delta, Delta Y)`` that shares no code with the package;
* **integration**: the reports, the covariance, the band and persistence at three arms.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.estimators.targeting as targeting
from cleverly import ATE, CausalStudy, DRTMLEMethod, PointTreatment, load
from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.reduced import MissingOutcomeReducedSet
from cleverly.estimators.tmle import correction_parts
from cleverly.fluctuation.iterative import InitialFit
from cleverly.inference.influence import missing_outcome_correction_parts
from cleverly.sensitivity import missingness_tilt
from tests import discrete_law_mar_multi as law
from tests.studies.mar_arm_indexed_laws import LawOutcome, LawResponse, LawTreatment, sample

#: The module, not the ``cleverly.estimators.tmle`` attribute, which a function shadows.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
REDUCED = {
    "reduced_outcome_learner": LinearRegression,
    "reduced_treatment_learner": lambda: LogisticRegression(C=1e6, max_iter=2000),
}


def _reduced_learners() -> dict[str, Any]:
    return {name: build() for name, build in REDUCED.items()}


def _law_fit(
    frame: pd.DataFrame,
    *,
    mu: np.ndarray | None = None,
    pi: np.ndarray | None = None,
    known: bool = True,
    estimands: tuple[str, ...] = ("ey", "ate", "rr", "or"),
    max_outer: int = 40,
) -> Any:
    """A K = 3 fit with law-table primaries and the binary study's reduction learners."""
    probabilities = law.probabilities_by_label(frame) if known else None
    return (
        DRTMLE(
            randomized=True,
            cross_fit=False,
            outcome_learner=LawOutcome(law.LAW, mu=mu),
            treatment_learner=LawTreatment(law.LAW),
            missingness_learner=LawResponse(law.LAW, pi=pi),
            **_reduced_learners(),
            estimands=estimands,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            max_outer=max_outer,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=probabilities,
        )
        .single()
    )


def _parts(result: Any) -> Any:
    repeat = result.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    return correction_parts(
        result.data,
        repeat.nuisance,
        fluctuation,
        fluctuation.targeted,
        repeat.nuisance.scaler.scale(result.data.outcome),
    )


def _mean_fluctuation(result: Any) -> Any:
    return result.repeats[0].fluctuations["mean"]


# ------------------------------------------------------------------ the exact-law fits


@pytest.fixture(scope="module")
def oracle_fit() -> Any:
    return _law_fit(law.frame())


@pytest.fixture(scope="module")
def outcome_drift_fit() -> Any:
    return _law_fit(law.frame(), mu=law.WRONG_Q)


@pytest.fixture(scope="module")
def observation_drift_fit() -> Any:
    return _law_fit(law.frame(), pi=law.WRONG_PI)


@pytest.fixture(scope="module")
def both_wrong_fit() -> Any:
    return _law_fit(law.frame(), mu=law.WRONG_Q, pi=law.WRONG_PI)


class TestTheSampleRealisesTheLaw:
    def test_every_cell_is_an_integer_count(self) -> None:
        exact = law.PROBS * law.N
        np.testing.assert_array_equal(exact, np.rint(exact))
        assert int(law.COUNTS.min()) >= 2
        assert len(law.frame()) == law.N

    def test_the_empirical_conditionals_are_the_tables(self) -> None:
        frame = law.frame()
        for w in range(3):
            rows = frame["W"] == w
            for column, label in enumerate(law.LABELS):
                arm = rows & (frame["A"] == label)
                observed = arm & (frame["Delta"] == 1.0)
                assert arm.sum() / rows.sum() == law.G[w, column]
                assert observed.sum() / arm.sum() == law.PI[w, column]
                assert frame.loc[observed, "Y"].mean() == law.Q[w, column]

    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        for name in law.NAMES:
            assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_no_bound_binds(self) -> None:
        assert float((law.G * law.PI).min()) == 1 / 32
        assert float(law.G.min()) > G_BOUNDS[0]
        assert float(law.PI.min()) > NUISANCE_BOUND


class TestW1TheOracleCurveIsTheEIF:
    """Oracle nuisances on the exact frame: the estimate is the truth, the curve the EIF.

    This check is blind to the three correction blocks, because each vanishes at the truth.
    The drift classes below make each block nonzero.
    """

    def test_every_name_is_reported(self, oracle_fit) -> None:
        assert set(oracle_fit.estimates) == set(law.NAMES)

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_estimate_is_the_truth(self, oracle_fit, name: str) -> None:
        assert oracle_fit.estimates[name].psi == pytest.approx(law.natural(name), abs=1e-12)

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_curve_is_the_gateaux_derivative(self, oracle_fit, name: str) -> None:
        reported = np.asarray(oracle_fit.estimates[name].influence_curve)
        cells = law.cell_of_row()
        per_cell = np.array(
            [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.SUPPORT))]
        )
        np.testing.assert_allclose(per_cell, law.eif(name), atol=1e-12, rtol=0)

    def test_the_five_reductions_vanish(self, oracle_fit) -> None:
        reduced = _mean_fluctuation(oracle_fit).reduction.reduced
        for name in ("r_a", "r_m", "e"):
            assert float(np.max(np.abs(getattr(reduced, name)))) < 1e-12
        parts = _parts(oracle_fit)
        for arm in oracle_fit.data.arm_codes:
            for block in (parts.d_a, parts.d_m, parts.d_y):
                assert float(np.max(np.abs(block[arm]))) < 1e-12


class TestW2OutcomeDrift:
    r"""``Q -> 1 - Q`` with oracle ``g`` and ``pi``: ``D_A`` and ``D_Delta`` are live.

    The estimate stays exact. With ``g pi`` correct, the solved outcome equation
    ``P_n[1{A=a} Delta / (g_a pi_a) (Y - Q*_a)] = 0`` reduces on the exact frame to
    ``sum_w P_W(w) (Q_a(w) - Q*_a(w)) = 0``. So the plug-in ``sum_w P_W(w) Q*_a(w)`` equals the
    truth, whatever the shape of ``Q*``.
    """

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_estimate_is_the_truth(self, outcome_drift_fit, name: str) -> None:
        estimate = outcome_drift_fit.estimates[name].psi
        assert estimate == pytest.approx(law.natural(name), abs=1e-9)

    def test_the_treatment_and_observation_blocks_are_nonzero_at_every_arm(
        self, outcome_drift_fit
    ) -> None:
        parts = _parts(outcome_drift_fit)
        for arm in outcome_drift_fit.data.arm_codes:
            assert float(np.max(np.abs(parts.d_a[arm]))) > 0.05
            assert float(np.max(np.abs(parts.d_m[arm]))) > 0.05

    def test_the_stored_scores_are_the_reported_corrections(self, outcome_drift_fit) -> None:
        check = outcome_drift_fit.diagnostics.corrections()
        assert check.passed
        assert len(check.rows) == 3 * law.K
        assert check.identity_failures() == ()


class TestW3ObservationDrift:
    r"""A wrong ``pi`` with the oracle ``Q``: ``D_Y`` is live.

    With ``Q`` correct, every residual has mean zero in each ``(a, w, Delta = 1)`` cell of the
    exact frame. So ``epsilon = 0`` solves every outcome equation, and the estimate is exact.
    """

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_estimate_is_the_truth(self, observation_drift_fit, name: str) -> None:
        estimate = observation_drift_fit.estimates[name].psi
        assert estimate == pytest.approx(law.natural(name), abs=1e-12)

    def test_the_outcome_block_is_nonzero_at_every_arm(self, observation_drift_fit) -> None:
        parts = _parts(observation_drift_fit)
        for arm in observation_drift_fit.data.arm_codes:
            assert float(np.max(np.abs(parts.d_y[arm]))) > 0.05

    def test_the_identity_holds(self, observation_drift_fit) -> None:
        check = observation_drift_fit.diagnostics.corrections()
        assert check.passed
        assert check.identity_failures() == ()


class TestW4BothWrong:
    """The control for W2 and W3: with both drifts, the estimate can miss.

    The three-level ``W`` keeps the outcome tilt from saturating, so this is possible.
    """

    def test_the_outcome_and_observation_tilts_moved(self, both_wrong_fit) -> None:
        fluctuation = _mean_fluctuation(both_wrong_fit)
        initial_q = both_wrong_fit.nuisance.outcome
        for arm in both_wrong_fit.data.arm_codes:
            moved = np.abs(fluctuation.targeted.arms[arm] - initial_q.arms[arm])
            assert float(np.max(moved)) > 1e-3
        observation = np.asarray(fluctuation.reduction.observation.propensity)
        assert float(np.max(np.abs(observation - both_wrong_fit.nuisance.missingness))) > 1e-3

    def test_at_least_one_arm_mean_misses_the_truth(self, both_wrong_fit) -> None:
        misses = [
            abs(both_wrong_fit.estimates[f"ey[{label}]"].psi - law.natural(f"ey[{label}]"))
            for label in law.LABELS
        ]
        assert max(misses) > 0.01


# ------------------------------------------------------------- the live finite-sample fit


def _trial(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """A three-arm randomized trial with Gaussian ``W``, MAR outcomes and a binary ``Y``."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    code = rng.integers(0, 3, size=n)
    pi = expit(0.3 + 0.4 * code - 0.6 * w1)
    observed = rng.binomial(1, pi).astype(float)
    p = expit(-0.4 + 0.5 * code + 1.5 * w1 - 0.5 * w2)
    y = rng.binomial(1, p).astype(float)
    y[observed == 0.0] = np.nan
    labels = np.array(["low", "mid", "high"], dtype=object)[code]
    return pd.DataFrame({"W1": w1, "W2": w2, "A": labels, "Delta": observed, "Y": y})


def _live_estimator(**settings: Any) -> DRTMLE:
    return DRTMLE(
        randomized=True,
        cross_fit=False,
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LogisticRegression(max_iter=1000),
        max_outer=40,
        random_state=0,
        **settings,
    )


def _live_fit(**settings: Any) -> Any:
    return (
        _live_estimator(**settings)
        .fit(_trial(), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta")
        .single()
    )


@pytest.fixture(scope="module")
def live_fit() -> Any:
    return _live_fit()


def _rows(check: Any, equation: str) -> list[Any]:
    return [row for row in check.rows if row.equation == equation]


def _identity_broken(check: Any, equation: str) -> list[float]:
    """The arms whose stored and reported ``equation`` fail the check's identity threshold."""
    return [row.arm for row in check.identity_failures() if row.equation == equation]


class TestTheLiveFitIsTheControl:
    """The precondition every mutation below rests on: the fit moved every tilt."""

    def test_every_tilt_moved(self, live_fit) -> None:
        fluctuation = _mean_fluctuation(live_fit)
        initial_g = np.asarray(live_fit.nuisance.propensity.values)
        targeted_g = np.asarray(fluctuation.mechanism.propensity)
        assert targeted_g.shape == initial_g.shape == (live_fit.data.n, law.K)
        observation = np.asarray(fluctuation.reduction.observation.propensity)
        initial_q = live_fit.nuisance.outcome
        for column, arm in enumerate(live_fit.data.arm_codes):
            assert float(np.max(np.abs(targeted_g[:, column] - initial_g[:, column]))) > 1e-4
            assert (
                float(
                    np.max(
                        np.abs(observation[:, column] - live_fit.nuisance.missingness[:, column])
                    )
                )
                > 1e-4
            )
            moved = fluctuation.targeted.arms[arm] - initial_q.arms[arm]
            assert float(np.max(np.abs(moved))) > 1e-4

    def test_the_unmutated_identity_holds(self, live_fit) -> None:
        check = live_fit.diagnostics.corrections()
        assert check.passed
        assert len(check.rows) == 3 * law.K


def _rolled_covariate(
    reduced: MissingOutcomeReducedSet, nuisance: Any, bounds: tuple[float, float]
) -> np.ndarray:
    g_a = nuisance.bounded_propensity(bounds)
    e = np.asarray(reduced.e, dtype=float)
    return np.asarray(np.roll(e, 1, axis=1) / np.roll(g_a, 1, axis=1), dtype=float)


def _negated_treatment_block(original: Callable[..., Any]) -> Callable[..., Any]:
    """The reported curve adds arm code 0's treatment correction instead of subtracting it."""

    def mutated(*args: Any, **kwargs: Any) -> Any:
        parts = original(*args, **kwargs)
        d_a, d_g = dict(parts.d_a), dict(parts.d_g)
        d_a[0.0] = -d_a[0.0]
        d_g[0.0] = d_a[0.0] + parts.d_m[0.0]
        return replace(parts, d_a=d_a, d_g=d_g)

    return mutated


def _response_swap(original: Callable[..., Any]) -> Callable[..., Any]:
    """Column ``j`` regresses ``1(A = arms[(j + 1) % K])``; covariate and mechanism still match."""

    def mutated(
        treatment: Any, propensity: Any, covariate: Any, weights: Any, arms: Any, **kw: Any
    ):
        codes = np.asarray(treatment, dtype=float)
        swapped = codes.copy()
        for j, arm in enumerate(arms):
            swapped[codes == arms[(j + 1) % len(arms)]] = arm
        return original(swapped, propensity, covariate, weights, arms, **kw)

    return mutated


def _rolled_observation(original: Callable[..., Any]) -> Callable[..., Any]:
    def mutated(*args: Any, **kwargs: Any) -> Any:
        values = list(args)
        values[6] = np.roll(np.asarray(values[6], dtype=float), 1, axis=1)
        return original(*values, **kwargs)

    return mutated


def _rolled_r_delta(original: Callable[..., Any]) -> Callable[..., Any]:
    def mutated(treatment: Any, reduced: MissingOutcomeReducedSet, **kwargs: Any) -> Any:
        rolled = replace(reduced, r_m=np.roll(np.asarray(reduced.r_m), 1, axis=1))
        return original(treatment, rolled, **kwargs)

    return mutated


class TestMutationControls:
    """Each mutation is a monkeypatch, run on the live fit. No source file is edited."""

    def test_w5_an_arm_mechanism_swap_breaks_the_treatment_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(targeting, "_missing_treatment_covariate", _rolled_covariate)
        check = _live_fit().diagnostics.corrections()
        assert len(_identity_broken(check, "D*_A")) >= 2

    def test_w5b_a_response_swap_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            targeting,
            "solve_armwise_bounded_mechanism",
            _response_swap(targeting.solve_armwise_bounded_mechanism),
        )
        check = _live_fit().diagnostics.corrections()
        unsolved = [row.arm for row in _rows(check, "D*_A") if abs(row.reported) > check.threshold]
        assert len(unsolved) >= 2
        assert not check.passed

    def test_w7_an_observation_column_swap_breaks_the_observation_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tmle_module,
            "missing_outcome_correction_parts",
            _rolled_observation(tmle_module.missing_outcome_correction_parts),
        )
        check = _live_fit().diagnostics.corrections()
        assert len(_identity_broken(check, "D*_M")) >= 2

    def test_w7_an_outcome_drift_column_swap_breaks_the_outcome_identity(
        self, live_fit, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            targeting,
            "missing_outcome_outcome_submodel",
            _rolled_r_delta(targeting.missing_outcome_outcome_submodel),
        )
        mutated = _live_fit()
        check = mutated.diagnostics.corrections()
        assert len(_identity_broken(check, "D*_Y")) >= 2
        moved = [
            abs(mutated.estimates[name].psi - live_fit.estimates[name].psi)
            for name in live_fit.estimates
        ]
        assert max(moved) > 1e-4


#: A treatment bound that binds on the live trial's initial mechanism. The suite's other
#: K = 3 fits keep every ``g`` inside its bounds, so without this fit the column clip of ``g``
#: and the ``g_bounds`` clip of ``gamma_a`` in ``D_Y`` would have no nonzero witness.
_BINDING = (0.30, 0.99)


def _loose_bounds(original: Callable[..., Any]) -> Callable[..., Any]:
    """The reported curve ignores ``g_bounds`` above two arms."""

    def mutated(*args: Any, **kwargs: Any) -> Any:
        if np.asarray(args[5]).ndim == 2:
            kwargs = {**kwargs, "g_bounds": (1e-12, 1.0 - 1e-12)}
        return original(*args, **kwargs)

    return mutated


class TestABindingTreatmentBound:
    def test_the_bound_binds_and_every_check_passes(self) -> None:
        result = _live_fit(g_bounds=_BINDING)
        initial = np.asarray(result.nuisance.propensity.values)
        assert float(np.mean(initial < _BINDING[0])) > 0.1
        assert result.diagnostics.corrections().initial_clip_share > 0.0
        assert result.diagnostics.corrections().passed
        assert result.diagnostics.score_equations().passed

    def test_a_curve_that_ignores_the_bound_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tmle_module,
            "missing_outcome_correction_parts",
            _loose_bounds(tmle_module.missing_outcome_correction_parts),
        )
        check = _live_fit(g_bounds=_BINDING).diagnostics.corrections()
        assert check.identity_failures()

    def test_and_at_a_bound_that_does_not_bind_it_is_invisible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control on the control: the suite's ordinary bounds cannot see this mutation."""
        monkeypatch.setattr(
            tmle_module,
            "missing_outcome_correction_parts",
            _loose_bounds(tmle_module.missing_outcome_correction_parts),
        )
        assert _live_fit().diagnostics.corrections().identity_failures() == ()


def test_w8_the_three_blocks_match_an_independent_computation() -> None:
    """``d_a``, ``d_m`` and ``d_y`` per arm at K = 3, on synthetic arrays."""
    arms = (0.0, 1.0, 2.0)
    e = np.array([[0.2, -0.1, 0.3], [0.3, -0.2, -0.1], [0.4, -0.3, 0.2], [0.5, -0.4, -0.2]])
    reduced = MissingOutcomeReducedSet(
        gamma_a=np.array([[0.3, 0.4, 0.3], [0.35, 0.3, 0.35], [0.2, 0.5, 0.3], [0.4, 0.3, 0.3]]),
        gamma_m=np.array([[0.7, 0.6, 0.8], [0.5, 0.7, 0.6], [0.9, 0.4, 0.7], [0.6, 0.8, 0.5]]),
        r_a=np.array([[0.2, -0.1, 0.1], [0.1, 0.3, -0.2], [-0.2, 0.1, 0.4], [0.3, -0.3, 0.0]]),
        r_m=np.array([[-0.1, 0.2, 0.3], [0.2, -0.1, 0.1], [0.0, 0.3, -0.2], [0.1, 0.1, 0.2]]),
        e=e,
        arms=arms,
        g_bounds=(1e-6, 1 - 1e-6),
        missingness_bound=0.01,
    )
    treatment = np.array([0.0, 1.0, 2.0, 1.0])
    observed = np.array([True, True, False, True])
    y = np.array([0.1, 0.6, 0.0, 0.4])
    targeted = InitialFit(
        np.array([0.2, 0.5, 0.4, 0.3]),
        {arm: np.full(4, 0.3 + 0.1 * arm) for arm in arms},
    )
    g = np.array([[0.3, 0.45, 0.3], [0.25, 0.35, 0.45], [0.4, 0.3, 0.35], [0.2, 0.5, 0.25]])
    pi = np.array([[0.7, 0.6, 0.8], [0.5, 0.75, 0.6], [0.9, 0.4, 0.7], [0.6, 0.8, 0.55]])
    parts = missing_outcome_correction_parts(
        y,
        targeted,
        treatment,
        observed,
        reduced,
        g,
        pi,
        g_bounds=(1e-6, 1 - 1e-6),
        missingness_bound=0.01,
        guard=("Q", "g"),
    )
    assert parts.d_a is not None and parts.d_m is not None and parts.d_y is not None
    delta = observed.astype(float)
    w2 = reduced.r_a / (reduced.gamma_a * reduced.gamma_m) + reduced.r_m / reduced.gamma_m
    for j, arm in enumerate(arms):
        indicator = (treatment == arm).astype(float)
        expected_a = e[:, j] / g[:, j] * (indicator - g[:, j])
        expected_m = indicator * e[:, j] / (g[:, j] * pi[:, j]) * (delta - pi[:, j])
        expected_y = indicator * delta * w2[:, j] * (y - targeted.observed)
        np.testing.assert_allclose(parts.d_a[arm], expected_a, rtol=1e-14, atol=0)
        np.testing.assert_allclose(parts.d_m[arm], expected_m, rtol=1e-14, atol=0)
        np.testing.assert_allclose(parts.d_y[arm], expected_y, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(parts.d_g[arm], parts.d_a[arm] + parts.d_m[arm])
        assert float(np.max(np.abs(expected_a))) > 0.05
    # A K-arm mechanism is not a vector; the binary complement must not be taken.
    with pytest.raises(ValueError, match=r"must be \(4, 3\)"):
        missing_outcome_correction_parts(
            y,
            targeted,
            treatment,
            observed,
            reduced,
            g[:, 1],
            pi,
            g_bounds=(1e-6, 1 - 1e-6),
            missingness_bound=0.01,
            guard=("Q", "g"),
        )


# ------------------------------------------------------- the one-indicator reference (W9)


def _logistic_tilt(
    response: np.ndarray, offset: np.ndarray, covariates: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """A weighted logistic MLE with an offset and no intercept, by Newton steps."""
    h = np.asarray(covariates, dtype=float).reshape(len(response), -1)
    epsilon = np.zeros(h.shape[1])
    for _ in range(100):
        fitted = expit(offset + h @ epsilon)
        score = h.T @ (weights * (response - fitted))
        information = (h * (weights * fitted * (1.0 - fitted))[:, None]).T @ h
        step = np.linalg.solve(information, score)
        epsilon = epsilon + step
        if float(np.max(np.abs(step))) < 1e-14:
            break
    return epsilon


def _one_indicator(
    frame: pd.DataFrame, label: str, *, mu: np.ndarray, pi: np.ndarray, rounds: int = 60
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Steps 1-5 of Díaz and van der Laan (2017), pp. 18-20, for ``1{A = label}``.

    Written from the paper, sharing no code with the package. Step 3 fits ``epsilon_A`` by
    a logistic regression of ``A`` on ``Z_A = e / g_A`` with an offset ``logit g_A`` on all
    rows. ``g_M`` is tilted among ``A = 1``, and ``m`` among ``(A, M) = (1, 1)``.
    Returns the estimate, its corrected curve and the summed absolute tilt coefficients.
    """
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    a = (frame["A"].to_numpy() == label).astype(float)
    delta = frame["Delta"].to_numpy(dtype=float)
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    column = law.LABELS.index(label)
    q = mu[w, column].copy()
    g = law.G[w, column].copy()
    m = pi[w, column].copy()
    complete = (a == 1.0) & (delta == 1.0)
    at_arm = a == 1.0

    def bounded(g: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(g, *G_BOUNDS), np.clip(m, NUISANCE_BOUND, 1.0)

    def reductions(q: np.ndarray, g: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, ...]:
        gb, mb = bounded(g, m)
        x = q.reshape(-1, 1)
        classifier = REDUCED["reduced_treatment_learner"]
        gamma_a = classifier().fit(x, a).predict_proba(x)[:, 1]
        gamma_m = classifier().fit(x[at_arm], delta[at_arm]).predict_proba(x)[:, 1]
        r_a = LinearRegression().fit(x, (a - gb) / gb).predict(x)
        r_m = LinearRegression().fit(x[at_arm], ((delta - mb) / (gb * mb))[at_arm]).predict(x)
        joint = (g * m).reshape(-1, 1)
        e = LinearRegression().fit(joint[complete], (y - q)[complete]).predict(joint)
        return gamma_a, gamma_m, r_a, r_m, e

    def drift_weight(gamma_a: np.ndarray, gamma_m: np.ndarray, r_a: np.ndarray, r_m: np.ndarray):
        gamma_a = np.clip(gamma_a, *G_BOUNDS)
        gamma_m = np.clip(gamma_m, NUISANCE_BOUND, 1.0)
        return r_a / (gamma_a * gamma_m) + r_m / gamma_m

    state = reductions(q, g, m)
    moved = {"Y": 0.0, "Delta": 0.0, "A": 0.0, "W2": 0.0}
    for _ in range(rounds):
        gamma_a, gamma_m, r_a, r_m, e = state
        gb, mb = bounded(g, m)
        h = np.column_stack([1.0 / (gb * mb), drift_weight(gamma_a, gamma_m, r_a, r_m)])
        eps_y = _logistic_tilt(y, logit(q), h, complete.astype(float))
        q = expit(logit(q) + h @ eps_y)
        z_m = e / (gb * mb)
        eps_m = _logistic_tilt(delta, logit(m), z_m, at_arm.astype(float))
        m = expit(logit(m) + z_m * eps_m[0])
        z_a = e / gb
        eps_a = _logistic_tilt(a, logit(g), z_a, np.ones_like(a))
        g = expit(logit(g) + z_a * eps_a[0])
        moved["Y"] += abs(float(eps_y[0]))
        moved["W2"] += abs(float(eps_y[1]))
        moved["Delta"] += abs(float(eps_m[0]))
        moved["A"] += abs(float(eps_a[0]))
        state = reductions(q, g, m)
        if max(float(np.max(np.abs(eps_y))), abs(float(eps_m[0])), abs(float(eps_a[0]))) < 1e-13:
            break
    psi = float(np.mean(q))
    gamma_a, gamma_m, r_a, r_m, e = state
    gb, mb = bounded(g, m)
    d_star = a * delta / (gb * mb) * (y - q) + q - psi
    d_a = e / gb * (a - gb)
    d_m = a * e / (gb * mb) * (delta - mb)
    d_y = a * delta * drift_weight(gamma_a, gamma_m, r_a, r_m) * (y - q)
    return psi, d_star - d_a - d_m - d_y, moved


#: Two live drifts on a finite sample of the exact law. The observation drift is the
#: midpoint of ``PI`` and ``WRONG_PI``: at ``WRONG_PI`` itself and n = 600 the
#: alternation of one arm does not settle in either implementation.
_W9_DRIFTS = {
    "outcome_drift": (law.WRONG_Q, law.PI),
    "observation_drift": (law.Q, (law.PI + law.WRONG_PI) / 2.0),
}


@pytest.mark.parametrize("drift", sorted(_W9_DRIFTS))
def test_w9_each_arm_is_the_one_indicator_estimator(drift: str) -> None:
    """The K-arm fit is the stack of per-arm Theorem 2 estimators, at K >= 3 only.

    Measured on this sample (seed 11, n = 600, 40 rounds against 60): the largest estimate
    gap was 3e-9 and the largest curve gap 2.3e-6. The tolerances are about 30x and 4x those.
    """
    mu, pi = _W9_DRIFTS[drift]
    frame = sample(law.LAW, 600, 11)
    result = _law_fit(frame, mu=mu, pi=pi, known=False, estimands=("ey",))
    for label in law.LABELS:
        psi, curve, moved = _one_indicator(frame, label, mu=mu, pi=pi)
        # The preconditions: every tilt of the reference is live, so agreement is not a
        # comparison of two untargeted plug-ins.
        assert min(moved.values()) > 1e-3, moved
        estimate = result.estimates[f"ey[{label}]"]
        assert estimate.psi == pytest.approx(psi, abs=1e-7)
        np.testing.assert_allclose(estimate.influence_curve, curve, atol=1e-5, rtol=0)


def test_w6_a_sign_mutation_in_the_curve_is_caught_by_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negating arm code 0's ``D_A`` in the reported curve; W9's reference sees it.

    The correction check cannot: a solved block has mean zero, so its negation does too. A sign
    flip of the solver's covariate at K >= 3 is a different matter: it is result-neutral,
    because one logistic tilt of one column solves the same equation with ``-h`` as with ``h``,
    by the opposite coefficient. The identity check sees that flip only through the stored
    residual. So the sign that matters is the curve's, and an independent curve is its witness.
    """
    monkeypatch.setattr(
        tmle_module,
        "missing_outcome_correction_parts",
        _negated_treatment_block(tmle_module.missing_outcome_correction_parts),
    )
    mu, pi = _W9_DRIFTS["outcome_drift"]
    frame = sample(law.LAW, 600, 11)
    result = _law_fit(frame, mu=mu, pi=pi, known=False, estimands=("ey",))
    gaps = {}
    for label in law.LABELS:
        _, curve, _ = _one_indicator(frame, label, mu=mu, pi=pi)
        reported = np.asarray(result.estimates[f"ey[{label}]"].influence_curve)
        gaps[label] = float(np.max(np.abs(reported - curve)))
    mutated = law.CODES[0]
    assert gaps[mutated] > 1e-2, gaps
    assert all(gap < 1e-5 for label, gap in gaps.items() if label != mutated), gaps


# ------------------------------------------------------------- covariance and the band


@pytest.fixture(scope="module")
def banded_fit() -> Any:
    return _live_fit(simultaneous=True)


class TestW10TheJointStack:
    def test_the_covariance_is_the_sample_covariance_of_the_curves(self, banded_fit) -> None:
        names = [f"ey[{label}]" for label in law.CODES]
        curves = np.column_stack([banded_fit.estimates[name].influence_curve for name in names])
        n = banded_fit.data.n
        centred = curves - curves.mean(axis=0)
        np.testing.assert_allclose(
            banded_fit.covariance(names), centred.T @ centred / (n * (n - 1)), rtol=1e-10, atol=0
        )
        assert abs(banded_fit.covariance(names)[0, 1]) > 0.0

    def test_the_contrast_reads_the_covariance(self, banded_fit) -> None:
        """Zeroing the off-diagonal moves the ATE's standard error by more than 1%."""
        ate = banded_fit.estimates["ate[low vs high]"]
        covariance = banded_fit.covariance(["ey[low]", "ey[high]"])
        joint = np.sqrt(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
        independent = np.sqrt(covariance[0, 0] + covariance[1, 1])
        assert ate.std_error == pytest.approx(joint, rel=1e-10)
        assert abs(independent / joint - 1.0) > 0.01

    def test_the_band_covers_every_name_and_reads_the_log_scale(self, banded_fit) -> None:
        bands = banded_fit.simultaneous
        assert bands is not None
        assert set(bands.bands) == set(law.NAMES)
        assert bands.critical_value > bands.pointwise_critical_value
        for name in law.NAMES:
            estimate = banded_fit.estimates[name]
            low, high = bands.bands[name]
            if name.startswith(("rr[", "or[")):
                assert estimate.log_psi is not None
                centre = estimate.log_psi
                half = bands.critical_value * estimate.std_error
                assert (low, high) == pytest.approx(
                    (np.exp(centre - half), np.exp(centre + half)), rel=1e-12
                )
            else:
                half = bands.critical_value * estimate.std_error
                assert (low, high) == pytest.approx(
                    (estimate.psi - half, estimate.psi + half), rel=1e-12
                )

    def test_the_status_supplies_inference(self, banded_fit) -> None:
        assert banded_fit.inference_status == "influence_curve"
        for name in law.NAMES:
            estimate = banded_fit.estimates[name]
            assert np.isfinite(estimate.std_error)
            assert np.isfinite(estimate.pvalue)
            low, high = estimate.ci
            assert low < high


# ---------------------------------------------------------------------- integration (W11)


class TestW11Integration:
    def test_every_equation_is_solved(self, live_fit) -> None:
        assert live_fit.diagnostics.score_equations().passed
        fluctuation = _mean_fluctuation(live_fit)
        assert fluctuation.mechanism.score.shape == (law.K,)
        assert fluctuation.reduction.observation.score.shape == (law.K,)
        assert fluctuation.reduction.score.shape == (law.K,)

    def test_the_fit_survives_serialization(self, live_fit, tmp_path) -> None:
        path = tmp_path / "multi-arm-missing.cleverly"
        live_fit.save(path)
        restored = load(path)
        for name, estimate in live_fit.estimates.items():
            assert restored.estimates[name].psi == estimate.psi
        assert restored.diagnostics.score_equations() == live_fit.diagnostics.score_equations()
        assert restored.diagnostics.corrections().rows == live_fit.diagnostics.corrections().rows

    def test_the_truncation_curve_at_the_fitted_bound_reproduces_the_fit(self, live_fit) -> None:
        curve = live_fit.diagnostics.truncation_curve(bounds=[live_fit.config.g_bounds[0]])
        np.testing.assert_allclose(curve["delta_from_fitted"], 0.0, atol=1e-12)
        assert set(curve["estimand"]) == set(live_fit.estimates)

    def test_the_positivity_report_has_a_row_per_arm_and_the_joint_row(self, live_fit) -> None:
        report = live_fit.diagnostics.support()
        assert "P(A=a,Delta=1|W)" in report.mechanisms
        summary = report.summary()
        for label in law.LABELS:
            assert f"g[{label}]" in summary

    def test_the_tilt_at_zero_reproduces_the_fit(self, live_fit) -> None:
        frame = missingness_tilt(live_fit, gamma=[0.0])
        assert len(frame) > 0
        for _, row in frame.iterrows():
            assert row["psi"] == pytest.approx(live_fit.estimates[row["estimand"]].psi, abs=1e-12)
        assert {f"ey[{label}]" for label in law.LABELS} <= set(frame["estimand"])


def test_w12_propensity_accepts_the_off_simplex_armwise_state() -> None:
    """The K-arm targeted state is off the simplex by design; ``Propensity`` must allow it.

    If a later ``__post_init__`` tightens the K-arm row-sum check, this test names the break.
    """
    values = np.array([[0.5, 0.4, 0.3], [0.2, 0.2, 0.2]])
    propensity = Propensity(values, (0.0, 1.0, 2.0))
    truncated = propensity.truncate((0.25, 0.75))
    np.testing.assert_array_equal(truncated.values, np.clip(values, 0.25, 0.75))


# --------------------------------------------------------------- the widened surface


def test_known_probabilities_as_an_array_match_the_mapping() -> None:
    frame = law.frame()
    keyed = _law_fit(frame, estimands=("ey",))
    mapping = law.probabilities_by_label(frame)
    array = np.column_stack([mapping[label] for label in law.CODES])
    positional = (
        DRTMLE(
            randomized=True,
            cross_fit=False,
            outcome_learner=LawOutcome(law.LAW),
            treatment_learner=LawTreatment(law.LAW),
            missingness_learner=LawResponse(law.LAW),
            **_reduced_learners(),
            estimands=("ey",),
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            max_outer=40,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=array,
        )
        .single()
    )
    np.testing.assert_array_equal(
        positional.nuisance.propensity.values, keyed.nuisance.propensity.values
    )
    for name, estimate in keyed.estimates.items():
        assert positional.estimates[name].psi == estimate.psi


def test_an_unguarded_fit_with_known_probabilities_is_the_plain_tmle() -> None:
    """``guard=()`` at K = 3 with the design mechanism is the ordinary TMLE, bit for bit."""
    frame = law.frame()
    unguarded = (
        DRTMLE(
            guard=(),
            cross_fit=False,
            outcome_learner=LawOutcome(law.LAW, mu=law.WRONG_Q),
            treatment_learner=LawTreatment(law.LAW),
            missingness_learner=LawResponse(law.LAW),
            estimands=("ey", "ate"),
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=law.probabilities_by_label(frame),
        )
        .single()
    )
    plain = (
        TMLE(
            cross_fit=False,
            outcome_learner=LawOutcome(law.LAW, mu=law.WRONG_Q),
            treatment_learner=LawTreatment(law.LAW),
            missingness_learner=LawResponse(law.LAW),
            estimands=("ey", "ate"),
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )
    assert unguarded.extra["drtmle"].guard == ()
    assert set(unguarded.estimates) == set(plain.estimates)
    for name, estimate in plain.estimates.items():
        assert unguarded.estimates[name].psi == estimate.psi
        np.testing.assert_array_equal(
            unguarded.estimates[name].influence_curve, estimate.influence_curve
        )


def test_the_study_route_fits_three_arms() -> None:
    """``DRTMLEMethod(randomized=True)`` through ``CausalStudy`` at three arms."""
    frame = _trial(n=300, seed=3)
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), missingness="Delta"
        ),
    ).identify(ATE())
    fitted = effect.estimate(
        method=DRTMLEMethod(
            randomized=True,
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
        ),
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    assert fitted.extra["drtmle"].reduction == "missing_outcome"
    assert {"ate[low vs high]", "ate[mid vs high]"} <= set(fitted.estimates)
    assert fitted.diagnostics.score_equations().passed
