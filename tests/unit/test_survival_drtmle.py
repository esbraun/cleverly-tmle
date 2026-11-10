"""The drift-corrected survival TMLE (X91 part F): the checks of the design, section 9.

``SurvivalDRTMLE`` follows Díaz (2019), Section 5: univariate regressions of residuals on
scalar indices of the initial fits, the clever covariates of eq. (9), three logistic
submodels per arm and horizon, and the corrected curve of Theorem 2.

The exact-law checks run on finite laws from :mod:`tests.survival_drift_law` with population
regressions by enumeration, so no smoother enters them.  The drift identity (check 5) is
the content: it holds to second order only with the correct signs and indices, and four
mutations each turn it first order.
"""

from __future__ import annotations

import functools
import pickle
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import LTMLE, DriftRecord, LongitudinalData, SurvivalDRTMLE, drift
from cleverly.longitudinal.hazard import HazardState, hazard_inputs, influence
from tests.survival_drift_law import FiniteLaw, expand, population_regression

ARMS = {"a1": 1, "a0": 0}
GB = (1e-12, 1.0)
K = 4
TAU = 4

# ---------------------------------------------------------------------- sampled data


def sample_frame(
    n: int,
    seed: int,
    *,
    n_times: int = 5,
    known: bool = False,
    sparse: bool = False,
) -> pd.DataFrame:
    """A time-to-event frame on the grid ``1..K`` with W-dependent censoring from node 2.

    With ``known=True`` the treatment is a fair coin and the frame carries the true
    treatment and retention probabilities.
    """
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    if sparse:
        w1 = np.where(rng.random(n) < 0.02, w1 + 6.0, w1)
    p1 = np.full(n, 0.5) if known else expit(0.6 * w1 + 0.3 * w2)
    a = rng.binomial(1, p1)
    retention = 1.0 - expit(-2.5 + 0.4 * w1 + 0.6 * w2)
    event_time = np.full(n, n_times + 1)
    censor_time = np.full(n, n_times + 1)
    for t in range(1, n_times + 1):
        alive = (event_time > n_times) & (censor_time > n_times)
        hazard = expit(-2.0 + 0.5 * a - w1 + 0.8 * w2 + 0.1 * t)
        fired = alive & (rng.random(n) < hazard)
        event_time = np.where(fired, t, event_time)
        # Censored after the event check at t (grid time t), for t < K.
        if t < n_times:
            alive = alive & ~fired
            leaves = alive & (rng.random(n) > retention)
            censor_time = np.where(leaves, t, censor_time)
    time = np.minimum(np.minimum(event_time, censor_time), n_times).astype(float)
    event = ((event_time <= censor_time) & (event_time <= n_times)).astype(int)
    frame = pd.DataFrame({"time": time, "event": event, "A": a, "W1": w1, "W2": w2})
    if known:
        frame["p1"] = 0.5
        frame["p0"] = 0.5
        for g in range(1, n_times):
            frame[f"r{g}"] = retention
    return frame


def container(frame: pd.DataFrame, **kwargs: Any) -> LongitudinalData:
    grid = [float(g) for g in range(1, int(frame["time"].max()) + 1)]
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=grid,
        **kwargs,
    )


def glm() -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=1000)


def fit(data: LongitudinalData, **kwargs: Any) -> Any:
    settings: dict[str, Any] = {
        "reference": "a0",
        "outcome_learner": glm(),
        "treatment_learner": glm(),
        "censoring_learner": glm(),
        "n_folds": 2,
        "random_state": 0,
        "horizons": [3.0],
    }
    settings.update(kwargs)
    return SurvivalDRTMLE(ARMS, **settings).fit(data)


@pytest.fixture(scope="module")
def sampled() -> LongitudinalData:
    return container(sample_frame(1500, 11))


@pytest.fixture(scope="module")
def drift_result(sampled: LongitudinalData) -> Any:
    return fit(sampled)


# ----------------------------------------------------------------------- finite laws

LEVELS = np.linspace(-1.0, 1.0, 20)
DIFFS = np.linspace(-1.0, 1.0, 10)
#: Law W5: 20 levels of s = W1 + 0.5 W2, each holding 10 points of distinct W2 - W1.
S = np.repeat(LEVELS, 10)
D = np.tile(DIFFS, 20)
P = np.full(200, 1 / 200)


def _psi(p: np.ndarray) -> np.ndarray:
    """A direction that varies inside each level set of s."""
    return D[p] ** 2 - 0.3 * D[p]


def _phi(p: np.ndarray) -> np.ndarray:
    """A direction that is a function of s."""
    return np.sin(2.0 * S[p]) + 0.5


def _zero_first(
    rate: Callable[[int, np.ndarray, int], np.ndarray],
) -> Callable[[int, np.ndarray, int], np.ndarray]:
    """``g_R(0) = 0``: on a ``TimeToEvent`` container node 1 is never censored."""

    def g(m: int, p: np.ndarray, a: int) -> np.ndarray:
        return np.zeros(p.size) if m == 0 else rate(m, p, a)

    return g


def law_view(
    law: FiniteLaw,
    g_a: Callable[[np.ndarray], np.ndarray],
    g_r: Callable[[int, np.ndarray, int], np.ndarray],
    h: Callable[[int, np.ndarray, int], np.ndarray],
    *,
    arm: int = 1,
    held_treatment: bool = False,
    held_censoring: tuple[bool, ...] = (True, False, False, False),
) -> drift.PaperView:
    points = law.point
    p_arm = g_a(points) if arm == 1 else 1.0 - g_a(points)
    return drift.PaperView(
        arm=(law.treatment == arm).astype(float),
        g_a=p_arm,
        g_r=np.column_stack([g_r(m, points, arm) for m in range(K)]),
        h=np.column_stack([h(t, points, arm) for t in range(1, K + 1)]),
        at_risk=law.at_risk,
        events=law.events,
        eligible=law.eligible,
        censored=law.censored,
        held_treatment=held_treatment,
        held_censoring=np.array(held_censoring),
        held_hazard=np.zeros(K, dtype=bool),
    )


def population_pieces(
    law: FiniteLaw,
    view: drift.PaperView,
    theta0: float,
    *,
    state: drift._State | None = None,
) -> dict[str, Any]:
    """Reductions at the view's nuisances by enumeration, eq. (9) and the scores at ``state``.

    Returns the remainder :math:`\\beta = \\Psi(\\eta) - \\theta_0 + P_0 D(\\eta)` and
    :math:`P_0 D_A`, :math:`P_0 D_R`, :math:`P_0 D_L`, all at ``state`` (default: the view).
    """
    regress = population_regression(law.weights)
    initial = drift._State(view.g_a, view.g_r, view.h)
    current = initial if state is None else state
    reductions = drift._Reductions(view, initial, TAU, GB, regress)
    coefficients = drift.fit_coefficients(reductions, TAU, GB[0])
    covariates = drift.clever_covariates(current, coefficients, TAU, GB)
    curve = drift.corrected_curve(view, current, covariates, TAU, GB)
    w = law.weights
    s_tau = current.survival(TAU)[:, TAU]
    augmentation = -curve.eif + (1.0 - curve.risk) - s_tau
    beta = float((w * s_tau).sum() - theta0 + (w * augmentation).sum())
    return {
        "beta": beta,
        "d_a": float((w * curve.d_a).sum()),
        "d_r": float((w * curve.d_r).sum()),
        "d_l": float((w * curve.d_l).sum()),
        "covariates": covariates,
        "curve": curve,
        "weights": w,
    }


def theta_of(h: Callable[[int, np.ndarray, int], np.ndarray], arm: int = 1) -> float:
    points = np.arange(P.size)
    survival = np.prod([1.0 - h(t, points, arm) for t in range(1, TAU + 1)], axis=0)
    return float((P * survival).sum())


# h-wrong branch: g through s, the truth's g moves off s by delta.
def _f_a(p: np.ndarray) -> np.ndarray:
    return 0.6 * S[p]


def _f_r(m: int, p: np.ndarray, a: int) -> np.ndarray:
    return -2.0 - 0.5 * S[p] + 0.1 * m + 0.3 * a


def _h0(t: int, p: np.ndarray, a: int) -> np.ndarray:
    return expit(-1.5 + 0.5 * a + D[p] + 0.1 * t + 0.3 * S[p])


def _h1(t: int, p: np.ndarray, a: int) -> np.ndarray:
    return expit(-1.0 + 0.5 * a + 0.1 * t + 0.3 * S[p])


G_A_HAT = lambda p: expit(_f_a(p))  # noqa: E731
G_R_HAT = _zero_first(lambda m, p, a: expit(_f_r(m, p, a)))


def h_wrong_remainder(delta: float, direction: Callable[[np.ndarray], np.ndarray]) -> float:
    """:math:`\\beta - P_0(D_A + D_R)` with :math:`h` wrong and the truth's :math:`g` moved."""
    g_a0 = lambda p: expit(_f_a(p) + delta * direction(p))  # noqa: E731
    g_r0 = _zero_first(lambda m, p, a: expit(_f_r(m, p, a) + delta * direction(p)))
    law = expand(P, g_a0, g_r0, _h0, K)
    pieces = population_pieces(law, law_view(law, G_A_HAT, G_R_HAT, _h1), theta_of(_h0))
    return pieces["beta"] - pieces["d_a"] - pieces["d_r"]


# g-wrong branch (the mirror): h through s, the truth's h moves off s by delta.
G_A0_M = lambda p: expit(0.3 + 0.8 * D[p])  # noqa: E731
G_R0_M = _zero_first(lambda m, p, a: expit(-2.0 + 0.7 * D[p] + 0.1 * m))
G_A1_M = lambda p: np.full(p.size, 0.55)  # noqa: E731
G_R1_M = _zero_first(lambda m, p, a: np.full(p.size, 0.12))


def _f_h(t: int, p: np.ndarray, a: int) -> np.ndarray:
    return -1.5 + 0.5 * a + 0.1 * t + 0.8 * S[p]


def g_wrong_remainder(delta: float, direction: Callable[[np.ndarray], np.ndarray]) -> float:
    """:math:`\\beta - P_0 D_L` with :math:`g` wrong and the truth's :math:`h` moved."""
    h0 = lambda t, p, a: expit(_f_h(t, p, a) + delta * direction(p))  # noqa: E731
    h_hat = lambda t, p, a: expit(_f_h(t, p, a))  # noqa: E731
    law = expand(P, G_A0_M, G_R0_M, h0, K)
    pieces = population_pieces(law, law_view(law, G_A1_M, G_R1_M, h_hat), theta_of(h0))
    return pieces["beta"] - pieces["d_l"]


# A law whose S(tau) and M = sum_t S(t) carry different level sets: S(tau) a function of s,
# M a function of W2 - W1.  S(1) = S(2) = S(3) = x and S(4) = F, so M = 3x + F.
def _split_f(p: np.ndarray, a: int, delta: float) -> np.ndarray:
    return 0.5 * expit(-0.5 + 0.8 * S[p] + 0.3 * a) * (1.0 + 0.3 * delta * _phi(p))


def _split_hazard(delta: float) -> Callable[[int, np.ndarray, int], np.ndarray]:
    def h(t: int, p: np.ndarray, a: int) -> np.ndarray:
        f = _split_f(p, a, delta)
        x = (2.4 + 0.25 * np.tanh(D[p]) - f) / 3.0
        if t == 1:
            return 1.0 - x
        if t < TAU:
            return np.zeros(p.size)
        return 1.0 - f / x

    return h


def split_remainder(delta: float) -> float:
    """:math:`\\beta - P_0 D_L` on the split law, :math:`g` wrong, the truth's :math:`S(\\tau)`
    moved along a function of s."""
    h0 = _split_hazard(delta)
    g_a0 = lambda p: expit(0.3 + S[p])  # noqa: E731
    g_r0 = _zero_first(lambda m, p, a: expit(-2.0 + 0.7 * S[p] + 0.1 * m))
    law = expand(P, g_a0, g_r0, h0, K)
    view = law_view(law, G_A1_M, G_R1_M, _split_hazard(0.0))
    pieces = population_pieces(law, view, theta_of(h0))
    return pieces["beta"] - pieces["d_l"]


def ratios(remainder: Callable[[float], float]) -> np.ndarray:
    values = [remainder(d) for d in (0.2, 0.1, 0.05, 0.025)]
    return np.array([values[i] / values[i + 1] for i in range(3)])


# ------------------------------------------------------------- check 1: the reduction


def test_guard_empty_is_hazard_ltmle_at_rtol_zero(sampled: LongitudinalData) -> None:
    ours = fit(sampled, guard=())
    reference = LTMLE(
        ARMS,
        reference="a0",
        survival_model="hazard",
        simultaneous=False,
        outcome_learner=glm(),
        treatment_learner=glm(),
        censoring_learner=glm(),
        n_folds=2,
        random_state=0,
        horizons=[3.0],
        max_iter=200,
    ).fit(sampled)
    assert ours.drift is None
    for name in ours.estimates:
        np.testing.assert_array_equal(ours.estimates[name].psi, reference.estimates[name].psi)
        np.testing.assert_array_equal(
            ours.estimates[name].influence_curve, reference.estimates[name].influence_curve
        )


def test_loop_without_drift_covariates_agrees_with_hazard_targeting(
    sampled: LongitudinalData, drift_result: Any
) -> None:
    """With H_A, H_R, H_L forced to zero only Z moves the hazards: the LTMLE score root."""
    reference = fit(sampled, guard=())
    block = _views(sampled)
    for label, code in ARMS.items():
        view = block[code]
        out = drift.target_drift(
            view,
            3,
            label=label,
            g_bounds=(0.01, 1.0),
            alpha=0.9995,
            max_iter=200,
            n_folds=2,
            drift=False,
        )
        assert out.record.epsilon[:, :3].tolist() == [[0.0, 0.0, 0.0]] * out.record.n_iter
        name = f"risk_regimen[{label} @ t=3]"
        assert out.curve.risk == pytest.approx(reference.estimates[name].psi, abs=10.0 / sampled.n)


def _views(data: LongitudinalData) -> dict[int, drift.PaperView]:
    """The paper views of a fit's own out-of-fold nuisances, by arm code."""
    estimator = SurvivalDRTMLE(
        ARMS,
        reference="a0",
        outcome_learner=glm(),
        treatment_learner=glm(),
        censoring_learner=glm(),
        n_folds=2,
        random_state=0,
        horizons=[3.0],
    )
    captured: dict[int, drift.PaperView] = {}
    original = drift.run_drift

    def spy(prepared: Any, plans: Any, mechanism: Any, initial: Any, **kwargs: Any) -> Any:
        for code in (0, 1):
            captured[code] = drift.paper_view(
                prepared,
                mechanism,
                initial,
                code,
                alpha=kwargs["alpha"],
                g_bounds=kwargs["g_bounds"],
            )
        return original(prepared, plans, mechanism, initial, **kwargs)

    drift.run_drift = spy
    try:
        estimator.fit(data)
    finally:
        drift.run_drift = original
    return captured


# ------------------------------------------------------------- check 2: the order map


def test_paper_view_relabels_the_container(sampled: LongitudinalData) -> None:
    from cleverly.longitudinal.collaborative import eligible_censoring

    views = _views(sampled)
    eligible, censored = eligible_censoring(sampled)
    for view in views.values():
        np.testing.assert_array_equal(view.eligible, eligible)
        np.testing.assert_array_equal(view.censored, censored.astype(float))
        # g_R(0) is the container's node 1, never censored on a TimeToEvent container.
        assert view.held_censoring[0]
        np.testing.assert_array_equal(view.g_r[:, 0], 0.0)
        assert not view.held_censoring[1:].any()
        # I_t is the container's at-risk set; the J_m rows contain the I_{m+1} rows.
        assert np.all(view.at_risk[:, 1:] <= view.eligible[:, 1:])


def test_retention_index_is_the_container_cumulative(sampled: LongitudinalData) -> None:
    """G(t) is the container's G_t with no shift: g_A G(t) is the cumulative at node t."""
    views = _views(sampled)
    view = views[1]
    state = drift._State(view.g_a, view.g_r, view.h)
    pi = state.pi(4)
    # The cumulative mechanism product of node t is g_A prod_{s <= t} c_s.
    expected = view.g_a[:, None] * np.cumprod(1.0 - view.g_r[:, :4], axis=1)
    np.testing.assert_allclose(pi[:, 1:], expected, rtol=1e-12)
    shifted = view.g_a[:, None] * np.concatenate(
        [np.ones((view.n, 1)), np.cumprod(1.0 - view.g_r[:, :3], axis=1)], axis=1
    )
    assert np.max(np.abs(pi[:, 1:] - shifted)) > 1e-3


# -------------------------------------------------------------- check 3: the exact law


def test_exact_law_has_no_drift() -> None:
    law = expand(P, G_A_HAT, G_R_HAT, _h0, K)
    pieces = population_pieces(law, law_view(law, G_A_HAT, G_R_HAT, _h0), theta_of(_h0))
    cov = pieces["covariates"]
    for array in (cov.h_a, cov.h_r, cov.h_l):
        assert np.max(np.abs(array)) < 1e-12
    assert abs(pieces["beta"]) < 1e-12
    # The Z score, the EIF's, is zero at the truth: the first epsilon_L2 is zero.
    w = pieces["weights"]
    assert abs(float((w * pieces["curve"].eif).sum())) < 1e-12


# --------------------------------------------- check 4: the vanishing pattern (Thm 2)


def _norm(law: FiniteLaw, values: np.ndarray) -> float:
    squared = values**2 if values.ndim == 1 else (values**2).sum(axis=1)
    return float(np.sqrt((law.weights * squared).sum()))


def test_exact_g_wrong_h_zeroes_h_l_only() -> None:
    law = expand(P, G_A_HAT, G_R_HAT, _h0, K)
    view = law_view(law, G_A_HAT, G_R_HAT, _h1)
    pieces = population_pieces(law, view, theta_of(_h0))
    cov = pieces["covariates"]
    assert np.max(np.abs(cov.h_l)) < 1e-12
    # The printed Theorem 2 would zero D_A and D_R here; they are live.
    assert _norm(law, cov.h_a) > 1e-2
    assert _norm(law, cov.h_r) > 1e-2
    # At a perturbed g the scores move off zero.
    moved = drift._State(expit(logit(view.g_a) + 0.3), view.g_r, view.h)
    perturbed = population_pieces(law, view, theta_of(_h0), state=moved)
    assert abs(perturbed["d_a"]) > 1e-3
    moved = drift._State(
        view.g_a,
        np.where(view.g_r > 0, expit(logit(np.clip(view.g_r, 1e-9, 1)) + 0.3), 0.0),
        view.h,
    )
    perturbed = population_pieces(law, view, theta_of(_h0), state=moved)
    assert abs(perturbed["d_r"]) > 1e-3


def test_exact_h_wrong_g_zeroes_h_a_and_h_r() -> None:
    h0 = lambda t, p, a: expit(_f_h(t, p, a))  # noqa: E731
    law = expand(P, G_A0_M, G_R0_M, h0, K)
    view = law_view(law, G_A1_M, G_R1_M, h0)
    pieces = population_pieces(law, view, theta_of(h0))
    cov = pieces["covariates"]
    assert np.max(np.abs(cov.h_a)) < 1e-12
    assert np.max(np.abs(cov.h_r)) < 1e-12
    assert _norm(law, cov.h_l) > 1e-2
    moved = drift._State(view.g_a, view.g_r, expit(logit(view.h) + 0.3))
    perturbed = population_pieces(law, view, theta_of(h0), state=moved)
    assert abs(perturbed["d_l"]) > 1e-3


# ------------------------------------------ check 5: the drift identity and its witness


def test_drift_identity_is_second_order_when_level_sets_are_kept() -> None:
    assert np.all(ratios(lambda d: h_wrong_remainder(d, _phi))[1:] > 3.5)
    assert np.all(ratios(lambda d: g_wrong_remainder(d, _phi))[1:] > 3.5)
    # e_A on S(tau): the split law separates it from M.
    assert np.all(ratios(split_remainder) > 3.5)


def test_drift_identity_is_first_order_when_level_sets_break() -> None:
    """The nonzero witness: the regression is on the estimated index, which ties points whose
    true mechanism (or hazard) differs.  The remainder is then first order."""
    for remainder in (
        lambda d: h_wrong_remainder(d, _psi),
        lambda d: g_wrong_remainder(d, _psi),
    ):
        values = ratios(remainder)
        assert np.all((values[1:] > 1.7) & (values[1:] < 2.3)), values
        assert abs(remainder(0.1)) > 1e-4


@pytest.mark.parametrize(
    ("mutation", "branch"),
    [
        ("d_r_sign", "h"),
        ("u_for_v", "h"),
        ("e_r_sign", "g"),
        ("a_term_index", "g"),
        ("e_a_on_m", "split"),
    ],
)
def test_drift_identity_mutations_turn_first_order(
    mutation: str, branch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mutation == "d_r_sign":
        original = drift.corrected_curve

        def flipped(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            return drift.DriftCurve(out.risk, out.eif, out.d_l, -out.d_r, out.d_a, out.curve)

        monkeypatch.setattr(drift, "corrected_curve", flipped)
        # population_pieces calls the module attribute.
    elif mutation == "u_for_v":
        monkeypatch.setattr(drift._Reductions, "v", lambda self, t, m: self.u(t, m))
    elif mutation == "e_r_sign":
        original_e_r = drift._Reductions.e_r
        monkeypatch.setattr(drift._Reductions, "e_r", lambda self, k: -original_e_r(self, k))
    elif mutation == "a_term_index":
        monkeypatch.setattr(drift, "_a_term_index", lambda t: t)
    else:

        def e_a_on_m(self: Any) -> np.ndarray:
            view = self.view
            index = self.initial.survival(TAU)[:, 1:].sum(axis=1)
            outcome = (view.arm - self.initial.g_a) / self.initial.g_a
            return self.fit("e_A", -1, -1, index, outcome, np.ones(view.n, dtype=bool))

        monkeypatch.setattr(drift._Reductions, "e_a", e_a_on_m)
    if branch == "split":
        values = ratios(split_remainder)
    else:
        remainder = h_wrong_remainder if branch == "h" else g_wrong_remainder
        values = ratios(lambda d: remainder(d, _phi))
    assert values[-1] < 2.6, values


# ------------------------------------------------------- check 7: scores and stopping


def test_scores_are_solved_at_the_stop(drift_result: Any) -> None:
    for (label, horizon), record in drift_result.drift.items():
        name = f"risk_regimen[{label} @ t={horizon:g}]"
        sigma = float(np.std(drift_result.estimates[name].influence_curve))
        bound = sigma / (10.0 * np.sqrt(drift_result.n))
        assert record.converged
        assert all(abs(score) < bound for score in record.scores), record.scores
        # The stop is the first iteration below the threshold.
        steps = np.max(np.abs(record.epsilon), axis=1)
        assert steps[-1] < record.threshold
        assert np.all(steps[:-1] >= record.threshold)
        assert record.threshold == pytest.approx(1e-4 * drift_result.n ** (-0.6))


def test_reductions_are_fitted_once(sampled: LongitudinalData) -> None:
    calls: list[int] = []
    original = drift.smooth

    def counting(*args: Any) -> Any:
        calls.append(1)
        return original(*args)

    drift.smooth = counting  # type: ignore[assignment]
    try:
        result = fit(sampled)
    finally:
        drift.smooth = original  # type: ignore[assignment]
    expected = sum(
        sum(fit.bandwidth is not None or fit.degenerate for fit in record.reductions)
        for record in result.drift.values()
    )
    assert len(calls) == expected
    assert all(record.n_iter > 1 for record in result.drift.values())


# ------------------------------------------------------------------- check 8: curve


def test_curve_is_the_rowwise_formula(sampled: LongitudinalData) -> None:
    views = _views(sampled)
    result = fit(sampled)
    for label, code in ARMS.items():
        out = drift.target_drift(
            views[code], 3, label=label, g_bounds=(0.01, 1.0), alpha=0.9995, max_iter=200, n_folds=2
        )
        name = f"risk_regimen[{label} @ t=3]"
        np.testing.assert_allclose(
            result.estimates[name].influence_curve, out.curve.curve, rtol=1e-10, atol=1e-12
        )
        expected = out.curve.eif + out.curve.d_l + out.curve.d_r + out.curve.d_a
        np.testing.assert_allclose(out.curve.curve, expected, rtol=1e-12, atol=1e-14)
    difference = result.estimates["ate_regimen[a1 vs a0 @ t=3]"]
    np.testing.assert_allclose(
        difference.influence_curve,
        result.estimates["risk_regimen[a1 @ t=3]"].influence_curve
        - result.estimates["risk_regimen[a0 @ t=3]"].influence_curve,
        rtol=1e-12,
        atol=1e-14,
    )


def test_eif_part_equals_the_hazard_influence(sampled: LongitudinalData) -> None:
    """The EIF of the risk at the targeted state equals hazard LTMLE's influence function."""
    views = _views(sampled)
    view = views[1]
    out = drift.target_drift(
        view, 3, label="a1", g_bounds=(0.01, 1.0), alpha=0.9995, max_iter=200, n_folds=2
    )
    state = out.state
    pi = np.clip(state.pi(sampled.n_times)[:, 1:], 0.01, 1.0)
    levels = [pi, pi]
    from cleverly.longitudinal.drift import _level_plan

    inputs = hazard_inputs(sampled, (_level_plan(sampled, 1),), levels, alpha=0.9995)
    hazards = np.zeros((sampled.n, sampled.n_times, 2, 1))
    hazards[:, :, 1, 0] = state.h
    hazards[:, :, 0, 0] = state.h
    targeted = HazardState.from_hazards(hazards, inputs.bounds, inputs.empty, inputs.full)
    _, curves = influence(targeted, inputs, (3,))
    np.testing.assert_allclose(curves[0, 0, 0], out.curve.eif, rtol=1e-8, atol=1e-10)


# -------------------------------------------------------- check 9: declared factors


def test_declared_treatment_and_censoring_are_held() -> None:
    frame = sample_frame(1200, 5, known=True)
    data = container(
        frame,
        treatment_probabilities={1: "p1", 0: "p0"},
        censoring_probabilities={float(g): f"r{g}" for g in range(1, 4)},
    )
    result = fit(data, outcome_learner=LogisticRegression(C=1e6, max_iter=1000))
    for record in result.drift.values():
        assert "treatment" in record.held_factors
        assert record.curve_terms == ("D_L",)
        assert record.censoring_terms == ()
        np.testing.assert_array_equal(record.epsilon[:, :2], 0.0)


def test_partial_censoring_declaration_sums_estimated_nodes_only() -> None:
    frame = sample_frame(1200, 6, known=True)
    data = container(
        frame,
        treatment_probabilities={1: "p1", 0: "p0"},
        censoring_probabilities={1.0: "r1"},
    )
    result = fit(data)
    record = result.drift[("a1", 3)]
    assert record.censoring_terms == (3,)
    assert record.curve_terms == ("D_L", "D_R")
    assert "censoring node 2" in record.held_factors


def test_declared_factor_in_view_is_untouched(sampled: LongitudinalData) -> None:
    views = _views(sampled)
    view = views[1]
    held = drift.PaperView(**{**view.__dict__, "held_treatment": True})
    out = drift.target_drift(
        held, 3, label="a1", g_bounds=(0.01, 1.0), alpha=0.9995, max_iter=200, n_folds=2
    )
    np.testing.assert_array_equal(out.state.g_a, view.g_a)
    assert "D_A" not in out.record.curve_terms
    np.testing.assert_array_equal(out.curve.d_a, 0.0)


def test_censoring_probabilities_map_grid_time_to_next_node() -> None:
    frame = sample_frame(400, 7, known=True)
    data = container(frame, censoring_probabilities={2.0: "r2"})
    known = data.known_mechanisms
    assert known is not None
    assert known.censoring_columns[2] == "r2"
    assert known.censoring[1] is None and known.censoring[3] is None
    with pytest.raises(Exception, match="not a grid time"):
        container(frame, censoring_probabilities={5.0: "r2"})


# -------------------------------------------------------------- check 10: denominators


def test_floors_keep_the_covariates_finite() -> None:
    data = container(sample_frame(1500, 8, sparse=True))
    result = fit(data, g_bounds=(0.01, 1.0))
    for record in result.drift.values():
        assert record.converged
    for estimate in result.estimates.values():
        assert np.all(np.isfinite(estimate.influence_curve))


# ------------------------------------------------------ check 12: refusals and records


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"survival_model": "iterated_mean"}, "survival_model"),
        ({"targeting": "one_step"}, "universal submodel"),
        ({"q_bounds": (0.0, 1.0)}, "q_bounds"),
        ({"simultaneous": True}, "band"),
        ({"isotonic": True}, "isotonic"),
        ({"ratio_trim": 0.99}, "ratio_trim"),
        ({"tol": 1e-6}, "stopping"),
        ({"guard": ("Q",)}, "One block"),
        ({"reduction": "bivariate"}, "reduced-regression"),
        ({"randomized": True}, "randomized"),
        ({"n_bootstrap": 10}, "bootstrap"),
        ({"repeats": 2}, "bootstrap"),
        ({"density_bins": 10}, "density_bins"),
        ({"competing_learner": LogisticRegression()}, "competing_learner"),
    ],
)
def test_settings_refusals(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(CapabilityError, match=match):
        SurvivalDRTMLE(ARMS, **kwargs)


def test_design_refusals() -> None:
    frame = sample_frame(300, 9)
    frame["A"] = np.where(frame["W1"] > 1.0, 2, frame["A"])
    with pytest.raises(CapabilityError, match="binary"):
        fit(container(frame))


def test_record_round_trip_and_determinism(sampled: LongitudinalData, drift_result: Any) -> None:
    record = next(iter(drift_result.drift.values()))
    assert isinstance(record, DriftRecord)
    copy = pickle.loads(pickle.dumps(record))
    np.testing.assert_array_equal(copy.epsilon, record.epsilon)
    assert copy.reductions == record.reductions
    assert "iteration" in record.to_frame()
    assert "Drift correction" in str(record.summary())
    again = fit(sampled)
    for name, estimate in drift_result.estimates.items():
        assert again.estimates[name].psi == estimate.psi


def test_status_is_influence_curve(drift_result: Any) -> None:
    assert drift_result.inference_status == "influence_curve"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isfinite(drift_result.estimates["risk_regimen[a1 @ t=3]"].std_error)


# ------------------------------------------------------- on-call refusals, study layer


def test_curve_replays_are_refused(drift_result: Any) -> None:
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve
    from cleverly.sensitivity._simulated_confounding_longitudinal import longitudinal_fit_refusal

    with pytest.raises(CapabilityError, match="restricted mean survival time"):
        drift_result.rmst("a1", 3)
    with pytest.raises(CapabilityError, match="no truncation curve"):
        longitudinal_truncation_curve(drift_result, [0.05])
    assert longitudinal_fit_refusal(drift_result) is drift.REPLAY_REFUSAL


def test_hazard_ltmle_fits_keep_their_replays(sampled: LongitudinalData) -> None:
    """The witness: the same calls on the guard-free fit are not refused by these rules."""
    from cleverly.sensitivity._simulated_confounding_longitudinal import longitudinal_fit_refusal

    plain = fit(sampled, guard=(), horizons=[1.0, 2.0, 3.0])
    assert plain.drift is None
    assert np.isfinite(plain.rmst("a1", 3).psi)
    assert longitudinal_fit_refusal(plain) is not drift.REPLAY_REFUSAL


def _study_frame() -> pd.DataFrame:
    return sample_frame(600, 21)


def test_the_study_layer_fits_a_time_to_event_design_by_drift_corrected_tmle() -> None:
    from cleverly import CausalStudy, RegimeContrast, TimeToEvent
    from cleverly.methods import DRTMLEMethod

    study = CausalStudy(
        _study_frame(),
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )
    effect = study.identify(RegimeContrast(regimens=ARMS, reference="a0"))
    record = next(r for r in effect.available_methods() if r.name == "drtmle")
    assert record.available
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = effect.estimate(
            method=DRTMLEMethod(),
            outcome_learner=glm(),
            treatment_learner=glm(),
            censoring_learner=glm(),
            cross_fit=False,
            random_state=0,
        )
    assert type(result.estimator) is SurvivalDRTMLE
    assert result.drift is not None
    assert result.inference_status == "influence_curve"


def test_the_study_layer_refuses_drift_correction_on_a_sequential_design() -> None:
    from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
    from cleverly.datasets import make_longitudinal

    frame, _ = make_longitudinal(n=200, seed=0)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        ),
    )
    effect = study.identify(RegimeContrast(regimens={"always": 1, "never": 0}, reference="never"))
    record = next(r for r in effect.available_methods() if r.name == "drtmle")
    assert not record.available
    assert record.refusal is not None and record.refusal.roadmap == "F5"


def test_summary_names_the_drift_correction(drift_result: Any, sampled: LongitudinalData) -> None:
    assert "survival DR-TMLE" in str(drift_result.summary())
    assert "DR-TMLE" not in str(fit(sampled, guard=()).summary())


# ------------------------------------------------------- check 6: independent reference


def _reference_newton(y: np.ndarray, offset: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Plain Newton for an unpenalized logistic fit with an offset; zero columns stay zero."""
    eps = np.zeros(x.shape[1])
    live = [j for j in range(x.shape[1]) if np.any(x[:, j] != 0.0)]
    if not live or y.size == 0:
        return eps
    design = x[:, live]
    beta = np.zeros(len(live))
    for _ in range(200):
        mu = expit(offset + design @ beta)
        step = np.linalg.solve((design.T * (mu * (1.0 - mu))) @ design, design.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-15:
            break
    eps[live] = beta
    return eps


def reference_drift(
    view: drift.PaperView,
    tau: int,
    *,
    g_bounds: tuple[float, float] = (0.01, 1.0),
    alpha: float = 0.9995,
    max_iter: int = 200,
    d_r_sign: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Steps 1 to 5 of Díaz (2019), Section 5, written from the design's section 4.

    It shares only the smoother with the engine.  Products are formed as the ratios eq. (9)
    prints, the clever covariates in plain loops over times, and the corrected signs and
    indices of the module docstring are written out where they enter.
    """
    from cleverly.longitudinal._kernel import smooth

    n, (lo, hi) = view.n, g_bounds
    a = view.arm
    treated = a > 0.0
    everyone = np.ones(n, dtype=bool)

    def surv(h: np.ndarray) -> np.ndarray:
        out = np.ones((n, tau + 1))
        for t in range(1, tau + 1):
            out[:, t] = out[:, t - 1] * (1.0 - h[:, t - 1])
        return out

    def ret(g_r: np.ndarray) -> np.ndarray:
        out = np.ones((n, tau + 1))
        for t in range(1, tau + 1):
            out[:, t] = out[:, t - 1] * (1.0 - g_r[:, t - 1])
        return out

    def reg(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return smooth(x, y, rows).values

    g_a, g_r, h = view.g_a.copy(), view.g_r.copy(), view.h.copy()
    L, R, I, J = view.events, view.censored, view.at_risk, view.eligible

    # Step 1: the indices and the reductions, from the initial fits, held fixed.
    s0, big_g0 = surv(h), ret(g_r)
    pi0 = np.clip(g_a[:, None] * big_g0, lo, hi)
    c_g = pi0
    c_h = s0[:, [tau]] / s0
    e_r = {}
    for k in range(tau):
        if view.held_censoring[k]:
            e_r[k] = np.zeros(n)
        else:
            e_r[k] = reg(c_h[:, k], (R[:, k] - g_r[:, k]) / pi0[:, k + 1], J[:, k] & treated)
    e_l = {
        t: reg(c_g[:, t], c_h[:, t] * (L[:, t - 1] - h[:, t - 1]), I[:, t - 1] & treated)
        for t in range(1, tau + 1)
    }
    e_a = np.zeros(n) if view.held_treatment else reg(s0[:, tau], (a - g_a) / g_a, everyone)
    q = np.clip(reg(s0[:, tau], a, everyone), lo, 1.0 - lo)

    @functools.cache
    def u(t: int, m: int) -> np.ndarray:
        return reg(c_g[:, t], R[:, m], J[:, m] & treated)

    @functools.cache
    def v(t: int, m: int) -> np.ndarray:
        return reg(c_g[:, t], L[:, m - 1], I[:, m - 1] & treated)

    @functools.cache
    def b(k: int, m: int) -> np.ndarray:
        return reg(c_h[:, k], L[:, m - 1], I[:, m - 1] & treated)

    @functools.cache
    def d(k: int, m: int) -> np.ndarray:
        return reg(c_h[:, k], R[:, m], J[:, m] & treated)

    def prod(factors: list[np.ndarray]) -> np.ndarray:
        out = np.ones(n)
        for f in factors:
            out = out * (1.0 - f)
        return out

    def big_u(t: int, upto: int) -> np.ndarray:  # U_t(upto) = prod_{m=0}^{upto-1}
        return prod([u(t, m) for m in range(upto)])

    def big_v(t: int, upto: int) -> np.ndarray:  # V_t(upto) = prod_{m=1}^{upto}
        return prod([v(t, m) for m in range(1, upto + 1)])

    def big_b(k: int, upto: int) -> np.ndarray:  # B_k(upto) = prod_{m=1}^{upto}
        return prod([b(k, m) for m in range(1, upto + 1)])

    def big_d(k: int, upto: int) -> np.ndarray:  # D_k(upto) = prod_{m=0}^{upto-1}
        return prod([d(k, m) for m in range(upto)])

    def covariates(
        g_a: np.ndarray, g_r: np.ndarray, h: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s, big_g = surv(h), ret(g_r)
        # G(k)/G(t) is read as the cumulative ratio g_A G(k) / g_A G(t), so the floored
        # cumulative is the only denominator, as the design's floor states.
        raw = g_a[:, None] * big_g
        pi = np.clip(raw, lo, hi)
        h_a = np.zeros(n)
        for t in range(1, tau + 1):
            h_a += big_u(t, t) * big_v(t, t - 1) * e_l[t] / pi[:, t]
        h_r = np.zeros((n, tau))
        for k in range(tau):
            total = np.zeros(n)
            for t in range(k + 1, tau + 1):
                total += (
                    big_v(t, t - 1)
                    / big_v(t, k)
                    * big_u(t, t)
                    / big_u(t, k)
                    * raw[:, k]
                    / pi[:, t]
                    * e_l[t]
                )
            h_r[:, k] = total / pi[:, k + 1]
        h_l = np.zeros((n, tau))
        for t in range(1, tau + 1):
            inner = np.zeros(n)
            for k in range(t):
                # Corrected sign: the e_R sum enters with the sign opposite to the e_A term.
                inner -= (
                    s[:, t - 1]
                    / s[:, k]
                    * big_b(k, k)
                    / big_b(k, t - 1)
                    * big_d(k, k)
                    / big_d(k, t)
                    * e_r[k]
                )
            # Corrected index: the e_A term's products sit on C_h(0) = S(tau).
            inner += e_a * s[:, t - 1] / (q * big_d(0, t) * big_b(0, t - 1))
            h_l[:, t - 1] = s[:, tau] / s[:, t] * inner
        z = s[:, [tau]] / s[:, 1:] / pi[:, 1:]
        return h_a, h_r, h_l, z

    clip = float(logit(alpha))
    threshold = 1e-4 * n ** (-0.6)
    for _ in range(max_iter):
        h_a, h_r, h_l, z = covariates(g_a, g_r, h)
        eps_a = 0.0
        if not view.held_treatment:
            eps_a = _reference_newton(a, logit(g_a), h_a[:, None])[0]
        nodes_r = [m for m in range(tau) if not view.held_censoring[m]]
        eps_r = 0.0
        if nodes_r:
            pick = [(J[:, m] & treated, m) for m in nodes_r]
            eps_r = _reference_newton(
                np.concatenate([R[rows, m] for rows, m in pick]),
                np.concatenate([logit(g_r[rows, m]) for rows, m in pick]),
                np.concatenate([h_r[rows, m] for rows, m in pick])[:, None],
            )[0]
        nodes_l = [t for t in range(1, tau + 1) if not view.held_hazard[t - 1]]
        pick_l = [(I[:, t - 1] & treated, t) for t in nodes_l]
        eps_l = _reference_newton(
            np.concatenate([L[rows, t - 1] for rows, t in pick_l]),
            np.concatenate([logit(h[rows, t - 1]) for rows, t in pick_l]),
            np.vstack([np.column_stack([h_l[rows, t - 1], z[rows, t - 1]]) for rows, t in pick_l]),
        )
        if not view.held_treatment:
            g_a = expit(np.clip(logit(g_a) + eps_a * h_a, -clip, clip))
        g_r = g_r.copy()
        for m in nodes_r:
            g_r[:, m] = expit(np.clip(logit(g_r[:, m]) + eps_r * h_r[:, m], -clip, clip))
        h = h.copy()
        for t in nodes_l:
            shift = eps_l[0] * h_l[:, t - 1] + eps_l[1] * z[:, t - 1]
            h[:, t - 1] = expit(np.clip(logit(h[:, t - 1]) + shift, -clip, clip))
        if max(abs(eps_a), abs(eps_r), *np.abs(eps_l)) < threshold:
            break

    # Step 5 and the corrected curve of Theorem 2.
    h_a, h_r, h_l, z = covariates(g_a, g_r, h)
    s = surv(h)
    theta = float(np.mean(s[:, tau]))
    eif = s[:, tau] - theta
    d_l = np.zeros(n)
    for t in range(1, tau + 1):
        resid = np.where(I[:, t - 1], L[:, t - 1] - h[:, t - 1], 0.0)
        eif -= a * z[:, t - 1] * resid
        d_l -= a * h_l[:, t - 1] * resid
    d_r = np.zeros(n)
    for k in range(tau):
        if not view.held_censoring[k]:
            # Corrected sign: D_R = + sum_k A J_k H_R(k) (R_k - g_R(k)).
            d_r += d_r_sign * np.where(J[:, k], a * h_r[:, k] * (R[:, k] - g_r[:, k]), 0.0)
    d_a = np.zeros(n) if view.held_treatment else -h_a * (a - g_a)
    return 1.0 - theta, -(eif - d_l - d_r - d_a)


@pytest.fixture(scope="module")
def reference_views(sampled: LongitudinalData) -> dict[int, drift.PaperView]:
    return _views(sampled)


@pytest.mark.parametrize("code", [1, 0])
def test_independent_reference_matches_the_engine(
    reference_views: dict[int, drift.PaperView], code: int
) -> None:
    view = reference_views[code]
    tau, bounds = TAU, (0.01, 1.0)
    out = drift.target_drift(
        view, tau, label="a", g_bounds=bounds, alpha=0.9995, max_iter=200, n_folds=2
    )
    record = out.record
    # Live drift in every block; no product floor binds, so the printed ratios apply.
    assert record.converged and record.n_iter >= 2
    assert np.all(np.abs(record.epsilon[0]) > 1e-8), record.epsilon[0]
    assert all(count == 0 for _, count in record.denominator_hits)
    risk, curve = reference_drift(view, tau, g_bounds=bounds)
    assert abs(risk - out.curve.risk) < 1e-10
    np.testing.assert_allclose(curve, out.curve.curve, rtol=0.0, atol=1e-10)
    # Mutation: the D_R sign of the transcribed eq. (9) moves the curve.
    _, flipped = reference_drift(view, tau, g_bounds=bounds, d_r_sign=-1.0)
    assert np.max(np.abs(flipped - out.curve.curve)) > 1e-4


# ------------------------------------------------- review follow-ups (X91-F-impl-review)


def _wide(frame: pd.DataFrame, n_times: int = 5) -> pd.DataFrame:
    """The time-to-event frame as a wide panel: censoring after the event check at ``t`` is
    ``C_{t+1} = 0``, and every node after an event or a censoring is missing."""
    wide = frame[["W1", "W2", "A"]].copy()
    event, time = frame["event"].to_numpy(), frame["time"].to_numpy()
    for k in range(1, n_times + 1):
        event_before = (event == 1) & (time < k)
        censored_before = (event == 0) & (time < k)
        long_gone = event_before | ((event == 0) & (time < k - 1))
        wide[f"C{k}"] = np.where(long_gone, np.nan, np.where(censored_before, 0.0, 1.0))
        wide[f"Y{k}"] = np.where(
            event_before | censored_before, np.nan, ((event == 1) & (time == k)).astype(float)
        )
    return wide


def test_time_to_event_and_wide_panel_give_the_same_fit() -> None:
    """Check 2's independent witness: two constructions of the same law agree."""
    frame = sample_frame(1200, 31)
    wide = LongitudinalData.from_frame(
        _wide(frame),
        outcome=[f"Y{k}" for k in range(1, 6)],
        treatment="A",
        baseline=["W1", "W2"],
        censoring=[f"C{k}" for k in range(1, 6)],
    )
    one, two = fit(container(frame)), fit(wide)
    for name, estimate in one.estimates.items():
        other = two.estimates[name]
        np.testing.assert_allclose(other.psi, estimate.psi, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(other.std_error, estimate.std_error, rtol=1e-8)


def test_declared_censoring_with_an_estimated_treatment_is_refused() -> None:
    """A declared retention is read at the received arm, but H_A needs every arm's (review H1)."""
    frame = sample_frame(400, 32, known=True)
    data = container(frame, censoring_probabilities={float(g): f"r{g}" for g in range(1, 4)})
    with pytest.raises(CapabilityError, match="treatment_probabilities"):
        fit(data)
    assert drift.drift_design_refusal(data, ARMS) is drift.DECLARED_CENSORING_REFUSAL


@pytest.fixture(scope="module")
def partial_view() -> drift.PaperView:
    frame = sample_frame(1500, 33, known=True)
    data = container(
        frame,
        treatment_probabilities={1: "p1", 0: "p0"},
        censoring_probabilities={1.0: "r1"},
    )
    return _views(data)[1]


def test_curve_sums_d_r_over_estimated_nodes_only(partial_view: drift.PaperView) -> None:
    """The node-2 guard of the curve's D_R, with the sigma mutation of check 9."""
    view = partial_view
    assert view.held_censoring[1] and not view.held_censoring[2]
    out = drift.target_drift(
        view, 3, label="a1", g_bounds=(0.01, 1.0), alpha=0.9995, max_iter=200, n_folds=2
    )
    state, h_r = out.state, out.covariates.h_r

    def term(m: int) -> np.ndarray:
        rows = view.eligible[:, m]
        return np.where(rows, view.arm * h_r[:, m] * (view.censored[:, m] - state.g_r[:, m]), 0.0)

    np.testing.assert_allclose(out.curve.d_r, term(2), rtol=0.0, atol=1e-14)
    held = term(1)
    assert np.std(held) > 1e-6
    mutated = out.curve.curve + held
    assert abs(np.std(mutated) / np.std(out.curve.curve) - 1.0) > 1e-6


def test_declared_factors_zero_their_residual_regressions(partial_view: drift.PaperView) -> None:
    """e_R at a declared node and e_A under a declared treatment are zero, and would not be."""
    view = partial_view
    state = drift._State(view.g_a.copy(), view.g_r.copy(), view.h.copy())
    guarded = drift._Reductions(view, state, 3, (0.01, 1.0))
    np.testing.assert_array_equal(guarded.e_r(1), 0.0)
    np.testing.assert_array_equal(guarded.e_a(), 0.0)
    free_view = drift.PaperView(
        **{
            **view.__dict__,
            "held_treatment": False,
            "held_censoring": np.zeros_like(view.held_censoring),
        }
    )
    free = drift._Reductions(free_view, state, 3, (0.01, 1.0))
    assert np.max(np.abs(free.e_r(1))) > 1e-6
    assert np.max(np.abs(free.e_a())) > 1e-6


def test_every_g_declared_leaves_h_l_identically_zero() -> None:
    frame = sample_frame(1200, 34, known=True)
    data = container(
        frame,
        treatment_probabilities={1: "p1", 0: "p0"},
        censoring_probabilities={float(g): f"r{g}" for g in range(1, 5)},
    )
    view = _views(data)[1]
    out = drift.target_drift(
        view, 3, label="a1", g_bounds=(0.01, 1.0), alpha=0.9995, max_iter=200, n_folds=2
    )
    np.testing.assert_array_equal(out.covariates.h_l, 0.0)
    np.testing.assert_array_equal(out.record.epsilon[:, :3], 0.0)


def _saturating(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Binary regressions return one, so the B and D products are exactly zero."""
    from cleverly.longitudinal._kernel import smooth

    if set(np.unique(y[rows])) <= {0.0, 1.0}:
        return np.ones(x.shape[0])
    return smooth(x, y, rows).values


def test_product_floor_binds_and_keeps_the_fit_finite(
    reference_views: dict[int, drift.PaperView], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check 10 with a binding floor, and its mutation (review M2)."""
    view = reference_views[1]
    settings: dict[str, Any] = {
        "label": "a1",
        "g_bounds": (0.01, 1.0),
        "alpha": 0.9995,
        "max_iter": 200,
        "n_folds": 2,
        "regress": _saturating,
        "warn": False,
    }
    out = drift.target_drift(view, TAU, **settings)
    hits = dict(out.record.denominator_hits)
    assert hits["B"] > 0 and hits["D"] > 0
    assert np.isfinite(out.curve.risk) and np.all(np.isfinite(out.curve.curve))
    monkeypatch.setattr(drift, "DENOMINATOR_FLOOR", 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            broken = drift.target_drift(view, TAU, **settings)
        except Exception:
            return
    assert not (np.isfinite(broken.curve.risk) and np.all(np.isfinite(broken.curve.curve)))


def test_a_newton_failure_raises_a_recordable_error(
    reference_views: dict[int, drift.PaperView], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submodel with no finite maximum is a failed fit, not a reported one (review M4)."""
    from cleverly.exceptions import LongitudinalError
    from cleverly.validation._failures import recordable

    monkeypatch.setattr(
        drift, "_logistic_epsilon", lambda y, offset, x: np.full(x.shape[1], np.nan)
    )
    with pytest.raises(LongitudinalError, match="no finite maximum") as raised:
        drift.target_drift(
            reference_views[1],
            3,
            label="a1",
            g_bounds=(0.01, 1.0),
            alpha=0.9995,
            max_iter=5,
            n_folds=2,
        )
    assert recordable(raised.value)
