"""Data-tuned caps of a modified treatment policy, longitudinal fits (X40 part (e)).

``X40-plan.md`` sections 2.4 and 2.5.  The population target ``Psi(P) = Phi(P, theta(P))``
adds the chain-rule term ``sum_t dPhi_t' IF_t`` to the curve; T-e4 checks that curve on law
Q (``tests/law_q_tuned.py``) as the pathwise derivative along exponential tilts, by
quadrature, and checks the fitted curve against the same formula computed longhand.  T-e7
checks the training-tuner construction of a cross-fitted fit on both ratio routes (M3,
N3).  The fold-average target fits each fold's map on that fold's training rows and
targets each fold on its own rows: T-e1 to T-e3 on the exact grid law.  T-e5 checks the
kernel estimate of the derivative by its rate, T-e6 pins the refusals R-e1 to R-e4, and
T-e8 checks that the full-refit bootstrap refits the tuner.  Each mutation control M-e*
is a monkeypatch of the shipped code, and each must move its witness by more than
``CAUGHT``.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import interpolate, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor

from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    DoseQuantile,
    LinearQuantileRegression,
    Scale,
    Shift,
    SupportBoundary,
)
from cleverly.interventions import tuning as tuning_module
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, sequential
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import regimen as regimen_module

from .. import law_q_tuned as lawq

CAUGHT = 1e-4
IDENTITY = 1e-6
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
QUANTILE = (lawq.Tuner(), lawq.Tuner())
REGRESSION = (lawq.Tuner(features=("W",)), lawq.Tuner(features=("W", "L2", "A1")))
MISSPECIFIED = (lawq.Tuner(features=("W",)), lawq.Tuner(features=("W",)))
ORACLE_TUNERS = {"quantile": QUANTILE, "regression": REGRESSION, "misspecified": MISSPECIFIED}


class _Raises(BaseEstimator, RegressorMixin):
    """A learner that fails the test if any fit reaches it."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Raises:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, X: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used before the refusal")


def _caps(kind: str) -> tuple[Any, Any]:
    if kind == "quantile":
        return DoseQuantile(0.5), DoseQuantile(0.5)
    if kind == "regression":
        return (
            SupportBoundary(LinearQuantileRegression(["W"]), 0.5),
            SupportBoundary(LinearQuantileRegression(["W", "L2", "A1"]), 0.5),
        )
    return (
        SupportBoundary(LinearQuantileRegression(["W"]), 0.5),
        SupportBoundary(LinearQuantileRegression(["W"]), 0.5),
    )


def _regimen(target: str | None, kind: str = "quantile", label: str = "tuned") -> DynamicRegimen:
    cap1, cap2 = _caps(kind)
    return DynamicRegimen(
        label, (Shift(1.0, cap=cap1, target=target), Shift(1.0, cap=cap2, target=target))
    )


def _settings(**overrides: Any) -> dict[str, Any]:
    return {
        "q_bounds": (0.0, 1.0),
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "density_bins": 20,
        "random_state": 1,
        "simultaneous": False,
        **overrides,
    }


def _fit(estimator: type, regimens: Any, frame: pd.DataFrame, **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(regimens, **_settings(**settings)).fit(frame, **COLUMNS)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return lawq.sample(1500, 11)


# ------------------------------------------------------------------ oracle


def test_the_oracle_does_not_import_cleverly() -> None:
    path = Path(lawq.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [name for name in imported if name.split(".")[0] == "cleverly"]


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_the_oracle_derivative_is_the_derivative_of_the_mean_in_theta(kind: str) -> None:
    """The analytic dPhi/dtheta_t of plan section 2.5 equals a finite difference of Phi."""
    law = lawq.LawQ(tuners=ORACLE_TUNERS[kind])
    theta = law.theta
    for node in (1, 2):
        analytic = law.derivative(node)
        numeric = []
        for j in range(theta[node - 1].size):

            def at(step: float, node: int = node, j: int = j) -> float:
                moved = [value.copy() for value in theta]
                moved[node - 1][j] += step
                return law.psi((moved[0], moved[1]))

            h = 1e-4
            numeric.append((8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h))
        # The difference step leaves an error near 1e-8 where a split point of the
        # quadrature moves with theta; the identity test reads 1e-6.
        np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-6)
        assert np.max(np.abs(analytic)) > 1e-3


@pytest.mark.parametrize("path", ["w", "a1", "l2", "a2"])
@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_t_e4_the_population_curve_is_the_pathwise_derivative(kind: str, path: str) -> None:
    """T-e4 (i): d/de Psi(P_e) along a tilt equals E[D_Psi s], by quadrature, to 1e-6.

    The tilt of a dose law, of ``L_2`` and of ``W`` each moves the tuner (the quantile, or the
    pseudo-true beta), so the identity needs the chain-rule term; the misspecified set
    reads ``W`` alone at node 2 (m3).
    """
    law = lawq.LawQ(tuners=ORACLE_TUNERS[kind])
    pathwise = lawq.pathwise_derivative(law, path)
    assert abs(pathwise - law.score_pairing(path)) < IDENTITY


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_m_e4_the_identity_fails_without_the_tuning_term(kind: str) -> None:
    """M-e4 at the oracle: dropping the term breaks the identity on the dose paths."""
    law = lawq.LawQ(tuners=ORACLE_TUNERS[kind])
    for path in ("a1", "a2"):
        gap = abs(lawq.pathwise_derivative(law, path) - law.score_pairing(path, tuning=False))
        assert gap > CAUGHT


def test_t_e4_law_q_carries_a_large_tuning_term() -> None:
    """The nonzero requirement: SD of the term over SD of D_Phi is above the 0.1 floor."""
    for kind in ("quantile", "regression"):
        term, phi = lawq.LawQ(tuners=ORACLE_TUNERS[kind]).spreads()
        assert term / phi > 0.1


# ------------------------------------------------------------------ refusals (T-e6)


R_E1 = (
    "policy '+1': a data-tuned map reads the sample, so which parameter it estimates is a "
    "declaration. Declare target='population', the mean under the map at the population "
    "value of the tuner (a dose quantile, or LinearQuantileRegression), or "
    "target='fold_average', the average over folds of the means under each fold's own map, "
    "fitted on that fold's training rows"
)
R_E2 = (
    "policy '+1' tunes its map with SupportBoundary, whose learner "
    "'GradientBoostingRegressor' has no declared finite-dimensional parameter. The "
    "population target needs the influence function of the tuned boundary, and only a "
    "boundary indexed by a finite vector that solves an estimating equation has one here (a "
    "dose quantile, or LinearQuantileRegression with its estimating equation). A learner of "
    "growing dimension has no such curve in any result this package uses. Declare "
    "target='fold_average', which estimates the average over folds of the means under each "
    "fold's own map, or tune the boundary with DoseQuantile or LinearQuantileRegression"
)
R_E3 = (
    "policy '+1' tunes its map with the outcome. A map chosen to optimize the outcome is a "
    "learned policy, whose value needs a limiting-policy condition and its own contract "
    "(docs/roadmap.md X11). A data-tuned map in this release reads the treatment law only: a "
    "dose quantile or a learned support boundary"
)
R_E4 = (
    "target='fold_average' builds each fold's map on that fold's training rows and "
    "estimates its mean on the held-out rows, so it needs n_folds of at least 2. For an "
    "in-sample fit, tune the map with a finite-dimensional tuner and declare "
    "target='population'"
)


def _raising(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": _Raises(),
        "pseudo_learner": _Raises(),
        "treatment_learner": _Raises(),
        **overrides,
    }


def _boosted() -> SupportBoundary:
    return SupportBoundary(
        GradientBoostingRegressor(loss="quantile", alpha=0.5), 0.5, features=("W",)
    )


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e6_r_e1_a_tuned_map_declares_its_target(estimator: type, frame: Any) -> None:
    """R-e1 is raised when the regimen is declared, so no learner is fitted."""
    with pytest.raises(CapabilityError) as caught:
        regimen = DynamicRegimen("t", (Shift(1.0, cap=DoseQuantile(0.5)), Shift(0.0, cap=None)))
        _fit(estimator, [regimen], frame, **_raising())
    assert str(caught.value) == R_E1


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e6_r_e2_the_population_target_needs_a_finite_dimensional_boundary(
    estimator: type, frame: Any
) -> None:
    with pytest.raises(CapabilityError) as caught:
        regimen = DynamicRegimen(
            "t", (Shift(1.0, cap=_boosted(), target="population"), Shift(0.0, cap=None))
        )
        _fit(estimator, [regimen], frame, **_raising())
    assert str(caught.value) == R_E2


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e6_r_e2_does_not_refuse_a_linear_quantile_regression(
    estimator: type, frame: Any
) -> None:
    result = _fit(estimator, [_regimen("population", "regression")], frame, n_folds=1)
    assert np.isfinite(result["ey_regimen[tuned]"].psi)


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e6_r_e2_does_not_refuse_a_boundary_learner_on_the_fold_average(
    estimator: type, frame: Any
) -> None:
    regimen = DynamicRegimen(
        "t", (Shift(1.0, cap=_boosted(), target="fold_average"), Shift(0.0, cap=None))
    )
    result = _fit(estimator, [regimen], frame, n_folds=2)
    assert np.isfinite(result["ey_regimen[t, fold average]"].psi)


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e6_r_e3_a_tuner_does_not_read_the_outcome(estimator: type, frame: Any) -> None:
    cap = SupportBoundary(LinearQuantileRegression(["W", "Y"]), 0.5)
    regimen = DynamicRegimen("t", (Shift(1.0, cap=cap, target="population"), Shift(0.0, cap=None)))
    with pytest.raises(CapabilityError) as caught:
        _fit(estimator, [regimen], frame, **_raising())
    assert str(caught.value) == R_E3


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e6_r_e4_the_fold_average_needs_two_folds(estimator: type, frame: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        _fit(estimator, [_regimen("fold_average")], frame, n_folds=1, **_raising())
    assert str(caught.value) == R_E4


def test_the_fold_average_refuses_the_full_refit_bootstrap(frame: Any) -> None:
    with pytest.raises(CapabilityError, match="refuses the full-refit bootstrap"):
        _fit(LTMLE, [_regimen("fold_average")], frame, n_folds=2, n_bootstrap=2, **_raising())


def test_the_fold_average_refuses_a_working_model(frame: Any) -> None:
    from cleverly.msm import MSM

    model = MSM(
        design=lambda label, horizon, w: np.ones((len(w), 1)),
        terms=("intercept",),
        design_kind="known",
    )
    with pytest.raises(CapabilityError, match="is not combined with msm="):
        _fit(LTMLE, [_regimen("fold_average")], frame, n_folds=2, msm=model, **_raising())


def test_a_target_declares_a_tuned_map_only() -> None:
    with pytest.raises(DataError, match="a target declares the parameter of a data-tuned map"):
        Shift(1.0, cap=2.0, target="population")
    with pytest.raises(DataError, match="target must be 'population', 'fold_average' or None"):
        Scale(1.1, cap=DoseQuantile(0.5), target="both")


def test_one_regimen_declares_one_target(frame: Any) -> None:
    regimen = DynamicRegimen(
        "t",
        (
            Shift(1.0, cap=DoseQuantile(0.5), target="population"),
            Shift(1.0, cap=DoseQuantile(0.5), target="fold_average"),
        ),
    )
    with pytest.raises(CapabilityError, match="declares the targets"):
        _fit(LTMLE, [regimen], frame, n_folds=2, **_raising())


def test_an_unresolved_tuned_map_has_no_map() -> None:
    with pytest.raises(CapabilityError, match="has no map until a fit resolves the tuner"):
        Shift(1.0, cap=DoseQuantile(0.5), target="population").apply(np.zeros(3))


# ------------------------------------------------------------------ the curve (T-e4 ii)


def _silverman(values: np.ndarray, power: float = 0.2) -> float:
    sd = float(np.std(values))
    iqr = float(
        np.quantile(values, 0.75, method="inverted_cdf")
        - np.quantile(values, 0.25, method="inverted_cdf")
    )
    return 0.9 * min(sd, iqr / 1.34) * values.size ** (-power)


def _longhand_tuner(frame: pd.DataFrame, node: int, kind: str) -> tuple[Any, ...]:
    """theta_n, the influence function, the boundary and the gradient, written out."""
    dose = frame[f"A{node}"].to_numpy()
    n = dose.size
    if kind == "quantile":
        theta = np.array([np.quantile(dose, 0.5, method="inverted_cdf")])
        h = _silverman(dose)
        density = float(np.mean(stats.norm.pdf((theta[0] - dose) / h)) / h)
        influence = (-((dose <= theta[0]).astype(float) - 0.5) / density)[:, None]
        gradient = np.ones((n, 1))
        return theta, influence, np.full(n, theta[0]), gradient
    names = ["W"] if node == 1 or kind == "misspecified" else ["W", "L2", "A1"]
    features = frame[names].to_numpy()
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(features, dose)
    theta = np.concatenate([[model.intercept_], model.coef_])
    design = np.column_stack([np.ones(n), features])
    residual = dose - design @ theta
    h = _silverman(residual, power=1.0 / 3.0)
    inside = (np.abs(residual) <= h).astype(float)
    jacobian = (design * inside[:, None]).T @ design / (2.0 * h * n)
    score = design * (0.5 - (residual <= 0.0).astype(float))[:, None]
    influence = np.linalg.solve(jacobian, score.T).T
    return theta, influence, design @ theta, design


def _longhand_term(frame: pd.DataFrame, fit: Any, kind: str) -> np.ndarray:
    term = np.zeros(len(frame))
    by_time = {step.time: step for step in fit.steps}
    for node in (1, 2):
        _, influence, boundary, gradient = _longhand_tuner(frame, node, kind)
        dose = frame[f"A{node}"].to_numpy()
        edge = boundary - 1.0
        step = by_time[node]
        prefix = np.ones(len(frame)) if node == 1 else by_time[node - 1].clever
        gap = dose - edge
        h = _silverman(gap)
        kernel = stats.norm.pdf(gap / h) / h
        derivative = (prefix * kernel * (step.probes[:, 0] - step.probes[:, 1])) @ gradient
        term = term + influence @ (derivative / len(frame))
    return term


@pytest.mark.parametrize(
    ("estimator", "folds", "kind"),
    [
        (LTMLE, 1, "quantile"),
        (LTMLE, 3, "quantile"),
        (LTMLE, 3, "regression"),
        (SDR, 1, "regression"),
        (SDR, 3, "quantile"),
        (LTMLE, 3, "misspecified"),
    ],
)
def test_t_e4_the_reported_curve_adds_the_tuning_term_longhand(
    estimator: type, folds: int, kind: str, frame: Any
) -> None:
    """T-e4 (ii): the curve is the curve at theta_n plus sum_t dPhi_n' IF_t, longhand."""
    result = _fit(estimator, [_regimen("population", kind)], frame, n_folds=folds)
    fit = result.fits["tuned"]
    term = _longhand_term(frame, fit, kind)
    np.testing.assert_allclose(fit.tuning_term, term, rtol=1e-8, atol=1e-10)
    first = fit.steps[0]
    if estimator is SDR:
        base = first.transform - fit.psi_scaled
    else:
        base = first.value - fit.psi_scaled
        for step in fit.steps:
            base = base + step.clever * (step.pseudo_outcome - step.targeted)
    np.testing.assert_allclose(fit.influence_curve_scaled, base + term, rtol=0, atol=1e-12)
    assert np.std(term) > 0.1 * np.std(base)


def test_m_e4_dropping_the_term_moves_the_reported_curve(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e4 in the library: without the term the curve misses the longhand curve."""
    kept = _fit(LTMLE, [_regimen("population")], frame, n_folds=1).fits["tuned"]
    monkeypatch.setattr(sequential, "tuning_term", lambda data, plan, steps: (None, ()))
    dropped = _fit(LTMLE, [_regimen("population")], frame, n_folds=1).fits["tuned"]
    gap = np.max(np.abs(kept.influence_curve_scaled - dropped.influence_curve_scaled))
    assert gap > CAUGHT
    np.testing.assert_allclose(
        dropped.influence_curve_scaled + _longhand_term(frame, kept, "quantile"),
        kept.influence_curve_scaled,
        atol=1e-12,
    )


# ------------------------------------------------------------------ the kernel slope (T-e5)

SIZES = (2_000, 8_000, 32_000, 128_000)


def _kernel_slope(seeds: int) -> float:
    """The slope of the log RMSE of the kernel dPhi_1 against log n, with oracle m and theta.

    The kernel estimate alone is under test: the probes are the oracle's node-1 regression at
    the boundary and the last moved dose, and the tuner is the population quantile.
    """
    law = lawq.LawQ()
    truth = float(law.derivative(1)[0])
    top = float(law.theta[0][0])
    # The oracle's node-1 regression is smooth in W, so a cubic spline on a fine grid
    # carries it to the sample at an error far below the kernel's (checked off the knots).
    grid = np.linspace(*lawq.W_RANGE, 2001)
    between = 0.5 * (grid[1:] + grid[:-1])
    probes = []
    for c in (top, top - 1.0):
        spline = interpolate.CubicSpline(grid, law.q1(grid, np.full(grid.size, c)))
        exact = law.q1(between, np.full(between.size, c))
        np.testing.assert_allclose(spline(between), exact, rtol=0, atol=1e-9)
        probes.append(spline)
    errors = []
    for n in SIZES:
        squared = []
        for seed in range(seeds):
            draw = lawq.sample_node1(n, seed)
            estimate = tuning_module.cap_derivative(
                weights=np.ones(n),
                prefix=np.ones(n),
                dose=draw["A1"].to_numpy(),
                edge=np.full(n, top - 1.0),
                high=probes[0](draw["W"].to_numpy()),
                low=probes[1](draw["W"].to_numpy()),
                gradient=np.ones((n, 1)),
                slope=1.0,
                at_risk=np.ones(n, dtype=bool),
                rule="silverman",
            )
            squared.append((float(estimate[0]) - truth) ** 2)
        errors.append(np.sqrt(np.mean(squared)))
    return float(np.polyfit(np.log(SIZES), np.log(errors), 1)[0])


def test_t_e5_the_kernel_derivative_is_consistent() -> None:
    """T-e5: Silverman's bandwidth gives the rate n^(-2/5); the band reads [-0.6, -0.25]."""
    assert -0.6 <= _kernel_slope(32) <= -0.25


def test_m_e5_a_fixed_bandwidth_is_not_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-e5: a fixed bandwidth of 1.0 leaves a bias that does not fall with n."""
    monkeypatch.setattr(tuning_module, "kernel_bandwidth", lambda *args, **kwargs: 1.0)
    assert _kernel_slope(8) > -0.1


# ------------------------------------------------------------------ instruments


def _median(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.5, method="inverted_cdf"))


def _up(dose: np.ndarray, cap: float) -> np.ndarray:
    """``Shift(1.0, cap=cap)`` written out."""
    return np.where(dose + 1.0 <= cap, dose + 1.0, dose)


def _cap(plan: Any, time: int) -> float:
    return float(plan.regimen.plan[time - 1].cap)


class _Log:
    """What a fit hands its tuner and its learners, recorded by monkeypatch."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.plans: list[Any] = []
        self.folds: Any = None
        self.tuners: list[tuple[np.ndarray, np.ndarray]] = []
        self.regressions: list[dict[str, Any]] = []
        resolve = estimator_module.resolve_tuned_plans

        def resolved(regimens: Any, data: Any, folds: Any) -> Any:
            plans = resolve(regimens, data, folds)
            self.plans.append(plans)
            self.folds = folds
            return plans

        monkeypatch.setattr(estimator_module, "resolve_tuned_plans", resolved)
        fit_tuner = regimen_module.fit_tuner

        def tuner(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
            fitted = fit_tuner(declared, doses, frame, rows, weights)
            self.tuners.append((np.array(rows, dtype=bool), np.array(fitted.theta)))
            return fitted

        monkeypatch.setattr(regimen_module, "fit_tuner", tuner)
        node = sequential._fit_node_regression

        def regression(data: Any, plan: Any, carried: Any, time: int, *args: Any, **kw: Any) -> Any:
            out = node(data, plan, carried, time, *args, **kw)
            self.regressions.append(
                {
                    "plan": plan,
                    "carried": np.array(carried, dtype=float),
                    "time": time,
                    "fold": kw.get("outer_fold"),
                    "rows": kw.get("fit_rows"),
                    "carry": kw.get("carry_plan"),
                    "out": out,
                }
            )
            return out

        monkeypatch.setattr(sequential, "_fit_node_regression", regression)

    def trains(self) -> list[np.ndarray]:
        out = []
        for train, _ in self.folds:
            rows = np.zeros(len(self.folds.assignment), dtype=bool)
            rows[train] = True
            out.append(rows)
        return out


# ------------------------------------------------------------------ the training tuner (T-e7)


def _doses(frame: pd.DataFrame, time: int) -> np.ndarray:
    return frame[f"A{time}"].to_numpy()


def _check_training_recursion(log: _Log, frame: pd.DataFrame) -> None:
    """T-e7 on the outcome recursion: fold k regresses its own map's doses (M3)."""
    trains = log.trains()
    (plan,) = log.plans[-1]
    folded = [entry for entry in log.regressions if entry["fold"] is not None]
    assert len(folded) == 2 * len(trains)
    for entry in folded:
        k, time = entry["fold"], entry["time"]
        own = _median(_doses(frame, time)[trains[k]])
        assert _cap(entry["carry"], time) == own
        # The held-out predictions and the targeting read the plan's own map, theta_n.
        assert _cap(entry["plan"], time) == _median(_doses(frame, time))
        assert entry["plan"].label == plan.label
    gaps = []
    for k, train in enumerate(trains):
        late = next(e for e in folded if e["fold"] == k and e["time"] == 2)
        early = next(e for e in folded if e["fold"] == k and e["time"] == 1)
        carried = late["out"].carry_marginal
        np.testing.assert_array_equal(early["carried"][train], carried[train])
        gaps.append(np.max(np.abs(late["out"].initial_marginal[train] - carried[train])))
    # The nonzero witness: in some fold the map at theta_n gives a different pseudo-outcome.
    # A fold whose training median is the full-sample median has no gap.
    assert max(gaps) > CAUGHT


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_e7_each_fold_trains_at_its_own_tuner(
    ratio: str, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-e7 (M3, N3): training at theta^(-k), held-out predictions and targeting at theta_n."""
    log = _Log(monkeypatch)
    branches: list[float] = []
    stacked: list[Any] = []
    if ratio == "density":
        branch_ratio = sequential.branch_ratio

        def recorded(branch: Any, *args: Any, **kwargs: Any) -> Any:
            branches.append(float(branch.cap))
            return branch_ratio(branch, *args, **kwargs)

        monkeypatch.setattr(sequential, "branch_ratio", recorded)
    else:
        classifier = sequential.classifier_ratio

        def recorded(*args: Any, **kwargs: Any) -> Any:
            stacked.append((np.array(args[2]), kwargs.get("assigned_by_fold")))
            return classifier(*args, **kwargs)

        monkeypatch.setattr(sequential, "classifier_ratio", recorded)
    _fit(LTMLE, [_regimen("population")], frame, n_folds=3, ratio=ratio)
    _check_training_recursion(log, frame)
    trains = log.trains()
    full = [_median(_doses(frame, time)) for time in (1, 2)]
    if ratio == "density":
        # N3, density route: fold k's training density is evaluated under the map at theta_n.
        assert branches
        assert set(branches) <= set(full)
        assert set(branches) == set(full)
        return
    # N3, classifier route: fold k's classifier is fitted on the doses of its own map.
    assert len(stacked) == 2
    for time, (dose, by_fold) in zip((1, 2), stacked, strict=True):
        np.testing.assert_array_equal(dose, _doses(frame, time))
        assert by_fold is not None
        assert len(by_fold) == len(trains)
        for k, assigned in enumerate(by_fold):
            ((probability, target),) = assigned
            assert probability == 1.0
            own = _median(_doses(frame, time)[trains[k]])
            np.testing.assert_array_equal(target, _up(dose, own))
        assert len({_median(_doses(frame, time)[train]) for train in trains}) > 1


def test_m_e7_a_training_recursion_at_theta_n_fails(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e7: the fold recursion carries the map at theta_n, and T-e7 catches it."""
    log = _Log(monkeypatch)
    recorded = sequential._fit_node_regression

    def mutated(data: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("carry_plan") is not None:
            kwargs["carry_plan"] = plan
        return recorded(data, plan, *args, **kwargs)

    monkeypatch.setattr(sequential, "_fit_node_regression", mutated)
    _fit(LTMLE, [_regimen("population")], frame, n_folds=3)
    with pytest.raises(AssertionError):
        _check_training_recursion(log, frame)


def test_m_e9_a_density_ratio_at_the_fold_tuner_fails(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e9: the density route reads fold k's map, and T-e7's density check catches it."""
    from dataclasses import replace

    log = _Log(monkeypatch)
    numerator = sequential._density_numerator
    branches: list[float] = []
    branch_ratio = sequential.branch_ratio

    def recorded(branch: Any, *args: Any, **kwargs: Any) -> Any:
        branches.append(float(branch.cap))
        return branch_ratio(branch, *args, **kwargs)

    monkeypatch.setattr(sequential, "branch_ratio", recorded)
    monkeypatch.setattr(
        sequential,
        "_density_numerator",
        lambda plan, *args: numerator(replace(plan, fold_average=True), *args),
    )
    _fit(LTMLE, [_regimen("population")], frame, n_folds=3)
    full = {_median(_doses(frame, time)) for time in (1, 2)}
    assert log.plans
    assert not set(branches) <= full


def test_t_e7_sdr_trains_at_its_own_tuner(frame: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """T-e7 on SDR: each fold's recursion carries its own map and returns the map at theta_n."""
    sdr_module = importlib.import_module("cleverly.longitudinal.sdr")
    log = _Log(monkeypatch)
    calls: list[tuple[Any, Any, Any]] = []
    recursion = sdr_module._recursion

    def recorded(data: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((plan, kwargs.get("carry_plan"), kwargs.get("fit_rows")))
        return recursion(data, plan, *args, **kwargs)

    monkeypatch.setattr(sdr_module, "_recursion", recorded)
    _fit(SDR, [_regimen("population")], frame, n_folds=3)
    trains = log.trains()
    folded = [call for call in calls if call[1] is not None]
    assert len(folded) == len(trains)
    for plan, carry, rows in folded:
        k = next(k for k, train in enumerate(trains) if np.array_equal(train, rows))
        for time in (1, 2):
            assert _cap(carry, time) == _median(_doses(frame, time)[trains[k]])
            assert _cap(plan, time) == _median(_doses(frame, time))


# ------------------------------------------------------------------ fold locality (T-e2, T-e3)


def _check_fold_maps(log: _Log, frame: pd.DataFrame) -> None:
    """T-e2: fold k's map is the tuner fitted on fold k's training rows alone."""
    trains = log.trains()
    (plan,) = log.plans[-1]
    assert plan.fold_average
    for k, train in enumerate(trains):
        for time in (1, 2):
            assert _cap(plan.fold_plans[k], time) == _median(_doses(frame, time)[train])
    fold_fits = [rows for rows, _ in log.tuners if not rows.all()]
    assert len(fold_fits) == 2 * len(trains)
    for rows in fold_fits:
        assert any(np.array_equal(rows, train) for train in trains)


def test_t_e2_each_fold_map_reads_its_training_rows(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log(monkeypatch)
    _fit(LTMLE, [_regimen("fold_average")], frame, n_folds=3)
    _check_fold_maps(log, frame)


def test_m_e2_a_full_sample_fold_map_fails(frame: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """M-e2: every fold's map is the full-sample tuner, and T-e2 catches it."""
    fit_tuner = regimen_module.fit_tuner
    monkeypatch.setattr(
        regimen_module,
        "fit_tuner",
        lambda declared, doses, frame, rows, weights: fit_tuner(
            declared, doses, frame, np.ones_like(rows, dtype=bool), weights
        ),
    )
    log = _Log(monkeypatch)
    _fit(LTMLE, [_regimen("fold_average")], frame, n_folds=3)
    with pytest.raises(AssertionError):
        _check_fold_maps(log, frame)


def _fold_scores(fit: Any, owner: np.ndarray) -> np.ndarray:
    """Each fold's targeting score at each node: the mean over its rows of H (Y* - Q*)."""
    return np.array(
        [
            [
                float(np.mean((step.clever * (step.pseudo_outcome - step.targeted))[owner == k]))
                for k in range(int(owner.max()) + 1)
            ]
            for step in fit.steps
        ]
    )


def test_t_e3_each_fold_solves_its_own_score(frame: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """T-e3: one fluctuation per fold and node, each solving its own validation score."""
    log = _Log(monkeypatch)
    result = _fit(LTMLE, [_regimen("fold_average")], frame, n_folds=3)
    fit = result.fits["tuned, fold average"]
    owner = np.asarray(log.folds.assignment)
    assert np.max(np.abs(_fold_scores(fit, owner))) < 1e-10
    assert len(fit.fold_estimates) == 3
    assert fit.psi_scaled == pytest.approx(np.mean(fit.fold_estimates), abs=1e-15)
    for step in fit.steps:
        assert len(set(step.fold_epsilon)) == 3


def test_m_e3_a_pooled_fluctuation_leaves_a_fold_score(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e3: one pooled coefficient per node leaves some fold's own score unsolved."""

    def pooled(data: Any, plan: Any, node: Any, *, time: int, **kwargs: Any) -> Any:
        solve = sequential._fluctuate_node(
            node.pseudo_outcome,
            node.initial,
            data.weights * node.counterfactual,
            node.fitted_on,
            label=plan.label,
            time=time,
            arms=node.initial_by_arm,
            **kwargs,
        )
        return solve, tuple(float(solve.epsilon[0]) for _ in range(plan.folds.n_folds))

    monkeypatch.setattr(sequential, "_fold_fluctuations", pooled)
    log = _Log(monkeypatch)
    result = _fit(LTMLE, [_regimen("fold_average")], frame, n_folds=3)
    fit = result.fits["tuned, fold average"]
    assert np.max(np.abs(_fold_scores(fit, np.asarray(log.folds.assignment)))) > CAUGHT


# ------------------------------------------------------------------ the bootstrap (T-e8)


def _replicate_tuners(log: _Log, n_bootstrap: int) -> list[np.ndarray]:
    """The node-1 tuner of each replicate fit, after the fit on the sample."""
    thetas = [theta for _, theta in log.tuners]
    assert len(thetas) == 2 * (1 + n_bootstrap)
    return [thetas[2 * (b + 1)] for b in range(n_bootstrap)]


def test_t_e8_the_full_refit_bootstrap_refits_the_tuner(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-e8 (m6): each replicate resolves theta on its own resample."""
    log = _Log(monkeypatch)
    _fit(LTMLE, [_regimen("population")], frame, n_folds=1, n_bootstrap=4, n_jobs=1)
    theta_n = log.tuners[0][1]
    replicates = _replicate_tuners(log, 4)
    assert any(not np.array_equal(theta, theta_n) for theta in replicates)


def test_m_e8_a_frozen_tuner_fails(frame: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """M-e8: theta frozen at theta_n in every replicate, and T-e8 catches it."""
    from dataclasses import replace

    fit_tuner = regimen_module.fit_tuner
    seen: list[np.ndarray] = []

    def frozen(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
        fitted = fit_tuner(declared, doses, frame, rows, weights)
        if len(seen) < 2:
            seen.append(fitted.theta)
            return fitted
        node = len(seen) % 2
        seen.append(fitted.theta)
        return replace(fitted, theta=seen[node])

    monkeypatch.setattr(regimen_module, "fit_tuner", frozen)
    log = _Log(monkeypatch)
    _fit(LTMLE, [_regimen("population")], frame, n_folds=1, n_bootstrap=4, n_jobs=1)
    theta_n = log.tuners[0][1]
    assert all(np.array_equal(theta, theta_n) for theta in _replicate_tuners(log, 4))


# ------------------------------------------------------------------ the fold average (T-e1)

#: The fixed fold maps of T-e1: fold 1's cap and fold 2's, on the unit grid.
FOLD_CAPS = (2.0, 3.0)


def _grid_branches(cap: float) -> tuple[Any, Any]:
    """``Shift(1.0, cap=cap)`` on the grid law's support indices, at both nodes."""

    def up(a: int) -> int:
        return a + 1 if a + 1 <= cap else a

    return (
        ((1.0, tuple(tuple(up(a) for a in range(4)) for _ in range(2))),),
        (
            (
                1.0,
                tuple(
                    tuple(tuple(tuple(up(a) for a in range(4)) for _ in range(2)) for _ in range(4))
                    for _ in range(2)
                ),
            ),
        ),
    )


def _grid_truth(cap: float) -> tuple[float, np.ndarray]:
    """The fold map's mean and its complex-step Gateaux derivative at every support point."""
    from .. import discrete_law_longitudinal_mtp as grid

    plan = _grid_branches(cap)
    base = np.asarray(grid.PROBS, dtype=complex)
    step = 1e-30
    curve = []
    for point in range(len(grid.SUPPORT)):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        value = grid.functional_mtp((1.0 - 1j * step) * base + 1j * step * mass, plan)
        curve.append(float(np.imag(value) / step))
    return float(np.real(grid.functional_mtp(base, plan))), np.array(curve)


def _fold_average_on_two_copies(
    monkeypatch: pytest.MonkeyPatch, caps: tuple[float, float], estimator: type = LTMLE
) -> Any:
    """The fold average on two copies of the exact grid law, each copy one fold.

    Each fold's training rows are the other copy, which realises the law, so the saturated
    learners are exact.  The fold maps are fixed at ``caps`` (T-e1 fixes them), in the order
    of the folds that hold the training copy out.
    """
    from .. import discrete_law_longitudinal as binary_law
    from .. import discrete_law_longitudinal_mtp as grid
    from .. import discrete_law_longitudinal_multivalue as multivalue
    from .. import longitudinal_mtp as mtp

    one = grid.frame()
    data = pd.concat([one, one], ignore_index=True)
    owner = np.repeat([0, 1], len(one))
    monkeypatch.setattr(
        estimator_module._SequentialEstimator,
        "_folds",
        lambda self, prepared: Folds(assignment=owner, n_folds=2),
    )
    fit_tuner = regimen_module.fit_tuner

    def fixed(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
        from dataclasses import replace

        fitted = fit_tuner(declared, doses, frame, rows, weights)
        if rows.all():
            return fitted
        # Fold k's training rows are the copy it does not hold out.
        k = 0 if rows[owner == 1].all() else 1
        return replace(fitted, theta=np.array([caps[k]]))

    monkeypatch.setattr(regimen_module, "fit_tuner", fixed)
    settings = {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "n_folds": 2,
        "g_bounds": (1e-8, 1.0),
        "simultaneous": False,
    }
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return estimator([_regimen("fold_average")], **settings).fit(
            data,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        )


def _check_fold_average(result: Any) -> None:
    from .. import discrete_law_longitudinal_mtp as grid

    name = "ey_regimen[tuned, fold average]"
    fit = result.fits["tuned, fold average"]
    rows = grid.first_row_of()
    curve = result.influence_curves[name]
    truths = []
    for k, cap in enumerate(FOLD_CAPS):
        truth, gateaux = _grid_truth(cap)
        truths.append(truth)
        assert fit.fold_estimates[k] == pytest.approx(truth, abs=1e-12)
        np.testing.assert_allclose(curve[rows + k * grid.N], gateaux, rtol=0, atol=1e-10)
    assert result.psi(name) == pytest.approx(np.mean(truths), abs=1e-12)


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e1_the_fold_average_is_the_average_of_exact_fold_functionals(
    estimator: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-e1: each fold's estimate and curve are its own map's functional and derivative."""
    result = _fold_average_on_two_copies(monkeypatch, FOLD_CAPS, estimator)
    _check_fold_average(result)
    # The nonzero witness: the fold caps are more than 0.05 apart, and so are their means.
    assert abs(FOLD_CAPS[1] - FOLD_CAPS[0]) > 0.05
    first, second = (_grid_truth(cap)[0] for cap in FOLD_CAPS)
    assert abs(first - second) > CAUGHT


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_m_e1_one_fold_map_everywhere_fails(
    estimator: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e1: fold 1's map in every fold, and T-e1 catches it."""
    result = _fold_average_on_two_copies(monkeypatch, (FOLD_CAPS[0], FOLD_CAPS[0]), estimator)
    with pytest.raises(AssertionError):
        _check_fold_average(result)
