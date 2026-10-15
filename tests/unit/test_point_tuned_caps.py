"""Data-tuned caps of a point-treatment modified treatment policy (X40 part (e)).

``X40-plan.md`` sections 2.4 and 2.5, with the construction of the longitudinal fits
(``tests/unit/test_mtp_tuned_caps.py``).  The population target adds the chain-rule term
``dPhi' IF`` to the curve: T-e4 checks the curve on the point law of
``tests/law_q_point.py`` as the pathwise derivative along exponential tilts, by
quadrature, and checks the fitted term against the same formula computed longhand.  The
fold-average target fits each fold's map on that fold's training rows and targets each
fold on its own rows: T-e1 on two copies of an exact grid law, T-e2 and T-e3 on a sample.
T-e6 pins the refusals, T-e7 the training tuner of the classifier route, and T-e8 the
bootstrap.  Each mutation control M-e* is a monkeypatch of the shipped code, and each
must move its witness by more than ``CAUGHT``.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor

import tests.discrete_law_policy_point as grid_law
from cleverly.estimators import (
    CTMLE,
    DRTMLE,
    TMLE,
)
from cleverly.estimators import _tuned_policies as tuned_module
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    DoseQuantile,
    LinearQuantileRegression,
    Scale,
    Shift,
    SupportBoundary,
)
from cleverly.interventions import tuning as tuning_module
from cleverly.learners import density_ratio as density_ratio_module
from cleverly.learners.crossfit import Folds
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from .. import law_q_point as lawp
from .. import longitudinal_mtp as mtp
from ..law_q_tuned import Tuner

tmle_module = importlib.import_module("cleverly.estimators.tmle")
CAUGHT = 1e-4
IDENTITY = 1e-6
NAME = "ey_policy[tuned]"
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W"],
    "treatment_kind": "continuous",
}
ORACLE_TUNERS = {"quantile": Tuner(), "regression": Tuner(features=("W",))}


class _Raises(BaseEstimator, RegressorMixin):
    """A learner that fails if a fit reaches it."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Raises:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, X: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("unreachable")


def _cap(kind: str) -> Any:
    if kind == "quantile":
        return DoseQuantile(0.5)
    return SupportBoundary(LinearQuantileRegression(("W",)), 0.5)


def _policies(target: str, kind: str = "quantile") -> list[Any]:
    return [Shift(1.0, cap=_cap(kind), target=target, name="tuned"), Shift(0.0, cap=None)]


def _settings(**overrides: Any) -> dict[str, Any]:
    return {
        "q_bounds": (0.0, 1.0),
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "density_bins": 20,
        "random_state": 1,
        "simultaneous": False,
        **overrides,
    }


def _fit(policies: Any, frame: pd.DataFrame, estimator: type = TMLE, **settings: Any) -> Any:
    folds = settings.pop("n_folds", 3)
    settings.setdefault("cross_fit", folds > 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = estimator(policies=policies, n_folds=max(folds, 2), **_settings(**settings))
        return model.fit(frame, **COLUMNS).single()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return lawp.sample(1500, 11)


def test_the_oracle_does_not_import_cleverly() -> None:
    """The longhand law shares no code with ``src/``."""
    source = Path(lawp.__file__).read_text(encoding="utf-8")
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(name.startswith("cleverly") for name in names)


# ------------------------------------------------------------------ the identity (T-e4 (i))


@pytest.mark.parametrize("kind", ["quantile", "regression"])
@pytest.mark.parametrize("path", ["w", "a1"])
def test_t_e4_the_point_population_curve_is_the_pathwise_derivative(kind: str, path: str) -> None:
    """T-e4 (i): ``d/de Psi(P_e) = E[D_Psi s]`` along tilts of the W law and the dose law."""
    law = lawp.PointLawQ(tuner=ORACLE_TUNERS[kind])
    assert lawp.pathwise_derivative(law, path) == pytest.approx(
        law.score_pairing(path), abs=IDENTITY
    )


@pytest.mark.parametrize("kind", ["quantile", "regression"])
def test_m_e4_the_point_identity_fails_without_the_tuning_term(kind: str) -> None:
    """M-e4 on the oracle: the curve without the tuner's term misses the derivative."""
    law = lawp.PointLawQ(tuner=ORACLE_TUNERS[kind])
    gaps = [
        abs(lawp.pathwise_derivative(law, path) - law.score_pairing(path, tuning=False))
        for path in ("w", "a1")
    ]
    assert max(gaps) > CAUGHT


def test_t_e4_the_point_law_carries_a_large_tuning_term() -> None:
    """The nonzero requirement: the term is a material share of the curve's spread."""
    for kind in ORACLE_TUNERS:
        term, phi = lawp.PointLawQ(tuner=ORACLE_TUNERS[kind]).spreads()
        assert term > 0.1 * phi


# ------------------------------------------------------------------ the fitted term (T-e4 (ii))


def _silverman(values: np.ndarray, power: float = 0.2) -> float:
    sd = float(np.std(values))
    iqr = float(
        np.quantile(values, 0.75, method="inverted_cdf")
        - np.quantile(values, 0.25, method="inverted_cdf")
    )
    return 0.9 * min(sd, iqr / 1.34) * values.size ** (-power)


def _longhand_term(
    frame: pd.DataFrame,
    nuisance: Any,
    kind: str,
    *,
    q: float = 0.5,
    factor: float | None = None,
) -> np.ndarray:
    """``dPhi_n' IF`` written out: the tuner, its curve, and the kernel derivative.

    A ``Shift(1.0)`` has the edge ``c - 1`` and ``|e'| = 1``; a ``Scale(f)`` has the edge
    ``c / f`` and ``|e'| = 1 / f``, both written here.
    """
    dose = frame["A"].to_numpy()
    n = dose.size
    if kind == "quantile":
        theta = np.array([np.quantile(dose, q, method="inverted_cdf")])
        h = _silverman(dose)
        density = float(np.mean(stats.norm.pdf((theta[0] - dose) / h)) / h)
        influence = (-((dose <= theta[0]).astype(float) - q) / density)[:, None]
        design = np.ones((n, 1))
    else:
        features = frame[["W"]].to_numpy()
        model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(features, dose)
        theta = np.concatenate([[model.intercept_], model.coef_])
        design = np.column_stack([np.ones(n), features])
        residual = dose - design @ theta
        hj = _silverman(residual, power=1.0 / 3.0)
        inside = (np.abs(residual) <= hj).astype(float)
        jacobian = (design * inside[:, None]).T @ design / (2.0 * hj * n)
        score = design * (q - (residual <= 0.0).astype(float))[:, None]
        influence = np.linalg.solve(jacobian, score.T).T
    boundary = design @ theta
    edge = boundary - 1.0 if factor is None else boundary / factor
    slope = 1.0 if factor is None else 1.0 / factor
    gap = dose - edge
    h = _silverman(gap)
    kernel = stats.norm.pdf(gap / h) / h
    high = nuisance.outcome_probes["tuned0:boundary"]
    low = nuisance.outcome_probes["tuned0:edge"]
    np.testing.assert_allclose(nuisance.policies.curve_terms[:, 1], 0.0, atol=0)
    derivative = slope * (kernel * (high - low)) @ design / n
    return np.asarray(influence @ derivative)


@pytest.mark.parametrize(
    ("folds", "kind", "ratio"),
    [
        (1, "quantile", "density"),
        (3, "quantile", "density"),
        (3, "regression", "density"),
        (3, "quantile", "classifier"),
        (1, "regression", "classifier"),
    ],
)
def test_t_e4_the_reported_point_curve_adds_the_tuning_term_longhand(
    folds: int, kind: str, ratio: str, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-e4 (ii) and M-e4 in the library: the curve is the curve at theta_n plus the term."""
    kept = _fit(_policies("population", kind), frame, n_folds=folds, ratio=ratio)
    term = _longhand_term(frame, kept.nuisance, kind)
    np.testing.assert_allclose(
        kept.nuisance.policies.curve_terms[:, 0], term, rtol=1e-8, atol=1e-10
    )
    resolve = tuned_module.resolve_point_policies
    monkeypatch.setattr(
        tmle_module,
        "resolve_point_policies",
        lambda policies, data, folds: replace(resolve(policies, data, folds), tunings=()),
    )
    dropped = _fit(_policies("population", kind), frame, n_folds=folds, ratio=ratio)
    assert dropped.nuisance.policies.curve_terms is None
    assert dropped.psi(NAME) == pytest.approx(kept.psi(NAME), abs=1e-14)
    gap = kept.influence_curves[NAME] - dropped.influence_curves[NAME]
    np.testing.assert_allclose(gap, term, rtol=0, atol=1e-12)
    assert np.max(np.abs(gap)) > CAUGHT
    assert np.std(term) > 0.1 * np.std(dropped.influence_curves[NAME])


def test_the_population_estimate_is_the_fit_at_theta_n(frame: Any) -> None:
    """The point estimate needs no new targeting: it is the fit with the cap at theta_n."""
    tuned = _fit(_policies("population"), frame, n_folds=3)
    theta = float(np.quantile(frame["A"], 0.5, method="inverted_cdf"))
    declared = _fit([Shift(1.0, cap=theta, name="tuned"), Shift(0.0, cap=None)], frame, n_folds=3)
    assert tuned.psi(NAME) == pytest.approx(declared.psi(NAME), abs=1e-14)
    gap = tuned.influence_curves[NAME] - declared.influence_curves[NAME]
    np.testing.assert_allclose(gap, tuned.nuisance.policies.curve_terms[:, 0], atol=1e-12)


# ------------------------------------------------------------------ instrumented tuners


class _Log:
    """What a fit hands its tuner and its classifier, recorded by monkeypatch."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tuners: list[tuple[np.ndarray, np.ndarray]] = []
        self.folds: Any = None
        self.stacked: list[dict[str, Any]] = []
        fit_tuner = tuned_module.fit_tuner

        def tuner(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
            fitted = fit_tuner(declared, doses, frame, rows, weights)
            self.tuners.append((np.array(rows, dtype=bool), np.array(fitted.theta)))
            return fitted

        monkeypatch.setattr(tuned_module, "fit_tuner", tuner)
        resolve = tuned_module.resolve_point_policies

        def resolved(policies: Any, data: Any, folds: Any) -> Any:
            self.folds = folds
            return resolve(policies, data, folds)

        monkeypatch.setattr(tmle_module, "resolve_point_policies", resolved)
        classifier = density_ratio_module.classifier_ratio

        def recorded(*args: Any, **kwargs: Any) -> Any:
            self.stacked.append(
                {
                    "assigned": args[3],
                    "by_fold": kwargs.get("assigned_by_fold"),
                    "at": kwargs.get("evaluate_at"),
                }
            )
            return classifier(*args, **kwargs)

        monkeypatch.setattr(density_ratio_module, "classifier_ratio", recorded)

    def trains(self) -> list[np.ndarray]:
        out = []
        for train, _ in self.folds:
            rows = np.zeros(len(self.folds.assignment), dtype=bool)
            rows[train] = True
            out.append(rows)
        return out


def _median(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.5, method="inverted_cdf"))


def _up(dose: np.ndarray, cap: float) -> np.ndarray:
    return np.where(dose + 1.0 <= cap, dose + 1.0, dose)


def _check_fold_maps(log: _Log, result: Any, frame: pd.DataFrame) -> None:
    """T-e2: fold k's map is the tuner fitted on fold k's training rows alone."""
    dose = frame["A"].to_numpy()
    trains = log.trains()
    policies = result.nuisance.policies
    owner = np.asarray(log.folds.assignment)
    assert policies.component_rows is not None
    for k, train in enumerate(trains):
        component = k
        np.testing.assert_array_equal(policies.component_rows[:, component], owner == k)
        np.testing.assert_array_equal(
            policies.shifted[:, component], _up(dose, _median(dose[train]))
        )
    fold_fits = [rows for rows, _ in log.tuners if not rows.all()]
    assert len(fold_fits) >= len(trains)
    for rows in fold_fits:
        assert any(np.array_equal(rows, train) for train in trains)


def test_t_e2_each_point_fold_map_reads_its_training_rows(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log(monkeypatch)
    result = _fit(_policies("fold_average"), frame, n_folds=3)
    _check_fold_maps(log, result, frame)
    assert "ey_policy[tuned, fold average]" in result.estimates
    assert len({_median(frame["A"].to_numpy()[train]) for train in log.trains()}) > 1


def test_m_e2_a_full_sample_point_fold_map_fails(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e2: every fold's map is the full-sample tuner, and T-e2 catches it."""
    fit_tuner = tuned_module.fit_tuner
    monkeypatch.setattr(
        tuned_module,
        "fit_tuner",
        lambda declared, doses, frame, rows, weights: fit_tuner(
            declared, doses, frame, np.ones_like(rows, dtype=bool), weights
        ),
    )
    log = _Log(monkeypatch)
    result = _fit(_policies("fold_average"), frame, n_folds=3)
    with pytest.raises(AssertionError):
        _check_fold_maps(log, result, frame)


def _fold_scores(result: Any, frame: pd.DataFrame) -> np.ndarray:
    """Each fold component's score on its own rows: the mean of h_k (Y - Q*(A))."""
    policies = result.nuisance.policies
    owner = np.asarray(result.nuisance.folds.assignment)
    targeted = result.fluctuations["mtp"].targeted
    residual = frame["Y"].to_numpy() - targeted.observed
    return np.array(
        [
            float(np.mean((policies.ratio[:, k] * residual)[owner == k]))
            for k in range(result.nuisance.folds.n_folds)
        ]
    )


def test_t_e3_each_point_fold_solves_its_own_score(frame: Any) -> None:
    """T-e3: one fluctuation coefficient per fold, each solving its own held-out score."""
    result = _fit(_policies("fold_average"), frame, n_folds=3)
    assert np.max(np.abs(_fold_scores(result, frame))) < 1e-10
    epsilon = np.asarray(result.fluctuations["mtp"].epsilon, dtype=float)
    assert len(set(np.round(epsilon[:3], 14))) == 3


def test_m_e3_a_pooled_point_fluctuation_leaves_a_fold_score(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e3: the fold components read every row's ratio, which pools their fluctuation."""
    finish = tuned_module.PointTunedPolicies.finish

    def pooled(self: Any, evaluated: Any, *args: Any) -> Any:
        out = finish(self, evaluated, *args)
        return replace(out, ratio=evaluated.ratio, ratio_at=evaluated.ratio_at)

    monkeypatch.setattr(tuned_module.PointTunedPolicies, "finish", pooled)
    result = _fit(_policies("fold_average"), frame, n_folds=3)
    owner = np.asarray(result.nuisance.folds.assignment)
    targeted = result.fluctuations["mtp"].targeted
    residual = frame["Y"].to_numpy() - targeted.observed
    ratio = result.nuisance.policies.ratio
    scores = [float(np.mean((ratio[:, k] * residual)[owner == k])) for k in range(3)]
    assert max(abs(score) for score in scores) > CAUGHT


# ------------------------------------------------------------------ the fold average (T-e1)

#: The fixed fold maps of T-e1: fold 1's cap and fold 2's, on the unit grid.
FOLD_CAPS = (2.0, 3.0)


def _grid_truth(cap: float) -> tuple[float, np.ndarray]:
    """The fold map's mean, and its complex-step Gateaux derivative at each support point."""
    grid = grid_law.UNIT
    branches = grid_law._deterministic(grid, lambda w, a: a + 1.0 if a + 1.0 <= cap else a)

    def functional(probs: Any) -> Any:
        joint = probs[:, :, 0] + probs[:, :, 1]
        return grid_law._mean_under(joint, probs[:, :, 1] / joint, branches)

    base = grid.probs.astype(complex)
    step = 1e-30
    curve = []
    for point in grid.support:
        mass = np.zeros_like(base)
        mass[point] = 1.0
        curve.append(float(np.imag(functional((1.0 - 1j * step) * base + 1j * step * mass)) / step))
    return float(np.real(functional(base))), np.array(curve)


def _fold_average_on_two_copies(monkeypatch: pytest.MonkeyPatch, caps: tuple[float, float]) -> Any:
    """The fold average on two copies of the exact unit grid law, each copy one fold."""
    one = grid_law.UNIT.frame()
    data = pd.concat([one, one], ignore_index=True)
    owner = np.repeat([0, 1], len(one))
    monkeypatch.setattr(
        TMLE, "_folds", lambda self, prepared, seed=None: Folds(assignment=owner, n_folds=2)
    )
    fit_tuner = tuned_module.fit_tuner

    def fixed(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
        fitted = fit_tuner(declared, doses, frame, rows, weights)
        if rows.all():
            return fitted
        k = 0 if rows[owner == 1].all() else 1
        return replace(fitted, theta=np.array([caps[k]]))

    monkeypatch.setattr(tuned_module, "fit_tuner", fixed)
    settings = {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "n_folds": 2,
        "g_bounds": (1e-8, 1.0 - 1e-8),
        "simultaneous": False,
        "policies": [Shift(1.0, cap=DoseQuantile(0.5), target="fold_average", name="tuned")],
    }
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return TMLE(**settings).fit(data, **COLUMNS).single()


def _check_fold_average(result: Any) -> None:
    grid = grid_law.UNIT
    name = "ey_policy[tuned, fold average]"
    rows = grid.first_row_of()
    present = grid.counts.reshape(-1) > 0
    curve = result.influence_curves[name]
    truths = []
    for k, cap in enumerate(FOLD_CAPS):
        truth, gateaux = _grid_truth(cap)
        truths.append(truth)
        held = slice(k * grid_law.N, (k + 1) * grid_law.N)
        fold_rows = np.asarray(result.nuisance.policies.component_rows[:, k])[held]
        assert fold_rows.all()
        np.testing.assert_allclose(
            curve[rows[present] + k * grid_law.N], gateaux[present], rtol=0, atol=1e-10
        )
    assert result.psi(name) == pytest.approx(np.mean(truths), abs=1e-12)


@pytest.mark.usefixtures("exact_grid_law")
def test_t_e1_the_point_fold_average_is_the_average_of_exact_fold_functionals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-e1: each fold's curve is its own map's Gateaux derivative; the estimate their mean."""
    result = _fold_average_on_two_copies(monkeypatch, FOLD_CAPS)
    _check_fold_average(result)
    assert abs(FOLD_CAPS[1] - FOLD_CAPS[0]) > 0.05
    first, second = (_grid_truth(cap)[0] for cap in FOLD_CAPS)
    assert abs(first - second) > CAUGHT


@pytest.mark.usefixtures("exact_grid_law")
def test_m_e1_one_point_fold_map_everywhere_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-e1: fold 1's map in both folds, and T-e1 catches it."""
    result = _fold_average_on_two_copies(monkeypatch, (FOLD_CAPS[0], FOLD_CAPS[0]))
    with pytest.raises(AssertionError):
        _check_fold_average(result)


# ------------------------------------------------------------------ the training tuner (T-e7)


def _check_classifier_folds(log: _Log, frame: pd.DataFrame) -> None:
    """T-e7 (N3): fold k's classifier stacks its own map; the ratio reads the map at theta_n."""
    dose = frame["A"].to_numpy()
    full = _up(dose, _median(dose))
    tuned = [entry for entry in log.stacked if np.array_equal(entry["assigned"][0][1], full)]
    assert len(tuned) == 1
    (entry,) = tuned
    assert entry["by_fold"] is not None
    trains = log.trains()
    assert len(entry["by_fold"]) == len(trains)
    for k, assigned in enumerate(entry["by_fold"]):
        ((probability, target),) = assigned
        assert probability == 1.0
        np.testing.assert_array_equal(target, _up(dose, _median(dose[trains[k]])))
    assert any(np.array_equal(at, full) for at in entry["at"])


def test_t_e7_each_point_fold_classifier_stacks_its_own_tuner(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log(monkeypatch)
    _fit(_policies("population"), frame, n_folds=3, ratio="classifier")
    _check_classifier_folds(log, frame)
    dose = frame["A"].to_numpy()
    assert len({_median(dose[train]) for train in log.trains()}) > 1


def test_m_e7_a_point_classifier_at_theta_n_fails(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e7: every fold's classifier stacks the map at theta_n, and T-e7 catches it."""
    resolve = tuned_module.resolve_point_policies
    log = _Log(monkeypatch)
    monkeypatch.setattr(
        tmle_module,
        "resolve_point_policies",
        lambda policies, data, folds: replace(resolve(policies, data, folds), by_fold=None),
    )
    log.folds = None
    result = _fit(_policies("population"), frame, n_folds=3, ratio="classifier")
    log.folds = result.nuisance.folds
    with pytest.raises(AssertionError):
        _check_classifier_folds(log, frame)


def test_the_density_route_reads_the_map_at_theta_n(frame: Any) -> None:
    """N3, density route: every fold's density is evaluated under the map at theta_n."""
    result = _fit(_policies("population"), frame, n_folds=3)
    dose = frame["A"].to_numpy()
    np.testing.assert_array_equal(result.nuisance.policies.shifted[:, 0], _up(dose, _median(dose)))
    assert result.nuisance.policies.component_rows is None


# ------------------------------------------------------------------ the bootstrap (T-e8)


def test_t_e8_the_point_bootstrap_refits_the_tuner(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-e8 (m6): each replicate resolves theta on its own resample."""
    log = _Log(monkeypatch)
    small = frame.iloc[:600].reset_index(drop=True)
    _fit(_policies("population"), small, n_folds=1, n_bootstrap=4, n_jobs=1)
    full = [theta for rows, theta in log.tuners if rows.all()]
    theta_n = full[0]
    assert len(full) >= 1 + 4
    assert any(not np.array_equal(theta, theta_n) for theta in full[-4:])


def test_m_e8_a_frozen_point_tuner_fails(frame: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """M-e8: theta frozen at theta_n in every replicate, and T-e8 catches it."""
    fit_tuner = tuned_module.fit_tuner
    seen: list[np.ndarray] = []

    def frozen(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
        fitted = fit_tuner(declared, doses, frame, rows, weights)
        if not seen:
            seen.append(fitted.theta)
        return replace(fitted, theta=seen[0])

    monkeypatch.setattr(tuned_module, "fit_tuner", frozen)
    log = _Log(monkeypatch)
    small = frame.iloc[:600].reset_index(drop=True)
    _fit(_policies("population"), small, n_folds=1, n_bootstrap=4, n_jobs=1)
    full = [theta for rows, theta in log.tuners if rows.all()]
    assert all(np.array_equal(theta, full[0]) for theta in full)


# ------------------------------------------------------------------ refusals (T-e6)


def _raising(**overrides: Any) -> dict[str, Any]:
    return {"outcome_learner": _Raises(), "treatment_learner": _Raises(), **overrides}


def _boosted() -> SupportBoundary:
    return SupportBoundary(
        GradientBoostingRegressor(loss="quantile", alpha=0.5), 0.5, features=("W",)
    )


def test_t_e6_r_e1_a_point_tuned_map_declares_its_target() -> None:
    with pytest.raises(CapabilityError, match="which parameter it estimates is a declaration"):
        TMLE(policies=[Shift(1.0, cap=DoseQuantile(0.5))])


def test_t_e6_r_e2_the_point_population_target_needs_a_finite_dimensional_boundary() -> None:
    with pytest.raises(CapabilityError, match="has no declared finite-dimensional parameter"):
        TMLE(policies=[Shift(1.0, cap=_boosted(), target="population")])


def test_t_e6_r_e2_does_not_refuse_a_point_boundary_learner_on_the_fold_average(
    frame: Any,
) -> None:
    result = _fit([Shift(1.0, cap=_boosted(), target="fold_average", name="tuned")], frame)
    assert "ey_policy[tuned, fold average]" in result.estimates


def test_t_e6_r_e3_a_point_tuner_does_not_read_the_outcome(frame: Any) -> None:
    cap = SupportBoundary(LinearQuantileRegression(("Y",)), 0.5)
    with pytest.raises(CapabilityError, match="tunes its map with the outcome"):
        _fit([Shift(1.0, cap=cap, target="population", name="tuned")], frame, **_raising())


def test_t_e6_r_e4_the_point_fold_average_needs_two_folds(frame: Any) -> None:
    with pytest.raises(CapabilityError, match="needs n_folds of at least 2"):
        _fit(_policies("fold_average"), frame, n_folds=1, **_raising())


COMPOSITIONS = [
    ({"cv_evaluation": True}, "cv_evaluation=True"),
    ({"targeting_scheme": "fold"}, "targeting_scheme='fold'"),
    ({"n_bootstrap": 4}, "the bootstrap on the fold-average target"),
    ({"repeats": 2}, "repeated splits on the fold-average target"),
]


@pytest.mark.parametrize(("settings", "named"), COMPOSITIONS)
def test_point_tuned_compositions_are_refused_before_any_learner(
    settings: dict[str, Any], named: str, frame: Any
) -> None:
    with pytest.raises(CapabilityError, match=f"is not combined with {named}"):
        _fit(_policies("fold_average"), frame, **_raising(**settings))


@pytest.mark.parametrize(
    ("estimator", "named"),
    [
        (DRTMLE, "DRTMLE and policies= are not combined"),
        (CTMLE, "CTMLE and policies= are not combined"),
    ],
)
def test_point_tuned_caps_refuse_a_variant(estimator: type, named: str, frame: Any) -> None:
    """The variants refuse a policy fit by their own text, before any learner."""
    with pytest.raises((CapabilityError, DataError), match=named):
        _fit(_policies("population"), frame, estimator=estimator, **_raising())


def test_point_tuned_caps_refuse_strata(frame: Any) -> None:
    data = frame.assign(S=(frame["W"] > 0).astype(int))
    with (
        pytest.raises(CapabilityError, match="is not combined with baseline strata"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        TMLE(policies=_policies("population"), **_settings(**_raising())).fit(
            data, strata=["S"], **{**COLUMNS, "covariates": ["W", "S"]}
        )


def test_point_tuned_caps_refuse_a_missing_outcome(frame: Any) -> None:
    data = frame.assign(D=(np.arange(len(frame)) % 7 != 0).astype(int))
    with (
        pytest.raises(CapabilityError, match="is not combined with a missing treatment or"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        TMLE(policies=_policies("population"), **_settings(**_raising())).fit(
            data, delta="D", **COLUMNS
        )


def test_the_point_kernel_bandwidth_is_the_shared_rule() -> None:
    """The point fit reads the one bandwidth rule of the tuning module."""
    assert tuned_module.cap_derivative is tuning_module.cap_derivative
    assert tuned_module.tuner_influence is tuning_module.tuner_influence


def test_point_tuned_caps_refuse_a_categorical_treatment(frame: Any) -> None:
    data = frame.assign(A=np.round(frame["A"]).clip(-1, 1))
    settings = {**COLUMNS, "treatment_kind": "discrete"}
    with pytest.raises(CapabilityError, match="is not one continuous dose"):
        TMLE(policies=_policies("population"), **_settings(**_raising())).fit(data, **settings)


# ------------------------------------------------------------------ the tuned Scale (M1)

FACTOR = 0.5
SCALE_Q = 0.3
SCALE_TUNERS = {
    "quantile": Tuner(q=SCALE_Q),
    "regression": Tuner(features=("W",), q=SCALE_Q),
}


def _scale_policies(kind: str) -> list[Any]:
    if kind == "quantile":
        cap: Any = DoseQuantile(SCALE_Q)
    else:
        cap = SupportBoundary(LinearQuantileRegression(("W",)), SCALE_Q)
    return [Scale(FACTOR, cap=cap, target="population", name="tuned"), Shift(0.0, cap=None)]


@pytest.fixture
def unit_slope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation of the review: ``|e'| = 1`` on every tuned map."""
    original = tuned_module.tuned_cap_parts

    def forced(policy: Any, fitted: Any, frame: Any, n: int) -> Any:
        boundary, edge, gradient, _ = original(policy, fitted, frame, n)
        return boundary, edge, gradient, 1.0

    monkeypatch.setattr(tuned_module, "tuned_cap_parts", forced)


@pytest.mark.parametrize("path", ["w", "a1"])
@pytest.mark.parametrize("kind", sorted(SCALE_TUNERS))
def test_t_e4_the_point_scale_curve_is_the_pathwise_derivative(kind: str, path: str) -> None:
    """T-e4 (i) for ``Scale(0.5)`` on the point law, with the slope ``1/f`` in the oracle."""
    law = lawp.PointLawQ(tuner=SCALE_TUNERS[kind], factor=FACTOR)
    assert abs(lawp.pathwise_derivative(law, path) - law.score_pairing(path)) < IDENTITY
    assert abs(lawp.pathwise_derivative(law, "a1") - law.score_pairing("a1", tuning=False)) > CAUGHT
    term, phi = law.spreads()
    assert term / phi > 0.1


@pytest.mark.parametrize(
    ("folds", "kind", "ratio"),
    [(1, "quantile", "density"), (3, "regression", "density"), (3, "quantile", "classifier")],
)
def test_t_e4_the_point_scale_curve_adds_the_tuning_term_longhand(
    folds: int, kind: str, ratio: str, frame: Any
) -> None:
    """T-e4 (ii) for ``Scale``: the longhand computes its own edge ``c / f`` and slope ``1/f``."""
    fit = _fit(_scale_policies(kind), frame, n_folds=folds, ratio=ratio)
    term = _longhand_term(frame, fit.nuisance, kind, q=SCALE_Q, factor=FACTOR)
    np.testing.assert_allclose(fit.nuisance.policies.curve_terms[:, 0], term, rtol=1e-8, atol=1e-10)
    assert np.std(term) > 0.05 * np.std(fit.influence_curves[NAME])


@pytest.mark.parametrize("kind", ["quantile", "regression"])
def test_m_e4_a_unit_slope_moves_the_point_scale_curve(
    kind: str, frame: Any, unit_slope: None
) -> None:
    """M1: with ``|e'| = 1`` the reported term misses the longhand term by half of it."""
    fit = _fit(_scale_policies(kind), frame, n_folds=1)
    term = _longhand_term(frame, fit.nuisance, kind, q=SCALE_Q, factor=FACTOR)
    gap = float(np.max(np.abs(fit.nuisance.policies.curve_terms[:, 0] - term)))
    assert gap > 0.3 * float(np.max(np.abs(term)))


def test_the_point_scale_term_has_the_oracle_spread() -> None:
    """The library term has the oracle's SD (T-e4 (i) at the library), flexible regressions."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    big = lawp.sample(8_000, 11)
    fit = _fit(
        _scale_policies("regression"),
        big,
        n_folds=1,
        outcome_learner=HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=0
        ),
    )
    oracle, _ = lawp.PointLawQ(tuner=SCALE_TUNERS["regression"], factor=FACTOR).spreads()
    spread = float(np.sqrt(np.mean(fit.nuisance.policies.curve_terms[:, 0] ** 2)))
    assert abs(spread / oracle - 1.0) < 0.25


# ------------------------------------------------------------------ declared atoms (M3)


def test_m3_a_point_tuned_cap_with_declared_atoms_is_refused_before_any_learner() -> None:
    """M3: the median sits on a 41% atom; the text names the missing derivation."""
    rng = np.random.default_rng(5)
    n = 1500
    w = rng.normal(size=n)
    atom = rng.uniform(size=n) < 0.41
    dose = np.where(atom, 3.0, 3.0 + 0.5 * w + rng.normal(size=n))
    held = pd.DataFrame({"W": w, "A": dose, "Y": rng.uniform(size=n)})
    for target in ("population", "fold_average"):
        policies = [
            Shift(0.5, cap=DoseQuantile(0.5), target=target, name="tuned"),
            Shift(0.0, cap=None),
        ]
        with pytest.raises(CapabilityError) as caught, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TMLE(policies=policies, n_folds=2, **_settings(**_raising())).fit(
                held, dose_atoms=(3.0,), **COLUMNS
            )
        assert str(caught.value) == (
            "a data-tuned cap is not combined with declared dose atoms: the treatment declares "
            "the atom(s) 3. The cap is a quantile of the dose law. Where that law has an atom, "
            "the quantile does not move with the law while its level stays inside the atom's "
            "jump, and it is not pathwise differentiable at the jump's edge. The influence "
            "function of the tuner and the derivative of the mean in the cap both need a "
            "density of the dose at the cap, and this release does not derive them at an atom. "
            "Declare the cap as a number, or tune the cap at a node with no declared atoms"
        )
