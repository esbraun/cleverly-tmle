"""A data-tuned cap at a dose component of a ``JointPolicy`` (X40 review M5).

The population target :math:`\\Psi(P) = \\Phi(P, \\theta(P))` has the curve
:math:`D_\\Phi + \\sum_t \\partial_t\\Phi' \\mathrm{IF}_t` at a mixed node too.  The tuner is
the Z-estimator of the component's natural dose on the frame its map reads, and the
derivative is the Leibniz term with the outcome regression read at the full assigned vector,
the tuned component set to the boundary and to the last moved dose.

- T-e4 (i): on law Q with a flag at node 2 (``tests/law_q_vector.py``), the curve is the
  pathwise derivative along five tilts, by quadrature.  M-e4: dropping the tuning term, or
  reading the natural flag in the derivative, breaks the identity.
- T-e4 (ii): the fitted curve adds the tuning term computed longhand, for a ``Shift`` and a
  ``Scale``; the probes are the assigned vector with the component set (M-e4 in the library).
- T-e7: each outer fold trains at its own tuner on both routes.
- T-e1: the fold average at two mixed nodes of the exact grid law is the average of the
  exact fold functionals; M-e1 holds one fold map everywhere.
- The refusals: declared atoms (M3), and a tuner that reads the tuned dose.
  ``tests/unit/test_mtp_joint_tuning.py`` covers two tuned doses and a randomizer, and
  ``tests/unit/test_point_tuned_components.py`` the point vector.
"""

from __future__ import annotations

import ast
import importlib
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor

from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    DoseQuantile,
    JointPolicy,
    LinearQuantileRegression,
    ModifiedPolicy,
    Scale,
    Shift,
    SupportBoundary,
)
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, sequential
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import regimen as regimen_module

from .. import law_q_vector as qv
from ..law_q_tuned import Tuner

CAUGHT = 1e-4
IDENTITY = 1e-6
FACTOR = 0.5
SCALE_Q = 0.3
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", ["F2", "A2"]],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
NODE2 = ("W", "L2", "A1", "F2")
QUANTILE = (Tuner(), Tuner())
REGRESSION = (Tuner(features=("W",)), Tuner(features=NODE2))
ORACLE_TUNERS = {"quantile": QUANTILE, "regression": REGRESSION}


def set_flag(a: Any, h: Any) -> np.ndarray:
    return np.ones(len(a))


FLAG = ModifiedPolicy("flag", apply=set_flag, policy_kind="known")


def _caps(kind: str, q: float = 0.5) -> tuple[Any, Any]:
    if kind == "quantile":
        return DoseQuantile(q), DoseQuantile(q)
    return (
        SupportBoundary(LinearQuantileRegression(["W"]), q),
        SupportBoundary(LinearQuantileRegression(list(NODE2)), q),
    )


def _regimen(
    target: str = "population", kind: str = "quantile", *, factor: float | None = None
) -> DynamicRegimen:
    """``Shift(1.0)`` (or ``Scale(factor)``) at node 1, and the flag set with the dose tuned."""
    q = 0.5 if factor is None else SCALE_Q
    cap1, cap2 = _caps(kind, q)

    def tuned(cap: Any) -> Any:
        if factor is None:
            return Shift(1.0, cap=cap, target=target)
        return Scale(factor, cap=cap, target=target)

    node2 = JointPolicy(
        "flag, tuned dose", components={"F2": FLAG, "A2": tuned(cap2)}, policy_kind="known"
    )
    return DynamicRegimen("tuned", (tuned(cap1), node2))


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
    return qv.sample(1500, 11)


# ------------------------------------------------------------------ the oracle (T-e4 (i))


def test_the_oracle_does_not_import_cleverly() -> None:
    tree = ast.parse(Path(qv.__file__).read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [name for name in imported if name.split(".")[0] == "cleverly"]


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_the_oracle_derivative_is_the_derivative_of_the_mean_in_theta(kind: str) -> None:
    """The analytic Leibniz term at the mixed node equals a finite difference of Phi."""
    law = qv.LawQV(tuners=ORACLE_TUNERS[kind])
    theta = law.theta
    for node in (1, 2):
        numeric = []
        for j in range(theta[node - 1].size):

            def at(step: float, node: int = node, j: int = j) -> float:
                moved = [value.copy() for value in theta]
                moved[node - 1][j] += step
                return law.psi((moved[0], moved[1]))

            h = 1e-4
            numeric.append((8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h))
        np.testing.assert_allclose(law.derivative(node), numeric, rtol=0, atol=1e-6)


@pytest.mark.parametrize("path", ["w", "a1", "l2", "f2", "a2"])
@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_t_e4_the_component_curve_is_the_pathwise_derivative(kind: str, path: str) -> None:
    """T-e4 (i): d/de Psi(P_e) equals E[D_Psi s] along each tilt, the flag's included."""
    law = qv.LawQV(tuners=ORACLE_TUNERS[kind])
    assert abs(qv.pathwise_derivative(law, path) - law.score_pairing(path)) < IDENTITY


@pytest.mark.parametrize("path", ["a1", "a2"])
def test_t_e4_the_scale_component_curve_is_the_pathwise_derivative(path: str) -> None:
    tuners = (Tuner(q=SCALE_Q, features=("W",)), Tuner(q=SCALE_Q, features=NODE2))
    law = qv.LawQV(tuners=tuners, factor=FACTOR)
    assert abs(qv.pathwise_derivative(law, path) - law.score_pairing(path)) < IDENTITY


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_m_e4_the_component_identity_fails_without_the_term_or_the_assigned_flag(
    kind: str,
) -> None:
    """M-e4: no tuning term, or a derivative at the natural flag, breaks the identity."""
    law = qv.LawQV(tuners=ORACLE_TUNERS[kind])
    natural = qv.LawQV(tuners=ORACLE_TUNERS[kind], assigned_flag=False)
    for path in ("a1", "a2"):
        pathwise = qv.pathwise_derivative(law, path)
        assert abs(pathwise - law.score_pairing(path, tuning=False)) > CAUGHT
    assert abs(qv.pathwise_derivative(law, "a2") - natural.score_pairing("a2")) > 1e-3


# ------------------------------------------------------------------ the fitted curve (T-e4 (ii))


def _silverman(values: np.ndarray, power: float = 0.2) -> float:
    sd = float(np.std(values))
    iqr = float(
        np.quantile(values, 0.75, method="inverted_cdf")
        - np.quantile(values, 0.25, method="inverted_cdf")
    )
    return 0.9 * min(sd, iqr / 1.34) * values.size ** (-power)


def _longhand_tuner(frame: pd.DataFrame, node: int, kind: str, q: float) -> tuple[Any, ...]:
    """theta_n, the influence function, the boundary and the gradient, written out.

    At node 2 the tuner is the dose component's quantile, or its linear quantile regression
    on ``W, L2, A1`` and the natural flag ``F2``.
    """
    dose = frame[f"A{node}"].to_numpy()
    n = dose.size
    if kind == "quantile":
        theta = np.array([np.quantile(dose, q, method="inverted_cdf")])
        h = _silverman(dose)
        density = float(np.mean(stats.norm.pdf((theta[0] - dose) / h)) / h)
        influence = (-((dose <= theta[0]).astype(float) - q) / density)[:, None]
        return theta, influence, np.full(n, theta[0]), np.ones((n, 1))
    names = ["W"] if node == 1 else list(NODE2)
    features = frame[names].to_numpy()
    model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(features, dose)
    theta = np.concatenate([[model.intercept_], model.coef_])
    design = np.column_stack([np.ones(n), features])
    residual = dose - design @ theta
    h = _silverman(residual, power=1.0 / 3.0)
    inside = (np.abs(residual) <= h).astype(float)
    jacobian = (design * inside[:, None]).T @ design / (2.0 * h * n)
    score = design * (q - (residual <= 0.0).astype(float))[:, None]
    return theta, np.linalg.solve(jacobian, score.T).T, design @ theta, design


def _longhand_term(
    frame: pd.DataFrame, fit: Any, kind: str, *, factor: float | None = None
) -> np.ndarray:
    """sum_t dPhi_n' IF_t, with node 2's kernel read at the component's natural dose."""
    q = 0.5 if factor is None else SCALE_Q
    term = np.zeros(len(frame))
    by_time = {step.time: step for step in fit.steps}
    for node in (1, 2):
        _, influence, boundary, gradient = _longhand_tuner(frame, node, kind, q)
        dose = frame[f"A{node}"].to_numpy()
        edge = boundary - 1.0 if factor is None else boundary / factor
        slope = 1.0 if factor is None else 1.0 / factor
        step = by_time[node]
        prefix = np.ones(len(frame)) if node == 1 else by_time[node - 1].clever
        gap = dose - edge
        kernel = stats.norm.pdf(gap / _silverman(gap)) / _silverman(gap)
        derivative = slope * (prefix * kernel * (step.probes[:, 0] - step.probes[:, 1])) @ gradient
        term = term + influence @ (derivative / len(frame))
    return term


def _off_the_fit(frame: pd.DataFrame, kind: str, q: float = 0.5) -> np.ndarray:
    """The rows off a linear quantile fit's hyperplane, at both nodes.

    A linear-programming fit interpolates ``p`` rows, where the indicator of the score sits
    at its jump: two solvers that agree on theta to rounding can put such a row on either
    side.  Every other row is compared.
    """
    keep = np.ones(len(frame), dtype=bool)
    if kind == "quantile":
        return keep
    for node in (1, 2):
        _, _, boundary, gradient = _longhand_tuner(frame, node, kind, q)
        on = np.abs(frame[f"A{node}"].to_numpy() - boundary) < 1e-9
        assert on.sum() <= gradient.shape[1]
        keep &= ~on
    return keep


@pytest.mark.parametrize(
    ("estimator", "folds", "kind", "ratio"),
    [
        (LTMLE, 1, "quantile", "density"),
        (LTMLE, 3, "regression", "classifier"),
        (LTMLE, 3, "quantile", "density"),
        (SDR, 1, "regression", "density"),
        (SDR, 3, "quantile", "classifier"),
    ],
)
def test_t_e4_the_component_curve_adds_the_tuning_term_longhand(
    estimator: type, folds: int, kind: str, ratio: str, frame: Any
) -> None:
    result = _fit(estimator, [_regimen("population", kind)], frame, n_folds=folds, ratio=ratio)
    fit = result.fits["tuned"]
    term = _longhand_term(frame, fit, kind)
    off = _off_the_fit(frame, kind)
    np.testing.assert_allclose(fit.tuning_term[off], term[off], rtol=1e-8, atol=1e-10)
    first = fit.steps[0]
    if estimator is SDR:
        base = first.transform - fit.psi_scaled
    else:
        base = first.value - fit.psi_scaled
        for step in fit.steps:
            base = base + step.clever * (step.pseudo_outcome - step.targeted)
    np.testing.assert_allclose(
        fit.influence_curve_scaled[off], (base + term)[off], rtol=0, atol=1e-12
    )
    assert np.std(term) > 0.1 * np.std(base)


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e4_the_scale_component_curve_adds_the_tuning_term_longhand(
    estimator: type, frame: Any
) -> None:
    """``Scale(0.5)``: the longhand reads its own edge ``c / f`` and slope ``1 / f``."""
    result = _fit(estimator, [_regimen(kind="regression", factor=FACTOR)], frame, n_folds=3)
    fit = result.fits["tuned"]
    term = _longhand_term(frame, fit, "regression", factor=FACTOR)
    off = _off_the_fit(frame, "regression", SCALE_Q)
    np.testing.assert_allclose(fit.tuning_term[off], term[off], rtol=1e-8, atol=1e-10)
    assert np.std(term) > 0.05 * np.std(fit.influence_curve_scaled)


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_t_e4_the_component_derivative_matches_the_oracle(kind: str) -> None:
    """Node 2's ``fit.tuning_derivatives`` against the Leibniz term by quadrature.

    Flexible regressions make ``m_2`` consistent near the boundary, which the kernel
    estimate needs.  Measured relative gaps at n = 8,000: 0.056 (regression) and 0.058
    (quantile).  The natural-flag derivative differs from the oracle by about 6%, inside
    this tolerance, so the probe check above is the witness for the assigned flag.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    learner = HGB(max_iter=300, learning_rate=0.05, random_state=0)
    result = _fit(
        LTMLE,
        [_regimen("population", kind)],
        qv.sample(8_000, 11),
        n_folds=1,
        outcome_learner=learner,
        pseudo_learner=learner,
    )
    estimate = result.fits["tuned"].tuning_derivatives[1]
    truth = qv.LawQV(tuners=ORACLE_TUNERS[kind]).derivative(2)
    assert np.linalg.norm(estimate - truth) / np.linalg.norm(truth) < 0.15


def _recorded_probes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, np.ndarray]]:
    seen: list[dict[str, np.ndarray]] = []
    original = sequential.cross_fit_predictions

    def recorded(*args: Any, **kwargs: Any) -> Any:
        designs = kwargs.get("predict_designs") or {}
        if "probe_0" in designs:
            seen.append({key: np.array(value) for key, value in designs.items()})
        return original(*args, **kwargs)

    monkeypatch.setattr(sequential, "cross_fit_predictions", recorded)
    return seen


def _natural_flag_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: a probe sets the tuned component and keeps the natural flag."""
    original = sequential.outcome_design

    def mutated(data: Any, plan: Any, time: int, arm: Any = None, **kwargs: Any) -> Any:
        if kwargs.get("component") is None:
            return original(data, plan, time, arm, **kwargs)
        design = np.array(original(data, plan, time), dtype=float)
        design[:, -1] = kwargs["dose"]
        return design

    monkeypatch.setattr(sequential, "outcome_design", mutated)


def _check_probes(seen: list[dict[str, np.ndarray]], frame: pd.DataFrame) -> None:
    """The node-2 probes are the observed design with the flag at 1 and the dose set."""
    (designs,) = [entry for entry in seen if entry["history"].shape[1] > 4]
    _, _, boundary, _ = _longhand_tuner(frame, 2, "quantile", 0.5)
    for key, dose in (("probe_0", boundary), ("probe_1", boundary - 1.0)):
        expected = np.array(designs["history"], dtype=float)
        expected[:, -2] = 1.0
        expected[:, -1] = dose
        np.testing.assert_allclose(designs[key], expected, rtol=0, atol=1e-12)


def test_t_e4_the_probes_read_the_full_assigned_vector(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recorded_probes(monkeypatch)
    kept = _fit(LTMLE, [_regimen()], frame, n_folds=1).fits["tuned"]
    _check_probes(seen, frame)
    # The nonzero witness: about a half of the rows hold the natural flag 0.
    assert 0.2 < float(np.mean(frame["F2"] == 0.0)) < 0.8
    seen.clear()
    _natural_flag_probes(monkeypatch)
    mutated = _fit(LTMLE, [_regimen()], frame, n_folds=1).fits["tuned"]
    with pytest.raises(AssertionError):
        _check_probes(seen, frame)
    assert np.max(np.abs(kept.tuning_derivatives[1] - mutated.tuning_derivatives[1])) > CAUGHT


# ------------------------------------------------------------------ the training tuner (T-e7)


def _median(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.5, method="inverted_cdf"))


def _up(dose: np.ndarray, cap: float) -> np.ndarray:
    return np.where(dose + 1.0 <= cap, dose + 1.0, dose)


def _trains(folds: Folds) -> list[np.ndarray]:
    out = []
    for train, _ in folds:
        rows = np.zeros(len(folds.assignment), dtype=bool)
        rows[train] = True
        out.append(rows)
    return out


def _t_e7_log(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the split, node 2's fold regressions, and the stacked fold maps of node 2."""
    logged: dict[str, Any] = {"regressions": [], "stacked": []}
    resolve = estimator_module.resolve_tuned_plans

    def resolved(regimens: Any, data: Any, folds: Any) -> Any:
        logged["folds"] = folds
        return resolve(regimens, data, folds)

    monkeypatch.setattr(estimator_module, "resolve_tuned_plans", resolved)
    node = sequential._fit_node_regression

    def regression(data: Any, plan: Any, carried: Any, time: int, *args: Any, **kw: Any) -> Any:
        if kw.get("carry_plan") is not None and time == 2:
            logged["regressions"].append((kw["outer_fold"], plan, kw["carry_plan"]))
        return node(data, plan, carried, time, *args, **kw)

    monkeypatch.setattr(sequential, "_fit_node_regression", regression)
    classifier = sequential.classifier_ratio

    def stacked(*args: Any, **kwargs: Any) -> Any:
        if np.ndim(args[2]) == 2:
            logged["stacked"].append(kwargs.get("assigned_by_fold"))
        return classifier(*args, **kwargs)

    monkeypatch.setattr(sequential, "classifier_ratio", stacked)
    return logged


def _check_t_e7(logged: dict[str, Any], frame: pd.DataFrame, ratio: str) -> None:
    """Fold k's recursion and classifier read the flag set and the dose at fold k's cap."""
    trains = _trains(logged["folds"])
    dose = frame["A2"].to_numpy()
    owns = [_median(dose[train]) for train in trains]
    assert len(logged["regressions"]) == len(trains)
    for k, plan, carry in logged["regressions"]:
        (values,) = carry.carry_targets(2)
        np.testing.assert_array_equal(values[:, 0], np.ones(len(frame)))
        np.testing.assert_array_equal(values[:, 1], _up(dose, owns[k]))
        (held,) = plan.carry_targets(2)
        np.testing.assert_array_equal(held[:, 1], _up(dose, _median(dose)))
    if ratio == "classifier":
        (by_fold,) = logged["stacked"]
        assert by_fold is not None and len(by_fold) == len(trains)
        for k, ((probability, design),) in enumerate(by_fold):
            assert probability == 1.0
            np.testing.assert_array_equal(design[:, 0], np.ones(len(frame)))
            np.testing.assert_array_equal(design[:, 1], _up(dose, owns[k]))


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_e7_each_fold_trains_at_its_own_component_tuner(
    ratio: str, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged = _t_e7_log(monkeypatch)
    _fit(LTMLE, [_regimen()], frame, n_folds=3, ratio=ratio)
    _check_t_e7(logged, frame, ratio)
    # The nonzero witness: the training medians differ across folds.
    dose = frame["A2"].to_numpy()
    assert len({_median(dose[train]) for train in _trains(logged["folds"])}) > 1


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_m_e7_a_component_training_recursion_at_theta_n_fails(
    ratio: str, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-e7: every fold carries the map at theta_n, and the T-e7 check catches it."""
    logged = _t_e7_log(monkeypatch)
    original = sequential._fit_node_regression

    def mutated(data: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("carry_plan") is not None:
            kwargs["carry_plan"] = plan
        return original(data, plan, *args, **kwargs)

    monkeypatch.setattr(sequential, "_fit_node_regression", mutated)
    if ratio == "classifier":
        classifier = sequential.classifier_ratio
        monkeypatch.setattr(
            sequential,
            "classifier_ratio",
            lambda *args, **kwargs: classifier(*args, **{**kwargs, "assigned_by_fold": None}),
        )
    _fit(LTMLE, [_regimen()], frame, n_folds=3, ratio=ratio)
    with pytest.raises(AssertionError):
        _check_t_e7(logged, frame, ratio)


def test_t_e7_sdr_reads_the_fold_joint_density_at_theta_n(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDR, density route: each fold's joint density is evaluated under the map at theta_n."""
    sdr_module = importlib.import_module("cleverly.longitudinal.sdr")
    calls: list[tuple[Any, np.ndarray]] = []
    original = sdr_module._mixed_numerator

    def recorded(data: Any, plan: Any, time: int, **kwargs: Any) -> Any:
        calls.append((plan, np.array(kwargs["fit_rows"])))
        return original(data, plan, time, **kwargs)

    monkeypatch.setattr(sdr_module, "_mixed_numerator", recorded)
    logged = _t_e7_log(monkeypatch)
    _fit(SDR, [_regimen()], frame, n_folds=3, ratio="density")
    trains = _trains(logged["folds"])
    dose = frame["A2"].to_numpy()
    assert len(calls) == len(trains)
    for plan, rows in calls:
        assert any(np.array_equal(rows, train) for train in trains)
        (values,) = plan.carry_targets(2)
        np.testing.assert_array_equal(values[:, 1], _up(dose, _median(dose)))


# ------------------------------------------------------------------ the fold average (T-e1)

FOLD_CAPS = (1.0, 2.0)


def _grid_plan(cap: float) -> Any:
    """Set the flag and ``Shift(1.0, cap=cap)`` the dose, at both nodes of the mixed law."""
    from .. import discrete_law_longitudinal_mtp as grid

    def assign(k: int) -> int:
        _, d = grid.MIXED_CELLS[k]
        return grid._index_of(1, d + 1 if d + 1 <= cap else d)

    return grid._node_pair(assign)


def _grid_truth(cap: float) -> tuple[float, np.ndarray]:
    from .. import discrete_law_longitudinal_mtp as grid

    plan = _grid_plan(cap)
    base = np.asarray(grid.MIXED_PROBS["end"], dtype=complex)
    curve = []
    for point in range(base.size):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        value = grid.functional_mixed((1.0 - 1e-30j) * base + 1e-30j * mass, plan)
        curve.append(float(np.imag(value) / 1e-30))
    return float(np.real(grid.functional_mixed(base, plan))), np.array(curve)


def _grid_regimen() -> DynamicRegimen:
    def node(time: int) -> JointPolicy:
        dose = Shift(1.0, cap=DoseQuantile(0.5), target="fold_average")
        return JointPolicy(
            "flag, tuned dose",
            components={f"F{time}": FLAG, f"D{time}": dose},
            policy_kind="known",
        )

    return DynamicRegimen("tuned", (node(1), node(2)))


def _fold_average_on_two_copies(
    monkeypatch: pytest.MonkeyPatch, caps: tuple[float, float], estimator: type, ratio: str
) -> Any:
    """Two copies of the exact mixed grid law, each copy one fold, the fold caps fixed."""
    from .. import discrete_law_longitudinal as binary_law
    from .. import discrete_law_longitudinal_mtp as grid
    from .. import discrete_law_longitudinal_multivalue as multivalue
    from .. import longitudinal_mtp as mtp

    one = grid.mixed_frame("end")
    data = pd.concat([one, one], ignore_index=True)
    owner = np.repeat([0, 1], len(one))
    monkeypatch.setattr(
        estimator_module._SequentialEstimator,
        "_folds",
        lambda self, prepared, seed=None: Folds(assignment=owner, n_folds=2),
    )
    fit_tuner = regimen_module.fit_tuner

    def fixed(declared: Any, doses: Any, frame: Any, rows: Any, weights: Any) -> Any:
        from dataclasses import replace

        fitted = fit_tuner(declared, doses, frame, rows, weights)
        if rows.all():
            return fitted
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
        "ratio": ratio,
    }
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return estimator([_grid_regimen()], **settings).fit(data, **mtp.MIXED_COLUMNS)


def _check_fold_average(result: Any) -> None:
    from .. import discrete_law_longitudinal_mtp as grid

    name = "ey_regimen[tuned, fold average]"
    fit = result.fits["tuned, fold average"]
    rows = grid.mixed_first_row_of("end")
    n = int(grid.MIXED_N["end"])
    truths = []
    for k, cap in enumerate(FOLD_CAPS):
        truth, gateaux = _grid_truth(cap)
        truths.append(truth)
        assert fit.fold_estimates[k] == pytest.approx(truth, abs=1e-12)
        curve = result.influence_curves[name][rows + k * n]
        np.testing.assert_allclose(curve, gateaux, rtol=0, atol=1e-10)
    assert result.psi(name) == pytest.approx(np.mean(truths), abs=1e-12)


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize("ratio", ["density", "classifier"])
@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_t_e1_the_component_fold_average_is_the_average_of_exact_fold_functionals(
    estimator: type, ratio: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fold_average_on_two_copies(monkeypatch, FOLD_CAPS, estimator, ratio)
    _check_fold_average(result)
    first, second = (_grid_truth(cap)[0] for cap in FOLD_CAPS)
    assert abs(first - second) > CAUGHT


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_m_e1_one_component_fold_map_everywhere_fails(
    ratio: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fold_average_on_two_copies(monkeypatch, (FOLD_CAPS[0],) * 2, LTMLE, ratio)
    with pytest.raises(AssertionError):
        _check_fold_average(result)


# ------------------------------------------------------------------ refusals


def _node2(components: dict[str, Any], **declared: Any) -> DynamicRegimen:
    first = Shift(1.0, cap=DoseQuantile(0.5), target="population")
    return DynamicRegimen(
        "tuned",
        (first, JointPolicy("node 2", components=components, policy_kind="known", **declared)),
    )


TUNED = Shift(1.0, cap=DoseQuantile(0.5), target="population")


def test_a_tuned_component_with_declared_atoms_is_refused(frame: Any) -> None:
    """M3 kept: the derivative term has no form where the boundary meets an atom."""
    from cleverly.interventions.tuning import tuned_cap_at_atoms

    data = frame.copy()
    data.loc[data.index[::5], "A2"] = 0.0
    text = tuned_cap_at_atoms("component 'A2' of the node 'F2+A2'", (0.0,)).message()
    with pytest.raises(CapabilityError, match=re.escape(text)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LTMLE([_node2({"A2": TUNED})], **_settings(n_folds=1)).fit(
            data, dose_atoms={"A2": (0.0,)}, **COLUMNS
        )


def test_a_tuner_reads_only_what_its_map_reads(frame: Any) -> None:
    reads_itself = Shift(
        1.0, cap=SupportBoundary(LinearQuantileRegression(["W", "A2"]), 0.5), target="population"
    )
    with pytest.raises(DataError, match="which the tuned component's map may not read"):
        _fit(LTMLE, [_node2({"A2": reads_itself})], frame, n_folds=1)
