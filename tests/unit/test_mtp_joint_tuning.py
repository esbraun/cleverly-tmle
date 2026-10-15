"""Several tuned caps in one ``JointPolicy`` (X40 review remediation 2).

The value of a joint policy is linear in its components' terms.  With
:math:`\\Psi = \\sum_b p_b \\Phi_b(\\theta)`, the chain rule gives the curve
:math:`D_\\Phi + \\sum_j \\partial_j\\Phi' \\mathrm{IF}_j` with one Leibniz term per branch
``b`` and tuned dose ``j``, weighted by :math:`p_b`, and read at the vector the branch
assigns with dose ``j`` set to its boundary and to its last moved dose.

- T-e4 (i): on law V (``tests/law_v_joint.py``), one node :math:`(F, A, B)`, the curve is the
  pathwise derivative along four tilts by quadrature, for two tuned doses.  Each term is
  nonzero, and the M-e4 mutation that drops either dose's term breaks the identity.
- T-e4 (ii): the fitted curve adds the terms computed longhand from the fit's own probes,
  and the probes read the vector the branch assigns.  The same mutations in the library
  fail the longhand.
- T-e7: each fold's plan holds every tuned map at that fold's training tuner.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor

from cleverly.interventions import (
    DoseQuantile,
    JointPolicy,
    LinearQuantileRegression,
    ModifiedPolicy,
    Shift,
    SupportBoundary,
)
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, sequential
from cleverly.longitudinal import estimator as estimator_module

from .. import law_v_joint as lv
from ..law_q_tuned import Tuner

CAUGHT = 1e-4
IDENTITY = 1e-6
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": [["F", "A", "B"]],
    "baseline": ["W"],
    "time_varying": [[]],
    "continuous_treatment": ["A", "B"],
}
FEATURES = {"A": ("W", "F"), "B": ("W", "F", "A")}
ORACLE_TUNERS = {
    "quantile": (Tuner(), Tuner()),
    "regression": (Tuner(features=FEATURES["A"]), Tuner(features=FEATURES["B"])),
}
TWO = (lv.Branch(1.0, 1.0, ("A", "B")),)
PATHS = ("w", "f", "a", "b")
#: The design columns of the node's regression: ``W`` and the components ``F, A, B``.
DESIGN = {"F": 1, "A": 2, "B": 3}


class _SetFlag:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, a: Any, h: Any) -> np.ndarray:
        return np.full(len(a), self.value)


def _flag(value: float) -> ModifiedPolicy:
    return ModifiedPolicy(f"flag {value:g}", apply=_SetFlag(value), policy_kind="known")


def _tuned(dose: str, kind: str, target: str) -> Shift:
    if kind == "quantile":
        return Shift(1.0, cap=DoseQuantile(0.5), target=target)
    cap = SupportBoundary(LinearQuantileRegression(list(FEATURES[dose])), 0.5)
    return Shift(1.0, cap=cap, target=target)


def _two(kind: str = "quantile", target: str = "population") -> DynamicRegimen:
    """The flag set to 1 and both doses moved by their own tuned shift."""
    components = {"F": _flag(1.0), "A": _tuned("A", kind, target), "B": _tuned("B", kind, target)}
    return DynamicRegimen("two", (JointPolicy("two", components=components, policy_kind="known"),))


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
    return lv.sample(1500, 5)


# ------------------------------------------------------------------ the oracle (T-e4 (i))


def test_the_oracle_does_not_import_cleverly() -> None:
    tree = ast.parse(Path(lv.__file__).read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [name for name in imported if name.split(".")[0] == "cleverly"]


@pytest.mark.parametrize("branches", [TWO], ids=["two"])
@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_the_oracle_derivatives_are_derivatives_of_the_mean_in_theta(
    kind: str, branches: Any
) -> None:
    """Each analytic term, summed over the branches, is a finite difference of Psi."""
    law = lv.LawV(branches=branches, tuners=ORACLE_TUNERS[kind])
    theta = law.theta
    for dose in law.tuned_doses():
        numeric = []
        for j in range(theta[dose].size):

            def at(step: float, dose: str = dose, j: int = j) -> float:
                moved = {key: value.copy() for key, value in theta.items()}
                moved[dose][j] += step
                return law.psi(moved)

            h = 1e-4
            numeric.append((8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h))
        np.testing.assert_allclose(law.derivative(dose), numeric, rtol=0, atol=1e-6)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_t_e4_two_tuned_doses_give_the_pathwise_derivative(kind: str, path: str) -> None:
    law = lv.LawV(branches=TWO, tuners=ORACLE_TUNERS[kind])
    assert abs(lv.pathwise_derivative(law, path) - law.score_pairing(path)) < IDENTITY


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_m_e4_dropping_either_dose_term_breaks_the_identity(kind: str) -> None:
    """M-e4: the curve without A's term fails along A's tilt, and without B's along B's."""
    law = lv.LawV(branches=TWO, tuners=ORACLE_TUNERS[kind])
    for dose in ("A", "B"):
        # The nonzero witness: each dose's term.
        assert np.linalg.norm(law.derivative(dose)) > 1e-2
    for dropped, path in (("A", "a"), ("B", "b")):
        kept = tuple(dose for dose in lv.DOSES if dose != dropped)
        mutated = law.score_pairing(path, lv.Curve(doses=kept))
        assert abs(lv.pathwise_derivative(law, path) - mutated) > CAUGHT


# ------------------------------------------------------------------ the fitted curve (T-e4 (ii))


def _silverman(values: np.ndarray, power: float = 0.2) -> float:
    sd = float(np.std(values))
    iqr = float(
        np.quantile(values, 0.75, method="inverted_cdf")
        - np.quantile(values, 0.25, method="inverted_cdf")
    )
    return 0.9 * min(sd, iqr / 1.34) * values.size ** (-power)


def _longhand_tuner(frame: pd.DataFrame, dose: str, kind: str) -> tuple[np.ndarray, ...]:
    """The influence function, the boundary and the gradient of a dose's tuner."""
    x = frame[dose].to_numpy()
    n = x.size
    if kind == "quantile":
        theta = np.quantile(x, 0.5, method="inverted_cdf")
        h = _silverman(x)
        density = float(np.mean(stats.norm.pdf((theta - x) / h)) / h)
        influence = (-((x <= theta).astype(float) - 0.5) / density)[:, None]
        return influence, np.full(n, theta), np.ones((n, 1))
    features = frame[list(FEATURES[dose])].to_numpy()
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(features, x)
    beta = np.concatenate([[model.intercept_], model.coef_])
    design = np.column_stack([np.ones(n), features])
    residual = x - design @ beta
    h = _silverman(residual, power=1.0 / 3.0)
    inside = (np.abs(residual) <= h).astype(float)
    jacobian = (design * inside[:, None]).T @ design / (2.0 * h * n)
    score = design * (0.5 - (residual <= 0.0).astype(float))[:, None]
    return np.linalg.solve(jacobian, score.T).T, design @ beta, design


def _longhand_term(
    frame: pd.DataFrame, probes: np.ndarray, kind: str, slots: list[tuple[str, float]]
) -> np.ndarray:
    """``sum_j IF_j' sum_b p_b dPhi_bj``: slot ``i`` reads probe columns ``2i`` and ``2i + 1``."""
    term = np.zeros(len(frame))
    for dose in dict.fromkeys(dose for dose, _ in slots):
        influence, boundary, gradient = _longhand_tuner(frame, dose, kind)
        gap = frame[dose].to_numpy() - (boundary - 1.0)
        kernel = stats.norm.pdf(gap / _silverman(gap)) / _silverman(gap)
        derivative = np.zeros(gradient.shape[1])
        for index, (own, probability) in enumerate(slots):
            if own == dose:
                jump = probes[:, 2 * index] - probes[:, 2 * index + 1]
                derivative = derivative + probability * (kernel * jump) @ gradient / len(frame)
        term = term + influence @ derivative
    return term


def _off_the_fit(frame: pd.DataFrame, kind: str, doses: tuple[str, ...]) -> np.ndarray:
    """The rows off each linear quantile fit's hyperplane (a linear program interpolates them)."""
    keep = np.ones(len(frame), dtype=bool)
    if kind == "quantile":
        return keep
    for dose in doses:
        _, boundary, gradient = _longhand_tuner(frame, dose, kind)
        on = np.abs(frame[dose].to_numpy() - boundary) < 1e-9
        assert on.sum() <= gradient.shape[1]
        keep &= ~on
    return keep


def _check_curve(
    estimator: type, fit: Any, frame: pd.DataFrame, kind: str, slots: list[tuple[str, float]]
) -> np.ndarray:
    """The curve is the fit's own base curve plus the longhand terms."""
    (step,) = fit.steps
    term = _longhand_term(frame, step.probes, kind, slots)
    off = _off_the_fit(frame, kind, tuple(dict.fromkeys(dose for dose, _ in slots)))
    np.testing.assert_allclose(fit.tuning_term[off], term[off], rtol=1e-8, atol=1e-10)
    if estimator is SDR:
        base = step.transform - fit.psi_scaled
    else:
        base = step.value - fit.psi_scaled + step.clever * (step.pseudo_outcome - step.targeted)
    np.testing.assert_allclose(
        fit.influence_curve_scaled[off], (base + term)[off], rtol=0, atol=1e-12
    )
    return term


ROUTES = [
    (LTMLE, 1, "quantile", "density"),
    (LTMLE, 3, "regression", "classifier"),
    (LTMLE, 3, "quantile", "density"),
    (SDR, 1, "regression", "density"),
    (SDR, 3, "quantile", "classifier"),
]
TWO_SLOTS = [("A", 1.0), ("B", 1.0)]


@pytest.mark.parametrize(("estimator", "folds", "kind", "ratio"), ROUTES)
def test_t_e4_two_tuned_doses_add_both_terms_longhand(
    estimator: type, folds: int, kind: str, ratio: str, frame: Any
) -> None:
    fit = _fit(estimator, [_two(kind)], frame, n_folds=folds, ratio=ratio).fits["two"]
    _check_curve(estimator, fit, frame, kind, TWO_SLOTS)
    # The nonzero witness: each dose's term moves the curve.
    (step,) = fit.steps
    for slot in TWO_SLOTS:
        assert np.std(_longhand_term(frame, step.probes, kind, [slot])) > 1e-3
    assert len(fit.tuning_derivatives) == 2


def _drop_slot(monkeypatch: pytest.MonkeyPatch, dropped: int) -> None:
    """The mutation: one slot of the node carries no term."""
    original = sequential._probe_slots

    def mutated(plan: Any, time: int) -> Any:
        slots = list(original(plan, time))
        return tuple(slot for index, slot in enumerate(slots) if index != dropped)

    monkeypatch.setattr(sequential, "_probe_slots", mutated)


@pytest.mark.parametrize("dropped", [0, 1])
def test_m_e4_a_dose_without_its_term_fails_the_longhand(
    dropped: int, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _fit(LTMLE, [_two()], frame, n_folds=1).fits["two"]
    (step,) = reference.steps
    expected = _longhand_term(frame, step.probes, "quantile", TWO_SLOTS)
    _drop_slot(monkeypatch, dropped)
    mutated = _fit(LTMLE, [_two()], frame, n_folds=1).fits["two"]
    assert np.max(np.abs(mutated.tuning_term - expected)) > CAUGHT


def _recorded_designs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, np.ndarray]]:
    seen: list[dict[str, np.ndarray]] = []
    original = sequential.cross_fit_predictions

    def recorded(*args: Any, **kwargs: Any) -> Any:
        designs = kwargs.get("predict_designs") or {}
        if "probe_0" in designs:
            seen.append({key: np.array(value) for key, value in designs.items()})
        return original(*args, **kwargs)

    monkeypatch.setattr(sequential, "cross_fit_predictions", recorded)
    return seen


def _up(dose: np.ndarray, cap: np.ndarray) -> np.ndarray:
    return np.where(dose + 1.0 <= cap, dose + 1.0, dose)


def _check_two_probes(designs: dict[str, np.ndarray], frame: pd.DataFrame) -> None:
    """Each dose's probes set it and read the flag at 1 and the other dose as assigned."""
    caps = {dose: _longhand_tuner(frame, dose, "quantile")[1] for dose in ("A", "B")}
    assigned = {dose: _up(frame[dose].to_numpy(), caps[dose]) for dose in ("A", "B")}
    for index, dose in enumerate(("A", "B")):
        for side, value in enumerate((caps[dose], caps[dose] - 1.0)):
            expected = np.array(designs["history"], dtype=float)
            expected[:, DESIGN["F"]] = 1.0
            for other in ("A", "B"):
                expected[:, DESIGN[other]] = value if other == dose else assigned[other]
            np.testing.assert_allclose(
                designs[f"probe_{2 * index + side}"], expected, rtol=0, atol=1e-12
            )


def _natural_other_dose(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: a probe sets its dose and the flag, and keeps the other dose natural."""
    original = sequential.outcome_design

    def mutated(data: Any, plan: Any, time: int, arm: Any = None, **kwargs: Any) -> Any:
        if kwargs.get("component") is None:
            return original(data, plan, time, arm, **kwargs)
        design = np.array(original(data, plan, time), dtype=float)
        design[:, DESIGN["F"]] = 1.0
        design[:, 1 + kwargs["component"]] = kwargs["dose"]
        return design

    monkeypatch.setattr(sequential, "outcome_design", mutated)


def test_t_e4_each_probe_reads_the_other_dose_as_assigned(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _recorded_designs(monkeypatch)
    kept = _fit(LTMLE, [_two()], frame, n_folds=1).fits["two"]
    (designs,) = seen
    _check_two_probes(designs, frame)
    # The nonzero witness: the policy moves about a half of each dose.
    for dose in ("A", "B"):
        cap = _longhand_tuner(frame, dose, "quantile")[1]
        moved = _up(frame[dose].to_numpy(), cap) != frame[dose].to_numpy()
        assert 0.2 < float(np.mean(moved)) < 0.8
    seen.clear()
    _natural_other_dose(monkeypatch)
    mutated = _fit(LTMLE, [_two()], frame, n_folds=1).fits["two"]
    with pytest.raises(AssertionError):
        _check_two_probes(seen[0], frame)
    gap = max(
        np.max(np.abs(a - b))
        for a, b in zip(kept.tuning_derivatives, mutated.tuning_derivatives, strict=True)
    )
    assert gap > CAUGHT


# ------------------------------------------------------------------ the training tuners (T-e7)


def _resolved(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    logged: dict[str, Any] = {}
    original = estimator_module.resolve_tuned_plans

    def recorded(regimens: Any, data: Any, folds: Any) -> Any:
        logged["folds"] = folds
        logged["plans"] = original(regimens, data, folds)
        return logged["plans"]

    monkeypatch.setattr(estimator_module, "resolve_tuned_plans", recorded)
    return logged


def _median(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.5, method="inverted_cdf"))


def _trains(folds: Any) -> list[np.ndarray]:
    out = []
    for train, _ in folds:
        rows = np.zeros(len(folds.assignment), dtype=bool)
        rows[train] = True
        out.append(rows)
    return out


def _check_fold_maps(logged: dict[str, Any], frame: pd.DataFrame, doses: dict[str, int]) -> None:
    """Fold k's plan moves every tuned dose, in every branch, at fold k's training median."""
    (plan,) = logged["plans"]
    trains = _trains(logged["folds"])
    assert len(plan.fold_plans) == len(trains)
    for train, fold_plan in zip(trains, plan.fold_plans, strict=True):
        for values in fold_plan.carry_targets(1):
            for dose, column in doses.items():
                natural = frame[dose].to_numpy()
                expected = _up(natural, np.full(len(frame), _median(natural[train])))
                np.testing.assert_array_equal(values[:, column], expected)


@pytest.mark.parametrize("target", ["population", "fold_average"])
def test_t_e7_each_fold_holds_both_doses_at_its_own_tuner(
    target: str, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged = _resolved(monkeypatch)
    _fit(LTMLE, [_two(target=target)], frame, n_folds=3)
    _check_fold_maps(logged, frame, {"A": 1, "B": 2})
    # The nonzero witness: the training medians differ across folds.
    for dose in ("A", "B"):
        natural = frame[dose].to_numpy()
        assert len({_median(natural[train]) for train in _trains(logged["folds"])}) > 1
