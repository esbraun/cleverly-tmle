"""A data-tuned cap at a dose component of a point vector treatment (X40 remediation 2).

The point analogue of ``tests/unit/test_mtp_tuned_components.py``.  The treatment is the
vector :math:`(F, A, B)` of law V (``tests/law_v_joint.py``), and the policy sets the flag
to 1 and moves ``A`` by ``Shift(1.0, cap=c(h))``.  The tuner of ``c`` reads what the map
of ``A`` reads, :math:`W` and the natural flag.  The curve is
:math:`D_\\Phi + \\partial\\Phi' \\mathrm{IF}`, with the Leibniz term read at the vector the
policy assigns, ``A`` set to its boundary and to its last moved dose.

- T-e4 (i): the curve is the pathwise derivative along four tilts, by quadrature.  M-e4:
  dropping the term, or reading the natural flag in it, breaks the identity.
- T-e4 (ii): the reported curve is the curve at :math:`\\theta_n` plus the term computed
  longhand; the probes are the assigned vector with ``A`` set.  At n = 8,000 the estimated
  term matches the oracle's.
- T-e7 and T-e2: fold ``k``'s classifier stacks the map at fold ``k``'s tuner, and each fold
  map of the fold average reads its own training rows.  M-e7: the map at
  :math:`\\theta_n` in every fold fails the check.
- Two tuned doses and a randomizer, as at a mixed node, and the refusals: a tuned
  categorical component, declared atoms, a tuner that reads a later dose, and two targets.
"""

from __future__ import annotations

import importlib
import re
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.estimators import TMLE
from cleverly.estimators import _tuned_policies as tuned_module
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    DoseQuantile,
    JointPolicy,
    LinearQuantileRegression,
    Shift,
    SupportBoundary,
)
from cleverly.interventions.tuning import tuned_cap_at_atoms
from cleverly.learners import density_ratio as density_ratio_module

from .. import law_v_joint as lv
from .test_mtp_joint_tuning import ORACLE_TUNERS, PATHS, _flag, _tuned

tmle_module = importlib.import_module("cleverly.estimators.tmle")
CAUGHT = 1e-4
IDENTITY = 1e-6
NAME = "ey_policy[point]"
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["F", "A", "B"],
    "covariates": ["W"],
    "continuous_treatment": ["A", "B"],
}
POINT = (lv.Branch(1.0, 1.0, ("A",)),)
#: The columns of a point vector: the components ``F, A, B`` in order.
COLUMN = {"F": 0, "A": 1, "B": 2}


def _point(kind: str = "quantile", target: str = "population") -> JointPolicy:
    components = {"F": _flag(1.0), "A": _tuned("A", kind, target)}
    return JointPolicy("point", components=components, policy_kind="known")


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


def _fit(policies: Any, frame: pd.DataFrame, **settings: Any) -> Any:
    folds = settings.pop("n_folds", 3)
    settings.setdefault("cross_fit", folds > 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TMLE(policies=policies, n_folds=max(folds, 2), **_settings(**settings))
        return model.fit(frame, **COLUMNS).single()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return lv.sample(1500, 5)


# ------------------------------------------------------------------ the oracle (T-e4 (i))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_t_e4_the_point_component_curve_is_the_pathwise_derivative(kind: str, path: str) -> None:
    law = lv.LawV(branches=POINT, tuners=ORACLE_TUNERS[kind])
    assert abs(lv.pathwise_derivative(law, path) - law.score_pairing(path)) < IDENTITY


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_m_e4_the_point_identity_fails_without_the_term_or_the_assigned_flag(kind: str) -> None:
    law = lv.LawV(branches=POINT, tuners=ORACLE_TUNERS[kind])
    pathwise = lv.pathwise_derivative(law, "a")
    assert np.linalg.norm(law.derivative("A")) > 1e-2
    assert abs(pathwise - law.score_pairing("a", lv.Curve(doses=()))) > CAUGHT
    natural = law.score_pairing("a", lv.Curve(assigned_flag=False))
    assert abs(pathwise - natural) > CAUGHT


# ------------------------------------------------------------------ the fitted curve (T-e4 (ii))


def _silverman(values: np.ndarray, power: float = 0.2) -> float:
    sd = float(np.std(values))
    iqr = float(
        np.quantile(values, 0.75, method="inverted_cdf")
        - np.quantile(values, 0.25, method="inverted_cdf")
    )
    return 0.9 * min(sd, iqr / 1.34) * values.size ** (-power)


def _longhand_tuner(frame: pd.DataFrame, kind: str) -> tuple[np.ndarray, ...]:
    """The influence function, the boundary and the gradient of the tuner of ``A``."""
    dose = frame["A"].to_numpy()
    n = dose.size
    if kind == "quantile":
        theta = np.quantile(dose, 0.5, method="inverted_cdf")
        h = _silverman(dose)
        density = float(np.mean(stats.norm.pdf((theta - dose) / h)) / h)
        influence = (-((dose <= theta).astype(float) - 0.5) / density)[:, None]
        return influence, np.full(n, theta), np.ones((n, 1))
    features = frame[["W", "F"]].to_numpy()
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(features, dose)
    beta = np.concatenate([[model.intercept_], model.coef_])
    design = np.column_stack([np.ones(n), features])
    residual = dose - design @ beta
    h = _silverman(residual, power=1.0 / 3.0)
    inside = (np.abs(residual) <= h).astype(float)
    jacobian = (design * inside[:, None]).T @ design / (2.0 * h * n)
    score = design * (0.5 - (residual <= 0.0).astype(float))[:, None]
    return np.linalg.solve(jacobian, score.T).T, design @ beta, design


def _longhand_term(frame: pd.DataFrame, nuisance: Any, kind: str) -> np.ndarray:
    influence, boundary, gradient = _longhand_tuner(frame, kind)
    gap = frame["A"].to_numpy() - (boundary - 1.0)
    kernel = stats.norm.pdf(gap / _silverman(gap)) / _silverman(gap)
    high = nuisance.outcome_probes["tuned0:0:0:boundary"]
    low = nuisance.outcome_probes["tuned0:0:0:edge"]
    derivative = (kernel * (high - low)) @ gradient / len(frame)
    return np.asarray(influence @ derivative)


def _off_the_fit(frame: pd.DataFrame, kind: str) -> np.ndarray:
    """The rows off a linear quantile fit's hyperplane (a linear program interpolates them)."""
    if kind == "quantile":
        return np.ones(len(frame), dtype=bool)
    _, boundary, gradient = _longhand_tuner(frame, kind)
    on = np.abs(frame["A"].to_numpy() - boundary) < 1e-9
    assert on.sum() <= gradient.shape[1]
    return ~on


def _without_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: the fit resolves the map at theta_n and carries no tuning term."""
    resolve = tuned_module.resolve_point_policies
    monkeypatch.setattr(
        tmle_module,
        "resolve_point_policies",
        lambda policies, data, folds: replace(resolve(policies, data, folds), tunings=()),
    )


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
def test_t_e4_the_point_component_curve_adds_the_tuning_term_longhand(
    folds: int, kind: str, ratio: str, frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept = _fit([_point(kind)], frame, n_folds=folds, ratio=ratio)
    term = _longhand_term(frame, kept.nuisance, kind)
    off = _off_the_fit(frame, kind)
    np.testing.assert_allclose(
        kept.nuisance.policies.curve_terms[off, 0], term[off], rtol=1e-8, atol=1e-10
    )
    _without_terms(monkeypatch)
    dropped = _fit([_point(kind)], frame, n_folds=folds, ratio=ratio)
    assert dropped.nuisance.policies.curve_terms is None
    assert dropped.psi(NAME) == pytest.approx(kept.psi(NAME), abs=1e-14)
    gap = kept.influence_curves[NAME] - dropped.influence_curves[NAME]
    np.testing.assert_allclose(gap[off], term[off], rtol=0, atol=1e-12)
    assert np.max(np.abs(gap)) > CAUGHT
    assert np.std(term) > 0.1 * np.std(dropped.influence_curves[NAME])


def _up(dose: np.ndarray, cap: Any) -> np.ndarray:
    return np.where(dose + 1.0 <= cap, dose + 1.0, dose)


def _recorded_probes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []
    resolve = tuned_module.resolve_point_policies

    def resolved(policies: Any, data: Any, folds: Any) -> Any:
        out = resolve(policies, data, folds)
        seen.append(out)
        return out

    monkeypatch.setattr(tmle_module, "resolve_point_policies", resolved)
    return seen


def _check_probes(tuned: Any, frame: pd.DataFrame) -> None:
    """The probes are the natural vector with the flag at 1 and ``A`` at its two doses."""
    _, boundary, _ = _longhand_tuner(frame, "quantile")
    probes = tuned.probe_doses
    assert sorted(probes) == ["tuned0:0:0:boundary", "tuned0:0:0:edge"]
    for key, dose in (("boundary", boundary), ("edge", boundary - 1.0)):
        expected = np.array(frame[["F", "A", "B"]], dtype=float)
        expected[:, COLUMN["F"]] = 1.0
        expected[:, COLUMN["A"]] = dose
        np.testing.assert_allclose(probes[f"tuned0:0:0:{key}"], expected, rtol=0, atol=1e-12)


def _natural_flag_probes(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    """The mutation: a probe sets ``A`` and keeps the natural flag."""
    original = tuned_module.PointTuning.probe_values

    def mutated(self: Any, slot: int) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for values in original(self, slot):
            moved = np.array(values, dtype=float, copy=True)
            moved[:, COLUMN["F"]] = frame["F"].to_numpy()
            out.append(moved)
        return out[0], out[1]

    monkeypatch.setattr(tuned_module.PointTuning, "probe_values", mutated)


def test_t_e4_the_point_probes_read_the_full_assigned_vector(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The outcome regression reads the flag, so a natural-flag probe moves the term."""
    flexible = make_pipeline(PolynomialFeatures(2), LinearRegression())
    seen = _recorded_probes(monkeypatch)
    kept = _fit([_point()], frame, n_folds=1, outcome_learner=flexible)
    _check_probes(seen[0], frame)
    # The nonzero witness: about a half of the rows hold the natural flag 0.
    assert 0.2 < float(np.mean(frame["F"] == 0.0)) < 0.8
    seen.clear()
    _natural_flag_probes(monkeypatch, frame)
    mutated = _fit([_point()], frame, n_folds=1, outcome_learner=flexible)
    with pytest.raises(AssertionError):
        _check_probes(seen[0], frame)
    gap = kept.nuisance.policies.curve_terms - mutated.nuisance.policies.curve_terms
    assert np.max(np.abs(gap)) > CAUGHT


@pytest.mark.parametrize("kind", sorted(ORACLE_TUNERS))
def test_t_e4_the_point_component_derivative_matches_the_oracle(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel term against law V's by quadrature, at n = 8,000.

    Measured relative gaps: 0.059 (quantile) and 0.032 (regression).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    derivatives: list[np.ndarray] = []
    original = tuned_module.cap_derivative

    def recorded(**kwargs: Any) -> np.ndarray:
        derivatives.append(original(**kwargs))
        return derivatives[-1]

    monkeypatch.setattr(tuned_module, "cap_derivative", recorded)
    learner = HGB(max_iter=300, learning_rate=0.05, random_state=0)
    _fit([_point(kind)], lv.sample(8_000, 11), n_folds=2, outcome_learner=learner)
    truth = lv.LawV(branches=POINT, tuners=ORACLE_TUNERS[kind]).derivative("A")
    (estimate,) = derivatives
    assert np.linalg.norm(estimate - truth) / np.linalg.norm(truth) < 0.15


# ------------------------------------------------------------------ the fold maps (T-e7, T-e2)


def _median(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.5, method="inverted_cdf"))


class _Log:
    """The split, and what each stacked classifier stacks per fold."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.folds: Any = None
        self.stacked: list[Any] = []
        resolve = tmle_module.resolve_point_policies

        def resolved(policies: Any, data: Any, folds: Any) -> Any:
            self.folds = folds
            return resolve(policies, data, folds)

        monkeypatch.setattr(tmle_module, "resolve_point_policies", resolved)
        classifier = density_ratio_module.classifier_ratio

        def recorded(*args: Any, **kwargs: Any) -> Any:
            self.stacked.append(kwargs.get("assigned_by_fold"))
            return classifier(*args, **kwargs)

        monkeypatch.setattr(density_ratio_module, "classifier_ratio", recorded)

    def trains(self) -> list[np.ndarray]:
        out = []
        for train, _ in self.folds:
            rows = np.zeros(len(self.folds.assignment), dtype=bool)
            rows[train] = True
            out.append(rows)
        return out


def _check_stacked(log: _Log, frame: pd.DataFrame) -> None:
    """Fold k's classifier stacks the flag at 1 and ``A`` at fold k's training median."""
    dose = frame["A"].to_numpy()
    trains = log.trains()
    (by_fold,) = log.stacked
    assert by_fold is not None and len(by_fold) == len(trains)
    for train, ((probability, design),) in zip(trains, by_fold, strict=True):
        assert probability == 1.0
        np.testing.assert_array_equal(design[:, COLUMN["F"]], np.ones(len(frame)))
        np.testing.assert_array_equal(design[:, COLUMN["A"]], _up(dose, _median(dose[train])))


def test_t_e7_each_point_fold_classifier_stacks_its_own_component_tuner(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log(monkeypatch)
    _fit([_point()], frame, n_folds=3, ratio="classifier")
    _check_stacked(log, frame)
    dose = frame["A"].to_numpy()
    assert len({_median(dose[train]) for train in log.trains()}) > 1


def test_m_e7_the_map_at_theta_n_in_every_point_fold_fails(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolve = tuned_module.resolve_point_policies
    monkeypatch.setattr(
        tmle_module,
        "resolve_point_policies",
        lambda policies, data, folds: replace(resolve(policies, data, folds), by_fold=None),
    )
    log = _Log(monkeypatch)
    _fit([_point()], frame, n_folds=3, ratio="classifier")
    with pytest.raises(AssertionError):
        _check_stacked(log, frame)


def test_t_e2_each_point_fold_component_reads_its_training_rows(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log(monkeypatch)
    result = _fit([_point(target="fold_average")], frame, n_folds=3)
    policies = result.nuisance.policies
    owner = np.asarray(log.folds.assignment)
    dose = frame["A"].to_numpy()
    assert policies.component_rows is not None
    for k, train in enumerate(log.trains()):
        np.testing.assert_array_equal(policies.component_rows[:, k], owner == k)
        np.testing.assert_array_equal(
            policies.shifted[:, k, COLUMN["A"]], _up(dose, _median(dose[train]))
        )
    assert "ey_policy[point, fold average]" in result.estimates


# ------------------------------------------------------------------ the joint compositions


def test_two_point_tuned_doses_and_a_point_randomizer_sum_their_terms(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point curve adds one term per slot, as at a mixed node: checked longhand."""
    two = _tuned("A", "quantile", "population"), _tuned("B", "quantile", "population")
    policies = [
        JointPolicy(
            "two", components={"F": _flag(1.0), "A": two[0], "B": two[1]}, policy_kind="known"
        ),
        _randomized_point(),
    ]
    seen = _recorded_probes(monkeypatch)
    result = _fit(policies, frame, n_folds=1)
    probes = result.nuisance.outcome_probes
    expected = np.zeros((len(frame), 2))
    for column, slots in enumerate(
        ([("A", "tuned0:0:0", 1.0), ("B", "tuned0:1:0", 1.0)],
         [("A", "tuned1:0:0", 0.6), ("A", "tuned1:0:1", 0.4)])
    ):  # fmt: skip
        for dose in dict.fromkeys(dose for dose, _, _ in slots):
            x = frame[dose].to_numpy()
            theta = np.quantile(x, 0.5, method="inverted_cdf")
            h = _silverman(x)
            density = float(np.mean(stats.norm.pdf((theta - x) / h)) / h)
            influence = -((x <= theta).astype(float) - 0.5) / density
            gap = x - (theta - 1.0)
            kernel = stats.norm.pdf(gap / _silverman(gap)) / _silverman(gap)
            derivative = sum(
                p * float(np.mean(kernel * (probes[f"{key}:boundary"] - probes[f"{key}:edge"])))
                for own, key, p in slots
                if own == dose
            )
            expected[:, column] += influence * derivative
    np.testing.assert_allclose(result.nuisance.policies.curve_terms, expected, atol=1e-10)
    # The nonzero witness: both columns, and the flag each randomized branch sets.
    assert np.all(np.std(expected, axis=0) > 1e-3)
    vectors = seen[0].probe_doses
    np.testing.assert_array_equal(vectors["tuned1:0:0:boundary"][:, COLUMN["F"]], 1.0)
    np.testing.assert_array_equal(vectors["tuned1:0:1:boundary"][:, COLUMN["F"]], 0.0)


def _randomized_point() -> JointPolicy:
    from cleverly.interventions import Randomizer

    tuned = _tuned("A", "quantile", "population")
    return JointPolicy(
        "randomized",
        components={"x": {"F": _flag(1.0), "A": tuned}, "y": {"F": _flag(0.0), "A": tuned}},
        randomizer=Randomizer(("x", "y"), (0.6, 0.4)),
        policy_kind="known",
    )


def test_a_tuned_categorical_point_component_is_refused(frame: Any) -> None:
    policy = JointPolicy(
        "flag", components={"F": _tuned("A", "quantile", "population")}, policy_kind="known"
    )
    with pytest.raises(CapabilityError, match="applies to a continuous component"):
        _fit([policy], frame, n_folds=1)


def test_a_tuned_point_component_with_declared_atoms_is_refused(frame: Any) -> None:
    data = frame.copy()
    data.loc[data.index[::5], "A"] = 0.0
    text = tuned_cap_at_atoms("component 'A' of the treatment", (0.0,)).message()
    head, tail = text.split("of the treatment", 1)
    pattern = re.escape(head + "of the treatment") + ".*" + re.escape(tail)
    with pytest.raises(CapabilityError, match=pattern), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TMLE(policies=[_point()], n_folds=2, cross_fit=False, **_settings()).fit(
            data, dose_atoms={"A": (0.0,)}, **COLUMNS
        )


def test_a_point_tuner_reads_only_what_its_map_reads(frame: Any) -> None:
    reads_later = Shift(
        1.0, cap=SupportBoundary(LinearQuantileRegression(["W", "B"]), 0.5), target="population"
    )
    policy = JointPolicy(
        "later", components={"F": _flag(1.0), "A": reads_later}, policy_kind="known"
    )
    with pytest.raises(DataError, match="which the tuned component's map may not read"):
        _fit([policy], frame, n_folds=1)


def test_two_targets_in_one_point_joint_policy_are_refused(frame: Any) -> None:
    policy = JointPolicy(
        "mixed targets",
        components={
            "A": Shift(1.0, cap=DoseQuantile(0.5), target="population"),
            "B": Shift(1.0, cap=DoseQuantile(0.5), target="fold_average"),
        },
        policy_kind="known",
    )
    with pytest.raises(CapabilityError, match="One policy is one parameter"):
        _fit([policy], frame, n_folds=3)
