"""X82 part F, longitudinal latent: one time-fixed latent shared by every node.

Each node takes the DoWhy law of its kind and the outcome takes the point law.  The exact
witness fits saturated sequential regressions, so the cell equals the sequential g-formula
computed by hand on the perturbed table.  Its mutation control perturbs node 1 alone, and
the hand value then differs from the cell.  The other tests pin the per-node association,
the nested flips, the held design, the declared node-1 mechanism, and the refusals.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.longitudinal import IPSI, LTMLE, SDR
from cleverly.longitudinal.estimator import (
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity._simulated_confounding_longitudinal import (
    _CENSORING_REFUSAL,
    _CLUSTERED_REFUSAL,
    _IPSI_REFUSAL,
    _KNOWN_LATER_NODE_REFUSAL,
    _LATER_COVARIATE_REFUSAL,
    _NODE_LAW_REFUSAL,
    _OUTCOME_KIND_REFUSAL,
    _frozen_seeds,
    _refuse_node_law,
    longitudinal_fit_refusal,
    perturb_nodes,
)
from cleverly.sensitivity._simulated_confounding_request import (
    _fit_wide_refusal,
    simulated_confounding_refusal,
)
from cleverly.sensitivity.simulated_confounding import (
    _binary_calibration,
    _draw_latent,
    _latent_child_seed,
)
from tests.unit._confounding_support import forbid_draw_and_refit

CONTRAST = "ate_regimen[never vs always]"
REGIMENS = {"always": 1, "never": 0}
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], []],
}


class CellMean(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """A saturated regression: the weighted mean of the target in each covariate cell."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMean:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.means_: dict[tuple[float, ...], float] = {}
        for key in {tuple(row) for row in X}:
            mask = (np.asarray(key) == X).all(axis=1)
            self.means_[key] = float(np.average(y[mask], weights=w[mask]))
        self.fallback_ = float(np.average(y, weights=w))
        return self

    def predict(self, X: Any) -> np.ndarray[Any, Any]:
        return np.array([self.means_.get(tuple(row), self.fallback_) for row in np.asarray(X)])


def _panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A two-node uncensored panel with a later node that reads the earlier one."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a1 = rng.binomial(1, 1 / (1 + np.exp(-w1))).astype(float)
    a2 = rng.binomial(1, 1 / (1 + np.exp(-(w1 + a1 - 0.5)))).astype(float)
    y = w1 + a1 + a2 + rng.normal(size=n)
    dose1 = w1 + rng.normal(size=n)
    dose2 = dose1 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "W1": w1,
            "W2": w2,
            "A1": a1,
            "A2": a2,
            "Y": y,
            "D1": dose1,
            "D2": dose2,
            "YD": w1 + dose1 + dose2 + rng.normal(size=n),
            "A": a1,
        }
    )


def _ltmle(**settings: Any) -> LTMLE:
    options: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 3,
        "simultaneous": False,
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
    }
    options.update(settings)
    return LTMLE(REGIMENS, **options)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _panel()


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return _ltmle().fit(frame, **COLUMNS)


def _latent(result: Any, seed: int) -> np.ndarray[Any, Any]:
    return _draw_latent(result.data.n, _latent_child_seed(seed), None, "row", None)


def _capture(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[Any]:
    seen: list[Any] = []
    original = result.estimator.refit

    def refit(data: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append((data, kwargs))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(result.estimator, "refit", refit)
    return seen


# ------------------------------------------------------------------ the exact witness


def _discrete_panel(n: int = 2000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.binomial(1, 0.5, n).astype(float)
    w2 = rng.binomial(1, 0.4, n).astype(float)
    a1 = rng.binomial(1, 0.3 + 0.4 * w1).astype(float)
    a2 = rng.binomial(1, 0.2 + 0.3 * a1 + 0.3 * w2).astype(float)
    y = 0.5 * w1 - 0.3 * w2 + a1 + 0.7 * a2 + rng.normal(size=n)
    return pd.DataFrame({"W1": w1, "W2": w2, "A1": a1, "A2": a2, "Y": y})


def _hand_gformula(
    w: np.ndarray[Any, Any], a1: np.ndarray[Any, Any], a2: np.ndarray[Any, Any], y: Any, arm: int
) -> float:
    """The two-node sequential g-formula of a static regimen with cell-mean regressions."""
    second = np.column_stack([w, a1, a2])
    q2 = CellMean().fit(second, y)
    pseudo = q2.predict(np.column_stack([w, a1, np.full_like(a2, arm)]))
    q1 = CellMean().fit(np.column_stack([w, a1]), pseudo)
    return float(np.mean(q1.predict(np.column_stack([w, np.full_like(a1, arm)]))))


def test_a_cell_equals_the_sequential_g_formula_on_the_perturbed_table() -> None:
    frame = _discrete_panel()
    estimator = SequentialGComputation(
        REGIMENS, random_state=3, simultaneous=False, outcome_learner=CellMean()
    )
    result = estimator.fit(frame, **COLUMNS)
    w = frame[["W1", "W2"]].to_numpy()
    a1, a2, y = (frame[name].to_numpy() for name in ("A1", "A2", "Y"))
    # The saturated fit reproduces the hand g-formula on the observed table.
    observed = _hand_gformula(w, a1, a2, y, 0) - _hand_gformula(w, a1, a2, y, 1)
    assert result[CONTRAST].psi == pytest.approx(observed, abs=1e-10)
    grid = ConfounderStrengthGrid(
        treatment={"A1": (0.0, 0.2), "A2": (0.0, 0.1)}, outcome=(0.0, 0.6)
    )
    surface = simulated_confounding(result, CONTRAST, grid=grid, random_state=11)
    assert surface.complete
    latent = _latent(result, surface.root_seed)
    tail1 = latent >= NormalDist().inv_cdf(0.8)
    tail2 = latent >= NormalDist().inv_cdf(0.9)
    moved_a1 = np.where(tail1, 1 - a1, a1)
    moved_a2 = np.where(tail2, 1 - a2, a2)
    moved_y = y - 0.6 * latent
    expected = _hand_gformula(w, moved_a1, moved_a2, moved_y, 0) - _hand_gformula(
        w, moved_a1, moved_a2, moved_y, 1
    )
    corner = surface.cells[-1]
    assert corner.treatment_strength == (0.2, 0.1)
    assert corner.estimate == pytest.approx(expected, abs=1e-10)
    # Mutation control: a law that moves node 1 alone gives another value.
    node_one_only = _hand_gformula(w, moved_a1, a2, moved_y, 0) - _hand_gformula(
        w, moved_a1, a2, moved_y, 1
    )
    assert abs(corner.estimate - node_one_only) > 1e-3
    assert abs(corner.estimate - surface.original_estimate) > 1e-3


# ------------------------------------------------------------------ the law, node by node


def test_each_node_reports_its_own_association(fitted: Any) -> None:
    grid = ConfounderStrengthGrid(treatment={"A1": (0.0, 0.2), "A2": (0.0, 0.3)}, outcome=(0.0,))
    surface = simulated_confounding(fitted, CONTRAST, grid=grid, random_state=4)
    assert surface.treatment_components == ("A1", "A2")
    assert surface.treatment_family == "longitudinal"
    latent = _latent(fitted, surface.root_seed)
    treatment = fitted.data.treatment
    corner = surface.cells[-1]
    moved = perturb_nodes(fitted.data, latent, (0.2, 0.3), ("binary", "binary"), False)
    for position in range(2):
        expected = np.corrcoef(latent, moved[:, position])[0, 1]
        assert corner.induced_treatment_association[position] == pytest.approx(expected, abs=1e-12)
    anchor = surface.cells[0]
    assert anchor.estimate == surface.original_estimate
    assert anchor.induced_treatment_association[1] == pytest.approx(
        np.corrcoef(latent, treatment[:, 1])[0, 1], abs=1e-12
    )


def test_the_flipped_row_sets_are_nested_in_strength_order(fitted: Any) -> None:
    latent = _latent(fitted, 9)
    original = fitted.data.treatment
    moved = perturb_nodes(fitted.data, latent, (0.1, 0.25), ("binary", "binary"), False)
    first = moved[:, 0] != original[:, 0]
    second = moved[:, 1] != original[:, 1]
    assert first.any() and (second & ~first).any()
    # One latent and one tail rule: the weaker node's flipped rows sit inside the stronger's.
    assert np.all(second[first])
    assert np.array_equal(first, latent >= NormalDist().inv_cdf(0.9))


def test_a_one_axis_grid_moves_every_node_and_the_refit_sees_it(
    fitted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture(monkeypatch, fitted)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.4))
    surface = simulated_confounding(fitted, CONTRAST, grid=grid, random_state=4)
    assert [cell.treatment_strength for cell in surface.cells] == [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.15, 0.15),
        (0.15, 0.15),
    ]
    latent = _latent(fitted, surface.root_seed)
    data, options = seen[-1]
    expected = perturb_nodes(fitted.data, latent, (0.15, 0.15), ("binary", "binary"), False)
    assert np.array_equal(data.treatment, expected)
    assert np.allclose(data.outcome, fitted.data.outcome - 0.4 * latent, rtol=0, atol=1e-12)
    assert options["random_state"] == surface.refit_seed
    outcome_only, _ = seen[0]
    assert np.array_equal(outcome_only.treatment, fitted.data.treatment)


def test_a_dose_node_adds_the_scaled_latent(frame: pd.DataFrame) -> None:
    data = SimpleNamespace(treatment=frame[["D1", "D2"]].to_numpy(dtype=float))
    latent = np.linspace(-1.0, 1.0, len(frame))
    moved = perturb_nodes(data, latent, (0.5, -0.25), ("continuous", "continuous"), False)
    assert np.allclose(moved[:, 0], data.treatment[:, 0] + 0.5 * latent, rtol=0, atol=1e-12)
    assert np.allclose(moved[:, 1], data.treatment[:, 1] - 0.25 * latent, rtol=0, atol=1e-12)


def test_a_held_design_moves_node_one_and_the_identity_nodes_follow(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ltmle().fit(
        frame, outcome="Y", treatment="A", baseline=["W1", "W2"], time_varying=[[], []]
    )
    assert result.data.treatment_held
    seen = _capture(monkeypatch, result)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
    surface = simulated_confounding(result, CONTRAST, grid=grid, random_state=2)
    assert surface.complete
    assert surface.held_decision
    assert surface.treatment_components == (result.data.treatment_names[0],)
    assert any(line.startswith("held decision") for line in surface.population_lines())
    (data, _), *_ = seen
    latent = _latent(result, surface.root_seed)
    flipped = np.where(
        latent >= NormalDist().inv_cdf(0.8),
        1 - result.data.treatment[:, 0],
        result.data.treatment[:, 0],
    )
    assert np.array_equal(data.treatment[:, 0], flipped)
    assert np.array_equal(data.treatment[:, 1], flipped)
    # A held design names one treatment column, so a mapping keys that column alone.
    keyed = simulated_confounding(
        result,
        CONTRAST,
        grid=ConfounderStrengthGrid(treatment={"A": (0.0, 0.2)}, outcome=(0.0,)),
        random_state=2,
    )
    assert [cell.estimate for cell in keyed.cells] == [cell.estimate for cell in surface.cells]


def test_a_declared_first_node_mechanism_is_carried_flipped(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    g1 = 1 / (1 + np.exp(-frame["W1"].to_numpy()))
    declared_frame = frame.assign(p1_0=1 - g1, p1_1=g1)
    result = _ltmle().fit(
        declared_frame,
        **COLUMNS,
        treatment_probabilities={"A1": {0: "p1_0", 1: "p1_1"}},
    )
    assert result.data.known_mechanisms is not None
    seen = _capture(monkeypatch, result)
    grid = ConfounderStrengthGrid(treatment={"A1": (0.0, 0.2)}, outcome=(0.0,))
    surface = simulated_confounding(result, CONTRAST, grid=grid, random_state=2)
    assert surface.complete
    assert surface.known_mechanism_carried
    (data, _), *_ = seen
    declared = np.column_stack([1 - g1, g1])
    expected = 0.8 * declared + 0.2 * declared[:, ::-1]
    assert np.allclose(data.known_mechanisms.treatment[0], expected, rtol=0, atol=1e-12)
    # Mutation control: the unflipped declaration differs from the carried one.
    assert not np.allclose(data.known_mechanisms.treatment[0], declared)


def test_a_declared_later_node_mechanism_refuses_before_work(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ltmle().fit(
        frame.assign(p2_0=0.5, p2_1=0.5),
        **COLUMNS,
        treatment_probabilities={"A2": {0: "p2_0", 1: "p2_1"}},
    )
    forbid_draw_and_refit(monkeypatch, result.estimator)
    assert longitudinal_fit_refusal(result) == _KNOWN_LATER_NODE_REFUSAL
    with pytest.raises(CapabilityError) as caught:
        simulated_confounding(result, CONTRAST, grid=ConfounderStrengthGrid((0.0,), (0.0,)))
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert caught.value.refusal.roadmap == "F13"


# ------------------------------------------------------------------ every estimator


@pytest.mark.parametrize(
    "build",
    [
        lambda: _ltmle(),
        lambda: SDR(
            REGIMENS,
            n_folds=1,
            random_state=3,
            simultaneous=False,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        ),
        lambda: SequentialGComputation(
            REGIMENS, random_state=3, simultaneous=False, outcome_learner=LinearRegression()
        ),
        lambda: SequentialIPW(
            REGIMENS,
            random_state=3,
            simultaneous=False,
            treatment_learner=LogisticRegression(max_iter=1000),
        ),
        lambda: ParametricGFormula(REGIMENS, random_state=3, simultaneous=False),
    ],
    ids=["LTMLE", "SDR", "SequentialGComputation", "SequentialIPW", "ParametricGFormula"],
)
def test_every_covered_estimator_reproduces_and_moves(build: Any, frame: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = build().fit(frame, **COLUMNS)
        assert _fit_wide_refusal(result) is None
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        surface = simulated_confounding(result, CONTRAST, grid=grid, random_state=3)
    assert surface.complete
    assert surface.cells[0].estimate == result[CONTRAST].psi
    assert abs(surface.cells[1].estimate - surface.original_estimate) > 1e-3
    # The refit seam reproduces the fit at zero strength, so movement is the law's.
    replay = result.estimator.refit(
        result.data, random_state=3, frozen_seeds=_frozen_seeds(result, 3)
    )
    assert replay[CONTRAST].psi == pytest.approx(result[CONTRAST].psi, abs=1e-12)


def test_a_cross_fitted_bounded_fit_keeps_its_folds_and_its_bounds(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ltmle(n_folds=2, q_bounds=(-10.0, 10.0)).fit(frame, **COLUMNS)
    seeds = [item.seed for item in result.repeats]
    assert _frozen_seeds(result, 3) == seeds
    assert _frozen_seeds(result, 4) is None
    replay = result.estimator.refit(result.data, random_state=3, frozen_seeds=seeds)
    assert replay[CONTRAST].psi == pytest.approx(result[CONTRAST].psi, abs=1e-12)
    seen = _capture(monkeypatch, result)
    surface = simulated_confounding(
        result,
        CONTRAST,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 3.0)),
        random_state=3,
    )
    assert surface.bounded_outcome_law == "logit"
    assert surface.outcome_bounds == (-10.0, 10.0)
    ((data, options),) = seen
    assert options["frozen_seeds"] == seeds
    assert np.all((data.outcome > -10.0) & (data.outcome < 10.0))


# ------------------------------------------------------------------ calibration and report


def test_calibration_reads_each_node_and_the_outcome(fitted: Any) -> None:
    surface = simulated_confounding(
        fitted,
        CONTRAST,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("W1",),
        random_state=1,
    )
    rows = surface.calibrations
    assert [(row.role, row.component) for row in rows] == [
        ("treatment", "A1"),
        ("treatment", "A2"),
        ("outcome", None),
    ]
    from sklearn.preprocessing import StandardScaler

    design = StandardScaler().fit_transform(fitted.data.baseline)
    strength, _ = _binary_calibration(
        design, fitted.data.treatment[:, 1], [0], np.asarray(fitted.data.weights)
    )
    assert rows[1].strength == strength
    frame = surface.calibration_frame()
    components = list(frame["component"])
    assert components[:2] == ["A1", "A2"]
    assert pd.isna(components[2])


def test_the_frame_and_summary_carry_one_column_per_node(fitted: Any) -> None:
    surface = simulated_confounding(
        fitted,
        CONTRAST,
        grid=ConfounderStrengthGrid(treatment={"A2": (0.0, 0.1)}, outcome=(0.0,)),
        random_state=1,
    )
    columns = list(surface.to_frame().columns)
    for name in ("A1", "A2"):
        assert f"treatment_strength[{name}]" in columns
        assert f"induced_treatment_association[{name}]" in columns
    assert [cell.treatment_strength for cell in surface.cells] == [(0.0, 0.0), (0.0, 0.1)]
    text = surface.summary()
    assert "treatment strength (A1, A2)" in text
    assert "path through the earlier perturbed nodes" in text


def test_the_assessment_row_is_available_and_asks_for_the_estimand(fitted: Any) -> None:
    capability = fitted.sensitivity.capability("simulated_confounding")
    assert capability.available
    assert capability.requires_arguments == ("grid", "estimand")
    assert simulated_confounding_refusal(fitted, CONTRAST) is None
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
    facade = fitted.sensitivity.simulated_confounding(estimand=CONTRAST, grid=grid, random_state=3)
    direct = simulated_confounding(fitted, CONTRAST, grid=grid, random_state=3)
    assert [cell.estimate for cell in facade.cells] == [cell.estimate for cell in direct.cells]


# ------------------------------------------------------------------ request checks


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"estimand": "ate"}, "was not requested in this fit"),
        ({"arm_pair": (0, 1)}, "pairwise swap of a multi-arm point treatment"),
        ({"latent_level": "cluster"}, "declares no cluster"),
        ({"observation_reads": ("W1",)}, "observes every node and the outcome"),
        ({"bounded_outcome_law": "logit"}, "declares none"),
        ({"benchmark_covariates": ("A1",)}, "baseline covariates of this panel"),
        ({"grid": ConfounderStrengthGrid({"B": (0.0, 0.1)}, (0.0,))}, "not treatment nodes"),
        ({"grid": ConfounderStrengthGrid((0.0, 0.6), (0.0,))}, "between 0 and 0.5"),
        (
            {"grid": ConfounderStrengthGrid((0.0,), (0.0,), intermediate=(0.0, 0.1))},
            "declares no intermediate",
        ),
    ],
)
def test_a_malformed_request_fails_before_work(
    fitted: Any, monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any], fragment: str
) -> None:
    forbid_draw_and_refit(monkeypatch, fitted.estimator)
    call: dict[str, Any] = {
        "estimand": CONTRAST,
        "grid": ConfounderStrengthGrid((0.0, 0.1), (0.0,)),
    }
    call.update(arguments)
    with pytest.raises(ValueError, match=fragment):
        simulated_confounding(fitted, **call)


# ------------------------------------------------------------------ refusals


def _refused(result: Any, constant: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert longitudinal_fit_refusal(result) == constant
    assert _fit_wide_refusal(result) == constant
    assert result.sensitivity.capability("simulated_confounding").refusal == constant
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError) as caught:
        simulated_confounding(
            result, next(iter(result.estimates)), grid=ConfounderStrengthGrid((0.0,), (0.0,))
        )
    assert str(caught.value) == constant.message()


def test_censoring_refuses_before_work(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _ = make_longitudinal(n=240, seed=4)
    result = _ltmle(censoring_learner=LogisticRegression(max_iter=1000)).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    _refused(result, _CENSORING_REFUSAL, monkeypatch)


def test_a_survival_outcome_refuses_before_work(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _ = make_longitudinal_survival(n=300, seed=2)
    result = _ltmle(censoring_learner=LogisticRegression(max_iter=1000)).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    _refused(result, _OUTCOME_KIND_REFUSAL, monkeypatch)


def test_a_covariate_after_the_first_node_refuses_before_work(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ltmle().fit(
        frame.assign(L2=frame["W1"] + frame["A1"]),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
    )
    _refused(result, _LATER_COVARIATE_REFUSAL, monkeypatch)


def test_an_ipsi_fit_refuses_before_work(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cleverly.interventions import Incremental

    result = IPSI(
        [Incremental(1.0), Incremental(2.0)],
        n_folds=1,
        random_state=3,
        simultaneous=False,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ).fit(frame, **COLUMNS)
    _refused(result, _IPSI_REFUSAL, monkeypatch)


def test_a_clustered_panel_refuses_before_work(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ltmle().fit(frame.assign(G=np.arange(len(frame)) // 4), **COLUMNS, id="G")
    _refused(result, _CLUSTERED_REFUSAL, monkeypatch)


def test_a_node_with_more_than_two_levels_refuses_before_work(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(1)
    three = frame.assign(A2=rng.integers(0, 3, len(frame)).astype(float))
    result = LTMLE(
        {"low": [0, 0], "high": [1, 2]},
        n_folds=1,
        random_state=3,
        simultaneous=False,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ).fit(three, **COLUMNS)
    _refused(result, _NODE_LAW_REFUSAL, monkeypatch)


def test_a_vector_node_with_a_dose_is_refused_by_the_node_law() -> None:
    data = SimpleNamespace(
        n_times=2,
        node_kind=lambda time: "mixed" if time == 2 else "categorical",
        treatment_levels=((0, 1), ()),
    )
    assert _refuse_node_law(SimpleNamespace(data=data)) == _NODE_LAW_REFUSAL


def test_a_shapeless_longitudinal_object_refuses_as_a_defect() -> None:
    probe = SimpleNamespace(assessment_family="longitudinal")
    refusal = _fit_wide_refusal(probe)
    assert refusal is not None
    assert refusal.reason is RefusalReason.INVALID_DECLARATION
    with pytest.raises(CapabilityError, match="fitted LongitudinalResult"):
        simulated_confounding(probe, grid=ConfounderStrengthGrid((0.0,), (0.0,)))
