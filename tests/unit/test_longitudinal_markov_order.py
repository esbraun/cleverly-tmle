"""``LongitudinalTreatment(markov_order=k)``: windowed nuisance designs (X33 part e).

The order ``k`` restricts the observed-data model.  At node ``t`` the mechanisms read
``W, L_{t-k}..L_t, A_{t-k-1}..A_{t-1}`` (plus ``A_t`` for censoring), and the outcome
regression reads ``W, L_{t-k}..L_t`` plus the rule and policy blocks of the nodes in
``[t-k, t]``.  The frames a rule or a policy reads keep the full history.

The exact tests check each design's columns against the source columns.  The witnesses run
on laws that a sample realises exactly (:mod:`tests.unit._markov_law_support`).  At
``P_n = P_0`` a fit returns the truth when either nuisance family is right, so a check on
the outcome side holds the mechanism wrong, and a check on the mechanism side holds the
outcome regression wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, LongitudinalData
from cleverly.longitudinal import sequential as sequential_module
from cleverly.longitudinal.data import markov_restriction
from cleverly.longitudinal.regimen import resolve_plans
from cleverly.longitudinal.sequential import outcome_design
from tests.discrete_law_longitudinal import CellMeans
from tests.unit import _markov_law_support as law

EXACT = 1e-12
NODES = 4


def _panel(seed: int = 0, n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns: dict[str, Any] = {"W": rng.normal(size=n)}
    for t in range(1, NODES + 1):
        columns[f"L{t}"] = rng.normal(size=n)
        columns[f"A{t}"] = rng.integers(0, 2, size=n).astype(float)
    columns["Y"] = rng.normal(size=n)
    return pd.DataFrame(columns)


def _design(order: int | None, **extra: Any) -> LongitudinalTreatment:
    return LongitudinalTreatment(
        outcome="Y",
        treatment=[f"A{t}" for t in range(1, NODES + 1)],
        baseline=["W"],
        time_varying=[[f"L{t}"] for t in range(1, NODES + 1)],
        markov_order=order,
        **extra,
    )


def _columns(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    return frame[names].to_numpy(dtype=float)


def _window(time: int, order: int) -> range:
    return range(max(1, time - order), time + 1)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


class TestTheWindows:
    """Each call site's design holds exactly the window's columns, in design order."""

    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("time", range(1, NODES + 1))
    def test_the_mechanism_reads_w_the_window_and_the_arms_before_it(
        self, panel: pd.DataFrame, order: int, time: int
    ) -> None:
        data = _design(order).prepare(panel)
        nodes = _window(time, order)
        covariates = ["W", *(f"L{s}" for s in nodes)]
        arms = [f"A{s}" for s in range(max(1, nodes[0] - 1), time)]
        np.testing.assert_array_equal(data.history_design(time), _columns(panel, covariates + arms))
        np.testing.assert_array_equal(
            data.history_design(time, include_current=True),
            _columns(panel, [*covariates, *arms, f"A{time}"]),
        )

    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("time", range(1, NODES + 1))
    def test_a_static_plan_adds_no_treatment_block(
        self, panel: pd.DataFrame, order: int, time: int
    ) -> None:
        data = _design(order).prepare(panel)
        (plan,) = resolve_plans(_regimens({"always": 1}), data)
        expected = ["W", *(f"L{s}" for s in _window(time, order))]
        np.testing.assert_array_equal(outcome_design(data, plan, time), _columns(panel, expected))

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_a_rule_node_in_the_window_adds_its_assigned_arm(
        self, panel: pd.DataFrame, order: int
    ) -> None:
        data = _design(order).prepare(panel)
        rule = DynamicRegimen("rule", (1, _sign_of("L1"), 0, _sign_of("L1")), rule_kind="known")
        (plan,) = resolve_plans((rule,), data)
        assigned = (panel["L1"].to_numpy() > 0).astype(float)
        for time in range(1, NODES + 1):
            nodes = _window(time, order)
            expected = [_columns(panel, ["W", *(f"L{s}" for s in nodes)])]
            expected += [assigned[:, None] for s in nodes if s in (2, 4)]
            np.testing.assert_array_equal(outcome_design(data, plan, time), np.hstack(expected))

    def test_a_policy_node_keeps_its_block_only_inside_the_window(
        self, panel: pd.DataFrame
    ) -> None:
        data = _design(1).prepare(panel)
        half = Stochastic(lambda h: np.full((len(h), 2), 0.5), "half", density_kind="known")
        mix = DynamicRegimen("mix", (half, 1, half, 1))
        (plan,) = resolve_plans((mix,), data)
        base = _columns(panel, ["W", "L3", "L4"])
        np.testing.assert_array_equal(
            outcome_design(data, plan, 4), np.hstack([base, _columns(panel, ["A3"])])
        )
        np.testing.assert_array_equal(
            outcome_design(data, plan, 3, arm=1),
            np.hstack([_columns(panel, ["W", "L2", "L3"]), np.ones((data.n, 1))]),
        )
        full = _design(None).prepare(panel)
        (unrestricted,) = resolve_plans((mix,), full)
        np.testing.assert_array_equal(
            outcome_design(full, unrestricted, 4),
            np.hstack(
                [_columns(panel, ["W", "L1", "L2", "L3", "L4"]), _columns(panel, ["A1", "A3"])]
            ),
        )

    def test_the_frames_keep_the_full_history(self, panel: pd.DataFrame) -> None:
        restricted = _design(0).prepare(panel)
        full = _design(None).prepare(panel)
        for time in range(1, NODES + 1):
            pd.testing.assert_frame_equal(restricted.history_frame(time), full.history_frame(time))
            pd.testing.assert_frame_equal(restricted.policy_frame(time), full.policy_frame(time))

    def test_no_order_keeps_every_design_bit_for_bit(self, panel: pd.DataFrame) -> None:
        data = _design(None).prepare(panel)
        (plan,) = resolve_plans(_regimens({"always": 1}), data)
        for time in range(1, NODES + 1):
            np.testing.assert_array_equal(
                outcome_design(data, plan, time), data.covariate_history(time)
            )
            assert data.window_covariates(time) is not None
            np.testing.assert_array_equal(
                data.window_covariates(time), data.covariate_history(time)
            )

    def test_a_row_subset_keeps_the_order(self, panel: pd.DataFrame) -> None:
        data = _design(1).prepare(panel)
        assert data.subset(np.arange(40)).markov_order == 1


def _sign_of(name: str) -> Any:
    return lambda h: (np.asarray(h[name], dtype=float) > 0).astype(float)


def _regimens(spec: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(DynamicRegimen(label, (arm,) * NODES) for label, arm in spec.items())


# ------------------------------------------------------------------ exact-law witnesses

THREE = LongitudinalTreatment(
    outcome="Y",
    treatment=["A1", "A2", "A3"],
    baseline=["W"],
    time_varying=[["L1"], ["L2"], ["L3"]],
)


def _rule_of_l1() -> Any:
    return lambda h: 1 - np.asarray(h["L1"], dtype=float)


PLANS: dict[str, tuple[Any, Any, Any]] = {
    "never": (0, 0, 0),
    "always": (1, 1, 1),
    "rule": (1, 1, _rule_of_l1()),
}


def _declared_regimens() -> dict[str, Any]:
    return {
        label: (
            DynamicRegimen(label, plan, rule_kind="known")
            if any(callable(node) for node in plan)
            else plan[0]
        )
        for label, plan in PLANS.items()
    }


def _ltmle(
    *, mechanism: str = "cell", outcome: str = "cell", estimator: type[LTMLE] | type[SDR] = LTMLE
) -> LTMLE | SDR:
    def g() -> Any:
        return CellMeans() if mechanism == "cell" else DummyClassifier(strategy="prior")

    def q() -> Any:
        return CellMeans() if outcome == "cell" else DummyClassifier(strategy="prior")

    return estimator(
        _declared_regimens(),
        reference="never",
        n_folds=1,
        random_state=0,
        outcome_learner=q(),
        pseudo_learner=CellMeans() if outcome == "cell" else _MeanRegressor(),
        treatment_learner=g(),
        censoring_learner=g(),
        simultaneous=False,
    )


class _MeanRegressor(DummyClassifier):
    """The unconditional mean of a fractional pseudo-outcome: a wrong outcome regression."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _MeanRegressor:
        self.mean_ = float(np.average(np.asarray(y, dtype=float), weights=sample_weight))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(np.asarray(X)), self.mean_)


def _errors(
    name: str, order: int | None, estimator: type[LTMLE] | type[SDR] = LTMLE, **learners: str
) -> dict[str, float]:
    frame = law.frame(name)
    data = LongitudinalTreatment(
        outcome=THREE.outcome,
        treatment=THREE.treatment,
        baseline=THREE.baseline,
        time_varying=THREE.time_varying,
        markov_order=order,
    ).prepare(frame)
    result = _ltmle(estimator=estimator, **learners).fit(data)
    return {
        label: result.fits[label].psi_scaled - law.truth(name, plan)
        for label, plan in PLANS.items()
    }


def _exact(errors: dict[str, float]) -> bool:
    return max(abs(value) for value in errors.values()) < EXACT


class TestTheMarkovOrderOneLaw:
    """Every nuisance reads the order-1 window, and both families read past order 0."""

    def test_order_one_returns_the_truth(self) -> None:
        assert _exact(_errors("markov1", 1))

    def test_order_one_returns_the_truth_through_each_family_alone(self) -> None:
        assert _exact(_errors("markov1", 1, mechanism="wrong"))
        assert _exact(_errors("markov1", 1, outcome="wrong"))

    def test_order_zero_misses_the_truth(self) -> None:
        errors = _errors("markov1", 0)
        assert min(abs(value) for value in errors.values()) > 1e-4


class TestTheOneFamilyLaw:
    """The mechanisms read the order-0 window, and the outcome reads ``L1`` outside it."""

    def test_order_zero_returns_the_truth_through_the_mechanisms(self) -> None:
        assert _exact(_errors("one_family", 0, outcome="wrong"))

    def test_order_zero_outcome_regressions_alone_miss_the_truth(self) -> None:
        errors = _errors("one_family", 0, mechanism="wrong")
        assert max(abs(value) for value in errors.values()) > 1e-3


class TestSDR:
    """``SDR`` reads the same windows: its node regressions and mechanisms share the designs."""

    def test_order_one_returns_the_truth_through_each_family_alone(self) -> None:
        assert _exact(_errors("markov1", 1, SDR, mechanism="wrong"))
        assert _exact(_errors("markov1", 1, SDR, outcome="wrong"))

    def test_order_zero_misses_the_truth(self) -> None:
        errors = _errors("markov1", 0, SDR)
        assert min(abs(value) for value in errors.values()) > 1e-4

    def test_order_zero_returns_the_truth_through_the_mechanisms(self) -> None:
        assert _exact(_errors("one_family", 0, SDR, outcome="wrong"))

    def test_dropping_the_rule_blocks_misses_the_truth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sequential_module, "_is_rule_node", lambda plan, node: False)
        errors = _errors("markov1", 1, SDR, mechanism="wrong")
        assert abs(errors["rule"]) > 1e-3
        assert abs(errors["always"]) < EXACT


class TestMutationControls:
    """Each mutant breaks one window rule, and an exact-law check fails."""

    def test_a_window_one_node_short_misses_the_truth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _exact(_errors("markov1", 1))

        def short(self: LongitudinalData, time: int) -> int:
            return 1 if self.markov_order is None else max(1, time - self.markov_order + 1)

        monkeypatch.setattr(LongitudinalData, "window_start", short)
        assert not _exact(_errors("markov1", 1))

    def test_dropping_the_rule_blocks_misses_the_truth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _exact(_errors("markov1", 1, mechanism="wrong"))
        monkeypatch.setattr(sequential_module, "_is_rule_node", lambda plan, node: False)
        errors = _errors("markov1", 1, mechanism="wrong")
        assert abs(errors["rule"]) > 1e-3
        assert abs(errors["always"]) < EXACT


class TestRefusals:
    @pytest.mark.parametrize("order", [-1, 1.5, True, "1"])
    def test_an_order_that_is_not_a_non_negative_integer(self, order: Any) -> None:
        with pytest.raises(ValueError, match="markov_order must be a non-negative integer"):
            _design(order)

    @pytest.mark.parametrize("order", [NODES - 1, NODES + 3])
    def test_an_order_that_keeps_every_node(self, panel: pd.DataFrame, order: int) -> None:
        with pytest.raises(DataError, match=f"markov_order={order} keeps every earlier node"):
            _design(order).prepare(panel)

    def test_a_design_with_no_history_to_restrict(self, panel: pd.DataFrame) -> None:
        held = LongitudinalTreatment(
            outcome=["Y1", "Y2", "Y3"],
            treatment="A1",
            baseline=["W"],
            markov_order=0,
        )
        frame = panel.assign(Y1=0.0, Y2=0.0, Y3=(panel["Y"] > 0).astype(float))
        with pytest.raises(CapabilityError, match="this design has none"):
            held.prepare(frame)

    def test_a_prepared_container_takes_the_declared_order(self, panel: pd.DataFrame) -> None:
        plain = _design(None).prepare(panel)
        assert _design(1).prepare(plain).markov_order == 1
        restricted = _design(1).prepare(panel)
        with pytest.raises(DataError, match="markov_order"):
            _design(0).prepare(restricted)


def test_the_study_path_fits_the_windowed_designs() -> None:
    frame = law.frame("markov1")
    design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[["L1"], ["L2"], ["L3"]],
        markov_order=1,
    )
    result = (
        CausalStudy(frame, design=design)
        .identify(RegimeMean({"always": 1}))
        .estimate(
            outcome_learner=CellMeans(),
            pseudo_learner=CellMeans(),
            treatment_learner=CellMeans(),
            censoring_learner=CellMeans(),
            cross_fit=False,
        )
    )
    (estimate,) = result.estimates.values()
    assert estimate.psi == pytest.approx(law.truth("markov1", (1, 1, 1)), abs=EXACT)


def test_the_restriction_text_is_pinned() -> None:
    assert markov_restriction(1) == (
        "statistical-model restriction, Markov order 1: at every node the outcome regression "
        "and the treatment and censoring mechanisms depend on the history only through W and "
        "the last 2 nodes. It restricts the observed-data model and does not change the "
        "estimand. The fit is consistent when the outcome regressions fitted on the window "
        "are consistent, which needs the restriction for them, or when the treatment and "
        "censoring mechanisms fitted on the window are consistent, which needs the restriction "
        "for them. The interval needs the restriction for both families and both families "
        "consistent at the usual product rate."
    )


def test_the_parametric_g_formula_refuses_a_markov_order() -> None:
    from cleverly.longitudinal import ParametricGFormula
    from cleverly.longitudinal.estimator import GFORMULA_MARKOV_REFUSAL

    data = LongitudinalTreatment(
        outcome=THREE.outcome,
        treatment=THREE.treatment,
        baseline=THREE.baseline,
        time_varying=THREE.time_varying,
        markov_order=1,
    ).prepare(law.frame("markov1"))
    with pytest.raises(CapabilityError) as refused:
        ParametricGFormula({"never": 0, "always": 1}).fit(data)
    assert refused.value.refusal == GFORMULA_MARKOV_REFUSAL


def _gcomp_errors(order: int) -> list[float]:
    from cleverly.longitudinal import SequentialGComputation

    data = LongitudinalTreatment(
        outcome=THREE.outcome,
        treatment=THREE.treatment,
        baseline=THREE.baseline,
        time_varying=THREE.time_varying,
        markov_order=order,
    ).prepare(law.frame("markov1"))
    result = SequentialGComputation(
        _declared_regimens(),
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        simultaneous=False,
    ).fit(data)
    return [
        abs(result.fits[label].psi_scaled - law.truth("markov1", plan))
        for label, plan in PLANS.items()
    ]


def test_sequential_g_computation_reads_the_window() -> None:
    """The order-1 window returns the truth on the order-1 law, and the order-0 window misses."""
    assert max(_gcomp_errors(1)) < EXACT
    assert max(_gcomp_errors(0)) > 1e-3
