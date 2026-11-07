"""The risk under eliminated competing events, checked exactly on a finite law.

``tests/discrete_law_point_survival.py`` realises the two-cause law exactly, so a saturated
fit is the oracle.  The parameter is the risk of relapse had death been prevented, Young
et al. (2020) g-formula (23), with the order censoring, death, relapse within a node.  On
every engine the estimate equals that g-formula, and on every engine that reports an
influence curve the curve equals its Gateaux derivative.

The point estimate of a targeted engine is the g-formula whatever retention enters its
clever covariate, because the outcome fit is the oracle.  The competing factor is
therefore checked through the influence curve, and the mutations below must each break it.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import DataError
from cleverly.learners import GLM
from cleverly.longitudinal import (
    LTMLE,
    SDR,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.longitudinal import sequential as sequential_module
from cleverly.longitudinal.data import LongitudinalData
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    PointLaw,
    static,
    survival_law,
)

REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
HORIZONS = (1, 2, 3)
RELAPSE, DEATH = 1, 2
#: Cumulative bounds that never bind on these laws, so the influence curve is the EIF.
NO_TRUNCATION = (1e-9, 1.0)
#: A saturated GLM is the nonparametric model on this support.
SATURATED = GLM(interactions=8, tol=1e-13)
#: How far a mutated curve must sit from the EIF, far above the rounding of the identities.
MUTATION_FLOOR = 1e-3


def _data(law: PointLaw) -> LongitudinalData:
    return LongitudinalData.from_frame(law.frame(), eliminate=["death"], **law.fit_columns())


def _cells() -> dict[str, Any]:
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }


def _engines() -> dict[str, Any]:
    return {
        "ltmle": lambda: LTMLE(REGIMENS, n_folds=1, g_bounds=NO_TRUNCATION, **_cells()),
        "ltmle_hazard": lambda: LTMLE(
            REGIMENS, n_folds=1, survival_model="hazard", g_bounds=NO_TRUNCATION, **_cells()
        ),
        "sdr": lambda: SDR(REGIMENS, n_folds=1, g_bounds=NO_TRUNCATION, **_cells()),
        "ipw_hajek": lambda: SequentialIPW(
            REGIMENS,
            treatment_learner=SATURATED,
            censoring_learner=SATURATED,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
            normalization="hajek",
        ),
        "ipw_horvitz_thompson": lambda: SequentialIPW(
            REGIMENS,
            treatment_learner=SATURATED,
            censoring_learner=SATURATED,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
            normalization="horvitz_thompson",
        ),
        "gcomp": lambda: SequentialGComputation(
            REGIMENS,
            outcome_learner=SATURATED,
            treatment_learner=SATURATED,
            censoring_learner=SATURATED,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
        ),
    }


#: The engines and the laws they fit: the hazard path reads baseline covariates only.
CASES = [
    (engine, l2)
    for engine in _engines()
    for l2 in (False, True)
    if not (l2 and engine == "ltmle_hazard")
]


def _law(l2: bool) -> PointLaw:
    return survival_law(causes=2, with_l2=l2)


def _fit(engine: str, law: PointLaw) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _engines()[engine]().fit(_data(law))


def _name(label: str, horizon: int) -> str:
    return f"risk_regimen[{label} @ t={horizon}]"


def _curve(result: Any, name: str) -> np.ndarray:
    curves = getattr(result, "influence_curves", None)
    if curves is not None:
        return np.asarray(curves[name])
    return np.asarray(result[name].influence_curve)


def _psi(result: Any, name: str) -> float:
    return float(result.psi(name)) if callable(getattr(result, "psi", None)) else result[name].psi


def _worst_curve_gap(law: PointLaw, result: Any) -> float:
    rows = law.first_row_of()
    worst = 0.0
    for label, arm in REGIMENS.items():
        for horizon in HORIZONS:
            eif = law.eif(static(arm), horizon, RELAPSE, eliminate=DEATH)
            gap = np.max(np.abs(_curve(result, _name(label, horizon))[rows] - eif))
            worst = max(worst, float(gap))
    return worst


@pytest.fixture(scope="module")
def fits() -> dict[tuple[str, bool], Any]:
    return {(engine, l2): _fit(engine, _law(l2)) for engine, l2 in CASES}


@pytest.mark.parametrize(("engine", "l2"), CASES)
def test_the_estimate_is_the_eliminated_g_formula(
    fits: dict[tuple[str, bool], Any], engine: str, l2: bool
) -> None:
    law, result = _law(l2), fits[(engine, l2)]
    for label, arm in REGIMENS.items():
        for horizon in HORIZONS:
            truth = law.functional(law.probs, static(arm), horizon, RELAPSE, eliminate=DEATH)
            assert _psi(result, _name(label, horizon)) == pytest.approx(truth, abs=1e-10)


@pytest.mark.parametrize(("engine", "l2"), CASES)
def test_the_influence_curve_is_the_gateaux_derivative(
    fits: dict[tuple[str, bool], Any], engine: str, l2: bool
) -> None:
    if engine == "gcomp":
        # The stacked g-computation curve reads no mechanism; it is checked here too, as the
        # chain of its outcome GLMs.
        pass
    assert _worst_curve_gap(_law(l2), fits[(engine, l2)]) < 1e-9


@pytest.mark.parametrize("l2", [False, True])
def test_the_eliminated_risk_is_not_the_total_effect_incidence(l2: bool) -> None:
    """Witness: the estimand differs from the cause-specific incidence on this law."""
    law = _law(l2)
    for arm in REGIMENS.values():
        for horizon in HORIZONS:
            eliminated = law.functional(law.probs, static(arm), horizon, RELAPSE, eliminate=DEATH)
            total = law.functional(law.probs, static(arm), horizon, RELAPSE)
            assert eliminated - total > 1e-2


def test_the_competing_factor_is_fitted_on_the_rows_that_passed_censoring() -> None:
    data = _data(_law(False))
    assert data.retained is not None
    left = data.retained & ~data.uncensored
    assert left.any(), "the law must hold prevented deaths"
    result = _fit("ltmle", _law(False))
    mechanism = result.mechanism
    assert len(mechanism.competing) == data.n_times
    assert mechanism.no_competing_nodes == ()


# ---------------------------------------------------------------- mutation controls


def _mutated_gap(monkeypatch: pytest.MonkeyPatch, engine: str, law: PointLaw) -> float:
    return _worst_curve_gap(law, _fit(engine, law))


@pytest.mark.parametrize("engine", ["ltmle", "ltmle_hazard", "sdr", "ipw_hajek"])
def test_dropping_the_competing_factor_breaks_the_curve(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    """Mutation (ii): the retention without the competing factor."""
    original = sequential_module._retention_factor

    def without_competing(*args: Any, **kwargs: Any) -> Any:
        factor = original(*args, **kwargs)
        if kwargs.get("key") != "observed_competing":
            return factor
        ones = np.ones_like(factor.observed)
        return replace(
            factor,
            by_plan=dict.fromkeys(factor.by_plan, ones),
            observed=ones,
        )

    monkeypatch.setattr(sequential_module, "_retention_factor", without_competing)
    assert _mutated_gap(monkeypatch, engine, _law(False)) > MUTATION_FLOOR


@pytest.mark.parametrize("engine", ["ltmle", "ltmle_hazard", "sdr", "ipw_hajek"])
def test_counting_the_competing_event_twice_breaks_the_curve(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    """Mutation (iii): the censoring factor fitted to the composite, times the competing one."""
    monkeypatch.setattr(
        LongitudinalData, "censoring_target", property(lambda self: self.uncensored)
    )
    assert _mutated_gap(monkeypatch, engine, _law(False)) > MUTATION_FLOOR


def test_dropping_the_ipw_competing_block_breaks_the_stacked_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation (iv): the stacked IPW curve without the competing factor's score block."""
    original = sequential_module.fit_mechanism

    def without_models(*args: Any, **kwargs: Any) -> Any:
        mechanism = original(*args, **kwargs)
        return replace(mechanism, competing_models=(None,) * len(mechanism.competing_models))

    monkeypatch.setattr(sequential_module, "fit_mechanism", without_models)
    from cleverly.longitudinal import baselines, estimator

    if hasattr(baselines, "fit_mechanism"):
        monkeypatch.setattr(baselines, "fit_mechanism", without_models)
    if hasattr(estimator, "fit_mechanism"):
        monkeypatch.setattr(estimator, "fit_mechanism", without_models)
    assert _mutated_gap(monkeypatch, "ipw_hajek", _law(False)) > MUTATION_FLOOR


def test_the_other_within_node_order_misses_the_truth() -> None:
    """Mutation (i): a prevented death stays in its node's risk set with a zero event."""
    law = _law(False)
    data = _data(law)
    assert data.retained is not None
    prevented = data.retained & ~data.uncensored
    first = prevented & np.concatenate(
        [np.ones((data.n, 1), dtype=bool), ~prevented[:, :-1]], axis=1
    )
    # Uncensored at its own node (in the risk set of that node's event), out after it.
    uncensored = data.uncensored | first
    mutated = replace(data, uncensored=uncensored, retained=data.retained)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _engines()["ltmle"]().fit(mutated)
    worst = 0.0
    for label, arm in REGIMENS.items():
        for horizon in HORIZONS:
            truth = law.functional(law.probs, static(arm), horizon, RELAPSE, eliminate=DEATH)
            worst = max(worst, abs(_psi(result, _name(label, horizon)) - truth))
    assert worst > MUTATION_FLOOR


# ------------------------------------------------------------------ reductions


def test_with_no_prevented_event_the_fit_is_the_total_effect_fit() -> None:
    """A sample with no death: the eliminated fit equals the fit of relapse alone."""
    law = _law(False)
    frame = law.frame()
    frame = frame[(frame[[f"D{k}" for k in (1, 2, 3)]].fillna(0) == 0).all(axis=1)]
    columns = law.fit_columns()
    eliminated = LongitudinalData.from_frame(frame, eliminate=["death"], **columns)
    single = {**columns, "outcome": columns["outcome"]["relapse"]}
    plain = LongitudinalData.from_frame(frame, **single)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        left = _engines()["ltmle"]().fit(eliminated)
        right = _engines()["ltmle"]().fit(plain)
    for label in REGIMENS:
        for horizon in HORIZONS:
            name = _name(label, horizon)
            np.testing.assert_allclose(left[name].psi, right[name].psi, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(
                left[name].influence_curve, right[name].influence_curve, rtol=1e-10, atol=1e-12
            )


def test_a_hand_built_composite_censoring_column_gives_the_same_estimate() -> None:
    """The documented workaround: fold death into censoring by hand, fit relapse alone."""
    law = _law(False)
    frame = law.frame().copy()
    for node in (1, 2, 3):
        died = frame[f"D{node}"] == 1
        frame.loc[died, f"C{node}"] = 0.0
        frame.loc[died, [f"R{node}", f"D{node}"]] = np.nan
    columns = law.fit_columns()
    single = {**columns, "outcome": columns["outcome"]["relapse"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_hand = _engines()["ltmle"]().fit(frame, **single)
        declared = _fit("ltmle", law)
    for label in REGIMENS:
        for horizon in HORIZONS:
            name = _name(label, horizon)
            assert by_hand[name].psi == pytest.approx(declared[name].psi, abs=1e-10)


# ------------------------------------------------------------------- refusals


def test_eliminate_needs_competing_causes() -> None:
    law = survival_law()
    with pytest.raises(DataError, match="declares none"):
        LongitudinalData.from_frame(law.frame(), eliminate=["death"], **law.fit_columns())


def test_eliminate_refuses_an_unknown_cause() -> None:
    law = _law(False)
    with pytest.raises(DataError, match="unknown cause"):
        LongitudinalData.from_frame(law.frame(), eliminate=["stroke"], **law.fit_columns())


def test_eliminate_refuses_every_cause() -> None:
    law = _law(False)
    with pytest.raises(DataError, match="names every cause"):
        LongitudinalData.from_frame(
            law.frame(), eliminate=["death", "relapse"], **law.fit_columns()
        )


def test_eliminating_twice_is_refused() -> None:
    data = LongitudinalData.from_frame(_law(False).frame(), **_law(False).fit_columns())
    once = data.eliminating(["death"])
    with pytest.raises(DataError, match="already eliminates"):
        once.eliminating(["relapse"])


def test_the_recode_is_carried_by_subset_and_separates_rows() -> None:
    data = _data(_law(False))
    assert data.retained is not None
    half = data.subset(np.arange(data.n // 2))
    assert half.retained is not None
    assert np.array_equal(half.retained, data.retained[: data.n // 2])
    assert half.eliminated == ("death",)


# ------------------------------------------------------------------ study path


def test_the_study_path_fits_an_eliminated_time_to_event_design() -> None:
    from cleverly import CausalStudy, RegimeContrast, TimeToEvent

    law = survival_law(causes=2, censor_first=False)
    grid = (1.0, 2.0, 3.0)
    frame = law.long_frame(grid, offset=0.5)
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=grid,
        causes={1: "relapse", 2: "death"},
        eliminate=["death"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CausalStudy(frame, design=design).estimate(
            RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0", horizons=[3.0]),
            outcome_learner=CellMeans(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
            cross_fit=False,
        )
    truth = law.functional(law.probs, static(1), 3, RELAPSE, eliminate=DEATH) - law.functional(
        law.probs, static(0), 3, RELAPSE, eliminate=DEATH
    )
    assert result["ate_regimen[a1 vs a0 @ t=3]"].psi == pytest.approx(truth, abs=1e-10)


def test_a_design_refuses_a_container_that_already_eliminates() -> None:
    from cleverly import CausalStudy, LongitudinalTreatment

    law = _law(False)
    columns = law.fit_columns()
    design = LongitudinalTreatment(eliminate=["death"], **columns)
    with pytest.raises(DataError, match="already eliminates"):
        CausalStudy(_data(law), design=design)


# ------------------------------------------------------------------- refute


@pytest.mark.parametrize("name", ["dummy_outcome", "simulated_outcome", "plasmode"])
def test_refute_refuses_a_generated_outcome_on_an_eliminated_fit(name: str) -> None:
    from cleverly.validation.outcome_processes import MultinomialHazardOutcome
    from cleverly.validation.refute import refute_refusal

    process = MultinomialHazardOutcome(intercepts=(-1.5,), effects=(0.8,))
    keyword = {"plasmode": "simulated_outcome"}.get(name, name)
    refusal = refute_refusal(
        _fit("ltmle", _law(False)),
        estimand="ate_regimen[a1 vs a0 @ t=3]",
        tests=(name,),
        n_replicates=400,
        **{keyword: process},
    )
    assert refusal is not None
    assert "eliminates death" in refusal.condition
    assert refusal.roadmap == "X91"


# ------------------------------------------------------------------- summary


def test_the_setup_summary_reports_prevented_events_apart_from_censoring() -> None:
    from cleverly import CausalStudy, LongitudinalTreatment
    from cleverly._summary import vocabulary

    law = _law(False)
    frame = law.frame()
    design = LongitudinalTreatment(eliminate=["death"], **law.fit_columns())
    header = dict(CausalStudy(frame, design=design).summary().header)
    data = _data(law)
    times, codes = data.to_time_to_event()
    lost = (codes == 0) & (times < data.n_times)
    assert data.retained is not None
    kept = np.cumprod(data.retained, axis=1, dtype=bool)
    observed = np.cumprod(data.uncensored, axis=1, dtype=bool)
    prevented = lost & (kept & ~observed).any(axis=1)
    # The witness: the law has deaths, so the two shares differ from the pooled one.
    assert prevented.mean() > 0.05
    assert header[vocabulary.PREVENTED_BEFORE_END] == f"{prevented.mean():.3f}"
    assert header[vocabulary.CENSORED_BEFORE_END] == f"{(lost & ~prevented).mean():.3f}"
