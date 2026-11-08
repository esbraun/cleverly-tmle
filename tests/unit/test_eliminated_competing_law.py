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
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, DataError
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


def test_the_competing_factor_is_fitted_on_the_rows_that_passed_censoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data(_law(False))
    assert data.retained is not None
    assert (data.retained & ~data.uncensored).any(), "the law must hold prevented deaths"
    original = sequential_module._retention_factor
    seen: dict[tuple[str, int], np.ndarray] = {}

    def recording(data: Any, plans: Any, time: int, fit_at: Any, **kwargs: Any) -> Any:
        seen[(kwargs["key"], time)] = np.asarray(fit_at).copy()
        return original(data, plans, time, fit_at, **kwargs)

    monkeypatch.setattr(sequential_module, "_retention_factor", recording)
    result = _fit("ltmle", _law(False))
    masks = data.regimen_masks(data.treatment)
    for time in range(1, data.n_times + 1):
        censoring = seen[("observed_censoring", time)]
        competing = seen[("observed_competing", time)]
        at_risk = masks.uncensored[:, time - 1] & masks.event_free[:, time - 1]
        np.testing.assert_array_equal(censoring, at_risk)
        np.testing.assert_array_equal(competing, at_risk & data.retained[:, time - 1])
        # Witness: some unit at risk was censored, so the two row sets differ.
        assert competing.sum() < censoring.sum()
    assert len(result.mechanism.competing) == data.n_times
    assert result.mechanism.no_competing_nodes == ()


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


def test_ratio_trim_reads_the_two_factor_product() -> None:
    """With ratio_trim= binding, the eliminated fit equals the hand-built composite fit.

    Saturated cells fit the product of the two factors exactly as one composite model, so the
    trimmed ratios agree only if the trim reads the product.
    """
    law = _law(False)
    frame = law.frame().copy()
    for node in (1, 2, 3):
        died = frame[f"D{node}"] == 1
        frame.loc[died, f"C{node}"] = 0.0
        frame.loc[died, [f"R{node}", f"D{node}"]] = np.nan
    columns = law.fit_columns()
    single = {**columns, "outcome": columns["outcome"]["relapse"]}

    def engine(trim: float | None) -> Any:
        return LTMLE(REGIMENS, n_folds=1, g_bounds=NO_TRUNCATION, ratio_trim=trim, **_cells())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_hand = engine(0.95).fit(frame, **single)
        declared = engine(0.95).fit(law.frame(), eliminate=["death"], **columns)
        untrimmed = engine(None).fit(law.frame(), eliminate=["death"], **columns)
    # The saturated outcome fit is the oracle, so the trim shows in the influence curve.
    moved = 0.0
    for label in REGIMENS:
        for horizon in HORIZONS:
            name = _name(label, horizon)
            assert declared[name].psi == pytest.approx(by_hand[name].psi, abs=1e-10)
            np.testing.assert_allclose(
                declared[name].influence_curve, by_hand[name].influence_curve, atol=1e-10
            )
            gap = declared[name].influence_curve - untrimmed[name].influence_curve
            moved = max(moved, float(np.abs(gap).max()))
    # The witness: the trim binds, so the curve depends on the trimmed product.
    assert moved > 1e-3


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


def test_the_study_path_passes_the_competing_learner() -> None:
    from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
    from cleverly._summary import vocabulary

    law = _law(False)
    design = LongitudinalTreatment(eliminate=["death"], **law.fit_columns())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CausalStudy(law.frame(), design=design).estimate(
            RegimeContrast(regimens=REGIMENS, reference="a0", horizons=[3]),
            competing_learner=_MarkedCells(),
            cross_fit=False,
            **_cells(),
        )
    assert dict(result.summary().header)[vocabulary.PREVENTED_MODEL] == "_MarkedCells"


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


# ------------------------------------------------------------------ competing_learner


class _MarkedCells(CellMeans):
    """The saturated learner under another class, to tell the competing factor apart."""


class _Marginal(CellMeans):
    """A learner that reads no covariate: the training mean in every row."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Marginal:
        return super().fit(np.zeros((len(y), 1)), y, sample_weight)

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(np.zeros((len(X), 1)))

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(np.zeros((len(X), 1)))


def _learners_by_factor(monkeypatch: pytest.MonkeyPatch) -> dict[str, set[type]]:
    seen: dict[str, set[type]] = {"observed_censoring": set(), "observed_competing": set()}
    original = sequential_module._retention_factor

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen.setdefault(kwargs["key"], set()).add(type(kwargs["learner"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(sequential_module, "_retention_factor", recording)
    return seen


@pytest.mark.parametrize(
    "build",
    [
        lambda: LTMLE(REGIMENS, n_folds=1, competing_learner=_MarkedCells(), **_cells()),
        # Two folds: each training fold fits its own mechanism, the competing factor too.
        lambda: SDR(REGIMENS, n_folds=2, competing_learner=_MarkedCells(), **_cells()),
        lambda: SequentialIPW(
            REGIMENS, censoring_learner=CellMeans(), competing_learner=_MarkedCells()
        ),
    ],
    ids=["ltmle", "sdr_cross_fitted", "ipw"],
)
def test_competing_learner_fits_the_competing_factor_only(
    monkeypatch: pytest.MonkeyPatch, build: Any
) -> None:
    seen = _learners_by_factor(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build().fit(_data(_law(False)))
    assert seen["observed_competing"] == {_MarkedCells}
    assert _MarkedCells not in seen["observed_censoring"]


def test_without_competing_learner_the_censoring_learner_fits_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _learners_by_factor(monkeypatch)
    _fit("ltmle", _law(False))
    assert seen["observed_competing"] == seen["observed_censoring"] == {CellMeans}


def test_a_competing_learner_that_reads_no_covariate_breaks_the_curve() -> None:
    """The witness that the option reaches the fit: death depends on the history here."""
    law = _law(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            REGIMENS, n_folds=1, g_bounds=NO_TRUNCATION, competing_learner=_Marginal(), **_cells()
        ).fit(_data(law))
    assert _worst_curve_gap(law, result) > MUTATION_FLOOR


# ------------------------------------------------------------------ compositions


def _truth(arm: int, horizon: int) -> float:
    law = _law(False)
    return law.functional(law.probs, static(arm), horizon, RELAPSE, eliminate=DEATH)


def test_rmst_and_ratio_compose_from_the_eliminated_risks(
    fits: dict[tuple[str, bool], Any],
) -> None:
    result = fits[("ltmle", False)]
    # RMST(3) = 3 - F(1) - F(2) on the node scale.
    assert result.rmst("a1", 3).psi == pytest.approx(3 - _truth(1, 1) - _truth(1, 2), abs=1e-10)
    ratio = result.ratio(_name("a1", 3), _name("a0", 3))
    assert ratio.psi == pytest.approx(_truth(1, 3) / _truth(0, 3), abs=1e-10)
    curve = result.curve()
    row = curve[curve["estimand"] == _name("a2", 2)]
    assert float(row["psi"].iloc[0]) == pytest.approx(_truth(2, 2), abs=1e-10)


def _truncated_cells(result: Any) -> list[int]:
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve

    bounds = [NO_TRUNCATION, *((b, 1.0) for b in (0.1, 0.2, 0.3, 0.4, 0.5))]
    table = longitudinal_truncation_curve(result, bounds, estimands=[_name("a1", 3)])
    return [int(count) for count in table["truncated_score_cells"]]


def test_the_truncation_curve_replays_both_factors(
    monkeypatch: pytest.MonkeyPatch, fits: dict[tuple[str, bool], Any]
) -> None:
    result = fits[("ltmle", False)]
    name = _name("a1", 3)
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve

    table = longitudinal_truncation_curve(result, [NO_TRUNCATION], estimands=[name])
    assert float(table["psi"].iloc[0]) == pytest.approx(result[name].psi, abs=1e-12)
    # The point estimate cannot see the mechanism on this law, so the witness is which
    # cells a bound clips: a replay without the competing factor clips other cells.
    full = _truncated_cells(result)
    original = sequential_module._retention_factor

    def without_competing(*args: Any, **kwargs: Any) -> Any:
        factor = original(*args, **kwargs)
        if kwargs.get("key") != "observed_competing":
            return factor
        ones = np.ones_like(factor.observed)
        return replace(factor, by_plan=dict.fromkeys(factor.by_plan, ones), observed=ones)

    monkeypatch.setattr(sequential_module, "_retention_factor", without_competing)
    assert _truncated_cells(_fit("ltmle", _law(False))) != full


def test_a_saturated_working_model_returns_the_eliminated_risks() -> None:
    from cleverly.msm import MSM

    cells = [(label, horizon) for label in REGIMENS for horizon in HORIZONS]

    def design(label: str, horizon: int, frame: Any) -> np.ndarray:
        return np.column_stack(
            [np.full(len(frame), float((label, horizon) == cell)) for cell in cells]
        )

    msm = MSM(design=design, terms=tuple(f"{l}@{h}" for l, h in cells), design_kind="known")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            REGIMENS, n_folds=1, reference=None, msm=msm, g_bounds=NO_TRUNCATION, **_cells()
        ).fit(_data(_law(False)))
    for label, arm in REGIMENS.items():
        for horizon in HORIZONS:
            estimate = result[f"msm_regimen[{label}@{horizon}]"].psi
            assert estimate == pytest.approx(_truth(arm, horizon), abs=1e-10)


def test_the_bootstrap_refits_the_competing_factor_in_every_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {"calls": 0}
    original = sequential_module._retention_factor

    def counting(*args: Any, **kwargs: Any) -> Any:
        if kwargs["key"] == "observed_competing":
            seen["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sequential_module, "_retention_factor", counting)
    replicates = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(REGIMENS, n_folds=1, n_bootstrap=replicates, random_state=1, **_cells()).fit(
            _data(_law(False))
        )
    # One fit per node for the original and for each replicate.
    assert seen["calls"] == (replicates + 1) * len(HORIZONS)
    assert result.bootstrap is not None


def test_a_learned_regimen_refuses_an_eliminated_fit() -> None:
    from cleverly.longitudinal import LearnedRegimen

    with pytest.raises(CapabilityError, match="survival"):
        LTMLE({"learned": LearnedRegimen()}, n_folds=2, **_cells()).fit(_data(_law(False)))


def test_the_result_summary_names_the_prevented_cause(
    fits: dict[tuple[str, bool], Any],
) -> None:
    from cleverly._summary import vocabulary

    summary = fits[("ltmle", False)].summary()
    header = dict(summary.header)
    assert header[vocabulary.PREVENTED].startswith("death;")
    assert header[vocabulary.OUTCOME].startswith("relapse at")
    assert header[vocabulary.PREVENTED_MODEL] == "CellMeans"
    text = str(summary)
    assert "probability of relapse by time 1 if every unit followed plan a0, had death" in (
        " ".join(text.split())
    )


# ------------------------------------------------------------------ parametric g-formula


def _gformula() -> Any:
    from cleverly.longitudinal import ParametricGFormula

    return ParametricGFormula(
        REGIMENS,
        hazard_learner=SATURATED,
        treatment_learner=SATURATED,
        censoring_learner=SATURATED,
        simultaneous=False,
    )


@pytest.mark.parametrize("eliminate", [True, False], ids=["eliminated", "total_effect"])
def test_the_parametric_g_formula_returns_the_g_formula(eliminate: bool) -> None:
    """The hazard rows read the composite risk set, which is the eliminated risk set.

    The total-effect case pins the held baseline treatment, which the design check once
    read past the plan's one node.
    """
    law = _law(False)
    data = (
        _data(law) if eliminate else LongitudinalData.from_frame(law.frame(), **law.fit_columns())
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _gformula().fit(data)
    for label, arm in REGIMENS.items():
        for horizon in HORIZONS:
            if eliminate:
                truth = law.functional(law.probs, static(arm), horizon, RELAPSE, eliminate=DEATH)
                name = _name(label, horizon)
            else:
                truth = law.functional(law.probs, static(arm), horizon, RELAPSE)
                name = f"cif_regimen[{label}, relapse @ t={horizon}]"
            assert result[name].psi == pytest.approx(truth, abs=1e-9)


# ------------------------------------------------------------------ review follow-ups


def test_the_diagnostics_score_each_retention_factor_on_its_own_rows() -> None:
    from cleverly.validation.longitudinal import _longitudinal_nuisances

    law = _law(False)
    data = _data(law)
    result = _fit("ltmle", law)
    frame = _longitudinal_nuisances(result).to_frame()
    masks = data.regimen_masks(data.treatment)
    assert data.retained is not None
    for time in range(1, data.n_times + 1):
        at_risk = masks.uncensored[:, time - 1] & masks.event_free[:, time - 1]
        rows = frame.loc[frame["time"] == time].set_index("role")
        assert int(rows.loc["censoring", "n"]) == int(at_risk.sum())
        assert int(rows.loc["competing", "n"]) == int((at_risk & data.retained[:, time - 1]).sum())


def test_without_censoring_only_the_prevented_factor_is_reported() -> None:
    from cleverly._summary import vocabulary
    from cleverly.validation.longitudinal import (
        LONGITUDINAL_NO_CENSORING,
        _longitudinal_nuisances,
    )

    law = _law(False)
    frame = law.frame()
    frame = frame[(frame[[f"C{k}" for k in (1, 2, 3)]].fillna(1) == 1).all(axis=1)]
    columns = {key: value for key, value in law.fit_columns().items() if key != "censoring"}
    data = LongitudinalData.from_frame(frame, eliminate=["death"], **columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _engines()["ltmle"]().fit(data)
    diagnostics = _longitudinal_nuisances(result)
    roles = set(diagnostics.to_frame()["role"])
    assert "competing" in roles and "censoring" not in roles
    reasons = {(o.role, o.time, o.reason) for o in diagnostics.omissions}
    assert all(("censoring", t, LONGITUDINAL_NO_CENSORING) in reasons for t in (1, 2, 3))
    header = dict(result.summary().header)
    assert vocabulary.CENSORING not in header and vocabulary.CENSORING_MODEL not in header
    assert header[vocabulary.PREVENTED].startswith("death;")
    # No competing learner was passed, so the censoring learner fitted the prevented cause.
    assert header[vocabulary.PREVENTED_MODEL] == "CellMeans"


def test_a_declared_zero_retention_is_checked_on_the_censoring_factor() -> None:
    """A unit that passed censoring and then died still contradicts P(C = 1) = 0."""
    law = _law(False)
    frame = law.frame()
    died = (frame["C1"] == 1) & (frame["D1"] == 1)
    retention = np.full((len(frame), 3), 0.75)
    retention[died.to_numpy(), 0] = 0.0
    with pytest.raises(DataError, match="retention probability 0"):
        LTMLE(REGIMENS, n_folds=1, g_bounds=NO_TRUNCATION, **_cells()).fit(
            frame, eliminate=["death"], censoring_probabilities=retention, **law.fit_columns()
        )


@pytest.mark.parametrize("engine", ["ltmle", "ltmle_hazard"])
def test_the_pooled_competing_factor_gives_the_eliminated_curve(engine: str) -> None:
    law = _law(False)
    survival_model = "hazard" if engine == "ltmle_hazard" else "iterated_mean"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            REGIMENS,
            n_folds=1,
            survival_model=survival_model,
            hazard_pooling="pooled",
            g_bounds=NO_TRUNCATION,
            **_cells(),
        ).fit(_data(law))
    assert _worst_curve_gap(law, result) < 1e-9


def test_a_decision_at_each_node_reads_the_eliminated_curve() -> None:
    """The two-cause law with a treatment at each node: estimate and curve are exact."""
    from tests import discrete_law_competing as competing
    from tests.studies import eliminated_competing_law as tv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            tv.TV_REGIMENS, reference=tv.TV_REFERENCE, n_folds=1, g_bounds=NO_TRUNCATION, **_cells()
        ).fit(tv.tv_exact_frame(), eliminate=["death"], **tv.TV_FIT_COLUMNS)
    rows = competing.first_row_of()
    for name in tv.TV_MEANS:
        assert result[name].psi == pytest.approx(tv.TV_TRUTH[name], abs=1e-10)
        curve = np.asarray(result[name].influence_curve)[rows]
        assert np.max(np.abs(curve - tv.tv_eif(name))) < 1e-9


def test_ipsi_on_an_eliminated_fit_is_the_hand_built_composite_fit() -> None:
    """IPSI composes: the same estimate and curve as death folded into censoring by hand."""
    from tests import discrete_law_competing as competing
    from tests import discrete_law_longitudinal_ipsi as ipsi_law
    from tests.unit.test_ipsi_exact_laws import EVENT_COLUMNS, fit_ipsi

    frame = competing.frame()
    columns = {"outcome": competing.outcome_columns(), **EVENT_COLUMNS}
    declared = fit_ipsi(
        frame,
        ipsi_law.EVENT_PLANS,
        columns,
        horizons=list(competing.HORIZONS),
        declared={"eliminate": ["death"]},
    )
    by_hand = frame.copy()
    for node in (1, 2):
        dead = by_hand[f"D{node}"] == 1
        by_hand.loc[dead, f"C{node}"] = 0.0
        later = [f"R{node}", f"D{node}"] + (["L2", "A2", "C2", "R2", "D2"] if node == 1 else [])
        by_hand.loc[dead, later] = np.nan
    single = {**columns, "outcome": competing.outcome_columns()["relapse"]}
    composite = fit_ipsi(by_hand, ipsi_law.EVENT_PLANS, single, horizons=list(competing.HORIZONS))
    names = [name for name in declared.estimates if name in composite.estimates]
    assert names, "the two fits must report the same parameters"
    for name in names:
        assert declared[name].psi == pytest.approx(composite[name].psi, abs=1e-12)
        np.testing.assert_allclose(
            declared[name].influence_curve, composite[name].influence_curve, atol=1e-12
        )


# ------------------------------------------------------------------ refusals


@pytest.fixture(scope="module")
def eliminated_fit() -> Any:
    law = _law(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _engines()["ltmle"]().fit(_data(law))


def test_with_events_refuses_eliminated_data(eliminated_fit: Any) -> None:
    data = eliminated_fit.data
    with pytest.raises(DataError, match="with_events draws observed events"):
        data.with_events(np.zeros((data.n, data.n_times)), np.ones((data.n, data.n_times)))


def test_to_long_panel_refuses_eliminated_data() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for unit in range(200):
        for t in (1, 2):
            relapse, death = (float(v) for v in rng.multinomial(1, (0.1, 0.1, 0.8))[:2])
            rows.append(
                {"id": unit, "t": t, "w": unit % 3, "a": t % 2, "relapse": relapse, "death": death}
            )
            if relapse or death:
                break
    data = LongitudinalData.from_long_panel(
        pd.DataFrame(rows),
        unit="id",
        time="t",
        treatment="a",
        baseline=["w"],
        event={"relapse": "relapse", "death": "death"},
        eliminate=["death"],
    )
    with pytest.raises(DataError, match="to_long_panel exports observed rows"):
        data.to_long_panel()


def test_the_evalue_refuses_an_eliminated_fit(eliminated_fit: Any) -> None:
    from cleverly.sensitivity import evalue

    name = next(iter(eliminated_fit.estimates))
    with pytest.raises(CapabilityError, match=r"this fit eliminates \['death'\]"):
        evalue(eliminated_fit, name)


def test_simulated_confounding_refuses_an_eliminated_fit(eliminated_fit: Any) -> None:
    from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding

    name = next(iter(eliminated_fit.estimates))
    treatments = eliminated_fit.data.treatment_names
    grid = ConfounderStrengthGrid(treatment=dict.fromkeys(treatments, (0.0, 0.1)), outcome=(0.0,))
    with pytest.raises(CapabilityError, match="eliminates competing causes"):
        simulated_confounding(eliminated_fit, name, grid=grid, random_state=0)


def test_structural_nested_means_refuse_eliminate() -> None:
    from cleverly.longitudinal import StructuralNestedMean

    law = _law(False)
    with pytest.raises(CapabilityError):
        StructuralNestedMean().fit(law.frame(), eliminate=["death"], **law.fit_columns())
