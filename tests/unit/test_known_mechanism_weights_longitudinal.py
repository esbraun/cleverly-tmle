"""Declared longitudinal factors beside observation weights: the two readings.

A weighted longitudinal fit estimates the regimen mean in the tilted law ``dP_w = w dP / E w``
and divides by that law's node factors.  At a treatment node Bayes' rule gives
``P_w(A_t = a | H_t) = g0_t m / sum_a' g0_t m`` with ``m(a, H_t) = E[w | A_t = a, H_t]``, and a
retention node is the same with ``C_t`` given ``(H_t, A_t)``.  ``mechanism_law="target"`` divides
by the declaration as the tilted law's factor; ``mechanism_law="sample"`` estimates the factor
as ``g0_t r_t / sum g0_t r_t`` with ``r_t`` the node learner's classification on the rows at risk
with weights ``w / g0_t(A_t | H_t)``.

The exact-law tests run on the two-node SMART law of
:mod:`tests.unit._known_mechanism_support` with censoring after node one, whose empirical law
is the law, and a weight on observed data only: ``L0``, ``A1``, ``C1`` and ``L1`` where the unit
stayed.  It reads the first treatment, the first retention and a covariate measured after
the first treatment, so the tilted factors differ from the declared ones at the first
treatment and retention nodes.  It reads nothing at or after ``A2``, so the tilted factor
there is the declaration, while the classifier trained with ``w / g0`` is one half: a fit
that dropped ``g0`` from the composition misses there.  The second retention is one for every unit still observed, a
0/1 declaration that both readings divide by.  The node classifier is the trainable
:class:`~tests.unit._known_mechanism_support.WeightedCellClassifier`, so a ``"sample"`` fit
divides by the tilted factors themselves, and with a wrong outcome regression its estimate
equals the longhand g-formula on ``P_w``.  The mutation controls change what the classifier
is trained on or read where.

The test names follow ``rows/Fix-16/plan.md`` section 5 (W1, W10, W12) of the plans
repository; :mod:`tests.unit.test_known_mechanism_weights` holds the point-treatment tests.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from cleverly import (
    CausalStudy,
    DataError,
    LongitudinalTreatment,
    LongPanel,
    RegimeMean,
    TimeToEvent,
)
from cleverly._summary import vocabulary
from cleverly.data.known_mechanism import (
    KNOWN_NODE_WEIGHTED_LAW_REFUSAL,
    tilts_node_declarations,
    unweighted_tilt_refusal,
)
from cleverly.exceptions import CapabilityError, MethodConfigurationError
from cleverly.inference.bootstrap import run_bootstrap
from cleverly.interventions import Incremental
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    DynamicRegimen,
    LongitudinalData,
    ParametricGFormula,
    sequential,
)
from cleverly.longitudinal.regimen import resolve_plans
from cleverly.longitudinal.sequential import TiltedNodeModel, fit_mechanism
from cleverly.validation._interpretation import fit_configuration
from cleverly.validation.longitudinal import _longitudinal_nuisances
from tests.unit import _known_mechanism_support as law
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit

#: The module, which the package's ``gformula`` function shadows as an attribute.
gf = importlib.import_module("cleverly.longitudinal.gformula")

#: The exact-law tolerance: each node's fluctuation stops at ``tol = 1e-10``.
EXACT = 1e-9
#: How far a mutated fit must miss.
MISS = 1e-3
COLUMNS: dict[str, Any] = {**law.SMART_COLUMNS, "censoring": ["C1", "C2"]}
DECLARED: dict[str, Any] = {
    "treatment_probabilities": law.SMART_TREATMENT,
    "censoring_probabilities": law.SMART_CENSORING,
}
#: The static regimens and the arms they set at the two nodes.
ARMS: dict[str, tuple[int, int]] = {"always": (1, 1), "never": (0, 0), "early": (1, 0)}
REGIMENS: dict[str, Any] = {"always": 1, "never": 0, "early": (1, 0)}

#: The exact laws carry whole-number weights, which the weight check reads as possible counts.
pytestmark = pytest.mark.filterwarnings("ignore::cleverly.exceptions.WeightingWarning")


def _frame() -> pd.DataFrame:
    """The SMART law with censoring and the weight on observed data only."""
    frame = law.smart_frame(censoring=True)
    stayed = frame["C1"].to_numpy()
    l1 = np.nan_to_num(frame["L1"].to_numpy()) * stayed
    a1 = frame["A1"].to_numpy()
    frame["w"] = (1.0 + frame["L0"]) * (1.0 + 2.0 * a1) * (1.0 + stayed) * (1.0 + l1 * (1.0 + a1))
    return frame


def _estimator(cls: Any, regimens: Any = None, **settings: Any) -> Any:
    base: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": law.WeightedCellClassifier(),
        "censoring_learner": law.WeightedCellClassifier(),
        "simultaneous": False,
    }
    base.update(settings)
    return cls(REGIMENS if regimens is None else regimens, **base)


def _fit(model: Any, mechanism_law: str, frame: pd.DataFrame | None = None) -> Any:
    return model.fit(
        _frame() if frame is None else frame,
        **COLUMNS,
        weights="w",
        **DECLARED,
        mechanism_law=mechanism_law,
    )


# --------------------------------------------------------------------------- the tilted law


def _share(rows: pd.DataFrame, column: str) -> float:
    """The weighted share of ``column == 1`` among ``rows``."""
    return float((rows["w"] * rows[column]).sum() / rows["w"].sum())


def _tilted_truth(
    frame: pd.DataFrame,
    first: Callable[[float], float],
    second: Callable[[float], float],
) -> float:
    """The longhand g-formula on ``P_w``.

    ``first(g)`` and ``second(g)`` give the regimen's probability of arm one at each node from
    the tilted law's own probability ``g`` there, so a static arm ignores ``g`` and an odds
    tilt reads it.  Every conditional is a weighted share of the frame, whose empirical law is
    the law.
    """
    total = 0.0
    weight = float(frame["w"].sum())
    for l0 in (0.0, 1.0):
        cell = frame[frame["L0"] == l0]
        p_l0 = float(cell["w"].sum()) / weight
        q1 = first(_share(cell, "A1"))
        for a1, p_a1 in ((0.0, 1.0 - q1), (1.0, q1)):
            if p_a1 == 0.0:
                continue
            kept = cell[(cell["A1"] == a1) & (cell["C1"] == 1.0)]
            for l1 in (0.0, 1.0):
                history = kept[kept["L1"] == l1]
                p_l1 = float(history["w"].sum() / kept["w"].sum())
                q2 = second(_share(history, "A2"))
                for a2, p_a2 in ((0.0, 1.0 - q2), (1.0, q2)):
                    if p_a2 == 0.0:
                        continue
                    arm = history[history["A2"] == a2]
                    total += p_l0 * p_a1 * p_l1 * p_a2 * _share(arm, "Y")
    return total


def _static_truth(frame: pd.DataFrame) -> dict[str, float]:
    return {
        f"ey_regimen[{name}]": _tilted_truth(
            frame, lambda g, a=a1: float(a), lambda g, a=a2: float(a)
        )
        for name, (a1, a2) in ARMS.items()
    }


def _odds(delta: float) -> Callable[[float], float]:
    return lambda g: delta * g / (delta * g + 1.0 - g)


def _errors(result: Any, truth: dict[str, float]) -> dict[str, float]:
    return {name: abs(float(result[name].psi) - value) for name, value in truth.items()}


def _assert_exact(result: Any, truth: dict[str, float]) -> None:
    errors = _errors(result, truth)
    assert max(errors.values()) <= EXACT, errors


def _assert_misses(result: Any, truth: dict[str, float]) -> None:
    errors = _errors(result, truth)
    assert max(errors.values()) > MISS, errors


# ------------------------------------------------------------------------------------ W1


def _long_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """The SMART frame as a long panel: a row per unit and node while the unit is observed."""
    rows: list[dict[str, float]] = []
    for unit, row in frame.reset_index(drop=True).iterrows():
        nodes = (1, 2) if row["C1"] == 1.0 else (1,)
        for t in nodes:
            rows.append(
                {
                    "id": float(unit),
                    "t": float(t),
                    "L0": row["L0"],
                    "L": row["L1"] if t == 2 else np.nan,
                    "A": row[f"A{t}"],
                    "C": row[f"C{t}"],
                    "Y": row["Y"],
                    "w": row["w"],
                    "g0": row[f"g{t}_0"],
                    "g1": row[f"g{t}_1"],
                    "r": row[f"r{t}"],
                }
            )
    return pd.DataFrame(rows)


_PANEL_ROLES: dict[str, Any] = {
    "unit": "id",
    "time": "t",
    "treatment": "A",
    "baseline": ["L0"],
    "time_varying": ["L"],
    "censoring": "C",
    "outcome": "Y",
    "weights": "w",
}


def _time_to_event() -> tuple[pd.DataFrame, dict[str, Any]]:
    """A one-row-per-unit survival frame with a declared baseline mechanism and a weight."""
    from tests.discrete_law_point_survival import G, survival_law

    frame = survival_law(censor_first=False).long_frame((1.0, 2.0, 3.0))
    matrix = G[frame["W"].to_numpy().astype(int)]
    frame = frame.assign(
        p0=matrix[:, 0], p1=matrix[:, 1], p2=matrix[:, 2], w=1.0 + frame["A"].to_numpy()
    )
    roles = {
        "time": "time",
        "event": "event",
        "treatment": "A",
        "baseline": ["W"],
        "grid": (1.0, 2.0, 3.0),
        "weights": "w",
        "treatment_probabilities": {0.0: "p0", 1.0: "p1", 2.0: "p2"},
    }
    return frame, roles


def _never(cls: Any = LTMLE, regimens: Any = None, **settings: Any) -> Any:
    return _estimator(
        cls,
        regimens,
        outcome_learner=NeverFit(),
        pseudo_learner=NeverFit(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        **settings,
    )


def _divided(cls: Any = LTMLE, regimens: Any = None) -> Any:
    """A fit that divides by every declared factor, so no node classifier may be fitted."""
    return _estimator(cls, regimens, treatment_learner=NeverFit(), censoring_learner=NeverFit())


def _static_regimens() -> tuple[Any, ...]:
    return tuple(DynamicRegimen(name, arms) for name, arms in ARMS.items())


def _refused(build: Callable[[], Any]) -> None:
    NeverFit.calls = 0
    assert_refused_before_any_call(
        build, None, "learner", KNOWN_NODE_WEIGHTED_LAW_REFUSAL, error=DataError
    )


def test_w1_every_longitudinal_entry_refuses_a_declaration_beside_weights() -> None:
    """W1: each entry refuses, before any learner, with the pinned text."""
    frame = _frame()
    panel = _long_panel(frame)
    survival, survival_roles = _time_to_event()
    arrays = {
        "treatment_probabilities": frame[["g1_1", "g2_1"]].to_numpy(dtype=float),
        "censoring_probabilities": frame[["r1", "r2"]].to_numpy(dtype=float),
    }
    weighted = LongitudinalData.from_frame(frame, **COLUMNS, weights="w")
    unweighted = LongitudinalData.from_frame(frame, **COLUMNS)
    wide_design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["L0"],
        time_varying=[[], ["L1"]],
        censoring=["C1", "C2"],
        weights="w",
        **DECLARED,
    )
    panel_design = LongPanel(
        **_PANEL_ROLES,
        treatment_probabilities={0.0: "g0", 1.0: "g1"},
        censoring_probabilities="r",
    )
    survival_design = TimeToEvent(**survival_roles)
    tilt = {"x2": DynamicRegimen("x2", (1, Incremental(2.0)))}
    entries: dict[str, Callable[[], Any]] = {
        "wide": lambda: LongitudinalData.from_frame(frame, **COLUMNS, weights="w", **DECLARED),
        "long": lambda: LongitudinalData.from_long_panel(
            panel, **_PANEL_ROLES, treatment_probabilities={0.0: "g0", 1.0: "g1"}
        ),
        "time_to_event": lambda: LongitudinalData.from_time_to_event(survival, **survival_roles),
        "with_known_mechanisms": lambda: weighted.with_known_mechanisms(**arrays),
        "ltmle_frame": lambda: _never().fit(frame, **COLUMNS, weights="w", **DECLARED),
        "ltmle_container": lambda: _never().fit(weighted, **arrays),
        "sdr": lambda: _never(SDR).fit(frame, **COLUMNS, weights="w", **DECLARED),
        "ipsi": lambda: _never(IPSI, tilt).fit(frame, **COLUMNS, weights="w", **DECLARED),
        "longitudinal_design": lambda: CausalStudy(frame, design=wide_design),
        "panel_design": lambda: CausalStudy(panel, design=panel_design),
        "time_to_event_design": lambda: CausalStudy(survival, design=survival_design),
        # A container built by hand that carries both reaches the fit's own check.
        "hand_built": lambda: _never().fit(
            dataclasses.replace(
                unweighted.with_known_mechanisms(**arrays), weights=weighted.weights
            )
        ),
    }
    for name, build in entries.items():
        try:
            _refused(build)
        except AssertionError as error:  # pragma: no cover - names the entry that failed
            raise AssertionError(f"{name}: {error}") from error


def test_w1_the_keyword_is_validated_and_needs_a_declaration() -> None:
    frame = _frame()
    with pytest.raises(DataError, match="mechanism_law must be"):
        LongitudinalData.from_frame(frame, **COLUMNS, weights="w", **DECLARED, mechanism_law="P_w")
    with pytest.raises(DataError, match="declares neither"):
        LongitudinalData.from_frame(frame, **COLUMNS, weights="w", mechanism_law="target")
    with pytest.raises(DataError, match="declares neither"):
        LongitudinalTreatment(
            outcome="Y", treatment=["A1", "A2"], baseline=["L0"], mechanism_law="sample"
        )
    data = LongitudinalData.from_frame(frame, **COLUMNS, weights="w")
    with pytest.raises(ValueError, match="passes neither"):
        _never().fit(data, mechanism_law="sample")


def test_w1b_a_zero_one_declaration_and_constant_weights_need_no_reading() -> None:
    """W1b: a declaration whose every value is 0 or 1 is one law under both readings.

    The second retention is one for every unit still observed and missing otherwise.  Its
    ``law`` stays ``None``, it reads as ``"target"``, and a ``"sample"`` fit divides by it
    too.  Constant weights tilt nothing, so the full declaration needs no reading either.
    """
    frame = _frame()
    degenerate = LongitudinalData.from_frame(
        frame, **COLUMNS, weights="w", censoring_probabilities={"C2": "r2"}
    )
    assert degenerate.known_mechanisms is not None
    assert degenerate.known_mechanisms.law is None
    assert not tilts_node_declarations(degenerate)
    constant = frame.assign(w=2.0)
    data = LongitudinalData.from_frame(constant, **COLUMNS, weights="w", **DECLARED)
    assert not tilts_node_declarations(data)
    sample = _fit(_divided(), "sample", constant)
    assert sample.config.known_factors == ("A1", "C1", "A2", "C2")


# ------------------------------------------------------------------------------------ W10


@pytest.mark.parametrize("cls", [LTMLE, SDR], ids=["ltmle", "sdr"])
def test_w10_sample_recovers_the_tilted_truth(cls: Any) -> None:
    """W10: ``"sample"`` with the trainable classifier is exact; ``"target"`` misses.

    The outcome regressions are constants, so the estimate is exact only through the
    factors: a correct sequential regression would make the target reading exact too.
    """
    frame = _frame()
    truth = _static_truth(frame)
    wrong = {"outcome_learner": DummyClassifier(), "pseudo_learner": DummyRegressor()}
    sample = _fit(_estimator(cls, **wrong), "sample", frame)
    _assert_exact(sample, truth)
    assert sample.mechanism.known_treatment_nodes == ()
    assert sample.mechanism.known_censoring_nodes == (2,)
    target = _estimator(cls, treatment_learner=NeverFit(), censoring_learner=NeverFit(), **wrong)
    _assert_misses(_fit(target, "target", frame), truth)


def test_w10_ipsi_tilts_the_estimated_mechanism_with_its_term() -> None:
    """W10: an odds tilt at each node of a ``"sample"`` fit is exact for IPSI.

    Each tilt reads ``g_w``, which the fit estimates from the declaration, so the node keeps
    its mechanism term.
    """
    frame = _frame()
    plans = {
        "x1": ((Incremental(2.0), 1), (_odds(2.0), lambda g: 1.0)),
        "x2": ((1, Incremental(2.0)), (lambda g: 1.0, _odds(2.0))),
        "x12": ((Incremental(2.0), Incremental(0.5)), (_odds(2.0), _odds(0.5))),
    }
    for name, (spec, (first, second)) in plans.items():
        truth = {f"ey_regimen[{name}]": _tilted_truth(frame, first, second)}
        regimens = {name: DynamicRegimen(name, spec)}
        result = _fit(_estimator(IPSI, regimens), "sample", frame)
        _assert_exact(result, truth)
        assert result.mechanism.known_treatment_nodes == ()
        _assert_misses(
            _fit(_estimator(IPSI, regimens, treatment_learner=NeverFit()), "target", frame),
            truth,
        )


def test_w10_the_classifier_trains_on_the_rows_at_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: the tilt classifier fitted on every row, censored ones included, misses.

    A censored unit has no second treatment or ``L1``; the design fills both with zero, so a
    classifier that trains on it learns a different law at node two.
    """
    frame = _frame()
    truth = _static_truth(frame)
    original = sequential.cross_fit_predictions
    weights = frame["w"].to_numpy(dtype=float)

    def every_row(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("mean_one_weights"):
            learner, design, target, tilt, folds = args
            tilt = np.where(tilt > 0.0, tilt, 2.0 * weights)
            kwargs["fit_mask"] = np.ones(len(target), dtype=bool)
            args = (learner, design, target, tilt, folds)
        return original(*args, **kwargs)

    monkeypatch.setattr(sequential, "cross_fit_predictions", every_row)
    _assert_misses(_fit(_estimator(LTMLE), "sample", frame), truth)


def test_w10_a_fold_restricted_fit_trains_the_tilt_on_its_rows_only() -> None:
    """W10: ``fit_rows`` restricts the tilt classifier, as it restricts an estimated factor.

    The training rows drop every second treated unit with ``L0 = 0``.  The node-one factor of
    the restricted call is the tilted law of those rows, and it differs from the tilted law of
    every row, which a classifier that ignored ``fit_rows`` would return.
    """
    frame = _frame()
    data = LongitudinalData.from_frame(
        frame, **COLUMNS, weights="w", **DECLARED, mechanism_law="sample"
    )
    rows = np.ones(data.n, dtype=bool)
    treated = np.flatnonzero((frame["A1"] == 1.0).to_numpy() & (frame["L0"] == 0.0).to_numpy())
    rows[treated[::2]] = False
    plans = resolve_plans(_static_regimens(), data)
    mechanism = fit_mechanism(
        data,
        plans,
        treatment_learner=law.WeightedCellClassifier(),
        censoring_learner=law.WeightedCellClassifier(),
        folds=Folds.single(data.n),
        fit_rows=rows,
    )
    observed = mechanism.treatment_observed[0][:, 1]
    zero = (frame["L0"] == 0.0).to_numpy()
    restricted = frame[rows]
    assert observed[zero][0] == pytest.approx(
        _share(restricted[restricted["L0"] == 0.0], "A1"), abs=1e-12
    )
    every = _share(frame[frame["L0"] == 0.0], "A1")
    assert abs(observed[zero][0] - every) > MISS


def _kish(values: Any) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.sum() ** 2 / np.sum(values**2))


def test_w10_each_tilted_node_records_the_effective_sample_size_of_its_weights() -> None:
    """12.11: the Kish effective sample size of ``v = w / g0`` on each tilted node's rows.

    ``A1`` and ``C1`` train on every unit, ``A2`` on the units still observed.  ``C2`` is a
    0/1 declaration, which both readings divide by, so it records nothing, and a
    ``"target"`` fit records nothing at all.
    """
    frame = _frame()
    sample = _fit(_estimator(LTMLE), "sample", frame)
    recorded = sample.mechanism.tilt_effective_n
    assert set(recorded) == {"A1", "C1", "A2"}
    w = frame["w"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy() == 1.0
    stayed = frame["C1"].to_numpy() == 1.0
    g1 = np.where(a1, frame["g1_1"], frame["g1_0"])
    r1 = np.where(stayed, frame["r1"], 1.0 - frame["r1"])
    a2 = frame["A2"].to_numpy() == 1.0
    g2 = np.where(a2, frame["g2_1"], frame["g2_0"])
    assert recorded["A1"] == pytest.approx(_kish(w / g1), rel=1e-12)
    assert recorded["C1"] == pytest.approx(_kish(w / r1), rel=1e-12)
    assert recorded["A2"] == pytest.approx(_kish((w / g2)[stayed]), rel=1e-12)
    assert recorded["A2"] < stayed.sum()
    assert _fit(_divided(), "target", frame).mechanism.tilt_effective_n == {}


def test_w10_each_tilt_classifier_trains_on_weights_of_mean_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tilt weights are rescaled to mean one on each fit's training rows.

    A penalised learner's loss is not invariant to the weight scale, so a raw ``w / g0`` would
    change its regularisation; the trainable classifier is invariant, so the spy is the
    witness.  Two folds, three tilted nodes: six fits.
    """
    from cleverly.estimators import _nuisance

    seen: list[float] = []
    tilting = [False]
    original_fit = _nuisance.fit_on_rows
    original_cross = sequential.cross_fit_predictions

    def spy(learner: Any, design: Any, target: Any, weights: Any, rows: Any, *rest: Any) -> Any:
        if tilting[0]:
            seen.append(float(np.mean(np.asarray(weights, dtype=float)[rows])))
        return original_fit(learner, design, target, weights, rows, *rest)

    def cross(*args: Any, **kwargs: Any) -> Any:
        tilting[0] = bool(kwargs.get("mean_one_weights"))
        try:
            return original_cross(*args, **kwargs)
        finally:
            tilting[0] = False

    monkeypatch.setattr(_nuisance, "fit_on_rows", spy)
    monkeypatch.setattr(sequential, "cross_fit_predictions", cross)
    learner = LogisticRegression(C=0.1, max_iter=1000)
    _fit(
        _estimator(LTMLE, n_folds=2, treatment_learner=learner, censoring_learner=learner),
        "sample",
    )
    assert len(seen) == 6
    np.testing.assert_allclose(seen, 1.0, rtol=0, atol=1e-12)


def test_w10_the_tilted_retention_is_read_at_each_plan_s_history() -> None:
    """12.6: the classifier is predicted at the plan's history and the declaration at the row.

    On a unit with ``A1 = 0`` the ``"always"`` plan's retention factor is the tilted
    retention at ``A1 = 1``.  No shipped estimator reads that row's factor, since every
    clever covariate, ratio and IPSI weight is zero or evaluated at the observed history
    where the plan does not reach it; the mechanism itself is the witness.  The declared
    ``r1`` reads ``L0`` only, so composing it at the row is composing it at the plan's
    history here.
    """
    frame = _frame()
    data = LongitudinalData.from_frame(
        frame, **COLUMNS, weights="w", **DECLARED, mechanism_law="sample"
    )
    plans = resolve_plans(_static_regimens(), data)
    mechanism = fit_mechanism(
        data,
        plans,
        treatment_learner=law.WeightedCellClassifier(),
        censoring_learner=law.WeightedCellClassifier(),
        folds=Folds.single(data.n),
    )
    for l0 in (0.0, 1.0):
        untreated = ((frame["A1"] == 0.0) & (frame["L0"] == l0)).to_numpy()
        treated = frame[(frame["A1"] == 1.0) & (frame["L0"] == l0)]
        expected = _share(treated, "C1")
        always = mechanism.censoring[0]["always"][untreated]
        observed = mechanism.censoring_observed[0][untreated]
        np.testing.assert_allclose(always, expected, rtol=0, atol=1e-12)
        assert np.max(np.abs(observed - expected)) > MISS


def test_w10_an_ltmle_odds_tilt_on_a_sample_node_is_refused() -> None:
    """12.3: LTMLE reads a tilt as a known policy only from a declaration it divides by."""
    regimens = {"x2": DynamicRegimen("x2", (1, Incremental(2.0)))}
    NeverFit.calls = 0
    error = assert_refused_before_any_call(
        lambda: _fit(_never(LTMLE, regimens), "sample"), None, "learner", "IPSI"
    )
    assert isinstance(error, CapabilityError)
    # The target reading divides by the declaration and fits the tilt as a known policy.
    target = _fit(_estimator(LTMLE, regimens, treatment_learner=NeverFit()), "target")
    assert np.isfinite(target["ey_regimen[x2]"].psi)


def test_w10_the_key_and_the_summary_carry_no_known_tilted_factor() -> None:
    """12.4: a tilted node is no known factor in the record, the support key or the summary."""
    frame = _frame()
    sample = _fit(_estimator(LTMLE), "sample", frame)
    target = _fit(_divided(), "target", frame)
    assert sample.config.known_factors == ("C2",)
    assert target.config.known_factors == ("A1", "C1", "A2", "C2")
    assert sample.config.support_key is not None and target.config.support_key is not None
    assert "A1" not in sample.config.support_key.qualifiers
    assert {"A1", "C1", "A2", "C2"} <= set(target.config.support_key.qualifiers)
    sample_lines = " ".join(sample.summary(detail="full").section("Settings").lines)
    target_lines = " ".join(target.summary(detail="full").section("Settings").lines)
    assert "A1, C1, A2, C2" in target_lines
    assert "A1" not in sample_lines.split("Declared known")[-1]


def test_w10_the_g_formula_draws_the_natural_treatment_from_the_tilted_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """12.7: the natural law at a tilt node is ``g0 r / sum g0 r``, exact with the trainable
    classifier; under ``"target"`` it is the declaration.

    The tilt sits at node two, whose declaration reads ``L1``; at node one the declaration is
    one half for every row, so dropping ``g0`` there would change nothing.  A natural law read
    from the classifier alone misses.
    """
    frame = _frame()
    books: list[Any] = []
    original = gf.fit_models

    def capture(*args: Any, **kwargs: Any) -> Any:
        books.append(original(*args, **kwargs))
        return books[-1]

    monkeypatch.setattr(gf, "fit_models", capture)
    # The g-formula draws a censored unit's natural treatment too, so it needs a value there.
    left = frame["C1"] == 0.0
    frame.loc[left, ["g2_0", "g2_1"]] = 0.5
    regimens = {"x2": DynamicRegimen("x2", (1, Incremental(2.0)))}
    estimator = ParametricGFormula(
        regimens,
        random_state=0,
        simultaneous=False,
        treatment_learner=law.WeightedCellClassifier(),
        n_draws=2,
    )
    _fit(estimator, "sample", frame)
    book = books[-1]
    assert set(book.tilted) == {2} and book.known == {}
    units = np.arange(len(frame))
    natural = book.natural_law(2, units, book.design(book.treatment[2]))[:, 1]
    kept = (frame["C1"] == 1.0).to_numpy()
    for (l0, a1, l1), history in frame[kept].groupby(["L0", "A1", "L1"]):
        cell = kept & (frame[["L0", "A1", "L1"]] == (l0, a1, l1)).all(axis=1).to_numpy()
        expected = _share(history, "A2")
        np.testing.assert_allclose(natural[cell], expected, rtol=0, atol=1e-12)
    # The weight reads nothing at or after A2, so the tilted law there is the declaration,
    # and the classifier alone, trained with weights w / g0, is one half.
    classifier = book.treatment[2].class_probabilities(book.design(book.treatment[2]))[:, 1]
    assert np.max(np.abs(classifier[kept] - natural[kept])) > MISS
    _fit(estimator, "target", frame)
    assert set(books[-1].known) == {2} and books[-1].tilted == {}


def _composed_log_likelihood(
    model: Any, x: Any, drawn: Any, offset: Any, tilt: float | None = None
) -> Any:
    """Each path's log probability of ``drawn`` under ``exp(eta + offset)`` normalised."""
    logits = np.asarray(model.learner.linear_predictor(x), dtype=float)
    if logits.ndim == 1:
        logits = np.column_stack([np.zeros(logits.shape[0]), logits])
    logits = logits + np.asarray(offset, dtype=float)
    if tilt is not None:
        logits[:, -1] = logits[:, -1] + np.log(tilt)
    logits = logits - logits.max(axis=1, keepdims=True)
    log_law = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return log_law[np.arange(x.shape[0]), np.asarray(drawn, dtype=np.int64)]


@pytest.mark.parametrize(("levels", "tilt"), [(2, None), (2, 3.0), (3, None)])
def test_w10_the_g_formula_draw_score_with_a_declaration_is_the_composed_law_s_score(
    levels: int, tilt: float | None
) -> None:
    """12.7: the variance term of a draw from ``g0 r / sum g0 r`` is that law's score.

    A central finite difference of the composed log-likelihood in the GLM coefficients must
    equal :meth:`FittedModel.draw_score` with the declaration's log probabilities as the
    offset.  The offset varies over rows and levels, so a score that dropped it, or the
    reference level's term of a binary offset, would read the model's own law and differ.
    """
    from cleverly.learners import GLM

    rng = np.random.default_rng(7)
    n = 60
    x = rng.normal(size=(n, 2))
    y = rng.integers(0, levels, size=n).astype(float)
    glm = GLM(family="binomial" if levels == 2 else "multinomial").fit(x, y)
    model = gf.FittedModel(
        key="treatment:1",
        label="the natural-treatment model",
        role="treatment",
        response="binary" if levels == 2 else "multi",
        prefix=2,
        n_classes=levels,
        learner=glm,
        present=tuple(range(levels)),
        sigma2=0.0,
        rows=np.ones(n, dtype=bool),
        weights=np.ones(n),
        target=y,
    )
    declared = rng.dirichlet(np.ones(levels), size=n)
    offset = np.log(declared)
    drawn = rng.integers(0, levels, size=n).astype(float)
    score = model.draw_score(x, drawn, tilt=tilt, offset=offset)
    centre = np.array(glm.coef_, dtype=float, copy=True)
    flat = centre.reshape(-1)
    expected = np.zeros((n, flat.size))
    step = 1e-6
    for index in range(flat.size):
        sides = []
        for sign in (1.0, -1.0):
            moved = flat.copy()
            moved[index] += sign * step
            glm.coef_ = moved.reshape(centre.shape)
            sides.append(_composed_log_likelihood(model, x, drawn, offset, tilt))
        expected[:, index] = (sides[0] - sides[1]) / (2.0 * step)
    glm.coef_ = centre
    np.testing.assert_allclose(score, expected, rtol=0, atol=1e-7)
    # The witness is not trivial: the model's own score differs.
    own = model.draw_score(x, drawn, tilt=tilt)
    assert np.max(np.abs(own - expected)) > MISS


def test_w10_a_g_formula_sample_fit_with_glm_matches_the_target_fit_on_the_tilted_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """12.7: a ``"sample"`` g-formula with ``GLM()`` at node one is the tilted law's fit.

    The declaration at node one is one half for every unit, and the GLM on ``L0`` is
    saturated, so the tilt classifier returns the tilted law exactly and the composed natural
    law is ``P_w(A1 | L0)``.  A ``"target"`` fit that declares ``P_w(A1 | L0)`` itself draws
    from the same law with the same models, so its estimate is the same.  A ``"target"`` fit
    of the raw declaration draws from one half and differs.  The variance term of each drawn
    natural treatment receives the declaration as its offset, whose score
    :func:`test_w10_the_g_formula_draw_score_with_a_declaration_is_the_composed_law_s_score`
    checks.
    """
    from cleverly.learners import GLM

    offsets: list[Any] = []
    original = gf.FittedModel.draw_score

    def spy(self: Any, x: Any, drawn: Any, tilt: Any = None, offset: Any = None) -> Any:
        if self.key == "treatment:1":
            offsets.append(offset)
        return original(self, x, drawn, tilt, offset)

    monkeypatch.setattr(gf.FittedModel, "draw_score", spy)
    frame = _frame()
    regimens = {"x1": DynamicRegimen("x1", (Incremental(2.0), 1))}

    def gformula() -> Any:
        return ParametricGFormula(
            regimens,
            random_state=0,
            simultaneous=False,
            treatment_learner=GLM(),
            n_draws=4,
        )

    sample = _fit(gformula(), "sample", frame)
    assert offsets and all(
        offset is not None and np.allclose(offset, np.log(0.5)) for offset in offsets
    )
    tilted = frame.copy()
    for l0 in (0.0, 1.0):
        cell = (tilted["L0"] == l0).to_numpy()
        share = _share(frame[cell], "A1")
        tilted.loc[cell, "g1_1"] = share
        tilted.loc[cell, "g1_0"] = 1.0 - share
    target = _fit(gformula(), "target", tilted)
    raw = _fit(gformula(), "target", frame)
    name = "ey_regimen[x1]"
    assert abs(sample[name].psi - target[name].psi) <= 1e-8
    assert abs(sample[name].psi - raw[name].psi) > MISS
    assert np.isfinite(sample[name].std_error) and sample[name].std_error > 0.0


@pytest.mark.parametrize("role", ["treatment_learner", "censoring_learner"])
def test_w10_a_node_classifier_that_ignores_weights_is_refused(role: str) -> None:
    """12.5: without ``sample_weight`` the classifier returns the declaration itself."""
    model = _never()
    setattr(model, role, KNeighborsClassifier())
    NeverFit.calls = 0
    assert_refused_before_any_call(
        lambda: _fit(model, "sample"),
        None,
        "learner",
        unweighted_tilt_refusal("KNeighborsClassifier.fit"),
        error=MethodConfigurationError,
    )


def test_w10_the_declared_product_bound_reads_divided_factors_only() -> None:
    """A tilted factor is an estimate, which ``g_bounds`` truncates; it is not refused."""
    frame = _frame()
    with pytest.raises(CapabilityError, match="truncation bounds"):
        _fit(_never(g_bounds=(0.3, 1.0)), "target", frame)
    result = _fit(_estimator(LTMLE, g_bounds=(0.3, 1.0)), "sample", frame)
    assert np.isfinite(result["ey_regimen[always]"].psi)


# ------------------------------------------------------------------------------------ W12


def test_w12_a_subset_and_a_bootstrap_replicate_keep_the_reading() -> None:
    data = LongitudinalData.from_frame(
        _frame(), **COLUMNS, weights="w", **DECLARED, mechanism_law="sample"
    )
    assert data.subset(np.arange(0, data.n, 3)).known_mechanisms.law == "sample"
    seen: list[Any] = []

    def record(replicate: LongitudinalData) -> dict[str, tuple[float, float]]:
        seen.append(replicate.known_mechanisms.law)
        return {"x": (0.0, 1.0)}

    run_bootstrap(data, record, n_replicates=2, resampling="iid", random_state=4)
    assert seen == ["sample", "sample"]


def test_w12_the_markov_placebo_declares_target_unless_a_retention_is_tilted() -> None:
    """12.7: the placebo trajectory is independent of the history and the weight.

    So the kernel's law is each placebo node's tilted law (``"target"``).  One reading covers
    both factors, and a retention read in the sample stays tilted, so such data keeps
    ``"sample"``.
    """
    from cleverly.validation._markov_placebo import MarkovKernel, placebo_known_mechanisms

    frame = _frame()
    for declared, expected in (
        ({"treatment_probabilities": law.SMART_TREATMENT}, "target"),
        (DECLARED, "sample"),
    ):
        data = LongitudinalData.from_frame(
            frame, **COLUMNS, weights="w", **declared, mechanism_law="sample"
        )
        kernel = MarkovKernel.fit(data)
        placebo = kernel.placebo(data, np.random.default_rng(1))
        assert placebo_known_mechanisms(data, kernel, placebo).law == expected


@pytest.mark.parametrize(
    ("declared", "phrase"),
    [
        ({"treatment_probabilities": law.SMART_TREATMENT}, "keeps a known mechanism"),
        (DECLARED, "weight-tilted mechanism"),
    ],
    ids=["treatment_only", "tilted_retention"],
)
def test_w12_the_placebo_report_names_the_refit_s_reading(
    declared: dict[str, Any], phrase: str
) -> None:
    """The placebo detail says a refit keeps a known mechanism only when it divides by one.

    With a tilted retention the placebo law keeps ``"sample"``, so its treatment nodes are
    estimated tilts of the kernel and the detail says so.
    """
    from cleverly.validation import refute

    result = _estimator(LTMLE).fit(
        _frame(), **COLUMNS, weights="w", **declared, mechanism_law="sample"
    )
    report = refute(
        result,
        estimand="ate_regimen[never vs always]",
        tests=("placebo",),
        n_replicates=2,
        random_state=1,
    )
    placebo = next(test for test in report.tests if test.name == "placebo")
    assert phrase in placebo.detail
    if phrase != "keeps a known mechanism":
        assert "keeps a known mechanism" not in placebo.detail


def test_w12_simulated_confounding_carries_the_reading() -> None:
    """The flip of node one is independent of the weight given the unit's data, so the
    flipped declaration keeps its reading (Bayes' rule holds for the flipped treatment)."""
    from cleverly.sensitivity._simulated_confounding_longitudinal import _flipped_declaration

    data = LongitudinalData.from_frame(
        _frame(),
        **COLUMNS,
        weights="w",
        treatment_probabilities={"A1": law.SMART_TREATMENT["A1"]},
        mechanism_law="sample",
    )
    flipped = _flipped_declaration(data, 0.2)
    assert flipped.law == "sample"
    np.testing.assert_allclose(flipped.treatment[0][:, 1], 0.5)


def test_w12_the_nuisance_report_and_the_fingerprint_read_the_tilt_classifier() -> None:
    frame = _frame()
    sample = _fit(_estimator(LTMLE), "sample", frame)
    target = _fit(_divided(), "target", frame)
    rows = {(row.role, row.time) for row in _longitudinal_nuisances(sample).rows}
    assert {("treatment", 1), ("treatment", 2), ("censoring", 1)} <= rows
    omitted = {(item.role, item.time) for item in _longitudinal_nuisances(target).omissions}
    assert {("treatment", 1), ("treatment", 2), ("censoring", 1)} <= omitted
    name = "ey_regimen[always]"
    assert not fit_configuration(sample, name).known_mechanism
    assert fit_configuration(target, name).known_mechanism
    assert len(fit_configuration(sample, name).learners) > len(
        fit_configuration(target, name).learners
    )


def test_w12_the_data_summary_names_a_known_mechanism_for_target_only() -> None:
    frame = _frame()

    def header(mechanism_law: str) -> dict[str, str]:
        design = LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["L0"],
            time_varying=[[], ["L1"]],
            censoring=["C1", "C2"],
            weights="w",
            mechanism_law=mechanism_law,  # type: ignore[arg-type]
            **DECLARED,
        )
        return dict(CausalStudy(frame, design=design).summary().header)

    assert header("target")[vocabulary.TREATMENT_MODEL] == vocabulary.KNOWN_MECHANISM
    assert vocabulary.TREATMENT_MODEL not in header("sample")


def test_w12_a_design_carries_the_reading_and_checks_a_prepared_container() -> None:
    frame = _frame()
    design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["L0"],
        time_varying=[[], ["L1"]],
        censoring=["C1", "C2"],
        weights="w",
        mechanism_law="sample",
        **DECLARED,
    )
    study = CausalStudy(frame, design=design)
    assert study.data.known_mechanisms.law == "sample"
    result = study.identify(RegimeMean({"always": 1})).estimate(
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=law.WeightedCellClassifier(),
        censoring_learner=law.WeightedCellClassifier(),
        cross_fit=False,
    )
    (estimate,) = result.estimates.values()
    assert abs(estimate.psi - _static_truth(frame)["ey_regimen[always]"]) <= EXACT
    target = dataclasses.replace(design, mechanism_law="target")
    with pytest.raises(DataError, match="built with mechanism_law='sample'"):
        target.prepare(study.data)


def test_w12_a_reading_recorded_without_weights_takes_its_route_once_weighted() -> None:
    frame = _frame()
    data = LongitudinalData.from_frame(frame, **COLUMNS, **DECLARED, mechanism_law="sample")
    assert not tilts_node_declarations(data)
    weighted = LongitudinalData.from_frame(frame, **COLUMNS, weights="w")
    tilted = dataclasses.replace(data, weights=weighted.weights)
    assert tilts_node_declarations(tilted)
    result = _estimator(LTMLE).fit(tilted)
    _assert_exact(result, _static_truth(frame))


def test_w12_a_kept_tilt_model_is_not_read_as_the_factor_s_model() -> None:
    """Sequential IPW's stacked equations read a GLM factor's coefficients; a tilt is not one."""
    from cleverly.longitudinal.baselines import mechanism_reads_parametric

    data = LongitudinalData.from_frame(
        _frame(), **COLUMNS, weights="w", **DECLARED, mechanism_law="sample"
    )
    plans = resolve_plans(_static_regimens(), data)
    mechanism = fit_mechanism(
        data,
        plans,
        treatment_learner=LogisticRegression(),
        censoring_learner=LogisticRegression(),
        folds=Folds.single(data.n),
        keep_models=True,
    )
    assert all(isinstance(model, TiltedNodeModel) for model in mechanism.treatment_models)
    assert isinstance(mechanism.censoring_models[0], TiltedNodeModel)
    assert not mechanism_reads_parametric(mechanism)
