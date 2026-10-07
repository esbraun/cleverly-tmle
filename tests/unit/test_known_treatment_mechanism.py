"""A treatment mechanism the data declares known: exact laws, witnesses, readers and refusals.

The exact-law tests run on :mod:`tests.unit._known_mechanism_support`, whose empirical law is
the law.  There a TMLE that divides by the true mechanism solves ``P_n D*(Qbar*, g0) = 0`` and
has remainder zero, so its estimate equals the truth for any outcome regression (Moore and
van der Laan 2009, Section 2).  The fluctuation solver stops at a relative score of
``tol = 1e-10``; the tolerance below, :data:`EXACT`, is far above the error that leaves and
far below any difference a wrong mechanism makes.

The witnesses show that each piece is load-bearing: the known values change the estimate,
the curve and the DR-TMLE correction, and the bound rule fires.  The reader tests pin which
readers carry the declaration and which drop it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CapabilityError,
    CausalStudy,
    DataError,
    DRTMLEMethod,
    PointTreatment,
    TMLEMethod,
    load,
)
from cleverly.data import CausalData
from cleverly.data.known_mechanism import (
    CONTINUOUS_REFUSAL,
    KNOWN_CTMLE_REFUSAL,
    KNOWN_EVALUATION_REFUSAL,
    KNOWN_SCREEN_REFUSAL,
    KNOWN_TREATMENT_DELTA_REFUSAL,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.tmle import correction_parts
from cleverly.inference.bootstrap import run_bootstrap
from cleverly.interventions import Incremental, LearnedRule, Stochastic
from cleverly.validation import refute
from tests.unit import _known_mechanism_support as law
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The exact-law tolerance: the fluctuation stops at ``tol = 1e-10`` on its relative score.
EXACT = 1e-9
ROLES: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}


def _columns(arms: int) -> dict[float, str]:
    return {float(code): f"p{code}" for code in range(arms)}


def _forms(frame: pd.DataFrame, arms: int) -> dict[str, Any]:
    """The four declaration forms of the same mechanism.

    The two mappings list the levels in reverse order, so a parser that bound a mapping by
    position rather than by level would read another arm's column.
    """
    matrix = law.mechanism(frame, arms)
    forms: dict[str, Any] = {
        "columns": dict(reversed(list(_columns(arms).items()))),
        "mapping": {float(code): matrix[:, code] for code in reversed(range(arms))},
        "matrix": matrix,
    }
    if arms == 2:
        forms["vector"] = matrix[:, 1]
    return forms


def _wrong_outcome() -> LogisticRegression:
    """A main-terms logistic outcome regression, which the law's table is not."""
    return LogisticRegression(max_iter=1000)


def _tmle(**settings: Any) -> TMLE:
    base: dict[str, Any] = {
        "outcome_learner": _wrong_outcome(),
        "treatment_learner": NeverFit(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    base.update(settings)
    return TMLE(**base)


def _drtmle(**settings: Any) -> DRTMLE:
    base: dict[str, Any] = {
        "outcome_learner": _wrong_outcome(),
        "treatment_learner": NeverFit(),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ey0", "ey1", "ate"),
    }
    base.update(settings)
    return DRTMLE(**base)


class ReplicaFolds:
    """A mixin whose outer folds each hold an exact replica of the law.

    Row ``i`` goes to fold ``k mod V``, where ``k`` counts the rows before it with the same
    ``(cell, arm, Y)``.  Every such group is a multiple of ten rows, so each fold's empirical
    law is the law and the cross-fitted estimate is exact too.  The package draws its folds
    from a seed alone, so a test that needs this split overrides ``_folds``.
    """

    def _folds(self, data: CausalData, seed: int | None = None) -> Any:
        from cleverly.learners.crossfit import Folds

        keys = np.column_stack([data.covariates, data.treatment, data.outcome])
        _, groups = np.unique(keys, axis=0, return_inverse=True)
        order = np.zeros(data.n, dtype=np.int64)
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            order[rows] = np.arange(rows.size)
        return Folds(order % self.n_folds, self.n_folds)  # type: ignore[attr-defined]


class ReplicaTMLE(ReplicaFolds, TMLE):
    pass


class ReplicaDRTMLE(ReplicaFolds, DRTMLE):
    pass


def _assert_truth(result: Any, truth: dict[str, float]) -> None:
    compared = [name for name in result.estimates if name in truth]
    assert compared
    for name in compared:
        assert abs(result.estimates[name].psi - truth[name]) <= EXACT, name


# ----------------------------------------------------------------------------- E1


@pytest.mark.parametrize("arms", [2, 3])
@pytest.mark.parametrize("outcome", ["main_terms", "intercept"])
def test_a_known_mechanism_recovers_the_truth_under_a_wrong_outcome_regression(
    arms: int, outcome: str
) -> None:
    """E1: every arm mean and contrast equals the truth, whatever the outcome learner.

    The four declaration forms give bit-identical results.  The treatment learner is a
    ``NeverFit``, so the fit can only have divided by the declaration.
    """
    frame = law.point_frame(arms)
    truth = law.point_truth(arms)
    learner = _wrong_outcome() if outcome == "main_terms" else DummyClassifier(strategy="prior")
    results = {
        name: _tmle(outcome_learner=learner, estimands="all")
        .fit(frame, **ROLES, treatment_probabilities=form)
        .single()
        for name, form in _forms(frame, arms).items()
    }
    reference = results["columns"]
    _assert_truth(reference, truth)
    if arms == 2:
        for name in ("ey0", "ey1", "ate", "rr", "or", "att", "atc"):
            assert name in reference.estimates
    for result in results.values():
        assert result.estimates.keys() == reference.estimates.keys()
        for name, estimate in result.estimates.items():
            assert estimate.psi == reference.estimates[name].psi
            np.testing.assert_array_equal(
                estimate.influence_curve, reference.estimates[name].influence_curve
            )


def test_the_column_form_keeps_the_probability_columns_out_of_the_adjustment_set() -> None:
    frame = law.point_frame(2)
    data = CausalData.from_frame(
        frame.drop(columns=["wt"]),
        outcome="Y",
        treatment="A",
        treatment_probabilities=_columns(2),
    )
    assert data.covariate_names == ("W1", "W2")
    assert data.known_treatment is not None
    assert data.known_treatment.source == "columns"
    assert data.known_treatment.column_names == ("p0", "p1")


# ----------------------------------------------------------------------------- E2


@pytest.mark.parametrize(
    "settings",
    [
        {"targeting_scheme": "pooled"},
        {"targeting_scheme": "fold"},
        {"cv_evaluation": True},
    ],
    ids=["pooled", "fold", "cv_evaluation"],
)
@pytest.mark.parametrize("arms", [2, 3])
def test_a_cross_fitted_fit_on_replica_folds_recovers_the_truth(
    arms: int, settings: dict[str, Any]
) -> None:
    """E2: each CV-TMLE shape is exact when every fold is a replica of the law."""
    frame = law.point_frame(arms)
    estimands = ("ey", "ate", "att", "atc") if arms == 2 else ("ey", "ate")
    result = (
        ReplicaTMLE(
            outcome_learner=_wrong_outcome(),
            treatment_learner=NeverFit(),
            cross_fit=True,
            n_folds=5,
            estimands=estimands,
            simultaneous=False,
            random_state=0,
            **settings,
        )
        .fit(frame, **ROLES, treatment_probabilities=_columns(arms))
        .single()
    )
    _assert_truth(result, law.point_truth(arms))


# ----------------------------------------------------------------------------- E3


@pytest.mark.parametrize("reduction", ["univariate", "bivariate"])
@pytest.mark.parametrize("guard", [(), ("g",), ("Q",), ("Q", "g")], ids=["none", "g", "Q", "Qg"])
def test_complete_data_dr_tmle_with_a_known_mechanism_recovers_the_truth(
    guard: tuple[str, ...], reduction: str
) -> None:
    """E3: Benkeser et al. (2017), Theorem 1, at ``g_n = g0``.

    At guards ``()`` and ``("g",)`` the mechanism stays the declaration.  For the ``"Q"``
    guards the pass here is trivial: at ``P_n = P_0`` and ``g = g0``, ``P_n D_A = 0`` before
    any fluctuation, so equation (9) solves at ``epsilon_g = 0`` with or without the
    mechanism fluctuation.  :func:`test_the_known_mechanism_is_fluctuated_under_the_q_guard`
    covers that branch on a finite sample.
    """
    frame = law.point_frame(2)
    result = (
        _drtmle(guard=guard, reduction=reduction)
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    _assert_truth(result, law.point_truth(2))
    assert result.extra["missing_data"] == "complete"


# ----------------------------------------------------------------------------- E4


def _pair(make: Any, frame: pd.DataFrame, arms: int = 2, **roles: Any) -> tuple[Any, Any]:
    """A fit on the declaration and a fit whose treatment learner returns it."""
    roles = {**ROLES, **roles}
    declared = make(treatment_learner=NeverFit()).fit(
        frame, **roles, treatment_probabilities=_columns(arms)
    )
    fitted = make(treatment_learner=law.FixedMechanism(arms)).fit(frame, **roles)
    return declared.single(), fitted.single()


def _assert_same_fit(declared: Any, fitted: Any) -> None:
    assert declared.estimates.keys() == fitted.estimates.keys()
    for name, estimate in declared.estimates.items():
        other = fitted.estimates[name]
        np.testing.assert_allclose(estimate.psi, other.psi, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(estimate.std_error, other.std_error, rtol=1e-12)
        np.testing.assert_allclose(
            estimate.influence_curve, other.influence_curve, rtol=1e-12, atol=1e-14
        )


def _finite_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """A finite draw on the exact law's cells, so a fitted mechanism can equal the declared one."""
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, len(law.CELLS), size=n)
    w = np.array([law.CELLS[cell] for cell in cells])
    g1 = np.array([law.BINARY_MECHANISM[cell] for cell in cells])
    a = rng.binomial(1, g1).astype(float)
    q = np.array([law.OUTCOME[cell][int(arm)] for cell, arm in zip(cells, a, strict=True)])
    y = rng.binomial(1, q).astype(float)
    observed = rng.binomial(1, expit(1.2 + 0.5 * w[:, 0])).astype(float)
    return pd.DataFrame(
        {
            "W1": w[:, 0],
            "W2": w[:, 1],
            "A": a,
            "Y": y,
            "p0": 1.0 - g1,
            "p1": g1,
            "Delta": observed,
            "Ymiss": np.where(observed == 1.0, y, np.nan),
            "wt": 1.0 + 0.5 * (w[:, 1] == 1.0),
        }
    )


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda **kw: _tmle(estimands="all", **kw), id="tmle"),
        pytest.param(lambda **kw: _tmle(cross_fit=True, n_folds=5, **kw), id="cv_tmle"),
        pytest.param(lambda **kw: _drtmle(**kw), id="drtmle"),
        pytest.param(lambda **kw: _drtmle(cross_fit=True, n_folds=4, **kw), id="cv_drtmle"),
        pytest.param(
            lambda **kw: _drtmle(cross_fit=True, n_folds=4, reduced_crossfit="nested", **kw),
            id="nested_drtmle",
        ),
        pytest.param(lambda **kw: _tmle(n_bootstrap=4, estimands=("ate",), **kw), id="bootstrap"),
    ],
)
def test_a_known_mechanism_equals_a_learner_that_returns_it(make: Any) -> None:
    """E4: the declaration and a treatment learner returning the same values agree.

    The nested DR-TMLE reuses the declaration in every inner fold, which is what a learner
    that returns it gives, so the nested construction needs no refit of the mechanism.
    """
    declared, fitted = _pair(make, _finite_frame())
    _assert_same_fit(declared, fitted)
    if declared.bootstrap is not None:
        for name, draws in declared.bootstrap.draws.items():
            np.testing.assert_allclose(draws, fitted.bootstrap.draws[name], rtol=1e-12)


# ----------------------------------------------------------------------------- E5


def test_the_reported_curve_is_the_efficient_curve_at_the_known_mechanism() -> None:
    """E5: arm means, ATE, ATT and ATC curves equal ``D*(Qbar*, g0)`` written out here."""
    frame = _finite_frame()
    result = (
        _tmle(estimands=("ey1", "ey0", "ate", "att", "atc"))
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    a = frame["A"].to_numpy()
    y = frame["Y"].to_numpy()
    g1 = frame["p1"].to_numpy()
    mean = result.fluctuations["mean"].targeted
    q1, q0 = mean.arms[1.0], mean.arms[0.0]
    qa = np.where(a == 1.0, q1, q0)
    ey1 = a / g1 * (y - qa) + q1 - q1.mean()
    ey0 = (1 - a) / (1 - g1) * (y - qa) + q0 - q0.mean()
    np.testing.assert_allclose(result.estimates["ey1"].influence_curve, ey1, atol=1e-12)
    np.testing.assert_allclose(result.estimates["ey0"].influence_curve, ey0, atol=1e-12)
    np.testing.assert_allclose(result.estimates["ate"].influence_curve, ey1 - ey0, atol=1e-12)
    p = a.mean()
    att = result.fluctuations["att"].targeted
    t1, t0 = att.arms[1.0], att.arms[0.0]
    ta = np.where(a == 1.0, t1, t0)
    psi = result.estimates["att"].psi
    curve = (a - (1 - a) * g1 / (1 - g1)) * (y - ta) / p + a * (t1 - t0 - psi) / p
    np.testing.assert_allclose(result.estimates["att"].influence_curve, curve, atol=1e-12)
    atc = result.fluctuations["atc"].targeted
    c1, c0 = atc.arms[1.0], atc.arms[0.0]
    ca = np.where(a == 1.0, c1, c0)
    psi = result.estimates["atc"].psi
    curve = (a * (1 - g1) / g1 - (1 - a)) * (y - ca) / (1 - p) + (1 - a) * (c1 - c0 - psi) / (1 - p)
    np.testing.assert_allclose(result.estimates["atc"].influence_curve, curve, atol=1e-12)


# ----------------------------------------------------------------------------- E6, E7, E8, E9


def test_a_missing_outcome_tmle_divides_by_the_known_mechanism() -> None:
    """E6: ``delta=`` with a known mechanism is the ordinary missing-outcome TMLE at ``g0``."""
    frame = _finite_frame()

    def make(**kw: Any) -> TMLE:
        return _tmle(missingness_learner=LogisticRegression(max_iter=1000), **kw)

    declared, fitted = _pair(make, frame, delta="Delta", outcome="Ymiss")
    _assert_same_fit(declared, fitted)
    assert declared.extra["missing_data"] == "missing_outcome"


def test_a_learned_rule_fit_divides_by_the_known_mechanism() -> None:
    """E7: the learned rule reads the outcome regression only, so the mechanism is a plug-in."""

    def make(**kw: Any) -> TMLE:
        return _tmle(
            cross_fit=True, cv_evaluation=True, n_folds=4, learned_rule=LearnedRule(), **kw
        )

    declared, fitted = _pair(make, _finite_frame())
    _assert_same_fit(declared, fitted)


def _tilt(g1: np.ndarray, delta: float) -> np.ndarray:
    one = delta * g1 / (delta * g1 + 1.0 - g1)
    return np.column_stack([1.0 - one, one])


def _cell_mechanism(frame: Any) -> np.ndarray:
    values = np.asarray(frame[["W1", "W2"]], dtype=float)
    return np.array(
        [law.BINARY_MECHANISM[law.CELLS.index((float(a), float(b)))] for a, b in values]
    )


def test_an_incremental_fit_on_a_known_mechanism_is_the_known_stochastic_regime() -> None:
    """E8: ``incremental=`` on a declared mechanism is the regime ``q_delta(g0)``.

    With ``g0`` known the model has no mechanism scores, so the curve is the regime curve,
    without Kennedy's (2019) term in ``(A - g)``.  The witness: the same tilt on an
    estimated mechanism carries that term, so its curve differs.
    """
    frame = _finite_frame()
    delta = 2.0
    incremental = (
        _tmle(incremental=[Incremental(delta)])
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    regime = (
        _tmle(
            interventions=[
                Stochastic(lambda w: _tilt(_cell_mechanism(w), delta), "tilt", density_kind="known")
            ],
            treatment_learner=law.FixedMechanism(2),
        )
        .fit(frame, **ROLES)
        .single()
    )
    (ipsi,) = incremental.estimates.values()
    stochastic = regime.estimates["ey_regime[tilt]"]
    np.testing.assert_allclose(ipsi.psi, stochastic.psi, rtol=1e-10)
    np.testing.assert_allclose(ipsi.influence_curve, stochastic.influence_curve, atol=1e-10)
    estimated = (
        _tmle(incremental=[Incremental(delta)], treatment_learner=law.FixedMechanism(2))
        .fit(frame, **ROLES)
        .single()
    )
    (with_term,) = estimated.estimates.values()
    assert np.max(np.abs(with_term.influence_curve - ipsi.influence_curve)) > 1e-3


def test_a_weighted_fit_recovers_the_tilted_truth() -> None:
    """E9: weights that depend on ``W`` only keep the design mechanism the tilted one."""
    frame = law.point_frame(2)
    result = (
        _tmle(estimands="all")
        .fit(frame, **ROLES, weights="wt", treatment_probabilities=_columns(2))
        .single()
    )
    _assert_truth(result, law.point_truth(2, weighted=True))


# ----------------------------------------------------------------------------- witnesses


def _confounded_frame(n: int = 8000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    g1 = np.where(w2 == 1.0, 0.75, 0.2)
    a = rng.binomial(1, g1).astype(float)
    y = rng.binomial(1, expit(-0.4 + 0.5 * a + 1.5 * w2 + 0.8 * a * w2 - 0.3 * w1)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "p0": 1 - g1, "p1": g1})


def test_a_known_mechanism_and_an_intercept_only_estimate_differ() -> None:
    """W1: dividing by the declaration is not dividing by a poor estimate."""
    frame = _confounded_frame()
    outcome = DummyClassifier(strategy="prior")
    known = (
        _tmle(outcome_learner=outcome, estimands=("ate",))
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    estimated = (
        _tmle(
            outcome_learner=outcome,
            treatment_learner=DummyClassifier(strategy="prior"),
            estimands=("ate",),
        )
        .fit(frame, **ROLES)
        .single()
    )
    gap = abs(known.estimates["ate"].psi - estimated.estimates["ate"].psi)
    assert gap > 10 * known.estimates["ate"].std_error


def test_the_curve_moves_with_the_declared_values() -> None:
    """W2: moving ``g0`` by 0.05 moves every treated row's curve value."""
    frame = _finite_frame()
    g1 = frame["p1"].to_numpy()
    base = _tmle(estimands=("ey1",)).fit(frame, **ROLES, treatment_probabilities=g1).single()
    moved = (
        _tmle(estimands=("ey1",))
        .fit(frame, **ROLES, treatment_probabilities=np.clip(g1 + 0.05, 0.05, 0.95))
        .single()
    )
    treated = frame["A"].to_numpy() == 1.0
    gap = np.abs(base.estimates["ey1"].influence_curve - moved.estimates["ey1"].influence_curve)[
        treated
    ]
    assert np.min(gap) > 0.0
    assert np.max(gap) > 1e-3


def _continuous_frame(n: int = 800, seed: int = 2) -> pd.DataFrame:
    """A finite draw whose declared mechanism varies continuously with ``W1``."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    g1 = expit(0.9 * w1 - 0.5 * w2)
    a = rng.binomial(1, g1).astype(float)
    eta = -0.3 + 0.7 * a + 1.2 * w1 + 0.6 * w1 * a + 0.8 * w2
    y = rng.binomial(1, expit(eta)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "p0": 1 - g1, "p1": g1})


def _q_guard(guard: tuple[str, ...]) -> Any:
    """A DR-TMLE fit with an intercept-only outcome regression on :func:`_continuous_frame`."""
    return (
        _drtmle(guard=guard, outcome_learner=DummyClassifier(strategy="prior"), estimands=("ate",))
        .fit(_continuous_frame(), **ROLES, treatment_probabilities=_columns(2))
        .single()
    )


def test_the_mechanism_correction_is_nonzero_under_the_q_guard() -> None:
    """W3: with a wrong outcome regression, ``D_A`` is nonzero and moves the standard error."""
    guarded = _q_guard(("Q",))
    plain = _q_guard(())
    fluctuation = guarded.fluctuations["mean"]
    parts = correction_parts(
        guarded.data,
        guarded.nuisance,
        fluctuation,
        fluctuation.targeted,
        guarded.nuisance.scaler.scale(guarded.data.outcome),
    )
    assert parts is not None
    assert max(float(np.max(np.abs(values))) for values in parts.d_g.values()) > 1e-3
    gap = abs(guarded.estimates["ate"].std_error - plain.estimates["ate"].std_error)
    assert gap > 1e-3


def test_the_known_mechanism_is_fluctuated_under_the_q_guard() -> None:
    """W5: equation (9) moves ``g*`` off the declaration on a finite sample.

    The detector of the mutation that skips the mechanism fluctuation when the mechanism is
    known.  At the initial reductions, ``P_n D_A`` is far from zero; after the alternation the
    mechanism score is solved, and the targeted mechanism is no longer ``g0``.
    """
    result = _q_guard(("Q",))
    frame = _continuous_frame()
    a = frame["A"].to_numpy()
    g1 = frame["p1"].to_numpy()
    qr = np.asarray(result.nuisance.reduced.qr, dtype=float)
    initial = [
        float(np.mean(qr[:, column] / g * ((a == arm) - g)))
        for column, (arm, g) in enumerate([(0.0, 1.0 - g1), (1.0, g1)])
    ]
    assert max(abs(value) for value in initial) > 1e-3
    mechanism = result.fluctuations["mean"].mechanism
    assert mechanism is not None
    assert mechanism.relative_score <= 1e-6
    targeted = np.asarray(mechanism.propensity, dtype=float)
    targeted = targeted if targeted.ndim == 1 else targeted[:, 1]
    assert float(np.max(np.abs(targeted - g1))) > 1e-4


@pytest.mark.parametrize(
    "estimands, values, n, fragment",
    [
        (("ate",), (0.01,), 500, "[0.03598, 0.964]"),
        (("att",), (0.02,), 100_000, "[0.025, 0.975]"),
    ],
    ids=["ate_pair", "att_pair"],
)
def test_the_bound_rule_refuses_a_bound_that_moves_a_known_value(
    estimands: tuple[str, ...], values: tuple[float, ...], n: int, fragment: str
) -> None:
    """W4: a bound pair the fit uses that would move a declared value is refused first."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "W1": rng.normal(size=n),
            "A": np.tile([0.0, 1.0], n // 2),
            "Y": np.tile([0.0, 1.0], n // 2),
        }
    )
    g1 = np.full(n, 0.5)
    g1[0] = values[0]
    raised = assert_refused_before_any_call(
        lambda: TMLE(cross_fit=False, estimands=estimands, **never_fit_learners()).fit(
            frame, outcome="Y", treatment="A", covariates=["W1"], treatment_probabilities=g1
        ),
        None,
        "learner",
        "treatment_probabilities has 1 values outside the truncation bounds",
        fragment,
        "Pass g_bounds=(lower, upper) that contains every known probability.",
    )
    assert "smallest" in str(raised)


def test_a_bound_that_contains_every_known_value_reaches_the_learners() -> None:
    """W4, the other half: the ATE pair at ``n = 100,000`` keeps 0.02, and so does a pair given."""
    n = 100_000
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "W1": rng.normal(size=n),
            "A": np.tile([0.0, 1.0], n // 2),
            "Y": np.tile([0.0, 1.0], n // 2),
        }
    )
    g1 = np.full(n, 0.5)
    g1[0] = 0.02
    for settings in ({"estimands": ("ate",)}, {"estimands": ("att",), "g_bounds": (0.01, 0.99)}):
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            TMLE(cross_fit=False, **settings, **never_fit_learners()).fit(
                frame, outcome="Y", treatment="A", covariates=["W1"], treatment_probabilities=g1
            )
        assert NeverFit.calls > 0


# ----------------------------------------------------------------------------- readers


def test_a_subset_and_a_bootstrap_replicate_carry_their_own_rows() -> None:
    """W6: the declaration follows ``subset``, which the bootstrap and refutation use."""
    frame = _finite_frame()
    data = CausalData.from_frame(frame, **ROLES, treatment_probabilities=_columns(2))
    assert data.known_treatment is not None
    index = np.array([5, 5, 0, 17, 3, *range(20, 60)])
    sub = data.subset(index)
    assert sub.known_treatment is not None
    np.testing.assert_array_equal(sub.known_treatment.values, data.known_treatment.values[index])
    seen: list[np.ndarray] = []

    def record(replicate: CausalData) -> dict[str, float]:
        assert replicate.known_treatment is not None
        seen.append(np.asarray(replicate.known_treatment.values))
        return {"x": 0.0}

    run_bootstrap(data, record, n_replicates=2, resampling="iid", random_state=4)
    rng_index = [np.asarray(values) for values in seen]
    assert all(values.shape == data.known_treatment.values.shape for values in rng_index)
    assert not np.array_equal(rng_index[0], data.known_treatment.values)


def test_a_rewritten_treatment_drops_the_declaration() -> None:
    """W6: a permuted treatment has a mechanism of its own, so the copy declares none."""
    data = CausalData.from_frame(_finite_frame(), **ROLES, treatment_probabilities=_columns(2))
    permuted = data.with_treatment(np.random.default_rng(0).permutation(data.treatment))
    assert permuted.known_treatment is None
    same = data.with_treatment(data.treatment.copy())
    assert same.known_treatment is data.known_treatment


def test_the_placebo_refutation_names_the_dropped_declaration() -> None:
    frame = _finite_frame()
    result = (
        _tmle(treatment_learner=LogisticRegression(max_iter=1000), estimands=("ate",))
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    report = refute(result, tests=("placebo", "subset"), n_replicates=2, random_state=1)
    placebo = next(test for test in report.tests if test.name == "placebo")
    assert "The known treatment mechanism was dropped" in placebo.detail
    subset = next(test for test in report.tests if test.name == "subset")
    assert "dropped" not in subset.detail


def test_a_saved_result_reports_the_known_mechanism(tmp_path: Any) -> None:
    frame = _finite_frame()
    result = (
        _tmle(estimands=("ate",)).fit(frame, **ROLES, treatment_probabilities=_columns(2)).single()
    )
    assert result.config.treatment_mechanism == "known"
    assert "treatment mechanism: known (declared)" in result.summary()
    assert list(result.to_frame()["treatment_mechanism"]) == ["known"]
    assert result.diagnostics.support().treatment_mechanism == "known"
    path = tmp_path / "known.cleverly"
    result.save(path)
    restored = load(path)
    assert restored.config.treatment_mechanism == "known"
    assert restored.nuisance.treatment_mechanism == "known"


def test_a_refit_replays_a_known_mechanism_fit_bit_for_bit() -> None:
    frame = _finite_frame()
    result = (
        _tmle(estimands=("ate", "att"))
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    replayed = result.estimator.refit(result.data)
    for name, estimate in result.estimates.items():
        assert replayed.estimates[name].psi == estimate.psi
        np.testing.assert_array_equal(
            replayed.estimates[name].influence_curve, estimate.influence_curve
        )


def test_an_estimated_fit_records_an_estimated_mechanism() -> None:
    frame = _finite_frame()
    result = (
        _tmle(treatment_learner=LogisticRegression(max_iter=1000), estimands=("ate",))
        .fit(frame, **ROLES)
        .single()
    )
    assert result.config.treatment_mechanism == "estimated"
    assert "treatment_mechanism" not in result.to_frame().columns
    assert "treatment mechanism: known" not in result.summary()


# ----------------------------------------------------------------------------- refusals


def _trial(n: int = 200) -> pd.DataFrame:
    frame = _finite_frame(n=n)
    return frame.assign(DeltaA=np.where(np.arange(n) % 7 == 0, 0.0, 1.0))


REFUSALS: dict[str, tuple[Any, type[Exception], tuple[str, ...]]] = {
    "ctmle": (
        lambda: CTMLE(cross_fit=False, estimands=("ate",), **never_fit_learners()).fit(
            _trial(), **ROLES, treatment_probabilities=_columns(2)
        ),
        CapabilityError,
        (KNOWN_CTMLE_REFUSAL,),
    ),
    "continuous": (
        lambda: TMLE(cross_fit=False, **never_fit_learners()).fit(
            _trial(),
            outcome="Y",
            treatment="W1",
            covariates=["W2"],
            treatment_kind="continuous",
            treatment_probabilities=np.full(200, 0.5),
        ),
        CapabilityError,
        (CONTINUOUS_REFUSAL,),
    ),
    "treatment_delta_tmle": (
        lambda: TMLE(cross_fit=False, estimands=("ate",), **never_fit_learners()).fit(
            _trial(), **ROLES, treatment_delta="DeltaA", treatment_probabilities=_columns(2)
        ),
        CapabilityError,
        (KNOWN_TREATMENT_DELTA_REFUSAL,),
    ),
    "treatment_delta_drtmle": (
        lambda: DRTMLE(
            cross_fit=False,
            estimands=("ate",),
            reduced_outcome_learner=NeverFit(),
            reduced_treatment_learner=NeverFit(),
            **never_fit_learners(),
        ).fit(_trial(), **ROLES, treatment_delta="DeltaA", treatment_probabilities=_columns(2)),
        CapabilityError,
        (KNOWN_TREATMENT_DELTA_REFUSAL,),
    ),
    "screen_treatment": (
        lambda: TMLE(cross_fit=False, screen_treatment=True, **never_fit_learners()).fit(
            _trial(), **ROLES, treatment_probabilities=_columns(2)
        ),
        ValueError,
        (KNOWN_SCREEN_REFUSAL,),
    ),
    "evaluation_without_columns": (
        lambda: DRTMLE(
            cross_fit=False,
            estimands=("ate",),
            evaluation=_trial().drop(columns=["p0", "p1"]),
            reduced_outcome_learner=NeverFit(),
            reduced_treatment_learner=NeverFit(),
            **never_fit_learners(),
        ).fit(_trial(), **ROLES, treatment_probabilities=_columns(2)),
        CapabilityError,
        (KNOWN_EVALUATION_REFUSAL,),
    ),
    "evaluation_from_arrays": (
        lambda: DRTMLE(
            cross_fit=False,
            estimands=("ate",),
            evaluation=_trial(),
            reduced_outcome_learner=NeverFit(),
            reduced_treatment_learner=NeverFit(),
            **never_fit_learners(),
        ).fit(_trial(), **ROLES, treatment_probabilities=np.full(200, 0.5)),
        CapabilityError,
        (KNOWN_EVALUATION_REFUSAL,),
    ),
    "declared_twice": (
        lambda: TMLE(cross_fit=False, **never_fit_learners()).fit(
            CausalData.from_frame(_trial(), **ROLES, treatment_probabilities=_columns(2)),
            treatment_probabilities=np.full(200, 0.5),
        ),
        ValueError,
        ("treatment_probabilities is declared twice: on the data and at fit. Declare it once.",),
    ),
}


@pytest.mark.parametrize("row", sorted(REFUSALS))
def test_each_refusal_fires_before_any_learner(row: str) -> None:
    build, error, fragments = REFUSALS[row]
    assert_refused_before_any_call(build, None, "learner", *fragments, error=error)


def test_a_companion_with_the_declared_columns_reaches_the_learners() -> None:
    """The evaluation companion is admitted when it declares the same columns."""
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        DRTMLE(
            cross_fit=False,
            estimands=("ate",),
            evaluation=_trial(),
            reduced_outcome_learner=NeverFit(),
            reduced_treatment_learner=NeverFit(),
            **never_fit_learners(),
        ).fit(_trial(), **ROLES, treatment_probabilities=_columns(2))
    assert NeverFit.calls > 0


def test_the_omitted_variable_benchmark_is_refused_on_a_known_mechanism() -> None:
    from cleverly.sensitivity.omitted_variable import benchmark_refusal

    frame = _finite_frame()
    result = (
        _tmle(estimands=("ate",)).fit(frame, **ROLES, treatment_probabilities=_columns(2)).single()
    )
    reason = benchmark_refusal(result, ["W1"])
    assert reason is not None and "declares the mechanism known" in reason


# ----------------------------------------------------------------------------- study API


def test_a_point_design_declares_the_mechanism_for_every_method() -> None:
    frame = _finite_frame()
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        treatment_probabilities={0.0: "p0", 1.0: "p1"},
    )
    study = CausalStudy(frame, design=design)
    assert study.data.covariate_names == ("W1", "W2")
    effect = study.identify(ATE())
    availability = {item.name: item for item in effect.available_methods()}
    assert not availability["collaborative_tmle"].available
    assert availability["collaborative_tmle"].reason == KNOWN_CTMLE_REFUSAL
    with pytest.raises(CapabilityError, match="nothing to select"):
        effect.estimate("collaborative_tmle")
    learners = {"outcome_learner": _wrong_outcome(), "treatment_learner": NeverFit()}
    reduced = {
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
    }
    for method in (TMLEMethod(), DRTMLEMethod(guard=("Q",), **reduced)):
        result = effect.estimate(method.with_overrides(**learners, cross_fit=False, random_state=0))
        assert result.config.treatment_mechanism == "known"


def test_a_point_design_names_columns_only() -> None:
    with pytest.raises(DataError, match="as every other design field names columns"):
        PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1",), treatment_probabilities=np.ones(3)
        )


def test_the_evaluation_companion_divides_by_its_own_declaration() -> None:
    """Each fold's companion mechanism is the companion's declared values, not a prediction."""
    frame = _finite_frame()
    companion = _finite_frame(n=300, seed=99)
    result = (
        _drtmle(
            guard=("Q", "g"), cross_fit=True, n_folds=3, estimands=("ate",), evaluation=companion
        )
        .fit(frame, **ROLES, treatment_probabilities=_columns(2))
        .single()
    )
    evaluated = result.nuisance.companion
    assert evaluated is not None
    for propensity in evaluated.propensity:
        np.testing.assert_array_equal(propensity.values, law.mechanism(companion, 2))


def test_the_array_entry_point_reads_g1w_as_the_known_mechanism() -> None:
    """``tmle(Y, A, W, g1W=...)`` is R ``tmle``'s known ``P(A = 1 | W)``."""
    from cleverly.estimators.tmle import tmle

    frame = _finite_frame()
    g1 = frame["p1"].to_numpy()
    settings = {
        "outcome_learner": _wrong_outcome(),
        "treatment_learner": NeverFit(),
        "cross_fit": False,
        "simultaneous": False,
        "estimands": ("ate",),
        "random_state": 0,
    }
    arrays = tmle(
        frame["Y"].to_numpy(),
        frame["A"].to_numpy(),
        frame[["W1", "W2"]].to_numpy(),
        g1W=g1,
        **settings,
    ).single()
    columns = TMLE(**settings).fit(frame, **ROLES, treatment_probabilities=g1).single()
    assert arrays.estimates["ate"].psi == columns.estimates["ate"].psi


# ----------------------------------------------------------------------------- review additions


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda **kw: _tmle(**kw), id="tmle"),
        pytest.param(lambda **kw: _tmle(cross_fit=True, n_folds=5, **kw), id="cv_tmle"),
    ],
)
def test_a_three_arm_known_mechanism_equals_a_learner_that_returns_it(make: Any) -> None:
    """E4 on three arms: the declaration and a learner returning it give one fit."""
    declared, fitted = _pair(make, law.point_frame(3), arms=3)
    _assert_same_fit(declared, fitted)


def _treat_w2(a: Any, h: Any) -> Any:
    return np.where(np.asarray(h["W2"], dtype=float) == 1.0, 1.0, np.asarray(a, dtype=float))


def _drop_top(a: Any, h: Any) -> Any:
    values = np.asarray(a, dtype=float)
    return np.where(values == 2.0, 1.0, values)


@pytest.mark.parametrize("arms", [2, 3])
def test_a_discrete_modified_policy_divides_by_the_known_mechanism(arms: int) -> None:
    """A discrete MTP on a declared mechanism equals the fit whose learner returns it.

    The discrete formula reads ``g`` only in the ratio ``g^d / g``, so a declared ``g`` is
    the degenerate estimate ``g_n = g0`` and the shipped curve applies unchanged.
    """
    from cleverly.interventions import ModifiedPolicy

    apply = _treat_w2 if arms == 2 else _drop_top
    policy = ModifiedPolicy("mtp", apply=apply, policy_kind="known")
    frame = law.point_frame(arms) if arms == 3 else _finite_frame()
    declared, fitted = _pair(lambda **kw: _tmle(policies=[policy], **kw), frame, arms=arms)
    _assert_same_fit(declared, fitted)
    assert declared.config.treatment_mechanism == "known"


def test_the_classifier_ratio_is_refused_beside_a_declared_mechanism() -> None:
    from cleverly.data.known_mechanism import KNOWN_POLICY_CLASSIFIER_REFUSAL
    from cleverly.interventions import ModifiedPolicy

    policy = ModifiedPolicy("mtp", apply=_treat_w2, policy_kind="known")
    assert_refused_before_any_call(
        lambda: TMLE(
            cross_fit=False, policies=[policy], ratio="classifier", **never_fit_learners()
        ).fit(_trial(), **ROLES, treatment_probabilities=_columns(2)),
        None,
        "learner",
        KNOWN_POLICY_CLASSIFIER_REFUSAL,
        error=CapabilityError,
    )


def test_the_flipped_mechanism_is_the_mixture_the_flip_induces() -> None:
    from cleverly.sensitivity.simulated_confounding import flipped_mechanism

    g1 = np.array([0.2, 0.5, 0.75])
    declared = np.column_stack([1.0 - g1, g1])
    flipped = flipped_mechanism(declared, 0.1)
    np.testing.assert_allclose(flipped[:, 1], 0.9 * g1 + 0.1 * (1.0 - g1), rtol=1e-15)
    np.testing.assert_allclose(flipped.sum(axis=1), 1.0, rtol=1e-15)
    np.testing.assert_array_equal(flipped_mechanism(declared, 0.0), declared)


def test_simulated_confounding_carries_the_flipped_mechanism_in_every_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every cell divides by a known mechanism, so the surface measures confounding only.

    The treatment learner is a ``NeverFit``: a cell that dropped the declaration would fit
    it and fail.  At a 1% flip the displacement stays near zero, where an intercept-only
    refit of the mechanism moved it by about 0.056 (implementation review, probe 5).
    """
    from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding

    frame = _confounded_frame(n=2000)
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        treatment_probabilities=_columns(2),
    )
    result = (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            TMLEMethod().with_overrides(
                outcome_learner=_wrong_outcome(),
                treatment_learner=NeverFit(),
                cross_fit=False,
                random_state=0,
            )
        )
    )
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.01, 0.2), outcome=(0.0, 0.1))
    declared: list[np.ndarray] = []
    original = CausalData.with_known_mechanism

    def spy(self: CausalData, treatment_probabilities: Any) -> CausalData:
        declared.append(np.asarray(treatment_probabilities, dtype=float))
        return original(self, treatment_probabilities)

    monkeypatch.setattr(CausalData, "with_known_mechanism", spy)
    NeverFit.calls = 0
    surface = simulated_confounding(result, "ate", grid=grid, random_state=3)
    g0 = np.column_stack([frame["p0"], frame["p1"]])
    # Two outcome strengths at each nonzero treatment strength, in grid order.
    assert len(declared) == 4
    for values, strength in zip(declared, (0.01, 0.01, 0.2, 0.2), strict=True):
        np.testing.assert_allclose(values, (1 - strength) * g0 + strength * g0[:, ::-1], rtol=1e-15)
    assert surface.complete, [cell.failure for cell in surface.cells if cell.failure]
    assert NeverFit.calls == 0
    assert surface.known_mechanism_carried
    assert any("declares the flipped mechanism" in line for line in surface.population_lines())
    small = next(
        cell
        for cell in surface.cells
        if cell.treatment_strength == 0.01 and cell.outcome_strength == 0.0
    )
    assert abs(small.displacement) < 0.02
