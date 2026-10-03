r"""The composite-indicator construction for an observational missing outcome or treatment.

For arm ``a`` the fit targets :math:`C_a = \Delta_A \Delta 1\{A = a\}` with the mechanism
:math:`g_{c,a} = \pi_A(W) g(a \mid \Delta_A = 1, W) \pi(a, W)`, and stacks the per-arm
estimators (``docs/technical-reference/dr-tmle/theorem.md``).  The evidence here:

* **exact law** (:mod:`tests.discrete_law_composite`, E1-E4).  At the oracle factors the
  estimate is the truth and the curve is the EIF.  That check is blind to the corrections,
  so an outcome drift and a mechanism drift make each correction live while the estimate
  stays exact, and both drifts together make it miss;
* **nonzero witnesses and mutation controls** (E5-E12, E19).  Each removes or replaces one
  piece of the construction and lands on a limit the law module computes at import;
* **exact reductions** (E13, E14) to the shipped complete-data and missing-outcome fits;
* **an independent per-arm reference** (E18): Benkeser et al.'s steps on
  ``(W, C_a, C_a Y)``, written here and sharing no code with the package;
* **integration** (E15, E16, E20, E21).

Every mutation is a monkeypatch; no source file is edited.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.estimators._nuisance as nuisance_module
import cleverly.estimators.composite as composite
import cleverly.estimators.drtmle as drtmle_module
import cleverly.estimators.reduced as reduced_module
import cleverly.estimators.targeting as targeting
import cleverly.validation.drtmle as validation_drtmle
from cleverly import load
from cleverly.data import CausalData
from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.composite import (
    composite_bounds,
    composite_mechanism,
    composite_state,
    composite_view,
    targeting_inputs,
)
from cleverly.estimators.targeting import build_submodel
from cleverly.estimators.tmle import correction_parts
from tests import discrete_law_composite as dl
from tests import discrete_law_multi as multi
from tests.studies.mar_arm_indexed_laws import LawOutcome, LawResponse, LawTreatment, sample

#: The module, not the ``cleverly.estimators.tmle`` attribute, which a function shadows.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
LAWS = tuple(dl.LAWS)
GUARDS = (("Q", "g"), ("Q",), ("g",), ())


def _reduced_learners() -> dict[str, Any]:
    return {
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(C=1e6, max_iter=2000),
    }


def _estimands(law: dl.CompositeLaw) -> tuple[str, ...]:
    return ("ey0", "ey1", "ate", "rr", "or") if law.k == 2 else ("ey", "ate", "rr", "or")


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "cross_fit": False,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "nuisance_bound": NUISANCE_BOUND,
        "tol": 1e-10,
        "random_state": 0,
    }
    settings.update(overrides)
    return settings


def _engine(guard: tuple[str, ...] | None, reduction: str = "univariate", **settings: Any) -> Any:
    """``TMLE`` for ``guard=None``, else a ``DRTMLE`` at that guard and reduction."""
    if guard is None:
        return TMLE(**settings)
    settings = {"max_outer": 40, **_reduced_learners(), **settings}
    return DRTMLE(guard=guard, reduction=reduction, **settings)


def _oracle_fit(
    key: str,
    *,
    guard: tuple[str, ...] | None = ("Q", "g"),
    reduction: str = "univariate",
    mu: np.ndarray | None = None,
    drift_mechanism: bool = False,
    weighted: bool = False,
    frame: pd.DataFrame | None = None,
    **overrides: Any,
) -> Any:
    """A fit of one exact law with law-table nuisances, at a guard (``None`` is ``TMLE``)."""
    law = dl.LAWS[key]
    oracle = law.law()
    learners: dict[str, Any] = {
        "outcome_learner": LawOutcome(oracle, mu=mu),
        "treatment_learner": LawTreatment(oracle, g=law.wrong_g if drift_mechanism else None),
        "missingness_learner": LawResponse(
            oracle, recorded=law.wrong_recorded if drift_mechanism else None
        ),
        "estimands": _estimands(law),
    }
    estimator = _engine(guard, reduction, **_settings(**{**learners, **overrides}))
    return estimator.fit(
        law.frame() if frame is None else frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="Delta",
        treatment_delta="DeltaA",
        weights="weight" if weighted else None,
    ).single()


def _parts(result: Any) -> Any:
    repeat = result.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    data, nuisance = targeting_inputs(result)
    return correction_parts(
        data, nuisance, fluctuation, fluctuation.targeted, nuisance.scaler.scale(data.outcome)
    )


def _max_error(result: Any, law: dl.CompositeLaw, *, weighted: bool = False) -> float:
    return max(
        abs(result.estimates[name].psi - law.truth(name, weighted=weighted)) for name in law.names()
    )


def _arm_means(result: Any, law: dl.CompositeLaw) -> np.ndarray:
    """The arm means in table-column order."""
    if law.k == 2:
        return np.array([result.estimates[f"ey{int(label)}"].psi for label in law.labels])
    return np.array([result.estimates[f"ey[{label}]"].psi for label in law.labels])


# ------------------------------------------------------------------ the law module


class TestTheSampleRealisesTheLaw:
    @pytest.mark.parametrize("key", LAWS)
    def test_the_empirical_conditionals_are_the_tables(self, key: str) -> None:
        law = dl.LAWS[key]
        frame = law.frame()
        recorded = frame["DeltaA"] == 1.0
        for w in range(3):
            rows = frame["W"] == w
            assert recorded[rows].mean() == pytest.approx(law.recorded[w], abs=1e-15)
            for column, label in enumerate(law.labels):
                arm = rows & recorded & (frame["A"] == label)
                assert arm.sum() / (rows & recorded).sum() == pytest.approx(
                    law.conditional_g[w, column], abs=1e-15
                )
                observed = arm & (frame["Delta"] == 1.0)
                assert observed.sum() / arm.sum() == pytest.approx(law.pi[w, column], abs=1e-15)
                assert frame.loc[observed, "Y"].mean() == pytest.approx(law.q[w, column], abs=1e-15)

    @pytest.mark.parametrize("key", LAWS)
    def test_the_gateaux_derivative_has_mean_zero(self, key: str) -> None:
        law = dl.LAWS[key]
        for name in law.names():
            mean = float((law.probs.reshape(-1) * law.eif(name)).sum())
            assert mean == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("key", LAWS)
    def test_the_treatment_recording_depends_on_the_treatment(self, key: str) -> None:
        """The recording depends on ``A`` in some ``W`` cell, so ``P(A | W)`` is unidentified."""
        law = dl.LAWS[key]
        assert float(np.max(np.ptp(law.pi_treatment, axis=1))) > 0.0
        assert not np.allclose(law.conditional_g, law.g)


# ---------------------------------------------------------------------------- E1


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("guard", (None, *GUARDS))
@pytest.mark.parametrize("weighted", (False, True))
def test_e1_the_oracle_fit_is_exact(key: str, guard: Any, weighted: bool) -> None:
    """E1: the estimate is the truth, the curve the EIF, and the corrections vanish."""
    law = dl.LAWS[key]
    result = _oracle_fit(key, guard=guard, weighted=weighted)
    assert result.extra["missing_data"] == "composite"
    assert _max_error(result, law, weighted=weighted) < 1e-12
    if not weighted:
        cells = law.cell_of_row()
        for name in law.names():
            reported = np.asarray(result.estimates[name].influence_curve)
            per_cell = np.array(
                [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.support))]
            )
            np.testing.assert_allclose(per_cell, law.eif(name), atol=1e-12, rtol=0)
    if guard:
        parts = _parts(result)
        for arm in result.data.arm_codes:
            assert float(np.max(np.abs(parts.d_g[arm]))) < 1e-12
            assert float(np.max(np.abs(parts.d_q[arm]))) < 1e-12


@pytest.mark.parametrize("key", LAWS)
def test_e1_the_bivariate_oracle_fit_is_exact(key: str) -> None:
    law = dl.LAWS[key]
    result = _oracle_fit(key, reduction="bivariate")
    assert _max_error(result, law) < 1e-12


# ---------------------------------------------------------------------------- E2


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("guard", (None, *GUARDS))
@pytest.mark.parametrize("weighted", (False, True))
def test_e2_an_outcome_drift_leaves_the_estimate_exact(
    key: str, guard: Any, weighted: bool
) -> None:
    r"""E2: ``Q -> 1 - Q`` with oracle factors.  ``D_g`` is live, and the estimate exact.

    With :math:`g_c` correct, the solved equation :math:`P_n[C_a / g_{c,a} (Y - \bar Q^*_a)]
    = 0` reduces on the exact frame to :math:`\sum_w P_W(w) (\bar Q_a - \bar Q^*_a)(w) = 0`,
    so the plug-in is the truth whatever the shape of :math:`\bar Q^*`.  The weight depends
    on ``W`` only, so the same holds for the weighted target.
    """
    law = dl.LAWS[key]
    result = _oracle_fit(key, guard=guard, mu=1.0 - law.q, weighted=weighted)
    assert _max_error(result, law, weighted=weighted) < 1e-10
    if not guard:
        return
    parts = _parts(result)
    for arm in result.data.arm_codes:
        assert float(np.max(np.abs(parts.d_g[arm]))) > 0.05
    check = result.diagnostics.corrections()
    assert check.passed
    assert check.identity_failures() == ()
    # One term per arm is selected at a partial guard: the row of the unguarded term is
    # reported but not solved.
    solved = {(row.equation, row.solved) for row in check.rows}
    assert ("D*_g", "Q" in guard) in solved
    assert ("D*_Q", "g" in guard) in solved


@pytest.mark.parametrize("key", LAWS)
def test_e2_the_bivariate_reduction_under_an_outcome_drift(key: str) -> None:
    law = dl.LAWS[key]
    result = _oracle_fit(key, reduction="bivariate", mu=1.0 - law.q)
    assert _max_error(result, law) < 1e-10
    assert result.diagnostics.corrections().passed


# ---------------------------------------------------------------------------- E3


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("guard", GUARDS[:3])
def test_e3_a_mechanism_drift_leaves_the_estimate_exact(key: str, guard: tuple[str, ...]) -> None:
    """E3: a wrong ``P(A | Delta_A = 1, W)`` and a wrong ``P(Delta_A = 1 | W)``.

    With ``Q`` correct every residual has mean zero in each ``(a, w)`` cell of the exact
    frame, so the estimate is exact, and ``D_Q`` is live.
    """
    law = dl.LAWS[key]
    result = _oracle_fit(key, guard=guard, drift_mechanism=True)
    assert _max_error(result, law) < 1e-12
    parts = _parts(result)
    for arm in result.data.arm_codes:
        assert float(np.max(np.abs(parts.d_q[arm]))) > 0.05
    assert result.diagnostics.corrections().identity_failures() == ()


# ---------------------------------------------------------------------------- E4


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("guard", (None, ("Q", "g")))
def test_e4_both_wrong_misses(key: str, guard: Any) -> None:
    """E4: both drifts.  The tilts move, and at least one arm mean misses the truth."""
    law = dl.LAWS[key]
    result = _oracle_fit(key, guard=guard, mu=1.0 - law.q, drift_mechanism=True)
    fluctuation = result.repeats[0].fluctuations["mean"]
    initial = result.nuisance.outcome
    for arm in result.data.arm_codes:
        moved = np.abs(fluctuation.targeted.arms[arm] - initial.arms[arm])
        assert float(np.max(moved)) > 1e-3
    if guard is None:
        expected = law.tmle_limit(law.mutations()["mechanism_drift"], 1.0 - law.q)
        np.testing.assert_allclose(_arm_means(result, law), expected, atol=1e-9)
    gap = float(np.max(np.abs(_arm_means(result, law) - dl.WITNESS_LIMITS[key]["truth"])))
    assert gap > dl.MARGIN


# ----------------------------------------------------------- E5-E10: the witnesses


def _patched_mechanism(monkeypatch: pytest.MonkeyPatch, transform: Any) -> None:
    """Replace the one derivation site of the composite mechanism with a mutated copy."""
    original = composite.composite_mechanism

    def mutated(nuisance: Any, bounds: Any, nuisance_bound: Any, **kwargs: Any) -> Propensity:
        return transform(nuisance, original(nuisance, bounds, nuisance_bound, **kwargs))

    monkeypatch.setattr(composite, "composite_mechanism", mutated)


def _omit(field: str) -> Any:
    def transform(nuisance: Any, _: Propensity) -> Propensity:
        return composite_mechanism(replace(nuisance, **{field: None}), G_BOUNDS, NUISANCE_BOUND)

    return transform


def _renormalized(_: Any, mechanism: Propensity) -> Propensity:
    values = np.asarray(mechanism.values)
    return Propensity(values / values.sum(axis=1, keepdims=True), mechanism.arms, simplex=False)


def _rolled(_: Any, mechanism: Propensity) -> Propensity:
    return Propensity(np.roll(mechanism.values, 1, axis=1), mechanism.arms, simplex=False)


def _complement(_: Any, mechanism: Propensity) -> Propensity:
    values = np.array(mechanism.values, copy=True)
    upper = mechanism.column_for(1.0)
    values[:, mechanism.column_for(0.0)] = 1.0 - values[:, upper]
    return Propensity(values, mechanism.arms)


#: Each witness: the law-module limit it lands on, and the mutation that reaches it.
WITNESSES = {
    "omit_recorded": _omit("treatment_observation"),  # E5
    "omit_response": _omit("missingness"),  # E6
    "renormalized": _renormalized,  # E7
}


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_e5_to_e7_a_mutated_mechanism_lands_on_its_limit(
    monkeypatch: pytest.MonkeyPatch, key: str, witness: str
) -> None:
    """E5-E7: under the outcome drift, each mutated composite moves the estimate.

    The correct composite is exact (E2), so the move is the mutation's alone, and it lands
    on the limit :mod:`tests.discrete_law_composite` computes and checks at import.
    """
    law = dl.LAWS[key]
    _patched_mechanism(monkeypatch, WITNESSES[witness])
    result = _oracle_fit(key, guard=None, mu=1.0 - law.q)
    np.testing.assert_allclose(_arm_means(result, law), dl.WITNESS_LIMITS[key][witness], atol=1e-9)


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_e5_to_e7_the_mutated_dr_tmle_misses(
    monkeypatch: pytest.MonkeyPatch, key: str, witness: str
) -> None:
    """The same mutations on the guarded fit, which is exact unmutated (E2).

    The outcome is wrong and so is the mechanism, so nothing holds the estimate at the
    truth.  The smallest move measured over these six cells was 1.1e-3, so the 5e-4 bar keeps
    a margin of about two.  This guarded witness is the weak one: no limit is computed for
    it.  The unguarded cells above land on import-computed limits with a margin of 0.02.
    """
    law = dl.LAWS[key]
    _patched_mechanism(monkeypatch, WITNESSES[witness])
    result = _oracle_fit(key, mu=1.0 - law.q)
    gap = float(np.max(np.abs(_arm_means(result, law) - dl.WITNESS_LIMITS[key]["truth"])))
    assert gap > 5e-4


@pytest.mark.parametrize("key", LAWS)
def test_e7_a_renormalized_fit_fails_the_identity_against_the_derivation_site(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """E7: the readers recompute the composite at one site, so a fit that divided by another
    mechanism fails an identity once the site is restored.

    The guard ``("g",)`` tilts no mechanism, so the curve reads the composite from the
    derivation site, and the bivariate ``D*_Q`` divides by it.  With both guards the curve
    reads the stored tilted mechanism instead, and the identity holds by construction.
    """
    law = dl.LAWS[key]
    with monkeypatch.context() as patch:
        _patched_mechanism(patch, _renormalized)
        result = _oracle_fit(key, guard=("g",), reduction="bivariate", mu=1.0 - law.q)
    failures = result.diagnostics.corrections().identity_failures()
    assert any(row.equation == "D*_Q" for row in failures)
    honest = _oracle_fit(key, guard=("g",), reduction="bivariate", mu=1.0 - law.q)
    assert honest.diagnostics.corrections().identity_failures() == ()


def test_e9_the_complement_form_on_the_composite_moves_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E9: at two arms the composite's columns do not sum to one.  The complement lands on
    its own limit, and on the guarded fit the forced complement route breaks the identity."""
    law = dl.TWO
    with monkeypatch.context() as patch:
        _patched_mechanism(patch, _complement)
        result = _oracle_fit("two", guard=None, mu=1.0 - law.q)
    np.testing.assert_allclose(
        _arm_means(result, law), dl.WITNESS_LIMITS["two"]["complement"], atol=1e-9
    )
    with monkeypatch.context() as patch:
        patch.setattr(targeting, "complement_form", lambda propensity: propensity.n_arms == 2)
        forced = _oracle_fit("two", mu=1.0 - law.q)
    assert forced.diagnostics.corrections().identity_failures() != ()
    honest = _oracle_fit("two", mu=1.0 - law.q)
    assert honest.repeats[0].fluctuations["mean"].mechanism.propensity.ndim == 2
    gaps = [abs(forced.estimates[name].psi - honest.estimates[name].psi) for name in law.names()]
    assert max(gaps) > 1e-6


def test_e10_rolled_arm_columns_land_on_their_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """E10: rolling the composite's columns by one at three arms moves at least two arms."""
    law = dl.THREE
    _patched_mechanism(monkeypatch, _rolled)
    result = _oracle_fit("three", guard=None, mu=1.0 - law.q)
    means = _arm_means(result, law)
    np.testing.assert_allclose(means, dl.WITNESS_LIMITS["three"]["rolled"], atol=1e-9)
    assert int(np.sum(np.abs(means - dl.WITNESS_LIMITS["three"]["truth"]) > 0.01)) >= 2


# ----------------------------------------------------------------- E8: factor order


class CellFrequency(ClassifierMixin, BaseEstimator):
    """Class frequencies within each level of the first design column, weighted.

    A saturated categorical fit on the exact frame: it reproduces the conditional law of
    the rows it is trained on, so those rows decide what it returns.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellFrequency:
        w = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        target = np.asarray(y, dtype=float)
        weights = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(target)
        table = np.array(
            [
                [np.sum(weights[(w == level) & (target == c)]) for c in self.classes_]
                for level in range(int(w.max()) + 1)
            ]
        )
        self.table_ = table / table.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        w = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        return np.asarray(self.table_[w], dtype=float)


def _all_rows_treatment_fit(original: Any) -> Any:
    """The factor-order mutation: the treatment model fitted on every row, with an
    unrecorded row read as the reference arm."""

    def mutated(learner: Any, design: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("classes") is not None and kwargs.get("fit_mask") is not None:
            target = np.where(np.isfinite(target), target, 0.0)
            kwargs = {**kwargs, "fit_mask": None}
        return original(learner, design, target, *args, **kwargs)

    return mutated


@pytest.mark.parametrize("key", LAWS)
def test_e8_the_treatment_factor_is_fitted_on_the_recorded_rows(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """E8: a fitted categorical treatment model is exact on the recorded rows.  Fitting it
    on every row, an unrecorded row read as the reference, lands on its limit."""
    law = dl.LAWS[key]
    exact = _oracle_fit(key, guard=None, mu=1.0 - law.q, treatment_learner=CellFrequency())
    assert _max_error(exact, law) < 1e-10
    monkeypatch.setattr(
        nuisance_module,
        "cross_fit_companion",
        _all_rows_treatment_fit(nuisance_module.cross_fit_companion),
    )
    mutated = _oracle_fit(key, guard=None, mu=1.0 - law.q, treatment_learner=CellFrequency())
    np.testing.assert_allclose(
        _arm_means(mutated, law), dl.WITNESS_LIMITS[key]["factor_order"], atol=1e-9
    )


# ---------------------------------------------------------- E11: fill invariance


def _filled_with(code: float) -> Any:
    original = CausalData.treatment_design

    def design(self: CausalData, **kwargs: Any) -> Any:
        if kwargs.get("missing_as") is not None:
            kwargs = {**kwargs, "missing_as": code}
        return original(self, **kwargs)

    return design


def _live_frame(law: dl.CompositeLaw, n: int = 1200, seed: int = 5) -> pd.DataFrame:
    """A finite sample of the law, with the columns the exact frame carries."""
    frame = sample(law.law(), n, seed)
    w = np.rint(frame["W"].to_numpy()).astype(int)
    return frame.assign(weight=law.weight[w])


def _fitted_learners() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=2000),
        "treatment_learner": LogisticRegression(max_iter=2000),
        "missingness_learner": LogisticRegression(max_iter=2000),
    }


def _live_fit(
    law: dl.CompositeLaw, frame: pd.DataFrame, guard: Any = ("Q", "g"), **roles: Any
) -> Any:
    estimator = _engine(guard, **_settings(**_fitted_learners(), estimands=_estimands(law)))
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="Delta",
        treatment_delta="DeltaA",
        **roles,
    ).single()


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("guard", (None, ("Q", "g")))
def test_e11_the_fill_value_of_an_unrecorded_row_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, key: str, guard: Any
) -> None:
    """E11: every arm in turn as the fill of the observed design, for both learners.

    The outcome and response learners are fitted, so an unrecorded row's prediction moves
    with its fill.  Every consumer multiplies it by a residual that is zero there, so every
    estimate, curve and score is bit for bit the same.
    """
    law = dl.LAWS[key]
    frame = _live_frame(law)
    fits = []
    for code in range(law.k):
        with monkeypatch.context() as patch:
            patch.setattr(CausalData, "treatment_design", _filled_with(float(code)))
            fits.append(_live_fit(law, frame, guard))
    observed = [np.asarray(fit.nuisance.outcome.observed) for fit in fits]
    unrecorded = frame["DeltaA"].to_numpy() == 0.0
    assert float(np.max(np.abs(observed[0][unrecorded] - observed[1][unrecorded]))) > 1e-6
    reference = fits[0]
    for fit in fits[1:]:
        for name, estimate in reference.estimates.items():
            assert fit.estimates[name].psi == estimate.psi
            np.testing.assert_array_equal(
                fit.estimates[name].influence_curve, estimate.influence_curve
            )
        np.testing.assert_array_equal(
            fit.repeats[0].fluctuations["mean"].score,
            reference.repeats[0].fluctuations["mean"].score,
        )


# ------------------------------------------------------- E12: the drtmle slip


def _qr_training_rows(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Record the training mask of every ``Q_r`` fit, the one reduction with a mask."""
    seen: list[np.ndarray] = []
    original = reduced_module._reduced_column

    def spy(learner: Any, **kwargs: Any) -> Any:
        if kwargs.get("fit_mask") is not None:
            seen.append(np.asarray(kwargs["fit_mask"], dtype=bool).copy())
        return original(learner, **kwargs)

    monkeypatch.setattr(reduced_module, "_reduced_column", spy)
    return seen


def _coded(frame: pd.DataFrame, law: dl.CompositeLaw) -> pd.DataFrame:
    """The frame with the first label written into every unrecorded treatment."""
    unrecorded = frame["DeltaA"] == 0.0
    coded = frame.copy()
    if law.k == 2:
        coded["A"] = np.where(unrecorded, law.labels[0], coded["A"]).astype(float)
    else:
        coded["A"] = np.where(unrecorded, law.labels[0], coded["A"]).astype(object)
    return coded


def _composite_indicator(data: CausalData, arm: float) -> np.ndarray:
    return data.treatment_recorded & data.observed & (data.treatment == arm)


def _same_fit(first: Any, second: Any) -> None:
    for name, estimate in first.estimates.items():
        assert second.estimates[name].psi == estimate.psi
        np.testing.assert_array_equal(
            second.estimates[name].influence_curve, estimate.influence_curve
        )


@pytest.mark.parametrize("key", LAWS)
def test_e12_a_coded_treatment_on_an_unrecorded_row_is_ignored(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """E12: R ``drtmle``'s ``glm_Qr`` branch filters ``Q_r``'s rows on ``DeltaY`` alone.

    Here a coded treatment on an unrecorded row changes nothing, ``Q_r`` trains on exactly
    the rows with ``C_a = 1``, and a view that masks by ``Delta`` alone -- the slip -- moves
    both the rows and the estimate.
    """
    law = dl.LAWS[key]
    frame = _live_frame(law)
    seen = _qr_training_rows(monkeypatch)
    blank = _live_fit(law, frame)
    blank_rows = list(seen)
    seen.clear()
    coded = _live_fit(law, _coded(frame, law))
    _same_fit(blank, coded)
    data = blank.data
    expected = [_composite_indicator(data, arm) for arm in data.arm_codes]
    assert len(blank_rows) >= law.k
    for index, rows in enumerate(blank_rows):
        np.testing.assert_array_equal(rows, expected[index % law.k])
    assert not any(np.any(rows & ~data.treatment_recorded) for rows in blank_rows)

    # The second layer: the codes kept on the unrecorded rows of the prepared data.  The
    # view masks by Delta_A itself, so the rows and the fit are unchanged.
    kept = replace(data, treatment=np.where(data.treatment_recorded, data.treatment, 0.0))
    seen.clear()
    estimator = _engine(("Q", "g"), **_settings(**_fitted_learners(), estimands=_estimands(law)))
    replaced = estimator.fit(kept).single()
    _same_fit(blank, replaced)
    for index, rows in enumerate(seen):
        np.testing.assert_array_equal(rows, expected[index % law.k])

    # The mutation: a view that masks by the outcome indicator only.
    def slip(data: CausalData) -> CausalData:
        observed = np.asarray(data.observed, dtype=bool)
        return replace(
            data,
            treatment=np.where(observed, data.treatment, np.nan),
            outcome=np.where(observed, data.outcome, 0.0),
            treatment_observed=None,
            composite_view=True,
        )

    monkeypatch.setattr(composite, "composite_view", slip)
    seen.clear()
    slipped = (
        _engine(("Q", "g"), **_settings(**_fitted_learners(), estimands=_estimands(law)))
        .fit(kept)
        .single()
    )
    assert any(np.any(rows & ~data.treatment_recorded) for rows in seen)
    gaps = [
        abs(slipped.estimates[name].psi - blank.estimates[name].psi) for name in blank.estimates
    ]
    assert max(gaps) > 1e-4


# --------------------------------------------------------- E13, E14: reductions


def _force_composite(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (tmle_module, drtmle_module):
        monkeypatch.setattr(module, "missing_data_route", lambda estimator, data: "composite")


def test_e13_the_composite_reduces_to_the_complete_data_fit_at_three_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E13: on a complete three-arm law the forced composite route is the shipped DR-TMLE.

    Both routes are armwise at three arms, and an absent factor is not multiplied, so every
    estimate and curve is bit for bit the same.
    """
    frame = multi.frame()
    covariates = [name for name in frame.columns if name.startswith("W")]

    def fit() -> Any:
        return (
            DRTMLE(
                **_reduced_learners(),
                **_settings(
                    outcome_learner=LogisticRegression(max_iter=2000),
                    treatment_learner=LogisticRegression(max_iter=2000),
                    estimands=("ey", "ate"),
                ),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )

    shipped = fit()
    assert shipped.extra["missing_data"] == "complete"
    with monkeypatch.context() as patch:
        _force_composite(patch)
        forced = fit()
    assert forced.extra["drtmle"].missing_data == "composite"
    _same_fit(shipped, forced)


def _missing_outcome_frame() -> tuple[pd.DataFrame, list[str]]:
    from cleverly.datasets import make_missing_outcome

    frame, _ = make_missing_outcome(n=600, seed=4)
    return frame, [name for name in frame.columns if name.startswith("W")]


def _observational(engine: type, **settings: Any) -> Any:
    frame, covariates = _missing_outcome_frame()
    return (
        engine(
            **_settings(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=2000),
                missingness_learner=LogisticRegression(max_iter=2000),
                estimands=("ey0", "ey1", "ate"),
                **settings,
            )
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates, delta="Delta")
        .single()
    )


def test_e14_the_initial_composite_covariate_is_the_missing_outcome_covariate() -> None:
    """E14: on an observational ``delta=`` fit, ``C_a / g_c`` is ``1{A=a} Delta / (g pi)``.

    Bit for bit, on every row whose outcome is observed and at every arm's counterfactual
    column: the composite multiplies the separately bounded factors in the shipped order.
    """
    result = _observational(TMLE)
    assert result.extra["missing_data"] == "missing_outcome"
    data, nuisance = result.data, result.nuisance
    shipped = build_submodel(data, nuisance, "mean", bounds=G_BOUNDS, nuisance_bound=NUISANCE_BOUND)
    state = composite_state(data, nuisance, g_bounds=G_BOUNDS, nuisance_bound=NUISANCE_BOUND)
    assert state.bounds == composite_bounds(G_BOUNDS, NUISANCE_BOUND, 1)
    built = build_submodel(
        state.data, state.nuisance, "mean", bounds=state.bounds, nuisance_bound=NUISANCE_BOUND
    )
    observed = np.asarray(data.observed, dtype=bool)
    np.testing.assert_array_equal(built.observed[observed], shipped.observed[observed])
    assert not np.any(built.observed[~observed])
    for arm in data.arm_codes:
        np.testing.assert_array_equal(built.arms[arm], shipped.arms[arm])


def test_e14_the_forced_composite_tmle_is_the_missing_outcome_tmle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole composite TMLE on a ``delta=``-only fit is the shipped missing-outcome TMLE."""
    shipped = _observational(TMLE)
    with monkeypatch.context() as patch:
        _force_composite(patch)
        forced = _observational(TMLE)
    assert forced.extra["missing_data"] == "composite"
    _same_fit(shipped, forced)


# ----------------------------------------------------------------- E15: bootstrap


def test_e15_a_bootstrap_replicate_carries_the_treatment_indicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    law = dl.TWO
    frame = _live_frame(law, 600, 3)
    data = _live_fit(law, frame, None).data
    index = np.array([5, 1, 1, 400, 7, 9, 12, 33, 250, 251, 500, 2])
    subset = data.subset(index)
    np.testing.assert_array_equal(subset.treatment_observed, data.treatment_observed[index])
    np.testing.assert_array_equal(np.isnan(subset.treatment), ~subset.treatment_recorded)

    replicates: list[CausalData] = []
    original = TMLE._bootstrap_point_estimates

    def spy(self: Any, replicate: CausalData, intermediate_value: Any) -> Any:
        replicates.append(replicate)
        return original(self, replicate, intermediate_value)

    monkeypatch.setattr(TMLE, "_bootstrap_point_estimates", spy)
    estimator = TMLE(n_bootstrap=3, **_settings(**_fitted_learners(), estimands=_estimands(law)))
    result = estimator.fit(
        frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta", treatment_delta="DeltaA"
    ).single()
    assert result.bootstrap is not None and len(replicates) == 3
    for replicate in replicates:
        assert replicate.has_missing_treatment
        np.testing.assert_array_equal(np.isnan(replicate.treatment), ~replicate.treatment_recorded)


# ----------------------------------------------------------- E16: integration


def _gaussian_frame(k: int, n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """A random law with a Gaussian ``W``, a logistic treatment and logistic recordings."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    logits = np.column_stack([np.zeros(n)] + [0.4 * (j + 1) * w for j in range(k - 1)])
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    arm = (rng.random(n)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    y = (rng.random(n) < expit(-0.3 + 0.5 * w + 0.4 * arm)).astype(float)
    recorded = rng.random(n) < expit(1.0 + 0.5 * w - 0.3 * arm)
    observed = rng.random(n) < expit(0.8 - 0.4 * w + 0.2 * arm)
    labels = np.array(["a", "b", "c"][:k], dtype=object) if k > 2 else np.array([0.0, 1.0])
    treatment = labels[arm]
    return pd.DataFrame(
        {
            "W": w,
            "A": np.where(recorded, treatment, np.nan if k == 2 else None),
            "Y": np.where(observed, y, np.nan),
            "Delta": observed.astype(float),
            "DeltaA": recorded.astype(float),
            "weight": 0.5 + rng.random(n),
            "cluster": rng.integers(0, 300, n),
        }
    )


@pytest.fixture(scope="module", params=[2, 3], ids=["two_arms", "three_arms"])
def gaussian_frame(request: Any) -> pd.DataFrame:
    return _gaussian_frame(request.param)


@pytest.mark.parametrize("guard", (None, ("Q", "g")))
@pytest.mark.parametrize("roles", ({}, {"weights": "weight"}, {"id": "cluster"}), ids=str)
def test_e16_integration(gaussian_frame: pd.DataFrame, guard: Any, roles: dict, tmp_path) -> None:
    """E16: every equation solved per arm, persistence, positivity rows and the truncation
    curve at the fitted bound, on a random law with fitted logistic nuisances."""
    k = 2 if gaussian_frame["A"].dropna().map(type).eq(float).all() else 3
    estimands = ("ey0", "ey1", "ate", "rr", "or") if k == 2 else ("ey", "ate", "rr", "or")
    estimator = _engine(guard, **_settings(**_fitted_learners(), estimands=estimands))
    result = estimator.fit(
        gaussian_frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="Delta",
        treatment_delta="DeltaA",
        **roles,
    ).single()
    assert result.extra["missing_data"] == "composite"
    assert result.diagnostics.score_equations().passed
    fluctuation = result.repeats[0].fluctuations["mean"]
    assert np.asarray(fluctuation.score).shape == (k,)
    if guard:
        assert fluctuation.mechanism.propensity.shape == (result.data.n, k)
        assert result.diagnostics.corrections().passed
    path = tmp_path / "composite.cleverly"
    result.save(path)
    restored = load(path)
    for name, estimate in result.estimates.items():
        assert restored.estimates[name].psi == estimate.psi
    np.testing.assert_array_equal(restored.data.treatment_observed, result.data.treatment_observed)
    report = result.diagnostics.support()
    assert "P(A=a,Delta_A=1,Delta=1|W)" in report.mechanisms
    assert "P(Delta_A=1|W)" in report.mechanisms
    curve = result.diagnostics.truncation_curve(bounds=[result.config.g_bounds[0]])
    np.testing.assert_allclose(curve["delta_from_fitted"], 0.0, atol=1e-10)
    assert set(curve["truncated_factor"]) == {"P(A=a|W, Delta_A=1)"}


# ------------------------------------------------- E18: the per-arm reference


def _tilt(
    response: np.ndarray, offset: np.ndarray, covariate: np.ndarray, rows: np.ndarray
) -> float:
    """A one-parameter logistic MLE with an offset and no intercept, by Newton steps."""
    epsilon = 0.0
    for _ in range(200):
        fitted = expit(offset + epsilon * covariate)
        score = float(np.sum(rows * covariate * (response - fitted)))
        information = float(np.sum(rows * covariate**2 * fitted * (1.0 - fitted)))
        if information == 0.0:
            return 0.0
        step = score / information
        epsilon += step
        if abs(step) < 1e-15:
            break
    return epsilon


def _one_arm(
    frame: pd.DataFrame,
    law: dl.CompositeLaw,
    column: int,
    *,
    mu: np.ndarray,
    g_cond: np.ndarray,
    recorded: np.ndarray,
    guard: tuple[str, ...],
    rounds: int = 200,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Benkeser et al. (2017), Section 3.2, for one binary treatment ``C_a``, from the paper.

    The data are ``(W, C_a, C_a Y)``, ``C_a = Delta_A Delta 1{A = a}``.  The mechanism is
    ``P(C_a = 1 | W)``, the product of the working factors; the regression is
    ``E(Y | C_a = 1, W)``.  Each round fluctuates ``g`` along ``Q_r / g`` (equation 9, the
    ``"Q"`` guard), refits ``g_r1`` and ``g_r2`` and fluctuates ``Q`` along ``g_r2 / g_r1``
    among ``C_a = 1`` (equation 10, the ``"g"`` guard), fluctuates ``Q`` along ``1 / g`` among
    ``C_a = 1`` (equation 8), and refits ``Q_r``.  It shares no code with the package.
    """
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    label = law.labels[column]
    c = (
        (frame["DeltaA"].to_numpy() == 1.0)
        & (frame["Delta"].to_numpy() == 1.0)
        & (frame["A"].to_numpy() == label)
    ).astype(float)
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    q = mu[w, column].copy()
    g = np.clip(g_cond[w, column], *G_BOUNDS) * np.clip(law.pi[w, column], NUISANCE_BOUND, 1.0)
    g = g * np.clip(recorded[w], NUISANCE_BOUND, 1.0)
    lower, upper = composite_bounds(G_BOUNDS, NUISANCE_BOUND, 2)
    at = c == 1.0
    classifier = LogisticRegression(C=1e6, max_iter=2000)

    def bounded(values: np.ndarray) -> np.ndarray:
        return np.clip(values, lower, upper)

    def qr(q: np.ndarray, g: np.ndarray) -> np.ndarray:
        x = g.reshape(-1, 1)
        return LinearRegression().fit(x[at], (y - q)[at]).predict(x)

    def gr(q: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = q.reshape(-1, 1)
        gr1 = np.clip(classifier.fit(x, c).predict_proba(x)[:, 1], lower, upper)
        target = (c - bounded(g)) / bounded(g)
        gr2 = LinearRegression().fit(x, target).predict(x)
        return gr1, gr2

    # The reductions are first fitted at the initial pair, before equation (8) is solved.
    current_qr = qr(q, g)
    q = expit(logit(q) + _tilt(y, logit(q), 1.0 / bounded(g), c) / bounded(g))
    moved = {"g": 0.0, "Q_r": 0.0, "Q": 0.0}
    for _ in range(rounds):
        eps_g = eps_r = 0.0
        if "Q" in guard:
            h_g = current_qr / bounded(g)
            eps_g = _tilt(c, logit(g), h_g, np.ones_like(c))
            g = bounded(expit(logit(g) + eps_g * h_g))
        if "g" in guard:
            gr1, gr2 = gr(q, g)
            h_r = gr2 / gr1
            eps_r = _tilt(y, logit(q), h_r, c)
            q = expit(logit(q) + eps_r * h_r)
        h_q = 1.0 / bounded(g)
        eps_q = _tilt(y, logit(q), h_q, c)
        q = expit(logit(q) + eps_q * h_q)
        current_qr = qr(q, g)
        moved["g"] += abs(eps_g)
        moved["Q_r"] += abs(eps_r)
        moved["Q"] += abs(eps_q)
        if max(abs(eps_g), abs(eps_r), abs(eps_q)) < 1e-14:
            break
    psi = float(np.mean(q))
    gb = bounded(g)
    gr1, gr2 = gr(q, g)
    d_star = c / gb * (y - q) + q - psi
    d_g = current_qr / gb * (c - gb) if "Q" in guard else 0.0
    d_q = c * gr2 / gr1 * (y - q) if "g" in guard else 0.0
    return psi, d_star - d_g - d_q, {name: moved[name] for name in _TILTS[guard]}


#: The two live drifts of E18, each with the guard whose correction it makes live.  Under the
#: outcome drift the mechanism is right, so ``g_r2`` vanishes and equation (10) has a
#: near-zero covariate: its fixed point is not numerically identified, in the package or in
#: the reference (measured: both run to 200 rounds and stop 2e-3 apart on one arm).  The
#: mechanism drift is the mirror case for equation (9).
_REFERENCE_DRIFTS = {"outcome": ("Q",), "mechanism": ("g",)}
#: The tilts each guard moves, which the precondition asserts are live.
_TILTS = {("Q",): ("g", "Q"), ("g",): ("Q_r", "Q")}


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("drift", sorted(_REFERENCE_DRIFTS))
def test_e18_each_arm_is_the_one_indicator_estimator(key: str, drift: str) -> None:
    """E18: the stacked fit is Benkeser et al.'s estimator on ``(W, C_a, C_a Y)``, per arm.

    Under live drift only, with every tilt of the reference moving: at the oracle every
    correction vanishes and the comparison would be blind.  The package stops at its
    numerical bar and the reference iterates to 1e-14, so they agree to the stopping error.
    Measured on this sample (seed 11, n = 1200) over the four cells: the largest estimate
    gap was 8.8e-7 and the largest curve gap 3.3e-4, on curve values of order ten.  The
    tolerances are 1e-5 and 3e-3, against witness moves of at least 0.02 (E5-E10).

    The primary composition, both guards together, has no per-arm reference: under both
    drifts the alternation has more than one stable fixed point (the three-arm ``low`` arm
    settles 0.011 apart in the package and the reference), and some arms cycle.
    """
    law = dl.LAWS[key]
    frame = _live_frame(law, 1200, 11)
    mu = 1.0 - law.q if drift == "outcome" else law.q
    mechanism = drift == "mechanism"
    guard = _REFERENCE_DRIFTS[drift]
    result = _oracle_fit(
        key,
        frame=frame,
        guard=guard,
        mu=mu,
        drift_mechanism=mechanism,
        estimands=("ey0", "ey1") if law.k == 2 else ("ey",),
        max_outer=200,
    )
    g_cond = law.wrong_g if mechanism else law.conditional_g
    recorded = law.wrong_recorded if mechanism else law.recorded
    for column, label in enumerate(law.labels):
        psi, curve, moved = _one_arm(
            frame, law, column, mu=mu, g_cond=g_cond, recorded=recorded, guard=guard
        )
        assert min(moved.values()) > 1e-4, moved
        name = f"ey{int(label)}" if law.k == 2 else f"ey[{label}]"
        estimate = result.estimates[name]
        assert estimate.psi == pytest.approx(psi, abs=1e-5)
        np.testing.assert_allclose(estimate.influence_curve, curve, atol=3e-3, rtol=0)


def test_e18_a_sign_mutation_in_the_curve_is_caught_by_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negating arm code 0's ``D_g`` in the reported curve; E18's reference sees it.

    The correction check cannot: a solved block has mean zero, and so has its negation.
    """
    original = tmle_module.reduced_correction_parts

    def negated(*args: Any, **kwargs: Any) -> Any:
        parts = original(*args, **kwargs)
        flipped = {arm: (-values if arm == 0.0 else values) for arm, values in parts.d_g.items()}
        return replace(parts, d_g=flipped)

    monkeypatch.setattr(tmle_module, "reduced_correction_parts", negated)
    law = dl.THREE
    frame = _live_frame(law, 1200, 11)
    mu = 1.0 - law.q
    result = _oracle_fit(
        "three", frame=frame, guard=("Q",), mu=mu, estimands=("ey",), max_outer=200
    )
    gaps = {}
    for column, label in enumerate(law.labels):
        _, curve, _ = _one_arm(
            frame,
            law,
            column,
            mu=mu,
            g_cond=law.conditional_g,
            recorded=law.recorded,
            guard=("Q",),
        )
        reported = np.asarray(result.estimates[f"ey[{label}]"].influence_curve)
        gaps[label] = float(np.max(np.abs(reported - curve)))
    mutated = law.codes[0]
    assert gaps[mutated] > 0.1, gaps
    assert all(gap < 3e-3 for label, gap in gaps.items() if label != mutated), gaps


# --------------------------------------------------- E19: the reader mutation


@pytest.mark.parametrize("key", LAWS)
def test_e19_a_reader_of_the_raw_treatment_fails_the_identity(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """E19: the corrections read the composite indicator through ``targeting_inputs``.

    Fed the raw treatment and outcome indicator instead, ``1{A = a}`` counts the rows with a
    recorded treatment and no outcome, and the identity fails there.
    """
    law = dl.LAWS[key]
    result = _live_fit(law, _live_frame(law))
    assert result.diagnostics.corrections().identity_failures() == ()

    def raw(result: Any, nuisance: Any = None) -> Any:
        _, composite_nuisance = targeting_inputs(result, nuisance)
        return result.data, composite_nuisance

    monkeypatch.setattr(validation_drtmle, "targeting_inputs", raw)
    data = result.data
    assert np.any(data.treatment_recorded & ~data.observed)
    check = validation_drtmle.correction_check(result, tolerance=1e-6)
    assert check.identity_failures() != ()


# ------------------------------------------------------------- E20: clusters


@pytest.mark.parametrize("guard", (None, ("Q", "g")))
def test_e20_clusters_move_the_variance_and_not_the_estimate(guard: Any) -> None:
    law = dl.THREE
    frame = _live_frame(law, 900, 2).assign(cluster=lambda f: np.arange(len(f)) // 3)
    plain = _live_fit(law, frame, guard)
    clustered = _live_fit(law, frame, guard, id="cluster")
    codes = np.asarray(clustered.data.cluster)
    for name, estimate in plain.estimates.items():
        assert clustered.estimates[name].psi == estimate.psi
        curve = np.asarray(clustered.estimates[name].influence_curve)
        if estimate.log_psi is not None:
            continue
        sums = np.bincount(codes, weights=curve)
        n, m = len(curve), len(sums)
        expected = np.sqrt(m / (m - 1) * np.sum(sums**2)) / n
        assert clustered.estimates[name].std_error == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------ E21: view guard


def test_e21_a_nuisance_fit_refuses_the_composite_view() -> None:
    law = dl.TWO
    fitted = _live_fit(law, _live_frame(law, 300, 1), None)
    data = fitted.data
    view = composite_view(data)
    assert view.composite_view and not data.composite_view
    with pytest.raises(ValueError, match="composite-indicator view"):
        nuisance_module.fit_nuisances(
            view,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            intermediate_learner=None,
            folds=fitted.nuisance.folds,
            scaler=fitted.nuisance.scaler,
        )


# ------------------------------------------------------- 3.2: the route stamps

#: ``float.hex`` of each estimate and standard error, taken before the composite route existed, on
#: ``make_missing_outcome(n=400, seed=4)`` with the learners of :func:`_pinned`.  The
#: shipped missing-outcome TMLE, ``DRTMLE(guard=())`` with an observational ``delta=``,
#: and ``DRTMLE(randomized=True)`` must not move.
PINNED = {
    "missing_outcome": {
        "ate": ("0x1.1c1481f05746cp+0", "0x1.b500ad823e9e2p-4"),
        "ey1": ("0x1.24249d3373fccp+1", "0x1.7ad47c4091282p-4"),
        "ey0": ("0x1.2c34b87690b2cp+0", "0x1.81e703744dd2dp-4"),
    },
    "randomized_missing_outcome": {
        "ate": ("0x1.1b733f0769a50p+0", "0x1.b45a0a99daa01p-4"),
        "ey1": ("0x1.241ea3fe56ab4p+1", "0x1.79a67cc02f91cp-4"),
        "ey0": ("0x1.2cca08f543b18p+0", "0x1.82d00b80554e7p-4"),
    },
}


def _pinned(engine: type, **settings: Any) -> Any:
    from cleverly.datasets import make_missing_outcome

    frame, _ = make_missing_outcome(n=400, seed=4)
    covariates = [name for name in frame.columns if name.startswith("W")]
    return (
        engine(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
            estimands=["ey0", "ey1", "ate"],
            **settings,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates, delta="Delta")
        .single()
    )


@pytest.mark.parametrize(
    ("label", "build", "route"),
    [
        ("tmle", lambda: _pinned(TMLE), "missing_outcome"),
        ("unguarded", lambda: _pinned(DRTMLE, guard=()), "missing_outcome"),
        ("randomized", lambda: _pinned(DRTMLE, randomized=True), "randomized_missing_outcome"),
    ],
)
def test_the_shipped_missing_outcome_routes_did_not_move(
    label: str, build: Any, route: str
) -> None:
    result = build()
    assert result.extra["missing_data"] == route, label
    for name, (psi, std_error) in PINNED[route].items():
        assert float(result.estimates[name].psi).hex() == psi, (label, name)
        assert float(result.estimates[name].std_error).hex() == std_error, (label, name)


def test_the_observational_guarded_fit_is_composite_and_moves() -> None:
    """The control of the pins: the guarded observational fit is the new route."""
    result = _pinned(DRTMLE)
    assert result.extra["missing_data"] == "composite"
    assert float(result.estimates["ate"].psi).hex() != PINNED["missing_outcome"]["ate"][0]


def test_a_complete_fit_is_the_complete_route() -> None:
    frame = multi.frame()
    covariates = [name for name in frame.columns if name.startswith("W")]
    result = (
        DRTMLE(
            **_reduced_learners(),
            **_settings(
                outcome_learner=LogisticRegression(max_iter=2000),
                treatment_learner=LogisticRegression(max_iter=2000),
                estimands=("ey",),
            ),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )
    assert result.extra["missing_data"] == "complete"
    assert result.extra["drtmle"].missing_data == "complete"


def test_the_randomized_route_at_three_arms() -> None:
    from tests import discrete_law_mar_multi as mar_multi

    frame = mar_multi.frame()
    result = (
        DRTMLE(
            randomized=True,
            **_reduced_learners(),
            **_settings(
                outcome_learner=LawOutcome(mar_multi.LAW),
                treatment_learner=LawTreatment(mar_multi.LAW),
                missingness_learner=LawResponse(mar_multi.LAW),
                estimands=("ey",),
            ),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )
    assert result.extra["missing_data"] == "randomized_missing_outcome"
    assert result.extra["drtmle"].reduction == "missing_outcome"


def test_no_construction_is_chosen_by_the_outcome_flag() -> None:
    """Every DR-TMLE construction is chosen by the recorded route, not by
    ``has_missing_outcome``: an observational ``delta=`` fit would otherwise take the
    randomized construction.  The flag may still record a bound or gate a refusal."""
    import inspect

    for method in (DRTMLE._fit_reduced, DRTMLE._reduction):
        assert "has_missing_outcome" not in inspect.getsource(method), method.__name__
    lines = [
        line.strip()
        for line in inspect.getsource(DRTMLE._nuisances).splitlines()
        if "has_missing_outcome" in line
    ]
    assert lines == ["if data.has_missing_outcome or data.has_missing_treatment"]


# ----------------------------------------------------------- the study route


def _study_frame() -> pd.DataFrame:
    frame = _live_frame(dl.THREE, 900, 4)
    return frame.rename(columns={"Delta": "R", "DeltaA": "RA"})


def test_the_study_route_declares_a_missing_treatment() -> None:
    from cleverly import ATE, CausalStudy, DRTMLEMethod, PointTreatment

    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W",),
        missingness="R",
        treatment_missingness="RA",
    )
    effect = CausalStudy(_study_frame(), design=design).identify(ATE())
    assert "RA=1" in effect.functional.expression
    assumptions = " ".join(effect.identification.assumptions)
    assert "treatment missing at random for RA" in assumptions
    assert "composite positivity" in assumptions
    fitted = effect.estimate(
        method=DRTMLEMethod(
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
        ),
        **_fitted_learners(),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    assert fitted.extra["missing_data"] == "composite"
    assert fitted.diagnostics.score_equations().passed


def test_the_study_route_refuses_an_unidentified_target() -> None:
    from cleverly import ATT, CausalStudy, PointTreatment
    from cleverly.exceptions import CapabilityError

    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=("W",), missingness="R", treatment_missingness="RA"
    )
    study = CausalStudy(_study_frame().assign(A=lambda f: f["A"]), design=design)
    with pytest.raises(CapabilityError, match=r"P\(A = a \| W\) is not identified"):
        study.identify(ATT())


# ---------------------------------------------------- the inference status


@pytest.mark.parametrize(
    "kind",
    [
        "drtmle+composite",
        "missing_treatment",
        "tmle+missing_treatment+weights",
        "drtmle+missing_treatment+multi_arm",
        "drtmle+missing_treatment+id",
    ],
)
def test_each_composite_kind_supplies_an_influence_curve_interval(kind: str) -> None:
    from tests.unit._capability_sweep_support import KINDS

    result = KINDS[kind].build()
    assert result.extra["missing_data"] == "composite"
    assert result.inference_status == "influence_curve"
    for estimate in result.estimates.values():
        assert np.isfinite(estimate.std_error)


@pytest.mark.parametrize(
    ("guard", "status"), [(("Q", "g"), "estimated_weight_plugin"), ((), "influence_curve")]
)
def test_estimated_weights_on_a_guarded_composite_fit_withhold_the_interval(
    guard: tuple[str, ...], status: str
) -> None:
    """As on the complete-data DR-TMLE: the reductions of an estimated weight are not derived."""
    law = dl.TWO
    frame = _live_frame(law, 600, 8)
    data = CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="Delta",
        treatment_delta="DeltaA",
        weights="weight",
        weights_estimated=True,
    )
    result = _engine(guard, **_settings(**_fitted_learners(), estimands=("ate",))).fit(data)
    assert result.single().inference_status == status


@pytest.mark.parametrize("key", LAWS)
def test_the_positivity_report_reads_the_composite_covariate(key: str) -> None:
    """The report rebuilds the clever covariate a fit was fluctuated along: ``C_a / g_c``.

    Rebuilt from the raw data instead, it would divide by ``g pi`` alone and miss the
    treatment observation factor, so its largest value would be smaller.
    """
    from cleverly.sensitivity.positivity import _max_abs_covariate

    law = dl.LAWS[key]
    result = _oracle_fit(key, guard=None)
    expected = float(np.max(1.0 / law.composite))
    assert _max_abs_covariate(result, "mean") == pytest.approx(expected, rel=1e-12)
    raw = build_submodel(
        result.data, result.nuisance, "mean", bounds=G_BOUNDS, nuisance_bound=NUISANCE_BOUND
    )
    assert raw.max_abs < expected - 1.0


# ------------------------------------------------- regimes and MSMs over arms


def _rule(law: dl.CompositeLaw) -> tuple[Any, np.ndarray]:
    """A known ``W``-rule, and its table column at each ``W`` level."""
    from cleverly.interventions import Rule

    high, low = law.codes[1], law.codes[0]
    rule = Rule(lambda w: np.where(w["W"] >= 1, high, low), name="rule", rule_kind="known")
    columns = np.array([law.labels.index(high if w >= 1 else low) for w in range(3)])
    return rule, columns


@pytest.mark.parametrize("key", LAWS)
@pytest.mark.parametrize("drift", ("oracle", "outcome", "mechanism"))
def test_a_regime_mean_is_the_composite_regime_functional(key: str, drift: str) -> None:
    """A static and a ``W``-dependent regime with a missing treatment, on the exact law.

    The regime mean ``E[Qbar(d(W), W)]`` reads only the regression and the law of ``W``, so it
    is identified under the composite conditions.  Its covariate is ``C_d / g_c``, the arm
    covariate at the arm the rule assigns.  The estimate is exact at the oracle and under
    either drift, and the static regime's curve is the arm mean's EIF.
    """
    from cleverly.interventions import Static

    law = dl.LAWS[key]
    rule, columns = _rule(law)
    reference = law.codes[0]
    result = _oracle_fit(
        key,
        guard=None,
        mu=1.0 - law.q if drift == "outcome" else None,
        drift_mechanism=drift == "mechanism",
        interventions=[Static(reference, name="ref"), rule],
        estimands=("ey_regime", "ate_regime"),
    )
    assert result.extra["missing_data"] == "composite"
    truth_rule = float(dl.P_W @ law.q[np.arange(3), columns])
    truth_ref = float(dl.P_W @ law.q[:, law.labels.index(reference)])
    assert result.estimates["ey_regime[rule]"].psi == pytest.approx(truth_rule, abs=1e-10)
    assert result.estimates["ey_regime[ref]"].psi == pytest.approx(truth_ref, abs=1e-10)
    if drift == "oracle":
        name = "ey0" if law.k == 2 else f"ey[{reference}]"
        cells = law.cell_of_row()
        reported = np.asarray(result.estimates["ey_regime[ref]"].influence_curve)
        per_cell = np.array(
            [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.support))]
        )
        np.testing.assert_allclose(per_cell, law.eif(name), atol=1e-12, rtol=0)


def test_a_regime_fit_without_the_composite_misses() -> None:
    """The control: the same regime fit with the treatment observation factor omitted."""
    import cleverly.estimators.composite as composite_module

    law = dl.TWO
    rule, columns = _rule(law)
    truth = float(dl.P_W @ law.q[np.arange(3), columns])
    with pytest.MonkeyPatch.context() as patch:
        _patched_mechanism(patch, _omit("treatment_observation"))
        assert composite_module.composite_mechanism is not composite_mechanism
        result = _oracle_fit(
            "two", guard=None, mu=1.0 - law.q, interventions=[rule], estimands=("ey_regime",)
        )
    assert abs(result.estimates["ey_regime[rule]"].psi - truth) > 0.01


@pytest.mark.parametrize("drift", ("oracle", "outcome", "mechanism"))
def test_an_arm_msm_is_the_projection_of_the_arm_means(drift: str) -> None:
    """``MSM.linear()`` over two arms is ``(ey0, ey1 - ey0)``, exactly, with a missing treatment."""
    from cleverly.msm import MSM

    law = dl.TWO
    result = _oracle_fit(
        "two",
        guard=None,
        mu=1.0 - law.q if drift == "outcome" else None,
        drift_mechanism=drift == "mechanism",
        msm=MSM.linear(),
        estimands=("msm",),
    )
    means = dl.P_W @ law.q
    assert result.estimates["msm[(intercept)]"].psi == pytest.approx(means[0], abs=1e-10)
    assert result.estimates["msm[a]"].psi == pytest.approx(means[1] - means[0], abs=1e-10)


# ------------------------------------------ the admitted compositions, one test each


@pytest.mark.parametrize("key", LAWS)
def test_strata_on_the_composite_tmle_are_exact(key: str) -> None:
    """``strata=`` on the composite TMLE: each stratum's arm mean is its stratum truth."""
    law = dl.LAWS[key]
    oracle = law.law()
    result = (
        TMLE(
            **_settings(
                outcome_learner=LawOutcome(oracle),
                treatment_learner=LawTreatment(oracle),
                missingness_learner=LawResponse(oracle),
                estimands=("ey0", "ey1") if law.k == 2 else ("ey",),
            )
        )
        .fit(
            law.frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_delta="DeltaA",
            strata=["W"],
        )
        .single()
    )
    stratum = {name: estimate.psi for name, estimate in result.estimates.items() if "W=" in name}
    assert len(stratum) == 3 * law.k
    expected = sorted(float(value) for value in law.q.reshape(-1))
    np.testing.assert_allclose(sorted(stratum.values()), expected, atol=1e-12)


def test_screening_reads_the_recorded_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``screen_treatment=True`` screens the treatment on the rows whose treatment is recorded."""
    import cleverly.learners.screeners as screeners

    seen: list[int] = []
    original = screeners.screen_by_correlation

    def spy(design: Any, target: Any, **kwargs: Any) -> Any:
        values = np.asarray(target, dtype=float)
        assert np.all(np.isfinite(values))
        seen.append(values.shape[0])
        return original(design, target, **kwargs)

    monkeypatch.setattr(screeners, "screen_by_correlation", spy)
    law = dl.TWO
    frame = _live_frame(law, 600, 9).assign(V=lambda f: f["W"] ** 2)
    estimator = TMLE(screen_treatment=True, **_settings(**_fitted_learners(), estimands=("ate",)))
    estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W", "V"],
        delta="Delta",
        treatment_delta="DeltaA",
    )
    assert seen
    assert set(seen) == {int((frame["DeltaA"] == 1.0).sum())}


def test_the_array_entry_point_takes_delta_a() -> None:
    """``tmle(Y, A, W, Delta=, DeltaA=)`` is the frame fit with ``treatment_delta=``."""
    from cleverly.estimators.tmle import tmle

    law = dl.TWO
    frame = _live_frame(law, 600, 4)
    settings = _settings(**_fitted_learners(), estimands=("ate",))
    by_arrays = tmle(
        frame["Y"].to_numpy(),
        frame["A"].to_numpy(),
        frame[["W"]].to_numpy(),
        Delta=frame["Delta"].to_numpy(),
        DeltaA=frame["DeltaA"].to_numpy(),
        **settings,
    ).single()
    by_frame = _live_fit(law, frame, None)
    assert by_arrays.extra["missing_data"] == "composite"
    assert by_arrays.estimates["ate"].psi == pytest.approx(by_frame.estimates["ate"].psi, abs=1e-12)


def test_dr_tmle_default_targets_drop_the_unidentified_ones() -> None:
    """The default list on a missing treatment drops ATT and ATC, as TMLE's does."""
    result = _oracle_fit("two", estimands=None)
    assert {"att", "atc"}.isdisjoint(result.estimates)
    assert {"ey0", "ey1", "ate"} <= set(result.estimates)


def test_a_small_treatment_observation_probability_warns() -> None:
    """``P(Delta_A = 1 | W)`` below ``nuisance_bound`` warns, as the response mechanism does."""
    from cleverly.exceptions import PositivityWarning

    law = dl.TWO
    oracle = law.law()
    estimator = TMLE(
        **_settings(
            outcome_learner=LawOutcome(oracle),
            treatment_learner=LawTreatment(oracle),
            missingness_learner=LawResponse(oracle, recorded=np.array([0.005, 0.5, 0.5])),
            estimands=("ate",),
        )
    )
    with pytest.warns(PositivityWarning, match=r"P\(Delta_A = 1 \| W\)"):
        estimator.fit(
            law.frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_delta="DeltaA",
        )
