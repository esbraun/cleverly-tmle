"""A cross-fitted longitudinal working model: untargeted folds, one stacked update per node.

Each outer fold runs the untargeted recursion of every regimen-horizon cell on its training
rows, which is Step 1 of the cross-fitted construction of Díaz, Williams, Hoffman and
Schenck (2023, *JASA* 118(542), Section 5.2).  The held-out predictions are stitched into one
out-of-fold initial fit per node and cell.  One stacked fluctuation per node then targets
every follower of every live cell, with the out-of-fold mechanism in the loss weight, and the
projection of the stitched targeted fit gives the coefficients.  No fold solves a fluctuation
and no fold computes a ``beta``.

Every fit here has a misspecified or deliberately rough outcome regression, so each node's
``epsilon`` is away from zero.  An exact-law check is blind to a term that vanishes at the
truth, so T5 states that premise and the mutations below fail each claim that reads it.

=====  ======================================================================================
claim  what it checks
=====  ======================================================================================
T1     a saturated model reproduces the cross-fitted per-regimen report: survival, competing
       risks, a dynamic plan and a three-level node (end of study in ``test_ltmle_msm.py``)
T2     each cell's stitched initial fit is the per-regimen fold recursion, and the longhand
T3     the coefficient curve is the stacked delta-method curve, identity and logit links
T4     each node's stacked score is solved over every follower, and ``P_n D* = 0``
T5     every node moves in every term, and the cross-fitted ``beta`` is not the in-sample one
T6     one ``solver`` score row per node and term, and no stitching row or ``z`` column
T7     log and logit links: the last design, the round count, and no learner refit
T8     a held-out outcome reaches no initial fit of its own fold
T9     observation weights: the saturated reduction and the weighted score
T10    survival and competing risks: the curve and the score on an unsaturated grid
T11    clusters: whole-cluster folds, the cluster-summed SE, bands and the t reference
T12    bands, the bootstrap kind, the truncation replay and the ratio view
T13    the refusals that remain, and the deleted refusal nowhere in the tree
T14    agreement with the in-sample fit on the discrete law
M1-M8  mutations, each failing the claim named in its docstring
=====  ======================================================================================
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from scipy.special import expit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import cleverly.assessment
from cleverly._inference_status import NO_T_REFERENCE_BANDS
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
    make_longitudinal_weighted,
)
from cleverly.exceptions import CapabilityError, DataError, LongitudinalError, PositivityWarning
from cleverly.fluctuation.iterative import InitialFit
from cleverly.learners.crossfit import Folds, random_partition
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalResult, sequential
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal import msm as longitudinal_msm
from cleverly.longitudinal.msm import RegimenMSM
from cleverly.longitudinal.sequential import Mechanism
from cleverly.msm import MSM
from tests import discrete_law_longitudinal as discrete_law
from tests.studies import canonical_ltmle_crossfit as subject_study
from tests.studies import clustered_longitudinal_laws as cluster_law
from tests.studies import longitudinal_msm_properties as discrete_properties
from tests.studies.canonical_longitudinal_msm import declared_msm
from tests.unit.test_pooled_longitudinal_targeting import (
    _assert_clever_is_the_out_of_fold_reciprocal,
    _assert_stitched_longhand,
    refit_fold_mechanism,
)
from tests.unit.test_sequential_design import COLUMNS as CATEGORICAL_COLUMNS
from tests.unit.test_sequential_design import multivalue_panel

ROOT = Path(__file__).resolve().parents[2]

SHARED: dict[str, Any] = {
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}
SPEC: dict[str, Any] = {"always": 1, "never": 0, "early": (1, 0), "late": (0, 1)}
DURATION: dict[str, float] = {"always": 2.0, "never": 0.0, "early": 1.0, "late": 1.0}
#: A nonuniform projection measure, so a dropped ``h`` cannot hide behind a constant.
PROJECTION: dict[str, float] = {"always": 3.0, "never": 1.0, "early": 0.5, "late": 2.0}
TERMS = ("(intercept)", "duration")
K = 5
TOL = 1e-10
#: The reduction's solver tolerance.  One stacked solve can stop one Newton step away from
#: separate ones, which leaves a gap of the order of ``tol``: 1.1e-10 at ``tol=1e-10``.  At
#: ``1e-11`` every reduction fit here converges, and the largest gaps measured 4.7e-12 on a
#: coefficient and 4.7e-11 on a curve.  ``1e-12`` is below the attainable floor.
REDUCTION_TOL = 1e-11
#: The reduction pins, as multiples of the solver tolerance.
COEFFICIENT_GAP = 10 * REDUCTION_TOL
CURVE_GAP = 50 * REDUCTION_TOL


def settings(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "n_folds": K,
        "learner_folds": 2,
        "random_state": 0,
        "simultaneous": False,
        "tol": TOL,
        **overrides,
    }


def dose_msm(link: str = "identity", *, horizon_trend: bool = False) -> MSM:
    """Two or three terms over four regimens, so the projection is not saturated."""

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        n = len(frame)
        columns = [np.ones(n), np.full(n, DURATION[str(label)])]
        if horizon_trend:
            columns.append(np.full(n, float(horizon)))
        return np.column_stack(columns)

    def weight(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.full(len(frame), PROJECTION[str(label)])

    terms = (*TERMS, "horizon") if horizon_trend else TERMS
    return MSM(
        design=design,
        terms=terms,
        weights=weight,
        weights_kind="known",
        design_kind="known",
        link=link,
    )


def saturated(cells: tuple[tuple[str, int], ...]) -> MSM:
    """One indicator per ``(regimen, horizon)`` cell: the model that must reduce exactly."""

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        return np.eye(len(cells))[cells.index((str(label), int(horizon)))] * np.ones(
            (len(frame), 1)
        )

    return MSM(
        design=design,
        terms=tuple(f"{label}@{horizon}" for label, horizon in cells),
        design_kind="known",
    )


def end_of_study_frame(n: int = 1500, seed: int = 2) -> Any:
    frame, _ = make_longitudinal(n=n, seed=seed)
    return frame


def fit_dose(frame: Any = None, **overrides: Any) -> LongitudinalResult:
    frame = end_of_study_frame() if frame is None else frame
    msm = overrides.pop("msm", dose_msm())
    return LTMLE(SPEC, msm=msm, **settings(**overrides)).fit(frame, outcome="Y", **SHARED)


@pytest.fixture(scope="module")
def dose() -> LongitudinalResult:
    return fit_dose()


@pytest.fixture(scope="module")
def plain() -> LongitudinalResult:
    """The per-regimen cross-fitted fit of the same frame, split and learners."""
    return LTMLE(SPEC, **settings()).fit(end_of_study_frame(), outcome="Y", **SHARED)


def _cell_fit(msm_fit: Any, label: str, horizon: int) -> Any:
    return next(
        fit
        for fit, cell in zip(msm_fit.fits, msm_fit.model.cells, strict=True)
        if cell.label == label and cell.horizon == horizon
    )


def _plain_fit(result: LongitudinalResult, label: str, horizon: int, cause: str | None) -> Any:
    return next(
        fit
        for fit in result.fits.values()
        if fit.regimen.label == label and fit.horizon == horizon and fit.cause == cause
    )


def _assert_saturated_reduction(plain: LongitudinalResult, model: LongitudinalResult) -> None:
    """Every coefficient is its cell's per-regimen estimate, and its curve the cell's curve."""
    assert plain.folds.n_folds == model.folds.n_folds == K
    np.testing.assert_array_equal(plain.folds.assignment, model.folds.assignment)
    assert model.scaler.range == 1.0
    assert plain.converged
    for msm_fit in model.msm_fits:
        assert msm_fit.converged
        for column, cell in enumerate(msm_fit.model.cells):
            reference = _plain_fit(plain, cell.label, cell.horizon, msm_fit.cause)
            assert msm_fit.beta[column] == pytest.approx(
                reference.psi_scaled, rel=COEFFICIENT_GAP, abs=COEFFICIENT_GAP
            )
            np.testing.assert_allclose(
                msm_fit.influence_curves[:, column],
                reference.influence_curve_scaled,
                rtol=0.0,
                atol=CURVE_GAP,
            )


def _node_scores(result: LongitudinalResult) -> list[float]:
    """Each node's stacked relative score per term, against the design the pass used."""
    scores = []
    weights = np.asarray(result.data.weights, dtype=float)
    for msm_fit in result.msm_fits:
        model = msm_fit.model
        times = sorted({step.time for fit in msm_fit.fits for step in fit.steps})
        for time in times:
            score = np.zeros(model.n_terms)
            scale = np.zeros(model.n_terms)
            for index, (cell, fit) in enumerate(zip(model.cells, msm_fit.fits, strict=True)):
                if cell.horizon < time:
                    continue
                step = next(item for item in fit.steps if item.time == time)
                # Every follower: ``clever`` is nonzero exactly on ``trained_on``.
                assert np.array_equal(step.clever != 0.0, step.trained_on)
                multiplier = weights * model.weights[:, index] * step.clever
                covariate = msm_fit.fluctuation_design[:, index, :]
                residual = step.pseudo_outcome - step.targeted
                score += np.sum(multiplier[:, None] * covariate * residual[:, None], axis=0)
                scale += np.sum(np.abs(multiplier[:, None] * covariate), axis=0)
            # A term no live cell carries has no equation at this node.
            live = scale > 0.0
            scores.extend((np.abs(score[live]) / scale[live]).tolist())
    return scores


def _assert_curve_is_centred(result: LongitudinalResult) -> None:
    """``P_n D*_beta = 0``: the node scores and the normal equations, summed."""
    for msm_fit in result.msm_fits:
        curve = np.asarray(msm_fit.influence_curves, dtype=float)
        spread = 1.0 + np.std(curve, axis=0)
        assert np.all(np.abs(np.mean(curve, axis=0)) <= 1e-8 * spread), np.mean(curve, axis=0)


def _longhand_curve(result: LongitudinalResult, msm_fit: Any) -> np.ndarray:
    """The stacked delta-method curve, assembled cell by cell from the returned steps.

    ``M`` is built here from the projection's own definition rather than read off the
    solver: the weighted least-squares Gram matrix, plus the curvature term under a link.
    """
    model = msm_fit.model
    data = result.data
    assert result.scaler.range == 1.0 and result.scaler.lower == 0.0
    weights = np.asarray(data.weights, dtype=float)
    beta = np.asarray(msm_fit.beta, dtype=float)
    n, p = data.n, model.n_terms
    gram = np.zeros((p, p))
    contribution = np.zeros((n, p))
    for index, fit in enumerate(msm_fit.fits):
        phi = model.design[:, index, :]
        h = model.weights[:, index]
        eta = phi @ beta
        if model.link == "identity":
            mean, slope, bend = eta, np.ones(n), np.zeros(n)
        else:
            assert model.link == "logit"
            mean = expit(eta)
            slope = mean * (1.0 - mean)
            bend = slope * (1.0 - 2.0 * mean)
        first = fit.steps[0].targeted
        residual = np.zeros(n)
        for step in fit.steps:
            residual = residual + step.clever * (step.pseudo_outcome - step.targeted)
        bracket = residual + first - mean
        curvature = slope**2 - (first - mean) * bend
        gram += np.einsum("i,ip,iq->pq", weights * h * curvature, phi, phi)
        contribution += (weights * h * slope * bracket)[:, None] * phi
    gram /= float(np.sum(weights))
    return np.asarray(contribution @ np.linalg.inv(gram).T)


# --------------------------------------------------------------------------------------
# T1: a saturated model is the cross-fitted per-regimen report.
# --------------------------------------------------------------------------------------


def _survival_pair(**overrides: Any) -> tuple[LongitudinalResult, LongitudinalResult]:
    frame, _ = make_longitudinal_survival(n=1500, seed=4)
    spec = {"always": 1, "never": 0}
    cells = tuple((label, horizon) for horizon in (1, 2) for label in spec)
    configured = settings(tol=REDUCTION_TOL, **overrides)
    plain = LTMLE(spec, reference="never", **configured).fit(frame, outcome=("Y1", "Y2"), **SHARED)
    model = LTMLE(spec, msm=saturated(cells), **configured).fit(
        frame, outcome=("Y1", "Y2"), **SHARED
    )
    return plain, model


def _competing_pair() -> tuple[LongitudinalResult, LongitudinalResult]:
    frame, _ = make_longitudinal_competing(n=1800, seed=5)
    spec = {"always": 1, "never": 0}
    cells = tuple((label, horizon) for horizon in (1, 2) for label in spec)
    outcome = {"relapse": ("R1", "R2"), "death": ("D1", "D2")}
    configured = settings(tol=REDUCTION_TOL)
    plain = LTMLE(spec, reference="never", **configured).fit(frame, outcome=outcome, **SHARED)
    model = LTMLE(spec, msm=saturated(cells), **configured).fit(frame, outcome=outcome, **SHARED)
    return plain, model


CATEGORICAL_SPEC: dict[str, Any] = {
    "never": 0,
    "high": (2, 1),
    "dynamic": DynamicRegimen(
        "dynamic", (2, lambda history: (history["L2"] > 0).astype(float)), rule_kind="known"
    ),
}


def _categorical_pair() -> tuple[LongitudinalResult, LongitudinalResult]:
    """A three-level first node, a static plan on its third level, and a dynamic rule."""
    frame = multivalue_panel(n=1500, seed=8)
    cells = tuple((label, 2) for label in CATEGORICAL_SPEC)
    configured = settings(tol=REDUCTION_TOL)
    plain = LTMLE(CATEGORICAL_SPEC, reference="never", **configured).fit(
        frame, **CATEGORICAL_COLUMNS
    )
    model = LTMLE(CATEGORICAL_SPEC, msm=saturated(cells), **configured).fit(
        frame, **CATEGORICAL_COLUMNS
    )
    return plain, model


PAIRS = {
    "survival": _survival_pair,
    "competing": _competing_pair,
    "categorical_dynamic": _categorical_pair,
}


@pytest.fixture(scope="module", params=sorted(PAIRS))
def pair(request: pytest.FixtureRequest) -> tuple[LongitudinalResult, LongitudinalResult]:
    return PAIRS[request.param]()


@pytest.mark.filterwarnings("error::cleverly.exceptions.ConvergenceWarning")
def test_a_saturated_model_is_the_cross_fitted_per_regimen_report(
    pair: tuple[LongitudinalResult, LongitudinalResult],
) -> None:
    plain, model = pair
    if model.data.is_competing:
        assert [fit.cause for fit in model.msm_fits] == list(model.data.cause_labels)
    if any(len(levels) > 2 for levels in model.data.treatment_levels):
        assignment = np.asarray(_plain_fit(plain, "dynamic", 2, None).assignment)
        assert np.unique(assignment[:, 1]).size == 2
    _assert_saturated_reduction(plain, model)


def test_m1_a_fold_local_fluctuation_breaks_the_reduction_and_the_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1: a fold-local fluctuation over the stitched initial fit.

    Each fold solves on its training followers and keeps its held-out rows.  It is not the
    deleted construction exactly, which also fitted each fold's own initial fit and its own
    ``beta``, but it carries that construction's fold-local update.  It fails the saturated
    reduction (T1) and leaves the pooled score unsolved (T4).
    """
    plain, _ = _survival_pair()
    folds = plain.folds
    n = plain.n
    real = longitudinal_msm.solve_fluctuation

    def fold_local(
        outcome: Any, initial: Any, submodel: Any, weights: Any, observed: Any, **kw: Any
    ) -> Any:
        cells = len(outcome) // n
        stitched = np.empty(len(outcome))
        first = None
        for train, test in folds:
            training = np.zeros(n, dtype=bool)
            training[train] = True
            held = np.zeros(n, dtype=bool)
            held[test] = True
            solved = real(
                outcome, initial, submodel, weights, observed & np.tile(training, cells), **kw
            )
            first = solved if first is None else first
            rows = np.tile(held, cells)
            stitched[rows] = solved.targeted.arms[0.0][rows]
        assert first is not None
        return dataclasses.replace(first, targeted=InitialFit(stitched, {0.0: stitched}))

    monkeypatch.setattr(longitudinal_msm, "solve_fluctuation", fold_local)
    _, mutated = _survival_pair()
    with pytest.raises(AssertionError):
        _assert_saturated_reduction(plain, mutated)
    assert max(_node_scores(mutated)) > 1e-3


# --------------------------------------------------------------------------------------
# T2: the stitched initial fit is the per-regimen fold recursion.
# --------------------------------------------------------------------------------------


def test_every_cell_starts_from_the_per_regimen_fold_recursion(
    dose: LongitudinalResult, plain: LongitudinalResult
) -> None:
    np.testing.assert_array_equal(dose.folds.assignment, plain.folds.assignment)
    for cell, fit in zip(dose.msm.cells, dose.msm_fits[0].fits, strict=True):
        reference = plain.fits[cell.label]
        for left, right in zip(fit.steps, reference.steps, strict=True):
            assert np.array_equal(left.initial, right.initial)
            assert np.array_equal(left.regression_target, right.regression_target)
            assert np.array_equal(left.clever, right.clever)
    _assert_stitched_longhand(dose)
    _assert_clever_is_the_out_of_fold_reciprocal(dose)


def test_m3_a_fold_targeted_carry_moves_the_stitched_initial_fit(
    dose: LongitudinalResult, plain: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3: the pre-RM18 fold recursion, which carries a training-fold fluctuation, fails T2."""
    mechanism = plain.mechanism
    bounds = plain.config.g_bounds

    def fold_targeted(
        data: Any,
        cell: Any,
        *,
        scaler: Any,
        cause: Any,
        outcome_learner: Any,
        pseudo_learner: Any,
        outer_train: Any,
        fold: int,
        test: Any,
    ) -> Any:
        _, cumulative = mechanism.cumulative_with_unbounded(data, cell.plan, bounds)
        outputs = {}
        carried = sequential.seed_carried(data, scaler)
        for time in range(cell.horizon, 0, -1):
            node = sequential._fit_node_regression(
                data,
                cell.plan,
                carried,
                time,
                cell.horizon,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                folds=Folds.single(data.n),
                cause=cause,
                masks=cell.masks,
                fit_rows=outer_train,
                outer_fold=fold,
            )
            outputs[time] = (
                node.initial[test],
                node.pseudo_outcome[test],
                node.learner_diagnostics,
                None,
            )
            counterfactual, _ = sequential._clever_covariate(
                node.at_risk, node.trained_on, cumulative, time
            )
            solved = sequential._fluctuate_node(
                node.pseudo_outcome,
                node.initial,
                data.weights * counterfactual,
                node.fitted_on,
                label=cell.plan.label,
                time=time,
                alpha=0.9995,
                max_iter=20,
                tol=TOL,
            )
            carried = np.where(node.at_risk, solved.targeted.arms[0.0], 0.5)
        return outputs

    monkeypatch.setattr(sequential, "_untargeted_recursion_in_fold", fold_targeted)
    mutated = fit_dose()
    moved = max(
        float(np.max(np.abs(fit.steps[0].initial - plain.fits[cell.label].steps[0].initial)))
        for cell, fit in zip(mutated.msm.cells, mutated.msm_fits[0].fits, strict=True)
    )
    assert moved > 1e-6


def test_an_active_bound_reaches_the_clever_covariate_of_every_cell() -> None:
    """T2 where the lower bound replaces cells the stacked score reads.

    ``clever`` divides by the bounded out-of-fold prefix.  A pooled pass that divided by the
    raw product would fail the reciprocal check here, because some cells were clipped.
    """
    with pytest.warns(PositivityWarning, match="truncation"):
        result = fit_dose(g_bounds=(0.2, 1.0))
    clipped = 0
    for fit in result.msm_fits[0].fits:
        for step in fit.steps:
            column = step.time - 1
            clipped += int(
                np.count_nonzero(
                    fit.cumulative_unbounded[step.trained_on, column]
                    != fit.cumulative[step.trained_on, column]
                )
            )
    assert clipped > 0
    _assert_clever_is_the_out_of_fold_reciprocal(result)
    assert max(_node_scores(result)) < 1e-9


def test_m7_a_fold_model_read_as_the_out_of_fold_mechanism_fails_the_reciprocal(
    dose: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M7: dividing by fold 0's refit treatment model everywhere."""
    fold_zero = refit_fold_mechanism(dose, 0)
    real = Mechanism.cumulative_with_unbounded

    def fold_model(self: Mechanism, data: Any, plan: Any, bounds: Any) -> Any:
        return real(fold_zero, data, plan, bounds)

    monkeypatch.setattr(Mechanism, "cumulative_with_unbounded", fold_model)
    mutated = fit_dose()
    monkeypatch.undo()
    with pytest.raises(AssertionError):
        _assert_clever_is_the_out_of_fold_reciprocal(mutated)


# --------------------------------------------------------------------------------------
# T3 and T4: the curve is the stacked delta-method curve, and every score is solved.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logit_dose() -> LongitudinalResult:
    return fit_dose(msm=dose_msm("logit"))


@pytest.mark.parametrize("which", ["identity", "logit"])
def test_the_curve_is_the_stacked_delta_method_curve(
    which: str, dose: LongitudinalResult, logit_dose: LongitudinalResult
) -> None:
    result = dose if which == "identity" else logit_dose
    msm_fit = result.msm_fits[0]
    # Unsaturated and nonuniform: four cells, two terms, four distinct weights.
    assert msm_fit.model.n_cells > msm_fit.model.n_terms
    assert np.unique(msm_fit.model.weights).size == 4
    curve = msm_fit.influence_curves
    longhand = _longhand_curve(result, msm_fit)
    np.testing.assert_allclose(curve, longhand, rtol=0.0, atol=1e-12 * float(np.max(np.abs(curve))))


@pytest.mark.parametrize("which", ["identity", "logit"])
def test_every_node_solves_its_stacked_score_over_every_follower(
    which: str, dose: LongitudinalResult, logit_dose: LongitudinalResult
) -> None:
    result = dose if which == "identity" else logit_dose
    assert max(_node_scores(result)) < 1e-9
    _assert_curve_is_centred(result)


def test_m4_dropping_the_projection_weight_from_the_loss_fails_t4(
    dose: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4: the stacked loss weight without ``h``.  Every cell is live at both nodes."""
    stacked = dose.msm.weights.T.reshape(-1)
    real = longitudinal_msm.solve_fluctuation

    def without_h(
        outcome: Any, initial: Any, submodel: Any, weights: Any, observed: Any, **kw: Any
    ) -> Any:
        return real(outcome, initial, submodel, weights / stacked, observed, **kw)

    monkeypatch.setattr(longitudinal_msm, "solve_fluctuation", without_h)
    assert max(_node_scores(fit_dose())) > 1e-4


# --------------------------------------------------------------------------------------
# T5 and T6: the witnesses, and the score rows.
# --------------------------------------------------------------------------------------


def test_every_node_moves_in_every_term_and_the_folds_move_beta() -> None:
    frame = end_of_study_frame()
    rough = {"outcome_learner": DummyRegressor(), "pseudo_learner": DummyRegressor()}
    crossed = fit_dose(frame, **rough)
    inside = fit_dose(frame, n_folds=1, **rough)
    for node in crossed.msm_fits[0].nodes:
        assert np.all(np.abs(np.asarray(node.epsilon)) > 1e-4), node.epsilon
    shift = np.abs(crossed.msm_fits[0].beta - inside.msm_fits[0].beta)
    assert float(np.max(shift)) >= 1e-6


def test_score_rows_are_one_solver_row_per_node_and_term(dose: LongitudinalResult) -> None:
    report = dose.diagnostics.score_equations()
    assert [row.kind for row in report.rows] == ["solver"] * (dose.data.n_times * len(TERMS))
    assert all(row.passed for row in report.rows)
    assert {field.name for field in dataclasses.fields(report.rows[0])}.isdisjoint({"z"})
    assert "z" not in report.to_frame().columns
    assert not hasattr(cleverly.assessment, "STITCHED_SCORE_Z_TOLERANCE")
    assert not hasattr(report, "z_tolerance")


# --------------------------------------------------------------------------------------
# T7: links, and the learners are fitted once.
# --------------------------------------------------------------------------------------


class CountingLinear(LinearRegression):
    """A linear regression that counts its fits, on the class so it survives ``clone``."""

    fits: ClassVar[int] = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CountingLinear:
        CountingLinear.fits += 1
        return super().fit(X, y, sample_weight=sample_weight)


#: The unpatched method, for the comparisons a spy must not intercept.
DESIGN_AT = RegimenMSM.fluctuation_design_at


def _spy_designs(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray | None]:
    """Record the ``beta`` of every design a pass is built at, in order."""
    seen: list[np.ndarray | None] = []
    real = DESIGN_AT

    def spy(self: RegimenMSM, beta: Any) -> Any:
        seen.append(None if beta is None else np.array(beta, dtype=float))
        return real(self, beta)

    monkeypatch.setattr(RegimenMSM, "fluctuation_design_at", spy)
    return seen


def _assert_last_design_and_rounds(result: LongitudinalResult, seen: list[Any]) -> None:
    msm_fit = result.msm_fits[0]
    assert msm_fit.alternation.converged
    assert len(msm_fit.alternation.trace) >= 2
    beta_in = seen[-1]
    assert beta_in is not None
    # The design of the last pass is the one at that round's *input* beta.
    assert np.array_equal(msm_fit.fluctuation_design, DESIGN_AT(msm_fit.model, beta_in))
    shift = np.max(np.abs(beta_in - msm_fit.beta)) / (1.0 + np.max(np.abs(beta_in)))
    assert shift <= TOL


@pytest.mark.parametrize("link", ["log", "logit"])
def test_a_link_converges_on_the_last_design_and_refits_no_learner(
    link: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _spy_designs(monkeypatch)
    CountingLinear.fits = 0
    learners = {"outcome_learner": CountingLinear(), "pseudo_learner": CountingLinear()}
    result = fit_dose(msm=dose_msm(link), **learners)
    _assert_last_design_and_rounds(result, seen)
    # One regression per fold, cell and node, whatever the round count.
    cells = result.msm.cells
    assert CountingLinear.fits == K * sum(cell.horizon for cell in cells)
    _assert_curve_is_centred(result)


def test_a_survival_grid_refits_no_learner_across_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_designs(monkeypatch)
    CountingLinear.fits = 0
    frame, _ = make_longitudinal_survival(n=1500, seed=4)
    result = LTMLE(
        {"always": 1, "never": 0},
        msm=dose_msm("logit", horizon_trend=True),
        **settings(outcome_learner=CountingLinear(), pseudo_learner=CountingLinear()),
    ).fit(frame, outcome=("Y1", "Y2"), **SHARED)
    _assert_last_design_and_rounds(result, seen)
    assert CountingLinear.fits == K * sum(cell.horizon for cell in result.msm.cells) == K * 6


def test_m6_a_frozen_beta_in_the_covariate_fails_the_design_and_the_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6: the covariate stays at the first round's ``beta`` while ``beta`` moves."""
    real = DESIGN_AT
    frozen: list[np.ndarray] = []

    def stuck(self: RegimenMSM, beta: Any) -> Any:
        if beta is not None and not frozen:
            frozen.append(np.array(beta, dtype=float))
        return real(self, frozen[0] if beta is not None else None)

    monkeypatch.setattr(RegimenMSM, "fluctuation_design_at", stuck)
    mutated = fit_dose(msm=dose_msm("logit"))
    monkeypatch.undo()
    msm_fit = mutated.msm_fits[0]
    assert not np.array_equal(msm_fit.fluctuation_design, real(msm_fit.model, msm_fit.beta))
    with pytest.raises(AssertionError):
        _assert_curve_is_centred(mutated)


# --------------------------------------------------------------------------------------
# T8: a held-out outcome reaches no initial fit of its own fold.
# --------------------------------------------------------------------------------------

NEIGHBOURS = {
    "outcome_learner": KNeighborsClassifier(n_neighbors=1),
    "pseudo_learner": KNeighborsRegressor(n_neighbors=1),
}


def _flip_one_outcome() -> tuple[Any, Any, int]:
    frame = end_of_study_frame(n=600, seed=3)
    observed = frame["C1"].eq(1.0) & frame["C2"].eq(1.0) & frame["A1"].eq(1.0) & frame["A2"].eq(1.0)
    row = int(np.flatnonzero(observed.to_numpy())[0])
    flipped = frame.copy()
    flipped.loc[row, "Y"] = 1.0 - flipped.loc[row, "Y"]
    return frame, flipped, row


def _assert_no_leak(before: LongitudinalResult, after: LongitudinalResult, row: int) -> None:
    folds = before.folds.assignment
    np.testing.assert_array_equal(folds, after.folds.assignment)
    own = folds == folds[row]
    moved_elsewhere = False
    for left, right in zip(before.msm_fits[0].fits, after.msm_fits[0].fits, strict=True):
        for early, late in zip(left.steps, right.steps, strict=True):
            np.testing.assert_array_equal(early.initial[own], late.initial[own])
            moved_elsewhere |= bool(np.any(early.initial[~own] != late.initial[~own]))
    # The witness: the flip reaches a row of another fold, whose training rows hold it.
    assert moved_elsewhere


def test_a_held_out_outcome_reaches_no_initial_fit_of_its_own_fold() -> None:
    frame, flipped, row = _flip_one_outcome()
    _assert_no_leak(fit_dose(frame, **NEIGHBOURS), fit_dose(flipped, **NEIGHBOURS), row)


def test_m2_in_sample_nuisances_in_the_fold_recursion_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2: a fold regression fitted on every row reads its own held-out outcome."""
    real = sequential._fit_node_regression

    def in_sample(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **{**kwargs, "fit_rows": None})

    monkeypatch.setattr(sequential, "_fit_node_regression", in_sample)
    frame, flipped, row = _flip_one_outcome()
    with pytest.raises(AssertionError):
        _assert_no_leak(fit_dose(frame, **NEIGHBOURS), fit_dose(flipped, **NEIGHBOURS), row)


# --------------------------------------------------------------------------------------
# T9: observation weights.
# --------------------------------------------------------------------------------------


def _weighted_frame() -> Any:
    frame, _ = make_longitudinal_weighted(n=1400, seed=6)
    return frame


@pytest.mark.filterwarnings("error::cleverly.exceptions.ConvergenceWarning")
def test_weights_keep_the_saturated_reduction_and_the_weighted_score() -> None:
    frame = _weighted_frame()
    assert np.ptp(frame["w"].to_numpy()) > 0.5
    spec = {"always": 1, "never": 0}
    cells = tuple((label, 2) for label in spec)
    configured = settings(tol=REDUCTION_TOL)
    plain = LTMLE(spec, reference="never", **configured).fit(
        frame, outcome="Y", weights="w", **SHARED
    )
    model = LTMLE(spec, msm=saturated(cells), **configured).fit(
        frame, outcome="Y", weights="w", **SHARED
    )
    _assert_saturated_reduction(plain, model)
    weighted = LTMLE(SPEC, msm=dose_msm(), **settings()).fit(
        frame, outcome="Y", weights="w", **SHARED
    )
    assert max(_node_scores(weighted)) < 1e-9
    _assert_curve_is_centred(weighted)


def test_m5_dropping_the_observation_weights_from_the_loss_fails_t9(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _weighted_frame()
    reference = LTMLE(SPEC, msm=dose_msm(), **settings()).fit(
        frame, outcome="Y", weights="w", **SHARED
    )
    observation = np.asarray(reference.data.weights, dtype=float)
    real = longitudinal_msm.solve_fluctuation

    def untilted(
        outcome: Any, initial: Any, submodel: Any, weights: Any, observed: Any, **kw: Any
    ) -> Any:
        tilt = np.tile(observation, len(outcome) // len(observation))
        return real(outcome, initial, submodel, weights / tilt, observed, **kw)

    monkeypatch.setattr(longitudinal_msm, "solve_fluctuation", untilted)
    mutated = LTMLE(SPEC, msm=dose_msm(), **settings()).fit(
        frame, outcome="Y", weights="w", **SHARED
    )
    assert max(_node_scores(mutated)) > 1e-4


# --------------------------------------------------------------------------------------
# T10: survival and competing risks on an unsaturated grid.
# --------------------------------------------------------------------------------------


def test_a_survival_trend_model_has_the_stacked_curve_and_solved_scores() -> None:
    frame, _ = make_longitudinal_survival(n=1500, seed=4)
    result = LTMLE({"always": 1, "never": 0}, msm=dose_msm(horizon_trend=True), **settings()).fit(
        frame, outcome=("Y1", "Y2"), **SHARED
    )
    msm_fit = result.msm_fits[0]
    assert {cell.horizon for cell in msm_fit.model.cells} == {1, 2}
    assert msm_fit.model.n_cells == 4 and msm_fit.model.n_terms == 3
    curve = msm_fit.influence_curves
    np.testing.assert_allclose(
        curve,
        _longhand_curve(result, msm_fit),
        rtol=0.0,
        atol=1e-12 * float(np.max(np.abs(curve))),
    )
    assert max(_node_scores(result)) < 1e-9
    _assert_curve_is_centred(result)


def test_competing_causes_share_one_mechanism_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = longitudinal_estimator.fit_mechanism

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(longitudinal_estimator, "fit_mechanism", counted)
    frame, _ = make_longitudinal_competing(n=1800, seed=5)
    result = LTMLE({"always": 1, "never": 0}, msm=dose_msm(horizon_trend=True), **settings()).fit(
        frame, outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")}, **SHARED
    )
    assert len(calls) == 1
    assert len(result.msm_fits) == 2
    first, second = result.msm_fits
    for left, right in zip(first.fits, second.fits, strict=True):
        assert np.array_equal(left.cumulative, right.cumulative)
        assert left.steps[0].clever is not None
        np.testing.assert_array_equal(left.steps[0].clever, right.steps[0].clever)
    assert max(_node_scores(result)) < 1e-9
    _assert_curve_is_centred(result)


# --------------------------------------------------------------------------------------
# T11: clusters.
# --------------------------------------------------------------------------------------

THREE = {"always": 1, "never": 0, "early": (1, 0)}


def _clustered(clusters: int, **overrides: Any) -> LongitudinalResult:
    frame, _ = cluster_law.draw_end_of_study(clusters, "equal40", 1)
    configured = {
        "outcome_learner": subject_study.QuasiBinomialGLM(),
        "pseudo_learner": subject_study.QuasiBinomialGLM(),
        "treatment_learner": subject_study.KnownLongitudinalMechanism("treatment"),
        "censoring_learner": subject_study.KnownLongitudinalMechanism("censoring"),
        "n_folds": K,
        "learner_folds": 2,
        "g_bounds": subject_study.G_BOUNDS,
        "simultaneous": False,
        "max_iter": 100,
        "tol": TOL,
        "random_state": 0,
        **overrides,
    }
    return LTMLE(THREE, msm=dose_msm(), **configured).fit(
        frame,
        outcome="Y",
        id="id",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )


def _assert_cluster_se(result: LongitudinalResult) -> None:
    cluster = np.asarray(result.data.cluster)
    labels = np.unique(cluster)
    n = result.n
    for name in (f"msm_regimen[{term}]" for term in TERMS):
        curve = np.asarray(result[name].influence_curve, dtype=float)
        sums = np.array([curve[cluster == label].sum() for label in labels])
        expected = np.sqrt(labels.size * np.var(sums, ddof=1) / n**2)
        assert result[name].std_error == pytest.approx(expected, rel=1e-12)


def _assert_whole_clusters(result: LongitudinalResult) -> None:
    cluster = np.asarray(result.data.cluster)
    for label in np.unique(cluster):
        assert np.unique(result.folds.assignment[cluster == label]).size == 1


def test_forty_clusters_cross_fit_whole_clusters_with_a_normal_band() -> None:
    result = _clustered(40, simultaneous=True)
    _assert_whole_clusters(result)
    _assert_cluster_se(result)
    assert result.inference_status == "influence_curve"
    assert {result[f"msm_regimen[{term}]"].reference_df for term in TERMS} == {None}
    assert result.simultaneous is not None


def test_twenty_one_clusters_take_the_t_reference_and_skip_the_band() -> None:
    result = _clustered(21, simultaneous=True)
    _assert_whole_clusters(result)
    _assert_cluster_se(result)
    assert {result[f"msm_regimen[{term}]"].reference_df for term in TERMS} == {19}
    assert result.simultaneous is None
    assert NO_T_REFERENCE_BANDS in result.summary()


def test_nineteen_clusters_withhold_the_interval_of_every_coefficient() -> None:
    """Below the cross-fitted floor of 20 clusters, a working-model fit reports no interval."""
    result = _clustered(19)
    _assert_whole_clusters(result)
    assert result.inference_status == "few_cluster_plugin"
    for term in TERMS:
        estimate = result[f"msm_regimen[{term}]"]
        assert estimate.reference_df is None
        with pytest.raises(CapabilityError):
            _ = estimate.ci


def test_m8_a_row_level_split_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """M8: rows of one cluster in two folds.  The integrity check refuses before any learner."""

    def row_folds(self: LTMLE, data: Any) -> Folds:
        return random_partition(data.n, self.n_folds, seed=0)

    monkeypatch.setattr(LTMLE, "_folds", row_folds)
    with pytest.raises(DataError, match="cluster"):
        _clustered(40)


# --------------------------------------------------------------------------------------
# T12: what surrounds the fit.
# --------------------------------------------------------------------------------------


def test_the_band_is_built_from_the_coefficient_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ...]] = []
    real = longitudinal_estimator.simultaneous_bands

    def spy(estimates: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(tuple(estimates))
        return real(estimates, *args, **kwargs)

    monkeypatch.setattr(longitudinal_estimator, "simultaneous_bands", spy)
    first = fit_dose(simultaneous=True)
    second = fit_dose(simultaneous=True)
    assert seen and set(seen[0]) == {f"msm_regimen[{term}]" for term in TERMS}
    assert first.simultaneous is not None and second.simultaneous is not None
    assert first.simultaneous.critical_value == second.simultaneous.critical_value


def test_the_bootstrap_of_a_cross_fitted_working_model_is_a_diagnostic() -> None:
    frame = end_of_study_frame(n=600, seed=3)
    result = fit_dose(frame, n_bootstrap=20)
    assert result.folds.n_folds == K
    kind = longitudinal_estimator.bootstrap_design_kind(
        result.data, result.folds, [plan.regimen for plan in result.replay_recipe.plans], result.msm
    )
    assert kind is None
    assert result.bootstrap is not None
    assert result.bootstrap.n_requested == 20


def test_the_truncation_replay_is_a_fresh_fit_at_the_replay_bound(
    dose: LongitudinalResult,
) -> None:
    curve = dose.diagnostics.truncation_curve(bounds=[0.05])
    rows = curve[~curve["is_fitted_bound"]]
    fresh = fit_dose(g_bounds=(0.05, 1.0))
    assert len(rows) == len(TERMS)
    for row in rows.itertuples():
        assert row.psi == pytest.approx(fresh[row.estimand].psi, rel=1e-10, abs=1e-12)
    # The witness: the bound moved the estimate.
    assert any(row.psi != dose[row.estimand].psi for row in rows.itertuples())


def test_the_ratio_view_of_a_log_link_exponentiates_the_coefficients() -> None:
    result = fit_dose(msm=dose_msm("log"))
    table = result.coefficients(scale="ratio")
    for row in table.itertuples():
        assert row.psi == pytest.approx(np.exp(result[row.estimand].psi), rel=1e-12)


# --------------------------------------------------------------------------------------
# T13: what is still refused, and what no longer is.
# --------------------------------------------------------------------------------------


def test_the_removed_refusal_appears_nowhere() -> None:
    message = "msm= with n_folds" + " > 1 is not supported"
    hits = []
    for folder in ("src", "docs", "tests"):
        for path in (ROOT / folder).rglob("*"):
            if "_build" in path.parts or path.suffix not in {".py", ".md", ".rst", ".ipynb"}:
                continue
            if message in path.read_text(encoding="utf-8", errors="replace"):
                hits.append(path)
    assert hits == []


def test_a_cross_fitted_continuous_outcome_still_needs_q_bounds() -> None:
    frame = end_of_study_frame(n=600, seed=3)
    frame["Y"] = frame["Y"] + 0.5 * frame["W1"]
    with pytest.raises(LongitudinalError, match="needs a declared q_bounds"):
        fit_dose(frame)


def test_a_cell_nobody_follows_in_a_training_fold_names_the_cell() -> None:
    frame = end_of_study_frame(n=600, seed=3)
    # Every unit that reaches the second node repeats its first arm, so no unit follows
    # "early" or "late", in any training fold.
    frame["A2"] = frame["A1"].where(frame["A2"].notna())
    with pytest.raises(
        LongitudinalError,
        match=r"no unit followed regimen '(early|late)' through time 2 in outer training fold",
    ):
        fit_dose(frame)


def test_a_reference_regimen_is_still_refused_beside_a_working_model() -> None:
    with pytest.raises(ValueError, match="reference= names the regimen"):
        LTMLE(SPEC, reference="never", msm=dose_msm(), n_folds=K)


# --------------------------------------------------------------------------------------
# T14: agreement with the in-sample fit on the discrete law.
# --------------------------------------------------------------------------------------

#: Measured on seeds 0 to 4 before this pin was written: the largest
#: ``|beta_cf - beta_in| / SE`` was 0.104 (seed 3, the duration term).  The bound keeps at
#: least three times that.
IN_SAMPLE_AGREEMENT = 0.35


def _discrete_fit(frame: Any, n_folds: int) -> LongitudinalResult:
    return LTMLE(
        discrete_properties.REGIMENS,
        msm=declared_msm(discrete_properties.DURATION, discrete_properties.PROJECTION_WEIGHT),
        outcome_learner=discrete_law.CellMeans(),
        pseudo_learner=discrete_law.CellMeans(),
        treatment_learner=discrete_law.CellMeans(),
        censoring_learner=discrete_law.CellMeans(),
        n_folds=n_folds,
        g_bounds=discrete_properties.G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=TOL,
        random_state=0,
    ).fit(frame, **discrete_properties.COLUMNS)


@pytest.mark.parametrize("seed", range(5))
def test_the_cross_fitted_fit_agrees_with_the_in_sample_fit(seed: int) -> None:
    frame = discrete_properties.sample(discrete_law.PROBS, 20_000, seed)
    inside = _discrete_fit(frame, 1)
    crossed = _discrete_fit(frame, K)
    for name in discrete_properties.NAMES.values():
        ratio = abs(crossed[name].psi - inside[name].psi) / inside[name].std_error
        assert ratio <= IN_SAMPLE_AGREEMENT, (name, ratio)
        # The witness: the two are different estimators of the same coefficient.
        assert crossed[name].psi != inside[name].psi
