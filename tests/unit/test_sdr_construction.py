r"""T3, T4, T5: the SDR recursion's algebra and its cross-fit construction.

T5 compares the engine with :func:`tests.longitudinal_sdr.longhand_sdr`, a numpy recursion
that fits its own cell-mean mechanism and regressions per fold.  The two share no code, so
agreement to ``1e-12`` checks the transform, the partial weights, the training-fold
construction and the pooling together.  Two constructions the plan rejects then have to
move the estimate by more than ``1e-4``: the printed Step 2, which stitches each row's
transform from the folds that hold it out (M7a), and ``lmtp``'s training rows that read the
out-of-fold mechanism (M7b).  The sample is iid from the exact law, so each fold's cell
means differ from the out-of-fold ones and both alternatives have something to move.

T3 checks the bilinear remainder of the paper's Lemma 3 on the exact law: with every other
nuisance exact, scaling the node-2 ratio by :math:`1 + a h` and shifting the node-2
regression by :math:`b k` moves the estimate by exactly :math:`c\,ab`.  T4 checks the
one-step form of the recursion where no bound binds.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly.learners.density import bin_edges
from cleverly.longitudinal import SDR
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.regimen import attach_policy_numerators
from cleverly.longitudinal.sdr import FoldMechanism
from cleverly.longitudinal.sequential import fit_mechanism

from .. import discrete_law_longitudinal as law
from ..longitudinal_sdr import COLUMNS, longhand_sdr, sample
from ..studies import longitudinal_mtp_common as mtp_common
from ..studies.canonical_ltmle import declared_regimens

#: The floor below which two exactly computed numbers count as equal.
EXACT_FLOOR = 1e-12
#: How far a rejected construction must move the estimate to count as caught.
CAUGHT = 1e-4
PLANS = {"never": (0, 0), "always": (1, 1)}
REGIMENS = declared_regimens({label: law.REGIMEN_SPEC[label] for label in PLANS})
#: The iid sample and seed the construction checks read; measured gaps for M7a are at least
#: 9e-4 and for M7b at least 2e-4 on it.
N, SEED = 600, 3


def _fit(frame: Any, n_folds: int, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": law.CellMeans(),
        "pseudo_learner": law.CellMeans(),
        "treatment_learner": law.CellMeans(),
        "censoring_learner": law.CellMeans(),
        "n_folds": n_folds,
        "random_state": 4,
        "simultaneous": False,
    }
    settings.update(overrides)
    regimens = settings.pop("regimens", REGIMENS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDR(regimens, **settings).fit(frame, **COLUMNS)


@pytest.fixture(scope="module")
def frame() -> Any:
    return sample(N, SEED)


@pytest.fixture(scope="module", params=[1, 5])
def fitted(request: Any, frame: Any) -> tuple[int, Any]:
    return request.param, _fit(frame, request.param)


# ------------------------------------------------------------------ T5


@pytest.mark.parametrize("label", sorted(PLANS))
def test_the_engine_is_the_longhand_recursion(
    fitted: tuple[int, Any], frame: Any, label: str
) -> None:
    """Estimate, curve and each fold's node-2 transform, in sample and at five folds."""
    _, fit = fitted
    assignment = fit.folds.assignment
    longhand = longhand_sdr(frame, PLANS[label], assignment)
    name = f"ey_regimen[{label}]"
    assert fit[name].psi == pytest.approx(longhand.psi, abs=EXACT_FLOOR)
    np.testing.assert_allclose(fit.influence_curves[name], longhand.curve, atol=1e-12, rtol=0)
    second = fit.fits[label].steps[1]
    assert second.time == 2
    for fold, phi2 in longhand.fold_phi2.items():
        # Off the rows at risk the engine holds a filler that nothing reads.
        held = (assignment == fold) & second.at_risk
        np.testing.assert_allclose(second.transform[held], phi2[held], atol=1e-12, rtol=0)


def test_the_stitched_step_2_is_a_different_estimate(frame: Any) -> None:
    """M7a: the printed Step 2 moves every estimate the training-fold recursion reports."""
    fit = _fit(frame, 5)
    for label, arms in PLANS.items():
        stitched = longhand_sdr(frame, arms, fit.folds.assignment, construction="stitched")
        assert abs(fit[f"ey_regimen[{label}]"].psi - stitched.psi) > CAUGHT


def _recombined(data: Any, plans: Any, *, folds: Any, **settings: Any) -> tuple[FoldMechanism, ...]:
    """M7b: every fold reads the out-of-fold mechanism at its training rows."""
    settings.pop("n_jobs", None)
    mechanism = fit_mechanism(data, plans, folds=folds, **settings)
    attached = attach_policy_numerators(plans, mechanism.policy_numerators)
    return tuple(FoldMechanism(mechanism, attached) for _ in range(folds.n_folds))


def test_training_rows_that_read_the_out_of_fold_mechanism_move_the_estimate(
    frame: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M7b, ``lmtp``'s recombined ratio matrix, needs an estimated mechanism to show."""
    shipped = _fit(frame, 5)
    monkeypatch.setattr(estimator_module, "fit_fold_mechanisms", _recombined)
    mutated = _fit(frame, 5)
    for label in PLANS:
        name = f"ey_regimen[{label}]"
        assert abs(mutated[name].psi - shipped[name].psi) > CAUGHT


def test_each_fold_fits_its_mechanism_on_its_training_rows(frame: Any) -> None:
    """``fit_rows`` reaches the learner: node 1's treatment factor is the training cell mean."""
    fit = _fit(frame, 5)
    w = frame["W"].to_numpy(dtype=float)
    treated = (frame["A1"].to_numpy(dtype=float) == 1.0).astype(float)
    assert len(fit.fold_mechanisms) == 5
    for fold, (train, _) in enumerate(fit.folds):
        rows = np.zeros(len(frame), dtype=bool)
        rows[train] = True
        expected = np.array([treated[rows & (w == value)].mean() for value in w])
        reported = fit.fold_mechanisms[fold].mechanism.treatment[0]["always"]
        np.testing.assert_allclose(reported, expected, atol=1e-15, rtol=0)


def test_a_fold_density_is_the_out_of_fold_density_at_its_held_out_rows() -> None:
    """A continuous node's fold density reads the out-of-fold bin edges.

    The witness is the oracle hazard of ``longitudinal-mtp``, which answers for the
    whole-sample edges.  A fold density binned on its training rows alone handed it other
    bins, and its policy ratios lost about half their correlation with the truth.
    """
    sample = mtp_common.sample_continuous(400, 11)
    bins = 12
    edges = tuple(
        tuple(bin_edges(sample[column].to_numpy(dtype=float), bins)) for column in ("A1", "A2")
    )
    outcome, pseudo, hazard = mtp_common.continuous_learners("both_correct", edges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SDR(
            {"natural": mtp_common.NATURAL, "up": mtp_common.UP},
            reference="natural",
            outcome_learner=outcome,
            pseudo_learner=pseudo,
            treatment_learner=hazard,
            density_bins=bins,
            n_folds=3,
            g_bounds=(1e-8, 1.0),
            simultaneous=False,
            random_state=0,
        ).fit(
            sample,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        )
    pooled = fit.mechanism.policy_numerators["up"]
    for fold, (_, held) in enumerate(fit.folds):
        own = fit.fold_mechanisms[fold].mechanism.policy_numerators["up"]
        for node in (1, 2):
            np.testing.assert_array_equal(own[node][held], pooled[node][held])
            assert np.max(np.abs(own[node] - pooled[node])) == 0.0


def test_an_in_sample_fit_keeps_no_fold_mechanism(fitted: tuple[int, Any]) -> None:
    n_folds, fit = fitted
    assert len(fit.fold_mechanisms) == (0 if n_folds == 1 else n_folds)


# ------------------------------------------------------------------ T4


def test_without_binding_bounds_the_transform_is_the_one_step_recursion(frame: Any) -> None:
    r""":math:`\varphi_1 = v_1 + W_{1,1}(\varphi_2 - m_1)` on every row at risk at node 1."""
    fit = _fit(frame, 1, g_bounds=(1e-8, 1.0))
    for label in PLANS:
        first = fit.fits[label].steps[0]
        assert first.bound_binding == 0.0
        # At a label node the carried value is the prediction at the observed arm.
        assert first.marginal is None
        expected = first.initial + first.clever * (first.regression_target - first.initial)
        np.testing.assert_allclose(first.transform, expected, atol=1e-12, rtol=0)


def test_a_binding_bound_breaks_the_one_step_form(frame: Any) -> None:
    """The control: once a partial bound binds, the general sum and the one-step form part."""
    fit = _fit(frame, 1, g_bounds=(0.2, 1.0))
    first = fit.fits["always"].steps[0]
    assert first.bound_binding > 0.0
    one_step = first.initial + first.clever * (first.regression_target - first.initial)
    assert np.max(np.abs(first.transform - one_step)) > CAUGHT


# ------------------------------------------------------------------ T3


class _Shifted(BaseEstimator):
    """Cell means plus ``b * (L2 + 0.3)`` at the node-2 design ``[W, L2]``."""

    def __init__(self, b: float = 0.0) -> None:
        self.b = b

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Shifted:
        self.base_ = law.CellMeans().fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return self.base_.predict(matrix) + self.b * (matrix[:, -1] + 0.3)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


class _Tilted(BaseEstimator):
    """Cell means divided by ``1 + a (W + 0.5)`` at the node-2 treatment design.

    The treatment factor at the arm ``1`` then shrinks by that factor, so the node-2 ratio of
    the plan ``always`` grows by it.  Node 1's design is left exact.
    """

    def __init__(self, a: float = 0.0) -> None:
        self.a = a

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Tilted:
        self.base_ = law.CellMeans().fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        p = self.base_.predict(matrix)
        if matrix.shape[1] == 3:
            p = p / (1.0 + self.a * (matrix[:, 0] + 0.5))
        return np.column_stack([1.0 - p, p])


def _remainder_error(a: float, b: float) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SDR(
            declared_regimens({"always": law.REGIMEN_SPEC["always"]}),
            outcome_learner=_Shifted(b),
            pseudo_learner=law.CellMeans(),
            treatment_learner=_Tilted(a),
            censoring_learner=law.CellMeans(),
            n_folds=1,
            g_bounds=(1e-8, 1.0),
            simultaneous=False,
        ).fit(law.frame(), **COLUMNS)
    return fit["ey_regimen[always]"].psi - law.TRUTH["ey_regimen[always]"]


def test_the_remainder_is_bilinear_in_the_ratio_and_regression_errors() -> None:
    """Lemma 3: zero when either error is zero, and ``c a b`` with one constant otherwise."""
    assert abs(_remainder_error(0.0, 0.1)) < EXACT_FLOOR
    assert abs(_remainder_error(0.2, 0.0)) < EXACT_FLOOR
    constant = _remainder_error(0.1, 0.1) / 0.01
    assert abs(constant) > 1e-3
    for a, b in ((0.2, 0.05), (0.05, -0.1), (0.3, 0.08)):
        assert _remainder_error(a, b) == pytest.approx(constant * a * b, abs=1e-10)
