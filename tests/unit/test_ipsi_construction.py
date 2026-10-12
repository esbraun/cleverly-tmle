r"""T7, T8, T9: the IPSI recursion against three independent constructions.

T7 compares the engine with :func:`tests.longitudinal_ipsi.longhand_ipsi`, a numpy and
scikit-learn transcription of Kennedy's Algorithm 1 with Kim's dropout weights.  The
nuisances are a logistic treatment and censoring mechanism and a linear regression, none
saturated.  So :math:`P_n` of the mechanism term is not zero, and the comparison checks the
term's contribution to the estimate as well as to the curve.  Two constructions the plan
rejects must move the estimate by more than ``1e-4``: the stitched node-1 target (M9), and the
average of fold means in place of the pooled mean on unequal folds (M10).

T8 transcribes ``npcausal::ipsi`` (version 0.1.0, ``R/ipsi.R`` lines 150 to 202) on an
uncensored law:
:math:`\varphi = W_T Y + \sum_t W_t v_t \bar m_t`, with
:math:`v_t = (1 - \delta)\{A_t(1 - \pi_t) - (1 - A_t)\delta\pi_t\}/\delta`.  T9 transcribes
Kim, Kennedy and Naimi (2021, Appendix A, Algorithm 1, steps 3 to 5) on a censored law:
:math:`\varphi = W_T Y R_T + \sum_t W_t M_t`, with the step 4b augmentation :math:`M_t`.  Both
read the nuisances the engine reports, so each checks the engine's algebra of
:math:`\varphi` against a published form of it.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.longitudinal import IPSI

from .. import discrete_law_longitudinal as binary
from .. import discrete_law_longitudinal_ipsi as law
from ..longitudinal_ipsi import COLUMNS, longhand_ipsi, regimens
from .test_ipsi_exact_laws import THREE_COLUMNS

EXACT = 1e-12
CAUGHT = 1e-4
LOGISTIC = LogisticRegression(max_iter=1000)
LINEAR = LinearRegression()
PLANS = {"natural": law.PLANS["natural"], "per_node": law.PLANS["per_node"]}
DELTAS = {"natural": (1.0, 1.0), "per_node": (2.0, 0.5)}
#: The iid sample the cross-fitted checks read, drawn from the tilt law.
N, SEED = 600, 3


def _fit(frame: Any, plans: Any, n_folds: int, columns: Any = COLUMNS, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": LINEAR,
        "pseudo_learner": LINEAR,
        "treatment_learner": LOGISTIC,
        "censoring_learner": LOGISTIC if "censoring" in columns else None,
        "n_folds": n_folds,
        "random_state": 4,
        "simultaneous": False,
    }
    settings.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IPSI(regimens(plans), **settings).fit(frame, **columns)


def _sample(probs_frame: Any, seed: int = SEED, n: int = N) -> Any:
    return probs_frame.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module")
def frame() -> Any:
    return _sample(law.tilt_frame())


@pytest.fixture(scope="module", params=[1, 5])
def fitted(request: Any, frame: Any) -> tuple[int, Any]:
    return request.param, _fit(frame, PLANS, request.param)


def _assignment(fit: Any, n: int) -> np.ndarray:
    return np.zeros(n, dtype=int) if fit.config.n_folds == 1 else fit.folds.assignment


# ------------------------------------------------------------------ T7


@pytest.mark.parametrize("label", sorted(PLANS))
def test_the_engine_is_the_longhand_recursion(
    fitted: tuple[int, Any], frame: Any, label: str
) -> None:
    """Estimate, curve and each held-out row's node-1 target, in sample and at five folds."""
    _, fit = fitted
    longhand = longhand_ipsi(
        frame,
        DELTAS[label],
        _assignment(fit, len(frame)),
        treatment_learner=LOGISTIC,
        censoring_learner=LOGISTIC,
        outcome_learner=LINEAR,
    )
    name = f"ey_regimen[{label}]"
    assert fit.psi(name) == pytest.approx(longhand.psi, abs=EXACT)
    np.testing.assert_allclose(
        fit.influence_curves[name], longhand.phi - longhand.psi, atol=EXACT, rtol=0
    )
    observed = frame["C1"].to_numpy() == 1.0
    first = fit.fits[label].steps[0]
    np.testing.assert_allclose(
        first.pseudo_outcome[observed], longhand.target[observed], atol=EXACT, rtol=0
    )
    np.testing.assert_allclose(
        fit.fits[label].mechanism_term, longhand.mechanism_term, atol=EXACT, rtol=0
    )


def test_the_mechanism_term_moves_the_cross_fitted_estimate(frame: Any) -> None:
    """The estimate-side witness: at five folds :math:`P_n` of the mechanism term is not 0.

    Measured on this sample: ``7.3e-4`` under ``natural`` and ``1.7e-3`` under ``per_node``.
    """
    fit = _fit(frame, PLANS, 5)
    assert abs(float(np.mean(fit.fits["per_node"].mechanism_term))) > 1e-3
    assert abs(float(np.mean(fit.fits["natural"].mechanism_term))) > CAUGHT


def test_the_stitched_target_is_a_different_estimate(frame: Any) -> None:
    """M9: a node-1 target from the out-of-fold node-2 fit moves the estimate."""
    fit = _fit(frame, PLANS, 5)
    for label in PLANS:
        stitched = longhand_ipsi(
            frame,
            DELTAS[label],
            fit.folds.assignment,
            treatment_learner=LOGISTIC,
            censoring_learner=LOGISTIC,
            outcome_learner=LINEAR,
            construction="stitched",
        )
        assert abs(fit.psi(f"ey_regimen[{label}]") - stitched.psi) > CAUGHT


def test_the_estimate_pools_unequal_folds() -> None:
    """M10: on unequal cluster folds the pooled mean is not the average of fold means.

    The folds hold 154 to 200 rows.  Measured gaps: ``1.9e-3`` and ``8.3e-4``.
    """
    frame = _sample(law.tilt_frame(), seed=7, n=900)
    sizes = np.random.default_rng(11).integers(1, 30, size=len(frame))
    frame["cluster"] = np.repeat(np.arange(len(frame)), sizes)[: len(frame)]
    fit = _fit(frame, PLANS, 5, columns={**COLUMNS, "id": "cluster"})
    assignment = fit.folds.assignment
    counts = np.bincount(assignment)
    assert counts.max() - counts.min() > 40
    for label in PLANS:
        phi = fit.fits[label].steps[0].transform
        pooled = float(np.mean(phi))
        averaged = float(np.mean([np.mean(phi[assignment == k]) for k in np.unique(assignment)]))
        assert fit.psi(f"ey_regimen[{label}]") == pytest.approx(pooled, abs=EXACT)
        assert abs(pooled - averaged) > CAUGHT


# ------------------------------------------------------------------ T8 npcausal


def _propensity(regimen_fit: Any, time: int, delta: float) -> np.ndarray:
    """:math:`\\pi_t` from the reported tilt density, :math:`q = \\delta\\pi/(\\delta\\pi + 1 - \\pi)`."""
    q = np.asarray(regimen_fit.policy[time - 1], dtype=float)[:, 1]
    return q / (delta * (1.0 - q) + q)


def _npcausal_phi(fit: Any, frame: Any, label: str, deltas: tuple[float, ...]) -> np.ndarray:
    """``ifvals`` of ``npcausal::ipsi`` at the engine's nuisances (``R/ipsi.R`` 150 to 202)."""
    regimen_fit = fit.fits[label]
    y = frame["Y"].to_numpy(dtype=float)
    cumwt = np.ones(len(frame))
    total = np.zeros(len(frame))
    for time, delta in enumerate(deltas, start=1):
        a = frame[f"A{time}"].to_numpy(dtype=float)
        pi = _propensity(regimen_fit, time, delta)
        wt = (delta * a + 1.0 - a) / (delta * pi + 1.0 - pi)
        cumwt = cumwt * wt
        vt = (1.0 - delta) * (a * (1.0 - pi) - (1.0 - a) * delta * pi) / delta
        rt = regimen_fit.steps[time - 1].marginal
        total = total + cumwt * vt * rt
    return cumwt * y + total


THREE_DELTAS = {"natural": (1.0,) * 3, "half": (0.5,) * 3, "double": (2.0,) * 3}


@pytest.mark.parametrize("n_folds", [1, 5])
def test_the_curve_is_npcausals_influence_function(n_folds: int) -> None:
    """T8 on the three-node law: exact in sample, and cross-fitted on a sample."""
    plans = {label: law.THREE_PLANS[label] for label in THREE_DELTAS}
    if n_folds == 1:
        frame = law.three_frame()
        fit = _fit(
            frame,
            plans,
            1,
            THREE_COLUMNS,
            outcome_learner=binary.CellMeans(),
            pseudo_learner=binary.CellMeans(),
            treatment_learner=binary.CellMeans(),
            g_bounds=(1e-8, 1.0),
        )
    else:
        frame = _sample(law.three_frame(), seed=5, n=800)
        fit = _fit(frame, plans, 5, THREE_COLUMNS)
    for label, deltas in THREE_DELTAS.items():
        phi = fit.fits[label].steps[0].transform
        np.testing.assert_allclose(
            _npcausal_phi(fit, frame, label, deltas), phi, atol=EXACT, rtol=0, err_msg=label
        )


# ------------------------------------------------------------------ T9 Kim


def _kim_phi(fit: Any, frame: Any, label: str, deltas: tuple[float, float]) -> np.ndarray:
    """Kim et al. (2021) Algorithm 1, steps 3 to 5, at the engine's nuisances.

    Kim's :math:`R_t` is this package's :math:`C_t`, and :math:`\\omega_t` is the retention
    probability at node :math:`t`, read off the reported cumulative censoring product.
    """
    regimen_fit = fit.fits[label]
    n = len(frame)
    observed = [np.ones(n, dtype=bool)]
    cumulative = np.asarray(regimen_fit.cumulative_unbounded, dtype=float)
    cumw = np.ones(n)
    total = np.zeros(n)
    previous = np.ones(n)
    for time, delta in enumerate(deltas, start=1):
        rows = observed[-1]
        a = np.nan_to_num(frame[f"A{time}"].to_numpy(dtype=float))
        r = np.nan_to_num(frame[f"C{time}"].to_numpy(dtype=float))
        omega = np.where(rows, cumulative[:, time - 1] / previous, 1.0)
        previous = np.where(rows, cumulative[:, time - 1], previous)
        pi = _propensity(regimen_fit, time, delta)
        step = regimen_fit.steps[time - 1]
        m1 = np.where(rows, step.initial_by_arm[:, 1], 0.0)
        m0 = np.where(rows, step.initial_by_arm[:, 0], 0.0)
        ratio = (delta * a + 1.0 - a) / (delta * pi + 1.0 - pi)
        cumw = np.where(rows, cumw * ratio / omega, 0.0)
        augmentation = (
            (m1 - m0) * delta * (a - pi) * omega / (delta * pi + 1.0 - pi)
            + delta * m1 * (pi * omega - a * r)
            + m0 * ((1.0 - pi) * omega - (1.0 - a) * r)
        ) / (delta * a + 1.0 - a)
        total = total + np.where(rows, cumw * augmentation, 0.0)
        observed.append(rows & (r == 1.0))
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    return np.where(observed[-1], cumw * y, 0.0) + total


@pytest.mark.parametrize("n_folds", [1, 5])
def test_the_curve_is_kims_influence_function(n_folds: int) -> None:
    """T9 on the censored tilt law: exact in sample, and cross-fitted on a sample."""
    if n_folds == 1:
        frame = law.tilt_frame()
        fit = _fit(
            frame,
            PLANS,
            1,
            outcome_learner=binary.CellMeans(),
            pseudo_learner=binary.CellMeans(),
            treatment_learner=binary.CellMeans(),
            censoring_learner=binary.CellMeans(),
            g_bounds=(1e-8, 1.0),
        )
    else:
        frame = _sample(law.tilt_frame())
        fit = _fit(frame, PLANS, 5)
    for label, deltas in DELTAS.items():
        phi = fit.fits[label].steps[0].transform
        np.testing.assert_allclose(
            _kim_phi(fit, frame, label, deltas), phi, atol=1e-10, rtol=0, err_msg=label
        )
