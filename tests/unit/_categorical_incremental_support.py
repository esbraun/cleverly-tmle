"""Shared fits and readers for the categorical incremental tests.

The fits run on the exact three-arm laws of :mod:`tests.discrete_law_multi` (outcomes
complete) and :mod:`tests.discrete_law_mar_multi` (outcomes missing at random), with the tilts
of :mod:`tests.discrete_law_categorical_incremental`.  Oracle learners make the initial fit
exactly correct in the sample, so the reported curve is the influence function at ``P_0``;
the non-saturated learners make the targeting step do real work.  The test files import this
module and not each other.  The weak-learner fit and its exact-remainder reader live here
because both the targeting tests and the mutation controls read them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.interventions import Incremental
from tests import discrete_law_categorical_incremental as law
from tests import discrete_law_mar_multi as mar
from tests import discrete_law_multi as base
from tests.studies.mar_arm_indexed_laws import LawOutcome, LawResponse, LawTreatment

#: ``V`` is a function of ``W``, so a stratum is a union of ``W`` cells and the oracle
#: nuisances stay exact inside it.
STRATA_OF_W = np.array(law.STRATA_OF_W)

#: Weak enough that the targeting step must move, as the non-saturated witnesses need.
MAX_ITER = 200


def declared(tilts: dict[str, dict[str, float]] | None = None) -> list[Incremental]:
    """The library-side tilts that :data:`law.TILTS` describes, in the same order."""
    return [Incremental(factors, name=name) for name, factors in (tilts or law.TILTS).items()]


def frame(*, missing: bool = False, strata: bool = False) -> pd.DataFrame:
    """The sample that realises the law exactly, optionally with the stratum column ``V``."""
    data = mar.frame() if missing else base.frame()
    return data.assign(V=STRATA_OF_W[data["W"].to_numpy().astype(int)]) if strata else data


def cells(*, missing: bool = False) -> np.ndarray:
    """The support index of every row of :func:`frame`."""
    return mar.cell_of_row() if missing else base.cell_of_row()


def at_support(curve: np.ndarray, *, missing: bool = False) -> np.ndarray:
    """One value of a row-wise curve per support point, in support order.

    The curve is constant within a cell when the fit is exact, so the first row of each cell
    stands for it.
    """
    index = cells(missing=missing)
    support = mar.SUPPORT if missing else base.SUPPORT
    return np.array([np.asarray(curve)[np.flatnonzero(index == p)[0]] for p in range(len(support))])


def exact_fit(
    tilts: Sequence[Incremental] | None = None,
    *,
    missing: bool = False,
    strata: bool = False,
    known: bool = False,
    **settings: Any,
) -> Any:
    """A TMLE fit of the tilts with oracle nuisances on the exact law."""
    if missing:
        learners: dict[str, Any] = {
            "outcome_learner": LawOutcome(mar.LAW),
            "treatment_learner": LawTreatment(mar.LAW),
            "missingness_learner": LawResponse(mar.LAW),
        }
    else:
        learners = {
            "outcome_learner": base.OracleMultiOutcome(),
            "treatment_learner": base.OracleMultiTreatment(),
        }
    estimator = TMLE(
        **{
            **learners,
            "cross_fit": False,
            "incremental": list(declared() if tilts is None else tilts),
            "estimands": "all",
            "simultaneous": False,
            "random_state": 0,
            **settings,
        }
    )
    data = frame(missing=missing, strata=strata)
    roles: dict[str, Any] = {"covariates": ["W", "V"] if strata else ["W"]}
    if missing:
        roles["delta"] = "Delta"
    if strata:
        roles["strata"] = ["V"]
    if known:
        probabilities = mar.probabilities_by_label(data) if missing else _known_probabilities(data)
        roles["treatment_probabilities"] = probabilities
    return estimator.fit(data, outcome="Y", treatment="A", **roles).single()


def _known_probabilities(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Known ``g(a | W)`` of the base law as a mapping keyed by level, one array per arm."""
    w = np.rint(data["W"].to_numpy(dtype=float)).astype(int)
    return {label: base.G[w, base.LABELS.index(label)] for label in base.LABELS}


class MainTermsTreatment(BaseEstimator):
    """A main-terms multinomial logistic ``g``: deliberately not saturated in ``W``."""

    def __init__(self, columns: Sequence[int] = (0,)) -> None:
        self.columns = columns

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> MainTermsTreatment:
        design = np.asarray(X, dtype=float)[:, list(self.columns)]
        self.model_ = LogisticRegression(C=1e6, max_iter=2000).fit(
            design, y, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)[:, list(self.columns)]
        return np.asarray(self.model_.predict_proba(design))


class MainTermsOutcome(BaseEstimator):
    """A linear main-terms outcome regression on the arm block and ``W``."""

    def __init__(self, width: int = 3) -> None:
        self.width = width

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> MainTermsOutcome:
        design = np.asarray(X, dtype=float)[:, : self.width]
        self.model_ = LinearRegression().fit(design, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)[:, : self.width]
        return np.asarray(np.clip(self.model_.predict(design), 0.02, 0.98))


TOL = 1e-10
#: The solved score is relative to its largest possible size.
SCORE = 1e-8


def weak_fit(*, missing: bool = False, strata: bool = False, tilts: Any = None) -> Any:
    """A fit whose nuisances are main-terms, hence wrong, on the exact law."""
    # ``V`` is a function of ``W``: the design columns stop at ``W`` so that the stratified fit
    # is not saturated by accident.  The missing-outcome law gets the marginal shares for ``g``
    # (a wrong mechanism, as ``Qbar`` alone leaves too small a bias there to test anything).
    settings: dict[str, Any] = {
        "outcome_learner": MainTermsOutcome(width=3),
        "treatment_learner": (
            DummyClassifier(strategy="prior") if missing else MainTermsTreatment()
        ),
        "tol": TOL,
        "max_iter": MAX_ITER,
    }
    if missing:
        settings["missingness_learner"] = LawResponse(mar.LAW)
    return exact_fit(tilts, missing=missing, strata=strata, **settings)


def tables(missing: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(P(W), g, Qbar)`` of the law on its own arm axis."""
    if missing:
        return mar.P_W, mar.G, mar.Q
    return base.P_W, base.G, base.Q


def targeted(fit: Any, missing: bool) -> tuple[np.ndarray, np.ndarray]:
    """``g*`` and ``Q*`` as ``(n, K)`` arrays in the fit's arm-code order."""
    fluctuation = fit.fluctuations["ipsi"]
    g = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    q = np.column_stack([fluctuation.targeted.arms[float(c)] for c in range(g.shape[1])])
    return g, q


def remainder_at(fit: Any, tilt: dict[str, float], missing: bool, rows: np.ndarray) -> float:
    """``mean over rows of (m* - m_0)(D* - D_0)/D*`` at the targeted pair.

    The rows realise the law, so the sample mean is :math:`E_0`.  The truth :math:`(m_0, D_0)`
    is the law's, read at each row's ``W`` and reordered to the library's arm codes.
    """
    _, g0, q0 = tables(missing)
    g_star, q_star = targeted(fit, missing)
    w = np.rint(fit.data.covariates[:, 0]).astype(int)
    order = list(base.ARM_OF_CODE)
    f = np.asarray([tilt.get(base.SORTED_LABELS[c], 1.0) for c in range(base.K)])
    d_star = g_star @ f
    m_star = (g_star * f * q_star).sum(axis=1) / d_star
    g_true, q_true = g0[w][:, order], q0[w][:, order]
    d_true = g_true @ f
    m_true = (g_true * f * q_true).sum(axis=1) / d_true
    value = (m_star - m_true) * (d_star - d_true) / d_star
    return float(np.mean(value[rows]))
