r"""The fits and learners of the study ``categorical-incremental`` (X43).

Family P fits :class:`~cleverly.estimators.TMLE` with the tilts of
:mod:`tests.discrete_law_categorical_ipsi_point` on a three-level point treatment.  Family L
fits :class:`~cleverly.longitudinal.IPSI` with three-level tilt nodes on
:mod:`tests.discrete_law_categorical_ipsi_longitudinal`.  Every nuisance is saturated:
:class:`CellFrequencies` for the categorical mechanism and
:class:`~tests.discrete_law_longitudinal.CellMeans` for every regression.  Both are
consistent on these finite-support laws, so the declared cells test the estimator, not a
learner.

The softmax witness (``X43-plan.md`` rev 2, finding 1).  A saturated mechanism fitted in
sample already solves every mechanism score, so its fluctuation stays at zero.  Out of fold
it does not, and the softmax update moves :math:`\hat g`.  The library reports the
coefficient of the last alternation round, which is zero once the loop settles
(``X43-progress.md`` decision 7), so :func:`mechanism_movement` reads the movement itself:
the largest :math:`\lvert g^* - \hat g\rvert` over rows and levels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.estimators import TMLE
from cleverly.interventions import Incremental
from tests import discrete_law_categorical_ipsi_point as point
from tests.discrete_law_longitudinal import CellMeans
from tests.studies.intervention_study_helpers import incremental_estimates

#: The declared support of the continuous outcome.  Every outcome mean of laws P, P0, Pw, L
#: and L0 lies in [0, 5.5] and the noise is standard normal, so a draw leaves the support
#: with probability below 1e-15 per row.  A cross-fitted continuous-outcome fit needs it (the
#: shipped scale rule); the in-sample fits carry it too, so the two differ in splitting only.
Q_BOUNDS = (-8.0, 14.0)
#: The fold count of every cross-fitted fit.
N_FOLDS = 5
#: The targeting controls of every point fit.
MAX_ITER = 100
TOL = 1e-10


class CellFrequencies(ClassifierMixin, BaseEstimator):
    """The saturated categorical mechanism: the level frequencies within each design row.

    The multi-level sibling of :class:`~tests.discrete_law_longitudinal.CellMeans`.  A design
    row the fit never saw gets the training frequencies.  ``sample_weight`` is honoured.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellFrequencies:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        target = np.asarray(y)
        weights = (
            np.ones(len(target))
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        self.classes_ = np.unique(target)
        onehot = (target[:, None] == self.classes_[None, :]).astype(float) * weights[:, None]
        keys, inverse = np.unique(matrix, axis=0, return_inverse=True)
        totals = np.zeros((keys.shape[0], self.classes_.size))
        np.add.at(totals, inverse.reshape(-1), onehot)
        sizes = totals.sum(axis=1, keepdims=True)
        self.keys_ = keys
        self.frequencies_ = np.where(sizes > 0, totals / np.where(sizes > 0, sizes, 1.0), 0.0)
        self.default_ = onehot.sum(axis=0) / onehot.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        out = np.tile(self.default_, (matrix.shape[0], 1))
        for position, key in enumerate(self.keys_):
            out[np.all(matrix == key, axis=1)] = self.frequencies_[position]
        return out


def incrementals(tilts: Mapping[str, Mapping[str, float]] = point.TILTS) -> list[Incremental]:
    """The library tilts of ``tilts``, in order, each named by its key."""
    return [Incremental(dict(factors), name=name) for name, factors in tilts.items()]


def fit_point(
    frame: pd.DataFrame,
    *,
    cross_fit: bool,
    strata: bool = False,
    known: bool = False,
    outcome_learner: Any = None,
    treatment_learner: Any = None,
    tilts: Mapping[str, Mapping[str, float]] = point.TILTS,
) -> Any:
    """One point TMLE fit of the tilts, with saturated learners unless told otherwise.

    ``strata=True`` reads the stratum column ``V`` (a function of ``W``) as a baseline
    stratum.  ``known=True`` declares the law's own :math:`g(a \\mid W)` by level.
    """
    estimator = TMLE(
        incremental=incrementals(tilts),
        outcome_learner=CellMeans() if outcome_learner is None else outcome_learner,
        treatment_learner=CellFrequencies() if treatment_learner is None else treatment_learner,
        cross_fit=cross_fit,
        n_folds=N_FOLDS,
        q_bounds=Q_BOUNDS,
        simultaneous=False,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=0,
    )
    roles: dict[str, Any] = {"covariates": ["W", "V"] if strata else ["W"]}
    if strata:
        roles["strata"] = ["V"]
    if known:
        roles["treatment_probabilities"] = known_mechanism(frame)
    return estimator.fit(frame, outcome="Y", treatment="A", **roles).single()


def known_mechanism(frame: pd.DataFrame, law: point.PointLaw = point.LAW_P) -> dict[str, Any]:
    """The law's own :math:`g(a \\mid W)` at each row, keyed by level label."""
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    return {label: law.g[w, index] for index, label in enumerate(point.LABELS)}


def mechanism_movement(result: Any) -> float:
    r"""The largest :math:`\lvert g^* - \hat g\rvert` over rows and levels of a point fit."""
    targeted = np.asarray(result.fluctuations["ipsi"].mechanism.propensity, dtype=float)
    initial = np.asarray(result.nuisance.propensity.values, dtype=float)
    return float(np.max(np.abs(targeted - initial)))


def stratum_name(name: str, stratum: int) -> str:
    """The reported name of ``name`` in stratum ``V = stratum``."""
    return f"{name}[V={float(stratum)}]"


def standard_error_without(result: Any, name: str, term: np.ndarray) -> float:
    """The standard error of ``name`` after removing ``term`` from its influence curve."""
    curve = np.asarray(result[name].influence_curve, dtype=float) - term
    return float(np.std(curve, ddof=1) / np.sqrt(len(curve)))


def mechanism_terms(result: Any, name: str, *, levels: Sequence[int] | None = None) -> np.ndarray:
    r"""The mechanism term of ``name`` on each row, from the targeted pair.

    :math:`\sum_b c_b(W)\{1(A = b) - g^*_b(W)\}` with
    :math:`c_b = \delta_b(\bar Q^*_b - m^*)/D^*`, summed over ``levels`` (every level by
    default).  ``levels=(0, 1)`` is the binary-blip term of mutation M2.  A contrast's term is
    the difference of its two tilts' terms.
    """
    fluctuation = result.fluctuations["ipsi"]
    g = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    q = np.column_stack([fluctuation.targeted.arms[float(c)] for c in range(g.shape[1])])
    # The targeted arms are on the bounded scale of ``Q_BOUNDS``; the curve is not.
    q = np.asarray(result.nuisance.scaler.unscale_level(q), dtype=float)
    arms = np.asarray(result.data.treatment, dtype=float).astype(int)
    keep = range(g.shape[1]) if levels is None else levels
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")

    def one(label: str) -> np.ndarray:
        factors = np.array([point.TILTS[label].get(level, 1.0) for level in point.LABELS])
        normaliser = g @ factors
        tilted = (g * factors * q).sum(axis=1) / normaliser
        coefficient = factors[None, :] * (q - tilted[:, None]) / normaliser[:, None]
        out = np.zeros(len(arms))
        for b in keep:
            out += coefficient[:, b] * ((arms == b).astype(float) - g[:, b])
        return out

    term = one(parts[0])
    if stem == "ate_ipsi":
        term = term - one(parts[1])
    return term


def initial_estimate(result: Any, name: str, *, family: str) -> float:
    """The untargeted plug-in of ``name``: the initial fit's tilted mean, before correction."""
    if family == "P":
        return float(incremental_estimates(result)[name])
    return float(plug_in(result, name))


# --------------------------------------------------------------------------- family L

#: The treatment value of each level name at a longitudinal node.  The law codes the
#: levels 0, 1 and 2, and :class:`~cleverly.longitudinal.LongitudinalData` reports the
#: numeric column values as its ``treatment_levels``, so a tilt names its levels by those.
LEVEL_KEYS: dict[str, float] = {"a": 0.0, "b": 1.0, "c": 2.0}
#: The bounds of every longitudinal fit, the shipped default (X19's reason: a sparse training
#: cell with no row at an arm gives a cell-mean estimate of 0).
G_BOUNDS = (0.01, 1.0)


def node_tilt(factors: tuple[float, ...]) -> Incremental:
    """The estimator side of one oracle tilt node: a mapping keyed by treatment value."""
    return Incremental({LEVEL_KEYS[label]: f for label, f in zip(LEVEL_KEYS, factors, strict=True)})


def regimens(plans: Mapping[str, Sequence[tuple[str, tuple[float, ...]]]]) -> dict[str, Any]:
    """``regimens=`` of an :class:`~cleverly.longitudinal.IPSI` fit of the oracle's ``plans``."""
    return {
        label: tuple(node_tilt(factors) for _, factors in plan) for label, plan in plans.items()
    }


def fit_longitudinal(
    frame: pd.DataFrame,
    target: Any,
    *,
    n_folds: int,
    plans: Mapping[str, Any] | None = None,
    reference: str | None = point.REFERENCE,
    censoring: bool | None = None,
) -> Any:
    """One :class:`~cleverly.longitudinal.IPSI` fit with saturated learners at every slot.

    ``censoring=False`` drops the censoring model from a fit to complete cases; by default
    a fit models censoring when ``target`` has censoring axes.
    """
    from cleverly.longitudinal import IPSI
    from tests import discrete_law_categorical_ipsi_longitudinal as long_law

    censored = any(axis.kind == "censoring" for axis in target.axes)
    if censoring is None:
        censoring = censored
    declared = long_law.columns(target)
    if not censoring:
        declared.pop("censoring", None)
    estimator = IPSI(
        regimens(long_law.PLANS if plans is None else plans),
        reference=reference,
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=CellFrequencies(),
        censoring_learner=CellMeans() if censoring else None,
        n_folds=n_folds,
        learner_folds=2,
        g_bounds=G_BOUNDS,
        q_bounds=Q_BOUNDS,
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(frame, **declared)


def plug_in(result: Any, name: str) -> float:
    """The mean of the first-node regression's tilted value of an IPSI fit, unscaled."""
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")

    def first(label: str) -> float:
        return float(np.mean(result.fits[label].steps[0].marginal))

    if stem == "ey_regimen":
        return float(result.scaler.unscale_level(first(parts[0])))
    return float(result.scaler.unscale_difference(first(parts[0]) - first(parts[1])))
