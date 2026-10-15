r"""Data-tuned caps of a point-treatment modified treatment policy (X40 part (e)).

A :class:`~cleverly.interventions.Shift` or :class:`~cleverly.interventions.Scale` whose cap
is a tuner (:mod:`cleverly.interventions.tuning`) has no map until a fit reads the treatment
law.  :func:`resolve_point_policies` fits each tuner before any learner and returns the
policies :class:`~cleverly.interventions.PolicySet` evaluates, with the construction of the
longitudinal fits (``X40-plan.md`` sections 2.4 and 2.5):

.. list-table::
   :header-rows: 1

   * - ``target``
     - the map the fit evaluates
     - the curve
   * - ``"population"``
     - the map at :math:`\theta_n`, the tuner fitted on every row.  On the classifier route
       with ``n_folds > 1``, fold ``k``'s classifier stacks the map at :math:`\theta^{(-k)}`,
       the tuner fitted on its training rows (N3); the density route reads its fold density
       under the map at :math:`\theta_n`.
     - the curve at :math:`\theta_n` plus :math:`\partial\Phi_n' \mathrm{IF}` (the chain rule)
   * - ``"fold_average"``
     - one component per outer fold ``k``, the map at :math:`\theta^{(-k)}`, read on fold
       ``k``'s held-out rows only, with its own fluctuation coefficient
     - :math:`n w_i (D_i - \hat\psi_k) / (V W_k)` on fold ``k``'s rows (Hubbard,
       Kherad-Pajouh and van der Laan 2016, Theorem 1)

A point treatment has one node, so every row is at risk and :math:`W_{1:0} = 1` in the
derivative :math:`\partial\Phi/\partial\theta = E[g(e \mid W) |e'| \{\bar Q(c, W) - \bar Q(e,
W)\} \partial c/\partial\theta]`.  The kernel estimate reads the cross-fitted initial outcome
regression at the boundary :math:`c(W)` and at the last moved dose :math:`e(W)`.  Only its
consistency enters the curve, and the targeted update moves the regression by
:math:`O_P(n^{-1/2})`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import CapabilityError
from ..interventions.policy import (
    Scale,
    Shift,
    describe_policy,
    lazy_frame,
    resolve_tuned_policy,
    tuned_cap_parts,
    tuned_policy,
)
from ..interventions.tuning import (
    FOLD_AVERAGE_ONE_FOLD,
    DoseQuantile,
    FittedTuner,
    SupportBoundary,
    cap_derivative,
    fit_tuner,
    refuse_outcome_tuners,
    tuned_cap_at_atoms,
    tuner_influence,
)

if TYPE_CHECKING:
    from ..data import CausalData
    from ..interventions import PolicySet
    from ..learners.crossfit import Folds

__all__ = [
    "FOLD_AVERAGE_SUFFIX",
    "PointTunedPolicies",
    "PointTuning",
    "declares_tuned",
    "in_sample_policies",
    "refuse_tuned_compositions",
    "resolve_point_policies",
]

#: What a fold-average policy appends to its name in every reported parameter.
FOLD_AVERAGE_SUFFIX = ", fold average"


def declares_tuned(policies: Sequence[object]) -> bool:
    """Whether any declared policy has a data-tuned cap.

    Parameters
    ----------
    policies : sequence of object
        The declared policies.

    Returns
    -------
    bool
        ``True`` when some policy is a tuned :class:`~cleverly.interventions.Shift` or
        :class:`~cleverly.interventions.Scale`.
    """
    return any(tuned_policy(policy) for policy in policies)


def _composition(reason: str, remedy: str) -> CapabilityError:
    return CapabilityError(
        f"a point-treatment policy with a data-tuned cap is not combined with {reason}. {remedy}"
    )


def refuse_tuned_compositions(estimator: Any, data: CausalData) -> None:
    """Refuse every composition the point tuned-cap construction does not cover.

    Parameters
    ----------
    estimator : TMLE
        The point-treatment estimator.
    data : CausalData
        The validated data.

    Raises
    ------
    CapabilityError
        R-e3 for a tuner that reads the outcome.  For ``cv_evaluation=True``,
        ``targeting_scheme="fold"``, baseline strata, a missing treatment or outcome, an
        intermediate, a treatment that is not one continuous dose, declared dose atoms, and, on
        the fold-average target, the bootstrap and repeated splits.
    """
    refuse_outcome_tuners(
        [
            (str(policy.name), cast(DoseQuantile | SupportBoundary, policy.cap))
            for policy in estimator.policies
            if tuned_policy(policy)
        ],
        (str(data.outcome_name),),
    )
    if estimator.cv_evaluation:
        raise _composition(
            "cv_evaluation=True",
            "The fold-evaluated estimate averages fold estimates of one map, and a tuned "
            "map differs by fold. Use cv_evaluation=False; target='fold_average' already "
            "targets each fold's map on that fold's held-out rows",
        )
    if estimator.targeting_scheme == "fold":
        raise _composition(
            "targeting_scheme='fold'",
            "Use the pooled scheme; target='fold_average' fits one fluctuation per fold",
        )
    if data.has_strata:
        raise _composition(
            "baseline strata",
            "A stratum mean would need the tuner's influence function within each stratum. "
            "Fit without strata=",
        )
    if data.has_missing_treatment or data.has_missing_outcome:
        raise _composition(
            "a missing treatment or outcome",
            "The tuner reads the dose of every row, and the derivative term is derived "
            "for complete data. Fit the complete rows",
        )
    if data.has_intermediate:
        raise _composition("an intermediate", "Fit the policy without intermediate=")
    if not data.is_continuous_treatment or data.is_vector_treatment:
        raise _composition(
            "a treatment that is not one continuous dose",
            "A tuned cap reads a quantile of the dose law, which has a density only at a "
            "continuous dose",
        )
    if data.dose_atoms:
        raise CapabilityError(tuned_cap_at_atoms("the treatment", data.dose_atoms))
    fold_average = any(
        tuned_policy(policy) and getattr(policy, "target", None) == "fold_average"
        for policy in estimator.policies
    )
    if fold_average and estimator.n_bootstrap:
        raise _composition(
            "the bootstrap on the fold-average target",
            "The fold-average parameter depends on the split, so a resample redefines it. "
            "Set n_bootstrap=0, or declare target='population' with a finite-dimensional "
            "tuner",
        )
    if fold_average and estimator.repeats > 1:
        raise _composition(
            "repeated splits on the fold-average target",
            "Each split defines its own fold-average parameter, so a median over splits "
            "estimates none of them. Set repeats=1",
        )


@dataclass(frozen=True)
class PointTuning:
    """One population-target tuned policy: its tuner fitted on every row, and its curve parts.

    Parameters
    ----------
    index : int
        The policy's position among the declared policies.
    policy : Shift or Scale
        The declared tuned map.
    fitted : FittedTuner
        The tuner fitted on every row: :math:`\\theta_n`.
    influence : ndarray
        ``(n, p)`` the tuner's influence function.
    boundary : ndarray
        ``(n,)`` :math:`c(W)` at :math:`\\theta_n`.
    edge : ndarray
        ``(n,)`` :math:`e(W)`, the last dose the cap lets move.
    gradient : ndarray
        ``(n, p)`` :math:`\\partial c / \\partial\\theta`.
    slope : float
        :math:`|e'|`.
    """

    index: int
    policy: Any
    fitted: FittedTuner
    influence: FloatArray
    boundary: FloatArray
    edge: FloatArray
    gradient: FloatArray
    slope: float

    @property
    def keys(self) -> tuple[str, str]:
        """The probe keys of the boundary and of the last moved dose."""
        return f"tuned{self.index}:boundary", f"tuned{self.index}:edge"


@dataclass(frozen=True)
class PointTunedPolicies:
    """The declared policies of a point fit, with every tuned cap resolved.

    Parameters
    ----------
    declared : tuple of object
        The declared policies.
    expanded : tuple of object
        The policies :class:`~cleverly.interventions.PolicySet` evaluates: a declared
        policy as given, a population-target policy at :math:`\\theta_n`, and one policy per
        outer fold for a fold-average target.
    groups : tuple of tuple of int
        Per declared policy, its positions in ``expanded``.
    labels : tuple of str
        Per declared policy, its reported name.
    rows : ndarray
        ``(n, len(expanded))`` the rows each expanded policy reads.
    by_fold : tuple of tuple of object or None
        Per outer fold, the expanded policies that fold's classifier stacks, or ``None``
        when no population-target policy is cross-fitted.
    tunings : tuple of PointTuning
        The population-target policies.
    fold_fits : dict of int to tuple of FittedTuner
        Per tuned declared policy, the tuner fitted on each outer fold's training rows.
    """

    declared: tuple[object, ...]
    expanded: tuple[object, ...]
    groups: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    rows: BoolArray
    by_fold: tuple[tuple[object, ...], ...] | None
    tunings: tuple[PointTuning, ...]
    fold_fits: Mapping[int, tuple[FittedTuner, ...]]

    @property
    def probe_doses(self) -> dict[str, FloatArray]:
        """The doses the outcome regression is predicted at for the derivative terms."""
        out: dict[str, FloatArray] = {}
        for tuning in self.tunings:
            high, low = tuning.keys
            out[high] = tuning.boundary
            out[low] = tuning.edge
        return out

    def finish(
        self,
        evaluated: PolicySet,
        probes: Mapping[str, FloatArray],
        data: CausalData,
        reference: str | None,
    ) -> PolicySet:
        """The policy set of the declared policies, from the evaluated expanded one.

        Parameters
        ----------
        evaluated : PolicySet
            The expanded policies, evaluated on the data and the fitted mechanism.
        probes : mapping of str to ndarray
            The outcome regression at :attr:`probe_doses`, on the scaled outcome scale.
        data : CausalData
            The validated data.
        reference : str or None
            The declared name of the reference policy, or ``None`` for the first.

        Returns
        -------
        PolicySet
            One policy per declared policy, with the fold restriction and the chain-rule
            terms.
        """
        from ..exceptions import DataError

        weights = evaluated.weights
        owner = np.argmax(weights > 0.0, axis=0)
        rows = self.rows[:, owner]
        mixing = np.vstack([weights[list(group)].mean(axis=0) for group in self.groups])
        ratio = np.asarray(evaluated.ratio, dtype=float) * rows
        ratio_at = np.asarray(evaluated.ratio_at, dtype=float) * rows[:, None, :]
        terms = np.zeros((data.n, len(self.declared)))
        w = np.asarray(data.weights, dtype=float)
        dose = np.asarray(data.treatment, dtype=float).reshape(-1)
        everyone = np.ones(data.n, dtype=bool)
        for tuning in self.tunings:
            high, low = tuning.keys
            derivative = cap_derivative(
                weights=w,
                prefix=np.ones(data.n),
                dose=dose,
                edge=tuning.edge,
                high=np.asarray(probes[high], dtype=float),
                low=np.asarray(probes[low], dtype=float),
                gradient=tuning.gradient,
                slope=tuning.slope,
                at_risk=everyone,
                rule=tuning.fitted.tuner.bandwidth,
            )
            terms[:, tuning.index] = w * (tuning.influence @ derivative)
        names = tuple(str(getattr(policy, "name", "")) for policy in self.declared)
        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(f"reference={reference!r} is not one of the policies {list(names)}")
            code = float(names.index(reference))
        identity = mixing.shape[0] == mixing.shape[1] and np.array_equal(
            mixing, np.eye(mixing.shape[0])
        )
        return replace(
            evaluated,
            names=self.labels,
            descriptions=tuple(describe_policy(policy) for policy in self.declared),
            ratio=ratio,
            ratio_at=ratio_at,
            reference=code,
            mixing=None if identity else mixing,
            component_rows=None if bool(rows.all()) else np.asarray(rows, dtype=bool),
            curve_terms=terms if self.tunings else None,
        )


def _frame(data: CausalData) -> Any:
    return lazy_frame(
        lambda: data.frame_like(
            {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        )
    )


def _named(policy: object, name: str) -> object:
    return replace(cast(Any, policy), name=name)


def in_sample_policies(policies: Sequence[object], data: CausalData) -> tuple[object, ...]:
    """Every tuned cap resolved at the tuner fitted on every row, for the pre-fit checks.

    Parameters
    ----------
    policies : sequence of object
        The declared policies.
    data : CausalData
        The validated data, with one continuous dose.

    Returns
    -------
    tuple of object
        The policies, each tuned one at :math:`\\theta_n`.
    """
    frame = _frame(data)
    dose = np.asarray(data.treatment, dtype=float).reshape(-1)
    everyone = np.ones(data.n, dtype=bool)
    out: list[object] = []
    for policy in policies:
        if not tuned_policy(policy):
            out.append(policy)
            continue
        tuned = cast(Shift | Scale, policy)
        tuner = cast(DoseQuantile | SupportBoundary, tuned.cap)
        out.append(
            resolve_tuned_policy(tuned, fit_tuner(tuner, dose, frame, everyone, data.weights))
        )
    return tuple(out)


def resolve_point_policies(
    policies: Sequence[object], data: CausalData, folds: Folds
) -> PointTunedPolicies | None:
    """Fit every tuned cap before any learner, and expand the policies a fit evaluates.

    Parameters
    ----------
    policies : sequence of object
        The declared policies.
    data : CausalData
        The validated data, with one continuous dose.
    folds : Folds
        The outer split of the nuisance fits.

    Returns
    -------
    PointTunedPolicies or None
        ``None`` when no policy has a tuned cap.

    Raises
    ------
    CapabilityError
        R-e3 for a tuner that reads the outcome, and R-e4 for the fold-average target on
        one fold.
    """
    declared = tuple(policies)
    if not declares_tuned(declared):
        return None
    tuned = [
        (index, cast(Shift | Scale, policy))
        for index, policy in enumerate(declared)
        if tuned_policy(policy)
    ]
    refuse_outcome_tuners(
        [(policy.name, cast(DoseQuantile | SupportBoundary, policy.cap)) for _, policy in tuned],
        (str(data.outcome_name),),
    )
    if any(policy.target == "fold_average" for _, policy in tuned) and folds.is_single:
        raise CapabilityError(FOLD_AVERAGE_ONE_FOLD)
    frame = _frame(data)
    dose = np.asarray(data.treatment, dtype=float).reshape(-1)
    n = data.n
    everyone = np.ones(n, dtype=bool)
    owner = np.asarray(folds.assignment)
    trains: list[BoolArray] = []
    if not folds.is_single:
        for train, _ in folds:
            rows = np.zeros(n, dtype=bool)
            rows[train] = True
            trains.append(rows)
    expanded: list[object] = []
    groups: list[tuple[int, ...]] = []
    labels: list[str] = []
    masks: list[BoolArray] = []
    by_fold: list[list[object]] = [[] for _ in trains]
    tunings: list[PointTuning] = []
    fold_fits: dict[int, tuple[FittedTuner, ...]] = {}
    for index, policy in enumerate(declared):
        name = str(getattr(policy, "name", ""))
        if not tuned_policy(policy):
            groups.append((len(expanded),))
            labels.append(name)
            expanded.append(policy)
            masks.append(everyone)
            for fold in by_fold:
                fold.append(policy)
            continue
        policy = cast(Shift | Scale, policy)
        tuner = cast(DoseQuantile | SupportBoundary, policy.cap)
        fitted = fit_tuner(tuner, dose, frame, everyone, data.weights)
        fits = tuple(fit_tuner(tuner, dose, frame, rows, data.weights) for rows in trains)
        fold_fits[index] = fits
        if policy.target == "fold_average":
            start = len(expanded)
            for k, fold_fit in enumerate(fits):
                component = _named(resolve_tuned_policy(policy, fold_fit), f"{name} [fold {k + 1}]")
                expanded.append(component)
                masks.append(owner == k)
                for fold in by_fold:
                    fold.append(component)
            groups.append(tuple(range(start, len(expanded))))
            labels.append(f"{name}{FOLD_AVERAGE_SUFFIX}")
            continue
        groups.append((len(expanded),))
        labels.append(name)
        expanded.append(resolve_tuned_policy(policy, fitted))
        masks.append(everyone)
        for fold, fold_fit in zip(by_fold, fits, strict=True):
            fold.append(resolve_tuned_policy(policy, fold_fit))
        boundary, edge, gradient, slope = tuned_cap_parts(policy, fitted, frame(), n)
        tunings.append(
            PointTuning(
                index=index,
                policy=policy,
                fitted=fitted,
                influence=tuner_influence(fitted, dose, frame, everyone, data.weights),
                boundary=boundary,
                edge=edge,
                gradient=gradient,
                slope=slope,
            )
        )
    population_folds = bool(trains) and bool(tunings)
    return PointTunedPolicies(
        declared=declared,
        expanded=tuple(expanded),
        groups=tuple(groups),
        labels=tuple(labels),
        rows=np.column_stack(masks),
        by_fold=tuple(tuple(fold) for fold in by_fold) if population_folds else None,
        tunings=tuple(tunings),
        fold_fits=fold_fits,
    )
