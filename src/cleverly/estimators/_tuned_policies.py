r"""Data-tuned caps of a point-treatment modified treatment policy (X40 part (e)).

A :class:`~cleverly.interventions.Shift` or :class:`~cleverly.interventions.Scale` whose cap
is a tuner (:mod:`cleverly.interventions.tuning`) has no map until a fit reads the treatment
law.  :func:`resolve_point_policies` fits each tuner before any learner and returns the
policies :class:`~cleverly.interventions.PolicySet` evaluates, with the construction of the
longitudinal fits (``tests/unit/test_point_tuned_caps.py`` and the technical reference, "Modified
treatment policies"):

.. list-table::
   :header-rows: 1

   * - ``target``
     - the map the fit evaluates
     - the curve
   * - ``"population"``
     - the map at :math:`\theta_n`, the tuner fitted on every row.  On the classifier route
       with ``n_folds > 1``, fold ``k``'s classifier stacks the map at :math:`\theta^{(-k)}`,
       the tuner fitted on its training rows; the density route reads its fold density
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

At a vector treatment a :class:`~cleverly.interventions.JointPolicy` may tune the cap of a
dose component, as at a mixed node of ``LTMLE``.  The tuner reads what the component's map
reads, and each branch that applies the map adds its own term, weighted by the branch
probability and read at the vector the branch assigns with the component set
(``tests/unit/test_point_tuned_components.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import CapabilityError, DataError
from ..interventions.policy import (
    ForbiddenRead,
    JointComponent,
    JointPolicy,
    component_source,
    describe_policy,
    joint_assignments,
    lazy_frame,
    resolve_tuned_policy,
    tuned_cap_parts,
    tuned_maps,
    with_resolved_component,
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
        :class:`~cleverly.interventions.Scale`, or a
        :class:`~cleverly.interventions.JointPolicy` with a tuned component.
    """
    return any(tuned_maps(policy) for policy in policies)


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
        intermediate, a treatment that is not one continuous dose or a vector with the tuned
        component continuous, declared atoms on the tuned dose, two targets in one joint
        policy, and, on the fold-average target, the bootstrap and repeated splits.
    """
    refuse_outcome_tuners(
        [
            (str(tuned.name), cast(DoseQuantile | SupportBoundary, tuned.cap))
            for policy in estimator.policies
            for _, tuned, _ in tuned_maps(policy)
        ],
        (str(data.outcome_name),),
    )
    for policy in estimator.policies:
        _policy_target(policy)
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
    if data.treatment_kind == "vector":
        _refuse_vector_tunings(estimator.policies, data)
    elif not data.is_continuous_treatment or data.is_vector_treatment:
        raise _composition(
            "a treatment that is not one continuous dose",
            "A tuned cap reads a quantile of the dose law, which has a density only at a "
            "continuous dose",
        )
    elif data.dose_atoms:
        raise CapabilityError(tuned_cap_at_atoms("the treatment", data.dose_atoms))
    fold_average = any(_policy_target(policy) == "fold_average" for policy in estimator.policies)
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


def _refuse_vector_tunings(policies: Sequence[object], data: CausalData) -> None:
    """Refuse a tuned cap at a vector that is not on a continuous component without atoms."""
    components = _components(data)
    names = [component.name for component in components]
    for policy in policies:
        for component, _, _ in tuned_maps(policy):
            if component is None:
                raise _composition(
                    "a treatment that is not one continuous dose",
                    "A tuned cap reads a quantile of the dose law. At a vector treatment it "
                    "applies to a continuous component of a JointPolicy",
                )
            if component not in names or not components[names.index(component)].continuous:
                raise CapabilityError(
                    f"policy {getattr(policy, 'name', policy)!r} tunes the cap of "
                    f"{component!r}, but the continuous components of the treatment are "
                    f"{[c.name for c in components if c.continuous]}. A tuned cap reads a "
                    "quantile of a dose's law, so it applies to a continuous component"
                )
            atoms = data.component_atoms.get(component, ())
            if atoms:
                raise CapabilityError(
                    tuned_cap_at_atoms(
                        f"component {component!r} of the treatment {data.treatment_name!r}",
                        atoms,
                    )
                )


def _policy_target(policy: object) -> str | None:
    """The target of a policy's tuned maps, ``None`` without one."""
    targets = {str(tuned.target) for _, tuned, _ in tuned_maps(policy)}
    if len(targets) > 1:
        raise CapabilityError(
            f"policy {getattr(policy, 'name', policy)!r} declares the targets "
            f"{sorted(targets)} on its tuned maps. One policy is one parameter, so its tuned "
            "maps declare one target"
        )
    return next(iter(targets)) if targets else None


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
    key : str
        The stem of the probe keys.
    column : int or None
        At a vector treatment, the column of the tuned component; ``None`` at a dose.
    slots : tuple of tuple
        ``(probability, assigned)`` of each branch of the policy that applies the map, with
        ``assigned`` the ``(n, d)`` vector the branch assigns at :math:`\theta_n`, or
        ``None`` at a dose.  The curve carries one derivative term per slot, weighted by the
        branch probability.
    """

    index: int
    policy: Any
    fitted: FittedTuner
    influence: FloatArray
    boundary: FloatArray
    edge: FloatArray
    gradient: FloatArray
    slope: float
    key: str
    column: int | None = None
    slots: tuple[tuple[float, FloatArray | None], ...] = ((1.0, None),)

    def keys(self, slot: int = 0) -> tuple[str, str]:
        """The probe keys of a slot's boundary and last moved dose.

        Parameters
        ----------
        slot : int
            The position in :attr:`slots`.

        Returns
        -------
        tuple of str
            The keys of the boundary and of the last moved dose.
        """
        stem = self.key if self.column is None else f"{self.key}:{slot}"
        return f"{stem}:boundary", f"{stem}:edge"

    def probe_values(self, slot: int) -> tuple[FloatArray, FloatArray]:
        """The treatment a slot's probes set: the boundary and the last moved dose.

        At a vector treatment it is the vector the branch assigns, with the tuned component
        set.

        Parameters
        ----------
        slot : int
            The position in :attr:`slots`.

        Returns
        -------
        tuple of ndarray
            ``(n,)`` doses, or ``(n, d)`` vectors.
        """
        _, assigned = self.slots[slot]
        if assigned is None:
            return self.boundary, self.edge
        out = []
        for value in (self.boundary, self.edge):
            vector = np.array(assigned, dtype=float, copy=True)
            assert self.column is not None
            vector[:, self.column] = value
            out.append(vector)
        return out[0], out[1]


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
        The tuned maps of the population-target policies.
    """

    declared: tuple[object, ...]
    expanded: tuple[object, ...]
    groups: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    rows: BoolArray
    by_fold: tuple[tuple[object, ...], ...] | None
    tunings: tuple[PointTuning, ...]

    @property
    def probe_doses(self) -> dict[str, FloatArray]:
        """The doses the outcome regression is predicted at for the derivative terms."""
        out: dict[str, FloatArray] = {}
        for tuning in self.tunings:
            for slot in range(len(tuning.slots)):
                high, low = tuning.keys(slot)
                out[high], out[low] = tuning.probe_values(slot)
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
        weights = evaluated.weights
        owner = np.argmax(weights > 0.0, axis=0)
        rows = self.rows[:, owner]
        mixing = np.vstack([weights[list(group)].mean(axis=0) for group in self.groups])
        ratio = np.asarray(evaluated.ratio, dtype=float) * rows
        ratio_at = np.asarray(evaluated.ratio_at, dtype=float) * rows[:, None, :]
        terms = np.zeros((data.n, len(self.declared)))
        w = np.asarray(data.weights, dtype=float)
        everyone = np.ones(data.n, dtype=bool)
        for tuning in self.tunings:
            dose = (
                np.asarray(data.treatment, dtype=float).reshape(-1)
                if tuning.column is None
                else np.asarray(data.treatment_components, dtype=float)[:, tuning.column]
            )
            # A policy's value is linear in its slots' terms: each branch that applies the
            # map adds its own Leibniz term, weighted by the branch probability.
            derivative = np.zeros(tuning.gradient.shape[1])
            for slot, (probability, _) in enumerate(tuning.slots):
                high, low = tuning.keys(slot)
                derivative = derivative + probability * cap_derivative(
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
            terms[:, tuning.index] += w * (tuning.influence @ derivative)
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


def _components(data: CausalData) -> tuple[JointComponent, ...]:
    return tuple(
        JointComponent(name, levels)
        for name, levels in zip(data.component_names, data.component_levels, strict=True)
    )


def _frame_with(data: CausalData) -> Any:
    def frame_with(extra: Mapping[str, Any]) -> Any:
        payload = {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        payload.update(extra)
        return data.frame_like(payload)

    return frame_with


@dataclass(frozen=True)
class _Source:
    """What a tuned map reads: the natural dose, the frame of its map, and its column."""

    dose: FloatArray
    frame: Any
    column: int | None


def _source(policy: object, component: str | None, data: CausalData) -> _Source:
    """The natural values of a tuned dose, and the frame its tuner reads.

    At a dose the tuner reads the covariates.  At a vector treatment it reads what the
    component's map reads: the natural categorical components and the doses before it in
    the policy's order (:func:`~cleverly.interventions.policy.component_source`).
    """
    if component is None:
        return _Source(np.asarray(data.treatment, dtype=float).reshape(-1), _frame(data), None)
    assert isinstance(policy, JointPolicy)
    components = _components(data)
    names = [item.name for item in components]
    natural = np.asarray(data.treatment_components, dtype=float)
    where = f"treatment {data.treatment_name!r}"
    frame = component_source(policy, component, components, natural, _frame_with(data), where)
    column = names.index(component)
    return _Source(natural[:, column], frame, column)


def _fitted(
    policy: object, source: _Source, tuner: DoseQuantile | SupportBoundary, data: CausalData
) -> FittedTuner:
    """Fit a tuner on every row, and refuse one that reads what its map may not read."""
    try:
        fitted = fit_tuner(
            tuner, source.dose, source.frame, np.ones(data.n, dtype=bool), data.weights
        )
        check = getattr(source.frame, "check", None)
        if check is not None:
            check()
    except ForbiddenRead as error:
        raise DataError(
            f"the tuned cap of policy {getattr(policy, 'name', policy)!r} reads {error}, "
            "which the tuned component's map may not read. The tuner reads what the map "
            "reads: the natural categorical components and the doses before the component in "
            "the policy's order. Declare an order that puts the read dose first, or drop the "
            "feature"
        ) from error
    return fitted


def _at(policy: object, maps: Sequence[Any], fits: Sequence[FittedTuner]) -> object:
    """The policy with every tuned map resolved at its fitted tuner."""
    out = policy
    for (component, tuned, _), fitted in zip(maps, fits, strict=True):
        out = with_resolved_component(out, component, tuned, resolve_tuned_policy(tuned, fitted))
    return out


def in_sample_policies(policies: Sequence[object], data: CausalData) -> tuple[object, ...]:
    """Every tuned cap resolved at the tuner fitted on every row, for the pre-fit checks.

    Parameters
    ----------
    policies : sequence of object
        The declared policies.
    data : CausalData
        The validated data, with one continuous dose or a vector treatment.

    Returns
    -------
    tuple of object
        The policies, each tuned one at :math:`\\theta_n`.
    """
    out: list[object] = []
    for policy in policies:
        maps = tuned_maps(policy)
        fits = [
            _fitted(
                policy,
                _source(policy, component, data),
                cast(DoseQuantile | SupportBoundary, tuned.cap),
                data,
            )
            for component, tuned, _ in maps
        ]
        out.append(_at(policy, maps, fits))
    return tuple(out)


def _assigned(policy: object, data: CausalData) -> tuple[FloatArray, ...]:
    """The ``(n, d)`` vector each branch of a joint policy assigns."""
    assert isinstance(policy, JointPolicy)
    evaluated = joint_assignments(
        policy,
        _components(data),
        np.asarray(data.treatment_components, dtype=float),
        _frame_with(data),
        np.ones(data.n, dtype=bool),
        where=f"treatment {data.treatment_name!r}",
        supports=data.component_supports,
        atoms=data.component_atoms,
    )
    return tuple(np.asarray(values, dtype=float) for _, values in evaluated)


def resolve_point_policies(
    policies: Sequence[object], data: CausalData, folds: Folds
) -> PointTunedPolicies | None:
    """Fit every tuned cap before any learner, and expand the policies a fit evaluates.

    A :class:`~cleverly.interventions.JointPolicy` at a vector treatment may tune the cap
    of any dose component, in any branch of its randomizer.  Each tuned map is one tuner,
    fitted on the frame its map reads, and it carries one derivative term per branch that
    applies it.

    Parameters
    ----------
    policies : sequence of object
        The declared policies.
    data : CausalData
        The validated data, with one continuous dose or a vector treatment.
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
    refuse_outcome_tuners(
        [
            (tuned.name, cast(DoseQuantile | SupportBoundary, tuned.cap))
            for policy in declared
            for _, tuned, _ in tuned_maps(policy)
        ],
        (str(data.outcome_name),),
    )
    if any(_policy_target(policy) == "fold_average" for policy in declared) and folds.is_single:
        raise CapabilityError(FOLD_AVERAGE_ONE_FOLD)
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
    for index, policy in enumerate(declared):
        name = str(getattr(policy, "name", ""))
        maps = tuned_maps(policy)
        if not maps:
            groups.append((len(expanded),))
            labels.append(name)
            expanded.append(policy)
            masks.append(everyone)
            for fold in by_fold:
                fold.append(policy)
            continue
        tuners = [cast(DoseQuantile | SupportBoundary, tuned.cap) for _, tuned, _ in maps]
        sources = [_source(policy, component, data) for component, _, _ in maps]
        fitted = [
            _fitted(policy, source, tuner, data)
            for source, tuner in zip(sources, tuners, strict=True)
        ]
        fits = [
            tuple(
                fit_tuner(tuner, source.dose, source.frame, rows, data.weights) for rows in trains
            )
            for source, tuner in zip(sources, tuners, strict=True)
        ]
        at_fold = [_at(policy, maps, [own[k] for own in fits]) for k in range(len(trains))]
        if _policy_target(policy) == "fold_average":
            start = len(expanded)
            for k, resolved in enumerate(at_fold):
                component = _named(resolved, f"{name} [fold {k + 1}]")
                expanded.append(component)
                masks.append(owner == k)
                for fold in by_fold:
                    fold.append(component)
            groups.append(tuple(range(start, len(expanded))))
            labels.append(f"{name}{FOLD_AVERAGE_SUFFIX}")
            continue
        main = _at(policy, maps, fitted)
        groups.append((len(expanded),))
        labels.append(name)
        expanded.append(main)
        masks.append(everyone)
        for fold, resolved in zip(by_fold, at_fold, strict=True):
            fold.append(resolved)
        assigned = _assigned(main, data) if isinstance(policy, JointPolicy) else None
        for position, ((_, tuned, slots), source, fit) in enumerate(
            zip(maps, sources, fitted, strict=True)
        ):
            boundary, edge, gradient, slope = tuned_cap_parts(tuned, fit, source.frame(), n)
            tunings.append(
                PointTuning(
                    index=index,
                    policy=tuned,
                    fitted=fit,
                    influence=tuner_influence(
                        fit, source.dose, source.frame, everyone, data.weights
                    ),
                    boundary=boundary,
                    edge=edge,
                    gradient=gradient,
                    slope=slope,
                    key=f"tuned{index}" if assigned is None else f"tuned{index}:{position}",
                    column=source.column,
                    slots=tuple(
                        (probability, None if assigned is None else assigned[branch])
                        for branch, probability in slots
                    ),
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
    )
