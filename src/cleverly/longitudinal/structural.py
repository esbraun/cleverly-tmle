r"""Structural nested mean models: g-estimation of the effect of each treatment node.

:class:`StructuralNestedMean` fits the linear structural nested mean model (SNMM) of Robins
by cross-fitted g-estimation, in the Neyman orthogonal form of Lewis and Syrgkanis (2021).
The blip of node :math:`t` is the effect of the treatment at node :math:`t` on the end-of-study
outcome, with every later treatment at its reference level:

.. math::

    \gamma_t(h_t, a_t) = E[Y(\bar a_{t-1}, a_t, 0) - Y(\bar a_{t-1}, 0, 0) \mid H_t = h_t,
        A_t = a_t] = c(a_t)^\top \Psi_t \phi_t(h_t).

:math:`c(a_t)` is the coding of the level and :math:`\phi_t` is a declared finite feature map.
:math:`\Psi_t` is a matrix of constants.  This is a model and not a nonparametric target: the
coefficients are blips of the law only when the model holds.  A level or contrast of static
regimens read from the blips is regular under the model, and
:meth:`StructuralNestedMeanResult.check_model` compares it with the nonparametric
:class:`~cleverly.longitudinal.LTMLE` contrast on the same rows.

The nuisances condition on the full history :math:`H_t`.  They are the outcome mean
:math:`q_t(H_t) = E[Y \mid H_t]` and, for each :math:`j \ge t`, the product means
:math:`m_{j,t}(H_t) = E[Z_j \mid H_t]` with :math:`Z_j = \phi_j(H_j) \otimes c(A_j)`.  The
score of node :math:`t` is :math:`\tilde Z_{t,t}(\tilde Y_t - \sum_{j \ge t}
\tilde Z_{j,t}^\top \psi_j)`, and the nodes are solved from the last to the first.  A design
with a declared ``markov_order`` is refused: its restriction does not cover these means.
``docs/technical-reference/structural-dynamic-effects.md`` gives the derivation, the influence
curves and the conditions.
"""

from __future__ import annotations

import copy
import itertools
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np
from scipy import stats

from .._inference_status import (
    ADAPTIVE_BASELINE_STATUS,
    MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS,
    NO_SIMULTANEOUS_BANDS,
    NO_T_REFERENCE_BANDS,
    InferenceStatus,
    precedent_status,
    status_record,
    supplies_inference,
)
from .._support import (
    UNCLASSIFIED,
    Classification,
    Support,
    SupportKey,
    band_support,
    classify_structural_fit,
    combine_support,
    record_fit,
    support_for,
)
from .._typing import FloatArray, Learner
from ..data.validate import arm_indicators
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import (
    CapabilityError,
    DataError,
    Refusal,
    RefusalReason,
)
from ..inference.cluster import (
    cluster_inference_status,
    cluster_sums,
    fewest_clusters,
    influence_covariance,
    positive_mass_clause,
)
from ..inference.delta import Transform
from ..inference.influence import (
    ParameterEstimate,
    Scale,
    has_t_reference,
    make_estimate,
    reference_label,
    support_column,
    support_notes,
)
from ..inference.multiplier import SimultaneousBands, simultaneous_bands
from ..inference.results import (
    estimate_covariance,
    estimate_curves,
    reported_status,
    smooth_contrast,
    sole_estimate,
)
from ..learners.crossfit import Folds, _fresh_seed, check_integrity, random_partition
from ..learners.glm import GLM
from ..learners.library import _validate_learner
from ..learners.super_learner import SuperLearner, resolve_learner
from ..provenance import Provenance, fingerprint_array
from ..provenance import build as provenance_build
from ..utils.text import format_pvalue, format_table
from .data import LongitudinalData, Repeated
from .estimator import LTMLE, _reference_df, refuse_unsupported
from .regimen import DynamicRegimen, Regimen, _same_label, resolve_regimens

__all__ = [
    "SNMM_LABEL",
    "StructuralConfig",
    "StructuralDiagnostics",
    "StructuralModelCheck",
    "StructuralNestedMean",
    "StructuralNestedMeanResult",
    "snmm",
]

#: The condition that every estimate of a :class:`StructuralNestedMean` fit carries beside its
#: inference status.  The status says that the interval is an influence-curve interval.  The
#: label says what the interval is valid for.
SNMM_LABEL = "valid under the linear structural nested mean model with the declared blip features"

#: The smallest eigenvalue of a standardized diagonal block of the Jacobian, relative to one,
#: below which the fit refuses.  A block that small holds no treatment variation given the
#: history.
DEFAULT_MIN_EIGENVALUE = 1e-8

#: The ratio of the smallest eigenvalue of the model check's difference covariance to the
#: largest eigenvalue of the two contrast covariances, below which the check reports no test.
RANK_FLOOR = 1e-6

#: The name of the reference-level mean in a result.
REFERENCE_NAME = "ey_reference"

_INTERCEPT = "intercept"

#: The kinds of reported parameter.
_KINDS = ("blip", "reference", "level", "contrast")

# ---------------------------------------------------------------------------- refusals

_DYNAMIC_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "a rule, a modified treatment policy or a known stochastic plan is identified from the "
    "blips and the law of the history under the policy (Lewis and Syrgkanis 2021, supplement "
    "Lemma 6). StructuralNestedMean does not estimate that law, so it reports static levels "
    "and contrasts only.",
    alternative="Fit the regimen with LTMLE or SDR.",
)
_TILT_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "an incremental policy depends on the observed treatment law, which StructuralNestedMean "
    "does not use for a regimen value.",
    alternative="Fit IPSI(regimens=[...]).",
)
_CENSORING_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "StructuralNestedMean fits an end-of-study outcome with no censoring. A g-estimator with "
    "inverse probability of censoring weights has derivative terms for the estimated weights "
    "that this estimator does not build.",
    alternative="Fit LTMLE or SDR, which model censoring.",
)
_SURVIVAL_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "StructuralNestedMean fits one end-of-study outcome. A structural nested failure-time or "
    "cumulative model is a different model, and the blip of this estimator is a mean "
    "difference.",
    alternative="Fit LTMLE, which has survival and competing-risk paths.",
)
_REPEATED_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "StructuralNestedMean fits one end-of-study outcome and no repeated outcome.",
    alternative="Fit LTMLE, which fits a repeated outcome with one pass per horizon.",
)
_HELD_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "a design with one baseline decision held over every node has no treatment variation "
    "after node 1 for the blips of the later nodes to use.",
    alternative="Fit LTMLE or SDR, which read a held decision.",
)
#: Why a fit refuses a declared Markov order.  The restriction of
#: :func:`~cleverly.longitudinal.data.markov_restriction` names the regimen outcome regression and
#: the treatment and censoring mechanisms.  The g-estimation nuisances are other conditional
#: means, so a window on them would rest on a restriction nobody declared.
_MARKOV_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "StructuralNestedMean conditions the outcome mean E[Y | H_t] and the product means "
    "E[Z_j | H_t] on the full history. The Markov restriction of markov_order= covers the "
    "regimen outcome regressions and the treatment and censoring mechanisms only, and not "
    "these means.",
    alternative="Omit markov_order=, or fit LTMLE or SDR, whose nuisances read the window.",
)
_KNOWN_MECHANISM_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "StructuralNestedMean estimates every treatment nuisance and reads no declared mechanism.",
    alternative="Fit LTMLE, which reads a declared treatment mechanism.",
)
_MSM_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "a working model over regimens is a setting of the regimen estimators, and the blips of "
    "this estimator are the structural model.",
    alternative="Declare it where the regimens are: LTMLE(regimens, msm=MSM(...)).",
)
_COVARIATE_FEATURE_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "a blip feature reads a time-varying covariate, so a static level needs the counterfactual "
    "law of that covariate under the regimen, which this estimator does not estimate. The "
    "blips and the reference mean E[Y(0)] are reported.",
    alternative="Fit the static contrast with LTMLE or SDR, or declare blip features that read "
    "baseline columns and earlier treatments only.",
    roadmap="X72",
)


def _mixed_refusal(data: LongitudinalData) -> Refusal:
    """The refusal of a vector node with a continuous component, naming the first one."""
    time = next(t for t in range(1, data.n_times + 1) if data.node_kind(t) == "mixed")
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"node {data.treatment_names[time - 1]!r} at time {time} is a vector with a continuous "
        "component. StructuralNestedMean codes a node as one dose or one set of level "
        "indicators, and the blip of a flag with a dose has no coding here.",
        alternative="Fit LTMLE or SDR, which read a mixed node, or declare the flag and the "
        "dose as separate nodes.",
    )


def _invalid(condition: str, *, alternative: str | None = None) -> CapabilityError:
    """An :attr:`~cleverly.exceptions.RefusalReason.INVALID_DECLARATION` refusal."""
    return CapabilityError(
        condition, reason=RefusalReason.INVALID_DECLARATION, alternative=alternative
    )


# ------------------------------------------------------------------- blip feature layout


@dataclass(frozen=True)
class _Factor:
    """One factor of a blip feature column.

    Parameters
    ----------
    kind : str
        ``"baseline"``, ``"covariate"`` or ``"treatment"``.
    node : int
        ``0`` for a baseline column, else the node whose covariate block or treatment it is.
    index : int
        The column of the baseline matrix or covariate block, or the component of the
        treatment coding.
    """

    kind: str
    node: int
    index: int

    def determined_at(self, time: int) -> bool:
        """Whether the history :math:`H_t` holds this factor."""
        if self.kind == "baseline":
            return True
        if self.kind == "treatment":
            return self.node < time
        return self.node <= time


@dataclass(frozen=True)
class _Feature:
    """One column of :math:`\\phi_t`: a product of factors, with an empty product for 1."""

    name: str
    factors: tuple[_Factor, ...]


@dataclass(frozen=True)
class _Node:
    """The blip layout of one treatment node.

    Parameters
    ----------
    time : int
        The node, counted from one.
    treatment : str
        The treatment column of the node.
    continuous : bool
        Whether the node holds a dose.
    n_levels : int
        The number of levels of a categorical node, and ``0`` for a dose.
    levels : tuple of str
        The name of each coding component: the non-reference level labels, or ``"dose"``.
    features : tuple of _Feature
        The columns of :math:`\\phi_t`, the intercept first.
    """

    time: int
    treatment: str
    continuous: bool
    n_levels: int
    levels: tuple[str, ...]
    features: tuple[_Feature, ...]

    @property
    def p(self) -> int:
        """The number of blip features."""
        return len(self.features)

    @property
    def d(self) -> int:
        """The dimension of the coding of the node's treatment."""
        return len(self.levels)

    @property
    def size(self) -> int:
        """The number of coefficients of the node."""
        return self.p * self.d


def _is_dose(data: LongitudinalData, time: int) -> bool:
    """Whether node ``time`` holds one dose.  The fit refuses a mixed node before this runs."""
    return data.node_kind(time) == "continuous"


def _level_labels(data: LongitudinalData, time: int) -> tuple[object, ...]:
    """The labels of a categorical node's levels, in code order.

    The level set is read from the units that hold each level.  Each code ``0 .. K-1`` occurs
    at the node because the level set is built from the column itself.  A dense code set is
    checked rather than assumed.
    """
    codes = data.node_codes(time)
    present = data.node_present(time)
    present_codes, first = np.unique(codes[present], return_index=True)
    if not np.array_equal(present_codes, np.arange(present_codes.size)):
        raise _invalid(
            f"the treatment codes of node {time} are not dense; "
            f"the codes {present_codes.tolist()} occur"
        )
    return tuple(data.observed_labels(time)[present][first])


def _n_levels(data: LongitudinalData, time: int) -> int:
    """The number of levels of a categorical node: one more than its highest code."""
    return int(np.nanmax(data.node_codes(time))) + 1


def _treatment_components(data: LongitudinalData, time: int) -> tuple[str, ...]:
    """The name of each coding component of node ``time``'s treatment."""
    name = data.treatment_names[time - 1]
    if _is_dose(data, time):
        return (name,)
    levels = _level_labels(data, time)
    if len(levels) == 2:
        return (name,)
    return tuple(f"{name}={level}" for level in levels[1:])


def _node_levels(data: LongitudinalData, time: int) -> tuple[str, ...]:
    if _is_dose(data, time):
        return ("dose",)
    return tuple(str(level) for level in _level_labels(data, time)[1:])


def _terms(value: Any) -> list[tuple[str, ...]]:
    """The declared terms of one node: a name, or a tuple of names for a product."""
    if value is None:
        return []
    if isinstance(value, str):
        return [(value,)]
    terms: list[tuple[str, ...]] = []
    for term in value:
        if isinstance(term, str):
            terms.append((term,))
        elif isinstance(term, (tuple, list)) and term and all(isinstance(n, str) for n in term):
            terms.append(tuple(term))
        else:
            raise _invalid(
                f"a blip feature term is a column name or a tuple of column names for a "
                f"product; got {term!r}"
            )
    return terms


def _atoms(data: LongitudinalData, name: str, time: int) -> list[tuple[str, _Factor]]:
    """The columns one name stands for in the features of node ``time``."""
    for index, column in enumerate(data.baseline_names):
        if column == name:
            return [(name, _Factor("baseline", 0, index))]
    for node, block in enumerate(data.time_varying_names, start=1):
        for index, column in enumerate(block):
            if column == name:
                if node > time:
                    raise _invalid(
                        f"the blip feature {name!r} of node {time} reads a covariate "
                        f"measured at node {node}, after the treatment it modifies"
                    )
                return [(name, _Factor("covariate", node, index))]
    for node, column in enumerate(data.treatment_names, start=1):
        if column == name:
            if node >= time:
                raise _invalid(
                    f"the blip feature {name!r} of node {time} reads the treatment of node "
                    f"{node}. A blip reads the baseline columns, the covariates through node "
                    f"{time} and the treatments before node {time}"
                )
            labels = _treatment_components(data, node)
            return [(label, _Factor("treatment", node, c)) for c, label in enumerate(labels)]
    known = [
        *data.baseline_names,
        *(c for block in data.time_varying_names for c in block),
        *data.treatment_names,
    ]
    raise _invalid(
        f"the blip feature {name!r} of node {time} is not a column of the data; the columns "
        f"are {known}"
    )


def _node_features(
    data: LongitudinalData, time: int, terms: Sequence[tuple[str, ...]]
) -> tuple[_Feature, ...]:
    features = [_Feature(_INTERCEPT, ())]
    for term in terms:
        options = [_atoms(data, name, time) for name in term]
        for combination in itertools.product(*options):
            label = "*".join(part[0] for part in combination)
            features.append(_Feature(label, tuple(part[1] for part in combination)))
    names = [feature.name for feature in features]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise _invalid(f"the blip features of node {time} repeat {repeated}")
    return tuple(features)


def _parse_blip_features(
    data: LongitudinalData, declared: Mapping[Any, Any] | None
) -> tuple[_Node, ...]:
    """Read ``blip_features`` into one :class:`_Node` per treatment node.

    A key is a node counted from one or a treatment column name.  A node with no key holds
    the intercept only.
    """
    by_node: dict[int, list[tuple[str, ...]]] = {}
    for key, value in (declared or {}).items():
        if isinstance(key, str):
            if key not in data.treatment_names:
                raise _invalid(
                    f"blip_features names {key!r}, which is not a treatment column; the "
                    f"treatment columns are {list(data.treatment_names)}"
                )
            node = data.treatment_names.index(key) + 1
        elif isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            node = int(key)
            if not 1 <= node <= data.n_times:
                raise _invalid(
                    f"blip_features names node {node}; the nodes are 1 to {data.n_times}"
                )
        else:
            raise _invalid(
                f"a blip_features key is a node counted from one or a treatment column; got {key!r}"
            )
        if node in by_node:
            raise _invalid(f"blip_features declares node {node} twice")
        by_node[node] = _terms(value)
    return tuple(
        _Node(
            time=time,
            treatment=data.treatment_names[time - 1],
            continuous=_is_dose(data, time),
            n_levels=0 if _is_dose(data, time) else _n_levels(data, time),
            levels=_node_levels(data, time),
            features=_node_features(data, time, by_node.get(time, [])),
        )
        for time in range(1, data.n_times + 1)
    )


def _reads_covariate(nodes: Sequence[_Node]) -> bool:
    return any(
        factor.kind == "covariate"
        for node in nodes
        for feature in node.features
        for factor in feature.factors
    )


# ------------------------------------------------------------------------ the coding


def _encode(codes: FloatArray, n_levels: int, continuous: bool) -> FloatArray:
    """The coding :math:`c(a)` of treatment codes: ``(n, d)``.

    A dose is itself.  A categorical node is the drop-first indicators, so the reference is
    the first level in sorted order.
    """
    if continuous:
        return np.asarray(codes, dtype=float).reshape(-1, 1)
    return arm_indicators(codes, n_levels)


def _coding(data: LongitudinalData, time: int, codes: FloatArray | None = None) -> FloatArray:
    """The coding of node ``time``'s treatment, observed or at ``codes``: ``(n, d)``."""
    raw = data.node_codes(time) if codes is None else codes
    filled = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0)
    dose = _is_dose(data, time)
    return _encode(filled, 0 if dose else _n_levels(data, time), dose)


def _factor_value(
    data: LongitudinalData, factor: _Factor, overrides: Mapping[int, FloatArray] | None
) -> FloatArray:
    if factor.kind == "baseline":
        return np.asarray(data.baseline[:, factor.index], dtype=float)
    if factor.kind == "covariate":
        return np.asarray(data.time_varying[factor.node - 1][:, factor.index], dtype=float)
    codes = None if overrides is None else overrides.get(factor.node)
    return _coding(data, factor.node, codes)[:, factor.index]


def _product(
    data: LongitudinalData,
    factors: Sequence[_Factor],
    overrides: Mapping[int, FloatArray] | None = None,
) -> FloatArray:
    out = np.ones(data.n)
    for factor in factors:
        out = out * _factor_value(data, factor, overrides)
    return out


def _phi(
    data: LongitudinalData, node: _Node, overrides: Mapping[int, FloatArray] | None = None
) -> FloatArray:
    """The feature matrix :math:`\\phi_t`, ``(n, p)``, at the treatments in ``overrides``."""
    return np.column_stack([_product(data, f.factors, overrides) for f in node.features])


def _outer(phi: FloatArray, coding: FloatArray) -> FloatArray:
    """:math:`\\phi \\otimes c` with the component :math:`k d + l`: ``(n, p d)``."""
    return (phi[:, :, None] * coding[:, None, :]).reshape(phi.shape[0], -1)


def _z_observed(data: LongitudinalData, node: _Node) -> FloatArray:
    return _outer(_phi(data, node), _coding(data, node.time))


# --------------------------------------------------------------------------- nuisances


def _outcome_design(data: LongitudinalData, time: int) -> FloatArray:
    """The design of :math:`q_t`: the history :math:`H_t`."""
    return _with_constant(data.history_design(time), data.n)


def _treatment_design(data: LongitudinalData, time: int) -> FloatArray:
    """The design of the treatment-side nuisances :math:`m_{j,t}`: the history :math:`H_t`."""
    return _with_constant(data.history_design(time), data.n)


def _with_constant(design: FloatArray, n: int) -> FloatArray:
    """A design with a column, so a learner is never handed zero covariates."""
    if design.shape[1]:
        return design
    return np.ones((n, 1))


class _Learners:
    """The learners of a fit, resolved on first use for each role and task."""

    def __init__(
        self,
        outcome: Learner | None,
        treatment: Learner | None,
        *,
        learner_folds: int,
        random_state: int | None,
        regression: Learner | None = None,
    ) -> None:
        self._specs = {"outcome": outcome, "treatment": treatment, "regression": regression}
        self._folds = learner_folds
        self._seed = random_state
        self._cache: dict[tuple[str, str], Learner] = {}

    def get(self, role: str, task: str) -> Learner:
        # A regression with no learner of its own is the outcome learner's.
        if role == "regression" and self._specs["regression"] is None:
            role = "outcome"
        key = (role, task)
        if key not in self._cache:
            spec = self._specs[role]
            if spec is None:
                # A regression target is not on [0, 1], so the default ensemble clips only a
                # probability.
                self._cache[key] = SuperLearner(
                    task=task,  # type: ignore[arg-type]
                    n_folds=self._folds,
                    clip=(0.0, 1.0) if task == "classification" else None,
                    random_state=self._seed,
                    n_jobs=1,
                )
            else:
                self._cache[key] = resolve_learner(
                    spec,
                    task=task,  # type: ignore[arg-type]
                    n_folds=self._folds,
                    random_state=self._seed,
                )
        return self._cache[key]

    @property
    def used(self) -> list[Learner]:
        return list(self._cache.values())


@dataclass
class _Nuisances:
    """The out-of-fold nuisances of a fit.

    Parameters
    ----------
    q : list of ndarray
        ``q[t - 1]`` is :math:`\\hat q_t(H_t)`, ``(n,)``.
    m : dict
        ``m[(t, j)]`` is :math:`\\hat m_{j,t}(H_t)` for ``j >= t``, ``(n, p_j d_j)``.
    own : list of ndarray
        ``own[t - 1]`` is the ``(n, d_t)`` estimated class probabilities of the coding of a
        categorical node, and an empty array for a dose.
    n_constant : int
        The count of regressions with a fold whose training rows held one class.
    diagnostics : list
        The Super Learner diagnostics of every regression, in fitting order.
    """

    q: list[FloatArray]
    m: dict[tuple[int, int], FloatArray]
    own: list[FloatArray]
    n_constant: int
    diagnostics: list[Any] = field(default_factory=list)


def _is_binary(target: FloatArray) -> bool:
    return bool(np.all((target == 0.0) | (target == 1.0)))


def _regress(
    data: LongitudinalData,
    learners: _Learners,
    role: str,
    design: FloatArray,
    target: FloatArray,
    folds: Folds,
    n_jobs: int,
    sink: _Nuisances,
) -> FloatArray:
    """One out-of-fold regression of ``target`` on ``design``.

    A target in {0, 1} is a classification.  The outcome of a binary-outcome fit is one, and
    so is a treatment indicator.  Any other target is a regression.
    """
    classification = _is_binary(target)
    task = "classification" if classification else "regression"
    constant: list[int] = []
    # A classifier is the treatment learner for a treatment indicator and the outcome learner
    # for a binary outcome.  Any regression is the regression learner's, which defaults to the
    # outcome learner, because a treatment learner may be a classifier that takes no target
    # outside {0, 1}.
    predictions, diagnostics = cross_fit_predictions(
        learners.get(role if classification else "regression", task),
        design,
        target,
        data.weights,
        folds,
        task=task,  # type: ignore[arg-type]
        predict_designs={"history": design},
        groups=data.cluster,
        n_jobs=1,
        constant_folds=constant if classification else None,
    )
    sink.n_constant += len(constant)
    sink.diagnostics.extend(diagnostics)
    return np.asarray(predictions["history"], dtype=float)


def _fit_nuisances(
    data: LongitudinalData,
    nodes: Sequence[_Node],
    learners: _Learners,
    folds: Folds,
    n_jobs: int,
) -> _Nuisances:
    """Cross-fit :math:`q_t` and every :math:`m_{j,t}`, ``j >= t``, on the history :math:`H_t`.

    A feature of node ``j`` splits into the factors the history :math:`H_t` holds and the
    factors it does not hold.  The regression is of the undetermined product times the
    coding on :math:`H_t`, and the mean multiplies the determined product back in.  Regressions
    that share a target are fitted once.
    """
    sink = _Nuisances(q=[], m={}, own=[np.empty((data.n, 0)) for _ in nodes], n_constant=0)
    y = np.asarray(data.outcome, dtype=float)
    codings = {node.time: _coding(data, node.time) for node in nodes}
    for t in range(1, len(nodes) + 1):
        sink.q.append(
            _regress(data, learners, "outcome", _outcome_design(data, t), y, folds, n_jobs, sink)
        )
        design = _treatment_design(data, t)
        cache: dict[tuple[int, tuple[_Factor, ...], int], FloatArray] = {}
        for j in range(t, len(nodes) + 1):
            node = nodes[j - 1]
            mean = np.empty((data.n, node.size))
            for k, feature in enumerate(node.features):
                held = tuple(f for f in feature.factors if f.determined_at(t))
                rest = tuple(f for f in feature.factors if not f.determined_at(t))
                determined = _product(data, held)
                undetermined = _product(data, rest)
                for l in range(node.d):
                    key = (j, rest, l)
                    if key not in cache:
                        cache[key] = _regress(
                            data,
                            learners,
                            "treatment",
                            design,
                            undetermined * codings[j][:, l],
                            folds,
                            n_jobs,
                            sink,
                        )
                    mean[:, k * node.d + l] = determined * cache[key]
            sink.m[(t, j)] = mean
        if not nodes[t - 1].continuous:
            sink.own[t - 1] = np.column_stack([cache[(t, (), l)] for l in range(nodes[t - 1].d)])
    return sink


# ---------------------------------------------------------------------- second stage


@dataclass(frozen=True)
class _Stage:
    """The residuals the g-estimating equations read.

    Parameters
    ----------
    z : list of ndarray
        ``z[j - 1]`` is :math:`Z_j = \\phi_j \\otimes c(A_j)`, ``(n, P_j)``.
    y_res : list of ndarray
        ``y_res[t - 1]`` is :math:`\\tilde Y_t`.
    z_res : dict
        ``z_res[(t, j)]`` is :math:`\\tilde Z_{j,t}` for ``j >= t``.
    y : ndarray
        The outcome.
    """

    y: FloatArray
    z: list[FloatArray]
    y_res: list[FloatArray]
    z_res: dict[tuple[int, int], FloatArray]

    @property
    def sizes(self) -> list[int]:
        return [block.shape[1] for block in self.z]

    @property
    def offsets(self) -> list[int]:
        return [0, *np.cumsum(self.sizes).tolist()]


def _build_stage(data: LongitudinalData, nodes: Sequence[_Node], nuisances: _Nuisances) -> _Stage:
    y = np.asarray(data.outcome, dtype=float)
    z = [_z_observed(data, node) for node in nodes]
    y_res = [y - q for q in nuisances.q]
    z_res = {key: z[key[1] - 1] - mean for key, mean in nuisances.m.items()}
    return _Stage(y=y, z=z, y_res=y_res, z_res=z_res)


def _g_estimation(
    y_res: Sequence[FloatArray], z_res: Mapping[tuple[int, int], FloatArray], w: FloatArray
) -> list[FloatArray]:
    r"""Solve the g-estimating equations from the last node to the first.

    Node :math:`t` regresses :math:`\tilde Y_t - \sum_{j>t} \tilde Z_{j,t}^\top \psi_j` on
    :math:`\tilde Z_{t,t}` with weights ``w``.  Pure numpy on the residuals, so a test can
    perturb the weights with the nuisances held fixed.
    """
    n_times = len(y_res)
    n = w.shape[0]
    psi: list[FloatArray] = [np.empty(0)] * n_times
    for t in range(n_times, 0, -1):
        adjusted = y_res[t - 1].copy()
        for j in range(t + 1, n_times + 1):
            adjusted = adjusted - z_res[(t, j)] @ psi[j - 1]
        own = z_res[(t, t)]
        gram = (own * w[:, None]).T @ own / n
        rhs = (own * w[:, None]).T @ adjusted / n
        psi[t - 1] = np.linalg.solve(gram, rhs)
    return psi


def _jacobian(
    z_res: Mapping[tuple[int, int], FloatArray], w: FloatArray, offsets: Sequence[int]
) -> FloatArray:
    r"""The Jacobian :math:`G = E_n[-\partial s / \partial \psi]`, block upper triangular.

    Block :math:`(t, j)` is :math:`E_n[w \tilde Z_{t,t} \tilde Z_{j,t}^\top]` for
    :math:`j \ge t`: the derivative of the score of node :math:`t` in :math:`\psi_j`.
    """
    n_times = len(offsets) - 1
    n = w.shape[0]
    size = offsets[-1]
    gram = np.zeros((size, size))
    for t in range(1, n_times + 1):
        own = z_res[(t, t)]
        for j in range(t, n_times + 1):
            block = (own * w[:, None]).T @ z_res[(t, j)] / n
            gram[offsets[t - 1] : offsets[t], offsets[j - 1] : offsets[j]] = block
    return gram


def _scores(stage: _Stage, psi: Sequence[FloatArray], w: FloatArray) -> FloatArray:
    """The per-row scores of all nodes, stacked: ``(n, P)``."""
    n_times = len(stage.y_res)
    blocks = []
    for t in range(1, n_times + 1):
        residual = stage.y_res[t - 1].copy()
        for j in range(t, n_times + 1):
            residual = residual - stage.z_res[(t, j)] @ psi[j - 1]
        blocks.append(w[:, None] * stage.z_res[(t, t)] * residual[:, None])
    return np.hstack(blocks)


@dataclass(frozen=True)
class _Solution:
    """The second-stage solution on one set of residuals and weights."""

    weights: FloatArray
    psi: list[FloatArray]
    gram: FloatArray
    psi_curve: FloatArray
    mu0: float
    mu0_curve: FloatArray


def _reference_curve(
    stage: _Stage, psi: Sequence[FloatArray], w: FloatArray, psi_curve: FloatArray
) -> tuple[float, FloatArray]:
    r"""The reference mean :math:`E[Y(0)]` and its curve.

    The mean is :math:`E_n[w(Y - \sum_t Z_t^\top \psi_t)]`.  Its curve is the centered
    summand minus :math:`E_n[w Z]` times the curve of :math:`\psi`.
    """
    adjusted = stage.y.copy()
    for t, block in enumerate(stage.z):
        adjusted = adjusted - block @ psi[t]
    mu0 = float(np.mean(w * adjusted))
    mean_z = np.concatenate([np.mean(w[:, None] * block, axis=0) for block in stage.z])
    curve = w * (adjusted - mu0) - psi_curve @ mean_z
    return mu0, curve


def _solve(stage: _Stage, weights: FloatArray) -> _Solution:
    """Solve the equations and build the curves of :math:`\\psi` and of :math:`E[Y(0)]`.

    The weights are normalized to mean one inside, so a caller may pass any positive weights.
    """
    w = np.asarray(weights, dtype=float)
    w = w / np.mean(w)
    psi = _g_estimation(stage.y_res, stage.z_res, w)
    gram = _jacobian(stage.z_res, w, stage.offsets)
    scores = _scores(stage, psi, w)
    psi_curve = np.asarray(np.linalg.solve(gram, scores.T).T, dtype=np.float64)
    mu0, mu0_curve = _reference_curve(stage, psi, w, psi_curve)
    return _Solution(
        weights=w, psi=psi, gram=gram, psi_curve=psi_curve, mu0=mu0, mu0_curve=mu0_curve
    )


@dataclass(frozen=True)
class _RegimenInputs:
    """The feature rows and codings of one static regimen.

    Parameters
    ----------
    phi : list of ndarray
        ``phi[t - 1]`` is :math:`\\phi_t(X_0, \\bar a_{t-1})` for every row, ``(n, p_t)``.
    coding : list of ndarray
        ``coding[t - 1]`` is :math:`c(a_t)`, ``(d_t,)``.
    """

    phi: list[FloatArray]
    coding: list[FloatArray]


def _level_curve(solution: _Solution, inputs: _RegimenInputs) -> tuple[float, FloatArray]:
    r"""The level :math:`L(a) = \mu_0 + \sum_t c(a_t)^\top \Psi_t \bar\phi_t(a)` and its curve.

    The curve is the curve of :math:`\mu_0`, plus :math:`w(\phi_t - \bar\phi_t)` times the
    blip :math:`\Psi_t c(a_t)`, plus :math:`g(a)` times the curve of :math:`\psi`, where
    :math:`g_t(a) = \bar\phi_t(a) \otimes c(a_t)`.
    """
    w = solution.weights
    value = solution.mu0
    curve = solution.mu0_curve.copy()
    gradient = []
    for t, phi in enumerate(inputs.phi):
        coding = inputs.coding[t]
        mean_phi = np.mean(w[:, None] * phi, axis=0)
        blip = solution.psi[t].reshape(phi.shape[1], coding.shape[0])
        curve = curve + (w[:, None] * (phi - mean_phi)) @ (blip @ coding)
        gradient.append(np.kron(mean_phi, coding))
        value += float(mean_phi @ blip @ coding)
    curve = curve + solution.psi_curve @ np.concatenate(gradient)
    return value, curve


def _regimen_inputs(
    data: LongitudinalData, nodes: Sequence[_Node], codes: Sequence[float]
) -> _RegimenInputs:
    """The features and codings of the static regimen with treatment codes ``codes``."""
    phi = []
    coding = []
    for node in nodes:
        overrides = {s: np.full(data.n, codes[s - 1], dtype=float) for s in range(1, node.time)}
        phi.append(_phi(data, node, overrides))
        coding.append(
            _encode(
                np.asarray([codes[node.time - 1]], dtype=float),
                node.n_levels,
                node.continuous,
            )[0]
        )
    return _RegimenInputs(phi=phi, coding=coding)


# --------------------------------------------------------------------- the regimens


@dataclass(frozen=True)
class _StaticRegimen:
    """A static regimen, resolved against the levels of each node.

    Parameters
    ----------
    label : str
        The name the regimen is reported under.
    values : tuple
        The label or dose the regimen assigns at each node.
    codes : tuple of float
        The dense level code, or the dose, at each node.
    """

    label: str
    values: tuple[object, ...]
    codes: tuple[float, ...]


def _code_of(data: LongitudinalData, regimen: Regimen, time: int) -> float:
    value = regimen.values[time - 1]
    if _is_dose(data, time):
        try:
            dose = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise DataError(
                f"regimen {regimen.label!r} assigns {value!r} at node {time}, which holds a "
                "dose; a static dose is a number"
            ) from None
        observed = np.asarray(data.node_codes(time), dtype=float)
        low, high = float(np.nanmin(observed)), float(np.nanmax(observed))
        if not low <= dose <= high:
            raise _invalid(
                f"regimen {regimen.label!r} assigns the dose {dose:g} at node {time}, outside "
                f"the observed range [{low:g}, {high:g}]. The static dose level is regular "
                "under the structural model only inside that range"
            )
        return dose
    levels = _level_labels(data, time)
    for index, level in enumerate(levels):
        if _same_label(value, level):
            return float(index)
    raise DataError(
        f"regimen {regimen.label!r} assigns {value!r} at node {time}, which has the levels "
        f"{list(levels)}"
    )


def _static_regimens(spec: Any, data: LongitudinalData) -> tuple[_StaticRegimen, ...]:
    """The static regimens of ``regimens=``, refusing a rule or a policy."""
    if spec is None:
        return ()
    resolved = resolve_regimens(spec, data.n_decisions)
    for regimen in resolved:
        if isinstance(regimen, DynamicRegimen):
            if regimen.has_tilt:
                raise CapabilityError.from_refusal(_TILT_REFUSAL)
            raise CapabilityError.from_refusal(_DYNAMIC_REFUSAL)
    return tuple(
        _StaticRegimen(
            label=regimen.label,
            values=tuple(regimen.values),
            codes=tuple(_code_of(data, regimen, t) for t in range(1, data.n_times + 1)),
        )
        for regimen in resolved
        if isinstance(regimen, Regimen)
    )


# ------------------------------------------------------------------ result containers


@dataclass(frozen=True)
class StructuralConfig:
    """The settings a fit ran with.

    Parameters
    ----------
    n_times : int
        The number of treatment nodes.
    treatment_names : tuple of str
        The treatment column of each node.
    nodes : tuple
        The blip layout of each node: features and coding components.
    regimens : tuple
        The static regimens reported, as ``(label, values)`` pairs.
    reference : str or None
        The regimen every contrast is taken against, or ``None`` with no regimen.
    n_folds : int
        The number of outer folds.
    min_eigenvalue : float
        The refusal floor of the diagonal Jacobian blocks.
    positivity_margin : float
        The distance from 0 or 1 that the positivity share counts.
    alpha_sig : float
        The significance level of the intervals.
    family : str
        The outcome family.
    random_state : int or None
        The seed of the folds and of the default learners.
    reads_covariates : bool
        Whether a blip feature reads a time-varying covariate.
    support_key : SupportKey or None, default None
        The support key the fit was classified under, read before any nuisance was fitted.
    """

    n_times: int
    treatment_names: tuple[str, ...]
    nodes: tuple[_Node, ...]
    regimens: tuple[tuple[str, tuple[object, ...]], ...]
    reference: str | None
    n_folds: int
    min_eigenvalue: float
    positivity_margin: float
    alpha_sig: float
    family: str
    random_state: int | None
    reads_covariates: bool
    support_key: SupportKey | None = None

    def describe(self) -> list[str]:
        """The settings as summary lines."""
        lines = [f"model: {SNMM_LABEL}"]
        for node in self.nodes:
            features = ", ".join(feature.name for feature in node.features)
            lines.append(
                f"node {node.time} ({node.treatment}): blip features [{features}] x "
                f"{list(node.levels)}"
            )
        if self.regimens:
            lines.append(
                "regimens = "
                + ", ".join(
                    f"{label} ({'/'.join(str(v) for v in values)})"
                    for label, values in self.regimens
                )
                + f"; reference = {self.reference}"
            )
        lines.append(f"cross-fitting = {self.n_folds} folds; outcome family = {self.family}")
        if self.reads_covariates:
            lines.append(
                "a blip feature reads a time-varying covariate: static levels and contrasts "
                "are not reported (X72 in docs/roadmap.md)"
            )
        return lines


@dataclass(frozen=True)
class StructuralDiagnostics:
    """What the fit measured about positivity and nuisance fits.

    Parameters
    ----------
    smallest_eigenvalue : tuple of float
        For each node, the smallest eigenvalue of its diagonal Jacobian block after the block
        is standardized to unit diagonal.  A value near zero says that the treatment varies
        little given the history.
    extreme_share : tuple of float
        For each categorical node, the weighted share of rows whose estimated class
        probability lies within ``positivity_margin`` of 0 or 1.  ``nan`` for a dose.
    n_constant_fits : int
        The count of classification regressions with a fold whose training rows held one
        class.
    learner_diagnostics : tuple
        The Super Learner diagnostics of every regression, empty for other learners.
    """

    smallest_eigenvalue: tuple[float, ...]
    extreme_share: tuple[float, ...]
    n_constant_fits: int
    learner_diagnostics: tuple[Any, ...] = ()


def _split_seeds(random_state: int | None, count: int) -> list[int]:
    """The seeds of the further fold splits of a model check.

    A seeded fit derives them from its seed, so the check is reproducible.  An unseeded fit
    draws them from operating-system entropy.
    """
    if random_state is None:
        return [_fresh_seed() for _ in range(count)]
    sequence = np.random.SeedSequence([int(random_state), 0x5EED])
    return [int(seed) % (2**31 - 1) for seed in sequence.generate_state(count)]


def _split_covariance(
    differences: FloatArray, curves: Sequence[FloatArray], cluster: Any
) -> FloatArray:
    """The covariance of the split-averaged difference of a model check.

    ``differences`` is the ``(S, k)`` difference of the contrasts on each split, and
    ``curves`` holds the ``(n, k)`` difference curve of each split.  Given the data, the
    splits are independent, so the variance of the mean difference is the variance of its
    conditional mean plus the between-split covariance over ``S``.  The first part is the
    mean covariance of curves from two different splits: each curve's own split noise is
    independent of the other's, so the cross terms do not carry it.  One split reads its
    own curve.
    """
    count = len(curves)
    if count == 1:
        return influence_covariance(curves[0], cluster)
    total = influence_covariance(np.sum(curves, axis=0), cluster)
    own = np.sum([influence_covariance(curve, cluster) for curve in curves], axis=0)
    first_order = (total - own) / (count * (count - 1))
    k = differences.shape[1]
    between = np.cov(differences, rowvar=False, ddof=1).reshape(k, k)
    covariance = first_order + between / count
    return np.asarray((covariance + covariance.T) / 2.0, dtype=float)


def _satterthwaite_df(
    differences: FloatArray, curves: Sequence[FloatArray], cluster: Any, covariance: FloatArray
) -> float:
    """The Welch-Satterthwaite degrees of freedom of the model-check covariance.

    The Wald statistic has its chi-squared law only when the covariance is estimated with
    negligible error.  Each diagonal entry is a sum of per-unit terms plus, over several
    splits, the between-split variance over ``S``.  Under weak positivity a few rows carry
    large inverse weights, and they carry most of that sum.  Its relative error is then
    large at any number of splits.  The Satterthwaite (1946) value is ``2 v**2 / var(v)``.
    It uses the empirical variance of the per-unit terms and ``2 b**2 / (S - 1)`` for the
    between-split part ``b``.  It approximates each entry by a scaled chi-squared law, as
    Bell and McCaffrey (2002) and Imbens and Kolesar (2016) do for a robust variance with
    high-leverage rows.  The joint test takes the smallest value over the contrasts.
    """
    count = len(curves)
    units = [
        np.asarray(curve if cluster is None else cluster_sums(curve, cluster), dtype=float)
        for curve in curves
    ]
    centered = [unit - unit.mean(axis=0) for unit in units]
    m = centered[0].shape[0]
    n = curves[0].shape[0]
    # influence_covariance scales the centered sum of squares by m / ((m - 1) n**2).
    scale = m / ((m - 1) * n**2)
    diagonal = np.diag(covariance)
    if count == 1:
        terms = centered[0] ** 2
        between_variance = np.zeros_like(diagonal)
    else:
        total = np.sum(centered, axis=0)
        terms = (total**2 - np.sum([unit**2 for unit in centered], axis=0)) / (count * (count - 1))
        between = np.var(differences, axis=0, ddof=1) / count
        between_variance = 2.0 * between**2 / (count - 1)
    term_variance = scale**2 * np.sum((terms - terms.mean(axis=0)) ** 2, axis=0)
    error = term_variance + between_variance
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(error > 0.0, 2.0 * diagonal**2 / error, np.inf)
    return float(np.min(values))


def _wald_pvalue(statistic: float, k: int, reference_df: float) -> float:
    """The tail probability of a Wald statistic over ``k`` contrasts.

    The covariance has ``reference_df`` degrees of freedom.  The Hotelling law gives
    ``T (nu - k + 1) / (nu k)`` the law ``F(k, nu - k + 1)``.  An infinite ``nu`` reads
    chi-squared on ``k``.  With ``nu`` at most ``k - 1`` there is no reference law, and the
    value is ``nan``.
    """
    if not np.isfinite(reference_df):
        return float(stats.chi2.sf(statistic, df=k))
    if reference_df <= k - 1:
        return float("nan")
    ratio = statistic * (reference_df - k + 1) / (reference_df * k)
    return float(stats.f.sf(ratio, k, reference_df - k + 1))


def _smallest_eigenvalue(block: FloatArray) -> float:
    """The smallest eigenvalue of ``block`` standardized to unit diagonal."""
    diagonal = np.diag(block)
    if not np.all(np.isfinite(block)) or np.any(diagonal <= 0.0):
        return 0.0
    scale = 1.0 / np.sqrt(diagonal)
    standardized = block * scale[:, None] * scale[None, :]
    return float(np.linalg.eigvalsh((standardized + standardized.T) / 2.0)[0])


def _extreme_share(own: FloatArray, weights: FloatArray, margin: float) -> float:
    if own.shape[1] == 0:
        return float("nan")
    full = np.column_stack([1.0 - own.sum(axis=1), own])
    flagged = np.any((full < margin) | (full > 1.0 - margin), axis=1)
    return float(np.average(flagged, weights=weights))


# ----------------------------------------------------------------------- the estimator


class StructuralNestedMean:
    """G-estimation of a linear structural nested mean model over treatment nodes.

    The estimator reports the blip coefficients of each node, the mean outcome with every
    treatment at its reference level, and, for declared static regimens, their means and
    contrasts.  The coefficients are the parameters of a restricted model.  A level or a
    contrast is the same estimand as the one :class:`~cleverly.longitudinal.LTMLE` reports
    only when the model holds.

    Parameters
    ----------
    regimens : mapping, sequence, or None
        Static regimens, as for :class:`~cleverly.longitudinal.LTMLE`.  ``None`` reports the
        blips and the reference mean only.  A rule, a modified treatment policy or an incremental
        policy is refused.
    reference : str or None, default=None
        The regimen every contrast is taken against.  ``None`` takes the first.
    blip_features : mapping or None, default=None
        The features :math:`\\phi_t` of each node.  A key is a node counted from one or the
        treatment column.  A value lists terms.  A term is a column name, or a tuple of names
        for a product.  A blip may read baseline columns, earlier treatments and the
        time-varying covariates through its own node.  The intercept is always included.  A
        feature that reads a time-varying covariate allows the blips and ``ey_reference``
        only.
    outcome_learner : learner or None, default=None
        The learner of :math:`q_t`.  ``None`` fits a Super Learner.
    treatment_learner : learner or None, default=None
        The learner of the treatment-side nuisances :math:`m_{j,t}`.  A target in {0, 1}
        takes this learner as a classifier.  ``None`` fits a Super Learner.
    regression_learner : learner or None, default=None
        The learner of every target outside {0, 1}: a continuous outcome, a dose, a product
        with a dose or with a continuous feature.  ``None`` takes ``outcome_learner``.  A
        binary outcome with a dose node or a continuous feature needs this setting when
        ``outcome_learner`` is a classifier.
    n_folds : int, default=10
        The outer folds, drawn over trajectories.  ``1`` fits in sample.
    learner_folds : int, default=5
        The inner folds of a default Super Learner.
    positivity_margin : float, default=0.01
        The distance from 0 or 1 that the diagnostic share of extreme class probabilities
        counts.  It truncates nothing.
    min_eigenvalue : float, default=1e-8
        The floor of the smallest standardized eigenvalue of each diagonal Jacobian block.
        A smaller block raises.
    alpha_sig : float, default=0.05
        The significance level of the intervals.
    simultaneous : bool, default=True
        Whether to report simultaneous bands.
    n_multiplier : int, default=2000
        Multiplier draws of the bands.
    multiplier_kind : str, default="rademacher"
        The multiplier law.
    random_state : int or None, default=None
        The seed of the folds, the default learners and the bands.
    run_id : str or None, default=None
        A label for the provenance record.
    n_jobs : int, default=1
        Workers for the nuisance fits.
    msm : None
        Refused.  A working model over regimens is declared on ``LTMLE`` or ``SDR``.
    pseudo_learner : None
        Refused.  The estimator builds no pseudo-outcome.
    censoring_learner : None
        Refused.  The estimator fits an end-of-study outcome with no censoring.
    horizons : None
        Refused.  The outcome is one end-of-study value.

    Raises
    ------
    CapabilityError
        When a refused setting is passed, with the reason code of the refusal.

    See Also
    --------
    snmm : The one-call form.
    cleverly.longitudinal.LTMLE : The nonparametric estimator of static regimen values.
    StructuralNestedMeanResult.check_model : The comparison with ``LTMLE``.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.learners import GLM
    >>> from cleverly.longitudinal import StructuralNestedMean
    >>> frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    >>> result = StructuralNestedMean(
    ...     {"always": 1, "never": 0},
    ...     outcome_learner=GLM(family="binomial"),
    ...     treatment_learner=GLM(family="binomial"),
    ...     n_folds=2,
    ...     random_state=0,
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ... )
    >>> sorted(result)  # doctest: +NORMALIZE_WHITESPACE
    ['ate_regimen[never vs always]', 'blip[A1:1.0:intercept]', 'blip[A2:1.0:intercept]',
     'ey_reference', 'ey_regimen[always]', 'ey_regimen[never]']
    """

    def __init__(
        self,
        regimens: Any = None,
        *,
        reference: str | None = None,
        blip_features: Mapping[Any, Any] | None = None,
        outcome_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        regression_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        positivity_margin: float = 0.01,
        min_eigenvalue: float = DEFAULT_MIN_EIGENVALUE,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        msm: Any = None,
        pseudo_learner: Any = None,
        censoring_learner: Any = None,
        horizons: Any = None,
        **refused: Any,
    ) -> None:
        refuse_unsupported(refused, where="StructuralNestedMean")
        if msm is not None:
            raise CapabilityError.from_refusal(_MSM_REFUSAL)
        for name, value in (
            ("pseudo_learner", pseudo_learner),
            ("censoring_learner", censoring_learner),
            ("horizons", horizons),
        ):
            if value is not None:
                reasons = {
                    "pseudo_learner": "builds no pseudo-outcome, so it has no learner for one",
                    "censoring_learner": "fits an end-of-study outcome with no censoring",
                    "horizons": "fits one end-of-study outcome",
                }
                raise _invalid(
                    f"StructuralNestedMean {reasons[name]}, so {name}= has nothing to set"
                )
        _validate_learner(outcome_learner, "outcome_learner")
        _validate_learner(treatment_learner, "treatment_learner")
        _validate_learner(regression_learner, "regression_learner")
        if not 0.0 < alpha_sig < 0.5:
            raise ValueError(f"alpha_sig must lie in (0, 0.5); got {alpha_sig}")
        if n_folds < 1:
            raise ValueError(f"n_folds must be at least 1; got {n_folds}")
        if not 0.0 < positivity_margin < 0.5:
            raise ValueError(f"positivity_margin must lie in (0, 0.5); got {positivity_margin}")
        if not min_eigenvalue >= 0.0:
            raise ValueError(f"min_eigenvalue must not be negative; got {min_eigenvalue}")
        self.regimens = regimens
        self.reference = reference
        self.blip_features = blip_features
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.regression_learner = regression_learner
        self.n_folds = n_folds
        self.learner_folds = learner_folds
        self.positivity_margin = positivity_margin
        self.min_eigenvalue = min_eigenvalue
        self.alpha_sig = alpha_sig
        self.simultaneous = simultaneous
        self.n_multiplier = n_multiplier
        self.multiplier_kind = multiplier_kind
        self.random_state = random_state
        self.run_id = run_id
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | Repeated | None = None,
        treatment: str | Sequence[str] | None = None,
        baseline: Sequence[str] | None = None,
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        family: str = "auto",
        continuous_treatment: Sequence[str] = (),
        **refused: Any,
    ) -> StructuralNestedMeanResult:
        """Fit on a wide dataframe, or on an already-built :class:`LongitudinalData`.

        The columns are read as :meth:`cleverly.longitudinal.LTMLE.fit` reads them.  The
        design must hold one end-of-study outcome and no censoring.  ``weights=`` names known
        observation weights: every nuisance is fitted with them, every mean is weighted, and
        the curve carries the weight of its row.  ``id=`` names a cluster column: the folds
        hold whole clusters and the variance sums the curve within a cluster.

        Parameters
        ----------
        data : DataFrame or LongitudinalData
            One wide frame, or a container that has already read the columns.
        outcome : str or None
            The end-of-study outcome column.
        treatment : str, sequence of str, or None
            The treatment column of each node, in time order.
        baseline : sequence of str or None
            The covariates measured before any treatment.
        time_varying : sequence of sequence of str or None
            The covariates measured before the treatment of each node.
        censoring : sequence of str or None
            Refused when it names a column.
        id : str or None
            A cluster column.
        weights : str or None
            A column of known observation weights.
        weights_type : str
            How to read ``weights``.
        weights_estimated : bool
            Declare that the weights came out of a fitted model.  The fit then takes the
            ``estimated_weight_plugin`` status.
        family : {"auto", "binomial", "gaussian"}
            The outcome family.
        continuous_treatment : sequence of str
            The treatment columns that hold a dose.  A dose enters the blip linearly, and the
            reference dose 0 must lie in the observed range.
        **refused : Any
            A point-treatment keyword, refused by name.

        Returns
        -------
        StructuralNestedMeanResult
            The blips, the reference mean, and the declared regimen means and contrasts.
        """
        refuse_unsupported(refused, where="StructuralNestedMean.fit")
        from .estimator import _SequentialEstimator

        prepared = _SequentialEstimator._prepare(
            data,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            id=id,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            family=family,
            continuous_treatment=continuous_treatment,
        )
        self._refuse_design(prepared)
        nodes = _parse_blip_features(prepared, self.blip_features)
        regimens = _static_regimens(self.regimens, prepared)
        reads_covariates = _reads_covariate(nodes)
        if regimens and reads_covariates:
            raise CapabilityError.from_refusal(_COVARIATE_FEATURE_REFUSAL)
        reference = self._reference(regimens)
        self._refuse_reference_dose(prepared)
        folds = self._folds(prepared)
        check_integrity(folds, cluster=prepared.cluster)
        # No refusal of this estimator is a composition that the code computes and no study
        # covers, so the classification carries no condition: the label rests on the rows.
        classification = classify_structural_fit(
            self,
            prepared,
            source="StructuralNestedMean.fit",
            cross_fitted=not folds.is_single,
            has_regimens=bool(regimens),
            covariate_features=reads_covariates,
        )
        learners = _Learners(
            self.outcome_learner,
            self.treatment_learner,
            learner_folds=self.learner_folds,
            random_state=self.random_state,
            regression=self.regression_learner,
        )
        nuisances = _fit_nuisances(prepared, nodes, learners, folds, self.n_jobs)
        stage = _build_stage(prepared, nodes, nuisances)
        eigenvalues = self._check_positivity(nodes, stage, prepared)
        solution = _solve(stage, prepared.weights)
        status = _status(
            prepared,
            folds,
            parametric=all(isinstance(learner, GLM) for learner in learners.used),
        )
        diagnostics = StructuralDiagnostics(
            smallest_eigenvalue=tuple(eigenvalues),
            extreme_share=tuple(
                _extreme_share(
                    nuisances.own[node.time - 1], prepared.weights, self.positivity_margin
                )
                for node in nodes
            ),
            n_constant_fits=nuisances.n_constant,
            learner_diagnostics=tuple(nuisances.diagnostics),
        )
        estimates, kinds = self._estimates(
            prepared, nodes, regimens, reference, stage, solution, status, classification
        )
        _warn_statuses(estimates)
        config = StructuralConfig(
            n_times=prepared.n_times,
            treatment_names=tuple(prepared.treatment_names),
            nodes=nodes,
            regimens=tuple((r.label, r.values) for r in regimens),
            reference=None if reference is None else reference.label,
            n_folds=folds.n_folds,
            min_eigenvalue=self.min_eigenvalue,
            positivity_margin=self.positivity_margin,
            alpha_sig=self.alpha_sig,
            family=prepared.family,
            random_state=self.random_state,
            reads_covariates=reads_covariates,
            support_key=None if classification is None else classification.key,
        )
        result = StructuralNestedMeanResult(
            estimates=estimates,
            kinds=kinds,
            coefficients=tuple(
                solution.psi[node.time - 1].reshape(node.p, node.d) for node in nodes
            ),
            psi_curve=solution.psi_curve,
            data=prepared,
            config=config,
            folds=folds,
            provenance=self._provenance(prepared, folds),
            simultaneous=self._bands(estimates, kinds, prepared, status, classification),
            diagnostics=diagnostics,
            static_regimens=regimens,
            estimator=copy.copy(self),
            label=SNMM_LABEL,
        )
        record_fit(classification, result, source="StructuralNestedMean.fit")
        return result

    # -------------------------------------------------------------- internals

    @staticmethod
    def _refuse_design(data: LongitudinalData) -> None:
        if data.is_competing:
            raise CapabilityError.from_refusal(_SURVIVAL_REFUSAL)
        if data.is_survival:
            raise CapabilityError.from_refusal(_SURVIVAL_REFUSAL)
        if data.is_repeated:
            raise CapabilityError.from_refusal(_REPEATED_REFUSAL)
        if data.censoring_names or data.has_censoring:
            raise CapabilityError.from_refusal(_CENSORING_REFUSAL)
        if data.treatment_held:
            raise CapabilityError.from_refusal(_HELD_REFUSAL)
        if data.known_mechanisms is not None:
            raise CapabilityError.from_refusal(_KNOWN_MECHANISM_REFUSAL)
        if data.markov_order is not None:
            raise CapabilityError.from_refusal(_MARKOV_REFUSAL)
        if data.has_mixed_node:
            raise CapabilityError.from_refusal(_mixed_refusal(data))

    def _reference(self, regimens: Sequence[_StaticRegimen]) -> _StaticRegimen | None:
        if not regimens:
            if self.reference is not None:
                raise _invalid("reference= names a regimen, and no regimen is declared")
            return None
        if self.reference is None:
            return regimens[0]
        for regimen in regimens:
            if regimen.label == self.reference:
                return regimen
        raise KeyError(
            f"reference={self.reference!r} is not one of the declared regimens "
            f"{[regimen.label for regimen in regimens]}"
        )

    @staticmethod
    def _refuse_reference_dose(data: LongitudinalData) -> None:
        """The reference dose 0 of a continuous node must lie in the observed range."""
        for node in range(1, data.n_times + 1):
            if not _is_dose(data, node):
                continue
            observed = np.asarray(data.node_codes(node), dtype=float)
            low, high = float(np.nanmin(observed)), float(np.nanmax(observed))
            if not low <= 0.0 <= high:
                raise _invalid(
                    f"the dose at node {node} ({data.treatment_names[node - 1]}) has the "
                    f"observed range [{low:g}, {high:g}], which excludes the reference dose "
                    "0. The blip is the effect of a dose against dose 0, and E[Y(0)] sets "
                    "every dose to 0",
                    alternative="Center the dose so that the reference lies in the range.",
                )

    def _folds(self, data: LongitudinalData) -> Folds:
        if self.n_folds <= 1:
            return Folds.single(data.n)
        return random_partition(
            data.n,
            self.n_folds,
            cluster=data.cluster,
            seed=_fresh_seed() if self.random_state is None else int(self.random_state),
        )

    def _check_positivity(
        self, nodes: Sequence[_Node], stage: _Stage, data: LongitudinalData
    ) -> list[float]:
        eigenvalues = []
        w = data.weights
        for node in nodes:
            own = stage.z_res[(node.time, node.time)]
            block = (own * w[:, None]).T @ own / data.n
            value = _smallest_eigenvalue(block)
            eigenvalues.append(value)
            if not value > self.min_eigenvalue:
                raise _invalid(
                    f"the diagonal Jacobian block of node {node.time} "
                    f"({node.treatment}) has the smallest standardized eigenvalue "
                    f"{value:.3g}, below the floor {self.min_eigenvalue:g}. The treatment "
                    "varies too little given the history, or a blip feature is collinear "
                    "with the other features given the history, so the node's blip is not "
                    "identified",
                    alternative="Drop a feature, merge treatment levels, or check positivity.",
                )
        return eigenvalues

    def _estimates(
        self,
        data: LongitudinalData,
        nodes: Sequence[_Node],
        regimens: Sequence[_StaticRegimen],
        reference: _StaticRegimen | None,
        stage: _Stage,
        solution: _Solution,
        status: InferenceStatus,
        classification: Classification | None = None,
    ) -> tuple[dict[str, ParameterEstimate], dict[str, str]]:
        df = _reference_df(data, status)
        support = support_for(classification, status, df)

        def build(name: str, value: float, curve: FloatArray, scale: Scale) -> ParameterEstimate:
            return make_estimate(
                name,
                value,
                curve,
                n=data.n,
                cluster=data.cluster,
                scale=scale,
                alpha=self.alpha_sig,
                inference=status,
                reference_df=df,
                support=support,
            )

        estimates: dict[str, ParameterEstimate] = {}
        kinds: dict[str, str] = {}
        offsets = stage.offsets
        for node in nodes:
            for l, level in enumerate(node.levels):
                for k, feature in enumerate(node.features):
                    name = blip_name(node, level, feature.name)
                    index = offsets[node.time - 1] + k * node.d + l
                    estimates[name] = build(
                        name,
                        float(solution.psi[node.time - 1][k * node.d + l]),
                        solution.psi_curve[:, index],
                        "difference",
                    )
                    kinds[name] = "blip"
        estimates[REFERENCE_NAME] = build(REFERENCE_NAME, solution.mu0, solution.mu0_curve, "level")
        kinds[REFERENCE_NAME] = "reference"
        levels = {
            regimen.label: _level_curve(solution, _regimen_inputs(data, nodes, regimen.codes))
            for regimen in regimens
        }
        for regimen in regimens:
            name = f"ey_regimen[{regimen.label}]"
            estimates[name] = build(name, *levels[regimen.label], "level")
            kinds[name] = "level"
        if reference is not None:
            base_value, base_curve = levels[reference.label]
            for regimen in regimens:
                if regimen.label == reference.label:
                    continue
                value, curve = levels[regimen.label]
                name = f"ate_regimen[{regimen.label} vs {reference.label}]"
                estimates[name] = build(name, value - base_value, curve - base_curve, "difference")
                kinds[name] = "contrast"
        return estimates, kinds

    def _bands(
        self,
        estimates: Mapping[str, ParameterEstimate],
        kinds: Mapping[str, str],
        data: LongitudinalData,
        status: InferenceStatus,
        classification: Classification | None = None,
    ) -> SimultaneousBands | None:
        """Joint bands over the regimen means and contrasts, or over the blips with none.

        The band takes the support of :func:`~cleverly._support.band_support`: ``supported``
        only when a row licenses a ``band`` and every estimate is supported on that row.
        """
        family = [name for name, kind in kinds.items() if kind in ("level", "contrast")]
        if not family:
            family = [name for name, kind in kinds.items() if kind == "blip"]
        chosen = {name: estimates[name] for name in family if estimates[name].supplies_inference}
        if not self.simultaneous or len(chosen) < 2 or has_t_reference(chosen):
            return None
        bands = simultaneous_bands(
            chosen,
            alpha=self.alpha_sig,
            n_replicates=self.n_multiplier,
            kind=self.multiplier_kind,  # type: ignore[arg-type]
            random_state=self.random_state,
            cluster=data.cluster,
        )
        return replace(
            bands,
            support=band_support(
                classification, status, (estimate.support for estimate in chosen.values())
            ),
        )

    def _provenance(self, data: LongitudinalData, folds: Folds) -> Provenance:
        return provenance_build(
            n=data.n,
            n_covariates=len(data.baseline_names)
            + sum(len(names) for names in data.time_varying_names),
            n_clusters=None if data.cluster is None else data.n_clusters,
            data_fingerprint=fingerprint_array(
                data.outcome,
                data.treatment,
                data.baseline,
                data.weights,
                *data.time_varying,
                np.asarray(
                    [repr(level) for levels in data.treatment_levels for level in levels],
                    dtype=str,
                ),
            ),
            fold_fingerprint=fingerprint_array(folds.assignment),
            random_state=self.random_state,
            run_id=self.run_id,
        )


def blip_name(node: _Node, level: str, feature: str) -> str:
    """The reported name of one blip coefficient.

    Parameters
    ----------
    node : _Node
        The node.
    level : str
        The coding component: a non-reference level label, or ``"dose"``.
    feature : str
        The feature name, ``"intercept"`` for the constant.

    Returns
    -------
    str
        ``blip[<treatment>:<level>:<feature>]``.
    """
    return f"blip[{node.treatment}:{level}:{feature}]"


def _status(data: LongitudinalData, folds: Folds, *, parametric: bool) -> InferenceStatus:
    """The fit status of a structural fit.

    A cross-fitted fit takes ``influence_curve`` for any learner.  An in-sample fit takes it
    when every learner is a :class:`~cleverly.learners.GLM`, and
    ``adaptive_baseline_plugin`` otherwise.  Known weights declared estimated take
    ``estimated_weight_plugin``, and few clusters take ``few_cluster_plugin``.
    """
    cluster = cluster_inference_status(
        data.cluster,
        weights=data.weights if data.is_weighted else None,
        minimum=MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS,
    )
    adaptive = (
        "influence_curve" if (not folds.is_single or parametric) else ADAPTIVE_BASELINE_STATUS
    )
    weighted = "estimated_weight_plugin" if data.weight_spec.estimated else "influence_curve"
    return precedent_status((cluster, adaptive, weighted))


def _warn_statuses(estimates: Mapping[str, ParameterEstimate]) -> None:
    statuses = {estimate.inference for estimate in estimates.values()}
    for status in sorted(s for s in statuses if not supplies_inference(s)):
        warnings.warn(
            f"this StructuralNestedMean fit takes the {status!r} status: "
            + status_record(status).reason,
            UserWarning,
            stacklevel=3,
        )


# ------------------------------------------------------------------------ the result


@dataclass(frozen=True)
class StructuralModelCheck:
    """The comparison of the structural contrasts with the nonparametric ones.

    The difference of the two contrast vectors has mean zero when the structural model holds.
    The statistic is a joint Wald statistic over the contrasts against the reference.  It is a
    necessary-condition check and not a test of the model.  On a cross-fitted fit, every
    estimate is the mean over ``n_splits`` fold splits, the first of which is the fit's own.

    Parameters
    ----------
    names : tuple of str
        The contrasts, ``ate_regimen[<regimen> vs <reference>]``.
    structural : ndarray
        The structural point estimates, averaged over the splits.
    nonparametric : ndarray
        The :class:`~cleverly.longitudinal.LTMLE` point estimates on the same rows and
        splits, averaged over the splits.
    covariance : ndarray
        The covariance of the split-averaged difference: the cross-split covariance of the
        difference curves plus the between-split covariance of the difference over
        ``n_splits``.
    statistic : float
        The Wald statistic, ``nan`` when no test is reported.
    df : int
        The degrees of freedom: the number of contrasts.
    pvalue : float
        The tail probability of the Hotelling-type reference law, ``nan`` when no test is
        reported.
    reason : str or None
        Why no test is reported, or ``None``.
    ltmle : LongitudinalResult
        The nonparametric fit on the folds of the structural fit.
    n_splits : int
        The fold splits behind the estimates: ``1`` for an in-sample fit.
    reference_df : float, default=nan
        The Welch-Satterthwaite degrees of freedom of the covariance, the smallest over the
        contrasts.  The p-value reads the Hotelling-type law
        ``F(df, reference_df - df + 1)`` of the scaled statistic, and ``inf`` reads
        chi-squared.  ``nan`` when no statistic is reported.
    support : Support, default=UNCLASSIFIED
        What the package states about the evidence behind the statistic: the combination of
        the support of the structural contrasts and of the nonparametric contrasts that the
        statistic compares.  The check is ``supported`` only when every one of them is
        supported on one row.
    """

    names: tuple[str, ...]
    structural: FloatArray
    nonparametric: FloatArray
    covariance: FloatArray
    statistic: float
    df: int
    pvalue: float
    reason: str | None
    ltmle: Any
    n_splits: int
    reference_df: float = float("nan")
    support: Support = UNCLASSIFIED

    @property
    def difference(self) -> FloatArray:
        """The structural minus the nonparametric contrasts."""
        return np.asarray(self.structural - self.nonparametric)

    @property
    def tested(self) -> bool:
        """Whether the check reports a statistic."""
        return self.reason is None

    def summary(self) -> str:
        """A printable table of the two sets of contrasts and the joint statistic.

        Returns
        -------
        str
            One row per contrast, then the Wald statistic or the reason for none.
        """
        standard = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        rows = [
            [
                name,
                f"{a:.4f}",
                f"{b:.4f}",
                f"{a - b:.4f}",
                f"{s:.4f}",
            ]
            for name, a, b, s in zip(
                self.names, self.structural, self.nonparametric, standard, strict=True
            )
        ]
        table = format_table(
            ["contrast", "structural", "nonparametric", "difference", "se of difference"], rows
        )
        if self.tested:
            tail = (
                f"joint Wald statistic = {self.statistic:.3f} on {self.df} contrasts, "
                f"covariance df = {self.reference_df:.1f}, p = {format_pvalue(self.pvalue)}"
            )
        else:
            tail = f"no test is reported: {self.reason}"
        if self.n_splits > 1:
            tail += f"\nestimates and covariance over {self.n_splits} fold splits"
        marker = (
            [f"experimental: {self.support.reason}"]
            if (self.support.label == "experimental")
            else []
        )
        return "\n".join(
            [
                "Structural model check (a necessary condition, not a test of the model)",
                "",
                table,
                "",
                tail,
                *marker,
            ]
        )


@dataclass
class StructuralNestedMeanResult:
    """The fitted blips, the reference mean and the declared regimen values.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`.  The names are
    ``blip[<treatment>:<level>:<feature>]``, ``ey_reference``, ``ey_regimen[<label>]`` and
    ``ate_regimen[<label> vs <reference>]``.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        The estimates in report order.
    kinds : dict of str to str
        ``"blip"``, ``"reference"``, ``"level"`` or ``"contrast"`` for each name.
    coefficients : tuple of ndarray
        The blip coefficients of each node as :math:`\\Psi_t^\\top`, shape ``(p_t, d_t)``:
        one row per feature and one column per coding component.
    psi_curve : ndarray
        The ``(n, P)`` influence curves of the stacked coefficients.
    data : LongitudinalData
        The prepared data.
    config : StructuralConfig
        The settings of the fit.
    folds : Folds
        The realized outer folds.
    provenance : Provenance
        The runtime and data record.
    simultaneous : SimultaneousBands or None
        Joint bands over the regimen parameters, or over the blips with no regimen.
    diagnostics : StructuralDiagnostics
        Positivity and nuisance diagnostics.
    static_regimens : tuple
        The resolved static regimens.
    estimator : StructuralNestedMean
        The settings the fit ran with, which :meth:`check_model` reuses.
    label : str
        The structural condition under which the intervals are valid.

    See Also
    --------
    StructuralNestedMean : The estimator.
    cleverly.longitudinal.LongitudinalResult : The result of the nonparametric estimators.
    """

    estimates: dict[str, ParameterEstimate]
    kinds: dict[str, str]
    coefficients: tuple[FloatArray, ...]
    psi_curve: FloatArray
    data: LongitudinalData
    config: StructuralConfig
    folds: Folds
    provenance: Provenance
    diagnostics: StructuralDiagnostics
    static_regimens: tuple[_StaticRegimen, ...]
    estimator: StructuralNestedMean
    label: str
    simultaneous: SimultaneousBands | None = None

    #: The method identity of the result.  The result declares no ``assessment_family``, so
    #: ``assessment_capabilities`` does not accept it until X74 in ``docs/roadmap.md`` adds one.
    fitted_method: ClassVar[str] = "snmm"

    # ------------------------------------------------------------- mapping API

    def __getitem__(self, name: str) -> ParameterEstimate:
        try:
            return self.estimates[name]
        except KeyError:
            raise KeyError(
                f"{name!r} was not estimated; this fit reports {list(self.estimates)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    def keys(self) -> Any:
        """The parameter names in report order."""
        return self.estimates.keys()

    def values(self) -> Any:
        """The estimates in report order."""
        return self.estimates.values()

    def items(self) -> Any:
        """The ``(name, estimate)`` pairs in report order."""
        return self.estimates.items()

    def __contains__(self, name: object) -> bool:
        return name in self.estimates

    # --------------------------------------------------------------- inference

    @property
    def estimate(self) -> ParameterEstimate:
        """The sole estimate, refusing ambiguity on a multi-parameter result."""
        return sole_estimate(self.estimates)

    @property
    def inference_status(self) -> InferenceStatus:
        """The inference status of the fit.

        ``"influence_curve"`` is the interval status.  The structural condition
        :attr:`label` states what the interval is valid for.
        """
        return reported_status(self.estimates)

    @property
    def n(self) -> int:
        """The number of rows."""
        return self.data.n

    def psi(self, name: str | None = None) -> float:
        """Return a point estimate by parameter name.

        Parameters
        ----------
        name : str or None
            The parameter name.  ``None`` is allowed when the fit reported one.

        Returns
        -------
        float
            The point estimate.
        """
        return self.estimate.psi if name is None else self[name].psi

    @property
    def influence_curves(self) -> dict[str, FloatArray]:
        """The influence curves by parameter name."""
        return estimate_curves(self.estimates)

    def covariance(self, names: Sequence[str] | None = None) -> FloatArray:
        """Return the joint covariance of selected estimates.

        Parameters
        ----------
        names : sequence of str or None
            Names in output order.  ``None`` selects all estimates.

        Returns
        -------
        ndarray
            The covariance of the estimators, at the cluster level on a clustered fit.
        """
        return estimate_covariance(self.estimates, names, cluster=self.data.cluster)

    def contrast(
        self,
        function: Callable[[FloatArray], float],
        names: Sequence[str],
        *,
        name: str | None = None,
        scale: Scale = "difference",
        gradient: Callable[[FloatArray], FloatArray] | None = None,
        transform: Transform | None = None,
    ) -> ParameterEstimate:
        """Return a smooth function of estimates with joint delta-method inference.

        Parameters
        ----------
        function : callable
            A smooth scalar function of the selected estimates.
        names : sequence of str
            The names passed to ``function`` in order.
        name : str or None
            The name of the derived estimate.
        scale : str
            The reported scale.
        gradient : callable or None
            The analytic gradient.  ``None`` uses central differences.
        transform : Transform or None
            The map the interval is computed on.

        Returns
        -------
        ParameterEstimate
            The derived estimate, with the inference status of the selected estimates.
        """
        return smooth_contrast(
            self.estimates,
            function,
            names,
            n=self.n,
            cluster=self.data.cluster,
            alpha=self.config.alpha_sig,
            name=name,
            scale=scale,
            gradient=gradient,
            transform=transform,
        )

    def blip(self, node: int | str) -> FloatArray:
        """The blip coefficients of one node, as :math:`\\Psi_t^\\top`.

        Parameters
        ----------
        node : int or str
            The node counted from one, or its treatment column.

        Returns
        -------
        ndarray
            The ``(p_t, d_t)`` matrix: one row per feature and one column per coding component.
            It is the transpose of :math:`\\Psi_t` in the model, which has one row per coding
            component.
        """
        time = node if isinstance(node, int) else self.config.treatment_names.index(node) + 1
        return np.asarray(self.coefficients[time - 1])

    # -------------------------------------------------------------- model check

    def check_model(
        self,
        *,
        q_bounds: tuple[float, float] | None = None,
        rank_floor: float = RANK_FLOOR,
        n_splits: int = 5,
    ) -> StructuralModelCheck:
        """Compare the structural contrasts with the nonparametric contrasts.

        The check fits :class:`~cleverly.longitudinal.LTMLE` on the same rows, weights,
        clusters and folds.  Both estimators are asymptotically linear, so the difference of
        the two contrast vectors is asymptotically normal with the covariance of the
        difference of the two curves.  The mean is zero under the structural model.  The
        joint Wald statistic has one degree of freedom per contrast.  A rejection says that
        the model fails for these contrasts.  A non-rejection does not confirm the model.

        The two contrasts nearly cancel, so the difference has a much smaller variance than
        either estimator.  On a cross-fitted fit, the second-order remainder of each
        estimator depends on the fold split, and the curves do not carry it.  It is
        negligible against either estimator but not against the difference, so at 2 folds a
        check on one split rejects a correct model too often.  The check therefore
        refits both estimators on ``n_splits - 1`` further splits and tests the mean
        difference (Chernozhukov et al. 2018, *Econometrics Journal* 21(1), Section 3.4).
        Given the data, the splits are independent draws, so the between-split covariance
        over ``n_splits`` is the split part of the variance.  The covariance of curves from
        two different splits estimates the first-order part, free of each curve's own split
        noise.  An in-sample fit has no split, and its check reads the curves of the fit.

        The covariance is itself an estimate.  Under weak positivity a few rows carry large
        inverse weights and most of the variance, so the covariance has few effective degrees
        of freedom, and a chi-squared reference rejects a correct model too often.  The check
        computes the Welch-Satterthwaite degrees of freedom ``nu`` of each diagonal entry and
        takes the smallest.  It reads ``T (nu - k + 1) / (nu k)`` against ``F(k, nu - k + 1)``,
        the Hotelling law of a Wald statistic whose covariance has ``nu`` degrees of freedom.
        With ``nu`` at most ``k - 1`` it reports no test.  For large ``nu`` this is the
        chi-squared law.  The correction concerns the reference law only.  When the
        nonparametric estimator under-covers at this sample size, a non-rejection carries
        less information, and ``reference_df`` says how much.

        Parameters
        ----------
        q_bounds : tuple of float or None, default=None
            The known support of a continuous outcome.  A cross-fitted fit of a continuous
            outcome needs it, as :class:`~cleverly.longitudinal.LTMLE` does.
        rank_floor : float, default=1e-6
            The ratio of the smallest eigenvalue of the difference covariance to the largest
            eigenvalue of the two contrast covariances, below which the check reports no
            test.  A feature map that saturates the contrasts makes the two curves nearly
            equal.
        n_splits : int, default=5
            The fold splits of a cross-fitted check, the fit's own included.  It must be at
            least 2, because one split cannot estimate the between-split covariance.  Each
            further split refits both estimators.  An in-sample fit ignores it.

        Returns
        -------
        StructuralModelCheck
            The contrasts, their difference, the covariance and the statistic.

        Raises
        ------
        CapabilityError
            For a continuous node (``INVALID_DECLARATION``), a blip feature that reads a
            time-varying covariate (``IMPLEMENTATION_GAP``), a status that supplies no
            inference (``INFERENCE_GAP``), fewer than two regimens, or a cross-fitted
            continuous outcome with no ``q_bounds``.
        ValueError
            For ``n_splits`` below 2.
        """
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError(f"n_splits must be an integer of at least 2; got {n_splits!r}")
        data = self.data
        if self.config.reads_covariates:
            raise CapabilityError.from_refusal(_COVARIATE_FEATURE_REFUSAL)
        if any(node.continuous for node in self.config.nodes):
            raise _invalid(
                "the model check compares the structural contrast with the nonparametric "
                "contrast, and a continuous static dose has no nonparametric counterpart: it is "
                "not pathwise differentiable in the nonparametric model. The structural "
                "model is what makes the static dose level regular"
            )
        if len(self.static_regimens) < 2:
            raise _invalid(
                "the model check compares contrasts, so it needs at least two static regimens"
            )
        for estimate in self.estimates.values():
            if not estimate.supplies_inference:
                record = status_record(estimate.inference)
                raise CapabilityError(
                    f"the fit takes the {estimate.inference!r} status, so its curve is a "
                    "diagnostic and a Wald statistic over it has no reference law.",
                    reason=RefusalReason.INFERENCE_GAP,
                    alternative=record.alternative or "Cross-fit the structural fit.",
                )
        if data.family != "binomial" and not self.folds.is_single and q_bounds is None:
            raise _invalid(
                "a cross-fitted nonparametric fit of a continuous outcome needs the known "
                "outcome support",
                alternative="Pass q_bounds=(lower, upper) to check_model.",
            )
        names = tuple(name for name, kind in self.kinds.items() if kind == "contrast")
        splits = [self.folds]
        if not self.folds.is_single:
            splits += [
                random_partition(data.n, self.folds.n_folds, cluster=data.cluster, seed=seed)
                for seed in _split_seeds(self.estimator.random_state, n_splits - 1)
            ]
        structural_rows: list[FloatArray] = []
        nonparametric_rows: list[FloatArray] = []
        curves: list[FloatArray] = []
        first: Any = None
        for index, folds in enumerate(splits):
            try:
                structural_fit = self if index == 0 else self._refit(folds)
                fit = self._nonparametric(folds, q_bounds)
            except DataError as error:
                if index == 0:
                    raise
                raise DataError(
                    f"the model check refits both estimators on {len(splits) - 1} further "
                    f"fold splits, and split {index + 1} failed: {error}"
                ) from error
            for name in names:
                if not fit[name].supplies_inference:
                    raise CapabilityError(
                        f"the nonparametric fit takes the {fit[name].inference!r} status, so "
                        "its curve supplies no interval to compare with.",
                        reason=RefusalReason.INFERENCE_GAP,
                        alternative=(
                            "Fit LTMLE with more clusters, or compare the point estimates."
                        ),
                    )
            if index == 0:
                first = fit
            structural_rows.append(np.array([structural_fit[name].psi for name in names]))
            nonparametric_rows.append(np.array([fit[name].psi for name in names]))
            curves.append(
                np.column_stack(
                    [
                        structural_fit[name].influence_curve - fit[name].influence_curve
                        for name in names
                    ]
                )
            )
        structural = np.mean(structural_rows, axis=0)
        nonparametric = np.mean(nonparametric_rows, axis=0)
        cluster = data.cluster
        covariance = _split_covariance(
            np.asarray(structural_rows) - np.asarray(nonparametric_rows), curves, cluster
        )
        scale = max(
            float(
                np.linalg.eigvalsh(
                    influence_covariance(
                        np.column_stack([self[n].influence_curve for n in names]), cluster
                    )
                )[-1]
            ),
            float(
                np.linalg.eigvalsh(
                    influence_covariance(
                        np.column_stack([first[n].influence_curve for n in names]), cluster
                    )
                )[-1]
            ),
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        reason: str | None = None
        statistic = pvalue = reference_df = float("nan")
        if not eigenvalues[0] > rank_floor * scale:
            reason = (
                f"the difference covariance has the smallest eigenvalue {eigenvalues[0]:.3g}, "
                f"below {rank_floor:g} times the contrast covariance scale {scale:.3g}. The "
                "blip features make the structural contrasts nearly equal to the "
                "nonparametric ones, so the difference carries no information"
            )
            if len(splits) > 1:
                reason += (
                    f", or {len(splits)} splits estimate the covariance too coarsely. "
                    "Pass a larger n_splits to tell the two apart"
                )
        else:
            gap = structural - nonparametric
            statistic = float(gap @ np.linalg.solve(covariance, gap))
            k = len(names)
            reference_df = _satterthwaite_df(
                np.asarray(structural_rows) - np.asarray(nonparametric_rows),
                curves,
                cluster,
                covariance,
            )
            pvalue = _wald_pvalue(statistic, k, reference_df)
            if np.isnan(pvalue):
                statistic = float("nan")
                reason = (
                    f"the difference covariance has {reference_df:.2f} Satterthwaite degrees "
                    f"of freedom, not more than {k - 1}, so the Wald statistic over {k} "
                    "contrasts has no reference law. A few rows carry most of the variance, "
                    "which weak positivity of a regimen causes"
                )
        return StructuralModelCheck(
            names=names,
            structural=structural,
            nonparametric=nonparametric,
            covariance=covariance,
            statistic=statistic,
            df=len(names),
            pvalue=pvalue,
            reason=reason,
            ltmle=first,
            n_splits=len(splits),
            reference_df=reference_df,
            support=combine_support(
                [
                    *(self[name].support for name in names),
                    *(first[name].support for name in names),
                ]
            ),
        )

    def _refit(self, folds: Folds) -> StructuralNestedMeanResult:
        """Refit the structural estimator on these rows with another fold split."""
        settings = copy.copy(self.estimator)
        # The check reads the curves and the estimates; a band would be discarded.
        settings.simultaneous = False
        settings._folds = lambda _data, _seed=None: folds  # type: ignore[method-assign, misc]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return settings.fit(self.data)

    def _nonparametric(self, folds: Folds, q_bounds: tuple[float, float] | None) -> Any:
        """Fit LTMLE for the static regimens on these rows with this fold split."""
        settings = self.estimator
        nonparametric_fit = LTMLE(
            [Regimen(r.label, r.values) for r in self.static_regimens],
            reference=self.config.reference,
            outcome_learner=settings.outcome_learner,
            treatment_learner=settings.treatment_learner,
            n_folds=folds.n_folds,
            learner_folds=settings.learner_folds,
            q_bounds=q_bounds,
            alpha_sig=settings.alpha_sig,
            simultaneous=False,
            random_state=settings.random_state,
            n_jobs=settings.n_jobs,
        )
        # The split is the structural fit's own, so both estimators hold out the same rows.
        nonparametric_fit._folds = lambda _data, _seed=None: folds  # type: ignore[assignment]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nonparametric_fit.fit(self.data)

    # ----------------------------------------------------------------- reports

    def save(self, path: Any) -> Any:
        """Persist the fitted result to a trusted joblib artifact.

        The artifact records :data:`cleverly.__version__`.  A load by a different version
        warns with :class:`~cleverly.exceptions.VersionMismatchWarning`.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path.
        """
        from ..estimators.serialize import save as _save

        return _save(self, path)

    def to_frame(self) -> Any:
        """One row per reported parameter, in the backend the data came from.

        Returns
        -------
        dataframe
            The columns of a point-treatment report, plus ``kind`` (``blip``, ``reference``,
            ``level`` or ``contrast``) and ``model``, the structural condition under which the
            interval is valid.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        keys = list(dict.fromkeys(key for row in rows for key in row))
        for row, estimate in zip(rows, self.estimates.values(), strict=True):
            if "inference" in keys:
                row.setdefault("inference", estimate.inference)
        payload: dict[str, list[Any]] = {
            key: [row.get(key, np.nan) for row in rows] for key in keys
        }
        payload["kind"] = [self.kinds[name] for name in self.estimates]
        payload["model"] = [self.label for _ in self.estimates]
        return self.data.frame_like(payload)

    def summary(self) -> str:
        """A printable report: the estimates, the model condition, then the settings.

        Returns
        -------
        str
            The table, the structural condition, the settings and the diagnostics.
        """
        level = f"{(1 - self.config.alpha_sig) * 100:g}%"
        status = self.inference_status
        record = None if supplies_inference(status) else status_record(status)
        t_reference = has_t_reference(self.estimates)
        support_cells = support_column(self.estimates)
        rows = []
        for name, estimate in self.estimates.items():
            if record is not None:
                rows.append([name, f"{estimate.psi:.4f}", f"{estimate.plugin_std_error:.4f}"])
                continue
            low, high = estimate.ci
            rows.append(
                [
                    name,
                    f"{estimate.psi:.4f}",
                    f"{estimate.std_error:.4f}",
                    f"[{low:.4f}, {high:.4f}]",
                    format_pvalue(estimate.pvalue),
                    *([reference_label(estimate)] if t_reference else []),
                ]
            )
        if record is not None:
            headers = ["parameter", "estimate", record.summary_label]
        else:
            headers = [
                "parameter",
                "estimate",
                "std. error",
                f"{level} CI",
                "p-value",
                *(["df"] if t_reference else []),
            ]
        if support_cells is not None:
            headers.append("support")
            rows = [[*row, cell] for row, cell in zip(rows, support_cells, strict=True)]
        table = format_table(headers, rows)
        facts = list(self.config.describe())
        if self.data.cluster is not None:
            count = f"{self.data.n_clusters}" + positive_mass_clause(
                self.data.cluster, self.data.weights if self.data.is_weighted else None
            )
            facts.append(f"clusters = {count} ({self.data.cluster_name}, cluster-robust variance)")
        if self.data.is_weighted:
            report = self.data.weight_report()
            facts.append(
                f"observation weights ({report.name or 'weights'}, "
                + ("estimated" if report.estimated else "fixed")
                + f"): effective n = {report.effective_n:.1f}, "
                f"design effect = {report.design_effect:.2f}"
            )
        smallest = min(self.diagnostics.smallest_eigenvalue)
        shares = [s for s in self.diagnostics.extreme_share if not np.isnan(s)]
        facts.append(f"smallest standardized Jacobian eigenvalue = {smallest:.3g}")
        if shares:
            facts.append(
                f"largest share of rows with a class probability within "
                f"{self.config.positivity_margin:g} of 0 or 1 = {max(shares):.1%}"
            )
        facts.extend(self.provenance.describe())
        lines = [
            f"Structural nested mean model ({self.data.n_times} time points, n = {self.n})",
            f"The intervals are {self.label}.",
            "",
            table,
            *(["", *support_notes(self.estimates)] if support_cells is not None else []),
            *(["", record.summary_note()] if record is not None else []),
            "",
            *(f"  {line}" for line in facts),
        ]
        if self.simultaneous is None and record is not None and len(self.estimates) > 1:
            lines.extend(["", f"  {NO_SIMULTANEOUS_BANDS}"])
        elif self.simultaneous is None and t_reference and len(self.estimates) > 1:
            lines.extend(["", f"  {NO_T_REFERENCE_BANDS}"])
        if self.simultaneous is not None:
            band_marker = (
                " [experimental]" if self.simultaneous.support.label == "experimental" else ""
            )
            lines.extend(
                [
                    "",
                    f"  simultaneous {level} bands (multiplier bootstrap, critical value "
                    f"{self.simultaneous.critical_value:.3f} vs "
                    f"{self.simultaneous.pointwise_critical_value:.3f} pointwise){band_marker}:",
                ]
            )
            for name, (low, high) in self.simultaneous.bands.items():
                lines.append(f"    {name}  [{low:.4f}, {high:.4f}]")
        if t_reference and self.data.cluster is not None:
            positive = fewest_clusters(
                self.data.cluster, weights=self.data.weights if self.data.is_weighted else None
            )
            lines.append(f"  clusters with positive mass = {positive}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StructuralNestedMeanResult({', '.join(self.estimates)})"


# ---------------------------------------------------------------------- one-call form


def snmm(
    data: Any,
    *,
    outcome: str,
    treatment: Sequence[str],
    baseline: Sequence[str],
    regimens: Any = None,
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    family: str = "auto",
    continuous_treatment: Sequence[str] = (),
    **kwargs: Any,
) -> StructuralNestedMeanResult:
    """One-call structural nested mean model, mirroring :func:`cleverly.longitudinal.ltmle`.

    Parameters
    ----------
    data : DataFrame or LongitudinalData
        One wide frame, or a container that has already read the columns.
    outcome : str
        The end-of-study outcome column.
    treatment : sequence of str
        The treatment column of each node, in time order.
    baseline : sequence of str
        The covariates measured before any treatment.
    regimens : mapping, sequence, or None
        Static regimens, as for :class:`StructuralNestedMean`.
    time_varying : sequence of sequence of str or None
        The covariates measured before the treatment of each node.
    censoring : sequence of str or None
        Refused when it names a column.
    id : str or None
        A cluster column.
    weights : str or None
        A column of known observation weights.
    weights_type : str
        How to read ``weights``.
    weights_estimated : bool
        Declare that the weights came out of a fitted model.
    family : {"auto", "binomial", "gaussian"}
        The outcome family.
    continuous_treatment : sequence of str
        The treatment columns that hold a dose.
    **kwargs : Any
        The remaining :class:`StructuralNestedMean` settings.

    Returns
    -------
    StructuralNestedMeanResult
        The blips, the reference mean, and the declared regimen means and contrasts.

    See Also
    --------
    StructuralNestedMean : The estimator class and its settings.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.learners import GLM
    >>> from cleverly.longitudinal import snmm
    >>> frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    >>> result = snmm(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     outcome_learner=GLM(family="binomial"),
    ...     treatment_learner=GLM(family="binomial"),
    ...     n_folds=2,
    ...     random_state=0,
    ... )
    >>> "ey_reference" in result
    True
    """
    return StructuralNestedMean(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
        continuous_treatment=continuous_treatment,
    )
