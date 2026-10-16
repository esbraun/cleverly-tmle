"""The support label, the key a fit is classified by, and the phase-1 record switch.

A fit is *supported* when a registered study covers its configuration, and *experimental*
otherwise. This module holds the pieces of that decision that do not depend on any fitted
estimate: the :class:`Support` record, the :class:`SupportKey` read from an estimator and its
data before any learner runs, :func:`classify`, which matches a key against the registered rows, and
:func:`combine_support`, which says what a derived estimate carries.

A fit classifies itself before any learner runs. :func:`support_for` then computes what each
estimate carries from the classification, the estimate's final status and its reference. The
record switch is a measurement aid. When the environment variable
:data:`RECORD_ENVIRONMENT_VARIABLE` names a directory, each classification and each refusal site
that a later change will move appends one JSON line to ``<directory>/<pid>.jsonl``. The switch
changes no output.

The module imports the standard library, :mod:`cleverly._refusal` and
:mod:`cleverly._inference_status` at module level. The key readers import the cluster counter
inside the function, so the estimate classes can import this module without a cycle.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from ._inference_status import (
    FEW_CLUSTER_THRESHOLD,
    MINIMUM_INTERVAL_CLUSTERS,
    MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS,
    STATUS_RECORDS,
    supplies_inference,
)
from ._refusal import Refusal

__all__ = [
    "KEY_AXES",
    "RECORD_ENVIRONMENT_VARIABLE",
    "REGIMEN_KINDS",
    "UNCLASSIFIED",
    "Classification",
    "Kind",
    "Preflight",
    "Support",
    "SupportKey",
    "SupportLabel",
    "SupportRow",
    "classify",
    "classify_key",
    "classify_longitudinal_fit",
    "classify_point_fit",
    "classify_structural_fit",
    "combine_support",
    "longitudinal_key",
    "point_key",
    "record_fit",
    "record_refusal_site",
    "recording_directory",
    "regimen_kinds",
    "registered_rows",
    "structural_key",
    "support_for",
]

#: The environment variable that turns the phase-1 record on. Its value is a directory.
RECORD_ENVIRONMENT_VARIABLE: Final[str] = "CLEVERLY_SUPPORT_RECORD"

SupportLabel = Literal["supported", "experimental"]

#: What a support row can license for an estimate: a point estimate, a Wald interval on the
#: normal or the t reference, a simultaneous band, a percentile bootstrap interval, a BCa
#: bootstrap interval, a studentized bootstrap sup-t band, or a derived estimate.
#: A derived estimate is a ratio, a contrast, a restricted mean or an E-value built from
#: estimates of one fit. No row lists a hypothesis-test kind, because a test statistic is the
#: ``wald`` interval read against a null and carries the support of that interval.
Kind = Literal[
    "point", "wald", "wald_t", "band", "bootstrap", "bootstrap_bca", "bootstrap_band", "derived"
]


@dataclass(frozen=True)
class Support:
    """What the package states about the evidence behind one estimate.

    Parameters
    ----------
    label : {"supported", "experimental"}
        ``"supported"`` when a registered study covers the configuration.
    reason : str
        Whole sentences that say why.
    evidence : tuple of str
        Study slugs when supported, and roadmap ids when experimental.
    row : str or None
        The id of the support row that matched, or ``None``.
    limits : tuple of str
        The stated limits of the row and its assumptions.
    """

    label: SupportLabel
    reason: str
    evidence: tuple[str, ...] = ()
    row: str | None = None
    limits: tuple[str, ...] = ()


#: The support of an estimate built outside a classified fit.
UNCLASSIFIED: Final[Support] = Support(
    "experimental", "This estimate was built outside a classified fit."
)

_TWO_ROWS_REASON: Final[str] = "This estimate combines estimates of two support rows."


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    """The distinct items in first-seen order."""
    return tuple(dict.fromkeys(items))


def combine_support(supports: Iterable[Support], kind: Kind | None = None) -> Support:
    """The support of an estimate built from several estimates.

    Equal records stay as they are. When any input is experimental, the result is
    experimental and joins the reasons and the evidence of every experimental input, in first-seen
    order. Two supported inputs of different rows give an experimental result. With ``kind``,
    every row an input names must also license that kind, and each row that does not adds its
    reason and makes the result experimental.

    Parameters
    ----------
    supports : iterable of Support
        The support of each input estimate. At least one is required.
    kind : str or None, default None
        A member of :data:`Kind`: the kind the new estimate needs from the rows of its
        inputs. A derived estimate passes ``"derived"``. ``None`` combines the inputs only.

    Returns
    -------
    Support
        The combined record.

    Raises
    ------
    ValueError
        When ``supports`` is empty.

    Examples
    --------
    >>> from cleverly._support import Support, combine_support
    >>> a = Support("supported", "A study covers it.", ("study-a",), row="R1")
    >>> combine_support([a, a]) == a
    True
    >>> combine_support([a, Support("supported", "Covered.", ("study-b",), row="R2")]).label
    'experimental'
    """
    records = tuple(supports)
    if not records:
        raise ValueError("combine_support needs at least one Support")
    combined = _combine(records)
    if kind is None:
        return combined
    rows = {row.id: row for row in registered_rows()}
    reasons: list[str] = []
    for row_id in _unique(record.row for record in records if record.row is not None):
        row = rows.get(row_id)
        if row is None:
            continue
        if kind not in row.kinds:
            reasons.append(f"Row {row.id} licenses no {_KIND_NAMES[kind]} for this estimate.")
        else:
            reasons.extend(
                text for shortfall_kind, text in row.shortfalls if shortfall_kind == kind
            )
    if not reasons:
        return combined
    return Support(
        "experimental",
        " ".join(
            _unique([*(() if combined.label == "supported" else (combined.reason,)), *reasons])
        ),
        combined.evidence if combined.label == "experimental" else (),
        limits=combined.limits,
    )


def _combine(records: tuple[Support, ...]) -> Support:
    """The combination of :func:`combine_support` without a kind check."""
    first = records[0]
    if all(record == first for record in records[1:]):
        return first
    experimental = [record for record in records if record.label == "experimental"]
    if experimental:
        return Support(
            "experimental",
            " ".join(_unique(record.reason for record in experimental)),
            _unique(item for record in experimental for item in record.evidence),
            limits=_unique(item for record in records for item in record.limits),
        )
    return Support(
        "experimental",
        _TWO_ROWS_REASON,
        limits=_unique(item for record in records for item in record.limits),
    )


@dataclass(frozen=True)
class Preflight:
    """What the configuration checks of a fit return before any learner runs.

    A group-A rule appends its :class:`~cleverly._refusal.Refusal` to ``conditions`` and the
    checks go on, so a condition never hides a later refusal. A refusal of any other group
    raises as before.

    Parameters
    ----------
    estimands : tuple of str
        The resolved estimand names.
    conditions : tuple of Refusal, default ()
        The group-A conditions the fit meets. Each one makes the fit experimental.
    """

    estimands: tuple[str, ...]
    conditions: tuple[Refusal, ...] = ()


# ----------------------------------------------------------------------- the key

#: The axes of a key, in the order a nearest-row report names them.
KEY_AXES: Final[tuple[str, ...]] = (
    "kind",
    "engine",
    "family",
    "outcome",
    "missing",
    "treatment",
    "weighting",
    "splitting",
    "dependence",
    "qualifiers",
    "options",
)


@dataclass(frozen=True)
class SupportKey:
    """The configuration of a fit, read before any learner runs.

    Every field is hashable, so two keys compare by value. The learner classes are not part
    of the key. A row names them separately, and :class:`Classification` carries them.

    Parameters
    ----------
    kind : {"point", "longitudinal"}
        The estimator family.
    engine : tuple of str
        The estimation procedure: the assessment method and, for C-TMLE, its strategy and design.
        A longitudinal key holds the recursion name.
    family : frozenset of str
        The target families the fit reports, as the ids ``P1`` to ``P11`` of the support
        inventory. A row covers a fit when its family set holds every family of the fit. A
        longitudinal key holds the outcome kind, the working-model flag and the regimen-kind
        flags.
    outcome : tuple of str
        The outcome family and whether a continuous outcome declares its bounds.
    missing : tuple of str
        The missing-data structure: ``"complete"`` or the names of what is missing, and
        ``"intermediate"`` for a declared intermediate.
    treatment : tuple of str
        The treatment kind and arm count, or ``vector`` for a vector with a dose. A
        longitudinal key holds the level count of each node, led by ``held`` when one baseline
        treatment is held over every node.
    weighting : {"none", "fixed", "estimated"}
        How the observation weights are read.
    splitting : tuple of str
        ``in_sample``, ``stacked``, ``fold_evaluated``, ``fold_targeted``,
        ``fold_targeted_evaluated`` or ``cross_fitted``, followed by ``repeated`` and
        ``supplied_plan`` where they apply.
    dependence : {"iid", "clustered", "t", "below_floor"}
        ``clustered`` from 40 clusters with positive weight mass, ``t`` from the interval floor
        to 39, ``below_floor`` under the floor. The floor is 10 for a point fit and 20 for a
        longitudinal one.
    qualifiers : frozenset of str
        ``strata`` and ``known_mechanism``. A longitudinal key holds the known factors.
    options : tuple of tuple
        Sorted ``(name, value)`` pairs of the options that change the estimator's arithmetic. A
        point key on the policy axis holds the policy density ``ratio``. A longitudinal key
        holds ``multiplier_kind``, the inverse-probability ``normalization``, the policy density
        ``ratio``, the ``survival_model``, the hazard ``targeting`` algorithm and the
        ``isotonic`` projection.
    """

    kind: str
    engine: tuple[str, ...]
    family: frozenset[str]
    outcome: tuple[str, ...]
    missing: tuple[str, ...]
    treatment: tuple[str, ...]
    weighting: str
    splitting: tuple[str, ...]
    dependence: str
    qualifiers: frozenset[str]
    options: tuple[tuple[str, Any], ...]

    def axes(self) -> dict[str, Any]:
        """The value of each axis, by name, in :data:`KEY_AXES` order.

        Returns
        -------
        dict of str to Any
            The axis values. A frozenset appears as a sorted tuple, so the mapping is stable.
        """
        values: dict[str, Any] = {}
        for name in KEY_AXES:
            value = getattr(self, name)
            values[name] = tuple(sorted(value)) if isinstance(value, frozenset) else value
        return values

    def is_covered_by(self, row_key: SupportKey) -> bool:
        """Whether ``row_key`` covers this key.

        Every axis must be equal, except the family axis: the row covers the fit when the row's
        families include every family of the fit.

        Parameters
        ----------
        row_key : SupportKey
            The key of a support row.

        Returns
        -------
        bool
            ``True`` when the fit falls inside the row.
        """
        return not self.differing_axes(row_key)

    def differing_axes(self, other: SupportKey) -> tuple[str, ...]:
        """The names of the axes on which this key is not covered by ``other``.

        The family axis differs when this key holds a family that ``other`` lacks. Every other
        axis differs when the values are unequal.

        Parameters
        ----------
        other : SupportKey
            The key of a support row.

        Returns
        -------
        tuple of str
            The axis names, in :data:`KEY_AXES` order.
        """
        mine, theirs = self.axes(), other.axes()
        differing = []
        for name in KEY_AXES:
            if name == "family":
                if not self.family <= other.family:
                    differing.append(name)
            elif mine[name] != theirs[name]:
                differing.append(name)
        return tuple(differing)

    def to_json(self) -> dict[str, Any]:
        """The key as a JSON-ready mapping.

        Returns
        -------
        dict of str to Any
            Each axis as a string, a list, or a list of pairs.
        """
        return json.loads(json.dumps(self.axes(), default=_jsonable))


def _jsonable(value: Any) -> Any:
    """Serialise a value a key can hold that the json module does not."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _dependence(clusters: int | None, floor: int) -> str:
    """The dependence tier of a fit that reads ``clusters`` positive-mass clusters."""
    if clusters is None:
        return "iid"
    if clusters >= FEW_CLUSTER_THRESHOLD:
        return "clustered"
    if clusters >= floor:
        return "t"
    return "below_floor"


def _positive_mass_clusters(data: Any) -> int | None:
    """The fewest positive-mass clusters one estimate reads, or ``None`` without clusters."""
    cluster = getattr(data, "cluster", None)
    if cluster is None:
        return None
    from .inference.cluster import fewest_clusters

    return int(fewest_clusters(cluster, getattr(data, "strata", None), data.weights))


def _weighting(data: Any) -> str:
    """``"estimated"``, ``"fixed"`` or ``"none"``, as the data declares its weights."""
    if not data.is_weighted:
        return "none"
    return "estimated" if data.weight_spec.estimated else "fixed"


def _option(value: Any) -> Any:
    """A hashable stand-in for an option value."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_option(item) for item in value)
    return repr(value)


def _learner_type(learner: Any) -> type | None:
    """The class of a learner, or ``None`` when the estimator uses its default."""
    return None if learner is None else type(learner)


#: The target family of each arm-axis estimand, as the ids of the support inventory.
_ARM_FAMILIES: Final[Mapping[str, str]] = {
    "ey": "P1",
    "ey1": "P1",
    "ey0": "P1",
    "ate": "P1",
    "rr": "P2",
    "or": "P2",
    "att": "P3",
    "atc": "P3",
    "ey_obs": "P4",
    "par": "P4",
    "paf": "P4",
}

#: The target family of each other parameter axis. The regime axis is ``P5`` unless a
#: regime declares a known stochastic density, which makes it ``P6``.
_AXIS_FAMILIES: Final[Mapping[str, str]] = {
    "regime": "P5",
    "policy": "P7",
    "rr_tilt": "P8",
    "ipsi": "P9",
    "msm": "P10",
    "learned_rule": "P11",
}


def _point_families(estimator: Any, estimands: Sequence[str]) -> frozenset[str]:
    """The target families of a point fit: its parameter axis and, on the arm axis, its names."""
    axis = str(estimator._axis)
    if axis == "arm":
        return frozenset(_ARM_FAMILIES.get(str(name), f"?{name}") for name in estimands)
    if axis == "regime" and any(
        type(item).__name__ == "Stochastic" for item in estimator.interventions
    ):
        return frozenset({"P6"})
    return frozenset({_AXIS_FAMILIES[axis]})


def _outcome(data: Any, q_bounds: Any) -> tuple[str, ...]:
    """The outcome family, and for a non-binary outcome whether its support is declared."""
    if str(data.family) == "binomial":
        return ("binomial",)
    return (str(data.family), "bounded" if q_bounds is not None else "unbounded")


def point_key(
    estimator: Any, data: Any, estimands: Sequence[str]
) -> tuple[SupportKey, dict[str, type | None]]:
    """The key and the learner classes of a point-treatment fit.

    Parameters
    ----------
    estimator : TMLE
        The estimator, as constructed. It is read and never modified.
    data : CausalData
        The prepared data.
    estimands : sequence of str
        The resolved estimand names the fit reports.

    Returns
    -------
    tuple of SupportKey and dict
        The key, and the learner class per nuisance role (``None`` for a default learner).
    """
    method = str(getattr(estimator, "_assessment_method", type(estimator).__name__))
    engine: tuple[str, ...] = (method,)
    if method == "collaborative_tmle":
        engine += (str(estimator.strategy), str(estimator.oat_design))
        from .estimators.ctmle import per_arm_parametric_support

        if per_arm_parametric_support(estimator, data).admitted:
            # The stacked curve of X58 is another estimator of the interval, so it is its own
            # engine: a row for it never matches a per-arm fit that keeps the shipped curve.
            engine += ("parametric_stack",)

    if estimator.cross_fit:
        folded = estimator.targeting_scheme != "pooled"
        if estimator.cv_evaluation:
            scheme = "fold_targeted_evaluated" if folded else "fold_evaluated"
        else:
            scheme = "fold_targeted" if folded else "stacked"
    else:
        scheme = "in_sample"
    splitting = [scheme]
    if estimator.repeats != 1:
        splitting.append("repeated")
    if estimator.split_plan is not None:
        splitting.append("supplied_plan")

    missing: list[str] = []
    if data.has_missing_outcome:
        missing.append("outcome")
    if data.has_missing_treatment:
        missing.append("treatment")
    if data.has_intermediate:
        missing.append("intermediate")

    if data.treatment_kind == "vector":
        # A vector with a dose has no arms, and its one-column readers refuse (X31).
        treatment: tuple[str, ...] = ("vector",)
    elif data.is_continuous_treatment:
        treatment = ("continuous",)
    else:
        treatment = ("binary",) if data.n_arms == 2 else (f"arms{data.n_arms}",)

    qualifiers = set()
    if data.has_strata:
        qualifiers.add("strata")
    if data.known_treatment is not None:
        qualifiers.add("known_mechanism")

    options: dict[str, Any] = {
        "fluctuation": estimator.fluctuation,
        "targeting": estimator.targeting,
        "target_weights": estimator.target_weights,
        "multiplier_kind": estimator.multiplier_kind,
    }
    if method == "drtmle":
        for name in ("guard", "reduction", "reduced_crossfit", "update_order", "randomized"):
            options[name] = getattr(estimator, name)
        options["evaluation"] = estimator.evaluation is not None
    if str(estimator._axis) == "policy":
        # The policy density ratio changes the clever covariate, so a classifier fit never
        # matches a row whose study fitted the density route.
        options["ratio"] = estimator.ratio

    key = SupportKey(
        kind="point",
        engine=engine,
        family=_point_families(estimator, estimands),
        outcome=_outcome(data, estimator.q_bounds),
        missing=tuple(missing) or ("complete",),
        treatment=treatment,
        weighting=_weighting(data),
        splitting=tuple(splitting),
        dependence=_dependence(_positive_mass_clusters(data), MINIMUM_INTERVAL_CLUSTERS),
        qualifiers=frozenset(qualifiers),
        options=tuple(sorted((name, _option(value)) for name, value in options.items())),
    )
    learners = {
        "outcome": _learner_type(estimator.outcome_learner),
        "treatment": _learner_type(estimator.treatment_learner),
        "missingness": _learner_type(estimator.missingness_learner),
        "intermediate": _learner_type(estimator.intermediate_learner),
    }
    return key, learners


#: The regimen kinds of a longitudinal key: a static or dynamic regimen, a known stochastic
#: policy, a modified treatment policy and an odds tilt.
REGIMEN_KINDS: Final[tuple[str, ...]] = ("static", "stochastic", "mtp", "tilt")


def regimen_kinds(plans: Iterable[Any]) -> frozenset[str]:
    """The kinds of regimen in resolved longitudinal plans.

    Parameters
    ----------
    plans : iterable of RegimenPlan
        The resolved plans. Each one is read for ``has_mtp``, ``has_tilt`` and ``has_policy``.

    Returns
    -------
    frozenset of str
        Members of :data:`REGIMEN_KINDS`. A plan with no policy node is ``static``. A plan
        with a policy node and no modified treatment policy or tilt is ``stochastic``.
    """
    kinds: set[str] = set()
    for plan in plans:
        mtp, tilt = bool(plan.has_mtp), bool(plan.has_tilt)
        if mtp:
            kinds.add("mtp")
        if tilt:
            kinds.add("tilt")
        if not (mtp or tilt):
            kinds.add("stochastic" if plan.has_policy else "static")
    return frozenset(kinds)


def longitudinal_key(
    estimator: Any,
    data: Any,
    *,
    outcome_kind: str,
    cross_fitted: bool,
    known_factors: Iterable[str],
    has_msm: bool,
    regimen_kinds: Iterable[str],
) -> tuple[SupportKey, dict[str, type | None]]:
    """The key and the learner classes of a longitudinal fit.

    Parameters
    ----------
    estimator : LTMLE
        The estimator. It is read and never modified.
    data : LongitudinalData
        The prepared panel.
    outcome_kind : str
        ``"end_of_study"``, ``"survival"``, ``"competing"`` or ``"repeated"``.
    cross_fitted : bool
        Whether the outer split holds more than one fold.
    known_factors : iterable of str
        The treatment and censoring columns whose factor the data declares known.
    has_msm : bool
        Whether the fit reports working-model coefficients.
    regimen_kinds : iterable of str
        The kinds of regimen the fit reports: members of :data:`REGIMEN_KINDS`.

    Returns
    -------
    tuple of SupportKey and dict
        The key, and the learner class per nuisance role (``None`` for a default learner).
    """
    levels = tuple(
        kind if kind != "categorical" else str(len(node))
        for kind, node in (
            (data.node_kind(time), node) for time, node in enumerate(data.treatment_levels, 1)
        )
    )
    key = SupportKey(
        kind="longitudinal",
        engine=(str(estimator._estimator),),
        family=frozenset(
            (
                outcome_kind,
                "msm" if has_msm else "regimens",
                *regimen_kinds,
            )
        ),
        outcome=_outcome(data, estimator.q_bounds),
        missing=("censoring",) if data.has_censoring else ("complete",),
        treatment=("held", *levels) if data.treatment_held else levels,
        weighting=_weighting(data),
        splitting=("cross_fitted",) if cross_fitted else ("in_sample",),
        dependence=_dependence(
            _positive_mass_clusters(data), MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS
        ),
        qualifiers=frozenset(known_factors),
        options=(
            ("multiplier_kind", _option(estimator.multiplier_kind)),
            ("normalization", _option(estimator._normalization())),
            ("ratio", _option(getattr(estimator, "ratio", "density"))),
            ("survival_model", _option(estimator.survival_model)),
            ("targeting", _option(estimator.targeting)),
            ("isotonic", _option(estimator.isotonic)),
        ),
    )
    learners = {
        "outcome": _learner_type(estimator.outcome_learner),
        "pseudo": _learner_type(estimator.pseudo_learner),
        "treatment": _learner_type(estimator.treatment_learner),
        "censoring": _learner_type(estimator.censoring_learner),
    }
    return key, learners


def structural_key(
    estimator: Any,
    data: Any,
    *,
    cross_fitted: bool,
    has_regimens: bool,
    covariate_features: bool,
) -> tuple[SupportKey, dict[str, type | None]]:
    """The key and the learner classes of a structural nested mean fit.

    The key reads the axes of a longitudinal key. The engine is ``("snmm",)``, and the family
    holds the outcome kind ``end_of_study``, the flag ``blips`` and, with a declared regimen,
    the flags ``regimens`` and ``static``. The estimator refuses censoring, a held decision, a
    declared mechanism and a working model before it classifies, so the missing and qualifier
    axes are fixed.

    Parameters
    ----------
    estimator : StructuralNestedMean
        The estimator. It is read and never modified.
    data : LongitudinalData
        The prepared panel.
    cross_fitted : bool
        Whether the outer split holds more than one fold.
    has_regimens : bool
        Whether the fit declares static regimens.
    covariate_features : bool
        Whether a blip feature reads a time-varying covariate.

    Returns
    -------
    tuple of SupportKey and dict
        The key, and the learner class per nuisance role (``None`` for a default learner).
    """
    levels = tuple(
        kind if kind != "categorical" else str(len(node))
        for kind, node in (
            (data.node_kind(time), node) for time, node in enumerate(data.treatment_levels, 1)
        )
    )
    key = SupportKey(
        kind="longitudinal",
        engine=("snmm",),
        family=frozenset(
            ("end_of_study", "blips", *(("regimens", "static") if has_regimens else ()))
        ),
        outcome=_outcome(data, None),
        missing=("complete",),
        treatment=levels,
        weighting=_weighting(data),
        splitting=("cross_fitted",) if cross_fitted else ("in_sample",),
        dependence=_dependence(
            _positive_mass_clusters(data), MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS
        ),
        qualifiers=frozenset(),
        options=(
            ("covariate_features", bool(covariate_features)),
            ("multiplier_kind", _option(estimator.multiplier_kind)),
        ),
    )
    learners = {
        "outcome": _learner_type(estimator.outcome_learner),
        "treatment": _learner_type(estimator.treatment_learner),
        "regression": _learner_type(estimator.regression_learner),
    }
    return key, learners


# ------------------------------------------------------------------ rows and classify


@dataclass(frozen=True)
class SupportRow:
    """One configuration a registered study covers.

    Parameters
    ----------
    id : str
        The row id.
    key : SupportKey
        The configuration, as one explicit key.
    kinds : frozenset of str
        What the row licenses: members of :data:`Kind`.
    evidence : tuple of str
        Registered study slugs.
    cells : tuple of str
        The licensing cells or truth rows.
    learners : mapping of str to type or None, default None
        The exact learner class per nuisance role, where the row needs one.
    limits : tuple of str, default ()
        The stated limits of the row.
    shortfalls : tuple of tuple of str, default ()
        Pairs of a kind and a sentence. The sentence states the measured number of a red
        cell inside the configuration of the row. A fit that reads the kind from this row is
        experimental, and the sentence is a reason of its label.
    kind_families : tuple of tuple, default ()
        Pairs of a kind and the target families that kind covers. A kind with no pair covers
        every family of the row. A fit that holds a family outside the pair reads the kind as
        unlicensed.
    """

    id: str
    key: SupportKey
    kinds: frozenset[str]
    evidence: tuple[str, ...]
    cells: tuple[str, ...] = ()
    learners: Mapping[str, type] | None = None
    limits: tuple[str, ...] = ()
    shortfalls: tuple[tuple[str, str], ...] = ()
    kind_families: tuple[tuple[str, frozenset[str]], ...] = ()

    def licenses(self, kind: str, families: frozenset[str]) -> bool:
        """Whether the row licenses ``kind`` for a fit that holds ``families``.

        Parameters
        ----------
        kind : str
            A member of :data:`Kind`.
        families : frozenset of str
            The target families of the fit.

        Returns
        -------
        bool
            ``True`` when the row lists the kind and the kind covers every family of the fit.
        """
        if kind not in self.kinds:
            return False
        covered = dict(self.kind_families).get(kind)
        return covered is None or families <= covered

    def matches(self, key: SupportKey, learners: Mapping[str, type | None]) -> bool:
        """Whether a fit with ``key`` and ``learners`` falls inside this row.

        Parameters
        ----------
        key : SupportKey
            The key of the fit.
        learners : mapping of str to type or None
            The learner class per nuisance role. A role the row names compares with ``is``.

        Returns
        -------
        bool
            ``True`` when the row covers the key and every learner the row names is the fit's.
        """
        if not key.is_covered_by(self.key):
            return False
        return all(learners.get(role) is cls for role, cls in (self.learners or {}).items())


def registered_rows() -> tuple[SupportRow, ...]:
    """The registered support rows.

    Returns
    -------
    tuple of SupportRow
        The point-treatment and longitudinal rows of :mod:`cleverly._support_rows`.
    """
    from ._support_rows import LONGITUDINAL_ROWS, POINT_ROWS

    return (*POINT_ROWS, *LONGITUDINAL_ROWS)


@dataclass(frozen=True)
class Classification:
    """The outcome of matching one fit against the rows.

    Parameters
    ----------
    key : SupportKey
        The key of the fit.
    learners : mapping of str to type or None
        The learner class per nuisance role.
    row : SupportRow or None
        The matching row, or ``None``.
    nearest : tuple of str and tuple of str, or None
        With no match, the id of the row that differs on the fewest axes and the names of those
        axes. ``None`` when there are no rows.
    conditions : tuple of Refusal
        The group-A conditions the fit meets. Each one makes the fit experimental.
    """

    key: SupportKey
    learners: Mapping[str, type | None] = field(default_factory=dict)
    row: SupportRow | None = None
    nearest: tuple[str, tuple[str, ...]] | None = None
    conditions: tuple[Refusal, ...] = ()

    @property
    def label(self) -> SupportLabel:
        """``"supported"`` when a row matched and no condition applies."""
        return "supported" if self.row is not None and not self.conditions else "experimental"


def classify_key(
    key: SupportKey,
    learners: Mapping[str, type | None],
    conditions: Sequence[Refusal] = (),
    *,
    rows: Sequence[SupportRow] | None = None,
) -> Classification:
    """Match a key against the rows. It never raises for an unmatched key.

    Parameters
    ----------
    key : SupportKey
        The key of the fit.
    learners : mapping of str to type or None
        The learner class per nuisance role.
    conditions : sequence of Refusal, default ()
        The group-A conditions the fit meets.
    rows : sequence of SupportRow or None, default None
        The rows to match. ``None`` reads :func:`registered_rows`.

    Returns
    -------
    Classification
        The matching row, or the nearest row and its differing axes.

    Raises
    ------
    RuntimeError
        When two rows match. The row tests make this unreachable.
    """
    table = registered_rows() if rows is None else tuple(rows)
    matched = [row for row in table if row.matches(key, learners)]
    if len(matched) > 1:
        raise RuntimeError(f"support rows {[row.id for row in matched]} all match one fit")
    if matched:
        return Classification(key, learners, matched[0], None, tuple(conditions))
    nearest: tuple[str, tuple[str, ...]] | None = None
    for row in table:
        differing = key.differing_axes(row.key)
        if (
            learners
            and row.learners
            and any(learners.get(role) is not cls for role, cls in row.learners.items())
        ):
            differing = (*differing, "learners")
        if nearest is None or len(differing) < len(nearest[1]):
            nearest = (row.id, differing)
    return Classification(key, learners, None, nearest, tuple(conditions))


_KIND_NAMES: Final[Mapping[str, str]] = {
    "point": "point estimate",
    "wald": "Wald interval",
    "wald_t": "Student t interval",
    "band": "simultaneous band",
    "bootstrap": "bootstrap interval",
    "bootstrap_bca": "BCa bootstrap interval",
    "bootstrap_band": "studentized bootstrap band",
    "derived": "derived estimate",
}


def _sentence(text: str) -> str:
    """``text`` as a sentence: opened with a capital letter and closed with a full stop."""
    text = text.strip()
    text = text[:1].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def support_for(
    classification: Classification | None,
    status: str,
    reference_df: int | None = None,
    kind: Kind | None = None,
) -> Support:
    """What one estimate carries, from its fit's classification and its final status.

    Called where the status is final: ``stamp_inference`` for a point fit, and the longitudinal
    estimate builders after the parameter status. The estimate needs a ``point`` kind of the
    matched row when its status withholds the interval, a ``wald_t`` kind when it reads a t
    reference, and a ``wald`` kind otherwise. A shown status, a missing row, a missing kind, a
    shortfall of the row for the kind, and each group-A condition make the estimate
    experimental. Each adds its reason, and a condition or a shown status adds its roadmap id as
    evidence.

    Parameters
    ----------
    classification : Classification or None
        The classification of the fit, or ``None`` for an estimate built outside a classified
        fit, which keeps :data:`UNCLASSIFIED`.
    status : str
        The final inference status of the estimate.
    reference_df : int or None, default None
        The degrees of freedom of the estimate's t reference, or ``None`` for the normal
        reference.
    kind : str or None, default None
        The kind the row must license instead of the one the status and the reference give:
        ``"band"`` for a simultaneous band, ``"bootstrap"`` for a percentile interval,
        ``"bootstrap_bca"`` for a BCa interval, and ``"bootstrap_band"`` for a studentized
        bootstrap sup-t band.

    Returns
    -------
    Support
        ``supported`` with the row's evidence and limits when nothing makes the estimate
        experimental, and ``experimental`` with every reason otherwise.

    Examples
    --------
    >>> from cleverly._support import support_for
    >>> support_for(None, "influence_curve").label
    'experimental'
    """
    if classification is None:
        return UNCLASSIFIED
    record = None if status == "influence_curve" else STATUS_RECORDS[status]
    withheld = record is not None and record.interval == "withheld"
    if kind is None:
        kind = "point" if withheld else ("wald_t" if reference_df is not None else "wald")
    row = classification.row
    reasons: list[str] = []
    evidence: list[str] = []
    if row is None:
        text = "No registered study covers this configuration."
        if classification.nearest is not None:
            nearest_id, axes = classification.nearest
            nearest = next((r for r in registered_rows() if r.id == nearest_id), None)
            study = "" if nearest is None or not nearest.evidence else f" ({nearest.evidence[0]})"
            text += f" Nearest row: {nearest_id}{study}, differing on {', '.join(axes)}."
        reasons.append(text)
    elif not row.licenses(kind, classification.key.family):
        if kind in row.kinds:
            covered = sorted(dict(row.kind_families)[kind])
            reasons.append(
                f"Row {row.id} licenses the {_KIND_NAMES[kind]} for the families "
                f"{', '.join(covered)} only."
            )
        else:
            reasons.append(f"Row {row.id} licenses no {_KIND_NAMES[kind]} for this estimate.")
    else:
        reasons.extend(text for shortfall_kind, text in row.shortfalls if shortfall_kind == kind)
    for condition in classification.conditions:
        reasons.append(_sentence(condition.condition))
        if condition.roadmap is not None:
            evidence.append(condition.roadmap)
    if record is not None and record.interval == "shown":
        applied = {status}
        if classification.key.dependence == "below_floor":
            applied.add("few_cluster_plugin")
        for name, shown in STATUS_RECORDS.items():
            if name in applied and shown.shown_reason is not None:
                reasons.append(shown.shown_reason)
                evidence.append(shown.reopened_by)
    if reasons:
        return Support(
            "experimental",
            " ".join(reasons),
            _unique(evidence),
            row=None if row is None else row.id,
            limits=() if row is None else row.limits,
        )
    assert row is not None
    return Support(
        "supported",
        f"Row {row.id} covers this configuration.",
        row.evidence,
        row=row.id,
        limits=row.limits,
    )


def classify(
    estimator: Any,
    data: Any,
    estimands: Sequence[str],
    conditions: Sequence[Refusal] = (),
) -> Classification:
    """Classify a point-treatment fit before any learner runs.

    Parameters
    ----------
    estimator : TMLE
        The estimator. It is read and never modified.
    data : CausalData
        The prepared data.
    estimands : sequence of str
        The resolved estimand names.
    conditions : sequence of Refusal, default ()
        The group-A conditions the fit meets.

    Returns
    -------
    Classification
        The row or the nearest row, with the key and the conditions.
    """
    key, learners = point_key(estimator, data, estimands)
    return classify_key(key, learners, conditions)


def band_support(
    classification: Classification | None,
    status: str,
    supports: Iterable[Support],
    kind: Kind = "band",
) -> Support:
    """What a simultaneous band carries: its row's band kind, combined with its estimates.

    Parameters
    ----------
    classification : Classification or None
        The classification of the fit, or ``None`` for a band built outside a classified fit.
    status : str
        The inference status of the fit.
    supports : iterable of Support
        The support of each estimate the band covers.
    kind : {"band", "bootstrap_band"}, default="band"
        The band's kind: the influence-curve multiplier band, or the studentized bootstrap
        sup-t band.

    Returns
    -------
    Support
        ``supported`` only when the row licenses ``kind`` and every estimate is supported on
        that row. ``experimental`` otherwise.
    """
    kind_support = support_for(classification, status, None, kind=kind)
    return combine_support([kind_support, *supports])


# --------------------------------------------------------------------- record switch


def recording_directory() -> Path | None:
    """The directory the record switch names, or ``None`` when it is off.

    Returns
    -------
    Path or None
        The value of :data:`RECORD_ENVIRONMENT_VARIABLE` as a path, or ``None`` when it is
        unset or empty.
    """
    value = os.environ.get(RECORD_ENVIRONMENT_VARIABLE, "")
    return Path(value) if value else None


def _first_test_frame() -> str | None:
    """The first caller frame in a ``tests`` directory, as ``file:line``."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if f"{os.sep}tests{os.sep}" in name:
            return f"{name}:{frame.f_lineno}"
        frame = frame.f_back  # type: ignore[assignment]
    return None


def _append(directory: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line with the context every record shares."""
    payload = {
        **payload,
        "pytest": os.environ.get("PYTEST_CURRENT_TEST"),
        "test_frame": _first_test_frame(),
        "argv0": sys.argv[0] if sys.argv else None,
        "subdirectory": directory.name,
        "pid": os.getpid(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_jsonable, sort_keys=True) + "\n")


def record_refusal_site(
    site: str,
    refusal_code: str,
    roadmap: str | None,
    *,
    flagged: bool = False,
) -> None:
    """Record that a group-A or flagged refusal site was reached. A no-op with the switch off.

    Call it where the refusal is built, before the raise. It never changes the raise.

    Parameters
    ----------
    site : str
        A literal id written at the call site.
    refusal_code : str
        The reason code of the refusal.
    roadmap : str or None
        The roadmap id the refusal names.
    flagged : bool, default False
        Whether the plan holds the site as a refusal until the owner decides.
    """
    directory = recording_directory()
    if directory is None:
        return
    _append(
        directory,
        {
            "record": "refusal_site",
            "site": site,
            "code": str(refusal_code),
            "roadmap": roadmap,
            "flagged": flagged,
        },
    )


def _fit_outcome(result: Any, estimates: Mapping[str, Any], key: SupportKey) -> dict[str, Any]:
    """What a finished fit shows today, in the terms the phase-1 tables need."""
    statuses = sorted({str(estimate.inference) for estimate in estimates.values()})
    clusters_known = key.dependence != "iid"
    bootstrap = [
        getattr(estimate.bootstrap, "inferential", None)
        for estimate in estimates.values()
        if getattr(estimate, "bootstrap", None) is not None
    ]
    return {
        "statuses": statuses,
        "supplies_inference": [bool(supplies_inference(s)) for s in statuses],
        "clustered": clusters_known,
        "bootstrap_attached": bool(bootstrap),
        "bootstrap_inferential_today": bool(bootstrap) and all(bootstrap),
        "band_built_today": getattr(result, "simultaneous", None) is not None,
        "n_estimates": len(estimates),
    }


def record_fit(
    classification: Classification | None,
    result: Any = None,
    *,
    estimates: Mapping[str, Any] | None = None,
    source: str = "fit",
) -> None:
    """Record one classification, with what the finished fit shows today.

    A no-op with the switch off. Call it once with the classification when the fit starts,
    and again with the result when it ends.

    Parameters
    ----------
    classification : Classification or None
        The classification, or ``None`` when it could not be computed.
    result : Any, default None
        The finished result. ``None`` records the classification alone, unless ``estimates``
        is given.
    estimates : mapping of str to ParameterEstimate or None, default None
        The estimates of a call that returns no result, such as ``retarget``.
    source : str, default "fit"
        The entry point that classified.
    """
    directory = recording_directory()
    if directory is None or classification is None:
        return
    outcome = estimates if estimates is not None else getattr(result, "estimates", None)
    payload: dict[str, Any] = {
        "record": "classification" if outcome is None else "fit_outcome",
        "source": source,
        "key": classification.key.to_json(),
        "learners": {
            role: None if cls is None else f"{cls.__module__}.{cls.__qualname__}"
            for role, cls in classification.learners.items()
        },
        "label": classification.label,
        "row": None if classification.row is None else classification.row.id,
        "nearest": classification.nearest,
        "conditions": [condition.roadmap for condition in classification.conditions],
    }
    if outcome is not None:
        payload.update(_fit_outcome(result, outcome, classification.key))
    _append(directory, payload)


def _record_error(directory: Path, source: str, error: Exception) -> None:
    """Record that the classifier failed, so a measurement run shows the failure."""
    _append(
        directory,
        {"record": "classification_error", "source": source, "error": repr(error)[:500]},
    )


def classify_point_fit(
    estimator: Any,
    data: Any,
    estimands: Sequence[str],
    *,
    source: str,
    conditions: Sequence[Refusal] = (),
) -> Classification | None:
    """Classify a point-treatment fit before any learner runs.

    A classifier failure never reaches the caller. It is recorded when the switch is on, and
    the function returns ``None``, so the fit runs and its estimates carry
    :data:`UNCLASSIFIED`.

    Parameters
    ----------
    estimator : TMLE
        The estimator.
    data : CausalData
        The prepared data.
    estimands : sequence of str
        The resolved estimand names.
    source : str
        The entry point, such as ``"TMLE._fit_single"``.
    conditions : sequence of Refusal, default ()
        The group-A conditions the fit meets.

    Returns
    -------
    Classification or None
        The classification, to pass to :func:`record_fit` with the finished result.
    """
    try:
        classification = classify(estimator, data, estimands, conditions)
    except Exception as error:
        directory = recording_directory()
        if directory is not None:
            _record_error(directory, source, error)
        return None
    record_fit(classification, source=source)
    return classification


def classify_longitudinal_fit(
    estimator: Any,
    data: Any,
    *,
    source: str,
    outcome_kind: str,
    cross_fitted: bool,
    known_factors: Iterable[str],
    has_msm: bool,
    regimen_kinds: Iterable[str],
    conditions: Sequence[Refusal] = (),
) -> Classification | None:
    """Classify a longitudinal fit before any mechanism is fitted.

    A classifier failure never reaches the caller, as in :func:`classify_point_fit`.

    Parameters
    ----------
    estimator : LTMLE
        The estimator.
    data : LongitudinalData
        The prepared panel.
    source : str
        The entry point.
    outcome_kind : str
        ``"end_of_study"``, ``"survival"``, ``"competing"`` or ``"repeated"``.
    cross_fitted : bool
        Whether the outer split holds more than one fold.
    known_factors : iterable of str
        The known treatment and censoring factors.
    has_msm : bool
        Whether the fit reports working-model coefficients.
    regimen_kinds : iterable of str
        The kinds of regimen the fit reports: members of :data:`REGIMEN_KINDS`.
    conditions : sequence of Refusal, default ()
        The conditions the fit meets, such as a declared stopping rule. Each one makes the fit
        experimental.

    Returns
    -------
    Classification or None
        The classification, to pass to :func:`record_fit` with the finished result.
    """
    try:
        key, learners = longitudinal_key(
            estimator,
            data,
            outcome_kind=outcome_kind,
            cross_fitted=cross_fitted,
            known_factors=known_factors,
            has_msm=has_msm,
            regimen_kinds=regimen_kinds,
        )
        classification = classify_key(key, learners, conditions)
    except Exception as error:
        directory = recording_directory()
        if directory is not None:
            _record_error(directory, source, error)
        return None
    record_fit(classification, source=source)
    return classification


def classify_structural_fit(
    estimator: Any,
    data: Any,
    *,
    source: str,
    cross_fitted: bool,
    has_regimens: bool,
    covariate_features: bool,
) -> Classification | None:
    """Classify a structural nested mean fit before any nuisance is fitted.

    A classifier failure never reaches the caller, as in :func:`classify_point_fit`.

    Parameters
    ----------
    estimator : StructuralNestedMean
        The estimator.
    data : LongitudinalData
        The prepared panel.
    source : str
        The entry point.
    cross_fitted : bool
        Whether the outer split holds more than one fold.
    has_regimens : bool
        Whether the fit declares static regimens.
    covariate_features : bool
        Whether a blip feature reads a time-varying covariate.

    Returns
    -------
    Classification or None
        The classification, to pass to :func:`record_fit` with the finished result.
    """
    try:
        key, learners = structural_key(
            estimator,
            data,
            cross_fitted=cross_fitted,
            has_regimens=has_regimens,
            covariate_features=covariate_features,
        )
        classification = classify_key(key, learners)
    except Exception as error:
        directory = recording_directory()
        if directory is not None:
            _record_error(directory, source, error)
        return None
    record_fit(classification, source=source)
    return classification
