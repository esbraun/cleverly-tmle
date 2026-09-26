"""Fold policy and fold construction for cross-fitting.

Nuisance models are fit out-of-fold so that no observation contributes to the
model used to predict it.  Two structural constraints matter and are enforced
here rather than left to the caller:

* **stratification** on the binary label keeps rare-treatment folds usable; a
  fold with no treated units cannot produce a propensity score;
* **cluster integrity** keeps every observation of a cluster in the same fold,
  otherwise cross-fitting leaks information between correlated rows and the
  out-of-fold predictions are optimistic.

The second is *checked* here rather than assumed, which is the difference between a split
that happens to be sound and one that is known to be.  What can be checked depends on
what the caller has in hand, so the checks come in two kinds:
:meth:`Folds.__post_init__` covers what an assignment alone can be wrong about, and
:func:`check_integrity` covers what needs the cluster vector beside it.  Two prohibitions
are on neither list, for opposite reasons.  No row can appear in two validation folds,
because an assignment holds one fold index per row and two-fold membership has no
representation -- worth stating, not worth checking.  And every stratum appearing in
every fold is not checkable as a *guarantee*: :func:`resolve_n_folds` caps the fold count
to make it achievable, but a cluster is atomic, so ``StratifiedGroupKFold`` cannot always
deliver it and an imbalanced split is still a usable one.

Four objects, and the distinction between the first two is the point of the module.
:class:`CrossFitPlan` is what a caller *declares*: a policy, made of numbers, that says
nothing about any particular dataset.  :class:`Folds` is what that policy *realises* on
one: an actual assignment of rows to folds, which depends on the row order as much as on
the seed, and for a stratified split on the scikit-learn version that made it -- which is why
:mod:`cleverly.provenance` fingerprints the realisation separately from the seed, and why
a fit records the plan it declared beside the fold count it got.  :func:`make_folds` is
the map from one to the other, and calls :func:`check_integrity` on its way out, so every
split this library builds -- the outer cross-fitting folds, Super Learner's inner folds,
C-TMLE's selection folds -- is checked at construction without any of the three knowing
about it.

:class:`SplitPlan` is the fourth, and it exists because that realisation is the thing a
second fit has to be handed: an immutable, reusable record of every realised repeat, which
a caller reads off one result and gives to the next.  Handing back the seed would not do
it, for the reason above.  The generated and the supplied paths both check the assignments
before any nuisance is fitted, by different routes -- :func:`make_folds` checks what it
built, and :meth:`SplitPlan.validate` checks what it was given against the data it is
about to label.  A supplied plan must also carry the :class:`FoldOrigin` of every repeat,
and :meth:`SplitPlan.verify` draws each repeat again from that record: labels that no
recorded draw produces could have been chosen by looking at the outcome.  The record
holds the fold count and the seed the caller declared, so that check rules out a
hand-built, a stratified and an edited assignment, and it cannot audit the declaration
itself.  ``result.split_plan`` of an earlier fit is the source it is written for.

The unstratified outer splits, row-level and grouped, are the package's own:
:func:`random_partition` draws them from the seed alone and records how on
:class:`FoldOrigin`.  Their assignment depends on ``n``, the cluster labels and the seed,
and not on the installed scikit-learn.  The stratified splits stay on scikit-learn.
"""

from __future__ import annotations

import warnings

# Imported at run time rather than under ``TYPE_CHECKING``: ``Sequence`` appears in a
# dataclass field annotation, and a documentation build that resolves those annotations
# needs the name to exist.
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, cast

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import DataError
from ..utils.records import _DefaultingUnpickle

__all__ = [
    "CrossFitPlan",
    "FoldOrigin",
    "Folds",
    "SplitPlan",
    "check_integrity",
    "make_folds",
    "missing_training_support",
    "random_partition",
    "refuse_scheme",
    "resolve_n_folds",
]

#: The generator name and version :func:`random_partition` writes on every
#: :class:`FoldOrigin`.  The version changes when the same inputs would give a different
#: assignment, so a recorded origin never names an algorithm that no longer runs.
RANDOM_PARTITION_GENERATOR = "cleverly.random_partition/1"

#: The largest seed :func:`random_partition` accepts, which is the range
#: :class:`numpy.random.RandomState` accepts.
_MAX_SEED = 2**32 - 1


#: How a caller turns cross-fitting off, in both spellings, for a refusal that names it.
#: A refusal raised *after* a split was drawn never names a redraw: a fold count and a
#: seed that happen to succeed were chosen by looking at the data the split must not read.
_IN_SAMPLE_REMEDY = "fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))"

#: What a refusal raised *after* the split was drawn may offer, and why it offers no
#: redraw. A fold count or a seed that happens to give every complement what it needs was
#: chosen by looking at the treatment and the outcome, which is the dependence the
#: unstratified draw exists to remove; searching for one would put it back by hand.
#: Written here rather than beside one of its callers because four of them raise it:
#: the engine's preflights, the collaborative fold loop, the nuisance fold loop and the
#: reduced-regression fold loop, and none of those modules can import another.
_POST_DRAW_REMEDY = (
    "The split is drawn from the seed alone and reads no treatment or outcome, so trying "
    "fold counts or seeds until one fits would choose the partition by the values it must "
    "not read. Either {remedy}, or collect more observations at the rare level."
)

#: Why a plan without a generator record is refused, and how to get one that has it.
#: The record names a fold count and a seed the caller declared, so drawing the labels
#: again rules out an assignment nothing generated. It does not audit the declaration,
#: which is why the message names ``result.split_plan`` as the source rather than any
#: pair of numbers that happens to reproduce the labels.
_UNRECORDED_PLAN_REASON = (
    "split_plan carries no generator record, so the fit cannot draw its labels again and "
    "check that they are the split the record describes. A hand-built assignment has no "
    "record, and neither has a stratified split. Pass result.split_plan from a fit with "
    "unstratified folds (stratify_by='none'), or build the plan with "
    "SplitPlan.from_folds over random_partition draws"
)


def fold_strata_refusal(stratify_folds: str, *, collaborative: bool) -> str | None:
    """Return why a fold-stratification policy cannot run, or ``None``.

    ``"none"`` is the only policy any fit draws folds under. ``"treatment"`` and
    ``"treatment+outcome"`` request strata based on the treatment, and on the outcome
    as well. A split drawn from those strata would depend on data that the fit then
    analyses, while the cross-fitting argument conditions on the split. No shipped
    result covers a partition read off the data it is then used to analyse
    (``docs/technical-reference/cv-tmle.md``, fold and outcome-scale rules).

    The caller says whether the fit draws a split at all. An ordinary point-treatment fit
    and an outcome-adaptive C-TMLE fit draw one only under cross-fitting, so
    ``cross_fit=False`` leaves nothing for a policy to apply to. Selector-based C-TMLE
    draws selection and nested folds at every setting.

    Parameters
    ----------
    stratify_folds : str
        The declared policy.
    collaborative : bool
        Whether the fit draws collaborative selection folds even without cross-fitting.

    Returns
    -------
    str or None
        The reason to refuse, or ``None`` when the policy can run.
    """
    if stratify_folds == "none":
        return None
    reads = (
        "the treatment and the outcome"
        if stratify_folds == "treatment+outcome"
        else "the treatment"
    )
    where = (
        "the selection and nested folds a collaborative search draws at every setting"
        if collaborative
        else "the outer folds"
    )
    tail = (
        "A collaborative fit draws those folds whether or not cross_fit is set, so "
        "cross_fit=False does not make this policy available."
        if collaborative
        else f"Otherwise {_IN_SAMPLE_REMEDY}, which draws no split for a policy to apply to."
    )
    return (
        f"stratify_folds={stratify_folds!r} requests stratification of {where} on {reads}. "
        "A split drawn from those strata would make the partition a function of the "
        "data the fit then conditions on. No shipped result "
        "covers that split "
        "(docs/technical-reference/cv-tmle.md, fold and outcome-scale rules). "
        "Set stratify_folds='none' "
        f"(CrossFitting(stratify_by='none')), which is the default. {tail}"
    )


def _cross_fit_policy_refusal(
    *,
    cross_fit: bool,
    n_folds: int,
    repeats: int,
    split_plan: object,
    n_bootstrap: int = 0,
    stratify_folds: str = "none",
    collaborative: bool = False,
    option_name: str,
) -> str | None:
    """Return why a declared cross-fitting policy cannot run, or ``None``.

    One ordered message source for two callers with two exception contracts:
    :class:`~cleverly.CrossFitting` and :class:`~cleverly.TMLEMethod` raise
    :class:`~cleverly.exceptions.MethodConfigurationError`, and the engine raises
    :class:`ValueError`. The checks run in one order, so one input earns one reason at
    both layers:

    1. ``repeats`` below one;
    2. a ``split_plan`` that is not a :class:`SplitPlan`;
    3. a plan that cannot serve the declared policy (:meth:`SplitPlan._policy_refusal`);
    4. a plan combined with the targeted bootstrap;
    5. ``repeats`` above one without cross-fitting;
    6. cross-fitting declared with fewer than two folds;
    7. a fold-stratification policy this package draws no split under
       (:func:`fold_strata_refusal`).

    ``n_bootstrap`` belongs to a different configuration group than the other arguments.
    :class:`~cleverly.CrossFitting` does not hold it and leaves it at zero, and
    :class:`~cleverly.TMLEMethod` passes it once it holds both groups.

    Parameters
    ----------
    cross_fit : bool
        Whether the declaration enables cross-fitting.
    n_folds : int
        Outer folds the declaration asks for.
    repeats : int
        Independent draws the declaration asks for.
    split_plan : object
        The supplied plan, or ``None``. Any other type is refused.
    n_bootstrap : int, default=0
        Targeted-bootstrap replicates the declaration asks for.
    stratify_folds : str, default="none"
        The declared fold-stratification policy.
    collaborative : bool, default=False
        Whether the fit draws collaborative selection folds.
    option_name : str
        The caller's spelling of the cross-fitting switch, ``"enabled"`` on
        :class:`~cleverly.CrossFitting` and ``"cross_fit"`` on the engine.

    Returns
    -------
    str or None
        The reason to refuse, or ``None`` when the policy can run.
    """
    if repeats < 1:
        return f"repeats must be at least 1; got {repeats}"
    if split_plan is not None:
        if not isinstance(split_plan, SplitPlan):
            return "split_plan must be a SplitPlan"
        reason = split_plan._policy_refusal(cross_fit=cross_fit, n_folds=n_folds, repeats=repeats)
        if reason is not None:
            return reason
        if n_bootstrap:
            return (
                "n_bootstrap cannot be combined with split_plan: targeted bootstrap "
                "replicates duplicate sampled rows, while the supplied assignments "
                "identify only the original row positions"
            )
    if repeats > 1 and not cross_fit:
        return (
            "repeats takes the median over independent cross-fitting splits, and "
            f"{option_name}=False makes no split to draw or repeat. Enable cross-fitting or "
            "set repeats=1"
        )
    if cross_fit and n_folds < 2:
        return (
            f"{option_name}=True with n_folds={n_folds} leaves one fold, so every nuisance "
            "is fitted on the rows it predicts while the fit reports the cross-fitted "
            "estimator's name and variance rule. Set n_folds to at least 2, or fit in "
            "sample with CrossFitting(enabled=False)"
        )
    if cross_fit or collaborative:
        return fold_strata_refusal(stratify_folds, collaborative=collaborative)
    return None


@dataclass(frozen=True)
class FoldOrigin(_DefaultingUnpickle):
    """The record of how :func:`random_partition` drew one split.

    The four fields and the rows' cluster labels determine the assignment.  A caller
    can therefore draw the split again from this record and compare the result.

    Parameters
    ----------
    generator : str
        The generator name and version, ``"cleverly.random_partition/1"`` for every
        split this package draws.
    scheme : str
        ``"vfold"`` for a row-level split, and ``"grouped"`` for a split of whole
        clusters.
    requested_n_folds : int
        The fold count the caller asked for, before :func:`resolve_n_folds` capped it.
    seed : int
        The seed of the draw.

    See Also
    --------
    random_partition : The generator that writes this record.
    Folds : The split that carries this record as its ``origin``.
    """

    generator: str
    scheme: str
    requested_n_folds: int
    seed: int


@dataclass(frozen=True)
class Folds(_DefaultingUnpickle):
    """A cross-fitting partition.

    ``assignment[i]`` is the index of the fold that holds out observation ``i``.
    Iterating yields ``(train_index, test_index)`` pairs like scikit-learn.

    Parameters
    ----------
    assignment : ndarray
        ``assignment[i]`` is the index of the fold that holds out observation ``i``.
    n_folds : int
        Number of folds the assignment ranges over.
    origin : FoldOrigin or None, default=None
        How :func:`random_partition` drew the split.  ``None`` for a split from any other
        source: a stratified split, a hand-built one, or a pickle that predates the field.
        Equality ignores it, because two splits with the same labels hold out the same
        rows.
    """

    assignment: IntArray
    n_folds: int
    origin: FoldOrigin | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Check what an assignment on its own can be wrong about.

        Two things, and no more.  A fold index outside ``[0, n_folds)`` names a fold that
        does not exist, and an empty fold produces no out-of-fold predictions for anyone.
        Cluster integrity needs a vector this object does not hold, and lives in
        :func:`check_integrity`.

        Note what is *not* checked, on purpose.  "Every row is held out exactly once" has
        no check because it has no counterexample: an assignment holds one fold index per
        row, and two-fold membership is unrepresentable.  And an empty *training* fold is
        only reachable at ``n_folds == 1``, which is exactly what :meth:`single` builds --
        the ``cross_fit=False`` path, R's ``cvQinit = FALSE``, where the initial fit is
        deliberately evaluated on the rows that produced it.  Above one fold, every fold
        being non-empty already makes every complement non-empty.
        """
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1; got {self.n_folds}")
        assignment = np.asarray(self.assignment)
        if assignment.ndim != 1:
            raise DataError(
                f"fold assignment must be one index per row; got an array of shape "
                f"{assignment.shape}"
            )
        if assignment.size == 0:
            raise DataError("fold assignment is empty; there is nothing to cross-fit")
        low, high = int(assignment.min()), int(assignment.max())
        if low < 0 or high >= self.n_folds:
            raise DataError(
                f"fold assignment holds index/indices outside [0, {self.n_folds}): "
                f"the range present is [{low}, {high}]"
            )
        counts = np.bincount(assignment, minlength=self.n_folds)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            raise DataError(
                f"fold(s) {empty.tolist()} hold no rows, so they can produce no "
                f"out-of-fold predictions; {self.n_folds} folds were declared but only "
                f"{int((counts > 0).sum())} are populated"
            )

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(self.assignment.shape[0])

    def __len__(self) -> int:
        return self.n_folds

    def __iter__(self):  # type: ignore[no-untyped-def]
        # Recomputed on every pass rather than cached, and measured before being left
        # that way: 40 full iterations of a 10-fold split -- a generous count for one
        # fit, which iterates once per nuisance plus once per targeting solve -- cost
        # 2.5 ms at n=1000 and 6.8 ms at n=5000, against ~1.2 s for the whole fit with
        # the *cheapest* nuisance library (glm). That is under 1% of a fit, and an
        # order of magnitude less with the default SuperLearner. Caching it would mean
        # mutable state on a frozen dataclass for no measurable gain.
        for fold in range(self.n_folds):
            test = np.flatnonzero(self.assignment == fold)
            train = np.flatnonzero(self.assignment != fold)
            yield train, test

    def test_index(self, fold: int) -> IntArray:
        """Return held-out row indices for one fold.

        Parameters
        ----------
        fold : int
            Fold index.

        Returns
        -------
        ndarray
            Positions of the rows that fold holds out.
        """
        return np.flatnonzero(self.assignment == fold)

    @classmethod
    def single(cls, n: int) -> Folds:
        """A degenerate partition that trains and predicts on all rows.

        This is the ``cross_fit=False`` path -- it reproduces R's
        ``cvQinit = FALSE`` behaviour, where the initial fit is evaluated on the
        same data that produced it.

        Parameters
        ----------
        n : int
            Number of observations.

        Returns
        -------
        Folds
            A one-fold partition holding every row.
        """
        return cls(np.zeros(n, dtype=np.int64), 1)

    @property
    def is_single(self) -> bool:
        """Return whether this object contains one fold."""
        return self.n_folds == 1


@dataclass(frozen=True, repr=False)
class SplitPlan(_DefaultingUnpickle):
    """Reusable cross-fitting assignments for every repeat.

    A fit accepts a plan only when the plan records how :func:`random_partition` drew
    each repeat. :attr:`~cleverly.estimators.TMLEResult.split_plan` and
    :meth:`from_folds` write that record. Before any learner runs, :meth:`verify` draws
    each repeat again and refuses labels that differ. A plan built from labels alone
    stays constructible, and every fit refuses it.

    Parameters
    ----------
    assignments : sequence of sequence of int, or ndarray
        Fold labels in repeat-major order, one label per row in each repeat. Any nested
        sequence is accepted, and a two-dimensional integer array is read as one row per
        repeat. The value is copied into nested tuples, so the plan does not change when
        the source does.
    source_fingerprint : str or None, default=None
        The data fingerprint the labels were realised on, as
        :attr:`~cleverly.Provenance.data_fingerprint` records it.
        :attr:`~cleverly.estimators.TMLEResult.split_plan` fills it in, and
        :meth:`validate` then refuses data that fingerprint differently. ``None`` is a
        plan bound to no data, which is what :meth:`unbound` returns.
    provenance : tuple of FoldOrigin or None, default=None
        One generator record per repeat, in repeat order. ``None`` for labels with no
        record: a hand-built plan, a stratified split, or a pickle that predates the
        field. A fit refuses a plan whose provenance is ``None``.

    Attributes
    ----------
    n : int
    n_folds : int
    n_repeats : int
    fingerprint : str

    See Also
    --------
    cleverly.learners.CrossFitPlan : Policy that generates folds from data.
    cleverly.learners.Folds : Mutable-array materialization of one repeat.
    cleverly.learners.random_partition : The generator a plan must record.

    Examples
    --------
    >>> from cleverly import SplitPlan
    >>> from cleverly.learners import random_partition
    >>> plan = SplitPlan.from_folds([random_partition(6, 3, seed=0)])
    >>> plan.assignments
    ((2, 1, 0, 1, 2, 0),)
    >>> plan.provenance[0].seed
    0
    >>> plan.verify(n=6)
    """

    assignments: Sequence[Sequence[int]] | Sequence[IntArray] | IntArray
    source_fingerprint: str | None = None
    provenance: tuple[FoldOrigin, ...] | None = None

    def __post_init__(self) -> None:
        """Copy and validate the complete repeat-major assignment.

        The rule this enforces on a repeat is stronger than the one
        :meth:`Folds.__post_init__` enforces on an assignment, so it is checked here and
        not delegated: contiguous zero-based labels already imply that no label falls
        outside the fold range and that no fold is empty.  What it adds is the message --
        a plan is a *record*, and a caller who mistyped one needs the repeat named.
        :meth:`to_folds` builds the ``Folds`` when there is a use for one.
        """
        if self.source_fingerprint is not None and not isinstance(self.source_fingerprint, str):
            raise DataError(
                "split-plan source_fingerprint must be a provenance digest string or None; "
                f"got {type(self.source_fingerprint).__name__}"
            )
        raw = self.assignments
        if isinstance(raw, np.ndarray):
            if raw.ndim != 2:
                raise DataError(
                    "split-plan assignments must be a repeat-major two-dimensional "
                    f"array; got shape {raw.shape}"
                )
            rows = tuple(raw)
        else:
            if isinstance(raw, (str, bytes)):
                raise DataError("split-plan assignments must be a sequence of repeats")
            try:
                rows = tuple(raw)
            except TypeError as exc:
                raise DataError("split-plan assignments must be a sequence of repeats") from exc
        if not rows:
            raise DataError("split-plan assignments contain no repeats")

        normalized: list[tuple[int, ...]] = []
        expected_n: int | None = None
        expected_folds: int | None = None
        for repeat, row in enumerate(rows):
            array = np.asarray(row)
            if array.ndim != 1:
                raise DataError(
                    "each split-plan repeat must contain one fold label per row; "
                    f"repeat {repeat} has shape {array.shape}"
                )
            values = tuple(array.tolist())
            if not values:
                raise DataError(f"split-plan repeat {repeat} is empty")
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in values
            ):
                raise DataError(
                    "split-plan fold labels must be integers; "
                    f"repeat {repeat} contains a non-integer label"
                )
            assignment = tuple(int(value) for value in values)
            labels = sorted(set(assignment))
            if labels != list(range(len(labels))):
                raise DataError(
                    "split-plan fold labels must be a contiguous zero-based range; "
                    f"repeat {repeat} uses {labels}"
                )
            if expected_n is None:
                expected_n = len(assignment)
                expected_folds = len(labels)
            elif len(assignment) != expected_n:
                raise DataError(
                    "all split-plan repeats must describe the same rows; "
                    f"repeat 0 has {expected_n} rows but repeat {repeat} has {len(assignment)}"
                )
            elif len(labels) != expected_folds:
                raise DataError(
                    "all split-plan repeats must use the same number of folds; "
                    f"repeat 0 uses {expected_folds} but repeat {repeat} uses {len(labels)}"
                )
            normalized.append(assignment)
        object.__setattr__(self, "assignments", tuple(normalized))

        if self.provenance is not None:
            if isinstance(self.provenance, (str, bytes)):
                raise DataError("split-plan provenance must be a sequence of FoldOrigin records")
            try:
                records = tuple(self.provenance)
            except TypeError as exc:
                raise DataError(
                    "split-plan provenance must be a sequence of FoldOrigin records"
                ) from exc
            if len(records) != len(normalized):
                raise DataError(
                    f"split-plan provenance holds {len(records)} record(s) for "
                    f"{len(normalized)} repeat(s); a plan records one origin per repeat"
                )
            for repeat, record in enumerate(records):
                if not isinstance(record, FoldOrigin):
                    raise DataError(
                        f"split-plan provenance for repeat {repeat} must be a FoldOrigin; "
                        f"got {type(record).__name__}"
                    )
            object.__setattr__(self, "provenance", records)

    @property
    def n(self) -> int:
        """Return the number of rows in each repeat."""
        return len(self.assignments[0])

    @property
    def n_folds(self) -> int:
        """Return the number of folds in each repeat."""
        return max(self.assignments[0]) + 1

    @property
    def n_repeats(self) -> int:
        """Return the number of repeats."""
        return len(self.assignments)

    @property
    def fingerprint(self) -> str:
        """Return the provenance-compatible fingerprint of every repeat."""
        from ..provenance import fold_fingerprint

        return fold_fingerprint(self.assignments)

    def __repr__(self) -> str:
        """Return a summary rather than every label.

        A plan holds one label per row per repeat, so the generated ``__repr__`` prints
        the whole dataset back: 150 KB for a 50,000-row plan, in a traceback or a
        notebook cell that asked for one line.  What identifies a plan is its shape and
        its fingerprint, and both are here.  The generator says whether a fit can accept
        the plan at all: ``generator=None`` is a plan with no record, which every fit
        refuses.
        """
        if self.provenance is None:
            generator = "None"
        else:
            generator = "+".join(sorted({origin.generator for origin in self.provenance}))
        bound = "" if self.source_fingerprint is None else f", source={self.source_fingerprint}"
        return (
            f"SplitPlan(n={self.n}, n_folds={self.n_folds}, n_repeats={self.n_repeats}, "
            f"fingerprint={self.fingerprint}, generator={generator}{bound})"
        )

    def to_folds(self) -> tuple[Folds, ...]:
        """Return fresh mutable-array fold objects for every repeat.

        Each fold object carries its repeat's :class:`FoldOrigin` when the plan records
        one, so a fit on this plan returns a plan with the same record.

        Returns
        -------
        tuple of Folds
            Independent materializations in repeat order.
        """
        origins = (None,) * self.n_repeats if self.provenance is None else self.provenance
        return tuple(
            Folds(np.asarray(assignment, dtype=np.int64), self.n_folds, origin=origin)
            for assignment, origin in zip(self.assignments, origins, strict=True)
        )

    @classmethod
    def from_folds(
        cls, folds: Iterable[Folds], *, source_fingerprint: str | None = None
    ) -> SplitPlan:
        """Copy repeat assignments, and their generator records, from realized folds.

        Parameters
        ----------
        folds : iterable of Folds
            Realized folds in repeat order.
        source_fingerprint : str or None, default=None
            The data fingerprint these folds were realised on. ``None`` leaves the plan
            bound to no data.

        Returns
        -------
        SplitPlan
            Immutable copies of the assignments. :attr:`provenance` holds each repeat's
            :attr:`Folds.origin` when every repeat has one, and is ``None`` otherwise.

        Examples
        --------
        >>> from cleverly import SplitPlan
        >>> from cleverly.learners import random_partition
        >>> plan = SplitPlan.from_folds([random_partition(4, 2, seed=s) for s in (1, 2)])
        >>> plan.n_repeats, [origin.seed for origin in plan.provenance]
        (2, [1, 2])
        """
        draws = tuple(folds)
        origins = tuple(draw.origin for draw in draws)
        return cls(
            tuple(draw.assignment for draw in draws),
            source_fingerprint=source_fingerprint,
            provenance=(
                None
                if any(origin is None for origin in origins)
                else cast("tuple[FoldOrigin, ...]", origins)
            ),
        )

    def unbound(self) -> SplitPlan:
        """Return this plan bound to no data, with its generator record kept.

        A refit on the same rows with one column replaced changes the data fingerprint
        and moves no row. The unbound plan serves that refit, and a caller who means to
        reuse the labels on other rows asks for it by name. :meth:`verify` still draws
        every repeat again, so an unbound plan cannot carry labels the record does not
        produce.

        Returns
        -------
        SplitPlan
            The same assignments and :attr:`provenance`, with
            :attr:`source_fingerprint` set to ``None``.

        Examples
        --------
        >>> from cleverly import SplitPlan
        >>> from cleverly.learners import random_partition
        >>> bound = SplitPlan.from_folds([random_partition(4, 2, seed=3)], source_fingerprint="ab")
        >>> free = bound.unbound()
        >>> free.source_fingerprint is None, free.provenance == bound.provenance
        (True, True)
        """
        return replace(self, source_fingerprint=None)

    def verify(self, *, n: int, cluster: IntArray | None = None) -> None:
        """Refuse labels that the recorded generator does not produce on these rows.

        Each repeat is drawn again with :func:`random_partition`, from its recorded fold
        count and seed, and compared label for label. The recorded scheme must be
        ``"grouped"`` when the data declare clusters and ``"vfold"`` when they do not. The
        draw reads ``n`` and the cluster labels only, so a plan that passes holds labels
        the recorded fold count and seed produce.

        What this rules out is an assignment no draw of this package made: a hand-built
        one, a stratified one, and one edited after the draw. What it cannot rule out is
        the declaration itself. The fold count and the seed come from the caller, so a
        caller who searched for a seed gets a plan this method accepts. A fit records
        that declaration and has nothing local to audit it against.
        ``result.split_plan`` of an earlier fit is the source this contract is written
        for.

        Parameters
        ----------
        n : int
            Number of data rows.
        cluster : ndarray or None, default=None
            Cluster code for each row, or ``None`` for data without clusters.

        Raises
        ------
        DataError
            If the plan has no generator record, if the row count differs, or if a
            repeat names another generator, the other scheme, or labels its record does
            not produce. The message names the repeat.

        Examples
        --------
        >>> from cleverly import SplitPlan
        >>> from cleverly.learners import random_partition
        >>> SplitPlan.from_folds([random_partition(6, 2, seed=5)]).verify(n=6)
        """
        if self.provenance is None:
            raise DataError(_UNRECORDED_PLAN_REASON)
        if self.n != n:
            raise DataError(f"split plan has {self.n} rows but the data have {n} rows")
        scheme = "vfold" if cluster is None else "grouped"
        for repeat, (assignment, origin) in enumerate(
            zip(self.assignments, self.provenance, strict=True)
        ):
            if origin.generator != RANDOM_PARTITION_GENERATOR:
                raise DataError(
                    f"split-plan repeat {repeat} records generator {origin.generator!r}, and "
                    f"this version draws with {RANDOM_PARTITION_GENERATOR!r}, so it cannot "
                    "draw the labels again to check them"
                )
            if origin.scheme != scheme:
                declared = "declare clusters" if cluster is not None else "declare no clusters"
                raise DataError(
                    f"split-plan repeat {repeat} records a {origin.scheme!r} draw, and these "
                    f"data {declared}, which a {scheme!r} draw serves. A row-level draw cuts "
                    "across clusters, and a grouped draw needs the cluster labels it split"
                )
            # The cap warning belongs to the fit that drew the split. This draw repeats it
            # to check the labels, and resolves no fold count for the fit.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    redrawn = random_partition(
                        n, origin.requested_n_folds, cluster=cluster, seed=origin.seed
                    )
                except (ValueError, DataError) as exc:
                    raise DataError(
                        f"split-plan repeat {repeat} records a draw that cannot run on these "
                        f"rows: {exc}"
                    ) from exc
            differing = int(np.count_nonzero(redrawn.assignment != np.asarray(assignment)))
            if differing:
                raise DataError(
                    f"split-plan repeat {repeat} differs from the split its record draws "
                    f"(seed {origin.seed}, {origin.requested_n_folds} requested folds) at "
                    f"{differing} row(s). A fit accepts only the labels the recorded fold "
                    "count and seed draw, because other labels could have been chosen by "
                    "reading the outcome"
                )

    def _policy_refusal(self, *, cross_fit: bool, n_folds: int, repeats: int) -> str | None:
        """Return why this plan cannot serve a declared fold policy, or ``None``.

        One message source for two callers with two exception contracts:
        :class:`~cleverly.CrossFitting` raises
        :class:`~cleverly.exceptions.MethodConfigurationError` and the engine raises
        :class:`ValueError`, and neither should paraphrase the other.

        The fold count is compared as an upper bound rather than for equality, because
        the count a fit *ran* is the count it *declared* capped at what the data support:
        :func:`resolve_n_folds` caps at the rarest stratum and again at the cluster count.
        A plan realised under a cap therefore holds fewer folds than the declaration that
        produced it, and refusing that here would refuse a plan this package itself
        wrote. More folds than declared is the direction no cap can produce, and it is
        the only fold-count question a declaration can answer without the data. Whether
        the labels can serve a particular dataset is :meth:`validate`'s question, and
        :meth:`~cleverly.estimators.TMLE._repeat_draws` asks it there.

        A plan with no generator record is refused first, whatever else it holds. No
        declaration can make such a plan acceptable, so a caller who fixes its shape
        would only meet this refusal next. Whether the record reproduces the labels is
        :meth:`verify`'s question, and it needs the rows.

        Parameters
        ----------
        cross_fit : bool
            Whether the declaration enables cross-fitting.
        n_folds : int
            Outer folds the declaration asks for.
        repeats : int
            Independent draws the declaration asks for.

        Returns
        -------
        str or None
            The reason to refuse, or ``None`` when the plan can serve the policy.
        """
        if self.provenance is None:
            return _UNRECORDED_PLAN_REASON
        if not cross_fit or n_folds < 2:
            return "split_plan requires enabled cross-fitting with at least two folds"
        if self.n_folds > n_folds:
            return (
                f"split_plan uses {self.n_folds} folds but n_folds is {n_folds}; a supplied "
                "plan may hold fewer folds than the declaration, because the data can cap "
                "the count, and never more"
            )
        if self.n_repeats != repeats:
            return f"split_plan has {self.n_repeats} repeats but repeats is {repeats}"
        return None

    def validate(
        self,
        *,
        n: int,
        cluster: IntArray | None = None,
        treatment: FloatArray | None = None,
        stratify: FloatArray | None = None,
        source_fingerprint: str | None = None,
    ) -> tuple[Folds, ...]:
        """Validate assignments against the data and return fresh folds.

        Parameters
        ----------
        n : int
            Number of data rows.
        cluster : ndarray, optional
            Cluster code for each row.
        treatment : ndarray, optional
            Categorical treatment code for each row.
        stratify : ndarray, optional
            Requested stratification code for each row. Named as in :func:`make_folds`,
            which balances on the same vector; ``strata`` is the survey design role
            :attr:`~cleverly.data.CausalData.strata` carries.
        source_fingerprint : str or None, default=None
            Fingerprint of the data being validated. Compared with
            :attr:`source_fingerprint` when both are present.

        Returns
        -------
        tuple of Folds
            Fresh fold realizations in repeat order.
        """
        if self.n != n:
            raise DataError(f"split plan has {self.n} rows but the data have {n} rows")
        if (
            self.source_fingerprint is not None
            and source_fingerprint is not None
            and self.source_fingerprint != source_fingerprint
        ):
            raise DataError(
                f"split plan was realised on data fingerprinting {self.source_fingerprint}, "
                f"and these data fingerprint {source_fingerprint}. A plan labels rows by "
                "position, so a reordering, a replaced column or an added covariate leaves "
                "every label pointing at a different unit, and the row count cannot see it. "
                "Fit without split_plan= to draw a split for these data, or, when the rows "
                "are the same units in the same order, hand over plan.unbound() to reuse "
                "the labels unbound"
            )
        # Each vector is read once here rather than once per (repeat x fold): none of them
        # changes as the loop below walks the folds, and neither does the set of values a
        # training complement has to contain.
        prepared: dict[str, Any] = {}
        for name, vector in (
            ("cluster", cluster),
            ("treatment arm", treatment),
            ("stratum", stratify),
        ):
            if vector is None:
                continue
            values = np.asarray(vector).reshape(-1)
            if values.shape[0] != n:
                raise DataError(f"{name} has {values.shape[0]} rows but the data have {n}")
            prepared[name] = values
        # The cluster vector is checked by ``check_integrity`` rather than for support:
        # a cluster is atomic, so it is meant to be absent from the folds it is not in.
        codes = prepared.pop("cluster", None)
        support = tuple((name, values, np.unique(values)) for name, values in prepared.items())

        realized = self.to_folds()
        for repeat, folds in enumerate(realized):
            check_integrity(folds, cluster=codes)
            gap = missing_training_support(folds, support)
            if gap is not None:
                fold, name, missing = gap
                raise DataError(
                    f"split-plan repeat {repeat}, fold {fold} has no {name}(s) "
                    f"{missing.tolist()} in its training complement"
                )
        return realized


def missing_training_support(
    folds: Folds,
    support: Sequence[tuple[str, FloatArray | BoolArray, FloatArray | BoolArray]],
) -> tuple[int, str, FloatArray | BoolArray] | None:
    """Find the first training complement that lacks a value the whole sample holds.

    A nuisance fitted on a complement that never saw a value cannot predict it on the
    held-out rows. :meth:`SplitPlan.validate` and the cross-fitted natural-course
    preflight in :class:`~cleverly.TMLE` both ask this question, so it is answered once.

    Parameters
    ----------
    folds : Folds
        One realized partition. A single fold trains on every row.
    support : sequence of tuple
        ``(name, values, present)`` triples: a label for the error, one value per row,
        and the values every training complement must contain.

    Returns
    -------
    tuple or None
        ``(fold, name, missing)`` for the first fold, in fold order and then support
        order, whose complement lacks a value. ``None`` when every complement has them all.
    """
    training_sets = (
        (np.arange(folds.n, dtype=np.int64),)
        if folds.is_single
        else tuple(train for train, _ in folds)
    )
    for fold, train in enumerate(training_sets):
        for name, values, present in support:
            missing = np.setdiff1d(present, np.unique(values[train]))
            if missing.size:
                return fold, name, missing
    return None


def check_integrity(folds: Folds, *, cluster: IntArray | None = None) -> None:
    """Check the prohibition that needs a vector beside the assignment.

    Every row of a cluster must land in the same fold.  This is the prohibition that
    actually leaks: correlated rows split across the boundary let a nuisance model see,
    in training, rows that stand in for the ones it is about to predict, and the
    out-of-fold predictions come back optimistic.  An unclustered split passes trivially
    and pays nothing.

    Called as a post-condition of :func:`make_folds`, so every split this library builds
    is checked at construction.  It should never fire -- :func:`random_partition` and
    ``StratifiedGroupKFold`` both guarantee it -- which is the point: the unstratified
    grouped split is this library's own code, and a post-condition is what turns its
    guarantee from a hope into a checked claim.  Also exposed for the three holders of a
    :class:`Folds` that
    :func:`make_folds` never saw: a result reloaded from disk, a caller who built one by
    hand, and a :class:`SplitPlan` handed back to a later fit, which
    :meth:`SplitPlan.validate` puts through this check once per repeat.

    Stratum coverage is deliberately *not* checked here.  Every stratum in every fold is
    guaranteed only when the strata alone constrain the split -- ``resolve_n_folds`` caps
    the count to make it so -- and ``StratifiedGroupKFold`` cannot promise it, because a
    cluster is atomic and a rare class concentrated in few clusters has nowhere else to
    go.  Asserting it would refuse splits that are imbalanced but perfectly usable.

    Runs once per split rather than on the iteration path: one ``np.unique`` over ``n``
    rows, against the many nuisance fits a split goes on to serve.
    :meth:`Folds.__iter__` records that iterating a split at all is under 1% of a fit,
    and this is well below that.
    """
    if cluster is None:
        return
    assignment = np.asarray(folds.assignment)
    codes = np.asarray(cluster).reshape(-1)
    if codes.shape[0] != assignment.shape[0]:
        raise DataError(
            f"cluster has {codes.shape[0]} row(s) but the fold assignment has {assignment.shape[0]}"
        )
    # The distinct (cluster, fold) pairs: a cluster that stayed intact contributes
    # exactly one.  One sort beats an unbuffered scatter -- the same reason
    # inference.cluster.cluster_sums uses np.bincount rather than np.add.at.
    unique, inverse = np.unique(codes, return_inverse=True)
    pairs = np.unique(np.stack([inverse.reshape(-1), assignment], axis=1), axis=0)
    per_cluster = np.bincount(pairs[:, 0], minlength=unique.size)
    split = np.flatnonzero(per_cluster > 1)
    if split.size:
        raise DataError(
            f"{split.size} cluster(s) have rows in more than one fold, which is the "
            f"leakage grouped cross-fitting exists to prevent; the first is cluster code "
            f"{int(unique[split[0]])}, spread over {int(per_cluster[split[0]])} folds"
        )


def resolve_n_folds(
    n_folds: int,
    n: int,
    stratify: FloatArray | None = None,
    *,
    cluster: IntArray | None = None,
) -> int:
    """Cap the requested number of folds at what the data can support.

    A stratified split needs at least one member of the rarer class per fold, so
    the cap is the rarer class count.  Silently exceeding it would raise deep
    inside scikit-learn, or worse, produce a fold with a single treatment arm.

    A cluster is atomic, so a grouped split cannot make more folds than there are
    clusters, and ``cluster`` applies that second cap.  Both caps live here rather than
    one here and one in :func:`make_folds`, so that one function answers "how many folds
    can a generated split make on these data" and every generated split asks it once.

    This is a question about a split that is about to be *generated*.  A supplied
    :class:`SplitPlan` is not held to the count it returns: a plan may hold more folds
    than this, because a rare stratum has to reach every training complement rather than
    appear once per fold, and :meth:`SplitPlan.validate` checks that property on the
    labels themselves.

    Parameters
    ----------
    n_folds : int
        Folds requested.
    n : int
        Number of observations.
    stratify : ndarray or None
        Labels the split must balance, usually the treatment indicator.
    cluster : ndarray or None
        Cluster codes, when every row of a cluster must land in the same fold.

    Returns
    -------
    int
        The fold count the data support, at most ``n_folds``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2 for cross-fitting; got {n_folds}")
    cap = n
    if stratify is not None:
        counts = np.unique(np.asarray(stratify), return_counts=True)[1]
        cap = int(counts.min())
    resolved = int(min(n_folds, cap))
    # Two callers, two caps, and the message has to name the one that bound. An
    # unstratified draw caps at ``n``, and :func:`random_partition` is now the only outer
    # path, so "the rarer class" would describe a stratum nobody asked for.
    binding = "the rarer class has only" if stratify is not None else "the data hold only"
    counted = "member(s)" if stratify is not None else "row(s)"
    if resolved < 2:
        raise ValueError(
            f"cannot cross-fit: {binding} {cap} {counted}, so no split into two folds exists"
        )
    if resolved < n_folds:
        warnings.warn(
            f"reducing n_folds from {n_folds} to {resolved}: {binding} {cap} {counted}",
            UserWarning,
            stacklevel=2,
        )
    if cluster is not None:
        n_groups = int(np.unique(cluster).size)
        if n_groups < resolved:
            warnings.warn(
                f"reducing n_folds from {resolved} to {n_groups}: only {n_groups} clusters",
                UserWarning,
                stacklevel=2,
            )
            resolved = n_groups
        if resolved < 2:
            raise ValueError("cluster-respecting cross-fitting needs at least 2 clusters")
    return resolved


def make_folds(
    n: int,
    n_folds: int = 10,
    *,
    stratify: FloatArray | None = None,
    cluster: IntArray | None = None,
    random_state: int | np.random.Generator | None = None,
) -> Folds:
    """Build a cross-fitting partition of ``n`` observations.

    Parameters
    ----------
    n : int
        Number of observations to partition.
    n_folds : int
        Folds requested. Reduced, with a warning, when the data cannot support it.
    stratify : ndarray or None
        Labels to balance across folds.  Pass the treatment indicator.
    cluster : ndarray or None
        Cluster codes; every row of a cluster lands in the same fold.
    random_state : int, Generator, or None
        Seed or generator.  The split is deterministic given an integer seed, so a fit
        is reproducible.  ``None`` draws a fresh seed for an unstratified split, which
        the returned :attr:`Folds.origin` records.

    Returns
    -------
    Folds
        A checked partition, cluster-respecting and stratified as asked.

    Notes
    -----
    Without ``stratify`` the split comes from :func:`random_partition`, which does not
    depend on the installed scikit-learn.  With ``stratify`` it comes from
    ``StratifiedKFold`` or, with ``cluster``, from ``StratifiedGroupKFold``.
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations to cross-fit; got {n}")
    seed = _as_seed(random_state)
    if stratify is None:
        return random_partition(
            n, n_folds, cluster=cluster, seed=_fresh_seed() if seed is None else seed
        )
    resolved = resolve_n_folds(n_folds, n, stratify, cluster=cluster)
    x = np.zeros((n, 1))
    assignment = np.empty(n, dtype=np.int64)

    if cluster is not None:
        splitter: StratifiedGroupKFold | StratifiedKFold = StratifiedGroupKFold(
            n_splits=resolved, shuffle=True, random_state=seed
        )
        iterator = splitter.split(x, np.asarray(stratify), groups=cluster)
    else:
        splitter = StratifiedKFold(n_splits=resolved, shuffle=True, random_state=seed)
        iterator = splitter.split(x, np.asarray(stratify))

    for fold, (_, test) in enumerate(iterator):
        assignment[test] = fold
    folds = Folds(assignment, resolved)
    check_integrity(folds, cluster=cluster)
    return folds


def random_partition(
    n: int,
    n_folds: int,
    *,
    cluster: IntArray | None = None,
    seed: int,
) -> Folds:
    """Draw an unstratified split of rows, or of whole clusters, from a seed.

    The draw reads no outcome, treatment or covariate.  The assignment depends on ``n``,
    the cluster labels and ``seed`` alone, so the same inputs give the same split on
    every supported scikit-learn.

    Parameters
    ----------
    n : int
        Number of rows to split.
    n_folds : int
        Folds requested.  :func:`resolve_n_folds` caps it at ``n`` and at the number of
        clusters, with a warning.
    cluster : ndarray or None, default=None
        One cluster code per row.  Every row of a cluster lands in the same fold.
    seed : int
        Seed of the draw, in ``[0, 2**32 - 1]``.

    Returns
    -------
    Folds
        A checked split whose :attr:`~Folds.origin` records the generator, the scheme,
        the requested fold count and the seed.

    Raises
    ------
    ValueError
        If ``n`` is below two, the seed is not an integer in range, or fewer than two
        folds are possible.

    See Also
    --------
    make_folds : Builds a split with or without strata, and calls this function
        for an unstratified split.
    FoldOrigin : The record this function attaches to the split.

    Notes
    -----
    The row-level draw shuffles ``numpy.arange(n)`` with
    ``numpy.random.RandomState(seed)``.  It then cuts the order into ``K`` contiguous
    blocks of ``n // K`` rows, and the first ``n % K`` blocks take one extra row.  This
    is the split ``sklearn.model_selection.KFold(K, shuffle=True, random_state=seed)``
    makes.

    The grouped draw permutes the sorted distinct cluster labels with
    ``numpy.random.RandomState(seed)`` and cuts the permutation into ``K`` parts with
    ``numpy.array_split``.  This is the split
    ``sklearn.model_selection.GroupKFold(K, shuffle=True, random_state=seed)`` makes in
    scikit-learn 1.6 and later.  The number of clusters in two folds differs by at most
    one, and a cluster's size does not change where it lands.

    Examples
    --------
    >>> from cleverly.learners import random_partition
    >>> folds = random_partition(6, 3, seed=0)
    >>> folds.assignment.tolist()
    [2, 1, 0, 1, 2, 0]
    >>> folds.origin.scheme, folds.origin.seed
    ('vfold', 0)
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations to cross-fit; got {n}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError(f"seed must be an integer; got {type(seed).__name__}")
    if not 0 <= int(seed) <= _MAX_SEED:
        raise ValueError(f"seed must lie in [0, {_MAX_SEED}]; got {seed}")
    seed = int(seed)
    resolved = resolve_n_folds(n_folds, n, cluster=cluster)
    rng = np.random.RandomState(seed)
    if cluster is None:
        order = np.arange(n)
        rng.shuffle(order)
        assignment = np.empty(n, dtype=np.int64)
        assignment[order] = _contiguous_blocks(n, resolved)
        scheme = "vfold"
    else:
        codes = np.asarray(cluster).reshape(-1)
        if codes.shape[0] != n:
            raise DataError(f"cluster has {codes.shape[0]} row(s) but n is {n}")
        labels, inverse = np.unique(codes, return_inverse=True)
        permuted = rng.permutation(labels)
        label_fold = np.empty(labels.size, dtype=np.int64)
        label_fold[np.searchsorted(labels, permuted)] = _contiguous_blocks(labels.size, resolved)
        assignment = label_fold[inverse.reshape(-1)]
        scheme = "grouped"
    origin = FoldOrigin(
        generator=RANDOM_PARTITION_GENERATOR,
        scheme=scheme,
        requested_n_folds=int(n_folds),
        seed=seed,
    )
    folds = Folds(assignment, resolved, origin=origin)
    check_integrity(folds, cluster=cluster)
    return folds


def _contiguous_blocks(size: int, n_folds: int) -> IntArray:
    """Return fold labels for ``size`` ordered items cut into ``n_folds`` blocks.

    The blocks are contiguous.  Each holds ``size // n_folds`` items, and the first
    ``size % n_folds`` blocks hold one more.  ``KFold`` and ``numpy.array_split`` both
    cut this way.

    Parameters
    ----------
    size : int
        Number of ordered items.
    n_folds : int
        Number of blocks.

    Returns
    -------
    ndarray
        The block label of each item, in order.
    """
    sizes = np.full(n_folds, size // n_folds, dtype=np.int64)
    sizes[: size % n_folds] += 1
    return np.repeat(np.arange(n_folds, dtype=np.int64), sizes)


def _fresh_seed() -> int:
    """Draw a concrete seed from operating-system entropy.

    The value lies in ``[0, 2**31 - 2]``, the range :meth:`CrossFitPlan.seeds` reduces
    to, so a resolved seed is valid wherever a declared one is.

    Returns
    -------
    int
        A new seed.
    """
    return int(np.random.SeedSequence().generate_state(1)[0]) % (2**31 - 1)


@dataclass(frozen=True)
class CrossFitPlan:
    """The fold policy a fit declared, as distinct from the split it got.

    Every field is a number or a string, so a plan is comparable, hashable and
    serialisable, and says nothing about any particular dataset.  What a plan realises on
    one is a :class:`Folds`, and the two can differ: :func:`resolve_n_folds` caps
    ``n_folds`` at the rarest stratum and again at the cluster count, both with a warning
    at fit time and no trace afterwards.  Recording the plan beside the realised count is
    what makes "why did my 10-fold fit run 3 folds?" answerable from a saved result.

    Built from the estimator's own keyword arguments by
    :meth:`~cleverly.estimators.tmle.TMLE.crossfit_plan` and held on
    :class:`~cleverly.estimators.base.TMLEConfig`, exactly as
    :class:`~cleverly.estimators.targeting.TargetingSpec` is -- so the settings appear
    once and cannot drift.

    Parameters
    ----------
    n_folds : int
        Outer folds, the ones that make the nuisance predictions out of fold.  ``1``
        means ``cross_fit=False``.
    learner_folds : int
        Inner folds, which score Super Learner's candidates *inside* one outer training
        fold.  A separate declaration rather than the same one: this is model selection,
        not what makes a prediction out of fold, and its stratum is the learner's own
        target -- the treatment for the mechanism, the outcome for the regression.
    scheme : str
        Which family of split the fit used.  ``"supplied"`` means a caller handed over
        exact assignments and nothing was generated.  ``"none"`` means the fit drew no
        split.  The other two are resolved from what the data declared rather than
        chosen: ``"grouped"`` whenever ``id=`` named clusters, and ``"vfold"`` otherwise.
        A result restored from an earlier version can carry a stratified value, which this
        version draws no split under.
    stratify_by : tuple of str
        What the outer folds were checked against, as user-facing names.  Empty on every
        fit this version runs, because it draws no split that balances the data it then
        conditions on.  A result restored from an earlier version can carry names here.
    random_state : int or None
        Seed for generated outer splits and repeat-specific learner state. Under
        ``scheme="supplied"``, the assignments ignore it while learner and collaborative
        selection folds remain seeded. A seed is not enough to reproduce a generated
        split on its own; see :mod:`cleverly.provenance`.
    repeats : int
        How many independent draws of the whole split the fit combines by median. ``1`` is
        an ordinary fit.  A count layered over whichever ``scheme`` the data resolved to,
        not a scheme of its own -- repeating a grouped split gives grouped splits.
    """

    n_folds: int = 10
    learner_folds: int = 5
    scheme: str = "stratified"
    stratify_by: tuple[str, ...] = ()
    random_state: int | None = None
    repeats: int = 1

    @property
    def cross_fit(self) -> bool:
        """Whether the outer nuisance fits are cross-fitted at all."""
        return self.n_folds > 1

    @property
    def repeated(self) -> bool:
        """Whether more than one draw of the split was combined."""
        return self.repeats > 1

    def describe(self) -> str:
        """Return a readable description.

        Returns
        -------
        str
            One line naming the fold count, the repeats, and the stratification.
        """
        by = f" stratified on {', '.join(self.stratify_by)}" if self.stratify_by else ""
        if not self.cross_fit:
            return "declared: no cross-fitting (cross_fit=False)"
        over = f", median over {self.repeats} draws" if self.repeated else ""
        return f"declared: {self.n_folds}-fold {self.scheme}{by}{over}"

    def seeds(self) -> tuple[int | None, ...]:
        """One seed per repeat, for the fold draws to combine.

        Spawned from ``random_state`` rather than derived by addition, so the draws are
        independent rather than merely different, and a repeated fit stays reproducible
        under a seed.  ``random_state=None`` yields ``None`` per repeat, because the plan
        is a declaration and the caller declared no seed.  The engine does not draw with
        ``None``: :meth:`~cleverly.estimators.TMLE._repeat_draws` replaces each ``None``
        with a fresh seed from operating-system entropy, in the same ``[0, 2**31 - 2]``
        range, before it draws the folds.  That seed reaches the folds, the learners and
        the C-TMLE selection folds, and
        :attr:`~cleverly.estimators._nuisance.RepeatFit.seed` records it on the result.
        The plan keeps ``random_state=None``, so the result still says that the caller
        fixed no seed.

        One repeat passes ``random_state`` straight through rather than spawning from it,
        which is what makes ``repeats=1`` bit-for-bit an ordinary fit rather than merely
        an equivalent one.  ``tests/unit/test_repeated_crossfit.py`` enforces that.

        Returns
        -------
        tuple of int or None
            One seed per repeat, derived from :attr:`random_state`.
        """
        if not self.repeated:
            return (self.random_state,)
        if self.random_state is None:
            return (None,) * self.repeats
        state = np.random.SeedSequence(self.random_state).generate_state(self.repeats)
        return tuple(int(value) % (2**31 - 1) for value in state)


def refuse_scheme(kind: str) -> None:
    """Raise for a fold scheme this package will not fake.

    Three refusals, for three different reasons -- which is why they are spelled out
    rather than collected under one "not implemented".  ``"blocked"`` is missing a data
    layer; ``"rolling_origin"`` is *incompatible with the storage contract* and would
    still be after that layer arrived, which is the one worth reading; and
    ``"row_within_cluster"`` is refused outright rather than unimplemented.

    A fourth name is handled here and is no longer a refusal at all.  ``"repeated"``
    shipped as ``repeats=``, and it keeps a branch only to say that it was never a
    *scheme* -- so a caller who reaches for it by name is redirected to the option that
    exists rather than told the feature does not.
    """
    if kind == "blocked":
        raise NotImplementedError(
            "blocked temporal folds are not implemented. A contiguous-in-time split is "
            "perfectly expressible as a fold assignment -- what is missing is the "
            "ordering it would need. CausalData declares an outcome, a treatment, "
            "baseline covariates and a cluster, and no node carries a time index; id= "
            "is the independent sampling unit, not a row or time key. LongitudinalData "
            "does order its nodes, but along a within-unit axis: its rows are still "
            "exchangeable units, and a fold there splits units rather than time, so it "
            "supplies the ordering a *panel* would need and not the one this scheme does."
        )
    if kind == "rolling_origin":
        raise NotImplementedError(
            "rolling-origin folds are not implemented, and not for want of a time index. "
            "Their training sets are nested prefixes: some rows are never held out and "
            "others fall in several evaluation windows, so there is no one fold that "
            "holds out each row. Cross-fitting here rests on exactly that -- Folds is an "
            "assignment of one fold per row, and NuisanceEstimates stores one "
            "out-of-fold prediction per row because of it. A rolling origin needs a "
            "different storage contract, not a different splitter."
        )
    if kind in {"repeated", "repeats"}:
        raise ValueError(
            "repeated cross-fitting is implemented, and it is not a scheme: it is a count "
            "layered over whichever scheme the data resolved to, so repeating a grouped "
            "split gives grouped splits. Pass repeats= to the estimator rather than "
            "naming it here; the count is recorded on CrossFitPlan.repeats."
        )
    if kind == "row_within_cluster":
        raise ValueError(
            "splitting a cluster across folds is refused. id= declares the independent "
            "sampling unit, and it deliberately fuses the variance unit with the fold "
            "unit: dealing a cluster's correlated rows into different folds lets a "
            "nuisance model train on rows that stand in for the ones it predicts, which "
            "is the leakage grouped cross-fitting exists to prevent. Too few clusters to "
            "split is a reason to reduce n_folds, not to unfuse them."
        )
    raise ValueError(f"unknown cross-fitting scheme {kind!r}")


def _as_seed(random_state: int | np.random.Generator | None) -> int | None:
    """Turn a generator or seed into an int scikit-learn accepts."""
    if random_state is None:
        return None
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**31 - 1))
    return int(random_state)
