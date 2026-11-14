r"""A two-node treatment regimen learned inside each training fold (X11 part (b)).

A :class:`~cleverly.longitudinal.DynamicRegimen` holds *known* rules, and a rule learned from the
analysis sample is refused there.  :class:`LearnedRegimen` declares the one learned regimen that
:class:`~cleverly.longitudinal.LTMLE` estimates instead: two decision nodes with a binary treatment
at each, and an end-of-study outcome.

Each outer training fold :math:`v` learns its own rule pair :math:`d_v = (d_{v,1}, d_{v,2})` by
backward plug-in Q-learning on its training rows (van der Laan and Luedtke 2015, Theorem 1):

1. at the second node, for each first-node arm :math:`a_1`, regress the outcome on the history
   :math:`(L_1, L_2)` among the rows that took :math:`a_1` and then each arm :math:`a_2`, which
   gives :math:`\bar Q_{v,2}(H_2, a_1, a_2)`, and set
   :math:`d^{a_1}_{v,2}(H_2) = 1\{\bar Q_{v,2}(H_2, a_1, 1) - \bar Q_{v,2}(H_2, a_1, 0) > 0\}`;
2. at the first node, for each arm :math:`a_1`, regress
   :math:`\bar Q_{v,2}(H_2, a_1, d^{a_1}_{v,2}(H_2))` on :math:`L_1` among the rows that took
   :math:`a_1`, which gives :math:`\bar Q_{v,1}(L_1, a_1)`, and set
   :math:`d_{v,1}(L_1) = 1\{\bar Q_{v,1}(L_1, 1) - \bar Q_{v,1}(L_1, 0) > 0\}`;
3. the second-node rule of the pair reads the arm the first rule assigns,
   :math:`d_{v,2}(H_2) = d^{d_{v,1}(L_1)}_{v,2}(H_2)`.

``1`` and ``0`` are the higher and the lower of a node's two sorted arm codes, and a tie assigns
the lower one.  The second node's regressions are stratified by the first-node arm because the
outcome design of a deterministic node holds no block for an earlier arm, and the first node's
target is each row's prediction from its own first-node branch at the arm the learned
second-node rule assigns there.  So the first-node rule is learned under the learned second-node
rule, not under a fixed one.  Every regression is fitted on the fold's training rows and predicted
on every row, so a fold's rules read no validation row (Luedtke and van der Laan 2016,
*International Journal of Biostatistics*, Sections 2.2 and 3.2).

The estimator is the cross-fitted LTMLE of a regimen whose rule differs by fold.  Fold
:math:`v`'s untargeted recursion runs under its own rule pair on its training rows, every row
follows the rule pair of the fold that holds it out, the mechanism is evaluated at that rule, and
one pooled logistic fluctuation per node targets the stitched predictions, node 2 then node 1 (van
der Laan and Luedtke 2015, Appendix B, Equations 21 to 24).  The report is the pooled mean, so the
target is

.. math::

    \tilde\psi^{s}_{0n} = \sum_v \frac{n_v}{n}\, \Psi_{d_v}(P_0),

the stacked weighting of the point-treatment learned rule, and the variance is the curve's second
moment centred at the pooled estimate, as for any cross-fitted LTMLE fit.  Theorem 6 of van der
Laan and Luedtke (2015) gives asymptotic linearity with no Donsker condition when the fold rules
settle to a fixed limit, and Corollary 3 adds the term of an estimated mechanism.

This module adds the declaration, the fold-local Q-learning, the result record and the refusals.
The recursion, the targeting and the curve are the shipped ones, run under per-row assignments
(:meth:`~cleverly.longitudinal.regimen.Plan.with_assignment`,
:func:`~cleverly.longitudinal.sequential.stitch_fold_assignments` and
:attr:`~cleverly.longitudinal.sequential.FoldRecursionCell.fold_assignments`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import CapabilityError, DataError, Refusal, RefusalReason
from ..learners.crossfit import Folds
from ..utils.bounds import OutcomeScaler
from ..utils.parallel import map_parallel
from ..utils.phases import PhaseProfile, collect_phases, merge_worker_phases, phase, profiling
from .data import LongitudinalData
from .regimen import Plan
from .sequential import (
    _FILLER,
    FoldRecursionCell,
    Mechanism,
    NodeBound,
    RegimenFit,
    _finish_regimen_fit,
    _fit_node_regression,
    _pooled_targeting,
    _single_arm_predictions,
    seed_carried,
    stitch_fold_assignments,
    untargeted_fold_recursions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..learners.super_learner import Learner

__all__ = [
    "LEARNED_REGIMEN_NODES",
    "LEARNED_REGIMEN_QUANTILE_LEVELS",
    "LEARNED_REGIMEN_TARGET",
    "LearnedRegimen",
    "LearnedRegimenNode",
    "LearnedRegimenRecord",
    "LearnedRules",
    "branch_plans",
    "fit_learned_regimen",
    "learn_fold_rules",
    "learned_plan",
    "learned_regimen_in",
    "learned_regimen_record",
    "learned_regimen_refusal",
    "refuse_learned_regimen_data",
    "refuse_learned_regimen_result",
    "refuse_learned_regimen_settings",
]

#: The decision nodes a learned regimen has.  More are X11 part (b2).
LEARNED_REGIMEN_NODES = 2

#: The quantile levels of each node's gap record.
LEARNED_REGIMEN_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5)

#: What the reported value of a learned regimen is.
LEARNED_REGIMEN_TARGET = (
    "fold-size-weighted data-adaptive value: sum over the outer folds of (n_v / n) times the "
    "value of the rule pair learned on that fold's training rows"
)

#: The roadmap part every learned-regimen refusal of an unbuilt composition cites.
_B2 = "X11 (b2)"

#: The working call every refusal of a learned-regimen setting can offer.
_LONE_FIT = (
    "Fit LTMLE(regimens=LearnedRegimen(...), n_folds=V) with V of at least 2, repeats=1, "
    "n_bootstrap=0, an end-of-study outcome and two binary treatment nodes."
)


@dataclass(frozen=True)
class LearnedRegimen:
    """Declare the two-node regimen whose fold-size-weighted value an ``LTMLE`` fit estimates.

    Each outer training fold learns its own pair of rules by backward plug-in Q-learning on its
    training rows: the second-node rule treats where the fold's outcome regression at the second
    node predicts a higher outcome under treatment, given the first-node arm, and the first-node
    rule treats where the first-node regression of that learned value predicts a higher one.  A
    tie assigns the lower arm code.  Every row then follows the rule pair of the fold that held it
    out.

    The fit reports one parameter, ``ey_regimen[<label>]``, the value of those rules weighted by
    each fold's share of the rows.  The target depends on the data, because the rules do (van der
    Laan and Luedtke 2015, Section 7).  ``result.learned_regimen`` records the rules.

    Pass it as ``LTMLE(regimens=LearnedRegimen())``, alone.  The fit needs two treatment nodes,
    each binary, an end-of-study outcome, at least two outer folds and explicit learners.  A
    ``SuperLearner`` or a cross-validated selector as the first node's regression is refused,
    because its internal risk would read a second-node rule trained on its own validation rows.

    Parameters
    ----------
    label : str
        The label that the reported parameter carries, as in
        ``ey_regimen[learned regimen]``.

    Raises
    ------
    DataError
        If ``label`` is not a non-empty string.

    See Also
    --------
    cleverly.longitudinal.LTMLE : The estimator that fits it.
    cleverly.longitudinal.DynamicRegimen : A regimen of known rules, fixed before the fit.

    Examples
    --------
    >>> from cleverly.longitudinal import LearnedRegimen
    >>> LearnedRegimen().label
    'learned regimen'
    """

    label: str = "learned regimen"

    #: Read by :func:`cleverly._support.regimen_kinds`, which keys the regimen kind ``learned``.
    learned: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise DataError(f"LearnedRegimen label= must be a non-empty string; got {self.label!r}")

    @property
    def has_policy(self) -> bool:
        """Return ``False``: every node of a learned regimen assigns its arm by a rule."""
        return False

    @property
    def has_tilt(self) -> bool:
        """Return ``False``: a learned regimen holds no odds tilt."""
        return False

    @property
    def has_mtp(self) -> bool:
        """Return ``False``: a learned regimen holds no modified treatment policy."""
        return False

    @property
    def has_rule(self) -> bool:
        """Return ``True``: every node holds a rule, learned in each training fold."""
        return True

    def is_rule(self, time: int) -> bool:
        """Return ``True``: node ``time`` holds a rule.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``True``.
        """
        return True

    def is_policy(self, time: int) -> bool:
        """Return ``False``: no node of a learned regimen draws its arm.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``False``.
        """
        return False

    def is_mtp(self, time: int) -> bool:
        """Return ``False``: no node holds a modified treatment policy.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``False``.
        """
        return False

    def is_tilt(self, time: int) -> bool:
        """Return ``False``: no node holds an odds tilt.

        Parameters
        ----------
        time : int
            The treatment node, counted from one.

        Returns
        -------
        bool
            ``False``.
        """
        return False


# --------------------------------------------------------------------------- refusals


def _gap(condition: str, alternative: str = _LONE_FIT, roadmap: str = "X11") -> CapabilityError:
    """The ``IMPLEMENTATION_GAP`` error of an unbuilt learned-regimen composition."""
    return CapabilityError.from_refusal(
        Refusal(
            RefusalReason.IMPLEMENTATION_GAP, condition, alternative=alternative, roadmap=roadmap
        )
    )


def _beside_refusal(others: Sequence[str]) -> CapabilityError:
    named = ", ".join(repr(label) for label in others)
    return _gap(
        f"a LearnedRegimen is fitted alone, and regimens= also declares {named}. A known regimen "
        "beside it, a stochastic, tilt or modified-treatment-policy node, their contrasts with "
        f"the learned regimen and the simultaneous band over them are not built ({_B2}): no "
        "witness covers the joint curve of a learned and a known regimen.",
        alternative=(
            "Fit LTMLE(regimens=LearnedRegimen(...)) alone, and the other regimens in a "
            "separate LTMLE fit."
        ),
    )


def learned_regimen_in(regimens: Any) -> LearnedRegimen | None:
    """The learned regimen that ``regimens=`` declares, or ``None`` when it declares none.

    A mapping value takes its key as its label, as a regimen in a mapping does.

    Parameters
    ----------
    regimens : Any
        The ``regimens=`` argument of a fit.

    Returns
    -------
    LearnedRegimen or None
        The declared learned regimen.

    Raises
    ------
    CapabilityError
        If another regimen is declared beside it (X11 (b2)).
    """
    if isinstance(regimens, LearnedRegimen):
        return regimens
    if isinstance(regimens, Mapping):
        items = [(str(key), value) for key, value in regimens.items()]
    elif isinstance(regimens, Sequence) and not isinstance(regimens, (str, bytes)):
        items = [(str(getattr(value, "label", value)), value) for value in regimens]
    else:
        return None
    learned = [(label, value) for label, value in items if isinstance(value, LearnedRegimen)]
    if not learned:
        return None
    if len(items) > 1:
        chosen = {label for label, _ in learned[:1]}
        raise _beside_refusal([label for label, _ in items if label not in chosen])
    label, value = learned[0]
    return value if isinstance(regimens, Sequence) else replace(value, label=label)


def _is_selector(learner: Any) -> bool:
    """Whether ``learner`` selects by an internal cross-validated risk, at any depth.

    A learner selects if it, or any estimator among its ``get_params(deep=True)`` values, is a
    ``SuperLearner``, has a class name ending in ``CV``, or holds a ``cv`` parameter that is
    not ``None``.  The parameter walk reaches ``Pipeline`` steps, ``TransformedTargetRegressor``
    and every other scikit-learn wrapper.
    """
    from ..learners.super_learner import SuperLearner

    seen: set[int] = set()

    def selects(candidate: Any) -> bool:
        if isinstance(candidate, type) or id(candidate) in seen:
            return False
        seen.add(id(candidate))
        if isinstance(candidate, SuperLearner) or type(candidate).__name__.endswith("CV"):
            return True
        get_params = getattr(candidate, "get_params", None)
        if not callable(get_params):
            return False
        try:
            params = get_params(deep=True)
        except TypeError:
            params = get_params()
        if params.get("cv") is not None:
            return True
        steps = params.get("steps")
        children = [step for _, step in steps] if isinstance(steps, list) else []
        children.extend(value for value in params.values() if hasattr(value, "fit"))
        return any(selects(child) for child in children)

    return selects(learner)


def refuse_learned_regimen_settings(estimator: Any) -> None:
    """Refuse an estimator setting that a learned regimen does not support, before any learner.

    The checks run in this order, and each refusal names its remedy:

    =====  ===========================================================  =================
    order  request                                                      item
    =====  ===========================================================  =================
    1      an estimator other than ``LTMLE``: ``SDR``, ``IPSI`` or a    X11 (b2)
           baseline
    2      ``msm=``                                                     X11 (b2)
    3      ``survival_model="hazard"`` or ``targeting="one_step"``      X11 (b2)
    4      ``repeats`` above 1                                          X11 (b2)
    5      ``n_folds=1``                                                X11 (b2)
    6      ``n_bootstrap`` above 0                                      F27
    7      a ``SuperLearner`` or a ``*CV`` selector as the first        X11 (b2)
           node's regression (``pseudo_learner``)
    =====  ===========================================================  =================

    Parameters
    ----------
    estimator : LTMLE
        The estimator, read and never modified.

    Raises
    ------
    CapabilityError
        If a setting meets a row of the table.
    """
    name = type(estimator).__name__
    if str(getattr(estimator, "_estimator", "")) != "tmle":
        raise _gap(
            f"{name} does not fit a LearnedRegimen ({_B2}). Only LTMLE learns a regimen in each "
            "training fold and targets it, with one logistic fluctuation per node.",
            alternative="Fit the learned regimen with LTMLE(regimens=LearnedRegimen(...)).",
        )
    if estimator.msm is not None:
        raise _gap(
            f"a LearnedRegimen is not projected onto a working model (msm=; {_B2}). Its one "
            "parameter is the value of the learned rules.",
            alternative="Omit msm=.",
        )
    if estimator.survival_model != "iterated_mean" or estimator.targeting != "iterative":
        raise _gap(
            "a LearnedRegimen is targeted by one logistic fluctuation per node, as van der Laan "
            "and Luedtke (2015), Appendix B, states it. The hazard model and its one-step "
            f"targeting (survival_model='hazard', targeting='one_step') are not built for it "
            f"({_B2}).",
            alternative="Keep survival_model='iterated_mean' and targeting='iterative'.",
        )
    if estimator.repeats > 1:
        raise _gap(
            "repeats above 1 is not combined with a LearnedRegimen. LTMLE reports the median of "
            "the draws, and the median of data-adaptive values over draws of the split "
            f"estimates no defined parameter ({_B2}).",
            alternative="Set repeats=1.",
        )
    if estimator.n_folds < 2:
        raise _gap(
            "a LearnedRegimen is learned inside each outer training fold, and n_folds=1 draws "
            "no split. The value of one regimen learned on every row is another target, which "
            f"van der Laan and Luedtke (2015), Theorem 5, covers at two nodes and which is not "
            f"built ({_B2}).",
            alternative="Set n_folds to 2 or more.",
        )
    if estimator.n_bootstrap:
        raise _gap(
            "the full-refit bootstrap is not combined with a LearnedRegimen: each resample would "
            "relearn the rules, so each draw has a different target (F27).",
            alternative=(
                "Set n_bootstrap=0. The fit reports the cross-fitted influence-curve interval."
            ),
            roadmap="F27",
        )
    pseudo = estimator._pseudo_learner(None)
    if _is_selector(pseudo):
        raise _gap(
            f"the first node's regression of a LearnedRegimen is {type(pseudo).__name__}, which "
            "selects by an internal cross-validated risk. That risk would read the second-node "
            "rule learned on the fold's whole training set, its own validation rows included, "
            "and the nested rule of Luedtke and van der Laan (2016), Sections 2.2 and 3.2, is "
            f"not built ({_B2}). A pseudo_learner of None takes outcome_learner, or the default "
            "Super Learner when that is None too.",
            alternative=(
                "Pass a fixed learner as pseudo_learner=, such as LinearRegression() or a "
                "gradient-boosting regressor with fixed settings."
            ),
        )


def refuse_learned_regimen_data(data: LongitudinalData) -> None:
    """Refuse a panel that a learned regimen does not support, before any learner.

    =====  ===========================================================  =================
    order  request                                                      item
    =====  ===========================================================  =================
    1      a survival, competing-risk or repeated outcome               X11 (b2)
    2      one baseline treatment held over every node                  X11 (b2)
    3      another number of decision nodes than two                    X11 (b2)
    4      a node that is not one binary treatment                      X11 (b2)
    5      ``weights=``                                                 X11 (b2)
    6      ``id=``                                                      X11 (b2)
    7      a Markov order                                               X11 (b2)
    =====  ===========================================================  =================

    Censoring after a node is admitted as part of that node's intervention, as in van der Laan
    and Luedtke (2015), and the fit reads experimental.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.

    Raises
    ------
    CapabilityError
        If the panel meets a row of the table.
    """
    if data.is_survival or data.is_competing or data.is_repeated:
        kind = (
            "a competing-risk"
            if data.is_competing
            else "a survival"
            if data.is_survival
            else "a repeated"
        )
        raise _gap(
            f"a LearnedRegimen needs one end-of-study outcome, and this fit declares {kind} "
            f"outcome. The rules of an event-indexed target are not built ({_B2}).",
        )
    if data.treatment_held:
        raise _gap(
            "a LearnedRegimen learns a rule at each of two treatment nodes, and this design holds "
            f"one baseline decision over every node ({_B2}).",
            alternative=(
                "For one decision, TMLE(learned_rule=LearnedRule()) learns a point-treatment rule."
            ),
        )
    if data.n_decisions != LEARNED_REGIMEN_NODES:
        raise _gap(
            f"a LearnedRegimen has two decision nodes, and this design has {data.n_decisions}. "
            "No source read by this project states the learned regimen beyond two nodes, and "
            f"one node is a point treatment ({_B2}).",
            alternative=(
                "Fit two nodes, or for one node TMLE(learned_rule=LearnedRule()) learns a "
                "point-treatment rule."
            ),
        )
    for time in range(1, data.n_times + 1):
        if data.node_kind(time) != "categorical" or len(data.treatment_levels[time - 1]) != 2:
            raise _gap(
                f"a LearnedRegimen learns a binary rule at each node, and node {time} "
                f"({data.treatment_names[time - 1]!r}) is not one binary treatment. A rule over "
                f"more arms or doses is not built ({_B2}).",
            )
    if data.is_weighted:
        raise _gap(
            "a LearnedRegimen is not combined with weights=: no construction of the learned "
            f"rules and their value is checked on a weighted panel ({_B2}).",
            alternative="Omit weights=.",
        )
    if data.cluster is not None:
        raise _gap(
            "a LearnedRegimen is not combined with id=: no construction of the learned rules "
            f"and their value is checked on a clustered panel ({_B2}).",
            alternative="Omit id=.",
        )
    if data.markov_order is not None:
        raise _gap(
            "a LearnedRegimen is not combined with a Markov order: the windowed outcome design "
            "reads a rule node from a declaration, and a learned rule has none "
            f"({_B2}).",
            alternative="Omit markov_order=.",
        )


def learned_regimen_refusal(result: Any, what: str) -> Refusal | None:
    """Why ``what`` is refused on a learned-regimen fit, or ``None``.

    A refit, a truncation replay and a refutation run the fit again, and so relearn the rules:
    each run then estimates the value of other rules (F27). The capability rows of those
    surfaces and the calls read this one record.

    Parameters
    ----------
    result : LongitudinalResult or LTMLE
        The fit, or the estimator a refit would run.
    what : str
        The surface, for the message, such as ``"the truncation curve"``.

    Returns
    -------
    Refusal or None
        The ``IMPLEMENTATION_GAP`` record (roadmap F27) when the fit holds a learned regimen
        or the estimator declares one, else ``None``.
    """
    if hasattr(result, "config"):
        learned = any(isinstance(regimen, LearnedRegimen) for regimen in result.config.regimens)
    else:
        learned = learned_regimen_in(getattr(result, "regimens", None)) is not None
    if not learned:
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"{what} is not available for a LearnedRegimen fit: it runs the fit again, which "
        "relearns the rules, so each run estimates the value of other rules (F27).",
        alternative=(
            "Fix the learned rules first, and fit them as DynamicRegimen(..., rule_kind='known') "
            "on independent data; result.learned_regimen.fold_rules holds them."
        ),
        roadmap="F27",
    )


def refuse_learned_regimen_result(result: Any, what: str) -> None:
    """Refuse a surface that would refit a learned-regimen fit.

    Parameters
    ----------
    result : LongitudinalResult or LTMLE
        The fit, or the estimator a refit would run.
    what : str
        The surface, for the message, such as ``"the truncation curve"``.

    Raises
    ------
    CapabilityError
        If the fit holds a learned regimen, or the estimator declares one, with the record
        of :func:`learned_regimen_refusal`.
    """
    refusal = learned_regimen_refusal(result, what)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)


# --------------------------------------------------------------------------- plans


def learned_plan(regimen: LearnedRegimen, data: LongitudinalData) -> Plan:
    """The plan of ``regimen`` before any rule is learned: every row assigned arm code 0.

    Parameters
    ----------
    regimen : LearnedRegimen
        The declaration.
    data : LongitudinalData
        The prepared panel.

    Returns
    -------
    Plan
        A placeholder whose assignment the fit replaces with each row's learned rule pair
        (:meth:`~cleverly.longitudinal.regimen.Plan.with_assignment`).
    """
    return Plan(regimen=regimen, values=np.zeros((data.n, data.n_times)))  # type: ignore[arg-type]


def branch_plans(plan: Plan) -> tuple[Plan, ...]:
    """The four static plans whose followers the Q-learning regressions are fitted on.

    The second node's regression at arms ``(a1, a2)`` is fitted on the followers of the static
    plan ``(a1, a2)``, and the first node's regression at ``a1`` on the followers of either plan
    that starts with ``a1``.  So the followers of these four plans are what every training fold
    needs, and the follower preflight runs on them.

    Parameters
    ----------
    plan : Plan
        The learned regimen's plan.

    Returns
    -------
    tuple of Plan
        The plans ``(0, 0)``, ``(0, 1)``, ``(1, 0)`` and ``(1, 1)``, under the regimen's label.
    """
    plans = []
    for first in (0.0, 1.0):
        for second in (0.0, 1.0):
            values = np.empty_like(plan.values, dtype=float)
            values[:, 0], values[:, 1] = first, second
            plans.append(plan.with_assignment(values))
    return tuple(plans)


# --------------------------------------------------------------------------- learning


@dataclass(frozen=True)
class LearnedRules:
    """Every fold's learned rule pair, on every row.

    Parameters
    ----------
    assignments : tuple of FloatArray
        Per outer fold, the ``(n, 2)`` arm codes of the fold's rule pair on every row.  The
        second column is ``d^{d_1(L_1)}_2(H_2)``, the second-node rule of the first-node arm
        the pair assigns.  It is 0 on a row censored after the first node, which no estimate
        reads.
    blips : tuple of FloatArray
        Per outer fold, the ``(n, 2)`` blips behind the rules: column 0 is
        ``Qbar_1(L_1, 1) - Qbar_1(L_1, 0)`` and column 1 is
        ``Qbar_2(H_2, d_1, 1) - Qbar_2(H_2, d_1, 0)`` at the first-node arm the pair assigns,
        ``nan`` on a row censored after the first node.
    second_node_blips : tuple of FloatArray
        Per outer fold, the ``(n, 2)`` second-node blips under each first-node arm ``a1``
        (column ``a1``), ``nan`` on a row censored after the first node.
    folds : Folds
        The outer split.
    """

    assignments: tuple[FloatArray, ...]
    blips: tuple[FloatArray, ...]
    second_node_blips: tuple[FloatArray, ...]
    folds: Folds

    def stitched(self) -> FloatArray:
        """Return each row's arm codes from the fold that held it out.

        Returns
        -------
        FloatArray
            The ``(n, 2)`` stitched assignment.
        """
        return stitch_fold_assignments(self.assignments, self.folds)


def _rule(blip: FloatArray) -> FloatArray:
    """``1{blip > 0}`` as arm codes: a tie, and an undefined blip, assign the lower arm."""
    return np.where(np.nan_to_num(blip, nan=0.0) > 0.0, 1.0, 0.0)


def _learn_in_fold(
    data: LongitudinalData,
    plan: Plan,
    *,
    scaler: OutcomeScaler,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    train: BoolArray,
    fold: int | None,
    bounds: Mapping[int, NodeBound] | None,
    second_rule: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One fold's backward plug-in Q-learning on its training rows.

    ``second_rule`` replaces the learned second-node rule in the first node's target, for the
    witness that the first-node rule reads the learned one.  ``None`` is the estimator.

    Returns
    -------
    tuple of FloatArray
        The ``(n, 2)`` assignment, the ``(n, 2)`` blips at it, and the ``(n, 2)`` second-node
        blips under each first-node arm.
    """
    inner = Folds.single(data.n)
    bounds = {} if bounds is None else bounds
    seed = seed_carried(data, scaler, horizon=LEARNED_REGIMEN_NODES)
    first_arm = np.asarray(data.node_codes(1), dtype=float)
    # The second node's regressions run under the observed first-node arm, so every row
    # uncensored after the first node is at risk there and is predicted.  The fit rows of arm
    # ``a1`` are restricted to the training rows that took ``a1``: the regression is the
    # stratified one of the static plan ``(a1, a2)``, and the prediction reaches every row.
    observed = np.zeros_like(plan.values, dtype=float)
    observed[:, 0] = first_arm
    observed_plan = plan.with_assignment(observed)
    at_risk = observed_plan.masks(data).at_risk(2)
    rows = np.arange(data.n)
    second = np.full((data.n, 2), np.nan)
    first = np.full((data.n, 2), np.nan)
    second_rules = np.zeros((data.n, 2))
    for arm in (0, 1):
        took = np.asarray(first_arm == arm)
        with phase("outcome_learner_fit"):
            arms = _single_arm_predictions(
                data,
                observed_plan,
                seed,
                2,
                LEARNED_REGIMEN_NODES,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                folds=inner,
                cause=None,
                fit_rows=train & took,
                outer_fold=fold,
                n_jobs=1,
                # The Q-learning clips to [0, 1], the scale every recursion runs on: a binary
                # outcome, or a continuous one rescaled by q_bounds, so it agrees with the
                # estimator's own clip there.
                clip=(0.0, 1.0),
                bound=bounds.get(2),
            )
        blip = np.where(at_risk, arms[:, 1] - arms[:, 0], np.nan)
        rule = _rule(blip) if second_rule is None else np.asarray(second_rule, dtype=float)
        second[:, arm] = blip
        second_rules[:, arm] = rule
        carried = np.where(at_risk, arms[rows, rule.astype(np.int64)], _FILLER)
        values = np.zeros_like(plan.values, dtype=float)
        values[:, 0] = float(arm)
        values[:, 1] = rule
        branch = plan.with_assignment(values)
        node = _fit_node_regression(
            data,
            branch,
            carried,
            1,
            LEARNED_REGIMEN_NODES,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=inner,
            cause=None,
            masks=branch.masks(data),
            fit_rows=train,
            outer_fold=fold,
            bound=bounds.get(1),
        )
        first[:, arm] = node.initial
    first_blip = first[:, 1] - first[:, 0]
    first_rule = _rule(first_blip)
    chosen = first_rule.astype(np.int64)
    assignment = np.column_stack([first_rule, second_rules[rows, chosen]])
    blips = np.column_stack([first_blip, second[rows, chosen]])
    return assignment, blips, second


def learn_fold_rules(
    data: LongitudinalData,
    plan: Plan,
    *,
    folds: Folds,
    scaler: OutcomeScaler,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    n_jobs: int = 1,
    bounds: Mapping[int, NodeBound] | None = None,
) -> LearnedRules:
    """Learn every outer fold's rule pair on its own training rows, one job per fold.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The learned regimen's plan.
    folds : Folds
        The outer split, with at least two folds.
    scaler : OutcomeScaler
        The outcome transformation.
    outcome_learner : Learner
        The second node's regression.
    pseudo_learner : Learner
        The first node's regression.
    n_jobs : int, default=1
        Parallel workers across the outer folds.
    bounds : mapping of int to NodeBound or None, default=None
        The declared outcome bound of each bounded node.

    Returns
    -------
    LearnedRules
        Every fold's rule pair and blips.
    """
    wanted = profiling()

    def run_fold(
        fold: int, train: IntArray
    ) -> tuple[tuple[FloatArray, FloatArray, FloatArray], PhaseProfile | None]:
        rows = np.zeros(data.n, dtype=bool)
        rows[train] = True
        with collect_phases(wanted) as profile:
            learned = _learn_in_fold(
                data,
                plan,
                scaler=scaler,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                train=rows,
                fold=fold,
                bounds=bounds,
            )
        return learned, profile

    jobs = [(fold, train) for fold, (train, _) in enumerate(folds)]
    with phase("outer_fold_recursion"):
        outcomes = map_parallel(run_fold, jobs, n_jobs=n_jobs)
    for _, profile in outcomes:
        merge_worker_phases(profile)
    return LearnedRules(
        assignments=tuple(learned[0] for learned, _ in outcomes),
        blips=tuple(learned[1] for learned, _ in outcomes),
        second_node_blips=tuple(learned[2] for learned, _ in outcomes),
        folds=folds,
    )


def fit_learned_regimen(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    rules: LearnedRules,
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    alpha: float,
    max_iter: int,
    tol: float,
    n_jobs: int = 1,
    stopping: str = "relative",
    step_bound: float | None = None,
    step_search: bool = True,
    iterate: str = "last",
    node_bounds: Mapping[int, NodeBound] | None = None,
) -> RegimenFit:
    """Target the learned regimen: each fold's recursion under its rules, then one pooled pass.

    Each fold's untargeted recursion runs under that fold's rule pair on its training rows
    (:attr:`~cleverly.longitudinal.sequential.FoldRecursionCell.fold_assignments`), which is the
    shipped recursion of a known rule.  Each row's held-out prediction is stitched, and one
    pooled logistic fluctuation per node, node 2 then node 1, targets them under the stitched
    plan, with the mechanism evaluated at each row's own rule.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The stitched plan: each row's arms from the fold that held it out.
    mechanism : Mechanism
        The out-of-fold mechanism, evaluated at ``plan``.
    rules : LearnedRules
        Every fold's rule pair.
    outcome_learner : Learner
        The regression at the second node.
    pseudo_learner : Learner
        The regression at the first node.
    scaler : OutcomeScaler
        The outcome transformation.
    g_bounds : tuple of float
        The cumulative mechanism bounds.
    alpha : float
        Probability bound of the logistic submodel.
    max_iter : int
        Largest number of Newton iterations per node.
    tol : float
        Relative-score convergence tolerance.
    n_jobs : int, default=1
        Parallel workers across the outer folds.
    stopping : {"relative", "scaled_var", "sample_size"}, default="relative"
        The rule that decides each node's pooled score equation is solved.
    step_bound : float or None, default=None
        The longest step of one iteration of a node's targeting.
    step_search : bool, default=True
        Whether a bounded step searches its length.
    iterate : {"last", "best"}, default="last"
        Which iterate of a node's walk is reported.
    node_bounds : mapping of int to NodeBound or None, default=None
        The declared outcome bound of each bounded node.

    Returns
    -------
    RegimenFit
        The pooled-targeted fit.
    """
    horizon = LEARNED_REGIMEN_NODES
    cumulative_unbounded, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    with phase("mask_construction"):
        masks = plan.masks(data)
    (stitched,) = untargeted_fold_recursions(
        data,
        (FoldRecursionCell(plan, masks, horizon, node_bounds, fold_assignments=rules.assignments),),
        folds=rules.folds,
        scaler=scaler,
        cause=None,
        outcome_learner=outcome_learner,
        pseudo_learner=pseudo_learner,
        n_jobs=n_jobs,
    )
    steps = _pooled_targeting(
        data,
        plan,
        masks,
        cumulative,
        stitched,
        numerator=None,
        scaler=scaler,
        horizon=horizon,
        cause=None,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        stopping=stopping,
        step_bound=step_bound,
        step_search=step_search,
        iterate=iterate,
        node_bounds=node_bounds,
    )
    return _finish_regimen_fit(
        data,
        plan,
        steps,
        cumulative_unbounded,
        cumulative,
        horizon=horizon,
        cause=None,
        numerator=None,
    )


# --------------------------------------------------------------------------- record


@dataclass(frozen=True)
class LearnedRegimenNode:
    """Record one decision node of a learned regimen: what its fold rules did.

    Parameters
    ----------
    node : int
        The decision node, counted from one.
    treatment : str
        The node's treatment column.
    arms : tuple of str
        The node's two treatment levels in arm-code order; the rule treats with the second.
    treated_shares : tuple of float
        For each fold, the share of its validation rows at risk at this node under the fold's
        own rule pair that the fold's rule treats.  At the first node every row is at risk.
    gap_quantiles : tuple of tuple of float
        For each fold, quantiles of the absolute blip behind the rule on the same rows, at
        :data:`LEARNED_REGIMEN_QUANTILE_LEVELS`, on the scale the regressions are fitted on.
        Mass near 0 speaks against the margin condition of a fixed rule limit.
    rule_disagreement : float
        The share of rows at which the fold rules do not all assign the same arm, read from
        every fold's rule on every row (at the second node, on the rows uncensored after the
        first node).  At the second node a fold's code is the second rule of the first-node
        arm that fold chose, so the share also counts rows whose folds chose different
        first-node arms.  Rules that settle to one limit agree almost everywhere.
    """

    node: int
    treatment: str
    arms: tuple[str, str]
    treated_shares: tuple[float, ...]
    gap_quantiles: tuple[tuple[float, ...], ...]
    rule_disagreement: float


@dataclass(frozen=True, eq=False)
class LearnedRegimenRecord:
    """Record what a learned-regimen fit estimated, under ``result.learned_regimen``.

    The fitted fold models are not kept.  Each fold's rule pair on every row is, so a refit of
    a fold's rules can be checked against it, and the stitched assignment each row followed is
    ``result.fits[...].assignment``.

    Parameters
    ----------
    label : str
        The label of the regimen.
    seed : int or None
        The seed of the split and the learners, as ``result.repeats[0].seed`` records it.
    fold_sizes : tuple of int
        The validation rows of each outer fold.
    nodes : tuple of LearnedRegimenNode
        One record per decision node, in time order.
    fold_rules : numpy.ndarray
        The ``(V, n, 2)`` arm codes of every fold's rule pair on every row, as
        :attr:`LearnedRules.assignments` holds them.
    fold_blips : numpy.ndarray
        The ``(V, n, 2)`` blips behind them, as :attr:`LearnedRules.blips` holds them.

    Attributes
    ----------
    n_folds : int
    fold_weights : tuple of float
    target : str
    quantile_levels : tuple of float
    """

    label: str
    seed: int | None
    fold_sizes: tuple[int, ...]
    nodes: tuple[LearnedRegimenNode, ...]
    fold_rules: np.ndarray = field(repr=False)
    fold_blips: np.ndarray = field(repr=False)

    @property
    def n_folds(self) -> int:
        """Return the number of outer folds."""
        return len(self.fold_sizes)

    @property
    def fold_weights(self) -> tuple[float, ...]:
        """Return each fold's weight ``n_v / n`` in the target and the estimate."""
        total = float(sum(self.fold_sizes))
        return tuple(float(size) / total for size in self.fold_sizes)

    @property
    def target(self) -> str:
        """Return what the reported value is (:data:`LEARNED_REGIMEN_TARGET`)."""
        return LEARNED_REGIMEN_TARGET

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the levels of each node's gap quantiles."""
        return LEARNED_REGIMEN_QUANTILE_LEVELS

    def describe(self) -> str:
        """Return the line that a summary prints for this fit.

        Returns
        -------
        str
            The target, the fold count, how the rules are learned, the range of each node's
            treated share over the folds, and the share of rows on which the fold rules
            disagree.
        """
        weights = self.fold_weights
        shares = " ".join(
            f"Node {node.node} ({node.treatment}): the rules treat {min(node.treated_shares):.3g}"
            f" to {max(node.treated_shares):.3g} of the rows at risk by fold, and disagree on "
            f"{node.rule_disagreement:.3g} of the rows."
            for node in self.nodes
        )
        return (
            f"Target: the values of the {self.n_folds} rule pairs learned on the training folds, "
            "each pair evaluated on the fold it did not train on and pooled over every "
            f"validation row, weighted by each fold's share of the rows (fold weights "
            f"{min(weights):.3g} to {max(weights):.3g}). The target depends on the data, "
            "because the rules do (van der Laan and Luedtke 2015, Section 7). Each pair is "
            "learned by backward plug-in Q-learning of the fold's own regressions: the "
            "second-node rule given the first-node arm, then the first-node rule under it. "
            f"{shares}"
        )


def _quantiles(values: FloatArray) -> tuple[float, ...]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return tuple(float("nan") for _ in LEARNED_REGIMEN_QUANTILE_LEVELS)
    return tuple(
        float(value) for value in np.quantile(np.abs(finite), LEARNED_REGIMEN_QUANTILE_LEVELS)
    )


def _disagreement(columns: FloatArray, rows: BoolArray) -> float:
    if not rows.any():
        return 0.0
    picked = columns[:, rows]
    return float(np.mean(np.any(picked != picked[:1], axis=0)))


def learned_regimen_record(
    data: LongitudinalData,
    plan: Plan,
    rules: LearnedRules,
    *,
    seed: int | None,
) -> LearnedRegimenRecord:
    """Build the record of a learned-regimen fit from its fold rules.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The learned regimen's plan.
    rules : LearnedRules
        Every fold's rule pair.
    seed : int or None
        The seed of the split.

    Returns
    -------
    LearnedRegimenRecord
        The record.
    """
    folds = rules.folds
    assignments = np.stack(rules.assignments)
    blips = np.stack(rules.blips)
    uncensored = plan.with_assignment(np.zeros_like(plan.values)).masks(data).uncensored[:, 1]
    first_arm = np.asarray(data.node_codes(1), dtype=float)
    shares: list[list[float]] = [[], []]
    gaps: list[list[tuple[float, ...]]] = [[], []]
    for fold, (_, test) in enumerate(folds):
        held = np.zeros(data.n, dtype=bool)
        held[test] = True
        at_risk = (
            held,
            held & uncensored & (first_arm == assignments[fold, :, 0]),
        )
        for column in (0, 1):
            rows = at_risk[column]
            treated = assignments[fold, rows, column]
            shares[column].append(float(np.mean(treated)) if treated.size else float("nan"))
            gaps[column].append(_quantiles(blips[fold, rows, column]))
    nodes = tuple(
        LearnedRegimenNode(
            node=column + 1,
            treatment=str(data.treatment_names[column]),
            arms=tuple(str(level) for level in data.treatment_levels[column]),  # type: ignore[arg-type]
            treated_shares=tuple(shares[column]),
            gap_quantiles=tuple(gaps[column]),
            rule_disagreement=_disagreement(
                assignments[:, :, column],
                np.ones(data.n, dtype=bool) if column == 0 else np.asarray(uncensored),
            ),
        )
        for column in (0, 1)
    )
    return LearnedRegimenRecord(
        label=plan.label,
        seed=seed,
        fold_sizes=tuple(int(np.sum(folds.assignment == fold)) for fold in range(folds.n_folds)),
        nodes=nodes,
        fold_rules=assignments.astype(np.int8),
        fold_blips=blips,
    )
