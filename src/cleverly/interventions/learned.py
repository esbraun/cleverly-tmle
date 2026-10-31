r"""The value of a treatment rule learned inside each training fold.

A :class:`~cleverly.interventions.Rule` is a *known* function of the covariates, and a rule
learned from the analysis sample is refused there.  This module holds
the one learned-rule target the package estimates instead.  The target is data-adaptive:

.. math::

    \tilde\psi_{0n} = \frac1V \sum_{v=1}^V \Psi_{d_{nv}}(P_0),

where :math:`d_{nv}` is the rule learned on the training rows of outer fold :math:`v`.  The
default rule is the plug-in rule
:math:`d_{nv}(w) = \mathbb 1\{\bar Q_{nv}(1, w) - \bar Q_{nv}(0, w) > 0\}` of the outcome
regression :math:`\bar Q_{nv}` fitted on those rows.  ``LearnedRule(loss="blip")`` and
``LearnedRule(loss="weighted_classification")`` learn it with a declared learner from a doubly
robust pseudo-outcome instead (Luedtke and van der Laan 2016, *International Journal of
Biostatistics*, Sections 4.1 to 4.3), which :mod:`cleverly.interventions._rule_learners` fits.
Van der Laan and Luedtke (2015), Section 7 and Appendix B, define the target and its
cross-validated TMLE for any rule learned on the training rows.  The estimator reuses the
regime fluctuation with one pooled :math:`\varepsilon`, the fold-evaluated report and the
cross-validated variance (``TMLE(cv_evaluation=True)``).  This module adds the fold-local rule,
the result record, the mean over a fixed number of splits (:func:`mean_over_splits`) and the
refusals.  No new submodel, fluctuation or influence curve enters.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import CapabilityError, DataError, Refusal, RefusalReason
from ._rule_learners import InnerFit, inner_seed, rule_learner_treat
from .base import RegimeSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators._nuisance import NuisanceEstimates
    from ..inference.influence import ParameterEstimate
    from ..learners.crossfit import Folds

__all__ = [
    "BLIP_QUANTILE_LEVELS",
    "LEARNED_RULE_BENCHMARK_REFUSAL",
    "LEARNED_RULE_CONFIGURATION",
    "LEARNED_RULE_EVALUE_REFUSAL",
    "LEARNED_RULE_LIMITS",
    "LEARNED_RULE_REMEDY",
    "LEARNED_RULE_TILT_REFUSAL",
    "LOSSES",
    "RULE_CLASS",
    "TARGET_KIND",
    "FrozenLearnedRule",
    "LearnedRule",
    "LearnedRuleRecord",
    "LearnedRuleSplit",
    "check_fold_rules",
    "fold_rules",
    "freeze_learned_rule",
    "frozen_rule_replay",
    "learned_rule_configuration_refusal",
    "learned_rule_record",
    "learned_rule_scheme_refusal",
    "learned_rule_split",
    "learned_rule_splits_refusal",
    "learned_rule_treat",
    "mean_over_splits",
    "refuse_learned_rule_composition",
]

#: The loss of each rule learner, the ``loss=`` of :class:`LearnedRule`.
LOSSES: tuple[str, ...] = ("plug_in", "blip", "weighted_classification")

#: The rule class of each loss, in words.  The record quotes the one of its fit.  ``1`` and
#: ``0`` stand for the higher and the lower of the two sorted arm codes.
_TIE = "where 1 is the higher arm code and 0 the lower; a tie assigns the lower arm code"
_PSEUDO = (
    "the doubly robust blip pseudo-outcome D = (2A - 1) / g(A | W) (Y - Qbar(A, W)) + "
    "Qbar(1, W) - Qbar(0, W) of each training row, on the scaled outcome, with Qbar and g "
    "from a 5-fold cross-fit inside the training rows (one fold per cluster when they hold "
    "fewer than 5 clusters) and g truncated at the fit's resolved g_bounds, resolved at the "
    "full sample's effective size, the value result.config records"
)
_WEIGHTED = (
    "Under weights= every fit inside the training rows is weighted, so the rule is the rule "
    "of the weighted population"
)
RULE_CLASS: dict[str, str] = {
    "plug_in": (
        "the plug-in rule 1{Qbar_v(1, W) - Qbar_v(0, W) > 0} of the outcome regression fitted "
        f"on the training rows of each outer fold, {_TIE}. Under weights= the regression is "
        "the weighted one, so the rule is the plug-in rule of the weighted population"
    ),
    "blip": (
        "the rule 1{b_v(W) > 0} of the declared learner b_v, a regression of "
        f"{_PSEUDO} on W, fitted on the training rows of each outer fold, {_TIE}. {_WEIGHTED}"
    ),
    "weighted_classification": (
        "the rule 1{P_v(D > 0 | W) > 1/2} of the declared classifier P_v of the sign of "
        f"{_PSEUDO}, fitted on the training rows of each outer fold with sample weights |D|, "
        f"{_TIE}. {_WEIGHTED}"
    ),
}

#: What the reported value is.  The record and the summary quote it.
TARGET_KIND = "fold-average data-adaptive value"

#: The quantile levels of the estimated blip that the record keeps for each fold.
BLIP_QUANTILE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

#: The configuration that fits the learned-rule value.  Every refusal whose remedy is a
#: configuration names it, and ``tests/unit/test_learned_rule_refusals.py`` fits it.
LEARNED_RULE_CONFIGURATION = (
    "cross_fit=True, cv_evaluation=True on the engine, or "
    "CrossFitting(enabled=True, fold_evaluation=True) on the method, with n_folds of at least "
    "2 and n_bootstrap=0 (Inference(n_bootstrap=0) on the method)"
)

#: The way out of a shared refusal that a drawn split raises.  The learned-rule value has
#: no in-sample fit (X11 (a)), so the in-sample remedy of those refusals does not apply.
LEARNED_RULE_REMEDY = (
    "fit the learned-rule value on a larger sample, because it has no in-sample fit "
    "(X11 (a) in docs/roadmap.md)"
)

#: The roadmap part that holds the limits and the benchmark of the learned-rule bound.
_X11_N2 = "X11 (n2) in docs/roadmap.md"

#: The reason every sensitivity analysis of a learned-rule fit shares.  Each analysis
#: appends the roadmap row that holds its own missing work.
_LEARNED_RULE_TARGET = (
    "this sensitivity analysis of the learned-rule value is not implemented. The target averages "
    "the values of rules learned inside each training fold, and a refit relearns those "
    "rules, so a refitted or perturbed fit estimates a different target"
)

#: Why ``benchmark`` refuses a learned-rule fit.  The omitted-variable bound itself reads
#: the fit's own fold rules, held fixed, and needs no refit.
LEARNED_RULE_BENCHMARK_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "benchmark refits the model without the named covariates, and on a learned-rule fit the "
    "refit relearns the rule inside each training fold, so the short model would estimate "
    "the value of other rules. A short model that holds the fit's rules fixed needs every "
    "fold's rule at every row, and the frozen-rule replay holds one realized rule per row "
    f"({_X11_N2}).",
    alternative=(
        "Read omitted_variable_bounds() and robustness_value() at declared strengths, which "
        "hold the fit's fold rules fixed and refit nothing."
    ),
    roadmap="X11",
)

#: Why the omitted-variable bound of a learned-rule fit publishes no confidence limit.
LEARNED_RULE_LIMITS = (
    "a learned-rule fit: the omitted-variable bound of its value is a point bound. Its estimate of "
    "nu^2 evaluates every fold's rule at every row, so V - 1 of the V rules read rows they "
    "were learned on, and a standard error would need a Donsker-type condition on the rule "
    "learners, which condition C5 of the learned-rule contract does not make. Replacing the "
    "fold rules by their common limit would instead need a margin rate, which condition C3 "
    f"does not give ({_X11_N2})."
)

#: Why the E-value refuses a learned-rule fit.  No E-value of a regime contrast ships.
LEARNED_RULE_EVALUE_REFUSAL = (
    f"{_LEARNED_RULE_TARGET}. An E-value reads an arm contrast, and no bound maps a "
    "learned-rule contrast to one (F27 in docs/roadmap.md)"
)

#: Why the missingness tilt refuses a learned-rule fit, which refuses missing outcomes.
LEARNED_RULE_TILT_REFUSAL = (
    f"{_LEARNED_RULE_TARGET}. A learned-rule fit refuses missing outcomes before any "
    "learner, so it has no observation mechanism to tilt (F27 in docs/roadmap.md)"
)

_X11 = "X11 ({part}) in docs/roadmap.md"
_F27 = "F27 in docs/roadmap.md"

#: The estimator call that takes a fixed rule, the alternative of a learned-rule refusal that
#: a fixed rule does not meet.
_FIXED_RULE = "TMLE(interventions=[Rule(..., rule_kind='known')])"


def _gap(condition: str, alternative: str, roadmap: str) -> Refusal:
    """An ``IMPLEMENTATION_GAP`` record of the learned-rule refusal table.

    Parameters
    ----------
    condition : str
        The sentences that state the failed condition.
    alternative : str
        The sentence that names a working call.
    roadmap : str
        The ``docs/roadmap.md`` id that tracks the missing capability.

    Returns
    -------
    Refusal
        The record.
    """
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP, condition, alternative=alternative, roadmap=roadmap
    )


@dataclass(frozen=True)
class LearnedRule:
    """Declare the rule whose fold-average value a fit estimates.

    The rule is learned inside each outer training fold.  ``1`` and ``0`` stand for the
    higher and the lower of the two sorted arm codes, and a tie assigns the lower arm code.
    ``loss=`` picks how the rule is learned:

    - ``"plug_in"``, the default: the plug-in rule of the fit's own outcome regression,
      ``d_v(w) = 1{Qbar_v(1, w) - Qbar_v(0, w) > 0}``, where ``Qbar_v`` is fitted on the
      training rows of fold ``v``.  It needs no learner beyond the outcome regression and
      no fit beyond the cross-fitted nuisances.
    - ``"blip"``: ``learner`` regresses a doubly robust blip pseudo-outcome on the
      covariates, and the rule treats where the fitted blip is positive.
    - ``"weighted_classification"``: ``learner`` classifies the sign of the same
      pseudo-outcome with its absolute value as the sample weight, and the rule treats where
      the fitted probability of a positive sign exceeds one half.

    The pseudo-outcome of a training row is
    ``(2A - 1) / g(A | W) (Y - Qbar(A, W)) + Qbar(1, W) - Qbar(0, W)``, with ``Qbar`` and
    ``g`` from a 5-fold cross-fit inside the training rows that uses the fit's own outcome
    and treatment learners (Luedtke and van der Laan 2016, *International Journal of
    Biostatistics*, Sections 4.1 to 4.3).

    The object has no ``density`` method, so it never enters ``interventions=`` as a known
    regime.  Pass it as ``TMLE(learned_rule=LearnedRule())`` or
    ``LearnedRuleValue(rule=LearnedRule())``.

    Parameters
    ----------
    name : str
        The label that the reported parameter carries, as in
        ``ey_learned_rule[learned rule]``.
    loss : {"plug_in", "blip", "weighted_classification"}
        How the rule is learned.
    learner : object or None
        The rule learner of ``loss="blip"``, a regressor, or of
        ``loss="weighted_classification"``, a classifier whose ``fit`` takes
        ``sample_weight``.  It is a template, cloned for each fold.  ``None`` for
        ``loss="plug_in"``, which refuses a learner.

    Raises
    ------
    DataError
        If ``name`` is empty or ``loss`` is not one of the three.
    CapabilityError
        If ``learner`` does not suit ``loss``: a learner for the plug-in rule, no learner
        for a pseudo-outcome loss, a classifier for the blip, a regressor for weighted
        classification, a classifier without ``sample_weight``, or a ``SuperLearner`` whose
        task or clip contradicts the loss.

    See Also
    --------
    cleverly.LearnedRuleValue : The typed estimand that holds this rule.
    cleverly.interventions.Rule : A known rule, fixed before the fit.

    Examples
    --------
    >>> from cleverly.interventions import LearnedRule
    >>> LearnedRule().name
    'learned rule'

    A blip rule learned by a linear regression on the covariates:

    >>> from sklearn.linear_model import LinearRegression
    >>> LearnedRule(loss="blip", learner=LinearRegression()).loss
    'blip'
    """

    name: str = "learned rule"
    loss: str = field(default="plug_in", kw_only=True)
    learner: Any = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataError(f"LearnedRule name= must be a non-empty string; got {self.name!r}")
        if self.loss not in LOSSES:
            raise DataError(f"LearnedRule loss= must be one of {list(LOSSES)}; got {self.loss!r}")
        refusal = _learner_refusal(self.loss, self.learner)
        if refusal is not None:
            raise CapabilityError.from_refusal(refusal)


def _learner_refusal(loss: str, learner: Any) -> Refusal | None:
    """The refusal of a rule learner that does not suit ``loss``, or ``None``."""
    from sklearn.base import is_classifier, is_regressor

    from ..learners.library import _validate_learner
    from ..learners.super_learner import SuperLearner

    def invalid(condition: str, alternative: str) -> Refusal:
        return Refusal(RefusalReason.INVALID_DECLARATION, condition, alternative=alternative)

    if loss == "plug_in":
        if learner is None:
            return None
        return invalid(
            "LearnedRule(loss='plug_in') takes no learner=. The plug-in rule reads the fit's "
            "own outcome regression, so a rule learner has no role in it.",
            "Drop learner=, or pass loss='blip' or loss='weighted_classification' to learn "
            "the rule with it.",
        )
    if learner is None:
        return invalid(
            f"LearnedRule(loss={loss!r}) needs learner=, the model that learns the rule from "
            "the pseudo-outcome. No default applies, because the default nuisance library "
            "clips its predictions to [0, 1] and the blip takes negative values.",
            "Pass learner=, such as LinearRegression() for loss='blip' or "
            "LogisticRegression() for loss='weighted_classification'.",
        )
    _validate_learner(learner, "LearnedRule learner=", optional=False)
    super_task = learner.task if isinstance(learner, SuperLearner) else None
    if loss == "blip":
        if is_classifier(learner) or super_task == "classification":
            return invalid(
                "LearnedRule(loss='blip') regresses a real-valued pseudo-outcome, and "
                f"{type(learner).__name__} is a classifier.",
                "Pass a regressor, or loss='weighted_classification' for a classifier.",
            )
        if isinstance(learner, SuperLearner) and learner.clip is not None:
            return invalid(
                "LearnedRule(loss='blip') regresses a blip that takes both signs, and this "
                f"SuperLearner clips its predictions to {learner.clip}, which would move the "
                "sign of the fitted blip.",
                "Pass a SuperLearner with clip=None.",
            )
        return None
    if is_regressor(learner) or super_task == "regression":
        return invalid(
            "LearnedRule(loss='weighted_classification') classifies the sign of the "
            f"pseudo-outcome, and {type(learner).__name__} is a regressor.",
            "Pass a classifier, or loss='blip' for a regressor.",
        )
    unweighted = _unweighted_member(learner)
    if unweighted is not None:
        return invalid(
            "LearnedRule(loss='weighted_classification') weights each row by the absolute "
            f"pseudo-outcome, and {unweighted} takes no sample_weight. "
            "Fitted without the weights, the classifier learns the majority sign of the "
            "pseudo-outcome, which is a different rule.",
            "Pass a classifier whose fit takes sample_weight, such as LogisticRegression(), "
            "and for a SuperLearner a library whose every member's fit takes it.",
        )
    return None


def _unweighted_member(learner: Any) -> str | None:
    """Name the part of ``learner`` whose ``fit`` takes no ``sample_weight``, or ``None``.

    A ``SuperLearner`` passes ``sample_weight`` to each member that takes it, and fits a
    member that does not unweighted, with a warning.  So a ``SuperLearner`` with an
    explicit library is checked member by member.  Its default library takes weights.
    """
    from ..learners._fitting import supports_sample_weight
    from ..learners.super_learner import SuperLearner

    if not supports_sample_weight(learner):
        return f"{type(learner).__name__}.fit"
    if isinstance(learner, SuperLearner) and learner.library is not None:
        for member in learner.library:
            estimator = member[1] if isinstance(member, tuple) else member
            if not supports_sample_weight(estimator):
                return f"the SuperLearner member {type(estimator).__name__}, whose fit"
    return None


@dataclass(frozen=True)
class FrozenLearnedRule(LearnedRule):
    """The realized fold rules of a learned-rule fit, held fixed for a replay.

    A replay that relearns the rule inside each training fold estimates the value of other
    rules, so its target moves with the perturbation.  This rule carries the realized rule of
    every row of the original fit instead.  A refit with the original folds then evaluates
    the value of these fixed rules fold-wise, with the ``1/V`` weights of the learned-rule
    record.  Simulated confounding and the refutations that refit read it as confounding of
    the value of these fixed rules, not of the learned-rule procedure.

    Parameters
    ----------
    name : str
        The label of the rule, as the original fit reported it.
    loss : str
        The loss of the original rule, kept so that the replay's record names it.  The
        replay learns no rule.
    learner : object or None
        The rule learner of the original rule, kept for the same reason.
    treat : numpy.ndarray
        One boolean per row of the original fit: ``True`` where that row's fold rule assigns
        the higher arm code.  A subset or resampled refit passes the value of each kept row's
        source row.
    """

    treat: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        values = np.asarray(self.treat)
        if values.ndim != 1 or values.dtype != bool:
            raise DataError("FrozenLearnedRule treat= holds one boolean per row")
        object.__setattr__(self, "treat", values.copy())

    def rows(self, rows: np.ndarray) -> FrozenLearnedRule:
        """Return the frozen rule of the rows ``rows`` of the original fit, in that order.

        Parameters
        ----------
        rows : numpy.ndarray
            Integer positions in the original fit.  A position may repeat, and then the
            rule value repeats with it.

        Returns
        -------
        FrozenLearnedRule
            The rule of the selected rows.
        """
        return replace(self, treat=self.treat[np.asarray(rows, dtype=int)])


def learned_rule_splits_refusal(
    operation: str,
    n_splits: int,
    *,
    reads: str = "reads one learned rule per row",
    instead: str = "One rule per row would answer for the target of one split instead.",
) -> Refusal | None:
    """Return the refusal of a surface that reads one split of a learned-rule fit, or ``None``.

    A learned-rule fit with ``repeats=R`` above 1 reports the mean over ``R`` splits.  Each
    split learns its own fold rules, so a row has ``R`` rules, not one.  A surface that
    freezes, reports or replays one rule per row would answer for the target of one split,
    and a surface that combines the splits by their median would answer for no defined
    parameter.

    Parameters
    ----------
    operation : str
        The surface, named as the caller writes it.
    n_splits : int
        How many splits the fit combines.
    reads : str
        What the surface reads, after its name.
    instead : str
        The sentence that names what the surface would answer for.

    Returns
    -------
    Refusal or None
        A ``DIFFERENT_ESTIMAND`` record whose remedy is a refit with ``repeats=1``, or
        ``None`` for a fit of one split.
    """
    if n_splits == 1:
        return None
    return Refusal(
        RefusalReason.DIFFERENT_ESTIMAND,
        f"{operation} {reads}, and this learned-rule fit reports the mean over {n_splits} "
        f"splits (repeats={n_splits}). Each split learns its own fold rules, so each row has "
        f"{n_splits} rules, and the reported target is the mean of the {n_splits} split "
        f"targets. {instead}",
        alternative=(
            "Refit with repeats=1 (CrossFitting(repeats=1) on the method), and run "
            f"{operation} on that fit, which reads the rules of its one split."
        ),
    )


def freeze_learned_rule(result: Any) -> FrozenLearnedRule:
    """Return the realized fold rules of a learned-rule fit, one value per row.

    Parameters
    ----------
    result : TMLEResult
        A learned-rule fit of one split.

    Returns
    -------
    FrozenLearnedRule
        The rule of the fit under its own name, loss and learner. ``treat`` is ``True``
        where the row's fold rule assigns the higher arm code.

    Raises
    ------
    CapabilityError
        If the fit combines more than one split (:func:`learned_rule_splits_refusal`).
    """
    refusal = learned_rule_splits_refusal("freeze_learned_rule", result.n_repeats)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    values = np.asarray(result.nuisance.regimes.values[:, 1, 0], dtype=float) > 0.5
    rule = result.estimator.learned_rule
    return FrozenLearnedRule(rule.name, values, loss=rule.loss, learner=rule.learner)


def frozen_rule_replay(estimator: Any, rule: FrozenLearnedRule, rows: Any = None) -> Any:
    """Return a copy of a learned-rule estimator that evaluates the frozen rule.

    Parameters
    ----------
    estimator : TMLE
        The estimator of the learned-rule fit.
    rule : FrozenLearnedRule
        The realized rules of the original rows.
    rows : numpy.ndarray or None
        The original row of each refit row, for a subset or a resample. ``None`` keeps
        every original row in place.

    Returns
    -------
    TMLE
        The replay estimator.
    """
    replay = copy.copy(estimator)
    replay.learned_rule = rule if rows is None else rule.rows(rows)
    return replay


def learned_rule_treat(
    data: CausalData,
    nuisance: NuisanceEstimates,
    rule: LearnedRule,
    *,
    fit_inner: InnerFit,
    g_bounds: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return the rule of every row of a learned-rule fit, its margin, and every fold's rule.

    A frozen rule returns its values.  The plug-in rule reads the out-of-fold blip of the
    cross-fitted outcome regression.  A pseudo-outcome loss fits its rule learner inside
    each outer training fold (:func:`~cleverly.interventions._rule_learners.rule_learner_treat`),
    and never reads the outer out-of-fold predictions.

    Each fold's rule is also evaluated at every row while its fitted model is in memory: the
    plug-in rule by the outcome regression of the fold
    (:attr:`~cleverly.estimators._nuisance.NuisanceEstimates.fold_plug_in_rules`), and a
    rule learner by its fitted learner.  The omitted-variable bound reads that matrix, so it
    never refits a rule.

    Parameters
    ----------
    data : CausalData
        The prepared data of the fit.
    nuisance : NuisanceEstimates
        The split's cross-fitted nuisances, whose folds and scaler the rule reads.
    rule : LearnedRule
        The declared rule.
    fit_inner : callable
        The inner nuisance fit of a pseudo-outcome loss,
        ``(data, folds, seed) -> NuisanceEstimates``.
    g_bounds : tuple of float
        The fit's truncation bounds of the treatment mechanism.
    seed : int
        The split's seed.

    Returns
    -------
    treat : numpy.ndarray
        One boolean per row: ``True`` where the rule of the row's fold assigns the higher
        arm code.
    margin : numpy.ndarray or None
        For a pseudo-outcome loss, the score of the row's fold rule minus its threshold,
        :math:`\\hat b_v(W_i)` or :math:`\\hat P_v(\\tilde D > 0 \\mid W_i) - 1/2`.
        ``None`` for the plug-in rule, whose score is the blip, and for a frozen rule.
    fold_rules : numpy.ndarray or None
        ``(n, V)`` booleans: column ``v`` is fold ``v``'s rule at every row.  ``None`` for
        a frozen rule, which holds one realized rule per row and no fold rule.

    Raises
    ------
    DataError
        If a frozen rule holds another number of rows than the refit, or if an inner
        training complement of a pseudo-outcome loss lacks an arm or an outcome class.
    RuntimeError
        If a plug-in rule meets nuisances fitted without every fold's rule.  No
        declaration reaches it.
    """
    if isinstance(rule, FrozenLearnedRule):
        n = np.asarray(nuisance.outcome.arms[nuisance.outcome.levels[0]]).shape[0]
        if rule.treat.shape != (n,):
            raise DataError(
                f"the frozen learned rule holds {rule.treat.size} rows, and the refit has {n}"
            )
        return rule.treat, None, None
    if rule.loss == "plug_in":
        if nuisance.fold_plug_in_rules is None:
            raise RuntimeError(
                "a plug-in learned-rule fit reached its rule without every fold's rule at "
                "every row. This is a defect in cleverly: TMLE asks the nuisance fit for them."
            )
        return _blip(nuisance) > 0.0, None, np.asarray(nuisance.fold_plug_in_rules, dtype=bool)
    return rule_learner_treat(
        data,
        nuisance.folds,
        loss=rule.loss,
        learner=rule.learner,
        fit_inner=fit_inner,
        scaler=nuisance.scaler,
        g_bounds=g_bounds,
        seed=seed,
    )


@dataclass(frozen=True)
class LearnedRuleSplit:
    """Record one split of a learned-rule fit: its rules and its fold estimates.

    Parameters
    ----------
    seed : int
        The seed of the split's folds and learners, as ``result.repeats`` records it.
    estimate : float
        The split's fold-evaluated value, :math:`(1/V)\\sum_v \\psi^*_{nv}`.
    fold_sizes : tuple of int
        The validation rows of each outer fold.
    fold_estimates : tuple of float
        The targeted plug-in value of each fold on its own validation rows.
    treated_shares : tuple of float
        The share of each fold's validation rows that its rule assigns to the higher
        arm code.
    blip_quantiles : tuple of tuple of float
        For each fold, quantiles of the outer outcome regression's blip
        ``Qbar_v(1, W) - Qbar_v(0, W)`` on its validation rows, at
        :data:`BLIP_QUANTILE_LEVELS`, on the scale the outcome regression is fitted on.
        For the plug-in rule this is the rule's own score, whose mass at 0 speaks to C3.
        Under a pseudo-outcome loss it describes the outer outcome regression, not the rule.
    score_quantiles : tuple of tuple of float
        For a pseudo-outcome loss, for each fold, quantiles of the rule learner's score
        minus its threshold on the fold's validation rows, at :data:`BLIP_QUANTILE_LEVELS`:
        :math:`\\hat b_v(w)` for ``loss="blip"`` and :math:`\\hat P_v(\\tilde D > 0 \\mid w) - 1/2`
        for ``loss="weighted_classification"``.  The rule treats where this is positive, so
        its mass at 0 speaks to C3.  Empty for the plug-in rule and for a frozen rule.
    inner_seeds : tuple of int
        For a pseudo-outcome loss, the seed of each fold's inner cross-fit and rule learner
        (:func:`~cleverly.interventions._rule_learners.inner_seed`), so that a refit of one
        fold's rule reproduces it.  Empty for the plug-in rule.

    Attributes
    ----------
    n_folds : int
    fold_weights : tuple of float
    """

    seed: int
    estimate: float
    fold_sizes: tuple[int, ...]
    fold_estimates: tuple[float, ...]
    treated_shares: tuple[float, ...]
    blip_quantiles: tuple[tuple[float, ...], ...]
    score_quantiles: tuple[tuple[float, ...], ...]
    inner_seeds: tuple[int, ...]

    @property
    def n_folds(self) -> int:
        """Return the number of outer folds of the split."""
        return len(self.fold_sizes)

    @property
    def fold_weights(self) -> tuple[float, ...]:
        """Return the weight of each fold in the split's average, ``1/V`` for every fold."""
        return tuple(1.0 / self.n_folds for _ in self.fold_sizes)


@dataclass(frozen=True)
class LearnedRuleRecord:
    """Record what a learned-rule fit estimated, under ``result.extra["learned_rule"]``.

    The fitted fold models are not kept.  The realized rule of each row in split ``r`` is
    ``result.repeats[r].nuisance.regimes.values[:, 1, 0]``, and the folds, the learner
    templates and the seeds replay the fit.  A fit of ``repeats=R`` holds one
    :class:`LearnedRuleSplit` per split, and reports the mean of their estimates.

    Parameters
    ----------
    name : str
        The label of the rule.
    splits : tuple of LearnedRuleSplit
        One record per split, in the order of ``result.repeats``.
    loss : str
        The loss of the rule learner, one of :data:`LOSSES`.
    learner : str or None
        The plain name of the rule learner of a pseudo-outcome loss, or ``None`` for the
        plug-in rule.

    Attributes
    ----------
    n_splits : int
    rule : str
    target : str
    quantile_levels : tuple of float
    """

    name: str
    splits: tuple[LearnedRuleSplit, ...]
    loss: str
    learner: str | None

    @property
    def n_splits(self) -> int:
        """Return the number of splits the reported mean combines."""
        return len(self.splits)

    @property
    def rule(self) -> str:
        """Return the rule class of the fit's loss, in words (:data:`RULE_CLASS`)."""
        return RULE_CLASS[self.loss]

    @property
    def target(self) -> str:
        """Return what the reported value is (:data:`TARGET_KIND`)."""
        return TARGET_KIND

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the levels of each split's quantiles (:data:`BLIP_QUANTILE_LEVELS`)."""
        return BLIP_QUANTILE_LEVELS

    def describe(self) -> str:
        """Return the line that :meth:`~cleverly.TMLEResult.summary` prints for this fit.

        Returns
        -------
        str
            The target, the fold count, the source of the definition, how the rules are
            learned, which record field describes a rule learner's score, and the range of
            the treated share over the folds.  A fit of several
            splits also names the split count, and gives both ranges over every split.
        """
        shares = [share for split in self.splits for share in split.treated_shares]
        low, high = min(shares), max(shares)
        learned = {
            "plug_in": "Each rule is the plug-in rule of the fold's outcome regression.",
            "blip": (
                f"Each rule is learned by {self.learner}, a regression of a doubly robust "
                "blip pseudo-outcome on the covariates (loss='blip')."
            ),
            "weighted_classification": (
                f"Each rule is learned by {self.learner}, a classifier of the sign of a doubly "
                "robust blip pseudo-outcome weighted by its size "
                "(loss='weighted_classification')."
            ),
        }[self.loss]
        if self.loss != "plug_in":
            learned += (
                " The record's score_quantiles give the rule learner's score minus its "
                "threshold on each fold's validation rows, and its blip_quantiles describe "
                "the outer outcome regression, not the rule."
            )
        source = (
            "The target depends on the data, because the rules do (van der Laan and Luedtke "
            f"2015, Section 7). {learned}"
        )
        if self.n_splits == 1:
            return (
                f"Target: the average value of the {self.splits[0].n_folds} rules learned on the "
                "training folds, each estimated on the fold it did not train on. "
                f"{source} Unweighted share of rows the rules treat, by fold: "
                f"{low:.3g} to {high:.3g}."
            )
        estimates = [split.estimate for split in self.splits]
        return (
            f"Target: the mean over {self.n_splits} random splits of the average value of "
            "the rules learned on the training folds of each split, each estimated on the "
            f"fold it did not train on. {source} Estimate of each split: "
            f"{min(estimates):.4g} to {max(estimates):.4g}. Unweighted share of rows the "
            f"rules treat, by fold over every split: {low:.3g} to {high:.3g}."
        )


def mean_over_splits(
    per_repeat: Sequence[Mapping[str, ParameterEstimate]],
    *,
    variance: Callable[[np.ndarray], float],
) -> dict[str, ParameterEstimate]:
    r"""Combine the split reports of a learned-rule fit by their mean.

    Each split :math:`r` estimates its own target :math:`\tilde\psi^{(r)}_{0n}` with
    :math:`\hat\psi_r - \tilde\psi^{(r)}_{0n} = n^{-1}\sum_i \phi_{r,i} + o_P(n^{-1/2})`,
    where :math:`\phi_{r,i}` is the split's stored curve (van der Laan and Luedtke 2015,
    Theorem 6, per split).  The reported target is the mean of the split targets, and by
    linearity over a fixed :math:`R`,

    .. math::

        \hat\psi = \frac1R\sum_r \hat\psi_r, \qquad
        \bar\phi_i = \frac1R\sum_r \phi_{r,i}.

    The variance is ``variance(phibar)``, the split's own rule applied to the averaged
    curve: :math:`n^{-2}\sum_i \bar\phi_i^2` for the fold-evaluated report.  The median of
    the split estimates estimates no defined parameter, so it is never used here.

    A parameter status of one split passes to the mean, as it passes to the median
    (:func:`~cleverly.inference.influence.restamp_parameter_statuses`).  Every other
    field is fixed before the first learner: the learned-rule target declares no case in
    which a fold cannot evaluate it, so every split reports it, and the inference status,
    the support record and the level scale follow from the declaration and the data,
    which every split shares.  The guard below therefore finds a defect in the package,
    never a property of the data.

    Parameters
    ----------
    per_repeat : sequence of mappings
        One estimate mapping per split, each stamped by its own fit.
    variance : callable
        The variance rule of one split, applied to the averaged curve.

    Returns
    -------
    dict
        The mean estimate of each name, with the averaged curve and its variance. Every
        other field is the first split's.

    Raises
    ------
    RuntimeError
        When the splits report different names, fit statuses or support records, or an
        estimate on a ratio or transformed scale.  No declaration or data reaches it.
    """
    from ..inference.influence import restamp_parameter_statuses

    per_repeat = restamp_parameter_statuses(per_repeat)
    names = list(per_repeat[0])
    defect = (
        "This is a defect in cleverly, not in the data: the learned-rule value is a level "
        "that every split reports under one configuration."
    )
    if any(list(report) != names for report in per_repeat):
        raise RuntimeError(
            f"the splits of a learned-rule fit report different estimates "
            f"{[sorted(report) for report in per_repeat]}. {defect}"
        )
    out: dict[str, ParameterEstimate] = {}
    for name in names:
        parts = [report[name] for report in per_repeat]
        first = parts[0]
        if (
            len({part.inference for part in parts}) != 1
            or len({part.support for part in parts}) != 1
            or first.transform is not None
            or first.scale == "ratio"
        ):
            raise RuntimeError(
                f"{name} has different inference statuses or support records across the "
                f"splits, or is not on a level scale, so the mean over splits does not "
                f"apply. {defect}"
            )
        curve = np.mean([np.asarray(part.influence_curve, dtype=float) for part in parts], axis=0)
        out[name] = replace(
            first,
            psi=float(np.mean([part.psi for part in parts])),
            influence_curve=curve,
            variance=float(variance(curve)),
        )
    return out


def _blip(nuisance: NuisanceEstimates) -> np.ndarray:
    """The out-of-fold blip ``Qbar_v(1, W_i) - Qbar_v(0, W_i)`` at every row.

    Row ``i`` of fold ``v`` carries the prediction of the outcome regression fitted on the
    training complement of ``v``, so the blip at a row never reads that row.  The refusal
    table refuses a treatment with more than two arms first.
    """
    control, treated = nuisance.outcome.levels
    return np.asarray(nuisance.outcome.arms[treated], dtype=float) - np.asarray(
        nuisance.outcome.arms[control], dtype=float
    )


@dataclass(frozen=True)
class _LearnedRuleRegimes(RegimeSet):
    """The regime set of a learned-rule fit, with every fold's rule and the rule learner's margin.

    Parameters
    ----------
    names : tuple of str
        The one regime label.
    values : numpy.ndarray
        ``(n, 2, 1)`` one-hot densities of each row's own fold rule.
    reference : float
        The code of the one regime.
    margin : numpy.ndarray or None
        The score of each row's fold rule minus its threshold, for a pseudo-outcome loss
        (:func:`learned_rule_treat`), or ``None``.  :func:`learned_rule_split` records its
        quantiles.
    fold_rules : numpy.ndarray or None
        ``(n, V)`` booleans, ``True`` where fold ``v``'s rule assigns row ``i`` the higher
        arm code, or ``None`` for a frozen rule.  Row ``i``'s entry at its own fold is the
        realized rule ``values[i, 1, 0]``.  The omitted-variable bound reads every column
        (:func:`fold_rules`).
    """

    margin: np.ndarray | None = None
    fold_rules: np.ndarray | None = None


def fold_rules(nuisance: NuisanceEstimates) -> np.ndarray | None:
    """Return the rule of every fold at every row of one split of a learned-rule fit.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The nuisances of one split, whose regime set the fit built.

    Returns
    -------
    numpy.ndarray or None
        ``(n, V)`` booleans in the order of ``nuisance.folds``, or ``None`` when the split
        stores none: a frozen rule's replay, which holds one realized rule per row.

    Raises
    ------
    RuntimeError
        If the entry of some row at its own fold differs from the row's realized rule.  The
        fit evaluates both from one fitted model, so this is a defect in cleverly.
    """
    regimes = nuisance.regimes
    rules = getattr(regimes, "fold_rules", None)
    if rules is None:
        return None
    assert regimes is not None
    realized = np.asarray(regimes.values[:, 1, 0], dtype=float) > 0.5
    return check_fold_rules(rules, realized, nuisance.folds)


def check_fold_rules(rules: np.ndarray, realized: np.ndarray, folds: Folds) -> np.ndarray:
    """Check every fold's rule at every row against each row's realized rule.

    The fit runs it when it builds the regime set, and :func:`fold_rules` runs it again on
    every read, because a saved result can be edited.

    Parameters
    ----------
    rules : numpy.ndarray
        ``(n, V)`` booleans, column ``v`` the rule of fold ``v`` at every row.
    realized : numpy.ndarray
        ``(n,)`` booleans, the rule of each row's own fold.
    folds : Folds
        The split whose folds the columns follow.

    Returns
    -------
    numpy.ndarray
        ``rules`` as booleans.

    Raises
    ------
    RuntimeError
        If the shape does not match the split, or if the entry of some row at its own fold
        differs from its realized rule.  The fit evaluates both from one fitted model, so
        this is a defect in cleverly.
    """
    rules = np.asarray(rules, dtype=bool)
    assignment = np.asarray(folds.assignment, dtype=int)
    if rules.shape != (assignment.size, folds.n_folds):
        raise RuntimeError(
            f"the stored fold rules have shape {rules.shape}, and the split has "
            f"{assignment.size} rows and {folds.n_folds} folds. This is a defect in "
            "cleverly."
        )
    realized = np.asarray(realized, dtype=bool)
    own = rules[np.arange(assignment.size), assignment]
    if not np.array_equal(own, realized):
        raise RuntimeError(
            f"the stored rule of {int(np.sum(own != realized))} rows at their own fold differs "
            "from the realized rule. The fit evaluates both from one fitted model, so this is "
            "a defect in cleverly; report it."
        )
    return rules


def _learned_rule_regimes(
    treat: np.ndarray,
    name: str,
    margin: np.ndarray | None = None,
    fold_rules: np.ndarray | None = None,
) -> RegimeSet:
    """The fold-local rule of every row, as a one-regime :class:`RegimeSet`.

    The density is one-hot: ``values[i, 1, 0]`` is ``1`` when the rule of row ``i``'s fold
    assigns it the higher arm code, and ``values[i, 0, 0]`` is ``1`` otherwise.  The set
    is built directly, so no regime declaration is checked: the rule is learned, and the
    target that reads it is the learned-rule value, not a regime mean.

    Parameters
    ----------
    treat : numpy.ndarray
        The rule of every row (:func:`learned_rule_treat`).
    name : str
        The name of the regime, the rule's label.
    margin : numpy.ndarray or None
        The rule learner's score minus its threshold at every row, or ``None``.
    fold_rules : numpy.ndarray or None
        Every fold's rule at every row, ``(n, V)`` booleans, or ``None`` for a frozen rule.

    Returns
    -------
    RegimeSet
        One regime, named ``name``, over the two arms.
    """
    treat = np.asarray(treat, dtype=bool)
    values = np.zeros((treat.size, 2, 1), dtype=float)
    values[:, 1, 0] = treat
    values[:, 0, 0] = ~treat
    if margin is None and fold_rules is None:
        return RegimeSet((name,), values, 0.0)
    return _LearnedRuleRegimes(
        (name,),
        values,
        0.0,
        None if margin is None else np.asarray(margin, dtype=float),
        None if fold_rules is None else np.asarray(fold_rules, dtype=bool),
    )


def learned_rule_split(
    rule: LearnedRule,
    nuisance: NuisanceEstimates,
    fold_estimates: Sequence[float],
    *,
    seed: int,
    estimate: float,
) -> LearnedRuleSplit:
    """Build the record of one split of a learned-rule fit.

    Parameters
    ----------
    rule : LearnedRule
        The declared rule.
    nuisance : NuisanceEstimates
        The split's cross-fitted nuisances, whose folds and regimes the record reads.
    fold_estimates : sequence of float
        The fold plug-in values, in the order of ``nuisance.folds``.
    seed : int
        The split's seed.
    estimate : float
        The split's fold-evaluated value.

    Returns
    -------
    LearnedRuleSplit
        The per-fold summaries of the split.
    """
    blip = _blip(nuisance)
    assert nuisance.regimes is not None
    treat = np.asarray(nuisance.regimes.values[:, 1, 0], dtype=float) > 0.5
    tests = [np.asarray(test) for _, test in nuisance.folds]
    learns = rule.loss != "plug_in" and not isinstance(rule, FrozenLearnedRule)
    margin = getattr(nuisance.regimes, "margin", None) if learns else None
    return LearnedRuleSplit(
        seed=int(seed),
        estimate=float(estimate),
        fold_sizes=tuple(int(test.size) for test in tests),
        fold_estimates=tuple(float(value) for value in fold_estimates),
        treated_shares=tuple(float(np.mean(treat[test])) for test in tests),
        blip_quantiles=tuple(
            tuple(float(value) for value in np.quantile(blip[test], BLIP_QUANTILE_LEVELS))
            for test in tests
        ),
        score_quantiles=(
            ()
            if margin is None
            else tuple(
                tuple(float(value) for value in np.quantile(margin[test], BLIP_QUANTILE_LEVELS))
                for test in tests
            )
        ),
        inner_seeds=(tuple(inner_seed(seed, fold) for fold in range(len(tests))) if learns else ()),
    )


def learned_rule_record(rule: LearnedRule, splits: Sequence[LearnedRuleSplit]) -> LearnedRuleRecord:
    """Build the record of a learned-rule fit from the records of its splits.

    Parameters
    ----------
    rule : LearnedRule
        The declared rule.
    splits : sequence of LearnedRuleSplit
        One record per split, in the order of ``result.repeats``.

    Returns
    -------
    LearnedRuleRecord
        The record, with the rule's loss and the plain name of its learner.
    """
    from .._summary._learners import learner_name

    return LearnedRuleRecord(
        name=rule.name,
        splits=tuple(splits),
        loss=rule.loss,
        learner=None if rule.learner is None else learner_name(rule.learner),
    )


def _arm_estimands(estimands: Any) -> tuple[str, ...]:
    """The requested estimand names that belong to another parameter axis."""
    from ..targets import TARGETS

    if estimands is None or estimands == "all":
        return ()
    names = (estimands,) if isinstance(estimands, str) else tuple(estimands)
    return tuple(
        name for name in names if name in TARGETS and TARGETS[name].parameter_axis != "learned_rule"
    )


def learned_rule_configuration_refusal(estimator: Any) -> Refusal | None:
    """Return the refusal of a learned-rule estimator that no data can change, or ``None``.

    The first two rows of the learned-rule refusal table in
    ``docs/technical-reference/point-treatment-tmle.md``: a collaborative or doubly robust
    estimator, and ``learned_rule=`` beside another axis keyword, ``reference=`` or an arm
    estimand.  The engine asks it at construction and again first in its preflight.

    Parameters
    ----------
    estimator : TMLE
        The estimator, with ``learned_rule`` set.

    Returns
    -------
    Refusal or None
        The refusal, an ``IMPLEMENTATION_GAP`` record, or ``None``.
    """
    method = getattr(estimator, "_assessment_method", "tmle")
    if method != "tmle":
        label = {"collaborative_tmle": "CTMLE", "drtmle": "DRTMLE"}[method]
        return _gap(
            f"{label} does not estimate the learned-rule value. No collaborative or doubly "
            f"robust learned-rule result was reviewed ({_F27}).",
            f"Fit the ordinary TMLE with learned_rule= and {LEARNED_RULE_CONFIGURATION}.",
            "F27",
        )
    contrasts = [
        keyword
        for keyword, value in (
            ("interventions=", getattr(estimator, "interventions", ())),
            ("reference=", getattr(estimator, "reference", None) is not None),
        )
        if value
    ]
    axes = [
        keyword
        for keyword, value in (
            ("policies=", getattr(estimator, "policies", ())),
            ("incremental=", getattr(estimator, "incremental", ())),
            ("msm=", getattr(estimator, "msm", None) is not None),
        )
        if value
    ]
    arm = _arm_estimands(getattr(estimator, "estimands", None))
    if not (contrasts or axes or arm):
        return None
    reasons = []
    roadmap = "F17"
    if contrasts or arm:
        named = [*contrasts, *([f"estimands={list(arm)}"] if arm else [])]
        reasons.append(
            f"A contrast with a known regime or an arm ({', '.join(named)}) follows by "
            f"linearity of the fold-local curves, and {_X11.format(part='c')} holds it"
        )
        roadmap = "X11"
    if axes:
        reasons.append(
            f"Another parameter axis ({', '.join(axes)}) needs a fit of its own, because one "
            "fluctuation cannot solve the score equations of two axes, and "
            f"{_X11.format(part='c')} holds the joint fit"
        )
    # The remedy names what the caller declared: the keywords to drop, estimands=None for
    # an arm estimand, and a fit of its own for another target.  reference= names no target.
    steps = [
        *([f"without {' or '.join([*contrasts, *axes])}"] if contrasts or axes else []),
        *(["with estimands=None"] if arm else []),
    ]
    other = [keyword for keyword in [*contrasts, *axes] if keyword != "reference="] or arm
    return _gap(
        "learned_rule= estimates one data-adaptive value, the fold average of the values of "
        f"rules learned inside each training fold. {'. '.join(reasons)}.",
        f"Fit the learned-rule value alone, {' and '.join(steps)}"
        + (", and fit the other target in a fit of its own" if other else "")
        + ".",
        roadmap,
    )


def _data_refusal(data: CausalData) -> Refusal | None:
    """The data rows of the learned-rule refusal table: the declarations no setting repairs.

    ``docs/technical-reference/point-treatment-tmle.md``, "Learned rules", holds the table.
    """
    if data.is_continuous_treatment:
        return _gap(
            "a learned rule over a continuous treatment is refused. No reviewed source "
            f"defines a learned rule over a dose for the learned-rule value ({_F27}). The "
            "learned-rule value needs a binary treatment.",
            "A fit without learned_rule= can use "
            "TMLE(policies=[ModifiedPolicy(..., policy_kind='known')]) for a fixed policy on a "
            "continuous treatment.",
            "F27",
        )
    if not data.is_binary_treatment:
        return _gap(
            f"a learned rule over a treatment with {data.n_arms} arms is refused. Van der "
            "Laan and Luedtke (2015) derive the learned-rule value for a binary treatment, "
            f"and {_X11.format(part='d')} holds categorical treatments.",
            f"{_FIXED_RULE} estimates a fixed rule on a categorical treatment.",
            "X11",
        )
    if data.has_missing_outcome:
        return _gap(
            "a learned rule with missing outcomes (delta=) is refused. Van der Laan and "
            "Luedtke (2015) treat complete outcomes, and no audited result covers a "
            f"cross-fitted learned-rule fit with missing outcomes ({_F27}). "
            "The learned-rule value needs every outcome observed.",
            f"{_FIXED_RULE} takes delta= for a fixed rule.",
            "F27",
        )
    if data.has_intermediate:
        return _gap(
            "a learned rule with intermediate= is refused. A learned rule at the treatment and "
            "the static Z = z at the intermediate form a two-time-point rule pair, which is "
            f"not implemented ({_X11.format(part='j')}).",
            f"{_FIXED_RULE} takes intermediate= for a fixed rule.",
            "X11",
        )
    if data.declares_estimated_weights:
        return _gap(
            "a learned rule with estimated observation weights (weights_estimated=True) is "
            "refused. The learned-rule value covers known, fixed weights, and the variance "
            "of a fit that also estimates its weights is not implemented "
            f"({_X11.format(part='k2')}).",
            "Declare the weights known (weights_estimated=False) when they are fixed by the "
            f"design, or use {_FIXED_RULE}, which takes estimated weights for a fixed rule.",
            "X11",
        )
    if data.has_strata:
        return _gap(
            "a learned rule with baseline strata (strata=) is refused. The value inside each "
            "of a fixed number of strata needs fold evaluation with strata, which X28 holds "
            f"({_X11.format(part='m')}).",
            f"{_FIXED_RULE} takes strata= for a fixed rule.",
            "X11",
        )
    return None


def learned_rule_scheme_refusal(estimator: Any, data: CausalData | None = None) -> Refusal | None:
    """Return the refusal of a learned-rule fit whose remedy is a setting, or ``None``.

    The setting rows of the learned-rule refusal table in
    ``docs/technical-reference/point-treatment-tmle.md``.  Each fold-policy row names the
    configuration that fits, :data:`LEARNED_RULE_CONFIGURATION`.  The row of
    ``repeats`` above 1 with clusters names ``repeats=1``, and it needs the data, so a
    caller without them skips it.

    ``repeats=R`` above 1 on rows is no refusal: the fit reports the mean over the ``R``
    splits (:func:`mean_over_splits`).  The median over splits estimates no defined
    parameter, so no setting reaches it.

    Parameters
    ----------
    estimator : TMLE
        The estimator, with ``learned_rule`` set.
    data : CausalData or None
        The prepared data, or ``None`` before the data exist.

    Returns
    -------
    Refusal or None
        The refusal, or ``None``. The bootstrap row is an ``INFERENCE_GAP`` record, and
        the others are ``IMPLEMENTATION_GAP`` records.
    """
    remedy = f"To estimate the learned-rule value, fit it with {LEARNED_RULE_CONFIGURATION}."
    if not estimator.cross_fit or estimator.n_folds < 2:
        return _gap(
            "the learned-rule value needs cross-fitting with at least two outer folds. A "
            "rule learned on every row is evaluated on the rows it was learned on, and the "
            "value of that full-sample rule is a different target "
            f"({_X11.format(part='a')}).",
            remedy,
            "X11",
        )
    if not estimator.cv_evaluation:
        return _gap(
            "the learned-rule value needs the fold-evaluated CV-TMLE (cv_evaluation=True). "
            "The stacked report weights the folds by n_v / n and centres the curve at the "
            "pooled estimate, which is a different construction "
            f"({_X11.format(part='e')}).",
            remedy,
            "X11",
        )
    if estimator.repeats != 1 and data is not None and data.cluster is not None:
        return _gap(
            f"the mean over {estimator.repeats} splits (repeats={estimator.repeats}) of a "
            "learned-rule value with clusters (id=) is refused. The averaged curve is centred "
            "on different cluster partitions, and it loses about 2V - V/R - 1 cluster degrees "
            "of freedom, for which no finite-sample correction is derived "
            f"({_X11.format(part='i2')}).",
            "Fit the clustered learned-rule value with repeats=1 (CrossFitting(repeats=1) on "
            "the method).",
            "X11",
        )
    if estimator.n_bootstrap:
        return Refusal(
            RefusalReason.INFERENCE_GAP,
            "the learned-rule value refuses the full-refit bootstrap (n_bootstrap=0). Each "
            "resample relearns the rules, so each draw has a different target "
            f"({_F27}).",
            alternative=f"{remedy} That fit reports the influence-curve interval.",
            roadmap="F27",
        )
    return None


def refuse_learned_rule_composition(data: CausalData, estimator: Any = None) -> None:
    """Refuse a learned-rule request that the learned-rule contract does not cover.

    The checks follow the refusal table of the learned-rule contract in
    ``docs/technical-reference/point-treatment-tmle.md``.  A refusal that no setting repairs comes
    before a refusal whose remedy is a setting.  So a caller who follows one remedy never meets an
    earlier refusal next.

    ======  ===================================================  ===================
    order   request                                              item
    ======  ===================================================  ===================
    1       ``CTMLE`` or ``DRTMLE``                              F27
    2       another axis keyword, ``reference=``, arm estimands  X11 (c)
    3       a continuous treatment                               F27
    4       more than two arms                                   X11 (d)
    5       missing outcomes, ``delta=``                         F27
    6       ``intermediate=``                                    X11 (j)
    7       estimated weights, ``weights_estimated=True``        X11 (k2)
    8       ``strata=``                                          X11 (m)
    9       ``cross_fit=False``, or one fold                     X11 (a)
    10      ``cv_evaluation=False``                              X11 (e)
    11      ``repeats`` above 1 with clusters, ``id=``           X11 (i2)
    12      ``n_bootstrap`` above 0                              F27
    13      ``loss="blip"`` with ``weights=`` and a learner      none, a declaration
            whose ``fit`` takes no ``sample_weight``
    ======  ===================================================  ===================

    ``TMLE`` runs it first in its preflight, before any learner.
    ``CausalStudy.identify`` runs it with no estimator, so the data rows refuse there.

    Parameters
    ----------
    data : CausalData
        The prepared data of the fit.
    estimator : TMLE or None
        The estimator, or ``None`` for the checks that read the data alone.

    Raises
    ------
    CapabilityError
        If the request meets a row of the table.
    """
    refusal = None if estimator is None else learned_rule_configuration_refusal(estimator)
    if refusal is None:
        refusal = _data_refusal(data)
    if refusal is None and estimator is not None:
        refusal = learned_rule_scheme_refusal(estimator, data)
    if refusal is None and estimator is not None:
        refusal = _weighted_learner_refusal(estimator.learned_rule, data)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)


def _weighted_learner_refusal(rule: LearnedRule, data: CausalData) -> Refusal | None:
    """The last row of the table: a weighted blip fit whose learner takes no weights.

    Under ``weights=`` the rule is the rule of the weighted population, so the blip
    regression is weighted.  A learner whose ``fit`` takes no ``sample_weight`` would be fitted
    unweighted, and its rule would be the rule of the unweighted population.  Weighted
    classification refuses such a learner whatever the data, when the rule is declared.
    """
    if rule.loss != "blip" or not data.is_weighted:
        return None
    unweighted = _unweighted_member(rule.learner)
    if unweighted is None:
        return None
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        "LearnedRule(loss='blip') with weights= fits a weighted blip regression, so that "
        f"the rule is the rule of the weighted population, and {unweighted} takes no "
        "sample_weight. Fitted without the weights, it would learn the rule of the "
        "unweighted population.",
        alternative=(
            "Pass a regressor whose fit takes sample_weight, such as LinearRegression(), and "
            "for a SuperLearner a library whose every member's fit takes it."
        ),
    )
