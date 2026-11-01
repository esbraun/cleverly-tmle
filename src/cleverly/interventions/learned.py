r"""The value of a treatment rule learned inside each training fold.

A :class:`~cleverly.interventions.Rule` is a *known* function of the covariates, and a rule
learned from the analysis sample is refused there.  This module holds
the one learned-rule target the package estimates instead.  The target is data-adaptive.
The fold-evaluated report (``TMLE(cv_evaluation=True)``) targets

.. math::

    \tilde\psi_{0n} = \frac1V \sum_{v=1}^V \Psi_{d_{nv}}(P_0),

where :math:`d_{nv}` is the rule learned on the training rows of outer fold :math:`v`, and the
stacked report (``cv_evaluation=False``) targets
:math:`\tilde\psi^{s}_{0n} = \sum_v (W_v / W)\,\Psi_{d_{nv}}(P_0)`, with :math:`W_v` the weight
mass of validation fold :math:`v` (its row count without weights) and :math:`W` their sum.  The
two coincide at equal masses.  The
default rule is the plug-in rule
:math:`d_{nv}(w) = \arg\max_a \bar Q_{nv}(a, w)` of the outcome regression
:math:`\bar Q_{nv}` fitted on those rows, over every arm of the treatment, with a tie
assigned to the lowest arm code (polle, Nordland and Holst 2026).  At two arms it is
:math:`\mathbb 1\{\bar Q_{nv}(1, w) - \bar Q_{nv}(0, w) > 0\}`, and at more than two arms
(X11 (d)) it is the argmax over all of them.  ``LearnedRule(loss="blip")`` and
``LearnedRule(loss="weighted_classification")`` learn it with a declared learner from a doubly
robust pseudo-outcome instead (Luedtke and van der Laan 2016, *International Journal of
Biostatistics*, Sections 4.1 to 4.3), which :mod:`cleverly.interventions._rule_learners` fits.
Both losses are binary, so they refuse a treatment with more than two arms (X11 (d2)).  Van der
Laan and Luedtke (2015), Section 7 and Appendix B, define the target and its cross-validated
TMLE for any rule learned on the training rows, and polle's Algorithm 4 (Nordland and Holst
2026) gives the stacked weighting :math:`n_v / n`.  The estimator reuses the
regime fluctuation with one pooled :math:`\varepsilon`, and either the fold-evaluated report
with the cross-validated variance or the stacked report with the curve centred at the pooled
estimate.  This module adds the fold-local rule, the result record, the mean over a fixed
number of splits (:func:`mean_over_splits`) and the refusals.  No new submodel, fluctuation or
influence curve enters.  Known regimes declared beside the rule (X11 (c)) are more columns of
the same regime set, and a contrast's curve is the difference of the two columns' curves.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import CapabilityError, DataError, Refusal, RefusalReason
from ..learners.crossfit import LEARNED_RULE_IN_SAMPLE
from ._rule_learners import InnerFit, inner_seed, rule_learner_treat
from .base import RegimeSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators._nuisance import NuisanceEstimates
    from ..inference.influence import ParameterEstimate
    from ..learners.crossfit import Folds

__all__ = [
    "BLIP_QUANTILE_LEVELS",
    "CONTRAST_TARGETS",
    "LEARNED_RULE_ARM_REMEDY",
    "LEARNED_RULE_BENCHMARK_REFUSAL",
    "LEARNED_RULE_CONFIGURATION",
    "LEARNED_RULE_EVALUE_REFUSAL",
    "LEARNED_RULE_IN_SAMPLE_BENCHMARK_REFUSAL",
    "LEARNED_RULE_LIMITS",
    "LEARNED_RULE_REMEDY",
    "LEARNED_RULE_TILT_REFUSAL",
    "LOSSES",
    "RULE_CLASS",
    "SCHEMES",
    "TARGET_KIND",
    "FrozenLearnedRule",
    "LearnedRule",
    "LearnedRuleRecord",
    "LearnedRuleSplit",
    "check_fold_rules",
    "fold_rules",
    "freeze_learned_rule",
    "frozen_rule_replay",
    "in_sample_learner_condition",
    "learned_rule_configuration_refusal",
    "learned_rule_estimands",
    "learned_rule_record",
    "learned_rule_retarget_refusal",
    "learned_rule_scheme_refusal",
    "learned_rule_split",
    "learned_rule_splits_refusal",
    "learned_rule_treat",
    "mean_over_splits",
    "realized_arms",
    "refuse_learned_rule_composition",
    "stacked_fold_estimates",
]

#: The loss of each rule learner, the ``loss=`` of :class:`LearnedRule`.
LOSSES: tuple[str, ...] = ("plug_in", "blip", "weighted_classification")

#: The rule class of each loss, in words.  The record quotes the one of its fit.  For the
#: pseudo-outcome losses, which read two arms only, ``1`` and ``0`` stand for the higher and
#: the lower of the two sorted arm codes.
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
        "the plug-in rule argmax_a Qbar_v(a, W) of the outcome regression fitted on the "
        "training rows of each outer fold, over every arm of the treatment, where a tie "
        "between two or more arms assigns the lowest of their arm codes. At two arms it is "
        f"1{{Qbar_v(1, W) - Qbar_v(0, W) > 0}}, {_TIE}. Under weights= the regression is the "
        "weighted one, so the rule is the plug-in rule of the weighted population"
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

#: What the reported value is under each report, keyed by :attr:`LearnedRuleSplit.scheme`.
#: The record and the summary quote it.
TARGET_KIND: dict[str, str] = {
    "fold_evaluated": "fold-average data-adaptive value",
    "stacked": "fold-mass-weighted data-adaptive value",
    "in_sample": "full-sample data-adaptive value",
}

#: The three schemes of a learned-rule fit, the values of :attr:`LearnedRuleSplit.scheme`:
#: the two reports of a cross-fitted fit, and the in-sample fit of one rule learned on
#: every row (``cross_fit=False``), whose target is the value of that full-sample rule.
SCHEMES: tuple[str, ...] = tuple(TARGET_KIND)

#: The quantile levels of the top-two gap and of a rule learner's score that the record
#: keeps for each fold.
BLIP_QUANTILE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

#: The configuration that fits the learned-rule value.  Every refusal whose remedy is a
#: configuration names it, and ``tests/unit/test_learned_rule_refusals.py`` fits it.  Either
#: report fits, so the requirement names neither ``cv_evaluation`` value.  Its last clause
#: names the fold-evaluated report, because the default is the stacked report, which no
#: registered study covers.
LEARNED_RULE_CONFIGURATION = (
    "cross_fit=True on the engine, or CrossFitting(enabled=True) on the method, with n_folds "
    "of at least 2 and n_bootstrap=0 (Inference(n_bootstrap=0) on the method); "
    "cv_evaluation=True on the engine (CrossFitting(fold_evaluation=True) on the method) "
    "gives the fold-evaluated report that the registered studies cover"
)

#: The way out of a shared refusal that a drawn split raises.  The shared remedy is a bare
#: in-sample fit, and on a learned-rule fit that changes the target to the value of the
#: full-sample rule (X11 (a)), so this remedy names the change.
LEARNED_RULE_REMEDY = LEARNED_RULE_IN_SAMPLE

#: The way out of the arm minimum on a learned-rule fit.  The in-sample fit changes the
#: target, so the remedy that keeps the cross-fitted target comes first: more units at the
#: rare arm.  A one-unit arm gives no sound fit under either scheme.
LEARNED_RULE_ARM_REMEDY = f"collect more observations at the rare arm, or {LEARNED_RULE_IN_SAMPLE}"

#: The roadmap part that holds the limits and the benchmark of the learned-rule bound.
_X11_N2 = "X11 (n2) in docs/roadmap.md"

#: The reason every sensitivity analysis of a learned-rule fit shares.  Each analysis
#: appends the roadmap row that holds its own missing work.
_LEARNED_RULE_TARGET = (
    "this sensitivity analysis of the learned-rule value is not implemented. The target is the "
    "value of the rules the fit learned, and a refit relearns them, so a refitted or perturbed "
    "fit estimates a different target"
)

#: Why ``benchmark`` refuses a cross-fitted learned-rule fit.  The omitted-variable bound
#: itself reads the fit's own fold rules, held fixed, and needs no refit.
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

#: Why ``benchmark`` refuses an in-sample learned-rule fit (``cross_fit=False``).  Its one
#: rule is every row's rule, so the frozen-rule replay holds it, and a short model that keeps
#: it fixed is well posed.  That benchmark is not built.
LEARNED_RULE_IN_SAMPLE_BENCHMARK_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "benchmark refits the model without the named covariates, and on a learned-rule fit the "
    "refit relearns the rule, so the short model would estimate the value of another rule. "
    "This in-sample fit (cross_fit=False) has one rule, the rule of every row, so a short "
    "model that holds it fixed, as the frozen-rule replay does, is well posed. That "
    f"benchmark is not built ({_X11_N2}).",
    alternative=(
        "Read omitted_variable_bounds() and robustness_value() at declared strengths, which "
        "hold the fit's rule fixed and refit nothing."
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

    The rule is learned inside each outer training fold.  ``loss=`` picks how the rule is
    learned:

    - ``"plug_in"``, the default: the plug-in rule of the fit's own outcome regression,
      ``d_v(w) = argmax_a Qbar_v(a, w)`` over every arm, where ``Qbar_v`` is fitted on the
      training rows of fold ``v`` and a tie between arms assigns the lowest arm code.  At
      two arms it is ``1{Qbar_v(1, w) - Qbar_v(0, w) > 0}``.  It needs no learner beyond
      the outcome regression and no fit beyond the cross-fitted nuisances.
    - ``"blip"``: ``learner`` regresses a doubly robust blip pseudo-outcome on the
      covariates, and the rule treats where the fitted blip is positive.
    - ``"weighted_classification"``: ``learner`` classifies the sign of the same
      pseudo-outcome with its absolute value as the sample weight, and the rule treats where
      the fitted probability of a positive sign exceeds one half.

    The two pseudo-outcome losses read a binary treatment: ``1`` and ``0`` stand for the
    higher and the lower of the two sorted arm codes, a tie assigns the lower one, and a
    treatment with more than two arms refuses them (X11 (d2)).

    The pseudo-outcome of a training row is
    ``(2A - 1) / g(A | W) (Y - Qbar(A, W)) + Qbar(1, W) - Qbar(0, W)``, with ``Qbar`` and
    ``g`` from a 5-fold cross-fit inside the training rows that uses the fit's own outcome
    and treatment learners (Luedtke and van der Laan 2016, *International Journal of
    Biostatistics*, Sections 4.1 to 4.3).

    The object has no ``density`` method, so it never enters ``interventions=`` as a known
    regime.  Pass it as ``TMLE(learned_rule=LearnedRule())`` or
    ``LearnedRuleValue(rule=LearnedRule())``.  Known regimes passed beside it,
    ``TMLE(learned_rule=LearnedRule(), interventions=[Static(1)])``, are reported with their
    contrasts against the learned rule.

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
    the value of these fixed rules fold-wise, with the record's fold weights.  Simulated
    confounding and the refutations that refit read it as confounding of the value of these
    fixed rules, not of the learned-rule procedure.

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
        One ``int8`` arm code per row of the original fit: the code
        (:attr:`~cleverly.data.CausalData.arm_codes`, ``0`` to ``K - 1`` in the sorted
        order of the treatment levels) of the arm that row's fold rule assigns.  A subset
        or resampled refit passes the value of each kept row's source row.
    levels : tuple
        The treatment levels of the original fit, in code order.  A refit whose levels
        differ refuses, because its codes would name other arms.  Empty skips the check.
    """

    treat: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8), compare=False)
    levels: tuple[Any, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        values = np.asarray(self.treat)
        if values.ndim != 1 or values.dtype != np.int8 or np.any(values < 0):
            raise DataError("FrozenLearnedRule treat= holds one int8 arm code per row")
        object.__setattr__(self, "treat", values.copy())
        object.__setattr__(self, "levels", tuple(self.levels))

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
        The rule of the fit under its own name, loss and learner. ``treat`` is the arm
        code that each row's fold rule assigns, and ``levels`` the fit's treatment levels.

    Raises
    ------
    CapabilityError
        If the fit combines more than one split (:func:`learned_rule_splits_refusal`).
    """
    refusal = learned_rule_splits_refusal("freeze_learned_rule", result.n_repeats)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    values = realized_arms(result.nuisance.regimes)
    rule = result.estimator.learned_rule
    return FrozenLearnedRule(
        rule.name,
        values,
        tuple(result.data.treatment_levels),
        loss=rule.loss,
        learner=rule.learner,
    )


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

    A frozen rule returns its values.  The plug-in rule reads the out-of-fold predictions of
    the cross-fitted outcome regression at every arm or, in sample, those of the regression
    fitted on every row, and takes the best arm, a tie going to the lowest arm code.  A
    pseudo-outcome loss fits its rule learner inside each outer training fold, or on every row
    in sample (:func:`~cleverly.interventions._rule_learners.rule_learner_treat`),
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
        One ``int8`` arm code per row (:attr:`~cleverly.data.CausalData.arm_codes`): the
        arm that the rule of the row's fold assigns.
    margin : numpy.ndarray or None
        For a pseudo-outcome loss, the score of the row's fold rule minus its threshold,
        :math:`\\hat b_v(W_i)` or :math:`\\hat P_v(\\tilde D > 0 \\mid W_i) - 1/2`.
        ``None`` for the plug-in rule, whose score is the top-two gap, and for a frozen
        rule.
    fold_rules : numpy.ndarray or None
        ``(n, V)`` ``int8`` arm codes: column ``v`` is fold ``v``'s rule at every row.
        ``None`` for a frozen rule, which holds one realized rule per row and no fold rule.

    Raises
    ------
    DataError
        If a frozen rule holds another number of rows than the refit, or treatment levels
        other than the refit's, or if an inner training complement of a pseudo-outcome
        loss lacks an arm or an outcome class.
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
        levels = tuple(data.treatment_levels)
        if rule.levels and rule.levels != levels:
            raise DataError(
                f"the frozen learned rule was fitted on the treatment levels "
                f"{list(rule.levels)}, and the refit has {list(levels)}, so its arm codes "
                "would name other arms"
            )
        if int(np.max(rule.treat, initial=0)) >= data.n_arms:
            raise DataError(
                f"the frozen learned rule assigns an arm code above {data.n_arms - 1}, the "
                f"last code of the refit's {data.n_arms} arms"
            )
        return rule.treat, None, None
    if rule.loss == "plug_in":
        if nuisance.fold_plug_in_rules is None:
            raise RuntimeError(
                "a plug-in learned-rule fit reached its rule without every fold's rule at "
                "every row. This is a defect in cleverly: TMLE asks the nuisance fit for them."
            )
        realized, _ = _plug_in_rule(nuisance)
        return realized, None, np.asarray(nuisance.fold_plug_in_rules, dtype=np.int8)
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
    r"""Record one split of a learned-rule fit: its rules and its fold estimates.

    Parameters
    ----------
    seed : int
        The seed of the split's folds and learners, as ``result.repeats`` records it.
    scheme : str
        The scheme of the fit, one of :data:`SCHEMES`: ``"fold_evaluated"`` under
        ``cv_evaluation=True``, ``"stacked"`` under ``cv_evaluation=False``, and
        ``"in_sample"`` under ``cross_fit=False``, whose one pseudo-fold holds every row.
    estimate : float
        The split's estimate.  Under the fold-evaluated report it is
        :math:`(1/V)\sum_v \psi^*_{nv}`, and under the stacked report it is the pooled plug-in
        :math:`\sum_v (W_v/W)\psi^*_{nv}`, which :attr:`fold_weights` gives.  In sample it
        is the targeted plug-in of the full-sample rule on every row.
    fold_sizes : tuple of int
        The validation rows of each outer fold.
    fold_masses : tuple of float
        The observation-weight mass :math:`W_v` of each fold's validation rows, on the fit's
        mean-one weight scale.  Without weights it is the fold's row count.
    fold_estimates : tuple of float
        The targeted plug-in value :math:`\psi^*_{nv}` of each fold on its own validation rows.
        Under the fold-evaluated report each is a term of the estimate, and with baseline
        strata it is the share mixture of the fold's stratum plug-ins.  Under the stacked
        report it is descriptive: the weighted mean of :math:`\bar Q^*(d_i, W_i)` over the
        fold's rows at the fit's own fluctuation.  Pooled targeting does not solve each
        fold's score, so a fold value is not an estimate of :math:`\Psi_{d_{nv}}(P_0)`
        with its own interval, and only their weighted sum is reported.  In sample the one
        value is the estimate.
    arm_shares : tuple of tuple of float
        For each fold, the share of its validation rows that its rule assigns to each arm,
        in arm-code order (:attr:`LearnedRuleRecord.arms`).  Each fold's shares sum to 1.
        At two arms the second share is the share the rule treats.
    gap_quantiles : tuple of tuple of float
        For each fold, quantiles of the top-two gap of the outer outcome regression,
        :math:`\max_a \bar Q_v(a, W) - \max_{a \ne d_v(W)} \bar Q_v(a, W)`, on its
        validation rows, at :data:`BLIP_QUANTILE_LEVELS`, on the scale the outcome
        regression is fitted on.  At two arms it is the absolute blip
        :math:`|\bar Q_v(1, W) - \bar Q_v(0, W)|`.  For the plug-in rule it is the margin
        of the rule's own argmax, whose mass at 0 speaks to C3.  Under a pseudo-outcome
        loss it describes the outer outcome regression, not the rule.
    score_quantiles : tuple of tuple of float
        For a pseudo-outcome loss, for each fold, quantiles of the rule learner's score
        minus its threshold on the fold's validation rows, at :data:`BLIP_QUANTILE_LEVELS`:
        :math:`\hat b_v(w)` for ``loss="blip"`` and :math:`\hat P_v(\tilde D > 0 \mid w) - 1/2`
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
    scheme: str
    estimate: float
    fold_sizes: tuple[int, ...]
    fold_masses: tuple[float, ...]
    fold_estimates: tuple[float, ...]
    arm_shares: tuple[tuple[float, ...], ...]
    gap_quantiles: tuple[tuple[float, ...], ...]
    score_quantiles: tuple[tuple[float, ...], ...]
    inner_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}; got {self.scheme!r}")

    @property
    def n_folds(self) -> int:
        """Return the number of outer folds of the split."""
        return len(self.fold_sizes)

    @property
    def fold_weights(self) -> tuple[float, ...]:
        r"""Return the weight of each fold in the split's target and estimate.

        ``1/V`` for every fold under the fold-evaluated report, and the fold's share
        :math:`W_v/W` of the weight mass under the stacked report (``n_v/n`` without weights).
        The one pseudo-fold of an in-sample fit has weight 1.
        """
        if self.scheme == "fold_evaluated":
            return tuple(1.0 / self.n_folds for _ in self.fold_sizes)
        total = float(sum(self.fold_masses))
        return tuple(float(mass) / total for mass in self.fold_masses)


@dataclass(frozen=True)
class LearnedRuleRecord:
    """Record what a learned-rule fit estimated, under ``result.extra["learned_rule"]``.

    The fitted fold models are not kept.  The realized rule of each row in split ``r`` is
    the arm whose entry of the one-hot density
    ``result.repeats[r].nuisance.regimes.values[:, :, 0]`` is 1
    (:func:`realized_arms`), and the folds, the learner templates and the seeds replay the
    fit.  A fit of ``repeats=R`` holds one
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
    arms : tuple of str
        The treatment levels in arm-code order, the order of each split's
        :attr:`LearnedRuleSplit.arm_shares`.

    Attributes
    ----------
    n_splits : int
    scheme : str
    rule : str
    target : str
    quantile_levels : tuple of float
    """

    name: str
    splits: tuple[LearnedRuleSplit, ...]
    loss: str
    learner: str | None
    arms: tuple[str, ...]

    @property
    def n_splits(self) -> int:
        """Return the number of splits the reported mean combines."""
        return len(self.splits)

    @property
    def scheme(self) -> str:
        """Return the report of the fit, one of :data:`SCHEMES`, which every split shares."""
        return self.splits[0].scheme

    @property
    def rule(self) -> str:
        """Return the rule class of the fit's loss, in words (:data:`RULE_CLASS`)."""
        return RULE_CLASS[self.loss]

    @property
    def target(self) -> str:
        """Return what the reported value is under the fit's report (:data:`TARGET_KIND`)."""
        return TARGET_KIND[self.scheme]

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
            the treated share over the folds, or at more than two arms of each arm's share.
            The stacked report also gives the range of the fold weights.  A fit of several
            splits also names the split count, and gives both ranges over every split.
        """
        if self.scheme == "in_sample":
            return self._describe_in_sample()
        learned = {
            "plug_in": (
                "Each rule is the plug-in rule of the fold's outcome regression."
                if len(self.arms) == 2
                else "Each rule is the plug-in rule of the fold's outcome regression: the "
                f"arm of {len(self.arms)} with the highest predicted outcome, a tie going to "
                "the first arm in sorted order."
            ),
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
                "threshold on each fold's validation rows, and its gap_quantiles describe "
                "the outer outcome regression, not the rule."
            )
        source = (
            "The target depends on the data, because the rules do (van der Laan and Luedtke "
            f"2015, Section 7). {learned}"
        )
        if self.scheme == "stacked":
            weights = [weight for split in self.splits for weight in split.fold_weights]
            # Each fold mass is the fold's row count exactly when no weights were passed.
            weighted = any(
                tuple(float(size) for size in split.fold_sizes) != split.fold_masses
                for split in self.splits
            )
            share = "the total weight" if weighted else "the rows"
            average = "the average of the values"
            weighting = (
                f", weighted by each fold's share of {share} (fold weights "
                f"{min(weights):.3g} to {max(weights):.3g})"
            )
            evaluated = (
                "each evaluated on the fold it did not train on and pooled over every "
                "validation row (the stacked report)"
            )
        else:
            average, weighting = "the average value", ""
            evaluated = "each estimated on the fold it did not train on"
        if self.n_splits == 1:
            return (
                f"Target: {average} of the {self.splits[0].n_folds} rules learned on the "
                f"training folds{weighting}, {evaluated}. "
                f"{source} {self._shares('by fold')}"
            )
        estimates = [split.estimate for split in self.splits]
        return (
            f"Target: the mean over {self.n_splits} random splits of {average} of "
            f"the rules learned on the training folds of each split{weighting}, {evaluated}. "
            f"{source} Estimate of each split: "
            f"{min(estimates):.4g} to {max(estimates):.4g}. "
            f"{self._shares('by fold over every split')}"
        )

    def _shares(self, over: str) -> str:
        """The sentence of the rules' arm shares: the treated share at two arms, else each arm's.

        ``over`` names what the range runs over, such as ``"by fold"``.  An in-sample fit
        (``over=""``) has one share per arm and no range.
        """
        columns = list(
            zip(*(fold for split in self.splits for fold in split.arm_shares), strict=True)
        )
        if len(self.arms) == 2:
            low, high = min(columns[1]), max(columns[1])
            if not over:
                return f"Unweighted share of rows the rule treats: {low:.3g}."
            return f"Unweighted share of rows the rules treat, {over}: {low:.3g} to {high:.3g}."
        if not over:
            parts = "; ".join(
                f"A = {arm}: {column[0]:.3g}"
                for arm, column in zip(self.arms, columns, strict=True)
            )
            return f"Unweighted share of rows the rule assigns to each arm: {parts}."
        parts = "; ".join(
            f"A = {arm}: {min(column):.3g} to {max(column):.3g}"
            for arm, column in zip(self.arms, columns, strict=True)
        )
        return f"Unweighted share of rows the rules assign to each arm, {over}: {parts}."

    def _describe_in_sample(self) -> str:
        """The summary line of an in-sample fit (``cross_fit=False``), which has one rule."""
        learned = {
            "plug_in": (
                "The rule is the plug-in rule of the outcome regression fitted on every row."
                + (
                    ""
                    if len(self.arms) == 2
                    else f" It assigns the arm of {len(self.arms)} with the highest predicted "
                    "outcome, a tie going to the first arm in sorted order."
                )
            ),
            "blip": (
                f"The rule is learned on every row by {self.learner}, a regression of a doubly "
                "robust blip pseudo-outcome on the covariates (loss='blip')."
            ),
            "weighted_classification": (
                f"The rule is learned on every row by {self.learner}, a classifier of the sign "
                "of a doubly robust blip pseudo-outcome weighted by its size "
                "(loss='weighted_classification')."
            ),
        }[self.loss]
        return (
            "Target: the value of the one rule learned on every row, estimated on the same "
            "rows (cross_fit=False). It is the value of that full-sample rule, a different "
            "target from the fold average of a cross-fitted fit (van der Laan and Luedtke "
            f"2015, Section 6). {learned} {self._shares('')}"
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
    curve: :math:`n^{-2}\sum_i \bar\phi_i^2` for the fold-evaluated report, and
    :func:`~cleverly.inference.influence_variance`, the centred rule, for the
    stacked report.  The median of
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


def _plug_in_rule(nuisance: NuisanceEstimates) -> tuple[np.ndarray, np.ndarray]:
    r"""The out-of-fold plug-in rule at every row and its top-two gap.

    Row ``i`` of fold ``v`` carries the predictions of the outcome regression fitted on the
    training complement of ``v`` at every arm, so the rule at a row never reads that row.
    The rule is :math:`\arg\max_a \bar Q_v(a, W_i)`, a tie going to the lowest arm code
    (:func:`~cleverly.estimators._nuisance.plug_in_arm`, the reduction the fold workers
    apply to :attr:`~cleverly.estimators._nuisance.NuisanceEstimates.fold_plug_in_rules`).
    The gap is the best prediction minus the second best, 0 at a tie, and at two arms the
    absolute blip.

    Returns
    -------
    tuple of numpy.ndarray
        The ``int8`` arm code of every row, and the gap of every row.
    """
    from ..estimators._nuisance import plug_in_arm

    predictions = [
        np.asarray(nuisance.outcome.arms[level], dtype=float) for level in nuisance.outcome.levels
    ]
    ordered = np.sort(np.vstack(predictions), axis=0)
    return plug_in_arm(predictions), ordered[-1] - ordered[-2]


def realized_arms(regimes: RegimeSet | None) -> np.ndarray:
    """Return the arm code of each row's realized rule, from the learned column's density.

    Parameters
    ----------
    regimes : RegimeSet
        The regime set of a learned-rule fit, whose column 0 is the one-hot density of each
        row's fold rule over the arms.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` ``int8`` arm codes, the arm whose density entry is 1.

    Raises
    ------
    ValueError
        If ``regimes`` is ``None``.
    """
    if regimes is None:
        raise ValueError("realized_arms needs the regime set of a learned-rule fit")
    return np.asarray(np.argmax(np.asarray(regimes.values[:, :, 0]), axis=1), dtype=np.int8)


@dataclass(frozen=True)
class _LearnedRuleRegimes(RegimeSet):
    """The regime set of a learned-rule fit, with every fold's rule and the rule learner's margin.

    Parameters
    ----------
    names : tuple of str
        The one regime label.
    values : numpy.ndarray
        ``(n, K, R)`` densities: column 0 is the one-hot density of each row's own fold rule
        over the ``K`` arms, and any further column a known regime's.
    reference : float
        The code of the one regime.
    margin : numpy.ndarray or None
        The score of each row's fold rule minus its threshold, for a pseudo-outcome loss
        (:func:`learned_rule_treat`), or ``None``.  :func:`learned_rule_split` records its
        quantiles.
    fold_rules : numpy.ndarray or None
        ``(n, V)`` ``int8`` arm codes, the arm fold ``v``'s rule assigns row ``i``, or
        ``None`` for a frozen rule.  Row ``i``'s entry at its own fold is its realized rule,
        the arm of the 1 in ``values[i, :, 0]``.  The omitted-variable bound reads every
        column (:func:`fold_rules`).
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
        ``(n, V)`` ``int8`` arm codes in the order of ``nuisance.folds``, or ``None`` when
        the split stores none: a frozen rule's replay, which holds one realized rule per row.

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
    if regimes is None:
        raise ValueError("realized_arms needs the regime set of a learned-rule fit")
    values = np.asarray(regimes.values)
    return check_fold_rules(rules, realized_arms(regimes), nuisance.folds, n_arms=values.shape[1])


def check_fold_rules(
    rules: np.ndarray, realized: np.ndarray, folds: Folds, *, n_arms: int | None = None
) -> np.ndarray:
    """Check every fold's rule at every row against each row's realized rule.

    The fit runs it when it builds the regime set, and :func:`fold_rules` runs it again on
    every read, because a saved result can be edited.

    Parameters
    ----------
    rules : numpy.ndarray
        ``(n, V)`` ``int8`` arm codes, column ``v`` the rule of fold ``v`` at every row.
    realized : numpy.ndarray
        ``(n,)`` ``int8`` arm codes, the rule of each row's own fold.
    folds : Folds
        The split whose folds the columns follow.
    n_arms : int or None
        The number of arms, which every code must stay below, or ``None`` to skip that
        check.

    Returns
    -------
    numpy.ndarray
        ``rules`` as ``int8`` arm codes.

    Raises
    ------
    RuntimeError
        If the dtype or the shape does not match the split, if a code names no arm, or if
        the entry of some row at its own fold differs from its realized rule.  The fit
        evaluates both from one fitted model, so this is a defect in cleverly.
    """
    rules = np.asarray(rules)
    assignment = np.asarray(folds.assignment, dtype=int)
    if rules.dtype != np.int8 or rules.shape != (assignment.size, folds.n_folds):
        raise RuntimeError(
            f"the stored fold rules have dtype {rules.dtype} and shape {rules.shape}, and the "
            f"split needs int8 arm codes for {assignment.size} rows and {folds.n_folds} "
            "folds. This is a defect in cleverly."
        )
    if np.any(rules < 0) or (n_arms is not None and np.any(rules >= n_arms)):
        named = (
            "a negative arm code" if n_arms is None else f"an arm code outside 0 to {n_arms - 1}"
        )
        raise RuntimeError(f"a stored fold rule names {named}. This is a defect in cleverly.")
    realized = np.asarray(realized, dtype=np.int8)
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
    known: RegimeSet | None = None,
    reference: float = 0.0,
    *,
    n_arms: int = 2,
) -> RegimeSet:
    """The fold-local rule of every row, then any known regimes, as a :class:`RegimeSet`.

    The density of the rule is one-hot over the arms: ``values[i, a, 0]`` is ``1`` for the
    arm code ``a`` that the rule of row ``i``'s fold assigns, and ``0`` at every other arm.
    The rule's column is built directly, so no regime declaration is checked: the rule is
    learned, and the target that reads it is the learned-rule value, not a regime mean.
    The known regimes of ``interventions=`` (X11 (c)) follow as codes ``1`` to ``K``, with
    the densities :meth:`RegimeSet.evaluate` gave them, so one fluctuation targets every
    column on the same folds.

    Parameters
    ----------
    treat : numpy.ndarray
        The ``int8`` arm code of the rule of every row (:func:`learned_rule_treat`).
    name : str
        The name of the regime, the rule's label.
    margin : numpy.ndarray or None
        The rule learner's score minus its threshold at every row, or ``None``.
    fold_rules : numpy.ndarray or None
        Every fold's rule at every row, ``(n, V)`` ``int8`` arm codes, or ``None`` for a
        frozen rule.
    known : RegimeSet or None
        The evaluated known regimes declared beside the rule, or ``None``.
    reference : float
        The code contrasts are taken against: ``0`` when ``reference=`` names the learned
        rule or no known regime is declared, and ``1`` otherwise, which the contrast target
        reads as "the learned rule minus each known regime".
    n_arms : int
        The number of arms of the treatment.

    Returns
    -------
    RegimeSet
        The rule, named ``name``, then the known regimes, over the ``n_arms`` arms.
    """
    treat = np.asarray(treat, dtype=np.int8)
    values = np.zeros((treat.size, n_arms, 1), dtype=float)
    values[np.arange(treat.size), treat, 0] = 1.0
    names: tuple[str, ...] = (name,)
    if known is not None:
        values = np.concatenate([values, np.asarray(known.values, dtype=float)], axis=2)
        names = (name, *known.names)
    if margin is None and fold_rules is None:
        return RegimeSet(names, values, float(reference))
    return _LearnedRuleRegimes(
        names,
        values,
        float(reference),
        None if margin is None else np.asarray(margin, dtype=float),
        None if fold_rules is None else np.asarray(fold_rules, dtype=np.int8),
    )


def stacked_fold_estimates(
    nuisance: NuisanceEstimates, targeted: Any, weights: np.ndarray
) -> tuple[float, ...]:
    r"""Return each fold's weighted mean of the targeted value of its rule, on the outcome scale.

    The stacked report's estimate is the pooled weighted mean of
    :math:`\bar Q^*(d_i, W_i)` over every validation row, which equals
    :math:`\sum_v (W_v/W)\psi^*_{nv}` with these fold values.  The values are descriptive:
    pooled targeting solves the pooled score, not each fold's.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The split's cross-fitted nuisances, whose folds and one-hot regime the values read.
    targeted : InitialFit
        The targeted outcome regression of the split's fluctuation, on the fitted scale.
    weights : numpy.ndarray
        The observation weights of the rows.

    Returns
    -------
    tuple of float
        One value per fold, in the order of ``nuisance.folds``.
    """
    assert nuisance.regimes is not None
    density = np.asarray(nuisance.regimes.values[:, :, 0], dtype=float)
    predictions = np.column_stack(
        [np.asarray(targeted.arms[level], dtype=float) for level in targeted.levels]
    )
    mixture = np.einsum("ij,ij->i", density, predictions)
    weights = np.asarray(weights, dtype=float)
    scaler = nuisance.scaler
    out = []
    for _, test in nuisance.folds:
        value = float(np.average(mixture[test], weights=weights[test]))
        out.append(value if scaler.is_identity else float(scaler.unscale_level(value)))
    return tuple(out)


def learned_rule_split(
    rule: LearnedRule,
    nuisance: NuisanceEstimates,
    fold_estimates: Sequence[float],
    *,
    seed: int,
    estimate: float,
    scheme: str,
    weights: np.ndarray,
) -> LearnedRuleSplit:
    """Build the record of one split of a learned-rule fit.

    Parameters
    ----------
    rule : LearnedRule
        The declared rule.
    nuisance : NuisanceEstimates
        The split's cross-fitted nuisances, whose folds and regimes the record reads.
    fold_estimates : sequence of float
        The fold plug-in values, in the order of ``nuisance.folds``: the fold-evaluated
        report's own, or :func:`stacked_fold_estimates` under the stacked report.
    seed : int
        The split's seed.
    estimate : float
        The split's estimate under its report.
    scheme : str
        The report, one of :data:`SCHEMES`.
    weights : numpy.ndarray
        The observation weights of the rows, whose fold sums are the fold masses.

    Returns
    -------
    LearnedRuleSplit
        The per-fold summaries of the split.
    """
    _, gap = _plug_in_rule(nuisance)
    assert nuisance.regimes is not None
    treat = realized_arms(nuisance.regimes)
    n_arms = int(np.asarray(nuisance.regimes.values).shape[1])
    tests = [np.asarray(test) for _, test in nuisance.folds]
    learns = rule.loss != "plug_in" and not isinstance(rule, FrozenLearnedRule)
    margin = getattr(nuisance.regimes, "margin", None) if learns else None
    weights = np.asarray(weights, dtype=float)
    return LearnedRuleSplit(
        seed=int(seed),
        scheme=scheme,
        estimate=float(estimate),
        fold_sizes=tuple(int(test.size) for test in tests),
        fold_masses=tuple(float(np.sum(weights[test])) for test in tests),
        fold_estimates=tuple(float(value) for value in fold_estimates),
        arm_shares=tuple(
            tuple(float(share) for share in np.bincount(treat[test], minlength=n_arms) / test.size)
            for test in tests
        ),
        gap_quantiles=tuple(
            tuple(float(value) for value in np.quantile(gap[test], BLIP_QUANTILE_LEVELS))
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


def learned_rule_record(
    rule: LearnedRule, splits: Sequence[LearnedRuleSplit], arms: Sequence[Any]
) -> LearnedRuleRecord:
    """Build the record of a learned-rule fit from the records of its splits.

    Parameters
    ----------
    rule : LearnedRule
        The declared rule.
    splits : sequence of LearnedRuleSplit
        One record per split, in the order of ``result.repeats``.
    arms : sequence
        The treatment levels in arm-code order
        (:attr:`~cleverly.data.CausalData.treatment_levels`).

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
        arms=tuple(str(arm) for arm in arms),
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


#: The targets of a learned-rule fit that read a known regime declared beside the rule
#: (X11 (c)).  A fit without ``interventions=`` reports neither, and refuses either by name.
#: No simultaneous band covers them: ``simultaneous_bands()`` refuses them with
#: :data:`~cleverly.inference.multiplier.KNOWN_REGIME_BAND_REFUSAL`, and a fit skips its
#: default band (X11 (c2)).
CONTRAST_TARGETS: tuple[str, ...] = ("ey_known_regime", "ate_learned_rule")


def learned_rule_estimands(estimator: Any, names: Sequence[str]) -> tuple[str, ...]:
    """Resolve the estimands of a learned-rule fit: the value first, then its contrasts.

    The learned-rule axis registers :data:`CONTRAST_TARGETS` in its default report, and they
    read the known regimes of ``interventions=``.  A fit without them reports the learned-rule
    value alone.  An explicit request of either without ``interventions=`` refuses first, in
    :func:`learned_rule_configuration_refusal`.  The learned-rule value is always reported,
    first: the record ``result.extra["learned_rule"]`` describes it, and every contrast
    reads it.

    Parameters
    ----------
    estimator : TMLE
        The estimator.
    names : sequence of str
        The resolved estimand names.

    Returns
    -------
    tuple of str
        ``names`` with ``ey_learned_rule`` first, and without the contrast targets on a
        learned-rule fit with no known regime.  Any other fit's ``names`` unchanged.
    """
    if getattr(estimator, "learned_rule", None) is None:
        return tuple(names)
    known = bool(getattr(estimator, "interventions", ()))
    rest = [
        name
        for name in names
        if name != "ey_learned_rule" and (known or name not in CONTRAST_TARGETS)
    ]
    return ("ey_learned_rule", *rest)


def _requested(estimands: Any) -> tuple[str, ...]:
    """The estimand names a caller wrote, or ``()`` for the default or ``"all"``."""
    if estimands is None or estimands == "all":
        return ()
    return (estimands,) if isinstance(estimands, str) else tuple(estimands)


def learned_rule_retarget_refusal(estimator: Any, requested: Sequence[str]) -> Refusal | None:
    """The refusal of a retarget that asks a learned-rule fit for a target it cannot report.

    :meth:`~cleverly.estimators.TMLE.retarget` can name any target of the fit's axis, and a
    known-regime target on a fit without ``interventions=`` would otherwise return no
    estimate.  It refuses as the constructor refuses the same request.

    Parameters
    ----------
    estimator : TMLE
        The estimator.
    requested : sequence of str
        The targets the retarget names.

    Returns
    -------
    Refusal or None
        The ``INVALID_DECLARATION`` refusal, or ``None``.
    """
    if getattr(estimator, "learned_rule", None) is None:
        return None
    if getattr(estimator, "interventions", ()):
        return None
    if not any(name in CONTRAST_TARGETS for name in requested):
        return None
    return _contrast_refusal(estimator, requested)


def _contrast_refusal(estimator: Any, requested: Sequence[str] | None = None) -> Refusal | None:
    """The refusal of a known-regime contrast that the learned-rule fit cannot report.

    Three declarations: a known-regime target without ``interventions=``, ``reference=``
    without ``interventions=``, and ``reference=`` naming a known regime, which would ask for
    a contrast of two known regimes.  A known regime that takes the learned rule's name is
    refused too, because every reported name would be ambiguous.  ``requested`` replaces
    the estimator's ``estimands=`` for a retarget, which names its own targets.
    """
    interventions = tuple(getattr(estimator, "interventions", ()) or ())
    reference = getattr(estimator, "reference", None)
    rule = estimator.learned_rule
    names = [str(item.name) for item in interventions]
    if not interventions:
        written = (
            _requested(getattr(estimator, "estimands", None))
            if requested is None
            else tuple(requested)
        )
        asked = [name for name in written if name in CONTRAST_TARGETS]
        if asked or reference is not None:
            named = [
                *([f"estimands={asked}"] if asked else []),
                *(["reference="] if reference is not None else []),
            ]
            return Refusal(
                RefusalReason.INVALID_DECLARATION,
                f"{' and '.join(named)} on a learned-rule fit read a known regime declared "
                "beside the rule with interventions=, and this fit declares none. Its one "
                "estimand is the learned-rule value, which has no contrast.",
                alternative=(
                    "Declare the known regimes to compare against, for example "
                    "interventions=[Static(1)], or drop "
                    f"{' and '.join(named)} to fit the learned-rule value alone."
                ),
            )
        return None
    if rule.name in names:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"the learned rule and a known regime of interventions= are both named "
            f"{rule.name!r}, so the reported values and contrasts could not tell them apart.",
            alternative=(
                "Give the learned rule another name, LearnedRule(name=...), or rename the regime."
            ),
        )
    if reference is not None and str(reference) != rule.name:
        if str(reference) in names:
            return Refusal(
                RefusalReason.INVALID_DECLARATION,
                f"reference={reference!r} names a known regime. On a learned-rule fit every "
                "contrast reads the learned rule: the learned rule minus each known regime by "
                "default, and each known regime minus the learned rule with reference= naming "
                "the learned rule. A contrast of two known regimes is a regime contrast, which "
                "a learned-rule fit does not report.",
                alternative=(
                    f"Leave reference=None, or pass reference={rule.name!r}. Fit the contrast "
                    "of two known regimes with "
                    "TMLE(interventions=[...], reference=...), a fit of its own."
                ),
            )
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"reference={reference!r} is neither the learned rule {rule.name!r} nor one of "
            f"the known regimes {names}.",
            alternative=f"Leave reference=None, or pass reference={rule.name!r}.",
        )
    return None


def learned_rule_configuration_refusal(estimator: Any) -> Refusal | None:
    """Return the refusal of a learned-rule estimator that no data can change, or ``None``.

    The first two rows of the learned-rule refusal table in
    ``docs/technical-reference/point-treatment-tmle.md``: a collaborative or doubly robust
    estimator, and ``learned_rule=`` beside another axis keyword or an arm estimand.  Known
    regimes beside the rule (``interventions=``, and ``reference=`` naming the learned rule)
    pass (X11 (c)); a contrast declaration they cannot report refuses here too.  The engine
    asks it at construction and again first in its preflight.

    Parameters
    ----------
    estimator : TMLE
        The estimator, with ``learned_rule`` set.

    Returns
    -------
    Refusal or None
        The refusal, an ``IMPLEMENTATION_GAP`` or ``INVALID_DECLARATION`` record, or
        ``None``.
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
    axes = [
        keyword
        for keyword, value in (
            ("policies=", getattr(estimator, "policies", ())),
            ("incremental=", getattr(estimator, "incremental", ())),
            ("msm=", getattr(estimator, "msm", None) is not None),
        )
        if value
    ]
    if axes:
        return _gap(
            "learned_rule= estimates a data-adaptive value, and beside it known regimes of "
            f"interventions=. Another parameter axis ({', '.join(axes)}) needs a fit of its "
            "own, because one fluctuation cannot solve the score equations of two axes, and "
            f"{_X11.format(part='c2')} holds the joint fit.",
            f"Fit the learned-rule value without {' or '.join(axes)}, and fit the other "
            "target in a fit of its own.",
            "X11",
        )
    other = _arm_estimands(getattr(estimator, "estimands", None))
    regime = [name for name in other if name in ("ey_regime", "ate_regime")]
    if regime:
        return Refusal(
            RefusalReason.API_REDIRECTION,
            f"estimands={regime} are the estimands of a regime fit without learned_rule=. On "
            "a learned-rule fit, a known regime of interventions= reports its value as "
            "ey_known_regime and its contrast with the learned rule as ate_learned_rule.",
            alternative=(
                "Request estimands=['ey_learned_rule', 'ey_known_regime', 'ate_learned_rule'], "
                "or estimands=None for all three."
            ),
        )
    arm = [name for name in other if name not in regime]
    if arm:
        return Refusal(
            RefusalReason.API_REDIRECTION,
            f"learned_rule= reports the learned-rule value and its contrasts with known "
            f"regimes, and estimands={list(arm)} name arm estimands, which index the "
            "treatment arms rather than regimes. A contrast with a static arm is a contrast "
            "with the known regime that always assigns that arm.",
            alternative=(
                "Declare the arm as a static known regime beside the rule, "
                "interventions=[Static(1)] for the higher arm code, with estimands=None, and "
                "read ate_learned_rule[...] for the contrast."
            ),
        )
    return _contrast_refusal(estimator)


def _data_refusal(data: CausalData, rule: LearnedRule | None = None) -> Refusal | None:
    """The data rows of the learned-rule refusal table: the declarations no setting repairs.

    ``docs/technical-reference/point-treatment-tmle.md``, "Learned rules", holds the table.
    ``rule`` is the declared rule, which the row of a rule learner on more than two arms
    reads.  ``None`` skips that row.
    """
    if data.is_continuous_treatment:
        return _gap(
            "a learned rule over a continuous treatment is refused. No reviewed source "
            f"defines a learned rule over a dose for the learned-rule value ({_F27}). The "
            "learned-rule value needs an arm-coded treatment.",
            "A fit without learned_rule= can use "
            "TMLE(policies=[ModifiedPolicy(..., policy_kind='known')]) for a fixed policy on a "
            "continuous treatment.",
            "F27",
        )
    if (
        rule is not None
        and rule.loss != "plug_in"
        and not isinstance(rule, FrozenLearnedRule)
        and not data.is_binary_treatment
    ):
        return _gap(
            f"LearnedRule(loss={rule.loss!r}) over a treatment with {data.n_arms} arms is "
            "refused. Its pseudo-outcome (2A - 1) / g(A | W) (Y - Qbar(A, W)) + Qbar(1, W) - "
            "Qbar(0, W) and its rule 1{score > threshold} compare two arms, and no multi-arm "
            "loss is implemented (Luedtke and van der Laan 2016 state both losses for a "
            f"binary treatment; {_X11.format(part='d2')}).",
            "LearnedRule() (loss='plug_in') learns the argmax rule over every arm from the "
            f"fit's own outcome regression, and {_FIXED_RULE} estimates a fixed rule on a "
            "categorical treatment.",
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
    if data.has_strata and data.cluster is not None:
        return _gap(
            "a learned rule with baseline strata (strata=) and clusters (id=) is refused under "
            "either report. Each stratum's variance would compare the cluster totals of its "
            "stratum, inside every validation fold under cv_evaluation=True and over the "
            "pooled validation rows under the stacked report, and neither variance nor its "
            "Student t reference has been checked (X28 in docs/roadmap.md).",
            "Fit the clustered learned-rule value without strata=, or the stratified "
            "learned-rule value without id=.",
            "X28",
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
    parameter, so no setting reaches it.  ``cv_evaluation=False`` is no refusal either: the
    fit reports the stacked value, whose target weights fold :math:`v` by :math:`W_v/W`.
    Nor is ``cross_fit=False``: the fit reports the value of the one rule learned on every
    row (X11 (a)).  ``cross_fit=True`` with one fold meets the shared fold-policy refusal,
    whose in-sample offer names that change of target.

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
        # The remedy keeps the caller's target: a cross-fitted caller is named the
        # configuration that fits, and an in-sample caller (X11 (a)) keeps its in-sample fit.
        alternative = (
            f"{remedy} That fit reports the influence-curve interval."
            if estimator.cross_fit
            else "Fit the value of the full-sample rule with n_bootstrap=0 "
            "(Inference(n_bootstrap=0) on the method), which reports its influence-curve "
            "interval."
        )
        return Refusal(
            RefusalReason.INFERENCE_GAP,
            "the learned-rule value refuses the full-refit bootstrap (n_bootstrap=0). Each "
            "resample relearns the rules, so each draw has a different target "
            f"({_F27}).",
            alternative=alternative,
            roadmap="F27",
        )
    return None


#: The nuisance role of each learner-family option of an in-sample key, in words.
_FAMILY_ROLES: dict[str, str] = {
    "outcome_learner_family": "outcome learner",
    "treatment_learner_family": "treatment learner",
    "rule_learner_family": "rule learner",
}


def in_sample_learner_condition(estimator: Any, data: CausalData) -> Refusal | None:
    """Return the support condition of an in-sample learned-rule fit with an adaptive learner.

    An in-sample fit (``cross_fit=False``, X11 (a)) estimates the value of the one rule
    learned on every row.  Its interval needs condition C5a: the curve
    :math:`D^*(d_n, \bar Q^*_n, g_n)` lies in a :math:`P_0`-Donsker class with probability
    tending to one (van der Laan and Luedtke 2015, Section 6, Theorem 5).  A parametric
    learner, a fixed-dimension model that selects nothing
    (:func:`~cleverly._support.learner_family`), gives it, and an adaptive learner fitted
    on every row need not.  The fit runs, labelled experimental, and warns.

    Parameters
    ----------
    estimator : TMLE
        The estimator.
    data : CausalData
        The prepared data.

    Returns
    -------
    Refusal or None
        An ``INFERENCE_GAP`` condition that names the adaptive learners, or ``None`` for a
        cross-fitted fit, a fit that is not a learned-rule fit, or one whose learners are
        all parametric or known.
    """
    if getattr(estimator, "learned_rule", None) is None or estimator.cross_fit:
        return None
    from .._support import in_sample_learned_rule_learners

    families = in_sample_learned_rule_learners(estimator, data)
    adaptive = [_FAMILY_ROLES[name] for name, family in families.items() if family == "adaptive"]
    if not adaptive:
        return None
    roles = " and ".join(adaptive)
    return Refusal(
        RefusalReason.INFERENCE_GAP,
        f"this in-sample learned-rule fit (cross_fit=False) uses an adaptive {roles}. The "
        "interval of the full-sample rule's value needs the influence curve at the fitted "
        "rule, outcome regression and treatment model to stay in a Donsker class (condition "
        "C5a; van der Laan and Luedtke 2015, Theorem 5), which an adaptive learner fitted on "
        "every row need not do, and then the interval can be too narrow",
        alternative=(
            "Cross-fitting (cross_fit=True with n_folds of at least 2; CrossFitting(enabled=True) "
            "on the method) removes condition C5a, and it estimates the cross-fitted "
            "learned-rule value, a different target. In "
            "sample, a parametric learner meets C5a: LogisticRegression, LinearRegression, "
            "GLM, or a fixed-dimension class that declares cleverly_learner_family = "
            "'parametric'."
        ),
    )


def refuse_learned_rule_composition(
    data: CausalData, estimator: Any = None, *, rule: LearnedRule | None = None
) -> None:
    """Refuse a learned-rule request that the learned-rule contract does not cover.

    The checks follow the refusal table of the learned-rule contract in
    ``docs/technical-reference/point-treatment-tmle.md``.  A refusal that no setting repairs comes
    before a refusal whose remedy is a setting.  So a caller who follows one remedy never meets an
    earlier refusal next.

    ======  ===================================================  ===================
    order   request                                              item
    ======  ===================================================  ===================
    1       ``CTMLE`` or ``DRTMLE``                              F27
    2       another axis keyword                                 X11 (c2)
            an arm estimand or ``ey_regime`` (a redirection)     none, a redirection
            a contrast declaration the fit cannot report         none, a declaration
    3       a continuous treatment                               F27
    4       a rule learner (``loss="blip"`` or                   X11 (d2)
            ``"weighted_classification"``) on more than two
            arms
    5       missing outcomes, ``delta=``                         F27
    6       ``intermediate=``                                    X11 (j)
    7       estimated weights, ``weights_estimated=True``        X11 (k2)
    8       ``strata=`` with clusters, ``id=``                   X28
    9       ``repeats`` above 1 with clusters, ``id=``           X11 (i2)
    10      ``n_bootstrap`` above 0                              F27
    11      ``loss="blip"`` with ``weights=`` and a learner      none, a declaration
            whose ``fit`` takes no ``sample_weight``
    ======  ===================================================  ===================

    The simultaneous band is no row. A fit with known regimes skips the band over their
    values and contrasts, and ``summary()`` says why;
    :func:`~cleverly.inference.simultaneous_bands` called on them refuses (X11 (c2)).

    ``cross_fit=False`` is no row: it fits the value of the one rule learned on every row
    (X11 (a)).  ``cross_fit=True`` with one fold, and ``repeats`` above 1 without
    cross-fitting, meet the shared fold-policy refusals, whose in-sample offer names that
    change of target.

    More than two arms is no row: the plug-in rule is the argmax over every arm (X11 (d)).

    ``TMLE`` runs it first in its preflight, before any learner.
    ``CausalStudy.identify`` runs it with no estimator and the estimand's rule, so the data
    rows refuse there.

    Parameters
    ----------
    data : CausalData
        The prepared data of the fit.
    estimator : TMLE or None
        The estimator, or ``None`` for the checks that read the data alone.
    rule : LearnedRule or None
        The declared rule when no estimator is passed, for the row of a rule learner on
        more than two arms.  The estimator's ``learned_rule`` otherwise.

    Raises
    ------
    CapabilityError
        If the request meets a row of the table.
    """
    refusal = None if estimator is None else learned_rule_configuration_refusal(estimator)
    if estimator is not None:
        rule = estimator.learned_rule
    if refusal is None:
        refusal = _data_refusal(data, rule)
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
