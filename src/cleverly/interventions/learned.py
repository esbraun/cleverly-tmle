r"""The value of a treatment rule learned inside each training fold.

A :class:`~cleverly.interventions.Rule` is a *known* function of the covariates, and a rule
learned from the analysis sample is refused there.  This module holds
the one learned-rule target the package estimates instead.  The target is data-adaptive:

.. math::

    \tilde\psi_{0n} = \frac1V \sum_{v=1}^V \Psi_{d_{nv}}(P_0), \qquad
    d_{nv}(w) = \mathbb 1\{\bar Q_{nv}(1, w) - \bar Q_{nv}(0, w) > 0\},

where :math:`\bar Q_{nv}` is the outcome regression fitted on the training rows of outer
fold :math:`v`.  Van der Laan and Luedtke (2015), Section 7 and Appendix B, define the
target and its cross-validated TMLE.  The estimator reuses the regime fluctuation with one
pooled :math:`\varepsilon`, the fold-evaluated report and the cross-validated variance
(``TMLE(cv_evaluation=True)``).  This module adds the fold-local rule, the result record and
the refusals.  No new submodel, fluctuation, influence curve, variance or learner protocol
enters.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import CapabilityError, DataError, Refusal, RefusalReason
from .base import RegimeSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators._nuisance import NuisanceEstimates

__all__ = [
    "BLIP_QUANTILE_LEVELS",
    "LEARNED_RULE_BOUND_REFUSAL",
    "LEARNED_RULE_CONFIGURATION",
    "LEARNED_RULE_EVALUE_REFUSAL",
    "LEARNED_RULE_REFIT_REFUSAL",
    "LEARNED_RULE_REMEDY",
    "LEARNED_RULE_TILT_REFUSAL",
    "RULE_CLASS",
    "TARGET_KIND",
    "FrozenLearnedRule",
    "LearnedRule",
    "LearnedRuleRecord",
    "freeze_learned_rule",
    "frozen_rule_replay",
    "learned_rule_configuration_refusal",
    "learned_rule_record",
    "learned_rule_scheme_refusal",
    "refuse_learned_rule_composition",
]

#: The rule class of version one, in words.  The record quotes it.  ``1`` and ``0`` stand
#: for the higher and the lower of the two sorted arm codes.
RULE_CLASS = (
    "the plug-in rule 1{Qbar_v(1, W) - Qbar_v(0, W) > 0} of the outcome regression fitted "
    "on the training rows of each outer fold, where 1 is the higher arm code and 0 the "
    "lower; a tie assigns the lower arm code"
)

#: What the reported value is.  The record and the summary quote it.
TARGET_KIND = "fold-average data-adaptive value"

#: The quantile levels of the estimated blip that the record keeps for each fold.
BLIP_QUANTILE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

#: The configuration that fits the learned-rule value.  Every refusal whose remedy is a
#: configuration names it, and ``tests/unit/test_learned_rule_refusals.py`` fits it.
LEARNED_RULE_CONFIGURATION = (
    "cross_fit=True, cv_evaluation=True on the engine, or "
    "CrossFitting(enabled=True, fold_evaluation=True) on the method, with n_folds of at least "
    "2, repeats=1, targeting_scheme='pooled' and n_bootstrap=0 (Inference(n_bootstrap=0) on "
    "the method)"
)

#: The way out of a shared refusal that a drawn split raises.  The learned-rule value has
#: no in-sample fit (X11 (a)), so the in-sample remedy of those refusals does not apply.
LEARNED_RULE_REMEDY = (
    "fit the learned-rule value on a larger sample, because it has no in-sample fit "
    "(X11 (a) in docs/roadmap.md)"
)

#: The reason every sensitivity analysis of a learned-rule fit shares.  Each analysis
#: appends the roadmap row that holds its own missing work.
_LEARNED_RULE_TARGET = (
    "this sensitivity analysis of the learned-rule value is not implemented. The target averages "
    "the values of rules learned inside each training fold, and a refit relearns those "
    "rules, so a refitted or perturbed fit estimates a different target"
)

#: Why the omitted-variable bound refuses a learned-rule fit.  With the fold rules frozen,
#: the value is a linear functional, and the bound of a known regime is X82 part G's open work.
LEARNED_RULE_BOUND_REFUSAL = (
    f"{_LEARNED_RULE_TARGET}. X11 (n) in docs/roadmap.md tracks the bound of the frozen "
    "fold rules, after X82 part G covers a known regime"
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

#: Why a refutation refuses a learned-rule fit.
LEARNED_RULE_REFIT_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "bootstrap_measurement_error perturbs a covariate of a learned-rule fit. The refit holds "
    "the realized rule of each original row fixed, and the fit keeps no fold rule that "
    "assigns a perturbed covariate row. X11 (p) in docs/roadmap.md tracks a replay that "
    "keeps the fold rules themselves.",
    alternative=(
        "refute(result, estimand=..., tests=('placebo', 'random_common_cause', 'subset', "
        "'bootstrap_stability')) holds the realized rules fixed."
    ),
    roadmap="X11",
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

    The rule is learned inside each outer training fold.  Version one has one rule class:
    the plug-in rule of the fit's own outcome regression,
    ``d_v(w) = 1{Qbar_v(1, w) - Qbar_v(0, w) > 0}``, where ``Qbar_v`` is fitted on the
    training rows of fold ``v``.  ``1`` and ``0`` stand for the higher and the lower of
    the two sorted arm codes, so a tie assigns the lower arm code.  The rule needs no
    learner beyond the outcome regression, and no fit beyond the cross-fitted
    nuisances.

    The object has no ``density`` method, so it never enters ``interventions=`` as a known
    regime.  Pass it as ``TMLE(learned_rule=LearnedRule())`` or
    ``LearnedRuleValue(rule=LearnedRule())``.

    Parameters
    ----------
    name : str
        The label that the reported parameter carries, as in
        ``ey_learned_rule[learned rule]``.

    See Also
    --------
    cleverly.LearnedRuleValue : The typed estimand that holds this rule.
    cleverly.interventions.Rule : A known rule, fixed before the fit.

    Examples
    --------
    >>> from cleverly.interventions import LearnedRule
    >>> LearnedRule().name
    'learned rule'
    """

    name: str = "learned rule"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataError(f"LearnedRule name= must be a non-empty string; got {self.name!r}")


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
        return FrozenLearnedRule(self.name, self.treat[np.asarray(rows, dtype=int)])


def freeze_learned_rule(result: Any) -> FrozenLearnedRule:
    """Return the realized fold rules of a learned-rule fit, one value per row.

    Parameters
    ----------
    result : TMLEResult
        A learned-rule fit.

    Returns
    -------
    FrozenLearnedRule
        The rule of the fit under its own name. ``treat`` is ``True`` where the row's fold
        rule assigns the higher arm code.
    """
    values = np.asarray(result.nuisance.regimes.values[:, 1, 0], dtype=float) > 0.5
    return FrozenLearnedRule(result.estimator.learned_rule.name, values)


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


def _rule_treat(nuisance: NuisanceEstimates, rule: LearnedRule) -> np.ndarray:
    """The rule of every row: the frozen values, or the out-of-fold plug-in rule."""
    if isinstance(rule, FrozenLearnedRule):
        n = np.asarray(nuisance.outcome.arms[nuisance.outcome.levels[0]]).shape[0]
        if rule.treat.shape != (n,):
            raise DataError(
                f"the frozen learned rule holds {rule.treat.size} rows, and the refit has {n}"
            )
        return rule.treat
    return _blip(nuisance) > 0.0


@dataclass(frozen=True)
class LearnedRuleRecord:
    """Record what a learned-rule fit estimated, under ``result.extra["learned_rule"]``.

    The fitted fold models are not kept.  The realized rule of each row is
    ``result.nuisance.regimes.values[:, 1, 0]``, and the folds, the learner templates and
    the seeds replay the fit.

    Parameters
    ----------
    name : str
        The label of the rule.
    fold_sizes : tuple of int
        The validation rows of each outer fold.
    fold_estimates : tuple of float
        The targeted plug-in value of each fold on its own validation rows.
    treated_shares : tuple of float
        The share of each fold's validation rows that its rule assigns to the higher
        arm code.
    blip_quantiles : tuple of tuple of float
        For each fold, quantiles of the estimated blip ``Qbar_v(1, W) - Qbar_v(0, W)`` on
        its validation rows, at :attr:`quantile_levels`, on the scale the outcome
        regression is fitted on.

    Attributes
    ----------
    n_folds : int
    fold_weights : tuple of float
    rule : str
    target : str
    quantile_levels : tuple of float
    """

    name: str
    fold_sizes: tuple[int, ...]
    fold_estimates: tuple[float, ...]
    treated_shares: tuple[float, ...]
    blip_quantiles: tuple[tuple[float, ...], ...]

    @property
    def n_folds(self) -> int:
        """Return the number of outer folds."""
        return len(self.fold_sizes)

    @property
    def fold_weights(self) -> tuple[float, ...]:
        """Return the weight of each fold in the reported average, ``1/V`` for every fold."""
        return tuple(1.0 / self.n_folds for _ in self.fold_sizes)

    @property
    def rule(self) -> str:
        """Return the rule class, in words (:data:`RULE_CLASS`)."""
        return RULE_CLASS

    @property
    def target(self) -> str:
        """Return what the reported value is (:data:`TARGET_KIND`)."""
        return TARGET_KIND

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the levels of ``blip_quantiles`` (:data:`BLIP_QUANTILE_LEVELS`)."""
        return BLIP_QUANTILE_LEVELS

    def describe(self) -> str:
        """Return the line that :meth:`~cleverly.TMLEResult.summary` prints for this fit.

        Returns
        -------
        str
            The target, the fold count, the source of the definition, and the range of the
            treated share over the folds.
        """
        low, high = min(self.treated_shares), max(self.treated_shares)
        return (
            f"target: average over {self.n_folds} training-fold rules of each rule's value "
            "(data-adaptive; van der Laan and Luedtke 2015, Section 7); "
            f"treated share by fold {low:.3g} to {high:.3g}"
        )


def _blip(nuisance: NuisanceEstimates) -> np.ndarray:
    """The out-of-fold blip ``Qbar_v(1, W_i) - Qbar_v(0, W_i)`` at every row.

    Row ``i`` of fold ``v`` carries the prediction of the outcome regression fitted on the
    training complement of ``v``, so the blip at a row never reads that row.  Row 4 of
    the refusal table refuses a treatment with more than two arms first.
    """
    control, treated = nuisance.outcome.levels
    return np.asarray(nuisance.outcome.arms[treated], dtype=float) - np.asarray(
        nuisance.outcome.arms[control], dtype=float
    )


def _learned_rule_regimes(nuisance: NuisanceEstimates, rule: LearnedRule) -> RegimeSet:
    """The fold-local plug-in rule of every row, as a one-regime :class:`RegimeSet`.

    The density is one-hot: ``values[i, 1, 0]`` is ``1`` when the rule of row ``i``'s fold
    assigns it the higher arm code, and ``values[i, 0, 0]`` is ``1`` otherwise.  A tie
    assigns the lower arm code.  The set
    is built directly, so no regime declaration is checked: the rule is learned, and the
    target that reads it is the learned-rule value, not a regime mean.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The cross-fitted nuisances of the fit.
    rule : LearnedRule
        The declared rule, which names the regime.

    Returns
    -------
    RegimeSet
        One regime, named ``rule.name``, over the two arms.
    """
    treat = _rule_treat(nuisance, rule)
    values = np.zeros((treat.size, 2, 1), dtype=float)
    values[:, 1, 0] = treat
    values[:, 0, 0] = ~treat
    return RegimeSet((rule.name,), values, 0.0)


def learned_rule_record(
    rule: LearnedRule, nuisance: NuisanceEstimates, fold_estimates: Sequence[float]
) -> LearnedRuleRecord:
    """Build the record of a learned-rule fit from its nuisances and its fold estimates.

    Parameters
    ----------
    rule : LearnedRule
        The declared rule.
    nuisance : NuisanceEstimates
        The cross-fitted nuisances, whose folds and regimes the record reads.
    fold_estimates : sequence of float
        The fold plug-in values, in the order of ``nuisance.folds``.

    Returns
    -------
    LearnedRuleRecord
        The per-fold summaries of the fit.
    """
    blip = _blip(nuisance)
    treat = _rule_treat(nuisance, rule)
    tests = [np.asarray(test) for _, test in nuisance.folds]
    return LearnedRuleRecord(
        name=rule.name,
        fold_sizes=tuple(int(test.size) for test in tests),
        fold_estimates=tuple(float(value) for value in fold_estimates),
        treated_shares=tuple(float(np.mean(treat[test])) for test in tests),
        blip_quantiles=tuple(
            tuple(float(value) for value in np.quantile(blip[test], BLIP_QUANTILE_LEVELS))
            for test in tests
        ),
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

    Rows 1 and 2 of the learned-rule refusal table in
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
    """Rows 3 to 9 of the learned-rule refusal table: the data declarations no setting repairs.

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
    if data.weights_name is not None or data.is_weighted:
        return _gap(
            "a learned rule with observation weights (weights=) is refused. Van der Laan "
            "and Luedtke (2015) treat unweighted iid rows, and the chain rule with a known "
            f"weight is not implemented ({_X11.format(part='k')}).",
            f"{_FIXED_RULE} takes weights= for a fixed rule.",
            "X11",
        )
    if data.cluster is not None:
        return _gap(
            "a learned rule with clusters (id=) is refused. Van der Laan and Luedtke (2015) "
            "treat iid rows, and a cluster as the unit with whole-cluster folds is not "
            f"implemented ({_X11.format(part='l')}).",
            f"{_FIXED_RULE} takes id= for a fixed rule.",
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


def learned_rule_scheme_refusal(estimator: Any) -> Refusal | None:
    """Return the refusal of a learned-rule fit whose remedy is a configuration, or ``None``.

    Rows 10 to 14 of the learned-rule refusal table in
    ``docs/technical-reference/point-treatment-tmle.md``.  Each one names the configuration
    that fits, :data:`LEARNED_RULE_CONFIGURATION`.

    Parameters
    ----------
    estimator : TMLE
        The estimator, with ``learned_rule`` set.

    Returns
    -------
    Refusal or None
        The refusal, or ``None``. Rows 10 to 13 are ``IMPLEMENTATION_GAP`` records, and
        row 14 is an ``INFERENCE_GAP`` record.
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
    if estimator.targeting_scheme != "pooled":
        return _gap(
            "the learned-rule value needs one pooled fluctuation (targeting_scheme="
            "'pooled'). A fluctuation fitted inside each validation fold is published for "
            f"this target and is not implemented ({_X11.format(part='h')}).",
            remedy,
            "X11",
        )
    if estimator.repeats != 1:
        return _gap(
            "the learned-rule value needs one split (repeats=1). Each split defines a "
            "different target. The mean over a fixed number of splits targets the mean of "
            f"those targets and is not implemented ({_X11.format(part='i')}), and the median "
            "has no defined target.",
            remedy,
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
    7       ``weights=``                                         X11 (k)
    8       ``id=``                                              X11 (l)
    9       ``strata=``                                          X11 (m)
    10      ``cross_fit=False``, or one fold                     X11 (a)
    11      ``cv_evaluation=False``                              X11 (e)
    12      ``targeting_scheme="fold"``                          X11 (h)
    13      ``repeats`` above 1                                  X11 (i)
    14      ``n_bootstrap`` above 0                              F27
    ======  ===================================================  ===================

    ``TMLE`` runs it first in its preflight, before any learner.
    ``CausalStudy.identify`` runs it with no estimator, so rows 3 to 9 refuse there.

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
        refusal = learned_rule_scheme_refusal(estimator)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
