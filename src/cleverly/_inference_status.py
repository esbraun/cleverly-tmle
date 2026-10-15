"""The inference statuses, and the one table of what each labelled status says.

An estimate declares one :data:`InferenceStatus`. ``"influence_curve"`` is the ordinary
case: the package supplies ``std_error``, ``ci`` and ``pvalue``. Every other status is
labelled, and :data:`STATUS_RECORDS` holds its reason with every text a report prints for it.
A record has an ``interval`` of ``"shown"`` or ``"withheld"``. A shown status supplies
``std_error``, ``ci`` and ``pvalue`` from the plug-in curve, and the estimate carries the
experimental label with the record's ``shown_reason``. A withheld status supplies none of the
three and names the reason. A consumer reads the table and never branches on a status name, so
a new status is one new entry here and one new member of the Literal. The one exception is the
estimator that stamps a parameter status (:data:`PARAMETER_STATUSES`), which names it through
:data:`CONSTANT_NODE_STATUS`.

A leaf module. It imports the standard library and the leaf :mod:`cleverly._refusal` only, so
:mod:`cleverly.exceptions`, which the rest of the package imports, can read the table at
module level without an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from ._refusal import NO_ALTERNATIVE, RefusalReason

__all__ = [
    "ADAPTIVE_BASELINE_STATUS",
    "CONSTANT_NODE_STATUS",
    "FEW_CLUSTER_THRESHOLD",
    "HELD_OUT_SCALE",
    "MINIMUM_INTERVAL_CLUSTERS",
    "MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS",
    "MINIMUM_T_REFERENCE_CLUSTERS",
    "NO_SIMULTANEOUS_BANDS",
    "NO_T_REFERENCE_BANDS",
    "PARAMETER_STATUSES",
    "STACK_FAILURE_STATUS",
    "STATUS_RECORDS",
    "T_REFERENCE_NOTE",
    "InferenceStatus",
    "Interval",
    "StatusRecord",
    "is_labelled",
    "precedent_status",
    "status_record",
    "supplies_inference",
]

#: Whether the package supplies an interval for an estimate, and what label it carries.
#: ``"influence_curve"`` is the ordinary case. Each other member is a key of
#: :data:`STATUS_RECORDS`, and ``tests/unit/test_inference_status_registry.py`` pins that the
#: two lists agree.
InferenceStatus = Literal[
    "influence_curve",
    "working_mechanism_plugin",
    "generated_design_plugin",
    "estimated_weight_plugin",
    "adaptive_baseline_plugin",
    "no_cluster_variance_plugin",
    "few_cluster_plugin",
    "constant_node_plugin",
    "stack_failure_plugin",
]

# Below this many clusters with positive weight mass, in the fit or in the stratum an
# estimate reads, the estimate takes a Student t reference with J - 2 degrees of freedom.
# Nugent, Marquez, Charlebois, Abbott and Balzer (2024), Biostatistics 25(3):599-616,
# Section 2.2, last paragraph: "In CRTs with fewer than 40 clusters randomized (N < 40),
# we recommend using the Student's t distribution with N - 2 degrees of freedom", citing
# Hayes and Moulton (2009). That is the only explicit threshold in a source read here.
# Benitez et al. (2023), Stat Med 42(19):3443-3466, Section 3.1.2, paragraph on inference,
# and Section 3.2.1, last paragraph, recommend t with J - 2 degrees of freedom at every
# cluster count. At this count and above, the normal reference stays.
#: The cluster count below which a clustered estimate takes a Student t reference.
FEW_CLUSTER_THRESHOLD: Final[int] = 40

#: The fewest clusters with positive weight mass for which a clustered fit reports an
#: interval without the few-cluster label. It is the smallest count a registered study measures:
#: ``clustered-few-cluster-tmle`` starts at 10 clusters, because at 4 a share of draws admits no
#: fit at all. From :data:`MINIMUM_T_REFERENCE_CLUSTERS` to one below it the fit takes
#: ``"few_cluster_plugin"``, which shows a labelled interval on ``t(J - 2)``. X54 in
#: ``docs/roadmap.md`` holds the evidence for 4 to 9 clusters.
MINIMUM_INTERVAL_CLUSTERS: Final[int] = 10

#: The fewest clusters with positive weight mass for which a Student t reference has a positive
#: degree of freedom, ``J - 2 >= 1``. A fit that reads fewer takes ``"no_cluster_variance_plugin"``
#: and reports its point estimate only.
MINIMUM_T_REFERENCE_CLUSTERS: Final[int] = 3

#: The same floor for a longitudinal fit. The registered few-cluster study measures in-sample
#: ``LTMLE`` from 20 clusters: at 10, about 1 draw in 4,000 admits no LTMLE fit, so those cells
#: are not in its grid. Below this count an ``LTMLE`` fit takes ``"few_cluster_plugin"``.
MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS: Final[int] = 20

#: The paragraph a result summary prints under a table whose estimates use a t reference.
#: ``{clusters}`` is the positive-mass cluster count of the fit.
T_REFERENCE_NOTE: Final[str] = (
    "An estimate that reads fewer than "
    f"{FEW_CLUSTER_THRESHOLD} clusters with positive weight mass reports its interval and "
    "p-value on a Student t reference with J - 2 degrees of freedom, where J is the count it "
    "reads (Nugent et al. (2024), Section 2.2); a fold-evaluated estimate over V folds takes "
    "min(J - 2, J - V). The fit reads {clusters} such clusters, and the fewest in one "
    "reported stratum is {fewest}. The df column gives each estimate's degrees of freedom, "
    "and 'normal' marks the normal reference. The registered few-cluster evidence uses "
    "parametric nuisance learners; Wang et al. (2024), Remark 3, caution against complex "
    "learners at about 20 clusters."
)

#: Where ``q_bounds=None`` takes the scale of a continuous outcome from. Both outcome-scale
#: refusals of :class:`~cleverly.TMLE` say it, so the two name one fact in one wording.
HELD_OUT_SCALE: Final[str] = "from every observed outcome, held-out rows included"

#: The line a result summary prints when a fit that supplies no inference builds no
#: simultaneous band. ``TMLEResult.summary`` and ``LongitudinalResult.summary`` both
#: print it, and both estimators skip the default band on such a fit.
NO_SIMULTANEOUS_BANDS: Final[str] = (
    "no simultaneous bands: a band is a joint confidence statement, and this fit reports none."
)

#: The line a result summary prints when a fit skips its default band because an estimate
#: carries a Student t reference.
NO_T_REFERENCE_BANDS: Final[str] = (
    "no simultaneous bands: an estimate of this fit reads fewer than "
    f"{FEW_CLUSTER_THRESHOLD} clusters with positive weight mass and carries a t reference, "
    "and no source gives a t-calibrated band, so the fit reports pointwise intervals only."
)


#: Whether a labelled status shows an interval (``"shown"``) or withholds it (``"withheld"``).
Interval = Literal["shown", "withheld"]


@dataclass(frozen=True)
class StatusRecord:
    """Every text the package prints for one labelled status.

    Parameters
    ----------
    reason : str
        The sentences that say what the fit reports, what result is missing, and which
        roadmap item reopens it. They open with a capital letter. Every refusal of a withheld
        status ends with them, and ``summary()`` prints them under the table.
    assessment_note : str
        The clause the nuisance report and the assessment's nuisance-model item add.
    summary_label : str
        The header of the spread column in ``TMLEResult.summary()`` and
        ``LongitudinalResult.summary()``, and the label in ``ParameterEstimate.__repr__``,
        for a withheld status.
    bootstrap_note : str
        The parenthesis ``summary()`` prints beside a bootstrap percentile range.
    diagnostic_noun : str
        The noun a coverage-study report uses for the spread it measured.
    reopened_by : str
        The roadmap id whose anchor in ``docs/roadmap.md`` holds the reopen route.
    alternative : str
        The sentence a refusal of an inferential operation at this status ends with: the
        estimator that supplies an interval for the same question, or
        :data:`~cleverly._refusal.NO_ALTERNATIVE`.
    code : RefusalReason
        The reason code of that refusal. A status the theory leaves without inference takes
        :attr:`~cleverly._refusal.RefusalReason.INFERENCE_GAP`. A status that one sample's
        numerical failure sets takes
        :attr:`~cleverly._refusal.RefusalReason.INVALID_DECLARATION`.
    interval : {"shown", "withheld"}, default "withheld"
        ``"shown"`` supplies ``std_error``, ``ci`` and ``pvalue`` from the plug-in curve under
        the experimental label. ``"withheld"`` refuses all three.
    shown_reason : str or None, default None
        For a shown status, the sentences the experimental label adds to say what the interval
        does not cover. Required when ``interval`` is ``"shown"``.

    Raises
    ------
    ValueError
        When a shown record has no ``shown_reason``, or a withheld record has one.
    """

    reason: str
    assessment_note: str
    summary_label: str
    bootstrap_note: str
    diagnostic_noun: str
    reopened_by: str
    alternative: str
    code: RefusalReason = RefusalReason.INFERENCE_GAP
    interval: Interval = "withheld"
    shown_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.interval == "shown") != (self.shown_reason is not None):
            raise ValueError("a shown status states a shown_reason, and a withheld one states none")

    def summary_note(self) -> str:
        """The paragraph a result summary prints under its table of diagnostics.

        The refusal's own reason rather than a paraphrase of it, so the summary and every
        raise say one thing. Only the pointer to the table's spread column is the
        summary's own. ``TMLEResult.summary`` and ``LongitudinalResult.summary`` both
        print it for a withheld status. A shown status prints :attr:`shown_reason` as its
        label note instead.

        Returns
        -------
        str
            :attr:`reason`, then :attr:`alternative` when one exists, then a sentence that
            names :attr:`summary_label` as that diagnostic.
        """
        alternative = "" if self.alternative == NO_ALTERNATIVE else f" {self.alternative}"
        return (
            self.reason
            + alternative
            + f' The "{self.summary_label}" column above is that diagnostic, and it '
            "is not a standard error for this estimate."
        )


#: One record per labelled status. The insertion order is the precedence inside a tier: when a
#: fit meets more than one withheld status, it takes the first one here. When it meets none,
#: it takes the first shown one. :func:`precedent_status` applies that rule. The text of each
#: record is asserted by importing it, so a report and a raise cannot drift apart.
STATUS_RECORDS: Mapping[str, StatusRecord] = MappingProxyType(
    {
        "working_mechanism_plugin": StatusRecord(
            reason=(
                "The greedy, ordered, discrete, lasso and learners collaborative paths report "
                "no confidence interval, no p-value and no standard error. The reported curve "
                "is the ordinary efficient influence curve at the candidate the search "
                "stopped at, or at the final-step candidate, and no result shows it is this "
                "estimator's influence curve when that "
                "working mechanism is not consistent for the treatment law. The point "
                "estimate and the selection path stand. The plug-in standard error of that "
                "curve remains as a diagnostic under plugin_std_error and plugin_interval. "
                "F18 in docs/roadmap.md reopens this when it supplies the estimator's "
                "influence curve. A discrete fit that declares one candidate, equal to the "
                "full adjustment set, selects nothing and takes the ordinary TMLE status. A "
                "learners fit that declares no candidate does the same."
            ),
            assessment_note=(
                "the reported curve is a working-mechanism diagnostic: no confidence "
                "interval or p-value is available for this path, and F18 in the roadmap is "
                "the condition that reopens it"
            ),
            summary_label="working-mechanism se",
            bootstrap_note=(
                "a diagnostic; the full-refit bootstrap reruns the selection, and no result "
                "validates its coverage for this path"
            ),
            diagnostic_noun="working-mechanism plug-in diagnostic",
            reopened_by="F18",
            alternative="For an interval, fit TMLE or DRTMLE on the same design.",
        ),
        "generated_design_plugin": StatusRecord(
            reason=(
                "This outcome-adaptive collaborative fit, strategy='oat', reports no confidence "
                "interval, no p-value and no standard error. The shared design, "
                "oat_design='shared', fits one treatment mechanism on the outcome predictions of "
                "every arm, and no result covers it. The per-arm design, oat_design='per_arm', "
                "applies the construction of Benkeser, Cai and van der Laan (2020), Theorem 1, "
                "to each arm, inside each baseline stratum, and to the composite indicator of a "
                "missing outcome or treatment. That theorem needs its condition (v) (preprint "
                "Appendix F, the remainder R24), which fails for an estimated outcome "
                "regression, so the reported curve omits a first-order generated-design term. "
                "The registered study ctmle-oat-per-arm measured standard errors 8 to 24 percent "
                "too small in its calibration, primary coverage, estimated-design and band "
                "cells, flat in n, while its oracle-design cells were calibrated. The point "
                "estimate stands. The plug-in standard error of the ordinary curve remains as a "
                "diagnostic under plugin_std_error and plugin_interval. F19 in docs/roadmap.md "
                "tracks the generated-design term of both designs."
            ),
            assessment_note=(
                "the reported curve is a generated-design diagnostic: no confidence interval "
                "or p-value is available for this path, and F19 in the roadmap is the "
                "condition that reopens it"
            ),
            summary_label="generated-design se",
            bootstrap_note=(
                "a diagnostic; the full-refit bootstrap reruns the generated design, and no "
                "result validates its coverage for this path"
            ),
            diagnostic_noun="generated-design plug-in diagnostic",
            reopened_by="F19",
            alternative=(
                "For an interval, fit TMLE, or fit the in-sample per-arm design with "
                "ParametricOutcome and ParametricMechanism (X58). "
                "per_arm_parametric_support(estimator, data) in cleverly.estimators.ctmle "
                "names the first requirement a fit does not meet."
            ),
        ),
        "estimated_weight_plugin": StatusRecord(
            reason=(
                "A DR-TMLE fit with a guard and weights declared estimated "
                "(weights_estimated=True) reports its interval, p-value and standard error "
                "from the plug-in curve with the weights held fixed, under the experimental "
                "label. The DR-TMLE fit adds reduced-dimension regressions of the weighted law, "
                "and no result read here gives the contribution of the weight estimate to its "
                "curve. X50 in docs/roadmap.md reopens this for DRTMLE: the conditional reading "
                "of the weights does not use efficiency, so it covers the fixed-weight DR-TMLE "
                "curve. A per-arm outcome-adaptive C-TMLE fit (strategy='oat', "
                "oat_design='per_arm') with estimated weights takes the generated_design_plugin "
                "status, because per_arm_parametric_support refuses observation weights, and "
                "F19 in docs/roadmap.md holds its interval. A sequential g-computation or IPW "
                "fit with estimated weights takes this status too, because its stacked "
                "estimating-equation curve conditions on the weights."
            ),
            assessment_note=(
                "the reported interval conditions on the weights and is experimental: no "
                "result gives the weight-estimation term, and X50 in the roadmap is the "
                "condition that reopens it for DRTMLE"
            ),
            summary_label="fixed-weight se",
            bootstrap_note=(
                "a diagnostic; the bootstrap resamples rows and does not estimate the weights "
                "again, and no result validates its coverage for this fit"
            ),
            diagnostic_noun="fixed-weight plug-in interval",
            reopened_by="X50",
            alternative=(
                "The ordinary TMLE (DRTMLE with guard=(), or TMLE) keeps an interval that "
                "conditions on the weights."
            ),
            interval="shown",
            shown_reason=(
                "The interval conditions on the weights, and no result gives the contribution "
                "of the weight estimate to the curve. X50 in docs/roadmap.md holds the result "
                "that reopens it."
            ),
        ),
        "adaptive_baseline_plugin": StatusRecord(
            reason=(
                "A sequential g-computation or IPW fit reports no confidence interval, no "
                "p-value and no standard error when a regression it reads is fitted by a "
                "learner other than cleverly.learners.GLM. That covers a data-adaptive learner, "
                "and a parametric scikit-learn model whose score the package does not stack. "
                "The estimator solves no efficient influence-curve equation, so its "
                "first-order bias is of the order of the learner's error, and no result read "
                "here makes that bias negligible over several time points. The point estimate "
                "stands. The diagnostic curve remains under plugin_std_error and "
                "plugin_interval: for g-computation it is the efficient influence curve at the "
                "untargeted fits, and for IPW it is the fixed-mechanism curve. F35 in "
                "docs/roadmap.md reopens this when a result supplies the asymptotic linearity "
                "of these estimators with data-adaptive nuisances."
            ),
            assessment_note=(
                "the reported curve is a baseline diagnostic: no confidence interval or "
                "p-value is available for a g-computation or IPW fit with a data-adaptive "
                "learner, and F35 in the roadmap is the condition that reopens it"
            ),
            summary_label="baseline diagnostic se",
            bootstrap_note=(
                "a diagnostic; no result validates the bootstrap of a g-computation or IPW "
                "fit with a data-adaptive learner"
            ),
            diagnostic_noun="baseline plug-in diagnostic",
            reopened_by="F35",
            alternative="For an interval, pass GLM models, or fit LTMLE or SDR.",
        ),
        "no_cluster_variance_plugin": StatusRecord(
            reason=(
                "A clustered fit reports no confidence interval, no p-value and no standard "
                f"error when it reads fewer than {MINIMUM_T_REFERENCE_CLUSTERS} clusters with "
                "positive weight mass, in the fit or in one reported baseline stratum. A Student "
                "t reference with J - 2 degrees of freedom needs at least one degree of "
                "freedom, and at 1 or 2 clusters no independent-cluster argument applies. The "
                "point estimate stands. The plug-in standard error of the reported curve "
                "remains as a diagnostic under plugin_std_error and plugin_interval. F28 in "
                "docs/roadmap.md holds fits below 3 clusters."
            ),
            assessment_note=(
                "the reported curve is a no-cluster-variance diagnostic: no confidence "
                "interval or p-value is available for a fit with fewer than "
                f"{MINIMUM_T_REFERENCE_CLUSTERS} clusters, and F28 in the roadmap is the "
                "condition that reopens it"
            ),
            summary_label="no-df se",
            bootstrap_note=(
                "a diagnostic; no result validates the bootstrap of a clustered fit with fewer "
                f"than {MINIMUM_T_REFERENCE_CLUSTERS} clusters"
            ),
            diagnostic_noun="no-cluster-variance plug-in diagnostic",
            reopened_by="F28",
            alternative=NO_ALTERNATIVE,
        ),
        "few_cluster_plugin": StatusRecord(
            reason=(
                "A clustered fit that reads from "
                f"{MINIMUM_T_REFERENCE_CLUSTERS} clusters with positive weight mass up to one "
                f"below {MINIMUM_INTERVAL_CLUSTERS} (one below "
                f"{MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS} on a longitudinal fit), in the fit "
                "or in one reported baseline stratum, reports its interval and p-value on a "
                "Student t reference with J - 2 degrees of freedom, as Nugent et al. (2024), "
                "Section 2.2, last paragraph, recommend. The registered few-cluster study "
                f"measures point-treatment fits from {MINIMUM_INTERVAL_CLUSTERS} clusters and "
                f"longitudinal fits from {MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS}, and no "
                "registered study measures an interval below those counts. The estimate "
                "carries the experimental label. X54 in docs/roadmap.md holds the registered "
                "measurements that reopen counts from "
                f"{MINIMUM_T_REFERENCE_CLUSTERS} clusters up to the floor."
            ),
            assessment_note=(
                "the reported interval is experimental: no registered study measures it for a "
                f"fit with fewer than {MINIMUM_INTERVAL_CLUSTERS} clusters "
                f"({MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS} for a longitudinal fit), and X54 in "
                "the roadmap is the condition that supports it"
            ),
            summary_label="normal-reference se",
            bootstrap_note=(
                "a diagnostic; no result validates the bootstrap coverage of a clustered fit "
                "with few clusters"
            ),
            diagnostic_noun="few-cluster plug-in interval",
            reopened_by="X54",
            alternative=NO_ALTERNATIVE,
            interval="shown",
            shown_reason=(
                "The interval uses a Student t reference with J - 2 degrees of freedom, and no "
                f"registered study measures a clustered interval below {MINIMUM_INTERVAL_CLUSTERS} "
                f"clusters ({MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS} for a longitudinal fit). "
                "X54 in docs/roadmap.md holds the measurements."
            ),
        ),
        "constant_node_plugin": StatusRecord(
            reason=(
                "A longitudinal parameter reports no confidence interval, no p-value and no "
                "standard error when a node regression of its recursion read a constant "
                "target, such as a grid node at which no follower of the regimen had the "
                "event, or a repeated-outcome node at which every follower held one outcome "
                "value. A hazard fit applies the same rule to each node, level and cause. "
                "That regression is its maximum-likelihood value, 0 or 1, so the "
                "point estimate stands. The node's term of the influence curve is then "
                "identically zero, although the true outcome variance there can be "
                "positive, so the plug-in standard error leaves that node's variance out. "
                "A contrast or a restricted mean that reads such a parameter takes this status "
                "too, and the "
                "simultaneous band leaves it out. The plug-in standard error remains as a "
                "diagnostic under "
                "plugin_std_error and plugin_interval. X54 in docs/roadmap.md reopens this "
                "at a fixed law with a declared node-variance construction and a registered "
                "study, and F31 holds the rare-event regime in which the node hazard "
                "shrinks with n."
            ),
            assessment_note=(
                "the reported curve is a constant-node diagnostic: no confidence interval or "
                "p-value is available for a parameter whose recursion read a constant node "
                "regression, and X54 in the roadmap is the condition that reopens it"
            ),
            summary_label="constant-node se",
            bootstrap_note=(
                "a diagnostic; a resample can hold an event at the node that the sample "
                "lacks, and no result validates the bootstrap coverage at a boundary node"
            ),
            diagnostic_noun="constant-node plug-in diagnostic",
            reopened_by="X54",
            alternative=NO_ALTERNATIVE,
        ),
        "stack_failure_plugin": StatusRecord(
            reason=(
                "A per-arm outcome-adaptive C-TMLE fit with parametric learners "
                "(ParametricOutcome and ParametricMechanism) reports no confidence interval, "
                "no p-value and no standard error for an estimate that reads an arm whose "
                "stacked estimating equation failed on this sample. The stack fails when the "
                "fluctuation did not converge, when a diagonal block of its Jacobian is "
                "numerically singular, or when the targeted predictions cannot be recomputed "
                "from the fluctuation coefficient because the solver path met the outer "
                "re-shrink. The fit's warning names the failed condition. The point estimate "
                "stands. The plug-in standard error of the shipped curve remains as a "
                "diagnostic under plugin_std_error and plugin_interval, and it omits the "
                "generated-design term. X58 in docs/roadmap.md describes the stack."
            ),
            assessment_note=(
                "the reported curve is a stacked-equation failure diagnostic: no confidence "
                "interval or p-value is available for an estimate whose arm stack failed on "
                "this sample, and X58 in the roadmap describes the stack"
            ),
            summary_label="stack-failure se",
            bootstrap_note=(
                "a diagnostic; no result validates the bootstrap coverage of the per-arm "
                "outcome-adaptive fit"
            ),
            diagnostic_noun="stack-failure plug-in diagnostic",
            reopened_by="X58",
            alternative=NO_ALTERNATIVE,
            code=RefusalReason.INVALID_DECLARATION,
        ),
    }
)


#: The status of a sequential g-computation or IPW fit that reads a regression fitted by a
#: learner other than :class:`~cleverly.learners.GLM`.
ADAPTIVE_BASELINE_STATUS: Final = "adaptive_baseline_plugin"

#: The status of a longitudinal parameter whose recursion read a node regression as a
#: constant 0 or 1. The estimator stamps it by name, so the name lives here once.
CONSTANT_NODE_STATUS: Final = "constant_node_plugin"

#: The statuses one parameter takes beside the status of its fit. Every other status holds
#: for the whole fit. :data:`CONSTANT_NODE_STATUS` marks a longitudinal parameter whose
#: recursion read a constant node regression, and the other parameters of that fit keep their
#: status. A derived estimate that reads such a parameter takes it too. A reader therefore
#: asks each estimate's ``supplies_inference``, and never the fit's status alone.
PARAMETER_STATUSES: Final[frozenset[str]] = frozenset(
    {CONSTANT_NODE_STATUS, "stack_failure_plugin"}
)

#: The status of an estimate of a per-arm outcome-adaptive fit whose arm stack failed on
#: one sample (:mod:`cleverly.estimators._oat_stack`). A parameter status, like
#: :data:`CONSTANT_NODE_STATUS`.
STACK_FAILURE_STATUS: Final = "stack_failure_plugin"


def supplies_inference(status: str) -> bool:
    """Whether the package supplies an interval at an inference status.

    The one test a consumer applies before it reads ``std_error``, ``ci`` or ``pvalue``. An
    estimate, a fit, a frame and a report each ask it of the status they hold. It is ``True``
    for ``"influence_curve"`` and for a labelled status whose record shows its interval. Any
    other string fails closed at :func:`status_record` rather than reporting an interval.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus`.

    Returns
    -------
    bool
        ``True`` when ``std_error``, ``ci`` and ``pvalue`` answer. ``False`` for every
        status that withholds its interval.

    Raises
    ------
    KeyError
        When ``status`` is neither ``"influence_curve"`` nor a key of :data:`STATUS_RECORDS`.
    """
    if status == "influence_curve":
        return True
    return STATUS_RECORDS[status].interval == "shown"


def is_labelled(status: str) -> bool:
    """Whether a status is a labelled status, shown or withheld.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus`.

    Returns
    -------
    bool
        ``False`` for ``"influence_curve"`` and ``True`` for every other status.
    """
    return status != "influence_curve"


def status_record(status: str) -> StatusRecord:
    """The record of a labelled status.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus` other than ``"influence_curve"``.

    Returns
    -------
    StatusRecord
        The texts the package prints for ``status``.

    Raises
    ------
    KeyError
        When ``status`` has no record, which includes ``"influence_curve"``.
    """
    return STATUS_RECORDS[status]


def precedent_status(statuses: Iterable[str]) -> InferenceStatus:
    """The one status a fit takes when several apply.

    Two tiers. The first applied status whose interval is withheld, in the order of
    :data:`STATUS_RECORDS`, wins. When none is withheld, the first applied shown status wins.
    ``"influence_curve"`` when none applies. A withheld status wins over a shown one, so a fit
    that loses an interval for one reason does not regain it for another. A hook that finds
    more than one reason passes them all here, so the precedence lives in the table and not in
    the order a subclass happens to call its parent.

    Parameters
    ----------
    statuses : iterable of str
        The statuses that apply to one fit. ``"influence_curve"`` entries are ignored.

    Returns
    -------
    str
        One of :data:`InferenceStatus`.

    Raises
    ------
    KeyError
        When an entry is neither ``"influence_curve"`` nor a key of :data:`STATUS_RECORDS`.
    """
    applied = {status for status in statuses if is_labelled(status)}
    for status in applied:
        status_record(status)
    for tier in ("withheld", "shown"):
        for status, record in STATUS_RECORDS.items():
            if status in applied and record.interval == tier:
                return cast(InferenceStatus, status)
    return "influence_curve"
