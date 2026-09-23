"""The inference statuses, and the one table of what each non-inferential status says.

An estimate declares one :data:`InferenceStatus`. ``"influence_curve"`` is the ordinary
case: the package supplies ``std_error``, ``ci`` and ``pvalue``. Every other status names a
reason the package supplies none, and :data:`NON_INFERENTIAL` holds that reason with every
text a report prints for it. A consumer reads the table and never branches on a status
name, so a new status is one new entry here and one new member of the Literal.

A leaf module. It imports the standard library only, as :mod:`cleverly._typing` does, so
:mod:`cleverly.exceptions`, which the rest of the package imports, can read the table at
module level without an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

__all__ = [
    "FEW_CLUSTER_THRESHOLD",
    "NON_INFERENTIAL",
    "NO_SIMULTANEOUS_BANDS",
    "InferenceStatus",
    "StatusRecord",
    "precedent_status",
    "status_record",
    "supplies_inference",
]

#: Whether the package supplies inference for an estimate, or only a point estimate and a
#: named diagnostic. ``"influence_curve"`` supplies inference. Each other member is a key
#: of :data:`NON_INFERENTIAL`, and ``tests/unit/test_inference_status_registry.py`` pins
#: that the two lists agree.
InferenceStatus = Literal[
    "influence_curve",
    "working_mechanism_plugin",
    "generated_design_plugin",
    "estimated_weight_plugin",
    "unequal_cluster_plugin",
    "few_cluster_plugin",
]

# A clustered fit with fewer clusters than this, in total or in one reported baseline
# stratum, takes the ``"few_cluster_plugin"`` status.
# Nugent, Marquez, Charlebois, Abbott and Balzer (2024), Biostatistics 25(3):599-616,
# Section 2.2, last paragraph: "In CRTs with fewer than 40 clusters randomized (N < 40),
# we recommend using the Student's t distribution with N - 2 degrees of freedom", citing
# Hayes and Moulton (2009). That is the only explicit threshold in a source read here.
# Benitez et al. (2023), Stat Med 42(19):3443-3466, Section 3.1.2, paragraph on inference,
# and Section 3.2.1, last paragraph, recommend t with J - 2 degrees of freedom at every
# cluster count. Neither paper compares
# the normal reference with t. The package keeps its normal reference and withholds the
# interval below this count; roadmap row RM20 records the decision and F22 the reopen route.
#: The cluster count below which a clustered fit reports no interval (RM20).
FEW_CLUSTER_THRESHOLD: Final[int] = 40

#: The line a result summary prints when a fit that supplies no inference builds no
#: simultaneous band. ``TMLEResult.summary`` and ``LongitudinalResult.summary`` both
#: print it, and both estimators skip the default band on such a fit.
NO_SIMULTANEOUS_BANDS: Final[str] = (
    "no simultaneous bands: a band is a joint confidence statement, and this fit reports none."
)


@dataclass(frozen=True)
class StatusRecord:
    """Every text the package prints for one non-inferential status.

    Parameters
    ----------
    reason : str
        The sentences that say what the fit reports, what result is missing, and which
        roadmap item reopens it. They open with a capital letter. Every refusal ends with
        them, and ``summary()`` prints them under the table.
    assessment_note : str
        The clause the nuisance report and the assessment's nuisance-model item add.
    summary_label : str
        The header of the spread column in ``TMLEResult.summary()`` and
        ``LongitudinalResult.summary()``, and the label in ``ParameterEstimate.__repr__``.
    bootstrap_note : str
        The parenthesis ``summary()`` prints beside a bootstrap percentile range.
    diagnostic_noun : str
        The noun a coverage-study report uses for the spread it measured.
    reopened_by : str
        The roadmap id whose anchor in ``docs/roadmap.md`` holds the reopen route.
    """

    reason: str
    assessment_note: str
    summary_label: str
    bootstrap_note: str
    diagnostic_noun: str
    reopened_by: str

    def summary_note(self) -> str:
        """The paragraph a result summary prints under its table of diagnostics.

        The refusal's own reason rather than a paraphrase of it, so the summary and every
        raise say one thing. Only the pointer to the table's spread column is the
        summary's own. ``TMLEResult.summary`` and ``LongitudinalResult.summary`` both
        print it.

        Returns
        -------
        str
            :attr:`reason`, then a sentence that names :attr:`summary_label` as that
            diagnostic.
        """
        return (
            self.reason + f' The "{self.summary_label}" column above is that diagnostic, and it '
            "is not a standard error for this estimate."
        )


#: One record per non-inferential status. The insertion order is the precedence: when a
#: fit meets more than one status, it takes the first one here, as the ordered refusals
#: in ``TMLE._resolve_estimands_for_data`` give "a fit that breaks several rules" the
#: first. :func:`precedent_status` applies that order. The text of each record is
#: asserted by importing it, so a report and a raise cannot drift apart.
NON_INFERENTIAL: Mapping[str, StatusRecord] = MappingProxyType(
    {
        "working_mechanism_plugin": StatusRecord(
            reason=(
                "The greedy, ordered and discrete collaborative paths report no confidence "
                "interval, no p-value and no standard error. The reported curve is the "
                "ordinary efficient influence curve at the candidate the search stopped at, "
                "and no result shows it is this estimator's influence curve when that "
                "working mechanism is not consistent for the treatment law. The point "
                "estimate and the selection path stand. The plug-in standard error of that "
                "curve remains as a diagnostic under plugin_std_error and plugin_interval. "
                "F18 in docs/roadmap.md reopens this when it supplies the estimator's "
                "influence curve."
            ),
            assessment_note=(
                "the reported curve is a working-mechanism diagnostic: no confidence "
                "interval or p-value is available for this path, and F18 in the roadmap is "
                "the condition that reopens it"
            ),
            summary_label="working-mechanism se",
            bootstrap_note=(
                "a diagnostic; the refit bootstrap reruns the selection, and no result "
                "validates its coverage for this path"
            ),
            diagnostic_noun="working-mechanism plug-in diagnostic",
            reopened_by="F18",
        ),
        "generated_design_plugin": StatusRecord(
            reason=(
                "The outcome-adaptive collaborative path, strategy='oat', reports no "
                "confidence interval, no p-value and no standard error. It fits one treatment "
                "mechanism on the estimated outcome predictions of every arm, and it targets "
                "every arm mean jointly. Benkeser, Cai and van der Laan (2020), Theorem 1, "
                "prove the ordinary curve for one binary treatment-specific mean with one "
                "scalar design. No result covers this joint construction. Missing outcomes, "
                "weights and repeated splits are also outside that theorem. The point "
                "estimate stands. The plug-in standard error of that curve remains as a "
                "diagnostic under plugin_std_error and plugin_interval. F19 in "
                "docs/roadmap.md reopens this when it supplies that result."
            ),
            assessment_note=(
                "the reported curve is a generated-design diagnostic: no confidence interval "
                "or p-value is available for this path, and F19 in the roadmap is the "
                "condition that reopens it"
            ),
            summary_label="generated-design se",
            bootstrap_note=(
                "a diagnostic; the refit bootstrap reruns the generated design, and no result "
                "validates its coverage for this path"
            ),
            diagnostic_noun="generated-design plug-in diagnostic",
            reopened_by="F19",
        ),
        "estimated_weight_plugin": StatusRecord(
            reason=(
                "A DR-TMLE fit with a guard and weights declared estimated "
                "(weights_estimated=True) reports no confidence interval, no p-value and no "
                "standard error. The argument that an interval conditions on the weights "
                "concerns the efficient influence curve. No result read here gives the "
                "reduced-dimension regressions of an estimated weight, or the contribution "
                "of the weight estimate to the curve. The point estimate stands. The plug-in "
                "standard error of that curve remains as a diagnostic under plugin_std_error "
                "and plugin_interval. F5 in docs/roadmap.md reopens this when a paper "
                "supplies that contribution. guard=() fits the ordinary TMLE, whose interval "
                "conditions on the weights."
            ),
            assessment_note=(
                "the reported curve is a fixed-weight diagnostic: no confidence interval or "
                "p-value is available for this fit, and F5 in the roadmap is the condition "
                "that reopens it"
            ),
            summary_label="fixed-weight se",
            bootstrap_note=(
                "a diagnostic; the bootstrap resamples rows and does not estimate the weights "
                "again, and no result validates its coverage for this fit"
            ),
            diagnostic_noun="fixed-weight plug-in diagnostic",
            reopened_by="F5",
        ),
        "unequal_cluster_plugin": StatusRecord(
            reason=(
                "A cross-fitted clustered fit reports no confidence interval, no p-value and "
                "no standard error when its clusters hold different numbers of rows, or "
                "different weight mass on a weighted fit, overall or within a reported "
                "baseline stratum. The point estimator remains row weighted. The package's "
                "grouped cross-fitting argument and registered study cover equal cluster "
                "sizes and weight masses only (docs/technical-reference/cv-tmle.md, grouped "
                "folds). No result here validates its cross-fitted interval at unequal "
                "sizes. The point estimate stands. The plug-in standard error "
                "of the reported curve remains as a diagnostic under plugin_std_error and "
                "plugin_interval. The same clusters fitted in sample keep the interval when "
                f"there are at least {FEW_CLUSTER_THRESHOLD} of them in the fit and in each "
                "reported stratum with positive weight mass. Benitez et al. (2023), "
                "Section 3.2.1, give cluster-sum aggregation for a row-weighted estimand "
                "but no result for this cross-fitted construction. F22 in "
                "docs/roadmap.md reopens this when a result covers unequal cluster sizes."
            ),
            assessment_note=(
                "the reported curve is an unequal-cluster diagnostic: no confidence interval "
                "or p-value is available for this fit, and F22 in the roadmap is the "
                "condition that reopens it"
            ),
            summary_label="cluster-robust plug-in se",
            bootstrap_note=(
                "a diagnostic; no result validates the bootstrap coverage of a cross-fitted "
                "fit at unequal cluster sizes"
            ),
            diagnostic_noun="unequal-cluster plug-in diagnostic",
            reopened_by="F22",
        ),
        "few_cluster_plugin": StatusRecord(
            reason=(
                "A clustered fit reports no confidence interval, no p-value and no standard "
                f"error when it has fewer than {FEW_CLUSTER_THRESHOLD} clusters with "
                "positive weight mass, or when one baseline stratum it reports has fewer. "
                "The package uses a "
                "normal reference distribution. Nugent, Marquez, Charlebois, Abbott and "
                "Balzer (2024), Section 2.2, recommend a Student t reference with J - 2 "
                f"degrees of freedom below {FEW_CLUSTER_THRESHOLD} clusters. Benitez et al. "
                "(2023) recommend it at every cluster count, in Section 3.1.2, paragraph on "
                "inference, and Section 3.2.1, last paragraph. No registered study covers a "
                "clustered fit with few clusters. The point estimate stands. The plug-in "
                "standard error of the reported curve remains as a diagnostic under "
                "plugin_std_error and plugin_interval. F22 in docs/roadmap.md reopens this "
                "with a t reference and a registered study at few clusters."
            ),
            assessment_note=(
                "the reported curve is a few-cluster diagnostic: no confidence interval or "
                "p-value is available for this fit, and F22 in the roadmap is the condition "
                "that reopens it"
            ),
            summary_label="normal-reference se",
            bootstrap_note=(
                "a diagnostic; no result validates the bootstrap coverage of a clustered fit "
                "with few clusters"
            ),
            diagnostic_noun="few-cluster plug-in diagnostic",
            reopened_by="F22",
        ),
    }
)


def supplies_inference(status: str) -> bool:
    """Whether the package supplies inference at an inference status.

    The one comparison against ``"influence_curve"``. An estimate, a fit, a frame and a
    report each ask it of the status they hold. Any other string is non-inferential, so
    a status with no record fails closed at :func:`status_record` rather than reporting
    an interval.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus`.

    Returns
    -------
    bool
        ``True`` for ``"influence_curve"``, which is when ``std_error``, ``ci`` and
        ``pvalue`` answer. ``False`` for every diagnostic status.
    """
    return status == "influence_curve"


def status_record(status: str) -> StatusRecord:
    """The record of a non-inferential status.

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
    return NON_INFERENTIAL[status]


def precedent_status(statuses: Iterable[str]) -> InferenceStatus:
    """The one status a fit takes when several apply.

    The first of ``statuses`` in the order of :data:`NON_INFERENTIAL`, or
    ``"influence_curve"`` when none of them is non-inferential. A hook that finds more
    than one reason to withhold inference passes them all here, so the precedence lives
    in the table and not in the order a subclass happens to call its parent.

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
        When an entry is neither ``"influence_curve"`` nor a key of
        :data:`NON_INFERENTIAL`.
    """
    applied = {status for status in statuses if not supplies_inference(status)}
    for status in applied:
        status_record(status)
    for status in NON_INFERENTIAL:
        if status in applied:
            return cast(InferenceStatus, status)
    return "influence_curve"
