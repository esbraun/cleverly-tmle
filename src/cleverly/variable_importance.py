"""Exposure-wise variable importance from explicitly repeated target parameters.

This is orchestration, not a new influence function: candidate ``X_j`` is assigned the
treatment role in its own fit and the requested causal parameter is estimated while the
remaining declared variables are adjustment covariates.  Consequently each row states
its own intervention and adjustment set, and no shared-nuisance shortcut is implied.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np
from scipy import stats

from ._assessment_cache import AssessmentCache
from ._inference_status import FEW_CLUSTER_THRESHOLD, is_labelled, supplies_inference
from ._support import Support
from ._warnings import muted
from .data.causal_data import CausalData
from .estimators.base import TMLEResult
from .estimators.tmle import TMLE
from .exceptions import CapabilityError, DataError, Refusal, RefusalReason, refuse_inference
from .inference.influence import ParameterEstimate, support_column
from .inference.results import estimate_covariance, reported_status
from .protocol import StudyProtocol
from .targets import parameter_stem
from .utils.frames import as_frame, backend_of, emit_frame, has_nulls, is_dataframe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._planned_fits import PlannedFits
    from ._summary import Detail, Summary

__all__ = [
    "ADJUSTMENTS",
    "CORRELATION_TOLERANCE",
    "KEEP_FITS_REFUSAL",
    "MEASURED_N",
    "MEASURED_ROWS",
    "MINIMUM_MULTIPLIER_DRAWS",
    "PENDING_REASON",
    "AdjustmentKey",
    "ImportanceFacts",
    "VariableImportanceEntry",
    "VariableImportanceResult",
    "variable_importance",
    "variable_importance_plan",
    "variable_importance_support",
]

#: Why a ``keep_fits=False`` run refuses :attr:`VariableImportanceResult.fits` and every
#: diagnostic and sensitivity analysis, each of which runs on a candidate's fit.
KEEP_FITS_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "This variable-importance run used keep_fits=False, which keeps each candidate's "
    "estimates and drops its fit.",
    alternative="Run variable_importance again with keep_fits=True.",
)

Adjustment = Literal["max_t", "bh"]
ErrorRate = Literal["fwer", "fdr"]

#: The multiplicity adjustments :func:`variable_importance` offers: the step-down max-T, which
#: controls the family-wise error rate, and Benjamini-Hochberg, whose false-discovery claim is
#: experimental.
ADJUSTMENTS: Final[tuple[Adjustment, ...]] = ("max_t", "bh")

#: The fewest multiplier draws the step-down max-T accepts. At 1,000 draws the Monte Carlo
#: standard error of an adjusted p-value near 0.05 is about 0.007.
MINIMUM_MULTIPLIER_DRAWS: Final[int] = 1_000

#: The smallest eigenvalue of the rows' correlation matrix the step-down max-T accepts. Below
#: it the matrix is numerically singular: two rows are, to working precision, one test.
CORRELATION_TOLERANCE: Final[float] = 1e-8

#: The study that measures the step-down max-T adjustment of :func:`variable_importance`.
VARIABLE_IMPORTANCE_STUDY: Final[str] = "variable-importance"

#: The sample size the variable-importance study declares. A run with fewer rows reads
#: experimental: at n = 2,000 the design probe measured a family-wise error rate of 0.080 at
#: level 0.05 on its law with correlated nulls tied to signals, from per-candidate tests that
#: are liberal at that size.
MEASURED_N: Final[int] = 4_000

#: The most adjusted rows the variable-importance study measures.
MEASURED_ROWS: Final[int] = 12

#: The per-estimate support rows of the candidates the study measures: in sample, a binary
#: candidate (``T01.binary``) or a three-level one (``T08``); cross-fitted, a binary candidate
#: (``T02.binary``) only.
MEASURED_ENTRY_ROWS: Final[Mapping[bool, frozenset[str]]] = {
    False: frozenset({"T01.binary", "T08"}),
    True: frozenset({"T02.binary"}),
}

#: The fold count of the study's cross-fitted configuration.
MEASURED_FOLDS: Final[int] = 5

#: Why a configuration the study covers still reads experimental: the study is declared and
#: its run has not landed. ``None`` once the run's verdicts are committed.
PENDING_REASON: Final[str | None] = (
    f"The {VARIABLE_IMPORTANCE_STUDY} study that measures this adjustment is declared and has "
    "not run yet, so no study measures its family-wise error rate."
)

#: Why the Benjamini-Hochberg adjustment always reads experimental.
BH_REASON: Final[str] = (
    "Benjamini-Hochberg has no theorem for two-sided tests that share rows, and correlated "
    "two-sided Gaussian tests can exceed its false discovery rate (Dobriban 2026)."
)


#: The ``method`` a result records for each adjustment.
METHOD_NAMES: Final[Mapping[str, str]] = {
    "max_t": "step-down max-T",
    "bh": "Benjamini-Hochberg",
}


@dataclass(frozen=True)
class AdjustmentKey:
    """The configuration of a variable-importance run that its adjustment label reads.

    Parameters
    ----------
    engine : str
        The estimator class, such as ``"TMLE"``.
    estimand : str
        The estimand stem fitted for each candidate.
    adjust_for_other_candidates : bool
        Whether the other candidates joined each candidate's adjustment set.
    clustered : bool
        Whether the run passed ``id=``.
    weighted : bool
        Whether the run passed ``weights=``.
    missing_outcome : bool
        Whether the run passed ``delta=``.
    n : int
        The number of rows.
    cross_fitted : bool
        Whether the nuisances are fitted out of fold.
    n_folds : int
        The number of folds of a cross-fitted run.
    logistic_learners : bool
        Whether the outcome and the treatment learner are both scikit-learn
        ``LogisticRegression``.
    binary_outcome : bool
        Whether every estimate is a functional of a binary outcome.
    """

    engine: str
    estimand: str
    adjust_for_other_candidates: bool
    clustered: bool
    weighted: bool
    missing_outcome: bool
    n: int
    cross_fitted: bool
    n_folds: int
    logistic_learners: bool
    binary_outcome: bool


def variable_importance_support(
    *, adjustment: str, key: AdjustmentKey, entries: Sequence[Support]
) -> Support:
    """The support of a variable-importance run's multiplicity claim.

    The step-down max-T claim reads ``supported`` when the ``variable-importance`` study
    measures the run's configuration: a :class:`~cleverly.estimators.TMLE` of the ATE of a
    binary outcome, with the other candidates adjusted for, no ``id=``, ``weights=`` or
    ``delta=``, ``LogisticRegression`` for both learners, at least :data:`MEASURED_N` rows and
    at most :data:`MEASURED_ROWS` adjusted rows. In sample, each candidate is binary or has
    three levels; cross-fitted with :data:`MEASURED_FOLDS` folds, each candidate is binary.
    Every row's own estimate must read ``supported`` too. While :data:`PENDING_REASON` is set,
    the study has not run and every configuration reads ``experimental``. The
    Benjamini-Hochberg claim always reads ``experimental``: no theorem covers it.

    Parameters
    ----------
    adjustment : {"max_t", "bh"}
        The adjustment the run applied.
    key : AdjustmentKey
        The run's configuration.
    entries : sequence of Support
        The support of each adjusted row's own estimate.

    Returns
    -------
    Support
        The support of the adjusted p-values' error-rate claim.

    Examples
    --------
    >>> from cleverly._support import Support
    >>> from cleverly.variable_importance import AdjustmentKey, variable_importance_support
    >>> key = AdjustmentKey(
    ...     engine="TMLE", estimand="ate", adjust_for_other_candidates=True, clustered=False,
    ...     weighted=False, missing_outcome=False, n=4_000, cross_fitted=False, n_folds=10,
    ...     logistic_learners=True, binary_outcome=True,
    ... )
    >>> row = Support("supported", "Covered.", ("canonical-tmle",), row="T01.binary")
    >>> variable_importance_support(adjustment="bh", key=key, entries=[row]).label
    'experimental'
    """
    if adjustment == "bh":
        return Support("experimental", BH_REASON)
    reasons: list[str] = []
    study = f"The {VARIABLE_IMPORTANCE_STUDY} study"
    if key.engine != "TMLE":
        reasons.append(f"{study} measures TMLE only, and this run fits {key.engine}.")
    if key.estimand != "ate":
        reasons.append(f"{study} measures the ATE only, and this run targets {key.estimand}.")
    if not key.binary_outcome:
        reasons.append(f"{study} measures a binary outcome only.")
    if not key.adjust_for_other_candidates:
        reasons.append(
            f"{study} adjusts each candidate for the others, and this run does not, so its "
            "rows answer a different question for each candidate."
        )
    for passed, option in (
        (key.clustered, "clustered data (id=)"),
        (key.weighted, "weights="),
        (key.missing_outcome, "missing outcomes (delta=)"),
    ):
        if passed:
            reasons.append(f"No study covers {option} under adjustment='max_t'.")
    if not key.logistic_learners:
        reasons.append(
            f"{study} fits LogisticRegression for the outcome and the treatment, and this run "
            "uses another learner."
        )
    if key.cross_fitted and key.n_folds != MEASURED_FOLDS:
        reasons.append(
            f"{study} cross-fits with {MEASURED_FOLDS} folds, and this run uses {key.n_folds}."
        )
    if key.n < MEASURED_N:
        reasons.append(
            f"The run has {key.n:,} rows, and the {VARIABLE_IMPORTANCE_STUDY} study measures "
            f"n = {MEASURED_N:,}; "
            "at n = 2,000 its design probe found the per-candidate tests liberal enough to "
            "lift the family-wise error rate to 0.080 at level 0.05."
        )
    if len(entries) > MEASURED_ROWS:
        reasons.append(
            f"The run adjusts {len(entries)} rows, and the {VARIABLE_IMPORTANCE_STUDY} study "
            f"measures at most {MEASURED_ROWS}."
        )
    measured = MEASURED_ENTRY_ROWS[key.cross_fitted]
    if any(support.label != "supported" for support in entries):
        reasons.append("A candidate's own estimate is experimental.")
    elif any(support.row not in measured for support in entries):
        reasons.append(
            f"{study} measures "
            + (
                "cross-fitted binary candidates only."
                if key.cross_fitted
                else "binary and three-level candidates only."
            )
        )
    if not reasons and PENDING_REASON is not None:
        reasons.append(PENDING_REASON)
    if reasons:
        return Support("experimental", " ".join(reasons))
    return Support(
        "supported",
        f"{study} measures the family-wise error rate of this configuration.",
        (VARIABLE_IMPORTANCE_STUDY,),
    )


@dataclass(frozen=True)
class VariableImportanceEntry:
    """One candidate exposure's target estimate and declared adjustment set.

    Parameters
    ----------
    candidate : str
        Candidate exposure column.
    parameter : str
        Estimand alias targeted for it.
    adjustment_set : tuple of str
        Covariates adjusted for, as declared.
    estimate : ParameterEstimate
        The estimate and its influence curve.
    adjusted_pvalue : float or None
        Its p-value after the multiplicity adjustment. ``None`` on an entry whose estimate
        supplies no inference. :func:`variable_importance` never builds such an entry,
        because it refuses the estimator before its first fit.
    """

    candidate: str
    parameter: str
    adjustment_set: tuple[str, ...]
    estimate: ParameterEstimate
    adjusted_pvalue: float | None


@dataclass(frozen=True)
class ImportanceFacts:
    """What a variable-importance summary reads from the candidates' fits.

    Kept on every result, so the summary does not need the fits themselves.

    Parameters
    ----------
    candidates : tuple of str
        The candidates, in the order they were fitted.
    first_fit : mapping of str to str
        The header facts of the first candidate's fit summary.
    outcome : str
        The outcome column.
    n : int
        The number of rows.
    cautions : mapping of str to tuple of str
        For each candidate whose fit summary carries a warning note, in candidate order, the
        text of those warnings.
    fitted_method : str
        The fitted method of every candidate's fit, such as ``"tmle"``: the run fits each
        candidate with one estimator.

    Attributes
    ----------
    cautious : tuple of str
    """

    candidates: tuple[str, ...]
    first_fit: Mapping[str, str]
    outcome: str
    n: int
    cautions: Mapping[str, tuple[str, ...]]
    fitted_method: str

    @property
    def cautious(self) -> tuple[str, ...]:
        """The candidates whose fit summary carries a warning note, in candidate order."""
        return tuple(self.cautions)


@dataclass(frozen=True)
class VariableImportanceResult:
    """Candidate-exposure estimates with multiplicity-adjusted null tests.

    Parameters
    ----------
    entries : tuple of VariableImportanceEntry
        One entry per candidate and parameter.
    facts : ImportanceFacts
        What the summary reads from the fits, kept whether or not the fits are.
    adjustment_key : AdjustmentKey
        The run's configuration, which :attr:`adjustment_support` reads.
    keep_fits : bool
        Whether the run kept each candidate's fit. ``False`` keeps the entries and the facts
        only, and :attr:`fits` then refuses.
    method : str
        Multiplicity adjustment applied to the p-values: ``"step-down max-T"`` or
        ``"Benjamini-Hochberg"``.
    error_rate : {"fwer", "fdr"}
        What the adjusted p-values aim to control: the family-wise error rate under the
        step-down max-T, the false discovery rate under Benjamini-Hochberg.
    correlation : ndarray or None
        The ``(m, m)`` correlation of the rows' estimates, read from their influence curves,
        in the order of the entries, under the step-down max-T. ``None`` under
        Benjamini-Hochberg.
    multiplier_draws : int or None
        The number of Gaussian draws of the step-down max-T, ``None`` under
        Benjamini-Hochberg.
    multiplier_seed : int or None
        The seed of those draws, ``None`` under Benjamini-Hochberg.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns.
    kept_fits : mapping of str to TMLEResult
        The fits a run with ``keep_fits=True`` kept, empty otherwise. Read them through
        :attr:`fits`, which refuses when they were not kept.
    protocol : StudyProtocol or None
        Scientific study protocol the run was given. ``None`` records that no protocol was
        supplied. A full summary prints it as the study description.

    Attributes
    ----------
    adjustment_support : Support
    fits : mapping of str to TMLEResult
    fitted_method : str
    diagnostics : CandidateDiagnostics
    sensitivity : CandidateSensitivity
    replayability : Replayability
    """

    entries: tuple[VariableImportanceEntry, ...]
    facts: ImportanceFacts
    adjustment_key: AdjustmentKey
    keep_fits: bool = True
    method: str = "step-down max-T"
    error_rate: ErrorRate = "fwer"
    correlation: np.ndarray | None = field(default=None, repr=False, compare=False)
    multiplier_draws: int | None = None
    multiplier_seed: int | None = None
    backend: str | None = None
    kept_fits: Mapping[str, TMLEResult] = field(default_factory=dict, repr=False)
    protocol: StudyProtocol | None = None
    #: Assessment results keyed by operation and normalized arguments. ``init=False``, so a
    #: mapping handed to the constructor by ``dataclasses.replace`` cannot answer for
    #: another run.
    assessment_cache: AssessmentCache = field(
        default_factory=AssessmentCache, init=False, repr=False, compare=False
    )

    #: The assessment family: each operation runs on every candidate's own fit.
    assessment_family: ClassVar[str] = "variable_importance"

    def __post_init__(self) -> None:
        kept = set(self.kept_fits)
        if self.keep_fits and kept != set(self.facts.candidates):
            raise ValueError("a result with keep_fits=True holds one fit per candidate")
        if not self.keep_fits and kept:
            raise ValueError("a result with keep_fits=False holds no fits")
        _check_protocol(self.protocol)
        max_t = (self.method, self.error_rate) == (METHOD_NAMES["max_t"], "fwer")
        bh = (self.method, self.error_rate) == (METHOD_NAMES["bh"], "fdr")
        if not max_t and not bh:
            raise ValueError(
                "method and error_rate must be 'step-down max-T' with 'fwer' or "
                "'Benjamini-Hochberg' with 'fdr'"
            )
        recorded = (self.correlation, self.multiplier_draws, self.multiplier_seed)
        if bh and any(value is not None for value in recorded):
            raise ValueError("a Benjamini-Hochberg result records no correlation or draws")
        if max_t:
            m = len(self.entries)
            if self.correlation is None or np.shape(self.correlation) != (m, m):
                raise ValueError("a step-down max-T result records the (m, m) correlation")
            if self.multiplier_draws is None or self.multiplier_seed is None:
                raise ValueError("a step-down max-T result records its draws and seed")

    @property
    def adjustment_support(self) -> Support:
        """The support of the adjusted p-values' error-rate claim.

        It reads :func:`variable_importance_support` on the run's configuration and on each
        row's own estimate. The step-down max-T claim reads ``supported`` only on the
        configuration the ``variable-importance`` study measures, and the Benjamini-Hochberg
        claim always reads ``experimental``.
        """
        return variable_importance_support(
            adjustment="max_t" if self.error_rate == "fwer" else "bh",
            key=self.adjustment_key,
            entries=[entry.estimate.support for entry in self.entries],
        )

    @property
    def fits(self) -> Mapping[str, TMLEResult]:
        """The fit behind each candidate.

        Any read refuses on a ``keep_fits=False`` result, ``hasattr(result, "fits")``
        included, which therefore raises rather than returning ``False``.

        Raises
        ------
        CapabilityError
            When the run used ``keep_fits=False``, which kept no fit.
        """
        if not self.keep_fits:
            raise CapabilityError(refusal=KEEP_FITS_REFUSAL)
        return self.kept_fits

    @property
    def fitted_method(self) -> str:
        """The fitted method every candidate's fit shares, recorded when the run fitted."""
        return self.facts.fitted_method

    @property
    def diagnostics(self) -> Any:
        """The diagnostics of each candidate's fit, asked together.

        Each operation runs on every candidate's own fit and returns one report per
        candidate.  A ``keep_fits=False`` run refuses every one.
        """
        from ._importance_assessment import CandidateDiagnostics

        return CandidateDiagnostics(self)

    @property
    def sensitivity(self) -> Any:
        """The sensitivity analyses of each candidate's fit, asked together.

        Each analysis reads one candidate's own estimate and unadjusted interval, not the
        multiplicity-adjusted test.  A ``keep_fits=False`` run refuses every one.
        """
        from ._importance_assessment import CandidateSensitivity

        return CandidateSensitivity(self)

    @property
    def replayability(self) -> Any:
        """What every candidate's fit can redo; nothing on a ``keep_fits=False`` run."""
        from .assessment import replayability

        return replayability(self)

    def validate(self) -> Any:
        """Run the inexpensive default battery on every candidate's fit.

        Returns
        -------
        ValidationReport
            One row per check, combining the candidates' rows and naming each candidate
            that needs attention.
        """
        from .assessment import validate_result

        return validate_result(self)

    def assess(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> Any:
        """Run the post-fit assessment battery on every candidate's fit.

        Each row runs the operation on every candidate's own fit, with the same arguments
        and seed.  It reads the worst status any candidate reached and names each candidate
        that needs attention.  An operation that some candidate's fit refuses is refused,
        with that candidate's reason.

        Parameters
        ----------
        include_refits : bool
            Run requested operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets.
        arguments : mapping or None
            Per-operation keyword arguments, passed to every candidate.
        random_state : int or None
            Common seed for stochastic refit operations, the same for every candidate.

        Returns
        -------
        AssessmentReport
            Validation, diagnostics, and sensitivity in one report.
        """
        from .assessment import assess_result

        return assess_result(
            self,
            include_refits=include_refits,
            include_retargets=include_retargets,
            arguments=arguments,
            random_state=random_state,
        )

    def __getitem__(self, index: int) -> VariableImportanceEntry:
        return self.entries[index]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VariableImportanceEntry]:
        return iter(self.entries)

    def to_frame(self) -> Any:
        """One row per candidate/parameter, in the order of the entries.

        A run sorts its entries by adjusted p-value. An ordinary result emits ``std_err``,
        ``ci_lower``, ``ci_upper``, ``p_value`` and ``p_value_adjusted``. A result at a
        labelled status adds an ``inference`` column that names the status. A result whose
        entries supply no inference emits an ``inference`` column and the three spread
        columns under their
        diagnostic names, as ``TMLEResult.to_frame`` does. It emits neither p-value column.
        Every fit of one run shares one estimator and one set of rows, so the entries share
        one status.

        Returns
        -------
        dataframe
            One row per candidate and parameter, in the order of the entries.

        Raises
        ------
        ValueError
            When the entries declare different inference statuses.
        """
        status = reported_status(
            {str(index): entry.estimate for index, entry in enumerate(self.entries)}
        )
        inferential = supplies_inference(status)
        spreads = [entry.estimate.spread_columns() for entry in self.entries]
        payload: dict[str, Any] = {
            "candidate": [entry.candidate for entry in self.entries],
            "parameter": [entry.parameter for entry in self.entries],
            "psi": [entry.estimate.psi for entry in self.entries],
        }
        if is_labelled(status):
            payload["inference"] = [status] * len(self.entries)
        labels = support_column({str(i): entry.estimate for i, entry in enumerate(self.entries)})
        if labels is not None:
            payload["support"] = labels
        names = list(spreads[0]) if spreads else []
        payload.update({name: [spread[name] for spread in spreads] for name in names})
        if inferential:
            payload["p_value_adjusted"] = [entry.adjusted_pvalue for entry in self.entries]
        payload["adjustment_set"] = [", ".join(entry.adjustment_set) for entry in self.entries]
        return emit_frame(payload, backend=self.backend)

    def summary(self, *, detail: Detail = "short") -> Summary:
        """A printable report of the candidates, like a statsmodels summary.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"full"`` adds each candidate's adjustment set.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text, with the parts as data.
        """
        from ._summary import check_detail

        return _importance_summary(self, check_detail(detail))

    def __repr__(self) -> str:
        from ._summary import pvalue_text, short_repr

        facts: list[tuple[str, Any]] = [("candidates", len(self.facts.candidates))]
        if self.entries and self.entries[0].adjusted_pvalue is not None:
            top = self.entries[0]
            facts.append(("top", f"{top.candidate} adj. p={pvalue_text(top.adjusted_pvalue)}"))
        return short_repr("VariableImportanceResult", facts)

    def _repr_html_(self) -> str:
        return self.summary().as_html()


def _check_protocol(protocol: object) -> None:
    if protocol is not None and not isinstance(protocol, StudyProtocol):
        raise TypeError("protocol must be a StudyProtocol or None")


def _row_labels(entries: Sequence[VariableImportanceEntry]) -> list[str]:
    """One label per entry: the candidate, with the parameter when a candidate has several."""
    per_candidate: dict[str, int] = {}
    for entry in entries:
        per_candidate[entry.candidate] = per_candidate.get(entry.candidate, 0) + 1
    return [
        entry.candidate
        if per_candidate[entry.candidate] == 1
        else f"{entry.candidate}: {entry.parameter}"
        for entry in entries
    ]


def _importance_summary(result: VariableImportanceResult, detail: Detail) -> Summary:
    from ._summary import (
        Cell,
        Summary,
        SummarySection,
        SummaryTable,
        count_text,
        note,
        pvalue_text,
        vocabulary,
    )
    from ._summary._estimates import estimates_table
    from ._summary._setup import _protocol_section

    entries = result.entries
    labels = _row_labels(entries)
    estimates = {label: entry.estimate for label, entry in zip(labels, entries, strict=True)}
    level = 1.0 - entries[0].estimate.alpha if entries else 0.95
    table, notes = estimates_table(estimates, level=level)
    inferential = all(entry.adjusted_pvalue is not None for entry in entries)
    rows = []
    for row, entry in zip(table.rows, entries, strict=True):
        key = Cell((entry.candidate, entry.parameter), row[0].text)
        extra: tuple[Cell, ...] = ()
        if inferential:
            adjusted = float(entry.adjusted_pvalue)  # type: ignore[arg-type]
            extra = (Cell(adjusted, pvalue_text(adjusted)),)
        rows.append((key, *row[1:], *extra))
    columns = table.columns + ((vocabulary.ADJUSTED_P_VALUE,) if inferential else ())
    table = SummaryTable(columns, tuple(rows), title=table.title, kind="estimates")
    kept = result.facts
    header: list[tuple[str, str]] = []
    if kept.candidates:
        facts = kept.first_fit
        header.append(
            (
                vocabulary.QUESTION,
                f"effect of each candidate on {kept.outcome}, one fit per candidate",
            )
        )
        header.append((vocabulary.OUTCOME, facts.get(vocabulary.OUTCOME, "")))
        for label in (vocabulary.METHOD, vocabulary.OUTCOME_MODEL, vocabulary.TREATMENT_MODEL):
            if label in facts:
                header.append((label, facts[label]))
        header.append((vocabulary.ROWS, count_text(kept.n)))
    header.append((vocabulary.CANDIDATES, count_text(len(kept.candidates))))
    max_t = result.error_rate == "fwer"
    if inferential:
        rate = "family-wise error rate" if max_t else "false discovery rate"
        header.append(
            (vocabulary.ADJUSTMENT, f"{result.method}, {count_text(len(entries))} tests ({rate})")
        )
    notes = list(notes)
    if inferential:
        code = "multiple_testing" if max_t else "multiple_testing_bh"
        text = vocabulary.NOTE_TEXTS[code].format(count=count_text(len(entries)))
        support = result.adjustment_support
        if max_t and support.label == "experimental":
            text = f"{text} {vocabulary.ADJUSTMENT_EXPERIMENTAL} {support.reason}"
        notes.append(note(code, text))
    notes.append(note("separate_fits", vocabulary.NOTE_TEXTS["separate_fits"]))
    cautious = kept.cautious
    if cautious and result.keep_fits:
        notes.append(
            note(
                "fit_cautions",
                vocabulary.NOTE_TEXTS["fit_cautions"].format(names=", ".join(cautious)),
            )
        )
    elif cautious:
        # Without kept fits the user cannot print each fit's summary, so the warnings are
        # listed in place.
        listing = " ".join(f"{name}: {' '.join(texts)}" for name, texts in kept.cautions.items())
        notes.append(
            note(
                "fit_cautions_listed",
                vocabulary.NOTE_TEXTS["fit_cautions_listed"].format(
                    names=", ".join(cautious), listing=listing
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(_protocol_section(result.protocol))
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                tuple(
                    f"{label} is adjusted for {', '.join(entry.adjustment_set) or 'nothing'}."
                    for label, entry in zip(labels, entries, strict=True)
                ),
            )
        )
    return Summary.build(
        title=vocabulary.IMPORTANCE_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def _bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values, monotone in rank."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    m = values.size
    order = np.argsort(values, kind="stable")
    ranked = values[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _max_t_statistics(pvalues: Sequence[float]) -> np.ndarray:
    """The two-sided normal quantile of each row's own p-value, ``isf(p / 2)``.

    The upper-tail form keeps a p-value below about ``1e-16`` finite, where
    ``ppf(1 - p / 2)`` rounds to infinity and ties every such row.
    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    return np.asarray(stats.norm.isf(values / 2.0), dtype=float)


def _null_draws(correlation: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """``draws`` absolute draws of ``N(0, correlation)``, by Cholesky, one row per draw.

    This is the exact conditional law, given the data, of the Gaussian-multiplier bootstrap of
    the studentized, centred influence curves.
    """
    factor = np.linalg.cholesky(correlation)
    normals = np.random.default_rng(seed).standard_normal((draws, correlation.shape[0]))
    return np.abs(normals @ factor.T)


def _step_down(
    statistics: np.ndarray, pvalues: Sequence[float], null_draws: np.ndarray
) -> np.ndarray:
    """Step-down max-T adjusted p-values on fixed null draws.

    Rows are ranked by statistic, largest first. The row at rank ``k`` reads the share of draws
    whose maximum over ranks ``k`` to ``m`` reaches its statistic. A running maximum in rank
    makes the values monotone, each is floored at the row's own p-value, which absorbs Monte
    Carlo error only, and capped at one.
    """
    floor = np.asarray(pvalues, dtype=float)
    order = np.argsort(-statistics, kind="stable")
    ranked = null_draws[:, order]
    remaining_max = np.maximum.accumulate(ranked[:, ::-1], axis=1)[:, ::-1]
    exceed = np.mean(remaining_max >= statistics[order], axis=0)
    adjusted = np.empty(statistics.size, dtype=float)
    adjusted[order] = np.maximum.accumulate(exceed)
    return np.asarray(np.minimum(np.maximum(adjusted, floor), 1.0), dtype=float)


def _row_correlation(
    estimates: Sequence[ParameterEstimate], labels: Sequence[str], cluster: Any
) -> np.ndarray:
    """The correlation of the rows' estimates from their joint influence curves.

    Refuses a row whose curve has no variance and a correlation whose smallest eigenvalue is
    below :data:`CORRELATION_TOLERANCE`, naming the rows involved.
    """
    covariance = estimate_covariance(
        {str(index): estimate for index, estimate in enumerate(estimates)}, None, cluster=cluster
    )
    variances = np.diag(covariance)
    flat = [label for label, value in zip(labels, variances, strict=True) if not value > 0]
    if flat:
        raise CapabilityError(
            f"adjustment='max_t' needs each row's influence curve to vary, and {flat} do not, "
            "so their correlation with the other rows is undefined.",
            reason=RefusalReason.INFERENCE_GAP,
            alternative="Drop the constant candidates, or pass adjustment='bh' (experimental).",
        )
    scale = np.sqrt(variances)
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues[0] < CORRELATION_TOLERANCE:
        loading = np.abs(eigenvectors[:, 0])
        involved = [
            label
            for label, value in zip(labels, loading, strict=True)
            if value >= 0.1 * loading.max()
        ]
        raise CapabilityError(
            f"adjustment='max_t' needs the rows' correlation matrix to be nonsingular, and the "
            f"rows {involved} are linearly dependent to working precision (smallest "
            f"eigenvalue {eigenvalues[0]:.2e}, below {CORRELATION_TOLERANCE:.0e}): they are "
            "one test.",
            reason=RefusalReason.INFERENCE_GAP,
            alternative="Drop one of the dependent candidates, or pass adjustment='bh' "
            "(experimental).",
        )
    return np.asarray(correlation, dtype=float)


def _check_adjustment(
    adjustment: object, multiplier_draws: object, multiplier_seed: object
) -> None:
    if adjustment not in ADJUSTMENTS:
        raise ValueError(f"adjustment must be one of {list(ADJUSTMENTS)}; got {adjustment!r}")
    for name, value in (
        ("multiplier_draws", multiplier_draws),
        ("multiplier_seed", multiplier_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer; got {value!r}")
    assert isinstance(multiplier_draws, (int, np.integer))
    if multiplier_draws < MINIMUM_MULTIPLIER_DRAWS:
        raise ValueError(
            f"multiplier_draws must be at least {MINIMUM_MULTIPLIER_DRAWS:,}, where the Monte "
            f"Carlo error of an adjusted p-value near 0.05 is about 0.007; got {multiplier_draws}"
        )


#: Why the step-down max-T refuses estimated weights.
ESTIMATED_WEIGHTS_REFUSAL: Final[str] = (
    "adjustment='max_t' reads the joint law of the rows from their influence curves, and with "
    "weights_estimated=True the curves omit the term of the weight estimation, so the "
    "correlation misses a component of that law."
)

#: Why the step-down max-T refuses few clusters.
FEW_CLUSTER_REFUSAL: Final[str] = (
    "adjustment='max_t' rests on a Gaussian limit in the number of independent units, and "
    "with {clusters} clusters, fewer than {floor}, each row's test reads a Student t "
    "reference, which has no Gaussian joint law to draw from."
)

_MAX_T_ALTERNATIVE: Final[str] = "Pass adjustment='bh' (experimental)."


@dataclass(frozen=True)
class _Candidate:
    """One candidate after the pre-pass: its adjustment set, prepared data and engine."""

    name: str
    adjustment_set: tuple[str, ...]
    data: CausalData
    engine: TMLE
    prepare: Callable[[str, tuple[str, ...]], CausalData]


def _checked_candidates(
    data: Any,
    *,
    outcome: str,
    candidates: Sequence[str],
    covariates: Sequence[str],
    estimand: str,
    estimator: TMLE | None,
    adjust_for_other_candidates: bool,
    delta: str | None,
    weights: str | None,
    weights_type: str,
    weights_estimated: bool,
    id: str | None,
    adjustment: str,
    multiplier_draws: int,
    multiplier_seed: int,
) -> Iterator[_Candidate]:
    """Run every refusal of a variable-importance call and yield each candidate in turn.

    :func:`variable_importance` drains it before its first fit, and
    :func:`variable_importance_plan` drains it before it prices a candidate, so both refuse
    identically and in the same order. The refusals of the step-down max-T that the
    configuration decides come after every candidate's own checks.
    """
    _check_adjustment(adjustment, multiplier_draws, multiplier_seed)
    if not is_dataframe(data):
        raise TypeError(
            "variable_importance needs the original dataframe because each candidate "
            "takes a turn as treatment"
        )
    candidate_names = tuple(candidates)
    if not candidate_names or len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidates must be a non-empty sequence of distinct column names")
    base_covariates = tuple(dict.fromkeys(covariates))
    if outcome in candidate_names:
        raise DataError("the outcome cannot also be a candidate exposure")
    template = TMLE(estimands=estimand) if estimator is None else estimator
    # The fold-policy and declaration refusals of every fit come first, as they do in
    # ``fit``, so a copied or modified estimator meets the refusal that names its remedy.
    template._refuse_fold_policy()
    template._refuse_undeclared_functions()
    if getattr(template, "_has_derivative_arrays", lambda: False)():
        # Each candidate becomes the treatment, and a declared propensity array describes
        # the treatment it was fitted for.
        raise CapabilityError(
            "The refit changes the data (the covariates, the treatment or the rows), and a "
            "propensity array entry was fitted without that change.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="Pass learners to refit.",
        )
    # Every candidate's data is prepared, and its status asked, before the first fit.
    # This procedure ends in a Benjamini--Hochberg adjustment of one p-value per
    # candidate, so an estimator that supplies no p-value leaves it with nothing to
    # adjust.  ``_prepare`` fits no learner, so the refusal still arrives before the first
    # fit when the status depends on the data as well as on the configuration.  It is
    # asked of the estimator's own hook on the prepared data, which is what
    # ``_retarget_detailed`` stamps the estimates with, so the refusal and the estimates
    # the adjustment would read cannot disagree.
    frame_view = as_frame(data)

    def engine() -> TMLE:
        # A fresh copy of the template for each use. A candidate treatment has no declared
        # mechanism, so the known-score ATT and ATC (which weight by one) do not apply; the
        # candidate fit summarizes over the arm.
        fitted = copy(template)
        fitted.estimands = estimand
        fitted.conditional_form = "arm_mean"
        return fitted

    def prepare(candidate: str, adjustment_set: tuple[str, ...]) -> CausalData:
        return template._prepare(
            data,
            outcome=outcome,
            treatment=candidate,
            covariates=adjustment_set,
            delta=delta,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            id=id,
            intermediate=None,
            treatment_kind="discrete",
        )

    clusters: int | None = None
    for candidate in candidate_names:
        chosen = [name for name in base_covariates if name != candidate]
        if adjust_for_other_candidates:
            chosen.extend(name for name in candidate_names if name != candidate)
        adjustment_set = tuple(dict.fromkeys(chosen))
        if not adjustment_set:
            raise DataError(
                f"candidate {candidate!r} has an empty adjustment set; cleverly's "
                "point-treatment estimator requires at least one baseline covariate"
            )
        if candidate in frame_view.columns and has_nulls(frame_view, candidate):
            raise DataError(
                f"candidate {candidate!r} has missing values. variable_importance takes no "
                "missing treatment: each candidate's refit reads the treatment law of every "
                "row, which a missing treatment leaves unidentified. Drop those rows or "
                "impute the candidate first."
            )
        candidate_data = prepare(candidate, adjustment_set)
        # The outcome-scale refusal of every fit comes before the status, as the fold
        # policy does.
        template._refuse_unbounded_cross_fitted_scale(candidate_data)
        # The status reads the estimand the fit reports, not the template's own list.
        refuse_inference(
            engine()._inference_status(candidate_data),
            operation="variable_importance()",
        )
        if id is not None:
            clusters = candidate_data.n_clusters
        yield _Candidate(candidate, adjustment_set, candidate_data, engine(), prepare)
        del candidate_data
    if adjustment != "max_t":
        return
    if weights_estimated:
        raise CapabilityError(
            ESTIMATED_WEIGHTS_REFUSAL,
            reason=RefusalReason.INFERENCE_GAP,
            alternative="Pass known weights, or adjustment='bh' (experimental).",
        )
    if clusters is not None and clusters < FEW_CLUSTER_THRESHOLD:
        raise CapabilityError(
            FEW_CLUSTER_REFUSAL.format(clusters=clusters, floor=FEW_CLUSTER_THRESHOLD),
            reason=RefusalReason.INFERENCE_GAP,
            alternative=_MAX_T_ALTERNATIVE,
        )


def variable_importance(
    data: Any,
    *,
    outcome: str,
    candidates: Sequence[str],
    covariates: Sequence[str],
    estimand: str = "ate",
    estimator: TMLE | None = None,
    adjust_for_other_candidates: bool = True,
    adjustment: Adjustment = "max_t",
    multiplier_draws: int = 10_000,
    multiplier_seed: int = 0,
    delta: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    id: str | None = None,
    keep_fits: bool = True,
    protocol: StudyProtocol | None = None,
) -> VariableImportanceResult:
    """Estimate one causal importance parameter per candidate exposure.

    Each candidate is fitted as the treatment.  With
    ``adjust_for_other_candidates=True`` (the default), the other candidates join the
    supplied baseline covariates, matching the interpretation "intervene on this
    exposure while adjusting for the others."  Set it false when the other candidates
    are descendants, colliders, or otherwise should not be conditioned on.

    Tests are two-sided on each parameter's native null (zero for differences/levels,
    one for ratios), then adjusted jointly. This corrects the one-tail normal probability
    used by the historical ``tmle3_vim`` helper.

    The default adjustment, ``adjustment="max_t"``, is a studentized step-down max-T that
    controls the family-wise error rate, the chance of any false rejection among the rows,
    and so also bounds the false discovery rate. It reads the correlation of the rows'
    estimates from their influence curves, which every candidate's fit computes on the same
    rows. Each row's statistic is the two-sided normal quantile of its own p-value,
    ``norm.isf(p / 2)``. ``multiplier_draws`` draws of the absolute values of a normal vector
    with that correlation, seeded by ``multiplier_seed``, give the step-down critical values:
    this is the Gaussian-multiplier bootstrap of the studentized curves computed through its
    exact conditional law. The adjusted p-values are made monotone in rank and floored at
    each row's own p-value, so the result is deterministic for a given seed. The control is
    asymptotic in the number of independent units (Dudoit, van der Laan and Pollard, Part II,
    Theorems 1, 3 and 6; Chernozhukov, Chetverikov and Kato 2013, Theorem 5.1).
    ``adjustment="bh"`` applies Benjamini--Hochberg, whose false-discovery claim is
    experimental: no theorem covers two-sided tests that share rows, and correlated
    two-sided Gaussian tests can exceed its rate.
    :attr:`VariableImportanceResult.adjustment_support` labels either claim.

    The step-down max-T refuses, before the first fit, ``weights_estimated=True`` (the curves
    omit the weight-estimation term) and clustered data with fewer than 40 clusters (each
    test reads a Student t reference). After the fits it refuses a row whose test still reads
    a t reference, a row whose curve does not vary, and rows whose correlation matrix is
    singular to working precision. Benjamini--Hochberg runs in each of these cases.

    Refuses, before the first fit, when the estimator's inference status on any
    candidate's prepared data supplies no inference. A
    :class:`~cleverly.estimators.CTMLE` whose ``strategy`` is ``"greedy"``, ``"ordered"``,
    ``"discrete"``, ``"lasso"`` or ``"learners"`` is one such case, and the
    :doc:`inference reference </technical-reference/inference>` lists each status. The
    adjustment above needs one p-value per candidate, and such a fit supplies none, so
    this procedure has no diagnostic form to fall back to. A ``"derivative"`` fit
    (C-TMLE0) supplies inference and is admitted, unless it declares propensity arrays,
    which describe one treatment only. Before that check, an estimator with a rule, a
    regime density or an MSM function that is not declared known meets the refusal that its
    own fit gives.

    Parameters
    ----------
    data : dataframe
        Observed study data.
    outcome : str
        Outcome column.
    candidates : sequence of str
        Columns to fit as the treatment, one at a time.
    covariates : sequence of str
        Baseline adjustment columns, shared by every candidate.
    estimand : str
        Alias targeted for each candidate.
    estimator : TMLE or None
        Configured estimator to reuse. ``None`` builds one with the defaults. An
        estimator whose status supplies no inference is refused here, because it reports
        no p-value.
    adjust_for_other_candidates : bool
        Whether the other candidates join the baseline covariates. Set it false when
        they are descendants, colliders, or otherwise should not be conditioned on.
    adjustment : {"max_t", "bh"}
        The multiplicity adjustment: the step-down max-T (family-wise error rate) or
        Benjamini--Hochberg (false discovery rate, experimental).
    multiplier_draws : int
        Gaussian draws of the step-down max-T, at least 1,000. The Monte Carlo error of an
        adjusted p-value near 0.05 is about 0.002 at the default 10,000. Read only under
        ``"max_t"``.
    multiplier_seed : int
        Seed of those draws. Read only under ``"max_t"``.
    delta : str or None
        Outcome-observation indicator column.
    weights : str or None
        Probability-weight column.
    weights_type : {"probability"}
        Interpretation of ``weights``.
    weights_estimated : bool
        Whether the supplied weights were estimated from these data.
    id : str or None
        Independent-cluster identifier used for variance estimation.
    keep_fits : bool
        Whether the result keeps each candidate's fit. ``False`` keeps the estimates, the
        adjusted p-values and the summary, and releases each candidate's data and fit before
        the next one starts. Each estimate keeps its length-n influence curve, which the
        adjustment reads, so memory still grows by one curve per row but no longer by a fit
        per candidate. Each candidate's data is then prepared a second time for its fit.
    protocol : StudyProtocol or None
        Scientific study protocol for the whole ranking, recorded on the result and printed
        by ``summary(detail="full")`` as the study description. ``None`` records that no
        protocol was supplied. Each candidate's fit stores the protocol's fingerprint in its
        provenance, as a fit from a :class:`~cleverly.CausalStudy` does.

    Returns
    -------
    VariableImportanceResult
        One entry per candidate, with its estimate and adjusted p-value.
    """
    _check_protocol(protocol)
    prepared: dict[str, tuple[tuple[str, ...], CausalData | None]] = {}
    engines: dict[str, TMLE] = {}
    prepare: Callable[[str, tuple[str, ...]], CausalData] | None = None
    for checked in _checked_candidates(
        data,
        outcome=outcome,
        candidates=candidates,
        covariates=covariates,
        estimand=estimand,
        estimator=estimator,
        adjust_for_other_candidates=adjust_for_other_candidates,
        delta=delta,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        id=id,
        adjustment=adjustment,
        multiplier_draws=multiplier_draws,
        multiplier_seed=multiplier_seed,
    ):
        # Without kept fits, the data is prepared again for its fit, so no candidate's data
        # outlives its own turn.
        prepared[checked.name] = (checked.adjustment_set, checked.data if keep_fits else None)
        engines[checked.name], prepare = checked.engine, checked.prepare
        del checked
    assert prepare is not None  # the candidates are non-empty
    fits: dict[str, TMLEResult] = {}
    first_fit: Mapping[str, str] = {}
    outcome_name, n_rows, fitted_method = outcome, 0, ""
    cautions: dict[str, tuple[str, ...]] = {}
    raw: list[tuple[str, str, tuple[str, ...], ParameterEstimate]] = []
    cluster: Any = None
    first_engine = next(iter(engines.values()))
    for candidate in list(prepared):
        adjustment_set, kept_data = prepared[candidate]
        if kept_data is None:
            # The user saw this data's warnings when the pre-pass built it; the fit still
            # records them.
            with muted():
                kept_data = prepare(candidate, adjustment_set)
        else:
            prepared[candidate] = (adjustment_set, None)
        codes = kept_data.cluster
        if not raw:
            cluster = codes
        elif not _same_rows(cluster, codes, n_rows, int(kept_data.n)):
            # The joint curves need every fit on the same rows and units.
            raise RuntimeError(
                f"the fit for {candidate!r} reads other rows or clusters than the first fit"
            )
        result = engines.pop(candidate).fit(kept_data).single()
        del kept_data
        if protocol is not None and result.provenance is not None:
            # Each candidate's fit stores the digest, as a fit from a study does.
            result = replace(
                result,
                provenance=replace(result.provenance, protocol_fingerprint=protocol.fingerprint),
            )
        fit_summary = result.summary()
        if not raw:
            first_fit = dict(fit_summary.facts)
            outcome_name, n_rows = result.data.outcome_name, int(result.data.n)
            fitted_method = result.fitted_method
        warned = tuple(item.text for item in fit_summary.notes if item.kind == "warning")
        if warned:
            cautions[candidate] = warned
        if keep_fits:
            fits[candidate] = result
        estimates = [
            estimate
            for name, estimate in result.estimates.items()
            if parameter_stem(name) == estimand
        ]
        if not estimates:
            raise RuntimeError(
                f"the fit for {candidate!r} produced no parameter from target {estimand!r}"
            )
        raw.extend((candidate, estimate.name, adjustment_set, estimate) for estimate in estimates)
        del result, estimates, fit_summary

    estimates = [item[3] for item in raw]
    if any(estimate.influence_curve.shape != (n_rows,) for estimate in estimates):
        raise RuntimeError("every candidate's influence curve must have one value per row")
    pvalues = [estimate.pvalue for estimate in estimates]
    correlation: np.ndarray | None = None
    if adjustment == "max_t":
        labels = [f"{candidate}: {parameter}" for candidate, parameter, _, _ in raw]
        t_rows = [
            label
            for label, estimate in zip(labels, estimates, strict=True)
            if estimate.reference_df is not None
        ]
        if t_rows:
            raise CapabilityError(
                f"adjustment='max_t' rests on a Gaussian limit, and the tests of {t_rows} read "
                "a Student t reference (a clustered fold-evaluated fit, or fewer than 40 "
                "clusters with positive weight), which has no Gaussian joint law to draw from.",
                reason=RefusalReason.INFERENCE_GAP,
                alternative=_MAX_T_ALTERNATIVE,
            )
        correlation = _row_correlation(estimates, labels, cluster)
        statistics = _max_t_statistics(pvalues)
        adjusted = _step_down(
            statistics, pvalues, _null_draws(correlation, int(multiplier_draws), multiplier_seed)
        )
    else:
        adjusted = _bh_adjust(pvalues)
    order = sorted(range(len(raw)), key=lambda index: float(adjusted[index]))
    entries = [VariableImportanceEntry(*raw[index], float(adjusted[index])) for index in order]
    if correlation is not None:
        correlation = correlation[np.ix_(order, order)].copy()
        correlation.setflags(write=False)
    key = AdjustmentKey(
        engine=type(first_engine).__name__,
        estimand=estimand,
        adjust_for_other_candidates=adjust_for_other_candidates,
        clustered=id is not None,
        weighted=weights is not None,
        missing_outcome=delta is not None,
        n=n_rows,
        cross_fitted=bool(getattr(first_engine, "cross_fit", False)),
        n_folds=int(getattr(first_engine, "n_folds", 0)),
        logistic_learners=_logistic_learners(first_engine),
        binary_outcome=all(estimate.outcome_family == "binomial" for estimate in estimates),
    )
    facts = ImportanceFacts(
        candidates=tuple(prepared),
        first_fit=first_fit,
        outcome=outcome_name,
        n=n_rows,
        cautions=cautions,
        fitted_method=fitted_method,
    )
    max_t = adjustment == "max_t"
    return VariableImportanceResult(
        tuple(entries),
        facts,
        key,
        keep_fits=keep_fits,
        method=METHOD_NAMES[adjustment],
        error_rate="fwer" if max_t else "fdr",
        correlation=correlation,
        multiplier_draws=int(multiplier_draws) if max_t else None,
        multiplier_seed=int(multiplier_seed) if max_t else None,
        backend=backend_of(as_frame(data)),
        kept_fits=fits,
        protocol=protocol,
    )


def _same_rows(first: Any, codes: Any, n_first: int, n: int) -> bool:
    """Whether a candidate's data has the first candidate's rows and cluster codes."""
    if n_first and n != n_first:
        return False
    if first is None or codes is None:
        return first is None and codes is None
    return bool(np.array_equal(np.asarray(first), np.asarray(codes)))


def _logistic_learners(engine: TMLE) -> bool:
    from sklearn.linear_model import LogisticRegression

    return isinstance(getattr(engine, "outcome_learner", None), LogisticRegression) and (
        isinstance(getattr(engine, "treatment_learner", None), LogisticRegression)
    )


def _candidate_scope(name: str, scope: str) -> str:
    return f"{scope}, candidate {name!r}"


def variable_importance_plan(
    data: Any,
    *,
    outcome: str,
    candidates: Sequence[str],
    covariates: Sequence[str],
    estimand: str = "ate",
    estimator: TMLE | None = None,
    adjust_for_other_candidates: bool = True,
    adjustment: Adjustment = "max_t",
    multiplier_draws: int = 10_000,
    multiplier_seed: int = 0,
    delta: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    id: str | None = None,
    protocol: StudyProtocol | None = None,
) -> PlannedFits:
    """Count the model fits :func:`variable_importance` would make, without fitting any.

    Takes the arguments of :func:`variable_importance` except ``keep_fits``, which does not
    change the fits. It runs the same refusals in the same order, prepares each candidate's
    data as the run does, and counts the fits of each candidate's own TMLE fit. The
    multiplicity adjustment fits no model, so ``adjustment`` and the multiplier settings
    change no count; the plan raises the refusals the configuration decides before any fit.
    The refusals the step-down max-T can make only after the fits (a row on a t reference,
    a constant curve, singular correlation) are not previewed.

    Parameters
    ----------
    data : DataFrame
        The original pandas or polars dataframe.
    outcome : str
        Outcome column.
    candidates : sequence of str
        Candidate exposures, each fitted in turn as the treatment.
    covariates : sequence of str
        Baseline covariates.
    estimand : str
        Target parameter stem fitted for each candidate.
    estimator : TMLE or None
        Template estimator; ``None`` uses ``TMLE(estimands=estimand)``.
    adjust_for_other_candidates : bool
        Whether the other candidates join each candidate's adjustment set.
    adjustment : {"max_t", "bh"}
        The multiplicity adjustment, checked as the run checks it.
    multiplier_draws : int
        Gaussian draws of the step-down max-T, checked as the run checks them.
    multiplier_seed : int
        Seed of those draws, checked as the run checks it.
    delta : str or None
        Outcome-observation indicator column.
    weights : str or None
        Probability-weight column.
    weights_type : {"probability"}
        Interpretation of ``weights``.
    weights_estimated : bool
        Whether the supplied weights were estimated from these data.
    id : str or None
        Independent-cluster identifier used for variance estimation.
    protocol : StudyProtocol or None
        Scientific study protocol, checked as the run checks it. It changes no fit.

    Returns
    -------
    PlannedFits
        One row per model, scope and candidate. Each scope ends with the candidate it
        belongs to, and the multiplier ``"candidate exposures"`` gives their number. A multiplier
        whose value differs between candidates, such as the number of arms, is left out and
        named in a note.

    See Also
    --------
    variable_importance : The fits this counts.
    cleverly.PlannedFits : The record returned.
    cleverly.IdentifiedEffect.planned_fits : The same preview for one estimate.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import variable_importance_plan
    >>> from cleverly.estimators import TMLE
    >>> rng = np.random.default_rng(0)
    >>> frame = pd.DataFrame(
    ...     {
    ...         "W": rng.normal(size=200),
    ...         "X1": rng.integers(0, 2, 200),
    ...         "X2": rng.integers(0, 2, 200),
    ...     }
    ... )
    >>> frame["Y"] = frame["X1"] + frame["W"] + rng.normal(size=200)
    >>> plan = variable_importance_plan(
    ...     frame,
    ...     outcome="Y",
    ...     candidates=["X1", "X2"],
    ...     covariates=["W"],
    ...     estimator=TMLE(
    ...         outcome_learner=LinearRegression(),
    ...         treatment_learner=LogisticRegression(),
    ...         cross_fit=False,
    ...         random_state=0,
    ...     ),
    ... )
    >>> plan.total_fits, plan.exact
    (4, True)
    """
    from ._planned_fits import PlanBuilder

    _check_protocol(protocol)
    # Every check runs before any candidate is priced, as every check runs before the first
    # fit, so the preview raises the run's first refusal. A candidate's preview can itself
    # refuse (it runs the fit's own preflights), as the run's fit of that candidate would.
    checked_candidates: list[tuple[str, tuple[str, ...], TMLE]] = []
    prepare: Callable[[str, tuple[str, ...]], CausalData] | None = None
    for checked in _checked_candidates(
        data,
        outcome=outcome,
        candidates=candidates,
        covariates=covariates,
        estimand=estimand,
        estimator=estimator,
        adjust_for_other_candidates=adjust_for_other_candidates,
        delta=delta,
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        id=id,
        adjustment=adjustment,
        multiplier_draws=multiplier_draws,
        multiplier_seed=multiplier_seed,
    ):
        checked_candidates.append((checked.name, checked.adjustment_set, checked.engine))
        prepare = checked.prepare
        del checked
    assert prepare is not None  # the candidates are non-empty
    builder = PlanBuilder()
    plans: dict[str, PlannedFits] = {}
    for name, adjustment_set, engine in checked_candidates:
        # The data is prepared again, one candidate at a time, as under keep_fits=False; the
        # user saw its warnings in the checks above. ``_planned_fits`` mirrors the run's
        # ``engine.fit(prepared)`` and fits nothing.
        with muted():
            candidate_data = prepare(name, adjustment_set)
        plans[name] = engine._planned_fits(candidate_data)
        del candidate_data
        builder.absorb(plans[name], functools.partial(_candidate_scope, name))
    first = next(iter(plans.values()))
    # The fold state reads only the template's seeds and split plan, so it is the same for
    # every candidate.
    assert all(plan.folds == first.folds for plan in plans.values())
    builder.multiply("candidate exposures", len(plans))
    values: dict[str, dict[str, int]] = {}
    for name, plan in plans.items():
        for factor, value in plan.multipliers:
            values.setdefault(factor, {})[name] = value
    for factor, by_candidate in values.items():
        if len(by_candidate) == len(plans) and len(set(by_candidate.values())) == 1:
            builder.multiply(factor, next(iter(by_candidate.values())))
        else:
            listed = ", ".join(f"{name} {by_candidate.get(name, 'none')}" for name in plans)
            builder.note(
                f"The candidates differ in {factor} ({listed}), so it is not listed as one "
                "multiplier; each candidate's rows include its own."
            )
    return builder.build(first.folds)
