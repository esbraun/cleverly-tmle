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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ._assessment_cache import AssessmentCache
from ._inference_status import is_labelled, supplies_inference
from ._warnings import muted
from .data.causal_data import CausalData
from .estimators.base import TMLEResult
from .estimators.tmle import TMLE
from .exceptions import CapabilityError, DataError, Refusal, RefusalReason, refuse_inference
from .inference.influence import ParameterEstimate, support_column
from .inference.results import reported_status
from .targets import parameter_stem
from .utils.frames import as_frame, backend_of, emit_frame, has_nulls, is_dataframe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._planned_fits import PlannedFits
    from ._summary import Detail, Summary

__all__ = [
    "KEEP_FITS_REFUSAL",
    "ImportanceFacts",
    "VariableImportanceEntry",
    "VariableImportanceResult",
    "variable_importance",
    "variable_importance_plan",
]

#: Why a ``keep_fits=False`` run refuses :attr:`VariableImportanceResult.fits` and every
#: diagnostic and sensitivity analysis, each of which runs on a candidate's fit.
KEEP_FITS_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "This variable-importance run used keep_fits=False, which keeps each candidate's "
    "estimates and drops its fit.",
    alternative="Run variable_importance again with keep_fits=True.",
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
    keep_fits : bool
        Whether the run kept each candidate's fit. ``False`` keeps the entries and the facts
        only, and :attr:`fits` then refuses.
    method : str
        Multiplicity adjustment applied to the p-values.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns.
    kept_fits : mapping of str to TMLEResult
        The fits a run with ``keep_fits=True`` kept, empty otherwise. Read them through
        :attr:`fits`, which refuses when they were not kept.

    Attributes
    ----------
    fits : mapping of str to TMLEResult
    fitted_method : str
    diagnostics : CandidateDiagnostics
    sensitivity : CandidateSensitivity
    replayability : Replayability
    """

    entries: tuple[VariableImportanceEntry, ...]
    facts: ImportanceFacts
    keep_fits: bool = True
    method: str = "Benjamini-Hochberg"
    backend: str | None = None
    kept_fits: Mapping[str, TMLEResult] = field(default_factory=dict, repr=False)
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
        Benjamini-Hochberg adjusted test.  A ``keep_fits=False`` run refuses every one.
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
    if inferential:
        header.append((vocabulary.ADJUSTMENT, f"{result.method}, {count_text(len(entries))} tests"))
    notes = list(notes)
    if inferential:
        notes.append(
            note(
                "multiple_testing",
                vocabulary.NOTE_TEXTS["multiple_testing"].format(
                    count=count_text(len(entries)), method=result.method
                ),
            )
        )
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
) -> Iterator[_Candidate]:
    """Run every refusal of a variable-importance call and yield each candidate in turn.

    :func:`variable_importance` drains it before its first fit, and
    :func:`variable_importance_plan` prices each candidate as it is yielded, so both refuse
    identically and in the same order.
    """
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

    for candidate in candidate_names:
        adjustment = [name for name in base_covariates if name != candidate]
        if adjust_for_other_candidates:
            adjustment.extend(name for name in candidate_names if name != candidate)
        adjustment_set = tuple(dict.fromkeys(adjustment))
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
        yield _Candidate(candidate, adjustment_set, candidate_data, engine(), prepare)
        del candidate_data


def variable_importance(
    data: Any,
    *,
    outcome: str,
    candidates: Sequence[str],
    covariates: Sequence[str],
    estimand: str = "ate",
    estimator: TMLE | None = None,
    adjust_for_other_candidates: bool = True,
    delta: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    id: str | None = None,
    keep_fits: bool = True,
) -> VariableImportanceResult:
    """Estimate one causal importance parameter per candidate exposure.

    Each candidate is fitted as the treatment.  With
    ``adjust_for_other_candidates=True`` (the default), the other candidates join the
    supplied baseline covariates, matching the interpretation "intervene on this
    exposure while adjusting for the others."  Set it false when the other candidates
    are descendants, colliders, or otherwise should not be conditioned on.

    Tests are two-sided on each parameter's native null (zero for differences/levels,
    one for ratios), then adjusted jointly by Benjamini--Hochberg.  This corrects the
    one-tail normal probability used by the historical ``tmle3_vim`` helper.

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
        the next one starts, so memory no longer grows with the number of candidates. Each
        candidate's data is then prepared a second time for its fit.

    Returns
    -------
    VariableImportanceResult
        One entry per candidate, with its estimate and adjusted p-value.
    """
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
    for candidate in list(prepared):
        adjustment_set, kept_data = prepared[candidate]
        if kept_data is None:
            # The user saw this data's warnings when the pre-pass built it; the fit still
            # records them.
            with muted():
                kept_data = prepare(candidate, adjustment_set)
        else:
            prepared[candidate] = (adjustment_set, None)
        result = engines.pop(candidate).fit(kept_data).single()
        del kept_data
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

    adjusted = _bh_adjust([item[3].pvalue for item in raw])
    ranked = sorted(zip(raw, adjusted, strict=True), key=lambda item: float(item[1]))
    entries = [
        VariableImportanceEntry(candidate, parameter, adjustment, estimate, float(pvalue))
        for (candidate, parameter, adjustment, estimate), pvalue in ranked
    ]
    facts = ImportanceFacts(
        candidates=tuple(prepared),
        first_fit=first_fit,
        outcome=outcome_name,
        n=n_rows,
        cautions=cautions,
        fitted_method=fitted_method,
    )
    return VariableImportanceResult(
        tuple(entries),
        facts,
        keep_fits=keep_fits,
        backend=backend_of(as_frame(data)),
        kept_fits=fits,
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
    delta: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
    id: str | None = None,
) -> PlannedFits:
    """Count the model fits :func:`variable_importance` would make, without fitting any.

    Takes the arguments of :func:`variable_importance` except ``keep_fits``, which does not
    change the fits. It runs the same refusals in the same order, prepares each candidate's
    data as the run does, and counts the fits of each candidate's own TMLE fit.

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
