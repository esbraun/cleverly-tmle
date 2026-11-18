"""Exposure-wise variable importance from explicitly repeated target parameters.

This is orchestration, not a new influence function: candidate ``X_j`` is assigned the
treatment role in its own fit and the requested causal parameter is estimated while the
remaining declared variables are adjustment covariates.  Consequently each row states
its own intervention and adjustment set, and no shared-nuisance shortcut is implied.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._inference_status import is_labelled, supplies_inference
from ._warnings import muted
from .data.causal_data import CausalData
from .estimators.base import TMLEResult
from .estimators.tmle import TMLE
from .exceptions import CapabilityError, DataError, RefusalReason, refuse_inference
from .inference.influence import ParameterEstimate, support_column
from .inference.results import reported_status
from .targets import parameter_stem
from .utils.frames import as_frame, backend_of, emit_frame, has_nulls, is_dataframe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._summary import Detail, Summary

__all__ = [
    "ImportanceFacts",
    "VariableImportanceEntry",
    "VariableImportanceResult",
    "variable_importance",
]


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

    Attributes
    ----------
    cautious : tuple of str
    """

    candidates: tuple[str, ...]
    first_fit: Mapping[str, str]
    outcome: str
    n: int
    cautions: Mapping[str, tuple[str, ...]]

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
    """

    entries: tuple[VariableImportanceEntry, ...]
    facts: ImportanceFacts
    keep_fits: bool = True
    method: str = "Benjamini-Hochberg"
    backend: str | None = None
    kept_fits: Mapping[str, TMLEResult] = field(default_factory=dict, repr=False)

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
            raise CapabilityError(
                "This variable-importance run used keep_fits=False, which keeps each "
                "candidate's estimates and drops its fit.",
                reason=RefusalReason.API_REDIRECTION,
                alternative="Run variable_importance again with keep_fits=True.",
            )
        return self.kept_fits

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
    prepared: dict[str, tuple[tuple[str, ...], CausalData | None]] = {}
    frame_view = as_frame(data)

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
        # The status reads the estimand the fit below reports, not the template's own list.
        fitted_estimator = copy(template)
        fitted_estimator.estimands = estimand
        # A candidate treatment has no declared mechanism, so the known-score ATT and ATC
        # (which weight by one) do not apply; the candidate fit summarizes over the arm.
        fitted_estimator.conditional_form = "arm_mean"
        refuse_inference(
            fitted_estimator._inference_status(candidate_data),
            operation="variable_importance()",
        )
        # Without kept fits, the data is prepared again for its fit, so no candidate's data
        # outlives its own turn.
        prepared[candidate] = (adjustment_set, candidate_data if keep_fits else None)
        del candidate_data
    fits: dict[str, TMLEResult] = {}
    first_fit: Mapping[str, str] = {}
    outcome_name, n_rows = outcome, 0
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
        fitted_estimator = copy(template)
        fitted_estimator.estimands = estimand
        # A candidate treatment has no declared mechanism, so the known-score ATT and ATC
        # (which weight by one) do not apply; the candidate fit summarizes over the arm.
        fitted_estimator.conditional_form = "arm_mean"
        result = fitted_estimator.fit(kept_data).single()
        del kept_data
        fit_summary = result.summary()
        if not raw:
            first_fit = dict(fit_summary.facts)
            outcome_name, n_rows = result.data.outcome_name, int(result.data.n)
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
    )
    return VariableImportanceResult(
        tuple(entries),
        facts,
        keep_fits=keep_fits,
        backend=backend_of(as_frame(data)),
        kept_fits=fits,
    )
