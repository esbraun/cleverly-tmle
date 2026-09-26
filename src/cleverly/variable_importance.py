"""Exposure-wise variable importance from explicitly repeated target parameters.

This is orchestration, not a new influence function: candidate ``X_j`` is assigned the
treatment role in its own fit and the requested causal parameter is estimated while the
remaining declared variables are adjustment covariates.  Consequently each row states
its own intervention and adjustment set, and no shared-nuisance shortcut is implied.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._inference_status import supplies_inference
from .data.causal_data import CausalData
from .estimators.base import TMLEResult
from .estimators.tmle import TMLE
from .exceptions import DataError, refuse_inference
from .inference.influence import ParameterEstimate
from .inference.results import reported_status
from .targets import parameter_stem
from .utils.frames import as_frame, backend_of, emit_frame, is_dataframe

__all__ = ["VariableImportanceEntry", "VariableImportanceResult", "variable_importance"]


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
        Its p-value after the multiplicity adjustment. ``None`` on a restored entry whose
        estimate supplies no inference, as :class:`VariableImportanceResult` withholds it
        when it loads. A live run never builds such an entry, because
        :func:`variable_importance` refuses the estimator before its first fit.
    """

    candidate: str
    parameter: str
    adjustment_set: tuple[str, ...]
    estimate: ParameterEstimate
    adjusted_pvalue: float | None

    def _restamped(self, fit: TMLEResult | None) -> VariableImportanceEntry:
        """This entry with the estimate of its restored fit, when that fit re-stamped it.

        ``TMLEResult.__setstate__`` re-stamps the estimates of the fit alone, so an entry
        saved beside its fit keeps the status it was saved with. The fit's estimate of the
        same name carries the new status and the same numbers. When that status supplies
        no inference, the adjusted p-value is withheld as ``None``.

        Parameters
        ----------
        fit : TMLEResult or None
            The restored fit of this entry's candidate, or ``None`` when the result holds
            none.

        Returns
        -------
        VariableImportanceEntry
            This entry, or a copy that carries the fit's estimate.
        """
        stamped = None if fit is None else fit.estimates.get(self.parameter)
        if stamped is None or stamped.inference == self.estimate.inference:
            return self
        adjusted = self.adjusted_pvalue if stamped.supplies_inference else None
        return replace(self, estimate=stamped, adjusted_pvalue=adjusted)


@dataclass(frozen=True)
class VariableImportanceResult:
    """Candidate-exposure estimates with multiplicity-adjusted null tests.

    Parameters
    ----------
    entries : tuple of VariableImportanceEntry
        One entry per candidate and parameter.
    fits : mapping of str to TMLEResult
        The fit behind each candidate.
    method : str
        Multiplicity adjustment applied to the p-values.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns.
    """

    entries: tuple[VariableImportanceEntry, ...]
    fits: Mapping[str, TMLEResult]
    method: str = "Benjamini-Hochberg"
    backend: str | None = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a result, and give each entry the status its re-stamped fit carries.

        Each fit re-stamps itself as it loads (``TMLEResult.__setstate__``), whichever
        status this version gives its configuration. An entry holds its own copy of the
        estimate, so without this step it would keep the status it was saved with, and
        :meth:`to_frame` would publish a p-value that its fit refuses.

        Parameters
        ----------
        state : dict of str to Any
            The pickled instance state.
        """
        self.__dict__.update(state)
        fits = state.get("fits") or {}
        self.__dict__["entries"] = tuple(
            entry._restamped(fits.get(entry.candidate)) for entry in state.get("entries", ())
        )

    def __getitem__(self, index: int) -> VariableImportanceEntry:
        return self.entries[index]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VariableImportanceEntry]:
        return iter(self.entries)

    def to_frame(self) -> Any:
        """One row per candidate/parameter, in the order of the entries.

        A live run sorts its entries by adjusted p-value, and a restored result keeps the
        saved order. An ordinary result emits ``std_err``, ``ci_lower``, ``ci_upper``,
        ``p_value`` and ``p_value_adjusted``. A restored result whose entries supply no
        inference emits an ``inference`` column and the three spread columns under their
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
        if not inferential:
            payload["inference"] = [status] * len(self.entries)
        names = list(spreads[0]) if spreads else []
        payload.update({name: [spread[name] for spread in spreads] for name in names})
        if inferential:
            payload["p_value_adjusted"] = [entry.adjusted_pvalue for entry in self.entries]
        payload["adjustment_set"] = [", ".join(entry.adjustment_set) for entry in self.entries]
        return emit_frame(payload, backend=self.backend)


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
    :class:`~cleverly.estimators.CTMLE` whose ``strategy`` is ``"greedy"``, ``"ordered"``
    or ``"discrete"`` is one such case, and the
    :doc:`inference reference </technical-reference/inference>` lists each status. The
    adjustment above needs one p-value per candidate, and such a fit supplies none, so
    this procedure has no diagnostic form to fall back to. Before that check, an estimator
    with a rule, a regime density or an MSM function that is not declared known meets the
    refusal that its own fit gives.

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
    # The fold-policy refusal of every fit comes first, as it does in ``fit``.  Asked below,
    # the status hook would report a restored stratified split as the status of a saved
    # result (roadmap row RM31), and its reason does not name the remedy.
    template._refuse_fold_policy()
    # The declaration refusals of every fit come next, as they do in ``fit``.  Asked
    # below, the status hook would report an undeclared function as the status of a
    # restored result (roadmap row RM28), and its reason does not name the remedy.
    template._refuse_undeclared_functions()
    # Every candidate's data is prepared, and its status asked, before the first fit.
    # This procedure ends in a Benjamini--Hochberg adjustment of one p-value per
    # candidate, so an estimator that supplies no p-value leaves it with nothing to
    # adjust.  ``_prepare`` fits no learner, so the refusal still arrives before the first
    # fit when the status depends on the data as well as on the configuration.  It is
    # asked of the estimator's own hook on the prepared data, which is what
    # ``_retarget_detailed`` stamps the estimates with, so the refusal and the estimates
    # the adjustment would read cannot disagree.
    prepared: dict[str, tuple[tuple[str, ...], CausalData]] = {}
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
        candidate_data = template._prepare(
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
        # The outcome-scale refusal of every fit comes before the status, as the fold
        # policy does.  Asked below, the status hook would report an undeclared scale as the
        # status of a saved result (roadmap row RM33), and its reason does not name the
        # remedy.
        template._refuse_unbounded_cross_fitted_scale(candidate_data)
        refuse_inference(
            template._inference_status(candidate_data), operation="variable_importance()"
        )
        prepared[candidate] = (adjustment_set, candidate_data)
    fits: dict[str, TMLEResult] = {}
    raw: list[tuple[str, str, tuple[str, ...], ParameterEstimate]] = []
    for candidate, (adjustment_set, candidate_data) in prepared.items():
        fitted_estimator = copy(template)
        fitted_estimator.estimands = estimand
        result = fitted_estimator.fit(candidate_data).single()
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

    adjusted = _bh_adjust([item[3].pvalue for item in raw])
    ranked = sorted(zip(raw, adjusted, strict=True), key=lambda item: float(item[1]))
    entries = [
        VariableImportanceEntry(candidate, parameter, adjustment, estimate, float(pvalue))
        for (candidate, parameter, adjustment, estimate), pvalue in ranked
    ]
    return VariableImportanceResult(
        tuple(entries),
        fits,
        backend=backend_of(as_frame(data)),
    )
