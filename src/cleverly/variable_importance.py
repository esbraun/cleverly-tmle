"""Exposure-wise variable importance from explicitly repeated target parameters.

This is orchestration, not a new influence function: candidate ``X_j`` is assigned the
treatment role in its own fit and the requested causal parameter is estimated while the
remaining declared variables are adjustment covariates.  Consequently each row states
its own intervention and adjustment set, and no shared-nuisance shortcut is implied.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .estimators.base import TMLEResult
from .estimators.tmle import TMLE
from .exceptions import DataError
from .inference.influence import ParameterEstimate
from .targets import parameter_stem
from .utils.frames import as_frame, backend_of, emit_frame, is_dataframe

__all__ = ["VariableImportanceEntry", "VariableImportanceResult", "variable_importance"]


@dataclass(frozen=True)
class VariableImportanceEntry:
    """One candidate exposure's target estimate and declared adjustment set."""

    candidate: str
    parameter: str
    adjustment_set: tuple[str, ...]
    estimate: ParameterEstimate
    adjusted_pvalue: float


@dataclass(frozen=True)
class VariableImportanceResult:
    """Candidate-exposure estimates with multiplicity-adjusted null tests."""

    entries: tuple[VariableImportanceEntry, ...]
    fits: Mapping[str, TMLEResult]
    method: str = "Benjamini-Hochberg"
    backend: str | None = None

    def __getitem__(self, index: int) -> VariableImportanceEntry:
        return self.entries[index]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VariableImportanceEntry]:
        return iter(self.entries)

    def to_frame(self) -> Any:
        """One row per candidate/parameter, sorted by adjusted p-value."""
        payload: dict[str, Any] = {
            "candidate": [entry.candidate for entry in self.entries],
            "parameter": [entry.parameter for entry in self.entries],
            "psi": [entry.estimate.psi for entry in self.entries],
            "std_err": [entry.estimate.std_error for entry in self.entries],
            "ci_lower": [entry.estimate.ci[0] for entry in self.entries],
            "ci_upper": [entry.estimate.ci[1] for entry in self.entries],
            "p_value": [entry.estimate.pvalue for entry in self.entries],
            "p_value_adjusted": [entry.adjusted_pvalue for entry in self.entries],
            "adjustment_set": [", ".join(entry.adjustment_set) for entry in self.entries],
        }
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
    fits: dict[str, TMLEResult] = {}
    raw: list[tuple[str, str, tuple[str, ...], ParameterEstimate]] = []
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
        fitted_estimator = copy(template)
        fitted_estimator.estimands = estimand
        result = fitted_estimator.fit(
            data,
            outcome=outcome,
            treatment=candidate,
            covariates=adjustment_set,
            delta=delta,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            id=id,
            treatment_kind="discrete",
        ).single()
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
    entries = [
        VariableImportanceEntry(candidate, parameter, adjustment, estimate, float(pvalue))
        for (candidate, parameter, adjustment, estimate), pvalue in zip(raw, adjusted, strict=True)
    ]
    entries.sort(key=lambda entry: entry.adjusted_pvalue)
    return VariableImportanceResult(
        tuple(entries),
        fits,
        backend=backend_of(as_frame(data)),
    )
