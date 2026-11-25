"""The assessment of a variable-importance run: each candidate's fit, asked the same question.

:attr:`cleverly.VariableImportanceResult.diagnostics` and ``.sensitivity`` route here.  A run
is a finite partition: each candidate is its own point fit, so each operation runs on each
candidate's fit through that fit's own facade, with that fit's gates, and returns a
:class:`~cleverly.assessment.CandidateReports`.  Nothing here reads the run as one point fit.

A row is composed from the candidates' rows for the same request.  It is available when every
candidate's row is, deferred when no candidate refuses and some defers, and otherwise carries
the first refusing candidate's status and record, named after that candidate.  A run with
``keep_fits=False`` keeps no fit, so every row refuses with
:data:`~cleverly.variable_importance.KEEP_FITS_REFUSAL`.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import cached_property
from typing import Any

from .assessment import (
    _FAMILY_METHODS,
    ASSESSMENT_CAPABILITIES,
    SENSITIVITY_ROUTES,
    AssessmentCapability,
    AssessmentStatus,
    CandidateReports,
    DiagnosticsFacade,
    SensitivityFacade,
    _capability,
    _CapabilityFacade,
    _require_argument_mapping,
    _single_pass,
)
from .exceptions import CapabilityError, Refusal, RefusalReason

__all__ = ["CandidateDiagnostics", "CandidateSensitivity"]

_FAMILY = "variable_importance"
_COST_RANK = {"cheap": 0, "moderate": 1, "expensive": 2}
_EXECUTION_RANK = {"summarize": 0, "retarget": 1, "refit": 2}


def _named(candidate: str, refusal: Refusal | None, row: AssessmentCapability) -> Refusal:
    """A candidate's refusal, its condition prefixed with the candidate's name."""
    record = refusal or Refusal(RefusalReason.INVALID_DECLARATION, row.interpretation)
    return replace(record, condition=f"candidate {candidate}: {record.condition}")


def _keep_fits_error() -> CapabilityError:
    from .variable_importance import KEEP_FITS_REFUSAL

    return CapabilityError(refusal=KEEP_FITS_REFUSAL)


def compose_rows(operation: str, rows: Mapping[str, AssessmentCapability]) -> AssessmentCapability:
    """One row for ``operation`` from each candidate's row for the same request.

    Parameters
    ----------
    operation : str
        The operation.
    rows : mapping of str to AssessmentCapability
        Each candidate's resolved row, in candidate order.

    Returns
    -------
    AssessmentCapability
        Available when every candidate's row is; deferred, on every argument any candidate
        waits on, when none refuses and some defers; otherwise the first refusing
        candidate's row with its record named after that candidate.  The cost, the
        execution and the seed are the most expensive candidate's.
    """

    def stamped(row: AssessmentCapability, **fields: Any) -> AssessmentCapability:
        return replace(row, result_family=_FAMILY, methods=_FAMILY_METHODS[_FAMILY], **fields)

    for candidate, row in rows.items():
        if not row.available and row.status is not AssessmentStatus.DEFERRED:
            return stamped(row, refusal=_named(candidate, row.refusal, row))
    values = list(rows.values())
    costliest = max(values, key=lambda row: (_COST_RANK[row.cost], _EXECUTION_RANK[row.execution]))
    shared = {
        "requires_arguments": tuple(
            dict.fromkeys(name for row in values for name in row.requires_arguments)
        ),
        "required_artifacts": tuple(
            dict.fromkeys(name for row in values for name in row.required_artifacts)
        ),
        "deterministic_from_saved": all(row.deterministic_from_saved for row in values),
        "accepts_random_state": any(row.accepts_random_state for row in values),
    }
    deferred = [
        (name, row) for name, row in rows.items() if row.status is AssessmentStatus.DEFERRED
    ]
    if not deferred:
        return stamped(costliest, **shared)
    candidate, row = deferred[0]
    return stamped(
        costliest,
        available=False,
        status=AssessmentStatus.DEFERRED,
        refusal=None if row.refusal is None else _named(candidate, row.refusal, row),
        **shared,
    )


class _CandidateFacade(_CapabilityFacade):
    """The composition both variable-importance facades share.

    Every row, every request and every call goes to the candidates' own facades, which apply
    their method, replay, estimand and request gates.  No gate runs at the level of the run.
    """

    #: The operations a facade of this kind declares on a point fit, for a run that keeps
    #: no fit to ask.
    _operations: tuple[str, ...]

    def _candidates(self) -> dict[str, Any]:
        """Each candidate's facade of this kind, in candidate order; empty without fits."""
        return {name: getattr(fit, self._attribute) for name, fit in self._result.kept_fits.items()}

    @cached_property
    def _facades(self) -> dict[str, Any]:
        return self._candidates()

    def _composed(self, operation: str, read: str, *arguments: Any) -> AssessmentCapability:
        """Compose what ``facade.<read>(operation, *arguments)`` gives on every candidate."""
        if not self._facades:
            return self._without_fits(operation)
        return compose_rows(
            operation,
            {
                name: getattr(facade, read)(operation, *arguments)
                for name, facade in self._facades.items()
            },
        )

    def _without_fits(self, operation: str) -> AssessmentCapability:
        from .variable_importance import KEEP_FITS_REFUSAL

        return _capability(
            operation,
            _FAMILY,
            artifacts=("candidate fits",),
            interpretation=f"each candidate fit's {operation}",
            available=False,
            status=AssessmentStatus.UNAVAILABLE,
            refusal=KEEP_FITS_REFUSAL,
        )

    def _names(self) -> tuple[str, ...]:
        if not self._facades:
            return self._operations
        return tuple(next(iter(self._facades.values()))._capability_map)

    @property
    def _declared(self) -> tuple[AssessmentCapability, ...]:
        return tuple(self._gated_map.values())

    @cached_property
    def _gated_map(self) -> dict[str, AssessmentCapability]:
        return {operation: self._composed(operation, "_gated") for operation in self._names()}

    @cached_property
    def _capability_map(self) -> dict[str, AssessmentCapability]:
        return {operation: self._composed(operation, "capability") for operation in self._names()}

    def _capability_for_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        self._lookup(self._gated_map, operation)
        return self._composed(operation, "_capability_for_arguments", arguments)

    def _estimand_candidates(self, operation: str) -> tuple[str, ...]:
        """The estimands every candidate's row would choose between, in the first's order."""
        if not self._facades:
            return ()
        each = [f._estimand_candidates(operation) for f in self._facades.values()]
        return tuple(name for name in each[0] if all(name in other for other in each[1:]))

    def _template(self) -> Any:
        """The first candidate's facade, which binds a request as every candidate does."""
        return next(iter(self._facades.values()))

    def _bind_arguments(
        self,
        operation: str,
        kwargs: Mapping[str, Any],
        *,
        partial: bool,
        args: tuple[Any, ...] = (),
        resolve: bool = True,
    ) -> inspect.BoundArguments:
        if not self._facades:
            return super()._bind_arguments(
                operation, kwargs, partial=partial, args=args, resolve=resolve
            )
        return self._template()._bind_arguments(
            operation, kwargs, partial=partial, args=args, resolve=resolve
        )

    def _effective_arguments(
        self,
        operation: str,
        kwargs: Mapping[str, Any],
        report: Any,
        *,
        args: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        if not self._facades:
            # A run without fits has no signature to bind against; its rows refuse first.
            return dict(kwargs)
        first = next(iter(self._facades))
        own = report[first] if isinstance(report, CandidateReports) else None
        effective: dict[str, Any] = self._template()._effective_arguments(
            operation, kwargs, own, args=args
        )
        return effective

    def _validated_arguments(
        self,
        arguments: Mapping[str, Mapping[str, Any]] | None,
        random_state: int | None,
    ) -> dict[str, dict[str, Any]]:
        """Each candidate's facade clears the request; a run without fits checks its shape."""
        if self._facades:
            cleared: dict[str, dict[str, Any]] = {}
            for facade in self._facades.values():
                cleared = facade._validated_arguments(arguments, random_state)
            return cleared
        _require_argument_mapping(arguments)
        if arguments is None:
            return {}
        unknown = sorted(set(arguments) - set(self._operations))
        if unknown:
            raise KeyError(f"unknown {self._kind} operation(s): {unknown}")
        validated: dict[str, dict[str, Any]] = {}
        for operation, values in arguments.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"arguments[{operation!r}] must be a mapping")
            _, validated[operation] = _single_pass((), values)
        return validated

    def _require(
        self,
        operation: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        called_as: str | None = None,
    ) -> AssessmentCapability:
        """Refuse a call that some candidate's own call refuses, naming that candidate.

        ``called_as`` is accepted for the base signature; the candidate's refusal names the
        operation its own facade was asked for.
        """
        del called_as
        self._require_call(operation, (), arguments or {})
        return self._capability_for_arguments(operation, arguments or {})

    def _require_call(
        self, operation: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        self._lookup(self._gated_map, operation)
        if not self._facades:
            raise _keep_fits_error()
        for name, facade in self._facades.items():
            try:
                facade._require_call(operation, args, kwargs)
            except CapabilityError as error:
                raise self._candidate_error(operation, name, facade, error) from error

    def _candidate_error(
        self, operation: str, name: str, facade: Any, error: CapabilityError
    ) -> CapabilityError:
        """A candidate's refusal, raised as this facade raises the composed row's.

        A candidate facade that refuses by its row puts ``"<kind> '<operation>' is
        <status>: "`` before the condition, or ``"<kind> '<preview>' previews
        '<operation>', which is <status>: "`` from a preview.  The composed row's record
        names the candidate inside that prefix, so the raise reads ``"<kind> '<operation>' is
        <status>: candidate <name>: <condition>"``, which is the composed row's message under
        the same prefix.  A record raised bare by the call itself stays bare, named after the
        candidate.
        """
        record = _named(name, error.refusal, self._gated(operation))
        bare = (error.refusal.condition if error.refusal is not None else "") or ""
        # A preview (``planned_refits`` of ``refute``) names itself before the operation.
        match = re.match(
            rf"{re.escape(facade._kind)} (?:'\w+' previews {re.escape(repr(operation))}, which "
            rf"|{re.escape(repr(operation))} )is (\w+): ",
            bare,
        )
        if match is None:
            return CapabilityError.from_refusal(record)
        condition = (
            f"{self._kind} {operation!r} is {match.group(1)}: candidate {name}: "
            f"{bare[match.end() :]}"
        )
        return CapabilityError.from_refusal(replace(record, condition=condition))

    def _invoke(self, operation: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        """Run ``operation`` on every candidate's fit, with the same arguments and seed.

        Every candidate's refusal is asked first, so no candidate refits before another
        refuses.  A candidate whose call still refuses makes the whole call refuse, named
        after it, so no partial report is returned.
        """
        args, kwargs = _single_pass(args, kwargs)
        self._require_call(operation, args, kwargs)
        reports = {}
        for name, facade in self._facades.items():
            try:
                reports[name] = getattr(facade, operation)(
                    *args, **_candidate_checkpoint(operation, name, kwargs, record=True)
                )
            except CapabilityError as error:
                raise self._candidate_error(operation, name, facade, error) from error
        return CandidateReports(operation, reports, self._result.backend)


def _candidate_checkpoint(
    operation: str, name: str, kwargs: Mapping[str, Any], *, record: bool
) -> dict[str, Any]:
    """``kwargs`` for one candidate: its own checkpoint subdirectory, its name on progress.

    One directory records one request, so each candidate keeps its draws in a subdirectory of
    the one passed (X106 G3), and each progress record names the candidate.
    """
    out = dict(kwargs)
    if operation != "refute":
        return out
    checkpoint = out.get("checkpoint")
    if checkpoint is not None:
        out["checkpoint"] = checkpoint.child(name, record=record)
    progress = out.get("progress")
    if progress is not None:
        out["progress"] = lambda record_: progress(replace(record_, candidate=name))
    return out


class CandidateDiagnostics(_CandidateFacade, DiagnosticsFacade):
    """The diagnostics of a variable-importance run, read on each candidate's fit.

    Each operation runs on ``result.fits[candidate].diagnostics`` for every candidate and
    returns a :class:`~cleverly.assessment.CandidateReports`.  Each report uses that
    candidate's own estimate and unadjusted interval, not the Benjamini-Hochberg adjusted
    test.

    Parameters
    ----------
    result : VariableImportanceResult
        The run.
    """

    _operations = tuple(
        dict.fromkeys(
            row.operation for row in ASSESSMENT_CAPABILITIES if row.result_family == "point"
        )
    )

    def _routed_callable(self, operation: str) -> tuple[Callable[..., Any], bool]:
        function, result_first = super()._routed_callable(operation)
        if not result_first:
            # The signature of the call each candidate runs, not this facade's forwarder.
            function = getattr(DiagnosticsFacade, operation)
        return function, result_first

    def support(self) -> CandidateReports:
        """Each candidate fit's support report.

        Returns
        -------
        CandidateReports
            One support report per candidate.
        """
        return self._invoke("support", (), {})

    def nuisance_models(self) -> CandidateReports:
        """Each candidate fit's held-out nuisance diagnostics.

        Returns
        -------
        CandidateReports
            One nuisance report per candidate.
        """
        return self._invoke("nuisance_models", (), {})

    def score_equations(self, **kwargs: Any) -> CandidateReports:
        """Each candidate fit's score-equation check.

        Parameters
        ----------
        **kwargs
            Passed to each fit's ``diagnostics.score_equations``.

        Returns
        -------
        CandidateReports
            One score check per candidate.
        """
        return self._invoke("score_equations", (), kwargs)

    def corrections(self, **kwargs: Any) -> CandidateReports:
        """Each candidate fit's correction-equation check.

        Parameters
        ----------
        **kwargs
            Passed to each fit's ``diagnostics.corrections``.

        Returns
        -------
        CandidateReports
            One correction check per candidate.
        """
        return self._invoke("corrections", (), kwargs)

    def truncation_curve(self, *args: Any, **kwargs: Any) -> CandidateReports:
        """Each candidate fit's truncation curve.

        Parameters
        ----------
        *args, **kwargs
            Passed to each fit's ``diagnostics.truncation_curve``.

        Returns
        -------
        CandidateReports
            One curve per candidate.
        """
        return self._invoke("truncation_curve", args, kwargs)

    def refute(self, **kwargs: Any) -> CandidateReports:
        """Each candidate fit's refutation, with the same tests and seed.

        Parameters
        ----------
        **kwargs
            Passed to each fit's ``diagnostics.refute``. With ``sample_limit=``, each
            candidate refutes its own sample; the candidates draw the same units only when
            they share a root seed, that is under an explicit ``random_state=`` or seeded
            fits.

        Returns
        -------
        CandidateReports
            One refutation result per candidate.
        """
        return self._invoke("refute", (), kwargs)

    def planned_refits(self, **kwargs: Any) -> CandidateReports:
        """Each candidate fit's planned refits, counted as :meth:`refute` would make them.

        Parameters
        ----------
        **kwargs
            Passed to each fit's ``diagnostics.planned_refits``.

        Returns
        -------
        CandidateReports
            One :class:`~cleverly.validation.PlannedRefits` per candidate. Its ``to_frame``
            stacks the candidates' test rows.
        """
        _, kwargs = _single_pass((), kwargs)
        try:
            self._require_call("refute", (), kwargs)
        except CapabilityError as error:
            raise self._as_preview(error) from error
        reports = {}
        for name, facade in self._facades.items():
            try:
                reports[name] = facade.planned_refits(
                    **_candidate_checkpoint("refute", name, kwargs, record=False)
                )
            except CapabilityError as error:
                composed = self._candidate_error("refute", name, facade, error)
                raise self._as_preview(composed) from error
        return CandidateReports("planned_refits", reports, self._result.backend)

    def _as_preview(self, error: CapabilityError) -> CapabilityError:
        """The composed refusal of ``refute``, named as :meth:`planned_refits` previews it.

        The per-fit facade names the preview (``"<kind> 'planned_refits' previews
        'refute', which is <status>: "``), so the composed call does too.
        """
        refusal = error.refusal
        prefix = f"{self._kind} 'refute' is "
        if refusal is None or not (refusal.condition or "").startswith(prefix):
            return error
        condition = (
            f"{self._kind} 'planned_refits' previews 'refute', which is "
            f"{refusal.condition[len(prefix) :]}"
        )
        return CapabilityError.from_refusal(replace(refusal, condition=condition))

    def influence(self) -> Any:
        """Refused: each candidate's influence curves are its own fit's.

        Returns
        -------
        InfluenceReport
            Never; the call always raises.

        Raises
        ------
        CapabilityError
            An ``API_REDIRECTION`` to ``result.fits[candidate].diagnostics.influence()``, or
            the ``keep_fits=False`` refusal on a run that kept no fit.
        """
        if not self._facades:
            raise _keep_fits_error()
        raise CapabilityError(
            "a variable-importance run holds one fit per candidate, and each fit's estimates "
            "carry its own influence curves.",
            reason=RefusalReason.API_REDIRECTION,
            alternative="Call result.fits[candidate].diagnostics.influence() for one candidate.",
        )


class CandidateSensitivity(_CandidateFacade, SensitivityFacade):
    """The sensitivity analyses of a variable-importance run, run on each candidate's fit.

    Each analysis runs on ``result.fits[candidate].sensitivity`` for every candidate and
    returns a :class:`~cleverly.assessment.CandidateReports`.  A bound or a benchmark of one
    candidate reads that candidate's own estimate and unadjusted interval, never the
    Benjamini-Hochberg adjusted test.

    Parameters
    ----------
    result : VariableImportanceResult
        The run.
    """

    _operations = tuple(SENSITIVITY_ROUTES)

    def _dispatch(self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return self._invoke(operation, args, kwargs)

    def _with_default_parameter(
        self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not self._facades:
            return args, kwargs
        substituted: tuple[tuple[Any, ...], dict[str, Any]] = (
            self._template()._with_default_parameter(operation, args, kwargs)
        )
        return substituted
