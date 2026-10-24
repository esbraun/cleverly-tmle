"""The plain facts and summaries of the method records a point fit keeps (X89, thread T1).

A collaborative fit keeps its search (``result.extra["ctmle"]``), a DR-TMLE fit its extra
regressions (``extra["drtmle"]``) and an Adaptive IPTW fit its weight adjustment
(``extra["adaptive_iptw"]``). :func:`method_facts` reads those records, not the estimator,
so a loaded result with no estimator prints the same summary as the fit that made it.
Each record also prints its own summary, built here.

Labels and note texts come from :mod:`cleverly._summary.vocabulary`. The records'
``describe()`` sentences stay for the reports of other modules; no summary prints them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._inference_status import supplies_inference
from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    count_text,
    decimals_for,
    note,
    number_text,
    short_repr,
    vocabulary,
)
from .._summary._cells import number_cell
from .._summary._estimates import estimates_table, plain_support_reason

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import CVTargeting, TMLEResult, TMLEResultSet
    from .ctmle import CTMLEDerivativeFit, CTMLEOutcomeAdaptiveFit, CTMLESelection
    from .drtmle import ReducedFit
    from .iptw import AdaptiveIPTWFit

#: A search table longer than this shows only its landmark candidates (``rows_shown``).
_MOST_CANDIDATES = 20
#: The most covariate names a treatment-model fact lists before it counts the rest.
_MOST_NAMES = 8
_SHOWN_NAMES = 6
#: The strategies whose candidates are labelled rather than covariate sets.
_INDEXED = ("lasso", "learners")


def _listed(words: Sequence[str]) -> str:
    items = list(words)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _names_text(names: Sequence[str]) -> str:
    if len(names) > _MOST_NAMES:
        return f"{', '.join(names[:_SHOWN_NAMES])} and {len(names) - _SHOWN_NAMES} more"
    return ", ".join(names)


@dataclass(frozen=True)
class MethodFacts:
    """What a method record adds to a point summary.

    Parameters
    ----------
    method : str or None
        The Method header value, or ``None`` to keep the method's plain name.
    treatment_model : str or None
        The Treatment model header value, or ``None`` to keep the learner's name.
    header : tuple of tuple of str
        More header facts.
    details : tuple of str
        The "Method details" lines.
    notes : tuple of Note
        More notes.
    """

    method: str | None = None
    treatment_model: str | None = None
    header: tuple[tuple[str, str], ...] = ()
    details: tuple[str, ...] = ()
    notes: tuple[Note, ...] = ()


def method_facts(result: TMLEResult, treatment_learner: str | None) -> MethodFacts:
    """The plain facts of the method records ``result.extra`` holds.

    Parameters
    ----------
    result : TMLEResult
        The fitted result.
    treatment_learner : str or None
        The plain name of the treatment learner, when the fit records one.

    Returns
    -------
    MethodFacts
        The facts; empty for an ordinary TMLE fit.
    """
    from .ctmle import CTMLEDerivativeFit, CTMLEOutcomeAdaptiveFit, CTMLESelection

    record = result.extra.get("ctmle")
    data = result.data
    if isinstance(record, CTMLESelection):
        return _selection_facts(record, treatment_learner)
    if isinstance(record, CTMLEOutcomeAdaptiveFit):
        return _oat_facts(record, data)
    if isinstance(record, CTMLEDerivativeFit):
        return _derivative_facts(record)
    reduced = result.extra.get("drtmle")
    if reduced is not None:
        return _reduced_facts(reduced, data.treatment_adjustment_sources)
    iptw = result.extra.get("adaptive_iptw")
    if iptw is not None:
        everything = tuple(
            dict.fromkeys(data.indicator_sources.get(n, n) for n in data.covariate_names)
        )
        return _iptw_facts(iptw, data.treatment_adjustment_sources, everything)
    return MethodFacts()


# ----------------------------------------------------------------- collaborative search


def _search_words(selection: CTMLESelection) -> str:
    words = vocabulary.CTMLE_SEARCHES.get(selection.strategy, selection.strategy)
    if selection.strategy == "ordered" and selection.preorder is not None:
        order = vocabulary.CTMLE_PREORDERS.get(selection.preorder, selection.preorder)
        words += f", {order}"
    return words


def _collaborative(words: str) -> str:
    return f"{vocabulary.METHOD_NAMES['collaborative_tmle']}, {words}"


def _criterion(selection: CTMLESelection) -> str:
    loss = vocabulary.LOSS_WORDS.get(selection.loss, selection.loss)
    penalty = ", penalized for variance and bias" if selection.penalized else ""
    return f"cross-validated {loss}{penalty}"


def _candidate_learner(label: str) -> str:
    """The plain learner name of a ``learners`` candidate label such as ``"Ridge[1]"``."""
    kind = label.split("[", 1)[0]
    if kind == "arrays":
        return "declared probabilities"
    return vocabulary.LEARNER_NAMES.get(kind, kind)


def _selection_model(selection: CTMLESelection, learner: str | None) -> str:
    if selection.strategy == "lasso":
        return "lasso logistic regression at the chosen penalty"
    if selection.strategy == "learners":
        if not selection.labels:
            return learner or vocabulary.DEFAULT_LEARNER
        return _candidate_learner(selection.labels[selection.reported])
    chosen = selection.selected_covariates
    name = learner or vocabulary.DEFAULT_LEARNER
    return f"{name} on {_names_text(chosen)}" if chosen else f"{name}, intercept only"


def _chosen(selection: CTMLESelection) -> str:
    text = f"candidate {selection.reported + 1} of {len(selection.path)}"
    return text + " (final step)" if selection.final is not None else text


def _patience_text(selection: CTMLESelection) -> str | None:
    if selection.patience is None:
        return None
    if selection.stopped_at is None:
        return vocabulary.NOTE_TEXTS["search_never_stopped"].format(patience=selection.patience)
    later = "fitted but not scored" if selection.strategy in _INDEXED else "not fitted or scored"
    count = "1 candidate" if selection.patience == 1 else f"{selection.patience} candidates"
    return vocabulary.NOTE_TEXTS["search_stopped"].format(
        stopped=selection.stopped_at + 1, patience=count, later=later
    )


def _left_out_text(selection: CTMLESelection) -> str | None:
    if selection.strategy in _INDEXED or not selection.dropped:
        return None
    return vocabulary.NOTE_TEXTS["left_out"].format(names=_names_text(selection.dropped))


def _risk_order_text(selection: CTMLESelection) -> str | None:
    if selection.strategy != "learners" or not selection.treatment_risk_increases:
        return None
    return vocabulary.NOTE_TEXTS["risk_order"].format(
        names=_listed([str(k + 1) for k in selection.treatment_risk_increases])
    )


def _search_sentence(selection: CTMLESelection) -> str:
    sentence = (
        f"The search built {len(selection.path)} candidate treatment models "
        f"({_search_words(selection)}). It chose candidate {selection.selected + 1}, the one "
        f"whose adjusted outcome predictions had the lowest {_criterion(selection)}."
    )
    if selection.final is not None:
        sentence += f" A final step then used candidate {selection.final + 1}."
    return sentence


def _selection_facts(selection: CTMLESelection, learner: str | None) -> MethodFacts:
    details = [_search_sentence(selection)]
    for text in (_left_out_text(selection), _patience_text(selection)):
        if text is not None:
            details.append(text)
    if selection.restricted_pool is not None:
        details.append(vocabulary.SEARCH_POOL.format(names=_names_text(selection.restricted_pool)))
    order = _risk_order_text(selection)
    return MethodFacts(
        method=_collaborative(_search_words(selection)),
        treatment_model=_selection_model(selection, learner),
        header=((vocabulary.CHOSEN, _chosen(selection)),),
        details=tuple(details),
        notes=(note("risk_order", order),) if order is not None else (),
    )


def _shown_candidates(selection: CTMLESelection) -> list[int]:
    """The candidates a search table prints: all of a short path, landmarks of a long one."""
    count = len(selection.path)
    if count <= _MOST_CANDIDATES:
        return list(range(count))
    keep = {0, count - 1, selection.selected}
    if selection.final is not None:
        keep.add(selection.final)
    keep.update(k for k in range(1, count) if selection.path[k] != selection.path[k - 1])
    return sorted(keep)


def selection_summary(selection: CTMLESelection, detail: Detail) -> Summary:
    """The summary of a collaborative search.

    Parameters
    ----------
    selection : CTMLESelection
        The search record.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    shown = _shown_candidates(selection)
    decimals = decimals_for(
        [
            abs(float(a) - float(b))
            for a, b in zip(selection.cv_risk[1:], selection.cv_risk[:-1], strict=True)
        ]
        or [float(selection.cv_risk[0])]
    )
    decimals = max(decimals, 3)
    rows = []
    for index in shown:
        mark = ""
        if index == selection.final:
            mark = vocabulary.FINAL_MARK
        elif index == selection.selected:
            mark = vocabulary.CHOSEN_MARK
        names = selection.path[index]
        model = (
            selection._label(index)
            if selection.strategy in _INDEXED
            else (", ".join(names) if names else "intercept only")
        )
        train, cv = float(selection.train_risk[index]), float(selection.cv_risk[index])
        rows.append(
            (
                Cell(index, str(index + 1)),
                Cell(model, model),
                number_cell(train, decimals),
                number_cell(cv, decimals),
                Cell(mark, mark),
            )
        )
    table = SummaryTable(
        (
            "candidate",
            vocabulary.TREATMENT_MODEL_COLUMN,
            vocabulary.IN_SAMPLE_RISK,
            vocabulary.CV_RISK,
            vocabulary.CHOSEN.lower(),
        ),
        tuple(rows),
        kind="other",
    )
    header = [(vocabulary.SEARCH, _search_words(selection))]
    if tuple(selection.target_names) != (selection.estimand,):
        header.append(
            (vocabulary.ESTIMAND, f"{selection.estimand} ({', '.join(selection.target_names)})")
        )
    else:
        header.append((vocabulary.ESTIMAND, selection.estimand))
    pool = _names_text(selection.covariates) or "none"
    if selection.restricted_pool is not None:
        pool += " (declared)"
    header += [
        (vocabulary.CRITERION, _criterion(selection)),
        (vocabulary.CANDIDATES, count_text(len(selection.path))),
        (vocabulary.CHOSEN, _chosen(selection)),
        (vocabulary.FOLDS, count_text(selection.folds.n_folds)),
        (vocabulary.COVARIATES_SEARCHED, pool),
    ]
    notes: list[Note] = []
    for code, text in (
        ("left_out", _left_out_text(selection)),
        ("search_stopped", _patience_text(selection)),
        ("risk_order", _risk_order_text(selection)),
    ):
        if text is not None:
            notes.append(note(code, text))
    if len(shown) < len(selection.path):
        notes.append(
            note(
                "rows_shown",
                vocabulary.NOTE_TEXTS["rows_shown"].format(
                    shown=len(shown), total=len(selection.path)
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    "Each risk is the loss of the outcome predictions after the adjustment "
                    "step that uses the candidate's treatment model. The in-sample risk "
                    "uses the rows each model was fitted on; the CV risk uses held-out rows "
                    "and is the one the search minimizes.",
                    "The outcome model sees every covariate; only the treatment model is searched.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.SEARCH_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def selection_repr(selection: CTMLESelection) -> str:
    """The one-line repr of a collaborative search.

    Parameters
    ----------
    selection : CTMLESelection
        The search record.

    Returns
    -------
    str
        For example ``CTMLESelection(search=greedy, candidates=4, chosen=3)``.
    """
    return short_repr(
        "CTMLESelection",
        [
            ("search", selection.strategy),
            ("candidates", len(selection.path)),
            ("chosen", selection.reported + 1),
        ],
    )


# ----------------------------------------------------------------- outcome-adaptive


def _factor_words(factor: str) -> str:
    for start, words in vocabulary.OAT_FACTORS.items():
        if factor.startswith(start):
            return words
    return factor


def _stacked_note(fit: CTMLEOutcomeAdaptiveFit) -> Note | None:
    if fit.stacked_interval is not None:
        weights = ""
        if "declared estimated" in fit.stacked_interval:
            weights = vocabulary.NOTE_TEXTS["stacked_weights_estimated"]
        elif "data weights" in fit.stacked_interval:
            weights = vocabulary.NOTE_TEXTS["stacked_weights"]
        return note(
            "stacked_interval", vocabulary.NOTE_TEXTS["stacked_interval"].format(weights=weights)
        )
    if fit.stacked_refusal is not None:
        return note(
            "stacked_refused",
            vocabulary.NOTE_TEXTS["stacked_refused"].format(
                reason=plain_support_reason(fit.stacked_refusal).rstrip(".") + "."
            ),
        )
    return None


def _stacked_lines(fit: CTMLEOutcomeAdaptiveFit) -> list[str]:
    """How the stacked interval of an admitted fit was built, or why there is none."""
    if fit.stacked_refusal is not None:
        reason = plain_support_reason(fit.stacked_refusal).rstrip(".")
        return [vocabulary.NOTE_TEXTS["stacked_refused"].format(reason=reason) + "."]
    if fit.stacked_interval is None:
        return []
    lines = []
    if (fit.stacked_folds or 1) > 1:
        lines.append(
            f"The stacked equation is cross-fitted over {fit.stacked_folds} folds, with one "
            "outcome model and one treatment model per fold."
        )
    if (fit.stacked_repeats or 1) > 1:
        lines.append(
            f"Each estimate is the median of {fit.stacked_repeats} cross-fitting repeats, "
            "and its std. error allows for the spread across them."
        )
    lines.append(
        "A simultaneous band covers the estimates that have an interval jointly, from the "
        "same stacked equation."
    )
    return lines


def _oat_design(fit: CTMLEOutcomeAdaptiveFit) -> str:
    if fit.design == "per_arm":
        return "one per arm, on outcome predictions"
    return "one, on every arm's outcome prediction"


def _oat_sentence(fit: CTMLEOutcomeAdaptiveFit) -> str:
    if fit.design == "per_arm":
        return (
            "The treatment model is one model per arm, each fitted on the outcome model's "
            "prediction for its arm."
        )
    return (
        "The treatment model is one model fitted on the outcome model's predictions for every arm."
    )


def _oat_facts(fit: CTMLEOutcomeAdaptiveFit, data: Any) -> MethodFacts:
    details = [_oat_sentence(fit)]
    if len(fit.factors) > 1:
        details.append(
            "Each arm's model has "
            + str(len(fit.factors))
            + " parts: "
            + ", ".join(_factor_words(factor) for factor in fit.factors)
            + "."
        )
    if fit.strata:
        details.append(f"It is fitted inside each of {len(fit.strata)} baseline strata.")
    details.extend(_stacked_lines(fit))
    stacked = _stacked_note(fit) if fit.stacked_interval is not None else None
    return MethodFacts(
        method=_collaborative(vocabulary.CTMLE_SEARCHES["oat"]),
        treatment_model=_oat_design(fit),
        details=tuple(details),
        notes=(stacked,) if stacked is not None else (),
    )


def oat_summary(fit: CTMLEOutcomeAdaptiveFit, detail: Detail) -> Summary:
    """The summary of an outcome-adaptive treatment model.

    Parameters
    ----------
    fit : CTMLEOutcomeAdaptiveFit
        The record.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    risks = (
        {str(arm): float(value) for arm, value in fit.treatment_risk_by_arm.items()}
        if fit.treatment_risk_by_arm is not None
        else {"every arm": float(fit.treatment_risk)}
    )
    decimals = max(3, decimals_for([abs(v) for v in risks.values()]))
    column = "treatment-model risk"
    tables = [
        SummaryTable(
            ("arm", column),
            tuple((Cell(arm, arm), number_cell(value, decimals)) for arm, value in risks.items()),
            kind="other",
        )
    ]
    if len(fit.factors) > 1 and fit.risk_by_factor is not None:
        tables.append(
            SummaryTable(
                ("part", column),
                tuple(
                    (
                        Cell(factor, _factor_words(factor)),
                        Cell(
                            float(fit.risk_by_factor[factor]),
                            number_text(fit.risk_by_factor[factor], decimals),
                        ),
                    )
                    for factor in fit.factors
                ),
                title="Risk of each part, summed over arms",
                kind="other",
            )
        )
    header = [(vocabulary.DESIGN, _oat_design(fit))]
    if fit.strata:
        header.append((vocabulary.STRATA, _names_text(fit.strata)))
    stacked = _stacked_note(fit)
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    "The risk is the negative log-likelihood of the treatment model, "
                    "weighted as the fit is.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.OAT_TITLE,
        header=header,
        tables=tables,
        notes=[stacked] if stacked is not None else [],
        sections=sections,
    )


def oat_repr(fit: CTMLEOutcomeAdaptiveFit) -> str:
    """The one-line repr of an outcome-adaptive treatment model.

    Parameters
    ----------
    fit : CTMLEOutcomeAdaptiveFit
        The record.

    Returns
    -------
    str
        For example ``CTMLEOutcomeAdaptiveFit(design=per_arm, risk=1.31)``.
    """
    facts: list[tuple[str, Any]] = [
        ("design", fit.design),
        ("risk", f"{fit.treatment_risk:.3g}"),
    ]
    if fit.stacked_interval is not None:
        facts.append(("", "stacked interval"))
    return short_repr("CTMLEOutcomeAdaptiveFit", facts)


# ----------------------------------------------------------------- derivative form


def _derivative_model(fit: CTMLEDerivativeFit) -> str:
    if fit.source == "lasso" and fit.penalties:
        chosen = sorted({pair[0] for pair in fit.penalties})
        if len(chosen) == 1:
            return f"lasso logistic regression at penalty {chosen[0]:.3g}"
        return (
            f"lasso logistic regression at penalties {chosen[0]:.3g} to {chosen[-1]:.3g} "
            f"over {len(fit.penalties)} training sets"
        )
    return vocabulary.DERIVATIVE_SOURCES.get(fit.source, fit.source)


def _derivative_sentence(fit: CTMLEDerivativeFit) -> str:
    partner = ""
    partners = sorted({pair[1] for pair in fit.penalties})
    if fit.source == "lasso" and len(partners) == 1:
        partner = f" at penalty {partners[0]:.3g}"
    return (
        f"A second, partner treatment model{partner} adds one more equation per arm, which "
        "the fit solves along with the usual one."
    )


def _derivative_facts(fit: CTMLEDerivativeFit) -> MethodFacts:
    return MethodFacts(
        method=_collaborative(vocabulary.CTMLE_SEARCHES["derivative"]),
        treatment_model=_derivative_model(fit),
        details=(_derivative_sentence(fit),),
    )


def derivative_summary(fit: CTMLEDerivativeFit, detail: Detail) -> Summary:
    """The summary of a derivative-form collaborative fit.

    Parameters
    ----------
    fit : CTMLEDerivativeFit
        The record.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    before = [float(v) for v in fit.derivative_score_before]
    after = [float(v) for v in fit.derivative_score_after]
    decimals = max(3, decimals_for([abs(v) for v in before if math.isfinite(v)] or [1.0]))
    tables = [
        SummaryTable(
            ("arm", "score before", "score after"),
            tuple(
                (
                    Cell(k, f"arm {k}"),
                    number_cell(b, decimals),
                    number_cell(a, decimals),
                )
                for k, (b, a) in enumerate(zip(before, after, strict=True))
            ),
            title="Partner equation score, before and after the adjustment",
            kind="other",
        )
    ]
    if fit.source == "lasso" and fit.penalties:
        tables.append(
            SummaryTable(
                ("training set", "penalty", "partner penalty"),
                tuple(
                    (
                        Cell(k, str(k + 1)),
                        Cell(float(m), f"{m:.3g}"),
                        Cell(float(p), f"{p:.3g}"),
                    )
                    for k, (m, p) in enumerate(fit.penalties)
                ),
                title="Lasso penalties",
                kind="other",
            )
        )
    header = [
        (vocabulary.SOURCE, vocabulary.DERIVATIVE_SOURCES.get(fit.source, fit.source)),
        (vocabulary.TREATMENT_MODEL, _derivative_model(fit)),
    ]
    if fit.source == "lasso":
        header.append((vocabulary.TRAINING_SETS, count_text(len(fit.penalties))))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    _derivative_sentence(fit),
                    "The score measures how far the outcome predictions are from solving "
                    "that equation; the fit drives it toward zero.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.DERIVATIVE_TITLE, header=header, tables=tables, sections=sections
    )


def derivative_repr(fit: CTMLEDerivativeFit) -> str:
    """The one-line repr of a derivative-form collaborative fit.

    Parameters
    ----------
    fit : CTMLEDerivativeFit
        The record.

    Returns
    -------
    str
        For example ``CTMLEDerivativeFit(source=lasso, training sets=1)``.
    """
    facts: list[tuple[str, Any]] = [("source", fit.source)]
    if fit.source == "lasso":
        facts.append(("training sets", len(fit.penalties)))
    return short_repr("CTMLEDerivativeFit", facts)


# ----------------------------------------------------------------- DR-TMLE


def _guard_text(fit: ReducedFit) -> str:
    if not fit.guard:
        return "nothing (no extra equations)"
    return ", ".join(vocabulary.GUARD_WORDS.get(g, g) for g in fit.guard)


def _guard_note(fit: ReducedFit) -> Note:
    if not fit.guard:
        return note("ordinary_tmle", vocabulary.NOTE_TEXTS["ordinary_tmle"])
    return note("guard_scope", " ".join(vocabulary.GUARD_SENTENCES.get(g, g) for g in fit.guard))


def _reduced_lines(fit: ReducedFit, sources: Sequence[str] | None) -> list[str]:
    if not fit.guard:
        return []
    words = vocabulary.REDUCTION_WORDS.get(fit.reduction, fit.reduction)
    lines = [f"The extra equations come from the regressions in the table ({words})."]
    if fit.missing_data == "composite":
        lines.append(
            "Missing treatments and missing outcomes enter through one combined indicator "
            "of a fully recorded row in the arm."
        )
    if sources is not None:
        lines.append(
            vocabulary.DRTMLE_RESTRICTED.format(names=_names_text(sources) or "no covariates")
        )
    return lines


def _reduced_facts(fit: ReducedFit, sources: Sequence[str] | None) -> MethodFacts:
    return MethodFacts(
        header=((vocabulary.GUARD, _guard_text(fit)),),
        details=tuple(_reduced_lines(fit, sources)),
        notes=(_guard_note(fit),),
    )


def _best(diagnostics: Any) -> tuple[str, float]:
    """The lowest-risk candidate of one super learner fit and its cross-validated risk."""
    names = list(diagnostics.names)
    if diagnostics.best in names:
        return diagnostics.best, float(diagnostics.cv_risk[names.index(diagnostics.best)])
    return str(diagnostics.best), math.nan


def reduced_summary(fit: ReducedFit, detail: Detail) -> Summary:
    """The summary of the DR-TMLE extra regressions.

    Parameters
    ----------
    fit : ReducedFit
        The record.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary. A regression fitted by a super learner names the best candidate
        of its first fit (the first arm, stratum and round) and that candidate's
        cross-validated risk.
    """
    rows = []
    for key, entries in fit.diagnostics.items():
        name = vocabulary.REDUCED_REGRESSIONS.get(key, key)
        best, risk = _best(entries[0]) if entries else ("", math.nan)
        rows.append(
            (
                Cell(key, name),
                Cell(best, best or "not a super learner"),
                number_cell(risk, 4),
            )
        )
    if not rows:
        rows.append((Cell("none", "none"), Cell("", ""), Cell(None, "")))
    table = SummaryTable(
        ("regression", vocabulary.BEST.lower(), vocabulary.CV_RISK),
        tuple(rows),
        kind="other",
    )
    low, high = fit.g_bounds
    header = [
        (vocabulary.GUARD, _guard_text(fit)),
        (
            vocabulary.REDUCTION,
            vocabulary.REDUCTION_WORDS.get(fit.reduction, fit.reduction) if fit.guard else "none",
        ),
        (vocabulary.PROBABILITY_LIMITS, f"[{low:.4g}, {high:.4g}]"),
    ]
    sections: list[SummarySection] = []
    if detail == "full":
        lines = _reduced_lines(fit, None) or ["No extra regressions were fitted."]
        if fit.missingness_bound is not None:
            lines.append(
                f"The recording probabilities of two regressions are bounded below at "
                f"{fit.missingness_bound:.4g}."
            )
        sections.append(SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines)))
    return Summary.build(
        title=vocabulary.REDUCED_TITLE,
        header=header,
        tables=[table],
        notes=[_guard_note(fit)],
        sections=sections,
    )


def reduced_repr(fit: ReducedFit) -> str:
    """The one-line repr of the DR-TMLE extra regressions.

    Parameters
    ----------
    fit : ReducedFit
        The record.

    Returns
    -------
    str
        For example ``ReducedFit(guard=Q+g, reduction=univariate, regressions=3)``.
    """
    return short_repr(
        "ReducedFit",
        [
            ("guard", "+".join(fit.guard) or "none"),
            ("reduction", fit.reduction),
            ("regressions", len(fit.diagnostics)),
        ],
    )


# ----------------------------------------------------------------- Adaptive IPTW


def _rounds_note(fit: AdaptiveIPTWFit) -> Note | None:
    if fit.stop != "cap":
        return None
    rounds = "1 round" if fit.rounds == 1 else f"{fit.rounds} rounds"
    return note("rounds_capped", vocabulary.NOTE_TEXTS["rounds_capped"].format(rounds=rounds))


def _iptw_facts(
    fit: AdaptiveIPTWFit, sources: Sequence[str] | None, everything: Sequence[str]
) -> MethodFacts:
    stopped = (
        "until the weighted score of every estimate met its tolerance"
        if fit.stop == "tolerance"
        else "and stopped at the round limit before every weighted score met its tolerance"
    )
    rounds = "1 round" if fit.rounds == 1 else f"{fit.rounds} rounds"
    details = [f"The treatment model was adjusted in {rounds}, {stopped}."]
    if fit.form == "one_step":
        details.append("The estimate is the one-step correction of the weighted mean.")
    else:
        details.append("The estimate is the weighted mean at the adjusted treatment model.")
    if sources is not None:
        omitted = [name for name in everything if name not in sources]
        line = vocabulary.IPTW_RESTRICTED.format(names=_names_text(sources) or "no covariates")
        if omitted:
            line += vocabulary.IPTW_RESTRICTED_OMITTED.format(omitted=_names_text(omitted))
        details.append(line)
    capped = _rounds_note(fit)
    return MethodFacts(details=tuple(details), notes=(capped,) if capped is not None else ())


def iptw_summary(fit: AdaptiveIPTWFit, detail: Detail) -> Summary:
    """The summary of an Adaptive IPTW weight adjustment.

    Parameters
    ----------
    fit : AdaptiveIPTWFit
        The record.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary. The plug-in spreads of the two forms are left to the fitted result,
        which prints them under the label its inference status gives.
    """
    adjusted = fit.targeted if fit.form == "targeted" else fit.one_step
    names = list(fit.untargeted)
    values = [float(fit.untargeted[n]) for n in names] + [
        float(adjusted[n][0]) for n in names if n in adjusted
    ]
    decimals = max(3, decimals_for([abs(v) for v in values if math.isfinite(v)] or [1.0]))
    rows = []
    for name in names:
        first = float(fit.untargeted[name])
        final = float(adjusted[name][0]) if name in adjusted else math.nan
        rows.append(
            (
                Cell(name, f"arm {name}"),
                number_cell(first, decimals),
                number_cell(final, decimals),
            )
        )
    tables = [
        SummaryTable(("", "initial weighted mean", "adjusted estimate"), tuple(rows), kind="other")
    ]
    if any(fit.bound_active.values()):
        tables.append(
            SummaryTable(
                ("part", "rows at a probability limit"),
                tuple(
                    (Cell(key, key), Cell(int(count), count_text(int(count))))
                    for key, count in fit.bound_active.items()
                ),
                title="Rows whose adjusted treatment probability sits at a limit",
                kind="other",
            )
        )
    header = [
        (vocabulary.FORM, vocabulary.IPTW_FORMS.get(fit.form, fit.form)),
        (vocabulary.ROUNDS, count_text(fit.rounds)),
        (vocabulary.STOPPED, vocabulary.IPTW_STOPS.get(fit.stop, fit.stop)),
    ]
    capped = _rounds_note(fit)
    sections: list[SummarySection] = []
    if detail == "full":
        lines = [
            "The initial weighted mean uses the first treatment model; the adjusted "
            "estimate uses the treatment model after the last round."
        ]
        if fit.constant_design:
            lines.append(
                "These problems had a constant design in a training set, so their extra "
                f"regression was a weighted mean: {', '.join(fit.constant_design)}."
            )
        sections.append(SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines)))
    return Summary.build(
        title=vocabulary.IPTW_TITLE,
        header=header,
        tables=tables,
        notes=[capped] if capped is not None else [],
        sections=sections,
    )


def iptw_repr(fit: AdaptiveIPTWFit) -> str:
    """The one-line repr of an Adaptive IPTW weight adjustment.

    Parameters
    ----------
    fit : AdaptiveIPTWFit
        The record.

    Returns
    -------
    str
        For example ``AdaptiveIPTWFit(form=targeted, rounds=4, stopped=score tolerance met)``.
    """
    return short_repr(
        "AdaptiveIPTWFit",
        [
            ("form", fit.form),
            ("rounds", fit.rounds),
            ("stopped", vocabulary.IPTW_STOPS.get(fit.stop, fit.stop)),
        ],
    )


# ----------------------------------------------------------------- cross-validated report


def cv_summary(report: CVTargeting, detail: Detail) -> Summary:
    """The summary of the fold-level report of a cross-validated fit.

    Parameters
    ----------
    report : CVTargeting
        The report.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    from ..inference.influence import spread_name

    status = report.inference
    inferential = supplies_inference(status)
    cv_label = vocabulary.CV_SPREAD_LABELS[spread_name("cv_std_err", status)]
    whole_label = vocabulary.CV_SPREAD_LABELS[spread_name("pooled_std_err", status)]
    errors = report.plugin_std_error
    names = list(report.variance)
    decimals = decimals_for([errors[n] for n in names])
    rows = []
    spread_rows = []
    for name in names:
        folds = report.fold_estimates.get(name, ())
        sd = float(np.std(folds, ddof=1)) if len(folds) > 1 else math.nan
        averaged = float(report.fold_evaluated[name].psi)
        whole = float(report.pooled[name].psi)
        whole_se = float(report.pooled[name].plugin_std_error)
        rows.append(
            (
                Cell(name, name),
                number_cell(averaged, decimals),
                number_cell(errors[name], decimals),
                number_cell(whole, decimals),
                number_cell(whole_se, decimals),
            )
        )
        if folds:
            low, high = float(min(folds)), float(max(folds))
            spread_rows.append(
                (
                    Cell(name, name),
                    number_cell(sd, decimals),
                    Cell(
                        (low, high),
                        f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
                    ),
                )
            )
    tables = [
        SummaryTable(
            ("", vocabulary.FOLD_AVERAGED, cv_label, vocabulary.WHOLE_SAMPLE, whole_label),
            tuple(rows),
            kind="other",
        )
    ]
    if spread_rows:
        tables.append(
            SummaryTable(
                ("", vocabulary.FOLD_SD, vocabulary.FOLD_RANGE),
                tuple(spread_rows),
                title="Estimates of the single folds, first split",
                kind="other",
            )
        )
    sizes = report.fold_sizes
    header = [
        (
            vocabulary.FOLDS,
            f"{report.n_folds} (sizes {min(sizes):,} to {max(sizes):,})"
            if sizes
            else str(report.n_folds),
        )
    ]
    if report.repeats > 1:
        header.append((vocabulary.SPLITS, count_text(report.repeats)))
    words = vocabulary.CV_SPREAD_WORDS["inference" if inferential else "withheld"]
    notes = [note("cv_two_reports", vocabulary.NOTE_TEXTS["cv_two_reports"].format(spread=words))]
    if report.repeats > 1:
        notes.append(
            note(
                "cv_first_split",
                vocabulary.NOTE_TEXTS["cv_first_split"].format(repeats=count_text(report.repeats)),
            )
        )
    if report.fold_evaluated:
        level = 1.0 - next(iter(report.fold_evaluated.values())).alpha
        _, status_notes = estimates_table(report.fold_evaluated, level=level)
        notes.extend(status_notes)
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    "The fold SD and fold range describe the estimates of the single folds "
                    "of the first split.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.CV_TITLE, header=header, tables=tables, notes=notes, sections=sections
    )


def cv_repr(report: CVTargeting) -> str:
    """The one-line repr of the fold-level report.

    Parameters
    ----------
    report : CVTargeting
        The report.

    Returns
    -------
    str
        For example ``CVTargeting(estimands=3, folds=5, splits=1)``.
    """
    return short_repr(
        "CVTargeting",
        [
            ("estimands", len(report.variance)),
            ("folds", report.n_folds),
            ("splits", report.repeats),
        ],
    )


# ----------------------------------------------------------------- result sets


def _level_text(result_set: TMLEResultSet, value: float | None) -> str:
    return f"{value:g}" if value is not None else "none"


def result_set_summary(result_set: TMLEResultSet, detail: Detail) -> Summary:
    """The summary of a result set: one result's, or one table over the levels.

    Parameters
    ----------
    result_set : TMLEResultSet
        The results.
    detail : {"short", "full"}
        Passed to each result's summary.

    Returns
    -------
    Summary
        A single entry's own summary, or a combined summary over the levels of the
        intermediate variable.
    """
    levels = result_set.levels
    if len(levels) == 1:
        return result_set.results[levels[0]].summary(detail=detail)
    parts = {value: result_set.results[value].summary(detail=detail) for value in levels}
    first = result_set.results[levels[0]]
    z_name = result_set.intermediate_name or "Z"
    level_words = {value: _level_text(result_set, value) for value in levels}
    data = first.data
    header: list[tuple[str, str]] = [
        (
            vocabulary.QUESTION,
            vocabulary.RESULT_SET_QUESTION.format(
                A=data.treatment_name, Y=data.outcome_name, Z=z_name
            ),
        ),
        (vocabulary.INTERMEDIATE, f"{z_name}: {', '.join(level_words.values())}"),
    ]
    for label, fact in parts[levels[0]].header:
        if label == vocabulary.QUESTION:
            continue
        if all(dict(parts[v].header).get(label) == fact for v in levels):
            header.append((label, fact))
    tables = [part.table() for part in parts.values()]
    columns = tables[0].columns
    rows = []
    for value, table in zip(levels, tables, strict=True):
        if table.columns != columns:
            raise ValueError("the levels of a result set print different estimate columns")
        for row in table.rows:
            label = vocabulary.LEVEL_ROW.format(name=row[0].text, Z=z_name, z=level_words[value])
            rows.append((Cell((row[0].value, value), label), *row[1:]))
    combined = SummaryTable(columns, tuple(rows), kind="estimates")
    notes: dict[str, list[tuple[float | None, Note]]] = {}
    for value, part in parts.items():
        for item in part.notes:
            notes.setdefault(item.code, []).append((value, item))
    merged: list[Note] = []
    for code, entries in notes.items():
        texts = {item.text for _, item in entries}
        if len(texts) == 1 and len(entries) == len(levels):
            merged.append(entries[0][1])
        else:
            text = " ".join(
                f"At {z_name} = {level_words[value]}: {item.text}" for value, item in entries
            )
            merged.append(Note(code, text, entries[0][1].kind))
    merged.append(
        note("levels_separate", vocabulary.NOTE_TEXTS["levels_separate"].format(Z=z_name))
    )
    sections: list[SummarySection] = []
    if detail == "full":
        titles = list(dict.fromkeys(sec.title for part in parts.values() for sec in part.sections))
        for title in titles:
            found = {
                value: next((sec for sec in part.sections if sec.title == title), None)
                for value, part in parts.items()
            }
            if len(set(found.values())) == 1:
                section = found[levels[0]]
                assert section is not None
                sections.append(section)
                continue
            for value, section in found.items():
                if section is not None:
                    sections.append(
                        SummarySection(f"{title} ({z_name} = {level_words[value]})", section.lines)
                    )
    return Summary.build(
        title=vocabulary.RESULT_SET_TITLE.format(
            method=vocabulary.method_short_name(first.fitted_method)
        ),
        header=header,
        tables=[combined],
        notes=merged,
        sections=sections,
    )


def result_set_repr(result_set: TMLEResultSet) -> str:
    """The one-line repr of a result set.

    Parameters
    ----------
    result_set : TMLEResultSet
        The results.

    Returns
    -------
    str
        ``TMLEResultSet(TMLEResult(...))`` for one entry, otherwise for example
        ``TMLEResultSet(levels=2, Z=[0, 1], method=TMLE)``.
    """
    levels = result_set.levels
    if len(levels) == 1:
        return f"TMLEResultSet({result_set.results[levels[0]]!r})"
    first = result_set.results[levels[0]]
    z_name = result_set.intermediate_name or "Z"
    return short_repr(
        "TMLEResultSet",
        [
            ("levels", len(levels)),
            (z_name, "[" + ", ".join(_level_text(result_set, v) for v in levels) + "]"),
            ("method", vocabulary.method_short_name(first.fitted_method)),
        ],
    )
