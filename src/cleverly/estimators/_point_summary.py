"""The printed summary of a point-treatment result (:meth:`TMLEResult.summary`).

The result supplies the facts every point-treatment method shares: the question, the
data, the estimates and their notes. The estimator's ``_summary_facts`` hook supplies
the method's own name, its models and any facts only it has, so a method adds what it
needs without rewriting the summary. :func:`base_summary_facts` is the hook's default.

What a summary never prints, at either detail (Eric, 2026-10-24): hashes, fingerprints,
versions, timestamps, seeds, run ids, the targeting step's internals, registry or rung
ids, and cache keys. They stay on the result for programs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .._inference_status import (
    MINIMUM_INTERVAL_CLUSTERS,
    NO_SIMULTANEOUS_BANDS,
    NO_T_REFERENCE_BANDS,
    T_REFERENCE_NOTE,
    T_REFERENCE_STRATUM,
    supplies_inference,
)
from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummaryFacts,
    SummarySection,
    SummaryTable,
    count_text,
    note,
    number_text,
    share_text,
    vocabulary,
)
from .._summary._estimates import estimates_table, plain_support_reason, without_roadmap
from .._summary._learners import is_parametric, learner_name
from ..data.treatment_covariates import treatment_mechanism_restriction
from ..inference.cluster import (
    cluster_sizes,
    cluster_weight_mass,
    fewest_clusters,
    positive_mass_clause,
    unequal_cluster_sizes,
)
from ..inference.influence import (
    BOOTSTRAP_INTERVAL_LABELS,
    bootstrap_is_diagnostic,
)
from ..targets import TARGETS
from ._method_summary import method_facts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from .base import TMLEResult

#: The most covariate names the Adjusted for fact lists before it counts the rest.
_MOST_COVARIATES = 8
_SHOWN_COVARIATES = 6


# ----------------------------------------------------------------- the hook's default


def _learner_specs(result: TMLEResult) -> dict[str, Any] | None:
    """The learner specifications the fit ran with, or ``None`` when none is recorded."""
    for source in (result.estimator, getattr(getattr(result, "method", None), "models", None)):
        if source is not None and hasattr(source, "outcome_learner"):
            missingness = getattr(source, "missingness_learner", None)
            return {
                "outcome": source.outcome_learner,
                "treatment": source.treatment_learner,
                # The response and recording models fall back to the treatment learner.
                "missingness": source.treatment_learner if missingness is None else missingness,
                "intermediate": getattr(source, "intermediate_learner", None),
            }
    return None


def base_summary_facts(result: TMLEResult) -> SummaryFacts:
    """The facts every point-treatment method reports, and the default of the hook.

    Parameters
    ----------
    result : TMLEResult
        The fitted result.

    Returns
    -------
    SummaryFacts
        The title, the method, the models, and a "Method details" section holding the
        lines of the method-specific records (a collaborative selection, a DR-TMLE
        reduction, an adaptive IPTW fit, a learned rule).
    """
    method = result.fitted_method
    iptw = result.extra.get("adaptive_iptw")
    specs = _learner_specs(result)
    config = result.config
    outcome_model: str | None = None
    treatment_model: str | None = None
    if iptw is None and specs is not None:
        outcome_model = learner_name(specs["outcome"])
    if config.treatment_mechanism == "known":
        treatment_model = vocabulary.KNOWN_MECHANISM
    elif config.fits_treatment and specs is not None:
        treatment_model = learner_name(specs["treatment"])
        sources = result.data.treatment_adjustment_sources
        if sources is not None:
            # The design restricts what the treatment model reads.
            treatment_model += " on " + (", ".join(sources) if sources else "no covariates")
    header: list[tuple[str, str]] = []
    data = result.data
    if specs is not None:
        if data.has_missing_outcome:
            header.append((vocabulary.MISSINGNESS_MODEL, learner_name(specs["missingness"])))
        if data.has_missing_treatment:
            header.append((vocabulary.RECORDING_MODEL, learner_name(specs["missingness"])))
        if data.intermediate is not None and specs["intermediate"] is not None:
            header.append((vocabulary.INTERMEDIATE_MODEL, learner_name(specs["intermediate"])))
    record = method_facts(
        result,
        learner_name(specs["treatment"]) if specs is not None and config.fits_treatment else None,
    )
    if record.treatment_model is not None:
        treatment_model = record.treatment_model
    header.extend(record.header)
    details = list(record.details)
    learned = result.extra.get("learned_rule")
    if learned is not None:
        details.append(learned.describe())
    return SummaryFacts(
        title=vocabulary.ESTIMATE_TITLE.format(method=vocabulary.method_short_name(method)),
        method=record.method or vocabulary.method_name(method),
        outcome_model=outcome_model,
        treatment_model=treatment_model,
        header=tuple(header),
        notes=record.notes,
        sections=(
            (SummarySection(vocabulary.SECTION_METHOD, tuple(map(without_roadmap, details))),)
            if details
            else ()
        ),
        question=vocabulary.LEARNED_RULE_QUESTION if learned is not None else None,
    )


# ----------------------------------------------------------------- header


def _arm_text(result: TMLEResult, code: Any) -> str:
    if code is None:
        return "?"
    try:
        return str(result.data.arm_label(float(code)))
    except (TypeError, ValueError, KeyError, IndexError):
        return str(code)


#: The arm the unkeyed mean rows of a binary fit read. A contrast reads the arm that is not
#: the reference.
_UNKEYED_ARM = {"ey1": 1.0, "ey0": 0.0}


def _fill(template: str, result: TMLEResult, key: Any, target: str | None = None) -> str:
    data = result.data
    value = getattr(key, "value", None)
    reference = getattr(key, "reference", None)
    if reference is None:
        reference = result.config.reference_arm
    if value is None and key is None:
        value = _UNKEYED_ARM.get(target or "", 0.0 if float(reference) == 1.0 else 1.0)
    text = template.format(
        Y=data.outcome_name,
        A=data.treatment_name,
        v=_arm_text(result, value),
        r=_arm_text(result, reference),
    )
    stratum = getattr(key, "stratum", None)
    if stratum is not None and data.has_strata:
        try:
            text += f", within {data.stratum_label(int(stratum))}"
        except (TypeError, ValueError, KeyError, IndexError):
            text += f", within stratum {stratum}"
    return text


def _target_name(result: TMLEResult, name: str) -> str | None:
    key = result.parameter_keys.get(name)
    target = getattr(key, "estimand", None)
    if target is None and name in TARGETS:
        target = name
    return target


def _meaning(result: TMLEResult, name: str) -> str:
    """What one row estimates, in plain words."""
    stratum = _stratum_row(result, name)
    if stratum is not None:
        base, label = stratum
        return vocabulary.STRATUM_MEANING.format(meaning=_meaning(result, base), stratum=label)
    key = result.parameter_keys.get(name)
    target = _target_name(result, name)
    axis = getattr(key, "axis", "arm")
    template = vocabulary.ESTIMAND_MEANINGS.get(target or "")
    if template is not None and axis == "arm":
        return _fill(template, result, key, target)
    if target in TARGETS and TARGETS[target].description:
        return TARGETS[target].description.split(",")[0]
    return name


def _stratum_row(result: TMLEResult, name: str) -> tuple[str, str] | None:
    """The base row name and stratum label of a per-stratum row such as ``ate[S='a']``."""
    data = result.data
    if data.strata is None or not name.endswith("]") or "[" not in name:
        return None
    base, _, label = name[:-1].partition("[")
    labels = {data.stratum_label(code) for code in range(data.n_strata)}
    return (base, label) if label in labels else None


def _question(result: TMLEResult, facts: SummaryFacts) -> str:
    data = result.data
    if facts.question is not None:
        return facts.question
    if result.intermediate_value is not None:
        return vocabulary.DIRECT_EFFECT_QUESTION.format(
            A=data.treatment_name,
            Y=data.outcome_name,
            Z=data.intermediate_name or "Z",
            z=f"{result.intermediate_value:g}",
        )
    if len(result.estimates) == 1:
        (name,) = result.estimates
        key = result.parameter_keys.get(name)
        target = _target_name(result, name)
        phrase = vocabulary.QUESTION_PHRASES.get(target or "")
        if phrase is not None and getattr(key, "axis", "arm") == "arm":
            return _fill(phrase, result, key, target)
        return _meaning(result, name)
    return vocabulary.SEVERAL_ROWS_QUESTION.format(A=data.treatment_name, Y=data.outcome_name)


def _covariates(result: TMLEResult) -> str | None:
    data = result.data
    sources = tuple(dict.fromkeys(data.indicator_sources.get(n, n) for n in data.covariate_names))
    if not sources:
        return None
    if len(sources) > _MOST_COVARIATES:
        shown = ", ".join(sources[:_SHOWN_COVARIATES])
        return f"{shown} and {len(sources) - _SHOWN_COVARIATES} more"
    return ", ".join(sources)


def _treatment_fact(result: TMLEResult) -> tuple[str, str]:
    data = result.data
    weighted = vocabulary.WEIGHTED if data.is_weighted else ""
    if data.treatment_kind == "vector":
        columns = data._component_columns()
        parts: list[str] = []
        for name, levels in zip(data.component_names, data.component_levels, strict=True):
            values = np.asarray(columns[name])
            if levels:
                listed = ", ".join(
                    f"{level}={float(np.average(values == level, weights=data.weights)):.3f}"
                    for level in levels
                )
                parts.append(f"{name}: {listed}")
            else:
                dose = np.asarray(values, dtype=float)
                parts.append(f"{name}: range [{dose.min():.3g}, {dose.max():.3g}]")
        return vocabulary.ARM_SHARES + weighted, "; ".join(parts)
    if data.is_continuous_treatment:
        dose = np.asarray(data.treatment, dtype=float)
        mean = float(np.average(dose, weights=data.weights))
        fact = f"mean {mean:.4g}, range [{dose.min():.3g}, {dose.max():.3g}]"
        return vocabulary.DOSE + weighted, fact
    if data.is_binary_treatment and _zero_one_arms(data):
        return vocabulary.TREATED_SHARE + weighted, share_text(data.treated_fraction)
    shares = ", ".join(
        f"{data.arm_label(arm)}={share:.3f}"
        for arm, share in zip(data.arm_codes, data.arm_fractions, strict=True)
    )
    return vocabulary.ARM_SHARES + weighted, shares


def _zero_one_arms(data: Any) -> bool:
    """Whether the two arms are labelled 0 and 1, so "treated" names arm 1 unambiguously."""
    levels = tuple(data.treatment_levels or ())
    try:
        return len(levels) == 2 and {float(level) for level in levels} == {0.0, 1.0}
    except (TypeError, ValueError):
        return False


def _outcome_fact(result: TMLEResult) -> str:
    data = result.data
    observed = np.asarray(data.outcome, dtype=float)[np.asarray(data.observed, dtype=bool)]
    binary = observed.size > 0 and bool(np.isin(observed, (0.0, 1.0)).all())
    text = f"{data.outcome_name} ({'binary' if binary else 'continuous'})"
    if data.has_missing_outcome:
        text += f"; {count_text(int(data.observed.sum()))} of {count_text(data.n)} observed"
    return text


def _rows_fact(result: TMLEResult) -> str:
    data = result.data
    text = count_text(data.n)
    if data.has_missing_treatment:
        text += f" ({count_text(int(data.treatment_recorded.sum()))} with a recorded treatment)"
    return text


def _folds_fact(result: TMLEResult) -> str:
    config = result.config
    if not config.cross_fit:
        return "no"
    # The repeated_splits note says how the repeats combine.
    return f"{config.n_folds} folds"


def _header(
    result: TMLEResult, facts: SummaryFacts, clipping: _Clipping
) -> tuple[list[tuple[str, str]], int]:
    """The header facts and how many of them go in the left column."""
    data = result.data
    left: list[tuple[str, str]] = [
        (vocabulary.QUESTION, _question(result, facts)),
        (vocabulary.OUTCOME, _outcome_fact(result)),
    ]
    covariates = _covariates(result)
    if covariates is not None:
        left.append((vocabulary.ADJUSTED_FOR, covariates))
    left.append((vocabulary.METHOD, facts.method))
    if data.cluster is not None:
        # The count the few-cluster rules read: a cluster with zero weight mass adds nothing.
        clusters = count_text(int(np.unique(data.cluster).size)) + positive_mass_clause(
            data.cluster, data.weights if data.is_weighted else None
        )
        left.append((vocabulary.CLUSTERS, clusters))
        if data.has_strata:
            assert data.strata is not None
            fewest = fewest_clusters(
                data.cluster, data.strata, data.weights if data.is_weighted else None
            )
            left.append((vocabulary.FEWEST_IN_STRATUM, f"{fewest} clusters"))
    if data.is_weighted:
        report = data.weight_report()
        kind = ", estimated" if report.estimated else ""
        left.append(
            (
                vocabulary.WEIGHTS,
                f"{report.name or 'weights'}{kind}; effective sample size "
                f"{report.effective_n:,.0f}",
            )
        )
    right: list[tuple[str, str]] = [
        (vocabulary.ROWS, _rows_fact(result)),
        _treatment_fact(result),
        (vocabulary.OUT_OF_FOLD, _folds_fact(result)),
    ]
    if facts.outcome_model is not None:
        right.append((vocabulary.OUTCOME_MODEL, facts.outcome_model))
    if facts.treatment_model is not None:
        right.append((vocabulary.TREATMENT_MODEL, facts.treatment_model))
    # The limits of the means and the average effect, or of the fit's one estimand group
    # when it has no means; Settings gives every pair.
    limits = [pair for pair, _, groups in clipping.pairs if "mean" in groups] or [
        pair for pair, _, _ in clipping.pairs
    ]
    if limits:
        low, high = limits[0]
        right.append((vocabulary.PROBABILITY_LIMITS, f"[{low:.4g}, {high:.4g}]"))
    right.extend(facts.header)
    return [*left, *right], len(left)


# ----------------------------------------------------------------- more tables


def _key_table(result: TMLEResult) -> SummaryTable | None:
    if len(result.estimates) < 2:
        return None
    rows = tuple((Cell(name, name), Cell.of(_meaning(result, name))) for name in result.estimates)
    return SummaryTable(("", vocabulary.MEANING), rows, title=vocabulary.KEY_TITLE)


def _decimals_of(table: SummaryTable) -> int:
    text = table.rows[0][1].text if table.rows else "0.0000"
    return len(text.split(".")[1]) if "." in text else 0


def _bands(result: TMLEResult, decimals: int, level: str) -> tuple[SummaryTable, Note] | None:
    bands = result.simultaneous
    if bands is None or not bands.bands:
        return None
    rows = tuple(
        (
            Cell(name, name),
            Cell(
                (float(low), float(high)),
                f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
            ),
        )
        for name, (low, high) in bands.bands.items()
    )
    table = SummaryTable(("", vocabulary.BAND), rows, title=f"Simultaneous {level} bands")
    sentences = [
        vocabulary.NOTE_TEXTS["simultaneous_bands"].format(count=len(bands.bands), level=level)
    ]
    if bands.support.label == vocabulary.EXPERIMENTAL:
        sentences.append(vocabulary.NOTE_TEXTS["bands_experimental"])
    outside = [name for name in result.estimates if name not in bands.bands]
    if outside:
        sentences.append(vocabulary.NOTE_TEXTS["bands_leave_out"].format(names=", ".join(outside)))
    return table, note("simultaneous_bands", " ".join(sentences))


def _bootstrap(result: TMLEResult, decimals: int, level: str) -> tuple[SummaryTable, list[Note]]:
    bootstrap = result.bootstrap
    assert bootstrap is not None
    rows: list[tuple[Cell, ...]] = []
    diagnostic: dict[str, list[str]] = {}
    experimental: list[str] = []
    interval_name = ""
    for name, estimate in result.estimates.items():
        summary = estimate.bootstrap
        if summary is None:
            continue
        interval_name = BOOTSTRAP_INTERVAL_LABELS[summary.interval]
        low, high = summary.ci
        flagged = bootstrap_is_diagnostic(estimate)
        if flagged:
            if not estimate.supplies_inference:
                reason = "withheld"
            elif summary.below_floor:
                reason = "below_floor"
            else:
                reason = "declared"
            diagnostic.setdefault(reason, []).append(name)
        elif summary.support.label == vocabulary.EXPERIMENTAL:
            experimental.append(name)
        rows.append(
            (
                Cell(name, name + (vocabulary.NO_INFERENCE_MARK if flagged else "")),
                Cell(float(summary.std_error), number_text(summary.std_error, decimals)),
                Cell(
                    (float(low), float(high)),
                    f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
                ),
            )
        )
    usable = bootstrap.n_requested - bootstrap.n_failed
    every_diagnostic = sum(len(names) for names in diagnostic.values()) == len(rows)
    if every_diagnostic:
        # A table of diagnostics says so in its labels, and needs no row marks.
        columns = (
            "",
            vocabulary.SPREAD_LABELS["bootstrap sd"],
            f"{interval_name} range (diagnostic)".strip(),
        )
        rows = [(Cell(row[0].value, str(row[0].value)), *row[1:]) for row in rows]
    else:
        columns = (
            "",
            vocabulary.SPREAD_LABELS["bootstrap se"],
            f"{interval_name} {level} CI".strip(),
        )
    table = SummaryTable(
        columns,
        tuple(rows),
        title=f"Bootstrap ({bootstrap.resampling} resampling, {usable:,} usable replicates)",
    )
    notes: list[Note] = []
    if diagnostic:
        notes.append(
            note(
                "bootstrap_diagnostic",
                " ".join(
                    vocabulary.BOOTSTRAP_DIAGNOSTIC[reason].format(
                        names=", ".join(names), clusters=MINIMUM_INTERVAL_CLUSTERS
                    )
                    for reason, names in diagnostic.items()
                ),
            )
        )
    if experimental:
        notes.append(
            note(
                "bootstrap_experimental",
                vocabulary.NOTE_TEXTS["bootstrap_experimental"].format(
                    names=", ".join(experimental)
                ),
            )
        )
    return table, notes


# ----------------------------------------------------------------- notes


@dataclass(frozen=True)
class _Clipping:
    """What the treatment-probability bounds did on this fit, per bound pair."""

    #: ``(pair, most units moved in one draw, the groups that read the pair)``.
    pairs: tuple[tuple[tuple[float, float], int, tuple[str, ...]], ...] = ()
    #: The groups whose count was not retained, because their score used a treatment
    #: model changed while the fit ran.
    not_retained: tuple[str, ...] = ()
    #: The share of recorded-outcome probabilities the missingness bound moved, or ``None``.
    missingness_share: float | None = None


def _bounded(result: TMLEResult) -> bool:
    """Whether a treatment-probability bound applies to this fit at all."""
    config = result.config
    return (
        config.fits_treatment
        and config.treatment_mechanism != "known"
        and not (config.parameter_axis == "policy" and config.continuous_treatment)
    )


def _clipping(result: TMLEResult) -> _Clipping:
    """The rows each bound moved, read through the positivity report's own rule."""
    from ..sensitivity.positivity import _clipped_fraction, group_clipping

    pairs: dict[tuple[float, float], list[Any]] = {}
    not_retained: list[str] = []
    if _bounded(result):
        for group in result.fluctuations:
            if group == "ipsi":
                continue
            most = 0
            for repeat in result.repeats:
                clipping = group_clipping(result, group, repeat.nuisance)
                if clipping.omission is not None:
                    not_retained.append(group)
                    break
                most = max(most, int(clipping.count))
            else:
                held = pairs.setdefault(clipping.bounds, [0, []])
                held[0] = max(held[0], most)
                held[1].append(group)
    share = None
    config = result.config
    if config.bounded_mechanisms:
        share = _clipped_fraction(result, (float(config.missingness_bound), 1.0), True)
    return _Clipping(
        tuple((pair, count, tuple(groups)) for pair, (count, groups) in pairs.items()),
        tuple(not_retained),
        share,
    )


def _mechanism_words(labels: Sequence[str]) -> str:
    """Plain words for the bounded recording and intermediate probabilities."""
    words = []
    for label in labels:
        if label.startswith("P(Delta_A"):
            words.append("that the treatment is recorded")
        elif label.startswith("P(Delta"):
            words.append("that the outcome is recorded")
        elif label.startswith("P(Z"):
            words.append("of the intermediate value")
        else:
            words.append(label)
    return _listed(dict.fromkeys(words))


def _capped_note(result: TMLEResult, clipping: _Clipping) -> Note | None:
    """The note on rows whose weights a bound capped, or ``None`` when no bound moved one."""
    n = count_text(result.data.n)
    split = " in one split" if result.n_repeats > 1 else ""
    up_to = "up to " if result.n_repeats > 1 else ""
    sentences = []
    for (low, high), count, groups in clipping.pairs:
        if not count:
            continue
        reads = _listed(_GROUP_WORDS.get(group, group) for group in groups)
        sentences.append(
            vocabulary.NOTE_TEXTS["weights_capped"].format(
                low=f"{low:.3g}",
                high=f"{high:.3g}",
                count=f"{up_to}{count_text(count)}",
                n=n,
                split=split,
                reads=reads,
            )
        )
    if clipping.missingness_share:
        sentences.append(
            vocabulary.NOTE_TEXTS["weights_capped_missingness"].format(
                what=_mechanism_words(result.config.bounded_mechanisms),
                bound=f"{result.config.missingness_bound:.3g}",
                share=f"{clipping.missingness_share:.1%}",
            )
        )
    if not sentences:
        return None
    sentences.append(vocabulary.NOTE_TEXTS["weights_capped_effect"])
    return note("weights_capped", " ".join(sentences))


def _listed(words: Iterable[str]) -> str:
    """``a``, ``a and b``, or ``a, b and c``."""
    items = list(words)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


#: Plain names of the target groups a bound pair serves, for the weights note.
_GROUP_WORDS = {
    "mean": "the means and the average effect",
    "att": "the effect among the treated",
    "atc": "the effect among the untreated",
    "regime": "the rule means",
    "msm": "the projection",
    "mtp": "the shifted-treatment mean",
}


def _fit_notes(
    result: TMLEResult, facts: SummaryFacts, clipping: _Clipping, spread: str
) -> list[Note]:
    notes: list[Note] = []
    config = result.config
    specs = _learner_specs(result)
    iptw = result.extra.get("adaptive_iptw") is not None
    if not config.cross_fit and specs is not None:
        data = result.data
        learners = [] if iptw else [specs["outcome"]]
        if config.fits_treatment and config.treatment_mechanism != "known":
            learners.append(specs["treatment"])
        if data.has_missing_outcome or data.has_missing_treatment:
            learners.append(specs["missingness"])
        if data.intermediate is not None:
            learners.append(specs["intermediate"])
        if any(not is_parametric(learner) for learner in learners):
            notes.append(note("in_sample_flexible", vocabulary.NOTE_TEXTS["in_sample_flexible"]))
    capped = _capped_note(result, clipping)
    if capped is not None:
        notes.append(capped)
    if result.data.is_weighted:
        notes.append(note("weighted_population", vocabulary.NOTE_TEXTS["weighted_population"]))
    if result.n_repeats > 1:
        notes.append(
            note(
                "repeated_splits",
                vocabulary.NOTE_TEXTS["repeated_splits"].format(
                    repeats=result.n_repeats, spread=spread
                ),
            )
        )
    if config.cross_fit and config.crossfit.n_folds != config.n_folds:
        notes.append(
            note(
                "folds_reduced",
                vocabulary.NOTE_TEXTS["folds_reduced"].format(
                    declared=config.crossfit.n_folds, used=config.n_folds
                ),
            )
        )
    notes.extend(_score_notes(result))
    notes.extend(facts.notes)
    # A note that states what a recorded warning says takes the warning's place, and is
    # shown as a warning when the fit warned.
    config = result.config
    low, high = config.g_bounds
    context = {
        "declared": config.crossfit.n_folds,
        "used": config.n_folds,
        "low": f"{low:.4g}",
        "high": f"{high:.4g}",
    }
    # The positivity warning counts at the mean bounds, so only a note that prints that
    # pair states it.
    mean_pair = any("mean" in groups for _, _, groups in clipping.pairs)
    kept = list(result.fit_warnings)
    for index, item in enumerate(notes):
        phrases = [
            phrase.format(**context)
            for phrase in vocabulary.RESTATED_WARNINGS.get(item.code, ())
            if item.code != "weights_capped" or mean_pair or "{low}" not in phrase
        ]
        restated = [r for r in kept if any(phrase in r.message for phrase in phrases)]
        if restated:
            kept = [r for r in kept if r not in restated]
            notes[index] = replace(item, kind="warning")
    notes.extend(_warning_notes(kept))
    return notes


def _score_notes(result: TMLEResult) -> list[Note]:
    """The notes on how well the fit solved its estimating equation."""
    from ..fluctuation._score import wald_coverage_bound

    verdict = result.score_verdict
    notes: list[Note] = []
    if verdict.declared:
        # The remainder in the standard error the interval itself uses, which can be
        # cluster-robust, cross-validated or the median over splits.
        remainders = {
            name: abs(float(estimate.score)) / float(estimate.std_error)
            for name, estimate in result.estimates.items()
            if estimate.supplies_inference
            and np.isfinite(estimate.score)
            and float(estimate.std_error) > 0.0
        }
        if remainders:
            name, b = max(remainders.items(), key=lambda item: item[1])
            alpha = float(result.estimates[name].alpha)
            # A lower bound, so it is rounded down; a remainder below 0.001 is stated as
            # that ceiling and its coverage computed there.
            shown = max(b, _SMALLEST_REMAINDER)
            bound = math.floor(wald_coverage_bound(shown, alpha) * 10000) / 100
            notes.append(
                note(
                    "stopping_rule",
                    vocabulary.NOTE_TEXTS["stopping_rule"].format(
                        b=f"less than {_SMALLEST_REMAINDER:g}"
                        if b < _SMALLEST_REMAINDER
                        else f"{b:.3g}",
                        name=name,
                        level=f"{(1.0 - alpha) * 100:g}%",
                        bound=f"{bound:.2f}%",
                    ),
                )
            )
        else:
            notes.append(note("stopping_rule", vocabulary.NOTE_TEXTS["stopping_rule_plain"]))
    if not verdict.passed:
        failing = verdict.rule_failures if verdict.declared else verdict.failures
        named = list(dict.fromkeys(_score_row_words(result, row.name) for row in failing))
        spread = (
            "interval"
            if all(result.estimates[name].supplies_inference for name in result.estimates)
            else "spread"
        )
        key = "score_rule_unmet" if verdict.declared and verdict.rule_failures else "score_unsolved"
        notes.append(
            note(
                "score_unsolved",
                vocabulary.NOTE_TEXTS[key].format(
                    names=_listed(named) if named else _listed(result.estimates), spread=spread
                ),
            )
        )
    return notes


#: The smallest stopping-rule remainder a note prints as a number.
_SMALLEST_REMAINDER = 0.001


def _score_row_words(result: TMLEResult, name: str) -> str:
    """What a score-check row covers, in the words of the summary's rows."""
    if name in result.estimates:
        return name
    group = name.split("[", 1)[0]
    if group in _GROUP_WORDS:
        return _GROUP_WORDS[group]
    return vocabulary.SCORE_MODEL_ROW


#: The most distinct messages one warning note quotes before it counts the rest.
_MOST_MESSAGES = 3


def _warning_notes(records: Sequence[Any]) -> list[Note]:
    """One note per recorded warning category, each distinct message once, in first-seen
    order. The messages are quoted as the user saw them, without roadmap pointers."""
    grouped: dict[str, list[Any]] = {}
    for record in records:
        code = vocabulary.WARNING_NOTE_CODES.get(record.category, "fit_warning_other")
        grouped.setdefault(code, []).append(record)
    notes = []
    for code, held in grouped.items():
        quoted = [
            without_roadmap(record.message)
            + ("" if record.count == 1 else vocabulary.REPEATS.format(count=record.count))
            for record in held[:_MOST_MESSAGES]
        ]
        text = vocabulary.NOTE_TEXTS["fit_warning"].format(messages=" ".join(quoted))
        if len(held) > _MOST_MESSAGES:
            text += " " + vocabulary.NOTE_TEXTS["fit_warning_more"].format(
                count=len(held) - _MOST_MESSAGES
            )
        notes.append(note(code, text))
    return notes


# ----------------------------------------------------------------- full detail


def _assumptions(result: TMLEResult) -> SummarySection:
    effect = result.identified_effect
    lines: list[str] = []
    if effect is not None:
        identification = effect.identification
        lines.append(vocabulary.ASSUMPTIONS_LEAD.format(functional=effect.functional.expression))
        lines.extend(f"- {item}" for item in identification.assumptions)
        lines.extend(
            f"- {reading.label}: {reading.meaning} (needs {'; '.join(reading.assumptions)})"
            for reading in identification.interpretations
        )
        restrictions = identification.model_restrictions
        if restrictions and result.fitted_method != "adaptive_iptw":
            lines.extend(f"- model restriction: {item}" for item in restrictions)
    else:
        lines.append(vocabulary.ASSUMPTIONS_LEAD_GENERIC)
        sources = result.data.treatment_adjustment_sources
        if sources is not None and result.fitted_method != "adaptive_iptw":
            # A fit with no study carries no identification record, so the data's own
            # declaration states the restriction a design records.
            lines.append(f"- model restriction: {treatment_mechanism_restriction(sources)}")
        seen: list[str] = []
        for name in result.estimates:
            target = _target_name(result, name)
            if target in TARGETS:
                for item in TARGETS[target].identification.assumptions:
                    if item not in seen:
                        seen.append(item)
        lines.extend(f"- {_with_arm_label(result, item)}" for item in seen)
        lines.extend(f"- {item}" for item in _missing_data_assumptions(result))
    try:
        available = result.sensitivity.capability("omitted_confounding").available
    except (KeyError, AttributeError):
        available = False
    if available:
        lines.append(vocabulary.SENSITIVITY_POINTER)
    return SummarySection(vocabulary.SECTION_ASSUMPTIONS, tuple(lines))


def _with_arm_label(result: TMLEResult, item: str) -> str:
    """An assumption written for arms 0 and 1, with the fit's own label for arm 1."""
    data = result.data
    if not data.is_binary_treatment or _zero_one_arms(data):
        return item
    return item.replace("P(A = 1 | W)", f"P(A = {data.arm_label(data.arm_codes[1])} | W)")


def _missing_data_assumptions(result: TMLEResult) -> list[str]:
    """The conditions a fit with no study needs for its missing data and intermediate."""
    data = result.data
    names = {"Y": data.outcome_name, "A": data.treatment_name}
    lines: list[str] = []
    if data.has_missing_outcome:
        lines.extend(text.format(**names) for text in vocabulary.MISSING_OUTCOME_ASSUMPTIONS)
    if data.has_missing_treatment:
        lines.extend(text.format(**names) for text in vocabulary.MISSING_TREATMENT_ASSUMPTIONS)
    if result.intermediate_value is not None:
        lines.extend(
            text.format(
                **names, Z=data.intermediate_name or "Z", z=f"{result.intermediate_value:g}"
            )
            for text in vocabulary.INTERMEDIATE_ASSUMPTIONS
        )
    return lines


def _settings(result: TMLEResult, clipping: _Clipping) -> SummarySection:
    config = result.config
    data = result.data
    lines: list[str] = []
    up_to = "up to " if result.n_repeats > 1 else ""
    for (low, high), count, groups in clipping.pairs:
        reads = _listed(_GROUP_WORDS.get(group, group) for group in groups)
        reached = (
            "no row" if not count else f"{up_to}{count_text(count)} row{'' if count == 1 else 's'}"
        )
        lines.append(
            f"Estimated treatment probabilities limited to [{low:.4g}, {high:.4g}] for "
            f"{reads}; {reached} reached a limit."
        )
    if any("mean" in groups for _, _, groups in clipping.pairs) and (
        config.auto_bounds_n is not None
    ):
        lines.append(
            "The limits for the means and the average effect come from the sample size, here "
            f"the effective sample size of the weights, {config.auto_bounds_n:,.0f}."
        )
    if clipping.not_retained:
        reads = _listed(_GROUP_WORDS.get(group, group) for group in clipping.not_retained)
        lines.append(
            f"For {reads} the fit changed the treatment model while it ran, and how many "
            "rows reached a limit was not kept."
        )
    if data.cluster is not None:
        lines.append(f"Clusters: {_cluster_fact(data)}.")
    if config.treatment_mechanism == "known":
        lines.append("Treatment probabilities are the declared known values; no limit moved one.")
    if config.bounded_mechanisms:
        lines.append(
            f"Estimated probabilities {_mechanism_words(config.bounded_mechanisms)} raised to "
            f"at least {config.missingness_bound:.4g}."
        )
    if config.q_bounds is not None and result.extra.get("adaptive_iptw") is None:
        low, high = config.q_bounds
        lines.append(f"Outcome predictions kept inside the outcome range [{low:.4g}, {high:.4g}].")
    if config.cross_fit:
        if config.crossfit.n_folds != config.n_folds:
            lines.append(f"Folds asked for: {config.crossfit.n_folds}; used: {config.n_folds}.")
        else:
            lines.append(f"Folds: {config.n_folds}.")
        lines.append(
            vocabulary.CROSS_FIT_CONSTRUCTION.format(
                targeting=vocabulary.CROSS_FIT_TARGETING[config.targeting_scheme == "pooled"],
                evaluation=vocabulary.CROSS_FIT_EVALUATION[config.cv_evaluation],
            )
        )
    lines.extend(_band_settings(result))
    estimates = result.estimates
    if any(estimate.reference_df is not None for estimate in estimates.values()):
        assert data.cluster is not None
        lines.append(
            T_REFERENCE_NOTE.format(
                clusters=fewest_clusters(
                    data.cluster, weights=data.weights if data.is_weighted else None
                ),
                stratum=""
                if data.strata is None
                else T_REFERENCE_STRATUM.format(
                    fewest=fewest_clusters(
                        data.cluster, data.strata, data.weights if data.is_weighted else None
                    )
                ),
            )
        )
    if result.n_repeats > 1:
        spreads = result.repeat_spread()
        listed = ", ".join(
            f"{name} {spread:.3g}" for name, spread in spreads.items() if np.isfinite(spread)
        )
        missing = [name for name, spread in spreads.items() if not np.isfinite(spread)]
        if listed:
            lines.append(
                f"Spread of each row across the {result.n_repeats} splits (a diagnostic, on "
                f"the scale the interval is computed on): {listed}."
            )
        if missing:
            lines.append(
                f"{_listed(missing)} {'has' if len(missing) == 1 else 'have'} no spread on "
                "that scale: a split gave a value the scale cannot take."
            )
    reasons: dict[str, list[str]] = {}
    for name, estimate in estimates.items():
        if estimate.support.label == vocabulary.EXPERIMENTAL and estimate.support.reason:
            reasons.setdefault(plain_support_reason(estimate.support.reason), []).append(name)
    for reason, names in reasons.items():
        lines.append(
            f"Why {', '.join(names)} {'is' if len(names) == 1 else 'are'} experimental: {reason}"
        )
    return SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines))


def _cluster_fact(data: CausalData) -> str:
    """The cluster count, with the size and positive-mass facts a reader needs.

    A weighted fit whose clusters hold equal rows and unequal weight mass prints the mass
    range in place of the size range. The stratum count applies to a fit with baseline
    strata. The facts also name unequal sizes within a stratum and clusters that have
    zero weight mass. Equal sizes without strata print the count alone.
    """
    cluster = data.cluster
    assert cluster is not None
    counts = cluster_sizes(cluster)
    smallest, largest = int(counts.min()), int(counts.max())
    fact = f"{counts.size}"
    if smallest != largest:
        fact += f", sizes {smallest} to {largest}"
    elif data.is_weighted and unequal_cluster_sizes(cluster, data.weights):
        mass = cluster_weight_mass(cluster, data.weights)
        fact += f", weight mass {mass.min():.4g} to {mass.max():.4g}"
    fact += positive_mass_clause(cluster, data.weights if data.is_weighted else None)
    if data.has_strata:
        assert data.strata is not None
        masks = (data.strata == level for level in np.unique(data.strata))
        parts = [(cluster[inside], data.weights[inside]) for inside in masks]
        uneven = [cluster_sizes(part) for part, _ in parts]
        uneven = [part for part in uneven if int(part.min()) != int(part.max())]
        if uneven:
            fact += (
                f", within-stratum sizes {min(int(part.min()) for part in uneven)}"
                f" to {max(int(part.max()) for part in uneven)}"
            )
        elif data.is_weighted:
            uneven_mass = [
                cluster_weight_mass(part, weight)
                for part, weight in parts
                if unequal_cluster_sizes(part, weight)
            ]
            if uneven_mass:
                low_mass = min(float(part.min()) for part in uneven_mass)
                high_mass = max(float(part.max()) for part in uneven_mass)
                fact += f", within-stratum weight mass {low_mass:.4g} to {high_mass:.4g}"
        stratum_count = fewest_clusters(
            cluster, data.strata, data.weights if data.is_weighted else None
        )
        fact += f"; fewest in a stratum: {stratum_count}"
    return fact


def _band_settings(result: TMLEResult) -> list[str]:
    """Why a fit with several rows has no simultaneous band, when it would have had one."""
    estimates = result.estimates
    if result.simultaneous is not None or len(estimates) < 2:
        return []
    if not bool(getattr(result.estimator, "simultaneous", True)):
        # Bands were not asked for, so no reason why the fit has none applies.
        return []
    t_reference = any(estimate.reference_df is not None for estimate in estimates.values())
    if not supplies_inference(result.inference_status):
        return [NO_SIMULTANEOUS_BANDS]
    if t_reference:
        return [NO_T_REFERENCE_BANDS]
    withheld = [name for name, estimate in estimates.items() if not estimate.supplies_inference]
    if withheld:
        return [
            "No simultaneous band: fewer than two rows have an interval. Left out: "
            + ", ".join(withheld)
            + "."
        ]
    return []


def _study(result: TMLEResult) -> SummarySection:
    effect = result.identified_effect
    record = getattr(effect, "protocol", None)
    if record is None:
        return SummarySection(vocabulary.SECTION_STUDY, (vocabulary.NO_PROTOCOL,))
    return SummarySection(vocabulary.SECTION_STUDY, tuple(record.summary_lines()))


# ----------------------------------------------------------------- the summary


def point_summary(result: TMLEResult, facts: SummaryFacts, detail: Detail) -> Summary:
    """Build the summary of a point-treatment result.

    Parameters
    ----------
    result : TMLEResult
        The fitted result.
    facts : SummaryFacts
        The method's facts, from the estimator's hook.
    detail : {"short", "full"}
        ``"full"`` adds the Assumptions, Settings, Method details and Study description
        sections.

    Returns
    -------
    Summary
        The summary.
    """
    level_value = 1.0 - result.config.alpha_sig
    level = f"{level_value * 100:g}%"
    clipping = _clipping(result)
    header, split = _header(result, facts, clipping)
    table, notes = estimates_table(result.estimates, level=level_value)
    tables = [table]
    key = _key_table(result)
    if key is not None:
        tables.append(key)
    decimals = _decimals_of(table)
    bands = _bands(result, decimals, level)
    if bands is not None:
        tables.append(bands[0])
        notes.append(bands[1])
    if result.bootstrap is not None:
        boot_table, boot_notes = _bootstrap(result, decimals, level)
        if boot_table.rows:
            tables.append(boot_table)
            notes.extend(boot_notes)
    # The spread column's label, which names a diagnostic when no row has an interval.
    notes.extend(_fit_notes(result, facts, clipping, table.columns[2]))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(_assumptions(result))
        settings = _settings(result, clipping)
        if settings.lines:
            sections.append(settings)
        sections.extend(facts.sections)
        sections.append(_study(result))
    return Summary.build(
        title=facts.title,
        header=header,
        header_split=split,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def point_repr(result: TMLEResult) -> str:
    """The one-line repr of a point-treatment result.

    Parameters
    ----------
    result : TMLEResult
        The fitted result.

    Returns
    -------
    str
        For example ``TMLEResult(ate=1.513, 95% CI [1.422, 1.603], n=2000, method=TMLE)``.
    """
    from .._summary import short_repr
    from .._summary._render import decimals_for

    items = list(result.estimates.items())
    facts: list[tuple[str, Any]] = []
    if items:
        name, estimate = items[0]
        if estimate.supplies_inference:
            decimals = decimals_for([estimate.std_error])
            low, high = estimate.ci
            facts.append((name, number_text(estimate.psi, decimals)))
            facts.append(
                (
                    "",
                    f"{(1 - result.config.alpha_sig) * 100:g}% CI "
                    f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
                )
            )
        else:
            decimals = decimals_for([estimate.plugin_std_error])
            facts.append((name, f"{number_text(estimate.psi, decimals)} (no interval)"))
        if len(items) > 1:
            facts.append(("", f"+{len(items) - 1} more"))
    facts.append(("n", result.data.n))
    facts.append(("method", vocabulary.method_short_name(result.fitted_method)))
    if any(e.support.label == vocabulary.EXPERIMENTAL for _, e in items):
        facts.append(("", vocabulary.EXPERIMENTAL))
    return short_repr(type(result).__name__, facts)
