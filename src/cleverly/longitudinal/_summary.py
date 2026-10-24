"""The printed summary of a longitudinal result (:meth:`LongitudinalResult.summary`).

A longitudinal summary has the shape of a point-treatment one: a title, a header of facts
in two columns, the estimates table, the tables a reader needs to read it (what each row
estimates, what each plan assigns and who followed it, the bands and the bootstrap), then
numbered warnings and notes. ``detail="full"`` adds the assumptions, the settings, how the
estimator works and the study description.

It never prints library bookkeeping, at either detail: data or fold digests, the digest of
a rule's assigned arms, versions, time stamps, seeds and run ids stay on the result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .._inference_status import (
    NO_SIMULTANEOUS_BANDS,
    NO_T_REFERENCE_BANDS,
    T_REFERENCE_NOTE,
    supplies_inference,
)
from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    count_text,
    note,
    number_text,
    short_repr,
    vocabulary,
)
from .._summary._estimates import estimates_table, plain_support_reason
from .._summary._learners import is_parametric, learner_name
from .._summary._render import decimals_for
from ..estimators._point_summary import (
    _bands,
    _bootstrap,
    _decimals_of,
    _listed,
    _study,
    _warning_notes,
)
from ..inference.cluster import fewest_clusters, positive_mass_clause
from .estimator import HORIZON_INFIX
from .regimen import DynamicRegimen, Regimen, _is_mtp, _is_tilt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .estimator import LongitudinalResult

#: The most covariate names the Adjusted for fact lists before it counts the rest.
_MOST_COVARIATES = 8
_SHOWN_COVARIATES = 6
#: The smallest stopping-rule remainder a note prints as a number.
_SMALLEST_REMAINDER = 0.001
#: The baselines whose interval rests on every model the estimate reads being right.
_MODEL_BASED = {
    "gcomp": "model_based_interval_gcomp",
    "ipw": "model_based_interval_ipw",
    "parametric_gformula": "model_based_interval_gformula",
}


# ----------------------------------------------------------------- the method


def method_key(result: LongitudinalResult) -> str:
    """The key of the fit's estimator in the vocabulary's method maps."""
    config = result.config
    if config.estimator == "tmle" and config.msm_terms is not None:
        return "msm"
    return config.estimator


def _method(result: LongitudinalResult) -> str:
    key = method_key(result)
    name = vocabulary.LONGITUDINAL_METHOD_NAMES.get(
        key, vocabulary.LONGITUDINAL_METHOD_NAMES["tmle"]
    )
    normalization = result.config.normalization
    form = vocabulary.IPW_FORMS.get(normalization) if normalization is not None else None
    if key == "ipw" and form is not None:
        name += f", {form}"
    return name


def _short(result: LongitudinalResult) -> str:
    key = method_key(result)
    return vocabulary.LONGITUDINAL_SHORT_NAMES.get(key, vocabulary.LONGITUDINAL_SHORT_NAMES["tmle"])


# ----------------------------------------------------------------- learners


def _learners(result: LongitudinalResult) -> dict[str, Any] | None:
    """The learners the estimate reads, by role, or ``None`` when no estimator is recorded."""
    estimator = result.estimator
    if estimator is None:
        return None
    config = result.config
    data = result.data
    known = set(config.known_factors)
    roles: dict[str, Any] = {}
    key = config.estimator
    if key != "ipw":
        roles["outcome"] = getattr(estimator, "outcome_learner", None)
        pseudo = getattr(estimator, "pseudo_learner", None)
        uses_pseudo = config.survival_model != "hazard" and (
            data.n_times > 1 or len(config.outcome_names) > 1
        )
        if pseudo is not None and uses_pseudo and key != "parametric_gformula":
            roles["pseudo"] = pseudo
    if key not in ("gcomp", "parametric_gformula"):
        if not set(data.treatment_names) <= known:
            roles["treatment"] = getattr(estimator, "treatment_learner", None)
        if data.censoring_names and not set(data.censoring_names) <= known:
            roles["censoring"] = getattr(estimator, "censoring_learner", None)
    return roles


def _model_fact(result: LongitudinalResult, role: str, learners: Mapping[str, Any]) -> str | None:
    config = result.config
    data = result.data
    known = set(config.known_factors)
    if role == "treatment":
        if set(data.treatment_names) <= known:
            return vocabulary.KNOWN_MECHANISM
        if role not in learners:
            return None
        return learner_name(learners[role])
    if role == "censoring":
        if not data.censoring_names:
            return None
        if set(data.censoring_names) <= known:
            return vocabulary.KNOWN_MECHANISM
        if role not in learners:
            return None
        return learner_name(learners[role])
    if role not in learners:
        return None
    text = learner_name(learners["outcome"])
    if config.survival_model == "hazard":
        return f"{text} (hazards)"
    pseudo = learners.get("pseudo")
    if pseudo is not None and learner_name(pseudo) != text:
        text += f"; {learner_name(pseudo)} for earlier time points"
    return text


def _gformula_models(result: LongitudinalResult) -> dict[str, str]:
    """The model facts of a parametric g-formula fit, by header label.

    The outcome side is the outcome or event-hazard model. The natural-treatment model is
    fitted only for the time points a plan leaves to the data, so it is listed only then.
    """
    report = result.simulation
    if report is None:
        return {}

    def names(records: list[Any]) -> str:
        return ", ".join(
            dict.fromkeys(
                vocabulary.LEARNER_NAMES.get(record.learner, record.learner) for record in records
            )
        )

    facts: dict[str, str] = {}
    outcome = [record for record in report.models if record.role in ("outcome", "hazard")]
    if outcome:
        facts[vocabulary.OUTCOME_MODEL] = names(outcome)
    treatment = [
        record for record in report.models if record.role == "treatment" and record.stacked
    ]
    if treatment:
        facts[vocabulary.TREATMENT_MODEL] = names(treatment)
    covariates = [record for record in report.models if record.role == "covariate"]
    if covariates:
        count = len({record.label for record in covariates})
        facts[vocabulary.COVARIATE_MODELS] = (
            f"{names(covariates)} for {count} covariate{'' if count == 1 else 's'}"
        )
    return facts


# ----------------------------------------------------------------- names and meanings


def _plans(result: LongitudinalResult) -> set[str]:
    return {fit.regimen.label for fit in result.fits.values()}


def _row(name: str, result: LongitudinalResult) -> tuple[str, str, str | None, str | None, Any]:
    """``(prefix, plan, reference, cause, time)`` of one reported row.

    Read from the parameter index each name was built with, so a label that holds a comma or
    a time grid's times print as the fit means them. A contrast's two plans are the split of
    its label whose sides are both plans of the fit.
    """
    prefix = name.partition("[")[0].removesuffix("_regimen")
    key = (result.parameter_index or {}).get(name)
    if key is None:
        inside = name.partition("[")[2].removesuffix("]")
        cause = None
        if result.config.outcome_kind == "competing" and ", " in inside:
            inside, _, cause = inside.rpartition(", ")
        return prefix, inside, None, cause, None
    label, cause, horizon = key
    time = result._time_of(horizon) if HORIZON_INFIX in name else None
    plans = _plans(result)
    if label not in plans:
        parts = label.split(" vs ")
        for cut in range(1, len(parts)):
            plan, reference = " vs ".join(parts[:cut]), " vs ".join(parts[cut:])
            if plan in plans and reference in plans:
                return prefix, plan, reference, cause, time
    return prefix, label, None, cause, time


def _outcome_word(result: LongitudinalResult, time: Any) -> tuple[str, str]:
    """The outcome a row reads and the ``at`` phrase that follows it."""
    data = result.data
    config = result.config
    if time is None:
        return data.outcome_name, ""
    if config.outcome_kind == "repeated":
        for horizon in config.horizons:
            if result._time_of(horizon) == time and horizon - 1 < len(config.outcome_names):
                return config.outcome_names[horizon - 1], ""
    return data.outcome_name, f" at time {time:g}"


def meaning(result: LongitudinalResult, name: str, *, question: bool = False) -> str:
    """What one reported row estimates, in plain words.

    Parameters
    ----------
    result : LongitudinalResult
        The fitted result.
    name : str
        A key of ``result.estimates``.
    question : bool
        ``True`` for the shorter form the Question header fact prints.

    Returns
    -------
    str
        The meaning, or the name itself when the summary has no words for it.
    """
    templates = vocabulary.LONGITUDINAL_QUESTIONS if question else vocabulary.LONGITUDINAL_MEANINGS
    prefix, plan, reference, cause, time = _row(name, result)
    if prefix == "msm":
        text = templates["msm"].format(term=plan)
        return text if cause is None else vocabulary.MSM_CAUSE.format(text=text, cause=cause)
    survival = result.config.outcome_kind in ("survival", "competing")
    key = prefix
    if prefix == "ate" and survival:
        key = "cif_ate" if cause is not None else "risk_ate"
    if prefix in ("ey", "risk") and survival:
        key = "cif" if cause is not None else "risk"
    template = templates.get(key)
    contrast = reference is not None
    if template is None or contrast != (key in ("ate", "risk_ate", "cif_ate")):
        return name
    if survival and time is None:
        return name
    outcome, at = _outcome_word(result, time)
    return template.format(
        Y=outcome,
        at=at,
        d=plan,
        r=reference or "",
        t="" if time is None else f"{time:g}",
        cause=cause or "",
    )


def _question(result: LongitudinalResult) -> str:
    data = result.data
    if result.config.msm_terms is not None:
        return vocabulary.LONGITUDINAL_MSM_QUESTION.format(Y=data.outcome_name)
    if len(result.estimates) == 1:
        (name,) = result.estimates
        return meaning(result, name, question=True)
    if result.config.outcome_kind in ("survival", "competing"):
        return vocabulary.LONGITUDINAL_SEVERAL_ROWS.format(Y="the event risk")
    if result.config.outcome_kind == "repeated":
        return vocabulary.LONGITUDINAL_SEVERAL_ROWS.format(Y=vocabulary.REPEATED_OUTCOME)
    return vocabulary.LONGITUDINAL_SEVERAL_ROWS.format(Y=data.outcome_name)


# ----------------------------------------------------------------- plans


def _node_words(node: Any) -> str:
    """One node of a plan in plain words."""
    from ..interventions import Stochastic
    from ..interventions.policy import describe_policy

    if isinstance(node, Stochastic):
        return (
            vocabulary.PLAN_NAMED_POLICY.format(name=node.name)
            if node.name
            else vocabulary.PLAN_POLICY
        )
    if _is_tilt(node):
        delta = node.delta
        if isinstance(delta, Mapping):
            factors = ", ".join(f"{level} {float(value):g}" for level, value in delta.items())
            return vocabulary.PLAN_TILT_LEVELS.format(factors=factors)
        if float(delta) == 1.0:
            return vocabulary.PLAN_NATURAL
        return vocabulary.PLAN_TILT.format(delta=f"{float(delta):g}")
    if _is_mtp(node):
        return _policy_words(node, describe_policy)
    if callable(node):
        name = getattr(node, "__name__", "<lambda>")
        return (
            vocabulary.PLAN_RULE
            if name == "<lambda>"
            else vocabulary.PLAN_NAMED_RULE.format(name=name)
        )
    return str(node)


def _policy_words(policy: Any, fallback: Any) -> str:
    """A modified treatment policy in words; its class text when no plain form exists."""
    from ..interventions.policy import ModifiedPolicy, Scale, Shift

    bound = getattr(policy, "cap", None)
    if bound is None or not isinstance(bound, (int, float)):
        bound_text = None if bound is None else str(bound)
    else:
        bound_text = f"{float(bound):g}"
    cap = "" if bound_text is None else f", capped at {bound_text}"
    if isinstance(policy, Shift) and policy.target is None:
        if float(policy.delta) == 0.0:
            return vocabulary.PLAN_NATURAL
        return vocabulary.PLAN_SHIFT.format(delta=f"{float(policy.delta):+g}", cap=cap)
    if isinstance(policy, Scale) and policy.target is None:
        if float(policy.factor) == 1.0:
            return vocabulary.PLAN_NATURAL
        return vocabulary.PLAN_SCALE.format(factor=f"{float(policy.factor):g}", cap=cap)
    if isinstance(policy, ModifiedPolicy):
        return vocabulary.PLAN_NAMED_MTP.format(name=policy.name)
    return str(fallback(policy))


def _without_time(words: str, t: int) -> str | None:
    """``words`` without a trailing time tag such as `` at 2`` or ``@2``; None without one."""
    for tag in (f" at {t}", f"@{t}", f"_{t}"):
        if words.endswith(tag):
            return words[: -len(tag)]
    return None


def plan_words(result: LongitudinalResult, regimen: Any) -> str:
    """What one plan assigns at each time point, in plain words.

    Parameters
    ----------
    result : LongitudinalResult
        The fitted result.
    regimen : Regimen or DynamicRegimen
        The plan.

    Returns
    -------
    str
        For example ``"1 at every time point"`` or ``"1 at time 1; a declared rule at time 2"``.
    """
    if isinstance(regimen, Regimen):
        nodes: Sequence[Any] = regimen.values
    elif isinstance(regimen, DynamicRegimen):
        nodes = regimen.plan
    else:  # pragma: no cover - every resolved plan is one of the two
        return str(regimen)
    words = [_node_words(node) for node in nodes]
    # Per-node names such as "price_up at 1" or "d@1" name one plan: print it once.
    stripped = [_without_time(text, index) for index, text in enumerate(words, 1)]
    if len(words) > 1 and None not in stripped and len(set(stripped)) == 1:
        words = [str(stripped[0])] * len(words)
    if result.config.held_treatment is not None and len(set(words)) == 1:
        return vocabulary.PLAN_HELD.format(words=words[0])
    if len(set(words)) == 1:
        return vocabulary.PLAN_EVERY_TIME.format(words=words[0])
    return "; ".join(
        vocabulary.PLAN_AT_TIME.format(words=text, t=index) for index, text in enumerate(words, 1)
    )


def _plan_rows(result: LongitudinalResult) -> list[tuple[str, int, float, float, Any]]:
    """``(label, units in the plan to the end, largest weight, effective n, fit)`` per plan.

    Read from the fit that runs deepest, as a horizon-k fit stops at k. Competing causes share
    one row when their numbers agree.
    """
    deepest = max(fit.horizon for fit in result.fits.values())
    grouped: dict[str, list[tuple[str | None, tuple[int, float, float], Any]]] = {}
    for fit in result.fits.values():
        if fit.horizon != deepest:
            continue
        numbers = (
            int(fit.scored_nodes[-1].trained_on.sum()),
            float(fit.max_weight),
            float(fit.effective_n),
        )
        grouped.setdefault(fit.regimen.label, []).append((fit.cause, numbers, fit))
    rows: list[tuple[str, int, float, float, Any]] = []
    for label, entries in grouped.items():
        if len({numbers for _, numbers, _ in entries}) == 1:
            _, numbers, fit = entries[0]
            rows.append((label, *numbers, fit))
            continue
        rows.extend((f"{label} ({cause})", *numbers, fit) for cause, numbers, fit in entries)
    return rows


def _plans_table(result: LongitudinalResult) -> SummaryTable:
    regimens = {regimen.label: regimen for regimen in result.config.regimens}
    simulated = result.config.estimator == "parametric_gformula"
    # An estimate that divides by no treatment probability has no weights to report.
    weighted = result.config.estimator not in ("gcomp", "parametric_gformula")
    # The words of a plan go last, the one column the renderer wraps to keep the width.
    tilted = any(
        isinstance(regimen, DynamicRegimen) and regimen.has_tilt
        for regimen in result.config.regimens
    )
    count_label = (
        vocabulary.FOLLOWED_IN_DATA
        if simulated
        else vocabulary.OBSERVED_TO_END
        if tilted
        else vocabulary.IN_PLAN
    )
    columns = (
        vocabulary.PLAN,
        count_label,
        *((vocabulary.LARGEST_WEIGHT, vocabulary.EFFECTIVE_N) if weighted else ()),
        vocabulary.ASSIGNS,
    )
    n = count_text(result.n)
    rows = []
    for label, followers, weight, effective, fit in _plan_rows(result):
        regimen = regimens.get(fit.regimen.label, fit.regimen)
        words = plan_words(result, regimen)
        weights = (
            (Cell(weight, number_text(weight, 1)), Cell(effective, f"{effective:,.0f}"))
            if weighted
            else ()
        )
        rows.append(
            (
                Cell(label, label),
                Cell(followers, f"{count_text(followers)} of {n}"),
                *weights,
                Cell(words, words),
            )
        )
    title = (
        vocabulary.PLANS_SPLIT_TITLE.format(repeats=result.n_repeats)
        if result.n_repeats > 1
        else vocabulary.PLANS_TITLE
    )
    return SummaryTable(columns, tuple(rows), title=title)


# ----------------------------------------------------------------- header


def _covariates(result: LongitudinalResult) -> str | None:
    data = result.data
    baseline = list(dict.fromkeys(data.baseline_names))
    later = list(dict.fromkeys(name for names in data.time_varying_names for name in names))
    listed = baseline + later
    if not listed:
        return None
    if len(listed) > _MOST_COVARIATES:
        shown = ", ".join(listed[:_SHOWN_COVARIATES])
        return f"{shown} and {len(listed) - _SHOWN_COVARIATES} more"
    text = ", ".join(baseline)
    if later:
        text = (text + "; " if text else "") + "later: " + ", ".join(later)
    return text


def _outcome_fact(result: LongitudinalResult) -> str:
    data = result.data
    config = result.config
    kind = config.outcome_kind
    if kind == "competing":
        causes = "; ".join(
            f"{cause} at {', '.join(names)}" for cause, names in _cause_columns(result).items()
        )
        return vocabulary.COMPETING_FACT.format(causes=causes)
    if kind == "survival":
        if data.time_name is not None and data.event_name is not None:
            return f"{data.event_name} by {data.time_name}"
        return f"event at {', '.join(config.outcome_names)}"
    if kind == "repeated":
        return f"{', '.join(config.outcome_names)}, measured at each time point"
    observed = np.asarray(data.observed_outcomes(), dtype=float)
    observed = observed[np.isfinite(observed)]
    binary = observed.size > 0 and bool(np.isin(observed, (0.0, 1.0)).all())
    return f"{data.outcome_name} ({'binary' if binary else 'continuous'}), at the end"


def _cause_columns(result: LongitudinalResult) -> dict[str, list[str]]:
    data = result.data
    causes = list(data.cause_labels)
    names = list(result.config.outcome_names)
    if not causes:
        return {"event": names}
    per = len(names) // len(causes) if causes else len(names)
    return {cause: names[index * per : (index + 1) * per] for index, cause in enumerate(causes)}


def _treatment_fact(result: LongitudinalResult) -> str:
    held = result.config.held_treatment
    if held is not None:
        return f"{held}, given at baseline"
    return ", ".join(result.data.treatment_names)


def _time_points(result: LongitudinalResult) -> str:
    data = result.data
    text = f"{data.n_times}"
    if data.time_grid is not None:
        text += f" (at {', '.join(f'{time:g}' for time in data.time_grid)})"
    return text


def _folds_fact(result: LongitudinalResult) -> str:
    folds = result.config.n_folds
    return "no" if folds <= 1 else f"{folds} folds"


def _header(result: LongitudinalResult) -> tuple[list[tuple[str, str]], int]:
    """The header facts and how many of them go in the left column."""
    data = result.data
    config = result.config
    left: list[tuple[str, str]] = [
        (vocabulary.QUESTION, _question(result)),
        (vocabulary.OUTCOME, _outcome_fact(result)),
        (vocabulary.TREATMENT, _treatment_fact(result)),
    ]
    covariates = _covariates(result)
    if covariates is not None:
        left.append((vocabulary.ADJUSTED_FOR, covariates))
    if data.censoring_names:
        left.append(
            (
                vocabulary.CENSORING,
                vocabulary.CENSORING_FACT.format(names=", ".join(data.censoring_names)),
            )
        )
    left.append((vocabulary.METHOD, _method(result)))
    if data.cluster is not None:
        clusters = count_text(int(np.unique(data.cluster).size)) + positive_mass_clause(
            data.cluster, data.weights if data.is_weighted else None
        )
        left.append((vocabulary.CLUSTERS, clusters))
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
        (vocabulary.UNITS if data.panel is not None else vocabulary.ROWS, count_text(result.n)),
        (vocabulary.TIME_POINTS, _time_points(result)),
        (vocabulary.OUT_OF_FOLD, _folds_fact(result)),
    ]
    if config.estimator == "parametric_gformula":
        right.extend(_gformula_models(result).items())
    else:
        learners = _learners(result)
        if learners is not None:
            for label, role in (
                (vocabulary.OUTCOME_MODEL, "outcome"),
                (vocabulary.TREATMENT_MODEL, "treatment"),
                (vocabulary.CENSORING_MODEL, "censoring"),
            ):
                fact = _model_fact(result, role, learners)
                if fact is not None:
                    right.append((label, fact))
    if config.estimator not in ("gcomp", "parametric_gformula"):
        low, high = config.g_bounds
        right.append((vocabulary.PROBABILITY_LIMITS, f"[{low:.4g}, {high:.4g}]"))
    return [*left, *right], len(left)


# ----------------------------------------------------------------- notes


def _inferential(result: LongitudinalResult) -> bool:
    return any(estimate.supplies_inference for estimate in result.estimates.values())


def _in_sample_note(result: LongitudinalResult) -> Note | None:
    """The in-sample note of an LTMLE, SDR or IPSI fit that used a flexible learner.

    The baselines' intervals need parametric models, so the note never applies to them. A
    loaded or hand-built result with no estimator record gets no note: the summary cannot
    tell which learners it used.
    """
    if result.config.n_folds > 1 or not _inferential(result):
        return None
    if result.config.estimator in _MODEL_BASED:
        return None
    learners = _learners(result)
    if learners is None:
        return None
    flexible = any(
        learner is not None and not is_parametric(learner) for learner in learners.values()
    )
    if not flexible:
        return None
    return note("in_sample_flexible", vocabulary.NOTE_TEXTS["in_sample_flexible_longitudinal"])


#: The estimators whose estimate divides by the bounded treatment and censoring
#: probabilities. G-computation and the g-formula fit them for the overlap check only.
_WEIGHTED = ("tmle", "sdr", "ipsi", "ipw")
#: The estimators that bound partial-path products rather than the full prefix.
_PATH_BOUNDED = ("sdr", "ipsi")


def max_truncated(result: LongitudinalResult) -> list[tuple[str, float, int]]:
    """``(plan, largest share of the units still following it whose probability a limit
    moved, time)``, over every split.

    One entry per plan with a positive share, from the deepest fit of each plan. On a
    sequentially doubly robust or incremental fit the share is of the followed paths whose
    partial product the limits moved, and the time is the node the paths start from.
    """
    from .estimator import LongitudinalResult as _Result

    best: dict[str, tuple[float, int]] = {}
    for repeat in result.repeats:
        fits = repeat.fits
        deepest = max(fit.horizon for fit in fits.values())
        for fit in fits.values():
            if fit.horizon != deepest:
                continue
            label = fit.regimen.label
            if result.config.estimator in _PATH_BOUNDED:
                shares = [
                    (float(step.bound_binding), int(step.time))
                    for step in fit.steps
                    if step.bound_binding is not None
                ]
                if not shares:
                    continue
                share, time = max(shares, key=lambda item: (item[0], -item[1]))
            else:
                share, time = _Result._max_truncated(fit)
            if share > best.get(label, (0.0, 0))[0]:
                best[label] = (float(share), int(time))
    return [(label, share, time) for label, (share, time) in best.items() if share > 0.0]


def _capped_note(result: LongitudinalResult) -> Note | None:
    estimator = result.config.estimator
    if estimator not in _WEIGHTED:
        return None
    low, high = result.config.g_bounds
    template = "weights_capped_paths" if estimator in _PATH_BOUNDED else "weights_capped_cumulative"
    split = vocabulary.IN_ONE_SPLIT if result.n_repeats > 1 else ""
    sentences = [
        vocabulary.NOTE_TEXTS[template].format(
            plan=plan,
            low=f"{low:.4g}",
            high=f"{high:.4g}",
            share=f"{share:.1%}",
            time=result._time_of(time),
            split=split,
        )
        for plan, share, time in max_truncated(result)
    ]
    if not sentences:
        return None
    sentences.append(vocabulary.NOTE_TEXTS["weights_capped_cumulative_effect"])
    return note("weights_capped", " ".join(sentences))


def _rule_misses(result: LongitudinalResult) -> list[str]:
    """Each update that stopped above its declared threshold, in plain words."""
    seen: set[int] = set()
    missed: list[str] = []
    draws = [repeat.fits for repeat in result.repeats] if result.repeats else [result.fits]
    fits = [(key, fit) for draw in draws for key, fit in draw.items()]
    for key, fit in fits:
        record = fit.hazard
        if (
            record is not None
            and id(record) not in seen
            and record.rule_ratio is not None
            and record.rule_ratio > 1.0
        ):
            seen.add(id(record))
            where = (
                "the hazard update of every horizon"
                if record.algorithm == "one_step"
                else f"the hazard update at time {result._time_of(fit.horizon):g}"
            )
            missed.append(f"{where} stopped at {record.rule_ratio:.3g} times its threshold")
        for step in fit.steps:
            update = step.fluctuation
            if (
                update is None
                or id(update) in seen
                or update.rule_ratio is None
                or update.rule_ratio <= 1.0
            ):
                continue
            seen.add(id(update))
            missed.append(
                f"{key} at time {result._time_of(step.time):g} stopped at "
                f"{update.rule_ratio:.3g} times its threshold"
            )
    return missed


def _score_notes(result: LongitudinalResult, spread: str) -> list[Note]:
    """The notes on how well the fit solved its estimating equations."""
    from ..fluctuation._score import wald_coverage_bound

    notes: list[Note] = []
    rule = result._declared_rule()
    draw = " in at least one split" if result.n_repeats > 1 else ""
    if rule != "relative":
        remainders = {
            name: abs(float(estimate.score)) / float(estimate.std_error)
            for name, estimate in result.estimates.items()
            if estimate.supplies_inference
            and np.isfinite(estimate.score)
            and float(estimate.std_error) > 0.0
        }
        if remainders:
            name, b = max(remainders.items(), key=lambda item: item[1])
            alpha = float(result.config.alpha_sig)
            shown = max(b, _SMALLEST_REMAINDER)
            bound = np.floor(wald_coverage_bound(shown, alpha) * 10000) / 100
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
    if not result.converged:
        what = "interval" if spread == vocabulary.SPREAD_LABELS["std_err"] else spread
        misses = _rule_misses(result) if rule != "relative" else []
        if misses:
            text = vocabulary.NOTE_TEXTS["score_rule_unmet_longitudinal"].format(
                draw=draw, misses="; ".join(misses), what=what
            )
        else:
            text = vocabulary.NOTE_TEXTS["score_unsolved_longitudinal"].format(draw=draw, what=what)
        notes.append(note("score_unsolved", text))
    return notes


def _isotonic_note(result: LongitudinalResult) -> Note | None:
    # Every split's projection, each moved level once.
    moves = list(
        dict.fromkeys(
            (regimen, cause, horizon)
            for repeat in result.repeats
            for regimen, cause, horizon, _, _ in repeat.isotonic_moves
        )
    )
    if not result.config.isotonic or not moves:
        return None
    names = []
    for regimen, cause, horizon in moves:
        where = (
            f"{regimen}" + (f", {cause}" if cause else "") + f" at time {result._time_of(horizon)}"
        )
        names.append(where)
    return note(
        "isotonic_moves",
        vocabulary.NOTE_TEXTS["isotonic_moves"].format(count=len(moves), names="; ".join(names)),
    )


def _bootstrap_band(
    result: LongitudinalResult, decimals: int, level: str
) -> tuple[SummaryTable, Note] | None:
    band = result.bootstrap_band
    if band is None or not band.bands:
        return None
    rows = tuple(
        (
            Cell(name, name),
            Cell(
                (float(low), float(high)),
                f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
            ),
        )
        for name, (low, high) in band.bands.items()
    )
    table = SummaryTable(
        ("", vocabulary.BAND), rows, title=vocabulary.BOOTSTRAP_BANDS_TITLE.format(level=level)
    )
    sentences = [
        vocabulary.NOTE_TEXTS["bootstrap_bands"].format(
            count=len(band.bands), level=level, replicates=count_text(band.n_replicates)
        )
    ]
    if band.support.label == vocabulary.EXPERIMENTAL:
        sentences.append(vocabulary.NOTE_TEXTS["bands_experimental"])
    return table, note("bootstrap_bands", " ".join(sentences))


def _fit_notes(result: LongitudinalResult, spread: str) -> list[Note]:
    config = result.config
    notes: list[Note] = []
    flexible = _in_sample_note(result)
    if flexible is not None:
        notes.append(flexible)
    capped = _capped_note(result)
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
    key = _MODEL_BASED.get(config.estimator)
    if key is not None and _inferential(result):
        mechanisms = {*result.data.treatment_names, *result.data.censoring_names}
        known = [name for name in config.known_factors if name in mechanisms]
        # An IPW fit with every factor declared known reads no fitted model.
        if not (config.estimator == "ipw" and set(known) >= mechanisms):
            text = vocabulary.NOTE_TEXTS[key]
            if known and config.estimator == "ipw":
                text += " " + vocabulary.NOTE_TEXTS["known_factors"].format(factors=_listed(known))
            notes.append(note("model_based_interval", text))
    if config.msm_terms is not None:
        notes.append(
            note(
                "working_model",
                vocabulary.NOTE_TEXTS["working_model"].format(
                    terms=", ".join(config.msm_terms), link=config.msm_link
                ),
            )
        )
    isotonic = _isotonic_note(result)
    if isotonic is not None:
        notes.append(isotonic)
    notes.extend(_score_notes(result, spread))
    kept = list(result.fit_warnings)
    if capped is not None and method_key(result) not in _PATH_BOUNDED:
        # The capped note says what the truncation warning said, with the numbers. On SDR
        # and IPSI the note reads the share of paths, another number, so the warning stays.
        restated = [
            record
            for record in kept
            if record.message.startswith(vocabulary.CUMULATIVE_TRUNCATION_WARNING)
        ]
        if restated:
            kept = [record for record in kept if record not in restated]
            notes[notes.index(capped)] = replace(capped, kind="warning")
    notes.extend(_warning_notes(kept))
    return notes


# ----------------------------------------------------------------- full detail


def _assumptions(result: LongitudinalResult) -> SummarySection:
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
        lines.extend(f"- model restriction: {item}" for item in identification.model_restrictions)
    else:
        lines.append(vocabulary.ASSUMPTIONS_LEAD_GENERIC)
        assumptions = list(vocabulary.LONGITUDINAL_ASSUMPTIONS)
        if result.config.estimator == "ipsi":
            assumptions = [*assumptions[:3], *vocabulary.TILT_ASSUMPTIONS]
        lines.extend(f"- {item}" for item in assumptions)
    return SummarySection(vocabulary.SECTION_ASSUMPTIONS, tuple(lines))


def _bound_target(result: LongitudinalResult) -> str:
    estimator = result.config.estimator
    if estimator == "sdr":
        return "each partial-path probability of treatment and staying under observation"
    if estimator == "ipsi":
        return (
            "each cumulative probability of staying under observation and of a non-tilt "
            "treatment; the tilt factor is not limited"
        )
    return "each cumulative probability of the plan's treatment and of staying under observation"


def _settings(result: LongitudinalResult) -> SummarySection:
    from .estimator import DEFAULT_LTMLE_G_BOUNDS, NO_ISOTONIC_REPEAT_BANDS

    config = result.config
    data = result.data
    lines: list[str] = []
    if config.estimator not in ("gcomp", "parametric_gformula"):
        low, high = config.g_bounds
        default = " (the default)" if config.g_bounds == DEFAULT_LTMLE_G_BOUNDS else ""
        lines.append(
            f"Probability limits [{low:.4g}, {high:.4g}]{default} on {_bound_target(result)}."
        )
    if config.known_factors:
        lines.append(
            "Declared known (no learner fits them, and no limit moves a declared value "
            f"itself): {', '.join(config.known_factors)}."
        )
    if config.q_bounds is not None:
        low, high = config.q_bounds
        lines.append(f"Outcome predictions kept inside the outcome range [{low:.4g}, {high:.4g}].")
    if config.node_bounds:
        bounds = "; ".join(
            f"{'' if cause is None else f'{cause} '}time {time:g} in [{lower:g}, {upper:g}]"
            for cause, time, lower, upper in config.node_bounds
        )
        scale = "hazard" if config.survival_model == "hazard" else "outcome"
        lines.append(f"Declared bounds on the {scale} scale: {bounds}.")
    if config.n_folds > 1:
        repeats = f", in each of {result.n_repeats} splits" if result.n_repeats > 1 else ""
        lines.append(f"Folds: {config.n_folds}{repeats}.")
    if len(config.outcome_names) > 1 and config.outcome_kind != "repeated":
        if config.survival_model == "hazard":
            how = (
                "one update for every horizon at once"
                if config.targeting == "one_step"
                else "each horizon updated on its own"
            )
            lines.append(f"Survival curve: from cause-specific hazards, {how}.")
        else:
            lines.append("Survival curve: one backward regression sequence per horizon.")
        lines.append("Each curve made non-decreasing: " + ("yes." if config.isotonic else "no."))
    if config.held_treatment is not None:
        lines.append(
            f"One baseline treatment, {config.held_treatment}, held over "
            f"{config.n_times} time points."
        )
    if config.panel is not None:
        lines.append(config.panel[0].upper() + config.panel[1:] + ".")
    for label, nodes in config.policy_point_mass_nodes:
        listed = ", ".join(str(node) for node in nodes)
        lines.append(
            f"Plan {label}: the random draw puts all its mass on one arm at time {listed}, so "
            "it is fitted as the rule it equals."
        )
    if any(isinstance(regimen, DynamicRegimen) and regimen.has_rule for regimen in config.regimens):
        lines.append(vocabulary.RULE_FOLLOWERS)
    if any(
        isinstance(regimen, DynamicRegimen) and regimen.has_policy for regimen in config.regimens
    ):
        lines.append(vocabulary.POLICY_FOLLOWERS)
    lines.extend(_hazard_lines(result))
    if result.simulation is not None:
        lines.extend(simulation_lines(result.simulation))
    estimates = result.estimates
    t_reference = any(estimate.reference_df is not None for estimate in estimates.values())
    if result.simultaneous is None and len(estimates) > 1:
        if result.n_repeats > 1 and config.isotonic:
            lines.append(NO_ISOTONIC_REPEAT_BANDS)
        elif not supplies_inference(result.inference_status):
            lines.append(NO_SIMULTANEOUS_BANDS)
        elif t_reference:
            lines.append(NO_T_REFERENCE_BANDS)
    if t_reference and data.cluster is not None:
        lines.append(
            T_REFERENCE_NOTE.format(
                clusters=fewest_clusters(
                    data.cluster, weights=data.weights if data.is_weighted else None
                ),
                stratum="",
            )
        )
    if result.n_repeats > 1:
        spreads = result.repeat_spread()
        listed = ", ".join(
            f"{name} {spread:.4g}" for name, spread in spreads.items() if np.isfinite(spread)
        )
        if listed:
            lines.append(
                f"Spread of each row across the {result.n_repeats} splits (a diagnostic, on "
                f"the scale the interval is computed on): {listed}."
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


def _hazard_lines(result: LongitudinalResult) -> list[str]:
    records = {id(fit.hazard): fit.hazard for fit in result.fits.values() if fit.hazard}
    if not records:
        return []
    converged = all(record.converged for record in records.values())
    largest = max(record.relative_score for record in records.values())
    verdict = "solved its equations" if converged else "did not solve its equations"
    first = next(iter(records.values()))
    if first.algorithm == "one_step":
        lines = [
            f"The hazard update took {first.n_iter} steps over {len(first.targets)} estimates and "
            f"{verdict}; largest relative score {largest:.2g}."
        ]
    else:
        steps = max(record.n_iter for record in records.values())
        lines = [
            f"The hazard update of each of {len(records)} horizons took at most {steps} "
            f"steps and {verdict}; largest relative score {largest:.2g}."
        ]
    if first.share_clipped:
        node, cause, share = max(first.share_clipped, key=lambda item: item[2])
        time = result._time_of(node)
        lines.append(
            f"The node bounds clipped up to {share:.1%} of the initial hazards (time {time:g}, "
            f"cause {cause + 1}); {len(first.bound_excludes_data)} cells lie outside a bound."
        )
    return lines


def simulation_lines(report: Any) -> list[str]:
    """The settings of a parametric g-formula simulation, in plain words.

    Parameters
    ----------
    report : GFormulaReport
        The fit's simulation record.

    Returns
    -------
    list of str
        One sentence per setting. No seed and no stream numbers.
    """
    from .gformula import SUPPORT_Z

    lines = [
        f"The simulation draws {report.n_draws} histories per unit for the estimate and "
        f"{report.n_derivative_draws} for the derivatives; the interval includes the "
        "simulation error."
    ]
    typed = [f"{variable} ({kind}, time {node})" for node, variable, kind in report.covariate_types]
    if typed:
        lines.append("Covariate models: " + ", ".join(typed) + ".")
    if report.monte_carlo_share:
        worst = max(report.monte_carlo_share, key=lambda name: report.monte_carlo_share[name])
        lines.append(
            f"Share of the variance from the simulation: at most {report.max_share:.1%} ({worst})."
        )
    if report.model_reliance is not None:
        lines.append(f"Model reliance: {report.model_reliance}")
    for row in report.support:
        dose = "the rule-set dose" if row.dose is None else f"dose {row.dose:g}"
        lines.append(
            f"Support of {row.plan} at {row.name}, {dose}: {row.off_support:.1%} of simulated "
            f"histories lie more than {SUPPORT_Z:g} fitted SD from the fitted mean dose; "
            f"{row.window:.1%} of observed doses lie in the same window."
        )
    return lines


def _method_details(result: LongitudinalResult) -> SummarySection:
    lines = [
        vocabulary.LONGITUDINAL_CONSTRUCTIONS.get(
            result.config.estimator, vocabulary.LONGITUDINAL_CONSTRUCTIONS["tmle"]
        )
    ]
    if result.config.n_folds > 1:
        lines.append("Each fold's models predict the units the fold leaves out.")
    return SummarySection(vocabulary.SECTION_METHOD, tuple(lines))


# ----------------------------------------------------------------- the summary


def longitudinal_summary(result: LongitudinalResult, detail: Detail) -> Summary:
    """Build the summary of a longitudinal result.

    Parameters
    ----------
    result : LongitudinalResult
        The fitted result.
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
    header, split = _header(result)
    table, notes = estimates_table(result.estimates, level=level_value)
    tables = [table]
    if len(result.estimates) > 1:
        tables.append(
            SummaryTable(
                ("", vocabulary.MEANING),
                tuple(
                    (Cell(name, name), Cell.of(meaning(result, name))) for name in result.estimates
                ),
                title=vocabulary.KEY_TITLE,
            )
        )
    if result.fits:
        tables.append(_plans_table(result))
    decimals = _decimals_of(table)
    bands = _bands(result, decimals, level)  # type: ignore[arg-type]
    if bands is not None:
        tables.append(bands[0])
        notes.append(bands[1])
    if result.bootstrap is not None:
        boot_table, boot_notes = _bootstrap(result, decimals, level)  # type: ignore[arg-type]
        if boot_table.rows:
            tables.append(boot_table)
            notes.extend(boot_notes)
    band = _bootstrap_band(result, decimals, level)
    if band is not None:
        tables.append(band[0])
        notes.append(band[1])
    notes.extend(_fit_notes(result, table.columns[2]))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(_assumptions(result))
        settings = _settings(result)
        if settings.lines:
            sections.append(settings)
        sections.append(_method_details(result))
        sections.append(_study(result))  # type: ignore[arg-type]
    return Summary.build(
        title=vocabulary.LONGITUDINAL_TITLE.format(method=_short(result)),
        header=header,
        header_split=split,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def estimates_repr(
    name: str, estimates: Mapping[str, Any], *, n: int, method: str, alpha: float
) -> str:
    """The one-line repr of a result that reports estimates.

    Parameters
    ----------
    name : str
        The class name.
    estimates : mapping of str to ParameterEstimate
        The rows.
    n : int
        The units.
    method : str
        The short method name.
    alpha : float
        The significance level of the intervals.

    Returns
    -------
    str
        For example ``LongitudinalResult(ey_regimen[always]=0.762, 95% CI [0.661, 0.863],
        +1 more, n=300, method=LTMLE)``.
    """
    items = list(estimates.items())
    facts: list[tuple[str, Any]] = []
    if items:
        first, estimate = items[0]
        if estimate.supplies_inference:
            decimals = decimals_for([estimate.std_error])
            low, high = estimate.ci
            facts.append((first, number_text(estimate.psi, decimals)))
            facts.append(
                (
                    "",
                    f"{(1 - alpha) * 100:g}% CI "
                    f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
                )
            )
        else:
            decimals = decimals_for([estimate.plugin_std_error])
            facts.append((first, f"{number_text(estimate.psi, decimals)} (no interval)"))
        if len(items) > 1:
            facts.append(("", f"+{len(items) - 1} more"))
    facts.append(("n", n))
    facts.append(("method", method))
    if any(e.support.label == vocabulary.EXPERIMENTAL for _, e in items):
        facts.append(("", vocabulary.EXPERIMENTAL))
    return short_repr(name, facts)


def longitudinal_repr(result: LongitudinalResult) -> str:
    """The one-line repr of a longitudinal result.

    Parameters
    ----------
    result : LongitudinalResult
        The fitted result.

    Returns
    -------
    str
        The repr.
    """
    return estimates_repr(
        type(result).__name__,
        result.estimates,
        n=result.n,
        method=_short(result),
        alpha=result.config.alpha_sig,
    )
