"""The printed summaries of the sensitivity analyses (X89 T4).

Each summary states the question its analysis answers, prints the numbers in tables, and
says in plain words what the analysis assumes. Symbols (``sigma^2``, ``nu^2``, the Riesz
representer) and the formal statements wait for ``detail="full"``. Every word another
summary shares comes from :mod:`cleverly._summary.vocabulary`.

Each summary carries one standing ``sensitivity_assumption`` note. That departs from the
rule that a note fires only when its condition holds, as statsmodels prints its standard
error assumption on every OLS fit: a sensitivity table cannot be read without knowing what
its strengths mean, and each text is specific to its analysis.

The builders read only fields the result already holds, so a summary computes nothing new
and a notebook display raises no warning.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

from .._inference_status import status_record, supplies_inference
from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    check_detail,
    count_text,
    note,
    number_text,
    short_repr,
    vocabulary,
)
from .._summary._cells import number_cell
from .._summary._estimates import plain_support_reason, without_roadmap
from ..inference.influence import spread_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._inference_status import InferenceStatus
    from ._mechanism_stack import MechanismRefusal
    from ._objects import MissingnessTilt, RobustnessValue, SensitivityContour
    from .evalue import EValue
    from .marginal import MarginalSensitivityBounds
    from .omitted_variable import BenchmarkResult, SensitivityBounds, SensitivityElements
    from .simulated_confounding import SimulatedConfoundingResult

V = vocabulary
TEXTS = vocabulary.NOTE_TEXTS

#: The plain noun of each second mechanism a bound's weight reads, keyed by
#: ``omitted_variable._FEATURE_MECHANISMS`` values.
_MECHANISM_NOUNS = {
    "response": "whether the outcome is recorded",
    "recording": "whether the treatment is recorded",
    "intermediate": "the intermediate variable",
}

#: The plug-in refusals that name a treatment learner the rule does not admit, and those
#: whose cause is the second-order correction (``_mechanism_stack``). Every other key is a
#: property of this fit's stored mechanism.
_LEARNER_KEYS = frozenset(
    {
        "screened",
        "cv_penalty",
        "super_learner",
        "other_learner",
        "pipeline",
        "pipeline_intercept",
        "l1_penalty",
        "liblinear",
        "class_weight",
    }
)
_CORRECTION_KEYS = frozenset({"corrected_nonpositive", "correction_unstable"})

#: The note code and plain text of each part of an E-value's note (``EValue.note_parts``).
#: A derived ratio carries no condition, so it prints only under the full settings.
_EVALUE_PARTS = {
    "conversion_or": ("approximate_scale", "evalue_conversion_or"),
    "conversion_rd": ("approximate_scale", "evalue_conversion_rd"),
    "conversion_smd": ("approximate_scale", "evalue_conversion_smd"),
    "censoring": ("censoring_conditions", "censoring_conditions"),
    "per_arm": ("per_arm_conditions", "per_arm_conditions"),
}

#: The most rows and columns a contour table prints.
_CONTOUR_ROWS = 6
_CONTOUR_COLUMNS = 5

#: A default object repr's address, which no summary prints.
#: A sentence that points into the roadmap, such as "docs/roadmap.md F26 tracks this stop."
_ROADMAP_CITE = re.compile(r"\s*[^.]*docs/roadmap\.md[^.]*\.")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


# ----------------------------------------------------------------- numbers


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _decimals(values: Iterable[Any]) -> int:
    """Decimals that show four significant figures of the largest value, within 2 to 6."""
    sizes = [abs(float(value)) for value in values if _finite(value) and float(value) != 0.0]
    if not sizes:
        return 3
    return max(2, min(6, 3 - math.floor(math.log10(max(sizes)))))


def _number(value: Any, decimals: int) -> Cell:
    return number_cell(value, decimals)


def _interval(low: Any, high: Any, decimals: int) -> Cell:
    text = f"[{number_text(low, decimals)}, {number_text(high, decimals)}]"
    if not (_finite(low) and _finite(high)):
        return Cell(None, text)
    return Cell((float(low), float(high)), text)


def _label(text: str) -> Cell:
    return Cell(text, text)


def _strength(value: Any) -> str:
    """A strength, share or alignment at 3 decimals."""
    return number_text(value, 3)


def _level(level: float) -> str:
    return f"{level * 100:g}%"


def _null(value: float) -> str:
    return f"{value:g}"


def _names(names: Sequence[Any]) -> str:
    return ", ".join(str(name) for name in names)


def _estimate_fact(estimand: str, psi: float) -> tuple[str, str]:
    return (V.ESTIMATE_ANALYSED, f"{estimand} = {number_text(psi, _decimals([psi]))}")


# ----------------------------------------------------------------- shared notes


def _alignment(rho: float) -> str:
    text = f"{_strength(rho)} (rho)"
    return text + ", worst case" if abs(rho) == 1.0 else text


def _alignment_note(rho: float) -> Note | None:
    """What a stated alignment below the worst case assumes."""
    if abs(rho) == 1.0:
        return None
    return note("alignment_stated", TEXTS["alignment_stated"].format(rho=_strength(rho)))


def _alignment_clause(rho: float) -> str:
    return "" if abs(rho) == 1.0 else f" at alignment {_strength(rho)}"


def _joint_note(features: Sequence[str]) -> Note | None:
    from .omitted_variable import _FEATURE_MECHANISMS

    nouns = [
        _MECHANISM_NOUNS[noun]
        for name in features
        if (noun := _FEATURE_MECHANISMS.get(name)) is not None and noun in _MECHANISM_NOUNS
    ]
    if not nouns:
        return None
    named = nouns[0] if len(nouns) == 1 else f"{', '.join(nouns[:-1])} and {nouns[-1]}"
    return note("joint_strength", TEXTS["joint_strength"].format(mechanisms=named))


def _learned_rule_note(features: Sequence[str]) -> Note | None:
    """What a bound of a learned-rule value holds fixed, or ``None`` on any other bound."""
    if "learned_rule" not in features:
        return None
    return note("learned_rule_bound", TEXTS["learned_rule_bound"])


def _status_note(status: str, names: str, label: str) -> Note:
    """The shared note of a table with a diagnostic spread column."""
    record = status_record(status)
    return note(
        "no_inference",
        TEXTS["no_inference"].format(names=names, reason=record.plain_reason, label=label),
    )


def _limit_status_note(status: str, names: str, label: str) -> Note:
    """The note of a bound whose limits are a diagnostic at the estimate's status."""
    record = status_record(status)
    return note(
        "no_inference_sensitivity",
        TEXTS["no_inference_sensitivity"].format(
            names=names, reason=record.plain_reason, label=label
        ),
    )


def _limits_withheld(refusal: MechanismRefusal | None) -> Note:
    """Why a bound has no one-sided limits, worded by the rule that withheld them."""
    if refusal is None:
        return note("limits_withheld", TEXTS["limits_withheld_zero"])
    if refusal.key == "consistent_mechanism":
        key = "limits_withheld_consistent"
    elif refusal.key == "learned_rule":
        key = "limits_withheld_learned_rule"
    elif refusal.key == "configuration":
        key = "limits_withheld_configuration"
    elif refusal.key in _CORRECTION_KEYS:
        key = "limits_withheld_correction"
    elif refusal.key in _LEARNER_KEYS:
        key = "limits_withheld_learner"
    else:
        key = "limits_withheld_fit"
    alternative = without_roadmap(refusal.alternative) if refusal.alternative else ""
    return note("limits_withheld", TEXTS[key].format(alternative=alternative).strip())


def _descriptive(what: str) -> Note:
    return note("limits_descriptive", TEXTS["limits_descriptive"].format(what=what))


def _definitions_bounds() -> SummarySection:
    return SummarySection(
        "Definitions",
        (
            "sigma^2: the outcome variance left after adjustment, E[(Y - E[Y | A, W])^2].",
            "nu^2: the second moment E[alpha^2] of the Riesz representer alpha, the weight "
            "that turns an outcome into the estimate (for the ATE, A / g(W) - (1 - A) / "
            "(1 - g(W))).",
            "cf_y: the share of sigma^2 that the left-out variables would explain.",
            "cf_d: the share of nu^2 that only the left-out variables would add, "
            "1 - nu_short^2 / nu^2.",
            "rho: the correlation of the two parts of the bias the left-out variables would "
            "cause; |rho| = 1 is the worst case.",
        ),
    )


_BOUND_ASSUMPTION = (
    "|bias| <= |rho| sqrt(cf_y cf_d / (1 - cf_d)) sqrt(sigma^2 nu^2) "
    "(Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis, Long story short, Theorem 2)."
)
_LIMIT_ASSUMPTION = (
    "The one-sided limits add the sampling error of each end of the range, from the "
    "influence curves of the estimate and of sqrt(sigma^2 nu^2) (same source, Theorem 4)."
)


def _nu2_setting(nu2_estimator: str, correction: float | None, mechanism: str | None = None) -> str:
    if nu2_estimator == "doubly_robust":
        return "Estimate of nu^2: doubly robust."
    if correction is None:
        return "Estimate of nu^2: plug-in, uncorrected."
    text = f"Estimate of nu^2: plug-in, second-order corrected by {correction:.5g}"
    if mechanism is None:
        return text + "."
    source = (
        "the declared known mechanism"
        if mechanism == "known"
        else "the score of the fitted LogisticRegression mechanism, stacked"
    )
    return f"{text}; its standard error carries {source}."


def _formal(text: str) -> str:
    """A stored formal sentence, without roadmap pointers or object addresses."""
    return _ADDRESS.sub("", _ROADMAP_CITE.sub("", without_roadmap(text))).strip()


# ----------------------------------------------------------------- SensitivityBounds


def _rv_verdict(rv: float | None, null: float, rho: float) -> str:
    alignment = _alignment_clause(rho)
    if rv is None:
        return TEXTS["verdict_rv_unreached"].format(null=_null(null), alignment=alignment)
    if rv == 0.0:
        return TEXTS["verdict_rv_at_null"].format(null=_null(null))
    return TEXTS["verdict_rv_reached"].format(
        rv=_strength(rv), null=_null(null), alignment=alignment
    )


def _rv_limit_text(inferential: bool, limit: str, level: str, null: float) -> str:
    """What the limit robustness value means: a confidence limit's, or a diagnostic's."""
    if not inferential:
        return TEXTS["rv_limit_plugin"].format(null=_null(null))
    return TEXTS["rv_limit"].format(limit=limit, level=level, null=_null(null))


def _rv_cell(value: float | None) -> Cell:
    if value is None:
        return Cell(None, "none below 1")
    return Cell(float(value), _strength(value))


def bounds_summary(bounds: SensitivityBounds, detail: Detail) -> Summary:
    """The summary of an omitted-confounder bound.

    Parameters
    ----------
    bounds : SensitivityBounds
        The bound.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    status = bounds.inference
    null = bounds.null_hypothesis
    derived = bounds._limits_derived
    inferential = supplies_inference(status)
    excludes = (bounds.lower - null) * (bounds.upper - null) > 0
    header = [
        (V.QUESTION, f"how far left-out confounders could move {bounds.estimand}"),
        _estimate_fact(bounds.estimand, bounds.psi),
        (V.ALIGNMENT, _alignment(bounds.rho)),
        (V.OUTCOME_STRENGTH, f"{_strength(bounds.cf_y)} (cf_y)"),
        (V.TREATMENT_STRENGTH, f"{_strength(bounds.cf_d)} (cf_d)"),
        (V.LARGEST_SHIFT, number_text(bounds.bias, _decimals([bounds.psi, bounds.bias]))),
        (V.CONCLUSION, f"range {'excludes' if excludes else 'includes'} {_null(null)}"),
    ]
    values: list[Any] = [bounds.psi, bounds.lower, bounds.upper]
    if derived:
        values += [bounds._ci_lower, bounds._ci_upper]
    decimals = _decimals(values)
    columns = ["", V.ESTIMATE, "range"]
    row = [
        _label(bounds.estimand),
        _number(bounds.psi, decimals),
        _interval(bounds.lower, bounds.upper, decimals),
    ]
    level = _level(bounds.level)
    if derived:
        if inferential:
            columns.append(f"{level} one-sided limits")
        else:
            columns.append(V.SPREAD_LABELS["plugin_interval"])
        row.append(_interval(bounds._ci_lower, bounds._ci_upper, decimals))
    tables = [SummaryTable(tuple(columns), (tuple(row),))]
    rv_rows = [(_label("estimate"), _rv_cell(bounds.robustness_value))]
    if derived:
        name = f"{level} limit" if inferential else "plug-in limit (diagnostic)"
        rv_rows.append((_label(name), _rv_cell(bounds._robustness_value_ci)))
    tables.append(SummaryTable(("", "robustness value"), tuple(rv_rows), title="Robustness value"))

    assumption = TEXTS["assumption_bounds"]
    if derived and inferential:
        assumption += " " + TEXTS["assumption_bounds_limits"].format(level=level)
    notes = [note("sensitivity_assumption", assumption)]
    if derived:
        notes.append(
            note(
                "rv_limit",
                _rv_limit_text(inferential, rv_rows[1][0].text, level, null),
            )
        )
    alignment = _alignment_note(bounds.rho)
    if alignment is not None:
        notes.append(alignment)
    if derived and not inferential:
        notes.append(
            _limit_status_note(status, bounds.estimand, V.SPREAD_LABELS["plugin_interval"])
        )
    if not derived:
        notes.append(_limits_withheld(bounds.limit_refusal))
    elif bounds.limit_descriptive:
        notes.append(_descriptive("one-sided limits"))
    joint = _joint_note(bounds.features)
    if joint is not None:
        notes.append(joint)
    learned = _learned_rule_note(bounds.features)
    if learned is not None:
        notes.append(learned)
    if bounds.assumption is not None:
        notes.append(note("consistent_mechanism", TEXTS["consistent_mechanism"]))

    sections: list[SummarySection] = []
    if detail == "full":
        assumptions = [_BOUND_ASSUMPTION]
        if derived:
            assumptions.append(_LIMIT_ASSUMPTION)
        if bounds.assumption is not None:
            assumptions.append(_formal(bounds.assumption))
        settings = [
            f"Bias scale sqrt(sigma^2 nu^2): {bounds.max_bias:.4g}.",
            _nu2_setting(bounds.nu2_estimator, bounds.nu2_correction, bounds.plugin_mechanism),
            f"Null value: {_null(null)}.",
            f"Confidence level of the limits: {level}.",
        ]
        reason = bounds._limit_reason()
        if reason is not None:
            settings.append(f"Why the limits are withheld: {_formal(reason)}")
        if bounds.limit_reading is not None and derived:
            settings.append(f"Evidence for the limits: {_formal(bounds.limit_reading.sentence)}")
        sections = [
            SummarySection(V.SECTION_ASSUMPTIONS, tuple(assumptions)),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Bounds under omitted confounding",
        header=header,
        header_split=3,
        tables=tables,
        notes=notes,
        reading=_rv_verdict(bounds.robustness_value, null, bounds.rho),
        sections=sections,
    )


def bounds_repr(bounds: SensitivityBounds) -> str:
    """The one-line repr of an omitted-confounder bound."""
    decimals = _decimals([bounds.psi, bounds.lower, bounds.upper])
    return short_repr(
        "SensitivityBounds",
        [
            (bounds.estimand, number_text(bounds.psi, decimals)),
            (
                "",
                f"range [{number_text(bounds.lower, decimals)}, "
                f"{number_text(bounds.upper, decimals)}]",
            ),
            ("cf_y", f"{bounds.cf_y:g}"),
            ("cf_d", f"{bounds.cf_d:g}"),
        ],
    )


# ----------------------------------------------------------------- RobustnessValue


def robustness_summary(value: RobustnessValue, detail: Detail) -> Summary:
    """The summary of a robustness value.

    Parameters
    ----------
    value : RobustnessValue
        The values.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    status = cast("InferenceStatus", value.get("inference", "influence_curve"))
    inferential = supplies_inference(status)
    null = value.null_hypothesis
    level = _level(value.level)
    limit_key = spread_name("rva", status)
    derived = "limit_refusal" not in value
    header = [
        (V.QUESTION, f"how strong left-out confounders must be to move {value.estimand}"),
        _estimate_fact(value.estimand, value.psi),
        (V.ALIGNMENT, _alignment(value.rho)),
    ]
    rows = [(_label("estimate"), _rv_cell(value["rv"]))]
    if derived:
        name = f"{level} limit" if inferential else "plug-in limit (diagnostic)"
        rows.append((_label(name), _rv_cell(value[limit_key])))
    tables = [SummaryTable(("", "robustness value"), tuple(rows))]
    notes = [
        note("sensitivity_assumption", TEXTS["assumption_robustness"].format(null=_null(null)))
    ]
    if derived:
        notes.append(
            note(
                "rv_limit",
                _rv_limit_text(inferential, rows[1][0].text, level, null),
            )
        )
    alignment = _alignment_note(value.rho)
    if alignment is not None:
        notes.append(alignment)
    if derived and not inferential:
        notes.append(_limit_status_note(status, value.estimand, V.SPREAD_LABELS[limit_key]))
    if not derived:
        notes.append(_limits_withheld(value.refusal))
        if value.refusal is not None and value.refusal.key == "learned_rule":
            notes.append(note("learned_rule_bound", TEXTS["learned_rule_bound"]))
    elif value.get("limit_licensed") is False:
        notes.append(_descriptive("limit robustness value"))
    sections: list[SummarySection] = []
    if detail == "full":
        settings = [
            f"Bias scale sqrt(sigma^2 nu^2): {value['max_bias']:.4g}.",
            _nu2_setting(value.get("nu2_estimator", "doubly_robust"), value.get("nu2_correction")),
            f"Null value: {_null(null)}.",
        ]
        if not derived:
            settings.append(
                f"Why the limit value is withheld: {_formal(str(value['limit_refusal']))}"
            )
        if "limit_sentence" in value:
            settings.append(f"Evidence for the limit: {_formal(value['limit_sentence'])}")
        sections = [
            SummarySection(V.SECTION_ASSUMPTIONS, (_BOUND_ASSUMPTION,)),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Robustness value",
        header=header,
        header_split=2,
        tables=tables,
        notes=notes,
        reading=_rv_verdict(value["rv"], null, value.rho),
        sections=sections,
    )


def robustness_repr(value: RobustnessValue) -> str:
    """The one-line repr of a robustness value."""
    facts: list[tuple[str, Any]] = [("", value.estimand)]
    for key, item in value.items():
        if key.startswith("rv"):
            facts.append((key, "none" if item is None else _strength(item)))
    return short_repr("RobustnessValue", facts)


# ----------------------------------------------------------------- SensitivityElements


def elements_summary(elements: SensitivityElements, detail: Detail) -> Summary:
    """The summary of the inputs to an omitted-confounder bound.

    Parameters
    ----------
    elements : SensitivityElements
        The inputs.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    header = [(V.QUESTION, f"the scales of the bias bound for {elements.estimand}")]
    decimals = _decimals([elements.sigma2, elements.nu2, elements.max_bias])
    rows = [
        (_label("outcome variance left after adjustment"), _number(elements.sigma2, decimals)),
        (_label("average squared weight"), _number(elements.nu2, decimals)),
        (_label("bias scale"), _number(elements.max_bias, decimals)),
    ]
    if elements.nu2_correction is not None:
        rows.append((_label("plug-in correction"), _number(elements.nu2_correction, decimals)))
    tables = [SummaryTable(("", "value"), tuple(rows))]
    notes = [note("sensitivity_assumption", TEXTS["assumption_elements"])]
    if elements.psi_max_bias is None:
        notes.append(_limits_withheld(elements.limit_refusal))
    joint = _joint_note(elements.features)
    if joint is not None:
        notes.append(joint)
    learned = _learned_rule_note(elements.features)
    if learned is not None:
        notes.append(learned)
    sections: list[SummarySection] = []
    if detail == "full":
        settings = [
            _nu2_setting(elements.nu2_estimator, elements.nu2_correction, elements.plugin_mechanism)
        ]
        if elements.limit_refusal is not None:
            settings.append(f"Why the limits are withheld: {_formal(elements.limit_refusal.text)}")
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "The rows are sigma^2, nu^2 and sqrt(sigma^2 nu^2). With influence "
                    "curves for each, a bound built from them has one-sided limits.",
                    _BOUND_ASSUMPTION,
                ),
            ),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Inputs to the omitted-confounding bounds",
        header=header,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def elements_repr(elements: SensitivityElements) -> str:
    """The one-line repr of the bound inputs; the influence curves never print."""
    return short_repr(
        "SensitivityElements",
        [
            ("", elements.estimand),
            ("sigma2", f"{elements.sigma2:.4g}"),
            ("nu2", f"{elements.nu2:.4g}"),
            ("max_bias", f"{elements.max_bias:.4g}"),
        ],
    )


# ----------------------------------------------------------------- BenchmarkResult


def benchmark_summary(record: BenchmarkResult, detail: Detail) -> Summary:
    """The summary of a benchmark against observed covariates.

    Parameters
    ----------
    record : BenchmarkResult
        The benchmark.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    covariates = _names(record.covariates)
    decimals = _decimals([record.psi_long, record.psi_short])
    header = [
        (V.QUESTION, f"how strong confounders like {covariates} would be"),
        (V.ESTIMATE_ANALYSED, f"{record.estimand} = {number_text(record.psi_long, decimals)}"),
        (V.BENCHMARK_COVARIATES, covariates),
        (V.OUTCOME_STRENGTH, f"{_strength(record.cf_y)} (cf_y)"),
        (V.TREATMENT_STRENGTH, f"{_strength(record.cf_d)} (cf_d)"),
        (V.IMPLIED_ALIGNMENT, f"{_strength(record.rho)} (rho)"),
    ]
    scales = _decimals([record.sigma2_long, record.sigma2_short, record.nu2_long, record.nu2_short])
    rows = (
        (
            _label("estimate"),
            _number(record.psi_long, decimals),
            _number(record.psi_short, decimals),
        ),
        (
            _label("outcome variance left after adjustment"),
            _number(record.sigma2_long, scales),
            _number(record.sigma2_short, scales),
        ),
        (
            _label("average squared weight"),
            _number(record.nu2_long, scales),
            _number(record.nu2_short, scales),
        ),
    )
    tables = [
        SummaryTable(
            ("", "with", "without"), rows, title="With and without the benchmark covariates"
        )
    ]
    clipped_d = record.gain_d >= 1.0
    assumption = TEXTS["assumption_benchmark"].format(
        covariates=covariates, delta=f"{record.delta_psi:+.{decimals}f}"
    )
    if not clipped_d:
        assumption += " " + TEXTS["benchmark_pointer"]
    notes = [note("sensitivity_assumption", assumption)]
    clipped = []
    if record.gain_y > 1.0:
        clipped.append(
            TEXTS["strength_clipped_y"].format(covariates=covariates, gain=f"{record.gain_y:.3g}")
        )
    if clipped_d:
        clipped.append(TEXTS["strength_clipped_d"].format(gain=f"{record.gain_d:.3g}"))
    if clipped:
        notes.append(note("strength_clipped", " ".join(clipped)))
    joint = _joint_note(record.features)
    if joint is not None:
        notes.append(joint)
    sections: list[SummarySection] = []
    if detail == "full":
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "cf_y is the outcome gain (sigma^2_short / sigma^2_long - 1) and cf_d the "
                    "weight gain (nu^2_long / nu^2_short - 1), each clipped to [0, 1]. The "
                    "gains are proxies for the strengths (Chernozhukov et al., Long story "
                    "short, Appendix E.5, Remark 8).",
                    "rho is the alignment those covariates imply. A value well below 1 says "
                    "the worst case rho = 1 is pessimistic for confounders like these.",
                ),
            ),
            SummarySection(
                V.SECTION_SETTINGS,
                (
                    f"Outcome gain: {record.gain_y:.4g}; weight gain: {record.gain_d:.4g}.",
                    f"Change in the estimate when they are left out: {record.delta_psi:+.4g}.",
                ),
            ),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Benchmark against observed covariates",
        header=header,
        header_split=2,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def benchmark_repr(record: BenchmarkResult) -> str:
    """The one-line repr of a benchmark."""
    return short_repr(
        "BenchmarkResult",
        [
            ("", _names(record.covariates)),
            ("cf_y", _strength(record.cf_y)),
            ("cf_d", _strength(record.cf_d)),
            ("rho", _strength(record.rho)),
        ],
    )


# ----------------------------------------------------------------- SensitivityContour


def _spread_indices(size: int, most: int) -> list[int]:
    """At most ``most`` evenly spaced indices of ``range(size)``, both ends included."""
    if size <= most:
        return list(range(size))
    return sorted({round(step * (size - 1) / (most - 1)) for step in range(most)})


def contour_summary(contour: SensitivityContour, detail: Detail) -> Summary:
    """The summary of a grid of bias-adjusted bounds.

    Parameters
    ----------
    contour : SensitivityContour
        The grid.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    cf_d = contour.column("cf_d")
    cf_y = contour.column("cf_y")
    value = contour.column("value")
    d_values = sorted(set(cf_d))
    y_values = sorted(set(cf_y))
    cells = {(d, y): v for d, y, v in zip(cf_d, cf_y, value, strict=True)}
    header = [
        (V.QUESTION, f"the {contour.bound} end of the range over strengths"),
        _estimate_fact(contour.estimand, contour.psi),
        (V.ALIGNMENT, _alignment(contour.rho)),
        (
            V.GRID,
            f"cf_y to {_strength(max(y_values))}, cf_d to {_strength(max(d_values))}",
        ),
    ]
    rows_shown = [y_values[index] for index in _spread_indices(len(y_values), _CONTOUR_ROWS)]
    columns_shown = [d_values[index] for index in _spread_indices(len(d_values), _CONTOUR_COLUMNS)]
    decimals = _decimals(value)
    rows = tuple(
        (Cell(y, _strength(y)), *(_number(cells.get((d, y)), decimals) for d in columns_shown))
        for y in rows_shown
    )
    columns = ("cf_y \\ cf_d", *(f"{_strength(d)}" for d in columns_shown))
    tables = [SummaryTable(columns, rows, title=f"{contour.bound.capitalize()} end of the range")]
    notes = [
        note(
            "sensitivity_assumption",
            TEXTS["assumption_contour"].format(
                bound=contour.bound, estimand=contour.estimand, rho=_strength(contour.rho)
            ),
        )
    ]
    if len(rows_shown) < len(y_values) or len(columns_shown) < len(d_values):
        notes.append(
            note(
                "grid_thinned",
                TEXTS["grid_thinned"].format(
                    rows=len(rows_shown),
                    columns=len(columns_shown),
                    y_size=len(y_values),
                    d_size=len(d_values),
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections = [
            SummarySection(V.SECTION_ASSUMPTIONS, (_BOUND_ASSUMPTION,)),
            SummarySection(
                V.SECTION_SETTINGS,
                (f"Grid: {len(y_values)} cf_y values by {len(d_values)} cf_d values.",),
            ),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Bias-adjusted bounds over a grid of strengths",
        header=header,
        header_split=2,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def contour_repr(contour: SensitivityContour) -> str:
    """The one-line repr of a grid of bounds."""
    d_size = len(set(contour.column("cf_d")))
    y_size = len(set(contour.column("cf_y")))
    return short_repr(
        "SensitivityContour",
        [("", contour.estimand), ("", f"{contour.bound} end"), ("", f"{y_size} x {d_size}")],
    )


# ----------------------------------------------------------------- EValue


def evalue_summary(value: EValue, detail: Detail) -> Summary:
    """The summary of an E-value.

    Parameters
    ----------
    value : EValue
        The E-value.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    scale = value.scale + (", converted" if value.approximate else "")
    header = [
        (V.QUESTION, f"how strong a confounder would explain away {value.estimand}"),
        (V.ESTIMATE_ANALYSED, value.estimand),
        (V.SCALE, scale),
    ]
    low, high = value.risk_ratio_ci
    nearest = low if value.risk_ratio >= 1.0 else high
    limit_name = f"{_level(value.level)} limit nearest the null"
    rows = (
        (_label("estimate"), _number(value.risk_ratio, 3), _number(value.point, 2)),
        (_label(limit_name), _number(nearest, 3), _number(value.limit, 2)),
    )
    tables = [SummaryTable(("", "risk ratio", "E-value"), rows)]
    limit_sentence = (
        TEXTS["evalue_limit_reached"].format(limit=number_text(value.limit, 2))
        if value.limit > 1.0
        else TEXTS["evalue_limit_covers"]
    )
    single = value.bias_factor == "single"
    key = "assumption_evalue" if single else "assumption_evalue_per_arm"
    verdict_key = "verdict_evalue" if single else "verdict_evalue_per_arm"
    notes = [note("sensitivity_assumption", TEXTS[key])]
    for kind, _ in value.note_parts:
        if kind in _EVALUE_PARTS:
            code, text_key = _EVALUE_PARTS[kind]
            notes.append(note(code, TEXTS[text_key]))
    if value.truncated_bound is not None:
        notes.append(
            note(
                "bound_truncated",
                TEXTS[
                    "bound_truncated" if value.risk_ratio >= 1.0 else "bound_truncated_upper"
                ].format(value=f"{value.truncated_bound:.3g}"),
            )
        )
    if value.support.label == "experimental":
        notes.append(note("experimental", plain_support_reason(value.support.reason)))
    if value.limit_descriptive:
        notes.append(_descriptive("E-value of the limit"))
    sections: list[SummarySection] = []
    if detail == "full":
        settings = [
            f"Risk ratio {value.risk_ratio:.4g}, {_level(value.level)} interval "
            f"[{low:.4g}, {high:.4g}].",
        ]
        for kind, _ in value.note_parts:
            if kind == "derived":
                settings.append(
                    TEXTS["evalue_derived"].format(
                        name=value.estimand, source=value.source_estimand
                    )
                )
        if value.limit_reading is not None:
            settings.append(f"Evidence for the limit: {_formal(value.limit_reading.sentence)}")
        if value.support.label == "experimental":
            settings.append(f"Why experimental: {plain_support_reason(value.support.reason)}")
        formal = tuple(text for kind, text in value.note_parts if kind != "derived")
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (_EVALUE_FORMULA if single else _EVALUE_PER_ARM_FORMULA, *formal),
            ),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
        ]
    return Summary.build(
        title="E-value",
        header=header,
        header_split=1,
        tables=tables,
        notes=notes,
        reading=TEXTS[verdict_key].format(
            point=number_text(value.point, 2), limit_sentence=limit_sentence
        ),
        sections=sections,
    )


_EVALUE_FORMULA = (
    "E-value = RR + sqrt(RR (RR - 1)) for a risk ratio RR >= 1 (its inverse below 1): the "
    "smallest equal strength, on the risk-ratio scale, of the confounder's association with "
    "the treatment and with the outcome that could move RR to 1 (VanderWeele and Ding, 2017)."
)
_EVALUE_PER_ARM_FORMULA = (
    "E-value = sqrt(RR) + sqrt(sqrt(RR) (sqrt(RR) - 1)) for a risk ratio RR >= 1 (its inverse "
    "below 1): the bias is a product of one bounding factor per arm (Ding and VanderWeele, "
    "2016; Smith, Mathur and VanderWeele, 2021), and with four equal strengths E the product "
    "is BF(E, E)^2, so the E-value inverts the square root of RR."
)


def evalue_repr(value: EValue) -> str:
    """The one-line repr of an E-value."""
    facts: list[tuple[str, Any]] = [
        ("", value.estimand),
        ("point", number_text(value.point, 2)),
        ("limit", number_text(value.limit, 2)),
    ]
    if value.support.label == "experimental":
        facts.append(("", "experimental"))
    return short_repr("EValue", facts)


# ----------------------------------------------------------------- MarginalSensitivityBounds


def _breakdown_row(name: str, gamma: float | None, grid: Sequence[float]) -> tuple[Cell, ...]:
    """The grid values around the first one whose range or interval includes the null.

    The smallest such gamma lies above the last grid value that excludes the null and at
    or below the first that includes it, so the row prints both.
    """
    if gamma is None:
        last = max(grid, default=1.0)
        return (_label(name), Cell(None, f"none up to {last:g}"), Cell(last, f"{last:g}"))
    earlier = [value for value in grid if value < gamma]
    before = Cell(max(earlier), f"{max(earlier):g}") if earlier else Cell(None, V.MISSING)
    return (_label(name), Cell(gamma, f"{gamma:g}"), before)


def marginal_summary(bounds: MarginalSensitivityBounds, detail: Detail) -> Summary:
    """The summary of marginal sensitivity model bounds.

    Parameters
    ----------
    bounds : MarginalSensitivityBounds
        The bounds.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    grid = sorted(row.gamma for row in bounds.rows)
    header = [
        (V.QUESTION, f"the range of {bounds.estimand} as treatment odds may shift"),
        (V.OUTCOME, bounds.outcome_kind),
    ]
    decimals = _decimals(
        value
        for row in bounds.rows
        for value in (row.lower, row.upper, row.lower_limit, row.upper_limit)
    )
    level = _level(bounds.level)
    rows = tuple(
        (
            Cell(row.gamma, f"{row.gamma:g}"),
            _number(row.lower, decimals),
            _number(row.upper, decimals),
            _interval(row.lower_limit, row.upper_limit, decimals),
        )
        for row in bounds.rows
    )
    tables = [SummaryTable(("gamma", "lower", "upper", f"{level} interval for the range"), rows)]
    if bounds.null is not None:
        null = _null(bounds.null)
        tables.append(
            SummaryTable(
                ("", f"first grid gamma that includes {null}", "last that excludes it"),
                (
                    _breakdown_row("range", bounds.breakdown, grid),
                    _breakdown_row(f"{level} interval", bounds.limit_breakdown, grid),
                ),
                title=f"Where the range reaches {null}",
            )
        )
    assumption = TEXTS["assumption_marginal"].format(estimand=bounds.estimand)
    if not bounds.limit_descriptive:
        assumption += " " + TEXTS["marginal_interval"].format(level=level)
    notes = [note("sensitivity_assumption", assumption)]
    if bounds.limit_descriptive:
        notes.append(_descriptive("interval for the range"))
    sections: list[SummarySection] = []
    if detail == "full":
        settings = []
        if bounds.null is not None:
            settings.append(
                f"Null value: {_null(bounds.null)}. The smallest gamma at which the range "
                "or interval reaches it lies between the two grid values its row prints."
            )
        if bounds.quantile_learner:
            settings.append(f"Quantile learners: {_names(bounds.quantile_learner)}.")
        if bounds.pseudo_outcome_learner:
            settings.append(f"Tail-regression learners: {_names(bounds.pseudo_outcome_learner)}.")
        if bounds.scale == "log":
            settings.append("The standard errors of the ends are on the log scale.")
        if bounds.limit_reading is not None:
            settings.append(f"Evidence for the interval: {_formal(bounds.limit_reading.sentence)}")
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "Marginal sensitivity model: the odds ratio between P(A = a | W, U) and "
                    "P(A = a | W) lies in [1 / gamma, gamma] for every unit (Tan, 2006).",
                    "The interval covers the identified set with asymptotic probability at "
                    "least the level, by a union bound over its two ends, when either the "
                    "treatment model or the outcome quantile model is consistent.",
                ),
            ),
            SummarySection(
                V.SECTION_SETTINGS, tuple(settings) or ("No settings beyond the grid.",)
            ),
        ]
    return Summary.build(
        title="Bounds under the marginal sensitivity model",
        header=header,
        header_split=1,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def marginal_repr(bounds: MarginalSensitivityBounds) -> str:
    """The one-line repr of marginal sensitivity model bounds."""
    gammas = [row.gamma for row in bounds.rows]
    facts: list[tuple[str, Any]] = [("", bounds.estimand)]
    if gammas:
        facts.append(("", f"gamma {min(gammas):g} to {max(gammas):g}"))
    if bounds.null is not None:
        facts.append(("breakdown", "none" if bounds.breakdown is None else f"{bounds.breakdown:g}"))
    return short_repr("MarginalSensitivityBounds", facts)


# ----------------------------------------------------------------- MissingnessTilt


def _value(payload: dict[str, tuple[Any, ...]], key: str, index: int) -> Any:
    column = payload.get(key)
    return None if column is None else column[index]


def _spread_decimals(spreads: Iterable[Any]) -> int:
    """Decimals that show two significant figures of the smallest spread, within 2 to 6."""
    sizes = [abs(float(s)) for s in spreads if _finite(s) and float(s) > 0]
    return max(2, min(6, 1 - math.floor(math.log10(min(sizes or [1e-2])))))


def tilt_summary(tilt: MissingnessTilt, detail: Detail) -> Summary:
    """The summary of a missingness tilt.

    Parameters
    ----------
    tilt : MissingnessTilt
        The tilt.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    names = tilt.estimands
    if tilt.arm_gamma is None:
        direction = "every arm alike"
    else:
        direction = ", ".join(f"arm {arm} x {value:g}" for arm, value in tilt.arm_gamma)
    header = [
        (V.QUESTION, "how the estimates move if missing outcomes differ"),
        (V.MISSING_OUTCOMES, f"{count_text(tilt.n_missing)} of {count_text(tilt.n)}"),
        (V.DIRECTION, direction),
    ]
    payload = dict(tilt.rows)
    count = len(tilt)
    statuses = dict(tilt.statuses)
    tables = []
    diagnostic: list[str] = []
    withheld: list[str] = []
    for name in names:
        indices = [index for index in range(count) if payload["estimand"][index] == name]
        status = cast("InferenceStatus", statuses.get(name, "influence_curve"))
        inferential = supplies_inference(status)
        if not inferential:
            withheld.append(name)
        spread_key = spread_name("std_err", status)
        low_key = spread_name("ci_lower", status)
        high_key = spread_name("ci_upper", status)
        rows_values = []
        moved_diagnostic = False
        for index in indices:
            spread = _value(payload, spread_key, index)
            low, high = _value(payload, low_key, index), _value(payload, high_key, index)
            if not _finite(spread) and _finite(_value(payload, "plugin_std_err", index)):
                # Off gamma = 0 a parameter that claims no interval carries the diagnostic.
                spread = _value(payload, "plugin_std_err", index)
                low = _value(payload, "plugin_interval_lower", index)
                high = _value(payload, "plugin_interval_upper", index)
                moved_diagnostic = True
            rows_values.append((index, spread, low, high))
        if moved_diagnostic:
            diagnostic.append(name)
        decimals = _spread_decimals(spread for _, spread, _, _ in rows_values)
        spread_label = (
            V.SPREAD_LABELS[spread_key] if inferential else V.SPREAD_LABELS["plugin_std_err"]
        )
        interval_label = (
            V.interval_label(tilt.level) if inferential else V.SPREAD_LABELS["plugin_interval"]
        )
        rows = []
        for index, spread, low, high in rows_values:
            gamma = float(payload["gamma"][index])
            mar = bool(payload["is_mar"][index])
            label = f"{gamma:g}"
            mark = moved_diagnostic and inferential and not mar
            rows.append(
                (
                    Cell(gamma, label + (V.NO_INFERENCE_MARK if mark else "")),
                    _number(payload["psi"][index], decimals),
                    _number(spread, decimals),
                    _interval(low, high, decimals),
                )
            )
        tables.append(
            SummaryTable(
                ("gamma", V.ESTIMATE, spread_label, interval_label), tuple(rows), title=name
            )
        )
    arm_clause = (
        ""
        if tilt.arm_gamma is None
        else ", multiplied for each arm by its declared factor (" + direction + ")"
    )
    direction_text = TEXTS["tilt_direction" if tilt.arm_gamma is None else "tilt_direction_arms"]
    moved = any(float(gamma) != 0.0 for gamma in payload["gamma"])
    reused = moved and any(name not in diagnostic and name not in withheld for name in names)
    if reused:
        direction_text += " " + TEXTS["tilt_interval"]
    assumption = TEXTS["assumption_tilt"].format(arm_clause=arm_clause, direction=direction_text)
    notes = [note("sensitivity_assumption", assumption)]
    if withheld:
        first = statuses[withheld[0]]
        notes.append(_status_note(first, _names(withheld), V.SPREAD_LABELS["plugin_std_err"]))
    if diagnostic:
        log = all(name in tilt.ratios for name in diagnostic)
        some = any(name in tilt.ratios for name in diagnostic)
        scale = (
            ", on the log scale"
            if log
            else f", on the log scale for {_names([n for n in diagnostic if n in tilt.ratios])}"
            if some
            else ""
        )
        notes.append(
            note(
                "tilt_diagnostic",
                TEXTS["tilt_diagnostic"].format(names=_names(diagnostic), scale=scale),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        settings = [
            f"Interval level: {_level(tilt.level)}.",
            "Off gamma = 0 each interval is centred on the tilted estimate and reuses the "
            "standard error of the missing-at-random analysis, with no new correction step.",
        ]
        if tilt.arm_gamma is not None:
            settings.append(f"Each arm's tilt is its multiplier times gamma: {direction}.")
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "Among units with a missing outcome, the mean of the outcome rescaled to "
                    "[0, 1] is expit(logit(Q(a, W)) + gamma), where Q is the fitted mean among "
                    "recorded units; the full mean mixes the two by the fitted chance of being "
                    "recorded (Scharfstein, Rotnitzky and Robins, 1999).",
                ),
            ),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
        ]
    return Summary.build(
        title="Sensitivity to outcomes missing not at random",
        header=header,
        header_split=1,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def tilt_repr(tilt: MissingnessTilt) -> str:
    """The one-line repr of a missingness tilt."""
    gammas = [float(value) for value in tilt.column("gamma")]
    span = f"gamma {min(gammas):g} to {max(gammas):g}" if gammas else "no gamma"
    return short_repr("MissingnessTilt", [("", _names(tilt.estimands)), ("", span)])


# ----------------------------------------------------------------- SimulatedConfoundingResult


def _values_text(value: Any, spec: str) -> str:
    if value is None:
        return V.MISSING
    if isinstance(value, tuple):
        return ", ".join(_values_text(item, spec) for item in value)
    return format(float(value), spec)


def _surface_laws(result: SimulatedConfoundingResult) -> tuple[str, str]:
    """What each cell moves, and the plain treatment and outcome laws."""
    if result.intermediate_name is not None:
        moved = "the intermediate"
        treatment = TEXTS["law_intermediate"].format(name=result.intermediate_name)
    elif result.treatment_family == "categorical" and result.arm_pair is not None:
        moved = "the treatment"
        pair = result.arm_pair
        first, second = (pair[0], pair[1]) if len(pair) == 2 else (pair, "")
        treatment = TEXTS["law_categorical"].format(a=first, b=second)
    elif result.treatment_family == "vector":
        moved = "each treatment component"
        treatment = TEXTS["law_vector"]
    elif result.treatment_family == "longitudinal":
        moved = "each treatment node"
        treatment = TEXTS["law_longitudinal"]
    elif result.treatment_family == "binary":
        moved = "the treatment"
        treatment = TEXTS["law_binary"]
    else:
        moved = "the treatment"
        treatment = TEXTS["law_continuous"]
    if result.bounded_outcome_law == "logit":
        outcome = TEXTS["law_outcome_logit"]
    elif result.bounded_outcome_law == "normal_score":
        outcome = TEXTS["law_outcome_normal_score"]
    elif result.outcome_family == "gaussian":
        outcome = TEXTS["law_outcome_gaussian"]
    else:
        outcome = TEXTS["law_outcome_binomial"]
    return moved, f"{treatment} {outcome}"


def _surface_caveats(result: SimulatedConfoundingResult) -> list[Note]:
    """The conditions that change how a movement reads, each when it holds."""
    notes = []
    if result.estimand.startswith(("ey_ipsi[", "ate_ipsi[")):
        notes.append(note("incremental_reading", TEXTS["incremental_reading"]))
    if result.estimand.startswith("ey_learned_rule["):
        notes.append(note("learned_rule_reading", TEXTS["learned_rule_reading"]))
    if result.observation_reads is not None:
        reads = _names(result.observation_reads) or "nothing"
        notes.append(note("observation_reads", TEXTS["observation_reads"].format(reads=reads)))
    if result.bounded_share is not None and result.bounded_share > 0.0:
        notes.append(
            note("bounded_rows", TEXTS["bounded_rows"].format(share=f"{result.bounded_share:.3f}"))
        )
    if result.refit_folds == "redrawn":
        notes.append(note("redrawn_folds", TEXTS["redrawn_folds"]))
    return notes


def _association_guard(result: SimulatedConfoundingResult) -> str:
    """The reading lines of the association columns."""
    lines = [line for line in result._reading_guard() if line]
    intermediate = " or intermediate" if result.intermediate_name is not None else ""
    stratum = "" if result.stratum is None else ", within the selected baseline stratum"
    defined = TEXTS["association_defined"].format(intermediate=intermediate, stratum=stratum)
    return " ".join([defined, *lines])


def surface_summary(result: SimulatedConfoundingResult, detail: Detail) -> Summary:
    """The summary of a simulated-confounding surface.

    Parameters
    ----------
    result : SimulatedConfoundingResult
        The surface.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    failed = len(result.failures)
    total = len(result.cells)
    decimals = _decimals(
        [result.original_estimate, *(cell.estimate for cell in result.successful_cells)]
    )
    header = [
        (V.QUESTION, f"how {result.estimand} moves under a simulated confounder"),
        _estimate_fact(result.estimand, result.original_estimate),
        (
            V.MOVEMENT,
            "difference in log ratio"
            if result.movement_scale == "log_ratio"
            else "difference in estimate",
        ),
        (V.CELLS, f"{total}, {failed} failed"),
    ]
    arm_phrase = None
    if result.population == "perturbed_treatment_group":
        arm_phrase = f"treatment {result.conditioning_arm}"
        header.append((V.POPULATION, f"units with {arm_phrase}"))
    if result.n_repeats > 1:
        header.append((V.REPEATED_SPLITS, f"{result.n_repeats}, median"))

    components = result.treatment_components
    labels = [f"({name})" for name in components] if components else [""]
    movement_columns = ["cell", *(f"treatment strength {label}".strip() for label in labels)]
    if result.intermediate_name is not None:
        movement_columns.append("intermediate strength")
    movement_columns += ["outcome strength", V.ESTIMATE, "movement"]
    association_columns = ["cell", *(f"treatment {label}".strip() for label in labels)]
    if result.intermediate_name is not None:
        association_columns.append("intermediate")
    if result.bounded_outcome_law is not None:
        association_columns.append("outcome")
    if result.population == "perturbed_treatment_group":
        association_columns.append("population share")
    movement_rows = []
    association_rows = []
    for number, cell in enumerate(result.cells, start=1):
        label = Cell(number, str(number))
        strengths = cell.treatment_strength
        strength_values = strengths if isinstance(strengths, tuple) else (strengths,)
        row: list[Cell] = [label, *(Cell(item, f"{float(item):g}") for item in strength_values)]
        if result.intermediate_name is not None:
            row.append(
                Cell(cell.intermediate_strength, _values_text(cell.intermediate_strength, "g"))
            )
        row.append(Cell(cell.outcome_strength, f"{float(cell.outcome_strength):g}"))
        if cell.failure is None:
            row.append(_number(cell.estimate, decimals))
            shift = cell.displacement
            text = "-" if shift is None else f"{float(shift):+.{decimals}f}"
            row.append(Cell(shift, text))
        else:
            row.append(Cell(None, "failed"))
            row.append(Cell(None, cell.failure.error_type))
        movement_rows.append(tuple(row))
        association = cell.induced_treatment_association
        if isinstance(association, tuple):
            values: tuple[Any, ...] = association
        else:
            values = (association,) * len(labels)
        associations: list[Cell] = [
            label,
            *(Cell(item, _values_text(item, "+.3f")) for item in values),
        ]
        if result.intermediate_name is not None:
            item = cell.induced_intermediate_association
            associations.append(Cell(item, _values_text(item, "+.3f")))
        if result.bounded_outcome_law is not None:
            item = cell.induced_outcome_association
            associations.append(Cell(item, _values_text(item, "+.3f")))
        if result.population == "perturbed_treatment_group":
            item = cell.target_population_fraction
            associations.append(Cell(item, _values_text(item, ".3f")))
        association_rows.append(tuple(associations))
    tables = [
        SummaryTable(tuple(movement_columns), tuple(movement_rows), title="Movement"),
        SummaryTable(tuple(association_columns), tuple(association_rows), title="Associations"),
    ]
    if result.calibrations:
        calibration_columns = ["covariate", "role"]
        if components:
            calibration_columns.append("component")
        calibration_columns.append("strength")
        calibration_rows = []
        for calibration in result.calibrations:
            cells = [_label(calibration.covariate), _label(calibration.role)]
            if components:
                cells.append(Cell(calibration.component, calibration.component or V.MISSING))
            cells.append(_number(calibration.strength, 3))
            calibration_rows.append(tuple(cells))
        tables.append(
            SummaryTable(
                tuple(calibration_columns),
                tuple(calibration_rows),
                title="Observed covariates on the same scale",
            )
        )
    moved, laws = _surface_laws(result)
    notes = [
        note("sensitivity_assumption", TEXTS["assumption_surface"].format(moved=moved, laws=laws)),
        note("association_reading", _association_guard(result)),
    ]
    if arm_phrase is not None:
        notes.append(
            note("population_moves", TEXTS["population_moves"].format(arm_phrase=arm_phrase))
        )
    notes += _surface_caveats(result)
    if failed:
        notes.append(note("cells_failed", TEXTS["cells_failed"].format(failed=failed, total=total)))
    calibration_text = " ".join(
        line[:1].upper() + line[1:] + ("" if line.endswith(".") else ".")
        for line in result._calibration_lines()
    )
    if calibration_text:
        notes.append(note("calibration_nan", calibration_text))
    sections: list[SummarySection] = []
    if detail == "full":
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                tuple(part for part in (result.treatment_law, result.outcome_law) if part),
            ),
            SummarySection(V.SECTION_SETTINGS, tuple(_population_facts(result))),
        ]
    return Summary.build(
        title="Simulated confounding",
        header=header,
        header_split=2,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def _population_facts(result: SimulatedConfoundingResult) -> list[str]:
    """The population and refit facts of a surface, in plain words, without seeds."""
    facts = [
        "Every cell refits on all fitted rows; the associations are measured on "
        + ("the selected baseline stratum." if result.stratum is not None else "all fitted rows."),
    ]
    if result.stratum is not None:
        pairs = zip(result.strata_names, result.stratum, strict=False)
        facts.append(
            f"Baseline stratum: {', '.join(f'{name} = {value}' for name, value in pairs)}."
        )
    if result.weight_report.name is not None:
        facts.append(f"Each cell uses the fixed observation weights {result.weight_report.name}.")
    if result.n_repeats > 1:
        facts.append(
            f"Each estimate is the median over {result.n_repeats} random splits into folds."
        )
    for line in result._population_conditions():
        facts.append(line[:1].upper() + line[1:] + ("" if line.endswith(".") else "."))
    return facts


def surface_repr(result: SimulatedConfoundingResult) -> str:
    """The one-line repr of a simulated-confounding surface."""
    decimals = _decimals([result.original_estimate])
    return short_repr(
        "SimulatedConfoundingResult",
        [
            (result.estimand, number_text(result.original_estimate, decimals)),
            ("", f"{len(result.cells)} cells"),
            ("", f"{len(result.failures)} failed"),
        ],
    )
