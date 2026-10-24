"""The printed summaries of the sensitivity analyses (X89 T4).

Each summary states the question its analysis answers, prints the numbers in tables, and
says in plain words what the analysis assumes. Symbols (``sigma^2``, ``nu^2``, the Riesz
representer) and the formal statements wait for ``detail="full"``. Every word another
summary shares comes from :mod:`cleverly._summary.vocabulary`.

The builders read only fields the result already holds, so a summary computes nothing new
and a notebook display raises no warning.
"""

from __future__ import annotations

import math
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
from .._summary._estimates import plain_support_reason, without_roadmap
from ..inference.influence import spread_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._inference_status import InferenceStatus
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

#: The most rows and columns a contour table prints.
_CONTOUR_ROWS = 6
_CONTOUR_COLUMNS = 5


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
    if value is None or not _finite(value):
        return Cell(None if value is None else float(value), number_text(value, decimals))
    return Cell(float(value), number_text(value, decimals))


def _interval(low: Any, high: Any, decimals: int) -> Cell:
    return Cell(
        (float(low), float(high)),
        f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
    )


def _strength(value: Any) -> str:
    """A strength, share or alignment at 3 decimals."""
    return number_text(value, 3)


def _level(level: float) -> str:
    return f"{level * 100:g}%"


def _null(value: float) -> str:
    return f"{value:g}"


def _alignment(rho: float) -> str:
    text = f"{_strength(rho)} (rho)"
    return text + " (worst case)" if abs(rho) == 1.0 else text


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


def _status_note(status: str, names: str, label: str) -> Note:
    record = status_record(status)
    return note(
        "no_inference",
        TEXTS["no_inference"].format(names=names, reason=record.plain_reason, label=label),
    )


def _limits_withheld(reason_is_plugin: bool) -> Note:
    key = "limits_withheld_plugin" if reason_is_plugin else "limits_withheld_zero"
    return note("limits_withheld", TEXTS[key])


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


def _nu2_setting(nu2_estimator: str, correction: float | None) -> str:
    if nu2_estimator == "doubly_robust":
        return "Estimate of nu^2: doubly robust."
    if correction is not None:
        return f"Estimate of nu^2: plug-in, second-order corrected by {correction:.4g}."
    return "Estimate of nu^2: plug-in, uncorrected."


# ----------------------------------------------------------------- SensitivityBounds


def _rv_sentence(rv: float | None, null: float) -> str:
    if rv is None:
        return TEXTS["rv_unreached"].format(null=_null(null))
    return TEXTS["rv_reached"].format(rv=_strength(rv), null=_null(null))


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
    status = cast("InferenceStatus", bounds.inference)
    null = bounds.null_hypothesis
    reason = bounds._limit_reason()
    derived = reason is None
    inferential = supplies_inference(status)
    excludes = (bounds.lower - null) * (bounds.upper - null) > 0
    header = [
        (
            V.QUESTION,
            f"how far left-out confounders of the stated strength could move {bounds.estimand}",
        ),
        (
            V.ESTIMATE_ANALYSED,
            f"{bounds.estimand} = {number_text(bounds.psi, _decimals([bounds.psi]))}",
        ),
        (V.OUTCOME_STRENGTH, f"{_strength(bounds.cf_y)} (cf_y)"),
        (V.TREATMENT_STRENGTH, f"{_strength(bounds.cf_d)} (cf_d)"),
        (V.ALIGNMENT, _alignment(bounds.rho)),
        (V.LARGEST_SHIFT, number_text(bounds.bias, _decimals([bounds.psi, bounds.bias]))),
        (
            V.CONCLUSION,
            f"the range {'excludes' if excludes else 'includes'} {_null(null)}",
        ),
    ]
    values = [bounds.psi, bounds.lower, bounds.upper]
    if derived:
        values += [bounds._ci_lower, bounds._ci_upper]
    decimals = _decimals(values)
    columns = ["", V.ESTIMATE, "range"]
    row = [
        Cell(bounds.estimand, bounds.estimand),
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
    rv_rows = [(Cell("estimate", "estimate"), _rv_cell(bounds.robustness_value))]
    if derived:
        name = f"{level} limit" if inferential else "plug-in limit (diagnostic)"
        rv_rows.append((Cell(name, name), _rv_cell(bounds._robustness_value_ci)))
    tables.append(SummaryTable(("", "robustness value"), tuple(rv_rows), title="Robustness value"))

    assumption = TEXTS["assumption_bounds"].format(
        rv_sentence=_rv_sentence(bounds.robustness_value, null)
    )
    if derived:
        assumption += " " + TEXTS["assumption_bounds_limits"].format(level=level)
    notes = [note("sensitivity_assumption", assumption)]
    if derived and not inferential:
        notes.append(_status_note(status, bounds.estimand, V.SPREAD_LABELS["plugin_interval"]))
    if not derived:
        notes.append(_limits_withheld(bounds.limit_refusal is not None))
    elif bounds.limit_descriptive:
        notes.append(_descriptive("one-sided limits"))
    joint = _joint_note(bounds.features)
    if joint is not None:
        notes.append(joint)
    if bounds.assumption is not None:
        notes.append(note("consistent_mechanism", without_roadmap(bounds.assumption)))

    sections: list[SummarySection] = []
    if detail == "full":
        assumptions = [_BOUND_ASSUMPTION]
        if derived:
            assumptions.append(_LIMIT_ASSUMPTION)
        settings = [
            f"Bias scale sqrt(sigma^2 nu^2): {bounds.max_bias:.4g}.",
            _nu2_setting(bounds.nu2_estimator, bounds.nu2_correction),
            f"Null value: {_null(null)}.",
            f"Confidence level of the limits: {level}.",
        ]
        if reason is not None:
            settings.append(f"Why the limits are withheld: {without_roadmap(reason)}")
        if bounds.limit_reading is not None:
            settings.append(
                f"Evidence for the limits: {without_roadmap(bounds.limit_reading.sentence)}"
            )
        sections = [
            SummarySection(V.SECTION_ASSUMPTIONS, tuple(assumptions)),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Bounds under omitted confounding",
        header=header,
        tables=tables,
        notes=notes,
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
        (
            V.QUESTION,
            f"how strong left-out confounders would need to be to move "
            f"{value.estimand} to {_null(null)}",
        ),
        (
            V.ESTIMATE_ANALYSED,
            f"{value.estimand} = {number_text(value.psi, _decimals([value.psi]))}",
        ),
        (V.ALIGNMENT, _alignment(value.rho)),
    ]
    rows = [(Cell("estimate", "estimate"), _rv_cell(value["rv"]))]
    if derived:
        name = f"{level} limit" if inferential else "plug-in limit (diagnostic)"
        rows.append((Cell(name, name), _rv_cell(value[limit_key])))
    tables = [SummaryTable(("", "robustness value"), tuple(rows))]
    sentence = _rv_sentence(value["rv"], null)
    if derived:
        sentence += " " + TEXTS["rv_limit_reached"].format(
            limit=rows[1][0].text, level=level, null=_null(null)
        )
    notes = [
        note(
            "sensitivity_assumption",
            TEXTS["assumption_robustness"].format(null=_null(null), rv_sentence=sentence),
        )
    ]
    if derived and not inferential:
        notes.append(_status_note(status, value.estimand, "plug-in limit (diagnostic)"))
    if not derived:
        notes.append(_limits_withheld(str(value["limit_refusal"]).startswith("nu2_estimator")))
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
                f"Why the limit value is withheld: {without_roadmap(str(value['limit_refusal']))}"
            )
        if "limit_sentence" in value:
            settings.append(f"Evidence for the limit: {without_roadmap(value['limit_sentence'])}")
        sections = [
            SummarySection(V.SECTION_ASSUMPTIONS, (_BOUND_ASSUMPTION,)),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Robustness value", header=header, tables=tables, notes=notes, sections=sections
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
    header = [
        (
            V.QUESTION,
            f"the scales that turn a confounder strength into a bias bound for {elements.estimand}",
        ),
    ]
    values = [elements.sigma2, elements.nu2, elements.max_bias]
    decimals = _decimals(values)
    rows = [
        (
            Cell(
                "outcome variance left after adjustment", "outcome variance left after adjustment"
            ),
            _number(elements.sigma2, decimals),
        ),
        (Cell("average squared weight", "average squared weight"), _number(elements.nu2, decimals)),
        (Cell("bias scale", "bias scale"), _number(elements.max_bias, decimals)),
    ]
    if elements.nu2_correction is not None:
        rows.append(
            (
                Cell("plug-in correction", "plug-in correction"),
                _number(elements.nu2_correction, decimals),
            )
        )
    tables = [SummaryTable(("", "value"), tuple(rows))]
    notes = [note("sensitivity_assumption", TEXTS["assumption_elements"])]
    if elements.psi_max_bias is None:
        notes.append(_limits_withheld(elements.limit_refusal is not None))
    joint = _joint_note(elements.features)
    if joint is not None:
        notes.append(joint)
    sections: list[SummarySection] = []
    if detail == "full":
        settings = [_nu2_setting(elements.nu2_estimator, elements.nu2_correction)]
        if elements.limit_refusal is not None:
            settings.append(
                f"Why the limits are withheld: {without_roadmap(elements.limit_refusal.text)}"
            )
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "The rows are sigma^2, nu^2 and sqrt(sigma^2 nu^2), each with an influence "
                    "curve, so a bound built from them has one-sided limits.",
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


def _names(names: Sequence[str]) -> str:
    return ", ".join(str(name) for name in names)


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
        (
            V.QUESTION,
            f"how strong left-out confounders would be if they mattered as much as {covariates}",
        ),
        (V.ESTIMATE_ANALYSED, f"{record.estimand} = {number_text(record.psi_long, decimals)}"),
        (V.BENCHMARK_COVARIATES, covariates),
        (V.IMPLIED_STRENGTHS, f"{_strength(record.cf_y)} (cf_y), {_strength(record.cf_d)} (cf_d)"),
        (V.IMPLIED_ALIGNMENT, f"{_strength(record.rho)} (rho)"),
    ]
    scales = _decimals([record.sigma2_long, record.sigma2_short, record.nu2_long, record.nu2_short])
    rows = (
        (
            Cell("estimate", "estimate"),
            _number(record.psi_long, decimals),
            _number(record.psi_short, decimals),
        ),
        (
            Cell(
                "outcome variance left after adjustment", "outcome variance left after adjustment"
            ),
            _number(record.sigma2_long, scales),
            _number(record.sigma2_short, scales),
        ),
        (
            Cell("average squared weight", "average squared weight"),
            _number(record.nu2_long, scales),
            _number(record.nu2_short, scales),
        ),
    )
    tables = [
        SummaryTable(
            ("", "with", "without"), rows, title="With and without the benchmark covariates"
        )
    ]
    notes = [
        note(
            "sensitivity_assumption",
            TEXTS["assumption_benchmark"].format(
                covariates=covariates,
                delta=f"{record.delta_psi:+.{decimals}f}",
            ),
        )
    ]
    clipped = []
    if record.gain_y > 1.0:
        clipped.append(
            TEXTS["strength_clipped_y"].format(covariates=covariates, gain=f"{record.gain_y:.3g}")
        )
    if record.gain_d >= 1.0:
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
                    "weight gain (nu^2_long / nu^2_short - 1), each clipped to [0, 1]; the "
                    "gains serve as the strengths themselves (Chernozhukov et al., Long story "
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
    size = max(len(d_values), len(y_values))
    header = [
        (
            V.QUESTION,
            f"the {contour.bound} end of the range for {contour.estimand} over a grid "
            "of confounder strengths",
        ),
        (
            V.ESTIMATE_ANALYSED,
            f"{contour.estimand} = {number_text(contour.psi, _decimals([contour.psi]))}",
        ),
        (V.ALIGNMENT, _alignment(contour.rho)),
        (
            V.GRID,
            f"{len(y_values)} x {len(d_values)}, cf_y to {_strength(max(y_values))}, "
            f"cf_d to {_strength(max(d_values))}",
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
                    rows=len(rows_shown), columns=len(columns_shown), size=size
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections = [
            SummarySection(V.SECTION_ASSUMPTIONS, (_BOUND_ASSUMPTION,)),
            _definitions_bounds(),
        ]
    return Summary.build(
        title="Bias-adjusted bounds over a grid of strengths",
        header=header,
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
    scale = value.scale + (", converted to a risk ratio" if value.approximate else "")
    header = [
        (
            V.QUESTION,
            "how strongly an unmeasured confounder would need to be associated with "
            f"both the treatment and the outcome to explain away {value.estimand}",
        ),
        (V.ESTIMATE_ANALYSED, value.estimand),
        (V.SCALE, scale),
    ]
    low, high = value.risk_ratio_ci
    nearest = low if value.risk_ratio >= 1.0 else high
    limit_name = f"{_level(value.level)} limit nearest the null"
    rows = (
        (Cell("estimate", "estimate"), _number(value.risk_ratio, 3), _number(value.point, 2)),
        (Cell(limit_name, limit_name), _number(nearest, 3), _number(value.limit, 2)),
    )
    tables = [SummaryTable(("", "risk ratio", "E-value"), rows)]
    limit_sentence = (
        TEXTS["evalue_limit_reached"].format(limit=number_text(value.limit, 2))
        if value.limit > 1.0
        else TEXTS["evalue_limit_covers"]
    )
    key = "assumption_evalue" if value.bias_factor == "single" else "assumption_evalue_per_arm"
    notes = [
        note(
            "sensitivity_assumption",
            TEXTS[key].format(point=number_text(value.point, 2), limit_sentence=limit_sentence),
        )
    ]
    from .evalue import _PER_ARM_CONDITIONS

    conversion = value.note.replace(_PER_ARM_CONDITIONS, "").strip()
    if value.approximate and conversion:
        notes.append(note("approximate_scale", without_roadmap(conversion)))
    if value.truncated_bound is not None:
        notes.append(
            note(
                "bound_truncated",
                TEXTS["bound_truncated"].format(value=f"{value.truncated_bound:.3g}"),
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
        if value.source_estimand is not None and value.source_estimand != value.estimand:
            settings.append(f"Converted from the reported {value.source_estimand}.")
        if value.limit_reading is not None:
            settings.append(
                f"Evidence for the limit: {without_roadmap(value.limit_reading.sentence)}"
            )
        if value.support.label == "experimental":
            settings.append(f"Why experimental: {plain_support_reason(value.support.reason)}")
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "E-value = RR + sqrt(RR (RR - 1)) for a risk ratio RR >= 1 (its inverse "
                    "below 1): the smallest equal strength, on the risk-ratio scale, of the "
                    "confounder's association with the treatment and with the outcome that "
                    "could move RR to 1 (VanderWeele and Ding, 2017).",
                    *((_PER_ARM_CONDITIONS,) if value.bias_factor == "per_arm_product" else ()),
                ),
            ),
            SummarySection(V.SECTION_SETTINGS, tuple(settings)),
        ]
    return Summary.build(
        title="E-value", header=header, tables=tables, notes=notes, sections=sections
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
    largest = max((row.gamma for row in bounds.rows), default=1.0)
    header = [
        (
            V.QUESTION,
            f"the range of {bounds.estimand} if an unmeasured confounder could change "
            "each unit's odds of treatment by up to a factor Gamma",
        ),
        (V.OUTCOME, bounds.outcome_kind),
    ]
    if bounds.null is not None:
        for label, gamma in (
            (V.BREAKDOWN, bounds.breakdown),
            (V.BREAKDOWN_INTERVAL, bounds.limit_breakdown),
        ):
            header.append((label, f"none up to {largest:g}" if gamma is None else f"{gamma:g}"))
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
    tables = [SummaryTable(("Gamma", "lower", "upper", f"{level} interval for the range"), rows)]
    notes = [
        note(
            "sensitivity_assumption",
            TEXTS["assumption_marginal"].format(estimand=bounds.estimand, level=level),
        )
    ]
    if bounds.limit_descriptive:
        notes.append(_descriptive("interval for the range"))
    sections: list[SummarySection] = []
    if detail == "full":
        settings = []
        if bounds.quantile_learner:
            settings.append(f"Quantile learners: {_names(bounds.quantile_learner)}.")
        if bounds.pseudo_outcome_learner:
            settings.append(f"Tail-regression learners: {_names(bounds.pseudo_outcome_learner)}.")
        if bounds.scale == "log":
            settings.append("The standard errors of the ends are on the log scale.")
        if bounds.limit_reading is not None:
            settings.append(
                f"Evidence for the interval: {without_roadmap(bounds.limit_reading.sentence)}"
            )
        sections = [
            SummarySection(
                V.SECTION_ASSUMPTIONS,
                (
                    "Marginal sensitivity model: the odds ratio between P(A = a | W, U) and "
                    "P(A = a | W) lies in [1 / Gamma, Gamma] for every unit (Tan, 2006).",
                ),
            ),
            SummarySection(
                V.SECTION_SETTINGS, tuple(settings) or ("No settings beyond the grid.",)
            ),
        ]
    return Summary.build(
        title="Bounds under the marginal sensitivity model",
        header=header,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def marginal_repr(bounds: MarginalSensitivityBounds) -> str:
    """The one-line repr of marginal sensitivity model bounds."""
    gammas = [row.gamma for row in bounds.rows]
    facts: list[tuple[str, Any]] = [("", bounds.estimand)]
    if gammas:
        facts.append(("", f"Gamma {min(gammas):g} to {max(gammas):g}"))
    if bounds.null is not None:
        facts.append(("breakdown", "none" if bounds.breakdown is None else f"{bounds.breakdown:g}"))
    return short_repr("MarginalSensitivityBounds", facts)


# ----------------------------------------------------------------- MissingnessTilt


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
        (
            V.QUESTION,
            f"how {_names(names)} would move if units with a missing outcome differed "
            "from recorded units with the same treatment and covariates",
        ),
        (V.MISSING_OUTCOMES, f"{count_text(tilt.n_missing)} of {count_text(tilt.n)}"),
        (V.DIRECTION, direction),
    ]
    payload = dict(tilt.rows)
    count = len(tilt)
    statuses = dict(tilt.statuses)
    tables = []
    diagnostic: list[str] = []
    withheld: list[str] = []
    level = _level(tilt.level)
    for name in names:
        indices = [index for index in range(count) if payload["estimand"][index] == name]
        status = statuses.get(name, "influence_curve")
        inferential = supplies_inference(status)
        if not inferential:
            withheld.append(name)
        spread_key = spread_name("std_err", cast("InferenceStatus", status))
        low_key = spread_name("ci_lower", cast("InferenceStatus", status))
        high_key = spread_name("ci_upper", cast("InferenceStatus", status))
        spreads = [_value(payload, spread_key, index) for index in indices]
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
        decimals = max(
            2,
            min(
                6,
                1
                - math.floor(
                    math.log10(
                        min(
                            [
                                abs(float(s))
                                for _, s, _, _ in rows_values
                                if _finite(s) and float(s) > 0
                            ]
                            or [1e-2]
                        )
                    )
                ),
            ),
        )
        spread_label = (
            V.SPREAD_LABELS[spread_key] if inferential else V.SPREAD_LABELS["plugin_std_err"]
        )
        interval_label = (
            V.interval_label(tilt.level) if inferential else V.SPREAD_LABELS["plugin_interval"]
        )
        rows = []
        for index, spread, low, high in rows_values:
            gamma = float(payload["gamma"][index])
            label = f"{gamma:g}"
            if payload.get("is_mar", [False] * count)[index]:
                label += " (missing at random)"
            mark = moved_diagnostic and inferential and not bool(payload["is_mar"][index])
            rows.append(
                (
                    Cell(gamma, label + (V.NO_INFERENCE_MARK if mark else "")),
                    _number(payload["psi"][index], decimals),
                    _number(spread, decimals),
                    _interval(low, high, decimals),
                )
            )
        del spreads
        tables.append(
            SummaryTable(
                ("gamma", V.ESTIMATE, spread_label, interval_label), tuple(rows), title=name
            )
        )
    notes = [note("sensitivity_assumption", TEXTS["assumption_tilt"])]
    if withheld:
        first = statuses[withheld[0]]
        notes.append(_status_note(first, _names(withheld), V.SPREAD_LABELS["plugin_std_err"]))
    if diagnostic:
        notes.append(
            note("tilt_diagnostic", TEXTS["tilt_diagnostic"].format(names=_names(diagnostic)))
        )
    sections: list[SummarySection] = []
    if detail == "full":
        settings = [f"Interval level: {level}."]
        if tilt.arm_gamma is not None:
            settings.append(
                "Each arm's tilt is its multiplier times gamma: "
                + ", ".join(f"arm {arm} x {value:g}" for arm, value in tilt.arm_gamma)
                + "."
            )
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
        tables=tables,
        notes=notes,
        sections=sections,
    )


def _value(payload: dict[str, tuple[Any, ...]], key: str, index: int) -> Any:
    column = payload.get(key)
    return None if column is None else column[index]


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
        (
            V.QUESTION,
            f"how {result.estimand} moves when the data are refit with a simulated "
            "common cause of the treatment and the outcome",
        ),
        (
            V.ESTIMATE_ANALYSED,
            f"{result.estimand} = {number_text(result.original_estimate, decimals)}",
        ),
        (
            V.MOVEMENT,
            "difference in the log ratio"
            if result.movement_scale == "log_ratio"
            else "difference in the estimate",
        ),
        (V.CELLS, f"{total}, {failed} failed"),
    ]
    arm_phrase = None
    if result.population == "perturbed_treatment_group":
        arm_phrase = f"treatment {result.conditioning_arm}"
        header.append((V.POPULATION, f"units with {arm_phrase} in each cell"))
    if result.n_repeats > 1:
        header.append((V.REPEATED_SPLITS, f"{result.n_repeats}, median of the splits"))

    components = result.treatment_components
    columns: list[str] = []
    if components:
        columns += [f"treatment strength ({name})" for name in components]
    else:
        columns.append("treatment strength")
    if result.intermediate_name is not None:
        columns.append("intermediate strength")
    columns += ["outcome strength", V.ESTIMATE, "movement"]
    if components:
        columns += [f"treatment association ({name})" for name in components]
    else:
        columns.append("treatment association")
    if result.intermediate_name is not None:
        columns.append("intermediate association")
    if result.bounded_outcome_law is not None:
        columns.append("outcome association")
    if result.population == "perturbed_treatment_group":
        columns.append("population share")
    rows = []
    for cell in result.cells:
        strengths = cell.treatment_strength
        strength_values = strengths if isinstance(strengths, tuple) else (strengths,)
        row = [Cell(item, f"{float(item):g}") for item in strength_values]
        if result.intermediate_name is not None:
            row.append(
                Cell(cell.intermediate_strength, _values_text(cell.intermediate_strength, "g"))
            )
        row.append(Cell(cell.outcome_strength, f"{float(cell.outcome_strength):g}"))
        if cell.failure is None:
            row.append(_number(cell.estimate, decimals))
            row.append(Cell(cell.displacement, f"{float(cell.displacement):+.{decimals}f}"))
        else:
            row.append(Cell(None, "failed"))
            row.append(Cell(None, cell.failure.error_type))
        association = cell.induced_treatment_association
        association_values = (
            association
            if isinstance(association, tuple)
            else (association,) * max(1, len(components))
        )
        if not components:
            association_values = (association,)
        row += [Cell(item, _values_text(item, "+.3f")) for item in association_values]
        if result.intermediate_name is not None:
            item = cell.induced_intermediate_association
            row.append(Cell(item, _values_text(item, "+.3f")))
        if result.bounded_outcome_law is not None:
            item = cell.induced_outcome_association
            row.append(Cell(item, _values_text(item, "+.3f")))
        if result.population == "perturbed_treatment_group":
            item = cell.target_population_fraction
            row.append(Cell(item, _values_text(item, ".3f")))
        rows.append(tuple(row))
    tables = [SummaryTable(tuple(columns), tuple(rows))]
    if result.calibrations:
        calibration_columns = ["covariate", "role"]
        if components:
            calibration_columns.append("component")
        calibration_columns.append("strength")
        calibration_rows = []
        for item in result.calibrations:
            values = [Cell(item.covariate, item.covariate), Cell(item.role, item.role)]
            if components:
                values.append(Cell(item.component, item.component or V.MISSING))
            values.append(_number(item.strength, 3))
            calibration_rows.append(tuple(values))
        tables.append(
            SummaryTable(
                tuple(calibration_columns),
                tuple(calibration_rows),
                title="Observed covariates on the same scale",
            )
        )
    laws = " ".join(part for part in (result.treatment_law, result.outcome_law) if part)
    notes = [note("sensitivity_assumption", TEXTS["assumption_surface"].format(laws=laws))]
    guard = " ".join(line for line in result._reading_guard() if line)
    if guard:
        notes.append(note("association_reading", guard))
    if arm_phrase is not None:
        notes.append(
            note("population_moves", TEXTS["population_moves"].format(arm_phrase=arm_phrase))
        )
    if failed:
        notes.append(note("cells_failed", TEXTS["cells_failed"].format(failed=failed, total=total)))
    calibration = " ".join(result._calibration_lines())
    if calibration:
        notes.append(note("calibration_nan", calibration))
    sections: list[SummarySection] = []
    if detail == "full":
        sections = [SummarySection(V.SECTION_SETTINGS, tuple(_population_facts(result)))]
    return Summary.build(
        title="Simulated confounding",
        header=header,
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
    for line in result.population_lines()[7:]:
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
