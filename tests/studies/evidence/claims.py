"""Every number a study's prose quotes, resolvable from the committed artefacts.

A measured value typed into a document is a copy with no gate on it.  It was wrong here on
the first attempt -- a bound of 0.0145269 rendered as 0.0146 -- and a wrong figure in a
document whose whole claim is "verify every value in the summary" is worse than a missing
one.  So the documents name quantities instead of repeating numbers, and this module is the
vocabulary: a name resolves against the artefacts, and the documentation gate checks that
what is printed is what resolves, at the precision it was printed to.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import pandas as pd

from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import StudyRecord

#: Artefact file names, and the columns each is keyed by.
ARTIFACTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "replicates": ("replicates.csv.gz", ()),
    "summary": ("summary.csv", ("implementation", "scenario", "estimand")),
    "performance": ("performance-tests.csv", ("implementation", "scenario", "estimand")),
    "equivalence": ("equivalence.csv", ("scenario", "estimand")),
    "property_replicates": ("property-replicates.csv.gz", ()),
    "properties": ("properties.csv", ("property", "cell")),
}

_REFERENCE = re.compile(r"^(?P<artifact>\w+)\[(?P<keys>[^\]]*)\]:(?P<column>\w+)$")


def load(record: StudyRecord) -> dict[str, pd.DataFrame]:
    """Every committed artefact of a study, keyed by its short name."""
    return {
        name: pd.read_csv(record.artifact(filename)) for name, (filename, _) in ARTIFACTS.items()
    }


def _subject(frame: pd.DataFrame, implementation: str) -> pd.DataFrame:
    return frame.loc[frame["implementation"] == implementation]


def _displacement(frame: pd.DataFrame, implementation: str) -> pd.Series:
    r"""How far targeting moved each estimate, in units of its own standard error.

    :math:`|\hat\psi - \hat\psi^{0}| / \widehat{SE}`, the scale-free form, because the
    absolute one is a number about the estimand's units and says nothing across studies.

    What ``initial_estimate`` means is the study's to state and is **not** uniform: for a
    sequential estimator it is the earliest node's regression of an *already targeted* later
    node, so this measures the final fluctuation rather than the whole targeting step.  A
    study whose plug-in is genuinely untargeted measures the whole of it.  Published so a
    reader can see which, rather than inferred from a shared pass/fail that cannot tell them
    apart.
    """
    rows = _subject(frame, implementation)
    return (rows["estimate"] - rows["initial_estimate"]).abs() / rows["std_error"]


def _aggregates(record: StudyRecord) -> dict[str, Callable[[Mapping[str, pd.DataFrame]], float]]:
    """Study-independent headline quantities, derived rather than declared."""
    subject = record.implementation

    def count(frame: pd.DataFrame) -> float:
        return float(len(frame))

    return {
        "max_targeting_displacement": lambda data: float(
            _displacement(data["replicates"], subject).max()
        ),
        "median_targeting_displacement": lambda data: float(
            _displacement(data["replicates"], subject).median()
        ),
        "replicates": lambda data: float(record.replicates),
        "n": lambda data: float(record.n),
        "independent_tests_total": lambda data: count(data["performance"]),
        "independent_tests_passed": lambda data: float(data["performance"]["passed"].sum()),
        "subject_tests_total": lambda data: count(_subject(data["performance"], subject)),
        "subject_tests_passed": lambda data: float(
            _subject(data["performance"], subject)["passed"].sum()
        ),
        "paired_tests_total": lambda data: count(data["equivalence"]),
        "paired_tests_passed": lambda data: float(data["equivalence"]["passed"].sum()),
        "property_cells_total": lambda data: count(data["properties"]),
        # Both columns, not just the row's own.  A property whose claim needs more than one
        # cell -- ``crossfit_overfitting``, whose coverage-gain clause is about the pair --
        # can have every row pass its own rule while the joint clause fails, and a headline
        # reading only ``passed`` would publish "14/14 cells pass" over exactly that.
        "property_cells_passed": lambda data: float(
            (data["properties"]["passed"] & data["properties"]["property_passed"]).sum()
        ),
        "min_coverage_ci_lower": lambda data: float(data["performance"]["coverage_ci_lower"].min()),
        "min_coverage": lambda data: float(data["performance"]["coverage"].min()),
        "min_se_ratio_ci_lower": lambda data: float(data["performance"]["se_ratio_ci_lower"].min()),
        "max_se_ratio_ci_upper": lambda data: float(data["performance"]["se_ratio_ci_upper"].max()),
        "max_se_ratio_resolution": lambda data: float(
            data["performance"]["se_ratio_resolution"].max()
        ),
        "max_standardized_bias": lambda data: float(
            data["performance"]["standardized_bias"].abs().max()
        ),
        "max_rmse_ratio_upper": lambda data: float(data["equivalence"]["rmse_ratio_upper"].max()),
        "min_coverage_difference_lower": lambda data: float(
            data["equivalence"]["coverage_difference_lower"].min()
        ),
        "max_calibration_excess_upper": lambda data: float(
            data["equivalence"]["calibration_excess_upper"].max()
        ),
        "max_margin_utilization": lambda data: float(
            data["equivalence"]["margin_utilization"].max()
        ),
        "cells_with_se_ratio_below_one": lambda data: float(
            (data["summary"]["se_ratio"] < 1.0).sum()
        ),
        "cells_with_coverage_below_nominal": lambda data: float(
            (data["summary"]["coverage"] < 1.0 - record.margins.alpha).sum()
        ),
        "summary_cells": lambda data: count(data["summary"]),
    }


def _scenario(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    return frame.loc[frame["scenario"] == scenario]


def _scenario_aggregates(
    record: StudyRecord,
) -> dict[str, Callable[[Mapping[str, pd.DataFrame]], float]]:
    """The same counts restricted to one scenario, for a law-specific observation."""
    out: dict[str, Callable[[Mapping[str, pd.DataFrame]], float]] = {}
    for scenario in record.scenarios:

        def below_one(data: Mapping[str, pd.DataFrame], scenario: str = scenario) -> float:
            subset = _scenario(data["summary"], scenario)
            return float((subset["se_ratio"] < 1.0).sum())

        def below_nominal(data: Mapping[str, pd.DataFrame], scenario: str = scenario) -> float:
            subset = _scenario(data["summary"], scenario)
            return float((subset["coverage"] < 1.0 - record.margins.alpha).sum())

        def cells(data: Mapping[str, pd.DataFrame], scenario: str = scenario) -> float:
            return float(len(_scenario(data["summary"], scenario)))

        out[f"{scenario}_cells_with_se_ratio_below_one"] = below_one
        out[f"{scenario}_cells_with_coverage_below_nominal"] = below_nominal
        out[f"{scenario}_summary_cells"] = cells
    return out


def thresholds(record: StudyRecord) -> dict[str, float]:
    """Every *declared* number a study's prose may quote, under the name it is declared as.

    The vocabulary above resolves what a study measured.  This resolves what it decided in
    advance, which had the same problem and no gate at all: the rules were restated in prose,
    so moving ``MINIMUM_POWER`` or ``OVERFIT_SE_FLOOR`` left a document asserting a threshold
    the study had not applied, and nothing failed.  Naming them makes the prose resolve
    against the declaration the same way a measured value resolves against an artefact.

    Derived limits are listed as limits rather than as their parts.  A reader checking "the
    interval must lie in [-0.6250, -0.3750]" should not have to compose it from a centre and
    a margin, and the composition is exactly where a hand-typed copy goes wrong.
    """
    margins = record.margins
    declared: dict[str, float] = {
        "margin:confidence_level": margins.confidence_level,
        "margin:alpha": margins.alpha,
        "margin:nominal_coverage": 1.0 - margins.alpha,
        "margin:bootstrap_replicates": float(margins.bootstrap_replicates),
        "margin:standardized_bias": margins.standardized_bias,
        "margin:coverage_floor": margins.coverage_floor,
        "margin:over_coverage_ceiling": margins.over_coverage_ceiling,
        "margin:se_ratio_sanity_lower": margins.se_ratio_sanity[0],
        "margin:se_ratio_sanity_upper": margins.se_ratio_sanity[1],
        "margin:calibration_se_ratio_lower": margins.calibration_se_ratio[0],
        "margin:calibration_se_ratio_upper": margins.calibration_se_ratio[1],
        "margin:calibration_coverage_lower": margins.calibration_coverage[0],
        "margin:calibration_coverage_upper": margins.calibration_coverage[1],
        "margin:type_i_ceiling": margins.alpha + margins.type_i_margin,
        "margin:paired_difference": margins.paired_difference,
        "margin:rmse_noninferiority": margins.rmse_noninferiority,
        "margin:coverage_noninferiority": margins.coverage_noninferiority,
        "margin:calibration_noninferiority": margins.calibration_noninferiority,
        "margin:minimum_power": property_verdicts.MINIMUM_POWER,
        "margin:root_n_slope": property_verdicts.ROOT_N_SLOPE,
        "margin:root_n_slope_lower": (
            property_verdicts.ROOT_N_SLOPE - property_verdicts.ROOT_N_SLOPE_MARGIN
        ),
        "margin:root_n_slope_upper": (
            property_verdicts.ROOT_N_SLOPE + property_verdicts.ROOT_N_SLOPE_MARGIN
        ),
        "margin:excluded_slope": property_verdicts.EXCLUDED_SLOPE,
    }
    if "crossfit_overfitting" in record.property_cells:
        declared.update(
            {
                "margin:overfit_se_floor": property_verdicts.OVERFIT_SE_FLOOR,
                "margin:overfit_control_ceiling": property_verdicts.OVERFIT_SE_CONTROL_CEILING,
                "margin:overfit_coverage_gain": property_verdicts.OVERFIT_COVERAGE_GAIN,
            }
        )
    if "generated_design" in record.property_cells:
        declared["margin:generated_design_deficit"] = record.properties().GENERATED_DESIGN_DEFICIT
    if "selector_necessity" in record.property_cells:
        # Through the record, not by name: this constant belongs to the study that declares
        # the cells, unlike the overfitting margins above, which are shared across families
        # and genuinely live in one module.  ``test_method_evidence`` reads it the same way,
        # so the two cannot come to disagree about which module owns it.
        declared["margin:selector_rmse_ratio"] = record.properties().SELECTOR_RMSE_RATIO
    # Off the declared cells, like the three blocks above, rather than off ``hasattr`` on the
    # module.  A duck-typed guard publishes a threshold because a constant happens to be
    # importable, which is a fact about a file rather than about what the study claims -- and
    # it goes quiet the day the constant is renamed, taking the published row with it.  A
    # noise control is the band's own negative arm: it exists to land *outside* the upper
    # edge, so a study that declares one has declared a band for it to fail.
    if any(
        cell.endswith("noise_control")
        for cell in record.property_cells.get("interval_calibration", ())
    ):
        properties = record.properties()
        low, high = properties.EFFICIENCY_RATIO_BAND
        declared.update(
            {
                "margin:efficiency_ratio_lower": low,
                "margin:efficiency_ratio_upper": high,
                "margin:shrunken_se_factor": properties.SHRUNKEN_SE_FACTOR,
            }
        )
    if "targeting_necessity" in record.property_cells:
        declared["margin:targeting_displacement"] = record.properties().TARGETING_DISPLACEMENT
    # Its own entry rather than a second reader of the one above.  The two families are gated
    # on separate displacements computed from separate arms, and a page that published one
    # threshold for both would describe one of them wrongly however the numbers happened to
    # compare.
    if "survival_recursion_necessity" in record.property_cells:
        declared["margin:recursion_displacement"] = record.properties().RECURSION_DISPLACEMENT
    return declared


def quantities(record: StudyRecord) -> dict[str, Callable[[Mapping[str, pd.DataFrame]], float]]:
    declared = thresholds(record)
    return {
        **_aggregates(record),
        **_scenario_aggregates(record),
        **{name: (lambda data, value=value: value) for name, value in declared.items()},
    }


def value(record: StudyRecord, name: str, data: Mapping[str, pd.DataFrame] | None = None) -> float:
    """Resolve one quantity name against the committed artefacts.

    Two forms.  A bare name is an aggregate from :func:`quantities`.  A reference of the form
    ``properties[double_robustness/both_wrong]:bias`` selects one artefact row by its key
    columns and reads one column, which covers every per-cell number a document can quote
    without a per-study list of them.
    """
    data = load(record) if data is None else data
    match = _REFERENCE.match(name)
    if match is None:
        table = quantities(record)
        if name not in table:
            raise KeyError(
                f"{name!r} is not a known quantity for {record.slug}; "
                f"known aggregates are {sorted(table)}"
            )
        return float(table[name](data))

    artifact = match.group("artifact")
    if artifact not in ARTIFACTS:
        raise KeyError(
            f"{name!r} names artefact {artifact!r}, which is not one of {sorted(ARTIFACTS)}"
        )
    _, key_columns = ARTIFACTS[artifact]
    keys = [part for part in match.group("keys").split("/") if part]
    if len(keys) != len(key_columns):
        raise KeyError(f"{name!r} gives {len(keys)} keys but {artifact} is keyed by {key_columns}")
    frame = data[artifact]
    selected = frame
    for column, key in zip(key_columns, keys, strict=True):
        selected = selected.loc[selected[column].astype(str) == key]
    if len(selected) != 1:
        raise KeyError(f"{name!r} selects {len(selected)} rows of {artifact}, expected exactly one")
    column = match.group("column")
    if column not in selected.columns:
        raise KeyError(f"{artifact} has no column {column!r}")
    return float(selected.iloc[0][column])


_NUMBER = re.compile(r"^-?[\d,]+(?:\.(?P<decimals>\d+))?$")

#: A figure small enough that decimals cannot carry it.  Rounding a scientific figure is
#: rounding to *significant* digits rather than to a place, which is why it needs its own
#: pattern rather than a wider decimal one: 4.45e-08 written to four decimals is ``0.0000``,
#: and a table full of zeros is not a rounding of anything a reader can check.
_SCIENTIFIC = re.compile(r"^-?\d(?:\.(?P<mantissa>\d+))?[eE][-+]?\d+$")


def _significant(text: str) -> int | None:
    """Digits after the mantissa's point, or ``None`` if ``text`` is not scientific."""
    match = _SCIENTIFIC.match(text)
    return None if match is None else len(match.group("mantissa") or "")


def matches(printed: str, computed: float) -> bool:
    """Is ``printed`` what ``computed`` rounds to at the precision it was printed to?

    The rule the wrong figure broke: 0.0145269 printed to four decimals is 0.0145, and a
    document may round as far as it likes so long as rounding is what it did.
    """
    text = printed.strip().rstrip("%")
    scale = 100.0 if printed.strip().endswith("%") else 1.0
    figures = _significant(text)
    if figures is not None:
        return float(f"{computed * scale:.{figures}e}") == float(text)
    match = _NUMBER.match(text)
    if match is None:
        raise ValueError(f"{printed!r} is not a number this gate can check")
    decimals = len(match.group("decimals") or "")
    return round(computed * scale, decimals) == float(text.replace(",", ""))


def describe(computed: float, printed: str) -> str:
    text = printed.strip().rstrip("%")
    scale = 100.0 if printed.strip().endswith("%") else 1.0
    figures = _significant(text)
    if figures is not None:
        return f"printed {printed!r}, computed {float(f'{computed * scale:.{figures}e}')!r}"
    match = _NUMBER.match(text)
    decimals = len(match.group("decimals") or "") if match else 6
    return f"printed {printed!r}, computed {round(computed * scale, decimals)!r} (raw {computed!r})"
