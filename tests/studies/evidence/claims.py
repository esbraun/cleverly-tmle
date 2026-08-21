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


def _aggregates(record: StudyRecord) -> dict[str, Callable[[Mapping[str, pd.DataFrame]], float]]:
    """Study-independent headline quantities, derived rather than declared."""
    subject = record.implementation

    def count(frame: pd.DataFrame) -> float:
        return float(len(frame))

    return {
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


def quantities(record: StudyRecord) -> dict[str, Callable[[Mapping[str, pd.DataFrame]], float]]:
    return {**_aggregates(record), **_scenario_aggregates(record)}


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


def matches(printed: str, computed: float) -> bool:
    """Is ``printed`` what ``computed`` rounds to at the precision it was printed to?

    The rule the wrong figure broke: 0.0145269 printed to four decimals is 0.0145, and a
    document may round as far as it likes so long as rounding is what it did.
    """
    text = printed.strip().rstrip("%")
    match = _NUMBER.match(text)
    if match is None:
        raise ValueError(f"{printed!r} is not a number this gate can check")
    decimals = len(match.group("decimals") or "")
    scale = 100.0 if printed.strip().endswith("%") else 1.0
    return round(computed * scale, decimals) == float(text.replace(",", ""))


def describe(computed: float, printed: str) -> str:
    text = printed.strip().rstrip("%")
    match = _NUMBER.match(text)
    decimals = len(match.group("decimals") or "") if match else 6
    scale = 100.0 if printed.strip().endswith("%") else 1.0
    return f"printed {printed!r}, computed {round(computed * scale, decimals)!r} (raw {computed!r})"
