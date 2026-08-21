"""Render each method-evidence study's reader-facing result tables.

The committed CSV artifacts are the source of truth. Study pages keep their explanatory prose
by hand, while the block between :data:`RESULTS_START` and :data:`RESULTS_END` is regenerated from
those artifacts. Run this explicitly after regenerating a study::

    python -m tests.studies.evidence.document

It remains separate from the statistical regeneration command so a documentation change is
reviewed as a documentation change rather than appearing as a side effect of a simulation run.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping

import pandas as pd

from tests.studies.evidence.claims import load
from tests.studies.evidence.registry import StudyRecord, registered

RESULTS_START = "<!-- BEGIN GENERATED STUDY RESULTS -->"
RESULTS_END = "<!-- END GENERATED STUDY RESULTS -->"


def number(value: object) -> str:
    """Render a result compactly without rounding a small nonzero value to zero."""
    numeric = float(value)
    if math.isnan(numeric):
        return "N/A"
    if numeric == int(numeric):
        return f"{int(numeric):,}"
    if 0.0 < abs(numeric) < 0.001:
        return f"{numeric:.6f}"
    return f"{numeric:.4f}"


def verdict(value: object) -> str:
    """Use a textual verdict so the result never depends on color."""
    return "Pass" if bool(value) else "Fail"


def interval(row: pd.Series, low: str, high: str) -> str:
    return f"[{number(row[low])}, {number(row[high])}]"


def _heading(value: object) -> str:
    text = str(value)
    labels = {
        "double_robustness": "Double robustness",
        "root_n_and_efficiency": "Root-n and efficiency",
        "root_n_rate": "Root-n rate",
        "interval_calibration": "Interval calibration",
        "type_i_error": "Type-I error",
        "power": "Power",
        "crossfit_overfitting": "Cross-fit overfitting",
    }
    return labels.get(text, text.replace("_", " ").title())


def _performance(record: StudyRecord, frame: pd.DataFrame) -> list[str]:
    lines = ["### Performance versus truth", ""]
    for scenario in record.scenarios:
        lines.extend(
            [
                f"#### {_heading(scenario)} outcome law",
                "",
                "| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | "
                "coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |",
            ]
        )
        selected = frame.loc[frame["scenario"] == scenario].copy()
        order = {name: index for index, name in enumerate(record.scenarios[scenario])}
        implementations = {name: index for index, name in enumerate(record.implementations)}
        selected["_estimand_order"] = selected["estimand"].map(order)
        selected["_implementation_order"] = selected["implementation"].map(implementations)
        selected = selected.sort_values(["_implementation_order", "_estimand_order"])
        for _, row in selected.iterrows():
            lines.append(
                "| {implementation} | `{estimand}` | {scale} | {bias} {bias_ci} | "
                "±{bias_margin} | {bias_test} | {coverage} {coverage_ci} | {coverage_test} | "
                "{se_ratio} {se_ci} | {se_test} | {overall} |".format(
                    implementation=row["implementation"],
                    estimand=row["estimand"],
                    scale=row["inference_scale"],
                    bias=number(row["bias"]),
                    bias_ci=interval(row, "bias_ci_lower", "bias_ci_upper"),
                    bias_margin=number(row["bias_margin"]),
                    bias_test=verdict(row["bias_equivalent"]),
                    coverage=number(row["coverage"]),
                    coverage_ci=interval(row, "coverage_ci_lower", "coverage_ci_upper"),
                    coverage_test=verdict(row["coverage_valid"]),
                    se_ratio=number(row["se_ratio"]),
                    se_ci=interval(row, "se_ratio_ci_lower", "se_ratio_ci_upper"),
                    se_test=verdict(row["se_calibrated"]),
                    overall=verdict(row["passed"]),
                )
            )
        lines.append("")
    return lines


def _equivalence(record: StudyRecord, frame: pd.DataFrame) -> list[str]:
    lines = ["### Cross-implementation tests", ""]
    if record.reference is None:
        lines.extend(
            [
                "No external comparison is registered for this study; its committed comparison "
                "artifact therefore contains zero tests.",
                "",
            ]
        )
        return lines

    for scenario in record.scenarios:
        lines.extend(
            [
                f"#### {_heading(scenario)} outcome law",
                "",
                "| estimand | paired difference (99% CI) | similarity margin | similarity | "
                "RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | "
                "coverage NI | calibration excess (99% upper) | calibration NI | overall |",
                "| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |",
            ]
        )
        selected = frame.loc[frame["scenario"] == scenario].copy()
        order = {name: index for index, name in enumerate(record.scenarios[scenario])}
        selected["_estimand_order"] = selected["estimand"].map(order)
        selected = selected.sort_values("_estimand_order")
        for _, row in selected.iterrows():
            comparable = bool(row["se_comparable"])
            calibration_value = (
                f"{number(row['subject_calibration_excess'])} "
                f"({number(row['calibration_excess_upper'])})"
                if comparable
                else "N/A"
            )
            calibration_test = (
                verdict(row["calibration_excess_upper"] <= row["calibration_noninferiority_margin"])
                if comparable
                else "N/A"
            )
            lines.append(
                "| `{estimand}` | {difference} {difference_ci} | ±{similarity_margin} | "
                "{similarity} | {rmse} ({rmse_upper}) | {rmse_test} | "
                "{coverage} ({coverage_lower}) | "
                "{coverage_test} | {calibration} | {calibration_test} | {overall} |".format(
                    estimand=row["estimand"],
                    difference=number(row["mean_difference"]),
                    difference_ci=interval(row, "paired_ci_lower", "paired_ci_upper"),
                    similarity_margin=number(row["mean_margin"]),
                    similarity=verdict(row["paired_similarity"]),
                    rmse=number(row["rmse_ratio"]),
                    rmse_upper=number(row["rmse_ratio_upper"]),
                    rmse_test=verdict(row["rmse_ratio_upper"] <= row["rmse_noninferiority_margin"]),
                    coverage=number(row["coverage_difference"]),
                    coverage_lower=number(row["coverage_difference_lower"]),
                    coverage_test=verdict(
                        row["coverage_difference_lower"] >= row["coverage_noninferiority_margin"]
                    ),
                    calibration=calibration_value,
                    calibration_test=calibration_test,
                    overall=verdict(row["passed"]),
                )
            )
        lines.append("")
    return lines


def _property_result(record: StudyRecord, row: pd.Series) -> tuple[str, str, str]:
    margins = record.margins
    property_name = str(row["property"])
    cell = str(row["cell"])
    if property_name == "double_robustness":
        result = f"bias {number(row['bias'])}"
        evidence = f"bias CI {interval(row, 'bias_ci_lower', 'bias_ci_upper')}"
        direction = "outside" if cell == "both_wrong" else "inside"
        decision = f"99% bias CI {direction} ±{number(row['bias_margin'])}"
    elif property_name == "root_n_and_efficiency":
        result = (
            f"bias {number(row['bias'])}; coverage {number(row['coverage'])}; "
            f"SE ratio {number(row['se_ratio'])}"
        )
        evidence = (
            f"bias CI {interval(row, 'bias_ci_lower', 'bias_ci_upper')}; "
            f"coverage CI {interval(row, 'coverage_ci_lower', 'coverage_ci_upper')}"
        )
        decision = (
            f"bias equivalent; coverage lower ≥ {number(margins.coverage_floor)}; "
            f"SE ratio in [{number(margins.se_ratio_sanity[0])}, "
            f"{number(margins.se_ratio_sanity[1])}]"
        )
    elif property_name == "root_n_rate":
        result = f"slope {number(row['slope'])}"
        evidence = f"slope CI {interval(row, 'slope_ci_lower', 'slope_ci_upper')}"
        decision = "99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500"
    elif property_name == "interval_calibration":
        result = f"coverage {number(row['coverage'])}; SE ratio {number(row['se_ratio'])}"
        evidence = (
            f"coverage CI {interval(row, 'coverage_ci_lower', 'coverage_ci_upper')}; "
            f"SE-ratio CI {interval(row, 'se_ratio_ci_lower', 'se_ratio_ci_upper')}"
        )
        decision = (
            f"coverage CI in [{number(margins.calibration_coverage[0])}, "
            f"{number(margins.calibration_coverage[1])}]; SE-ratio CI in "
            f"[{number(margins.calibration_se_ratio[0])}, "
            f"{number(margins.calibration_se_ratio[1])}]"
        )
    elif property_name == "type_i_error":
        result = f"rejection {number(row['rejection_rate'])}; coverage {number(row['coverage'])}"
        evidence = (
            f"rejection upper {number(row['rejection_ci_upper'])}; "
            f"coverage lower {number(row['coverage_ci_lower'])}"
        )
        decision = (
            f"rejection upper ≤ {number(margins.alpha + margins.type_i_margin)}; "
            f"coverage lower ≥ {number(margins.coverage_floor)}"
        )
    elif property_name == "power":
        result = f"rejection {number(row['rejection_rate'])}"
        evidence = f"rejection CI {interval(row, 'rejection_ci_lower', 'rejection_ci_upper')}"
        decision = "99% rejection lower ≥ 0.8000"
    elif property_name == "crossfit_overfitting":
        result = f"coverage {number(row['coverage'])}; SE ratio {number(row['se_ratio'])}"
        evidence = (
            f"SE-ratio CI {interval(row, 'se_ratio_ci_lower', 'se_ratio_ci_upper')}; "
            f"coverage-gain CI "
            f"{interval(row, 'coverage_gain_ci_lower', 'coverage_gain_ci_upper')}"
        )
        decision = (
            "cross-fit SE-ratio CI in [0.8500, 1.2000]; control upper ≤ 0.7500; "
            "coverage-gain lower ≥ 0.1500"
        )
    else:  # pragma: no cover - a new property must teach the renderer how it is read
        raise ValueError(f"no reader-facing result format for property {property_name!r}")
    return result, evidence, decision


def _properties(record: StudyRecord, frame: pd.DataFrame) -> list[str]:
    lines = [
        "### Scientific-property and control tests",
        "",
        "| test | cell | n | replications | observed result | uncertainty | decision rule | status |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for property_name, cells in record.property_cells.items():
        for cell in cells:
            selected = frame.loc[(frame["property"] == property_name) & (frame["cell"] == cell)]
            if len(selected) != 1:
                raise ValueError(
                    f"{record.slug} has {len(selected)} rows for {property_name}/{cell}, expected 1"
                )
            row = selected.iloc[0]
            result, evidence, decision = _property_result(record, row)
            lines.append(
                f"| {_heading(property_name)} | `{cell}` | {number(row['n'])} | "
                f"{number(row['replicates'])} | {result} | {evidence} | {decision} | "
                f"{verdict(row['passed'])} |"
            )
    lines.append("")
    return lines


def render_results(record: StudyRecord, data: Mapping[str, pd.DataFrame] | None = None) -> str:
    """Render all committed test rows for one study in registered order."""
    loaded = load(record) if data is None else data
    lines = [
        *_performance(record, loaded["performance"]),
        *_equivalence(record, loaded["equivalence"]),
        *_properties(record, loaded["properties"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def generated_results(document: str) -> str:
    """Return the generated block from a study page, excluding its markers."""
    _, separator, rest = document.partition(RESULTS_START)
    if not separator:
        raise ValueError(f"document has no {RESULTS_START!r} marker")
    body, separator, _ = rest.partition(RESULTS_END)
    if not separator:
        raise ValueError(f"document has no {RESULTS_END!r} marker")
    return body.strip("\n") + "\n"


def refresh(record: StudyRecord) -> bool:
    """Refresh one page's generated result block and report whether it changed."""
    document = record.document_path
    original = document.read_text(encoding="utf-8")
    expected = render_results(record)
    current = generated_results(original)
    if current == expected:
        return False
    prefix, _, rest = original.partition(RESULTS_START)
    _, _, suffix = rest.partition(RESULTS_END)
    updated = prefix + RESULTS_START + "\n" + expected + RESULTS_END + suffix
    document.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", help="only this study; default is every registered one")
    arguments = parser.parse_args()
    for record in registered():
        if arguments.slug and record.slug != arguments.slug:
            continue
        changed = refresh(record)
        print(f"{record.slug}: {'updated' if changed else 'already current'}")


if __name__ == "__main__":
    main()
