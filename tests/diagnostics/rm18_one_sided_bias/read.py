"""The one-sided robustness bias reading, computed from committed DR-TMLE rows alone.

RM18 of ``docs/roadmap.md`` declares this reading in "The one-sided robustness reading,
declared before it is computed".  The subsection fixes four statistics and two rules.  This
module computes them from the committed ``replicates.csv.gz`` and ``property-replicates.csv.gz``
of ``canonical-drtmle`` and ``canonical-multi-arm-drtmle``.  It fits no estimator, and it changes
no verdict.

* (i) the ``cleverly`` bias on the primary rows of a configuration.
* (ii) the R ``drtmle`` bias on the same rows.  Binary study only.
* (iii) the paired difference, ``cleverly`` minus R, per replication.  Binary study only, on each
  red configuration and on ``both_correct``.
* (iv) a Welch interval for the property-cell bias minus statistic (i).  Multi-arm study only.

Every declared interval is a 99% interval.  ``--output`` is required, so a bare run cannot
overwrite the committed ``readings.csv``.

The same table also carries supplementary rows, marked ``supplementary`` in its ``scope``
column.  The declaration names none of them, they carry no reading, and no rule reads them.
They were added after the declared reading ran, in answer to a review:

* the multi-arm ``double_robust_contraction/treatment_correct_n2000`` rung, the second sample
  of the same configuration at the same size, against (i) and against the property cell, and
  both samples pooled against (i);
* each binary (iii) at a Bonferroni level over the three paired intervals, because the
  declaration adjusted for no multiplicity;
* the binary ``treatment_correct`` (iii) against the ``outcome_correct`` and ``both_correct``
  (iii), by Welch, because the scenarios draw independent samples.

    python -m tests.diagnostics.rm18_one_sided_bias.read --output <scratch>/readings.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from tests.studies.evidence.inference import Interval, student_interval
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent

#: The level of every bias gate in these studies, and of every interval here.
CONFIDENCE_LEVEL = 0.99

BINARY_STUDY = "canonical-drtmle"
MULTI_ARM_STUDY = "canonical-multi-arm-drtmle"

BINARY_PRIMARY = ROOT / "tests" / "canonical" / "drtmle" / "replicates.csv.gz"
MULTI_ARM_PRIMARY = ROOT / "tests" / "canonical" / "multi_arm_drtmle" / "replicates.csv.gz"
MULTI_ARM_PROPERTIES = (
    ROOT / "tests" / "canonical" / "multi_arm_drtmle" / "property-replicates.csv.gz"
)

BINARY_CLEVERLY = "cleverly"
BINARY_REFERENCE = "drtmle-r"
BINARY_ESTIMAND = "ate"
BINARY_N = 3_000
#: The binary property cells that fail bias equivalence with one nuisance correct.  Each one
#: has a primary scenario of the same name, which R ``drtmle`` also fits.
BINARY_RED = ("outcome_correct", "treatment_correct")
BOTH_CORRECT = "both_correct"

MULTI_ARM_CLEVERLY = "cleverly-multi-arm-drtmle"
MULTI_ARM_SCENARIO = "multi_arm_binary_drtmle"
MULTI_ARM_ESTIMAND = "ate[medium vs high]"
MULTI_ARM_N = 2_000
MULTI_ARM_PROPERTY = "double_robustness"
MULTI_ARM_CELL = "treatment_correct"
#: The contraction rung that carries the same nuisance configuration at the same size.
MULTI_ARM_RUNG_PROPERTY = "double_robust_contraction"
MULTI_ARM_RUNG = "treatment_correct_n2000"

#: The Bonferroni level over the three binary paired intervals.
BONFERRONI_LEVEL = 1.0 - (1.0 - CONFIDENCE_LEVEL) / 3.0

DECLARED = "declared"
SUPPLEMENTARY = "supplementary"

SHARED = "shared"
ESTIMATOR_SPECIFIC = "estimator-specific"
MIXED = "mixed"
UNRESOLVED = "unresolved"
CONSISTENT = "Monte Carlo consistent"
NOT_CONSISTENT = "not consistent"

COVERS = "covers zero"
ABOVE = "above zero"
BELOW = "below zero"

COLUMNS = (
    "study",
    "configuration",
    "statistic",
    "rows",
    "replications",
    "degrees_of_freedom",
    "point",
    "ci_lower",
    "ci_upper",
    "side",
    "reading",
    "scope",
)


@dataclass(frozen=True)
class Statistic:
    """One declared statistic: its point estimate, its interval and the rows it read."""

    name: str
    rows: str
    replications: int
    degrees_of_freedom: float
    point: float
    interval: Interval


def side(interval: Interval) -> str:
    """Where an interval sits relative to zero."""
    if interval.low > 0.0:
        return ABOVE
    if interval.high < 0.0:
        return BELOW
    return COVERS


def on_side_of(interval: Interval, first: Statistic) -> bool:
    """The declaration's test: ``interval`` excludes zero with the sign of the (i) point."""
    sign = np.sign(first.point)
    return (sign > 0 and interval.low > 0.0) or (sign < 0 and interval.high < 0.0)


def binary_reading(
    first: Statistic, second: Statistic, paired: Statistic, both_correct_paired: Statistic
) -> str:
    """The declared binary rule, for one red configuration.

    An (i) interval that covers zero, and a ``both_correct`` (iii) interval that excludes zero,
    each read as unresolved before the three named patterns are tried.
    """
    if side(first.interval) == COVERS or side(both_correct_paired.interval) != COVERS:
        return UNRESOLVED
    reference_on_side = on_side_of(second.interval, first)
    paired_on_side = on_side_of(paired.interval, first)
    reference_covers = side(second.interval) == COVERS
    paired_covers = side(paired.interval) == COVERS
    if reference_on_side and paired_covers:
        return SHARED
    if reference_covers and paired_on_side:
        return ESTIMATOR_SPECIFIC
    if reference_on_side and paired_on_side:
        return MIXED
    return UNRESOLVED


def multi_arm_reading(difference: Statistic) -> str:
    """The declared multi-arm rule, which reads statistic (iv) alone."""
    return CONSISTENT if side(difference.interval) == COVERS else NOT_CONSISTENT


def _student(
    name: str, rows: str, values: np.ndarray, level: float = CONFIDENCE_LEVEL
) -> Statistic:
    interval = student_interval(values, confidence_level=level)
    return Statistic(
        name, rows, len(values), float(len(values) - 1), float(np.mean(values)), interval
    )


def welch(name: str, rows: str, first: np.ndarray, second: np.ndarray) -> Statistic:
    """A Welch interval for ``mean(first) - mean(second)``, with Satterthwaite's df."""
    first_variance = float(np.var(first, ddof=1)) / len(first)
    second_variance = float(np.var(second, ddof=1)) / len(second)
    standard_error = math.sqrt(first_variance + second_variance)
    degrees_of_freedom = (first_variance + second_variance) ** 2 / (
        first_variance**2 / (len(first) - 1) + second_variance**2 / (len(second) - 1)
    )
    point = float(np.mean(first) - np.mean(second))
    half_width = float(t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, degrees_of_freedom)) * standard_error
    return Statistic(
        name,
        rows,
        len(first) + len(second),
        degrees_of_freedom,
        point,
        Interval(point - half_width, point + half_width),
    )


def _errors(
    rows: pd.DataFrame, implementation: str, scenario: str, estimand: str, n: int
) -> pd.Series:
    """Estimate minus truth, indexed by replication, for one primary configuration."""
    selected = rows.loc[
        (rows["implementation"] == implementation)
        & (rows["scenario"] == scenario)
        & (rows["estimand"] == estimand)
        & (rows["n"] == n)
    ]
    if selected.empty:
        raise ValueError(f"no primary rows for {implementation} {scenario} {estimand} n={n}")
    if selected["replicate"].duplicated().any():
        raise ValueError(f"duplicate replications for {implementation} {scenario} {estimand}")
    errors = (selected["estimate"] - selected["truth"]).astype(float)
    errors.index = selected["replicate"].to_numpy()
    return errors.sort_index()


def _paired(cleverly: pd.Series, reference: pd.Series, scenario: str) -> np.ndarray:
    if not cleverly.index.equals(reference.index):
        raise ValueError(f"the two implementations read different replications on {scenario}")
    return (cleverly - reference).to_numpy()


def binary_statistics(rows: pd.DataFrame) -> dict[str, dict[str, Statistic]]:
    """Statistics (i), (ii) and (iii) per red configuration, and (iii) on ``both_correct``."""
    out: dict[str, dict[str, Statistic]] = {}
    for scenario in (*BINARY_RED, BOTH_CORRECT):
        cleverly = _errors(rows, BINARY_CLEVERLY, scenario, BINARY_ESTIMAND, BINARY_N)
        reference = _errors(rows, BINARY_REFERENCE, scenario, BINARY_ESTIMAND, BINARY_N)
        label = f"primary {scenario}, {BINARY_ESTIMAND}, n={BINARY_N}"
        paired = _student(
            "(iii)",
            f"{label}, {BINARY_CLEVERLY} minus {BINARY_REFERENCE}",
            _paired(cleverly, reference, scenario),
        )
        if scenario == BOTH_CORRECT:
            out[scenario] = {"(iii)": paired}
            continue
        out[scenario] = {
            "(i)": _student("(i)", f"{label}, {BINARY_CLEVERLY}", cleverly.to_numpy()),
            "(ii)": _student("(ii)", f"{label}, {BINARY_REFERENCE}", reference.to_numpy()),
            "(iii)": paired,
        }
    return out


def _cell_errors(
    primary: pd.DataFrame, properties: pd.DataFrame, family: str, name: str
) -> np.ndarray:
    """Estimate minus truth for one multi-arm property cell, checked against the primary law."""
    cell = properties.loc[(properties["property"] == family) & (properties["cell"] == name)]
    if set(cell["n"]) != {MULTI_ARM_N}:
        raise ValueError(f"{family}/{name} runs at n={sorted(set(cell['n']))}, not {MULTI_ARM_N}")
    primary_truth = set(
        primary.loc[primary["estimand"] == MULTI_ARM_ESTIMAND, "truth"].round(12).unique()
    )
    if set(cell["truth"].round(12).unique()) != primary_truth:
        raise ValueError(f"{family}/{name} and the primary estimand have different truths")
    return (cell["estimate"] - cell["truth"]).to_numpy(dtype=float)


def multi_arm_statistics(primary: pd.DataFrame, properties: pd.DataFrame) -> dict[str, Statistic]:
    """Statistic (i) on the primary rows, the property-cell bias it is set against, and (iv)."""
    first_errors = _errors(
        primary, MULTI_ARM_CLEVERLY, MULTI_ARM_SCENARIO, MULTI_ARM_ESTIMAND, MULTI_ARM_N
    )
    cell_errors = _cell_errors(primary, properties, MULTI_ARM_PROPERTY, MULTI_ARM_CELL)
    first_label = f"primary {MULTI_ARM_SCENARIO}, {MULTI_ARM_ESTIMAND}, n={MULTI_ARM_N}"
    cell_label = f"property {MULTI_ARM_PROPERTY}/{MULTI_ARM_CELL}, n={MULTI_ARM_N}"
    return {
        "(i)": _student("(i)", f"{first_label}, {MULTI_ARM_CLEVERLY}", first_errors.to_numpy()),
        "property-cell bias": _student("property-cell bias", cell_label, cell_errors),
        "(iv)": welch(
            "(iv)",
            f"{cell_label} minus {first_label}",
            cell_errors,
            first_errors.to_numpy(),
        ),
    }


def supplementary_statistics(
    binary: pd.DataFrame, primary: pd.DataFrame, properties: pd.DataFrame
) -> list[tuple[str, str, Statistic]]:
    """The supplementary rows: ``(study, configuration, statistic)``, in table order.

    None of them enters a reading.  The module docstring says why each one is here.
    """
    out: list[tuple[str, str, Statistic]] = []
    paired: dict[str, np.ndarray] = {}
    for scenario in (*BINARY_RED, BOTH_CORRECT):
        cleverly = _errors(binary, BINARY_CLEVERLY, scenario, BINARY_ESTIMAND, BINARY_N)
        reference = _errors(binary, BINARY_REFERENCE, scenario, BINARY_ESTIMAND, BINARY_N)
        paired[scenario] = _paired(cleverly, reference, scenario)
        label = f"primary {scenario}, {BINARY_ESTIMAND}, n={BINARY_N}"
        out.append(
            (
                BINARY_STUDY,
                scenario,
                _student(
                    "(iii) at the Bonferroni level over three",
                    f"{label}, {BINARY_CLEVERLY} minus {BINARY_REFERENCE}, "
                    f"level {BONFERRONI_LEVEL:.6f}",
                    paired[scenario],
                    BONFERRONI_LEVEL,
                ),
            )
        )
    for other in ("outcome_correct", BOTH_CORRECT):
        out.append(
            (
                BINARY_STUDY,
                "treatment_correct",
                welch(
                    f"(iii) minus {other} (iii)",
                    f"paired differences on primary treatment_correct minus those on {other}",
                    paired["treatment_correct"],
                    paired[other],
                ),
            )
        )
    first = _errors(
        primary, MULTI_ARM_CLEVERLY, MULTI_ARM_SCENARIO, MULTI_ARM_ESTIMAND, MULTI_ARM_N
    ).to_numpy()
    cell = _cell_errors(primary, properties, MULTI_ARM_PROPERTY, MULTI_ARM_CELL)
    rung = _cell_errors(primary, properties, MULTI_ARM_RUNG_PROPERTY, MULTI_ARM_RUNG)
    first_label = f"primary {MULTI_ARM_SCENARIO}, {MULTI_ARM_ESTIMAND}, n={MULTI_ARM_N}"
    cell_label = f"property {MULTI_ARM_PROPERTY}/{MULTI_ARM_CELL}, n={MULTI_ARM_N}"
    rung_label = f"property {MULTI_ARM_RUNG_PROPERTY}/{MULTI_ARM_RUNG}, n={MULTI_ARM_N}"
    for statistic in (
        _student("rung bias", rung_label, rung),
        welch("rung bias minus (i)", f"{rung_label} minus {first_label}", rung, first),
        welch("property-cell bias minus rung bias", f"{cell_label} minus {rung_label}", cell, rung),
        welch(
            "pooled bias minus (i)",
            f"{cell_label} and {rung_label} pooled, minus {first_label}",
            np.concatenate([cell, rung]),
            first,
        ),
    ):
        out.append((MULTI_ARM_STUDY, MULTI_ARM_CELL, statistic))
    return out


def _record(
    study: str,
    configuration: str,
    statistic: Statistic,
    reading: str,
    scope: str = DECLARED,
) -> dict[str, object]:
    return {
        "study": study,
        "configuration": configuration,
        "statistic": statistic.name,
        "rows": statistic.rows,
        "replications": statistic.replications,
        "degrees_of_freedom": statistic.degrees_of_freedom,
        "point": statistic.point,
        "ci_lower": statistic.interval.low,
        "ci_upper": statistic.interval.high,
        "side": side(statistic.interval),
        "reading": reading,
        "scope": scope,
    }


def readings(
    binary: pd.DataFrame, multi_arm_primary: pd.DataFrame, multi_arm_properties: pd.DataFrame
) -> pd.DataFrame:
    """Every declared statistic as one row, each carrying the reading of its configuration.

    The ``both_correct`` row enters both binary readings and carries none of its own.  The
    supplementary rows follow the declared ones, and carry no reading.
    """
    records = []
    statistics = binary_statistics(binary)
    both_correct = statistics[BOTH_CORRECT]["(iii)"]
    for scenario in BINARY_RED:
        chosen = statistics[scenario]
        reading = binary_reading(chosen["(i)"], chosen["(ii)"], chosen["(iii)"], both_correct)
        records.extend(
            _record(BINARY_STUDY, scenario, statistic, reading) for statistic in chosen.values()
        )
    records.append(_record(BINARY_STUDY, BOTH_CORRECT, both_correct, ""))
    multi_arm = multi_arm_statistics(multi_arm_primary, multi_arm_properties)
    reading = multi_arm_reading(multi_arm["(iv)"])
    records.extend(
        _record(MULTI_ARM_STUDY, MULTI_ARM_CELL, statistic, reading)
        for statistic in multi_arm.values()
    )
    records.extend(
        _record(study, configuration, statistic, "", SUPPLEMENTARY)
        for study, configuration, statistic in supplementary_statistics(
            binary, multi_arm_primary, multi_arm_properties
        )
    )
    return pd.DataFrame(records, columns=list(COLUMNS))


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three committed artefacts the reading consumes."""
    return (
        pd.read_csv(BINARY_PRIMARY),
        pd.read_csv(MULTI_ARM_PRIMARY),
        pd.read_csv(MULTI_ARM_PROPERTIES),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"where to write the table; the committed record is {HERE / 'readings.csv'}",
    )
    arguments = parser.parse_args()
    frame = readings(*load())
    write_csv(frame, arguments.output)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(frame.drop(columns="rows").to_string(index=False))
    print(f"wrote {len(frame)} rows to {arguments.output}")


if __name__ == "__main__":
    main()
