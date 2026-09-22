"""The RM18 one-sided robustness reading: committed output, declared rules and mutations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from tests.diagnostics.rm18_one_sided_bias.read import (
    BINARY_ESTIMAND,
    BINARY_REFERENCE,
    BOTH_CORRECT,
    CONSISTENT,
    ESTIMATOR_SPECIFIC,
    HERE,
    MIXED,
    MULTI_ARM_CELL,
    MULTI_ARM_SCENARIO,
    MULTI_ARM_STUDY,
    NOT_CONSISTENT,
    SHARED,
    UNRESOLVED,
    Statistic,
    binary_reading,
    load,
    multi_arm_reading,
    readings,
    welch,
)
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.registry import ROOT

COMMITTED = HERE / "readings.csv"
CANONICAL = ROOT / "tests" / "canonical"
TEXT = ("study", "configuration", "statistic", "rows", "side", "reading")
NUMBERS = ("replications", "degrees_of_freedom", "point", "ci_lower", "ci_upper")


@pytest.fixture(scope="module")
def artefacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load()


@pytest.fixture(scope="module")
def rebuilt(artefacts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return readings(*artefacts)


def _reading(frame: pd.DataFrame, study: str, configuration: str) -> str:
    chosen = frame.loc[(frame["study"] == study) & (frame["configuration"] == configuration)]
    values = set(chosen["reading"])
    assert len(values) == 1, values
    return str(values.pop())


def test_committed_readings_are_rebuilt_from_the_committed_artefacts(
    rebuilt: pd.DataFrame,
) -> None:
    committed = pd.read_csv(COMMITTED, keep_default_na=False)
    assert list(committed.columns) == list(rebuilt.columns)
    assert len(committed) == len(rebuilt)
    for column in TEXT:
        assert committed[column].astype(str).tolist() == rebuilt[column].astype(str).tolist()
    for column in NUMBERS:
        np.testing.assert_allclose(
            committed[column].to_numpy(dtype=float),
            rebuilt[column].to_numpy(dtype=float),
            rtol=1e-12,
            atol=0.0,
        )


def _statistic(point: float, low: float, high: float) -> Statistic:
    return Statistic("x", "synthetic", 800, 799.0, point, Interval(low, high))


ABOVE = _statistic(0.002, 0.001, 0.003)
BELOW = _statistic(-0.002, -0.003, -0.001)
COVERS = _statistic(0.0001, -0.001, 0.001)
COVERS_NEGATIVE = _statistic(-0.0001, -0.001, 0.001)


@pytest.mark.parametrize(
    ("first", "second", "paired", "both_correct", "expected"),
    [
        pytest.param(ABOVE, ABOVE, COVERS, COVERS, SHARED, id="shared"),
        pytest.param(ABOVE, COVERS, ABOVE, COVERS, ESTIMATOR_SPECIFIC, id="estimator-specific"),
        pytest.param(ABOVE, ABOVE, ABOVE, COVERS, MIXED, id="mixed"),
        pytest.param(BELOW, BELOW, BELOW, COVERS, MIXED, id="mixed-below"),
        pytest.param(BELOW, BELOW, COVERS, COVERS, SHARED, id="shared-below"),
        pytest.param(COVERS, ABOVE, COVERS, COVERS, UNRESOLVED, id="first-covers-zero"),
        pytest.param(ABOVE, ABOVE, COVERS, ABOVE, UNRESOLVED, id="both-correct-excludes"),
        pytest.param(ABOVE, ABOVE, COVERS, BELOW, UNRESOLVED, id="both-correct-excludes-below"),
        pytest.param(ABOVE, BELOW, COVERS, COVERS, UNRESOLVED, id="reference-opposite-side"),
        pytest.param(ABOVE, COVERS, BELOW, COVERS, UNRESOLVED, id="paired-opposite-side"),
        pytest.param(ABOVE, COVERS, COVERS, COVERS, UNRESOLVED, id="both-cover"),
    ],
)
def test_binary_rule_follows_the_declaration(
    first: Statistic, second: Statistic, paired: Statistic, both_correct: Statistic, expected: str
) -> None:
    assert binary_reading(first, second, paired, both_correct) == expected


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        pytest.param(COVERS, CONSISTENT, id="covers"),
        pytest.param(COVERS_NEGATIVE, CONSISTENT, id="covers-negative-point"),
        pytest.param(ABOVE, NOT_CONSISTENT, id="above"),
        pytest.param(BELOW, NOT_CONSISTENT, id="below"),
    ],
)
def test_multi_arm_rule_follows_the_declaration(difference: Statistic, expected: str) -> None:
    assert multi_arm_reading(difference) == expected


def test_welch_interval_matches_scipy() -> None:
    rng = np.random.default_rng(18)
    first = rng.normal(0.01, 0.03, size=600)
    second = rng.normal(0.0, 0.02, size=800)
    ours = welch("(iv)", "synthetic", first, second)
    theirs = ttest_ind(first, second, equal_var=False).confidence_interval(0.99)
    assert ours.interval.low == pytest.approx(theirs.low, rel=1e-10)
    assert ours.interval.high == pytest.approx(theirs.high, rel=1e-10)


def _shift_reference(binary: pd.DataFrame, scenario: str, shift: float) -> pd.DataFrame:
    mutated = binary.copy()
    rows = (
        (mutated["implementation"] == BINARY_REFERENCE)
        & (mutated["scenario"] == scenario)
        & (mutated["estimand"] == BINARY_ESTIMAND)
    )
    mutated.loc[rows, "estimate"] += shift
    return mutated


@pytest.mark.parametrize(
    ("scenario", "shift", "configuration", "expected"),
    [
        # R drtmle's bias moves to zero: the whole cleverly bias becomes its increment.
        pytest.param(
            "treatment_correct", -0.0021, "treatment_correct", ESTIMATOR_SPECIFIC, id="to-zero"
        ),
        # R drtmle takes on the paired increment: the bias becomes shared.
        pytest.param("treatment_correct", 0.001, "treatment_correct", SHARED, id="to-shared"),
        # R drtmle loses its outcome_correct bias, so the shared bias becomes cleverly's own.
        pytest.param(
            "outcome_correct", -0.0019, "outcome_correct", ESTIMATOR_SPECIFIC, id="outcome-to-zero"
        ),
        # The implementations differ where neither nuisance is wrong.
        pytest.param(BOTH_CORRECT, 0.001, "treatment_correct", UNRESOLVED, id="both-correct"),
        pytest.param(BOTH_CORRECT, 0.001, "outcome_correct", UNRESOLVED, id="both-correct-o"),
    ],
)
def test_shifting_reference_rows_changes_the_reading(
    artefacts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    rebuilt: pd.DataFrame,
    scenario: str,
    shift: float,
    configuration: str,
    expected: str,
) -> None:
    binary, primary, properties = artefacts
    before = _reading(rebuilt, "canonical-drtmle", configuration)
    after = _reading(
        readings(_shift_reference(binary, scenario, shift), primary, properties),
        "canonical-drtmle",
        configuration,
    )
    assert after == expected
    assert after != before


def test_shifting_the_multi_arm_property_cell_changes_the_reading(
    artefacts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rebuilt: pd.DataFrame
) -> None:
    binary, primary, properties = artefacts
    mutated = properties.copy()
    cell = (mutated["property"] == "double_robustness") & (mutated["cell"] == MULTI_ARM_CELL)
    mutated.loc[cell, "estimate"] -= 0.0049
    before = _reading(rebuilt, MULTI_ARM_STUDY, MULTI_ARM_CELL)
    after = _reading(readings(binary, primary, mutated), MULTI_ARM_STUDY, MULTI_ARM_CELL)
    assert (before, after) == (NOT_CONSISTENT, CONSISTENT)


def test_multi_arm_treatment_correct_has_no_comparator_row(
    artefacts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rebuilt: pd.DataFrame
) -> None:
    _, primary, properties = artefacts
    # R drtmle fits the primary scenario alone, where both nuisances are correct.
    assert set(primary["scenario"]) == {MULTI_ARM_SCENARIO}
    # The property rows carry no implementation column: only cleverly fits a property cell.
    assert "implementation" not in properties.columns
    multi_arm = rebuilt.loc[rebuilt["study"] == MULTI_ARM_STUDY, "statistic"]
    assert set(multi_arm).isdisjoint({"(ii)", "(iii)"})


def _interval(frame: pd.DataFrame, study: str, configuration: str, statistic: str) -> Interval:
    row = frame.loc[
        (frame["study"] == study)
        & (frame["configuration"] == configuration)
        & (frame["statistic"] == statistic)
    ].iloc[0]
    return Interval(float(row["ci_lower"]), float(row["ci_upper"]))


def _published(path: Path, low: str, high: str, **keys: object) -> Interval:
    frame = pd.read_csv(path)
    for column, value in keys.items():
        frame = frame.loc[frame[column] == value]
    assert len(frame) == 1, keys
    return Interval(float(frame[low].iloc[0]), float(frame[high].iloc[0]))


def test_statistics_equal_the_intervals_each_study_publishes(rebuilt: pd.DataFrame) -> None:
    binary = CANONICAL / "drtmle"
    multi_arm = CANONICAL / "multi_arm_drtmle"
    bias = ("bias_ci_lower", "bias_ci_upper")
    paired = ("paired_ci_lower", "paired_ci_upper")
    expected = []
    for scenario in ("outcome_correct", "treatment_correct"):
        for statistic, implementation in (("(i)", "cleverly"), ("(ii)", BINARY_REFERENCE)):
            published = _published(
                binary / "performance-tests.csv",
                *bias,
                implementation=implementation,
                scenario=scenario,
                estimand=BINARY_ESTIMAND,
            )
            expected.append(("canonical-drtmle", scenario, statistic, published))
    for scenario in ("outcome_correct", "treatment_correct", BOTH_CORRECT):
        published = _published(
            binary / "equivalence.csv", *paired, scenario=scenario, estimand=BINARY_ESTIMAND
        )
        expected.append(("canonical-drtmle", scenario, "(iii)", published))
    expected.append(
        (
            MULTI_ARM_STUDY,
            MULTI_ARM_CELL,
            "(i)",
            _published(
                multi_arm / "performance-tests.csv",
                *bias,
                implementation="cleverly-multi-arm-drtmle",
                estimand="ate[medium vs high]",
            ),
        )
    )
    expected.append(
        (
            MULTI_ARM_STUDY,
            MULTI_ARM_CELL,
            "property-cell bias",
            _published(
                multi_arm / "properties.csv",
                *bias,
                property="double_robustness",
                cell=MULTI_ARM_CELL,
            ),
        )
    )
    for study, configuration, statistic, published in expected:
        ours = _interval(rebuilt, study, configuration, statistic)
        assert ours.low == pytest.approx(published.low, rel=1e-9), (configuration, statistic)
        assert ours.high == pytest.approx(published.high, rel=1e-9), (configuration, statistic)
