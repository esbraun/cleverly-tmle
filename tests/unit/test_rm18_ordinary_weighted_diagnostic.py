"""The RM18 OW diagnostic: the exact-law witness, seeds, the three reading rules, the harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_fixed_weights.run import BOUNDS
from tests.diagnostics.rm18_ordinary_weighted import run as ow
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.evidence.inference import Interval


def test_the_exact_frame_realises_the_selected_law_with_its_weights() -> None:
    frame = shared.exact_selected_frame()
    assert len(frame) == law.N // 2 * 1 + law.N // 2 * 3
    expected = np.where(frame["W"] > 0, 1.0 / weighted.SELECTION_LOW, 1.0 / weighted.SELECTION_HIGH)
    assert np.array_equal(frame["obs_weight"].to_numpy(), expected)


def test_the_exact_untargeted_value_is_the_longhand_follower_mean() -> None:
    held, largest = ow.exact_precondition()
    assert held, largest


def test_dropping_the_weights_moves_the_exact_untargeted_value() -> None:
    """The nonzero witness: the precondition is not blind to the weights."""
    weighted_values = ow.exact_untargeted("mechanism_correct")
    unweighted_values = ow.exact_untargeted("mechanism_correct", use_weights=False)
    for label in ("always", "never"):
        assert unweighted_values[label] == pytest.approx(
            ow.follower_mean(label, use_weights=False), abs=ow.EXACT_TOLERANCE
        )
        assert abs(unweighted_values[label] - ow.follower_mean(label)) > 1e-3
        assert abs(weighted_values[label] - unweighted_values[label]) > 1e-3


def test_b_inf_is_the_exact_untargeted_contrast_minus_its_truth() -> None:
    values = ow.exact_untargeted("mechanism_correct")
    assert ow.b_inf() == values["always"] - values["never"] - law.TRUTH[shared.STATIC]


def test_the_discrimination_probability_follows_its_formula() -> None:
    quantile = float(student.ppf(0.995, 1_199))
    # At the discrimination threshold of one Student half-width the chance is one half.
    threshold = ow.MARGIN + quantile / np.sqrt(1_200)
    assert ow.discrimination_probability(threshold, 1_200) == pytest.approx(0.5, abs=1e-6)
    assert ow.discrimination_probability(ow.MARGIN, 1_200) < 0.01
    assert ow.discrimination_probability(0.5, 1_200) > 0.99


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (Interval(0.975, 1.02), ow.CONTRACTING),
        (Interval(0.90, 0.969), ow.PERSISTENT),
        (Interval(0.96, 0.99), ow.UNRESOLVED),
        (Interval(1.0, 1.04), ow.UNRESOLVED),
    ],
)
def test_the_calibration_rule(interval: Interval, expected: str) -> None:
    assert ow.calibration_label(interval) == expected


@pytest.mark.parametrize(
    ("rejection", "coverage", "expected"),
    [
        (Interval(0.04, 0.10), Interval(0.90, 0.97), ow.WITHIN),
        (Interval(0.101, 0.13), Interval(0.90, 0.97), ow.OUTSIDE),
        (Interval(0.04, 0.09), Interval(0.85, 0.899), ow.OUTSIDE),
        (Interval(0.04, 0.11), Interval(0.90, 0.97), ow.UNRESOLVED),
        (Interval(0.04, 0.09), Interval(0.89, 0.95), ow.UNRESOLVED),
    ],
)
def test_the_sharp_null_rule(rejection: Interval, coverage: Interval, expected: str) -> None:
    assert ow.null_label(rejection, coverage) == expected


@pytest.mark.parametrize(
    ("standardized", "power", "expected"),
    [
        (0.25, 0.0, ow.INERT),
        (0.30, 0.79, ow.UNDERPOWERED),
        (0.30, 0.80, ow.DISCRIMINABLE),
    ],
)
def test_the_population_rule(standardized: float, power: float, expected: str) -> None:
    assert ow.population_label(standardized, power) == expected


# ------------------------------------------------------------------ table-level mutations


def _ladder(
    sizes: tuple[int, ...], count: int, se_at_read: float, covered: int, rejected: int
) -> pd.DataFrame:
    """Ladder rows whose W arm at the read size has SE ratio ``se_at_read``, and whose W arm
    at the null's read size covers ``covered`` and rejects ``rejected`` of ``count``."""
    rng = np.random.default_rng(12)
    frames = []
    for arm in ow.ARMS:
        for n in sizes:
            draws = rng.normal(size=count)
            estimate = draws / draws.std(ddof=1) * 0.03
            ratio = (
                se_at_read if (arm, n) in {("W", ow.CALIBRATION_READ), ("W", ow.NULL_READ)} else 1.0
            )
            flags = np.arange(count)
            frames.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "n": n,
                        "estimate": estimate,
                        "std_error": 0.03 * ratio,
                        "covered": (flags < covered).astype(int),
                        "rejected": (flags < rejected).astype(int),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _result(rows: list[dict[str, object]], statistic: str) -> str:
    return str(next(row["result"] for row in rows if row["statistic"] == statistic))


def test_the_calibration_table_reads_w_at_32000(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ow, "CALIBRATION_SIZES", (ow.CALIBRATION_READ,))
    monkeypatch.setattr(ow, "CALIBRATION_REPLICATES", 8_000)
    rows = _ladder((ow.CALIBRATION_READ,), 8_000, 1.0, 7_600, 400)
    assert _result(ow.calibration_table(rows), "reading") == ow.CONTRACTING
    rows = _ladder((ow.CALIBRATION_READ,), 8_000, 0.94, 7_600, 400)
    assert _result(ow.calibration_table(rows), "reading") == ow.PERSISTENT
    assert BOUNDS["W"] > BOUNDS["U"]


def test_the_null_table_reads_w_at_4000(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ow, "NULL_SIZES", (ow.NULL_READ,))
    monkeypatch.setattr(ow, "NULL_REPLICATES", 3_200)
    assert (
        _result(ow.null_table(_ladder((ow.NULL_READ,), 3_200, 1.0, 3_040, 160)), "reading")
        == ow.WITHIN
    )
    assert (
        _result(ow.null_table(_ladder((ow.NULL_READ,), 3_200, 1.0, 3_040, 400)), "reading")
        == ow.OUTSIDE
    )
    # One row short of the budget turns the reading into a smoke reading.
    short = _ladder((ow.NULL_READ,), 3_199, 1.0, 3_040, 160)
    assert _result(ow.null_table(short), "reading") == shared.SMOKE


def _family(control_bias: float, count: int = 400) -> pd.DataFrame:
    """Family rows whose positive arms are unbiased and whose controls carry ``control_bias``
    in units of their own spread."""
    rng = np.random.default_rng(5)
    frames = []
    for label in weighted.CONTRASTS:
        noise = rng.normal(size=count)
        noise = (noise - noise.mean()) / noise.std(ddof=1) * 0.03
        for arm, shift in (("targeted", 0.0), ("untargeted", control_bias * 0.03)):
            frames.append(
                pd.DataFrame(
                    {
                        "property": ow.FAMILY,
                        "cell": f"{label}__{arm}",
                        "role": "positive" if arm == "targeted" else "control",
                        "replicate": np.arange(count),
                        "n": ow.FAMILY_N,
                        "requested_replicates": count,
                        "failed_replicates": 0,
                        "truth": 0.2,
                        "estimate": 0.2 + noise + shift,
                        "std_error": 0.03,
                        "covered": 1,
                        "rejected": 1,
                        "seed": np.arange(count),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_the_family_rule_and_a_mutation_of_the_control(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ow, "FAMILY_REPLICATES", 400)
    assert _result(ow.family_table(_family(1.0), -0.05), "fresh reading") == ow.RESOLVED
    # A control at a tenth of its spread no longer discriminates, so the family fails.
    assert _result(ow.family_table(_family(0.1), -0.05), "fresh reading") == ow.NOT_RESOLVED
    assert _result(ow.family_table(_family(1.0), -0.001), "population reading") == ow.INERT
    assert _result(ow.family_table(_family(1.0, 399), -0.05), "fresh reading") == shared.SMOKE


def test_the_supplementary_row_brackets_p_1200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ow, "FAMILY_REPLICATES", 400)
    rows = ow.family_table(_family(1.0), -0.011)
    point = next(row for row in rows if row["statistic"] == "p_1200")["value"]
    ends = next(row for row in rows if row["statistic"] == "p_1200 at the SD interval's ends")
    assert ends["result"] == "supplementary"
    assert ends["ci_lower"] < point < ends["ci_upper"]
    sd = next(row for row in rows if row["statistic"] == "fresh SD")
    assert sd["ci_lower"] < sd["value"] < sd["ci_upper"]


# --------------------------------------------------------------------------- seeds and R4


def test_the_parts_draw_their_declared_seeds() -> None:
    seeds = rm18_seeds.weighted_seeds()
    assert [p[-1] for p in ow.calibration_payloads()] == list(seeds["OW-A"])
    assert [p[-1] for p in ow.null_payloads()] == list(seeds["OW-B"])
    assert [p[2][5] for p in ow.family_payloads()] == list(seeds["OW-C"])
    assert [p[2][5] for p in ow.family_payloads(3)] == list(seeds["OW-C"][:3])


@pytest.fixture(scope="module")
def validations() -> dict[str, pd.DataFrame]:
    """One registered replicate of each part, validated once for the module."""
    return {part: ow.validate(part, 1, 1) for part in ow.PARTS}


@pytest.mark.parametrize("part", ow.PARTS)
def test_one_registered_replicate_of_each_part_reproduces(
    validations: dict[str, pd.DataFrame], part: str
) -> None:
    assert validations[part]["result"].eq(shared.HOLDS).all(), validations[part].to_string()


BUDGETS = {
    "OW-A": ("arm", "n", "CALIBRATION_REPLICATES", 6),
    "OW-B": ("arm", "n", "NULL_REPLICATES", 4),
    "OW-C": ("cell", None, "FAMILY_REPLICATES", 4),
}


@pytest.mark.parametrize("part", ow.PARTS)
def test_each_committed_reading_follows_from_its_rows(part: str) -> None:
    rows_path, _, reading_path = shared.part_paths(ow.HERE, part)
    if not reading_path.exists():
        pytest.skip(f"{part} has not run")
    first, second, budget, groups = BUDGETS[part]
    rows = shared.read_rows(rows_path)
    keys = [first] if second is None else [first, second]
    sizes = rows.groupby(keys).size()
    assert len(sizes) == groups and sizes.eq(getattr(ow, budget)).all()
    rebuilt = ow.table(part, ow.HERE, 1)
    pd.testing.assert_frame_equal(
        shared.as_committed(rebuilt), shared.read_rows(reading_path), check_dtype=False, rtol=1e-12
    )
    assert shared.SMOKE not in set(rebuilt["result"])
