"""The RM18 OW diagnostic: the exact-law witness, seeds, the three reading rules, the harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_ordinary_weighted import run as ow
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_weighted_ltmle import STUDY as ORDINARY
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
    weighted_values = shared.exact_untargeted("mechanism_correct")
    unweighted_values = shared.exact_untargeted("mechanism_correct", use_weights=False)
    for label in ("always", "never"):
        assert unweighted_values[label] == pytest.approx(
            shared.follower_mean(label, use_weights=False), abs=ow.EXACT_TOLERANCE
        )
        assert abs(unweighted_values[label] - shared.follower_mean(label)) > 1e-3
        assert abs(weighted_values[label] - unweighted_values[label]) > 1e-3


def test_b_inf_is_the_exact_untargeted_contrast_minus_its_truth() -> None:
    values = shared.exact_untargeted("mechanism_correct")
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


def _family(control_bias: float) -> pd.DataFrame:
    """Family rows whose positive arms are unbiased and whose controls carry ``control_bias``
    in units of their own spread."""
    rng = np.random.default_rng(5)
    frames = []
    for label in weighted.CONTRASTS:
        noise = rng.normal(size=400)
        noise = (noise - noise.mean()) / noise.std(ddof=1) * 0.03
        for arm, shift in (("targeted", 0.0), ("untargeted", control_bias * 0.03)):
            frames.append(
                pd.DataFrame(
                    {
                        "property": ow.FAMILY,
                        "cell": f"{label}__{arm}",
                        "role": "positive" if arm == "targeted" else "control",
                        "replicate": np.arange(400),
                        "n": ow.FAMILY_N,
                        "requested_replicates": 400,
                        "failed_replicates": 0,
                        "truth": 0.2,
                        "estimate": 0.2 + noise + shift,
                        "std_error": 0.03,
                        "covered": 1,
                        "rejected": 1,
                        "seed": np.arange(400),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _result(rows: list[dict[str, object]], statistic: str) -> str:
    return str(next(row["result"] for row in rows if row["statistic"] == statistic))


def test_the_family_rule_and_a_mutation_of_the_control() -> None:
    assert _result(ow.family_table(_family(1.0), -0.05), "fresh reading") == ow.RESOLVED
    # A control at a tenth of its spread no longer discriminates, so the family fails.
    assert _result(ow.family_table(_family(0.1), -0.05), "fresh reading") == ow.NOT_RESOLVED
    assert _result(ow.family_table(_family(1.0), -0.001), "population reading") == ow.INERT


def test_fresh_seeds_miss_every_registered_seed_and_each_other() -> None:
    registered = shared.weighted_registered_seeds(ORDINARY)
    parts = {
        "OW-A": [payload[-1] for payload in ow.calibration_payloads(ow.CALIBRATION_REPLICATES)],
        "OW-B": [payload[-1] for payload in ow.null_payloads(ow.NULL_REPLICATES)],
        "OW-C": [payload[5] for payload in ow.family_payloads(ow.FAMILY_REPLICATES)],
    }
    for seeds in parts.values():
        assert len(seeds) == len(set(seeds))
        assert set(seeds).isdisjoint(registered)
    # OW-A and OW-C draw the same selected law, so their draws must not repeat each other.
    assert set(parts["OW-A"]).isdisjoint(parts["OW-C"])


@pytest.mark.parametrize("part", ow.PARTS)
def test_one_registered_replicate_of_each_part_reproduces(part: str) -> None:
    validation = ow.validate(part, 1, 1)
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()


@pytest.mark.parametrize("part", ow.PARTS)
def test_each_committed_reading_follows_from_its_rows(part: str) -> None:
    rows_path, validation_path, reading_path = shared.part_paths(ow.HERE, part)
    if not reading_path.exists():
        pytest.skip(f"{part} has not run")
    rebuilt = ow.reading_table(
        part, shared.optional_rows(rows_path), shared.read_rows(validation_path)
    )
    pd.testing.assert_frame_equal(
        rebuilt, shared.read_rows(reading_path), check_dtype=False, rtol=1e-12
    )
