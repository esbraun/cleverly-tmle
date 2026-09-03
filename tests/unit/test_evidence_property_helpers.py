"""Focused contracts for shared finite-law and property-row helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law as point
from tests import discrete_law_cde as cde
from tests import discrete_law_mar as mar
from tests.studies import canonical_cde_tmle
from tests.studies.cde_study_helpers import sample_discrete as sample_cde
from tests.studies.evidence import descriptions
from tests.studies.evidence.properties import (
    ReplicationSpec,
    paired_spread_ratio_interval,
    property_role,
    replication_payloads,
)
from tests.studies.intervention_study_helpers import sample_discrete as sample_point
from tests.studies.missing_outcome_study_helpers import sample_discrete as sample_mar


def test_each_law_adapter_preserves_its_public_sample_schema() -> None:
    point_frame = sample_point(point.PROBS, 20, 17)
    mar_frame = sample_mar(mar.PROBS, 20, 17)
    cde_frame = sample_cde(cde.PROBS, 20, 17)

    assert list(point_frame) == ["W", "A", "Y"]
    assert list(mar_frame) == ["W", "A", "Y", "Delta"]
    assert list(cde_frame) == ["W", "A", "Z", "Y", "Delta"]
    assert mar_frame["Y"].isna().equals(mar_frame["Delta"].eq(0.0))
    assert cde_frame["Y"].isna().equals(cde_frame["Delta"].eq(0.0))


def test_property_payloads_keep_seeded_tuple_shape_and_role_rules() -> None:
    specs = [ReplicationSpec("family", "cell", 100, 2, "wrong")]
    payloads = replication_payloads(canonical_cde_tmle.STUDY, specs)

    assert len(payloads) == 2
    assert [payload[0][2] for payload in payloads] == [0, 1]
    assert payloads[0][0][:2] == ("family", "cell")
    assert payloads[0][0][3:5] == (100, 2)
    assert payloads[0][0][-1] == "wrong"
    assert payloads[0][0][5] != payloads[1][0][5]
    assert (
        property_role(
            "wrong",
            controls={"wrong"},
            property_name="family",
            n=100,
            rate_sizes=(100, 400),
        )
        == "control"
    )
    assert (
        property_role(
            "correct",
            controls={"wrong"},
            property_name="root_n_and_efficiency",
            n=100,
            rate_sizes=(100, 400),
        )
        == "control"
    )


def test_calibration_description_uses_declared_nuisance_count() -> None:
    two = descriptions.cell("interval_calibration", "correctly_specified", nuisance_count=2)[0]
    three = descriptions.cell("interval_calibration", "correctly_specified", nuisance_count=3)[0]
    four = descriptions.cell("interval_calibration", "z0__correctly_specified", nuisance_count=4)[0]

    assert two == "both nuisances are correctly specified"
    assert three == "all three required nuisance functions are correctly specified"
    assert four.endswith("all four required nuisance functions are correctly specified")


def test_shared_sampler_keeps_the_declared_draw_deterministic() -> None:
    first = sample_cde(cde.PROBS, 50, 81)
    second = sample_cde(cde.PROBS, 50, 81)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first[["W", "A", "Z", "Delta"]].to_numpy()).all()


def test_paired_spread_ratio_uses_one_shared_bootstrap_index_matrix() -> None:
    keys = np.arange(20)
    denominator = pd.DataFrame({"replicate": keys, "estimate": np.linspace(-2.0, 3.0, 20)})
    numerator = pd.DataFrame(
        {"replicate": keys, "estimate": 0.4 * denominator["estimate"] + np.sin(keys)}
    )

    result = paired_spread_ratio_interval(
        numerator,
        denominator,
        replicates=500,
        confidence_level=0.99,
        seed=81,
    )

    values = np.column_stack([numerator["estimate"], denominator["estimate"]])
    picks = np.random.default_rng(81).integers(0, len(values), size=(500, len(values)))
    expected = values[picks].std(axis=1, ddof=1)
    expected_ratios = expected[:, 0] / expected[:, 1]
    assert result.ratio == values[:, 0].std(ddof=1) / values[:, 1].std(ddof=1)
    assert result.interval.low == np.quantile(expected_ratios, 0.005)
    assert result.interval.high == np.quantile(expected_ratios, 0.995)


def test_paired_spread_ratio_refuses_unpaired_duplicate_and_degenerate_rows() -> None:
    subject = pd.DataFrame({"replicate": [0, 1, 2], "estimate": [0.0, 1.0, 2.0]})
    kwargs = {"replicates": 20, "confidence_level": 0.99, "seed": 17}

    with np.testing.assert_raises_regex(ValueError, "not paired"):
        paired_spread_ratio_interval(
            subject,
            pd.DataFrame({"replicate": [0, 1, 3], "estimate": [0.0, 1.0, 2.0]}),
            **kwargs,
        )
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        paired_spread_ratio_interval(
            subject,
            pd.DataFrame({"replicate": [0, 0, 2], "estimate": [0.0, 1.0, 2.0]}),
            **kwargs,
        )
    with np.testing.assert_raises_regex(ValueError, "denominator spread is zero"):
        paired_spread_ratio_interval(
            subject,
            pd.DataFrame({"replicate": [0, 1, 2], "estimate": [1.0, 1.0, 1.0]}),
            **kwargs,
        )


def test_paired_spread_ratio_refuses_a_zero_denominator_bootstrap_draw() -> None:
    subject = pd.DataFrame({"replicate": [0, 1, 2], "estimate": [0.0, 2.0, 1.0]})
    denominator = pd.DataFrame({"replicate": [0, 1, 2], "estimate": [0.0, 1.0, 2.0]})

    with np.testing.assert_raises_regex(ValueError, "bootstrap draw has zero denominator"):
        paired_spread_ratio_interval(
            subject,
            denominator,
            replicates=20,
            confidence_level=0.99,
            seed=17,
        )


@pytest.mark.parametrize("floor", [0.9, 0.925])
def test_contraction_coverage_boundaries_and_unrelated_rows(floor) -> None:
    from dataclasses import replace

    from tests.studies.evidence.property_verdicts import contraction_verdicts

    record = replace(
        canonical_cde_tmle.STUDY,
        margins=replace(canonical_cde_tmle.STUDY.margins, coverage_floor=floor),
    )
    below, above = np.nextafter(floor, 0.0), np.nextafter(floor, 1.0)
    summary = pd.DataFrame(
        {
            "property": ["double_robust_contraction"] * 6 + ["other"],
            "role": ["positive"] * 3 + ["control"] * 3 + ["positive"],
            "coverage_ci_lower": [below, floor, above] * 2 + [0.0],
            "coverage_ci_upper": [below, floor, above] * 2 + [0.0],
            "passed": pd.Series([None] * 6 + ["unchanged"], dtype=object),
        }
    )
    contraction_verdicts(summary, record)
    assert summary["passed"].tolist() == [False, True, True, True, False, False, "unchanged"]


@pytest.mark.parametrize("upper", [-1e-12, 0.0, 1e-12])
def test_contraction_rates_keep_strict_direction_control_inversion_and_streams(
    monkeypatch, upper
) -> None:
    from tests.studies.evidence import property_verdicts
    from tests.studies.evidence.inference import Interval
    from tests.studies.evidence.properties import Rate
    from tests.studies.evidence.seeds import stream_seed

    record = canonical_cde_tmle.STUDY
    scenarios = ("both_wrong", "treatment_correct", "outcome_correct")
    rows = pd.DataFrame(
        [
            {"property": "double_robust_contraction", "cell": f"{scenario}_n{n}", "n": n}
            for scenario in scenarios
            for n in (20, 40, 80)
        ]
        + [{"property": "other", "cell": "both_wrong_n999", "n": 999}]
    )
    calls = []

    def fitted(selected, **kwargs):
        calls.append((selected.copy(), kwargs))
        return Rate(-0.5, Interval(-1.0, upper))

    monkeypatch.setattr(property_verdicts, "rate", fitted)
    result = property_verdicts.contraction_rates(
        rows, record, ["extra_column"], scenarios=scenarios
    )
    assert [row["cell"] for row in result] == [f"rate_{scenario}" for scenario in scenarios]
    assert [row["role"] for row in result] == ["control", "positive", "positive"]
    assert [row["passed"] for row in result] == [upper >= 0.0, upper < 0.0, upper < 0.0]
    for scenario, row, (selected, arguments) in zip(scenarios, result, calls, strict=True):
        assert selected["cell"].tolist() == [f"{scenario}_n{n}" for n in (20, 40, 80)]
        assert arguments == {
            "property_name": "double_robust_contraction",
            "statistic": "bias",
            "bootstrap_replicates": record.margins.bootstrap_replicates,
            "confidence_level": record.margins.confidence_level,
            "seed": stream_seed(record, "double_robust_contraction", scenario),
        }
        assert row["rate_sizes"] == "20;40;80"
        assert row["n"] == 80
        assert row["replicates"] == 3
        assert np.isnan(row["extra_column"])
        assert row["slope_ci_upper"] == upper
