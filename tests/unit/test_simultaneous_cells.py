"""The shared joint-coverage row builder and its verdict rule."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cleverly.inference import simultaneous_bands
from cleverly.inference.influence import make_estimate
from tests.studies.evidence.property_verdicts import simultaneous_coverage_verdicts
from tests.studies.evidence.registry import Margins
from tests.studies.evidence.simultaneous import inference_scale, joint_coverage_rows

N = 400
POINTWISE = float(norm.ppf(0.975))


class _Result:
    def __init__(self, estimates: dict[str, object], bands: object) -> None:
        self._estimates = estimates
        self.simultaneous = bands

    def __getitem__(self, name: str) -> object:
        return self._estimates[name]


def _fit(truth_shift: float) -> tuple[_Result, dict[str, float], float]:
    rng = np.random.default_rng(20261002)
    curves = rng.standard_normal((N, 2)) @ np.array([[1.0, 0.3], [0.0, 1.0]])
    curves -= curves.mean(axis=0)
    estimates = {
        "ate": make_estimate("ate", 0.1, curves[:, 0], n=N),
        "rr": make_estimate("rr", 1.0, curves[:, 1] * 8.0, n=N, scale="ratio", log_psi=0.0),
    }
    bands = simultaneous_bands(estimates, random_state=0)
    critical = float(bands.critical_value)
    truth = {
        "ate": 0.1 + truth_shift * critical * estimates["ate"].std_error,
        "rr": math.exp(truth_shift * critical * estimates["rr"].std_error),
    }
    return _Result(estimates, bands), truth, critical


def _rows(result: _Result, truth: dict[str, float]) -> list[dict[str, object]]:
    return joint_coverage_rows(
        result,
        truth,
        ("ate", "rr"),
        label="pair",
        replicate=3,
        n=N,
        requested=10,
        pointwise_critical=POINTWISE,
    )


def test_a_truth_between_the_two_critical_values_is_covered_by_the_band_alone() -> None:
    result, truth, critical = _fit(0.95)
    assert critical > POINTWISE
    band, control = _rows(result, truth)
    assert band["cell"] == "pair__simultaneous_band" and band["role"] == "positive"
    assert control["cell"] == "pair__pointwise_joint_control" and control["role"] == "control"
    assert (band["covered"], band["rejected"]) == (1, 0)
    assert (control["covered"], control["rejected"]) == (0, 1)
    assert band["std_error"] == critical and control["std_error"] == POINTWISE
    assert band["truth"] == control["truth"] == 0.0
    assert band["estimate"] == pytest.approx(0.95 * critical)


def test_a_ratio_is_read_on_the_log_scale_its_band_is_built_on() -> None:
    """The scale witness: the ratio truth is inside the exponentiated band.

    The same truth is outside ``psi +- c * se`` on the natural scale.  A builder that read the
    ratio there would call this replication uncovered and report a larger statistic.
    """
    result, truth, critical = _fit(0.95)
    ratio = result["rr"]
    natural_upper = ratio.psi + critical * ratio.std_error
    assert truth["rr"] > natural_upper
    low, high = result.simultaneous.bands["rr"]
    assert low <= truth["rr"] <= high
    target, point = inference_scale(ratio, truth["rr"])
    assert (target, point) == (pytest.approx(math.log(truth["rr"])), 0.0)
    band, _ = _rows(result, truth)
    assert band["covered"] == 1


def test_a_truth_outside_the_band_is_covered_by_neither() -> None:
    result, truth, _ = _fit(1.05)
    band, control = _rows(result, truth)
    assert (band["covered"], control["covered"]) == (0, 0)


def test_a_band_over_another_family_is_refused() -> None:
    result, truth, _ = _fit(0.5)
    with pytest.raises(AssertionError, match="band covers"):
        joint_coverage_rows(
            result,
            truth,
            ("ate",),
            label="pair",
            replicate=0,
            n=N,
            requested=1,
            pointwise_critical=POINTWISE,
        )
    result.simultaneous = None
    with pytest.raises(AssertionError, match="no simultaneous band"):
        _rows(result, truth)


@pytest.mark.parametrize(
    ("role", "low", "high", "passed"),
    [
        ("positive", 0.93, 0.96, True),
        ("positive", 0.91, 0.94, False),
        ("positive", 0.96, 0.985, False),
        ("control", 0.80, 0.949, True),
        ("control", 0.90, 0.951, False),
    ],
)
def test_the_joint_verdict_reads_each_role_against_its_own_rule(
    role: str, low: float, high: float, passed: bool
) -> None:
    summary = pd.DataFrame(
        {
            "property": ["simultaneous_coverage"],
            "role": [role],
            "coverage_ci_lower": [low],
            "coverage_ci_upper": [high],
            "passed": [None],
        }
    )
    simultaneous_coverage_verdicts(summary, margins=Margins())
    assert bool(summary.loc[0, "passed"]) is passed


def test_the_cross_fitted_competing_study_fits_no_band() -> None:
    """The shared generator adds joint cells for the ordinary competing study alone.

    The cross-fitted study passes ``joint_cells=False``, so its fit function is still called
    with the frame and the configuration only, and it declares no joint cell.
    """
    from tests.studies import canonical_ltmle_competing, ltmle_competing_properties

    assert "simultaneous_coverage" not in canonical_ltmle_competing.property_cells(crossfit=True)
    assert "simultaneous_coverage" in canonical_ltmle_competing.property_cells(crossfit=False)
    calls: list[dict[str, object]] = []

    def spy(frame: pd.DataFrame, configuration: str, **options: object) -> object:
        calls.append(options)
        return ltmle_competing_properties.fit(frame, configuration, n_folds=5)

    payload = ("interval_calibration", "correctly_specified", 0, 4_000, 1, 7, "both_correct")
    rows = ltmle_competing_properties._fit_replication(
        payload, study=ltmle_competing_properties.STUDY, fit_fn=spy
    )
    assert calls == [{}]
    assert {row["property"] for row in rows} == {"interval_calibration"}


def test_the_multi_arm_cells_keep_their_integer_roots() -> None:
    """The joint pair is appended on its own stream; the sampled cells are unchanged."""
    from tests.studies import multi_arm_tmle_properties

    sampled = multi_arm_tmle_properties.cells()
    declared = multi_arm_tmle_properties.declared_cells()

    def key(cell: object) -> tuple[object, ...]:
        return tuple(
            getattr(cell, name)
            for name in ("property", "cell", "n", "replicates", "seed", "role", "estimand")
        )

    assert [key(cell) for cell in declared[: len(sampled)]] == [key(cell) for cell in sampled]
    joint = declared[len(sampled) :]
    assert [cell.cell for cell in joint] == [
        "arms__simultaneous_band",
        "arms__pointwise_joint_control",
    ]
    assert {cell.seed for cell in joint}.isdisjoint({cell.seed for cell in sampled})
    batches = multi_arm_tmle_properties.sampling_batches()
    assert [key(cell) for cell in batches[0].cells] == [key(cell) for cell in sampled]
