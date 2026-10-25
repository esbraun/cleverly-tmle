"""The simultaneous band of an in-sample LTMLE fit on a repeated outcome, and of its narrowing.

Row G27 licenses the band: the repeated-outcome study gates the default band of its in-sample
calibration fits over every parameter its law reports (``simultaneous_coverage/all_reported``).
A ``CausalStudy`` band narrowed to the parameters an effect reports is a different family, which
no study measures, so it stays experimental and says why.
"""

from __future__ import annotations

import warnings

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    ModelSpec,
    RegimeContrast,
    Repeated,
    Runtime,
    TMLEMethod,
)
from cleverly._support_rows import LONGITUDINAL_ROWS
from cleverly.datasets import make_weekly_orders
from cleverly.longitudinal import LTMLE

WEEKS = (1, 2, 3)


def _row(row_id: str):
    return next(row for row in LONGITUDINAL_ROWS if row.id == row_id)


def test_the_repeated_outcome_row_licenses_the_band_its_study_gates() -> None:
    row = _row("G27")
    assert "band" in row.kinds
    assert "all_reported" in row.cells


@pytest.fixture(scope="module")
def frame():
    data, _ = make_weekly_orders(3_000, seed=11)
    return data


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def test_the_engine_band_over_every_parameter_is_supported(frame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            {"never": 0, "always": 1},
            reference="never",
            outcome_learner=_logistic(),
            pseudo_learner=LinearRegression(),
            treatment_learner=_logistic(),
            censoring_learner=_logistic(),
            n_folds=1,
            simultaneous=True,
            random_state=0,
        ).fit(
            frame,
            outcome=Repeated([f"order_{week}" for week in WEEKS]),
            treatment=[f"offer_{week}" for week in WEEKS],
            baseline=["tenure", "order_0"],
            time_varying=[[f"engaged_{week}"] for week in WEEKS],
            censoring=[f"tracked_{week}" for week in WEEKS],
        )
    assert {estimate.support.row for estimate in result.estimates.values()} == {"G27"}
    assert result.simultaneous.support.label == "supported"


def test_a_band_narrowed_to_the_contrasts_is_experimental_and_says_why(frame) -> None:
    design = LongitudinalTreatment(
        outcome=Repeated([f"order_{week}" for week in WEEKS]),
        treatment=tuple(f"offer_{week}" for week in WEEKS),
        baseline=("tenure", "order_0"),
        time_varying=tuple((f"engaged_{week}",) for week in WEEKS),
        censoring=tuple(f"tracked_{week}" for week in WEEKS),
    )
    models = ModelSpec(
        outcome_learner=_logistic(),
        pseudo_learner=LinearRegression(),
        treatment_learner=_logistic(),
        censoring_learner=_logistic(),
    )
    method = TMLEMethod(
        models=models, cross_fitting=CrossFitting(enabled=False), runtime=Runtime(random_state=0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            CausalStudy(frame, design=design)
            .identify(RegimeContrast({"never": (0, 0, 0), "always": (1, 1, 1)}, reference="never"))
            .estimate(method=method)
        )
    assert {estimate.support.label for estimate in result.estimates.values()} == {"supported"}
    band = result.simultaneous.support
    assert band.label == "experimental"
    assert "subset of the parameters" in band.reason
