"""The bootstrap docstrings of IPSI and the baselines say what the bootstrap publishes (X124).

They said the draws report "a bootstrap sd and no percentile inference", while a fit that
publishes its interval also publishes a percentile (or BCa) bootstrap interval under the
``experimental`` label, and a baseline fit that withholds its interval reports the bootstrap
as diagnostics. Each test pairs the behaviour with the docstring that describes it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from cleverly.datasets import make_longitudinal
from cleverly.interventions import Incremental
from cleverly.learners import GLM
from cleverly.longitudinal import IPSI
from cleverly.longitudinal.estimator import (
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.methods import IPSIMethod, SDRMethod
from tests.unit._inference_status_support import LONGITUDINAL_NODES, longitudinal_learners


@pytest.fixture(scope="module")
def panel() -> Any:
    return make_longitudinal(n=300, seed=0)[0]


def _fit(build: Any, panel: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build().fit(panel, outcome="Y", **LONGITUDINAL_NODES)


@pytest.mark.parametrize("interval", ["percentile", "bca"])
def test_an_inferential_baseline_publishes_an_experimental_bootstrap_interval(
    panel: Any, interval: str
) -> None:
    result = _fit(
        lambda: SequentialGComputation(
            {"always": 1, "never": 0},
            outcome_learner=GLM(family="binomial"),
            n_bootstrap=5,
            bootstrap_interval=interval,
            random_state=0,
            simultaneous=False,
        ),
        panel,
    )
    frame = result.to_frame()
    assert {"bootstrap_ci_lower", "bootstrap_ci_upper"} <= set(frame.columns)
    assert set(frame["bootstrap_interval"]) == {interval}
    assert set(frame["bootstrap_support"]) == {"experimental"}


def test_an_adaptive_baseline_reports_the_bootstrap_as_diagnostics(panel: Any) -> None:
    learners = longitudinal_learners(n_bootstrap=5)
    learners.pop("pseudo_learner")
    learners.pop("treatment_learner")
    frame = _fit(
        lambda: SequentialGComputation({"always": 1, "never": 0}, **learners), panel
    ).to_frame()
    assert "bootstrap_ci_lower" not in frame.columns
    assert {"bootstrap_sd", "bootstrap_range_lower"} <= set(frame.columns)


def test_ipsi_publishes_an_experimental_bootstrap_interval(panel: Any) -> None:
    frame = _fit(
        lambda: IPSI([Incremental(1.0), Incremental(2.0)], **longitudinal_learners(n_bootstrap=5)),
        panel,
    ).to_frame()
    assert {"bootstrap_ci_lower", "bootstrap_ci_upper"} <= set(frame.columns)
    assert set(frame["bootstrap_support"]) == {"experimental"}


@pytest.mark.parametrize(
    "owner",
    [IPSI, SequentialGComputation, SequentialIPW, ParametricGFormula, SDRMethod, IPSIMethod],
)
def test_no_docstring_denies_the_percentile_interval(owner: type) -> None:
    text = " ".join((owner.__doc__ or "").split())
    assert "no percentile" not in text
    assert "experimental" in text


def test_the_sdr_docstrings_name_the_rows_that_license_a_bootstrap_kind() -> None:
    """The SDR docstrings said no registered row lists a bootstrap kind for SDR.

    The ``G46.sdr`` rows license ``bootstrap_bca`` and no ``bootstrap_band``, so each
    docstring must name those rows and the BCa interval, and keep the band experimental.
    """
    from cleverly._support_rows import LONGITUDINAL_ROWS
    from cleverly.longitudinal import SDR

    rows = [row for row in LONGITUDINAL_ROWS if row.id.startswith("G46.sdr.")]
    assert any("bootstrap_bca" in row.kinds for row in rows)
    assert not any("bootstrap_band" in row.kinds for row in rows)
    for owner in (SDR, SDRMethod):
        text = " ".join((owner.__doc__ or "").split())
        assert "no registered row lists a bootstrap kind" not in text
        assert "G46.sdr" in text and "BCa" in text
        assert "every bootstrap band, carries the label ``experimental``" in text
