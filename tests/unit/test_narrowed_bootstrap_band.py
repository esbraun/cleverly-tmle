"""A study effect's bootstrap band covers the rows it reports, and no others.

The longitudinal engines band every estimate they computed.  A contrast identified from a fit
that also computed the plan means kept the engine's band over all three rows, so ``summary()``
said "the bootstrap bands cover all 3 rows" under a table of one.  The study now narrows the
bootstrap band as it narrows the multiplier band: recomputed over the retained members from the
same replicates, absent below two members, and experimental when narrowed.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest

from cleverly import (
    CausalStudy,
    Inference,
    LongitudinalTreatment,
    ModelSpec,
    ParametricGFormulaMethod,
    RegimeContrast,
    RegimeMean,
    Runtime,
    study,
)
from cleverly.datasets import make_recovery_score
from cleverly.inference.bootstrap import bootstrap_band
from cleverly.learners import GLM

pytestmark = pytest.mark.filterwarnings("ignore::cleverly.MonteCarloWarning")

PLANS = {"never": (0, 0, 0), "always": (1, 1, 1)}
MEANS = ("ey_regimen[never]", "ey_regimen[always]")
GAIN = "ate_regimen[always vs never]"


@cache
def _study() -> CausalStudy:
    frame, _ = make_recovery_score(600, seed=3100)
    design = LongitudinalTreatment(
        outcome="recovery_day30",
        treatment=("offer_discharge", "offer_day10", "offer_day20"),
        baseline=("lives_alone", "discharge_risk"),
        time_varying=((), ("symptoms_day10",), ("symptoms_day20",)),
    )
    return CausalStudy(frame, design=design)


def _method() -> ParametricGFormulaMethod:
    return ParametricGFormulaMethod(
        models=ModelSpec(outcome_learner=GLM(family="gaussian"), treatment_learner=GLM()),
        covariate_learner=GLM(),
        n_draws=2,
        runtime=Runtime(random_state=1, n_jobs=1),
        inference=Inference(n_bootstrap=20),
    )


def _fit(estimand: Any) -> Any:
    return _study().identify(estimand).estimate(method=_method())


def test_a_one_row_contrast_has_no_bootstrap_band() -> None:
    result = _fit(RegimeContrast(PLANS, reference="never"))
    assert list(result.estimates) == [GAIN]
    assert result.bootstrap_band is None
    assert set(result.bootstrap.draws) == {GAIN}
    assert set(result.bootstrap.std_errors) == {GAIN}
    assert "bootstrap bands" not in str(result.summary())


def test_two_means_get_the_band_the_engine_builds_for_them_alone() -> None:
    result = _fit(RegimeMean(PLANS))
    band = result.bootstrap_band
    assert band is not None
    assert set(band.bands) == set(MEANS)
    assert band.support.label == "experimental"
    alone = bootstrap_band(
        result.bootstrap,
        {name: result[name].psi for name in MEANS},
        {name: result[name].plugin_std_error for name in MEANS},
        alpha=band.alpha,
    )
    assert band.critical_value == pytest.approx(alone.critical_value, rel=1e-12)
    for name in MEANS:
        assert band.bands[name] == pytest.approx(alone.bands[name], rel=1e-12)
    assert "cover all 2 rows" in " ".join(str(result.summary()).split())


def test_without_the_narrowing_the_band_counts_the_hidden_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the engine's own band names rows the contrast does not report."""
    monkeypatch.setattr(
        study, "_narrow_bootstrap_band", lambda result, retained: result.bootstrap_band
    )
    result = _fit(RegimeContrast(PLANS, reference="never"))
    assert result.bootstrap_band is not None
    assert set(result.bootstrap_band.bands) == {*MEANS, GAIN}
