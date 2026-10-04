"""Continuous-dose working models with baseline strata, against the expanded-design oracle.

A continuous dose has no arms, so the stratum blocks read no arm share.  Each block is the
density-ratio covariate of the marginal working model, ``I(V = s) h phi / g(a | W)`` on the
observed dose and on every grid dose, divided by ``P_n(V = s)``.  Under the identity link the
blocks are solved in the one fluctuation; under the logit link in the nested stratum
fluctuation.

The oracle is the unstratified fit of the working model with the expanded design
``(I(V = s) phi)_s``.  The identity link reads a linear dose response whose slope changes
with ``V``; the logit link reads a binary outcome whose dose response is not linear on any
scale the model declares.  A marginal column in place of the blocks must fail the comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.estimators import TMLE
from cleverly.msm import MSM
from tests.studies import stratified_law as law
from tests.unit._declaration_support import assert_every_witness_fails, tmle_module
from tests.unit._stratified_alternating_support import (
    GRID,
    expanded_msm,
    frame,
    logistic,
    marginal_blocks,
)

TERMS = ("(intercept)", "a")
#: The outcome column each link reads: a linear dose response, and a binary outcome.
OUTCOMES = {"identity": "Yc", "logit": "Yd"}


def model(link: str) -> MSM:
    return MSM.linear(doses=GRID, link=link)  # type: ignore[arg-type]


def fit(rows: pd.DataFrame, msm: MSM, link: str, *, strata: bool = True) -> Any:
    roles: dict[str, Any] = {"strata": ["V"]} if strata else {}
    outcome_learner = LinearRegression() if link == "identity" else logistic()
    return (
        TMLE(
            msm=msm,
            outcome_learner=outcome_learner,
            treatment_learner=logistic(),
            density_bins=6,
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            rows,
            outcome=OUTCOMES[link],
            treatment="D",
            treatment_kind="continuous",
            covariates=["W", "V"],
            **roles,
        )
        .single()
    )


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return frame(2_000)


@pytest.fixture(scope="module", params=tuple(OUTCOMES))
def fits(request: pytest.FixtureRequest, rows: pd.DataFrame) -> dict[str, Any]:
    link = request.param
    return {
        "link": link,
        "stratified": fit(rows, model(link), link),
        "oracle": fit(rows, expanded_msm(model(link)), link, strata=False),
    }


def check_oracle(stratified: Any, oracle: Any) -> None:
    for s in law.STRATA:
        for term in TERMS:
            estimate = stratified[f"msm[{term}][V={s}]"]
            expected = oracle[f"msm[{term}|{s}]"]
            assert estimate.psi == pytest.approx(expected.psi, abs=1e-8), (term, s)
            np.testing.assert_allclose(
                estimate.influence_curve, expected.influence_curve, rtol=0.0, atol=1e-6
            )


def test_the_slopes_differ_by_stratum(fits: dict[str, Any]) -> None:
    """The nonzero witness: ``V`` modifies the dose response."""
    slopes = [fits["stratified"][f"msm[a][V={s}]"].psi for s in law.STRATA]
    assert np.ptp(slopes) > 0.05, slopes


def test_the_stratum_coefficients_are_the_expanded_design_ones(fits: dict[str, Any]) -> None:
    check_oracle(fits["stratified"], fits["oracle"])


def test_every_block_is_solved(fits: dict[str, Any]) -> None:
    fluctuation = fits["stratified"].fluctuations["msm"]
    solved = fluctuation if fits["link"] == "identity" else fluctuation.stratified
    assert solved is not None
    assert solved.epsilon.shape == (len(law.STRATA) * len(TERMS),)
    assert solved.relative_score_norm < 1e-9


def test_a_marginal_column_fails(
    fits: dict[str, Any], rows: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tmle_module, "stratify", marginal_blocks)
    mutated = fit(rows, model(fits["link"]), fits["link"])
    assert_every_witness_fails([lambda: check_oracle(mutated, fits["oracle"])])
