"""The replay audit of the stratified alternating targets.

``simulated_confounding`` replays a confounded copy of the data through a complete refit.  A
stratified incremental, linked-MSM, continuous-MSM or DR-TMLE fit targets one block per stratum,
and the refit targets the same blocks.  So the zero-strength cell of a stratum alias must be the
fit's own stratum estimate exactly, and every other cell must be the manual refit of its
replaced data.  ``DRTMLE`` runs at every guard and at ``guard=()``.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CrossFitting, DRTMLEMethod, IncrementalEffect, TMLEMethod
from cleverly.interventions import Incremental
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from tests.unit._confounding_support import (
    alias_for,
    confounding_estimate,
    confounding_study,
    replacement,
)
from tests.unit._simulated_confounding_support import _alias, _fit_msm
from tests.unit.test_simulated_confounding_msm import _fit_continuous

_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17))
_IN_SAMPLE = TMLEMethod(cross_fitting=CrossFitting(enabled=False))
_TILTS = (
    Incremental(0.5, name="down"),
    Incremental(1.0, name="natural"),
    Incremental(2.0, name="up"),
)


@cache
def _incremental() -> Any:
    return confounding_estimate(
        confounding_study(strata=True),
        IncrementalEffect(_TILTS, reference="natural"),
        method=_IN_SAMPLE,
    )


@cache
def _drtmle(guard: tuple[str, ...]) -> Any:
    return (
        confounding_study(strata=True, binary=True)
        .identify(ATE())
        .estimate(
            DRTMLEMethod(
                guard=guard,
                reduced_outcome_learner=LinearRegression(),
                reduced_treatment_learner=LogisticRegression(max_iter=1000),
            ),
            cross_fit=False,
            simultaneous=False,
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
        )
    )


def check_replay(result: Any, alias: str) -> None:
    """Zero strength replays the fit; every other cell is the manual refit."""
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    assert surface.cells[0].estimate == pytest.approx(result[alias].psi, abs=1e-12)
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=surface.refit_seed,
        )
        assert cell.failure is None
        assert cell.estimate == pytest.approx(manual[alias].psi, abs=1e-12)
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-5


def test_an_incremental_stratum_replays() -> None:
    result = _incremental()
    check_replay(result, alias_for(result, value="up", stratum=("small",)))


@pytest.mark.parametrize("link", ["log", "logit"])
def test_a_linked_msm_stratum_replays(link: str) -> None:
    result = _fit_msm(binary=True, link=link, strata=True)
    stratum = result.data.strata_levels[0]
    check_replay(result, _alias(result, stratum=stratum))


@pytest.mark.parametrize("link", ["identity", "logit"])
def test_a_continuous_msm_stratum_replays(link: str) -> None:
    result = _fit_continuous(link, strata=True, binary=link == "logit")
    stratum = result.data.strata_levels[0]
    check_replay(result, alias_for(result, stratum=stratum, coefficient="a"))


@pytest.mark.parametrize("guard", [("Q", "g"), ("Q",), ("g",), ()])
def test_a_drtmle_stratum_replays(guard: tuple[str, ...]) -> None:
    result = _drtmle(guard)
    check_replay(result, alias_for(result, target="ate", stratum=("small",)))
