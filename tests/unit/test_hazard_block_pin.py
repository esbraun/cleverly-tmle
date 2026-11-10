"""Hazard ``LTMLE`` values pinned at the commit before the overridable hazard block.

``SurvivalCTMLE`` overrides one block of the hazard fit, ``LTMLE._hazard_block``, which the
change that added it split out of the shared fit.  These numbers were computed at commit
``57024984``, the parent of that change, and this test holds every later commit to them, so a
refactor of the shared fit that moves a hazard ``LTMLE`` estimate fails here.  The wider
replay that checked the split at relative tolerance zero is::

    python -m tests.canonical.replay_committed replay \\
        --studies canonical-ltmle-survival,canonical-ltmle-competing,survival-node-bounds,\\
separable-effects,canonical-ctmle-selector --replicates 2 --jobs 4 --write DIR
    python -m tests.canonical.replay_committed compare BASE HEAD --rtol 0 --same-source

run once at ``57024984`` and once at the change.  The tolerance here is relative ``1e-9``, so
rounding differences between library versions pass and a real change fails.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly.longitudinal import LTMLE, LongitudinalData

#: ``configuration -> estimand -> (psi, standard error)`` at ``57024984``.
PINNED: dict[str, dict[str, tuple[float, float]]] = {
    "one_cause": {
        "risk_regimen[a1 @ t=3]": (0.39468467895193965, 0.038114370854315294),
        "risk_regimen[a0 @ t=3]": (0.5797347714227872, 0.036700498934978204),
        "ate_regimen[a1 vs a0 @ t=3]": (-0.18505009247084753, 0.051938904691713667),
    },
    "one_cause_crossfit": {
        "risk_regimen[a1 @ t=3]": (0.40058128736366755, 0.0408527717830427),
        "risk_regimen[a0 @ t=3]": (0.5786007807543353, 0.03817849177586778),
        "ate_regimen[a1 vs a0 @ t=3]": (-0.17801949339066775, 0.05514028620477745),
    },
    "competing": {
        "cif_regimen[a1, relapse @ t=3]": (0.21482032784089022, 0.031993153818685935),
        "cif_regimen[a1, death @ t=3]": (0.16896513774764949, 0.02926777115854468),
        "cif_regimen[a0, relapse @ t=3]": (0.2407234834114975, 0.0322041349853042),
        "cif_regimen[a0, death @ t=3]": (0.3322290972559918, 0.03469382109352245),
        "ate_regimen[a1 vs a0, relapse @ t=3]": (-0.025903155570607267, 0.04533763572673896),
        "ate_regimen[a1 vs a0, death @ t=3]": (-0.16326395950834233, 0.044926549243671295),
    },
    "one_step": {
        "risk_regimen[a1 @ t=3]": (0.3946015741530301, 0.03811311316969909),
        "risk_regimen[a0 @ t=3]": (0.5797266563336833, 0.03670422422730076),
        "ate_regimen[a1 vs a0 @ t=3]": (-0.18512508218065316, 0.05194219710270148),
    },
}

#: ``configuration -> (causes, n_folds, targeting)``.
CONFIGURATIONS = {
    "one_cause": (1, 1, "iterative"),
    "one_cause_crossfit": (1, 2, "iterative"),
    "competing": (2, 1, "iterative"),
    "one_step": (1, 1, "one_step"),
}


def _frame(causes: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, expit(0.4 * w[:, 0]))
    time = np.full(n, 3.0)
    event = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for k in (1, 2, 3):
        hit = alive & (rng.uniform(size=n) < expit(-1.2 + 0.5 * w[:, 1] - 0.5 * a))
        time[hit] = k
        event[hit] = rng.integers(1, causes + 1, size=hit.sum())
        alive &= ~hit
        if k < 3:
            gone = alive & (rng.uniform(size=n) < expit(-2.0 + 0.4 * w[:, 0]))
            time[gone] = k
            alive &= ~gone
    frame = pd.DataFrame(w, columns=["W1", "W2"])
    frame["A"], frame["time"], frame["event"] = a, time, event
    return frame


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_hazard_ltmle_matches_the_values_before_the_block_split(name: str) -> None:
    causes, n_folds, targeting = CONFIGURATIONS[name]
    extra = {"causes": {1: "relapse", 2: "death"}} if causes == 2 else {}
    data = LongitudinalData.from_time_to_event(
        _frame(causes), time="time", event="event", treatment="A", baseline=["W1", "W2"], **extra
    )
    learner = LogisticRegression(max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            {"a1": 1, "a0": 0},
            reference="a0",
            survival_model="hazard",
            targeting=targeting,
            outcome_learner=learner,
            treatment_learner=learner,
            censoring_learner=learner,
            n_folds=n_folds,
            random_state=0,
            simultaneous=False,
        ).fit(data)
    for estimand, (psi, std_error) in PINNED[name].items():
        estimate = result.estimates[estimand]
        assert float(estimate.psi) == pytest.approx(psi, rel=1e-9), estimand
        assert float(np.sqrt(estimate.variance)) == pytest.approx(std_error, rel=1e-9), estimand
