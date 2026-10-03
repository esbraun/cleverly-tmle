"""Bootstrap replicates read the point fit's resolved estimands.

An ``estimands="all"`` fit with ``delta=`` or ``intermediate=`` drops ``ey_obs``, ``par`` and
``paf`` in its preflight.  The replicate refit used to resolve ``"all"`` again, so every
replicate asked for the dropped names.  With missing outcomes each replicate raised the
population-intervention refusal, ``run_bootstrap`` counted it as a failure, and the fit
reported that all replicates failed.  The replicate now reads the point fit's tuple.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators import TMLE
from tests.conftest import fast_tmle


def _frame(n: int = 400, seed: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.4, size=n).astype(float)
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * w1)), size=n).astype(float)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.3 + 0.4 * a + 0.5 * w1 - 0.3 * w2))), size=n)
    delta = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.0 + 0.5 * w1 - 0.4 * a))), size=n)
    outcome = np.where(delta == 1, y.astype(float), np.nan)
    return pd.DataFrame({"Y": outcome, "A": a, "W1": w1, "W2": w2, "Delta": delta.astype(float)})


class TestTheBootstrapKeepsThePointFitsNames:
    """D7: a replicate reads the point fit's resolved estimands, not ``"all"`` again."""

    def test_an_all_fit_with_missing_outcomes_bootstraps_its_own_names(self) -> None:
        result = (
            fast_tmle(estimands="all", cross_fit=False, n_bootstrap=3, random_state=3)
            .fit(_frame(), outcome="Y", treatment="A", covariates=("W1", "W2"), delta="Delta")
            .single()
        )
        assert result.bootstrap is not None
        assert set(result.bootstrap.draws) == set(result.estimates)
        assert result.bootstrap.n_failed == 0

    def test_an_all_fit_beside_an_intermediate_bootstraps_its_own_names(self) -> None:
        frame = _frame().assign(
            Y=lambda f: f["Y"].fillna(0.0), Z=(np.arange(400) % 2).astype(float)
        )
        result = (
            fast_tmle(estimands="all", cross_fit=False, n_bootstrap=3, random_state=3)
            .fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2"), intermediate="Z")
            .get(0.0)
        )
        assert result is not None
        assert result.bootstrap is not None
        assert set(result.bootstrap.draws) == set(result.estimates)
        assert result.bootstrap.n_failed == 0

    def test_a_failing_replicate_is_still_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = TMLE._bootstrap_point_estimates
        calls = {"n": 0}

        def flaky(self: TMLE, *args: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("a replicate that fails for its own reason")
            return original(self, *args)

        monkeypatch.setattr(TMLE, "_bootstrap_point_estimates", flaky)
        with pytest.warns(UserWarning, match="1 of 3 bootstrap replicates failed"):
            result = (
                fast_tmle(estimands=("ey1", "ey0"), cross_fit=False, n_bootstrap=3, n_jobs=1)
                .fit(_frame(), outcome="Y", treatment="A", covariates=("W1", "W2"), delta="Delta")
                .single()
            )
        assert result.bootstrap.n_failed == 1
