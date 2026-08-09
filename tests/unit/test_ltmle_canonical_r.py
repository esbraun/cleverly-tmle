r"""A bounded, nonzero witness against the canonical R ``ltmle`` implementation.

The exact finite-support laws and Gateaux checks remain the scientific oracles for the
parameter and EIF.  They deliberately land at ``epsilon = 0`` and use nonbinding bounds,
however, so they cannot distinguish two finite-sample targeting paths with the same score
or two truncation conventions that agree away from the boundary.  This fixture fills only
that gap: R ``ltmle`` 1.3-0 and cleverly receive identical intercept-only Q regressions,
fixed g predictions, and ``gbounds=(0.2, 0.99)``.  The second prefix binds for one baseline
stratum and the deepest fluctuation is nonzero in both the end-of-study and survival fits.

Regenerate the frozen CSVs with ``tests/canonical/ltmle/generate_reference.R`` in the
pinned container described beside it.  Cross-language agreement is secondary acceptance
evidence, never a substitute for the independent exact-law derivations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.fluctuation.submodel import Submodel
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LongitudinalData, resolve_plans, resolve_regimens
from cleverly.longitudinal import sequential as longitudinal_sequential
from cleverly.longitudinal.sequential import Mechanism, RegimenFit, fit_regimen
from cleverly.utils.bounds import OutcomeScaler

FIXTURES = Path(__file__).parents[1] / "canonical" / "ltmle"
BOUNDS = (0.2, 0.99)
LABEL = "always"


def _mechanism(frame: pd.DataFrame) -> Mechanism:
    def values(name: str) -> dict[str, np.ndarray]:
        return {LABEL: frame[name].to_numpy(dtype=float)}

    return Mechanism(
        treatment=(values("g_A1"), values("g_A2")),
        censoring=(values("g_C1"), values("g_C2")),
    )


def _fit(variant: str) -> tuple[RegimenFit, pd.DataFrame]:
    frame = pd.read_csv(FIXTURES / f"{variant}.csv")
    outcome: str | list[str]
    time_varying: list[list[str]]
    if variant == "longitudinal":
        outcome = "Y"
        time_varying = [[], ["L2"]]
    else:
        # Both APIs store the absorbing event cumulatively: once Y1 is one, Y2 stays one.
        outcome = ["Y1", "Y2"]
        time_varying = [[], []]
    data = LongitudinalData.from_frame(
        frame,
        outcome=outcome,
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=time_varying,
        censoring=["C1", "C2"],
    )
    plan = resolve_plans(resolve_regimens({LABEL: 1}, data.n_times), data)[0]
    fit = fit_regimen(
        data,
        plan,
        _mechanism(frame),
        outcome_learner=DummyClassifier(strategy="prior"),
        pseudo_learner=DummyRegressor(strategy="mean"),
        folds=Folds.single(data.n),
        scaler=OutcomeScaler.identity(),
        g_bounds=BOUNDS,
        horizon=data.n_times,
        max_iter=100,
        tol=1e-12,
    )
    return fit, frame


@pytest.mark.parametrize("variant", ["longitudinal", "survival"])
def test_plain_ltmle_matches_the_frozen_r_reference(variant: str) -> None:
    fit, _ = _fit(variant)
    reference = pd.read_csv(FIXTURES / "reference.csv").query("variant == @variant")

    # Canonical R's glm update stops at its 1e-8 IRLS criterion; cleverly's direct
    # Newton solve is tighter, so the frozen comparison is at the resulting 1e-9 scale.
    assert fit.psi_scaled == pytest.approx(reference["estimate"].iloc[0], abs=1e-9)
    np.testing.assert_allclose(
        fit.influence_curve_scaled,
        reference["influence_curve"].to_numpy(),
        rtol=0,
        atol=3e-9,
    )
    np.testing.assert_allclose(fit.cumulative[:, 0], reference["cumulative_g_t1"], atol=1e-15)
    # Canonical R freezes cumulative g after an event; cleverly can keep evaluating the
    # supplied numeric mechanism, but the second value is used only on this at-risk set.
    at_risk_t2 = fit.steps[1].at_risk
    np.testing.assert_allclose(
        fit.cumulative[at_risk_t2, 1],
        reference.loc[at_risk_t2, "cumulative_g_t2"],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        [step.fluctuation.epsilon[0] for step in fit.steps],
        reference[["epsilon_t1", "epsilon_t2"]].iloc[0].to_numpy(),
        rtol=0,
        atol=3e-9,
    )


@pytest.mark.parametrize("variant", ["longitudinal", "survival"])
def test_the_reference_exercises_both_corrected_algorithmic_choices(variant: str) -> None:
    fit, frame = _fit(variant)
    raw_first = frame["g_A1"].to_numpy() * frame["g_C1"].to_numpy()
    raw_second = raw_first * frame["g_A2"].to_numpy() * frame["g_C2"].to_numpy()
    old_factorwise_second = (
        np.clip(frame["g_A1"], *BOUNDS)
        * np.clip(frame["g_C1"], *BOUNDS)
        * np.clip(frame["g_A2"], *BOUNDS)
        * np.clip(frame["g_C2"], *BOUNDS)
    )

    np.testing.assert_allclose(fit.cumulative[:, 0], np.clip(raw_first, *BOUNDS))
    np.testing.assert_allclose(fit.cumulative[:, 1], np.clip(raw_second, *BOUNDS))
    assert np.max(np.abs(fit.cumulative[:, 1] - old_factorwise_second)) > 0.03
    assert abs(fit.steps[-1].fluctuation.epsilon[0]) > 0.4


@pytest.mark.parametrize("variant", ["longitudinal", "survival"])
def test_putting_inverse_probability_back_in_the_submodel_fails_the_reference(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate mutation: same score equation, wrong finite-sample path."""
    original = longitudinal_sequential.solve_fluctuation

    def old_path(outcome, initial, submodel, weights, observed, *args, **kwargs):  # type: ignore[no-untyped-def]
        inverse_probability = np.asarray(weights, dtype=float)
        mutated = Submodel(
            inverse_probability[:, None],
            {0.0: inverse_probability[:, None]},
            submodel.names,
            submodel.group,
        )
        return original(
            outcome,
            initial,
            mutated,
            np.ones_like(inverse_probability),
            observed,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(longitudinal_sequential, "solve_fluctuation", old_path)
    mutated, _ = _fit(variant)
    reference = pd.read_csv(FIXTURES / "reference.csv").query("variant == @variant")
    assert abs(mutated.psi_scaled - reference["estimate"].iloc[0]) > 1e-3
