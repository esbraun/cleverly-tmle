"""The natural-course mean with missing outcomes and baseline strata, against its exact form.

A stratum parameter is ``E[Y | V = s]`` under outcomes missing at random given ``(A, W)``.
Inside the stratum its efficient influence function is

    D_s = Delta / pi(A, W) {Y - m(A, W)} + m(A, W) - psi_s,

with ``m`` the outcome regression and ``pi`` the observation mechanism, and its full-law
gradient is ``I(V = s) D_s / P(V = s)``.  The fit solves one natural-course block per stratum
in sample.  The nuisances are fixed functions, the outcome one deliberately wrong, so every
block moves.

The tests check the blocks, the curves at the targeted regression, the subset fits, the
marginal mixture, and the joint route that stacks the natural course with arm means for PAR.
Three mutations must fail: one marginal column, no ``P_n(V = s)`` in the un-scaling, and no
``n / n_s`` in the embedding.  The cross-fitted natural-course mean refuses strata before any
learner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from tests.studies import stratified_law as law
from tests.unit._declaration_support import (
    assert_every_witness_fails,
    assert_refused_before_any_call,
    tmle_module,
)
from tests.unit._natural_course_support import never_fit_learners
from tests.unit._stratified_alternating_support import marginal_blocks
from tests.unit.test_stratified_drtmle_exact import (
    ArmOutcome,
    FixedResponse,
    WOnlyTreatment,
    rows,
)

N = 2_000
SEED = 20261105


def fit(
    frame: pd.DataFrame,
    estimands: tuple[str, ...] = ("ey_obs",),
    *,
    stratum: int | None = None,
) -> Any:
    """The stratified fit, or with ``stratum`` the unstratified fit of that stratum's rows."""
    roles: dict[str, Any] = {} if stratum is not None else {"strata": ["V"]}
    return (
        TMLE(
            estimands=estimands,
            outcome_learner=ArmOutcome(2, stratum),
            treatment_learner=WOnlyTreatment(2),
            missingness_learner=FixedResponse(stratum=stratum),
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Yobs", treatment="A", covariates=["W", "V"], delta="Delta", **roles)
        .single()
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return rows(N, SEED)


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return fit(frame)


def observation(frame: pd.DataFrame) -> np.ndarray:
    """``pi(A, W)``, the law's observation mechanism at each row."""
    a, w, v = (frame[name].to_numpy(dtype=float) for name in ("A", "W", "V"))
    return np.asarray(expit(1.1 - 0.3 * (a > 0) + 0.2 * w - 0.25 * v))


def pieces(result: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``m*(A, W)``, the residual weight ``Delta / pi`` and the residual ``Y - m*``."""
    targeted = np.asarray(result.fluctuations["natural_course"].targeted.observed, dtype=float)
    delta = frame["Delta"].to_numpy(dtype=float)
    y = np.nan_to_num(frame["Yobs"].to_numpy(dtype=float))
    return targeted, delta / observation(frame), np.where(delta == 1.0, y - targeted, 0.0)


def check_blocks(result: Any, frame: pd.DataFrame) -> None:
    _, weight, residual = pieces(result, frame)
    v = frame["V"].to_numpy()
    for s in law.STRATA:
        assert abs(float(np.mean((v == s) * weight * residual))) < 1e-10, s


def check_curves(
    result: Any, frame: pd.DataFrame, *, unscale: bool = True, embed: bool = True
) -> None:
    m, weight, residual = pieces(result, frame)
    v = frame["V"].to_numpy()
    for s in law.STRATA:
        inside = v == s
        mass = float(inside.mean())
        psi = float(m[inside].mean())
        h = weight if unscale else weight / mass
        curve = np.where(inside, (h * residual + m - psi) / (mass if embed else 1.0), 0.0)
        estimate = result[f"ey_obs[V={s}]"]
        assert estimate.psi == pytest.approx(psi, abs=1e-12)
        np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=1e-10)


class TestTheStratumNaturalCourse:
    def test_every_block_moved(self, fitted: Any) -> None:
        epsilon = np.asarray(fitted.fluctuations["natural_course"].epsilon)
        assert epsilon.shape == (len(law.STRATA),)
        assert np.min(np.abs(epsilon)) > 1e-3, epsilon

    def test_each_block_is_zero(self, fitted: Any, frame: pd.DataFrame) -> None:
        check_blocks(fitted, frame)

    def test_each_curve_is_the_analytic_one(self, fitted: Any, frame: pd.DataFrame) -> None:
        check_curves(fitted, frame)

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_each_stratum_is_its_subset_fit(
        self, fitted: Any, frame: pd.DataFrame, stratum: int
    ) -> None:
        alone = fit(frame[frame["V"] == stratum].reset_index(drop=True), stratum=stratum)
        inside = frame["V"].to_numpy() == stratum
        pooled = fitted[f"ey_obs[V={stratum}]"]
        assert pooled.psi == pytest.approx(alone["ey_obs"].psi, abs=1e-10)
        np.testing.assert_allclose(
            pooled.influence_curve[inside] * float(inside.mean()),
            alone["ey_obs"].influence_curve,
            rtol=0.0,
            atol=1e-9,
        )

    def test_the_marginal_is_the_mixture(self, fitted: Any, frame: pd.DataFrame) -> None:
        v = frame["V"].to_numpy()
        mixture = sum(float(np.mean(v == s)) * fitted[f"ey_obs[V={s}]"].psi for s in law.STRATA)
        assert fitted["ey_obs"].psi == pytest.approx(mixture, abs=1e-12)


class TestTheJointRoute:
    @pytest.fixture(scope="class")
    def joint(self, frame: pd.DataFrame) -> Any:
        return fit(frame, ("ey", "par", "ey_obs"))

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_the_stratum_natural_course_is_the_scalar_fits(
        self, joint: Any, fitted: Any, stratum: int
    ) -> None:
        name = f"ey_obs[V={stratum}]"
        assert joint[name].psi == pytest.approx(fitted[name].psi, abs=1e-12)
        np.testing.assert_allclose(
            joint[name].influence_curve, fitted[name].influence_curve, rtol=0.0, atol=1e-12
        )

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_the_stratum_par_is_the_difference(self, joint: Any, stratum: int) -> None:
        par = joint[f"par[V={stratum}]"]
        observed = joint[f"ey_obs[V={stratum}]"]
        reference = joint[f"ey[0][V={stratum}]"]
        assert par.psi == pytest.approx(observed.psi - reference.psi, abs=1e-14)
        np.testing.assert_allclose(
            par.influence_curve,
            observed.influence_curve - reference.influence_curve,
            rtol=0.0,
            atol=1e-12,
        )


class TestTheMutationsFail:
    def test_a_marginal_column(self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tmle_module, "stratify", marginal_blocks)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_blocks(mutated, frame)])

    def test_no_mass_in_the_unscaling(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "unscale_block", lambda values, probability: values)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_curves(mutated, frame)])

    def test_no_mass_in_the_embedding(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unembedded(curve: Any, index: Any, n: int) -> np.ndarray:
            out = np.zeros(n)
            out[index] = curve
            return out

        monkeypatch.setattr(tmle_module, "embed_stratum_curve", unembedded)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_curves(mutated, frame)])

    def test_the_analytic_curve_sees_both_masses(self, fitted: Any, frame: pd.DataFrame) -> None:
        assert_every_witness_fails(
            [
                lambda: check_curves(fitted, frame, unscale=False),
                lambda: check_curves(fitted, frame, embed=False),
            ]
        )


def test_the_cross_fitted_natural_course_refuses_strata(frame: pd.DataFrame) -> None:
    assert_refused_before_any_call(
        lambda: TMLE(estimands=("ey_obs",), **never_fit_learners(), simultaneous=False).fit(
            frame,
            outcome="Yobs",
            treatment="A",
            covariates=["W", "V"],
            delta="Delta",
            strata=["V"],
        ),
        None,
        "learner",
        "second-moment variance term has no stratum form",
        "docs/roadmap.md F21",
        error=CapabilityError,
    )
