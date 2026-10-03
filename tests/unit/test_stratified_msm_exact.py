"""Linked working models with baseline strata, against the expanded-design oracle.

Stratum ``s``'s coefficients ``beta_s`` are the projection of the counterfactual means on the
law given ``V = s``.  The projection loss separates by stratum, so ``(beta_s)_s`` are the
coefficients of one working model with the expanded design ``(I(V = s) phi)_s``.  That model
is an ordinary unstratified MSM, and the package fits it through its shipped path.  It is the
independent oracle here: a stratified fit must report the same coefficients and the same
curves, to the solver's tolerance.

The marginal coefficients come from the shipped unstratified solve, so they must equal an
unstratified fit bit for bit.

Two mutations must fail the oracle comparison: blocks built at the marginal coefficients
instead of each stratum's, and stratum estimates read from the marginal fluctuation instead of
the nested stratum one.  The rank check refuses a design that is singular inside a stratum
before any learner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.msm import MSM
from tests import discrete_law
from tests.studies import stratified_law as law
from tests.unit._declaration_support import assert_every_witness_fails, tmle_module
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._stratified_alternating_support import (
    PerturbedOutcome,
    PerturbedTreatment,
    expanded_msm,
)
from tests.unit.test_stratified_influence_exact import KnownTreatment

N = 3_000
SEED = 20261103
LINKS = ("logit", "log")
TERMS = ("(intercept)", "a", "W", "a:W")


def model(link: str) -> MSM:
    return MSM.linear(modifiers=("W",), link=link)  # type: ignore[arg-type]


def fit(frame: pd.DataFrame, msm: MSM, *, strata: bool = True) -> Any:
    roles: dict[str, Any] = {"strata": ["V"]} if strata else {}
    return (
        TMLE(
            msm=msm,
            outcome_learner=PerturbedOutcome(),
            treatment_learner=PerturbedTreatment(),
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], **roles)
        .single()
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return law.sample(N, SEED)


@pytest.fixture(scope="module", params=LINKS)
def fits(request: pytest.FixtureRequest, frame: pd.DataFrame) -> dict[str, Any]:
    link = request.param
    return {
        "link": link,
        "stratified": fit(frame, model(link)),
        "marginal": fit(frame, model(link), strata=False),
        "oracle": fit(frame, expanded_msm(model(link)), strata=False),
    }


def check_oracle(stratified: Any, oracle: Any) -> None:
    """Each stratum coefficient and curve equals the expanded-design oracle's."""
    for s in law.STRATA:
        for term in TERMS:
            estimate = stratified[f"msm[{term}][V={s}]"]
            expected = oracle[f"msm[{term}|{s}]"]
            assert estimate.psi == pytest.approx(expected.psi, abs=1e-8), (term, s)
            np.testing.assert_allclose(
                estimate.influence_curve, expected.influence_curve, rtol=0.0, atol=1e-6
            )


class TestTheStratumCoefficients:
    def test_the_stratum_coefficients_differ(self, fits: dict[str, Any]) -> None:
        """The nonzero witness: the strata modify the effect, so ``beta_s`` differ."""
        stratified = fits["stratified"]
        slopes = [stratified[f"msm[a][V={s}]"].psi for s in law.STRATA]
        assert np.ptp(slopes) > 0.1, slopes

    def test_they_equal_the_expanded_design_oracle(self, fits: dict[str, Any]) -> None:
        check_oracle(fits["stratified"], fits["oracle"])

    def test_the_nested_fluctuation_solves_every_block(self, fits: dict[str, Any]) -> None:
        nested = fits["stratified"].fluctuations["msm"].stratified
        assert nested is not None
        assert nested.epsilon.shape == (len(law.STRATA) * len(TERMS),)
        assert np.min(np.abs(nested.epsilon)) > 1e-4
        assert nested.relative_score_norm < 1e-9
        assert nested.projection.converged

    def test_the_marginal_record_is_the_unstratified_fit(self, fits: dict[str, Any]) -> None:
        stratified, marginal = fits["stratified"], fits["marginal"]
        for term in TERMS:
            name = f"msm[{term}]"
            assert stratified[name].psi == marginal[name].psi
            np.testing.assert_array_equal(
                stratified[name].influence_curve, marginal[name].influence_curve
            )
        np.testing.assert_array_equal(
            stratified.fluctuations["msm"].epsilon, marginal.fluctuations["msm"].epsilon
        )

    def test_the_score_check_reads_both_records(self, fits: dict[str, Any]) -> None:
        from cleverly.validation import score_check

        rows = {row.name: row for row in score_check(fits["stratified"]).rows}
        assert {"msm", "msm (strata)"} <= set(rows)
        assert rows["msm (strata)"].passed and rows["msm"].passed


class TestTheMutationsFail:
    def test_blocks_at_the_marginal_coefficients(
        self, fits: dict[str, Any], frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = tmle_module.solve_with_stratified_projection

        def at_the_marginal(data: Any, nuisance: Any, *args: Any, **kwargs: Any) -> Any:
            blocks_at = kwargs.pop("blocks_at")
            beta = tmle_module.reported_beta(nuisance, nuisance.outcome, data.weights)
            return original(
                data,
                nuisance,
                *args,
                blocks_at=lambda betas: blocks_at([beta for _ in betas]),
                **kwargs,
            )

        monkeypatch.setattr(tmle_module, "solve_with_stratified_projection", at_the_marginal)
        mutated = fit(frame, model(fits["link"]))
        assert_every_witness_fails([lambda: check_oracle(mutated, fits["oracle"])])

    def test_stratum_estimates_from_the_marginal_fluctuation(
        self, fits: dict[str, Any], frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "stratum_source", lambda fluctuation: fluctuation)
        mutated = fit(frame, model(fits["link"]))
        assert_every_witness_fails([lambda: check_oracle(mutated, fits["oracle"])])


def test_no_effect_modification_converges() -> None:
    """W-M2: with ``beta_s = beta`` in the population, both records converge and solve."""
    rows = law.sample(2_000, SEED + 1)
    a, w = rows["A"].to_numpy(dtype=float), rows["W"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    rows["Y"] = (rng.random(len(rows)) < expit(-0.8 + a + 0.6 * w)).astype(int)
    result = (
        TMLE(
            msm=MSM.linear(modifiers=("W",), link="logit"),
            outcome_learner=LogisticRegression(tol=1e-12, max_iter=10_000),
            treatment_learner=LogisticRegression(tol=1e-12, max_iter=10_000),
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
        )
        .fit(rows, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
        .single()
    )
    fluctuation = result.fluctuations["msm"]
    assert fluctuation.projection.converged and fluctuation.relative_score_norm < 1e-9
    nested = fluctuation.stratified
    assert nested.projection.converged and nested.relative_score_norm < 1e-9


class TestTheStratumRankCheck:
    """W-M4: a design singular inside a stratum refuses before any learner."""

    @staticmethod
    def refuse(msm: MSM) -> DataError:
        NeverFit.calls = 0
        with pytest.raises(DataError) as raised:
            TMLE(msm=msm, **never_fit_learners(), cross_fit=False, simultaneous=False).fit(
                law.sample(400, SEED),
                outcome="Y",
                treatment="A",
                covariates=["W", "V"],
                strata=["V"],
            )
        assert NeverFit.calls == 0
        return raised.value

    @pytest.mark.parametrize("link", ("identity", "logit"))
    def test_the_stratum_as_a_term(self, link: str) -> None:
        error = self.refuse(MSM.linear(modifiers=("V",), link=link))  # type: ignore[arg-type]
        assert "inside baseline stratum V=" in str(error)
        assert "collinear" in str(error)

    def test_an_interaction_that_copies_its_main_effect(self) -> None:
        """``a:V`` is ``a`` times a constant inside each stratum."""
        msm = MSM(
            design=lambda arm, frame: np.column_stack(
                [
                    np.ones(len(frame)),
                    np.full(len(frame), float(arm)),
                    float(arm) * np.asarray(frame["V"], dtype=float),
                ]
            ),
            terms=("(intercept)", "a", "a:V"),
            design_kind="known",
        )
        error = self.refuse(msm)
        assert "inside baseline stratum" in str(error) and "collinear" in str(error)

    def test_a_design_of_full_rank_in_every_stratum_passes(self) -> None:
        with pytest.raises(AssertionError, match="before any learner"):
            TMLE(
                msm=model("logit"), **never_fit_learners(), cross_fit=False, simultaneous=False
            ).fit(
                law.sample(400, SEED),
                outcome="Y",
                treatment="A",
                covariates=["W", "V"],
                strata=["V"],
            )


class KnownOutcome(BaseEstimator, ClassifierMixin):
    """The law's outcome regression, read from the ``(A, W, V)`` outcome design."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> KnownOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        p = law.outcome(values[:, 0], values[:, 1], values[:, 2])
        return np.column_stack([1.0 - p, p])


def stratum_beta(probs: Any, stratum: int, link: str) -> Any:
    """``beta_s`` as an analytic function of the 36 cell probabilities.

    The projection of ``E[Y | A = a, W, V = s]`` on ``phi = (1, a, W, a W)`` under ``link``,
    with ``h = 1``, over the law of ``W`` given ``V = s``.  Newton runs a fixed number of
    steps with the exact Jacobian, as :func:`tests.discrete_law.functional` does, so a
    complex step differentiates through it.
    """
    inverse, slope, curvature = discrete_law.MSM_LINKS[link]
    p = np.asarray(probs).reshape(3, 3, 2, 2)  # (v, w, a, y)
    q = p[stratum, :, :, 1] / p[stratum].sum(axis=2)
    p_w = p[stratum].sum(axis=(1, 2))
    w = np.arange(3, dtype=float)[:, None] * np.ones((1, 2))
    a = np.ones((3, 1)) * np.array([[0.0, 1.0]])
    phi = np.stack([np.ones_like(w), a, w, a * w], axis=2)
    beta = np.zeros(4, dtype=p.dtype)
    for _ in range(discrete_law.MSM_NEWTON_STEPS):
        m = inverse(np.einsum("wap,p->wa", phi, beta))
        residual = q - m
        first, second = slope(m), curvature(m)
        score = np.einsum("wap,wa,w->p", phi, first * residual, p_w)
        jacobian = np.einsum("wap,waq,wa,w->pq", phi, phi, first**2 - residual * second, p_w)
        beta = beta + np.linalg.solve(jacobian, score)
    return beta


class TestTheStratumCurveIsTheGateauxDerivative:
    """W-M3: the embedded stratum curve is the Gateaux derivative of ``beta_s``.

    The 36 support points, weighted by their probabilities, realise the law exactly, and
    the learners are the law's own regressions.  So nothing is left to target, and the
    reported curve is the efficient influence function at the law.  The reported curve
    carries each row's normalised weight.
    """

    @pytest.fixture(scope="class", params=LINKS)
    def exact(self, request: pytest.FixtureRequest) -> tuple[str, Any, pd.DataFrame]:
        support = pd.DataFrame(np.asarray(law.SUPPORT), columns=["V", "W", "A", "Y"])
        support["wt"] = law.PROBS
        result = (
            TMLE(
                msm=model(request.param),
                outcome_learner=KnownOutcome(),
                treatment_learner=KnownTreatment(),
                cross_fit=False,
                simultaneous=False,
                g_bounds=(1e-6, 1.0 - 1e-6),
                max_iter=100,
                tol=1e-12,
            )
            .fit(
                support,
                outcome="Y",
                treatment="A",
                covariates=["W", "V"],
                strata=["V"],
                weights="wt",
            )
            .single()
        )
        return request.param, result, support

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_each_curve_is_the_derivative(
        self, exact: tuple[str, Any, pd.DataFrame], stratum: int
    ) -> None:
        link, result, _ = exact
        derivative = discrete_law.contamination_eif(
            lambda p: stratum_beta(p, stratum, link),
            law.PROBS,
            [(i,) for i in range(len(law.SUPPORT))],
        )
        weight = law.PROBS / law.PROBS.mean()
        for j, term in enumerate(TERMS):
            estimate = result[f"msm[{term}][V={stratum}]"]
            assert estimate.psi == pytest.approx(
                float(np.real(stratum_beta(law.PROBS, stratum, link)[j])), abs=1e-10
            )
            np.testing.assert_allclose(
                estimate.influence_curve / weight, derivative[:, j], rtol=0.0, atol=1e-8
            )
