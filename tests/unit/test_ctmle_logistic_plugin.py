r"""``logistic_plugin``: the ``calc_varIC(ICg = TRUE)`` variance R ``ctmle`` reports.

Three witnesses tie it to its definition.

* A line-by-line transcription of ``calc_varIC`` in numpy, on a fit.
* The term :math:`\text{term1}` at a *misspecified* outcome regression, where it does not
  vanish, against the complex-step derivative of :math:`\beta \mapsto P_n D^*(\beta)` at
  the fitted logistic coefficients.  At a correct outcome regression the term is near
  zero, which is why the registered selector study cannot see it.
* R ``ctmle``'s own ``calc_varIC``, run in the pinned image on the inputs of one fit, in
  ``tests/canonical/ctmle_logistic_plugin``.

The status of every selector fit stays ``working_mechanism_plugin``.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import make_binary_outcome, make_multi_arm
from cleverly.estimators import CTMLE, TMLE, logistic_plugin
from cleverly.exceptions import CapabilityError
from tests.canonical.ctmle_logistic_plugin.regenerate import inputs_of

WITNESS = Path(__file__).parents[1] / "canonical" / "ctmle_logistic_plugin"
COVARIATES = ["W1", "W2", "W3"]


class ConstantOutcome(ClassifierMixin, BaseEstimator):
    """A misspecified outcome regression: the marginal mean, whatever ``A`` and ``W`` are."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantOutcome:
        self.classes_ = np.array([0.0, 1.0])
        self.mean_ = float(np.mean(y))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.full(np.asarray(X).shape[0], self.mean_)
        return np.column_stack([1.0 - p, p])


def _fit(strategy: str = "discrete", **settings: Any) -> Any:
    frame, _ = make_binary_outcome(n=400, seed=3)
    options: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(C=1e6, max_iter=2000),
        "strategy": strategy,
        "cross_fit": False,
        "estimands": ("ate", "ey1", "ey0"),
        "simultaneous": False,
        "g_bounds": (0.01, 0.99),
        "random_state": 0,
    }
    if strategy == "discrete":
        options["candidates"] = (("W1", "W2"),)
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**options).fit(frame, outcome="Y", treatment="A", covariates=COVARIATES).single()
        )


@pytest.fixture(scope="module")
def correct() -> Any:
    return _fit()


@pytest.fixture(scope="module")
def misspecified() -> Any:
    return _fit(outcome_learner=ConstantOutcome())


def calc_varic(frame: pd.DataFrame) -> tuple[float, float]:
    """``calc_varIC(Y, Q, h, g1W, A, W, ICg=TRUE)``, transcribed line by line."""
    y, a, g = frame["Y"].to_numpy(), frame["A"].to_numpy(), frame["g1W"].to_numpy()
    qaw, q0, q1 = frame["QAW"].to_numpy(), frame["Q0W"].to_numpy(), frame["Q1W"].to_numpy()
    h = a / g - (1 - a) / (1 - g)
    dstar = (y - qaw) * h + q1 - q0 - (q1.mean() - q0.mean())
    var_dstar = float(np.var(dstar, ddof=1))
    w = frame[[c for c in frame.columns if c.startswith("W")]].to_numpy()
    if w.shape[1] == 0:
        return var_dstar, var_dstar
    w = np.column_stack([np.ones(len(frame)), w])
    term1 = np.mean(-(y - qaw)[:, None] * w * (a * (1 - g) / g + (1 - a) * g / (1 - g))[:, None], 0)
    term2 = np.linalg.inv(
        np.mean([np.outer(row, row) * gi * (1 - gi) for row, gi in zip(w, g, strict=True)], 0)
    )
    ic = dstar + term1 @ term2 @ ((a - g)[:, None] * w).T
    return var_dstar, float(np.var(ic, ddof=1))


class TestTheFormula:
    def test_it_is_the_transcription(self, misspecified: Any) -> None:
        frame = inputs_of(misspecified)
        var_dstar, var_ic = calc_varic(frame)
        n = len(frame)
        diagnostic = logistic_plugin(misspecified)["ate"]
        assert diagnostic.correction_applied
        assert diagnostic.plugin_logistic_std_error == pytest.approx(np.sqrt(var_ic / n), rel=1e-12)
        assert misspecified["ate"].plugin_std_error == pytest.approx(
            np.sqrt(var_dstar / n), rel=1e-12
        )

    def test_term1_is_the_derivative_of_the_mean_curve(self, misspecified: Any) -> None:
        """Nonzero at a misspecified ``Q``, and the complex-step derivative there."""
        frame = inputs_of(misspecified)
        y, a = frame["Y"].to_numpy(), frame["A"].to_numpy()
        qaw = frame["QAW"].to_numpy()
        w = np.column_stack([np.ones(len(frame)), frame[["W1", "W2"]].to_numpy()])
        fitted = LogisticRegression(C=1e6, max_iter=2000).fit(frame[["W1", "W2"]], a)
        beta = np.concatenate([fitted.intercept_, fitted.coef_[0]])
        g = 1.0 / (1.0 + np.exp(-(w @ beta)))
        np.testing.assert_allclose(g, frame["g1W"].to_numpy(), atol=1e-6, rtol=0)

        def mean_curve(b: np.ndarray) -> Any:
            gb = 1.0 / (1.0 + np.exp(-(w @ b)))
            return np.mean((a / gb - (1 - a) / (1 - gb)) * (y - qaw))

        step = 1e-30
        derivative = np.array(
            [
                np.imag(mean_curve(beta + 1j * step * np.eye(len(beta))[j])) / step
                for j in range(len(beta))
            ]
        )
        # The module's term, read back off its curve: IC - D* = S I^{-1} term1.
        diagnostic = logistic_plugin(misspecified)["ate"]
        gf = frame["g1W"].to_numpy()
        score = w * (a - gf)[:, None]
        information = (w * (gf * (1 - gf))[:, None]).T @ w / len(frame)
        projected = np.linalg.lstsq(
            score @ np.linalg.inv(information),
            diagnostic.influence_curve - misspecified["ate"].influence_curve,
            rcond=None,
        )[0]
        assert np.max(np.abs(projected)) > 1e-2
        np.testing.assert_allclose(projected, derivative, atol=1e-4, rtol=1e-3)

    def test_the_mutations_are_caught(self, misspecified: Any) -> None:
        frame = inputs_of(misspecified)
        y, a, g = frame["Y"].to_numpy(), frame["A"].to_numpy(), frame["g1W"].to_numpy()
        qaw = frame["QAW"].to_numpy()
        dstar = misspecified["ate"].influence_curve
        reported = logistic_plugin(misspecified)["ate"].influence_curve
        n = len(frame)
        correction = np.max(np.abs(reported - dstar))
        assert correction > 1e-3

        def curve(intercept: bool, sign: float, inverse: bool) -> np.ndarray:
            cols = frame[["W1", "W2"]].to_numpy()
            w = np.column_stack([np.ones(n), cols]) if intercept else cols
            term1 = np.mean(
                -(y - qaw)[:, None] * w * (a * (1 - g) / g + (1 - a) * g / (1 - g))[:, None], 0
            )
            information = (w * (g * (1 - g))[:, None]).T @ w / n
            middle = np.linalg.inv(information) if inverse else np.eye(w.shape[1])
            return dstar + sign * (term1 @ middle @ ((a - g)[:, None] * w).T)

        np.testing.assert_allclose(curve(True, 1.0, True), reported, atol=1e-12, rtol=0)
        # A flipped sign, a dropped inverse and a dropped intercept each move the curve
        # ten orders of magnitude past the tolerance of the identity above.
        for mutated in (curve(True, -1.0, True), curve(True, 1.0, False), curve(False, 1.0, True)):
            assert np.max(np.abs(mutated - reported)) > 1e-2

    def test_the_arm_terms_sum_to_the_contrast(self, misspecified: Any) -> None:
        out = logistic_plugin(misspecified)
        np.testing.assert_allclose(
            out["ate"].influence_curve,
            out["ey1"].influence_curve - out["ey0"].influence_curve,
            atol=1e-12,
            rtol=0,
        )

    def test_an_intercept_only_candidate_applies_no_correction(self) -> None:
        result = _fit(candidates=((),))
        diagnostic = logistic_plugin(result)["ate"]
        assert not diagnostic.correction_applied
        assert diagnostic.plugin_logistic_std_error == result["ate"].plugin_std_error

    def test_the_plugin_interval_is_the_dstar_interval(self, correct: Any) -> None:
        frame = inputs_of(correct)
        a, g = frame["A"].to_numpy(), frame["g1W"].to_numpy()
        dstar = (a / g - (1 - a) / (1 - g)) * (frame["Y"] - frame["QAW"]).to_numpy() + (
            frame["Q1W"] - frame["Q0W"]
        ).to_numpy()
        dstar = dstar - dstar.mean()
        assert correct["ate"].plugin_std_error == pytest.approx(
            np.sqrt(np.var(dstar, ddof=1) / len(frame)), rel=1e-12
        )


class TestTheRWitness:
    """R ``ctmle``'s own ``calc_varIC`` on the inputs of one fit, committed with hashes."""

    def test_the_artifacts_match_their_hashes(self) -> None:
        manifest = json.loads((WITNESS / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256((WITNESS / name).read_bytes()).hexdigest() == digest, name

    def test_the_fit_reproduces_the_committed_inputs(self) -> None:
        from tests.canonical.ctmle_logistic_plugin.regenerate import witness_inputs

        committed = pd.read_csv(WITNESS / "inputs.csv", float_precision="round_trip")
        fresh = witness_inputs()
        assert list(fresh.columns) == list(committed.columns)
        np.testing.assert_allclose(fresh.to_numpy(), committed.to_numpy(), atol=1e-12, rtol=0)

    def test_the_diagnostic_is_r_s_variance(self) -> None:
        from tests.canonical.ctmle_logistic_plugin.regenerate import REPLICATE, SCENARIO
        from tests.studies import canonical_ctmle_selector as selector

        output = pd.read_csv(WITNESS / "output.csv", float_precision="round_trip").iloc[0]
        frame, _ = selector.draw_scenario(SCENARIO, selector.PRIMARY_N, REPLICATE)
        result = selector.fit_cleverly(frame, SCENARIO)
        diagnostic = logistic_plugin(result)["ate"]
        n = int(output["n"])
        assert diagnostic.correction_applied
        # Declared before the comparison was first run.
        assert diagnostic.plugin_logistic_std_error == pytest.approx(
            np.sqrt(output["var_ic"] / n), rel=1e-6
        )
        assert result["ate"].plugin_std_error == pytest.approx(
            np.sqrt(output["var_dstar"] / n), rel=1e-6
        )
        assert output["var_ic"] != output["var_dstar"]


class TestStatus:
    @pytest.mark.parametrize("strategy", ["greedy", "ordered", "discrete"])
    def test_the_fit_still_refuses_inference(self, strategy: str) -> None:
        settings = {"ordering": tuple(COVARIATES)} if strategy == "ordered" else {}
        result = _fit(strategy, **settings)
        logistic_plugin(result)
        assert result.inference_status == "working_mechanism_plugin"
        estimate = result["ate"]
        for read in (
            lambda: estimate.ci,
            lambda: estimate.pvalue,
            lambda: estimate.std_error,
            lambda: estimate.wald_test(),
        ):
            with pytest.raises(CapabilityError):
                read()


class TestRefusals:
    SCOPE = (
        "the logistic-estimation term of R ctmle's calc_varIC is defined for a binary "
        "treatment whose propensity is fitted once on all rows without weights, missing "
        "outcomes or an intermediate variable. This fit "
    )

    def _refused(self, result: Any, reason: str) -> None:
        with pytest.raises(CapabilityError, match=re.escape(self.SCOPE + reason)):
            logistic_plugin(result)

    def test_a_plain_tmle_fit(self) -> None:
        frame, _ = make_binary_outcome(n=200, seed=1)
        result = (
            TMLE(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                estimands=("ate",),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1"])
            .single()
        )
        self._refused(
            result, "is not a selector C-TMLE fit (strategy 'greedy', 'ordered' or 'discrete')"
        )

    def test_an_oat_fit(self) -> None:
        result = _fit("oat")
        self._refused(
            result, "used strategy='oat', which fits no logistic propensity on covariates"
        )

    def test_a_cross_fitted_fit(self) -> None:
        result = _fit(cross_fit=True, n_folds=2)
        self._refused(result, "cross-fits its propensity")

    def test_a_weighted_fit(self) -> None:
        frame, _ = make_binary_outcome(n=300, seed=5)
        frame = frame.assign(w=np.linspace(0.5, 1.5, len(frame)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                CTMLE(
                    outcome_learner=LogisticRegression(max_iter=1000),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    strategy="discrete",
                    candidates=(("W1",),),
                    cross_fit=False,
                    estimands=("ate",),
                    simultaneous=False,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, weights="w")
                .single()
            )
        self._refused(result, "carries observation weights")

    def test_a_multi_arm_fit(self) -> None:
        frame, _ = make_multi_arm(n=300, seed=2, family="binomial")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = (
                    CTMLE(
                        outcome_learner=LogisticRegression(max_iter=1000),
                        treatment_learner=LogisticRegression(max_iter=1000),
                        strategy="greedy",
                        cross_fit=False,
                        simultaneous=False,
                    )
                    .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                    .single()
                )
            except (CapabilityError, ValueError):
                pytest.skip("a multi-arm selector fit is refused before it reaches here")
        self._refused(result, f"has {result.data.n_arms} treatment arms")
