r"""Double robustness, the property the targeting step exists to buy.

TMLE is consistent if *either* the outcome regression or the treatment mechanism is
consistently estimated.  That is a testable claim, and this is the grid that tests it:

======================  ===========================  ==================
outcome model ``Qbar``   treatment model ``g``        expected
======================  ===========================  ==================
correct                  correct                      unbiased (efficient)
correct                  wrong                        unbiased
wrong                    correct                      unbiased
wrong                    wrong                        biased
======================  ===========================  ==================

"Wrong" here means a deliberately misspecified model on a process where a linear model
cannot work -- nonlinear terms, interactions, and a heterogeneous effect.  The last row
matters as much as the others: if a test suite only ever checks that the estimator is
unbiased, it cannot distinguish a working implementation from one that is accidentally
insensitive to its inputs.

Because a single sample confounds bias with sampling noise, these tests average over
replications and compare the bias against its own Monte Carlo standard error.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import nonlinear_dgp
from cleverly.validation import CoverageStudy
from tests.conftest import OracleTreatment


class _OracleOutcomeContinuous:
    """The true conditional mean for a continuous outcome, on the scaled scale.

    The estimator scales ``Y`` onto ``[0, 1]`` before fitting ``Qbar``, so an oracle for a
    continuous outcome has to apply the same transformation.  The scaler is derived from
    the training outcome it is handed, which is exactly what the estimator does.
    """

    def __init__(self, dgp: object) -> None:
        self.dgp = dgp

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"dgp": self.dgp}

    def set_params(self, **params: object) -> _OracleOutcomeContinuous:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        # y arrives already scaled onto [0, 1]; recover the affine map from the raw
        # structural mean so predictions can be returned on the same scale.
        design = np.asarray(X, dtype=float)
        raw = self._raw_mean(design)
        keep = np.isfinite(y)
        slope, intercept = np.polyfit(raw[keep], np.asarray(y)[keep], 1)
        self._slope, self._intercept = float(slope), float(intercept)
        return self

    def _raw_mean(self, design: np.ndarray) -> np.ndarray:
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.outcome_mean(w, 1.0, None), dtype=float)  # type: ignore[attr-defined]
        zero = np.asarray(self.dgp.outcome_mean(w, 0.0, None), dtype=float)  # type: ignore[attr-defined]
        return np.where(a == 1.0, one, zero)

    def predict(self, X: np.ndarray) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        return np.clip(self._intercept + self._slope * self._raw_mean(design), 1e-9, 1 - 1e-9)


def _study(
    outcome_learner: object, treatment_learner: object, n: int, reps: int, n_jobs: int = 2
) -> object:
    return CoverageStudy(
        dgp=nonlinear_dgp(),
        estimator=lambda: TMLE(
            outcome_learner=outcome_learner,
            treatment_learner=treatment_learner,
            n_folds=4,
            learner_folds=3,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
        ),
        n=n,
        n_replicates=reps,
        estimands=("ate",),
        seed=11,
        n_jobs=n_jobs,
    ).run()


class TestDoubleRobustnessGrid:
    """Both nuisances wrong is the only cell that should be biased."""

    @pytest.mark.parametrize(
        "label,outcome_learner,treatment_learner",
        [
            ("both correct", "oracle_q", "oracle_g"),
            ("outcome correct only", "oracle_q", "glm"),
            ("treatment correct only", "glm", "oracle_g"),
        ],
    )
    def test_one_correct_nuisance_suffices(
        self, label: str, outcome_learner: str, treatment_learner: str
    ) -> None:
        dgp = nonlinear_dgp()
        q = _OracleOutcomeContinuous(dgp) if outcome_learner == "oracle_q" else "glm"
        g = OracleTreatment(dgp) if treatment_learner == "oracle_g" else "glm"
        summary = _study(q, g, n=700, reps=40)["ate"]
        # The bias must be indistinguishable from zero at the Monte Carlo resolution.
        assert abs(summary.bias) < max(3.5 * summary.bias_se, 0.03), (
            f"{label}: bias {summary.bias:+.4f} +- {summary.bias_se:.4f}"
        )

    def test_both_nuisances_wrong_is_visibly_biased(self) -> None:
        summary = _study("glm", "glm", n=700, reps=40)["ate"]
        # This is the control condition: with no consistent nuisance, double robustness
        # has nothing to work with and the bias is real and detectable. Without this
        # row, a suite of "is unbiased" assertions could not distinguish a working
        # estimator from one insensitive to its own inputs.
        assert abs(summary.bias) > 4.0 * summary.bias_se
        assert abs(summary.bias) > 0.1

    def test_the_standard_error_shrinks_at_the_root_n_rate(self) -> None:
        small = _study("glm", "glm", n=500, reps=30)["ate"]
        large = _study("glm", "glm", n=2000, reps=30)["ate"]
        # Quadrupling n should halve the standard error.
        ratio = large.mean_std_error / small.mean_std_error
        assert ratio == pytest.approx(0.5, abs=0.1)


@pytest.mark.slow
class TestFlexibleLearners:
    """The practical case for the Super Learner, at sizes the fast tier cannot afford."""

    def test_a_flexible_learner_removes_the_bias_the_glm_leaves(self) -> None:
        misspecified = _study("glm", "glm", n=1000, reps=60)["ate"]
        flexible = _study("fast", "fast", n=1000, reps=60)["ate"]
        # Same estimator, same data: the bias largely disappears once the nuisance
        # models are able to fit the process.
        assert abs(flexible.bias) < 0.5 * abs(misspecified.bias)

    def test_bias_shrinks_with_sample_size(self) -> None:
        small = _study("fast", "fast", n=500, reps=60)["ate"]
        large = _study("fast", "fast", n=2000, reps=60)["ate"]
        assert abs(large.bias) < abs(small.bias)
        # Root-n consistency: sqrt(n) * bias stays bounded rather than growing.
        assert abs(large.root_n_bias) < 2.0 * max(abs(small.root_n_bias), 0.5)
