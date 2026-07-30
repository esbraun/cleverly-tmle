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

That table has a premise it does not show: **positivity**.  Every row of it is measured at
comfortable overlap, and read alone it invites the slogan "either nuisance will do".
:class:`TestDoubleRobustnessIsNotSymmetricUnderWeakOverlap` runs the same grid where
positivity nearly fails, and there the two halves come apart -- the outcome-model half
still delivers, the propensity half does not.  Which half you are relying on turns out to
matter, and that is not visible anywhere in the table above.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly import TMLE
from cleverly.datasets import DGP, nonlinear_dgp
from cleverly.utils.bounds import expit
from cleverly.validation import CoverageStudy
from tests.conftest import OracleOutcomeContinuous, OracleTreatment


def weak_overlap_nonlinear(strength: float = 2.2) -> DGP:
    """:func:`~cleverly.datasets.nonlinear_dgp` with the propensity pushed into the tails.

    The bundled :func:`~cleverly.datasets.weak_overlap_dgp` cannot serve here: its outcome
    mean is linear, so ``"glm"`` is *correct* for ``Qbar`` and the two "wrong" cells of the
    grid would not be wrong.  Scaling the nonlinear process's linear predictor keeps both
    nuisances misspecifiable while moving 11% of the population below ``g = 0.05`` and 2%
    above 0.95 -- a practical positivity violation rather than a nominal one.
    """
    base = nonlinear_dgp()

    def propensity(w: Any) -> Any:
        return expit(
            strength
            * (0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0))
        )

    return DGP(
        name=f"weak_overlap_nonlinear(strength={strength})",
        n_latent=4,
        covariate_names=("W1", "W2", "W3", "W4"),
        propensity=propensity,
        outcome_mean=base.outcome_mean,
    )


def _study(
    outcome_learner: object,
    treatment_learner: object,
    n: int,
    reps: int,
    n_jobs: int = 2,
    dgp: DGP | None = None,
) -> object:
    return CoverageStudy(
        dgp=dgp if dgp is not None else nonlinear_dgp(),
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
        q = OracleOutcomeContinuous(dgp) if outcome_learner == "oracle_q" else "glm"
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


class TestDoubleRobustnessIsNotSymmetricUnderWeakOverlap:
    """The same grid where positivity nearly fails -- and the two halves stop matching.

    The grid above runs at comfortable overlap, and there both single-correct cells come
    out unbiased, which invites the summary "either nuisance will do".  That summary drops
    the premise.  Double robustness is consistency *given positivity*, and the two halves
    do not lean on positivity equally: with ``Qbar`` right the estimand is recovered by
    integrating a regression over the covariate distribution, which needs no overlap at
    all, while with only ``g`` right everything rests on ``1/g`` weights that positivity is
    exactly what bounds.

    On a process with 11% of the population below ``g = 0.05`` the asymmetry is plain, and
    it is stable across seeds (measured at three, ``n = 700``, 40 replications):

    ========================  ==================  ==================
    cell                      bias                significance
    ========================  ==================  ==================
    ``Qbar`` correct only     -0.005 to -0.013    under 1 sigma
    ``g`` correct only        -0.118 to -0.152    3.5 to 4.9 sigma
    neither correct           -0.510 to -0.540    22 to 29 sigma
    ========================  ==================  ==================

    Two things this is *not*.  It is not a truncation artefact: sweeping ``g_bounds`` from
    ``auto`` down to ``1e-5`` moves the middle row's bias by less than 0.01 while inflating
    its standard error by 75%, so the bound is buying variance and not causing the bias.
    And it is not a bug -- it is the positivity premise doing what premises do when they
    fail.  Worth pinning precisely because the reassuring version of the double-robustness
    slogan is the one people remember: sweeping ``n`` to 11200 at 40 replications left the
    middle row at -0.079 +- 0.011, still several standard errors from zero.
    """

    N = 700
    REPS = 40

    @pytest.fixture(scope="class")
    def dgp(self) -> DGP:
        return weak_overlap_nonlinear()

    def test_the_overlap_really_is_weak(self, dgp) -> None:
        # The premise. If the propensity stayed away from the boundary this class would be
        # a slower copy of the grid above.
        import numpy as np

        rng = np.random.default_rng(0)
        latent = rng.normal(size=(40_000, dgp.n_latent))
        g = dgp.propensity(latent)
        assert float(np.mean(g < 0.05)) > 0.08
        assert float(np.mean(g > 0.95)) > 0.01

    @pytest.fixture(scope="class")
    def cells(self, dgp) -> dict[str, object]:
        return {
            "outcome correct only": _study(
                OracleOutcomeContinuous(dgp), "glm", n=self.N, reps=self.REPS, dgp=dgp
            )["ate"],
            "treatment correct only": _study(
                "glm", OracleTreatment(dgp), n=self.N, reps=self.REPS, dgp=dgp
            )["ate"],
            "neither correct": _study("glm", "glm", n=self.N, reps=self.REPS, dgp=dgp)["ate"],
        }

    def test_the_outcome_half_still_delivers(self, cells) -> None:
        # A correct outcome regression is unaffected by the overlap problem: it never
        # divides by g.
        summary = cells["outcome correct only"]
        assert abs(summary.bias) < max(3.5 * summary.bias_se, 0.05), summary

    def test_the_propensity_half_does_not(self, cells) -> None:
        # The finding. Not asserted as a bound on how bad it gets -- that would be
        # over-fitting to this process -- but as detectably nonzero, which is the claim
        # the comfortable-overlap grid cannot make.
        summary = cells["treatment correct only"]
        assert abs(summary.bias) > 2.5 * summary.bias_se, summary
        assert abs(summary.bias) > 0.05, summary

    def test_neither_correct_is_worse_than_either(self, cells) -> None:
        # The control, and the ordering. If the middle row were as bad as the bottom one
        # the grid would be saying nothing about g at all.
        outcome = abs(cells["outcome correct only"].bias)
        treatment = abs(cells["treatment correct only"].bias)
        neither = abs(cells["neither correct"].bias)
        assert outcome < treatment < neither
        assert neither > 0.3
        assert neither > 2.0 * treatment
