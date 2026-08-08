r"""The coverage study's instrument, at the arithmetic its conclusions rest on.

``benchmarks/drtmle_coverage.py`` is a characterisation and asserts nothing, so almost nothing
in it is under test here.  What *is* under test is the part a rule reads, and for piece C1 that
is a short list with a sharp reason behind each entry:

* **the injection is what it says it is.**  Tier 1's whole claim is that the nuisance sequence
  was *prescribed*, so if the estimator does not receive
  :math:`\bar Q_0 + n^{-\alpha}h_a` exactly then every remainder column below is describing a
  different sequence from the one the design committed to.  Read off a **real fit's**
  out-of-fold arrays rather than off the learner, since what matters is what the estimator got.
* **the drift coefficients are the declared ones, and none of them vanishes.**
  ``docs/drtmle/validation-plan.md`` §5's correction is that the remainder is an **inner
  product** rather than a norm, so :math:`c_a` can be zero with :math:`\|h_a\| > 0` and
  :math:`c_1 - c_0` can be zero with both arm coefficients nonzero.  A design that only
  declared a rate would satisfy every other test here and measure nothing.
* **the three invalid-fit accountings do what the frozen rule says.**  The primary one counts
  an algorithmically invalid fit as a coverage **failure**, and the rule has to be written down
  before the numbers exist; this is where "written down" becomes "checked".
* **a fit that raised reaches the table.**  A sweep that dropped it would report the coverage of
  the draws that happened to work.

Piece C3 added three more, and each is a rule a gate reads rather than a convenience:

* **the pooled coverage number stays pooled.**  §5's fourth operational rule -- a mixed cell is
  reported pooled, with the two contract populations *beside* it as description.  A stratum
  quietly becoming the primary number is the failure this pins, and it is built so the two
  disagree completely so that a slip reads ``0.0`` or ``1.0`` rather than something plausible.
* **the two causes of an invalid fit are counted apart.**  Gate 1's clause 2 is *zero
  state-identity failures* and its clause 3 is *every required score negligible*; one boolean
  answers neither, and B1a worded the two apart precisely because they send a reader to
  different places.
* **every table's rows are the width of its headers.**  A structural pin against the one
  mistake that produces a complete, plausible, *wrong* table rather than a failure.

Everything but the first two bullets is arithmetic on hand-built records and fits nothing, which
is why this module is cheap despite testing a study.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

from benchmarks import drtmle_coverage as study  # noqa: E402
from benchmarks import drtmle_injection as injection  # noqa: E402

from cleverly import DRTMLE, TMLE  # noqa: E402
from cleverly.utils.bounds import OutcomeScaler  # noqa: E402
from cleverly.validation import score  # noqa: E402

#: Small enough to be a fast-tier fit and large enough that the injected sequence's ``n``
#: means something.  The injection makes the primary nuisances free -- they are function
#: evaluations rather than learner fits -- so what this pays for is the alternation.
N = 300


@pytest.fixture(scope="module")
def fitted() -> dict:
    """One fit per cell, shared: every test below reads a different part of the same pair.

    ``TMLE`` rather than ``DRTMLE`` wherever the claim is about the *injection*, which is
    ``TMLE``'s nuisance layer and not the variant's -- the reductions cost the alternation and
    nothing here needs them.  ``DRTMLE`` appears once, for the settings refusal.
    """
    dgp = injection.base_law()
    frame, _ = dgp.sample(N, seed=5)
    return {
        cell: TMLE(**injection.settings(cell, N), random_state=0)
        .fit(frame, outcome="Y", treatment="A")
        .single()
        for cell in injection.CELLS
    }


class TestTheEstimatorReceivesThePrescribedSequence:
    """What the estimator got, not what the learner returns -- which are two claims.

    A learner is cloned per fold and its predictions are stitched back by index, so reading
    ``predict`` directly would test the function and leave the plumbing untested: an injected
    learner handed the wrong design column, or predicting at the wrong arm, would pass a
    direct call and put a different sequence into the fit.
    """

    @pytest.mark.parametrize("cell", injection.CELLS)
    @pytest.mark.parametrize("arm", (1.0, 0.0))
    def test_the_outcome_regression_is_the_truth_plus_the_prescribed_perturbation(
        self, fitted, cell, arm
    ) -> None:
        """Exactly, on the scaled scale, at both arms.

        ``pytest.approx`` with a loose tolerance would pass against
        ``OracleOutcomeContinuous``'s recovered affine map, whose error is ``O(n^(-1/2))`` from
        the outcome noise -- **the same order as the injected drift**.  That is the whole
        reason ``q_bounds=`` is declared rather than derived, so the assertion is equality to
        floating point.
        """
        fit = fitted[cell]
        covariates = np.asarray(fit.data.covariates, dtype=float)
        scaler = OutcomeScaler(*injection.Q_BOUNDS)
        expected = scaler.scale(injection.injected_outcome(cell, N, covariates, arm))

        assert np.allclose(fit.nuisance.outcome.arms[arm], expected, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_mechanism_is_the_prescribed_one(self, fitted, cell) -> None:
        """The wrong limit in ``q-drift``, the truth plus a drift in ``g-drift``."""
        fit = fitted[cell]
        expected = injection.injected_mechanism(cell, N, np.asarray(fit.data.covariates, float))

        assert np.allclose(fit.nuisance.propensity.arm(1.0), expected, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_scaler_is_the_declared_support_and_not_the_draws_range(self, fitted, cell) -> None:
        """Which is what makes the equality above possible at all.

        ``q_bounds=None`` would widen the *observed* range by 10% per draw, so the affine map
        would move with the sample and the injected perturbation would be scaled by a random
        factor -- a drift the design did not commit to.
        """
        scaler = fitted[cell].nuisance.scaler

        assert (scaler.lower, scaler.upper) == injection.Q_BOUNDS

    def test_the_mechanism_stays_interior_at_the_smallest_size(self) -> None:
        """So the cells are inside the supported contract by construction rather than by luck.

        ``g-drift``'s perturbation carries a ``g(1 - g)`` factor precisely for this, and the
        smallest size is where the ``n^(-alpha)`` factor is largest and so the check bites.
        The learner *raises* rather than clipping, so this is a check on the design's margin
        and not on a guard.
        """
        dgp = injection.base_law()
        frame, _ = dgp.sample(4000, seed=1)
        covariates = np.column_stack(
            [np.asarray(frame[name], dtype=float) for name in dgp.covariate_names]
        )
        for cell in injection.CELLS:
            values = injection.injected_mechanism(cell, 200, covariates)
            assert values.min() > 0.05, cell
            assert values.max() < 0.95, cell


class TestTheDriftCoefficientsAreTheOnesTheDesignDeclared:
    r"""§5's *"commit the coefficient calculation with the design"*, as arithmetic.

    The coefficient is what a rate alone does not give you.  With
    :math:`\hat Q_a - \bar Q_{0,a} = n^{-\alpha}h_a` the root-``n`` drift is
    :math:`n^{1/2-\alpha}c_a` and :math:`c_a = P_0[(g_{1,a} - g_{0,a})/g_{1,a} \cdot h_a]` is an
    **inner product**: it can vanish because :math:`h_a` is orthogonal to the misspecification
    weight even with :math:`\|h_a\| > 0`, and :math:`c_1 - c_0` can vanish in the ATE with both
    arm coefficients nonzero.  So the design aligns :math:`h_a` with that weight and gives the
    arms **opposite signs**, and this class is what says it worked.
    """

    def test_the_q_drift_cell_realises_its_declared_per_arm_coefficients(self) -> None:
        """Exactly, because ``h_a`` is normalised by the quadrature that defines them."""
        realised = injection.drift_coefficients("q-drift")

        assert realised["c1"] == pytest.approx(injection.Q_DRIFT_C[1.0], rel=1e-9)
        assert realised["c0"] == pytest.approx(injection.Q_DRIFT_C[0.0], rel=1e-9)

    def test_the_g_drift_cell_realises_its_declared_ate_coefficient(self) -> None:
        """One target rather than two, because a binary mechanism has one free function.

        The estimator reads :math:`\\hat g(1|W)` off a classifier and takes the complement, so
        one perturbation determines both arms' coefficients and only their combination can be
        set.  Which arm coefficients fall out is then a *finding* about the design, and the
        next test is what holds them to a floor.
        """
        realised = injection.drift_coefficients("g-drift")

        assert realised["c_ate"] == pytest.approx(injection.G_DRIFT_C_ATE, rel=1e-9)

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_no_coefficient_vanishes_in_either_arm_or_in_the_contrast(self, cell) -> None:
        """The cancellation §5 warns about, as an assertion rather than a hope.

        The ATE's is the sharp one: the arms are given opposite signs, so ``c_ate`` is a **sum
        of magnitudes** and a cancellation there would mean the sign convention had been lost.
        """
        realised = injection.drift_coefficients(cell)

        assert abs(realised["c1"]) > injection.C_MIN
        assert abs(realised["c0"]) > injection.C_MIN
        assert abs(realised["c_ate"]) > injection.C_MIN
        assert realised["c1"] > 0.0 > realised["c0"], "opposite signs, so the ATE cannot cancel"
        assert realised["c_ate"] == pytest.approx(realised["c1"] - realised["c0"])

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_remainder_approaches_the_coefficient_at_the_declared_rate(self, cell) -> None:
        r"""§5's *"verify empirically that* :math:`n^{\alpha}R_2 \to c`\ *"*, exactly.

        Exact rather than empirical here, and that is Tier 1's point: both nuisances are
        prescribed functions of ``W``, so the remainder is a quadrature and not a simulation.
        Three sizes because a rate needs three, and the tightening tolerance is the claim --
        a construction with a first-order error in it would sit at a constant offset instead.
        """
        declared = injection.drift_coefficients(cell)
        for size in (600, 1200, 2400):
            scaled = size**injection.ALPHA * injection.exact_remainder(cell, size)["r2_ate"]
            assert scaled == pytest.approx(declared["c_ate"], rel=5e-3)

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_drifting_nuisance_falls_at_the_declared_rate_and_the_wrong_one_does_not(
        self, cell
    ) -> None:
        """§5's regime-entry pair, and it is a *pair* for a reason.

        A study reporting only the shrinking norm could not tell a product going to zero
        because one factor does from one going to zero because both do -- and the second is the
        regime a plain ``TMLE``'s interval is already valid in, so it would demonstrate
        nothing.  The misspecified nuisance's distance to the truth has to stay **bounded away
        from zero**, which is the second assertion here.
        """
        errors = {size: injection.nuisance_error(cell, size) for size in (600, 1200, 2400)}
        drifting = "q_error_1" if cell == "q-drift" else "g_error"
        fixed = "g_error" if cell == "q-drift" else "q_error_1"
        logs = np.log([600.0, 1200.0, 2400.0])

        slope = np.polyfit(logs, [np.log(errors[size][drifting]) for size in errors], 1)[0]
        assert slope == pytest.approx(-injection.ALPHA, abs=1e-6)
        assert min(errors[size][fixed] for size in errors) > 0.01
        assert len({round(errors[size][fixed], 9) for size in errors}) == 1


#: The floor each arm's **targeted** coefficient clears, as a share of its cell's declared ATE
#: one rather than as an absolute number.  A share because the two cells declare different
#: targeted coefficients -- ``g-drift``'s is bounded by positivity at a quarter of
#: ``q-drift``'s -- so one absolute floor would be a different demand in each.  Fixed by the
#: rule and not by what was measured: an arm carrying under a tenth of the contrast's drift is
#: an arm the contrast is not really about.
B_ARM_SHARE = 0.10


class TestTheTargetedCoefficientsAreTheOnesTheDesignDeclared:
    r"""C3b's repair, as arithmetic: the coefficient a fit's **bias** has, not the plug-in one.

    The class above is necessary and was never sufficient, and C3a's pilot is what showed it.
    Targeting solves :math:`P_0[w_a(\bar Q^*_a - \bar Q_{0,a})] = 0` with
    :math:`w_a = g_{0,a}/\hat g_a`, and the estimator's bias is the same offset against
    :math:`u_a = 1 - w_a` -- so a shape chosen to make :math:`c_a = P_0[u_a h_a]` large has
    constrained *nothing* about what survives the fluctuation.  Measured at the old design, it
    survived ``0.00092`` of a declared ``0.40``.

    So the design declares both, and this class is what says the second one worked.  The
    quadrature is the same Sobol rule everything else here uses, so a coefficient and the
    coverage it explains cannot disagree through two quadratures.
    """

    def test_the_q_drift_cell_realises_its_declared_per_arm_coefficients(self) -> None:
        """Exactly, because the shape is solved for both conditions rather than scaled for one."""
        realised = injection.targeted_coefficients("q-drift")

        assert realised["b1"] == pytest.approx(injection.Q_DRIFT_B[1.0], rel=1e-9)
        assert realised["b0"] == pytest.approx(injection.Q_DRIFT_B[0.0], rel=1e-9)

    def test_the_g_drift_cell_realises_its_declared_ate_coefficient(self) -> None:
        """One target rather than two, for the plug-in coefficient's structural reason."""
        realised = injection.targeted_coefficients("g-drift")

        assert realised["b_ate"] == pytest.approx(injection.G_DRIFT_B_ATE, rel=1e-9)

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_no_targeted_coefficient_vanishes_or_cancels_in_the_contrast(self, cell) -> None:
        """The opposite-sign property, which the *plug-in* coefficients having it did not buy.

        This is the sharpest thing the pilot found and it had no column: at the old design the
        arms' plug-in coefficients were opposite-signed, so ``c_ate`` was a sum of magnitudes --
        and their targeted ones came out **both positive**, so ``b_ate`` was a difference.  The
        design's own no-cancellation property did not survive the step it was never checked
        through.  Declaring ``b`` per arm is what restores it, and this is the assertion.
        """
        realised = injection.targeted_coefficients(cell)

        assert abs(realised["b_ate"]) > injection.C_MIN
        assert abs(realised["b1"]) > B_ARM_SHARE * abs(realised["b_ate"])
        assert abs(realised["b0"]) > B_ARM_SHARE * abs(realised["b_ate"])
        assert realised["b1"] > 0.0 > realised["b0"], "opposite signs, so the ATE cannot cancel"
        assert realised["b_ate"] == pytest.approx(realised["b1"] - realised["b0"])

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_bias_approaches_the_targeted_coefficient_at_the_declared_rate(self, cell) -> None:
        r"""Pre-flight condition 1: :math:`n^{\alpha}R_2(\bar Q^*) \to b`, at the study's sizes.

        ``docs/drtmle/validation-plan.md`` §5 requires this be read **before** a coverage
        dispatch rather than inferred from one afterwards, and requires it at the *targeted*
        regression -- the clause whose absence C3a's pilot failed on.  Exact here rather than
        empirical, because :func:`population_epsilon` solves the score by quadrature.

        The tolerance is looser than the plug-in column's ``5e-3`` and that is a statement
        about ``g-drift``: its mechanism perturbation is inside a probability, so the
        :math:`o(n^{-\alpha})` terms it carries do not vanish the way ``q-drift``'s do.
        """
        declared = injection.targeted_coefficients(cell)
        for size in (600, 1200, 2400):
            scaled = (
                size**injection.ALPHA * injection.exact_targeted_remainder(cell, size)["r2_ate"]
            )
            assert scaled == pytest.approx(declared["b_ate"], rel=1e-2)

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_targeted_weight_is_a_direction_the_fluctuation_cannot_reach(self, cell) -> None:
        r"""The measurement that says Tier 1 can be a demonstration at all.

        The design note kept one alternative live and refused to talk itself out of it: that
        *no* injection into a single nuisance produces a first-order shortfall, because a
        ``TMLE`` with one consistent nuisance is consistent and that is double robustness
        working.  Under that reading the repair would be a scope correction and not a new shape.

        It is decided by whether the targeted weight is degenerate.  :math:`v_a` vanishes
        identically only if :math:`w_a` is constant -- only if :math:`\hat g_a \propto g_{0,a}`
        -- and if it did, the fluctuation would reach every direction the design can inject and
        no shape would survive it.  So this measures :math:`\|v_a\|` and the conditioning of the
        2x2 solve, and it is the test that would fail if the alternative were true.
        """
        dgp = injection.base_law()
        for arm in (1.0, 0.0):
            plugin = dgp.expectation(lambda w, a=arm: injection.plugin_weight(cell, w, a) ** 2)
            targeted = dgp.expectation(lambda w, a=arm: injection.targeted_weight(cell, w, a) ** 2)
            cross = dgp.expectation(
                lambda w, a=arm: (
                    injection.plugin_weight(cell, w, a) * injection.targeted_weight(cell, w, a)
                )
            )
            assert np.sqrt(targeted) > 0.01, "the fluctuation reaches every injectable direction"
            gram = np.array([[plugin, cross], [cross, targeted]])
            assert np.linalg.cond(gram) < 1e3, "the two conditions are nearly the same condition"

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_injected_outcome_stays_inside_its_declared_support(self, cell) -> None:
        """The repair's own hazard, and the reason ``g-drift``'s coefficient is the smaller one.

        A shape that survives targeting is a **larger** shape, since only a fraction of it does
        -- ``q-drift``'s perturbation is four times what it was -- so the injection has more room
        to leave the support it has to stay inside.  The mechanism's side of this is
        ``test_the_mechanism_stays_interior_at_the_smallest_size`` above and predates C3b; this
        is the outcome's, which had no test because the old injection came nowhere near.

        The two cells have very different room, which is exactly why they declare different
        targeted coefficients: ``q-drift`` moves the outcome regression, whose support is
        declared as :data:`~benchmarks.drtmle_injection.Q_BOUNDS` and wide on purpose, and
        ``g-drift`` moves a probability.  :class:`~benchmarks.drtmle_injection.InjectedOutcome`
        **raises** rather than clipping, so a breach would surface as a study that cannot run
        rather than as a silently distorted drift -- but it would surface at dispatch time,
        which is what this is here to prevent.
        """
        dgp = injection.base_law()
        frame, _ = dgp.sample(4_000, seed=11)
        covariates = np.column_stack(
            [np.asarray(frame[name], dtype=float) for name in dgp.covariate_names]
        )
        scaler = OutcomeScaler(*injection.Q_BOUNDS)
        for size in (200, 600, 1200, 2400):
            for arm in (1.0, 0.0):
                outcome = scaler.scale(injection.injected_outcome(cell, size, covariates, arm))
                assert outcome.min() > 0.0 and outcome.max() < 1.0


class TestTheReducedRegressionsAreFittedAndNotInjected:
    """The refusal ``DRTMLE``'s own docstring warns about, as a test rather than a comment.

    The reduced learners default to the primary *specification*, so omitting them would hand
    this cell's injected instance to :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}` -- making
    the reductions prescribed rather than fitted and the estimator a different object from the
    one the study reports on.  :func:`benchmarks.drtmle_injection.settings` therefore does not
    carry them, and the harness names them; this pins both halves of that.
    """

    def test_the_shared_settings_do_not_name_a_reduced_learner(self) -> None:
        settings = injection.settings("q-drift", N)

        assert "reduced_outcome_learner" not in settings
        assert "reduced_treatment_learner" not in settings

    def test_the_harness_names_one_and_it_is_a_library_rather_than_an_instance(self) -> None:
        """A string, so the reductions get a real cross-fitted ``glm`` per round."""
        assert isinstance(study.REDUCED_LEARNER, str)

    def test_omitting_them_does_not_quietly_prescribe_the_reductions(self) -> None:
        """The mutation, run, with the outcome recorded rather than assumed.

        It **raises**, and it is worth being precise about why, because the reason is not a
        check: a reduced regression's design is *univariate* -- :math:`\\hat{\\bar Q}(a, W)` and
        nothing else -- while the injected learner reads an arm out of column 0 and covariates
        out of the rest, so there is no column left for it to evaluate the law at and numpy
        says so.  ``IndexError`` rather than a message naming the problem.

        So the guard here is the *width of a design* and not an argument check, which is why
        :func:`benchmarks.drtmle_injection.settings` omits the reduced learners deliberately
        and the harness names them: relying on this would be relying on an accident.  What the
        assertion pins is that the accident is a failure and not a silently prescribed
        reduction -- which is the outcome that would be unrecoverable, since the arrays would
        look entirely plausible.
        """
        dgp = injection.base_law()
        frame, _ = dgp.sample(N, seed=5)

        with pytest.raises(IndexError):
            DRTMLE(**injection.settings("q-drift", N), random_state=0).fit(
                frame, outcome="Y", treatment="A"
            )

    def test_and_the_named_reductions_are_regressions_of_the_data(self) -> None:
        """The other half: with them named, the reductions are fitted and are residual-scaled.

        :math:`Q_r = \\bar Q_0 - \\bar Q^*` on the ``[0, 1]`` scale, so its magnitude is that of
        a *residual*: it is made of the injected perturbation and the fluctuation's own step,
        and an **injected** :math:`Q_r` would carry the conditional mean's magnitude instead.

        So the bar is read off the design rather than written down as a constant.  It used to
        be a bare ``0.05``, calibrated to the injection C3b replaced -- whose perturbation is
        four times larger, since it now has to survive targeting rather than only exist -- and
        a number like that goes stale silently every time the design moves.  The comparison
        that carries the claim is against the *scaled perturbation*, which moves with it.
        """
        dgp = injection.base_law()
        frame, _ = dgp.sample(N, seed=5)
        fit = (
            DRTMLE(
                **injection.settings("q-drift", N),
                reduced_outcome_learner=study.REDUCED_LEARNER,
                reduced_treatment_learner=study.REDUCED_LEARNER,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        reduced = fit.repeats[0].fluctuations["mean"].reduction.reduced

        covariates = np.asarray(fit.data.covariates, dtype=float)
        scaled_range = injection.Q_BOUNDS[1] - injection.Q_BOUNDS[0]
        injected = (
            max(
                float(np.abs(injection.outcome_perturbation("q-drift", N, covariates, arm)).max())
                for arm in (1.0, 0.0)
            )
            / scaled_range
        )
        assert np.abs(np.asarray(reduced.qr)).max() < 2.0 * injected
        assert np.abs(np.asarray(fit.nuisance.outcome.arms[1.0])).min() > 0.05


def check_row(name: str, kind: str, passed: bool) -> score.ScoreCheckRow:
    """A hand-built row, since only ``kind`` and ``passed`` are read by the split."""
    return score.ScoreCheckRow(
        name=name,
        kind=kind,
        score=0.0 if passed else 1.0,
        threshold=1.0 if passed else 1e-12,
        std_error=1.0,
        passed=passed,
        converged=True,
        n_iter=1,
        method="newton",
    )


def _score_row(**overrides) -> study.ScoreRow:
    """A hand-built score row, so the artefact can be tested without fitting anything."""
    defaults: dict[str, object] = {
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "fold_seed": 2,
        "estimator": "drtmle",
        "tolerance": 1e-3,
        "corrected": True,
        "passed_overall": True,
        "name": "ate",
        "kind": "correction",
        "score": 1e-11,
        "threshold": 1e-6,
        "std_error": 0.1,
        "passed": True,
        "converged": True,
        "n_iter": 3,
        "method": "newton",
        "score_initial": 1e-3,
        "hessian_condition": 12.0,
        "failure": "",
        "folds_converged": 0,
        "folds_total": 0,
        "ratio": 1e-5,
        "reduction": 1e8,
    }
    return study.ScoreRow(**{**defaults, **overrides})  # type: ignore[arg-type]


def record(**overrides) -> study.Replicate:
    """A hand-built replicate, so the accounting can be tested without fitting anything."""
    defaults: dict[str, object] = {
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "fold_seed": 2,
        "estimator": "drtmle",
        "estimand": "ate",
        "truth": 1.5,
        "psi": 1.5,
        "std_error": 0.1,
        "lower": 1.4,
        "upper": 1.6,
        "covered": True,
        "valid": True,
        "identity_failures": 0,
        "score_failures": 0,
        "contract": "theorem",
        "initial_clip_share": 0.0,
        "margin": 0.1,
        "gr1_margin": 0.2,
        "exit_reason": "tolerance",
        "failure": "",
        "rounds": 4,
        "seconds": 1.0,
    }
    return study.Replicate(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestTheInvalidFitAccountingIsTheFrozenRule:
    """Three accountings, and which one is primary is a decision made before the numbers.

    A coverage number computed over the surviving fits is conditional on a non-random subset
    selected on a diagnostic correlated with the fit having gone wrong, and reporting *that* as
    the coverage is the same class of error as reporting a per-protocol analysis as
    intention-to-treat.  So the primary report counts an invalid fit as a failure of the
    procedure, the excluded number is reported beside it with the exclusion rate, and the rate
    is itself the third accounting.
    """

    def test_an_invalid_fit_counts_against_the_primary_number(self) -> None:
        """Even when its interval *does* contain the truth, which is the case that decides it."""
        rows = [record(data_seed=i) for i in range(8)] + [
            record(data_seed=8, valid=False, covered=True),
            record(data_seed=9, valid=False, covered=True),
        ]
        accounted = study.account(rows)

        assert accounted.trials == 10
        assert accounted.primary == pytest.approx(0.8)
        assert accounted.excluded == pytest.approx(1.0)
        assert accounted.invalid_share == pytest.approx(0.2)
        assert accounted.valid_trials == 8

    def test_a_fit_that_raised_is_in_the_denominator(self) -> None:
        """A sweep that dropped it would report the coverage of the draws that worked."""
        rows = [record(data_seed=i) for i in range(9)] + [
            record(data_seed=9, valid=False, covered=False, psi=float("nan"), error="LinAlgError")
        ]
        accounted = study.account(rows)

        assert accounted.trials == 10
        assert accounted.primary == pytest.approx(0.9)
        assert accounted.invalid_share == pytest.approx(0.1)

    def test_the_summary_describes_the_fits_that_produced_an_estimate(self) -> None:
        """``bias`` and ``se_ratio`` cannot be computed from a fit that raised.

        Which is the division of labour: the summary is over the estimates that exist and the
        coverage columns re-derive their own numerator, so a raised fit lowers the coverage
        without turning the bias into ``nan``.
        """
        rows = [record(data_seed=i, psi=1.5 + 0.01 * i) for i in range(6)] + [
            record(data_seed=6, valid=False, covered=False, psi=float("nan"), error="ValueError")
        ]
        summary = study.summarise(rows)

        assert summary is not None
        assert summary.n_replicates == 6
        assert np.isfinite(summary.bias)


class TestTheCoverageRulesAreTheFrozenOnes:
    def test_the_wilson_interval_stays_inside_zero_and_one(self) -> None:
        """The reason §5 prefers it: a Wald interval at 0.98 over 50 reaches above 1.

        An upper limit that cannot be attained makes "compatible with 0.95" satisfied by
        construction on the high side, which is a rule that cannot fail rather than a rule.
        """
        low, high = study.wilson(49, 50)

        assert 0.0 <= low < high <= 1.0

    def test_compatibility_is_the_rules_own_wald_form(self) -> None:
        """Restated exactly, because a rule restated differently is a rule changed.

        The frozen text is ``|coverage-hat - 0.95| <= 1.96 sqrt(p(1-p)/M)``; the Wilson
        interval is reported *beside* it and does not replace it.
        """
        assert study.compatible(48, 50)
        assert not study.compatible(35, 50)

    def test_a_tiny_replication_count_cannot_read_as_success_unnoticed(self) -> None:
        """Compatible and useless at once, which is why the width is on the table.

        Four replicates all covering satisfy nothing -- ``p-hat = 1`` makes the Wald tolerance
        zero -- and the Wilson interval is what says how little the number is worth.
        """
        low, high = study.wilson(4, 4)

        assert not study.compatible(4, 4)
        assert high - low > 0.4

    def test_the_shortfall_is_paired_on_the_draw(self) -> None:
        """Which is what makes gate 2's 0.05 resolvable at 250 rather than at 1,000.

        The fixture is the extreme case and is the one that shows the difference: two
        estimators that agree on every draw but one have a paired standard error of order
        ``1/M``, where treating them as independent would put it at order ``1/sqrt(M)``.
        """
        records = []
        for seed in range(20):
            records.append(record(data_seed=seed, estimator="tmle", covered=seed > 0))
            records.append(record(data_seed=seed, estimator="drtmle", covered=True))
        difference, error, pairs = study.paired_shortfall(records, "q-drift", 600, "ate")

        assert pairs == 20
        assert difference == pytest.approx(0.05)
        assert error < 0.06

    def test_an_unpaired_draw_is_dropped_from_the_comparison_and_not_matched_by_position(
        self,
    ) -> None:
        """The comparison keys on the draw, so a missing arm cannot be silently paired.

        A positional zip would pair ``drtmle``'s draw 3 with ``tmle``'s draw 4 the moment one
        estimator raised, and the difference would then be draw-to-draw variation wearing the
        estimator's name.
        """
        records = [
            record(data_seed=0, estimator="tmle"),
            record(data_seed=1, estimator="tmle"),
            record(data_seed=2, estimator="tmle"),
            record(data_seed=1, estimator="drtmle"),
            record(data_seed=2, estimator="drtmle"),
        ]
        _, _, pairs = study.paired_shortfall(records, "q-drift", 600, "ate")

        assert pairs == 2


class TestTheContractColumnIsReportedPerCell:
    """Gate 1's clause 0, at the arithmetic that turns per-fit labels into a cell's verdict."""

    def test_one_bound_active_draw_makes_the_cell_bound_active(self) -> None:
        """A count and not a median, because the label is a statement about the cell.

        A cell with one bound-active draw in twelve is a cell whose coverage number is evidence
        about two estimators, and a median of the margins would report it as though it were
        about one.  That is exactly what the B2b dispatch's medians could not show.
        """
        records = [record(data_seed=i) for i in range(11)] + [
            record(data_seed=11, contract="bound-active", margin=0.0)
        ]
        (row,) = study.contract_rows(records)

        assert row[3] == "1/12"
        assert row[4] == "BOUND-ACTIVE"

    def test_a_cell_with_no_active_truncation_reads_theorem(self) -> None:
        records = [record(data_seed=i) for i in range(4)]
        (row,) = study.contract_rows(records)

        assert row[4] == "theorem"

    def test_a_plain_fit_contributes_no_contract_row(self) -> None:
        """It has no mechanism tilt, so there is nothing for the label to be about."""
        records = [record(data_seed=i, estimator="drtmle", contract="none") for i in range(4)]

        assert study.contract_rows(records) == []


class TestTheTierIsSelectedRatherThanRefused:
    r"""``--tier 2`` runs now, and the refusal that stood here is **deleted** rather than kept.

    It read *"tier 2 is not implemented here and is refused rather than approximated"*, and
    the thing it named -- prescribed-rate learners plus the fold-retained nuisances
    :math:`P_0\hat D` needs -- is what piece C2 landed.  Deleted rather than guarded, which
    is `lesson 12 <../../docs/drtmle/investigation-log.md>`_ of the investigation log applied
    a second time: a branch kept "in case" is a branch that hides the next mutation.

    What replaces it is a *selection*, and the confusion the refusal existed to prevent is
    now handled by the banner the run prints: the two tiers answer different questions and
    every table says which one it is reporting.
    """

    def test_a_tier_selects_a_module_and_both_supply_one_interface(self) -> None:
        assert set(study.TIERS) == {1, 2}
        for tier in study.TIERS.values():
            for name in (
                "CELLS",
                "ALPHA",
                "base_law",
                "settings",
                "drift_coefficients",
                # C3b's two: a tier that supplied only the plug-in coefficient would print a
                # regime-entry table about a quantity no fit's bias is, which is the whole of
                # what C3a's pilot measured.
                "targeted_coefficients",
                "exact_targeted_remainder",
            ):
                assert hasattr(tier, name)

    def test_tier_two_runs_and_reports_the_same_tables(self, monkeypatch, tmp_path) -> None:
        """Driven through ``main`` rather than through the pieces.

        A smoke size, because what is checked is that a tier-2 dispatch produces the tables
        rather than what the numbers in them are -- the numbers are the study's, and the
        study runs on a runner.
        """
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "drtmle_coverage.py",
                "--tier",
                "2",
                "--cells",
                "q-drift",
                "--sizes",
                "200",
                "--replicates",
                "2",
                "--jobs",
                "1",
                "--evaluation-n",
                "400",
                "--out",
                str(tmp_path),
            ],
        )
        study.main()

    def test_an_unknown_tier_is_still_refused_by_argparse(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["drtmle_coverage.py", "--tier", "3"])
        with pytest.raises(SystemExit):
            study.main()


class TestTheEvaluationRuleIsChosenRatherThanAssumed:
    """Two rules for the companion, and a row that says which one produced it.

    The i.i.d. draw is what C3c ran and is the default, so every recorded invocation
    reproduces; the quasi-random rule is
    :func:`~benchmarks.drtmle_remainder.quadrature_frame`, whose error is orders smaller and,
    under an **independent scramble per replicate**, is mean-zero noise the study averages
    down rather than the bias a fixed grid left it -- see
    [E1b](../../docs/roadmap.md#what-e1b-measures).  What is under test here is the wiring
    and the refusal, not the statistics: those are
    ``tests/unit/test_drtmle_remainder_study.py``'s, where the identities are.
    """

    def test_the_deterministic_rule_runs_end_to_end(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "drtmle_coverage.py",
                "--tier",
                "1",
                "--cells",
                "q-drift",
                "--sizes",
                "200",
                "--replicates",
                "2",
                "--jobs",
                "1",
                "--evaluation-n",
                "0",
                "--quadrature-points",
                "128",
                "--out",
                str(tmp_path),
            ],
        )
        study.main()

        written = sorted(tmp_path.glob("*.jsonl"))
        assert len(written) == 2
        (replicates,) = [path for path in written if not path.stem.endswith("-scores")]
        rows = [
            json.loads(line)
            for line in replicates.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["estimator"] == "drtmle"
        ]
        assert {row["companion_rule"] for row in rows} == {"sobol"}
        # Two arms per point, times the scrambles: the companion holds every replicate of the
        # rule, which is what puts a measured error on the row rather than a derived one.
        assert {row["companion_rows"] for row in rows} == {2 * 128 * study.QUADRATURE_SCRAMBLES}

    def test_the_scramble_is_per_replicate_and_the_rule_error_is_measured(
        self, monkeypatch, tmp_path
    ) -> None:
        """E1b's rule, at the two places a row can show it.

        The scramble varies **across draws**, which is what makes the rule's error mean-zero
        noise the study averages down instead of the same bias on every row; and
        ``companion_replicate_se`` is finite, which is what says the error was measured from
        this fit's own scrambles rather than derived from a halving witness.  A single fixed
        grid passes neither.
        """
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "drtmle_coverage.py",
                "--tier",
                "1",
                "--cells",
                "q-drift",
                "--sizes",
                "200",
                "--replicates",
                "2",
                "--jobs",
                "1",
                "--evaluation-n",
                "0",
                "--quadrature-points",
                "128",
                "--out",
                str(tmp_path),
            ],
        )
        study.main()

        (replicates,) = [
            path for path in sorted(tmp_path.glob("*.jsonl")) if not path.stem.endswith("-scores")
        ]
        rows = [
            json.loads(line)
            for line in replicates.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["estimator"] == "drtmle"
        ]

        assert len({row["companion_scramble"] for row in rows}) == 2
        assert all(np.isfinite(row["companion_replicate_se"]) for row in rows)

    def test_both_rules_at_once_are_refused_by_name(self, monkeypatch) -> None:
        """A run under both would integrate against a rule nobody chose."""
        # Sized so that a *removed* refusal fails this test quickly rather than dispatching
        # the study's defaults and hanging -- a mutation that has to be waited out is one
        # nobody watches fail.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "drtmle_coverage.py",
                "--cells",
                "q-drift",
                "--sizes",
                "200",
                "--replicates",
                "1",
                "--jobs",
                "1",
                "--evaluation-n",
                "400",
                "--quadrature-points",
                "128",
            ],
        )
        with pytest.raises(SystemExit):
            study.main()


class TestTheScoreRowsAreKeptWholeRatherThanCounted:
    """``valid`` plus two counts is what gate 1 reads; it is not what E3 can classify from.

    The counts stay exactly as they are -- clauses 2 and 3 read them apart and every table
    below does too -- and the rows are **in addition**, in a second artefact.  Which equation
    missed, by what ratio against its threshold, and whether it started large and was driven
    down or started near zero: none of that survives a count, and all of it is what
    [E3](../../docs/roadmap.md#e-what-c3c-handed-back) has to replay the invalid fits with.
    """

    def test_the_two_artefacts_share_one_timestamp(self, tmp_path) -> None:
        """Joinable by name, which is the whole value of the second file.

        Two ``strftime`` calls a second apart produce a pair a reader cannot tell belongs
        together, so the two paths come out of one call rather than two.
        """
        replicates = [record(data_seed=1), record(data_seed=2)]
        scores = [_score_row(data_seed=1), _score_row(data_seed=2)]

        path, score_path = study.write_records(replicates, scores, tmp_path)

        assert score_path.name == f"{path.stem}-scores.jsonl"
        assert len(score_path.read_text(encoding="utf-8").splitlines()) == 2
        assert json.loads(score_path.read_text(encoding="utf-8").splitlines()[0])["data_seed"] == 1

    def test_every_field_of_the_librarys_row_reaches_the_artefact(self) -> None:
        """A **structural** pin, and it is the one that keeps the artefact honest.

        The library's ``ScoreCheckRow`` is what a fit records; if it grows a field and this
        record does not, the artefact silently stops carrying it and nobody finds out until
        somebody needs it. ``folds_converged`` is the one shape change -- a pair in JSON is a
        list whose order a reader has to know -- so it becomes two named ``int`` fields, and
        both are checked for here rather than exempted.
        """
        library = {field.name for field in fields(score.ScoreCheckRow)}
        recorded = {field.name for field in fields(study.ScoreRow)}

        assert library <= recorded, library - recorded
        assert {"folds_converged", "folds_total"} <= recorded
        # The two derived properties as well: a reader of the artefact should not have to
        # rebuild the ratio a threshold is read against.
        assert {"ratio", "reduction"} <= recorded

    def test_every_row_is_written_and_not_only_the_failing_ones(self) -> None:
        """A passing row carries ``score_initial``, which is what says targeting had work."""
        check = SimpleNamespace(
            tolerance=1e-3,
            corrected=True,
            passed=False,
            rows=(
                check_row("ate", "correction", True),
                check_row("ate", "identity", False),
                check_row("ate", "diagnostic", True),
            ),
        )
        payload = study.Payload("q-drift", 600, 1, 2)

        rows = study._score_rows(payload, "drtmle", check)

        assert [row.kind for row in rows] == ["correction", "identity", "diagnostic"]
        assert all(row.data_seed == 1 and row.estimator == "drtmle" for row in rows)
        assert all(row.passed_overall is False for row in rows)


class TestTheRemainderColumnsAreItemThirteens:
    """What the remainder table reports, and what it does when it cannot report it."""

    def test_the_table_reads_the_drtmle_rows_only(self) -> None:
        """A plain ``TMLE`` fit has no companion, so it contributes no remainder row.

        Item 13 is a condition of *Theorem 1*, which is the doubly-robust estimator's; a
        remainder column filled in for the plain arm would be a number about a different
        expansion under the same heading.
        """
        records = [
            record(data_seed=i, estimator=name, remaining=float("nan") if name == "tmle" else 0.01)
            for i in range(4)
            for name in ("tmle", "drtmle")
        ]
        rows = study.remainder_rows(records)

        assert rows
        assert all(len(row) == len(study.REMAINDER_HEADERS) for row in rows)

    def test_a_run_without_an_evaluation_draw_reports_no_rows(self) -> None:
        """Absent rather than blank: a column of ``nan`` reads as a measurement that failed."""
        records = [record(data_seed=i, estimator="drtmle") for i in range(4)]

        assert study.remainder_rows(records) == []

    def test_an_unresolved_branch_is_reported_as_such_rather_than_as_zero(self) -> None:
        """``nan`` in, ``-`` out.

        The branch columns are binned approximations and
        ``benchmarks/drtmle_remainder.py`` returns ``nan`` where they did not separate from
        their own discretisation error.  Printing a zero there would report "the branch is
        negligible" for a branch nobody measured, which is the one reading gate 1's clause 4
        must not be given.
        """
        records = [
            record(
                data_seed=i,
                estimator="drtmle",
                remaining=0.01,
                root_n_remaining=0.2,
                branch_q=float("nan"),
                branch_g=float("nan"),
                branch_movement=0.5,
            )
            for i in range(4)
        ]
        (row,) = [entry for entry in study.remainder_rows(records) if entry[2] == "ate"]
        cell = dict(zip(study.REMAINDER_HEADERS, row, strict=True))

        assert cell["R_Q"] == "-"
        assert cell["R_g"] == "-"
        # The cancellation ratio is clause 4's second half and is a ratio *of* the branches, so
        # where they did not resolve it must not resolve either: a number here would say the
        # total rests on no cancellation, which is a claim about two quantities nobody measured.
        assert cell["cancel"] == "-"
        assert cell["branches settled"] == "0/4"


class TestEveryTablesRowsMatchItsHeaders:
    """A structural pin, and the hazard it exists for is specific rather than hypothetical.

    The whole output of a dispatch is a log a human reads down a column.  A header tuple one
    place out from its row builder does not fail, raise or print a ragged table -- it
    **relabels every number underneath it**, and the study whose numbers those are is the one
    run on this page whose cost makes redoing it a decision rather than an errand.

    It is checked here rather than trusted to review because it is the mistake that was
    actually made: inserting ``sqrt(n) R2`` into :func:`remainder_rows` while its headers lived
    at the call site in ``main`` produced a complete, plausible, wrong table.
    """

    def test_each_builder_returns_rows_the_width_of_its_headers(self) -> None:
        records = [
            record(
                data_seed=i,
                estimator=name,
                contract="theorem" if i % 2 else "bound-active",
                remaining=float("nan") if name == "tmle" else 0.01,
                root_n_remaining=float("nan") if name == "tmle" else 0.2,
                branch_q=float("nan") if name == "tmle" else 0.004,
                branch_g=float("nan") if name == "tmle" else -0.001,
                branch_movement=1e-5,
                # On **both** arms, unlike every other remainder column: the targeted
                # remainder needs no companion, and the row a shortfall is read against is
                # the plain `TMLE`'s.
                r2_targeted=0.08,
            )
            for i in range(4)
            for name in ("tmle", "drtmle")
        ]
        builders = {
            study.REGIME_HEADERS: study.regime_rows(records, [600]),
            study.ENTRY_HEADERS: study.entry_rows(records),
            study.COVERAGE_HEADERS: study.coverage_rows(records),
            study.SHORTFALL_HEADERS: study.shortfall_rows(records),
            study.REMAINDER_HEADERS: study.remainder_rows(records),
            study.CONTRACT_HEADERS: study.contract_rows(records),
            study.STRATUM_HEADERS: study.stratum_rows(records),
            study.VALIDITY_HEADERS: study.validity_rows(records),
            study.PREFLIGHT_HEADERS: study.preflight_rows(records),
            study.REPLICATE_HEADERS: study.replicate_rows(records),
        }

        for headers, rows in builders.items():
            assert rows, headers
            for row in rows:
                assert len(row) == len(headers), (headers, row)


class TestTheTwoCausesOfAnInvalidFitAreCountedApart:
    """Gate 1 asks for them apart, so one boolean cannot answer it.

    Clause 2 is *zero state-identity failures across the whole study* and clause 3 is *every
    required final score negligible*.  They are different findings -- an identity residual is a
    software defect that iterating longer cannot fix, and a score above its threshold is a fit
    that did not converge -- and B1a shipped ``correction_check`` precisely to keep the wording
    of the two apart.  A study that collapsed them into ``valid`` would report a clause-2
    failure as though it were a clause-3 one, which sends a reader to ``one_step`` and a smaller
    step size for a fit whose solver did its job.
    """

    def test_an_identity_failure_is_not_counted_as_a_score_failure(self) -> None:
        records = [
            record(data_seed=i, valid=False, identity_failures=1, score_failures=0)
            for i in range(3)
        ]

        cells = [
            dict(zip(study.VALIDITY_HEADERS, r, strict=True)) for r in study.validity_rows(records)
        ]
        (cell,) = [c for c in cells if c["estimator"] == "drtmle"]

        assert cell["identity"] == "3"
        assert cell["score"] == "0"
        assert cell["invalid share"] == "1.000"

    def test_a_score_failure_is_not_counted_as_an_identity_failure(self) -> None:
        records = [
            record(data_seed=i, valid=False, identity_failures=0, score_failures=2)
            for i in range(3)
        ]

        cells = [
            dict(zip(study.VALIDITY_HEADERS, r, strict=True)) for r in study.validity_rows(records)
        ]
        (cell,) = [c for c in cells if c["estimator"] == "drtmle"]

        assert cell["identity"] == "0"
        assert cell["score"] == "6"

    def test_the_split_is_taken_off_the_checks_own_selector(self) -> None:
        """The arithmetic in the harness, not just the column it feeds.

        Built by hand rather than by fitting: what is under test is that the two counts
        partition ``failures``, so a check carrying one failing identity row and two failing
        score rows must come back ``(1, 2)`` and never ``(1, 3)`` or ``(3, 0)``.
        """
        rows = (
            check_row("ok", "score", passed=True),
            check_row("identity[1.0]", "identity", passed=False),
            check_row("eq8", "score", passed=False),
            check_row("correction[1.0]", "correction", passed=False),
        )
        check = score.ScoreCheck(rows=rows, tolerance=study.VALIDITY_TOLERANCE, n=600)

        assert study._failure_counts(check) == (1, 2)

    def test_a_check_with_nothing_failing_carries_neither_count(self) -> None:
        rows = (check_row("ok", "score", passed=True),)
        check = score.ScoreCheck(rows=rows, tolerance=study.VALIDITY_TOLERANCE, n=600)

        assert study._failure_counts(check) == (0, 0)

    def test_a_fit_that_raised_carries_neither_count(self) -> None:
        """There is no returned state for an identity to be checked against.

        Such a row is read by ``error`` and by ``invalid share``, both of which already carry
        it; putting it in ``identity`` as well would report a software defect the study has no
        evidence for.
        """
        payload = study.Payload(cell="q-drift", n=600, data_seed=1, fold_seed=2)

        rows = study._failed(payload, "drtmle", "ValueError", dict.fromkeys(study.ESTIMANDS, 0.0))

        assert all(r.identity_failures == 0 and r.score_failures == 0 for r in rows)
        assert all(r.error == "ValueError" and not r.valid for r in rows)


class TestTheMixedCellRuleIsPooledPrimaryWithStrataBeside:
    """C3's frozen decision, at the arithmetic rather than at the prose.

    ``docs/drtmle/validation-plan.md`` §5's fourth operational rule.  C1's witness found cells
    are **mixed**, so gate 1's clause 0 is a share and there is no cell-level label a coverage
    number can be read under.  The rule: the pooled number stays primary and is what clauses 5
    and 6 read; the two contract populations are reported beside it as *description*, because
    the label is a post-fit property of the draw and selecting on it conditions on a non-random
    subset exactly as excluding invalid fits would.
    """

    def test_the_pooled_number_is_over_every_draw_and_not_over_a_stratum(self) -> None:
        """The load-bearing one: a stratum's coverage must not become the cell's.

        Built so the two disagree -- every bound-active draw covers and no theorem-side one
        does -- so a pooled number that had quietly become either stratum's reads 0.0 or 1.0
        rather than the 0.5 the cell actually has.
        """
        records = [
            record(
                data_seed=i,
                contract="bound-active" if i % 2 else "theorem",
                covered=bool(i % 2),
            )
            for i in range(8)
        ]

        cells = [
            dict(zip(study.COVERAGE_HEADERS, r, strict=True)) for r in study.coverage_rows(records)
        ]
        (pooled,) = [c for c in cells if c["estimator"] == "drtmle"]

        assert pooled["coverage"] == "0.500"
        assert pooled["reps"] == "8"

    def test_the_strata_partition_the_cell(self) -> None:
        records = [
            record(data_seed=i, contract="bound-active" if i % 2 else "theorem", covered=i % 2 == 1)
            for i in range(8)
        ]

        rows = [
            dict(zip(study.STRATUM_HEADERS, r, strict=True)) for r in study.stratum_rows(records)
        ]
        by_population = {r["population"]: r for r in rows}

        assert sum(int(r["reps"]) for r in rows) == 8
        assert by_population["theorem"]["coverage"] == "0.000"
        assert by_population["bound-active"]["coverage"] == "1.000"
        assert by_population["theorem"]["share"] == "0.500"

    def test_a_plain_fit_contributes_no_stratum_row(self) -> None:
        """A ``TMLE`` fit has no mechanism tilt, so it is on neither side of the contract."""
        records = [record(data_seed=i, estimator="tmle", contract="none") for i in range(4)]

        assert study.stratum_rows(records) == []


class TestTheRemainderCarriesTheColumnsTheGatesRead:
    """``sqrt(n) R2`` is gate 2's and ``cancel`` is the second half of gate 1's clause 4."""

    def test_root_n_r2_is_root_n_times_the_mean_plain_remainder(self) -> None:
        records = [
            record(data_seed=i, estimator="drtmle", remaining=0.01, r2=0.02) for i in range(4)
        ]

        cells = [
            dict(zip(study.REMAINDER_HEADERS, r, strict=True))
            for r in study.remainder_rows(records)
        ]
        (cell,) = [c for c in cells if c["estimand"] == "ate"]

        assert cell["sqrt(n) R2"] == f"{np.sqrt(600) * 0.02:+.4f}"

    def test_branches_of_one_sign_show_no_cancellation(self) -> None:
        assert study._cancellation(0.004, 0.002) == "1.00x"

    def test_opposed_branches_report_how_much_the_total_rests_on_cancelling(self) -> None:
        """The failure mode clause 4 names: a small total built from two large branches.

        ``|0.010| + |-0.009| = 0.019`` over a total of ``0.001`` is a total that is 19 times
        smaller than the numbers it came from, which is exactly *"large with the other
        cancelling it"* and is invisible in ``sqrt(n) R_rem`` alone.
        """
        assert study._cancellation(0.010, -0.009) == "19.00x"
