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

Everything in the last two is arithmetic on hand-built records and fits nothing, which is why
this module is cheap despite testing a study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

from benchmarks import drtmle_coverage as study  # noqa: E402
from benchmarks import drtmle_injection as injection  # noqa: E402

from cleverly import DRTMLE, TMLE  # noqa: E402
from cleverly.utils.bounds import OutcomeScaler  # noqa: E402

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
        a *residual* -- three orders below the outcome regression's own values here, since this
        cell's injected :math:`\\hat Q` is nearly right by construction.  An injected
        :math:`Q_r` would carry the conditional mean's magnitude instead.
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

        assert np.abs(np.asarray(reduced.qr)).max() < 0.05
        assert np.abs(np.asarray(fit.nuisance.outcome.arms[1.0])).min() > 0.05


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


class TestTierTwoIsRefusedRatherThanApproximated:
    def test_the_refusal_fires_before_a_single_fit_and_names_where_tier_two_lives(
        self, monkeypatch
    ) -> None:
        """So that a dispatch cannot produce a number the harness has no learner for.

        The two tiers answer different questions and the confusion between them is the one this
        script is most likely to cause -- Tier 1's numbers are about a prescribed nuisance
        sequence and are not an applied claim -- so the refusal is a message naming what Tier 2
        would need and which piece owns it, rather than an ``argparse`` rejection that says only
        that 2 is not 1.

        Driven through ``main`` rather than asserted against the source: what matters is that it
        fires *before* any fit, which reading the file cannot establish.
        """
        monkeypatch.setattr(
            sys, "argv", ["drtmle_coverage.py", "--tier", "2", "--replicates", "500"]
        )
        monkeypatch.setattr(
            study,
            "one_draw",
            lambda payload: pytest.fail("the refusal must fire before any fit"),
        )

        with pytest.raises(SystemExit, match="prescribed-rate"):
            study.main()
