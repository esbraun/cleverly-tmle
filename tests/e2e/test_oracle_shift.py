r"""End-to-end behaviour of a shift fit, with the true outcome regression plugged in.

The companion to :mod:`tests.e2e.test_oracle`, whose fixtures are binary all the way
down, and it makes the same three claims in the same order:

1. **the score equation is solved exactly** -- if it is not, the targeting step is
   broken;
2. **the estimate matches an independently written one-step estimator** -- if it does
   not, the plug-in step or the clever covariate is wrong;
3. **the estimate lands within sampling error of the truth** -- if it does not, the
   estimand is misdefined.

What differs is which nuisance is given away.  A shift's mechanism is a conditional
*density*, and there is no oracle for it here: the estimator factorises the density into
bin hazards fitted by a classifier, so an "oracle density" would have to be an oracle
for that discretisation rather than for :math:`g`.  Q-bar is supplied instead, which is
the honest half to fix -- double robustness then says the point estimate is consistent
whatever the density does, so claim 3 is a test of the parameter and the plug-in rather
than of the density estimator's finite-sample behaviour.  The density is separately
checked in :mod:`tests.unit.test_density`, and the clever covariate that consumes it
against a longhand law in :mod:`tests.unit.test_influence_gateaux_shift`.

Two degeneracies are asserted alongside, because both are exact rather than statistical
and both would survive a wrong density: the natural course reports ``E[Y]``, and so does
a shift the cap holds back for everybody.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import tests.conftest as conftest
from cleverly import load
from cleverly.datasets import shift_dgp
from cleverly.estimators import TMLE
from cleverly.interventions import Shift

#: The cap sits just above the bulk of the dose, so it holds back about one row in a
#: hundred.  Deliberate: the ``1{a <= u}`` factor in the clever covariate is invisible
#: whenever the cap sits above the largest dose, which is the case a test can pass
#: without ever exercising the term.
CAP = 5.0

#: The policies the fixture declares, as ``(delta, cap, name)`` for the DGP's ``truth``
#: and as :class:`~cleverly.interventions.Shift` for the estimator.  Written once so the
#: two cannot drift; the names have to agree for the truth to be keyed as reported.
POLICIES: tuple[tuple[float, float | None, str], ...] = (
    (0.0, None, "natural course"),
    (0.25, CAP, "+0.25"),
    (0.5, CAP, "+0.5"),
)

SHIFTS = [Shift(delta, cap=cap) for delta, cap, _ in POLICIES]

N = 4000

#: Finer than the default 20.  The dose spans about eight units here, so 20 bins are
#: 0.4 wide and a 0.25 shift moves most rows *within* a bin, where a binned density is
#: constant and the clever covariate is exactly one.  40 bins resolve both policies, and
#: they also thin the ratio's upper tail -- which is what makes the comparison against
#: the longhand one-step below a sharp test rather than a loose one.
BINS = 40


class OracleShiftOutcome(BaseEstimator):
    """The true ``E[Y | A, W]`` for a dose, on the scale the fluctuation works on.

    The same construction as :class:`tests.conftest.OracleOutcomeContinuous` and for the
    same reason -- the estimator maps ``Y`` onto ``[0, 1]`` before fitting Q-bar, so the
    oracle has to follow it there, and recovers the affine map by regressing the scaled
    outcome it is handed on the raw structural mean.  Both are affine images of one
    quantity, so that fit is a line through the points rather than an approximation.

    What differs is reading the design: a continuous treatment's ``treatment_block`` is
    the dose itself rather than arm indicators, so column 0 is ``a`` and the rest is
    ``W``, and the structural mean is evaluated at that dose directly instead of at two
    arms and selected between.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleShiftOutcome:
        raw = self._raw_mean(np.asarray(X, dtype=float))
        keep = np.isfinite(y)
        slope, intercept = np.polyfit(raw[keep], np.asarray(y)[keep], 1)
        self._slope, self._intercept = float(slope), float(intercept)
        return self

    def _raw_mean(self, design: Any) -> Any:
        a, w = design[:, 0], design[:, 1:]
        return np.asarray(self.dgp.outcome_mean(w, a), dtype=float)

    def predict(self, X: Any) -> Any:
        return self._intercept + self._slope * self._raw_mean(np.asarray(X, dtype=float))


def _fit(*, n: int = N, seed: int = 5, curvature: float = 0.25, **kwargs: Any):  # type: ignore[no-untyped-def]
    dgp = shift_dgp(curvature=curvature)
    frame, truth = dgp.sample(n, shifts=POLICIES, seed=seed)
    settings: dict[str, Any] = {
        "outcome_learner": OracleShiftOutcome(dgp),
        "treatment_learner": "glm",
        "cross_fit": False,
        "shifts": SHIFTS,
        "density_bins": BINS,
        "random_state": 0,
        "simultaneous": False,
    }
    settings.update(kwargs)
    result = (
        TMLE(**settings)
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )
    return result, truth, frame


@pytest.fixture(scope="module")
def oracle_fit():  # type: ignore[no-untyped-def]
    return _fit()


def shift_one_step(result, index: int, delta: float, cap: float | None) -> float:  # type: ignore[no-untyped-def]
    r"""``E[Y^d]`` written longhand, sharing no code with ``src/``.

    .. math::

        \psi = P_n\left[h(A, W)\,(Y - \bar Q(A, W)) + \bar Q(d(A, W), W)\right]

    with :math:`h(a, w) = g(a - \delta \mid w) / g(a \mid w)\,1\{a \le u\} + 1\{a > u -
    \delta\}`.

    The nuisance *predictions* are read off the fit -- the estimated density, and the
    untargeted Q-bar at the observed and shifted doses -- because re-estimating them
    here would compare two fits rather than two estimators.  What is written out is
    everything between them: the shift map, the density ratio, the cap's two indicators
    and the combination.  A plug-in evaluated at the observed dose instead of the shifted
    one, a missing ``1{a <= u}`` factor, or a ratio the wrong way up all show up as a
    disagreement rather than as two copies of one mistake.
    """
    data, nuisance = result.data, result.nuisance
    density, scaler = nuisance.density, nuisance.scaler
    a = np.asarray(data.treatment, dtype=float)
    weights = np.asarray(data.weights, dtype=float)

    denominator = density.density_at(a)
    ratio = np.where(denominator > 0.0, density.density_at(a - delta) / denominator, 0.0)
    if cap is not None:
        ratio = ratio * (a <= cap) + (a > cap - delta)

    # Assembled on the [0, 1] scale the fluctuation works on and unscaled once at the
    # end. The map is affine, so that is the same number as unscaling each term -- and
    # it uses the estimator's own scaler rather than a second reconstruction of it.
    qbar = np.asarray(nuisance.outcome.observed, dtype=float)
    at_shift = np.asarray(nuisance.outcome.arms[float(index)], dtype=float)
    scaled = scaler.scale(np.asarray(data.outcome, dtype=float))
    return scaler.unscale_level(
        float(np.average(ratio * (scaled - qbar) + at_shift, weights=weights))
    )


class TestTheScoreEquationIsSolved:
    def test_the_score_is_zero_to_floating_point(self, oracle_fit) -> None:  # type: ignore[no-untyped-def]
        result, _, frame = oracle_fit
        check = result.validation.score_check()
        assert bool(check)

        # Not merely "within tolerance" -- the fluctuation is a maximum-likelihood
        # solution, so the score is at floating-point zero.
        #
        # Stated relative to the size of the terms being averaged rather than as the flat
        # 1e-12 tests.e2e.test_oracle uses. That module's outcome is binary and its
        # clever covariate is an inverse probability truncated at 0.026, so its summands
        # are O(1) and an absolute bound *is* a relative one. Here the outcome spans some
        # 23 units and the density ratio reaches ~45, so the same claim in absolute terms
        # would be an arbitrary constant. The scale is computed here rather than read
        # from `row.threshold`, which would make the assertion circular.
        scale = float(np.ptp(np.asarray(frame["Y"]))) * float(np.max(result.nuisance.shifts.ratio))
        for row in check.rows:
            assert abs(row.score) < 1e-12 * scale

    def test_the_fluctuation_barely_moves_qbar(self, oracle_fit) -> None:  # type: ignore[no-untyped-def]
        """With the true Q-bar there is nothing for the targeting step to correct."""
        result, _, _ = oracle_fit
        assert np.max(np.abs(result.fluctuations["mtp"].epsilon)) < 0.15


class TestAgainstAnIndependentEstimator:
    @pytest.mark.parametrize("index", range(len(POLICIES)))
    def test_matches_a_longhand_one_step(self, oracle_fit, index: int) -> None:  # type: ignore[no-untyped-def]
        """The two estimators are different, and must still agree to well inside one SE.

        A TMLE plugs the *targeted* Q-bar into the parameter; the one-step adds an
        augmentation term to the untargeted one.  Their difference is exactly what the
        targeting step moved, ``P_n[h (Qbar - Qbar*)] + P_n[Qbar*(d) - Qbar(d)]``, so it
        is not zero and a tolerance of ``0`` would be wrong.  With the true Q-bar there
        is nothing for targeting to correct and the observed gap is ~0.05 standard
        errors; ``5e-3`` on this fixture is about a tenth of one, which a wrong clever
        covariate or a plug-in read at the observed dose would blow through immediately.
        """
        result, _, _ = oracle_fit
        delta, cap, name = POLICIES[index]
        reference = shift_one_step(result, index, delta, cap)
        assert result.psi(f"ey_shift[{name}]") == pytest.approx(reference, abs=5e-3)


class TestTheTruthIsRecovered:
    @pytest.mark.parametrize(
        "estimand",
        [
            "ey_shift[natural course]",
            "ey_shift[+0.25]",
            "ey_shift[+0.5]",
            "ate_shift[+0.25 vs natural course]",
            "ate_shift[+0.5 vs natural course]",
        ],
    )
    def test_within_sampling_error(self, oracle_fit, estimand: str) -> None:  # type: ignore[no-untyped-def]
        result, truth, _ = oracle_fit
        estimate = result[estimand]
        deviation = abs(estimate.psi - truth[estimand])
        assert deviation < 4.0 * estimate.std_error, (
            f"{estimand}: {estimate.psi:.4f} vs truth {truth[estimand]:.4f}, "
            f"se {estimate.std_error:.4f}"
        )

    def test_a_bigger_shift_moves_the_mean_further(self, oracle_fit) -> None:  # type: ignore[no-untyped-def]
        """The response is increasing over the dose range these policies reach."""
        result, _, _ = oracle_fit
        assert (
            result.psi("ey_shift[natural course]")
            < result.psi("ey_shift[+0.25]")
            < result.psi("ey_shift[+0.5]")
        )


class TestTheDegenerateShifts:
    """Two policies whose answer is ``E[Y]`` exactly, whatever the density says."""

    def test_the_natural_course_reports_the_outcome_mean(self, oracle_fit) -> None:  # type: ignore[no-untyped-def]
        result, _, frame = oracle_fit
        # delta = 0 makes the clever covariate identically one and the plug-in read the
        # observed dose, so the influence curve collapses to Y - psi: the estimator of
        # E[Y]. Exact, not statistical -- a tolerance here would hide a real error.
        assert result.psi("ey_shift[natural course]") == pytest.approx(
            float(np.mean(np.asarray(frame["Y"]))), abs=1e-10
        )

    def test_a_shift_nobody_can_take_also_reports_the_outcome_mean(self) -> None:
        """A cap below the smallest dose holds every unit at its own.

        The covariate's ratio term is zeroed by ``1{a <= u}`` and its indicator is one
        everywhere, which is the case the cap's derivation exists to get right -- and the
        one that a cap sitting above the largest dose never exercises.
        """
        dgp = shift_dgp()
        frame, _ = dgp.sample(1500, shifts=POLICIES, seed=11)
        low = float(np.min(np.asarray(frame["A"]))) - 1.0
        estimator = TMLE(
            outcome_learner=OracleShiftOutcome(dgp),
            treatment_learner="glm",
            cross_fit=False,
            shifts=[Shift(1.0, cap=low, name="unreachable")],
            random_state=0,
            simultaneous=False,
        )
        # The resolution warning is *correct* here and is asserted rather than silenced:
        # a policy that moves nobody is exactly what it exists to report, and an analyst
        # who wrote this cap by accident should be told the intervention is invisible.
        with pytest.warns(UserWarning, match="across a bin edge"):
            result = estimator.fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"]
            ).single()
        assert result.psi("ey_shift[unreachable]") == pytest.approx(
            float(np.mean(np.asarray(frame["Y"]))), abs=1e-10
        )


class TestTheDensityIsLoadBearing:
    """Why :class:`~cleverly.datasets.ShiftDGP`'s dose response is quadratic.

    With Q-bar linear in the dose, an uncapped shift's effect is ``beta * delta`` however
    ``A`` is distributed, so the density cancels out of the truth and an oracle test
    built on such a process would pass with the density estimator returning nonsense.
    The curvature is what makes the parameter depend on the conditional law of ``A``.
    """

    @staticmethod
    def _uncapped_effect(curvature: float) -> float:
        dgp = shift_dgp(curvature=curvature)
        truth = dgp.truth([(0.0, None, "natural course"), (1.0, None, "+1")])
        return truth["ate_shift[+1 vs natural course]"]

    def test_a_linear_response_gives_the_shift_a_density_free_effect(self) -> None:
        # beta * delta = 0.5 * 1, exactly, for any mechanism.
        assert self._uncapped_effect(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_a_quadratic_response_does_not(self) -> None:
        # beta*delta + curvature*(2*E[A] + delta) -- a functional of the dose's own law.
        assert self._uncapped_effect(0.25) == pytest.approx(0.5 + 0.25 * (2 * 2.0 + 1.0), abs=1e-3)


class _MissingAtRandom:
    r"""A missingness mechanism for the dose fixture, and an oracle for it.

    :math:`\pi(a, w) = \mathrm{expit}(2.4 - 0.9 a + 1.2 w_1)` -- decreasing in the dose
    and increasing in the confounder, which is what makes a complete-case analysis
    *biased* rather than merely less efficient: dropping the unrecorded rows tilts the
    joint law of ``(A, W)`` the shifted predictions are averaged against.

    The coefficients are not decorative.  The dose is itself confounded by :math:`w_1`
    with coefficient 0.7, so a mechanism whose two slopes push along that direction very
    nearly cancels: at :math:`(-0.45, 0.7)` the population bias on ``ey_shift[+0.5]`` is
    ``-0.025``, a third of a standard error, and the negative control would have been
    vacuous while reading as though it were not.  These leave ``-0.173`` and keep
    :math:`\pi` above 0.014 everywhere -- clear of ``nuisance_bound``, so nothing is
    truncated and the bias is the mechanism's rather than the bound's.

    ``truth`` needs no adjustment at all, and that is a claim rather than a convenience:
    :math:`\pi` is not in :math:`E[\bar Q(d(A, W), W)]`, so the MAR-identified parameter
    is the one the complete-data fixture already reports.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def probability(self, w: Any, a: Any) -> Any:
        latent = np.asarray(w, dtype=float)
        dose = np.asarray(a, dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-(2.4 - 0.9 * dose + 1.2 * latent[:, 0])))

    # The name conftest.OracleDoseMechanism reaches for.
    def missingness(self, w: Any, a: Any) -> Any:
        return self.probability(w, a)


def _mar_frame(frame, dgp, seed: int):  # type: ignore[no-untyped-def]
    """The fixture's frame with outcomes knocked out at random given ``(A, W)``."""
    latent = np.column_stack([np.asarray(frame[name], dtype=float) for name in ("W1", "W2", "W3")])
    dose = np.asarray(frame["A"], dtype=float)
    pi = _MissingAtRandom(dgp).probability(latent, dose)
    observed = np.random.default_rng(seed).random(len(frame)) < pi
    return (
        frame.assign(
            Delta=observed.astype(float),
            Y=np.where(observed, np.asarray(frame["Y"], dtype=float), np.nan),
        ),
        observed,
    )


class TestAShiftWithOutcomesMissingAtRandom:
    """The same three claims as above, with a third nuisance in the covariate.

    The oracle law in ``tests/discrete_law_shift_cde.py`` settles the arithmetic exactly;
    what a real dose adds is that ``pi`` is now *estimated* at each shifted dose by a
    classifier reading the dose as a numeric feature -- which is the thing the refusal
    this replaced said could not be done.
    """

    #: Three times the complete-data fixture's, and the reason is the negative control
    #: rather than the estimate: the complete-case bias is ``0.173`` in population, so at
    #: ``N`` its standard error is the same size and "biased" would not be a statement a
    #: single fit could make. Costs a second, since every nuisance here is a glm or an
    #: oracle.
    MAR_N = 3 * N

    @pytest.fixture(scope="class")
    def mar_fit(self):  # type: ignore[no-untyped-def]
        dgp = shift_dgp(curvature=0.25)
        frame, truth = dgp.sample(self.MAR_N, shifts=POLICIES, seed=5)
        holed, observed = _mar_frame(frame, dgp, seed=11)
        result = (
            TMLE(
                outcome_learner=OracleShiftOutcome(dgp),
                treatment_learner="glm",
                missingness_learner=conftest.OracleDoseMechanism(_MissingAtRandom(dgp)),
                cross_fit=False,
                shifts=SHIFTS,
                density_bins=BINS,
                random_state=0,
                simultaneous=False,
            )
            .fit(holed, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], delta="Delta")
            .single()
        )
        return result, truth, holed, observed

    def test_a_useful_share_of_outcomes_is_missing(self, mar_fit) -> None:  # type: ignore[no-untyped-def]
        _, _, _, observed = mar_fit
        assert 0.15 < 1.0 - float(observed.mean()) < 0.6

    def test_the_score_equation_is_still_solved(self, mar_fit) -> None:  # type: ignore[no-untyped-def]
        result, _, _, _ = mar_fit
        assert result.fluctuations["mtp"].converged
        assert bool(result.validation.score_check())

    @pytest.mark.parametrize(
        "estimand",
        ["ey_shift[natural course]", "ey_shift[+0.5]", "ate_shift[+0.5 vs natural course]"],
    )
    def test_the_truth_is_recovered(self, mar_fit, estimand: str) -> None:  # type: ignore[no-untyped-def]
        result, truth, _, _ = mar_fit
        estimate = result[estimand]
        deviation = abs(estimate.psi - truth[estimand])
        assert deviation < 4.0 * estimate.std_error, (
            f"{estimand}: {estimate.psi:.4f} vs truth {truth[estimand]:.4f}, "
            f"se {estimate.std_error:.4f}"
        )

    def test_dropping_the_incomplete_rows_is_biased(self, mar_fit) -> None:  # type: ignore[no-untyped-def]
        """The reason the mechanism is in the covariate at all.

        A complete-case fit is a perfectly ordinary shift fit run on a *different* joint
        law of ``(A, W)``, so it converges to a different number -- and nothing in its
        own output says so.  Measured here rather than argued.
        """
        result, truth, holed, observed = mar_fit
        dgp = shift_dgp(curvature=0.25)
        complete = holed[np.asarray(observed)]
        naive = (
            TMLE(
                outcome_learner=OracleShiftOutcome(dgp),
                treatment_learner="glm",
                cross_fit=False,
                shifts=SHIFTS,
                density_bins=BINS,
                random_state=0,
                simultaneous=False,
            )
            .fit(complete, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )
        name = "ey_shift[+0.5]"
        corrected = abs(result.psi(name) - truth[name])
        dropped = abs(naive.psi(name) - truth[name])
        # Three rather than four standard errors, and the gap is Monte Carlo slack rather
        # than a weakened claim: the *population* complete-case bias is 0.173, which at
        # this n is 4.2 standard errors, and one draw of it came out at 3.7. Raising the
        # threshold to four would make the test a coin flip on the seed.
        assert dropped > 3.0 * naive[name].std_error
        assert dropped > 3.0 * corrected

    def test_the_overlap_report_is_of_the_whole_weight(self, mar_fit) -> None:  # type: ignore[no-untyped-def]
        result, _, _, _ = mar_fit
        report = result.sensitivity.shift_support()
        assert report["+0.5"].min_mechanism is not None
        assert report["+0.5"].min_mechanism < 1.0
        # The natural course's ratio is one everywhere, so with no mechanism its ESS is
        # exactly n; here the missingness alone brings it down, which is the claim.
        assert report["natural course"].max_ratio > 1.0
        assert report["natural course"].ess_ratio < 1.0

    def test_the_fit_survives_a_round_trip(self, mar_fit, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result, _, _, _ = mar_fit
        path = tmp_path / "shift_mar.npz"
        result.save(path)
        back = load(path)
        assert back.nuisance.missingness.shape == (result.data.n, len(SHIFTS) + 1)
        np.testing.assert_array_equal(back.nuisance.missingness, result.nuisance.missingness)
        for name in result.estimates:
            np.testing.assert_array_equal(back[name].influence_curve, result[name].influence_curve)


class TestTheFitSurvivesARoundTrip:
    def test_treatment_kind_density_and_shifts_all_come_back(self, oracle_fit, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result, _, _ = oracle_fit
        path = tmp_path / "shift.npz"
        result.save(path)
        back = load(path)

        # Without treatment_kind the dose reloads as a discrete treatment with no
        # levels, and is_continuous_treatment silently flips to False.
        assert back.data.treatment_kind == "continuous"
        assert back.data.is_continuous_treatment and back.data.n_arms == 0
        assert back.config.parameter_axis == "shift"
        assert back.nuisance.shifts.names == result.nuisance.shifts.names
        np.testing.assert_array_equal(
            back.nuisance.density.bin_probabilities, result.nuisance.density.bin_probabilities
        )
        for name in result.estimates:
            np.testing.assert_array_equal(back[name].influence_curve, result[name].influence_curve)
