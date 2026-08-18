"""What an incremental fit reports, and what it refuses.

The exactness lives in ``test_influence_gateaux_ipsi.py`` and ``test_remainder_ipsi.py``;
this module is about the estimator wiring -- the names, the axis, the diagnostics, and
every combination that is refused rather than approximated.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cleverly import load
from cleverly.datasets import make_nonlinear_ate
from cleverly.estimators import CTMLE, TMLE
from cleverly.exceptions import DataError
from cleverly.interventions import Incremental, Shift, Static
from tests.conftest import FAST_KWARGS

TILTS = [Incremental(1.0), Incremental(2.0), Incremental(0.5)]


@pytest.fixture(scope="module")
def frame():
    return make_nonlinear_ate(n=800, seed=0)[0]


@pytest.fixture(scope="module")
def fit(frame):
    """One fit, shared: every test below reads a different part of the same result."""
    return TMLE(**FAST_KWARGS, incremental=TILTS).fit(frame, outcome="Y", treatment="A").single()


class TestWhatItReports:
    def test_one_mean_per_tilt_and_one_contrast_per_non_reference(self, fit) -> None:
        assert set(fit.estimates) == {
            "ey_ipsi[natural course]",
            "ey_ipsi[odds x2]",
            "ey_ipsi[odds x0.5]",
            "ate_ipsi[odds x2 vs natural course]",
            "ate_ipsi[odds x0.5 vs natural course]",
        }

    def test_the_axis_is_recorded_on_the_config(self, fit) -> None:
        assert fit.config.parameter_axis == "ipsi"

    def test_the_natural_course_recovers_the_sample_mean(self, fit) -> None:
        """The identity that holds whatever the nuisances are; see the Gateaux module."""
        assert fit.estimates["ey_ipsi[natural course]"].psi == pytest.approx(
            float(np.mean(fit.data.outcome)), abs=1e-8
        )

    def test_tilting_toward_treatment_moves_the_mean_the_way_treatment_does(self, fit) -> None:
        low = fit.estimates["ey_ipsi[odds x0.5]"].psi
        middle = fit.estimates["ey_ipsi[natural course]"].psi
        high = fit.estimates["ey_ipsi[odds x2]"].psi
        assert low < middle < high

    def test_both_score_equations_are_reported_and_solved(self, fit) -> None:
        check = fit.validation.score_check()
        names = {row.name for row in check.rows}
        assert "ipsi" in names and "ipsi (mechanism)" in names
        assert check.passed

    def test_the_alternation_records_what_it_did(self, fit) -> None:
        mechanism = fit.repeats[0].fluctuations["ipsi"].mechanism
        assert mechanism is not None
        assert mechanism.trace, "the outer loop must record its per-round scores"
        joint = [row[3] for row in mechanism.trace]
        # Coordinate ascent on one joint likelihood: it cannot go down.
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(joint))

    def test_the_reported_mechanism_is_the_initial_one(self, fit) -> None:
        """The targeted g lives on the fluctuation, so the diagnostics stay honest."""
        initial = fit.nuisance.propensity.arm(1.0)
        np.testing.assert_array_equal(fit.nuisance.incremental.propensity, initial)
        targeted = fit.repeats[0].fluctuations["ipsi"].mechanism.propensity
        assert not np.allclose(targeted, initial), "targeting must have moved something"


class TestTheOverlapReport:
    def test_it_states_the_bound_the_tilt_guarantees(self, fit) -> None:
        reports = fit.sensitivity.incremental_support()
        assert reports["odds x2"].guaranteed == (pytest.approx(0.5), 2.0)
        assert reports["odds x2"].max_ratio <= 2.0 + 1e-12

    def test_the_natural_course_loses_no_effective_sample_size(self, fit) -> None:
        report = fit.sensitivity.incremental_support()["natural course"]
        assert report.effective_sample_size == pytest.approx(fit.data.n, abs=1e-6)

    def test_positivity_still_reports_because_the_mechanism_still_matters(self, fit) -> None:
        """No doubly-robust fallback here, so g's quality matters *more*, not less."""
        assert fit.sensitivity.positivity() is not None


class TestTheRefusals:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"interventions": [Static(0)]}, "cannot solve their score equations at once"),
            ({"shifts": [Shift(0.5, cap=None)]}, "cannot solve their score equations at once"),
        ],
    )
    def test_two_counterfactual_axes_together_are_refused(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            TMLE(**FAST_KWARGS, incremental=TILTS, **kwargs)

    def test_a_working_model_over_tilts_is_refused(self) -> None:
        from cleverly.msm import MSM

        model = MSM(
            design=lambda a, w: np.column_stack([np.ones(len(w)), np.full(len(w), float(a))]),
            terms=("(intercept)", "a"),
        )
        with pytest.raises(ValueError, match="cannot be combined"):
            TMLE(**FAST_KWARGS, incremental=TILTS, msm=model)

    def test_an_explicit_propensity_bound_is_refused_with_the_reason(self) -> None:
        with pytest.raises(ValueError, match=r"part of the \*estimand\*"):
            TMLE(**FAST_KWARGS, incremental=TILTS, g_bounds=0.01)

    def test_the_default_bound_is_not_refused(self) -> None:
        TMLE(**FAST_KWARGS, incremental=TILTS)  # g_bounds="auto" must still construct

    def test_an_arm_estimand_is_refused_on_a_tilt_fit(self, frame) -> None:
        with pytest.raises(ValueError, match="indexed by"):
            TMLE(**FAST_KWARGS, incremental=TILTS, estimands=["ate"]).fit(
                frame, outcome="Y", treatment="A"
            )

    def test_a_tilt_estimand_with_no_tilt_declared_is_refused(self, frame) -> None:
        with pytest.raises(ValueError, match="indexed by"):
            TMLE(**FAST_KWARGS, estimands=["ey_ipsi"]).fit(frame, outcome="Y", treatment="A")

    def test_a_multi_arm_treatment_is_refused(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 200
        data = pd.DataFrame(
            {
                "Y": rng.normal(size=n),
                "A": rng.choice(["low", "mid", "high"], n),
                "W1": rng.normal(size=n),
            }
        )
        with pytest.raises(DataError, match="odds"):
            TMLE(**FAST_KWARGS, incremental=[Incremental(2.0)]).fit(
                data, outcome="Y", treatment="A"
            )

    def test_ctmle_is_refused_because_each_candidate_is_a_different_parameter(self, frame) -> None:
        with pytest.raises(ValueError, match=r"different\s+parameter"):
            CTMLE(**FAST_KWARGS, incremental=TILTS).fit(frame, outcome="Y", treatment="A")

    def test_an_unknown_reference_names_the_tilts_there_are(self, frame) -> None:
        with pytest.raises(DataError, match="natural course"):
            TMLE(**FAST_KWARGS, incremental=TILTS, reference="nope").fit(
                frame, outcome="Y", treatment="A"
            )

    def test_the_regime_and_shift_reports_refuse_and_name_the_right_one(self, fit) -> None:
        with pytest.raises(ValueError, match="incremental_support"):
            fit.sensitivity.support()
        with pytest.raises(ValueError, match="declared none"):
            fit.sensitivity.shift_support()

    def test_the_truncation_sweep_refuses_because_it_would_move_the_estimand(self, fit) -> None:
        with pytest.raises(ValueError, match=r"\*inside\* the estimand"):
            fit.sensitivity.truncation_curve()


class TestAMissingOutcomeIsAccepted:
    """``delta=`` used to be refused here; what replaced the refusal is the wiring.

    That the composition is *right* is ``test_influence_gateaux_ipsi_mar.py``'s business
    and that the guarantee changes is ``test_remainder_ipsi_mar.py``'s.  What this class
    covers is the estimator plumbing around them, and the two sensitivity reports whose
    answers the missingness mechanism changes.
    """

    @pytest.fixture(scope="class")
    def with_missing(self, frame):
        rng = np.random.default_rng(1)
        out = frame.copy()
        # Missingness that depends on the arm, so a model fitted without `A` would differ
        # -- the same reason tests/discrete_law_mar.py builds PI that way.
        probability = np.where(out["A"] > 0.5, 0.75, 0.9)
        out["D"] = rng.binomial(1, probability)
        out.loc[out["D"] == 0, "Y"] = np.nan
        return out

    @pytest.fixture(scope="class")
    def missing_fit(self, with_missing):
        return (
            TMLE(**FAST_KWARGS, incremental=TILTS)
            .fit(with_missing, outcome="Y", treatment="A", delta="D", covariates=["W1", "W2"])
            .single()
        )

    def test_it_reports_the_same_five_parameters(self, missing_fit) -> None:
        assert set(missing_fit.estimates) == {
            "ey_ipsi[natural course]",
            "ey_ipsi[odds x2]",
            "ey_ipsi[odds x0.5]",
            "ate_ipsi[odds x2 vs natural course]",
            "ate_ipsi[odds x0.5 vs natural course]",
        }

    def test_the_missingness_mechanism_is_fitted(self, missing_fit) -> None:
        assert missing_fit.nuisance.missingness is not None

    def test_both_score_equations_are_still_solved(self, missing_fit) -> None:
        for fluctuation in missing_fit.fluctuations.values():
            assert fluctuation.mechanism is not None
            assert fluctuation.mechanism.relative_score < 1.0

    def test_the_natural_course_is_no_longer_the_complete_case_mean(self, missing_fit) -> None:
        """The identity that holds without ``delta=`` becomes a different one with it.

        ``psi(1)`` is the MAR-identified ``E[Y]``, so averaging the recorded outcomes is
        exactly the mistake it must not make; the exact statement of what it *is* lives in
        ``test_influence_gateaux_ipsi_mar.py``, on a law where both sides are known.
        """
        outcome = np.asarray(missing_fit.data.outcome, dtype=float)
        recorded = np.asarray(missing_fit.data.observed, dtype=bool)
        complete_case = float(outcome[recorded].mean())
        assert missing_fit.estimates["ey_ipsi[natural course]"].psi != pytest.approx(
            complete_case, abs=1e-8
        )

    def test_nuisance_bound_is_accepted_where_g_bounds_is_not(self, with_missing) -> None:
        """``pi`` is a denominator and not part of the estimand, so bounding it is allowed."""
        TMLE(**FAST_KWARGS, incremental=TILTS, nuisance_bound=0.05).fit(
            with_missing, outcome="Y", treatment="A", delta="D", covariates=["W1", "W2"]
        )

    def test_the_mechanism_sweep_is_allowed_but_the_propensity_sweep_is_not(
        self, missing_fit
    ) -> None:
        with pytest.raises(ValueError, match=r"\*inside\* the estimand"):
            missing_fit.sensitivity.truncation_curve()
        curve = missing_fit.sensitivity.truncation_curve([0.01, 0.05], mechanism=True)
        assert len(curve["bound"]) == 2 * len(missing_fit.estimates)

    def test_the_mnar_tilt_refuses_and_says_which_report_to_use(self, missing_fit) -> None:
        with pytest.raises(ValueError, match="incremental_support"):
            missing_fit.sensitivity.missingness_tilt()


class TestTheReferenceCanBeMoved:
    def test_contrasts_are_taken_against_the_named_tilt(self, frame) -> None:
        fit = (
            TMLE(**FAST_KWARGS, incremental=TILTS, reference="odds x2")
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert "ate_ipsi[natural course vs odds x2]" in fit.estimates
        assert "ate_ipsi[odds x0.5 vs odds x2]" in fit.estimates


class TestTheFitSurvivesARoundTrip:
    def test_every_array_and_the_axis_come_back(self, fit, tmp_path) -> None:
        path = tmp_path / "ipsi.joblib"
        fit.save(path)
        back = load(path)
        assert back.config.parameter_axis == "ipsi"
        assert back.nuisance.incremental.names == fit.nuisance.incremental.names
        assert back.nuisance.incremental.deltas == fit.nuisance.incremental.deltas
        for name, estimate in fit.estimates.items():
            np.testing.assert_array_equal(
                back.estimates[name].influence_curve, estimate.influence_curve
            )

    def test_retargeting_a_loaded_fit_reproduces_it(self, fit, tmp_path) -> None:
        """The whole point of storing arrays rather than the declaration."""
        path = tmp_path / "ipsi.joblib"
        fit.save(path)
        back = load(path)
        estimates, _ = back.estimator.retarget(
            back.data, back.nuisance, estimands=("ey_ipsi", "ate_ipsi")
        )
        for name, estimate in estimates.items():
            assert estimate.psi == pytest.approx(fit.estimates[name].psi, abs=1e-12)
