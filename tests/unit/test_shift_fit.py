"""Reaching the shift estimand from ``TMLE.fit``: what it accepts and what it refuses.

The arithmetic of the clever covariate lives in ``test_shift_submodel.py`` and its
influence curve in ``test_influence_gateaux_shift.py``; the end-to-end recovery of a
known truth lives in ``tests/e2e/test_oracle_shift.py``.  What is left, and what this
module covers, is the seam between them: which keyword declares a shift, what a fit
reports once one is declared, and the six ways of asking for something incoherent.

Every refusal here is a case where the alternative is a *silently wrong* report rather
than an error -- a shift fit offered ``ate``, a dose read as twenty arms, a continuous
treatment with no policy to compare against -- which is why they are asserted by message
rather than merely by type.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.data import CausalData
from cleverly.datasets import GENERATORS, make_shift_dose
from cleverly.exceptions import DataError
from cleverly.interventions import Shift, Static

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SHIFTS = [Shift(0.0, cap=None), Shift(0.5, cap=5.0)]


def frame(n: int = 300, seed: int = 0):  # type: ignore[no-untyped-def]
    return make_shift_dose(n=n, seed=seed)[0]


def binary_frame(n: int = 300, seed: int = 0):  # type: ignore[no-untyped-def]
    """A treatment that really does have arms, for the refusals that need one."""
    return GENERATORS["linear_ate"](n=n, seed=seed)[0]


def fit_binary(**kwargs):  # type: ignore[no-untyped-def]
    data = binary_frame()
    covariates = [c for c in data.columns if c.startswith("W")]
    return estimator(**kwargs).fit(data, outcome="Y", treatment="A", covariates=covariates).single()


def estimator(**kwargs):  # type: ignore[no-untyped-def]
    settings = {
        "outcome_learner": "glm",
        "treatment_learner": "glm",
        "n_folds": 4,
        "random_state": 0,
        "simultaneous": False,
    }
    settings.update(kwargs)
    return TMLE(**settings)


def fit(**kwargs):  # type: ignore[no-untyped-def]
    data = kwargs.pop("data", None)
    if data is None:
        data = frame()
        return (
            estimator(**kwargs)
            .fit(data, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )
    return estimator(**kwargs).fit(data).single()


class TestWhatAShiftFitReports:
    def test_one_mean_per_shift_and_one_contrast_per_non_reference_shift(self) -> None:
        result = fit(shifts=SHIFTS)
        assert set(result.estimates) == {
            "ey_shift[natural course]",
            "ey_shift[+0.5]",
            "ate_shift[+0.5 vs natural course]",
        }

    def test_the_parameters_are_named_by_the_policy_not_by_an_arm(self) -> None:
        """Always labelled, even with exactly two of them.

        Two *arms* get the historical short names ``ate``/``ey1``; two shifts have
        neither history nor an unambiguous reading, so "the ATE" of one policy against
        another is not a name a reader can resolve without the labels.
        """
        result = fit(shifts=SHIFTS)
        assert not any(name in result.estimates for name in ("ate", "ey1", "ey0", "ey"))

    def test_the_fit_solves_the_mtp_score_equation(self) -> None:
        result = fit(shifts=SHIFTS)
        assert "mtp" in result.fluctuations
        assert result.fluctuations["mtp"].converged
        assert bool(result.validation.score_check())

    def test_the_config_records_the_axis(self) -> None:
        assert fit(shifts=SHIFTS).config.parameter_axis == "shift"
        assert fit_binary().config.parameter_axis == "arm"

    def test_the_reference_selects_which_contrast_is_reported(self) -> None:
        result = fit(shifts=SHIFTS, reference="+0.5")
        assert "ate_shift[natural course vs +0.5]" in result.estimates
        # The means do not depend on the reference; only the contrast does.
        other = fit(shifts=SHIFTS)
        assert result.psi("ey_shift[+0.5]") == pytest.approx(other.psi("ey_shift[+0.5]"))

    def test_the_natural_course_reports_the_outcome_mean(self) -> None:
        """Exact, and independent of every nuisance: ``delta = 0`` makes ``h`` one."""
        data = frame()
        result = fit(shifts=[Shift(0.0, cap=None)])
        assert result.psi("ey_shift[natural course]") == pytest.approx(
            float(np.mean(np.asarray(data["Y"]))), abs=1e-10
        )


class TestDeclaringTheAxis:
    def test_shifts_alone_declares_the_treatment_continuous(self) -> None:
        """From a dataframe there is no other way to say it, and a shift names no arm."""
        result = fit(shifts=SHIFTS)
        assert result.data.treatment_kind == "continuous"
        assert result.data.n_arms == 0

    def test_treatment_kind_can_be_declared_explicitly(self) -> None:
        result = (
            estimator(shifts=SHIFTS)
            .fit(
                frame(),
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                treatment_kind="continuous",
            )
            .single()
        )
        assert result.data.is_continuous_treatment

    def test_a_prepared_causal_data_refuses_a_second_declaration(self) -> None:
        data = CausalData.from_frame(
            frame(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            treatment_kind="continuous",
        )
        with pytest.raises(ValueError, match="already assigned"):
            estimator(shifts=SHIFTS).fit(data, treatment_kind="continuous")


class TestTheRefusals:
    def test_shifts_and_interventions_together_are_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot solve their score equations at once"):
            estimator(shifts=SHIFTS, interventions=[Static(0), Static(1)])

    def test_a_shift_of_an_arm_coded_treatment_is_refused(self) -> None:
        data = binary_frame()
        with pytest.raises(DataError, match="which is a Rule"):
            estimator(shifts=SHIFTS).fit(
                data,
                outcome="Y",
                treatment="A",
                covariates=[c for c in data.columns if c.startswith("W")],
                treatment_kind="discrete",
            )

    def test_a_continuous_treatment_with_no_shifts_is_refused(self) -> None:
        """There is no arm-indexed estimand it could report instead."""
        data = CausalData.from_frame(
            frame(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            treatment_kind="continuous",
        )
        with pytest.raises(DataError, match=r"Shift\(0\.0, cap=None\) is the natural course"):
            estimator().fit(data)

    def test_a_shift_fit_cannot_be_asked_for_an_arm_estimand(self) -> None:
        with pytest.raises(ValueError, match="indexed by treatment arm"):
            fit(shifts=SHIFTS, estimands=["ate"])

    def test_a_shift_fit_cannot_be_asked_for_a_regime_estimand(self) -> None:
        """The failure the three-valued axis exists to produce.

        With a two-valued ``requires_intervention`` both a regime fit and a shift fit
        would read as "declared an intervention", so this would be accepted here and
        fail several steps later inside the regime fluctuation, complaining about a
        missing ``regimes=`` rather than about the estimand.
        """
        with pytest.raises(ValueError, match="indexed by declared regime"):
            fit(shifts=SHIFTS, estimands=["ey_regime"])

    def test_an_arm_fit_cannot_be_asked_for_a_shift_estimand(self) -> None:
        with pytest.raises(ValueError, match="indexed by declared shift"):
            fit_binary(estimands=["ey_shift"])

    def test_an_unknown_reference_names_the_declared_shifts(self) -> None:
        with pytest.raises(DataError, match="is not one of the shifts"):
            fit(shifts=SHIFTS, reference="+2")

    @pytest.mark.parametrize("bins", [0, 2])
    def test_too_few_density_bins_are_refused(self, bins: int) -> None:
        with pytest.raises(ValueError, match="density_bins must be at least 3"):
            estimator(density_bins=bins)


class TestTheDiagnosticsMatchTheAxis:
    def test_positivity_refuses_and_names_the_report_that_applies(self) -> None:
        """It would otherwise return an empty table with ``simplex_deviation=1.0``.

        Every field of the arm-level report is per arm, and a dose has none -- so the
        multi-arm branch computed its simplex deviation from a zero-column mechanism and
        got the largest value the field can take, then raised on an empty ``min()``.
        """
        result = fit(shifts=SHIFTS)
        with pytest.raises(DataError, match="shift_support"):
            result.sensitivity.positivity()

    def test_shift_support_reports_the_density_ratio_per_shift(self) -> None:
        result = fit(shifts=SHIFTS)
        report = result.sensitivity.shift_support()
        assert set(report) == {"natural course", "+0.5"}
        # A shift of zero divides a density by itself, so its ratio is one everywhere and
        # it costs no effective sample size at all.
        assert report["natural course"].max_ratio == pytest.approx(1.0)
        assert report["natural course"].ess_ratio == pytest.approx(1.0)
        assert report["+0.5"].max_ratio > 1.0

    def test_shift_support_refuses_an_arm_indexed_fit(self) -> None:
        with pytest.raises(ValueError, match="this fit declared none"):
            fit_binary().sensitivity.shift_support()
