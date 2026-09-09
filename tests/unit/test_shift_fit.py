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

import warnings

import numpy as np
import pytest
import sklearn.linear_model

import tests.conftest as conftest
import tests.discrete_law_shift_cde as shift_law
from cleverly.data import CausalData
from cleverly.datasets import GENERATORS, make_shift_dose
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.fluctuation.iterative import InitialFit, solve_fluctuation
from cleverly.fluctuation.submodel import submodel_for
from cleverly.interventions import Shift, ShiftSet, Static, check_shift_support
from cleverly.learners.density import ConditionalDensity

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SHIFTS = [Shift(0.0, cap=None), Shift(0.5, cap=5.0)]

#: The crossed oracle law's three policies, in the order their codes run.
LAW_SHIFTS = [
    Shift(0.0, cap=shift_law.CAP, name="natural course"),
    Shift(shift_law.DELTA, cap=shift_law.CAP, name="+1"),
    Shift(shift_law.DELTA, cap=shift_law.CAP_TIGHT, name="+1 (cap 2)"),
]
LAW_DGP = shift_law.DiscreteShiftCoarsenedLaw()


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
        "outcome_learner": sklearn.linear_model.LinearRegression(),
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
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
        assert bool(result.diagnostics.score_equations())

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


class TestACoarsenedShiftFit:
    """``delta=`` and ``intermediate=`` on a dose, at the seam this module covers.

    The influence curve is checked against a Gateaux derivative in
    ``test_influence_gateaux_shift_cde.py`` and the covariate's blocks longhand in
    ``test_shift_submodel.py``.  What is left here is what ``fit_nuisances`` produces --
    which doses the mechanisms were evaluated at -- and the one behavioural claim neither
    of the others can make: that the *counterfactual* blocks reach the reported number.
    """

    def _law_fit(self, **kwargs):  # type: ignore[no-untyped-def]
        """A fit on the crossed oracle law, with the mechanisms given away exactly.

        The *density* is not given away and cannot be: the estimator factorises it into
        bin hazards over quantile edges chosen from the sample, which on four tied doses
        is not this law's unit-width partition. Nothing here depends on it -- the shifted
        doses come from ``Shift.apply`` and the mechanisms from the oracle.
        """
        kwargs.setdefault("shifts", LAW_SHIFTS)
        kwargs.setdefault("outcome_learner", conftest.OracleDoseOutcome(LAW_DGP))
        kwargs.setdefault("missingness_learner", conftest.OracleDoseMechanism(LAW_DGP))
        settings = {"cross_fit": False, "density_bins": 4, **kwargs}
        fit_kwargs = settings.pop("fit_kwargs", {})
        level = settings.pop("level", None)
        results = estimator(**settings).fit(
            shift_law.frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            **fit_kwargs,
        )
        return results.single() if level is None else results[float(level)]

    def _cde_fit(self, level: int, **kwargs):  # type: ignore[no-untyped-def]
        return self._law_fit(
            level=level,
            outcome_learner=conftest.OracleDoseOutcome(LAW_DGP, has_intermediate=True),
            intermediate_learner=conftest.OracleDoseMechanism(LAW_DGP, role="intermediate_mean"),
            fit_kwargs={"intermediate": "Z"},
            **kwargs,
        )

    def _cells(self):  # type: ignore[no-untyped-def]
        law_frame = shift_law.frame()
        covariate = law_frame["W"].to_numpy().astype(int)
        index = np.rint(law_frame["A"].to_numpy(dtype=float)).astype(int)
        return covariate, index

    def test_the_missingness_is_evaluated_at_the_observed_and_each_shifted_dose(self) -> None:
        result = self._law_fit()
        covariate, index = self._cells()
        missingness = result.nuisance.missingness
        assert missingness is not None
        assert missingness.shape == (shift_law.N, len(LAW_SHIFTS) + 1)

        # Block 0 is the row's own dose; block s + 1 is the dose policy s assigns. The
        # order is ShiftSet.design's first axis, and the correspondence is the contract.
        np.testing.assert_allclose(
            missingness[:, 0], shift_law.PI_EXACT[covariate, index], atol=1e-9, rtol=0
        )
        for position, name in enumerate(("natural course", "+1", "+1 (cap 2)")):
            mapping = np.asarray(shift_law.POLICIES[name])
            np.testing.assert_allclose(
                missingness[:, position + 1],
                shift_law.PI_EXACT[covariate, mapping[index]],
                atol=1e-9,
                rtol=0,
            )
        # And the shifted blocks really do differ from the observed one, so the four
        # assertions above are not all the same assertion.
        assert np.max(np.abs(missingness[:, 2] - missingness[:, 0])) > 1e-2

    @pytest.mark.parametrize("level", (None, 0, 1))
    def test_the_truth_comes_back_exactly(self, level: int | None) -> None:
        """The end-to-end statement, and it is an equality rather than an interval.

        The sample realises the law exactly and the nuisances are the law's own, so every
        score is already zero within its cell -- whatever the density's binning does --
        ``epsilon`` is zero, and the plug-in *is* the functional.  The mechanism's
        binning is why this is the strongest available end-to-end check and a coverage
        study is not: nothing here is a sampling statement.
        """
        result = self._law_fit() if level is None else self._cde_fit(level)
        assert float(np.max(np.abs(result.fluctuations["mtp"].epsilon))) == pytest.approx(
            0.0, abs=1e-9
        )
        for name, truth in shift_law.TRUTH[level].items():
            assert result.psi(name) == pytest.approx(truth, abs=1e-9)

    def test_the_controlled_direct_effects_are_not_the_marginal_one(self) -> None:
        # On this law the shift effect changes sign between the levels, and neither equals
        # the parameter that leaves Z alone -- so a fit that ignored `intermediate=` would
        # be reporting a visibly different number rather than a subtly different one.
        name = "ate_shift[+1 vs natural course]"
        marginal = self._law_fit().psi(name)
        at_zero, at_one = self._cde_fit(0).psi(name), self._cde_fit(1).psi(name)
        assert at_zero < 0.0 < at_one
        assert abs(marginal - at_zero) > 1e-2
        assert abs(marginal - at_one) > 1e-2

    def test_the_intermediate_density_is_evaluated_there_too(self) -> None:
        result = self._cde_fit(1)
        covariate, index = self._cells()
        intermediate = result.nuisance.intermediate
        assert intermediate is not None
        assert intermediate.shape == (shift_law.N, len(LAW_SHIFTS) + 1)
        np.testing.assert_allclose(
            intermediate[:, 0], shift_law.QZ_EXACT[covariate, index], atol=1e-9, rtol=0
        )
        mapping = np.asarray(shift_law.POLICIES["+1"])
        np.testing.assert_allclose(
            intermediate[:, 2], shift_law.QZ_EXACT[covariate, mapping[index]], atol=1e-9, rtol=0
        )
        assert np.max(np.abs(intermediate[:, 2] - intermediate[:, 0])) > 1e-2

    def test_the_counterfactual_blocks_reach_the_reported_estimate(self) -> None:
        """The claim the other two instruments cannot make.

        The Gateaux module runs at ``epsilon = 0``, where the curve reads the observed
        block and the untargeted ``Qbar``, so a mechanism at the wrong dose in a
        counterfactual block moves nothing there; the submodel test pins the arrays but
        says nothing about whether they are used.  Here the initial fit is deliberately
        wrong, so the fluctuation actually moves -- and then perturbing *only* the
        counterfactual blocks has to move ``psi``.
        """
        covariate, index = self._cells()
        law_frame = shift_law.frame()
        dose = law_frame["A"].to_numpy(dtype=float)
        outcome = law_frame["Y"].to_numpy(dtype=float)
        observed = law_frame["Delta"].to_numpy(dtype=float).astype(bool)

        density = ConditionalDensity(shift_law.G_EXACT[covariate], shift_law.EDGES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = CausalData.from_arrays(
                outcome,
                dose,
                covariate.reshape(-1, 1).astype(float),
                treatment_kind="continuous",
                delta=observed.astype(float),
            )
            shifts = ShiftSet.evaluate(LAW_SHIFTS, data, density)

        maps = [np.asarray(shift_law.POLICIES[n]) for n in shifts.labels.values()]
        # Deliberately wrong by a constant on the logit scale, so epsilon is not zero.
        qbar = np.clip(shift_law.QBAR_MARGINAL_EXACT + 0.15, 1e-6, 1 - 1e-6)
        initial = InitialFit(
            qbar[covariate, index],
            {
                float(code): qbar[covariate, mapping[index]]
                for code, mapping in zip(shifts.codes, maps, strict=True)
            },
        )
        at = np.column_stack([index] + [mapping[index] for mapping in maps])
        truthful = shift_law.PI_EXACT[covariate[:, None], at]
        # Same observed block, different counterfactual ones.
        perturbed = truthful.copy()
        perturbed[:, 1:] = np.clip(perturbed[:, 1:] * 0.8, 1e-6, 1.0)

        def psi_of(pi: np.ndarray) -> float:
            submodel = submodel_for(
                "mtp",
                dose,
                np.zeros((dose.size, 0)),
                arms=(),
                shifts=shifts.design,
                missingness=pi,
            )
            fluctuation = solve_fluctuation(
                np.nan_to_num(outcome), initial, submodel, np.ones(dose.size), observed
            )
            return float(np.mean(fluctuation.targeted.arms[1.0]))

        assert np.max(np.abs(np.asarray(psi_of(truthful) - psi_of(perturbed)))) > 1e-3

    def test_the_diagnostics_read_the_mechanism_at_the_rows_own_dose(self) -> None:
        result = self._law_fit()
        covariate, index = self._cells()
        report = {model.name: model for model in result.diagnostics.nuisance_models().models}
        assert "missingness" in report
        # A calibration report compares a prediction against an outcome, and Delta is
        # evidence about pi at the dose the row actually took -- block 0, no selection.
        support = result.diagnostics.support()
        assert support["+1"].min_mechanism == pytest.approx(
            float(np.min(np.maximum(shift_law.PI_EXACT[covariate, index], 0.01))), abs=1e-9
        )
        # The two reweightings multiply, so the weight's ESS is below the bare ratio's.
        bare = check_shift_support(
            result.nuisance.shifts, result.nuisance.density, result.data.treatment
        )
        assert support["+1"].ess_ratio < bare["+1"].ess_ratio
        assert bare["+1"].min_mechanism is None

    def test_the_mnar_tilt_refuses_this_axis_by_name(self) -> None:
        result = self._law_fit()
        with pytest.raises(ValueError, match="continuous dose with shifts="):
            result.sensitivity.missingness()

    def test_the_mechanism_truncation_curve_is_not_flat(self) -> None:
        """The diagnostic that would be vacuous if ``pi`` were baked in at fit time.

        ``truncation_curve(mechanism=True)`` admits the sweep as soon as a missingness
        mechanism exists and passes the swept bound into ``retarget``.  A flat curve reads
        as "the estimate does not hinge on the truncation choice", so it has to be a real
        sweep or it is a wrong conclusion reported silently.  This is the diagnostic that
        would be vacuous by construction had the mechanism been folded into ``ShiftSet``
        at nuisance-fit time rather than kept on ``NuisanceEstimates`` and bounded here.

        The outcome learner is deliberately **not** the oracle: with an exact ``Qbar``
        every score is zero whatever the covariate is, ``epsilon`` is zero, and the curve
        is flat for a reason that has nothing to do with the bound.
        """
        result = self._law_fit(outcome_learner=sklearn.linear_model.LinearRegression())
        assert float(np.max(np.abs(result.fluctuations["mtp"].epsilon))) > 1e-6
        curve = result.diagnostics.truncation_curve(mechanism=True, bounds=[0.01, 0.3, 0.45])
        values = np.asarray(curve["psi"])[np.asarray(curve["estimand"]) == "ey_shift[+1]"]
        assert float(np.ptp(values)) > 1e-6


class TestAWeightedShiftFit:
    """``weights=`` on a dose, at the seam. The estimand is checked on the oracle law in
    ``test_influence_gateaux_shift.py``; what is left is that the tilt reaches every
    nuisance, the density included, and that a constant weight changes nothing.
    """

    def _weighted(self, weights, **kwargs):  # type: ignore[no-untyped-def]
        data = frame()
        data = data.assign(wt=weights(data))
        return (
            estimator(shifts=SHIFTS, **kwargs)
            .fit(data, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], weights="wt")
            .single()
        )

    def test_a_constant_weight_reproduces_the_unweighted_fit(self) -> None:
        # The package's convention is mean-one weights, so a constant weight is exactly
        # one and the whole fit -- density included -- must be the one it always was.
        flat = self._weighted(lambda d: np.ones(len(d)))
        plain = fit(shifts=SHIFTS)
        for name in plain.estimates:
            assert flat.psi(name) == pytest.approx(plain.psi(name), abs=1e-12)
            assert flat.estimates[name].std_error == pytest.approx(
                plain.estimates[name].std_error, abs=1e-12
            )

    def test_the_tilt_reaches_the_density_and_moves_the_estimate(self) -> None:
        tilted = self._weighted(lambda d: np.exp(0.6 * np.asarray(d["W1"])))
        plain = fit(shifts=SHIFTS)
        assert abs(tilted.psi("ey_shift[+0.5]") - plain.psi("ey_shift[+0.5]")) > 1e-3
        # The clever covariate is the *tilted* density's ratio, so the mechanism itself
        # has to have seen the weights -- not only the outcome regression and the average.
        ratio = np.asarray(tilted.nuisance.shifts.ratio[:, 1])
        assert np.max(np.abs(ratio - np.asarray(plain.nuisance.shifts.ratio[:, 1]))) > 1e-3

    def test_the_auto_bound_resolves_at_the_effective_sample_size(self) -> None:
        # Not a no-op line: `_bounds_n` returns Kish's effective n, and the summary says so
        # when it bites. On this axis g is not truncated at all, so the bound reaches only
        # the mechanisms -- which is why the docs say `nuisance_bound=` is the one to watch.
        tilted = self._weighted(lambda d: np.exp(0.6 * np.asarray(d["W1"])))
        assert tilted.config.auto_bounds_n is not None
        assert tilted.config.auto_bounds_n < tilted.data.n
        assert tilted.config.auto_bounds_n == pytest.approx(tilted.data.effective_n)

    def test_the_summary_reports_the_weighted_dose_mean_and_observed_range(self) -> None:
        tilted = self._weighted(lambda data: np.exp(0.8 * np.asarray(data["A"])))
        dose = np.asarray(tilted.data.treatment)
        weighted_mean = float(np.average(dose, weights=tilted.data.weights))
        unweighted_mean = float(np.mean(dose))
        assert f"{weighted_mean:.4g}" != f"{unweighted_mean:.4g}"
        summary = tilted.summary()
        assert f"dose: mean {weighted_mean:.4g}," in summary
        assert f"dose: mean {unweighted_mean:.4g}," not in summary
        assert f"range [{dose.min():.3g}, {dose.max():.3g}]" in summary


class TestTheDiagnosticsMatchTheAxis:
    def test_support_dispatches_to_the_density_ratio_report(self) -> None:
        result = fit(shifts=SHIFTS)
        assert set(result.diagnostics.support()) == {"natural course", "+0.5"}

    def test_shift_support_reports_the_density_ratio_per_shift(self) -> None:
        result = fit(shifts=SHIFTS)
        report = result.diagnostics.support()
        assert set(report) == {"natural course", "+0.5"}
        # A shift of zero divides a density by itself, so its ratio is one everywhere and
        # it costs no effective sample size at all.
        assert report["natural course"].max_ratio == pytest.approx(1.0)
        assert report["natural course"].ess_ratio == pytest.approx(1.0)
        assert report["+0.5"].max_ratio > 1.0

    def test_support_dispatches_to_propensity_overlap_on_an_arm_fit(self) -> None:
        report = fit_binary().diagnostics.support()
        assert report.propensity_quantiles
