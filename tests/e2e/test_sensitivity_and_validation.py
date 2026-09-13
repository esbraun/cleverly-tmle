"""Sensitivity and validation analyses on real fits.

These modules are the reason the library exists in the shape it does, so the tests
check that they *say the right thing*, not merely that they return a frame.  Each
analysis is exercised on a case where the correct answer is known by construction:

* positivity diagnostics on data with and without an overlap problem;
* the robustness value on a fit with a *known* confounder withheld;
* the missingness tilt at ``gamma = 0``, which must reproduce the MAR estimate exactly;
* refutation tests, where a placebo treatment must yield no effect.

Every fixture above carries unit observation weights, so none of those cases can see a
diagnostic or a tilt that drops the weights.  ``weighted_missing_fit`` is the one that
can: :class:`TestPositivityUnderObservationWeights` and
:class:`TestTheTiltUnderObservationWeights` run on it, and each writes its expectation out
from the displayed formula rather than from the helper it is checking.
"""

from __future__ import annotations

import dataclasses
import warnings

import narwhals as nw
import numpy as np
import pytest
import sklearn.linear_model

from cleverly import AssessmentStatus, SuperLearner
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_missing_outcome,
    make_nonlinear_ate,
    make_shift_dose,
    make_weak_overlap,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, PositivityWarning
from cleverly.interventions import Shift
from cleverly.sensitivity.omitted_variable import benchmark
from cleverly.validation.refute import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    RelativeGaussianNoise,
    refute,
)
from tests.conftest import fast_tmle


@pytest.fixture(scope="module")
def good_overlap() -> object:
    frame, _ = make_linear_ate(n=1500, seed=71)
    return (
        fast_tmle(estimands=("ate", "att", "ey1", "ey0"))
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@pytest.fixture(scope="module")
def poor_overlap() -> object:
    frame, _ = make_weak_overlap(n=1500, seed=72)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        return fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()


def _weight_by_covariate_rank(column: np.ndarray) -> np.ndarray:
    """``np.linspace(0.5, 1.5, n)``, handed out in the order ``column`` ranks the rows.

    ``linspace`` gives the profile a fixed nonconstant shape, and the covariate order
    gives it something to say.  Handed out by row index it would be independent of
    everything the fit uses, so the weighted and the unweighted answer would differ only
    by sampling noise and the witnesses below could state no margin at all.
    """
    values = np.asarray(column, dtype=float)
    return np.linspace(0.5, 1.5, values.size)[np.argsort(np.argsort(values))]


def _kish(weights: np.ndarray) -> float:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``, written out.

    A sibling of the helper of the same name in
    :mod:`tests.unit.test_sensitivity_multi_arm`.  Six lines duplicated rather than
    shared, because the only home both modules could import from is ``tests/conftest.py``
    and this change does not own it.
    """
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def _top_share(weights: np.ndarray, fraction: float) -> float:
    """Share of the total weight held by the largest ``fraction`` of the rows."""
    w = np.asarray(weights, dtype=float)
    count = max(1, int(np.ceil(fraction * w.size)))
    return float(np.sort(w)[-count:].sum() / w.sum())


@pytest.fixture(scope="module")
def weighted_missing_fit() -> object:
    """One weighted fit, carrying missing outcomes, for the two weighted witnesses.

    Missingness costs the positivity witness nothing and the tilt cannot run without it,
    so the two share a fit rather than paying for one each.
    """
    frame, _ = make_missing_outcome(n=800, seed=11)
    weighted = frame.assign(obs_weight=_weight_by_covariate_rank(frame["W1"].to_numpy()))
    return (
        fast_tmle(estimands=("ate", "att", "atc", "ey1", "ey0"))
        .fit(
            weighted,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
            weights="obs_weight",
        )
        .single()
    )


class TestPositivity:
    def test_good_overlap_is_reported_as_adequate(self, good_overlap) -> None:
        report = good_overlap.diagnostics.support()
        # Asserted on the machine-readable tier rather than on a word in the prose. The
        # verdict sentence reports the effective-sample-size share it does not grade, so
        # matching its wording pins the wrong half of the contract.
        assert report.severity == "adequate"
        assert report.truncated["fraction"] == 0.0
        for arm in ("treated", "control"):
            # Almost all of the nominal sample size survives the reweighting.
            assert report.effective_sample_size[arm]["ratio"] > 0.8

    def test_poor_overlap_is_flagged(self, poor_overlap) -> None:
        report = poor_overlap.diagnostics.support()
        assert report.severity != "adequate"
        assert report.truncated["fraction"] > 0.0
        worst = min(ess["ratio"] for ess in report.effective_sample_size.values())
        # The weighted analysis is using far fewer observations than it appears to.
        assert worst < 0.6

    def test_the_effective_sample_size_never_exceeds_the_arm_size(self, poor_overlap) -> None:
        report = poor_overlap.diagnostics.support()
        for ess in report.effective_sample_size.values():
            assert ess["effective"] <= ess["n"] + 1e-9

    def test_weight_concentration_is_higher_under_poor_overlap(
        self, good_overlap, poor_overlap
    ) -> None:
        good = good_overlap.diagnostics.support().weight_share["treated"]["top_1pct"]
        poor = poor_overlap.diagnostics.support().weight_share["treated"]["top_1pct"]
        assert poor > good

    def test_the_report_renders_and_tabulates(self, good_overlap) -> None:
        report = good_overlap.diagnostics.support()
        text = report.summary()
        assert "Positivity" in text
        assert "VERDICT" in text
        frame = report.to_frame(good_overlap.data)
        assert set(nw.from_native(frame, eager_only=True)["group"].unique().to_list()) == {
            "overall",
            "treated",
            "control",
        }


class TestPositivityUnderObservationWeights:
    """The leverage a weighted point-treatment fit reports is the *product* of the two.

    The sibling of
    ``tests/e2e/test_ltmle.py::TestObservationWeights::test_the_diagnostics_fold_the_weights_into_the_leverage``,
    which makes this statement for the longitudinal estimator and cites
    ``result.diagnostics.support()`` as having made the same choice.  This is the check
    that holds the point-treatment side to it.  Both the module docstring of
    :mod:`cleverly.sensitivity.positivity` and
    ``docs/technical-reference/validation-methods.md`` state the claim in prose.

    Every check in :class:`TestPositivity` above passes unchanged on an implementation
    that ignores the observation weights, because none of its fits carries any.
    """

    def test_the_diagnostics_fold_the_weights_into_the_leverage(self, weighted_missing_fit) -> None:
        report = weighted_missing_fit.diagnostics.support()
        data = weighted_missing_fit.data
        lower, upper = weighted_missing_fit.config.g_bounds
        bounded = np.clip(weighted_missing_fit.nuisance.propensity.arm(1.0), lower, upper)
        treated = np.asarray(data.treatment == 1.0)

        for arm, mask, covariate in (
            ("treated", treated, 1.0 / bounded),
            ("control", ~treated, 1.0 / (1.0 - bounded)),
        ):
            clever = covariate[mask]
            leverage = clever * data.weights[mask]
            nominal = float(mask.sum())

            ess = report.effective_sample_size[arm]
            assert ess["effective"] == pytest.approx(_kish(leverage), abs=0)
            assert ess["ratio"] == pytest.approx(_kish(leverage) / nominal, abs=0)
            share = report.weight_share[arm]
            assert share["top_1pct"] == pytest.approx(_top_share(leverage, 0.01), abs=0)
            assert share["top_5pct"] == pytest.approx(_top_share(leverage, 0.05), abs=0)

            # The observation weighting materially changes the leverage rather than
            # merely carrying an unused array alongside it.
            assert not np.allclose(leverage, clever)
            # And it changes what is *reported*.  The smaller of the two arms moves by 15
            # units of effective sample size and by 0.017 of the top-5% share; the other
            # moves by 65 and by 0.037.
            assert abs(_kish(leverage) - _kish(clever)) > 10.0
            assert abs(_top_share(leverage, 0.05) - _top_share(clever, 0.05)) > 0.01

    def test_the_mechanism_rows_fold_them_in_too(self, weighted_missing_fit) -> None:
        """Both the fitted factor and the derived product, not only the propensity rows.

        The mechanism rows report the same leverage on the same scale, so a row that
        divided by the mechanism and stopped there would understate exactly the design a
        reader turned to this table to weigh.  The unweighted quantity is the control:
        every assertion above is silent about these rows, and the weights are a term that
        vanishes on any fixture that does not carry them.
        """
        result = weighted_missing_fit
        data, nuisance = result.data, result.nuisance
        lower = result.config.missingness_bound
        observation = np.clip(np.asarray(nuisance.missingness, dtype=float), lower, 1.0)
        product = nuisance.propensity.truncate(result.config.g_bounds).values * observation
        contributing = np.asarray(data.observed, dtype=bool)
        treated = np.asarray(data.treatment == 1.0)
        mechanisms = result.diagnostics.support().mechanisms

        for name, bounded in (
            ("P(Delta=1|A,W)", observation),
            ("P(A=a,Delta=1|W)", product),
        ):
            at_arm = np.where(treated, bounded[:, 1], bounded[:, 0])[contributing]
            clever = 1.0 / at_arm
            leverage = data.weights[contributing] / at_arm
            stats = mechanisms[name]

            assert stats["ess_ratio"] == pytest.approx(_kish(leverage) / leverage.size, abs=0)
            assert stats["top_1pct"] == pytest.approx(_top_share(leverage, 0.01), abs=0)
            assert stats["top_5pct"] == pytest.approx(_top_share(leverage, 0.05), abs=0)

            # The control: dropping the observation weights moves both reported numbers.
            assert abs(_kish(leverage) / leverage.size - _kish(clever) / clever.size) > 0.01
            assert abs(_top_share(leverage, 0.05) - _top_share(clever, 0.05)) > 0.005


class TestThePerGroupCovariateLeverageIsReported:
    """Each targeted group's own clever covariate, read as a load and not as a maximum.

    ``effective_sample_size`` is keyed by *arm* and built from ``1 / g`` alone, and
    ``mechanisms`` is keyed by mechanism name and reports *denominators*.  Neither one
    describes the weighting an ``att`` fit performs, because the ATT covariate is not an
    inverse arm probability: it divides by ``P(A = a)`` and reweights the reference arm by
    the propensity odds.  ``clever_covariate_max`` is keyed by group but reports a single
    number, unweighted and over every row.  So a conditional-arm fit had no load measure
    at all, and ``group_leverage`` is the row that supplies one.

    Every check below rebuilds the covariate from the formula in
    :func:`~cleverly.fluctuation.submodel.mean_submodel` and
    :func:`~cleverly.fluctuation.submodel.att_submodel` rather than from the helper under
    test, and each mutation control changes the *shape* of the load rather than its scale.
    Kish's effective sample size is scale invariant, as ``345e5d4`` recorded, so a control
    that only rescales the load passes on any implementation however wrong the row is.
    """

    @pytest.fixture(scope="class")
    def loaded_fit(self) -> object:
        """A weighted fit with missing outcomes, targeting one marginal and one conditional group.

        Three things vary here, and each is what lets one control below fail.  The design
        weights vary with ``W1``.  ``pi`` varies with ``A`` and ``W``, so the ``mean``
        covariate is not ``1 / g`` rescaled.  And the fit targets an ``att``, whose
        covariate loads the reference arm through the propensity odds rather than through
        its own arm.
        """
        frame, _ = make_missing_outcome(n=400, seed=13, strength=1.5)
        weighted = frame.assign(obs_weight=_weight_by_covariate_rank(frame["W1"].to_numpy()))
        return (
            fast_tmle(estimands=("ate", "att"))
            .fit(
                weighted,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
                weights="obs_weight",
            )
            .single()
        )

    @pytest.fixture(scope="class")
    def thin_overlap_fit(self) -> object:
        """A fit whose propensity reaches both truncation bounds, and only for some rows.

        ``g_bounds="auto"`` resolves to ``[0.036, 0.964]`` at this size and
        ``g_bounds_conditional`` to the fixed ``[0.025, 0.975]``, so the two clip
        different subsets of the rows.  That is the one condition under which rebuilding
        the ``att`` row at the wrong bound can report a different number: on a fit where
        neither bound binds the two rebuilds are the same array.
        """
        frame, _ = make_weak_overlap(n=500, seed=72)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            return (
                fast_tmle(estimands=("ate", "att")).fit(frame, outcome="Y", treatment="A").single()
            )

    @pytest.fixture(scope="class")
    def reversed_reference_fit(self) -> object:
        """The same law as ``loaded_fit``, targeted against arm 1 rather than arm 0.

        ``reference=1`` is the one thing that differs, and it is the only condition under
        which the rebuild's ``reference=`` argument is observable: everywhere else in this
        class the fit's reference is the lowest arm, which is also the fallback.
        """
        frame, _ = make_missing_outcome(n=400, seed=13, strength=1.5)
        weighted = frame.assign(obs_weight=_weight_by_covariate_rank(frame["W1"].to_numpy()))
        return (
            fast_tmle(estimands=("ate", "att"), reference=1)
            .fit(
                weighted,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
                weights="obs_weight",
            )
            .single()
        )

    @staticmethod
    def _pieces(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(contributing rows, treated indicator, bounded missingness)`` for one fit.

        The rows are the ones whose residual the covariate multiplies.  ``mean_submodel``
        divides by ``pi`` but carries no ``Delta`` indicator, so an unmasked ratio would
        average in rows the estimating equation never weights.
        """
        data, nuisance = result.data, result.nuisance
        contributing = np.asarray(data.observed, dtype=bool)
        treated = np.asarray(data.treatment == 1.0)
        missingness = (
            np.ones((data.n, 2))
            if nuisance.missingness is None
            else np.clip(
                np.asarray(nuisance.missingness, dtype=float),
                result.config.missingness_bound,
                1.0,
            )
        )
        return contributing, treated, missingness

    def _mean_covariate(self, result) -> np.ndarray:
        """``1 / (g_a(W) pi_a(W) q_a(W))`` at each unit's observed arm, over every row."""
        _, treated, pi = self._pieces(result)
        g = result.nuisance.propensity.bounded(result.config.g_bounds)
        return 1.0 / (np.where(treated, g[:, 1], g[:, 0]) * np.where(treated, pi[:, 1], pi[:, 0]))

    def _att_covariate(self, result, *, bounds=None, reference: float = 0.0) -> np.ndarray:
        """``(1{A=a}/pi_a - 1{A=r}/pi_r g_a/g_r) / P(A = a)``, signed, over every row.

        ``bounds`` and ``reference`` are arguments so that two of the controls below can
        rebuild the same formula at the wrong one.  The defaults are what the fit used.
        """
        contrast = 1.0 - reference
        _, treated, pi = self._pieces(result)
        g = result.nuisance.propensity.bounded(bounds or result.config.g_bounds_conditional)
        share = result.data.arm_fractions[int(contrast)]
        at_contrast = treated if contrast == 1.0 else ~treated
        own = 1.0 / (share * pi[:, int(contrast)])
        against = (g[:, int(contrast)] / g[:, int(reference)]) / (share * pi[:, int(reference)])
        return np.where(at_contrast, own, -against)

    def _selected_artifact(self, result, group: str) -> tuple[np.ndarray, int]:
        """Exact score-load column selected by the report's concentration rule."""
        artifact = result.fluctuations[group].absolute_score_weights
        assert artifact is not None
        loads = np.asarray(artifact)
        ratios = np.array([_kish(loads[:, j]) / loads.shape[0] for j in range(loads.shape[1])])
        selected = int(np.argmin(ratios))
        return loads[:, selected], selected

    def test_the_mean_row_uses_one_exact_fitted_score_equation(self, loaded_fit) -> None:
        """Every row value comes from the most concentrated stored score column."""
        report = loaded_fit.diagnostics.support()
        load, selected = self._selected_artifact(loaded_fit, "mean")
        row = report.group_leverage["mean"]

        assert row["equation"] == loaded_fit.fluctuations["mean"].names[selected]
        assert row["n_targeted"] == float(load.size)
        assert row["n_total"] == float(loaded_fit.data.n)
        assert row["effective"] == pytest.approx(_kish(load), abs=0)
        assert row["targeted_ratio"] == pytest.approx(_kish(load) / load.size, abs=0)
        assert row["total_ratio"] == pytest.approx(_kish(load) / loaded_fit.data.n, abs=0)
        assert row["top_1pct"] == pytest.approx(_top_share(load, 0.01), abs=0)
        assert row["top_5pct"] == pytest.approx(_top_share(load, 0.05), abs=0)
        assert row["max_load"] == pytest.approx(float(load.max()), abs=0)
        assert row["zero_load"] == float(np.count_nonzero(load == 0.0))

    def test_the_att_row_uses_the_exact_fitted_score_artifact(self, loaded_fit) -> None:
        """The diagnostic consumes fitted score rows instead of rebuilding a submodel."""
        report = loaded_fit.diagnostics.support()
        load, selected = self._selected_artifact(loaded_fit, "att")
        row = report.group_leverage["att"]

        assert row["equation"] == loaded_fit.fluctuations["att"].names[selected]
        assert row["n_targeted"] == float(load.size)
        assert row["effective"] == pytest.approx(_kish(load), abs=0)
        assert row["targeted_ratio"] == pytest.approx(_kish(load) / load.size, abs=0)
        assert row["top_1pct"] == pytest.approx(_top_share(load, 0.01), abs=0)
        assert row["top_5pct"] == pytest.approx(_top_share(load, 0.05), abs=0)
        assert row["max_load"] == pytest.approx(float(load.max()), abs=0)
        assert row["zero_load"] == float(np.count_nonzero(load == 0.0))

    def test_dropping_the_design_weights_reports_a_different_row(self, loaded_fit) -> None:
        """Control: the load without ``data.weights``.

        This can fail because the weight profile is a function of ``W1``, and ``W1`` also
        drives ``g`` and ``pi``.  So dropping it does not divide every row by one number;
        it moves the heavy rows relative to the light ones.  A profile handed out by row
        index would rescale the load and Kish's ratio would not notice.
        """
        report = loaded_fit.diagnostics.support()
        contributing, _, _ = self._pieces(loaded_fit)
        covariate = self._att_covariate(loaded_fit, reference=loaded_fit.config.reference_arm)

        for group, values in (
            ("mean", self._mean_covariate(loaded_fit)),
            ("att", np.abs(covariate)),
        ):
            unweighted = values[contributing]
            row = report.group_leverage[group]
            # Measured: the mean row moves by 51 units of effective sample size and the
            # att row by 51 as well, on 309 targeted rows.
            assert abs(_kish(unweighted) - row["effective"]) > 20.0
            assert abs(_top_share(unweighted, 0.05) - row["top_5pct"]) > 0.02

    def test_the_arm_tables_inverse_propensity_is_not_this_covariate(self, loaded_fit) -> None:
        """Control: ``1 / g`` at the observed arm, which is what the arm table weights by.

        This can fail because ``pi`` depends on ``A`` and on ``W``, so dividing by it
        multiplies each row by its own factor rather than by a common one.  On a
        complete-outcome fit the two arrays coincide, and this control would be blind
        there, which is why the fixture carries missing outcomes.
        """
        report = loaded_fit.diagnostics.support()
        contributing, treated, _ = self._pieces(loaded_fit)
        g = loaded_fit.nuisance.propensity.bounded(loaded_fit.config.g_bounds)
        at_arm = np.where(treated, g[:, 1], g[:, 0])
        arm_weight = loaded_fit.data.weights[contributing] * (1.0 / at_arm)[contributing]
        row = report.group_leverage["mean"]

        # Measured: 265 effective rows against the 200 the mechanism-bearing covariate
        # leaves, and a top-5% share of 0.104 against 0.168.
        assert abs(_kish(arm_weight) - row["effective"]) > 25.0
        assert abs(_top_share(arm_weight, 0.05) - row["top_5pct"]) > 0.02
        assert abs(float(arm_weight.max()) - row["max_load"]) > 1.0

    def test_the_att_row_is_built_at_the_conditional_bound(self, thin_overlap_fit) -> None:
        """ATT reports the bound and clipped units that its fitted score actually used."""
        result = thin_overlap_fit
        assert result.config.g_bounds != result.config.g_bounds_conditional
        row = result.diagnostics.support().group_leverage["att"]
        conditional = result.nuisance.propensity.truncate(result.config.g_bounds_conditional).units
        marginal = result.nuisance.propensity.truncate(result.config.g_bounds).units

        assert (row["lower_bound"], row["upper_bound"]) == result.config.g_bounds_conditional
        assert row["clipped_count"] == float(np.count_nonzero(conditional))
        assert row["clipped_fraction"] == pytest.approx(float(np.mean(conditional)), abs=0)
        assert np.count_nonzero(conditional) != np.count_nonzero(marginal)
        assert row["clipped_count"] != float(np.count_nonzero(marginal))

    def test_the_att_row_is_built_against_the_fits_own_reference_arm(self, loaded_fit) -> None:
        """Control: the ``att`` covariate contrasted against arm 1 instead of arm 0.

        This can fail because swapping the reference swaps which arm carries ``own`` and
        which carries ``against``, and inverts the propensity odds.  The result is a
        different function of the row, not a multiple of the same one, so the ordering of
        the loads changes and Kish's ratio moves with it.

        It is a control on the *formula* and not on the argument that carries the
        reference.  This fixture's reference is arm 0, which is also what
        ``_reference_index(None, ...)`` returns, so dropping ``reference=`` from the
        rebuild leaves this row where it is.
        ``test_a_non_default_reference_arm_is_the_one_the_row_is_rebuilt_at`` is the case
        that separates the two.
        """
        report = loaded_fit.diagnostics.support()
        contributing, _, _ = self._pieces(loaded_fit)
        assert loaded_fit.config.reference_arm == 0.0
        flipped = self._att_covariate(loaded_fit, reference=1.0)
        load = loaded_fit.data.weights[contributing] * np.abs(flipped)[contributing]
        row = report.group_leverage["att"]

        # Measured: 246 effective rows against 144, and a top-5% share of 0.130 against
        # 0.215. The wrong reference reports a comfortable row for a strained fit.
        assert abs(_kish(load) - row["effective"]) > 50.0
        assert abs(_top_share(load, 0.05) - row["top_5pct"]) > 0.05

    def test_a_non_default_reference_arm_is_the_one_the_row_is_rebuilt_at(
        self, reversed_reference_fit
    ) -> None:
        """The deliberate-mutation control on ``reference=`` itself, not on the formula.

        ``_group_submodel`` passes the fit's own ``reference_arm`` to the rebuild.  Drop
        that argument and ``_reference_index`` falls back to the lowest arm, which is 0 on
        every other fixture in this class and so is the same array they already assert.
        Only a fit whose reference is **not** the lowest arm can tell the two apart, and
        that is what this one is: ``reference=1`` makes ``att`` the effect among the units
        in arm 0, whose covariate reweights arm 1 by the propensity odds rather than the
        other way round.

        Measured over seeds 11, 13, 17, 21 and 29 at ``n=400``. The reported row matches
        the rebuild at arm 1 exactly, and the rebuild at arm 0 differs by 66 to 123
        effective rows of about 300, by 0.069 to 0.105 in the top-5% share and by 5.8 to
        31 in the largest single load.  The windows hold across that whole range.
        """
        result = reversed_reference_fit
        report = result.diagnostics.support()
        contributing, _, _ = self._pieces(result)
        weights = result.data.weights[contributing]
        row = report.group_leverage["att"]

        assert result.config.reference_arm == 1.0
        at_fit = weights * np.abs(self._att_covariate(result, reference=1.0))[contributing]
        at_default = weights * np.abs(self._att_covariate(result, reference=0.0))[contributing]

        assert row["effective"] == pytest.approx(_kish(at_fit), abs=0)
        assert row["top_1pct"] == pytest.approx(_top_share(at_fit, 0.01), abs=0)
        assert row["top_5pct"] == pytest.approx(_top_share(at_fit, 0.05), abs=0)
        assert row["max_load"] == pytest.approx(float(at_fit.max()), abs=0)

        assert abs(_kish(at_default) - row["effective"]) > 50.0
        assert abs(_top_share(at_default, 0.05) - row["top_5pct"]) > 0.05
        assert abs(float(at_default.max()) - row["max_load"]) > 5.0

    def test_the_rows_the_equation_never_weights_stay_out_of_the_ratio(self, loaded_fit) -> None:
        """The artifact's score mask defines n_targeted; n_total remains visible."""
        report = loaded_fit.diagnostics.support()
        contributing, _, _ = self._pieces(loaded_fit)
        assert not contributing.all(), "the fixture must carry rows the equation drops"
        for group in ("mean", "att"):
            artifact = loaded_fit.fluctuations[group].absolute_score_weights
            assert artifact is not None
            row = report.group_leverage[group]
            assert row["n_targeted"] == float(artifact.shape[0]) == float(contributing.sum())
            assert row["n_total"] == float(loaded_fit.data.n)
            assert row["n_targeted"] < row["n_total"]
            assert row["total_ratio"] == pytest.approx(row["effective"] / row["n_total"], abs=0)

    def test_a_group_can_retain_far_less_than_any_arm_does(self, loaded_fit) -> None:
        """The nonzero witness: every arm looks comfortable and the ``att`` group does not.

        This is the claim the slice makes, so it needs a law where the two disagree rather
        than a fit where both are fine.  Measured across seeds 11-17 at this size and
        strength: the narrowest arm retains 0.75 to 0.86 of its rows while the ``att``
        group retains 0.36 to 0.68, and the gap runs 0.17 to 0.39. The windows below hold
        across that whole range rather than around the one seed the fixture uses.
        """
        report = loaded_fit.diagnostics.support()
        narrowest_arm = min(ess["ratio"] for ess in report.effective_sample_size.values())
        conditional = report.group_leverage["att"]["targeted_ratio"]

        assert report.truncated["fraction"] == 0.0
        assert narrowest_arm > 0.70
        assert conditional < 0.70
        assert narrowest_arm - conditional > 0.15

    def test_group_concentration_is_not_equated_with_mechanism_ess(self, loaded_fit) -> None:
        """A selected score equation and a pooled denominator have different units."""
        report = loaded_fit.diagnostics.support()
        group_ratio = report.group_leverage["mean"]["targeted_ratio"]
        mechanism_ratio = report.mechanisms["P(A=a,Delta=1|W)"]["ess_ratio"]
        assert group_ratio != pytest.approx(mechanism_ratio, abs=1e-6)
        assert report.composed_excluded == ("att",)

    def test_the_keys_are_the_groups_the_fit_targeted(
        self, poor_overlap, good_overlap, loaded_fit
    ) -> None:
        """One row per fluctuation, on a mean-only fit and on a mean-and-att fit.

        Keyed exactly as ``clever_covariate_max`` is, because a reader compares the
        maximum and the load of one group side by side and a table with its own key set
        would silently pair the wrong two.
        """
        for result, groups in (
            (poor_overlap, {"mean"}),
            (good_overlap, {"mean", "att"}),
            (loaded_fit, {"mean", "att"}),
        ):
            report = result.diagnostics.support()
            assert set(result.fluctuations) == groups
            assert report.group_leverage.keys() == report.clever_covariate_max.keys()
            assert report.group_leverage.keys() == result.fluctuations.keys()

    def test_the_table_renders_beside_the_maxima_it_keeps(self, loaded_fit) -> None:
        """The table names equations, denominators, exact bounds, and descriptive units."""
        report = loaded_fit.diagnostics.support()
        summary = report.summary()
        lines = summary.splitlines()
        header = next(line for line in lines if line.startswith("group "))

        assert "equation" in header and "target rows" in header
        assert header.index("Kish-equivalent rows") < header.index("max |w h|")
        assert "g bound" in header and "clipped" in header
        assert "residual-multiplier concentration, not residual contributions" in summary
        for group, load in report.group_leverage.items():
            row = next(line for line in lines if line.startswith(f"{group} "))
            assert str(load["equation"]) in row
            assert f"{load['max_load']:.4g}" in row
            assert f"{load['n_targeted']:.0f}/{load['n_total']:.0f}" in row
        # The composed note now says where the estimands it refuses to cover are covered.
        # The refusal is unchanged: this fit forms the product for its `ate` and never for
        # its `att`, so the derived row is present and does not describe the `att`.
        assert "P(A=a,Delta=1|W) does not describe att" in summary
        assert "Their score-weight load is in the group table above" in summary

    def test_the_verdict_names_the_narrowest_group_and_keeps_the_arm_sentence(
        self, loaded_fit
    ) -> None:
        """Reported beside the arm share, not instead of it.

        The two are different weightings and the narrower is not always the arm's, so the
        verdict carries both. Neither is graded: this fit clips nothing and reads
        ``adequate`` with a group retaining under half of the rows it weights.
        """
        report = loaded_fit.diagnostics.support()
        verdict = report.verdict()
        group, narrow = min(
            report.group_leverage.items(), key=lambda item: item[1]["targeted_ratio"]
        )
        narrowest_arm = min(ess["ratio"] for ess in report.effective_sample_size.values())

        assert f"group {group!r}, equation {narrow['equation']!r}" in verdict
        assert f"{narrow['effective']:.1f} Kish-equivalent rows" in verdict
        assert f"Kish-equivalent weight count of {narrowest_arm:.0%}" in verdict
        assert "not estimator effective sample size" in verdict
        assert report.truncated["fraction"] == 0.0
        assert narrow["targeted_ratio"] < 0.5
        assert report.severity == "adequate"

    def test_a_complete_outcome_fit_reports_no_mechanism_and_the_same_tier(
        self, good_overlap
    ) -> None:
        """Non-regression: the new table is added beside the old ones and moves neither.

        ``good_overlap`` has no fitted factor beside ``g``, so there is no derived row and
        nothing for a group to be outside of, even though it targets an ``att``. Its
        every load row is still filled, and its tier is what it was.
        """
        report = good_overlap.diagnostics.support()
        assert report.mechanisms == {}
        assert report.composed_excluded == ()
        assert report.severity == "adequate"
        assert set(report.group_leverage) == {"mean", "att"}
        for group, load in report.group_leverage.items():
            assert np.isfinite(load["targeted_ratio"])
            selected = good_overlap.fluctuations[group].names.index(str(load["equation"]))
            artifact = good_overlap.fluctuations[group].absolute_score_weights
            assert artifact is not None
            assert load["zero_load"] == float(np.count_nonzero(artifact[:, selected] == 0.0))

    def test_the_table_survives_a_save_and_a_load(self, loaded_fit, tmp_path) -> None:
        """The report is recomputed from the restored result, so the rebuild has to travel.

        ``_covariate_leverage`` reads the data, the nuisance estimates and the config
        rather than the estimator, which is what lets a reloaded fit answer at all. Exact
        equality, because a restored fit that rebuilt the covariate from anything else
        would land nearby rather than on the number.
        """
        import cleverly

        path = tmp_path / "loaded-fit.joblib"
        loaded_fit.save(path)
        restored = cleverly.load(path)
        assert restored.diagnostics.support().group_leverage == (
            loaded_fit.diagnostics.support().group_leverage
        )

    def test_the_combined_report_retains_the_same_table(self, loaded_fit) -> None:
        """``assess()`` retains the report it interpreted, rather than a summary of it."""
        report = loaded_fit.diagnostics.support()
        assert loaded_fit.assess().report("support").group_leverage == report.group_leverage


class TestTruncationCurve:
    def test_the_curve_is_flat_when_overlap_is_good(self, good_overlap) -> None:
        curve = nw.from_native(
            good_overlap.diagnostics.truncation_curve([0.001, 0.01, 0.05], estimands=["ate"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        # No propensity is near the boundary, so truncation cannot bite.
        assert float(values.max() - values.min()) < 1e-9

    def test_the_curve_moves_when_overlap_is_poor(self, poor_overlap) -> None:
        headline = poor_overlap.psi("ate")
        curve = nw.from_native(
            poor_overlap.diagnostics.truncation_curve([0.001, 0.01, 0.05, 0.15], estimands=["ate"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        errors = np.array(curve["std_err"].to_list())
        truncated = np.array(curve["truncated_fraction"].to_list())
        assert float(values.max() - values.min()) > 1e-3
        # Tighter bounds truncate more units and buy variance with bias.
        assert np.all(np.diff(truncated) > 0)
        assert errors[0] > errors[-1]
        # The estimate moves because the finite-sample procedure changes. The requested
        # parameter does not: every row remains the fitted ATE, and the result still
        # records that same estimand after the retargeting sweep.
        assert set(curve["estimand"].to_list()) == {"ate"}
        assert poor_overlap.config.estimands == ("ate",)
        assert tuple(poor_overlap.estimates) == ("ate",)
        assert poor_overlap.psi("ate") == headline

    def test_the_fitted_bound_is_marked(self, good_overlap) -> None:
        curve = nw.from_native(
            good_overlap.diagnostics.truncation_curve(estimands=["ate"]), eager_only=True
        )
        assert sum(curve["is_fitted_bound"].to_list()) >= 1

    def test_an_invalid_bound_is_refused(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            good_overlap.diagnostics.truncation_curve([0.7])

    def test_an_unreported_estimand_is_refused_before_any_retargeting(
        self, poor_overlap, monkeypatch
    ) -> None:
        """A name the fit never reported has no fitted pair for a row to reference.

        The sweep used to widen the name to its registered target, retarget the whole
        grid, and only then fail on a bare ``KeyError``. That named neither the request
        nor the parameters the fit does carry, and it charged the caller a full sweep for
        a typing mistake. The stub below fails the test if any retargeting happens.
        """

        def never(*args: object, **kwargs: object) -> object:
            raise AssertionError("the sweep retargeted before it checked estimands=")

        monkeypatch.setattr(type(poor_overlap.estimator), "retarget", never)
        with pytest.raises(CapabilityError) as refusal:
            poor_overlap.diagnostics.truncation_curve(estimands=["att"])

        assert "['att']" in str(refusal.value)
        assert "['ate']" in str(refusal.value)

    def test_an_empty_selection_is_refused_before_any_retargeting(
        self, poor_overlap, monkeypatch
    ) -> None:
        """An empty request used to sweep the whole grid and then die at ``rows[0]``.

        It emits no row, so it has nothing to report and no reason to pay for a sweep.
        The refusal joins the unreported name above, at the same point and in the same
        form, rather than arriving as a bare ``IndexError`` afterwards.
        """

        def never(*args: object, **kwargs: object) -> object:
            raise AssertionError("the sweep retargeted before it checked estimands=")

        monkeypatch.setattr(type(poor_overlap.estimator), "retarget", never)
        with pytest.raises(CapabilityError) as refusal:
            poor_overlap.diagnostics.truncation_curve(bounds=[0.05], estimands=[])

        assert "selected no parameter" in str(refusal.value)
        assert "['ate']" in str(refusal.value)

    def test_a_continuous_treatment_withholds_the_truncated_fraction(self) -> None:
        """No arms, no share of units the bound moved. The column says so.

        A shift fit's propensity is ``(n, 0)``: the mechanism it truncates is a density
        ratio, and no column of it is a treatment probability the ``g_bounds`` pair clips.
        ``np.any`` over that empty axis is ``False``, so counting units returned a
        well-formed ``0.0`` beside a real resolved ``fitted_lower_bound`` -- a positive
        claim that the bound moved no unit, about a mechanism with no unit to move. The
        previous code averaged the empty matrix and produced ``nan`` with a
        ``RuntimeWarning``; the answer withheld is the same, and the warning is not.
        """
        frame, _ = make_shift_dose(n=300, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = (
                fast_tmle(shifts=(Shift(0.0, cap=None), Shift(0.5, cap=5.0)))
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
            curve = nw.from_native(
                result.diagnostics.truncation_curve(bounds=[0.01, 0.05]), eager_only=True
            )

        assert result.nuisance.propensity.values.shape == (300, 0)
        assert np.all(np.isnan(np.array(curve["truncated_fraction"].to_list())))
        # The rest of the row is answered, so the withheld column is a judgement about
        # this mechanism and not a curve that failed to run.
        assert np.all(np.isfinite(np.array(curve["psi"].to_list())))
        assert set(curve["fitted_lower_bound"].to_list()) == {result.config.g_bounds[0]}


class TestOmittedVariableBias:
    def test_the_bound_grows_with_the_assumed_confounding(self, good_overlap) -> None:
        weak = good_overlap.sensitivity.omitted_confounding("ate", cf_y=0.01, cf_d=0.01)
        strong = good_overlap.sensitivity.omitted_confounding("ate", cf_y=0.10, cf_d=0.10)
        assert strong.bias > weak.bias
        assert strong.lower < weak.lower
        assert strong.upper > weak.upper

    def test_zero_confounding_reproduces_the_point_estimate(self, good_overlap) -> None:
        bounds = good_overlap.sensitivity.omitted_confounding("ate", cf_y=0.0, cf_d=0.0)
        assert bounds.lower == pytest.approx(bounds.psi)
        assert bounds.upper == pytest.approx(bounds.psi)

    def test_the_robustness_value_is_where_the_bound_reaches_the_null(self, good_overlap) -> None:
        values = good_overlap.sensitivity.robustness_value("ate")
        rv = values["rv"]
        assert 0.0 < rv < 1.0
        # By definition, setting cf_y = cf_d = RV must put the bound at zero.
        at_rv = good_overlap.sensitivity.omitted_confounding("ate", cf_y=rv, cf_d=rv)
        edge = at_rv.lower if at_rv.psi > 0 else at_rv.upper
        assert edge == pytest.approx(0.0, abs=1e-4)

    def test_the_confidence_robustness_value_is_the_smaller_one(self, good_overlap) -> None:
        values = good_overlap.sensitivity.robustness_value("ate")
        # It takes less confounding to make an interval touch the null than to move the
        # point estimate there.
        assert values["rva"] < values["rv"]

    def test_a_weaker_effect_has_a_smaller_robustness_value(self) -> None:
        strong_frame, _ = make_linear_ate(n=1500, seed=73, effect=2.0)
        weak_frame, _ = make_linear_ate(n=1500, seed=73, effect=0.2)
        strong = (
            fast_tmle(estimands=("ate",)).fit(strong_frame, outcome="Y", treatment="A").single()
        )
        weak = fast_tmle(estimands=("ate",)).fit(weak_frame, outcome="Y", treatment="A").single()
        assert (
            weak.sensitivity.robustness_value("ate")["rv"]
            < strong.sensitivity.robustness_value("ate")["rv"]
        )

    def test_poor_overlap_inflates_the_maximal_bias(self, good_overlap, poor_overlap) -> None:
        # nu^2 is the second moment of the Riesz representer, so it blows up exactly when
        # overlap fails -- the same quantity that drives the clever covariate.
        assert (
            poor_overlap.sensitivity.elements("ate").nu2
            > good_overlap.sensitivity.elements("ate").nu2
        )

    @pytest.mark.parametrize("estimand", ["ate", "ey1", "ey0", "att"])
    def test_every_linear_estimand_is_supported(self, good_overlap, estimand: str) -> None:
        elements = good_overlap.sensitivity.elements(estimand)
        assert elements.sigma2 > 0
        assert elements.nu2 > 0
        assert elements.max_bias == pytest.approx(np.sqrt(elements.sigma2 * elements.nu2))

    def test_the_doubly_robust_and_plugin_nu2_agree(self, good_overlap) -> None:
        doubly_robust = good_overlap.sensitivity.elements("ate", nu2_estimator="doubly_robust")
        plugin = good_overlap.sensitivity.elements("ate", nu2_estimator="plugin")
        # Both estimate E[alpha^2]; they differ only by sampling error.
        assert doubly_robust.nu2 == pytest.approx(plugin.nu2, rel=0.1)

    def test_a_ratio_estimand_is_refused_with_a_pointer_to_the_evalue(self) -> None:
        frame, _ = make_binary_outcome(n=800, seed=74)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match=r"sensitivity\.evalue"):
            result.sensitivity.omitted_confounding("rr")

    def test_benchmarking_a_real_confounder_reports_its_strength(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=75)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        # W1 drives both the outcome and treatment in this process, so dropping it must
        # register as a substantial confounder on both the cf_y and cf_d scales.
        benchmark = result.sensitivity.benchmark(["W1"], estimand="ate")
        assert benchmark.cf_y > 0.05
        assert benchmark.cf_d > 0.0
        assert abs(benchmark.delta_psi) > 0.0
        assert benchmark.sigma2_short > benchmark.sigma2_long
        assert "W1" in benchmark.summary()

    def test_benchmarking_a_pure_noise_covariate_reports_almost_nothing(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=76)
        noisy = frame.assign(noise=np.random.default_rng(0).normal(size=len(frame)))
        result = fast_tmle(estimands=("ate",)).fit(noisy, outcome="Y", treatment="A").single()
        benchmark = result.sensitivity.benchmark(["noise"], estimand="ate")
        assert benchmark.cf_y < 0.02
        assert benchmark.cf_d < 0.05

    def test_a_benchmark_repeats_from_the_seed_it_reports(self) -> None:
        """A benchmark refits, so it carries the reproducibility question a refutation does.

        The result is cached on the fit and survives ``save``, so a benchmark that named no
        seed would persist a number nobody could recompute.  These call the free function,
        because ``sensitivity.benchmark`` memoises and two facade calls return one object.
        """
        frame, _ = make_linear_ate(n=500, seed=88)
        unseeded = (
            fast_tmle(estimands=("ate",), random_state=None)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert unseeded.estimator.random_state is None
        first = benchmark(unseeded, ["W1"], estimand="ate")
        assert isinstance(first.random_state, int)
        replay = benchmark(unseeded, ["W1"], estimand="ate", random_state=first.random_state)
        assert replay.psi_short == first.psi_short
        assert replay.cf_y == first.cf_y
        # The seed applies to a copy, exactly as it does for a refutation.
        assert unseeded.estimator.random_state is None

    def test_a_benchmark_of_a_seeded_fit_reports_that_fits_seed(self) -> None:
        frame, _ = make_linear_ate(n=500, seed=88)
        seeded = (
            fast_tmle(estimands=("ate",), random_state=11)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert benchmark(seeded, ["W1"], estimand="ate").random_state == 11

    def test_a_zero_seed_still_overrides_the_fit_for_a_benchmark(self) -> None:
        """``random_state=0`` is falsy, so the resolution has to read ``is None``."""
        frame, _ = make_linear_ate(n=500, seed=88)
        seeded = (
            fast_tmle(estimands=("ate",), random_state=11)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert benchmark(seeded, ["W1"], estimand="ate", random_state=0).random_state == 0

    def test_the_contour_grid_is_monotone(self, good_overlap) -> None:
        grid = nw.from_native(good_overlap.sensitivity.contour("ate", grid_size=5), eager_only=True)
        assert len(grid) == 25
        # The lower bound falls as either sensitivity parameter grows.
        at_origin = [
            row
            for row in zip(
                grid["cf_d"].to_list(), grid["cf_y"].to_list(), grid["value"].to_list(), strict=True
            )
            if row[0] == 0.0 and row[1] == 0.0
        ]
        assert at_origin[0][2] == pytest.approx(good_overlap.psi("ate"))
        assert min(grid["value"].to_list()) < good_overlap.psi("ate")


class TestEValue:
    @pytest.fixture(scope="class")
    def binary_fit(self) -> object:
        """One binary-outcome fit for the three tests that only need a ratio to read.

        Nothing asserted below turns on the sample or the seed -- each test reads a
        different field off the same kind of result -- so three fits would be three
        copies of one.
        """
        frame, _ = make_binary_outcome(n=2000, seed=77)
        return fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A").single()

    def test_a_binary_outcome_uses_the_risk_ratio_directly(self, binary_fit) -> None:
        evalue = binary_fit.sensitivity.evalue("rr")
        assert not evalue.approximate
        assert evalue.risk_ratio == pytest.approx(binary_fit.psi("rr"))
        assert evalue.point > evalue.limit >= 1.0

    def test_the_default_prefers_the_risk_ratio(self, binary_fit) -> None:
        assert binary_fit.sensitivity.evalue().estimand == "rr"

    def test_a_continuous_outcome_is_converted_and_flagged(self, good_overlap) -> None:
        evalue = good_overlap.sensitivity.evalue("ate")
        assert evalue.approximate
        assert "Chinn" in evalue.note
        assert evalue.point > 1.0

    def test_the_odds_ratio_conversion_is_flagged_for_common_outcomes(self, binary_fit) -> None:
        evalue = binary_fit.sensitivity.evalue("or")
        assert evalue.approximate
        assert "common outcomes" in evalue.note
        assert "understates" not in evalue.note
        assert evalue.risk_ratio == pytest.approx(np.sqrt(binary_fit.psi("or")))


class TestMissingnessTilt:
    @pytest.fixture(scope="class")
    def missing_fit(self) -> object:
        frame, _ = make_missing_outcome(n=1500, seed=80)
        return (
            fast_tmle(estimands=("ate", "ey1"))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            )
            .single()
        )

    def test_no_tilt_reproduces_the_reported_estimate(self, missing_fit) -> None:
        curve = nw.from_native(
            missing_fit.sensitivity.missingness([0.0], estimands=["ate"]), eager_only=True
        )
        # gamma = 0 is the MAR analysis, so the curve must pass exactly through the
        # reported point estimate. Anything else means the tilt is mis-parameterised.
        assert float(curve["psi"][0]) == pytest.approx(missing_fit.psi("ate"), rel=1e-12)
        assert bool(curve["is_mar"][0])

    def test_the_tilt_moves_the_estimate_monotonically(self, missing_fit) -> None:
        curve = nw.from_native(
            missing_fit.sensitivity.missingness([-1.0, -0.5, 0.0, 0.5, 1.0], estimands=["ey1"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        # The direction, not just that there is one.  `all(diff > 0) or all(diff < 0)`
        # accepted either, so a flipped gamma passed it: exactly the kind of error a
        # sign-blind check cannot expose.
        #
        # Which direction is read off the derivation rather than off a run.  The tilt is
        # Q_miss = expit(logit(Q*) + gamma), mixed in at weight (1 - pi_a), and the module
        # docstring states what positive gamma means: the unobserved outcomes were
        # systematically *higher*.  So E[Y^1] increases with gamma, strictly, wherever
        # any row is unobserved -- which the `missing_fit` fixture guarantees.
        assert np.all(np.diff(values) > 0), values

    def test_the_tipping_point_is_reported_or_absent(self, missing_fit) -> None:
        tipping = missing_fit.sensitivity.tipping_gamma("ate")
        # This process has a substantial effect, so no plausible tilt nulls it.
        assert tipping is None or abs(tipping) > 1.0

    def test_the_confidence_interval_tips_at_the_nearest_limit(self, missing_fit) -> None:
        """The signed CI objective reaches the null inside an asymmetric search."""
        direction = {0: 0.0, 1: -1.0}
        tipping = missing_fit.sensitivity.tipping_gamma(
            "ate", search=(-0.5, 4.0), use_ci=True, arm_gamma=direction
        )

        assert tipping is not None
        assert 2.0 < tipping < 4.0
        curve = missing_fit.sensitivity.missingness(
            [tipping], estimands=["ate"], arm_gamma=direction
        )
        assert float(curve["ci_lower"].iloc[0]) == pytest.approx(0.0, abs=1e-4)

    def test_the_upper_confidence_limit_can_tip_from_below(self, missing_fit) -> None:
        """A null above the baseline interval keeps the opposite signed boundary."""
        direction = {0: -1.0, 1: 1.0}
        tipping = missing_fit.sensitivity.tipping_gamma(
            "ate", null_hypothesis=2.0, use_ci=True, arm_gamma=direction
        )

        assert tipping is not None
        curve = missing_fit.sensitivity.missingness(
            [tipping], estimands=["ate"], arm_gamma=direction
        )
        assert float(curve["ci_upper"].iloc[0]) == pytest.approx(2.0, abs=1e-4)

    def test_the_ci_tipping_search_handles_baseline_and_no_crossing(self, missing_fit) -> None:
        assert (
            missing_fit.sensitivity.tipping_gamma(
                "ate", null_hypothesis=missing_fit.psi("ate"), use_ci=True
            )
            == 0.0
        )
        assert (
            missing_fit.sensitivity.tipping_gamma("ate", null_hypothesis=100.0, use_ci=True) is None
        )

    def test_a_nonmonotone_direction_finds_the_nearest_interior_crossing(self, missing_fit) -> None:
        """Equal endpoint signs do not hide two crossings inside the bracket."""
        direction = {0: -190.71029260054688, 1: -161.20708526870136}
        null = 1.5587107797809199

        tipping = missing_fit.sensitivity.tipping_gamma(
            "ate", null_hypothesis=null, arm_gamma=direction
        )

        witness = missing_fit.sensitivity.missingness(
            [0.0, 0.015, 0.0625], estimands=["ate"], arm_gamma=direction
        )
        assert float(witness["psi"].iloc[0]) < null
        assert float(witness["psi"].iloc[1]) > null
        assert float(witness["psi"].iloc[2]) < null
        assert tipping is not None and 0.006 < tipping < 0.008
        curve = missing_fit.sensitivity.missingness(
            [tipping], estimands=["ate"], arm_gamma=direction
        )
        assert float(curve["psi"].iloc[0]) == pytest.approx(null, abs=1e-4)

    def test_the_tipping_search_must_bracket_mar(self, missing_fit) -> None:
        with pytest.raises(ValueError, match=r"must be a finite, increasing.*contains gamma=0"):
            missing_fit.sensitivity.tipping_gamma("ate", search=(1.0, 2.0))
        with pytest.raises(ValueError, match="arm_gamma multipliers must be finite"):
            missing_fit.sensitivity.tipping_gamma("ate", arm_gamma={0: 0.0, 1: np.inf})

    def test_the_mnar_analyses_wait_to_be_asked_for(self, missing_fit) -> None:
        """A fit that *can* run the tilt still does not run it by default.

        ``tipping_gamma`` searches for a root by retargeting the whole tilt at every
        probe, so a bare combined report must not pay for it -- and must say which flag
        would.
        """
        default = missing_fit.sensitivity.run_all()
        for operation in ("missingness", "tipping_gamma"):
            assert default[operation].status is AssessmentStatus.DEFERRED
            assert "pass include_retargets=True" in default[operation].detail

        asked = missing_fit.sensitivity.run_all(include_retargets=True)
        assert asked["missingness"].status is AssessmentStatus.COMPLETED
        assert asked["tipping_gamma"].status is AssessmentStatus.COMPLETED

    def test_the_tilt_needs_missing_outcomes(self, good_overlap) -> None:
        with pytest.raises(CapabilityError, match="not_applicable"):
            good_overlap.sensitivity.missingness()

    def test_a_ratio_estimand_is_excluded(self, missing_fit) -> None:
        with pytest.raises(ValueError, match="no tiltable estimands"):
            missing_fit.sensitivity.missingness(estimands=["rr"])


class TestTheTiltUnderObservationWeights:
    r"""The tilted curve averages over the population the weights describe.

    A weighted fit reports a weighted estimand, so the MNAR curve through it has to be
    weighted at every ``gamma`` and not only at the origin.  The tilt takes three separate
    averages -- one for a mean, one for an unconditional contrast, one for a conditional
    effect whose weight is the arm indicator times the observation weight -- and none of
    them is reachable from the ``gamma = 0`` identity alone, because at ``gamma = 0`` the
    mixture collapses to the fit's own targeted regression and the ``ate`` moves by 0.0004
    where the weighted and unweighted averages of the *tilted* regression differ by 0.065.

    So the check is the mixing formula from the module docstring of
    :mod:`cleverly.sensitivity.missingness`, written out at a nonzero ``gamma``:

    .. math::

        \bar Q^{\text{full}}_\gamma(a, W)
          = \pi_a(W) \bar Q^*(a, W)
          + (1 - \pi_a(W)) \operatorname{expit}(\operatorname{logit} \bar Q^*(a, W) + \gamma).

    ``tipping_gamma`` needs no witness of its own: it reaches the same three averages
    through :func:`~cleverly.sensitivity.missingness.missingness_tilt`, one probe at a
    time, and has no arithmetic of its own past the root search.
    """

    #: Far enough from the origin that the mixture is doing work, and inside the range an
    #: analyst reads: the module's own default grid runs to 2.
    GAMMA = 1.25

    def _longhand(self, result, name: str, gamma: float, *, weighted: bool) -> float:
        """One tilted estimand, from the displayed formula and the fit's own nuisances."""
        from cleverly.utils.bounds import expit, logit

        data = result.data
        weights = data.weights if weighted else np.ones(data.n)
        group = "mean" if name in ("ey1", "ate") else name
        draws = []
        for repeat in result.repeats:
            scaler = repeat.nuisance.scaler
            arms = repeat.nuisance.arms
            observed = repeat.nuisance.bounded_missingness(result.config.missingness_bound)
            targeted = repeat.fluctuations[group].targeted

            def full(arm: float, targeted=targeted, observed=observed, arms=arms):
                q = np.asarray(targeted.arms[arm], dtype=float)
                pi = observed[:, arms.index(arm)]
                return pi * q + (1.0 - pi) * expit(logit(q) + gamma)

            if name == "ey1":
                scaled = float(np.average(full(1.0), weights=weights))
                draws.append(scaler.unscale_level(scaled) if not scaler.is_identity else scaled)
                continue
            contrast = full(1.0) - full(0.0)
            if name == "ate":
                over = weights
            else:
                conditioning = 1.0 if name == "att" else 0.0
                over = weights * np.asarray(data.treatment == conditioning, dtype=float)
            scaled = float(np.average(contrast, weights=over))
            draws.append(scaler.unscale_difference(scaled) if not scaler.is_identity else scaled)
        return float(np.median(draws))

    @pytest.mark.parametrize("name", ["ey1", "ate", "att", "atc"])
    def test_the_tilted_estimate_averages_over_the_weighted_population(
        self, weighted_missing_fit, name: str
    ) -> None:
        curve = nw.from_native(
            weighted_missing_fit.sensitivity.missingness([self.GAMMA], estimands=[name]),
            eager_only=True,
        )
        expected = self._longhand(weighted_missing_fit, name, self.GAMMA, weighted=True)
        assert float(curve["psi"][0]) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("name", ["ey1", "ate", "att", "atc"])
    def test_the_unweighted_average_is_a_different_number(
        self, weighted_missing_fit, name: str
    ) -> None:
        """The control that makes the comparison above worth making.

        Both sides come from this module's own arithmetic.  The measured gaps are 0.348
        for ``ey1``, 0.065 for ``ate``, 0.058 for ``att`` and 0.066 for ``atc``, against a
        ``rel=1e-12`` tolerance above.
        """
        weighted = self._longhand(weighted_missing_fit, name, self.GAMMA, weighted=True)
        unweighted = self._longhand(weighted_missing_fit, name, self.GAMMA, weighted=False)
        assert abs(weighted - unweighted) > 0.02

    def test_the_curve_still_passes_through_the_weighted_estimate(
        self, weighted_missing_fit
    ) -> None:
        """``gamma = 0`` is the MAR analysis of the *weighted* fit, not of the sample."""
        for name in ("ey1", "ate", "att", "atc"):
            curve = nw.from_native(
                weighted_missing_fit.sensitivity.missingness([0.0], estimands=[name]),
                eager_only=True,
            )
            assert float(curve["psi"][0]) == pytest.approx(
                weighted_missing_fit.psi(name), rel=1e-12
            )


class TestValidation:
    def test_the_score_check_passes_and_reports(self, good_overlap) -> None:
        check = good_overlap.diagnostics.score_equations()
        assert check.passed
        assert bool(check)
        assert check.failures == ()
        assert "PASS" in check.summary()
        check.raise_if_failed()

    def test_the_score_check_can_be_made_to_fail(self, good_overlap) -> None:
        # An absurd tolerance turns the check into a failure, exercising the reporting
        # path that a real convergence problem would take.
        strict = good_overlap.diagnostics.score_equations(tolerance=1e-30)
        assert not strict.passed
        assert strict.failures
        with pytest.raises(AssertionError, match="score equation was not solved"):
            strict.raise_if_failed()

    def test_a_passing_fit_prints_no_verdict(self, good_overlap) -> None:
        """Silent on the common path, which is what keeps every transcript untouched."""
        assert good_overlap.score_verdict.passed
        assert "score check" not in good_overlap.summary()

    def test_a_failing_score_check_is_visible_in_the_summary(self, good_overlap) -> None:
        """An unlicensed interval must not be formatted like any other.

        The fit is grafted rather than found: `weak_overlap_dgp` fails this check 23 times
        in 24 but costs a sweep to reach, and what is under test is the reporting rather
        than the cause.  A score the targeting could not have left is exactly the state
        the validation contract describes arriving in practice.
        """
        fluctuation = good_overlap.repeats[0].fluctuations["mean"]
        broken = dataclasses.replace(
            good_overlap,
            repeats=(
                dataclasses.replace(
                    good_overlap.repeats[0],
                    fluctuations={
                        "mean": dataclasses.replace(
                            fluctuation, score=np.full_like(fluctuation.score, 0.5)
                        )
                    },
                ),
            ),
        )

        assert not broken.score_verdict.passed
        summary = broken.summary()
        assert "score check: FAIL" in summary
        assert "mean" in summary.split("score check: FAIL")[1]
        assert "do not describe this estimate" in summary
        # The interval is still printed -- the line says it is not licensed, it does not
        # withhold it. Predeclaring which regimes are refused outright needs the
        # targeting-and-exit study's evidence; see
        # docs/technical-reference/dr-tmle/validation-programme.md.
        assert "95% CI" in summary

    def test_nuisance_diagnostics_cover_every_model(self, good_overlap) -> None:
        diagnostics = good_overlap.diagnostics.nuisance_models()
        names = {model.name for model in diagnostics.models}
        assert names == {"propensity", "outcome"}
        assert 0.0 < diagnostics["propensity"].metrics["auc"] < 1.0
        assert diagnostics["outcome"].metrics["r2"] > 0.1
        assert "VERDICT" in diagnostics.summary()

    def test_an_almost_randomised_treatment_is_read_as_good_overlap(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=81)
        # Replace treatment with pure coin flips: nothing in W predicts it.
        rng = np.random.default_rng(0)
        randomised = frame.assign(A=rng.binomial(1, 0.5, len(frame)).astype(float))
        result = fast_tmle(estimands=("ate",)).fit(randomised, outcome="Y", treatment="A").single()
        verdict = result.diagnostics.nuisance_models().verdict()
        assert "overlap is excellent" in verdict

    def test_calibration_is_reported_per_model(self, good_overlap) -> None:
        diagnostics = good_overlap.diagnostics.nuisance_models()
        frame = nw.from_native(
            diagnostics.calibration_frame("propensity", good_overlap.data), eager_only=True
        )
        assert {"bin", "n", "mean_predicted", "mean_observed"} <= set(frame.columns)
        assert len(frame) >= 2

    def test_super_learner_weights_are_summarised(self) -> None:
        frame, _ = make_nonlinear_ate(n=500, seed=82)
        # A one-model SuperLearner isolates the reporting path without paying for flexible
        # candidates that add no coverage here.
        result = (
            TMLE(
                outcome_learner=SuperLearner(
                    [sklearn.linear_model.LinearRegression()],
                    n_folds=3,
                ),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                learner_folds=3,
                estimands=("ate",),
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        weights = result.diagnostics.nuisance_models()["outcome"].learner_weights
        assert weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_combined_report_renders(self, good_overlap) -> None:
        assert "score_equations" in good_overlap.diagnostics.run_all().summary()
        assert "nuisance_models" in good_overlap.diagnostics.run_all().summary()


class TestRefutation:
    @pytest.fixture(scope="class")
    def refutation(self) -> object:
        frame, _ = make_linear_ate(n=700, seed=83)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        return result.diagnostics.refute(n_replicates=3, random_state=0)

    def test_all_default_tests_behave(self, refutation) -> None:
        assert refutation.passed
        assert {test.name for test in refutation.tests} == {
            "placebo",
            "random_common_cause",
            "subset",
        }

    def test_a_placebo_treatment_shows_no_effect(self, refutation) -> None:
        placebo = refutation["placebo"]
        # Permuting treatment destroys the effect while preserving its marginal.
        assert abs(placebo.mean) < 0.2 * abs(placebo.original)

    def test_an_irrelevant_covariate_does_not_move_the_estimate(self, refutation) -> None:
        noise = refutation["random_common_cause"]
        assert noise.mean == pytest.approx(noise.original, rel=0.05)

    def test_the_results_tabulate(self, refutation) -> None:
        frame = nw.from_native(refutation.to_frame(), eager_only=True)
        assert len(frame) == 3
        assert "VERDICT" in refutation.summary()

    def test_a_negative_control_outcome_shows_no_effect(self) -> None:
        frame, _ = make_linear_ate(n=700, seed=84)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        # An outcome built from the covariates alone, with no treatment component.
        rng = np.random.default_rng(0)
        control = frame["W1"].to_numpy() * 0.5 + rng.normal(size=len(frame))
        outcome = result.diagnostics.refute(
            tests=["negative_control_outcome"],
            negative_control_outcome=control,
            random_state=0,
        )
        assert outcome.passed

    def test_the_negative_control_test_needs_an_outcome(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=85)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match="needs an outcome array"):
            result.diagnostics.refute(tests=["negative_control_outcome"])

    def test_an_unknown_test_is_refused(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="unknown refutation test"):
            good_overlap.diagnostics.refute(tests=["magic"])

    def test_nonzero_measurement_error_changes_real_refits(self) -> None:
        frame, _ = make_linear_ate(n=120, seed=91)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        rule = EmpiricalInclusionRule(alpha=0.5, minimum_draws=4)

        def run(noise: float):
            return result.diagnostics.refute(
                tests=("bootstrap_measurement_error",),
                n_replicates=4,
                bootstrap_measurement_error=BootstrapMeasurementError(
                    ("W1",), numeric_noise=RelativeGaussianNoise(noise)
                ),
                measurement_error_rule=rule,
                random_state=17,
            )["bootstrap_measurement_error"]

        active = run(0.5)
        zero = run(0.0)
        child_seeds = tuple(
            int(sequence.generate_state(1)[0]) for sequence in np.random.SeedSequence(17).spawn(4)
        )
        assert active.child_seeds == zero.child_seeds == child_seeds
        assert active.failures == zero.failures == ()
        assert active.family == zero.family == "gaussian"
        assert len(active.records) == len(zero.records) == 4
        active_estimates = np.asarray([record.estimate for record in active.records])
        zero_estimates = np.asarray([record.estimate for record in zero.records])
        assert np.max(np.abs(active_estimates - zero_estimates)) > 1e-3


class TestARefutationInheritsTheFitsSeed:
    """``refute()`` used to draw from OS entropy whenever the caller named no seed.

    It was the only public stochastic operation in the package that did.  Every one of
    these tests calls the module function rather than ``result.diagnostics.refute``,
    because the facade memoises through ``_cached``: two facade calls with the same
    keywords return the *same object*, so an equality assertion written through the facade
    compares one object with itself and holds whatever the function does.
    """

    @staticmethod
    def _placebo(result: object, **kwargs: object) -> tuple[float, ...]:
        report = refute(result, n_replicates=1, tests=["placebo"], **kwargs)  # type: ignore[arg-type]
        return report["placebo"].values

    @pytest.fixture(scope="class")
    def seeded(self) -> object:
        frame, _ = make_linear_ate(n=500, seed=86)
        return fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()

    def test_two_refutations_of_one_seeded_fit_agree(self, seeded) -> None:
        assert self._placebo(seeded) == self._placebo(seeded)

    def test_the_inherited_seed_is_the_fit_own(self, seeded) -> None:
        """Not merely repeatable: repeatable *from the seed the fit records*."""
        assert seeded.estimator.random_state == 0
        assert self._placebo(seeded) == self._placebo(
            seeded, random_state=seeded.estimator.random_state
        )

    def test_an_explicit_seed_overrides_the_fit(self, seeded) -> None:
        assert self._placebo(seeded, random_state=3) != self._placebo(seeded)

    def test_a_zero_seed_still_overrides_the_fit(self) -> None:
        """``random_state=0`` is falsy, so the fallback has to read ``is None``.

        The other tests in this class hold under a truthiness test as well, because the
        fit they use carries seed 0 itself.  This one fails under it.
        """
        frame, _ = make_linear_ate(n=500, seed=86)
        fit = (
            fast_tmle(estimands=("ate",), random_state=21)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert self._placebo(fit, random_state=0) != self._placebo(fit)

    def test_an_unseeded_fit_still_draws_from_entropy(self) -> None:
        """The guarantee is conditional on the fit carrying a seed, and says so."""
        unseeded = self._unseeded_fit()
        assert unseeded.estimator.random_state is None
        assert self._placebo(unseeded) != self._placebo(unseeded)

    @staticmethod
    def _unseeded_fit() -> object:
        frame, _ = make_linear_ate(n=500, seed=87)
        return (
            fast_tmle(estimands=("ate",), random_state=None)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_the_report_records_the_seed_it_resolved(self, seeded) -> None:
        """Every branch of the resolution, so the field cannot drift from the generator."""
        assert refute(seeded, n_replicates=1, tests=["placebo"]).random_state == 0
        assert refute(seeded, n_replicates=1, tests=["placebo"], random_state=3).random_state == 3

    def test_replaying_a_recorded_seed_repeats_a_seeded_fits_report(self, seeded) -> None:
        """What the recorded seed is for: the report says how to obtain itself again."""
        report = refute(seeded, n_replicates=1, tests=["placebo"])
        replay = refute(seeded, n_replicates=1, tests=["placebo"], random_state=report.random_state)
        assert replay["placebo"].values == report["placebo"].values

    def test_replaying_a_recorded_seed_repeats_an_unseeded_fits_report(self) -> None:
        """The case the recorded seed exists for, and the one that needed the refit seeded.

        A perturbation is half of a refutation.  ``refit`` re-learns the nuisances, and an
        estimator carrying no ``random_state`` redraws its folds every time, so seeding the
        draws alone left this report unrepeatable.  The report is cached on the result and
        survives ``save``, so an unrepeatable one is a number nobody can check.
        """
        unseeded = self._unseeded_fit()
        report = refute(unseeded, n_replicates=1, tests=["placebo"])
        assert isinstance(report.random_state, int)
        replay = refute(
            unseeded, n_replicates=1, tests=["placebo"], random_state=report.random_state
        )
        assert replay["placebo"].values == report["placebo"].values

    def test_a_seeded_refit_leaves_the_estimator_alone(self) -> None:
        """The seed applies to a copy.  A refutation may not silently seed the fit itself."""
        unseeded = self._unseeded_fit()
        refute(unseeded, n_replicates=1, tests=["placebo"])
        assert unseeded.estimator.random_state is None

    def test_a_seeded_refit_matches_a_fit_that_carried_that_seed(self) -> None:
        """``refit(random_state=s)`` is the fit the estimator would have run at ``s``."""
        frame, _ = make_linear_ate(n=500, seed=87)
        unseeded = self._unseeded_fit()
        genuine = (
            fast_tmle(estimands=("ate",), random_state=4242)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        borrowed = unseeded.estimator.refit(
            unseeded.data,
            intermediate_value=unseeded.intermediate_value,
            random_state=4242,
        )
        assert (
            borrowed["ate"].psi
            == genuine.estimator.refit(genuine.data, intermediate_value=genuine.intermediate_value)[
                "ate"
            ].psi
        )


class TestTheDefaultEstimandOfTheOmittedVariableBound:
    """These analyses default to ``estimand="ate"``, which not every fit reports."""

    def test_a_sole_reported_parameter_is_supplied_without_being_named(self) -> None:
        frame, _ = make_linear_ate(n=800, seed=73)
        fit = fast_tmle(estimands=("ey1",)).fit(frame, outcome="Y", treatment="A").single()
        assert "ate" not in fit.estimates
        assert fit.sensitivity.robustness_value() == fit.sensitivity.robustness_value("ey1")

    def test_a_choice_between_parameters_is_the_callers_to_make(self) -> None:
        """``ey1`` and ``ey0`` are both linear and both reported, and they are different
        questions.

        Filling the gap by position would answer about the treated arm's counterfactual
        mean for a caller who asked nothing about arms, and the returned bound names no
        estimand for them to notice with.
        """
        frame, _ = make_linear_ate(n=800, seed=73)
        fit = fast_tmle(estimands=("ey1", "ey0")).fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match="was not requested in this fit"):
            fit.sensitivity.robustness_value()
        assert fit.sensitivity.robustness_value("ey0")["rv"] > 0.0


class TestCombinedSensitivityReport:
    def test_the_report_gathers_what_it_can(self, good_overlap) -> None:
        report = good_overlap.sensitivity.run_all().summary()
        assert "omitted_confounding" in report
        assert "robustness_value" in report
        assert "evalue" in report


class TestTheMechanismDenominatorsAreDiagnosed:
    r"""``P(Delta = 1 | A, W)`` divides the clever covariate; it needs the same scrutiny as ``g``.

    Nothing used to report it.  ``positivity()`` described only the propensity,
    ``truncation_curve()`` swept only ``g_bounds``, the positivity warning inspected only
    ``g``, and ``summary()`` printed only ``g_bounds`` and ``q_bounds`` -- so a fit could
    be resting on a handful of rows that were very unlikely to have been observed at all,
    with immaculate propensity overlap and nothing anywhere saying so.
    """

    @pytest.fixture(scope="class")
    def strained(self) -> object:
        # strength=2 sharpens the mechanism on W1: the first percentile of pi is ~0.13.
        frame, _ = make_missing_outcome(n=2000, seed=91, strength=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            return (
                fast_tmle(estimands=("ate",))
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2", "W3"],
                    delta="Delta",
                )
                .single()
            )

    def test_the_report_carries_the_mechanism(self, strained) -> None:
        report = strained.diagnostics.support()
        assert "P(Delta=1|A,W)" in report.mechanisms
        assert "P(A=a,Delta=1|W)" in report.mechanisms
        stats = report.mechanisms["P(Delta=1|A,W)"]
        assert 0.0 < stats["min"] < stats["q01"] < stats["q05"] < stats["median"] < 1.0
        assert "P(Delta=1|A,W)" in report.summary()

    def test_the_composed_row_reports_ess_and_concentration(self, strained) -> None:
        report = strained.diagnostics.support()
        stats = report.mechanisms["P(A=a,Delta=1|W)"]
        assert {"ess_ratio", "top_1pct", "top_5pct"} <= stats.keys()
        assert 0.0 < stats["ess_ratio"] <= 1.0
        assert 0.0 < stats["top_1pct"] <= stats["top_5pct"] <= 1.0
        retained = strained.assess().report("support")
        assert retained.mechanisms == report.mechanisms

    def test_the_mechanism_explains_leverage_the_propensity_does_not(self, strained) -> None:
        """The case the diagnostic exists for, asserted as a whole.

        On this fit the propensity overlap is immaculate -- nothing truncated, effective
        sample size near 90% of nominal in both arms -- and yet the largest clever
        covariate is in the hundreds.  Every bit of that comes from ``pi`` reaching
        0.04, an order of magnitude below the smallest propensity.  Before this the
        report had nothing to say about it: a reader saw a three-figure covariate next
        to a clean bill of health and no way to connect them.
        """
        # Measured across seeds 91-95 at this n and strength: pi bottoms out at
        # 0.019-0.039 against a smallest propensity of 0.105-0.165, the largest clever
        # covariate runs 53-195, the propensity ESS stays above 0.88 and nothing is
        # truncated. The windows below are set to hold across that whole range rather
        # than to the one seed the fixture happens to use.
        report = strained.diagnostics.support()
        mechanism = report.mechanisms["P(Delta=1|A,W)"]
        assert report.truncated["fraction"] == 0.0
        assert min(ess["ratio"] for ess in report.effective_sample_size.values()) > 0.88
        assert report.clever_covariate_max["mean"] > 40.0
        # The mechanism is where the leverage lives, and its ESS says so on the same
        # scale the propensity's is reported on.
        assert mechanism["min"] < 0.5 * float(np.min(strained.nuisance.propensity.values))
        assert mechanism["ess_ratio"] < 0.90

    def test_clipping_the_mechanism_reaches_the_verdict(self) -> None:
        """The verdict's truncation branch, forced deterministically.

        Driving it through the data instead -- a process sharp enough for the mechanism's
        effective sample size to fall past 0.6 -- lands at 0.58-0.65 depending on the
        seed, because the statistic is governed by the extreme tail of a normal
        covariate. That is a coin flip dressed as a test, so the bound is raised until it
        bites instead, which is deterministic and exercises the same verdict.
        """
        frame, _ = make_missing_outcome(n=1500, seed=94, strength=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            result = (
                fast_tmle(estimands=("ate",), nuisance_bound=0.35)
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2", "W3"],
                    delta="Delta",
                )
                .single()
            )
        verdict = result.diagnostics.support().verdict()
        assert "P(Delta=1|A,W) strains the estimate" in verdict
        assert "truncation_curve(mechanism=True)" in verdict

    def test_a_low_mechanism_ess_reaches_the_verdict(self, strained) -> None:
        # The other branch, checked on the rule rather than through a process: what a
        # data-driven version would be measuring is the tail of a normal, not the rule.
        report = strained.diagnostics.support()
        assert report.severity == "adequate"
        degenerate = dataclasses.replace(
            report,
            mechanisms={
                "P(Delta=1|A,W)": {**report.mechanisms["P(Delta=1|A,W)"], "ess_ratio": 0.4}
            },
        )
        assert "P(Delta=1|A,W) strains the estimate" in degenerate.verdict()
        assert "leaves an effective 40%" in degenerate.verdict()
        # Reported, not graded: this mechanism clips nothing, so the sentence appears and
        # the tier does not move. A cutoff on a Kish ratio would be an invented rule.
        assert degenerate.mechanisms["P(Delta=1|A,W)"]["clipped_fraction"] <= 0.01
        assert degenerate.severity == "adequate"

    def test_propensity_clipping_keeps_the_propensity_verdict(self, strained) -> None:
        """A derived row's clipping is the union over its factors, so it can be all ``g``.

        Triggering the mechanism branch on that union put a mechanism's name on a verdict
        whose every number came from the propensity, and closed it with
        ``mechanism=True``, which sweeps ``nuisance_bound`` alone. The propensity branch
        below reports the same cells accurately and names the curve that moves them.
        """
        report = strained.diagnostics.support()
        composed = "P(A=a,Delta=1|W)"

        def with_clipping(name: str) -> object:
            return dataclasses.replace(
                report,
                truncated={**report.truncated, "fraction": 0.09},
                mechanisms={
                    **report.mechanisms,
                    name: {**report.mechanisms[name], "clipped_fraction": 0.2},
                },
            )

        verdict = with_clipping(composed).verdict()
        assert f"{composed} strains the estimate" not in verdict
        assert "truncation is carrying this estimate" in verdict

        # The control: a *factor* row's clipping is its own, and still earns the sentence
        # ahead of the propensity's, which is the ordering this loop exists for.
        assert "P(Delta=1|A,W) strains the estimate" in with_clipping("P(Delta=1|A,W)").verdict()

        # And joint leverage still reaches the verdict through the derived row.
        leveraged = dataclasses.replace(
            report,
            mechanisms={
                **report.mechanisms,
                composed: {**report.mechanisms[composed], "ess_ratio": 0.4},
            },
        )
        assert f"{composed} strains the estimate" in leveraged.verdict()

    def test_a_fit_without_missingness_reports_no_mechanism(self, good_overlap) -> None:
        report = good_overlap.diagnostics.support()
        assert report.mechanisms == {}
        assert "P(Delta=1|A,W)" not in report.summary()

    def test_a_conditional_arm_fit_reports_its_factors_and_no_product(self) -> None:
        """``att`` divides by ``P(A = a)``, never by ``g_a(W) pi_a(W)``.

        ``att_submodel`` builds ``1 / (P(A=a) pi_a pz_a)`` for the conditioning arm and
        reweights the reference arm by the propensity odds, so the product is a
        denominator that fit never forms. Reporting it would be the convenient
        approximation to a different estimand -- and it would quote ``g_bounds`` where
        the covariate was held to ``g_bounds_conditional``. The observation mechanism is
        a denominator in both, so its factor row stays.
        """
        frame, _ = make_missing_outcome(n=800, seed=91, strength=1.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            result = (
                fast_tmle(estimands=("att",))
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2", "W3"],
                    delta="Delta",
                )
                .single()
            )
        report = result.diagnostics.support()
        assert set(result.fluctuations) == {"att"}
        assert set(report.mechanisms) == {"P(Delta=1|A,W)"}
        assert "P(A=a,Delta=1|W)" not in report.summary()

        # The omission says so. A silently absent row reads exactly like a fit with
        # nothing to report, and those are the two readings that must not be confused.
        assert report.composed_excluded == ("att",)
        summary = report.summary()
        assert "no derived denominator row is reported for att" in summary
        assert "divides by P(A=a) rather than g(W)" in summary

    def test_a_mixed_fit_says_which_estimands_the_product_row_omits(self) -> None:
        """The row is reported for the `ate` and does not describe the `att` beside it.

        The harder half of the same contract: the row is present, so its absence cannot
        carry the warning, and a reader who takes it for the whole fit reads a
        denominator two of these estimands never form.
        """
        frame, _ = make_missing_outcome(n=800, seed=91, strength=1.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            result = (
                fast_tmle(estimands=("ate", "att", "atc"))
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2", "W3"],
                    delta="Delta",
                )
                .single()
            )
        report = result.diagnostics.support()
        assert "P(A=a,Delta=1|W)" in report.mechanisms
        assert set(report.composed_excluded) == {"att", "atc"}
        summary = report.summary()
        assert "P(A=a,Delta=1|W) does not describe" in summary
        for group in ("att", "atc"):
            assert f"{group}, which divides by" in summary

    def test_a_covered_fit_and_a_complete_fit_say_nothing(self, strained, good_overlap) -> None:
        """The note is silent where there is nothing to explain.

        Two different silences, and both must hold. Every group the `strained` fit
        targets forms the product, so the row covers it. `good_overlap` has no fitted
        factor beside `g` at all, so there is no derived row for any group to be outside
        of, even though it targets an `att`.
        """
        covered = strained.diagnostics.support()
        assert covered.composed_excluded == ()
        assert "does not describe" not in covered.summary()

        complete = good_overlap.diagnostics.support()
        assert "att" in good_overlap.fluctuations
        assert complete.mechanisms == {}
        assert complete.composed_excluded == ()
        assert "no derived denominator row" not in complete.summary()

    def test_the_bound_appears_in_the_fit_summary(self, strained, good_overlap) -> None:
        # Traceability: a reported number must be traceable to every bound that shaped
        # it, not just the one with a familiar name.
        assert "P(Delta=1|A,W) truncated to" in strained.summary()
        assert "truncated to [0.01, 1]" not in good_overlap.summary()

    def test_the_curve_sweeps_the_mechanism_bound(self, strained) -> None:
        curve = nw.from_native(
            strained.diagnostics.truncation_curve(
                [0.01, 0.1, 0.25], estimands=["ate"], mechanism=True
            ),
            eager_only=True,
        )
        truncated = np.array(curve["truncated_fraction"].to_list())
        values = np.array(curve["psi"].to_list())
        # A tighter bound on pi clips more rows and moves the estimate, exactly as a
        # tighter bound on g does -- which is the whole reason it deserves a curve.
        assert np.all(np.diff(truncated) > 0)
        assert float(values.max() - values.min()) > 1e-3

    def test_the_mechanism_curve_is_flat_when_the_bound_never_binds(self, strained) -> None:
        # Below the smallest fitted pi nothing is clipped, so the estimate cannot move.
        smallest = float(np.min(strained.nuisance.missingness))
        grid = [smallest / 8.0, smallest / 4.0, smallest / 2.0]
        curve = nw.from_native(
            strained.diagnostics.truncation_curve(grid, estimands=["ate"], mechanism=True),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        assert float(values.max() - values.min()) < 1e-9

    def test_sweeping_the_mechanism_needs_a_mechanism(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="needs a fit with missing outcomes"):
            good_overlap.diagnostics.truncation_curve(mechanism=True)

    def test_a_degenerate_mechanism_warns(self) -> None:
        """The warning half: a fit that leans on the bound has to say so at fit time."""
        frame, _ = make_missing_outcome(n=1500, seed=92, strength=2.0)
        with pytest.warns(PositivityWarning, match=r"P\(Delta = 1 \| A, W\)"):
            fast_tmle(estimands=("ate",), nuisance_bound=0.35).fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            ).single()

    def test_an_untroubled_mechanism_does_not_warn(self) -> None:
        frame, _ = make_missing_outcome(n=1500, seed=93)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PositivityWarning)
            fast_tmle(estimands=("ate",)).fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            ).single()
