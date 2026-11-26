"""Influence-curve measures and ``SparseInfluenceWarning`` (X39, T4)."""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    ModelSpec,
    RegimeContrast,
    RegimeMean,
    Runtime,
    SparseInfluenceWarning,
    TMLEMethod,
)
from cleverly.datasets import make_longitudinal, make_longitudinal_competing
from cleverly.inference.sparse import (
    SPARSE_INFLUENCE_THRESHOLD,
    InfluenceReport,
    influence_report,
    influence_row,
    warn_sparse_influence,
)
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.estimator import _ReportedRows, _unreported_readers
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_


class TestTheMeasures:
    def test_a_fixture_curve_with_known_moments(self) -> None:
        # Centred: m2 = 3, m3 = 6, m4 = 21.
        row = influence_row("x", np.array([3.0, -1.0, -1.0, -1.0]))
        assert row.skewness == pytest.approx(6.0 / 3.0**1.5, abs=1e-12)
        assert row.kurtosis == pytest.approx(21.0 / 9.0, abs=1e-12)
        assert row.kish == pytest.approx(4.0 / (21.0 / 9.0), abs=1e-12)
        assert row.variance_rel_sd == pytest.approx(math.sqrt((21.0 / 9.0 - 1.0) / 4.0))

    def test_the_kish_count_is_the_squared_curve_formula(self) -> None:
        curve = np.random.default_rng(1).exponential(size=500)
        centred = curve - curve.mean()
        row = influence_row("x", curve)
        assert row.kish == pytest.approx(np.sum(centred**2) ** 2 / np.sum(centred**4), rel=1e-12)

    def test_the_edgeworth_loss_is_the_two_sided_term(self) -> None:
        curve = np.random.default_rng(2).exponential(size=300) ** 2
        row = influence_row("x", curve, alpha=0.05)
        z = norm.ppf(0.975)
        gamma, kappa = row.skewness, row.kurtosis - 3.0
        q2 = (
            kappa * (z**3 - 3 * z) / 6
            - gamma**2 * (z**5 + 2 * z**3 - 3 * z) / 9
            - (z**3 + 3 * z) / 2
        )
        assert row.edgeworth_loss == pytest.approx(-norm.pdf(z) * q2 / 300, abs=1e-15)
        # At z = 1.96 the research note's coefficients: 0.275 kappa - 4.23 gamma^2 - 6.70.
        assert q2 == pytest.approx(0.275 * kappa - 4.23 * gamma**2 - 6.70, rel=2e-3)

    def test_bootstrap_measures_need_draws(self) -> None:
        curve = np.random.default_rng(3).exponential(size=50)
        assert math.isnan(influence_row("x", curve).bca_z0)
        # Two of five draws lie below the estimate, so z0 = Phi^-1(0.4), which is not zero and
        # changes sign with a sign slip.
        draws = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        row = influence_row("x", curve, estimate=0.25, draws=draws)
        assert row.bca_z0 == pytest.approx(norm.ppf(0.4), abs=1e-15)
        assert row.bca_z0 < -0.2
        assert row.bca_acceleration > 0

    def test_a_tied_draw_counts_for_half(self) -> None:
        curve = np.random.default_rng(3).exponential(size=50)
        row = influence_row("x", curve, estimate=0.2, draws=np.array([0.1, 0.2, 0.2, 0.3]))
        assert row.bca_z0 == 0.0

    def test_a_clustered_curve_is_read_at_the_cluster_level(self) -> None:
        curve = np.random.default_rng(5).normal(size=40) ** 3
        cluster = np.repeat(np.arange(8), 5)
        row = influence_row("x", curve, cluster=cluster)
        sums = curve.reshape(8, 5).sum(axis=1)
        reference = influence_row("x", sums)
        assert row.n == 8
        for field in ("n", "kish", "skewness", "kurtosis", "variance_rel_sd", "edgeworth_loss"):
            assert getattr(row, field) == getattr(reference, field), field
        assert influence_row("x", curve).n == 40
        assert influence_row("x", curve).kish != row.kish

    def test_a_flat_curve(self) -> None:
        row = influence_row("x", np.ones(10))
        assert row.kish == math.inf and not row.sparse


class TestTheWarning:
    @staticmethod
    def _report(nonzero: int, n: int = 2000) -> InfluenceReport:
        curve = np.zeros(n)
        curve[:nonzero] = np.random.default_rng(4).normal(size=nonzero)
        return InfluenceReport((influence_row("x", curve),), 0.05)

    def test_the_threshold_is_a_kish_count_of_64(self) -> None:
        assert SPARSE_INFLUENCE_THRESHOLD**2 == 1 / 64
        for nonzero in (8, 30, 60, 70, 200, 1500):
            row = self._report(nonzero).rows[0]
            # (kappa - 1) / n > 1/64 differs from n / kappa < 64 only by 1/n.
            assert row.sparse == ((row.kurtosis - 1) / row.n > 1 / 64)
            assert row.sparse == (row.kish < 64) or abs(row.kish - 64) < 64 * 64 / row.n

    def test_a_dense_curve_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseInfluenceWarning)
            warn_sparse_influence(self._report(1500), cross_fitted=False)

    def test_a_sparse_curve_warns_with_the_bootstrap_advice(self) -> None:
        with pytest.warns(SparseInfluenceWarning, match="bootstrap_interval='bca'"):
            warn_sparse_influence(self._report(20), cross_fitted=False)

    def test_a_cross_fitted_fit_is_not_told_to_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=True)
        text = str(record[0].message)
        assert "X20" in text and "n_bootstrap" not in text

    def test_the_advice_says_199_and_experimental(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False)
        text = str(record[0].message)
        assert "n_bootstrap=199" in text and "399" not in text
        assert "BCa interval is experimental" in text

    def test_a_percentile_bootstrap_is_told_to_use_bca(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False, percentile_bootstrap=True)
        text = str(record[0].message)
        assert "percentile bootstrap interval keeps the skew" in text
        assert "bootstrap_interval='bca'" in text and "n_bootstrap" not in text
        assert "experimental" in text

    def test_a_flagged_row_shows_why_it_is_flagged(self) -> None:
        report = self._report(20)
        (row,) = report.rows
        assert row.sparse
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(report, cross_fitted=False)
        text = str(record[0].message)
        assert f"relative SD {row.variance_rel_sd:.3f}" in text
        assert f"Edgeworth coverage loss {row.edgeworth_loss:.3f}" in text

    def test_a_flagged_row_can_carry_a_negative_edgeworth_loss(self) -> None:
        """The flag reads the relative SD only, so a negative loss does not clear it."""
        # Six symmetric spikes in 300 rows: kurtosis 50 lifts q2 above zero, so the Edgeworth
        # term predicts over-coverage while the variance estimate is noisy.
        curve = np.zeros(300)
        curve[:6] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        flagged_negative = influence_row("y", curve)
        assert flagged_negative.sparse and flagged_negative.edgeworth_loss < 0
        with pytest.warns(SparseInfluenceWarning, match="relative SD .* Edgeworth coverage loss -"):
            warn_sparse_influence(InfluenceReport((flagged_negative,), 0.05), cross_fitted=False)

    def test_the_warning_points_at_the_caller_outside_the_package(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False)
        assert record[0].filename == __file__

    def test_an_unmeasured_fit_is_not_told_to_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False, bootstrap_measured=False)
        assert "n_bootstrap" not in str(record[0].message)

    def test_an_mtp_fit_the_option_accepts_is_told_about_the_option(self) -> None:
        """Fix-5: the advice names ``variance="policy_weighted"`` and not the bootstrap."""
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False, policy_available=True)
        text = str(record[0].message)
        assert "variance='policy_weighted'" in text
        assert "0.980 to 0.985" in text and "BCa interval 0.908 to 0.923" in text
        assert "n_bootstrap" not in text and "bootstrap_interval" not in text

    def test_control_a_cross_fitted_fit_keeps_its_advice_over_the_option(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=True, policy_available=True)
        text = str(record[0].message)
        assert "X20" in text and "policy_weighted" not in text

    def test_control_without_the_flag_the_option_is_not_named(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False)
        assert "policy_weighted" not in str(record[0].message)

    def test_the_warning_is_a_user_warning(self) -> None:
        assert issubclass(SparseInfluenceWarning, UserWarning)


def _fit(**settings: Any) -> Any:
    model = LTMLE(
        plans_.regimens(("low", "high", "respond"), 3, 2),
        outcome_learner=GLM(),
        pseudo_learner=GLM(),
        g_bounds=(1e-8, 1.0),
        random_state=0,
        **settings,
    )
    return model.fit(
        law_.draw(3, 2, 600, 9, law_.L_R), **law_.roles(3, 2), **law_.probability_columns(3, 2)
    )


class TestTheFit:
    def test_a_sparse_wald_fit_warns(self) -> None:
        with pytest.warns(SparseInfluenceWarning, match="n_bootstrap=199"):
            result = _fit(n_folds=1)
        report = result.diagnostics.influence()
        assert report.sparse
        assert set(report.sparse) <= set(result.estimates)

    @pytest.mark.filterwarnings("ignore:.*constant_node_plugin")
    @pytest.mark.parametrize(
        "settings",
        [{"isotonic": True}, {"survival_model": "hazard"}],
        ids=["isotonic", "hazard"],
    )
    def test_a_hazard_or_isotonic_fit_is_not_told_to_bootstrap(
        self, settings: dict[str, Any]
    ) -> None:
        from tests.unit.test_isotonic_projection import _fit as survival_fit

        with pytest.warns(SparseInfluenceWarning) as record:
            survival_fit(36, **settings)
        sparse = [w for w in record if issubclass(w.category, SparseInfluenceWarning)]
        text = str(sparse[0].message)
        assert "No registered study measures a bootstrap" in text
        assert "n_bootstrap" not in text and "bca" not in text

    def test_a_cross_fitted_fit_warns_without_the_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            _fit(n_folds=2)
        sparse = [w for w in record if issubclass(w.category, SparseInfluenceWarning)]
        assert sparse and "n_bootstrap" not in str(sparse[0].message)

    def test_a_percentile_bootstrap_fit_warns_with_the_bca_advice(self) -> None:
        with pytest.warns(SparseInfluenceWarning, match="percentile bootstrap interval"):
            _fit(n_folds=1, n_bootstrap=3)

    def test_a_clustered_fit_does_not_warn(self) -> None:
        from tests.unit.test_bootstrap_intervals import _clustered

        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseInfluenceWarning)
            _clustered(n_bootstrap=0, bootstrap_interval="percentile")

    def test_a_bca_bootstrap_fit_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", SparseInfluenceWarning)
            result = _fit(n_folds=1, n_bootstrap=3, bootstrap_interval="bca")
        report = result.diagnostics.influence()
        assert all(np.isfinite(row.bca_acceleration) for row in report.rows)
        assert report.rows[0].name == next(iter(result.estimates))

    def test_influence_is_a_direct_call_and_not_in_run_all(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _fit(n_folds=1)
        assert result.diagnostics.influence().rows
        battery = result.diagnostics.run_all()
        assert "influence" not in {item.name for item in battery.items}
        assert "influence" not in {row.operation for row in result.diagnostics.capabilities}

    def test_the_report_reads_every_estimate(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _fit(n_folds=1)
        report = influence_report(result.estimates)
        assert [row.name for row in report.rows] == list(result.estimates)
        frame = report.to_frame()
        assert list(frame["name"]) == list(result.estimates)


def test_linear_regression_is_frequency_exact() -> None:
    from cleverly.longitudinal.frequency import frequency_exact

    assert frequency_exact(LinearRegression()) and frequency_exact(GLM())


class TestRowsTheResultDoesNotReport:
    """N1 of X120: a study fit warns about a flagged row its result drops, and says so.

    A :class:`~cleverly.CausalEffect` fits an engine that builds every level and contrast,
    then keeps the identified rows. The warning still fires on the same rows; only its text
    changes, naming a dropped row as one the fit does not report, with the reported
    contrasts whose interval reads it through their MOVER components.
    """

    @staticmethod
    def _report(*names: str) -> InfluenceReport:
        sparse = np.zeros(2000)
        sparse[:20] = np.random.default_rng(4).normal(size=20)
        dense = np.random.default_rng(5).normal(size=2000)
        rows = [
            influence_row(name, sparse if index == 0 else dense) for index, name in enumerate(names)
        ]
        return InfluenceReport(tuple(rows), 0.05)

    @staticmethod
    def _text(report: InfluenceReport, **settings: Any) -> str:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(report, cross_fitted=False, **settings)
        (warning,) = [w for w in record if issubclass(w.category, SparseInfluenceWarning)]
        return str(warning.message)

    def test_an_unreported_row_names_the_contrast_that_reads_it(self) -> None:
        text = self._text(self._report("level", "contrast"), unreported={"level": ("contrast",)})
        assert (
            "level, which the fit does not report and whose interval contrast reads (Kish" in text
        )
        assert text.endswith("The influence table lists the reported estimates only.")

    def test_two_readers_are_joined(self) -> None:
        text = self._text(self._report("level"), unreported={"level": ("one", "two")})
        assert "whose interval one and two read (Kish" in text

    def test_an_unreported_row_with_no_reader_claims_none(self) -> None:
        text = self._text(self._report("level"), unreported={"level": ()})
        assert "level, which the fit does not report (Kish" in text
        assert "whose interval" not in text

    def test_a_flagged_reported_row_reads_as_before(self) -> None:
        """The flagged row is reported; another, dense row is not. The text is unchanged."""
        report = self._report("level", "dense")
        assert report.rows[0].sparse and not report.rows[1].sparse
        assert self._text(report, unreported={"dense": ()}) == self._text(report)
        assert "does not report" not in self._text(report)

    def test_control_unreported_rows_without_a_flag_do_not_warn(self) -> None:
        dense = InfluenceReport(
            (influence_row("level", np.random.default_rng(6).normal(size=2000)),), 0.05
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseInfluenceWarning)
            warn_sparse_influence(dense, cross_fitted=False, unreported={"level": ("x",)})

    def test_a_policy_note_also_says_the_table_lists_the_reported_rows(self) -> None:
        text = self._text(self._report("level"), unreported={"level": ()}, policy_variance=True)
        assert "larger of the influence-curve and policy-weighted variances" in text
        assert text.endswith("The influence table lists the reported estimates only.")


def _competing_study() -> tuple[Any, Any, Any]:
    """The ``longitudinal-survival`` tutorial's data, design and learners."""
    frame, _ = make_longitudinal_competing(n=4_000, seed=53, censoring=True)
    design = LongitudinalTreatment(
        outcome={"readmission": ("R1", "R2"), "death": ("D1", "D2")},
        censoring=("C1", "C2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
    )
    logistic = LogisticRegression(max_iter=1000, random_state=41)
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=logistic,
            pseudo_learner=LinearRegression(),
            treatment_learner=logistic,
            censoring_learner=logistic,
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=41, n_jobs=1),
    )
    return CausalStudy(frame, design=design), design, method


def _sparse_texts(fit: Any) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = fit()
    texts = [str(w.message) for w in record if issubclass(w.category, SparseInfluenceWarning)]
    return result, texts


class TestAStudyFitNamesTheRowsItDrops:
    LEVEL = "cif_regimen[always, death @ t=1]"
    READER = "ate_regimen[always vs never, death @ t=1]"

    def test_a_contrast_fit_names_the_level_and_its_one_reader(self) -> None:
        study, _, method = _competing_study()
        estimand = RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
        result, (text,) = _sparse_texts(lambda: study.identify(estimand).estimate(method=method))
        assert self.LEVEL not in result.estimates
        assert self.LEVEL not in {row.name for row in result.diagnostics.influence().rows}
        assert (
            f"{self.LEVEL}, which the fit does not report and whose interval {self.READER} reads"
            in text
        )
        # Witness: only the contrast built on the flagged level is named, not its siblings.
        for other in (
            "ate_regimen[always vs never, death @ t=2]",
            "ate_regimen[always vs never, readmission @ t=1]",
            "ate_regimen[always vs never, readmission @ t=2]",
        ):
            assert other in result.estimates and other not in text
        assert result.estimates[self.READER].interval_scale == "mover"
        # The result keeps the same text, and its summary prints it.
        assert [w.message for w in result.fit_warnings] == [text]
        summary = " ".join(str(result.summary()).split())
        assert f"{self.LEVEL}, which the fit does not report" in summary

    def test_a_level_fit_keeps_its_text(self) -> None:
        """The flagged level is reported, so the text is the one a caller who sees every row
        gets, byte for byte."""
        study, _, method = _competing_study()
        estimand = RegimeMean({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
        result, (text,) = _sparse_texts(lambda: study.identify(estimand).estimate(method=method))
        assert self.LEVEL in result.estimates
        assert "does not report" not in text and "influence table lists" not in text
        assert text.startswith(
            "1 estimate(s) read their influence curve from few rows, so the variance "
            f"estimate's relative SD is above 0.125: {self.LEVEL} (Kish 42, relative SD 0.153, "
            "Edgeworth coverage loss 0.005). The Wald interval may under-cover."
        )
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(result.diagnostics.influence(), cross_fitted=False)
        assert text == str(record[0].message)

    def test_a_direct_engine_fit_keeps_its_text(self) -> None:
        """A direct ``LTMLE`` caller sees every row, so its text names no dropped row. The
        goldens of ``tests/unit/_longitudinal_repeats_golden.py`` pin more direct texts."""
        study, _, method = _competing_study()
        data = study.data
        assert isinstance(data, LongitudinalData)
        logistic = method.models.treatment_learner
        engine = LTMLE(
            {"always": 1, "never": 0},
            reference="never",
            horizons=(1, 2),
            outcome_learner=logistic,
            pseudo_learner=LinearRegression(),
            treatment_learner=logistic,
            censoring_learner=logistic,
            n_folds=1,
            random_state=41,
        )
        result, (text,) = _sparse_texts(lambda: engine.fit(data))
        assert self.LEVEL in result.estimates and self.READER in result.estimates
        assert "does not report" not in text and f"{self.LEVEL} (Kish 42" in text

    def test_an_unbounded_contrast_claims_no_reader(self) -> None:
        """Mutation control for "components, not contributors": an unbounded continuous
        contrast reads its own curve, so its levels are named as dropped with no reader,
        although they are the contrast's contributors."""
        frame, _ = make_longitudinal(n=300, seed=7, censoring=False)
        frame["Y"] = frame["Y"] * 2.0 + np.random.default_rng(1).exponential(size=300) ** 2
        design = LongitudinalTreatment(
            outcome="Y", treatment=("A1", "A2"), baseline=("W1", "W2"), time_varying=((), ("L2",))
        )
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=LinearRegression(),
                pseudo_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000, random_state=0),
            ),
            cross_fitting=CrossFitting(enabled=False),
            runtime=Runtime(random_state=0, n_jobs=1),
        )
        study = CausalStudy(frame, design=design)
        plans = {"always": 1, "never": 0}
        contrast = "ate_regimen[always vs never]"
        result, (text,) = _sparse_texts(
            lambda: study.identify(RegimeContrast(plans, reference="never")).estimate(method=method)
        )
        assert result.estimates[contrast].components is None
        assert "ey_regimen[always], which the fit does not report (Kish" in text
        assert "ey_regimen[never], which the fit does not report (Kish" in text
        assert "whose interval" not in text
        # The reported contrast is itself flagged, and reads as a reported row.
        assert f"; {contrast} (Kish" in text
        # The mirror case: a level fit drops the contrast it built, which no row reads.
        _, (mirror,) = _sparse_texts(
            lambda: study.identify(RegimeMean(plans, reference="never")).estimate(method=method)
        )
        assert f"{contrast}, which the fit does not report (Kish" in mirror
        assert "ey_regimen[always] (Kish" in mirror


class TestTheReportedRowsRule:
    def test_readers_come_from_components_and_horizons_filter(self) -> None:
        study, _, method = _competing_study()
        effect = study.identify(
            RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
        )
        engine = effect._longitudinal_engine(method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseInfluenceWarning)
            raw = engine.fit(study.data)
        index, data = raw.parameter_index, raw.data
        contrasts = _ReportedRows(("ate_regimen[", "risk_difference["))
        readers = _unreported_readers(contrasts, raw.estimates, index, data)
        assert readers is not None
        for schedule in ("always", "never"):
            for cause in ("readmission", "death"):
                for horizon in (1, 2):
                    level = f"cif_regimen[{schedule}, {cause} @ t={horizon}]"
                    assert readers[level] == (
                        f"ate_regimen[always vs never, {cause} @ t={horizon}]",
                    )
        # A contrast stripped of its components names no reader (mutation control).
        stripped = {
            name: dataclasses.replace(value, components=None) if name.startswith("ate_") else value
            for name, value in raw.estimates.items()
        }
        assert set(_unreported_readers(contrasts, stripped, index, data).values()) == {()}
        # The mirror rule drops the contrasts, which nothing reads.
        levels = _ReportedRows(("ey_regimen[", "risk_regimen[", "cif_regimen["))
        mirror = _unreported_readers(levels, raw.estimates, index, data)
        assert mirror is not None and set(mirror) == {n for n in raw.estimates if "ate_" in n}
        assert set(mirror.values()) == {()}
        # Restricted means add grid times the result does not report: a horizon filter drops
        # the day-30 rows too, and names no reader for a level whose contrast it drops.
        later = _ReportedRows(("ate_regimen[",), horizons=(2.0,))
        shown = later.select(raw.estimates, index, data)
        assert shown == [n for n in raw.estimates if n.startswith("ate_") and "t=2" in n]
        dropped = _unreported_readers(later, raw.estimates, index, data)
        assert dropped is not None
        assert dropped["cif_regimen[always, death @ t=1]"] == ()
        assert dropped["ate_regimen[always vs never, death @ t=1]"] == ()
        assert _unreported_readers(None, raw.estimates, index, data) is None

    def test_a_refit_template_drops_the_rule(self) -> None:
        study, _, method = _competing_study()
        estimand = RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseInfluenceWarning)
            result = study.identify(estimand).estimate(method=method)
        assert result.estimator._reported_rows is None
