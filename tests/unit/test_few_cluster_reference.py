"""A clustered estimate takes a Student t reference with J - 2 degrees of freedom below 40.

Nugent, Marquez, Charlebois, Abbott and Balzer (2024), Section 2.2, last paragraph, give the
rule. ``J`` counts the clusters with positive weight mass among the rows the estimate reads:
the whole fit for a marginal estimate, and the stratum for a stratum estimate. A derived
estimate takes the smallest degrees of freedom of its inputs. At 40 clusters or more the normal
reference stays, bit for bit. Below 10, the smallest count a registered study measures, the fit takes ``"few_cluster_plugin"``, and every
estimate keeps the normal reference of its diagnostic.

Each class names its nonzero witness and its mutation control. The fits use explicit linear
learners and in-sample nuisances unless a class says otherwise.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import variable_importance
from cleverly._inference_status import (
    NO_T_REFERENCE_BANDS,
    T_REFERENCE_BOOTSTRAP_NOTE,
    T_REFERENCE_NOTE,
)
from cleverly.datasets import (
    make_binary_outcome,
    make_clustered,
    make_linear_ate,
    make_missing_outcome,
    make_missing_outcome_binary,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference import cluster as cluster_module
from cleverly.inference import influence as influence_module
from cleverly.inference import make_estimate, median_estimates, simultaneous_bands, wald_ci
from cleverly.inference.multiplier import T_REFERENCE_BAND_REFUSAL
from cleverly.inference.results import ratio_contrast, smooth_contrast
from cleverly.interventions import Incremental, Shift
from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM
from cleverly.sensitivity.missingness import missingness_tilt
from cleverly.sensitivity.omitted_variable import omitted_variable_bounds
from tests import discrete_law, regimes
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import (
    LONGITUDINAL_N,
    LONGITUDINAL_NODES,
    longitudinal_learners,
)
from tests.unit._inference_status_support import cluster_labels as labels
from tests.unit.test_longitudinal_msm import DOSE
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

pytestmark = pytest.mark.xdist_group("few_cluster_reference")

tmle_module = importlib.import_module("cleverly.estimators.tmle")
longitudinal_estimator = importlib.import_module("cleverly.longitudinal.estimator")

ROOT = Path(__file__).resolve().parents[2]


def with_clusters(frame: Any, k: int) -> Any:
    """``frame`` with ``k`` clusters of consecutive rows."""
    return frame.assign(cluster=labels(len(frame), k))


def fit_clustered(frame: Any, k: int, **settings: Any) -> Any:
    roles = settings.pop("roles", {})
    return (
        TMLE(**linear_in_sample(**settings))
        .fit(with_clusters(frame, k), outcome="Y", treatment="A", id="cluster", **roles)
        .single()
    )


def reference_dfs(result: Any) -> dict[str, int | None]:
    return {name: estimate.reference_df for name, estimate in result.estimates.items()}


@pytest.fixture(scope="module")
def twelve() -> Any:
    """``make_binary_outcome(400, 17)`` in 12 clusters: J - 2 = 10."""
    frame, _ = make_binary_outcome(n=400, seed=17)
    return fit_clustered(frame, 12, estimands=("ey0", "ey1", "ate", "rr"), n_bootstrap=0)


class TestTheReferenceQuantile:
    """Nonzero witness: the t(10) interval is 2.2281 / 1.9600 times the normal one."""

    def test_the_interval_and_the_pvalue_read_t_with_ten_degrees_of_freedom(
        self, twelve: Any
    ) -> None:
        critical = float(stats.t.ppf(0.975, 10))
        for name in ("ey0", "ey1", "ate"):
            estimate = twelve[name]
            assert estimate.reference_df == 10
            low, high = estimate.ci
            assert low == estimate.psi - critical * estimate.std_error
            assert high == estimate.psi + critical * estimate.std_error
            z = estimate.psi / estimate.std_error
            assert estimate.pvalue == pytest.approx(2.0 * stats.t.sf(abs(z), 10), rel=1e-12)
            assert estimate.plugin_interval == estimate.ci
            assert estimate.to_dict()["reference_df"] == 10
            assert estimate.wald_test().reference_df == 10
            width = (high - low) / (2.0 * 1.959963984540054 * estimate.std_error)
            assert width == pytest.approx(2.228138851986274 / 1.959963984540054, rel=1e-12)

    def test_a_ratio_is_exponentiated_from_the_log_scale(self, twelve: Any) -> None:
        rr = twelve["rr"]
        assert rr.reference_df == 10
        critical = float(stats.t.ppf(0.975, 10))
        assert rr.log_psi is not None
        assert rr.ci == pytest.approx(
            (
                float(np.exp(rr.log_psi - critical * rr.std_error)),
                float(np.exp(rr.log_psi + critical * rr.std_error)),
            ),
            rel=1e-12,
        )

    @pytest.mark.parametrize(
        "mutant", [lambda count: count - 1, lambda count: None], ids=["J - 1", "normal"]
    )
    def test_another_reference_fails(self, mutant: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def rule(cluster: Any, weights: Any = None, rows: Any = None) -> int | None:
            keep = np.ones(len(cluster), dtype=bool) if rows is None else rows
            return mutant(int(np.unique(np.asarray(cluster)[keep]).size))

        monkeypatch.setattr(influence_module, "cluster_reference_df", rule)
        frame, _ = make_binary_outcome(n=400, seed=17)
        mutated = fit_clustered(frame, 12, estimands=("ate",))
        with pytest.raises(AssertionError):
            self.test_the_interval_and_the_pvalue_read_t_with_ten_degrees_of_freedom(
                _with_names(mutated)
            )


def _with_names(result: Any) -> Any:
    """``result`` with ``ey0`` and ``ey1`` aliased to ``ate``, for a one-estimate mutant fit."""

    class View:
        def __getitem__(self, name: str) -> Any:
            return result["ate"]

    return View()


def equal_clusters(k: int, size: int = 10) -> Any:
    return make_clustered(n=size * k, cluster_size=size, seed=7)[0]


def fit_k(k: int) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=("ate", "ey0")))
        .fit(equal_clusters(k), outcome="Y", treatment="A", id="cluster")
        .single()
    )


def below_or_at(threshold_name: str) -> Any:
    """The mutant of ``cluster_reference_df`` that compares with ``<=`` at one constant."""

    original = cluster_module.cluster_reference_df

    def mutant(cluster: Any, weights: Any = None, rows: Any = None) -> int | None:
        keep = np.ones(len(cluster), dtype=bool) if rows is None else rows
        count = int(np.unique(np.asarray(cluster)[keep]).size)
        if threshold_name == "FEW_CLUSTER_THRESHOLD" and count == 40:
            return count - 2
        return original(cluster, weights, rows)

    return mutant


class TestTheBoundaries:
    def test_thirty_nine_clusters_give_thirty_seven(self) -> None:
        assert set(reference_dfs(fit_k(39)).values()) == {37}

    def test_forty_clusters_keep_the_normal_reference_bit_for_bit(self) -> None:
        result = fit_k(40)
        for estimate in result.estimates.values():
            assert estimate.reference_df is None
            assert estimate.ci == wald_ci(estimate.psi, estimate.std_error, 0.05)

    def test_ten_clusters_give_eight(self) -> None:
        result = fit_k(10)
        assert result.inference_status == "influence_curve"
        assert set(reference_dfs(result).values()) == {8}

    def test_nine_clusters_take_the_status(self) -> None:
        result = fit_k(9)
        assert result.inference_status == "few_cluster_plugin"
        assert set(reference_dfs(result).values()) == {None}

    def test_an_at_or_below_threshold_fails_the_forty_cluster_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            influence_module, "cluster_reference_df", below_or_at("FEW_CLUSTER_THRESHOLD")
        )
        with pytest.raises(AssertionError):
            self.test_forty_clusters_keep_the_normal_reference_bit_for_bit()


class TestThePositiveMassCount:
    def test_zero_weight_clusters_do_not_count(self) -> None:
        frame = with_clusters(make_binary_outcome(n=420, seed=3)[0], 14)
        frame = frame.assign(w=np.where(frame["cluster"] < 2, 0.0, 1.0))
        result = (
            TMLE(**linear_in_sample(estimands=("ate",)))
            .fit(frame, outcome="Y", treatment="A", id="cluster", weights="w")
            .single()
        )
        # The witness: the labels alone would give 12.
        assert result.data.n_clusters == 14
        assert result["ate"].reference_df == 10

    def test_counting_every_label_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = cluster_module.cluster_reference_df
        monkeypatch.setattr(
            influence_module,
            "cluster_reference_df",
            lambda cluster, weights=None, rows=None: original(cluster, None, rows),
        )
        with pytest.raises(AssertionError):
            self.test_zero_weight_clusters_do_not_count()


def stratified(k: int, small: int, seed: int = 7) -> Any:
    """``k`` clusters of 10 rows, with the first ``small`` clusters in stratum ``S = "small"``."""
    frame = make_clustered(n=10 * k, cluster_size=10, seed=seed)[0]
    return frame.assign(S=np.where(frame["cluster"] < small, "small", "big"))


def fit_strata(frame: Any) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=("ate",)))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "S"],
            id="cluster",
            strata=["S"],
        )
        .single()
    )


class TestTheStratumCount:
    def test_a_small_stratum_inside_a_large_fit(self) -> None:
        result = fit_strata(stratified(44, small=12))
        assert reference_dfs(result) == {"ate": None, "ate[S='small']": 10, "ate[S='big']": 30}

    def test_two_strata_below_forty(self) -> None:
        result = fit_strata(stratified(32, small=12))
        assert reference_dfs(result) == {"ate": 30, "ate[S='small']": 10, "ate[S='big']": 18}

    def test_a_dropped_stratum_mapping_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = tmle_module.TMLE._stratum_estimates

        def no_mapping(self: Any, *args: Any, **kwargs: Any) -> Any:
            estimates, _ = original(self, *args, **kwargs)
            return estimates, {}

        monkeypatch.setattr(tmle_module.TMLE, "_stratum_estimates", no_mapping)
        with pytest.raises(AssertionError):
            self.test_two_strata_below_forty()


class TestAFewClusterStratumDoesNotCrash:
    def test_a_three_cluster_stratum_gives_the_status_and_normal_diagnostics(self) -> None:
        result = fit_strata(stratified(13, small=3))
        assert result.inference_status == "few_cluster_plugin"
        assert set(reference_dfs(result).values()) == {None}
        for estimate in result.estimates.values():
            low, _ = estimate.plugin_interval
            assert low == pytest.approx(
                estimate.psi - 1.959963984540054 * estimate.plugin_std_error
            )

    def test_a_reference_computed_before_the_status_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = influence_module.stamp_inference

        def eager(estimates: Any, status: Any, reference: Any = None) -> Any:
            stamped = original(estimates, status, reference)
            if reference is None:
                return stamped
            return {
                name: influence_module.replace(e, reference_df=reference.reference_df(name))
                for name, e in stamped.items()
            }

        monkeypatch.setattr(tmle_module, "stamp_inference", eager)
        with pytest.raises((AssertionError, ValueError)):
            self.test_a_three_cluster_stratum_gives_the_status_and_normal_diagnostics()


def curve_estimate(name: str, df: int | None, psi: float = 0.4) -> Any:
    curve = np.array([0.1, -0.1, 0.2, -0.2, 0.05, -0.05])
    return make_estimate(name, psi, curve, n=6, scale="level", reference_df=df)


class TestDerivedEstimatesTakeTheMinimum:
    @pytest.mark.parametrize(
        ("left", "right", "expected"), [(6, 9, 6), (None, 6, 6), (None, None, None)]
    )
    def test_a_smooth_contrast(self, left: Any, right: Any, expected: Any) -> None:
        estimates = {"a": curve_estimate("a", left), "b": curve_estimate("b", right, 0.3)}
        derived = smooth_contrast(estimates, lambda p: p[0] - p[1], ["a", "b"], n=6)
        assert derived.reference_df == expected
        ratio = ratio_contrast(estimates, "a", "b", n=6, name="rr[a vs b]")
        assert ratio.reference_df == expected

    def test_a_median_over_repeats(self) -> None:
        reports = [{"a": curve_estimate("a", 9)}, {"a": curve_estimate("a", 6)}]
        assert median_estimates(reports)["a"].reference_df == 6

    def test_the_fit_conveniences(self, twelve: Any) -> None:
        assert twelve.contrast(lambda p: p[0] - p[1], ["ey1", "ey0"]).reference_df == 10
        assert twelve.ratio("ey1", "ey0").reference_df == 10
        assert twelve.ratio("ey1", "ey0", kind="or").reference_df == 10

    def test_a_maximum_rule_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            influence_module,
            "minimum_reference_df",
            lambda estimates: max(
                (e.reference_df for e in estimates if e.reference_df is not None), default=None
            ),
        )
        import cleverly.inference.results as results_module

        monkeypatch.setattr(
            results_module, "minimum_reference_df", influence_module.minimum_reference_df
        )
        with pytest.raises(AssertionError):
            self.test_a_smooth_contrast(6, 9, 6)


def _binary(k: int, **settings: Any) -> Any:
    return fit_clustered(make_binary_outcome(n=400, seed=17)[0], k, **settings)


def _linear(k: int, **settings: Any) -> Any:
    return fit_clustered(make_linear_ate(n=400, seed=2)[0], k, **settings)


def _missing(k: int, **settings: Any) -> Any:
    return fit_clustered(
        make_missing_outcome(n=400, seed=4)[0], k, roles={"delta": "Delta"}, **settings
    )


def _missing_binary(k: int, **settings: Any) -> Any:
    return fit_clustered(
        make_missing_outcome_binary(n=400, seed=4)[0], k, roles={"delta": "Delta"}, **settings
    )


def _shift(k: int) -> Any:
    return fit_clustered(make_shift_dose(n=300, seed=0)[0], k, shifts=[Shift(0.5, cap=5.0)])


def _regime(k: int) -> Any:
    return fit_clustered(
        discrete_law.frame(),
        k,
        interventions=regimes.interventions(),
        roles={"covariates": ("W",)},
    )


def _multi_arm(k: int) -> Any:
    return fit_clustered(make_multi_arm(n=600, seed=0)[0], k)


def _drtmle(k: int) -> Any:
    frame = with_clusters(make_linear_ate(n=400, seed=2)[0], k)
    return (
        DRTMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
            learner_folds=2,
            n_folds=3,
            q_bounds=(-20.0, 20.0),
            estimands=("ey0", "ey1", "ate"),
            random_state=0,
            simultaneous=False,
        )
        .fit(frame, outcome="Y", treatment="A", id="cluster")
        .single()
    )


def _ltmle_end_of_study(k: int) -> Any:
    frame = multivalue_panel(n=LONGITUDINAL_N, seed=43).assign(cluster=labels(LONGITUDINAL_N, k))
    return LTMLE({"never": 0, "always": 1}, reference="never", **longitudinal_learners()).fit(
        frame, **COLUMNS, id="cluster"
    )


def _ltmle_msm(k: int) -> Any:
    from cleverly.datasets import make_longitudinal

    frame, _ = make_longitudinal(n=LONGITUDINAL_N, seed=0)
    frame = frame.assign(cluster=labels(LONGITUDINAL_N, k))
    return LTMLE({"always": 1, "never": 0}, msm=DOSE, **longitudinal_learners()).fit(
        frame, outcome="Y", **LONGITUDINAL_NODES, id="cluster"
    )


SWEEP = {
    "binary means, contrasts and ratios": lambda k: _binary(
        k, estimands=("ey0", "ey1", "ate", "att", "atc", "rr", "or")
    ),
    "attributable effects": lambda k: _binary(k, estimands=("ey0", "par", "paf")),
    "continuous shift": _shift,
    "incremental": lambda k: _linear(k, incremental=[Incremental(2.0)]),
    "regimes": _regime,
    "working model": lambda k: _linear(k, msm=MSM.linear()),
    "missing outcome": _missing,
    "natural course": lambda k: _missing_binary(k, estimands=("ey_obs", "ey0", "par", "paf")),
    "multi-arm": _multi_arm,
    "DRTMLE": _drtmle,
    "LTMLE end of study": _ltmle_end_of_study,
    "LTMLE working model": _ltmle_msm,
}


class TestEveryEstimateCarriesItsReference:
    """Every estimate a clustered fit reports at J = 12 carries exactly 10 degrees of freedom."""

    @pytest.mark.parametrize("kind", list(SWEEP))
    def test_every_estimate_reads_ten(self, kind: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SWEEP[kind](12)
        assert result.inference_status == "influence_curve"
        assert result.estimates
        assert set(reference_dfs(result).values()) == {10}, reference_dfs(result)
        report = getattr(result, "cv_targeting", None)
        if report is not None:
            for estimates in (report.pooled, report.fold_evaluated):
                assert {e.reference_df for e in estimates.values()} == {10}

    def test_retarget_and_the_truncation_curve_keep_the_reference(self, twelve: Any) -> None:
        estimates, _ = twelve.estimator.retarget(
            twelve.data, twelve.nuisance, estimands=("ate",), g_bounds=(0.02, 0.98)
        )
        assert estimates["ate"].reference_df == 10
        curve = twelve.diagnostics.truncation_curve(bounds=[0.05])
        fitted = curve[curve["is_fitted_bound"]]
        for row in fitted.itertuples():
            assert (row.ci_lower, row.ci_upper) == pytest.approx(twelve[row.estimand].ci)

    def test_a_longitudinal_replay_keeps_the_reference(self) -> None:
        result = _ltmle_end_of_study(12)
        curve = result.diagnostics.truncation_curve(bounds=[0.05])
        assert len(curve) == len(result.estimates)


class TestTheFoldEvaluatedReference:
    """A fold-evaluated estimate over V folds takes min(J - 2, J - V).

    The fold-evaluated variance centres the cluster totals inside each fold, so it estimates
    one mean per fold and has J - V degrees of freedom, the pooled within-group count. The
    stacked report beside it keeps J - 2.
    """

    @pytest.mark.parametrize(
        ("clusters", "folds", "expected"), [(10, 5, 5), (12, 3, 9), (20, 4, 16), (30, 10, 20)]
    )
    def test_the_fold_evaluated_report_reads_j_minus_v(
        self, clusters: int, folds: int, expected: int
    ) -> None:
        result = _binary(
            clusters, cross_fit=True, n_folds=folds, cv_evaluation=True, estimands=("ey1", "ate")
        )
        assert set(reference_dfs(result).values()) == {expected}
        report = result.cv_targeting
        assert {e.reference_df for e in report.fold_evaluated.values()} == {expected}
        assert {e.reference_df for e in report.pooled.values()} == {clusters - 2}

    def test_two_folds_keep_j_minus_2(self) -> None:
        result = _binary(12, cross_fit=True, n_folds=2, cv_evaluation=True, estimands=("ate",))
        assert result["ate"].reference_df == 10

    def test_fold_targeting_keeps_j_minus_2_on_its_headline(self) -> None:
        result = _binary(
            12, cross_fit=True, n_folds=3, targeting_scheme="fold", estimands=("ey1", "ate")
        )
        assert set(reference_dfs(result).values()) == {10}
        assert {e.reference_df for e in result.cv_targeting.fold_evaluated.values()} == {9}

    def test_forty_clusters_keep_the_normal_reference(self) -> None:
        result = _binary(40, cross_fit=True, n_folds=5, cv_evaluation=True, estimands=("ate",))
        assert result["ate"].reference_df is None

    def test_dropping_the_fold_rule_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            influence_module.ClusterReference,
            "with_fold_evaluated",
            lambda self, names, folds: self,
        )
        with pytest.raises(AssertionError):
            self.test_the_fold_evaluated_report_reads_j_minus_v(10, 5, 5)


class TestTheSummaryNamesEachCount:
    def test_a_small_stratum_inside_a_large_fit(self) -> None:
        """44 clusters, 12 in one stratum: the note names both counts."""
        result = fit_strata(stratified(44, small=12))
        text = result.summary()
        assert T_REFERENCE_NOTE.format(clusters=44, fewest=12) in text
        assert NO_T_REFERENCE_BANDS in text
        assert "the fit reads 44 clusters with positive weight mass, fewer than" not in text


class TestTheLongitudinalDerivedEstimates:
    def test_rmst_contrast_and_ratio_take_ten(self) -> None:
        from cleverly.datasets import make_longitudinal_survival

        frame, _ = make_longitudinal_survival(n=LONGITUDINAL_N, seed=2)
        frame = frame.assign(cluster=labels(LONGITUDINAL_N, 12))
        result = LTMLE({"always": 1, "never": 0}, reference="never", **longitudinal_learners()).fit(
            frame, outcome=["Y1", "Y2"], **LONGITUDINAL_NODES, id="cluster"
        )
        assert set(reference_dfs(result).values()) == {10}
        assert result.rmst("always", 2).reference_df == 10
        assert result.rmst("always", 2, versus="never").reference_df == 10
        names = [name for name in result.estimates if name.startswith("risk_regimen[")][:2]
        assert result.contrast(lambda p: p[0] - p[1], names).reference_df == 10
        risks = sorted(n for n in result.estimates if n.startswith("risk_regimen[") and "t=2" in n)
        assert result.ratio(risks[0], risks[1]).reference_df == 10


class TestBandsRefuseTheTReference:
    def test_the_default_band_is_skipped_and_named(self) -> None:
        result = _binary(12, estimands=("ey0", "ey1", "ate"), simultaneous=True)
        assert result.simultaneous is None
        assert NO_T_REFERENCE_BANDS in result.summary()

    def test_an_explicit_band_raises(self, twelve: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            simultaneous_bands(twelve.estimates, cluster=twelve.data.cluster)
        assert str(raised.value) == T_REFERENCE_BAND_REFUSAL

    def test_forty_clusters_keep_the_default_band(self) -> None:
        result = _binary(40, estimands=("ey0", "ey1", "ate"), simultaneous=True)
        assert result.simultaneous is not None
        assert NO_T_REFERENCE_BANDS not in result.summary()


class TestTheBootstrapIsADiagnosticBelow40:
    def test_twelve_clusters_print_a_percentile_range(self) -> None:
        result = _binary(12, estimands=("ate",), n_bootstrap=20)
        text = result.summary()
        assert "percentile range" in text
        assert T_REFERENCE_BOOTSTRAP_NOTE in text
        assert "percentile CI" not in text
        row = result["ate"].to_dict()
        assert {"bootstrap_sd", "bootstrap_range_lower", "bootstrap_range_upper"} <= set(row)
        assert "bootstrap_ci_lower" not in row
        # The t note and the df column appear with the table.
        assert T_REFERENCE_NOTE.format(clusters=12, fewest=12) in text
        assert "p_value  df" in text

    def test_forty_clusters_keep_the_percentile_interval(self) -> None:
        result = _binary(40, estimands=("ate",), n_bootstrap=20)
        assert "percentile CI" in result.summary()
        assert "bootstrap_ci_lower" in result["ate"].to_dict()

    def test_a_label_that_reads_the_status_only_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            influence_module,
            "bootstrap_is_diagnostic",
            lambda estimate: (
                not (
                    estimate.supplies_inference
                    and estimate.bootstrap is not None
                    and estimate.bootstrap.inferential
                )
            ),
        )
        with pytest.raises(AssertionError):
            self.test_twelve_clusters_print_a_percentile_range()


class TestTheConsumersReadTheReference:
    def test_the_missingness_tilt_interval(self) -> None:
        result = _missing(12, estimands=("ey0", "ey1", "ate"))
        frame = missingness_tilt(result, [0.0], estimands=["ate"])
        row = frame.iloc[0]
        critical = float(stats.t.ppf(0.975, 10))
        assert row["ci_lower"] == pytest.approx(row["psi"] - critical * row["std_err"], rel=1e-12)

    def test_the_omitted_variable_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The limit's half-width is the t(10) quantile times its standard error."""
        result = _linear(12, estimands=("ate",))
        t_bounds = omitted_variable_bounds(result, "ate", cf_y=0.02, cf_d=0.02)
        import cleverly.sensitivity.omitted_variable as omitted

        monkeypatch.setattr(
            omitted, "one_sided_quantile", lambda level, df=None: stats.norm.ppf(level)
        )
        normal_bounds = omitted_variable_bounds(result, "ate", cf_y=0.02, cf_d=0.02)
        ratio = (t_bounds.lower - t_bounds.ci_lower) / (
            normal_bounds.lower - normal_bounds.ci_lower
        )
        assert ratio == pytest.approx(stats.t.ppf(0.95, 10) / stats.norm.ppf(0.95), rel=1e-10)

    def test_the_wald_test(self, twelve: Any) -> None:
        test = twelve["ate"].wald_test(null=0.05)
        assert test.pvalue == pytest.approx(2.0 * stats.t.sf(abs(test.statistic), 10), rel=1e-12)

    def test_variable_importance(self) -> None:
        frame = with_clusters(make_binary_outcome(n=400, seed=17)[0], 12)
        result = variable_importance(
            frame,
            outcome="Y",
            candidates=["A"],
            covariates=["W1", "W2"],
            estimator=TMLE(**linear_in_sample(estimands=("ate",))),
            id="cluster",
        )
        (entry,) = result.entries
        z = entry.estimate.psi / entry.estimate.std_error
        assert entry.estimate.reference_df == 10
        assert entry.adjusted_pvalue == pytest.approx(2.0 * stats.t.sf(abs(z), 10), rel=1e-12)


#: The modules allowed to read a normal or t quantile directly, and why.
ALLOWED = {
    "inference/delta.py": "the one home of the reference quantile",
    "inference/multiplier.py": "the pointwise critical value of a band, which refuses a t estimate",
    "validation/nuisance.py": "a Bonferroni calibration screen, not an interval",
    "datasets/synthetic.py": "a data generator",
    "sensitivity/simulated_confounding.py": "a latent perturbation threshold",
    "learners/screeners.py": "a screening p-value, not a reported interval",
}

FORBIDDEN_ATTRIBUTES = {"ppf", "sf", "cdf", "isf"}
FORBIDDEN_NAMES = {"ndtr", "ndtri", "erfinv", "NormalDist"}


def quantile_reads(source: str) -> list[str]:
    """Every normal or t quantile, tail, ``NormalDist`` or ``1.96`` literal in ``source``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                owner = node.value
                owner_name = (
                    owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
                )
                if owner_name in {"norm", "t"}:
                    found.append(f"{owner_name}.{node.attr}")
            if node.attr in FORBIDDEN_NAMES:
                found.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append(node.id)
        elif isinstance(node, ast.Constant) and node.value == 1.96:
            found.append("1.96")
    return found


class TestNoSecondNormalQuantile:
    def test_only_the_allowed_modules_read_a_quantile(self) -> None:
        package = ROOT / "src" / "cleverly"
        offenders = {}
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(package).as_posix()
            if relative in ALLOWED:
                continue
            reads = quantile_reads(path.read_text(encoding="utf-8"))
            if reads:
                offenders[relative] = reads
        assert not offenders

    def test_every_allowed_module_still_reads_one(self) -> None:
        package = ROOT / "src" / "cleverly"
        for relative in ALLOWED:
            assert quantile_reads((package / relative).read_text(encoding="utf-8")), relative

    def test_a_stray_call_is_found(self) -> None:
        stray = "from scipy import stats\nz = stats.norm.ppf(0.975)\nw = 1.96\n"
        assert sorted(quantile_reads(stray)) == ["1.96", "norm.ppf"]


def test_the_frame_stays_pandas() -> None:
    """The sweep frames carry the cluster column the fits declare."""
    assert isinstance(with_clusters(pd.DataFrame({"x": range(4)}), 2), pd.DataFrame)
