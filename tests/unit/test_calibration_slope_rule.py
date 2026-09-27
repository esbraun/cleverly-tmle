"""The calibration-slope rule of RM15 in ``docs/roadmap.md``.

Before RM15, ``NuisanceDiagnostics.findings`` warned when a pooled one-intercept recalibration
slope fell outside the fixed band [0.7, 1.4], and it said that the model "biases the weights".
The band warned on fits of the true propensity and of a correct weak-signal model, and on every
mean-only fit of a randomized law, where the pooled intercept makes the slope -2.

This module pins the replacement:

* the statistic is a weighted logistic recalibration with one intercept per validation fold and a
  common slope, and its standard error is the sandwich read through ``influence_variance`` with
  the fit's cluster codes.  W1 checks both against an independent numpy fit to 1e-10, and W2, W3
  and the pooled-fit contrast are nonzero witnesses for the weights, the clusters and the fold
  intercepts;
* ``nuisance_diagnostics`` passes the propensity and outcome reports the fit's folds and cluster
  codes, and the response model's largest inverse weight reads respondents only;
* a prediction that is constant within every fold, or that separates the labels within every
  fold, has no slope, and the report names the reason;
* a finding needs the Bonferroni interval over the eligible models to lie above 0 and exclude 1;
* an in-sample fit carries no test;
* the message reports the interval, the AUC and, for a weight model, the largest untruncated
  inverse weight, and it makes no claim that the weights are biased;
* a conditional-mean report names its linear slope ``regression_slope``.

Every name is read off the module inside each test, so the file collects on a tree without the
rule and each test fails for its own reason.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_linear_ate
from cleverly.estimators import TMLE
from cleverly.validation import nuisance as nuisance_module

# ---------------------------------------------------------------------------- helpers


def _logistic() -> LogisticRegression:
    """An unpenalized main-effects logistic model."""
    return LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=1000, tol=1e-12)


class Tempered(BaseEstimator, ClassifierMixin):
    """An unpenalized logistic model whose logit is multiplied by ``k``.

    ``k = 2`` makes the predictions too extreme, for a limit slope of 1/2. ``k = 1/2`` makes
    them too moderate, for a limit slope of 2.
    """

    def __init__(self, k: float = 1.0) -> None:
        self.k = k

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Tempered:
        self.model_ = _logistic().fit(design, target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = expit(self.k * self.model_.decision_function(design))
        return np.column_stack([1.0 - p, p])


class MeanOnly(BaseEstimator, ClassifierMixin):
    """Predicts the training mean for every row, so each fold's prediction is one constant."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> MeanOnly:
        self.classes_ = np.array([0.0, 1.0])
        self.p_ = float(np.average(target, weights=sample_weight))
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = np.full(np.asarray(design).shape[0], self.p_)
        return np.column_stack([1.0 - p, p])


def _law(n: int, p: int, coefficients: tuple[float, ...], seed: int) -> pd.DataFrame:
    """``logit g0 = sum_j c_j W_j`` and ``Y ~ Bernoulli(expit(0.5 A + 0.5 W1 - 0.25))``."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, p))
    eta = sum(c * w[:, j] for j, c in enumerate(coefficients))
    a = rng.binomial(1, expit(eta)).astype(float)
    y = rng.binomial(1, expit(0.5 * a + 0.5 * w[:, 0] - 0.25)).astype(float)
    frame = pd.DataFrame(w, columns=[f"W{j + 1}" for j in range(p)])
    frame["A"] = a
    frame["Y"] = y
    return frame


def _fit(frame: pd.DataFrame, treatment_learner: Any, *, seed: int, **settings: Any) -> Any:
    options: dict[str, Any] = {"n_folds": 3, **settings}
    return (
        TMLE(
            outcome_learner=_logistic(),
            treatment_learner=treatment_learner,
            simultaneous=False,
            estimands=("ate",),
            random_state=seed,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def _calibration_notes(report: Any) -> list[str]:
    return [note for note in report.findings if "calibration slope" in note]


def _with_metrics(report: Any, name: str, **metrics: float) -> Any:
    """``report`` with the named model's metrics overwritten."""
    models = tuple(
        replace(model, metrics={**model.metrics, **metrics}) if model.name == name else model
        for model in report.models
    )
    return replace(report, models=models)


def _reference(
    predicted: np.ndarray,
    actual: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    cluster: np.ndarray | None = None,
    *,
    pooled: bool = False,
) -> tuple[float, float]:
    """An independent fold-intercept recalibration and its hand-built sandwich SE.

    Plain Newton steps on the weighted log likelihood, and the sandwich
    ``B^-1 M B^-1`` with the small-sample factor ``G / (G - 1)`` that a variance with
    ``ddof=1`` over ``G`` independent units carries.
    """
    lp = logit(predicted)
    levels = np.unique(folds)
    columns = [np.ones_like(lp)] if pooled else [(folds == k).astype(float) for k in levels]
    x = np.column_stack([*columns, lp])
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        mu = expit(x @ beta)
        information = x.T @ (x * (weights * mu * (1.0 - mu))[:, None])
        step = np.linalg.solve(information, x.T @ (weights * (actual - mu)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-14:
            break
    mu = expit(x @ beta)
    information = x.T @ (x * (weights * mu * (1.0 - mu))[:, None])
    inverse = np.linalg.inv(information)
    score = x * (weights * (actual - mu))[:, None]
    if cluster is None:
        units = score
    else:
        codes = np.unique(cluster)
        units = np.vstack([score[cluster == code].sum(axis=0) for code in codes])
    count = units.shape[0]
    sandwich = count / (count - 1) * (inverse @ (units.T @ units) @ inverse)
    return float(beta[-1]), float(np.sqrt(sandwich[-1, -1]))


def _sample(n: int = 600, seed: int = 11) -> dict[str, np.ndarray]:
    """A compact miscalibrated sample with three folds, non-uniform weights and clusters.

    Each fold's predictions carry their own offset, so the fold intercepts differ and a pooled
    intercept gives a different slope.  Each cluster of 20 rows shares an effect on the label,
    so the cluster-sum variance differs from the row-level one.
    """
    rng = np.random.default_rng(seed)
    folds = np.arange(n) % 3
    cluster = np.arange(n) // 20
    signal = rng.standard_normal(n)
    offsets = np.array([-0.6, 0.0, 0.7])[folds]
    predicted = expit(0.9 * signal + offsets)
    shared = rng.standard_normal(cluster.max() + 1)[cluster]
    actual = rng.binomial(1, expit(0.6 * signal + 1.2 * shared)).astype(float)
    weights = rng.uniform(0.3, 3.0, n)
    return {
        "predicted": predicted,
        "actual": actual,
        "weights": weights,
        "folds": folds,
        "cluster": cluster,
    }


def _report(
    sample: dict[str, np.ndarray],
    *,
    weights: np.ndarray | None = None,
    cluster: np.ndarray | None = None,
) -> Any:
    return nuisance_module._binary_report(
        "propensity",
        sample["predicted"],
        sample["actual"],
        sample["weights"] if weights is None else weights,
        None,
        folds=sample["folds"],
        cluster=cluster,
    )


# ------------------------------------------------------------ W1 to W4: the statistic


class TestTheStatistic:
    def test_w1_the_slope_and_its_sandwich_se_equal_an_independent_fit(self) -> None:
        sample = _sample()
        report = _report(sample)
        slope, se = _reference(
            sample["predicted"], sample["actual"], sample["weights"], sample["folds"]
        )

        assert report.metrics["calibration_slope"] == pytest.approx(slope, rel=1e-10, abs=1e-10)
        assert report.metrics["calibration_slope_se"] == pytest.approx(se, rel=1e-10, abs=1e-10)
        assert report.calibration_omission is None

        # Nonzero witness for the fold intercepts: on this sample one pooled intercept gives
        # another slope, so a fit that dropped the fold indicators would miss W1.
        pooled, _ = _reference(
            sample["predicted"],
            sample["actual"],
            sample["weights"],
            sample["folds"],
            pooled=True,
        )
        assert abs(pooled - slope) > 1e-2

    def test_w2_the_weights_move_the_slope_and_the_se(self) -> None:
        sample = _sample()
        weighted = _report(sample)
        unit = _report(sample, weights=np.ones_like(sample["weights"]))
        slope, se = _reference(
            sample["predicted"], sample["actual"], np.ones_like(sample["weights"]), sample["folds"]
        )

        assert unit.metrics["calibration_slope"] == pytest.approx(slope, rel=1e-10)
        assert unit.metrics["calibration_slope_se"] == pytest.approx(se, rel=1e-10)
        assert abs(unit.metrics["calibration_slope"] - weighted.metrics["calibration_slope"]) > 1e-3
        assert (
            abs(unit.metrics["calibration_slope_se"] - weighted.metrics["calibration_slope_se"])
            > 1e-3
        )

    def test_w3_the_clusters_move_the_se_to_the_cluster_sum_sandwich(self) -> None:
        sample = _sample()
        clustered = _report(sample, cluster=sample["cluster"])
        rows = _report(sample)
        _, se = _reference(
            sample["predicted"],
            sample["actual"],
            sample["weights"],
            sample["folds"],
            sample["cluster"],
        )

        assert clustered.metrics["calibration_slope_se"] == pytest.approx(se, rel=1e-10)
        # The slope is the same fit either way. Only its standard error reads the clusters.
        assert clustered.metrics["calibration_slope"] == pytest.approx(
            rows.metrics["calibration_slope"], rel=1e-12
        )
        ratio = clustered.metrics["calibration_slope_se"] / rows.metrics["calibration_slope_se"]
        assert abs(ratio - 1.0) > 0.10

    def test_one_cluster_gives_no_standard_error_and_names_why(self) -> None:
        sample = _sample()
        report = _report(sample, cluster=np.zeros_like(sample["cluster"]))

        assert np.isnan(report.metrics["calibration_slope"])
        assert np.isnan(report.metrics["calibration_slope_se"])
        assert report.calibration_omission is not None
        assert "cluster" in report.calibration_omission

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_predictions_that_separate_the_labels_have_no_slope(self, sign: float) -> None:
        """Complete separation, in either direction, leaves the slope without a maximum.

        The labels are ``1{sign * z > 0}`` and the predictions ``expit(z)``.  Without the
        separation check the Newton fit meets its score tolerance at a slope near
        ``sign * 8800`` with a finite standard error, and a positive slope gives a "more
        moderate" finding.
        """
        rng = np.random.default_rng(0)
        n = 900
        signal = rng.standard_normal(n)
        model = nuisance_module._binary_report(
            "propensity",
            expit(signal),
            (sign * signal > 0.0).astype(float),
            np.ones(n),
            None,
            folds=np.arange(n) % 3,
            cluster=None,
        )

        assert np.isnan(model.metrics["calibration_slope"])
        assert np.isnan(model.metrics["calibration_slope_se"])
        assert model.calibration_omission == nuisance_module.CALIBRATION_SEPARATED
        assert nuisance_module._calibration_finding(model, 1) is None

    def test_a_fold_with_one_class_keeps_the_slope_of_the_other_folds(self) -> None:
        """One fold whose labels are all one class adds nothing about the slope.

        Its intercept grows without bound, and the slope stays the one the other folds give.
        The separation check must not refuse this fit.
        """
        sample = _sample()
        actual = sample["actual"].copy()
        actual[sample["folds"] == 2] = 1.0
        whole = nuisance_module._binary_report(
            "propensity",
            sample["predicted"],
            actual,
            sample["weights"],
            None,
            folds=sample["folds"],
            cluster=None,
        )
        keep = sample["folds"] != 2
        rest = nuisance_module._binary_report(
            "propensity",
            sample["predicted"][keep],
            actual[keep],
            sample["weights"][keep],
            None,
            folds=sample["folds"][keep],
            cluster=None,
        )

        assert whole.calibration_omission is None
        assert whole.metrics["calibration_slope"] == pytest.approx(
            rest.metrics["calibration_slope"], rel=1e-8
        )

    def test_w4_a_fold_constant_prediction_has_no_slope(self) -> None:
        frame = _law(2000, 4, (0.0,), seed=7)
        result = _fit(frame, MeanOnly(), seed=7)
        report = result.diagnostics.nuisance_models()
        propensity = report["propensity"]

        assert np.isnan(propensity.metrics["calibration_slope"])
        assert np.isnan(propensity.metrics["calibration_slope_se"])
        assert propensity.calibration_omission == nuisance_module.CALIBRATION_CONSTANT_WITHIN_FOLDS
        assert _calibration_notes(report) == []

        # The pooled recalibration of these very predictions reads -(V - 1) = -2 at three
        # folds, which is the value the band used to warn on.
        predicted = np.clip(result.nuisance.propensity.arm(1.0), 1e-12, 1.0 - 1e-12)
        folds = np.asarray(result.nuisance.folds.assignment)
        pooled, _ = _reference(
            predicted, result.data.treatment, np.ones(result.n), folds, pooled=True
        )
        assert pooled == pytest.approx(-2.0, abs=5e-2)

        summary = report.summary()
        assert (
            f"calibration slope unavailable for propensity: "
            f"{nuisance_module.CALIBRATION_CONSTANT_WITHIN_FOLDS}" in summary
        )


class TestTheCallSites:
    """``nuisance_diagnostics`` passes each probability report the fit's folds and clusters.

    W1 to W4 drive ``_binary_report`` directly, so they cannot see a call site that drops an
    argument.  These tests read the reports of a clustered cross-fitted fit with a binary
    outcome.  On this fit the clusters move each standard error and the fold intercepts move
    each slope, so a call site that dropped either would fail the equality.
    """

    @pytest.fixture(scope="class")
    def clustered(self) -> Any:
        from cleverly.datasets import make_clustered

        frame, _ = make_clustered(n=400, cluster_size=10, seed=7, family="binomial")
        return (
            TMLE(
                outcome_learner=_logistic(),
                treatment_learner=_logistic(),
                n_folds=3,
                simultaneous=False,
                estimands=("ate",),
                random_state=7,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2"), id="cluster")
            .single()
        )

    @pytest.mark.parametrize("name", ["propensity", "outcome"])
    def test_a_report_reads_the_fit_folds_and_cluster_codes(
        self, clustered: Any, name: str
    ) -> None:
        data = clustered.data
        folds = np.asarray(clustered.nuisance.folds.assignment)
        if name == "propensity":
            arguments = (name, clustered.nuisance.propensity.arm(1.0), data.treatment)
        else:
            arguments = (name, clustered.nuisance.outcome.observed, data.outcome)
        options = {"mask": None if name == "propensity" else data.observed}

        def direct(fold: np.ndarray, cluster: Any) -> dict[str, float]:
            return nuisance_module._binary_report(
                *arguments, data.weights, None, folds=fold, cluster=cluster, **options
            ).metrics

        reported = clustered.diagnostics.nuisance_models()[name].metrics
        expected = direct(folds, data.cluster)

        assert reported["calibration_slope"] == expected["calibration_slope"]
        assert reported["calibration_slope_se"] == expected["calibration_slope_se"]
        rows = direct(folds, None)
        assert abs(rows["calibration_slope_se"] - expected["calibration_slope_se"]) > 1e-3
        pooled = direct(np.zeros_like(folds), data.cluster)
        assert abs(pooled["calibration_slope"] - expected["calibration_slope"]) > 1e-2


# ------------------------------------------------------------ W5 to W9: the rule


class TestTheRule:
    def test_w5_the_correct_weak_signal_law_gives_no_finding_where_the_band_fired(self) -> None:
        """The roadmap law, at a seed where the fixed band fires on both slopes.

        A correct unpenalized logistic model on ``W1`` for ``logit g0 = 0.15 W1``.  At seed
        193 the fold-intercept slope is about 0.62 and the pooled slope about 0.60, both
        outside [0.7, 1.4].  The Bonferroni interval lies above 0 and contains 1, so the
        rule, not the gate at 0, keeps the report quiet.
        """
        result = _fit(_law(2000, 1, (0.15,), seed=193), _logistic(), seed=193)
        report = result.diagnostics.nuisance_models()
        propensity = report["propensity"].metrics
        slope, se = propensity["calibration_slope"], propensity["calibration_slope_se"]
        predicted = np.clip(result.nuisance.propensity.arm(1.0), 1e-12, 1.0 - 1e-12)
        pooled, _ = _reference(
            predicted,
            result.data.treatment,
            np.ones(result.n),
            np.asarray(result.nuisance.folds.assignment),
            pooled=True,
        )

        assert not 0.7 <= slope <= 1.4
        assert not 0.7 <= pooled <= 1.4
        z = float(stats.norm.ppf(1.0 - 0.05 / 4.0))
        assert slope - z * se > 0.0
        assert slope - z * se <= 1.0 <= slope + z * se
        assert _calibration_notes(report) == []
        assert result.diagnostics.run_all()["nuisance_models"].status.value == "completed"

    @pytest.mark.parametrize(
        ("k", "direction", "other"),
        [(2.0, "more extreme", "more moderate"), (0.5, "more moderate", "more extreme")],
    )
    def test_w6_a_miscalibrated_learner_warns_in_its_own_direction(
        self, k: float, direction: str, other: str
    ) -> None:
        result = _fit(_law(2000, 2, (1.0, -0.5), seed=0), Tempered(k), seed=0)
        report = result.diagnostics.nuisance_models()
        notes = _calibration_notes(report)

        assert len(notes) == 1
        assert notes[0].startswith("propensity: ")
        assert direction in notes[0]
        assert other not in notes[0]

        item = result.diagnostics.run_all()["nuisance_models"]
        assert item.status.value == "warning"
        assert notes[0] in item.detail

    def test_w7_an_interval_that_reaches_zero_gives_no_finding(self) -> None:
        """A randomized law, where a fitted propensity has no signal to recalibrate.

        At seed 3 the Bonferroni interval excludes 1 and contains 0.  Without the gate at 0,
        this fit would warn that the predictions are more extreme than the observed rates.
        """
        result = _fit(_law(2000, 4, (0.0,), seed=3), _logistic(), seed=3)
        report = result.diagnostics.nuisance_models()
        propensity = report["propensity"].metrics
        z = float(stats.norm.ppf(1.0 - 0.05 / 4.0))
        low = propensity["calibration_slope"] - z * propensity["calibration_slope_se"]
        high = propensity["calibration_slope"] + z * propensity["calibration_slope_se"]

        assert low < 0.0 < high < 1.0
        assert _calibration_notes(report) == []

    def test_an_interval_below_zero_gives_no_finding(self) -> None:
        result = _fit(_law(2000, 4, (0.0,), seed=14), _logistic(), seed=14)
        report = result.diagnostics.nuisance_models()
        propensity = report["propensity"].metrics
        z = float(stats.norm.ppf(1.0 - 0.05 / 4.0))

        assert propensity["calibration_slope"] + z * propensity["calibration_slope_se"] < 0.0
        assert _calibration_notes(report) == []

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            (0.2, 0.8, "more extreme"),
            (1.2, 1.8, "more moderate"),
            (-0.3, 0.5, None),
            (0.5, 1.3, None),
            (-1.2, -0.2, None),
        ],
    )
    def test_w8_the_truth_table_of_the_rule(
        self, low: float, high: float, expected: str | None
    ) -> None:
        z = float(stats.norm.ppf(0.975))
        model = nuisance_module.NuisanceModelReport(
            name="propensity",
            kind="probability",
            metrics={
                "auc": 0.6,
                "calibration_slope": (low + high) / 2.0,
                "calibration_slope_se": (high - low) / (2.0 * z),
                "largest_inverse_weight": 3.0,
            },
            calibration={},
            learner_weights={},
            learner_risks={},
        )
        note = nuisance_module._calibration_finding(model, 1)

        if expected is None:
            assert note is None
        else:
            assert note is not None
            assert expected in note
            assert f"interval {low:.2f} to {high:.2f}" in note

    def test_w8_a_missing_slope_gives_no_finding(self) -> None:
        model = nuisance_module.NuisanceModelReport(
            name="propensity",
            kind="probability",
            metrics={"calibration_slope": float("nan"), "calibration_slope_se": float("nan")},
            calibration={},
            learner_weights={},
            learner_risks={},
        )
        assert nuisance_module._calibration_finding(model, 1) is None

    def test_w9_the_level_is_split_over_the_eligible_models(self) -> None:
        """Two eligible models test at 97.5% each, and one alone at 95%.

        The propensity slope sits 2.1 standard errors below 1, between the two critical
        values 1.960 and 2.241.  The outcome model leaves the count when its standard error
        is missing, which is the one change between the two reports.
        """
        result = _fit(_law(600, 2, (1.0, -0.5), seed=1), _logistic(), seed=1)
        base = result.diagnostics.nuisance_models()
        two = _with_metrics(
            _with_metrics(base, "propensity", calibration_slope=0.79, calibration_slope_se=0.1),
            "outcome",
            calibration_slope=1.0,
            calibration_slope_se=0.1,
        )
        one = _with_metrics(two, "outcome", calibration_slope_se=float("nan"))

        assert _calibration_notes(two) == []
        notes = _calibration_notes(one)
        assert len(notes) == 1
        assert "95.0% interval" in notes[0]

        far = _with_metrics(two, "propensity", calibration_slope=0.5)
        notes = _calibration_notes(far)
        assert len(notes) == 1
        assert "97.5% interval" in notes[0]

    def test_w10_an_in_sample_fit_carries_no_test(self) -> None:
        """An unpenalized logistic model with an intercept has an in-sample slope of 1.

        Riley et al. (2021), Section 2.1.2.  A replaced slope of 0.3 with a standard error of
        0.01 would warn on an out-of-fold report, and it does not warn here.
        """
        frame = _law(2000, 2, (1.0, -0.5), seed=2)
        result = _fit(frame, _logistic(), seed=2, cross_fit=False)
        report = result.diagnostics.nuisance_models()

        assert report.evaluation == "in_sample"
        assert report["propensity"].metrics["calibration_slope"] == pytest.approx(1.0, abs=1e-8)
        forced = _with_metrics(
            report, "propensity", calibration_slope=0.3, calibration_slope_se=0.01
        )
        assert _calibration_notes(forced) == []
        assert _calibration_notes(replace(forced, evaluation="out_of_fold")) != []


# ------------------------------------------------------------ W11 to W13: the surfaces


class TestTheSurfaces:
    def test_w11_the_message_reports_the_interval_and_claims_no_bias(self) -> None:
        result = _fit(_law(2000, 2, (1.0, -0.5), seed=0), Tempered(2.0), seed=0)
        report = result.diagnostics.nuisance_models()
        forced = _with_metrics(report, "outcome", calibration_slope=0.5, calibration_slope_se=0.05)
        notes = _calibration_notes(forced)

        assert len(notes) == 2
        propensity = next(note for note in notes if note.startswith("propensity: "))
        outcome = next(note for note in notes if note.startswith("outcome: "))
        auc = report["propensity"].metrics["auc"]
        weight = report["propensity"].metrics["largest_inverse_weight"]
        assert f"AUC {auc:.3f}" in propensity
        assert f"largest untruncated inverse weight {weight:.3g}" in propensity
        assert "inverse weight" not in outcome
        for note in forced.findings:
            assert "biases the weights" not in note
            assert "poorly calibrated" not in note

    def test_the_largest_inverse_weight_reads_the_observed_treatment(self) -> None:
        result = _fit(_law(600, 2, (1.0, -0.5), seed=4), _logistic(), seed=4)
        report = result.diagnostics.nuisance_models()
        g = np.clip(result.nuisance.propensity.arm(1.0), 1e-12, 1.0 - 1e-12)
        observed = np.where(result.data.treatment == 1.0, g, 1.0 - g)

        assert report["propensity"].metrics["largest_inverse_weight"] == pytest.approx(
            float(np.max(1.0 / observed)), rel=1e-12
        )
        assert "largest_inverse_weight" not in report["outcome"].metrics

    def test_the_missingness_weight_reads_the_respondents_only(self) -> None:
        """Only a row with a recorded outcome carries ``1 / P(Delta = 1 | A, W)``.

        On this draw the largest ``1 / (1 - p)`` over the nonrespondents is larger than the
        largest ``1 / p`` over the respondents, so a report that read every row would differ.
        The fit is in sample, because a cross-fitted continuous outcome with missing values
        needs declared outcome bounds, and the mask does not depend on the folds.
        """
        from cleverly.datasets import make_missing_outcome

        frame, _ = make_missing_outcome(n=400, seed=4)
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=_logistic(),
                missingness_learner=_logistic(),
                cross_fit=False,
                simultaneous=False,
                estimands=("ate",),
                random_state=4,
            )
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        report = result.diagnostics.nuisance_models()
        response = np.asarray(result.nuisance.missingness, dtype=float)
        arm = np.where(result.data.treatment == 1.0, response[:, 1], response[:, 0])
        p = np.clip(arm, 1e-12, 1.0 - 1e-12)
        observed = result.data.observed

        expected = float(np.max(1.0 / p[observed]))
        assert report["missingness"].metrics["largest_inverse_weight"] == pytest.approx(
            expected, rel=1e-12
        )
        assert float(np.max(1.0 / (1.0 - p[~observed]))) > expected

    def test_w12_a_conditional_mean_report_names_a_regression_slope(self) -> None:
        frame, _ = make_linear_ate(n=300, seed=1)
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=_logistic(),
                cross_fit=False,
                simultaneous=False,
                estimands=("ate",),
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        report = result.diagnostics.nuisance_models()
        outcome = report["outcome"]

        assert "regression_slope" in outcome.metrics
        assert "calibration_slope" not in outcome.metrics
        assert "calibration_slope_se" not in outcome.metrics
        frame_out = report.to_frame()
        row = frame_out.loc[frame_out["model"] == "outcome"].iloc[0]
        assert row["regression_slope"] == pytest.approx(outcome.metrics["regression_slope"])
        assert np.isnan(row["calibration_slope"])

    def test_w13_the_summary_states_the_basis_of_the_slope(self) -> None:
        crossed = _fit(_law(600, 2, (1.0, -0.5), seed=5), _logistic(), seed=5)
        summary = crossed.diagnostics.nuisance_models().summary()
        header = next(line for line in summary.splitlines() if line.startswith("model"))
        for column in ("cal_slope", "cal_se", "reg_slope"):
            assert column in header
        assert "one intercept per fold" in summary
        assert "97.5% interval" in summary

        in_sample = _fit(_law(600, 2, (1.0, -0.5), seed=5), _logistic(), seed=5, cross_fit=False)
        summary = in_sample.diagnostics.nuisance_models().summary()
        assert "in sample and carry no test" in summary
        assert "one intercept per fold" not in summary
