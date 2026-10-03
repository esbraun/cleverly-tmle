"""A cross-fitted clustered ``LTMLE`` fit: whole-cluster folds and a cluster-summed curve.

Díaz, Williams, Hoffman and Schenck (2023), Section 5.2 and Theorem 3, give the cross-fitted
TMLE over a random partition of iid units. With the cluster as the unit, the partition is
whole-cluster, every inner Super Learner split is grouped on the same labels, the targeting
is the shipped pooled fluctuation, and the variance is
:math:`J\\,\\widehat{\\mathrm{var}}(S_c)/n^2` over the cluster sums :math:`S_c` of the row
curve. Each class below pairs a witness with a mutation control that must fail it.

The fits use the clustered study law of ``tests/studies/clustered_longitudinal_laws.py``,
whose outcome is correlated within a cluster, and the ``canonical-ltmle-crossfit`` subject
(quasi-binomial node regressions and the law's own mechanism), which fits in about 0.1 s.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    AssessmentStatus,
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    RegimeMean,
    TMLEMethod,
)
from cleverly._inference_status import (
    FEW_CLUSTER_THRESHOLD,
    MINIMUM_INTERVAL_CLUSTERS,
    NO_T_REFERENCE_BANDS,
)
from cleverly.datasets import make_longitudinal_competing
from cleverly.exceptions import CapabilityError, DataError
from cleverly.inference import cluster as cluster_module
from cleverly.learners.crossfit import Folds, random_partition
from cleverly.longitudinal import LTMLE
from tests.studies import canonical_ltmle_crossfit as subject_study
from tests.studies import clustered_longitudinal_laws as law
from tests.unit._inference_status_support import cluster_labels
from tests.unit.test_longitudinal_msm import DOSE
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

pytestmark = pytest.mark.xdist_group("clustered_crossfit_ltmle")

longitudinal_estimator = importlib.import_module("cleverly.longitudinal.estimator")
sequential_module = importlib.import_module("cleverly.longitudinal.sequential")

NODES: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
SURVIVAL: dict[str, Any] = {"outcome": ["Y1", "Y2"], **NODES}
COMPETING: dict[str, Any] = {"outcome": {"relapse": ["R1", "R2"], "death": ["D1", "D2"]}, **NODES}
CONTRAST = "ate_regimen[always vs never]"
#: Seeds of the law draws below. Each one draws a split whose training folds support every
#: node, which is a property of the draw and not a choice of result.
SEED = 1
UNEQUAL_SEED = 20261101


def subject(regimens: Any = None, **settings: Any) -> LTMLE:
    """The ``canonical-ltmle-crossfit`` subject at five whole-cluster folds."""
    configuration: dict[str, Any] = {
        "reference": subject_study.REFERENCE,
        "outcome_learner": subject_study.QuasiBinomialGLM(),
        "pseudo_learner": subject_study.QuasiBinomialGLM(),
        "treatment_learner": subject_study.KnownLongitudinalMechanism("treatment"),
        "censoring_learner": subject_study.KnownLongitudinalMechanism("censoring"),
        "n_folds": 5,
        "learner_folds": 2,
        "g_bounds": subject_study.G_BOUNDS,
        "simultaneous": False,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
        **settings,
    }
    return LTMLE(subject_study.REGIMENS if regimens is None else regimens, **configuration)


def fit_law(clusters: int, sizes: str = "equal40", seed: int = SEED, **settings: Any) -> Any:
    """One cross-fitted clustered fit on the end-of-study study law."""
    frame, _ = law.draw_end_of_study(clusters, sizes, seed)
    return subject(**settings).fit(frame, outcome="Y", id="id", **NODES)


def cluster_variance(curve: Any, cluster: Any) -> float:
    """:math:`J\\,\\widehat{\\mathrm{var}}(S_c)/n^2`, written out without the package."""
    curve = np.asarray(curve, dtype=float)
    sums = np.bincount(np.asarray(cluster), weights=curve)
    return float(sums.size * np.var(sums, ddof=1) / curve.size**2)


def assert_cluster_summed(estimates: Any, cluster: Any) -> None:
    """Every estimate's variance is the cluster-sum variance of its own curve."""
    assert estimates
    for name, estimate in estimates.items():
        assert estimate.variance == pytest.approx(
            cluster_variance(estimate.influence_curve, cluster), rel=1e-12
        ), name


def assert_whole_clusters(folds: Folds, cluster: Any) -> None:
    frame = pd.DataFrame({"cluster": np.asarray(cluster), "fold": np.asarray(folds.assignment)})
    assert frame.groupby("cluster")["fold"].nunique().max() == 1


class CountingClassifier(LogisticRegression):
    """A logistic regression that counts its fits, to show a refusal came first."""

    calls: ClassVar[int] = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CountingClassifier:
        CountingClassifier.calls += 1
        return super().fit(X, y, sample_weight=sample_weight)


#: The offset of the ``site`` column: the cluster code plus this value, so that a learner can
#: find the column by its range without knowing its position in a node's design.
SITE_OFFSET = 1000.0
_GROUPS_SEEN: list[tuple[np.ndarray, np.ndarray | None]] = []


class SiteMean(BaseEstimator):
    """The training mean of the rows that share a row's ``site``, or the training mean.

    It records, at every ``fit``, the site codes of the rows it received and the ``groups``
    the package passed. A learner that names ``groups`` is the one the package routes the
    cluster codes to.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None, groups: Any = None) -> SiteMean:
        del sample_weight
        site = self._site(X)
        _GROUPS_SEEN.append((site, None if groups is None else np.asarray(groups)))
        target = np.asarray(y, dtype=float)
        self.classes_ = np.array([0.0, 1.0])
        self.fallback_ = float(target.mean())
        self.means_ = pd.Series(target).groupby(site).mean().to_dict()
        return self

    @staticmethod
    def _site(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        columns = [j for j in range(matrix.shape[1]) if np.all(matrix[:, j] >= SITE_OFFSET)]
        assert len(columns) == 1, "the site column is not in this design"
        return matrix[:, columns[0]] - SITE_OFFSET

    def predict(self, X: Any) -> np.ndarray:
        site = self._site(X)
        return np.array([self.means_.get(code, self.fallback_) for code in site])

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


def site_frame(clusters: int = 40, sizes: str = "equal10", seed: int = SEED) -> pd.DataFrame:
    frame, _ = law.draw_end_of_study(clusters, sizes, seed)
    return frame.assign(site=frame["id"] + SITE_OFFSET)


SITE_NODES = {**NODES, "baseline": ["W1", "W2", "site"]}


class TestTheFoldsKeepEveryClusterWhole:
    def test_the_split_is_the_grouped_draw(self) -> None:
        result = fit_law(40, "equal10", random_state=7)
        assert result.folds.origin.scheme == "grouped"
        assert_whole_clusters(result.folds, result.data.cluster)
        expected = random_partition(result.n, 5, cluster=result.data.cluster, seed=7)
        assert np.array_equal(result.folds.assignment, expected.assignment)

    def test_few_clusters_cap_the_fold_count_at_the_cluster_count(self) -> None:
        with pytest.warns(UserWarning, match="reducing n_folds from 10 to 4"):
            result = fit_law(4, "equal100", n_folds=10)
        assert result.folds.n_folds == 4
        assert_whole_clusters(result.folds, result.data.cluster)

    @pytest.fixture(autouse=True)
    def _reset_counter(self) -> None:
        CountingClassifier.calls = 0

    def test_a_split_that_drops_the_clusters_is_refused_before_any_learner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the draw forgets ``cluster=``. The check at the call site refuses it."""

        def rows(n: int, n_folds: int, *, cluster: Any = None, seed: Any = None) -> Folds:
            del cluster
            return random_partition(n, n_folds, seed=seed)

        monkeypatch.setattr(longitudinal_estimator, "random_partition", rows)
        with pytest.raises(DataError, match="have rows in more than one fold"):
            fit_law(40, "equal10", outcome_learner=CountingClassifier(max_iter=1000))
        assert CountingClassifier.calls == 0

    def test_an_overridden_draw_is_held_to_the_same_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: ``_folds`` itself draws rows. The check sits outside it, so it still runs."""

        def row_folds(self: LTMLE, data: Any) -> Folds:
            return random_partition(data.n, self.n_folds, seed=0)

        monkeypatch.setattr(LTMLE, "_folds", row_folds)
        with pytest.raises(DataError, match="have rows in more than one fold"):
            fit_law(40, "equal10", outcome_learner=CountingClassifier(max_iter=1000))
        assert CountingClassifier.calls == 0


def _site_subject(**settings: Any) -> LTMLE:
    return LTMLE(
        {"always": 1, "never": 0},
        reference="never",
        outcome_learner=SiteMean(),
        pseudo_learner=SiteMean(),
        treatment_learner=settings.pop("treatment_learner", SiteMean()),
        censoring_learner=settings.pop("censoring_learner", SiteMean()),
        n_folds=5,
        simultaneous=False,
        random_state=0,
        **settings,
    )


class TestTheInnerFoldsAreGrouped:
    """Every nuisance fit receives the cluster codes of exactly the rows it trains on."""

    @staticmethod
    def recorded(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray | None]]:
        _GROUPS_SEEN.clear()
        result = _site_subject().fit(frame, outcome="Y", id="id", **SITE_NODES)
        assert result.folds.n_folds == 5
        return list(_GROUPS_SEEN)

    def test_every_fit_receives_the_codes_of_its_rows(self) -> None:
        frame = site_frame()
        seen = self.recorded(frame)
        # Two mechanisms at two nodes and two regimens at two nodes, five folds each, and
        # nothing else would receive the codes.
        assert len(seen) >= 4 * 5
        codes = pd.factorize(frame["id"], sort=True)[0]
        mapping = dict(zip(frame["id"].to_numpy(), codes, strict=True))
        for site, groups in seen:
            assert groups is not None
            expected = np.array([mapping[value] for value in site])
            assert np.array_equal(groups, expected)

    def test_dropping_the_codes_at_the_fold_fits_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nuisance = importlib.import_module("cleverly.estimators._nuisance")
        original = nuisance.fit_on_rows

        def without_groups(*args: Any, **kwargs: Any) -> Any:
            args = (*args[:6], None) if len(args) > 6 else args
            kwargs.pop("groups", None)
            if len(args) == 6:
                kwargs["groups"] = None
            return original(*args, **kwargs)

        monkeypatch.setattr(nuisance, "fit_on_rows", without_groups)
        for module in (sequential_module, longitudinal_estimator):
            if hasattr(module, "fit_on_rows"):
                monkeypatch.setattr(module, "fit_on_rows", without_groups)
        seen = self.recorded(site_frame())
        assert any(groups is None for _, groups in seen)


class TestAHeldOutClusterIsUnseen:
    """The leakage witness: a learner that memorizes a site cannot see its own cluster."""

    @staticmethod
    def final_initial(frame: pd.DataFrame, **settings: Any) -> tuple[Any, np.ndarray]:
        result = _site_subject(
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            **settings,
        ).fit(frame, outcome="Y", id="id", **SITE_NODES)
        return result, np.asarray(result.fits["always"].steps[-1].initial)

    @staticmethod
    def pair(result: Any, *, same_fold: bool) -> tuple[int, int]:
        """Two followers of ``always`` with an observed outcome in one cluster."""
        step = result.fits["always"].steps[-1]
        followers = np.flatnonzero(step.trained_on)
        cluster = np.asarray(result.data.cluster)
        fold = np.asarray(result.folds.assignment)
        for i in followers:
            for j in followers:
                if i != j and cluster[i] == cluster[j] and (fold[i] == fold[j]) == same_fold:
                    return int(i), int(j)
        raise AssertionError("no such pair in this draw")

    def test_the_flip_of_a_cluster_mate_does_not_reach_the_prediction(self) -> None:
        frame = site_frame()
        result, before = self.final_initial(frame)
        i, j = self.pair(result, same_fold=True)
        flipped = frame.copy()
        flipped.loc[j, "Y"] = 1.0 - flipped.loc[j, "Y"]
        _, after = self.final_initial(flipped)
        assert after[i] == before[i]
        # Row i's site is unseen in its training fold, so it gets the training mean.
        step = result.fits["always"].steps[-1]
        fold = np.asarray(result.folds.assignment)
        training = step.trained_on & (fold != fold[i])
        assert before[i] == pytest.approx(
            np.clip(frame["Y"].to_numpy()[training].mean(), 0.01, 0.99), abs=1e-15
        )
        # The nonzero witness: the same flip moves every row of another fold, whose
        # training set holds row j, so the learner does read the outcome.
        other = step.trained_on & (fold != fold[j])
        assert np.all(after[other] != before[other])

    def test_a_row_level_split_lets_the_cluster_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: the post-condition and the grouping both removed."""

        def row_folds(self: LTMLE, data: Any) -> Folds:
            return random_partition(data.n, self.n_folds, seed=0)

        monkeypatch.setattr(LTMLE, "_folds", row_folds)
        monkeypatch.setattr(longitudinal_estimator, "check_integrity", lambda *a, **k: None)
        frame = site_frame()
        result, before = self.final_initial(frame)
        i, j = self.pair(result, same_fold=False)
        flipped = frame.copy()
        flipped.loc[j, "Y"] = 1.0 - flipped.loc[j, "Y"]
        _, after = self.final_initial(flipped)
        assert after[i] != before[i]


@pytest.fixture(scope="module")
def unequal() -> Any:
    """60 clusters of 10 to 70 rows: the size term and the clustering both matter here."""
    return fit_law(60, "unequal40", UNEQUAL_SEED)


class TestTheCurveIsClusterSummed:
    def test_every_estimate_reads_the_cluster_sums(self, unequal: Any) -> None:
        assert unequal.inference_status == "influence_curve"
        assert_cluster_summed(unequal.estimates, unequal.data.cluster)

    def test_the_cluster_variance_exceeds_the_row_variance(self, unequal: Any) -> None:
        """Measured 4.62 on the contrast at this seed; the floor sits well below it."""
        curve = unequal[CONTRAST].influence_curve
        row = float(np.var(curve, ddof=1) / curve.size)
        assert unequal[CONTRAST].variance > 3.0 * row

    @pytest.mark.parametrize("rule", ["rows", "means"])
    def test_another_aggregation_fails(self, rule: str, monkeypatch: pytest.MonkeyPatch) -> None:
        original = cluster_module.cluster_sums

        def mutated(curve: Any, cluster: Any) -> Any:
            if rule == "rows":
                return np.asarray(curve, dtype=float)
            sizes = np.bincount(np.asarray(cluster))
            return original(curve, cluster) / sizes[sizes > 0]

        monkeypatch.setattr(cluster_module, "cluster_sums", mutated)
        result = fit_law(60, "unequal40", UNEQUAL_SEED)
        with pytest.raises(AssertionError):
            assert_cluster_summed(result.estimates, result.data.cluster)


def _uncentred_sums(result: Any, regimen: str) -> tuple[np.ndarray, np.ndarray, float]:
    """The cluster sums :math:`A_c` of the uncentred curve, rebuilt from the steps."""
    fit = result.fits[regimen]
    steps = fit.steps
    uncentred = np.asarray(steps[0].targeted, dtype=float).copy()
    for step in steps:
        uncentred += step.clever * (step.pseudo_outcome - step.targeted)
    cluster = np.asarray(result.data.cluster)
    sums = np.bincount(cluster, weights=uncentred)
    sizes = np.bincount(cluster).astype(float)
    return sums, sizes, float(fit.psi_scaled)


class TestTheSizeTermIsCarried:
    """At unequal sizes the curve carries :math:`-N_c\\hat\\psi`, the ratio-of-sums term."""

    NAME = "ey_regimen[always]"

    def test_the_variance_reads_the_centred_sums(self, unequal: Any) -> None:
        sums, sizes, psi = _uncentred_sums(unequal, "always")
        n, j = unequal.n, sums.size
        centred = j * np.var(sums - sizes * psi, ddof=1) / n**2
        uncentred = j * np.var(sums, ddof=1) / n**2
        assert unequal[self.NAME].variance == pytest.approx(centred, rel=1e-10)
        assert abs(uncentred / centred - 1.0) > 0.10
        # The share of the size term in the cluster sums.
        assert np.var(sizes * psi) / np.var(sums - sizes * psi) > 0.05

    def test_a_curve_without_the_size_term_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = sequential_module._finish_regimen_fit

        def uncentred(*args: Any, **kwargs: Any) -> Any:
            fit = original(*args, **kwargs)
            curve = fit.influence_curve_scaled + fit.obs_weights * fit.psi_scaled
            return replace(fit, influence_curve_scaled=curve)

        monkeypatch.setattr(sequential_module, "_finish_regimen_fit", uncentred)
        result = fit_law(60, "unequal40", UNEQUAL_SEED)
        sums, sizes, psi = _uncentred_sums(result, "always")
        centred = sums.size * np.var(sums - sizes * psi, ddof=1) / result.n**2
        assert result[self.NAME].variance != pytest.approx(centred, rel=1e-10)


class TestPooledTargetingSolvesTheClusteredScore:
    def test_the_score_equations_pass(self, unequal: Any) -> None:
        assert unequal.validate()["score_equations"].status is AssessmentStatus.PASSED
        for estimate in unequal.estimates.values():
            if estimate.scale == "level":
                assert abs(float(np.mean(estimate.influence_curve))) < 1e-8


# ---------------------------------------------------------------- every target kind


def _end_of_study(clusters: int, **settings: Any) -> Any:
    return fit_law(clusters, **settings)


def _categorical(clusters: int, **settings: Any) -> Any:
    frame = multivalue_panel(n=1200, seed=43).assign(cluster=cluster_labels(1200, clusters))
    return LTMLE(
        {"never": 0, "always": 1, "high": (2, 1)},
        reference="never",
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=5,
        random_state=0,
        **{"simultaneous": False, **settings},
    ).fit(frame, **COLUMNS, id="cluster")


def _survival(clusters: int, **settings: Any) -> Any:
    frame, _ = law.draw_survival(clusters, "equal40", SEED)
    return LTMLE(
        {"always": 1, "never": 0},
        reference="never",
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=5,
        random_state=0,
        **{"simultaneous": False, **settings},
    ).fit(frame, **SURVIVAL, id="id")


def _competing(clusters: int, **settings: Any) -> Any:
    frame, _ = make_longitudinal_competing(n=40 * clusters, seed=3)
    frame = frame.assign(cluster=cluster_labels(len(frame), clusters))
    return LTMLE(
        {"always": 1, "never": 0},
        reference="never",
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=5,
        random_state=0,
        **{"simultaneous": False, **settings},
    ).fit(frame, **COMPETING, id="cluster")


def _weighted(clusters: int, **settings: Any) -> Any:
    """One zero-mass cluster: code 0. Its rows stay in their fold and add nothing."""
    frame, _ = law.draw_end_of_study(clusters, "equal40", SEED)
    frame = frame.assign(w=np.where(frame["id"] == 0.0, 0.0, 1.0 + 0.5 * (frame["W1"] > 0)))
    return subject(**settings).fit(frame, outcome="Y", id="id", weights="w", **NODES)


KINDS = {
    "end of study, static and dynamic": _end_of_study,
    "end of study, categorical": _categorical,
    "survival": _survival,
    "competing risks": _competing,
    "weighted": _weighted,
}


def derived(result: Any) -> dict[str, Any]:
    """Every derived estimate the result's builders report, keyed by a label."""
    out: dict[str, Any] = {}
    levels = [name for name, e in result.estimates.items() if e.scale == "level"]
    out["contrast"] = result.contrast(lambda psi: psi[0] - psi[1], levels[:2])
    if result.data.is_survival:
        index = result.parameter_index
        horizon = max(h for _, _, h in index.values())
        if not result.data.is_competing:
            name = f"risk_regimen[always @ t={horizon}]"
            reference = f"risk_regimen[never @ t={horizon}]"
            out["ratio"] = result.ratio(name, reference)
        out["rmst"] = result.rmst("always", horizon, versus="never")
        cause = result.config.causes[0] if result.data.is_competing else None
        out["rmtl"] = result.rmtl("always", horizon, **({"cause": cause} if cause else {}))
    elif result.data.family == "binomial" and not result.data.is_weighted:
        out["ratio"] = result.ratio("ey_regimen[always]", "ey_regimen[never]", kind="or")
    return out


def positive_mass_clusters(result: Any) -> int:
    weights = np.asarray(result.data.weights) if result.data.is_weighted else None
    cluster = np.asarray(result.data.cluster)
    if weights is None:
        return int(np.unique(cluster).size)
    return int(np.count_nonzero(np.bincount(cluster, weights=weights) > 0))


class TestEveryTargetKind:
    """Every name a cross-fitted clustered fit reports, and every derived one, is clustered."""

    @pytest.mark.parametrize("clusters", [FEW_CLUSTER_THRESHOLD, 12])
    @pytest.mark.parametrize("kind", list(KINDS))
    def test_every_reported_estimate(self, kind: str, clusters: int) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = KINDS[kind](clusters)
        assert result.folds.origin.scheme == "grouped"
        assert_whole_clusters(result.folds, result.data.cluster)
        assert result.inference_status == "influence_curve"
        everything = {**result.estimates, **derived(result)}
        assert_cluster_summed(everything, result.data.cluster)
        count = positive_mass_clusters(result)
        expected = None if count >= FEW_CLUSTER_THRESHOLD else count - 2
        assert {e.reference_df for e in everything.values()} == {expected}
        if kind == "weighted":
            assert count == clusters - 1
        if result.data.is_competing:
            total = result.incidence_total()
            assert "std_err" in total.columns

    def test_the_band_at_forty_clusters_draws_cluster_multipliers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        original = longitudinal_estimator.simultaneous_bands

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("cluster"))
            return original(*args, **kwargs)

        monkeypatch.setattr(longitudinal_estimator, "simultaneous_bands", spy)
        result = fit_law(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        assert result.simultaneous is not None
        assert seen and all(np.array_equal(c, result.data.cluster) for c in seen)

    def test_the_band_below_forty_clusters_is_skipped_and_named(self) -> None:
        result = fit_law(12, simultaneous=True)
        assert result.simultaneous is None
        assert NO_T_REFERENCE_BANDS in result.summary()
        assert {e.reference_df for e in result.estimates.values()} == {10}

    def test_below_the_floor_the_fit_withholds_its_interval(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_law(MINIMUM_INTERVAL_CLUSTERS - 1)
        assert result.inference_status == "few_cluster_plugin"
        for estimate in result.estimates.values():
            with pytest.raises(CapabilityError):
                _ = estimate.ci

    @pytest.mark.parametrize("mutation", ["J - 1", "label count"])
    def test_another_reference_fails(self, mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
        if mutation == "J - 1":
            original = longitudinal_estimator._reference_df

            def shifted(data: Any, inference: Any) -> Any:
                df = original(data, inference)
                return None if df is None else df + 1

            monkeypatch.setattr(longitudinal_estimator, "_reference_df", shifted)
        else:
            monkeypatch.setattr(
                longitudinal_estimator,
                "_inference_status",
                lambda data, folds: cluster_module.cluster_inference_status(data.cluster),
            )
            original_df = cluster_module.cluster_reference_df
            monkeypatch.setattr(
                longitudinal_estimator,
                "cluster_reference_df",
                lambda cluster, weights=None, rows=None: original_df(cluster, None, rows),
            )
        with pytest.raises(AssertionError):
            self.test_every_reported_estimate("weighted", 12)


class TestTheWorkingModelStaysRefused:
    def test_msm_above_one_fold_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError) as raised:
            LTMLE({"always": 1, "never": 0}, msm=DOSE, n_folds=5)
        message = str(raised.value)
        assert "docs/roadmap.md X27 tracks this work" in message
        assert "cluster" not in message


class TestTheTruncationReplayKeepsTheClusters:
    def test_the_replay_is_clustered_at_every_bound(self, unequal: Any) -> None:
        curve = unequal.diagnostics.truncation_curve(bounds=[0.3])
        fitted = curve[curve["is_fitted_bound"]]
        for row in fitted.itertuples():
            assert row.psi == unequal[row.estimand].psi
        replay = longitudinal_estimator._refit_bound(
            unequal, unequal.replay_recipe, (0.3, subject_study.G_BOUNDS[1])
        )
        assert_cluster_summed(replay.estimates, unequal.data.cluster)
        moved = [name for name, e in replay.estimates.items() if e.psi != unequal[name].psi]
        assert moved, "a bound of 0.3 truncates no weight, so the replay tests nothing"


class TestTheClusteredCrossFitBootstrapIsADiagnostic:
    def test_the_kinds_are_split(self) -> None:
        cross_fitted = fit_law(FEW_CLUSTER_THRESHOLD, regimens={"always": 1, "never": 0})
        in_sample = fit_law(FEW_CLUSTER_THRESHOLD, regimens={"always": 1, "never": 0}, n_folds=1)
        kind = longitudinal_estimator.bootstrap_design_kind
        regimens = [fit.regimen for fit in cross_fitted.fits.values()]
        assert (
            kind(cross_fitted.data, cross_fitted.folds, regimens, None)
            == "end_of_study/cluster_cross_fit"
        )
        assert kind(in_sample.data, in_sample.folds, regimens, None) == "end_of_study/cluster"
        assert "end_of_study/cluster_cross_fit" not in (
            longitudinal_estimator.LICENSED_BOOTSTRAP_DESIGNS
        )

    @staticmethod
    def bootstrapped() -> tuple[Any, list[Any]]:
        seen: list[Any] = []
        original = longitudinal_estimator.random_partition

        def spy(n: int, n_folds: int, *, cluster: Any = None, seed: Any = None) -> Folds:
            seen.append(np.asarray(cluster))
            return original(n, n_folds, cluster=cluster, seed=seed)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(longitudinal_estimator, "random_partition", spy)
            result = fit_law(
                FEW_CLUSTER_THRESHOLD, regimens={"always": 1, "never": 0}, n_bootstrap=4
            )
        return result, seen

    def test_the_percentile_output_is_a_diagnostic(self) -> None:
        result, seen = self.bootstrapped()
        assert "percentile range" in result.summary()
        assert "percentile CI" not in result.summary()
        for estimate in result.estimates.values():
            assert {"bootstrap_range_lower", "bootstrap_range_upper"} <= set(estimate.to_dict())
        # The fit's own draw, then one per replicate: each replicate gives every drawn
        # occurrence of a cluster its own code, so two copies can land in different folds.
        assert len(seen) == 1 + 4
        for codes in seen[1:]:
            assert np.unique(codes).size == FEW_CLUSTER_THRESHOLD

    def test_one_shared_kind_would_license_the_cross_fitted_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the in-sample clustered kind, licensed, read by the cross-fitted fit."""
        original = longitudinal_estimator.bootstrap_design_kind

        def one_kind(*args: Any) -> Any:
            kind = original(*args)
            return None if kind is None else kind.replace("/cluster_cross_fit", "/cluster")

        monkeypatch.setattr(longitudinal_estimator, "bootstrap_design_kind", one_kind)
        monkeypatch.setattr(
            longitudinal_estimator,
            "LICENSED_BOOTSTRAP_DESIGNS",
            longitudinal_estimator.LICENSED_BOOTSTRAP_DESIGNS | {"end_of_study/cluster"},
        )
        with pytest.raises(AssertionError):
            self.test_the_percentile_output_is_a_diagnostic()


class TestTheStudyWorkflowCrossFits:
    def test_the_study_matches_the_engine(self) -> None:
        frame, _ = law.draw_end_of_study(FEW_CLUSTER_THRESHOLD, "equal40", SEED)
        frame = frame.assign(team=frame["id"])
        learners = {
            "outcome_learner": LogisticRegression(max_iter=1000),
            "pseudo_learner": LinearRegression(),
            "treatment_learner": LogisticRegression(max_iter=1000),
        }
        study = (
            CausalStudy(
                frame,
                design=LongitudinalTreatment(
                    outcome="Y",
                    treatment=("A1", "A2"),
                    baseline=("W1", "W2"),
                    time_varying=((), ("L2",)),
                    censoring=("C1", "C2"),
                    cluster="team",
                ),
            )
            .identify(RegimeMean({"always": 1, "never": 0}, reference="never"))
            .estimate(
                method=TMLEMethod(cross_fitting=CrossFitting(n_folds=5)),
                random_state=0,
                simultaneous=False,
                **learners,
            )
        )
        assert study.folds.origin.scheme == "grouped"
        engine = LTMLE(
            {"always": 1, "never": 0},
            reference="never",
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=5,
            random_state=0,
            simultaneous=False,
            **learners,
        ).fit(frame, outcome="Y", id="team", **NODES)
        assert np.array_equal(study.folds.assignment, engine.folds.assignment)
        assert set(study.estimates) <= set(engine.estimates)
        for name in study.estimates:
            estimate = engine[name]
            assert study[name].psi == pytest.approx(estimate.psi, abs=1e-12)
            assert study[name].std_error == pytest.approx(estimate.std_error, rel=1e-10)


class TestOneClusterCannotBeSplit:
    def test_one_label_is_refused_before_any_learner(self) -> None:
        """The container refuses one cluster before a split is drawn, in sample or not."""
        CountingClassifier.calls = 0
        frame, _ = law.draw_end_of_study(1, "equal200", SEED)
        with pytest.raises(
            DataError, match="id identifies a single cluster; cluster-robust variance is undefined"
        ):
            subject(outcome_learner=CountingClassifier(max_iter=1000)).fit(
                frame, outcome="Y", id="id", **NODES
            )
        assert CountingClassifier.calls == 0
