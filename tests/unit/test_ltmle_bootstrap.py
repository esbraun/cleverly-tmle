"""The full-refit bootstrap on ``LTMLE``: resampling, replay and draw order.

The contract in ``docs/technical-reference/longitudinal-tmle.md`` says what a replicate
resamples and refits.  These tests pin each clause: the container subset carries every
field, a replicate refits the mechanism, the bootstrap stream draws nothing the fit uses,
and a truncation-curve replay ignores the bootstrap.  Coverage is the registered study
``full-refit-bootstrap-and-derived-contrasts``.
"""

from __future__ import annotations

import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.longitudinal.estimator as estimator_module
from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.inference.bootstrap import _bootstrap_design
from cleverly.inference.results import attach_bootstrap
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.data import _ROW_FIELDS, _SHARED_FIELDS
from tests.studies import canonical_ltmle

COLUMNS: dict[str, Any] = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _estimator(**settings: Any) -> LTMLE:
    base: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 3,
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }
    base.update(settings)
    return LTMLE({"always": 1, "never": 0}, **base)


@pytest.fixture(scope="module")
def survival_frame() -> pd.DataFrame:
    return make_longitudinal_survival(n=400, seed=0)[0]


@pytest.fixture(scope="module")
def plain(survival_frame: pd.DataFrame) -> Any:
    return _estimator().fit(survival_frame, **COLUMNS)


@pytest.fixture(scope="module")
def bootstrapped(survival_frame: pd.DataFrame) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(n_bootstrap=4).fit(survival_frame, **COLUMNS)


def _equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(left == right)


class TestTheSubset:
    def test_every_field_has_a_subset_rule(self) -> None:
        names = {item.name for item in fields(LongitudinalData)}
        assert set(_ROW_FIELDS) | set(_SHARED_FIELDS) == names
        assert not set(_ROW_FIELDS) & set(_SHARED_FIELDS)

    def test_the_identity_subset_reproduces_every_field(self, plain: Any) -> None:
        data = plain.data
        copy = data.subset(np.arange(data.n))
        for item in fields(LongitudinalData):
            assert _equal(getattr(copy, item.name), getattr(data, item.name)), item.name

    def test_a_competing_and_clustered_container_round_trips(self) -> None:
        from tests import discrete_law_competing as law

        frame = law.frame().assign(cluster=np.arange(len(law.frame())) % 50)
        data = LongitudinalData.from_frame(
            frame,
            outcome=law.outcome_columns(),
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
            id="cluster",
        )
        assert data.cause_event is not None and data.cluster is not None
        copy = data.subset(np.arange(data.n))
        for item in fields(LongitudinalData):
            assert _equal(getattr(copy, item.name), getattr(data, item.name)), item.name

    def test_a_censored_row_keeps_its_pattern(self, plain: Any) -> None:
        data = plain.data
        censored = int(np.flatnonzero(~data.uncensored[:, 0])[0])
        copy = data.subset([censored, censored, *range(10)])
        assert np.isnan(copy.treatment[0, 1]) and np.isnan(copy.treatment[1, 1])
        np.testing.assert_array_equal(copy.uncensored[0], data.uncensored[censored])
        np.testing.assert_array_equal(copy.event[0], data.event[censored])

    def test_weights_are_renormalised(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=2)
        frame = frame.assign(w=np.linspace(0.5, 2.0, len(frame)))
        data = LongitudinalData.from_frame(frame, weights="w", **_end_columns())
        copy = data.subset(np.arange(100))
        assert copy.weights.mean() == pytest.approx(1.0, abs=1e-12)
        assert copy.weight_spec.scale == pytest.approx(
            data.weight_spec.scale * data.weights[:100].mean(), rel=1e-12
        )

    def test_cluster_codes_are_distinct_per_occurrence(self) -> None:
        frame, _ = make_longitudinal(n=300, seed=4, cluster_size=10)
        data = LongitudinalData.from_frame(frame, id="id", **_end_columns())
        design = _bootstrap_design(data, n_replicates=3, resampling="auto", random_state=1)
        sample = design.sample(data, design.draws[0])
        assert design.resampling == "cluster"
        assert np.unique(sample.cluster).size == data.n_clusters
        assert sample.n == data.n


def _end_columns() -> dict[str, Any]:
    return {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }


class TestTheReplicates:
    def test_every_analytic_field_is_unchanged(self, plain: Any, bootstrapped: Any) -> None:
        assert list(plain.estimates) == list(bootstrapped.estimates)
        for name, estimate in plain.estimates.items():
            other = bootstrapped[name]
            assert other.psi == estimate.psi
            assert other.variance == estimate.variance
            np.testing.assert_array_equal(other.influence_curve, estimate.influence_curve)
            assert other.bootstrap is not None and estimate.bootstrap is None
        assert plain.simultaneous is not None
        assert bootstrapped.simultaneous.bands == plain.simultaneous.bands
        assert bootstrapped.simultaneous.critical_value == plain.simultaneous.critical_value

    def test_each_replicate_refits_the_mechanism(
        self, survival_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        original = estimator_module.fit_mechanism

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs["folds"].n_folds)
            return original(*args, **kwargs)

        monkeypatch.setattr(estimator_module, "fit_mechanism", counting)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _estimator(n_bootstrap=3, simultaneous=False).fit(survival_frame, **COLUMNS)
        assert result.bootstrap.n_failed == 0
        # One for the fit and one per replicate.  A bootstrap that reused the fit's
        # mechanism would count one.
        assert len(calls) == 4

    def test_a_replicate_is_a_plain_fit_on_its_resample(
        self, plain: Any, bootstrapped: Any
    ) -> None:
        design = _bootstrap_design(plain.data, n_replicates=4, resampling="auto", random_state=3)
        sample = design.sample(plain.data, design.draws[1])
        refit = _estimator(simultaneous=False).fit(sample)
        for name, draws in bootstrapped.bootstrap.draws.items():
            assert draws[1] == refit[name].psi

    def test_draws_are_deterministic_and_do_not_depend_on_workers(
        self, survival_frame: pd.DataFrame, bootstrapped: Any
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = _estimator(n_bootstrap=4, n_jobs=2).fit(survival_frame, **COLUMNS)
        for name, draws in bootstrapped.bootstrap.draws.items():
            np.testing.assert_array_equal(again.bootstrap.draws[name], draws)

    def test_the_fit_matches_a_committed_canonical_row(self) -> None:
        """At ``n_bootstrap=0`` and above it, the registered fit is unchanged."""
        frame, _ = canonical_ltmle.draw_scenario(canonical_ltmle.SCENARIO, 2_000, 0)
        committed = pd.read_csv(
            Path(__file__).parents[1] / "canonical" / "ltmle" / "replicates.csv.gz",
            float_precision="round_trip",
        )
        rows = committed[
            (committed["implementation"] == "cleverly") & (committed["replicate"] == 0)
        ]
        fitted = canonical_ltmle.fit_cleverly(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = LTMLE(
                canonical_ltmle.REGIMENS,
                reference=canonical_ltmle.REFERENCE,
                outcome_learner=canonical_ltmle.QuasiBinomialGLM(),
                pseudo_learner=canonical_ltmle.QuasiBinomialGLM(),
                treatment_learner=canonical_ltmle.KnownLongitudinalMechanism("treatment"),
                censoring_learner=canonical_ltmle.KnownLongitudinalMechanism("censoring"),
                n_folds=1,
                g_bounds=canonical_ltmle.G_BOUNDS,
                simultaneous=False,
                max_iter=100,
                tol=1e-10,
                random_state=0,
                n_bootstrap=2,
            ).fit(frame, **_end_columns())
        for row in rows.itertuples():
            assert fitted[row.estimand].psi == row.estimate
            assert boot[row.estimand].psi == row.estimate
            assert boot[row.estimand].variance == fitted[row.estimand].variance


class TestReporting:
    def test_the_summary_and_the_frame_carry_the_bootstrap(self, bootstrapped: Any) -> None:
        text = bootstrapped.summary()
        assert "full-refit bootstrap (iid resampling, 4 usable replicates, 0 failed)" in text
        assert "percentile CI" in text
        columns = list(bootstrapped.to_frame().columns)
        assert {"bootstrap_std_err", "bootstrap_ci_lower", "bootstrap_ci_upper"} <= set(columns)

    def test_an_unlicensed_bootstrap_is_a_diagnostic(self, bootstrapped: Any) -> None:
        demoted = attach_bootstrap(
            replace(
                bootstrapped,
                estimates={
                    n: replace(e, bootstrap=None) for n, e in bootstrapped.estimates.items()
                },
            ),
            bootstrapped.bootstrap,
            inferential=False,
        )
        text = demoted.summary()
        assert "bootstrap sd" in text and "percentile range" in text
        assert "percentile CI" not in text
        row = next(iter(demoted.estimates.values())).to_dict()
        assert "bootstrap_sd" in row and "bootstrap_range_lower" in row
        assert "bootstrap_ci_lower" not in row
        # The analytic columns keep their names: the status still supplies inference.
        assert "std_err" in row

    def test_a_point_bootstrap_takes_the_same_rule(self) -> None:
        from cleverly.datasets import make_binary_outcome
        from cleverly.estimators import TMLE

        frame, _ = make_binary_outcome(n=200, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                TMLE(
                    outcome_learner=LogisticRegression(max_iter=1000),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    cross_fit=False,
                    estimands=("ate",),
                    n_bootstrap=3,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        assert "percentile CI" in result.summary()
        bare = replace(
            result,
            estimates={n: replace(e, bootstrap=None) for n, e in result.estimates.items()},
        )
        demoted = attach_bootstrap(bare, result.bootstrap, inferential=False)
        assert "percentile range" in demoted.summary()
        assert "bootstrap_sd" in demoted["ate"].to_dict()

    def test_a_replay_ignores_the_bootstrap(self, plain: Any, bootstrapped: Any) -> None:
        bounds = [plain.config.g_bounds, (0.05, 1.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            left = plain.diagnostics.truncation_curve(bounds)
            right = bootstrapped.diagnostics.truncation_curve(bounds)
        pd.testing.assert_frame_equal(left, right)

    def test_a_study_routes_the_bootstrap_and_narrows_it(self) -> None:
        frame, _ = make_longitudinal(n=300, seed=8)
        study = CausalStudy(
            frame,
            design=LongitudinalTreatment(
                outcome="Y",
                treatment=("A1", "A2"),
                baseline=("W1", "W2"),
                time_varying=((), ("L2",)),
                censoring=("C1", "C2"),
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.identify(RegimeMean({"always": 1, "never": 0})).estimate(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                n_bootstrap=3,
                random_state=0,
            )
        assert result.bootstrap is not None
        assert set(result.bootstrap.draws) == set(result.estimates)
        assert all(e.bootstrap is not None for e in result.estimates.values())


class TestRefusals:
    def test_one_replicate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap must be 0 or at least 2; got 1"):
            _estimator(n_bootstrap=1)

    def test_cluster_resampling_without_ids_is_refused_before_any_learner(
        self, survival_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a learner ran before the refusal")

        monkeypatch.setattr(estimator_module, "fit_mechanism", forbidden)
        with pytest.raises(
            ValueError, match="resampling='cluster' requires the data to carry cluster ids"
        ):
            _estimator(n_bootstrap=2, bootstrap_resampling="cluster").fit(survival_frame, **COLUMNS)
