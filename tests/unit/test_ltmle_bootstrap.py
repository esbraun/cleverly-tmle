"""The full-refit bootstrap on ``LTMLE``: resampling, replay and draw order.

The contract in ``docs/technical-reference/longitudinal-tmle.md`` says what a replicate
resamples and refits.  These tests pin each clause: the container subset carries every
field, a replicate refits the mechanism, the bootstrap stream draws nothing the fit uses,
and a truncation-curve replay ignores the bootstrap.  Coverage is the registered study
``full-refit-bootstrap-and-derived-contrasts``.
"""

from __future__ import annotations

import warnings
from dataclasses import fields
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
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.data import _ROW_FIELDS, _SHARED_FIELDS
from tests.studies import canonical_ltmle
from tests.studies.evidence.manifest import read_rows

COLUMNS: dict[str, Any] = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _bootstrap_table(summary: Any) -> Any:
    """The bootstrap table of a summary."""
    return next(t for t in summary.tables if (t.title or "").startswith("Bootstrap ("))


def _interval_column(result: Any) -> str:
    """The interval column of the summary's bootstrap table."""
    (table,) = [t for t in result.summary().tables if (t.title or "").startswith("Bootstrap")]
    return table.columns[-1]


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
        committed = read_rows(
            Path(__file__).parents[1] / "canonical" / "ltmle" / "replicates.parquet"
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
            # Against the committed artifact at a stated tolerance, because a refit on another
            # platform can move the last bits.  Within one process the check is exact.
            assert fitted[row.estimand].psi == pytest.approx(row.estimate, rel=1e-10, abs=0)
            assert boot[row.estimand].psi == fitted[row.estimand].psi
            assert boot[row.estimand].variance == fitted[row.estimand].variance


def without_bootstrap_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: every registered row stops listing ``bootstrap`` among its kinds."""
    from dataclasses import replace

    from cleverly import _support

    rows = tuple(
        replace(row, kinds=row.kinds - {"bootstrap"}) for row in _support.registered_rows()
    )
    monkeypatch.setattr(_support, "registered_rows", lambda: rows)


def canonical_boot(n_bootstrap: int = 2) -> Any:
    """The registered ``canonical-ltmle`` fit (row G01) with a bootstrap, at 400 units."""
    frame, _ = canonical_ltmle.draw_scenario(canonical_ltmle.SCENARIO, 400, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
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
            n_bootstrap=n_bootstrap,
        ).fit(frame, **_end_columns())


class TestReporting:
    def test_the_designs_the_old_constant_licensed_map_to_rows_with_bootstrap(self) -> None:
        """The in-sample point, end-of-study and survival kinds each have a row with ``bootstrap``."""
        from cleverly._support import registered_rows

        by_id = {row.id: row for row in registered_rows()}
        listed = {row_id for row_id, row in by_id.items() if "bootstrap" in row.kinds}
        assert {"T01.binary", "G01", "G11.nodes4"} <= listed
        assert by_id["G01"].key.splitting == ("in_sample",)
        assert "survival" in by_id["G11.nodes4"].key.family
        assert by_id["G11.nodes4"].key.splitting == ("in_sample",)

    def test_a_registered_fit_has_a_supported_bootstrap(self) -> None:
        result = canonical_boot()
        for estimate in result.estimates.values():
            support = estimate.bootstrap.support
            assert support.label == "supported"
            assert support.row == "G01"
            assert estimate.bootstrap.diagnostic is False

    def test_a_row_without_the_bootstrap_kind_makes_the_interval_experimental(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: with no row listing ``bootstrap`` no interval is supported."""
        without_bootstrap_kind(monkeypatch)
        result = canonical_boot()
        estimate = next(iter(result.estimates.values()))
        assert estimate.bootstrap.support.label == "experimental"
        assert "bootstrap interval" in estimate.bootstrap.support.reason
        summary = result.summary()
        table = _bootstrap_table(summary)
        assert table.title == "Bootstrap (iid resampling, 2 usable replicates)"
        assert result.bootstrap.n_failed == 0
        assert "percentile 95% CI" in table.columns
        assert "bootstrap_experimental" in summary.note_codes
        row = estimate.to_dict()
        assert {"bootstrap_std_err", "bootstrap_ci_lower", "bootstrap_ci_upper"} <= set(row)
        assert row["bootstrap_support"] == "experimental"

    def test_a_supported_bootstrap_prints_no_marker(self) -> None:
        result = canonical_boot()
        summary = result.summary()
        assert "percentile 95% CI" in _bootstrap_table(summary).columns
        assert "bootstrap_experimental" not in summary.note_codes
        columns = set(result.to_frame().columns)
        assert {"bootstrap_std_err", "bootstrap_ci_lower", "bootstrap_ci_upper"} <= columns
        assert set(next(iter(result.estimates.values())).to_dict()["bootstrap_support"]) == set(
            "supported"
        )

    def test_a_cross_fitted_bootstrap_is_experimental(self, survival_frame: pd.DataFrame) -> None:
        """No row lists ``bootstrap`` for a cross-fitted fit, so the interval is experimental."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crossfit = _estimator(n_bootstrap=3, n_folds=2).fit(survival_frame, **COLUMNS)
        estimate = next(iter(crossfit.estimates.values()))
        assert estimate.bootstrap.support.label == "experimental"
        assert "bootstrap_experimental" in crossfit.summary().note_codes

    def test_an_unmeasured_composition_is_experimental(self) -> None:
        """A competing-risk fit has no row listing ``bootstrap``, so its interval is experimental."""
        from tests import discrete_law_competing as law

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                {"always": 1, "never": 0},
                n_folds=1,
                random_state=0,
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
                n_bootstrap=2,
                simultaneous=False,
            ).fit(
                law.frame(),
                outcome=law.outcome_columns(),
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                censoring=["C1", "C2"],
            )
        estimate = next(iter(result.estimates.values()))
        assert estimate.bootstrap.support.label == "experimental"
        summary = result.summary()
        assert "percentile 95% CI" in _bootstrap_table(summary).columns
        assert "bootstrap_experimental" in summary.note_codes

    def test_the_point_row_decides_the_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An in-sample ``TMLE`` bootstrap is supported by its row, and experimental without it."""
        from cleverly.datasets import make_binary_outcome
        from cleverly.estimators import TMLE

        frame, _ = make_binary_outcome(n=200, seed=1)

        def fit(cross_fit: bool = False) -> Any:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return (
                    TMLE(
                        outcome_learner=LogisticRegression(max_iter=1000),
                        treatment_learner=LogisticRegression(max_iter=1000),
                        cross_fit=cross_fit,
                        n_folds=2,
                        estimands=("ate",),
                        n_bootstrap=3,
                        random_state=0,
                        simultaneous=False,
                    )
                    .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                    .single()
                )

        supported = fit()
        assert supported["ate"].bootstrap.support.label == "supported"
        assert _interval_column(supported) == "percentile 95% CI"
        assert "bootstrap_experimental" not in supported.summary().note_codes
        cross_fitted = fit(True)
        assert cross_fitted["ate"].bootstrap.support.label == "experimental"
        assert cross_fitted["ate"].bootstrap.diagnostic is False
        assert _interval_column(cross_fitted) == "percentile 95% CI"
        assert "bootstrap_experimental" in cross_fitted.summary().note_codes
        assert "bootstrap_ci_lower" in cross_fitted["ate"].to_dict()
        without_bootstrap_kind(monkeypatch)
        demoted = fit()
        assert demoted["ate"].bootstrap.support.label == "experimental"
        assert "bootstrap_experimental" in demoted.summary().note_codes

    @pytest.mark.parametrize(
        "configuration",
        [
            pytest.param({"estimator": "DRTMLE"}, id="DRTMLE"),
            pytest.param({"estimator": "CTMLE", "strategy": "greedy"}, id="CTMLE greedy"),
            pytest.param(
                {"estimator": "CTMLE", "strategy": "discrete", "candidates": [["W1", "W2"]]},
                id="CTMLE discrete with the full adjustment set",
            ),
            pytest.param({"estimator": "CTMLE", "strategy": "oat"}, id="CTMLE oat"),
        ],
    )
    def test_a_ctmle_bootstrap_is_a_diagnostic_and_a_drtmle_one_is_experimental(
        self, configuration: dict[str, Any]
    ) -> None:
        """Only the collaborative fit declares its bootstrap a diagnostic.

        The registered ``boot_ate_point_tmle`` cell fits a point-treatment ``TMLE``. An
        in-sample ``DRTMLE`` fit publishes the inferential names under the experimental label,
        because no row lists ``bootstrap`` for it. Every ``CTMLE`` strategy publishes the
        diagnostic names, because ``CTMLE._bootstrap_diagnostic`` is ``True``.
        """
        from cleverly.datasets import make_binary_outcome
        from cleverly.estimators import CTMLE, DRTMLE

        frame, _ = make_binary_outcome(n=200, seed=1)
        settings = dict(configuration)
        estimator = {"DRTMLE": DRTMLE, "CTMLE": CTMLE}[settings.pop("estimator")]
        if estimator is DRTMLE:
            settings.update(
                reduced_outcome_learner=LinearRegression(),
                reduced_mechanism_residual_learner=LinearRegression(),
                reduced_treatment_learner=LogisticRegression(max_iter=1000),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                estimator(
                    outcome_learner=LogisticRegression(max_iter=1000),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    cross_fit=False,
                    estimands=("ate",),
                    simultaneous=False,
                    n_bootstrap=3,
                    random_state=0,
                    **settings,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        row = result["ate"].to_dict()
        if estimator is CTMLE:
            assert result["ate"].bootstrap.diagnostic is True
            assert _interval_column(result) == "percentile range (diagnostic)"
            assert "bootstrap_sd" in row
            assert "bootstrap_ci_lower" not in row
        else:
            assert result["ate"].bootstrap.diagnostic is False
            assert result["ate"].bootstrap.support.label == "experimental"
            assert _interval_column(result) == "percentile 95% CI"
            assert "bootstrap_ci_lower" in row


class TestDerivedBootstrap:
    """``rmst``, ``ratio`` and ``contrast`` carry the bootstrap the replicates imply."""

    def test_rmst_reads_the_replicate_risks(self, bootstrapped: Any) -> None:
        derived = bootstrapped.rmst("always", 3)
        draws = bootstrapped.bootstrap.draws
        expected = 3.0 - draws["risk_regimen[always @ t=1]"] - draws["risk_regimen[always @ t=2]"]
        np.testing.assert_allclose(derived.bootstrap.draws, expected, atol=1e-12, rtol=0)
        low, high = np.quantile(expected, [0.025, 0.975])
        assert derived.bootstrap.ci == pytest.approx((low, high), abs=1e-12)
        assert derived.bootstrap.std_error == pytest.approx(np.std(expected, ddof=1), abs=1e-12)

    def test_ratio_and_contrast_read_the_replicates(self, bootstrapped: Any) -> None:
        a, b = "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]"
        draws = bootstrapped.bootstrap.draws
        ratio = bootstrapped.ratio(a, b, view="survival")
        np.testing.assert_allclose(
            ratio.bootstrap.draws, (1 - draws[a]) / (1 - draws[b]), atol=1e-12, rtol=0
        )
        difference = bootstrapped.contrast(lambda p: p[0] - p[1], [a, b])
        np.testing.assert_allclose(
            difference.bootstrap.draws, draws[a] - draws[b], atol=1e-12, rtol=0
        )

    def test_the_support_is_inherited(self, bootstrapped: Any) -> None:
        derived = bootstrapped.rmst("always", 3)
        parent = bootstrapped["risk_regimen[always @ t=1]"].bootstrap
        assert derived.bootstrap.diagnostic is False
        # The parent's reason is kept, and the derived kind adds its own when the row lacks it.
        assert derived.bootstrap.support.label == parent.support.label == "experimental"
        assert parent.support.reason in derived.bootstrap.support.reason
        assert "licenses no derived estimate" in derived.bootstrap.support.reason
        assert "bootstrap_ci_lower" in derived.to_dict()

    def test_a_contrast_of_supported_draws_is_supported_and_a_mix_is_experimental(self) -> None:
        from dataclasses import replace

        from cleverly._support import Support

        result = canonical_boot()
        names = list(result.estimates)[:2]
        derived = result.contrast(lambda p: p[0] - p[1], names)
        assert derived.bootstrap.support.label == "supported"
        # The control: one experimental parent makes the derived bootstrap experimental.
        experimental = Support("experimental", "No row.", ("X1",))
        mixed = dict(result.estimates)
        mixed[names[1]] = replace(
            mixed[names[1]], bootstrap=replace(mixed[names[1]].bootstrap, support=experimental)
        )
        probe = replace(result, estimates=mixed)
        assert probe.contrast(lambda p: p[0] - p[1], names).bootstrap.support.label == (
            "experimental"
        )

    def test_a_fit_without_a_bootstrap_attaches_none(self, plain: Any) -> None:
        assert plain.rmst("always", 3).bootstrap is None

    def test_a_replay_ignores_the_bootstrap(self, plain: Any, bootstrapped: Any) -> None:
        bounds = [plain.config.g_bounds, (0.05, 1.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            left = plain.diagnostics.truncation_curve(bounds)
            right = bootstrapped.diagnostics.truncation_curve(bounds)
        pd.testing.assert_frame_equal(left.to_frame(), right.to_frame())

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
