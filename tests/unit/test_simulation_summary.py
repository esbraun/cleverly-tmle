"""What :class:`~cleverly.validation.EstimandSummary` reports, on which scale.

One property here and it is arithmetic rather than statistical: ``se_ratio`` divides a
reported standard error by an observed spread, and the two have to be on the same scale.
They were not for a ratio estimand -- ``psi`` is the odds ratio while ``std_error`` is
``SE(log OR)`` -- so the quotient came back at roughly ``1 / psi`` and a perfectly
calibrated ``or`` of ``0.42`` read as an interval 2.8 times too wide.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cleverly._inference_status import STATUS_RECORDS
from cleverly.datasets import make_clustered
from cleverly.estimators import TMLE
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    LongitudinalError,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.inference import make_estimate
from cleverly.validation import (
    CoverageStudy,
    EstimandSummary,
    ReplicationCheckpoint,
    ReplicationFailure,
    StudyResult,
    simulation,
)
from tests.conftest import linear_in_sample

FEW = "few_cluster_plugin"


def _summary(**overrides: object) -> EstimandSummary:
    defaults: dict[str, object] = {
        "estimand": "or",
        "truth": 0.5,
        "n": 500,
        "n_replicates": 4,
        "estimates": np.array([0.40, 0.50, 0.60, 0.55]),
        "std_errors": np.array([0.20, 0.22, 0.24, 0.21]),
        "covered": np.ones(4),
        "rejected": np.zeros(4),
    }
    return EstimandSummary(**{**defaults, "estimand": "or", **overrides})  # type: ignore[arg-type]


def test_the_ratio_is_taken_on_the_scale_the_error_is_reported_on() -> None:
    estimates = np.array([0.40, 0.50, 0.60, 0.55])
    logs = np.log(estimates)
    summary = _summary(inference_estimates=logs)
    assert summary.se_ratio == pytest.approx(
        float(np.mean(summary.std_errors)) / float(np.std(logs, ddof=1))
    )


def test_the_two_scales_disagree_enough_to_matter() -> None:
    """The mutation control: without it the assertion above could be a tautology."""
    estimates = np.array([0.40, 0.50, 0.60, 0.55])
    on_log = _summary(inference_estimates=np.log(estimates)).se_ratio
    on_ratio = _summary().se_ratio
    # Roughly `1 / psi` apart, which is the whole defect.
    assert on_ratio / on_log == pytest.approx(1.0 / float(np.mean(estimates)), rel=0.1)


def test_a_difference_estimand_is_arithmetically_what_it_always_was() -> None:
    """``inference_estimates=None`` means the scales coincide, which is every non-ratio.

    This is the regression guard for the ``se_ratio`` bands registered studies assert on
    ``ate``, ``ey_regimen`` and ``risk_regimen``: none of them may move by so much as a
    rounding.
    """
    summary = _summary(estimand="ate")
    assert summary.inference_estimates is None
    np.testing.assert_array_equal(summary.inference_scale_estimates, summary.estimates)
    assert summary.se_ratio == float(np.mean(summary.std_errors)) / summary.monte_carlo_se


def test_a_degenerate_spread_reports_nothing_rather_than_dividing_by_zero() -> None:
    summary = _summary(estimand="ate", estimates=np.full(4, 0.3))
    assert np.isnan(summary.se_ratio)


class _Result:
    def __init__(self, *, alpha: float) -> None:
        estimate = make_estimate(
            "ate",
            0.0,
            np.array([-1.0, 1.0]),
            n=2,
            scale="difference",
            alpha=alpha,
        )
        self.estimates = {"ate": estimate}

    def __getitem__(self, name: str):  # type: ignore[no-untyped-def]
        return self.estimates[name]


class _Estimator:
    def __init__(
        self,
        *,
        alpha: float = 0.05,
        error: str | None = None,
        singular: frozenset[int] = frozenset(),
    ) -> None:
        self.alpha = alpha
        self.error = error
        self.singular = singular

    def fit(self, frame, **kwargs):  # type: ignore[no-untyped-def]
        if self.error is not None:
            raise np.linalg.LinAlgError(self.error)
        if getattr(frame, "seed", None) in self.singular:
            raise np.linalg.LinAlgError("singular on this draw")
        return _Result(alpha=self.alpha)


def test_coverage_study_uses_the_estimates_non_default_alpha() -> None:
    study = CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
        estimator=lambda: _Estimator(alpha=0.10),
        n=2,
        n_replicates=3,
        seed=4,
    ).run()
    assert study.alpha == 0.10
    assert {record.alpha for record in study.replications} == {0.10}
    assert not any(record.rejected for record in study.replications)


def test_failed_replications_retain_the_seed_and_cause() -> None:
    seeds = np.random.SeedSequence(7).generate_state(3)
    study = CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
        estimator=lambda: _Estimator(singular=frozenset({int(seeds[0])})),
        n=2,
        n_replicates=3,
        seed=7,
    ).run()
    assert study.n_failed == 1
    assert study.failures[0].replicate == 0
    assert study.failures[0].seed == int(seeds[0])
    assert study.failures[0].error_type == "LinAlgError"
    assert study.failures[0].message == "singular on this draw"
    assert study.failures[0].call.endswith(":fit")
    assert {record.replicate for record in study.replications} == {1, 2}


def test_an_error_in_the_draw_stops_the_study() -> None:
    def dgp(n, seed):  # type: ignore[no-untyped-def]
        raise np.linalg.LinAlgError("a law defect, not a fit")

    study = CoverageStudy(dgp=dgp, estimator=lambda: _Estimator(), n=2, n_replicates=3, seed=7)
    with pytest.raises(np.linalg.LinAlgError, match="a law defect"):
        study.run()


class _Raising(_Estimator):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.raised = error

    def fit(self, frame, **kwargs):  # type: ignore[no-untyped-def]
        raise self.raised


@pytest.mark.parametrize(
    "error",
    [
        TypeError("a bug"),
        KeyError("a bug"),
        ValueError("raised by this test module, not by a fitting library"),
        RuntimeError("raised by this test module, not by a fitting library"),
        CapabilityError("not implemented", reason=RefusalReason.INVALID_DECLARATION),
        MethodConfigurationError("misdeclared"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_non_recordable_fit_error_stops_the_study(error: BaseException) -> None:
    study = CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
        estimator=lambda: _Raising(error),
        n=2,
        n_replicates=3,
        seed=7,
    )
    with pytest.raises(type(error)):
        study.run()


@pytest.mark.parametrize(
    "error",
    [DataError("no overlap"), LongitudinalError("nobody followed"), ZeroDivisionError("zero")],
    ids=lambda error: type(error).__name__,
)
def test_a_recordable_fit_error_is_a_failure(error: BaseException) -> None:
    study = CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
        estimator=lambda: _Raising(error),
        n=2,
        n_replicates=2,
        seed=7,
    )
    with pytest.raises(RuntimeError, match=f"{type(error).__name__}"):
        study.run()
    outcome = study.run_replication(0, 11)
    assert isinstance(outcome, ReplicationFailure)
    assert outcome.error_type == type(error).__name__


class _Store:
    def __init__(self) -> None:
        self.outcomes: dict[int, object] = {}

    def load(self, replicate):  # type: ignore[no-untyped-def]
        return self.outcomes.get(replicate)

    def save(self, replicate, outcome):  # type: ignore[no-untyped-def]
        self.outcomes[replicate] = outcome


def test_a_checkpointed_study_resumes_without_refitting() -> None:
    fits: list[int] = []

    class Counting(_Estimator):
        def fit(self, frame, **kwargs):  # type: ignore[no-untyped-def]
            fits.append(frame.seed)
            return super().fit(frame, **kwargs)

    store = _Store()

    def study() -> CoverageStudy:
        return CoverageStudy(
            dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
            estimator=lambda: Counting(),
            n=2,
            n_replicates=4,
            seed=9,
            checkpoint=store,
        )

    assert isinstance(store, ReplicationCheckpoint)
    first = study().run()
    assert len(fits) == 4
    del store.outcomes[2]
    second = study().run()
    assert len(fits) == 5
    assert second.replications == first.replications


def test_a_checkpoint_under_another_seed_is_refused() -> None:
    store = _Store()
    CoverageStudy(
        dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
        estimator=lambda: _Estimator(),
        n=2,
        n_replicates=3,
        seed=9,
        checkpoint=store,
    ).run()
    with pytest.raises(RuntimeError, match="under seed"):
        CoverageStudy(
            dgp=lambda n, seed: (SimpleNamespace(seed=seed), {"ate": 0.0}),
            estimator=lambda: _Estimator(),
            n=2,
            n_replicates=3,
            seed=10,
            checkpoint=store,
        ).run()


def test_an_all_failed_study_reports_the_first_cause() -> None:
    study = CoverageStudy(
        dgp=lambda n, seed: (None, {"ate": 0.0}),
        estimator=lambda: _Estimator(error="deliberate failure"),
        n=2,
        n_replicates=2,
        seed=8,
    )
    with pytest.raises(RuntimeError, match="LinAlgError: deliberate failure"):
        study.run()


@pytest.mark.parametrize(
    ("inference", "noun"),
    [
        ("influence_curve", "reported standard error"),
        ("working_mechanism_plugin", "plug-in standard error"),
    ],
)
def test_the_undercoverage_verdict_names_the_spread_by_its_status(
    inference: str, noun: str
) -> None:
    """A study that covers nothing with a spread ten times too small reaches the noun."""
    summary = _summary(
        estimand="ate",
        std_errors=np.full(4, 0.01),
        covered=np.zeros(4),
        inference=inference,
    )
    assert summary.se_ratio < 0.95
    study = StudyResult(
        summaries={"ate": summary},
        replications=(),
        failures=(),
        n=500,
        n_replicates=4,
        alpha=0.05,
        label="spread noun",
    )
    assert f"; the {noun} is " in study.verdict()


def _straddling_dgp(n: int, seed: int) -> tuple[object, dict[str, float]]:
    """A clustered law whose cluster count is drawn from 8 to 12, around the floor of 10."""
    j = int(np.random.default_rng(seed).integers(8, 13))
    frame, truth = make_clustered(n=40 * j, cluster_size=40, seed=int(seed))
    return frame, truth


def _straddling_study() -> StudyResult:
    """Eight in-sample clustered fits, on both sides of the floor of 10 clusters."""
    return CoverageStudy(
        dgp=_straddling_dgp,
        estimator=lambda: TMLE(**linear_in_sample(estimands=("ate",))),
        n=400,
        n_replicates=8,
        seed=0,
        fit_kwargs={"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"], "id": "cluster"},
    ).run()


class TestAStudyWhoseReplicatesMixStatuses:
    """A draw-dependent status summarizes under the precedent status, not a refusal.

    Every record measures the plug-in numbers of its replicate, so the mix averages one
    kind of number. The summary takes the status that withholds, or the first shown status
    when none withholds, and names the mix.
    """

    def test_a_mix_of_a_shown_status_and_the_ordinary_one_reads_each_reported_interval(
        self,
    ) -> None:
        study = _straddling_study()
        records = study.replications
        # The witness: the draws straddle the threshold, so both statuses occur.
        taken = {record.inference for record in records}
        assert taken == {FEW, "influence_curve"}
        summary = study["ate"]
        assert summary.inference == FEW
        few = sum(record.inference == FEW for record in records)
        assert summary.status_counts == ((FEW, few), ("influence_curve", len(records) - few))
        # Every replicate counts, whichever status it took.
        assert summary.n_replicates == len(records) == 8
        assert summary.coverage == np.mean([record.covered for record in records])
        text = study.summary()
        # Both statuses show an interval, so the study measures no diagnostic.
        assert "supply no interval" not in text
        assert (
            f"ate: the replicates took more than one status (few_cluster_plugin in {few}, "
            f"influence_curve in {8 - few}), and every column reads the interval each "
            "replicate reported"
        ) in text
        row = study.to_frame().iloc[0]
        assert row["inference"] == FEW
        assert row["mixed_statuses"] == summary.mixed_statuses
        assert "mean_std_error" in row.index

    def test_a_mix_with_a_withheld_status_summarizes_as_the_diagnostic(self) -> None:
        summary = _summary(
            estimand="ate",
            inference="no_cluster_variance_plugin",
            status_counts=(("no_cluster_variance_plugin", 3), ("influence_curve", 1)),
        )
        study = StudyResult(
            summaries={"ate": summary},
            replications=(),
            failures=(),
            n=500,
            n_replicates=4,
            alpha=0.05,
            label="mixed",
        )
        text = study.summary()
        assert "estimates supply no interval on some replicates" in text
        assert (
            "every column reads them as the "
            f"{STATUS_RECORDS['no_cluster_variance_plugin'].diagnostic_noun}"
        ) in text
        assert "mean_std_error" not in study.to_frame().iloc[0].index

    def test_a_mix_read_as_inference_fails_the_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mutation: a summary that labels the mix with the inferential status."""
        monkeypatch.setattr(simulation, "precedent_status", lambda statuses: "influence_curve")
        with pytest.raises(AssertionError):
            assert _straddling_study()["ate"].inference == FEW

    def test_a_uniform_study_names_no_mix(self) -> None:
        summary = _summary(inference=FEW)
        assert summary.status_counts == ()
        assert summary.mixed_statuses == ""
        assert "mixed_statuses" not in summary.to_dict()

    def test_two_diagnostic_statuses_keep_their_distinct_names(self) -> None:
        summary = _summary(
            estimand="ate",
            inference="adaptive_baseline_plugin",
            status_counts=(("adaptive_baseline_plugin", 2), ("no_cluster_variance_plugin", 2)),
        )
        study = StudyResult(
            summaries={"ate": summary},
            replications=(),
            failures=(),
            n=500,
            n_replicates=4,
            alpha=0.05,
            label="two diagnostics",
        )
        text = study.summary()
        assert "measuring a plug-in diagnostic" in text
        assert "every column reads them as the plug-in diagnostic" in text
        assert "on some replicates" not in text
        assert "adaptive_baseline_plugin in 2, no_cluster_variance_plugin in 2" in text
