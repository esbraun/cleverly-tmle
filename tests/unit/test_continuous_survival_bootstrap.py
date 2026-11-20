"""The full-refit bootstrap of the continuous-time TMLE (X6 part (d), plan 4.4: T-d0 to T-d4).

Every check here reads the bootstrap's own records: the fold plans the replicates were refitted
with, the replicate draws, the failure count and the labels.  Each check that a component could
pass by accident carries a nonzero witness: copies of a unit exist in the training rows, some
replicates fail, the BCa limits differ from the percentile ones.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import ConvergenceWarning, DataError, MethodConfigurationError
from cleverly.inference.bootstrap import (
    _bootstrap_design,
    _bootstrap_draw,
    bca_acceleration,
    run_index_bootstrap,
)
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from cleverly.survival import _memory as memory
from cleverly.survival import engine as engine_module
from cleverly.survival import result as result_module
from cleverly.survival.data import ContinuousSurvivalData

HORIZONS = (0.5, 1.0, 2.0)
BASELINE = ["x1", "x2"]


def _frame(n: int = 200, seed: int = 11, *, early_cause2: int | None = None) -> pd.DataFrame:
    """Two causes with exponential hazards, a binary treatment and two continuous covariates.

    ``x1`` is continuous, so its value names each unit, and the copies of a unit in a resample
    are the rows that share it.  ``early_cause2`` keeps that many control units' cause-2 events
    before the first target time and moves every other cause-2 event of the control arm past
    it, so a resample that draws none of those units has no early event of that cause.
    """
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n)
    h1 = np.exp(-0.8 + 0.4 * x1 - 0.4 * a)
    h2 = np.exp(-1.4 - 0.3 * x2 + 0.2 * a)
    t1, t2 = rng.exponential(1 / h1), rng.exponential(1 / h2)
    censor = rng.exponential(4.0, size=n)
    time = np.minimum(np.minimum(t1, t2), censor)
    event = np.where(censor <= np.minimum(t1, t2), 0, np.where(t1 <= t2, 1, 2))
    if early_cause2 is not None:
        early = np.flatnonzero((a == 0) & (event == 2) & (time <= HORIZONS[0]))
        late = early[early_cause2:]
        time[late] = HORIZONS[0] + 0.05 + rng.uniform(0.0, 1.0, size=late.size)
        assert np.count_nonzero((a == 0) & (event == 2) & (time <= HORIZONS[0])) == early_cause2
    return pd.DataFrame({"time": time, "event": event, "a": a, "x1": x1, "x2": x2})


def _engine(**settings: Any) -> ContinuousSurvivalTMLE:
    base: dict[str, Any] = {
        "cross_fit": False,
        "hazard_library": [CoxHazard("main")],
        "censoring_library": [CoxHazard("main")],
        "random_state": 3,
        "simultaneous": False,
    }
    if settings.get("cross_fit"):
        from sklearn.linear_model import LogisticRegression

        base["treatment_learner"] = LogisticRegression(max_iter=1000)
        base["n_folds"] = 5
    base.update(settings)
    return ContinuousSurvivalTMLE(HORIZONS, **base)


def _fit(frame: pd.DataFrame, estimator: ContinuousSurvivalTMLE) -> Any:
    return estimator.fit(frame, time="time", event="event", treatment="a", baseline=BASELINE)


def _container(frame: pd.DataFrame) -> ContinuousSurvivalData:
    return ContinuousSurvivalData.from_frame(
        frame, time="time", event="event", treatment="a", baseline=BASELINE
    )


@pytest.fixture(scope="module")
def in_sample() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(_frame(), _engine(n_bootstrap=24, simultaneous=True))


# ----------------------------------------------------------------------------- T-d0 labels


def test_an_in_sample_bootstrap_reads_experimental_with_the_unread_theory(in_sample: Any) -> None:
    assert in_sample.bootstrap_status == "ran"
    for estimate in in_sample.values():
        support = estimate.bootstrap.support
        assert support.label == "experimental"
        assert support.reason == result_module.IN_SAMPLE_BOOTSTRAP_REASON
        assert support.reason.startswith(
            "The bootstrap delta-method conditions are not yet read first-hand."
        )
        assert support.evidence == ("X6",)
    band = in_sample.bootstrap_band
    assert band is not None and band.support.label == "experimental"
    assert band.support.reason.endswith(result_module.BOOTSTRAP_BAND_REASON)
    assert set(band.bands) == set(
        result_module.risk_names(in_sample.arms, in_sample.causes, in_sample.horizons)
    )


def test_a_cross_fitted_bootstrap_reads_experimental_with_the_missing_theory() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _fit(_frame(), _engine(cross_fit=True, n_bootstrap=2))
    assert fit.bootstrap_status == "ran"
    for estimate in fit.values():
        support = estimate.bootstrap.support
        assert support.label == "experimental"
        assert support.reason == result_module.CROSS_FITTED_BOOTSTRAP_REASON
        assert "Cai and van der Laan (2020) warn against it" in support.reason


def test_no_bootstrap_kind_reads_supported() -> None:
    for kind in ("bootstrap", "bootstrap_bca", "bootstrap_band"):
        for cross_fitted in (False, True):
            support = result_module.continuous_bootstrap_support(
                kind=kind,  # type: ignore[arg-type]
                cross_fitted=cross_fitted,
            )
            assert support.label == "experimental"
    with pytest.raises(ValueError, match="unknown bootstrap kind"):
        result_module.continuous_bootstrap_support(kind="wald", cross_fitted=False)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- T-d1 the folds


def _captured_plans(monkeypatch: pytest.MonkeyPatch, replicates: int) -> list[tuple[Any, Any]]:
    """Every replicate's resample and fold plan, read as the replicate is refitted.

    The replicate refit stops right after the plan is read, so each counts as a failure and the
    fit withholds its bootstrap; only the plans are read here.
    """
    seen: list[tuple[Any, Any]] = []
    original = ContinuousSurvivalTMLE._fit_data

    def spy(self: Any, data: Any, *, folds: Any = None, plan: Any = None) -> Any:
        if self.n_bootstrap == 0 and plan is not None:
            seen.append((data, plan))
            # A refusal on the drawn data is a counted failure, so the plans are all read.
            raise DataError("plan read")
        return original(self, data, folds=folds, plan=plan)

    monkeypatch.setattr(ContinuousSurvivalTMLE, "_fit_data", spy)
    with pytest.warns(ConvergenceWarning, match=f"{replicates} of {replicates} bootstrap refits"):
        fit = _fit(_frame(), _engine(cross_fit=True, n_bootstrap=replicates))
    assert fit.bootstrap_status == "bootstrap_unstable"
    assert len(seen) == replicates
    return seen


def test_copies_of_a_unit_share_a_cross_fitting_fold_and_a_learner_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_training = 0
    for data, plan in _captured_plans(monkeypatch, 20):
        unit = np.asarray(data.baseline)[:, 0]
        assert np.unique(plan.outer).size == 5
        for value in np.unique(unit):
            rows = unit == value
            assert np.unique(plan.outer[rows]).size == 1
        for position, fold in enumerate(np.unique(plan.outer)):
            training = plan.outer != fold
            inner = plan.inner[position]
            assert inner.shape == (int(training.sum()),)
            assert np.unique(inner).size == 5
            units = unit[training]
            for value in np.unique(units):
                copies = units == value
                assert np.unique(inner[copies]).size == 1, (position, value)
                shared_training += int(copies.sum() > 1)
    # Nonzero witness: the training rows hold copies, whose inner labels the check read.
    assert shared_training > 1_000


def test_an_in_sample_replicate_draws_its_learner_folds_on_units() -> None:
    index = np.array([0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 9])
    event = np.array([1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 1, 0, 0, 0])
    plan = engine_module.replicate_plan(
        index, event, cross_fit=False, n_folds=10, learner_folds=3, seed=5
    )
    assert np.all(plan.outer == 0) and len(plan.inner) == 1
    (inner,) = plan.inner
    for value in np.unique(index):
        assert np.unique(inner[index == value]).size == 1
    assert np.unique(inner).size == 3


# ------------------------------------------------------------- T-d2 a replicate is a refit


def test_a_replicate_is_the_fit_of_its_resampled_frame() -> None:
    frame = _frame()
    estimator = _engine(n_bootstrap=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _fit(frame, estimator)
    assert fit.bootstrap.n_failed == 0
    seed = int(fit.estimator.random_state)
    n = len(frame)
    design = _bootstrap_design(
        engine_module._Units(n), n_replicates=3, resampling="iid", random_state=seed
    )
    # The rows are the ones run_index_bootstrap draws for the fit's seed.
    drawn: list[np.ndarray] = []

    def record(index: np.ndarray) -> dict[str, tuple[float, float]]:
        drawn.append(index.copy())
        return {"x": (0.0, 1.0)}

    run_index_bootstrap(engine_module._Units(n), record, n_replicates=3, random_state=seed)
    for b, draw in enumerate(design.draws):
        index = _bootstrap_draw(n, None, np.random.default_rng(draw.sequence))[0]
        np.testing.assert_array_equal(index, drawn[b])
        resampled = frame.iloc[index].reset_index(drop=True)
        plan = engine_module.replicate_plan(
            index,
            resampled["event"].to_numpy(),
            cross_fit=False,
            n_folds=estimator.n_folds,
            learner_folds=estimator.learner_folds,
            seed=draw.seed,
        )
        direct = _engine(causes={1: "1", 2: "2"}).refit(
            _container(resampled), random_state=draw.seed, _frozen_plan=plan
        )
        for name, estimate in direct.items():
            assert fit.bootstrap.draws[name][b] == estimate.psi, (b, name)
            assert fit.bootstrap.std_errors[name][b] == estimate.std_error, (b, name)
    # Witness: the replicates differ from the fit and from each other.
    name = next(iter(fit))
    values = fit.bootstrap.draws[name]
    assert np.unique(values).size == 3 and fit[name].psi not in values


def test_an_assessment_refit_runs_no_bootstrap(in_sample: Any) -> None:
    stored = in_sample.estimator
    assert stored.n_bootstrap == 24
    plain = stored._for_estimate_refits()
    assert plain is not stored
    assert plain.n_bootstrap == 0 and plain.random_state == stored.random_state
    # Without a bootstrap the estimator itself is refitted, so callers that key their refits
    # on the estimator (the refuter-calibration study's shared refits) see one estimator.
    assert plain._for_estimate_refits() is plain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = plain.refit(in_sample.data, random_state=stored.random_state)
    assert refit.bootstrap is None and refit.bootstrap_status is None
    for name, estimate in refit.items():
        assert estimate.psi == in_sample[name].psi


# ------------------------------------------------------------------ failed refits


def test_too_many_failed_refits_withhold_every_bootstrap_interval() -> None:
    frame = _frame(early_cause2=2)
    estimator = _engine(n_bootstrap=20)
    critical = np.flatnonzero(
        (frame["a"] == 0) & (frame["event"] == 2) & (frame["time"] <= HORIZONS[0])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = _fit(frame, _engine())
    seed = int(plain.estimator.random_state)
    design = _bootstrap_design(
        engine_module._Units(len(frame)), n_replicates=20, resampling="iid", random_state=seed
    )
    expected = sum(
        not np.isin(
            critical, _bootstrap_draw(len(frame), None, np.random.default_rng(d.sequence))[0]
        ).any()
        for d in design.draws
    )
    # Nonzero witness: some resamples draw neither unit with an early cause-2 event.
    assert 1 <= expected <= 6
    with pytest.warns(ConvergenceWarning) as caught:
        fit = _fit(frame, estimator)
    texts = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    (text,) = [t for t in texts if "bootstrap refits failed" in t]
    assert text.startswith(f"{expected} of 20 bootstrap refits failed ({expected} x DataError: ")
    assert text.endswith(
        "the percentile interval would condition on the replicates that fit, so it is withheld. "
        "The Wald interval stands."
    )
    assert fit.bootstrap_status == "bootstrap_unstable"
    assert fit.bootstrap.n_failed == expected
    assert len(fit.bootstrap_failures) == expected
    assert all(reason.startswith("DataError: Arm '0'") for reason in fit.bootstrap_failures)
    assert fit.bootstrap_band is None
    for name, estimate in fit.items():
        assert estimate.bootstrap is None
        assert estimate.ci == plain[name].ci


def test_the_share_of_failed_refits_decides_the_withholding(in_sample: Any) -> None:
    # One failure in 100 is within the 1% share and keeps the intervals; two are not.
    row = {name: (estimate.psi, estimate.std_error) for name, estimate in in_sample.items()}
    plain = result_module.replace(in_sample, bootstrap=None, bootstrap_status=None)
    for failed, status in ((1, "ran"), (2, "bootstrap_unstable")):
        outcomes = [(None, "DataError: lost")] * failed + [(row, None)] * (100 - failed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = result_module.attach_bootstrap(
                plain,
                outcomes,
                interval="percentile",
                simultaneous=False,
                alpha=0.05,
                cross_fitted=False,
            )
        assert out.bootstrap_status == status
        assert out.bootstrap.n_failed == failed
        withheld = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert len(withheld) == (status == "bootstrap_unstable")
    # Every refit failing withholds the intervals and keeps no draws, but keeps the counts.
    with pytest.warns(ConvergenceWarning, match=r"3 of 3 bootstrap refits failed \(3 x E\)"):
        out = result_module.attach_bootstrap(
            plain,
            [(None, "E")] * 3,
            interval="bca",
            simultaneous=False,
            alpha=0.05,
            cross_fitted=False,
        )
    assert out.bootstrap_status == "bootstrap_unstable"
    assert (out.bootstrap.n_requested, out.bootstrap.n_failed) == (3, 3)
    assert all(draws.size == 0 for draws in out.bootstrap.draws.values())
    record = out.to_dict()["bootstrap"]
    assert (record["n_requested"], record["n_failed"], record["n_kept"]) == (3, 3, 0)


# ---------------------------------------------------------------------------- T-d4 BCa


def test_the_bca_interval_is_the_shared_bca_of_the_stored_draws() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _fit(_frame(), _engine(n_bootstrap=24, bootstrap_interval="bca"))
    moved = 0
    for name, estimate in fit.items():
        expected = fit.bootstrap.interval(
            name,
            estimate.alpha,
            interval="bca",
            estimate=estimate.psi,
            acceleration=bca_acceleration(estimate.influence_curve),
        )
        assert estimate.bootstrap.interval == "bca"
        assert estimate.bootstrap.ci == expected.ci
        assert estimate.bootstrap.acceleration == expected.acceleration
        assert estimate.bootstrap.support.reason == result_module.IN_SAMPLE_BOOTSTRAP_REASON
        percentile = fit.bootstrap.interval(name, estimate.alpha)
        moved += int(percentile.ci != expected.ci)
    # Witness: the bias correction and acceleration move the limits of some estimate.
    assert moved > 0


def test_the_percentile_interval_is_the_shared_percentile_of_the_stored_draws(
    in_sample: Any,
) -> None:
    for name, estimate in in_sample.items():
        expected = in_sample.bootstrap.interval(name, estimate.alpha)
        assert estimate.bootstrap.ci == expected.ci
        assert estimate.bootstrap.std_error == expected.std_error
    frame = in_sample.to_frame()
    assert {"bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_support"} <= set(frame.columns)
    record = in_sample.to_dict()["bootstrap"]
    assert record["status"] == "ran" and record["n_requested"] == 24 and record["failures"] == []


# ------------------------------------------------------------------------ settings, memory


def test_the_settings_are_checked() -> None:
    for value in (1, -2, 2.5, True):
        with pytest.raises(ValueError, match="n_bootstrap is 0 or an integer of at least 2"):
            ContinuousSurvivalTMLE(HORIZONS, n_bootstrap=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bootstrap_interval is 'percentile' or 'bca'"):
        ContinuousSurvivalTMLE(HORIZONS, n_bootstrap=10, bootstrap_interval="basic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_bootstrap=0 runs none"):
        ContinuousSurvivalTMLE(HORIZONS, bootstrap_interval="bca")


def test_the_study_route_passes_n_bootstrap() -> None:
    from cleverly import Inference, TMLEMethod
    from cleverly.survival.data import continuous_kwargs

    assert continuous_kwargs(TMLEMethod(inference=Inference(n_bootstrap=50)))["n_bootstrap"] == 50
    assert continuous_kwargs(TMLEMethod())["n_bootstrap"] == 0


def _shape(**changes: Any) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "n": 1_000,
        "causes": 2,
        "times": 600,
        "rows": 3,
        "covariates": 4,
        "train_size": 900,
        "design_width": 9,
        "simultaneous": True,
        "n_multiplier": 10_000,
        "switched": True,
        "n_jobs": 4,
        "n_bootstrap": 500,
        "bootstrap_workers": 1,
        "estimates": memory.estimate_count(2, 3, 0),
    }
    shape.update(changes)
    return shape


def test_the_bootstrap_holds_the_result_beside_one_replicate_per_worker() -> None:
    assert memory.estimate_count(2, 3, 0) == 24 and memory.estimate_count(2, 3, 1) == 33
    one = memory.peak_bytes(**_shape())
    two = memory.peak_bytes(**_shape(bootstrap_workers=2))
    replicate = memory.peak_bytes(
        **_shape(train_size=1_000, simultaneous=False, n_jobs=1, n_bootstrap=0, bootstrap_workers=1)
    ).bytes
    held = memory.bootstrap_held(n=1_000, estimates=24, rows=3, covariates=4, n_bootstrap=500)
    fixed = one.phases["nuisance"] - memory.peak_bytes(**_shape(n_bootstrap=0)).phases["nuisance"]
    assert fixed == 0
    assert two.phases["bootstrap"] - one.phases["bootstrap"] == replicate
    assert one.phases["bootstrap"] > held + replicate
    # The walk's own threads do not multiply the replicates: each replicate walks on one.
    assert memory.peak_bytes(**_shape(n_jobs=16)).phases["bootstrap"] == one.phases["bootstrap"]
    assert "bootstrap" not in memory.peak_bytes(**_shape(n_bootstrap=0)).phases


def test_a_budget_below_the_bootstrap_phase_refuses_before_any_learner() -> None:
    frame = _frame()
    estimator = _engine(n_bootstrap=50)
    data = _container(frame)
    shape = estimator._memory_shape(len(frame), 2, 150, np.eye(3), 2, np.zeros(len(frame)))
    estimate = memory.peak_bytes(**shape)
    assert estimate.phase == "bootstrap"
    budget = max(value for key, value in estimate.phases.items() if key != "bootstrap") + 1
    with pytest.raises(MethodConfigurationError, match="in the full-refit bootstrap") as caught:
        _engine(n_bootstrap=50, max_memory_bytes=budget).fit_data(data)
    assert "With a smaller n_jobs the bootstrap holds fewer replicate fits" in str(caught.value)


def test_each_replicate_takes_its_share_of_the_budget() -> None:
    estimator = _engine(n_bootstrap=50, max_memory_bytes=10**9, n_jobs=2)
    shape = estimator._memory_shape(200, 2, 150, np.eye(3), 2, np.zeros(200))
    held = memory.bootstrap_held(**{key: shape[key] for key in memory.BOOTSTRAP_HELD})
    assert estimator._replicate_budget(shape) == (10**9 - held) // 2
    replica = estimator._replica({1: "1", 2: "2"}, 123)
    assert (replica.n_bootstrap, replica.n_jobs, replica.simultaneous) == (0, 1, False)
    assert replica.max_memory_bytes == 123 and replica.causes == {1: "1", 2: "2"}


def test_under_the_default_budget_a_replicate_takes_its_share_of_what_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The memory available after the fit already leaves out the resident result, so the
    # default budget is not reduced by the result a second time.
    monkeypatch.setattr(engine_module, "available_memory", lambda: 10**9)
    estimator = _engine(n_bootstrap=50, n_jobs=2)
    shape = estimator._memory_shape(200, 2, 150, np.eye(3), 2, np.zeros(200))
    assert estimator._replicate_budget(shape) == 10**9 // 2
    monkeypatch.setattr(engine_module, "available_memory", lambda: None)
    assert estimator._replicate_budget(shape) is None


# ----------------------------------------------------------------- the replicates' fold draws


def _spread(labels: np.ndarray, codes: np.ndarray) -> int:
    """The largest difference, over event codes, between two folds' unit counts."""
    folds = np.unique(labels)
    worst = 0
    for code in np.unique(codes):
        counts = [np.count_nonzero((labels == fold) & (codes == code)) for fold in folds]
        worst = max(worst, max(counts) - min(counts))
    return worst


def test_the_replicate_folds_are_stratified_by_event_on_units() -> None:
    rng = np.random.default_rng(5)
    n = 240
    event = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    index = np.sort(rng.integers(0, n, size=n))
    plan = engine_module.replicate_plan(
        index, event[index], cross_fit=True, n_folds=5, learner_folds=4, seed=9
    )
    units, first = np.unique(index, return_index=True)
    unit_event, unit_outer = event[units], plan.outer[first]
    # Each cause's units are spread over the outer folds to within one unit.
    assert _spread(unit_outer, unit_event) <= 1
    for position, fold in enumerate(np.unique(plan.outer)):
        training_rows = plan.outer != fold
        inner = np.full(index.size, -1)
        inner[training_rows] = plan.inner[position]
        training = unit_outer != fold
        assert _spread(inner[first][training], unit_event[training]) <= 1
    # Witness: the rare cause has few units, so a draw that ignores the event code would
    # leave some fold without them or with several more.
    assert 5 <= np.count_nonzero(unit_event == 2) <= 30


def test_a_programming_error_in_a_replicate_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(data: Any, index: Any) -> Any:
        raise TypeError("a defect, not a failed refit")

    monkeypatch.setattr(engine_module, "resample", broken)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="a defect, not a failed refit"):
            _fit(_frame(), _engine(n_bootstrap=4))


# ------------------------------------------------------------------------------- the plots


def test_the_curve_draws_the_bootstrap_band_on_request(in_sample: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cleverly import plotting
    from cleverly.plotting import _continuous_curve

    rows, _, multiplier = _continuous_curve.curve_rows(in_sample, "risk")
    limits = _continuous_curve.bootstrap_limits(in_sample, rows, "risk")
    assert limits == {
        name: (float(low), float(high))
        for name, (low, high) in in_sample.bootstrap_band.bands.items()
    }
    # Nonzero witness: the bootstrap band is not the multiplier band.
    assert multiplier is not None and limits != multiplier
    # The survival view mirrors each limit as result.curve(scale="survival") does (this
    # two-cause fit refuses that view, so the arithmetic is read on its risk rows).
    mirrored = _continuous_curve.bootstrap_limits(in_sample, rows, "survival")
    for row in rows:
        low, high = in_sample.bootstrap_band.bands[row.parameter]
        expected = (1.0 - high, 1.0 - low) if row.level else (-high, -low)
        assert mirrored[row.parameter] == expected
    fig = plotting.curve(in_sample, band="bootstrap")
    notes = plotting.notes(fig)
    assert any(note.startswith("The simultaneous band drawn is the studentized") for note in notes)
    plt.close(fig)
    fig = plotting.curve(in_sample)
    assert "The fit also holds a bootstrap band; pass band='bootstrap' to draw it." in (
        plotting.notes(fig)
    )
    plt.close(fig)
    plain = result_module.replace(in_sample, bootstrap_band=None)
    with pytest.raises(ValueError, match="this fit holds no bootstrap band"):
        plotting.curve(plain, band="bootstrap")


def test_the_summary_names_the_bootstrap_or_says_it_was_withheld() -> None:
    from tests.unit.summary_fixtures import survival_continuous

    ran = survival_continuous.bootstrapped().summary(detail="full")
    assert "continuous_bootstrap" in ran.note_codes
    assert "continuous_bootstrap_withheld" not in ran.note_codes
    assert "40 of 40 refits" in ran.note("continuous_bootstrap").text
    assert "Why the bootstrap band is experimental" in str(ran)
    withheld = survival_continuous.bootstrap_withheld().summary()
    assert "continuous_bootstrap_withheld" in withheld.note_codes
    assert "continuous_bootstrap" not in withheld.note_codes
    assert withheld.note("continuous_bootstrap_withheld").text.startswith("2 of 40")
    assert "continuous_bootstrap" not in survival_continuous.single_cause().summary().note_codes


# ---------------------------------------------------------------- the preview of the refits


def _planned_and_realized(estimator: ContinuousSurvivalTMLE, frame: pd.DataFrame) -> Any:
    from tests.unit import _planned_fits_support as support

    return support.witness(
        lambda: estimator._planned_fits(
            frame, time="time", event="event", treatment="a", baseline=BASELINE
        ),
        lambda: _fit(frame, estimator),
    )


def test_the_preview_counts_every_bootstrap_replicate(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    estimator = _engine(
        cross_fit=True,
        n_folds=3,
        hazard_library=[CoxHazard("main"), CoxHazard("treatment")],
        n_bootstrap=5,
        # One process, so the class-wide count sees every replicate's fits.
        n_jobs=1,
    )
    seen = _planned_and_realized(estimator, frame)
    plan, result = seen.plan, seen.result
    assert result.bootstrap_status == "ran" and result.bootstrap.n_failed == 0
    realized = sum(seen.class_wide.values())
    assert plan.exact and realized == plan.total_candidate_fits, (dict(seen.class_wide), plan)
    assert ("bootstrap replicates", 5) in plan.multipliers
    replicate = [row for row in plan.rows if row.scope.endswith("per bootstrap replicate")]
    assert replicate and sum(row.fits for row in replicate) > 0
    # Each replicate refits the same models as the fit, so the replicate rows hold five times
    # the fit's own fits.
    own = sum(row.fits for row in plan.rows if row not in replicate)
    assert sum(row.fits for row in replicate) == 5 * own
    # A replicate whose refit fails makes at most the counted fits.  With the interaction model
    # two of five refits raise on a combined jump, which the engine checks after every initial
    # fit, so here they make all of them.
    failing = _engine(
        cross_fit=True,
        n_folds=3,
        hazard_library=[CoxHazard("main"), CoxHazard("interaction")],
        n_bootstrap=5,
        n_jobs=1,
    )
    short = _planned_and_realized(failing, frame)
    assert short.result.bootstrap.n_failed == 2
    assert sum(short.class_wide.values()) <= short.plan.total_candidate_fits
    assert any("at most the fits counted" in text for text in short.plan.notes)
    # Mutation control: a preview that drops the replicate rows misses the refits.
    monkeypatch.setattr(
        engine_module.ContinuousSurvivalTMLE, "_plan_bootstrap", lambda *args, **kwargs: None
    )
    dropped = _planned_and_realized(estimator, frame)
    assert dropped.plan.total_candidate_fits < sum(dropped.class_wide.values())
