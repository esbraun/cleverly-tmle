"""The continuous-time longitudinal TMLE on draws: reduction, refusals, warnings, folds, the cap.

T-e5, T-e14, T-e17, T-e18, T-e21, T-e22 and T-e23 of X6 part (e), and the result's methods. The
exact-law checks are in ``test_continuous_longitudinal_exact.py``.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    PositivityWarning,
)
from cleverly.interventions import Incremental, Rule, Shift, Static, Stochastic
from cleverly.sensitivity import evalue
from cleverly.survival import (
    ContinuousLongitudinalData,
    ContinuousLongitudinalResult,
    ContinuousLongitudinalTMLE,
    ContinuousSurvivalTMLE,
    CoxHazard,
    longitudinal,
)
from cleverly.survival import _longitudinal_nuisance as nuisance
from cleverly.survival.diagnostics import continuous_truncation_curve
from cleverly.survival.longitudinal_result import PENDING_REASON, USER_LEARNER_REASON
from cleverly.validation import refute
from tests.unit._continuous_longitudinal_support import CT_COLUMNS, simulate_ct

pytestmark = pytest.mark.xdist_group("continuous_longitudinal")

COX = {
    "hazard_library": [CoxHazard("main")],
    "visit_library": [CoxHazard("main")],
    "censoring_library": [CoxHazard("main")],
}
REGIMES = {"always": 1, "never": 0}


def _fit(frame: pd.DataFrame, horizons: Any = (10.0, 20.0), **kwargs: Any) -> Any:
    settings: dict[str, Any] = {
        "regimes": REGIMES,
        "reference": "never",
        "cross_fit": False,
        "random_state": 0,
        **COX,
    }
    settings.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ContinuousLongitudinalTMLE(list(horizons), **settings).fit(frame, **CT_COLUMNS)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return simulate_ct(300, 1)


@pytest.fixture(scope="module")
def result(frame: pd.DataFrame) -> Any:
    return _fit(frame)


# ---------------------------------------------------------------------- T-e5


def test_reduces_to_the_baseline_engine() -> None:
    """T-e5 (ii): with no visit, the initial fits and the targeted risks match the baseline."""
    frame = simulate_ct(300, 3, visits=False)
    data = ContinuousLongitudinalData.from_frame(
        frame,
        id="id",
        start="start",
        stop="stop",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        time_varying=["L"],
    )
    libraries = {"hazard_library": [CoxHazard("main")], "censoring_library": [CoxHazard("main")]}
    for steps, tol in ((0, None), (500, 1e-3)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline = ContinuousSurvivalTMLE(
                [10.0, 20.0],
                cross_fit=False,
                random_state=0,
                variance="influence_curve",
                max_iter=steps,
                tol=tol,
                **libraries,
            ).fit_data(data.baseline_view())
            engine = ContinuousLongitudinalTMLE(
                [10.0, 20.0],
                regimes={"0": 0, "1": 1},
                cross_fit=False,
                random_state=0,
                max_iter=steps,
                tol=tol,
                visit_library=[CoxHazard("main")],
                **libraries,
            ).fit_data(data)
        for name, estimate in baseline.items():
            other = engine[name]
            if steps == 0:
                # The untargeted plug-in and its influence curve at the initial hazards.
                assert other.psi == pytest.approx(estimate.psi, abs=1e-10)
                np.testing.assert_allclose(
                    other.influence_curve, estimate.influence_curve, atol=1e-10
                )
            elif name.startswith(("cif_", "ate_")):
                # Each engine stops within its own cutoff, one regime at a time here.
                sigma = estimate.std_error * np.sqrt(300)
                assert abs(other.psi - estimate.psi) <= 2 * 1e-3 * sigma


def test_saturated_fits_reduce_to_the_baseline_engine(monkeypatch: Any) -> None:
    """T-e5 (i): with no visit and saturated learners in sample, no walk moves and every
    estimate, standard error and influence curve equals the baseline engine's to 1e-10.

    1,500 units keep every cell of ``(A, W1, W2, L)`` populated with survivors, so no hazard
    jump reaches one and the saturated initial fits solve every score. The baseline refuses a
    user treatment learner in sample, so the test sets ``cross_fit`` after construction, as
    the longitudinal side lifts C13.
    """
    from tests.unit._continuous_longitudinal_support import (
        CellClassifier,
        CellHazard,
        CellRegressor,
    )

    monkeypatch.setattr(longitudinal, "_user_learners_in_sample", lambda estimator: None)
    data = ContinuousLongitudinalData.from_frame(
        simulate_ct(1500, 3, visits=False),
        id="id",
        start="start",
        stop="stop",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        time_varying=["L"],
    )
    saturated = {
        "hazard_library": [CellHazard()],
        "censoring_library": [CellHazard()],
        "treatment_learner": CellClassifier(),
        "g_bounds": 1e-6,
        "tol": 1e-12,
        "random_state": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline_estimator = ContinuousSurvivalTMLE(
            [10.0, 20.0], variance="influence_curve", **saturated
        )
        baseline_estimator.cross_fit = False
        baseline = baseline_estimator.fit_data(data.baseline_view())
        engine = ContinuousLongitudinalTMLE(
            [10.0, 20.0],
            regimes={"0": 0, "1": 1},
            cross_fit=False,
            visit_library=[CellHazard()],
            continuation_learner=CellRegressor(),
            **saturated,
        ).fit_data(data)
    assert baseline.record.steps == 0 and baseline.record.converged
    assert engine.record.steps == ((0,), (0,)) and engine.record.converged
    for name, estimate in baseline.items():
        other = engine[name]
        assert other.psi == pytest.approx(estimate.psi, abs=1e-10), name
        assert other.std_error == pytest.approx(estimate.std_error, abs=1e-10), name
        np.testing.assert_allclose(other.influence_curve, estimate.influence_curve, atol=1e-10)


# ---------------------------------------------------------------------- the result


def test_result_reads_every_regime(result: Any) -> None:
    assert isinstance(result, ContinuousLongitudinalResult)
    assert result.arms == ("never", "always")
    curve = result.curve()
    assert list(curve["arm"]).count("always vs never") == 4
    rd = result["ate_regimen[always vs never, 1 @ t=10]"]
    always, never = (result[f"cif_regimen[{a}, 1 @ t=10]"] for a in ("always", "never"))
    assert rd.psi == pytest.approx(always.psi - never.psi)
    ratio = result.ratio("cif_regimen[always, 1 @ t=10]", "cif_regimen[never, 1 @ t=10]")
    assert ratio.psi == pytest.approx(always.psi / never.psi)
    total = result.contrast(
        lambda x: x[0] + x[1], ["cif_regimen[always, 1 @ t=20]", "cif_regimen[always, 2 @ t=20]"]
    )
    assert total.support.label == "experimental"
    json.dumps(result.to_dict(), allow_nan=False)
    assert len(result.to_frame()) == len(result)
    assert set(result.initial_estimates) <= set(result)


def test_initial_estimates_follow_the_arms(frame: pd.DataFrame) -> None:
    """With no step, each initial risk is its estimate, whatever order the regimes came in.

    The record keeps the regimes in the order given and the result puts the reference first;
    here the two orders differ, so reading the record in the result's order would swap them.
    """
    fit = _fit(frame, regimes={"never": 0, "always": 1}, reference="always", max_iter=0)
    assert fit.arms == ("always", "never")
    assert fit.record.regimes == ("never", "always")
    initial = fit.initial_estimates
    assert set(initial) == {name for name in fit if not name.startswith("rr_")}
    for name, value in initial.items():
        assert value == pytest.approx(fit[name].psi, abs=1e-12)
    always, never = (fit[f"cif_regimen[{a}, 1 @ t=10]"].psi for a in ("always", "never"))
    assert always != pytest.approx(never)


def test_every_estimate_is_experimental(result: Any) -> None:
    """The labels of 3.5.9 before E2: every estimate names the pending study."""
    for estimate in result.values():
        assert estimate.support.label == "experimental"
        assert PENDING_REASON in estimate.support.reason


def test_a_chosen_shipped_library_names_the_user_learner_reason(result: Any) -> None:
    """Review S6: the label reads the default libraries, not any shipped learner class.

    The fixture's one-member Cox libraries are shipped learners, which C13 lets through in
    sample, but they are not the default libraries a study would measure.
    """
    assert all(USER_LEARNER_REASON in e.support.reason for e in result.values())
    assert ContinuousLongitudinalTMLE([10.0], regimes=REGIMES)._defaults
    assert not ContinuousLongitudinalTMLE([10.0], regimes=REGIMES, **COX)._defaults


def test_restricted_means_and_assessments_refuse(result: Any) -> None:
    """C9, and the refits and sensitivity analyses the plan refuses (3.5.11)."""
    with pytest.raises(CapabilityError, match="restricted mean is not one of its targets"):
        result.rmtl("always", 10.0)
    with pytest.raises(CapabilityError, match="restricted mean"):
        result.rmst("always", 10.0)
    with pytest.raises(CapabilityError, match="no adapter replays") as caught:
        result.refute()
    assert caught.value.refusal.roadmap == "X105"
    with pytest.raises(CapabilityError, match="point exposure") as caught:
        result.sensitivity.evalue()
    assert caught.value.refusal.reason.value == "inference_gap"
    with pytest.raises(CapabilityError, match="not a target time"):
        result.curve(times=[15.0])


def test_summary_diagnostics_and_batteries_read_the_record(result: Any) -> None:
    """T-e19: the summary, the record's diagnostics, and batteries that name each refusal."""
    record = result.record
    assert "irregular visits" in str(result.summary())
    assert "Assumptions" in str(result.summary(detail="full"))
    support = result.diagnostics.support()
    assert support.bounded_share == record.bounded_share
    assert support.caps == dict(zip(record.regimes, record.caps, strict=True))
    scores = result.diagnostics.score_equations()
    assert scores.passed == record.converged
    assert scores.cutoff == record.cutoff
    with pytest.raises(CapabilityError, match="s_n / N"):
        result.diagnostics.score_equations(tolerance=1e-3)
    nuisances = result.diagnostics.nuisance_models()
    assert [row.role for row in nuisances.rows][-2:] == ["censoring hazard", "treatment"]
    report = result.assess()
    refused = {item.name: item.refusal for item in report.diagnostics.items if item.refusal}
    assert refused["refute"].roadmap == "X105"
    assert refused["truncation_curve"].roadmap == "X105"
    reasons = {item.refusal.reason.value for item in report.sensitivity.items}
    assert reasons == {"inference_gap"}
    assert result.validate().passed
    with pytest.raises(CapabilityError, match="no adapter replays") as caught:
        refute(result)
    assert caught.value.refusal.roadmap == "X105"
    with pytest.raises(CapabilityError, match="no adapter replays"):
        continuous_truncation_curve(result)
    with pytest.raises(CapabilityError, match="point exposure"):
        evalue(result)


def test_record_states_the_window_and_the_cap(result: Any) -> None:
    """The record of 3.5.3: the history window, K* and the share past it."""
    record = result.record
    assert "window" in record.history_restriction
    assert all(cap >= 1 for cap in record.caps)
    assert all(share >= 0.0 for row in record.past_cap_share for share in row)


# ---------------------------------------------------------------------- T-e14, T-e17


@pytest.mark.parametrize("seed", range(20))
def test_risks_are_monotone_and_sum_below_one(seed: int) -> None:
    """T-e14: each regime's plug-in risks rise with the target time; the causes sum to <= 1.

    The plan names the default learners. The hazards here are one-member Cox libraries in
    sample, with no step, for the fast tier. The property rests on the binned continuation,
    which keeps its default learner here.
    """
    fit = _fit(simulate_ct(120, 100 + seed), horizons=(20.0, 25.0, 30.0), max_iter=0)
    for arm in fit.arms:
        risks = np.array(
            [[fit[f"cif_regimen[{arm}, {c} @ t={t}]"].psi for t in (20, 25, 30)] for c in (1, 2)]
        )
        assert np.all(np.diff(risks, axis=1) >= -1e-12)
        assert np.all(risks.sum(axis=0) <= 1 + 1e-12)


def test_threads_and_memory_budget_do_not_change_the_fit(frame: pd.DataFrame) -> None:
    """T-e17: n_jobs and a budget above the prediction leave every estimate bit-identical."""
    one = _fit(frame)
    many = _fit(frame, n_jobs=4, max_memory_bytes=10**10)
    for name, estimate in one.items():
        assert many[name].psi == estimate.psi
        np.testing.assert_array_equal(many[name].influence_curve, estimate.influence_curve)
    with pytest.raises(Exception, match="memory"):
        _fit(frame, max_memory_bytes=10**4)


# ---------------------------------------------------------------------- T-e18


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (Stochastic(lambda x: 0.5, name="half", density_kind="known"), "a known stochastic policy"),
        (Shift(1.0, None), "a modified treatment policy"),
        (Incremental(2.0), "an odds tilt"),
        ({"visits": 1}, "an intervention on the visit intensity"),
    ],
)
def test_other_policies_refuse(value: Any, kind: str) -> None:
    """C7: every policy other than a static level or a deterministic rule."""
    with pytest.raises(CapabilityError, match=kind) as caught:
        ContinuousLongitudinalTMLE([10.0], regimes={"a": value, "never": 0})
    assert "not built in continuous time" in str(caught.value)


def _known(function: Any) -> Rule:
    return Rule(function, "rule", rule_kind="known")


def test_rules_need_the_known_declaration(frame: pd.DataFrame) -> None:
    """S1: a Rule declared known is a regime; a bare callable or an undeclared Rule refuses."""

    def treat_when_l(f: Any) -> Any:
        return (f["L"] == 1).astype(int)

    fit = _fit(frame, regimes={"rule": _known(treat_when_l), "never": 0})
    assert fit.arms == ("never", "rule")
    with pytest.raises(CapabilityError, match="needs a declaration") as caught:
        ContinuousLongitudinalTMLE([10.0], regimes={"rule": treat_when_l, "never": 0})
    assert caught.value.refusal.reason.value == "invalid_declaration"
    # A Rule whose declaration was changed after it was built refuses at the estimator.
    undeclared = _known(treat_when_l)
    object.__setattr__(undeclared, "rule_kind", None)
    with pytest.raises(CapabilityError, match="needs a declaration"):
        ContinuousLongitudinalTMLE([10.0], regimes={"rule": undeclared, "never": 0})
    object.__setattr__(undeclared, "rule_kind", "estimated")
    with pytest.raises(CapabilityError, match="learned from the analysis sample"):
        ContinuousLongitudinalTMLE([10.0], regimes={"rule": undeclared, "never": 0})


def test_static_intervention_reads_its_level(frame: pd.DataFrame) -> None:
    fit = _fit(frame, regimes={"always": Static(1), "never": Static(0)})
    assert fit.arms == ("never", "always")


def test_continuous_and_multilevel_treatments_refuse(frame: pd.DataFrame) -> None:
    """C7 for a continuous treatment, C14 for more than two levels."""
    continuous = frame.assign(A=frame["A"] + np.linspace(0.01, 0.5, len(frame)))
    with pytest.raises(CapabilityError, match="a continuous treatment"):
        _fit(continuous)
    three = frame.assign(A=np.where(frame["L"] == 1, 2, frame["A"]))
    with pytest.raises(CapabilityError, match="treatment with 3 levels is not built"):
        _fit(three, regimes={"always": 1, "never": 0})


def test_rule_outside_the_levels_refuses(frame: pd.DataFrame) -> None:
    """C10."""
    with pytest.raises(DataError, match="rule 'bad' returned 2 at"):
        _fit(frame, regimes={"bad": _known(lambda f: np.full(len(f), 2)), "never": 0})
    with pytest.raises(DataError, match="returned None"):
        _fit(frame, regimes={"bad": _known(lambda f: [None] * len(f)), "never": 0})


def test_follow_up_and_first_events_refuse(frame: pd.DataFrame) -> None:
    """C11 past the follow-up, C12 with no event of a cause by the first target time."""
    with pytest.raises(DataError, match="followed only to"):
        _fit(frame, horizons=(10.0, 40.0))
    with pytest.raises(DataError, match="have no event of cause"):
        _fit(frame, horizons=(1e-4, 10.0))


def test_user_learners_in_sample_refuse() -> None:
    """C13: a user learner for any nuisance with cross_fit=False."""
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(MethodConfigurationError, match="Lemma S8"):
        ContinuousLongitudinalTMLE(
            [10.0], regimes=REGIMES, cross_fit=False, treatment_learner=LogisticRegression()
        )
    ContinuousLongitudinalTMLE([10.0], regimes=REGIMES, cross_fit=True, treatment_learner=None)


def test_visit_cap_below_one_refuses() -> None:
    """C15: a training fold with fewer than five followers who visit."""
    sparse = simulate_ct(80, 7, visit_rate=0.004)
    with pytest.raises(DataError, match=r"training fold \d+ holds \d+ units"):
        _fit(sparse, cross_fit=True, n_folds=10, horizons=(20.0, 30.0))


def test_few_followers_and_past_cap_warn() -> None:
    """W1 below 30 followers at the last target; W2 above 0.05 past the cap."""
    frame = simulate_ct(150, 11, visit_rate=0.4)
    with pytest.warns(UserWarning) as caught:
        ContinuousLongitudinalTMLE(
            [10.0, 20.0], regimes=REGIMES, cross_fit=False, random_state=0, **COX
        ).fit(frame, **CT_COLUMNS)
    texts = [str(w.message) for w in caught]
    assert any("fewer than 30" in t and "experimental" in t for t in texts)
    assert any("visit more than K*" in t for t in texts)


def test_visit_at_a_death_is_dropped_with_a_warning() -> None:
    """The order rule of 3.5.1: a visit recorded at the death time follows the death."""
    frame = pd.DataFrame(
        {
            "id": [0, 0, 1],
            "start": [0.0, 2.0, 0.0],
            "stop": [2.0, 2.0, 3.0],
            "event": [0, 1, 0],
            "W1": [0, 0, 1],
            "W2": [1, 1, 2],
            "L": [0, 1, 0],
            "A": [1, 1, 0],
        }
    )
    with pytest.warns(UserWarning, match="1 unit records a visit and a death"):
        data = ContinuousLongitudinalData.from_frame(
            frame,
            id="id",
            start="start",
            stop="stop",
            event="event",
            treatment="A",
            baseline=["W1", "W2"],
            time_varying=["L"],
        )
    assert data.dropped_visits == 1


# ---------------------------------------------------------------------- T-e21, T-e23


class _Stop(Exception):
    """Raised after the first hazard fit, which holds every fold label the test reads."""


def _first_hazard_fit(frame: pd.DataFrame, monkeypatch: Any) -> tuple[Any, Any, Any]:
    """The segments, the outer folds by unit and the first hazard fit's inner labels by row."""
    seen: list[Any] = []
    outer: list[Any] = []
    calls: list[Any] = []
    original_segments, original_folds = nuisance.segments, nuisance.unit_folds

    def segments(*args: Any) -> Any:
        seen.append(original_segments(*args))
        return seen[-1]

    def unit_folds(event: Any, n_folds: int, rng: Any) -> Any:
        outer.append(original_folds(event, n_folds, rng))
        return outer[-1]

    def fit_hazard(library: Any, X: Any, time: Any, event: Any, entry: Any, folds: Any, w: Any):
        calls.append(folds.copy())
        raise _Stop

    with monkeypatch.context() as patch:
        patch.setattr(nuisance, "segments", segments)
        patch.setattr(nuisance, "unit_folds", unit_folds)
        patch.setattr(nuisance, "fit_hazard", fit_hazard)
        with pytest.raises(_Stop):
            _fit(frame, cross_fit=True, n_folds=3)
    return seen[0], outer[0], calls[0]


def _one_label_per_unit(seg: Any, folds: Any, inner: Any) -> bool:
    """Whether the first outer fold's training rows carry one inner label per unit."""
    train = folds[seg.unit] != np.unique(folds)[0]
    units = seg.unit[train]
    if inner.size != units.size:
        return False
    return all(np.unique(inner[units == unit]).size == 1 for unit in np.unique(units))


def test_folds_are_drawn_by_unit(frame: pd.DataFrame, monkeypatch: Any) -> None:
    """T-e21: every unit's rows share one outer fold and one inner label.

    The fit stops after its first hazard fit, which holds the labels. Mutation: the labels
    copied to the rows by row rather than by unit; the assertion fails.
    """
    seg, folds, inner = _first_hazard_fit(frame, monkeypatch)
    assert folds.shape == (frame["id"].nunique(),)
    assert _one_label_per_unit(seg, folds, inner)

    def by_row(labels: Any, unit: Any) -> Any:
        return np.random.default_rng(0).permutation(labels[unit])

    monkeypatch.setattr(longitudinal, "by_unit", by_row)
    assert not _one_label_per_unit(*_first_hazard_fit(frame, monkeypatch))


def _cap_by_hand(frame: pd.DataFrame, folds: Any, level: int, last: float) -> int:
    """K* counted from the frame: the largest K whose every training fold holds 5 followers.

    A follower through visit K - 1 took ``level`` at baseline and at visits 1 to K - 1, and
    counts when its visit K falls before the last target time.
    """
    ids = {unit: i for i, unit in enumerate(np.sort(frame["id"].unique()))}
    histories = [
        (ids[unit], rows["start"].to_numpy(), rows["A"].to_numpy())
        for unit, rows in frame.sort_values(["id", "start"]).groupby("id")
    ]
    K = 0
    while True:
        K += 1
        units = np.array(
            [
                i
                for i, start, a in histories
                if start.size > K and start[K] < last and np.all(a[:K] == level)
            ],
            dtype=int,
        )
        if units.size == 0:
            return K - 1
        if min(np.count_nonzero(folds[units] != fold) for fold in np.unique(folds)) < 5:
            return K - 1


def test_visit_cap_under_cross_fitting(monkeypatch: Any) -> None:
    """T-e23: one unit holds the most visits; the cross-fitted fit runs at K* counted by hand.

    A 200-unit draw (the plan's 400 halved for the fast tier; one unit still holds the most
    visits). The share's hand computation is on PL2, in the exact file. Mutation: the cap at
    the full-sample maximum; a training fold holds no follower that far and the fit fails.
    """
    frame = simulate_ct(200, 4)
    counts = frame.groupby("id").size() - 1
    assert (counts == counts.max()).sum() == 1
    outer: list[Any] = []
    original = nuisance.unit_folds

    def unit_folds(event: Any, n_folds: int, rng: Any) -> Any:
        outer.append(original(event, n_folds, rng))
        return outer[-1]

    horizons = [10.0, 20.0, 30.0]
    estimator = ContinuousLongitudinalTMLE(
        horizons, regimes=REGIMES, reference="never", n_folds=10, random_state=0, **COX
    )
    with monkeypatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(nuisance, "unit_folds", unit_folds)
        fit = estimator.fit(frame, **CT_COLUMNS)
    assert all(np.isfinite(e.psi) and np.isfinite(e.std_error) for e in fit.values())
    hand = tuple(
        _cap_by_hand(frame, outer[0], REGIMES[r], horizons[-1]) for r in fit.record.regimes
    )
    assert fit.record.caps == hand
    assert max(hand) < counts.max()

    monkeypatch.setattr(nuisance, "visit_cap", lambda *args, **kwargs: (int(counts.max()), -1, 0))
    # The continuation of the first node no training fold's follower reaches has labels of
    # one class only, which its logistic learners refuse.
    with warnings.catch_warnings(), pytest.raises(ValueError, match="2 classes"):
        warnings.simplefilter("ignore")
        estimator.fit(frame, **CT_COLUMNS)


# ---------------------------------------------------------------------- T-e22


def _scheduled(off_schedule: bool, seed: int = 5) -> pd.DataFrame:
    """Every unit with W2 above -1 visits at day 5 unless it left before.

    A Cox visit model on W2 then puts a jump of one or more at day 5 on the units with the
    largest W2. With ``off_schedule`` the unit with the largest W2 skips the visit, so a row
    whose fitted jump reaches one does not visit there.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=200)
    top = int(np.argmax(x))
    rows = []
    for unit in range(200):
        w1 = int(rng.random() < 0.5)
        a = int(rng.random() < 0.5)
        l0 = int(rng.random() < 0.5)
        death = rng.exponential(1 / 0.04)
        cause = 1 + int(rng.random() < 0.4)
        end = min(death, 12.0)
        event = cause if death < 12.0 else 0
        base = {"id": unit, "W1": w1, "W2": x[unit], "A": a}
        visits = x[unit] > -1 and not (off_schedule and unit == top)
        if end > 5.0 and visits:
            rows.append({**base, "start": 0.0, "stop": 5.0, "event": 0, "L": l0})
            rows.append({**base, "start": 5.0, "stop": end, "event": event, "L": 1 - l0})
        else:
            rows.append({**base, "start": 0.0, "stop": end, "event": event, "L": l0})
    return pd.DataFrame.from_records(rows)


def test_visit_jump_bound(monkeypatch: Any) -> None:
    """T-e22: a visit jump of one stays where every row at risk visits, and is bounded elsewhere."""
    largest: list[float] = []
    original = longitudinal.visit_exit

    def spy(deaths: Any, visit: Any) -> Any:
        largest.append(float(visit.max()))
        return original(deaths, visit)

    monkeypatch.setattr(longitudinal, "visit_exit", spy)
    scheduled = _fit(_scheduled(False), horizons=(4.0, 10.0))
    # Witness: the fitted jump reaches one at day 5, where every row at risk visits.
    assert max(largest) == 1.0
    assert scheduled.record.visit_bound_share == 0.0
    assert all(np.isfinite(e.std_error) for e in scheduled.values())
    with pytest.warns(PositivityWarning, match="a visit jump of one or more"):
        ContinuousLongitudinalTMLE(
            [4.0, 10.0], regimes=REGIMES, cross_fit=False, random_state=0, **COX
        ).fit(_scheduled(True), **CT_COLUMNS)
    skipped = _fit(_scheduled(True), horizons=(4.0, 10.0))
    assert skipped.record.visit_bound_share > 0.0
    assert all("bounded below one" in e.support.reason for e in skipped.values())
    assert all(np.isfinite(e.std_error) for e in skipped.values())


def test_visit_jump_bound_under_cross_fitting(monkeypatch: Any) -> None:
    """T-e22 under cross-fitting: the scheduled-visit judgment reads the full-sample rows.

    The unit with the largest W2 skips the day-5 visit. It is held out of one fold, whose
    training rows all visit at day 5, so its stitched jump reaches one there although it does
    not visit; the full-sample judgment then bounds the jump. That witness carries the check.
    The plan's mutation (a jump of one kept everywhere) does not make the score non-finite:
    the S_k = 0 guard keeps every term finite, and the bound share it would zero is computed
    from the mutated rule itself, so the test does not run it.
    """
    calls: list[tuple[Any, Any, Any, Any]] = []
    original = longitudinal.scheduled_visits

    def spy(factual: Any, at_risk: Any, exits: Any) -> Any:
        out = original(factual, at_risk, exits)
        calls.append((factual, at_risk, exits, out))
        return out

    monkeypatch.setattr(longitudinal, "scheduled_visits", spy)
    estimator = ContinuousLongitudinalTMLE(
        [4.0, 10.0], regimes=REGIMES, n_folds=5, random_state=0, **COX
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", PositivityWarning)
        with pytest.raises(PositivityWarning, match="a visit jump of one or more"):
            estimator.fit(_scheduled(True), **CT_COLUMNS)
    # Witness: a held-out row's stitched jump reaches one where it does not visit.
    skipped = [bool(((f >= 1.0) & r & ~e).any() and not out.any()) for f, r, e, out in calls]
    assert any(skipped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = estimator.fit(_scheduled(True), **CT_COLUMNS)
    assert fit.record.cross_fitted
    assert fit.record.visit_bound_share > 0.0
    assert all("bounded below one" in e.support.reason for e in fit.values())
    assert all(np.isfinite(e.std_error) for e in fit.values())


def test_in_sample_fits_take_the_default_libraries(monkeypatch: Any) -> None:
    """C13 lets the shipped hazard learners through in sample."""
    estimator = ContinuousLongitudinalTMLE([10.0], regimes=REGIMES, cross_fit=False, **COX)
    assert estimator._default_nuisances_only()
    assert longitudinal.IN_SAMPLE_LEARNER_TEXT.startswith("an in-sample fit")


class _Captured(Exception):
    pass


def test_default_treatment_model_reads_the_baseline_interaction(
    frame: pd.DataFrame, monkeypatch: Any
) -> None:
    """The default treatment selection fits on ``[X, b, b X]``; a supplied learner on ``X``."""
    seen: list[Any] = []

    def spy(learner: Any, design: Any, *args: Any) -> Any:
        seen.append((learner, design))
        raise _Captured

    monkeypatch.setattr(nuisance, "_fit_propensity", spy)
    with pytest.raises(_Captured):
        _fit(frame)
    learner, design = seen[0]
    assert learner is None
    p = (design.shape[1] - 1) // 2
    X, b, bX = design[:, :p], design[:, p], design[:, p + 1 :]
    # The visit count is the design's second-to-last column; it is zero exactly at baseline.
    assert set(np.unique(b)) == {0.0, 1.0}
    np.testing.assert_array_equal(b, (X[:, -2] == 0).astype(float))
    np.testing.assert_array_equal(bX, b[:, None] * X)

    from sklearn.linear_model import LogisticRegression

    supplied = LogisticRegression()
    seen.clear()
    monkeypatch.setattr(longitudinal, "_user_learners_in_sample", lambda estimator: None)
    with pytest.raises(_Captured):
        _fit(frame, treatment_learner=supplied)
    assert seen[0][0] is supplied
    np.testing.assert_array_equal(seen[0][1], X)


def test_continuation_inner_fold_without_events_predicts_the_constant() -> None:
    """An inner training fold of the default continuation can hold no event.

    The whole training set has events (so the outer guard of ``ContinuationModel.fit`` passes),
    but every event sits in inner fold 0, so the fit on folds 1 and 2 sees one label.  Before
    the guard, that raised scikit-learn's single-class error and lost the fit; the declared
    run of part (e) lost one fit of 3,200 to it.  The single-label member predicts its label,
    and the cross-validated choice and the final fit proceed.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(90, 2))
    folds = np.repeat([0, 1, 2], 30)
    y = np.where(folds == 0, rng.uniform(0.05, 0.4, size=90), 0.0)
    w = np.ones(90)

    for value in (0.0, 1.0):
        member = nuisance._SoftLogistic(np.inf, 0).fit(X[:20], np.full(20, value), w[:20])
        np.testing.assert_array_equal(member.predict(X), np.full(90, value))

    fitted = nuisance._DefaultContinuation(folds, 0).fit(X, y, w)
    held = folds == 0
    # Both members train on the same event-free folds for held fold 0, so both predict zero
    # there and carry the same clipped loss on it.
    zero = nuisance._SoftLogistic(np.inf, 0).fit(X[~held], y[~held], w[~held])
    np.testing.assert_array_equal(zero.predict(X[held]), np.zeros(int(held.sum())))
    assert np.all(np.isfinite(fitted.losses_))
    prediction = fitted.predict(X)
    assert np.all((prediction > 0) & (prediction < 1))
