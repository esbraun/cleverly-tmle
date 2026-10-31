"""Quantile trimming of the per-node density ratios, ``LTMLE(ratio_trim=q)`` (X33 part d).

The exact checks run on the two-node SMART law of :mod:`tests.unit._known_mechanism_support`
with declared treatment and retention probabilities, so every per-node ratio is known by
hand. The law has units that leave the static plan at node one and follow it at node two,
and units censored after node one, so the zero pattern of ``lmtp``'s ratio matrix
(``density_ratios.R:55-61``: zero by node ``t`` alone) is visible. Each mutation control
replaces one step of the rule and shows that the hand check fails.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    IPSIMethod,
    LongitudinalTreatment,
    MethodConfigurationError,
    PointTreatment,
    RegimeMean,
    Targeting,
    TMLEMethod,
)
from cleverly._inference_status import STATUS_RECORDS, TRIMMED_RATIO_STATUS
from cleverly.datasets import make_linear_ate
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR, LongitudinalResult
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal import ratio_trim as trim_module
from cleverly.longitudinal.ratio_trim import RatioTrim, node_ratio_entries
from cleverly.utils.bounds import bound
from tests.unit import _known_mechanism_support as law
from tests.unit._natural_course_support import NeverFit
from tests.unit.test_known_node_mechanisms import SmartMechanism

#: The module, which the package's ``sdr`` function shadows as an attribute.
sdr_module = importlib.import_module("cleverly.longitudinal.sdr")

REGIMENS = {"always": 1, "never": 0}
#: A quantile that binds on the SMART law's ratios. The ratios take the values 0, 4/3, 8/3
#: and 4, and the 0.85 quantile is 8/3 for both plans, so every entry of 4 is trimmed.
BINDING = 0.85
#: A quantile at which type 7 interpolates between the plan's two largest ratio values, so
#: the quantile of one outer fold's rows differs from the pooled one.
PER_FOLD = 0.8965
#: The exact-law tolerance: each node's fluctuation stops at ``tol = 1e-10``.
EXACT = 1e-9


def _ltmle(**settings: Any) -> LTMLE:
    base: dict[str, Any] = {
        "reference": "never",
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
        "simultaneous": False,
    }
    base.update(settings)
    return LTMLE(REGIMENS, **base)


def _fit(frame: pd.DataFrame, **settings: Any) -> LongitudinalResult:
    return _ltmle(**settings).fit(
        frame,
        **law.SMART_COLUMNS,
        censoring=["C1", "C2"],
        treatment_probabilities=law.SMART_TREATMENT,
        censoring_probabilities=law.SMART_CENSORING,
    )


class SmartRetention(BaseEstimator, ClassifierMixin):
    """A censoring learner that returns the SMART law's node-one retention from ``L0``."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> SmartRetention:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        l0 = np.asarray(design, dtype=float)[:, 0]
        one = np.where(l0 == 1.0, law.SMART_C1[1], law.SMART_C1[0])
        return np.column_stack([1.0 - one, one])


def _fitted(frame: pd.DataFrame, **settings: Any) -> LongitudinalResult:
    """The fit with learners that return the law, which ``g_bounds`` may truncate."""
    return _ltmle(
        treatment_learner=SmartMechanism(), censoring_learner=SmartRetention(), **settings
    ).fit(frame, **law.SMART_COLUMNS, censoring=["C1", "C2"])


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return law.smart_frame(censoring=True)


def _hand_entries(frame: pd.DataFrame, arm: int) -> np.ndarray:
    """``lmtp``'s per-node ratio of the static plan ``arm`` at both nodes, from the columns.

    Entry ``(i, t)`` is ``1{A_t = arm} 1{C_t = 1} / (g_t(arm) c_t)``, zero where the unit is
    not at risk at ``t``. Node two reads node two alone: a unit that left the plan at node
    one and takes ``arm`` at node two has a positive entry.
    """
    a1 = frame["A1"].to_numpy(dtype=float)
    a2 = frame["A2"].to_numpy(dtype=float)
    c1 = frame["C1"].to_numpy(dtype=float)
    c2 = np.nan_to_num(frame["C2"].to_numpy(dtype=float), nan=0.0)
    g1 = frame[f"g1_{arm}"].to_numpy(dtype=float)
    g2 = np.nan_to_num(frame[f"g2_{arm}"].to_numpy(dtype=float), nan=1.0)
    r1 = frame["r1"].to_numpy(dtype=float)
    r2 = np.nan_to_num(frame["r2"].to_numpy(dtype=float), nan=1.0)
    first = np.where((a1 == arm) & (c1 == 1.0), 1.0 / (g1 * r1), 0.0)
    second = np.where((c1 == 1.0) & (a2 == arm) & (c2 == 1.0), 1.0 / (g2 * r2), 0.0)
    return np.column_stack([first, second])


def _type7(values: np.ndarray, q: float) -> float:
    """R's default ``quantile(x, q)`` (type 7) written out: ``x[j] + h (x[j+1] - x[j])``."""
    ordered = np.sort(values.ravel())
    position = (ordered.size - 1) * q
    low = int(np.floor(position))
    high = min(low + 1, ordered.size - 1)
    return float(ordered[low] + (position - low) * (ordered[high] - ordered[low]))


def _check_trim(result: LongitudinalResult, frame: pd.DataFrame, quantile: float) -> None:
    """The fit's trim and clever covariates equal the hand construction, plan by plan."""
    lower, upper = result.config.g_bounds
    for label, arm in (("always", 1), ("never", 0)):
        fit = result.fits[label]
        trim = fit.ratio_trim
        assert trim is not None
        entries = _hand_entries(frame, arm)
        np.testing.assert_allclose(trim.entries, entries, rtol=1e-15, atol=0.0)
        threshold = _type7(entries, quantile)
        assert trim.threshold == pytest.approx(threshold, rel=1e-15, abs=0.0)
        trimmed = np.minimum(entries, threshold)
        cumulative = np.cumprod(trimmed, axis=1)
        np.testing.assert_allclose(trim.cumulative, cumulative, rtol=1e-15, atol=0.0)
        for step in fit.steps:
            followers = step.trained_on
            expected = 1.0 / bound(1.0 / cumulative[followers, step.time - 1], lower, upper)
            np.testing.assert_allclose(step.clever[followers], expected, rtol=1e-12)
            assert np.all(step.clever[~followers] == 0.0)


class TestTheRule:
    def test_the_quantile_is_r_type_7_over_every_entry_zeros_and_ties_included(self) -> None:
        """``numpy.quantile(method="linear")`` is R's type 7 on a matrix with zeros and ties."""
        matrix = np.array(
            [[0.0, 2.0, 2.0], [4.0, 0.0, 2.0], [1.5, 4.0, 0.0], [2.0, 2.0, 8.0]], dtype=float
        )
        for q in (0.1, 0.5, 0.75, 0.9, 0.999):
            assert float(np.quantile(matrix.ravel(), q, method="linear")) == pytest.approx(
                _type7(matrix, q), rel=1e-15, abs=0.0
            )

    def test_entries_threshold_and_clever_covariates_equal_the_hand_construction(
        self, frame: pd.DataFrame
    ) -> None:
        _check_trim(_fit(frame, ratio_trim=BINDING), frame, BINDING)

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.PositivityWarning")
    def test_the_composition_with_an_upper_bound_below_one(self, frame: pd.DataFrame) -> None:
        """``num / bound(num / R~)``: an ``upper < 1`` that binds on an untrimmed row stays."""
        bounds = (0.05, 0.3)
        result = _fitted(frame, ratio_trim=BINDING, g_bounds=bounds)
        _check_trim(result, frame, BINDING)
        # The shipped composition at the same bounds, on the rows the trim never moved.
        untrimmed = _fitted(frame, g_bounds=bounds)
        for label in REGIMENS:
            trim = result.fits[label].ratio_trim
            assert trim is not None
            moved = np.logical_or.accumulate(trim.trimmed, axis=1)
            for step, shipped in zip(
                result.fits[label].steps, untrimmed.fits[label].steps, strict=True
            ):
                kept = step.trained_on & ~moved[:, step.time - 1]
                assert kept.any()
                np.testing.assert_array_equal(step.clever[kept], shipped.clever[kept])

    def test_a_unit_that_rejoins_the_plan_has_a_positive_entry(self, frame: pd.DataFrame) -> None:
        """Zero by node ``t`` alone: ``A1 != 1``, ``A2 = 1`` gives a positive node-two entry."""
        result = _fit(frame, ratio_trim=BINDING)
        trim = result.fits["always"].ratio_trim
        assert trim is not None
        rejoined = (frame["A1"].to_numpy() == 0.0) & (frame["A2"].to_numpy() == 1.0)
        assert rejoined.any()
        assert np.all(trim.entries[rejoined, 1] > 0.0)
        assert np.all(trim.cumulative[rejoined, 1] == 0.0)

    def test_a_censored_unit_has_zero_entries_from_its_censoring_node(
        self, frame: pd.DataFrame
    ) -> None:
        result = _fit(frame, ratio_trim=BINDING)
        trim = result.fits["always"].ratio_trim
        assert trim is not None
        censored = frame["C1"].to_numpy() == 0.0
        assert censored.any()
        assert np.all(trim.entries[censored] == 0.0)

    def test_the_trimmed_share_is_over_positive_entries(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, ratio_trim=BINDING)
        for label, arm in (("always", 1), ("never", 0)):
            entries = _hand_entries(frame, arm)
            threshold = _type7(entries, BINDING)
            expected = np.count_nonzero(entries > threshold) / np.count_nonzero(entries > 0.0)
            assert result.fits[label].trimmed_share == pytest.approx(expected, rel=1e-15)
            assert result.fits[label].trim_threshold == pytest.approx(threshold, rel=1e-15)

    def test_a_cross_fitted_fit_trims_the_pooled_out_of_fold_matrix_once(
        self, frame: pd.DataFrame
    ) -> None:
        _check_trim(_fit(frame, ratio_trim=BINDING, n_folds=3), frame, BINDING)


class TestTheWitness:
    def test_a_binding_trim_moves_the_estimate_and_withholds_the_interval(
        self, frame: pd.DataFrame
    ) -> None:
        """With every factor known the untrimmed fit returns the truth; the trimmed one moves."""
        plain = _fit(frame)
        trimmed = _fit(frame, ratio_trim=BINDING)
        truth = law.smart_truth(1, 1)
        assert abs(plain["ey_regimen[always]"].psi - truth) <= EXACT
        assert abs(trimmed["ey_regimen[always]"].psi - truth) > 1e-4
        assert plain.inference_status == "influence_curve"
        for estimate in trimmed.estimates.values():
            assert estimate.inference == TRIMMED_RATIO_STATUS
            assert np.isfinite(estimate.plugin_std_error)
            with pytest.raises(CapabilityError, match="ratio_trim="):
                _ = estimate.std_error

    def test_a_trim_that_moves_no_ratio_keeps_the_interval_and_the_numbers(
        self, frame: pd.DataFrame
    ) -> None:
        """At a quantile equal to the largest entry no ratio moves, and the fit is the plain one."""
        quantile = 0.999
        for label, arm in (("always", 1), ("never", 0)):
            entries = _hand_entries(frame, arm)
            assert _type7(entries, quantile) == entries.max(), label
        plain = _fit(frame)
        kept = _fit(frame, ratio_trim=quantile)
        assert kept.inference_status == "influence_curve"
        for name, estimate in plain.estimates.items():
            assert kept[name].psi == estimate.psi
            assert kept[name].std_error == estimate.std_error
        for fit in kept.fits.values():
            assert fit.trimmed_share == 0.0

    def test_the_default_fit_carries_no_trim(self, frame: pd.DataFrame) -> None:
        result = _fit(frame)
        for fit in result.fits.values():
            assert fit.ratio_trim is None
            assert fit.trim_threshold is None
            assert fit.trimmed_share == 0.0


def _mutant_cumulative(
    data: Any, plans: Any, mechanism: Any, quantile: float
) -> dict[str, RatioTrim]:
    """Mutation: trim the cumulative ratio instead of the per-node ratios."""
    trims = trim_module.trim_ratios(data, plans, mechanism, quantile)
    out = {}
    for label, trim in trims.items():
        cumulative = np.cumprod(trim.entries, axis=1)
        threshold = float(np.quantile(cumulative.ravel(), quantile))
        out[label] = RatioTrim(
            quantile, threshold, trim.entries, np.minimum(cumulative, threshold), trim.trimmed
        )
    return out


def _mutant_positive_only(
    data: Any, plans: Any, mechanism: Any, quantile: float
) -> dict[str, RatioTrim]:
    """Mutation: compute the quantile over the positive entries only."""
    trims = trim_module.trim_ratios(data, plans, mechanism, quantile)
    out = {}
    for label, trim in trims.items():
        threshold = float(np.quantile(trim.entries[trim.entries > 0.0], quantile))
        kept = np.minimum(trim.entries, threshold)
        out[label] = RatioTrim(
            quantile, threshold, trim.entries, np.cumprod(kept, axis=1), trim.entries > threshold
        )
    return out


def _mutant_without_censoring(data: Any, plan: Any, mechanism: Any) -> np.ndarray:
    """Mutation: leave the retention indicator and probability out of each entry."""
    entries = node_ratio_entries(data, plan, mechanism)
    density = plan.intervention_density(data)
    columns = []
    for time in range(1, data.n_times + 1):
        observed = mechanism.treatment_observed[time - 1]
        codes = np.nan_to_num(data.treatment[:, time - 1], nan=0.0).astype(np.int64)
        treatment = observed[np.arange(data.n), codes]
        columns.append(np.where(treatment > 0.0, density[:, time - 1] / treatment, 0.0))
    del entries
    return np.column_stack(columns)


def _mutant_followed_through(data: Any, plan: Any, mechanism: Any) -> np.ndarray:
    """Mutation: zero an entry unless the unit followed the plan at every node through ``t``."""
    entries = node_ratio_entries(data, plan, mechanism)
    support = np.logical_and.accumulate(entries > 0.0, axis=1)
    return np.where(support, entries, 0.0)


class TestMutationControls:
    @pytest.mark.parametrize(
        ("target", "mutant"),
        [
            ("estimator.trim_ratios", _mutant_cumulative),
            ("estimator.trim_ratios", _mutant_positive_only),
            ("ratio_trim.node_ratio_entries", _mutant_without_censoring),
            ("ratio_trim.node_ratio_entries", _mutant_followed_through),
        ],
        ids=["cumulative", "positive_only", "without_censoring", "followed_through"],
    )
    def test_each_mutation_fails_the_hand_check(
        self,
        frame: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        target: str,
        mutant: Callable[..., Any],
    ) -> None:
        module, name = target.split(".")
        owner = longitudinal_estimator if module == "estimator" else trim_module
        monkeypatch.setattr(owner, name, mutant)
        with pytest.raises(AssertionError):
            _check_trim(_fit(frame, ratio_trim=BINDING), frame, BINDING)

    def test_a_per_fold_trim_fails_the_hand_check(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: trim each outer fold's out-of-fold rows at its own quantile."""
        _check_trim(_fit(frame, ratio_trim=PER_FOLD, n_folds=3), frame, PER_FOLD)
        original = trim_module.trim_ratios
        result_folds: list[np.ndarray] = []

        def per_fold(data: Any, plans: Any, mechanism: Any, quantile: float) -> Any:
            trims = original(data, plans, mechanism, quantile)
            ids = result_folds[0]
            out = {}
            for label, trim in trims.items():
                kept = np.array(trim.entries, copy=True)
                for fold in np.unique(ids):
                    rows = ids == fold
                    threshold = float(np.quantile(trim.entries[rows].ravel(), quantile))
                    kept[rows] = np.minimum(kept[rows], threshold)
                out[label] = RatioTrim(
                    quantile,
                    trim.threshold,
                    trim.entries,
                    np.cumprod(kept, axis=1),
                    trim.entries > kept,
                )
            return out

        estimator = _ltmle(ratio_trim=PER_FOLD, n_folds=3)
        draw = estimator._folds

        def recording(prepared: Any, seed: int | None = None) -> Any:
            folds = draw(prepared, seed)
            result_folds.append(np.asarray(folds.assignment))
            return folds

        monkeypatch.setattr(estimator, "_folds", recording)
        monkeypatch.setattr(longitudinal_estimator, "trim_ratios", per_fold)
        result = estimator.fit(
            frame,
            **law.SMART_COLUMNS,
            censoring=["C1", "C2"],
            treatment_probabilities=law.SMART_TREATMENT,
            censoring_probabilities=law.SMART_CENSORING,
        )
        with pytest.raises(AssertionError):
            _check_trim(result, frame, PER_FOLD)


class TestTheStatus:
    def test_the_record_holds_the_pinned_texts(self) -> None:
        record = STATUS_RECORDS[TRIMMED_RATIO_STATUS]
        assert record.reason.startswith(
            "A longitudinal fit that trims its per-node density ratios at a declared quantile "
            "(ratio_trim=) reports no confidence interval, no p-value and no standard error."
        )
        assert "it is the standard error lmtp reports" in record.reason
        assert record.diagnostic_noun == "trimmed-ratio plug-in diagnostic"
        assert record.reopened_by == "F30"
        assert record.bootstrap_note.startswith("a diagnostic; the bootstrap repeats the trim")
        assert "F30 in the roadmap" in record.assessment_note

    def test_a_truncation_replay_keeps_the_trim(self, frame: pd.DataFrame) -> None:
        """The replay reuses the trimmed mechanism: each grid point equals a trimmed refit."""
        seeded = LogisticRegression(max_iter=1000, random_state=0)
        result = _fit(frame, ratio_trim=BINDING, outcome_learner=seeded)
        bounds = (0.001, 1.0)
        curve = result.diagnostics.truncation_curve([bounds])
        refit = _fit(frame, ratio_trim=BINDING, outcome_learner=seeded, g_bounds=bounds)
        values = dict(zip(curve["estimand"], curve["psi"], strict=True))
        for name, estimate in refit.estimates.items():
            assert values[name] == pytest.approx(estimate.psi, rel=1e-12)


class TestRefusals:
    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, True, "0.9"])
    def test_a_quantile_outside_the_open_unit_interval_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            LTMLE(REGIMENS, ratio_trim=value)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            Targeting(ratio_trim=value)

    def test_a_threshold_of_zero_is_refused(self, frame: pd.DataFrame) -> None:
        """At a quantile below the share of zero entries every ratio would be zero."""
        with pytest.raises(DataError, match="trims the per-node density ratios of regimen"):
            _fit(frame, ratio_trim=0.05)

    def test_a_point_treatment_fit_refuses_the_trim(self) -> None:
        method = TMLEMethod(targeting=Targeting(ratio_trim=0.9))
        with pytest.raises(MethodConfigurationError, match="Declare the point treatment"):
            method.estimator_kwargs()
        frame, _ = make_linear_ate(n=120, seed=19)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
        )
        with pytest.raises(MethodConfigurationError, match="Declare the point treatment"):
            study.identify(ATE()).estimate(method=method)

    def test_the_longitudinal_translation_passes_the_trim(self) -> None:
        method = TMLEMethod(targeting=Targeting(ratio_trim=0.9))
        assert method.estimator_kwargs(longitudinal=True)["ratio_trim"] == 0.9
        assert TMLEMethod().with_overrides(ratio_trim=0.8).targeting.ratio_trim == 0.8


def test_the_study_path_reaches_the_engine(frame: pd.DataFrame) -> None:
    design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["L0"],
        time_varying=[[], ["L1"]],
        censoring=["C1", "C2"],
        treatment_probabilities=law.SMART_TREATMENT,
        censoring_probabilities=law.SMART_CENSORING,
    )
    study = CausalStudy(frame, design=design)
    result = study.identify(RegimeMean({"always": 1})).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        cross_fit=False,
        ratio_trim=BINDING,
    )
    (estimate,) = result.estimates.values()
    assert estimate.inference == TRIMMED_RATIO_STATUS


def test_the_support_frame_reports_the_trimmed_share_per_node(frame: pd.DataFrame) -> None:
    plain = _fit(frame).diagnostics.support().to_frame()
    assert "share_trimmed" not in plain.columns
    result = _fit(frame, ratio_trim=BINDING)
    support = result.diagnostics.support().to_frame()
    for row in support.itertuples():
        fit = result.fits[row.regimen]
        step = fit.steps[row.time - 1]
        trim = fit.ratio_trim
        assert trim is not None
        expected = float(np.mean(trim.trimmed[step.trained_on, row.time - 1]))
        assert row.share_trimmed == pytest.approx(expected, rel=1e-15)
    assert support["share_trimmed"].max() > 0.0


def _sdr(**settings: Any) -> SDR:
    base: dict[str, Any] = {
        "reference": "never",
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
        "simultaneous": False,
    }
    base.update(settings)
    return SDR(REGIMENS, **base)


def _fit_sdr(frame: pd.DataFrame, **settings: Any) -> LongitudinalResult:
    return _sdr(**settings).fit(
        frame,
        **law.SMART_COLUMNS,
        censoring=["C1", "C2"],
        treatment_probabilities=law.SMART_TREATMENT,
        censoring_probabilities=law.SMART_CENSORING,
    )


def _recorded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[np.ndarray, int, int, tuple[float, float], list[np.ndarray]]]:
    """Record each ``partial_weights`` call of the SDR recursion: factors, span, weights."""
    calls: list[tuple[np.ndarray, int, int, tuple[float, float], list[np.ndarray]]] = []
    original = sdr_module.partial_weights

    def recording(
        numerator: Any, factors: Any, masks: Any, start: int, stop: int, bounds: Any
    ) -> Any:
        out = original(numerator, factors, masks, start, stop, bounds)
        calls.append((np.asarray(numerator), start, stop, bounds, out[0]))
        return out

    monkeypatch.setattr(sdr_module, "partial_weights", recording)
    return calls


def _check_sdr_weights(
    calls: list[tuple[np.ndarray, int, int, tuple[float, float], list[np.ndarray]]],
    frame: pd.DataFrame,
    quantile: float,
) -> None:
    """Each weight is ``lmtp``'s ``prod_{k=t}^{s} min(r_k, Q_q)`` through the bounds.

    The plan is read off the numerator: ``always`` has density one where ``A_t = 1``. The
    hand entries come from the columns, so a weight that ignores the trim, or a threshold
    that is not the pooled type-7 quantile, fails here. A unit that left the plan before
    ``t`` has a zero weight from ``t``, as the shipped recursion masks it.
    """
    assert calls
    a1 = frame["A1"].to_numpy(dtype=float)
    for numerator, start, stop, (lower, upper), weights in calls:
        arm = 1 if np.array_equal(numerator[:, 0] > 0.0, a1 == 1.0) else 0
        entries = _hand_entries(frame, arm)
        trimmed = np.minimum(entries, _type7(entries, quantile))
        running = np.ones(len(frame))
        for node, weight in zip(range(start, stop + 1), weights, strict=True):
            running = running * trimmed[:, node - 1]
            # The recursion weights the units that follow the plan from node one through
            # ``node``. A unit that rejoins the plan has a positive entry and a zero weight.
            following = np.prod(entries[:, :node], axis=1) > 0.0
            expected = 1.0 / bound(1.0 / running[following], lower, upper)
            np.testing.assert_allclose(weight[following], expected, rtol=1e-12)
            assert np.all(weight[~following] == 0.0)


class TestSDR:
    """``SDR(ratio_trim=q)``: each partial weight is a product of trimmed per-node ratios."""

    def test_the_partial_weights_equal_the_lmtp_weights(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _recorded_weights(monkeypatch)
        _fit_sdr(frame, ratio_trim=BINDING)
        _check_sdr_weights(calls, frame, BINDING)

    def test_every_cross_fitted_fold_trims_at_the_pooled_threshold(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At ``PER_FOLD`` a threshold from one fold's rows differs from the pooled one."""
        calls = _recorded_weights(monkeypatch)
        _fit_sdr(frame, ratio_trim=PER_FOLD, n_folds=3)
        assert len(calls) == 3 * len(REGIMENS) * 2
        _check_sdr_weights(calls, frame, PER_FOLD)

    def test_a_weight_that_ignores_the_trim_fails_the_hand_check(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the recursion divides by the untrimmed factors."""
        calls = _recorded_weights(monkeypatch)
        monkeypatch.setattr(sdr_module, "trimmed_factors", lambda numerator, factors, _: factors)
        _fit_sdr(frame, ratio_trim=BINDING)
        with pytest.raises(AssertionError):
            _check_sdr_weights(calls, frame, BINDING)

    def test_a_per_fold_threshold_fails_the_hand_check(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: each fold trims at the quantile of its own training rows' ratios."""
        calls = _recorded_weights(monkeypatch)
        original = sdr_module.trimmed_factors

        def per_fold(numerator: Any, factors: Any, threshold: float) -> Any:
            positive = (numerator > 0.0) & (factors > 0.0)
            ratio = np.where(positive, numerator / np.where(positive, factors, 1.0), 0.0)
            # The SMART factors are fold-free, so the rows differ only by the fold: read the
            # quantile over a fixed two-thirds of the rows as one fold's training rows.
            rows = np.arange(ratio.shape[0]) % 3 != 0
            return original(numerator, factors, float(np.quantile(ratio[rows], PER_FOLD)))

        monkeypatch.setattr(sdr_module, "trimmed_factors", per_fold)
        _fit_sdr(frame, ratio_trim=PER_FOLD, n_folds=3)
        with pytest.raises(AssertionError):
            _check_sdr_weights(calls, frame, PER_FOLD)

    def test_a_binding_trim_moves_the_estimate_and_withholds_the_interval(
        self, frame: pd.DataFrame
    ) -> None:
        plain = _fit_sdr(frame)
        trimmed = _fit_sdr(frame, ratio_trim=BINDING)
        truth = law.smart_truth(1, 1)
        assert abs(plain["ey_regimen[always]"].psi - truth) <= EXACT
        assert abs(trimmed["ey_regimen[always]"].psi - truth) > 1e-4
        assert plain.inference_status == "influence_curve"
        for estimate in trimmed.estimates.values():
            assert estimate.inference == TRIMMED_RATIO_STATUS
        assert trimmed.fits["always"].trim_threshold == pytest.approx(
            _type7(_hand_entries(frame, 1), BINDING), rel=1e-15
        )

    def test_a_trim_that_moves_no_ratio_keeps_the_numbers(self, frame: pd.DataFrame) -> None:
        plain = _fit_sdr(frame)
        kept = _fit_sdr(frame, ratio_trim=0.999)
        assert kept.inference_status == "influence_curve"
        for name, estimate in plain.estimates.items():
            assert kept[name].psi == estimate.psi
            assert kept[name].std_error == estimate.std_error


class TestIPSIRefusesTheTrim:
    def test_the_engine_refuses_it(self) -> None:
        with pytest.raises(CapabilityError, match="IPSI does not apply ratio_trim="):
            IPSI([Incremental(2.0)], ratio_trim=0.9)
        assert IPSI([Incremental(2.0)], ratio_trim=None).ratio_trim is None

    def test_the_method_refuses_it(self) -> None:
        method = IPSIMethod(targeting=Targeting(ratio_trim=0.9))
        with pytest.raises(MethodConfigurationError, match="IPSI does not apply ratio_trim="):
            method.estimator_kwargs(longitudinal=True)
        assert IPSIMethod().estimator_kwargs(longitudinal=True)["ratio_trim"] is None
