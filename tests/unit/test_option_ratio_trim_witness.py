"""Exact-law witnesses of ``option-ratio-trim`` (X76 part (e), Fix-8 plan section 4).

``ratio_trim=q`` replaces every per-node density ratio above the quantile ``Q_q`` of all
``n x T`` entries by ``Q_q``.  Targeting solves the trimmed score, so the limit of a trimmed
fit is the truth plus ``sum_t P_0[(R~_t - R_t)(Q*_{t+1} - Q*_t)]``, which vanishes when every
outcome regression is correct and not otherwise.  The witnesses fit the quadrature form of the
study law ``LE`` with probability weights, at the law's own quantile
(:func:`tests.studies._option_trim_law.exact_limit`), so every fit returns its population limit:

* with correct outcome regressions the trimmed LTMLE and SDR fits return the truth to ``1e-8``;
* with the W-blind outcome regression the trimmed fits miss it by more than ``0.01``, and the
  LTMLE miss equals the remainder computed from the true ratios at ``1e-8``;
* untrimmed, the W-blind fits return the truth to ``1e-8``: the term is the trim's, and the
  untrimmed fit is the comparison that keeps double robustness.

The trim binds: the clever covariate of the trimmed fit is the trimmed cumulative ratio, a
sixth of the positive entries lie above the threshold, and on a sample the threshold is the
type-7 quantile of every entry, zeros included.  The mutation controls drop the trim from the
clever covariate, carry a trim at node 1 to node 2 only where node 2 was trimmed too, and take
the quantile of the positive entries; each fails a witness.

The support tests read the rows: a trimmed fit with ``LinearRegression`` outcome regressions
lands on its ``G56`` row, reads supported for the point and withholds its interval.  The rows pin
the outcome learners to ``LinearRegression``, so a trimmed fit with another outcome learner
matches no row, and the untrimmed fit stays on ``G01``.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest

from cleverly._inference_status import TRIMMED_RATIO_STATUS
from cleverly.exceptions import CapabilityError
from tests.studies import _option_trim_law as law

estimator_module = importlib.import_module("cleverly.longitudinal.estimator")
trim_module = importlib.import_module("cleverly.longitudinal.ratio_trim")

EXACT = 1e-8
FLOOR = 0.01


@functools.cache
def cached_fit(name: str) -> tuple[Any, Any]:
    return law.limit_fit(name)


def bias(name: str, *, cache: bool = True) -> float:
    _, result = cached_fit(name) if cache else law.limit_fit(name)
    return float(result.estimates[law.ESTIMAND].psi) - law.TRUTH


def check_remainder_witness() -> None:
    """The trimmed W-blind LTMLE limit misses by the remainder, and by more than the floor."""
    frame, result = law.limit_fit("ltmle_in_sample_wrong_trim95")
    miss = float(result.estimates[law.ESTIMAND].psi) - law.TRUTH
    assert miss > FLOOR
    assert miss == pytest.approx(law.trim_remainder(frame, result, law.TRIM), abs=EXACT)


def mutated(module: Any, owner: Any, name: str, old: str, new: str) -> Any:
    """``owner.name`` recompiled with ``old`` replaced by ``new`` once in its source."""
    source = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    assert source.count(old) == 1, f"{name} no longer holds {old!r}"
    namespace: dict[str, Any] = {}
    exec(compile(source.replace(old, new), f"<mutated {name}>", "exec"), vars(module), namespace)
    return namespace[name]


class TestTheLaw:
    def test_the_weights_are_a_probability_law(self) -> None:
        frame = law.quadrature_frame()
        assert frame["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        assert (frame["weight"] > 0).all()

    def test_the_truth_is_the_weighted_g_formula(self) -> None:
        frame = law.quadrature_frame()
        w = frame["W"].to_numpy()
        weight = frame["weight"].to_numpy()
        q1 = 0.2 + 0.6 * w + 0.15 * law.p_covariate(w, 1.0)
        assert np.sum(weight * q1) == pytest.approx(law.TRUTH, abs=1e-12)
        assert law.truth() == pytest.approx(law.TRUTH, abs=1e-15)

    def test_the_population_quantile_is_the_weighted_entry_quantile(self) -> None:
        """The quadrature law's weighted entries put ``1 - q`` of their mass above ``Q_q``."""
        for quantile in (law.TRIM, law.REPORTED_TRIM):
            threshold = law.population_threshold(quantile)
            frame = law.quadrature_frame(cuts=law.cut_points(threshold))
            ratios = law.true_ratios(frame)
            weight = np.repeat(frame["weight"].to_numpy()[:, None], 2, axis=1) / 2.0
            assert weight[ratios > threshold].sum() == pytest.approx(1.0 - quantile, abs=1e-10)


class TestTheTrimWitness:
    @pytest.mark.parametrize("name", ["ltmle_in_sample_right_trim95", "sdr_in_sample_right_trim95"])
    def test_a_correct_outcome_regression_returns_the_truth_under_the_trim(self, name: str) -> None:
        assert abs(bias(name)) < EXACT

    def test_the_trimmed_w_blind_fit_misses_by_the_remainder(self) -> None:
        check_remainder_witness()

    def test_the_trimmed_w_blind_sdr_fit_misses(self) -> None:
        assert bias("sdr_in_sample_wrong_trim95") > FLOOR

    @pytest.mark.parametrize(
        "name", ["ltmle_in_sample_wrong_untrimmed", "sdr_in_sample_wrong_untrimmed"]
    )
    def test_the_untrimmed_w_blind_fit_returns_the_truth(self, name: str) -> None:
        """The comparison: without the trim double robustness removes the error."""
        assert abs(bias(name)) < EXACT

    def test_the_remainder_vanishes_without_the_trim(self) -> None:
        frame, result = cached_fit("ltmle_in_sample_wrong_untrimmed")
        assert abs(law.trim_remainder(frame, result, None)) < 1e-15

    def test_the_trim_binds(self) -> None:
        assert law.trimmed_share(law.TRIM) > 0.1
        frame, result = cached_fit("ltmle_in_sample_right_trim95")
        ratios = law.true_ratios(frame)
        trimmed = np.cumprod(np.minimum(ratios, law.population_threshold(law.TRIM)), axis=1)
        moved = 0
        for step in result.fits[law.REGIMEN].steps:
            follow = step.at_risk & (trimmed[:, step.time - 1] > 0)
            assert np.asarray(step.clever)[follow] == pytest.approx(
                trimmed[follow, step.time - 1], rel=1e-9
            )
            moved += int(
                np.sum(
                    follow & (trimmed[:, step.time - 1] < np.cumprod(ratios, 1)[:, step.time - 1])
                )
            )
        assert moved > 0

    def test_mutation_dropping_the_trim_from_the_clever_covariate_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        check_remainder_witness()
        untrimmed = mutated(
            trim_module,
            trim_module.RatioTrim,
            "implied_cumulative",
            "return np.where(touched & positive, numerator / safe, raw)",
            "return raw",
        )
        monkeypatch.setattr(trim_module.RatioTrim, "implied_cumulative", untrimmed)
        with pytest.raises(AssertionError):
            check_remainder_witness()

    def test_mutation_forgetting_an_earlier_trim_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Node 2's clever covariate must carry a node-1 trim even where node 2 was not trimmed."""
        check_remainder_witness()
        forgetful = mutated(
            trim_module,
            trim_module.RatioTrim,
            "implied_cumulative",
            "touched = np.logical_or.accumulate(self.trimmed, axis=1)",
            "touched = self.trimmed",
        )
        monkeypatch.setattr(trim_module.RatioTrim, "implied_cumulative", forgetful)
        with pytest.raises(AssertionError):
            check_remainder_witness()


def check_sample_threshold() -> None:
    """On a sample the threshold is the type-7 quantile of every entry, zeros included."""
    result = law.fit(law.draw(2_000, 0), "ltmle_in_sample_right_trim95")
    trim = result.fits[law.REGIMEN].ratio_trim
    entries = np.asarray(trim.entries)
    assert np.mean(entries == 0.0) > 0.3
    assert trim.threshold == np.quantile(entries.ravel(), law.TRIM, method="linear")
    assert np.array_equal(trim.trimmed, entries > trim.threshold)
    kept = np.cumprod(np.minimum(entries, trim.threshold), axis=1)
    assert np.allclose(np.asarray(trim.cumulative), kept, rtol=1e-12, atol=0.0)


class TestTheSampleTrim:
    def test_the_threshold_reads_every_entry(self) -> None:
        check_sample_threshold()

    def test_mutation_leaving_the_running_product_untrimmed_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        untrimmed = mutated(
            trim_module,
            trim_module,
            "trim_ratios",
            "cumulative=np.cumprod(kept, axis=1, dtype=np.float64)",
            "cumulative=np.cumprod(entries, axis=1, dtype=np.float64)",
        )
        monkeypatch.setattr(estimator_module, "trim_ratios", untrimmed)
        with pytest.raises(AssertionError):
            check_sample_threshold()

    def test_mutation_taking_the_quantile_of_positive_entries_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        positive = mutated(
            trim_module,
            trim_module,
            "trim_ratios",
            "np.quantile(entries.ravel(), quantile",
            "np.quantile(entries[entries > 0.0], quantile",
        )
        monkeypatch.setattr(estimator_module, "trim_ratios", positive)
        with pytest.raises(AssertionError):
            check_sample_threshold()


class TestTheSupportKey:
    @pytest.mark.parametrize(
        ("name", "row"),
        [
            ("ltmle_in_sample_right_trim95", "G56.ltmle.in_sample"),
            ("ltmle_stacked_right_trim95", "G56.ltmle.cross_fitted"),
            ("sdr_in_sample_right_trim95", "G56.sdr.in_sample"),
            ("sdr_stacked_right_trim95", "G56.sdr.cross_fitted"),
        ],
    )
    def test_a_trimmed_linear_fit_reads_its_g56_row(self, name: str, row: str) -> None:
        """The point estimate is supported on ``G56``, and the interval stays withheld."""
        estimate = law.fit(law.draw(400, 0), name).estimates[law.ESTIMAND]
        assert (estimate.support.label, estimate.support.row) == ("supported", row)
        assert estimate.inference == TRIMMED_RATIO_STATUS
        assert any("F30" in limit for limit in estimate.support.limits)

    @pytest.mark.parametrize("name", ["ltmle_in_sample_right_trim95", "sdr_in_sample_right_trim95"])
    def test_a_trimmed_fit_with_another_outcome_learner_matches_no_row(self, name: str) -> None:
        """``G56`` pins ``LinearRegression``: a boosted outcome regression reads experimental."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        boosted = HistGradientBoostingRegressor(max_iter=20, random_state=0)
        estimate = law.fit(
            law.draw(400, 0), name, outcome_learner=boosted, pseudo_learner=boosted
        ).estimates[law.ESTIMAND]
        assert (estimate.support.label, estimate.support.row) == ("experimental", None)
        assert "ratio_trim=True" in estimate.support.reason

    def test_the_study_s_wrong_outcome_learner_matches_no_row(self) -> None:
        """The W-blind control's learner class is not ``LinearRegression``, so it leaves ``G56``."""
        support = (
            law.fit(law.draw(400, 0), "ltmle_in_sample_wrong_trim95")
            .estimates[law.ESTIMAND]
            .support
        )
        assert (support.label, support.row) == ("experimental", None)

    def test_the_untrimmed_fit_is_the_default_row(self) -> None:
        frame = law.draw(400, 0)
        support = law.fit(frame, "ltmle_in_sample_right_untrimmed").estimates[law.ESTIMAND].support
        assert support.row == "G01"

    def test_a_trimmed_fit_withholds_its_interval(self) -> None:
        estimate = law.fit(law.draw(400, 0), "ltmle_in_sample_right_trim95").estimates[law.ESTIMAND]
        assert estimate.inference == TRIMMED_RATIO_STATUS
        assert np.isfinite(estimate.plugin_std_error)
        with pytest.raises(CapabilityError, match="ratio_trim="):
            _ = estimate.std_error
