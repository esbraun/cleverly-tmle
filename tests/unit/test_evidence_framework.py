"""The evidence framework's own instruments, checked without any committed artefact.

The tests that matter here are the last two classes: they pin the property the whole package
is built around -- a verdict must not get harder to earn as the Monte Carlo evidence grows --
and they do it by exhibiting the rule that fails it beside the rule that does not.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from tests.studies.evidence.claims import matches
from tests.studies.evidence.inference import (
    Interval,
    bootstrap,
    clopper_pearson,
    coverage_for_se_ratio,
    lower_bound,
    percentile_interval,
    se_ratio_for_coverage,
    standardized_bias_verdict,
    student_interval,
    upper_bound,
)
from tests.studies.evidence.pairing import paired_wide
from tests.studies.evidence.properties import rate, require_complete
from tests.studies.evidence.registry import Margins

CONFIDENCE = 0.99


class TestInterval:
    def test_within_and_outside_are_not_each_others_negation(self) -> None:
        interval = Interval(-0.1, 0.4)
        assert not interval.within(-0.25, 0.25)
        assert not interval.outside(-0.25, 0.25)

    def test_resolution_is_the_tightest_margin_the_interval_could_have_supported(self) -> None:
        assert Interval(0.85, 1.18).resolution(1.0) == pytest.approx(0.18)


class TestExactCoverageInterval:
    @pytest.mark.parametrize(("successes", "trials"), [(380, 400), (1500, 1600), (17, 20)])
    def test_the_endpoints_solve_the_binomial_tail_equations(
        self, successes: int, trials: int
    ) -> None:
        """The defining identities, not a comparison with another implementation.

        The Clopper--Pearson lower endpoint is the ``p`` at which observing at least this
        many successes has probability ``alpha/2``; the upper endpoint is its mirror.
        """
        interval = clopper_pearson(successes, trials, confidence_level=CONFIDENCE)
        tail = (1.0 - CONFIDENCE) / 2.0
        assert binom.sf(successes - 1, trials, interval.low) == pytest.approx(tail, rel=1e-9)
        assert binom.cdf(successes, trials, interval.high) == pytest.approx(tail, rel=1e-9)

    def test_a_degenerate_column_is_reported_as_degenerate(self) -> None:
        assert clopper_pearson(0, 400, confidence_level=CONFIDENCE).low == 0.0
        assert clopper_pearson(400, 400, confidence_level=CONFIDENCE).high == 1.0


class TestStandardErrorCoverageCorrespondence:
    def test_a_correctly_scaled_standard_error_gives_nominal_coverage(self) -> None:
        assert coverage_for_se_ratio(1.0, alpha=0.05) == pytest.approx(0.95)

    def test_the_map_inverts(self) -> None:
        for coverage in (0.88, 0.90, 0.93, 0.95):
            ratio = se_ratio_for_coverage(coverage, alpha=0.05)
            assert coverage_for_se_ratio(ratio, alpha=0.05) == pytest.approx(coverage)

    def test_a_ten_percent_understatement_costs_the_coverage_the_docs_claim(self) -> None:
        assert coverage_for_se_ratio(0.90, alpha=0.05) == pytest.approx(0.9223, abs=5e-5)

    def test_margins_refuse_a_sanity_band_that_would_bind_before_the_coverage_floor(self) -> None:
        with pytest.raises(ValueError, match="tighter than"):
            Margins(coverage_floor=0.90, se_ratio_sanity=(0.95, 1.20))


class TestBootstrapEngine:
    def _arrays(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        values = rng.normal(size=64)
        return {"left": values, "right": values.copy()}

    def test_columns_are_resampled_on_one_index_draw(self) -> None:
        """Two copies of one column must stay identical under resampling.

        This is the pairing the cross-implementation comparison rests on: resample each
        column separately and every paired statistic silently compares replication *i* of one
        implementation with replication *j* of the other.
        """
        samples = bootstrap(
            self._arrays(),
            {"difference": lambda draw: np.abs(draw["left"] - draw["right"]).max(axis=1)},
            replicates=200,
            seed=3,
        )["difference"]
        assert samples.max() == 0.0

    def test_the_same_seed_gives_the_same_interval(self) -> None:
        statistic = {"mean": lambda draw: draw["left"].mean(axis=1)}
        first = bootstrap(self._arrays(), statistic, replicates=500, seed=11)["mean"]
        second = bootstrap(self._arrays(), statistic, replicates=500, seed=11)["mean"]
        assert np.array_equal(first, second)

    def test_unequal_columns_are_refused_rather_than_broadcast(self) -> None:
        with pytest.raises(ValueError, match="different lengths"):
            bootstrap(
                {"a": np.zeros(4), "b": np.zeros(5)},
                {"mean": lambda draw: draw["a"].mean(axis=1)},
                replicates=10,
                seed=0,
            )

    def test_one_sided_bounds_sit_where_the_percentiles_do(self) -> None:
        samples = np.arange(1000, dtype=float)
        assert upper_bound(samples, confidence_level=0.99) == pytest.approx(989.01)
        assert lower_bound(samples, confidence_level=0.99) == pytest.approx(9.99)
        interval = percentile_interval(samples, confidence_level=0.99)
        assert (interval.low, interval.high) == pytest.approx((4.995, 994.005))


class TestPairing:
    def _rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "replicate": [0, 0, 1, 1, 2, 2],
                "implementation": ["a", "b"] * 3,
                "estimate": [1.0, 1.1, 2.0, 2.2, 3.0, 3.3],
                "std_error": [0.1, 0.1, 0.2, math.nan, 0.3, 0.3],
            }
        )

    def test_a_gap_in_one_column_drops_the_replication_from_all_of_them(self) -> None:
        paired = paired_wide(
            self._rows(),
            ("estimate", "std_error"),
            implementations=("a", "b"),
            tolerated_drops=1,
        )
        assert paired.dropped == 1
        assert paired.column("estimate", "a").tolist() == [1.0, 3.0]
        assert paired.column("estimate", "b").tolist() == [1.1, 3.3]

    def test_an_undeclared_drop_is_an_error_rather_than_a_shorter_array(self) -> None:
        with pytest.raises(ValueError, match="incomplete"):
            paired_wide(self._rows(), ("estimate", "std_error"), implementations=("a", "b"))

    def test_a_missing_implementation_is_named(self) -> None:
        with pytest.raises(KeyError, match="absent"):
            paired_wide(self._rows(), ("estimate",), implementations=("a", "c"))


def _biased_sample(bias: float, replicates: int, *, seed: int = 5) -> np.ndarray:
    """Errors of an estimator whose bias is a fixed fraction of its sampling spread."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=replicates)
    return bias + (draws - draws.mean()) / draws.std(ddof=1)


class TestVerdictsDoNotPunishEvidence:
    """The property the framework exists to hold: more replications never lose a verdict.

    Each test runs the same estimator -- a fixed bias of a fifth of a standard deviation, or
    a fixed 93% coverage -- at four Monte Carlo budgets, and asserts the bounded-margin rule
    is stable while exhibiting the significance-shaped rule that flips.  The flipping rule is
    not a strawman: both were in the study this framework replaced.
    """

    BUDGETS = (200, 800, 3200, 12800)

    @staticmethod
    def _never_lost(flags: list[bool]) -> bool:
        """A verdict, once earned, is not taken away by adding replications."""
        return all(later >= earlier for earlier, later in itertools.pairwise(flags))

    def test_standardized_bias_equivalence_survives_more_replications(self) -> None:
        verdicts = [
            standardized_bias_verdict(
                _biased_sample(0.20, replicates), margin=0.25, confidence_level=CONFIDENCE
            ).equivalent
            for replicates in self.BUDGETS
        ]
        assert self._never_lost(verdicts), verdicts
        assert verdicts[0] is False, "the smallest budget should not already be able to say this"
        assert verdicts[-1] is True, verdicts

    def test_the_significance_rule_it_replaced_flips_the_other_way(self) -> None:
        """``|bias| <= 3.5 * bias_se`` on the same estimator, for contrast.

        Not a strawman: this was the ``truth_compatible`` rule in the study that this
        framework replaced, and it is why quadrupling the replication count would have turned
        the published suite red without a line of estimator code changing.
        """
        flags = []
        for replicates in self.BUDGETS:
            errors = _biased_sample(0.20, replicates)
            bias_se = float(np.std(errors, ddof=1) / math.sqrt(replicates))
            flags.append(abs(float(np.mean(errors))) <= 3.5 * bias_se)
        assert self._never_lost(list(reversed(flags))), flags
        assert flags[0] is True and flags[-1] is False, flags

    def test_a_real_deficiency_is_established_rather_than_tolerated(self) -> None:
        """The margin cuts both ways: a bias beyond it is *shown* to be beyond it."""
        verdicts = [
            standardized_bias_verdict(
                _biased_sample(0.35, replicates), margin=0.25, confidence_level=CONFIDENCE
            ).discriminated
            for replicates in self.BUDGETS
        ]
        assert self._never_lost(verdicts), verdicts
        assert verdicts[0] is False and verdicts[-1] is True, verdicts
        assert not any(
            standardized_bias_verdict(
                _biased_sample(0.35, replicates), margin=0.25, confidence_level=CONFIDENCE
            ).equivalent
            for replicates in self.BUDGETS
        )

    @staticmethod
    def _covered(rate_: float, replicates: int) -> int:
        return round(rate_ * replicates)

    def test_a_coverage_floor_survives_more_replications(self) -> None:
        verdicts = [
            clopper_pearson(
                self._covered(0.93, replicates), replicates, confidence_level=CONFIDENCE
            ).low
            >= 0.90
            for replicates in self.BUDGETS
        ]
        assert self._never_lost(verdicts), verdicts
        assert verdicts[-1] is True, verdicts

    def test_requiring_the_interval_to_contain_nominal_coverage_flips(self) -> None:
        """The clause this replaced, on an estimator whose true coverage is 93%."""
        flags = [
            clopper_pearson(
                self._covered(0.93, replicates), replicates, confidence_level=CONFIDENCE
            ).contains(0.95)
            for replicates in self.BUDGETS
        ]
        assert self._never_lost(list(reversed(flags))), flags
        assert flags[0] is True and flags[-1] is False, flags


class TestRateEstimator:
    def _rows(self, exponent: float, *, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        frames = []
        for size in (500, 2000, 8000):
            spread = size**exponent
            estimates = rng.normal(scale=spread, size=400)
            frames.append(
                pd.DataFrame(
                    {
                        "property": "rate",
                        "cell": f"n_{size}",
                        "n": size,
                        "truth": 0.0,
                        "estimate": estimates,
                        "std_error": np.full(400, spread * 1.96 / norm.ppf(0.975)),
                        "covered": 1,
                        "rejected": 0,
                        "requested_replicates": 400,
                        "failed_replicates": 0,
                        "replicate": np.arange(400),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_a_root_n_sampling_distribution_is_recognised(self) -> None:
        fitted = rate(
            self._rows(-0.5),
            property_name="rate",
            statistic="spread",
            bootstrap_replicates=2000,
            confidence_level=CONFIDENCE,
            seed=0,
        )
        assert fitted.consistent_with(-0.5)
        assert fitted.excludes(-0.25)

    def test_a_slower_contraction_is_rejected(self) -> None:
        """The check has to be able to fail, which the ratio of two mean standard errors could not."""
        fitted = rate(
            self._rows(-0.25),
            property_name="rate",
            statistic="spread",
            bootstrap_replicates=2000,
            confidence_level=CONFIDENCE,
            seed=0,
        )
        assert not fitted.consistent_with(-0.5)

    def test_two_sizes_are_refused_because_a_ratio_is_not_a_rate(self) -> None:
        rows = self._rows(-0.5)
        with pytest.raises(ValueError, match="at least three"):
            rate(
                rows.loc[rows["n"] != 2000],
                property_name="rate",
                statistic="spread",
                bootstrap_replicates=100,
                confidence_level=CONFIDENCE,
                seed=0,
            )


class TestReplicationAccounting:
    def test_a_lost_replication_stops_the_summary(self) -> None:
        rows = pd.DataFrame(
            {
                "property": "p",
                "cell": "c",
                "replicate": [0, 1],
                "n": 100,
                "requested_replicates": 3,
                "failed_replicates": 1,
                "truth": 0.0,
                "estimate": [0.1, -0.1],
                "std_error": [0.1, 0.1],
                "covered": [1, 1],
                "rejected": [0, 0],
            }
        )
        with pytest.raises(ValueError, match="fits failed"):
            require_complete(rows)


class TestPrintedValues:
    @pytest.mark.parametrize(
        ("printed", "expected"),
        [("0.0145", True), ("0.0146", False), ("0.015", True), ("0.01", True), ("0.02", False)],
    )
    def test_a_quoted_figure_must_be_the_rounding_of_the_computed_one(
        self, printed: str, expected: bool
    ) -> None:
        assert matches(printed, 0.0145269) is expected

    def test_percentages_and_thousands_separators_are_understood(self) -> None:
        assert matches("88.75%", 0.8875)
        assert matches("1,600", 1600.0)

    def test_a_student_interval_needs_more_than_one_value(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            student_interval(np.array([1.0]), confidence_level=CONFIDENCE)
