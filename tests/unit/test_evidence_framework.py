"""The evidence framework's own instruments, checked without any committed artefact.

The tests that matter here are the last two classes: they pin the property the whole package
is built around -- a verdict must not get harder to earn as the Monte Carlo evidence grows --
and they do it by exhibiting the rule that fails it beside the rule that does not.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from tests.studies.evidence.claims import matches
from tests.studies.evidence.document import render
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
from tests.studies.evidence.properties import Rate, rate, require_complete, summarize_cells
from tests.studies.evidence.property_verdicts import (
    ROOT_N_SLOPE_MARGIN,
    alternative_target_necessity_verdicts,
    calibration_verdicts,
)
from tests.studies.evidence.registry import Margins, StudyRecord, registered
from tests.studies.evidence.seeds import stream_seed

CONFIDENCE = 0.99


def test_calibration_verdicts_refuse_an_unknown_control_kind() -> None:
    summary = pd.DataFrame(
        {
            "property": ["interval_calibration"],
            "cell": ["static__unknown_control"],
            **{
                f"{prefix}_ci_{endpoint}": [1.0]
                for prefix in (
                    "se_ratio",
                    "efficiency_empirical",
                    "efficiency_reported",
                    "coverage",
                )
                for endpoint in ("lower", "upper")
            },
        }
    )

    with pytest.raises(ValueError, match="unknown calibration cell kind 'unknown_control'"):
        calibration_verdicts(summary, margins=Margins(), efficiency_band=(0.9, 1.1))


def test_an_alternative_target_control_must_recover_the_target_it_claims() -> None:
    rng = np.random.default_rng(4)
    noise = rng.normal(scale=0.1, size=200)
    rows = pd.DataFrame.from_records(
        [
            {
                "property": "weight_necessity",
                "cell": f"ate__{cell}",
                "role": role,
                "replicate": replicate,
                "n": 1000,
                "requested_replicates": len(noise),
                "failed_replicates": 0,
                "truth": 0.0,
                "estimate": float(value + (0.0 if role == "positive" else 1.0)),
                "std_error": 0.1,
                "covered": 1,
                "rejected": 0,
            }
            for replicate, value in enumerate(noise)
            for cell, role in (("weighted", "positive"), ("omitted_control", "control"))
        ]
    )
    margins = Margins()
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    summary["passed"] = False
    summary["property_passed"] = pd.Series([None] * len(summary), dtype=object, index=summary.index)
    record = StudyRecord(
        name="test",
        slug="test",
        artifacts=Path("."),
        document="test.md",
        anchor="test",
        scenarios={"test": ("ate",)},
        replicates=len(noise),
        n=1000,
        seed=0,
    )

    alternative_target_necessity_verdicts(
        summary,
        rows,
        record,
        family="weight_necessity",
        labels=("ate",),
        arms=("weighted", "omitted_control"),
        alternative_truths={"ate": 1.0},
        column="necessity_displacement",
        threshold=0.5,
    )
    assert summary["passed"].all()
    assert summary["property_passed"].all()

    changed = summary.copy()
    alternative_target_necessity_verdicts(
        changed,
        rows,
        record,
        family="weight_necessity",
        labels=("ate",),
        arms=("weighted", "omitted_control"),
        alternative_truths={"ate": 0.5},
        column="necessity_displacement",
        threshold=0.5,
    )
    control = changed.loc[changed["role"] == "control"]
    assert not bool(control["alternative_bias_equivalent"].iloc[0])
    assert not changed["property_passed"].any()


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

    def test_the_calibration_band_has_to_be_tighter_than_the_screen_it_sits_inside(self) -> None:
        """Otherwise the calibration cell restates the screen and catches nothing new."""
        with pytest.raises(ValueError, match="adds no claim"):
            Margins(se_ratio_sanity=(0.80, 1.20), calibration_se_ratio=(0.75, 1.25))

    def test_a_calibration_band_no_valid_estimator_could_satisfy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="correctly scaled"):
            Margins(calibration_se_ratio=(1.01, 1.07))
        with pytest.raises(ValueError, match="nominal rate"):
            Margins(alpha=0.05, calibration_coverage=(0.96, 0.99))

    def test_the_calibration_band_is_the_one_that_catches_a_ten_percent_understatement(
        self,
    ) -> None:
        """The gap finding 4 named, stated as an arithmetic fact rather than a policy.

        A standard error 10% too small leaves coverage at 0.922 -- above the 0.90 floor and
        inside the sanity band, which :meth:`Margins.__post_init__` will not let anyone
        tighten past 0.8392.  Only the calibration band separates the two.
        """
        margins = Margins()
        assert coverage_for_se_ratio(0.90, alpha=margins.alpha) > margins.coverage_floor
        assert margins.se_ratio_sanity[0] < 0.90
        assert margins.calibration_se_ratio[0] > 0.90


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
    def _rows(self, exponent: float, *, replicates: int = 400, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        frames = []
        for size in (500, 2000, 8000):
            spread = size**exponent
            estimates = rng.normal(scale=spread, size=replicates)
            frames.append(
                pd.DataFrame(
                    {
                        "property": "rate",
                        "cell": f"n_{size}",
                        "n": size,
                        "truth": 0.0,
                        "estimate": estimates,
                        "std_error": np.full(replicates, spread * 1.96 / norm.ppf(0.975)),
                        "covered": 1,
                        "rejected": 0,
                        "requested_replicates": replicates,
                        "failed_replicates": 0,
                        "replicate": np.arange(replicates),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def _fitted(self, exponent: float, *, replicates: int = 400, seed: int = 1) -> Rate:
        return rate(
            self._rows(exponent, replicates=replicates, seed=seed),
            property_name="rate",
            statistic="spread",
            bootstrap_replicates=2000,
            confidence_level=CONFIDENCE,
            seed=0,
        )

    def test_a_root_n_sampling_distribution_is_recognised(self) -> None:
        fitted = self._fitted(-0.5)
        assert fitted.equivalent_to(-0.5, ROOT_N_SLOPE_MARGIN)
        assert fitted.excludes(-0.25)

    def test_a_slower_contraction_is_rejected(self) -> None:
        """The check has to be able to fail, which the ratio of two mean standard errors could not."""
        fitted = self._fitted(-0.25)
        assert not fitted.equivalent_to(-0.5, ROOT_N_SLOPE_MARGIN)
        assert not fitted.excludes(-0.25)

    def test_the_margin_bounded_rate_verdict_survives_more_replications(self) -> None:
        """The rule the studies use, on a distribution that really does contract at root-n."""
        verdicts = [
            self._fitted(-0.5, replicates=replicates).equivalent_to(-0.5, ROOT_N_SLOPE_MARGIN)
            for replicates in (200, 800, 3200)
        ]
        assert all(verdicts), verdicts

    def test_requiring_the_interval_to_contain_the_exact_rate_flips_the_other_way(self) -> None:
        """The clause this replaced, on a sampling distribution that is root-n for any purpose.

        An exponent of -0.52 is inside any margin a study of this kind declares, and the
        published reported-SE rate cleared exact containment of -1/2 by 4.4e-5 -- so the rule
        was one quadrupling away from turning a passing study red with no estimator change.
        """
        flags = [
            self._fitted(-0.52, replicates=replicates).consistent_with(-0.5)
            for replicates in (200, 800, 3200)
        ]
        assert flags[0] is True and flags[-1] is False, flags
        assert all(
            self._fitted(-0.52, replicates=replicates).equivalent_to(-0.5, ROOT_N_SLOPE_MARGIN)
            for replicates in (200, 800, 3200)
        )

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


class TestResamplingStreams:
    """Two published intervals must not share their bootstrap indices."""

    def test_every_named_stream_of_every_study_is_distinct(self) -> None:
        labels = [
            ("root_n_rate", "empirical_sd"),
            ("root_n_rate", "reported_se"),
            ("interval_calibration", "correctly_specified"),
            ("crossfit_overfitting", "in_sample_control"),
            ("crossfit_overfitting", "coverage_gain"),
            ("performance", "cleverly", "binary", "ate"),
            ("equivalence", "binary", "ate"),
        ]
        studies = registered()
        seeds = [stream_seed(study, *label) for study in studies for label in labels]
        assert len(set(seeds)) == len(seeds), "two analyses would resample on the same stream"

    def test_the_scheme_it_replaced_collides_across_studies(self) -> None:
        """Why labels rather than offsets, stated as an executable fact.

        The three registered study seeds are consecutive integers, so cell *k* of one study
        and cell *k-1* of the next landed on the same ``seed + offset + index``.  Rows a
        reader compares side by side then share their Monte Carlo error, which is exactly
        what an independent stream per analysis is supposed to prevent.
        """
        studies = registered()
        offsets = [study.seed + 10_000 + index for study in studies for index in range(3)]
        assert len(set(offsets)) < len(offsets)

    def test_the_two_rate_rows_no_longer_share_a_per_size_stream(self) -> None:
        """``rate`` spawns its per-size children instead of adding the size index.

        Adding it made ``empirical_sd``'s stream for one size identical to ``reported_se``'s
        for the next, because the two callers' base seeds were themselves one apart -- so
        two of the three sizes behind each published slope were resampled identically.
        """
        study = registered()[0]

        def children(cell: str) -> set[tuple[int, ...]]:
            root = np.random.SeedSequence(stream_seed(study, "root_n_rate", cell))
            return {(*child.spawn_key, child.entropy) for child in root.spawn(3)}

        assert not children("empirical_sd") & children("reported_se")


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

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [("4.449e-08", True), ("4.45e-08", True), ("4e-08", True), ("4.448e-08", False)],
    )
    def test_a_scientific_figure_is_checked_to_its_significant_digits(
        self, printed: str, expected: bool
    ) -> None:
        assert matches(printed, 4.448596227441203e-08) is expected

    def test_a_figure_too_small_for_decimals_is_not_printed_as_zero(self) -> None:
        """The rule the six-decimal floor broke.

        ``0.000000`` is what a paired margin utilization of ``4.45e-08`` used to print as, and
        it is a *correct* rounding at six decimals -- ``matches`` was never wrong.  What was
        wrong is that a literal zero rounds to the same six decimals, so the published cell
        could not distinguish a comparison agreeing to solver precision from one never made.
        ``render`` is what has to tell them apart, and now does.
        """
        assert matches("0.000000", 4.448596227441203e-08)
        assert render(4.448596227441203e-08) == "4.449e-08"
        assert render(0.0) == "0"
        assert render(4.448596227441203e-08) != render(0.0)

    def test_a_student_interval_needs_more_than_one_value(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            student_interval(np.array([1.0]), confidence_level=CONFIDENCE)
