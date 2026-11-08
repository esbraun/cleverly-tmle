"""The ``dose-response`` declaration: truths, the tail dose, limits, sizes, controls and cells.

The truths are recomputed by a second quadrature and checked against identities of the law, the
exact efficient-influence spread and the expected Kish size against Monte Carlo, the
double-robustness limits by a population computation, and the replicate counts against the
shared sizing rules at the committed pilot.  Nothing here reads a study run.
"""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import beta, norm
from sklearn.linear_model import LogisticRegression

from cleverly.dose_response import equivalent_kernel, pair_name
from cleverly.learners.density import bin_edges, default_density_bins
from tests.studies import dose_response as study
from tests.studies import dose_response_law as law
from tests.studies import dose_response_pilot as pilot
from tests.studies import dose_response_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.default_band_properties import (
    MINIMUM_CONTROL_POWER,
    control_power,
    design,
)
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("dose_response_design")

MARGINS = study.STUDY.margins

#: The design numbers the declaration rests on, recomputed below.
BAND_P0 = 0.6757
BAND_CRITICAL = 2.751
#: The population limits of the robustness cells, in efficient SEs at n = 4,000.
POSITIVE_LIMIT_BOUND = 0.10
CONTROL_LIMIT_FLOOR = 2.7
MAIN_TERMS_LIMIT_CEILING = 0.45


def _se(h: float, a0: float, n: int = study.PRIMARY_N) -> float:
    return law.efficiency_sd(h, a0) / np.sqrt(n)


class TestTheTruths:
    @pytest.mark.parametrize("scenario", list(study.SCENARIOS))
    def test_every_truth_is_a_second_quadratures(self, scenario: str) -> None:
        spec = study.SCENARIOS[scenario]
        for h, a0 in spec.curve.pairs:
            lower, upper = law.SUPPORT
            points = sorted({lower, upper, *(x for x in (a0 - h, a0 + h) if lower < x < upper)})
            for stratum in (0, 1) if spec.strata else (None,):

                def integrand(
                    a: float, h: float = h, a0: float = a0, stratum: int | None = stratum
                ) -> float:
                    weight = equivalent_kernel(np.array([a]), a0, h, law.SUPPORT)[0]
                    value = law.curve(a, gaussian=spec.gaussian, stratum=stratum)[0]
                    return float(weight * value)

                total = sum(
                    integrate.quad(integrand, left, right, epsabs=1e-13, epsrel=1e-12)[0]
                    for left, right in pairwise(points)
                )
                name = pair_name(h, a0) if stratum is None else f"{pair_name(h, a0)}[V={stratum}]"
                assert study.TRUTH[scenario][name] == pytest.approx(total, abs=1e-10)

    def test_the_strata_average_to_the_marginal_curve(self) -> None:
        # P(V = 1) = 1/2, so the two stratum curves average to the unstratified one.
        for a0 in study.STRATA_DOSES:
            pair = f"dose_response[h=1,a={a0:.10g}]"
            halves = study.TRUTH[study.STRATA]
            average = 0.5 * (halves[f"{pair}[V=0]"] + halves[f"{pair}[V=1]"])
            assert average == pytest.approx(study.TRUTH[study.BINARY][pair], abs=1e-12)
            assert abs(halves[f"{pair}[V=1]"] - halves[f"{pair}[V=0]"]) > 0.05

    def test_the_smoothed_curve_is_linear_reproducing(self) -> None:
        # The equivalent kernel integrates to one with zero first moment on D, so a curve
        # linear on the kernel's support is its own smoothed value, at boundary doses too.
        for h, a0 in ((0.5, 0.25), (1.5, 4.75), (1.0, 2.5)):
            a, mass = law._dose_rule((h, a0))
            weight = equivalent_kernel(a, a0, h, law.SUPPORT)
            assert float(weight @ mass) == pytest.approx(1.0, abs=1e-12)
            assert float((weight * (a - a0)) @ mass) == pytest.approx(0.0, abs=1e-12)

    def test_the_efficient_spread_is_its_monte_carlo(self) -> None:
        h, a0 = properties.CALIBRATION_PAIR
        frame = law.sample(2_000_000, stream_seed(study.STUDY, "design", "efficiency"))
        a, w, y = frame["A"].to_numpy(), frame[list(study.COVARIATES)].to_numpy(), frame["Y"]
        nodes, mass = law._dose_rule((h, a0))
        weight = equivalent_kernel(nodes, a0, h, law.SUPPORT)
        linear = law.f(nodes)[None, :] + law.c(nodes)[None, :] * law.index(w)[:, None]
        integral = expit(linear) @ (weight * mass)
        clever = equivalent_kernel(a, a0, h, law.SUPPORT) / law.density(a, w)
        curve = clever * (y.to_numpy() - law.q_bar(a, w)) + integral
        assert float(curve.mean()) == pytest.approx(law.smoothed(h, a0), abs=4e-3)
        assert float(curve.std()) == pytest.approx(
            properties.EFFICIENCY_SD[properties.CALIBRATION_LABEL], rel=5e-3
        )

    def test_the_dose_interval_lies_inside_every_sample(self) -> None:
        # The fit refuses a D outside the observed dose range.  P(A < 0) and P(A > 5), each
        # at least 0.05, leave no realistic sample of 1,000 rows without a dose outside D.
        for edge, side in ((law.SUPPORT[0], "below"), (law.SUPPORT[1], "above")):
            nodes, mass = law._w_rule()
            z = (edge - law.dose_mean(nodes)) / law.SIGMA
            tail = float(mass @ (norm.cdf(z) if side == "below" else norm.sf(z)))
            assert tail > 0.05
            assert (1.0 - tail) ** study.TAIL_N < 1e-20

    def test_the_gaussian_outcome_stays_inside_q_bounds(self) -> None:
        # P(|Y| > 15) by a tensor rule over W and the dose noise, times every Gaussian row the
        # declared run draws.
        w_nodes, w_mass = law._w_rule()
        e_nodes, e_mass = np.polynomial.hermite_e.hermegauss(40)
        e_mass = e_mass / e_mass.sum()
        dose = law.dose_mean(w_nodes)[:, None] + law.SIGMA * e_nodes[None, :]
        mean = law.f(dose) + law.c(dose) * law.index(w_nodes)[:, None]
        low, high = study.Q_BOUNDS
        outside = float(w_mass @ ((norm.sf(high - mean) + norm.cdf(low - mean)) @ e_mass))
        rows = study.PRIMARY_REPLICATES * study.PRIMARY_N
        assert outside * rows < 1e-9


class TestTheTailDose:
    def test_the_tail_dose_puts_the_expected_kish_size_just_above_the_floor(self) -> None:
        floor = study.SCENARIOS[study.TAIL].curve.support_floor
        assert floor == 50.0
        kish = law.expected_kish(study.TAIL_BANDWIDTH, study.TAIL_DOSE, study.TAIL_N)
        assert 55.0 <= kish <= 65.0
        assert kish == pytest.approx(62.26, abs=0.01)
        # The declared dose is the one on the 0.05 grid whose Kish size lands in [55, 65].
        grid = np.round(np.arange(0.25, 1.0001, 0.05), 2)
        inside = [
            float(dose)
            for dose in grid
            if 55.0 <= law.expected_kish(study.TAIL_BANDWIDTH, float(dose), study.TAIL_N) <= 65.0
        ]
        assert inside == [study.TAIL_DOSE]

    def test_the_kish_formula_is_its_monte_carlo(self) -> None:
        h, a0 = study.TAIL_BANDWIDTH, study.TAIL_DOSE
        frame = law.sample(4_000_000, stream_seed(study.STUDY, "design", "kish"))
        a, w = frame["A"].to_numpy(), frame[list(study.COVARIATES)].to_numpy()
        clever = equivalent_kernel(a, a0, h, law.SUPPORT) / law.density(a, w)
        kish = np.abs(clever).sum() ** 2 / (clever * clever).sum()
        assert kish / len(frame) == pytest.approx(law.expected_kish(h, a0, 1), rel=0.02)

    def test_the_tail_pair_is_a_boundary_pair(self) -> None:
        # Its kernel reaches below D, so the weight is the truncated boundary kernel: it puts
        # mass at the edge of D, where the density is smallest, and is not symmetric about a0.
        assert law.SUPPORT[0] > study.TAIL_DOSE - study.TAIL_BANDWIDTH
        lower, a0, h = law.SUPPORT[0], study.TAIL_DOSE, study.TAIL_BANDWIDTH
        edge, mirror = equivalent_kernel(np.array([lower, 2 * a0 - lower]), a0, h, law.SUPPORT)
        assert edge > 0.5
        assert edge != pytest.approx(mirror, abs=0.1)


class TestTheClaimBoundary:
    """The pointwise interval read against the unsmoothed curve at dose 2.5, reported only."""

    EXPECTED: ClassVar[dict[float, float]] = {0.5: 0.9379, 1.0: 0.5972, 1.5: 0.0103}

    def test_the_expected_coverage_of_the_unsmoothed_curve(self) -> None:
        unsmoothed = float(law.curve(2.5)[0])
        for h, expected in self.EXPECTED.items():
            gap = (law.smoothed(h, 2.5) - unsmoothed) / _se(h, 2.5)
            coverage = norm.cdf(1.96 - gap) - norm.cdf(-1.96 - gap)
            assert coverage == pytest.approx(expected, abs=1e-4)

    def test_the_mse_optimal_bandwidth_is_nearest_the_smallest_declared(self) -> None:
        unsmoothed = float(law.curve(2.5)[0])
        grid = np.round(np.arange(0.2, 2.0001, 0.02), 2)
        mse = [(law.smoothed(h, 2.5) - unsmoothed) ** 2 + _se(h, 2.5) ** 2 for h in grid]
        best = float(grid[int(np.argmin(mse))])
        assert best == pytest.approx(0.60, abs=0.03)
        assert min(study.BANDWIDTHS, key=lambda h: abs(h - best)) == 0.5


# ---------------------------------------------------------------------------- robustness


@pytest.fixture(scope="module")
def population() -> dict[str, float]:
    """Each robustness configuration's limit at the cell's pair, in efficient SEs at 4,000.

    The regressions are fitted on a 1,000,000-row fixed-seed sample; the density is the binned
    marginal density at the 32 bins of n = 4,000 on that sample's quantiles; the fluctuation
    solves its population score against :math:`\\bar Q_0` rather than :math:`Y`; and the plug-in
    over 100,000 rows is read against the true plug-in over the same rows.
    """
    frame = law.sample(1_000_000, stream_seed(study.STUDY, "design", "robustness"))
    a, w = frame["A"].to_numpy(), frame[list(study.COVARIATES)].to_numpy()
    y = frame["Y"].to_numpy()
    design_matrix = np.column_stack([a, w])
    truth = law.q_bar(a, w)
    edges = bin_edges(a, default_density_bins(study.PRIMARY_N))
    width = np.diff(edges)
    bins = len(width)

    def marginal(doses: np.ndarray) -> np.ndarray:
        k = np.clip(np.searchsorted(edges, doses, side="right") - 1, 0, bins - 1)
        return np.asarray((1.0 / bins) / width[k])

    h, a0 = properties.ROBUSTNESS_PAIR
    nodes, mass = law._dose_rule((h, a0))
    weight = equivalent_kernel(nodes, a0, h, law.SUPPORT)
    rows = 100_000
    true_nodes = expit(law.f(nodes)[None, :] + law.c(nodes)[None, :] * law.index(w[:rows])[:, None])

    def limit(model: object) -> float:
        fitted = np.clip(model.predict_proba(design_matrix)[:, 1], 1e-9, 1 - 1e-9)  # type: ignore[attr-defined]
        clever = equivalent_kernel(a, a0, h, law.SUPPORT) / marginal(a)
        epsilon = brentq(
            lambda e: float(np.mean(clever * (truth - expit(logit(fitted) + e * clever)))),
            -10.0,
            10.0,
        )
        predicted = np.column_stack(
            [
                model.predict_proba(np.column_stack([np.full(rows, dose), w[:rows]]))[:, 1]  # type: ignore[attr-defined]
                for dose in nodes
            ]
        )
        targeted = expit(
            logit(np.clip(predicted, 1e-9, 1 - 1e-9)) + epsilon * (weight / marginal(nodes))
        )
        return float(np.mean((targeted - true_nodes) @ (weight * mass))) / _se(h, a0)

    return {
        "positive": limit(study.outcome_learner().fit(design_matrix, y)),
        "control": limit(study.outcome_learner(interactions=False).fit(design_matrix, y)),
        "main_terms": limit(LogisticRegression(C=1e8, max_iter=2000).fit(design_matrix, y)),
    }


class TestTheRobustnessLimits:
    def test_the_positive_limit_is_the_truth(self, population: dict[str, float]) -> None:
        assert abs(population["positive"]) < POSITIVE_LIMIT_BOUND

    def test_the_control_limit_is_far_from_the_truth(self, population: dict[str, float]) -> None:
        assert population["control"] > CONTROL_LIMIT_FLOOR

    def test_the_rejected_main_terms_control_was_too_close(
        self, population: dict[str, float]
    ) -> None:
        # The first control design, a main-terms logistic outcome with the marginal density,
        # sits under half an efficient SE off at this pair: too close for the floor.
        assert 0.25 < population["main_terms"] < MAIN_TERMS_LIMIT_CEILING
        probability = sizing.bias_discriminated_probability(
            MARGINS, population["main_terms"], properties.ROBUSTNESS_REPLICATES
        )
        assert probability < sizing.discrimination_floor(DISCRIMINATION_CHECKS)


#: The discrimination checks of the record: the robustness control, the band's pointwise
#: control and the two derived calibration controls.
DISCRIMINATION_CHECKS = 4
#: The failure share of the band and of the calibration cell, which split one sized
#: instrument (the C4 reading of :mod:`tests.studies.repeated_sizing`).
CALIBRATION_TERM_SHARE = 1.0 - sizing.SIZED_FAIL / 2.0
#: The calibration cell's pass probability at its declared size, read with its pilot noise.
CALIBRATION_PREDICTIVE_PASS = 0.888


# ---------------------------------------------------------------------------- the band


def _band_correlation() -> np.ndarray:
    covariance = law.efficiency_covariance(study.SCENARIOS[study.BINARY].curve.pairs)
    sd = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(sd, sd))


class TestTheBand:
    def test_the_pointwise_control_discriminates(self) -> None:
        numbers = design(_band_correlation(), properties.BAND_REPLICATES)
        assert numbers.p0 == pytest.approx(BAND_P0, abs=5e-4)
        assert numbers.critical == pytest.approx(BAND_CRITICAL, abs=2e-3)
        # The correlation is exact, so p0 needs no design allowance; the RM36 one is kept.
        assert (
            control_power(numbers.p0 + 0.005, properties.BAND_REPLICATES) >= MINIMUM_CONTROL_POWER
        )


# ---------------------------------------------------------------------------- sizes

PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()
#: The draws every primary scenario's pilot shares; the clauses read these.
COMMON_DRAWS = 1_000


def _lower(count: int, draws: int) -> float:
    """The one-sided 80% lower Clopper-Pearson bound of a pilot rate."""
    return float(beta.ppf(1.0 - sizing.PILOT_LEVEL, count, draws - count + 1)) if count else 0.0


def _calibration_ratio() -> tuple[float, float]:
    """The pooled pilot SE ratio of the calibration pair, and its noise SD."""
    labels = ("calibration_4k", "binary_4k")
    rows = PILOT.loc[
        PILOT["label"].isin(labels) & (PILOT["estimand"] == properties.CALIBRATION_NAME)
    ]
    by = rows.groupby("label")
    ratio = float(by["std_error"].mean().mean() / np.sqrt((by["estimate"].var(ddof=1)).mean()))
    return ratio, ratio / np.sqrt(2.0 * (len(rows) - len(labels)))


@pytest.mark.skipif(PILOT.empty, reason="the pilot has not been committed")
class TestTheSizes:
    def test_the_pilot_lost_no_fit(self) -> None:
        assert PILOT["refusal"].fillna("").eq("").all()
        for label, (_, n, draws, _) in pilot.PILOTS.items():
            rows = PILOT.loc[PILOT["label"] == label]
            assert rows["replicate"].nunique() == draws, label
            assert int(rows["n"].iloc[0]) == n, label

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` at the X32b shares.

        One implementation (the comparator is pending), clauses (a) to (c) on the covered
        counts of the first :data:`COMMON_DRAWS` pilot draws of each scenario, and clause (d)
        on their standardized biases, one group per scenario with its noise covariance read
        from the same draws.  Both shares are
        :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`.
        """
        labels = {
            study.BINARY: "binary_4k",
            study.STRATA: "strata_8k",
            study.GAUSSIAN: "gaussian_4k",
            study.TAIL: "tail_1k",
        }
        covered: dict[str, int] = {}
        bias_groups = []
        for scenario, label in labels.items():
            rows = PILOT.loc[(PILOT["label"] == label) & (PILOT["replicate"] < COMMON_DRAWS)]
            assert int(rows["n"].iloc[0]) == study.STUDY.n_of(scenario)
            names = list(study.ESTIMANDS[scenario])
            hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
            assert len(hits) == COMMON_DRAWS
            covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
            error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
                index="replicate", columns="estimand", values="error"
            )[names]
            standardized = error / error.std(ddof=1)
            bias_groups.append(
                sizing.PilotGroup(
                    standardized.mean().to_numpy(),
                    np.atleast_2d(standardized.cov().to_numpy()) / COMMON_DRAWS,
                )
            )
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                COMMON_DRAWS,
                implementations=1,
                coverage_pass=share,
                bias_groups=bias_groups,
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )

    def test_the_band_replicates_follow_the_joint_rule(self) -> None:
        """The smallest multiple of 400 at which the band cell passes with its term's share.

        The true joint coverage is the one-sided 80% lower bound of the pilot's band coverage
        over its 1,000 binary draws, read by
        :func:`~tests.studies.repeated_sizing.joint_band_pass_probability`.
        """
        rows = PILOT.loc[PILOT["label"] == "binary_4k"]
        joint = rows.groupby("replicate")["band_covered"].min()
        coverage = _lower(int(joint.sum()), len(joint))
        declared = properties.BAND_REPLICATES
        assert declared % sizing.BUDGET_STEP == 0

        def passing(replicates: int) -> float:
            return sizing.joint_band_pass_probability(MARGINS, replicates, coverage)

        assert passing(declared) >= CALIBRATION_TERM_SHARE
        assert declared == sizing.BUDGET_STEP or passing(declared - sizing.BUDGET_STEP) < (
            CALIBRATION_TERM_SHARE
        )

    def test_the_calibration_replicates_follow_the_point_rule(self) -> None:
        """The smallest multiple of 400 at which the cell passes with its term's share.

        The true SE ratio is read at the pooled pilot ratio of the calibration pair: the mean
        reported SE over the root-mean-square empirical SD of the single-pair pilot and of the
        same pair in the 15-pair binary pilot.  The two fits share the nuisance fits and the
        fluctuation and differ only in the quadrature partition, so they measure one ratio.
        Read with that ratio's pilot noise, the pass probability levels off below the share,
        because the pilot cannot exclude a ratio near 1.07; the red-cell reading of
        ``regenerate.py`` declares the route for that event, and
        :meth:`test_the_calibration_pass_with_pilot_noise_is_stated` pins it.
        """
        declared = properties.CALIBRATION_REPLICATES
        assert declared % sizing.BUDGET_STEP == 0
        ratio = _calibration_ratio()[0]

        def passing(replicates: int) -> float:
            return float(sizing.calibration_pass(MARGINS, replicates, np.array([ratio]))[0])

        assert passing(declared) >= CALIBRATION_TERM_SHARE
        assert declared == sizing.BUDGET_STEP or (
            passing(declared - sizing.BUDGET_STEP) < CALIBRATION_TERM_SHARE
        )

    def test_the_calibration_pass_with_pilot_noise_is_stated(self) -> None:
        ratio, noise = _calibration_ratio()
        groups = [sizing.PilotGroup(np.array([ratio]), np.array([[noise**2]]))]
        predictive = sizing.calibration_family_pass_probability(
            MARGINS, properties.CALIBRATION_REPLICATES, groups, 0.0
        )
        assert predictive == pytest.approx(CALIBRATION_PREDICTIVE_PASS, abs=0.01)

    def test_the_calibration_efficiency_ratios_sit_inside_the_band(self) -> None:
        """The efficiency conjunct passes at the declared replicates, at the pilot's ratios."""
        low, high = properties.EFFICIENCY_RATIO_BAND
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        replicates = properties.CALIBRATION_REPLICATES
        rows = PILOT.loc[PILOT["label"] == "calibration_4k"]
        bound = properties.EFFICIENCY_SD[properties.CALIBRATION_LABEL] / np.sqrt(
            properties.CALIBRATION_N
        )
        for ratio in (
            float(rows["estimate"].std(ddof=1)) / bound,
            float(rows["std_error"].mean()) / bound,
        ):
            spread = ratio / np.sqrt(2.0 * (replicates - 1))
            passing = norm.cdf((high - z * spread - ratio) / spread) - norm.cdf(
                (low + z * spread - ratio) / spread
            )
            assert passing > CALIBRATION_TERM_SHARE, (ratio, passing)

    def test_the_derived_controls_leave_the_band(self) -> None:
        replicates = properties.CALIBRATION_REPLICATES
        floor = sizing.discrimination_floor(DISCRIMINATION_CHECKS)
        assert (
            sizing.calibration_control_probability(
                MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR
            )
            > floor
        )
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        noise = np.sqrt(2.0)
        spread = noise / np.sqrt(2.0 * (replicates - 1))
        upper = properties.EFFICIENCY_RATIO_BAND[1]
        assert norm.cdf((noise - z * spread - upper) / spread) > floor

    def test_the_robustness_replicates(self, population: dict[str, float]) -> None:
        """The positive passes at its pilot bias and the control clears the margin.

        The positive's true standardized bias is the one-sided 80% upper bound of its pilot
        bias over 300 draws; its pass probability must reach the property instrument's share.
        The control's gap is its population limit over its own pilot spread, read at 1.25 times
        that spread for the pilot's error in it; its discrimination must reach the floor.
        """
        replicates = properties.ROBUSTNESS_REPLICATES
        positive = PILOT.loc[PILOT["label"] == "dr_positive_4k"]
        error = positive["estimate"] - positive["truth"]
        bias = abs(float(error.mean() / error.std(ddof=1)))
        upper = bias + float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(positive))
        assert sizing.bias_equivalent_probability(MARGINS, upper, replicates) > sizing.SIZED_SHARE
        assert replicates % 200 == 0
        assert sizing.bias_equivalent_probability(MARGINS, upper, replicates - 200) <= (
            sizing.SIZED_SHARE
        )
        control = PILOT.loc[PILOT["label"] == "dr_control_4k"]
        spread = float(control["estimate"].std(ddof=1)) / _se(*properties.ROBUSTNESS_PAIR)
        gap = population["control"] / (1.25 * spread)
        assert sizing.bias_discriminated_probability(MARGINS, gap, replicates) > (
            sizing.discrimination_floor(DISCRIMINATION_CHECKS)
        )


class TestTheCells:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_declares_every_cell_the_module_writes(self) -> None:
        written = {(family, cell) for family, cell, _, _ in properties.declared_truths()}
        declared = {
            (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
        }
        assert written == declared

    def test_each_key_names_the_cells_it_fills(self) -> None:
        names = {name for key in properties._keys(2) for name in properties._cells_of(key)}
        assert names == {
            f"{family}/{cell}"
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
        }

    def test_the_record_declares_the_scenario_sizes(self) -> None:
        for scenario, spec in study.SCENARIOS.items():
            assert study.STUDY.n_of(scenario) == spec.n
            assert set(study.ESTIMANDS[scenario]) == set(study.TRUTH[scenario])

    def test_the_sample_seeds_are_distinct(self) -> None:
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, label, r)
            for family, label, *_ in {(k[0], k[1]) for k in properties._keys(1)}
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "design", label)
            for label in ("efficiency", "kish", "robustness")
        ]
        assert len(set(seeds)) == len(seeds)
