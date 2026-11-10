"""The ``dose-response-npcausal`` declaration: truths, matched kernels, grid, seeds, sizes, runner.

The truths are checked against the composite rule and against a density-weighted Gaussian
local-linear fit, with a curved-curve control where the two targets part; the matched bandwidths
against the kernels' roughness; the ``ctseff`` grid against declared draws; the bandwidths and
the replicate count against the pilot rule; the transcription's ``ctseff`` failure path against
constructed draws; and the R runner's formulas and the reference's provenance against the study
module.  Once the declared run lands, the pre-registered red set is checked against its
``equivalence.csv``.  Nothing here runs R.
"""

from __future__ import annotations

import inspect
import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from cleverly.dose_response import equivalent_kernel, kernel
from tests.canonical.dose_response_npcausal import regenerate
from tests.studies import dose_response as s1
from tests.studies import dose_response_npcausal as study
from tests.studies import dose_response_npcausal_law as law
from tests.studies import dose_response_npcausal_pilot as pilot
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("dose_response_npcausal_design")

MARGINS = study.STUDY.margins
RUNNER = ROOT / "tests" / "canonical" / "dose_response_npcausal" / "run_study.R"
DOCKERFILE = ROOT / "tests" / "canonical" / "npcausal_incremental" / "Dockerfile"
PILOT = pilot.read(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()
ARTIFACTS = study.STUDY.artifacts
#: The largest count any declaration can reach, for the bounds that grow with the count.
LARGEST = max(study.PRIMARY_REPLICATES, pilot.PAIRED_CAP)


def _marginal_density(a: np.ndarray) -> np.ndarray:
    """The dose's marginal density: normal, mean 2.5, variance s^2 + sigma^2."""
    return norm.pdf(a, loc=law.DOSE_INTERCEPT, scale=np.sqrt(law.DOSE_MEAN_VARIANCE + law.SIGMA**2))


def _density_weighted_fit(curve: object, a0: float, h: float) -> float:
    """``ctseff``'s population target: the Gaussian local-linear fit against the dose density."""
    grid = np.linspace(a0 - 12.0 * h, a0 + 12.0 * h, 20_001)
    weight = norm.pdf((grid - a0) / h) * _marginal_density(grid)
    design = np.column_stack([np.ones_like(grid), grid - a0])
    values = np.asarray(curve(grid))  # type: ignore[operator]
    coefficients = np.linalg.solve(
        design.T @ (design * weight[:, None]), design.T @ (weight * values)
    )
    return float(coefficients[0])


class TestTheTruths:
    def test_each_truth_is_the_line_at_its_dose(self) -> None:
        for name, (a0, _, _) in study.PILOT_PAIRS.items():
            assert study.TRUTH[name] == pytest.approx(0.2 + 0.5 * a0, abs=1e-15)

    def test_the_subject_target_is_the_truth(self) -> None:
        # The triweight equivalent kernel on D integrates to one with zero first moment, so the
        # line is its own smoothed value; a second quadrature confirms the composite rule.
        for name, (a0, h, _) in study.PILOT_PAIRS.items():
            assert law.smoothed(h, a0, study.KERNEL) == pytest.approx(study.TRUTH[name], abs=1e-12)
            total = integrate.quad(
                lambda a, a0=a0, h=h: float(
                    equivalent_kernel(np.array([a]), a0, h, law.SUPPORT, study.KERNEL)[0]
                    * law.curve(a)
                ),
                a0 - h,
                a0 + h,
                epsabs=1e-13,
            )[0]
            assert total == pytest.approx(study.TRUTH[name], abs=1e-10)

    def test_the_reference_target_is_the_truth(self) -> None:
        for name, (a0, _, h_gauss) in study.PILOT_PAIRS.items():
            fitted = _density_weighted_fit(law.curve, a0, h_gauss)
            assert fitted == pytest.approx(study.TRUTH[name], abs=1e-10)

    def test_a_curved_curve_separates_the_two_targets(self) -> None:
        # The control: on the curve sin(2a) the Lebesgue triweight target and the
        # density-weighted Gaussian target differ by more than half the subject's efficient
        # standard error at n = 4,000 at some candidate pair,
        # so the agreement above is the law's linearity and not an identity of the two
        # smoothers.  The matched roughness nearly equalizes the two kernels' second moments,
        # so most of the gap is the density weighting.
        def curved(a: np.ndarray) -> np.ndarray:
            return np.sin(2.0 * np.asarray(a, dtype=float))

        gaps = []
        for a0, h, h_gauss in study.PILOT_PAIRS.values():
            a, mass = law.dose_rule((h, a0))
            weight = equivalent_kernel(a, a0, h, law.SUPPORT, study.KERNEL)
            lebesgue = float(np.sum(mass * weight * curved(a)))
            weighted = _density_weighted_fit(curved, a0, h_gauss)
            se = law.efficiency_sd(h, a0, study.KERNEL) / np.sqrt(study.PRIMARY_N)
            gaps.append(abs(lebesgue - weighted) / se)
        assert max(gaps) > 0.5

    def test_the_outcome_regression_is_in_the_learners_span(self) -> None:
        frame = law.sample(5_000, stream_seed(study.STUDY, "design", "span"))
        design = frame[["A", *study.COVARIATES]].to_numpy()
        learner = study.outcome_learner().fit(design, frame["Y"])
        truth = law.q_bar(frame["A"].to_numpy(), frame[list(study.COVARIATES)].to_numpy())
        target = study.outcome_learner().fit(design, truth)
        assert float(np.max(np.abs(target.predict(design) - truth))) < 1e-10
        assert float(np.mean((learner.predict(design) - truth) ** 2)) < 0.01

    def test_the_inverse_density_mean_is_its_quadrature(self) -> None:
        nodes, weights = np.polynomial.hermite_e.hermegauss(30)
        weights = weights / weights.sum()
        grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        mass = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
        for a in (0.5, 1.5, 2.5, 3.5, 4.5):
            numeric = float(mass @ (1.0 / law.density(np.full(len(grid), a), grid)))
            assert float(law.inverse_density_mean(a)) == pytest.approx(numeric, rel=1e-8)

    def test_the_gaussian_outcome_stays_inside_q_bounds(self) -> None:
        # P(Y outside Q_BOUNDS) by a tensor rule over W and the dose noise, times every row
        # the declared run draws.
        nodes, weights = np.polynomial.hermite_e.hermegauss(30)
        weights = weights / weights.sum()
        w = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        w_mass = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
        e_nodes, e_mass = np.polynomial.hermite_e.hermegauss(40)
        e_mass = e_mass / e_mass.sum()
        dose = law.dose_mean(w)[:, None] + law.SIGMA * e_nodes[None, :]
        mean = law.curve(dose) + law.c(dose) * law.index(w)[:, None]
        low, high = study.Q_BOUNDS
        outside = float(w_mass @ ((norm.sf(high - mean) + norm.cdf(low - mean)) @ e_mass))
        assert outside * LARGEST * study.PRIMARY_N < 1e-9


class TestTheMatchedKernels:
    def test_the_roughness_constants_are_the_package_kernels(self) -> None:
        triweight = integrate.quad(
            lambda u: float(kernel("triweight", np.array([u]))[0] ** 2), -1, 1
        )[0]
        assert triweight == pytest.approx(study.TRIWEIGHT_ROUGHNESS, rel=1e-12)
        gaussian = integrate.quad(lambda u: norm.pdf(u) ** 2, -np.inf, np.inf)[0]
        assert gaussian == pytest.approx(study.GAUSSIAN_ROUGHNESS, rel=1e-12)

    def test_each_reference_bandwidth_has_the_subjects_roughness_per_unit_bandwidth(self) -> None:
        for h in study.CANDIDATE_BANDWIDTHS:
            h_gauss = study.NPCAUSAL_BANDWIDTHS[h]
            assert study.GAUSSIAN_ROUGHNESS / h_gauss == pytest.approx(
                study.TRIWEIGHT_ROUGHNESS / h, rel=1e-14
            )

    def test_every_pair_is_interior_so_the_weight_is_the_kernel(self) -> None:
        lower, upper = law.SUPPORT
        for a0, h, _ in study.PILOT_PAIRS.values():
            assert lower <= a0 - h and a0 + h <= upper
            a = np.linspace(a0 - h, a0 + h, 101)
            weight = equivalent_kernel(a, a0, h, law.SUPPORT, study.KERNEL)
            assert np.allclose(weight, kernel("triweight", (a - a0) / h) / h, atol=1e-12)

    def test_the_leading_variances_agree_with_the_exact_efficient_variance(self) -> None:
        # R(K) E_W{1/g0(a0|W)} / h is the residual variance both estimators share to leading
        # order.  It sits within 10% of the subject's exact residual term at bandwidths 1 and
        # 1.5 and within 25% at 2.5, where the inverse density is no longer flat over the
        # window; the RMSE leg carries what the leading order leaves out.
        for a0, h, _ in study.PILOT_PAIRS.values():
            a, mass = law.dose_rule((h, a0))
            weight = equivalent_kernel(a, a0, h, law.SUPPORT, study.KERNEL)
            exact = float(np.sum(mass * weight**2 * law.inverse_density_mean(a)))
            leading = study.TRIWEIGHT_ROUGHNESS * float(law.inverse_density_mean(a0)) / h
            assert exact == pytest.approx(leading, rel=0.10 if h <= 1.5 else 0.25)


class TestTheGrid:
    def test_the_grid_holds_every_dose_and_spans_every_declared_draw(self) -> None:
        for replicate in range(25):
            frame, _ = study.draw_scenario(study.SCENARIO, study.PRIMARY_N, replicate)
            a = frame["A"].to_numpy()
            grid = study.npcausal_grid(a)
            assert set(study.DOSES) <= set(grid.tolist())
            assert grid[0] <= a.min() < grid[0] + study.GRID_STEP
            assert grid[-1] - study.GRID_STEP < a.max() <= grid[-1]
            assert np.all(np.diff(grid) == study.GRID_STEP)

    def test_the_dose_interval_lies_inside_every_sample(self) -> None:
        # The subject refuses a D outside the observed dose range: the chance that any declared
        # sample has no dose past either edge of D = [-1.5, 6.5] is below one in a million.
        scale = np.sqrt(law.DOSE_MEAN_VARIANCE + law.SIGMA**2)
        for edge, tail in ((law.SUPPORT[0], norm.cdf), (law.SUPPORT[1], norm.sf)):
            outside = float(tail((edge - law.DOSE_INTERCEPT) / scale))
            assert (1.0 - outside) ** study.PRIMARY_N * LARGEST < 1e-6

    def test_the_interval_is_its_own_reflection(self) -> None:
        # The reflected pilot fits A -> 5 - A on the same curve.
        lower, upper = law.SUPPORT
        assert (pilot.MIRROR - upper, pilot.MIRROR - lower) == law.SUPPORT
        assert sorted(pilot.MIRROR - a0 for a0 in study.DOSES) == list(study.DOSES)


class TestTheRunner:
    TEXT = RUNNER.read_text(encoding="utf-8")

    CODE = "\n".join(line.split("#", 1)[0] for line in TEXT.splitlines())

    def test_the_runner_parses_the_pair_names_the_study_writes(self) -> None:
        stated = re.search(r'^PAIR_NAME <- "(.*)"$', self.TEXT, re.M)
        assert stated is not None
        # R's string escapes: "\\[" in the source is the regex \[.
        assert stated.group(1).replace("\\\\", "\\") == study.PAIR_NAME.pattern
        for name, (a0, h, _) in study.PILOT_PAIRS.items():
            match = study.PAIR_NAME.match(name)
            assert match is not None
            assert (float(match.group(1)), float(match.group(2))) == (h, a0)

    def test_the_runner_computes_the_gaussian_bandwidth_in_the_studys_order(self) -> None:
        assert "gaussian_bandwidth <- function(h) h * (1 / (2 * sqrt(pi))) / (350 / 429)" in (
            self.TEXT
        )
        assert "return h * GAUSSIAN_ROUGHNESS / TRIWEIGHT_ROUGHNESS" in inspect.getsource(
            study.gaussian_bandwidth
        )
        assert 1 / (2 * math.sqrt(math.pi)) == study.GAUSSIAN_ROUGHNESS
        assert study.TRIWEIGHT_ROUGHNESS == 350 / 429
        assert study.NPCAUSAL_BANDWIDTHS[1.0] == 0.3457676162028392

    def test_the_runner_states_the_grid_step_library_and_scenario(self) -> None:
        assert f"GRID_STEP <- {study.GRID_STEP}" in self.TEXT
        assert f'SL_LIBRARY <- "{study.SL_LIBRARY}"' in self.TEXT
        assert f'scenario <- "{study.SCENARIO}"' in self.TEXT

    def test_the_runner_widens_the_range_as_the_grid_mirror_does(self) -> None:
        # npcausal_grid's three lines, in R: floor and ceiling to 1/8, and the point count.
        lines = [
            r"low <- floor\(min\(frame\$A\) / GRID_STEP\) \* GRID_STEP",
            r"high <- ceiling\(max\(frame\$A\) / GRID_STEP\) \* GRID_STEP",
            r"points <- as\.integer\(round\(\(high - low\) / GRID_STEP\)\) \+ 1L",
        ]
        for line in lines:
            assert len(re.findall(rf"^\s*{line}$", self.CODE, re.M)) == 1, line
        mirror = inspect.getsource(study.npcausal_grid)
        assert "math.floor(float(np.min(a)) / GRID_STEP) * GRID_STEP" in mirror
        assert "math.ceil(float(np.max(a)) / GRID_STEP) * GRID_STEP" in mirror
        assert "round((high - low) / GRID_STEP) + 1" in mirror

    def test_every_fit_check_stops_inside_a_stage(self) -> None:
        # A stop() in a fit is a recorded failure only inside study_stage.  Outside it there
        # are the image check, before any group, and pairs_for's two input checks, which are
        # deliberately fatal.  Comments are stripped before counting.
        code = self.CODE
        body = code.split("fit_bandwidth <- function", 1)[1].split("fit_one <- function", 1)[0]
        staged = body.split('study_stage("ctseff", {', 1)
        assert len(staged) == 2 and "stop(" not in staged[0]
        pairs = code.split("pairs_for <- function", 1)[1].split("fit_bandwidth <- function", 1)[0]
        assert pairs.count("stop(") == 2
        assert code.count("stop(") == 1 + 2 + staged[1].count("stop(")

    def test_the_reference_runs_the_pinned_npcausal_image(self) -> None:
        assert regenerate.REFERENCE.image == study.NPCAUSAL_IMAGE
        here = ROOT / "tests" / "canonical" / "dose_response_npcausal"
        files = regenerate.REFERENCE.files(here)
        assert DOCKERFILE in files and RUNNER in files
        assert all(path.exists() for path in files)
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        assert study.NPCAUSAL_COMMIT in dockerfile
        assert study.R_BASE_IMAGE in dockerfile
        assert f"== '{study.NPCAUSAL_VERSION}'" in dockerfile
        assert '"/fixture/study_harness.R"' in self.TEXT


class TestTheSizes:
    def test_the_pilot_lost_no_subject_fit(self) -> None:
        assert not PILOT.empty
        counts = PILOT.groupby("implementation")["replicate"].nunique().to_dict()
        assert counts == {
            study.IMPLEMENTATION: pilot.DRAWS,
            pilot.SKETCH: pilot.DRAWS,
            pilot.REFLECTED: pilot.REFLECTED_DRAWS,
        }
        subject = PILOT.loc[PILOT["implementation"] != pilot.SKETCH]
        assert int((subject["refusal"].fillna("") != "").sum()) == 0

    def test_the_declaration_is_the_pilot_rule(self) -> None:
        """The bandwidths and the count are :func:`declaration_from_pilot`'s output.

        It reads the committed R pilot once that exists, and the transcription until then, so
        the declared values are provisional until the R pilot is committed.
        """
        declaration = pilot.declaration_from_pilot()
        expected = study.REFERENCE if pilot.R_OUTPUT.exists() else pilot.SKETCH
        assert declaration.source == expected
        assert declaration.bandwidths == study.BANDWIDTHS
        assert declaration.replicates == study.PRIMARY_REPLICATES == study.STUDY.replicates
        assert declaration.replicates >= 1_000
        assert declaration.failure_exceed <= pilot.FAILURE_EXCEED

    def test_the_failure_bound_sizes_the_selection(self) -> None:
        # One pilot failure in 1,000 keeps the reference inside floor(0.01 R) at R = 2,800; ten
        # do not.  The rule is the chance of exceeding the bound at the rate's 95% upper bound.
        assert pilot.failure_upper(0, 1_000) == pytest.approx(1 - 0.05 ** (1 / 1_000), rel=1e-9)
        assert pilot.failure_exceed(1, 1_000, 2_800) <= pilot.FAILURE_EXCEED
        assert pilot.failure_exceed(10, 1_000, 2_800) > pilot.FAILURE_EXCEED

    def test_the_transcription_fails_where_an_extreme_dose_is_isolated(self) -> None:
        # The deliberate-mutation control for the B1 path: isolate the lowest dose of a pilot
        # draw by 2.5, more than four Gaussian bandwidths at h = 1 and 1.5 and fewer at 2 and
        # 2.5.  The risk is NA exactly at the first two, and the unchanged draw fits at all four.
        frame = pilot.draw(0)
        a = frame["A"].to_numpy()
        order = np.argsort(a)
        isolated = frame.copy()
        isolated.loc[order[0], "A"] = a[order[1]] - 2.5
        unchanged, moved = pilot.ctseff_inputs(frame), pilot.ctseff_inputs(isolated)
        for h in study.CANDIDATE_BANDWIDTHS:
            h_gauss = study.NPCAUSAL_BANDWIDTHS[h]
            assert math.isfinite(pilot.leave_one_out_risk(unchanged, h_gauss))
            assert math.isnan(pilot.leave_one_out_risk(moved, h_gauss)) == (4 * h_gauss < 2.5)
        with pytest.raises(pilot.CtseffRefusal, match=r"which\.min"):
            pilot.ctseff_fit(moved, study.NPCAUSAL_BANDWIDTHS[1.0], study.DOSES)

    def test_the_binned_fit_is_the_local_linear_fit_at_an_interior_node(self) -> None:
        # locpoly's binning moves an interior estimate by far less than a standard error.
        frame = pilot.draw(1)
        inputs = pilot.ctseff_inputs(frame)
        h_gauss = study.NPCAUSAL_BANDWIDTHS[1.0]
        nodes, fit = pilot.locpoly(inputs.a, inputs.pseudo, h_gauss)
        u = (inputs.a - 2.5) / h_gauss
        weight = norm.pdf(u)
        design = np.column_stack([np.ones_like(u), u])
        exact = np.linalg.solve(
            design.T @ (design * weight[:, None]), design.T @ (weight * inputs.pseudo)
        )
        assert float(pilot.approx(nodes, fit, np.array([2.5]))[0]) == pytest.approx(
            exact[0], abs=1e-3
        )
        assert nodes[0] == inputs.a.min() and nodes[-1] == inputs.a.max()

    @staticmethod
    def _se_over_efficient(implementation: str) -> dict[tuple[float, float], float]:
        rows = PILOT.loc[PILOT["implementation"] == implementation]
        means = rows.groupby(["bandwidth", "dose"])[["std_error", "efficient_sd"]].mean()
        return (means["std_error"] / means["efficient_sd"]).to_dict()

    def test_the_over_report_follows_the_upper_end_of_the_dose_axis(self) -> None:
        """The M2 witness: the reflected fits move the law, and the excess stays at dose 3.5.

        The reflected draws are the original law's draws with the dose mirrored, so the
        reflected law at dose 3.5 is the original law at dose 1.5.  A law-driven excess would
        move to the reflected dose 1.5.  It stays at the upper end of the dose axis instead, the
        direction in which the pooled-hazard density compounds its bin hazards.
        """
        for implementation in (study.IMPLEMENTATION, pilot.REFLECTED):
            ratio = self._se_over_efficient(implementation)
            for h in study.BANDWIDTHS:
                assert ratio[(h, 3.5)] > study.UPPER_DOSE_SE_RATIO, (implementation, h)
                assert ratio[(h, 3.5)] - ratio[(h, 1.5)] > 0.05, (implementation, h)

    def test_the_stated_pilot_readings_are_the_pilots(self) -> None:
        """The pilot numbers the regeneration docstring and the evidence page state."""
        reference = pilot.sketch_reference()
        assert {h: len(reference.failed[h]) for h in study.CANDIDATE_BANDWIDTHS} == {
            1.0: 74,
            1.5: 7,
            2.0: 1,
            2.5: 0,
        }
        declaration = pilot.declaration_from_pilot(reference=reference)
        assert (declaration.assumed, declaration.own, declaration.paired) == (2_400, 2_400, 4_800)
        ratio = self._se_over_efficient(study.IMPLEMENTATION)
        assert [round(ratio[(h, 3.5)], 2) for h in study.BANDWIDTHS] == [1.10, 1.11]
        lost = reference.failed[2.0] | reference.failed[2.5]
        rows = reference.rows.loc[~reference.rows["replicate"].isin(lost)]
        legs = {
            name: pilot.paired_legs(pilot.subject_rows().rows, rows, name)
            for name in study.ESTIMANDS
        }
        rmse = {name: round(cell.rmse_ratio, 2) for name, cell in legs.items()}
        assert rmse["dose_response[h=2.5,a=3.5]"] == 1.10
        assert rmse["dose_response[h=2,a=3.5]"] == 1.05
        assert max(rmse[name] for name in study.ESTIMANDS if "a=3.5" not in name) <= 1.0
        calibration = {name: cell.calibration_excess for name, cell in legs.items()}
        assert max(calibration.values()) == pytest.approx(0.020, abs=0.0005)
        lower = [name for name in study.ESTIMANDS if "a=3.5" not in name]
        assert max(calibration[name] for name in lower) <= 0.015
        replicates = study.PRIMARY_REPLICATES
        assert min(legs[name].pass_probability(replicates) for name in lower) >= 0.96
        subject = PILOT.loc[PILOT["implementation"] == study.IMPLEMENTATION]
        upper = subject.loc[(subject["bandwidth"] == 2.5) & (subject["dose"] == 3.5)]
        assert round(upper["estimate"].std() / upper["efficient_sd"].mean(), 2) == 1.11

    def test_the_subject_targets_in_the_pilot(self) -> None:
        # A nonzero targeting witness: the fluctuation moves the plug-in, so agreement with
        # ctseff cannot come from two untargeted fits.  ctseff reports no plug-in.
        rows = PILOT.loc[PILOT["implementation"] == study.IMPLEMENTATION]
        moved = (rows["estimate"] - rows["initial_estimate"]).abs() / rows["std_error"]
        assert float(moved.median()) > 0.2


class TestTheDeclaration:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_pairs_every_cell_with_npcausal(self) -> None:
        record = study.STUDY
        assert record.reference == "npcausal"
        assert record.comparison == "related target"
        assert record.comparator_state == "compared"
        assert record.paired_cells() == {(study.SCENARIO, name) for name in study.ESTIMANDS}
        assert set(record.scenarios[study.SCENARIO]) == set(study.ESTIMANDS)
        assert set(study.ESTIMANDS) <= set(study.TRUTH)
        assert record.publication_policy == "reporting"
        assert record.failure_bound == {study.IMPLEMENTATION: 0.01, study.REFERENCE: 0.01}
        assert record.property_cells == {}
        assert record.replicates >= 1_000

    def test_the_seed_streams_are_distinct(self) -> None:
        assert study.SEED != study.RESAMPLING_SEED
        assert study.SEED not in (s1.SEED, s1.RESAMPLING_SEED)
        seeds = [replicate_seed(study.STUDY, study.SCENARIO, r) for r in range(200)]
        seeds += [stream_seed(study.STUDY, "pilot", "primary", r) for r in range(200)]
        seeds += [stream_seed(study.STUDY, "design", "span")]
        seeds += [
            replicate_seed(s1.STUDY, scenario, r) for scenario in s1.SCENARIOS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)

    def test_the_seeds_are_no_registered_studys_seeds(self) -> None:
        from tests.studies.evidence.registry import registered

        taken = {seed for record in registered() for seed in (record.seed, record.resampling_seed)}
        assert not {study.SEED, study.RESAMPLING_SEED} & taken

    def test_a_draw_is_its_seed(self) -> None:
        frame, truth = study.draw_scenario(study.SCENARIO, 200, 3)
        again = law.sample(200, replicate_seed(study.STUDY, study.SCENARIO, 3))
        assert frame.equals(again)
        assert truth == {name: study.TRUTH[name] for name in study.ESTIMANDS}

    def test_the_reference_input_carries_every_truth_and_no_subject_row(self) -> None:
        key = (study.SCENARIO, 0, 200)
        sample, truths, rows = study._reference_input(key, study._draw(key))
        assert list(sample.columns) == ["scenario", "replicate", *study.COVARIATES, "A", "Y"]
        assert {row["estimand"] for row in truths} == set(study.ESTIMANDS)
        assert rows == []


class TestThePreregisteredRedSet:
    """The red paired rows the declaration expects, checked once the declared run lands.

    The record is ``reporting``, so a red row publishes.  This test is what keeps
    ``reporting`` from absorbing a red nobody predicted: every red paired row must carry the
    signature declared before the run, and any other red fails here and forces a diagnosis.
    Owner rows are filed after the run, against the rows this test admits.
    """

    @staticmethod
    def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        path = ARTIFACTS / "equivalence.csv"
        if not path.exists():
            pytest.skip("the declared run has not landed")
        equivalence = pd.read_csv(path)
        summary = pd.read_csv(ARTIFACTS / "summary.csv").set_index(["implementation", "estimand"])
        performance = pd.read_csv(ARTIFACTS / "performance-tests.csv").set_index(
            ["implementation", "estimand"]
        )
        return equivalence, summary, performance

    @staticmethod
    def _failed_legs(row: pd.Series) -> set[str]:
        legs = {
            "similarity": bool(row["paired_similarity"]),
            "RMSE": row["rmse_ratio_upper"] <= row["rmse_noninferiority_margin"],
            "coverage": row["coverage_difference_lower"] >= row["coverage_noninferiority_margin"],
            "calibration": row["calibration_excess_upper"]
            <= row["calibration_noninferiority_margin"],
        }
        return {name for name, held in legs.items() if not held}

    def test_every_red_paired_row_has_a_declared_signature(self) -> None:
        equivalence, summary, performance = self._tables()
        straddle_declared = pilot.declaration_from_pilot().paired is None
        subject, reference = study.IMPLEMENTATION, study.REFERENCE
        for _, row in equivalence.loc[~equivalence["passed"].astype(bool)].iterrows():
            name = str(row["estimand"])
            a0, h, _ = study.PILOT_PAIRS[name]
            mine, theirs = summary.loc[(subject, name)], summary.loc[(reference, name)]
            failed = self._failed_legs(row)
            assert bool(performance.loc[(subject, name), "passed"]), f"{name}: subject truth red"
            if a0 == study.EXPECTED_RED_DOSE:
                # The pooled-hazard density's upper-dose excess, read as an over-reported SE
                # (calibration) or as extra spread (RMSE), with similarity and coverage green.
                efficient = law.efficiency_sd(h, a0, study.KERNEL) / math.sqrt(study.PRIMARY_N)
                assert failed and failed <= {"RMSE", "calibration"}, f"{name}: {failed}"
                assert mine["mean_std_error"] / efficient > study.UPPER_DOSE_SE_RATIO, name
                assert mine["coverage"] >= 0.95, name
                continue
            # The straddle reading, declared only where the paired-leg rule hit its cap.
            assert straddle_declared, f"{name}: a red outside the pre-registered set"
            assert failed == {"calibration"}, f"{name}: {failed}"
            high = study.STUDY.margins.calibration_se_ratio[1]
            assert 1.0 <= mine["se_ratio"] <= high and theirs["se_ratio"] < 1.0, name
            assert bool(performance.loc[(reference, name), "passed"]), name
