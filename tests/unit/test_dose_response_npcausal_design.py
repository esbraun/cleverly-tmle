"""The ``dose-response-npcausal`` declaration: truths, matched kernels, grid, seeds, sizes, runner.

The truths are checked against the composite rule and against a density-weighted Gaussian
local-linear fit, with a curved-curve control where the two targets part; the matched bandwidths
against the kernels' roughness; the ``ctseff`` grid against declared draws; the replicate count
against the shared sizing rule at the committed pilot; and the R runner's literals and the
reference's provenance against the study module.  Nothing here reads a study run or runs R.
"""

from __future__ import annotations

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
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("dose_response_npcausal_design")

MARGINS = study.STUDY.margins
RUNNER = ROOT / "tests" / "canonical" / "dose_response_npcausal" / "run_study.R"
DOCKERFILE = ROOT / "tests" / "canonical" / "npcausal_incremental" / "Dockerfile"
PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()


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
        for name, (a0, _, _) in study.PAIRS.items():
            assert study.TRUTH[name] == pytest.approx(0.2 + 0.5 * a0, abs=1e-15)

    def test_the_subject_target_is_the_truth(self) -> None:
        # The triweight equivalent kernel on D integrates to one with zero first moment, so the
        # line is its own smoothed value; a second quadrature confirms the composite rule.
        for name, (a0, h, _) in study.PAIRS.items():
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
        for name, (a0, _, h_gauss) in study.PAIRS.items():
            fitted = _density_weighted_fit(law.curve, a0, h_gauss)
            assert fitted == pytest.approx(study.TRUTH[name], abs=1e-10)

    def test_a_curved_curve_separates_the_two_targets(self) -> None:
        # The control: on the curve sin(2a) the Lebesgue triweight target and the
        # density-weighted Gaussian target differ by more than half the subject's efficient
        # standard error at n = 4,000 at some declared pair (0.75 at bandwidth 1.5, dose 2.5),
        # so the agreement above is the law's linearity and not an identity of the two
        # smoothers.  The matched roughness nearly equalizes the two kernels' second moments,
        # so most of the gap is the density weighting.
        def curved(a: np.ndarray) -> np.ndarray:
            return np.sin(2.0 * np.asarray(a, dtype=float))

        gaps = []
        for a0, h, h_gauss in study.PAIRS.values():
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
        assert outside * study.PRIMARY_REPLICATES * study.PRIMARY_N < 1e-9


class TestTheMatchedKernels:
    def test_the_roughness_constants_are_the_package_kernels(self) -> None:
        triweight = integrate.quad(
            lambda u: float(kernel("triweight", np.array([u]))[0] ** 2), -1, 1
        )[0]
        assert triweight == pytest.approx(study.TRIWEIGHT_ROUGHNESS, rel=1e-12)
        gaussian = integrate.quad(lambda u: norm.pdf(u) ** 2, -np.inf, np.inf)[0]
        assert gaussian == pytest.approx(study.GAUSSIAN_ROUGHNESS, rel=1e-12)

    def test_each_reference_bandwidth_has_the_subjects_roughness_per_unit_bandwidth(self) -> None:
        for h in study.BANDWIDTHS:
            h_gauss = study.NPCAUSAL_BANDWIDTHS[h]
            assert study.GAUSSIAN_ROUGHNESS / h_gauss == pytest.approx(
                study.TRIWEIGHT_ROUGHNESS / h, rel=1e-14
            )

    def test_every_pair_is_interior_so_the_weight_is_the_kernel(self) -> None:
        lower, upper = law.SUPPORT
        for a0, h, _ in study.PAIRS.values():
            assert lower <= a0 - h and a0 + h <= upper
            a = np.linspace(a0 - h, a0 + h, 101)
            weight = equivalent_kernel(a, a0, h, law.SUPPORT, study.KERNEL)
            assert np.allclose(weight, kernel("triweight", (a - a0) / h) / h, atol=1e-12)

    def test_the_leading_variances_agree_with_the_exact_efficient_variance(self) -> None:
        # R(K) E_W{1/g0(a0|W)} / h is the residual variance both estimators share to leading
        # order; at these bandwidths it sits within 10% of the subject's exact residual term.
        for a0, h, _ in study.PAIRS.values():
            a, mass = law.dose_rule((h, a0))
            weight = equivalent_kernel(a, a0, h, law.SUPPORT, study.KERNEL)
            exact = float(np.sum(mass * weight**2 * law.inverse_density_mean(a)))
            leading = study.TRIWEIGHT_ROUGHNESS * float(law.inverse_density_mean(a0)) / h
            assert exact == pytest.approx(leading, rel=0.10)


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
        # The subject refuses a D outside the observed dose range.
        scale = np.sqrt(law.DOSE_MEAN_VARIANCE + law.SIGMA**2)
        for edge, tail in ((law.SUPPORT[0], norm.cdf), (law.SUPPORT[1], norm.sf)):
            outside = float(tail((edge - law.DOSE_INTERCEPT) / scale))
            assert (1.0 - outside) ** study.PRIMARY_N * study.PRIMARY_REPLICATES < 1e-20


class TestTheRunner:
    TEXT = RUNNER.read_text(encoding="utf-8")

    def test_the_runner_restates_the_pairs_digit_for_digit(self) -> None:
        names = re.findall(r'"(dose_response\[[^"]+\])"', self.TEXT)
        doses = re.search(r"dose = c\(([^)]*)\)", self.TEXT)
        bandwidths = re.search(r"bandwidth = c\(([^)]*)\)", self.TEXT, re.S)
        assert doses is not None and bandwidths is not None
        stated = [float(value) for value in doses.group(1).split(",")]
        widths = [value.strip() for value in bandwidths.group(1).split(",")]
        assert names == list(study.PAIRS)
        assert stated == [a0 for a0, _, _ in study.PAIRS.values()]
        assert widths == [repr(h_gauss) for _, _, h_gauss in study.PAIRS.values()]

    def test_the_runner_states_the_grid_step_library_and_scenario(self) -> None:
        assert f"GRID_STEP <- {study.GRID_STEP}" in self.TEXT
        assert f'SL_LIBRARY <- "{study.SL_LIBRARY}"' in self.TEXT
        assert f'scenario <- "{study.SCENARIO}"' in self.TEXT

    def test_every_fit_check_stops_inside_a_stage(self) -> None:
        # A stop() in a fit is a recorded failure only inside study_stage; the one stop()
        # outside a function is the image check, which runs before any group.
        body = self.TEXT.split("fit_bandwidth <- function", 1)[1].split("fit_one <- function", 1)[0]
        staged = body.split('study_stage("ctseff", {', 1)
        assert len(staged) == 2 and "stop(" not in staged[0]
        assert self.TEXT.count("stop(") == 1 + 1 + staged[1].count("stop(")

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
    def test_the_pilot_lost_no_fit(self) -> None:
        assert not PILOT.empty
        assert int(PILOT["refusal"].fillna("").astype(bool).sum()) == 0
        counts = PILOT.groupby("implementation")["replicate"].nunique()
        assert counts.to_dict() == {study.IMPLEMENTATION: pilot.DRAWS, pilot.SKETCH: pilot.DRAWS}

    @staticmethod
    def _reading(implementation: str) -> tuple[dict[str, int], sizing.PilotGroup]:
        rows = PILOT.loc[PILOT["implementation"] == implementation]
        names = list(study.ESTIMANDS)
        hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
        assert len(hits) == pilot.DRAWS
        error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
            index="replicate", columns="estimand", values="error"
        )[names]
        standardized = error / error.std(ddof=1)
        group = sizing.PilotGroup(
            standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(standardized)
        )
        return {f"{implementation}/{name}": int(hits[name].sum()) for name in names}, group

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` at the X32b shares, twice.

        Reading A takes the subject's pilot coverage and biases for both implementations (R did
        not run in the pilot).  Reading B takes each implementation's own pilot rows, the
        transcription standing in for ``ctseff``.  The record declares the larger count.
        """
        share = sizing.PRIMARY_CLAUSE_SHARE
        subject, subject_group = self._reading(study.IMPLEMENTATION)
        sketch, sketch_group = self._reading(pilot.SKETCH)
        assumed = sizing.primary_replicates(
            MARGINS,
            subject,
            pilot.DRAWS,
            implementations=2,
            coverage_pass=share,
            bias_groups=[subject_group],
            bias_pass=share,
        )
        own = sizing.primary_replicates(
            MARGINS,
            {**subject, **sketch},
            pilot.DRAWS,
            implementations=1,
            coverage_pass=share,
            bias_groups=[subject_group, sketch_group],
            bias_pass=share,
        )
        assert (assumed, own) == (2_400, 2_800)
        assert max(assumed, own) == study.PRIMARY_REPLICATES

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
        assert set(record.scenarios[study.SCENARIO]) == set(study.TRUTH)
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

    def test_a_draw_is_its_seed(self) -> None:
        frame, truth = study.draw_scenario(study.SCENARIO, 200, 3)
        again = law.sample(200, replicate_seed(study.STUDY, study.SCENARIO, 3))
        assert frame.equals(again)
        assert truth == study.TRUTH

    def test_the_reference_input_carries_every_truth_and_no_subject_row(self) -> None:
        key = (study.SCENARIO, 0, 200)
        sample, truths, rows = study._reference_input(key, study._draw(key))
        assert list(sample.columns) == ["scenario", "replicate", *study.COVARIATES, "A", "Y"]
        assert {row["estimand"] for row in truths} == set(study.ESTIMANDS)
        assert rows == []
