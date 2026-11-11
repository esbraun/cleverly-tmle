"""The ``survival-dr-tmle`` declaration: truths, spreads, sizes and cells.

The truth and the efficient spread are checked against Monte Carlo on the law, and the
replicate counts against the shared sizing rules at the committed pilot.  Nothing here reads a
study run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.studies import repeated_sizing as sizing
from tests.studies import survival_dr_tmle as study
from tests.studies import survival_dr_tmle_law as law
from tests.studies import survival_dr_tmle_pilot as pilot
from tests.studies import survival_dr_tmle_properties as properties
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins


class TestTheLaw:
    def test_the_truth_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(0)
        w1, w2 = rng.normal(size=400_000), rng.normal(size=400_000)
        for name, a in ((law.ESTIMANDS[0], 1), (law.ESTIMANDS[1], 0)):
            survival = np.prod([1.0 - law.hazard(t, a, w1, w2) for t in range(1, 6)], axis=0)
            spread = survival.std() / np.sqrt(survival.size)
            assert abs((1.0 - survival.mean()) - law.TRUTH[name]) < 4.0 * spread

    def test_the_quadrature_has_converged(self) -> None:
        coarse = law.truth(nodes=300)
        for name, value in law.TRUTH.items():
            assert coarse[name] == pytest.approx(value, abs=1e-8)
        assert law.efficiency_sd(trial=False, nodes=60) == pytest.approx(
            law.efficiency_sd(trial=False), rel=1e-9
        )

    def test_the_efficient_spread_matches_monte_carlo(self) -> None:
        # The martingale part's conditional variance, summed directly on draws of W.
        rng = np.random.default_rng(1)
        w1, w2 = rng.normal(size=400_000), rng.normal(size=400_000)
        g = law.treatment(w1, w2, trial=False)
        leave = law.censoring(w1, w2)
        survival = [np.ones_like(w1)]
        for t in range(1, 6):
            survival.append(survival[-1] * (1.0 - law.hazard(t, 1, w1, w2)))
        total = np.zeros_like(w1)
        for t in range(1, 6):
            h = law.hazard(t, 1, w1, w2)
            ratio = survival[5] / survival[t]
            total += survival[t - 1] * ratio**2 * h * (1.0 - h) / (g * (1.0 - leave) ** (t - 1))
        sd = np.sqrt(survival[5].var() + total.mean())
        assert sd == pytest.approx(law.efficiency_sd(trial=False)[law.ESTIMANDS[0]], rel=5e-3)

    def test_the_trial_frame_declares_the_true_probabilities(self) -> None:
        frame = law.sample(200, 3, trial=True)
        np.testing.assert_array_equal(frame["p1"], 0.5)
        np.testing.assert_allclose(
            frame["r2"], 1.0 - law.censoring(frame["W1"].to_numpy(), frame["W2"].to_numpy())
        )


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()


def _pilot_rows(label: str, estimand: str | None = None) -> pd.DataFrame:
    rows = PILOT.loc[PILOT["label"] == label]
    if estimand is not None:
        rows = rows.loc[rows["estimand"] == estimand]
    return rows


@pytest.mark.skipif(PILOT.empty, reason="the pilot has not been committed")
class TestTheSizes:
    def test_the_pilot_lost_no_fit(self) -> None:
        assert PILOT["refusal"].fillna("").eq("").all()

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` at the X32b shares.

        One implementation, clauses (a) to (c) on the pilot's covered counts and clause (d) on
        its standardized biases, one group per cell with its noise covariance read from the
        pilot draws.
        """
        covered: dict[str, int] = {}
        bias_groups = []
        draws = set()
        for scenario in study.SCENARIOS:
            rows = _pilot_rows(scenario)
            assert int(rows["n"].iloc[0]) == study.STUDY.n_of(scenario)
            names = list(law.ESTIMANDS)
            hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
            covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
            error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
                index="replicate", columns="estimand", values="error"
            )[names]
            standardized = error / error.std(ddof=1)
            draws.add(len(hits))
            bias_groups.append(
                sizing.PilotGroup(
                    standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
                )
            )
        (pilot_draws,) = draws
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                pilot_draws,
                implementations=1,
                coverage_pass=share,
                bias_groups=bias_groups,
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )

    @staticmethod
    def _calibration() -> pd.DataFrame:
        rows = _pilot_rows("calibration_both", properties.RD)
        assert int(rows["n"].iloc[0]) == properties.CALIBRATION_N
        assert len(rows) == sizing.CALIBRATION_PILOT_DRAWS
        return rows

    def test_the_calibration_replicates_follow_the_pooled_rule(self) -> None:
        rows = self._calibration()
        ratio = float(rows["std_error"].mean() / rows["estimate"].std(ddof=1))
        noise = ratio**2 / (2.0 * (len(rows) - 1))
        groups = [sizing.PilotGroup(np.array([ratio]), np.array([[noise]]))]
        tau = sizing.heterogeneity_bound(groups)
        assert tau == 0.0

        def family_pass(replicates: int) -> float:
            return sizing.calibration_family_pass_probability(MARGINS, replicates, groups, tau)

        declared = properties.CALIBRATION_REPLICATES
        assert declared % sizing.BUDGET_STEP == 0
        assert family_pass(declared) >= sizing.SIZED_SHARE
        assert family_pass(declared - sizing.BUDGET_STEP) < sizing.SIZED_SHARE

    def test_the_calibration_efficiency_ratios_sit_inside_the_band(self) -> None:
        low, high = properties.EFFICIENCY_RATIO_BAND
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        replicates = properties.CALIBRATION_REPLICATES
        rows = self._calibration()
        bound = properties.EFFICIENCY_SD["both_rd"] / np.sqrt(properties.CALIBRATION_N)
        for ratio in (
            float(rows["estimate"].std(ddof=1)) / bound,
            float(rows["std_error"].mean()) / bound,
        ):
            spread = ratio / np.sqrt(2.0 * (replicates - 1))
            passing = norm.cdf((high - z * spread - ratio) / spread) - norm.cdf(
                (low + z * spread - ratio) / spread
            )
            assert passing > 0.999, (ratio, passing)

    def test_the_derived_controls_leave_the_band(self) -> None:
        replicates = properties.CALIBRATION_REPLICATES
        assert (
            sizing.calibration_control_probability(
                MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR
            )
            > 0.999
        )
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        noise = np.sqrt(2.0)
        spread = noise / np.sqrt(2.0 * (replicates - 1))
        upper = properties.EFFICIENCY_RATIO_BAND[1]
        assert norm.cdf((noise - z * spread - upper) / spread) > 0.999

    def test_the_robustness_cells_pass_at_their_pilot_bias(self) -> None:
        """Each positive at its one-sided 80% upper bias bound; the control at its lower one."""
        controls = [c for c, (_, role) in properties.ROBUSTNESS_CELLS.items() if role == "control"]
        smaller: list[float] = []
        for cell, (configuration, role) in properties.ROBUSTNESS_CELLS.items():
            rows = _pilot_rows(f"dr_{configuration}", properties.ARM1)
            assert int(rows["n"].iloc[0]) == properties.ROBUSTNESS_N
            error = rows["estimate"] - rows["truth"]
            bias = abs(float(error.mean() / error.std(ddof=1)))
            slack = float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(rows))
            replicates = properties.ROBUSTNESS_REPLICATES
            if role == "positive":
                probability = sizing.bias_equivalent_probability(MARGINS, bias + slack, replicates)
                assert probability > 0.98, (cell, bias)
                smaller.append(
                    sizing.bias_equivalent_probability(MARGINS, bias + slack, replicates - 400)
                )
            else:
                probability = sizing.bias_discriminated_probability(
                    MARGINS, bias - slack, replicates
                )
                assert probability >= sizing.discrimination_floor(len(controls)), (cell, bias)
        # The declared count is the smallest multiple of 400 that clears every positive.
        assert properties.ROBUSTNESS_REPLICATES % 400 == 0
        assert min(smaller) <= 0.98


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

    def test_the_sample_seeds_are_distinct(self) -> None:
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, label, r)
            for family, label in {(k[0], k[1]) for k in properties._keys(1)}
            for r in range(50)
        ]
        seeds += [
            properties.reported_seed(cell, r)
            for cell in properties.REPORTED_CELLS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)
