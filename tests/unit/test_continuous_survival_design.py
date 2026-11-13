"""The ``continuous-time-survival`` declaration: truths, spreads, sizes and cells.

The truth and the efficient spread are checked against Monte Carlo on the law, and the
replicate counts against the shared sizing rules at the committed pilot.  Nothing here reads a
study run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from tests.studies import continuous_survival as study
from tests.studies import continuous_survival_law as law
from tests.studies import continuous_survival_pilot as pilot
from tests.studies import continuous_survival_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins


def _covariates(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    factor = rng.integers(0, 3, n)
    return (factor == 1).astype(float), (factor == 2).astype(float), rng.normal(size=n)


def _risk(z: float, cause: int, t: float, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):  # type: ignore[no-untyped-def]
    h1, h2 = law.rates(x1, x2, x3, z)
    h = h1 + h2
    return (h1 if cause == 1 else h2) / h * (1.0 - np.exp(-h * t))


class TestTheLaw:
    def test_the_truth_matches_monte_carlo(self) -> None:
        x1, x2, x3 = _covariates(400_000, 0)
        arms = {"treated": 1.0, "control": 0.0}
        for (arm, z), cause, t in (
            (pair, cause, t) for pair in arms.items() for cause in (1, 2) for t in law.TIMES
        ):
            values = _risk(z, cause, t, x1, x2, x3)
            spread = values.std() / np.sqrt(values.size)
            name = law.level_name(arm, law.CAUSES[cause], t)
            assert abs(values.mean() - law.TRUTH[name]) < 4.0 * spread, name

    def test_the_differences_are_the_level_gaps(self) -> None:
        for cause in law.CAUSES.values():
            for t in law.TIMES:
                gap = (
                    law.TRUTH[law.level_name("treated", cause, t)]
                    - law.TRUTH[law.level_name("control", cause, t)]
                )
                assert law.TRUTH[law.difference_name(cause, t)] == pytest.approx(gap, abs=1e-14)

    def test_the_quadrature_has_converged(self) -> None:
        for cause, t in ((1, law.TIMES[1]), (2, law.TIMES[0])):
            coarse = law.efficiency_sd(cause, t, nodes=80)
            fine = law.efficiency_sd(cause, t)
            for key, value in fine.items():
                assert coarse[key] == pytest.approx(value, rel=1e-8)

    def test_the_efficient_spread_matches_simulated_influence_curves(self) -> None:
        # The efficient influence curve of each arm's risk of cause 1 at the middle time,
        # evaluated with the true hazards, treatment law and censoring on simulated draws.
        cause, t = 1, law.TIMES[1]
        frame = law.sample(400_000, 8)
        x1, x2, x3 = (frame[c].to_numpy() for c in ("X1", "X2", "X3"))
        a, time, event = frame["Z"].to_numpy(), frame["time"].to_numpy(), frame["event"].to_numpy()
        p1 = expit(law.OMEGA[0] * x1 + law.OMEGA[1] * x2 + law.OMEGA[2] * x3)
        u, wu = np.polynomial.legendre.leggauss(60)
        curves = {}
        for arm, z in (("treated", 1.0), ("control", 0.0)):
            h1, h2 = law.rates(x1, x2, x3, z)
            h = h1 + h2
            share = (h1 if cause == 1 else h2) / h

            def clever(l: int, s: np.ndarray, h: np.ndarray = h, share: np.ndarray = share):  # type: ignore[no-untyped-def]
                shape = (-1,) + (1,) * (s.ndim - 1)
                return float(l == cause) - share.reshape(shape) * (
                    1.0 - np.exp(-h.reshape(shape) * (t - s))
                )

            stop = np.minimum(time, t)
            nodes = stop[:, None] / 2.0 * (u[None, :] + 1.0)
            weights = stop[:, None] / 2.0 * wu[None, :]
            retention = np.exp(-law.CENSORING_RATE * nodes)
            compensator = np.sum(
                weights
                * (h1[:, None] * clever(1, nodes) + h2[:, None] * clever(2, nodes))
                / retention,
                axis=1,
            )
            jumped = (time <= t) & (event > 0)
            jump = np.where(
                jumped,
                clever(1, time) * (event == 1) + clever(2, time) * (event == 2),
                0.0,
            ) / np.exp(-law.CENSORING_RATE * time)
            pi = p1 if z == 1.0 else 1.0 - p1
            level = _risk(z, cause, t, x1, x2, x3)
            curves[arm] = (a == z) / pi * (jump - compensator) + level - level.mean()
        expected = law.efficiency_sd(cause, t)
        for arm in ("treated", "control"):
            assert curves[arm].std() == pytest.approx(expected[arm], rel=0.015)
        difference = curves["treated"] - curves["control"]
        assert difference.std() == pytest.approx(expected["difference"], rel=0.015)

    def test_the_target_times_are_the_event_time_quartiles(self) -> None:
        assert law.event_time_quartiles(n=400_000) == pytest.approx(law.TIMES, abs=0.011)

    def test_the_gated_and_early_targets_partition_the_truths(self) -> None:
        assert set(law.GATED) | set(law.EARLY) == set(law.TRUTH)
        assert not set(law.GATED) & set(law.EARLY)
        assert len(law.EARLY) == 6
        assert all(f"@ t={law.TIMES[0]:g}]" in name for name in law.EARLY)


PILOT = pd.read_csv(pilot.OUTPUT, low_memory=False) if pilot.OUTPUT.exists() else pd.DataFrame()


def _kept(label: str, estimand: str | None = None) -> pd.DataFrame:
    rows = PILOT.loc[(PILOT["label"] == label) & PILOT["refusal"].isna()]
    if estimand is not None:
        rows = rows.loc[rows["estimand"] == estimand]
    return rows


@pytest.mark.skipif(PILOT.empty, reason="the pilot has not been committed")
class TestTheSizes:
    def test_the_gated_pilot_labels_lost_fewer_fits_than_the_failure_bound(self) -> None:
        for label in (study.GATED_CELL, "calibration_1k", *properties_labels()):
            draws = PILOT.loc[PILOT["label"] == label, "replicate"].nunique()
            refused = PILOT.loc[
                (PILOT["label"] == label) & PILOT["refusal"].notna(), "replicate"
            ].nunique()
            assert draws == pilot.PILOTS[label][3]
            assert refused / draws <= study.FAILURE_BOUND, label

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the gated cell's rows.

        Clauses (a) to (c) on its covered counts and clause (d) on its standardized biases,
        over the targets the record gates.
        """
        names = list(law.GATED)
        rows = _kept(study.GATED_CELL)
        assert int(rows["n"].iloc[0]) == study.STUDY.n
        hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
        covered = {f"{study.GATED_CELL}/{name}": int(hits[name].sum()) for name in names}
        error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
            index="replicate", columns="estimand", values="error"
        )[names]
        standardized = error / error.std(ddof=1)
        group = sizing.PilotGroup(
            standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
        )
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                len(hits),
                implementations=1,
                coverage_pass=share,
                bias_groups=[group],
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )

    @staticmethod
    def _calibration() -> pd.DataFrame:
        rows = _kept("calibration_1k", properties.RD)
        assert int(rows["n"].iloc[0]) == properties.CALIBRATION_N
        assert pilot.PILOTS["calibration_1k"][3] == sizing.CALIBRATION_PILOT_DRAWS
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
            rows = _kept(f"dr_{configuration}_1k", properties.TREATED_C1)
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

    def test_the_wrong_hazard_is_wrong_where_it_matters(self) -> None:
        # The design's pilot gate: the untargeted plug-in of the treated arm's risk of cause 1
        # sits at least 2 SE from the truth in the mechanism-correct cell.
        rows = _kept("dr_mechanism_correct_1k", properties.TREATED_C1)
        gap = (rows["initial_estimate"] - rows["truth"]).abs() / rows["std_error"]
        assert float(gap.median()) >= 2.0


def properties_labels() -> tuple[str, ...]:
    return tuple(
        f"dr_{configuration}_1k" for configuration, _ in properties.ROBUSTNESS_CELLS.values()
    )


class TestTheCells:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_gates_one_cell_at_its_n(self) -> None:
        assert set(study.STUDY.scenarios) == {study.GATED_CELL}
        assert study.STUDY.n == study.PILOT_CELLS[study.GATED_CELL][1]
        assert set(study.STUDY.scenarios[study.GATED_CELL]) == set(law.GATED)
        reported = set(properties.REPORTED_CELLS)
        assert reported == {*study.PILOT_CELLS, "positivity_4k"} - {study.GATED_CELL}
        for cell, (configuration, n, _) in properties.REPORTED_CELLS.items():
            law_name, settings = properties.CONFIGURATIONS[configuration]
            if cell in study.PILOT_CELLS:
                cross_fit, size = study.PILOT_CELLS[cell]
                assert (law_name, settings.get("cross_fit", True), n) == ("base", cross_fit, size)
            else:
                assert (law_name, settings, n) == ("positivity", {}, 4_000)

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

    def test_the_early_table_reads_the_first_time(self) -> None:
        rows = []
        for replicate in range(3):
            fit = {
                f"early{i}_estimate": law.TRUTH[name] + 0.01 * (replicate - 1)
                for i, name in enumerate(law.EARLY)
            }
            fit.update({f"early{i}_std_error": 0.01 for i in range(len(law.EARLY))})
            fit.update({f"early{i}_covered": 1.0 for i in range(len(law.EARLY))})
            rows.append({"scenario": study.GATED_CELL, "replicate": replicate, **fit})
        table = study.early_time(pd.DataFrame(rows))
        assert list(table["estimand"]) == list(law.EARLY)
        np.testing.assert_allclose(table["bias"], 0.0, atol=1e-12)
        np.testing.assert_allclose(table["se_ratio"], 1.0)

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
