"""The ``eliminated-competing-events`` declaration: truths, limits, identities, sizes and cells.

The truths are checked against a fit on each law's exact sample, the double-robustness limits
and the targeting witnesses against the same exact sample, and the replicate counts against the
shared sizing rules at the committed pilot's coverage.  Nothing here reads a study run.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.longitudinal import SDR, SequentialIPW
from tests import discrete_law_competing as competing
from tests import discrete_law_point_survival as point
from tests.discrete_law_point_survival import CellMeans, CellProbabilities
from tests.studies import eliminated_competing_events as study
from tests.studies import eliminated_competing_events_pilot as pilot
from tests.studies import eliminated_competing_events_properties as properties
from tests.studies import eliminated_competing_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.canonical_point_survival_hazard import initial_estimates
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _exact(scenario: str) -> pd.DataFrame:
    key, _, _ = study.SCENARIOS[scenario]
    return law.tv_exact_frame() if key == "time_varying" else law.hb_exact_frame()


def _scale(name: str, n: int) -> float:
    return study.EFFICIENCY_SD[name] / np.sqrt(n)


@pytest.fixture(scope="module")
def limits() -> dict[str, Any]:
    """Each robustness cell's fit on its law's exact sample."""
    return {
        cell: _quiet(properties.robustness_fit, _exact(scenario), cell)
        for cell, (scenario, *_) in properties.ROBUSTNESS_CELLS.items()
    }


class TestTheTruths:
    @pytest.mark.parametrize("scenario", [study.LTMLE_TV, study.HAZARD_HB])
    def test_a_fit_on_the_exact_sample_returns_the_g_formula(self, scenario: str) -> None:
        result = _quiet(study.fit, _exact(scenario), scenario)
        for name in study.ESTIMANDS[scenario]:
            assert result[name].psi == pytest.approx(study.TRUTH[scenario][name], abs=1e-10)

    def test_the_eliminated_risk_is_not_the_cause_incidence(self) -> None:
        # Preventing death leaves more units at risk of relapse, so every eliminated risk at
        # t = 2 exceeds the competing-risk incidence of relapse under the same plan.
        for label in ("never", "always", "continue_if_l2"):
            eliminated = law.TV_TRUTH[f"risk_regimen[{label} @ t=2]"]
            incidence = competing.TRUTH[f"cif_regimen[{label}, relapse @ t=2]"]
            assert eliminated > incidence + 0.05

    def test_the_bounds_never_bind(self) -> None:
        # Every treatment and censoring probability lies in [0.25, 0.75] and every death
        # probability in [0.25, 0.5], so each node keeps at least 0.25 * 0.5 * 0.5 of its
        # units under a plan.  The cumulative bound 1e-3 sits below the product at the last
        # node of either law.
        for table in (competing.G1, competing.G2, competing.C1, competing.C2):
            assert table.min() >= 0.25 and table.max() <= 0.75
        assert competing.H1[1].max() <= 0.5 and competing.H2[1].max() <= 0.5
        assert point.G.min() >= 0.25
        assert max(death for _, death in point._PAIRS) <= 0.5
        time_varying = (0.25 * 0.5 * 0.5) ** 2
        held_baseline = point.G.min() * (0.5 * 0.5) ** 3
        assert study.G_BOUNDS[0] < min(time_varying, held_baseline)


class TestTheSaturatedIdentities:
    """Configurations the record does not run, because they coincide with one it does."""

    @pytest.fixture(scope="class")
    def frame(self) -> pd.DataFrame:
        return law.tv_sample(4_000, 20261191)

    def test_sdr_reduces_to_ltmle_under_saturated_nuisances(self, frame: pd.DataFrame) -> None:
        ltmle = _quiet(study.fit, frame, study.LTMLE_TV)
        sdr = _quiet(
            lambda: SDR(
                law.TV_REGIMENS,
                reference=law.TV_REFERENCE,
                n_folds=1,
                g_bounds=study.G_BOUNDS,
                simultaneous=False,
                outcome_learner=CellMeans(),
                treatment_learner=CellProbabilities(),
                censoring_learner=CellMeans(),
                competing_learner=CellMeans(),
            ).fit(frame, eliminate=list(study.ELIMINATE), **law.TV_FIT_COLUMNS)
        )
        for name in law.TV_ESTIMANDS:
            assert sdr[name].psi == pytest.approx(ltmle[name].psi, abs=1e-10)
            assert sdr[name].std_error == pytest.approx(ltmle[name].std_error, rel=1e-6)

    def test_a_correct_outcome_model_never_reads_the_death_factor(
        self, frame: pd.DataFrame
    ) -> None:
        both = _quiet(study.fit, frame, study.LTMLE_TV)
        wrong = _quiet(
            study.fit, frame, study.LTMLE_TV, competing_learner=DummyClassifier(strategy="prior")
        )
        for name in law.TV_ESTIMANDS:
            assert wrong[name].psi == pytest.approx(both[name].psi, abs=1e-10)

    def test_ipw_is_the_mechanism_correct_fit(self, frame: pd.DataFrame) -> None:
        cell = "tv_rule_t2__mechanism_correct"
        targeted = _quiet(properties.robustness_fit, frame, cell)
        ipw = _quiet(
            lambda: SequentialIPW(
                law.TV_REGIMENS,
                treatment_learner=CellProbabilities(),
                censoring_learner=CellMeans(),
                competing_learner=CellMeans(),
                g_bounds=study.G_BOUNDS,
                simultaneous=False,
            ).fit(frame, eliminate=list(study.ELIMINATE), **law.TV_FIT_COLUMNS)
        )
        for name in law.TV_MEANS:
            assert ipw[name].psi == pytest.approx(targeted[name].psi, abs=1e-8)


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(self, limits: dict[str, Any]) -> None:
        for cell, (scenario, _, name, _, role) in properties.ROBUSTNESS_CELLS.items():
            if role == "positive":
                truth = study.TRUTH[scenario][name]
                assert limits[cell][name].psi == pytest.approx(truth, abs=1e-9), cell

    def test_each_control_is_discriminated(self, limits: dict[str, Any]) -> None:
        controls = {
            cell: spec for cell, spec in properties.ROBUSTNESS_CELLS.items() if spec[4] == "control"
        }
        for cell, (scenario, n, name, _, _) in controls.items():
            gap = limits[cell][name].psi - study.TRUTH[scenario][name]
            # Read at the efficient spread and again at 1.5 times it, since a wrong fit's own
            # spread can exceed the bound.
            for spread in (_scale(name, n), 1.5 * _scale(name, n)):
                probability = sizing.bias_discriminated_probability(
                    MARGINS, gap / spread, properties.ROBUSTNESS_REPLICATES
                )
                assert probability >= sizing.discrimination_floor(len(controls)), cell

    def test_the_death_factor_carries_each_control(self, limits: dict[str, Any]) -> None:
        # The same fit with the death factor right lands on the truth, so the control's gap
        # is the death factor's.
        for cell, (scenario, _, name, _, role) in properties.ROBUSTNESS_CELLS.items():
            if role != "control":
                continue
            right = _quiet(
                study.fit,
                _exact(scenario),
                scenario,
                **properties.wrong_learners("mechanism_correct", scenario),
            )
            assert right[name].psi == pytest.approx(study.TRUTH[scenario][name], abs=1e-9)
            assert abs(limits[cell][name].psi - right[name].psi) > 0.01

    def test_each_positive_cell_witnesses_targeting(self) -> None:
        # The same regressions without the fluctuation would read red: the untargeted limit is
        # at least twice the margin from the truth at the cell's n.
        for cell, (scenario, n, name, configuration, role) in properties.ROBUSTNESS_CELLS.items():
            if role != "positive":
                continue
            frame = _exact(scenario)
            wrong = properties.wrong_learners(configuration, scenario)
            if scenario == study.HAZARD_HB:
                fitted = _quiet(study.fit, frame, scenario, **wrong)
                plug_in = initial_estimates(fitted, (name,))[name]
            else:
                plug_in = _quiet(study.untargeted, frame, scenario, **wrong)[name].psi
            gap = (plug_in - study.TRUTH[scenario][name]) / _scale(name, n)
            assert abs(gap) >= 2 * MARGINS.standardized_bias, cell
            assert (
                sizing.bias_discriminated_probability(
                    MARGINS, gap, properties.ROBUSTNESS_REPLICATES
                )
                > 0.99
            )


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()
CALIBRATION_PILOTS = {
    "ltmle_always_t2": "calibration_ltmle_16k",
    "hazard_a1_t3": "calibration_hazard_16k",
}


@pytest.mark.skipif(PILOT.empty, reason="the pilot has not been committed")
class TestTheSizes:
    def test_the_pilot_lost_no_fit(self) -> None:
        assert PILOT["refusal"].fillna("").eq("").all()

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` at the X32b shares.

        One implementation (no comparator), clauses (a) to (c) on the pilot's covered counts
        and clause (d) on its standardized biases, one group per scenario whose noise
        covariance is read from the pilot draws; both shares are
        :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`.
        """
        labels = {study.LTMLE_TV: "ltmle_16k", study.CROSSFIT_TV: "crossfit_16k"}
        labels[study.HAZARD_HB] = "hazard_16k"
        covered: dict[str, int] = {}
        bias_groups = []
        draws = set()
        for scenario, label in labels.items():
            rows = PILOT.loc[PILOT["label"] == label]
            assert int(rows["n"].iloc[0]) == study.STUDY.n_of(scenario)
            names = list(study.ESTIMANDS[scenario])
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
    def _calibration_pilot(cell: str) -> pd.DataFrame:
        _, _, name = properties.CALIBRATION_CELLS[cell]
        label = CALIBRATION_PILOTS[cell]
        rows = PILOT.loc[(PILOT["label"] == label) & (PILOT["estimand"] == name)]
        assert int(rows["n"].iloc[0]) == properties.CALIBRATION_N
        assert len(rows) == sizing.CALIBRATION_PILOT_DRAWS
        return rows

    def test_the_calibration_replicates_follow_the_pooled_rule(self) -> None:
        """The smallest multiple of 400 whose family-wise calibration pass reaches the share.

        The two cells sit on different laws, so each is its own group and the spread between
        cells does not enter (:func:`~tests.studies.repeated_sizing.heterogeneity_bound` reads
        no group of one cell).  Each cell's true SE ratio is its pilot ratio with the pilot's
        sampling noise.  The family-wise target is
        :data:`~tests.studies.repeated_sizing.SIZED_SHARE`.
        """
        groups = []
        for cell in properties.CALIBRATION_LABELS:
            rows = self._calibration_pilot(cell)
            ratio = float(rows["std_error"].mean() / rows["estimate"].std(ddof=1))
            noise = ratio**2 / (2.0 * (len(rows) - 1))
            groups.append(sizing.PilotGroup(np.array([ratio]), np.array([[noise]])))
        tau = sizing.heterogeneity_bound(groups)
        assert tau == 0.0

        def family_pass(replicates: int) -> float:
            return sizing.calibration_family_pass_probability(MARGINS, replicates, groups, tau)

        declared = properties.CALIBRATION_REPLICATES
        assert declared % sizing.BUDGET_STEP == 0
        assert family_pass(declared) >= sizing.SIZED_SHARE
        assert family_pass(declared - sizing.BUDGET_STEP) < sizing.SIZED_SHARE

    def test_the_calibration_efficiency_ratios_sit_inside_the_band(self) -> None:
        """The efficiency conjunct of each positive cell passes at the declared replicates."""
        low, high = properties.EFFICIENCY_RATIO_BAND
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        replicates = properties.CALIBRATION_REPLICATES
        for cell in properties.CALIBRATION_LABELS:
            rows = self._calibration_pilot(cell)
            bound = properties.EFFICIENCY_SD[cell] / np.sqrt(properties.CALIBRATION_N)
            for ratio in (
                float(rows["estimate"].std(ddof=1)) / bound,
                float(rows["std_error"].mean()) / bound,
            ):
                spread = ratio / np.sqrt(2.0 * (replicates - 1))
                passing = norm.cdf((high - z * spread - ratio) / spread) - norm.cdf(
                    (low + z * spread - ratio) / spread
                )
                assert passing > 0.999, (cell, ratio, passing)

    def test_the_derived_controls_leave_the_band(self) -> None:
        """The shrunken-SE arm leaves the SE-ratio band and the noise arm the efficiency band."""
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

    def test_the_positive_robustness_cells_pass_at_their_pilot_bias(self) -> None:
        """The pilot of each positive cell, at its one-sided 80% upper bias bound."""
        labels = {
            "tv_rule_t2__mechanism_correct": "dr_tv_rule_16k",
            "hb_a1_t2__mechanism_correct": "dr_hb_a1_16k",
        }
        for cell, label in labels.items():
            _, n, name, _, role = properties.ROBUSTNESS_CELLS[cell]
            assert role == "positive"
            rows = PILOT.loc[(PILOT["label"] == label) & (PILOT["estimand"] == name)]
            assert int(rows["n"].iloc[0]) == n
            error = rows["estimate"] - rows["truth"]
            bias = abs(float(error.mean() / error.std(ddof=1)))
            upper = bias + float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(rows))
            assert (
                sizing.bias_equivalent_probability(MARGINS, upper, properties.ROBUSTNESS_REPLICATES)
                > 0.98
            ), (cell, upper)


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
            for family, label, *_ in {(k[0], k[1]) for k in properties._keys(1)}
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)
