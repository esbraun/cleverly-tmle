"""The ``separable-effects`` declaration: truths, limits, witnesses and cells.

The truths are checked against a fit on the law's exact sample, and each robustness cell's
limit against the same exact sample.  The sizes read the committed pilot; nothing reads a
study run.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.studies import repeated_sizing as sizing
from tests.studies import separable_effects as study
from tests.studies import separable_effects_law as law
from tests.studies import separable_effects_pilot as pilot
from tests.studies import separable_effects_properties as properties
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins

#: The robustness cells' size, in efficient standard errors of one fit.
SCALE = study.EFFICIENCY_SD[properties.ROBUSTNESS_ESTIMAND] / np.sqrt(properties.ROBUSTNESS_N)


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def limits() -> dict[str, float]:
    """Each robustness cell's estimate on the exact sample."""
    frame = law.exact_frame()
    return {
        cell: float(
            _quiet(properties.robustness_fit, frame, cell)[properties.ROBUSTNESS_ESTIMAND].psi
        )
        for cell in properties.ROBUSTNESS_CELLS
    }


class TestTheTruths:
    def test_a_fit_on_the_exact_sample_returns_identification_8(self) -> None:
        result = _quiet(study.fit, law.exact_frame(), study.IN_SAMPLE)
        for name in study.ESTIMANDS[study.IN_SAMPLE]:
            assert result[name].psi == pytest.approx(study.TRUTH[study.IN_SAMPLE][name], abs=1e-10)

    def test_a_cross_fitted_fit_on_the_exact_sample_lands_near_the_truth(self) -> None:
        """Each fold of the exact sample is not exact; the largest miss is 0.13 SE at 16k."""
        result = _quiet(study.fit, law.exact_frame(), study.CROSS_FITTED)
        for name in study.ESTIMANDS[study.CROSS_FITTED]:
            scale = study.EFFICIENCY_SD[name] / np.sqrt(study.PRIMARY_N)
            miss = result[name].psi - study.TRUTH[study.CROSS_FITTED][name]
            assert abs(miss) < 0.25 * scale, name

    def test_each_separable_risk_sits_away_from_both_static_risks(self) -> None:
        """The witness of the decomposition: several efficient SEs from either arm at 16k."""
        for label in law.SEPARABLE_ARMS:
            for cause in law.CAUSES:
                name = f"cif_regimen[{label}, {cause} @ t=3]"
                scale = study.EFFICIENCY_SD[name] / np.sqrt(study.PRIMARY_N)
                for arm in law.STATIC:
                    static = law.TRUTH[f"cif_regimen[{arm}, {cause} @ t=3]"]
                    assert abs(law.TRUTH[name] - static) > 3 * scale

    def test_no_reported_contrast_is_degenerate(self) -> None:
        assert min(study.EFFICIENCY_SD.values()) > 0.1

    def test_the_bounds_never_bind(self) -> None:
        # g >= 0.25 and each node keeps at least half its units uncensored, so the product
        # through three nodes is at least 0.25 * 0.5**3, far above the bound 1e-3.
        assert np.min(1.0 / law.exact_inverse(law.exact_frame())) >= 0.25 * 0.5**3
        assert study.G_BOUNDS[0] < 0.25 * 0.5**3


class TestTheRobustnessCells:
    def test_a_wrong_factor_with_the_exact_mechanism_has_the_truth_as_limit(
        self, limits: dict[str, float]
    ) -> None:
        truth = law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
        for cell, spec in properties.ROBUSTNESS_CELLS.items():
            if spec is not None and spec[2] == "positive":
                assert limits[cell] == pytest.approx(truth, abs=1e-8), cell

    def test_the_wrong_mechanism_cell_lands_near_the_truth(self, limits: dict[str, float]) -> None:
        """Cross-fitted on the exact sample, so only a small miss is checked (see the truths)."""
        miss = (
            limits["sep10_relapse_t3__mechanism_wrong"] - law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
        )
        assert abs(miss) < 0.25 * SCALE

    def test_in_sample_the_wrong_mechanism_changes_no_estimate(self) -> None:
        """Saturated in-sample hazards already solve every score, so the cell is cross-fitted."""
        from sklearn.dummy import DummyClassifier

        frame = law.sample(4_000, 11)
        wrong = _quiet(study.fit, frame, study.IN_SAMPLE, treatment_learner=DummyClassifier())
        right = _quiet(study.fit, frame, study.IN_SAMPLE)
        name = properties.ROBUSTNESS_ESTIMAND
        assert wrong[name].psi == pytest.approx(right[name].psi, abs=1e-12)

    def test_cross_fitted_the_wrong_mechanism_moves_the_estimate(self) -> None:
        """The nonzero witness: out of fold the treatment model changes the targeting."""
        frame = law.sample(4_000, 11)
        wrong = _quiet(properties.robustness_fit, frame, "sep10_relapse_t3__mechanism_wrong")
        right = _quiet(study.fit, frame, study.CROSS_FITTED)
        name = properties.ROBUSTNESS_ESTIMAND
        assert abs(wrong[name].psi - right[name].psi) > 1e-6

    def test_the_control_is_discriminated(self, limits: dict[str, float]) -> None:
        """Read at the efficient spread and at 1.5 times it, since a wrong fit's own spread
        can exceed the bound."""
        truth = law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
        gap = limits["sep10_relapse_t3__both_hazards_wrong"] - truth
        controls = [role for role in properties.ROBUSTNESS_ROLES.values() if role == "control"]
        for spread in (SCALE, 1.5 * SCALE):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.ROBUSTNESS_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(len(controls))

    @pytest.mark.parametrize(
        "cell",
        ["sep10_relapse_t3__event_hazard_wrong", "sep10_relapse_t3__competing_hazard_wrong"],
    )
    def test_the_targeting_step_moves_a_wrong_start_by_more_than_one_se(self, cell: str) -> None:
        """The untargeted control: the shifted start sits away from the truth."""
        shift_d, shift_y, _ = properties.ROBUSTNESS_CELLS[cell]  # type: ignore[misc]
        frame = law.exact_frame()
        inputs, start = law.engine_inputs(frame)
        zero = np.zeros_like(start.beta_d)
        shifted = properties.SeparableState(
            start.beta_d + (law.logit_shift(frame, law.SHIFT_D) if shift_d else zero),
            start.beta_y + (law.logit_shift(frame, law.SHIFT_Y) if shift_y else zero),
        )
        before = properties.influence(shifted, inputs, [properties.TARGET])[0][0]
        assert abs(before - law.TRUTH[properties.ROBUSTNESS_ESTIMAND]) > SCALE

    def test_one_target_cannot_pin_a_factor(self) -> None:
        """Design R2: fewer targets than a factor's (W, node) cells at a level."""
        targets = [properties.TARGET]
        assert len(targets) < 2 * len(law.HORIZONS)

    def test_the_engine_cells_read_the_exact_mechanism(self) -> None:
        frame = law.sample(400, 0)
        inputs, _ = law.engine_inputs(frame)
        np.testing.assert_allclose(inputs.inverse, law.exact_inverse(frame))


def _fits(cell: str, failures: list[str], phase: str = "iterative") -> pd.DataFrame:
    """One row per fit: ``""`` converged, any other failure stopped short at 1e-5."""
    return pd.DataFrame(
        {
            "cell": cell,
            "converged": [int(f == "") for f in failures],
            "failure": [f or np.nan for f in failures],
            "relative_score": [1e-12 if f == "" else 1e-5 for f in failures],
            "phase": phase,
        }
    )


class TestTheDeclaredSuspects:
    """The declared readings of fits that stop short, read by code (``regenerate.py``)."""

    def test_separated_fits_within_the_share_are_no_suspect(self) -> None:
        fits = _fits("c", ["cell_separated"] + [""] * 99)
        assert study.convergence_suspects(fits, {"c": True}) == []

    def test_each_declared_suspect_is_found(self) -> None:
        crowded = _fits("crowded", ["cell_separated"] * 2 + [""] * 98)
        stalled = _fits("stalled", ["newton_stalled"] + [""] * 99)
        high = _fits("high", ["cell_separated"] + [""] * 99)
        high.loc[0, "relative_score"] = 2e-4
        walked = _fits("walked", [""] * 10, phase="walk")
        left = _fits("left", [""] * 10)
        short = _fits("short", ["max_iter_reached"] + [""] * 9, phase="walk")
        fits = pd.concat([crowded, stalled, high, walked, left, short], ignore_index=True)
        flags = dict.fromkeys(["crowded", "stalled", "high"], True)
        flags |= dict.fromkeys(["walked", "left", "short"], False)
        found = study.convergence_suspects(fits, flags)
        assert [line.split(":")[0] for line in found] == [
            "crowded",
            "high",
            "left",
            "short",
            "stalled",
        ]

    def test_an_investigated_stall_passes_only_as_recorded(self) -> None:
        """A listed fit passes while it stalls at its recorded score; a listed fit that
        converges or stalls elsewhere is a suspect, and an unlisted stall still is one."""
        fits = _fits("c", ["newton_stalled", "newton_stalled"] + [""] * 98)
        fits["replicate"] = range(len(fits))
        fits.loc[0, "relative_score"] = 1.5e-10
        listed = {("c", 0): 1.5e-10}
        assert study.convergence_suspects(fits, {"c": True}, {**listed, ("c", 1): 1e-5}) == []
        assert study.convergence_suspects(fits, {"c": True}, listed) == [
            "c: 1 fits stopped short with ['newton_stalled']"
        ]
        moved = {("c", 0): 1.6e-10, ("c", 1): 1e-5}
        assert study.convergence_suspects(fits, {"c": True}, moved) == [
            "c: 1 investigated fits no longer stall as recorded"
        ]
        healed = {("c", 0): 1.5e-10, ("c", 1): 1e-5, ("c", 2): 1e-10}
        assert study.convergence_suspects(fits, {"c": True}, healed) == [
            "c: 1 investigated fits no longer stall as recorded"
        ]
        separated = fits.copy()
        separated.loc[0, "failure"] = "cell_separated"
        both = {**listed, ("c", 1): 1e-5}
        assert study.convergence_suspects(separated, {"c": True}, both) == [
            "c: 1 investigated fits no longer stall as recorded"
        ]

    def test_the_property_check_counts_fitted_cells_only(self) -> None:
        base = "cross_fitted_sep01_death_t3__correctly_specified"
        derived = "cross_fitted_sep01_death_t3__noise_control"
        fits = pd.concat(
            [_fits(base, [""] * 100), _fits(derived, ["newton_stalled"] * 100)],
            ignore_index=True,
        )
        assert properties.convergence_suspects(fits) == []

    def test_the_rerun_pilot_holds_no_suspect(self) -> None:
        """The readings were declared from the cross-fitted pilot labels.

        ``dr_mechanism_16k`` stopped short in 7 of 300 fits (2.3%), above the share; refitted
        with the right treatment model the same 7 draws still separate, so the label's rate is
        a fluctuation of its 300 draws, and the share reads the declared cells of 2,000 fits.
        """
        labels = ["cross_fitted_16k", "calibration_cross_fitted_16k"]
        pilot_rows = pd.read_csv(pilot.OUTPUT, low_memory=False)
        fits = (
            pilot_rows.loc[pilot_rows["label"].isin(labels)]
            .drop_duplicates(["label", "replicate"])
            .rename(columns={"label": "cell"})
        )
        assert study.convergence_suspects(fits, dict.fromkeys(labels, True)) == []

    def test_the_refit_check_counts_clipped_rows_that_pull_inward(self) -> None:
        """A targeted cell held at the lower clip over rows with deaths is counted, and so is
        one held at the upper clip over rows without; the same clip at a level no target reads
        is not, since that factor enters no score."""
        from cleverly.longitudinal.separable import SeparableState
        from tests.studies.separable_effects_separation import inward_rows

        inputs, start = law.engine_inputs(law.sample(4_000, 3))
        low = inputs.bounds[0]
        rows = (inputs.observed == 0) & inputs.at_risk[:, 2]
        deaths = int(inputs.event_d[rows, 2].sum())
        assert deaths > 0
        targeted = start.beta_d.copy()
        targeted[:, 2, 0] = low
        untargeted = start.beta_d.copy()
        untargeted[:, 2, 2] = low
        target = [properties.TARGET]
        assert inward_rows(SeparableState(targeted, start.beta_y), inputs, target) == deaths
        assert inward_rows(SeparableState(untargeted, start.beta_y), inputs, target) == 0
        survivors = int(np.sum(rows & inputs.rows_d()[:, 2] & (inputs.event_d[:, 2] == 0.0)))
        assert survivors > 0
        upper = start.beta_d.copy()
        upper[:, 2, 0] = inputs.bounds[1]
        assert inward_rows(SeparableState(upper, start.beta_y), inputs, target) == survivors

    def test_the_committed_run_holds_no_suspect(self) -> None:
        """The declared run's own artifacts pass every declared reading."""
        root = study.STUDY.artifacts
        assert study.primary_suspects(pd.read_csv(root / "convergence.csv")) == []
        rows = pd.read_parquet(root / "property-replicates.parquet")
        assert properties.convergence_suspects(rows) == []
        # Every listed stall is a fit of this run that stalled, and every stall is listed.
        primary = pd.read_csv(root / "convergence.csv").rename(columns={"scenario": "cell"})
        fitted = rows.loc[rows["cell"].isin(list(properties.CROSS_FITTED_CELLS))]
        fits = pd.concat([primary, fitted.drop_duplicates(["cell", "replicate"])])
        stalled = fits.loc[fits["failure"].eq("newton_stalled"), ["cell", "replicate"]]
        assert set(study.INVESTIGATED_STALLS) == {
            (str(cell), int(replicate)) for cell, replicate in stalled.itertuples(index=False)
        }


class TestTheConvergence:
    def test_the_convergence_file_holds_one_row_per_fit(self) -> None:
        frame = law.sample(4_000, 3)
        result = _quiet(study.fit, frame, study.IN_SAMPLE)
        names = study.ESTIMANDS[study.IN_SAMPLE]
        plug_in = dict.fromkeys(names, 0.0)
        rows = pd.DataFrame(
            study.result_rows(result, study.TRUTH[study.IN_SAMPLE], study.IN_SAMPLE, 7, plug_in)
        )
        written = study.extra_artifacts(rows)
        assert set(written) == set(study.STUDY.extra_artifacts)
        (table,) = written.values()
        assert list(table.columns) == ["scenario", "replicate", *study.CONVERGENCE_COLUMNS]
        assert table.to_dict("records") == [
            {
                "scenario": study.IN_SAMPLE,
                "replicate": 7,
                "converged": 1,
                "failure": "",
                "relative_score": rows["relative_score"].iloc[0],
                "phase": rows["phase"].iloc[0],
            }
        ]
        assert set(rows["phase"].iloc[0].split(",")) == {"walk"}

    def test_the_property_summary_counts_unconverged_fits(self) -> None:
        rows = pd.DataFrame(
            {
                "property": ["p"] * 3,
                "cell": ["c"] * 3,
                "converged": [1, 0, 1],
                "relative_score": [0.0, 1e-3, 0.0],
            }
        )
        summary = pd.DataFrame({"property": ["p"], "cell": ["c"]})
        out = properties.convergence_reading(summary, rows)
        assert out["nonconverged_fits"].tolist() == [1.0]

    def test_the_pilot_reads_each_fits_convergence(self) -> None:
        converged = _quiet(study.fit, law.sample(4_000, 3), study.IN_SAMPLE)
        assert study.convergence(converged)[:2] == (True, "")
        stalled = _quiet(properties.engine_fit, law.sample(4_000, 3), True, True)
        object.__setattr__(stalled.record, "converged", False)
        object.__setattr__(stalled.record, "failure", "newton_stalled")
        assert study.convergence(stalled)[:2] == (False, "newton_stalled")


class TestTheCells:
    def test_every_published_cell_is_declared(self) -> None:
        assert set(study.PROPERTY_CELLS["double_robustness"]) == set(properties.ROBUSTNESS_CELLS)
        declared = {cell for _, cell, _, _ in properties.declared_truths()}
        for family in study.PROPERTY_CELLS.values():
            assert set(family) <= declared

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
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)

    def test_the_seed_streams_are_this_studys_own(self) -> None:
        from tests.studies.evidence.registry import registered

        others = [record for record in registered() if record.slug != study.STUDY.slug]
        assert study.SEED not in {record.seed for record in others}
        assert study.RESAMPLING_SEED not in {record.resampling_seed for record in others}


PILOT = pd.read_csv(pilot.OUTPUT, low_memory=False) if pilot.OUTPUT.exists() else pd.DataFrame()
PRIMARY_PILOTS = {study.IN_SAMPLE: "in_sample_16k", study.CROSS_FITTED: "cross_fitted_16k"}
CALIBRATION_PILOTS = {
    "in_sample_sde_relapse_t3": "calibration_in_sample_16k",
    "cross_fitted_sep01_death_t3": "calibration_cross_fitted_16k",
}
ROBUSTNESS_PILOTS = {
    "sep10_relapse_t3__event_hazard_wrong": "dr_event_16k",
    "sep10_relapse_t3__competing_hazard_wrong": "dr_competing_16k",
    "sep10_relapse_t3__mechanism_wrong": "dr_mechanism_16k",
}


@pytest.mark.skipif(PILOT.empty, reason="the pilot has not been committed")
class TestTheSizes:
    """The declared sizes read the pilot by the shared rules.

    The pilot's unconverged fits (0.6% of the rerun cross-fitted draws, all
    ``cell_separated``) are read as drawn, as the declared run keeps them.
    """

    def test_the_pilot_lost_no_fit(self) -> None:
        assert PILOT["refusal"].fillna("").eq("").all()

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates`, one implementation.

        Clauses (a) to (c) read the pilot's covered counts and clause (d) its standardized
        biases, one group per scenario; both shares are
        :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`.
        """
        covered: dict[str, int] = {}
        bias_groups = []
        draws = set()
        for scenario, label in PRIMARY_PILOTS.items():
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
        _, name = properties.CALIBRATION_CELLS[cell]
        label = CALIBRATION_PILOTS[cell]
        rows = PILOT.loc[(PILOT["label"] == label) & (PILOT["estimand"] == name)]
        assert int(rows["n"].iloc[0]) == properties.CALIBRATION_N
        assert len(rows) == sizing.CALIBRATION_PILOT_DRAWS
        return rows

    def test_the_calibration_replicates_follow_the_pooled_rule(self) -> None:
        """The smallest multiple of 400 whose family-wise calibration pass reaches the share."""
        groups = []
        for cell in properties.CALIBRATION_LABELS:
            rows = self._calibration_pilot(cell)
            ratio = float(rows["std_error"].mean() / rows["estimate"].std(ddof=1))
            noise = ratio**2 / (2.0 * (len(rows) - 1))
            groups.append(sizing.PilotGroup(np.array([ratio]), np.array([[noise]])))
        tau = sizing.heterogeneity_bound(groups)

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
        """Each positive cell at its one-sided 80% upper pilot bias, the three together within
        the sized failure share, and no fewer replicates."""
        failing = 0.0
        for cell, label in ROBUSTNESS_PILOTS.items():
            assert properties.ROBUSTNESS_ROLES[cell] == "positive"
            rows = PILOT.loc[PILOT["label"] == label]
            assert int(rows["n"].iloc[0]) == properties.ROBUSTNESS_N
            error = rows["estimate"] - rows["truth"]
            bias = abs(float(error.mean() / error.std(ddof=1)))
            upper = bias + float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(rows))
            replicates = properties.ROBUSTNESS_REPLICATES
            passing = sizing.bias_equivalent_probability(MARGINS, upper, replicates)
            assert passing > 0.98, cell
            failing += 1.0 - passing
        assert failing <= sizing.SIZED_FAIL
        smaller = properties.ROBUSTNESS_REPLICATES - sizing.PROPERTY_STEP
        uppers = []
        for label in ROBUSTNESS_PILOTS.values():
            error = (
                PILOT.loc[PILOT["label"] == label, "estimate"]
                - law.TRUTH[properties.ROBUSTNESS_ESTIMAND]
            )
            uppers.append(
                abs(float(error.mean() / error.std(ddof=1)))
                + float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(error))
            )
        assert sizing.bias_equivalent_probability(MARGINS, max(uppers), smaller) <= 0.98
