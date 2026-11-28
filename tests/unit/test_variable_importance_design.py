"""The declared design of the ``variable-importance`` study (X125, design rev 2).

The study is declared and has not run. These checks hold its sizing rule, laws, seeds, cells,
gates and verdict directions to the design before the pilot, and run one property key and one
primary replicate so the path a runner takes is known to work.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.variable_importance import _bh_adjust, _null_draws, _step_down
from tests.studies import variable_importance as study
from tests.studies import variable_importance_law as law_module
from tests.studies import variable_importance_properties as properties
from tests.studies.evidence import descriptions
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.evidence.schema import REPLICATE_COLUMNS as PRIMARY_COLUMNS
from tests.studies.evidence.seeds import replicate_seed, stream_seed

# ------------------------------------------------------------------------ the sizing rule


def test_the_provisional_r_is_the_smallest_multiple_of_400_passing_at_095() -> None:
    replicates, probability = study.size()
    assert replicates == study.PROVISIONAL_REPLICATES == 3_200
    assert probability >= study.SIZING_TARGET
    assert study.family_pass(2_800) < study.SIZING_TARGET
    # The design's table: 0.890 at 2,800 and 0.957 at 3,200, at planning coverage 0.95.
    assert study.family_pass(2_800) == pytest.approx(0.890, abs=0.005)
    assert study.family_pass(3_200) == pytest.approx(0.957, abs=0.005)


def test_the_se_ratio_curve_is_anchored_at_2400_and_conservative_below() -> None:
    # The project curve reads 0.16 at 800, 0.80 at 1,600 and 0.95 at 2,400 (guide/studies.md).
    # The normal fit is anchored at 2,400 (kappa 0.756 in the design) and plans no better than
    # the project curve below it.
    assert pytest.approx(0.756, abs=0.005) == study.PLANNING_KAPPA
    assert study.se_ratio_pass(2_400, study.PLANNING_KAPPA) == pytest.approx(0.95, abs=1e-6)
    assert study.se_ratio_pass(1_600, study.PLANNING_KAPPA) <= 0.80
    assert study.se_ratio_pass(800, study.PLANNING_KAPPA) <= 0.16


def test_the_complete_null_band_passes_at_alpha_and_fails_an_inflation() -> None:
    for alpha in study.ALPHAS:
        low, high = alpha * (1 - study.DELTA), alpha * (1 + study.DELTA)
        assert study.pass_band(alpha, 3_200, low, high) >= 0.995
        assert study.pass_band(alpha * 1.5, 3_200, low, high) < 0.6


def test_the_cross_fitted_three_level_rows_are_not_sized() -> None:
    in_sample = study.cell_passes(3_200, configuration="in_sample")
    cross_fitted = study.cell_passes(3_200, configuration="cross_fitted")
    assert "power/independent_mixed/Z:2" in in_sample
    assert "power/independent_mixed/Z:2" not in cross_fitted
    assert set(in_sample) - set(cross_fitted) == {"power/independent_mixed/Z:2"}


def test_every_sized_cell_is_a_gated_cell_and_back() -> None:
    """The sizing product and the declared cells name the same gated cells."""
    for configuration in study.CONFIGURATIONS:
        sized = set(study.cell_passes(3_200, configuration=configuration))
        gated = set()
        for law in study.LAW_NAMES:
            for family, cell, role in study.law_cells(law, configuration):
                if role == study.DIAGNOSTIC:
                    continue
                parts = cell.split("__")
                if family == study.CALIBRATION:
                    row = parts[0].split("_", 1)[0]
                    gated.add(f"{family}/{row}")
                    gated |= {
                        f"{family}/{row}/{kind}"
                        for kind in ("shrunken_se_control", "noise_control")
                    }
                    continue
                if family == study.POWER:
                    name = next(n for n in study.ESTIMANDS if study.row_label(n) == parts[2])
                    suffix = "/one_sided" if len(parts) == 4 else ""
                    gated.add(f"{family}/{law}/{name}{suffix}")
                    continue
                alpha = properties.cell_alpha(cell)
                suffix = f"/{parts[3]}" if len(parts) == 4 else ""
                gated.add(f"{family}/{law}/{alpha}{suffix}")
        assert sized == gated


def _arm_readings(coverage: float, ratio: float, replicates: int = 4_000) -> list[dict]:
    rows = []
    for configuration, row in study.CALIBRATION_ARMS:
        common = {"law": "", "configuration": configuration, "alpha": np.nan, "row": row}
        rows += [
            {"kind": "coverage_by_configuration", "value": coverage, "replicates": replicates}
            | common,
            {"kind": "se_ratio", "value": ratio, "replicates": np.nan} | common,
            {"kind": "se_ratio_kappa_by_configuration", "value": 0.69, "replicates": np.nan}
            | common,
        ]
    return rows


def test_a_pilot_moves_fwer_planning_upward_only_and_can_stop_the_declaration() -> None:
    fwer = {"kind": "fwer", "law": "independent_null", "alpha": 0.05, "row": "", "value": 0.01}
    pilot = pd.DataFrame([fwer, *_arm_readings(0.95, 1.0, 800)])
    assert study.planning_fwer(pilot) == study.PLANNING_FWER
    assert study.declared_replicates(pilot)[0] == 3_200
    liberal = pilot.copy()
    liberal.loc[0, "value"] = 0.068
    assert study.planning_fwer(liberal)[("independent_null", 0.05)] == 0.068
    with pytest.raises(RuntimeError, match="stop and report"):
        study.declared_replicates(liberal)


def test_the_primary_floors_join_the_family_and_fit_it_at_1600() -> None:
    # Review B1: 120 gated primary coverage floors. At 1,000 replicates and the probe's
    # in-sample coverage they alone fail with probability about 0.12; at 1,600 about 0.0002.
    assert study.PRIMARY_REPLICATES == 1_600
    assert study.primary_pass(1_000) < 0.90
    assert study.primary_pass() >= 0.999
    without = study.family_pass(3_200) / study.primary_pass()
    assert without == pytest.approx(
        np.prod(
            [
                np.prod(list(study.cell_passes(3_200, configuration=name).values()))
                for name in study.CONFIGURATIONS
            ]
        )
    )


def test_an_arm_far_from_target_needs_the_pilot_extension() -> None:
    # Rev 3: the committed pilot read X1 in sample at coverage 0.935 and SE ratio 0.949 on 800
    # draws. At the at-target R its plug-in pass is far below 0.90, so the declaration waits
    # for the extension (guide/studies.md, "Calibration budgets across arms").
    pilot = study.read_pilot()
    assert pilot is not None
    with pytest.raises(RuntimeError, match="pilot extension"):
        study.declared_replicates(pilot)
    arms = study.read_arms(pilot)
    assert set(arms) == set(study.CALIBRATION_ARMS)
    assert study.plug_in_arm_pass(3_600, arms[("in_sample", "X1")]) < study.EXTENSION_TRIGGER


def test_the_extension_sizes_the_calibration_arms_by_the_pooled_predictive_rule() -> None:
    # Readings near the rev-3 probe (rows/X125/probes/coverage_rev3.py, 3,200 fresh draws per
    # configuration with the pilot's 800): R = 6,800 when pooled within each configuration
    # (review S1), where a perfectly calibrated set of arms would need far less, and the
    # unpooled rule more.
    extension = pd.DataFrame(_arm_readings(0.95, 1.0))
    perfect, _ = study.declared_replicates(study.read_pilot(), extension)
    assert perfect <= 4_000
    probe = {
        ("in_sample", "X1"): (0.9415, 0.975),
        ("in_sample", "X5"): (0.9424, 0.9856),
        ("cross_fitted", "X1"): (0.947, 0.996),
        ("cross_fitted", "X5"): (0.953, 1.012),
    }
    arms = {
        arm: study.ArmReading(coverage=c, se_ratio=q, kappa=0.6863, replicates=4_000)
        for arm, (c, q) in probe.items()
    }
    assert study.calibration_pass(6_000, arms) > study.calibration_pass(6_000, arms, pooled=False)
    fwer = study.planning_fwer(study.read_pilot())
    replicates, probability = study.size(fwer=fwer, kappa=0.6863, arms=arms)
    assert replicates == 6_800
    assert probability >= study.SIZING_TARGET
    assert study.family_pass(6_400, fwer=fwer, kappa=0.6863, arms=arms) < study.SIZING_TARGET
    # A shortfall the band can resolve only at a very large R stops the declaration.
    short = {arm: dataclasses.replace(arms[arm], se_ratio=0.945) for arm in arms}
    assert study.size(fwer=fwer, kappa=0.6863, arms=short)[0] is None


# ------------------------------------------------------------------------ the laws


def test_the_null_rows_are_exact_nulls_and_the_signals_match_the_review() -> None:
    truth = study.truths("nulls_beside_signals")
    assert [round(truth[name], 4) for name in ("X1", "X2", "X3", "X4")] == [
        0.1214,
        0.1205,
        0.1204,
        -0.1132,
    ]
    assert all(truth[name] == 0.0 for name in study.ESTIMANDS[4:])
    for law in study.LAW_NAMES:
        assert np.linalg.eigvalsh(study.exact_covariance(law)).min() > 0.0
    for law in ("independent_null", "negative_pairs_null"):
        assert all(study.is_null(law, name) for name in study.ESTIMANDS)


def test_the_witness_truths_differ_on_the_confounded_nulls() -> None:
    adjusted = study.truths(study.WITNESS_LAW)
    unadjusted = study.truths(study.WITNESS_LAW, adjust=False)
    moved = [
        name for name in study.ESTIMANDS if adjusted[name] == 0.0 and abs(unadjusted[name]) > 0.01
    ]
    assert moved, "the False run's truths equal the True run's on every null row"


def test_the_oracle_at_n_4000_keeps_the_planning_values_conservative() -> None:
    for law in study.GATED_LAWS:
        reading = law_module.oracle(
            law_module.LAWS[law],
            study.N,
            datasets=1_500,
            draws=2_000,
            seed=stream_seed(study.STUDY, "gaussian_oracle", law, str(study.N)),
        )
        for alpha in study.ALPHAS:
            assert reading[alpha]["max_t_fwer"] <= alpha * (1 + study.DELTA)
        for (owner, row), planned in study.PLANNING_POWER.items():
            if owner == law:
                assert reading[0.05]["max_t_power"][row] >= planned - 0.03


def test_the_fixed_multiplier_seed_puts_the_first_step_near_its_level() -> None:
    # Review R2-3: one draw matrix serves every replicate, so its Monte Carlo error is common.
    rate = law_module.fixed_draw_fwer(
        law_module.LAWS["independent_null"],
        draws=10_000,
        seed=0,
        fresh=100_000,
        fresh_seed=stream_seed(study.STUDY, "fixed_draw"),
    )
    assert 0.040 < rate < 0.060


# ------------------------------------------------------------------------ the procedures


def test_the_study_procedures_agree_with_cleverly_and_scipy() -> None:
    rng = np.random.default_rng(stream_seed(study.STUDY, "identity_test"))
    for _ in range(20):
        pvalues = rng.uniform(size=12) ** 3
        bh = study.rank_bh(pvalues)
        np.testing.assert_allclose(_bh_adjust(pvalues), bh, rtol=1e-12)
        if hasattr(stats, "false_discovery_control"):
            np.testing.assert_allclose(stats.false_discovery_control(pvalues), bh, rtol=1e-10)
            np.testing.assert_allclose(
                stats.false_discovery_control(pvalues, method="by"),
                study.by_adjusted(pvalues),
                rtol=1e-10,
            )
        covariance = rng.normal(size=(12, 12))
        correlation = law_module.correlation(covariance @ covariance.T + np.eye(12))
        draws = _null_draws(correlation, 2_000, 3)
        statistics = stats.norm.isf(pvalues / 2.0)
        mine = np.minimum(np.maximum(law_module.step_down(statistics, draws), pvalues), 1.0)
        np.testing.assert_allclose(_step_down(statistics, pvalues, draws), mine, rtol=1e-12)


def test_the_study_step_down_steps_down_and_is_monotone_in_rank() -> None:
    """A nonzero witness for the study's own loop against its single-step mutant."""
    rng = np.random.default_rng(stream_seed(study.STUDY, "step_down_witness"))
    correlation = law_module.correlation(study.exact_covariance("nulls_beside_signals"))
    draws = _null_draws(correlation, 4_000, 5)
    statistics = np.array([4.0, 3.5, 3.0, 2.6, *np.abs(rng.normal(size=8))])
    adjusted = law_module.step_down(statistics, draws)
    single = np.array([np.mean(draws.max(axis=1) >= value) for value in statistics])
    assert np.all(adjusted <= single + 1e-15)
    assert np.any(adjusted < single - 1e-3)
    order = np.argsort(-statistics)
    assert np.all(np.diff(adjusted[order]) >= 0.0)


# ------------------------------------------------------------------------ seeds


def test_every_seed_of_the_declared_run_is_distinct() -> None:
    record = study.STUDY
    primary = {
        replicate_seed(record, scenario, r)
        for scenario in study.SCENARIOS
        if scenario not in study.SCENARIO_SEED_OWNERS
        for r in range(study.PRIMARY_REPLICATES)
    }
    property_seeds = {
        properties.seed(law, r) for law in study.LAW_NAMES for r in range(study.MAXIMUM_REPLICATES)
    }
    pilot = {
        properties.seed(law, r, pilot=True)
        for law in study.LAW_NAMES
        for r in range(
            study.PILOT_REPLICATES + study.PILOT_EXTENSION_REPLICATES
            if law == study.CALIBRATION_LAW
            else study.PILOT_REPLICATES
        )
    }
    expected = (
        study.PRIMARY_REPLICATES * len(study.LAW_NAMES)
        + study.MAXIMUM_REPLICATES * len(study.LAW_NAMES)
        + study.PILOT_REPLICATES * len(study.LAW_NAMES)
        + study.PILOT_EXTENSION_REPLICATES
    )
    assert len(primary) + len(property_seeds) + len(pilot) == expected
    assert len(primary | property_seeds | pilot) == expected
    # Both configurations fit the same primary samples.
    for law in study.LAW_NAMES:
        assert replicate_seed(record, study.scenario_name(law, "cross_fitted"), 7) == (
            replicate_seed(record, study.scenario_name(law, "in_sample"), 7)
        )


def test_the_record_entropy_is_used_nowhere_else() -> None:
    mine = {study.SEED, study.RESAMPLING_SEED}
    for record in registered():
        assert not mine & {record.seed, record.resampling_seed}, record.slug
    pattern = re.compile(r"^PILOT_SEED(?::[^=]*)?=\s*([0-9_]+)", re.MULTILINE)
    for path in (ROOT / "tests" / "studies").glob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            assert int(match.group(1).replace("_", "")) not in mine, path.name
    allowed = {"variable_importance.py", Path(__file__).name}
    for path in (ROOT / "tests").rglob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "2026112601" not in text and "2026112602" not in text, path


# ------------------------------------------------------------------------ declared, not run


def test_the_declaration_is_the_sizing_rule_on_the_committed_pilots() -> None:
    pilot, extension = study.read_pilot(), study.read_extension()
    assert pilot is not None and extension is not None
    replicates, probability = study.declared_replicates(pilot, extension)
    assert study.DECLARED_REPLICATES == replicates == 12_000
    assert probability >= study.SIZING_TARGET
    assert set(extension.loc[extension["kind"] == "coverage_by_configuration", "replicates"]) == {
        float(study.PILOT_EXTENSION_REPLICATES)
    }
    assert study.SLUG not in {record.slug for record in registered()}
    assert properties.requested("independent_null") == 12_000
    assert properties.requested("collinear_pair") == study.REPORTED_REPLICATES
    assert (study.STUDY.artifacts / "regenerate.py").is_file()


def test_the_declared_files_fit_under_the_ceiling() -> None:
    # Each replicates file at the declared R, from the measured size of one formatted record.
    per_fit = 2_200 / 1e6
    for (_, law), _name in study.REPLICATES_FILES.items():
        extra = study.WITNESS_REPLICATES if law == study.WITNESS_LAW else 0
        assert per_fit * (properties.planned(law) + extra) < study.ARTIFACT_CEILING_MB


def test_the_record_lists_its_modules_and_cells() -> None:
    record = study.STUDY
    for module in record.modules:
        assert (ROOT / module).is_file(), module
    assert record.property_cells == study.property_cells()
    for family, cells in record.property_cells.items():
        for cell in cells:
            assert "." not in cell and ":" not in cell, (family, cell)
    assert set(record.scenarios) == set(study.SCENARIOS)
    assert record.reported_scenarios == study.REPORTED_SCENARIOS
    assert record.property_artifacts == tuple(study.REPLICATES_FILES.values())


def test_the_reported_families_are_diagnostic_everywhere() -> None:
    for configuration in study.CONFIGURATIONS:
        rows = [
            *(item for law in study.LAW_NAMES for item in study.law_cells(law, configuration)),
            *study.witness_cells(configuration),
        ]
        for family, cell, role in rows:
            if family in study.REPORTED_FAMILIES or cell.startswith("collinear_pair"):
                assert role == study.DIAGNOSTIC, (family, cell)


# ------------------------------------------------------------------------ verdicts


def _proportion_rows(family: str, cell: str, role: str, share: float, count: int) -> pd.DataFrame:
    hits = round(share * count)
    return pd.DataFrame(
        {
            "property": family,
            "cell": cell,
            "role": role,
            "replicate": range(count),
            "n": study.N,
            "requested_replicates": count,
            "failed_replicates": 0,
            "truth": 0.05,
            "estimate": np.linspace(0.0, 1.0, count),
            "std_error": 1.0,
            "covered": 0,
            "rejected": [1] * hits + [0] * (count - hits),
        }
    )


@pytest.mark.parametrize(
    ("family", "cell", "role", "share", "passed"),
    [
        ("fwer_calibration", "independent_null__in_sample__alpha05", "positive", 0.05, True),
        ("fwer_calibration", "independent_null__in_sample__alpha05", "positive", 0.08, False),
        ("fwer_calibration", "independent_null__in_sample__alpha05", "positive", 0.02, False),
        ("fwer_calibration", "independent_null__in_sample__alpha10", "positive", 0.10, True),
        ("family_wise_error", "dobriban_like__in_sample__alpha05", "positive", 0.03, True),
        ("family_wise_error", "dobriban_like__in_sample__alpha05", "positive", 0.08, False),
        (
            "family_wise_error",
            "independent_null__in_sample__alpha05__unadjusted",
            "control",
            0.42,
            True,
        ),
        (
            "family_wise_error",
            "independent_null__in_sample__alpha05__shrunken_se",
            "control",
            0.05,
            False,
        ),
        ("power", "independent_mixed__in_sample__X1", "positive", 0.95, True),
        ("power", "independent_mixed__in_sample__X1", "positive", 0.80, False),
        ("power", "independent_mixed__in_sample__X4__one_sided", "control", 0.0, True),
        ("power", "independent_mixed__in_sample__X4__one_sided", "control", 0.95, False),
    ],
)
def test_the_verdicts_and_the_rule_read_each_cell_in_its_direction(
    family: str, cell: str, role: str, share: float, passed: bool
) -> None:
    rows = _proportion_rows(family, cell, role, share, 3_200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = properties.summarize_properties(rows)
    row = summary.iloc[0]
    assert bool(row["passed"]) is passed
    rule = pf.rule_for(family, properties)
    assert rule is not None and rule.shared == "multiplicity_verdicts"
    ctx = pf.Context(record=study.STUDY, rows=rows, summary=summary)
    clauses = [clause for clause in rule.clauses if clause.applies(ctx, row)]
    assert len(clauses) == 1
    assert clauses[0].read(ctx, row).passed is passed


def test_the_differential_names_whole_gated_groups() -> None:
    cells = properties.differential_cells()
    families = {name.split("/", 1)[0] for name in cells}
    assert families == {
        "fwer_calibration",
        "family_wise_error",
        "power",
        "interval_calibration",
    }
    assert properties.DIFFERENTIAL_BUDGET == 1_000


# ------------------------------------------------------------------------ one key of each path


@pytest.fixture(scope="module")
def one_key() -> tuple[tuple, list[dict], list[dict]]:
    key = (
        (
            "nulls_beside_signals",
            0,
            1,
            properties.seed("nulls_beside_signals", 0, pilot=True),
            True,
            ("in_sample",),
        ),
    )
    rows, records = properties.fit_replication(key, properties._draw(key))
    return key, rows, records


def test_one_property_key_fills_exactly_its_cells(one_key) -> None:
    key, rows, records = one_key
    names = {f"{row['property']}/{row['cell']}" for row in rows}
    controls = {name for names in properties.DERIVED_CELLS.values() for name in names}
    assert names == set(properties._cells_of(key)) - controls
    assert all(set(row) == set(REPLICATE_COLUMNS) for row in rows)
    assert [record["adjust_for_other_candidates"] for record in records] == [True, False]
    assert len(records[0]) == 4 + 6 * 12 + 66


def test_the_controls_move_in_their_predicted_direction(one_key) -> None:
    _, rows, _ = one_key
    by_cell = {row["cell"]: row for row in rows}
    stem = "nulls_beside_signals__in_sample"
    # X4 has beta < 0: its two-sided max-T test rejects at n = 4,000 and the one-sided does not.
    assert by_cell[f"{stem}__X4"]["rejected"] == 1
    assert by_cell[f"{stem}__X4__one_sided"]["rejected"] == 0
    assert by_cell[f"{stem}__X4"]["truth"] < 0


def test_the_shrunken_control_rejects_at_least_as_often_on_the_same_draws() -> None:
    frame = law_module.LAWS["independent_null"].sample(
        study.N, properties.seed("independent_null", 1, pilot=True)
    )
    result = study.fit(frame, "in_sample")
    readings = study.check_identity(result)
    read = properties.statistics(result)
    controls = properties.control_pvalues(result, read)
    assert np.all(controls["shrunken_se"] <= readings["max_t"] + 1e-12)
    assert np.any(controls["shrunken_se"] < readings["max_t"])
    # The redrawn draws are the fit's own: the step-down on them gives back the shipped values.
    redrawn = properties.stepped(read["distance"], properties.null_draws(result))
    np.testing.assert_allclose(redrawn, readings["max_t"], rtol=1e-12)
    # The identity gate stops a run whose adjusted values were made on other draws.
    other = dataclasses.replace(result, multiplier_seed=1)
    with pytest.raises(AssertionError, match="identity"):
        study.check_identity(other)


def test_one_primary_replicate_writes_the_declared_rows() -> None:
    key = (study.scenario_name("independent_mixed", "in_sample"), 0, study.N)
    rows = study._replicate(key, study._draw(key))
    assert [row["estimand"] for row in rows] == list(study.ESTIMANDS)
    assert all(set(row) == set(PRIMARY_COLUMNS) for row in rows)


def test_the_controls_are_recomputed_from_the_replicates_artifact(one_key) -> None:
    """A re-summary reads the in-fit controls back from the stored record (review N8).

    The correlation is stored to eight digits, so the check is a tolerance, not an identity.
    """
    _, rows, records = one_key
    record = properties.formatted(pd.DataFrame(records[:1])).iloc[0]
    labels = [study.row_label(name) for name in study.ESTIMANDS]
    m = len(labels)
    correlation = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            value = float(record[f"r_{labels[i]}_{labels[j]}"])
            correlation[i, j] = correlation[j, i] = value
    pvalues = np.array([float(record[f"p_{label}"]) for label in labels])
    psi = np.array([float(record[f"psi_{label}"]) for label in labels])
    distance = stats.norm.isf(pvalues / 2.0)
    draws = _null_draws(correlation, 10_000, 0)
    shrunken = properties.stepped(distance / study.SHRUNKEN_FACTOR, draws)
    one_sided = properties.stepped(
        stats.norm.isf(stats.norm.sf(np.sign(psi) * distance) / 2.0), draws
    )
    by_cell = {row["cell"]: row for row in rows}
    stem = "nulls_beside_signals__in_sample"
    nulls = np.array([study.is_null("nulls_beside_signals", name) for name in study.ESTIMANDS])
    for alpha in study.ALPHAS:
        cell = f"{stem}__{study.alpha_label(alpha)}"
        assert by_cell[cell]["estimate"] >= shrunken[nulls].min() - 1e-3
    x4 = study.ESTIMANDS.index("X4")
    assert by_cell[f"{stem}__X4__one_sided"]["rejected"] == int(one_sided[x4] <= 0.05)
    maxt = np.array([float(record[f"maxt_{label}"]) for label in labels])
    np.testing.assert_allclose(properties.stepped(distance, draws), maxt, atol=1e-3)


def test_the_replicates_record_splits_by_configuration_under_the_size_ceiling(one_key) -> None:
    _, _, records = one_key
    frame = properties.formatted(pd.DataFrame(records))
    split = properties.split_records(frame)
    assert set(split) == set(study.REPLICATES_FILES.values())
    assert sum(len(part) for part in split.values()) == len(frame)
    for (configuration, law), name in study.REPLICATES_FILES.items():
        assert set(split[name]["configuration"]) <= {configuration}
        assert set(split[name]["law"]) <= {law}


def test_every_cell_scenario_and_estimand_is_described() -> None:
    """The evidence tables read their words from descriptions.py, so each key must resolve."""
    for family, cells in study.PROPERTY_CELLS.items():
        assert descriptions.claim(family)
        for cell in cells:
            tested, required = descriptions.cell(family, cell, slug=study.SLUG)
            assert tested and required
    for derived in properties.declared_cells():
        tested, required = descriptions.cell(derived.property, derived.cell, slug=study.SLUG)
        assert tested and required
    for scenario in study.SCENARIOS:
        assert descriptions.scenario(scenario, slug=study.SLUG)
    for name in (*(f"X{j}" for j in range(1, 11)), "Z:1", "Z:2"):
        assert descriptions.estimand(name, slug=study.SLUG)
