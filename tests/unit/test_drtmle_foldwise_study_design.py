"""Design of the registered study ``drtmle-foldwise`` (X83 plan section 7.3).

The numbers here are fixed before any run.  Each test names the part of the declaration it pins:
the record, the pairing of the primary draws with ``canonical-drtmle``, the pairing of the
multi-arm cell with the registered multi-arm cell, the fresh streams, the failure bound, the
budgets against their sizing rule, the leak control and its nonzero witness, and the
descriptions and clause bindings of every cell.  No test fits more than a few small samples,
and none runs the study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly.validation._seeds import sample_seed_streams
from tests.studies import canonical_drtmle as canonical
from tests.studies import canonical_multi_arm_drtmle, multi_arm_drtmle_properties
from tests.studies import drtmle_foldwise as runner
from tests.studies import drtmle_foldwise_properties as properties
from tests.studies.evidence import descriptions, property_verdicts
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

SLUG = "drtmle-foldwise"


def _study() -> StudyRecord:
    (record,) = (study for study in registered() if study.slug == SLUG)
    return record


def test_the_record_registers_with_the_declared_design() -> None:
    record = _study()
    assert record is runner.STUDY
    assert record.reference is None
    assert record.comparator_state == "none"
    assert record.publication_policy == "gated"
    assert record.implementation == runner.IMPLEMENTATION
    assert record.n == runner.PRIMARY_N == 3_000
    assert record.replicates == runner.PRIMARY_REPLICATES == properties.BUDGETS["primary"]
    assert record.replicates >= 1_000
    assert record.scenarios == dict.fromkeys(canonical.SCENARIOS, canonical.ESTIMANDS)
    assert record.artifacts == ROOT / "tests" / "canonical" / "drtmle_foldwise"
    assert (record.artifacts / "__init__.py").is_file()
    assert (record.artifacts / "regenerate.py").is_file()
    assert record.runner() is runner
    assert record.properties() is properties
    assert record.property_cells == properties.property_cells()


def test_the_primary_fit_is_canonical_drtmle_with_foldwise_reduced_regressions() -> None:
    frame, _ = runner.draw_scenario("treatment_correct", 300, 0)
    subject = runner.estimator(frame["fold"].to_numpy(dtype=np.int64), "treatment_correct")
    assert subject.reduced_crossfit == "foldwise"
    assert subject.cross_fit is True
    assert subject.n_folds == canonical.N_FOLDS == 10
    assert subject.update_order == canonical.CONFIGURATION["update_order"]
    assert subject.reduction == canonical.CONFIGURATION["reduction"]
    assert subject.guard == tuple(canonical.CONFIGURATION["guard"])
    assert subject.max_outer == canonical.MAX_OUTER
    assert canonical.CONFIGURATION["reduced_crossfit"] == "pooled"
    assert runner.CONFIGURATION["reduced_crossfit"] == "foldwise"


@pytest.mark.parametrize("scenario", canonical.SCENARIOS)
def test_the_primary_draws_are_the_registered_canonical_drtmle_draws(scenario: str) -> None:
    """Replication ``r`` is ``canonical-drtmle``'s replication ``r``: rows and fold vector."""
    for replicate in (0, 1, canonical.PRIMARY_REPLICATES - 1):
        ours, truth = runner.draw_scenario(scenario, 400, replicate)
        theirs, _ = canonical.draw_scenario(scenario, 400, replicate)
        pd.testing.assert_frame_equal(ours, theirs)
        assert truth == canonical.truth()
    # Past the registered budget the same stream continues.
    beyond = canonical.PRIMARY_REPLICATES
    assert replicate_seed(runner.STUDY, scenario, beyond) == replicate_seed(
        canonical.STUDY, scenario, beyond
    )


def test_the_resampling_and_property_streams_are_fresh() -> None:
    others = [study for study in registered() if study is not runner.STUDY]
    taken = {study.seed for study in others} | {
        study.resampling_seed for study in others if study.resampling_seed is not None
    }
    assert runner.SEED == canonical.SEED
    assert runner.RESAMPLING_SEED not in taken
    assert runner.RESAMPLING_SEED != runner.SEED
    assert runner.STUDY.resampling_seed == runner.RESAMPLING_SEED
    primary = {
        replicate_seed(study, scenario, replicate)
        for study in others
        for scenario in study.scenarios
        for replicate in range(50)
    }
    own = {
        properties.sample_seed(group, replicate)
        for group in properties.GROUPS
        for replicate in range(50)
    }
    assert not own & primary
    assert len(own) == 50 * len(properties.GROUPS)
    assert stream_seed(runner.STUDY, "performance", "x") != stream_seed(
        canonical.STUDY, "performance", "x"
    )


def test_the_multi_arm_cell_reads_the_registered_multi_arm_draws_and_settings() -> None:
    registered_cell = properties._registered_multi_arm()
    cell = properties.multi_arm_cell()
    assert registered_cell.n == cell.n == properties.MULTI_ARM_N
    assert cell.seed == registered_cell.seed
    assert cell.estimand == registered_cell.estimand
    assert cell.replicates == properties.BUDGETS["multi_arm"]
    ours = properties.multi_arm_settings(cell)
    theirs = multi_arm_drtmle_properties._settings(registered_cell)
    assert theirs["reduced_crossfit"] == "pooled"
    assert ours["reduced_crossfit"] == "foldwise"
    assert {key for key in ours if key not in ("outcome_learner", "treatment_learner")} == {
        key for key in theirs if key not in ("outcome_learner", "treatment_learner")
    }
    for key in set(ours) - {"outcome_learner", "treatment_learner", "reduced_crossfit"}:
        assert repr(ours[key]) == repr(theirs[key]), key
    # The registered cell's sample seeds are the head of this cell's stream.
    budgets: dict[int, int] = {}
    for other in multi_arm_drtmle_properties.cells():
        budgets[other.seed] = max(budgets.get(other.seed, 0), other.replicates)
    registered_seeds = sample_seed_streams(budgets)[registered_cell.seed]
    own_seeds = sample_seed_streams({cell.seed: cell.replicates})[cell.seed]
    assert own_seeds[: registered_cell.replicates] == registered_seeds
    assert canonical_multi_arm_drtmle.STUDY.slug == "canonical-multi-arm-drtmle"


def test_the_failure_bound_is_one_percent_in_every_module() -> None:
    assert runner.FAILURE_BOUND == 0.01
    assert properties.FAILURE_BOUND == 0.01
    assert dict(runner.STUDY.failure_bound) == {runner.IMPLEMENTATION: 0.01}
    assert properties.PROPERTY_WITNESS is property_verdicts.SAMPLE_WITNESS
    (rule,) = properties.PROPERTY_RULES
    assert rule.families == frozenset({"fold_locality"})


def test_every_budget_is_the_output_of_the_declared_rule() -> None:
    assert set(properties.BUDGETS) == set(properties.UNITS)
    for unit, declared in properties.BUDGETS.items():
        replicates, _, _ = properties.budget(unit)
        assert declared == replicates, unit
        assert declared % properties.STEP == 0
    assert properties.BUDGETS["primary"] >= properties.PRIMARY_FLOOR
    assert set(properties.GROUPS) | {"primary", "multi_arm"} == set(properties.UNITS)
    for name, group in properties.GROUPS.items():
        assert group.replicates == properties.BUDGETS[name]
    gated = {row for rows, _ in properties.UNITS.values() for row in rows}
    assert gated == set(properties.PLANNING)


def test_the_unit_rule_is_monotone_and_the_target_shares_the_study_wise_rate() -> None:
    assert pytest.approx(0.95 ** (1 / len(properties.UNITS))) == properties.UNIT_TARGET
    for unit in properties.UNITS:
        for judge in (properties.unit_pass, properties.point_pass):
            values = [judge(unit, r) for r in (400, 1_200, 4_000)]
            assert values == sorted(values), unit
    assert properties.study_pass(point=True) >= properties.STUDY_TARGET
    assert 0.0 < properties.study_pass() < properties.study_pass(point=True)


def test_the_leak_control_is_declared_and_its_threshold_resolves_p1() -> None:
    leak = properties.LEAK
    assert leak.family == "fold_locality"
    assert leak.arms == ("foldwise", "pooled")
    assert leak.threshold == properties.FOLD_LOCALITY_DISPLACEMENT
    mean, spread = properties.LEAK_PAIRED
    assert 0.0 < leak.threshold < mean / 2
    # Five P1 standard errors between the threshold and P1's paired value.
    assert (mean - leak.threshold) / 0.025 > 5
    fits = {cell: (role, fit) for _, cell, role, fit in properties.GROUPS["fold_locality"].fits}
    assert fits["ate__foldwise"] == ("positive", runner.Fit())
    assert fits["ate__pooled"] == ("control", runner.Fit(reduced_crossfit="pooled"))
    assert properties.GROUPS["fold_locality"].n == properties.LEAK_N == 2_000
    assert spread > 0


def test_the_leak_pair_is_a_nonzero_witness_on_one_draw() -> None:
    """The two arms read one draw and differ, so a pooled wiring cannot pass the displacement."""
    frame, truth = runner.draw_paper("treatment_correct", 600, 7)
    foldwise = runner.fit_cleverly(frame, "treatment_correct", runner.Fit())
    pooled = runner.fit_cleverly(frame, "treatment_correct", runner.Fit(reduced_crossfit="pooled"))
    assert foldwise.estimates["ate"].psi != pooled.estimates["ate"].psi
    assert np.isfinite(truth["ate"])


def test_each_variant_changes_one_setting_of_the_primary_fit() -> None:
    base = runner.Fit()
    changed = {
        "benkeser_order": {"update_order": "benkeser"},
        "bivariate": {"reduction": "bivariate"},
        "k5": {"n_folds": 5},
        "adaptive_reduced": {"reduced": "spline"},
    }
    for name, fields in changed.items():
        ((_, _, role, fit),) = properties.GROUPS[name].fits
        assert role == "positive"
        assert fit == runner.Fit(**fields), name
        differing = {field for field in vars(base) if getattr(fit, field) != getattr(base, field)}
        assert differing == set(fields), name
    frame, _ = runner.draw_paper("treatment_correct", 300, 11, n_folds=5)
    assert int(frame["fold"].max()) + 1 == 5
    again, _ = runner.draw_paper("treatment_correct", 300, 11)
    assert frame.drop(columns="fold").equals(again.drop(columns="fold"))


def test_a_key_writes_the_cells_it_names() -> None:
    for name, group in properties.GROUPS.items():
        names = properties._cells_of(((name, 0, 5),))
        assert names == [f"{family}/{cell}" for family, cell, _, _ in group.fits]
    declared = {
        f"{family}/{cell}"
        for family, cells in runner.STUDY.property_cells.items()
        for cell in cells
    }
    produced = {
        name for group in properties.GROUPS for name in properties._cells_of(((group, 0, 5),))
    }
    assert declared == produced | {f"double_robustness/{properties.MULTI_ARM_CELL}"}


def test_every_cell_and_the_implementation_are_described() -> None:
    assert descriptions.implementation(runner.IMPLEMENTATION)
    for family, cells in runner.STUDY.property_cells.items():
        assert descriptions.claim(family)
        for cell in cells:
            tested, required = descriptions.cell(family, cell, slug=SLUG, role="positive")
            assert tested and required


def test_every_used_clause_has_a_binding_row() -> None:
    bindings = pd.read_csv(
        ROOT / "tests" / "canonical" / "clause-bindings.csv", dtype=str, keep_default_na=False
    )
    own = bindings.loc[bindings["subject"] == SLUG]
    assert set(zip(own["property"], own["clause"], strict=True)) == {
        ("double_robustness", "bias equivalent"),
        ("double_robustness", "se ratio in union band"),
        ("fold_locality", "bias discriminated"),
        ("fold_locality", "bias equivalent"),
        ("fold_locality", "least displacement over labels"),
    }
