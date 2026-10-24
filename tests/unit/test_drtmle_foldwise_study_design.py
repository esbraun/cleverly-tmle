"""Design of the registered study ``drtmle-foldwise`` (X83 plan section 7.3).

The numbers here are fixed before any run.  Each test names the part of the declaration it pins:
the record, the pairing of the primary draws with ``canonical-drtmle``, the pairing of the
multi-arm cell with the registered multi-arm cell, the fresh streams, the failure bound, the
budgets against their sizing rule, the leak control and its nonzero witness, and the
descriptions and clause bindings of every cell.  No test fits more than a few small samples,
and none runs the study.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from cleverly.validation._seeds import sample_seed_streams
from tests.studies import canonical_drtmle as canonical
from tests.studies import canonical_multi_arm_drtmle, multi_arm_drtmle_properties
from tests.studies import canonical_stratified_drtmle as stratified
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


def test_the_pilot_constants_are_the_declared_planning_summary() -> None:
    """The rule's inputs, pinned: a changed planning value must change this test too."""
    pilots = properties.PILOTS
    assert {name: pilots[name].bias for name in pilots} == {
        "primary/treatment_correct": (0.1032, -0.0469, 0.0939),
        "primary/outcome_correct": (0.1138, -0.0901, 0.0628),
        "primary/both_correct": (0.0659, -0.0969, -0.0035),
        "fold_locality": (0.1403, 0.3956),
        "benkeser_order": (0.0364,),
        "bivariate": (0.1427,),
        "bivariate_oc": (0.0298,),
        "k5": (0.0702,),
        "spline_qr": (0.1042,),
        "multi_arm": (0.0691,),
        "stratified": (0.0965, 0.0916, 0.0354, 0.0261),
        "both_wrong": (12.6089,),
    }
    assert {name: pilots[name].replicates for name in pilots} == {
        **dict.fromkeys(pilots, 400),
        "primary/treatment_correct": 1_200,
        "primary/outcome_correct": 800,
        "primary/both_correct": 800,
        "fold_locality": 2_000,
        "both_wrong": 800,
        "stratified": 800,
    }
    for name, pilot in pilots.items():
        correlation = np.asarray(pilot.correlation)
        assert correlation.shape == (len(pilot.rows), len(pilot.rows)), name
        assert np.allclose(correlation, correlation.T), name
        assert np.allclose(np.diag(correlation), 1.0), name
        assert len(pilot.se_ratio) == len(pilot.coverage), name
        assert pilot.se_ratio == () or name.startswith("primary/"), name
    assert pilots["fold_locality"].controls == ("pooled",)
    assert pilots["both_wrong"].controls == ("ate",)
    assert properties.REGIME[0] == "primary/treatment_correct"
    assert properties.PLANNING_DRAWS == 20_000
    assert properties.PLANNING_SEED == 2026102283
    assert properties.TAU_LEVEL == 0.80
    assert properties.MARGIN == 0.25


def test_the_regime_tau_is_its_q_profile_upper_bound() -> None:
    tau, mean, spread = properties.regime_tau()
    assert tau == pytest.approx(0.03797, abs=1e-5)
    assert mean == pytest.approx(0.09286, abs=1e-5)
    assert spread == pytest.approx(0.02622, abs=1e-5)
    # The posterior of each regime arm moves toward the pooled mean.
    draws = properties.planning_draws()
    for name in properties.REGIME:
        observed = properties.PILOTS[name].bias[0]
        planned = float(draws.bias[name][:, 0].mean())
        assert abs(planned - mean) < abs(observed - mean) + 0.002, name


def test_every_budget_is_the_output_of_the_declared_rule() -> None:
    assert set(properties.BUDGETS) == set(properties.UNITS)
    assert pytest.approx(0.95 ** (1 / len(properties.UNITS))) == properties.UNIT_TARGET
    unsized = []
    for unit, declared in properties.BUDGETS.items():
        replicates, value, sized = properties.budget(unit)
        assert declared == replicates, unit
        assert declared % properties.STEP == 0
        assert (value >= properties.UNIT_TARGET) == sized, unit
        if not sized:
            unsized.append(unit)
    assert tuple(unsized) == properties.FALLBACK
    assert properties.BUDGETS == {
        "primary": 6_000,
        "fold_locality": 4_400,
        "benkeser_order": 1_600,
        "bivariate": 4_000,
        "bivariate_oc": 1_600,
        "k5": 2_000,
        "spline_qr": 2_800,
        "both_wrong": 400,
        "multi_arm": 3_600,
        "stratified": 3_200,
    }
    assert properties.FALLBACK == ()
    assert properties.BUDGETS["primary"] >= properties.PRIMARY_FLOOR
    assert set(properties.GROUPS) | {"primary", "multi_arm"} == set(properties.UNITS)
    for name, group in properties.GROUPS.items():
        assert group.replicates == properties.BUDGETS[name]
        assert properties.UNITS[name][0] == (name,)


def test_the_family_wise_predictive_pass_reaches_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = properties.family_pass()
    assert value == pytest.approx(0.963, abs=0.003)
    assert value >= properties.STUDY_TARGET
    properties.assert_sized()
    # The guard is live: an open fallback refuses the run.
    monkeypatch.setattr(properties, "FALLBACK", ("fold_locality",))
    with pytest.raises(RuntimeError, match="fallback"):
        properties.assert_sized()


def test_the_unit_rule_is_monotone_and_reads_every_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for unit in properties.UNITS:
        values = [properties.unit_pass(unit, r) for r in (400, 1_200, 4_000, 8_000)]
        assert values == sorted(values), unit
    # A nonzero witness for each clause: dropping it raises the pass where it binds.
    full = properties.unit_pass("primary", 2_000)
    for clause in ("bias", "se_ratio", "coverage"):
        kept = tuple(c for c in properties.CLAUSES if c != clause)
        assert properties.unit_pass("primary", 2_000, kept) > full, clause
    assert properties.unit_pass("fold_locality", 2_000, ("displacement",)) == 1.0
    monkeypatch.setattr(properties, "FOLD_LOCALITY_DISPLACEMENT", 0.266)
    assert properties.unit_pass("fold_locality", 2_000, ("displacement",)) < 0.6
    # The control rows read the discrimination clause: a positive reading of them fails.
    assert properties.unit_pass("both_wrong", 400) == 1.0
    for name in ("both_wrong", "fold_locality"):
        pilot = properties.PILOTS[name]
        monkeypatch.setitem(properties.PILOTS, name, dataclasses.replace(pilot, controls=()))  # type: ignore[arg-type]
        assert properties.unit_pass(name, 2_000, ("bias",)) < 0.01, name


def test_stratified_treatment_correct_is_not_gated_by_the_pilot() -> None:
    gate = properties.STRATIFIED_TC_GATE
    assert gate == 0.12
    foldwise = [abs(pair[0]) for pair in properties.STRATIFIED_TC_PILOT.values()]
    assert max(foldwise) > properties.MARGIN > gate
    assert properties.STRATIFIED_SCENARIO == "outcome_correct"
    assert properties.GROUPS["stratified"].scenario == "outcome_correct"


def test_the_stratified_fit_is_canonical_stratified_fit_cleverly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cell calls the registered study's fit, with only ``reduced_crossfit`` replaced."""
    calls: list[tuple[object, ...]] = []

    def spy(frame: object, configuration: str, *, reduced_crossfit: str) -> str:
        calls.append((frame, configuration, reduced_crossfit))
        return "fitted"

    monkeypatch.setattr(stratified, "fit_cleverly", spy)
    frame = object()
    assert properties.stratified_fit(frame, runner.Fit()) == "fitted"  # type: ignore[arg-type]
    assert properties.stratified_fit(frame, runner.Fit(reduced_crossfit="pooled")) == "fitted"  # type: ignore[arg-type]
    assert calls == [
        (frame, "outcome_correct", "foldwise"),
        (frame, "outcome_correct", "pooled"),
    ]
    monkeypatch.undo()
    default = inspect.signature(stratified.fit_cleverly).parameters["reduced_crossfit"].default
    assert default == stratified.CONFIGURATION["reduced_crossfit"] == "pooled"


def test_the_leak_control_is_declared_and_its_threshold_resolves_the_pilot() -> None:
    leak = properties.LEAK
    assert leak.family == "fold_locality"
    assert leak.arms == ("foldwise", "pooled")
    assert leak.threshold == properties.FOLD_LOCALITY_DISPLACEMENT == 0.10
    foldwise, pooled = properties.PILOTS["fold_locality"].bias
    displacement = properties.LEAK_SPREAD_RATIO * pooled - foldwise
    assert displacement == pytest.approx(0.2656, abs=5e-4)
    assert 0.0 < leak.threshold < displacement / 2
    # Thirteen pilot standard errors (0.547 / sqrt(2000)) between the threshold and the pilot.
    assert (displacement - leak.threshold) / (0.547 / np.sqrt(2000)) > 13
    fits = {cell: (role, fit) for _, cell, role, fit in properties.GROUPS["fold_locality"].fits}
    assert fits["ate__foldwise"] == ("positive", runner.Fit())
    assert fits["ate__pooled"] == ("control", runner.Fit(reduced_crossfit="pooled"))
    assert properties.GROUPS["fold_locality"].n == properties.LEAK_N == 1_000
    assert properties.GROUPS["fold_locality"].scenario == "treatment_correct"


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
        "benkeser_order": ("treatment_correct", {"update_order": "benkeser"}),
        "bivariate": ("treatment_correct", {"reduction": "bivariate"}),
        "bivariate_oc": ("outcome_correct", {"reduction": "bivariate"}),
        "k5": ("treatment_correct", {"n_folds": 5}),
        "spline_qr": ("treatment_correct", {"reduced": "spline_qr"}),
    }
    for name, (scenario, fields) in changed.items():
        group = properties.GROUPS[name]
        ((family, cell, role, fit),) = group.fits
        assert (family, cell, role) == ("double_robustness", f"ate__{name}", "positive")
        assert group.scenario == scenario, name
        assert group.n == properties.PAPER_N == 3_000
        assert fit == runner.Fit(**fields), name
        differing = {field for field in vars(base) if getattr(fit, field) != getattr(base, field)}
        assert differing == set(fields), name
    frame, _ = runner.draw_paper("treatment_correct", 300, 11, n_folds=5)
    assert int(frame["fold"].max()) + 1 == 5
    again, _ = runner.draw_paper("treatment_correct", 300, 11)
    assert frame.drop(columns="fold").equals(again.drop(columns="fold"))


def test_the_both_wrong_control_is_the_primary_fit_with_both_nuisances_wrong() -> None:
    group = properties.GROUPS["both_wrong"]
    assert (group.law, group.scenario, group.n) == ("paper", "both_wrong", 3_000)
    assert group.fits == (("double_robustness", "ate__both_wrong", "control", runner.Fit()),)
    assert group.replicates == 400
    outcome, treatment = canonical._learners("both_wrong")
    assert outcome.columns is not None and treatment.columns is not None


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
            role = "control" if cell in ("ate__pooled", "ate__both_wrong") else "positive"
            tested, required = descriptions.cell(family, cell, slug=SLUG, role=role)
            assert tested and required


def test_every_used_clause_has_a_binding_row() -> None:
    bindings = pd.read_csv(
        ROOT / "tests" / "canonical" / "clause-bindings.csv", dtype=str, keep_default_na=False
    )
    own = bindings.loc[bindings["subject"] == SLUG]
    assert set(zip(own["property"], own["clause"], strict=True)) == {
        ("double_robustness", "bias discriminated"),
        ("double_robustness", "bias equivalent"),
        ("double_robustness", "se ratio in union band"),
        ("fold_locality", "bias discriminated"),
        ("fold_locality", "bias equivalent"),
        ("fold_locality", "least displacement over labels"),
    }
