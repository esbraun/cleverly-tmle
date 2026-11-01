"""X59: the long-horizon comparator inputs are the quantities the committed fits read.

The R runners re-implement no rule of the law or the plans: they read ratios, shifts, ``abar``
columns and folds that :mod:`tests.studies.long_horizon_reference` writes.  Each test here holds
one written input to ``cleverly``'s own value on a small fit, so a wrong mask, arm or fold
fails before any R process reads it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_
from tests.studies import canonical_long_horizon_categorical_ltmle as s2
from tests.studies import canonical_long_horizon_ltmle as s1
from tests.studies import canonical_long_horizon_sdr as s3
from tests.studies import long_horizon_common as common
from tests.studies import long_horizon_primary as primary
from tests.studies import long_horizon_reference as reference_
from tests.studies.canonical_longitudinal_sdr import check_ratios, plans_of
from tests.studies.evidence.registry import StudyRecord

MODULES = (s1, s2, s3)


def _scenario(module, name: str, n: int) -> primary.Scenario:  # type: ignore[no-untyped-def]
    scenario = next(s for s in module.SCENARIOS if s.name == name)
    return dataclasses.replace(scenario, n=n)


def _frame(scenario: primary.Scenario, seed: int = 11):  # type: ignore[no-untyped-def]
    return law_.draw(scenario.K, scenario.m, scenario.n, seed, scenario.law)


@pytest.mark.parametrize(
    ("module", "name"), [(s3, "sdr_binary_k3"), (s3, "sdr_policy_k5"), (s3, "sdr_categorical_k5")]
)
def test_the_written_ratios_are_the_weights_the_fit_divides_by(module, name) -> None:  # type: ignore[no-untyped-def]
    scenario = _scenario(module, name, 600)
    frame = _frame(scenario)
    result = primary.fit(frame, scenario, "SDR", bootstrap=False)
    declared = plans_.regimens(scenario.plans, scenario.K, scenario.m)
    fitted = plans_of(result, declared)
    assert {str(plan.label) for plan in fitted} == set(scenario.plans)
    for plan in fitted:
        check_ratios(
            result, plan, reference_.ratios(frame, str(plan.label), scenario.K, scenario.m)
        )


def test_a_ratio_off_by_one_node_fails_the_check() -> None:
    """A deliberate mutation: the check reads the ratios, it does not pass on any input."""
    scenario = _scenario(s3, "sdr_binary_k3", 600)
    frame = _frame(scenario)
    result = primary.fit(frame, scenario, "SDR", bootstrap=False)
    plan = next(
        p
        for p in plans_of(result, plans_.regimens(scenario.plans, scenario.K, scenario.m))
        if str(p.label) == "respond"
    )
    written = reference_.ratios(frame, "respond", scenario.K, scenario.m)
    written[:, 1] = written[:, 1] / 1.05
    with pytest.raises(AssertionError, match="node 2"):
        check_ratios(result, plan, written)


def test_the_written_folds_are_the_fit_s_folds() -> None:
    scenario = _scenario(s2, "categorical_k5_crossfit", 500)
    result = primary.fit(_frame(scenario), scenario, "LTMLE", bootstrap=False)
    written = reference_.folds(scenario.n, scenario.n_folds)
    assert np.array_equal(np.asarray(result.folds.assignment), written)
    assert len(set(written)) == 5
    assert not reference_.folds(10, 1).any()


@pytest.mark.parametrize(
    ("module", "name"), [(s1, "binary_k3"), (s2, "categorical_k5_mix"), (s3, "sdr_policy_k5")]
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.filterwarnings("ignore::tests.studies.fractional_glm.QuasiBinomialSeparationWarning")
def test_the_plugin_is_the_untargeted_sequential_regression(module, name) -> None:  # type: ignore[no-untyped-def]
    scenario = _scenario(module, name, 800)
    frame = _frame(scenario)
    for plan in scenario.plans:
        value, low, high = reference_.plugin(frame, plan, scenario.K, scenario.m)
        assert value == common.untargeted_plugin(frame, plan, scenario.K, scenario.m)
        assert 0.0 < low <= high < 1.0


@pytest.mark.parametrize(("module", "name"), [(s2, "categorical_k5"), (s3, "sdr_policy_k5")])
def test_the_oracle_density_is_the_plan_s_density(module, name) -> None:  # type: ignore[no-untyped-def]
    scenario = _scenario(module, name, 400)
    frame = _frame(scenario)
    for plan in scenario.plans:
        for t in range(1, scenario.K + 1):
            here = reference_.reached(frame, t)
            oracle = reference_.density(frame, plan, scenario.K, scenario.m, t)
            stated = reference_._reached_density(frame, plan, scenario.K, scenario.m, t)
            np.testing.assert_allclose(oracle[here], stated[here], rtol=0, atol=1e-15)


def test_a_policy_plan_takes_four_copies_that_reproduce_its_density() -> None:
    scenario = _scenario(s3, "sdr_policy_k5", 400)
    frame = _frame(scenario)
    columns = reference_.lmtp_columns(frame, scenario)
    for plan in scenario.plans:
        mask = int(columns[f"policy__{plan}"][0])
        assert mask == reference_.policy_mask(plan, scenario.K)
        copies = reference_.COPIES if mask else 1
        for t in range(1, scenario.K + 1):
            shifts = np.column_stack(
                [columns[f"shift__{plan}__{t}__{c + 1}"] for c in range(copies)]
            )
            share = np.column_stack([(shifts == k).mean(axis=1) for k in range(scenario.m)])
            here = reference_.reached(frame, t)
            density = reference_.density(frame, plan, scenario.K, scenario.m, t)
            np.testing.assert_allclose(share[here], density[here], atol=1e-15)
    assert reference_.policy_mask("mix", 5) and not reference_.policy_mask("low", 5)
    with pytest.raises(ValueError, match="multiple of"):
        reference_.allocate(np.array([[0.3, 0.7]]), 4)


def test_the_ltmle_columns_follow_the_plan_and_the_law() -> None:
    scenario = _scenario(s1, "binary_k5", 400)
    frame = _frame(scenario)
    columns = reference_.ltmle_columns(frame, scenario)
    for plan in ("never", "always", "respond", "switch"):
        arms = reference_.assigned(frame, plan, scenario.K, scenario.m)
        for t in range(1, scenario.K + 1):
            here = reference_.reached(frame, t)
            stated = reference_._reached_density(frame, plan, scenario.K, scenario.m, t)
            assert np.array_equal(arms[here, t - 1], np.argmax(stated[here], axis=1))
            assert np.array_equal(columns[f"abar__{plan}__{t}"], arms[:, t - 1])
    assert not any(key.startswith(("abar__coin__", "abar__coinlast__")) for key in columns)
    for t in range(1, scenario.K + 1):
        here = reference_.reached(frame, t)
        np.testing.assert_array_equal(
            columns[f"gA{t}"][here], frame.loc[here, f"g{t}_1"].to_numpy(float)
        )
        assert np.all((columns[f"gA{t}"] > 0) & (columns[f"gA{t}"] < 1))


def test_a_declared_mutation_changes_the_input(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario(s3, "sdr_binary_k3", 300)
    frame = _frame(scenario)
    plain = reference_.lmtp_columns(frame, scenario)["ratio__always__2"]
    monkeypatch.setenv(reference_.MUTATION_ENV, "gform_node2")
    mutated = reference_.lmtp_columns(frame, scenario)["ratio__always__2"]
    assert np.any(plain != mutated) and np.allclose(mutated * 1.05, plain)
    monkeypatch.setenv(reference_.MUTATION_ENV, "unknown")
    with pytest.raises(ValueError, match="not a declared mutation"):
        reference_.mutation()


@pytest.mark.filterwarnings("ignore::tests.studies.fractional_glm.QuasiBinomialSeparationWarning")
@pytest.mark.parametrize("module", MODULES, ids=[m.STUDY.slug for m in MODULES])
def test_the_sample_carries_every_column_its_runner_reads(module) -> None:  # type: ignore[no-untyped-def]
    scenario = module.SCENARIOS[0]
    sample = module.reference_sample(scenario.name, scenario.n, 0)
    columns = set(sample.frame.columns)
    assert {"W", "Y", "fold", "design__K", "design__m", "design__mutation"} <= columns
    assert set(sample.truths) >= set(
        module.COMPARATOR_DECLARATION["reference_scenarios"].get(scenario.name, ())
    )
    again = module.reference_sample(scenario.name, scenario.n, 0)
    assert sample.frame.equals(again.frame), "a re-drawn sample is not its seeded sample"
    assert not sample.frame.equals(module.reference_sample(scenario.name, scenario.n, 1).frame)


@pytest.mark.parametrize("module", MODULES, ids=[m.STUDY.slug for m in MODULES])
def test_the_comparator_declaration_validates(module) -> None:  # type: ignore[no-untyped-def]
    record: StudyRecord = dataclasses.replace(
        module.STUDY,
        reference=module.STUDY.pending_reference,
        pending_reference=None,
        pending_reason="",
        **module.COMPARATOR_DECLARATION,
    )
    assert record.paired_role == "descriptive"
    assert set(module.REFERENCE_TIERS) == set(record.reference_scenarios)
    for cell in module.PARITY.cells:
        assert cell.estimand in record.reference_scenarios[cell.scenario], cell


# ---- the read chunk of a reference-only run (review item 1) ----------------------------------
def _fitted_per_chunk(lines: int, replicates: int, chunk: int) -> list[tuple[int, bool]]:
    """``(groups fitted, whether the read was full)`` for each read of ``study_stream``.

    A read takes ``chunk`` lines after what was carried; every replicate it holds is fitted
    except the last, which is carried into the next read, until a read finds nothing
    (study_harness.R).
    """
    total = lines * replicates
    start, end, fitted = 0, 0, []
    while True:
        read = min(chunk, total - end)
        if read == 0:
            fitted.append((replicates - start // lines, False))
            return [(count, full) for count, full in fitted if count]
        end += read
        last = (end - 1) // lines
        fitted.append((last - start // lines, read == chunk))
        start = last * lines


def _declared_scenarios():  # type: ignore[no-untyped-def]
    from tests.studies import canonical_long_horizon_lmtp_parity as lmtp_parity
    from tests.studies import canonical_long_horizon_ltmle_parity as ltmle_parity

    for module in (s1, s2, s3, ltmle_parity, lmtp_parity):
        record = module.STUDY
        reference = str(record.pending_reference or record.reference)
        for name in record.reported(reference):
            yield module, name, record.n_of(name)


def test_an_archive_line_is_one_unit_so_a_replicate_holds_n_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The plug-in of a small sample can leave the Q bound; only the frame's length is read.
    monkeypatch.setattr(reference_, "plugins", lambda frame, scenario: {})
    for module, name, _ in _declared_scenarios():
        assert len(module.reference_sample(name, 400, 0).frame) == 400, name


@pytest.mark.parametrize("cores", [1, 2, 4, 8, 12])
def test_every_read_at_the_declared_sizes_keeps_every_worker_busy(cores: int) -> None:
    from tests.canonical import regenerate

    for _, name, n in _declared_scenarios():
        chunk = regenerate.chunk_lines(n, cores)
        fitted = _fitted_per_chunk(n, 30, chunk)
        assert sum(count for count, _ in fitted) == 30, name
        # Every full read fits at least one replicate per worker; only the archive's tail
        # holds fewer.
        assert min(count for count, full in fitted if full) >= cores, (name, fitted)


def test_the_harness_default_chunk_fits_one_replicate_at_a_time() -> None:
    # The control: the 64,000-line default holds less than two replicates of the large
    # scenarios, so their fits would run one at a time on any number of workers.
    largest = max(n for _, _, n in _declared_scenarios())
    assert max(count for count, _ in _fitted_per_chunk(largest, 30, 64_000)) <= 1


# ---- the policy screen's zero pattern (review item 12) -----------------------------------------
def _support(frame, plan: str, K: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    """``lmtp_9996b04_policy_adapter.R``'s support: some copy takes the observed arm, kept."""
    shifts = sorted(c for c in frame.columns if c.startswith(f"shift__{plan}__1__"))
    out = np.zeros((len(frame), K), dtype=bool)
    for t in range(1, K + 1):
        observed = frame[f"A{t}"].to_numpy(float)
        draws = np.column_stack(
            [frame[f"shift__{plan}__{t}__{c + 1}"].to_numpy(float) for c in range(len(shifts))]
        )
        taken = (draws == np.where(np.isnan(observed), -1.0, observed)[:, None]).any(axis=1)
        kept = ~np.isnan(observed) & (np.nan_to_num(frame[f"C{t}"].to_numpy(float)) == 1.0)
        out[:, t - 1] = kept & taken
    return out


def test_the_policy_screen_reads_a_zero_pattern_built_apart_from_the_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference_, "plugins", lambda frame, scenario: {})
    checked = 0
    for name in ("sdr_policy_k5", "sdr_policy_k5_mix"):
        frame = s3.reference_sample(name, 2_000, 0).frame
        K = int(frame["design__K"].iloc[0])
        for column in [c for c in frame.columns if c.startswith("policy__")]:
            plan = column.removeprefix("policy__")
            if not int(frame[column].iloc[0]):
                continue
            ratios = np.column_stack(
                [frame[f"ratio__{plan}__{t}"].to_numpy(float) for t in range(1, K + 1)]
            )
            support = _support(frame, plan, K)
            assert np.array_equal(ratios > 0, support), (name, plan)
            # The control: a ratio zeroed where the plan has mass is a disagreement.
            broken = ratios.copy()
            broken[np.argwhere(support)[0][0], np.argwhere(support)[0][1]] = 0.0
            assert np.logical_xor(broken > 0, support).sum() == 1
            checked += 1
    assert checked
