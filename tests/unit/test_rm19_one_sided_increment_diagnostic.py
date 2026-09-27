"""The RM19 localization design: its transcription, its rules, its seeds and its record."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student

from tests.canonical.drtmle import regenerate
from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm19_one_sided_increment import rtrans
from tests.diagnostics.rm19_one_sided_increment import run as rm19
from tests.diagnostics.rm19_one_sided_increment.rtrans import Switches, transcribe
from tests.studies import canonical_drtmle as study
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.manifest import ROOT
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The parts whose record is committed.  Phase 3 of RM19 adds each part here in the commit that
#: adds its files, which turns on the budget and rebuild checks below for that part.  Until then
#: the same tests assert that the part has no committed record, so no record can land without
#: its check.
RECORDED: tuple[str, ...] = ()

COMMITTED = shared.read_rows(study.STUDY.artifact("replicates.csv.gz"))


def _committed(scenario: str, replicate: int, implementation: str) -> pd.DataFrame:
    rows = COMMITTED.loc[
        (COMMITTED["scenario"] == scenario)
        & (COMMITTED["replicate"] == replicate)
        & (COMMITTED["implementation"] == implementation)
    ]
    return rows.set_index("estimand")


def _registered(replicate: int) -> tuple[pd.DataFrame, float, object]:
    seed = replicate_seed(study.STUDY, rm19.SCENARIO, replicate)
    return rm19.draw_payload(rm19.SCENARIO, study.PRIMARY_N, seed)


@pytest.fixture(scope="module")
def draw_one() -> tuple[pd.DataFrame, float]:
    """Registered ``treatment_correct`` draw 1, where R converges in 11 rounds, and its ``C``."""
    payload, _, result = _registered(1)
    return payload, float(result.estimates["ate"].psi)  # type: ignore[attr-defined]


# ------------------------------------------------------------------------ the declaration


def test_the_budget_and_the_declared_inputs_come_from_the_committed_rows() -> None:
    assert rm19.budget_rule() == 1_954
    assert rm19.REPLICATES == 2_000 >= rm19.budget_rule()
    assert round(rm19.MARGIN / 5.0, 6) == rm19.RESOLUTION
    assert round(float(student.ppf(0.995, 1e9)), 6) == rm19.Z
    equivalence = shared.read_rows(study.STUDY.artifact("equivalence.csv")).set_index(
        ["scenario", "estimand"]
    )
    assert round(float(equivalence.loc[(rm19.SCENARIO, "ate"), "mean_margin"]), 6) == rm19.MARGIN
    for scenario, declared in {rm19.SCENARIO: rm19.COMMITTED_SD, **rm19.CONTROL_SD}.items():
        rows = COMMITTED.loc[(COMMITTED["scenario"] == scenario) & (COMMITTED["estimand"] == "ate")]
        wide = rows.pivot(index="replicate", columns="implementation", values="estimate")
        paired = wide[study.STUDY.implementation] - wide[study.STUDY.reference]
        assert round(float(paired.std(ddof=1)), 6) == declared


def test_the_arms_are_the_declared_arms() -> None:
    labels = [arm.label for arm in rm19.FACTORIAL]
    assert labels == [f"T{j}{p}{s}" for j, p, s in itertools.product("01", repeat=3)]
    assert [arm.label for arm in rm19.arms("B", rm19.SCENARIO)] == [*labels, "T000+G"]
    assert [arm.label for arm in rm19.arms("A", "outcome_correct")] == [
        "T000",
        "T100",
        "T111",
        "T000+G",
    ]
    assert [arm.label for arm in rm19.arms("C", rm19.SCENARIO)] == ["T000"]
    assert rm19.SIZES == {"B": (3_000,), "C": (1_500, 6_000)}


def test_part_a_draws_the_registered_primary_samples() -> None:
    scenario, n, replicate, seed = rm19.draws("A", 1)[1]
    frame, _ = study.draw_from_seed(scenario, n, seed)
    registered, _ = study.draw_scenario(scenario, n, replicate)
    pd.testing.assert_frame_equal(frame, registered)
    assert len(rm19.draws("A")) == 3 * study.PRIMARY_REPLICATES


# ----------------------------------------------------------------------------- seeds


def test_every_fresh_seed_is_new_and_assigned_by_the_declared_rule() -> None:
    fresh = rm19.fresh_seeds()
    everything = [*fresh["B"], *fresh["C"]]
    assert len(fresh["B"]) == rm19.REPLICATES and len(fresh["C"]) == 2 * rm19.REPLICATES
    assert len(set(everything)) == len(everything)
    taken = rm18_seeds.binary_registered() | set(rm18_seeds.binary_seeds())
    assert taken.isdisjoint(everything)
    assert rm19.LAST_SEED in rm19.taken_seeds() and rm19.LAST_SEED not in everything
    labels = [*rm19.labels("B"), *rm19.labels("C")]
    # At the declared budget no label collides, so every seed is its label's own stream.
    assert everything == [stream_seed(study.STUDY, *label) for label in labels]
    # A smoke run takes the first draws of each size, with the declared seeds.
    assert [call[3] for call in rm19.draws("C", 2)] == [
        fresh["C"][0],
        fresh["C"][1],
        fresh["C"][rm19.REPLICATES],
        fresh["C"][rm19.REPLICATES + 1],
    ]


def test_a_forced_collision_moves_to_the_retry_label() -> None:
    label = rm19.labels("B")[0]
    natural = stream_seed(study.STUDY, *label)
    assert shared.fresh_seeds(study.STUDY, [label], {natural}) == [
        stream_seed(study.STUDY, *label, "retry", 1)
    ]
    assert shared.fresh_seeds(study.STUDY, [label], {natural - 1, natural + 1}) == [natural]


# ------------------------------------------------------------------- the transcription


@pytest.mark.parametrize("replicate", [1, 3])
def test_the_reference_arm_reproduces_the_committed_r_and_cleverly_rows(replicate: int) -> None:
    payload, _, result = _registered(replicate)
    reference = transcribe(payload)
    r_rows = _committed(rm19.SCENARIO, replicate, study.STUDY.reference)
    c_rows = _committed(rm19.SCENARIO, replicate, study.STUDY.implementation)
    cleverly = rm19.cleverly_row(result)
    for name in study.ESTIMANDS:
        assert abs(reference.estimates[name] - r_rows.loc[name, "estimate"]) <= 1e-9
        assert abs(reference.std_errors[name] - r_rows.loc[name, "std_error"]) <= 1e-9
        assert abs(cleverly[name] - c_rows.loc[name, "estimate"]) <= 1e-9
        assert abs(cleverly[f"se_{name}"] - c_rows.loc[name, "std_error"]) <= 1e-9
    assert reference.exit == "tolIC" and reference.guard_events == 0


def test_the_tilt_and_the_prime_are_nonzero_witnesses(
    draw_one: tuple[pd.DataFrame, float],
) -> None:
    payload, _ = draw_one
    reference = transcribe(payload).estimates["ate"]
    assert abs(transcribe(payload, Switches(joint_tilt=True)).estimates["ate"] - reference) > 1e-4
    assert abs(transcribe(payload, Switches(prime=True)).estimates["ate"] - reference) > 1e-4


def test_the_exit_switch_changes_the_exit_and_the_estimate(
    draw_one: tuple[pd.DataFrame, float],
) -> None:
    payload, _ = draw_one
    reference = transcribe(payload)
    exited = transcribe(payload, Switches(cleverly_exit=True))
    assert (reference.exit, reference.closing) == ("tolIC", 0)
    assert exited.exit == "tolerance" and exited.closing > 1
    assert abs(exited.estimates["ate"] - reference.estimates["ate"]) > 1e-8


def test_the_bracket_holds_on_draw_one_and_a_mutated_tilt_breaks_it(
    draw_one: tuple[pd.DataFrame, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, cleverly = draw_one
    full = Switches(True, True, True)
    assert abs(transcribe(payload, full).estimates["ate"] - cleverly) < 1e-5

    def mutated(a: np.ndarray, gn: list[np.ndarray], qrn: list[np.ndarray]) -> list[np.ndarray]:
        # The arm-0 column divided by g_1 rather than g_0.  A sign flip of that column would
        # not do: it reparameterizes the tilt and solves the same equation.
        upper = gn[1]
        design = np.column_stack([-qrn[0] / upper, qrn[1] / upper])
        offset = rtrans.trim_logit(upper, rtrans.LOGIT_GUARD)
        coef, _ = rtrans.glm_binomial((a == 1.0).astype(float), design, offset, np.zeros(2))
        tilted = np.clip(rtrans.expit(offset + design @ coef), rtrans.TOLG, 1.0 - rtrans.TOLG)
        return [1.0 - tilted, tilted]

    monkeypatch.setattr(rtrans, "fluctuate_g_joint", mutated)
    assert abs(transcribe(payload, full).estimates["ate"] - cleverly) > 1e-4


def _guarded_payload() -> pd.DataFrame:
    """A draw on which every ``A = 1`` row has a negative ``grn1``, so R skips equation (10)."""
    rng = np.random.default_rng(5)
    n = 400
    treatment = rng.binomial(1, 0.2, n).astype(float)
    return pd.DataFrame(
        {
            "Y": rng.binomial(1, 0.5, n).astype(float),
            "A": treatment,
            "fold": np.arange(n) % 5,
            "qn0": rng.uniform(0.3, 0.7, n),
            "qn1": rng.uniform(0.3, 0.7, n),
            "gn1": rng.uniform(0.75, 0.85, n),
        }
    )


def test_the_guards_fire_and_removing_them_changes_the_fit() -> None:
    payload = _guarded_payload()
    a = payload["A"].to_numpy()
    qn = [payload["qn0"].to_numpy(), payload["qn1"].to_numpy()]
    gn = [1.0 - payload["gn1"].to_numpy(), payload["gn1"].to_numpy()]
    signed, _ = rtrans.estimate_grn(a, qn, gn, rtrans.fold_rows(payload["fold"].to_numpy()))[1]
    assert np.all(signed[a == 1.0] < 0.0)
    guarded = transcribe(payload)
    unguarded = transcribe(payload, Switches(no_guard=True))
    assert guarded.guard_events > 0 and unguarded.guard_events == 0
    assert abs(guarded.estimates["ate"] - unguarded.estimates["ate"]) > 1e-3


def test_the_guard_arm_is_a_copy_only_where_no_guard_fired(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = {"guard_events": 0}

    def fake(payload: pd.DataFrame, switches: Switches) -> dict[str, object]:
        del payload
        return {"arm": switches.label, "ate": float(switches.no_guard), **rows}

    monkeypatch.setattr(rm19, "draw_payload", lambda *_: (pd.DataFrame(), 0.0, None))
    monkeypatch.setattr(rm19, "cleverly_row", lambda _: {"arm": "C"})
    monkeypatch.setattr(rm19, "_initial", lambda _: {})
    monkeypatch.setattr(rm19, "transcribed_row", fake)
    copied = rm19.fit_draw(("C", "outcome_correct", 3_000, 0, 1))
    assert [row["arm"] for row in copied] == ["C", "T000"]
    copied = rm19.fit_draw(("A", "outcome_correct", 3_000, 0, 1))
    assert copied[-1]["arm"] == "T000+G" and copied[-1]["ate"] == 0.0
    rows["guard_events"] = 2
    fitted = rm19.fit_draw(("A", "outcome_correct", 3_000, 0, 1))
    assert fitted[-1]["arm"] == "T000+G" and fitted[-1]["ate"] == 1.0


# ---------------------------------------------------------------------- the reading rules


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (Interval(0.0001, 0.0011), rm19.INCREMENT),
        (Interval(-0.0011, -0.0001), rm19.REVERSE),
        (Interval(-0.0005, 0.000696), rm19.NO_INCREMENT),
        (Interval(-0.000598, 0.000598), rm19.NO_INCREMENT),
        (Interval(-0.000599, 0.000599), rm19.UNRESOLVED),
        (Interval(0.0, 0.002), rm19.UNRESOLVED),
    ],
)
def test_the_primary_rule_and_each_boundary(interval: Interval, expected: str) -> None:
    assert rm19.primary_label(interval) == expected


ABOVE = Interval(0.0002, 0.0008)
BELOW = Interval(-0.0008, -0.0002)
ACROSS = Interval(-0.0003, 0.0003)
HELD = Interval(-0.00005, 0.00005)


@pytest.mark.parametrize(
    ("primary", "interval", "point", "bracket", "effects", "expected"),
    [
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": ABOVE, "P": ACROSS, "S": ACROSS},
            "attributed to J",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": ABOVE, "P": ABOVE, "S": ACROSS},
            "shared by J and P",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": ABOVE, "P": ABOVE, "S": ABOVE},
            "shared by J, P and S",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": BELOW, "P": ACROSS, "S": ACROSS},
            rm19.INTERACTION_ONLY,
        ),
        (rm19.INCREMENT, ABOVE, 5e-4, Interval(-0.0002, 0.00005), {"J": ABOVE}, rm19.UNEXPLAINED),
        (rm19.NO_INCREMENT, ACROSS, 1e-4, HELD, {"J": ABOVE}, rm19.NOTHING),
        (
            rm19.NO_INCREMENT,
            ACROSS,
            1e-4,
            Interval(0.00001, 0.00011),
            {"J": ABOVE},
            rm19.UNEXPLAINED,
        ),
        (rm19.REVERSE, BELOW, -5e-4, HELD, {"G": BELOW, "J": ACROSS}, "attributed to G"),
        (rm19.REVERSE, BELOW, -5e-4, HELD, {"J": ABOVE}, rm19.NO_LABEL),
        (rm19.UNRESOLVED, ACROSS, 2e-4, HELD, {"P": ABOVE}, "attributed to P"),
        (rm19.UNRESOLVED, ACROSS, -2e-4, HELD, {"P": ABOVE}, rm19.NO_LABEL),
        (rm19.UNRESOLVED, ACROSS, 0.0, HELD, {"P": ABOVE}, rm19.NO_LABEL),
        (rm19.UNRESOLVED, ACROSS, 2e-4, HELD, {"P": ACROSS}, rm19.NO_LABEL),
    ],
)
def test_the_localization_rule_reads_from_the_top(
    primary: str,
    interval: Interval,
    point: float,
    bracket: Interval,
    effects: dict[str, Interval],
    expected: str,
) -> None:
    assert rm19.localization_label(primary, interval, point, bracket, effects) == expected


def _synthetic(
    effect: dict[str, float] | None = None,
    *,
    residual: float = 0.0,
    guard: int = 0,
    replicates: int = rm19.REPLICATES,
) -> pd.DataFrame:
    """Part B rows in which each factor at level 1 adds its ``effect`` to ``ate``.

    ``C`` is ``T(1, 1, 1)`` plus ``residual``, so ``C - T000`` is the sum of the J, P and S
    effects plus ``residual``.  Each arm carries its own small noise.
    """
    rng = np.random.default_rng(11)
    effect = effect or {}
    base = rng.normal(0.02, 0.02, replicates)
    arms = {
        arm.label: base
        + sum(effect.get(name, 0.0) for name, key in rm19.FACTORS.items() if getattr(arm, key))
        + rng.normal(0.0, 1e-5, replicates)
        for arm in rm19.FACTORIAL
    }
    arms["T000+G"] = arms["T000"] + effect.get("G", 0.0)
    arms["C"] = arms["T111"] + residual + rng.normal(0.0, 1e-6, replicates)
    frames = [
        pd.DataFrame(
            {
                "part": "B",
                "scenario": rm19.SCENARIO,
                "n": 3_000,
                "replicate": np.arange(replicates),
                "arm": arm,
                "ate": values,
                "truth": 0.0,
                "iterations": 10,
                "exit": "tolIC",
                "closing": 0,
                "guard_events": guard if arm == "T000" else 0,
                "at_bound": 0,
                "score_max": 1e-9,
            }
        )
        for arm, values in arms.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _result(table: list[dict[str, object]], statistic: str) -> str:
    return str(next(row["result"] for row in table if row["statistic"] == statistic))


def test_the_factorial_contrasts_are_the_longhand_2_cubed_contrasts() -> None:
    rows = _synthetic({"J": 0.003, "P": -0.001, "S": 0.0005})
    wide = rows.pivot(index="replicate", columns="arm", values="ate")
    effects = rm19.main_effects(wide, guard=False)
    for name in ("J", "P", "S"):
        index = list(rm19.FACTORS).index(name)
        high = [f"T{''.join(c)}" for c in itertools.product("01", repeat=3) if c[index] == "1"]
        low = [f"T{''.join(c)}" for c in itertools.product("01", repeat=3) if c[index] == "0"]
        np.testing.assert_allclose(effects[name], wide[high].mean(axis=1) - wide[low].mean(axis=1))
    assert float(np.mean(effects["J"])) == pytest.approx(0.003, abs=1e-6)
    jp = rm19.interaction(wide, ("J", "P"))
    longhand = (
        sum(
            (1 if j == p else -1) * wide[f"T{j}{p}{s}"]
            for j, p, s in itertools.product("01", repeat=3)
        )
        / 4.0
    )
    np.testing.assert_allclose(jp, longhand)


@pytest.mark.parametrize(("guard", "k"), [(0, 3), (1, 4)])
def test_the_family_level_is_bonferroni_over_the_family(guard: int, k: int) -> None:
    rows = _synthetic({"J": 0.001}, guard=guard)
    table = rm19.factorial_table("B", rows, rm19.REPLICATES)
    effect = next(row for row in table if row["statistic"] == "main effect J")
    assert effect["result"] == f"level {1 - 0.01 / k:.6g}"
    wide = rows.pivot(index="replicate", columns="arm", values="ate")
    values = rm19.main_effects(wide, guard=bool(guard))["J"]
    half = (
        student.ppf(1 - 0.01 / k / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    assert effect["ci_upper"] - effect["value"] == pytest.approx(half, rel=1e-9)
    assert ("main effect G" in {row["statistic"] for row in table}) == bool(guard)


def test_the_factorial_table_reads_and_each_mutation_moves_it() -> None:
    table = rm19.factorial_table("B", _synthetic({"J": 0.001}), rm19.REPLICATES)
    assert _result(table, "C - T000") == rm19.INCREMENT
    assert _result(table, "localization") == "attributed to J"
    assert _result(table, "T111 - C") == "P-bracket holds"
    moved = rm19.factorial_table("B", _synthetic({"J": 0.001}, residual=0.0002), rm19.REPLICATES)
    assert _result(moved, "localization") == rm19.UNEXPLAINED
    assert _result(moved, "T111 - C") == "P-bracket fails"
    nothing = rm19.factorial_table("B", _synthetic(), rm19.REPLICATES)
    assert _result(nothing, "C - T000") == rm19.NO_INCREMENT
    assert _result(nothing, "localization") == rm19.NOTHING
    short = rm19.factorial_table("B", _synthetic({"J": 0.001}, replicates=50), rm19.REPLICATES)
    assert _result(short, "localization") == shared.SMOKE
    assert _result(short, "C - T000") == shared.SMOKE


def test_p_ctrl_reads_the_interval_and_the_declared_sd() -> None:
    rows = _synthetic(replicates=study.PRIMARY_REPLICATES)
    frames = []
    for scenario, spread in (("outcome_correct", 0.001), ("both_correct", 0.002)):
        wide = rows.loc[rows["arm"].isin(["C", "T000", "T111"])].copy()
        wide["scenario"] = scenario
        jitter = np.random.default_rng(3).normal(0.0, spread, study.PRIMARY_REPLICATES)
        t100 = wide.loc[wide["arm"] == "T000"].assign(arm="T100")
        t100["ate"] = t100["ate"].to_numpy() + jitter - jitter.mean()
        frames += [wide, t100]
    table = rm19.control_table(pd.concat(frames, ignore_index=True))
    results = [row["result"] for row in table if row["statistic"] == "T100 - T000"]
    # 0.001 is inside the outcome_correct bound 0.002223; 0.002 exceeds both_correct's 0.000869.
    assert results == ["P-ctrl holds", "P-ctrl fails"]


def test_the_scaling_rows_use_three_sizes_and_label_only_an_excluding_interval() -> None:
    frames = []
    for n, shift in ((1_500, 0.0), (6_000, 0.002)):
        rows = _synthetic({"J": shift})
        rows["n"] = n
        frames.append(rows.loc[rows["arm"].isin(["C", "T000"])])
    b_rows = _synthetic()
    table = rm19.scaling_table(pd.concat(frames, ignore_index=True), b_rows)
    assert [row["scope"] for row in table] == [
        f"{rm19.SCENARIO}, n = {n}" for n in (1_500, 3_000, 6_000)
    ]
    assert [row["result"] for row in table] == ["", "", "increment"]
    wide = frames[1].pivot(index="replicate", columns="arm", values="ate")
    values = np.sqrt(6_000) * (wide["C"] - wide["T000"]).to_numpy()
    half = (
        student.ppf(1 - 0.01 / 3 / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    assert table[2]["ci_upper"] - table[2]["value"] == pytest.approx(half, rel=1e-9)


def _write_part(
    directory: Path, part: str, rows: pd.DataFrame, validation: list[tuple[str, bool]]
) -> None:
    rows_path, validation_path, _ = shared.part_paths(directory, part)
    shared.write_table(rows, rows_path)
    shared.write_table(
        shared.validation_frame(
            shared.validation_row(part, check, (held, 0.0, 1)) for check, held in validation
        ),
        validation_path,
    )


def test_part_b_reads_only_after_v4_holds(tmp_path: Path) -> None:
    rows = _synthetic({"J": 0.001})
    _write_part(tmp_path, "B", rows, [("Part A V1", True)])
    table = rm19.table("B", tmp_path, 1)
    assert list(table["result"])[-1] == shared.NOT_VALIDATED
    _write_part(tmp_path, "V4", rows.iloc[:0], [("V4 R rows against T000", False)])
    assert list(rm19.table("B", tmp_path, 1)["result"])[-1] == shared.NOT_VALIDATED
    _write_part(tmp_path, "V4", rows.iloc[:0], [("V4 R rows against T000", True)])
    table = rm19.table("B", tmp_path, 1)
    localization = table.loc[table["statistic"] == "localization", "result"]
    assert list(localization) == ["attributed to J"]


def test_a_part_a_miss_stops_the_later_parts(tmp_path: Path) -> None:
    _write_part(tmp_path, "A", _synthetic().iloc[:0], [("V1 cleverly rows", False)])
    checks = rm19.prior_checks("C", tmp_path, smoke=True)
    assert not shared.validated(checks, "C")
    assert list(checks["check"]) == ["Part A V1 cleverly rows"]
    assert rm19.prior_checks("C", tmp_path / "missing", smoke=True).empty


@pytest.mark.parametrize("matched", [True, False])
def test_a_v4_exit_mismatch_stops_part_c(tmp_path: Path, matched: bool) -> None:
    _write_part(tmp_path, "A", _synthetic().iloc[:0], [("V1 cleverly rows", True)])
    _write_part(
        tmp_path,
        "V4",
        _synthetic().iloc[:0],
        [("V4 R exit status", matched), ("V4 R maxIter draws", False)],
    )
    checks = rm19.prior_checks("C", tmp_path, smoke=True)
    assert list(checks["check"]) == ["Part A V1 cleverly rows", "V4 R exit status"]
    assert shared.validated(checks, "C") == matched
    # Part B carries the Part A checks alone; its reading reads V4 in full.
    assert list(rm19.prior_checks("B", tmp_path, smoke=True)["check"]) == [
        "Part A V1 cleverly rows"
    ]


def test_a_declared_later_part_needs_the_pushed_part_a_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = "tests/diagnostics/rm19_one_sided_increment"
    assert rm19.pushed("A") == ()
    assert rm19.pushed("B") == (f"{directory}/a-validation.csv",)
    assert rm19.pushed("C") == (
        f"{directory}/a-validation.csv",
        f"{directory}/v4-validation.csv",
    )
    assert rm19.pushed("V4") == (f"{directory}/b-rows.csv.gz",)
    answers = {
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): "abc",
        ("rev-parse", "@{u}"): "abc",
        ("cat-file", "-e", f"@{{u}}:{directory}/a-validation.csv"): shared.UNKNOWN,
    }
    monkeypatch.setattr(shared, "_git", lambda *arguments: answers.get(arguments, ""))
    refused = shared.refusals(smoke=False, pushed=rm19.pushed("B"))
    assert refused == [f"{directory}/a-validation.csv is not in the pushed upstream"]


# ------------------------------------------------------------- V2 and V4 by exit status


def _exit_case(
    own_exit: str, score_max: float, shift: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One exact ``tolIC`` draw, and one draw whose T000 ``ate`` sits ``shift`` from R's."""
    transcribed = pd.DataFrame(
        {
            "scenario": rm19.SCENARIO,
            "replicate": [0, 1],
            "arm": "T000",
            "exit": ["tolIC", own_exit],
            **{name: [0.1, 0.1 + (shift if name == "ate" else 0.0)] for name in study.ESTIMANDS},
            **{f"se_{name}": [0.02, 0.02] for name in study.ESTIMANDS},
        }
    )
    reference = pd.DataFrame(
        [
            {
                "scenario": rm19.SCENARIO,
                "replicate": k,
                "estimand": name,
                "estimate": 0.1,
                "std_error": 0.02,
            }
            for k in (0, 1)
            for name in study.ESTIMANDS
        ]
    )
    scores = pd.DataFrame(
        {"scenario": rm19.SCENARIO, "replicate": [0, 1], "score_max": [5e-9, score_max]}
    )
    return transcribed, reference, scores


@pytest.mark.parametrize(
    ("own_exit", "score_max", "shift", "failing"),
    [
        ("cap", 2e-6, 1e-5, None),
        ("tolIC", 5e-9, 1e-8, "V2 tolIC draws"),
        ("cap", 5e-9, 0.0, "V2 exit status"),
        ("tolIC", 2e-6, 0.0, "V2 exit status"),
        ("cap", 2e-6, 2e-4, "V2 maxIter draws"),
    ],
)
def test_v2_and_v4_read_the_tolerance_of_each_exit_status(
    own_exit: str, score_max: float, shift: float, failing: str | None
) -> None:
    shared._NOTES.clear()
    rows = shared.validation_frame(
        rm19.compare_by_exit("A", "V2", *_exit_case(own_exit, score_max, shift))
    )
    failed = list(rows.loc[rows["result"] == shared.FAILS, "check"])
    assert failed == ([] if failing is None else [failing])
    counts = dict(zip(rows["check"], rows["compared"], strict=True))
    matched = own_exit == ("tolIC" if score_max <= 1e-8 else "cap")
    assert counts == {
        "V2 exit status": 2,
        "V2 tolIC draws": 1 + int(matched and own_exit == "tolIC"),
        "V2 maxIter draws": int(matched and own_exit == "cap"),
    }
    assert [
        f"V2: {counts['V2 maxIter draws']} maxIter draws, largest scaled difference "
        f"{rows.set_index('check').loc['V2 maxIter draws', 'largest_difference']:.3g}"
    ] == shared._NOTES
    shared._NOTES.clear()


def test_a_draw_on_one_side_only_stops_the_check() -> None:
    transcribed, reference, scores = _exit_case("tolIC", 5e-9, 0.0)
    with pytest.raises(RuntimeError, match="one side only"):
        rm19.compare_by_exit("A", "V2", transcribed.iloc[:1], reference, scores)


# ------------------------------------------------------------------ the R reference (V4)


def test_v4_calls_the_registered_runner_with_its_hashed_files() -> None:
    manifest = pd.read_json(study.STUDY.artifact("manifest.json"), typ="series")
    files = {
        path.relative_to(ROOT).as_posix() for path in regenerate.REFERENCE.files(regenerate.HERE)
    }
    assert files == set(manifest["reference_sha256"])
    assert regenerate.REFERENCE.image == "cleverly-drtmle-reference:538a3a2"
    # The hoist is result-neutral: no manifest hashes the regeneration script.
    for path in (ROOT / "tests" / "canonical").glob("*/manifest.json"):
        hashed = pd.read_json(path, typ="series")
        for group in ("study_module_sha256", "reference_sha256", "sha256"):
            assert "tests/canonical/drtmle/regenerate.py" not in dict(hashed.get(group) or {})


# -------------------------------------------------------------------- the committed record


@pytest.mark.parametrize("part", rm19.PARTS)
def test_each_part_is_recorded_only_with_its_checks(part: str) -> None:
    rows, validation, reading = shared.part_paths(rm19.HERE, part)
    if part not in RECORDED:
        assert not any(path.exists() for path in (rows, validation, reading)), part
        return
    record = shared.read_rows(validation)
    assert shared.validated(record, part), record.to_string()
    rebuilt = rm19.table(part, rm19.HERE, 1)
    pd.testing.assert_frame_equal(
        shared.as_committed(rebuilt), shared.read_rows(reading), check_dtype=False, rtol=1e-12
    )
    assert shared.SMOKE not in set(rebuilt["result"].dropna())


@pytest.mark.parametrize("part", rm19.PARTS)
def test_each_recorded_part_meets_its_declared_budget(part: str) -> None:
    rows_path = shared.part_paths(rm19.HERE, part)[0]
    if part not in RECORDED:
        assert not rows_path.exists(), part
        return
    rows = shared.read_rows(rows_path)
    if part == "V4":
        assert rows["replicate"].nunique() == rm19.V4_DRAWS
        return
    declared = {"A": study.PRIMARY_REPLICATES}.get(part, rm19.REPLICATES)
    counts = rows.groupby(["scenario", "n", "arm"]).size()
    assert counts.eq(declared).all(), counts.to_string()
    expected = {(scenario, n, k, seed) for scenario, n, k, seed in rm19.draws(part)}
    observed = set(zip(rows["scenario"], rows["n"], rows["replicate"], rows["seed"], strict=True))
    assert observed == expected
