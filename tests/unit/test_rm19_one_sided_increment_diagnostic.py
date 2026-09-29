"""The RM19 localization design: its transcription, its rules, its seeds and its record."""

from __future__ import annotations

import functools
import inspect
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student

from cleverly.estimators import targeting
from cleverly.estimators.targeting import TargetingSpec
from cleverly.fluctuation import mechanism
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
RECORDED: tuple[str, ...] = ("A", "AV", "B", "V4", "C")

COMMITTED = shared.read_rows(study.STUDY.artifact("replicates.csv.gz"))
FULL = Switches(True, True, True, True)


def _committed(scenario: str, replicate: int, implementation: str) -> pd.DataFrame:
    rows = COMMITTED.loc[
        (COMMITTED["scenario"] == scenario)
        & (COMMITTED["replicate"] == replicate)
        & (COMMITTED["implementation"] == implementation)
    ]
    return rows.set_index("estimand")


@functools.cache
def _registered(
    replicate: int, scenario: str = rm19.SCENARIO
) -> tuple[pd.DataFrame, float, object]:
    """A registered draw, its payload and its ``C`` fit, fitted once per test session."""
    seed = replicate_seed(study.STUDY, scenario, replicate)
    return rm19.draw_payload(scenario, study.PRIMARY_N, seed)


def _cleverly(replicate: int, scenario: str = rm19.SCENARIO) -> dict[str, float]:
    result = _registered(replicate, scenario)[2]
    return {name: float(result.estimates[name].psi) for name in study.ESTIMANDS}  # type: ignore[attr-defined]


def _miss(out: rtrans.Transcribed, cleverly: dict[str, float]) -> float:
    return max(abs(out.estimates[name] - cleverly[name]) for name in study.ESTIMANDS)


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
    assert labels == [f"T{''.join(levels)}" for levels in itertools.product("01", repeat=4)]
    parts = [f"T1110[{part}]" for part in rtrans.K_PARTS]
    assert [arm.label for arm in rm19.arms("A", rm19.SCENARIO)] == [*labels, "T0000+G", *parts]
    assert [arm.label for arm in rm19.arms("B", rm19.SCENARIO)] == [*labels, "T0000+G"]
    assert [arm.label for arm in rm19.arms("A", "outcome_correct")] == [
        "T0000",
        "T1000",
        "T1111",
        "T0000+G",
    ]
    assert [arm.label for arm in rm19.arms("C", rm19.SCENARIO)] == ["T0000"]
    assert rm19.SIZES == {"B": (3_000,), "C": (1_500, 6_000)}
    assert rm19.INTERACTIONS == (
        ("J", "P"),
        ("J", "S"),
        ("J", "K"),
        ("P", "S"),
        ("P", "K"),
        ("S", "K"),
    )
    with pytest.raises(ValueError, match="part must be one of"):
        Switches(cleverly_numerics=True, part="outcome solver")


def test_part_a_draws_the_registered_primary_samples() -> None:
    scenario, n, replicate, seed = rm19.draws("A", 1)[1]
    frame, _ = study.draw_from_seed(scenario, n, seed)
    registered, _ = study.draw_scenario(scenario, n, replicate)
    pd.testing.assert_frame_equal(frame, registered)
    assert len(rm19.draws("A")) == 3 * study.PRIMARY_REPLICATES


def test_the_cleverly_side_constants_are_the_package_values() -> None:
    """D5: every constant the transcription copies from ``cleverly`` or the study equals it."""
    spec = TargetingSpec()
    config = study.CONFIGURATION
    assert spec.alpha == rtrans.ALPHA
    assert config["max_iter"] == rtrans.OUTCOME_MAX_ITER
    assert rtrans.NEGLIGIBLE == targeting._NEGLIGIBLE
    assert rtrans.STALL_FACTOR == targeting._STALL_FACTOR
    closing = inspect.signature(targeting._close_at_frozen_reductions).parameters["max_steps"]
    assert closing.default == rtrans.CLOSING_STEPS
    assert rtrans.LOGIT_GUARD == mechanism._LOGIT_GUARD
    assert rtrans.MAX_ITER == study.MAX_OUTER == config["max_outer"]
    assert (rtrans.TOLG, 1.0 - rtrans.TOLG) == study.G_BOUNDS
    # The registered fit's own settings, read back from a fit rather than from the source.
    fitted = _registered(1)[2].config.targeting_spec  # type: ignore[attr-defined]
    assert (fitted.tol, fitted.alpha, fitted.max_iter) == (
        rtrans.CLEVERLY_TOL,
        rtrans.ALPHA,
        rtrans.OUTCOME_MAX_ITER,
    )
    reduced = inspect.signature(mechanism.solve_bounded_mechanism).parameters["max_iter"]
    assert reduced.default == 50


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
    # The cached seeds are read-only, so no caller can move them for the next one.
    with pytest.raises(TypeError):
        fresh["B"] = ()  # type: ignore[index]


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
    r_rows = _committed(rm19.SCENARIO, replicate, str(study.STUDY.reference))
    c_rows = _committed(rm19.SCENARIO, replicate, str(study.STUDY.implementation))
    cleverly = rm19.cleverly_row(result)
    for name in study.ESTIMANDS:
        assert abs(reference.estimates[name] - r_rows.loc[name, "estimate"]) <= 1e-9
        assert abs(reference.std_errors[name] - r_rows.loc[name, "std_error"]) <= 1e-9
        assert abs(cleverly[name] - c_rows.loc[name, "estimate"]) <= 1e-9
        assert abs(cleverly[f"se_{name}"] - c_rows.loc[name, "std_error"]) <= 1e-9
    assert reference.exit == "tolIC" and reference.guard_events == 0


#: V5's pins: A 14 and A 25, the draws on which the review found the old bracket open;
#: ``treatment_correct`` 19, on which ``T(0, 0, 0, 0)`` records a guard event; and
#: ``outcome_correct`` 21, on which ``C``'s mechanism sits at its bound.
V5_DRAWS = [
    (rm19.SCENARIO, 14),
    (rm19.SCENARIO, 25),
    (rm19.SCENARIO, 19),
    ("outcome_correct", 21),
]


@pytest.mark.parametrize(("scenario", "replicate"), V5_DRAWS)
def test_v5_the_full_transcription_is_cleverly(scenario: str, replicate: int) -> None:
    payload = _registered(replicate, scenario)[0]
    out = transcribe(payload, FULL)
    assert _miss(out, _cleverly(replicate, scenario)) <= 1e-9
    if (scenario, replicate) == (rm19.SCENARIO, 19):
        assert transcribe(payload).guard_events > 0


@pytest.mark.parametrize(
    ("part", "scenario", "replicate"),
    [
        ("outcome solver", rm19.SCENARIO, 25),
        ("mechanism root", "outcome_correct", 21),
        ("reduction bounds", rm19.SCENARIO, 48),
        ("reduction learners", rm19.SCENARIO, 69),
    ],
)
def test_each_part_of_k_is_needed_for_v5(
    part: str, scenario: str, replicate: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate mutation per part of K: without it, ``T(1, 1, 1, 1)`` misses ``C``."""
    payload = _registered(replicate, scenario)[0]
    monkeypatch.setattr(
        Switches, "uses", lambda self, name: self.cleverly_numerics and name != part
    )
    assert _miss(transcribe(payload, FULL), _cleverly(replicate, scenario)) > 1e-9


def test_the_tilt_the_prime_and_k_are_nonzero_witnesses() -> None:
    payload = _registered(1)[0]
    reference = transcribe(payload).estimates["ate"]
    assert abs(transcribe(payload, Switches(joint_tilt=True)).estimates["ate"] - reference) > 1e-4
    assert abs(transcribe(payload, Switches(prime=True)).estimates["ate"] - reference) > 1e-4
    payload = _registered(25)[0]
    without = transcribe(payload, rm19.WITHOUT_K).estimates["ate"]
    assert abs(transcribe(payload, FULL).estimates["ate"] - without) > 1e-4


def test_the_exit_switch_changes_the_exit_and_the_estimate() -> None:
    payload = _registered(1)[0]
    reference = transcribe(payload)
    exited = transcribe(payload, Switches(cleverly_exit=True))
    assert (reference.exit, reference.closing) == ("tolIC", 0)
    assert exited.exit == "tolerance" and exited.closing > 1
    assert abs(exited.estimates["ate"] - reference.estimates["ate"]) > 1e-8


def test_a_mutated_tilt_breaks_the_bracket(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _registered(1)[0]
    cleverly = _cleverly(1)
    assert _miss(transcribe(payload, FULL), cleverly) <= 1e-9

    def mutated(qrn: list[np.ndarray], upper: np.ndarray) -> np.ndarray:
        # The arm-0 column divided by g_1 rather than g_0.  A sign flip of that column would
        # not do: it reparameterizes the tilt and solves the same equation.
        bounded = np.clip(upper, rtrans.TOLG, 1.0 - rtrans.TOLG)
        return np.column_stack([-qrn[0] / bounded, qrn[1] / bounded])

    monkeypatch.setattr(rtrans, "joint_design", mutated)
    assert abs(transcribe(payload, FULL).estimates["ate"] - cleverly["ate"]) > 1e-4


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
    split = rtrans.Split.of(payload["fold"].to_numpy())
    signed, _ = rtrans.estimate_grn(a, qn, gn, split)[1]
    assert np.all(signed[a == 1.0] < 0.0)
    guarded = transcribe(payload)
    unguarded = transcribe(payload, Switches(no_guard=True))
    assert guarded.guard_events > 0 and unguarded.guard_events == 0
    assert abs(guarded.estimates["ate"] - unguarded.estimates["ate"]) > 1e-3
    # K2: at k = 1 the outcome and mechanism steps are cleverly's, so no R guard acts.
    assert transcribe(payload, Switches(cleverly_numerics=True)).guard_events == 0


def test_the_mechanism_fallback_predicts_like_r() -> None:
    """F2: when both ``glm`` attempts fail, R predicts with the failed retry's fitted values.

    ``fluctuate-g-fallback.csv`` is R ``drtmle`` 1.1.2's ``fluctuateG`` in the pinned image on a
    constructed case (``fluctuate_g_fallback.R``).  Arm 1 separates, so R's coefficient is 0 and
    its prediction is not ``g``.
    """
    case = shared.read_rows(rm19.HERE / "fluctuate-g-fallback.csv")
    a = case["A"].to_numpy(dtype=float)
    gn = [case["g0"].to_numpy(), case["g1"].to_numpy()]
    log = rtrans._Log()
    out = rtrans.fluctuate_g_armwise(
        a, gn, [case["qr0"].to_numpy(), case["qr1"].to_numpy()], True, log
    )
    assert list(case["r_eps1"].unique()) == [0.0] and log.guard_events == 1
    np.testing.assert_allclose(out[0], case["r_g0"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(out[1], case["r_g1"], rtol=0.0, atol=1e-12)
    # The rule this replaced, a zero tilt of g, is far from R there.
    assert np.max(np.abs(np.clip(gn[1], rtrans.TOLG, 1.0 - rtrans.TOLG) - case["r_g1"])) > 0.5


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
    assert [row["arm"] for row in copied] == ["C", "T0000"]
    copied = rm19.fit_draw(("A", "outcome_correct", 3_000, 0, 1))
    assert copied[-1]["arm"] == "T0000+G" and copied[-1]["ate"] == 0.0
    rows["guard_events"] = 2
    fitted = rm19.fit_draw(("A", "outcome_correct", 3_000, 0, 1))
    assert fitted[-1]["arm"] == "T0000+G" and fitted[-1]["ate"] == 1.0


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
            {"J": ABOVE, "P": ACROSS, "S": ACROSS, "K": ACROSS},
            "attributed to J",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": ACROSS, "P": ACROSS, "S": ACROSS, "K": ABOVE},
            "attributed to K",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": ABOVE, "P": ABOVE, "S": ACROSS, "K": ACROSS},
            "shared by J and P",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": ABOVE, "P": ABOVE, "S": ABOVE, "K": ABOVE},
            "shared by J, P, S and K",
        ),
        (
            rm19.INCREMENT,
            ABOVE,
            5e-4,
            HELD,
            {"J": BELOW, "P": ACROSS, "S": ACROSS, "K": ACROSS},
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

    ``C`` is ``T(1, 1, 1, 1)`` plus ``residual``, so ``C - T0000`` is the sum of the J, P, S and
    K effects plus ``residual``.  Each arm carries its own small noise.
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
    arms["T0000+G"] = arms["T0000"] + effect.get("G", 0.0)
    for part in rm19.DECOMPOSITION:
        arms[part.label] = arms["T1110"] + effect.get(str(part.part), 0.0)
    arms["C"] = arms["T1111"] + residual + rng.normal(0.0, 1e-6, replicates)
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
                "guard_events": guard if arm == "T0000" else 0,
                "at_bound": 0,
                "score_max": 1e-9,
            }
        )
        for arm, values in arms.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _result(table: list[dict[str, object]], statistic: str) -> str:
    return str(next(row["result"] for row in table if row["statistic"] == statistic))


def test_the_factorial_contrasts_are_the_longhand_2_to_the_4_contrasts() -> None:
    rows = _synthetic({"J": 0.003, "P": -0.001, "S": 0.0005, "K": 0.002})
    wide = rows.pivot(index="replicate", columns="arm", values="ate")
    effects = rm19.main_effects(wide, guard=False)
    names = list(rm19.FACTORS)
    for name in names:
        index = names.index(name)
        high = [f"T{''.join(c)}" for c in itertools.product("01", repeat=4) if c[index] == "1"]
        low = [f"T{''.join(c)}" for c in itertools.product("01", repeat=4) if c[index] == "0"]
        np.testing.assert_allclose(effects[name], wide[high].mean(axis=1) - wide[low].mean(axis=1))
    assert float(np.mean(effects["K"])) == pytest.approx(0.002, abs=1e-6)
    for first, second in rm19.INTERACTIONS:
        i, j = names.index(first), names.index(second)
        longhand = (
            sum(
                (1 if levels[i] == levels[j] else -1) * wide[f"T{''.join(levels)}"]
                for levels in itertools.product("01", repeat=4)
            )
            / 8.0
        )
        np.testing.assert_allclose(rm19.interaction(wide, (first, second)), longhand)


@pytest.mark.parametrize(("guard", "k"), [(0, 4), (1, 5)])
def test_the_family_level_is_bonferroni_over_the_family(guard: int, k: int) -> None:
    rows = _synthetic({"J": 0.001}, guard=guard)
    table = rm19.factorial_table("B", rows, rm19.REPLICATES, guard=rm19.guard_fired(rows))
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
    table = rm19.factorial_table("B", _synthetic({"K": 0.001}), rm19.REPLICATES, guard=False)
    assert _result(table, "C - T0000") == rm19.INCREMENT
    assert _result(table, "localization") == "attributed to K"
    assert _result(table, "T1111 - C") == "bracket holds"
    moved = rm19.factorial_table(
        "B", _synthetic({"J": 0.001}, residual=0.0002), rm19.REPLICATES, guard=False
    )
    assert _result(moved, "localization") == rm19.UNEXPLAINED
    assert _result(moved, "T1111 - C") == "bracket fails"
    nothing = rm19.factorial_table("B", _synthetic(), rm19.REPLICATES, guard=False)
    assert _result(nothing, "C - T0000") == rm19.NO_INCREMENT
    assert _result(nothing, "localization") == rm19.NOTHING
    short = rm19.factorial_table(
        "B", _synthetic({"J": 0.001}, replicates=50), rm19.REPLICATES, guard=False
    )
    assert _result(short, "localization") == shared.SMOKE
    assert _result(short, "C - T0000") == shared.SMOKE


def test_the_k_decomposition_reads_each_part_against_t1110() -> None:
    effect = {part: 0.0001 * (index + 1) for index, part in enumerate(rtrans.K_PARTS)}
    table = rm19.decomposition_table(_synthetic(effect))
    assert [row["statistic"] for row in table] == [
        *(f"T1110[{part}] - T1110" for part in rtrans.K_PARTS),
        "T1111 - T1110",
    ]
    assert {row["result"] for row in table} == {rm19.SUPPLEMENTARY}
    for row, part in zip(table, rtrans.K_PARTS, strict=False):
        assert row["value"] == pytest.approx(effect[part], abs=1e-9)


def _controls(spread: dict[str, float], shift: dict[str, float] | None = None) -> pd.DataFrame:
    rows = _synthetic(replicates=study.PRIMARY_REPLICATES)
    frames = []
    for scenario, width in spread.items():
        wide = rows.loc[rows["arm"].isin(["C", "T0000", "T1111"])].copy()
        wide["scenario"] = scenario
        jitter = np.random.default_rng(3).normal(0.0, width, study.PRIMARY_REPLICATES)
        t1000 = wide.loc[wide["arm"] == "T0000"].assign(arm="T1000")
        t1000["ate"] = (
            t1000["ate"].to_numpy() + jitter - jitter.mean() + (shift or {}).get(scenario, 0.0)
        )
        frames += [wide, t1000]
    return pd.concat(frames, ignore_index=True)


def test_p_ctrl_reads_the_interval_and_the_declared_sd() -> None:
    # 0.001 is inside the outcome_correct bound 0.002223; 0.002 exceeds both_correct's 0.000869.
    table = rm19.control_table(_controls({"outcome_correct": 0.001, "both_correct": 0.002}))
    results = [row["result"] for row in table if row["statistic"] == "T1000 - T0000"]
    assert results == ["P-ctrl holds", "P-ctrl fails"]
    # A spread inside both bounds with a mean far from 0: the interval branch fails alone.
    table = rm19.control_table(
        _controls(
            {"outcome_correct": 0.0005, "both_correct": 0.0005},
            {"outcome_correct": 0.001},
        )
    )
    results = [row["result"] for row in table if row["statistic"] == "T1000 - T0000"]
    assert results == ["P-ctrl fails", "P-ctrl holds"]


def test_the_scaling_rows_use_three_sizes_and_label_only_an_excluding_interval() -> None:
    frames = []
    for n, shift in ((1_500, 0.0), (6_000, 0.002)):
        rows = _synthetic({"J": shift})
        rows["n"] = n
        frames.append(rows.loc[rows["arm"].isin(["C", "T0000"])])
    b_rows = _synthetic()
    table = rm19.scaling_table(pd.concat(frames, ignore_index=True), b_rows)
    assert [row["scope"] for row in table] == [
        f"{rm19.SCENARIO}, n = {n}" for n in (1_500, 3_000, 6_000)
    ]
    assert [row["result"] for row in table] == ["", "", "increment"]
    wide = frames[1].pivot(index="replicate", columns="arm", values="ate")
    values = np.sqrt(6_000) * (wide["C"] - wide["T0000"]).to_numpy()
    half = (
        student.ppf(1 - 0.01 / 3 / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    assert table[2]["ci_upper"] - table[2]["value"] == pytest.approx(half, rel=1e-9)
    # A short run labels every row, with or without a label of its own.
    short = rm19.scaling_table(
        pd.concat(frames, ignore_index=True).query("replicate < 50"), b_rows.query("replicate < 50")
    )
    assert {row["result"] for row in short} == {shared.SMOKE}


def test_the_context_rows_are_the_two_references_alone() -> None:
    rows = _synthetic(replicates=4)
    rows = pd.concat([rows.assign(n=1_500), rows.assign(n=6_000)], ignore_index=True)
    table = rm19.context_table(rows)
    assert [(row["scope"], row["statistic"]) for row in table] == [
        (f"{rm19.SCENARIO}, n = 1500", "BD-1 bias"),
        (f"{rm19.SCENARIO}, n = 6000", "committed rung bias"),
    ]
    bd1 = shared.read_rows(ROOT / "tests/diagnostics/rm18_boundary/bd-1-reading.csv")
    bias = bd1.loc[
        bd1["scope"].str.startswith(study.STUDY.slug) & (bd1["statistic"] == "bias")
    ].iloc[0]
    assert (table[0]["value"], table[0]["ci_lower"], table[0]["ci_upper"]) == (
        bias["value"],
        bias["ci_lower"],
        bias["ci_upper"],
    )
    assert rm19.context_table(rows.loc[rows["n"] == 6_000])[0]["statistic"] == "committed rung bias"


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
    _write_part(tmp_path, "V4", rows.iloc[:0], [("V4 R rows against T0000", False)])
    assert list(rm19.table("B", tmp_path, 1)["result"])[-1] == shared.NOT_VALIDATED
    _write_part(tmp_path, "V4", rows.iloc[:0], [("V4 R rows against T0000", True)])
    table = rm19.table("B", tmp_path, 1)
    localization = table.loc[table["statistic"] == "localization", "result"]
    assert list(localization) == ["attributed to J"]


@pytest.mark.parametrize("stopping", rm19.STOPPING)
@pytest.mark.parametrize("matched", [True, False])
def test_a_v4_stopping_miss_stops_the_part_a_reading(
    tmp_path: Path, matched: bool, stopping: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3, X1, X2 and F6: A folds in each V4 row that stops the design, selected by its class
    constant, and G's family reads every Part A row."""
    rows = _synthetic()
    control = rows.loc[rows["arm"] == "T0000"].assign(scenario="outcome_correct", guard_events=3)
    _write_part(tmp_path, "A", pd.concat([rows, control]), [("V1 cleverly rows", True)])
    _write_part(
        tmp_path,
        "V4",
        rows.iloc[:0],
        [(f"V4 R {stopping}", matched), (f"V4 R {rm19.SAME_MAX_ITER}", False)],
    )
    seen: dict[str, bool] = {}

    def factorial(*_: object, guard: bool, **__: object) -> list[dict[str, object]]:
        seen["guard"] = guard
        return []

    monkeypatch.setattr(rm19, "_committed_reference", lambda _: (None, None, None))
    monkeypatch.setattr(rm19, "max_iter_drift", lambda *_: (0, float("nan")))
    monkeypatch.setattr(rm19, "_conditioned", lambda *_: None)
    monkeypatch.setattr(rm19, "excused", lambda *_: (0, float("nan")))
    monkeypatch.setattr(rm19, "factorial_table", factorial)
    for name in ("decomposition_table", "control_table", "arm_summaries"):
        monkeypatch.setattr(rm19, name, lambda *_, **__: [])
    table = rm19.table("A", tmp_path, 1)
    assert f"V4 R {stopping}" in set(table["scope"])
    assert f"V4 R {rm19.SAME_MAX_ITER}" not in set(table["scope"])
    assert (list(table["result"])[-1] == shared.NOT_VALIDATED) == (not matched)
    assert seen == ({"guard": True} if matched else {})


def test_a_part_a_miss_stops_the_later_parts(tmp_path: Path) -> None:
    _write_part(tmp_path, "A", _synthetic().iloc[:0], [("V1 cleverly rows", False)])
    checks = rm19.prior_checks("C", tmp_path, smoke=True)
    assert not shared.validated(checks, "C")
    assert list(checks["check"]) == ["Part A V1 cleverly rows"]
    assert rm19.prior_checks("C", tmp_path / "missing", smoke=True).empty


@pytest.mark.parametrize("matched", [True, False])
def test_a_v4_stopping_miss_stops_part_c(tmp_path: Path, matched: bool) -> None:
    _write_part(tmp_path, "A", _synthetic().iloc[:0], [("V1 cleverly rows", True)])
    v4 = [(f"V4 R {kind}", matched) for kind in rm19.STOPPING]
    _write_part(tmp_path, "V4", _synthetic().iloc[:0], [*v4, (f"V4 R {rm19.SAME_MAX_ITER}", False)])
    checks = rm19.prior_checks("C", tmp_path, smoke=True)
    assert list(checks["check"]) == ["Part A V1 cleverly rows", *(check for check, _ in v4)]
    assert shared.validated(checks, "C") == matched
    # Part B carries the Part A checks alone; its reading reads V4 in full.
    assert list(rm19.prior_checks("B", tmp_path, smoke=True)["check"]) == [
        "Part A V1 cleverly rows"
    ]


def test_part_c_reads_the_part_b_rows_of_its_own_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: a smoke C reads B from its own directory, a declared C from the pushed record."""
    record = tmp_path / "record"
    record.mkdir()
    monkeypatch.setattr(rm19, "HERE", record)
    rows = _synthetic(replicates=4)
    c_rows = pd.concat([rows.assign(n=1_500), rows.assign(n=6_000)], ignore_index=True)
    c_rows = c_rows.loc[c_rows["arm"].isin(["C", "T0000"])]
    _write_part(tmp_path, "C", c_rows, [("Part A V1 cleverly rows", True)])
    with pytest.raises(RuntimeError, match="Part B must run first"):
        rm19.table("C", tmp_path, 1)
    _write_part(tmp_path, "B", rows, [("Part A V1 cleverly rows", True)])
    table = rm19.table("C", tmp_path, 1)
    scaling = table.loc[table["statistic"] == "sqrt(n) (C - T0000)", "scope"]
    assert list(scaling) == [f"{rm19.SCENARIO}, n = {n}" for n in (1_500, 3_000, 6_000)]
    # A declared C (2,000 rows per size) reads the record directory, here one with no B rows.
    full = _synthetic()
    declared = pd.concat([full.assign(n=1_500), full.assign(n=6_000)], ignore_index=True)
    _write_part(tmp_path, "C", declared.loc[declared["arm"].isin(["C", "T0000"])], [("x", True)])
    with pytest.raises(RuntimeError, match=str(rm19.HERE.name)):
        rm19.table("C", tmp_path, 1)


def test_a_declared_later_part_needs_the_pushed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = "tests/diagnostics/rm19_one_sided_increment"
    assert rm19.pushed("A") == ()
    assert rm19.pushed("AV") == (f"{directory}/a-rows.csv.gz",)
    assert rm19.pushed("B") == (f"{directory}/a-validation.csv",)
    assert rm19.pushed("C") == (
        f"{directory}/a-validation.csv",
        f"{directory}/b-rows.csv.gz",
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


def test_record_path_reads_the_record_for_a_declared_run(tmp_path: Path) -> None:
    """D7: one helper, shared with ``rm18_boundary``, picks the record or the scratch copy."""
    assert shared.record_path(rm19.HERE, "x.csv", tmp_path, smoke=True) == tmp_path / "x.csv"
    assert shared.record_path(rm19.HERE, "x.csv", tmp_path, smoke=False) == rm19.HERE / "x.csv"


# ----------------------------------------------------- V2 and V4 by round and conditioning


def _case(
    *,
    own_exit: str = "tolIC",
    rounds: tuple[int, int] = (10, 10),
    score_max: float = 5e-9,
    gap: float = 0.0,
    twin_shift: float = 0.0,
    twin_rounds: int | None = None,
    twin_exit: str | None = None,
    drift: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw 0 exact and well conditioned; draw 1 as named.

    Draw 1's ``T(0, 0, 0, 0)`` stops in ``rounds[0]`` with ``own_exit`` and sits ``gap`` from R in
    ``ate``; R stops in ``rounds[1]`` with ``score_max``.  Its twin sits ``twin_shift`` from
    the same run's refit and stops in ``twin_rounds`` with ``twin_exit``, by default T's own.
    The recorded ``ate`` of draw 1 then moves by ``drift``, away from the refit (X1).
    """
    own_rounds, r_rounds = rounds
    transcribed = pd.DataFrame(
        {
            "scenario": rm19.SCENARIO,
            "replicate": [0, 1],
            "arm": "T0000",
            "exit": ["tolIC", own_exit],
            "iterations": [10, own_rounds],
            **{name: [0.1, 0.1 + (gap if name == "ate" else 0.0)] for name in study.ESTIMANDS},
            **{f"se_{name}": [0.02, 0.02] for name in study.ESTIMANDS},
        }
    )
    twin = transcribed.assign(arm=rm19.TWIN)
    twin.loc[1, "ate"] += twin_shift
    twin.loc[1, "iterations"] = own_rounds if twin_rounds is None else twin_rounds
    twin.loc[1, "exit"] = own_exit if twin_exit is None else twin_exit
    twins = pd.concat([transcribed, twin], ignore_index=True)
    transcribed.loc[1, "ate"] += drift
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
    r_rows = pd.DataFrame(
        {
            "scenario": rm19.SCENARIO,
            "replicate": [0, 1],
            "rounds": [10, r_rounds],
            "score_max": [5e-9, score_max],
        }
    )
    return transcribed, reference, rm19.conditions(r_rows, r_rows, twins)


KINDS = (rm19.SAME_TOL_IC, rm19.SAME_MAX_ITER, rm19.SENSITIVE, rm19.DIFFERS)
CHECKS = [f"V2 {kind}" for kind in (rm19.REFIT, *KINDS, rm19.EXCUSED)]
REFIT_ROW, TOL_IC_ROW, MAX_ITER_ROW, SENSITIVE_ROW, DIFFERS_ROW, EXCUSED_ROW = CHECKS


@pytest.mark.parametrize(
    ("case", "kind", "failing", "excused"),
    [
        # Rule 3: same round and exit, not sensitive.
        ({"gap": 5e-10}, rm19.SAME_TOL_IC, None, 0),
        ({"gap": 2e-9}, rm19.SAME_TOL_IC, TOL_IC_ROW, 0),
        (
            {"own_exit": "cap", "rounds": (100, 100), "score_max": 2e-6, "gap": 1e-5},
            rm19.SAME_MAX_ITER,
            None,
            0,
        ),
        (
            {"own_exit": "cap", "rounds": (100, 100), "score_max": 2e-6, "gap": 2e-4},
            rm19.SAME_MAX_ITER,
            MAX_ITER_ROW,
            0,
        ),
        # A draw at maxIter on both sides keeps 1e-4 when it is sensitive, too.
        (
            {
                "own_exit": "cap",
                "rounds": (100, 100),
                "score_max": 2e-6,
                "gap": 2e-5,
                "twin_shift": 1e-6,
            },
            rm19.SAME_MAX_ITER,
            None,
            0,
        ),
        # Rule 4: same round and exit, sensitive.
        ({"gap": 5e-9, "twin_shift": 1e-9}, rm19.SENSITIVE, None, 1),
        ({"gap": 5e-9, "twin_shift": 1e-10}, rm19.SENSITIVE, SENSITIVE_ROW, 0),
        ({"gap": 2e-7, "twin_shift": 1e-6}, rm19.SENSITIVE, SENSITIVE_ROW, 0),
        ({"gap": 5e-9, "twin_rounds": 11}, rm19.SENSITIVE, SENSITIVE_ROW, 0),
        # Within rule 3's 1e-9, a sensitive draw passes and is not excused.
        ({"gap": 5e-10, "twin_shift": 2e-12}, rm19.SENSITIVE, None, 0),
        # Rule 5: another round or exit, excused only when the twin moves too.
        ({"rounds": (81, 85), "gap": 7.6e-5, "twin_rounds": 80}, rm19.DIFFERS, None, 1),
        ({"rounds": (81, 85), "gap": 0.0, "twin_shift": 1e-9}, rm19.DIFFERS, DIFFERS_ROW, 0),
        (
            {"own_exit": "cap", "rounds": (100, 74), "gap": 2e-6, "twin_rounds": 90},
            rm19.DIFFERS,
            None,
            1,
        ),
        (
            {"own_exit": "cap", "rounds": (100, 74), "gap": 2e-6, "twin_exit": "tolIC"},
            rm19.DIFFERS,
            None,
            1,
        ),
        ({"rounds": (81, 85), "gap": 2e-4, "twin_rounds": 80}, rm19.DIFFERS, DIFFERS_ROW, 0),
        (
            {"own_exit": "cap", "rounds": (100, 100), "score_max": 5e-9, "twin_rounds": 90},
            rm19.DIFFERS,
            None,
            1,
        ),
    ],
)
def test_v2_and_v4_read_each_rule_by_round_and_conditioning(
    case: dict[str, object], kind: str, failing: str | None, excused: int
) -> None:
    transcribed, reference, conditioned = _case(**case)  # type: ignore[arg-type]
    classes = rm19.classify(transcribed, reference, conditioned).set_index("replicate")
    assert classes.loc[0, "class"] == rm19.SAME_TOL_IC and classes.loc[0, "passes"]
    assert classes.loc[1, "class"] == kind
    before = len(shared.notes())
    rows = shared.validation_frame(
        rm19.compare_by_exit("A", "V2", transcribed, reference, conditioned)
    )
    assert list(rows["check"]) == CHECKS
    # One excused draw of two is above rule 6's 1%, so its limit row fails beside the class.
    assert list(rows.loc[rows["result"] == shared.FAILS, "check"]) == (
        ([] if failing is None else [failing]) + ([EXCUSED_ROW] if excused else [])
    )
    counts = dict(zip(rows["check"], rows["compared"], strict=True))
    assert sum(counts[f"V2 {name}"] for name in KINDS) == 2 and counts[f"V2 {kind}"] >= 1
    assert counts[REFIT_ROW] == counts[EXCUSED_ROW] == 2
    count, mean = rm19.excused(transcribed, reference, conditioned)
    assert count == excused
    gap = float(case.get("gap", 0.0))  # type: ignore[arg-type]
    assert (mean == pytest.approx(gap)) if excused else np.isnan(mean)
    notes = list(shared.notes()[before:])
    assert notes[-1] == f"V2: {excused} excused draws, signed mean ate difference {mean:.3g}"
    assert notes[0].startswith(f"V2: {counts[MAX_ITER_ROW]} maxIter draws")


def test_a_draw_without_its_round_count_refit_or_twin_stops_the_check() -> None:
    transcribed, reference, conditioned = _case()
    with pytest.raises(RuntimeError, match="one side only"):
        rm19.compare_by_exit("A", "V2", transcribed.iloc[:1], reference, conditioned)
    r_rows = conditioned[["scenario", "replicate", "rounds", "score_max"]]
    twins = pd.concat([transcribed, transcribed.assign(arm=rm19.TWIN)], ignore_index=True)
    missing = "no R round count, no refit or no twin"
    with pytest.raises(RuntimeError, match=missing):
        rm19.conditions(r_rows, r_rows.iloc[:1], twins)
    for arm in ("T0000", rm19.TWIN):
        with pytest.raises(RuntimeError, match=missing):
            rm19.conditions(r_rows, r_rows, twins.loc[twins["arm"] != arm])


@pytest.mark.parametrize("drift", [0.0, 1e-8])
def test_a_recorded_row_that_drifted_from_the_refit_stops_the_check(drift: float) -> None:
    """X1: the recorded ``T(0, 0, 0, 0)`` moves by 1e-8 between runs.  Compared with the old
    rule's recorded-row twin, that drift read as sensitivity; the refit row now fails."""
    transcribed, reference, conditioned = _case(drift=drift)
    rows = shared.validation_frame(
        rm19.compare_by_exit("A", "V2", transcribed, reference, conditioned)
    )
    failed = list(rows.loc[rows["result"] == shared.FAILS, "check"])
    refit = rows.set_index("check").loc[REFIT_ROW]
    assert float(refit["largest_difference"]) == pytest.approx(drift)
    # The drift also sits 1e-8 from R on a draw the twin calls well conditioned: rule 3 fails.
    assert failed == ([] if drift == 0.0 else [REFIT_ROW, TOL_IC_ROW])
    assert rm19.excused(transcribed, reference, conditioned)[0] == 0
    # A refit in another round fails the row as well, even at an equal estimate.
    transcribed, reference, conditioned = _case()
    conditioned.loc[1, "refit_iterations"] = 11
    conditioned.loc[1, "twin_iterations"] = 11
    rows = shared.validation_frame(
        rm19.compare_by_exit("A", "V2", transcribed, reference, conditioned)
    )
    assert rows.set_index("check").loc[REFIT_ROW, "result"] == shared.FAILS


def _many(draws: int, excused: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``draws`` exact draws, of which the first ``excused`` stop four rounds from R with a
    twin that moves too, 1e-5 away: each passes rule 5 and is excused."""
    transcribed, reference, conditioned = _case()
    one = [frame.loc[frame["replicate"] == 0] for frame in (transcribed, reference, conditioned)]
    frames = [
        pd.concat([frame.assign(replicate=k) for k in range(draws)], ignore_index=True)
        for frame in one
    ]
    transcribed, reference, conditioned = frames
    chosen = transcribed["replicate"] < excused
    transcribed.loc[chosen, "iterations"] = 14
    transcribed.loc[chosen, "ate"] += 1e-5
    marked = conditioned["replicate"] < excused
    conditioned.loc[marked, ["refit_iterations", "twin_iterations"]] = [14, 13]
    conditioned.loc[marked, "refit_ate"] += 1e-5
    conditioned.loc[marked, "twin_ate"] += 1e-5
    return transcribed, reference, conditioned


@pytest.mark.parametrize(
    ("draws", "excused", "held"),
    [(200, 2, True), (200, 3, False), (2_400, 24, True), (2_400, 25, False)],
)
def test_excused_draws_above_one_percent_stop_the_check(
    draws: int, excused: int, held: bool
) -> None:
    """X2 at its boundary: 2 of 200 and 24 of 2,400 hold; one more fails."""
    case = _many(draws, excused)
    assert rm19.excused(*case)[0] == excused
    rows = shared.validation_frame(rm19.compare_by_exit("V4", "V4 R", *case)).set_index("check")
    assert rows.loc[f"V4 R {rm19.DIFFERS}", "result"] == shared.HOLDS
    row = rows.loc[f"V4 R {rm19.EXCUSED}"]
    assert (row["result"] == shared.HOLDS) == held
    assert (int(row["compared"]), float(row["largest_difference"])) == (draws, excused / draws)


def test_the_max_iter_drift_is_the_signed_mean_over_max_iter_draws() -> None:
    """F8: reported beside the count and the largest difference, and read by no rule."""
    capped = _case(own_exit="cap", rounds=(100, 100), score_max=2e-6, gap=-3e-5)
    assert rm19.max_iter_drift(*capped) == (1, pytest.approx(-3e-5))
    count, drift = rm19.max_iter_drift(*_case(gap=1e-8))
    assert count == 0 and np.isnan(drift)
    reading = rm19.drift_reading("A", 1, -3e-5)
    assert (reading["value"], reading["result"]) == (-3e-5, rm19.SUPPLEMENTARY)
    reading = rm19.excused_reading("A", 2, 4e-5)
    assert (reading["scope"], reading["value"], reading["result"]) == (
        "2 excused draws",
        4e-5,
        rm19.SUPPLEMENTARY,
    )


def _v4_record(directory: Path, shift: float = 2e-6) -> None:
    """Part B rows and a V4 record on 4 draws; draw 1 reaches ``maxIter`` on both sides."""
    b_rows = _synthetic(replicates=4)
    b_rows = b_rows.assign(
        ey0=b_rows["ate"], ey1=b_rows["ate"], **{f"se_{name}": 0.02 for name in study.ESTIMANDS}
    )
    is_t = b_rows["arm"] == "T0000"
    b_rows.loc[is_t, "exit"] = ["tolIC", "cap", "tolIC", "tolIC"]
    b_rows.loc[is_t, "iterations"] = [10, 100, 10, 10]
    t0000 = b_rows.loc[is_t].set_index("replicate")
    r_rows = pd.DataFrame(
        [
            {
                "scenario": rm19.SCENARIO,
                "replicate": k,
                "estimand": name,
                "estimate": float(t0000.loc[k, "ate"])
                - (shift if (k == 1 and name == "ate") else 0.0),
                "std_error": 0.02,
                "score_max": 1e-5 if k == 1 else 1e-9,
                "rounds": int(t0000.loc[k, "iterations"]),
            }
            for k in range(4)
            for name in study.ESTIMANDS
        ]
    )
    _write_part(directory, "B", b_rows, [("x", True)])
    _write_part(directory, "V4", r_rows, [(f"V4 R {rm19.DIFFERS}", True)])
    refit = t0000.reset_index()
    twins = pd.concat([refit, refit.assign(arm=rm19.TWIN)], ignore_index=True)
    shared.write_table(twins, rm19.twin_path(directory, "V4"))


def test_the_v4_reading_reports_the_drift_of_its_t0000_rows(tmp_path: Path) -> None:
    """F8 and rule 6 on V4: they read the Part B ``T0000`` rows of the V4 draws."""
    _v4_record(tmp_path)
    table = rm19.table("V4", tmp_path, 1)
    drift = table.loc[table["statistic"] == "T0000 - R, signed mean ate difference"]
    assert list(drift["scope"]) == ["1 maxIter draws", "0 excused draws"]
    assert float(drift["value"].iloc[0]) == pytest.approx(2e-6)


def test_v4_runs_the_counting_step_and_the_rule_of_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V4 takes the path of step AV: ``r_phase``, then ``compare_by_exit`` with its conditions."""
    _v4_record(tmp_path)
    b_rows = shared.read_rows(shared.part_paths(tmp_path, "B")[0]).assign(n=3_000, seed=7)
    shared.write_table(b_rows, shared.part_paths(tmp_path, "B")[0])
    r_rows = shared.read_rows(shared.part_paths(tmp_path, "V4")[0])
    twins = shared.read_rows(rm19.twin_path(tmp_path, "V4"))
    calls: list[object] = []

    def phase(requested: list[tuple[object, ...]], *_: object) -> tuple[pd.DataFrame, ...]:
        calls.extend(requested)
        return r_rows, b_rows.loc[b_rows["arm"] == "C"], twins

    monkeypatch.setattr(rm19, "r_phase", phase)
    rm19.run_v4(tmp_path, 4, 1)
    assert calls == [("V4", rm19.SCENARIO, 3_000, k, 7) for k in range(4)]
    checks = shared.read_rows(shared.part_paths(tmp_path, "V4")[1])
    assert list(checks["check"]) == ["V4 payload C refit"] + [
        f"V4 R {kind}" for kind in (rm19.REFIT, *KINDS, rm19.EXCUSED)
    ]
    assert (checks["result"] == shared.HOLDS).all()
    # Step AV runs the same phase and the same rule.
    source = inspect.getsource(rm19.condition_part_a) + inspect.getsource(rm19.validate_committed)
    assert "r_phase(" in source and "compare_by_exit(" in source


def test_the_draw_step_fits_the_refit_and_the_twin_in_one_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """X1 and X4: ``condition_draw`` fits ``T(0, 0, 0, 0)`` and its twin together, and
    ``r_phase`` records the image ID for AV and V4 alike."""
    payload = pd.DataFrame({"Y": [1.0]})
    seen: list[str] = []

    def row(_: object, switches: Switches, *, solver: str = "gelsd") -> dict[str, object]:
        seen.append(solver)
        arm = switches.label if solver == "gelsd" else rm19.TWIN
        return {"arm": arm, **dict.fromkeys(rm19.VALUES, 0.1), "iterations": 10}

    monkeypatch.setattr(rm19, "draw_payload", lambda *_: (payload, 0.0, None))
    monkeypatch.setattr(rm19, "cleverly_row", lambda _: {"arm": "C"})
    monkeypatch.setattr(rm19, "_initial", lambda _: {})
    monkeypatch.setattr(rm19, "transcribed_row", row)
    *_, rows = rm19.condition_draw(("AV", rm19.SCENARIO, 3_000, 5, 11))
    assert seen == ["gelsd", "qr"]
    assert [entry["arm"] for entry in rows] == ["T0000", rm19.TWIN]
    assert all(entry["replicate"] == 5 and entry["part"] == "AV" for entry in rows)
    monkeypatch.setattr(rm19, "image_id", lambda: "sha256:test")
    monkeypatch.setattr(rm19.shared, "pool", lambda *_: [])
    monkeypatch.setattr(rm19, "count_rounds", lambda *_: pd.DataFrame())
    before = len(shared.notes())
    rm19.r_phase([("AV", rm19.SCENARIO, 3_000, 0, 1)], tmp_path, 1)
    assert list(shared.notes()[before:]) == [f"image {regenerate.REFERENCE.image}: sha256:test"]


class _Runner:
    """A stand-in for the counting wrapper: it writes the rows and the round counts it is given."""

    def __init__(self, rows: pd.DataFrame, rounds: pd.DataFrame) -> None:
        self.rows, self.rounds = rows, rounds
        self.cores: list[int] = []

    def run(self, _: Path, samples: Path, truths: Path, output: Path, *, cores: int) -> None:
        assert samples.exists() and truths.exists()
        self.cores.append(cores)
        shared.write_table(self.rows, output)
        shared.write_table(self.rounds, output.with_name(output.stem + "-rounds.csv"))


def test_count_rounds_merges_one_count_per_draw_and_stops_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = pd.DataFrame(
        {"scenario": rm19.SCENARIO, "replicate": [0, 0, 1, 1], "estimate": [0.1, 0.2, 0.3, 0.4]}
    )
    built: list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = [
        (pd.DataFrame({"Y": [1.0], "replicate": [k]}), {"replicate": k}, {}, []) for k in (0, 1)
    ]
    runner = _Runner(
        rows, pd.DataFrame({"scenario": rm19.SCENARIO, "replicate": [0, 1], "rounds": [7, 9]})
    )
    monkeypatch.setattr(rm19, "COUNTING", runner)
    counted = rm19.count_rounds(built, tmp_path)
    assert list(counted["rounds"]) == [7, 7, 9, 9]
    assert runner.cores == [rm19.R_WORKERS] and rm19.R_WORKERS <= 8
    runner.rounds = runner.rounds.iloc[:1]
    with pytest.raises(RuntimeError, match="different draws"):
        rm19.count_rounds(built, tmp_path)


def test_the_counting_wrapper_sources_the_unchanged_runner_in_the_pinned_image() -> None:
    """Rule 1 and W5: the same image and context, the canonical runner sourced, not copied."""
    counting, reference = rm19.COUNTING, regenerate.REFERENCE
    assert (counting.image, counting.build_context) == (reference.image, reference.build_context)
    root = counting.runner_root or ROOT
    wrapper = (root / counting.runner).read_text(encoding="utf-8")
    runner = reference.files(regenerate.HERE)[1]
    assert (root / "canonical" / "drtmle" / "run_drtmle.R") == runner
    assert 'source("/fixture/canonical/drtmle/run_drtmle.R")' in wrapper
    # run_drtmle.R stops when a worker returns no result; the wrapper when a count is missing.
    assert "returned no result" in runner.read_text(encoding="utf-8")
    assert "a draw has more than one round count" in wrapper


def test_the_twin_is_the_same_fit_with_another_rounding() -> None:
    """Rule 2: the QR solve agrees with ``gelsd`` to rounding, and the twin uses it."""
    rng = np.random.default_rng(3)
    design = np.column_stack([np.ones(50), rng.normal(size=50)])
    response = rng.normal(size=50)
    plain = rtrans.least_squares(design, response)
    with rtrans.solving_with("qr"):
        twin = rtrans.least_squares(design, response)
    assert np.max(np.abs(twin - plain)) <= 1e-13
    with pytest.raises(ValueError, match="solver"), rtrans.solving_with("svd"):  # type: ignore[arg-type]
        pass
    payload = _registered(1)[0]
    reference, qr = transcribe(payload), transcribe(payload, solver="qr")
    assert (qr.iterations, qr.exit) == (reference.iterations, reference.exit)
    shifts = [abs(qr.estimates[name] - reference.estimates[name]) for name in study.ESTIMANDS]
    assert 0.0 < max(shifts) <= rm19.SENSITIVE_SHIFT


#: R's round count on registered ``treatment_correct`` draw 1, counted by ``count_rounds.R`` in
#: the pinned image (the RM19 V2 diagnosis; step AV records it in ``av-rows.csv.gz``).
R_ROUNDS_DRAW_1 = 11


def test_a_one_step_defect_of_the_review_size_fails_rule_3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule 7: a looser ``glm`` tolerance on the ``gr1`` reduction fails V2 by rule 3.

    The harness review found a ``gr1`` refit that moved 4.2e-9 on equal inputs.  This defect
    moves the round-0 ``gr1`` refit of registered draw 1 by less than 1e-8.  The rows come from
    ``transcribed_row`` and the verdict from ``compare_by_exit``, as in step AV (X6).
    """
    payload = _registered(1)[0]
    reference = _committed(rm19.SCENARIO, 1, str(study.STUDY.reference)).reset_index()
    reference = reference.assign(scenario=rm19.SCENARIO, replicate=1)
    original = rtrans.glm_binomial

    def looser(*args: object, epsilon: float = rtrans.GLM_EPSILON, **kwargs: object) -> object:
        design = np.asarray(args[1])
        if design.ndim == 2 and design.shape[1] == 2:  # the gr1 fit: an intercept and a slope
            epsilon = 1e-6
        return original(*args, epsilon=epsilon, **kwargs)  # type: ignore[arg-type]

    def verdict() -> pd.DataFrame:
        draw = {"scenario": rm19.SCENARIO, "replicate": 1}
        refit = {**draw, **rm19.transcribed_row(payload, rm19.REFERENCE_ARM)}
        twin = {**draw, **rm19.transcribed_row(payload, rm19.REFERENCE_ARM, solver="qr")}
        scores = pd.DataFrame([{**draw, "score_max": 5e-9, "rounds": R_ROUNDS_DRAW_1}])
        conditioned = rm19.conditions(scores, scores, pd.DataFrame([refit, twin]))
        rows = rm19.compare_by_exit("A", "V2", pd.DataFrame([refit]), reference, conditioned)
        return shared.validation_frame(rows).set_index("check")

    clean = verdict()
    assert (clean["result"] == shared.HOLDS).all()
    assert int(clean.loc[TOL_IC_ROW, "compared"]) == 1
    split = rtrans.Split.of(payload["fold"].to_numpy())
    a = payload["A"].to_numpy(dtype=float)
    qn = [payload["qn0"].to_numpy(dtype=float), payload["qn1"].to_numpy(dtype=float)]
    upper = payload["gn1"].to_numpy(dtype=float)
    gn = [np.where(g < rtrans.TOLG, rtrans.TOLG, g) for g in (1.0 - upper, upper)]
    before = rtrans.estimate_grn(a, qn, gn, split)
    monkeypatch.setattr(rtrans, "glm_binomial", looser)
    after = rtrans.estimate_grn(a, qn, gn, split)
    step = max(float(np.max(np.abs(x[1] - y[1]))) for x, y in zip(before, after, strict=True))
    assert 1e-9 < step < 1e-8
    mutated = verdict()
    assert list(mutated.index[mutated["result"] == shared.FAILS]) == [TOL_IC_ROW]
    assert int(mutated.loc[TOL_IC_ROW, "compared"]) == 1
    assert float(mutated.loc[TOL_IC_ROW, "largest_difference"]) > rm19.SENSITIVE_TOLERANCE


def test_run_log_clears_the_notes_of_a_part(tmp_path: Path) -> None:
    """D6: a note made before a part's block opens does not leak into it."""
    shared.note("stale")
    with shared.run_log(tmp_path, "test"):
        assert shared.notes() == ()
        shared.note("fresh")
    assert shared.notes() == ()
    block = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "note: fresh" in block and "stale" not in block


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
    # A part that did not validate is still recorded: its reading then says so.
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
    if part in ("AV", "V4"):
        # The counting rerun of R and the twins: every draw of the part once, with its count.
        expected = (
            {(scenario, k) for scenario in study.SCENARIOS for k in range(study.PRIMARY_REPLICATES)}
            if part == "AV"
            else {(rm19.SCENARIO, k) for k in range(rm19.V4_DRAWS)}
        )
        assert set(zip(rows["scenario"], rows["replicate"], strict=True)) == expected
        assert rows.groupby(["scenario", "replicate"]).size().eq(len(study.ESTIMANDS)).all()
        assert rows["rounds"].between(1, rtrans.MAX_ITER).all()
        twins = shared.read_rows(rm19.twin_path(rm19.HERE, part))
        assert set(zip(twins["scenario"], twins["replicate"], strict=True)) == expected
        assert set(twins["arm"]) == {"T0000", rm19.TWIN} and len(twins) == 2 * len(expected)
        return
    declared = {"A": study.PRIMARY_REPLICATES}.get(part, rm19.REPLICATES)
    counts = rows.groupby(["scenario", "n", "arm"]).size()
    assert counts.eq(declared).all(), counts.to_string()
    declared_draws = {(scenario, n, k, seed) for scenario, n, k, seed in rm19.draws(part)}
    observed = set(zip(rows["scenario"], rows["n"], rows["replicate"], rows["seed"], strict=True))
    assert observed == declared_draws


def test_the_part_a_validation_is_computed_from_its_committed_inputs() -> None:
    """W3: ``a-validation.csv`` is V1 to V5 over the committed Part A rows and the step AV files."""
    if "AV" not in RECORDED:
        assert not rm19.twin_path(rm19.HERE, "AV").exists()
        return
    rows = shared.read_rows(shared.part_paths(rm19.HERE, "A")[0])
    r_rows = shared.read_rows(shared.part_paths(rm19.HERE, "AV")[0])
    twins = shared.read_rows(rm19.twin_path(rm19.HERE, "AV"))
    rebuilt = shared.as_committed(rm19.validate_committed(rows, r_rows, twins))
    committed = shared.read_rows(shared.part_paths(rm19.HERE, "A")[1])
    pd.testing.assert_frame_equal(rebuilt, committed, check_dtype=False, rtol=1e-12)
    counted = r_rows.drop_duplicates(["scenario", "replicate"]).set_index(["scenario", "replicate"])
    assert counted.loc[(rm19.SCENARIO, 1), "rounds"] == R_ROUNDS_DRAW_1
