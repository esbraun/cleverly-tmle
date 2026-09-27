"""The RM19 localization design: its transcription, its rules, its seeds and its record."""

from __future__ import annotations

import functools
import inspect
import itertools
from pathlib import Path

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
RECORDED: tuple[str, ...] = ()

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
    r_rows = _committed(rm19.SCENARIO, replicate, study.STUDY.reference)
    c_rows = _committed(rm19.SCENARIO, replicate, study.STUDY.implementation)
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


@pytest.mark.parametrize("matched", [True, False])
def test_a_v4_exit_mismatch_stops_the_part_a_reading(
    tmp_path: Path, matched: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3 and F6: A folds in the V4 exit status, and G's family reads every Part A row."""
    rows = _synthetic()
    control = rows.loc[rows["arm"] == "T0000"].assign(scenario="outcome_correct", guard_events=3)
    _write_part(tmp_path, "A", pd.concat([rows, control]), [("V1 cleverly rows", True)])
    _write_part(tmp_path, "V4", rows.iloc[:0], [("V4 R exit status", matched)])
    seen: dict[str, bool] = {}

    def factorial(*_: object, guard: bool, **__: object) -> list[dict[str, object]]:
        seen["guard"] = guard
        return []

    monkeypatch.setattr(rm19, "_committed_reference", lambda _: (None, None, None))
    monkeypatch.setattr(rm19, "max_iter_drift", lambda *_: (0, float("nan")))
    monkeypatch.setattr(rm19, "factorial_table", factorial)
    for name in ("decomposition_table", "control_table", "arm_summaries"):
        monkeypatch.setattr(rm19, name, lambda *_, **__: [])
    table = rm19.table("A", tmp_path, 1)
    assert "V4 R exit status" in set(table["scope"])
    assert (list(table["result"])[-1] == shared.NOT_VALIDATED) == (not matched)
    assert seen == ({"guard": True} if matched else {})


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


def test_part_c_reads_the_part_b_rows_of_its_own_record(tmp_path: Path) -> None:
    """F5: a smoke C reads B from its own directory, a declared C from the pushed record."""
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
    # A declared C (2,000 rows per size) reads the committed record, which holds no B rows yet.
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


# ------------------------------------------------------------- V2 and V4 by exit status


def _exit_case(
    own_exit: str, score_max: float, shift: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One exact ``tolIC`` draw, and one draw whose T0000 ``ate`` sits ``shift`` from R's."""
    transcribed = pd.DataFrame(
        {
            "scenario": rm19.SCENARIO,
            "replicate": [0, 1],
            "arm": "T0000",
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
    before = len(shared.notes())
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
    maxiter = int(matched and own_exit == "cap")
    drift = f"{shift:.3g} over 1" if maxiter else "nan over 0"
    assert list(shared.notes()[before:]) == [
        f"V2: {counts['V2 maxIter draws']} maxIter draws, largest scaled difference "
        f"{rows.set_index('check').loc['V2 maxIter draws', 'largest_difference']:.3g}, "
        f"signed mean ate difference {drift}"
    ]


def test_the_max_iter_drift_is_the_signed_mean_over_max_iter_draws() -> None:
    """F8: reported beside the count and the largest difference, and read by no rule."""
    assert rm19.max_iter_drift(*_exit_case("cap", 2e-6, -3e-5)) == (1, pytest.approx(-3e-5))
    count, drift = rm19.max_iter_drift(*_exit_case("tolIC", 5e-9, 1e-8))
    assert count == 0 and np.isnan(drift)
    reading = rm19.drift_reading("A", 1, -3e-5)
    assert (reading["value"], reading["result"]) == (-3e-5, rm19.SUPPLEMENTARY)


def test_the_v4_reading_reports_the_drift_of_its_t0000_rows(tmp_path: Path) -> None:
    """F8 on V4: the drift reads the Part B ``T0000`` rows of the V4 draws, one row per draw."""
    b_rows = _synthetic(replicates=4)
    b_rows = b_rows.assign(
        ey0=b_rows["ate"], ey1=b_rows["ate"], **{f"se_{name}": 0.02 for name in study.ESTIMANDS}
    )
    t0000 = b_rows.loc[b_rows["arm"] == "T0000"].set_index("replicate")["ate"]
    b_rows.loc[b_rows["arm"] == "T0000", "exit"] = ["tolIC", "cap", "tolIC", "tolIC"]
    r_rows = pd.DataFrame(
        [
            {
                "scenario": rm19.SCENARIO,
                "replicate": k,
                "estimand": name,
                "estimate": float(t0000[k]) - (2e-6 if (k == 1 and name == "ate") else 0.0),
                "std_error": 0.02,
                "score_max": 1e-5 if k == 1 else 1e-9,
            }
            for k in range(4)
            for name in study.ESTIMANDS
        ]
    )
    _write_part(tmp_path, "B", b_rows, [("x", True)])
    _write_part(tmp_path, "V4", r_rows, [("V4 R exit status", True)])
    table = rm19.table("V4", tmp_path, 1)
    drift = table.loc[table["statistic"] == "T0000 - R, signed mean ate difference"]
    assert list(drift["scope"]) == ["1 maxIter draws"]
    assert float(drift["value"].iloc[0]) == pytest.approx(2e-6)


def test_a_draw_on_one_side_only_stops_the_check() -> None:
    transcribed, reference, scores = _exit_case("tolIC", 5e-9, 0.0)
    with pytest.raises(RuntimeError, match="one side only"):
        rm19.compare_by_exit("A", "V2", transcribed.iloc[:1], reference, scores)


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
    if part == "V4":
        assert sorted(rows["replicate"].unique()) == list(range(rm19.V4_DRAWS))
        assert rows.groupby("replicate").size().eq(len(study.ESTIMANDS)).all()
        return
    declared = {"A": study.PRIMARY_REPLICATES}.get(part, rm19.REPLICATES)
    counts = rows.groupby(["scenario", "n", "arm"]).size()
    assert counts.eq(declared).all(), counts.to_string()
    expected = {(scenario, n, k, seed) for scenario, n, k, seed in rm19.draws(part)}
    observed = set(zip(rows["scenario"], rows["n"], rows["replicate"], rows["seed"], strict=True))
    assert observed == expected
