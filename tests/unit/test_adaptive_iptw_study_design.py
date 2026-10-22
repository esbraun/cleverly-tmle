"""The declared design of ``adaptive-iptw`` (X21, ``X21-plan.md`` 3.5, design test items 1-8).

The test fits no estimator.  Each item reads a law as a table of cells with exact
probabilities (:mod:`tests.unit._adaptive_iptw_study_support`) and computes what the plan asks
for.  The study modules carry the results as constants, and this test fails when a constant
differs from the computation, so the record cannot drift from the design.

======  ==================================================================================
item    what the test computes
======  ==================================================================================
1       the untargeted displacement of each necessity row, and the chance a run of the
        scene's size reaches the declared threshold of 1.0
2       the predicted bias ``b`` of every row that reads ``bias_equivalent``, and the role
        it implies
3       the range of each term's mechanism, against ``g_bounds``
4       the pass probability of every verdict row, and the replication count that gives
        each cell at least 0.90 (0.95 for the spread cell and the necessity rows)
5       the distinct values of each saturated basis, and the fold-to-fold identifiability
        of the cross-fitted finite laws
6       the separation of the regime truths in the unit of the estimator's SD
7       ``L2(Q_0 - Qr_0)`` and the declared-basis L2 error per law and key
8       the chance a binary arm mean leaves ``(0, 1)`` at the declared size
======  ==================================================================================

**The role rule** (item 2, amended in rev 3.9).  ``b`` is the population fixed point of the
targeting loop at the learner's limit minus the truth, over the SD of the estimator at that
limit, times ``sqrt(n)``.  ``|b| <= 0.05`` is positive and ``|b| >= 0.5`` is a control.  A row
between the two stays positive only when two things hold.  Some replication count of at most
:data:`R_CAP` passes it with probability 0.90, and that count becomes the scene's.  The bias
falls with the size, ``|b(4n)| < |b(n)|``, which is the sign of a second-order remainder.  A
row that fails either condition is routed to reporting before the run
(:data:`~tests.studies.canonical_adaptive_iptw.DESIGN_ROUTED`).  A bias that grows with ``n``
is the first-order validity term, and no replication count gives it a valid positive.

**The SD** is the SD of the Theorem 1 curve with the declared-basis ``Qr``, less the projection
onto the score of the mechanism model when the mechanism is a fitted logistic regression (laws
P, PD, VL).  Law M uses the Theorem 1 SD, because its multinomial mechanism has no score here.

**Records, not checks.**  :data:`PLAN_SIZES` holds the sizes the plan declares.  The test
verifies each against the items and fails when a size fails its item.  The sizes of laws C, WT
and ST follow :func:`unit_rule`.  The fold-gap simulation uses 1,000 draws, where the plan's
probe used 200, because the 200-draw ratio at one size moves by 0.03 between seeds.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators import AdaptiveIPTW
from tests import discrete_law_composite as composite_law
from tests.studies import adaptive_iptw_laws as laws
from tests.studies import adaptive_iptw_properties as properties
from tests.studies import canonical_adaptive_iptw as study
from tests.studies import clustered_unequal_laws as clustered
from tests.studies import mar_arm_indexed_laws as mar
from tests.studies import multi_arm_common as multi
from tests.studies import stratified_law as stratified
from tests.studies import weighted_point_common as weighted
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.unit import _adaptive_iptw_study_support as support
from tests.unit.test_study_provenance import _required_from

pytestmark = pytest.mark.xdist_group("adaptive_iptw_study_design")

#: The pass probability every verdict row must reach, and the one the spread cell and the
#: necessity rows must reach (``X21-plan.md`` item 4 and item 1).
ROW_TARGET = 0.90
STRICT_TARGET = 0.95
#: The role thresholds of item 2.
POSITIVE_LIMIT = 0.05
CONTROL_LIMIT = 0.5
#: The largest replication count a bias scene may need before its row is routed to reporting.
R_CAP = 1_600
R_STEP = 200
#: The relative spread of one reported SE in the calibration rule (the planning probe's value).
SE_NOISE = 0.03
#: The displacement floor of a necessity row (``NECESSITY_THRESHOLD``).
DISPLACEMENT_FLOOR = 1.0
#: The ratio of the plug-in SE to the true SD that the verifier simulated for law VL1 under the
#: correct logistic mechanism, by size (``_x21_validity_sim.log``): 1.084 at n = 1,000 and 1.093
#: at n = 4,000.  The asymptotic ratio is computed here.
VERIFIER_RATIOS = {1_000: 1.084, 4_000: 1.093}
#: The sizes the plan declares, checked by the items.
PLAN_SIZES = {
    "N_PD": 2_000,
    "N_P": 3_000,
    "LADDER_SIZES": (500, 1_500, 4_500),
    "N_M": 2_000,
    "N_CN": 2_000,
    "N_RM": 3_000,
    "N_MA": 2_000,
    "N_MA_CF": 7_000,
    "N_VL": 2_000,
    "N_CL": 200,
}
#: The ladder of candidate sizes for laws the plan does not size, and the rows a saturated
#: fit needs in its smallest covariate cell.
SIZE_LADDER = (2_000, 3_000, 4_000, 6_000, 8_000)
UNIT_ROWS = 100
#: The bias scenes that also host a cell with a larger count: the scene runs that count and
#: reads its bias rows from all of its draws (the spread cell sets VL1, the calibration cells
#: set the known-mechanism scenes).
HOSTS_LARGER = {
    "vl1": study.R_VL,
    "wt_known": study.R_CAL,
    "cl_known": study.R_CAL,
    "st_known": study.R_CAL,
}
#: The expected number of rows in a whole run that may sit outside ``g_bounds`` (item 3).
CLIP_BUDGET = 0.01
#: The regime-separation floors of item 6.
CALIBRATION_SEPARATION = 0.6
FOLD_DRAWS = 1_000
FOLD_THRESHOLD = 1.05
FOLD_MARGIN = 0.03


# --------------------------------------------------------------------------------- populations


@dataclass(frozen=True)
class Spec:
    """A population, the size it is read at, and whether its SD is the projected one."""

    population: support.Population
    n: int
    projected: bool


@functools.cache
def specs() -> dict[str, Spec]:
    """Every population the items read, by name."""
    out: dict[str, Spec] = {}

    def add(name: str, population: support.Population, n: int, projected: bool) -> None:
        out[name] = Spec(population, n, projected)

    degree = study.DEGREE
    add("p_correct", support.build_p("p", "correct", study.N_P, laws, degree["p"]), study.N_P, True)
    add("p_rate", support.build_p("p", "rate", study.N_P, laws, degree["p"]), study.N_P, True)
    add("p_wrong", support.build_p("p", "wrong", study.N_P, laws, degree["p"]), study.N_P, True)
    for n in study.LADDER_SIZES:
        add(f"p_ladder_{n}", support.build_p("p", "rate", n, laws, degree["p"]), n, True)
    add(
        "pd_known",
        support.build_p("pd", "correct", study.N_PD, laws, degree["pd"], regimes=True),
        study.N_PD,
        False,
    )
    add(
        "pd_rate",
        support.build_p("pd", "rate_scaled", study.N_PD, laws, degree["pd"], regimes=True),
        study.N_PD,
        True,
    )
    add("m_correct", support.build_m("correct", study.N_M, multi, degree["m"]), study.N_M, False)
    add("m_rate", support.build_m("rate", study.N_M, multi, degree["m"]), study.N_M, False)
    add("cn", support.build_cn(study.N_CN, mar, (degree["cn"], 2)), study.N_CN, False)
    add(
        "rm",
        support.build_rm(study.N_RM, mar, degree["rm"], study.RM_RULE, study.RM_TILT),
        study.N_RM,
        False,
    )
    add(
        "ma",
        support.build_ma(composite_law.TWO, composite_law.P_W, degree["ma"], study.RM_RULE),
        study.N_MA,
        False,
    )
    for law, kind in (("vl1", "glm"), ("vl1", "lsb"), ("vl0", "lsb")):
        add(
            f"{law}_{kind}",
            support.build_vl(law, kind, study.N_VL, laws, degree["vl"]),
            study.N_VL,
            True,
        )
    return out


#: Where each ``(scene, fit)`` that reads a bias gets its ``b``: a population, ``None`` for a
#: learner whose limit is the true mechanism (the loop starts at its fixed point, so ``b = 0``
#: exactly), or ``"flexible"`` for the boosted learner, which has no closed limit.  ``b`` is
#: the population limit at the declared size.  It does not model the generated-design term of a
#: cross-fitted reduced regression with an estimated mechanism.  That term is why a
#: cross-fitted fit trains fold ``k``'s regression on leave-two-folds-out designs (Rev4.0).
#: Under the pooled designs of run 1 the term was about 0.4 SD at law WT
#: (``X21-red-cell-diagnosis.md``, plans repository).
SOURCES: dict[tuple[str, str], str | None] = {
    ("p_correct", "p_correct"): "p_correct",
    ("p_rate", "p_rate"): "p_rate",
    ("p_insample", "p_insample"): None,
    ("p_wrong", "p_wrong"): "p_wrong",
    ("p_flexible", "p_flexible"): "flexible",
    **{(f"p_ladder_{n}", "p_rate"): f"p_ladder_{n}" for n in study.LADDER_SIZES},
    ("m_correct", "m_correct"): "m_correct",
    ("m_rate", "m_rate"): "m_rate",
    ("c_l1", "c_l1"): None,
    ("c_l3", "c_l3"): None,
    ("c_l1_insample", "c_l1_insample"): None,
    ("c_l3_insample", "c_l3_insample"): None,
    ("wt_known", "wt_estimated"): None,
    ("cl_known", "cl_estimated"): None,
    ("st_known", "st_estimated"): None,
    ("pd_rate", "pd_regimes_estimated"): "pd_rate",
    ("cn", "cn"): "cn",
    ("rm", "rm"): "rm",
    ("ma_insample", "ma_arms"): None,
    ("ma_insample", "ma_regime"): None,
    ("ma_crossfit", "ma_crossfit"): None,
    ("vl1", "vl1_lsb"): "vl1_lsb",
    ("vl1", "vl1_glm"): "vl1_glm",
    ("vl0", "vl0_lsb"): "vl0_lsb",
}


def bias_of(reader: study.Reader) -> dict[str, float] | str:
    """``b`` of a bias-reading row, ``"flexible"`` for the boosted learner."""
    source = SOURCES[(reader.scene, reader.fit)]
    if source is None:
        return {"b": 0.0, "untargeted": 0.0, "sd": math.nan, "source": "oracle"}
    if source == "flexible":
        return "flexible"
    spec = specs()[source]
    result = spec.population.key_bias(reader.key, spec.n, projected=spec.projected)
    return {**result, "source": source}


def untargeted_ratio(spec: Spec, key: str) -> float:
    """The SD of the untargeted curve over the SD of the targeted one."""
    population = spec.population
    return population.untargeted_sd(key) / population.key_sd(key, projected=spec.projected)


# -------------------------------------------------------------------------------- the readers


def bias_readers() -> list[study.Reader]:
    """Every row that reads a standardized bias, the necessity targeted arms included."""
    return [
        r
        for r in study.READERS
        if r.family in ("bias", "root_n_points")
        or (r.family == "targeting_necessity" and r.cell.endswith("__targeted"))
    ]


def _passes(reader: study.Reader, b: float, replicates: int) -> float:
    if reader.role == "control":
        return support.control_fail_probability(abs(b), replicates)
    return support.bias_pass_probability(b, replicates)


def necessity_table() -> dict[str, dict[str, float]]:
    """Item 1: the displacement of each candidate necessity row and its pass probability."""
    out = {}
    for label, (scene, fit, key, _) in study.NECESSITY_LABELS.items():
        source = SOURCES[(scene, fit)]
        assert source is not None
        spec = specs()[source]
        result = spec.population.key_bias(key, spec.n, projected=spec.projected)
        ratio = untargeted_ratio(spec, key)
        out[label] = {
            "displacement": abs(float(result["untargeted"])),
            "targeted_b": float(result["b"]),
            "ratio": ratio,
            "exact_variance": float(spec.population.paired_variance(key)),
            "n": spec.n,
        }
    return out


def displacement_probability(row: dict[str, float], replicates: int) -> float:
    """The chance the paired displacement reaches the floor, under independent arms.

    The variance of the per-replicate difference is the sum of the two arms' variances, in the
    targeted arm's units.  The independent sum is the largest the variance can be with
    non-negative correlation, so the probability is a floor.
    """
    return support.displacement_pass_probability(
        row["displacement"], 1.0 + row["ratio"] ** 2, replicates, floor=DISPLACEMENT_FLOOR
    )


def untargeted_b(row: dict[str, float]) -> float:
    """The untargeted arm's bias in its own SD (the control's standardized bias)."""
    return row["displacement"] / row["ratio"]


def scene_replicates(scene: str) -> int:
    """Item 4: the replication count the scene's rows need.

    The smallest multiple of :data:`R_STEP`, at least ``R_BIAS``, that gives every positive
    row not routed to reporting a pass probability of 0.90, every control a failure probability
    of 0.90, and every necessity row on the scene a displacement probability of 0.95.
    """
    rows = [r for r in bias_readers() if r.scene == scene]
    routed = routed_cells()
    labels = [label for label, (s, _, _, _) in study.NECESSITY_LABELS.items() if s == scene]
    table = necessity_table()
    for replicates in range(study.R_BIAS, R_CAP + 1, R_STEP):
        ok = True
        for r in rows:
            name = f"{r.family}/{r.cell}"
            if name in routed:
                continue
            result = bias_of(r)
            if isinstance(result, str):
                continue
            ok &= _passes(r, result["b"], replicates) >= ROW_TARGET
        for label in labels:
            ok &= displacement_probability(table[label], replicates) >= STRICT_TARGET
            ok &= (
                support.control_fail_probability(untargeted_b(table[label]), replicates)
                >= ROW_TARGET
            )
        if ok:
            return replicates
    raise AssertionError(f"{scene}: no replication count up to {R_CAP} passes its rows")


def population_at(source: str, n: int) -> Spec:
    """The population of a middle-band source at size ``n``, for the trend condition."""
    degree = study.DEGREE
    if source == "p_rate" or source.startswith("p_ladder_"):
        return Spec(support.build_p("p", "rate", n, laws, degree["p"]), n, True)
    if source == "m_rate":
        return Spec(support.build_m("rate", n, multi, degree["m"]), n, False)
    raise KeyError(f"no trend is defined for the source {source}")


def trend(reader: study.Reader) -> tuple[float, float]:
    """``(b(n), b(4n))`` of a bias-reading row, from one population at two sizes."""
    source = SOURCES[(reader.scene, reader.fit)]
    assert source is not None and source != "flexible"
    spec = specs()[source]
    larger = population_at(source, 4 * spec.n)
    at = spec.population.key_bias(reader.key, spec.n, projected=spec.projected)["b"]
    at_four = larger.population.key_bias(reader.key, larger.n, projected=larger.projected)["b"]
    return float(at), float(at_four)


@functools.cache
def routed_cells() -> dict[str, float]:
    """Item 2: the middle-band positive rows routed to reporting, with their ``b``.

    A row is routed when no count up to :data:`R_CAP` passes it with probability 0.90 or when
    its bias does not fall from ``n`` to ``4 n``.  A row with ``|b|`` at most
    :data:`POSITIVE_LIMIT` is never routed.
    """
    out = {}
    for r in bias_readers():
        result = bias_of(r)
        if isinstance(result, str) or r.role != "positive":
            continue
        if abs(result["b"]) <= POSITIVE_LIMIT:
            continue
        at, at_four = trend(r)
        unreachable = support.bias_pass_probability(result["b"], R_CAP) < ROW_TARGET
        if unreachable or abs(at_four) >= abs(at):
            out[f"{r.family}/{r.cell}"] = float(result["b"])
    return out


# ------------------------------------------------------------------------------------ item 1


class TestNecessityPower:
    """Item 1: each necessity row's untargeted displacement and its power."""

    def test_the_exact_displacements_reproduce_the_plan_probes(self) -> None:
        """The plan quotes PD ``d`` 2.08, ``s`` 1.98, CN ``ey_obs`` 3.31 and RM ``d`` 1.11."""
        spec = specs()["pd_rate"]
        # The probe divides by the Theorem 1 SD; the test divides by the projected SD below.
        theorem1 = {
            key: abs(spec.population.key_bias(key, spec.n)["untargeted"])
            for key in ("ey_regime[d]", "ey_regime[s]")
        }
        assert theorem1["ey_regime[d]"] == pytest.approx(2.08, abs=0.02)
        assert theorem1["ey_regime[s]"] == pytest.approx(1.98, abs=0.02)
        table = necessity_table()
        assert table["cn_ey_obs"]["displacement"] == pytest.approx(3.31, abs=0.01)
        assert table["rm_d"]["displacement"] == pytest.approx(1.11, abs=0.01)

    def test_every_declared_row_reaches_the_floor_with_probability_095(self) -> None:
        table = necessity_table()
        for label, row in table.items():
            scene = study.NECESSITY_LABELS[label][0]
            replicates = study.SCENES[scene].replicates
            assert row["displacement"] >= DISPLACEMENT_FLOOR, label
            assert displacement_probability(row, replicates) >= STRICT_TARGET, (label, replicates)

    def test_the_exact_paired_variance_gives_the_same_verdict_as_the_bound(self) -> None:
        """Report the exact paired variance beside the independent-arms bound (m7)."""
        for label, row in necessity_table().items():
            replicates = study.SCENES[study.NECESSITY_LABELS[label][0]].replicates
            exact = support.displacement_pass_probability(
                row["displacement"], row["exact_variance"], replicates, floor=DISPLACEMENT_FLOOR
            )
            assert exact >= STRICT_TARGET, (label, exact)
            assert exact >= displacement_probability(row, replicates) - 1e-9, label

    def test_the_rows_are_the_new_key_kinds_with_a_passing_item_and_the_p_arms(self) -> None:
        """RM's S11 key fails the item, so it has no row; the other new kinds pass."""
        spec = specs()["rm"]
        s_key = spec.population.key_bias("ey_regime[s]", spec.n)
        assert abs(float(s_key["untargeted"])) < DISPLACEMENT_FLOOR
        assert set(study.NECESSITY_LABELS) == {
            "p_ey0",
            "p_ey1",
            "p_ate",
            "pd_d",
            "pd_s",
            "cn_ey_obs",
            "rm_d",
        }

    def test_a_targeting_step_that_moved_nothing_would_be_found(self) -> None:
        """Nonzero witness: the loop moves the estimate by a margin on every row.

        A row whose targeted and untargeted values agree would pass the displacement
        trivially if the loop did nothing.  The untargeted value differs from the truth by
        the displacement, and the targeted value by far less.
        """
        for label, row in necessity_table().items():
            assert abs(row["targeted_b"]) < 0.25 * row["displacement"], label


# ------------------------------------------------------------------------------------ item 2


class TestPredictedBiasAndRole:
    """Item 2: ``b`` of every row that reads ``bias_equivalent``, and its role."""

    def test_the_loop_reproduces_the_plan_probes(self) -> None:
        lsb = specs()["vl1_lsb"]
        assert lsb.population.key_bias("ey1", lsb.n, projected=True)["b"] == pytest.approx(
            -0.697, abs=0.003
        )
        h1 = specs()["vl0_lsb"]
        assert h1.population.key_bias("ey1", h1.n, projected=True)["b"] == pytest.approx(
            0.045, abs=0.002
        )

    def test_a_learner_at_the_true_mechanism_has_zero_bias_and_a_perturbed_one_does_not(
        self,
    ) -> None:
        """Nonzero witness and mutation control for the zero-bias rows."""
        for name in ("p_correct", "m_correct", "vl1_glm"):
            spec = specs()[name]
            for key in spec.population.keys:
                assert spec.population.key_bias(key, spec.n)["b"] == pytest.approx(0.0, abs=1e-9)
            assert all(s["rounds"] == 0 for s in spec.population.solved().values())
        rate = specs()["p_rate"]
        assert abs(rate.population.key_bias("ate", rate.n)["b"]) > 0.05
        wrong = specs()["p_wrong"]
        assert abs(wrong.population.key_bias("ate", wrong.n)["b"]) > 5.0

    def test_every_bias_row_has_a_source(self) -> None:
        for reader in bias_readers():
            assert (reader.scene, reader.fit) in SOURCES, reader

    def test_the_controls_are_beyond_the_control_limit_and_the_positives_are_not(self) -> None:
        for reader in bias_readers():
            result = bias_of(reader)
            if isinstance(result, str):
                continue
            if reader.role == "control":
                assert abs(result["b"]) >= CONTROL_LIMIT, reader.cell
            else:
                assert abs(result["b"]) < CONTROL_LIMIT, reader.cell

    def test_a_row_between_the_limits_is_positive_only_with_a_count_and_a_falling_bias(
        self,
    ) -> None:
        """The middle band: positive if a count passes it and its bias falls, else routed."""
        middle = {}
        for reader in bias_readers():
            result = bias_of(reader)
            if isinstance(result, str) or reader.role == "control":
                continue
            if abs(result["b"]) > POSITIVE_LIMIT:
                middle[f"{reader.family}/{reader.cell}"] = (reader, float(result["b"]))
        assert middle, "the middle band is empty, so the trend condition reads nothing"
        for name, (reader, b) in middle.items():
            reachable = support.bias_pass_probability(b, R_CAP) >= ROW_TARGET
            at, at_four = trend(reader)
            falling = abs(at_four) < abs(at)
            assert (name in routed_cells()) is (not (reachable and falling)), name
        design_routed = set(routed_cells())
        assert design_routed == set(study.DESIGN_ROUTED)
        for name, reading in study.DESIGN_ROUTED.items():
            assert f"b = {routed_cells()[name]:+.3f}" in reading, name

    def test_every_middle_band_row_of_the_study_has_a_falling_bias(self) -> None:
        """The rows the study keeps positive on the middle band are second order.

        Law P ``rate`` is H1, so its bias is the second-order remainder of the learner and it
        falls with ``n``.  The table is the evidence page's.
        """
        rows = {}
        for reader in bias_readers():
            result = bias_of(reader)
            if isinstance(result, str) or reader.role == "control":
                continue
            if abs(result["b"]) > POSITIVE_LIMIT and reader.family == "bias":
                rows[reader.cell] = trend(reader)
        for cell, (at, at_four) in rows.items():
            assert abs(at_four) < abs(at), (cell, at, at_four)

    def test_the_trend_condition_catches_a_level_set_breaking_error(self) -> None:
        """Deliberate-mutation control: a bias that grows with ``n`` fails the condition.

        Law VL1 with the ``lsb`` learner shifts the logit along a covariate, which breaks the
        level sets of ``g_0``.  Its ``b`` grows with ``n``, as the old law M rate error did
        (``b`` of ``ey[low]``: -0.198 at 2,000, -0.375 at 32,000).
        """
        at = specs()["vl1_lsb"]
        larger = support.build_vl("vl1", "lsb", 4 * at.n, laws, study.DEGREE["vl"])
        small = at.population.key_bias("ey1", at.n, projected=True)["b"]
        large = larger.key_bias("ey1", 4 * at.n, projected=True)["b"]
        assert abs(large) > abs(small)
        # The shrunk multinomial of the study keeps every level set: b is below 1e-3 at both sizes (the cubic basis is not exact).
        for n in (study.N_M, 4 * study.N_M):
            population = support.build_m("rate", n, multi, study.DEGREE["m"])
            assert all(abs(population.key_bias(k, n)["b"]) < 1e-3 for k in population.keys)
            # Nonzero witness: the untargeted mean is displaced, so the zero is the targeting.
            assert abs(population.key_bias("ey[low]", n)["untargeted"]) > 0.05

    def test_the_declared_roles_are_the_roles_the_rule_gives(self) -> None:
        """No role override: the readers declare the rule's roles, and the records agree."""
        assert study.ROLE_OVERRIDES == {}
        for reader in bias_readers():
            result = bias_of(reader)
            if isinstance(result, str):
                assert reader.cell.endswith("__flexible")
                assert reader.role == "positive"
                continue
            expected = "control" if abs(result["b"]) >= CONTROL_LIMIT else "positive"
            assert reader.role == expected, (reader.cell, result["b"])

    def test_the_predicted_bias_table(self) -> None:
        """Pin the table: every row's ``b`` to two decimals, the numbers the evidence page cites."""
        got = {}
        for reader in bias_readers():
            result = bias_of(reader)
            if not isinstance(result, str):
                got[f"{reader.family}/{reader.cell}"] = round(float(result["b"]), 2)
        assert got["bias/ey1__level_set_breaking"] == pytest.approx(-0.70, abs=0.01)
        assert got["bias/ey1__level_set_breaking_h1"] == pytest.approx(0.04, abs=0.01)
        assert got["bias/ey1__parametric_q_ne_qr"] == 0.0
        assert got["bias/ate__wrong_mechanism"] > 10.0
        assert got["bias/ate__rate"] == pytest.approx(0.09, abs=0.01)
        assert got["bias/ey_d__regime"] == pytest.approx(0.01, abs=0.01)
        assert all(
            abs(got[f"bias/{label}__regime"]) <= POSITIVE_LIMIT
            for label in ("ey_never", "ey_d", "ey_s", "ate_d", "ate_s")
        )
        for cell in ("natural_course", "insample_regime_missing"):
            assert all(abs(v) < 1e-9 for k, v in got.items() if k.endswith("__" + cell))


# ------------------------------------------------------------------------------------ item 3


class TestBounds:
    """Item 3: no mechanism meets ``g_bounds`` under its learner."""

    def test_no_term_limit_or_fixed_point_meets_the_bounds(self) -> None:
        """Expected clipped rows over the longest run are below :data:`CLIP_BUDGET`.

        The quadrature grids of laws P, PD, M and VL reach into the Gaussian tail.  A tail
        point can sit outside ``g_bounds`` for the true mechanism, the learner's limit or the
        fixed point.  A draw clips only on such a point, so the mass of those points, times the
        rows of a run, is the expected number of clipped rows.
        """
        low, high = study.G_BOUNDS
        for name, spec in specs().items():
            for term in spec.population.terms:
                mass = spec.population.outside_mass(term, low, high)
                expected = mass * spec.n * study.R_CAL
                assert expected < CLIP_BUDGET, (name, term, mass, expected)

    def test_the_laws_without_a_tail_stay_strictly_inside_the_bounds(self) -> None:
        low, high = study.G_BOUNDS
        for name in ("p_correct", "p_rate", "pd_known", "pd_rate", "cn", "rm", "ma"):
            for term in specs()[name].population.terms:
                smallest, largest = specs()[name].population.mechanism_range(term)
                assert low < smallest and largest < high, (name, term, smallest, largest)

    def test_the_finite_composites_are_inside_the_bounds(self) -> None:
        low, high = study.G_BOUNDS
        for law in (mar.LAWS["l1"], mar.LAWS["l3"], composite_law.TWO):
            composite = law.composite
            assert low < composite.min() and composite.max() < high

    def test_the_sampled_laws_keep_their_true_mechanism_inside_the_bounds(self) -> None:
        """Laws WT, CL and ST: the true mechanism over a large draw."""
        low, high = study.G_BOUNDS
        assert low < weighted.G.min() and weighted.G.max() < high
        frame = clustered.draw_informative(20_000, np.random.default_rng(3))
        g = np.asarray(clustered.informative_propensity(frame["W1"], frame["W2"]))
        assert low < g.min() and g.max() < high
        grid = [(w, v) for w in stratified.LEVELS_W for v in stratified.STRATA]
        values = [float(stratified.propensity(w, v)) for w, v in grid]
        assert low < min(values) and max(values) < high


# ------------------------------------------------------------------------------------ item 4


class TestPassProbability:
    """Item 4: every verdict row passes with probability 0.90, and R is the smallest that does."""

    def test_r_cal_is_the_smallest_multiple_of_200_that_passes_a_calibrated_cell(self) -> None:
        def passes(replicates: int) -> float:
            return calibration_pass_probability(replicates, true_se_ratio=1.0, se_noise=SE_NOISE)

        smallest = next(r for r in range(R_STEP, 20_000, R_STEP) if passes(r) >= ROW_TARGET)
        assert smallest == study.R_CAL
        assert passes(study.R_CAL) >= ROW_TARGET > passes(study.R_CAL - R_STEP)

    def test_the_band_row_passes_on_the_coverage_rule_at_r_cal(self) -> None:
        probability = calibration_pass_probability(
            study.R_CAL, true_se_ratio=1.0, se_band=(0.0, 10.0), se_noise=0.0
        )
        assert probability >= ROW_TARGET

    def test_every_calibration_row_passes_at_its_scene_count(self) -> None:
        for reader in study.READERS:
            if reader.family not in ("interval_calibration", "simultaneous_coverage"):
                continue
            scene = study.SCENES[reader.scene]
            assert scene.replicates == study.R_CAL
            probability = calibration_pass_probability(
                scene.replicates, true_se_ratio=1.0, se_noise=SE_NOISE
            )
            assert probability >= ROW_TARGET, reader.cell

    def test_r_bias_is_the_smallest_multiple_of_200_that_passes_an_unbiased_row(self) -> None:
        def passes(replicates: int) -> float:
            return support.bias_pass_probability(0.0, replicates)

        smallest = next(r for r in range(R_STEP, 20_000, R_STEP) if passes(r) >= ROW_TARGET)
        assert smallest == study.R_BIAS

    def test_each_bias_scene_runs_the_smallest_count_that_passes_its_rows(self) -> None:
        """A scene runs the count its rows need, or the larger count of a cell it also hosts."""
        raised = {}
        scenes = {r.scene for r in bias_readers() if not r.scene.startswith("p_ladder_")}
        for scene in sorted(scenes):
            needed = scene_replicates(scene)
            if needed != study.R_BIAS:
                raised[scene] = needed
            hosted = study.SCENES[scene]
            assert hosted.replicates >= needed, scene
            if scene not in HOSTS_LARGER:
                assert hosted.replicates == needed, scene
        assert raised == study.R_RAISED
        for scene, count in HOSTS_LARGER.items():
            assert study.SCENES[scene].replicates == count, scene

    def test_every_positive_bias_row_passes_and_every_control_fails_at_its_count(self) -> None:
        routed = routed_cells()
        for reader in bias_readers():
            result = bias_of(reader)
            if isinstance(result, str) or f"{reader.family}/{reader.cell}" in routed:
                continue
            scene = study.SCENES[reader.scene]
            replicates = (
                scene.extra_replicates if reader.fit in scene.extra_fits else scene.replicates
            )
            assert _passes(reader, result["b"], replicates) >= ROW_TARGET, reader.cell

    def test_r_ladder_is_the_smallest_multiple_of_200_that_passes_every_rung_and_the_slope(
        self,
    ) -> None:
        rungs = {n: specs()[f"p_ladder_{n}"] for n in study.LADDER_SIZES}
        b = {
            n: spec.population.key_bias("ate", n, projected=True)["b"] for n, spec in rungs.items()
        }

        def ok(replicates: int) -> bool:
            return (
                all(
                    support.bias_pass_probability(value, replicates) >= ROW_TARGET
                    for value in b.values()
                )
                and slope_pass_probability(replicates) >= STRICT_TARGET
            )

        smallest = next(r for r in range(R_STEP, 20_000, R_STEP) if ok(r))
        assert smallest == study.R_LADDER
        assert not ok(study.R_LADDER - R_STEP)

    def test_the_ladder_sizes_are_the_plan_declared_three_in_a_geometric_ladder(self) -> None:
        sizes = np.array(study.LADDER_SIZES, dtype=float)
        assert tuple(study.LADDER_SIZES) == PLAN_SIZES["LADDER_SIZES"]
        assert np.allclose(np.diff(np.log(sizes)), np.log(3.0))

    def test_the_spread_cell_excludes_one_with_probability_095_at_r_vl(self) -> None:
        spec = specs()["vl1_glm"]
        population = spec.population
        asymptotic = population.key_sd("ey1") / population.key_sd("ey1", projected=True)
        assert asymptotic == pytest.approx(1.1145, abs=0.0005)
        sizes = sorted(VERIFIER_RATIOS)
        low, high = sizes[0], sizes[-1]
        weight = (math.log(study.N_VL) - math.log(low)) / (math.log(high) - math.log(low))
        finite = VERIFIER_RATIOS[low] + weight * (VERIFIER_RATIOS[high] - VERIFIER_RATIOS[low])
        assert 1.08 < finite < asymptotic
        smallest = next(
            r
            for r in range(R_STEP, 20_000, R_STEP)
            if support.spread_exclusion_probability(finite, r) >= STRICT_TARGET
        )
        assert smallest == study.R_VL
        assert support.spread_exclusion_probability(finite, study.R_VL) >= STRICT_TARGET

    def test_the_spread_companion_passes_as_a_positive_bias_row(self) -> None:
        reader = next(r for r in study.READERS if r.cell == "ey1__parametric_q_ne_qr")
        assert reader.family == "bias" and reader.role == "positive"
        assert support.bias_pass_probability(0.0, study.R_VL) >= ROW_TARGET


def slope_pass_probability(replicates: int) -> float:
    """The chance the fitted slope of log SD on log n lies in -0.5 +- 0.125 and excludes -0.25.

    The slope is a contrast of the rung log SDs, each with variance ``1 / (2 (R - 1))``.  The
    99% interval is the slope plus or minus 2.576 of its SE, and it must lie inside the band.
    """
    from scipy.stats import norm

    x = np.log(np.array(study.LADDER_SIZES, dtype=float))
    weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    scale = float(np.sqrt(np.sum(weights**2) / (2.0 * (replicates - 1))))
    half = 2.576 * scale
    margin = 0.125 - half
    inside = norm.cdf(margin / scale) - norm.cdf(-margin / scale)
    excludes = 1.0 if half < (0.5 - 0.25) else 0.0
    return float(inside * excludes)


def expected_chance_reds() -> dict[str, float]:
    """The expected number of red verdict rows when every row behaves as designed.

    Each verdict row has a design pass probability ``p`` (item 4), so it is red by chance with
    probability ``1 - p``.  The rows are independent in this sum.  Rows that share draws are
    positively correlated, which raises the chance that none is red and lowers the expected
    count by nothing, so the sum is the expected count.  Rows pre-routed to reporting add
    nothing, because a red there is the declared reading.
    """
    routed = set(routed_cells()) | set(study.PRE_ROUTED)
    calibration = 0.0
    for reader in study.READERS:
        if reader.family in ("interval_calibration", "simultaneous_coverage"):
            scene = study.SCENES[reader.scene]
            if reader.family == "simultaneous_coverage":
                p = calibration_pass_probability(
                    scene.replicates, true_se_ratio=1.0, se_band=(0.0, 10.0), se_noise=0.0
                )
            else:
                p = calibration_pass_probability(
                    scene.replicates, true_se_ratio=1.0, se_noise=SE_NOISE
                )
            calibration += 1.0 - p
    bias = 0.0
    for reader in bias_readers():
        result = bias_of(reader)
        if isinstance(result, str) or f"{reader.family}/{reader.cell}" in routed:
            continue
        scene = study.SCENES[reader.scene]
        replicates = scene.extra_replicates if reader.fit in scene.extra_fits else scene.replicates
        bias += 1.0 - _passes(reader, result["b"], replicates)
    table = necessity_table()
    for label, row in table.items():
        replicates = study.SCENES[study.NECESSITY_LABELS[label][0]].replicates
        bias += 1.0 - displacement_probability(row, replicates)
    bias += 1.0 - slope_pass_probability(study.R_LADDER)
    return {"calibration": calibration, "bias_ladder_necessity": bias}


class TestChanceReds:
    """The record states how many red rows a correct estimator is expected to show."""

    def test_the_declared_expected_counts_are_the_computed_ones(self) -> None:
        got = expected_chance_reds()
        assert {k: round(v, 1) for k, v in got.items()} == study.EXPECTED_CHANCE_REDS
        assert round(sum(got.values()), 1) == study.EXPECTED_CHANCE_REDS_TOTAL

    def test_a_reading_exists_for_each_kind_of_red_row(self) -> None:
        kinds = {"calibration", "bias", "ladder", "necessity", "control", "primary"}
        assert kinds <= set(study.RED_READING)


# ------------------------------------------------------------------------------------ item 5


def distinct_counts() -> dict[str, int]:
    """The number of distinct mechanism values of each saturated basis."""
    two, l1, l3 = composite_law.TWO, mar.LAWS["l1"], mar.LAWS["l3"]
    return {
        "cn_obs": len(np.unique(np.round(l1.pi, 10))),
        "cn_zero": len(np.unique(np.round(l1.g[:, 0] * l1.pi[:, 0], 10))),
        "rm_d": len({round(float(x), 10) for x in specs()["rm"].population.terms["d"].limit}),
        "ma_arm0": len(np.unique(np.round(two.composite[:, 0], 10))),
        "ma_arm1": len(np.unique(np.round(two.composite[:, 1], 10))),
        "c_l1_arm1": len(np.unique(np.round(l1.composite[:, 1], 10))),
        "c_l3_arm0": len(np.unique(np.round(l3.composite[:, 0], 10))),
    }


@functools.cache
def ma_ratio(n: int) -> float:
    """The identifiability ratio of MA arm 1 at size ``n``: four seeds of :data:`FOLD_DRAWS` draws."""
    two = composite_law.TWO
    draws = [
        support.composite_gap_ratios(
            composite_law.P_W,
            two.g,
            two.pi_treatment,
            two.pi,
            n=n,
            draws=FOLD_DRAWS,
            seed=seed,
        )["arm1"]
        for seed in (100, 101, 102, 103)
    ]
    return float(np.mean(draws))


class TestIdentifiability:
    """Item 5: the saturated bases, and the fold-to-fold SD of the cross-fitted finite laws."""

    def test_the_distinct_values_allow_the_saturated_basis(self) -> None:
        counts = distinct_counts()
        assert counts["cn_obs"] == 6 and counts["cn_zero"] == 3
        assert counts["ma_arm0"] == counts["ma_arm1"] == 3
        assert counts["c_l1_arm1"] == counts["c_l3_arm0"] == 3
        # The degree caps at distinct - 1, which is the saturated regression in sample and for
        # a known mechanism.  A cross-fitted estimated mechanism trains each fold on up to
        # (N_FOLDS - 1) * distinct leave-two-out values, so there the cap does not saturate.
        assert min(study.DEGREE["cn"], counts["cn_obs"] - 1) == counts["cn_obs"] - 1

    def test_n_ma_cf_is_the_smallest_multiple_of_500_that_clears_the_threshold_with_margin(
        self,
    ) -> None:
        """The plan's probe moved by 0.03 between seeds at 200 draws, so the margin is 0.03.

        At 1,000 draws the seed spread is 0.007.  Size 6,500 clears the bare threshold of 1.05
        (ratio 1.061) but not the margin.  Size 7,000 clears both (ratio 1.096).
        """
        assert PLAN_SIZES["N_MA_CF"] == study.N_MA_CF
        required = FOLD_THRESHOLD + FOLD_MARGIN
        smallest = next(n for n in range(5_000, 12_000, 500) if ma_ratio(n) >= required)
        assert smallest == study.N_MA_CF
        assert ma_ratio(6_500) >= FOLD_THRESHOLD
        assert ma_ratio(6_000) < FOLD_THRESHOLD
        assert ma_ratio(2_000) < 1.0

    def test_the_natural_course_cells_are_identified_at_n_cn(self) -> None:
        l1 = mar.LAWS["l1"]
        ratios = support.composite_gap_ratios(
            l1.p_w, l1.g, None, l1.pi, n=study.N_CN, draws=FOLD_DRAWS
        )
        assert ratios["response"] >= FOLD_THRESHOLD
        assert ratios["arm0"] >= FOLD_THRESHOLD


# ------------------------------------------------------------------------------------ item 6


def separation(spec: Spec, keys: list[str], extra: dict[str, float] | None = None) -> float:
    """The smallest gap between two truths, in the larger SD of the pair over ``sqrt(n)``."""
    population = spec.population
    truth = {key: population.key_truth(key) for key in keys}
    sd = {key: population.key_sd(key, projected=spec.projected) for key in keys}
    if extra:
        truth.update(extra)
    out = math.inf
    names = sorted(truth)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            larger = max(sd.get(left, 0.0), sd.get(right, 0.0))
            out = min(out, abs(truth[left] - truth[right]) / larger * math.sqrt(spec.n))
    return out


class TestRegimeSeparation:
    """Item 6: regime truths are apart by more than the cell's rule can resolve."""

    def test_pd_calibration_regimes_clear_the_calibration_floor(self) -> None:
        spec = specs()["pd_known"]
        keys = [
            f"ey_regime[{name}]" for name in (laws.NEVER_NAME, laws.RULE_NAME, laws.STOCHASTIC_NAME)
        ]
        assert separation(spec, keys) >= CALIBRATION_SEPARATION

    def test_pd_bias_regimes_clear_the_bias_floor(self) -> None:
        spec = specs()["pd_rate"]
        keys = [
            f"ey_regime[{name}]" for name in (laws.NEVER_NAME, laws.RULE_NAME, laws.STOCHASTIC_NAME)
        ]
        floor = 0.25 + 2.576 / math.sqrt(study.SCENES["pd_rate"].replicates)
        assert separation(spec, keys) >= floor

    def test_rm_regimes_and_the_natural_course_clear_the_bias_floor(self) -> None:
        spec = specs()["rm"]
        keys = ["ey_regime[d]", "ey_regime[s]"]
        never = {"ey_regime[never]": mar.truths(mar.LAWS["l1"])["ey0"]}
        floor = 0.25 + 2.576 / math.sqrt(study.SCENES["rm"].replicates)
        assert separation(spec, keys, never) >= floor

    def test_ma_regime_and_arms_clear_the_bias_floor(self) -> None:
        spec = specs()["ma"]
        keys = ["ey0", "ey1", "ey_regime[d]"]
        floor = 0.25 + 2.576 / math.sqrt(study.SCENES["ma_insample"].replicates)
        assert separation(spec, keys) >= floor


# ------------------------------------------------------------------------------------ item 7


class TestBasisError:
    """Item 7: ``L2(Q_0 - Qr_0)`` and the declared-basis error, by law and key."""

    def test_the_h1_laws_have_no_gap_and_the_others_do(self) -> None:
        def gap(name: str, term: str, *, declared: bool) -> float:
            return specs()[name].population.l2_gap(term, declared=declared)

        for term in ("a0", "a1"):
            assert gap("p_correct", term, declared=False) < 1e-9
            assert gap("p_correct", term, declared=True) < 1e-9
        assert gap("pd_known", "a0", declared=True) < 1e-9
        for term in ("a1", "d", "s1"):
            assert gap("pd_known", term, declared=False) > 1e-3, term
        assert gap("vl1_glm", "a1", declared=False) == pytest.approx(0.329, abs=0.001)
        assert gap("vl1_glm", "a1", declared=True) == pytest.approx(0.329, abs=0.001)
        assert gap("vl0_lsb", "a1", declared=True) < 1e-9
        for term in ("high", "low", "medium"):
            assert gap("m_correct", term, declared=False) > 1e-3, term
        for term in ("obs", "zero"):
            assert gap("cn", term, declared=True) < 1e-9
        for term in ("d", "s1", "s0"):
            assert gap("rm", term, declared=True) < 1e-9
        for term in ("a0", "a1", "d"):
            assert gap("ma", term, declared=True) < 1e-9


# ------------------------------------------------------------------------------------ item 8


class TestArmMeansStayInTheInterval:
    """Item 8: the chance a binary arm mean leaves ``(0, 1)`` at the declared size."""

    def test_the_chance_is_negligible_for_every_term(self) -> None:
        for name, spec in specs().items():
            population = spec.population
            for term_name, term in population.terms.items():
                if not term.binary:
                    continue
                mass = population.prob * term.ind
                p1 = float(np.sum(mass * term.target))
                p0 = float(np.sum(mass * (1.0 - term.target)))
                leave = (1.0 - p1) ** spec.n + (1.0 - p0) ** spec.n
                assert leave < 1e-12, (name, term_name, leave)


# ---------------------------------------------------------------------------------- the sizes


def unit_rule(n_minimum: float) -> int:
    """The smallest size in :data:`SIZE_LADDER` with :data:`UNIT_ROWS` rows in the smallest cell."""
    return next(n for n in SIZE_LADDER if n * n_minimum >= UNIT_ROWS)


class TestSizes:
    """The sizes: the plan's, verified by the items; the rest by :func:`unit_rule`."""

    def test_the_plan_declared_sizes_are_the_study_sizes(self) -> None:
        for name, value in PLAN_SIZES.items():
            assert getattr(study, name) == value, name

    def test_laws_c_wt_and_st_follow_the_unit_rule(self) -> None:
        smallest_c = min(float(mar.LAWS[k].p_w.min()) for k in ("l1", "l3"))
        smallest_wt = float(weighted.SELECTED_P_W.min())
        smallest_st = float(
            min(
                stratified.P_V[v] * stratified.P_W_GIVEN_V[v, w] for v in range(3) for w in range(3)
            )
        )
        assert unit_rule(smallest_c) == study.N_C
        assert unit_rule(smallest_wt) == study.N_WT
        assert unit_rule(smallest_st) == study.N_ST
        # The smaller size in the ladder fails the rule where the declared one is not the first.
        assert study.N_ST == 3_000 and 2_000 * smallest_st < UNIT_ROWS


# ----------------------------------------------------------------------- the record is complete


class TestRecord:
    """The record names every module, cell and description the design needs."""

    def test_the_manifest_lists_every_study_module_the_runner_and_properties_import(self) -> None:
        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_declared_cells_and_the_record_agree(self) -> None:
        declared = {(c.property, c.cell) for c in properties.declared_cells()}
        recorded = {
            (family, cell)
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
            if family != "root_n_rate"
        }
        assert declared == recorded

    def test_every_declared_truth_is_the_law_s_own(self) -> None:
        for cell in properties.declared_cells():
            truth = cell.dgp.truth()
            assert cell.estimand in truth, (cell.property, cell.cell)


# ------------------------------------------------------------------------ convergence is reported


class TestConvergenceAccounting:
    """A fit that stops at ``max_iter`` keeps its rows, and the summary counts it."""

    def test_the_row_columns_read_every_fluctuation(self) -> None:
        def fluctuation(n_iter: int, converged: bool) -> SimpleNamespace:
            return SimpleNamespace(n_iter=n_iter, converged=converged)

        result = SimpleNamespace(
            repeats=[
                SimpleNamespace(
                    fluctuations={"a": fluctuation(3, True), "b": fluctuation(7, False)}
                )
            ]
        )
        assert properties.convergence(result) == {"rounds": 7, "converged": 0}
        result.repeats[0].fluctuations["b"] = fluctuation(2, True)
        assert properties.convergence(result) == {"rounds": 3, "converged": 1}

    def test_a_capped_fit_reports_itself_unconverged(self) -> None:
        """Nonzero witness on the shipped estimator: one round against the study's ``tol``.

        Cross-fitted, so the loop does not start at its fixed point: an in-sample cell frequency
        already solves the score, and its fit takes no round.
        """
        rng = np.random.default_rng(3)
        w = rng.integers(0, 3, size=400)
        a = rng.binomial(1, np.array([0.3, 0.55, 0.75])[w])
        y = rng.binomial(1, 0.2 + 0.2 * w / 2 + 0.1 * a)
        frame = pd.DataFrame({"Y": y.astype(float), "A": a.astype(float), "W": w.astype(float)})

        def fit(max_iter: int) -> Any:
            return AdaptiveIPTW(
                treatment_learner=laws.CellFrequency(),
                reduced_outcome_learner=laws.DesignBasisGLM(1),
                cross_fit=True,
                n_folds=3,
                random_state=0,
                g_bounds=study.G_BOUNDS,
                max_iter=max_iter,
                tol=study.TOL,
                estimands=["ey1"],
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            capped = fit(1).fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()
            free = (
                fit(study.MAX_ITER)
                .fit(frame, outcome="Y", treatment="A", covariates=["W"])
                .single()
            )
        assert properties.convergence(capped) == {"rounds": 1, "converged": 0}
        assert properties.convergence(free)["converged"] == 1

    def test_the_summary_counts_a_nonconverged_fit_and_old_rows_add_no_column(self) -> None:
        rows = pd.DataFrame(
            {
                "property": ["bias"] * 4,
                "cell": ["x", "x", "y", "y"],
                "rounds": [3, 100, 2, 4],
                "converged": [1, 0, 1, 1],
            }
        )
        summary = pd.DataFrame({"property": ["bias", "bias"], "cell": ["x", "y"]})
        read = properties.convergence_reading(summary, rows)
        assert read["nonconverged_fits"].tolist() == [1.0, 0.0]
        assert read["max_rounds"].tolist() == [100.0, 4.0]
        old = properties.convergence_reading(summary, rows.drop(columns=["rounds", "converged"]))
        assert list(old.columns) == ["property", "cell"]
