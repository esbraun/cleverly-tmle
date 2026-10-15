"""The two readings of a modified-treatment-policy functional (X40 part (d), T-d1).

Díaz, Williams, Hoffman and Schenck (2023), Theorem 1: under strong sequential randomization
(Assumption 3) the functional is the modified treatment policy effect, and under standard
sequential randomization (Assumption 2) the same number is the LMTP-SI.  Williams and Díaz
(2025) correct the software paper's exchangeability to the strong form and leave positivity
unchanged.  The point and the longitudinal identification records carry both readings, each
with its own exchangeability assumption and source, and ``IdentifiedEffect.summary()`` prints
both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleverly import (
    CausalStudy,
    LongitudinalTreatment,
    ModifiedTreatmentPolicy,
    PointTreatment,
    RegimeContrast,
)
from cleverly.interventions import Shift
from cleverly.targets import TARGETS, Interpretation

from .. import law_q_point as lawp
from .. import law_q_tuned as lawq
from .. import longitudinal_mtp as mtp

POLICY = "modified treatment policy effect"
STOCHASTIC = "stochastic intervention (LMTP-SI)"
SOURCE = Path(__file__).resolve().parents[2] / "src" / "cleverly"


def _point_effect() -> object:
    study = CausalStudy(
        lawp.sample(200, 1),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )
    shifts = (Shift(0.0, cap=None), Shift(0.5, cap=None))
    return study.identify(ModifiedTreatmentPolicy(policies=shifts))


def _longitudinal_effect() -> object:
    study = CausalStudy(
        lawq.sample(200, 1),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W",),
            time_varying=((), ("L2",)),
            continuous_treatment=("A1", "A2"),
        ),
    )
    return study.identify(RegimeContrast(mtp.regimens(("natural", "up")), reference="natural"))


def _readings(effect: object) -> dict[str, Interpretation]:
    readings = effect.identification.interpretations  # type: ignore[attr-defined]
    assert all(isinstance(item, Interpretation) for item in readings)
    return {item.label: item for item in readings}


@pytest.mark.parametrize("build", [_point_effect, _longitudinal_effect])
def test_t_d1_each_record_carries_both_readings_with_their_own_exchangeability(
    build: object,
) -> None:
    """T-d1: two labels, two distinct exchangeability texts, Theorem 1 (i) and (ii) cited."""
    effect = build()  # type: ignore[operator]
    readings = _readings(effect)
    assert list(readings) == [POLICY, STOCHASTIC]
    strong, standard = readings[POLICY], readings[STOCHASTIC]
    assert strong.assumptions != standard.assumptions
    assert strong.assumptions[0].startswith("strong sequential randomization")
    assert standard.assumptions[0].startswith("standard sequential randomization")
    assert any("Theorem 1 (ii) and Assumption 3" in ref for ref in strong.references)
    assert "Williams & Díaz (2025)" in strong.references
    assert any("Theorem 1 (i) and Assumption 2" in ref for ref in standard.references)
    assert strong.meaning != standard.meaning
    summary = effect.summary()  # type: ignore[attr-defined]
    assert "interpretations, each with the assumption it adds:" in summary
    for item in (strong, standard):
        assert f"  - {item.label}: {item.meaning}" in summary
        assert f"      needs {item.assumptions[0]}" in summary
    lines = effect.summary_lines()  # type: ignore[attr-defined]
    assert sum(line.startswith("interpretation ") for line in lines) == 2


def test_t_d1_the_point_record_says_the_two_exchangeabilities_coincide() -> None:
    """At one node Assumption 3 has no later dose, so the point record says they coincide."""
    for item in _readings(_point_effect()).values():
        assert (
            "at one node the strong and the standard sequential randomization coincide"
            in (item.assumptions[0])
        )


def test_t_d1_the_longitudinal_readings_differ_beyond_one_node() -> None:
    """The longitudinal strong form names the later doses, which the standard form omits."""
    readings = _readings(_longitudinal_effect())
    assert "later covariates and doses" in readings[POLICY].assumptions[0]
    assert "later covariates and doses" not in readings[STOCHASTIC].assumptions[0]
    assert "coincide" not in " ".join(readings[POLICY].assumptions)


def test_t_d1_only_the_policy_targets_carry_readings() -> None:
    """An arm target has one reading, so its record carries none."""
    carrying = {name for name, target in TARGETS.items() if target.identification.interpretations}
    assert carrying == {"ey_policy", "ate_policy"}


def test_t_d1_the_single_reading_comment_is_gone() -> None:
    """The source no longer calls the functional one number with two readings."""
    hits = [
        path
        for path in SOURCE.rglob("*.py")
        if "One number, two readings" in path.read_text(encoding="utf-8")
    ]
    assert hits == []
