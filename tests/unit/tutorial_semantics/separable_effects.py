"""The reviewed semantic callback for ``docs/examples/separable-effects``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`, and the slow tier adds :func:`check_refits`.
Each assertion pins a reading the prose makes about this draw.  A relation on one draw certifies
no method.  The true values come from :func:`cleverly.datasets.heart_failure_truth`, whose own
test (``tests/unit/test_datasets_heart_failure.py``) checks them against a Monte Carlo draw of
the world where the two parts of the dose are set apart.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly.datasets import heart_failure_protocol, heart_failure_truth
from tests.unit.tutorial_semantics import EXAMPLES, assert_protocol_recorded, covers

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refute",)

NOTEBOOK = EXAMPLES / "separable-effects.ipynb"

PLANS = ("high dose", "no pill", "high dose with supplement")
PARTS = ("whole medicine", "fluid part", "potassium part")

#: Decimals the prose writes that no stored output prints, each with its reason.
UNPRINTED_DECIMALS = {
    "1.3": "the upper end of the scatter-ratio range that 20 refits easily give, 0.7 to 1.3, a "
    "reading aid rather than a result; _check_data asserts the ratio lies inside it",
}


def _near(value: float, printed: float, places: int) -> bool:
    """``value`` prints as ``printed`` to ``places`` decimals, with slack for library rounding."""
    return abs(value - printed) <= 0.6 * 10.0**-places


def _day90(readmission_dose: int, death_dose: int, first: str = "death") -> Any:
    return heart_failure_truth(readmission_dose, death_dose, first=first).set_index("day").loc[90]


def _check_data(namespace: dict[str, Any]) -> None:
    frame, known = namespace["frame"], namespace["known"]
    assert frame.shape == (8_000, 5)
    assert list(frame.columns) == ["older", "prior_admissions", "dose", "day", "event"]
    # "such as day 70 for the death in row 0".
    assert frame.loc[0, "event"] == 2 and frame.loc[0, "day"] == 70.0
    # "when event is 0, day is the last check that found the patient with neither".
    assert set(frame.loc[frame["event"].eq(0), "day"]) <= {30.0, 60.0, 90.0}
    # The known risks are the exact truth of each plan.
    for plan, doses in zip(PLANS, ((2, 2), (0, 0), (2, 0)), strict=True):
        exact = _day90(*doses)
        for cause in ("readmission", "death"):
            assert known.loc[plan, cause] == pytest.approx(exact[cause], abs=1e-15)


def _check_association(namespace: dict[str, Any]) -> None:
    frame, known, shares = namespace["frame"], namespace["known"], namespace["shares"]
    means = frame.groupby("dose")[["older", "prior_admissions"]].mean()
    # "High-dose patients had more prior admissions and were less often 75 or older."
    assert means.loc[2.0, "prior_admissions"] > means.loc[0.0, "prior_admissions"]
    assert means.loc[2.0, "older"] < means.loc[0.0, "older"]
    # "the simple readmission difference, -0.1358, falls short of the true -0.1770".
    simple = shares.loc[2.0, 1] - shares.loc[0.0, 1]
    true = known.loc["high dose", "readmission"] - known.loc["no pill", "readmission"]
    assert _near(simple, -0.1358, 4) and _near(true, -0.1770, 4)
    assert true < simple < 0.0


def _check_fit(namespace: dict[str, Any]) -> None:
    protocol, result, risk = namespace["protocol"], namespace["result"], namespace["risk"]
    assert protocol == heart_failure_protocol()
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, result)
    assert namespace["design"].grid == (30.0, 60.0, 90.0)
    assert result.data.n_times == 3 and result.data.treatment_held
    # "Every estimate reads supported, which means a simulation study checks this kind of fit."
    for name in result:
        support = result[name].support
        assert support.label == "supported" and support.row == "G49.in_sample"
    # "0.1935, against 0.3792 with no pill".
    assert _near(result[risk("high dose", "readmission")].psi, 0.1935, 4)
    assert _near(result[risk("no pill", "readmission")].psi, 0.3792, 4)
    # "the supplement plan reads 2,150 patients in the last month: the high-dose and no-pill
    # patients together".
    last = namespace["support"].query("cause == 'readmission' and time == 3")
    followed = last.set_index("regimen")["n_followed"]
    assert followed["high dose with supplement"] == 2_150
    assert followed["high dose with supplement"] == followed["high dose"] + followed["no pill"]


def _check_effects(namespace: dict[str, Any]) -> None:
    effects, known = namespace["effects"], namespace["known"]
    pairs = namespace["pairs"]
    assert tuple(pairs) == PARTS
    for cause in ("readmission", "death"):
        # The whole medicine is the sum of the two parts' differences.
        whole = effects[cause, "whole medicine"].psi
        parts = effects[cause, "fluid part"].psi + effects[cause, "potassium part"].psi
        assert whole == pytest.approx(parts, abs=1e-12)
        for part, (plan, base) in pairs.items():
            true = known.loc[plan, cause] - known.loc[base, cause]
            # At a glance: "each interval contains its value", for the three it quotes; the
            # other three rows hold on this draw too.
            assert covers(effects[cause, part], true)
    readmission = {part: 100 * effects["readmission", part].psi for part in PARTS}
    for part, printed in zip(PARTS, (-18.6, -13.6, -5.0), strict=True):
        assert _near(readmission[part], printed, 1)
    # The potassium part's share is clear of zero: "(-5.8 to -4.1)".
    assert effects["readmission", "potassium part"].ci[1] < 0.0
    death = {part: 100 * effects["death", part].psi for part in PARTS}
    for part, printed in zip(PARTS, (23.4, 1.0, 22.5), strict=True):
        assert _near(death[part], printed, 1)
    # "The fluid part adds +1.0, because patients who avoid readmission stay at risk of dying
    # first": the supplement plan's death hazard is the no-pill one, so the exact truth moves
    # only through the readmission hazard.
    assert effects["death", "fluid part"].ci[0] > 0.0
    assert _day90(2, 0)["death"] > _day90(0, 0)["death"]


def _check_assessment(namespace: dict[str, Any]) -> None:
    assessment, support = namespace["assessment"], namespace["support"]
    assert not assessment.attention
    ledger = assessment.to_frame().set_index("check")["status"]
    assert ledger["score_equations"] == "passed"
    # "The largest weight is 15.2, in the supplement plan's last month".
    top = support.loc[support["max_weight"].idxmax()]
    assert top["regimen"] == "high dose with supplement" and top["time"] == 3
    assert _near(float(top["max_weight"]), 15.2, 1)
    assert _near(namespace["ratio"], 0.803, 3)
    assert (support["share_truncated"] == 0.0).all()
    assert bool(support["converged"].all())


def _check_evalue(namespace: dict[str, Any]) -> None:
    bound, result, risk = namespace["bound"], namespace["result"], namespace["risk"]
    high, none = (result[risk(plan, "readmission")].psi for plan in ("high dose", "no pill"))
    # "the high-dose readmission risk is 0.510 times the no-pill risk"; "3.33 ... and 2.88".
    assert bound.scale == "risk ratio"
    assert bound.risk_ratio == pytest.approx(high / none, rel=1e-9)
    assert _near(bound.risk_ratio, 0.510, 3)
    assert _near(bound.point, 3.33, 2) and _near(bound.limit, 2.88, 2)
    assert "does not affect censoring" in bound.note
    assert "read it as descriptive" in str(bound.summary())
    # "cleverly has no such bound for the supplement plan" and the made-up confounder "refuses
    # this fit".
    from cleverly.exceptions import CapabilityError

    with pytest.raises(Exception, match="not static arms"):
        result.sensitivity.evalue(
            (risk("high dose with supplement", "readmission"), risk("no pill", "readmission"))
        )
    from cleverly.sensitivity import ConfounderStrengthGrid

    with pytest.raises(CapabilityError, match="common cause of both events"):
        result.sensitivity.simulated_confounding(
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1))
        )


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    moves = {(m["left out"], m["readmission difference"]): m for m in namespace["moves"]}
    # "Leaving out prior_admissions moves the fluid part's effect from -0.1359 to -0.0814, or
    # 3.3 standard errors, and the whole medicine's by 3.1".
    fluid = moves["prior_admissions", "fluid part"]
    assert _near(fluid["with both"], -0.1359, 4) and _near(fluid["left out one"], -0.0814, 4)
    assert _near(fluid["move in SEs"], 3.3, 1)
    assert _near(moves["prior_admissions", "whole medicine"]["move in SEs"], 3.1, 1)
    # "Leaving out older moves each by less than half a standard error."
    for part in ("whole medicine", "fluid part"):
        assert abs(moves["older", part]["move in SEs"]) < 0.5


def _check_order(namespace: dict[str, Any]) -> None:
    other, result = namespace["other_order"], namespace["result"]
    names = namespace["names"]
    estimates = {}
    for order, fit in (("death", result), ("readmission", other)):
        estimate = fit.contrast(lambda psi: psi[0] - psi[1], names, name=order)
        true = _day90(2, 0, order)["readmission"] - _day90(0, 0, order)["readmission"]
        # "each near its own true value, -13.1 and -15.8".
        assert covers(estimate, true)
        estimates[order] = estimate
    assert _near(100 * estimates["death"].psi, -13.6, 1)
    assert _near(100 * estimates["readmission"].psi, -16.5, 1)
    # The two orders' truths differ: "they answer two questions".
    assert abs(_day90(2, 0, "readmission")["readmission"] - _day90(2, 0)["readmission"]) > 0.02


def check(namespace: dict[str, Any]) -> None:
    """Pin the readings of the separable fit on its seeded draw."""
    _check_data(namespace)
    _check_association(namespace)
    _check_fit(namespace)
    _check_effects(namespace)
    _check_assessment(namespace)
    _check_evalue(namespace)
    _check_sensitivity(namespace)
    _check_order(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation refits of the fluid part's effect on readmission."""
    contrast_fit, fluid_name = namespace["contrast_fit"], namespace["fluid_name"]
    fluid = namespace["effects"]["readmission", "fluid part"]
    # The typed contrast that `refute` reads is the `result.contrast` difference.
    assert contrast_fit[fluid_name].psi == pytest.approx(fluid.psi, abs=1e-6)
    assert contrast_fit[fluid_name].support.label == "supported"
    tests = {test.name: test for test in namespace["refutation"].tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause"]
    for test in tests.values():
        assert test.verdict == "descriptive" and test.passed is None
        assert test.n_replicates == 20
    # "The random doses give a mean of -0.0020, the noise column barely moves the estimate, and
    # the subset refits spread with a scatter ratio of 0.91".
    assert _near(tests["placebo"].mean, -0.0020, 4)
    assert abs(tests["random_common_cause"].statistic) < 0.05
    assert _near(tests["subset"].statistic, 0.91, 2)
    assert 0.7 < tests["subset"].statistic < 1.3
