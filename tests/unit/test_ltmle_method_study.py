"""Fast structural checks for the registered longitudinal TMLE evidence study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_longitudinal as law
from tests.studies import canonical_ltmle as study
from tests.studies import ltmle_properties as properties


def test_quasibinomial_irls_solves_the_fractional_response_score() -> None:
    design = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.0, -1.0],
            [0.5, 0.5],
            [1.0, -0.5],
            [1.5, 1.0],
        ]
    )
    coefficient = np.array([-0.4, 0.7, -0.3])
    augmented = np.column_stack([np.ones(len(design)), design])
    target = 1.0 / (1.0 + np.exp(-(augmented @ coefficient)))
    fitted = study.QuasiBinomialGLM(tol=1e-12).fit(design, target)
    np.testing.assert_allclose(
        np.concatenate([fitted.intercept_, fitted.coef_[0]]), coefficient, atol=1e-10
    )


def baseline_only(probs: np.ndarray, a1: int, a2: int) -> float:
    """The mean outcome an analysis that never conditions on ``L2`` would report.

    Written longhand off the support rather than by disabling something in the estimator, for
    the reason every deliberate-mutation control in this suite is: a flag on the code under
    audit makes the control a statement about a branch in it.  The only difference from
    :func:`law.functional` is that the outcome regression conditions on ``(W, A1, A2, C2 = 1)``
    with ``L2`` marginalised out of the *observed* law rather than on ``L2`` as well.  Since
    ``A2`` and ``C2`` both depend on ``L2``, conditioning on them reweights it, which is the
    bias a longitudinal fit exists to remove.
    """
    return sum(
        (law._mass(probs, w=w) / law._mass(probs))
        * law._mass(probs, w=w, a1=a1, c1=1, a2=a2, c2=1, y=1)
        / law._mass(probs, w=w, a1=a1, c1=1, a2=a2, c2=1)
        for w in (0, 1)
    )


def test_the_sharp_null_is_exact_and_still_needs_the_longitudinal_adjustment() -> None:
    """Both contrasts are exactly zero, and getting there still takes the whole recursion.

    The last clause is the one that matters and the one the first draft of this law failed.
    With ``P(Y = 1 | W, L2, A2, C2 = 1) = 0.25 + 0.5 W`` the outcome is independent of
    everything but ``W``, so standardising over the baseline alone returns the truth exactly
    and the type-I cell cannot tell a longitudinal fit from one that ignores ``L2``.  A null
    is only a null *for an estimator*; one no estimator has to work for measures nothing.
    """
    assert properties.NULL_TRUTH == 0.0
    assert law.functional(properties.NULL_PROBS, properties.CONTRASTS["dynamic"]) == 0.0
    assert properties.NULL_PROBS.sum() == 1.0

    # Nonzero witnesses: the outcome moves with the baseline, with the time-varying confounder,
    # and with the first arm.  Any one of them constant would collapse a piece of the problem.
    assert law.G1[0] != law.G1[1]
    conditional = {
        (w, l2): law._mass(properties.NULL_PROBS, w=w, a1=1, l2=l2, a2=1, c2=1, y=1)
        / law._mass(properties.NULL_PROBS, w=w, a1=1, l2=l2, a2=1, c2=1)
        for w in (0, 1)
        for l2 in (0, 1)
    }
    assert conditional[(0, 0)] != conditional[(0, 1)], "Y does not move with L2"
    assert conditional[(0, 0)] != conditional[(1, 0)], "Y does not move with W"
    assert properties.NULL_OUTCOME[0, 1, 0, 1] != properties.NULL_OUTCOME[0, 0, 0, 1], (
        "Y does not move with the first arm"
    )

    # And the deliberate-mutation control: a baseline-only standardisation, which is what an
    # analysis that dropped L2 would compute, does *not* recover the null.
    null = properties.NULL_PROBS
    assert abs(baseline_only(null, 1, 1) - baseline_only(null, 0, 0)) > 1e-3


def test_the_untargeted_plug_in_is_the_fit_without_its_fluctuation() -> None:
    """The control is a second implementation, so it has to agree where agreement is forced.

    Hand both the saturated learner and the plug-in stops being a control: on a sample that
    realises the law exactly the initial fit already solves every score, so the fluctuation is
    a no-op and the two must return the same number.  That is what says the difference the
    ``mechanism_correct`` cells report is the targeting step rather than a second estimator
    written differently.
    """
    frame = law.frame()
    result = properties.fit(frame, "both_correct")
    for label in properties.REGIMENS:
        plug_in = properties.untargeted(frame, label, "both_correct")
        assert plug_in == pytest.approx(float(result[f"ey_regimen[{label}]"].psi), abs=1e-9)


def test_the_frozen_r_study_matches_beyond_its_acceptance_margin() -> None:
    rows = pd.read_csv(study.STUDY.artifact("replicates.csv.gz"))
    paired = rows.pivot(
        index=["scenario", "replicate", "estimand"],
        columns="implementation",
        values=["estimate", "std_error", "initial_estimate"],
    )
    tolerances = {"estimate": 3e-7, "std_error": 1e-8, "initial_estimate": 3e-7}
    for column, tolerance in tolerances.items():
        difference = paired[(column, "cleverly")] - paired[(column, "ltmle")]
        assert np.max(np.abs(difference)) < tolerance


def test_the_primary_dynamic_plan_assigns_both_second_node_arms() -> None:
    frame, _ = study.draw_scenario(study.SCENARIO, study.PRIMARY_N, 0)
    result = study.fit_cleverly(frame)
    assignment = result.fits[study.RULE_LABEL].assignment[:, 1]
    assert set(np.unique(assignment)) == {0.0, 1.0}
