r"""The survival influence curve, against a numerically differentiated functional.

``tests/discrete_law_survival.py`` holds a two-time-point law with an absorbing event at
each node, a sample that realises it exactly, and a closed-form statement of the
cumulative risk at each horizon.  Its Gateaux derivative -- taken by complex step, so it
is exact to double precision -- *is* the efficient influence function, and it is derived
from the functional alone: no cumulative product, no clever covariate, nothing the
library supplies.

This is the module that says the survival derivation is right, and it carries the whole
claim, because the thing a survival outcome adds is a *population*: at every node the
regression is fitted on the units at risk entering it, which is one event node earlier
than the censoring factor runs to.  A unit that has the event at ``t`` belongs in node
``t``'s regression -- it is the observation that the event happened -- and not in node
``t + 1``'s.  Get that one index wrong and the fit still converges, every score still
comes back at ``1e-16``, and the answer is quietly biased.  Nothing but a comparison
against an independently written truth catches it, which is why the two negative controls
at the foot are here and why each is checked to be capable of failing.

Two deliberate mutations were run against this module and both were seen to fail it, which
is the only thing that makes the claims above evidence rather than assertion:

* ``following(t)`` reading ``event_free_through(t)`` instead of ``t - 1`` -- the tidying a
  reader is most likely to make, since the *censoring* factor really does run to ``t``.
  It drops every unit that had the event from its own node's regression, and it takes 26
  of the 30 tests here with it.
* ``fit_mechanism`` fitting on the units that had already had the event.  Their ``A_t`` is
  missing and the design fills a missing arm with zero, so they train as untreated
  observations and bias ``g``.  This one is **silent**: it leaves every point estimate
  green -- with an exact initial fit ``epsilon`` is zero, ``psi`` is the plug-in, and no
  error in the mechanism can move it -- and is caught by the Gateaux comparison alone, at
  the five parameters whose horizon reaches the second node.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE

from .. import discrete_law_survival as law

#: Truncation wide enough never to bind: the law's conditionals all lie in [0.25, 0.75].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)

COLUMNS = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _oracle_fit(frame: object, **overrides: object) -> object:
    """A fit of ``frame`` with the saturated learner at every node."""
    settings: dict[str, object] = {
        "reference": law.REGIMEN_REFERENCE,
        "outcome_learner": law.CellMeans(),
        "pseudo_learner": law.CellMeans(),
        "treatment_learner": law.CellMeans(),
        "censoring_learner": law.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        # Nothing here reads the bands, and three regimens over two horizons make ten
        # parameters -- a multiplier bootstrap over a matrix nobody looks at.
        "simultaneous": False,
    }
    columns = dict(COLUMNS)
    for key in ("outcome", "censoring"):
        if key in overrides:
            columns[key] = overrides.pop(key)  # type: ignore[assignment]
    settings.update(overrides)
    regimens = settings.pop("regimens", law.REGIMEN_SPEC)
    return LTMLE(regimens, **settings).fit(frame, **columns)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def fit() -> object:
    """One fit of the exact law, shared by every test in the module."""
    return _oracle_fit(law.frame())


def test_every_reported_parameter_has_an_oracle_and_no_more(fit: object) -> None:
    """The bidirectional gate, as the end-of-study law has for its own parameters.

    A survival fit reports a parameter per regimen *per horizon*, so the count moves with
    two things rather than one and the reverse direction earns its keep: adding a horizon
    to what ``_estimates`` reports now fails here until a longhand functional exists for
    it in ``tests/discrete_law_survival.py``.
    """
    assert set(fit) == set(law.NAMES)  # type: ignore[call-overload]
    # ``functional`` raises on a name it has no branch for, so this is not a tautology.
    assert set(law.TRUTH) == set(law.NAMES)
    for name in law.NAMES:
        assert law.functional(law.PROBS, name) == pytest.approx(law.TRUTH[name], abs=0)


@pytest.mark.parametrize("name", law.NAMES)
def test_point_estimate_is_the_g_formula(fit: object, name: str) -> None:
    """With exact nuisances the estimate is the truth, to the last bit."""
    assert fit.psi(name) == pytest.approx(law.TRUTH[name], abs=1e-12)  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", law.NAMES)
def test_influence_curve_is_the_gateaux_derivative(fit: object, name: str) -> None:
    reported = fit.influence_curves[name][law.first_row_of()]  # type: ignore[attr-defined]
    # ``rtol=0`` as in every sibling module: these curves reach order 20, so a default
    # relative tolerance would quietly loosen this to ~1e-6 -- six orders short of what
    # the comparison actually holds to, on the module's central claim.
    np.testing.assert_allclose(reported, law.eif(name), atol=1e-14, rtol=0)


def test_the_curve_is_a_function_of_the_support_point_alone(fit: object) -> None:
    """Two rows with the same observed history must carry the same influence curve."""
    curve = fit.influence_curves["risk_regimen[always @ t=2]"]  # type: ignore[attr-defined]
    for position, start in enumerate(law.first_row_of()):
        stop = start + law.COUNTS[position]
        np.testing.assert_allclose(curve[start:stop], curve[start], atol=1e-12, rtol=0)


def test_targeting_had_nothing_to_do(fit: object) -> None:
    """An exact initial fit already solves every score equation, so ``epsilon`` is zero.

    On a survival fit that is a statement about ``T(T+1)/2`` targeting steps rather than
    ``T``, and one of them -- the last node of the horizon-1 pass -- is a node no
    end-of-study fit ever targets as a *terminal* regression.
    """
    for regimen_fit in fit.fits.values():  # type: ignore[attr-defined]
        for step in regimen_fit.steps:
            assert step.fluctuation.converged
            assert abs(float(step.fluctuation.epsilon[0])) < 1e-8
            np.testing.assert_allclose(step.targeted, step.initial, atol=1e-9, rtol=0)


def test_the_contrast_curve_is_the_difference_of_the_two(fit: object) -> None:
    """Exactly, not approximately, and at the *same* horizon."""
    for horizon in law.HORIZONS:
        left = fit.influence_curves[f"risk_regimen[always @ t={horizon}]"]  # type: ignore[attr-defined]
        right = fit.influence_curves[f"risk_regimen[never @ t={horizon}]"]  # type: ignore[attr-defined]
        name = f"ate_regimen[always vs never @ t={horizon}]"
        difference = fit.influence_curves[name]  # type: ignore[attr-defined]
        np.testing.assert_allclose(difference, left - right, atol=1e-14, rtol=0)


def test_the_risk_curve_is_monotone(fit: object) -> None:
    """A cumulative risk cannot fall as the horizon lengthens.

    Nothing in the estimator imposes this -- each horizon is its own backward pass, with
    its own regressions and its own targeting -- so it is a real check on the recursion
    rather than a restatement of a constraint.  It is also the one property of the curve
    a reader would notice broken, which makes it worth an assertion of its own.
    """
    for label in law.REGIMEN_ARMS:
        risks = [fit.psi(f"risk_regimen[{label} @ t={h}]") for h in law.HORIZONS]  # type: ignore[attr-defined]
        assert risks == sorted(risks)


def test_a_rule_that_matches_a_constant_at_a_node_matches_it_at_that_horizon(
    fit: object,
) -> None:
    """``continue_if_l2`` treats at the first node, so at ``t = 1`` it *is* ``always``.

    Bit for bit, not approximately: the horizon-1 pass sees the same arms, so the same
    masks, the same design and the same fluctuation.  This is the survival analogue of a
    history-ignoring rule reproducing the constant plan it equals, and it fails if the
    horizon ever leaks into a mask or a design it has no business in.
    """
    rule = fit.fits["continue_if_l2 @ t=1"]  # type: ignore[attr-defined]
    constant = fit.fits["always @ t=1"]  # type: ignore[attr-defined]
    assert rule.psi_scaled == constant.psi_scaled
    np.testing.assert_array_equal(rule.influence_curve_scaled, constant.influence_curve_scaled)
    # And it must *not* match at the horizon where the rule has acted, or the parameter
    # would be one no rule-specific code path could move.
    assert (
        abs(fit.psi("risk_regimen[continue_if_l2 @ t=2]") - fit.psi("risk_regimen[always @ t=2]"))
        > 1e-3
    )  # type: ignore[attr-defined]


def test_the_rule_is_a_parameter_no_static_plan_reaches() -> None:
    """On the oracle, so the test is capable of failing.

    Checked against the *truth* rather than the estimates: an assertion that two
    estimates differ is satisfied by two wrong numbers.  This found a real defect -- with
    an earlier ``H2`` the rule's curve was exactly ``never``'s at both horizons, the two
    cells they differ in cancelling to the last bit, which would have left a parameter no
    bug could have moved.
    """
    rule = law.TRUTH["risk_regimen[continue_if_l2 @ t=2]"]
    for label in ("never", "always"):
        assert abs(rule - law.TRUTH[f"risk_regimen[{label} @ t=2]"]) > 1e-3


def test_no_truth_sits_on_the_filler() -> None:
    """No parameter may land on ``sequential._FILLER``.

    A half is what a prediction at a row nobody reads is filled with, so a truth of
    exactly ``0.5`` would let a fit that read those rows agree with the oracle for the
    wrong reason.  The end-of-study dataset records the same trap; with two horizons per
    regimen there are twice as many chances to fall into it.
    """
    for name, value in law.TRUTH.items():
        assert abs(value - 0.5) > 1e-3, name


class TestTheControlsBite:
    """Two ways of getting a survival fit wrong, each shown to move the answer.

    Both are mistakes that leave the fit convergent and every score at machine zero, so
    neither is caught by anything else in this module.  They are checked here at four
    orders of magnitude past the window the real assertions use.
    """

    def test_dropping_the_censoring_factor_would_be_wrong(self) -> None:
        """Treating the uncensored as the whole sample misses the truth.

        The same control the end-of-study law carries, and it has to be repeated here
        rather than inherited: censoring now interleaves with the event nodes, so the
        sweep that validates it is a different one.
        """
        frame = law.frame()
        # Drop the censored and *only* the censored: a unit that had the event at the
        # first node has no C2, and dropping it too would be a second mistake on top of
        # the one under test.
        kept = frame[(frame["C1"] == 1) & ((frame["Y1"] == 1) | (frame["C2"] == 1))].reset_index(
            drop=True
        )
        naive = _oracle_fit(kept, censoring=None, regimens={"always": 1}, reference="always")
        assert (
            abs(naive.psi("risk_regimen[always @ t=2]") - law.TRUTH["risk_regimen[always @ t=2]"])  # type: ignore[attr-defined]
            > 1e-2
        )

    def test_ignoring_the_first_event_node_would_be_wrong(self) -> None:
        """Analysing the survivors' ``Y2`` as an end-of-study outcome misses the truth.

        This is the mistake a survival outcome exists to prevent, written out: drop the
        units that had the event at the first node -- which is what any end-of-study
        analysis of this data must do, since they have no ``L2``, ``A2`` or ``C2`` -- and
        estimate ``E[Y2]`` on the rest.  It answers a question about the survivors, not
        the cumulative risk, and the gap says so.
        """
        frame = law.frame()
        survivors = frame[(frame["C1"] != 1) | (frame["Y1"] == 0)].reset_index(drop=True)
        naive = LTMLE(
            {"always": 1},
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=law.CellMeans(),
            censoring_learner=law.CellMeans(),
            n_folds=1,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
        ).fit(
            survivors,
            outcome="Y2",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        assert abs(naive.psi("ey_regimen[always]") - law.TRUTH["risk_regimen[always @ t=2]"]) > 1e-2
