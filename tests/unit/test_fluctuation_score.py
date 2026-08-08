r"""A fluctuation's recorded score is its score *at the state it returned*.

One invariant, three solvers, and it is the seam a good deal of this package rests on
without ever having pinned it.  Every consumer of a :class:`~cleverly.fluctuation.iterative
.Fluctuation` treats ``score`` as a statement about ``targeted``:
:func:`~cleverly.validation.score_check` reports the pair as a verdict on the fit it returns,
:func:`~cleverly.estimators.targeting.solve_with_reduction`'s exit test compares it against a
tolerance and stops, and
:func:`~cleverly.estimators.targeting._restated_outcome_score` re-derives it in place on the
strength of the two being the same expression.  None of them would notice if a solver
recorded a score from an iterate it then moved away from.

**The mutation this exists for is one line away, and it is not the obvious one.**  The
obvious candidate was measured and is **inert**:
:func:`~cleverly.fluctuation.iterative.solve_fluctuation` computes a score inside its loop
and recomputes it after, and the two agree because the in-loop evaluation is taken *after*
the step, at the iterate the loop then returns -- so recording the first in place of the
second changes nothing.  What is live is ``scoring_submodel = submodel``, kept deliberately
distinct from the ``fit_submodel`` the solve uses under ``target_weights=True``: scoring on
the weighted form instead is the plausible edit, it looks like removing a duplicate, and it
was run here and reddens exactly the two ``target_weights=True`` cases below.  It would
leave every *unweighted* fit in this package untouched, which is
the numerical-versus-reported-score distinction -- a change almost no assertion about a fitted result
can see.

**Why it is checked with exact equality.**  ``score_columns`` is a pure function of plain
arrays with no caching and no randomness, and the invariant is not that the two are close but
that they are *the same evaluation of one expression* --
:mod:`cleverly.validation.drtmle` makes the same distinction between an identity, whose right
value is zero, and a tolerance, which is a judgement about magnitude.  A tolerance here would
pass against a solver that recorded a score one Newton step early.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation._score import score_columns, score_scale
from cleverly.fluctuation.iterative import InitialFit, solve_fluctuation
from cleverly.fluctuation.one_step import solve_one_step
from cleverly.fluctuation.submodel import Submodel

#: Enough rows for a covariate with real spread and few enough to keep the module instant.
N = 200


def _inputs(seed: int = 0, *, missing: bool = False):
    """A two-arm ``mean`` submodel and an initial fit, on the ``[0, 1]`` scaled outcome.

    Built by hand rather than by fitting: the claim is about the *solver*, so a nuisance
    estimate would be a slower way of producing the same arrays and would tie this module to
    the estimator's own conventions.
    """
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, size=N).astype(float)
    propensity = np.clip(0.35 + 0.3 * rng.random(N), 0.05, 0.95)
    initial_mean = np.clip(0.3 + 0.4 * rng.random(N), 0.02, 0.98)
    outcome = np.clip(initial_mean + 0.2 * rng.standard_normal(N), 0.0, 1.0)

    covariate = np.column_stack(
        [(treatment == 1.0) / propensity, (treatment == 0.0) / (1.0 - propensity)]
    )
    submodel = Submodel(
        covariate,
        {
            1.0: np.column_stack([1.0 / propensity, np.zeros(N)]),
            0.0: np.column_stack([np.zeros(N), 1.0 / (1.0 - propensity)]),
        },
        ("h1", "h0"),
        "mean",
        {1.0: 0, 0.0: 1},
    )
    initial = InitialFit(observed=initial_mean, arms={1.0: initial_mean, 0.0: initial_mean})
    observed = np.ones(N, dtype=bool)
    if missing:
        observed[rng.random(N) < 0.2] = False
    # Non-unit weights, so that `target_weights=True` genuinely reweights the fit submodel.
    # With unit weights the weighted and unweighted forms coincide and that case is inert.
    return outcome, initial, submodel, 0.5 + rng.random(N), observed


@pytest.mark.parametrize("kind", ["logistic", "linear"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("target_weights", [False, True])
def test_the_recorded_score_is_the_score_at_the_returned_state(
    kind, missing, target_weights
) -> None:
    """Exactly, not approximately, and on every branch the solver can take.

    Two of the three parameters are not decoration.  ``missing=True`` matters because the
    score carries a ``Delta`` factor, so a solver that scored at the wrong mask would still
    agree with a recomputation on a fully observed fixture.  ``target_weights=True`` matters
    because it is the one setting under which the submodel the solve *fits* and the submodel
    the score is *taken on* are different objects -- ``solve_fluctuation`` keeps
    ``scoring_submodel = submodel`` deliberately, and scoring on the weighted fit submodel
    instead is the plausible edit this module exists to catch.  With unit weights the two
    coincide and the case is silent.
    """
    outcome, initial, submodel, weights, observed = _inputs(missing=missing)

    solved = solve_fluctuation(
        outcome,
        initial,
        submodel,
        weights,
        observed=observed,
        kind=kind,
        target_weights=target_weights,
    )
    recomputed = score_columns(
        outcome, solved.targeted.observed, submodel.observed, weights, observed
    )

    assert np.array_equal(solved.score, recomputed), "the score is not the returned state's"
    assert np.array_equal(solved.score_scale, score_scale(submodel.observed, weights, observed))


def test_it_holds_for_the_one_step_solver_too() -> None:
    """The third branch: the universal least-favourable walk, which records its score
    inside its own loop rather than after one, and so is the branch where this is least
    obviously true."""
    outcome, initial, submodel, weights, observed = _inputs(seed=3)

    solved = solve_one_step(outcome, initial, submodel, weights, observed=observed)
    recomputed = score_columns(
        outcome, solved.targeted.observed, submodel.observed, weights, observed
    )

    assert np.array_equal(solved.score, recomputed)


def test_the_invariant_is_not_vacuous() -> None:
    """The control: the fixture must be one where a *stale* score would be a different number.

    A fluctuation that moved nothing would satisfy the assertion above under any solver, so
    without this the module could pass on an inert fixture.  Scoring the **initial** state
    rather than the targeted one has to disagree, and by a wide margin.
    """
    outcome, initial, submodel, weights, observed = _inputs()

    solved = solve_fluctuation(outcome, initial, submodel, weights, observed=observed)
    stale = score_columns(outcome, initial.observed, submodel.observed, weights, observed)

    assert np.max(np.abs(stale)) > 1e-3
    assert np.max(np.abs(solved.score)) < 1e-8
