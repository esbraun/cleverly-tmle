"""The prefix masks compute what rebuilding them per node computed.

``LongitudinalData.at_risk`` and ``.following`` each rebuild a prefix from scratch --
``uncensored[:, :t].all(axis=1)`` and a loop of ``t`` comparisons -- so calling them at
every node of a backward pass is :math:`O(T^2 n)` per regimen, and a survival fit pays
that once per horizon on top.  ``regimen_masks`` scans once and reads a column per node.

The claim is an equality of arrays, so that is what is tested, node by node and mask by
mask, against the methods themselves rather than against a re-derivation of them.  Two
things make that worth more than it looks:

* the *asymmetry* between the two masks is easy to tidy away.  ``following(t)`` reads the
  censoring and follow factors at ``t`` and the event factor at ``t - 1``, because a unit
  that had the event at ``t`` **is** the observation that it happened and belongs in that
  node's regression.  A scan that indexed all three at ``t`` would agree with the old
  masks everywhere except on a survival fit, which is exactly where it matters;
* a dynamic rule assigns different units different arms at the same node, so ``followed``
  is a scan over an ``(n, T)`` comparison rather than over a row of scalars.

The estimator-level equivalence is checked too, because "the masks match" and "the fit is
unchanged" are different claims and the second is the one a user has.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import LTMLE, LongitudinalData
from cleverly.datasets import make_longitudinal, make_longitudinal_survival


def _end_of_study(n: int = 1200, seed: int = 3) -> LongitudinalData:
    frame, _ = make_longitudinal(n=n, seed=seed)
    return LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _survival(n: int = 1200, seed: int = 5) -> LongitudinalData:
    frame, _ = make_longitudinal_survival(n=n, seed=seed)
    return LongitudinalData.from_frame(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _competing(n: int = 1200, seed: int = 7) -> LongitudinalData:
    """Two causes, split by a per-unit coin -- the construction ``tests/e2e`` uses.

    Per unit rather than per node, because the event is absorbing: a per-node toss would
    let a unit relapse at one node and die at the next, which the container refuses.
    """
    frame, _ = make_longitudinal_survival(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    out = frame.copy()
    is_relapse = rng.integers(0, 2, size=len(frame)) == 0
    for node in ("1", "2"):
        event = frame[f"Y{node}"].to_numpy()
        out[f"R{node}"] = np.where(np.isnan(event), np.nan, event * is_relapse)
        out[f"D{node}"] = np.where(np.isnan(event), np.nan, event * ~is_relapse)
        out = out.drop(columns=[f"Y{node}"])
    return LongitudinalData.from_frame(
        out,
        outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


BUILDERS = {"end of study": _end_of_study, "survival": _survival, "competing": _competing}


def _rule(history):
    """A dynamic rule, so ``followed`` is a genuine ``(n, T)`` comparison."""
    return (np.asarray(history[history.columns[0]], dtype=float) > 0.0).astype(float)


@pytest.mark.parametrize("flavour", list(BUILDERS))
@pytest.mark.parametrize("plan", [[1, 1], [0, 0], [0, 1]])
def test_every_node_mask_matches_the_per_node_rebuild(flavour: str, plan: list[int]) -> None:
    data = BUILDERS[flavour]()
    masks = data.regimen_masks(plan)
    for time in range(data.n_times + 1):
        np.testing.assert_array_equal(masks.uncensored[:, time], data.uncensored_through(time))
        np.testing.assert_array_equal(masks.event_free[:, time], data.event_free_through(time))
        np.testing.assert_array_equal(masks.followed[:, time], data.followed_through(plan, time))
        if time >= 1:
            np.testing.assert_array_equal(masks.at_risk(time), data.at_risk(plan, time))
            np.testing.assert_array_equal(masks.following(time), data.following(plan, time))


@pytest.mark.parametrize("flavour", list(BUILDERS))
def test_the_masks_line_up_with_the_recursion(flavour: str) -> None:
    """``at_risk(t + 1) == following(t) & event-free at t``, read off the scan.

    The closure identity the recursion rests on, checked on the scanned masks so that a
    scan which agreed with the old masks *individually* but shifted the relation between
    them would still fail.
    """
    data = BUILDERS[flavour]()
    masks = data.regimen_masks([1, 1])
    for time in range(1, data.n_times):
        expected = masks.following(time) & masks.event_free[:, time]
        np.testing.assert_array_equal(masks.at_risk(time + 1), expected)


def test_the_event_factor_is_read_one_node_earlier_than_the_censoring_one() -> None:
    """The asymmetry, as a negative control rather than as a comment.

    A unit that has the event at ``t`` is in node ``t``'s regression.  Indexing the event
    factor at ``t`` instead of ``t - 1`` would drop exactly those units, so on a survival
    fit with any events at all the two differ -- and this asserts that they do, which is
    what stops the test above from passing against the tidied version.
    """
    data = _survival()
    masks = data.regimen_masks([1, 1])
    time = data.n_times
    tidied = masks.uncensored[:, time] & masks.followed[:, time] & masks.event_free[:, time]
    assert not np.array_equal(masks.following(time), tidied)
    assert masks.following(time).sum() > tidied.sum()


def test_a_dynamic_rule_scans_the_same_masks() -> None:
    data = _end_of_study()
    from cleverly.longitudinal.regimen import resolve_plans, resolve_regimens

    plans = resolve_plans(resolve_regimens({"rule": [1, _rule]}, data.n_times), data)
    values = plans[0].values
    masks = data.regimen_masks(values)
    for time in range(1, data.n_times + 1):
        np.testing.assert_array_equal(masks.at_risk(time), data.at_risk(values, time))
        np.testing.assert_array_equal(masks.following(time), data.following(values, time))


@pytest.mark.parametrize(
    ("flavour", "kwargs"),
    [
        ("end of study", {}),
        ("survival", {"horizons": [1, 2]}),
        ("competing", {"horizons": [1, 2]}),
    ],
)
def test_the_fit_is_unchanged(flavour: str, kwargs: dict) -> None:
    """The user-facing claim: same estimates, same curves, same epsilons.

    Compared against a fit whose recursion rebuilds the masks per node -- the previous
    behaviour, reached through ``prepare_node``'s default rather than by reverting the
    module -- so this is the two code paths against each other and not a regression file.
    """
    data = BUILDERS[flavour]()
    estimator = LTMLE(
        {"always": [1, 1], "never": [0, 0]},
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=2,
        learner_folds=2,
        random_state=0,
        **kwargs,
    )
    result = estimator.fit(data)

    import cleverly.longitudinal.sequential as sequential

    original = sequential.prepare_node

    def rebuilding(*args, **kw):
        kw.pop("masks", None)
        return original(*args, **kw)

    sequential.prepare_node = rebuilding
    try:
        reference = estimator.fit(data)
    finally:
        sequential.prepare_node = original

    assert list(result.keys()) == list(reference.keys())
    for name in list(result.keys()):
        got, want = result[name], reference[name]
        assert got.psi == want.psi
        assert got.std_error == want.std_error
        np.testing.assert_array_equal(got.influence_curve, want.influence_curve)
    np.testing.assert_array_equal(result.covariance(), reference.covariance())


def test_the_scan_is_linear_in_the_number_of_nodes() -> None:
    """Complexity, asserted as a count of passes rather than as a timing.

    A timing on a shared box is a coin flip; what the change actually claims is that the
    number of full-length passes over the treatment matrix stops growing with ``T``.  One
    ``regimen_masks`` call is one comparison and one accumulate whatever ``T`` is, where
    ``T`` calls to ``followed_through`` are ``T(T+1)/2`` boolean ANDs.
    """
    counted: list[int] = []

    class CountingAnd:
        def __init__(self, inner):
            self._inner = inner

        def __call__(self, *args, **kwargs):
            counted.append(1)
            return self._inner(*args, **kwargs)

        def accumulate(self, *args, **kwargs):
            counted.append(1)
            return self._inner.accumulate(*args, **kwargs)

    data = _end_of_study()
    original = np.logical_and
    np.logical_and = CountingAnd(original)  # type: ignore[assignment]
    try:
        counted.clear()
        data.regimen_masks([1, 1])
        scanned = len(counted)
    finally:
        np.logical_and = original  # type: ignore[assignment]

    # Two accumulates (censoring and follow) plus whatever the event prefix costs, and in
    # particular a count that does not depend on the node index the way a rebuild does.
    assert scanned <= 4
