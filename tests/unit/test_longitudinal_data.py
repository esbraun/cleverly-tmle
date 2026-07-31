"""What the longitudinal container accepts, and what it refuses rather than guesses at."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import DataError
from cleverly.longitudinal import (
    DynamicRegimen,
    LongitudinalData,
    Regimen,
    resolve_regimens,
)


def panel(n: int = 40, *, seed: int = 0) -> pd.DataFrame:
    """A small, valid two-time-point panel with monotone censoring."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    a1 = rng.integers(0, 2, n).astype(float)
    c1 = (rng.random(n) < 0.8).astype(float)
    l2 = np.where(c1 == 1, rng.standard_normal(n), np.nan)
    a2 = np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    observed = (c1 == 1) & (c2 == 1)
    y = np.where(observed, rng.integers(0, 2, n).astype(float), np.nan)
    return pd.DataFrame({"W1": w1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})


#: The column declaration :func:`panel` answers to, shared by the container tests and by
#: the fits that go through ``LTMLE`` rather than restated at each of them.
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def build(frame: pd.DataFrame, **overrides: Any) -> LongitudinalData:
    return LongitudinalData.from_frame(frame, **{**COLUMNS, **overrides})


def test_reads_the_panel() -> None:
    data = build(panel())
    assert data.n_times == 2
    assert data.baseline_names == ("W1",)
    assert data.time_varying_names == ((), ("L2",))
    assert data.family == "binomial"
    assert data.has_censoring


def test_masks_line_up_with_the_recursion() -> None:
    """``at_risk(t)`` at one node is the set the *previous* node is fitted on.

    That identity is what makes the backward recursion close, so it is checked here
    rather than left as a comment in the estimator.
    """
    data = build(panel())
    plan = (1.0, 1.0)
    for time in range(1, data.n_times):
        np.testing.assert_array_equal(data.at_risk(plan, time + 1), data.following(plan, time))
    np.testing.assert_array_equal(data.at_risk(plan, 1), np.ones(data.n, dtype=bool))


def test_history_grows_one_block_at_a_time() -> None:
    data = build(panel())
    assert data.covariate_history(1).shape == (data.n, 1)
    assert data.covariate_history(2).shape == (data.n, 2)
    # The mechanism at the second node conditions on the first treatment; the censoring
    # model at the same node additionally conditions on the second.
    assert data.history_design(2).shape == (data.n, 3)
    assert data.history_design(2, include_current=True).shape == (data.n, 4)
    at_regimen = data.history_design(2, treatment=(1.0, 0.0), include_current=True)
    np.testing.assert_array_equal(at_regimen[:, -2], np.ones(data.n))
    np.testing.assert_array_equal(at_regimen[:, -1], np.zeros(data.n))


class TestWhatARuleIsHandedAndWhatItMayReturn:
    """The contract between a dynamic rule and the container, checked from both ends."""

    def test_the_history_names_are_the_history_columns(self) -> None:
        """Otherwise the frame a rule reads is silently mislabelled.

        The correspondence is an accident of two independent loops -- the matrix skips a
        zero-width block and a zero-width block's names are the empty tuple -- so it holds
        until someone reorders one of them. A rule handed a mislabelled frame reads the
        wrong covariate and returns a perfectly valid-looking arm, which is exactly the
        kind of wrongness no downstream assertion can see.
        """
        data = build(panel())
        for time in range(1, data.n_times + 1):
            assert len(data.history_names(time)) == data.covariate_history(time).shape[1]
        assert data.history_names(1) == ("W1",)
        assert data.history_names(2) == ("W1", "L2")

    def test_a_rule_sees_the_history_and_nothing_else(self) -> None:
        """No outcome and no observed treatment, enforced by what the frame contains.

        Reading ``Y`` is not an intervention, and reading ``A1`` is a different object:
        under the regimen the earlier treatment is what the rule itself assigned, so a
        rule reading the *observed* one would be reading the treatment of a unit that
        deviated. Both are kept out by omission rather than by a check, which is what
        this pins.
        """
        from cleverly.longitudinal import LTMLE

        seen: list[list[str]] = []

        def watcher(history: Any) -> Any:
            seen.append(list(history.columns))
            return np.zeros(len(history))

        LTMLE(
            {"watched": (watcher, watcher)},
            outcome_learner="glm",
            pseudo_learner="glm",
            treatment_learner="glm",
            n_folds=2,
            learner_folds=2,
            random_state=0,
        ).fit(panel(n=200, seed=1), **COLUMNS)
        assert seen == [["W1"], ["W1", "L2"]]

    def test_refuses_a_rule_that_returns_the_wrong_number_of_rows(self) -> None:
        data = build(panel())
        regimen = DynamicRegimen("short", (lambda history: np.zeros(3), 0.0))
        with pytest.raises(DataError, match="one arm per row"):
            regimen.assignment(data)

    def test_refuses_a_rule_that_returns_something_other_than_an_arm(self) -> None:
        data = build(panel())
        regimen = DynamicRegimen("dose", (lambda history: np.full(data.n, 0.5), 0.0))
        with pytest.raises(DataError, match="must return 0 or 1"):
            regimen.assignment(data)

    def test_a_rule_that_raises_says_which_columns_it_had(self) -> None:
        """The commonest trap: one rule broadcast to every node, reading a late covariate.

        ``lambda h: h["L2"]`` is a perfectly good rule at the second node and a
        ``KeyError`` at the first, so the message has to name the node and the columns
        rather than let a bare ``KeyError`` surface.
        """
        data = build(panel())
        regimen = DynamicRegimen("too early", (lambda history: history["L2"],) * 2)
        with pytest.raises(DataError, match=r"time 1 .*raised KeyError"):
            regimen.assignment(data)
        with pytest.raises(DataError, match=r"\['W1'\]"):
            regimen.assignment(data)

    def test_a_rule_is_not_asked_to_answer_for_a_censored_unit(self) -> None:
        """Off the recorded set the history is the zero fill, so the arm is coerced.

        Not validated, coerced: such a row is masked out of every regression and every
        influence curve, so the only way its arm could matter is by putting a ``nan`` into
        a design matrix a learner is then called on. A rule that divides by a filled zero
        is the realistic way that happens.  ``L2 / L2`` is exactly that rule and nothing
        else: a valid arm of ``1`` wherever ``L2`` was recorded, and ``0 / 0`` -- ``nan``
        -- on precisely the rows the fill created.
        """
        data = build(panel())
        regimen = DynamicRegimen("reciprocal", (0.0, lambda history: history["L2"] / history["L2"]))
        assigned = regimen.assignment(data)
        censored = ~data.uncensored_through(1)
        assert censored.any()
        assert np.all(np.isfinite(assigned))
        np.testing.assert_array_equal(assigned[censored, 1], 0.0)


def test_the_fill_for_censored_rows_carries_no_information() -> None:
    """The zero-fill is confined to rows nothing reads, and cannot encode anything.

    The container zero-fills the covariates of a unit censored before the node, so that
    a learner can be *called* on the whole matrix in one pass.  Two things together make
    that safe, and both are checked: the fill lands only where the unit is not at risk,
    and the container refuses a recorded value there in the first place
    (``test_refuses_a_node_recorded_after_censoring``), so there is no number for the
    fill to have replaced.
    """
    data = build(panel())
    reachable = data.uncensored_through(1)
    design = data.covariate_history(2)
    assert np.all(np.isfinite(design))
    np.testing.assert_array_equal(design[~reachable, 1], 0.0)
    np.testing.assert_array_equal(design[reachable, 1], data.time_varying[1][reachable, 0])


def test_the_fill_cannot_reach_the_estimate() -> None:
    """The claim the module docstring makes: replace the filled entries, get the same fit.

    ``data.py`` said this test existed; it did not. What was checked was *where* the fill
    lands, which is a statement about the container. This is the statement about the
    estimator, and it is the one that would fail if a mask slipped: the filled rows flow
    into ``covariate_history`` and hence into every mechanism model's training matrix, and
    only ``fit_mask=at_risk`` keeps them inert. Fill with something wild rather than
    something plausible, so a leak is a visible move rather than a rounding difference.

    A **dynamic rule** is in the regimen dict for the same reason, and it makes the test
    sharper than it was: the rule reads ``L2``, so replacing the filled entries with
    ``1e6`` flips the arm it assigns on every one of those rows. If a filled row could
    reach the estimate, this is not a rounding difference but the opposite arm.
    """
    from cleverly.longitudinal import LTMLE

    regimens: dict[str, Any] = {
        "always": 1,
        "never": 0,
        "treat if l2 rises": (0, lambda history: (history["L2"] > 0.0).astype(float)),
    }
    frame = panel(n=400, seed=3)
    settings: dict[str, Any] = {
        "outcome_learner": "glm",
        "pseudo_learner": "glm",
        "treatment_learner": "glm",
        "n_folds": 2,
        "learner_folds": 2,
        "random_state": 0,
    }
    plain = LTMLE(regimens, **settings).fit(frame, **COLUMNS)

    # Every node after a unit's censoring time is nan and stays nan -- the container
    # refuses a recorded value there.  What the fill replaces those nans with is what is
    # under test, so it is reached through the built container rather than the frame.
    data = LongitudinalData.from_frame(frame, **COLUMNS)
    perturbed = [block.copy() for block in data.time_varying]
    censored = ~data.uncensored_through(1)
    assert censored.any()
    perturbed[1][censored, 0] = 1e6
    moved = LTMLE(regimens, **settings).fit(
        dataclasses.replace(data, time_varying=tuple(perturbed))
    )
    for name in plain:
        assert plain.psi(name) == moved.psi(name)
        np.testing.assert_array_equal(plain.influence_curves[name], moved.influence_curves[name])


def test_refuses_a_unit_that_returns_after_censoring() -> None:
    frame = panel()
    frame.loc[frame.index[0], ["C1", "C2"]] = [0.0, 1.0]
    with pytest.raises(DataError, match="monotone"):
        build(frame)


def test_refuses_a_node_recorded_after_censoring() -> None:
    frame = panel()
    censored = frame.index[frame["C1"] == 0][0]
    frame.loc[censored, "L2"] = 0.3
    with pytest.raises(DataError, match="had already been censored"):
        build(frame)


def test_refuses_a_node_missing_before_censoring() -> None:
    frame = panel()
    present = frame.index[frame["C1"] == 1][0]
    frame.loc[present, "A2"] = np.nan
    with pytest.raises(DataError, match="still under observation"):
        build(frame)


def test_refuses_an_outcome_missing_for_any_other_reason() -> None:
    """A missing outcome on an uncensored unit is a node, not an absence."""
    frame = panel()
    complete = frame.index[(frame["C1"] == 1) & (frame["C2"] == 1)][0]
    frame.loc[complete, "Y"] = np.nan
    with pytest.raises(DataError, match="never censored but have no"):
        build(frame)


def test_refuses_a_non_binary_treatment() -> None:
    frame = panel()
    frame.loc[frame.index[0], "A1"] = 2.0
    with pytest.raises(DataError, match="binary treatment at every node"):
        build(frame)


def test_refuses_a_column_used_twice() -> None:
    with pytest.raises(DataError, match="more than one node"):
        build(panel(), baseline=["W1", "A1"])


def test_refuses_a_censoring_column_per_node_mismatch() -> None:
    with pytest.raises(DataError, match="one censoring column per time point"):
        build(panel(), censoring=["C1"])


def test_works_without_censoring_columns() -> None:
    frame = panel()
    frame = frame[frame["C1"] * frame["C2"] == 1].reset_index(drop=True)
    data = build(frame, censoring=None)
    assert not data.has_censoring
    assert data.uncensored.all()


class TestRegimens:
    def test_scalar_broadcasts_across_the_nodes(self) -> None:
        (always,) = resolve_regimens({"always": 1}, 3)
        assert always.values == (1.0, 1.0, 1.0)

    def test_keeps_declaration_order(self) -> None:
        regimens = resolve_regimens({"b": 0, "a": 1}, 2)
        assert [regimen.label for regimen in regimens] == ["b", "a"]

    def test_refuses_a_plan_of_the_wrong_length(self) -> None:
        with pytest.raises(DataError, match="assigns 3 arm"):
            resolve_regimens({"odd": (1, 0, 1)}, 2)

    def test_refuses_a_non_binary_arm(self) -> None:
        with pytest.raises(DataError, match="must be 0 or 1"):
            Regimen("dose", (0.5, 1.0))

    def test_a_rule_that_reads_the_history_is_a_dynamic_regimen(self) -> None:
        """This used to be a refusal, and the shape of the replacement is the point.

        A callable plan is broadcast across the nodes exactly as a scalar arm is, and
        comes back a ``DynamicRegimen`` rather than a ``Regimen`` -- so which class
        ``resolve_regimens`` returns *is* the statement about whether the followers are a
        fixed slice or a set this sample determines.
        """
        resolved = resolve_regimens({"dynamic": lambda history: history["W"]}, 2)
        assert isinstance(resolved[0], DynamicRegimen)
        assert len(resolved[0].plan) == 2
        assert all(callable(node) for node in resolved[0].plan)

    def test_a_plan_with_no_rule_in_it_stays_static(self) -> None:
        """The load-bearing half of the previous test.

        A static plan must come back a ``Regimen``, because that is what keeps it on the
        broadcast path whose arithmetic is bit-for-bit what it was before rules existed.
        """
        assert isinstance(resolve_regimens({"early": (1, 0)}, 2)[0], Regimen)

    def test_a_plan_may_mix_a_constant_node_with_a_rule(self) -> None:
        resolved = resolve_regimens({"then decide": (1, lambda history: history["L2"])}, 2)
        assert isinstance(resolved[0], DynamicRegimen)
        assert resolved[0].plan[0] == 1.0
        assert resolved[0].is_rule(2) and not resolved[0].is_rule(1)

    def test_an_array_plan_is_read_as_a_plan(self) -> None:
        """A numpy plan is a plan, and must not be diagnosed as a dynamic rule.

        ``np.ndarray`` does not register as a ``collections.abc.Sequence``, so a type
        test on that alone sends an ordinary plan to the message about rules -- the right
        refusal for the wrong input, which is worse than no message at all.
        """
        resolved = resolve_regimens({"early": np.array([1.0, 0.0])}, 2)
        assert resolved[0].values == (1.0, 0.0)

    def test_refuses_a_plan_that_is_neither_arm_nor_sequence(self) -> None:
        with pytest.raises(DataError, match="must be an arm"):
            resolve_regimens({"odd": object()}, 2)

    def test_refuses_no_regimens_at_all(self) -> None:
        with pytest.raises(DataError, match="needs regimens="):
            resolve_regimens(None, 2)

    def test_refuses_an_empty_mapping(self) -> None:
        with pytest.raises(DataError, match="no regimen reports no parameter"):
            resolve_regimens({}, 2)

    def test_refuses_a_duplicate_label(self) -> None:
        with pytest.raises(DataError, match="appears twice"):
            resolve_regimens([Regimen("a", (1.0, 1.0)), Regimen("a", (0.0, 0.0))], 2)

    def test_refuses_a_sequence_of_non_regimens(self) -> None:
        with pytest.raises(DataError, match="must hold Regimen or DynamicRegimen objects"):
            resolve_regimens([(1.0, 1.0)], 2)

    def test_refuses_a_spec_that_is_neither_mapping_nor_sequence(self) -> None:
        with pytest.raises(DataError, match="must be a mapping or a sequence"):
            resolve_regimens(1.0, 2)

    def test_refuses_a_plan_with_no_nodes(self) -> None:
        with pytest.raises(DataError, match="assigns no treatment at any time point"):
            Regimen("empty", ())

    def test_a_single_regimen_object_is_accepted(self) -> None:
        assert resolve_regimens(Regimen("a", (1.0, 0.0)), 2)[0].label == "a"


class TestTheContainerRefusesByName:
    """The branches of ``LongitudinalData`` with no test, one per message."""

    def test_refuses_something_that_is_not_a_dataframe(self) -> None:
        with pytest.raises(DataError, match="pandas or polars DataFrame"):
            build(np.zeros((10, 3)))  # type: ignore[arg-type]

    def test_refuses_an_empty_treatment_list(self) -> None:
        with pytest.raises(DataError, match="at least one node"):
            build(panel(), treatment=[], time_varying=[], censoring=[])

    def test_names_the_columns_it_could_not_find(self) -> None:
        with pytest.raises(DataError, match="columns not found"):
            build(panel(), baseline=["nope"])

    def test_refuses_too_few_observations(self) -> None:
        with pytest.raises(DataError, match="at least 10 observations"):
            build(panel(n=9))

    def test_refuses_an_empty_baseline(self) -> None:
        with pytest.raises(DataError, match="baseline= is empty"):
            build(panel(), baseline=[])

    def test_refuses_a_time_varying_block_per_node_mismatch(self) -> None:
        with pytest.raises(DataError, match="one \\(possibly empty\\) list per time point"):
            build(panel(), time_varying=[["L2"]])

    def test_refuses_an_unknown_family(self) -> None:
        with pytest.raises(DataError, match="family must be"):
            build(panel(), family="poisson")

    def test_refuses_a_non_binary_binomial_outcome(self) -> None:
        frame = panel()
        complete = frame.index[frame["Y"].notna()][0]
        frame.loc[complete, "Y"] = 4.0
        with pytest.raises(DataError, match="requires a 0/1 outcome"):
            build(frame, family="binomial")

    def test_refuses_a_non_binary_censoring_column(self) -> None:
        frame = panel()
        frame.loc[frame.index[0], "C1"] = 2.0
        with pytest.raises(DataError, match="still under observation after that time"):
            build(frame)

    def test_refuses_a_censoring_value_missing_while_under_observation(self) -> None:
        frame = panel()
        frame.loc[frame.index[0], "C1"] = np.nan
        with pytest.raises(DataError, match="were still under observation"):
            build(frame)

    def test_refuses_a_history_outside_the_node_range(self) -> None:
        data = build(panel())
        with pytest.raises(DataError, match="outside 1\\.\\.2"):
            data.covariate_history(3)

    def test_drops_a_constant_time_varying_covariate(self) -> None:
        """Screened on the same terms as a baseline covariate, and for a sharper reason.

        ``covariate_history`` stacks every block into one design, so a constant ``L_t``
        makes the history matrix singular at that node and at every node after it.  The
        screen runs on the rows still under observation: a censored unit's block is
        ``nan``, which is not missing data to be refused but a node that does not exist.
        """
        frame = panel()
        frame["const"] = np.where(frame["C1"] == 1, 1.0, np.nan)
        data = build(frame, time_varying=[[], ["L2", "const"]])
        assert data.time_varying_names == ((), ("L2",))
        assert "const" in data.dropped_covariates
