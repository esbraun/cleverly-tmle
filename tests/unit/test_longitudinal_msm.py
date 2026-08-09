"""What a working model over regimens accepts, and what it refuses rather than guesses at.

The call-site assertion here is structural for the reason
``test_sequential_design.py``'s is: a design handed the wrong frame reads the wrong
covariate and returns a perfectly valid-looking design matrix, so no downstream number
can tell you about it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import DataError
from cleverly.longitudinal import LongitudinalData, resolve_plans, resolve_regimens
from cleverly.longitudinal.msm import Cell, RegimenMSM, evaluate_regimen_msm
from cleverly.msm import MSM

#: Two regimens, and a "duration" for each -- the summary a working model over regimens
#: is a dose-response in. Declared as a table rather than parsed out of the label, which
#: is the whole content of the ``MSM.linear`` refusal.
DURATION = {"always": 2.0, "never": 0.0, "early": 1.0}


def panel(n: int = 60, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.integers(0, 2, n).astype(float)
    l1 = rng.standard_normal(n)
    a1 = rng.integers(0, 2, n).astype(float)
    c1 = (rng.random(n) < 0.9).astype(float)
    l2 = np.where(c1 == 1, rng.standard_normal(n), np.nan)
    a2 = np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    observed = (c1 == 1) & (c2 == 1)
    y = np.where(observed, rng.integers(0, 2, n).astype(float), np.nan)
    return pd.DataFrame(
        {"W1": w1, "W2": w2, "L1": l1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y}
    )


def build(frame: pd.DataFrame) -> LongitudinalData:
    # ``L1`` is time-varying at the *first* node, and it is here on purpose. With an empty
    # ``time_varying[0]`` -- which is what every other longitudinal fixture in this
    # repository has -- ``history_frame(1)`` and ``baseline_frame()`` are the same frame,
    # so the call-site pin below would pass under the very substitution it exists to
    # catch. Mutation-tested: swapping the two turns it red only because of this column.
    return LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[["L1"], ["L2"]],
        censoring=["C1", "C2"],
    )


def plans_for(data: LongitudinalData, spec: dict[str, Any] | None = None) -> Any:
    spec = {"always": 1, "never": 0} if spec is None else spec
    return resolve_plans(resolve_regimens(spec, data.n_times), data)


def dose_design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    del horizon
    n = len(frame)
    return np.column_stack([np.ones(n), np.full(n, DURATION[label])])


DOSE = MSM(design=dose_design, terms=("(intercept)", "duration"))


class TestItEvaluatesTheDeclaredModel:
    def test_the_cells_are_the_regimens_at_one_horizon(self) -> None:
        data = build(panel())
        plans = plans_for(data)
        model = evaluate_regimen_msm(DOSE, data, plans, (2,))
        assert model.cells == (Cell("always", 2), Cell("never", 2))
        assert model.labels == ("always", "never")
        assert model.design.shape == (data.n, 2, 2)
        assert model.weights.shape == (data.n, 2)
        np.testing.assert_array_equal(model.weights, np.ones((data.n, 2)))
        np.testing.assert_array_equal(model.design[:, 0, 1], np.full(data.n, 2.0))
        np.testing.assert_array_equal(model.design[:, 1, 1], np.zeros(data.n))

    def test_the_cells_cross_the_regimens_with_the_horizons(self) -> None:
        """A survival grid is (regimen, horizon), so a coefficient can be a trend in t."""
        data = build(panel())
        plans = plans_for(data)

        def with_time(label: Any, horizon: int, frame: Any) -> np.ndarray:
            n = len(frame)
            return np.column_stack(
                [np.ones(n), np.full(n, DURATION[label]), np.full(n, float(horizon))]
            )

        msm = MSM(design=with_time, terms=("(intercept)", "duration", "t"))
        model = evaluate_regimen_msm(msm, data, plans, (1, 2))
        assert model.cells == (
            Cell("always", 1),
            Cell("never", 1),
            Cell("always", 2),
            Cell("never", 2),
        )
        np.testing.assert_array_equal(model.design[0, :, 2], np.array([1.0, 1.0, 2.0, 2.0]))

    def test_a_declared_weight_reaches_the_evaluated_model(self) -> None:
        data = build(panel())
        plans = plans_for(data)
        msm = MSM(
            design=dose_design,
            terms=("(intercept)", "duration"),
            weights=lambda label, horizon, w: 1.0 + DURATION[label] + np.asarray(w["W2"]),
        )
        model = evaluate_regimen_msm(msm, data, plans, (2,))
        w2 = data.baseline[:, 1]
        np.testing.assert_allclose(model.weights[:, 0], 3.0 + w2)
        np.testing.assert_allclose(model.weights[:, 1], 1.0 + w2)


class TestTheDesignSeesTheBaselineAndNothingElse:
    def test_the_frame_handed_to_a_design_is_the_baseline(self) -> None:
        """Structural, and load-bearing: V has to be pre-treatment.

        A design reading a time-varying covariate would not be a model for
        ``E[Y^a-bar | V]`` -- it would be conditioning on a consequence of an earlier
        node's arm, which is a different parameter. Pinned at the call site because a
        design handed the richer frame returns a design matrix that looks perfectly
        ordinary, and no number downstream can tell you.

        ``L1`` is measured *before* the first node, so ``history_frame(1)`` carries it and
        ``baseline_frame()`` does not. That is what gives this assertion teeth.
        """
        data = build(panel())
        seen: list[tuple[Any, int, tuple[str, ...]]] = []

        def spy(label: Any, horizon: int, frame: Any) -> np.ndarray:
            seen.append((label, horizon, tuple(frame.columns)))
            return np.column_stack([np.ones(len(frame)), np.full(len(frame), DURATION[label])])

        evaluate_regimen_msm(
            MSM(design=spy, terms=("(intercept)", "duration")), data, plans_for(data), (2,)
        )
        assert [entry[0] for entry in seen] == ["always", "never"]
        assert {entry[1] for entry in seen} == {2}
        for _, _, columns in seen:
            assert columns == data.baseline_names == ("W1", "W2")
            assert "L1" not in columns and "L2" not in columns
            assert "A1" not in columns and "Y" not in columns
        # And the substitution this is here to catch is a visible one on this fixture.
        assert "L1" in data.history_names(1)


class TestItRefusesByName:
    def test_the_arm_shorthand_is_refused(self) -> None:
        """``MSM.linear`` reads its label as a dose, and a regimen has no dose."""
        data = build(panel())
        with pytest.raises(DataError, match="reads the label it is handed as a dose"):
            evaluate_regimen_msm(MSM.linear(), data, plans_for(data), (2,))

    def test_a_numerically_named_regimen_does_not_slip_past_it(self) -> None:
        """The refusal is a flag on the declaration, not a failure to parse the label.

        ``_numeric_level`` refuses a string it cannot read as a number, so most misuse is
        loud on its own. A regimen legitimately called ``"0"`` is the case that is not:
        it would be read as a dose of zero and reported without complaint.
        """
        data = build(panel())
        plans = plans_for(data, {"0": 0, "1": 1})
        with pytest.raises(DataError, match="a plan is a sequence of"):
            evaluate_regimen_msm(MSM.linear(), data, plans, (2,))

    def test_a_two_argument_design_says_what_the_signature_is(self) -> None:
        data = build(panel())
        msm = MSM(design=lambda label, w: np.ones((len(w), 1)), terms=("(intercept)",))
        with pytest.raises(DataError, match="takes three arguments"):
            evaluate_regimen_msm(msm, data, plans_for(data), (2,))

    def test_a_wrong_shape_names_the_regimen_it_came_from(self) -> None:
        data = build(panel())

        def short(label: Any, horizon: int, frame: Any) -> np.ndarray:
            rows = len(frame) if label == "always" else len(frame) - 1
            return np.column_stack([np.ones(rows), np.zeros(rows)])

        msm = MSM(design=short, terms=("(intercept)", "duration"))
        with pytest.raises(DataError, match="for regimen 'never' at horizon 2"):
            evaluate_regimen_msm(msm, data, plans_for(data), (2,))

    def test_a_term_count_mismatch_is_refused(self) -> None:
        data = build(panel())
        msm = MSM(design=dose_design, terms=("(intercept)", "duration", "spare"))
        with pytest.raises(DataError, match="but names 3 term"):
            evaluate_regimen_msm(msm, data, plans_for(data), (2,))

    def test_a_collinear_design_is_refused_where_it_is_built(self) -> None:
        """Left to the solve, ``lstsq`` would return a minimum-norm answer to a question
        that has none."""
        data = build(panel())

        def twice(label: Any, horizon: int, frame: Any) -> np.ndarray:
            del horizon
            dose = np.full(len(frame), DURATION[label])
            return np.column_stack([dose, 2.0 * dose])

        msm = MSM(design=twice, terms=("duration", "twice"))
        with pytest.raises(DataError, match="collinear across the regimens"):
            evaluate_regimen_msm(msm, data, plans_for(data), (2,))

    def test_a_negative_weight_is_refused(self) -> None:
        data = build(panel())
        msm = MSM(
            design=dose_design,
            terms=("(intercept)", "duration"),
            weights=lambda label, horizon, w: -np.ones(len(w)),
        )
        with pytest.raises(DataError, match="not a signed contrast"):
            evaluate_regimen_msm(msm, data, plans_for(data), (2,))

    def test_a_logit_link_needs_an_outcome_in_the_unit_interval(self) -> None:
        frame = panel()
        frame["Y"] = frame["Y"] * 4.0
        data = build(frame)
        msm = MSM(design=dose_design, terms=("(intercept)", "duration"), link="logit")
        with pytest.raises(DataError, match="needs an outcome in"):
            evaluate_regimen_msm(msm, data, plans_for(data), (2,))


class TestTheCovariateNumerator:
    def test_the_identity_link_ignores_beta(self) -> None:
        """Which is what keeps an identity-link fit one pass rather than an alternation."""
        data = build(panel())
        model = evaluate_regimen_msm(DOSE, data, plans_for(data), (2,))
        expected = model.design * model.weights[:, :, None]
        np.testing.assert_array_equal(model.weighted_design_at(None), expected)
        np.testing.assert_array_equal(model.weighted_design_at(np.array([3.0, -1.0])), expected)
        np.testing.assert_array_equal(model.fluctuation_design_at(None), model.design)
        np.testing.assert_array_equal(
            model.fluctuation_design_at(np.array([3.0, -1.0])), model.design
        )

    def test_a_link_scales_each_cell_by_dm_deta(self) -> None:
        data = build(panel())
        msm = RegimenMSM(
            DOSE.terms,
            evaluate_regimen_msm(DOSE, data, plans_for(data), (2,)).design,
            np.ones((data.n, 2)),
            (Cell("always", 2), Cell("never", 2)),
            "logit",
        )
        beta = np.array([0.25, -0.5])
        m = msm.fitted(beta)
        np.testing.assert_allclose(
            msm.weighted_design_at(beta), msm.design * (m * (1 - m))[..., None]
        )
        np.testing.assert_allclose(
            msm.fluctuation_design_at(beta), msm.design * (m * (1 - m))[..., None]
        )

    def test_a_link_needs_a_beta_to_build_the_covariate_at(self) -> None:
        data = build(panel())
        model = evaluate_regimen_msm(
            MSM(design=dose_design, terms=("(intercept)", "duration"), link="logit"),
            data,
            plans_for(data),
            (2,),
        )
        with pytest.raises(ValueError, match="cannot be built before one is available"):
            model.weighted_design_at(None)
        with pytest.raises(ValueError, match="cannot be built before one is available"):
            model.fluctuation_design_at(None)
