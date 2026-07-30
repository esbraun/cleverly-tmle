"""The working model, before it reaches a fluctuation.

A marginal structural model is a *declaration* -- a design, its term names and a known
weight function -- so everything here is exact: the evaluated arrays, the projection's
closed form, and the refusals.  Nothing is inferred from an estimate that used them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import DataError
from cleverly.msm import MSM, MSMSet, refuse_unsupported


def make_data(n: int = 40, *, levels: tuple = (0, 1), seed: int = 0) -> CausalData:
    rng = np.random.default_rng(seed)
    k = len(levels)
    frame = {
        "Y": rng.binomial(1, 0.4, n).astype(float),
        "A": np.asarray([levels[i % k] for i in range(n)]),
        "W1": rng.normal(size=n),
        "W2": rng.normal(size=n),
    }
    return CausalData.from_frame(
        pd.DataFrame(frame), outcome="Y", treatment="A", covariates=["W1", "W2"]
    )


def constant_design(terms: tuple[str, ...] = ("(intercept)", "a")) -> MSM:
    return MSM(
        design=lambda a, w: np.column_stack([np.ones(len(w)), np.full(len(w), float(a))]),
        terms=terms,
    )


# ------------------------------------------------------------------ declaration


class TestDeclaration:
    def test_terms_must_be_distinct_and_non_empty(self) -> None:
        with pytest.raises(DataError, match="distinct"):
            MSM(design=lambda a, w: np.ones((len(w), 2)), terms=("a", "a"))
        with pytest.raises(DataError, match="at least one term"):
            MSM(design=lambda a, w: np.ones((len(w), 0)), terms=())

    def test_a_non_identity_link_is_refused_with_the_reason(self) -> None:
        with pytest.raises(NotImplementedError, match="outer \\(beta, epsilon\\) iteration"):
            MSM(design=lambda a, w: np.ones((len(w), 1)), terms=("(intercept)",), link="logit")  # type: ignore[arg-type]

    def test_weights_that_are_not_a_known_function_are_refused(self) -> None:
        # An array is not a function of ``(a, V)``: it is one particular evaluation, and
        # nothing here can check it was not derived from the fitted mechanism.
        with pytest.raises(NotImplementedError, match="functional of P"):
            MSM(
                design=lambda a, w: np.ones((len(w), 1)),
                terms=("(intercept)",),
                weights=np.ones(10),  # type: ignore[arg-type]
            )

    def test_refuse_unsupported_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown refusal"):
            refuse_unsupported("nonsense")


class TestLinearShorthand:
    def test_terms_follow_the_modifiers(self) -> None:
        assert MSM.linear().terms == ("(intercept)", "a")
        assert MSM.linear(modifiers=("W1",), interaction=False).terms == (
            "(intercept)",
            "a",
            "W1",
        )
        assert MSM.linear(modifiers=("W1", "W2")).terms == (
            "(intercept)",
            "a",
            "W1",
            "W2",
            "a:W1",
            "a:W2",
        )

    def test_the_design_is_the_model_written_out(self) -> None:
        data = make_data(levels=(0, 2))
        evaluated = MSMSet.evaluate(MSM.linear(modifiers=("W1",)), data)
        w1 = data.covariates[:, data.covariate_names.index("W1")]
        for j, dose in enumerate((0.0, 2.0)):
            expected = np.column_stack([np.ones(data.n), np.full(data.n, dose), w1, dose * w1])
            assert np.allclose(evaluated.design[:, j, :], expected)

    def test_the_arm_enters_as_its_label_not_its_code(self) -> None:
        # Levels 0 and 5 are codes 0 and 1.  A shorthand reading the code would make the
        # slope per-step rather than per-unit-of-dose, which is a different parameter.
        data = make_data(levels=(0, 5))
        evaluated = MSMSet.evaluate(MSM.linear(), data)
        assert np.allclose(evaluated.design[:, 1, 1], 5.0)

    def test_a_string_labelled_treatment_is_refused_rather_than_coded(self) -> None:
        data = make_data(levels=("low", "medium", "high"))
        with pytest.raises(DataError, match="not one anybody chose"):
            MSMSet.evaluate(MSM.linear(), data)

    def test_a_modifier_that_is_not_a_covariate_says_so(self) -> None:
        data = make_data()
        with pytest.raises(DataError, match="not one of the encoded"):
            MSMSet.evaluate(MSM.linear(modifiers=("nope",)), data)


# -------------------------------------------------------------------- evaluation


class TestEvaluate:
    def test_shapes_and_defaults(self) -> None:
        data = make_data(levels=(0, 1, 2))
        evaluated = MSMSet.evaluate(constant_design(), data)
        assert evaluated.design.shape == (data.n, 3, 2)
        assert evaluated.weights.shape == (data.n, 3)
        assert np.all(evaluated.weights == 1.0)  # uniform by default
        assert evaluated.arms == data.arm_codes
        assert evaluated.codes == (0.0, 1.0)
        assert evaluated.labels == {0.0: "(intercept)", 1.0: "a"}

    def test_a_continuous_treatment_is_refused(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "Y": rng.normal(size=60),
                "A": rng.normal(size=60),
                "W1": rng.normal(size=60),
            }
        )
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W1"], treatment_kind="continuous"
        )
        with pytest.raises(DataError, match="which has none"):
            MSMSet.evaluate(constant_design(), data)

    def test_a_design_of_the_wrong_shape_says_which_shape_it_wanted(self) -> None:
        data = make_data()
        wrong = MSM(design=lambda a, w: np.ones((len(w), 3)), terms=("(intercept)", "a"))
        with pytest.raises(DataError, match=r"expected \(40, 2\)"):
            MSMSet.evaluate(wrong, data)

    def test_weights_of_the_wrong_length_say_so(self) -> None:
        data = make_data()
        wrong = MSM(
            design=lambda a, w: np.column_stack([np.ones(len(w)), np.full(len(w), float(a))]),
            terms=("(intercept)", "a"),
            weights=lambda a, w: np.ones(3),
        )
        with pytest.raises(DataError, match="one weight per unit"):
            MSMSet.evaluate(wrong, data)

    def test_negative_weights_are_refused(self) -> None:
        data = make_data()
        wrong = MSM(
            design=lambda a, w: np.column_stack([np.ones(len(w)), np.full(len(w), float(a))]),
            terms=("(intercept)", "a"),
            weights=lambda a, w: -np.ones(len(w)),
        )
        with pytest.raises(DataError, match="not a signed contrast"):
            MSMSet.evaluate(wrong, data)

    def test_a_collinear_design_is_refused_where_it_is_built(self) -> None:
        # ``a`` and ``2a`` span one direction, so the projection is not a single vector.
        # Left to lstsq this would silently return the minimum-norm solution.
        data = make_data(levels=(0, 1))
        collinear = MSM(
            design=lambda a, w: np.column_stack(
                [np.full(len(w), float(a)), np.full(len(w), 2.0 * float(a))]
            ),
            terms=("a", "twice_a"),
        )
        with pytest.raises(DataError, match="collinear across the arms"):
            MSMSet.evaluate(collinear, data)


# ------------------------------------------------------------------- projection


class TestTheGramMatrix:
    """``M``, which exists here only so a rank-deficient design can be refused.

    The projection itself lives in ``inference.influence.msm_coefficients``, where the
    observation weights and the unscaling belong, and is checked against an oracle in
    ``tests/unit/test_influence_gateaux_msm.py``.  There is deliberately no second copy
    of it on this object to keep in step.
    """

    def test_it_is_the_weighted_sum_over_arms(self) -> None:
        data = make_data(levels=(0, 1, 2))
        evaluated = MSMSet.evaluate(MSM.linear(modifiers=("W1",)), data)
        expected = np.zeros((4, 4))
        for i in range(data.n):
            for j in range(3):
                phi = evaluated.design[i, j, :]
                expected += evaluated.weights[i, j] * np.outer(phi, phi)
        assert np.allclose(evaluated.gram, expected / data.n)

    def test_a_saturated_model_makes_it_the_identity(self) -> None:
        """One indicator per arm and uniform weights: ``M = I``, so ``beta`` is the means."""
        data = make_data(levels=(0, 1, 2))
        saturated = MSM(
            design=lambda a, w, arms=(0.0, 1.0, 2.0): np.column_stack(
                [np.full(len(w), float(float(a) == level)) for level in arms]
            ),
            terms=("arm0", "arm1", "arm2"),
        )
        assert np.allclose(MSMSet.evaluate(saturated, data).gram, np.eye(3))

    def test_the_weighted_design_is_the_product_the_covariate_needs(self) -> None:
        data = make_data(levels=(0, 1, 2))
        model = MSM.linear(modifiers=("W1",), weights=lambda a, w: 1.0 + float(a) * np.ones(len(w)))
        evaluated = MSMSet.evaluate(model, data)
        assert np.allclose(
            evaluated.weighted_design, evaluated.design * evaluated.weights[:, :, None]
        )


class TestSubset:
    def test_rows_are_sliced_and_the_gram_recomputed(self) -> None:
        data = make_data(levels=(0, 1, 2))
        evaluated = MSMSet.evaluate(MSM.linear(modifiers=("W1",)), data)
        keep = np.arange(0, data.n, 2)
        subset = evaluated.subset(keep)
        assert subset.n == keep.size
        assert np.allclose(subset.design, evaluated.design[keep])
        # M is an empirical mean, so it moves with the rows rather than being carried.
        direct = MSMSet(
            evaluated.terms, evaluated.design[keep], evaluated.weights[keep], evaluated.arms
        )
        assert np.allclose(subset.gram, direct.gram)

    def test_a_boolean_mask_selects_the_same_rows(self) -> None:
        data = make_data(levels=(0, 1))
        evaluated = MSMSet.evaluate(MSM.linear(), data)
        mask = np.zeros(data.n, dtype=bool)
        mask[::3] = True
        assert np.allclose(
            evaluated.subset(mask).design, evaluated.subset(np.flatnonzero(mask)).design
        )

    def test_arm_column_is_keyed_by_code(self) -> None:
        data = make_data(levels=(0, 1, 2))
        evaluated = MSMSet.evaluate(MSM.linear(), data)
        assert np.allclose(evaluated.arm_column(2.0), evaluated.design[:, 2, :])
