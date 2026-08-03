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
from cleverly.msm import (
    MSM,
    Link,
    MSMSet,
    link_for,
    refuse_unsupported,
    register_link,
    solve_projection,
)


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

    @pytest.mark.parametrize("link", ["identity", "log", "logit"])
    def test_a_registered_link_is_accepted(self, link: str) -> None:
        model = MSM(design=lambda a, w: np.ones((len(w), 1)), terms=("(intercept)",), link=link)  # type: ignore[arg-type]
        assert model.link == link

    def test_an_unknown_link_is_refused_with_the_ones_that_exist(self) -> None:
        with pytest.raises(NotImplementedError, match="registered ones are"):
            MSM(design=lambda a, w: np.ones((len(w), 1)), terms=("(intercept)",), link="probit")  # type: ignore[arg-type]

    def test_a_link_cannot_be_registered_twice(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_link(Link("logit", inverse=np.exp, slope=np.exp, curvature=np.exp))

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


class TestTheLinkAlgebra:
    """``dm/deta`` and ``d2m/deta2`` against a numerical derivative of ``m``.

    Written as functions of the *mean* rather than of the linear predictor, which is
    cheaper everywhere they are used and one substitution away from wrong -- so the check
    is against ``m(eta)`` differentiated numerically, not against the formula restated.
    """

    @pytest.mark.parametrize("name", ["identity", "log", "logit"])
    def test_the_derivatives_are_the_derivatives(self, name: str) -> None:
        link = link_for(name)
        eta = np.linspace(-2.0, 2.0, 41)
        step = 1e-5
        m = link.inverse(eta)
        first = (link.inverse(eta + step) - link.inverse(eta - step)) / (2.0 * step)
        second = (link.inverse(eta + step) - 2.0 * m + link.inverse(eta - step)) / step**2
        np.testing.assert_allclose(link.slope(m), first, atol=1e-8)
        np.testing.assert_allclose(link.curvature(m), second, atol=1e-5)

    def test_only_the_identity_calls_itself_one(self) -> None:
        assert link_for("identity").is_identity
        assert not link_for("log").is_identity
        assert not link_for("logit").is_identity


class TestSolvingTheProjection:
    """``solve_projection`` is the one solver, so it is checked one link at a time.

    The oracle for the whole estimand is ``tests/discrete_law.py``; this is the algebra
    underneath it, checked against a statement of the estimating equation written out
    here so that a sign slip in ``U`` or ``M`` fails at the smallest scale it can.
    """

    @staticmethod
    def _problem(seed: int = 0, n: int = 120, k: int = 3, p: int = 3):
        rng = np.random.default_rng(seed)
        phi = rng.normal(size=(n, k, p))
        phi[:, :, 0] = 1.0
        h = rng.uniform(0.5, 2.0, size=(n, k))
        w = rng.uniform(0.5, 1.5, size=n)
        q = rng.uniform(0.1, 0.9, size=(n, k))
        return phi, h, q, w

    @staticmethod
    def _score(phi, h, q, w, link, beta):
        """``U(beta)``, longhand."""
        m = link.inverse(np.einsum("ijp,p->ij", phi, beta))
        return np.einsum("ijp,ij,i->p", phi, h * link.slope(m) * (q - m), w) / w.sum()

    @pytest.mark.parametrize("name", ["identity", "log", "logit"])
    def test_it_solves_the_estimating_equation(self, name: str) -> None:
        phi, h, q, w = self._problem()
        fit = solve_projection(phi, h, q, w, name)
        assert fit.converged
        residual = self._score(phi, h, q, w, link_for(name), fit.beta)
        assert np.max(np.abs(residual)) < 1e-11

    def test_the_identity_link_is_the_closed_form_bit_for_bit(self) -> None:
        """No iteration, and the same two einsums the projection has always used."""
        phi, h, q, w = self._problem()
        mass = w.sum()
        gram = np.einsum("ijp,ijq,ij,i->pq", phi, phi, h, w) / mass
        moment = np.einsum("ijp,ij,i->p", phi * h[:, :, None], q, w) / mass
        fit = solve_projection(phi, h, q, w, "identity")
        assert fit.n_iter == 0
        np.testing.assert_array_equal(fit.beta, np.linalg.solve(gram, moment))
        np.testing.assert_array_equal(fit.jacobian, gram)

    @pytest.mark.parametrize("name", ["log", "logit"])
    def test_the_jacobian_carries_the_curvature_term(self, name: str) -> None:
        """``M`` is ``-dU/dbeta``, checked by differentiating ``U`` numerically.

        The term that separates the two candidates is ``-(Qbar - m) d2m/deta2``, so it is
        large only where the working model fits *badly*: the counterfactual means here are
        ``0.95, 0.05, 0.95`` against a model linear in the dose, which cannot follow them
        at all. A Gram-only ``M`` is then out by 0.04 (logit) and 0.39 (log), four to five
        orders past the tolerance the numerical check uses.
        """
        n = 100
        phi = np.tile(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]), (n, 1, 1))
        h, w = np.ones((n, 3)), np.ones(n)
        q = np.tile(np.array([0.95, 0.05, 0.95]), (n, 1))
        link = link_for(name)
        fit = solve_projection(phi, h, q, w, name)
        step = 1e-6
        numerical = np.column_stack(
            [
                (
                    self._score(phi, h, q, w, link, fit.beta - step * np.eye(2)[j])
                    - self._score(phi, h, q, w, link, fit.beta + step * np.eye(2)[j])
                )
                / (2.0 * step)
                for j in range(2)
            ]
        )
        np.testing.assert_allclose(fit.jacobian, numerical, atol=1e-6)

        m = link.inverse(np.einsum("ijp,p->ij", phi, fit.beta))
        gram_only = np.einsum("ijp,ijq,ij,i->pq", phi, phi, h * link.slope(m) ** 2, w) / w.sum()
        assert np.max(np.abs(gram_only - fit.jacobian)) > 1e-2, (
            "the curvature term is too small here to tell the two matrices apart"
        )

    def test_a_saturated_design_recovers_the_means_through_the_link(self) -> None:
        """One indicator per arm: ``m(a) = Qbar(a)`` exactly, so ``beta = link(mean)``."""
        rng = np.random.default_rng(3)
        n = 80
        phi = np.tile(np.eye(3), (n, 1, 1))
        h = np.ones((n, 3))
        w = np.ones(n)
        q = rng.uniform(0.2, 0.8, size=(n, 3))
        fit = solve_projection(phi, h, q, w, "logit")
        np.testing.assert_allclose(
            1.0 / (1.0 + np.exp(-fit.beta)), q.mean(axis=0), rtol=0, atol=1e-10
        )

    def test_a_solve_that_cannot_converge_says_so_rather_than_raising(self) -> None:
        """One Newton step is not enough here, and the answer comes back labelled."""
        phi, h, q, w = self._problem()
        fit = solve_projection(phi, h, q, w, "logit", max_iter=1)
        assert not fit.converged
        assert fit.score > 1e-12


class TestTheLinkAndTheOutcomeMustAgree:
    def test_a_logit_model_needs_an_outcome_in_the_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {"Y": rng.normal(size=60) * 5.0, "A": np.tile([0, 1], 30), "W1": rng.normal(size=60)}
        )
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1"])
        with pytest.raises(DataError, match="needs an outcome in"):
            MSMSet.evaluate(MSM.linear(link="logit"), data)

    def test_a_log_model_needs_a_non_negative_outcome(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {"Y": rng.normal(size=60), "A": np.tile([0, 1], 30), "W1": rng.normal(size=60)}
        )
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1"])
        with pytest.raises(DataError, match="non-negative outcome"):
            MSMSet.evaluate(MSM.linear(link="log"), data)

    def test_a_binary_outcome_satisfies_both(self) -> None:
        data = make_data()
        for link in ("identity", "log", "logit"):
            assert MSMSet.evaluate(MSM.linear(link=link), data).link == link  # type: ignore[arg-type]


class TestTheCovariateNumerator:
    def test_the_identity_link_ignores_beta_entirely(self) -> None:
        data = make_data(levels=(0, 1, 2))
        evaluated = MSMSet.evaluate(MSM.linear(), data)
        np.testing.assert_array_equal(evaluated.weighted_design_at(None), evaluated.weighted_design)
        np.testing.assert_array_equal(
            evaluated.weighted_design_at(np.array([3.0, -2.0])), evaluated.weighted_design
        )

    def test_another_link_carries_the_slope(self) -> None:
        data = make_data(levels=(0, 1, 2))
        evaluated = MSMSet.evaluate(MSM.linear(link="logit"), data)
        beta = np.array([0.2, -0.5])
        m = evaluated.fitted(beta)
        expected = evaluated.design * (evaluated.weights * m * (1.0 - m))[:, :, None]
        np.testing.assert_allclose(evaluated.weighted_design_at(beta), expected)

    def test_a_covariate_without_a_beta_is_refused_under_a_link(self) -> None:
        data = make_data()
        evaluated = MSMSet.evaluate(MSM.linear(link="log"), data)
        with pytest.raises(DataError, match="depends on beta"):
            evaluated.weighted_design_at(None)


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
