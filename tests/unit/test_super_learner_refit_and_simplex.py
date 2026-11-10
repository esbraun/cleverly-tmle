"""SuperLearner refits and the simplex least-squares meta-learner (Codex review CDR-002, CDR-007).

A refit must replace every fitted attribute of the earlier fit, whichever branch either fit
took. ``"cc_ls"`` must return the least-squares minimiser on the simplex, which normalised
``"nnls"`` does not: the reviewer's two-point law is the witness that tells the two apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from cleverly import SuperLearner
from cleverly.learners import super_learner as module


class Fixed(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """A candidate that ignores the data: ``scale * x + bump * 1{x = 2}``."""

    def __init__(self, scale: float = 1.0, bump: float = 0.0) -> None:
        self.scale = scale
        self.bump = bump

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Fixed:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scale * X[:, 0] + self.bump * (X[:, 0] == 2.0)


def _two_point(meta_learner: str = "auto") -> SuperLearner:
    # P(X = 1) = P(X = 2) = 1/2 and Y = X. f1 = 2X has risk 2.5, f2 = X + 0.1 1{X = 2} has 0.005.
    # Unconstrained NNLS gives (0.5, 0) with zero loss; dividing by the sum gives (1, 0).
    x = np.tile([[1.0], [2.0]], (50, 1))
    return SuperLearner(
        library=[("twice", Fixed(2.0, 0.0)), ("near", Fixed(1.0, 0.1))],
        task="regression",
        meta_learner=meta_learner,  # type: ignore[arg-type]
        n_folds=5,
        random_state=1,
    ).fit(x, x[:, 0])


def _fitted_state(model: SuperLearner) -> dict[str, object]:
    return {key: value for key, value in vars(model).items() if key.endswith("_")}


class TestARefitReplacesTheEarlierFit:
    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(90, 2))
        three = np.tile([0.0, 1.0, 2.0], 30)
        two = (x[:, 0] > 0).astype(float)
        continuous = x[:, 0] + rng.normal(size=90)
        return x, three, two, continuous

    @staticmethod
    def fresh(task: str) -> SuperLearner:
        library = [LogisticRegression()] if task == "classification" else [LinearRegression()]
        return SuperLearner(library=library, task=task, n_folds=3, random_state=1)  # type: ignore[arg-type]

    def test_multiclass_then_binary(self, data) -> None:  # type: ignore[no-untyped-def]
        x, three, two, _ = data
        model = self.fresh("classification").fit(x, three)
        assert model.is_multiclass
        model.fit(x, two)
        reference = self.fresh("classification").fit(x, two)
        assert not model.is_multiclass
        assert model.classes_.tolist() == [0.0, 1.0]
        assert set(_fitted_state(model)) == set(_fitted_state(reference))
        np.testing.assert_array_equal(model.predict_proba(x), reference.predict_proba(x))

    def test_binary_then_multiclass(self, data) -> None:  # type: ignore[no-untyped-def]
        x, three, two, _ = data
        model = self.fresh("classification").fit(x, two)
        model.fit(x, three)
        reference = self.fresh("classification").fit(x, three)
        assert model.is_multiclass
        # The binary fit's coef_ and diagnostics_ would read as the one-vs-rest ensemble's own.
        assert not hasattr(model, "coef_")
        assert not hasattr(model, "diagnostics_")
        assert set(_fitted_state(model)) == set(_fitted_state(reference))
        np.testing.assert_array_equal(model.predict_proba(x), reference.predict_proba(x))

    def test_classification_then_regression(self, data) -> None:  # type: ignore[no-untyped-def]
        x, three, _, continuous = data
        model = SuperLearner(library=[LogisticRegression()], n_folds=3, random_state=1)
        model.fit(x, three)
        model.set_params(library=[LinearRegression()], task="regression")
        model.fit(x, continuous)
        reference = self.fresh("regression").fit(x, continuous)
        assert model.task_ == "regression"
        assert set(_fitted_state(model)) == set(_fitted_state(reference))
        np.testing.assert_array_equal(model.predict(x), reference.predict(x))

    def test_parameters_survive_the_reset(self, data) -> None:  # type: ignore[no-untyped-def]
        x, three, two, _ = data
        model = self.fresh("classification")
        before = model.get_params()
        model.fit(x, three).fit(x, two)
        assert model.get_params() == before


class TestTheSimplexLeastSquaresMetaLearner:
    def test_cc_ls_finds_the_simplex_minimiser_on_the_two_point_law(self) -> None:
        model = _two_point("cc_ls")
        np.testing.assert_allclose(model.coef_, [0.0, 1.0], atol=1e-12)
        assert model.cv_risk_.tolist() == pytest.approx([2.5, 0.005])
        assert model.diagnostics_.ensemble_cv_risk == pytest.approx(0.005, abs=1e-12)

    def test_normalised_nnls_still_picks_the_bad_candidate(self) -> None:
        # Mutation control: the named R heuristic keeps its behaviour, so the law can tell
        # the two meta-learners apart.
        model = _two_point("nnls")
        np.testing.assert_allclose(model.coef_, [1.0, 0.0], atol=1e-12)
        assert model.diagnostics_.ensemble_cv_risk == pytest.approx(2.5)

    def test_auto_picks_cc_ls_for_regression(self) -> None:
        np.testing.assert_array_equal(_two_point().coef_, _two_point("cc_ls").coef_)

    def test_auto_keeps_nnloglik_for_classification(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(200, 2))
        y = (x[:, 0] + rng.normal(size=200) > 0).astype(float)
        library = [LogisticRegression(), LogisticRegression(C=0.01)]
        auto = SuperLearner(library=library, n_folds=3, random_state=0).fit(x, y)
        named = SuperLearner(
            library=library, meta_learner="nnloglik", n_folds=3, random_state=0
        ).fit(x, y)
        np.testing.assert_array_equal(auto.coef_, named.coef_)

    def test_cc_ls_matches_a_brute_force_simplex_search(self) -> None:
        # Three correlated candidates with an interior optimum: the weights minimise the
        # cross-validated squared error over a fine grid of the simplex.
        rng = np.random.default_rng(11)
        x = rng.normal(size=(300, 3))
        y = x @ [1.0, 0.5, -0.5] + rng.normal(size=300)
        library = [
            ("ols", LinearRegression()),
            ("ridge", Ridge(alpha=200.0)),
            ("first", Fixed(1.0, 0.0)),
        ]
        model = SuperLearner(library=library, task="regression", n_folds=5, random_state=0)
        model.fit(x, y)
        z = model.cv_predictions_

        def risk(alpha: np.ndarray) -> float:
            return float(np.mean((y - z @ alpha) ** 2))

        grid = np.linspace(0.0, 1.0, 201)
        best = min(risk(np.array([a, b, 1.0 - a - b])) for a in grid for b in grid if a + b <= 1.0)
        assert model.coef_.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(model.coef_ >= 0.0)
        assert risk(model.coef_) <= best + 1e-12
        assert model.diagnostics_.ensemble_cv_risk <= model.cv_risk_.min() + 1e-12

    def test_a_solver_failure_falls_back_to_the_discrete_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Failed:
            success = False
            x = np.array([np.nan, np.nan])

        monkeypatch.setattr(module.optimize, "minimize", lambda *args, **kwargs: Failed())
        model = _two_point("cc_ls")
        np.testing.assert_array_equal(model.coef_, [0.0, 1.0])
        assert model.diagnostics_.best == "near"

    def test_nnloglik_falls_back_to_cc_ls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        real = module._solve_cc_ls

        def spy(*args: object) -> np.ndarray:
            calls.append("cc_ls")
            return real(*args)  # type: ignore[arg-type]

        class Failed:
            success = False
            x = np.array([np.nan, np.nan])

        real_minimize = module.optimize.minimize

        def failing_first(*args: object, **kwargs: object) -> object:
            # The log-likelihood solve fails; the least-squares fallback then runs normally.
            if not calls:
                return Failed()
            return real_minimize(*args, **kwargs)

        monkeypatch.setattr(module, "_solve_cc_ls", spy)
        monkeypatch.setattr(module.optimize, "minimize", failing_first)
        z = np.column_stack([np.full(40, 0.2), np.tile([0.1, 0.9], 20)])
        y = np.tile([0.0, 1.0], 20)
        alpha = module._solve_nnloglik(z, y, np.ones(40), np.array([0.7, 0.1]))
        assert calls == ["cc_ls"]
        np.testing.assert_allclose(alpha, [0.0, 1.0], atol=1e-12)

    def test_an_unknown_meta_learner_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'cc_ls'"):
            _two_point("simplex")


class TestTheDefaultReachesATMLEFit:
    """The fast tier's shared outcome learner names ``"nnls"``, so this fit runs ``"auto"``.

    A continuous outcome sends the outcome ensemble down the regression path, where ``"auto"``
    must resolve to ``"cc_ls"`` and return weights on the simplex.
    """

    def test_a_continuous_outcome_fit_combines_by_cc_ls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd

        from cleverly.estimators import TMLE

        solved: list[np.ndarray] = []
        real = module._solve_cc_ls

        def spy(*args: object) -> np.ndarray:
            alpha = real(*args)  # type: ignore[arg-type]
            solved.append(alpha)
            return alpha

        def forbidden(*args: object) -> np.ndarray:
            raise AssertionError("auto resolved to normalised nnls on a regression task")

        monkeypatch.setattr(module, "_solve_cc_ls", spy)
        monkeypatch.setattr(module, "_solve_nnls", forbidden)
        rng = np.random.default_rng(5)
        n = 400
        w = rng.normal(size=(n, 2))
        a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-w[:, 0]))).astype(float)
        y = 0.5 * a + w[:, 0] - 0.5 * w[:, 1] + rng.normal(scale=0.5, size=n)
        frame = pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": a, "Y": y})
        outcome = SuperLearner(
            library=[("ols", LinearRegression()), ("ridge", Ridge(alpha=50.0))],
            n_folds=3,
            random_state=0,
        )
        result = TMLE(
            outcome_learner=outcome,
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        assert solved, "the outcome ensemble never reached the simplex solver"
        for alpha in solved:
            assert alpha.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(alpha >= 0.0)
        assert np.isfinite(result.single()["ate"].psi)
