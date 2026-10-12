"""X16: the ``lasso`` and ``learners`` candidate sequences, and the declared selection split.

Four groups of checks.

* The indexed construction rule, as a pure function on a scripted criterion table: the
  jump, the stop and the record after the stop (witnesses W1 and W2), and the fluctuation
  count.
* The L1-penalized logistic solver: its Karush-Kuhn-Tucker residual at every penalty
  (W6), the ends of the default grid (W4), and the separation refusal.
* The fits: the status and refusals of both strategies, the final step, the admitted empty
  list (W10), the declared order (W8), the arrays pair (W9) and the row identities of
  every fit (W7).
* The surfaces around a fit: round trips, the method configuration and the public
  ``lambda_CV`` recipe.

The R ``ctmleGlmnet`` and ``ctmleGeneral`` comparison lives in
``tests/canonical/ctmle_sequence_witness``.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cleverly import CollaborativeTMLEMethod
from cleverly.datasets import instrument_dgp, make_instrument, make_multi_arm
from cleverly.estimators import CTMLE, TMLE, PropensityArrays, logistic_plugin
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators.ctmle import _indexed_path
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.learners import _l1_logistic as l1
from tests.conftest import FAST_KWARGS
from tests.unit.test_ctmle import FULL_CANDIDATE_CASES, FullCandidateCase, _full_candidate_frame

IN_SAMPLE: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "cross_fit": False,
    "simultaneous": False,
    "random_state": 0,
    "estimands": ("ate",),
    "selection_folds": 3,
}

#: A short grid that ends above 0, so the unpenalized end never meets separation.
SHORT_GRID = (0.2, 0.05, 0.02, 0.005)


def _fit(frame: Any, **settings: Any) -> Any:
    options = {**IN_SAMPLE, **settings}
    if "selection_split" in settings and "selection_folds" not in settings:
        options.pop("selection_folds")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CTMLE(**options).fit(frame, outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def frame() -> Any:
    return make_instrument(n=500, seed=3)[0]


@pytest.fixture(scope="module")
def lasso_fit(frame: Any) -> Any:
    return _fit(frame, strategy="lasso", lambdas=SHORT_GRID)


@pytest.fixture(scope="module")
def learners_fit(frame: Any) -> Any:
    return _fit(
        frame,
        strategy="learners",
        treatment_candidates=[LogisticRegression(C=0.001), LogisticRegression(C=0.05)],
    )


# --------------------------------------------------------------- the indexed rule


def _scripted(table: dict[tuple[str, int], float]) -> tuple[Any, list[tuple[str, int, int]]]:
    """A criterion table keyed by (initial fit, candidate); each targeted fit is a name."""
    calls: list[tuple[str, int, int]] = []

    def fluctuate(index: int, initial: str, n_steps: int) -> Any:
        calls.append((initial, index, n_steps))
        return SimpleNamespace(
            risk=table[(initial, index)],
            targeted=f"{initial}>{index}",
            origin=initial,
            n_steps=n_steps,
            index=index,
        )

    return fluctuate, calls


class TestTheIndexedRule:
    """``_indexed_path`` against a scripted criterion, so every branch is pinned exactly."""

    #: Round 1 from q accepts candidate 1 and records 0 and 1 (a jump over 0). Round 2
    #: from q>1 accepts candidate 3 and records 2 and 3 (a jump over 2). Round 3 from
    #: q>1>3 finds 2.2 > 2.0, stops, and records candidate 4 after the stop.
    TABLE: ClassVar[dict[tuple[str, int], float]] = {
        ("q", 0): 5.0,
        ("q", 1): 3.0,
        ("q", 2): 4.0,
        ("q", 3): 6.0,
        ("q", 4): 7.0,
        ("q>1", 2): 2.5,
        ("q>1", 3): 2.0,
        ("q>1", 4): 9.0,
        ("q>1>3", 4): 2.2,
    }

    def test_jumps_and_the_stop(self) -> None:
        fluctuate, _ = _scripted(self.TABLE)
        path = _indexed_path(5, "q", fluctuate)
        assert [candidate.index for candidate in path] == [0, 1, 2, 3, 4]
        # W1: two accepted candidates, each a jump wider than one index.
        assert [candidate.origin for candidate in path] == ["q", "q", "q>1", "q>1", "q>1>3"]
        # n_steps is one plus the number of accepted indices before the candidate.
        assert [candidate.n_steps for candidate in path] == [1, 1, 2, 2, 3]
        # W2: the stop bound, and candidate 4 is a fluctuation of the last initial fit.
        assert path[4].risk > path[3].risk

    def test_without_the_stop_every_remaining_try_is_accepted(self) -> None:
        table = {**self.TABLE, ("q>1>3", 4): 1.0}
        fluctuate, _ = _scripted(table)
        path = _indexed_path(5, "q", fluctuate)
        assert path[4].origin == "q>1>3"
        assert path[4].n_steps == 3

    def test_a_tie_with_the_best_score_is_accepted_and_the_first_argmin_wins(self) -> None:
        table = {("q", 0): 2.0, ("q", 1): 2.0, ("q>0", 1): 2.0}
        fluctuate, _ = _scripted(table)
        path = _indexed_path(2, "q", fluctuate)
        assert [candidate.origin for candidate in path] == ["q", "q>0"]

    def test_a_non_finite_criterion_is_never_chosen(self) -> None:
        table = {("q", 0): float("nan"), ("q", 1): 3.0, ("q>1", 2): 1.0}
        fluctuate, _ = _scripted({**table, ("q", 2): float("inf")})
        path = _indexed_path(3, "q", fluctuate)
        assert [candidate.origin for candidate in path] == ["q", "q", "q>1"]

    @pytest.mark.parametrize("count", [1, 4, 9])
    def test_the_fluctuation_count_is_at_most_k_k_plus_one_over_two(self, count: int) -> None:
        """A criterion that always improves at the first remaining try is the worst case."""
        table: dict[tuple[str, int], float] = {}
        initial = "q"
        for index in range(count):
            for later in range(index, count):
                table[(initial, later)] = float(count - index) - 0.5 * (later > index)
            initial = f"{initial}>{index}"
        # Each round's first try is its smallest, so each round accepts one candidate.
        for key in list(table):
            if key[1] != int(key[0].count(">")):
                table[key] = table[key] + 10.0
        fluctuate, calls = _scripted(table)
        _indexed_path(count, "q", fluctuate)
        assert len(calls) == count * (count + 1) // 2


# --------------------------------------------------------------------- the solver


def _newton_logistic(x: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """An independent unpenalized logistic fit, solved to gradient below 1e-12."""
    design = np.column_stack([np.ones(len(a)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(design @ beta)))
        gradient = design.T @ (w * (a - p))
        if np.max(np.abs(gradient)) / len(a) < 1e-12:
            break
        hessian = (design * (w * p * (1.0 - p))[:, None]).T @ design
        beta = beta + np.linalg.solve(hessian, gradient)
    return 1.0 / (1.0 + np.exp(-(design @ beta)))


def _newton_multinomial(x: np.ndarray, codes: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """An independent reference-coded multinomial fit, solved to gradient below 1e-12."""
    design = np.column_stack([np.ones(len(codes)), x])
    d = design.shape[1]
    beta = np.zeros((d, k - 1))
    y = np.eye(k)[codes][:, 1:]
    for _ in range(100):
        eta = np.column_stack([np.zeros(len(codes)), design @ beta])
        p = np.exp(eta - eta.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        gradient = design.T @ (w[:, None] * (y - p[:, 1:]))
        if np.max(np.abs(gradient)) / len(codes) < 1e-12:
            break
        hessian = np.zeros((d * (k - 1), d * (k - 1)))
        for a in range(k - 1):
            for b in range(k - 1):
                c = p[:, a + 1] * ((a == b) - p[:, b + 1])
                hessian[a * d : (a + 1) * d, b * d : (b + 1) * d] = (
                    design * (w * c)[:, None]
                ).T @ design
        step = np.linalg.solve(hessian, gradient.T.reshape(-1))
        beta = beta + step.reshape(k - 1, d).T
    eta = np.column_stack([np.zeros(len(codes)), design @ beta])
    p = np.exp(eta - eta.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def design() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    n = 600
    x = rng.normal(size=(n, 5)) * np.array([1.0, 3.0, 0.5, 2.0, 1.0]) + 1.0
    eta = 0.7 * x[:, 0] - 0.2 * x[:, 1] + 0.4 * x[:, 2]
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    scores = np.column_stack([np.zeros(n), eta, 0.5 * eta - 0.3 * x[:, 3]])
    p = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    a3 = np.array([rng.choice(3, p=row) for row in p]).astype(float)
    w = rng.uniform(0.4, 2.0, n)
    return x, a, a3, w


class TestTheSolver:
    @pytest.mark.parametrize("arms", [2, 3])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_every_penalty_meets_the_kkt_tolerance(
        self, design: Any, arms: int, weighted: bool
    ) -> None:
        """W6: active and inactive coordinates, weighted, and the symmetric multinomial."""
        x, a, a3, w = design
        target = a if arms == 2 else a3
        weights = w if weighted else None
        grid = l1.default_penalty_grid(x, target, weights)
        fits = l1.penalty_path(x, target, weights, grid)
        residuals = [l1.kkt_residual(x, target, weights, fit) for fit in fits]
        assert max(residuals) <= l1.KKT_TOLERANCE
        # The restated residual agrees with the solver's own record.
        np.testing.assert_allclose(residuals, [fit.kkt for fit in fits], atol=1e-13)
        # Both kinds of coordinate occur along the path.
        assert any(0 < len(fit.active) < x.shape[1] for fit in fits)

    def test_the_kkt_residual_sees_a_wrong_coefficient(self, design: Any) -> None:
        x, a, _, w = design
        fit = l1.penalty_path(x, a, w, [0.01])[0]
        moved = l1.L1Fit(
            penalty=fit.penalty,
            intercept=fit.intercept,
            coef=fit.coef * 1.001,
            center=fit.center,
            scale=fit.scale,
            constant=None,
            kkt=0.0,
        )
        assert l1.kkt_residual(x, a, w, moved) > 1e3 * l1.KKT_TOLERANCE

    @pytest.mark.parametrize("arms", [2, 3])
    def test_the_default_grid_starts_at_the_intercept_and_ends_at_the_mle(
        self, design: Any, arms: int
    ) -> None:
        """W4 on the solver: the short circuit, ``lambda_max``, and the unpenalized end."""
        x, a, a3, w = design
        target = a if arms == 2 else a3
        grid = l1.default_penalty_grid(x, target, w)
        assert grid.size == 101 and grid[-1] == 0.0
        assert np.all(np.diff(grid) < 0.0)
        np.testing.assert_allclose(grid[99] / grid[0], 1e-4, rtol=1e-12)
        fits = l1.penalty_path(x, target, w, grid)
        codes = np.searchsorted(np.unique(target), target)
        proportions = np.array([np.sum(w * (codes == k)) / np.sum(w) for k in range(arms)])
        assert fits[0].constant is not None
        assert np.array_equal(fits[0].predict_proba(x[:3])[0], proportions)
        # lambda_max is the smallest penalty with every coefficient zero.
        below = l1.penalty_path(x, target, w, [grid[0] * (1.0 - 1e-6)])[0]
        assert below.constant is None and below.active
        if arms == 2:
            reference = _newton_logistic(x, a, w)
            np.testing.assert_allclose(fits[-1].predict_proba(x)[:, 1], reference, atol=1e-8)
        else:
            reference = _newton_multinomial(x, codes, w, 3)
            np.testing.assert_allclose(fits[-1].predict_proba(x), reference, atol=1e-8)

    def test_a_separated_unpenalized_end_raises_with_its_layer(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 3))
        a = (x[:, 0] > 0).astype(float)
        with pytest.raises(l1.L1ConvergenceError, match=r"penalty 0 did not converge in outer"):
            l1.penalty_path(x, a, None, l1.default_penalty_grid(x, a), layer="outer fold 1")
        assert len(l1.penalty_path(x, a, None, [0.1, 0.01])) == 2

    @pytest.mark.parametrize(
        "grid", [[], [0.1, 0.2], [0.1, 0.1], [0.1, -0.1], [0.0, 0.1], [np.inf, 0.1], [0.1, np.nan]]
    )
    def test_a_bad_grid_is_refused(self, grid: list[float]) -> None:
        with pytest.raises(ValueError, match=r"^lambdas must be a non-empty sequence"):
            l1.check_penalty_grid(grid)

    def test_a_constant_column_takes_no_coefficient(self, design: Any) -> None:
        x, a, _, w = design
        widened = np.column_stack([x, np.full(len(a), 2.0)])
        fits = l1.penalty_path(widened, a, w, [0.01, 0.0])
        assert all(fit.coef[-1, 1] == 0.0 for fit in fits)
        np.testing.assert_allclose(
            fits[-1].predict_proba(widened), l1.penalty_path(x, a, w, [0.0])[0].predict_proba(x)
        )

    def test_the_classifier_matches_the_path(self, design: Any) -> None:
        x, a, _, w = design
        model = l1.L1Logistic(penalty=0.01).fit(x, a, sample_weight=w)
        path = l1.penalty_path(x, a, w, [0.01])[0]
        np.testing.assert_allclose(model.predict_proba(x), path.predict_proba(x), atol=1e-10)


# ---------------------------------------------------------------------- the fits


class TestLasso:
    def test_status_path_and_labels(self, lasso_fit: Any) -> None:
        selection = lasso_fit.extra["ctmle"]
        assert lasso_fit.inference_status == "working_mechanism_plugin"
        assert selection.strategy == "lasso"
        assert len(selection.path) == len(SHORT_GRID)
        assert selection.labels == tuple(f"lambda={value:.4g}" for value in SHORT_GRID)
        assert np.isfinite(lasso_fit["ate"].plugin_std_error)
        with pytest.raises(CapabilityError, match=r"discrete, lasso and learners"):
            _ = lasso_fit["ate"].ci

    def test_the_final_step_follows_the_selection(self, lasso_fit: Any) -> None:
        selection = lasso_fit.extra["ctmle"]
        if selection.selected == len(selection.path) - 1:
            assert selection.final is None
        else:
            assert selection.final is not None and selection.final > selection.selected
        frame = selection.to_frame()
        assert frame["selected"].count(True) == 1
        assert frame["final"].count(True) == (selection.final is not None)
        assert selection.selected_covariates == selection.path[selection.reported]
        assert "final step used candidate" in selection.describe() or selection.final is None

    def test_the_first_candidate_is_the_intercept_bit_for_bit(self, frame: Any) -> None:
        """W4: the short circuit makes the top of the default grid the intercept candidate."""
        lasso = _fit(frame, strategy="lasso")
        intercept = _fit(frame, strategy="discrete", candidates=[()])
        selection = lasso.extra["ctmle"]
        reference = intercept.extra["ctmle"]
        assert selection.path[0] == ()
        assert selection.treatment_risk[0] == reference.treatment_risk[0]
        assert selection.train_risk[0] == reference.train_risk[0]
        assert len(selection.path) == 101 and selection.labels[-1] == "lambda=0"

    def test_the_last_default_candidate_is_the_unpenalized_logistic_fit(self, frame: Any) -> None:
        lasso = _fit(frame, strategy="lasso").extra["ctmle"]
        full = _fit(
            frame,
            strategy="discrete",
            candidates=[("W1", "W2", "W3")],
            treatment_learner=LogisticRegression(C=1e12, tol=1e-12, max_iter=10_000),
        ).extra["ctmle"]
        assert lasso.path[-1] == ("W1", "W2", "W3")
        assert lasso.treatment_risk[-1] == pytest.approx(full.treatment_risk[0], rel=1e-7)

    def test_three_arms_and_cross_fitting(self) -> None:
        frame = make_multi_arm(n=450, seed=2, family="binomial")[0]
        result = _fit(
            frame,
            strategy="lasso",
            lambdas=SHORT_GRID,
            cross_fit=True,
            n_folds=3,
            outcome_learner=LogisticRegression(max_iter=1000),
            estimands=("ey",),
            ctmle_estimand="ey",
        )
        assert result.inference_status == "working_mechanism_plugin"
        assert len(result.extra["ctmle"].path) == len(SHORT_GRID)

    def test_logistic_plugin_refuses(self, lasso_fit: Any) -> None:
        with pytest.raises(CapabilityError, match=r"strategy='lasso'.*plugin_std_error"):
            logistic_plugin(lasso_fit)


class TestLearners:
    def test_status_and_the_terminal(self, learners_fit: Any, frame: Any) -> None:
        selection = learners_fit.extra["ctmle"]
        assert learners_fit.inference_status == "working_mechanism_plugin"
        assert selection.labels == (
            "LogisticRegression[0]",
            "LogisticRegression[1]",
            "LogisticRegression[terminal]",
        )
        assert all(names == ("W1", "W2", "W3") for names in selection.path)
        assert selection.final is None
        full = _fit(frame, strategy="discrete", candidates=[("W1", "W2", "W3")]).extra["ctmle"]
        # W4: the terminal is treatment_learner's own fit, bit for bit.
        assert selection.treatment_risk[-1] == full.treatment_risk[0]
        assert "treatment risk decreases along the declared order: yes" in selection.summary()

    def test_the_declared_order_matters(self, frame: Any) -> None:
        """W8: swapping two declared learners that differ changes the path."""
        first = [LogisticRegression(C=0.001), LogisticRegression(C=0.05)]
        swapped = [LogisticRegression(C=0.05), LogisticRegression(C=0.001)]
        a = _fit(frame, strategy="learners", treatment_candidates=first).extra["ctmle"]
        b = _fit(frame, strategy="learners", treatment_candidates=swapped).extra["ctmle"]
        assert not np.array_equal(a.treatment_risk, b.treatment_risk)
        assert "no" in b.summary().splitlines()[-1]

    def test_logistic_plugin_refuses(self, learners_fit: Any) -> None:
        with pytest.raises(CapabilityError, match=r"strategy='learners', which fits declared"):
            logistic_plugin(learners_fit)

    def test_a_random_common_cause_refit_gives_learners_the_added_covariate(
        self, learners_fit: Any
    ) -> None:
        noise = np.random.default_rng(0).normal(size=learners_fit.data.n)
        data = learners_fit.data.with_extra_covariate(noise, "_noise_0")
        refit = learners_fit.estimator.refit(data, intermediate_value=None, random_state=0)
        assert refit.extra["ctmle"].path[-1] == ("W1", "W2", "W3", "_noise_0")


@pytest.mark.parametrize("case", FULL_CANDIDATE_CASES, ids=lambda case: case.name)
def test_an_empty_learners_list_is_the_ordinary_tmle(case: FullCandidateCase) -> None:
    """W10: nothing to select, so the fit equals ``TMLE`` and takes its status."""
    if case.name == "candidate_order_is_free":
        pytest.skip("the order of a declared covariate set does not apply to learners")
    frame = _full_candidate_frame(case.frame)
    settings = {**FAST_KWARGS, **case.settings}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = (
            CTMLE(**settings, **case.ctmle, strategy="learners", treatment_candidates=[])
            .fit(frame, **case.fit)
            .single()
        )
        plain = TMLE(**settings).fit(frame, **case.fit).single()
    assert mine.inference_status == plain.inference_status == "influence_curve"
    for name, estimate in plain.estimates.items():
        assert mine[name].psi == pytest.approx(estimate.psi, abs=1e-12)
        assert mine[name].std_error == pytest.approx(estimate.std_error, abs=1e-12)
        np.testing.assert_allclose(mine[name].ci, estimate.ci, atol=1e-12, rtol=0.0)


# ------------------------------------------------------------------- the arrays


def _oof(learner: Any, x: np.ndarray, a: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.empty(len(a))
    for fold in np.unique(labels):
        held = labels == fold
        out[held] = learner.fit(x[~held], a[~held]).predict_proba(x[held])[:, 1]
    return out


@pytest.fixture(scope="module")
def split(frame: Any) -> np.ndarray:
    return np.random.default_rng(4).permutation(np.arange(len(frame)) % 3)


@pytest.fixture(scope="module")
def arrays(frame: Any, split: np.ndarray) -> tuple[PropensityArrays, np.ndarray, np.ndarray]:
    x = frame[["W1", "W2", "W3"]].to_numpy(float)
    a = frame["A"].to_numpy(float)
    learner = LogisticRegression(C=0.05, max_iter=1000)
    full = learner.fit(x, a).predict_proba(x)[:, 1]
    cv = _oof(LogisticRegression(C=0.05, max_iter=1000), x, a, split)
    return PropensityArrays(full=full, cv=cv), full, cv


class TestArrays:
    def test_the_split_is_the_declared_split(self, frame: Any, split: Any, arrays: Any) -> None:
        pair, _, _ = arrays
        result = _fit(
            frame, strategy="learners", treatment_candidates=[pair], selection_split=split
        )
        folds = result.extra["ctmle"].folds
        assert np.array_equal(folds.assignment, split)
        assert folds.origin is None
        assert result.extra["ctmle"].labels[0] == "arrays[0]"

    def test_the_analysis_path_reads_full_and_the_cv_reads_cv(
        self, frame: Any, split: Any, arrays: Any
    ) -> None:
        """W9: the same ``full`` with two ``cv`` columns moves the CV risk only."""
        pair, full, _ = arrays
        same = PropensityArrays(full=full, cv=full)
        a = _fit(frame, strategy="learners", treatment_candidates=[pair], selection_split=split)
        b = _fit(frame, strategy="learners", treatment_candidates=[same], selection_split=split)
        sa, sb = a.extra["ctmle"], b.extra["ctmle"]
        assert np.array_equal(sa.train_risk, sb.train_risk)
        assert not np.allclose(sa.cv_risk, sb.cv_risk, rtol=1e-6, atol=0.0)

    def test_a_learner_and_its_full_array_give_one_analysis_path(
        self, frame: Any, split: Any, arrays: Any
    ) -> None:
        pair, _, _ = arrays
        by_array = _fit(
            frame, strategy="learners", treatment_candidates=[pair], selection_split=split
        )
        by_learner = _fit(
            frame,
            strategy="learners",
            treatment_candidates=[LogisticRegression(C=0.05, max_iter=1000)],
            selection_split=split,
        )
        np.testing.assert_allclose(
            by_array.extra["ctmle"].train_risk, by_learner.extra["ctmle"].train_risk, rtol=1e-10
        )

    @pytest.mark.parametrize(
        ("settings", "error", "pattern"),
        [
            ({"treatment_candidates": [np.full(500, 0.5)]}, CapabilityError, "single propensity"),
            ({}, CapabilityError, r"need selection_split="),
            ({"cross_fit": True}, CapabilityError, "cross-fitted fit needs one fit"),
        ],
    )
    def test_refusals(
        self, arrays: Any, settings: dict[str, Any], error: type, pattern: str
    ) -> None:
        pair, _, _ = arrays
        options = {**IN_SAMPLE, "strategy": "learners", "treatment_candidates": [pair], **settings}
        if "cross_fit" in settings:
            options["selection_split"] = np.arange(500) % 3
        with pytest.raises(error, match=pattern):
            CTMLE(**options)

    def test_a_wrong_shape_names_the_entry(self, frame: Any, split: Any) -> None:
        pair = PropensityArrays(full=np.full(10, 0.5), cv=np.full(500, 0.5))
        with pytest.raises(ValueError, match=r"PropensityArrays entry 0: full and cv must"):
            _fit(frame, strategy="learners", treatment_candidates=[pair], selection_split=split)

    def test_a_refit_refuses(self, frame: Any, split: Any, arrays: Any) -> None:
        pair, _, _ = arrays
        result = _fit(
            frame, strategy="learners", treatment_candidates=[pair], selection_split=split
        )
        data = result.data.with_extra_covariate(np.zeros(result.data.n), "_noise_0")
        with pytest.raises(CapabilityError, match=r"PropensityArrays candidates were fitted"):
            result.estimator.refit(data, intermediate_value=None, random_state=0)


# ----------------------------------------------------------------- the split


class TestTheDeclaredSplit:
    @pytest.mark.parametrize("strategy", ["greedy", "lasso"])
    def test_every_selector_strategy_records_it(
        self, frame: Any, split: Any, strategy: str
    ) -> None:
        extra = {"lambdas": SHORT_GRID} if strategy == "lasso" else {}
        result = _fit(frame, strategy=strategy, selection_split=split, **extra)
        assert np.array_equal(result.extra["ctmle"].folds.assignment, split)

    @pytest.mark.parametrize(
        ("settings", "error", "pattern"),
        [
            ({"repeats": 2, "cross_fit": True}, CapabilityError, "repeats= redraws"),
            ({"n_bootstrap": 5}, CapabilityError, "bootstrap resample changes"),
            ({"selection_folds": 4}, ValueError, "replaces selection_folds"),
            ({"strategy": "oat"}, ValueError, r"selection_split= configure selector"),
        ],
    )
    def test_refusals(self, settings: dict[str, Any], error: type, pattern: str) -> None:
        options = {**IN_SAMPLE, "selection_split": np.arange(500) % 3, **settings}
        if options.get("strategy") == "oat":
            options.pop("selection_folds")
        with pytest.raises(error, match=pattern):
            CTMLE(**options)

    @pytest.mark.parametrize(
        "labels",
        [np.arange(499) % 3, np.arange(500) % 3 + 1, np.zeros(500), np.arange(500) % 3 * 0.5],
    )
    def test_a_bad_split_is_refused(self, frame: Any, labels: Any) -> None:
        with pytest.raises(ValueError, match=r"selection_split must hold one fold label"):
            _fit(frame, selection_split=labels)


# ------------------------------------------------------------- configuration


class TestConfiguration:
    @pytest.mark.parametrize(
        ("settings", "pattern"),
        [
            ({"strategy": "greedy", "lambdas": [0.1]}, r"lambdas= only applies"),
            ({"strategy": "greedy", "treatment_candidates": []}, r"treatment_candidates= only"),
            ({"strategy": "learners"}, r"needs an explicit treatment_candidates="),
            ({"strategy": "lasso", "lambdas": [0.1, 0.2]}, r"strictly decreasing"),
            ({"strategy": "nope"}, r"'lasso', 'learners' or 'oat'"),
        ],
    )
    def test_value_errors(self, settings: dict[str, Any], pattern: str) -> None:
        with pytest.raises(ValueError, match=pattern):
            CTMLE(**{**IN_SAMPLE, **settings})

    def test_the_method_carries_the_new_fields(self, split: Any) -> None:
        method = CollaborativeTMLEMethod(
            strategy="learners",
            treatment_candidates=[LogisticRegression(C=0.1)],
            selection_split=list(split),
        )
        kwargs = method.estimator_kwargs()
        assert isinstance(method.treatment_candidates, tuple)
        assert kwargs["selection_split"] == tuple(split)
        assert kwargs["lambdas"] is None
        lasso = CollaborativeTMLEMethod(strategy="lasso", lambdas=[0.1, 0.0])
        assert lasso.estimator_kwargs()["lambdas"] == (0.1, 0.0)

    @pytest.mark.parametrize("kind", ["lasso", "learners", "arrays"])
    def test_a_saved_fit_round_trips(
        self, kind: str, frame: Any, split: Any, arrays: Any, lasso_fit: Any, learners_fit: Any
    ) -> None:
        if kind == "lasso":
            result = lasso_fit
        elif kind == "learners":
            result = learners_fit
        else:
            result = _fit(
                frame, strategy="learners", treatment_candidates=[arrays[0]], selection_split=split
            )
        restored = loads(dumps(result))
        assert restored.inference_status == "working_mechanism_plugin"
        assert restored.extra["ctmle"].labels == result.extra["ctmle"].labels
        assert restored["ate"].psi == result["ate"].psi
        with pytest.raises(CapabilityError):
            _ = restored["ate"].std_error

    def test_the_public_lambda_cv_recipe(self, frame: Any) -> None:
        """The docs recipe for a grid that starts at a cross-validated penalty."""
        x = frame[["W1", "W2", "W3"]].to_numpy(float)
        a = frame["A"].to_numpy(int)
        scaler = StandardScaler().fit(x)
        cv = LogisticRegressionCV(
            Cs=10, l1_ratios=(1.0,), solver="saga", max_iter=5000, random_state=0
        ).fit(scaler.transform(x), a)
        start = 1.0 / (float(np.ravel(cv.C_)[0]) * len(a))
        grid = start * np.geomspace(1.0, 0.01, 6)
        result = _fit(frame, strategy="lasso", lambdas=grid)
        assert len(result.extra["ctmle"].path) == 6


# ---------------------------------------------------------- row identities (W7)


class RowSpy(ClassifierMixin, BaseEstimator):
    """A logistic learner that records the row identities of each fit and prediction."""

    log: ClassVar[list[tuple[str, frozenset[float]]]] = []

    def __init__(self, column: int = 0) -> None:
        self.column = column

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RowSpy:
        self.ids_ = frozenset(np.asarray(X)[:, self.column].tolist())
        RowSpy.log.append(("fit", self.ids_))
        self.model_ = LogisticRegression(max_iter=1000).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        RowSpy.log.append(("predict", frozenset(np.asarray(X)[:, self.column].tolist())))
        RowSpy.log.append(("model", self.ids_))
        return self.model_.predict_proba(X)


def _check_identities(pairs: list[tuple[frozenset[float], frozenset[float]]], every: set) -> None:
    for fitted, predicted in pairs:
        if fitted == every and predicted == every:
            continue  # the in-sample analysis fit
        assert not fitted & predicted, "a row reached the model that predicts it"


class TestRowIdentities:
    """W7: no selection-validation row reaches a fit that scores it."""

    @pytest.fixture
    def tagged(self, frame: Any) -> Any:
        tagged = frame.copy()
        tagged["ID"] = np.arange(len(frame), dtype=float)
        return tagged

    def test_learners(self, tagged: Any) -> None:
        RowSpy.log = []
        _fit(
            tagged,
            strategy="learners",
            treatment_candidates=[RowSpy(column=3)],
            treatment_learner=LogisticRegression(max_iter=1000),
        )
        pairs = []
        log = RowSpy.log
        for index in range(len(log) - 1):
            if log[index][0] == "predict" and log[index + 1][0] == "model":
                pairs.append((log[index + 1][1], log[index][1]))
        assert len(pairs) >= 1 + 3 * 3
        _check_identities(pairs, set(tagged["ID"].tolist()))

    def test_lasso(self, tagged: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[tuple[frozenset[float], frozenset[float]]] = []
        original = ctmle_module.penalty_path

        def spy(x: Any, *args: Any, **kwargs: Any) -> Any:
            fitted = frozenset(np.asarray(x)[:, 3].tolist())
            fits = original(x, *args, **kwargs)

            class Recorder:
                def __init__(self, fit: Any) -> None:
                    self.fit = fit
                    self.active = fit.active

                def predict_proba(self, z: Any) -> Any:
                    recorded.append((fitted, frozenset(np.asarray(z)[:, 3].tolist())))
                    return self.fit.predict_proba(z)

            return [Recorder(fit) for fit in fits]

        monkeypatch.setattr(ctmle_module, "penalty_path", spy)
        _fit(tagged, strategy="lasso", lambdas=SHORT_GRID)
        assert len(recorded) >= len(SHORT_GRID) * (1 + 3 * 3)
        _check_identities(recorded, set(tagged["ID"].tolist()))


# ------------------------------------------------------- forced selection (W5)


class TestSelectionIsForced:
    """W5 for both strategies: with a useless outcome model, ``g`` must carry the adjustment."""

    SEEDS = (0, 1, 2)
    N = 1500

    @pytest.fixture(scope="class")
    def fits(self) -> list[tuple[Any, Any, Any, float]]:
        dgp = instrument_dgp()
        truth = float(dgp.truth()["ate"])
        settings = {**IN_SAMPLE, "outcome_learner": DummyRegressor(strategy="mean")}
        select_w2 = make_pipeline(
            ColumnTransformer([("w2", "passthrough", [1])]), LogisticRegression(max_iter=1000)
        )
        out = []
        for seed in self.SEEDS:
            frame, _ = dgp.sample(self.N, seed=seed)
            lasso = _fit(frame, **settings, strategy="lasso", lambdas=(0.3, 0.1, 0.03, 0.01))
            learners = _fit(
                frame, **settings, strategy="learners", treatment_candidates=[select_w2]
            )
            nothing = _fit(frame, **settings, strategy="discrete", candidates=[()])
            out.append((lasso, learners, nothing, truth))
        return out

    def test_the_lasso_mechanism_includes_the_confounder(self, fits: Any) -> None:
        for lasso, _, _, _ in fits:
            chosen = lasso.extra["ctmle"].selected_covariates
            assert "W1" in chosen

    def test_both_beat_a_do_nothing_selector_by_five_times(self, fits: Any) -> None:
        lasso = np.mean([abs(f[0]["ate"].psi - f[3]) for f in fits])
        learners = np.mean([abs(f[1]["ate"].psi - f[3]) for f in fits])
        nothing = np.mean([abs(f[2]["ate"].psi - f[3]) for f in fits])
        assert nothing >= 5.0 * lasso
        assert nothing >= 5.0 * learners
