"""The optional glmstar accelerator of the L1-penalized logistic path (X106 K2c-2).

``CTMLE(l1_solver="glmstar")`` starts each positive penalty below the null gradient from
glmstar's solution and keeps it only once the built-in Karush-Kuhn-Tucker check passes. The
first block runs without glmstar: keyword validation, the refusals, the loader's errors, and
the fallback and polish branches with an injected adapter. The second block needs glmstar and
checks the adapter's mapping: positive bounds on the pre-polish residual, mutation controls
that each push it to 1e-6 or more, determinism, and agreement of the public fit.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import sys
import types
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly import CollaborativeTMLEMethod
from cleverly.estimators import CTMLE
from cleverly.learners import GLM, _glmstar
from cleverly.learners import _l1_logistic as l1
from cleverly.longitudinal import SurvivalCTMLE


def _has_glmstar() -> bool:
    try:
        _glmstar.load()
    except ImportError:
        return False
    return True


HAS_GLMSTAR = _has_glmstar()
#: The CI ``glmstar`` job sets ``CLEVERLY_REQUIRE_GLMSTAR`` so a broken install fails, not skips.
REQUIRED = bool(os.environ.get("CLEVERLY_REQUIRE_GLMSTAR"))
needs_glmstar = pytest.mark.skipif(
    not (HAS_GLMSTAR or REQUIRED),
    reason='glmstar is not installed (pip install "cleverly[glmstar]")',
)
COVARIATES = ["W1", "W2", "W3", "W4"]


def _frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 4))
    a = rng.binomial(1, expit(0.6 * w[:, 0] - 0.5 * w[:, 1]))
    y = rng.binomial(1, expit(0.3 * w[:, 0] + 0.4 * a))
    frame = pd.DataFrame(w, columns=COVARIATES)
    frame["A"] = a
    frame["Y"] = y
    return frame


def _fit(frame: pd.DataFrame, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": GLM(family="binomial"),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 5,
        "estimands": ("ate",),
        **overrides,
    }
    return CTMLE(**settings).fit(frame, outcome="Y", treatment="A").single()


def _binary(n: int = 3000, p: int = 6, seed: int = 11, weighted: bool = False) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)) * np.linspace(0.5, 3.0, p) + np.linspace(-1.0, 2.0, p)
    a = rng.binomial(1, expit(0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2]))
    w = rng.uniform(0.2, 3.0, n) if weighted else None
    return x, a, w


def _three_class(n: int = 3000, p: int = 5, seed: int = 12) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    eta = np.column_stack([np.zeros(n), x[:, 0] - 0.5 * x[:, 1], 0.6 * x[:, 2]])
    prob = np.exp(eta)
    prob /= prob.sum(axis=1, keepdims=True)
    a = (rng.uniform(size=(n, 1)) > np.cumsum(prob, axis=1)).sum(axis=1)
    return x, a, None


def _problem(x: Any, a: Any, w: Any) -> l1._Problem:
    order = np.unique(a)
    codes = np.searchsorted(order, a).astype(np.int64)
    weights = np.ones(a.size) if w is None else np.asarray(w, dtype=float)
    return l1._Problem(np.asarray(x, dtype=float), codes, weights, order.size)


def _positive(problem: l1._Problem, grid: Any) -> np.ndarray:
    gradient = problem.null_gradient()
    return np.array([v for v in grid if 0.0 < v < gradient])


def _max_kkt(problem: l1._Problem, penalties: Any, starts: list[Any]) -> float:
    return max(problem.kkt(b0, b, v) for v, (b0, b) in zip(penalties, starts, strict=True))


# --------------------------------------------------------------- without glmstar


class TestTheKeyword:
    def test_an_unknown_solver_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="l1_solver must be one of"):
            _fit(_frame(), strategy="lasso", l1_solver="cd")

    def test_penalty_path_names_its_solvers(self) -> None:
        x, a, _ = _binary(200)
        with pytest.raises(ValueError, match="solver must be one of"):
            l1.penalty_path(x, a, None, [0.1, 0.0], solver="cd")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"strategy": "greedy"},
            {"strategy": "ordered", "preorder": "logistic"},
            {"strategy": "discrete", "candidates": [["W1"], ["W1", "W2"]]},
            {"strategy": "learners", "treatment_candidates": [LogisticRegression()]},
            {"strategy": "oat"},
            {
                "strategy": "derivative",
                "treatment_candidates": [LogisticRegression(C=0.01), LogisticRegression()],
            },
        ],
        ids=["greedy", "ordered", "discrete", "learners", "oat", "derivative-learners"],
    )
    def test_strategies_without_an_l1_path_refuse_glmstar(self, overrides: Any) -> None:
        with pytest.raises(ValueError, match="l1_solver= only applies to strategy='lasso'"):
            _fit(_frame(), l1_solver="glmstar", **overrides)

    def test_survival_ctmle_refuses_glmstar(self) -> None:
        # The refusal table row is in tests/unit/test_survival_ctmle.py; this pins the default.
        assert SurvivalCTMLE({"a1": 1, "a0": 0}).l1_solver == "builtin"

    def test_the_facade_passes_the_keyword_to_both_engines(self) -> None:
        method = CollaborativeTMLEMethod(strategy="lasso", l1_solver="glmstar")
        assert method.estimator_kwargs()["l1_solver"] == "glmstar"
        assert method.estimator_kwargs(longitudinal=True)["l1_solver"] == "glmstar"
        assert CollaborativeTMLEMethod().estimator_kwargs()["l1_solver"] == "builtin"

    def test_assessment_cache_keys_read_the_keyword(self) -> None:
        from cleverly._assessment_cache import _cache_key

        keys = {
            _cache_key("assess", (), {"method": CollaborativeTMLEMethod(l1_solver=solver)})
            for solver in ("builtin", "glmstar")
        }
        assert len(keys) == 2

    def test_the_builtin_default_records_zero_work(self) -> None:
        selection = _fit(_frame(), strategy="lasso", lambdas=[0.2, 0.05, 0.01, 0.0]).extra["ctmle"]
        assert selection.l1_solver == "builtin"
        assert selection.l1_polished == (0,) * 6
        assert selection.l1_fallbacks == (0,) * 6
        assert selection.l1_fallback_reasons == ()

    def test_other_strategies_leave_the_fields_empty(self) -> None:
        selection = _fit(_frame()).extra["ctmle"]
        assert (selection.l1_polished, selection.l1_fallbacks) == ((), ())


@pytest.fixture
def fresh_loader() -> Any:
    """``load`` is cached; a loader test sees a cold cache and leaves one behind."""
    _glmstar.load.cache_clear()
    yield
    _glmstar.load.cache_clear()


@pytest.mark.usefixtures("fresh_loader")
class TestTheLoader:
    def test_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", missing)
        with pytest.raises(ImportError, match=r'pip install "cleverly\[glmstar\]"'):
            _glmstar.load()

    def test_old_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.6.9")
        with pytest.raises(ImportError, match=r"needs glmstar 0\.7\.1 or later; 0\.6\.9"):
            _glmstar.load()

    def test_a_stand_in_glmnet_module_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        stand_in = types.ModuleType("glmnet")
        stand_in.__file__ = str(tmp_path / "glmnet" / "__init__.py")
        monkeypatch.setitem(sys.modules, "glmnet", stand_in)
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.7.1")

        class Distribution:
            files: ClassVar[list[str]] = ["glmnet/__init__.py"]

            def locate_file(self, entry: str) -> str:
                return str(tmp_path / "site-packages" / entry)

        monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Distribution())
        with pytest.raises(ImportError, match="is not glmstar's"):
            _glmstar.load()

    def test_the_version_parser(self) -> None:
        assert _glmstar._version_tuple("0.7.1") == (0, 7, 1)
        assert _glmstar._version_tuple("1.0rc2") == (1, 0)
        assert _glmstar._version_tuple("0.7.1.post2") == (0, 7, 1)
        assert _glmstar._version_tuple("0.10.0.dev3") == (0, 10, 0)
        assert _glmstar._below_floor("0.7.0") and _glmstar._below_floor("0.7.1rc1")
        assert _glmstar._below_floor("0.7.1.dev0")
        assert not _glmstar._below_floor("0.7.1") and not _glmstar._below_floor("0.7.2rc1")
        assert not _glmstar._below_floor("0.7.1.post2")

    def test_absent_class_is_declared_before_glmstar_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable() -> Any:
            raise AssertionError("load ran before the absent-class check")

        monkeypatch.setattr(_glmstar, "load", unreachable)
        x, a, _ = _binary(200)
        problem = l1._Problem(x, a.astype(np.int64), np.ones(a.size), 3)
        with pytest.raises(_glmstar.GlmstarFallback, match="absent_class"):
            _glmstar.starts(problem, np.array([0.01]))


def _builtin_starts(perturb: float) -> Callable[[l1._Problem, Any], list[Any]]:
    """An injected adapter that returns the built-in solutions, moved by ``perturb``."""

    solve = l1._solve

    def starts(problem: l1._Problem, penalties: Any) -> list[Any]:
        out = []
        intercept, coef = problem.null_start()
        # The built-in path solves each penalty warm from the one before (the strong rule reads
        # it), starting after the null gradient, the default grid's first value. The same
        # chain of solves gives the same bits.
        previous = float(problem.null_gradient())
        for penalty in penalties:
            intercept, coef, _ = solve(problem, intercept, coef, float(penalty), previous)
            out.append((intercept.copy(), coef + perturb * (coef != 0.0)))
            previous = float(penalty)
        return out

    return starts


class TestInjectedOutcomes:
    def test_a_declared_failure_falls_back_warns_and_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, a, w = _binary(800, weighted=True)
        grid = l1.default_penalty_grid(x, a, w)

        def fail(problem: Any, penalties: Any) -> Any:
            raise _glmstar.GlmstarFallback("nonconvergence")

        monkeypatch.setattr(_glmstar, "starts", fail)
        work = l1.L1Work()
        with pytest.warns(UserWarning, match=r"glmstar did not solve .*\(nonconvergence\)"):
            fits = l1.penalty_path(x, a, w, grid, solver="glmstar", work=work)
        assert (work.fallbacks, work.reasons, work.polished) == (1, ["nonconvergence"], 0)
        builtin = l1.penalty_path(x, a, w, grid)
        for ours, theirs in zip(fits, builtin, strict=True):
            np.testing.assert_array_equal(ours.coef, theirs.coef)

    def test_an_unexpected_error_is_raised_not_absorbed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, a, _ = _binary(400)

        def broken(problem: Any, penalties: Any) -> Any:
            raise AttributeError("LogNet has no attribute coefs_")

        monkeypatch.setattr(_glmstar, "starts", broken)
        with pytest.raises(AttributeError, match="coefs_"):
            l1.penalty_path(x, a, None, l1.default_penalty_grid(x, a, None), solver="glmstar")

    def test_exact_starts_are_kept_and_moved_starts_are_polished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, a, _ = _binary(800)
        grid = l1.default_penalty_grid(x, a, None)
        builtin = l1.penalty_path(x, a, None, grid)
        monkeypatch.setattr(_glmstar, "starts", _builtin_starts(0.0))
        exact = l1.L1Work()
        kept = l1.penalty_path(x, a, None, grid, solver="glmstar", work=exact)
        assert exact.polished == 0
        for ours, theirs in zip(kept, builtin, strict=True):
            np.testing.assert_array_equal(ours.coef, theirs.coef)
        # A nonzero witness: moved starts fail the check and are re-solved to tolerance.
        monkeypatch.setattr(_glmstar, "starts", _builtin_starts(1e-3))
        moved = l1.L1Work()
        polished = l1.penalty_path(x, a, None, grid, solver="glmstar", work=moved)
        active = sum(1 for fit in builtin if fit.penalty > 0.0 and fit.active)
        assert moved.polished == active > 0
        assert max(fit.kkt for fit in polished) <= l1.KKT_TOLERANCE

    def test_a_failed_polish_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x, a, _ = _binary(800)
        grid = l1.default_penalty_grid(x, a, None)
        monkeypatch.setattr(_glmstar, "starts", _builtin_starts(1e-3))
        solve = l1._solve
        calls = {"n": 0}

        def first_fails(*args: Any) -> Any:
            calls["n"] += 1
            intercept, coef, residual = solve(*args)
            return intercept, coef, np.inf if calls["n"] == 1 else residual

        monkeypatch.setattr(l1, "_solve", first_fails)
        work = l1.L1Work()
        with pytest.warns(UserWarning, match=r"\(polish\)"):
            fits = l1.penalty_path(x, a, None, grid, solver="glmstar", work=work)
        assert (work.fallbacks, work.reasons) == (1, ["polish"])
        assert max(fit.kkt for fit in fits) <= l1.KKT_TOLERANCE

    def test_the_fit_records_fallbacks_per_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(problem: Any, penalties: Any) -> Any:
            raise _glmstar.GlmstarFallback("short_path")

        monkeypatch.setattr(_glmstar, "starts", fail)
        settings = {"strategy": "lasso", "lambdas": [0.2, 0.05, 0.01, 0.0]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            accelerated = _fit(_frame(), l1_solver="glmstar", **settings)
        builtin = _fit(_frame(), **settings)
        selection = accelerated.extra["ctmle"]
        assert selection.l1_solver == "glmstar"
        assert len(selection.l1_fallbacks) == 6 and min(selection.l1_fallbacks) > 0
        assert set(selection.l1_fallback_reasons) == {"short_path"}
        printed = selection.summary()
        assert "l1_fallback" in {item.code for item in printed.notes}
        assert "glmstar did not solve" in str(printed)
        assert "l1_fallback" not in {item.code for item in builtin.extra["ctmle"].summary().notes}
        # Every path fell back, so the fit is the built-in one exactly.
        assert accelerated.estimate.psi == builtin.estimate.psi
        assert selection.selected == builtin.extra["ctmle"].selected


# ------------------------------------------------------------------ with glmstar

#: Positive bounds on the pre-polish residual and the polish share, from the measurement under
#: ``thresh=1e-18`` on these fixtures (binary and weighted: max 6.5e-11, 0 of 99 penalties
#: polished; three-class: max 1.6e-10), with room for library rounding. The polish share is the
#: deterministic work guard: an adapter that stopped early would polish most penalties.
BINARY_KKT = 1e-9
MULTINOMIAL_KKT = 2e-9
POLISH_SHARE = 0.05
#: A wrong mapping pushes the pre-polish residual to at least this.
MUTATION_KKT = 1e-6


@needs_glmstar
class TestTheAdapter:
    @pytest.mark.parametrize("weighted", [False, True], ids=["unit", "weighted"])
    def test_binary_starts_are_near_optimal(self, weighted: bool) -> None:
        x, a, w = _binary(weighted=weighted)
        problem = _problem(x, a, w)
        penalties = _positive(problem, l1.default_penalty_grid(x, a, w))
        starts = _glmstar.starts(problem, penalties)
        residuals = [problem.kkt(b0, b, v) for v, (b0, b) in zip(penalties, starts, strict=True)]
        assert max(residuals) <= BINARY_KKT
        share = np.mean(np.asarray(residuals) > l1.KKT_TOLERANCE)
        assert share <= POLISH_SHARE

    def test_three_class_starts_are_near_optimal(self) -> None:
        x, a, w = _three_class()
        problem = _problem(x, a, w)
        penalties = _positive(problem, l1.default_penalty_grid(x, a, w))
        assert _max_kkt(problem, penalties, _glmstar.starts(problem, penalties)) <= (
            MULTINOMIAL_KKT
        )

    @pytest.mark.parametrize(
        "mutation",
        ["unit_weights", "permuted_weights", "flipped_labels", "raw_columns", "penalty_times_n"],
    )
    def test_each_binary_mapping_mutation_is_caught(self, mutation: str) -> None:
        x, a, w = _binary(weighted=True)
        problem = _problem(x, a, w)
        penalties = _positive(problem, l1.default_penalty_grid(x, a, w))
        wrong = copy.copy(problem)
        given = penalties
        if mutation == "unit_weights":
            wrong.w = np.ones_like(problem.w)
        elif mutation == "permuted_weights":
            wrong.w = np.random.default_rng(0).permutation(problem.w)
        elif mutation == "flipped_labels":
            wrong.codes = 1 - problem.codes
        elif mutation == "raw_columns":
            wrong.z = np.asarray(x, dtype=float)[:, problem.used]
        else:
            given = penalties * problem.n
        starts = _glmstar.starts(wrong, given)
        assert _max_kkt(problem, penalties, starts) >= MUTATION_KKT

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            ("warn_and_truncate", "nonconvergence"),
            ("truncate", "short_path"),
            ("nan", "nonfinite"),
            ("fatal_code", "fatal"),
        ],
    )
    def test_the_adapter_names_each_declared_failure(
        self, monkeypatch: pytest.MonkeyPatch, change: str, expected: str
    ) -> None:
        """glmstar's own observables, injected after its fit, map to their declared reasons."""
        log_net = _glmstar.load()[0]
        fit = log_net.fit

        def altered(self: Any, *args: Any, **kwargs: Any) -> Any:
            out = fit(self, *args, **kwargs)
            if change == "warn_and_truncate":
                warnings.warn(
                    "Error code -10001: convergence for 5th lambda value not reached", stacklevel=2
                )
            if change in ("warn_and_truncate", "truncate"):
                self.coefs_, self.intercepts_ = self.coefs_[:4], self.intercepts_[:4]
            elif change == "nan":
                self.coefs_ = self.coefs_.copy()
                self.coefs_[3, 0] = np.nan
            else:
                self._fit["jerr"] = 8001
            return out

        monkeypatch.setattr(log_net, "fit", altered)
        x, a, w = _binary(800)
        problem = _problem(x, a, w)
        penalties = _positive(problem, l1.default_penalty_grid(x, a, w))
        with pytest.raises(_glmstar.GlmstarFallback) as caught:
            _glmstar.starts(problem, penalties)
        assert caught.value.reason == expected

    def test_swapped_multinomial_axes_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A layout change in glmstar's multinomial output is an error, never a start."""
        multi_class_net = _glmstar.load()[1]
        fit = multi_class_net.fit

        def swapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            out = fit(self, *args, **kwargs)
            self.coefs_ = np.ascontiguousarray(np.swapaxes(self.coefs_, 1, 2))
            return out

        monkeypatch.setattr(multi_class_net, "fit", swapped)
        x, a, w = _three_class(p=5)
        problem = _problem(x, a, w)
        penalties = _positive(problem, l1.default_penalty_grid(x, a, w))
        with pytest.raises(ValueError, match="may have changed its output layout"):
            _glmstar.starts(problem, penalties)

    def test_an_error_inside_glmstar_names_the_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_net = _glmstar.load()[0]

        def broken(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise TypeError("fit() got an unexpected keyword argument 'sample_weight'")

        monkeypatch.setattr(log_net, "fit", broken)
        x, a, w = _binary(200)
        problem = _problem(x, a, w)
        with pytest.raises(TypeError, match=r"glmstar \d.*cleverly\[glmstar\]"):
            _glmstar.starts(problem, _positive(problem, l1.default_penalty_grid(x, a, w)))

    def test_a_class_below_glmnets_floor_falls_back_or_agrees(self) -> None:
        """A class with a weighted share below 1e-5 never crashes the glmstar arm."""
        x, a, _ = _binary(3000)
        w = np.ones(a.size)
        w[a == 1] = 1e-7
        grid = l1.default_penalty_grid(x, a, w)[::10]
        work = l1.L1Work()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fast = _outcome(lambda: l1.penalty_path(x, a, w, grid, solver="glmstar", work=work))
        slow = _outcome(lambda: l1.penalty_path(x, a, w, grid))
        assert fast[0] == slow[0], (fast, slow)

    def test_runs_are_bitwise_equal(self) -> None:
        x, a, w = _binary(weighted=True)
        grid = l1.default_penalty_grid(x, a, w)
        runs = [l1.penalty_path(x, a, w, grid, solver="glmstar") for _ in range(3)]
        for later in runs[1:]:
            for first, other in zip(runs[0], later, strict=True):
                np.testing.assert_array_equal(first.coef, other.coef)
                np.testing.assert_array_equal(first.intercept, other.intercept)

    @pytest.mark.parametrize("data", [_binary, _three_class], ids=["binary", "three-class"])
    def test_paths_agree_with_the_builtin_solver(self, data: Callable[..., Any]) -> None:
        x, a, w = data()
        grid = l1.default_penalty_grid(x, a, w)
        work = l1.L1Work()
        fast = l1.penalty_path(x, a, w, grid, solver="glmstar", work=work)
        slow = l1.penalty_path(x, a, w, grid)
        assert work.fallbacks == 0
        assert max(fit.kkt for fit in fast) <= l1.KKT_TOLERANCE
        gap = max(
            np.max(np.abs(f.predict_proba(x) - s.predict_proba(x)))
            for f, s in zip(fast, slow, strict=True)
        )
        assert gap <= 1e-8

    def test_the_public_lasso_fit_agrees(self) -> None:
        frame = _frame(1500)
        builtin = _fit(frame, strategy="lasso")
        fast = _fit(frame, strategy="lasso", l1_solver="glmstar")
        selection = fast.extra["ctmle"]
        assert selection.selected == builtin.extra["ctmle"].selected
        assert selection.l1_fallbacks == (0,) * len(selection.l1_fallbacks)
        assert abs(fast.estimate.psi - builtin.estimate.psi) <= 1e-8

    def test_the_public_derivative_fit_agrees(self) -> None:
        frame = _frame(1500)
        builtin = _fit(frame, strategy="derivative")
        fast = _fit(frame, strategy="derivative", l1_solver="glmstar")
        record = fast.extra["ctmle"]
        assert record.l1_solver == "glmstar"
        assert record.penalties == builtin.extra["ctmle"].penalties
        assert abs(fast.estimate.psi - builtin.estimate.psi) <= 1e-8


def _edge_case(name: str) -> tuple[Any, ...]:
    """The fixtures of the plan's edge-case list: ``(x, a, w, grid, classes)``."""
    rng = np.random.default_rng(21)
    n, p = 1200, 5
    x = rng.normal(size=(n, p))
    a = rng.binomial(1, expit(0.7 * x[:, 0] - 0.5 * x[:, 1]))
    w: Any = None
    classes: Any = None
    if name == "near_collinear":
        x[:, 1] = x[:, 0] + 0.045 * rng.normal(size=n)  # correlation about 0.999
    elif name == "quasi_separated":
        a = (x[:, 0] > 0).astype(int)
        a[np.argsort(np.abs(x[:, 0]))[:6]] ^= 1
    elif name == "rare_arm":
        a = (rng.uniform(size=n) < expit(-4.0 + 0.5 * x[:, 0])).astype(int)
    elif name == "constant_column":
        x[:, 2] = 1.5
    elif name == "zero_weights":
        w = rng.lognormal(size=n)
        w[rng.choice(n, 100, replace=False)] = 0.0
    elif name == "absent_class":
        a = rng.integers(0, 2, size=n)
        classes = [0, 1, 2]
    elif name == "p_greater_than_n":
        x = rng.normal(size=(60, 80))
        a = rng.binomial(1, expit(x[:, 0]))
    grid = l1.default_penalty_grid(x, a, w)
    if name in ("p_greater_than_n", "quasi_separated_positive"):
        grid = grid[:-1]  # end at the default's smallest positive penalty (1e-2 at n < p)
    if name == "quasi_separated_positive":
        a = (x[:, 0] > 0).astype(int)
        a[np.argsort(np.abs(x[:, 0]))[:6]] ^= 1
        grid = l1.default_penalty_grid(x, a, w)[:-1]
    if name == "grid_ending_at_zero":
        grid = np.append(grid[::10], 0.0)
    return x, a, w, grid, classes


EDGE_CASES = [
    "near_collinear",
    "quasi_separated",
    "quasi_separated_positive",
    "rare_arm",
    "constant_column",
    "zero_weights",
    "p_greater_than_n",
    "grid_ending_at_zero",
]


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "fits", call()
    except Exception as error:
        return type(error).__name__, str(error)


@needs_glmstar
@pytest.mark.parametrize("name", EDGE_CASES)
def test_edge_cases_match_the_builtin_solver(name: str) -> None:
    """Same refusal or success as the built-in solver, no fallback, and agreeing paths."""
    x, a, w, grid, classes = _edge_case(name)
    work = l1.L1Work()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fast = _outcome(
            lambda: l1.penalty_path(x, a, w, grid, classes, solver="glmstar", work=work)
        )
    slow = _outcome(lambda: l1.penalty_path(x, a, w, grid, classes))
    assert fast[0] == slow[0], (fast, slow)
    if slow[0] != "fits":
        assert fast[1] == slow[1]
        return
    assert work.fallbacks == 0
    gap = max(
        np.max(np.abs(f.predict_proba(x) - s.predict_proba(x)))
        for f, s in zip(fast[1], slow[1], strict=True)
    )
    assert gap <= 1e-8, gap
    for f, s in zip(fast[1], slow[1], strict=True):
        flipped = _membership_flips(f, s, x, a, w, classes)
        assert all(margin <= DELTA_ACTIVE for margin in flipped), (s.penalty, flipped)


#: The active-set near-tie threshold the replay uses (k2c-replay/pilot/tolerances.json,
#: ``delta_active``), rounded up.
DELTA_ACTIVE = 2e-9


def _margins(slow: Any, x: Any, a: Any, w: Any, classes: Any) -> tuple[Any, Any]:
    """The built-in active-set margin of every penalized coordinate, and the used columns.

    The margin is ``lambda - |g_j|`` for a coordinate the built-in solution sets to 0, and
    ``|b_j| c_j`` for a nonzero one, in gradient units (plan tolerances step 1).
    """
    matrix = np.asarray(x, dtype=float)
    order = np.unique(a) if classes is None else np.asarray(classes)
    codes = np.searchsorted(order, a)
    weights = l1._normalized(np.ones(a.size) if w is None else np.asarray(w, dtype=float))
    used = slow.scale > 0.0
    z = (matrix[:, used] - slow.center[used]) / slow.scale[used]
    prob = slow.predict_proba(matrix)
    y = np.zeros_like(prob)
    y[np.arange(codes.size), codes] = 1.0
    gradient = z.T @ (weights[:, None] * (prob - y)) / codes.size
    scale = (z**2).T @ (weights[:, None] * prob * (1.0 - prob)) / codes.size
    bs = slow.coef[used]
    if prob.shape[1] == 2:
        gradient, scale, bs = gradient[:, 1:], scale[:, 1:], bs[:, 1:]
    margin = np.where(bs == 0.0, slow.penalty - np.abs(gradient), np.abs(bs) * scale)
    return margin, used


def _membership_flips(fast: Any, slow: Any, x: Any, a: Any, w: Any, classes: Any) -> list[float]:
    """The built-in margin of each coordinate whose zero/nonzero membership differs."""
    if slow.constant is not None or slow.penalty <= 0.0:
        return []
    margin, used = _margins(slow, x, a, w, classes)
    bs, bf = slow.coef[used], fast.coef[used]
    if bs.shape[1] == 2:
        bs, bf = bs[:, 1:], bf[:, 1:]
    flipped = (bs == 0.0) != (bf == 0.0)
    return [float(m) for m in margin[flipped]]


@needs_glmstar
def test_a_coordinate_at_the_boundary_is_a_counted_near_tie() -> None:
    """Positive control: at a penalty a hair below the null gradient, the entering coordinate
    sits at the active-set boundary. Its built-in margin is inside delta, so a membership flip
    there is a near tie, and no coordinate above delta flips."""
    x, a, w = _binary(2000)
    problem = _problem(x, a, w)
    boundary = problem.null_gradient() * (1.0 - 1e-12)
    grid = np.array([problem.null_gradient() * 1.5, boundary, boundary * 0.5])
    fast = l1.penalty_path(x, a, w, grid, solver="glmstar")
    slow = l1.penalty_path(x, a, w, grid)
    margins, _ = _margins(slow[1], x, a, w, None)
    assert np.min(np.abs(margins)) <= DELTA_ACTIVE  # the fixture is a near tie
    for f, sl in zip(fast, slow, strict=True):
        assert all(m <= DELTA_ACTIVE for m in _membership_flips(f, sl, x, a, w, None))


@needs_glmstar
def test_an_absent_class_falls_back_to_the_builtin_path() -> None:
    """glmstar cannot size a class with no training row; the path is the built-in one."""
    x, a, w, grid, classes = _edge_case("absent_class")
    work = l1.L1Work()
    with pytest.warns(UserWarning, match=r"\(absent_class\)"):
        fast = l1.penalty_path(x, a, w, grid, classes, solver="glmstar", work=work)
    assert work.reasons == ["absent_class"]
    for ours, theirs in zip(fast, l1.penalty_path(x, a, w, grid, classes), strict=True):
        np.testing.assert_array_equal(ours.coef, theirs.coef)
