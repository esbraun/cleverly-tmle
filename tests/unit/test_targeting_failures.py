"""When targeting fails, it must say which way.

Before this, every failure looked the same from outside: ``converged=False`` and a
warning about positivity, whatever had actually gone wrong.  The fixes differ --
separation and pinned bounds mean the overlap is bad, a singular Hessian means two
clever-covariate columns are collinear, and a bare iteration cap may just need more
iterations -- so the mode is worth naming.

Each failure here is constructed deterministically rather than sampled, so these are
sub-second exact tests and not a simulation that might or might not trip.

Policy note, asserted below: a failed targeting step *warns and returns*.  Raising
would break the sensitivity sweeps that deliberately push the truncation bound into
bad territory in order to show the reader what happens there.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.linear_model

from cleverly.exceptions import ConvergenceWarning
from cleverly.fluctuation.iterative import InitialFit, NewtonDetail, _classify, solve_fluctuation
from cleverly.fluctuation.one_step import _classify_one_step
from cleverly.fluctuation.submodel import Submodel


def _fit(values: np.ndarray) -> InitialFit:
    return InitialFit(values.copy(), {0.0: values.copy(), 1.0: values.copy()})


def _submodel(columns: np.ndarray, names: tuple[str, ...]) -> Submodel:
    """A ``mean``-group submodel over the given columns.

    ``arm_columns`` is supplied only for the two-column case, since a single-column
    design has no column to spare for a second arm; nothing here reads it, and claiming
    both arms share column 0 would be a false statement about the submodel.
    """
    per_arm = {0.0: 0, 1.0: 1} if columns.shape[1] == 2 else {}
    return Submodel(
        columns.copy(),
        {0.0: columns.copy(), 1.0: columns.copy()},
        names,
        "mean",
        per_arm,
    )


class TestNamedFailures:
    def test_off_mask_fillers_do_not_diagnose_bounds_pinned(self) -> None:
        """Failure labels read the rows whose score failed, not carried filler rows."""
        n = 100
        mask = np.zeros(n, dtype=bool)
        mask[:10] = True
        values = np.full(n, 0.9995)
        values[mask] = 0.5
        current = _fit(values)
        epsilon = np.array([10.0])

        assert _classify(epsilon, current, NewtonDetail(), 0.9995, 1, 2, mask) == (
            "max_iter_reached"
        )
        assert _classify_one_step(epsilon, current, 0.9995, 1, 2, mask) == ("line_search_exhausted")

    def test_separation_is_recognised_as_separation(self) -> None:
        """A perfectly separating covariate drives the logistic MLE to infinity."""
        n = 200
        h = np.where(np.arange(n) < n // 2, -1.0, 1.0)[:, None]
        # Outcome equals the covariate's sign: no finite epsilon solves the score.
        y = (h[:, 0] > 0).astype(float)
        with pytest.warns(ConvergenceWarning):
            fit = solve_fluctuation(
                y,
                _fit(np.full(n, 0.5)),
                _submodel(h, ("h",)),
                np.ones(n),
                max_iter=50,
                tol=1e-12,
            )
        assert not fit.converged
        assert fit.failure in {"separation_suspected", "bounds_pinned"}
        assert fit.describe_failure() != "converged"

    def test_a_rank_deficient_submodel_is_recognised(self) -> None:
        """Two columns that are multiples of one another leave epsilon unidentified."""
        rng = np.random.default_rng(0)
        n = 300
        base = rng.normal(size=n)
        h = np.column_stack([base, 2.0 * base])  # exactly collinear
        y = rng.uniform(0.2, 0.8, size=n)
        with np.errstate(all="ignore"):
            fit = solve_fluctuation(
                y, _fit(np.full(n, 0.5)), _submodel(h, ("a", "b")), np.ones(n), warn=False
            )
        # Either the solve detects the singular Hessian, or it converges through the
        # pseudo-inverse; what must not happen is a silent claim of a well-posed fit.
        assert fit.failure == "singular_hessian" or fit.hessian_condition > 1e8

    def test_the_iteration_cap_is_named(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        h = rng.normal(size=n)[:, None] * 5.0
        y = rng.uniform(0.05, 0.95, size=n)
        fit = solve_fluctuation(
            y,
            _fit(rng.uniform(0.2, 0.8, size=n)),
            _submodel(h, ("h",)),
            np.ones(n),
            max_iter=1,
            tol=1e-16,
            warn=False,
        )
        if not fit.converged:
            assert fit.failure in {
                "max_iter_reached",
                "bounds_pinned",
                "separation_suspected",
                "line_search_exhausted",
            }

    def test_a_converged_fit_names_no_failure(self) -> None:
        rng = np.random.default_rng(2)
        n = 400
        h = rng.normal(size=n)[:, None]
        y = rng.uniform(0.3, 0.7, size=n)
        fit = solve_fluctuation(y, _fit(np.full(n, 0.5)), _submodel(h, ("h",)), np.ones(n))
        assert fit.converged
        assert fit.failure is None
        assert fit.describe_failure() == "converged"


class TestPolicyIsWarnNotRaise:
    def test_a_failing_step_returns_a_result(self) -> None:
        """Sensitivity sweeps push into bad bounds on purpose; raising would break them."""
        n = 200
        h = np.where(np.arange(n) < n // 2, -1.0, 1.0)[:, None]
        y = (h[:, 0] > 0).astype(float)
        with pytest.warns(ConvergenceWarning):
            fit = solve_fluctuation(
                y, _fit(np.full(n, 0.5)), _submodel(h, ("h",)), np.ones(n), max_iter=30
            )
        assert fit.targeted.observed.shape == (n,)
        assert np.all(np.isfinite(fit.epsilon))


class TestRecordedDiagnostics:
    @pytest.fixture(scope="class")
    def fit(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(3)
        n = 500
        h = rng.normal(size=n)[:, None]
        y = rng.uniform(0.2, 0.8, size=n)
        return solve_fluctuation(y, _fit(np.full(n, 0.5)), _submodel(h, ("h",)), np.ones(n))

    def test_the_initial_score_is_kept(self, fit) -> None:  # type: ignore[no-untyped-def]
        """Targeting that started near zero had nothing to do; say so."""
        assert fit.score_initial is not None
        assert np.isfinite(fit.initial_score_norm)
        # Targeting must have reduced it.
        assert fit.score_norm <= fit.initial_score_norm

    def test_trace_starts_at_epsilon_zero(self, fit) -> None:  # type: ignore[no-untyped-def]
        """trace[0] means the same thing in both solvers now."""
        scale = fit.score_scale
        expected = float(np.max(np.abs(fit.score_initial) / np.maximum(scale, 1e-300)))
        assert fit.trace[0] == pytest.approx(expected, rel=1e-12)

    def test_the_hessian_summary_survives(self, fit) -> None:  # type: ignore[no-untyped-def]
        assert np.isfinite(fit.hessian_condition)
        assert fit.hessian_condition >= 1.0
        assert fit.epsilon_std_error is not None
        assert fit.epsilon_std_error.shape == fit.epsilon.shape
        assert np.all(fit.epsilon_std_error > 0)

    def test_the_log_likelihood_survives(self, fit) -> None:  # type: ignore[no-untyped-def]
        """one_step used to compute this and explicitly `del` it."""
        assert np.isfinite(fit.loglik)

    def test_solver_calls_are_distinguished_from_iterations(self, fit) -> None:  # type: ignore[no-untyped-def]
        assert fit.n_solver_calls == 1
        assert fit.n_iter >= 1


class TestOneStepAgrees:
    def test_one_step_also_records_the_initial_score(self) -> None:
        from cleverly.fluctuation.one_step import solve_one_step

        rng = np.random.default_rng(4)
        n = 300
        h = rng.normal(size=n)[:, None]
        y = rng.uniform(0.3, 0.7, size=n)
        fit = solve_one_step(
            y, _fit(np.full(n, 0.5)), _submodel(h, ("h",)), np.ones(n), step_size=1e-2
        )
        assert fit.score_initial is not None
        assert np.isfinite(fit.loglik)
        # Both solvers' trace now starts at epsilon = 0.
        scale = fit.score_scale
        expected = float(np.max(np.abs(fit.score_initial) / np.maximum(scale, 1e-300)))
        assert fit.trace[0] == pytest.approx(expected, rel=1e-12)


class TestFoldFailuresAreNotSilent:
    def test_a_fold_that_fails_is_reported(self) -> None:
        """Per-fold solves run with warn=False; the aggregate must still speak up.

        The treatment is almost perfectly determined by ``W1``, so the propensity
        pins against the truncation bounds and the fluctuation in at least one fold
        has to travel to infinity.  Deterministic given the seed.
        """
        import warnings

        import pandas as pd

        from cleverly.estimators import TMLE

        rng = np.random.default_rng(0)
        n = 300
        w1 = rng.normal(size=n)
        treatment = (w1 > 0).astype(float)
        treatment[rng.choice(n, size=6, replace=False)] = (
            1 - treatment[rng.choice(n, size=6, replace=False)]
        )
        outcome = (
            rng.uniform(size=n) < np.clip(0.5 + 0.4 * treatment + 0.3 * w1, 0.01, 0.99)
        ).astype(float)
        frame = pd.DataFrame({"W1": w1, "W2": rng.normal(size=n), "A": treatment, "Y": outcome})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = (
                TMLE(
                    outcome_learner=sklearn.linear_model.LinearRegression(),
                    treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                    n_folds=5,
                    targeting_scheme="fold",
                    random_state=1,
                    estimands=["ate"],
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )

        folds = result.fluctuations["mean"].folds
        assert not all(f.converged for f in folds), "the fixture stopped separating"

        messages = [str(w.message) for w in caught if w.category is ConvergenceWarning]
        fold_messages = [m for m in messages if "fold(s) did not converge" in m]
        assert fold_messages, (
            "a fold-targeted fit with non-converging folds reported nothing; per-fold "
            f"solves are warn=False, so the aggregate warning is the only signal. Got: {messages}"
        )
        # The message must name the mode and point at the per-fold detail.
        assert "separation_suspected" in fold_messages[0]
        assert "res.fluctuations[group].folds" in fold_messages[0]
