"""Shared fixtures and helpers.

The fast tier keeps runtime down with small explicit parametric SuperLearners wherever
the test is about estimator machinery rather than flexible learning.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import cleverly
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.ctmle import CTMLEStrategy
from cleverly.learners import SuperLearner

ROOT = Path(__file__).resolve().parents[1]


def _check_source_matches_checkout() -> None:
    """Refuse to run these tests against a *different* checkout's ``src/cleverly``.

    One editable install is shared by every ``git worktree`` of this repository, and it
    points at whichever tree it was installed from.  So ``pytest`` run inside a worktree
    collects that worktree's tests and imports the *other* tree's source, and the run is
    a verdict on neither branch.  It does not look like a configuration error: it looks
    like twenty ordinary failures, because tests that arrived with a branch are asserting
    against a package that does not have it yet.  That is what happened -- a docs-only
    change appeared to break ``LongitudinalData.from_frame``, and the traceback's
    ``..\\..\\..\\..\\Documents\\Projects`` prefix was the only tell.

    Failing here rather than in a test is deliberate.  The mismatch invalidates the whole
    run, so there is nothing a ``-k`` or ``-m`` selection should be able to leave behind.

    An installed (non-source) copy is left alone.  Only a sibling checkout is refused,
    which is the mistake with no other symptom.
    """
    imported = Path(cleverly.__file__).resolve().parent
    if imported == (ROOT / "src" / "cleverly").resolve():
        return
    if imported.parent.name != "src":
        return  # a wheel or a plain install, deliberately not a checkout
    raise pytest.UsageError(
        f"tests here belong to {ROOT}, but `import cleverly` resolves to {imported}. "
        f"The editable install points at another checkout of this repository, so this "
        f"run would report that tree's behaviour under this tree's tests. Either "
        f'reinstall from here (`uv pip install -e ".[dev]"`) or pin the import for '
        f'one run (`PYTHONPATH="{ROOT / "src"}"`).'
    )


def pytest_configure(config: pytest.Config) -> None:
    _check_source_matches_checkout()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep one study's evidence checks on one xdist worker.

    The evidence module reads and summarizes the same immutable artifacts in several tests.
    Grouping by registered study lets its process-local caches remove that duplicate work.
    A serial run ignores the marker, so grouping changes scheduling and not correctness.
    """
    for item in items:
        if item.path.name != "test_method_evidence.py" or not hasattr(item, "callspec"):
            continue
        study = item.callspec.params.get("study")
        if study is not None:
            item.add_marker(pytest.mark.xdist_group(study.slug))


class _AdaptiveMean(BaseEstimator):
    """The old parametric fallback candidate, expressed as one estimator object."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _AdaptiveMean:
        self.model_ = (
            DummyClassifier(strategy="prior")
            if np.unique(np.asarray(y)).size <= 2
            else DummyRegressor(strategy="mean")
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(X)


class _AdaptiveGLM(BaseEstimator):
    """Linear or effectively unpenalized logistic regression, selected from ``y``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _AdaptiveGLM:
        if np.unique(np.asarray(y)).size <= 2:
            self.model_ = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=1e6, max_iter=1000)),
                ]
            )
            fit_kwargs = {} if sample_weight is None else {"model__sample_weight": sample_weight}
            self.model_.fit(X, y, **fit_kwargs)
        else:
            self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        return self.model_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(X)


#: Estimator settings for the fast tier: parametric nuisances, few folds, seeded.
FAST_KWARGS: dict[str, Any] = {
    "outcome_learner": SuperLearner(
        library=[
            ("mean", _AdaptiveMean()),
            ("glm", _AdaptiveGLM()),
        ],
        task=None,
        n_folds=3,
        clip=(0.0, 1.0),
        random_state=None,
    ),
    "treatment_learner": SuperLearner(
        library=[
            ("mean", DummyClassifier(strategy="prior")),
            (
                "glm",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(C=1e6, max_iter=1000)),
                    ]
                ),
            ),
        ],
        task="classification",
        n_folds=3,
        clip=(0.0, 1.0),
        random_state=None,
    ),
    "n_folds": 5,
    "learner_folds": 3,
    "random_state": 0,
    "simultaneous": False,
}


#: Fit every nuisance on all the rows. The setting a test reaches for when its subject is
#: not cross-fitting: a refutation, a sensitivity analysis, a diagnostic, a serialization
#: round trip. A cross-fitted fit of a continuous outcome needs a declared ``q_bounds``,
#: and the Gaussian laws here have unbounded support, so a test that neither declares one
#: nor turns cross-fitting off is refused before its first learner.
IN_SAMPLE: dict[str, Any] = {"cross_fit": False}

#: The known support of the bounded laws below, for a fit that *is* about cross-fitting.
#: Pass it only with a bounded or binary outcome: :meth:`cleverly.estimators.TMLE._scaler`
#: refuses ``q_bounds`` on a binary outcome, and declaring a finite support for a Gaussian
#: law would state something the law does not satisfy.
BOUNDED: dict[str, Any] = {"q_bounds": (0.0, 1.0)}


def bounded_frame(n: int = 200, seed: int = 0, **kwargs: Any) -> Any:
    """A proportion-outcome law a cross-fitted fit can declare the support of.

    :func:`~cleverly.datasets.make_nonlinear_bounded` draws ``Y`` from a Beta law, so its
    support is ``(0, 1)`` by construction and :data:`BOUNDED` states a fact rather than an
    assumption. Use it where a Gaussian fixture would have been cross-fitted.

    Parameters
    ----------
    n : int, default=200
        Rows to draw.
    seed : int, default=0
        Seed of the draw.
    **kwargs : Any
        Passed to :func:`~cleverly.datasets.make_nonlinear_bounded`.

    Returns
    -------
    tuple
        The frame and its truth mapping, as the generator returns them.
    """
    from cleverly.datasets import make_nonlinear_bounded

    return make_nonlinear_bounded(n=n, seed=seed, **kwargs)


def fast_tmle(**overrides: Any) -> TMLE:
    """A quick, reproducible estimator for tests.

    :data:`FAST_KWARGS` sets no ``cross_fit`` key, so a fit takes :class:`cleverly.TMLE`'s
    own default of ``True`` at the ``n_folds=5`` declared above. A caller whose subject is
    not cross-fitting passes ``**IN_SAMPLE``, and one whose subject is it passes a bounded
    or binary law. Deciding the switch here would put every caller's subject on one line
    neither of them wrote.
    """
    return TMLE(**{**FAST_KWARGS, **overrides})


def linear_in_sample(**overrides: Any) -> dict[str, Any]:
    """Explicit linear learners, fitted on every row, with no simultaneous bands.

    The block a test passes when its subject is what a fit *reports* rather than how it
    learns: a refusal, a renamed column, a round trip. Each call builds a fresh pair of
    learners, so no fit can carry another test's fitted state.
    """
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        **overrides,
    }


def linear_ctmle(strategy: CTMLEStrategy, **overrides: Any) -> CTMLE:
    """A collaborative estimator on :func:`linear_in_sample`, for the given ``strategy``."""
    return CTMLE(strategy=strategy, **linear_in_sample(**overrides))


#: One configuration per selector path, for a claim that must hold on all three.  Three
#: selection folds keep each search in the fast tier.  The ordered path ranks by a
#: logistic preorder, so a test can pass any covariate set without an explicit ``ordering``.
SELECTOR_CONFIGS: dict[str, dict[str, Any]] = {
    "greedy": {"selection_folds": 3},
    "ordered": {"selection_folds": 3, "preorder": "logistic"},
    "discrete": {"selection_folds": 3, "candidates": ((), ("W1",))},
}


def mean_one_weights(n: int, spread: tuple[float, float] = (0.5, 1.5)) -> Any:
    """``n`` nonconstant observation weights whose mean is exactly one.

    :func:`cleverly.data.validate.check_weights` returns ``w * (n / w.sum())``, so a
    profile whose mean is already one survives the constructor unchanged. That is what a
    weighted fixture wants: the stored mass is the mass the test wrote down, and a test
    that compares a hand computation against the estimator is comparing the same numbers.

    The profile is nonconstant for the opposite reason. A weight factor multiplied by a
    constant one is invisible, so a formula check run on unit weights agrees with a copy
    of the code that never reads the mass. Widening ``spread`` about one keeps the mean and
    increases the margin a mutation control has over its gate.
    """
    low, high = spread
    if not np.isclose(0.5 * (low + high), 1.0):
        raise ValueError(f"spread {spread} is not centred on one")
    return np.linspace(low, high, n)


def unweight(monkeypatch: pytest.MonkeyPatch, cls: type, method: str) -> None:
    """Run one production method against unit weights, unchanged otherwise.

    This is the deliberate-mutation control the fixed-weight compositions rest on. A
    weighted fit whose components silently ignore the row mass reports a number that reads
    like a weighted answer, so each component gets a case that unweights that component
    alone and asserts the result moves.

    ``np.ones_like`` has mean one, so the normalisation
    :func:`cleverly.data.validate.check_weights` applies is a no-op and the only difference
    is the mass the patched method sees.

    The mass reaches a method one of two ways, and the helper reads the signature to tell
    which. A method that takes a ``data`` argument gets a replaced argument. A method that
    reads ``self.data`` gets the attribute swapped for the duration of the call, which is
    what ``_Selector`` needs: :meth:`_Selector.target` reads ``self.data.weights`` twice,
    and a patch that covered one read would leave the other weighted.

    Either way the swap reaches whatever the patched method calls, so a case is a *dynamic
    extent* and not one function body. Two cases nest when one patched method delegates to
    another, and a caller that relies on the cases being disjoint has to say so.
    """
    original = getattr(cls, method)
    signature = inspect.signature(original)

    if "data" in signature.parameters:

        def unweighted(self: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(self, *args, **kwargs)
            data = bound.arguments["data"]
            bound.arguments["data"] = replace(data, weights=np.ones_like(data.weights))
            return original(*bound.args, **bound.kwargs)
    else:

        def unweighted(self: Any, *args: Any, **kwargs: Any) -> Any:
            saved = self.data
            self.data = replace(saved, weights=np.ones_like(saved.weights))
            try:
                return original(self, *args, **kwargs)
            finally:
                self.data = saved

    monkeypatch.setattr(cls, method, unweighted)


def assert_scale_normalizes_away(
    build: Callable[[float], Any],
    scales: Sequence[float] = (1.0, 13.0),
) -> list[Any]:
    """Fit the same weighted design at several common weight scales, and say what that shows.

    ``build(scale)`` returns a fitted result whose weight column is ``scale`` times one
    fixed mean-one profile. :func:`cleverly.data.validate.check_weights` divides the column
    by its own mean, so every scale produces the *same stored array* to within floating
    point. The first assertion states that up front rather than leaving a reader to infer
    it: a cell-by-cell comparison of two surfaces built on identical numbers passes for any
    deterministic implementation, weighted or not, so it is not evidence about weighting.

    What the comparison can still show is on the other two assertions. ``weight_spec.scale``
    records the mean of the column as supplied, so the normalisation stays recoverable. And
    every reported estimate agrees across the scales, which fails for an implementation that
    reads the raw column instead of the normalised one.

    Returns the fitted results, in ``scales`` order, so a caller can go on to compare
    whatever the scale is meant to leave alone.
    """
    results = [build(float(scale)) for scale in scales]
    reference = results[0]
    for scale, result in zip(scales, results, strict=True):
        spec = result.data.weight_spec
        assert spec is not None, f"scale {scale} produced a result carrying no weight_spec"
        assert spec.scale == pytest.approx(scale), (
            f"weight_spec.scale is {spec.scale} for a column supplied at {scale}"
        )
        # Stated, not hidden: the stored mass is scale-free, so everything below runs on
        # one set of numbers and cannot separate a weighted fit from an unweighted one.
        np.testing.assert_allclose(
            result.data.weights, reference.data.weights, rtol=0.0, atol=1e-15
        )
        for alias, estimate in result.estimates.items():
            assert estimate.psi == pytest.approx(reference.estimates[alias].psi, abs=1e-12), (
                f"{alias} moved between weight scales {scales[0]} and {scale}; the fit is "
                f"reading the raw weight column rather than the normalised one"
            )
    return results


class OracleTreatment(BaseEstimator):
    """A treatment model that returns the data-generating propensity exactly.

    Used to isolate the estimator from nuisance-estimation error: with the truth
    plugged in, any remaining discrepancy is the estimator's own.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleTreatment:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        p = np.clip(np.asarray(self.dgp.propensity(np.asarray(X, dtype=float))), 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p, p])


class ConstantProbability(BaseEstimator):
    """A binary classifier that predicts ``P(label = 1) = p`` on every row.

    A test uses it where a fitted model would only come *close* to a known value: a
    mechanism pinned to a known wrong value, or an intermediate density pinned to its
    truth.  Every term the test computes from ``p`` is then exact.
    """

    def __init__(self, p: float) -> None:
        self.p = p

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantProbability:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        p = np.full(np.asarray(X, dtype=float).shape[0], self.p)
        return np.column_stack([1.0 - p, p])


class OracleOutcome(BaseEstimator):
    """An outcome model returning the true conditional mean given ``[A, W]``.

    Only valid for a binary outcome, where the estimator does not rescale ``Y`` and the
    true conditional mean is directly on the ``[0, 1]`` scale the fluctuation uses.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.outcome_mean(w, 1.0, None), dtype=float)
        zero = np.asarray(self.dgp.outcome_mean(w, 0.0, None), dtype=float)
        return np.clip(np.where(a == 1.0, one, zero), 1e-9, 1.0 - 1e-9)

    def predict_proba(self, X: Any) -> Any:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> Any:
        return self._mean(X)


class OracleOutcomeContinuous(BaseEstimator):
    """The true conditional mean for a *continuous* outcome, on the scaled scale.

    The estimator maps ``Y`` onto ``[0, 1]`` before fitting ``Qbar``, so an oracle for a
    continuous outcome cannot simply return the structural mean -- it has to apply the same
    affine map, and it does not know the map in advance because the scaler is derived from
    the observed outcome range.  Recovering it by regressing the scaled outcome the
    estimator hands over on the raw structural mean is exact: both are affine images of the
    same quantity, so the fit is a line through the points rather than an approximation.

    :class:`OracleOutcome` is the binary counterpart, where the scaler is the identity and
    none of this is needed.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOutcomeContinuous:
        design = np.asarray(X, dtype=float)
        raw = self._raw_mean(design)
        keep = np.isfinite(y)
        slope, intercept = np.polyfit(raw[keep], np.asarray(y)[keep], 1)
        self._slope, self._intercept = float(slope), float(intercept)
        return self

    def _raw_mean(self, design: Any) -> Any:
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.outcome_mean(w, 1.0, None), dtype=float)
        zero = np.asarray(self.dgp.outcome_mean(w, 0.0, None), dtype=float)
        return np.where(a == 1.0, one, zero)

    def predict(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        return np.clip(self._intercept + self._slope * self._raw_mean(design), 1e-9, 1 - 1e-9)


class OracleOutcomeUnit(BaseEstimator):
    """The true conditional mean for an outcome the estimator does not rescale.

    Between :class:`OracleOutcome` and :class:`OracleOutcomeContinuous`, and it exists
    because a *bounded* continuous law sits between the two cases they cover.  A
    proportion has a known support, so a fit declares ``q_bounds=(0, 1)`` and
    :meth:`~cleverly.estimators.TMLE._scaler` builds the identity scaler from it.  The
    structural mean is then already on the scale ``Qbar`` is fitted on, and returning it
    is *exact* -- where :class:`OracleOutcomeContinuous` has to recover an affine map it
    cannot know in advance, and pays one regression's worth of arithmetic for a map that
    is the identity here.

    Two guards, because the class is exact only under two conditions and each fails in a
    way the other cannot see.

    :meth:`fit` refuses a law whose conditional means leave ``(0, 1)``.  That is the
    condition on the *law*: a Gaussian outcome mean of 2.5 is not a unit-interval mean,
    and clipping it would return one law's oracle while the study sampled another.

    The condition on the *fit* is that the scaler is the identity, and a learner cannot
    see the scaler: it is handed the already-scaled outcome, and a scaler derived from the
    observed range maps that outcome into ``[1/12, 11/12]``, which is a subset of the
    values the identity produces.  No input distinguishes them.  A study therefore asserts
    it on the fitted result instead, through
    :func:`tests.studies.bounded_cv_laws.assert_unit_outcome_scaler`.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOutcomeUnit:
        del y, sample_weight
        means = self._mean(np.asarray(X, dtype=float))
        low, high = float(np.min(means)), float(np.max(means))
        if not (low > 0.0 and high < 1.0):
            raise ValueError(
                f"OracleOutcomeUnit returns the structural mean unchanged, which is the "
                f"fitted quantity only for an outcome the estimator does not rescale. This "
                f"law's conditional mean reaches [{low:.4g}, {high:.4g}], outside the open "
                f"unit interval. Use OracleOutcomeContinuous for an unbounded outcome."
            )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, design: Any) -> Any:
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.outcome_mean(w, 1.0, None), dtype=float)
        zero = np.asarray(self.dgp.outcome_mean(w, 0.0, None), dtype=float)
        return np.where(a == 1.0, one, zero)

    def predict(self, X: Any) -> Any:
        return self._mean(np.asarray(X, dtype=float))

    def predict_proba(self, X: Any) -> Any:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


class OracleMissingness(BaseEstimator):
    """A missingness model returning the true ``P(Delta = 1 | A, W)``.

    Its design matrix is ``[A, W]``, not ``W`` -- the mechanism is allowed to depend on
    treatment, and the estimator predicts it at both arms -- so this follows
    :class:`OracleOutcome`'s convention of reading the arm out of the first column, not
    :class:`OracleTreatment`'s.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleMissingness:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.missingness(w, 1.0), dtype=float)
        zero = np.asarray(self.dgp.missingness(w, 0.0), dtype=float)
        p = np.clip(np.where(a == 1.0, one, zero), 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p, p])


class OracleIntermediate(BaseEstimator):
    """An intermediate model returning the true ``P(Z = 1 | A, W)``.

    Fitted on :meth:`~cleverly.data.causal_data.CausalData.treatment_design` -- ``[A, W]``
    -- and predicted at both arms, so it follows :class:`OracleMissingness`'s convention of
    reading the arm out of the first column rather than :class:`OracleTreatment`'s.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleIntermediate:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.intermediate_mean(w, 1.0), dtype=float)
        zero = np.asarray(self.dgp.intermediate_mean(w, 0.0), dtype=float)
        p = np.clip(np.where(a == 1.0, one, zero), 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p, p])


class OracleDoseMechanism(BaseEstimator):
    """``P(Delta = 1 | A, W)`` or ``P(Z = 1 | A, W)`` when ``A`` is a *dose*.

    The arm-indexed oracles above read the arm out of the design's first column and then
    ask the law for one arm at a time, because there are only ever two of them.  A modified
    treatment policy assigns a different dose to every unit, so the design's first column is
    a vector of treatments rather than a label -- and the mechanism has to be answered at
    all of them at once.  That is the whole difference, and it is the reason this cannot be
    :class:`OracleMissingness` with a wider ``arms`` tuple.

    ``role`` selects which of the law's accessors to call, so one class serves both
    mechanisms; they have the same signature and differ only in what they mean.
    """

    def __init__(self, dgp: Any, role: str = "missingness") -> None:
        self.dgp = dgp
        self.role = role

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleDoseMechanism:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        dose, w = design[:, 0], design[:, 1:]
        answer = getattr(self.dgp, self.role)
        p = np.clip(np.asarray(answer(w, dose), dtype=float), 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p, p])


class OracleDoseOutcome(BaseEstimator):
    """``E[Y | A, W]`` -- or ``E[Y | A, Z, W]`` -- when ``A`` is a dose.

    Reads the level out of the design's last column when the fit carries an intermediate,
    exactly as :class:`OracleDirectOutcome` does, and passes ``None`` when it does not --
    which on the crossed shift law is a *different regression* rather than a default, since
    a fit without ``intermediate=`` learns ``Qbar`` with ``Z`` marginalised out.

    ``has_intermediate`` has to be declared rather than inferred: the width of the design
    is not enough to tell an intermediate column from a further covariate.
    """

    def __init__(self, dgp: Any, has_intermediate: bool = False) -> None:
        self.dgp = dgp
        self.has_intermediate = has_intermediate

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleDoseOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        dose = design[:, 0]
        if not self.has_intermediate:
            return np.clip(
                np.asarray(self.dgp.outcome_mean(design[:, 1:], dose, None), dtype=float),
                1e-9,
                1.0 - 1e-9,
            )
        w, z = design[:, 1:-1], design[:, -1]
        values = np.empty(design.shape[0], dtype=float)
        for level in (0.0, 1.0):
            rows = z == level
            if not rows.any():
                continue
            values[rows] = np.asarray(
                self.dgp.outcome_mean(w[rows], dose[rows], level), dtype=float
            )
        return np.clip(values, 1e-9, 1.0 - 1e-9)

    def predict_proba(self, X: Any) -> Any:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> Any:
        return self._mean(X)


class OracleDirectOutcome(BaseEstimator):
    """An outcome model returning the true ``E[Y | A, Z, W]`` for a direct-effect fit.

    A controlled-direct-effect fit trains the outcome model on ``[A, W, Z]`` and predicts
    it at ``[a, W, z]`` for a *fixed* level ``z``, so the design carries the intermediate
    in its last column -- which is why :class:`OracleOutcome`, which reads everything after
    the arm as covariates, cannot be reused here.  Reading ``z`` per row rather than from a
    stored level is deliberate: the same object serves the observed design and both
    counterfactual ones.

    Only valid for a binary outcome, for the reason :class:`OracleOutcome` gives.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleDirectOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w, z = design[:, 0], design[:, 1:-1], design[:, -1]
        values = np.empty(design.shape[0], dtype=float)
        for arm in (0.0, 1.0):
            for level in (0.0, 1.0):
                rows = (a == arm) & (z == level)
                if not rows.any():
                    continue
                values[rows] = np.asarray(self.dgp.outcome_mean(w[rows], arm, level), dtype=float)
        return np.clip(values, 1e-9, 1.0 - 1e-9)

    def predict_proba(self, X: Any) -> Any:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> Any:
        return self._mean(X)


def aipw_ate(
    y: Any,
    a: Any,
    propensity: Any,
    q_one: Any,
    q_zero: Any,
    *,
    delta: Any = None,
    missingness: Any = None,
) -> float:
    """The augmented IPW (one-step) ATE, computed independently of the estimator.

    A second implementation of the same estimating equation, written out longhand:
    with the same nuisance inputs, TMLE and AIPW solve the identical efficient score
    equation and must agree up to the second-order difference between a substitution
    estimator and a one-step correction.

    ``delta`` and ``missingness`` extend that cross-check to missing outcomes.  Pass the
    observed-outcome indicator and an ``(n, 2)`` array of ``P(Delta = 1 | A = a, W)``:
    the indicator multiplies the residual term and the arm's observation probability
    joins the propensity in its denominator.  ``y`` is then read only where ``delta`` is
    one, so it may be anything (zero, ``nan``) elsewhere.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    g = np.asarray(propensity, dtype=float)
    q1 = np.asarray(q_one, dtype=float)
    q0 = np.asarray(q_zero, dtype=float)
    if delta is None:
        d, pi0, pi1 = np.ones_like(y), np.ones_like(y), np.ones_like(y)
    else:
        d = np.asarray(delta, dtype=float).reshape(-1)
        pi = np.ones((y.shape[0], 2)) if missingness is None else np.asarray(missingness, float)
        pi0, pi1 = pi[:, 0], pi[:, 1]
    # Read Y only where it exists, so an unobserved NaN cannot propagate through the
    # multiply-by-zero that the Delta factor is.
    residual = np.where(d == 1.0, y, 0.0)
    contribution = (
        q1
        - q0
        + a * d / (g * pi1) * (residual - q1)
        - (1.0 - a) * d / ((1.0 - g) * pi0) * (residual - q0)
    )
    return float(np.mean(contribution))


def binary_means(*args: Any, **kwargs: Any) -> tuple[float, Any, float, Any]:
    """``(psi1, IC1, psi0, IC0)`` from :func:`counterfactual_means`' arm mapping.

    Most of the influence-curve tests are *about* the two-arm contrast -- the Gateaux
    derivative of the ATE, the second-order remainder, the ``IC_ate == IC_ey1 - IC_ey0``
    identity -- and read better naming the two arms than indexing a mapping twice. The
    arm-general shape is exercised directly by the multi-arm tests rather than by making
    every binary test spell it out.
    """
    from cleverly.inference.influence import counterfactual_means

    means = counterfactual_means(*args, **kwargs)
    one, zero = means[1.0], means[0.0]
    return one.psi, one.influence_curve, zero.psi, zero.influence_curve


def binary_mean_parts(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """``(parts_one, parts_zero)`` from :func:`counterfactual_mean_parts`' arm mapping."""
    from cleverly.inference.influence import counterfactual_mean_parts

    parts = counterfactual_mean_parts(*args, **kwargs)
    return parts[1.0], parts[0.0]


def assert_estimate_coherent(
    estimate: Any,
    *,
    variance_from_curve: bool = True,
) -> None:
    r"""Everything a :class:`~cleverly.inference.ParameterEstimate` owes itself.

    Six fields of a reported estimate are *derived* -- ``std_error`` from ``variance``,
    ``ci`` and ``pvalue`` from ``psi`` and ``std_error``, ``score`` from the curve -- so
    each of them can be checked against the state the result carries, in the same process,
    with no simulation and no truth to compare to. The evidence rule is recorded in
    ``docs/architecture-invariants.md``. This is the check that settled the
    uncentred-curve question after two revisions had filed it behind
    a cross-language fixture: "recompute the recorded number from the returned state in the
    same process", thirty lines and one fit.  This is that check, written once so that
    every test producing an estimate can spend a line on it.

    What it pins, in the order the mistakes actually happen:

    * **the variance is the variance of the curve that was returned.**  Everything
      downstream -- the delta method, the cluster-robust variance, the simultaneous bands,
      the score diagnostic -- reads the curve rather than the variance, so a curve that has
      drifted from its variance makes those four disagree with the reported interval while
      each stays internally consistent.  Pass ``variance_from_curve=False`` for a
      ``targeting_scheme="fold"`` fit, whose variance is
      :func:`~cleverly.inference.cluster.cross_validated_variance` by construction --
      averaged over validation folds rather than taken over the pooled curve -- and for a
      clustered fit, whose cluster assignment the estimate does not carry;
    * **the interval and the p-value agree about the null.**  A Wald interval excludes the
      null exactly when ``p < alpha``, and the two are computed by different code down
      different branches for a ratio, where the interval is built on the log scale and
      exponentiated.  A scale confusion in either shows up here and in almost nothing else;
    * **NaN propagates rather than resolving to a number.**  Where the variance is not
      usable, ``std_error``, both interval endpoints and the p-value are all ``nan`` -- not
      zero, not a placeholder, and not an interval of width zero that reads as a certainty.

    It deliberately does *not* assert that ``score`` is small: whether targeting solved the
    score equation is a claim about the fit, which :mod:`cleverly.validation.score` makes
    with a tolerance, and folding it in here would make a diagnostic into an invariant.
    """
    import math
    import typing

    name = estimate.name
    ic = np.asarray(estimate.influence_curve, dtype=float)
    assert ic.ndim == 1, f"{name}: influence curve has shape {ic.shape}"
    assert ic.shape[0] == estimate.n, (
        f"{name}: curve has {ic.shape[0]} rows for an estimate reporting n={estimate.n}"
    )
    assert 0 < estimate.alpha < 1, f"{name}: alpha={estimate.alpha}"
    # Read off the Literal rather than written down, so a scale added there without a
    # branch below fails here rather than falling silently into the difference case.
    from cleverly.inference.influence import Scale

    assert estimate.scale in typing.get_args(Scale), f"{name}: scale={estimate.scale!r}"
    if estimate.scale == "ratio":
        assert estimate.log_psi is not None, f"{name}: a ratio with no log_psi"

    se = estimate.std_error
    low, high = estimate.ci
    pvalue = estimate.pvalue

    if not math.isfinite(se) or se <= 0.0:
        # The diff-diff rule, and the reason it is one assertion rather than three: an
        # unusable standard error that leaves a finite interval behind reads as a precise
        # answer rather than as a missing one.
        assert math.isnan(low) and math.isnan(high) and math.isnan(pvalue), (
            f"{name}: std_error={se} but ci={(low, high)} and p={pvalue}; "
            f"a non-usable variance has to reach every field it feeds"
        )
        return

    if variance_from_curve:
        from cleverly.inference.cluster import influence_variance

        recomputed = influence_variance(ic)
        assert estimate.variance == pytest.approx(recomputed, rel=1e-12), (
            f"{name}: reported variance {estimate.variance!r} is not the variance of the "
            f"curve it returned ({recomputed!r}); everything downstream reads the curve"
        )

    assert low < high, f"{name}: interval {(low, high)} is not ordered"
    centre = math.exp(estimate.log_psi) if estimate.scale == "ratio" else estimate.psi
    assert low <= centre <= high, f"{name}: interval {(low, high)} does not contain {centre}"

    null = 1.0 if estimate.scale == "ratio" else 0.0
    excludes_null = not (low <= null <= high)
    assert excludes_null == (pvalue < estimate.alpha), (
        f"{name}: interval {(low, high)} {'excludes' if excludes_null else 'contains'} the "
        f"null at {null} while p={pvalue} against alpha={estimate.alpha}. The interval and "
        f"the p-value are the same statement twice; on a ratio they are computed down "
        f"different branches, and this is where a scale confusion in either surfaces"
    )
    assert 0.0 <= pvalue <= 1.0, f"{name}: p={pvalue}"
