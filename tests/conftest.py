"""Shared fixtures and helpers.

The fast tier keeps runtime down by using parametric nuisance learners
(``library="glm"``) wherever the test is about the estimator's machinery rather than
about the Super Learner.  Tests that specifically exercise flexible learning say so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import cleverly
from cleverly import TMLE
from tests.doc_sections import ROOT


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("documentation execution")
    group.addoption(
        "--doc-section",
        action="append",
        default=None,
        metavar="ID",
        help="run one documentation section (repeatable)",
    )
    group.addoption(
        "--doc-changed-from",
        metavar="GIT_REV",
        help="run documentation sections affected since this git revision",
    )


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
    from tests.doc_sections import all_sections, git_changes, select_sections

    _check_source_matches_checkout()
    sections = all_sections()
    known = {section.section_id for section in sections}
    requested = config.getoption("--doc-section")
    base = config.getoption("--doc-changed-from")
    if requested:
        unknown = set(requested) - known
        if unknown:
            raise pytest.UsageError(f"unknown --doc-section value(s): {sorted(unknown)}")
    if base:
        selected = select_sections(git_changes(base), sections)
        config.option.doc_section = sorted(set(requested or ()) | selected)
    config._doc_selection_active = requested is not None or base is not None  # type: ignore[attr-defined]


#: Estimator settings for the fast tier: parametric nuisances, few folds, seeded.
FAST_KWARGS: dict[str, Any] = {
    "outcome_learner": "glm",
    "treatment_learner": "glm",
    "n_folds": 5,
    "learner_folds": 3,
    "random_state": 0,
    "simultaneous": False,
}


def fast_tmle(**overrides: Any) -> TMLE:
    """A quick, reproducible estimator for tests."""
    return TMLE(**{**FAST_KWARGS, **overrides})


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
