"""The quasibinomial study learner: separation keeps the last iterate, convergence is unchanged.

Study A of ``point-treatment-survival`` stopped on replication 189 of its ``competing``
scenario.  At one node an arm's ``W1 = 0`` cell had no event of one cause, so the
coefficient of ``W1`` diverged while the deviance settled.  R's ``glm`` returns such a fit with
a warning.  The learner raised.  It now keeps the last iterate with
``QuasiBinomialSeparationWarning`` when the deviance has converged by R's rule, and it still
raises when the deviance has not.

Complete separation of a binary target is a second case.  The deviance tends to zero there
and never meets R's relative rule, so the learner raised on 8% of the cross-fitted LTMLE draws of
``few_cluster_crossfit_properties`` at 4 clusters.  It now detects the case on the design and
the target, by a linear program for a strict separating hyperplane
(:func:`~tests.studies.fractional_glm.completely_separated`), and keeps the last iterate with
the same warning.  The draws that raised are must-not-raise fixtures below, and so are 120
random separated designs.

The learner is shared by many registered studies, so a converging fit must take the same
iterates as before.  ``_previous_fit`` is the learner as it was, and the bit-identity test
compares the two on converging fits.

A third case, opt-in through ``numerical_separation=True``, is a target binary to within
``1e-4``: a pseudo-outcome carried back from a separated later fit. One such fit from the
``boundary-node-ltmle`` pilot is a fixture below. It raises by default and fits with the flag,
and the flag leaves converging fits bit-identical.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from tests.studies import fractional_glm
from tests.studies.fractional_glm import (
    QuasiBinomialGLM,
    QuasiBinomialSeparationWarning,
    completely_separated,
    deviance,
    numerically_separated,
)


def _previous_fit(X: Any, y: Any, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """The learner before the separation branch, verbatim, returning its coefficients."""
    matrix = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    weights = np.ones_like(target)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    mean = float(np.average(target, weights=weights))
    coefficient = np.zeros(design.shape[1], dtype=float)
    coefficient[0] = math.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))
    for _ in range(max_iter):
        fitted = expit(design @ coefficient)
        variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        working = design @ coefficient + (target - fitted) / variance
        root_weight = np.sqrt(weights * variance)
        updated = np.linalg.lstsq(design * root_weight[:, None], working * root_weight, rcond=None)[
            0
        ]
        if np.max(np.abs(updated - coefficient)) <= tol:
            coefficient = updated
            break
        coefficient = updated
    else:
        raise RuntimeError("quasibinomial IRLS did not converge")
    return coefficient


def _separated(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A binary ``W1`` whose ``W1 = 0`` cell has no event: quasi-complete separation."""
    rng = np.random.default_rng(seed)
    w1 = rng.integers(0, 2, n).astype(float)
    w2 = rng.integers(0, 4, n).astype(float)
    y = np.where(w1 == 1.0, rng.binomial(1, 0.05, n), 0).astype(float)
    return np.column_stack([w1, w2]), y


@pytest.mark.parametrize("seed", range(20))
def test_a_converging_fit_takes_the_same_iterates(seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3))
    y = rng.binomial(1, expit(0.3 + X @ np.array([0.5, -0.4, 0.2]))).astype(float)
    if seed % 2:
        y = np.clip(y * 0.7 + rng.uniform(0, 0.3, size=y.size), 0.0, 1.0)  # fractional target
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = QuasiBinomialGLM().fit(X, y)
    expected = _previous_fit(X, y)
    assert np.array_equal(model.intercept_, expected[:1])
    assert np.array_equal(model.coef_[0], expected[1:])


def test_separation_keeps_the_last_iterate_with_a_warning() -> None:
    X, y = _separated()
    with pytest.raises(RuntimeError, match="did not converge"):
        _previous_fit(X, y)
    with pytest.warns(QuasiBinomialSeparationWarning, match="quasi-complete separation"):
        model = QuasiBinomialGLM().fit(X, y)
    empty = model.predict(X[X[:, 0] == 0.0])
    assert np.all(empty < 1e-8)
    events = model.predict(X[X[:, 0] == 1.0])
    assert abs(float(events.mean()) - float(y[X[:, 0] == 1.0].mean())) < 1e-6


def test_a_deviance_that_has_not_settled_still_raises() -> None:
    """Mutation control: a fallback that accepted every unconverged fit would pass this."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = rng.binomial(1, expit(1.0 + 2.0 * X[:, 0])).astype(float)
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM(max_iter=2).fit(X, y)


def test_the_failing_study_replication_now_fits() -> None:
    """Replication 189 of study A's ``competing`` scenario, the one that stopped the run."""
    from tests.studies import canonical_point_survival as study

    frame, _ = study.draw_scenario("competing", study.STUDY.n, 189)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = study.fit("competing", frame)
    assert any(issubclass(w.category, QuasiBinomialSeparationWarning) for w in caught)
    for name in study.ESTIMANDS["competing"]:
        assert np.isfinite(result[name].psi)
        assert np.isfinite(result[name].std_error)


# ------------------------------------------------------------------ complete separation

#: The cross-fitted LTMLE draws of ``few_cluster_crossfit_properties`` (law ``equal40``, 4
#: clusters) on which the learner raised at complete separation before it detected the case:
#: the first 12 of the seeds ``9_204_000 + r``.
SEPARATED_DRAW_SEEDS = (
    9_204_012,
    9_204_031,
    9_204_085,
    9_204_094,
    9_204_097,
    9_204_108,
    9_204_121,
    9_204_137,
    9_204_144,
    9_204_152,
    9_204_157,
    9_204_163,
)


def _completely_separated_design(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``Y = 1{x > 0}``, with a quarter of the rows on each side at distance ``0.005``.

    The rows at the margin are many, so the deviance of the last iterate is a sum of many
    small terms: above ``1e-6`` at 320 rows.  That lets the design tell direct detection from
    an absolute deviance floor.
    """
    quarter = n // 4
    side = np.concatenate([np.full(quarter, 0.005), np.linspace(0.005, 1.005, quarter)])
    x = np.concatenate([-side, side])[:, None]
    return x, (x[:, 0] > 0.0).astype(float)


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _last_iterate(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The coefficients after 100 unguarded IRLS steps, for the mutated rules to read."""
    mean = float(np.clip(target.mean(), 1e-8, 1.0 - 1e-8))
    coefficient = np.zeros(design.shape[1])
    coefficient[0] = math.log(mean / (1.0 - mean))
    for _ in range(100):
        fitted = expit(design @ coefficient)
        variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        working = design @ coefficient + (target - fitted) / variance
        root = np.sqrt(variance)
        coefficient = np.linalg.lstsq(design * root[:, None], working * root, rcond=None)[0]
    return coefficient


def _fits_without_raising(seeds: tuple[int, ...]) -> list[int]:
    """The seeds whose study fit raised ``RuntimeError``."""
    from tests.studies import few_cluster_crossfit_properties as study

    raised = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in seeds:
            try:
                estimate, _ = study.fit_drawn(study.draw("equal40", 4, seed))
            except RuntimeError:
                raised.append(seed)
            else:
                assert np.isfinite(estimate.psi)
    return raised


@pytest.mark.parametrize("n", [40, 320])
def test_complete_separation_keeps_the_last_iterate(n: int) -> None:
    X, y = _completely_separated_design(n)
    with pytest.raises(RuntimeError, match="did not converge"):
        _previous_fit(X, y)
    with pytest.warns(QuasiBinomialSeparationWarning, match="complete separation"):
        model = QuasiBinomialGLM().fit(X, y)
    fitted = model.predict(X)
    assert np.all(np.abs(fitted - y) < 1e-4)
    # The rows at the margin are near their class, not at it: a nonzero witness that the fit
    # kept an iterate rather than returning the labels.
    assert float(np.max(np.abs(fitted - y))) > 0.0


@pytest.mark.parametrize("n", [40, 320, 2000])
def test_random_separated_designs_never_raise(n: int) -> None:
    """A random hyperplane splits Gaussian rows: some rows lie close to it at every size.

    A rule that read how close the last iterate came to the labels missed 29 of 40 of these
    designs at 320 rows, because IRLS had not yet brought the rows near the hyperplane within
    its tolerance.
    """
    for seed in range(40):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(float)
        with pytest.warns(QuasiBinomialSeparationWarning, match="complete separation"):
            QuasiBinomialGLM().fit(X, y)


def test_complete_separation_on_the_study_draws_that_raised() -> None:
    """Review fixtures: every draw that raised at complete separation now fits."""
    assert _fits_without_raising(SEPARATED_DRAW_SEEDS) == []


def test_without_detection_the_study_draws_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: the study draws above are witnesses of the separation branch."""
    monkeypatch.setattr(fractional_glm, "completely_separated", lambda *args: False)
    assert _fits_without_raising(SEPARATED_DRAW_SEEDS) == list(SEPARATED_DRAW_SEEDS)


def test_a_rule_relative_to_the_null_deviance_misses_study_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: separation read as deviance at most ``1e-8`` of the null deviance."""

    def relative(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> bool:
        fitted = expit(design @ _last_iterate(design, target))
        null = np.full_like(target, np.average(target, weights=weights))
        return deviance(target, fitted, weights) <= 1e-8 * deviance(target, null, weights)

    monkeypatch.setattr(fractional_glm, "completely_separated", relative)
    assert len(_fits_without_raising(SEPARATED_DRAW_SEEDS)) >= 1


def test_an_absolute_deviance_floor_misses_the_large_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: separation read as a deviance of at most ``1e-6``."""

    def floor(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> bool:
        fitted = expit(design @ _last_iterate(design, target))
        return deviance(target, fitted, weights) <= 1e-6

    monkeypatch.setattr(fractional_glm, "completely_separated", floor)
    X, y = _completely_separated_design(320)
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM().fit(X, y)


def test_a_fractional_target_is_never_separated() -> None:
    """Values 0.001 and 0.999 split by a threshold: a finite fit, not the separation branch."""
    X, labels = _completely_separated_design(40)
    target = np.where(labels == 1.0, 0.999, 0.001)
    weights = np.ones_like(target)
    assert not completely_separated(_with_intercept(X), target, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = QuasiBinomialGLM().fit(X, target)
    assert np.all(np.isfinite(model.coef_))


def test_a_zero_weight_row_does_not_count_toward_separation() -> None:
    """The third row sits on the wrong side of every separator of the first two."""
    design = _with_intercept(np.array([[-1.0], [1.0], [2.0]]))
    target = np.array([0.0, 1.0, 0.0])
    assert completely_separated(design, target, np.array([1.0, 1.0, 0.0]))
    assert not completely_separated(design, target, np.ones(3))


def test_quasi_separation_is_not_complete_separation() -> None:
    """A row on the hyperplane: no strict separator exists, so the deviance rule decides.

    ``_separated`` puts every ``W1 = 0`` row at ``Y = 0`` while the ``W1 = 1`` cell holds both
    classes, which no hyperplane splits strictly.  Its fit keeps the quasi-complete message.
    """
    X, y = _separated()
    assert not completely_separated(_with_intercept(X), y, np.ones_like(y))


def test_a_deviance_that_rises_is_not_read_as_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: a separated design whose last step raised the deviance still raises.

    The steps alternate between a separating and a reversed coefficient and end on the
    reversed one.  Dropping the clause that the deviance fell would keep that iterate.
    """
    X, y = _completely_separated_design(40)
    steps = iter(np.array([[0.0, sign * 2.0] for sign in (1, -1) * 100]))

    def oscillate(*args: Any, **kwargs: Any) -> tuple[np.ndarray, None, None, None]:
        return next(steps), None, None, None

    monkeypatch.setattr(fractional_glm.np.linalg, "lstsq", oscillate)
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM().fit(X, y)


#: One of the four pseudo-outcome regressions that raised in the pilot's
#: ``interior_n200_crossfit`` cell, on its one covariate. Every value is within ``5e-8`` of 0 or 1, and the
#: one value near 1 sits at the smallest covariate. Unlisted targets are exactly 0.
_NUMERICAL_X = (
    -0.195934,
    1.270248,
    1.193440,
    0.504915,
    -0.283032,
    0.148482,
    0.787413,
    -1.278158,
    -1.475866,
    -0.470222,
    -0.185183,
    -0.063796,
    -0.747503,
    -0.153576,
    1.503990,
    -0.899555,
    1.420630,
    -1.784143,
    -1.281221,
    -0.787614,
    -1.846442,
    -0.822189,
    -0.067870,
    -1.780764,
    -0.430177,
    -0.150682,
    -0.967302,
    -1.581684,
    -0.658924,
    -1.318144,
    -0.325347,
    -0.987794,
    0.696619,
    0.399445,
    -1.261857,
    0.564534,
    -0.424567,
    0.707594,
)
_NUMERICAL_Y_NONZERO = {
    7: 1.2039173034120965e-127,
    8: 6.135937585849431e-81,
    12: 5.211938783598094e-253,
    15: 4.347967905069976e-217,
    17: 4.1329897223318884e-08,
    18: 6.370764678407521e-127,
    19: 1.5591146484510947e-243,
    20: 0.9999999536409566,
    21: 2.296990583384628e-235,
    23: 6.574974907546119e-09,
    26: 4.395597301813421e-201,
    27: 6.117070808897285e-56,
    28: 6.178401561166117e-274,
    29: 3.365172061530106e-118,
    31: 3.049152644507041e-196,
    34: 1.696460336471902e-131,
}


def _numerically_separated_fit() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(_NUMERICAL_X).reshape(-1, 1)
    y = np.zeros(len(_NUMERICAL_X))
    for row, value in _NUMERICAL_Y_NONZERO.items():
        y[row] = value
    return X, y


def test_a_numerically_binary_separated_target_raises_by_default() -> None:
    X, y = _numerically_separated_fit()
    assert not completely_separated(_with_intercept(X), y, np.ones_like(y))
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM().fit(X, y)


def test_numerical_separation_keeps_the_last_iterate() -> None:
    X, y = _numerically_separated_fit()
    assert numerically_separated(_with_intercept(X), y, np.ones_like(y))
    with pytest.warns(QuasiBinomialSeparationWarning, match="complete separation"):
        model = QuasiBinomialGLM(numerical_separation=True).fit(X, y)
    fitted = model.predict(X)
    assert np.max(np.abs(fitted - np.round(y))) < 1e-6


def test_numerical_separation_rests_on_its_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: without either detector the flagged fit raises again."""
    monkeypatch.setattr(fractional_glm, "numerically_separated", lambda *args: False)
    real = fractional_glm.completely_separated
    X, y = _numerically_separated_fit()
    rounded = np.round(y)
    monkeypatch.setattr(
        fractional_glm,
        "completely_separated",
        lambda design, target, weights: (
            False if np.array_equal(target, rounded) else real(design, target, weights)
        ),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM(numerical_separation=True).fit(X, y)


def test_a_target_off_binary_is_not_numerically_separated() -> None:
    """A value ``2e-4`` from 0 exceeds the tolerance, so the rounded design is not read."""
    X, y = _numerically_separated_fit()
    y[17] = 2e-4
    assert not numerically_separated(_with_intercept(X), y, np.ones_like(y))


@pytest.mark.parametrize("seed", range(10))
def test_the_numerical_flag_leaves_a_converging_fit_unchanged(seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3))
    y = rng.binomial(1, expit(0.3 + X @ np.array([0.5, -0.4, 0.2]))).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = QuasiBinomialGLM(numerical_separation=True).fit(X, y)
    expected = _previous_fit(X, y)
    assert np.array_equal(model.intercept_, expected[:1])
    assert np.array_equal(model.coef_[0], expected[1:])


#: A pseudo-outcome regression of property draw 130 of ``interior_n200_crossfit``, which
#: stopped the first declared run: separated once rounded, with one value ``4.2e-4`` from 1,
#: beyond :data:`~tests.studies.fractional_glm.NUMERICAL_BINARY_TOLERANCE`. Its maximum is finite
#: and distant, so IRLS needs about 200 steps. Values below ``1e-20`` are written as 0.
_EXTENDED_X = (
    -0.944388,
    0.163901,
    -0.208937,
    -0.159121,
    0.712430,
    0.110891,
    2.321552,
    -0.312164,
    -0.985751,
    -0.921071,
    2.729214,
    2.069248,
    0.357537,
    1.258227,
    -0.392028,
    0.407077,
    -0.442837,
    1.016585,
    -0.529769,
    0.767256,
    0.107041,
    0.923500,
    -1.061007,
    0.320218,
    1.042800,
    0.128777,
    0.603791,
    -0.209603,
    1.342066,
    1.258558,
    0.744884,
    -0.757811,
    -0.409675,
    -0.763399,
    -0.730881,
    -0.340801,
    1.419281,
    0.671333,
    -0.577598,
    -0.311114,
    0.879252,
    -1.620806,
    2.286791,
    -0.576001,
    -1.477433,
    -1.645567,
    0.833955,
    -1.675981,
    -1.161935,
    -1.323723,
    -1.146804,
    -0.969337,
    0.577222,
    2.425375,
    1.458983,
    0.851693,
    0.809495,
    0.376800,
    1.402052,
    0.433613,
    0.906432,
    0.337261,
    -0.054296,
    0.236460,
    -0.886785,
    0.291616,
    0.991522,
    1.354027,
    -0.540610,
    0.096669,
    0.986682,
    -1.327845,
    -0.776812,
    0.930688,
    -1.040652,
    0.472550,
    -0.271687,
    -0.210933,
    1.006554,
    1.022168,
    1.189479,
)
_EXTENDED_Y_NONZERO = {
    6: 1.0,
    10: 1.0,
    11: 1.0,
    28: 1.8225960584984293e-10,
    36: 0.9999999273842101,
    42: 1.0,
    53: 1.0,
    54: 0.9999999999999998,
    58: 0.9995763609261213,
    67: 7.501014435182158e-08,
}


def _extended_fit() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(_EXTENDED_X).reshape(-1, 1)
    y = np.zeros(len(_EXTENDED_X))
    for row, value in _EXTENDED_Y_NONZERO.items():
        y[row] = value
    return X, y


def test_a_target_separated_once_rounded_continues_and_fits() -> None:
    """The flag continues IRLS where the default raises, and reaches the target."""
    X, y = _extended_fit()
    assert not numerically_separated(_with_intercept(X), y, np.ones_like(y))
    assert completely_separated(_with_intercept(X), np.round(y), np.ones_like(y))
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM().fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QuasiBinomialGLM(numerical_separation=True).fit(X, y)
    assert np.max(np.abs(model.predict(X) - y)) < 1e-8


def test_the_continuation_rests_on_the_rounded_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: without the rounded-separation check the flagged fit raises.

    It raises the recordable error.
    """
    real = fractional_glm.completely_separated
    X, y = _extended_fit()
    rounded = np.round(y)
    monkeypatch.setattr(
        fractional_glm,
        "completely_separated",
        lambda design, target, weights: (
            False if np.array_equal(target, rounded) else real(design, target, weights)
        ),
    )
    with pytest.raises(fractional_glm.QuasiBinomialConvergenceError):
        QuasiBinomialGLM(numerical_separation=True).fit(X, y)


def test_a_flagged_failure_is_recorded_and_a_default_one_stops_the_run() -> None:
    """The flagged learner's error is one the study harness records as a lost replicate."""
    from cleverly.validation._failures import recordable

    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = rng.binomial(1, expit(1.0 + 2.0 * X[:, 0])).astype(float)
    with pytest.raises(fractional_glm.QuasiBinomialConvergenceError) as flagged:
        QuasiBinomialGLM(max_iter=2, numerical_separation=True).fit(X, y)
    assert recordable(flagged.value)
    with pytest.raises(RuntimeError) as default:
        QuasiBinomialGLM(max_iter=2).fit(X, y)
    assert type(default.value) is RuntimeError
    assert not recordable(default.value)
