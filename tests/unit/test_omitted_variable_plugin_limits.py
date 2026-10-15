r"""The curve of the plug-in :math:`\nu^2` with a logistic or a known treatment mechanism.

Under ``nu2_estimator="plugin"`` the estimate :math:`E_n[\hat\alpha^2]` squares the
representer of the fitted mechanism, so its curve carries the mechanism's own term
:math:`\psi^\gamma` (``cleverly.sensitivity._mechanism_stack``).  Each witness here compares
the package curve, row by row, with a comparator that reads no package curve.

* The exact-law witnesses (W1 to W4, W9) differentiate the plug-in functional of
  :mod:`tests.discrete_law_logistic` by a complex step, on the finite laws of
  :mod:`tests.discrete_law` and :mod:`tests.discrete_law_multi`.  The fitted rows are the
  law, so the derivative in the direction of a row is the curve at that row.
* The infinitesimal jackknife (W5, P2) refits each fold's scikit-learn model with one row's
  weight moved by :data:`EPSILON`, renormalised to mean one, on fixed folds.
* The finite-difference checks (W6) and the numerical sandwich (Saul and Hudgens 2020,
  Equation (2)) rebuild the pieces of the stacked estimating equation in numpy.

:class:`TestTheWitnessHasTeeth` commits the mutations M0 to M12 of the X26 plan.  Each one
patches one internal of the stack, and each floor is the smallest deviation measured under
that mutation, rounded down.
"""

from __future__ import annotations

import contextlib
import functools
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import cleverly.sensitivity._mechanism_stack as ms
import cleverly.sensitivity.omitted_variable as ov
from cleverly.estimators import TMLE
from cleverly.estimators._mechanism_fit import describe_mechanism
from cleverly.estimators._nuisance import Propensity, Truncation
from cleverly.inference.cluster import influence_variance
from tests import discrete_law as law
from tests import discrete_law_logistic as logistic
from tests import discrete_law_multi as multi
from tests.conftest import OracleOutcome
from tests.unit import _exact_sensitivity_support as support

#: The tolerance of every exact-law row comparison.  The largest deviation measured was
#: 8.6e-12, on the three-arm ``atc[2 vs 0]`` with a linear design; the scikit-learn root
#: at ``tol=1e-14`` sets it.
ATOL = 1e-9

#: The tolerance of the jackknife and finite-difference comparisons, relative to the largest
#: comparator value.  The largest deviation measured was 2.0e-9.
RTOL = 1e-6

#: The weight step of the infinitesimal jackknife.
EPSILON = 1e-4

#: The finite-difference step of W6 and of the numerical sandwich.
STEP = 1e-6

#: The rows of the simulated sample that the jackknife checks.
ROWS = np.arange(0, 200, 20)

#: The treatment learner settings of every fit: a tight root, so that the captured
#: coefficients solve the score to the precision the witnesses need.
TIGHT: dict[str, Any] = {"solver": "newton-cg", "tol": 1e-14, "max_iter": 1000}

#: Bounds that move no cell of any simulated fit.
NO_CLIP = (1e-6, 1.0 - 1e-6)

#: The weight function of the weighted two-arm fits.
WEIGHTING = "treatment_and_outcome"

#: The strength of the lower bound that the clustered witness reads.
CF_Y, CF_D = 0.5, 0.3
STRENGTH = float(np.sqrt(CF_Y * CF_D / (1.0 - CF_D)))

#: The cluster of each row of the two-arm law: 50 clusters of 20 rows, by row index.
CLUSTER = np.arange(law.N) % 50

#: The two-arm names, on the arm indices of :mod:`tests.discrete_law_logistic`.
BINARY_NAMES: dict[str, str] = {
    "ey1": "ey[1]",
    "ey0": "ey[0]",
    "ate": "ate[1 vs 0]",
    "att": "att[1 vs 0]",
    "atc": "atc[1 vs 0]",
}

#: The three-arm names, on the arm indices of :mod:`tests.discrete_law_multi`.
MULTI_NAMES: tuple[str, ...] = (
    "ey[0]",
    "ey[1]",
    "ey[2]",
    "ate[1 vs 0]",
    "ate[2 vs 0]",
    "att[1 vs 0]",
    "att[2 vs 0]",
    "atc[1 vs 0]",
    "atc[2 vs 0]",
)

# Floors: the smallest deviation each mutation produced on its named witnesses, rounded
# down.  Exact-law floors are absolute; jackknife floors are relative to the largest
# jackknife value.  The measured minimum follows each floor.
M1_EXACT_FLOOR = 3.7  # 3.731, W1 ey0
M1_JACKKNIFE_FLOOR = 0.8  # 0.8264, W5 binary att
M2_EXACT_FLOOR = 0.02  # 0.0233, W1w ate
M2_JACKKNIFE_FLOOR = 0.009  # 0.009111, W5 binary atc
M3_JACKKNIFE_FLOOR = 0.7  # 0.786, W5 ridge-multinomial att
M4_EXACT_FLOOR = 3.2  # 3.28, W4 ey0
M4_JACKKNIFE_FLOOR = 2.4  # 2.442, W5 clipped att
M5_EXACT_FLOOR = 3e-3  # 0.003004, W3 C=0.5 ey[mid], with the root check widened
M5_JACKKNIFE_FLOOR = 0.036  # 0.0364, W5 ridge-multinomial ate, with the root check widened
M6_EXACT_FLOOR = 7.5  # 7.503, W9 weighted att
M7_JACKKNIFE_FLOOR = 0.19  # 0.1917, P2 att
M8_EXACT_FLOOR = 3.7  # 3.731, W1 ey0
M9_EXACT_FLOOR = 4.8e-3  # 0.004843, W1r ey0
M9_JACKKNIFE_FLOOR = 0.039  # 0.03973, W5 ridge-in-sample att
M10_JACKKNIFE_FLOOR = 3.6e-3  # 0.003676, P2 att
M11_JACKKNIFE_FLOOR = 4.8e-3  # 0.004862, P2 att
M12_EXACT_FLOOR = 7.4  # 7.5, W9 ey0

#: The smallest :math:`\max_i |\psi^\gamma_i|` on the W1 law over the five estimands, rounded
#: down (3.731, ``ey0``).
W7_FLOOR = 3.7

#: The smallest relative gap between the variances with and without the mechanism term on
#: the misspecified W1 and W1w laws, rounded down (0.01411, W1 ``ey0``; identity test (b)).
IDENTITY_GAP_FLOOR = 0.014


@contextlib.contextmanager
def _silenced() -> Iterator[None]:
    """Ignore warnings inside a fit: ``newton-cg`` at ``tol=1e-14`` notes failed line searches."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _learner(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, **TIGHT)


# --------------------------------------------------------------------------- exact laws


class _SaturatedBinaryLaw(law.DiscreteLaw):
    """The two-arm law's oracle, read off the two indicator columns of ``W``."""

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        columns = np.asarray(covariates, dtype=float).reshape(-1, 2)
        return np.rint(columns[:, 0] + 2.0 * columns[:, 1]).astype(int)


class _SaturatedMultiOutcome(multi.OracleMultiOutcome):
    """The three-arm outcome oracle, read off the two indicator columns of ``W``."""

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        w = design[:, multi.K - 1] + 2.0 * design[:, multi.K]
        return super().predict(np.column_stack([design[:, : multi.K - 1], w]))


@dataclass(frozen=True)
class ExactFit:
    """One in-sample fit on a finite law."""

    law: str
    design: str = "linear"
    C: float = float("inf")
    weighted: bool = False
    bounds: tuple[float, float] | None = None
    known: bool = False
    clustered: bool = False

    @property
    def n_arms(self) -> int:
        return 2 if self.law == "binary" else multi.K

    @property
    def n(self) -> int:
        return law.N if self.law == "binary" else multi.N

    @property
    def support(self) -> np.ndarray:
        return np.array(law.SUPPORT if self.law == "binary" else multi.SUPPORT)

    @property
    def probs(self) -> np.ndarray:
        return np.asarray(law.PROBS if self.law == "binary" else multi.PROBS)

    def cell_design(self) -> np.ndarray:
        w = self.support[:, 0].astype(float)
        if self.design == "linear":
            return np.column_stack([np.ones_like(w), w])
        return np.column_stack([np.ones_like(w), w == 1, w == 2]).astype(float)

    def cell_weights(self) -> np.ndarray | None:
        if not self.weighted:
            return None
        if self.law == "binary":
            return law.cell_weights(law.WEIGHT_FUNCTIONS[WEIGHTING])
        return support.multi_cell_weights()

    def cell_mechanism(self) -> np.ndarray | None:
        if not self.known:
            return None
        w = self.support[:, 0]
        if self.law == "binary":
            return np.column_stack([1.0 - law.G_EXACT[w], law.G_EXACT[w]])
        return np.asarray(multi.G_EXACT[w])

    def penalty(self) -> np.ndarray:
        d = self.cell_design().shape[1]
        return np.asarray(logistic.penalty_matrix(self.n_arms, d, C=self.C, n=self.n))

    def first_rows(self) -> np.ndarray:
        return law.first_row_of() if self.law == "binary" else support.multi_first_row_of()

    def contamination(self, functional: Callable[[Any], Any]) -> np.ndarray:
        return law.contamination_eif(functional, self.probs, tuple(map(tuple, self.support)))

    def __str__(self) -> str:
        parts = [self.law, self.design]
        if not np.isinf(self.C):
            parts.append(f"C={self.C:g}")
        parts += [
            label
            for flag, label in (
                (self.weighted, "weighted"),
                (self.bounds is not None, "clipped"),
                (self.known, "known"),
                (self.clustered, "clustered"),
            )
            if flag
        ]
        return "-".join(parts)


@functools.cache
def exact_fit(spec: ExactFit) -> Any:
    """The in-sample TMLE fit of ``spec``, with the law's oracle outcome."""
    kwargs: dict[str, Any] = {}
    if spec.bounds is not None:
        kwargs["g_bounds"] = spec.bounds
    fit_kwargs: dict[str, Any] = {}
    covariates = ["W"]
    if spec.law == "binary":
        frame = law.frame()
        dgp = law.DiscreteLaw()
        if spec.weighted:
            cells = law.cell_weights(law.WEIGHT_FUNCTIONS[WEIGHTING])
            dgp = law.DiscreteLaw(law.tilt(law.PROBS, cells))
            frame = frame.assign(w=law.row_weights(cells))
            fit_kwargs["weights"] = "w"
        outcome: Any = OracleOutcome(dgp)
        if spec.design == "saturated":
            outcome = OracleOutcome(_SaturatedBinaryLaw(dgp.probs))
        if spec.known:
            fit_kwargs["treatment_probabilities"] = law.G_EXACT[frame["W"].to_numpy().astype(int)]
        estimator_kwargs: dict[str, Any] = {"estimands": tuple(BINARY_NAMES)}
    else:
        frame = multi.frame()
        if spec.weighted:
            raw = support.RAW_WEIGHT[frame["W"].to_numpy().astype(int)]
            frame = frame.assign(obs_weight=raw)
            fit_kwargs["weights"] = "obs_weight"
        outcome = multi.OracleMultiOutcome()
        if spec.design == "saturated":
            outcome = _SaturatedMultiOutcome()
        if spec.known:
            w = frame["W"].to_numpy().astype(int)
            for j, label in enumerate(multi.LABELS):
                frame[f"p_{label}"] = multi.G_EXACT[w, j]
            fit_kwargs["treatment_probabilities"] = {label: f"p_{label}" for label in multi.LABELS}
        estimator_kwargs = {"estimands": support.MULTI_ESTIMANDS, "reference": support.REFERENCE}
    if spec.design == "saturated":
        w = frame["W"].to_numpy()
        frame = frame.assign(W1=(w == 1).astype(float), W2=(w == 2).astype(float))
        covariates = ["W1", "W2"]
    if spec.clustered:
        frame = frame.assign(cluster=CLUSTER)
        fit_kwargs["id"] = "cluster"
    with _silenced():
        return (
            TMLE(
                outcome_learner=outcome,
                treatment_learner=_learner(spec.C),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
                **estimator_kwargs,
                **kwargs,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates, **fit_kwargs)
            .single()
        )


@dataclass(frozen=True)
class ExactCase:
    """One estimand of one exact-law fit, and the witness it belongs to."""

    witness: str
    fit: ExactFit
    oracle: str

    @property
    def logistic_name(self) -> str:
        return BINARY_NAMES[self.oracle] if self.fit.law == "binary" else self.oracle

    @property
    def reported(self) -> str:
        return self.oracle if self.fit.law == "binary" else multi.reported_name(self.oracle)

    @property
    def conditional(self) -> bool:
        return self.oracle.startswith(("att", "atc"))

    def __str__(self) -> str:
        return f"{self.witness}-{self.fit}-{self.reported}"


def _cases(witness: str, fit: ExactFit) -> tuple[ExactCase, ...]:
    names = tuple(BINARY_NAMES) if fit.law == "binary" else MULTI_NAMES
    return tuple(ExactCase(witness, fit, name) for name in names)


W1 = _cases("W1", ExactFit("binary"))
W1W = _cases("W1w", ExactFit("binary", weighted=True))
W1R = _cases("W1r", ExactFit("binary", C=1.0))
W2 = _cases("W2", ExactFit("binary", design="saturated"))
W3 = (
    _cases("W3", ExactFit("multi"))
    + _cases("W3", ExactFit("multi", weighted=True))
    + _cases("W3", ExactFit("multi", C=0.5))
    + _cases("W3", ExactFit("multi", design="saturated"))
)
W4 = _cases("W4", ExactFit("binary", design="saturated", bounds=(0.3, 0.7)))
W9 = (
    _cases("W9", ExactFit("binary", design="saturated", known=True))
    + _cases("W9", ExactFit("binary", design="saturated", known=True, weighted=True))
    + _cases("W9", ExactFit("multi", design="saturated", known=True))
    + _cases("W9", ExactFit("multi", design="saturated", known=True, weighted=True))
)
EXACT: tuple[ExactCase, ...] = W1 + W1W + W1R + W2 + W3 + W4 + W9


@functools.cache
def exact_curve(case: ExactCase) -> np.ndarray:
    """The complex-step Gateaux derivative of the plug-in functional at every support point."""
    spec = case.fit
    design = spec.cell_design()
    arms = spec.support[:, 1]
    weights = spec.cell_weights()
    penalty = spec.penalty()
    mechanism = spec.cell_mechanism()

    def functional(probs: Any) -> Any:
        return logistic.plugin_second_moment(
            probs,
            design,
            arms,
            spec.n_arms,
            case.logistic_name,
            weights=weights,
            penalty=penalty,
            bounds=spec.bounds,
            mechanism=mechanism,
        )

    return spec.contamination(functional)


def _elements(result: Any, name: str, estimator: str = "plugin") -> Any:
    return ov.sensitivity_elements(result, name, nu2_estimator=estimator)


def reported_curve(case: ExactCase) -> np.ndarray | None:
    """The package's plug-in curve at one row of each support point, or ``None``."""
    elements = _elements(exact_fit(case.fit), case.reported)
    if elements.psi_nu2 is None:
        return None
    return np.asarray(elements.psi_nu2)[case.fit.first_rows()]


def exact_deviation(case: ExactCase) -> float:
    """The largest distance between the package curve and the derivative."""
    curve = reported_curve(case)
    if curve is None:
        return float("inf")
    return float(np.max(np.abs(curve - exact_curve(case))))


def mechanism_term(result: Any, name: str) -> np.ndarray:
    """The :math:`\\psi^\\gamma` the bound adds, read off the call that builds it."""
    captured: list[Any] = []
    current = ov.plugin_mechanism_term

    def spy(*arguments: Any) -> Any:
        out = current(*arguments)
        captured.append(out)
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ov, "plugin_mechanism_term", spy)
        _elements(result, name)
    (term,) = captured
    assert not isinstance(term, ms.MechanismRefusal), term
    return np.asarray(term)


# ---------------------------------------------------------------------- simulated sample

#: The simulated sample size of W5, W6, W8, P1, P2 and the sandwich.
N_SIMULATED = 200


@functools.cache
def simulated_frame(n_arms: int) -> pd.DataFrame:
    """A sample of 200 rows whose mechanism is not linear in the covariates.

    It carries fixed weights in ``[0.5, 2]`` and pairs of rows in one group, so that
    cross-fitting draws grouped folds.
    """
    rng = np.random.default_rng(20261015)
    n = N_SIMULATED
    w = rng.normal(size=(n, 3))
    if n_arms == 2:
        p1 = 1.0 / (1.0 + np.exp(-(0.4 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2] ** 2)))
        a = (rng.random(n) < p1).astype(float)
    else:
        eta = np.column_stack(
            [
                np.zeros(n),
                0.3 * w[:, 0] - 0.2 * w[:, 1],
                -0.2 * w[:, 0] + 0.3 * w[:, 1] + 0.1 * w[:, 2] ** 2,
            ]
        )
        p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
        a = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).astype(float)
    y = a + w[:, 0] + rng.normal(size=n)
    return pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W1": w[:, 0],
            "W2": w[:, 1],
            "W3": w[:, 2],
            "wt": rng.uniform(0.5, 2.0, n),
            "grp": np.arange(n) // 2,
        }
    )


@dataclass(frozen=True)
class Variant:
    """One fit on the simulated sample."""

    name: str
    n_arms: int
    C: float = float("inf")
    cross_fit: bool = True
    bounds: tuple[float, float] = NO_CLIP
    scaled: bool = False
    duplicate: bool = False
    jackknife: tuple[str, ...] = ()

    def learner(self) -> Any:
        model = _learner(self.C)
        if self.scaled:
            return Pipeline([("scale", StandardScaler()), ("model", model)])
        return model

    @property
    def covariates(self) -> list[str]:
        return ["W1", "W2", "W3", "W3copy"] if self.duplicate else ["W1", "W2", "W3"]


VARIANTS: dict[str, Variant] = {
    variant.name: variant
    for variant in (
        Variant("binary", 2, jackknife=("ate", "att", "atc")),
        Variant(
            "multinomial",
            3,
            jackknife=("ate[2.0 vs 0.0]", "att[1.0 vs 0.0]", "atc[2.0 vs 0.0]"),
        ),
        Variant("ridge", 2, C=1.0, jackknife=("ate", "atc")),
        Variant("ridge-multinomial", 3, C=0.5, jackknife=("att[2.0 vs 0.0]", "ate[1.0 vs 0.0]")),
        Variant("ridge-in-sample", 2, C=1.0, cross_fit=False, jackknife=("ate", "att")),
        Variant("clipped", 2, bounds=(0.3, 0.7), jackknife=("ate", "att")),
        Variant("scaled-ridge", 2, C=1.0, scaled=True, jackknife=("ate", "att")),
        Variant("scaled", 2, scaled=True),
        Variant("duplicated", 2, duplicate=True),
        Variant("binary-in-sample", 2, cross_fit=False),
        Variant("multinomial-in-sample", 3, cross_fit=False),
    )
}


@functools.cache
def variant_fit(name: str) -> Any:
    """The TMLE fit of one variant, with four grouped folds when it cross-fits."""
    variant = VARIANTS[name]
    frame = simulated_frame(variant.n_arms)
    if variant.duplicate:
        frame = frame.assign(W3copy=frame["W3"])
    with _silenced():
        return (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=variant.learner(),
                cross_fit=variant.cross_fit,
                n_folds=4,
                q_bounds=(-30.0, 30.0),
                estimands=("ate", "att", "atc"),
                random_state=0,
                g_bounds=variant.bounds,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=variant.covariates,
                weights="wt",
                id="grp",
            )
            .single()
        )


def _fold_rows(result: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = result.repeats[0].nuisance.folds
    if folds.n_folds == 1:
        everything = np.arange(result.data.n)
        return [(everything, everything)]
    return [(np.asarray(train), np.asarray(test)) for train, test in folds]


def _logistic_name(name: str) -> str:
    """A reported simulated-sample name on integer arm indices."""
    if "[" not in name:
        return f"{name}[1 vs 0]"
    stem, _, rest = name.partition("[")
    arm, reference = (int(float(part)) for part in rest[:-1].split(" vs "))
    return f"{stem}[{arm} vs {reference}]"


def _plugin_nu2(name: str, mechanism: np.ndarray, arms: np.ndarray, weights: np.ndarray) -> float:
    """The plug-in second moment of mean-one ``weights`` and a bounded mechanism."""
    oracle = _logistic_name(name)
    conditioning = logistic.conditioning_arm(oracle)
    share = None if conditioning is None else float(np.mean(weights * (arms == conditioning)))
    alpha = logistic.representer(oracle, arms, mechanism, share)
    return float(np.mean(weights * alpha**2))


def _refit_mechanism(variant: Variant, result: Any, weights: np.ndarray) -> np.ndarray:
    """Each fold's model refitted by the test, at ``weights``, bounded as the fit bounds it."""
    x = np.asarray(result.data.covariates, dtype=float)
    a = np.asarray(result.data.treatment, dtype=float)
    out = np.zeros((result.data.n, variant.n_arms))
    with _silenced():
        for train, test in _fold_rows(result):
            model = clone(variant.learner())
            if isinstance(model, Pipeline):
                model.fit(x[train], a[train], model__sample_weight=weights[train])
            else:
                model.fit(x[train], a[train], sample_weight=weights[train])
            out[test] = model.predict_proba(x[test])
    bounded, _ = logistic.truncate(out, variant.bounds)
    return np.asarray(bounded)


@functools.cache
def jackknife(name: str) -> dict[str, np.ndarray]:
    """``n`` times the central difference of the plug-in estimate in each row's weight."""
    variant = VARIANTS[name]
    result = variant_fit(name)
    n = result.data.n
    base = np.asarray(result.data.weights, dtype=float)
    arms = np.asarray(result.data.treatment, dtype=float).astype(int)
    values: dict[str, list[float]] = {estimand: [] for estimand in variant.jackknife}
    for row in ROWS:
        moved = []
        for sign in (1.0, -1.0):
            weights = base.copy()
            weights[row] *= 1.0 + sign * EPSILON
            weights *= n / weights.sum()
            moved.append((weights, _refit_mechanism(variant, result, weights)))
        for estimand in variant.jackknife:
            plus, minus = (_plugin_nu2(estimand, g, arms, w) for w, g in moved)
            values[estimand].append(n * (plus - minus) / (2.0 * EPSILON))
    return {estimand: np.array(rows) for estimand, rows in values.items()}


@dataclass(frozen=True)
class JackknifeCase:
    """One estimand of one jackknife variant."""

    witness: str
    variant: str
    estimand: str

    def __str__(self) -> str:
        return f"{self.witness}-{self.variant}-{self.estimand}"


def jackknife_deviation(case: JackknifeCase) -> float:
    """The largest distance from the jackknife, relative to its largest value."""
    elements = _elements(variant_fit(case.variant), case.estimand)
    if elements.psi_nu2 is None:
        return float("inf")
    expected = jackknife(case.variant)[case.estimand]
    gap = np.abs(np.asarray(elements.psi_nu2)[ROWS] - expected)
    return float(np.max(gap) / np.max(np.abs(expected)))


JACKKNIFE: tuple[JackknifeCase, ...] = tuple(
    JackknifeCase("P2" if variant.scaled else "W5", variant.name, estimand)
    for variant in VARIANTS.values()
    for estimand in variant.jackknife
)


def _jackknife_cases(*variants: str) -> tuple[JackknifeCase, ...]:
    return tuple(case for case in JACKKNIFE if case.variant in variants)


# ------------------------------------------------------------------------------ witnesses


class TestTheExactLawCurve:
    """W1, W1w, W1r, W2, W3, W4 and W9: the curve is the Gateaux derivative, row by row."""

    @pytest.mark.parametrize("case", EXACT, ids=str)
    def test_the_curve_is_the_gateaux_derivative(self, case: ExactCase) -> None:
        curve = reported_curve(case)
        assert curve is not None
        np.testing.assert_allclose(curve, exact_curve(case), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("case", EXACT, ids=str)
    def test_the_estimate_is_the_functional(self, case: ExactCase) -> None:
        """The premise: at the law, the package's plug-in is the oracle's functional."""
        spec = case.fit
        expected = logistic.plugin_second_moment(
            spec.probs,
            spec.cell_design(),
            spec.support[:, 1],
            spec.n_arms,
            case.logistic_name,
            weights=spec.cell_weights(),
            penalty=spec.penalty(),
            bounds=spec.bounds,
            mechanism=spec.cell_mechanism(),
        )
        elements = _elements(exact_fit(spec), case.reported)
        assert elements.nu2 == pytest.approx(float(np.real(expected)), rel=1e-11)
        assert elements.plugin_mechanism == ("known" if spec.known else "logistic")

    def test_the_clip_binds_on_the_clipped_law(self) -> None:
        """W4 is not vacuous: the W = 2 cell, with g = 0.25, sits below the bound 0.3."""
        result = exact_fit(W4[0].fit)
        truncation = result.repeats[0].nuisance.propensity.truncate((0.3, 0.7))
        assert 0.0 < float(np.mean(truncation.units)) < 1.0

    @pytest.mark.parametrize("case", W2, ids=str)
    def test_the_saturated_curve_is_the_doubly_robust_curve(self, case: ExactCase) -> None:
        """W2: with a saturated model, the plug-in curve is the curve of ``nu^2`` itself."""
        result = exact_fit(case.fit)
        plugin = _elements(result, case.reported).psi_nu2
        robust = _elements(result, case.reported, "doubly_robust").psi_nu2
        assert plugin is not None and robust is not None
        np.testing.assert_allclose(plugin, robust, atol=ATOL, rtol=0)
        nonparametric = law.gateaux_eif(lambda probs: law.riesz_second_moment(probs, case.oracle))
        np.testing.assert_allclose(exact_curve(case), nonparametric, atol=ATOL, rtol=0)

    @pytest.mark.parametrize(
        "case", tuple(case for case in W3 if case.fit.design == "saturated"), ids=str
    )
    def test_the_saturated_three_arm_curve_is_the_doubly_robust_curve(
        self, case: ExactCase
    ) -> None:
        result = exact_fit(case.fit)
        plugin = _elements(result, case.reported).psi_nu2
        robust = _elements(result, case.reported, "doubly_robust").psi_nu2
        assert plugin is not None and robust is not None
        np.testing.assert_allclose(plugin, robust, atol=ATOL, rtol=0)
        nonparametric = support.multi_gateaux_eif(
            lambda probs: multi.riesz_second_moment(probs, case.oracle)
        )
        np.testing.assert_allclose(exact_curve(case), nonparametric, atol=ATOL, rtol=0)

    def test_the_oracle_projection_is_the_fitted_model(self) -> None:
        """The comparator's penalty is scikit-learn's: the projection of the law is the fit."""
        for spec in (ExactFit("binary", C=1.0), ExactFit("multi", C=0.5)):
            result = exact_fit(spec)
            q = spec.probs.reshape(-1)
            theta = logistic.logistic_projection(
                q, spec.cell_design(), spec.support[:, 1], spec.n_arms, penalty=spec.penalty()
            )
            expected = logistic.softmax(spec.cell_design(), theta)
            stored = np.asarray(result.repeats[0].nuisance.propensity.values)
            fitted = stored[spec.first_rows()]
            if spec.law == "multi":
                # The package orders arms by sorted label; this module by its own index.
                fitted = fitted[:, np.argsort(multi.ARM_OF_CODE)]
            np.testing.assert_allclose(fitted, expected, atol=1e-10, rtol=0)


class TestTheKnownMechanism:
    """W9: a declared known mechanism adds no term, and the curve keeps the share term."""

    @pytest.mark.parametrize("case", W9, ids=str)
    def test_the_mechanism_term_is_zero(self, case: ExactCase) -> None:
        term = mechanism_term(exact_fit(case.fit), case.reported)
        assert np.all(term == 0.0)


class TestTheClusteredLimit:
    """W1c: the clustered standard error of the lower limit is the cluster sum of the curve."""

    @pytest.mark.parametrize("name", ["ate", "att", "atc"])
    def test_the_standard_error_sums_the_exact_curve(self, name: str) -> None:
        spec = ExactFit("binary", clustered=True)
        result = exact_fit(spec)
        assert result.inference_status == "influence_curve"
        assert result[name].reference_df is None
        elements = _elements(result, name)
        exact_nu2 = exact_curve(ExactCase("W1c", ExactFit("binary"), name))[law.cell_of_row()]
        psi_bias = (
            elements.sigma2 * exact_nu2 + elements.nu2 * np.asarray(elements.psi_sigma2)
        ) / (2.0 * elements.max_bias)
        rows = np.asarray(result[name].influence_curve) - STRENGTH * psi_bias
        clustered = float(np.sqrt(influence_variance(rows, CLUSTER)))
        bounds = ov.omitted_variable_bounds(
            result, name, cf_y=CF_Y, cf_d=CF_D, rho=1.0, nu2_estimator="plugin"
        )
        reported = (bounds.lower - bounds.ci_lower) / stats.norm.ppf(0.95)
        assert reported == pytest.approx(clustered, rel=1e-10)
        row_level = float(np.sqrt(influence_variance(rows, None)))
        assert abs(clustered - row_level) > 1e-3 * row_level


class TestTheInfinitesimalJackknife:
    """W5 and P2: the curve is ``n`` times the derivative of a refitted estimate."""

    @pytest.mark.parametrize("case", JACKKNIFE, ids=str)
    def test_the_curve_is_the_jackknife(self, case: JackknifeCase) -> None:
        assert jackknife_deviation(case) < RTOL

    def test_the_estimate_is_the_refitted_estimate(self) -> None:
        for name, variant in VARIANTS.items():
            if not variant.jackknife:
                continue
            result = variant_fit(name)
            weights = np.asarray(result.data.weights, dtype=float)
            mechanism = _refit_mechanism(variant, result, weights)
            arms = np.asarray(result.data.treatment, dtype=float).astype(int)
            for estimand in variant.jackknife:
                expected = _plugin_nu2(estimand, mechanism, arms, weights)
                assert _elements(result, estimand).nu2 == pytest.approx(expected, rel=1e-10)

    def test_the_clipped_variant_clips_rows_away_from_the_bounds(self) -> None:
        """The clipped variant moves cells, and no cell is near enough a bound to cross it."""
        result = variant_fit("clipped")
        raw = np.asarray(result.repeats[0].nuisance.propensity.values)[:, 1]
        moved = (raw < 0.3) | (raw > 0.7)
        assert 0 < int(moved.sum()) < raw.size
        # A weight step of 1e-4 moves a probability by about 1e-6.
        assert float(np.min(np.abs(raw[:, None] - np.array([0.3, 0.7])))) > 1e-4

    def test_the_folds_are_grouped_and_cross_fitted(self) -> None:
        result = variant_fit("binary")
        group = simulated_frame(2)["grp"].to_numpy()
        rows = _fold_rows(result)
        assert len(rows) == 4
        for train, test in rows:
            assert not set(group[train]) & set(group[test])


def _row_function(stack: Any, theta: np.ndarray, arms: np.ndarray, weights: np.ndarray) -> Any:
    """``(n, p)`` rows ``w x (e_A - g)_{-0}`` and the fold's penalty allocation, in numpy."""
    g = logistic.softmax(stack.design, theta)
    indicator = np.eye(g.shape[1])[arms]
    residual = (indicator - g)[:, 1:]
    rows = weights[:, None] * np.einsum("im,ij->imj", residual, stack.design).reshape(len(arms), -1)
    allocation = (weights / len(arms))[:, None] * (stack.penalty @ theta.reshape(-1))[None, :]
    return rows, allocation


def _central(function: Callable[[np.ndarray], np.ndarray], point: np.ndarray) -> np.ndarray:
    """The central-difference Jacobian of ``function`` at ``point``, one column per entry."""
    flat = point.reshape(-1)
    columns = []
    for j in range(flat.size):
        up, down = flat.copy(), flat.copy()
        up[j] += STEP
        down[j] -= STEP
        columns.append(
            (np.asarray(function(up)) - np.asarray(function(down))).reshape(-1) / (2.0 * STEP)
        )
    return np.column_stack(columns)


def _fold_stacks(result: Any) -> list[Any]:
    nuisance = result.repeats[0].nuisance
    folds = ms._stacks(nuisance.treatment_fits, nuisance, result.data)
    assert isinstance(folds, list)
    return folds


class TestTheFiniteDifferenceJacobians:
    """W6: the information, ``D_k`` and ``dg/dtheta`` against central differences."""

    @pytest.mark.parametrize("name", ["ridge", "ridge-multinomial", "multinomial"])
    def test_the_information_is_the_derivative_of_the_summed_rows(self, name: str) -> None:
        result = variant_fit(name)
        weights = np.asarray(result.data.weights, dtype=float)
        arms = np.asarray(result.data.treatment, dtype=float).astype(int)
        for fold in _fold_stacks(result):
            shape = fold.stack.theta.shape

            def summed(flat: np.ndarray, fold: Any = fold, shape: Any = shape) -> np.ndarray:
                rows, allocation = _row_function(fold.stack, flat.reshape(shape), arms, weights)
                return np.asarray(rows[fold.train].sum(axis=0) - allocation.sum(axis=0))

            numeric = -_central(summed, fold.stack.theta)
            analytic = fold.stack.information(fold.train, weights)
            np.testing.assert_allclose(analytic, numeric, rtol=0, atol=RTOL * np.abs(numeric).max())

    @pytest.mark.parametrize(
        ("name", "estimand"),
        [("ridge", "atc"), ("ridge-multinomial", "att[2.0 vs 0.0]"), ("clipped", "ate")],
    )
    def test_the_mechanism_term_is_the_finite_difference_stack(
        self, name: str, estimand: str
    ) -> None:
        """``n sum_k D_k J_k^{-1} u_{k,i}`` with ``D_k`` and ``J_k`` by central differences."""
        variant = VARIANTS[name]
        result = variant_fit(name)
        n = result.data.n
        weights = np.asarray(result.data.weights, dtype=float)
        arms = np.asarray(result.data.treatment, dtype=float).astype(int)
        oracle = _logistic_name(estimand)
        conditioning = logistic.conditioning_arm(oracle)
        share = None if conditioning is None else float(np.mean(weights * (arms == conditioning)))
        expected = np.zeros(n)
        for fold in _fold_stacks(result):
            shape = fold.stack.theta.shape

            def contribution(flat: np.ndarray, fold: Any = fold, shape: Any = shape) -> Any:
                g = logistic.softmax(fold.stack.design, flat.reshape(shape))
                bounded, _ = logistic.truncate(g, variant.bounds)
                alpha = logistic.representer(oracle, arms, bounded, share)
                return np.array([np.sum((weights * alpha**2)[fold.test]) / n])

            def summed(flat: np.ndarray, fold: Any = fold, shape: Any = shape) -> np.ndarray:
                rows, allocation = _row_function(fold.stack, flat.reshape(shape), arms, weights)
                return np.asarray(rows[fold.train].sum(axis=0) - allocation.sum(axis=0))

            gradient = _central(contribution, fold.stack.theta)[0]
            information = -_central(summed, fold.stack.theta)
            rows, allocation = _row_function(fold.stack, fold.stack.theta, arms, weights)
            u = -allocation
            u[fold.train] += rows[fold.train]
            expected += n * u @ np.linalg.solve(information, gradient)
        reported = mechanism_term(result, estimand)
        np.testing.assert_allclose(reported, expected, rtol=0, atol=RTOL * np.abs(expected).max())

    @pytest.mark.parametrize("name", ["ridge", "ridge-multinomial"])
    def test_dg_dtheta_is_the_finite_difference_of_predict_proba(self, name: str) -> None:
        """The analytic ``g_c (1{c = m} - g_m) x`` against scikit-learn's own predictions."""
        result = variant_fit(name)
        nuisance = result.repeats[0].nuisance
        x = np.asarray(result.data.covariates, dtype=float)
        for record, fold in zip(nuisance.treatment_fits, _fold_stacks(result), strict=True):
            model = LogisticRegression()
            model.classes_ = np.asarray(record.classes)
            model.coef_ = np.asarray(record.coef, dtype=float)
            model.intercept_ = np.asarray(record.intercept, dtype=float)
            g = fold.stack.probabilities
            np.testing.assert_allclose(model.predict_proba(x), g, rtol=0, atol=1e-12)
            k, d = fold.stack.theta.shape
            for m in range(k):
                row = 0 if g.shape[1] == 2 else m + 1
                for j in range(d):
                    up, down = clone(model), clone(model)
                    for copy, sign in ((up, 1.0), (down, -1.0)):
                        copy.classes_ = model.classes_
                        copy.coef_ = model.coef_.copy()
                        copy.intercept_ = model.intercept_.copy()
                        if j == 0:
                            copy.intercept_[row] += sign * STEP
                        else:
                            copy.coef_[row, j - 1] += sign * STEP
                    numeric = (up.predict_proba(x) - down.predict_proba(x)) / (2.0 * STEP)
                    unit = np.zeros(g.shape[1])
                    unit[m + 1] = 1.0
                    analytic = g * (unit[None, :] - g[:, [m + 1]]) * fold.stack.design[:, [j]]
                    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-8)


class TestTheNonzeroWitness:
    """W7: the mechanism term is large on some row, for every estimand of the W1 law."""

    @pytest.mark.parametrize("name", list(BINARY_NAMES))
    def test_the_mechanism_term_is_not_zero(self, name: str) -> None:
        term = mechanism_term(exact_fit(ExactFit("binary")), name)
        assert float(np.max(np.abs(term))) > W7_FLOOR


class TestTheDesignInvariances:
    """W8 and P1: a duplicated column and a fitted scaler leave the curve where it was."""

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_a_duplicated_column_gives_the_same_curve(self, estimand: str) -> None:
        duplicated = _elements(variant_fit("duplicated"), estimand).psi_nu2
        plain = _elements(variant_fit("binary"), estimand).psi_nu2
        assert duplicated is not None and plain is not None
        np.testing.assert_allclose(duplicated, plain, rtol=0, atol=1e-10 * np.abs(plain).max())

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_a_scaler_before_an_unpenalized_model_gives_the_same_curve(self, estimand: str) -> None:
        scaled = _elements(variant_fit("scaled"), estimand).psi_nu2
        plain = _elements(variant_fit("binary"), estimand).psi_nu2
        assert scaled is not None and plain is not None
        np.testing.assert_allclose(scaled, plain, rtol=0, atol=1e-8 * np.abs(plain).max())


IDENTITY_SATURATED = tuple(
    case
    for case in W2 + W3
    if case.fit.design == "saturated" and not case.fit.weighted and not case.conditional
)
IDENTITY_MISSPECIFIED = tuple(case for case in W1 + W1W if not case.conditional)


def _variance_with_and_without_the_term(case: ExactCase) -> tuple[float, float]:
    result = exact_fit(case.fit)
    elements = _elements(result, case.reported)
    assert elements.psi_nu2 is not None
    curve = np.asarray(elements.psi_nu2)
    term = mechanism_term(result, case.reported)
    return influence_variance(curve, None), influence_variance(curve - term, None)


class TestTheReflectionIdentity:
    """Section 3.4: at a correct model the term changes the curve but not its variance."""

    @pytest.mark.parametrize("case", IDENTITY_SATURATED, ids=str)
    def test_a_saturated_model_keeps_the_variance(self, case: ExactCase) -> None:
        with_term, without_term = _variance_with_and_without_the_term(case)
        assert with_term == pytest.approx(without_term, rel=1e-10)
        term = mechanism_term(exact_fit(case.fit), case.reported)
        assert float(np.max(np.abs(term))) > 0.1

    @pytest.mark.parametrize("case", IDENTITY_MISSPECIFIED, ids=str)
    def test_a_misspecified_model_moves_the_variance(self, case: ExactCase) -> None:
        with_term, without_term = _variance_with_and_without_the_term(case)
        assert abs(with_term - without_term) > IDENTITY_GAP_FLOOR * with_term


SANDWICH = (
    ("binary-in-sample", "att"),
    ("binary-in-sample", "ate"),
    ("multinomial-in-sample", "att[2.0 vs 0.0]"),
    ("multinomial-in-sample", "ate[2.0 vs 0.0]"),
    ("ridge-in-sample", "atc"),
)


class TestTheNumericalSandwich:
    """Section 3.3: the stacked sandwich of Saul and Hudgens (2020), Equation (2)."""

    @pytest.mark.parametrize(("name", "estimand"), SANDWICH)
    def test_the_variance_is_the_stacked_sandwich(self, name: str, estimand: str) -> None:
        variant = VARIANTS[name]
        result = variant_fit(name)
        n = result.data.n
        x = np.asarray(result.data.covariates, dtype=float)
        design = np.column_stack([np.ones(n), x])
        arms = np.asarray(result.data.treatment, dtype=float).astype(int)
        weights = np.asarray(result.data.weights, dtype=float)
        k = variant.n_arms
        with _silenced():
            model = _learner(variant.C).fit(x, arms.astype(float), sample_weight=weights)
        full = np.column_stack([model.intercept_, model.coef_])
        theta = full[:1] if k == 2 else full[1:] - full[0]
        penalty = np.asarray(logistic.penalty_matrix(k, design.shape[1], C=variant.C, n=1))
        oracle = _logistic_name(estimand)
        conditioning = logistic.conditioning_arm(oracle)
        indicator = np.eye(k)[arms]
        p = theta.size

        def psi(parameters: np.ndarray) -> np.ndarray:
            coefficients = parameters[:p].reshape(theta.shape)
            g = logistic.softmax(design, coefficients)
            score = np.einsum("im,ij->imj", (indicator - g)[:, 1:], design).reshape(n, -1)
            score = score - (penalty @ parameters[:p])[None, :] / n
            share = None if conditioning is None else parameters[p + 1]
            alpha = logistic.representer(oracle, arms, g, share)
            columns = [score, (alpha**2 - parameters[p])[:, None]]
            if conditioning is not None:
                columns.append(((arms == conditioning) - parameters[p + 1])[:, None])
            return weights[:, None] * np.column_stack(columns)

        share = None if conditioning is None else float(np.mean(weights * (arms == conditioning)))
        g = logistic.softmax(design, theta)
        nu2 = float(np.mean(weights * logistic.representer(oracle, arms, g, share) ** 2))
        point = np.concatenate([theta.reshape(-1), [nu2], [] if share is None else [share]])
        rows = psi(point)
        assert np.max(np.abs(rows.mean(axis=0))) < 1e-9
        bread = -_central(lambda parameters: psi(parameters).mean(axis=0), point)
        meat = rows.T @ rows / n
        inverse = np.linalg.inv(bread)
        sandwich = inverse @ meat @ inverse.T / n
        elements = _elements(result, estimand)
        assert elements.psi_nu2 is not None
        assert elements.nu2 == pytest.approx(nu2, rel=1e-10)
        expected = influence_variance(np.asarray(elements.psi_nu2), None) * (n - 1) / n
        assert sandwich[p, p] == pytest.approx(expected, rel=RTOL)


# ------------------------------------------------------------------------------ mutations


def _exact_deviations(cases: tuple[ExactCase, ...]) -> dict[str, float]:
    return {str(case): exact_deviation(case) for case in cases}


def _jackknife_deviations(cases: tuple[JackknifeCase, ...]) -> dict[str, float]:
    return {str(case): jackknife_deviation(case) for case in cases}


def _fails(measured: dict[str, float], floor: float) -> None:
    assert min(measured.values()) > floor, measured


def _passes(measured: dict[str, float], tolerance: float) -> None:
    assert max(measured.values()) < tolerance, measured


def _only(cases: tuple[ExactCase, ...], *names: str) -> tuple[ExactCase, ...]:
    return tuple(case for case in cases if case.oracle.split("[")[0] in names)


RIDGE_JACKKNIFE = _jackknife_cases("ridge", "ridge-multinomial", "ridge-in-sample", "scaled-ridge")
CROSS_FIT_JACKKNIFE = tuple(case for case in JACKKNIFE if VARIANTS[case.variant].cross_fit)


class TestTheWitnessHasTeeth:
    """The mutations M0 to M12 of the X26 plan, each against its named witnesses."""

    def test_m0_a_rewrapped_term_changes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = ms.plugin_mechanism_term
        monkeypatch.setattr(ov, "plugin_mechanism_term", lambda *arguments: original(*arguments))
        _passes(_exact_deviations(EXACT), ATOL)
        _passes(_jackknife_deviations(JACKKNIFE), RTOL)

    def test_m1_dropping_the_mechanism_term_fails_w1_w2_w3_w5_w7(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms.plugin_mechanism_term

        def dropped(*arguments: Any) -> Any:
            out = original(*arguments)
            return out if isinstance(out, ms.MechanismRefusal) else np.zeros_like(out)

        monkeypatch.setattr(ov, "plugin_mechanism_term", dropped)
        _fails(_exact_deviations(W1 + W2 + W3), M1_EXACT_FLOOR)
        _fails(_jackknife_deviations(_jackknife_cases("binary", "multinomial")), M1_JACKKNIFE_FLOOR)
        _passes(_exact_deviations(W9), ATOL)
        for name in BINARY_NAMES:
            assert float(np.max(np.abs(mechanism_term(exact_fit(W1[0].fit), name)))) == 0.0

    def test_m2_an_unweighted_information_fails_w1w_and_w5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms.LogisticStack.information
        monkeypatch.setattr(
            ms.LogisticStack,
            "information",
            lambda self, rows, weights: original(self, rows, np.ones_like(weights)),
        )
        _fails(_exact_deviations(W1W), M2_EXACT_FLOOR)
        _fails(_jackknife_deviations(_jackknife_cases("binary", "multinomial")), M2_JACKKNIFE_FLOOR)
        _passes(_exact_deviations(W1), ATOL)

    def test_m3_the_score_on_the_test_rows_fails_the_cross_fitted_w5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms._stacks

        def on_test_rows(*arguments: Any) -> Any:
            out = original(*arguments)
            if isinstance(out, ms.MechanismRefusal):
                return out
            return [replace(fold, train=fold.test) for fold in out]

        monkeypatch.setattr(ms, "_stacks", on_test_rows)
        _fails(_jackknife_deviations(CROSS_FIT_JACKKNIFE), M3_JACKKNIFE_FLOOR)
        _passes(_jackknife_deviations(_jackknife_cases("ridge-in-sample")), RTOL)

    def test_m4_a_derivative_on_the_clipped_cells_fails_w4_and_clipped_w5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = Propensity.truncate

        def unmarked(self: Propensity, bounds: tuple[float, float]) -> Truncation:
            truncation = original(self, bounds)
            return Truncation(truncation.values, np.zeros_like(truncation.clipped))

        monkeypatch.setattr(Propensity, "truncate", unmarked)
        _fails(_exact_deviations(W4), M4_EXACT_FLOOR)
        _fails(_jackknife_deviations(_jackknife_cases("clipped")), M4_JACKKNIFE_FLOOR)
        _passes(_exact_deviations(W1), ATOL)

    def test_m5_a_penalty_without_its_symmetric_centring_fails_the_ridge_w3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The multinomial penalty as ``I ⊗ S`` rather than ``(I - 11'/K) ⊗ S``.

        That is scikit-learn's symmetric penalty read as if each reference coefficient were
        penalized alone.  The root check refuses it (0.131 standard errors on the three-arm
        law), so the test widens the check to see the curve move.
        """
        original = ms._penalty_matrix

        def uncentred(fit: Any, n_arms: int, d: int) -> np.ndarray:
            matrix = np.asarray(original(fit, n_arms, d))
            if n_arms == 2 or not matrix.any():
                return matrix
            slopes = np.eye(d)
            if fit.fit_intercept:
                slopes[0, 0] = 0.0
            return np.asarray(np.kron(np.eye(n_arms - 1), slopes) / fit.C)

        monkeypatch.setattr(ms, "_penalty_matrix", uncentred)
        ridge = tuple(case for case in W3 if not np.isinf(case.fit.C))
        for case in ridge:
            refusal = _elements(exact_fit(case.fit), case.reported).limit_refusal
            assert refusal is not None and refusal.key == "score_not_zero"
        monkeypatch.setattr(ms, "ROOT_TOLERANCE", np.inf)
        _fails(_exact_deviations(ridge), M5_EXACT_FLOOR)
        _fails(_jackknife_deviations(_jackknife_cases("ridge-multinomial")), M5_JACKKNIFE_FLOOR)
        _passes(_exact_deviations(tuple(case for case in W3 if np.isinf(case.fit.C))), ATOL)

    def test_m5_the_wrong_reference_refuses_under_a_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Coefficients against the last arm, while the stack drops the first column.

        With no penalty the curve does not depend on the reference, because the score, the
        information and ``D_k`` read the probabilities alone.  So the unpenalized witnesses
        pass.  Under a penalty the misread coefficients do not solve the score, and the
        root check refuses.
        """
        original_form = ms._reference_form
        original_softmax = ms._softmax_reference

        def last_reference(fit: Any, n_arms: int) -> np.ndarray:
            if n_arms == 2:
                return np.asarray(original_form(fit, n_arms))
            full = np.column_stack([fit.intercept, fit.coef]) if fit.fit_intercept else fit.coef
            return np.asarray(full[:-1] - full[-1], dtype=float)

        def last_column(design: np.ndarray, theta: np.ndarray) -> np.ndarray:
            if theta.shape[0] == 1:
                return np.asarray(original_softmax(design, theta))
            eta = np.column_stack([design @ theta.T, np.zeros(design.shape[0])])
            expd = np.exp(eta - eta.max(axis=1, keepdims=True))
            return np.asarray(expd / expd.sum(axis=1, keepdims=True))

        monkeypatch.setattr(ms, "_reference_form", last_reference)
        monkeypatch.setattr(ms, "_softmax_reference", last_column)
        unpenalized = tuple(case for case in W3 if np.isinf(case.fit.C))
        _passes(_exact_deviations(unpenalized), ATOL)
        for case in W3:
            if not np.isinf(case.fit.C):
                refusal = _elements(exact_fit(case.fit), case.reported).limit_refusal
                assert refusal is not None and refusal.key == "score_not_zero"

    def test_m6_dropping_the_share_term_fails_the_conditional_w1_w2_w9(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ov, "_conditioning_share_influence", lambda *arguments: 0.0)
        conditional = tuple(case for case in W1 + W2 + W9 if case.conditional)
        _fails(_exact_deviations(conditional), M6_EXACT_FLOOR)
        _passes(_exact_deviations(tuple(case for case in W1 if not case.conditional)), ATOL)

    def test_m7_a_scaled_mean_information_fails_the_cross_fitted_w5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms.LogisticStack.information

        def scaled_mean(self: Any, rows: Any, weights: np.ndarray) -> np.ndarray:
            unpenalized = original(self, rows, weights) - self.penalty
            return np.asarray(unpenalized * len(weights) / len(rows) + self.penalty)

        monkeypatch.setattr(ms.LogisticStack, "information", scaled_mean)
        _fails(_jackknife_deviations(CROSS_FIT_JACKKNIFE), M7_JACKKNIFE_FLOOR)
        _passes(_jackknife_deviations(_jackknife_cases("ridge-in-sample")), RTOL)

    def test_m8_a_reference_mechanism_free_of_theta_fails_w1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms.representer_g_derivative

        def frozen(*arguments: Any) -> tuple[np.ndarray, np.ndarray]:
            alpha, derivative = original(*arguments)
            arms = arguments[3]
            if len(arms) == 2:
                derivative = derivative.copy()
                derivative[:, list(arms).index(0.0)] = 0.0
            return alpha, derivative

        monkeypatch.setattr(ms, "representer_g_derivative", frozen)
        _fails(_exact_deviations(_only(W1, "ate", "atc", "ey0")), M8_EXACT_FLOOR)
        _passes(_exact_deviations(_only(W1, "ey1")), ATOL)

    def test_m9_an_information_without_the_penalty_fails_w1r_and_ridge_w5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms.LogisticStack.information
        monkeypatch.setattr(
            ms.LogisticStack,
            "information",
            lambda self, rows, weights: original(self, rows, weights) - self.penalty,
        )
        _fails(_exact_deviations(W1R), M9_EXACT_FLOOR)
        _fails(_jackknife_deviations(RIDGE_JACKKNIFE), M9_JACKKNIFE_FLOOR)
        _passes(_exact_deviations(W1), ATOL)

    def test_m10_dropping_the_penalty_allocation_fails_ridge_w5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = ms._stacks

        def unallocated(*arguments: Any) -> Any:
            out = original(*arguments)
            if isinstance(out, ms.MechanismRefusal):
                return out
            return [
                replace(fold, stack=replace(fold.stack, penalty=np.zeros_like(fold.stack.penalty)))
                for fold in out
            ]

        monkeypatch.setattr(ms, "_stacks", unallocated)
        _fails(_jackknife_deviations(RIDGE_JACKKNIFE), M10_JACKKNIFE_FLOOR)
        _passes(_jackknife_deviations(_jackknife_cases("binary")), RTOL)

    def test_m11_the_design_from_the_raw_covariates_fails_p2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline's coefficients re-expressed on the raw covariates.

        The probabilities do not change, so the stack still matches the stored mechanism, and
        the root check passes.  The penalty then acts on the raw slopes, which is not the
        model that was fitted, so only the witness sees it.
        """
        original_design = ms._design
        original_form = ms._reference_form

        def raw_design(fit: Any, covariates: np.ndarray) -> np.ndarray:
            return np.asarray(original_design(replace(fit, steps=()), covariates))

        def raw_form(fit: Any, n_arms: int) -> np.ndarray:
            theta = np.asarray(original_form(fit, n_arms))
            for step in reversed(fit.steps or ()):
                scale = np.ones(theta.shape[1] - 1) if step.scale is None else step.scale
                mean = np.zeros(theta.shape[1] - 1) if step.mean is None else step.mean
                slopes = theta[:, 1:] / scale
                theta = np.column_stack([theta[:, 0] - slopes @ mean, slopes])
            return theta

        monkeypatch.setattr(ms, "_design", raw_design)
        monkeypatch.setattr(ms, "_reference_form", raw_form)
        _fails(_jackknife_deviations(_jackknife_cases("scaled-ridge")), M11_JACKKNIFE_FLOOR)
        _passes(_jackknife_deviations(_jackknife_cases("ridge")), RTOL)

    def test_m12_a_known_mechanism_treated_as_estimated_fails_w9(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A test-side ``LogisticRegression`` capture on the known fit, past the known branch.

        The saturated model reproduces the known mechanism, so the stack accepts it.
        """
        original = ms.plugin_mechanism_term

        def estimated(result: Any, repeat: Any, *arguments: Any) -> Any:
            nuisance = repeat.nuisance
            if nuisance.treatment_mechanism != "known":
                return original(result, repeat, *arguments)
            data = result.data
            with _silenced():
                model = _learner(float("inf")).fit(
                    data.covariates, data.treatment, sample_weight=data.weights
                )
            values = np.asarray(nuisance.propensity.values)
            swapped = replace(
                nuisance,
                propensity=Propensity(values, nuisance.propensity.arms),
                treatment_fits=(describe_mechanism(model),),
            )
            return original(result, replace(repeat, nuisance=swapped), *arguments)

        monkeypatch.setattr(ov, "plugin_mechanism_term", estimated)
        _fails(_exact_deviations(W9), M12_EXACT_FLOOR)
        _passes(_exact_deviations(W1), ATOL)
