r"""Rule 2 of the X26 red-cell policy: does the package curve agree with two independent curves?

The declared run of ``omitted-variable-bound-plugin-se`` left two cells red:
``multinomial_logistic_crossfit_ate_2v0_lower__correctly_specified`` and
``multinomial_logistic_crossfit_att_2v0_lower__correctly_specified``.  Rule 2 of
``tests.studies.omitted_variable_bound_plugin.RED_CELL_POLICY`` asks for a comparison before the
cells take an owner.  On 20 replicates of the scenario, it compares the curve of the plug-in
:math:`\hat\nu^2` that the package reports with two curves that read no package curve.  A
mismatch is a defect.

**The draws.**  Replicate ``k`` is the draw that property replicate ``k`` of the declared run
fitted: ``stream_seed(STUDY, "property_sample", "interval_calibration", SCENARIO, k)``.  The
committed ``property-replicates.parquet`` holds the reported standard error of that draw, and
:func:`reported_standard_error` reads the lower-bound error that the declared fit gives.  The diagnostic reads the same
samples as the red cells.

**The fits.**  ``study`` is the declared fit, ``LogisticRegression(C=numpy.inf, max_iter=2000)``
with the default ``lbfgs`` solver.  ``tight`` is the same fit with ``newton-cg`` at
``tol=1e-14``, as the unit witnesses fit.  The two comparators differentiate a refit, so they
need a root to full precision, and the ``tight`` fit supplies it.  Both fits read the same
folds, which :func:`folds_of` asserts.

**The infinitesimal jackknife** (unit witness W5, ``tests/unit/test_omitted_variable_plugin_limits.py``).
For a row :math:`i`, the curve of :math:`\hat\nu^2 = E_n[w \alpha^2]` is :math:`n` times the
derivative of the refitted estimate in the weight of row :math:`i`. The test refits each fold's
model at weight :math:`w_i (1 \pm 10^{-4})`, renormalised to mean one, and takes the central
difference. The rows are every 20th row and the five rows of largest :math:`|\psi|` of each
parameter, 55 to 60 rows. The deviation is the largest gap to the package curve, over the largest
jackknife value.

**The numerical sandwich** (``TestTheNumericalSandwich``).  The unit witness stacks the score of
one in-sample fit with the estimating equation of :math:`\nu^2`.  A cross-fitted fit has one
coefficient vector for each fold.  The stack here holds the five scores, each summed over its
training rows, with :math:`\sum_i w_i (\alpha_i(\theta_{f(i)})^2 - \nu^2) = 0` and, for the ATT,
the share equation. :math:`f(i)` is the fold that predicts row :math:`i`.  The bread is a central
difference of the stacked mean, with step :math:`10^{-6}`.  The sandwich variance of
:math:`\hat\nu^2` must equal :math:`\frac{n-1}{n}` times the influence variance of the package
curve.

**Criteria, declared before the run.**  The cell agrees when, for every replicate and parameter,

* the ``tight`` jackknife deviation is below :data:`RTOL`, the tolerance of the unit witness; and
* the relative gap of the sandwich variance to the variance of the ``tight`` curve is below
  :data:`RTOL`.

Any other result is a mismatch, which the policy treats as a defect.

**The declared fit is measured and not gated.**  The ``lbfgs`` fit of the study solves the score
only to the tolerance of that solver, so a jackknife that refits to full precision departs from
the curve at that root by an amount the solver sets. A one-replicate probe showed 1.4e-3 and 2.2e-3
of the largest jackknife value for the ``study`` curve, against 1e-9 for the ``tight`` curve. The
reading therefore reports the ``study`` deviation, and the mean shift of the reported lower
standard error between the two fits, as measurements. They decide nothing. The probe came before
this paragraph, and no criterion above uses it.

The rows go to ``agreement-rows.csv`` and the verdict to ``agreement-reading.csv``.
``tests/unit/test_x26_red_cells_reading.py`` rebuilds the reading from the rows.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m tests.diagnostics.x26_red_cells.agreement --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import omitted_variable as ov
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_logistic as logistic
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent

SCENARIO = "multinomial_logistic_crossfit"
#: The parameters of the two red cells.
TOKENS = ("ate_2v0", "att_2v0")
REPLICATES = 20

#: The tolerance of the unit witnesses W5 and the sandwich, relative to the largest value.
RTOL = 1e-6
#: The weight step of the jackknife and the finite-difference step of the sandwich.
EPSILON = 1e-4
STEP = 1e-6
#: Every ``ROW_STRIDE``-th row, and the ``LARGEST`` rows of largest curve of each parameter.
ROW_STRIDE = 20
LARGEST = 5
#: The solver of the ``tight`` fit, as the unit witnesses fit.
TIGHT: dict[str, Any] = {"solver": "newton-cg", "tol": 1e-14, "max_iter": 1000}

ROW_COLUMNS = (
    "replicate",
    "parameter",
    "n",
    "jackknife_rows",
    "nu2_gap",
    "jackknife_deviation_tight",
    "jackknife_deviation_study",
    "sandwich_variance",
    "curve_variance_tight",
    "sandwich_gap",
    "curve_variance_study",
    "lower_se_study",
    "lower_se_tight",
)
READING_COLUMNS = (
    "parameter",
    "replicates",
    "jackknife_deviation_tight",
    "sandwich_gap",
    "jackknife_deviation_study",
    "mean_lower_se_shift",
    "agrees",
)


def sample_seed(replicate: int) -> int:
    """The seed of the draw that property replicate ``replicate`` of the declared run fitted."""
    return stream_seed(study.STUDY, "property_sample", "interval_calibration", SCENARIO, replicate)


def draw(replicate: int) -> pd.DataFrame:
    """The sample of replicate ``replicate``."""
    return study.draw_from_seed(SCENARIO, study.PRIMARY_N, sample_seed(replicate))[0]


def fit_tight(frame: pd.DataFrame) -> Any:
    """The declared fit of the scenario with the treatment model solved to full precision."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(C=np.inf, **TIGHT),
                simultaneous=False,
                estimands=("ate", "att", "atc"),
                random_state=0,
                cross_fit=True,
                n_folds=study.CROSSFIT_FOLDS,
                q_bounds=study.CROSSFIT_Q_BOUNDS,
                g_bounds=study.WIDE_G_BOUNDS,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )


def folds_of(result: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``(train, test)`` rows of each fold."""
    folds = result.repeats[0].nuisance.folds
    return [(np.asarray(train), np.asarray(test)) for train, test in folds]


def oracle_name(token: str) -> str:
    """The arm-index name of a parameter, for :mod:`tests.discrete_law_logistic`."""
    stem, _, rest = study.PARAMETERS[SCENARIO][token][0].partition("[")
    arm, reference = (int(float(part)) for part in rest[:-1].split(" vs "))
    return f"{stem}[{arm} vs {reference}]"


def bounds_of(result: Any, token: str) -> tuple[float, float]:
    """The truncation bounds the package applies to a parameter's representer."""
    group = study.PARAMETERS[SCENARIO][token][1]
    chosen = (
        result.config.g_bounds_conditional if group in ("att", "atc") else result.config.g_bounds
    )
    return float(chosen[0]), float(chosen[1])


def plugin_nu2(
    name: str,
    mechanism: np.ndarray,
    arms: np.ndarray,
    weights: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    """:math:`E_n[w \\alpha^2]` of the truncated ``mechanism``, with mean-one ``weights``."""
    bounded, _ = logistic.truncate(mechanism, bounds)
    conditioning = logistic.conditioning_arm(name)
    share = None if conditioning is None else float(np.mean(weights * (arms == conditioning)))
    alpha = logistic.representer(name, arms, bounded, share)
    return float(np.mean(weights * alpha**2))


def refit(
    x: np.ndarray,
    a: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
) -> np.ndarray:
    """Each fold's model refitted at ``weights``, predicting its test rows."""
    out = np.zeros((x.shape[0], 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, test in folds:
            model = LogisticRegression(C=np.inf, **TIGHT)
            model.fit(x[train], a[train], sample_weight=weights[train])
            out[test] = model.predict_proba(x[test])
    return out


def jackknife(
    x: np.ndarray,
    a: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    rows: np.ndarray,
    names: dict[str, str],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """:math:`n` times the central difference of the refitted estimate in each row's weight."""
    n = x.shape[0]
    arms = a.astype(int)
    out: dict[str, list[float]] = {token: [] for token in names}
    for row in rows:
        moved = []
        for sign in (1.0, -1.0):
            weights = np.ones(n)
            weights[row] *= 1.0 + sign * EPSILON
            weights *= n / weights.sum()
            moved.append((weights, refit(x, a, folds, weights)))
        for token, name in names.items():
            plus, minus = (plugin_nu2(name, g, arms, w, bounds[token]) for w, g in moved)
            out[token].append(n * (plus - minus) / (2.0 * EPSILON))
    return {token: np.array(values) for token, values in out.items()}


def _central(function: Callable[[np.ndarray], np.ndarray], point: np.ndarray) -> np.ndarray:
    columns = []
    for j in range(point.size):
        up, down = point.copy(), point.copy()
        up[j] += STEP
        down[j] -= STEP
        columns.append((function(up) - function(down)).reshape(-1) / (2.0 * STEP))
    return np.column_stack(columns)


def sandwich(
    x: np.ndarray,
    a: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    name: str,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """:math:`\\hat\\nu^2` and its sandwich variance, with one coefficient vector per fold.

    The stacked equations are the fold scores over their training rows, the equation of
    :math:`\\nu^2` over all rows, and the share equation of an ATT or an ATC.  The coefficients
    are the refits of :func:`refit` at unit weights.
    """
    n = x.shape[0]
    arms = a.astype(int)
    design = np.column_stack([np.ones(n), x])
    indicator = np.eye(3)[arms]
    weights = np.ones(n)
    thetas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, _ in folds:
            model = LogisticRegression(C=np.inf, **TIGHT).fit(x[train], a[train])
            full = np.column_stack([model.intercept_, model.coef_])
            thetas.append(full[1:] - full[0])
    shape = thetas[0].shape
    p = thetas[0].size
    k = len(folds)
    conditioning = logistic.conditioning_arm(name)
    in_train = np.zeros((k, n), dtype=bool)
    predicted_by = np.zeros(n, dtype=int)
    for index, (train, test) in enumerate(folds):
        in_train[index, train] = True
        predicted_by[test] = index

    def psi(parameters: np.ndarray) -> np.ndarray:
        coefficients = parameters[: k * p].reshape(k, *shape)
        columns = []
        mechanism = np.zeros((n, 3))
        for index in range(k):
            g = logistic.softmax(design, coefficients[index])
            score = np.einsum("im,ij->imj", (indicator - g)[:, 1:], design).reshape(n, -1)
            columns.append(score * in_train[index][:, None])
            mechanism[predicted_by == index] = g[predicted_by == index]
        bounded, _ = logistic.truncate(mechanism, bounds)
        share = None if conditioning is None else parameters[k * p + 1]
        alpha = logistic.representer(name, arms, bounded, share)
        columns.append((alpha**2 - parameters[k * p])[:, None])
        if conditioning is not None:
            columns.append(((arms == conditioning) - parameters[k * p + 1])[:, None])
        return weights[:, None] * np.column_stack(columns)

    share_value = None if conditioning is None else float(np.mean(arms == conditioning))
    mechanism = np.zeros((n, 3))
    for index, (_, test) in enumerate(folds):
        mechanism[test] = logistic.softmax(design, thetas[index])[test]
    nu2 = plugin_nu2(name, mechanism, arms, weights, bounds)
    pieces = [theta.reshape(-1) for theta in thetas] + [np.array([nu2])]
    if share_value is not None:
        pieces.append(np.array([share_value]))
    point = np.concatenate(pieces)
    rows = psi(point)
    if float(np.max(np.abs(rows.mean(axis=0)))) > 1e-8:
        raise RuntimeError("the refitted coefficients do not solve the stacked equations")
    bread = -_central(lambda parameters: psi(parameters).mean(axis=0), point)
    inverse = np.linalg.inv(bread)
    full_variance = inverse @ (rows.T @ rows / n) @ inverse.T / n
    return nu2, float(full_variance[k * p, k * p])


def selected_rows(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Every ``ROW_STRIDE``-th row and the ``LARGEST`` rows of largest curve of each parameter."""
    n = next(iter(curves.values())).size
    chosen = set(range(0, n, ROW_STRIDE))
    for curve in curves.values():
        chosen.update(int(row) for row in np.argsort(-np.abs(curve))[:LARGEST])
    return np.array(sorted(chosen))


def reported_standard_error(frame: pd.DataFrame, token: str, result: Any | None = None) -> float:
    """The standard error of the lower bound that the declared fit reports."""
    fitted = study.fit(frame, SCENARIO) if result is None else result
    return float(study.parameter_rows(fitted, SCENARIO, token).bounds["lower"].std_error)


def replicate_rows(replicate: int) -> list[dict[str, float | int | str]]:
    """The two comparisons of every parameter on one draw."""
    frame = draw(replicate)
    declared = study.fit(frame, SCENARIO)
    tight = fit_tight(frame)
    for (train_a, test_a), (train_b, test_b) in zip(
        folds_of(declared), folds_of(tight), strict=True
    ):
        if not (np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)):
            raise RuntimeError("the declared and the tight fit use different folds")
    x = np.asarray(tight.data.covariates, dtype=float)
    a = np.asarray(tight.data.treatment, dtype=float)
    n = x.shape[0]
    folds = folds_of(tight)
    names = {token: oracle_name(token) for token in TOKENS}
    bounds = {token: bounds_of(tight, token) for token in TOKENS}
    curves: dict[str, dict[str, np.ndarray]] = {"study": {}, "tight": {}}
    elements: dict[str, dict[str, Any]] = {"study": {}, "tight": {}}
    for label, result in (("study", declared), ("tight", tight)):
        for token in TOKENS:
            found = ov.sensitivity_elements(
                result, study.PARAMETERS[SCENARIO][token][0], nu2_estimator="plugin"
            )
            if found.psi_nu2 is None:
                raise RuntimeError(f"the plug-in curve of {token!r} is missing")
            elements[label][token] = found
            curves[label][token] = np.asarray(found.psi_nu2, dtype=float)
    rows = selected_rows(curves["tight"])
    expected = jackknife(x, a, folds, rows, names, bounds)
    out: list[dict[str, float | int | str]] = []
    for token in TOKENS:
        scale = float(np.max(np.abs(expected[token])))
        deviation = {
            label: float(np.max(np.abs(curves[label][token][rows] - expected[token])) / scale)
            for label in ("tight", "study")
        }
        nu2, variance = sandwich(x, a, folds, names[token], bounds[token])
        found = elements["tight"][token]
        raw = float(found.nu2) + float(found.nu2_correction or 0.0)
        curve_variance = float(influence_variance(curves["tight"][token], None)) * (n - 1) / n
        out.append(
            {
                "replicate": replicate,
                "parameter": token,
                "n": n,
                "jackknife_rows": int(rows.size),
                "nu2_gap": abs(raw - nu2) / abs(nu2),
                "jackknife_deviation_tight": deviation["tight"],
                "jackknife_deviation_study": deviation["study"],
                "sandwich_variance": variance,
                "curve_variance_tight": curve_variance,
                "sandwich_gap": abs(variance - curve_variance) / curve_variance,
                "curve_variance_study": float(influence_variance(curves["study"][token], None))
                * (n - 1)
                / n,
                "lower_se_study": reported_standard_error(frame, token, declared),
                "lower_se_tight": reported_standard_error(frame, token, tight),
            }
        )
    return out


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict per parameter: the largest deviation of each comparison, and whether it agrees.

    ``mean_lower_se_shift`` is the mean over replicates of the declared fit's reported lower
    standard error over the ``tight`` fit's, less one.  It measures the effect of the solver.
    """
    out = []
    for token in TOKENS:
        cell = rows.loc[rows["parameter"] == token]
        tight = float(cell["jackknife_deviation_tight"].max())
        gap = float(cell["sandwich_gap"].max())
        out.append(
            {
                "parameter": token,
                "replicates": int(cell["replicate"].nunique()),
                "jackknife_deviation_tight": tight,
                "sandwich_gap": gap,
                "jackknife_deviation_study": float(cell["jackknife_deviation_study"].max()),
                "mean_lower_se_shift": float(
                    (cell["lower_se_study"] / cell["lower_se_tight"]).mean()
                )
                - 1.0,
                "agrees": bool(tight < RTOL and gap < RTOL),
            }
        )
    return pd.DataFrame(out, columns=list(READING_COLUMNS))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    batches = map_parallel(replicate_rows, list(range(arguments.replicates)), n_jobs=arguments.jobs)
    rows = pd.DataFrame([row for batch in batches for row in batch], columns=list(ROW_COLUMNS))
    rows = rows.sort_values(["replicate", "parameter"], ignore_index=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        arguments.output / "agreement-rows.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    summary = reading(rows)
    summary.to_csv(
        arguments.output / "agreement-reading.csv",
        index=False,
        float_format="%.6g",
        lineterminator="\n",
    )
    print(summary.to_string(index=False))
    if not bool(summary["agrees"].all()):
        raise SystemExit("MISMATCH: the package curve departs from a comparator (rule 2)")


if __name__ == "__main__":
    main()
