r"""The ``pygformula`` fixtures of ``parametric-gformula`` (X45): laws, panels, the ``cleverly`` side
and the comparison verdicts.

``pygformula`` 1.2.0 cannot express the node-specific models of the primary laws.  It fits one
pooled-over-time model per covariate, parses bare column names and pools the residual SD over
time (``tests/canonical/parametric_gformula_study/README.md``).  So the comparison runs on small
fixture laws where the pooled model *is* the node-specific model.

=============  =======================================================================
fixture        why pooled equals node-specific
=============  =======================================================================
``cov_*``      :math:`T = 2`.  One node is simulated, so the pooled covariate fit has the
               rows of that node alone and the residual SD is the node's own.  The two
               fixtures differ in the outcome: a Gaussian mean and a binary probability
``survival``   :math:`T = 3`, no time-varying covariate.  The outcome statement is
               ``0 + C(t) + C(t):W + C(t):A + C(t):lag1_A + C(t):lag2_A``, a cell-means
               model in time, so it equals the three node-specific hazard fits
``competing``  :math:`T = 2`, one competing cause that acts at the second node only.  The
               sequential form draws the competing event first and then the event of
               interest among the survivors.  ``pygformula`` models it with a restriction
               to the second node, and ``cleverly`` fits it through a test-only hazard
               learner (:class:`SequentialHazard`)
=============  =======================================================================

**Gate.**  Each fixture compares coefficients first, the ``statsmodels`` fit of ``pygformula``
against the ``cleverly`` ``GLM`` (:data:`COEFFICIENT_RTOL`).  It then compares the estimates.
Both sides run :data:`SEEDS` seeds on the one panel, and the Monte Carlo SE of each side is the
standard deviation of its seeds over :math:`\sqrt{S}`.  An estimand passes when the difference
of the two means is inside :data:`GATE_SE` combined Monte Carlo SE.  ``pygformula`` runs at
``n_simul = n``, one simulated history per unit.  The coefficient gate reads the fits, so a
failure there is a model mismatch and not a Monte Carlo effect.

The module imports ``cleverly`` only inside the functions of the ``cleverly`` side.  The panels
and the verdicts need numpy and pandas alone.
"""

from __future__ import annotations

import importlib
import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator

#: The seeds of both sides on each panel: at least 20 (X45 plan, section R2.7).
SEEDS = tuple(range(1, 25))
#: The panel size of every fixture.
N = 2_000
#: The panel seed of each fixture: the data are drawn once, and every simulation seed reads them.
PANEL_SEED = 20264502
#: The gate: a difference of means inside this many combined Monte Carlo SE.
GATE_SE = 4.0
#: The coefficient gate.  ``atol`` only guards coefficients that are zero in both fits.
COEFFICIENT_RTOL = 1e-6
COEFFICIENT_ATOL = 1e-8
#: An estimand with no simulation noise on either side (no covariate is drawn, so the hazard
#: product is a function of the fitted models alone) must agree to rounding.
DETERMINISTIC_ATOL = 1e-10
#: The simulated histories per unit of the ``cleverly`` side: the estimator's default.
CLEVERLY_DRAWS = 10

REGIMENS = {"treat_all": 1.0, "treat_none": 0.0}
CONTRAST = "contrast"
INTERCEPT = "(intercept)"
SIGMA = "(sigma)"

# ----------------------------------------------------------------------------- the laws


@dataclass(frozen=True)
class Fixture:
    """One comparison fixture.

    Parameters
    ----------
    name : str
        The fixture name, also the ``fixture`` column of every artifact.
    kind : {"covariate", "survival", "competing"}
        The law family.
    outcome : {"gaussian", "binary", "survival"}
        The outcome of the panel.
    n_times : int
        The number of nodes.
    """

    name: str
    kind: str
    outcome: str
    n_times: int

    @property
    def horizons(self) -> tuple[int, ...]:
        """The horizons the estimates are read at: the last node for an end-of-study outcome."""
        return (self.n_times,) if self.kind == "covariate" else tuple(range(1, self.n_times + 1))


FIXTURES: dict[str, Fixture] = {
    "cov_gaussian": Fixture("cov_gaussian", "covariate", "gaussian", 2),
    "cov_binary": Fixture("cov_binary", "covariate", "binary", 2),
    "survival": Fixture("survival", "survival", "survival", 3),
    "competing": Fixture("competing", "competing", "survival", 2),
}


def _bern(rng: np.random.Generator, eta: np.ndarray) -> np.ndarray:
    return (rng.random(eta.shape[0]) < expit(eta)).astype(float)


def _covariate_panel(fixture: Fixture, n: int, seed: int) -> pd.DataFrame:
    """``T = 2``: a normal and a binary covariate per node, treatment, one outcome."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    z1 = 0.2 + 0.5 * w + rng.standard_normal(n)
    b1 = _bern(rng, -0.2 + 0.4 * w + 0.3 * z1)
    a1 = _bern(rng, -0.1 + 0.3 * w + 0.4 * z1 - 0.3 * b1)
    z2 = 0.1 + 0.4 * w + 0.5 * z1 + 0.3 * b1 + 0.4 * a1 + 0.8 * rng.standard_normal(n)
    b2 = _bern(rng, -0.3 + 0.3 * w + 0.2 * z1 + 0.4 * b1 - 0.3 * a1 + 0.5 * z2)
    a2 = _bern(rng, -0.2 + 0.5 * a1 + 0.4 * z2 + 0.3 * b2)
    mean = 0.5 * w + 0.3 * z1 - 0.2 * b1 + 0.4 * a1 + 0.5 * z2 + 0.4 * b2 + 0.6 * a2
    if fixture.outcome == "gaussian":
        y = 1.0 + mean + rng.standard_normal(n)
    else:
        y = _bern(rng, -0.5 + 0.6 * mean)
    return pd.DataFrame(
        {
            "W": w,
            "Z1": z1,
            "B1": b1,
            "A1": a1,
            "Z2": z2,
            "B2": b2,
            "A2": a2,
            "Y": y,
        }
    )


#: The hazard coefficients of ``survival`` by node: intercept, ``W``, then the treatments in order.
_SURVIVAL_HAZARD = {
    1: (-1.6, 0.4, (-0.5,)),
    2: (-1.4, 0.3, (0.2, -0.4)),
    3: (-1.2, 0.5, (-0.2, 0.3, -0.6)),
}
_SURVIVAL_TREATMENT = {1: (0.0, 0.2, ()), 2: (-0.1, 0.3, (0.5,)), 3: (0.1, 0.2, (0.3, 0.4))}


def _survival_panel(n: int, seed: int) -> pd.DataFrame:
    """``T = 3``: baseline ``W``, a treatment and a discrete event per node, no covariate."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    at_risk = np.ones(n, dtype=bool)
    columns: dict[str, np.ndarray] = {"W": w}
    treatments: list[np.ndarray] = []
    for node in (1, 2, 3):
        intercept, slope, past = _SURVIVAL_TREATMENT[node]
        eta = intercept + slope * w + sum(c * a for c, a in zip(past, treatments, strict=True))
        a_t = _bern(rng, eta)
        treatments.append(a_t)
        intercept, slope, coefficients = _SURVIVAL_HAZARD[node]
        eta = (
            intercept
            + slope * w
            + sum(c * a for c, a in zip(coefficients, treatments, strict=True))
        )
        event = _bern(rng, eta)
        columns[f"A{node}"] = np.where(at_risk, a_t, np.nan)
        columns[f"E{node}"] = np.where(at_risk, event, np.nan)
        at_risk = at_risk & (event == 0)
    return pd.DataFrame(columns)


def _competing_panel(n: int, seed: int) -> pd.DataFrame:
    """``T = 2``: the competing event ``D`` acts at node 2 and is drawn before the event ``Y``."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    a1 = _bern(rng, 0.1 + 0.3 * w)
    y1 = _bern(rng, -1.5 + 0.4 * w - 0.5 * a1)
    alive = y1 == 0
    a2 = _bern(rng, -0.2 + 0.2 * w + 0.5 * a1)
    d2 = _bern(rng, -1.8 + 0.3 * w + 0.4 * a2 - 0.3 * a1)
    y2 = _bern(rng, -1.3 + 0.3 * w - 0.5 * a2 + 0.2 * a1)
    y2 = np.where(d2 == 1, 0.0, y2)
    return pd.DataFrame(
        {
            "W": w,
            "A1": a1,
            "Y1": y1,
            "D1": np.zeros(n),
            "A2": np.where(alive, a2, np.nan),
            "Y2": np.where(alive, y2, np.nan),
            "D2": np.where(alive, d2, np.nan),
        }
    )


def sample(fixture: Fixture | str, n: int = N, seed: int = PANEL_SEED) -> pd.DataFrame:
    """The wide panel of a fixture, the form ``cleverly`` reads."""
    fixture = FIXTURES[fixture] if isinstance(fixture, str) else fixture
    if fixture.kind == "covariate":
        return _covariate_panel(fixture, n, seed)
    if fixture.kind == "survival":
        return _survival_panel(n, seed)
    return _competing_panel(n, seed)


# ----------------------------------------------------------------- the long form for pygformula

#: The columns of the long panel; a fixture leaves the others empty.
LONG_COLUMNS = ("fixture", "id", "t", "W", "Z", "B", "A", "Y", "D")


def to_long(fixture: Fixture | str, wide: pd.DataFrame) -> pd.DataFrame:
    """The long panel ``pygformula`` reads: one row per unit and time, ``t`` counted from zero.

    An end-of-study outcome sits on the last row only.  A discrete event ends a unit's rows at
    the node of the event.  The competing event masks the event of interest at its node.
    """
    fixture = FIXTURES[fixture] if isinstance(fixture, str) else fixture
    n = len(wide)
    frames: list[pd.DataFrame] = []
    for t in range(fixture.n_times):
        node = t + 1
        row = pd.DataFrame({"fixture": fixture.name, "id": np.arange(n), "t": t, "W": wide["W"]})
        if fixture.kind == "covariate":
            row["Z"] = wide[f"Z{node}"].to_numpy()
            row["B"] = wide[f"B{node}"].to_numpy()
            row["A"] = wide[f"A{node}"].to_numpy()
            row["Y"] = wide["Y"].to_numpy() if node == fixture.n_times else np.nan
        elif fixture.kind == "survival":
            row["A"] = wide[f"A{node}"].to_numpy()
            row["Y"] = wide[f"E{node}"].to_numpy()
            row = row.loc[np.isfinite(row["Y"])]
        else:
            row["A"] = wide[f"A{node}"].to_numpy()
            row["D"] = wide[f"D{node}"].to_numpy()
            event = wide[f"Y{node}"].to_numpy().copy()
            event[row["D"].to_numpy() == 1] = np.nan
            row["Y"] = event
            row = row.loc[np.isfinite(row["D"])]
        frames.append(row)
    long = pd.concat(frames, ignore_index=True).sort_values(["id", "t"], ignore_index=True)
    return long.reindex(columns=list(LONG_COLUMNS))


# ------------------------------------------------------------------------------ the cleverly side


def table_names(fixture: Fixture) -> tuple[str, ...]:
    """The design-table columns of a fixture, in the estimator's order (baseline, then nodes)."""
    names = ["W"]
    for node in range(1, fixture.n_times + 1):
        if fixture.kind == "covariate":
            names += [f"Z{node}", f"B{node}"]
        names.append(f"A{node}")
    return tuple(names)


def fit_columns(fixture: Fixture) -> dict[str, Any]:
    """The node declaration of a ``cleverly`` fit."""
    times = range(1, fixture.n_times + 1)
    columns: dict[str, Any] = {
        "treatment": [f"A{t}" for t in times],
        "baseline": ["W"],
        "time_varying": [[f"Z{t}", f"B{t}"] if fixture.kind == "covariate" else [] for t in times],
    }
    if fixture.kind == "covariate":
        columns["outcome"] = "Y"
    elif fixture.kind == "survival":
        columns["outcome"] = [f"E{t}" for t in times]
    else:
        columns["outcome"] = {
            "relapse": [f"Y{t}" for t in times],
            "death": [f"D{t}" for t in times],
        }
    return columns


def estimator(fixture: Fixture, seed: int) -> Any:
    """The ``ParametricGFormula`` of a fixture, with explicit learners."""
    from cleverly.learners import GLM
    from cleverly.longitudinal import ParametricGFormula

    options: dict[str, Any] = {
        "n_draws": CLEVERLY_DRAWS,
        "treatment_learner": GLM(),
        "simultaneous": False,
        "random_state": seed,
        "reference": "treat_none",
    }
    if fixture.kind == "covariate":
        options["covariate_types"] = {"Z2": "normal"}
        options["outcome_learner"] = GLM(
            family="gaussian" if fixture.outcome == "gaussian" else "binomial"
        )
    elif fixture.kind == "survival":
        options["horizons"] = list(fixture.horizons)
        options["hazard_learner"] = GLM()
    else:
        options["horizons"] = list(fixture.horizons)
        options["hazard_learner"] = SequentialHazard()
    return ParametricGFormula(dict(REGIMENS), **options)


def _estimand_names(fixture: Fixture) -> dict[tuple[str, int], str]:
    """``(regimen or contrast, horizon) -> the cleverly parameter name``."""
    out: dict[tuple[str, int], str] = {}
    for horizon in fixture.horizons:
        if fixture.kind == "covariate":
            where, cause = "", ""
        elif fixture.kind == "survival":
            where, cause = f" @ t={horizon}", ""
        else:
            where, cause = f" @ t={horizon}", ", relapse"
        for label in REGIMENS:
            if fixture.kind == "covariate":
                out[(label, horizon)] = f"ey_regimen[{label}]"
            elif fixture.kind == "survival":
                out[(label, horizon)] = f"risk_regimen[{label}{where}]"
            else:
                out[(label, horizon)] = f"cif_regimen[{label}{cause}{where}]"
        out[(CONTRAST, horizon)] = f"ate_regimen[treat_all vs treat_none{cause}{where}]"
    return out


def cleverly_estimates(fixture: Fixture | str, wide: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One ``cleverly`` fit: the estimate of each regimen and the contrast at each horizon.

    Returns
    -------
    pandas.DataFrame
        The columns ``fixture``, ``estimand``, ``horizon``, ``seed`` and ``value``.
    """
    fixture = FIXTURES[fixture] if isinstance(fixture, str) else fixture
    with warnings.catch_warnings():
        # A non-GLM hazard learner withholds the interval, which the comparison does not read.
        warnings.simplefilter("ignore")
        result = estimator(fixture, seed).fit(wide, **fit_columns(fixture))
    rows = []
    for (label, horizon), name in _estimand_names(fixture).items():
        rows.append(
            {
                "fixture": fixture.name,
                "estimand": label,
                "horizon": horizon,
                "seed": seed,
                "value": float(result[name].psi),
            }
        )
    return pd.DataFrame(rows)


def cleverly_coefficients(fixture: Fixture | str, wide: pd.DataFrame) -> pd.DataFrame:
    """The coefficients of every model the ``cleverly`` side fits, by canonical name.

    The fit stops after the models, with the exact-terms hook the design test uses
    (``tests/unit/test_parametric_gformula_design.py``), so no simulation runs.  A model is named
    ``<variable>@<node>`` (``Z@2`` for the column ``Z2``) and a term by its design column.  The residual SD of a Gaussian model is
    the term :data:`SIGMA`.  An aliased column reads zero, as it does in ``pygformula``'s
    minimum-norm solution.  The outcome reads on its own scale, not the scaled one it is fitted on.

    Returns
    -------
    pandas.DataFrame
        The columns ``fixture``, ``model``, ``term`` and ``value``.
    """
    gf = importlib.import_module("cleverly.longitudinal.gformula")

    fixture = FIXTURES[fixture] if isinstance(fixture, str) else fixture
    store: dict[str, Any] = {}

    def hook(book: Any, plan: Any, cells: Any) -> Any:
        store["book"] = book
        return gf.ExactTerms(
            np.zeros((book.data.n, len(cells))),
            {m.key: np.zeros((len(cells), m.n_parameters)) for m in book.stacked},
        )

    model = estimator(fixture, 0)
    model._exact = hook
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(wide, **fit_columns(fixture))
    book = store["book"]
    names = table_names(fixture)
    if book.table.shape[1] != len(names):
        raise RuntimeError(
            f"{fixture.name}: the design table has {book.table.shape[1]} columns, not {len(names)}"
        )
    rows = [
        {"fixture": fixture.name, "model": label, "term": term, "value": float(value)}
        for label, coefficients in _book_coefficients(book, fixture, names)
        for term, value in coefficients.items()
    ]
    return pd.DataFrame(rows, columns=["fixture", "model", "term", "value"])


def _glm_coefficients(glm: Any, columns: Sequence[str]) -> dict[str, float]:
    """A fitted ``GLM``'s coefficients by design column: the intercept, then ``columns``."""
    terms = [INTERCEPT, *columns]
    active = np.asarray(glm.active_, dtype=bool)
    values = np.zeros(len(terms))
    values[active] = np.asarray(glm.coef_, dtype=float).reshape(-1)
    return dict(zip(terms, values.tolist(), strict=True))


def _book_coefficients(
    book: Any, fixture: Fixture, names: Sequence[str]
) -> list[tuple[str, dict[str, float]]]:
    """The ``(model label, coefficients)`` pairs of a fitted model book."""
    out: list[tuple[str, dict[str, float]]] = []
    for fitted in book.models:
        columns = list(names[: fitted.prefix])
        role, _, rest = fitted.key.partition(":")
        if role == "covariate":
            node, _, column = rest.partition(":")
            variable = column.removesuffix(node)
            coefficients = _glm_coefficients(fitted.learner, columns)
            if fitted.response == "normal":
                coefficients[SIGMA] = math.sqrt(fitted.sigma2)
            out.append((f"{variable}@{node}", coefficients))
        elif role == "outcome":
            # Fitted on ``(Y - lower) / range``; the comparison reads the data scale.
            coefficients = _glm_coefficients(fitted.learner, columns)
            scaled = {
                term: value * float(book.scaler.range) for term, value in coefficients.items()
            }
            scaled[INTERCEPT] += float(book.scaler.lower)
            out.append(("Y@2", scaled))
        elif role == "hazard":
            learner = fitted.learner
            if isinstance(learner, SequentialHazard):
                for label, glm in (("D", learner.competing_), ("Y", learner.interest_)):
                    if not isinstance(glm, float):
                        out.append((f"{label}@{rest}", _glm_coefficients(glm, columns)))
            else:
                out.append((f"E@{rest}", _glm_coefficients(learner, columns)))
    return out


class SequentialHazard(BaseEstimator):
    r"""A test-only hazard learner for the sequential form of a competing cause.

    The hazard of node :math:`t` has three classes: 0 for no event, 1 for the event of interest
    and 2 for the competing event.  The sequential form draws the competing event first,
    :math:`q = P(D = 1 \mid x)`, and the event of interest among the survivors,
    :math:`p = P(Y = 1 \mid D = 0, x)`.  The class probabilities are
    :math:`((1 - q)(1 - p), (1 - q) p, q)`.  Each of :math:`q` and :math:`p` is a
    :class:`~cleverly.learners.GLM`.

    ``pygformula`` fits and simulates this form, and the estimator's default hazard model is the
    baseline-category multinomial instead.  So this learner exists for the comparison only.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> SequentialHazard:
        from cleverly.learners import GLM

        labels = np.asarray(y, dtype=float)
        weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.array([0.0, 1.0, 2.0])
        self.competing_: GLM | float
        self.interest_: GLM | float
        died = labels == 2.0
        self.competing_ = _binary_fit(
            GLM, x, died.astype(float), weights, np.ones(len(labels), bool)
        )
        survivors = ~died
        self.interest_ = _binary_fit(GLM, x, (labels == 1.0).astype(float), weights, survivors)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        q = _binary_predict(self.competing_, x)
        p = _binary_predict(self.interest_, x)
        return np.column_stack([(1.0 - q) * (1.0 - p), (1.0 - q) * p, q])


def _binary_fit(
    glm: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray, rows: np.ndarray
) -> Any:
    """A binomial ``GLM`` on ``rows``, or the constant 0 or 1 when the rows hold one class."""
    values = y[rows]
    if np.all(values == values[0]):
        return float(values[0])
    return glm(family="binomial").fit(x[rows], values, weights[rows])


def _binary_predict(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(x.shape[0], model)
    return np.asarray(model.predict(x), dtype=float)


# ------------------------------------------------------------------------------- the verdicts

COEFFICIENT_COLUMNS = ("fixture", "model", "term", "cleverly", "pygformula", "abs_error", "passed")
ESTIMATE_COLUMNS = (
    "fixture",
    "estimand",
    "horizon",
    "seeds",
    "cleverly_mean",
    "pygformula_mean",
    "cleverly_mc_se",
    "pygformula_mc_se",
    "difference",
    "combined_mc_se",
    "ratio",
    "gate",
    "deterministic",
    "passed",
)


def coefficient_rows(cleverly: pd.DataFrame, pygformula: pd.DataFrame) -> pd.DataFrame:
    """The coefficient comparison: every model of both sides, joined on ``(fixture, model, term)``.

    A model or term that only one side holds is a failed row, not a skipped one.  The residual
    SD is one more term of its model.
    """
    keys = ["fixture", "model", "term"]
    merged = cleverly.rename(columns={"value": "cleverly"}).merge(
        pygformula.rename(columns={"value": "pygformula"}), on=keys, how="outer"
    )
    error = (merged["cleverly"] - merged["pygformula"]).abs()
    scale = np.maximum(merged["cleverly"].abs(), merged["pygformula"].abs())
    merged["abs_error"] = error
    merged["passed"] = (error <= COEFFICIENT_ATOL + COEFFICIENT_RTOL * scale).fillna(False)
    return merged.loc[:, list(COEFFICIENT_COLUMNS)].sort_values(keys, ignore_index=True)


def estimate_rows(cleverly: pd.DataFrame, pygformula: pd.DataFrame) -> pd.DataFrame:
    """The estimate comparison per fixture, estimand and horizon, gated at :data:`GATE_SE`."""
    contrast = lambda frame: _with_contrast(frame)  # noqa: E731
    sides = {"cleverly": contrast(cleverly), "pygformula": contrast(pygformula)}
    keys = ["fixture", "estimand", "horizon"]
    summary = {}
    for name, frame in sides.items():
        grouped = frame.groupby(keys)["value"]
        summary[name] = pd.DataFrame(
            {
                "seeds": grouped.size(),
                "mean": grouped.mean(),
                "mc_se": grouped.std(ddof=1) / np.sqrt(grouped.size()),
            }
        )
    joined = summary["cleverly"].join(
        summary["pygformula"], lsuffix="_c", rsuffix="_p", how="outer"
    )
    rows = []
    for key, row in joined.iterrows():
        difference = row["mean_c"] - row["mean_p"]
        combined = math.sqrt(row["mc_se_c"] ** 2 + row["mc_se_p"] ** 2)
        seeds = min(row["seeds_c"], row["seeds_p"])
        deterministic = combined == 0.0
        if deterministic:
            ratio = 0.0 if abs(difference) <= DETERMINISTIC_ATOL else math.inf
        else:
            ratio = abs(difference) / combined
        passed = bool(seeds >= 20 and ratio <= GATE_SE)
        rows.append(
            {
                "fixture": key[0],
                "estimand": key[1],
                "horizon": int(key[2]),
                "seeds": int(seeds),
                "cleverly_mean": row["mean_c"],
                "pygformula_mean": row["mean_p"],
                "cleverly_mc_se": row["mc_se_c"],
                "pygformula_mc_se": row["mc_se_p"],
                "difference": difference,
                "combined_mc_se": combined,
                "ratio": ratio,
                "gate": GATE_SE,
                "deterministic": deterministic,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows, columns=list(ESTIMATE_COLUMNS))


def _with_contrast(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the per-seed contrast ``treat_all - treat_none`` to a table of regimen estimates."""
    keys = ["fixture", "horizon", "seed"]
    wide = frame.loc[frame["estimand"].isin(REGIMENS)].pivot_table(
        index=keys, columns="estimand", values="value"
    )
    contrast = (wide["treat_all"] - wide["treat_none"]).rename("value").reset_index()
    contrast["estimand"] = CONTRAST
    return pd.concat(
        [frame.loc[frame["estimand"].isin(REGIMENS)], contrast.loc[:, list(frame.columns)]],
        ignore_index=True,
    )


def findings(coefficients: pd.DataFrame, estimates: pd.DataFrame) -> list[str]:
    """Every failed row, as one line each; an empty list means the comparator agrees."""
    found = [
        f"coefficient {row.fixture}/{row.model}/{row.term}: cleverly {row.cleverly}, "
        f"pygformula {row.pygformula}"
        for row in coefficients.loc[~coefficients["passed"]].itertuples()
    ]
    found += [
        f"estimate {row.fixture}/{row.estimand}@{row.horizon}: difference {row.difference:.5f} is "
        f"{row.ratio:.2f} combined Monte Carlo SE (gate {row.gate:g}, {row.seeds} seeds)"
        for row in estimates.loc[~estimates["passed"]].itertuples()
    ]
    return found


# ---------------------------------------------------------------------------- the file contract

PANELS = "samples.csv.gz"
SPEC = "spec.json"
PYGFORMULA_RESULTS = "pygformula-results.csv"
PYGFORMULA_COLUMNS = ("fixture", "kind", "model", "term", "estimand", "horizon", "seed", "value")


def write_inputs(directory: Path, names: Sequence[str] | None = None) -> tuple[Path, Path]:
    """Write the long panels and the spec the runner reads; return their paths.

    The runner takes three paths, as every mounted runner does: the panels, the spec (where a
    runner of another study takes truths) and its output.
    """
    chosen = list(FIXTURES) if names is None else list(names)
    directory.mkdir(parents=True, exist_ok=True)
    panels = pd.concat([to_long(name, sample(name)) for name in chosen], ignore_index=True)
    samples = directory / PANELS
    panels.to_csv(
        samples, index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0}
    )
    spec = directory / SPEC
    spec.write_text(
        json.dumps({"fixtures": chosen, "seeds": list(SEEDS)}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return samples, spec


def read_pygformula(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The runner's coefficient and estimate tables."""
    table = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if tuple(table.columns) != PYGFORMULA_COLUMNS:
        raise ValueError(f"{path}: columns {tuple(table.columns)}, expected {PYGFORMULA_COLUMNS}")
    coefficients = table.loc[table["kind"] == "coefficient", ["fixture", "model", "term", "value"]]
    estimates = table.loc[
        table["kind"] == "estimate", ["fixture", "estimand", "horizon", "seed", "value"]
    ]
    return coefficients.reset_index(drop=True), estimates.reset_index(drop=True)


def cleverly_side(
    names: Sequence[str] | None = None,
    seeds: Sequence[int] = SEEDS,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every ``cleverly`` coefficient and per-seed estimate of the chosen fixtures."""
    coefficients, estimates = [], []
    for name in FIXTURES if names is None else names:
        wide = sample(name)
        coefficients.append(cleverly_coefficients(name, wide))
        for seed in seeds:
            estimates.append(cleverly_estimates(name, wide, seed))
        if progress is not None:
            progress(name)
    return pd.concat(coefficients, ignore_index=True), pd.concat(estimates, ignore_index=True)
