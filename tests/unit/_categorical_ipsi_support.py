"""Shared fits and readers for the categorical longitudinal incremental tests.

The fits run on the exact three-level laws of
:mod:`tests.discrete_law_longitudinal_categorical_incremental`.  A saturated learner returns the
law's own nuisances, so the reported estimate and curve are the functional and its efficient
influence function.  The test files import this module and not each other.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI
from tests import discrete_law_longitudinal as binary
from tests import discrete_law_longitudinal_categorical_incremental as law
from tests import discrete_law_longitudinal_ipsi as binary_law
from tests import discrete_law_longitudinal_multivalue as multi
from tests.longitudinal_ipsi import COLUMNS as BINARY_COLUMNS
from tests.longitudinal_ipsi import regimens as binary_regimens

POINT = 1e-12
CURVE = 1e-10
NO_TRUNCATION = (1e-8, 1.0)
CellMeans = binary.CellMeans

END_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
UNCENSORED_COLUMNS: dict[str, Any] = {
    key: value for key, value in END_COLUMNS.items() if key != "censoring"
}
EVENT_COLUMNS: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
SURVIVAL_COLUMNS: dict[str, Any] = {"outcome": ["Y1", "Y2"], **EVENT_COLUMNS}


def competing_columns() -> dict[str, Any]:
    """Columns of a competing-risk fit."""
    return {"outcome": law.competing_outcome_columns(), **EVENT_COLUMNS}


def node(entry: tuple[str, Any]) -> Any:
    """The estimator side of one oracle node: an ``Incremental`` or the arm's label."""
    kind, value = entry
    if kind == "tilt":
        return Incremental(dict(value))
    return law.ARM_LABELS[int(value)]


def regimens(plans: Mapping[str, tuple[Any, ...]]) -> dict[str, Any]:
    """``regimens=`` for an :class:`~cleverly.longitudinal.IPSI` fit of the oracle's ``plans``."""
    return {label: tuple(node(entry) for entry in plan) for label, plan in plans.items()}


def fit_ipsi(
    frame: Any,
    plans: Any,
    columns: dict[str, Any] = END_COLUMNS,
    *,
    declared: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """An in-sample fit with the saturated learner at every node.

    ``declared`` holds keywords of ``fit``: weights and known mechanisms.
    """
    settings: dict[str, Any] = {
        "reference": law.REFERENCE,
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "treatment_learner": multi.CellProbabilities(),
        "censoring_learner": CellMeans() if "censoring" in columns else None,
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IPSI(regimens(plans), **settings).fit(frame, **columns, **(declared or {}))


Expected = list[tuple[str, float, np.ndarray, np.ndarray]]


def expected(
    names: tuple[str, ...],
    function: Callable[[str], Any],
    probs: np.ndarray,
    counts: np.ndarray,
) -> Expected:
    """``(name, truth, curve at the support points with mass, first row of each)``."""
    rows = law.first_rows(counts)
    keep = probs > 0
    out = []
    for name in names:
        at = function(name)
        out.append((name, float(at(probs)), law.eif(at, probs)[keep], rows[keep]))
    return out


def binary_fit(declaration: Any) -> Any:
    """The binary tilt law, every node tilted by one declaration of an odds multiplier."""
    plans = {
        "natural": binary_law.PLANS["natural"],
        "tilted": binary_law.PLANS["natural"],
    }
    regimen = binary_regimens(plans)
    regimen["tilted"] = (Incremental(declaration), Incremental(declaration))
    settings: dict[str, Any] = {
        "reference": "natural",
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "treatment_learner": CellMeans(),
        "censoring_learner": CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IPSI(regimen, **settings).fit(binary_law.tilt_frame(), **BINARY_COLUMNS)


#: The labels as the data layer sorts them.
SORTED: tuple[str, ...] = tuple(sorted(law.ARM_LABELS))

#: ``fit`` keywords that declare the law's own mechanism at both nodes of :func:`known_frame`.
KNOWN_DECLARED: dict[str, Any] = {
    "treatment_probabilities": {
        "A1": {label: f"g1_{label}" for label in SORTED},
        "A2": {label: f"g2_{label}" for label in SORTED},
    }
}


def known_frame() -> pd.DataFrame:
    """The sample with the law's true mechanism as probability columns, in sorted-label order."""
    frame = law.uncensored_frame()
    tables = law.UNCENSORED_TABLES
    w = frame["W"].to_numpy(dtype=int)
    l2 = frame["L2"].to_numpy(dtype=int)
    raw1 = frame["A1"].map(law.ARM_LABELS.index).to_numpy(dtype=int)
    g1 = tables["G1"][w]
    g2 = tables["G2"][w, raw1, l2]
    order = [law.ARM_LABELS.index(label) for label in SORTED]
    for label, position in zip(SORTED, order, strict=True):
        frame[f"g1_{label}"] = g1[:, position]
        frame[f"g2_{label}"] = g2[:, position]
    return frame
