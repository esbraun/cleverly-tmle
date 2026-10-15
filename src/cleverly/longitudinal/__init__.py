"""Longitudinal TMLE: time-varying treatment, time-varying confounding, censoring.

>>> from cleverly.longitudinal import LTMLE
>>> from cleverly.datasets import make_longitudinal
>>> from sklearn.linear_model import LinearRegression, LogisticRegression
>>> frame, _ = make_longitudinal(n=200, seed=0)
>>> result = LTMLE(
...     {"always": 1, "never": 0},
...     n_folds=2,
...     random_state=0,
...     outcome_learner=LinearRegression(),
...     treatment_learner=LogisticRegression(max_iter=1000),
...     censoring_learner=LogisticRegression(max_iter=1000),
... ).fit(
...     frame,
...     outcome="Y",
...     treatment=["A1", "A2"],
...     baseline=["W1", "W2"],
...     time_varying=[[], ["L2"]],
...     censoring=["C1", "C2"],
... )
>>> sorted(result.estimates)
['ate_regimen[never vs always]', 'ey_regimen[always]', 'ey_regimen[never]']

See :mod:`cleverly.longitudinal.sequential` for the recursion and the influence
function, and :mod:`cleverly.longitudinal.estimator` for the assumptions it identifies
the parameter under.
"""

from __future__ import annotations

from ..exceptions import LongitudinalError
from .data import LongitudinalData, Repeated
from .estimator import (
    IPSI,
    LTMLE,
    SDR,
    LongitudinalConfig,
    LongitudinalResult,
    SequentialGComputation,
    SequentialIPW,
    gcomp,
    ipsi,
    ipw,
    ltmle,
    sdr,
)
from .panel import NodeRule, PanelLayout, each_node
from .regimen import (
    DynamicRegimen,
    Plan,
    Regimen,
    RegimenSpec,
    resolve_plans,
    resolve_regimens,
)
from .sequential import (
    Mechanism,
    RegimenFit,
    SequentialStep,
    fit_mechanism,
    fit_regimen,
)
from .structural import (
    StructuralModelCheck,
    StructuralNestedMean,
    StructuralNestedMeanResult,
    snmm,
)

__all__ = [
    "IPSI",
    "LTMLE",
    "SDR",
    "DynamicRegimen",
    "LongitudinalConfig",
    "LongitudinalData",
    "LongitudinalError",
    "LongitudinalResult",
    "Mechanism",
    "NodeRule",
    "PanelLayout",
    "Plan",
    "Regimen",
    "RegimenFit",
    "RegimenSpec",
    "Repeated",
    "SequentialGComputation",
    "SequentialIPW",
    "SequentialStep",
    "StructuralModelCheck",
    "StructuralNestedMean",
    "StructuralNestedMeanResult",
    "each_node",
    "fit_mechanism",
    "fit_regimen",
    "gcomp",
    "ipsi",
    "ipw",
    "ltmle",
    "resolve_plans",
    "resolve_regimens",
    "sdr",
    "snmm",
]
