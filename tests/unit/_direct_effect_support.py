"""The controlled-direct-effect fits that the sensitivity refusal tests share.

Every sensitivity surface refuses a fit with an intermediate variable: the E-value (RM21), the
omitted-variable bound, and the simulated-confounding replay. Their tests fit one law, the RM21
probe law, and build the two results that only ``dataclasses.replace`` can make: a level with no
intermediate column, and a column with no level. They live here once, so the surfaces are tested
on the same fits.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from cleverly.datasets import make_cde
from tests.conftest import IN_SAMPLE, fast_tmle

#: The baseline covariates of :func:`cleverly.datasets.make_cde`.
COVARIATES = ["W1", "W2", "W3"]


def cde_frame() -> Any:
    """The RM21 probe law, at the size of the fast tier."""
    frame, _ = make_cde(n=400, seed=3)
    return frame


def binary_cde_frame() -> Any:
    """The same law with the outcome split at its median."""
    frame = cde_frame()
    frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(int)
    return frame


def fit_cde(frame: Any, estimands: tuple[str, ...] | None = None, **overrides: Any) -> Any:
    """Fit a controlled direct effect in sample, and return the result at each level of ``Z``.

    ``overrides`` reach :func:`tests.conftest.fast_tmle`, for a fit that also declares an
    intervention such as a shift of a continuous dose.
    """
    return fast_tmle(**IN_SAMPLE, estimands=estimands, **overrides).fit(
        frame, outcome="Y", treatment="A", covariates=COVARIATES, intermediate="Z"
    )


def fit_without_intermediate(frame: Any, estimands: tuple[str, ...] | None = None) -> Any:
    """Fit the same frame in sample with no intermediate variable."""
    return (
        fast_tmle(**IN_SAMPLE, estimands=estimands)
        .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)
        .single()
    )


def with_intermediate_column(data: Any) -> Any:
    """The data with an intermediate column of zeros named ``Z``."""
    return replace(data, intermediate=np.zeros(data.n), intermediate_name="Z")


def column_only(result: Any) -> Any:
    """An intermediate column with no level, which only ``dataclasses.replace`` builds."""
    return replace(result, data=with_intermediate_column(result.data))


def level_only(result: Any) -> Any:
    """A level with no intermediate column, which only ``dataclasses.replace`` builds."""
    return replace(result, intermediate_value=0.0)
