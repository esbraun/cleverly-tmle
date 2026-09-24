"""Which fits the sensitivity surfaces read as a controlled direct effect.

Roadmap row RM21 refuses the E-value on a fit with an intermediate variable. The omitted-variable
bound, the simulated-confounding replay, and the derived risk ratio already refuse that fit, and
each surface wrote the test itself. :func:`~cleverly.estimators.direct_effect.declares_intermediate`
is now the one predicate, so the surfaces cannot drift apart on which fits they refuse.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_cde
from cleverly.estimators import direct_effect
from cleverly.estimators.direct_effect import declares_intermediate
from tests.conftest import IN_SAMPLE, fast_tmle

pytestmark = pytest.mark.xdist_group("evalue_direct_effect")

COVARIATES = ["W1", "W2", "W3"]

#: Each surface that refuses a fit with an intermediate variable. Imported by name, because
#: ``cleverly.sensitivity`` rebinds some of its submodule names to functions.
SURFACES = (
    "cleverly.sensitivity._derived",
    "cleverly.sensitivity.omitted_variable",
    "cleverly.sensitivity._simulated_confounding_request",
)


def binary_frame() -> Any:
    """The RM21 law, with the outcome split at its median."""
    frame, _ = make_cde(n=400, seed=3)
    frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(int)
    return frame


@pytest.fixture(scope="module")
def ratio_cde() -> Any:
    """The controlled direct effect at ``Z = 0``."""
    estimator = fast_tmle(**IN_SAMPLE, estimands=("ate", "rr", "or"))
    return estimator.fit(
        binary_frame(), outcome="Y", treatment="A", covariates=COVARIATES, intermediate="Z"
    )[0.0]


@pytest.fixture(scope="module")
def ratio_plain() -> Any:
    """The same law fitted without the intermediate variable."""
    estimator = fast_tmle(**IN_SAMPLE, estimands=("ate", "rr", "or"))
    return estimator.fit(binary_frame(), outcome="Y", treatment="A", covariates=COVARIATES).single()


def level_only(result: Any) -> Any:
    """A level with no intermediate column, which only ``dataclasses.replace`` builds."""
    return replace(result, intermediate_value=0.0)


def column_only(result: Any) -> Any:
    """An intermediate column with no level, which only ``dataclasses.replace`` builds."""
    column = np.zeros(result.data.n)
    return replace(result, data=replace(result.data, intermediate=column, intermediate_name="Z"))


class TestOnePredicate:
    def test_the_predicate_reads_either_half(self, ratio_cde: Any, ratio_plain: Any) -> None:
        assert declares_intermediate(ratio_cde)
        assert not declares_intermediate(ratio_plain)
        assert declares_intermediate(level_only(ratio_plain))
        assert declares_intermediate(column_only(ratio_plain))

    @pytest.mark.parametrize("module", SURFACES)
    def test_every_sensitivity_surface_reads_one_predicate(self, module: str) -> None:
        surface = importlib.import_module(module)
        assert surface.declares_intermediate is direct_effect.declares_intermediate
