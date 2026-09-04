"""The point and longitudinal constructors share one observation-weight contract."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.data import CausalData
from cleverly.data.weighting import warn_if_concentrated, warn_if_counts
from cleverly.exceptions import DataError, WeightingWarning
from cleverly.longitudinal import LongitudinalData


def _frame(weights: np.ndarray | None) -> pd.DataFrame:
    frame = pd.DataFrame({"W": np.arange(20.0), "A": np.tile([0, 1], 10), "Y": np.tile([1, 0], 10)})
    if weights is not None:
        frame["mass"] = weights
    return frame


def _construct(kind: str, frame: pd.DataFrame, **kwargs: Any) -> CausalData | LongitudinalData:
    if kind == "point":
        return CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"], **kwargs)
    return LongitudinalData.from_frame(
        frame, outcome="Y", treatment=["A"], baseline=["W"], **kwargs
    )


@pytest.mark.parametrize("weights", [None, np.full(20, 2.0), np.linspace(0.0, 3.0, 20)])
@pytest.mark.parametrize("alias", ["probability", " Sampling ", "pweight"])
@pytest.mark.parametrize("estimated", [True, False])
def test_constructors_preserve_normalization_and_weight_metadata(weights, alias, estimated) -> None:
    frame = _frame(weights)
    options = {"weights_type": alias, "weights_estimated": estimated}
    if weights is not None:
        options["weights"] = "mass"
    point = _construct("point", frame, **options)
    longitudinal = _construct("longitudinal", frame, **options)
    expected = np.ones(20) if weights is None else weights * (20 / weights.sum())
    np.testing.assert_array_equal(point.weights, expected)
    np.testing.assert_array_equal(longitudinal.weights, expected)
    assert point.weight_spec == longitudinal.weight_spec
    assert point.weight_spec.kind == "probability"
    # Both values, and the exact one: a truthiness check on a flag that is always ``True``
    # would pass against a container that hardcoded it.
    assert point.weight_spec.estimated is estimated
    assert point.weight_spec.name == (None if weights is None else "mass")
    assert point.weight_spec.scale == (1.0 if weights is None else float(weights.mean()))


# One heavy row, eighteen light ones and one negative: the design effect is about 19.3, so
# ``warn_if_concentrated`` fires on this vector, while ``check_weights`` refuses it for the
# negative entry.  Every other row below is silent under both emitters, so ``caught == []``
# holds there whatever the order is.  This row is the one that can tell the order apart.
REFUSED_BUT_WOULD_WARN = np.array([1000.0] + [1.0] * 18 + [-1.0])


def test_the_ordering_witness_does_warn_when_it_is_not_refused() -> None:
    """The row below is only a witness if the emitter really fires on it."""
    with pytest.warns(WeightingWarning, match="are concentrated"):
        warn_if_concentrated(REFUSED_BUT_WOULD_WARN, "mass")


@pytest.mark.parametrize("kind", ["point", "longitudinal"])
@pytest.mark.parametrize(
    ("weights", "weight_type", "error"),
    [
        (np.full(20, -1.0), "unknown", "unknown weights_type"),
        (np.full(20, -1.0), "frequency", "frequency"),
        (np.full(20, np.nan), "probability", "missing or non-finite"),
        (np.full(20, -1.0), "probability", "negative"),
        (np.zeros(20), "probability", "sums to zero"),
        (REFUSED_BUT_WOULD_WARN, "probability", "negative"),
    ],
)
def test_invalid_weights_fail_before_warnings(kind, weights, weight_type, error) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(DataError, match=error):
            _construct(kind, _frame(weights), weights="mass", weights_type=weight_type)
    assert caught == []


def _build_call_location(constructor) -> tuple[str, int]:
    lines, start = inspect.getsourcelines(constructor)
    offset = next(i for i, line in enumerate(lines) if "return cls._build(" in line)
    return str(Path(inspect.getfile(constructor)).resolve()), start + offset


def test_weight_warnings_keep_order_messages_and_constructor_locations() -> None:
    frame = _frame(np.array([1000.0] + [1.0] * 19))
    messages = []
    for kind, constructor in [
        ("point", CausalData.from_frame),
        ("longitudinal", LongitudinalData.from_frame),
    ]:
        with pytest.warns(WeightingWarning) as caught:
            _construct(kind, frame, weights="mass")
        assert len(caught) == 2
        current = [str(item.message) for item in caught]
        assert current[0].startswith("mass are whole numbers averaging")
        assert current[1].startswith("mass are concentrated:")
        assert all(item.category is WeightingWarning for item in caught)
        assert [(str(Path(item.filename).resolve()), item.lineno) for item in caught] == [
            _build_call_location(constructor)
        ] * 2
        messages.append(current)
    assert messages[0] == messages[1]


def test_array_constructor_keeps_its_warning_location() -> None:
    frame = _frame(np.array([1000.0] + [1.0] * 19))
    with pytest.warns(WeightingWarning) as caught:
        CausalData.from_arrays(
            frame["Y"].to_numpy(),
            frame["A"].to_numpy(),
            frame[["W"]].to_numpy(),
            weights=frame["mass"].to_numpy(),
        )
    assert len(caught) == 2
    assert [(str(Path(item.filename).resolve()), item.lineno) for item in caught] == [
        _build_call_location(CausalData.from_arrays)
    ] * 2


def _emit_warning(emitter, **options) -> None:
    emitter(np.array([1000.0] + [1.0] * 19), "mass", **options)


def _emit_one_frame_deeper(emitter, **options) -> None:
    _emit_warning(emitter, **options)


@pytest.mark.parametrize("emitter", [warn_if_counts, warn_if_concentrated])
def test_public_warning_helpers_keep_their_calling_convention(emitter) -> None:
    signature = inspect.signature(emitter)
    assert list(signature.parameters) == ["weights", "name", "stacklevel"]
    assert signature.parameters["stacklevel"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["stacklevel"].default == 3
    with pytest.warns(WeightingWarning) as caught:
        line = inspect.currentframe().f_lineno + 1
        _emit_warning(emitter)
    assert len(caught) == 1
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert caught[0].lineno == line


@pytest.mark.parametrize("emitter", [warn_if_counts, warn_if_concentrated])
def test_one_more_stacklevel_blames_exactly_one_frame_further_out(emitter) -> None:
    """What ``_prepare_weights`` relies on when it passes ``4`` rather than the default.

    The default blames the frame that called the helper's caller. ``_prepare_weights``
    sits one frame deeper than a direct call, so it asks for one level more and the
    constructor's ``_build`` call is blamed instead of ``_prepare_weights`` itself.
    """
    with pytest.warns(WeightingWarning) as default:
        _emit_one_frame_deeper(emitter)
    with pytest.warns(WeightingWarning) as deeper:
        line = inspect.currentframe().f_lineno + 1
        _emit_one_frame_deeper(emitter, stacklevel=4)
    lines, start = inspect.getsourcelines(_emit_one_frame_deeper)
    inner_call = start + next(i for i, text in enumerate(lines) if "_emit_warning(" in text)
    assert Path(default[0].filename).resolve() == Path(__file__).resolve()
    assert default[0].lineno == inner_call
    assert Path(deeper[0].filename).resolve() == Path(__file__).resolve()
    assert deeper[0].lineno == line
