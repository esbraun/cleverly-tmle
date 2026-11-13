"""The default band, and the band paths band-finite-sample leaves on their own options.

PR 2 of band-finite-sample moved the default band of :func:`simultaneous_bands`,
:func:`median_bands`, every estimator and :class:`cleverly.Inference` to the Mammen multiplier,
10,000 draws and the logit scale (plan section 4.2).  A projection curve's band
(:meth:`ProjectionCurve.band`, :meth:`TMLEResult.effect_curve`) keeps Rademacher at 1,000 draws,
and the ``targeting-curves`` rows that measure it stay byte-identical (plan section 4.3).  These
tests pin both sides, each with a witness that fails when the default reaches the curve path.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from cleverly import Inference
from cleverly.effect_curves import ProjectionCurve
from cleverly.estimators import TMLE
from cleverly.estimators.base import TMLEResult
from cleverly.estimators.dose_response import DoseResponseTMLE
from cleverly.inference import multiplier
from cleverly.inference.multiplier import (
    DEFAULT_BAND_SCALE,
    DEFAULT_KIND,
    DEFAULT_N_REPLICATES,
    median_bands,
    multiplier_critical_value,
    simultaneous_bands,
)
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    StructuralNestedMean,
    SurvivalCTMLE,
    SurvivalDRTMLE,
)

from .test_conditional_contrast import law_study, linear_contrast, oracle

#: Every public signature that sets a band default, with the names of its three options.
BAND_SIGNATURES: tuple[tuple[Any, str, str, str], ...] = (
    (simultaneous_bands, "n_replicates", "kind", "band_scale"),
    (median_bands, "n_replicates", "kind", "band_scale"),
    (TMLE, "n_multiplier", "multiplier_kind", "band_scale"),
    (LTMLE, "n_multiplier", "multiplier_kind", "band_scale"),
    (SDR, "n_multiplier", "multiplier_kind", "band_scale"),
    (IPSI, "n_multiplier", "multiplier_kind", "band_scale"),
    (SurvivalCTMLE, "n_multiplier", "multiplier_kind", "band_scale"),
    # It builds no band and refuses simultaneous=True; its options still take the default.
    (SurvivalDRTMLE, "n_multiplier", "multiplier_kind", "band_scale"),
    (StructuralNestedMean, "n_multiplier", "multiplier_kind", "band_scale"),
)


def test_the_constants_are_the_declared_default() -> None:
    assert (DEFAULT_BAND_SCALE, DEFAULT_KIND, DEFAULT_N_REPLICATES) == ("logit", "mammen", 10_000)


@pytest.mark.parametrize(
    ("target", "draws", "kind", "scale"),
    BAND_SIGNATURES,
    ids=lambda item: getattr(item, "__name__", ""),
)
def test_every_band_signature_defaults_to_the_declared_band(
    target: Any, draws: str, kind: str, scale: str
) -> None:
    parameters = inspect.signature(target).parameters
    assert parameters[draws].default == DEFAULT_N_REPLICATES
    assert parameters[kind].default == DEFAULT_KIND
    assert parameters[scale].default == DEFAULT_BAND_SCALE


def test_the_method_and_the_dose_response_curve_use_the_default() -> None:
    inference = Inference()
    assert (inference.n_multiplier, inference.multiplier_kind, inference.band_scale) == (
        DEFAULT_N_REPLICATES,
        DEFAULT_KIND,
        DEFAULT_BAND_SCALE,
    )
    assert inspect.signature(DoseResponseTMLE).parameters["n_replicates"].default == (
        DEFAULT_N_REPLICATES
    )


def test_the_critical_value_names_its_draws_and_kind() -> None:
    """No default reaches a caller that does not name the draws and the kind (plan 4.1)."""
    parameters = inspect.signature(multiplier_critical_value).parameters
    assert parameters["n_replicates"].default is inspect.Parameter.empty
    assert parameters["kind"].default is inspect.Parameter.empty


def test_the_curve_band_signatures_keep_rademacher_at_1000() -> None:
    for method in (ProjectionCurve.band, TMLEResult.effect_curve):
        parameters = inspect.signature(method).parameters
        assert parameters["n_replicates"].default == 1000, method
        assert parameters["kind"].default == "rademacher", method


@pytest.fixture(scope="module")
def projection_fit() -> Any:
    return law_study().identify(linear_contrast()).estimate(**oracle())


def test_the_curve_band_passes_rademacher_at_1000(
    projection_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path the ``targeting-curves`` rows read draws Rademacher at 1,000.

    The witness records the options the curve band hands the critical value.  A curve band that
    fell through to the module defaults, or forwarded the estimator's band, would record Mammen
    at 10,000 here.
    """
    calls: list[dict[str, Any]] = []
    real = multiplier_critical_value

    def recorder(*args: Any, **kwargs: Any) -> float:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("cleverly.effect_curves.multiplier_critical_value", recorder)
    grid = {"W": np.array([0.0, 1.0, 2.0])}
    curve = projection_fit.effect_curve(grid=grid, random_state=0)
    assert [(call["kind"], call["n_replicates"]) for call in calls] == [("rademacher", 1000)]
    assert (curve.band.kind, curve.band.n_replicates) == ("rademacher", 1000)


def test_the_multiplier_module_states_the_default() -> None:
    """The module docstring says why each default holds, which the inference page links."""
    text = multiplier.__doc__ or ""
    for topic in ("Why the transform", "Why Mammen", "same on either scale", "Why no bootstrap-t"):
        assert topic in text, topic


def test_the_support_tables_read_the_default_band() -> None:
    """The option and draws every band row lists are the library default (band-finite-sample)."""
    from cleverly._support import DEFAULT_BAND_OPTION
    from cleverly._support_rows import DEFAULT_BAND_DRAWS
    from cleverly.estimators.dose_response import DOSE_RESPONSE_BAND_DRAWS

    assert DEFAULT_BAND_OPTION == (DEFAULT_KIND, DEFAULT_BAND_SCALE)
    assert DEFAULT_BAND_DRAWS == DOSE_RESPONSE_BAND_DRAWS == DEFAULT_N_REPLICATES


def test_the_drift_corrected_survival_tmle_takes_the_band_options_and_builds_no_band() -> None:
    """Its multiplier options are inert, as on the other engines; only a band is refused."""
    from cleverly.exceptions import CapabilityError

    arms = {"a1": 1, "a0": 0}
    other = {"n_multiplier": 500, "multiplier_kind": "rademacher", "band_scale": "identity"}
    assert SurvivalDRTMLE(arms, **other)._settings_refusal() is None
    with pytest.raises(CapabilityError, match="band"):
        SurvivalDRTMLE(arms, simultaneous=True)._validate_settings()
