"""Validated data containers."""

from __future__ import annotations

from .causal_data import CategoricalEncoding, CausalData
from .weighting import WeightReport, WeightSpec

__all__ = ["CategoricalEncoding", "CausalData", "WeightReport", "WeightSpec"]
