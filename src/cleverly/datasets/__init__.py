"""Synthetic data-generating processes with known population estimands."""

from __future__ import annotations

from .synthetic import (
    DGP,
    GENERATORS,
    available,
    binary_outcome_dgp,
    cde_dgp,
    clustered_dgp,
    instrument_dgp,
    linear_dgp,
    make_binary_outcome,
    make_cde,
    make_clustered,
    make_instrument,
    make_linear_ate,
    make_missing_outcome,
    make_nonlinear_ate,
    make_weak_overlap,
    missing_outcome_dgp,
    nonlinear_dgp,
    weak_overlap_dgp,
)

__all__ = [
    "DGP",
    "GENERATORS",
    "available",
    "binary_outcome_dgp",
    "cde_dgp",
    "clustered_dgp",
    "instrument_dgp",
    "linear_dgp",
    "make_binary_outcome",
    "make_cde",
    "make_clustered",
    "make_instrument",
    "make_linear_ate",
    "make_missing_outcome",
    "make_nonlinear_ate",
    "make_weak_overlap",
    "missing_outcome_dgp",
    "nonlinear_dgp",
    "weak_overlap_dgp",
]
