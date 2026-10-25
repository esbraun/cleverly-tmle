"""Longitudinal checks whose summaries the contract test checks (X89 T3).

The three checks of a sequential fit: overlap, equations solved and models, each over time.
The fits are the longitudinal summary fixtures' (300 rows, linear and logistic learners),
plus a fit with a known policy and categorical treatment. Findings a correct compact fit
does not give (a time point whose every row the limit replaced, a trimmed ratio, a failed
step in another split) are built from real checks with ``dataclasses.replace``.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

from . import Case
from . import longitudinal as fits
from ._check_fidelity import (
    read_longitudinal_equations,
    read_longitudinal_models,
    read_longitudinal_overlap,
)

__all__ = ["FIXTURES"]


@functools.cache
def _policy_fit() -> Any:
    from tests.unit.test_longitudinal_policy_plumbing import (
        COLUMNS,
        LTMLE,
        _exact,
        law,
        policies,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(policies.regimens(("low", "mix")), **_exact(reference="low")).fit(
            law.frame(), **COLUMNS
        )


def _support(build: Any) -> Any:
    return build().diagnostics.support()


def _scores(build: Any) -> Any:
    return build().diagnostics.score_equations()


def _models(build: Any) -> Any:
    return build().diagnostics.nuisance_models()


def overlap_capped() -> Any:
    """Witness: the limit replaced every row at one time point, and a split failed."""
    check = _support(fits.repeats)
    rows = list(check.rows)
    rows[-1] = replace(rows[-1], share_truncated=1.0)
    return replace(
        check,
        rows=tuple(rows),
        draw_max_share_truncated=(1.0, 0.0, 1.0),
        draw_converged=(True, False, True),
    )


def overlap_trimmed() -> Any:
    """Witness: a ratio trim moved some rows."""
    check = _support(fits.means)
    rows = tuple(replace(row, share_trimmed=0.05 * index) for index, row in enumerate(check.rows))
    return replace(check, rows=rows)


_OVERLAP = {
    "means": fits.means,
    "competing": fits.competing,
    "sdr": fits.sdr,
    "ipsi": fits.ipsi,
    "repeats": fits.repeats,
    "capped": fits.capped,
    "unsolved": fits.unsolved,
    "ipw": fits.ipw,
    "policy": _policy_fit,
}
_EQUATIONS = {
    "means": fits.means,
    "competing": fits.competing,
    "repeats": fits.repeats,
    "working_model": fits.working_model,
    "unsolved": fits.unsolved,
}
_MODELS = {
    "means": fits.means,
    "competing": fits.competing,
    "repeats": fits.repeats,
    "ipw": fits.ipw,
    "policy": _policy_fit,
}

FIXTURES: dict[str, Case] = {
    **{
        f"overlap_{name}": Case(
            functools.partial(_support, build), fidelity=read_longitudinal_overlap
        )
        for name, build in _OVERLAP.items()
    },
    "overlap_capped_witness": Case(overlap_capped, fidelity=read_longitudinal_overlap),
    "overlap_trimmed_witness": Case(overlap_trimmed, fidelity=read_longitudinal_overlap),
    **{
        f"equations_{name}": Case(
            functools.partial(_scores, build), fidelity=read_longitudinal_equations
        )
        for name, build in _EQUATIONS.items()
    },
    **{
        f"models_{name}": Case(functools.partial(_models, build), fidelity=read_longitudinal_models)
        for name, build in _MODELS.items()
    },
}
