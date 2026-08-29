"""The one transcription of the point-treatment replication schema.

Every point-treatment study publishes the same fourteen columns, and five modules used to
build them: this one, ``missing_outcome_study_helpers``, ``intervention_study_helpers``,
``canonical_tmle`` and ``canonical_shift_policies``.  The copies differed only in details no
study had chosen deliberately.  One hardcoded the arm codes, one hardcoded
``inference_scale="identity"``, one called the truth ``reference``, one divided by ``ey0``
and ``1 - ey1`` for laws that report no ratio at all, and one point-treatment study imported
its builder from the missing-outcome module.  A schema written five times is a schema that
can change in one of them.

Two columns are *not* uniform, and the refit gate in ``tests/unit/test_method_evidence.py``
says so in as many words.  ``n`` is ``len(frame)`` in some studies and ``result.data.n`` in
others, and ``initial_estimate`` is a plug-in where a study reports one and ``math.nan``
where it does not.  Both stay arguments here, because unifying either would move a published
number with nothing anywhere to notice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: The contrasts whose transcription divides by an arm mean, so the arms have to be strictly
#: inside the unit interval for the row to mean anything.
RATIO_ESTIMANDS = frozenset({"rr", "or"})


def _odds(value: float) -> float:
    return value / (1.0 - value)


def initial_estimates(result: Any, estimands: Sequence[str] = ()) -> dict[str, float]:
    """Return the untargeted binary-arm parameters on their native reporting scales.

    ``rr`` and ``or`` are built only where ``estimands`` names one of them.  They were built
    unconditionally before, which divided by ``ey0`` and by ``1 - ey1`` for a study whose
    outcome is continuous and whose arm means need not lie in the unit interval.

    Where they are built, the guards are the ones ``tests/canonical/tmle_point_adapter.R``
    applies to the same rows before it forms the same two ratios: finite weights that are
    strictly positive, and finite arm predictions strictly inside the unit interval.  The R
    side refuses those rows, and a Python transcription that returned an infinity instead
    would put it in a published column.

    Parameters
    ----------
    result : Any
        One fitted single result.
    estimands : Sequence[str], optional
        The parameters the study reports.  Only ``rr`` and ``or`` change what is returned.

    Returns
    -------
    dict of str to float
        ``ey0``, ``ey1`` and ``ate``, plus each requested ratio.
    """
    weights = np.asarray(result.data.weights, dtype=float)
    ratios = RATIO_ESTIMANDS.intersection(estimands)
    if ratios and (not np.all(np.isfinite(weights)) or np.any(weights <= 0.0)):
        raise AssertionError("observation weights must be finite and strictly positive")
    arms: dict[float, float] = {}
    for arm in result.data.arm_codes:
        values = np.asarray(
            result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[arm]), dtype=float
        )
        if ratios and (
            not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0)
        ):
            raise AssertionError(
                "binary-outcome nuisance predictions must be finite and strictly between "
                "zero and one"
            )
        arms[float(arm)] = float(np.average(values, weights=weights))
    ey0, ey1 = arms[0.0], arms[1.0]
    initials = {"ey0": ey0, "ey1": ey1, "ate": ey1 - ey0}
    if "rr" in ratios:
        initials["rr"] = ey1 / ey0
    if "or" in ratios:
        initials["or"] = _odds(ey1) / _odds(ey0)
    return initials


def primary_rows(
    *,
    result: Any,
    truth: Mapping[str, float],
    implementation: str,
    scenario: str,
    replicate: int,
    estimands: Sequence[str],
    initials: Mapping[str, float] | None = None,
    n: int | None = None,
) -> list[dict[str, Any]]:
    """Convert one point-treatment fit to the registered primary-replication schema.

    Parameters
    ----------
    result : Any
        One fitted single result.
    truth : Mapping[str, float]
        The exact value of each reported estimand.
    implementation : str
        The name the study publishes its own rows under.
    scenario : str
        The scenario the sample was drawn from.
    replicate : int
        The replication index.
    estimands : Sequence[str]
        The parameters to transcribe, in publication order.
    initials : Mapping[str, float] or None, optional
        The untargeted plug-in of each estimand.  ``None`` writes ``math.nan``, which is
        what a study that reports no plug-in publishes.
    n : int or None, optional
        The size to publish.  Defaults to ``result.data.n``.

    Returns
    -------
    list of dict
        One row per estimand, carrying the fourteen published columns.
    """
    rows: list[dict[str, Any]] = []
    size = result.data.n if n is None else n
    for name in estimands:
        estimate = result[name]
        reference = float(truth[name])
        low, high = estimate.ci
        ratio = estimate.scale == "ratio"
        if ratio and estimate.log_psi is None:  # pragma: no cover - estimator contract guard
            raise AssertionError(f"ratio estimand {name!r} has no log-scale estimate")
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": size,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": (float(estimate.log_psi) if ratio else float(estimate.psi)),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if ratio else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": (math.nan if initials is None else float(initials[name])),
            }
        )
    return rows
