"""Header facts read off the data a study or a fit holds.

The study-setup summaries describe a :class:`~cleverly.data.causal_data.CausalData` or a
:class:`~cleverly.longitudinal.data.LongitudinalData` before any fit, so they read the data
itself. Each function returns the same text a fitted point result prints for the same fact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from . import vocabulary
from ._render import count_text, share_text

#: Above this many names, a list prints its first :data:`SHOWN_NAMES` and a count.
MOST_NAMES = 8
SHOWN_NAMES = 6
#: The most treatment levels a Treatment fact lists before it counts them.
MOST_LEVELS = 6


def listed(names: Iterable[Any], *, most: int = MOST_NAMES, shown: int = SHOWN_NAMES) -> str:
    """Names joined by commas, the first ``shown`` and a count above ``most``.

    Parameters
    ----------
    names : iterable
        The names, in order.
    most : int
        The most names printed in full.
    shown : int
        The names printed before the count when there are more than ``most``.

    Returns
    -------
    str
        For example ``"W1, W2, W3"`` or ``"W1, ..., W6 and 4 more"``.
    """
    items = [str(name) for name in names]
    if len(items) > most:
        return f"{', '.join(items[:shown])} and {len(items) - shown} more"
    return ", ".join(items)


def covariate_sources(data: Any) -> tuple[str, ...]:
    """The adjustment columns as the caller named them, before any indicator coding.

    A randomized design that adjusts for nothing holds one internal constant column, which
    names no column of the caller's and is left out.
    """
    from ..data.validate import RANDOMIZED_INTERCEPT

    sources = getattr(data, "indicator_sources", {}) or {}
    names = (sources.get(name, name) for name in data.covariate_names)
    return tuple(dict.fromkeys(name for name in names if name != RANDOMIZED_INTERCEPT))


def binary_outcome(values: Any) -> bool:
    """Whether the observed outcome takes only the values 0 and 1."""
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    return observed.size > 0 and bool(np.isin(observed, (0.0, 1.0)).all())


def outcome_fact(data: Any) -> str:
    """The Outcome fact of point data: the name, binary or continuous, and the observed count.

    Parameters
    ----------
    data : CausalData
        The data.

    Returns
    -------
    str
        For example ``"Y (continuous); 180 of 200 observed"``.
    """
    observed = np.asarray(data.outcome, dtype=float)[np.asarray(data.observed, dtype=bool)]
    kind = "binary" if binary_outcome(observed) else "continuous"
    text = f"{data.outcome_name} ({kind})"
    if data.has_missing_outcome:
        text += f"; {count_text(int(data.observed.sum()))} of {count_text(data.n)} observed"
    return text


def rows_fact(data: Any) -> str:
    """The Rows fact of point data, with the recorded-treatment count when some lack one."""
    text = count_text(data.n)
    if getattr(data, "has_missing_treatment", False):
        text += f" ({count_text(int(data.treatment_recorded.sum()))} with a recorded treatment)"
    return text


def _zero_one(levels: Sequence[Any]) -> bool:
    try:
        return len(levels) == 2 and {float(level) for level in levels} == {0.0, 1.0}
    except (TypeError, ValueError):
        return False


def treatment_fact(data: Any) -> str:
    """The Treatment fact of point data: the name and the kind of values it takes.

    Parameters
    ----------
    data : CausalData
        The data.

    Returns
    -------
    str
        For example ``"A (levels 0, 1)"``, ``"A (continuous dose)"`` or ``"D + K (vector)"``.
    """
    if data.treatment_kind == "vector":
        return f"{', '.join(data.component_names)} (several columns, one treatment)"
    if data.is_continuous_treatment:
        return f"{data.treatment_name} (continuous dose)"
    levels = tuple(data.treatment_levels or ())
    if len(levels) > MOST_LEVELS:
        return f"{data.treatment_name} ({len(levels)} levels)"
    return f"{data.treatment_name} (levels {', '.join(str(level) for level in levels)})"


def share_fact(data: Any) -> tuple[str, str]:
    """The treated share, the arm shares or the dose range of point data, with its label.

    Parameters
    ----------
    data : CausalData
        The data.

    Returns
    -------
    tuple of str
        ``(label, value)``, the label marked weighted on weighted data.
    """
    weighted = vocabulary.WEIGHTED if data.is_weighted else ""
    if data.treatment_kind == "vector":
        columns = data._component_columns()
        parts: list[str] = []
        for name, levels in zip(data.component_names, data.component_levels, strict=True):
            values = np.asarray(columns[name])
            if levels:
                shares = ", ".join(
                    f"{level}={float(np.average(values == level, weights=data.weights)):.3f}"
                    for level in levels
                )
                parts.append(f"{name}: {shares}")
            else:
                dose = np.asarray(values, dtype=float)
                parts.append(f"{name}: range [{dose.min():.3g}, {dose.max():.3g}]")
        return vocabulary.ARM_SHARES + weighted, "; ".join(parts)
    if data.is_continuous_treatment:
        dose = np.asarray(data.treatment, dtype=float)
        recorded = np.isfinite(dose)
        mean = float(np.average(dose[recorded], weights=data.weights[recorded]))
        low, high = float(dose[recorded].min()), float(dose[recorded].max())
        return vocabulary.DOSE + weighted, f"mean {mean:.4g}, range [{low:.3g}, {high:.3g}]"
    if data.is_binary_treatment and _zero_one(tuple(data.treatment_levels or ())):
        return vocabulary.TREATED_SHARE + weighted, share_text(data.treated_fraction)
    shares = ", ".join(
        f"{data.arm_label(arm)}={share:.3f}"
        for arm, share in zip(data.arm_codes, data.arm_fractions, strict=True)
    )
    return vocabulary.ARM_SHARES + weighted, shares


def clusters_fact(data: Any) -> str | None:
    """The Clusters fact: the count and the cluster column, or ``None`` without clusters."""
    cluster = getattr(data, "cluster", None)
    if cluster is None:
        return None
    name = getattr(data, "cluster_name", None)
    count = count_text(int(np.unique(cluster).size))
    return f"{count} ({name})" if name else count


def weights_fact(data: Any) -> str | None:
    """The Weights fact: the column, whether estimated, and the effective sample size."""
    if not getattr(data, "is_weighted", False):
        return None
    report = data.weight_report()
    kind = ", estimated" if report.estimated else ""
    return f"{report.name or 'weights'}{kind}; effective sample size {report.effective_n:,.0f}"
