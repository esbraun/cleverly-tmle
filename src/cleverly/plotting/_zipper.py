"""V18: the confidence intervals of a repeated-sampling study, one per replication, sorted.

Each replication of a :class:`~cleverly.validation.CoverageStudy` reports an interval for
a quantity whose true value the study knows. The zipper plot draws every interval as a
horizontal line, measured from that true value and sorted by where its centre falls, so
the intervals that miss the truth gather at the two ends. It shows what a coverage
number hides: whether the misses fall on one side (bias) or on both (too narrow an
interval).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import _core
from ._strips import fit_height, strip_figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

#: Panels per row.
_COLUMNS = 3
#: One panel's width and plotting-area height, in inches.
_PANEL = (3.3, 2.6)


#: Opacity of the intervals that cover. A light blue against a full orange differs in lightness
#: as well as hue, so the misses stand out in grey scale.
_COVER_ALPHA = 0.4


def _on_log_scale(summary: Any) -> bool:
    """Whether the study measured this estimand on the log scale, which is a ratio."""
    shifted = summary.inference_estimates
    if shifted is None:
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.asarray(summary.estimates, dtype=float))
    return bool(np.allclose(shifted, logs, equal_nan=True))


def _positions(records: Sequence[Any], *, ratio: bool) -> dict[str, Any]:
    """Each interval's limits and centre, measured from its own true value.

    A difference is measured as limit minus truth (the truth sits at 0) and a ratio as
    limit over truth (the truth sits at 1, and centres are geometric means).
    """
    from ..exceptions import DataError

    lower = np.asarray([record.lower for record in records], dtype=float)
    upper = np.asarray([record.upper for record in records], dtype=float)
    truth = np.asarray([record.truth for record in records], dtype=float)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise DataError(
            "the study's records hold no finite interval limits, so there is no interval to draw"
        )
    if ratio:
        low, high = lower / truth, upper / truth
        centre = np.sqrt(low * high)
        reference = 1.0
    else:
        low, high = lower - truth, upper - truth
        centre = (low + high) / 2.0
        reference = 0.0
    contains = (low <= reference) & (reference <= high)
    flagged = np.asarray([record.covered for record in records], dtype=bool)
    if not np.array_equal(contains, flagged):
        raise DataError(
            "a replication's interval limits disagree with its recorded coverage, so the "
            "figure would draw a different verdict than the study reported"
        )
    return {"low": low, "high": high, "centre": centre, "covered": contains}


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


@_core.labelled
def zipper(
    study: Any,
    *,
    estimand: str | Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot every replication's interval, sorted by centre, with the misses coloured.

    Use it to see how a study's coverage was lost. Each line is one replication's
    interval, drawn relative to the true value (the dashed line) and sorted by its
    centre. Intervals that miss are orange. Misses piled at one end mean the estimator
    is biased; misses split between the ends mean the intervals are too narrow. The
    subtitle gives the coverage and the level it was aimed at.

    A difference is drawn as the limit minus the true value, a ratio as the limit divided
    by it on a log axis. Both are drawn from the limits each record holds, and the figure
    refuses a study whose limits disagree with the coverage it recorded.

    Parameters
    ----------
    study : StudyResult
        A finished coverage study.
    estimand : str, sequence of str or None, default=None
        The estimands to draw, one panel each; every estimand of the study when omitted.
    labels : dict or None, default=None
        Words for each estimand, keyed by its alias.
    title : str, False or None, default=None
        The title; a short description of the figure when omitted, and none when ``False``.
    subtitle : str, False or None, default=None
        The line under the title; the plot's own, which names the estimate and its headline
        number, when omitted, and none when ``False``.
    xlabel : str, sequence of str or None, default=None
        The x-axis label. A string replaces the first labelled panel's label and the same
        label wherever it repeats; a sequence gives each labelled panel its own, in reading
        order.
    ylabel : str, sequence of str or None, default=None
        The y-axis label, given as for ``xlabel``.
    legend : {"top", "right", "bottom"} or False, default="top"
        Where the key goes: above the axes, right of them, or under the figure; ``False``
        draws no key.
    figsize : tuple of float or None, default=None
        Width and height in inches of a figure the plot creates.

    Returns
    -------
    Figure
        One panel per estimand, up to three across.

    Raises
    ------
    DataError
        When the study's records hold no finite interval limits, or the limits disagree
        with the recorded coverage.
    KeyError
        When ``estimand`` names an estimand the study did not measure.

    See Also
    --------
    cleverly.validation.CoverageStudy : The study that produces the records.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes count the intervals that miss
    below and above the true value and say which estimates are experimental.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import nonlinear_dgp
    >>> from cleverly.estimators import TMLE
    >>> from cleverly.plotting import zipper
    >>> from cleverly.validation import CoverageStudy
    >>> study = CoverageStudy(
    ...     dgp=nonlinear_dgp(),
    ...     estimator=lambda: TMLE(
    ...         outcome_learner=LinearRegression(),
    ...         treatment_learner=LogisticRegression(max_iter=1000),
    ...         cross_fit=False,
    ...     ),
    ...     n=200,
    ...     n_replicates=12,
    ...     estimands=("ate",),
    ...     seed=0,
    ... ).run()
    >>> fig = zipper(study)
    >>> fig.axes[0].get_title(loc="left")
    'Interval coverage'
    >>> plt.close(fig)
    """
    from .._summary._estimates import plain_support_reason

    names = (
        list(study.summaries)
        if estimand is None
        else ([estimand] if isinstance(estimand, str) else list(estimand))
    )
    for name in names:
        if name not in study.summaries:
            raise KeyError(f"the study did not measure {name!r}; it did {list(study.summaries)}")
    columns = min(len(names), _COLUMNS)
    fig, head, axes = strip_figure(len(names), columns, panel=_PANEL)
    coverage: list[str] = []
    found: list[str] = []
    experimental: dict[str, list[str]] = {}
    for axis, name in zip(axes, names, strict=False):
        records = [record for record in study.replications if record.estimand == name]
        ratio = _on_log_scale(study.summaries[name])
        place = _positions(records, ratio=ratio)
        order = np.argsort(place["centre"], kind="stable")
        height = np.arange(1, len(order) + 1)
        low, high, hit = place["low"][order], place["high"][order], place["covered"][order]
        # Thin lines once there are more replications than the panel has room for, so
        # neighbouring intervals stay apart rather than merging into a block.
        width = min(1.2, 180.0 / max(len(order), 1))
        for mask, color, text, gid in (
            (hit, _core.PALETTE[0], "Covers the true value", "cleverly-zipper-covered"),
            (~hit, _core.PALETTE[1], "Misses the true value", "cleverly-zipper-missed"),
        ):
            if mask.any():
                axis.hlines(
                    height[mask],
                    low[mask],
                    high[mask],
                    colors=color,
                    linewidth=width,
                    alpha=_COVER_ALPHA if text.startswith("Covers") else 1.0,
                    label=text,
                    gid=gid,
                    zorder=2.0,
                )
        reference = 1.0 if ratio else 0.0
        _core.reference_line(axis, reference, orientation="vertical", label="True value")
        axis.set_ylim(0.0, len(order) + 1.0)
        axis.set_yticks([])
        axis.set_ylabel("Replications (sorted)" if axis is axes[0] else "")
        shown = (labels or {}).get(name, _core.estimand_words(name))
        if ratio:
            _core.log_axis(axis, "x")
            axis.set_xlabel(f"{shown}\ninterval limit as a multiple of the true value")
        else:
            axis.set_xlabel(f"{shown}\ninterval limit minus the true value")
        coverage.append(
            f"{(labels or {}).get(name, name)} {_percent(float(np.mean(place['covered'])))}"
        )
        below = int(np.sum(place["high"] < reference))
        above = int(np.sum(place["low"] > reference))
        found.append(
            f"{(labels or {}).get(name, name)}: {below} "
            + ("interval falls" if below == 1 else "intervals fall")
            + f" wholly below the true value, {above} above."
        )
        if study.summaries[name].support == "experimental":
            reasons = " ".join(
                dict.fromkeys(
                    plain_support_reason(record.support_reason)
                    for record in records
                    if record.support == "experimental" and record.support_reason
                )
            )
            experimental.setdefault(reasons, []).append((labels or {}).get(name, name))
    _core.header(
        head,
        title or "Interval coverage",
        f"Coverage over {study.n_replicates} replications of n = {study.n}: "
        + ", ".join(coverage)
        + f" (aimed at {_percent(1.0 - float(study.alpha))})",
        key_sources=axes[:1],
    )
    found.insert(
        0,
        "Each line is one replication's interval, measured from the true value; the lines are "
        "sorted from bottom to top by the interval's centre.",
    )
    if study._diagnostic:
        found.append(
            f"These intervals are {study._diagnostic_noun}s, not confidence intervals; "
            "the figure shows what the study measured."
        )
    for reasons, shown_names in experimental.items():
        found.append(
            f"{', '.join(shown_names)}: experimental estimates. "
            + (f"{reasons} " if reasons else "")
            + "The study measures how often their intervals cover the true value."
        )
    if study.n_failed:
        found.append(f"{study.n_failed} replications failed and are not drawn.")
    _core.add_notes(fig, found)
    fit_height(fig, axes, columns, _PANEL[1])
    return fig


__all__ = ["zipper"]
