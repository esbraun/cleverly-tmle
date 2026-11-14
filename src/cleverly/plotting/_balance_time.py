"""V11 over time: the balance plot of a fit whose weights vary over time (X103).

:func:`~cleverly.plotting.balance` hands a :class:`~cleverly.sensitivity.BalanceOverTime` here.
Two views:

* ``"summary"``: one panel per exposure, the covariates down the side, each covariate's
  difference averaged over the earlier treatments (Jackson 2016, Table 2), before weighting as
  a hollow marker and after as a filled one, with a whisker over the strata's differences after
  weighting. The point balance plot's language, so a reader of one reads the other.
* ``"panels"``: Jackson's trellis (Figures 2 and 4): a grid of exposure times by covariate
  times, one dot per stratum of earlier treatments, sized by its weight, and the average.

Every number drawn is one the report holds; nothing is refitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..exceptions import CapabilityError, RefusalReason
from . import _core
from ._strips import fit_height, strip_figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..sensitivity.balance_time import BalanceOverTime, TimeBalanceRow

#: Panels per row of the summary view.
_COLUMNS = 3
#: The exposure times the trellis draws when ``nodes=`` is omitted.
_TRELLIS_NODES = 4

#: The words of each diagnostic in the subtitle.
DIAGNOSTIC_WORDS: dict[str, str] = {
    "confounding": "Covariates up to each treatment, across it, within earlier treatments",
    "feedback": "Covariates after each treatment, across it, within earlier treatments",
    "censoring": "Covariates across leaving the study and staying",
    "joint": "Covariates across the arms among units whose status is known",
}
#: What the weights of each diagnostic are, for the notes.
WEIGHT_WORDS: dict[str, str] = {
    "confounding": (
        "After weighting, each unit counts as one over the probability of its own treatments up "
        "to the treatment compared and of staying observed before it."
    ),
    "feedback": (
        "After weighting, each unit counts as one over the probability of the earlier treatment "
        "compared, times one over the probability of staying observed until the covariate."
    ),
    "censoring": (
        "After weighting, each unit counts as one over the probability of its treatments so far "
        "and of what it did at the time compared: stay or leave."
    ),
    "joint": (
        "After weighting, each unit whose status at the target time is known counts as one over "
        "the probability of its arm and of staying uncensored until its status was known."
    ),
}
#: What the weights of a continuous-time fit's censoring rows are.
CONTINUOUS_CENSORING_WORDS = (
    "Each arm's units whose status at the target time is known are compared with the whole "
    "arm; after weighting each counts as one over the probability of staying uncensored until "
    "its status was known."
)


def check_point_keywords(diagnostic: str | None, view: str, nodes: Any) -> None:
    """Refuse the over-time keywords on a point-treatment balance.

    Parameters
    ----------
    diagnostic : str or None
        The ``diagnostic=`` passed.
    view : str
        The ``view=`` passed.
    nodes : object
        The ``nodes=`` passed.
    """
    given = [
        name
        for name, value, default in (
            ("diagnostic", diagnostic, None),
            ("view", view, "summary"),
            ("nodes", nodes, None),
        )
        if value != default
    ]
    if given:
        raise ValueError(
            f"{', '.join(given)} applies to the balance of a fit over time; a point-treatment "
            "fit has one treatment and no time points."
        )


def _absolute(value: float) -> float:
    return abs(value) if np.isfinite(value) else float("nan")


def _chosen(summary: BalanceOverTime, diagnostic: str | None) -> str:
    if diagnostic is None:
        return summary.diagnostics[0]
    summary._check_diagnostic(diagnostic)
    return diagnostic


def _nodes(
    rows: Sequence[TimeBalanceRow], nodes: Sequence[int] | None, cap: int | None
) -> list[int]:
    present = sorted({row.exposure_node for row in rows})
    if nodes is not None:
        wanted = [int(node) for node in nodes]
        missing = [node for node in wanted if node not in present]
        if missing:
            raise ValueError(
                f"nodes= names {missing}, which have no rows; the time points with rows are "
                f"{present}"
            )
        return [node for node in present if node in wanted]
    if cap is not None and len(present) > cap:
        return present[-cap:]
    return present


def draw(
    summary: BalanceOverTime,
    *,
    diagnostic: str | None,
    view: str,
    nodes: Sequence[int] | None,
    max_covariates: int,
    treatment_label: str | None,
    labels: dict[str, str] | None,
    title: str | None,
    page: int | None = None,
) -> Figure:
    """Draw a balance over time.

    Parameters
    ----------
    summary : BalanceOverTime
        The report.
    diagnostic : str or None
        The diagnostic; the first one the report has when omitted.
    view : {"summary", "panels"}
        The view.
    nodes : sequence of int or None
        The exposure time points to draw.
    max_covariates : int
        The most covariates in one panel of the summary view.
    treatment_label : str or None
        Words for the treatment in panel titles.
    labels : dict or None
        Words for each covariate.
    title : str or None
        The title, or the plot's own.
    page : int or None, default=None
        The page of each panel's ranked covariates to draw, ``max_covariates`` to a page.

    Returns
    -------
    Figure
        The figure.
    """
    if view not in ("summary", "panels"):
        raise ValueError(f"view must be 'summary' or 'panels'; got {view!r}")
    chosen = _chosen(summary, diagnostic)
    rows = summary.rows_of(chosen)
    if view == "panels":
        if summary.family == "continuous_survival":
            raise CapabilityError(
                "view='panels' draws one dot per stratum of earlier treatments, and a "
                "continuous-time fit has none.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Draw the averaged comparisons with view='summary'.",
            )
        if page is not None:
            from .._refusal import RefusalReason
            from ..exceptions import CapabilityError

            raise CapabilityError(
                "page= pages the covariates of the summary view; view='panels' draws every "
                "covariate, so it has no pages",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="omit page=, or draw view='summary' with page=",
            )
        return _trellis(summary, chosen, rows, nodes, labels, title)
    return _summary_view(
        summary, chosen, rows, nodes, max_covariates, treatment_label, labels, title, page
    )


def _said(text: str, summary: BalanceOverTime, treatment_label: str | None) -> str:
    if not treatment_label or len(summary.treatments) != 1:
        return text
    return text.replace(f"{summary.treatments[0]} = ", f"{treatment_label} = ")


def _summary_view(
    summary: BalanceOverTime,
    diagnostic: str,
    rows: Sequence[TimeBalanceRow],
    nodes: Sequence[int] | None,
    max_covariates: int,
    treatment_label: str | None,
    labels: dict[str, str] | None,
    title: str | None,
    page: int | None = None,
) -> Figure:
    from ..sensitivity.balance_time import averages

    keep = _nodes(rows, nodes, None)
    rows = [row for row in rows if row.exposure_node in keep]
    groups = averages(rows, over="history")
    panels = list(dict.fromkeys((group["exposure"], group["contrast"]) for group in groups))
    by_panel: dict[tuple[str, str], list[dict[str, Any]]] = {panel: [] for panel in panels}
    for group in groups:
        by_panel[(group["exposure"], group["contrast"])].append(group)
    strata: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        strata.setdefault((row.exposure, row.contrast, row.covariate), []).append(row.smd_weighted)
    hidden = 0
    shown_by_panel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    ranked_by_panel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for panel, members in by_panel.items():
        unweighted = {m["covariate"]: _absolute(m["smd_unweighted"]) for m in members}
        weighted = {m["covariate"]: _absolute(m["smd_weighted"]) for m in members}
        ranked = sorted(
            members,
            key=lambda m: -np.nanmax([unweighted[m["covariate"]], weighted[m["covariate"]], 0.0]),
        )
        above = [m for m in ranked if weighted[m["covariate"]] > summary.threshold]
        rest = [m for m in ranked if m not in above]
        ranked_by_panel[panel] = above + rest
        chosen = (above + rest)[: max(max_covariates, len(above))]
        hidden += len(members) - len(chosen)
        order = [m for m in members if m in chosen]
        shown_by_panel[panel] = order
    empty: list[tuple[str, str]] = []
    pages = max(
        (math.ceil(len(ranked) / max_covariates) for ranked in ranked_by_panel.values()),
        default=1,
    )
    unpaged = dict(shown_by_panel)
    if page is not None:
        if page > pages:
            raise ValueError(f"page must be at most {pages}, the number of pages; got {page}")
        start = (page - 1) * max_covariates
        for panel, ranked in ranked_by_panel.items():
            on_page = ranked[start : start + max_covariates]
            shown_by_panel[panel] = [m for m in by_panel[panel] if m in on_page]
            if not on_page:
                empty.append(panel)
    columns = min(len(panels), _COLUMNS) or 1
    tallest = max((len(v) for v in shown_by_panel.values()), default=1)
    panel_height = 0.27 * tallest + 0.45
    names = [m["covariate"] for v in shown_by_panel.values() for m in v] or [""]
    longest = max(len((labels or {}).get(name, name)) for name in names)
    fig, head, axes = strip_figure(
        max(len(panels), 1),
        columns,
        panel=(3.6, panel_height),
        left=longest * _core.TICK_SIZE * 0.55 / 72.0,
    )
    values = [summary.threshold * 1.6]
    for panel, members in shown_by_panel.items():
        for m in members:
            values.extend(
                v
                for v in (_absolute(m["smd_unweighted"]), _absolute(m["smd_weighted"]))
                if np.isfinite(v)
            )
            values.extend(
                abs(v)
                for v in strata.get((panel[0], panel[1], m["covariate"]), [])
                if np.isfinite(v)
            )
    ceiling = max(values)
    opposite = 0
    for index, (axis, panel) in enumerate(zip(axes, panels, strict=False)):
        members = shown_by_panel[panel]
        heights = np.arange(len(members))[::-1] + 0.0
        before = np.asarray([_absolute(m["smd_unweighted"]) for m in members])
        after = np.asarray([_absolute(m["smd_weighted"]) for m in members])
        for height, m in zip(heights, members, strict=True):
            values_s = [
                v for v in strata.get((panel[0], panel[1], m["covariate"]), []) if np.isfinite(v)
            ]
            if len(values_s) > 1:
                low, high = min(abs(v) for v in values_s), max(abs(v) for v in values_s)
                axis.hlines(
                    height,
                    low,
                    high,
                    color=_core.CONTEXT,
                    linewidth=1.0,
                    zorder=1,
                    gid="cleverly-balance-strata",
                )
                sign = np.sign(m["smd_weighted"]) if np.isfinite(m["smd_weighted"]) else 0.0
                opposite += sum(1 for v in values_s if sign and np.sign(v) == -sign)
        axis.scatter(
            before,
            heights,
            s=34,
            facecolors=_core.FACE,
            edgecolors=_core.PALETTE[1],
            linewidths=1.6,
            zorder=3,
            gid="cleverly-balance-before",
            label="Before weighting" if index == 0 else None,
        )
        axis.scatter(
            after,
            heights,
            s=34,
            color=_core.PALETTE[0],
            zorder=4,
            gid="cleverly-balance-after",
            label="After weighting" if index == 0 else None,
        )
        _core.reference_line(
            axis,
            summary.threshold,
            orientation="vertical",
            label=f"Threshold, {summary.threshold:g}" if index == 0 else None,
        )
        axis.set_xlim(0.0, ceiling * 1.08)
        axis.set_yticks(heights)
        axis.set_yticklabels(
            [(labels or {}).get(m["covariate"], m["covariate"]) for m in members],
            fontsize=_core.TICK_SIZE,
            color=_core.INK,
        )
        axis.set_ylim(-0.7, max(len(members), 1) - 0.3)
        if not members and page is not None:
            axis.text(
                0.5,
                0.5,
                "No covariates on this page",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=_core.TICK_SIZE,
                color=_core.SECONDARY_INK,
                gid="cleverly-balance-empty-page",
            )
        axis.grid(False)
        axis.grid(True, axis="x", color=_core.GRID, linewidth=0.8)
        axis.set_xlabel("Absolute standardized mean difference")
        words = f"{panel[0]}: {panel[1]}" if diagnostic == "joint" else panel[1]
        axis.set_title(
            _said(words, summary, treatment_label),
            loc="left",
            fontsize=_core.SUBTITLE_SIZE,
            color=_core.INK,
        )
    for axis in axes[len(panels) :]:
        axis.set_axis_off()
    if page is not None:
        # The note counts the strata of the unpaged selection, as the figure without pages
        # does, so a page changes no note but its own.
        opposite = _opposite(unpaged, strata)
    finite = [g for g in groups if np.isfinite(g["smd_weighted"])]
    worst = max(finite, key=lambda g: abs(g["smd_weighted"])) if finite else None
    pieces = [DIAGNOSTIC_WORDS[diagnostic]]
    if worst is not None:
        name = (labels or {}).get(worst["covariate"], worst["covariate"])
        pieces.append(f"largest after weighting {name} {abs(worst['smd_weighted']):.3f}")
    _core.header(
        head,
        title or "Covariate balance over time",
        "; ".join(pieces),
        key_sources=axes[:1],
    )
    found = _notes(summary, diagnostic)
    found.append(
        "Each marker is the covariate's standardized difference averaged over the strata of "
        "earlier treatments, each stratum counting by its size, drawn as an absolute value; "
        "the whisker spans the strata's absolute differences after weighting."
    )
    if opposite:
        found.append(
            f"{opposite} strata differ after weighting in the direction opposite to their "
            "covariate's average, which the average offsets."
        )
    if page is not None:
        first = (page - 1) * max_covariates + 1
        last = page * max_covariates
        none = (
            "; "
            + ", ".join(_said(f"{e}: {c}", summary, treatment_label) for e, c in empty)
            + (" has" if len(empty) == 1 else " have")
            + " none on this page"
            if empty
            else ""
        )
        found.append(
            f"Page {page} of {pages}: each panel's covariates ranked {first} to {last}, "
            f"covariates above the threshold after weighting first{none}."
        )
    elif hidden:
        found.append(
            f"{hidden} covariate rows are not drawn. Every covariate above the threshold after "
            "weighting is drawn, then those with the largest difference before or after."
        )
    _core.add_notes(fig, found)
    fit_height(fig, axes, columns, panel_height)
    return fig


def _opposite(
    shown_by_panel: dict[tuple[str, str], list[dict[str, Any]]],
    strata: dict[tuple[str, str, str], list[float]],
) -> int:
    """The strata that differ after weighting opposite to their drawn covariate's average."""
    count = 0
    for panel, members in shown_by_panel.items():
        for m in members:
            values = [
                v for v in strata.get((panel[0], panel[1], m["covariate"]), []) if np.isfinite(v)
            ]
            if len(values) > 1:
                sign = np.sign(m["smd_weighted"]) if np.isfinite(m["smd_weighted"]) else 0.0
                count += sum(1 for v in values if sign and np.sign(v) == -sign)
    return count


def _notes(summary: BalanceOverTime, diagnostic: str) -> list[str]:
    words = (
        CONTINUOUS_CENSORING_WORDS
        if diagnostic == "censoring" and summary.family == "continuous_survival"
        else WEIGHT_WORDS[diagnostic]
    )
    found = [
        "The differences are descriptive: they carry no interval or test, and a small "
        "difference on the covariates listed does not show balance on covariates that were "
        "not. Each is divided by a spread read in the stratum before weighting: the square "
        "root of the mean of the two groups' variances, or, for units whose status is known "
        "against their whole arm, the arm's standard deviation.",
        words,
    ]
    if summary.limits is not None:
        found.append(
            f"Each cumulative probability is kept within [{summary.limits[0]:.3g}, "
            f"{summary.limits[1]:.3g}] before it is inverted; the limits moved "
            f"{summary.bounded_share:.1%} of the weighted rows."
        )
    dropped = [item for item in summary.dropped if item.diagnostic == diagnostic]
    if dropped:
        found.append(
            "Strata where one group is empty have no comparison: "
            + "; ".join(f"{item.exposure}: {item.strata} ({item.units} rows)" for item in dropped)
            + "."
        )
    found.extend(summary.notes)
    return found


def _trellis(
    summary: BalanceOverTime,
    diagnostic: str,
    rows: Sequence[TimeBalanceRow],
    nodes: Sequence[int] | None,
    labels: dict[str, str] | None,
    title: str | None,
) -> Figure:
    from ..sensitivity.balance_time import averages

    keep = _nodes(rows, nodes, _TRELLIS_NODES)
    capped = nodes is None and len({row.exposure_node for row in rows}) > len(keep)
    rows = [row for row in rows if row.exposure_node in keep]
    # One panel line per exposure and comparison: a node with several arms, or with both
    # censoring and an eliminated cause, has several.
    exposures = list(
        dict.fromkeys(
            (r.exposure_node, r.exposure, r.contrast)
            for r in sorted(rows, key=lambda r: r.exposure_node)
        )
    )
    several = len(exposures) > len({key[0] for key in exposures})
    covariate_nodes = sorted({row.covariate_node for row in rows})
    feedback = diagnostic == "feedback"
    # Figure 4 puts the covariate times down the side for feedback; Figure 2 the exposures.
    down, across = (covariate_nodes, exposures) if feedback else (exposures, covariate_nodes)
    plt = _core.pyplot()
    width = max(6.4, 2.4 * len(across) + 1.2)
    names_per = {
        node: list(dict.fromkeys(r.covariate for r in rows if r.covariate_node == node))
        for node in covariate_nodes
    }
    tallest = max((len(v) for v in names_per.values()), default=1)
    height = (0.25 * tallest + 0.6) * len(down) + 1.4
    chosen = _core.label_options().figsize
    figsize = chosen if chosen is not None else (width, height)
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.set_facecolor(_core.FACE)
    grid = fig.add_gridspec(len(down) + 1, len(across), height_ratios=[0.5] + [1.0] * len(down))
    head = fig.add_subplot(grid[0, :])
    head.set_axis_off()
    groups = {
        (g["exposure_node"], g["exposure"], g["contrast"], g["covariate_node"], g["covariate"]): g
        for g in averages(rows, over="history")
    }
    ceiling = max(
        [summary.threshold * 1.6]
        + [abs(r.smd_weighted) for r in rows if np.isfinite(r.smd_weighted)]
    )
    total = max((r.person_time_weighted for r in rows), default=1.0) or 1.0
    first_axis = None
    shared: Axes | None = None
    lowest: dict[int, Axes] = {}
    for i, row_node in enumerate(down):
        for j, column_node in enumerate(across):
            # One x scale for every panel, so the dots compare across cells.
            axis = fig.add_subplot(grid[i + 1, j], sharex=shared)
            shared = shared if shared is not None else axis
            _core.style_axes(axis)
            pair = (column_node, row_node) if feedback else (row_node, column_node)
            key = cast("tuple[int, str, str]", pair[0])
            covariate_node = cast(int, pair[1])
            cell = [
                r
                for r in rows
                if (r.exposure_node, r.exposure, r.contrast) == key
                and r.covariate_node == covariate_node
            ]
            if not cell:
                axis.set_axis_off()
                continue
            names = list(dict.fromkeys(r.covariate for r in cell))
            heights = {name: float(k) for k, name in enumerate(reversed(names))}
            axis.scatter(
                [r.smd_weighted for r in cell],
                [heights[r.covariate] for r in cell],
                s=[12 + 60 * r.person_time_weighted / total for r in cell],
                color=_core.PALETTE[0],
                alpha=0.55,
                zorder=3,
                gid="cleverly-balance-strata-dots",
                label="Stratum of earlier treatments" if first_axis is None else None,
            )
            averaged = [groups[(*key, covariate_node, name)] for name in names]
            axis.scatter(
                [g["smd_weighted"] for g in averaged],
                [heights[name] for name in names],
                s=46,
                marker="D",
                facecolors=_core.FACE,
                edgecolors=_core.INK,
                linewidths=1.3,
                zorder=4,
                gid="cleverly-balance-average",
                label="Average over strata" if first_axis is None else None,
            )
            axis.axvline(0.0, color=_core.CONTEXT, linewidth=0.9, zorder=1)
            for side in (-1.0, 1.0):
                axis.axvline(
                    side * summary.threshold, color=_core.GRID, linewidth=0.9, linestyle="--"
                )
            axis.set_xlim(-ceiling * 1.1, ceiling * 1.1)
            axis.set_yticks(list(heights.values()))
            axis.set_yticklabels(
                [(labels or {}).get(name, name) for name in heights],
                fontsize=_core.TICK_SIZE,
                color=_core.INK,
            )
            axis.set_ylim(-0.7, len(names) - 0.3)
            exposure_words = cell[0].contrast if several else cell[0].exposure
            covariate_words = "baseline" if covariate_node == 0 else f"time {covariate_node}"
            axis.set_title(
                f"{exposure_words}; covariates at {covariate_words}",
                loc="left",
                fontsize=_core.TICK_SIZE,
                color=_core.INK,
            )
            lowest[j] = axis
            if first_axis is None:
                first_axis = axis
    for axis in lowest.values():
        axis.set_xlabel("Standardized difference after weighting")
    _core.header(
        head,
        title or "Covariate balance over time",
        f"{DIAGNOSTIC_WORDS[diagnostic]}; one dot per stratum",
        key_sources=[first_axis] if first_axis is not None else None,
    )
    found = _notes(summary, diagnostic)
    found.append(
        "Each dot is one stratum of earlier treatments, sized by its weight; the diamond is the "
        "average over strata, each counting by its size. The dashed lines mark the threshold "
        "either side of zero."
    )
    if capped:
        found.append(
            f"Only the last {len(keep)} time points are drawn; pass nodes= to choose others."
        )
    _core.add_notes(fig, found)
    return fig
