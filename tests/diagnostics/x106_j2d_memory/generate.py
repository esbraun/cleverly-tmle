"""Generate the memory guard's literals and the reference's constants table from the calibration.

The memory guard of ``LTMLE``, ``SDR`` and ``IPSI`` (``cleverly.longitudinal._memory_guard``)
holds three tables that come from the X106 J2d calibration runs, copied here byte for byte from
the plans repository (``rows/X106/J2/evidence/2026-11-19-j2d/rev9/``):

=====================  =========================================================================
file                   what it holds
=====================  =========================================================================
``constants.json``     the per-family draw constants (bytes per unit per count, ``c0`` in bytes)
                       fitted by ``fit.py``; ``CONSTANTS``
``worker-share.json``  ``WORKER_BYTES``, ``HELPER_BYTES`` and the per-family ``(F, w, m)`` points
                       of the worker term, written by ``coldterm.py``; ``WORKER_SHARE``
``extbox.out``         the in-range predicate written by ``extbox.py``: each family's fitted
                       bounds and the box of each checked extension; ``FITTED`` and
                       ``EXTENSIONS``, then the floor check's verdict.  One correction is
                       applied: the time-to-event families' fitted cells were censored, which
                       ``extbox.py`` did not record
=====================  =========================================================================

``python -m tests.diagnostics.x106_j2d_memory.generate`` prints the literals to paste into the
module and the Markdown table of ``docs/technical-reference/longitudinal-tmle.md``.
``tests/unit/test_longitudinal_memory_guard.py`` checks that the module's literals and the
reference's table equal what this script reads.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).parent

#: The fields of the in-range predicate, in the order ``extbox.py`` writes them.
FIELDS = (
    "n",
    "T",
    "plans",
    "causes",
    "folds",
    "width",
    "bins",
    "paths",
    "horizons",
    "repeats",
    "bootstrap",
    "levels",
    "censored",
    "msm",
)

#: The families whose calibration cells were built from ``make_point_survival``, whose panels
#: are censored although the cases set no ``censor`` flag (:func:`predicate`).
POINT_SURVIVAL = ("hazard_one_node", "hazard_separable")

#: The count of each constant, as the reference's table names it.
TERM_COUNTS = {
    "c0": "fixed, MiB",
    "RSJ": "$R S J$",
    "RT": "$R T$",
    "RTJ": "$R T J$",
    "T": "$T$",
    "TJ": "$T J$",
    "RHJ": "$R H J$",
    "p": "$p$",
    "Rp": "$R p$",
    "Tp": "$T p$",
    "one": "$1$",
    "FRT": "$F R T$",
    "FT": "$F T$",
    "TB": "$T B$",
    "RTB": "$R T B$",
    "FTB": "$F T B$",
    "BB": "$B^2$",
    "TBB": "$T B^2$",
    "RS_msm": "$R S$ (working model)",
    "TU": "$T U$",
    "TUJ": "$T U J$",
    "Up": "$U p$",
}

#: The families in the order the reference lists them.
FAMILIES = (
    "ltmle",
    "ltmle_continuous",
    "sdr",
    "sdr_continuous",
    "ipsi",
    "hazard_one_node",
    "hazard_backward",
    "hazard_paths",
    "hazard_separable",
)


def constants() -> dict[str, dict[str, float]]:
    """The per-family draw constants, as ``constants.json`` holds them."""
    return dict(json.loads((HERE / "constants.json").read_text()))


def workers() -> tuple[int, int, dict[str, tuple[tuple[int, float, float], ...]]]:
    """``WORKER_BYTES``, ``HELPER_BYTES`` and the per-family ``(F, w, m)`` points."""
    record = json.loads((HERE / "worker-share.json").read_text())
    share = {
        family: tuple((int(f), float(w), float(m)) for f, w, m in points)
        for family, points in record["share"].items()
    }
    return int(record["worker_bytes"]), int(record["helper_bytes"]), share


def predicate() -> tuple[
    dict[str, dict[str, tuple[int, int]]],
    dict[str, tuple[tuple[tuple[str, ...], dict[str, int]], ...]],
]:
    """The fitted bounds and the checked-extension boxes of ``extbox.out``.

    Returns
    -------
    tuple of dict
        Per family, ``(low, high)`` of every field over the fitted cells; and per family, one
        ``(fields that leave the fitted bounds, the largest value of every field over the
        group's checked cells)`` pair per checked extension.  An excluded group has no box.
    """
    fitted: dict[str, dict[str, tuple[int, int]]] = {}
    boxes: dict[str, list[tuple[tuple[str, ...], dict[str, int]]]] = {}
    section = None
    for line in (HERE / "extbox.out").read_text().splitlines():
        if line.startswith("== fitted"):
            section = "fitted"
        elif line.startswith("== checked"):
            section = "checked"
        elif line.startswith("== floor check"):
            # The floor check's verdict (``none`` when no checked cell inside the range reads
            # below 1.32); ``extbox.py`` exits non-zero otherwise, so it adds no bound.
            section = "floor"
        elif not line.strip() or line.startswith("excluded") or section == "floor":
            continue
        elif section == "fitted":
            family, _, rest = line.partition(" ")
            bounds = ast.literal_eval(rest)
            fitted[family] = {field: (bounds[field][0], bounds[field][1]) for field in FIELDS}
        elif section == "checked":
            match = re.fullmatch(r"(\w+) (\(.*?\)) (\{.*\})", line)
            assert match is not None, line
            family = match.group(1)
            over = tuple(ast.literal_eval(match.group(2)))
            top = ast.literal_eval(match.group(3))
            boxes.setdefault(family, []).append(
                (over, {field: int(top[field]) for field in FIELDS})
            )
    # ``extbox.py`` read ``censored`` from each case's ``censor`` flag, which is off on the
    # time-to-event cells; their panels (``make_point_survival``) are censored, and the guard
    # reads ``censored`` from the panel, so every fitted cell of these families was censored.
    for family in POINT_SURVIVAL:
        low, high = fitted[family]["censored"]
        fitted[family]["censored"] = (low, max(high, 1))
    return fitted, {family: tuple(found) for family, found in boxes.items()}


def literal() -> str:
    """The Python source of the module's calibration literals."""
    worker, helper, share = workers()
    fitted, boxes = predicate()
    lines = [
        f"CONSTANTS: dict[str, dict[str, float]] = {constants()!r}",
        f"WORKER_BYTES = {worker}",
        f"HELPER_BYTES = {helper}",
        f"WORKER_SHARE: dict[str, tuple[tuple[int, float, float], ...]] = {share!r}",
        f"FITTED: dict[str, dict[str, tuple[int, int]]] = {fitted!r}",
        f"EXTENSIONS: dict[str, tuple[tuple[tuple[str, ...], dict[str, int]], ...]] = {boxes!r}",
    ]
    return "\n".join(lines)


def _number(value: float) -> str:
    """Three significant figures, without an exponent."""
    if value >= 100:
        return f"{value:,.0f}"
    return f"{float(f'{value:.3g}'):g}"


def constants_table() -> str:
    """The reference's table of the draw constants: one row per count, one column per family."""
    table = constants()
    used = [term for term in TERM_COUNTS if any(term in table[family] for family in FAMILIES)]
    header = "| count | " + " | ".join(f"`{family}`" for family in FAMILIES) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in FAMILIES) + " |"
    rows = [header, rule]
    for term in used:
        cells = []
        for family in FAMILIES:
            value = table[family].get(term)
            if value is None:
                cells.append("")
            elif term == "c0":
                cells.append(_number(value / 2**20))
            else:
                cells.append(_number(value))
        rows.append(f"| {TERM_COUNTS[term]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


#: How the reference's range tables name each field.
FIELD_NAMES = {
    "n": "units",
    "T": "nodes",
    "plans": "regimens",
    "causes": "causes",
    "folds": "folds",
    "width": "design columns",
    "bins": "density bins",
    "paths": "treatment paths",
    "horizons": "horizons",
    "repeats": "repeats",
    "bootstrap": "bootstrap replicates",
    "levels": "treatment levels",
    "censored": "censored",
    "msm": "working model",
}

#: The fields the fitted-range table shows; the others are named in the text.
RANGE_COLUMNS = ("n", "T", "plans", "causes", "folds", "width", "bins", "paths", "horizons")


def _span(low: int, high: int) -> str:
    def shown(value: int) -> str:
        return f"{value:,}"

    return shown(low) if low == high else f"{shown(low)} to {shown(high)}"


def ranges_table() -> str:
    """The reference's table of each family's fitted bounds."""
    fitted, _ = predicate()
    header = "| family | " + " | ".join(FIELD_NAMES[field] for field in RANGE_COLUMNS) + " |"
    rows = [header, "| --- | " + " | ".join("---" for _ in RANGE_COLUMNS) + " |"]
    for family in FAMILIES:
        bounds = fitted[family]
        cells = [_span(*bounds[field]) for field in RANGE_COLUMNS]
        rows.append(f"| `{family}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def extensions_table() -> str:
    """The reference's table of the checked extensions: what each reaches, and with what."""
    fitted, boxes = predicate()
    rows = ["| family | reaches | only with |", "| --- | --- | --- |"]
    for family in FAMILIES:
        for over, top in boxes.get(family, ()):
            reaches = ", ".join(
                "censoring" if field == "censored" else f"{FIELD_NAMES[field]} {top[field]:,}"
                for field in over
            )
            within = [
                "no working model"
                if field == "msm"
                else f"{FIELD_NAMES[field]} at most {top[field]:,}"
                for field in FIELDS
                if field not in over and top[field] < fitted[family][field][1]
            ]
            rows.append(f"| `{family}` | {reaches} | {', '.join(within)} |")
    return "\n".join(rows)


def main() -> None:
    """Print the literals and the table."""
    print(literal())
    print()
    print(constants_table())
    print()
    print(ranges_table())
    print()
    print(extensions_table())


if __name__ == "__main__":
    main()


__all__ = [
    "constants",
    "constants_table",
    "extensions_table",
    "literal",
    "predicate",
    "ranges_table",
    "workers",
]
