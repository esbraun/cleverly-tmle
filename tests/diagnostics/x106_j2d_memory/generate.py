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

The accuracy section of the reference quotes ranges from other calibration outputs, copied byte
for byte from the same plans directories into two subdirectories.  :func:`accuracy_readings`
reads every quoted number from them, and :func:`accuracy_phrases` gives the sentences that quote
them:

``rev9/`` (from ``rows/X106/J2/evidence/2026-11-19-j2d/rev9/``)
    ``packet10.out`` (resident set over traced peak), ``band_check.out`` and
    ``band_check-outside.out`` (the band term inside and outside the margin, before the part D
    refit), ``band-*.jsonl`` (band-on and band-off peaks), ``coldterm.out`` (parallel runs),
    ``fit11c-sum.out`` (the summed objective), ``bins.jsonl`` and ``dens.jsonl`` (the density
    routes' peaks) and ``witness11.jsonl`` (the fast-suite fixtures at 2,000 units).
``part_d/`` (from ``rows/X106/D/evidence/2026-11-25-j2d-recal/``)
    ``fit.out`` (every family's fitted and checked cells against the shipped constants),
    ``fit-no-ceiling.out`` (the refit without the 3.5 ceiling), ``band_check.out`` (the refitted
    families' band cells) and ``liblinear-readings.jsonl``.

``python -m tests.diagnostics.x106_j2d_memory.generate`` prints the literals to paste into the
module and the Markdown tables of ``docs/technical-reference/longitudinal-tmle.md``.
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

#: The families of the reference's main constants table, in its order.
MAIN_FAMILIES = (
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

#: The continuous families of the two density routes that build no indicator block (rev 11d),
#: which the reference's second constants table lists.
ROUTE_FAMILIES = (
    "ltmle_continuous_index",
    "sdr_continuous_index",
    "ltmle_continuous_classifier",
    "sdr_continuous_classifier",
)

#: Every family, in the order the reference's range tables list them.
FAMILIES = (
    "ltmle",
    "ltmle_continuous",
    "ltmle_continuous_index",
    "ltmle_continuous_classifier",
    "sdr",
    "sdr_continuous",
    "sdr_continuous_index",
    "sdr_continuous_classifier",
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


def constants_table(families: tuple[str, ...] = MAIN_FAMILIES) -> str:
    """The reference's table of the draw constants: one row per count, one column per family.

    Parameters
    ----------
    families : tuple of str
        The columns: :data:`MAIN_FAMILIES` for the main table, :data:`ROUTE_FAMILIES` for the
        density routes' table.

    Returns
    -------
    str
        The Markdown table.
    """
    table = constants()
    used = [term for term in TERM_COUNTS if any(term in table[family] for family in families)]
    header = "| count | " + " | ".join(f"`{family}`" for family in families) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in families) + " |"
    rows = [header, rule]
    for term in used:
        cells = []
        for family in families:
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


# The accuracy section's readings.  Each one is computed from a calibration output copied byte for
# byte into ``rev9/`` or ``part_d/`` (see the module docstring), and :func:`accuracy_phrases`
# returns the reference's sentences that quote it.

#: The record kind of ``part_d/fit.out``'s checked cells that ran with the default learner
#: library, which the prediction does not count.
GAP = "gap"

#: The four continuous families the part D refit changed.
REFITTED = (
    "ltmle_continuous",
    "sdr_continuous",
    "ltmle_continuous_index",
    "sdr_continuous_index",
)

_CELL = re.compile(
    r"^(?P<kind>\w+)\s+(?P<family>\w+)\s+(?P<n>\d+)\s+(?P<ratio>[\d.]+)\s+(?P<case>est=.*)$"
)
_FAMILY_ROW = re.compile(
    r"^(?P<family>\w+)\s+(?P<n>\d+)\s+(?P<cells>\d+)\s+(?P<low>[\d.]+)\s+[\d.]+\s+(?P<high>[\d.]+)$"
)


def _fit_summary(path: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """The per-family rows of fitted cells and the checked cells of a ``fit.py`` output."""
    families: list[dict[str, str]] = []
    cells: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if match := _FAMILY_ROW.match(line):
            families.append(match.groupdict())
        elif match := _CELL.match(line):
            case = dict(item.split("=", 1) for item in match["case"].split())
            cells.append({**match.groupdict(), **case, "ratio": float(match["ratio"])})
    return families, cells


def _excluded_extension(cell: dict[str, object]) -> bool:
    """Path fits at eight regimens, which the extension table leaves out."""
    return cell["family"] == "hazard_paths" and cell["R"] == "8"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _two(value: float) -> str:
    return f"{value:.2f}"


def accuracy_readings() -> dict[str, object]:
    """The ranges the accuracy and range sections quote, read from the copied outputs.

    The checked cells in the calibrated range are every kind but :data:`GAP`, less the path fits
    at eight regimens.  That holds for the copied rows only, so ``in_range_cells`` is returned for
    the test to pin.
    """
    families, cells = _fit_summary(HERE / "part_d" / "fit.out")
    fitted = [float(row[key]) for row in families for key in ("low", "high")]
    refitted = [
        float(row[key]) for row in families if row["family"] in REFITTED for key in ("low", "high")
    ]
    in_range = [c for c in cells if c["kind"] != GAP and not _excluded_extension(c)]
    low = min(in_range, key=lambda c: c["ratio"])
    high = max(in_range, key=lambda c: c["ratio"])
    gap = [c["ratio"] for c in cells if c["kind"] == GAP]
    held_out = [
        c["ratio"]
        for c in cells
        if c["kind"] in ("bins_check", "dens_check") and c["family"] in REFITTED
    ]
    probes = [c["ratio"] for c in cells if c["kind"] == "rev10"]
    paths = {c["ratio"] for c in cells if _excluded_extension(c)}

    rss = [
        float(m[1])
        for m in re.finditer(r"rss/traced ([\d.]+)", (HERE / "rev9" / "packet10.out").read_text())
    ]

    band = re.findall(
        r"^(\w+)\s+n\s+(\d+) T (\d+) R (\d+) sim ([01])\s+peak\s+([\d.]+) MiB\s+band\s+[\d.]+\s+ratio\s+([\d.]+)$",
        (HERE / "rev9" / "band_check.out").read_text(),
        flags=re.M,
    )
    band_on = [float(row[6]) for row in band if row[4] == "1"]
    band_off = [float(row[6]) for row in band if row[4] == "0"]
    over = sorted(
        (
            (row[0], int(row[1]), int(row[2]), float(row[5]), float(row[6]))
            for row in band
            if row[4] == "1" and float(row[6]) > 3.5
        ),
        key=lambda row: -row[4],
    )

    outside = re.search(
        r"band on: ([\d.]+)", (HERE / "rev9" / "band_check-outside.out").read_text()
    )
    part_d_band = (HERE / "part_d" / "band_check.out").read_text()
    part_d_off = re.search(r"band off: lowest ([\d.]+)", part_d_band)
    part_d_on = re.search(r"band on:\s+lowest [\d.]+ highest ([\d.]+)", part_d_band)

    parallel = [
        float(m[1])
        for m in re.finditer(
            r" j\d+\s+([\d.]+)$", (HERE / "rev9" / "coldterm.out").read_text(), flags=re.M
        )
    ]

    ceiling_families, _ = _fit_summary(HERE / "part_d" / "fit-no-ceiling.out")
    no_ceiling = {
        row["family"]: float(row["high"])
        for row in ceiling_families
        if row["family"].endswith("_index") and row["n"] == "20000"
    }

    summed = [
        (row["family"], int(row["n"]), float(row["high"]))
        for row in _fit_summary(HERE / "rev9" / "fit11c-sum.out")[0]
        if float(row["high"]) > 3.5
    ]

    liblinear = {
        (row["family"], row["B"], row["solver"]): float(row["ratio"])
        for row in _jsonl(HERE / "part_d" / "liblinear-readings.jsonl")
    }

    witness = [
        float(row["margined_ratio"])
        for row in _jsonl(HERE / "rev9" / "witness11.jsonl")
        if "continuous" not in str(row["family"])
    ]

    def traced(name: str, **where: object) -> list[float]:
        return [
            float(row["peak"]) / 2**20
            for row in _jsonl(HERE / "rev9" / name)
            if all(row.get(key) == value for key, value in where.items())
        ]

    route = {"n": 2000, "T": 2, "R": 4, "folds": 1}
    return {
        "fitted": (min(fitted), max(fitted)),
        "refitted": (min(refitted), max(refitted)),
        "in_range_cells": in_range,
        "checked": (low, high),
        "paths": paths,
        "rss": max(rss),
        "band_cells": len(band),
        "band_on": (min(band_on), max(band_on)),
        "band_off": min(band_off),
        "band_over": over,
        "outside": float(outside[1]) if outside else None,
        "parallel": (len(parallel), min(parallel), max(parallel)),
        "gap": (len(gap), min(gap), max(gap)),
        "held_out": (min(held_out), max(held_out)),
        "part_d_band": (
            float(part_d_off[1]) if part_d_off else None,
            float(part_d_on[1]) if part_d_on else None,
        ),
        "no_ceiling": no_ceiling,
        "summed": summed,
        "liblinear": liblinear,
        "witness": (min(witness), max(witness)),
        "probes": (min(probes), max(probes)),
        "binned_route": traced("bins.jsonl", B=320, **route),
        "index_route": traced("dens.jsonl", B=320, density="index", **route),
        "classifier_route": traced("dens.jsonl", density="classifier", **route),
    }


def accuracy_phrases() -> tuple[str, ...]:
    """The reference's phrases that quote :func:`accuracy_readings`, with spaces collapsed."""
    r = accuracy_readings()
    low, high = r["checked"]
    (paths,) = r["paths"]
    count, parallel_low, parallel_high = r["parallel"]
    gaps, gap_low, gap_high = r["gap"]
    over = r["band_over"]
    if len(over) != 3 or {row[0] for row in over} != {"hazard_separable"}:
        raise AssertionError(f"the band cells above 3.5 changed: {over}")
    lib = r["liblinear"]
    ((summed_family, summed_n, summed_high),) = r["summed"]
    return (
        f"Over every fitted cell the margined draw reads {_two(r['fitted'][0])} to {_two(r['fitted'][1])} times the traced peak",
        f"over every checked cell in the calibrated range {_two(low['ratio'])} (a censored `SDR` fit at {int(low['n']):,} units) to {_two(high['ratio'])} (a learned regimen,",
        f"Path fits at eight regimens read {_two(paths)} and are excluded.",
        f"The checked cells read at least {_two(low['ratio'])} times their traced peak.",
        f"any calibration run measured, {_two(r['rss'])} (`rev9/packet10.out`)",
        f"Over {r['band_cells']} cells run with the band on and off",
        f"the band-on cells read {_two(r['band_on'][0])} to {_two(r['band_on'][1])} and the band-off cells {_two(r['band_off'])} and up",
        f"{over[0][4]:.2f} and {over[1][4]:.2f} at {over[0][1]:,} units (outside that family's range) and {over[2][4]:.2f} at {over[2][1]:,}",
        f"traced about {round(max(r['binned_route']), -2):,.0f} MiB (`rev9/bins.jsonl`), {max(r['index_route']):.0f} MiB and {min(r['classifier_route']):.1f} to {max(r['classifier_route']):.1f} MiB (`rev9/dens.jsonl`)",
        f"band-on cells read down to {_two(r['outside'])} (`rev9/band_check-outside.out`)",
        f"Over the {count} parallel runs",
        f"the prediction reads {_two(parallel_low)} to {_two(parallel_high)} times the measured resident footprint",
        f"read {_two(gap_low)} to {_two(gap_high)} times the traced peak on {('eight' if gaps == 8 else str(gaps))} small cells",
        f"Without it the index families read up to {_two(r['no_ceiling']['ltmle_continuous_index'])} and {_two(r['no_ceiling']['sdr_continuous_index'])} at 20,000 units",
        f"Fitted cells now read {_two(r['refitted'][0])} to {_two(r['refitted'][1])}, the held-out cells {_two(r['held_out'][0])} to {_two(r['held_out'][1])}, and their band cells {_two(r['part_d_band'][0])} to {_two(r['part_d_band'][1])}",
        f"read {summed_high:.2f} on a fitted `{summed_family}` cell at {summed_n:,} units",
        f"`LTMLE` read {lib[('ltmle_continuous', 60, 'liblinear')]:.2f}, {lib[('ltmle_continuous', 200, 'liblinear')]:.2f} and {lib[('ltmle_continuous', 320, 'liblinear')]:.2f} times the traced peak at 60, 200 and 320 bins with liblinear ({lib[('ltmle_continuous', 60, 'lbfgs')]:.2f}, {lib[('ltmle_continuous', 200, 'lbfgs')]:.2f} and {lib[('ltmle_continuous', 320, 'lbfgs')]:.2f} with the default solver), and `SDR` read {lib[('sdr_continuous', 60, 'liblinear')]:.2f}, {lib[('sdr_continuous', 200, 'liblinear')]:.2f} and {lib[('sdr_continuous', 320, 'liblinear')]:.2f}",
        f"the `rev10` rows of `part_d/fit.out`, which read {_two(r['probes'][0])} to {_two(r['probes'][1])}",
        f"fast-suite fixtures at 2,000 units read {_two(r['witness'][0])} to {_two(r['witness'][1])} (`rev9/witness11.jsonl`)",
    )


def main() -> None:
    """Print the literals and the table."""
    print(literal())
    print()
    print(constants_table())
    print()
    print(constants_table(ROUTE_FAMILIES))
    print()
    print(ranges_table())
    print()
    print(extensions_table())


if __name__ == "__main__":
    main()


__all__ = [
    "accuracy_phrases",
    "accuracy_readings",
    "constants",
    "constants_table",
    "extensions_table",
    "literal",
    "predicate",
    "ranges_table",
    "workers",
]
