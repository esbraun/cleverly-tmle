"""The static checks for ``docs/examples/twins-causal-inference``.

The TWINS notebook downloads its data, so the offline fast tier cannot execute it.  This module
therefore defines no ``check(namespace)``.  It reads committed text only.
``tests/unit/test_documentation_runtime.py`` compares the narrated decimals against the stored
outputs with :data:`UNPRINTED_DECIMALS`, and calls :func:`check_stored` for the outputs the
readings rely on.  Pin a relation between the prose and the stored outputs here.  Do not pin a
number that the next networked execution can move.
"""

from __future__ import annotations

import math
import re

import nbformat

from tests.notebooks import code_cells
from tests.unit.tutorial_semantics import EXAMPLES, markdown_text, notebook_code, stored_output

NOTEBOOK = EXAMPLES / "twins-causal-inference.ipynb"

#: Decimals the prose writes that no stored output prints, each with the reason.
UNPRINTED_DECIMALS = {
    "4.3": "a section number of the cited Louizos et al. (2017) paper, not a result",
}


def _number(text: str, pattern: str) -> float:
    """The first number captured by ``pattern`` in ``text``."""
    match = re.search(pattern, text)
    assert match, f"the TWINS notebook no longer prints {pattern!r}"
    return float(match.group(1))


def check_stored() -> None:
    """The TWINS artifact retains the outputs and the relations its readings narrate."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = dict(notebook_code(NOTEBOOK))
    source = "\n".join(str(cell["source"]) for cell in notebook.cells)
    prose = markdown_text(NOTEBOOK)
    figures = [
        output
        for cell in code_cells(notebook)
        for output in cell.get("outputs", ())
        if "image/png" in output.get("data", {})
    ]
    assert len(figures) >= 3, "the TWINS notebook lost one or more evidence figures"

    # Every code cell is followed by its reading.
    cells = notebook.cells
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            following = cells[index + 1] if index + 1 < len(cells) else None
            assert following is not None and following.cell_type == "markdown", cell.id
            assert str(following.source).startswith("**What this output tells you.**"), cell.id

    # Setup: no global warning filter hides the library's positivity or convergence warnings.
    assert "filterwarnings" not in code["setup"]
    assert "from cleverly.learners import thread_limit" in code["setup"]

    # Data: first-year outcome, and no variable that the birth itself determines.
    load_text = stored_output(NOTEBOOK, "load-data")
    assert "mortality_3y" not in source and "three-year mortality" not in load_text
    assert "first-year mortality" in load_text
    assert "birth_order" not in code["load-data"]
    assert "birth_order" not in stored_output(NOTEBOOK, "identify")
    assert "encoded adjustment columns: 47" in load_text and "47 adjustment columns" in prose
    assert "1,290 discordant pairs" in stored_output(NOTEBOOK, "association-stress-test")

    # Protocol: printed in its own step, and carried by the identified effect and the fit.
    protocol_text = stored_output(NOTEBOOK, "protocol")
    match = re.search(r"causal study protocol: schema \d+; ([0-9a-f]{16})", protocol_text)
    assert match, "the protocol step no longer prints its fingerprint"
    fingerprint = match.group(1)
    identify_text = stored_output(NOTEBOOK, "identify")
    assert f"causal study protocol: schema 1; {fingerprint}" in identify_text
    assert "causal study protocol: absent" not in identify_text
    assert f"protocol {fingerprint}" in stored_output(NOTEBOOK, "production-fit")
    assert f"`{fingerprint}`" in prose

    # Hand-built TMLE: the nonzero witnesses the reading narrates stay asserted and printed.
    manual = code["manual-estimators"]
    assert "LogisticRegression(max_iter=3_000, random_state=SEED)" in manual
    assert manual.count("with thread_limit():") == 2
    assert "np.where(A == 1, q1_star, q0_star)" in manual
    assert "abs(initial_score) > 1e-5" in manual
    ladder = stored_output(NOTEBOOK, "manual-estimators")
    before = _number(ladder, r"residual score before targeting\s+(\S+)")
    after = _number(ladder, r"residual score after targeting\s+(\S+)")
    epsilon = _number(ladder, r"fluctuation epsilon\s+(\S+)")
    assert abs(before) > 1e-5 and abs(after) < 1e-10 and abs(epsilon) > 0

    # Package agreement: the gap is far below the targeting move it would otherwise hide in.
    production = stored_output(NOTEBOOK, "production-fit")
    gap = _number(production, r"ordinary package TMLE minus hand-built TMLE:\s+(\S+)")
    shift = _number(production, r"hand-built TMLE minus g-computation:\s+(\S+)")
    assert abs(gap) < 0.1 * abs(shift), (gap, shift)
    assert "0.1 * abs(targeting_shift)" in code["production-fit"]
    assert "clusters = 6000 (cluster-robust variance)" in production

    ordinary_tmle_row = next(
        line
        for line in stored_output(NOTEBOOK, "comparison-figure").splitlines()
        if "ordinary package TMLE" in line
    )
    assert "NaN" not in ordinary_tmle_row, (
        "the ordinary package TMLE lost its confidence interval in the comparison figure"
    )

    # Assessment: one assess() call, and the verdicts the readings quote.
    assert ".assess(" in code["diagnostics"] and "run_all" not in code["diagnostics"]
    diagnostics = stored_output(NOTEBOOK, "diagnostics")
    assert "needs attention: ()" in diagnostics
    assert "passed  1      validation.score_equations" in diagnostics
    assert "VERDICT: nuisance fits look reasonable." in diagnostics
    overlap = stored_output(NOTEBOOK, "overlap")
    assert "n = 12000; propensity truncated to [0.004859, 0.9951]" in overlap
    assert re.search(r"^0\.05\s+0\.0000\s+0\.0002\s*$", overlap, re.MULTILINE)
    assert "truncated: 0 unit(s) (0.00%)" in overlap

    # Sensitivity: the sign survives, and the benchmark gives no outcome-side scale.
    sensitivity = stored_output(NOTEBOOK, "sensitivity")
    assert "(the sign of the effect survives)" in sensitivity
    assert _number(sensitivity, r"implied cf_y = (\S+),") == 0.0
    cf_d = _number(sensitivity, r"implied cf_y = \S+, cf_d = (\S+),")
    assert 3.5 < 0.05 / cf_d < 4.5, "the reading says cf_d = 0.05 is about four times the benchmark"

    # Scales: the ratio interval is symmetric on the log scale, and the E-value follows the ratio.
    scales = stored_output(NOTEBOOK, "effect-scales")
    assert "risk ratio" in scales and "not shown" not in scales
    upper = _number(scales, r"to its upper limit: (\S+)")
    lower = _number(scales, r"to its lower limit: (\S+)")
    assert math.isclose(upper, lower, abs_tol=1e-4)
    ratio = _number(scales, r"risk ratio\s+(\S+)")
    evalue = _number(diagnostics, r"evalue\s+point=(\S+),")
    assert math.isclose(ratio + math.sqrt(ratio * (ratio - 1)), evalue, abs_tol=0.1)

    # LTMLE: the seeded relations the reading states, printed so this module can read them.
    ltmle = stored_output(NOTEBOOK, "semisynthetic-fit")
    assert "LTMLE interval contains the exact truth: True" in ltmle
    assert "LTMLE interval contains the naive contrast: True" in ltmle
    assert "method=sequential_method" in code["semisynthetic-fit"]
    long_diagnostics = stored_output(NOTEBOOK, "ltmle-diagnostics")
    assert "1 role omission(s) are recorded" in long_diagnostics
