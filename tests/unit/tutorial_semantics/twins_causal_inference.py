"""The static checks for ``docs/examples/twins-causal-inference``.

The TWINS notebook downloads its data, so the offline fast tier cannot execute it.  This module
therefore defines no ``check(namespace)``.  It reads committed text only.
``tests/unit/test_documentation_runtime.py`` compares the narrated decimals against the stored
outputs with :data:`UNPRINTED_DECIMALS`, and calls :func:`check_stored` for the outputs the
readings rely on.  Pin a relation between the prose and the stored outputs here.  Do not pin a
number that the next networked execution can move.
"""

from __future__ import annotations

import nbformat

from tests.notebooks import code_cells
from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "twins-causal-inference.ipynb"

#: Decimals the prose writes that no stored output prints, each with the reason.
UNPRINTED_DECIMALS = {
    "4.3": "a section number of the cited Louizos et al. (2017) paper, not a result",
    "0.7": (
        "the lower calibration-slope limit of the nuisance verdict "
        "(src/cleverly/validation/nuisance.py), which no output prints"
    ),
    "1.4": (
        "the upper calibration-slope limit of the nuisance verdict "
        "(src/cleverly/validation/nuisance.py), which no output prints"
    ),
}


def check_stored() -> None:
    """The TWINS artifact retains its figures, outcome, protocol, and typed contrasts."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(str(cell["source"]) for cell in notebook.cells)
    figures = [
        output
        for cell in code_cells(notebook)
        for output in cell.get("outputs", ())
        if "image/png" in output.get("data", {})
    ]
    ordinary_tmle_row = next(
        line
        for line in stored_output(NOTEBOOK, "comparison-figure").splitlines()
        if "ordinary package TMLE" in line
    )
    load_text = stored_output(NOTEBOOK, "load-data")

    assert len(figures) >= 3, "the TWINS notebook lost one or more evidence figures"
    assert "NaN" not in ordinary_tmle_row, (
        "the ordinary package TMLE lost its confidence interval in the comparison figure"
    )
    assert "mortality_3y" not in source and "three-year mortality" not in load_text
    assert "first-year mortality" in load_text
    identify_text = stored_output(NOTEBOOK, "identify")
    assert "causal study protocol: schema" in identify_text
    assert "causal study protocol: absent" not in identify_text
    scales_html = stored_output(NOTEBOOK, "effect-scales")
    assert "<td>risk ratio</td>" in scales_html
    assert "not shown" not in scales_html
