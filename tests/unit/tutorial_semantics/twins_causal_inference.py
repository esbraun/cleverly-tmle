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

_PROBE = "reviews/notebook-review/probes/twins-causal-inference-final"

#: Decimals the prose writes that no stored output prints, each with the reason.
UNPRINTED_DECIMALS = {
    "4.3": "a section number of the cited Louizos et al. (2017) paper, not a result",
    "4.091": f"smallest nu^2 over 31 pair samples, shallow booster: {_PROBE}/summary.log",
    "4.224": f"largest nu^2 over 31 pair samples, shallow booster: {_PROBE}/summary.log",
    "0.0019": f"largest benchmark cf_y over 31 pair samples: {_PROBE}/summary.log",
    "0.0023": f"smallest benchmark cf_d over 31 pair samples: {_PROBE}/summary.log",
    "0.0225": f"largest benchmark cf_d over 31 pair samples: {_PROBE}/summary.log",
    "0.0163": f"mean naive-contrast error over 60 synthetic draws: {_PROBE}/summary.log",
    "0.0134": f"mean LTMLE standard error over 60 synthetic draws: {_PROBE}/summary.log",
}

#: A printed residual such as 3.79e-13 is machine noise that changes between machines.
_MACHINE_NOISE = re.compile(r"\d\.\d+e-(0[89]|[1-9]\d)")


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
    # The source file removed equal-weight pairs; the eligibility and the readings say so.
    assert "pairs in the source file:   71,345" in load_text and "71,345 pairs" in prose
    assert "equal-weight source pairs:  0" in load_text
    # The stress test needs no model, so it runs on every discordant pair of the full file.
    association = stored_output(NOTEBOOK, "association-stress-test")
    assert "14,870 pairs, full file" in association and "14,870" in prose
    assert 'raw["T"]' in code["association-stress-test"]

    # Protocol: printed in its own step, and carried by the identified effect and the fit.
    protocol_text = stored_output(NOTEBOOK, "protocol")
    match = re.search(r"causal study protocol: schema \d+; ([0-9a-f]{16})", protocol_text)
    assert match, "the protocol step no longer prints its fingerprint"
    fingerprint = match.group(1)
    assert "the source file removed equal-weight pairs" in protocol_text
    identify_text = stored_output(NOTEBOOK, "identify")
    assert f"causal study protocol: schema 1; {fingerprint}" in identify_text
    assert "causal study protocol: absent" not in identify_text
    # The later summaries print the fingerprint line of the record and none of its fields.
    production = stored_output(NOTEBOOK, "production-fit")
    assert f"causal study protocol: schema 1; {fingerprint}" in production
    assert "assumption rationale:" not in identify_text + production
    assert f"`{fingerprint}`" in prose

    # Hand-built TMLE: the package construction, with the nonzero witnesses the reading narrates.
    manual = code["manual-estimators"]
    assert "LogisticRegression(max_iter=3_000, random_state=SEED)" in manual
    assert manual.count("with thread_limit():") == 2
    assert "np.where(A == 1, q1_star, q0_star)" in manual
    assert "H = np.column_stack([A / g1, (1 - A) / (1 - g1)])" in manual
    assert "G_BOUND = 5 / (np.sqrt(n) * np.log(n))" in manual and "Q_BOUND = 0.0005" in manual
    assert "np.abs(initial_scores).max() > 1e-5" in manual
    ladder = stored_output(NOTEBOOK, "manual-estimators")
    assert "g bound:  [0.004859, 0.9951]" in ladder and "Q bound:  [0.0005, 0.9995]" in ladder
    assert _number(ladder, r"the Q bound raises: (\d+)") > 0
    before = [_number(ladder, rf"score before targeting, arm {arm}:\s+(\S+)") for arm in (1, 0)]
    epsilon = [_number(ladder, rf"fluctuation epsilon, arm {arm}:\s+(\S+)") for arm in (1, 0)]
    assert max(abs(value) for value in before) > 1e-5 and min(map(abs, epsilon)) > 0
    assert "both scores after targeting below 1e-10: True" in ladder
    assert not _MACHINE_NOISE.search(ladder)

    # Package agreement: the gap is below a thousandth of the package standard error.
    package = _number(production, r"ordinary package TMLE:\s+(\S+)")
    hand = _number(production, r"hand-built TMLE:\s+(\S+)")
    assert package == hand, (package, hand)
    assert "|gap| / standard error below 0.001: True" in production
    assert _number(production, r"\|gap\| / standard error, rounded:\s+(\S+)") < 1e-3
    assert "assert gap_ratio < 1e-3" in code["production-fit"]
    assert "clusters = 6000 (cluster-robust variance)" in production
    # The booster is regularized: scikit-learn stops it early only above 10,000 rows.
    assert "max_depth=2" in code["production-fit"]
    assert "l2_regularization=1.0" in code["production-fit"]

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
    # "The rule flags neither model on this pair sample."
    assert "VERDICT: nuisance fits look reasonable." in diagnostics
    assert "more extreme than the observed rates (" not in diagnostics
    assert "rm15-calibration-slope-warning-rule" in prose, (
        "the reading must send the calibration flag to the roadmap item that replaced the band"
    )
    # The score check prints pass or fail and a rounded ratio, never a raw residual.
    assert diagnostics.count("passed=True; |score| / threshold = 0.0000") == 2
    assert not _MACHINE_NOISE.search(diagnostics)
    overlap = stored_output(NOTEBOOK, "overlap")
    assert "n = 12000; propensity truncated to [0.004859, 0.9951]" in overlap
    assert re.search(r"^0\.05\s+0\.0000\s+0\.0000\s*$", overlap, re.MULTILINE)
    assert re.search(r"^0\.1\s+0\.0000\s+0\.0002\s*$", overlap, re.MULTILINE)
    assert "truncated: 0 unit(s) (0.00%)" in overlap
    # "The maximum clever covariate is 1/(1 - g) for the largest control propensity."
    largest_control = _number(overlap, r"(?m)^control(?:\s+\S+){8}\s+(\S+)\s*$")
    clever = _number(overlap, r"max \|clever covariate\| \(mean\): (\S+)")
    assert math.isclose(clever, 1 / (1 - largest_control), abs_tol=0.01), (clever, largest_control)

    # Sensitivity: the bound runs on a nonnegative nu^2, the sign survives, and the benchmark
    # gives no outcome-side scale.
    sensitivity = stored_output(NOTEBOOK, "sensitivity")
    assert "(the sign of the effect survives)" in sensitivity
    assert _number(diagnostics, r"nu2=(\S+),") > 0
    cf_y = _number(sensitivity, r"implied cf_y = (\S+),")
    cf_d = _number(sensitivity, r"implied cf_y = \S+, cf_d = (\S+),")
    assert 0 <= cf_y < cf_d < 0.05, (cf_y, cf_d)
    sigma = re.search(r"(?m)^sigma\^2\s+(\S+)\s+(\S+)\s*$", sensitivity)
    assert sigma, "the benchmark no longer prints sigma^2 with and without the covariates"
    assert 0 <= float(sigma.group(2)) - float(sigma.group(1)) < 1e-4
    assert "about four times" not in prose

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
    assert len(re.findall(r"solver\s+True\s+0\.0000", long_diagnostics)) == 4
    assert not _MACHINE_NOISE.search(long_diagnostics)
