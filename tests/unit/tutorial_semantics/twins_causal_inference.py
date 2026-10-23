"""The static checks for ``docs/examples/twins-causal-inference``.

The TWINS notebook downloads its data, so the offline fast tier cannot execute it.  This module
therefore defines no ``check(namespace)``.  It reads committed text only.
``tests/unit/test_documentation_runtime.py`` compares the narrated decimals against the stored
outputs with :data:`UNPRINTED_DECIMALS`, and calls :func:`check_stored` for the outputs the
readings rely on.  Pin a relation between the prose and the stored outputs here.  Do not pin a
number that the next networked execution can move.
"""

from __future__ import annotations

import re

import nbformat

from tests.notebooks import code_cells
from tests.unit.tutorial_semantics import EXAMPLES, markdown_text, notebook_code, stored_output

NOTEBOOK = EXAMPLES / "twins-causal-inference.ipynb"

_PROBES = EXAMPLES.parents[1] / "reviews/notebook-review/probes/twins-birth-weight"

#: The review probe of the generated-outcome test at 400 draws.
PROBE_LOG = _PROBES / "generated_outcome_400.log"

#: The review probe that benchmarks each covariate and finds the two tipping strengths.
SENSITIVITY_LOG = _PROBES / "sensitivity_all.log"

#: Decimals the prose writes that no stored output prints, each with the reason.
_PROBE = "reviews/notebook-review/probes/twins-birth-weight/generated_outcome_400.log"

UNPRINTED_DECIMALS: dict[str, str] = {
    "0.03182": f"bias reading of the generated-outcome test, 400 draws: {_PROBE}",
    "-0.06618": f"lower limit of that bias interval, 400 draws: {_PROBE}",
    "0.1298": f"upper limit of that bias interval, 400 draws: {_PROBE}",
    "-0.2": f"lower limit of the level-2 equivalence band: {_PROBE}",
    "0.2": f"upper limit of the level-2 equivalence band: {_PROBE}",
}

#: A printed residual such as 3.79e-13 is machine noise that changes between machines.
_MACHINE_NOISE = re.compile(r"\d\.\d+e-(0[89]|[1-9]\d)")


def _number(text: str, pattern: str) -> float:
    """The first number captured by ``pattern`` in ``text``, without thousands separators."""
    match = re.search(pattern, text)
    assert match, f"the TWINS notebook no longer prints {pattern!r}"
    return float(match.group(1).replace(",", ""))


def _row(text: str, label: str) -> list[float]:
    """The numbers of the printed table row that starts with ``label``."""
    match = re.search(rf"(?m)^\s*{re.escape(label)}\s+(.+)$", text)
    assert match, f"the TWINS notebook no longer prints the row {label!r}"
    return [float(value.replace(",", "").lstrip("+")) for value in match.group(1).split()]


def _cell_prose(notebook: nbformat.NotebookNode, cell_id: str) -> str:
    """The source of the markdown cell ``cell_id``."""
    for cell in notebook.cells:
        if cell.get("id") == cell_id:
            assert cell.cell_type == "markdown", cell_id
            return str(cell.source)
    raise AssertionError(f"the TWINS notebook has no cell {cell_id!r}")


def check_stored() -> None:
    """The TWINS artifact retains the outputs and the relations its readings narrate."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = dict(notebook_code(NOTEBOOK))
    prose = markdown_text(NOTEBOOK)
    figures = [
        output
        for cell in code_cells(notebook)
        for output in cell.get("outputs", ())
        if "image/png" in output.get("data", {})
    ]
    assert len(figures) == 2, (
        "the TWINS notebook has a gestational-age figure and an overlap figure"
    )

    # Every code cell is followed by a markdown reading, and the page keeps its landmarks.
    cells = notebook.cells
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            following = cells[index + 1] if index + 1 < len(cells) else None
            assert following is not None and following.cell_type == "markdown", cell.id
    headings = re.findall(r"(?m)^#{1,3} .+$", prose)
    assert headings[1] == "## At a glance", headings[:2]
    assert not any(re.match(r"#+ \d", heading) for heading in headings), headings
    conclusion = prose.index("## What we can and can't conclude")
    assert conclusion < prose.index("## Go deeper")

    # The outcome is birth weight. No text names mortality as the outcome.
    assert "mortality" not in prose.lower() and "death_share" not in "".join(code.values())

    # The graph names gestational age as the mediator, and U as causes that smoking does not cause.
    assert "```text" in prose
    assert "M = gestational age at birth" in prose
    assert "that smoking does not\n      cause" in prose

    # The selection on equal weights, the live-birth condition, and the plausibility rule are
    # stated before the estimate.
    before_fit = prose[: prose.index("## Estimate the effect")]
    assert "dropped pairs with equal weights" in before_fit
    assert "two live births with unequal weights" in before_fit
    assert "### Are the birth weights plausible?" in before_fit

    # Setup: no global warning filter hides the library's warnings.
    assert "filterwarnings" not in code["setup"]

    # Data: the pinned source, the state-year rule, and the counts the reading quotes.
    load = code["load-data"]
    assert "AMLab-Amsterdam/CEVAE/{COMMIT}" in load
    assert 'COMMIT = "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd"' in load
    assert 'groupby(state_year).transform("mean")' in load and "missing_share <= 0.5" in load
    assert '"birth_weight_g": weights.mean(axis=1)' in load
    load_text = stored_output(NOTEBOOK, "load-data")
    for count in ("71,345", "16,739", "2,732", "51,874"):
        assert count in load_text and count in prose, count
    # The excluded residence-state codes are the states that the NCHS guides name.
    excluded = {int(code) for code in re.findall(r"(?m)^\s+(\d+)\s+19\d\d", load_text)}
    assert {5, 15, 19, 28, 33, 37, 42} <= excluded, excluded
    # No mediator enters the adjustment set.
    for mediator in ("gestat10", "phyper", "eclamp", "nprevistq", "adequacy"):
        assert mediator not in load, mediator

    # Plausibility: the range comes from the record layout, and the rule removes no pair.
    plausibility = code["plausibility"]
    assert "NCHS_RANGE = (227.0, 8_165.0)" in plausibility
    checks = stored_output(NOTEBOOK, "plausibility")
    assert _number(checks, r"missing birth weights:\s+(\d+)") == 0
    assert _number(checks, r"pairs with equal birth weights:\s+(\d+)") == 0
    inside = re.search(r"pairs inside the NCHS range:\s+([\d,]+) of ([\d,]+)", checks)
    assert inside and inside[1] == inside[2] == "51,874"
    for label, count in (
        ("twins at the floor of 227 g", "13"),
        ("twins over 5,000 g", "5"),
        ("pairs over 40% discordant", "731"),
        ("pairs at 20-27 weeks with a twin of 2,500 g+", "28"),
    ):
        assert _number(checks, rf"{re.escape(label)}:\s+(\d+)") == int(count), label
    references = (EXAMPLES.parent / "references.md").read_text(encoding="utf-8")
    assert '0227 to 8165, with 9999 for "not stated"' in references

    # Association: the two conditions of a confounder hold for marital status and education,
    # and birth order works the other way.
    association = stored_output(NOTEBOOK, "association")
    for label in ("unmarried", "mother_education 1-2"):
        smoking, non_smoking, with_it, without_it = _row(association, label)
        assert smoking > non_smoking and with_it < without_it, label
    smoking, non_smoking, with_it, without_it = _row(association, "birth_order 4+")
    assert smoking > non_smoking and with_it > without_it
    crude = _number(association, r"crude difference: (\S+) g,")

    # Protocol: printed in its own step, and carried by the identified effect and the fit.
    protocol_text = stored_output(NOTEBOOK, "protocol")
    match = re.search(r"causal study protocol: schema \d+; ([0-9a-f]{16})", protocol_text)
    assert match, "the protocol step no longer prints its fingerprint"
    fingerprint = match.group(1)
    identify_text = stored_output(NOTEBOOK, "identify")
    fit_text = stored_output(NOTEBOOK, "att-fit")
    for text in (identify_text, fit_text):
        assert f"causal study protocol: schema 1; {fingerprint}" in text
        assert "assumption rationale:" not in text
    intercurrent = "intercurrent-event handling: ['Fetal death of a twin: the cohort conditions on"
    assert intercurrent in protocol_text
    assert "No tobacco use at any time in the pregnancy" in protocol_text
    # The identify reading names two judgment assumptions beyond confounding and positivity.
    assert identify_text.count("\n  - ") == 5
    for assumption in ("consistency", "no interference", "no unmeasured confounding"):
        assert f"  - {assumption}:" in identify_text, assumption

    # Design: the pair is the unit, so no cluster; the ATT; the declared outcome support.
    assert "cluster=" not in code["identify"] and "ATT(reference=0)" in code["identify"]
    assert 'outcome="birth_weight_g"' in code["identify"]
    assert "Targeting(q_bounds=NCHS_RANGE)" in code["att-fit"]
    assert "CrossFitting(n_folds=3, learner_folds=3)" in code["att-fit"]
    assert "stacked CV-TMLE (Levy): nuisances cross-fitted over 3 folds" in fit_text
    assert "outcome scaled from [227, 8165] to [0, 1]" in fit_text
    assert "ATT/ATC to [0.025, 0.975]" in fit_text
    assert "clusters =" not in fit_text
    initial = re.search(r"initial outcome predictions: ([\d,]+) g to ([\d,]+) g", fit_text)
    assert initial
    low, high = (float(value.replace(",", "")) for value in initial.groups())
    assert 227.0 <= low <= high <= 8165.0

    # The ATT and the crude difference share the observed mean of the smoking pairs.
    observed = _number(fit_text, r"observed mean weight, smoking pairs:\s+(\S+) g")
    counterfactual = _number(fit_text, r"had they not smoked:\s+(\S+) g")
    reference = _number(fit_text, r"mean weight, non-smoking pairs:\s+(\S+) g")
    att = _number(fit_text, r"ATT: (\S+) g,")
    interval = re.search(r"ATT: \S+ g, 95% CI \[(\S+), (\S+)\]", fit_text)
    assert interval and float(interval[2]) < 0, "the ATT interval no longer excludes zero"
    assert abs(observed - counterfactual - att) <= 0.15
    assert counterfactual < reference and crude < att < 0
    # "At a glance" rounds the estimate and the interval of the fit.
    glance = _cell_prose(notebook, "title")
    assert f"{round(-att)} g on average" in glance
    assert f"interval of {round(-float(interval[2]))} to {round(-float(interval[1]))} g" in glance
    assert f"{round(-crude)} g, overstates it" in glance

    # Gestational age: smoking pairs weigh less in every band from 32 weeks on, and more at
    # 20 to 27 weeks.
    gestation = stored_output(NOTEBOOK, "gestational-age")
    shares = re.search(r"non-smoking (\S+), smoking (\S+)", gestation)
    assert shares and float(shares.group(2)) > float(shares.group(1))
    for band in ("32-35 weeks", "36 weeks", "37-39 weeks", "40 weeks", "41 weeks"):
        assert _row(gestation, band)[-1] < 0, band
    assert _row(gestation, "20-27 weeks")[-1] > 0
    assert "42 weeks or more" in gestation
    assert "kwj275" in references

    # Assessment: two checks need attention, the slope interval lies above 1, and the score
    # equation passed.
    diagnostics = stored_output(NOTEBOOK, "assessment")
    attention = re.search(r"checks that need attention: \(([^)]*)\)", diagnostics)
    assert attention, "the assessment no longer prints its attention list"
    assert sorted(re.findall(r"'(\w+)'", attention.group(1))) == ["nuisance_models", "support"]
    slope = re.search(r"calibration slope: (\S+) \(95% interval (\S+) to (\S+)\)", diagnostics)
    assert slope and float(slope[2]) > 1, "the reading says the slope interval lies above 1"
    assert diagnostics.rstrip().endswith(
        "PASS: the targeting step solved the estimated efficient score equation."
    )
    assert _number(diagnostics, r"highest probability of smoking:\s+(\S+)") < 0.975
    assert not _MACHINE_NOISE.search(diagnostics)
    assert "unavailable" not in diagnostics and "not_applicable" not in diagnostics

    # Overlap: almost all pairs at the lower limit are non-smoking pairs, and the limit does
    # not carry the estimate.
    overlap = stored_output(NOTEBOOK, "overlap")
    below = re.search(r"lower limit of 0.025: non-smoking ([\d,]+), smoking ([\d,]+)", overlap)
    assert below and int(below[1].replace(",", "")) > 50 * int(below[2].replace(",", ""))
    clipped = _number(diagnostics, r"raised to 0.025: ([\d,]+)")
    assert clipped == int(below[1].replace(",", "")) + int(below[2].replace(",", ""))
    curve = [
        float(match.group(1))
        for match in re.finditer(r"(?m)^\s*0\.0\d+\s+\d\.\d+\s+(-\d+\.\d+)\s*$", overlap)
    ]
    assert len(curve) == 5, curve
    standard_error = _number(fit_text, r"(?m)^att\s+\S+\s+(\S+)")
    assert max(curve) - min(curve) < 0.1 * standard_error, curve

    # Refutation: the three tests ran, every verdict is descriptive, and no refusal appears.
    refutation = stored_output(NOTEBOOK, "refute")
    for test in ("placebo", "subset", "random_common_cause"):
        assert re.search(rf"(?m)^\s*{test}\s.*\bdescriptive\s*$", refutation), test
    # The placebo mean lies within two of its own standard deviations over 20 draws of zero.
    placebo = re.search(r"(?m)^\s*placebo\s+\S+\s+(\S+)\s+(\S+)", refutation)
    assert placebo and abs(float(placebo[1])) < 2 * float(placebo[2]) / 20**0.5
    shift = _number(refutation, r"paired shift in standard errors = (\S+)")
    assert abs(shift) < 0.1, "the reading says the noise column moves under a tenth of an SE"
    rho = _number(refutation, r"scatter ratio rho = (\S+)")
    assert rho > 1, "the reading says the 70% refits spread more than the standard error predicts"
    for cell_id in code:
        text = stored_output(NOTEBOOK, cell_id)
        assert "CapabilityError" not in text and "refus" not in text.lower(), cell_id

    # Generated outcome: ten draws at level L1, below the calibrated minimum.
    generated = stored_output(NOTEBOOK, "refute-generated")
    assert "draws: 10; level L1; verdict descriptive" in generated
    assert (
        "BoundedAdjustmentOutcome(intercept=-1.0, effect=-0.1, adjustment_scale=0.2)"
        in code["refute-generated"]
    )

    # Sensitivity: every covariate is benchmarked, birth order is strongest on the outcome side
    # and alcohol on the smoking side, the bound at both excludes zero, and the tipping
    # strength is about twice birth order's.
    sensitivity = stored_output(NOTEBOOK, "sensitivity")
    assert "robustness value: " in sensitivity
    rows = re.findall(r"(?m)^\s*(\w+)\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(-?\d+\.\d+)\s*$", sensitivity)
    assert len(rows) == 18, rows
    assert rows[0][0] == "birth_order", rows[0]
    assert max(rows, key=lambda row: float(row[2]))[0] == "alcohol"
    limits = re.search(r"one-sided 95% limits: \[(\S+), (\S+)\] g", sensitivity)
    assert limits and float(limits[2]) < 0, "the bound at the strongest benchmarks reaches zero"
    tipping = re.search(r"pulls the upper limit to zero: (\S+), (\S+) times", sensitivity)
    assert tipping and 1.5 < float(tipping[2]) < 2.5, "the reading says about twice birth order"
    log = SENSITIVITY_LOG.read_text(encoding="utf-8")
    assert "cf_y tipping at the largest cf_d: 0.0278 (1.9x largest cf_y)" in log
    assert "cf_d tipping the 95% upper limit at cf_y = 1: 0.0062" in log

    # The E-value uses the standardized-difference approximation for a continuous outcome.
    evalue = stored_output(NOTEBOOK, "evalue")
    assert "(approximate conversion)" in evalue and "Standardised by sd(Y)" in evalue
    assert _number(evalue, r"E-value, confidence limit\s+:\s+(\S+)") > 1

    # Simulated confounding: nine cells calibrated on the two benchmark covariates.
    surface = stored_output(NOTEBOOK, "simulated-confounding")
    grid_rows = re.findall(
        r"(?m)^\s*(0\.0[025])\s+(-?0\.1|0\.0)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s*$",
        surface,
    )
    assert len(grid_rows) == 9, grid_rows
    assert 'benchmark_covariates=("mother_education", "married")' in code["simulated-confounding"]
    for covariate in ("mother_education", "married"):
        assert re.search(rf"(?m)^\s+{covariate}\s+treatment\s", surface), covariate
        assert re.search(rf"(?m)^\s+{covariate}\s+outcome\s", surface), covariate
    assert "outcome=(-0.1, 0.0, 0.1)" in code["simulated-confounding"]
    by_cell = {(row[0], row[1]): [float(value) for value in row[2:]] for row in grid_rows}
    # A positive outcome strength lowers the weight, and a negative one raises it.
    assert by_cell[("0.00", "0.1")][3] < 0 < by_cell[("0.00", "-0.1")][3]
    # The threat to the sign moves the estimate toward zero, and the estimate stays negative.
    threat = by_cell[("0.02", "-0.1")]
    assert threat[0] < 0 < threat[1]
    assert by_cell[("0.02", "0.1")][0] < 0

    # Over-adjustment: each refit moves the ATT by under one standard error.
    refit = stored_output(NOTEBOOK, "over-adjustment")
    moves = [float(value) for value in re.findall(r", ([+-]\d+\.\d+) standard errors", refit)]
    assert len(moves) == 2 and all(abs(move) < 1 for move in moves), moves
    assert '"previous_infant_over_4000g"' in code["over-adjustment"]

    # The review probe of 400 generated-outcome draws is committed with its reading.
    probe = PROBE_LOG.read_text(encoding="utf-8")
    assert "successful=400; failed=0" in probe
    assert "L2    yes" in probe
    assert "bias 0.03182 [-0.06618, 0.1298] inside [-0.2, 0.2]" in probe
