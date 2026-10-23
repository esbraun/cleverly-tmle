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

#: The review probe of the generated-outcome test at 400 draws.
PROBE_LOG = (
    EXAMPLES.parents[1]
    / "reviews/notebook-review/probes/twins-birth-weight/generated_outcome_400.log"
)

#: Decimals the prose writes that no stored output prints, each with the reason.
UNPRINTED_DECIMALS: dict[str, str] = {}

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

    # Every code cell is followed by its reading.
    cells = notebook.cells
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            following = cells[index + 1] if index + 1 < len(cells) else None
            assert following is not None and following.cell_type == "markdown", cell.id
            assert str(following.source).startswith("**What this output tells you.**"), cell.id

    # The outcome is birth weight. No text names mortality as the outcome.
    assert "mortality" not in prose.lower() and "death_share" not in "".join(code.values())
    assert "| outcome Y | the mean of the two twins' recorded birth weights, in grams |" in prose
    assert "\\| A = 1], in grams |" in prose  # the ATT row of the estimand table

    # The situation states a causal graph with one row per arrow or absent arrow, and names
    # gestational age as the mediator.
    assert "```text" in prose and "| arrow or absent arrow | what it asserts |" in prose
    assert "no U → A" in prose and "A → M ← U → Y" in prose
    assert "| M | gestational age at birth" in prose

    # The selection on equal weights, the live-birth condition, and the plausibility rule are
    # stated before the estimate.
    before_fit = prose[: prose.index("## 6. Estimate the ATT")]
    assert "selection on Y" in before_fit
    assert "does not change the chance that a pregnancy ends in two live births" in before_fit
    assert "### The plausibility rule" in before_fit

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
    references = (EXAMPLES.parent / "references.md").read_text(encoding="utf-8")
    assert '0227 to 8165, with 9999 for "not stated"' in references

    # Association: the two conditions of a confounder hold for marital status and education.
    association = stored_output(NOTEBOOK, "association")
    for label in ("unmarried", "mother_education 1-2"):
        smoking, non_smoking, with_it, without_it = _row(association, label)
        assert smoking > non_smoking and with_it < without_it, label
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
    assert f"`{fingerprint}`" in prose
    assert "intercurrent-event handling: ['Fetal death of a twin: while alive." in protocol_text

    # Design: the pair is the unit, so no cluster; the ATT; the declared outcome support.
    assert "cluster=" not in code["identify"] and "ATT(reference=0)" in code["identify"]
    assert 'outcome="birth_weight_g"' in code["identify"]
    assert "Targeting(q_bounds=NCHS_RANGE)" in code["att-fit"]
    assert "CrossFitting(n_folds=3, learner_folds=3)" in code["att-fit"]
    assert "stacked CV-TMLE (Levy): nuisances cross-fitted over 3 folds" in fit_text
    assert "outcome scaled from [227, 8165] to [0, 1]" in fit_text
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
    # "Adjustment for C moves that mean from 2,427.2 g to 2,405.0 g."
    assert counterfactual < reference and crude < att < 0

    # Gestational age: smoking pairs weigh less in every band from 32 weeks on.
    gestation = stored_output(NOTEBOOK, "gestational-age")
    shares = re.search(r"non-smoking (\S+), smoking (\S+)", gestation)
    assert shares and float(shares.group(2)) > float(shares.group(1))
    for band in ("32-35 weeks", "36 weeks", "37-39 weeks", "40 weeks", "41 weeks"):
        assert _row(gestation, band)[-1] < 0, band
    assert "42 weeks or more" in gestation
    assert "kwj275" in references

    # Assessment: the score check passed, and the reading explains every attention item.
    diagnostics = stored_output(NOTEBOOK, "assessment")
    assert re.search(r"validation\s+score_equations\s+passed", diagnostics)
    assert diagnostics.count("|score| / threshold = 0.0000") == 1
    assert not _MACHINE_NOISE.search(diagnostics)
    attention = re.search(r"needs attention: \(([^)]*)\)", diagnostics)
    assert attention, "the assessment no longer prints its attention list"
    for name in re.findall(r"'(\w+)'", attention.group(1)):
        assert f"| `{name}` |" in prose, f"the reading does not explain {name}"
    assert "unavailable" not in diagnostics and "not_applicable" not in diagnostics

    # Overlap: no smoking pair reaches the upper bound, and the bound does not carry the estimate.
    overlap = stored_output(NOTEBOOK, "overlap")
    assert _number(overlap, r"largest fitted propensity: (\S+)") < 0.975
    below = re.search(r"lower ATT bound 0.025: non-smoking ([\d,]+), smoking ([\d,]+)", overlap)
    assert below and int(below[1].replace(",", "")) > 50 * int(below[2].replace(",", ""))
    curve = [
        float(match.group(1))
        for match in re.finditer(r"(?m)^\s*0\.0\d+\s+\S+\s+(\S+)\s+\S+\s+\S+\s*$", overlap)
    ]
    assert len(curve) == 5, curve
    standard_error = _number(fit_text, r"(?m)^att\s+\S+\s+(\S+)")
    assert max(curve) - min(curve) < 0.1 * standard_error, curve

    # Refutation: the three tests ran, no verdict printed, and no refusal appears anywhere.
    refutation = stored_output(NOTEBOOK, "refute")
    for test in ("placebo", "subset", "random_common_cause"):
        assert re.search(rf"(?m)^{test}\s+\S+\s+\S+\s+\S+.*\bL0\b", refutation), test
    assert "VERDICT: none." in refutation
    # The placebo mean lies within two of its own standard deviations over 20 draws of zero.
    placebo = re.search(r"(?m)^placebo\s+\S+\s+(\S+)\s+(\S+)", refutation)
    assert placebo and abs(float(placebo[1])) < 2 * float(placebo[2]) / 20**0.5
    # The scope sentence names the nearest point row, not a longitudinal one.
    assert "Covered: methods ['TMLE']" in refutation and "LTMLE" not in refutation
    shift = _number(refutation, r"paired shift in standard errors = (\S+)")
    assert abs(shift) < 0.1, "the reading says the noise column moves under a tenth of an SE"
    generated = stored_output(NOTEBOOK, "refute-generated")
    assert "10 draws are below the calibrated minimum of 400" in generated
    assert re.search(r"(?m)^simulated_outcome\s.*\bL1\b", generated)
    assert (
        "BoundedAdjustmentOutcome(intercept=-1.0, effect=-0.1, adjustment_scale=0.2)"
        in (code["refute-generated"])
    )
    for cell_id in code:
        text = stored_output(NOTEBOOK, cell_id)
        assert "CapabilityError" not in text and "refus" not in text.lower(), cell_id

    # Sensitivity: the benchmark moves the estimate away from zero, the bound at the benchmark
    # excludes zero, and the largest grid value reaches zero.
    sensitivity = stored_output(NOTEBOOK, "sensitivity")
    assert _number(sensitivity, r"the estimate moved by (\S+) when") < 0
    implied = re.search(r"implied cf_y = (\S+),", sensitivity)
    assert implied and float(implied[1]) < 0.01
    assert "cf_d=benchmark.cf_d" in code["sensitivity"]
    assert 'estimand="att", benchmark=benchmark' in code["sensitivity"]
    at_benchmark = _row(sensitivity, implied[1])
    assert at_benchmark[-1] < 0, "the benchmark bound no longer excludes zero"
    assert _row(sensitivity, "0.1000")[-1] > 0, "the largest grid bound no longer reaches zero"
    for strength in ("0.0100", "0.0300"):
        assert _row(sensitivity, strength)[-1] < 0, strength
    assert "robustness value: " in sensitivity

    # The E-value uses the standardized-difference approximation for a continuous outcome.
    evalue = stored_output(NOTEBOOK, "evalue")
    assert "(approximate conversion)" in evalue and "Standardised by sd(Y)" in evalue
    assert _number(evalue, r"E-value, confidence limit\s+:\s+(\S+)") > 1

    # Simulated confounding: a qualitative surface over nine cells, calibrated on the two
    # benchmark covariates.
    surface = stored_output(NOTEBOOK, "simulated-confounding")
    assert "This surface is qualitative. It is not a bound" in surface
    cells_printed = re.findall(r"(?m)^(0(?:\.0[25])?)\s+(0(?:\.[12])?)\s+\S+\s+[+-]", surface)
    assert len(cells_printed) == 9, cells_printed
    assert 'benchmark_covariates=("mother_education", "married")' in code["simulated-confounding"]
    for covariate in ("mother_education", "married"):
        assert re.search(rf"(?m)^\s+{covariate}\s+treatment\s", surface), covariate
        assert re.search(rf"(?m)^\s+{covariate}\s+outcome\s", surface), covariate
    # "the estimate moves by -7.8858 g": away from zero at the benchmark-sized cell.
    near_benchmark = re.search(r"(?m)^0\.02\s+0\.1\s+\S+\s+(\S+)", surface)
    assert near_benchmark and float(near_benchmark[1]) < 0

    # Over-adjustment: each refit moves the ATT by under one standard error.
    refit = stored_output(NOTEBOOK, "over-adjustment")
    moves = [float(value) for value in re.findall(r", ([+-]\d+\.\d+) standard errors", refit)]
    assert len(moves) == 2 and all(abs(move) < 1 for move in moves), moves
    assert '"previous_infant_over_4000g"' in code["over-adjustment"]

    # The review probe of 400 generated-outcome draws is committed with its reading.
    probe = PROBE_LOG.read_text(encoding="utf-8")
    assert "400 draws" in probe or "successful=400" in probe
