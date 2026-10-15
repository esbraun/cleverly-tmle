"""The static checks for ``docs/examples/twins-causal-inference``.

The TWINS notebook downloads its data, so the offline fast tier cannot execute it.  This module
therefore defines no ``check(namespace)``.  It reads committed text only.
``tests/unit/test_documentation_runtime.py`` compares the narrated decimals against the stored
outputs with :data:`UNPRINTED_DECIMALS`, and calls :func:`check_stored` for the outputs the
readings rely on.  Pin a relation between the prose and the stored outputs here.  Do not pin a
number that the next networked execution can move.
"""

from __future__ import annotations

import csv
import re

import nbformat

from tests.notebooks import code_cells
from tests.unit.tutorial_semantics import EXAMPLES, markdown_text, notebook_code, stored_output

NOTEBOOK = EXAMPLES / "twins-causal-inference.ipynb"
_STUDY = "tests/canonical/tmle3_cvtmle"

#: Decimals the prose writes that no stored output prints, each with the reason.
UNPRINTED_DECIMALS = {
    "0.5": "a value of the outcome, the share of two twins who die, not a result",
    "0.9425": f"coverage of the cleverly continuous att row: {_STUDY}/summary.csv",
    "0.9956": f"se_ratio of the same row: {_STUDY}/summary.csv",
}

#: A printed residual such as 3.79e-13 is machine noise that changes between machines.
_MACHINE_NOISE = re.compile(r"\d\.\d+e-(0[89]|[1-9]\d)")


def _number(text: str, pattern: str) -> float:
    """The first number captured by ``pattern`` in ``text``."""
    match = re.search(pattern, text)
    assert match, f"the TWINS notebook no longer prints {pattern!r}"
    return float(match.group(1))


def _row(text: str, label: str) -> list[float]:
    """The numbers of the printed table row that starts with ``label``."""
    match = re.search(rf"(?m)^\s*{re.escape(label)}\s+(.+)$", text)
    assert match, f"the TWINS notebook no longer prints the row {label!r}"
    return [float(value) for value in match.group(1).split()]


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
    assert len(figures) == 2, "the TWINS notebook has a birth-weight figure and an overlap figure"

    # Every code cell is followed by its reading.
    cells = notebook.cells
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            following = cells[index + 1] if index + 1 < len(cells) else None
            assert following is not None and following.cell_type == "markdown", cell.id
            assert str(following.source).startswith("**What this output tells you.**"), cell.id

    # The situation states a causal graph with one row per arrow or absent arrow.
    assert "```text" in prose and "| arrow or absent arrow | what it asserts |" in prose
    assert "no U → A" in prose and "A → M ← U → Y" in prose

    # Setup: no global warning filter hides the library's warnings.
    assert "filterwarnings" not in code["setup"]

    # Data: the pinned source, the state-year rule, and the counts the reading quotes.
    load = code["load-data"]
    assert "AMLab-Amsterdam/CEVAE/{COMMIT}" in load
    assert 'COMMIT = "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd"' in load
    assert 'groupby(state_year).transform("mean")' in load and "missing_share <= 0.5" in load
    load_text = stored_output(NOTEBOOK, "load-data")
    for count in ("71,345", "16,739", "2,732", "51,874"):
        assert count in load_text and count in prose, count
    # The excluded residence-state codes are the states that the NCHS guides name.
    excluded = {int(code) for code in re.findall(r"(?m)^\s+(\d+)\s+19\d\d", load_text)}
    assert {5, 15, 19, 28, 33, 37, 42} <= excluded, excluded
    # No mediator enters the adjustment set.
    for mediator in ("dbirwt", "gestat10", "phyper", "eclamp", "nprevistq", "adequacy"):
        assert mediator not in load, mediator

    # Association: the two conditions of a confounder hold for marital status and education.
    association = stored_output(NOTEBOOK, "association")
    for label in ("unmarried", "mother_education 1-2"):
        smoking, non_smoking, with_it, without_it = _row(association, label)
        assert smoking > non_smoking and with_it > without_it, label
    crude = _number(association, r"crude difference: (\S+),")

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

    # Design: the pair is the unit, so no cluster; the ATT; the declared outcome support.
    assert "cluster=" not in code["identify"] and "ATT(reference=0)" in code["identify"]
    assert "Targeting(q_bounds=(0.0, 1.0))" in code["att-fit"]
    assert "CrossFitting(n_folds=3, learner_folds=3)" in code["att-fit"]
    assert "stacked CV-TMLE (Levy): nuisances cross-fitted over 3 folds" in fit_text
    assert "clusters =" not in fit_text
    # The bounded-outcome reading: the share is continuous, and the initial fit stays in [0, 1].
    assert "outcome family: gaussian; fluctuation: logistic" in fit_text
    initial = re.search(r"initial outcome predictions: (\S+) to (\S+)", fit_text)
    assert initial and 0.0 <= float(initial[1]) <= float(initial[2]) <= 1.0

    # The ATT and the crude difference share the observed risk of the smoking pairs.
    observed = _number(fit_text, r"observed risk, smoking pairs:\s+(\S+)")
    counterfactual = _number(fit_text, r"estimated risk had they not smoked:\s+(\S+)")
    reference = _number(fit_text, r"risk, non-smoking pairs:\s+(\S+)")
    att = _number(fit_text, r"ATT: (\S+),")
    assert abs(observed - counterfactual - att) <= 1.5e-4
    # "Adjustment for C moves that risk from 0.0348 to 0.0412."
    assert counterfactual > reference and att < crude

    # Birth weight: smoking shifts weight down, and the within-band contrast changes sign.
    paradox = stored_output(NOTEBOOK, "birth-weight-paradox")
    shares = re.search(r"non-smoking (\S+), smoking (\S+)", paradox)
    assert shares and float(shares.group(2)) > float(shares.group(1))
    lightest = _row(paradox, "under 1500 g")
    heaviest = _row(paradox, "3000 g or more")
    assert lightest[1] < lightest[0] and heaviest[1] > heaviest[0]
    assert "kwj275" in (EXAMPLES.parent / "references.md").read_text(encoding="utf-8")

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
    assert max(curve) - min(curve) <= 1e-4, curve

    # Sensitivity: the benchmark moves the estimate up, its outcome-side strength is below the
    # smallest grid value, and the grid bounds read as the prose says.
    sensitivity = stored_output(NOTEBOOK, "sensitivity")
    assert _number(sensitivity, r"the estimate moved by (\S+) when") > 0
    implied = re.search(r"implied cf_y = (\S+),", sensitivity)
    assert implied and float(implied[1]) < 0.001
    assert "cf_d=benchmark.cf_d" in code["sensitivity"]
    assert 'estimand="att", benchmark=benchmark' in code["sensitivity"]
    at_benchmark = _row(sensitivity, implied[1])
    assert at_benchmark[0] > att, "the bias bound at the benchmark no longer exceeds the estimate"
    assert at_benchmark[-1] > crude, "the benchmark bound no longer admits the crude difference"

    # Over-adjustment: dropping the three doubtful items moves the ATT by under half an SE.
    refit = stored_output(NOTEBOOK, "over-adjustment")
    shift = _number(refit, r"without the three items: (\S+)") - _number(
        refit, r"with the three items:\s+(\S+)"
    )
    assert 0 < shift < 0.5 * _number(refit, r"standard error of the ATT:\s+(\S+)")

    # The study numbers that the trust table quotes.
    with (EXAMPLES.parents[1] / _STUDY / "summary.csv").open(encoding="utf-8") as handle:
        (study_row,) = [
            row
            for row in csv.DictReader(handle)
            if (row["implementation"], row["scenario"], row["estimand"])
            == ("cleverly-stacked-cvtmle", "continuous", "att")
        ]
    assert f"{float(study_row['coverage']):.4f}" == "0.9425"
    assert f"{float(study_row['se_ratio']):.4f}" == "0.9956"
