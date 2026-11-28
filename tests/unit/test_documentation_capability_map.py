"""Every family of the implementation matrix has its own row on the capability map.

``docs/technical-reference/index.md`` holds the implementation matrix: one row per shipped
method family, with its theory and its evidence. ``docs/user-guide/capabilities.md`` is the
reader's map of the same ground, one row per question. A family added to the matrix and left off
the map is a method the reader cannot find, so :data:`MAP_ROWS` names, for each matrix family,
the question that opens its map row, and the test holds the two in agreement both ways.

The map may hold more rows than the matrix (variable importance, the EP learner and the
eliminated-event questions are not matrix families); it may not hold fewer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "technical-reference" / "index.md"
MAP = ROOT / "docs" / "user-guide" / "capabilities.md"

#: Matrix family (its first cell, verbatim) to the first cell of its map row.
MAP_ROWS: dict[str, str] = {
    "Counterfactual means; ATE; risk and odds ratios; multi-arm contrasts": "What would the average outcome be if everyone got each option (each arm), and how do the options compare?",
    "ATT and ATC": "What was the effect among the people who got the treatment, or among those who did not?",
    "Known treatment mechanism": "The chance of treatment was set by design, as in a randomized trial. Can the analysis use it?",
    "Natural-course mean, PAR, and PAF": "How much of the outcome would go away if no one were exposed?",
    "Missing outcomes and controlled direct effects": "Some outcomes are missing, or you want the effect with a later variable held at one value (a controlled direct effect)",
    "Deterministic and dynamic regimes": 'What if everyone followed a rule, such as "treat when the score is above 5"?',
    "Known stochastic regimes": "What if treatment were assigned at random with chances I choose?",
    "Value of a learned treatment rule": "How well would a treatment rule learned from these data work, and how does it compare with a fixed plan?",
    "Continuous modified treatment policies": "What if everyone's dose were raised by a fixed amount, or capped?",
    "Modified treatment policies on a vector treatment": "The treatment has several continuous parts, such as a price and a discount",
    "Continuous policies at declared boundaries, with learned and tuned inputs": "The changed dose must stay inside limits you declare, or uses a cap learned from the data",
    "Incremental propensity-score interventions": "What if everyone's odds of treatment were doubled?",
    "Incremental interventions over time": "What if the odds of treatment were changed at every time?",
    "Categorical incremental interventions": "The same, for a treatment with three or more levels",
    "Point and longitudinal MSM projections": "How does the average outcome trend across options, or across plans over time, summarized by a line or curve?",
    "Conditional effects on a declared basis": "How does the effect change with a baseline column?",
    "Targeting curves of a priority score": "Does a priority score put the people who benefit most first?",
    "External scores": "Does a score built elsewhere predict who benefits, and is one score better than another?",
    "Longitudinal end-of-study TMLE": "What would the end-of-study outcome be under each treatment plan?",
    "Sequentially doubly robust longitudinal estimation": "The same, with an estimator that stays on target when, at each time, either the outcome model or the treatment model is right",
    "Sequential g-computation and IPW over time": "What do simpler estimators say, beside the targeted one?",
    "Structural nested mean model": "What is the effect of each treatment time, and does it vary with a covariate?",
    "Parametric g-formula over time": "Can I simulate forward from models of every covariate, or fix a continuous dose at each time?",
    "Longitudinal survival and competing risks": "The same, when treatment changes over follow-up, or when another event can come first",
    "Longitudinal repeated outcomes, `IPSI`": "The same, under a change in the odds of treatment",
    "Longitudinal repeated outcomes, `SDR`": "The same, with the sequentially doubly robust estimator",
    "Longitudinal evidence at three to five decisions": "How far does the evidence for analyses over time reach, in number of treatment times?",
    "Longitudinal repeated outcomes": "The outcome is measured at every time, not only at the end",
    "Point-treatment survival": "What is the risk of the event by each check time under each treatment?",
    "Hazard targeting beyond a held baseline treatment": "Can the risk curve never dip, with one correction step for every check time at once?",
    "Collaborative TMLE": "Can the data choose which covariates the treatment model uses?",
    "Collaborative TMLE of a risk curve": "Can the data choose the treatment and censoring models?",
    "Drift-corrected survival TMLE": "Can the risk interval stay valid when the event model, or the treatment and censoring models, are wrong?",
    "Continuous-time survival and competing risks": "Can I use the exact event times instead of a grid?",
    "Continuous-time longitudinal TMLE": "What if the treatment changes at visits that fall at different times for each person?",
    "DR-TMLE": "The default interval needs both models close to right. Can it hold when one is wrong?",
    "Adaptive-propensity IPTW": "Can I estimate with treatment weights alone (IPTW), with no outcome model?",
    "Adaptive TMLE for a trial with external data": "Can outside rows, such as a registry, sharpen a small trial's estimate, correcting for how they differ?",
    "Smoothed dose-response curve": "How does the average outcome change across a range of doses?",
    "Cross-fitting and CV-TMLE": "How are the models trained out of fold, and when are the folds reused?",
    "Nuisance learners": "How do I use my own scikit-learn models?",
    "Targeting and fitting options": "How does the correction step stop, and which other fitting options exist?",
    "Inference, diagnostics, and sensitivity": "How are intervals, bands, diagnostics and sensitivity analyses computed?",
}


def _first_cells(text: str) -> list[str]:
    """The first cell of every body row of every pipe table in ``text``."""
    cells = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("| ") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.startswith("| ---"):
            continue  # a header row
        cells.append(line[2:].split(" | ", 1)[0].strip())
    return cells


def matrix_families(text: str) -> set[str]:
    """Each family of the implementation matrix, cut at its first colon.

    A family cell may run on after a colon with a list of what it covers; the name before the
    colon is the family.
    """
    start = text.index("## Implementation matrix")
    end = text.index("\n## ", start + 1)
    families = set()
    for cell in _first_cells(text[start:end]):
        families.add(cell.split(":", 1)[0].strip())
    return families


def map_questions(text: str) -> set[str]:
    """The first cell of every row of the capability map's tables."""
    return set(_first_cells(text))


def missing(matrix_text: str, map_text: str) -> list[str]:
    """Every disagreement between the matrix, :data:`MAP_ROWS` and the map page."""
    problems = []
    families = matrix_families(matrix_text)
    for family in sorted(families - MAP_ROWS.keys()):
        problems.append(f"matrix family with no map row in MAP_ROWS: {family!r}")
    for family in sorted(MAP_ROWS.keys() - families):
        problems.append(f"MAP_ROWS names a family the matrix no longer has: {family!r}")
    questions = map_questions(map_text)
    for family, question in sorted(MAP_ROWS.items()):
        if question not in questions:
            problems.append(f"{family!r}: no map row opens with {question!r}")
    if len(set(MAP_ROWS.values())) != len(MAP_ROWS):
        problems.append("two matrix families share one map row")
    return problems


def test_every_matrix_family_has_its_own_map_row():
    problems = missing(MATRIX.read_text(encoding="utf-8"), MAP.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("mutation", ["new_family", "dropped_row"])
def test_the_check_fails_when_the_map_falls_behind(mutation):
    """Deliberate-mutation controls: each doctored text must produce a finding."""
    matrix_text = MATRIX.read_text(encoding="utf-8")
    map_text = MAP.read_text(encoding="utf-8")
    if mutation == "new_family":
        header = "| --- | --- | --- | --- | --- |\n"
        matrix_text = matrix_text.replace(
            header, header + "| A family nobody mapped | t | c | e | v |\n", 1
        )
    else:
        question = MAP_ROWS["ATT and ATC"]
        map_text = map_text.replace(f"| {question} |", "| a reworded question |", 1)
    assert missing(matrix_text, map_text)
