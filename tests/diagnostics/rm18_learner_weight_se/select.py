"""The declared selection and comparison sets, read from committed standard errors alone.

RM18 of ``docs/roadmap.md`` declares both sets in "The learner-weight standard-error
diagnostic, declared before it runs".  The selection is every replicate whose committed
``std_error`` exceeds 1 in either ``learner_weight_necessity`` arm of
``weighted-ltmle-crossfit``.  The comparison is the 13 lowest replicate indices outside it.
This module fits nothing.  It prints both sets and any difference from the declared ones.

    python -m tests.diagnostics.rm18_learner_weight_se.select
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from tests.studies.evidence.registry import ROOT

PROPERTY_REPLICATES = (
    ROOT / "tests" / "canonical" / "weighted_lmtp_ltmle" / "property-replicates.csv.gz"
)
PROPERTY = "learner_weight_necessity"
ARMS = (
    "static__weighted_learners",
    "static__discarded_learner_weight_control",
)
#: A committed standard error above this selects its replicate.
EXPLOSION = 1.0

#: The set the declaration expects the rule to give.
DECLARED_SELECTED = (17, 20, 258, 278, 343, 487, 533, 711, 889, 978, 981, 998, 1099)
#: The declaration fixes the comparison size at 13, and names indices 0 to 12.
COMPARISON_SIZE = 13
DECLARED_COMPARISON = tuple(range(COMPARISON_SIZE))


def learner_weight_rows(path: Path = PROPERTY_REPLICATES) -> pd.DataFrame:
    """The committed rows of both ``learner_weight_necessity`` arms."""
    rows = pd.read_csv(path)
    chosen = rows.loc[rows["property"] == PROPERTY].reset_index(drop=True)
    if set(chosen["cell"]) != set(ARMS):
        raise ValueError(f"expected the arms {ARMS}, found {sorted(set(chosen['cell']))}")
    return chosen


def selected(rows: pd.DataFrame) -> tuple[int, ...]:
    """Every replicate whose committed ``std_error`` exceeds 1 in either arm."""
    exploded = rows.loc[rows["std_error"] > EXPLOSION, "replicate"]
    return tuple(sorted(int(replicate) for replicate in set(exploded)))


def comparison(rows: pd.DataFrame, chosen: Iterable[int]) -> tuple[int, ...]:
    """The :data:`COMPARISON_SIZE` lowest replicate indices outside ``chosen``."""
    excluded = set(chosen)
    outside = sorted(int(replicate) for replicate in set(rows["replicate"]) - excluded)
    if len(outside) < COMPARISON_SIZE:
        raise ValueError(f"only {len(outside)} replicates lie outside the selection")
    return tuple(outside[:COMPARISON_SIZE])


def main() -> None:
    rows = learner_weight_rows()
    chosen = selected(rows)
    compared = comparison(rows, chosen)
    print(f"selected ({len(chosen)}): {list(chosen)}")
    print(f"comparison ({len(compared)}): {list(compared)}")
    for name, got, declared in (
        ("selected", chosen, DECLARED_SELECTED),
        ("comparison", compared, DECLARED_COMPARISON),
    ):
        if got == declared:
            print(f"{name}: equals the declared set")
        else:
            print(
                f"{name}: differs from the declared set; "
                f"added {sorted(set(got) - set(declared))}, "
                f"missing {sorted(set(declared) - set(got))}"
            )


if __name__ == "__main__":
    main()
