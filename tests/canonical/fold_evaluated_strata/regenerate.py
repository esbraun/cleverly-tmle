"""Regenerate the fold-evaluated strata evidence (X11 PR E1, X28) under the X11 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import fold_evaluated_strata, fold_evaluated_strata_properties

if __name__ == "__main__":
    run(
        fold_evaluated_strata,
        fold_evaluated_strata_properties,
        here=Path(__file__).resolve().parent,
    )
