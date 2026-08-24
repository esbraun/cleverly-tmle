"""Regenerate fold-evaluated CV-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import fold_cvtmle_properties, fold_evaluated_cvtmle

if __name__ == "__main__":
    main(
        fold_evaluated_cvtmle,
        fold_cvtmle_properties,
        here=Path(__file__).resolve().parent,
    )
