"""Regenerate the learned two-node regimen evidence (X11 PR K) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_regimen_ltmle, learned_regimen_ltmle_properties

if __name__ == "__main__":
    run(
        learned_regimen_ltmle,
        learned_regimen_ltmle_properties,
        here=Path(__file__).resolve().parent,
    )
