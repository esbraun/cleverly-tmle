"""Regenerate the fold-local learned-rule value CV-TMLE evidence under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_cvtmle, learned_rule_cvtmle_properties

if __name__ == "__main__":
    run(
        learned_rule_cvtmle,
        learned_rule_cvtmle_properties,
        here=Path(__file__).resolve().parent,
    )
