"""Regenerate the learned-rule boundary evidence under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_cvtmle_boundary, learned_rule_cvtmle_boundary_properties

if __name__ == "__main__":
    run(
        learned_rule_cvtmle_boundary,
        learned_rule_cvtmle_boundary_properties,
        here=Path(__file__).resolve().parent,
    )
