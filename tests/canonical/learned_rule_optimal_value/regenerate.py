"""Regenerate the value-of-the-optimal-rule evidence (X11 PR I) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_optimal_value, learned_rule_optimal_value_properties

if __name__ == "__main__":
    run(
        learned_rule_optimal_value,
        learned_rule_optimal_value_properties,
        here=Path(__file__).resolve().parent,
    )
