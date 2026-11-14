"""Regenerate the optimal-rule compositions evidence (X11 PR 2) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import (
    learned_rule_optimal_compositions,
    learned_rule_optimal_compositions_properties,
)

if __name__ == "__main__":
    run(
        learned_rule_optimal_compositions,
        learned_rule_optimal_compositions_properties,
        here=Path(__file__).resolve().parent,
    )
