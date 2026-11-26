"""Regenerate the learned-rule comparator contrast evidence (T58) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import (
    learned_rule_contrast_comparators,
    learned_rule_contrast_comparators_properties,
)

if __name__ == "__main__":
    run(
        learned_rule_contrast_comparators,
        learned_rule_contrast_comparators_properties,
        here=Path(__file__).resolve().parent,
    )
