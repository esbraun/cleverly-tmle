"""Regenerate the learned-rule contrast evidence (X11 PR G) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_contrast, learned_rule_contrast_properties

if __name__ == "__main__":
    run(
        learned_rule_contrast,
        learned_rule_contrast_properties,
        here=Path(__file__).resolve().parent,
    )
