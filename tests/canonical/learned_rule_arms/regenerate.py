"""Regenerate the learned-rule evidence over three arms (X11 PR H) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_arms, learned_rule_arms_properties

if __name__ == "__main__":
    run(
        learned_rule_arms,
        learned_rule_arms_properties,
        here=Path(__file__).resolve().parent,
    )
