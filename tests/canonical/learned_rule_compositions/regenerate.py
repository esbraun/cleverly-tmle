"""Regenerate the learned-rule compositions evidence (X11 PR A) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_compositions, learned_rule_compositions_properties

if __name__ == "__main__":
    run(
        learned_rule_compositions,
        learned_rule_compositions_properties,
        here=Path(__file__).resolve().parent,
    )
