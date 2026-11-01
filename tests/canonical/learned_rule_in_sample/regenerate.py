"""Regenerate the in-sample learned-rule evidence (X11 PR F2) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_in_sample, learned_rule_in_sample_properties

if __name__ == "__main__":
    run(
        learned_rule_in_sample,
        learned_rule_in_sample_properties,
        here=Path(__file__).resolve().parent,
    )
