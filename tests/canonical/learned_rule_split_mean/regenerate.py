"""Regenerate the learned-rule split-mean evidence (X11 PR B) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_split_mean, learned_rule_split_mean_properties

if __name__ == "__main__":
    run(
        learned_rule_split_mean,
        learned_rule_split_mean_properties,
        here=Path(__file__).resolve().parent,
    )
