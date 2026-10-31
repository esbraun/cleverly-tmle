"""Regenerate the learned-rule rule-learner evidence (X11 PR C) under the RM30 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_learners, learned_rule_learners_properties

if __name__ == "__main__":
    run(
        learned_rule_learners,
        learned_rule_learners_properties,
        here=Path(__file__).resolve().parent,
    )
