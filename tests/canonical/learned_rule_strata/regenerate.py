"""Regenerate the learned-rule strata evidence (X11 PR E2, part (m)) under the X11 run form."""

from pathlib import Path

from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_strata, learned_rule_strata_properties

if __name__ == "__main__":
    run(
        learned_rule_strata,
        learned_rule_strata_properties,
        here=Path(__file__).resolve().parent,
    )
