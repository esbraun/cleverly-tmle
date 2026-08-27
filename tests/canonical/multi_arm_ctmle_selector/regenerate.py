"""Regenerate selector-based multi-arm C-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_multi_arm_ctmle_selector, multi_arm_ctmle_selector_properties

if __name__ == "__main__":
    main(
        canonical_multi_arm_ctmle_selector,
        multi_arm_ctmle_selector_properties,
        here=Path(__file__).resolve().parent,
    )
