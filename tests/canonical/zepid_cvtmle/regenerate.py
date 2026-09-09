"""Regenerate fold-targeted CV-TMLE evidence against Python zEpid."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import fold_targeted_cvtmle, fold_targeted_cvtmle_properties

if __name__ == "__main__":
    main(
        fold_targeted_cvtmle,
        fold_targeted_cvtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            # Derived, so the tag cannot drift away from the commit the study declares.
            image=f"cleverly-zepid-reference:{fold_targeted_cvtmle.ZEPID_COMMIT[:7]}",
            runner="run_zepid_cvtmle.py",
            mount_runner=True,
            extra_files=("requirements.txt",),
            interpreter="python",
        ),
    )
