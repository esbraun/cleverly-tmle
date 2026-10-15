"""The ``pygformula`` comparator runner of ``parametric-gformula``: a declared stub.

The comparison is pending: a follow-up PR from X45 registers it.  This file declares the runner whose bytes the
manifest records.  The comparator change replaces the body with the fixtures that
``README.md`` lists and removes ``pending_reference`` from the study record.  The driver runs
no reference process while the record is pending, so this stub never executes in a study run.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Refuse to run: the fixtures are not written."""
    sys.stderr.write(
        "run_pygformula.py is a declared stub: the pygformula comparison is pending (X45)\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
