"""Regenerate the part G omitted-variable limit coverage evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container.

Red-cell reading (declared before any run, ``X82-SG-plan.md`` rev 2, section 7.1).

* A red primary row is a defect suspect until it is diagnosed. A defect is fixed and the study
  re-runs under a fresh declaration. A red with no defect leaves the rule descriptive, and the
  docs page reports it.
* A control that does not detect leaves the primary verdict alone. The registry row then says
  that the component is not shown to matter at this size.
* No replicate count, law, learner, strength or seed moves after a verdict is seen. No rerun
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import ovb_limit_coverage, ovb_limit_coverage_properties

if __name__ == "__main__":
    run(
        ovb_limit_coverage,
        ovb_limit_coverage_properties,
        here=Path(__file__).resolve().parent,
    )
