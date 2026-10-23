"""Regenerate the X82 SH marginal sensitivity bound limit evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container.

Red-cell reading (declared before any run, ``X82-H-plan.md`` rev 2.5, section 8).

* A red primary row is a defect suspect until it is diagnosed. A defect is fixed and the study
  re-runs under a fresh declaration. A red with no defect leaves the rule descriptive, and the
  docs page reports it.
* A positive cell whose SE ratio lies above the band, with coverage inside its band, is a
  conservative interval. It leaves the rule descriptive, and the docs page reports it.
* A control that does not detect leaves the primary verdict alone.
* No replicate count, law, learner or seed moves after a verdict is seen. No rerun
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import marginal_bound_coverage, marginal_bound_coverage_properties

if __name__ == "__main__":
    run(
        marginal_bound_coverage,
        marginal_bound_coverage_properties,
        here=Path(__file__).resolve().parent,
    )
