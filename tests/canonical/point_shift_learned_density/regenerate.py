"""Regenerate the ``point-shift-learned-density`` evidence (X121 part b2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator, so the run starts no container. The law, the arms, the counts and the reading
declared before the run are those of :mod:`tests.studies.point_shift_learned_density` and
:mod:`tests.studies.point_shift_learned_density_properties`. Run the sizing pilot
(:mod:`tests.studies.point_shift_learned_density_pilot`) first.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import point_shift_learned_density, point_shift_learned_density_properties

if __name__ == "__main__":
    run(
        point_shift_learned_density,
        point_shift_learned_density_properties,
        here=Path(__file__).resolve().parent,
    )
