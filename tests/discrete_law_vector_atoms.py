"""Declared atoms at the doses of a vector treatment: the oracle of the mixed factors (X40).

The law is the geometric law of ``tests/discrete_law_vector_density.py``: ``A = (F, D1,
D2)``, a binary flag and two doses on ``{1, 2, 4, 8}``.  Here dose 1 declares the atoms
``1`` and ``2`` and dose 2 the atom ``8``.  The factor of each dose is then taken against
Lebesgue measure plus counting measure on its atoms: :math:`p_{k,j}` at an atom and
:math:`(1 - \\sum_j p_{k,j}) f_k` elsewhere (``cleverly.interventions.joint_density``).

**The exactness conditions.**  Every grid dose is a point mass of the law, so the binned
density of a dose equals the law on its non-atom values under the conditions of the
geometric law: a moved continuous dose has a source bin whose width is the inverse
derivative times the width of its target bin.  The mixed formula drops the Lebesgue mass a
smooth piece sends onto an atom, so no policy here sends a non-atom grid dose onto a
declared atom by a smooth piece.  The policies send atoms to atoms only, as R-b3 requires.

The nonzero requirements (``test_t_b1_the_component_witnesses_are_nonzero``):

* the atom ratio exceeds 2 on some cell: "atom up" sends the atom 1 of dose 1 to the atom
  2, so the ratio at 2 is :math:`(p_1 + p_2) / p_2`, which is 3 on the cell ``W = 1, F = 1``;
* a flag-dependent branch moves mass onto an atom: "flag up, atom up" sets the flag and
  sends the atom 1 to 2 on the natural flag 0, so the mass at ``(F, D1) = (1, 2)`` comes
  from both cells;
* "flag reads atom" sets the flag where the natural dose 1 is the atom 1, and dose 2's map
  reads the natural dose 1, so the atoms of dose 1 reach dose 2's frame and the flag's map.

"collapse to atom" sends the doses 2 and 4 of dose 1 to the atom 1 by
``Piece(1.5, 6, to_atom=1)``.  The density route refuses it at a vector, because dose 2's
factor conditions on dose 1, and the classifier route fits it.  "collapse last dose" sends
the doses 2 and 4 of dose 2, the last dose, to its atom 8 by ``Piece(1.5, 6, to_atom=8)``.
The interval ``[1.5, 6)`` is the union of the bins of 2 and 4, so the integral of the binned
density over it is the law's mass there, and both routes are exact.

``ALL_ATOMS`` declares every grid dose of dose 1 an atom.  Then dose 1's factor is the
discrete formula, and every plan of the geometric law is exact again: the vector analog of
a grid declared as atoms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import discrete_law_vector_density as geo
from .discrete_law_policy_point import Branches, _mean_under

#: The declared atoms of each dose, by component.
ATOMS: dict[str, tuple[float, ...]] = {"D1": (1.0, 2.0), "D2": (8.0,)}
#: Every grid dose of dose 1 declared an atom.
ALL_ATOMS: dict[str, tuple[float, ...]] = {"D1": geo.DOSES}

CELLS = geo.CELLS
PROBS = geo.PROBS
frame = geo.frame
sample = geo.sample
first_row_of = geo.first_row_of


def _index(f: int, i1: int, i2: int) -> int:
    return 16 * f + 4 * i1 + i2


def _d1_held(i1: int) -> int:
    """Dose 1 with its atoms 1 and 2 held: 4 doubles to 8, and 8 is held."""
    return 3 if i1 == 2 else i1


def _d1_up(i1: int) -> int:
    """Dose 1 with the atom 1 sent to the atom 2: 2 is held, 4 doubles, 8 is held."""
    return 1 if i1 == 0 else _d1_held(i1)


def _d2_held(i2: int) -> int:
    """Dose 2 doubled below 3, with 4 and the atom 8 held."""
    return i2 + 1 if i2 <= 1 else i2


def _atoms_held(f: int, i1: int, i2: int) -> int:
    return _index(f, _d1_held(i1), _d2_held(i2))


def _atom_up(f: int, i1: int, i2: int) -> int:
    return _index(f, _d1_up(i1), i2)


def _flag_up_atom_up(f: int, i1: int, i2: int) -> int:
    """Set the flag, and send the atom 1 of dose 1 to 2 where the natural flag is 0."""
    return _index(1, _d1_up(i1) if f == 0 else _d1_held(i1), i2)


def _flag_reads_atom(f: int, i1: int, i2: int) -> int:
    """The flag goes to one at the atom 1 of dose 1; dose 2 doubles below 3 at dose 1 <= 2."""
    flag = 1 if i1 == 0 else f
    return _index(flag, _d1_held(i1), _d2_held(i2) if i1 <= 1 else i2)


def _collapse(f: int, i1: int, i2: int) -> int:
    """Dose 1 sends the interval ``[1.5, 6)`` (the doses 2 and 4) to the atom 1."""
    return _index(f, 0 if i1 in (1, 2) else i1, i2)


def _collapse_last(f: int, i1: int, i2: int) -> int:
    """Dose 2 sends the interval ``[1.5, 6)`` (the doses 2 and 4) to its atom 8."""
    return _index(f, i1, 3 if i2 in (1, 2) else i2)


def _map(assign: Callable[[int, int, int], int]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(assign(*CELLS[k]) for k in range(len(CELLS))) for _ in range(2))


#: Every plan, as ``(probability, map)`` branches with ``map[w][k]`` the target index.
POLICIES: dict[str, Branches] = {
    "natural": ((1.0, _map(lambda f, i1, i2: _index(f, i1, i2))),),
    "atoms held": ((1.0, _map(_atoms_held)),),
    "atom up": ((1.0, _map(_atom_up)),),
    "flag up, atom up": ((1.0, _map(_flag_up_atom_up)),),
    "flag reads atom": ((1.0, _map(_flag_reads_atom)),),
    "random": ((0.3, _map(_flag_up_atom_up)), (0.7, _map(_atoms_held))),
}
#: A collapse of an interval onto an atom: the classifier route alone fits it at a vector.
COLLAPSE: dict[str, Branches] = {
    "collapse to atom": ((1.0, _map(_collapse)),),
    "collapse last dose": ((1.0, _map(_collapse_last)),),
}
LABELS = tuple(POLICIES)


def functional(probs: Any, label: str) -> Any:
    """The policy mean of ``label`` from the ``(2 * 32 * 2,)`` cell probabilities, longhand."""
    cells = np.asarray(probs).reshape(2, len(CELLS), 2)
    joint = cells[:, :, 0] + cells[:, :, 1]
    q = cells[:, :, 1] / joint
    return _mean_under(joint, q, {**POLICIES, **COLLAPSE}[label])


def truth(label: str) -> float:
    """The parameter of ``label`` at the law."""
    return float(np.real(functional(PROBS, label)))


def eif(label: str) -> np.ndarray:
    """The complex-step efficient influence function at every support point."""
    base = np.asarray(PROBS, dtype=complex)
    step = 1e-30
    out = []
    for point in range(base.size):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        out.append(float(np.imag(functional(perturbed, label)) / step))
    return np.array(out)


def atom_up_ratio() -> np.ndarray:
    """``(p_1 + p_2) / p_2`` of dose 1 on each ``(w, f)`` cell: the "atom up" ratio at 2."""
    out = np.zeros((2, 2))
    for w in range(2):
        for f in range(2):
            out[w, f] = (geo.g1(w, f, 0) + geo.g1(w, f, 1)) / geo.g1(w, f, 1)
    return out
