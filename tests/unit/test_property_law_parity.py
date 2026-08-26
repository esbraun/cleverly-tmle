"""A cross-fitted study's sharp null is its ordinary study's, and has to stay that way.

Three ordinary rows have a cross-fitted sibling, and each pair shares one law.  Two of the
three pairs state that law twice, in byte-identical blocks:
``ltmle_crossfit_properties`` copies ``ltmle_properties``, and
``ltmle_survival_crossfit_properties`` copies ``ltmle_survival_properties``, comments included.
The third, ``ltmle_competing_crossfit_properties``, imports it instead.

The copies are not an oversight that can simply be deleted.  Collapsing one makes the ordinary
properties module *result-determining* for the cross-fitted study, and neither ordinary module
appears in its sibling's :attr:`~tests.studies.evidence.registry.StudyRecord.modules`.
``test_method_evidence.py`` requires that tuple to equal the manifest's ``study_module_sha256``
key set, and only a regeneration rewrites a manifest.  So the copies stay until those two rows
are next regenerated for a reason of their own.

What this module does is make the duplication safe to leave.  A copy that drifts from its
source publishes a type-I cell measured against a law the page describes as the other one's,
and nothing else would notice: each study summarises its own rows, so both halves stay
internally consistent while disagreeing about what the null is.  Requiring the two to match
exactly means a future regeneration can collapse them by deleting a block, with this test as
the evidence that deleting it changes no number.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.studies import (
    ltmle_competing_crossfit_properties,
    ltmle_competing_properties,
    ltmle_crossfit_properties,
    ltmle_properties,
    ltmle_survival_crossfit_properties,
    ltmle_survival_properties,
)

#: The ordinary module and its cross-fitted sibling, for the pairs that state the law twice.
COPIED_PAIRS = (
    ("end-of-study", ltmle_properties, ltmle_crossfit_properties),
    ("survival", ltmle_survival_properties, ltmle_survival_crossfit_properties),
)

#: The pair that shares the law by import, which is what the other two should become.
SHARED_PAIR = (ltmle_competing_properties, ltmle_competing_crossfit_properties)


def law_constants(module: ModuleType) -> dict[str, Any]:
    """Every derived-law constant a properties module declares, by name.

    Discovered by prefix rather than listed, so a constant added to one copy and not the
    other fails as a differing key set.  A hand-written list would pass over exactly the
    addition it exists to catch.
    """
    return {
        name: value for name, value in vars(module).items() if name.startswith(("NULL_", "POWER_"))
    }


def same(left: Any, right: Any) -> bool:
    """Exact equality across the shapes these constants take: array, dict, or scalar."""
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return set(left) == set(right) and all(same(left[key], right[key]) for key in left)
    return bool(np.array_equal(left, right))


@pytest.mark.parametrize(
    ("family", "ordinary", "crossfit"), COPIED_PAIRS, ids=[pair[0] for pair in COPIED_PAIRS]
)
def test_the_copied_law_is_the_ordinary_one(
    family: str, ordinary: ModuleType, crossfit: ModuleType
) -> None:
    declared, copied = law_constants(ordinary), law_constants(crossfit)
    assert declared, f"{family}: the ordinary module declares no derived law to copy"
    assert set(declared) == set(copied), (
        f"{family}: the cross-fitted copy declares "
        f"{sorted(set(copied) ^ set(declared))} that its ordinary source does not, or the "
        f"reverse. The two blocks are meant to be identical text"
    )
    differing = sorted(name for name in declared if not same(declared[name], copied[name]))
    assert differing == [], (
        f"{family}: {differing} differ between the ordinary and cross-fitted property "
        f"modules. Both rows publish a type-I cell measured against 'the' null, so a null "
        f"that is two different laws makes one of those two pages wrong"
    )


def test_the_shared_law_is_not_restated() -> None:
    """The competing pair is the shape the other two should reach: one law, one statement."""
    ordinary, crossfit = SHARED_PAIR
    assert law_constants(ordinary), "the ordinary competing module declares no derived law"
    assert law_constants(crossfit) == {}, (
        "the cross-fitted competing study restates a derived law its ordinary sibling "
        "already declares. It reaches that law by import, which is what keeps the two rows "
        "measuring one null; a restated constant is a second null waiting to drift"
    )
