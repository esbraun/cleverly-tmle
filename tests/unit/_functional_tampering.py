"""The one tampering matrix for a :class:`~cleverly.study.BackdoorMeanContrast` record.

Every surface that refuses a forged functional parametrises over
:data:`FUNCTIONAL_TAMPERINGS`, so a new design-bound field on the record reaches each of
those surfaces at once. ``test_every_field_of_the_functional_is_classified_for_provenance``
checks that :data:`DESIGN_BOUND_FIELDS` and :data:`DESIGN_FREE_FIELDS` together name every
field of the class, so a new field that is in neither fails it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The fields a design declaration fixes beyond the estimand and the adjustment set. The
#: provenance matcher reconstructs each one from the design, so a forged value pairs a fit
#: with a record no design could have written.
DESIGN_BOUND_FIELDS: tuple[str, ...] = (
    "missingness",
    "intermediate_name",
    "treatment_levels",
    "treatment_value",
)

#: The remaining fields of the record. The matcher compares them too, and other tests forge
#: them where a surface reads one; they are not rows of this matrix.
DESIGN_FREE_FIELDS: tuple[str, ...] = (
    "outcome",
    "treatment",
    "adjustment",
    "target",
    "axis",
    "reference",
    "interventions",
    "horizons",
    "msm",
    "intermediate",
    "longitudinal",
)

#: One forged value per design-bound field. Each value has to differ from what a
#: reconstruction of the record produces.
_FORGED_FUNCTIONAL_VALUES: Mapping[str, Any] = {
    "missingness": "forged_delta",
    "intermediate_name": "forged_z",
    "treatment_levels": ("forged",),
    "treatment_value": 1,
}


def _tamperings(
    schema_fields: Sequence[str], forged: Mapping[str, Any]
) -> tuple[tuple[str, Any], ...]:
    """Pair every field in ``schema_fields`` with the value that must break provenance.

    Parameters
    ----------
    schema_fields : sequence of str
        The fields the provenance matcher reconstructs and compares.
    forged : mapping of str to Any
        One tampered value per field.

    Returns
    -------
    tuple of tuple of (str, Any)
        The parametrisation matrix, in the order of ``schema_fields``.

    Raises
    ------
    ValueError
        When the two do not describe the same set of fields. A matrix that is missing a
        row under-covers in silence, so a field added to the record would ship with no
        tampering case.
    """
    missing = [name for name in schema_fields if name not in forged]
    extra = [name for name in forged if name not in schema_fields]
    if missing or extra:
        raise ValueError(
            "the functional tampering matrix does not cover the design-bound fields: "
            f"missing {sorted(missing)}, unknown {sorted(extra)}; every field the "
            "provenance matcher compares needs a row, or the matrix passes while a "
            "forged value for that field goes unchecked"
        )
    return tuple((name, forged[name]) for name in schema_fields)


#: The tampering matrix. :func:`_tamperings` refuses a derivation that does not cover
#: :data:`DESIGN_BOUND_FIELDS`, so the matrix cannot fall behind the fields it names.
FUNCTIONAL_TAMPERINGS: tuple[tuple[str, Any], ...] = _tamperings(
    DESIGN_BOUND_FIELDS, _FORGED_FUNCTIONAL_VALUES
)
