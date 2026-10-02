"""Joint-coverage rows for a fit's simultaneous band and its pointwise control.

A joint cell reads one fit's whole family of estimates at once.  It has no scalar estimand,
so each row records the max-t statistic in place of an estimate:

============  ===================================================================
column        value
============  ===================================================================
``estimate``  ``max_j |psi_j - truth_j| / se_j``, each term on the scale its
              estimand's interval is built on (the log scale for a ratio)
``truth``     0
``std_error`` the critical value the row's intervals used
``covered``   positive row: every truth lies inside the package's own band.
              Control row: every truth lies inside the package's own pointwise
              interval
``rejected``  ``not covered``
============  ===================================================================

Only ``covered`` and ``rejected`` carry a verdict.  The statistic and the critical value are
published so a reader can see how far the family sat from its band.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tests.studies.evidence.properties import PropertyCell

#: The cell suffixes of a joint pair, and the role each answers to.
BAND = "simultaneous_band"
CONTROL = "pointwise_joint_control"
FAMILY = "simultaneous_coverage"


def joint_cells(label: str) -> tuple[str, str]:
    """The band cell and its pointwise control for one joint label."""
    return f"{label}__{BAND}", f"{label}__{CONTROL}"


@dataclass(frozen=True)
class JointLaw:
    """The law a joint cell declares: its rows read the max-t statistic against zero.

    The truth-binding gate reads every declared cell's truth back off the law it names.  A
    joint row's ``truth`` column is the zero of its max-t statistic, so this law integrates
    to zero.  Its name is unique per label, so two joint cells of one study never share a
    sample stream in the collision gate.

    Parameters
    ----------
    label : str
        The joint cell prefix.
    """

    label: str

    @property
    def name(self) -> str:
        """The law's name, unique per joint label."""
        return f"{self.label}__joint_max_t"

    def truth(self) -> dict[str, float]:
        """The zero every joint row's statistic is recorded against."""
        return {"max_t": 0.0}


def joint_property_cells(
    label: str, *, n: int, replicates: int, seed: int, control: bool = True
) -> tuple[PropertyCell, ...]:
    """The declared band cell, and its pointwise control unless ``control`` is false.

    ``seed`` names the root of the cell's per-replication stream.  Nothing here is passed to
    :func:`~tests.studies.evidence.properties.run_cells`.
    """
    roles = [(BAND, "positive")] + ([(CONTROL, "control")] if control else [])
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=f"{label}__{kind}",
            dgp=JointLaw(label),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=n,
            replicates=replicates,
            seed=seed,
            role=role,
            estimand="max_t",
        )
        for kind, role in roles
    )


def inference_scale(estimate: Any, truth: float) -> tuple[float, float]:
    """``(truth, estimate)`` on the scale the estimate's interval is built on.

    A ratio estimate builds its interval on the log scale and exponentiates it, so its
    standardized deviation is read there too.
    """
    if estimate.scale == "ratio":
        return math.log(truth), float(estimate.inference_value)
    return float(truth), float(estimate.psi)


def _row(
    *,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    statistic: float,
    critical: float,
    covered: bool,
) -> dict[str, Any]:
    return {
        "property": FAMILY,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": statistic,
        "std_error": critical,
        "covered": int(covered),
        "rejected": int(not covered),
    }


def joint_coverage_rows(
    result: Any,
    truth: Mapping[str, float],
    names: Sequence[str],
    *,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    pointwise_critical: float,
) -> list[dict[str, Any]]:
    """The band row and its pointwise control, from one fit's estimates.

    Parameters
    ----------
    result : object
        A fit result with ``simultaneous`` and one estimate per name.
    truth : mapping of str to float
        The true value of every name, on its natural scale.
    names : sequence of str
        The family the band must cover, exactly.
    label : str
        The joint cell prefix.
    replicate : int
        The replication index.
    n : int
        The sample size.
    requested : int
        The declared replication count of the cell.
    pointwise_critical : float
        The critical value of the pointwise intervals, recorded on the control row.

    Returns
    -------
    list of dict
        The positive band row, then the control row.

    Raises
    ------
    AssertionError
        If the fit reported no band, or a band over a different family.
    """
    bands = result.simultaneous
    if bands is None:
        raise AssertionError(f"the {label} fit reported no simultaneous band")
    if set(bands.bands) != set(names):
        raise AssertionError(f"the {label} band covers {sorted(bands.bands)}, not {sorted(names)}")
    deviations = []
    for name in names:
        target, point = inference_scale(result[name], truth[name])
        deviations.append(abs(point - target) / float(result[name].std_error))
    statistic = float(max(deviations))
    band = all(bands.bands[name][0] <= truth[name] <= bands.bands[name][1] for name in names)
    pointwise = all(result[name].ci[0] <= truth[name] <= result[name].ci[1] for name in names)
    band_cell, control_cell = joint_cells(label)
    return [
        _row(
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            statistic=statistic,
            critical=critical,
            covered=covered,
        )
        for cell, role, critical, covered in (
            (band_cell, "positive", float(bands.critical_value), band),
            (control_cell, "control", float(pointwise_critical), pointwise),
        )
    ]
