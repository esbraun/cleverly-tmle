"""Joint-coverage rows for a fit's simultaneous band and its pointwise control.

A joint cell reads one fit's whole family of estimates at once.  It has no scalar estimand,
so each row records the max-t statistic in place of an estimate:

============  ===================================================================
column        value
============  ===================================================================
``estimate``  ``max_j |f_j(psi_j) - f_j(truth_j)| / (se_j f_j'(psi_j))``, each term
              on the scale the row's own intervals are built on: the band's scale
              (``SimultaneousBands.scales``) on a band row, the pointwise interval's
              scale (the log scale for a ratio) on a control row
``truth``     0
``std_error`` the critical value the row's intervals used
``covered``   band row: every truth lies inside the package's own band.
              Control row: every truth lies inside the package's own pointwise
              interval
``rejected``  ``not covered``
============  ===================================================================

Each row satisfies ``covered == (estimate <= std_error)``, and asserts it before it returns,
so the statistic is the one the row's intervals answer to.  Only ``covered`` and ``rejected``
carry a verdict.  The statistic and the critical value are published so a reader can see how
far the family sat from its band.

A band the package refuses to build (a logit band over an estimate on or outside its range)
counts as not covered: its row records the critical value the band would have used and the
sentinel statistic ``critical + 1.0`` (:data:`REFUSED_MARGIN`).  It is not a lost replicate,
and it adds no column; :func:`scale_coverage_rows` reports the refusals in a diagnostic cell.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cleverly.inference.delta import Transform
from cleverly.inference.multiplier import out_of_range, simultaneous_bands
from tests.studies.evidence.properties import PropertyCell

#: The cell suffixes of a joint pair, and the role each answers to.
BAND = "simultaneous_band"
CONTROL = "pointwise_joint_control"
FAMILY = "simultaneous_coverage"
#: The suffix of a band cell's identity-scale control, and the role its own rule reads
#: (:func:`~tests.studies.evidence.property_verdicts.simultaneous_coverage_verdicts`).
SCALE_CONTROL = "identity_scale_control"
SCALE_ROLE = "scale_control"
#: What the 99% lower bound of the band's paired coverage gain over its identity-scale control
#: must reach.  A transform that does nothing gains exactly 0 on every replicate and fails.
SCALE_GAIN_FLOOR = 1e-4
#: The suffix of the diagnostic cell that counts refused bands.
REFUSED = "refused_band"
#: How far above its critical value a refused band's sentinel statistic sits.
REFUSED_MARGIN = 1.0
#: The role of a cell that reports beside a verdict and declares none; the value of
#: :data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE`.
DIAGNOSTIC = "diagnostic"
#: The relative distance from its critical value inside which a row's statistic may disagree
#: with its coverage indicator: the two are computed along different float paths.
_TIE_TOLERANCE = 1e-9


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


def pointwise_scale(estimate: Any) -> str:
    """The scale an estimate's pointwise interval is built on.

    Its own transform's name, ``"log"`` for a ratio, and ``"identity"`` otherwise, as
    ``SimultaneousBands.scales`` names them.
    """
    if estimate.transform is not None:
        return str(estimate.transform.name)
    return "log" if estimate.scale == "ratio" else "identity"


def standardized_deviation(estimate: Any, truth: float, scale: str) -> float:
    """``|f(psi) - f(truth)| / (se f'(psi))`` on the scale ``scale`` names.

    ``"identity"`` and ``"log"`` (a ratio) read the pointwise standard error as it is.  An
    explicit transform's name reads the standard error the estimate carries on that scale.
    ``"logit"`` and ``"range_logit"`` read the logit of the estimate's position in its
    ``bounds``, with the standard error carried there by the delta method, which is how the
    package builds that band.  A zero standard error gives 0 at the truth and infinity
    elsewhere: the band of such an estimate is the point.

    Raises
    ------
    ValueError
        If ``scale`` names no scale the estimate can be read on.
    """
    se = float(estimate.plugin_std_error)
    psi = float(estimate.psi)
    if scale == "identity":
        center, target, spread = psi, float(truth), se
    elif scale == "log":
        center, target, spread = float(estimate.inference_value), math.log(truth), se
    elif scale in {"logit", "range_logit"}:
        if estimate.bounds is None:
            raise ValueError(f"{estimate.name} has no bounds to read on the {scale} scale")
        ranged = Transform.range_logit(*estimate.bounds)
        center, target = float(ranged.forward(psi)), float(ranged.forward(truth))
        spread = se * float(ranged.slope(psi))
    elif estimate.transform is not None and estimate.transform.name == scale:
        transform = estimate.transform
        center, target, spread = (
            float(estimate.inference_value),
            float(transform.forward(truth)),
            se,
        )
    else:
        raise ValueError(f"{estimate.name} cannot be read on the {scale!r} scale")
    if spread == 0.0:
        return 0.0 if center == target else math.inf
    return abs(center - target) / spread


def _checked(statistic: float, critical: float, covered: bool, cell: str) -> None:
    """Assert that a row's coverage indicator is its statistic inside its critical value."""
    if abs(statistic - critical) <= _TIE_TOLERANCE * max(1.0, abs(critical)):
        return
    if covered != (statistic <= critical):
        raise AssertionError(
            f"{cell}: covered={covered} but the statistic {statistic:.9g} is "
            f"{'inside' if statistic <= critical else 'outside'} the critical value "
            f"{critical:.9g}; the statistic is not read on the scale of the row's intervals"
        )


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


def _judged_row(
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
    """:func:`_row`, after asserting ``covered == (statistic <= critical)``."""
    _checked(statistic, critical, covered, cell)
    return _row(
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        statistic=statistic,
        critical=critical,
        covered=covered,
    )


def _band_reading(
    bands: Any, estimates: Mapping[str, Any], truth: Mapping[str, float]
) -> tuple[float, float, bool]:
    """``(statistic, critical, covered)`` of a built band, the statistic on its scales."""
    statistic = max(
        standardized_deviation(estimates[name], truth[name], bands.scales[name])
        for name in estimates
    )
    covered = all(bands.bands[name][0] <= truth[name] <= bands.bands[name][1] for name in estimates)
    return float(statistic), float(bands.critical_value), bool(covered)


def _pointwise_reading(
    estimates: Mapping[str, Any], truth: Mapping[str, float]
) -> tuple[float, bool]:
    """``(statistic, covered)`` of the pointwise intervals, the statistic on their scales."""
    statistic = max(
        standardized_deviation(estimate, truth[name], pointwise_scale(estimate))
        for name, estimate in estimates.items()
    )
    covered = all(
        estimate.plugin_interval[0] <= truth[name] <= estimate.plugin_interval[1]
        for name, estimate in estimates.items()
    )
    return float(statistic), bool(covered)


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
        A fit result with ``simultaneous`` and one estimate per name. The row reads
        ``plugin_std_error`` and ``plugin_interval``, which give the numbers ``std_error`` and
        ``ci`` give on an inferential estimate and the retained diagnostic on any other.
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
        If the fit reported no band, a band over a different family, or a row whose
        coverage indicator is not its statistic inside its critical value.
    """
    bands = result.simultaneous
    if bands is None:
        raise AssertionError(f"the {label} fit reported no simultaneous band")
    if set(bands.bands) != set(names):
        raise AssertionError(f"the {label} band covers {sorted(bands.bands)}, not {sorted(names)}")
    estimates = {name: result[name] for name in names}
    band_statistic, band_critical, band = _band_reading(bands, estimates, truth)
    point_statistic, pointwise = _pointwise_reading(estimates, truth)
    band_cell, control_cell = joint_cells(label)
    return [
        _judged_row(
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            statistic=statistic,
            critical=critical,
            covered=covered,
        )
        for cell, role, statistic, critical, covered in (
            (band_cell, "positive", band_statistic, band_critical, band),
            (control_cell, "control", point_statistic, float(pointwise_critical), pointwise),
        )
    ]


@dataclass(frozen=True)
class BandOptions:
    """The configuration of one simultaneous band.

    Parameters
    ----------
    band_scale : str
        ``"logit"`` or ``"identity"``.
    kind : str
        The multiplier distribution.
    n_replicates : int
        The multiplier draws.
    random_state : int
        The seed of those draws.
    """

    band_scale: str
    kind: str
    n_replicates: int
    random_state: int


def scale_cells(
    label: str, *, n: int, replicates: int, seed: int, reporting: Sequence[str]
) -> tuple[PropertyCell, ...]:
    """The declared cells of a band read against its identity-scale control.

    The band and its pointwise control (:func:`joint_property_cells`), the identity-scale
    control, one diagnostic cell per ``reporting`` suffix, and the refused-band count.  Every
    cell reads the same fits.
    """
    extra = [(SCALE_CONTROL, SCALE_ROLE), *((name, DIAGNOSTIC) for name in reporting)]
    extra.append((REFUSED, DIAGNOSTIC))
    return (
        *joint_property_cells(label, n=n, replicates=replicates, seed=seed),
        *(
            PropertyCell(
                property=FAMILY,
                cell=f"{label}__{suffix}",
                dgp=JointLaw(label),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=replicates,
                seed=seed,
                role=role,
                estimand="max_t",
            )
            for suffix, role in extra
        ),
    )


def _options_reading(
    estimates: Mapping[str, Any],
    truth: Mapping[str, float],
    options: BandOptions,
    *,
    alpha: float,
) -> tuple[float, float, bool, bool]:
    """``(statistic, critical, covered, refused)`` of the band ``options`` build.

    A band the package refuses takes its critical value from the identity band with the same
    draws, which is the value it would have used, because the critical value does not depend
    on the scale.
    """
    refused = bool(out_of_range(estimates, options.band_scale))
    bands = simultaneous_bands(
        estimates,
        alpha=alpha,
        n_replicates=options.n_replicates,
        kind=options.kind,  # type: ignore[arg-type]
        random_state=options.random_state,
        band_scale="identity" if refused else options.band_scale,  # type: ignore[arg-type]
    )
    if refused:
        critical = float(bands.critical_value)
        return critical + REFUSED_MARGIN, critical, False, True
    return (*_band_reading(bands, estimates, truth), False)


def scale_coverage_rows(
    result: Any,
    truth: Mapping[str, float],
    names: Sequence[str],
    *,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    pointwise_critical: float,
    options: BandOptions,
    reporting: Mapping[str, BandOptions],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Every row :func:`scale_cells` declares, from one fit's estimates.

    The band row reads the fit's own band, which ``options`` must describe.  The
    identity-scale control is the identity band at the same draws, and the function asserts
    that its critical value is the band's: only the scale differs between the two rows.  Each
    ``reporting`` band is built from the same estimates.  A band the package refused counts
    as not covered (the module docstring), and the ``refused_band`` row records ``1`` in
    ``estimate`` and ``covered`` for it, with the band's critical value as ``std_error``.

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
        The declared replication count of the cells.
    pointwise_critical : float
        The critical value of the pointwise intervals.
    options : BandOptions
        The configuration the fit built its band with.
    reporting : mapping of str to BandOptions
        Each diagnostic cell suffix and the band it reports.
    alpha : float
        The joint significance level of every band.

    Returns
    -------
    list of dict
        The band, pointwise control, identity-scale control, reporting and refused rows.

    Raises
    ------
    AssertionError
        If the fit's band does not match ``options``, a band is missing that the package
        builds, the identity control's critical value differs from the band's, or a row's
        coverage indicator is not its statistic inside its critical value.
    """
    order = [name for name in result if name in set(names)]
    if sorted(order) != sorted(names):
        raise AssertionError(f"the {label} fit reports {sorted(order)}, not {sorted(names)}")
    estimates = {name: result[name] for name in order}
    refused = bool(out_of_range(estimates, options.band_scale))
    bands = result.simultaneous
    identity = simultaneous_bands(
        estimates,
        alpha=alpha,
        n_replicates=options.n_replicates,
        kind=options.kind,  # type: ignore[arg-type]
        random_state=options.random_state,
        band_scale="identity",
    )
    identity_reading = _band_reading(identity, estimates, truth)
    if refused:
        if bands is not None:
            raise AssertionError(f"the {label} fit built a band over an out-of-range estimate")
        critical = float(identity.critical_value)
        band_reading = (critical + REFUSED_MARGIN, critical, False)
    else:
        if bands is None:
            raise AssertionError(f"the {label} fit reported no simultaneous band")
        if set(bands.bands) != set(names):
            raise AssertionError(
                f"the {label} band covers {sorted(bands.bands)}, not {sorted(names)}"
            )
        built = (bands.band_scale, bands.kind, int(bands.n_replicates))
        declared = (options.band_scale, options.kind, options.n_replicates)
        if built != declared:
            raise AssertionError(f"the {label} band is {built}, not the declared {declared}")
        band_reading = _band_reading(bands, estimates, truth)
    if not math.isclose(band_reading[1], identity_reading[1], rel_tol=1e-12):
        raise AssertionError(
            f"the {label} identity control's critical value {identity_reading[1]!r} is not "
            f"the band's {band_reading[1]!r}"
        )
    point_statistic, pointwise = _pointwise_reading(estimates, truth)
    band_cell, control_cell = joint_cells(label)
    readings: list[tuple[str, str, float, float, bool]] = [
        (band_cell, "positive", *band_reading),
        (control_cell, "control", point_statistic, float(pointwise_critical), pointwise),
        (f"{label}__{SCALE_CONTROL}", SCALE_ROLE, *identity_reading),
    ]
    for suffix, configuration in reporting.items():
        statistic, critical, covered, _ = _options_reading(
            estimates, truth, configuration, alpha=alpha
        )
        readings.append((f"{label}__{suffix}", DIAGNOSTIC, statistic, critical, covered))
    rows = [
        _judged_row(
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            statistic=statistic,
            critical=critical,
            covered=covered,
        )
        for cell, role, statistic, critical, covered in readings
    ]
    rows.append(
        _row(
            cell=f"{label}__{REFUSED}",
            role=DIAGNOSTIC,
            replicate=replicate,
            n=n,
            requested=requested,
            statistic=float(refused),
            critical=band_reading[1],
            covered=refused,
        )
    )
    return rows
