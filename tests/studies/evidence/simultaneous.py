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
and it adds no column; :func:`scale_coverage_rows` reports the refusals in a diagnostic cell,
and :func:`joint_coverage_rows` records them the same way for a fit on the default band.

While :data:`BAND_OBSERVER` holds a list, :func:`joint_coverage_rows` appends one record per
band it reads (its label, whether the band was refused, the number of estimates outside their
range, the smallest Kish count of the banded influence curves and the band's options).  The
targeted band rerun's smoke reads it; it never changes a row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cleverly.inference import kish_count
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
#: Where :func:`joint_coverage_rows` reports each band it reads, or ``None`` when nobody reads.
BAND_OBSERVER: list[dict[str, Any]] | None = None


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

    Its :attr:`~cleverly.inference.ParameterEstimate.interval_scale`: its own transform's name,
    ``"log"`` for a ratio, ``"range_logit"`` for a bounded unclustered mean, ``"mover"`` for a
    bounded difference of two such means, ``"skew_t"`` for an unclustered estimate of a
    continuous outcome, and ``"identity"`` otherwise.
    """
    return str(estimate.interval_scale)


def standardized_deviation(estimate: Any, truth: float, scale: str) -> float:
    """``|f(psi) - f(truth)| / (se f'(psi))`` on the scale ``scale`` names.

    ``"identity"`` and ``"log"`` (a ratio) read the pointwise standard error as it is.  An
    explicit transform's name reads the standard error the estimate carries on that scale.
    ``"logit"`` and ``"range_logit"`` read the logit of the estimate's position in its
    ``bounds``, with the standard error carried there by the delta method, which is how the
    package builds that band.  ``"mover"`` and ``"skew_t"``, the asymmetric pointwise
    intervals, read ``|Phi^-1(1 - p/2)|`` of the test of the truth, which inverts the interval.
    A zero standard error gives 0 at the truth and infinity
    elsewhere: the band of such an estimate is the point.

    Raises
    ------
    ValueError
        If ``scale`` names no scale the estimate can be read on.
    """
    se = float(estimate.plugin_std_error)
    psi = float(estimate.psi)
    if se == 0.0:
        # The band of a zero-SE estimate is the point on every scale, even on a bound.
        return 0.0 if psi == float(truth) else math.inf
    transform = estimate.transform
    if transform is not None and transform.name == scale:
        center, target, spread = (
            float(estimate.inference_value),
            float(transform.forward(truth)),
            se,
        )
    elif scale == "identity":
        center, target, spread = psi, float(truth), se
    elif scale == "log":
        center, target, spread = float(estimate.inference_value), math.log(truth), se
    elif scale in {"mover", "skew_t"}:
        # An asymmetric interval has no single spread; its test inverts it, so the
        # statistic Phi^-1(1 - p/2) is at most z exactly when the interval holds the truth.
        return abs(float(estimate.wald_test(float(truth)).statistic))
    elif scale in {"logit", "range_logit"}:
        if estimate.bounds is None:
            raise ValueError(f"{estimate.name} has no bounds to read on the {scale} scale")
        # Written from the formula rather than the package's transform, so a wrong slope in
        # the package fails the row's own identity.
        low, high = (float(value) for value in estimate.bounds)
        center, target = _range_logit(psi, low, high), _range_logit(float(truth), low, high)
        spread = se * (high - low) / ((psi - low) * (high - psi))
    else:
        raise ValueError(f"{estimate.name} cannot be read on the {scale!r} scale")
    return abs(center - target) / spread


def _range_logit(value: float, low: float, high: float) -> float:
    """``logit((value - low) / (high - low))``."""
    return math.log((value - low) / (high - value))


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
    """``(statistic, covered)`` of the pointwise intervals, the statistic on their scales.

    An estimate whose plug-in interval is not reported (a constant arm cell, or an estimate on
    its bound) has no interval to cover the truth, so the row counts as a miss with an
    infinite statistic.
    """
    if any(
        not all(math.isfinite(limit) for limit in estimate.plugin_interval)
        for estimate in estimates.values()
    ):
        return math.inf, False
    statistic = max(
        standardized_deviation(estimate, truth[name], pointwise_scale(estimate))
        for name, estimate in estimates.items()
    )
    covered = all(
        estimate.plugin_interval[0] <= truth[name] <= estimate.plugin_interval[1]
        for name, estimate in estimates.items()
    )
    return float(statistic), bool(covered)


def refused_scale(result: Any, estimates: Mapping[str, Any]) -> str | None:
    """The band scale under which the fit refused its band, or ``None``.

    The fit's estimator names its ``band_scale``.  The band is refused when an estimate of the
    fit that supplies inference, inside ``estimates`` or not, lies on or outside its range on
    that scale (:func:`~cleverly.inference.multiplier.out_of_range`), as the fit decides it.
    """
    scale = getattr(getattr(result, "estimator", None), "band_scale", None)
    if scale is None:
        return None
    every = dict(getattr(result, "estimates", estimates))
    inferential = {name: value for name, value in every.items() if value.supplies_inference}
    return str(scale) if out_of_range(inferential, str(scale)) else None


def refused_critical_value(result: Any, estimates: Mapping[str, Any]) -> float:
    """The critical value a refused band would have used.

    The identity band over the same estimates at the fit's own alpha, multiplier kind, draws,
    seed and clusters: the critical value does not depend on the scale.  On a repeated fit it
    is the critical value of the combined estimates' curves, not the median over the draws of
    the split, which a refused band never computes.
    """
    estimator = result.estimator
    inferential = {name: value for name, value in estimates.items() if value.supplies_inference}
    data = getattr(result, "data", None)
    bands = simultaneous_bands(
        inferential,
        alpha=float(getattr(estimator, "alpha_sig", 0.05)),
        n_replicates=int(estimator.n_multiplier),
        kind=estimator.multiplier_kind,
        random_state=estimator.random_state,
        cluster=None if data is None else getattr(data, "cluster", None),
        band_scale="identity",
    )
    return float(bands.critical_value)


def _observe(
    label: str, result: Any, estimates: Mapping[str, Any], bands: Any, refused: str | None
) -> None:
    """Report one band to :data:`BAND_OBSERVER` when it holds a list."""
    if BAND_OBSERVER is None:
        return
    estimator = getattr(result, "estimator", None)
    BAND_OBSERVER.append(
        {
            "label": label,
            "refused": refused is not None,
            "out_of_range": len(out_of_range(estimates, "logit")),
            "kish_min": min(
                (kish_count(value.influence_curve) for value in estimates.values()),
                default=math.inf,
            ),
            "band_scale": getattr(bands, "band_scale", getattr(estimator, "band_scale", None)),
            "kind": getattr(bands, "kind", getattr(estimator, "multiplier_kind", None)),
            "n_replicates": getattr(
                bands, "n_replicates", getattr(estimator, "n_multiplier", None)
            ),
        }
    )


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
        If the fit reported no band and did not refuse one, a band over a different family,
        or a row whose coverage indicator is not its statistic inside its critical value.

    Notes
    -----
    A fit that refused its logit band (an estimate on or outside its range,
    :func:`refused_scale`) gets the not-covered band row of the module docstring, with
    :func:`refused_critical_value` as its critical value.
    """
    estimates = {name: result[name] for name in names}
    bands = result.simultaneous
    refused = refused_scale(result, estimates)
    if bands is None:
        if refused is None:
            raise AssertionError(f"the {label} fit reported no simultaneous band")
        critical = refused_critical_value(result, estimates)
        band_statistic, band_critical, band = critical + REFUSED_MARGIN, critical, False
    else:
        if refused is not None:
            raise AssertionError(f"the {label} fit built a band over an out-of-range estimate")
        if set(bands.bands) != set(names):
            raise AssertionError(
                f"the {label} band covers {sorted(bands.bands)}, not {sorted(names)}"
            )
        band_statistic, band_critical, band = _band_reading(bands, estimates, truth)
    _observe(label, result, estimates, bands, refused)
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


# ------------------------------------------------------------------ the t reference band

#: The suffix of a ``band-t-reference`` band cell and of its paired normal-reference control.
T_BAND = "t_band"
REFERENCE_CONTROL = "normal_reference_control"
#: The role of the normal-reference control, read by the paired-gain rule
#: (:func:`~tests.studies.evidence.property_verdicts.reference_gain_verdicts`).
REFERENCE_ROLE = "reference_control"
#: The study's own copy of the band's df floor (Fix-2 plan 2.4).  Not imported from the
#: package, so a changed floor there fails the agreement assertion of :func:`t_band_rows`.
T_DF_FLOOR = 5.0
#: Coordinates at or above this df must follow the first-order expansion
#: ``c_j / c - 1 = (c^2 + 1) / (4 nu_j)`` within :data:`EXPANSION_TOLERANCE` (plan 5.2).
EXPANSION_DF = 50.0
EXPANSION_TOLERANCE = 0.10
#: The multivariate t diagnostic's Monte Carlo draws and seed (common random numbers).
MVT_DRAWS = 20_000
MVT_SEED = 20261113


def t_cells(label: str) -> tuple[str, str, str]:
    """The ``t`` band cell, its normal-reference control and its pointwise control."""
    return f"{label}__{T_BAND}", f"{label}__{REFERENCE_CONTROL}", f"{label}__{CONTROL}"


def _own_kish(values: Any) -> float:
    """``(sum D^2)^2 / sum D^4`` of the centred ``D``, independent of the package's count."""
    centred = np.asarray(values, dtype=float) - np.mean(values)
    squares = centred**2
    fourth = float(np.sum(squares**2))
    return math.inf if fourth == 0.0 else float(np.sum(squares) ** 2 / fourth)


def study_kish(curve: Any, cluster: Any = None) -> float:
    """The study's own Kish count of a curve, of its cluster sums when ``cluster`` is given."""
    values = np.asarray(curve, dtype=float)
    if cluster is not None:
        _, inverse = np.unique(np.asarray(cluster), return_inverse=True)
        values = np.bincount(inverse, weights=values)
    return _own_kish(values)


def t_quantile(critical: float, nu: float) -> float:
    """``c_j = F_{t, nu}^{-1}(Phi(c))``, and ``c`` itself at ``nu = inf``."""
    from scipy.stats import norm, t

    if math.isinf(nu):
        return float(critical)
    return float(t.isf(norm.sf(critical), nu))


def t_statistic(deviation: float, nu: float) -> float:
    """``Phi^{-1}(F_{t, nu}(|T|))``: inside ``c`` exactly when ``|T|`` is inside ``c_j``."""
    from scipy.stats import norm, t

    if math.isinf(nu) or math.isinf(deviation):
        return float(deviation)
    return float(norm.isf(t.sf(deviation, nu)))


def check_expansion(critical: float, nus: Mapping[str, float], values: Mapping[str, float]) -> None:
    """Assert the first-order expansion on every coordinate with ``nu >= EXPANSION_DF``.

    A code-read gate (plan 5.2): deterministic given the fit, so a failure is an
    implementation defect, not a finite-sample reading.
    """
    for name, nu in nus.items():
        if math.isinf(nu) or nu < EXPANSION_DF:
            continue
        observed = values[name] / critical - 1.0
        predicted = (critical**2 + 1.0) / (4.0 * nu)
        if abs(observed / predicted - 1.0) > EXPANSION_TOLERANCE:
            raise AssertionError(
                f"{name}: c_j / c - 1 = {observed:.6g} departs from (c^2 + 1) / (4 nu) = "
                f"{predicted:.6g} by more than {EXPANSION_TOLERANCE:.0%} at nu = {nu:.6g}"
            )


def check_widening(
    critical: float, kish: Mapping[str, float], values: Mapping[str, float], label: str
) -> None:
    """Assert ``max_j (c_j / c - 1) > 0`` whenever some Kish count is finite (plan 5.2).

    The deterministic nonzero witness, read from the package's ``c_j``: a ``t`` band that
    collapsed to the normal band (``nu = inf`` on every coordinate, or ``c_j = c``) fails it on
    the first replicate, whatever the paired gain's power.
    """
    if all(math.isinf(value) for value in kish.values()):
        return
    if max(values[name] / critical - 1.0 for name in values) <= 0.0:
        raise AssertionError(
            f"the {label} t band does not widen any coordinate although some Kish count is finite"
        )


def moment_df(curve: Any) -> float:
    """Fix-3's pointwise df ``2K / (1 - K/m)`` over the rows that carry the curve.

    ``K`` is the Kish count of the centred curve over its nonzero rows and ``m`` their number.
    ``inf`` when every carrying term has one magnitude (``K = m``) or the curve is zero.
    """
    values = np.asarray(curve, dtype=float)
    carrying = values[values != 0.0]
    m = carrying.size
    if m == 0:
        return math.inf
    kish = _own_kish(carrying)
    if math.isinf(kish) or kish >= m:
        return math.inf
    return float(2.0 * kish / (1.0 - kish / m))


def model_df(curve: Any, cells: Any) -> float:
    """The model-assisted df: the Kish count of the cell means of ``d^2`` (D3 (b)).

    ``cells`` gives each row's covariate cell; a row's ``s_i^2`` is the mean of the squared
    centred curve over its cell (``rows/Fix-2/diag/dfpass.py``).
    """
    values = np.asarray(curve, dtype=float)
    squares = (values - values.mean()) ** 2
    _, codes = np.unique(np.asarray(cells), return_inverse=True)
    means = np.bincount(codes, squares) / np.bincount(codes)
    s2 = means[codes]
    fourth = float(np.sum(s2**2))
    return math.inf if fourth == 0.0 else float(np.sum(s2) ** 2 / fourth)


def common_mvt_critical(critical: float, nu: float, curves: np.ndarray, alpha: float) -> float:
    """``c q(nu) / q(inf)``: the exact multivariate t with one shared denominator (D3 (d)).

    ``q`` is the ``1 - alpha`` quantile of ``max_j |Z_j| / sqrt(chi2_nu / nu)`` with ``Z``
    drawn at the curves' own correlation, on :data:`MVT_DRAWS` common random numbers.
    """
    from scipy.stats import chi2

    if math.isinf(nu):
        return float(critical)
    centred = curves - curves.mean(axis=0)
    covariance = centred.T @ centred
    sd = np.sqrt(np.diag(covariance))
    sd[sd == 0.0] = 1.0
    correlation = covariance / np.outer(sd, sd)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    rng = np.random.default_rng(MVT_SEED)
    maxima = np.abs(rng.standard_normal((MVT_DRAWS, correlation.shape[0])) @ factor.T).max(axis=1)
    scale = np.sqrt(chi2.ppf(rng.uniform(size=MVT_DRAWS), nu) / nu)
    q_inf = float(np.quantile(maxima, 1.0 - alpha))
    return float(critical * np.quantile(maxima / scale, 1.0 - alpha) / q_inf)


@dataclass(frozen=True)
class TArm:
    """A diagnostic df arm of a ``t`` band cell, read on the same fits without a verdict.

    Parameters
    ----------
    suffix : str
        The diagnostic cell suffix.
    rule : str
        ``"floor"`` (Kish with ``floor``), ``"moment"`` (Fix-3's df, floored),
        ``"model"`` (the model-assisted df, floored), ``"common_copula"`` (one df
        ``max(min_j Kish_j, floor)`` through the per-coordinate mapping) or ``"common_mvt"``
        (that df through the exact multivariate t).
    floor : float
        The df floor of the arm.
    """

    suffix: str
    rule: str
    floor: float = T_DF_FLOOR


#: The diagnostic arms every ``t`` band cell reads (plan 5.2, D3 (b) to (d)).
T_ARMS: tuple[TArm, ...] = (
    TArm("model_df_diagnostic", "model"),
    TArm("moment_df_diagnostic", "moment"),
    TArm("common_df_copula_diagnostic", "common_copula"),
    TArm("common_df_mvt_diagnostic", "common_mvt"),
)
#: The floor arms the held-out cell reads besides them (plan 5.2, D4).
FLOOR_ARMS: tuple[TArm, ...] = (
    TArm("floor_2_diagnostic", "floor", 2.0),
    TArm("floor_10_diagnostic", "floor", 10.0),
)


def t_property_cells(
    label: str, *, n: int, replicates: int, seed: int, arms: Sequence[TArm]
) -> tuple[PropertyCell, ...]:
    """The declared cells of one ``t`` band label: band, two controls and ``arms``."""
    band, reference, control = t_cells(label)
    roles = [(band, "positive"), (reference, REFERENCE_ROLE), (control, "control")]
    roles += [(f"{label}__{arm.suffix}", DIAGNOSTIC) for arm in arms]
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=cell,
            dgp=JointLaw(label),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=n,
            replicates=replicates,
            seed=seed,
            role=role,
            estimand="max_t",
        )
        for cell, role in roles
    )


def _arm_dfs(
    arm: TArm,
    kish: Mapping[str, float],
    curves: Mapping[str, np.ndarray],
    cells: Any,
) -> dict[str, float]:
    if arm.rule == "floor":
        return {name: max(value, arm.floor) for name, value in kish.items()}
    if arm.rule == "moment":
        return {name: max(moment_df(curve), arm.floor) for name, curve in curves.items()}
    if arm.rule == "model":
        if cells is None:
            raise ValueError("the model-assisted arm needs each row's covariate cell")
        return {name: max(model_df(curve, cells), arm.floor) for name, curve in curves.items()}
    if arm.rule in {"common_copula", "common_mvt"}:
        common = max(min(kish.values()), arm.floor)
        return dict.fromkeys(kish, common)
    raise ValueError(f"unknown df rule {arm.rule!r}")


def t_band_rows(
    result: Any,
    truth: Mapping[str, float],
    names: Sequence[str],
    *,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    pointwise_critical: float,
    constant_cells: Sequence[str],
    arms: Sequence[TArm] = T_ARMS,
    cells: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every row :func:`t_property_cells` declares, and the replicate's diagnostics record.

    The fit runs once with the default band.  The ``t`` band is rebuilt by the package from the
    same estimates with the fit's own draws, seed, clusters and scale, so its ``c`` is the
    fit's and the two bands are paired by construction; the function asserts both.  The ``t``
    row's statistic is :func:`t_statistic` at the study's own df (:func:`study_kish`, floored
    at :data:`T_DF_FLOOR`), its coverage the package band's own endpoints, and
    :func:`_checked` ties the two.  The package's ``reference_df`` must equal the study's df
    and the expansion gate (:func:`check_expansion`) must hold.

    A refused logit band, or a replicate whose band family holds a constant arm cell
    (``constant_cells`` non-empty, the row module's own copy of Fix-3's check, plan 2.5), gives
    every band row the refused-band sentinel, not covered.

    Returns
    -------
    tuple
        The rows, and a per-replicate record of ``max_j (c_j / c - 1)``, ``min_j nu_j``,
        whether the floor bound, the first-order union term ``sum_j 2 phi(c) (c_j - c)``, each
        coordinate's df, moment df and standardized deviation, the constant cells and the
        coverage of the shown ``t`` band.

    Raises
    ------
    AssertionError
        If the fit reported no band and did not refuse one, the rebuilt band's ``c`` differs
        from the fit's, the package df differs from the study's, the expansion gate fails, or
        a row's coverage indicator is not its statistic inside its critical value.
    """
    from scipy.stats import norm

    estimates = {name: result[name] for name in names}
    fitted = result.simultaneous
    refused = refused_scale(result, estimates)
    band_cell, reference_cell, control_cell = t_cells(label)
    point_statistic, pointwise = _pointwise_reading(estimates, truth)
    readings: list[tuple[str, str, float, float, bool]] = []
    record: dict[str, Any] = {
        "label": label,
        "replicate": replicate,
        "refused": refused is not None,
        "constant_cells": ";".join(constant_cells),
    }
    if fitted is None:
        if refused is None:
            raise AssertionError(f"the {label} fit reported no simultaneous band")
        critical = refused_critical_value(result, estimates)
        sentinel = (critical + REFUSED_MARGIN, critical, False)
        readings += [
            (band_cell, "positive", *sentinel),
            (reference_cell, REFERENCE_ROLE, *sentinel),
        ]
        readings += [(f"{label}__{arm.suffix}", DIAGNOSTIC, *sentinel) for arm in arms]
        record.update(critical=critical, t_covered_shown=float("nan"))
    else:
        if refused is not None:
            raise AssertionError(f"the {label} fit built a band over an out-of-range estimate")
        if set(fitted.bands) != set(names):
            raise AssertionError(f"the {label} band covers {sorted(fitted.bands)}")
        estimator = result.estimator
        data = getattr(result, "data", None)
        cluster = None if data is None else getattr(data, "cluster", None)
        rebuilt = simultaneous_bands(
            estimates,
            alpha=float(fitted.alpha),
            n_replicates=int(fitted.n_replicates),
            kind=fitted.kind,
            random_state=estimator.random_state,
            cluster=cluster,
            band_scale=fitted.band_scale,
            band_reference="t",
        )
        critical = float(fitted.critical_value)
        if rebuilt.critical_value != critical:
            raise AssertionError(
                f"the {label} t band's c {rebuilt.critical_value!r} is not the fit's {critical!r}"
            )
        curves = {name: np.asarray(estimates[name].influence_curve, dtype=float) for name in names}
        kish = {name: study_kish(curve, cluster) for name, curve in curves.items()}
        nus = {name: max(value, T_DF_FLOOR) for name, value in kish.items()}
        for name in names:
            if not math.isclose(rebuilt.reference_df[name], nus[name], rel_tol=1e-12):
                raise AssertionError(
                    f"{label} {name}: the package df {rebuilt.reference_df[name]!r} is not the "
                    f"study's {nus[name]!r}"
                )
        own = {name: t_quantile(critical, nus[name]) for name in names}
        values = {name: float(rebuilt.critical_values[name]) for name in names}
        for name in names:
            if not math.isclose(values[name], own[name], rel_tol=1e-12):
                raise AssertionError(
                    f"{label} {name}: the package c_j {values[name]!r} is not the study's "
                    f"{own[name]!r}"
                )
        check_expansion(critical, nus, values)
        check_widening(critical, kish, values, label)
        deviations = {
            name: standardized_deviation(estimates[name], truth[name], rebuilt.scales[name])
            for name in names
        }
        t_stat = max(t_statistic(deviations[name], nus[name]) for name in names)
        t_covered = all(
            rebuilt.bands[name][0] <= truth[name] <= rebuilt.bands[name][1] for name in names
        )
        normal = _band_reading(fitted, estimates, truth)
        _checked(t_stat, critical, t_covered, band_cell)
        arm_readings: list[tuple[str, float, bool]] = []
        for arm in arms:
            arm_nus = _arm_dfs(arm, kish, curves, cells)
            if arm.rule == "common_mvt":
                stacked = np.column_stack([curves[name] for name in names])
                bound = common_mvt_critical(
                    critical, next(iter(arm_nus.values())), stacked, float(fitted.alpha)
                )
                statistic = max(deviations.values()) * critical / bound
                covered = all(value <= bound for value in deviations.values())
            else:
                statistic = max(t_statistic(deviations[name], arm_nus[name]) for name in names)
                covered = all(
                    deviations[name] <= t_quantile(critical, arm_nus[name]) for name in names
                )
            arm_readings.append((f"{label}__{arm.suffix}", statistic, covered))
        if constant_cells:
            sentinel = (critical + REFUSED_MARGIN, critical, False)
            readings += [
                (band_cell, "positive", *sentinel),
                (reference_cell, REFERENCE_ROLE, *sentinel),
            ]
            readings += [(cell, DIAGNOSTIC, *sentinel) for cell, _, _ in arm_readings]
        else:
            readings += [
                (band_cell, "positive", t_stat, critical, t_covered),
                (reference_cell, REFERENCE_ROLE, *normal),
            ]
            readings += [
                (cell, DIAGNOSTIC, statistic, critical, covered)
                for cell, statistic, covered in arm_readings
            ]
        finite = [nu for nu in nus.values() if not math.isinf(nu)]
        record.update(
            critical=critical,
            t_covered_shown=float(t_covered),
            normal_covered_shown=float(normal[2]),
            max_relative_widening=max(values[name] / critical - 1.0 for name in names),
            min_df=min(nus.values()),
            floor_bound=any(value < T_DF_FLOOR for value in kish.values()),
            union_gain=float(
                sum(2.0 * norm.pdf(critical) * (values[name] - critical) for name in names)
            ),
            df=";".join(f"{nus[name]:.6g}" for name in names),
            moment_df=";".join(f"{moment_df(curves[name]):.6g}" for name in names),
            deviations=";".join(f"{deviations[name]:.6g}" for name in names),
            beyond=";".join(str(int(deviations[name] > values[name])) for name in names),
            critical_values=";".join(f"{values[name]:.12g}" for name in names),
            finite_df=len(finite),
        )
    readings.append(
        (control_cell, "control", point_statistic, float(pointwise_critical), pointwise)
    )
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
    return rows, record
