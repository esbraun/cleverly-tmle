"""Qualitative stress surfaces for a simulated unobserved common cause.

The surface perturbs treatment and outcome through one shared latent vector, then
refits the complete estimator at each declared strength pair. It describes movement
of the fitted estimate on the scale ``movement_scale`` names. It does not provide a
bound, corrected estimate, test, or sensitivity-adjusted interval.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from numbers import Real
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.causal_data import arm_share
from ..utils.frames import emit_frame
from ..utils.random import resolve_assessment_seed
from ..utils.text import format_table
from ..validation.simulation import ReplicationFailure
from ._simulated_confounding_request import _validate_request, _ValidatedRequest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data import WeightReport
    from ..estimators.base import TMLEResult

__all__ = [
    "ConfounderStrengthGrid",
    "ObservedConfounderCalibration",
    "SimulatedConfoundingCell",
    "SimulatedConfoundingResult",
    "simulated_confounding",
]


@dataclass(frozen=True)
class ConfounderStrengthGrid:
    """Declare treatment and outcome perturbation strengths.

    Parameters
    ----------
    treatment : tuple of float, or mapping of str to tuple of float
        Treatment perturbation strengths. Continuous-treatment values are signed linear
        coefficients. Binary-treatment values are checked as flip probabilities from zero
        through 0.5 when the surface runs.  A vector treatment takes a mapping from each
        component column to its strengths, read by that component's law, and the surface
        evaluates every combination.  The mapping is stored as ``(column, strengths)``
        pairs in the declared order.
    outcome : tuple of float
        Outcome perturbation strengths. Gaussian strengths are signed. Binomial
        strengths are checked as flip probabilities when the surface runs.

    Attributes
    ----------
    components : tuple of str
    treatment_points : tuple

    See Also
    --------
    simulated_confounding : Evaluate the declared strength pairs.
    """

    treatment: tuple[float, ...] | tuple[tuple[str, tuple[float, ...]], ...]
    outcome: tuple[float, ...]

    def __post_init__(self) -> None:
        declared = self.treatment
        if isinstance(declared, Mapping):
            if not declared:
                raise ValueError("a treatment strength mapping must name at least one component")
            axes: list[tuple[str, tuple[float, ...]]] = []
            for column, values in declared.items():
                if not isinstance(column, str):
                    raise TypeError("a treatment strength mapping is keyed by component column")
                strengths = _numeric_strengths(values, f"treatment component {column!r}")
                if 0.0 not in strengths:
                    raise ValueError(f"treatment component {column!r} strengths must contain zero")
                axes.append((column, strengths))
            treatment: Any = tuple(axes)
        else:
            treatment = _numeric_strengths(declared, "treatment")
            if 0.0 not in treatment:
                raise ValueError("treatment and outcome strengths must each contain zero")
        outcome = _numeric_strengths(self.outcome, "outcome")
        if 0.0 not in outcome:
            raise ValueError("treatment and outcome strengths must each contain zero")
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)

    @property
    def components(self) -> tuple[str, ...]:
        """The component columns of a per-component grid, or ``()`` for one axis."""
        return tuple(axis[0] for axis in self.treatment if isinstance(axis, tuple))

    @property
    def treatment_points(self) -> tuple[float | tuple[float, ...], ...]:
        """Every treatment strength the surface evaluates, in grid order.

        One float per value of a one-axis grid.  A per-component grid gives one tuple per
        combination, in the order of :attr:`components`.
        """
        if not self.components:
            return tuple(float(value) for value in self.treatment)  # type: ignore[arg-type]
        axes = [axis[1] for axis in self.treatment if isinstance(axis, tuple)]
        return tuple(tuple(point) for point in product(*axes))


@dataclass(frozen=True)
class ObservedConfounderCalibration:
    """Record one model-dependent observed-covariate calibration.

    Parameters
    ----------
    covariate : str
        Numeric adjustment covariate used for the comparison.
    role : {"treatment", "outcome"}
        Perturbed variable whose source-style scale is reported.
    family : {"binomial", "gaussian"}
        Variable family that determines the calibration rule.
    strength : float
        Prediction-change fraction or signed standardized marginal coefficient.
    method : str
        Finite-sample rule used to calculate ``strength``.
    component : str or None
        The component of a vector treatment that a treatment row calibrates, or ``None``.
    """

    covariate: str
    role: Literal["treatment", "outcome"]
    family: Literal["binomial", "gaussian"]
    strength: float
    method: str
    component: str | None = None


@dataclass(frozen=True)
class SimulatedConfoundingCell:
    """Store one strength pair's estimate or structured failure.

    Parameters
    ----------
    treatment_strength : float or tuple of float
        Declared binary flip probability or continuous linear coefficient.  A vector
        treatment has one per component, in the grid's component order.
    outcome_strength : float
        Declared outcome perturbation strength.
    estimate : float or None
        Refitted estimate. ``None`` when the cell failed.
    displacement : float or None
        Refitted minus original value on the surface's movement scale. ``None`` when
        the cell failed.
    induced_treatment_association : float, tuple or None
        Pearson correlation between the shared latent vector and the treatment of this
        cell, one per component of a vector treatment. It is measured within the selected
        baseline population under the fixed row weights.
        Both treatment arms contribute even for ATT or ATC. The anchor cell reports the
        original treatment, which gives the null level of the same data. Every other cell
        reports the perturbed treatment. The value is ``None`` when that treatment has
        zero standard deviation, and when a cell failed before the surface built it.
    failure : ReplicationFailure or None
        Structured refit or replacement failure retained for this cell.
    target_population_fraction : float or None
        Weighted fraction of the selected baseline population in the conditioning arm.
        ATT and ATC recompute this fraction after treatment replacement. Other targets
        report one. ``None`` means the perturbation failed before the fraction was computed.
    """

    treatment_strength: float | tuple[float, ...]
    outcome_strength: float
    estimate: float | None
    displacement: float | None
    induced_treatment_association: float | tuple[float | None, ...] | None = None
    failure: ReplicationFailure | None = None
    target_population_fraction: float | None = None


@dataclass(frozen=True)
class SimulatedConfoundingResult:
    r"""Store a qualitative simulated-confounding stress surface.

    Parameters
    ----------
    estimand : str
        Binary-treatment ATE, ATT, ATC, counterfactual-mean, risk-ratio, odds-ratio, or
        population-attributable-contrast alias, named modified-policy mean alias, or named
        modified-policy contrast alias.
    original_estimate : float
        Point estimate from the unperturbed fitted result.
    movement_scale : {"estimate_difference", "log_ratio"}
        Scale used for the signed displacement. Ratio estimates use their stored log
        value. Other estimates use their reported value.
    grid : ConfounderStrengthGrid
        Explicit strength pairs evaluated by the surface.
    cells : tuple of SimulatedConfoundingCell
        Successful estimates and retained failures in grid order.
    calibrations : tuple of ObservedConfounderCalibration
        Optional source-style comparisons against numeric observed covariates.
    root_seed : int
        Seed resolved for the complete operation.
    latent_seed : int
        Tagged child seed used to draw the shared latent vector.
    refit_seed : int
        Seed reused by every estimator refit. It equals ``root_seed``. A refit reuses the
        repeat seed sequence of the original fit only when this seed equals the seed of
        that fit. ``random_state=None`` resolves to the seed of the fit when the fit
        declared one. An unseeded fit, or an explicit ``random_state`` other than the seed
        of the fit, gives the refits a different sequence, so movement near the anchor can
        carry a fold artifact. Perturbing a stratification variable can also change the
        realised folds.
    n_repeats : int
        Number of complete cross-fitting draws in the original and refitted estimates.
    repeat_aggregation : {"coordinatewise_median"}
        Estimator-owned rule that names how the estimator aggregates the complete
        cross-fitting draws when there is more than one draw. The estimator applies no
        aggregation to a single draw.
    treatment_family : str
        Treatment family covered by the perturbation law.
    outcome_family : str
        Original outcome family.
    treatment_law : str
        Human-readable treatment perturbation rule.
    outcome_law : str
        Human-readable outcome perturbation rule.
    weight_report : WeightReport
        Provenance for the fixed empirical row-mass vector used by every cell.
    backend : str or None
        Dataframe backend used by frame methods.
    stratum : tuple or None
        Selected baseline stratum values. ``None`` selects the full baseline population.
    strata_names : tuple of str
        Names of the columns defining the selected baseline stratum.
    population : {"baseline", "perturbed_treatment_group"}
        Whether the target averages over its baseline population or its cell's observed
        treatment group. ATT and ATC use the latter.
    conditioning_arm : Any or None
        Original treatment label defining the ATT or ATC population.
    treatment_components : tuple of str
        The component columns of a vector treatment, in the grid's order, or ``()``.  Each
        cell's ``treatment_strength`` and ``induced_treatment_association`` then hold one
        value per component in this order.
    arm_pair : tuple or None
        The two arm labels ``(a, b)`` that the pairwise swap of a multi-arm treatment
        exchanges, or ``None`` on every other treatment.  Each cell's
        ``induced_treatment_association`` then reads the indicator of arm ``a`` on the rows
        in arm ``a`` or ``b``.
    latent_level : {"row", "cluster", "mixed"}
        The declared level of the latent cause.  ``"cluster"`` draws one value per
        cluster.  ``"mixed"`` draws ``sqrt(rho) U_cluster + sqrt(1 - rho) e_row``, which
        keeps each row's latent standard normal.
    latent_rho : float or None
        The intra-cluster correlation of a ``"mixed"`` latent, else ``None``.
    n_clusters : int or None
        The cluster count of a clustered fit, else ``None``.
    weights_estimated : bool
        Whether the fit declared its observation weights estimated.  Every cell holds them
        fixed at their supplied values, so each cell targets the fit's weighted population.
    known_mechanism_carried : bool
        Whether the fitted data declared a known treatment mechanism.  The flip of
        strength :math:`s` is drawn from a latent normal independent of :math:`W`, so the
        flipped treatment's mechanism is known too:
        :math:`g_s(a \mid W) = (1 - s)\, g_0(a \mid W) + s\, g_0(1 - a \mid W)`.  Every
        cell that moves the treatment declares :math:`g_s`, and its refit divides by it.

    Attributes
    ----------
    target_measure : {"unweighted", "fixed_empirical_tilt"}
    association_population : {"selected_baseline_stratum", "full_fitted_population"}
    calibration_population : {"full_fitted_population"}
    refit_population : {"full_fitted_population"}

    See Also
    --------
    simulated_confounding : Build this result from a fitted estimator.
    """

    estimand: str
    original_estimate: float
    movement_scale: Literal["estimate_difference", "log_ratio"]
    grid: ConfounderStrengthGrid
    cells: tuple[SimulatedConfoundingCell, ...]
    calibrations: tuple[ObservedConfounderCalibration, ...]
    root_seed: int
    latent_seed: int
    refit_seed: int
    n_repeats: int
    repeat_aggregation: Literal["coordinatewise_median"]
    treatment_family: str
    outcome_family: str
    treatment_law: str
    outcome_law: str
    weight_report: WeightReport
    backend: str | None = None
    stratum: tuple[Any, ...] | None = None
    strata_names: tuple[str, ...] = ()
    population: Literal["baseline", "perturbed_treatment_group"] = "baseline"
    conditioning_arm: Any = None
    known_mechanism_carried: bool = False
    treatment_components: tuple[str, ...] = ()
    arm_pair: tuple[Any, Any] | None = None
    latent_level: Literal["row", "cluster", "mixed"] = "row"
    latent_rho: float | None = None
    n_clusters: int | None = None
    weights_estimated: bool = False

    @property
    def target_measure(self) -> Literal["unweighted", "fixed_empirical_tilt"]:
        """Name the empirical measure on which every cell is evaluated."""
        return "fixed_empirical_tilt" if self.weight_report.name is not None else "unweighted"

    @property
    def association_population(
        self,
    ) -> Literal["selected_baseline_stratum", "full_fitted_population"]:
        """Name the baseline rows used for the induced treatment association."""
        return "full_fitted_population" if self.stratum is None else "selected_baseline_stratum"

    @property
    def calibration_population(self) -> Literal["full_fitted_population"]:
        """Name the original rows used for observed-covariate calibration."""
        return "full_fitted_population"

    @property
    def refit_population(self) -> Literal["full_fitted_population"]:
        """Name the rows perturbed together before every complete estimator refit."""
        return "full_fitted_population"

    def population_lines(self) -> tuple[str, ...]:
        """Return the population provenance of this surface, one field to a line.

        The single field list and label vocabulary that :meth:`summary` prints and that
        the post-fit assessment row joins with ``"; "``. Written twice the two reports
        drift, and a reader cannot tell which population a movement belongs to.

        Returns
        -------
        tuple of str
            One ``label: value`` line for each population field, in a fixed order.
        """
        return (
            (
                f"target population: {self.population}",
                f"conditioning arm: {self.conditioning_arm!r}",
                f"baseline stratum: {self.stratum!r}",
                f"strata columns: {self.strata_names!r}",
                f"association population: {self.association_population}",
                f"calibration population: {self.calibration_population}",
                f"refit population: {self.refit_population}",
            )
            + (
                (
                    "treatment mechanism: known on the data; each cell that flips the treatment "
                    "declares the flipped mechanism (1 - s) g0 + s (1 - g0)"
                    if self.arm_pair is None
                    else "treatment mechanism: known on the data; each cell that swaps the arms "
                    "declares (1 - s) g_a + s g_b for arm a, (1 - s) g_b + s g_a for arm b, "
                    "and the other arms unchanged",
                )
                if self.known_mechanism_carried
                else ()
            )
            + (
                (
                    f"latent level: {self.latent_level}"
                    + (f" (rho {self.latent_rho:.4g})" if self.latent_rho is not None else "")
                    + f" over {self.n_clusters} clusters",
                )
                if self.n_clusters is not None
                else ()
            )
            + (
                (
                    "observation weights: declared estimated; every cell holds them fixed at "
                    "the supplied values, so each cell targets the fit's weighted population",
                )
                if self.weights_estimated
                else ()
            )
        )

    @property
    def complete(self) -> bool:
        """Return whether every declared cell produced an estimate."""
        return all(cell.failure is None for cell in self.cells)

    @property
    def failures(self) -> tuple[ReplicationFailure, ...]:
        """Return structured failures in grid order."""
        return tuple(cell.failure for cell in self.cells if cell.failure is not None)

    @property
    def successful_cells(self) -> tuple[SimulatedConfoundingCell, ...]:
        """Return cells that produced estimates in grid order."""
        return tuple(cell for cell in self.cells if cell.failure is None)

    def to_frame(self) -> Any:
        """Return one backend-native row per declared strength pair.

        Returns
        -------
        dataframe
            Estimates, the movement scale of the displacement, displacements, induced
            treatment associations, and retained failure details. Two population columns
            join a movement to the rows it belongs to. ``association_population`` repeats
            :attr:`association_population` on every row, which names the baseline rows the
            association was measured on. ``target_population_fraction`` carries
            :attr:`SimulatedConfoundingCell.target_population_fraction`, which is the
            conditioning-arm share that cell targets. A failed cell keeps its fraction,
            because the surface records it before the refit.
        """
        payload: dict[str, list[Any]] = {}
        if self.treatment_components:
            # A vector treatment: one strength and one association column per component.
            for index, column in enumerate(self.treatment_components):
                payload[f"treatment_strength[{column}]"] = [
                    _component(cell.treatment_strength, index) for cell in self.cells
                ]
        else:
            payload["treatment_strength"] = [cell.treatment_strength for cell in self.cells]
        payload.update(
            {
                "outcome_strength": [cell.outcome_strength for cell in self.cells],
                "movement_scale": [self.movement_scale for _ in self.cells],
                "estimate": [cell.estimate for cell in self.cells],
                "displacement": [cell.displacement for cell in self.cells],
            }
        )
        if self.treatment_components:
            for index, column in enumerate(self.treatment_components):
                payload[f"induced_treatment_association[{column}]"] = [
                    _component(cell.induced_treatment_association, index) for cell in self.cells
                ]
        else:
            payload["induced_treatment_association"] = [
                cell.induced_treatment_association for cell in self.cells
            ]
        payload.update(
            {
                "association_population": [self.association_population for _ in self.cells],
                "target_population_fraction": [
                    cell.target_population_fraction for cell in self.cells
                ],
                "error_type": [
                    None if cell.failure is None else cell.failure.error_type for cell in self.cells
                ],
                "message": [
                    None if cell.failure is None else cell.failure.message for cell in self.cells
                ],
            }
        )
        return emit_frame(payload, backend=self.backend)

    def calibration_frame(self) -> Any:
        """Return model-dependent observed-covariate comparisons.

        Returns
        -------
        dataframe
            One row per covariate and perturbed-variable role.
        """
        payload = {
            "covariate": [row.covariate for row in self.calibrations],
            "role": [row.role for row in self.calibrations],
            **(
                {"component": [row.component for row in self.calibrations]}
                if self.treatment_components
                else {}
            ),
            "family": [row.family for row in self.calibrations],
            "strength": [row.strength for row in self.calibrations],
            "method": [row.method for row in self.calibrations],
            "calibration_population": [self.calibration_population for _ in self.calibrations],
        }
        return emit_frame(payload, backend=self.backend)

    def summary(self) -> str:
        """Return a qualitative text summary without an inferential verdict.

        Returns
        -------
        str
            Original estimate, seeds, perturbation laws, cell movements, and the
            induced treatment association of each cell.
        """
        rows = []
        for cell in self.cells:
            reported = _format_values(cell.induced_treatment_association, "+.4f")
            fraction = cell.target_population_fraction
            population_fraction = "n/a" if fraction is None else f"{fraction:.4f}"
            strength = _format_values(cell.treatment_strength, ".4g")
            if cell.failure is None:
                rows.append(
                    [
                        strength,
                        f"{cell.outcome_strength:.4g}",
                        f"{cell.estimate:.5g}",
                        f"{cell.displacement:+.5g}",
                        reported,
                        population_fraction,
                    ]
                )
            else:
                rows.append(
                    [
                        strength,
                        f"{cell.outcome_strength:.4g}",
                        "failed",
                        cell.failure.error_type,
                        reported,
                        population_fraction,
                    ]
                )
        # ``inference.influence.median_estimates`` returns its input unchanged for a single
        # draw, so a one-draw report must not name an aggregation rule that never ran.
        crossfit = (
            f"cross-fitting: {self.n_repeats} draws, aggregation={self.repeat_aggregation}"
            if self.n_repeats > 1
            else "cross-fitting: 1 draw, no repeat aggregation"
        )
        return "\n".join(
            [
                f"Simulated common-cause stress surface for {self.estimand!r}",
                f"original estimate: {self.original_estimate:.5g}",
                (
                    f"seeds: root={self.root_seed}, latent={self.latent_seed}, "
                    f"refit={self.refit_seed}"
                ),
                crossfit,
                f"target measure: {self.target_measure}",
                *self.population_lines(),
                self.treatment_law,
                self.outcome_law,
                *self._calibration_lines(),
                "",
                format_table(
                    [
                        "treatment strength"
                        + (
                            f" ({', '.join(self.treatment_components)})"
                            if self.treatment_components
                            else ""
                        ),
                        "outcome strength",
                        "estimate",
                        f"movement ({self.movement_scale.replace('_', ' ')})",
                        "induced association",
                        "population fraction",
                    ],
                    rows,
                ),
                "",
                "The induced association is the correlation between the latent vector and the "
                "treatment of the cell within the selected baseline population.",
                *(
                    (
                        "ATT and ATC membership follows each cell's perturbed treatment. "
                        "Movement includes this change of population.",
                    )
                    if self.population == "perturbed_treatment_group"
                    else ()
                ),
                *self._reading_guard(),
                "This surface is qualitative. It is not a bound or sensitivity-adjusted inference.",
            ]
        )

    def _calibration_lines(self) -> tuple[str, ...]:
        """Name the calibration rows whose unpenalized fit separated, if any."""
        separated = sum(1 for row in self.calibrations if row.method == SEPARATION_METHOD)
        if not separated:
            return ()
        return (
            f"calibration: {separated} of {len(self.calibrations)} strengths are nan, because "
            "the unpenalized logistic fit separates the classes and has no finite solution",
        )

    def _reading_guard(self) -> tuple[str, ...]:
        """Return the lines that say which cells carry a confounding path."""
        if self.treatment_family == "vector":
            return (
                "Each component's association reports what its perturbation achieved on "
                "these data: a flip for a binary component, the linear shift for a dose. "
                "A cell whose outcome strength is zero has no confounding path, whatever "
                "its associations.",
                "A cell whose treatment strengths are all zero also has no confounding path. "
                "Its movement reports the outcome perturbation alone.",
            )
        if self.treatment_family == "categorical":
            return (
                "The association reads the indicator of arm {!r} on the rows in arms {!r} "
                "and {!r}.".format(self.arm_pair[0], *self.arm_pair)
                if self.arm_pair is not None
                else "",
                "An association near zero says the swap moved the estimate by "
                "misclassification between the two arms and not by confounding.",
                *(
                    (
                        "The estimate can also move because each cell rebuilds its ATT or "
                        "ATC population from the swapped treatment.",
                    )
                    if self.population == "perturbed_treatment_group"
                    else ()
                ),
            )
        if self.treatment_family == "binary":
            if self.population == "perturbed_treatment_group":
                return (
                    "An association near zero says the treatment axis opened no confounding path.",
                    "The estimate can still move, because each cell rebuilds its ATT or "
                    "ATC population from the perturbed treatment.",
                    "Read target_population_fraction beside the movement. It reports the "
                    "share of the selected baseline population this cell conditions on.",
                )
            return (
                "An association near zero says the treatment axis moved the estimate by "
                "misclassification and not by confounding.",
            )
        outcome_axis = (
            "Its movement reports the outcome level shift alone. A policy mean keeps that "
            "level shift, and a contrast removes most of it. A small residual stays, "
            "because each cell refits the outcome regression."
            if self.outcome_family == "gaussian"
            else "Its movement reports the outcome perturbation alone. The tail "
            "perturbation attenuates the fitted outcome regression. A policy mean and a "
            "contrast both move with that attenuation."
        )
        return (
            "The association reports what the continuous linear perturbation achieved on "
            "these data. A cell whose outcome strength is zero has no confounding path, "
            "whatever its association, and reports dose perturbation alone.",
            f"A cell whose treatment strength is zero also has no confounding path. {outcome_axis}",
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _component(value: Any, index: int) -> Any:
    """One component's entry of a per-component cell value, or ``None`` when absent."""
    return None if value is None else value[index]


def _format_values(value: Any, spec: str) -> str:
    """A cell value for the summary table: one number, or one per component."""
    if value is None:
        return "n/a"
    if isinstance(value, tuple):
        return "(" + ", ".join("n/a" if v is None else format(v, spec) for v in value) + ")"
    return format(value, spec)


def _pair_labels(data: Any, request: _ValidatedRequest) -> Any:
    """The labels of the swapped pair, ``a`` first.

    A one-column multi-arm treatment gives ``(a, b)``.  A vector treatment gives one
    ``(component, (a, b))`` entry per component that swaps a pair, in component order.
    """
    if request.component_pairs:
        return tuple(
            (
                name,
                (
                    data.component_levels[data.component_names.index(name)][int(first)],
                    data.component_levels[data.component_names.index(name)][int(second)],
                ),
            )
            for name, (first, second) in request.component_pairs
        )
    assert request.arm_codes is not None
    first, second = request.arm_codes
    return (data.arm_label(first), data.arm_label(second))


def _is_anchor(strength: float | tuple[float, ...], outcome: float) -> bool:
    """Whether a grid point moves nothing: every treatment strength and the outcome are zero."""
    treatment = strength if isinstance(strength, tuple) else (strength,)
    return outcome == 0.0 and all(value == 0.0 for value in treatment)


def _numeric_strengths(values: Any, role: str) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(f"{role} strengths must be a non-empty sequence") from error
    if not raw:
        raise ValueError(f"{role} strengths must not be empty")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw):
        raise TypeError(f"{role} strengths must contain only numeric values")
    strengths = tuple(float(value) for value in raw)
    if not all(np.isfinite(strengths)):
        raise ValueError(f"{role} strengths must be finite")
    if len(set(strengths)) != len(strengths):
        raise ValueError(f"{role} strengths must not contain duplicates")
    return strengths


_LATENT_SEED_TAG = 3


def _target_population_fraction(
    request: _ValidatedRequest,
    treatment: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> float:
    """Measure a cell's treatment-group mass within its fixed baseline population."""
    if request.conditioning_code is None:
        return 1.0
    return arm_share(treatment, weights, request.conditioning_code, mask=request.baseline_mask)


def _cell_diagnostics(
    request: _ValidatedRequest,
    latent: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> tuple[float | None, float]:
    """Measure the two per-cell diagnostics of one treatment vector together.

    The anchor cell and every perturbed cell report the same pair, on the same fixed
    baseline population, and differ only in the treatment vector they are given.

    Parameters
    ----------
    request : _ValidatedRequest
        Validated request holding the fixed baseline population and conditioning arm.
    latent : ndarray
        Shared latent vector drawn for the complete surface.
    treatment : ndarray
        Original or perturbed treatment of one cell.
    weights : ndarray
        Normalized fixed row masses. An unweighted fit supplies a vector of ones.

    Returns
    -------
    tuple of (float or None, float)
        The induced treatment association and the target population fraction.
    """
    return (
        _baseline_treatment_association(request, latent, treatment, weights),
        _target_population_fraction(request, treatment, weights),
    )


def _baseline_treatment_association(
    request: _ValidatedRequest,
    latent: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> float | None:
    """Report the realised correlation between the latent vector and one treatment.

    The same fixed baseline population is used at the anchor and at every perturbed
    cell. For a binary tail flip, the association depends on the treated fraction. A
    balanced design reaches zero association, where the treatment axis moves the
    estimate by misclassification alone. For a continuous linear perturbation, this is
    the achieved association between ``U`` and ``A'``.

    Parameters
    ----------
    request : _ValidatedRequest
        Validated request holding the fixed baseline population.
    latent : ndarray
        Shared latent vector drawn for the complete surface.
    treatment : ndarray
        Original or perturbed treatment of one cell.
    weights : ndarray
        Normalized fixed row masses. An unweighted fit supplies a vector of ones.

    Returns
    -------
    float or None
        Pearson correlation, or ``None`` when the treatment has zero standard deviation.
    """
    mask = request.baseline_mask
    if request.arm_codes is not None:
        # The swap acts on the rows of the two declared arms, and the association reads
        # the indicator of arm ``a`` among them.
        first, second = request.arm_codes
        mask = mask & ((treatment == first) | (treatment == second))
        return _weighted_correlation(
            latent[mask], (treatment[mask] == first).astype(float), weights[mask]
        )
    return _weighted_correlation(latent[mask], treatment[mask], weights[mask])


def _latent_child_seed(root_seed: int) -> int:
    """Derive the latent-draw seed from a tagged child of the root seed.

    The tag keeps this stream apart from every other child stream built on the same
    root. A plain ``SeedSequence(root_seed).spawn(...)`` reproduces the bootstrap
    replicate seeds in :mod:`cleverly.inference.bootstrap`, and the tagged form in
    :func:`cleverly.validation.refute._generated_child_seeds` uses tags one and two.

    Parameters
    ----------
    root_seed : int
        Seed resolved for the complete operation.

    Returns
    -------
    int
        Seed for the shared latent vector.
    """
    sequence = np.random.SeedSequence([root_seed, _LATENT_SEED_TAG])
    return int(sequence.generate_state(1)[0])


def _draw_latent(
    n: int,
    latent_seed: int,
    cluster: np.ndarray[Any, Any] | None,
    level: Literal["row", "cluster", "mixed"],
    rho: float | None,
) -> np.ndarray[Any, Any]:
    """Draw the shared latent vector at its declared level.

    ``"row"`` draws ``n`` standard normals, the stream every unclustered surface has used.
    ``"cluster"`` draws one standard normal per cluster code and gives it to every row of
    the cluster.  ``"mixed"`` draws the cluster values first and the row values second from
    the same generator, and returns ``sqrt(rho) U_c + sqrt(1 - rho) e``, so each row's
    latent stays standard normal and a flip strength stays a flip probability.

    Parameters
    ----------
    n : int
        The row count.
    latent_seed : int
        The tagged child seed of the latent draw.
    cluster : ndarray or None
        The contiguous cluster code of each row, or ``None`` on an unclustered fit.
    level : {"row", "cluster", "mixed"}
        The declared level.
    rho : float or None
        The intra-cluster correlation of a ``"mixed"`` latent.

    Returns
    -------
    ndarray
        One latent value per row.
    """
    generator = np.random.default_rng(latent_seed)
    if level == "row":
        return generator.normal(size=n)
    assert cluster is not None
    codes = np.asarray(cluster, dtype=int)
    shared = generator.normal(size=int(codes.max()) + 1)[codes]
    if level == "cluster":
        return shared
    assert rho is not None
    return np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * generator.normal(size=n)


def _swap_arms(
    values: np.ndarray[Any, Any], mask: np.ndarray[Any, Any], arms: tuple[float, float]
) -> np.ndarray[Any, Any]:
    """Exchange arms ``a`` and ``b`` on the masked rows, and keep every other arm.

    The pairwise swap of a multi-arm treatment: the binary tail flip applied to the two arms
    of the declared pair.  The mask does not read the arm, so the law is the same whichever
    of the two is named first.
    """
    first, second = arms
    swapped = values.copy()
    swapped[mask & (values == first)] = second
    swapped[mask & (values == second)] = first
    return swapped


def swapped_mechanism(
    values: np.ndarray[Any, Any], strength: float, arms: tuple[int, int]
) -> np.ndarray[Any, Any]:
    r"""The known mechanism of a multi-arm treatment after the pairwise swap.

    The swap mask is independent of :math:`W` and :math:`A` with probability :math:`s`, so
    :math:`g_s(a \mid W) = (1 - s)\, g_0(a \mid W) + s\, g_0(b \mid W)`, the same with
    :math:`a` and :math:`b` exchanged, and every other arm keeps :math:`g_0`.

    Parameters
    ----------
    values : ndarray
        The declared ``(n, K)`` mechanism, one column per arm in code order.
    strength : float
        The swap probability :math:`s`.
    arms : tuple of int
        The column indices of the two swapped arms.

    Returns
    -------
    ndarray
        The ``(n, K)`` mechanism of the swapped treatment.
    """
    declared = np.asarray(values, dtype=float)
    first, second = arms
    out = declared.copy()
    out[:, first] = (1.0 - strength) * declared[:, first] + strength * declared[:, second]
    out[:, second] = (1.0 - strength) * declared[:, second] + strength * declared[:, first]
    return out


def _flip_mask(latent: np.ndarray[Any, Any], strength: float) -> np.ndarray[Any, Any]:
    if strength == 0.0:
        return np.zeros(latent.shape, dtype=bool)
    threshold = -NormalDist().inv_cdf(strength)
    return latent >= threshold


def _flip_binary(values: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.where(mask, 1.0 - values, values)


def _linear_treatment(
    values: np.ndarray[Any, Any], latent: np.ndarray[Any, Any], strength: float
) -> np.ndarray[Any, Any]:
    """Apply the source-backed continuous treatment perturbation ``A' = A + k_A U``."""
    return values + strength * latent


def flipped_mechanism(values: np.ndarray[Any, Any], strength: float) -> np.ndarray[Any, Any]:
    r"""The known mechanism of a binary treatment flipped with probability ``strength``.

    The flip mask is :math:`1\{U \ge -\Phi^{-1}(s)\}` for a standard normal :math:`U` drawn
    independently of :math:`W` and :math:`A`, so :math:`P(\text{flip} \mid W, A) = s` and
    :math:`g_s(a \mid W) = (1 - s)\, g_0(a \mid W) + s\, g_0(1 - a \mid W)`.

    Parameters
    ----------
    values : ndarray
        The declared ``(n, 2)`` mechanism, one column per arm in code order.
    strength : float
        The flip probability :math:`s`.

    Returns
    -------
    ndarray
        The ``(n, 2)`` mechanism of the flipped treatment.
    """
    declared = np.asarray(values, dtype=float)
    return (1.0 - strength) * declared + strength * declared[:, ::-1]


def _perturb_treatment(
    values: np.ndarray[Any, Any],
    latent: np.ndarray[Any, Any],
    strength: float,
    family: Literal["binary", "categorical", "continuous"],
    arms: tuple[float, float] | None = None,
) -> np.ndarray[Any, Any]:
    if family == "continuous":
        return _linear_treatment(values, latent, strength)
    if family == "categorical":
        assert arms is not None
        return _swap_arms(values, _flip_mask(latent, strength), arms)
    return _flip_binary(values, _flip_mask(latent, strength))


def _perturb_components(
    data: Any,
    latent: np.ndarray[Any, Any],
    strengths: Mapping[str, float],
    pairs: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray[Any, Any]:
    """Perturb each component of a vector treatment through the one shared latent vector.

    A binary component is flipped in the upper latent tail, a categorical component with
    more than two levels swaps its declared pair of levels in that tail, and a dose takes
    ``D' = D + k_D U``, the laws of a one-column treatment.  Every component reads the same
    ``U``, so one simulated common cause reaches every component and the outcome.

    Parameters
    ----------
    data : CausalData
        The fitted data, with a vector treatment.
    latent : ndarray
        The shared latent vector.
    strengths : mapping of str to float
        Each component column's strength.
    pairs : mapping of str to tuple of float, or None
        The level codes ``(a, b)`` that each categorical component with more than two
        levels swaps.

    Returns
    -------
    ndarray
        The ``(n, d)`` perturbed component values, in the data's component order.
    """
    pairs = pairs or {}
    values = np.asarray(data.treatment_components, dtype=float).copy()
    for index, (name, levels) in enumerate(
        zip(data.component_names, data.component_levels, strict=True)
    ):
        family: Literal["binary", "categorical", "continuous"] = (
            "continuous" if not levels else "categorical" if name in pairs else "binary"
        )
        values[:, index] = _perturb_treatment(
            values[:, index], latent, strengths[name], family, pairs.get(name)
        )
    return values


def _component_associations(
    request: _ValidatedRequest,
    latent: np.ndarray[Any, Any],
    data: Any,
    values: np.ndarray[Any, Any],
    columns: tuple[str, ...],
) -> tuple[float | None, ...]:
    """The induced association of each component, in ``columns`` order.

    A component that swaps a declared pair reads the indicator of its arm ``a`` on the
    rows at level ``a`` or ``b``, as a one-column multi-arm treatment does.
    """
    pairs = dict(request.component_pairs)
    weights = data.weights
    out: list[float | None] = []
    for name in columns:
        column = values[:, data.component_names.index(name)]
        mask = request.baseline_mask
        if name in pairs:
            first, second = pairs[name]
            mask = mask & ((column == first) | (column == second))
            column = (column == first).astype(float)
        out.append(_weighted_correlation(latent[mask], column[mask], weights[mask]))
    return tuple(out)


def _weights_are_constant(weights: np.ndarray[Any, Any]) -> bool:
    """Return whether weighted calculations must preserve the old arithmetic path."""
    return bool(np.allclose(weights, 1.0))


def _weighted_std(values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> float:
    """Return a population standard deviation under the normalized empirical weights."""
    if _weights_are_constant(weights):
        return float(np.std(values))
    mean = float(np.average(values, weights=weights))
    return float(np.sqrt(np.average(np.square(values - mean), weights=weights)))


def _is_constant_under_weights(values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> bool:
    """Report whether a covariate takes one value on every row the weights keep.

    A zero weight is legal, and a row that carries no mass calibrates nothing. The
    comparison of distinct values on the positive-weight support is exact, where
    :func:`_weighted_std` leaves a floating-point residual in place of zero.

    Parameters
    ----------
    values : ndarray
        One numeric adjustment column of the analysis data.
    weights : ndarray
        Normalized fixed row masses.

    Returns
    -------
    bool
        ``True`` when the positive-weight rows carry at most one distinct value.
    """
    supported = values[weights > 0.0]
    return bool(supported.size == 0 or np.unique(supported).size <= 1)


def _weighted_correlation(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> float | None:
    """Return correlation under the empirical weight tilt, or ``None`` if undefined."""
    if _weights_are_constant(weights):
        if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])
    left_mean = float(np.average(left, weights=weights))
    right_mean = float(np.average(right, weights=weights))
    left_centered = left - left_mean
    right_centered = right - right_mean
    left_variance = float(np.average(np.square(left_centered), weights=weights))
    right_variance = float(np.average(np.square(right_centered), weights=weights))
    if left_variance == 0.0 or right_variance == 0.0:
        return None
    covariance = float(np.average(left_centered * right_centered, weights=weights))
    correlation = covariance / np.sqrt(left_variance * right_variance)
    return float(np.clip(correlation, -1.0, 1.0))


def _gaussian_outcome(
    values: np.ndarray[Any, Any], latent: np.ndarray[Any, Any], strength: float
) -> np.ndarray[Any, Any]:
    return values - strength * latent


#: The method text of a calibration whose unpenalized logistic fit has no finite solution.
SEPARATION_METHOD = "no finite unpenalized logistic fit (separation); strength not defined"


def _binary_calibration(
    design: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    block: list[int],
    weights: np.ndarray[Any, Any],
) -> tuple[float, str]:
    """Measure the class-prediction change of removing one covariate block.

    The fit is an unpenalized logistic regression (multinomial on more than two classes)
    of ``target`` on the standardized design.  The strength is the weighted share of rows
    whose predicted class changes when every column of ``block`` is set to its weighted
    mean, which is zero after standardizing.  An unpenalized fit is invariant to the
    parametrization of a categorical covariate, so its block strength does not depend on
    the dropped reference level (hard-stop review F10).  An L2 penalty is not invariant.

    A fit that classifies every row correctly has completely separated classes, so the
    maximum-likelihood fit does not exist.  The strength is then ``nan`` with
    :data:`SEPARATION_METHOD`.

    Parameters
    ----------
    design : ndarray
        The standardized adjustment design.
    target : ndarray
        The treatment codes or the binary outcome.
    block : list of int
        The design columns of the benchmarked covariate.
    weights : ndarray
        Normalized fixed row masses.

    Returns
    -------
    tuple of (float, str)
        The strength and the method text.
    """
    model = LogisticRegression(C=np.inf, max_iter=1000)
    constant = _weights_are_constant(weights)
    if constant:
        model.fit(design, target)
    else:
        model.fit(design, target, sample_weight=weights)
    baseline = model.predict(design)
    classes = np.unique(target).size
    method = (
        "unpenalized logistic class-prediction change fraction"
        if classes <= 2
        else "unpenalized multinomial class-prediction change fraction"
    )
    if bool(np.all(baseline == target)) or int(np.max(model.n_iter_)) >= model.max_iter:
        return float("nan"), SEPARATION_METHOD
    removed = design.copy()
    removed[:, block] = 0.0
    changed = model.predict(removed) != baseline
    if constant:
        return float(np.mean(changed)), method
    return float(np.average(changed, weights=weights)), method


def _continuous_calibration(
    covariate: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> float:
    correlation = _weighted_correlation(covariate, target, weights)
    if correlation is None:
        return float("nan")
    return correlation * _weighted_std(target, weights)


#: The method text of the Gaussian calibration of one numeric covariate.
_NUMERIC_GAUSSIAN_METHOD = "signed standardized marginal coefficient"
#: The method text of the Gaussian calibration of one categorical covariate's block.
_BLOCK_GAUSSIAN_METHOD = "block multiple correlation times target SD (unsigned)"


def _block_continuous_calibration(
    columns: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> float:
    """``R * sd(target)`` for a block, with ``R`` the multiple correlation of the block.

    ``R`` is the square root of the weighted coefficient of determination of an
    unpenalized least-squares fit of ``target`` on the block with an intercept.  A block
    has no sign, so the strength is unsigned.  For a block of one column ``R`` is the
    absolute correlation, so the strength is the absolute numeric strength.
    """
    w = weights / np.sum(weights)
    design = np.column_stack((np.ones(target.size), columns))
    root = np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    fitted = design @ coefficients
    mean = float(np.sum(w * target))
    total = float(np.sum(w * np.square(target - mean)))
    if total == 0.0:
        return float("nan")
    explained = float(np.sum(w * np.square(fitted - mean)))
    return float(np.sqrt(min(max(explained / total, 0.0), 1.0))) * _weighted_std(target, weights)


def _block_columns(data: Any, name: str) -> list[int]:
    """The design columns of one benchmark covariate: itself, or a categorical's indicators."""
    encoding = next((item for item in data.encodings if item.column == name), None)
    if encoding is None:
        return [data.covariate_names.index(name)]
    return [
        data.covariate_names.index(column)
        for column in encoding.generated
        if column in data.covariate_names
    ]


def _calibration_row(
    data: Any,
    design: np.ndarray[Any, Any],
    name: str,
    block: list[int],
    target: np.ndarray[Any, Any],
    binary: bool,
    role: Literal["treatment", "outcome"],
    component: str | None = None,
) -> ObservedConfounderCalibration:
    """One calibration row of one covariate against one perturbed variable."""
    categorical = any(encoding.column == name for encoding in data.encodings)
    if binary:
        strength, method = _binary_calibration(design, target, block, data.weights)
        family: Literal["binomial", "gaussian"] = "binomial"
    elif categorical:
        strength = _block_continuous_calibration(data.covariates[:, block], target, data.weights)
        method = _BLOCK_GAUSSIAN_METHOD
        family = "gaussian"
    else:
        strength = _continuous_calibration(data.covariates[:, block[0]], target, data.weights)
        method = _NUMERIC_GAUSSIAN_METHOD
        family = "gaussian"
    return ObservedConfounderCalibration(
        covariate=name,
        role=role,
        family=family,
        strength=strength,
        method=method,
        component=component,
    )


def _calibrate(result: Any, names: tuple[str, ...]) -> tuple[ObservedConfounderCalibration, ...]:
    if not names:
        return ()
    data = result.data
    if _weights_are_constant(data.weights):
        design = StandardScaler().fit_transform(data.covariates)
    else:
        scaler = StandardScaler().fit(data.covariates, sample_weight=data.weights)
        design = scaler.transform(data.covariates)
    rows: list[ObservedConfounderCalibration] = []
    for name in names:
        block = _block_columns(data, name)
        if data.treatment_kind == "vector":
            # One treatment row per component, by that component's law.
            for position, (column, levels) in enumerate(
                zip(data.component_names, data.component_levels, strict=True)
            ):
                rows.append(
                    _calibration_row(
                        data,
                        design,
                        name,
                        block,
                        data.treatment_components[:, position],
                        bool(levels),
                        "treatment",
                        component=column,
                    )
                )
        else:
            rows.append(
                _calibration_row(
                    data,
                    design,
                    name,
                    block,
                    data.treatment,
                    not data.is_continuous_treatment,
                    "treatment",
                )
            )
        rows.append(
            _calibration_row(
                data, design, name, block, data.outcome, data.family == "binomial", "outcome"
            )
        )
    return tuple(rows)


def simulated_confounding(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: tuple[str, ...] = (),
    random_state: int | None = None,
    arm_pair: tuple[Any, Any] | None = None,
    latent_level: Literal["row", "cluster", "mixed"] | None = None,
    latent_rho: float | None = None,
) -> SimulatedConfoundingResult:
    """Refit a selected parameter across a simulated common-cause strength grid.

    Parameters
    ----------
    result : TMLEResult
        Replayable backdoor-identified binary-treatment ATE, ATT, ATC, counterfactual
        mean, risk ratio, odds ratio, population attributable contrast, or continuous
        modified-policy fit. Ordinary binary TMLE also supports fixed regime means,
        regime contrasts, and incremental means and contrasts. Ordinary TMLE also supports
        binary and continuous MSM coefficients under built-in links.
        Baseline strata require estimator support for their targeting.
    estimand : str
        Parameter alias to report. The free function needs an explicit ``ey1``, ``ey0``,
        or ``ey[...]`` alias for a binary counterfactual mean. Binary ratio fits use
        ``rr`` or ``or``. Population attributable contrasts use ``par`` or ``paf``.
        A continuous fit requires an explicit ``ey_policy[...]`` alias
        of a nonzero-delta policy, an ``ate_policy[...]`` alias, or an ``msm[...]`` coefficient.
        Fixed regimes use ``ey_regime[...]`` or ``ate_regime[...]``.
        An MSM uses ``msm[term]``. Incremental targets use ``ey_ipsi[...]`` or
        ``ate_ipsi[...]``; a multiplier-one mean is refused.
        A conditional target requires its complete reported stratum alias.
    grid : ConfounderStrengthGrid
        Explicit treatment and outcome perturbation strengths.
    benchmark_covariates : tuple of str
        Numeric observed covariates for optional calibration on the full original population.
    random_state : int or None
        Root seed. ``None`` uses the fitted estimator's seed or draws a recorded seed.
    arm_pair : tuple or None
        The two arm labels ``(a, b)`` that the pairwise swap of a multi-arm treatment
        exchanges.  A two-arm contrast such as ``ate[a vs b]`` declares its pair, so the
        argument is needed only for another parameter of a multi-arm fit, such as an arm
        mean.  ``a`` names the arm whose indicator the induced association reads.
    latent_level : {"row", "cluster", "mixed"} or None
        The level of the latent cause.  A clustered fit must declare it.  ``None`` on an
        unclustered fit means ``"row"``.
    latent_rho : float or None
        The intra-cluster correlation of a ``"mixed"`` latent, strictly between 0 and 1.

    Returns
    -------
    SimulatedConfoundingResult
        Estimate movements on the scale ``movement_scale`` names, the induced treatment
        association of each cell, fixed-weight provenance, and retained cell failures.
        The result has no verdict or sensitivity-adjusted inference.

    See Also
    --------
    ConfounderStrengthGrid : The explicit strength declaration.
    cleverly.sensitivity.omitted_variable_bounds : A non-refit bias-bound analysis.

    Notes
    -----
    Every non-anchor cell refits under the resolved root seed, and the zero-strength anchor
    is the original fit itself. The root seed does not promise identical realised folds, for
    two reasons. First, a refit reuses the repeat seed sequence of the original fit only when
    the root seed equals the seed of that fit. ``random_state=None`` resolves to the seed of
    the fit when the fit declared one. An unseeded fit, or an explicit ``random_state`` other
    than the seed of the fit, gives every non-anchor cell a different sequence, so movement
    near the anchor can carry a fold artifact. Second, treatment-stratified or
    outcome-stratified splitting can change assignments after the surface perturbs that
    variable.

    An ordinary-TMLE fit can declare fixed probability weights. Binary complete-outcome
    collaborative-TMLE and DR-TMLE fits can also declare them. Every replacement and refit
    keeps the normalized weight on its original row. The induced association conditions
    this empirical law on the selected baseline stratum. Numeric calibration uses the full
    original population. Longitudinal, multi-arm, missing-outcome, intermediate,
    estimated-weight, and clustered fits are refused before calibration, a draw, or a
    refit. Execution and the assessment capability share the same ordered reasons.

    All strata share the same full-row latent draw and complete estimator refit. Baseline
    stratum membership stays fixed. ATT and ATC instead recompute their observed-treatment
    membership from each cell's replaced treatment, within the selected baseline population.
    Their movement therefore includes composition changes. Their induced association still
    includes both treatment arms, because treatment is constant within one treatment group.
    Each cell records that group's weighted fraction of its baseline population.

    PAR cells recompute the observed outcome mean minus the reference counterfactual mean.
    PAF cells recompute one minus the reference mean divided by the observed outcome risk.
    Both use estimate differences for movement. PAF keeps negative fractions without a
    log transform or clipping. A zero observed risk leaves a retained failed PAF cell.
    PAR and PAF support ordinary TMLE only. The identified effect's method catalog
    evidences no collaborative score and no reduced-dimension correction for these
    observed-law contrasts, so it offers neither C-TMLE nor DR-TMLE for them.

    Binary ordinary TMLE supports fixed Static, Rule, and Stochastic regimes.
    The operation validates declared policy arrays against every stored cross-fitting
    draw and freezes those arrays on the original baseline rows.
    MSMs support built-in identity, log, and logit links for binary and continuous
    treatments. Grid designs and raw projection weights stay fixed. Continuous MSMs
    reevaluate the declared deterministic functions at each perturbed observed dose;
    the estimator rebuilds observed weights and the integration-support mask.
    Coefficient movement is a difference on the stored coefficient scale, including
    log and logit coefficients. The operation does not exponentiate that difference.

    Incremental means and contrasts keep their odds multipliers, names, and reference.
    Complete refits rebuild intervention densities from the refitted treatment mechanism
    and retain its targeting contribution. Movement includes this intervention change.
    A multiplier-one mean is the natural course and is refused before a draw; contrasts
    against that reference remain supported. Custom intervention types and MSM links
    remain refused. All these extensions use exact ordinary TMLE only. Incremental
    targets and nonlinear or continuous MSMs refuse baseline strata upstream.

    Each cell reports ``induced_treatment_association``. It is the correlation between the
    shared latent vector and the treatment of that cell. For binary treatment, the flip is
    non-differential misclassification. Its induced association depends on the treated
    fraction. A balanced design reports an association near zero. For continuous
    treatment, the operation applies ``A' = A + k_A U`` and reports the correlation that
    linear perturbation achieves on the analysis data. That association grows with the
    treatment strength by construction, so it is not on its own evidence of confounding.
    A confounding path also needs the latent vector in the outcome, which happens only at
    a nonzero outcome strength. A cell whose outcome strength is zero therefore has no
    confounding path, whatever its association, and its movement reports dose perturbation
    alone.

    The zero treatment-strength column carries no confounding path either, because the
    latent vector never reaches the treatment there. For a Gaussian outcome that column
    reports the level shift ``Y' = Y - k_Y U`` alone. The level shift largely cancels in
    an ``ate_policy`` contrast. A policy mean keeps it, so read the zero treatment-strength
    column of an ``ey_policy`` surface as an artifact of the outcome law.

    A zero-delta shift is the natural course, its policy mean is ``E[Y]``, and it has no
    counterfactual treatment dependence. ``simulated_confounding`` refuses that mean before
    it draws the latent vector. It still accepts an ``ate_policy`` contrast that uses the
    natural course as its reference. The same refusal applies to ``NaturalCourseMean``.
    PAR and PAF retain counterfactual treatment dependence through their reference intervention.
    """
    if type(grid) is not ConfounderStrengthGrid:
        raise TypeError("grid must be an exact ConfounderStrengthGrid declaration")
    request = _validate_request(
        result,
        estimand,
        grid,
        benchmark_covariates,
        arm_pair=arm_pair,
        latent_level=latent_level,
        latent_rho=latent_rho,
    )
    calibrations = _calibrate(result, request.calibration_names)
    root_seed = resolve_assessment_seed(result, random_state)
    # Every non-anchor cell refits under the root seed, not a spawned child. Same convention
    # as ``cleverly.validation.refute`` and ``cleverly.sensitivity.omitted_variable``. This
    # does not freeze the realised folds. ``TMLE.refit`` reuses the estimator's own repeat
    # seed sequence only when the seed it receives equals ``estimator.random_state``, so an
    # unseeded fit or an explicit ``random_state`` other than the seed of the fit displaces
    # every non-anchor cell by pure fold noise. A perturbed treatment or outcome that supplies
    # a stratification variable moves the folds as well. The latent draw keeps its own tagged
    # child seed, so it stays independent of the splits.
    refit_seed = root_seed
    latent_seed = _latent_child_seed(root_seed)
    latent = _draw_latent(
        result.data.n,
        latent_seed,
        result.data.cluster,
        request.latent_level,
        request.latent_rho,
    )
    original_parameter = result[estimand]
    original = float(original_parameter.psi)
    original_inference = original_parameter.inference_value
    cells: list[SimulatedConfoundingCell] = []

    data = result.data
    columns = grid.components

    def diagnostics(values: np.ndarray[Any, Any]) -> tuple[Any, float]:
        """The cell's association and population fraction, per component on a vector."""
        if columns:
            return _component_associations(request, latent, data, values, columns), 1.0
        return _cell_diagnostics(request, latent, values, data.weights)

    observed = np.asarray(data.treatment_components if columns else data.treatment, dtype=float)
    for cell_index, (treatment_strength, outcome_strength) in enumerate(
        product(grid.treatment_points, grid.outcome)
    ):
        if _is_anchor(treatment_strength, outcome_strength):
            anchor_association, anchor_fraction = diagnostics(observed)
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    estimate=original,
                    displacement=0.0,
                    induced_treatment_association=anchor_association,
                    target_population_fraction=anchor_fraction,
                )
            )
            continue
        # A cell reports the association of the treatment the surface built for it, even
        # when the cell later fails.  An arm-loss cell reaches the zero-variance guard and
        # reports ``None``, because a constant treatment has no correlation.
        association: Any = None
        population_fraction: float | None = None
        try:
            if isinstance(treatment_strength, tuple):
                treatment = _perturb_components(
                    data,
                    latent,
                    dict(zip(columns, treatment_strength, strict=True)),
                    dict(request.component_pairs),
                )
            else:
                treatment = _perturb_treatment(
                    data.treatment,
                    latent,
                    treatment_strength,
                    request.treatment_family,  # type: ignore[arg-type]
                    request.arm_codes,
                )
            association, population_fraction = diagnostics(treatment)
            replacement = result.data.with_treatment(treatment)
            if (
                result.data.known_treatment is not None
                and replacement.known_treatment is None
                and not isinstance(treatment_strength, tuple)
            ):
                # A zero-strength axis keeps the treatment, and with it the declaration.
                declared = result.data.known_treatment.values
                replacement = replacement.with_known_mechanism(
                    flipped_mechanism(declared, treatment_strength)
                    if request.arm_codes is None
                    else swapped_mechanism(
                        declared,
                        treatment_strength,
                        (int(request.arm_codes[0]), int(request.arm_codes[1])),
                    )
                )
            if result.data.family == "gaussian":
                outcome = _gaussian_outcome(result.data.outcome, latent, outcome_strength)
            else:
                outcome = _flip_binary(result.data.outcome, _flip_mask(latent, outcome_strength))
            replacement = replacement.with_outcome(
                outcome,
                family=result.data.family,
                name="simulated-confounding outcome",
            )
            refitted = request.estimator.refit(replacement, random_state=refit_seed)
            # ``inference.influence.median_estimates`` drops a name that is absent from any
            # draw, so ``refitted[estimand]`` would raise "was not requested" for an estimand
            # this cell did request.  Retain the true reason instead.
            if result.n_repeats > 1 and estimand not in refitted.estimates:
                raise ValueError(
                    f"{estimand!r} is missing from at least one of the {result.n_repeats} "
                    "cross-fitting draws of this cell, so the median report omits it"
                )
            refitted_parameter = refitted[estimand]
            estimate = float(refitted_parameter.psi)
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    estimate=estimate,
                    displacement=refitted_parameter.inference_value - original_inference,
                    induced_treatment_association=association,
                    target_population_fraction=population_fraction,
                )
            )
        except Exception as error:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    estimate=None,
                    displacement=None,
                    induced_treatment_association=association,
                    target_population_fraction=population_fraction,
                    failure=ReplicationFailure(
                        replicate=cell_index,
                        seed=root_seed,
                        error_type=type(error).__name__,
                        message=str(error),
                    ),
                )
            )

    return SimulatedConfoundingResult(
        estimand=estimand,
        original_estimate=original,
        movement_scale=request.movement_scale,
        grid=grid,
        cells=tuple(cells),
        calibrations=calibrations,
        root_seed=root_seed,
        latent_seed=latent_seed,
        refit_seed=refit_seed,
        n_repeats=result.n_repeats,
        repeat_aggregation="coordinatewise_median",
        treatment_family=request.treatment_family,
        outcome_family=result.data.family,
        treatment_law=(
            "Treatment is flipped in the declared upper latent-normal tail."
            if request.treatment_family == "binary"
            else "Arms {!r} and {!r} are exchanged in the declared upper latent-normal tail; "
            "every other arm keeps its rows.".format(*_pair_labels(result.data, request))
            if request.treatment_family == "categorical"
            else "Each binary component is flipped in the declared upper latent-normal tail, "
            "and each dose adds signed strength times the shared latent value. Every "
            "component reads the same latent value."
            if request.treatment_family == "vector"
            else "Continuous treatment adds signed strength times the shared latent value."
        ),
        outcome_law=(
            "Gaussian outcome subtracts signed strength times the shared latent value."
            if result.data.family == "gaussian"
            else "Binomial outcome is flipped in the declared upper latent-normal tail."
        ),
        weight_report=result.data.weight_report(),
        known_mechanism_carried=result.data.known_treatment is not None,
        treatment_components=columns,
        arm_pair=(
            None
            if request.arm_codes is None and not request.component_pairs
            else _pair_labels(result.data, request)
        ),
        latent_level=request.latent_level,
        latent_rho=request.latent_rho,
        n_clusters=(
            None if result.data.cluster is None else int(np.unique(result.data.cluster).size)
        ),
        weights_estimated=result.data.declares_estimated_weights,
        backend=result.data.backend,
        stratum=request.stratum,
        strata_names=() if request.stratum is None else result.data.strata_names,
        population="baseline" if request.conditioning_code is None else "perturbed_treatment_group",
        conditioning_arm=(
            None
            if request.conditioning_code is None
            else result.data.arm_label(request.conditioning_code)
        ),
    )
