"""Qualitative stress surfaces for a simulated unobserved common cause.

The surface perturbs treatment and outcome through one shared latent vector, then
refits the complete estimator at each declared strength pair. It describes movement
of the fitted estimate on the scale ``movement_scale`` names. It does not provide a
bound, corrected estimate, test, or sensitivity-adjusted interval.
"""

from __future__ import annotations

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
    treatment : tuple of float
        Treatment perturbation strengths. Continuous-treatment values are signed linear
        coefficients. Binary-treatment values are checked as flip probabilities from zero
        through 0.5 when the surface runs.
    outcome : tuple of float
        Outcome perturbation strengths. Gaussian strengths are signed. Binomial
        strengths are checked as flip probabilities when the surface runs.

    See Also
    --------
    simulated_confounding : Evaluate the declared strength pairs.
    """

    treatment: tuple[float, ...]
    outcome: tuple[float, ...]

    def __post_init__(self) -> None:
        treatment = _numeric_strengths(self.treatment, "treatment")
        outcome = _numeric_strengths(self.outcome, "outcome")
        if 0.0 not in treatment or 0.0 not in outcome:
            raise ValueError("treatment and outcome strengths must each contain zero")
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)


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
    """

    covariate: str
    role: Literal["treatment", "outcome"]
    family: Literal["binomial", "gaussian"]
    strength: float
    method: str


@dataclass(frozen=True)
class SimulatedConfoundingCell:
    """Store one strength pair's estimate or structured failure.

    Parameters
    ----------
    treatment_strength : float
        Declared binary flip probability or continuous linear coefficient.
    outcome_strength : float
        Declared outcome perturbation strength.
    estimate : float or None
        Refitted estimate. ``None`` when the cell failed.
    displacement : float or None
        Refitted minus original value on the surface's movement scale. ``None`` when
        the cell failed.
    induced_treatment_association : float or None
        Pearson correlation between the shared latent vector and the treatment of this
        cell, measured within its selected baseline population under the fixed row weights.
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

    treatment_strength: float
    outcome_strength: float
    estimate: float | None
    displacement: float | None
    induced_treatment_association: float | None = None
    failure: ReplicationFailure | None = None
    target_population_fraction: float | None = None


@dataclass(frozen=True)
class SimulatedConfoundingResult:
    """Store a qualitative simulated-confounding stress surface.

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
            f"target population: {self.population}",
            f"conditioning arm: {self.conditioning_arm!r}",
            f"baseline stratum: {self.stratum!r}",
            f"strata columns: {self.strata_names!r}",
            f"association population: {self.association_population}",
            f"calibration population: {self.calibration_population}",
            f"refit population: {self.refit_population}",
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
        payload = {
            "treatment_strength": [cell.treatment_strength for cell in self.cells],
            "outcome_strength": [cell.outcome_strength for cell in self.cells],
            "movement_scale": [self.movement_scale for _ in self.cells],
            "estimate": [cell.estimate for cell in self.cells],
            "displacement": [cell.displacement for cell in self.cells],
            "induced_treatment_association": [
                cell.induced_treatment_association for cell in self.cells
            ],
            "association_population": [self.association_population for _ in self.cells],
            "target_population_fraction": [cell.target_population_fraction for cell in self.cells],
            "error_type": [
                None if cell.failure is None else cell.failure.error_type for cell in self.cells
            ],
            "message": [
                None if cell.failure is None else cell.failure.message for cell in self.cells
            ],
        }
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
            association = cell.induced_treatment_association
            reported = "n/a" if association is None else f"{association:+.4f}"
            fraction = cell.target_population_fraction
            population_fraction = "n/a" if fraction is None else f"{fraction:.4f}"
            if cell.failure is None:
                rows.append(
                    [
                        f"{cell.treatment_strength:.4g}",
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
                        f"{cell.treatment_strength:.4g}",
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
                "",
                format_table(
                    [
                        "treatment strength",
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

    def _reading_guard(self) -> tuple[str, ...]:
        """Return the lines that say which cells carry a confounding path."""
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


def _perturb_treatment(
    values: np.ndarray[Any, Any],
    latent: np.ndarray[Any, Any],
    strength: float,
    family: Literal["binary", "continuous"],
) -> np.ndarray[Any, Any]:
    if family == "continuous":
        return _linear_treatment(values, latent, strength)
    return _flip_binary(values, _flip_mask(latent, strength))


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


def _binary_calibration(
    design: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    index: int,
    weights: np.ndarray[Any, Any],
) -> float:
    model = LogisticRegression(max_iter=1000)
    if _weights_are_constant(weights):
        model.fit(design, target)
    else:
        model.fit(design, target, sample_weight=weights)
    baseline = model.predict(design)
    removed = design.copy()
    removed[:, index] = 0.0
    changed = model.predict(removed) != baseline
    if _weights_are_constant(weights):
        return float(np.mean(changed))
    return float(np.average(changed, weights=weights))


def _continuous_calibration(
    covariate: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> float:
    correlation = _weighted_correlation(covariate, target, weights)
    if correlation is None:
        return float("nan")
    return correlation * _weighted_std(target, weights)


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
        index = data.covariate_names.index(name)
        if data.is_continuous_treatment:
            treatment_strength = _continuous_calibration(
                data.covariates[:, index], data.treatment, data.weights
            )
            treatment_family: Literal["binomial", "gaussian"] = "gaussian"
            treatment_method = "signed standardized marginal coefficient"
        else:
            treatment_strength = _binary_calibration(design, data.treatment, index, data.weights)
            treatment_family = "binomial"
            treatment_method = "logistic class-prediction change fraction"
        rows.append(
            ObservedConfounderCalibration(
                covariate=name,
                role="treatment",
                family=treatment_family,
                strength=treatment_strength,
                method=treatment_method,
            )
        )
        if data.family == "binomial":
            outcome_strength = _binary_calibration(design, data.outcome, index, data.weights)
            method = "logistic class-prediction change fraction"
        else:
            outcome_strength = _continuous_calibration(
                data.covariates[:, index], data.outcome, data.weights
            )
            method = "signed standardized marginal coefficient"
        rows.append(
            ObservedConfounderCalibration(
                covariate=name,
                role="outcome",
                family=data.family,
                strength=outcome_strength,
                method=method,
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
        A continuous fit requires an explicit ``ey_shift[...]`` alias
        of a nonzero-delta policy, an ``ate_shift[...]`` alias, or an ``msm[...]`` coefficient.
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
    an ``ate_shift`` contrast. A policy mean keeps it, so read the zero treatment-strength
    column of an ``ey_shift`` surface as an artifact of the outcome law.

    A zero-delta shift is the natural course, its policy mean is ``E[Y]``, and it has no
    counterfactual treatment dependence. ``simulated_confounding`` refuses that mean before
    it draws the latent vector. It still accepts an ``ate_shift`` contrast that uses the
    natural course as its reference. The same refusal applies to ``NaturalCourseMean``.
    PAR and PAF retain counterfactual treatment dependence through their reference intervention.
    """
    if type(grid) is not ConfounderStrengthGrid:
        raise TypeError("grid must be an exact ConfounderStrengthGrid declaration")
    request = _validate_request(result, estimand, grid, benchmark_covariates)
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
    latent = np.random.default_rng(latent_seed).normal(size=result.data.n)
    original_parameter = result[estimand]
    original = float(original_parameter.psi)
    original_inference = original_parameter.inference_value
    cells: list[SimulatedConfoundingCell] = []

    for cell_index, (treatment_strength, outcome_strength) in enumerate(
        product(grid.treatment, grid.outcome)
    ):
        if treatment_strength == 0.0 and outcome_strength == 0.0:
            anchor_association, anchor_fraction = _cell_diagnostics(
                request, latent, result.data.treatment, result.data.weights
            )
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
        association: float | None = None
        population_fraction: float | None = None
        try:
            treatment = _perturb_treatment(
                result.data.treatment,
                latent,
                treatment_strength,
                request.treatment_family,
            )
            association, population_fraction = _cell_diagnostics(
                request, latent, treatment, result.data.weights
            )
            replacement = result.data.with_treatment(treatment)
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
            else "Continuous treatment adds signed strength times the shared latent value."
        ),
        outcome_law=(
            "Gaussian outcome subtracts signed strength times the shared latent value."
            if result.data.family == "gaussian"
            else "Binomial outcome is flipped in the declared upper latent-normal tail."
        ),
        weight_report=result.data.weight_report(),
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
