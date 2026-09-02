"""Qualitative stress surfaces for a simulated unobserved common cause.

The surface perturbs treatment and outcome through one shared latent vector, then
refits the complete estimator at each declared strength pair. It describes movement
of the fitted point estimate. It does not provide a bound, corrected estimate, test,
or sensitivity-adjusted interval.
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

from ..exceptions import CapabilityError
from ..utils.frames import emit_frame
from ..utils.random import resolve_assessment_seed
from ..utils.text import format_table
from ..validation.simulation import ReplicationFailure

if TYPE_CHECKING:  # pragma: no cover - typing only
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
        Binary-treatment flip probabilities. Each value is from zero through 0.5.
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
        if any(value < 0.0 or value > 0.5 for value in treatment):
            raise ValueError("treatment strengths must be between 0 and 0.5")
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
        Declared binary-treatment flip probability.
    outcome_strength : float
        Declared outcome perturbation strength.
    estimate : float or None
        Refitted estimate. ``None`` when the cell failed.
    displacement : float or None
        Estimate minus the original estimate. ``None`` when the cell failed.
    failure : ReplicationFailure or None
        Structured refit or replacement failure retained for this cell.
    """

    treatment_strength: float
    outcome_strength: float
    estimate: float | None
    displacement: float | None
    failure: ReplicationFailure | None = None


@dataclass(frozen=True)
class SimulatedConfoundingResult:
    """Store a qualitative simulated-confounding stress surface.

    Parameters
    ----------
    estimand : str
        Marginal binary-treatment ATE alias.
    original_estimate : float
        Point estimate from the unperturbed fitted result.
    grid : ConfounderStrengthGrid
        Explicit strength pairs evaluated by the surface.
    cells : tuple of SimulatedConfoundingCell
        Successful estimates and retained failures in grid order.
    calibrations : tuple of ObservedConfounderCalibration
        Optional source-style comparisons against numeric observed covariates.
    root_seed : int
        Seed resolved for the complete operation.
    latent_seed : int
        Child seed used to draw the shared latent vector.
    refit_seed : int
        Child seed reused by every estimator refit.
    treatment_family : str
        Treatment family covered by the perturbation law.
    outcome_family : str
        Original outcome family.
    treatment_law : str
        Human-readable treatment perturbation rule.
    outcome_law : str
        Human-readable outcome perturbation rule.
    backend : str or None
        Dataframe backend used by frame methods.

    See Also
    --------
    simulated_confounding : Build this result from a fitted estimator.
    """

    estimand: str
    original_estimate: float
    grid: ConfounderStrengthGrid
    cells: tuple[SimulatedConfoundingCell, ...]
    calibrations: tuple[ObservedConfounderCalibration, ...]
    root_seed: int
    latent_seed: int
    refit_seed: int
    treatment_family: str
    outcome_family: str
    treatment_law: str
    outcome_law: str
    backend: str | None = None

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
            Estimates, displacements, and retained failure details.
        """
        payload = {
            "treatment_strength": [cell.treatment_strength for cell in self.cells],
            "outcome_strength": [cell.outcome_strength for cell in self.cells],
            "estimate": [cell.estimate for cell in self.cells],
            "displacement": [cell.displacement for cell in self.cells],
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
        }
        return emit_frame(payload, backend=self.backend)

    def summary(self) -> str:
        """Return a qualitative text summary without an inferential verdict.

        Returns
        -------
        str
            Original estimate, seeds, perturbation laws, and cell movements.
        """
        rows = []
        for cell in self.cells:
            if cell.failure is None:
                rows.append(
                    [
                        f"{cell.treatment_strength:.4g}",
                        f"{cell.outcome_strength:.4g}",
                        f"{cell.estimate:.5g}",
                        f"{cell.displacement:+.5g}",
                    ]
                )
            else:
                rows.append(
                    [
                        f"{cell.treatment_strength:.4g}",
                        f"{cell.outcome_strength:.4g}",
                        "failed",
                        cell.failure.error_type,
                    ]
                )
        return "\n".join(
            [
                f"Simulated common-cause stress surface for {self.estimand!r}",
                f"original estimate: {self.original_estimate:.5g}",
                (
                    f"seeds: root={self.root_seed}, latent={self.latent_seed}, "
                    f"refit={self.refit_seed}"
                ),
                self.treatment_law,
                self.outcome_law,
                "",
                format_table(
                    ["treatment strength", "outcome strength", "estimate", "movement"], rows
                ),
                "",
                "This surface is qualitative. It is not a bound or sensitivity-adjusted inference.",
            ]
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


def _validate_request(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: Any,
) -> tuple[Any, tuple[str, ...]]:
    """Validate the complete supported boundary before a refit or random draw."""
    from ..estimators.base import TMLEResult
    from ..estimators.ctmle import CTMLE
    from ..estimators.drtmle import DRTMLE
    from ..estimators.tmle import TMLE
    from ..study import BackdoorMeanContrast, ExplicitAdjustmentProvider, ParameterKey
    from ..targets import TARGETS

    if type(result) is not TMLEResult:
        raise CapabilityError(
            "simulated_confounding supports point-treatment TMLEResult objects only; "
            "no longitudinal perturbation law is implemented"
        )
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError(
            "simulated_confounding needs a replayable fitted estimator; this restored or "
            "legacy result has no estimator configuration"
        )
    if type(estimator) not in {TMLE, CTMLE, DRTMLE}:
        raise CapabilityError(
            "simulated_confounding supports ordinary TMLE, collaborative TMLE, and "
            f"complete-outcome DR-TMLE; got {type(estimator).__name__}"
        )
    data = result.data
    if data.is_continuous_treatment or not data.is_binary_treatment:
        raise CapabilityError(
            "simulated_confounding requires a binary treatment; continuous and multi-arm "
            "treatments have no supported flip contrast"
        )
    if data.family not in {"gaussian", "binomial"}:
        raise CapabilityError(
            f"simulated_confounding has no perturbation law for outcome family={data.family!r}"
        )
    if data.family == "binomial" and any(value < 0.0 or value > 0.5 for value in grid.outcome):
        raise ValueError("binomial outcome strengths must be between 0 and 0.5")
    if data.has_missing_outcome:
        raise CapabilityError(
            "simulated_confounding has no missing-outcome perturbation and observation refit"
        )
    if data.has_intermediate or result.intermediate_value is not None:
        raise CapabilityError(
            "simulated_confounding has no controlled-direct-effect or intermediate-variable law"
        )
    if data.weights_name is not None or data.is_weighted:
        raise CapabilityError(
            "simulated_confounding does not support observation-weighted target populations"
        )
    if data.cluster is not None:
        raise CapabilityError("simulated_confounding does not support clustered fits")
    if result.n_repeats > 1:
        raise CapabilityError(
            "simulated_confounding does not support repeated cross-fitting; fit one split"
        )

    identified = result.identified_effect
    if identified is None:
        raise CapabilityError(
            "simulated_confounding needs identification metadata for a backdoor marginal ATE; "
            "this legacy fit records none"
        )
    functional = identified.functional
    if type(functional) is not BackdoorMeanContrast:
        raise CapabilityError(
            "simulated_confounding supports a backdoor-identified marginal ATE; "
            f"got {type(functional).__name__}"
        )
    provider = getattr(identified, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        raise CapabilityError(
            "simulated_confounding needs registered explicit-adjustment backdoor provenance"
        )
    if (
        functional.longitudinal
        or functional.axis != "arm"
        or functional.interventions
        or functional.msm is not None
    ):
        raise CapabilityError(
            "simulated_confounding supports a marginal arm-indexed ATE, not a regimen, "
            "stochastic, incremental, modified-policy, or MSM parameter"
        )
    if estimand not in result.estimates:
        raise ValueError(
            f"estimand {estimand!r} is unavailable; choose one of {list(result.estimates)}"
        )
    key = result.parameter_keys.get(estimand)
    if type(key) is not ParameterKey:
        raise CapabilityError(
            f"simulated_confounding needs a structured parameter key for {estimand!r}"
        )
    if key.estimand != "ate" or key.axis != "arm" or key.stratum is not None:
        raise CapabilityError(
            "simulated_confounding supports only a marginal ATE; ATT, ATC, means, ratios, "
            "conditional strata, and other parameters are outside its source boundary"
        )
    declared = getattr(getattr(identified, "estimand", None), "name", None)
    registered = TARGETS.get("ate")
    if (
        functional.target != "ate"
        or declared != "ate"
        or registered is None
        or identified.identification != registered.identification
    ):
        raise CapabilityError(
            "simulated_confounding found inconsistent registered ATE identification provenance"
        )

    names = tuple(
        [benchmark_covariates] if isinstance(benchmark_covariates, str) else benchmark_covariates
    )
    if len(set(names)) != len(names):
        raise ValueError("benchmark_covariates contains duplicates")
    categorical = {
        name for encoding in data.encodings for name in (encoding.column, *encoding.generated)
    }
    for name in names:
        if not isinstance(name, str):
            raise TypeError("benchmark_covariates must contain only column names")
        if name in categorical:
            raise CapabilityError(
                f"simulated_confounding cannot calibrate categorical covariate {name!r}; "
                "zeroing one encoded column does not define a logical-covariate benchmark"
            )
        if name not in data.covariate_names:
            raise ValueError(
                f"benchmark covariate {name!r} is unavailable; numeric adjustment columns are "
                f"{[name for name in data.covariate_names if name not in categorical]}"
            )
        column = data.covariates[:, data.covariate_names.index(name)]
        if float(np.std(column)) == 0.0:
            raise CapabilityError(
                f"simulated_confounding cannot calibrate constant covariate {name!r}"
            )
    return estimator, names


def _child_seeds(root_seed: int) -> tuple[int, int]:
    children = np.random.SeedSequence(root_seed).spawn(2)
    return tuple(int(child.generate_state(1)[0]) for child in children)  # type: ignore[return-value]


def _flip_mask(latent: np.ndarray[Any, Any], strength: float) -> np.ndarray[Any, Any]:
    if strength == 0.0:
        return np.zeros(latent.shape, dtype=bool)
    threshold = NormalDist().inv_cdf(1.0 - strength)
    return latent >= threshold


def _flip_binary(values: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.where(mask, 1.0 - values, values)


def _gaussian_outcome(
    values: np.ndarray[Any, Any], latent: np.ndarray[Any, Any], strength: float
) -> np.ndarray[Any, Any]:
    return values - strength * latent


def _binary_calibration(
    design: np.ndarray[Any, Any], target: np.ndarray[Any, Any], index: int
) -> float:
    model = LogisticRegression(max_iter=1000)
    model.fit(design, target)
    baseline = model.predict(design)
    removed = design.copy()
    removed[:, index] = 0.0
    return float(np.mean(model.predict(removed) != baseline))


def _calibrate(result: Any, names: tuple[str, ...]) -> tuple[ObservedConfounderCalibration, ...]:
    if not names:
        return ()
    data = result.data
    design = StandardScaler().fit_transform(data.covariates)
    rows: list[ObservedConfounderCalibration] = []
    for name in names:
        index = data.covariate_names.index(name)
        treatment_strength = _binary_calibration(design, data.treatment, index)
        rows.append(
            ObservedConfounderCalibration(
                covariate=name,
                role="treatment",
                family="binomial",
                strength=treatment_strength,
                method="logistic class-prediction change fraction",
            )
        )
        if data.family == "binomial":
            outcome_strength = _binary_calibration(design, data.outcome, index)
            method = "logistic class-prediction change fraction"
        else:
            outcome_strength = float(
                np.corrcoef(data.covariates[:, index], data.outcome)[0, 1] * np.std(data.outcome)
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
    """Refit a marginal ATE across a simulated common-cause strength grid.

    Parameters
    ----------
    result : TMLEResult
        Replayable backdoor-identified marginal binary-treatment ATE fit.
    estimand : str
        ATE alias to report.
    grid : ConfounderStrengthGrid
        Explicit treatment and outcome perturbation strengths.
    benchmark_covariates : tuple of str
        Numeric observed covariates for optional model-dependent calibration.
    random_state : int or None
        Root seed. ``None`` uses the fitted estimator's seed or draws a recorded seed.

    Returns
    -------
    SimulatedConfoundingResult
        Point-estimate movements and retained cell failures. The result has no verdict
        or sensitivity-adjusted inference.

    See Also
    --------
    ConfounderStrengthGrid : The explicit strength declaration.
    cleverly.sensitivity.omitted_variable_bounds : A non-refit bias-bound analysis.
    """
    if type(grid) is not ConfounderStrengthGrid:
        raise TypeError("grid must be an exact ConfounderStrengthGrid declaration")
    estimator, calibration_names = _validate_request(result, estimand, grid, benchmark_covariates)
    calibrations = _calibrate(result, calibration_names)
    root_seed = resolve_assessment_seed(result, random_state)
    latent_seed, refit_seed = _child_seeds(root_seed)
    latent = np.random.default_rng(latent_seed).normal(size=result.data.n)
    original = float(result[estimand].psi)
    cells: list[SimulatedConfoundingCell] = []

    for cell_index, (treatment_strength, outcome_strength) in enumerate(
        product(grid.treatment, grid.outcome)
    ):
        if treatment_strength == 0.0 and outcome_strength == 0.0:
            cells.append(
                SimulatedConfoundingCell(treatment_strength, outcome_strength, original, 0.0)
            )
            continue
        try:
            treatment = _flip_binary(result.data.treatment, _flip_mask(latent, treatment_strength))
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
            refitted = estimator.refit(replacement, random_state=refit_seed)
            estimate = float(refitted[estimand].psi)
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength,
                    outcome_strength,
                    estimate,
                    estimate - original,
                )
            )
        except Exception as error:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength,
                    outcome_strength,
                    None,
                    None,
                    ReplicationFailure(
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
        grid=grid,
        cells=tuple(cells),
        calibrations=calibrations,
        root_seed=root_seed,
        latent_seed=latent_seed,
        refit_seed=refit_seed,
        treatment_family="binary",
        outcome_family=result.data.family,
        treatment_law="Treatment is flipped in the declared upper latent-normal tail.",
        outcome_law=(
            "Gaussian outcome subtracts signed strength times the shared latent value."
            if result.data.family == "gaussian"
            else "Binomial outcome is flipped in the declared upper latent-normal tail."
        ),
        backend=result.data.backend,
    )
